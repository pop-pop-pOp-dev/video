"""One-at-a-time, fail-closed media preparation for inherited Stage2 decoding."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from typing import Iterator

from .task_inputs import sha256_file


MINIMUM_FREE_BYTES = 20 * 1024 ** 3
DATA_VOLUME_ROOT = Path("/root/autodl-tmp/lookaway-wm").resolve()


class BoundedStage2CacheError(RuntimeError):
    pass


def bounded_scratch_root(value: object) -> Path:
    """Resolve a non-symlink scratch directory under the approved data volume."""
    if not isinstance(value, str):
        raise BoundedStage2CacheError("bounded Stage2 scratch root must be absolute")
    path = Path(value)
    if not path.is_absolute():
        raise BoundedStage2CacheError("bounded Stage2 scratch root must be absolute")
    for ancestor in (path, *path.parents):
        if ancestor.exists() and ancestor.is_symlink():
            raise BoundedStage2CacheError("bounded Stage2 scratch root contains a symlink")
    resolved = path.resolve(strict=False)
    try:
        resolved.relative_to(DATA_VOLUME_ROOT)
    except ValueError as error:
        raise BoundedStage2CacheError("bounded Stage2 scratch root escapes the approved data volume") from error
    return resolved


def _load_resolver(path: Path):
    spec = importlib.util.spec_from_file_location("nc_rted_bounded_stage2_serial", path)
    if spec is None or spec.loader is None:
        raise BoundedStage2CacheError("cannot load the bound Stage2 resolver")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _ValidatorTemporaryGuard:
    """Bound the frozen validator's one-frame writer and retain it on failure."""

    def __init__(self, cache):
        self.cache = cache
        self.validator_maximum = None
        self.original_tempfile = None
        self.original_limit = None
        self.retained: list[Path] = []

    def __enter__(self):
        self.original_tempfile = self.cache.serial.tempfile
        self.original_limit = resource.getrlimit(resource.RLIMIT_FSIZE)

        def guarded_mkstemp(*args, **kwargs):
            if self.retained:
                raise BoundedStage2CacheError("frozen segment validator requested more than one temporary")
            # Recheck immediately before each creation, including the helper's
            # writer probe, so its temporary cannot cross the reserve boundary.
            self.cache._admit_temporary()
            kwargs["dir"] = str(self.cache.scratch_root)
            fd, name = self.original_tempfile.mkstemp(*args, **kwargs)
            original = Path(name)
            retained = original.with_suffix(original.suffix + ".validator-pending")
            try:
                os.link(original, retained)
            except OSError:
                os.close(fd)
                original.unlink(missing_ok=True)
                raise
            self.retained.append(retained)
            return fd, name

        # Complete every fallible setup step before replacing the frozen
        # resolver's module reference.  ``__exit__`` is not invoked if
        # ``__enter__`` raises, so restoring the rlimit here is essential.
        self.validator_maximum = self.cache._admit_temporary()
        try:
            resource.setrlimit(resource.RLIMIT_FSIZE, (self.validator_maximum, self.original_limit[1]))
        except BaseException:
            resource.setrlimit(resource.RLIMIT_FSIZE, self.original_limit)
            raise
        # The frozen validator imports ``tempfile`` as a module.  Replacing its
        # local reference confines the writer probe to this cache's scratch root.
        self.cache.serial.tempfile = SimpleNamespace(mkstemp=guarded_mkstemp)
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.cache.serial.tempfile = self.original_tempfile
        resource.setrlimit(resource.RLIMIT_FSIZE, self.original_limit)
        return False

    def release_success(self) -> None:
        for path in self.retained:
            path.unlink()

    def preserve_failure(self) -> list[str]:
        retained = []
        for path in self.retained:
            if path.exists():
                failed = path.with_suffix(path.suffix + ".failed")
                try:
                    os.replace(path, failed)
                    path = failed
                except OSError:
                    pass
                retained.append(str(path))
        return retained


class BoundedStage2Cache:
    """Adapt the reviewed frozen resolver to a single temporary lease.

    The object implements the same small protocol consumed by
    ``FailClosedStage2DatasetMixin``.  It never creates a compatibility tree or
    retains successful derived media after the original decoder returns.
    """

    def __init__(self, stage2: dict):
        self.stage2 = dict(stage2)
        self.module_path = Path(stage2["module"])
        self.config_path = Path(stage2["config"])
        self.scratch_root = bounded_scratch_root(stage2["scratch_root"])
        self.minimum_free_bytes = int(stage2["minimum_free_bytes"])
        self.overhead_bytes = int(stage2["overhead_bytes"])
        self.maximum_temporary_bytes = stage2.get("max_temporary_bytes")
        self.timeout_seconds = int(stage2.get("child_timeout_seconds", 600))
        self._validate_binding()
        self.serial = _load_resolver(self.module_path)
        self.config = json.loads(self.config_path.read_text(encoding="utf-8"))
        self._validate_frozen_inputs()
        self.requests = self.serial.FrozenTrainRequests(self.config)
        self._request_indices = self._frozen_request_indices()
        self._identities_by_request = {index: identity for identity, index in self._request_indices.items()}
        self.scratch_root.mkdir(parents=True, exist_ok=True)
        if self.scratch_root.is_symlink() or not self.scratch_root.is_dir():
            raise BoundedStage2CacheError("bounded Stage2 scratch root is unsafe")
        self.lock_path = self.scratch_root / ".nc-rted-bounded-stage2.lock"
        self.events_path = self.scratch_root / "bounded-stage2-events.jsonl"

    def _validate_binding(self) -> None:
        if sha256_file(self.module_path) != self.stage2["module_sha256"]:
            raise BoundedStage2CacheError("bound Stage2 resolver SHA-256 differs")
        if self.stage2.get("expected_resolver_sha256") != self.stage2["module_sha256"]:
            raise BoundedStage2CacheError("expected resolver SHA-256 differs from module binding")
        if sha256_file(self.config_path) != self.stage2["config_sha256"]:
            raise BoundedStage2CacheError("bound Stage2 configuration SHA-256 differs")
        if self.minimum_free_bytes < MINIMUM_FREE_BYTES or self.overhead_bytes <= 0:
            raise BoundedStage2CacheError("bounded Stage2 capacity policy is invalid")
        if self.maximum_temporary_bytes is not None and (type(self.maximum_temporary_bytes) is not int or self.maximum_temporary_bytes <= 0):
            raise BoundedStage2CacheError("bounded Stage2 temporary cap is invalid")
        if self.timeout_seconds <= 0:
            raise BoundedStage2CacheError("bounded Stage2 child timeout is invalid")

    def _validate_frozen_inputs(self) -> None:
        if self.config.get("status") != self.stage2["accepted_status"]:
            raise BoundedStage2CacheError("frozen Stage2 configuration is not accepted")
        for path_key, hash_key in self.serial.FROZEN:
            path, expected = self.config.get(path_key), self.config.get(hash_key)
            if not isinstance(path, str) or not isinstance(expected, str) or sha256_file(path) != expected:
                raise BoundedStage2CacheError(f"frozen Stage2 input hash differs: {path_key}")
        chunked = self.config.get("chunked_splitter")
        if chunked is not None:
            expected = self.config.get("chunked_splitter_sha256")
            if not isinstance(chunked, str) or not isinstance(expected, str) or sha256_file(chunked) != expected:
                raise BoundedStage2CacheError("frozen Stage2 input hash differs: chunked_splitter")
        child_memory = self.config.get("child_memory_bytes", 8 * 1024 ** 3)
        if type(child_memory) is not int or child_memory <= 0:
            raise BoundedStage2CacheError("frozen Stage2 child memory limit is invalid")

    def _frozen_request_indices(self) -> dict[tuple[object, str], int]:
        records = json.loads(Path(self.config["train_json"]).read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise BoundedStage2CacheError("frozen Stage2 train manifest is invalid")
        indices = {}
        for index, row in enumerate(records):
            if not isinstance(row, dict) or type(row.get("id")) not in {int, str} or not isinstance(row.get("video"), str):
                raise BoundedStage2CacheError("frozen Stage2 train identity is invalid")
            identity = (row["id"], row["video"])
            if identity in indices:
                raise BoundedStage2CacheError("frozen Stage2 train identity is ambiguous")
            indices[identity] = index
        return indices

    def _record(self, event: str, **fields) -> None:
        row = json.dumps({"event": event, **fields}, sort_keys=True)
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(row + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _available_bytes(self) -> int:
        stat = os.statvfs(self.scratch_root)
        return stat.f_bavail * stat.f_frsize

    def _admit_temporary(self, maximum: int | None = None) -> int:
        available = self._available_bytes()
        permitted = available - self.minimum_free_bytes - self.overhead_bytes
        cap = self.maximum_temporary_bytes if maximum is None else maximum
        if permitted <= 0 or (cap is not None and cap > permitted):
            raise BoundedStage2CacheError("temporary media would violate the required free-space reserve")
        return permitted if cap is None else cap

    def request_index_for(self, annotation) -> int:
        if not isinstance(annotation, dict):
            raise BoundedStage2CacheError("Stage2 annotation is invalid")
        relative = annotation.get("_reactvau_relative_video")
        # LazySupervisedDataset intentionally prepends ``data_root`` to
        # ``video``.  The adapter preserves the immutable instruction-relative
        # name separately, which is the resolver's actual identity binding.
        identity = (annotation.get("id"), relative)
        if (not isinstance(relative, str) or type(identity[0]) not in {int, str}):
            raise BoundedStage2CacheError("Stage2 annotation lost its frozen request identity")
        index = self._request_indices.get(identity)
        if index is None:
            raise BoundedStage2CacheError("Stage2 annotation does not bind one frozen request")
        try:
            self.requests.resolve(relative, index)
        except (ValueError, FileNotFoundError) as error:
            raise BoundedStage2CacheError("Stage2 annotation request cannot be resolved") from error
        return index

    def _request(self, relative_path: str, request_index: int) -> dict:
        identity = self._identities_by_request.get(request_index)
        if type(request_index) is not int or identity is None or identity[1] != relative_path:
            raise BoundedStage2CacheError("Stage2 request index does not bind the frozen path")
        try:
            request = self.requests.resolve(relative_path, request_index)
        except (ValueError, FileNotFoundError) as error:
            raise BoundedStage2CacheError("Stage2 request is not frozen") from error
        source = Path(request["source"])
        if sha256_file(source) != request["source_sha256"]:
            raise BoundedStage2CacheError("frozen Stage2 source SHA-256 differs")
        return request

    def provenance_for(self, relative_path: str, request_index: int) -> dict:
        """Return the revalidated source/range identity without retaining media."""
        request = self._request(relative_path, request_index)
        return {"mode": "bounded", "resolver_module_sha256": self.stage2["module_sha256"],
                "resolver_config_sha256": self.stage2["config_sha256"], "relative": request["relative"],
                "request_index": request_index, "kind": request["kind"], "source_sha256": request["source_sha256"],
                "segment": request.get("segment")}

    def _run_child(self, request: dict, output: Path, maximum_bytes: int):
        # These paths are executable. Rebind them immediately before spawning
        # the child instead of trusting construction-time validation.
        self._validate_binding()
        self._validate_frozen_inputs()
        command = [
            sys.executable, str(self.module_path), "--child", "--source", str(request["source"]),
            "--splitter", self.config["official_splitter"], "--segment", json.dumps(request["segment"]),
            "--output", str(output), "--max-bytes", str(maximum_bytes), "--max-memory",
            str(self.config.get("child_memory_bytes", 8 * 1024 ** 3)),
        ]
        if self.config.get("chunked_splitter"):
            command.extend(["--chunked-splitter", self.config["chunked_splitter"]])
        return subprocess.run(command, timeout=self.timeout_seconds, stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL, check=False)

    def _preserve(self, path: Path) -> str | None:
        if not path.exists():
            return None
        failed = path.with_suffix(path.suffix + ".failed")
        try:
            os.replace(path, failed)
            return str(failed)
        except OSError:
            return str(path)

    @contextmanager
    def acquire(self, relative_path: str, request_index: int) -> Iterator[Path]:
        self.scratch_root.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            request = self._request(relative_path, request_index)
            if request["kind"] == "videos":
                self._record("access_raw", relative=request["relative"], request_index=request_index,
                             source=str(request["source"]))
                yield Path(request["source"])
                return
            maximum_bytes = self._admit_temporary()
            fd, name = tempfile.mkstemp(prefix="stage2-", suffix=".partial.mp4", dir=self.scratch_root)
            os.close(fd)
            output = Path(name)
            guard = _ValidatorTemporaryGuard(self)
            try:
                try:
                    result = self._run_child(request, output, maximum_bytes)
                except subprocess.TimeoutExpired as error:
                    raise BoundedStage2CacheError("bounded Stage2 splitter child timed out") from error
                if result.returncode != 0:
                    raise BoundedStage2CacheError("bounded Stage2 splitter child failed")
                metadata = self.serial.full_decode(output)
                with guard:
                    self.serial.validate_segment_output(metadata, output, request["source"], request["segment"])
                if metadata["bytes"] > maximum_bytes:
                    raise BoundedStage2CacheError("bounded Stage2 output exceeds its admitted cap")
                self._record("prepared", relative=request["relative"], request_index=request_index,
                             output=str(output), **metadata)
                try:
                    yield output
                except BaseException:
                    preserved = self._preserve(output)
                    self._record("consumer_failure", relative=request["relative"], request_index=request_index,
                                 preserved=preserved)
                    raise
                else:
                    output.unlink()
                    guard.release_success()
                    self._record("released", relative=request["relative"], request_index=request_index)
            except BaseException as error:
                preserved = self._preserve(output)
                validator = guard.preserve_failure()
                self._record("failure", relative=request["relative"], request_index=request_index,
                             error=type(error).__name__, preserved=preserved, validator_temporaries=validator)
                raise
