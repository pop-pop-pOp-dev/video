"""Atomic, immutable prediction records with durable resume state."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import traceback
import uuid
from typing import Any

from .prediction_inputs import canonical_json
from .storage_lock import allocation_lock, ensure_directory


class PredictionStoreError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json(value: Any) -> Any:
    return asdict(value) if is_dataclass(value) else value


def _write_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    payload = canonical_json(value) + b"\n"
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _write_new_atomic(path: Path, value: dict) -> None:
    """Publish a record by linking a fsynced temporary file exactly once."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    payload = canonical_json(value) + b"\n"
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        descriptor = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


class PredictionPublicationError(PredictionStoreError):
    """The immutable record exists, but the mutable index was not updated."""


class PredictionStore:
    def __init__(self, root: str | Path, *, run_id: str, manifest_sha256: str, model_task: str, model_binding_sha256: str,
                 matrix_id: str | None = None, task_execution_binding_sha256: str | None = None,
                 storage_volume: str | Path | None = None, min_free_bytes: int = 0):
        self.root = Path(root)
        self.storage_volume = Path(storage_volume).resolve() if storage_volume is not None else None
        if type(min_free_bytes) is not int or min_free_bytes < 0:
            raise PredictionStoreError("prediction storage reserve is invalid")
        self.min_free_bytes = min_free_bytes
        if self.storage_volume is None and min_free_bytes:
            raise PredictionStoreError("prediction storage reserve requires an admitted volume")
        if not isinstance(model_binding_sha256, str) or len(model_binding_sha256) != 64:
            raise PredictionStoreError("selected model manifest hash is required")
        if (matrix_id is None) != (task_execution_binding_sha256 is None):
            raise PredictionStoreError("v2 prediction reuse requires both matrix and task execution bindings")
        if matrix_id is not None and (not isinstance(matrix_id, str) or len(matrix_id) != 64 or not isinstance(task_execution_binding_sha256, str) or len(task_execution_binding_sha256) != 64):
            raise PredictionStoreError("v2 prediction reuse bindings are invalid")
        self.run_id, self.manifest_sha256, self.model_task, self.model_binding_sha256 = run_id, manifest_sha256, model_task, model_binding_sha256
        self.matrix_id, self.task_execution_binding_sha256 = matrix_id, task_execution_binding_sha256
        self._assert_storage_root(create=True)
        self.records = self.root / "records"
        self.index_path = self.root / "index.json"
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.root / ".worker.lock"
        with self.lease():
            self._ensure_storage_directory(self.records)
            if self.records.is_symlink():
                raise PredictionStoreError("prediction records directory cannot be a symlink")
            if self.index_path.exists():
                self._read_index()
            else:
                index = ({"schema": "nc_rted_prediction_index/v1", "run_id": run_id, "manifest_sha256": manifest_sha256,
                          "model_task": model_task, "model_binding_sha256": model_binding_sha256, "records": {}, "updated_at": _now()}
                         if matrix_id is None else
                         {"schema": "nc_rted_prediction_index/v2", "run_id": run_id, "matrix_id": matrix_id,
                          "task_execution_binding_sha256": task_execution_binding_sha256, "model_task": model_task,
                          "model_binding_sha256": model_binding_sha256, "records": {}, "updated_at": _now()})
                self._write_atomic(self.index_path, index)
            self._reconcile_records()

    @contextmanager
    def lease(self):
        """Serialize an entire task worker, including record and index updates."""
        self._assert_storage_root(create=True)
        guard = allocation_lock(self.storage_volume) if self.storage_volume is not None else nullcontext()
        with guard:
            self._same_filesystem(self.lock_path.parent)
            try:
                prior = self.lock_path.lstat()
                if not stat.S_ISREG(prior.st_mode) or prior.st_nlink != 1:
                    raise PredictionStoreError("prediction worker lock must be a regular file")
            except FileNotFoundError:
                if self.storage_volume is not None and shutil.disk_usage(self.storage_volume).free < self.min_free_bytes + 8192:
                    raise PredictionStoreError("prediction worker lock would violate storage reserve")
            flags = os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
            try:
                try: descriptor = os.open(self.lock_path, flags | os.O_CREAT | os.O_EXCL, 0o600)
                except FileExistsError: descriptor = os.open(self.lock_path, flags)
            except OSError as error:
                raise PredictionStoreError("prediction worker lock cannot be opened safely") from error
            info = os.fstat(descriptor)
            if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or
                    (self.storage_volume is not None and info.st_dev != self.storage_volume.stat().st_dev)):
                os.close(descriptor)
                raise PredictionStoreError("prediction worker lock must be a regular admitted-volume file")
        with os.fdopen(descriptor, "a+") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _assert_storage_root(self, *, create: bool = False) -> None:
        if self.root.is_symlink():
            raise PredictionStoreError("prediction output root cannot be a symlink")
        if create:
            self._ensure_storage_directory(self.root)
        if self.storage_volume is not None:
            try:
                self.root.resolve().relative_to(self.storage_volume)
            except ValueError as error:
                raise PredictionStoreError("prediction output root escapes admitted volume") from error
            self._same_filesystem(self.root)

    def _mount_identity(self, path: Path) -> tuple[int, str]:
        target = path.resolve()
        while not target.exists(): target = target.parent
        best = None
        try:
            for line in Path("/proc/self/mountinfo").read_text().splitlines():
                fields = line.split()
                if len(fields) > 5:
                    point = Path(fields[4].replace("\\040", " "))
                    try:
                        target.relative_to(point)
                        if best is None or len(point.parts) > len(best[1].parts): best = (int(fields[0]), point)
                    except ValueError: pass
        except (OSError, ValueError) as error:
            raise PredictionStoreError("prediction filesystem mount identity is unavailable") from error
        if best is None: raise PredictionStoreError("prediction filesystem mount identity is unavailable")
        return target.stat().st_dev, str(best[1])

    def _same_filesystem(self, path: Path) -> None:
        if self.storage_volume is not None and self._mount_identity(path) != self._mount_identity(self.storage_volume):
            raise PredictionStoreError("prediction output uses a different filesystem or mount")

    def _ensure_storage_directory(self, path: Path) -> None:
        if path.is_symlink():
            raise PredictionStoreError("prediction output directory cannot be a symlink")
        if self.storage_volume is None:
            path.mkdir(parents=True, exist_ok=True)
            return
        with allocation_lock(self.storage_volume):
            ensure_directory(path, self.min_free_bytes)
            if path.is_symlink():
                raise PredictionStoreError("prediction output directory cannot be a symlink")
            self._same_filesystem(path)
            if shutil.disk_usage(self.storage_volume).free < self.min_free_bytes:
                raise PredictionStoreError("prediction publication would violate the admitted storage reserve")

    @contextmanager
    def _storage_publication(self, path: Path, value: dict):
        if self.storage_volume is None:
            yield
            return
        payload_bytes = len(canonical_json(value)) + 1
        with allocation_lock(self.storage_volume):
            self._assert_storage_root()
            if path.parent.is_symlink():
                raise PredictionStoreError("prediction publication directory cannot be a symlink")
            self._same_filesystem(path.parent)
            if shutil.disk_usage(self.storage_volume).free < self.min_free_bytes + payload_bytes + 8192:
                raise PredictionStoreError("prediction publication would violate the admitted storage reserve")
            yield

    def _write_atomic(self, path: Path, value: dict) -> None:
        with self._storage_publication(path, value):
            _write_atomic(path, value)

    def _write_new_atomic(self, path: Path, value: dict) -> None:
        with self._storage_publication(path, value):
            _write_new_atomic(path, value)

    def _read_index(self) -> dict:
        try:
            value = json.loads(self.index_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise PredictionStoreError("prediction index is unreadable") from error
        expected = {"run_id": self.run_id, "model_task": self.model_task, "model_binding_sha256": self.model_binding_sha256}
        expected |= ({"schema": "nc_rted_prediction_index/v1", "manifest_sha256": self.manifest_sha256} if self.matrix_id is None else
                     {"schema": "nc_rted_prediction_index/v2", "matrix_id": self.matrix_id, "task_execution_binding_sha256": self.task_execution_binding_sha256})
        if not isinstance(value, dict) or not isinstance(value.get("records"), dict) or any(value.get(key) != item for key, item in expected.items()):
            raise PredictionStoreError("prediction index schema differs")
        return value

    def _read_record_path(self, path: Path, *, identity: str | None = None) -> dict:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise PredictionStoreError("prediction record is unreadable") from error
        if not isinstance(record, dict) or not isinstance(record.get("record_sha256"), str):
            raise PredictionStoreError("prediction record schema differs")
        digest = hashlib.sha256(canonical_json({key: value for key, value in record.items() if key != "record_sha256"})).hexdigest()
        if record["record_sha256"] != digest:
            raise PredictionStoreError("prediction record digest differs")
        expected = {"schema": "nc_rted_prediction_record/v1", "run_id": self.run_id, "model_task": self.model_task, "model_binding_sha256": self.model_binding_sha256}
        if self.matrix_id is None:
            expected["manifest_sha256"] = self.manifest_sha256
        elif (not isinstance(record.get("manifest_sha256"), str) or len(record["manifest_sha256"]) != 64 or not isinstance(record.get("provenance"), dict) or
              record["provenance"].get("matrix_id") != self.matrix_id or record["provenance"].get("task_execution_binding_sha256") != self.task_execution_binding_sha256):
            raise PredictionStoreError("prediction record cannot be reused across v2 plan revisions")
        if any(record.get(key) != value for key, value in expected.items()) or not isinstance(record.get("identity"), str) or not record["identity"]:
            raise PredictionStoreError("prediction record identity differs from output store")
        if identity is not None and record["identity"] != identity:
            raise PredictionStoreError("prediction record identity differs from output store")
        if record.get("status") not in {"success", "failure"} or type(record.get("attempt")) is not int or record["attempt"] < 1:
            raise PredictionStoreError("prediction record terminal state differs")
        return record

    def _reconcile_records(self) -> None:
        """Rebuild mutable index entries from immutable committed attempts."""
        index = self._read_index()
        self._validate_index_records(index)
        attempts: dict[str, list[tuple[Path, dict]]] = {}
        for path in sorted(self.records.glob("*.json")):
            record = self._read_record_path(path)
            expected_name = f"{self.key(identity=record['identity'])}.attempt{record['attempt']}.json"
            if path.name != expected_name:
                raise PredictionStoreError("prediction record path differs from its identity and attempt")
            attempts.setdefault(record["identity"], []).append((path, record))
        recovered = {}
        for identity, entries in attempts.items():
            entries.sort(key=lambda item: item[1]["attempt"])
            for expected_attempt, (_, record) in enumerate(entries, start=1):
                if record["attempt"] != expected_attempt:
                    raise PredictionStoreError("prediction record attempts are not contiguous")
                if expected_attempt > 1:
                    previous = entries[expected_attempt - 2][1]
                    if previous["status"] != "failure" or previous.get("retryable") is not True:
                        raise PredictionStoreError("prediction record retry sequence is invalid")
            path, record = entries[-1]
            recovered[self.key(identity=identity)] = {"path": str(path.relative_to(self.root)), "record_sha256": record["record_sha256"],
                                                      "status": record["status"], "attempt": record["attempt"]}
        if index["records"] != recovered:
            index["records"] = recovered
            index["updated_at"] = _now()
            self._write_atomic(self.index_path, index)

    def _validate_index_records(self, index: dict) -> None:
        for key, entry in index["records"].items():
            if not isinstance(key, str) or not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
                raise PredictionStoreError("prediction index record entry differs")
            relative = Path(entry["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise PredictionStoreError("prediction index record path escapes output root")
            if type(entry.get("attempt")) is not int or entry["attempt"] < 1 or relative != Path("records") / f"{key}.attempt{entry['attempt']}.json":
                raise PredictionStoreError("prediction index record path differs from canonical attempt")
            path = self.root / relative
            try:
                path.resolve().relative_to(self.root.resolve())
            except ValueError as error:
                raise PredictionStoreError("prediction index record path escapes output root") from error
            if path.is_symlink():
                raise PredictionStoreError("prediction index record cannot be a symlink")
            record = self._read_record_path(path)
            if (key != self.key(identity=record["identity"]) or entry.get("record_sha256") != record["record_sha256"] or
                    entry.get("status") != record["status"] or entry.get("attempt") != record["attempt"]):
                raise PredictionStoreError("prediction index record differs from committed record")

    @staticmethod
    def key(*, identity: str) -> str:
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()

    def get(self, *, identity: str) -> dict | None:
        entry = self._read_index()["records"].get(self.key(identity=identity))
        if entry is None:
            return None
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str) or not isinstance(entry.get("record_sha256"), str):
            raise PredictionStoreError("prediction index record entry differs")
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise PredictionStoreError("prediction index record path escapes output root")
        record = self._read_record_path(self.root / relative, identity=identity)
        if record["record_sha256"] != entry.get("record_sha256"):
            raise PredictionStoreError("prediction record digest differs")
        return record

    def should_run(self, *, identity: str, max_retries: int, now: datetime | None = None) -> bool:
        record = self.get(identity=identity)
        if record is None:
            return True
        if record.get("status") != "failure" or record.get("retryable") is not True or record.get("attempt") > max_retries:
            return False
        retry_not_before = record.get("retry_not_before")
        if not isinstance(retry_not_before, str):
            raise PredictionStoreError("retryable prediction record omits retry deadline")
        try:
            deadline = datetime.fromisoformat(retry_not_before.replace("Z", "+00:00"))
        except ValueError as error:
            raise PredictionStoreError("retryable prediction record has invalid retry deadline") from error
        current = datetime.now(timezone.utc) if now is None else now
        if current.tzinfo is None:
            raise PredictionStoreError("retry clock requires timezone-aware time")
        return current >= deadline

    def publish(self, *, identity: str, attempt: int, status: str, provenance: dict, payload: dict | None = None,
                error: BaseException | None = None, retryable: bool = False, retry_not_before: datetime | None = None) -> dict:
        if status not in {"success", "failure"} or attempt < 1:
            raise PredictionStoreError("invalid terminal prediction record")
        existing = self.get(identity=identity)
        if existing is not None and not (existing["status"] == "failure" and existing["retryable"] and attempt == existing["attempt"] + 1):
            raise PredictionStoreError("prediction record is immutable or retry sequence is invalid")
        if retryable and (status != "failure" or retry_not_before is None or retry_not_before.tzinfo is None):
            raise PredictionStoreError("retryable failure requires a timezone-aware retry deadline")
        if not retryable and retry_not_before is not None:
            raise PredictionStoreError("non-retryable prediction record cannot have a retry deadline")
        record = {"schema": "nc_rted_prediction_record/v1", "run_id": self.run_id, "manifest_sha256": self.manifest_sha256,
                  "model_task": self.model_task, "model_binding_sha256": self.model_binding_sha256, "identity": identity, "attempt": attempt, "status": status, "retryable": bool(retryable),
                  "created_at": _now(), "provenance": provenance}
        if retryable:
            record["retry_not_before"] = retry_not_before.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        if status == "success":
            if not isinstance(payload, dict):
                raise PredictionStoreError("success records require a structured payload")
            record["payload"] = payload
        else:
            if error is None:
                raise PredictionStoreError("failure records require the original exception")
            rendered = "".join(traceback.format_exception(type(error), error, error.__traceback__))
            record["failure"] = {"stage": provenance.get("stage", "prediction"), "exception_class": type(error).__name__,
                                 "message": str(error), "traceback_sha256": hashlib.sha256(rendered.encode()).hexdigest()}
        record["record_sha256"] = hashlib.sha256(canonical_json(record)).hexdigest()
        relative = Path("records") / f"{self.key(identity=identity)}.attempt{attempt}.json"
        try:
            self._write_new_atomic(self.root / relative, record)
        except FileExistsError as error:
            raise PredictionStoreError("prediction attempt record already exists") from error
        try:
            index = self._read_index()
            index["records"][self.key(identity=identity)] = {"path": str(relative), "record_sha256": record["record_sha256"], "status": status, "attempt": attempt}
            index["updated_at"] = _now()
            self._write_atomic(self.index_path, index)
        except BaseException as error:
            raise PredictionPublicationError("prediction record committed but index publication failed") from error
        return record
