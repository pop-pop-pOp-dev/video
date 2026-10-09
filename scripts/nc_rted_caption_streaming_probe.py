#!/usr/bin/env python3
"""Measure one-at-a-time caption-media preparation against the frozen resolver.

This is a diagnostic/preparation tool.  It reuses the released Stage2 resolver's
request parser, splitter child, full decode, and segment validator.  It never
materializes a compatibility tree or changes production cache configuration.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import statistics
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace


class ProbeError(RuntimeError):
    pass


MINIMUM_RESERVE_BYTES = 20 * 1024 ** 3


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_serial(path: Path):
    spec = importlib.util.spec_from_file_location("nc_rted_stage2_serial", path)
    if spec is None or spec.loader is None:
        raise ProbeError(f"cannot load Stage2 serial resolver: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source_duration(path: Path) -> float:
    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if fps <= 0 or frames <= 0:
        raise ProbeError(f"cannot read source duration: {path}")
    return frames / fps


def _available_bytes(path: Path) -> int:
    stat = os.statvfs(path)
    return stat.f_bavail * stat.f_frsize


def _inventory(captions_path: Path, config: dict, serial) -> list[dict]:
    captions = json.loads(captions_path.read_text(encoding="utf-8"))
    train = json.loads(Path(config["train_json"]).read_text(encoding="utf-8"))
    if not isinstance(captions, list) or len(captions) != 2000:
        raise ProbeError("caption subset must contain exactly 2,000 rows")
    train_indices = {row.get("id"): index for index, row in enumerate(train)}
    if len(train_indices) != len(train):
        raise ProbeError("frozen training identities are not unique")
    resolver = serial.FrozenTrainRequests(config)
    entries: list[dict] = []
    for row in captions:
        if not isinstance(row, dict) or not isinstance(row.get("id"), int) or not isinstance(row.get("video"), str):
            raise ProbeError("caption subset row lacks a frozen identity")
        request_index = train_indices.get(row["id"])
        if request_index is None:
            raise ProbeError(f"caption identity absent from frozen train set: {row['id']}")
        request = resolver.resolve(row["video"], request_index)
        if request["kind"] == "videos":
            segment = (0.0, _source_duration(Path(request["source"])))
        else:
            segment = tuple(float(value) for value in request["segment"])
        start, end = segment
        if not 0 <= start < end:
            raise ProbeError(f"invalid resolved segment: {row['video']}")
        source = Path(request["source"])
        entries.append({
            "id": row["id"],
            "request_index": request_index,
            "relative_video": row["video"],
            "kind": request["kind"],
            "source": str(source),
            "source_bytes": source.stat().st_size,
            "source_sha256": request["source_sha256"],
            "segment_start_s": start,
            "segment_end_s": end,
            "segment_seconds": end - start,
        })
    return entries


def _selected(entries: list[dict]) -> dict[str, dict]:
    ordered = sorted(entries, key=lambda entry: (entry["segment_seconds"], entry["relative_video"]))
    derived = [entry for entry in ordered if entry["kind"] != "videos"]
    if not derived:
        raise ProbeError("caption subset has no derived clip/event media")
    return {
        "shortest": ordered[0],
        "middle": ordered[len(ordered) // 2],
        "longest": ordered[-1],
        "derived_median": derived[len(derived) // 2],
        "derived_longest": derived[-1],
        "direct_video": next(entry for entry in ordered if entry["kind"] == "videos"),
    }


def _summary(entries: list[dict], selected: dict[str, dict]) -> dict:
    durations = sorted(entry["segment_seconds"] for entry in entries)
    source_bytes = sorted(entry["source_bytes"] for entry in entries)
    return {
        "selected_captions": len(entries),
        "unique_sources": len({entry["source"] for entry in entries}),
        "kind_counts": {kind: sum(entry["kind"] == kind for entry in entries)
                        for kind in ("clips", "events", "videos")},
        "segment_seconds": {
            "min": durations[0], "median": statistics.median(durations),
            "p95": durations[round(0.95 * (len(durations) - 1))], "max": durations[-1],
        },
        "source_bytes": {
            "min": source_bytes[0], "median": statistics.median(source_bytes),
            "p95": source_bytes[round(0.95 * (len(source_bytes) - 1))], "max": source_bytes[-1],
        },
        "preselected": selected,
    }


def _admit_temporary(scratch_root: Path, reserve_bytes: int, overhead_bytes: int,
                     requested: int | None) -> dict:
    available = _available_bytes(scratch_root)
    permitted = available - reserve_bytes - overhead_bytes
    if permitted <= 0:
        raise ProbeError("no temporary-media budget remains above the required reserve")
    if requested is not None and requested > permitted:
        raise ProbeError("requested temporary-media cap would violate the required reserve")
    return {
        "available_bytes_before_creation": available,
        "reserve_bytes": reserve_bytes,
        "overhead_bytes": overhead_bytes,
        "max_one_temporary_bytes": requested if requested is not None else permitted,
    }


@contextmanager
def _scratch_lock(scratch_root: Path):
    lock_path = scratch_root / ".caption-streaming-probe.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def _preserve_failure(path: Path) -> str | None:
    if not path.exists():
        return None
    preserved = path.with_suffix(".failed.partial.mp4")
    try:
        os.replace(path, preserved)
        return str(preserved)
    except OSError:
        # The original path still names the failed material if the rename fails.
        return str(path)


class _ValidatorTemporaryGuard:
    """Keep a linked validator temporary only when validation itself fails."""
    def __init__(self, serial, scratch_root: Path, reserve_bytes: int, overhead_bytes: int,
                 requested_maximum: int | None, on_artifact):
        self.serial = serial
        self.scratch_root = scratch_root
        self.reserve_bytes = reserve_bytes
        self.overhead_bytes = overhead_bytes
        self.requested_maximum = requested_maximum
        self.on_artifact = on_artifact
        self._original_tempfile = None
        self._retained: list[Path] = []
        self.admissions: list[dict] = []
        self.failed = False
        self.limit_before = None
        self.admission = None

    def __enter__(self):
        self._original_tempfile = self.serial.tempfile
        self.limit_before = resource.getrlimit(resource.RLIMIT_FSIZE)

        def guarded_mkstemp(*args, **kwargs):
            if self._retained:
                raise ProbeError("frozen validator requested more than one auxiliary temporary")
            self.admission = _admit_temporary(self.scratch_root, self.reserve_bytes, self.overhead_bytes,
                                              self.requested_maximum)
            self.admissions.append(self.admission)
            resource.setrlimit(resource.RLIMIT_FSIZE,
                               (self.admission["max_one_temporary_bytes"], self.limit_before[1]))
            kwargs["dir"] = str(self.scratch_root)
            fd, name = self._original_tempfile.mkstemp(*args, **kwargs)
            original = Path(name)
            self.on_artifact("validator_original", original)
            retained = original.with_suffix(original.suffix + ".validator-pending.mp4")
            try:
                os.link(original, retained)
            except OSError:
                os.close(fd)
                raise
            self._retained.append(retained)
            self.on_artifact("validator_retained_link", retained)
            return fd, name

        self.serial.tempfile = SimpleNamespace(mkstemp=guarded_mkstemp)
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.serial.tempfile = self._original_tempfile
        resource.setrlimit(resource.RLIMIT_FSIZE, self.limit_before)
        self.failed = exc_type is not None
        return False

    def verify_and_release(self, source: Path) -> None:
        import cv2
        expected = cv2.VideoCapture(str(source))
        try:
            expected_size = (int(expected.get(cv2.CAP_PROP_FRAME_WIDTH)), int(expected.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        finally:
            expected.release()
        # The frozen validator has exactly one auxiliary writer.  Refusing a
        # second writer makes failed cleanup all-or-nothing for this probe.
        for path in self._retained:
            metadata = self.serial.full_decode(path)
            capture = cv2.VideoCapture(str(path))
            try:
                size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            finally:
                capture.release()
            if metadata["frames"] != 1 or metadata["fps"] <= 0 or size != expected_size:
                raise ProbeError("validator auxiliary did not retain the expected one-frame source format")
        for path in self._retained:
            if len(self._retained) != 1:
                raise ProbeError("validator auxiliary release requires exactly one verified file")
            try:
                path.unlink()
            except OSError as error:
                self.failed = True
                raise ProbeError(f"validator auxiliary cleanup failed: {path}") from error

    def audit(self) -> dict:
        return {
            "admissions": self.admissions,
            "enforced_max_bytes": None if self.admission is None else self.admission["max_one_temporary_bytes"],
            "released_success": [str(path) for path in self._retained if not path.exists()],
            "preserved_failure": [str(path) for path in self._retained if path.exists()],
        }


def _probe_derived(entry: dict, config: dict, serial_path: Path, serial, scratch_root: Path,
                   reserve_bytes: int, overhead_bytes: int, requested_maximum: int | None,
                   timeout_seconds: int, retain_success: bool, on_artifact) -> dict:
    with _scratch_lock(scratch_root):
        source = Path(entry["source"])
        actual_source_hash = _sha256(source)
        if actual_source_hash != entry["source_sha256"]:
            raise ProbeError(f"frozen source hash differs for {entry['relative_video']}")
        if entry["kind"] == "videos":
            metadata = serial.full_decode(source)
            return {"status": "PASS", "direct_raw_read": True, "source_sha256": actual_source_hash,
                    "source_metadata": metadata, "free_bytes_after": _available_bytes(scratch_root)}
        admission = _admit_temporary(scratch_root, reserve_bytes, overhead_bytes, requested_maximum)
        fd, temporary_name = tempfile.mkstemp(prefix="caption-stream-", suffix=".partial.mp4", dir=scratch_root)
        os.close(fd)
        temporary = Path(temporary_name)
        on_artifact("primary_temporary", temporary)
        command = [
            sys.executable, str(serial_path), "--child", "--source", str(source),
            "--splitter", config["official_splitter"],
            "--segment", json.dumps([entry["segment_start_s"], entry["segment_end_s"]]),
            "--output", str(temporary), "--max-bytes", str(admission["max_one_temporary_bytes"]),
            "--max-memory", str(config.get("child_memory_bytes", 8 * 1024 ** 3)),
        ]
        if config.get("chunked_splitter"):
            command.extend(["--chunked-splitter", config["chunked_splitter"]])
        started = time.monotonic()
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=timeout_seconds, check=False)
        except subprocess.TimeoutExpired as error:
            return {"status": "FAILED", "reason": "timeout", "seconds": time.monotonic() - started,
                    "admission": admission, "free_bytes_after": _available_bytes(scratch_root),
                    "preserved_failure": _preserve_failure(temporary), "error": str(error)}
        if result.returncode != 0:
            return {"status": "FAILED", "reason": "splitter_nonzero", "exit_code": result.returncode,
                    "seconds": time.monotonic() - started, "admission": admission,
                    "free_bytes_after": _available_bytes(scratch_root), "stdout": result.stdout[-4000:], "stderr": result.stderr[-4000:],
                    "preserved_failure": _preserve_failure(temporary)}
        validator_guard = _ValidatorTemporaryGuard(serial, scratch_root, reserve_bytes, overhead_bytes,
                                                    requested_maximum, on_artifact)
        try:
            metadata = serial.full_decode(temporary)
            with validator_guard:
                serial.validate_segment_output(metadata, temporary, source,
                                               [entry["segment_start_s"], entry["segment_end_s"]])
            validator_guard.verify_and_release(source)
            if metadata["bytes"] > admission["max_one_temporary_bytes"]:
                raise ProbeError("validated temporary object exceeded its admitted budget")
        except Exception as error:
            return {"status": "FAILED", "reason": "decode_or_segment_validation", "error": str(error),
                    "seconds": time.monotonic() - started, "admission": admission,
                    "free_bytes_after": _available_bytes(scratch_root), "validator_temporary": validator_guard.audit(),
                    "preserved_failure": _preserve_failure(temporary)}
        report = {"status": "PASS", "seconds": time.monotonic() - started,
                  "admission": admission, "free_bytes_after": _available_bytes(scratch_root),
                  "source_sha256": actual_source_hash, "derived": metadata,
                  "validator_temporary": validator_guard.audit()}
        if retain_success:
            retained = temporary.with_suffix(".verified.mp4")
            os.replace(temporary, retained)
            on_artifact("retained_success", retained)
            report["retained_success"] = str(retained)
        else:
            temporary.unlink()
            report["released_success"] = True
        report["free_bytes_after_release"] = _available_bytes(scratch_root)
        return report


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _reserve_report(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise ProbeError("refusing to replace an existing probe report; select a new report path") from error
    os.close(descriptor)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--captions", required=True, type=Path)
    parser.add_argument("--cache-config", required=True, type=Path)
    parser.add_argument("--serial-resolver", required=True, type=Path)
    parser.add_argument("--expected-resolver-sha256", required=True)
    parser.add_argument("--scratch-root", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--probe", default="", help="comma-separated: shortest,derived_median,derived_longest")
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--reserve-bytes", type=int, default=MINIMUM_RESERVE_BYTES)
    parser.add_argument("--overhead-bytes", type=int, default=16 * 1024 ** 2)
    parser.add_argument("--max-temporary-bytes", type=int)
    parser.add_argument("--retain-success", action="store_true")
    args = parser.parse_args()
    if args.timeout_seconds <= 0 or args.reserve_bytes < MINIMUM_RESERVE_BYTES or args.overhead_bytes <= 0:
        raise ProbeError("invalid capacity or timeout argument")
    if args.max_temporary_bytes is not None and args.max_temporary_bytes <= 0:
        raise ProbeError("temporary-media cap must be positive")
    args.scratch_root.mkdir(parents=True, exist_ok=True)
    _reserve_report(args.report)
    actual_resolver_hash = _sha256(args.serial_resolver)
    if actual_resolver_hash != args.expected_resolver_sha256:
        raise ProbeError("serial resolver SHA-256 differs from the expected frozen binding")
    serial = _load_serial(args.serial_resolver)
    config = json.loads(args.cache_config.read_text(encoding="utf-8"))
    for path_key, hash_key in serial.FROZEN:
        if _sha256(Path(config[path_key])) != config[hash_key]:
            raise ProbeError(f"frozen Stage2 input hash differs: {path_key}")
    if config.get("chunked_splitter") and _sha256(Path(config["chunked_splitter"])) != config.get("chunked_splitter_sha256"):
        raise ProbeError("frozen Stage2 chunked splitter hash differs")
    entries = _inventory(args.captions, config, serial)
    selected = _selected(entries)
    report = {
        "schema": "nc_rted_caption_streaming_probe/v1",
        "probe_script": str(Path(__file__).resolve()),
        "probe_script_sha256": _sha256(Path(__file__)),
        "cache_config": str(args.cache_config.resolve()),
        "cache_config_sha256": _sha256(args.cache_config),
        "caption_subset": str(args.captions.resolve()),
        "caption_subset_sha256": _sha256(args.captions),
        "resolver": str(args.serial_resolver.resolve()),
        "resolver_sha256": actual_resolver_hash,
        "expected_resolver_sha256": args.expected_resolver_sha256,
        "inventory": _summary(entries, selected),
        "probe_results": {},
        "attempts": [],
    }
    requested = [name for name in args.probe.split(",") if name]
    unknown = sorted(set(requested) - {"shortest", "derived_median", "derived_longest", "direct_video"})
    if unknown:
        raise ProbeError(f"unknown probe selections: {', '.join(unknown)}")
    if len(set(requested)) != len(requested):
        raise ProbeError("duplicate probe selections are not allowed")
    if requested:
        report["capacity_policy"] = {"minimum_reserve_bytes": args.reserve_bytes,
                                     "overhead_bytes": args.overhead_bytes,
                                     "requested_max_temporary_bytes": args.max_temporary_bytes,
                                     "re_admitted_after_source_hash_before_each_creation": True,
                                     "scratch_lock": str((args.scratch_root / ".caption-streaming-probe.lock").resolve())}
        report["status"] = "RUNNING"
        _write_json(args.report, report)
        for name in requested:
            attempt = {"selection": name, "status": "RUNNING", "artifacts": []}
            report["attempts"].append(attempt)
            _write_json(args.report, report)

            def record_artifact(role, path):
                attempt["artifacts"].append({"role": role, "path": str(path)})
                _write_json(args.report, report)

            try:
                outcome = _probe_derived(
                    selected[name], config, args.serial_resolver, serial, args.scratch_root,
                    args.reserve_bytes, args.overhead_bytes, args.max_temporary_bytes,
                    args.timeout_seconds, args.retain_success, record_artifact)
            except Exception as error:
                outcome = {"status": "FAILED", "reason": "terminal_exception", "error": str(error),
                           "free_bytes_after": _available_bytes(args.scratch_root)}
            report["probe_results"][name] = outcome
            attempt["status"] = outcome["status"]
            attempt["outcome"] = outcome
            _write_json(args.report, report)
        report["status"] = "PASS" if all(value["status"] == "PASS" for value in report["probe_results"].values()) else "FAILED"
    else:
        report["status"] = "INVENTORY_ONLY"
    _write_json(args.report, report)
    return 0 if report["status"] != "FAILED" else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ProbeError as error:
        print(f"caption streaming probe failed: {error}", file=sys.stderr)
        raise SystemExit(2)
