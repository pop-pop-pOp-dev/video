"""Seed a new observation journal from immutable, validated committed windows."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import uuid

from .observation_extraction import ExtractionError, ObservationJournal, atomic_json
from .storage_lock import allocation_lock
from .teacher_store import RESERVED_FREE_BYTES, load_teacher_store


ALLOWED_CODE_DELTA = frozenset({"src/nc_rted/detector.py", "src/nc_rted/frozen_vision.py"})


class ObservationSeedError(ExtractionError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(8 << 20), b""):
            digest.update(part)
    return digest.hexdigest()


def bound_json(path: str | Path, expected_sha256: str) -> dict:
    path = Path(path)
    if not path.is_file() or sha256_file(path) != expected_sha256:
        raise ObservationSeedError("bound seed input changed or is absent")
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ObservationSeedError("bound seed input is not JSON") from error
    if not isinstance(value, dict):
        raise ObservationSeedError("bound seed input must be a JSON object")
    return value


def _same_except(old: dict, new: dict, allowed: set[str]) -> bool:
    return {key: value for key, value in old.items() if key not in allowed} == {
        key: value for key, value in new.items() if key not in allowed
    }


def validate_config_delta(source: dict, target: dict, probe: dict, *, source_config_path: Path,
                          source_config_sha256: str, source_run_path: Path, source_run_sha256: str,
                          equivalence_path: Path, equivalence_sha256: str,
                          probe_config_path: Path, probe_config_sha256: str, probe_run_path: Path,
                          probe_run_sha256: str) -> None:
    if source.get("schema") != "nc_rted_observation_extraction/v2" or target.get("schema") != source["schema"]:
        raise ObservationSeedError("unsupported observation configuration schema")
    if not _same_except(source, target, {"output", "frame_cache", "cpu_threads", "code_sha256", "resume_parent"}):
        raise ObservationSeedError("new observation config changes a non-resumable input")
    if target.get("output") == source.get("output") or target.get("frame_cache") == source.get("frame_cache"):
        raise ObservationSeedError("new observation output and frame cache must be distinct")
    old_code, new_code = source.get("code_sha256"), target.get("code_sha256")
    if not isinstance(old_code, dict) or set(old_code) != set(new_code or {}):
        raise ObservationSeedError("observation source closure changed")
    changed = {name for name in old_code if old_code[name] != new_code[name]}
    if changed != ALLOWED_CODE_DELTA:
        raise ObservationSeedError("observation source delta is not exactly the accepted hotpath pair")
    parent = target.get("resume_parent")
    required = {"schema", "config", "run", "equivalence_report", "probe_config", "probe_run",
                "reuse_policy", "source_change_scope"}
    if not isinstance(parent, dict) or set(parent) != required or parent.get("schema") != "nc_rted_equivalent_observation_parent/v1":
        raise ObservationSeedError("new observation config lacks an exact resume-parent binding")
    bindings = (("config", source_config_path, source_config_sha256), ("run", source_run_path, source_run_sha256),
                ("equivalence_report", equivalence_path, equivalence_sha256),
                ("probe_config", probe_config_path, probe_config_sha256), ("probe_run", probe_run_path, probe_run_sha256))
    if any(parent[name] != {"path": str(path), "sha256": digest} for name, path, digest in bindings):
        raise ObservationSeedError("resume-parent binding differs from supplied immutable parents")
    if parent["source_change_scope"] != sorted(ALLOWED_CODE_DELTA):
        raise ObservationSeedError("resume-parent source-change scope is invalid")
    if not _same_except(probe, target, {"output", "frame_cache", "resume_parent"}):
        raise ObservationSeedError("target config differs from the measured probe outside output paths")


def validate_equivalence(report: dict, parent: dict) -> None:
    rows = report.get("three_windows")
    if report.get("status") != "PASS_EQUIVALENCE" or not isinstance(rows, list) or len(rows) != 3:
        raise ObservationSeedError("three-window equivalence evidence is not accepted")
    if any(not isinstance(row, dict) or not isinstance(row.get("window_id"), str)
           or not row["window_id"] or row.get("payload_equal") is not True
           or row.get("index_entry_equal") is not True for row in rows):
        raise ObservationSeedError("probe evidence has a non-identical observation window")
    for report_name, parent_name in (("source_config", "config"), ("source_run", "run"),
                                     ("probe_config", "probe_config"), ("probe_run", "probe_run")):
        if report.get(report_name) != parent.get(parent_name):
            raise ObservationSeedError("equivalence report does not bind the configured parent inputs")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _link_tree(source: Path, staging: Path) -> None:
    for item in source.rglob("*"):
        relative = item.relative_to(source)
        target = staging / relative
        if item.is_symlink():
            raise ObservationSeedError("committed observation store contains a symlink")
        if item.is_dir():
            target.mkdir()
        elif item.is_file():
            os.link(item, target)
        else:
            raise ObservationSeedError("committed observation store contains an unsupported entry")


def _publish_linked_window(source: Path, destination: Path) -> None:
    if destination.exists():
        raise ObservationSeedError("destination window already exists")
    staging = Path(tempfile.mkdtemp(prefix=".seed-", dir=destination.parent))
    try:
        _link_tree(source, staging)
        values = load_teacher_store(staging)
        if len(values) != 1:
            raise ObservationSeedError("source observation store must contain exactly one window")
        if (staging / "chunks").is_dir():
            _fsync_directory(staging / "chunks")
        _fsync_directory(staging)
        os.rename(staging, destination)
        _fsync_directory(destination.parent)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def seed_committed_prefix(*, source_root: str | Path, source_config: str | Path,
                          source_config_sha256: str, target_config: str | Path,
                          target_config_sha256: str, equivalence_report: str | Path,
                          equivalence_sha256: str, probe_output: str | Path,
                          probe_config_sha256: str, reserved_free_bytes: int = RESERVED_FREE_BYTES) -> dict:
    """Hardlink only source windows that are already committed and validate each destination."""
    source_config, target_config = Path(source_config), Path(target_config)
    source_cfg = bound_json(source_config, source_config_sha256)
    target_cfg = bound_json(target_config, target_config_sha256)
    source_root, probe_output = Path(source_root), Path(probe_output)
    source_run = source_root / "run.json"
    probe_run = probe_output / "run.json"
    source_run_sha256, probe_run_sha256 = sha256_file(source_run), sha256_file(probe_run)
    report = bound_json(equivalence_report, equivalence_sha256)
    parent = target_cfg.get("resume_parent")
    if not isinstance(parent, dict):
        raise ObservationSeedError("target config has no resume-parent binding")
    validate_equivalence(report, parent)
    source_document = bound_json(source_run, source_run_sha256)
    probe_document = bound_json(probe_run, probe_run_sha256)
    source_binding, window_ids = source_document.get("binding"), tuple(source_document.get("window_ids", ()))
    probe_binding = probe_document.get("binding")
    if not isinstance(parent.get("probe_config"), dict):
        raise ObservationSeedError("target config has no probe parent binding")
    probe_config_path = Path(parent["probe_config"].get("path", ""))
    probe_cfg = bound_json(probe_config_path, probe_config_sha256)
    validate_config_delta(source_cfg, target_cfg, probe_cfg, source_config_path=source_config,
                          source_config_sha256=source_config_sha256, source_run_path=source_run,
                          source_run_sha256=source_run_sha256, equivalence_path=Path(equivalence_report),
                          equivalence_sha256=equivalence_sha256, probe_config_path=probe_config_path,
                          probe_config_sha256=probe_config_sha256, probe_run_path=probe_run,
                          probe_run_sha256=probe_run_sha256)
    if (Path(source_cfg.get("output", "")).resolve() != source_root.resolve()
            or source_binding.get("config_sha256") != source_config_sha256):
        raise ObservationSeedError("source root is not the hash-bound parent observation run")
    if (source_document.get("schema") != "nc_rted_observation_journal/v1" or not isinstance(source_binding, dict)
            or not window_ids or probe_document.get("schema") != source_document["schema"]
            or not isinstance(probe_binding, dict) or probe_binding.get("config_sha256") != probe_config_sha256
            or tuple(probe_document.get("window_ids", ())) != window_ids):
        raise ObservationSeedError("source or probe journal identity is incompatible")
    binding = dict(probe_binding)
    binding["config_sha256"] = target_config_sha256
    destination = Path(target_cfg["output"])
    source_journal = ObservationJournal(source_root, source_binding, window_ids, reserved_free_bytes=reserved_free_bytes)
    probe_journal = ObservationJournal(probe_output, probe_binding, window_ids, reserved_free_bytes=reserved_free_bytes)
    for row in report["three_windows"]:
        window_id = row["window_id"]
        if window_id not in window_ids:
            raise ObservationSeedError("equivalence report names a foreign window")
        source_window, probe_window = source_journal._window_path(window_id), probe_journal._window_path(window_id)
        if not (source_window / "commit.json").is_file() or not (probe_window / "commit.json").is_file():
            raise ObservationSeedError("equivalence report lacks committed source or probe window")
        if _tree_digest(source_window) != _tree_digest(probe_window):
            raise ObservationSeedError("equivalence report bytes do not match its bound journals")
    destination_journal = ObservationJournal(destination, binding, window_ids, reserved_free_bytes=reserved_free_bytes)
    copied, reused = [], []
    with destination_journal.writer() as journal:
        for window_id in window_ids:
            source_window = source_journal._window_path(window_id)
            if not (source_window / "commit.json").is_file():
                continue
            source_value = source_journal.read(window_id)
            if source_value is None:
                continue
            destination_window = journal._window_path(window_id)
            if destination_window.exists() and not (destination_window / "commit.json").is_file():
                raise ObservationSeedError("destination has an uncommitted window and will not be overwritten")
            destination_value = journal.read(window_id)
            if destination_value is not None:
                if (destination_value.record.get("window_id") != source_value.record.get("window_id")
                        or _tree_digest(destination_window) != _tree_digest(source_window)):
                    raise ObservationSeedError("existing destination window differs from source")
                reused.append(window_id)
                continue
            with allocation_lock(destination):
                journal._reserve(3 << 20)
                _publish_linked_window(source_window, destination_window)
            validated = journal.read(window_id)
            if validated is None or validated.record.get("window_id") != window_id:
                raise ObservationSeedError("linked destination window did not validate")
            copied.append(window_id)
    receipt = {"schema": "nc_rted_observation_seed_receipt/v1", "status": "PARTIAL_RESUMABLE_OBSERVATIONS",
               "source_run_sha256": source_run_sha256, "target_config_sha256": target_config_sha256,
               "copied_window_ids": copied, "reused_window_ids": reused, "total": len(window_ids),
               "completed_after_seed": len(copied) + len(reused), "sealed": False}
    receipts = destination / "seed_receipts"
    receipts.mkdir(exist_ok=True)
    atomic_json(receipts / (uuid.uuid4().hex + ".json"), receipt, reserved_free_bytes=reserved_free_bytes)
    return receipt


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(root.rglob("*")):
        if item.is_symlink() or not item.is_file():
            continue
        digest.update(str(item.relative_to(root)).encode("utf-8") + b"\0")
        digest.update(sha256_file(item).encode("ascii"))
    return digest.hexdigest()
