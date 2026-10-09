"""Atomic, chunked persistence for compact NC-RTED teacher observations."""
from __future__ import annotations

import hashlib
import json
import os
import ctypes
import errno
from pathlib import Path
import shutil
import tempfile

import numpy as np

from .features import PROCESS_FEATURE_DIM
from .teacher_records import CompactTeacherWindow, TeacherRecordError, build_teachers_from_compact


STORE_SCHEMA = "nc_rted_teacher_store/v1"
RESERVED_FREE_BYTES = 20 * 1024 ** 3


class TeacherStoreError(ValueError):
    pass


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(8 << 20), b""):
            digest.update(part)
    return digest.hexdigest()


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _require_reserve(directory: Path, reserved_free_bytes: int) -> None:
    if shutil.disk_usage(directory).free <= reserved_free_bytes:
        raise TeacherStoreError("teacher store would violate the 20 GiB free-space reserve")


def _rename_noreplace(source: Path, destination: Path) -> None:
    if destination.exists():
        raise TeacherStoreError("refusing to overwrite an existing teacher store")
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1) != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise TeacherStoreError("refusing to overwrite an existing teacher store")
        raise OSError(error, os.strerror(error), str(destination))


def _identity(record: dict, relation_ids: tuple[str, ...]) -> dict:
    fields = ("dataset", "window_id", "source_family", "content_alias", "fold", "normal_permitted")
    if any(field not in record for field in fields):
        raise TeacherStoreError("accepted compact record lacks source/fold identity")
    return {field: record[field] for field in fields} | {"relation_ids": list(relation_ids)}


def _save_payload(path: Path, compact: CompactTeacherWindow) -> None:
    record = compact.record
    pairs = record["pairs"]
    count = len(pairs)
    if count > 16 or tuple(compact.relation_ids) != tuple(pair["pair_id"] for pair in pairs):
        raise TeacherStoreError("compact pair IDs do not match payload order")
    process = np.stack([np.asarray(pair["process_cells"]) for pair in pairs]) if pairs else np.empty((0, 4, PROCESS_FEATURE_DIM), dtype=np.float32)
    np.savez_compressed(
        path,
        background=np.asarray(record["background"]),
        class_composition=np.asarray(record["class_composition"]),
        pair_ids=np.asarray([pair["pair_id"] for pair in pairs], dtype=np.str_),
        class_pairs=np.asarray([pair["class_pair"] for pair in pairs], dtype=np.int16).reshape(count, 2),
        initial_geometry=np.asarray([pair["initial_geometry"] for pair in pairs]).reshape(count, 5),
        candidate_pair_count=np.asarray([pair["candidate_pair_count"] for pair in pairs], dtype=np.int16),
        valid_cells=np.asarray([pair["valid_cells"] for pair in pairs], dtype=np.bool_).reshape(count, 4),
        process_cells=process,
    )


def write_teacher_store(output: str | Path, compacts: tuple[CompactTeacherWindow, ...],
                        *, reserved_free_bytes: int = RESERVED_FREE_BYTES) -> None:
    """Write a new store atomically; a partially staged store is never visible."""
    destination = Path(output)
    if not compacts:
        raise TeacherStoreError("teacher store requires at least one compact window")
    if destination.exists():
        raise TeacherStoreError("refusing to overwrite an existing teacher store")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if reserved_free_bytes < 0:
        raise TeacherStoreError("reserved free bytes must be nonnegative")
    _require_reserve(destination.parent, reserved_free_bytes)
    identifiers = [item.record.get("window_id") for item in compacts]
    if any(not isinstance(item, str) or not item for item in identifiers) or len(set(identifiers)) != len(identifiers):
        raise TeacherStoreError("compact windows need unique nonempty IDs")
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent))
    try:
        chunks = staging / "chunks"
        chunks.mkdir()
        entries = []
        for number, compact in enumerate(sorted(compacts, key=lambda item: item.record["window_id"])):
            record = compact.record
            if compact.rejection is not None:
                entries.append({"window_id": record["window_id"], "dataset": record.get("dataset"),
                                "kind": "rejection", "rejection": record.get("rejection"),
                                "identity_sha256": _sha256_bytes(_canonical({"window_id": record["window_id"], "dataset": record.get("dataset")}))})
                continue
            identity = _identity(record, compact.relation_ids)
            relative = f"chunks/{number:06d}.npz"
            payload = staging / relative
            _require_reserve(destination.parent, reserved_free_bytes)
            _save_payload(payload, compact)
            _fsync_file(payload)
            _require_reserve(destination.parent, reserved_free_bytes)
            entries.append({"window_id": record["window_id"], "dataset": record["dataset"], "kind": "record",
                            "path": relative, "payload_sha256": _sha256_file(payload), "identity": identity,
                            "identity_sha256": _sha256_bytes(_canonical(identity))})
        index = {"schema": STORE_SCHEMA, "entries": entries}
        index_path = staging / "index.json"
        index_bytes = _canonical(index) + b"\n"
        commit_bytes = _canonical({"schema": STORE_SCHEMA, "index_sha256": _sha256_bytes(index_bytes)}) + b"\n"
        # Include block rounding and directory metadata, also for rejection-only
        # stores which create no NPZ chunks.
        allocation = sum(((len(value) + 4095) // 4096) * 4096 for value in (index_bytes, commit_bytes)) + 8192
        _require_reserve(destination.parent, reserved_free_bytes + allocation)
        index_path.write_bytes(index_bytes)
        _fsync_file(index_path)
        commit_path = staging / "commit.json"
        commit_path.write_bytes(commit_bytes)
        _fsync_file(commit_path)
        _fsync_directory(chunks)
        _fsync_directory(staging)
        _require_reserve(destination.parent, reserved_free_bytes)
        _rename_noreplace(staging, destination)
        _fsync_directory(destination.parent)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _load_index(root: Path) -> dict:
    commit_path, index_path = root / "commit.json", root / "index.json"
    if not root.is_dir() or not commit_path.is_file() or not index_path.is_file():
        raise TeacherStoreError("teacher store is absent or not committed")
    try:
        commit = json.loads(commit_path.read_text())
        index = json.loads(index_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise TeacherStoreError("teacher store metadata is unreadable") from error
    if commit.get("schema") != STORE_SCHEMA or index.get("schema") != STORE_SCHEMA:
        raise TeacherStoreError("unsupported teacher store schema")
    if commit.get("index_sha256") != _sha256_file(index_path):
        raise TeacherStoreError("teacher store index hash mismatch")
    if not isinstance(index.get("entries"), list) or not index["entries"]:
        raise TeacherStoreError("teacher store index has no entries")
    return index


def _load_payload(root: Path, entry: dict) -> CompactTeacherWindow:
    identity = entry.get("identity")
    if not isinstance(identity, dict) or entry.get("identity_sha256") != _sha256_bytes(_canonical(identity)):
        raise TeacherStoreError("teacher record identity hash mismatch")
    relative = entry.get("path")
    if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise TeacherStoreError("invalid teacher payload path")
    payload = root / relative
    if not payload.is_file() or entry.get("payload_sha256") != _sha256_file(payload):
        raise TeacherStoreError("teacher payload hash mismatch")
    try:
        with np.load(payload, allow_pickle=False) as data:
            expected = {"background", "class_composition", "pair_ids", "class_pairs", "initial_geometry",
                        "candidate_pair_count", "valid_cells", "process_cells"}
            if set(data.files) != expected:
                raise TeacherStoreError("teacher payload array schema mismatch")
            arrays = {name: data[name] for name in expected}
    except (OSError, ValueError) as error:
        raise TeacherStoreError("teacher payload is unreadable") from error
    pair_ids = arrays["pair_ids"].tolist()
    count = len(pair_ids)
    if (arrays["background"].shape != (1152,) or arrays["class_composition"].shape != (80,)
            or arrays["class_pairs"].shape != (count, 2) or arrays["initial_geometry"].shape != (count, 5)
            or arrays["candidate_pair_count"].shape != (count,) or arrays["valid_cells"].shape != (count, 4)
            or arrays["process_cells"].shape != (count, 4, PROCESS_FEATURE_DIM)):
        raise TeacherStoreError("teacher payload shapes are invalid")
    pairs = [{"pair_id": str(pair_ids[index]), "class_pair": arrays["class_pairs"][index].astype(int).tolist(),
              "initial_geometry": arrays["initial_geometry"][index], "candidate_pair_count": int(arrays["candidate_pair_count"][index]),
              "valid_cells": arrays["valid_cells"][index], "process_cells": arrays["process_cells"][index]}
             for index in range(count)]
    record = {field: identity[field] for field in ("dataset", "window_id", "source_family", "content_alias", "fold", "normal_permitted")}
    record |= {"background": arrays["background"], "background_valid": True,
               "class_composition": arrays["class_composition"], "pairs": pairs}
    return CompactTeacherWindow(record=record, relation_ids=tuple(str(item) for item in pair_ids))


def load_teacher_store(path: str | Path) -> tuple[CompactTeacherWindow, ...]:
    """Load all compact records after validating commit, identity, and payload hashes."""
    root = Path(path)
    index = _load_index(root)
    result = []
    for entry in index["entries"]:
        if entry.get("kind") == "record":
            result.append(_load_payload(root, entry))
        elif entry.get("kind") == "rejection":
            identity = {"window_id": entry.get("window_id"), "dataset": entry.get("dataset")}
            if entry.get("identity_sha256") != _sha256_bytes(_canonical(identity)) or not entry.get("rejection"):
                raise TeacherStoreError("teacher rejection index entry is invalid")
            result.append(CompactTeacherWindow(record={"window_id": identity["window_id"], "dataset": identity["dataset"],
                                                       "aux_valid": False, "rejection": entry["rejection"]},
                                               relation_ids=(), rejection={"reason": entry["rejection"]}))
        else:
            raise TeacherStoreError("unknown teacher store entry kind")
    if len({item.record["window_id"] for item in result}) != len(result):
        raise TeacherStoreError("duplicate teacher store window IDs")
    return tuple(result)


def build_teachers_from_store(path: str | Path) -> dict:
    """Read the committed chunks and construct teachers without materializing JSON vectors."""
    return build_teachers_from_compact(load_teacher_store(path))
