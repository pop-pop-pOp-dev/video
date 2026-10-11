"""Bounded, acknowledgement-gated transport for immutable frozen records.

The producer cache is intentionally separate from the remote training cache.
This module never changes a published record: it stages a tensor, verifies the
producer binding, then links data and publishes the binding under the existing
allocation lock.  A consumer-valid ACK is the only authority for deleting a
recomputable local record.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from typing import Callable, Mapping

from .frozen_sample_cache import (
    SCHEMA as RECORD_SCHEMA,
    FrozenSampleCacheError,
    FrozenSampleStore,
    _record_key,
    canonical,
    digest_path,
)
from .storage_lock import allocation_lock, ensure_directory
from .task_inputs import TrainingTask


ACK_SCHEMA = "nc_rted_frozen_sample_ack/v1"


class FrozenSampleTransportError(RuntimeError):
    pass


def _json_bytes(value: object, label: str) -> bytes:
    try:
        return canonical(value)
    except FrozenSampleCacheError as error:
        raise FrozenSampleTransportError(f"{label} is not canonical JSON") from error


def _document(content: bytes, label: str) -> dict:
    try:
        value = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FrozenSampleTransportError(f"{label} is invalid JSON") from error
    if not isinstance(value, dict) or _json_bytes(value, label) != content:
        raise FrozenSampleTransportError(f"{label} is not canonical JSON")
    return value


def task_document(task: TrainingTask) -> dict:
    if not isinstance(task, TrainingTask):
        raise FrozenSampleTransportError("transport task is invalid")
    return asdict(task)


def task_from_document(value: Mapping[str, object]) -> TrainingTask:
    expected = {"sample_id", "task", "dataset", "family", "media_key", "observed_seconds",
                "label", "instruction", "query_index"}
    if not isinstance(value, Mapping) or set(value) != expected:
        raise FrozenSampleTransportError("transport task document differs")
    try:
        task = TrainingTask(**dict(value))
    except (TypeError, ValueError) as error:
        raise FrozenSampleTransportError("transport task document is invalid") from error
    if not isinstance(task.sample_id, str) or task.task not in {"caption", "detection"}:
        raise FrozenSampleTransportError("transport task identity is invalid")
    return task


def _allocation(path: Path) -> int:
    stat = path.stat()
    return max(stat.st_size, getattr(stat, "st_blocks", 0) * 512)


def _tree_bytes(path: Path) -> int:
    if not path.is_dir():
        return 0
    return sum(_allocation(item) for item in path.iterdir() if item.is_file())


def cache_status(root: str | Path, *, max_bytes: int, min_free_bytes: int,
                 next_bytes: int = 0) -> dict:
    """Report one bounded cache admission decision without allocating data."""
    if (type(max_bytes) is not int or type(min_free_bytes) is not int or type(next_bytes) is not int or
            max_bytes <= 0 or min_free_bytes < 20 << 30 or next_bytes < 0):
        raise FrozenSampleTransportError("transport resource limits are invalid")
    location = Path(root).resolve()
    usage = _tree_bytes(location / "records") + _tree_bytes(location / "acks")
    free = shutil.disk_usage(location).free
    return {"schema": "nc_rted_frozen_sample_transport_status/v1", "root": str(location),
            "cache_bytes": usage, "free_bytes": free, "max_bytes": max_bytes,
            "min_free_bytes": min_free_bytes, "next_bytes": next_bytes,
            "admitted": usage + next_bytes <= max_bytes and free >= min_free_bytes + next_bytes}


def prepare_transport_root(root: str | Path, *, min_free_bytes: int) -> None:
    """Create only transport metadata directories under the shared reserve lock."""
    if type(min_free_bytes) is not int or min_free_bytes < 20 << 30:
        raise FrozenSampleTransportError("transport free-space reserve is invalid")
    location = Path(root).resolve()
    ensure_directory(location / ".incoming", min_free_bytes)
    ensure_directory(location / "acks", min_free_bytes)


def require_capacity(status: Mapping[str, object], *, where: str) -> None:
    if status.get("admitted") is not True:
        raise FrozenSampleTransportError(
            f"{where} capacity backpressure: cache/free-space admission is unavailable before provider")


def wait_for_capacity(status: Callable[[], Mapping[str, object]], *, timeout_seconds: float,
                      poll_seconds: float, where: str) -> dict:
    if timeout_seconds <= 0 or poll_seconds <= 0:
        raise FrozenSampleTransportError("capacity wait bounds are invalid")
    deadline = time.monotonic() + timeout_seconds
    while True:
        current = dict(status())
        if current.get("admitted") is True:
            return current
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise FrozenSampleTransportError(
                f"{where} capacity backpressure timed out before provider; advance or reclaim a completed window")
        time.sleep(min(poll_seconds, remaining))


def _binding_for(stage_binding: Path, stage_tensor: Path, task: TrainingTask) -> tuple[dict, bytes]:
    content = stage_binding.read_bytes()
    binding = _document(content, "staged producer binding")
    expected = {"schema", "sample_id", "identity", "tensor_path", "tensor_sha256"}
    if (set(binding) != expected or binding.get("schema") != RECORD_SCHEMA or
            binding.get("sample_id") != task.sample_id or not isinstance(binding.get("identity"), dict) or
            not binding["identity"] or binding.get("tensor_path") != f"{_record_key(task.sample_id)}.pt" or
            binding.get("tensor_sha256") != digest_path(stage_tensor)):
        raise FrozenSampleTransportError("staged tensor does not match its full producer binding")
    return binding, content


def ack_path(root: str | Path, sample_id: str) -> Path:
    return Path(root).resolve() / "acks" / f"{_record_key(sample_id)}.json"


def _ack(binding: dict, binding_content: bytes, tensor_path: Path) -> bytes:
    return _json_bytes({"schema": ACK_SCHEMA, "sample_id": binding["sample_id"],
                        "identity": binding["identity"], "tensor_path": tensor_path.name,
                        "tensor_sha256": binding["tensor_sha256"],
                        "binding_sha256": hashlib.sha256(binding_content).hexdigest()}, "acknowledgement")


def receive_staged(*, root: str | Path, staged_tensor: str | Path, staged_binding: str | Path,
                   task: TrainingTask, max_bytes: int = 80 << 30,
                   min_free_bytes: int = 20 << 30) -> dict:
    """Atomically publish a staged remote record, validate it, then ACK it.

    Staging files must be inside ``root/.incoming`` so the data hard-link stays
    on the same filesystem.  The final binding is the visibility boundary.
    """
    store_root = Path(root).resolve()
    incoming = store_root / ".incoming"
    stage_tensor = Path(staged_tensor).resolve()
    stage_binding = Path(staged_binding).resolve()
    try:
        stage_tensor.relative_to(incoming)
        stage_binding.relative_to(incoming)
    except ValueError as error:
        raise FrozenSampleTransportError("staged files must remain in the remote incoming directory") from error
    if not stage_tensor.is_file() or not stage_binding.is_file():
        raise FrozenSampleTransportError("staged transfer is incomplete")
    binding, binding_content = _binding_for(stage_binding, stage_tensor, task)
    store = FrozenSampleStore(store_root, identity=binding["identity"], max_bytes=max_bytes,
                              min_free_bytes=min_free_bytes)
    acks = store_root / "acks"
    ensure_directory(incoming, min_free_bytes)
    ensure_directory(acks, min_free_bytes)
    tensor_path, binding_path = store.paths(task.sample_id)
    acknowledgement = _ack(binding, binding_content, tensor_path)
    with allocation_lock(store_root):
        cache_usage = _tree_bytes(store.records) + _tree_bytes(acks)
        record_addition = 0 if tensor_path.exists() else _allocation(stage_tensor)
        binding_addition = 0 if binding_path.exists() else max(4096, len(binding_content))
        ack_target = ack_path(store_root, task.sample_id)
        ack_addition = 0 if ack_target.exists() else max(4096, len(acknowledgement))
        if cache_usage + record_addition + binding_addition + ack_addition > max_bytes:
            raise FrozenSampleTransportError("remote frozen cache upper bound would be exceeded")
        # The staged tensor already occupies its allocation; publishing it is a
        # hard link.  Only metadata and the ACK require additional free space.
        if shutil.disk_usage(store_root).free < min_free_bytes + binding_addition + ack_addition:
            raise FrozenSampleTransportError("remote frozen cache would violate the free-space reserve")
        try:
            os.link(stage_tensor, tensor_path)
        except FileExistsError:
            if digest_path(tensor_path) != binding["tensor_sha256"]:
                raise FrozenSampleTransportError("remote immutable tensor already differs")
        store._write_once(binding_path, binding_content)
    try:
        if store.get(task) is None:
            raise FrozenSampleTransportError("remote binding did not publish a readable record")
    except FrozenSampleCacheError as error:
        raise FrozenSampleTransportError("remote record failed consumer validation") from error
    with allocation_lock(store_root):
        store._write_once(ack_target, acknowledgement)
    stage_tensor.unlink(missing_ok=True)
    stage_binding.unlink(missing_ok=True)
    return {"schema": "nc_rted_frozen_sample_receive/v1", "record": store._record(task),
            "ack_path": str(ack_target), "ack_sha256": digest_path(ack_target),
            "sample_id": task.sample_id, "identity": binding["identity"]}


def verify_ack(*, root: str | Path, task: TrainingTask, identity: dict, ack: str | Path) -> dict:
    store = FrozenSampleStore(root, identity=identity)
    tensor_path, binding_path = store.paths(task.sample_id)
    try:
        binding_content = binding_path.read_bytes()
        binding = _document(binding_content, "local producer binding")
        acknowledgement = _document(Path(ack).read_bytes(), "remote acknowledgement")
    except OSError as error:
        raise FrozenSampleTransportError("acknowledgement or local record is absent") from error
    required = {"schema", "sample_id", "identity", "tensor_path", "tensor_sha256", "binding_sha256"}
    if (set(acknowledgement) != required or acknowledgement.get("schema") != ACK_SCHEMA or
            acknowledgement.get("sample_id") != task.sample_id or acknowledgement.get("identity") != store.identity or
            acknowledgement.get("tensor_path") != tensor_path.name or
            acknowledgement.get("tensor_sha256") != digest_path(tensor_path) or
            acknowledgement.get("binding_sha256") != hashlib.sha256(binding_content).hexdigest() or
            binding.get("identity") != store.identity):
        raise FrozenSampleTransportError("acknowledgement does not bind this exact local record")
    try:
        if store.get(task) is None:
            raise FrozenSampleTransportError("local record is absent during acknowledgement verification")
    except FrozenSampleCacheError as error:
        raise FrozenSampleTransportError("local record is invalid during acknowledgement verification") from error
    return acknowledgement


def install_ack(*, root: str | Path, task: TrainingTask, identity: dict, ack: str | Path,
                min_free_bytes: int = 20 << 30) -> Path:
    acknowledgement = verify_ack(root=root, task=task, identity=identity, ack=ack)
    target = ack_path(root, task.sample_id)
    ensure_directory(target.parent, min_free_bytes)
    with allocation_lock(root):
        FrozenSampleStore._write_once(target, _json_bytes(acknowledgement, "acknowledgement"))
    return target


def evict_local_after_ack(*, root: str | Path, task: TrainingTask, identity: dict,
                          ack: str | Path | None = None) -> None:
    """Delete only a recomputable local cache record after an exact ACK check."""
    acknowledgement = Path(ack) if ack is not None else ack_path(root, task.sample_id)
    verify_ack(root=root, task=task, identity=identity, ack=acknowledgement)
    store = FrozenSampleStore(root, identity=identity)
    tensor_path, binding_path = store.paths(task.sample_id)
    with allocation_lock(root):
        # Recheck inside the deletion transition; the acknowledgement is exact.
        verify_ack(root=root, task=task, identity=identity, ack=acknowledgement)
        binding_path.unlink()
        tensor_path.unlink()
        directory = os.open(store.records, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
