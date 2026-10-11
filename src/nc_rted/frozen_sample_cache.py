"""Immutable, portable frozen ``SampleMaterial`` records for GPU producers."""
from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import tempfile
from pathlib import Path

import torch

from .bridge import ObservationBatch
from .task_inputs import FrozenTaskContext, TaskInputError, TrainingTask, validate_observation_scope
from .train_worker import SampleMaterial
from .storage_lock import allocation_lock, ensure_directory


SCHEMA = "nc_rted_frozen_sample_record/v1"


class FrozenSampleCacheError(RuntimeError):
    pass


def canonical(value: object) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"
    except (TypeError, ValueError) as error:
        raise FrozenSampleCacheError("record metadata is not canonical JSON") from error


def digest_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _record_key(sample_id: str) -> str:
    if not isinstance(sample_id, str) or not sample_id:
        raise FrozenSampleCacheError("sample ID is invalid")
    return hashlib.sha256(sample_id.encode()).hexdigest()


def _cpu_tensor(value: torch.Tensor, label: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.requires_grad:
        raise FrozenSampleCacheError(f"{label} is not a frozen tensor")
    return value.detach().cpu().contiguous().clone()


def _payload(task: TrainingTask, material: SampleMaterial) -> tuple[dict, dict]:
    if not isinstance(material, SampleMaterial):
        raise FrozenSampleCacheError("producer did not return SampleMaterial")
    validate_observation_scope(task.task, material.context, material.observations)
    context, observations = material.context, material.observations
    if len(context.images) != 1 or not isinstance(context.images[0], torch.Tensor):
        raise FrozenSampleCacheError("material does not contain exactly one image tensor")
    metadata = {
        "sample_id": task.sample_id,
        "task": task.task,
        "context": {
            "image_sizes": [list(size) for size in context.image_sizes],
            "observed_seconds": context.observed_seconds,
            "sampled_frame_times": list(context.sampled_frame_times),
            "time_message": context.time_message,
        },
        "relation_ids": list(material.relation_ids),
        "detection_question": material.detection_question,
        "detection_scoring": material.detection_scoring,
    }
    tensors = {
        "visual_embeddings": _cpu_tensor(context.visual_embeddings, "visual embeddings"),
        "image_0": _cpu_tensor(context.images[0], "image tensor"),
        "features": _cpu_tensor(observations.features, "observation features"),
        "valid": _cpu_tensor(observations.valid, "observation mask"),
        "observed_times": _cpu_tensor(observations.observed_times, "observation times"),
    }
    return metadata, tensors


def _material(task: TrainingTask, metadata: dict, tensors: dict) -> SampleMaterial:
    if metadata.get("sample_id") != task.sample_id or metadata.get("task") != task.task:
        raise FrozenSampleCacheError("cached record names another training task")
    context_data = metadata.get("context")
    if not isinstance(context_data, dict):
        raise FrozenSampleCacheError("cached context is absent")
    try:
        context = FrozenTaskContext(
            tensors["visual_embeddings"], [tensors["image_0"]],
            [tuple(size) for size in context_data["image_sizes"]], context_data["observed_seconds"],
            tuple(context_data["sampled_frame_times"]), context_data["time_message"],
        )
        material = SampleMaterial(
            context, ObservationBatch(tensors["features"], tensors["valid"], tensors["observed_times"]),
            tuple(metadata["relation_ids"]), metadata["detection_question"], metadata["detection_scoring"],
        )
        validate_observation_scope(task.task, material.context, material.observations)
    except (KeyError, TypeError, ValueError, TaskInputError) as error:
        raise FrozenSampleCacheError("cached frozen material is malformed") from error
    return material


class FrozenSampleStore:
    """Append-only per-sample records; the JSON binding is the publish point."""

    def __init__(self, root: str | Path, *, identity: dict, max_bytes: int = 80 << 30,
                 min_free_bytes: int = 20 << 30):
        if not isinstance(identity, dict) or not identity:
            raise FrozenSampleCacheError("producer identity must be a nonempty mapping")
        if (type(max_bytes) is not int or type(min_free_bytes) is not int
                or max_bytes <= 0 or min_free_bytes < 20 << 30):
            raise FrozenSampleCacheError("cache resource limits are invalid")
        self.root = Path(root).resolve()
        self.identity = json.loads(canonical(identity))
        self.max_bytes, self.min_free_bytes = max_bytes, min_free_bytes
        self.records = self.root / "records"
        ensure_directory(self.records, self.min_free_bytes)

    def paths(self, sample_id: str) -> tuple[Path, Path]:
        key = _record_key(sample_id)
        return self.records / f"{key}.pt", self.records / f"{key}.json"

    @staticmethod
    def _allocated_bytes(path: Path) -> int:
        """Report filesystem allocation, falling back to the visible byte size."""
        stat = path.stat()
        return max(stat.st_size, getattr(stat, "st_blocks", 0) * 512)

    @staticmethod
    def _estimate_allocation(*contents: bytes) -> int:
        # Filesystems commonly allocate at least 4 KiB per new file.  Use that
        # lower bound before admission, then count real allocations thereafter.
        return sum(((len(content) + 4095) // 4096) * 4096 for content in contents)

    @staticmethod
    def _write_once(path: Path, content: bytes) -> None:
        temporary = None
        try:
            descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
            temporary = Path(temporary_name)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content); stream.flush(); os.fsync(stream.fileno())
            os.link(temporary, path)
            directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except FileExistsError:
            if path.read_bytes() != content:
                raise FrozenSampleCacheError(f"immutable record already differs: {path}")
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def _record(self, task: TrainingTask) -> dict:
        tensor_path, binding_path = self.paths(task.sample_id)
        return {
            "path": str(binding_path),
            "sha256": digest_path(binding_path),
            "bytes": self._allocated_bytes(tensor_path) + self._allocated_bytes(binding_path),
        }

    def put(self, task: TrainingTask, material: SampleMaterial) -> dict:
        metadata, tensors = _payload(task, material)
        body = {"schema": SCHEMA, "identity": self.identity, "metadata": metadata, "tensors": tensors}
        serialized = io.BytesIO(); torch.save(body, serialized)
        tensor_path, binding_path = self.paths(task.sample_id)
        tensor_content = serialized.getvalue()
        binding = {
            "schema": SCHEMA, "sample_id": task.sample_id, "identity": self.identity,
            "tensor_path": tensor_path.name, "tensor_sha256": hashlib.sha256(tensor_content).hexdigest(),
        }
        binding_content = canonical(binding)
        # Capacity admission and both publications share one process-wide lock.
        # The binding remains the publish point, after the immutable tensor data.
        with allocation_lock(self.root):
            if not tensor_path.exists():
                current = sum(self._allocated_bytes(path) for path in self.records.iterdir() if path.is_file())
                additional = self._estimate_allocation(tensor_content, binding_content)
                if current + additional > self.max_bytes:
                    raise FrozenSampleCacheError("frozen sample cache upper bound would be exceeded")
                if shutil.disk_usage(self.root).free < self.min_free_bytes + additional:
                    raise FrozenSampleCacheError("frozen sample cache would violate the free-space reserve")
            self._write_once(tensor_path, tensor_content)
            self._write_once(binding_path, binding_content)
            return self._record(task)

    def get(self, task: TrainingTask) -> SampleMaterial | None:
        tensor_path, binding_path = self.paths(task.sample_id)
        if not binding_path.is_file():
            return None
        try:
            binding = json.loads(binding_path.read_text())
            if (binding.get("schema") != SCHEMA or binding.get("sample_id") != task.sample_id
                    or binding.get("identity") != self.identity or binding.get("tensor_path") != tensor_path.name
                    or binding.get("tensor_sha256") != digest_path(tensor_path)):
                raise FrozenSampleCacheError("record binding differs")
            body = torch.load(tensor_path, map_location="cpu", weights_only=True)
            if not isinstance(body, dict) or body.get("schema") != SCHEMA or body.get("identity") != self.identity:
                raise FrozenSampleCacheError("record payload differs")
            return _material(task, body["metadata"], body["tensors"])
        except (OSError, KeyError, ValueError, RuntimeError, EOFError, FrozenSampleCacheError) as error:
            raise FrozenSampleCacheError(f"invalid frozen sample record: {binding_path}") from error
