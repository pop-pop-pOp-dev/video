"""Read-only consumer for immutable GPU-produced frozen sample records."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from .bridge import ObservationBatch
from .task_inputs import FrozenTaskContext, TaskInputError, TrainingTask, validate_observation_scope
from .train_worker import SampleMaterial


SCHEMA = "nc_rted_frozen_sample_record/v1"


class FrozenSampleStoreError(RuntimeError):
    pass


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"
    except (TypeError, ValueError) as error:
        raise FrozenSampleStoreError("producer identity is not canonical JSON") from error


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def _record_key(sample_id: str) -> str:
    if not isinstance(sample_id, str) or not sample_id:
        raise FrozenSampleStoreError("sample ID is invalid")
    return hashlib.sha256(sample_id.encode()).hexdigest()


class FrozenSampleStore:
    """Validate the producer's published binding before exposing CPU material.

    This is intentionally read-only.  The 4090 producer owns publication and
    capacity admission; consumers can neither repair nor regenerate a record.
    """

    def __init__(self, root: str | Path, *, identity: dict):
        if not isinstance(identity, dict) or not identity:
            raise FrozenSampleStoreError("producer identity must be a nonempty mapping")
        self.root = Path(root).resolve()
        self.records = self.root / "records"
        self.identity = json.loads(_canonical(identity))

    def paths(self, sample_id: str) -> tuple[Path, Path]:
        key = _record_key(sample_id)
        return self.records / f"{key}.pt", self.records / f"{key}.json"

    @staticmethod
    def _material(task: TrainingTask, metadata: dict, tensors: dict) -> SampleMaterial:
        if metadata.get("sample_id") != task.sample_id or metadata.get("task") != task.task:
            raise FrozenSampleStoreError("cached record names another training task")
        context_data = metadata.get("context")
        if not isinstance(context_data, dict):
            raise FrozenSampleStoreError("cached context is absent")
        required = {"visual_embeddings", "image_0", "features", "valid", "observed_times"}
        if not isinstance(tensors, dict) or set(tensors) != required:
            raise FrozenSampleStoreError("cached tensor set differs")
        if any(not isinstance(value, torch.Tensor) or value.device.type != "cpu" or value.requires_grad
               for value in tensors.values()):
            raise FrozenSampleStoreError("cached tensors must be detached CPU tensors")
        try:
            material = SampleMaterial(
                FrozenTaskContext(tensors["visual_embeddings"], [tensors["image_0"]],
                                  [tuple(size) for size in context_data["image_sizes"]],
                                  context_data["observed_seconds"], tuple(context_data["sampled_frame_times"]),
                                  context_data["time_message"]),
                ObservationBatch(tensors["features"], tensors["valid"], tensors["observed_times"]),
                tuple(metadata["relation_ids"]), metadata["detection_question"], metadata["detection_scoring"],
            )
            validate_observation_scope(task.task, material.context, material.observations)
        except (KeyError, TypeError, ValueError, TaskInputError) as error:
            raise FrozenSampleStoreError("cached frozen material is malformed") from error
        return material

    def get(self, task: TrainingTask) -> SampleMaterial | None:
        tensor_path, binding_path = self.paths(task.sample_id)
        if not binding_path.is_file():
            return None
        try:
            binding = json.loads(binding_path.read_text())
            if (binding.get("schema") != SCHEMA or binding.get("sample_id") != task.sample_id or
                    binding.get("identity") != self.identity or binding.get("tensor_path") != tensor_path.name or
                    binding.get("tensor_sha256") != _digest(tensor_path)):
                raise FrozenSampleStoreError("record binding differs")
            body = torch.load(tensor_path, map_location="cpu", weights_only=True)
            if (not isinstance(body, dict) or body.get("schema") != SCHEMA or
                    body.get("identity") != self.identity):
                raise FrozenSampleStoreError("record payload differs")
            return self._material(task, body["metadata"], body["tensors"])
        except (OSError, KeyError, ValueError, RuntimeError, EOFError, FrozenSampleStoreError) as error:
            raise FrozenSampleStoreError(f"invalid frozen sample record: {binding_path}") from error
