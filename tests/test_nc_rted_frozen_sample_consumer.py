import hashlib
import json
import time

import pytest
import torch

from nc_rted.bridge import ObservationBatch
from nc_rted.features import CELL_FEATURE_DIM
from nc_rted.frozen_sample_consumer import FrozenSampleConsumer, FrozenSampleConsumerError
from nc_rted.task_inputs import FrozenTaskContext, TrainingTask
from nc_rted.train_worker import SampleMaterial


def _task():
    return TrainingTask("detection:ucf-crime:sample:0", "detection", "ucf-crime", "family", "sample", 2.0, 1, None, 0)


def _material():
    return SampleMaterial(
        FrozenTaskContext(torch.ones((1, 2, 3), dtype=torch.bfloat16),
                          [torch.ones((1, 3, 4, 4), dtype=torch.bfloat16)], [(4, 4)], 2.0, (2.0,), "time"),
        ObservationBatch(torch.ones((1, 1, 1, 4, CELL_FEATURE_DIM), dtype=torch.bfloat16),
                         torch.ones((1, 1, 1, 4), dtype=torch.bool),
                         torch.tensor([[[[0.0, 0.5, 1.0, 2.0]]]], dtype=torch.float32)),
        ("relation",), "question",
    )


def _publish(root, task, material, identity):
    records = root / "records"; records.mkdir(parents=True)
    key = hashlib.sha256(task.sample_id.encode()).hexdigest()
    tensor = records / f"{key}.pt"; binding = records / f"{key}.json"
    metadata = {"sample_id": task.sample_id, "task": task.task,
                "context": {"image_sizes": [list(size) for size in material.context.image_sizes],
                            "observed_seconds": material.context.observed_seconds,
                            "sampled_frame_times": list(material.context.sampled_frame_times),
                            "time_message": material.context.time_message},
                "relation_ids": list(material.relation_ids), "detection_question": material.detection_question,
                "detection_scoring": material.detection_scoring}
    tensors = {"visual_embeddings": material.context.visual_embeddings,
               "image_0": material.context.images[0], "features": material.observations.features,
               "valid": material.observations.valid, "observed_times": material.observations.observed_times}
    torch.save({"schema": "nc_rted_frozen_sample_record/v1", "identity": identity,
                "metadata": metadata, "tensors": tensors}, tensor)
    digest = hashlib.sha256(tensor.read_bytes()).hexdigest()
    binding.write_text(json.dumps({"schema": "nc_rted_frozen_sample_record/v1", "sample_id": task.sample_id,
                                   "identity": identity, "tensor_path": tensor.name, "tensor_sha256": digest},
                                  sort_keys=True, separators=(",", ":")) + "\n")


def test_prefetched_record_is_cpu_validated_then_transferred_at_use(tmp_path):
    task, material, identity = _task(), _material(), {"producer": "gpu1", "runtime_sha256": "a" * 64}
    _publish(tmp_path, task, material, identity)
    consumer = FrozenSampleConsumer({task.sample_id: task}, root=str(tmp_path), identity=identity, device="cpu",
                                    wait_seconds=1, poll_seconds=.01, prefetch_depth=1)
    try:
        consumer.prefetch([task.sample_id])
        restored = consumer(task.sample_id)
    finally:
        consumer.close()
    assert restored.context.visual_embeddings.device.type == "cpu"
    assert torch.equal(restored.context.visual_embeddings, material.context.visual_embeddings)
    assert torch.equal(restored.observations.features, material.observations.features)


def test_missing_record_has_a_bounded_failure_and_no_provider_fallback(tmp_path):
    task = _task()
    consumer = FrozenSampleConsumer({task.sample_id: task}, root=str(tmp_path), identity={"producer": "gpu1"}, device="cpu",
                                    wait_seconds=.02, poll_seconds=.005, prefetch_depth=1)
    started = time.monotonic()
    try:
        with pytest.raises(FrozenSampleConsumerError, match="timed out waiting"):
            consumer(task.sample_id)
    finally:
        consumer.close()
    assert time.monotonic() - started < .5


def test_record_identity_mismatch_is_not_consumed(tmp_path):
    task, material = _task(), _material()
    _publish(tmp_path, task, material, {"producer": "other"})
    consumer = FrozenSampleConsumer({task.sample_id: task}, root=str(tmp_path), identity={"producer": "gpu1"}, device="cpu",
                                    wait_seconds=1, poll_seconds=.01, prefetch_depth=1)
    try:
        with pytest.raises(FrozenSampleConsumerError, match="invalid"):
            consumer(task.sample_id)
    finally:
        consumer.close()
