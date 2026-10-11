import shutil

import pytest
import torch

from nc_rted.bridge import ObservationBatch
from nc_rted.features import CELL_FEATURE_DIM
import nc_rted.frozen_sample_cache as cache_module
import nc_rted.storage_lock as storage_lock
from nc_rted.frozen_sample_cache import FrozenSampleStore
from nc_rted.frozen_sample_transport import (
    FrozenSampleTransportError,
    cache_status,
    evict_local_after_ack,
    install_ack,
    receive_staged,
    task_document,
)
from nc_rted.task_inputs import FrozenTaskContext, TrainingTask
from nc_rted.train_worker import SampleMaterial


@pytest.fixture(autouse=True)
def _reserve_aware_test_disk(monkeypatch):
    usage = type("Usage", (), {"free": 99 << 30})()
    monkeypatch.setattr(cache_module.shutil, "disk_usage", lambda _: usage)
    monkeypatch.setattr(storage_lock.shutil, "disk_usage", lambda _: usage)


def _task():
    return TrainingTask("detection:ucf-crime:transport:0", "detection", "ucf-crime", "family", "transport",
                        2.0, 1, None, 0)


def _material():
    return SampleMaterial(
        FrozenTaskContext(torch.ones((1, 2, 3), dtype=torch.bfloat16),
                          [torch.ones((1, 3, 4, 4), dtype=torch.bfloat16)], [(4, 4)], 2.0, (2.0,), "time"),
        ObservationBatch(torch.ones((1, 1, 1, 4, CELL_FEATURE_DIM), dtype=torch.bfloat16),
                         torch.ones((1, 1, 1, 4), dtype=torch.bool),
                         torch.tensor([[[[0.0, 0.5, 1.0, 2.0]]]], dtype=torch.float32)),
        ("relation",), "question",
    )


def _stage(local, remote, task):
    tensor, binding = local.paths(task.sample_id)
    incoming = remote.root / ".incoming"
    incoming.mkdir(parents=True, exist_ok=True)
    staged_tensor, staged_binding = incoming / "record.pt.part", incoming / "record.json.part"
    shutil.copy2(tensor, staged_tensor)
    shutil.copy2(binding, staged_binding)
    return staged_tensor, staged_binding


def test_receiver_rejects_incomplete_or_corrupt_staging_before_publication(tmp_path):
    identity, task = {"catalog_sha256": "a" * 64, "producer": "gpu1"}, _task()
    local = FrozenSampleStore(tmp_path / "local", identity=identity)
    local.put(task, _material())
    remote_root = tmp_path / "remote"
    remote = FrozenSampleStore(remote_root, identity=identity)
    staged_tensor, staged_binding = _stage(local, remote, task)
    staged_tensor.write_bytes(staged_tensor.read_bytes() + b"corrupt")
    with pytest.raises(FrozenSampleTransportError, match="does not match"):
        receive_staged(root=remote_root, staged_tensor=staged_tensor, staged_binding=staged_binding, task=task)
    tensor, binding = remote.paths(task.sample_id)
    assert not tensor.exists()
    assert not binding.exists()
    staged_binding.unlink()
    with pytest.raises(FrozenSampleTransportError, match="incomplete"):
        receive_staged(root=remote_root, staged_tensor=staged_tensor, staged_binding=staged_binding, task=task)


def test_local_eviction_requires_a_valid_consumer_acknowledgement(tmp_path):
    identity, task = {"catalog_sha256": "a" * 64, "producer": "gpu1"}, _task()
    local = FrozenSampleStore(tmp_path / "local", identity=identity)
    local.put(task, _material())
    with pytest.raises(FrozenSampleTransportError, match="absent"):
        evict_local_after_ack(root=local.root, task=task, identity=identity)
    assert local.get(task) is not None
    remote_root = tmp_path / "remote"
    remote = FrozenSampleStore(remote_root, identity=identity)
    staged_tensor, staged_binding = _stage(local, remote, task)
    receipt = receive_staged(root=remote_root, staged_tensor=staged_tensor, staged_binding=staged_binding, task=task)
    copied_ack = tmp_path / "copied-ack.json"
    shutil.copy2(receipt["ack_path"], copied_ack)
    installed = install_ack(root=local.root, task=task, identity=identity, ack=copied_ack)
    evict_local_after_ack(root=local.root, task=task, identity=identity, ack=installed)
    assert local.get(task) is None


def test_status_enforces_cache_and_free_space_reservations(tmp_path, monkeypatch):
    usage = type("Usage", (), {"free": 20 << 30})()
    monkeypatch.setattr("nc_rted.frozen_sample_transport.shutil.disk_usage", lambda _: usage)
    status = cache_status(tmp_path, max_bytes=1 << 30, min_free_bytes=20 << 30, next_bytes=1)
    assert status["admitted"] is False
    status = cache_status(tmp_path, max_bytes=1, min_free_bytes=20 << 30, next_bytes=2)
    assert status["admitted"] is False
    assert task_document(_task())["sample_id"] == _task().sample_id
