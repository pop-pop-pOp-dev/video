import json
import torch
from collections import namedtuple
from contextlib import contextmanager
import runpy
import sys
import types
from pathlib import Path

from nc_rted.bridge import ObservationBatch
from nc_rted.features import CELL_FEATURE_DIM
import nc_rted.frozen_sample_cache as cache_module
from nc_rted.frozen_sample_cache import FrozenSampleCacheError, FrozenSampleStore
from nc_rted.task_inputs import FrozenTaskContext, TrainingTask
from nc_rted.train_worker import SampleMaterial


def material():
    context = FrozenTaskContext(
        torch.ones((1, 2, 3), dtype=torch.bfloat16), [torch.ones((1, 3, 4, 4), dtype=torch.bfloat16)],
        [(4, 4)], 2.0, (2.0,), "time",
    )
    observations = ObservationBatch(
        torch.ones((1, 1, 1, 4, CELL_FEATURE_DIM), dtype=torch.bfloat16),
        torch.ones((1, 1, 1, 4), dtype=torch.bool),
        torch.tensor([[[[0.0, 0.5, 1.0, 2.0]]]], dtype=torch.float32),
    )
    return SampleMaterial(context, observations, ("relation",), "question")


def task():
    return TrainingTask("detection:ucf-crime:example:0", "detection", "ucf-crime", "family", "example", 2.0, 1, None, 0)


def ample_disk(monkeypatch):
    usage = namedtuple("Usage", "total used free")(100 << 30, 1 << 30, 99 << 30)
    monkeypatch.setattr(cache_module.shutil, "disk_usage", lambda _: usage)


def test_store_creates_record_directory_under_reserve_aware_admission(tmp_path, monkeypatch):
    admitted = []

    def ensure(path, reserve):
        admitted.append((path, reserve))
        path.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(cache_module, "ensure_directory", ensure)
    store = FrozenSampleStore(tmp_path, identity={"catalog_sha256": "a" * 64})
    assert admitted == [(store.records, 20 << 30)]


def test_roundtrip_preserves_cpu_frozen_material(tmp_path, monkeypatch):
    ample_disk(monkeypatch)
    store = FrozenSampleStore(tmp_path, identity={"catalog_sha256": "a" * 64, "producer": "gpu-1"})
    binding = store.put(task(), material())
    restored = store.get(task())
    assert binding["path"].endswith(".json")
    assert restored is not None
    assert restored.context.visual_embeddings.device.type == "cpu"
    assert restored.context.visual_embeddings.dtype == torch.bfloat16
    assert torch.equal(restored.context.images[0], material().context.images[0])
    assert torch.equal(restored.observations.features, material().observations.features)


def test_rejects_binding_with_other_tensor_bytes(tmp_path, monkeypatch):
    ample_disk(monkeypatch)
    store = FrozenSampleStore(tmp_path, identity={"catalog_sha256": "a" * 64, "producer": "gpu-1"})
    store.put(task(), material())
    tensor, _ = store.paths(task().sample_id)
    tensor.write_bytes(tensor.read_bytes() + b"changed")
    try:
        store.get(task())
    except FrozenSampleCacheError:
        pass
    else:
        raise AssertionError("corrupt immutable tensor record was accepted")


def test_rejects_record_above_configured_cache_limit(tmp_path, monkeypatch):
    ample_disk(monkeypatch)
    store = FrozenSampleStore(tmp_path, identity={"catalog_sha256": "a" * 64}, max_bytes=1)
    try:
        store.put(task(), material())
    except FrozenSampleCacheError:
        pass
    else:
        raise AssertionError("record above the configured cache cap was accepted")


def test_capacity_admission_and_publication_hold_the_shared_allocation_lock(tmp_path, monkeypatch):
    ample_disk(monkeypatch)
    entered = False

    @contextmanager
    def locked(_):
        nonlocal entered
        entered = True
        try:
            yield
        finally:
            entered = False

    store = FrozenSampleStore(tmp_path, identity={"catalog_sha256": "a" * 64})
    monkeypatch.setattr(cache_module, "allocation_lock", locked)
    original = store._write_once

    def checked_write(path, content):
        assert entered
        return original(path, content)

    monkeypatch.setattr(store, "_write_once", checked_write)
    result = store.put(task(), material())
    assert result["bytes"] >= 8192


def test_cli_reuses_valid_record_without_calling_provider(tmp_path, monkeypatch):
    ample_disk(monkeypatch)
    identity = {"catalog_sha256": "a" * 64, "producer": "gpu-1"}
    store = FrozenSampleStore(tmp_path, identity=identity)
    store.put(task(), material())
    calls = []

    def provider(sample_id):
        calls.append(sample_id)
        raise AssertionError("provider must not run for a valid cached sample")

    runtime = types.SimpleNamespace(
        catalog=types.SimpleNamespace(identity=identity["catalog_sha256"], tasks={task().sample_id: task()}),
        producer_identity=identity, provider=provider,
        execution={"torch": "test", "device": "cuda:1"},
    )
    module = types.ModuleType("nc_rted.frozen_provider_runtime")
    module.assemble_frozen_provider = lambda *args: runtime
    monkeypatch.setitem(sys.modules, "nc_rted.frozen_provider_runtime", module)
    report = tmp_path / "execution.json"
    monkeypatch.setattr(sys, "argv", ["producer", "--runtime", "runtime.py", "--runtime-sha256", "b" * 64,
                                       "--device", "cuda:1", "--output-root", str(tmp_path),
                                       "--sample-id", task().sample_id, "--execution-report", str(report)])
    runpy.run_path(str(Path(__file__).parents[1] / "scripts" / "nc_rted_produce_frozen_samples.py"),
                   run_name="__main__")
    assert calls == []
    assert json.loads(report.read_text()) == {
        "schema": "nc_rted_frozen_sample_execution/v1", "identity": identity,
        "execution": runtime.execution, "sample_ids": [task().sample_id],
        "records": [store._record(task())],
    }
