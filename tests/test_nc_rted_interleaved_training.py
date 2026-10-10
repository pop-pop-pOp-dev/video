import copy
import importlib.util
import random
import weakref
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from nc_rted.interleaved_training import Bundle, SameSeedBundleWorker, clone_with_shared_frozen
from nc_rted.training import IncrementalTrainer, Recipe, seed_run
from nc_rted.recovery import CheckpointStore, capture_rng, restore_rng
from nc_rted.production_runtime import RuntimeManifest

ROOT=Path(__file__).resolve().parents[1]
diagnostic_spec=importlib.util.spec_from_file_location("interleaved_diagnostic",ROOT / "scripts" / "nc_rted_interleaved_gpu_diagnostic.py")
diagnostic=importlib.util.module_from_spec(diagnostic_spec); diagnostic_spec.loader.exec_module(diagnostic)


def _same(left, right):
    if isinstance(left, torch.Tensor): return torch.equal(left, right)
    if isinstance(left, dict): return left.keys() == right.keys() and all(_same(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)): return len(left) == len(right) and all(_same(a, b) for a, b in zip(left, right))
    return left == right


class Toy(nn.Module):
    def __init__(self):
        super().__init__(); self.frozen = nn.Linear(3, 3, bias=False); self.frozen.requires_grad_(False)
        self.slow = nn.Module(); self.slow.block = nn.Module(); self.slow.block.lora_A = nn.Linear(3, 3, bias=False)
        self.evidence = nn.Linear(3, 1, bias=False); self.drop = nn.Dropout(.25)
    def forward(self, value): return self.evidence(self.drop(self.slow.block.lora_A(self.frozen(value)))).sum()


def _make(seed):
    seed_run(seed); baseline = Toy(); frozen = [p for p in baseline.parameters() if not p.requires_grad]
    models = {g: clone_with_shared_frozen(baseline, frozen) for g in "AUSF"}
    trainers = {g: IncrementalTrainer(models[g], [str(i) for i in range(8)], seed, Recipe(updates=4, accumulation=2, warmup=1, save_interval=2)) for g in models}
    return models, trainers


def _loss(model, group, material):
    return model(material) * {"A": 1., "U": 1.2, "S": .8, "F": 1.4}[group]


def _identity(group, seed=17, *, run_id="interleaved-test", config_digest=None):
    digest = "0" * 64
    return {"run_id":run_id,"group":group,"seed":str(seed),"code_sha256":digest,
            "config_sha256":config_digest or digest,"data_sha256":digest,"teacher_sha256":digest,
            "inherited_weights_sha256":digest,"runtime_sha256":digest}


def test_same_seed_interleaving_matches_four_serial_runs_and_reuses_material():
    models, serial = _make(17); initial_rng = capture_rng()
    oracle = {}
    for group, trainer in serial.items():
        restore_rng(initial_rng); trainer.run(lambda sid, g=group: _loss(models[g], g, torch.tensor([[int(sid) + 1., 2., 3.]])))
        oracle[group] = (copy.deepcopy(models[group].state_dict()), copy.deepcopy(trainer.optimizer.state_dict()), trainer.scheduler.state_dict())
    models, trainers = _make(17); calls = []
    def provider(sid): calls.append(sid); return torch.tensor([[int(sid) + 1., 2., 3.]])
    bundles = {g: Bundle(g, trainers[g], lambda sid, material, g=g: _loss(models[g], g, material), rng=copy.deepcopy(initial_rng)) for g in "AUSF"}
    worker = SameSeedBundleWorker(bundles, provider); worker.run()
    assert len(calls) == 8
    for group in "AUSF":
        assert _same(models[group].state_dict(), oracle[group][0])
        assert _same(trainers[group].optimizer.state_dict(), oracle[group][1])
        assert _same(trainers[group].scheduler.state_dict(), oracle[group][2])
    frozen_ids = {id(p) for p in models["A"].parameters() if not p.requires_grad}
    assert all({id(p) for p in models[g].parameters() if not p.requires_grad} == frozen_ids for g in "USF")
    assert all(id(models["A"].evidence.weight) != id(models[g].evidence.weight) for g in "USF")


def test_checkpointed_bundle_replay_matches_standalone_and_preserves_independent_state(tmp_path):
    serial_models, serial_trainers = _make(17); initial_rng = capture_rng(); serial = {}
    for group in "AUSF":
        restore_rng(initial_rng)
        store=CheckpointStore(tmp_path / "serial" / group,_identity(group),min_free_bytes=0)
        serial_trainers[group].run(lambda sid, g=group: _loss(serial_models[g],g,torch.tensor([[int(sid)+1.,2.,3.]])),save=store.save,stop_after=2)
        restored_models, restored_trainers=_make(17)
        store.restore(restored_trainers[group])
        restored_trainers[group].run(lambda sid, g=group: _loss(restored_models[g],g,torch.tensor([[int(sid)+1.,2.,3.]])),save=store.save)
        serial[group]=(copy.deepcopy(restored_models[group].state_dict()),copy.deepcopy(restored_trainers[group].optimizer.state_dict()),copy.deepcopy(restored_trainers[group].scheduler.state_dict()),capture_rng())

    models, trainers=_make(17); calls=[]
    stores={g:CheckpointStore(tmp_path / "bundled" / g,_identity(g),min_free_bytes=0) for g in "AUSF"}
    bundles={g:Bundle(g,trainers[g],lambda sid, material, g=g: _loss(models[g],g,material),store=stores[g],rng=copy.deepcopy(initial_rng)) for g in "AUSF"}
    worker=SameSeedBundleWorker(bundles,lambda sid: calls.append(sid) or torch.tensor([[int(sid)+1.,2.,3.]]), bundle_checkpoint_root=tmp_path / "bundled" / "bundle")
    worker.run(stop_after=2)
    resumed_models, resumed_trainers=_make(17)
    resumed={g:Bundle(g,resumed_trainers[g],lambda sid, material, g=g: _loss(resumed_models[g],g,material),store=stores[g]) for g in "AUSF"}
    replay=SameSeedBundleWorker(resumed,lambda sid: calls.append(sid) or torch.tensor([[int(sid)+1.,2.,3.]]), bundle_checkpoint_root=tmp_path / "bundled" / "bundle")
    replay.restore(); replay.run()

    assert calls == trainers["A"].order
    for group in "AUSF":
        assert _same(resumed_models[group].state_dict(),serial[group][0])
        assert _same(resumed_trainers[group].optimizer.state_dict(),serial[group][1])
        assert _same(resumed_trainers[group].scheduler.state_dict(),serial[group][2])
        assert _same(resumed[group].rng,serial[group][3])
    for name in ("evidence.weight","slow.block.lora_A.weight"):
        assert len({id(dict(resumed_models[g].named_parameters())[name]) for g in "AUSF"}) == 4
        assert len({id(resumed_trainers[g].master_parameters[name]) for g in "AUSF"}) == 4
    assert len({id(resumed_trainers[g].optimizer.state_dict()["state"][0]["exp_avg"]) for g in "AUSF"}) == 4


def test_gpu_diagnostic_requires_four_bound_runtime_members(tmp_path):
    assert diagnostic.SCHEMA == "nc_rted_interleaved_gpu_harness/v1"
    path = tmp_path / "bundle.json"
    path.write_text('{"bundle_checkpoint_root":"/tmp/bundle","diagnostic_checkpoint_interval":1,"diagnostic_updates":1,"members":{},"schema":"nc_rted_interleaved_gpu_harness/v1","source_manifest":"/tmp/source.json","source_manifest_sha256":"' + "0" * 64 + '"}')
    with pytest.raises(ValueError, match="exactly A/U/S/F"):
        diagnostic._load_bundle_manifest(path, diagnostic._sha256(path))


def _checkpointed_worker(tmp_path, *, callback=None):
    models, trainers = _make(17)
    initial_rng = capture_rng()
    stores = {group: CheckpointStore(tmp_path / "groups" / group, _identity(group), min_free_bytes=0) for group in "AUSF"}
    bundles = {group: Bundle(group, trainers[group],
                             lambda sid, material, group=group: _loss(models[group], group, material),
                             store=stores[group], rng=copy.deepcopy(initial_rng)) for group in "AUSF"}
    return models, trainers, SameSeedBundleWorker(
        bundles, lambda sid: torch.tensor([[int(sid) + 1., 2., 3.]]),
        bundle_checkpoint_root=tmp_path / "bundle", after_checkpoint_publication=callback)


@pytest.mark.parametrize("boundary,failed_group", [(2, group) for group in "AUSF"] + [(4, group) for group in "AUSF"])
def test_bundle_recovery_replays_ahead_immutable_checkpoints_after_each_publication(tmp_path, boundary, failed_group):
    expected_models, _, expected = _checkpointed_worker(tmp_path / "expected")
    expected.run()
    expected_state = {group: copy.deepcopy(expected_models[group].state_dict()) for group in "AUSF"}

    def fail(group, update, final):
        if group == failed_group and update == boundary:
            raise RuntimeError("injected crash after individual publication")

    _, _, interrupted = _checkpointed_worker(tmp_path / "interrupted", callback=fail)
    if boundary == 4:
        interrupted.run(stop_after=2)
    with pytest.raises(RuntimeError, match="injected crash"):
        interrupted.run()

    resumed_models, resumed_trainers, resumed = _checkpointed_worker(tmp_path / "interrupted")
    resumed.restore()
    assert {trainer.completed_updates for trainer in resumed_trainers.values()} == ({0} if boundary == 2 else {2})
    resumed.run()
    for group in "AUSF":
        assert _same(resumed_models[group].state_dict(), expected_state[group])


def test_bundle_restore_prevalidates_all_groups_before_mutating_any_trainer(tmp_path):
    _, _, source = _checkpointed_worker(tmp_path)
    source.run(stop_after=2)
    state_path = tmp_path / "groups" / "U" / "update_000002" / "state.pt"
    state_path.write_bytes(b"bad checkpoint")
    models, trainers, resumed = _checkpointed_worker(tmp_path)
    original = {group: copy.deepcopy(models[group].state_dict()) for group in "AUSF"}
    with pytest.raises(Exception):
        resumed.restore()
    assert all(trainer.completed_updates == 0 for trainer in trainers.values())
    assert all(_same(models[group].state_dict(), original[group]) for group in "AUSF")


def test_bundle_rejects_shared_state_grad_connected_material_and_material_mutation(tmp_path):
    models, trainers = _make(17)
    same = {group: Bundle(group, trainers["A"], lambda *_: _loss(models["A"], "A", torch.ones(1, 3))) for group in "AUSF"}
    with pytest.raises(ValueError, match="independent trainer"):
        SameSeedBundleWorker(same, lambda _: torch.ones(1, 3))

    bundles = {group: Bundle(group, trainers[group], lambda sid, material, group=group: _loss(models[group], group, material)) for group in "AUSF"}
    unsafe = SameSeedBundleWorker(bundles, lambda _: torch.ones(1, 3, requires_grad=True))
    with pytest.raises(ValueError, match="detached"):
        unsafe.run(stop_after=1)

    def mutating_loss(group):
        def loss(_, material):
            if group == "A": material.add_(1)
            return _loss(models[group], group, material)
        return loss
    mutating = {group: Bundle(group, trainers[group], mutating_loss(group)) for group in "AUSF"}
    worker = SameSeedBundleWorker(mutating, lambda _: torch.ones(1, 3))
    with pytest.raises(RuntimeError, match="mutated"):
        worker.run(stop_after=1)


def test_bundle_commit_binds_distinct_member_configurations_and_run_ids(tmp_path):
    models, trainers = _make(17)
    stores = {group: CheckpointStore(tmp_path / group, _identity(group, run_id=f"diagnostic:{group}",
                                                               config_digest=(f"{index:x}" * 64)), min_free_bytes=0)
              for index, group in enumerate("AUSF")}
    bundles = {group: Bundle(group, trainers[group], lambda sid, material, group=group: _loss(models[group], group, material),
                             store=stores[group]) for group in "AUSF"}
    worker = SameSeedBundleWorker(bundles, lambda sid: torch.tensor([[int(sid) + 1., 2., 3.]]),
                                  bundle_checkpoint_root=tmp_path / "bundle")
    worker.run(stop_after=2)
    manifest = next((tmp_path / "bundle" / "commits").glob("*.json"))
    document = __import__("json").loads(manifest.read_text())
    assert document["members"] == {group: stores[group].identity for group in "AUSF"}
    assert document["scientific_identity"]["data_sha256"] == "0" * 64


def test_prepared_guard_rejects_tensor_replacement_metadata_mutation_and_new_graph():
    models, trainers = _make(17)
    def bundles(loss):
        return {group: Bundle(group, trainers[group], loss(group)) for group in "AUSF"}
    def replacement(group):
        def loss(_, material):
            if group == "A": material["tensor"] = torch.ones(1, 3)
            return _loss(models[group], group, material["tensor"])
        return loss
    with pytest.raises(RuntimeError, match="mutated"):
        SameSeedBundleWorker(bundles(replacement), lambda _: {"tensor": torch.ones(1, 3), "meta": "fixed"}).run(stop_after=1)
    models, trainers = _make(17)
    def metadata(group):
        def loss(_, material):
            if group == "A": material["meta"] = "changed"
            return _loss(models[group], group, material["tensor"])
        return loss
    with pytest.raises(RuntimeError, match="mutated"):
        SameSeedBundleWorker(bundles(metadata), lambda _: {"tensor": torch.ones(1, 3), "meta": "fixed"}).run(stop_after=1)
    models, trainers = _make(17)
    def graph(group):
        def loss(_, material):
            if group == "A": material["new"] = torch.ones(1, 3, requires_grad=True) * 2
            return _loss(models[group], group, material["tensor"])
        return loss
    with pytest.raises(ValueError, match="detached"):
        SameSeedBundleWorker(bundles(graph), lambda _: {"tensor": torch.ones(1, 3), "meta": "fixed"}).run(stop_after=1)


def test_diagnostic_digests_content_and_complete_training_state():
    left = {"tensor": torch.tensor([1.]), "meta": {"source": "A"}}
    right = {"tensor": torch.tensor([2.]), "meta": {"source": "A"}}
    assert diagnostic._material_digest(left) != diagnostic._material_digest(right)
    assert diagnostic._material_digest([[1], 2]) != diagnostic._material_digest([[1, 2]])
    models, trainers = _make(17)
    trainer = trainers["A"]
    trainer.step(lambda sid: _loss(models["A"], "A", torch.tensor([[int(sid) + 1., 2., 3.]])))
    # AdamW has scalar step tensors only after a real optimizer update.
    rng = capture_rng(); initial = diagnostic._trainer_state_digest(models["A"], trainer, rng)
    optimizer = trainer.optimizer.state_dict(); first = next(iter(optimizer["state"].values()))
    first["exp_avg"].add_(.5); trainer.optimizer.load_state_dict(optimizer)
    assert diagnostic._trainer_state_digest(models["A"], trainer, rng) != initial
    changed_rng = copy.deepcopy(rng); changed_rng["torch"][0] ^= 1
    assert diagnostic._trainer_state_digest(models["A"], trainer, changed_rng) != diagnostic._trainer_state_digest(models["A"], trainer, rng)


def test_diagnostic_evidence_publication_is_durable_and_non_overwriting(tmp_path):
    target = tmp_path / "result.json"
    diagnostic._write_json(target, {"status": "ok"})
    diagnostic._write_json(target, {"status": "ok"})
    with pytest.raises(FileExistsError, match="already differs"):
        diagnostic._write_json(target, {"status": "different"})
    assert __import__("json").loads(target.read_text()) == {"status": "ok"}


def test_diagnostic_evidence_retry_syncs_existing_publication(tmp_path, monkeypatch):
    target = tmp_path / "result.json"; original = diagnostic.os.fsync; calls = [0]
    def fail_directory_once(descriptor):
        calls[0] += 1
        if calls[0] == 2: raise OSError("simulated crash after link")
        return original(descriptor)
    monkeypatch.setattr(diagnostic.os, "fsync", fail_directory_once)
    with pytest.raises(OSError, match="after link"):
        diagnostic._write_json(target, {"status": "ok"})
    assert target.is_file()
    monkeypatch.setattr(diagnostic.os, "fsync", original)
    diagnostic._write_json(target, {"status": "ok"})


def test_diagnostic_source_manifest_rejects_covered_source_change(tmp_path):
    root = tmp_path / "source"; root.mkdir()
    files = {}
    required = {"scripts/nc_rted_interleaved_gpu_diagnostic.py"}
    required.update(f"src/nc_rted/{path.name}" for path in (ROOT / "src" / "nc_rted").glob("*.py"))
    for relative in required:
        target = root / relative; target.parent.mkdir(parents=True, exist_ok=True); target.write_text(relative)
        files[relative] = diagnostic._sha256(target)
    document = {"schema": "nc_rted_interleaved_source_manifest/v1", "files": files,
                "code_sha256": diagnostic._source_identity(files)}
    manifest = tmp_path / "source.json"; manifest.write_text(__import__("json").dumps(document))
    harness = {"source_manifest": str(manifest), "source_manifest_sha256": diagnostic._sha256(manifest)}
    members = {group: type("Member", (), {"document": {"hashes": {"code_sha256": document["code_sha256"]}}})() for group in "AUSF"}
    diagnostic._verify_source_manifest(harness, members, root=root)
    (root / "src/nc_rted/bridge.py").write_text("changed")
    with pytest.raises(ValueError, match="source differs"):
        diagnostic._verify_source_manifest(harness, members, root=root)
    files.pop("src/nc_rted/bridge.py")
    document["files"] = files; document["code_sha256"] = diagnostic._source_identity(files)
    manifest.write_text(__import__("json").dumps(document)); harness["source_manifest_sha256"] = diagnostic._sha256(manifest)
    with pytest.raises(ValueError, match="coverage"):
        diagnostic._verify_source_manifest(harness, members, root=root)


def test_serial_factory_namespace_isolated_from_fault_checkpoint_root(tmp_path):
    document = {"run": {"checkpoint_root": str(tmp_path / "fault" / "A")}}
    manifest = RuntimeManifest(tmp_path / "member.json", "a" * 64, document)
    assembly = diagnostic._assembly_manifest(manifest, tmp_path / "serial")
    CheckpointStore(assembly.run["checkpoint_root"], _identity("A"), min_free_bytes=0)
    assert not (tmp_path / "fault" / "A").exists()
    assert (tmp_path / "serial" / "factory-A" / ".store.lock").exists()


def test_preparation_is_bounded_to_one_accumulation_window():
    models, trainers = _make(17)
    live, maximum = weakref.WeakSet(), [0]
    class Prepared:
        def __init__(self, value): self.value = torch.tensor([[value, 2., 3.]])
    def provider(sample_id):
        item = Prepared(float(int(sample_id) + 1)); live.add(item); maximum[0] = max(maximum[0], len(live)); return item
    bundles = {group: Bundle(group, trainers[group],
                             lambda _, material, group=group: _loss(models[group], group, material.value)) for group in "AUSF"}
    SameSeedBundleWorker(bundles, provider).run()
    assert maximum[0] <= trainers["A"].recipe.accumulation
