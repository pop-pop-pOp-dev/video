from __future__ import annotations

import hashlib
import ast
import importlib.util
import json
from pathlib import Path
import subprocess
from datetime import datetime, timezone
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nc_rted.evaluation import (EvaluationError, _caption_references, _read_task_store,
                                _task_execution_binding, _validate_vau_payload, _caption_metrics)
from nc_rted.prediction_inputs import canonical_json
from nc_rted.prediction_store import PredictionStore
import nc_rted.evaluation as evaluation


def _write_store(root: Path, *, binding: str | None = None, identities=("vad:ucf:alpha",)) -> tuple[SimpleNamespace, SimpleNamespace]:
    artifact = SimpleNamespace(task_id="A:seed17", manifest_sha256="a" * 64,
                               checkpoint_manifest_sha256="b" * 64, checkpoint_state_sha256="c" * 64)
    plan = SimpleNamespace(run_id="run", manifest_sha256="d" * 64, matrix_id="e" * 64,
                           protocol={"hivau": {"max_new_tokens": 5}}, bindings={"bound": "f" * 64})
    binding = binding or _task_execution_binding(matrix_id=plan.matrix_id, artifact=artifact)
    entries = {}
    for identity in identities:
        payload = {"total_frames": 2, "causal_smoothed_scores": [.1, .2], "queries": [{"query_index": 0, "frame_indices": [0], "fast_score": .1, "final_score": .1, "slow_score": None, "fused_score": None}]}
        record = {"schema": "nc_rted_prediction_record/v1", "run_id": plan.run_id, "manifest_sha256": plan.manifest_sha256,
                  "model_task": "A:seed17", "model_binding_sha256": artifact.manifest_sha256, "identity": identity,
                  "attempt": 1, "status": "success", "retryable": False, "created_at": "2026-10-10T00:00:00Z",
                  "provenance": {"matrix_id": plan.matrix_id, "task_execution_binding_sha256": binding, "model_group": "A", "model_seed": 17,
                                 "model_task": artifact.task_id, "model_manifest_sha256": artifact.manifest_sha256,
                                 "checkpoint_manifest_sha256": artifact.checkpoint_manifest_sha256, "checkpoint_state_sha256": artifact.checkpoint_state_sha256,
                                 "protocol": plan.protocol, "bindings": plan.bindings}, "payload": payload}
        record["record_sha256"] = hashlib.sha256(canonical_json(record)).hexdigest()
        key = hashlib.sha256(identity.encode()).hexdigest()
        path = root / "A:seed17" / "records" / f"{key}.attempt1.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(canonical_json(record) + b"\n")
        entries[key] = {"path": f"records/{path.name}", "record_sha256": record["record_sha256"], "status": "success", "attempt": 1}
    index = {"schema": "nc_rted_prediction_index/v2", "run_id": plan.run_id, "manifest_sha256": plan.manifest_sha256,
             "model_task": "A:seed17", "model_binding_sha256": artifact.manifest_sha256, "matrix_id": plan.matrix_id,
             "task_execution_binding_sha256": binding, "records": entries}
    (root / "A:seed17" / "index.json").write_bytes(canonical_json(index) + b"\n")
    return plan, artifact


def test_v2_store_requires_exact_task_binding_and_complete_identities(tmp_path):
    plan, artifact = _write_store(tmp_path)
    frozen = _read_task_store(tmp_path, plan=plan, group="A", seed=17, artifact=artifact,
                              expected_identities={"vad:ucf:alpha"})
    assert frozen.records["vad:ucf:alpha"]["payload"]["causal_smoothed_scores"] == [.1, .2]
    with pytest.raises(EvaluationError, match="denominator"):
        _read_task_store(tmp_path, plan=plan, group="A", seed=17, artifact=artifact,
                         expected_identities={"vad:ucf:alpha", "vau:missing"})
    stale_plan, stale_artifact = _write_store(tmp_path / "stale", binding="0" * 64)
    with pytest.raises(EvaluationError, match="matrix binding"):
        _read_task_store(tmp_path / "stale", plan=stale_plan, group="A", seed=17, artifact=stale_artifact,
                         expected_identities={"vad:ucf:alpha"})


def test_actual_prediction_store_retry_history_is_accepted(tmp_path):
    artifact = SimpleNamespace(task_id="A:seed17", manifest_sha256="a" * 64,
                               checkpoint_manifest_sha256="b" * 64, checkpoint_state_sha256="c" * 64)
    plan = SimpleNamespace(run_id="run", manifest_sha256="d" * 64, matrix_id="e" * 64,
                           protocol={"hivau": {"max_new_tokens": 5}}, bindings={"bound": "f" * 64})
    binding = _task_execution_binding(matrix_id=plan.matrix_id, artifact=artifact)
    store = PredictionStore(tmp_path / "A:seed17", run_id=plan.run_id, manifest_sha256=plan.manifest_sha256,
                            model_task=artifact.task_id, model_binding_sha256=artifact.manifest_sha256,
                            matrix_id=plan.matrix_id, task_execution_binding_sha256=binding)
    provenance = {"matrix_id": plan.matrix_id, "task_execution_binding_sha256": binding, "model_group": "A", "model_seed": 17,
                  "model_task": artifact.task_id, "model_manifest_sha256": artifact.manifest_sha256,
                  "checkpoint_manifest_sha256": artifact.checkpoint_manifest_sha256, "checkpoint_state_sha256": artifact.checkpoint_state_sha256,
                  "protocol": plan.protocol, "bindings": plan.bindings}
    identity = "vad:ucf:alpha"
    store.publish(identity=identity, attempt=1, status="failure", provenance=provenance, error=RuntimeError("retry"), retryable=True,
                  retry_not_before=datetime.now(timezone.utc))
    payload = {"total_frames": 2, "causal_smoothed_scores": [.1, .2], "queries": [{"query_index": 0, "frame_indices": [0], "fast_score": .1, "final_score": .1, "slow_score": None, "fused_score": None}]}
    store.publish(identity=identity, attempt=2, status="success", provenance=provenance, payload=payload)
    frozen = _read_task_store(tmp_path, plan=plan, group="A", seed=17, artifact=artifact, expected_identities={identity})
    assert frozen.records[identity]["attempt"] == 2


def test_caption_reference_join_requires_exact_original_instruction_set(tmp_path):
    path = tmp_path / "references.jsonl"
    path.write_text(json.dumps({"id": "one", "answer": "reference", "type": "clip"}) + "\n", encoding="utf-8")
    assert _caption_references(path, {"one"})["one"]["answers"] == ["reference"]
    with pytest.raises(EvaluationError, match="denominator"):
        _caption_references(path, {"one", "two"})


def test_caption_metrics_preserve_multiple_references():
    seen = {}
    class Tokenizer:
        def tokenize(self, value): return value
    class Score:
        def __init__(self, value): self.value = value
        def compute_score(self, refs, hyps):
            seen["refs"] = refs
            return self.value, []
    official = SimpleNamespace(PTBTokenizer=Tokenizer, Bleu=lambda _: Score([.1, .2, .3, .4]),
                               Rouge=lambda: Score(.5), Cider=lambda: Score(.6), Meteor=lambda: Score(.7))
    metrics = _caption_metrics(official, [{"pred": "prediction", "answers": ["first", "second"], "type": "clip"}])
    assert [item["caption"] for item in seen["refs"][0]] == ["first", "second"]
    assert metrics["by_granularity"]["clip"]["bleu_aggregate"] == pytest.approx(1.0)


def test_caption_bleu_uses_mean_of_per_question_vectors_not_corpus_value():
    class Tokenizer:
        def tokenize(self, value): return value
    class Bleu:
        calls = 0
        def compute_score(self, refs, hyps):
            self.__class__.calls += 1
            return ([.2] * 4 if self.calls == 1 else [.8] * 4), []
    class Constant:
        def compute_score(self, refs, hyps): return .5, []
    official = SimpleNamespace(PTBTokenizer=Tokenizer, Bleu=lambda _: Bleu(), Rouge=Constant, Cider=Constant, Meteor=Constant)
    metrics = _caption_metrics(official, [{"pred": "p1", "answers": ["a"], "type": "clip"}, {"pred": "p2", "answers": ["b"], "type": "clip"}])
    assert metrics["by_granularity"]["clip"]["bleu_1_to_4"] == pytest.approx([.5] * 4)


def test_generation_cap_is_enforced_before_any_reference_join():
    with pytest.raises(EvaluationError, match="bounded"):
        _validate_vau_payload({"text": "too long", "token_ids": [1, 2, 3]}, max_new_tokens=2)


def test_inherited_detection_metrics_match_all_ones_sklearn_semantics():
    root = Path("/root/autodl-tmp/lookaway-wm/external/ReactVAU-paper/eval_utils/vad/detect_utils.py")
    spec = importlib.util.spec_from_file_location("code04_detect_utils", root)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    metrics = module.compute_metrics([1.0, 1.0], [0, 1])
    assert metrics["roc_auc"] == pytest.approx(.5)
    assert metrics["ap"] == pytest.approx(.5)


def test_inherited_caption_metric_api_is_present_without_importing_optional_package():
    source = Path("/root/autodl-tmp/lookaway-wm/external/ReactVAU-paper/eval_utils/hivau/hivau_utils.py")
    functions = {node.name for node in ast.parse(source.read_text(encoding="utf-8")).body if isinstance(node, ast.FunctionDef)}
    assert {"hivau_BLEU", "hivau_ROUGE", "hivau_CIDEr", "hivau_METEOR"} <= functions


def test_cli_rejects_malformed_matrix_before_evaluator_inputs_are_opened(tmp_path):
    plan = tmp_path / "plan.json"
    plan.write_text("{}", encoding="utf-8")
    digest = hashlib.sha256(plan.read_bytes()).hexdigest()
    script = Path(__file__).resolve().parents[1] / "scripts" / "nc_rted_evaluate.py"
    result = subprocess.run([sys.executable, str(script), "--plan", str(plan), "--plan-sha256", digest,
                             "--stores-root", str(tmp_path / "stores"), "--reactvau-root", str(tmp_path / "not-opened"),
                             "--ucf-annotation", str(tmp_path / "not-opened-ucf"), "--xd-annotation", str(tmp_path / "not-opened-xd"),
                             "--hivau-references", str(tmp_path / "not-opened-refs"), "--output-dir", str(tmp_path / "out"),
                             "--bootstrap-seed", "1"], text=True, capture_output=True, check=False)
    assert result.returncode == 2
    assert "prediction plan is not an admitted full matrix" in result.stderr
    assert not (tmp_path / "out").exists()


def test_full_synthetic_matrix_freezes_old_plan_records_then_evaluates(tmp_path, monkeypatch):
    old_plan, current_plan = "1" * 64, "2" * 64
    vad = tuple(SimpleNamespace(dataset=dataset, media_id=media_id, identity=f"vad:{dataset}:{media_id}")
                for dataset, media_id in (("ucf", "u0"), ("ucf", "u1"), ("xd", "x0"), ("xd", "x1")))
    vau = tuple(SimpleNamespace(instruction_id=f"q{index}", identity=f"vau:q{index}") for index in range(3))
    tasks = tuple(SimpleNamespace(group=group, seed=seed, task_id=(group if seed is None else f"{group}:seed{seed}"), model_manifest=f"/{group}{seed}", model_manifest_sha256=(group[0] * 64))
                  for group, seed in (("R0", None), *((group, seed) for group in ("A", "U", "S", "F") for seed in (17, 42, 2026))))
    plan = SimpleNamespace(run_id="synthetic", manifest_sha256=current_plan, matrix_id="3" * 64, vad=vad, vau=vau, models=tasks,
                           protocol={"hivau": {"max_new_tokens": 4}}, bindings={"bound": "4" * 64})
    artifacts = {task.task_id: SimpleNamespace(task_id=task.task_id, manifest_sha256=task.model_manifest_sha256,
                 checkpoint_manifest_sha256=None if task.group == "R0" else "5" * 64, checkpoint_state_sha256=None if task.group == "R0" else "6" * 64,
                 final_checkpoint_attestation_sha256=None if task.group == "R0" else "7" * 64) for task in tasks}
    for task in tasks:
        artifact = artifacts[task.task_id]; binding = _task_execution_binding(matrix_id=plan.matrix_id, artifact=artifact)
        store = PredictionStore(tmp_path / "stores" / task.task_id, run_id=plan.run_id, manifest_sha256=current_plan,
                                model_task=task.task_id, model_binding_sha256=artifact.manifest_sha256,
                                matrix_id=plan.matrix_id, task_execution_binding_sha256=binding)
        provenance = {"matrix_id": plan.matrix_id, "task_execution_binding_sha256": binding, "model_group": task.group, "model_seed": task.seed,
                      "model_task": task.task_id, "model_manifest_sha256": artifact.manifest_sha256,
                      "checkpoint_manifest_sha256": artifact.checkpoint_manifest_sha256, "checkpoint_state_sha256": artifact.checkpoint_state_sha256,
                      "protocol": plan.protocol, "bindings": plan.bindings}
        for request in vad:
            score = .9 if request.media_id.endswith("1") else .1
            payload = {"total_frames": 1, "causal_smoothed_scores": [score], "queries": [{"query_index": 0, "frame_indices": [0], "fast_score": score, "final_score": score, "slow_score": None, "fused_score": None}]}
            store.publish(identity=request.identity, attempt=1, status="success", provenance=provenance, payload=payload)
        for request in vau:
            store.publish(identity=request.identity, attempt=1, status="success", provenance=provenance, payload={"text": "prediction", "token_ids": [1]})
        # Simulate an admitted prior plan revision: v2 index remains matrix-bound while records retain their plan hash.
        for record_path in (tmp_path / "stores" / task.task_id / "records").glob("*.json"):
            record = json.loads(record_path.read_text()); record["manifest_sha256"] = old_plan
            record["record_sha256"] = hashlib.sha256(canonical_json({k: v for k, v in record.items() if k != "record_sha256"})).hexdigest()
            record_path.write_bytes(canonical_json(record) + b"\n")
        index_path = tmp_path / "stores" / task.task_id / "index.json"; index = json.loads(index_path.read_text())
        for entry in index["records"].values():
            record = json.loads((index_path.parent / entry["path"]).read_text()); entry["record_sha256"] = record["record_sha256"]
        index_path.write_bytes(canonical_json(index) + b"\n")
    manifest = tmp_path / "plan.json"; manifest.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(evaluation, "_sha256", lambda _: current_plan)
    monkeypatch.setattr(evaluation, "load_prediction_plan", lambda *_args, **_kwargs: plan)
    monkeypatch.setattr(evaluation, "load_model_artifact", lambda _path, *, task, **_kwargs: artifacts[task.task_id])
    frozen = evaluation.freeze_prediction_matrix(plan_path=manifest, plan_sha256=current_plan, stores_root=tmp_path / "stores")
    assert frozen.tasks["A:seed17"].records["vad:ucf:u0"]["manifest_sha256"] == old_plan
    class Detect:
        @staticmethod
        def load_anno_txt(path, dataset, **kwargs): return {"u0": {"label": 0}, "u1": {"label": 1}} if dataset == "ucf-crime" else {"x0": {"label": 0}, "x1": {"label": 1}}
        @staticmethod
        def make_gt_labels_from_anno(anno, total): return [anno["label"]] * total
        @staticmethod
        def compute_metrics(scores, labels): return {"roc_auc": .5, "ap": .5}
    class Tokenizer:
        def tokenize(self, value): return value
    class Score:
        def __init__(self, value): self.value=value
        def compute_score(self, refs, hyps): return self.value, []
    fake = SimpleNamespace(PTBTokenizer=Tokenizer, Bleu=lambda _: Score([.1, .2, .3, .4]), Rouge=lambda: Score(.5), Cider=lambda: Score(.6), Meteor=lambda: Score(.7))
    root = tmp_path / "react"; (root / "eval_utils/vad").mkdir(parents=True); (root / "eval_utils/hivau").mkdir(parents=True)
    for path in (root / "eval_utils/vad/detect_utils.py", root / "eval_utils/hivau/hivau_utils.py", tmp_path / "ucf", tmp_path / "xd") : path.write_text("x")
    refs = tmp_path / "refs"; refs.write_text("\n".join(json.dumps({"id": f"q{i}", "answer": ["a", "b"], "type": kind}) for i, kind in enumerate(("clip", "event", "video"))) + "\n")
    monkeypatch.setattr(evaluation, "_official_modules", lambda _: (Detect, fake))
    result = evaluation.evaluate_frozen_matrix(frozen=frozen, reactvau_root=root, ucf_annotation=tmp_path / "ucf", xd_annotation=tmp_path / "xd", hivau_references=refs, output_dir=tmp_path / "out", bootstrap_seed=1, bootstrap_repeats=10)
    assert result["tasks"]["R0"]["completeness"]["successful_records"] == 7
    assert (tmp_path / "out" / "official-evaluation.json").exists()
