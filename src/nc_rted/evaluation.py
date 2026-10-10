"""Read-only official evaluation of a frozen NC-RTED prediction matrix.

The freeze boundary is intentional: all model/checkpoint and prediction-store
validation completes before this module opens an official annotation/reference
file or imports the inherited metric implementations.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import importlib.util
from importlib import metadata
import json
import math
import statistics
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Any, Mapping, Sequence

from .prediction_inputs import (FORMAL_SEEDS, GROUPS, PredictionInputError,
                                PredictionPlan, canonical_json, load_model_artifact,
                                load_prediction_plan)
from .statistics import Prediction, evaluate_six_primary


class EvaluationError(RuntimeError):
    pass


def _probability(value: object, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
        raise EvaluationError(f"{name} must be a finite probability")


def _validate_vad_payload(payload: object) -> None:
    if not isinstance(payload, dict) or type(payload.get("total_frames")) is not int or payload["total_frames"] < 1:
        raise EvaluationError("VAD payload omits original-frame geometry")
    total, scores, queries = payload["total_frames"], payload.get("causal_smoothed_scores"), payload.get("queries")
    if not isinstance(scores, list) or len(scores) != total or not isinstance(queries, list) or not queries:
        raise EvaluationError("VAD payload does not cover all original frames")
    for score in scores:
        _probability(score, name="VAD causal score")
    for index, query in enumerate(queries):
        if not isinstance(query, dict) or query.get("query_index") != index or not isinstance(query.get("frame_indices"), list):
            raise EvaluationError("VAD query geometry differs")
        for frame in query["frame_indices"]:
            if type(frame) is not int or frame < 0 or frame >= total:
                raise EvaluationError("VAD query frame differs")
        for name in ("fast_score", "final_score"):
            _probability(query.get(name), name=f"VAD {name}")
        for name in ("slow_score", "fused_score"):
            if query.get(name) is not None:
                _probability(query[name], name=f"VAD {name}")


def _validate_vau_payload(payload: object, *, max_new_tokens: int) -> None:
    if (not isinstance(payload, dict) or not isinstance(payload.get("text"), str) or not isinstance(payload.get("token_ids"), list) or
            any(type(token) is not int for token in payload["token_ids"]) or len(payload["token_ids"]) > max_new_tokens):
        raise EvaluationError("VAU payload omits bounded generated text")


@dataclass(frozen=True)
class FrozenTask:
    task_id: str
    group: str
    seed: int | None
    records: Mapping[str, Mapping[str, Any]]
    record_sha256: Mapping[str, str]


@dataclass(frozen=True)
class FrozenMatrix:
    plan: PredictionPlan
    tasks: Mapping[str, FrozenTask]


def _file_binding(path: str | Path) -> dict[str, str]:
    value = Path(path).resolve()
    return {"path": str(value), "sha256": _sha256(value)}


def _package_versions() -> dict[str, str | None]:
    result = {}
    for name in ("scikit-learn", "pycocoevalcap", "nltk", "loguru"):
        try: result[name] = metadata.version(name)
        except metadata.PackageNotFoundError: result[name] = None
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise EvaluationError(f"{name} is unreadable") from error
    if not isinstance(value, dict):
        raise EvaluationError(f"{name} must be a JSON object")
    return value


def _task_identity(group: str, seed: int | None) -> str:
    return group if seed is None else f"{group}:seed{seed}"


def _task_execution_binding(*, matrix_id: str, artifact: Any) -> str:
    """Match the v2 store binding introduced after the v35 source snapshot."""
    value = {"matrix_id": matrix_id, "task": artifact.task_id, "model_manifest_sha256": artifact.manifest_sha256,
             "checkpoint_manifest_sha256": artifact.checkpoint_manifest_sha256,
             "checkpoint_state_sha256": artifact.checkpoint_state_sha256}
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _expected_tasks() -> tuple[tuple[str, int | None], ...]:
    return (("R0", None),) + tuple((group, seed) for group in ("A", "U", "S", "F") for seed in FORMAL_SEEDS)


def _model_registry(plan_path: Path) -> object:
    document = _read_json(plan_path, name="prediction manifest")
    return document.get("model_tasks")


def _read_task_store(root: Path, *, plan: PredictionPlan, group: str, seed: int | None,
                     artifact: Any, expected_identities: set[str]) -> FrozenTask:
    task_id = _task_identity(group, seed)
    store = root / task_id
    index = _read_json(store / "index.json", name=f"prediction index for {task_id}")
    if index.get("schema") not in {"nc_rted_prediction_index/v1", "nc_rted_prediction_index/v2"}:
        raise EvaluationError(f"prediction index for {task_id} is not a frozen v2 store")
    expected = {"run_id": plan.run_id, "model_task": task_id, "model_binding_sha256": artifact.manifest_sha256}
    if index.get("schema") == "nc_rted_prediction_index/v1":
        expected["manifest_sha256"] = plan.manifest_sha256
    if index.get("schema") == "nc_rted_prediction_index/v2":
        if (index.get("matrix_id") != plan.matrix_id or
                index.get("task_execution_binding_sha256") != _task_execution_binding(matrix_id=plan.matrix_id, artifact=artifact)):
            raise EvaluationError(f"prediction v2 matrix binding differs for {task_id}")
    if any(index.get(name) != value for name, value in expected.items()):
        raise EvaluationError(f"prediction index binding differs for {task_id}")
    entries = index.get("records")
    if not isinstance(entries, dict) or len(entries) != len(expected_identities):
        raise EvaluationError(f"prediction denominator is incomplete for {task_id}")
    records: dict[str, Mapping[str, Any]] = {}
    digests: dict[str, str] = {}
    for key, entry in entries.items():
        if not isinstance(key, str) or not isinstance(entry, dict):
            raise EvaluationError(f"prediction index entry differs for {task_id}")
        relative = entry.get("path")
        if (not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts or
                Path(relative).parent != Path("records")):
            raise EvaluationError(f"prediction record path differs for {task_id}")
        record = _read_json(store / relative, name=f"prediction record for {task_id}")
        digest = hashlib.sha256(canonical_json({name: value for name, value in record.items() if name != "record_sha256"})).hexdigest()
        if (record.get("schema") != "nc_rted_prediction_record/v1" or record.get("record_sha256") != digest or
                record.get("record_sha256") != entry.get("record_sha256") or record.get("status") != "success" or
                any(record.get(name) != value for name, value in expected.items()) or
                type(record.get("attempt")) is not int or record["attempt"] < 1 or record.get("retryable") is not False or
                entry.get("status") != record.get("status") or entry.get("attempt") != record.get("attempt")):
            raise EvaluationError(f"prediction record is not a frozen successful result for {task_id}")
        if index.get("schema") == "nc_rted_prediction_index/v2":
            provenance = record.get("provenance")
            if (not isinstance(record.get("manifest_sha256"), str) or len(record["manifest_sha256"]) != 64 or not isinstance(provenance, dict) or provenance.get("matrix_id") != plan.matrix_id or
                    provenance.get("task_execution_binding_sha256") != index["task_execution_binding_sha256"] or
                    provenance.get("model_group") != group or provenance.get("model_seed") != seed or
                    provenance.get("model_task") != task_id or provenance.get("model_manifest_sha256") != artifact.manifest_sha256 or
                    provenance.get("checkpoint_manifest_sha256") != artifact.checkpoint_manifest_sha256 or
                    provenance.get("checkpoint_state_sha256") != artifact.checkpoint_state_sha256 or
                    provenance.get("protocol") != dict(plan.protocol) or provenance.get("bindings") != dict(plan.bindings)):
                raise EvaluationError(f"prediction v2 record binding differs for {task_id}")
        identity = record.get("identity")
        if not isinstance(identity, str) or hashlib.sha256(identity.encode("utf-8")).hexdigest() != key or identity in records:
            raise EvaluationError(f"prediction identity differs for {task_id}")
        payload = record.get("payload")
        if identity.startswith("vad:"):
            _validate_vad_payload(payload)
        elif identity.startswith("vau:"):
            _validate_vau_payload(payload, max_new_tokens=plan.protocol["hivau"]["max_new_tokens"])
        else:
            raise EvaluationError(f"unknown prediction identity for {task_id}")
        records[identity], digests[identity] = record, digest
    if set(records) != expected_identities:
        raise EvaluationError(f"prediction identities differ for {task_id}")
    return FrozenTask(task_id, group, seed, records, digests)


def freeze_prediction_matrix(*, plan_path: str | Path, plan_sha256: str, stores_root: str | Path) -> FrozenMatrix:
    """Validate all checkpoints and all thirteen output stores without labels."""
    manifest = Path(plan_path).resolve()
    if _sha256(manifest) != plan_sha256:
        raise EvaluationError("prediction manifest SHA-256 differs")
    try:
        plan = load_prediction_plan(manifest, expected_sha256=plan_sha256)
    except PredictionInputError as error:
        raise EvaluationError("prediction plan is not an admitted full matrix") from error
    if plan.matrix_id is None or tuple((task.group, task.seed) for task in plan.models) != _expected_tasks():
        raise EvaluationError("prediction plan is not the complete 13-task matrix")
    # This replays the existing strict final-checkpoint attestation validation for every A/U/S/F seed.
    artifacts = {task.task_id: load_model_artifact(task.model_manifest, expected_sha256=task.model_manifest_sha256, task=task)
                 for task in plan.models}
    if any(artifacts[task.task_id].final_checkpoint_attestation_sha256 is None for task in plan.models if task.group != "R0"):
        raise EvaluationError("all twelve trained tasks require admitted final checkpoints")
    vad_ids = {f"vad:{request.dataset}:{request.media_id}" for request in plan.vad}
    vau_ids = {f"vau:{request.instruction_id}" for request in plan.vau}
    identities = vad_ids | vau_ids
    if len(identities) != len(plan.vad) + len(plan.vau):
        raise EvaluationError("prediction plan identities are not unique")
    tasks = {_task_identity(group, seed): _read_task_store(Path(stores_root), plan=plan, group=group, seed=seed,
             artifact=artifacts[_task_identity(group, seed)], expected_identities=identities)
             for group, seed in _expected_tasks()}
    return FrozenMatrix(plan, tasks)


def _module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise EvaluationError(f"cannot load inherited evaluator {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _official_modules(reactvau_root: Path) -> tuple[ModuleType, ModuleType]:
    vad_root = reactvau_root / "eval_utils" / "vad"
    hivau_root = reactvau_root / "eval_utils" / "hivau"
    if not vad_root.is_dir() or not hivau_root.is_dir():
        raise EvaluationError("ReactVAU official evaluator root differs")
    # The upstream files import these sibling modules by their original names.
    sys.path.insert(0, str(vad_root)); sys.path.insert(0, str(hivau_root))
    return _module(vad_root / "detect_utils.py", "nc_rted_official_detect_utils"), _module(hivau_root / "hivau_utils.py", "nc_rted_official_hivau_utils")


def _caption_references(path: Path, expected: set[str]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise EvaluationError("official HIVAU references are unreadable") from error
    for line in lines:
        row = json.loads(line)
        answer = row.get("answer") if isinstance(row, dict) else None
        answers = [answer] if isinstance(answer, str) else answer
        if (not isinstance(row, dict) or set(row) != {"id", "answer", "type"} or not isinstance(row["id"], str) or
                not isinstance(answers, list) or not answers or any(not isinstance(item, str) for item in answers) or
                row["type"] not in {"clip", "event", "video"}):
            raise EvaluationError("official HIVAU reference schema differs")
        if row["id"] in rows:
            raise EvaluationError("official HIVAU references duplicate an instruction")
        rows[row["id"]] = {"id": row["id"], "answers": answers, "type": row["type"]}
    if set(rows) != expected:
        raise EvaluationError("official HIVAU reference denominator differs")
    return rows


def _detection_rows(frozen: FrozenTask, plan: PredictionPlan, annotations: Mapping[str, Mapping[str, Any]], maker: Any) -> list[Prediction]:
    output: list[Prediction] = []
    for request in plan.vad:
        annotation = annotations[request.dataset].get(request.media_id)
        if not isinstance(annotation, dict):
            raise EvaluationError(f"official annotation is absent for {request.identity}")
        payload = frozen.records[request.identity]["payload"]
        scores = payload["causal_smoothed_scores"]
        labels = maker(annotation, payload["total_frames"])
        if len(scores) != len(labels):
            raise EvaluationError(f"official frame denominator differs for {request.identity}")
        dataset = "ucf-crime" if request.dataset == "ucf" else "xd-violence"
        family = f"{dataset}:{request.media_id.split('__#', 1)[0] if dataset == 'xd-violence' else request.media_id}"
        output.extend(Prediction(f"{request.identity}:frame:{index}", family, dataset, int(label), float(score))
                      for index, (label, score) in enumerate(zip(labels, scores)))
    return output


def _caption_rows(frozen: FrozenTask, plan: PredictionPlan, refs: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [{"pred": str(frozen.records[request.identity]["payload"]["text"]), "answers": list(refs[request.instruction_id]["answers"]),
             "type": refs[request.instruction_id]["type"]} for request in plan.vau]


def _caption_metrics(hivau: ModuleType, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Use the inherited PTB/COCO scorers while retaining every reference."""
    by_type: dict[str, dict[str, Any]] = {}
    for kind in ("clip", "event", "video"):
        selected = [row for row in rows if row["type"] == kind]
        if not selected:
            by_type[kind] = {"bleu_1_to_4": [0.0] * 4, "bleu_aggregate": 0.0, "rouge_l": 0.0,
                             "cider": 0.0, "meteor": 0.0, "denominator": 0}
            continue
        refs = {index: [{"caption": answer} for answer in row["answers"]] for index, row in enumerate(selected)}
        hyps = {index: [{"caption": row["pred"]}] for index, row in enumerate(selected)}
        refs, hyps = hivau.PTBTokenizer().tokenize(refs), hivau.PTBTokenizer().tokenize(hyps)
        # The frozen wrapper scores BLEU once per question then averages vectors.
        bleu_vectors = []
        for index in refs:
            score, _ = hivau.Bleu(4).compute_score({0: refs[index]}, {0: hyps[index]})
            bleu_vectors.append([float(value) for value in score])
        rouge, _ = hivau.Rouge().compute_score(refs, hyps)
        cider, _ = hivau.Cider().compute_score(refs, hyps)
        meteor, _ = hivau.Meteor().compute_score(refs, hyps)
        vector = [sum(values[column] for values in bleu_vectors) / len(bleu_vectors) for column in range(4)]
        by_type[kind] = {"bleu_1_to_4": vector, "bleu_aggregate": float(sum(vector)), "rouge_l": float(rouge),
                         "cider": float(cider), "meteor": float(meteor), "denominator": len(selected)}
    return {"by_granularity": by_type,
            "official_aggregate": {name: float(sum(by_type[kind][name] for kind in by_type) / 3.0)
                                  for name in ("bleu_aggregate", "rouge_l", "cider", "meteor")}}


def evaluate_frozen_matrix(*, frozen: FrozenMatrix, reactvau_root: str | Path, ucf_annotation: str | Path,
                           xd_annotation: str | Path, hivau_references: str | Path, output_dir: str | Path,
                           bootstrap_seed: int, bootstrap_repeats: int = 10_000, xd_video_dir: str | Path | None = None) -> dict[str, Any]:
    """Open official inputs only after ``freeze_prediction_matrix`` has succeeded."""
    output = Path(output_dir)
    if output.exists():
        raise EvaluationError("evaluation output directory must not already exist")
    detect, hivau = _official_modules(Path(reactvau_root).resolve())
    ucf = detect.load_anno_txt(str(ucf_annotation), "ucf-crime")
    xd = detect.load_anno_txt(str(xd_annotation), "xd-violence", video_dir=None if xd_video_dir is None else str(xd_video_dir))
    refs = _caption_references(Path(hivau_references), {request.instruction_id for request in frozen.plan.vau})
    output.mkdir(parents=True)
    metrics: dict[str, Any] = {"schema": "nc_rted_official_evaluation/v1", "plan_sha256": frozen.plan.manifest_sha256,
                                "matrix_id": frozen.plan.matrix_id, "tasks": {},
                                "provenance": {"reactvau_root": str(Path(reactvau_root).resolve()),
                                               "detect_utils": _file_binding(Path(reactvau_root) / "eval_utils/vad/detect_utils.py"),
                                               "hivau_utils": _file_binding(Path(reactvau_root) / "eval_utils/hivau/hivau_utils.py"),
                                               "ucf_annotation": _file_binding(ucf_annotation), "xd_annotation": _file_binding(xd_annotation),
                                               "hivau_references": _file_binding(hivau_references),
                                               "xd_video_dir": None if xd_video_dir is None else str(Path(xd_video_dir).resolve()),
                                               "preprocessing": "inherited PTBTokenizer and pycocoevalcap scorer classes",
                                               "packages": _package_versions()},
                                "frozen_records": {task_id: dict(task.record_sha256) for task_id, task in frozen.tasks.items()}}
    statistical: dict[str, dict[int, Sequence[Prediction]]] = {group: {} for group in ("A", "U", "S", "F")}
    for task in frozen.tasks.values():
        rows = _detection_rows(task, frozen.plan, {"ucf": ucf, "xd": xd}, detect.make_gt_labels_from_anno)
        by_dataset = {dataset: [row for row in rows if row.dataset == dataset] for dataset in ("ucf-crime", "xd-violence")}
        ucf_metrics = detect.compute_metrics([r.score for r in by_dataset["ucf-crime"]], [r.label for r in by_dataset["ucf-crime"]])
        xd_metrics = detect.compute_metrics([r.score for r in by_dataset["xd-violence"]], [r.label for r in by_dataset["xd-violence"]])
        task_metrics = {"completeness": {"missing": 0, "technical_failures": 0, "successful_records": len(task.records)},
                        "detection": {"ucf_auroc": float(ucf_metrics["roc_auc"]), "xd_ap": float(xd_metrics["ap"]),
                                      "ucf_frame_denominator": len(by_dataset["ucf-crime"]), "xd_frame_denominator": len(by_dataset["xd-violence"]),
                                      "vad_video_denominator": len(frozen.plan.vad)}}
        caption_rows = _caption_rows(task, frozen.plan, refs)
        task_metrics["caption"] = _caption_metrics(hivau, caption_rows)
        task_metrics["caption"]["empty_predictions"] = sum(not row["pred"].strip() for row in caption_rows)
        metrics["tasks"][task.task_id] = task_metrics
        if task.group != "R0": statistical[task.group][task.seed] = rows
    comparisons = evaluate_six_primary(statistical, bootstrap_seed=bootstrap_seed, bootstrap_repeats=bootstrap_repeats)
    metrics["six_primary"] = {name: asdict(value) for name, value in comparisons.items()}
    metrics["group_summary"] = {group: {metric: {"mean": statistics.mean(values), "sd": statistics.stdev(values)}
                                          for metric, values in {"ucf_auroc": [metrics["tasks"][_task_identity(group, seed)]["detection"]["ucf_auroc"] for seed in FORMAL_SEEDS],
                                                                 "xd_ap": [metrics["tasks"][_task_identity(group, seed)]["detection"]["xd_ap"] for seed in FORMAL_SEEDS]}.items()}
                                for group in ("A", "U", "S", "F")}
    metrics["bootstrap"] = {"seed": bootstrap_seed, "repeats": bootstrap_repeats, "holm": "six_predeclared_comparisons"}
    report = output / "official-evaluation.json"
    report.write_bytes(canonical_json(metrics) + b"\n")
    return metrics
