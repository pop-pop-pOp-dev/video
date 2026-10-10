"""Checkpoint-time NC-RTED mechanism diagnostics.

This module never selects official test data.  It creates a fixed development
prefix manifest from the sealed source allocation, and supplies interventions
that alter the actual evidence resampler inputs before Slow preparation.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping

import torch
from torch import Tensor
from torch.nn import functional as F

from .manifests import canonical_family, causal_query_groups, stable_rank
from .model import EvidenceOutput, RelationTimeEvidence

if TYPE_CHECKING:
    from .bridge import EvidenceSlowBridge, ObservationBatch, PreparedSlow, SlowInputs


DIAGNOSTIC_SCHEMA = "nc_rted_mechanism_development_prefixes/v1"
RUNTIME_INPUT_SCHEMA = "nc_rted_mechanism_development_runtime_inputs/v1"
FAST_PLAN_SCHEMA = "nc_rted_mechanism_development_fast_plan/v1"


class DiagnosticError(ValueError):
    pass


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _prefix_rows(databases: Mapping[str, Mapping], splits: list[Mapping]) -> list[dict]:
    allocation = {(row["dataset"], row["key"]): row["allocation"] for row in splits}
    rows = []
    for dataset in ("ucf-crime", "xd-violence"):
        for key, item in databases.get(dataset, {}).items():
            if allocation.get((dataset, key)) != "development":
                continue
            labels, events = item.get("label"), item.get("events", [])
            if not isinstance(labels, list) or not isinstance(events, list):
                raise DiagnosticError(f"development metadata is invalid: {dataset}:{key}")
            if labels and not events:
                raise DiagnosticError(f"positive development source lacks intervals: {dataset}:{key}")
            intervals = [(float(start), float(end)) for start, end in events] if labels else []
            fps, frames = float(item["fps"]), int(item["n_frames"])
            if not math.isfinite(fps) or fps <= 0 or frames < 1:
                raise DiagnosticError(f"development geometry is invalid: {dataset}:{key}")
            family = canonical_family(dataset, key)
            for query_index, indices in causal_query_groups(frames, fps):
                observed = indices[-1] / fps
                label = "anomalous" if any(start <= index / fps <= end for index in indices for start, end in intervals) else "normal"
                rows.append({"sample_id": f"development:{dataset}:{key}:{query_index}", "dataset": dataset,
                             "key": key, "family": family, "query_index": query_index,
                             "observed_seconds": observed, "class": label,
                             "scope": "vad_causal_latest_8s"})
    return rows


def select_development_prefixes(databases: Mapping[str, Mapping], splits: list[Mapping], *, requested: int = 256,
                                per_family_cap: int = 4) -> tuple[list[dict], str]:
    """Deterministically select 256 source/class-stratified development prefixes.

    Four dataset/current-class strata receive equal quota.  If any stratum cannot
    supply its quota, all legal development candidates are retained and the
    manifest says so rather than silently backfilling from training or test data.
    """
    if requested < 1 or requested % 4 or per_family_cap < 1:
        raise DiagnosticError("requested prefixes must be positive, divisible by four, with a positive family cap")
    all_rows = _prefix_rows(databases, splits)
    strata = {(dataset, label): [] for dataset in ("ucf-crime", "xd-violence") for label in ("normal", "anomalous")}
    for row in all_rows:
        strata[(row["dataset"], row["class"])].append(row)
    chosen, quota = [], requested // 4
    for cell in sorted(strata):
        counts, picked = {}, []
        for row in sorted(strata[cell], key=lambda value: stable_rank("nc-rted-mechanism-dev-v1", value["sample_id"])):
            if counts.get(row["family"], 0) >= per_family_cap:
                continue
            counts[row["family"]] = counts.get(row["family"], 0) + 1
            picked.append(row)
            if len(picked) == quota:
                break
        if len(picked) != quota:
            return sorted(all_rows, key=lambda value: value["sample_id"]), "ALL_LEGAL_DEVELOPMENT_PREFIXES_INSUFFICIENT_FOR_STRATIFIED_256"
        chosen.extend(picked)
    return sorted(chosen, key=lambda value: value["sample_id"]), "FIXED_SOURCE_AND_CLASS_STRATIFIED_256"


def write_development_manifest(output: str | Path, *, splits_path: str | Path, ucf_database_path: str | Path,
                               xd_database_path: str | Path, requested: int = 256, per_family_cap: int = 4) -> dict:
    output = Path(output)
    if output.exists():
        raise DiagnosticError("refusing to overwrite development diagnostic manifest")
    paths = {"source_splits": Path(splits_path), "ucf_database": Path(ucf_database_path), "xd_database": Path(xd_database_path)}
    if any(not path.is_file() for path in paths.values()):
        raise DiagnosticError("required sealed development-selection input is missing")
    splits = json.loads(paths["source_splits"].read_text())
    databases = {"ucf-crime": json.loads(paths["ucf_database"].read_text()), "xd-violence": json.loads(paths["xd_database"].read_text())}
    selected, status = select_development_prefixes(databases, splits, requested=requested, per_family_cap=per_family_cap)
    document = {"schema": DIAGNOSTIC_SCHEMA, "status": status, "requested": requested,
                "per_family_cap": per_family_cap, "scope": "development allocation only; no official test identities/answers/metrics",
                "inputs": {name: {"path": str(path.resolve()), "sha256": sha256_file(path)} for name, path in paths.items()},
                "records": selected}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name("." + output.name + ".pending")
    temporary.write_text(json.dumps(document, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    return document


def write_development_runtime_inputs(output: str | Path, *, development_manifest: str | Path,
                                     splits_path: str | Path, ucf_database_path: str | Path,
                                     xd_database_path: str | Path, media_catalog: str | Path) -> dict:
    """Bind legal source/query prefixes to an immutable training-media catalog.

    The resulting document is an execution input, not a cache: a later frozen
    observer must derive the causal observations for exactly these records.
    """
    output = Path(output)
    if output.exists(): raise DiagnosticError("refusing to overwrite development runtime inputs")
    paths = {"development_manifest": Path(development_manifest), "source_splits": Path(splits_path),
             "ucf_database": Path(ucf_database_path), "xd_database": Path(xd_database_path),
             "media_catalog": Path(media_catalog)}
    if any(not path.is_file() for path in paths.values()): raise DiagnosticError("required development runtime input is missing")
    development = json.loads(paths["development_manifest"].read_text())
    splits = json.loads(paths["source_splits"].read_text())
    databases = {"ucf-crime": json.loads(paths["ucf_database"].read_text()),
                 "xd-violence": json.loads(paths["xd_database"].read_text())}
    catalog = json.loads(paths["media_catalog"].read_text())
    rows = catalog.get("media") if isinstance(catalog, dict) else None
    if not isinstance(rows, list): raise DiagnosticError("development media catalog must contain media rows")
    media = {}
    for row in rows:
        if not isinstance(row, dict) or not {"dataset", "media_key", "media_path", "media_sha256"} <= set(row):
            raise DiagnosticError("development media catalog row is invalid")
        identity = (row["dataset"], row["media_key"])
        if identity in media: raise DiagnosticError("development media catalog has duplicate source")
        media[identity] = {name: row[name] for name in ("media_path", "media_sha256")}
    allocation = {(row["dataset"], row["key"]): row["allocation"] for row in splits}
    records = []
    for row in development.get("records", []):
        if not isinstance(row, dict) or row.get("scope") != "vad_causal_latest_8s": raise DiagnosticError("development prefix record is invalid")
        dataset, key = row.get("dataset"), row.get("key")
        if allocation.get((dataset, key)) != "development" or key not in databases.get(dataset, {}):
            raise DiagnosticError("development prefix escapes the sealed allocation")
        item = databases[dataset][key]; observed = float(row["observed_seconds"])
        if not 0 <= observed <= (int(item["n_frames"]) - 1) / float(item["fps"]):
            raise DiagnosticError("development prefix exceeds source duration")
        if (dataset, key) not in media: raise DiagnosticError("development media catalog omits selected source")
        records.append({**row, **media[(dataset, key)]})
    if not records: raise DiagnosticError("development manifest has no records")
    document = {"schema": RUNTIME_INPUT_SCHEMA, "scope": "development allocation only; causal prefixes only",
                "inputs": {name: {"path": str(path.resolve()), "sha256": sha256_file(path)} for name, path in paths.items()},
                "records": records}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return document


def write_development_media_catalog(output: str | Path, *, development_manifest: str | Path,
                                    ucf_media_root: str | Path, xd_media_root: str | Path) -> dict:
    """Resolve only selected development source keys to original train media."""
    output = Path(output)
    if output.exists(): raise DiagnosticError("refusing to overwrite development media catalog")
    manifest = Path(development_manifest)
    if not manifest.is_file(): raise DiagnosticError("development manifest is missing")
    document = json.loads(manifest.read_text())
    roots = {"ucf-crime": Path(ucf_media_root), "xd-violence": Path(xd_media_root)}
    if any(not root.is_dir() for root in roots.values()): raise DiagnosticError("development raw-media root is missing")
    selected = {(row["dataset"], row["key"]) for row in document.get("records", [])}
    indexes = {}
    for dataset, root in roots.items():
        index = {}
        for path in root.rglob("*.mp4"):
            if path.stem in index: raise DiagnosticError(f"ambiguous development raw media key: {dataset}:{path.stem}")
            index[path.stem] = path
        indexes[dataset] = index
    rows = []
    for dataset, key in sorted(selected):
        path = indexes[dataset].get(key)
        if path is None: raise DiagnosticError(f"development raw media is absent: {dataset}:{key}")
        try:
            import cv2
            capture = cv2.VideoCapture(str(path))
            fps, frames = float(capture.get(cv2.CAP_PROP_FPS)), int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            height, width = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            capture.release()
        except Exception as error: raise DiagnosticError(f"cannot inspect development media: {dataset}:{key}") from error
        if not math.isfinite(fps) or fps <= 0 or min(frames, height, width) < 1:
            raise DiagnosticError(f"invalid development media geometry: {dataset}:{key}")
        rows.append({"dataset": dataset, "media_key": key, "media_path": str(path.resolve()), "media_sha256": sha256_file(path),
                     "fps": fps, "frame_count": frames, "height": height, "width": width})
    result = {"schema": "nc_rted_development_media_catalog/v1", "development_manifest": {"path": str(manifest.resolve()), "sha256": sha256_file(manifest)}, "media": rows}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return result


def write_development_fast_plan(output: str | Path, *, runtime_inputs: str | Path,
                                fast_identity: Mapping[str, Any], protocols: Mapping[str, Any]) -> dict:
    """Produce the exact bounded Stage1 worklist required for development prefixes."""
    output, source = Path(output), Path(runtime_inputs)
    if output.exists(): raise DiagnosticError("refusing to overwrite development Fast plan")
    if not source.is_file(): raise DiagnosticError("development runtime inputs are missing")
    document = json.loads(source.read_text())
    if document.get("schema") != RUNTIME_INPUT_SCHEMA: raise DiagnosticError("development runtime input schema differs")
    if not isinstance(fast_identity, Mapping) or not fast_identity or not isinstance(protocols, Mapping):
        raise DiagnosticError("frozen Stage1 identity and inherited protocols are required")
    work = {}
    for row in document.get("records", []):
        identity = (row.get("dataset"), row.get("key"))
        if identity[0] not in protocols or not isinstance(identity[1], str): raise DiagnosticError("development source lacks inherited Fast protocol")
        current = work.get(identity)
        candidate = {key: row[key] for key in ("dataset", "key", "media_path", "media_sha256")}
        candidate["max_query_index"] = int(row["query_index"])
        if current is None: work[identity] = candidate
        else: current["max_query_index"] = max(current["max_query_index"], candidate["max_query_index"])
    rows = [work[key] for key in sorted(work)]
    result = {"schema": FAST_PLAN_SCHEMA, "scope": "development allocation only; compute frozen Stage1 Fast scores through max_query_index only",
              "runtime_inputs": {"path": str(source.resolve()), "sha256": sha256_file(source)},
              "fast_identity": dict(fast_identity), "protocols": dict(protocols), "sources": rows}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return result


def score_development_fast(plan: Mapping[str, Any], *, score_source: Callable[[Mapping[str, Any], int], list[float]]) -> dict:
    """Build a standard frozen Fast snapshot, requesting no query past each bound."""
    if plan.get("schema") != FAST_PLAN_SCHEMA or not isinstance(plan.get("sources"), list):
        raise DiagnosticError("development Fast plan schema differs")
    media = []
    for source in plan["sources"]:
        maximum = source.get("max_query_index")
        if type(maximum) is not int or maximum < 0: raise DiagnosticError("development Fast maximum query is invalid")
        source_path = Path(source["media_path"])
        if not source_path.is_file() or sha256_file(source_path) != source.get("media_sha256"):
            raise DiagnosticError("development Fast source media binding differs")
        runtime = plan.get("runtime_inputs")
        if not isinstance(runtime, Mapping) or sha256_file(runtime.get("path", "")) != runtime.get("sha256"):
            raise DiagnosticError("development Fast plan runtime-input binding differs")
        scores = score_source(source, maximum)
        if not isinstance(scores, list) or len(scores) != maximum + 1 or any(not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1 for value in scores):
            raise DiagnosticError("frozen scorer omitted or corrupted a bounded query score")
        import cv2
        capture = cv2.VideoCapture(source["media_path"])
        fps, frames = float(capture.get(cv2.CAP_PROP_FPS)), int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        height, width = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)); capture.release()
        interval = max(1, int(fps / 4))
        queries = [{"index": index, "frame_indices": list(range(index * 4 * interval, min((index + 1) * 4 * interval, frames), interval)), "fast_score": float(score)} for index, score in enumerate(scores)]
        media.append({"dataset": source["dataset"], "media_key": source["key"], "media_path": source["media_path"], "media_sha256": source["media_sha256"], "fps": fps, "frame_count": frames, "height": height, "width": width, "target_fps": 4, "query_interval": 4, "max_query_index": maximum, "queries": queries})
    return {"schema": "nc_rted_development_fast/v1", "fast_identity": plan["fast_identity"], "media": media}


def _rotate(values: Tensor) -> Tensor:
    return values if values.numel() < 2 else torch.roll(values, shifts=1, dims=0)


def permute_position(position: Tensor, valid: Tensor, *, axis: str) -> Tensor:
    """Permute only valid pi mass along time or relation support; invalid stays zero."""
    if position.shape != valid.shape or position.ndim != 4:
        raise DiagnosticError("position and valid must be matching [B,N,E,T] tensors")
    output = torch.zeros_like(position)
    for b in range(position.shape[0]):
        for n in range(position.shape[1]):
            if axis == "time":
                for e in range(position.shape[2]):
                    index = valid[b, n, e].nonzero(as_tuple=False).flatten()
                    output[b, n, e, index] = _rotate(position[b, n, e, index])
            elif axis == "relation":
                for t in range(position.shape[3]):
                    index = valid[b, n, :, t].nonzero(as_tuple=False).flatten()
                    output[b, n, index, t] = _rotate(position[b, n, index, t])
            else:
                raise DiagnosticError("axis must be time or relation")
    if not torch.equal(output.masked_select(~valid), torch.zeros_like(output.masked_select(~valid))):
        raise DiagnosticError("permutation placed mass on invalid support")
    if not torch.allclose(torch.sort(position[valid]).values, torch.sort(output[valid]).values):
        raise DiagnosticError("permutation changed valid pi multiset")
    return output


def _resample(module: RelationTimeEvidence, encoded: Tensor, valid: Tensor, quality: Tensor, position: Tensor) -> Tensor:
    b, blocks, candidates, times, _ = encoded.shape
    flat_values = module.value_projection(encoded).reshape(b, blocks * candidates * times, -1)
    flat_keys = module.key_projection(encoded).reshape(b, blocks * candidates * times, -1)
    flat_position, flat_valid = position.reshape(b, -1), valid.reshape(b, -1)
    gate = quality.to(flat_values.dtype).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, candidates, times).reshape(b, -1)
    logits = torch.einsum("qh,bkh->bqk", module.resampler_queries, flat_keys) / math.sqrt(flat_keys.shape[-1])
    logits = logits + torch.log(flat_position.clamp_min(1e-20)).unsqueeze(1)
    logits = logits.masked_fill(~flat_valid.unsqueeze(1), float("-inf"))
    safe_logits = logits.masked_fill((~flat_valid).all(dim=1).view(b, 1, 1), 0)
    attention = torch.softmax(safe_logits.float(), dim=-1).to(flat_values.dtype)
    attention = attention * flat_valid.unsqueeze(1).to(attention.dtype)
    attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(attention.dtype).eps)
    return module.language_projection(torch.einsum("bqk,bkh->bqh", attention, flat_values * gate.unsqueeze(-1)))


def evidence_with_position(module: RelationTimeEvidence, features: Tensor, valid: Tensor, observed_times: Tensor, *,
                           axis: str | None = None) -> EvidenceOutput:
    """Reproduce the audited branch and intervene before its actual resampler."""
    baseline = module(features, valid, observed_times)
    if axis is None:
        return baseline
    if features.shape[2] == 0:
        return baseline
    # Reconstruct the encoded tensor using audited module components, then apply
    # the altered position in the same attention/value resampler used by forward.
    valid = valid.bool(); safe_features = torch.where(valid.unsqueeze(-1), torch.nan_to_num(features), torch.zeros_like(features))
    if observed_times.shape == valid.shape[:2] + (valid.shape[-1],): observed_times = observed_times.unsqueeze(2).expand_as(valid)
    safe_times = torch.where(valid, torch.nan_to_num(observed_times), torch.zeros_like(observed_times))
    encoded = module.feature_projection(safe_features)
    time = module._time_features(safe_times.float()).to(module.time_projection[0].weight.dtype)
    encoded = encoded + module.time_projection(time).to(encoded.dtype)
    b, blocks, candidates, times, _ = encoded.shape
    padding = ~valid.reshape(b * blocks * candidates, times); safe_padding = padding.clone(); safe_padding[padding.all(dim=1), 0] = False
    encoded = module.temporal_encoder(encoded.reshape(b * blocks * candidates, times, -1), src_key_padding_mask=safe_padding).reshape_as(encoded)
    encoded = encoded.masked_fill(~valid.unsqueeze(-1), 0)
    position = permute_position(baseline.position, valid, axis=axis)
    tokens = _resample(module, encoded, valid, baseline.quality, position)
    return EvidenceOutput(baseline.quality, position, tokens, baseline.valid_blocks, baseline.aux_valid, valid)


def equal_norm_random_delta(structured_delta: Tensor, *, seed: int = 0) -> Tensor:
    """Deterministic equal-Frobenius-norm token control; zero stays identity."""
    norm = torch.linalg.vector_norm(structured_delta.float())
    if not bool(torch.isfinite(norm)):
        raise DiagnosticError("structured token delta is nonfinite")
    if float(norm) == 0.0:
        return torch.zeros_like(structured_delta)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    raw = torch.randn(structured_delta.shape, generator=generator, dtype=torch.float32).to(
        device=structured_delta.device, dtype=structured_delta.dtype)
    return raw * (norm.to(raw.dtype) / torch.linalg.vector_norm(raw.float()).to(raw.dtype))


@dataclass(frozen=True)
class InterventionResult:
    name: str
    token_delta_norm: float
    position: Tensor | None
    evidence_tokens: Tensor | None


def intervention_tokens(module: RelationTimeEvidence, features: Tensor, valid: Tensor, observed_times: Tensor, *, name: str) -> InterventionResult:
    baseline = evidence_with_position(module, features, valid, observed_times)
    if name == "branch_disable":
        return InterventionResult(name, 0.0, None, None)
    if name in {"time_permute", "relation_permute"}:
        changed = evidence_with_position(module, features, valid, observed_times, axis=name.removesuffix("_permute"))
        return InterventionResult(name, float(torch.linalg.vector_norm((changed.evidence_tokens - baseline.evidence_tokens).float())), changed.position, changed.evidence_tokens)
    if name == "equal_norm_random":
        # Caller pairs this control with a structured intervention and records its norm.
        raise DiagnosticError("equal_norm_random requires a paired structured token delta")
    raise DiagnosticError("unknown intervention")


def prepare_intervened(bridge: "EvidenceSlowBridge", inputs: "SlowInputs", observations: "ObservationBatch", *,
                       name: str, paired_structured_name: str | None = None) -> tuple["PreparedSlow", InterventionResult]:
    """Send an intervention through the inherited Slow preparation boundary.

    This is deliberately diagnostic-only.  It preserves the frozen bridge and
    calls the same inherited ``prepare_inputs_labels_for_LLM`` method that the
    bridge calls in ordinary execution.  The returned norm is the exact token
    delta delivered to that method, including the deterministic random control.
    """
    from .bridge import PreparedSlow

    if name == "branch_disable":
        return bridge.prepare(inputs, None, enabled=False), InterventionResult(name, 0.0, None, None)
    if name not in {"baseline", "time_permute", "relation_permute", "equal_norm_random"}:
        raise DiagnosticError("unknown intervention")

    # The disabled call performs the bridge's complete frozen-input/configuration
    # validation and establishes the inherited original visual path.
    bridge.prepare(inputs, None, enabled=False)
    module = bridge.evidence
    features = observations.features.to(device=next(module.parameters()).device, dtype=next(module.parameters()).dtype)
    valid = observations.valid.to(device=features.device)
    observed_times = observations.observed_times.to(device=features.device, dtype=torch.float32)
    baseline = evidence_with_position(module, features, valid, observed_times)
    if name == "baseline":
        output, delta, position = baseline, torch.zeros_like(baseline.evidence_tokens), baseline.position
    elif name == "equal_norm_random":
        if paired_structured_name not in {"time_permute", "relation_permute"}:
            raise DiagnosticError("equal_norm_random requires paired_structured_name time_permute or relation_permute")
        structured = evidence_with_position(module, features, valid, observed_times,
                                            axis=paired_structured_name.removesuffix("_permute"))
        delta = structured.evidence_tokens - baseline.evidence_tokens
        seed = int(hashlib.sha256(paired_structured_name.encode("ascii")).hexdigest()[:16], 16)
        output = EvidenceOutput(baseline.quality, baseline.position,
                                baseline.evidence_tokens + equal_norm_random_delta(delta, seed=seed),
                                baseline.valid_blocks, baseline.aux_valid, baseline.cell_valid)
        position = None
    else:
        output = evidence_with_position(module, features, valid, observed_times, axis=name.removesuffix("_permute"))
        delta = output.evidence_tokens - baseline.evidence_tokens
        position = output.position
    norm = float(torch.linalg.vector_norm((output.evidence_tokens - baseline.evidence_tokens).float()))
    visual = module.inject(inputs.visual_embeddings, output)
    raw = bridge.raw_slow
    active = torch.ones_like(inputs.input_ids, dtype=torch.bool) if inputs.attention_mask is None else inputs.attention_mask.bool()
    expected_length = int(active.sum()) - 1 + visual.shape[1]
    limit = getattr(raw.config, "tokenizer_model_max_length", None)
    if limit is not None and expected_length > limit:
        raise ValueError("inherited preparation would truncate tokens; full input is required")
    result = raw.prepare_inputs_labels_for_LLM(
        inputs.input_ids, inputs.position_ids, inputs.attention_mask, None, inputs.labels,
        inputs.images, [visual], ["video"], image_sizes=inputs.image_sizes,
    )
    ids, positions, attention, past, embeds, labels = result
    if embeds is None or embeds.shape[1] != expected_length:
        raise RuntimeError("inherited preparation bypassed or truncated intervened visual insertion")
    if inputs.labels is not None:
        if labels is None or int((labels != -100).sum()) != int(((inputs.labels != -100) & active).sum()):
            raise RuntimeError("supervised targets changed during intervened visual insertion")
    prepared = PreparedSlow(dict(input_ids=ids, position_ids=positions, attention_mask=attention,
                                 past_key_values=past, inputs_embeds=embeds, labels=labels), output, visual)
    return prepared, InterventionResult(name, norm, position, output.evidence_tokens)


def execute_detection_prefix(*, bridge: "EvidenceSlowBridge", task: Any, reader: Callable,
                             observation_reader: Callable, tokenizer: Any, protocol: Any,
                             name: str, generation_config: Mapping[str, Any],
                             yes_token_ids: Sequence[int], no_token_ids: Sequence[int], meter=None) -> dict:
    """Evaluate one legal causal detection prefix through the real Slow path.

    ``reader`` is the inherited ``StreamingDetectionReader`` and ``tokenizer``
    is ``InheritedTaskTokenizer``.  This keeps question construction, CE labels,
    visual-memory creation, and model execution on their audited paths.
    """
    from .batches import pack_observation_blocks
    from .detection_provider import DetectionMemoryReplay
    from .task_inputs import TaskInputError

    if task.task != "detection" or task.query_index is None or task.label not in {0, 1}:
        raise DiagnosticError("diagnostic execution requires one labeled detection prefix")
    replay = DetectionMemoryReplay(bridge.slow, protocol)
    prefix = reader(task.dataset, task.media_key, task.query_index)
    captured = None; queries = iter(prefix.queries)
    try:
        for query in queries:
            if query.index > task.query_index: raise TaskInputError("prefix reader supplied future query")
            target = query.index == task.query_index
            captured = replay.step(query, capture=target, image_height=prefix.image_height, image_width=prefix.image_width)
            if target: break
    finally:
        close = getattr(queries, "close", None)
        if close is not None: close()
    if captured is None: raise DiagnosticError("prefix reader omitted selected query")
    context, question = captured
    if abs(context.observed_seconds - float(task.observed_seconds)) > 1e-6:
        raise DiagnosticError("replayed endpoint differs from development manifest")
    observed = observation_reader(task.dataset, task.media_key, context.observed_seconds)
    observations = pack_observation_blocks([observed.features], task="detection",
                                           dtype=context.visual_embeddings.dtype, device=context.visual_embeddings.device)
    inputs = tokenizer.encode(task, context, detection_question=question, detection_scoring=protocol.scoring)
    if (not isinstance(generation_config, Mapping) or generation_config.get("do_sample") is not False or
            generation_config.get("num_beams", 1) != 1):
        raise DiagnosticError("diagnostic text requires an explicit greedy generation configuration")
    pair = None
    report_name = name
    if name.endswith("_random"):
        pair = name.removesuffix("_random")
        name = "equal_norm_random"
    prepared, intervention = prepare_intervened(bridge, inputs, observations, name=name, paired_structured_name=pair)
    with torch.no_grad():
        invoke = lambda: bridge.slow(**prepared.arguments, use_cache=False, return_dict=True)
        result = invoke() if meter is None else meter.slow(invoke)
        if result.loss is None or result.loss.ndim != 0 or not bool(torch.isfinite(result.loss)):
            raise DiagnosticError("inherited original task loss is unavailable")
        # Stage2 CE masks intentionally leave template markers unmasked, so
        # labels cannot identify the answer boundary. Build the inherited
        # answer-free Qwen prompt independently.
        from .prediction_adapters import BlindPromptTokenizer
        from llava import conversation
        from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
        from llava.mm_utils import tokenizer_image_token
        prompt_encoder = BlindPromptTokenizer(tokenizer.tokenizer, tokenizer.data_args, conversation.conv_templates,
                                              tokenizer_image_token, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)
        prompt_inputs = prompt_encoder.encode(question=question, context=context)
        prompt_prepared, _ = prepare_intervened(bridge, prompt_inputs, observations, name=intervention.name,
                                                paired_structured_name=pair)
        from transformers import Qwen2ForCausalLM
        arguments = prompt_prepared.arguments
        forward = lambda: Qwen2ForCausalLM.forward(bridge.raw_slow, position_ids=arguments["position_ids"],
            attention_mask=arguments["attention_mask"], inputs_embeds=arguments["inputs_embeds"], use_cache=False,
            return_dict=True)
        scored = forward() if meter is None else meter.slow(forward)
        logits = scored.logits[:, -1].float()
        # Use the frozen configured variants from the inherited blind VAD route.
        # Literal tokenizer strings are insufficient because the prediction path
        # takes the maximum logit over each approved variant set.
        if (not yes_token_ids or not no_token_ids or not all(type(value) is int and value >= 0 for value in (*yes_token_ids, *no_token_ids))):
            raise DiagnosticError("configured Yes/No token variants are invalid")
        y, n = logits[0, list(yes_token_ids)].max(), logits[0, list(no_token_ids)].max()
        probability = float((y - torch.logaddexp(y, n)).exp())
        generation_arguments = {key: arguments[key] for key in ("position_ids", "attention_mask", "inputs_embeds")}
        generate = lambda: Qwen2ForCausalLM.generate(bridge.raw_slow, **generation_arguments, **dict(generation_config))
        generated = generate() if meter is None else meter.slow(generate)
        if not isinstance(generated, Tensor) or generated.ndim != 2 or generated.shape[0] != 1:
            raise DiagnosticError("inherited greedy generation returned invalid token IDs")
        generated_ids = [int(value) for value in generated[0].detach().cpu().tolist()]
        greedy = tokenizer.tokenizer.decode(generated_ids, skip_special_tokens=True)
    return {"sample_id": task.sample_id, "intervention": report_name, "observed_seconds": context.observed_seconds,
            "label": int(task.label), "no_candidate": not bool(observations.valid.any()),
            "original_task_loss": float(result.loss), "detection_probability": probability,
            "fixed_greedy_token_ids": generated_ids, "fixed_greedy_text": greedy,
            "token_delta_frobenius_norm": intervention.token_delta_norm}


def development_detection_task(record: Mapping[str, Any]):
    """Convert a sealed prefix record to the existing inherited task contract."""
    from .task_inputs import TrainingTask
    required = {"sample_id", "dataset", "key", "family", "query_index", "observed_seconds", "class", "scope"}
    if not required <= set(record) or record["scope"] != "vad_causal_latest_8s" or record["class"] not in {"normal", "anomalous"}:
        raise DiagnosticError("development record cannot form an inherited detection task")
    return TrainingTask(str(record["sample_id"]), "detection", str(record["dataset"]), str(record["family"]),
                        str(record["key"]), float(record["observed_seconds"]), int(record["class"] == "anomalous"),
                        None, int(record["query_index"]))


def run_development_suite(output: str | Path, *, runtime_inputs: str | Path, bridge: "EvidenceSlowBridge",
                          reader: Callable, observation_reader: Callable, tokenizer: Any,
                          protocols: Mapping[str, Any], generation_config: Mapping[str, Any],
                          yes_token_ids: Sequence[int], no_token_ids: Sequence[int]) -> dict:
    """Run and immutably record all predeclared interventions for sealed inputs."""
    output = Path(output)
    if bridge.training:
        raise DiagnosticError("mechanism diagnostics require bridge.eval()")
    if output.exists(): raise DiagnosticError("refusing to overwrite mechanism diagnostic results")
    source = Path(runtime_inputs)
    document = json.loads(source.read_text())
    if document.get("schema") != RUNTIME_INPUT_SCHEMA or not isinstance(document.get("records"), list):
        raise DiagnosticError("development runtime input schema differs")
    rows = []
    for record in document["records"]:
        task = development_detection_task(record)
        protocol = protocols.get(task.dataset)
        if protocol is None: raise DiagnosticError("development source has no inherited detection protocol")
        for name in ("baseline", "branch_disable", "time_permute", "time_permute_random",
                     "relation_permute", "relation_permute_random"):
            rows.append(execute_detection_prefix(bridge=bridge, task=task, reader=reader,
                observation_reader=observation_reader, tokenizer=tokenizer, protocol=protocol,
                name=name, generation_config=generation_config, yes_token_ids=yes_token_ids, no_token_ids=no_token_ids))
    result = {"schema": "nc_rted_mechanism_diagnostic_results/v1",
              "runtime_inputs": {"path": str(source.resolve()), "sha256": sha256_file(source)},
              "generation_config": dict(generation_config), "records": rows}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return result
