"""Record-driven, source-isolated NC-RTED teacher orchestration.

This is an offline training-teacher builder.  It never accepts official test
records, detector outputs, or learned/student values.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from .alignment import constrained_alignment, fit_reference_scale, process_cost_matrix
from .features import PROCESS_BLOCK_SLICES, PROCESS_FEATURE_DIM
from .retrieval import StaticPair, pair_compatible
from .teacher import calibrated_teacher, strength_matched_quality, stable_window_hash


PIPELINE_SCHEMA = "nc_rted_teacher_pipeline/v1"
CELL_COUNT = 4
PAIR_LIMIT = 16


class PipelineError(ValueError):
    pass


@dataclass(frozen=True)
class PairRecord:
    pair_id: str
    class_pair: tuple[int, int]
    initial_geometry: np.ndarray
    candidate_pair_count: int
    valid_cells: np.ndarray
    process_cells: np.ndarray


@dataclass(frozen=True)
class WindowRecord:
    dataset: str
    window_id: str
    source_family: str
    content_alias: str
    fold: int
    normal_permitted: bool
    background: np.ndarray
    background_valid: bool
    class_composition: np.ndarray
    pairs: tuple[PairRecord, ...]


@dataclass(frozen=True)
class WindowStatic:
    background: np.ndarray
    class_composition: np.ndarray
    aggregate_initial_geometry: np.ndarray
    support: np.ndarray


def _array(value: object, shape: tuple[int, ...], name: str, *, boolean: bool = False) -> np.ndarray:
    if boolean:
        raw = np.asarray(value, dtype=object)
        if raw.shape != shape or not all(isinstance(item, (bool, np.bool_)) for item in raw.flat):
            raise PipelineError(f"invalid {name}")
        return raw.astype(bool)
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise PipelineError(f"invalid {name}")
    return result


def _bool(value: object, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise PipelineError(f"invalid {name}")
    return bool(value)


def parse_frozen_records(document: dict) -> tuple[WindowRecord, ...]:
    """Parse an explicit train-only frozen-record schema without media access."""
    if document.get("schema") != "nc_rted_frozen_feature_records/v1" or document.get("split") != "train":
        raise PipelineError("input must be train-only nc_rted_frozen_feature_records/v1")
    rows = document.get("windows")
    if not isinstance(rows, list) or not rows:
        raise PipelineError("frozen feature input requires nonempty windows")
    output = []
    for row in rows:
        pairs = []
        for pair in row.get("pairs", []):
            valid = _array(pair.get("valid_cells"), (CELL_COUNT,), "pair valid cells", boolean=True)
            process = np.asarray(pair.get("process_cells"), dtype=np.float64)
            if process.shape != (CELL_COUNT, PROCESS_FEATURE_DIM):
                raise PipelineError("invalid pair process cells")
            if not np.isfinite(process[valid]).all() or np.isfinite(process[~valid]).any():
                raise PipelineError("valid process cells must be finite and missing cells must be NaN")
            pairs.append(PairRecord(str(pair["pair_id"]), tuple(pair["class_pair"]), _array(pair["initial_geometry"], (5,), "initial geometry"),
                                    int(pair["candidate_pair_count"]), valid, process))
        if len(pairs) > PAIR_LIMIT:
            raise PipelineError("each window permits at most 16 relation pairs")
        if len({pair.pair_id for pair in pairs}) != len(pairs):
            raise PipelineError("pair IDs must be unique within a window")
        output.append(WindowRecord(str(row["dataset"]), str(row["window_id"]), str(row["source_family"]),
                                   str(row["content_alias"]), int(row["fold"]), _bool(row["normal_permitted"], "normal_permitted"),
                                   _array(row["background"], (1152,), "background"), _bool(row["background_valid"], "background_valid"),
                                   _array(row["class_composition"], (80,), "class composition"), tuple(pairs)))
    if len({row.window_id for row in output}) != len(output):
        raise PipelineError("window IDs must be globally unique")
    for row in output:
        if not 0 <= row.fold < 5 or not np.isclose(row.class_composition.sum(), 1.0, atol=1e-6) or (row.class_composition < 0).any():
            raise PipelineError("invalid fold or normalized class composition")
    return tuple(output)


def _window_static(window: WindowRecord) -> WindowStatic:
    if not window.background_valid or np.linalg.norm(window.background) <= 1e-8:
        raise PipelineError("missing reliable window background")
    total_valid = sum(int(pair.valid_cells.sum()) for pair in window.pairs)
    if not total_valid:
        raise PipelineError("window has no valid relation-time support")
    return WindowStatic(window.background, window.class_composition,
                        np.mean(np.stack([pair.initial_geometry for pair in window.pairs]), axis=0),
                        np.asarray((len(window.pairs) / PAIR_LIMIT, total_valid / (PAIR_LIMIT * CELL_COUNT)), dtype=np.float64))


def _window_distance(first: WindowStatic, second: WindowStatic) -> float:
    cosine = np.dot(first.background, second.background) / (np.linalg.norm(first.background) * np.linalg.norm(second.background))
    blocks = [1.0 - float(np.clip(cosine, -1.0, 1.0))]
    for field in ("class_composition", "aggregate_initial_geometry", "support"):
        blocks.append(float(np.sqrt(np.mean((getattr(first, field) - getattr(second, field)) ** 2))))
    return float(np.mean(blocks))


def _process_blocks(pair: PairRecord) -> dict[str, np.ndarray]:
    return {name: pair.process_cells[:, part] for name, part in PROCESS_BLOCK_SLICES.items()}


def _validate_fold_provenance(records: tuple[WindowRecord, ...]) -> None:
    for field in ("source_family", "content_alias"):
        mapping: dict[str, int] = {}
        for record in records:
            value = getattr(record, field)
            if value in mapping and mapping[value] != record.fold:
                raise PipelineError(f"{field} crosses teacher folds")
            mapping[value] = record.fold


def _window_threshold(records: tuple[WindowRecord, ...]) -> float:
    values = []
    for window in records:
        descriptor = _window_static(window)
        neighbors = [_window_distance(descriptor, _window_static(other)) for other in records
                     if other.window_id != window.window_id and other.source_family != window.source_family
                     and other.content_alias != window.content_alias]
        if not neighbors:
            raise PipelineError(f"unsupported R window without leave-one-source static neighbor: {window.window_id}")
        values.append(min(neighbors))
    if not values:
        raise PipelineError("R threshold requires normal windows")
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    return float(ordered[int(np.ceil(.95 * len(ordered))) - 1])


def _static_eligible_records(records: tuple[WindowRecord, ...]) -> tuple[tuple[WindowRecord, ...], list[dict[str, str]]]:
    """Remove technical static failures before they can influence retrieval."""
    usable, excluded = [], []
    for window in records:
        try:
            _window_static(window)
        except PipelineError as error:
            excluded.append({"window_id": window.window_id, "reason": str(error)})
        else:
            usable.append(window)
    return tuple(usable), excluded


def _select_window_references(target: WindowRecord, candidates: tuple[WindowRecord, ...], threshold: float) -> tuple[WindowRecord, ...]:
    target_static = _window_static(target)
    ranked = []
    for candidate in candidates:
        if candidate.source_family == target.source_family or candidate.content_alias == target.content_alias:
            continue
        try:
            distance = _window_distance(target_static, _window_static(candidate))
        except PipelineError:
            continue
        if distance <= threshold:
            ranked.append((distance, hashlib.sha256((candidate.window_id + "\0window-static-v1").encode()).hexdigest(), candidate))
    selected, families, aliases = [], {target.source_family}, {target.content_alias}
    for _, _, candidate in sorted(ranked):
        if candidate.source_family in families or candidate.content_alias in aliases:
            continue
        selected.append(candidate); families.add(candidate.source_family); aliases.add(candidate.content_alias)
        if len(selected) == 3:
            return tuple(selected)
    raise PipelineError("need three threshold-compatible distinct normal R windows")


def _select_calibration_windows(target: WindowRecord, references: tuple[WindowRecord, ...], candidates: tuple[WindowRecord, ...], threshold: float) -> tuple[WindowRecord, ...]:
    target_static = _window_static(target)
    families = {target.source_family, *(item.source_family for item in references)}
    aliases = {target.content_alias, *(item.content_alias for item in references)}
    ranked = []
    for candidate in candidates:
        if candidate.source_family in families or candidate.content_alias in aliases:
            continue
        try:
            distance = _window_distance(target_static, _window_static(candidate))
        except PipelineError:
            continue
        if distance <= threshold:
            ranked.append((distance, hashlib.sha256((candidate.window_id + "\0window-calibration-v1").encode()).hexdigest(), candidate))
    selected, per_family, alias_owner = [], {}, {}
    for _, _, candidate in sorted(ranked):
        if per_family.get(candidate.source_family, 0) >= 4 or (candidate.content_alias in alias_owner and alias_owner[candidate.content_alias] != candidate.source_family):
            continue
        selected.append(candidate); per_family[candidate.source_family] = per_family.get(candidate.source_family, 0) + 1; alias_owner[candidate.content_alias] = candidate.source_family
        if len(selected) == 128:
            break
    if len(selected) < 64 or len(per_family) < 16:
        raise PipelineError("need 64..128 threshold-compatible C windows from >=16 families, max four per family")
    return tuple(selected)


def _fit_scales(r_records: tuple[WindowRecord, ...]) -> dict[str, object]:
    rows = {name: [] for name in PROCESS_BLOCK_SLICES}
    for window in r_records:
        if not window.normal_permitted:
            continue
        for pair in window.pairs:
            for name, block in _process_blocks(pair).items():
                rows[name].append(block[pair.valid_cells])
    try:
        return {name: fit_reference_scale(np.concatenate(values, axis=0)) for name, values in rows.items() if values}
    except ValueError as error:
        raise PipelineError(f"invalid R process scale input: {error}") from error


def _matched_pair(query: PairRecord, reference: WindowRecord) -> PairRecord:
    query_static = StaticPair(query.pair_id, query.class_pair, query.initial_geometry, query.candidate_pair_count, int(query.valid_cells.sum()))
    matches = []
    for candidate in reference.pairs:
        candidate_static = StaticPair(candidate.pair_id, candidate.class_pair, candidate.initial_geometry, candidate.candidate_pair_count, int(candidate.valid_cells.sum()))
        if pair_compatible(query_static, candidate_static):
            matches.append((float(np.sqrt(np.mean((query.initial_geometry - candidate.initial_geometry) ** 2))), candidate.pair_id, candidate))
    if not matches:
        raise PipelineError(f"fixed reference window {reference.window_id} has no compatible relation pair")
    return min(matches)[2]


def _pair_distances(query: PairRecord, references: tuple[WindowRecord, ...], scales: dict[str, object]) -> np.ndarray:
    values = []
    for reference_window in references:
        reference = _matched_pair(query, reference_window)
        cost = process_cost_matrix(_process_blocks(query), _process_blocks(reference), scales, query.valid_cells, reference.valid_cells)
        aligned = constrained_alignment(cost, query.valid_cells, reference.valid_cells)
        if not aligned.valid:
            raise PipelineError("no legal short-DTW alignment")
        values.append(aligned.cell_distance)
    return np.asarray(values, dtype=np.float64)


def _window_teacher_distances(window: WindowRecord, references: tuple[WindowRecord, ...], scales: dict[str, object],
                              support: dict[str, object] | None = None) -> tuple[np.ndarray, np.ndarray]:
    distances = np.full((3, PAIR_LIMIT * CELL_COUNT), np.nan, dtype=np.float64)
    valid = np.zeros(PAIR_LIMIT * CELL_COUNT, dtype=bool)
    rejected = []
    for pair_index, pair in enumerate(window.pairs):
        try:
            pair_distances = _pair_distances(pair, references, scales)
        except (PipelineError, ValueError) as error:
            rejected.append({"pair_id": pair.pair_id, "reason": str(error)})
            continue
        flat = slice(pair_index * CELL_COUNT, (pair_index + 1) * CELL_COUNT)
        distances[:, flat] = pair_distances
        valid[flat] = pair.valid_cells & np.isfinite(pair_distances).all(axis=0)
    if support is not None:
        support.update({"pair_count": len(window.pairs), "supported_pair_count": len(window.pairs) - len(rejected),
                        "valid_cell_count": int(valid.sum()), "rejected_pairs": rejected})
    return distances, valid


def relation_ids(window: WindowRecord) -> list[str]:
    """Audit-only pair order: ID i maps to flattened slots [4*i:4*i+4]."""
    return [pair.pair_id for pair in window.pairs]


def detection_window_id(dataset: str, key: str, query_index: int) -> str:
    """Stable manifest identity for a detection prefix; no source hash is used."""
    if not dataset or not key or not isinstance(query_index, int) or query_index < 0:
        raise PipelineError("invalid detection window identity")
    return f"detection:{dataset}:{key}:{query_index}"


def _window_m(window: WindowRecord, references: tuple[WindowRecord, ...], scales: dict[str, object],
              support: dict[str, object] | None = None) -> float:
    distances, valid = _window_teacher_distances(window, references, scales, support)
    if not valid.any():
        raise PipelineError("window has no valid relation-time cells")
    return float(np.median(distances[:, valid], axis=0).max())


def _calibration_maxima(target: WindowRecord, references: tuple[WindowRecord, ...], c_records: tuple[WindowRecord, ...], threshold: float,
                         scales: dict[str, object], support: dict[str, object] | None = None) -> np.ndarray:
    chosen = _select_calibration_windows(target, references, c_records, threshold)
    maxima = []
    unsupported = []
    for window in chosen:
        window_support = {}
        try:
            maxima.append(_window_m(window, references, scales, window_support))
        except PipelineError as error:
            unsupported.append({"window_id": window.window_id, "reason": str(error), "relation_support": window_support})
    if support is not None:
        support.update({"selected_window_count": len(chosen), "supported_window_count": len(maxima),
                        "unsupported_windows": unsupported})
    if not 64 <= len(maxima) <= 128:
        raise PipelineError("calibration maxima count differs from retrieval support after relation support filtering")
    return np.asarray(maxima, dtype=np.float64)


def build_teachers(records: tuple[WindowRecord, ...]) -> dict:
    """Construct F/S targets and rejection rows, then assign U per dataset globally."""
    _validate_fold_provenance(records)
    output, pending_u = [], []
    for fold in range(5):
        q_records = tuple(row for row in records if row.fold == fold)
        c_records = tuple(row for row in records if row.fold == (fold + 1) % 5 and row.normal_permitted)
        r_candidates = tuple(row for row in records if row.fold not in {fold, (fold + 1) % 5} and row.normal_permitted)
        r_records, r_excluded = _static_eligible_records(r_candidates)
        static_support = {"r_normal_candidate_count": len(r_candidates), "r_static_eligible_count": len(r_records),
                          "r_static_excluded": r_excluded}
        try:
            threshold, scales = _window_threshold(r_records), _fit_scales(r_records)
        except PipelineError as error:
            for row in q_records:
                output.append({"window_id": row.window_id, "dataset": row.dataset, "aux_valid": False,
                               "rejection": str(error), "static_support": static_support})
            continue
        for target in q_records:
            try:
                references = _select_window_references(target, r_records, threshold)
                relation_support, calibration_support = {}, {}
                target_distances, target_valid = _window_teacher_distances(target, references, scales, relation_support)
                if not target_valid.any():
                    raise PipelineError("window has no valid relation-time cells after reference matching")
                calibration = _calibration_maxima(target, references, c_records, threshold, scales, calibration_support)
                quality, positions, joint = calibrated_teacher(target_distances, calibration, target_valid)
                maximum = float(np.median(target_distances[:, target_valid], axis=0).max())
                row = {"window_id": target.window_id, "dataset": target.dataset, "aux_valid": True,
                       "mask": target_valid.tolist(), "relation_ids": relation_ids(target), "F_quality": quality, "S_quality": quality,
                       "F_positions": positions.tolist(), "S_positions": positions.tolist(), "joint": joint.tolist(),
                       "M": maximum, "threshold": threshold, "calibration_count": int(calibration.size),
                       "static_support": static_support, "relation_support": relation_support,
                       "calibration_support": calibration_support}
                output.append(row); pending_u.append(row)
            except (PipelineError, ValueError) as error:
                output.append({"window_id": target.window_id, "dataset": target.dataset, "aux_valid": False,
                               "rejection": str(error), "static_support": static_support})
    for dataset in sorted({row["dataset"] for row in pending_u}):
        rows = [row for row in pending_u if row["dataset"] == dataset]
        qualities = strength_matched_quality(np.asarray([row["F_quality"] for row in rows]), np.asarray([row["M"] for row in rows]),
                                            [stable_window_hash("teacher-only", row["window_id"], 0.0) for row in rows])
        for row, quality in zip(rows, qualities):
            row["U_quality"] = float(quality)
    for row in output:
        if not row["aux_valid"]:
            row["F_quality"] = row["S_quality"] = row["U_quality"] = None
            row["F_positions"] = row["S_positions"] = row["joint"] = None
    return {"schema": PIPELINE_SCHEMA, "rows": sorted(output, key=lambda row: row["window_id"])}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def publish_teacher_manifest(output: Path, manifest: dict, input_path: Path, config_path: Path) -> None:
    """Write once through an atomic directory replacement with input provenance."""
    if output.exists():
        raise PipelineError("refusing to overwrite existing teacher output")
    if not input_path.is_file() or not config_path.is_file():
        raise PipelineError("teacher input and frozen config paths must exist")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        value = {**manifest, "provenance": {"input_sha256": file_sha256(input_path), "config_sha256": file_sha256(config_path)}}
        (temporary / "teacher_manifest.json").write_text(json.dumps(value, sort_keys=True, allow_nan=False, indent=2) + "\n")
        os.replace(temporary, output)
    except Exception:
        for child in temporary.glob("*"):
            child.unlink()
        temporary.rmdir()
        raise
