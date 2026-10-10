"""Development-only teacher targets using immutable training compact windows."""
from __future__ import annotations

import numpy as np

from .teacher import calibrated_teacher, strength_matched_quality, stable_window_hash
from .teacher_pipeline import (PIPELINE_SCHEMA, PipelineError, _calibration_maxima, _fit_scales,
                               _select_window_references, _static_eligible_records, _window_teacher_distances,
                               _validate_fold_provenance, _window_threshold, parse_frozen_records, relation_ids)
from .teacher_records import CompactTeacherWindow


def _records(compacts: tuple[CompactTeacherWindow, ...], split: str):
    accepted = [item.record for item in compacts if item.rejection is None]
    records = () if not accepted else parse_frozen_records(
        {"schema": "nc_rted_frozen_feature_records/v1", "split": split, "windows": accepted})
    _validate_fold_provenance(records)
    return records


def summarize_development_teacher_targets(*, development_targets: tuple[CompactTeacherWindow, ...],
                                          training_compacts: tuple[CompactTeacherWindow, ...]) -> dict:
    """Emit development targets only; training compacts are references/calibration only."""
    if not development_targets or not training_compacts:
        raise ValueError("development targets and immutable training compacts are required")
    identifiers = [item.record.get("window_id") for item in development_targets]
    if any(not isinstance(value, str) or not value for value in identifiers) or len(set(identifiers)) != len(identifiers):
        raise ValueError("development compact windows need unique nonempty IDs")
    training = _records(training_compacts, "train")
    targets = _records(development_targets, "train")
    output = [dict(item.record) for item in development_targets if item.rejection is not None]
    pending = []
    prepared = {}
    for target in targets:
        fold = target.fold; c = tuple(x for x in training if x.fold == (fold + 1) % 5 and x.normal_permitted)
        if fold not in prepared:
            candidates = tuple(x for x in training if x.fold not in {fold, (fold + 1) % 5} and x.normal_permitted)
            refs, excluded = _static_eligible_records(candidates)
            prepared[fold] = (refs, c, {"r_normal_candidate_count": len(candidates), "r_static_eligible_count": len(refs), "r_static_excluded": excluded}, None, None)
        refs, c, support, threshold, scales = prepared[fold]
        relation_support, calibration_support = {}, {}
        try:
            if threshold is None:
                threshold, scales = _window_threshold(refs), _fit_scales(refs)
                prepared[fold] = (refs, c, support, threshold, scales)
            chosen = _select_window_references(target, refs, threshold)
            distances, valid = _window_teacher_distances(target, chosen, scales, relation_support)
            if not valid.any(): raise PipelineError("window has no valid relation-time cells after reference matching")
            calibration = _calibration_maxima(target, chosen, c, threshold, scales, calibration_support)
            quality, positions, joint = calibrated_teacher(distances, calibration, valid)
            row = {"window_id": target.window_id, "dataset": target.dataset, "aux_valid": True, "mask": valid.tolist(), "relation_ids": relation_ids(target), "F_quality": quality, "S_quality": quality, "F_positions": positions.tolist(), "S_positions": positions.tolist(), "joint": joint.tolist(), "M": float(np.median(distances[:, valid], axis=0).max()), "threshold": threshold, "calibration_count": int(calibration.size), "static_support": support, "relation_support": relation_support, "calibration_support": calibration_support}
            output.append(row); pending.append(row)
        except (PipelineError, ValueError) as error:
            output.append({"window_id": target.window_id, "dataset": target.dataset, "aux_valid": False, "rejection": str(error), "static_support": support, "relation_support": relation_support, "calibration_support": calibration_support})
    for dataset in {x["dataset"] for x in pending}:
        rows = [x for x in pending if x["dataset"] == dataset]
        for row, value in zip(rows, strength_matched_quality(np.asarray([x["F_quality"] for x in rows]), np.asarray([x["M"] for x in rows]), [stable_window_hash("teacher-only", x["window_id"], 0.) for x in rows])): row["U_quality"] = float(value)
    for row in output:
        if not row["aux_valid"]: row.update(F_quality=None, S_quality=None, U_quality=None, F_positions=None, S_positions=None, joint=None)
    return {"schema": PIPELINE_SCHEMA, "rows": sorted(output, key=lambda x: x["window_id"])}
