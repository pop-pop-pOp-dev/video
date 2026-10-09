"""Bind assembled train observations to compact teacher-pipeline records.

This module deliberately returns in-memory NumPy records.  Serializing all
process vectors to JSON would be prohibitively large; a later frozen writer
must use a chunked binary container and retain this exact schema.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np

from .batches import pack_observation_blocks
from .features import FeatureAssemblyResult, FeatureStatus, PROCESS_FEATURE_DIM
from .teacher_pipeline import (CELL_COUNT, PAIR_LIMIT, PIPELINE_SCHEMA, build_teachers,
                               detection_window_id, parse_frozen_records)


class TeacherRecordError(ValueError):
    pass


@dataclass(frozen=True)
class TeacherSourceTruth:
    """Trusted train-manifest binding for one detection prefix."""

    dataset: str
    key: str
    query_index: int
    family: str
    content_alias: str
    fold: int
    allocation: str
    teacher_reference_normal_eligible: bool
    observed_seconds: float

    @classmethod
    def from_manifest_rows(cls, prefix: Mapping[str, object], splitrow: Mapping[str, object]) -> "TeacherSourceTruth":
        """Bind a detection prefix to its one trusted source-split row."""
        required_prefix = ("dataset", "key", "family", "query_index", "observed_seconds",
                           "teacher_reference_normal_eligible")
        required_split = ("dataset", "key", "family", "allocation", "same_content_alias_group", "fold")
        if any(name not in prefix for name in required_prefix) or any(name not in splitrow for name in required_split):
            raise TeacherRecordError("manifest rows lack required teacher provenance")
        for name in ("dataset", "key", "family"):
            if prefix[name] != splitrow[name]:
                raise TeacherRecordError(f"prefix and split source {name} disagree")
        if splitrow["allocation"] != "train":
            raise TeacherRecordError("teacher source truth must be allocated to train")
        query_index, fold = prefix["query_index"], splitrow["fold"]
        observed = prefix["observed_seconds"]
        if (not isinstance(query_index, int) or query_index < 0 or not isinstance(fold, int) or not 0 <= fold < 5
                or not isinstance(observed, (int, float)) or isinstance(observed, bool)
                or not math.isfinite(float(observed)) or float(observed) < 0):
            raise TeacherRecordError("invalid prefix time or split fold")
        normal = _require_bool(prefix["teacher_reference_normal_eligible"], "teacher_reference_normal_eligible")
        alias = splitrow["same_content_alias_group"]
        if not isinstance(alias, str) or not alias:
            raise TeacherRecordError("split row lacks content alias")
        return cls(str(prefix["dataset"]), str(prefix["key"]), query_index, str(prefix["family"]), alias, fold,
                   "train", normal, float(observed))


@dataclass(frozen=True)
class CompactTeacherWindow:
    """Pipeline-shaped record retaining NumPy process cells and pair order."""

    record: dict
    relation_ids: tuple[str, ...]
    rejection: dict | None = None


def build_teachers_from_compact(compacts: tuple[CompactTeacherWindow, ...]) -> dict:
    """Run the pipeline for valid compact windows and retain every rejection."""
    if not compacts:
        raise TeacherRecordError("teacher compact batch must be nonempty")
    identifiers = [item.record.get("window_id") for item in compacts]
    if any(not isinstance(item, str) or not item for item in identifiers) or len(set(identifiers)) != len(identifiers):
        raise TeacherRecordError("compact records must have unique detection window IDs")
    accepted = [item.record for item in compacts if item.rejection is None]
    rejected = [item.record for item in compacts if item.rejection is not None]
    if accepted:
        pipeline = build_teachers(parse_frozen_records(
            {"schema": "nc_rted_frozen_feature_records/v1", "split": "train", "windows": accepted}
        ))
        rows = pipeline["rows"]
    else:
        rows = []
    rows.extend(rejected)
    if {row["window_id"] for row in rows} != set(identifiers) or len(rows) != len(identifiers):
        raise TeacherRecordError("compact teacher batch did not emit every window exactly once")
    return {"schema": PIPELINE_SCHEMA, "rows": sorted(rows, key=lambda row: row["window_id"])}


def _require_bool(value: object, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise TeacherRecordError(f"{name} must be a boolean")
    return bool(value)


def _rejection(truth: TeacherSourceTruth, reason: str) -> CompactTeacherWindow:
    return CompactTeacherWindow(
        record={"window_id": detection_window_id(truth.dataset, truth.key, truth.query_index),
                "dataset": truth.dataset, "aux_valid": False, "rejection": reason},
        relation_ids=(),
        rejection={"window_id": detection_window_id(truth.dataset, truth.key, truth.query_index), "reason": reason},
    )


def feature_assembly_to_teacher_window(result: FeatureAssemblyResult, truth: TeacherSourceTruth,
                                       relation_class_pairs: tuple[tuple[int, int], ...]) -> CompactTeacherWindow:
    """Adapt one real detection assembly without fabricating missing features.

    The source-truth object must come from the frozen train/split allocation
    and detection-prefix manifest.  Technical observations and absent
    backgrounds produce explicit rejection metadata rather than normal records.
    """
    if truth.allocation != "train":
        raise TeacherRecordError("teacher source truth must be allocated to train")
    if not 0 <= truth.fold < 5 or not truth.family or not truth.content_alias:
        raise TeacherRecordError("source truth lacks trusted family, alias, or fold")
    if not math.isfinite(truth.observed_seconds) or truth.observed_seconds < 0:
        raise TeacherRecordError("source truth lacks a finite observed time")
    normal = _require_bool(truth.teacher_reference_normal_eligible, "teacher_reference_normal_eligible")
    if result.status in {FeatureStatus.INVALID_INPUT, FeatureStatus.TRACKING_FAILURE}:
        return _rejection(truth, f"technical feature status: {result.status.value}")
    if result.status == FeatureStatus.NO_RELATION_PAIRS:
        if result.relations:
            raise TeacherRecordError("no-relation status contradicts assembled relations")
        return _rejection(truth, "no relation pairs in assembled observation")
    if result.status != FeatureStatus.OK:
        return _rejection(truth, f"unsupported feature status: {result.status}")
    if len(result.relations) > PAIR_LIMIT:
        raise TeacherRecordError("assembled relation count exceeds teacher limit")
    if len(relation_class_pairs) != len(result.relations):
        raise TeacherRecordError("trusted class-pair order must match assembled relations")
    if any(len(item) != 2 or any(not isinstance(value, int) or not 0 <= value < 80 for value in item)
           for item in relation_class_pairs):
        raise TeacherRecordError("trusted class pairs must be ordered COCO IDs")

    # This validates the student-side cells/masks without inventing a background
    # or changing the relation ordering used by task_inputs.teacher_batch.
    try:
        pack_observation_blocks([result], task="detection")
    except ValueError as error:
        return _rejection(truth, f"invalid assembled observation: {error}")

    pairs = []
    relation_ids = []
    static_anchors: list[tuple[str, np.ndarray, np.ndarray]] = []
    for position, relation in enumerate(result.relations):
        pair_id = f"{relation.first_track}:{relation.second_track}"
        if pair_id in relation_ids:
            raise TeacherRecordError("duplicate assembled relation identity")
        valid = np.asarray(relation.feature_valid)
        process = np.asarray(relation.process_cells, dtype=np.float64)
        if valid.shape != (CELL_COUNT,) or valid.dtype != np.bool_:
            raise TeacherRecordError("assembled relation validity mask is invalid")
        if process.shape != (CELL_COUNT, PROCESS_FEATURE_DIM):
            raise TeacherRecordError("assembled process cell shape is invalid")
        if not np.array_equal(valid, np.asarray(relation.cell_mask)):
            raise TeacherRecordError("assembled cell masks disagree")
        if not np.isfinite(process[valid]).all() or np.isfinite(process[~valid]).any():
            raise TeacherRecordError("assembled process cells violate masked-NaN policy")
        current_background = np.asarray(relation.static_background, dtype=np.float64)
        current_composition = np.asarray(relation.static_class_composition, dtype=np.float64)
        if current_background.shape != (1152,) or current_composition.shape != (80,):
            raise TeacherRecordError("assembled static feature shape is invalid")
        observed = np.asarray(relation.observed_times_s, dtype=np.float64)
        if observed.shape != (CELL_COUNT,) or not np.isfinite(observed[valid]).all():
            raise TeacherRecordError("assembled observed times are invalid")
        lower = max(0.0, truth.observed_seconds - 8.0)
        inside = observed[valid] >= lower if truth.observed_seconds < 8.0 else observed[valid] > lower
        if not inside.all() or (observed[valid] > truth.observed_seconds).any():
            raise TeacherRecordError("assembled observation lies outside its detection window")
        if (_require_bool(relation.background_valid, "background_valid") and np.isfinite(current_background).all()
                and np.isfinite(current_composition).all() and (current_composition >= 0).all()
                and current_composition.sum() > 0):
            static_anchors.append((pair_id, current_background, current_composition))
        pairs.append({"pair_id": pair_id, "class_pair": list(relation_class_pairs[position]),
                      "initial_geometry": np.asarray(relation.initial_geometry, dtype=np.float64).copy(),
                      "candidate_pair_count": len(result.relations), "valid_cells": valid.copy(),
                      "process_cells": process.copy()})
        relation_ids.append(pair_id)
    if not pairs:
        return _rejection(truth, "no relation pairs in assembled observation")
    if not static_anchors:
        return _rejection(truth, "missing reliable assembled background")
    # Anchor contexts can differ by relation birth time. Sorting by identity makes
    # this static aggregation invariant to a caller's candidate ordering.
    anchors = sorted(static_anchors, key=lambda item: item[0])
    background = np.mean(np.stack([item[1] for item in anchors]), axis=0)
    composition = np.mean(np.stack([item[2] for item in anchors]), axis=0)
    composition /= composition.sum()
    record = {"dataset": truth.dataset, "window_id": detection_window_id(truth.dataset, truth.key, truth.query_index),
              "source_family": truth.family, "content_alias": truth.content_alias, "fold": truth.fold,
              "normal_permitted": normal, "background": background.copy(), "background_valid": True,
              "class_composition": composition.copy(), "pairs": pairs}
    return CompactTeacherWindow(record=record, relation_ids=tuple(relation_ids))
