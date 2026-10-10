"""Convert sealed development observations into target-only teacher windows."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np

from .batches import pack_observation_blocks
from .features import FeatureAssemblyResult, FeatureStatus, PROCESS_FEATURE_DIM
from .teacher_pipeline import CELL_COUNT, PAIR_LIMIT, detection_window_id
from .teacher_records import CompactTeacherWindow, TeacherRecordError


@dataclass(frozen=True)
class DevelopmentTargetSource:
    """Trusted development-only provenance for one sealed causal prefix."""

    dataset: str
    key: str
    query_index: int
    family: str
    content_alias: str
    fold: int
    observed_seconds: float

    @classmethod
    def from_sealed_records(cls, prefix: Mapping[str, object], splitrow: Mapping[str, object]) -> "DevelopmentTargetSource":
        """Bind a sealed development prefix to its one sealed source-split row."""
        required_prefix = ("sample_id", "dataset", "key", "family", "query_index", "observed_seconds", "class", "scope")
        required_split = ("dataset", "key", "family", "allocation", "same_content_alias_group", "fold")
        if any(name not in prefix for name in required_prefix) or any(name not in splitrow for name in required_split):
            raise TeacherRecordError("sealed development records lack required target provenance")
        if prefix["scope"] != "vad_causal_latest_8s":
            raise TeacherRecordError("development prefix has an unsupported causal scope")
        if not isinstance(prefix["sample_id"], str) or not prefix["sample_id"]:
            raise TeacherRecordError("development prefix lacks a sample identity")
        if prefix["class"] not in {"normal", "anomalous"}:
            raise TeacherRecordError("development prefix lacks a supported source class")
        for name in ("dataset", "key", "family"):
            if not isinstance(prefix[name], str) or not prefix[name] or prefix[name] != splitrow[name]:
                raise TeacherRecordError(f"development prefix and split source {name} disagree")
        if splitrow["allocation"] != "development":
            raise TeacherRecordError("development target source must be allocated to development")
        query_index, fold, observed = prefix["query_index"], splitrow["fold"], prefix["observed_seconds"]
        if (type(query_index) is not int or query_index < 0 or type(fold) is not int or not 0 <= fold < 5
                or not isinstance(observed, (int, float)) or isinstance(observed, bool)
                or not math.isfinite(float(observed)) or float(observed) < 0):
            raise TeacherRecordError("invalid sealed development prefix time or fold")
        alias = splitrow["same_content_alias_group"]
        if not isinstance(alias, str) or not alias:
            raise TeacherRecordError("development split lacks a content alias")
        return cls(prefix["dataset"], prefix["key"], query_index, prefix["family"], alias, fold, float(observed))


def _require_bool(value: object, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise TeacherRecordError(f"{name} must be a boolean")
    return bool(value)


def _rejection(source: DevelopmentTargetSource, reason: str) -> CompactTeacherWindow:
    window_id = detection_window_id(source.dataset, source.key, source.query_index)
    return CompactTeacherWindow(
        record={"window_id": window_id, "dataset": source.dataset, "aux_valid": False, "rejection": reason},
        relation_ids=(), rejection={"window_id": window_id, "reason": reason},
    )


def development_feature_assembly_to_teacher_window(
        result: FeatureAssemblyResult, relation_class_pairs: tuple[tuple[int, int], ...], *,
        development_prefix: Mapping[str, object], development_source_split: Mapping[str, object]) -> CompactTeacherWindow:
    """Convert one sealed development observation without creating a train record.

    Development windows are teacher targets only.  Their source allocation and
    frozen fold are checked from the provided sealed rows, while their compact
    records retain only the fields parsed by ``parse_frozen_records``.
    """
    source = DevelopmentTargetSource.from_sealed_records(development_prefix, development_source_split)
    if result.status in {FeatureStatus.INVALID_INPUT, FeatureStatus.TRACKING_FAILURE}:
        return _rejection(source, f"technical feature status: {result.status.value}")
    if result.status == FeatureStatus.NO_RELATION_PAIRS:
        if result.relations:
            raise TeacherRecordError("no-relation status contradicts assembled relations")
        return _rejection(source, "no relation pairs in assembled observation")
    if result.status != FeatureStatus.OK:
        return _rejection(source, f"unsupported feature status: {result.status}")
    if len(result.relations) > PAIR_LIMIT:
        raise TeacherRecordError("assembled relation count exceeds teacher limit")
    if len(relation_class_pairs) != len(result.relations):
        raise TeacherRecordError("trusted class-pair order must match assembled relations")
    if any(len(item) != 2 or any(type(value) is not int or not 0 <= value < 80 for value in item)
           for item in relation_class_pairs):
        raise TeacherRecordError("trusted class pairs must be ordered COCO IDs")

    try:
        pack_observation_blocks([result], task="detection")
    except ValueError as error:
        return _rejection(source, f"invalid assembled observation: {error}")

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
        lower = max(0.0, source.observed_seconds - 8.0)
        inside = observed[valid] >= lower if source.observed_seconds < 8.0 else observed[valid] > lower
        if not inside.all() or (observed[valid] > source.observed_seconds).any():
            raise TeacherRecordError("assembled observation lies outside its development detection window")
        if (_require_bool(relation.background_valid, "background_valid")
                and np.isfinite(current_background).all() and np.isfinite(current_composition).all()
                and (current_composition >= 0).all() and current_composition.sum() > 0):
            static_anchors.append((pair_id, current_background, current_composition))
        pairs.append({"pair_id": pair_id, "class_pair": list(relation_class_pairs[position]),
                      "initial_geometry": np.asarray(relation.initial_geometry, dtype=np.float64).copy(),
                      "candidate_pair_count": len(result.relations), "valid_cells": valid.copy(),
                      "process_cells": process.copy()})
        relation_ids.append(pair_id)
    if not pairs:
        return _rejection(source, "no relation pairs in assembled observation")
    if not static_anchors:
        return _rejection(source, "missing reliable assembled background")
    anchors = sorted(static_anchors, key=lambda item: item[0])
    background = np.mean(np.stack([item[1] for item in anchors]), axis=0)
    composition = np.mean(np.stack([item[2] for item in anchors]), axis=0)
    composition /= composition.sum()
    record = {"dataset": source.dataset,
              "window_id": detection_window_id(source.dataset, source.key, source.query_index),
              "source_family": source.family, "content_alias": source.content_alias, "fold": source.fold,
              "normal_permitted": False, "background": background.copy(), "background_valid": True,
              "class_composition": composition.copy(), "pairs": pairs}
    return CompactTeacherWindow(record=record, relation_ids=tuple(relation_ids))
