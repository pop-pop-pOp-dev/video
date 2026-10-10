import numpy as np
import pytest

from nc_rted.development_teacher import summarize_development_teacher_targets
from nc_rted.features import PROCESS_FEATURE_DIM
from nc_rted.teacher_records import CompactTeacherWindow


def compact(name, fold, *, normal=True, valid=(True, True, False, False), family=None):
    process = np.full((4, PROCESS_FEATURE_DIM), np.nan); process[np.asarray(valid)] = 0.
    return CompactTeacherWindow({"dataset": "dev", "window_id": name, "source_family": family or name,
        "content_alias": name + "-alias", "fold": fold, "normal_permitted": normal,
        "background": np.r_[1., np.zeros(1151)], "background_valid": True,
        "class_composition": np.r_[1., np.zeros(79)], "pairs": [{"pair_id": name + "-pair", "class_pair": [0, 2],
        "initial_geometry": np.zeros(5), "candidate_pair_count": 1, "valid_cells": list(valid), "process_cells": process}]}, (name + "-pair",))


def training(*, calibration=64):
    rows = [compact("r2", 2), compact("r3", 3), compact("r4", 4)]
    rows.extend(compact(f"c{i}", 1, family=f"cal-{i // 4}") for i in range(calibration))
    return tuple(rows)


def test_development_targets_use_training_only_r_and_c_with_real_pipeline_masks():
    target = compact("dev-target", 0, normal=False)
    result = summarize_development_teacher_targets(development_targets=(target,), training_compacts=training())
    assert [row["window_id"] for row in result["rows"]] == ["dev-target"]
    row = result["rows"][0]
    assert row["aux_valid"] and row["mask"][:4] == [True, True, False, False]
    assert row["calibration_count"] == 64 and row["static_support"]["r_normal_candidate_count"] == 3


def test_development_target_never_becomes_reference_or_calibration_and_unsupported_coverage_is_explicit():
    target = compact("dev-target", 0, normal=True)
    result = summarize_development_teacher_targets(development_targets=(target,), training_compacts=training(calibration=4))
    row = result["rows"][0]
    assert not row["aux_valid"] and "64..128" in row["rejection"]
    assert row["window_id"] == "dev-target"
    assert row["relation_support"]["supported_pair_count"] == 1
    assert row["calibration_support"] == {}


def test_development_target_sharing_a_training_reference_family_is_not_reused():
    target = compact("dev-target", 0, family="r2")
    row = summarize_development_teacher_targets(development_targets=(target,), training_compacts=training())["rows"][0]
    assert not row["aux_valid"] and "three threshold-compatible" in row["rejection"]


def test_all_rejected_development_targets_are_emitted_without_mutating_compacts():
    target = CompactTeacherWindow({"window_id": "rejected", "dataset": "dev", "aux_valid": False, "rejection": "missing"}, (), {"reason": "missing"})
    result = summarize_development_teacher_targets(development_targets=(target,), training_compacts=training())
    assert result["rows"][0]["F_quality"] is None
    assert "F_quality" not in target.record


def test_duplicate_development_ids_are_rejected_across_accepted_and_rejected_rows():
    accepted = compact("same", 0)
    rejected = CompactTeacherWindow({"window_id": "same", "dataset": "dev", "aux_valid": False, "rejection": "missing"}, (), {"reason": "missing"})
    with pytest.raises(ValueError, match="unique"):
        summarize_development_teacher_targets(development_targets=(accepted, rejected), training_compacts=training())
