import numpy as np
import pytest
from pathlib import Path

from nc_rted.development_teacher_targets import development_feature_assembly_to_teacher_window
from nc_rted.features import (CELL_FEATURE_DIM, PROCESS_FEATURE_DIM, FeatureAssemblyResult,
                              FeatureStatus, RelationWindowFeatures)
from nc_rted.teacher_pipeline import parse_frozen_records
from nc_rted.teacher_records import TeacherRecordError


def development_prefix(*, query_index=7, observed_seconds=7.0):
    return {"sample_id": f"development:ucf-crime:Abuse001_x264:{query_index}", "dataset": "ucf-crime",
            "key": "Abuse001_x264", "family": "ucf:Abuse001", "query_index": query_index,
            "observed_seconds": observed_seconds, "class": "normal", "scope": "vad_causal_latest_8s"}


def development_split(*, allocation="development", fold=3):
    return {"dataset": "ucf-crime", "key": "Abuse001_x264", "family": "ucf:Abuse001",
            "allocation": allocation, "same_content_alias_group": "ucf:Abuse001-alias", "fold": fold}


def relation(*, background_valid=True):
    mask = np.array([True, True, False, False])
    process = np.full((4, PROCESS_FEATURE_DIM), np.nan)
    process[:2] = 0.0
    student = np.full((4, CELL_FEATURE_DIM), np.nan)
    student[:2] = 0.0
    return RelationWindowFeatures(
        first_track=3, second_track=4, cell_mask=mask, feature_valid=mask.copy(),
        observed_times_s=np.array([1., 2., np.nan, np.nan]), cell_right_boundaries_s=np.array([2., 4., 6., 8.]),
        student_cells=student, process_cells=process, static_background=np.r_[1., np.zeros(1151)],
        background_valid=background_valid, static_class_composition=np.r_[1., np.zeros(79)],
        initial_geometry=np.zeros(5),
    )


def test_development_converter_preserves_sealed_fold_and_parser_shape():
    compact = development_feature_assembly_to_teacher_window(
        FeatureAssemblyResult(FeatureStatus.OK, (relation(),)), ((0, 2),),
        development_prefix=development_prefix(), development_source_split=development_split())
    assert set(compact.record) == {"dataset", "window_id", "source_family", "content_alias", "fold",
                                   "normal_permitted", "background", "background_valid", "class_composition", "pairs"}
    assert compact.record["fold"] == 3
    assert compact.record["normal_permitted"] is False
    parsed = parse_frozen_records({"schema": "nc_rted_frozen_feature_records/v1", "split": "train",
                                   "windows": [compact.record]})
    assert parsed[0].window_id == "detection:ucf-crime:Abuse001_x264:7"


@pytest.mark.parametrize("result, pairs", [
    (FeatureAssemblyResult(FeatureStatus.TRACKING_FAILURE, (), "tracker failed"), ()),
    (FeatureAssemblyResult(FeatureStatus.NO_RELATION_PAIRS, ()), ()),
    (FeatureAssemblyResult(FeatureStatus.OK, (relation(background_valid=False),)), ((0, 2),)),
])
def test_development_failures_emit_explicit_window_rejections(result, pairs):
    compact = development_feature_assembly_to_teacher_window(
        result, pairs, development_prefix=development_prefix(), development_source_split=development_split())
    assert compact.rejection is not None
    assert compact.record["aux_valid"] is False
    assert compact.record["window_id"] == "detection:ucf-crime:Abuse001_x264:7"
    assert compact.record["dataset"] == "ucf-crime"


def test_development_converter_rejects_nondevelopment_allocation_and_unsealed_fold():
    with pytest.raises(TeacherRecordError, match="allocated to development"):
        development_feature_assembly_to_teacher_window(
            FeatureAssemblyResult(FeatureStatus.OK, (relation(),)), ((0, 2),),
            development_prefix=development_prefix(), development_source_split=development_split(allocation="train"))
    with pytest.raises(TeacherRecordError, match="invalid sealed development prefix time or fold"):
        development_feature_assembly_to_teacher_window(
            FeatureAssemblyResult(FeatureStatus.OK, (relation(),)), ((0, 2),),
            development_prefix=development_prefix(), development_source_split=development_split(fold=5))


def test_development_converter_never_constructs_training_source_truth():
    source = Path("src/nc_rted/development_teacher_targets.py").read_text(encoding="utf-8")
    assert "TeacherSourceTruth" not in source
    assert '"allocation": "train"' not in source
