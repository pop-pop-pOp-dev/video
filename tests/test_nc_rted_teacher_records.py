import numpy as np

from nc_rted.features import (CELL_FEATURE_DIM, PROCESS_FEATURE_DIM, FeatureAssemblyResult,
                              FeatureStatus, RelationWindowFeatures)
from nc_rted.teacher_records import (TeacherSourceTruth, build_teachers_from_compact,
                                     feature_assembly_to_teacher_window)


def truth():
    return TeacherSourceTruth("ucf-crime", "Abuse001_x264", 7, "ucf:Abuse001", "ucf:Abuse001", 2, "train", True, 7.0)


def relation(*, background_valid=True, first_track=3, second_track=4, background=None, composition=None, times=None):
    mask = np.array([True, True, False, False])
    process = np.full((4, PROCESS_FEATURE_DIM), np.nan)
    process[:2] = 0.0
    student = np.full((4, CELL_FEATURE_DIM), np.nan)
    student[:2] = 0.0
    return RelationWindowFeatures(
        first_track=first_track, second_track=second_track, cell_mask=mask, feature_valid=mask.copy(),
        observed_times_s=np.array([1., 2., np.nan, np.nan]) if times is None else times,
        cell_right_boundaries_s=np.array([2., 4., 6., 8.]),
        student_cells=student, process_cells=process,
        static_background=np.r_[1., np.zeros(1151)] if background is None else background,
        background_valid=background_valid,
        static_class_composition=np.r_[1., np.zeros(79)] if composition is None else composition,
        initial_geometry=np.zeros(5),
    )


def test_adapts_real_feature_result_without_changing_relation_order():
    result = FeatureAssemblyResult(FeatureStatus.OK, (relation(),))
    adapted = feature_assembly_to_teacher_window(result, truth(), ((0, 2),))
    assert adapted.relation_ids == ("3:4",)
    assert adapted.record["window_id"] == "detection:ucf-crime:Abuse001_x264:7"
    assert adapted.record["pairs"][0]["class_pair"] == [0, 2]
    assert adapted.record["pairs"][0]["valid_cells"].tolist() == [True, True, False, False]


def test_technical_or_missing_background_result_becomes_rejection():
    technical = FeatureAssemblyResult(FeatureStatus.TRACKING_FAILURE, (), "tracker failed")
    assert feature_assembly_to_teacher_window(technical, truth(), ()).record["aux_valid"] is False
    missing = FeatureAssemblyResult(FeatureStatus.OK, (relation(background_valid=False),))
    assert feature_assembly_to_teacher_window(missing, truth(), ((0, 2),)).record["aux_valid"] is False


def test_different_relation_anchor_contexts_are_aggregated_deterministically():
    first = relation(first_track=9, second_track=10, background=np.r_[2., np.zeros(1151)],
                     composition=np.r_[0., 1., np.zeros(78)], times=np.array([4., 6., np.nan, np.nan]))
    second = relation(first_track=3, second_track=4, background=np.r_[0., 2., np.zeros(1150)],
                      composition=np.r_[1., 0., np.zeros(78)])
    result = FeatureAssemblyResult(FeatureStatus.OK, (first, second))
    adapted = feature_assembly_to_teacher_window(result, truth(), ((1, 2), (0, 2)))
    assert adapted.relation_ids == ("9:10", "3:4")
    assert np.allclose(adapted.record["background"][:2], [1., 1.])
    assert np.allclose(adapted.record["class_composition"][:2], [.5, .5])


def test_source_truth_binds_matching_manifest_rows():
    prefix = {"dataset": "ucf-crime", "key": "Abuse001_x264", "family": "ucf:Abuse001",
              "query_index": 7, "observed_seconds": 7.0, "teacher_reference_normal_eligible": True}
    split = {"dataset": "ucf-crime", "key": "Abuse001_x264", "family": "ucf:Abuse001",
             "allocation": "train", "same_content_alias_group": "alias", "fold": 2}
    assert TeacherSourceTruth.from_manifest_rows(prefix, split).content_alias == "alias"


def test_compact_batch_keeps_mixed_and_all_rejected_windows():
    accepted = feature_assembly_to_teacher_window(FeatureAssemblyResult(FeatureStatus.OK, (relation(),)), truth(), ((0, 2),))
    rejected = feature_assembly_to_teacher_window(
        FeatureAssemblyResult(FeatureStatus.TRACKING_FAILURE, (), "failed"),
        TeacherSourceTruth("ucf-crime", "Abuse002_x264", 8, "ucf:Abuse002", "alias-2", 3, "train", True, 8.0), ())
    mixed = build_teachers_from_compact((accepted, rejected))
    assert {row["window_id"] for row in mixed["rows"]} == {accepted.record["window_id"], rejected.record["window_id"]}
    all_rejected = build_teachers_from_compact((rejected,))
    assert all_rejected["rows"] == [rejected.record]
