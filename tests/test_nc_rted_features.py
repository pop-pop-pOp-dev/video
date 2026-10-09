import numpy as np
import torch

from nc_rted.features import (
    CELL_FEATURE_DIM, PROCESS_FEATURE_DIM, FeatureStatus, FrozenFrameFeatures,
    PROCESS_BLOCK_SLICES, SIGLIP_FEATURE_DIM, STUDENT_BLOCK_SLICES,
    assemble_relation_features,
)
from nc_rted.tracking import Detection, FrameObservations, causal_tracks


def det(box, cls=0, appearance=(1.0, 0.0)):
    return Detection(box, cls, 0.9, appearance)


def patches(value):
    return torch.full((729, SIGLIP_FEATURE_DIM), value, dtype=torch.float32)


def test_feature_schema_keeps_missing_cells_masked_and_static_separate():
    tracked = causal_tracks([
        FrameObservations(2.5, (det((.1, .1, .3, .3)), det((.5, .1, .7, .3), cls=2))),
        FrameObservations(3.5, (det((.11, .1, .31, .3)), det((.5, .1, .7, .3), cls=2))),
        FrameObservations(4.5, (det((.12, .1, .32, .3)), det((.5, .1, .7, .3), cls=2))),
    ])
    output = assemble_relation_features(tracked, (
        FrozenFrameFeatures(2.5, patches(1)), FrozenFrameFeatures(3.5, patches(2)),
        FrozenFrameFeatures(4.5, patches(3)),
    ), 10.0)
    assert output.status is FeatureStatus.OK
    relation = output.relations[0]
    assert relation.student_cells.shape == (4, CELL_FEATURE_DIM)
    assert relation.process_cells.shape == (4, PROCESS_FEATURE_DIM)
    assert relation.cell_mask.tolist() == [True, True, False, False]
    assert relation.feature_valid.tolist() == [True, True, False, False]
    np.testing.assert_allclose(relation.observed_times_s[:2], [3.5, 4.5])
    assert np.isnan(relation.observed_times_s[2:]).all()
    np.testing.assert_allclose(relation.cell_right_boundaries_s, [4., 6., 8., 10.])
    assert np.isnan(relation.student_cells[2]).all()
    assert np.isnan(relation.process_cells[2]).all()
    assert relation.static_class_composition.shape == (80,)
    assert relation.initial_geometry.shape == (5,)
    assert relation.static_background.shape == (1152,)


def test_relation_observed_before_final_frame_is_retained_without_interpolation():
    tracked = causal_tracks([
        FrameObservations(4.0, (det((.1, .1, .3, .3)), det((.5, .1, .7, .3), cls=2))),
        FrameObservations(5.0, (det((.11, .1, .31, .3)),)),
    ])
    output = assemble_relation_features(tracked, (FrozenFrameFeatures(4.0, patches(1)), FrozenFrameFeatures(5.0, patches(2))), 10.0)
    assert output.status is FeatureStatus.OK
    assert len(output.relations) == 1
    assert output.relations[0].cell_mask.tolist() == [False, True, False, False]
    assert np.isnan(output.relations[0].student_cells[[0, 2, 3]]).all()


def test_invalid_siglip_dimension_and_future_frames_fail_explicitly():
    tracked = causal_tracks([FrameObservations(5.0, (det((.1, .1, .3, .3)), det((.5, .1, .7, .3), cls=2)))])
    bad_dimension = assemble_relation_features(tracked, (FrozenFrameFeatures(5.0, torch.zeros(729, 8)),), 10.0)
    assert bad_dimension.status is FeatureStatus.INVALID_INPUT
    future = assemble_relation_features(tracked, (FrozenFrameFeatures(10.1, patches(1)),), 10.0)
    assert future.status is FeatureStatus.INVALID_INPUT


def test_no_relation_pair_is_distinct_from_tracking_failure():
    tracked = causal_tracks([FrameObservations(5.0, (det((.1, .1, .3, .3)),))])
    output = assemble_relation_features(tracked, (FrozenFrameFeatures(5.0, patches(1)),), 10.0)
    assert output.status is FeatureStatus.NO_RELATION_PAIRS


def test_final_caption_block_uses_explicit_nonoverlapping_window_start():
    tracked = causal_tracks([FrameObservations(17.0, (det((.1, .1, .3, .3)), det((.5, .1, .7, .3), cls=2)))])
    legal = assemble_relation_features(tracked, (FrozenFrameFeatures(17.0, patches(1)),), 17.25, window_start_s=16.0)
    assert legal.status is FeatureStatus.OK
    assert legal.relations[0].cell_mask.tolist() == [True, False, False, False]
    np.testing.assert_allclose(legal.relations[0].cell_right_boundaries_s, [17.25, 17.25, 17.25, 17.25])
    stale = causal_tracks([FrameObservations(10.0, (det((.1, .1, .3, .3)), det((.5, .1, .7, .3), cls=2)))])
    rejected = assemble_relation_features(stale, (FrozenFrameFeatures(10.0, patches(1)),), 17.25, window_start_s=16.0)
    assert rejected.status is FeatureStatus.INVALID_INPUT


def _moving_relation(second_time, shift=0.0):
    tracked = causal_tracks([
        FrameObservations(2.2, (
            det((.1 + shift, .1, .3 + shift, .3)), det((.5 + shift, .1, .7 + shift, .3), cls=2),
        )),
        FrameObservations(second_time, (
            det((.1 + shift, .1, .3 + shift, .3)), det((.6 + shift, .1, .8 + shift, .3), cls=2),
        )),
    ])
    return assemble_relation_features(tracked, (
        FrozenFrameFeatures(2.2, patches(1)), FrozenFrameFeatures(second_time, patches(2)),
    ), 10.0).relations[0]


def test_velocity_uses_actual_elapsed_time_and_is_global_drift_invariant():
    fast = _moving_relation(2.7)
    slow = _moving_relation(3.2)
    shifted = _moving_relation(2.7, shift=.1)
    velocity = PROCESS_BLOCK_SLICES["relative_velocity"]
    np.testing.assert_allclose(fast.process_cells[0, velocity], [.2, 0.], atol=1e-6)
    np.testing.assert_allclose(slow.process_cells[0, velocity], [.1, 0.], atol=1e-6)
    np.testing.assert_allclose(fast.process_cells[0, velocity], shifted.process_cells[0, velocity], atol=1e-6)
    np.testing.assert_allclose(fast.student_cells[0, STUDENT_BLOCK_SLICES["relative_velocity"]], [.2, 0.], atol=1e-6)


def test_anchor_is_earliest_shared_observation_not_last_sample_of_its_cell():
    relation = _moving_relation(2.8)
    initial = PROCESS_BLOCK_SLICES["initial_geometry"]
    current = PROCESS_BLOCK_SLICES["current_geometry"]
    change = PROCESS_BLOCK_SLICES["geometry_change"]
    np.testing.assert_allclose(relation.process_cells[0, initial], [.4, 0., 0., 0., 0.], atol=1e-6)
    np.testing.assert_allclose(relation.process_cells[0, current], [.5, 0., 0., 0., 0.], atol=1e-6)
    np.testing.assert_allclose(relation.process_cells[0, change], [.1, 0., 0., 0., 0.], atol=1e-6)
    np.testing.assert_allclose(relation.initial_geometry, relation.process_cells[0, initial])
    np.testing.assert_allclose(relation.static_background, np.ones(SIGLIP_FEATURE_DIM))
    np.testing.assert_allclose(relation.observed_times_s[0], 2.8)
    np.testing.assert_allclose(relation.cell_right_boundaries_s[0], 4.0)
