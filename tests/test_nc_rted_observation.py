import numpy as np
import pytest
import torch

from nc_rted.observation import (
    background_feature, causal_frame_indices, patch_overlap_weights,
    pool_patch_regions, relative_geometry, temporal_cells,
)


def test_sampling_uses_actual_past_frames_and_is_future_invariant():
    pts = np.arange(0.0, 10.1, 0.1)
    expected = causal_frame_indices(pts, 10.0)
    extended = np.concatenate((pts, np.arange(10.2, 20.0, 0.1)))
    assert np.array_equal(expected, causal_frame_indices(extended, 10.0))
    assert len(expected) == 16
    assert (pts[expected] > 2).all() and (pts[expected] <= 10).all()
    low_fps = np.array([0.0, 1.0, 2.0])
    assert causal_frame_indices(low_fps, 2.0).tolist() == [0, 1, 2]
    assert causal_frame_indices(np.array([]), 0).size == 0


def test_four_time_cells_keep_missing_bins_and_reject_future():
    assert temporal_cells(np.array([2.5, 4, 6, 8, 10]), 10).tolist() == [0, 1, 2, 3, 3]
    with pytest.raises(ValueError):
        temporal_cells(np.array([10.01]), 10)
    with pytest.raises(ValueError):
        temporal_cells(np.array([2.0]), 10)


def test_roi_uses_real_patch_footprint_and_correct_row_major_axis():
    patches = torch.arange(729, dtype=torch.float32).unsqueeze(-1)
    boxes = torch.tensor([[14/384, 28/384, 28/384, 42/384],
                          [378/384, 0, 1, 1], [0, 0, 1, 1]])
    pooled, valid = pool_patch_regions(patches, boxes)
    assert valid.tolist() == [True, False, True]
    torch.testing.assert_close(pooled[:, 0], torch.tensor([55., 0., 364.]))
    area, _ = patch_overlap_weights(boxes[:1])
    assert (area > 0).sum().item() == 1
    torch.testing.assert_close(area.sum(), torch.tensor(14.0 * 14.0))


def test_partial_roi_area_weighting_and_invalid_box_masks():
    patches = torch.zeros(729, 2, dtype=torch.bfloat16)
    patches[0] = 2
    patches[1] = 6
    boxes = torch.tensor([[7/384, 0, 28/384, 14/384],
                          [float('nan'), 0, 1, 1], [0.3, 0, 0.2, 1]])
    output, valid = pool_patch_regions(patches, boxes)
    assert output.dtype == patches.dtype
    assert valid.tolist() == [True, False, False]
    torch.testing.assert_close(output[0].float(), torch.full((2,), 14/3), atol=0.02, rtol=0)
    assert torch.isfinite(output).all()


def test_background_failure_is_explicit_and_not_a_normal_label():
    patches = torch.ones(729, 2)
    feature, valid = background_feature(patches, torch.empty(0, 4))
    assert valid and torch.equal(feature, torch.ones(2))
    feature, valid = background_feature(patches, torch.tensor([[0., 0., 1., 1.]]))
    assert not valid and torch.equal(feature, torch.zeros(2))


def test_signed_relative_geometry_translation_and_direction():
    a = torch.tensor([0.1, 0.2, 0.3, 0.4])
    b = torch.tensor([0.3, 0.3, 0.6, 0.5])
    before = relative_geometry(a, b)
    torch.testing.assert_close(before, relative_geometry(a + 0.1, b + 0.1))
    after = relative_geometry(b, a)
    torch.testing.assert_close(before[:4], -after[:4])
    torch.testing.assert_close(before[4], after[4])
