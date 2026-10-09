import numpy as np
import pytest

from nc_rted.alignment import NormalScale, constrained_alignment, fit_reference_scale, process_cost_matrix


def test_signed_process_detects_reversal_and_block_dimension_is_not_weight():
    positive = np.arange(4.)[:, None]
    negative = -positive
    q = {"geometry": positive, "visual": np.repeat(positive, 100, axis=1)}
    r = {"geometry": negative, "visual": np.repeat(negative, 100, axis=1)}
    scales = {k: NormalScale(np.zeros(v.shape[1]), np.ones(v.shape[1])) for k, v in q.items()}
    distance = process_cost_matrix(q, r, scales, np.ones(4, bool), np.ones(4, bool))
    np.testing.assert_allclose(np.diag(distance), [0, 2, 4, 6])


def test_scale_uses_robust_reference_variability_and_fixed_floor():
    scale = fit_reference_scale(np.array([[0., 7.], [1., 7.], [2., 7.], [100., 7.]]))
    np.testing.assert_allclose(scale.center, [1.5, 7])
    np.testing.assert_allclose(scale.scale, [1.4826, 1e-3])
    with pytest.raises(ValueError):
        fit_reference_scale(np.array([[np.nan]]))


def test_dtw_ties_use_shortest_monotonic_path_with_exact_endpoints():
    result = constrained_alignment(np.zeros((4, 4)), np.ones(4, bool), np.ones(4, bool))
    assert result.valid
    assert result.path == ((0, 0), (1, 1), (2, 2), (3, 3))
    np.testing.assert_array_equal(result.cell_distance, np.zeros(4))


def test_missing_cells_remain_missing_and_impossible_warp_is_rejected():
    q = np.array([True, False, True, True])
    r = np.array([True, True, False, True])
    result = constrained_alignment(np.ones((4, 4)), q, r)
    assert result.valid and np.isnan(result.cell_distance[1])
    assert not result.cell_valid[1]
    assert all(i != 1 and j != 2 for i, j in result.path)
    impossible = constrained_alignment(np.ones((4, 4)), np.array([True, False, False, False]), np.ones(4, bool))
    assert not impossible.valid and impossible.reason == "no_legal_monotonic_path"


def test_invalid_descriptor_nan_is_not_zero_filled_or_allowed_into_cost():
    q = {"geometry": np.array([[0.], [np.nan], [2.], [3.]])}
    r = {"geometry": np.arange(4.)[:, None]}
    scales = {"geometry": NormalScale(np.zeros(1), np.ones(1))}
    result = process_cost_matrix(q, r, scales, np.array([True, False, True, True]), np.ones(4, bool))
    assert np.isinf(result[1]).all()
    assert np.isfinite(result[[0, 2, 3]]).all()
    with pytest.raises(ValueError):
        process_cost_matrix(q, r, scales, np.ones(4, bool), np.ones(4, bool))
