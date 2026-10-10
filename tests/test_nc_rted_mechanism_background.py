import numpy as np

from nc_rted.detector import CausalWindowObservation
from nc_rted.features import (CELL_FEATURE_DIM, PROCESS_FEATURE_DIM, FeatureAssemblyResult,
                               FeatureStatus, RelationWindowFeatures, STUDENT_BLOCK_SLICES)
from nc_rted.mechanism_background import _compatible_donor, _replace_global


def _observation(global_value, valid=None):
    valid = np.asarray([True, False, False, False] if valid is None else valid)
    student = np.full((4, CELL_FEATURE_DIM), np.nan, dtype=np.float32)
    for index in np.flatnonzero(valid):
        student[index] = np.arange(CELL_FEATURE_DIM, dtype=np.float32)
        student[index, STUDENT_BLOCK_SLICES["global"]] = global_value
    relation = RelationWindowFeatures(0, 1, valid, valid.copy(), np.asarray([1., np.nan, np.nan, np.nan]),
                                      np.asarray([2., 4., 6., 8.]), student,
                                      np.full((4, PROCESS_FEATURE_DIM), np.nan, dtype=np.float32),
                                      np.ones(1152, dtype=np.float32), True, np.r_[1., np.zeros(79)], np.zeros(5))
    return CausalWindowObservation(FeatureAssemblyResult(FeatureStatus.OK, (relation,)), (), ())


def test_global_context_replacement_changes_only_verified_student_global_block():
    original, donor = _observation(3.), _observation(7.)
    zeroed = _replace_global(original, donor=None)
    permuted = _replace_global(original, donor=donor)
    global_slice = STUDENT_BLOCK_SLICES["global"]
    original_values = original.features.relations[0].student_cells
    for altered, value in ((zeroed, 0.), (permuted, 7.)):
        changed = altered.features.relations[0].student_cells
        np.testing.assert_allclose(changed[0, global_slice], value)
        unchanged = np.r_[np.arange(global_slice.start), np.arange(global_slice.stop, CELL_FEATURE_DIM)]
        np.testing.assert_allclose(changed[0, unchanged], original_values[0, unchanged])
        assert np.array_equal(altered.features.relations[0].feature_valid, original.features.relations[0].feature_valid)
        np.testing.assert_allclose(altered.features.relations[0].process_cells, original.features.relations[0].process_cells,
                                   equal_nan=True)
        np.testing.assert_allclose(altered.features.relations[0].static_background,
                                   original.features.relations[0].static_background)


def test_global_context_donor_requires_matching_valid_layout():
    assert not _compatible_donor(_observation(3.), _observation(7., [False, True, False, False]))
