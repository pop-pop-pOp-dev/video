import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nc_rted.teacher import TeacherError, calibrated_teacher, strength_matched_quality, validate_retrieval_support

def test_calibrated_teacher_masks_invalid_cells_and_builds_joint_distribution():
    refs = np.array([[.1, 99, .4], [.2, 99, .5], [.3, 99, .6]])
    quality, positions, joint = calibrated_teacher(refs, np.zeros(64), np.array([True, False, True]))
    assert np.isclose(quality, (0.05 - 1 / 65) / .05)
    assert positions[1] == 0 and np.isclose(joint.sum(), 1) and np.argmax(positions) == 2

def test_strength_matched_control_uses_global_m_rank_and_stable_ties():
    result = strength_matched_quality(np.array([.2, .8, .5]), np.array([2., 1., 1.]), ["c", "b", "a"])
    assert np.array_equal(result, np.array([.8, .5, .2]))

def test_teacher_rejects_missing_support():
    with pytest.raises(TeacherError):
        calibrated_teacher(np.zeros((3, 2)), np.zeros(63), np.ones(2, dtype=bool))
    with pytest.raises(TeacherError):
        calibrated_teacher(np.zeros((3, 2)), np.zeros(64), np.zeros(2, dtype=bool))


def test_retrieval_support_checks_source_family_constraints():
    calibration = [f"family-{i}" for i in range(16) for _ in range(4)]
    validate_retrieval_support(["r0", "r1", "r2"], calibration, "target")
    with pytest.raises(TeacherError):
        validate_retrieval_support(["r0", "r0", "r2"], calibration, "target")
    with pytest.raises(TeacherError):
        validate_retrieval_support(["r0", "r1", "r2"], ["r0"] * 4 + calibration[4:], "target")

def test_teacher_rejects_impossible_calibration_and_u_domains():
    with pytest.raises(TeacherError): calibrated_teacher(np.zeros((3,1)), -np.ones(64), np.ones(1,dtype=bool))
    with pytest.raises(TeacherError): strength_matched_quality(np.array([1.1]),np.array([1.]),['a'])
