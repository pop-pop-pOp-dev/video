import numpy as np
import pytest

from nc_rted.retrieval import (
    RetrievedPair, RetrievalStatus, StaticPair, StaticWindow, r_leave_one_source_threshold,
    assert_role_isolation, select_calibration, select_references, static_descriptor, static_distance,
)


def window(name, family, alias, geometry=(.2, 0, 0, 0, 0), *, normal=True, background_valid=True, count=1, cells=4):
    return StaticWindow(
        name, family, alias, normal, np.r_[1., np.zeros(1151)], background_valid,
        np.r_[.5, .5, np.zeros(78)], (StaticPair(name + "-pair", (0, 2), np.array(geometry, dtype=float), count, cells),),
    )


def test_reference_selection_is_static_normal_and_provenance_isolated():
    query = window("q", "q-family", "q-alias")
    references = (
        window("bad-normal", "a", "a", normal=False), window("alias", "b", "q-alias"),
        window("r3", "r3", "a3"), window("r1", "r1", "a1"), window("r2", "r2", "a2"),
    )
    result = select_references(query, query.pairs[0], references, compatibility_threshold=1.0)
    assert result.status is RetrievalStatus.OK
    reversed_result = select_references(query, query.pairs[0], tuple(reversed(references)), compatibility_threshold=1.0)
    assert [(item.window.window_id, item.pair.pair_id) for item in result.selected] == [
        (item.window.window_id, item.pair.pair_id) for item in reversed_result.selected
    ]
    assert {item.window.source_family for item in result.selected} == {"r1", "r2", "r3"}
    assert len({item.window.content_alias for item in result.selected}) == 3


def test_missing_background_and_incompatible_orientation_refuse_retrieval():
    query = window("q", "q", "q", background_valid=False)
    assert select_references(query, query.pairs[0], (window("r", "r", "r"),), 1.0).status is RetrievalStatus.MISSING_BACKGROUND
    query = window("q", "q", "q")
    reverse = StaticWindow("reverse", "r", "r", True, np.r_[1., np.zeros(1151)], True, np.r_[.5, .5, np.zeros(78)],
                           (StaticPair("reverse-pair", (2, 0), np.array([.2, 0, 0, 0, 0.]), 1, 4),))
    assert select_references(query, query.pairs[0], (reverse,), 1.0).status is RetrievalStatus.INSUFFICIENT_REFERENCES


def test_calibration_enforces_qcr_disjoint_support_limits():
    query = window("q", "q", "q")
    refs = select_references(query, query.pairs[0], tuple(window(f"r{i}", f"r{i}", f"ra{i}") for i in range(3)), 1.0).selected
    candidates = tuple(window(f"c{family}-{sample}", f"c{family}", f"ca{family}") for family in range(16) for sample in range(4))
    calibration = select_calibration(query, query.pairs[0], refs, candidates, 1.0)
    assert calibration.status is RetrievalStatus.OK
    assert len(calibration.selected) == 64
    assert len({item.window.source_family for item in calibration.selected}) == 16
    assert max(sum(item.window.source_family == family for item in calibration.selected) for family in {item.window.source_family for item in calibration.selected}) == 4
    overlap = select_calibration(query, query.pairs[0], refs, candidates + (window("bad", "r0", "x"),), 1.0)
    assert all(item.window.source_family not in {"q", "r0", "r1", "r2"} for item in overlap.selected)


def test_static_distance_and_r_threshold_are_static_only_and_fixed_rank():
    windows = tuple(window(f"r{i}", f"f{i}", f"a{i}", geometry=(.2 + i * .01, 0, 0, 0, 0)) for i in range(4))
    records = tuple((item, item.pairs[0]) for item in windows)
    threshold = r_leave_one_source_threshold(records)
    assert threshold > 0
    first, second = static_descriptor(records[0][0], records[0][1]), static_descriptor(records[1][0], records[1][1])
    assert static_distance(first, second) == static_distance(second, first)


def test_threshold_gates_reference_and_calibration_candidates_and_skips_bad_candidates():
    query = window("q", "q", "q")
    bad = window("bad", "bad", "bad", background_valid=False)
    distant = window("far", "far", "far", geometry=(.4, 0, 0, 0, 0))
    result = select_references(query, query.pairs[0], (bad, distant), compatibility_threshold=.001)
    assert result.status is RetrievalStatus.INSUFFICIENT_REFERENCES
    assert "skipped_invalid=1" in result.reason
    refs = tuple(select_references(query, query.pairs[0], tuple(window(f"r{i}", f"r{i}", f"a{i}") for i in range(3)), 1.0).selected)
    calibration = select_calibration(query, query.pairs[0], refs, tuple(window(f"c{i}", f"c{i}", f"ca{i}", geometry=(.4, 0, 0, 0, 0)) for i in range(64)), .001)
    assert calibration.status is RetrievalStatus.INSUFFICIENT_CALIBRATION


def test_role_isolation_and_forged_reference_leaks_fail_closed():
    query = window("q", "q", "qa")
    reference = window("r", "r", "ra")
    calibration = window("c", "c", "ca")
    assert_role_isolation((query,), (calibration,), (reference,))
    try:
        assert_role_isolation((query,), (window("bad", "q", "other"),), (reference,))
        assert False, "expected Q/C family overlap failure"
    except ValueError:
        pass
    good = select_references(query, query.pairs[0], tuple(window(f"r{i}", f"r{i}", f"a{i}") for i in range(3)), 1.0).selected
    leak = window("leak", "q", "leak")
    forged = select_calibration(query, query.pairs[0], (
        good[0], good[1], RetrievedPair(leak, leak.pairs[0], 0.0),
    ), (), 1.0)
    assert forged.status is RetrievalStatus.QCR_ISOLATION_FAILURE


def test_unsupported_r_record_does_not_silently_drop_from_threshold_fit():
    solo = window("solo", "solo-family", "solo-alias")
    same_source = window("same", "solo-family", "other-alias")
    with pytest.raises(ValueError, match="unsupported R record"):
        r_leave_one_source_threshold(((solo, solo.pairs[0]), (same_source, same_source.pairs[0])))
