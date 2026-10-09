from dataclasses import dataclass

import numpy as np
import pytest

from nc_rted.alignment import NormalScale
from nc_rted.teacher_cache import (BoundedTeacherCache, normal_scale_key,
                                   ordered_reference_key, pair_payload_key,
                                   single_reference_alignment_key, window_payload_key,
                                   window_teacher_distances_key)


@dataclass(frozen=True)
class Pair:
    pair_id: str
    class_pair: tuple[int, int]
    initial_geometry: np.ndarray
    candidate_pair_count: int
    valid_cells: np.ndarray
    process_cells: np.ndarray


@dataclass(frozen=True)
class Window:
    dataset: str
    window_id: str
    source_family: str
    content_alias: str
    fold: int
    normal_permitted: bool
    background: np.ndarray
    background_valid: bool
    class_composition: np.ndarray
    pairs: tuple[Pair, ...]


def make_window(name: str, *, process_offset: float = 0.0, mask=None) -> Window:
    cells = np.arange(8, dtype=np.float64).reshape(4, 2) + process_offset
    valid = np.array([True, True, False, True] if mask is None else mask, dtype=bool)
    cells[~valid] = np.nan
    pair = Pair(name + "-pair", (0, 2), np.array([.2, .1, .3, .4, .5]), 2, valid, cells)
    return Window("ucf-crime", name, name + "-family", name + "-alias", 2, True,
                  np.array([1., 2.], dtype=np.float32), True,
                  np.array([.5, .5], dtype=np.float64), (pair,))


def test_payload_keys_bind_bytes_dtype_shape_masks_and_actual_scale_values():
    original = make_window("same-id")
    changed_cells = make_window("same-id", process_offset=1)
    changed_mask = make_window("same-id", mask=[True, False, False, True])
    changed_dtype = Window(**{**original.__dict__, "background": original.background.astype(np.float64)})
    assert len({window_payload_key(item) for item in (original, changed_cells, changed_mask, changed_dtype)}) == 4

    original_key = window_payload_key(original)
    pair_key = pair_payload_key(original_key, 0, original.pairs[0])
    other_pair_key = pair_payload_key(window_payload_key(changed_cells), 0, changed_cells.pairs[0])
    r_keys = ("a" * 64, "b" * 64, "c" * 64)
    first = normal_scale_key(2, r_keys, {"geometry": NormalScale(np.zeros(2), np.ones(2))})
    second = normal_scale_key(2, r_keys, {"geometry": NormalScale(np.zeros(2), np.array([1., 2.]))})
    assert pair_key != other_pair_key
    assert first != second


def test_equal_bytes_with_different_shape_are_distinct_and_zero_dimensional_arrays_round_trip():
    base = make_window("shape")
    reshaped_pair = Pair(base.pairs[0].pair_id, base.pairs[0].class_pair, base.pairs[0].initial_geometry,
                         base.pairs[0].candidate_pair_count, base.pairs[0].valid_cells,
                         base.pairs[0].process_cells.reshape(2, 4))
    reshaped = Window(**{**base.__dict__, "pairs": (reshaped_pair,)})
    assert window_payload_key(base) != window_payload_key(reshaped)

    cache = BoundedTeacherCache(max_bytes=10_000, max_entries=2)
    scalar = cache.get_or_compute("zero-d", lambda: np.asarray(3.0))
    assert scalar.shape == () and scalar.item() == 3.0

    with pytest.raises(TypeError, match="canonical numeric"):
        cache.get_or_compute("structured", lambda: np.asarray((1, 2), dtype=[("left", "i4"), ("right", "i4")]))

    metadata_dtype = np.dtype("f8", metadata={"mutable": bytearray(10_000)})
    with pytest.raises(TypeError, match="canonical numeric"):
        cache.get_or_compute("metadata", lambda: np.asarray([1.], dtype=metadata_dtype))


def test_window_result_key_requires_ordered_reference_triple_and_scale():
    references = ("a" * 64, "b" * 64, "c" * 64)
    reversed_references = tuple(reversed(references))
    ordered = ordered_reference_key(references)
    reversed_ordered = ordered_reference_key(reversed_references)
    scale = normal_scale_key(2, references, {"block": NormalScale(np.zeros(1), np.ones(1))})
    changed_scale = normal_scale_key(2, references, {"block": NormalScale(np.zeros(1), np.array([2.]))})
    window = "d" * 64
    assert ordered != reversed_ordered
    assert window_teacher_distances_key(scale, window, ordered) != window_teacher_distances_key(scale, window, reversed_ordered)
    assert window_teacher_distances_key(scale, window, ordered) != window_teacher_distances_key(changed_scale, window, ordered)


def test_cached_and_uncached_outcomes_match_and_repeated_dtw_work_is_saved():
    cache = BoundedTeacherCache(max_bytes=100_000, max_entries=8)
    calls = 0

    def uncached():
        nonlocal calls
        calls += 1
        return (np.array([[1., np.nan], [3., 4.]]), np.array([True, False]), {"rejected_pairs": []})

    expected = uncached()
    cached_first = cache.get_or_compute("dtw:exact", uncached)
    cached_second = cache.get_or_compute("dtw:exact", uncached)
    assert calls == 2  # one uncached reference computation and one cache miss
    np.testing.assert_equal(cached_first[0], expected[0])
    np.testing.assert_equal(cached_second[1], expected[1])
    assert cached_second[2] == expected[2]
    assert cache.stats.hits == 1 and cache.stats.misses == 1

    cached_first[0][0, 0] = 99
    cached_first[2]["rejected_pairs"].append({"pair_id": "mutated"})
    later = cache.get_or_compute("dtw:exact", uncached)
    assert later[0][0, 0] == 1
    assert later[2] == {"rejected_pairs": []}


def test_cached_builtin_value_error_replays_args_and_custom_failures_do_not_replace_the_original():
    cache = BoundedTeacherCache(max_bytes=10_000, max_entries=2)
    calls = 0

    def fail():
        nonlocal calls
        calls += 1
        raise ValueError("no legal short-DTW alignment", 7)

    with pytest.raises(ValueError) as first:
        cache.get_or_compute("failure", fail)
    with pytest.raises(ValueError) as second:
        cache.get_or_compute("failure", fail)
    assert first.value.args == second.value.args == ("no legal short-DTW alignment", 7)
    assert str(first.value) == str(second.value)
    assert calls == 1
    assert cache.stats.hits == 1 and cache.stats.misses == 1

    uncacheable_calls = 0

    def uncacheable():
        nonlocal uncacheable_calls
        uncacheable_calls += 1
        raise ValueError(lambda: "not a primitive exception argument")

    with pytest.raises(ValueError):
        cache.get_or_compute("uncacheable", uncacheable)
    with pytest.raises(ValueError):
        cache.get_or_compute("uncacheable", uncacheable)
    assert uncacheable_calls == 2

    class StatefulError(ValueError):
        payload = bytearray(100_000)

    stateful_calls = 0

    def stateful():
        nonlocal stateful_calls
        stateful_calls += 1
        raise StatefulError("custom state must not enter cache")

    with pytest.raises(StatefulError):
        cache.get_or_compute("stateful", stateful)
    with pytest.raises(StatefulError):
        cache.get_or_compute("stateful", stateful)
    assert stateful_calls == 2

    attributed_calls = 0

    def attributed():
        nonlocal attributed_calls
        attributed_calls += 1
        error = ValueError("attached state must not enter cache")
        error.context = {"attempt": attributed_calls}
        raise error

    with pytest.raises(ValueError) as first_attributed:
        cache.get_or_compute("attributed", attributed)
    with pytest.raises(ValueError) as second_attributed:
        cache.get_or_compute("attributed", attributed)
    assert attributed_calls == 2
    assert first_attributed.value.context == {"attempt": 1}
    assert second_attributed.value.context == {"attempt": 2}

    chained_calls = {"explicit": 0, "implicit": 0, "suppressed": 0}

    def chained(kind):
        chained_calls[kind] += 1
        try:
            raise RuntimeError(kind + " cause")
        except RuntimeError as cause:
            if kind == "explicit":
                raise ValueError(kind + " chain") from cause
            if kind == "suppressed":
                raise ValueError(kind + " chain") from None
            raise ValueError(kind + " chain")

    for kind in chained_calls:
        with pytest.raises(ValueError) as first_chained:
            cache.get_or_compute("chain-" + kind, lambda kind=kind: chained(kind))
        with pytest.raises(ValueError) as second_chained:
            cache.get_or_compute("chain-" + kind, lambda kind=kind: chained(kind))
        assert chained_calls[kind] == 2
        assert first_chained.value.__context__ is not None
        assert second_chained.value.__context__ is not None
        if kind == "explicit":
            assert first_chained.value.__cause__ is not None
            assert second_chained.value.__cause__ is not None
        if kind == "suppressed":
            assert first_chained.value.__suppress_context__
            assert second_chained.value.__suppress_context__


def test_lru_promotion_and_independent_entry_limit():
    cache = BoundedTeacherCache(max_bytes=100_000, max_entries=2)
    calls = {"first": 0, "second": 0, "third": 0}

    def compute(name):
        def inner():
            calls[name] += 1
            return np.arange(64, dtype=np.float64) + calls[name]
        return inner

    cache.get_or_compute("first", compute("first"))
    cache.get_or_compute("second", compute("second"))
    cache.get_or_compute("first", compute("first"))
    cache.get_or_compute("third", compute("third"))
    assert cache.stats.entries <= 2
    assert cache.stats.evictions >= 1
    cache.get_or_compute("first", compute("first"))
    assert calls["first"] == 1
    cache.get_or_compute("second", compute("second"))
    assert calls["second"] == 2


def test_byte_limit_and_oversized_support_payload_are_conservatively_bounded():
    cache = BoundedTeacherCache(max_bytes=2_000, max_entries=100)
    cache.get_or_compute("first", lambda: np.arange(64, dtype=np.float64))
    cache.get_or_compute("second", lambda: np.arange(64, dtype=np.float64))
    assert cache.stats.bytes_used <= 2_000
    assert cache.stats.evictions >= 1

    support_cache = BoundedTeacherCache(max_bytes=40_000, max_entries=4)
    empty_bytes = support_cache.stats.bytes_used
    calls = 0

    def support_heavy():
        nonlocal calls
        calls += 1
        return {"support": list(range(10_000))}

    support_cache.get_or_compute("support-heavy", support_heavy)
    support_cache.get_or_compute("support-heavy", support_heavy)
    assert calls == 2
    assert support_cache.stats.entries == 0
    assert support_cache.stats.bytes_used == empty_bytes
    assert support_cache.stats.oversized_skips == 2


def test_eviction_rebuilds_exact_tuple_storage_without_retained_mapping_capacity():
    cache = BoundedTeacherCache(max_bytes=8_000, max_entries=100)
    for index in range(20):
        cache.get_or_compute(f"fill-{index}", lambda index=index: np.array([index], dtype=np.float64))
    cache.get_or_compute("large", lambda: np.arange(256, dtype=np.float64))
    assert isinstance(cache._entries, tuple)
    assert cache.stats.bytes_used == cache._resident_bytes()
    assert cache.stats.bytes_used <= cache.max_bytes
    assert not hasattr(cache, "__dict__")
    assert all(not hasattr(entry, "__dict__") for entry in cache._entries)


def test_cache_rejects_budget_below_empty_accounted_footprint_and_skips_preserve_bound():
    probe = BoundedTeacherCache(max_bytes=1_000, max_entries=1)
    with pytest.raises(ValueError, match="empty cache footprint"):
        BoundedTeacherCache(max_bytes=probe.stats.bytes_used - 1, max_entries=1)
    baseline = probe.stats.bytes_used
    probe.get_or_compute("too-large", lambda: np.arange(10_000, dtype=np.float64))
    assert probe.stats.entries == 0
    assert probe.stats.bytes_used == baseline <= probe.max_bytes


def test_stateful_builtin_subclasses_are_rejected_from_keys_and_retained_values():
    class StatefulText(str):
        pass

    class StatefulInt(int):
        pass

    text = StatefulText("stateful-key")
    text.payload = bytearray(10_000)
    number = StatefulInt(7)
    number.payload = bytearray(10_000)
    cache = BoundedTeacherCache(max_bytes=10_000, max_entries=2)
    with pytest.raises(ValueError, match="cache key"):
        cache.get_or_compute(text, lambda: 1)
    with pytest.raises(TypeError, match="unsupported"):
        cache.get_or_compute("value", lambda: number)
    with pytest.raises(TypeError, match="string keys"):
        cache.get_or_compute("mapping", lambda: {text: 1})

    with pytest.raises(TypeError, match="canonical numeric"):
        cache.get_or_compute("string-dtype", lambda: np.empty(0, dtype="S4096"))

    class MisleadingDict(dict):
        def __iter__(self):
            return iter(())

    with pytest.raises(TypeError, match="exact dictionaries"):
        cache.get_or_compute("dict-subclass", lambda: MisleadingDict({text: 1}))


def test_identity_key_losslessly_accepts_parser_permitted_lone_surrogates():
    window = make_window("surrogate")
    pair = Pair("pair-\ud800", window.pairs[0].class_pair, window.pairs[0].initial_geometry,
                window.pairs[0].candidate_pair_count, window.pairs[0].valid_cells,
                window.pairs[0].process_cells)
    surrogate_window = Window(**{**window.__dict__, "window_id": "window-\ud800", "pairs": (pair,)})
    assert len(window_payload_key(surrogate_window)) == 64


def test_single_reference_key_cannot_share_changed_pair_payload():
    left = make_window("q")
    right = make_window("r")
    changed = make_window("r", process_offset=.25)
    left_key, right_key, changed_key = map(window_payload_key, (left, right, changed))
    scale = normal_scale_key(2, (right_key, "b" * 64, "c" * 64), {"block": NormalScale(np.zeros(2), np.ones(2))})
    query_pair = pair_payload_key(left_key, 0, left.pairs[0])
    reference_pair = pair_payload_key(right_key, 0, right.pairs[0])
    changed_pair = pair_payload_key(changed_key, 0, changed.pairs[0])
    assert single_reference_alignment_key(scale, query_pair, right_key, reference_pair) != single_reference_alignment_key(scale, query_pair, changed_key, changed_pair)
