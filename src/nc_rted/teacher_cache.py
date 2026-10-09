"""Exact, bounded in-memory reuse primitives for NC-RTED teacher computation.

This module deliberately does not alter teacher selection or numerical code.  A
future integration can wrap repeated process-cost/DTW calls with
``BoundedTeacherCache.get_or_compute`` using the content keys below.  Entries
are per-build only, byte-accounted, and never persisted.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import struct
import sys
from typing import Callable, TypeVar

import numpy as np


CACHE_KEY_SCHEMA = "nc_rted_teacher_cache/v1"
_ENTRY_OVERHEAD_BYTES = 256
_T = TypeVar("_T")
_SCALAR_TYPES = {type(None), bool, int, float, str, bytes}


def _write_bytes(digest: "hashlib._Hash", value: bytes) -> None:
    digest.update(len(value).to_bytes(8, "big"))
    digest.update(value)


def _write_text(digest: "hashlib._Hash", value: str) -> None:
    if type(value) is not str:
        raise TypeError("cache identity text fields must be strings")
    _write_bytes(digest, value.encode("utf-8", errors="surrogatepass"))


def _write_array(digest: "hashlib._Hash", value: object) -> None:
    array = np.asarray(value)
    _validate_array_dtype(array.dtype, "immutable teacher-cache keys")
    contiguous = np.ascontiguousarray(array)
    digest.update(b"A")
    _write_text(digest, array.dtype.str)
    _write_bytes(digest, np.asarray(array.shape, dtype=np.int64).tobytes())
    _write_bytes(digest, contiguous.tobytes())


def _validate_array_dtype(dtype: np.dtype, purpose: str) -> None:
    canonical = np.dtype(dtype.str)
    if (dtype.kind not in "biufc" or dtype.hasobject or dtype.fields is not None
            or dtype.subdtype is not None or dtype.metadata is not None or canonical != dtype):
        raise TypeError(f"only canonical numeric dtypes can form {purpose}")


def _digest(label: str, write: Callable[["hashlib._Hash"], None]) -> str:
    result = hashlib.sha256()
    _write_text(result, CACHE_KEY_SCHEMA)
    _write_text(result, label)
    write(result)
    return result.hexdigest()


def _write_pair_payload(digest: "hashlib._Hash", pair: object) -> None:
    _write_text(digest, str(pair.pair_id))
    _write_array(digest, pair.class_pair)
    _write_array(digest, pair.initial_geometry)
    _write_bytes(digest, int(pair.candidate_pair_count).to_bytes(8, "big", signed=True))
    _write_array(digest, pair.valid_cells)
    _write_array(digest, pair.process_cells)


def window_payload_key(window: object) -> str:
    """Return an immutable identity for every payload field used by the teacher.

    IDs are included for audit provenance, but are insufficient on their own:
    all arrays include their dtype, shape, and contiguous bytes, including masks.
    Pair ordering is represented explicitly.
    """
    def write(digest: "hashlib._Hash") -> None:
        for field in ("dataset", "window_id", "source_family", "content_alias"):
            _write_text(digest, str(getattr(window, field)))
        _write_bytes(digest, int(window.fold).to_bytes(8, "big", signed=True))
        digest.update(b"1" if bool(window.normal_permitted) else b"0")
        digest.update(b"1" if bool(window.background_valid) else b"0")
        _write_array(digest, window.background)
        _write_array(digest, window.class_composition)
        pairs = tuple(window.pairs)
        _write_bytes(digest, len(pairs).to_bytes(8, "big"))
        for index, pair in enumerate(pairs):
            _write_bytes(digest, index.to_bytes(8, "big"))
            _write_pair_payload(digest, pair)
    return _digest("window_payload", write)


def pair_payload_key(window_key: str, pair_index: int, pair: object) -> str:
    """Bind a pair payload to its containing window and ordered position."""
    if not isinstance(window_key, str) or len(window_key) != 64:
        raise ValueError("invalid containing window cache key")
    if not isinstance(pair_index, int) or pair_index < 0:
        raise ValueError("invalid pair index")

    def write(digest: "hashlib._Hash") -> None:
        _write_text(digest, window_key)
        _write_bytes(digest, pair_index.to_bytes(8, "big"))
        _write_pair_payload(digest, pair)
    return _digest("pair_payload", write)


def ordered_reference_key(reference_window_keys: tuple[str, ...] | list[str]) -> str:
    """Fingerprint a reference tuple in row order; it is intentionally not a set."""
    keys = tuple(reference_window_keys)
    if not keys or any(not isinstance(key, str) or len(key) != 64 for key in keys):
        raise ValueError("ordered references require nonempty SHA-256 window keys")

    def write(digest: "hashlib._Hash") -> None:
        _write_bytes(digest, len(keys).to_bytes(8, "big"))
        for key in keys:
            _write_text(digest, key)
    return _digest("ordered_reference_tuple", write)


def normal_scale_key(fold: int, r_window_keys: tuple[str, ...] | list[str], scales: dict[str, object]) -> str:
    """Bind a scale set to the ordered R payloads and exact NormalScale values."""
    if not isinstance(fold, int):
        raise ValueError("fold must be an integer")
    references = tuple(r_window_keys)
    if not references or any(not isinstance(key, str) or len(key) != 64 for key in references):
        raise ValueError("scale identity requires nonempty ordered R window keys")
    if not scales:
        raise ValueError("scale identity requires at least one process block")

    def write(digest: "hashlib._Hash") -> None:
        _write_bytes(digest, fold.to_bytes(8, "big", signed=True))
        for key in references:
            _write_text(digest, key)
        for name in sorted(scales):
            scale = scales[name]
            _write_text(digest, name)
            _write_array(digest, scale.center)
            _write_array(digest, scale.scale)
    return _digest("normal_scales", write)


def window_teacher_distances_key(scale_key: str, window_key: str, ordered_references_key: str) -> str:
    """Key a [3, 64] teacher-distance result by scale and ordered R triple."""
    for value in (scale_key, window_key, ordered_references_key):
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError("window teacher result requires SHA-256 component keys")

    def write(digest: "hashlib._Hash") -> None:
        for part in (scale_key, window_key, ordered_references_key):
            _write_text(digest, part)
    return _digest("window_teacher_distances", write)


def single_reference_alignment_key(scale_key: str, query_pair_key: str, reference_window_key: str,
                                   matched_reference_pair_key: str) -> str:
    """Key one matched process-cost/DTW result with the actual fitted scale."""
    parts = (scale_key, query_pair_key, reference_window_key, matched_reference_pair_key)
    if any(not isinstance(value, str) or len(value) != 64 for value in parts):
        raise ValueError("single-reference result requires SHA-256 component keys")

    def write(digest: "hashlib._Hash") -> None:
        for part in parts:
            _write_text(digest, part)
    return _digest("single_reference_alignment", write)


def threshold_key(value: float) -> str:
    """Identity for a retrieval threshold using IEEE-754 bytes, not formatting."""
    numeric = float(value)
    return _digest("threshold", lambda digest: _write_bytes(digest, struct.pack(">d", numeric)))


def derived_key(label: str, *parts: str) -> str:
    """Build a namespaced result key from already immutable component keys."""
    if any(not isinstance(part, str) or len(part) != 64 for part in parts):
        raise ValueError("derived cache keys require SHA-256 component keys")

    def write(digest: "hashlib._Hash") -> None:
        for part in parts:
            _write_text(digest, part)
    return _digest(label, write)


@dataclass(frozen=True, slots=True)
class CachedFailure:
    """Exception type/message captured so cached failures replay exactly."""
    error_args: tuple[object, ...]

    def raise_error(self) -> None:
        raise ValueError(*self.error_args)


@dataclass(frozen=True, slots=True)
class CacheStats:
    hits: int
    misses: int
    evictions: int
    entries: int
    bytes_used: int
    oversized_skips: int


@dataclass(frozen=True, slots=True)
class _Entry:
    key: str
    value: object


@dataclass(frozen=True, slots=True)
class _FrozenList:
    items: tuple[object, ...]


@dataclass(frozen=True, slots=True)
class _FrozenDict:
    items: tuple[tuple[str, object], ...]


def _freeze(value: object) -> object:
    if isinstance(value, np.ndarray):
        _validate_array_dtype(value.dtype, "teacher-cache values")
        stored = np.array(value, copy=True, order="C", subok=False)
        stored.setflags(write=False)
        return stored
    if isinstance(value, np.generic):
        raise TypeError("NumPy scalar cache values are unsupported; use an ndarray or Python scalar")
    if type(value) in _SCALAR_TYPES:
        return value
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, list):
        return _FrozenList(tuple(_freeze(item) for item in value))
    if isinstance(value, dict):
        if type(value) is not dict:
            raise TypeError("teacher-cache support dictionaries must be exact dictionaries")
        items = tuple(value.items())
        if any(type(key) is not str for key, _ in items):
            raise TypeError("teacher-cache support dictionaries require string keys")
        return _FrozenDict(tuple((key, _freeze(item)) for key, item in items))
    raise TypeError(f"unsupported teacher-cache value: {type(value)!r}")


def _thaw(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, _FrozenList):
        return [_thaw(item) for item in value.items]
    if isinstance(value, _FrozenDict):
        return {key: _thaw(item) for key, item in value.items}
    if isinstance(value, tuple):
        return tuple(_thaw(item) for item in value)
    return value


def _retained_bytes(value: object, seen: set[int] | None = None) -> int:
    """Conservatively charge every retained object reachable from an entry."""
    if seen is None:
        seen = set()
    marker = id(value)
    if marker in seen:
        return 0
    seen.add(marker)
    size = sys.getsizeof(value)
    if isinstance(value, np.ndarray):
        # Array headers and backing storage are both charged.  This can
        # over-account on a given NumPy build, which is acceptable for a hard
        # cache limit and avoids assuming a particular getsizeof convention.
        return size + value.nbytes
    if isinstance(value, _Entry):
        return size + _retained_bytes(value.key, seen) + _retained_bytes(value.value, seen)
    if isinstance(value, _FrozenList):
        return size + _retained_bytes(value.items, seen)
    if isinstance(value, _FrozenDict):
        return size + _retained_bytes(value.items, seen)
    if isinstance(value, tuple):
        return size + sum(_retained_bytes(item, seen) for item in value)
    if isinstance(value, CachedFailure):
        return size + _retained_bytes(value.error_args, seen)
    return size


def _cacheable_failure(error: Exception) -> CachedFailure | None:
    """Cache only the explicitly supported built-in failure contract.

    Future pipeline integration may recompute any unsupported failure.  That
    leaves the observed exception untouched and avoids retaining custom error
    types, constructor state, or class-owned payloads in this bounded cache.
    """
    if type(error) is not ValueError:
        return None
    if (getattr(error, "__dict__", None) or getattr(error, "__notes__", None)
            or error.__cause__ is not None or error.__context__ is not None
            or error.__suppress_context__):
        return None
    if not all(type(item) in _SCALAR_TYPES for item in error.args):
        return None
    return CachedFailure(tuple(error.args))


class BoundedTeacherCache:
    """Byte-bounded LRU for compact, immutable DTW outcomes and failures.

    The class has one fixed byte ceiling and one entry ceiling.  Oversized values
    are returned but not admitted, so eviction only changes recomputation count.
    ``get_or_compute`` stores exceptions derived from ``Exception`` and replays
    their original type/message on a hit.
    """
    __slots__ = ("max_bytes", "max_entries", "_entries", "_hits", "_misses", "_evictions", "_oversized_skips")

    def __init__(self, max_bytes: int, max_entries: int) -> None:
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError("max_bytes must be a positive integer")
        if type(max_entries) is not int or max_entries <= 0:
            raise ValueError("max_entries must be a positive integer")
        empty_footprint = _ENTRY_OVERHEAD_BYTES + _retained_bytes(())
        if max_bytes < empty_footprint:
            raise ValueError("max_bytes is smaller than the accounted empty cache footprint")
        self.max_bytes = max_bytes
        self.max_entries = max_entries
        # A tuple has no retained spare capacity.  Every hit/admission rebuilds
        # it in MRU-to-LRU order, making its live allocation measurable.
        self._entries: tuple[_Entry, ...] = ()
        self._hits = self._misses = self._evictions = self._oversized_skips = 0

    @property
    def stats(self) -> CacheStats:
        return CacheStats(self._hits, self._misses, self._evictions, len(self._entries),
                          self._resident_bytes(), self._oversized_skips)

    def _resident_bytes(self, entries: tuple[_Entry, ...] | None = None) -> int:
        """Measure the entire retained entry tuple, without capacity allowance."""
        current = self._entries if entries is None else entries
        return _ENTRY_OVERHEAD_BYTES + _retained_bytes(current)

    def get_or_compute(self, key: str, compute: Callable[[], _T]) -> _T:
        if type(key) is not str or not key:
            raise ValueError("cache key must be a nonempty string")
        for index, entry in enumerate(self._entries):
            if entry.key != key:
                continue
            self._hits += 1
            self._entries = (entry,) + self._entries[:index] + self._entries[index + 1:]
            if isinstance(entry.value, CachedFailure):
                entry.value.raise_error()
            return _thaw(entry.value)  # type: ignore[return-value]

        self._misses += 1
        try:
            value = compute()
        except Exception as error:
            failure = _cacheable_failure(error)
            if failure is not None:
                self._admit(key, failure)
            raise
        frozen = _freeze(value)
        self._admit(key, frozen)
        return _thaw(frozen)  # type: ignore[return-value]

    def _admit(self, key: str, frozen_value: object) -> None:
        candidate = (_Entry(key, frozen_value),) + self._entries
        if self._resident_bytes(candidate[:1]) > self.max_bytes:
            self._oversized_skips += 1
            return
        while len(candidate) > self.max_entries or self._resident_bytes(candidate) > self.max_bytes:
            candidate = candidate[:-1]
            self._evictions += 1
        self._entries = candidate
