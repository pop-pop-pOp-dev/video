"""Frozen, static-only normal-reference retrieval for NC-RTED teachers.

This module deliberately has no process descriptor, teacher distance, label, or
test-data input.  It selects only from caller-declared normal-permitted windows.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import math

import numpy as np


MAX_CANDIDATE_PAIRS = 16
CELL_COUNT = 4
BACKGROUND_EPSILON = 1e-8
GEOMETRY_COMPATIBILITY_RMS = 0.25
BACKGROUND_DIM = 1152
CLASS_COMPOSITION_DIM = 80


class RetrievalStatus(str, Enum):
    OK = "ok"
    MISSING_BACKGROUND = "missing_background"
    INVALID_STATIC_INPUT = "invalid_static_input"
    INSUFFICIENT_REFERENCES = "insufficient_references"
    INSUFFICIENT_CALIBRATION = "insufficient_calibration"
    QCR_ISOLATION_FAILURE = "qcr_isolation_failure"
    UNSUPPORTED_R_RECORDS = "unsupported_r_records"


class StaticRetrievalError(ValueError):
    pass


@dataclass(frozen=True)
class StaticPair:
    """Pair attributes admitted to matching; no motion or process field exists."""

    pair_id: str
    class_pair: tuple[int, int]
    initial_geometry: np.ndarray
    candidate_pair_count: int
    valid_cell_count: int


@dataclass(frozen=True)
class StaticWindow:
    window_id: str
    source_family: str
    content_alias: str
    normal_permitted: bool
    background: np.ndarray
    background_valid: bool
    class_composition: np.ndarray
    pairs: tuple[StaticPair, ...]


@dataclass(frozen=True)
class StaticDescriptor:
    background: np.ndarray
    class_composition: np.ndarray
    initial_geometry: np.ndarray
    support: np.ndarray


@dataclass(frozen=True)
class RetrievedPair:
    window: StaticWindow
    pair: StaticPair
    distance: float


@dataclass(frozen=True)
class RetrievalResult:
    status: RetrievalStatus
    selected: tuple[RetrievedPair, ...]
    reason: str | None = None


def stable_retrieval_hash(*parts: object) -> str:
    return hashlib.sha256("\0".join(map(str, parts)).encode()).hexdigest()


def _vector(value: np.ndarray, *, name: str, size: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or not array.size or (size is not None and array.size != size) or not np.isfinite(array).all():
        raise StaticRetrievalError(f"invalid {name}")
    return array


def static_descriptor(window: StaticWindow, pair: StaticPair) -> StaticDescriptor:
    """Build the only descriptor used by selection and threshold fitting."""
    if not window.background_valid:
        raise StaticRetrievalError("missing reliable background")
    background = _vector(window.background, name="background", size=BACKGROUND_DIM)
    if np.linalg.norm(background) < BACKGROUND_EPSILON:
        raise StaticRetrievalError("background norm below fixed epsilon")
    composition = _vector(window.class_composition, name="class composition", size=CLASS_COMPOSITION_DIM)
    if (composition < 0).any() or not np.isclose(composition.sum(), 1.0, atol=1e-6):
        raise StaticRetrievalError("class composition must be normalized")
    geometry = _vector(pair.initial_geometry, name="initial geometry", size=5)
    if len(pair.class_pair) != 2 or any(not isinstance(item, int) or not 0 <= item < 80 for item in pair.class_pair):
        raise StaticRetrievalError("class pair must contain ordered COCO classes")
    if not 1 <= pair.candidate_pair_count <= MAX_CANDIDATE_PAIRS or not 1 <= pair.valid_cell_count <= CELL_COUNT:
        raise StaticRetrievalError("candidate and valid-cell support counts are outside frozen bounds")
    support = np.asarray((pair.candidate_pair_count / MAX_CANDIDATE_PAIRS, pair.valid_cell_count / CELL_COUNT), dtype=np.float64)
    return StaticDescriptor(background, composition, geometry, support)


def static_distance(first: StaticDescriptor, second: StaticDescriptor) -> float:
    """Equal-weight frozen static distance: background, class, geometry, support."""
    a_background = _vector(first.background, name="first background")
    b_background = _vector(second.background, name="second background", size=a_background.size)
    cosine = float(np.dot(a_background, b_background) / max(np.linalg.norm(a_background) * np.linalg.norm(b_background), BACKGROUND_EPSILON))
    background = 1.0 - float(np.clip(cosine, -1.0, 1.0))
    blocks = [background]
    for name in ("class_composition", "initial_geometry", "support"):
        left = _vector(getattr(first, name), name=f"first {name}")
        right = _vector(getattr(second, name), name=f"second {name}", size=left.size)
        blocks.append(float(np.sqrt(np.mean((left - right) ** 2))))
    return float(np.mean(blocks))


def pair_compatible(query: StaticPair, candidate: StaticPair) -> bool:
    """Only ordered COCO classes and initial geometry determine pair matching."""
    if query.class_pair != candidate.class_pair:
        return False
    query_geometry = _vector(query.initial_geometry, name="query initial geometry", size=5)
    candidate_geometry = _vector(candidate.initial_geometry, name="candidate initial geometry", size=5)
    return float(np.sqrt(np.mean((query_geometry - candidate_geometry) ** 2))) <= GEOMETRY_COMPATIBILITY_RMS


def _threshold(value: float) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise StaticRetrievalError("compatibility threshold must be finite and nonnegative")
    return float(value)


def _unique_window_ids(windows: tuple[StaticWindow, ...]) -> None:
    identifiers = [window.window_id for window in windows]
    if len(set(identifiers)) != len(identifiers):
        raise StaticRetrievalError("candidate window IDs must be unique")


def _eligible(query_window: StaticWindow, query_pair: StaticPair, candidates: tuple[StaticWindow, ...], *, forbidden_families: set[str], forbidden_aliases: set[str], compatibility_threshold: float) -> tuple[list[RetrievedPair], int]:
    query = static_descriptor(query_window, query_pair)
    result = []
    skipped_invalid = 0
    for window in candidates:
        if not window.normal_permitted or window.source_family in forbidden_families or window.content_alias in forbidden_aliases:
            continue
        try:
            for pair in window.pairs:
                if pair_compatible(query_pair, pair):
                    distance = static_distance(query, static_descriptor(window, pair))
                    if distance <= compatibility_threshold:
                        result.append(RetrievedPair(window, pair, distance))
        except StaticRetrievalError:
            skipped_invalid += 1
    return sorted(result, key=lambda item: (item.distance, stable_retrieval_hash("nc-rted-static-retrieval-v1", item.window.window_id, item.pair.pair_id))), skipped_invalid


def select_references(query_window: StaticWindow, query_pair: StaticPair, candidates: tuple[StaticWindow, ...], compatibility_threshold: float) -> RetrievalResult:
    """Select exactly three normal R pairs with family and content-alias isolation."""
    try:
        compatibility_threshold = _threshold(compatibility_threshold)
        _unique_window_ids(candidates)
        selected = []
        used_families = {query_window.source_family}
        used_aliases = {query_window.content_alias}
        eligible, skipped = _eligible(query_window, query_pair, candidates, forbidden_families=used_families, forbidden_aliases=used_aliases, compatibility_threshold=compatibility_threshold)
        for item in eligible:
            if item.window.source_family in used_families or item.window.content_alias in used_aliases:
                continue
            selected.append(item)
            used_families.add(item.window.source_family)
            used_aliases.add(item.window.content_alias)
            if len(selected) == 3:
                return RetrievalResult(RetrievalStatus.OK, tuple(selected))
        return RetrievalResult(RetrievalStatus.INSUFFICIENT_REFERENCES, (), f"need three threshold-compatible distinct normal source families and aliases; skipped_invalid={skipped}")
    except StaticRetrievalError as error:
        status = RetrievalStatus.MISSING_BACKGROUND if "background" in str(error) else RetrievalStatus.INVALID_STATIC_INPUT
        return RetrievalResult(status, (), str(error))


def select_calibration(query_window: StaticWindow, query_pair: StaticPair, references: tuple[RetrievedPair, ...], candidates: tuple[StaticWindow, ...], compatibility_threshold: float) -> RetrievalResult:
    """Select C using fixed support limits and disjoint Q/C/R provenance."""
    try:
        compatibility_threshold = _threshold(compatibility_threshold)
        _unique_window_ids(candidates)
        if len(references) != 3:
            raise StaticRetrievalError("calibration requires exactly three references")
        reference_families = {item.window.source_family for item in references}
        reference_aliases = {item.window.content_alias for item in references}
        if (len(reference_families) != 3 or len(reference_aliases) != 3
                or query_window.source_family in reference_families or query_window.content_alias in reference_aliases
                or len({item.window.window_id for item in references}) != 3
                or any(not item.window.normal_permitted for item in references)):
            return RetrievalResult(RetrievalStatus.QCR_ISOLATION_FAILURE, (), "references are not provenance-distinct")
        query_descriptor = static_descriptor(query_window, query_pair)
        for item in references:
            if not pair_compatible(query_pair, item.pair) or static_distance(query_descriptor, static_descriptor(item.window, item.pair)) > compatibility_threshold:
                return RetrievalResult(RetrievalStatus.QCR_ISOLATION_FAILURE, (), "reference is not threshold-compatible with query")
        forbidden_families = reference_families | {query_window.source_family}
        forbidden_aliases = reference_aliases | {query_window.content_alias}
        selected, per_family, alias_owner, used_windows = [], {}, {}, set()
        eligible, skipped = _eligible(query_window, query_pair, candidates, forbidden_families=forbidden_families, forbidden_aliases=forbidden_aliases, compatibility_threshold=compatibility_threshold)
        for item in eligible:
            family = item.window.source_family
            alias = item.window.content_alias
            if item.window.window_id in used_windows or per_family.get(family, 0) >= 4:
                continue
            if alias in alias_owner and alias_owner[alias] != family:
                continue
            selected.append(item)
            used_windows.add(item.window.window_id)
            alias_owner[alias] = family
            per_family[family] = per_family.get(family, 0) + 1
            if len(selected) == 128:
                break
        if len(selected) < 64 or len(per_family) < 16:
            return RetrievalResult(RetrievalStatus.INSUFFICIENT_CALIBRATION, (), f"need 64..128 threshold-compatible normal windows from at least 16 families, max four per family; skipped_invalid={skipped}")
        return RetrievalResult(RetrievalStatus.OK, tuple(selected))
    except StaticRetrievalError as error:
        status = RetrievalStatus.MISSING_BACKGROUND if "background" in str(error) else RetrievalStatus.INVALID_STATIC_INPUT
        return RetrievalResult(status, (), str(error))


def assert_role_isolation(query_windows: tuple[StaticWindow, ...], calibration_windows: tuple[StaticWindow, ...], reference_windows: tuple[StaticWindow, ...]) -> None:
    """Verify complete Q/C/R pools, not merely the selected retrieval result."""
    pools = (query_windows, calibration_windows, reference_windows)
    for pool in pools:
        _unique_window_ids(pool)
    for left_index in range(3):
        for right_index in range(left_index + 1, 3):
            left_families = {window.source_family for window in pools[left_index]}
            right_families = {window.source_family for window in pools[right_index]}
            left_aliases = {window.content_alias for window in pools[left_index]}
            right_aliases = {window.content_alias for window in pools[right_index]}
            if left_families & right_families or left_aliases & right_aliases:
                raise StaticRetrievalError("Q/C/R pools must be family- and content-alias-disjoint")


def r_leave_one_source_threshold(records: tuple[tuple[StaticWindow, StaticPair], ...]) -> float:
    """Fixed nearest-rank 95th percentile of R leave-one-source static distances."""
    if len(records) < 2:
        raise StaticRetrievalError("R threshold requires at least two normal records")
    nearest = []
    for index, (window, pair) in enumerate(records):
        if not window.normal_permitted:
            raise StaticRetrievalError("R threshold requires normal-permitted records")
        descriptor = static_descriptor(window, pair)
        distances = []
        for other_index, (other_window, other_pair) in enumerate(records):
            if other_index == index or other_window.source_family == window.source_family or other_window.content_alias == window.content_alias:
                continue
            if other_window.normal_permitted and pair_compatible(pair, other_pair):
                distances.append(static_distance(descriptor, static_descriptor(other_window, other_pair)))
        if not distances:
            raise StaticRetrievalError(f"unsupported R record without leave-one-source neighbor: {window.window_id}")
        nearest.append(min(distances))
    ordered = np.sort(np.asarray(nearest, dtype=np.float64))
    return float(ordered[math.ceil(0.95 * ordered.size) - 1])
