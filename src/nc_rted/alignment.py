"""Fixed short-sequence comparison primitives for the NC-RTED teacher.

Reference selection, normal-label permission and Q/C/R isolation are enforced by
the manifest/retrieval layer, not inferred from these numerical arrays.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class NormalScale:
    center: np.ndarray
    scale: np.ndarray


@dataclass(frozen=True)
class Alignment:
    valid: bool
    reason: str
    path: tuple[tuple[int, int], ...]
    cell_distance: np.ndarray
    cell_valid: np.ndarray
    total_cost: float


def fit_reference_scale(normal_reference_rows: np.ndarray,
                        scale_floor: float = 1e-3) -> NormalScale:
    """Coordinate median / 1.4826 MAD on valid R-side normal rows only.

    Caller must bind the exact R manifest. Query or calibration observations may
    not be passed here. The floor is fixed before measuring task performance.
    """
    rows = np.asarray(normal_reference_rows, dtype=np.float64)
    if rows.ndim != 2 or not rows.shape[0] or not rows.shape[1]:
        raise ValueError("reference rows must be nonempty [normal observations,D]")
    if not np.isfinite(rows).all() or not np.isfinite(scale_floor) or scale_floor <= 0:
        raise ValueError("reference rows and scale floor must be finite")
    center = np.median(rows, axis=0)
    scale = np.maximum(1.4826 * np.median(np.abs(rows - center), axis=0), scale_floor)
    return NormalScale(center, scale)


def process_cost_matrix(query_blocks: dict[str, np.ndarray],
                        reference_blocks: dict[str, np.ndarray],
                        scales: dict[str, NormalScale], query_valid: np.ndarray,
                        reference_valid: np.ndarray) -> np.ndarray:
    """Equal block weights, each with dimension-normalized standardized L2.

    Each field is a signed per-cell descriptor [4,D]. NaNs in invalid cells are
    ignored; nonfinite valid descriptors fail closed. Invalid pairs cost infinity.
    """
    fields = sorted(query_blocks)
    if not fields or set(fields) != set(reference_blocks) or set(fields) != set(scales):
        raise ValueError("query/reference/scale feature fields must match exactly")
    qm, rm = np.asarray(query_valid, bool), np.asarray(reference_valid, bool)
    if qm.shape != (4,) or rm.shape != (4,):
        raise ValueError("exactly four cell validity flags are required")
    qi, ri = np.flatnonzero(qm), np.flatnonzero(rm)
    result = np.full((4, 4), np.inf, dtype=np.float64)
    total = np.zeros((len(qi), len(ri)), dtype=np.float64)
    for field in fields:
        q, r = np.asarray(query_blocks[field], float), np.asarray(reference_blocks[field], float)
        normal = scales[field]
        if q.ndim != 2 or q.shape[0] != 4 or r.shape != q.shape or q.shape[1] == 0:
            raise ValueError("each process field must have matching [4,D] arrays")
        if normal.scale.shape != (q.shape[1],) or normal.center.shape != normal.scale.shape:
            raise ValueError("reference scale shape mismatch")
        if (not np.isfinite(normal.scale).all() or np.any(normal.scale <= 0)
                or not np.isfinite(normal.center).all()):
            raise ValueError("invalid frozen reference scale")
        if not np.isfinite(q[qi]).all() or not np.isfinite(r[ri]).all():
            raise ValueError("nonfinite valid process descriptor")
        difference = (q[qi, None, :] - r[None, ri, :]) / normal.scale
        total += np.sqrt(np.mean(difference ** 2, axis=-1))
    result[np.ix_(qi, ri)] = total / len(fields)
    return result


def constrained_alignment(costs: np.ndarray, query_valid: np.ndarray,
                          reference_valid: np.ndarray) -> Alignment:
    """Exact deterministic DTW over available cells, without fake zero features.

    Endpoints are the first/last available observed cells. Steps are diagonal,
    horizontal or vertical, with no two consecutive horizontal or vertical
    steps. Optimize sum cost, then shorter path, then lexicographic original cell
    indices. Map multiple reference matches back by their mean local cost per
    original query cell. Missing query cells keep NaN with validity false.
    """
    costs = np.asarray(costs, dtype=np.float64)
    qm, rm = np.asarray(query_valid, bool), np.asarray(reference_valid, bool)
    if costs.shape != (4, 4) or qm.shape != (4,) or rm.shape != (4,):
        raise ValueError("DTW requires a four-by-four cost matrix and four-cell masks")
    qi, ri = np.flatnonzero(qm), np.flatnonzero(rm)
    empty = np.full(4, np.nan, dtype=np.float64)
    if not qi.size or not ri.size:
        return Alignment(False, "no_observed_cells", (), empty, np.zeros(4, bool), float("inf"))
    observed = costs[np.ix_(qi, ri)]
    if not np.isfinite(observed).all() or np.any(observed < 0):
        raise ValueError("observed comparisons must have finite nonnegative costs")
    paths = []

    def visit(i, j, last_step, path):
        if i == len(qi) - 1 and j == len(ri) - 1:
            original = tuple((int(qi[a]), int(ri[b])) for a, b in path)
            paths.append((sum(costs[a, b] for a, b in original), len(original), original))
            return
        for di, dj, step in ((1, 1, "D"), (0, 1, "H"), (1, 0, "V")):
            if step in {"H", "V"} and step == last_step:
                continue
            ni, nj = i + di, j + dj
            if ni < len(qi) and nj < len(ri):
                visit(ni, nj, step, path + [(ni, nj)])

    visit(0, 0, None, [(0, 0)])
    if not paths:
        return Alignment(False, "no_legal_monotonic_path", (), empty, np.zeros(4, bool), float("inf"))
    total, _, path = min(paths)
    mapped = empty.copy()
    for i in qi:
        mapped[i] = np.mean([costs[a, b] for a, b in path if a == i])
    return Alignment(True, "ok", path, mapped, qm.copy(), float(total))
