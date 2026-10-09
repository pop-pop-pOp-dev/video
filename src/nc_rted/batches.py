"""Pack frozen relation windows into a single sample without dropping blocks."""
from __future__ import annotations

import math

import numpy as np
import torch

from .bridge import ObservationBatch
from .features import CELL_FEATURE_DIM, FeatureAssemblyResult, FeatureStatus


def caption_block_endpoints(observed_seconds: float) -> tuple[float, ...]:
    """End points cover the complete observed media, including a partial tail.

    The final partial block is left-aligned to the 8-second partition, not
    reinterpreted as an overlapping last-eight-second sliding window. Callers
    pass its explicit left boundary when assigning its four temporal cells.
    """
    if not math.isfinite(observed_seconds) or observed_seconds <= 0:
        raise ValueError("positive observed media duration is required")
    count = math.ceil(observed_seconds / 8.)
    return tuple(min((i + 1) * 8., observed_seconds) for i in range(count))


def pack_observation_blocks(blocks: list[FeatureAssemblyResult], *, task: str,
                            dtype: torch.dtype = torch.bfloat16,
                            device: str | torch.device = "cpu") -> ObservationBatch:
    """Padding remains NaN under an exact mask; source/track IDs never enter tensors."""
    if task not in {"detection", "caption"} or not blocks:
        raise ValueError("nonempty observation blocks and known task are required")
    if task == "detection" and len(blocks) != 1:
        raise ValueError("detection uses one legal recent eight-second window")
    if dtype not in {torch.bfloat16, torch.float32}:
        raise ValueError("only original BF16 or diagnostic FP32 features are supported")
    if any(block.status in {FeatureStatus.INVALID_INPUT, FeatureStatus.TRACKING_FAILURE} for block in blocks):
        raise ValueError("technical observation failure must be resolved or recorded, not treated as normal")
    candidates = max((len(block.relations) for block in blocks), default=0)
    if candidates > 16:
        raise ValueError("candidate cap exceeded")
    shape = (1, len(blocks), candidates, 4)
    features = torch.full((*shape, CELL_FEATURE_DIM), float("nan"), dtype=dtype, device=device)
    valid = torch.zeros(shape, dtype=torch.bool, device=device)
    times = torch.full(shape, float("nan"), dtype=torch.float32, device=device)
    for index, block in enumerate(blocks):
        if block.status == FeatureStatus.NO_RELATION_PAIRS and block.relations:
            raise ValueError("no-candidate status contradicts relation data")
        for candidate, relation in enumerate(block.relations):
            mask = np.asarray(relation.feature_valid, dtype=bool)
            values = np.asarray(relation.student_cells)
            observed = np.asarray(relation.observed_times_s)
            if mask.shape != (4,) or values.shape != (4, CELL_FEATURE_DIM) or observed.shape != (4,):
                raise ValueError("relation feature schema mismatch")
            if not np.array_equal(mask, relation.cell_mask):
                raise ValueError("cell/feature validity masks disagree")
            if not np.isfinite(values[mask]).all() or not np.isfinite(observed[mask]).all():
                raise ValueError("valid relation data must be finite")
            if (observed[mask] < 0).any() or np.any(np.diff(observed[mask]) <= 0):
                raise ValueError("actual observed relation times must increase")
            features[0, index, candidate] = torch.as_tensor(values.copy(), device=device, dtype=dtype)
            valid[0, index, candidate] = torch.as_tensor(mask.copy(), device=device)
            times[0, index, candidate] = torch.as_tensor(observed.copy(), device=device, dtype=torch.float32)
    return ObservationBatch(features, valid, times)
