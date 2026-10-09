"""Deterministic observation boundaries for the frozen, causal NC-RTED input.

This module does not decode videos, run a detector, or infer anomaly labels.
Callers supply actual frame timestamps and the unchanged baseline SigLip features.
"""
from __future__ import annotations

import math
import numpy as np
import torch
from torch import Tensor


def causal_frame_indices(frame_pts: np.ndarray, query_s: float) -> np.ndarray:
    """Select unique frames on a global 2 FPS clock in ``(query-8, query]``.

    Each nominal tick selects the last decoded frame at or before that tick.
    Low-frame-rate media are not made into fake repeated observations. Appending
    later frames cannot change the returned prefix. Timestamps use media seconds.
    """
    pts = np.asarray(frame_pts, dtype=np.float64)
    if pts.ndim != 1 or not np.isfinite(pts).all() or np.any(np.diff(pts) <= 0):
        raise ValueError("frame timestamps must be finite and strictly increasing")
    if not math.isfinite(query_s) or query_s < 0:
        raise ValueError("query must be a nonnegative finite media timestamp")
    first_tick = max(0, math.floor((query_s - 8.0) * 2.0) + 1)
    last_tick = math.floor(query_s * 2.0)
    ticks = np.arange(first_tick, last_tick + 1, dtype=np.float64) / 2.0
    indices = np.searchsorted(pts, ticks, side="right") - 1
    indices = indices[indices >= 0]
    if indices.size:
        indices = indices[(pts[indices] > query_s - 8.0) & (pts[indices] <= query_s)]
    result = np.unique(indices).astype(np.int64)
    if result.size > 16:
        raise AssertionError("2 FPS / eight seconds permits at most sixteen observations")
    return result


def temporal_cells(observed_pts: np.ndarray, query_s: float) -> np.ndarray:
    """Map actual observations to four ordered two-second intervals.

    The window is (query-8, query]; an internal boundary belongs to the later
    interval and the query endpoint belongs to cell 3. No missing cell is filled.
    """
    pts = np.asarray(observed_pts, dtype=np.float64)
    if not math.isfinite(query_s) or pts.ndim != 1 or not np.isfinite(pts).all():
        raise ValueError("invalid timestamps")
    if np.any(pts <= query_s - 8.0) or np.any(pts > query_s):
        raise ValueError("observation outside the causal eight-second window")
    return np.minimum(np.floor((pts - (query_s - 8.0)) / 2.0).astype(np.int64), 3)


def patch_overlap_weights(boxes_xyxy: Tensor, image_size: int = 384,
                          patch_size: int = 14) -> tuple[Tensor, Tensor]:
    """Area overlap with physical patch receptive fields after direct resize.

    Boxes are normalized in original-image xyxy coordinates. The actual baseline
    directly resizes to 384 square, with no crop/letterbox. A valid-stride-14
    convolution yields 27 patches, covering [0,378), NOT the whole [0,384) image.
    """
    if image_size != 384 or patch_size != 14:
        raise ValueError("only the verified SigLip-384/14 preprocessing is supported")
    if boxes_xyxy.ndim != 2 or boxes_xyxy.shape[-1] != 4:
        raise ValueError("boxes must be [instances,4] normalized xyxy")
    boxes = boxes_xyxy.float()
    valid = (torch.isfinite(boxes).all(-1) & (boxes >= 0).all(-1)
             & (boxes <= 1).all(-1) & (boxes[:, 2] > boxes[:, 0])
             & (boxes[:, 3] > boxes[:, 1]))
    boxes = torch.where(valid[:, None], boxes, torch.zeros_like(boxes)) * image_size
    axis = torch.arange(image_size // patch_size, device=boxes.device,
                        dtype=torch.float32) * patch_size
    overlap_x = (torch.minimum(boxes[:, 2, None], axis + patch_size)
                 - torch.maximum(boxes[:, 0, None], axis)).clamp_min(0)
    overlap_y = (torch.minimum(boxes[:, 3, None], axis + patch_size)
                 - torch.maximum(boxes[:, 1, None], axis)).clamp_min(0)
    area = (overlap_y[:, :, None] * overlap_x[:, None, :]).flatten(1)
    valid = valid & (area.sum(-1) > 0)
    return area, valid


def pool_patch_regions(patches: Tensor, boxes_xyxy: Tensor) -> tuple[Tensor, Tensor]:
    """Pool from one [729,D] frozen frame, returning original dtype and validity."""
    if patches.ndim != 2 or patches.shape[0] != 729 or not patches.is_floating_point():
        raise ValueError("expected floating SigLip patch tensor [729,D]")
    if patches.device != boxes_xyxy.device:
        raise ValueError("patches and boxes must share a device")
    if not bool(torch.isfinite(patches).all()):
        raise ValueError("nonfinite observed patch features")
    weights, valid = patch_overlap_weights(boxes_xyxy)
    weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
    pooled = (weights @ patches.float()).to(patches.dtype)
    return pooled, valid


def background_feature(patches: Tensor, boxes_xyxy: Tensor,
                       min_uncovered_fraction: float = 0.1) -> tuple[Tensor, bool]:
    """Conservative static context from patches untouched by any valid detection.

    Insufficient background is explicitly invalid, never a zero-valued normal
    reference. Whole-frame context remains a separate student feature.
    """
    if patches.ndim != 2 or patches.shape[0] != 729:
        raise ValueError("expected [729,D] patches")
    if not 0 < min_uncovered_fraction <= 1:
        raise ValueError("background fraction must be in (0,1]")
    if patches.device != boxes_xyxy.device or not bool(torch.isfinite(patches).all()):
        raise ValueError("invalid patch device or values")
    overlap, valid = patch_overlap_weights(boxes_xyxy)
    covered = (overlap[valid] > 0).any(0)
    usable = ~covered
    reliable = bool(usable.float().mean() >= min_uncovered_fraction)
    if not reliable:
        return patches.new_zeros(patches.shape[-1]), False
    return patches[usable].float().mean(0).to(patches.dtype), True


def relative_geometry(box_a: Tensor, box_b: Tensor) -> Tensor:
    """Signed relative center, relative log size, and IoU; translation invariant."""
    if box_a.shape != box_b.shape or box_a.shape[-1] != 4:
        raise ValueError("pair boxes must share [...,4] xyxy shape")
    a, b = box_a.float(), box_b.float()
    size_a, size_b = a[..., 2:] - a[..., :2], b[..., 2:] - b[..., :2]
    if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
        raise ValueError("nonfinite geometry")
    if not bool((size_a > 0).all() and (size_b > 0).all()):
        raise ValueError("degenerate pair boxes")
    delta = (b[..., :2] + b[..., 2:] - a[..., :2] - a[..., 2:]) / 2
    relative_log_size = torch.log(size_b / size_a)
    intersection = (torch.minimum(a[..., 2:], b[..., 2:])
                    - torch.maximum(a[..., :2], b[..., :2])).clamp_min(0).prod(-1)
    union = size_a.prod(-1) + size_b.prod(-1) - intersection
    return torch.cat((delta, relative_log_size, (intersection / union).unsqueeze(-1)), -1)
