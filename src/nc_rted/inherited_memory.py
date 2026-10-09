"""Reuse the two existing visual-memory paths without a replacement memory algorithm.

Caption training uses the original projector forward (including its existing
spatial/time position additions). Detection uses its streaming MemoryManager and
only the original projector MLP. These paths intentionally remain distinct.
"""
from __future__ import annotations

import math
import torch
from torch import Tensor

from .task_inputs import TaskInputError


def _raw(model):
    return model.get_base_model() if hasattr(model, "get_base_model") else model


def _projector(model):
    raw = _raw(model)
    projector = raw.get_model().mm_projector
    if not hasattr(projector, "mlp") or any(p.requires_grad for p in projector.parameters()):
        raise TaskInputError("the complete inherited projector must be frozen")
    return raw, projector


def _ordinary_frozen(value: Tensor) -> Tensor:
    # Inference tensors cannot be saved for LoRA backward. Produce an ordinary
    # tensor of the original dtype, without changing the cached source tensor.
    with torch.inference_mode(False), torch.no_grad():
        result = value.clone().detach()
    if result.ndim != 3 or result.shape[0] != 1 or not bool(torch.isfinite(result).all()):
        raise TaskInputError("invalid original memory output")
    return result


def caption_memory_from_patches(model, patches: Tensor, aligned_pg_scores: list[float]) -> Tensor:
    """Patches must be the original frozen SigLIP outputs in original frame order."""
    raw, projector = _projector(model)
    local = getattr(raw.config, "mm_local_num_frames", None)
    if local != 1 or patches.ndim != 3 or patches.shape[1:] != (729, 1152) or not len(patches):
        raise TaskInputError("inherited caption memory requires original [frames,729,1152] patches")
    if patches.requires_grad or not bool(torch.isfinite(patches).all()):
        raise TaskInputError("caption patches must be finite and frozen")
    if len(aligned_pg_scores) != len(patches) or any(not math.isfinite(s) or not 0 <= s <= 1 for s in aligned_pg_scores):
        raise TaskInputError("every original caption frame requires its existing aligned Fast score")
    parameter = next(projector.mlp.parameters())
    if patches.dtype != parameter.dtype or patches.device != parameter.device:
        raise TaskInputError("cached patches and original projector must retain the same dtype/device")
    # The original encode_image_video_memory_batch groups each video by local=1
    # and calls this exact forward. Do not recreate MemoryManager or positions.
    with torch.no_grad():
        result = projector(patches, local_num_frames=1, pg_scores=aligned_pg_scores)
    return _ordinary_frozen(result)


def detection_memory_from_stream(model, memory_manager, *, rt_anomaly_tokens: Tensor | None = None) -> Tensor:
    """Caller advances the original memory/trigger/fusion loop; no tokens are written back."""
    _, projector = _projector(model)
    with torch.no_grad():
        memory = memory_manager.get_memory_tokens(rt_anomaly_tokens=rt_anomaly_tokens)
        parameter = next(projector.mlp.parameters())
        if memory.requires_grad or memory.dtype != parameter.dtype or memory.device != parameter.device:
            raise TaskInputError("streaming memory must be frozen with unchanged original dtype/device")
        result = projector.mlp(memory)
    return _ordinary_frozen(result)
