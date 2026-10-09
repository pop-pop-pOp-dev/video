"""Connect evidence to the inherited Slow multimodal preparation and task loss.

Inputs are frozen, causal cache products. This module neither builds memories
nor changes Fast triggering/fusion. Its disabled path calls the same inherited
preparation with the original visual tensor, without constructing extra tokens.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from .model import EvidenceOutput, RelationTimeEvidence


@dataclass(frozen=True)
class ObservationBatch:
    features: Tensor
    valid: Tensor
    observed_times: Tensor


@dataclass(frozen=True)
class TeacherBatch:
    quality: Tensor
    position: Tensor
    eligible: Tensor


@dataclass(frozen=True)
class SlowInputs:
    input_ids: Tensor
    visual_embeddings: Tensor  # [1, existing_tokens, language_hidden], after old projector
    images: list[Tensor]       # inherited preparation's image shape metadata
    image_sizes: list[tuple[int, int]]
    labels: Tensor | None = None
    attention_mask: Tensor | None = None
    position_ids: Tensor | None = None


@dataclass
class PreparedSlow:
    arguments: dict[str, Any]
    evidence: EvidenceOutput | None
    visual_embeddings: Tensor


@dataclass
class TaskLoss:
    loss: Tensor
    task_loss: Tensor
    auxiliary_loss: Tensor
    evidence: EvidenceOutput | None


class EvidenceSlowBridge(nn.Module):
    def __init__(self, slow: nn.Module, evidence: RelationTimeEvidence):
        super().__init__()
        self.slow = slow
        self.evidence = evidence
        reference = self.raw_slow.get_input_embeddings().weight
        if reference.dtype not in {torch.bfloat16, torch.float32}:
            raise ValueError("Slow must use the audited BF16 or CPU-test FP32 policy")
        self.evidence.to(device=reference.device, dtype=reference.dtype)

    @property
    def raw_slow(self):
        return self.slow.get_base_model() if hasattr(self.slow, "get_base_model") else self.slow

    def prepare(self, inputs: SlowInputs, observations: ObservationBatch | None,
                *, enabled: bool = True, teacher: TeacherBatch | None = None) -> PreparedSlow:
        raw = self.raw_slow
        if getattr(raw.config, "mm_patch_merge_type", "flat") != "flat":
            raise ValueError("bridge requires the audited inherited flat SigLIP merge")
        if "anyres" in getattr(raw.config, "frame_aspect_ratio", "square"):
            raise ValueError("unaudited frame preprocessing")
        if inputs.input_ids.ndim != 2 or inputs.input_ids.shape[0] != 1:
            raise ValueError("the fixed recipe uses microbatch one")
        if inputs.visual_embeddings.ndim != 3 or inputs.visual_embeddings.shape[0] != 1:
            raise ValueError("frozen visual embeddings must be [1,L,H]")
        if inputs.visual_embeddings.requires_grad:
            raise ValueError("old visual memory/projector must be frozen")
        reference = raw.get_input_embeddings().weight
        if inputs.visual_embeddings.device != reference.device or inputs.visual_embeddings.dtype != reference.dtype:
            raise ValueError("frozen visual embeddings must match Slow device/dtype without implicit recasting")
        if inputs.input_ids.device != reference.device:
            raise ValueError("text and Slow devices differ")
        if len(inputs.images) != 1 or len(inputs.image_sizes) != 1:
            raise ValueError("one observed video per sample is required")
        mask = inputs.attention_mask
        if mask is not None and mask.shape != inputs.input_ids.shape:
            raise ValueError("attention mask shape mismatch")
        active = torch.ones_like(inputs.input_ids, dtype=torch.bool) if mask is None else mask.bool()
        # This is the pinned llava.constants.IMAGE_TOKEN_INDEX, not a vocabulary ID.
        if int(((inputs.input_ids == -200) & active).sum()) != 1:
            raise ValueError("exactly one active inherited image sentinel is required")
        visual, output = inputs.visual_embeddings, None
        if enabled:
            if observations is None:
                raise ValueError("enabled evidence requires explicit observations, including an empty mask")
            if observations.features.shape[0] != 1:
                raise ValueError("observation batch mismatch")
            parameter = next(self.evidence.parameters())
            if parameter.device != reference.device or parameter.dtype != reference.dtype:
                raise ValueError("evidence and Slow device/dtype policy differs")
            output = self.evidence(observations.features.to(device=parameter.device, dtype=parameter.dtype),
                                   observations.valid.to(device=parameter.device),
                                   observations.observed_times.to(device=parameter.device, dtype=torch.float32),
                                   None if teacher is None else teacher.eligible.to(device=parameter.device))
            visual = self.evidence.inject(visual, output)
        expected_length = int(active.sum()) - 1 + visual.shape[1]
        limit = getattr(raw.config, "tokenizer_model_max_length", None)
        if limit is not None and expected_length > limit:
            raise ValueError("inherited preparation would truncate tokens; full input is required")
        result = raw.prepare_inputs_labels_for_LLM(
            inputs.input_ids, inputs.position_ids, mask, None, inputs.labels,
            inputs.images, [visual], ["video"], image_sizes=inputs.image_sizes,
        )
        ids, positions, attention, past, embeds, labels = result
        if embeds is None or embeds.shape[1] != expected_length:
            raise RuntimeError("inherited preparation bypassed or truncated visual insertion")
        if inputs.labels is not None:
            if labels is None or int((labels != -100).sum()) != int(((inputs.labels != -100) & active).sum()):
                raise RuntimeError("supervised targets changed during visual insertion")
        return PreparedSlow(dict(input_ids=ids, position_ids=positions, attention_mask=attention,
                                 past_key_values=past, inputs_embeds=embeds, labels=labels), output, visual)

    def forward(self, inputs: SlowInputs, observations: ObservationBatch,
                *, task: str, group: str, teacher: TeacherBatch | None = None) -> TaskLoss:
        if task not in {"detection", "caption"} or group not in {"A", "U", "S", "F"}:
            raise ValueError("unknown task or group")
        if task == "caption" and teacher is not None:
            raise ValueError("caption instructions cannot receive teacher supervision")
        if task == "detection" and observations.features.shape[1] != 1:
            raise ValueError("detection auxiliary supervision covers one legal eight-second window")
        if task == "detection" and group != "A" and teacher is None:
            raise ValueError("distilled groups require an explicit teacher, including rejection masks")
        if inputs.labels is None or not bool((inputs.labels != -100).any()):
            raise ValueError("the inherited task requires actual supervised target tokens")
        prepared = self.prepare(inputs, observations, teacher=teacher)
        result = self.slow(**prepared.arguments, use_cache=False, return_dict=True)
        if result.loss is None or result.loss.ndim != 0 or not bool(torch.isfinite(result.loss)):
            raise FloatingPointError("nonfinite or missing inherited task loss")
        aux = result.loss.float() * 0
        if task == "detection" and teacher is not None:
            aux = self.evidence.auxiliary_loss(prepared.evidence,
                teacher.quality.to(device=result.loss.device, dtype=torch.float32),
                teacher.position.to(device=result.loss.device, dtype=torch.float32), group)
        if not bool(torch.isfinite(aux)):
            raise FloatingPointError("nonfinite evidence loss")
        return TaskLoss(result.loss + .1 * aux, result.loss, aux, prepared.evidence)

    @torch.no_grad()
    def generate(self, inputs: SlowInputs, observations: ObservationBatch | None,
                 *, enabled: bool, generation_config: dict[str, Any]):
        """Use pinned Qwen generation with all inherited decoding arguments supplied.

        Do not call LlavaQwen.generate: it rejects inputs_embeds and would rebuild
        visual memories. The Qwen parent is also what original Slow uses here.
        """
        from transformers import Qwen2ForCausalLM
        if any(module.training for module in self.modules()):
            raise ValueError("generation requires bridge.eval() before preparing inputs")
        prepared = self.prepare(inputs, observations, enabled=enabled)
        if inputs.labels is not None:
            raise ValueError("generation inputs must not contain training answers")
        forbidden = set(generation_config) & set(prepared.arguments)
        if forbidden:
            raise ValueError(f"decoding may not replace prepared inputs: {sorted(forbidden)}")
        if not isinstance(self.raw_slow, Qwen2ForCausalLM):
            raise TypeError("generation is bound to inherited Qwen2")
        args = {key: prepared.arguments[key] for key in ("position_ids", "attention_mask", "inputs_embeds")}
        return Qwen2ForCausalLM.generate(self.raw_slow, **args, **generation_config)
