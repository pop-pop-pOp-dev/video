"""NC-RTED relation-time branch inserted after the frozen ReactVAU projector.

The branch consumes only causal, pre-extracted observations.  Its sixteen output
tokens are concatenated after the existing ``mm_projector.mlp`` visual embeddings
and before ``prepare_inputs_labels_for_LLM``.  The language projection is
bias-free: quality gates the actual values reaching Slow, rather than merely an
auxiliary head.  A zero quality therefore cannot become a learned constant token.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class EvidenceOutput:
    quality: Tensor              # [B, N]
    position: Tensor             # [B, N, E, T], zero on invalid cells
    evidence_tokens: Tensor      # [B, 16, language_hidden]
    valid_blocks: Tensor         # [B, N]
    aux_valid: Tensor            # [B, N], independent teacher-support mask
    cell_valid: Tensor           # [B, N, E, T], exact observation support


class RelationTimeEvidence(nn.Module):
    """Shared A/U/S/F architecture; the selected group changes only ``auxiliary_loss``."""
    def __init__(self, feature_dim: int, language_hidden_size: int, width: int = 384, queries: int = 16, heads: int = 6):
        super().__init__()
        if width % heads: raise ValueError("width must divide transformer heads")
        self.feature_projection = nn.Linear(feature_dim, width)
        layer = nn.TransformerEncoderLayer(width, heads, dim_feedforward=width * 4, batch_first=True, activation="gelu", norm_first=True)
        self.temporal_encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.time_projection = nn.Sequential(nn.Linear(3, width), nn.GELU(), nn.Linear(width, width))
        self.quality_head = nn.Linear(width, 1)
        self.position_head = nn.Linear(width, 1)
        self.resampler_queries = nn.Parameter(torch.empty(queries, width))
        nn.init.normal_(self.resampler_queries, std=width ** -0.5)
        self.key_projection = nn.Linear(width, width, bias=False)
        self.value_projection = nn.Linear(width, width, bias=False)
        # No bias: quality truly gates values delivered to the language model.
        self.language_projection = nn.Linear(width, language_hidden_size, bias=False)

    @staticmethod
    def _time_features(observed_times: Tensor) -> Tensor:
        """Absolute observed timestamps, with no video-duration or future-frame input."""
        scaled = torch.log1p(observed_times.clamp_min(0)) / math.log1p(3600.0)
        return torch.stack((scaled, torch.sin(observed_times / 8.0), torch.cos(observed_times / 8.0)), dim=-1)

    def forward(self, features: Tensor, valid: Tensor, observed_times: Tensor, aux_valid: Tensor | None = None) -> EvidenceOutput:
        """Encode [B, blocks, candidates<=16, four cells, feature_dim] causally."""
        if features.ndim != 5: raise ValueError("features must be [B,N,E,T,D]")
        b, blocks, candidates, times, _ = features.shape
        if candidates > 16 or times != 4: raise ValueError("NC-RTED supports <=16 candidates and exactly four time cells")
        if valid.shape != (b, blocks, candidates, times): raise ValueError("valid mask shape mismatch")
        if observed_times.shape == (b, blocks, times): observed_times = observed_times.unsqueeze(2).expand(-1, -1, candidates, -1)
        elif observed_times.shape != (b, blocks, candidates, times): raise ValueError("observed_times must be [B,N,4] or [B,N,E,4]")
        valid = valid.bool()
        if not torch.isfinite(features[valid]).all(): raise ValueError("valid feature cells must be finite")
        if not torch.isfinite(observed_times[valid]).all() or (observed_times[valid] < 0).any(): raise ValueError("valid timestamps must be finite and nonnegative")
        if candidates == 0:
            empty = features.new_zeros((b, blocks, 0, times)); blocks_valid = torch.zeros((b, blocks), device=features.device, dtype=torch.bool)
            return EvidenceOutput(features.new_zeros((b, blocks)), empty, features.new_zeros((b, self.resampler_queries.shape[0], self.language_projection.out_features)), blocks_valid, blocks_valid, valid)
        # Invalid placeholders never enter learned projections or attention.
        safe_features = torch.where(valid.unsqueeze(-1), torch.nan_to_num(features), torch.zeros_like(features))
        safe_times = torch.where(valid, torch.nan_to_num(observed_times), torch.zeros_like(observed_times))
        encoded = self.feature_projection(safe_features)
        # Keep raw timestamps FP32 for long-video precision; only derived channels
        # cross into the BF16/FP32 branch dtype at the learned projection boundary.
        time_features = self._time_features(safe_times.float()).to(self.time_projection[0].weight.dtype)
        encoded = encoded + self.time_projection(time_features).to(encoded.dtype)
        # Temporal processing is shared over relationships; no relation-ID embedding exists.
        sequence = encoded.reshape(b * blocks * candidates, times, -1)
        padding = ~valid.reshape(b * blocks * candidates, times)
        safe_padding = padding.clone()
        safe_padding[padding.all(dim=1), 0] = False
        encoded = self.temporal_encoder(sequence, src_key_padding_mask=safe_padding).reshape(b, blocks, candidates, times, -1)
        encoded = encoded.masked_fill(~valid.unsqueeze(-1), 0)
        valid_blocks = valid.any(dim=(-1, -2))
        if aux_valid is None: aux_valid = torch.zeros_like(valid_blocks)
        if aux_valid.shape != valid_blocks.shape: raise ValueError("aux_valid must be [B,N]")
        aux_valid = aux_valid.bool() & valid_blocks
        denominator = valid.sum(dim=(-1, -2), keepdim=True).clamp_min(1)
        pooled = encoded.sum(dim=(-2, -3)) / denominator.squeeze((-1, -2)).unsqueeze(-1)
        # Keep quality logits/sigmoid FP32: BF16 rounds moderate logits (for
        # example +8) to an exact probability of one and destroys Bernoulli KL.
        quality_logits = F.linear(pooled.float(), self.quality_head.weight.float(), self.quality_head.bias.float()).squeeze(-1)
        quality = torch.sigmoid(quality_logits) * valid_blocks.to(torch.float32)
        position_logits = self.position_head(encoded).squeeze(-1)
        position = self._masked_softmax(position_logits, valid)
        flat_values = self.value_projection(encoded).reshape(b, blocks * candidates * times, -1)
        flat_keys = self.key_projection(encoded).reshape(b, blocks * candidates * times, -1)
        flat_position = position.reshape(b, -1)
        flat_valid = valid.reshape(b, -1)
        gate = quality.to(flat_values.dtype).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, candidates, times).reshape(b, -1)
        logits = torch.einsum("qh,bkh->bqk", self.resampler_queries, flat_keys) / math.sqrt(flat_keys.shape[-1])
        logits = logits + torch.log(flat_position.clamp_min(1e-20)).unsqueeze(1)
        logits = logits.masked_fill(~flat_valid.unsqueeze(1), float("-inf"))
        safe_logits = logits.masked_fill((~flat_valid).all(dim=1).view(b, 1, 1), 0)
        attention = torch.softmax(safe_logits.float(), dim=-1).to(flat_values.dtype)
        attention = attention * flat_valid.unsqueeze(1).to(attention.dtype)
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(attention.dtype).eps)
        # Both the position distribution (bias) and a_hat (gated values) affect Slow tokens.
        values = flat_values * gate.unsqueeze(-1)
        resampled = torch.einsum("bqk,bkh->bqh", attention, values)
        return EvidenceOutput(quality, position, self.language_projection(resampled), valid_blocks, aux_valid, valid)

    @staticmethod
    def _masked_softmax(logits: Tensor, valid: Tensor) -> Tensor:
        result = torch.zeros_like(logits)
        flat_logits, flat_valid = logits.flatten(-2), valid.flatten(-2)
        any_valid = flat_valid.any(dim=-1, keepdim=True)
        safe = flat_logits.masked_fill(~flat_valid, float("-inf"))
        safe = torch.where(any_valid, safe, torch.zeros_like(safe))
        result.copy_((torch.softmax(safe.float(), dim=-1).to(logits.dtype) * flat_valid).reshape_as(logits))
        return result

    def inject(self, original_visual_embeddings: Tensor, output: EvidenceOutput, enabled: bool = True) -> Tensor:
        """Exact bypass for disabled/no-candidate paths; original visual tokens stay byte-identical."""
        if not enabled or not bool(output.valid_blocks.any()): return original_visual_embeddings
        if original_visual_embeddings.shape[0] != 1: raise ValueError("mixed candidate/no-candidate batches require ragged integration; use batch size one")
        if original_visual_embeddings.shape[0] != output.evidence_tokens.shape[0]: raise ValueError("batch mismatch")
        return torch.cat((original_visual_embeddings, output.evidence_tokens.to(original_visual_embeddings.dtype)), dim=1)

    @staticmethod
    def auxiliary_loss(output: EvidenceOutput, quality_target: Tensor, position_target: Tensor, mode: str) -> Tensor:
        """FP32 KL; masked samples contribute zero but remain in the full batch denominator."""
        if mode not in {"A", "U", "S", "F"}: raise ValueError("mode must be A/U/S/F")
        if quality_target.shape != output.quality.shape or position_target.shape != output.position.shape: raise ValueError("teacher target shape mismatch")
        if mode == "A": return output.quality.float().sum() * 0
        valid = output.aux_valid
        if not bool(valid.any()): return output.quality.float().sum() * 0
        q = torch.where(valid, quality_target.float(), torch.zeros_like(quality_target, dtype=torch.float32))
        if not torch.isfinite(q[valid]).all() or ((q[valid] < 0) | (q[valid] > 1)).any(): raise ValueError("invalid quality target")
        prediction = output.quality.float().clamp(1e-6, 1 - 1e-6)
        q_safe = torch.where(q > 0, q, torch.ones_like(q))
        nq = 1 - q; nq_safe = torch.where(nq > 0, nq, torch.ones_like(nq))
        quality_kl = torch.where(q > 0, q * (q_safe.log() - prediction.log()), torch.zeros_like(q)) + torch.where(nq > 0, nq * (nq_safe.log() - (1 - prediction).log()), torch.zeros_like(q))
        quality_kl = (quality_kl * valid).sum() / output.quality.shape[0]
        if mode == "S": return quality_kl
        target = position_target.float()
        block_mask = valid.unsqueeze(-1).unsqueeze(-1).expand_as(target)
        if not torch.isfinite(target[block_mask]).all(): raise ValueError("invalid position target")
        invalid_cells = target[~output.cell_valid]
        if (invalid_cells[torch.isfinite(invalid_cells)] > 0).any(): raise ValueError("teacher position mass on invalid observation cell")
        target = torch.where(block_mask, target, torch.zeros_like(target))
        if (target < 0).any() or not torch.allclose(target.sum(dim=(-1, -2))[valid], torch.ones_like(target.sum(dim=(-1, -2))[valid]), atol=1e-6): raise ValueError("position targets must be nonnegative normalized distributions")
        predicted = output.position.float().clamp_min(1e-20)
        target_safe = torch.where(target > 0, target, torch.ones_like(target))
        position_kl = torch.where(target > 0, target * (target_safe.log() - predicted.log()), torch.zeros_like(target)).sum(dim=(-1, -2))
        position_kl = (q * position_kl * valid).sum() / output.quality.shape[0]
        return quality_kl + position_kl
