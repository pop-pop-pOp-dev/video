"""ZeRO-3-only two-step capacity entrypoint with gathered snapshot evidence."""
from __future__ import annotations

import math
import os
import sys

import torch
from deepspeed import zero

from llava.train import train_mem_stage2_capacity_two_step as base


def _snapshot(parameter):
    """Read a complete ZeRO-3 parameter on every rank without changing it."""
    if not hasattr(parameter, "ds_id"):
        value = parameter.detach().cpu().clone()
    else:
        with zero.GatheredParameters([parameter], modifier_rank=None):
            value = parameter.detach().cpu().clone()
    if not bool(torch.isfinite(value).all().item()):
        raise RuntimeError("tracked gathered parameter contains non-finite values")
    return value


class Zero3EvidenceCallback(base.CapacityEvidenceCallback):
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        trainable = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
        lora_b = sorted(name for name in trainable if "lora_B" in name and "vision_tower" not in name)
        projector = sorted(name for name in trainable if "mm_projector" in name)
        vision_trainable = sorted(name for name, parameter in model.named_parameters()
                                  if "vision_tower" in name and parameter.requires_grad)
        if not lora_b or len(projector) < 4 or vision_trainable:
            raise RuntimeError("ZeRO-3 evidence requires LoRA-B, four projector tensors, and frozen vision")
        self.tracked = {"lora_B": [lora_b[0]], "projector": projector[:4]}
        selected = [*self.tracked["lora_B"], *self.tracked["projector"]]
        self.baseline = {name: _snapshot(trainable[name]) for name in selected}
        self.writer.write("train_begin", effective_argv=sys.argv, trainable_tensor_count=len(trainable),
                          tracked_tensors=self.tracked, vision_trainable=vision_trainable,
                          snapshot_method="deepspeed.zero.GatheredParameters(modifier_rank=None)")
        self.gradients = {}
        for category, names in self.tracked.items():
            for name in names:
                # Hooks receive the local ZeRO partition gradient; do not gather it.
                trainable[name].register_hook(self._gradient_hook(category, name))

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if self.baseline is None:
            raise RuntimeError("ZeRO-3 capacity evidence baseline was not captured")
        parameters = dict(model.named_parameters())
        changes = {}
        for category, names in self.tracked.items():
            changes[category] = {}
            for name in names:
                difference = (_snapshot(parameters[name]) - self.baseline[name]).abs()
                changes[category][name] = {"changed_elements": int(torch.count_nonzero(difference).item()),
                                           "max_abs_delta": float(difference.max().item())}
        nonzero_lr = any(rate != 0.0 for rate in self.step_lrs)
        category_changed = {category: any(item["changed_elements"] for item in records.values())
                            for category, records in changes.items()}
        gradients = self.gradients.get(state.global_step, {})
        local_gradient_norms = {category: {name: float(norm) for name, norm in records.items()}
                                for category, records in gradients.items()}
        local_gradient_positive = {category: any(norm > 0.0 for norm in gradients.get(category, {}).values())
                                   for category in self.tracked}
        if not all(local_gradient_positive.values()):
            raise RuntimeError("local ZeRO gradient hooks did not observe LoRA-B and projector gradients")
        if nonzero_lr and not all(category_changed.values()):
            raise RuntimeError("nonzero-LR ZeRO-3 step did not update gathered LoRA-B and projector evidence")
        if not nonzero_lr and any(category_changed.values()):
            raise RuntimeError("zero-LR ZeRO-3 step unexpectedly changed gathered tracked tensors")
        vision_trainable = sorted(name for name, parameter in model.named_parameters()
                                  if "vision_tower" in name and parameter.requires_grad)
        if vision_trainable:
            raise RuntimeError("vision freeze changed during ZeRO-3 diagnostic")
        engine = getattr(self.trainer, "model_wrapped", None)
        grad_norm = engine.get_global_grad_norm() if hasattr(engine, "get_global_grad_norm") else None
        if grad_norm is not None and not math.isfinite(float(grad_norm)):
            raise RuntimeError("DeepSpeed global gradient norm is non-finite")
        self.writer.write("optimizer_step", global_step=state.global_step,
                          elapsed_seconds=self._elapsed(), optimizer_learning_rates=self.step_lrs,
                          nonzero_learning_rate=nonzero_lr, category_changed=category_changed,
                          gathered_tracked_updates=changes, local_partition_gradient_norms=local_gradient_norms,
                          local_gradient_categories_positive=local_gradient_positive,
                          vision_trainable=vision_trainable,
                          gradient_norm=float(grad_norm) if grad_norm is not None else None,
                          gradient_norm_source="deepspeed.get_global_grad_norm" if grad_norm is not None else "unavailable",
                          snapshot_method="deepspeed.zero.GatheredParameters(modifier_rank=None)")

    def _elapsed(self):
        return base.time.perf_counter() - self.step_start if self.step_start else None


class Zero3InstrumentedLLaVATrainer(base.InstrumentedLLaVATrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.remove_callback(self._capacity_evidence)
        self._capacity_evidence = Zero3EvidenceCallback(self)
        base._MULTIMODAL_WRITER = self._capacity_evidence.writer
        self.add_callback(self._capacity_evidence)


def main():
    base._set_diagnostic_defaults()
    config_path = os.environ.get("REACTVAU_STAGE2_CACHE_CONFIG")
    if not config_path:
        raise RuntimeError("REACTVAU_STAGE2_CACHE_CONFIG is required")
    base.cache_entrypoint.install(config_path)
    base._install_multimodal_insertion_evidence()
    base.train_module.LLaVATrainer = Zero3InstrumentedLLaVATrainer
    base.cache_entrypoint.init_distributed_mode()
    base.train_module.train()


if __name__ == "__main__":
    main()
