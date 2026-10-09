"""Two-step Stage2 capacity entrypoint with evidence-only instrumentation.

This imports the existing cache entrypoint's resolver setup and the official
training function. It does not alter the model, data, loss, optimizer, or
checkpoint implementation.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import torch
from transformers import TrainerCallback

from llava.constants import IMAGE_TOKEN_INDEX
from llava.model.llava_arch import LlavaMetaForCausalLM
from llava.train import train as train_module
from llava.train import train_mem_stage2_cache as cache_entrypoint


def _argument_value(name: str):
    for index, value in enumerate(sys.argv):
        if value == name and index + 1 < len(sys.argv):
            return sys.argv[index + 1]
        if value.startswith(name + "="):
            return value.split("=", 1)[1]
    return None


def _set_diagnostic_defaults():
    max_steps = _argument_value("--max_steps")
    if max_steps is None:
        sys.argv.extend(["--max_steps", "2"])
    elif int(max_steps) != 2:
        raise RuntimeError("two-step capacity entrypoint requires --max_steps 2")
    save_steps = _argument_value("--save_steps")
    if save_steps is None:
        sys.argv.extend(["--save_steps", "2"])
    elif int(save_steps) != 2:
        raise RuntimeError("two-step capacity entrypoint requires --save_steps 2")
    if _argument_value("--logging_steps") is None:
        sys.argv.extend(["--logging_steps", "1"])


def _rank():
    return int(os.environ.get("RANK", "0"))


class EvidenceWriter:
    def __init__(self, output_dir: str):
        self.path = Path(output_dir) / "capacity_two_step_evidence" / f"rank{_rank()}.jsonl"

    def write(self, event: str, **fields):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"event": event, "monotonic_seconds": time.perf_counter(), "rank": _rank(), **fields}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def _tensor_shape(value):
    return list(value.shape) if isinstance(value, torch.Tensor) else None


class CapacityEvidenceCallback(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer
        self.writer = EvidenceWriter(trainer.args.output_dir)
        self.baseline = None
        self.step_start = None

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        trainable = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
        lora_b = sorted(name for name in trainable if "lora_B" in name and "vision_tower" not in name)
        projector = sorted(name for name in trainable if "mm_projector" in name)
        vision_trainable = sorted(name for name, parameter in model.named_parameters()
                                  if "vision_tower" in name and parameter.requires_grad)
        if not lora_b:
            raise RuntimeError("capacity evidence requires at least one trainable non-vision LoRA B tensor")
        if len(projector) < 4:
            raise RuntimeError("capacity evidence requires at least four trainable projector tensors")
        if vision_trainable:
            raise RuntimeError("capacity evidence found trainable vision parameters")
        self.tracked = {"lora_B": [lora_b[0]], "projector": projector[:4]}
        selected = [*self.tracked["lora_B"], *self.tracked["projector"]]
        self.baseline = {name: trainable[name].detach().cpu().clone() for name in selected}
        if any(not bool(torch.isfinite(value).all().item()) for value in self.baseline.values()):
            raise RuntimeError("tracked trainable baseline contains non-finite weights")
        self.writer.write("train_begin", effective_argv=sys.argv, trainable_tensor_count=len(trainable),
                          tracked_tensors=self.tracked, vision_trainable=vision_trainable)
        self.gradients = {}
        for category, names in self.tracked.items():
            for name in names:
                trainable[name].register_hook(self._gradient_hook(category, name))

    def _gradient_hook(self, category, name):
        def capture(gradient):
            norm = float(gradient.detach().float().norm().cpu().item())
            if not math.isfinite(norm):
                raise RuntimeError(f"tracked gradient is non-finite: {name}")
            step = self.trainer.state.global_step + 1
            self.gradients.setdefault(step, {}).setdefault(category, {})[name] = norm
        return capture

    def on_step_begin(self, args, state, control, **kwargs):
        self.step_start = time.perf_counter()
        self.step_lrs = [float(group["lr"]) for group in self.trainer.optimizer.param_groups]
        self.writer.write("step_begin", upcoming_global_step=state.global_step + 1,
                          optimizer_learning_rates=self.step_lrs)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if self.baseline is None:
            raise RuntimeError("capacity evidence baseline was not captured")
        parameters = dict(model.named_parameters())
        changes = {}
        for category, names in self.tracked.items():
            changes[category] = {}
            for name in names:
                current = parameters[name].detach().cpu()
                if not bool(torch.isfinite(current).all().item()):
                    raise RuntimeError(f"tracked trainable weight is non-finite: {name}")
                difference = (current - self.baseline[name]).abs()
                changes[category][name] = {"changed_elements": int(torch.count_nonzero(difference).item()),
                                           "max_abs_delta": float(difference.max().item())}
        nonzero_lr = any(rate != 0.0 for rate in self.step_lrs)
        category_changed = {category: any(item["changed_elements"] for item in records.values())
                            for category, records in changes.items()}
        gradients = self.gradients.get(state.global_step, {})
        gradient_categories = {category: {name: float(norm) for name, norm in records.items()}
                               for category, records in gradients.items()}
        gradient_positive = {category: any(norm > 0.0 for norm in gradients.get(category, {}).values())
                             for category in self.tracked}
        if not all(gradient_positive.values()):
            raise RuntimeError("backward pass did not produce finite nonzero LoRA-B and projector gradients")
        if nonzero_lr and not all(category_changed.values()):
            raise RuntimeError("nonzero-LR optimizer step did not update both LoRA-B and projector evidence categories")
        if not nonzero_lr and any(category_changed.values()):
            raise RuntimeError("zero-LR optimizer step unexpectedly changed tracked trainables")
        vision_trainable = sorted(name for name, parameter in model.named_parameters()
                                  if "vision_tower" in name and parameter.requires_grad)
        if vision_trainable:
            raise RuntimeError("vision freeze changed during capacity diagnostic")
        engine = getattr(self.trainer, "model_wrapped", None)
        grad_norm = None
        grad_norm_source = "unavailable"
        if hasattr(engine, "get_global_grad_norm"):
            value = engine.get_global_grad_norm()
            if value is not None:
                grad_norm = float(value)
                if not math.isfinite(grad_norm):
                    raise RuntimeError("DeepSpeed global gradient norm is non-finite")
                grad_norm_source = "deepspeed.get_global_grad_norm"
        self.writer.write("optimizer_step", global_step=state.global_step,
                          elapsed_seconds=time.perf_counter() - self.step_start if self.step_start else None,
                          optimizer_learning_rates=self.step_lrs, nonzero_learning_rate=nonzero_lr,
                          category_changed=category_changed, tracked_updates=changes,
                          tracked_gradient_norms=gradient_categories, gradient_categories_positive=gradient_positive,
                          vision_trainable=vision_trainable, gradient_norm=grad_norm,
                          gradient_norm_source=grad_norm_source,
                          model_wrapped_type=type(engine).__name__ if engine is not None else None)

    def on_train_end(self, args, state, control, **kwargs):
        if state.global_step != 2:
            raise RuntimeError(f"two-step capacity diagnostic ended at {state.global_step} steps")

class InstrumentedLLaVATrainer(train_module.LLaVATrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._capacity_evidence = CapacityEvidenceCallback(self)
        global _MULTIMODAL_WRITER
        _MULTIMODAL_WRITER = self._capacity_evidence.writer
        self.add_callback(self._capacity_evidence)

    def training_step(self, model, inputs, num_items_in_batch=None):
        input_ids = inputs.get("input_ids")
        if not isinstance(input_ids, torch.Tensor):
            raise RuntimeError("capacity evidence requires tensor input_ids")
        visual_tokens = (input_ids == IMAGE_TOKEN_INDEX).sum(dim=1).detach().cpu().tolist()
        if any(count != 1 for count in visual_tokens):
            raise RuntimeError(f"each capacity batch video requires one IMAGE_TOKEN_INDEX, got {visual_tokens}")
        images = inputs.get("images")
        if not isinstance(images, list) or not images:
            raise RuntimeError("capacity evidence requires real visual tensors")
        image_shapes = [_tensor_shape(image) for image in images]
        if any(shape is None or len(shape) != 4 or shape[0] <= 0 for shape in image_shapes):
            raise RuntimeError(f"invalid real video frame tensor shapes: {image_shapes}")
        self._capacity_evidence.writer.write("batch", upcoming_global_step=self.state.global_step + 1,
                                             image_token_counts=visual_tokens, image_shapes=image_shapes,
                                             image_sizes=inputs.get("image_sizes"), modalities=inputs.get("modalities"),
                                             pg_score_count=len(inputs.get("pg_scores") or []))
        loss = super().training_step(model, inputs, num_items_in_batch)
        if not isinstance(loss, torch.Tensor) or not bool(torch.isfinite(loss).all().item()):
            raise RuntimeError("capacity diagnostic training loss is non-finite")
        self._capacity_evidence.writer.write("training_loss", upcoming_global_step=self.state.global_step + 1,
                                             finite=True, loss=float(loss.detach().float().cpu().item()))
        return loss


_MULTIMODAL_WRITER = None


def _install_multimodal_insertion_evidence():
    def wrap(method_name):
        original = getattr(LlavaMetaForCausalLM, method_name)

        def instrumented(self, *args, **kwargs):
            result = original(self, *args, **kwargs)
            input_ids = kwargs.get("input_ids", args[0] if args else None)
            attention_mask = kwargs.get("attention_mask", args[2] if len(args) > 2 else None)
            labels = kwargs.get("labels", args[4] if len(args) > 4 else None)
            if not isinstance(input_ids, torch.Tensor) or not isinstance(labels, torch.Tensor):
                return result
            writer = _MULTIMODAL_WRITER
            if writer is None:
                raise RuntimeError("multimodal evidence writer was not installed")
            _, _, output_mask, _, inputs_embeds, output_labels = result
            if not isinstance(inputs_embeds, torch.Tensor) or not bool(torch.isfinite(inputs_embeds).all().item()):
                raise RuntimeError("multimodal preparation did not return finite embeddings")
            if not isinstance(output_labels, torch.Tensor) or not isinstance(output_mask, torch.Tensor):
                raise RuntimeError("multimodal preparation did not return labels and attention mask")
            input_mask = torch.ones_like(input_ids, dtype=torch.bool) if attention_mask is None else attention_mask.bool()
            input_lengths = input_mask.sum(dim=1).detach().cpu().tolist()
            output_lengths = output_mask.bool().sum(dim=1).detach().cpu().tolist()
            image_tokens = (input_ids == IMAGE_TOKEN_INDEX).sum(dim=1).detach().cpu().tolist()
            input_ignored = (labels == -100).sum(dim=1).detach().cpu().tolist()
            visual_masks = (output_labels == -100).sum(dim=1).detach().cpu().tolist()
            if any(tokens != 1 for tokens in image_tokens) or any(after <= before for before, after in zip(input_lengths, output_lengths)):
                raise RuntimeError("multimodal visual token was not expanded into a longer embedding sequence")
            if any(after <= before for before, after in zip(input_ignored, visual_masks)):
                raise RuntimeError("multimodal visual embeddings are not paired with masked labels")
            writer.write("multimodal_insertion", method=method_name, image_token_counts=image_tokens,
                         input_nonpadding_lengths=input_lengths, output_nonpadding_lengths=output_lengths,
                         inputs_embeds_shape=_tensor_shape(inputs_embeds), input_ignored_label_counts=input_ignored,
                         masked_label_counts=visual_masks)
            return result

        setattr(LlavaMetaForCausalLM, method_name, instrumented)

    wrap("prepare_inputs_labels_for_multimodal")
    wrap("prepare_inputs_labels_for_LLM")


def main():
    _set_diagnostic_defaults()
    config_path = os.environ.get("REACTVAU_STAGE2_CACHE_CONFIG")
    if not config_path:
        raise RuntimeError("REACTVAU_STAGE2_CACHE_CONFIG is required")
    cache_entrypoint.install(config_path)
    _install_multimodal_insertion_evidence()
    train_module.LLaVATrainer = InstrumentedLLaVATrainer
    cache_entrypoint.init_distributed_mode()
    train_module.train()


if __name__ == "__main__":
    main()
