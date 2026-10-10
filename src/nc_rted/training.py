"""Fixed single-device incremental training kernel, independent of job scheduling.

Sample construction/teacher lookup stays outside this loop. A worker must bind
their hashes and acceptance evidence before calling it for a formal run.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import time
import uuid
from typing import Callable

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class Recipe:
    updates: int = 1000
    accumulation: int = 8
    new_lr: float = 1e-4
    lora_lr: float = 5e-6
    weight_decay: float = .01
    warmup: int = 50
    clip: float = 1.
    save_interval: int = 50


def seed_run(seed: int) -> None:
    """Call before constructing new modules; groups never alter this seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_order(sample_ids: list[str], seed: int) -> list[str]:
    if not sample_ids or len(sample_ids) != len(set(sample_ids)):
        raise ValueError("sample IDs must be nonempty and unique")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return [sample_ids[i] for i in torch.randperm(len(sample_ids), generator=generator).tolist()]


def order_sha256(order: list[str]) -> str:
    return hashlib.sha256(json.dumps(order, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def publish_progress(path: str | Path, report: dict) -> None:
    """Atomic update-boundary progress for queue watchdogs; not a checkpoint."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}")
    with temporary.open("x") as stream:
        json.dump(dict(report, counter=report["update"], written_at=time.time()), stream, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, target)
    descriptor = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def trainable_groups(model: nn.Module, recipe: Recipe) -> list[dict]:
    groups = {"new": [], "lora": []}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("evidence."):
            groups["new"].append(parameter)
        elif name.startswith("slow.") and (".lora_A." in name or ".lora_B." in name):
            groups["lora"].append(parameter)
        else:
            raise ValueError(f"unexpected trainable parameter: {name}")
    if not all(groups.values()):
        raise ValueError("both new evidence parameters and inherited Slow LoRA must train")
    return [dict(params=groups["new"], lr=recipe.new_lr, name="new"),
            dict(params=groups["lora"], lr=recipe.lora_lr, name="lora")]


class IncrementalTrainer:
    def __init__(self, model: nn.Module, sample_ids: list[str], seed: int,
                 recipe: Recipe = Recipe()):
        if min(recipe.updates, recipe.accumulation, recipe.save_interval) < 1:
            raise ValueError("positive updates/accumulation/save interval required")
        if not 0 <= recipe.warmup < recipe.updates:
            raise ValueError("invalid warmup")
        if len(sample_ids) != recipe.updates * recipe.accumulation:
            raise ValueError("fixed single-pass recipe must consume every sample exactly once")
        self.model, self.recipe, self.seed = model, recipe, seed
        self.order = sample_order(sample_ids, seed)
        self.completed_updates = 0
        self.at_update_boundary = True
        groups = trainable_groups(model, recipe)
        self.forward_parameters = {n: p for n, p in model.named_parameters() if p.requires_grad}
        if any(p.dtype not in {torch.bfloat16, torch.float32} for p in self.forward_parameters.values()):
            raise ValueError("incremental training supports BF16 or FP32 forward parameters")
        # BF16 AdamW loses updates smaller than one representable weight step.
        # Keep inherited forward tensors unchanged while accumulating updates and
        # moments in FP32. Never recreate masters from rounded weights on resume.
        self.master_parameters = {n: nn.Parameter(p.detach().float().clone())
                                  for n, p in self.forward_parameters.items()}
        by_id = {id(p): self.master_parameters[n] for n, p in self.forward_parameters.items()}
        names_by_id = {id(p): n for n, p in self.forward_parameters.items()}
        for group in groups:
            group["parameter_names"] = [names_by_id[id(p)] for p in group["params"]]
            group["params"] = [by_id[id(p)] for p in group["params"]]
        self.optimizer = torch.optim.AdamW(groups,
                                          weight_decay=recipe.weight_decay)
        # Explicit linear warmup then linear decay, in effective update units.
        def scale(step):
            if step < recipe.warmup:
                return float(step + 1) / max(1, recipe.warmup)
            return max(0., (recipe.updates - step) / max(1, recipe.updates - recipe.warmup))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, scale)

    @torch.no_grad()
    def sync_forward_weights(self) -> None:
        for name, parameter in self.forward_parameters.items():
            parameter.copy_(self.master_parameters[name].to(parameter.dtype))

    @property
    def cursor(self) -> int:
        return self.completed_updates * self.recipe.accumulation

    def metadata(self) -> dict:
        return dict(seed=self.seed, recipe=asdict(self.recipe), order=self.order,
                    order_sha256=order_sha256(self.order), completed_updates=self.completed_updates,
                    cursor=self.cursor)

    def run(self, loss_for_sample: Callable[[str], torch.Tensor], *, save=None, progress=None,
            stop_after: int | None = None) -> None:
        target = self.recipe.updates if stop_after is None else min(stop_after, self.recipe.updates)
        if target < self.completed_updates:
            raise ValueError("cannot rewind completed updates")
        self.model.train()
        while self.completed_updates < target:
            report = self.step(loss_for_sample)
            if save and (self.completed_updates % self.recipe.save_interval == 0 or
                         self.completed_updates == self.recipe.updates):
                save(self, final=self.completed_updates == self.recipe.updates)
            if progress:
                progress(report)

    def step(self, loss_for_sample: Callable[[str], torch.Tensor]) -> dict:
        """Execute one complete optimizer update; shared by serial and bundled runs."""
        if not self.at_update_boundary:
            raise RuntimeError("previous update did not complete; restore a committed checkpoint")
        self.model.train(); self.at_update_boundary = False
        self.model.zero_grad(set_to_none=True); self.optimizer.zero_grad(set_to_none=True)
        losses, parameters = [], list(self.master_parameters.values())
        start = self.cursor
        for sid in self.order[start:start + self.recipe.accumulation]:
            loss = loss_for_sample(sid)
            if loss.ndim != 0 or not bool(torch.isfinite(loss)):
                raise FloatingPointError("nonfinite task loss; block run without optimizer update")
            (loss / self.recipe.accumulation).backward()
            for name, parameter in self.forward_parameters.items():
                if parameter.grad is None: continue
                if not bool(torch.isfinite(parameter.grad).all()):
                    raise FloatingPointError("nonfinite gradient; block run without optimizer update")
                master, gradient = self.master_parameters[name], parameter.grad.detach().float()
                if master.grad is None: master.grad = gradient.clone()
                else: master.grad.add_(gradient)
            self.model.zero_grad(set_to_none=True); losses.append(float(loss.detach()))
        gradient_groups = {"new": [], "lora": []}
        for name, parameter in self.master_parameters.items():
            bucket = "new" if name.startswith("evidence.") else "lora"
            if parameter.grad is not None:
                gradient_groups[bucket].append(parameter.grad.detach())
        gradient_summary = {name: dict(nonzero=sum(int(torch.count_nonzero(value)) for value in values),
                                       l1=float(sum(value.abs().sum() for value in values)))
                            for name, values in gradient_groups.items()}
        norm = torch.nn.utils.clip_grad_norm_(parameters, self.recipe.clip, error_if_nonfinite=True)
        if not any(p.grad is not None for p in parameters): raise RuntimeError("no trainable gradients")
        self.optimizer.step()
        if any(not bool(torch.isfinite(p).all()) for p in parameters): raise FloatingPointError("nonfinite optimizer parameter; block run")
        self.sync_forward_weights(); self.scheduler.step(); self.completed_updates += 1
        self.model.zero_grad(set_to_none=True); self.optimizer.zero_grad(set_to_none=True); self.at_update_boundary = True
        return dict(update=self.completed_updates, cursor=self.cursor, mean_loss=sum(losses) / len(losses),
                    gradient_norm=float(norm), gradient_summary=gradient_summary)
