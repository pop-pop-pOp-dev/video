"""Mixed-length ZeRO-3 runtime and periodic-checkpoint diagnostic entrypoint."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch.distributed as dist
from transformers import TrainerCallback, set_seed

from llava.train import train_mem_stage2_capacity_zero3_two_step as base


def _argument_value(name: str):
    for index, value in enumerate(sys.argv):
        if value == name and index + 1 < len(sys.argv):
            return sys.argv[index + 1]
        if value.startswith(name + "="):
            return value.split("=", 1)[1]
    return None


def _set_seed_42_before_model_construction():
    supplied = _argument_value("--seed")
    if supplied is None:
        sys.argv.extend(["--seed", "42"])
    elif int(supplied) != 42:
        raise RuntimeError("mixed runtime diagnostic requires seed 42")
    set_seed(42)


class RuntimeCheckpointTerminalCallback(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer

    def on_train_end(self, args, state, control, **kwargs):
        if state.global_step != 2:
            raise RuntimeError(f"mixed runtime diagnostic ended at {state.global_step}, expected 2")
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        checkpoint = Path(args.output_dir) / "checkpoint-2"
        deepspeed_checkpoint = checkpoint / "global_step2"
        rank = int(os.environ.get("RANK", "0"))
        required = {
            "adapter_model": checkpoint / "adapter_model.safetensors",
            "adapter_config": checkpoint / "adapter_config.json",
            "scheduler_state": checkpoint / "scheduler.pt",
            "rank_rng_state": checkpoint / f"rng_state_{rank}.pth",
            "trainer_state": checkpoint / "trainer_state.json",
            "rank_model_state": deepspeed_checkpoint / f"zero_pp_rank_{rank}_mp_rank_00_model_states.pt",
            "rank_optimizer_state": deepspeed_checkpoint / f"bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt",
        }
        absent = [name for name, path in required.items() if not path.is_file() or path.stat().st_size == 0]
        if absent:
            raise RuntimeError(f"periodic checkpoint is incomplete on rank {rank}: {absent}")
        with required["trainer_state"].open("r", encoding="utf-8") as handle:
            saved_state = json.load(handle)
        if saved_state.get("global_step") != 2:
            raise RuntimeError(f"checkpoint trainer_state has global_step={saved_state.get('global_step')!r}")

        self.trainer._capacity_evidence.writer.write(
            "runtime_periodic_checkpoint_terminal",
            purpose="RUNTIME_AND_PERIODIC_CHECKPOINT_ONLY",
            global_step=state.global_step,
            checkpoint=str(checkpoint),
            checkpoint_files={name: {"path": str(path), "bytes": path.stat().st_size}
                              for name, path in required.items()},
            full_final_export_skipped=True,
        )
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
        raise SystemExit(0)


class MixedRuntimeCheckpointTrainer(base.Zero3InstrumentedLLaVATrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_callback(RuntimeCheckpointTerminalCallback(self))


def main():
    _set_seed_42_before_model_construction()
    base.Zero3InstrumentedLLaVATrainer = MixedRuntimeCheckpointTrainer
    base.main()


if __name__ == "__main__":
    main()
