"""Formal Stage2 entrypoint with explicit deterministic initialization seed.

This leaves all diagnostic and released entrypoints untouched.  The seed is
applied before model and LoRA construction, rather than at Trainer creation.
"""
import os
import sys

from transformers import set_seed

from llava.dist_utils import init_distributed_mode
from llava.train import train as train_module
from llava.train.train_mem_stage2_cache import install


def explicit_seed(argv):
    values = [argv[index + 1] for index, value in enumerate(argv[:-1]) if value == "--seed"]
    if len(values) != 1:
        raise ValueError("formal Stage2 requires exactly one explicit --seed VALUE")
    try:
        return int(values[0])
    except ValueError as exc:
        raise ValueError("--seed must be an integer") from exc


def main(argv=None):
    seed = explicit_seed(sys.argv[1:] if argv is None else argv)
    set_seed(seed)
    config_path = os.environ.get("REACTVAU_STAGE2_CACHE_CONFIG")
    if not config_path:
        raise RuntimeError("REACTVAU_STAGE2_CACHE_CONFIG is required")
    install(config_path)
    init_distributed_mode()
    train_module.train()


if __name__ == "__main__":
    main()
