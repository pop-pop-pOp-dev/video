"""Explicit review-required Stage2 entrypoint; leaves train_mem.py unchanged."""
import importlib.util
import json
import os
from pathlib import Path

from llava.train import train as train_module
from llava.train.reactvau_stage2_cache_adapter import FailClosedStage2DatasetMixin
from llava.dist_utils import init_distributed_mode


def _resolver_module():
    location = os.environ.get("REACTVAU_STAGE2_CACHE_MODULE")
    if not location:
        raise RuntimeError("REACTVAU_STAGE2_CACHE_MODULE is required")
    spec = importlib.util.spec_from_file_location("reactvau_stage2_cache_serial", location)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def install(config_path):
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    resolver = _resolver_module().Cache(config)
    original = train_module.LazySupervisedDataset

    class FailClosedStage2Dataset(FailClosedStage2DatasetMixin, original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if self.pg_scores_dict is None:
                raise RuntimeError("Stage2 requires a loaded PG score cache")
            self.stage2_cache = resolver
            self._bind_stage2_requests()

    train_module.LazySupervisedDataset = FailClosedStage2Dataset
    return FailClosedStage2Dataset


if __name__ == "__main__":
    config_path = os.environ.get("REACTVAU_STAGE2_CACHE_CONFIG")
    if not config_path:
        raise RuntimeError("REACTVAU_STAGE2_CACHE_CONFIG is required")
    install(config_path)
    init_distributed_mode()
    train_module.train()
