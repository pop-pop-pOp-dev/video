#!/usr/bin/env python3
"""CPU validation of bounded caption leases through the inherited Stage2 path."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sys
from types import MethodType

import torch

from nc_rted.bounded_stage2_cache import BoundedStage2Cache
from nc_rted.caption_sampling import OriginalSamplingAuditReader
from nc_rted.production_runtime import _stage2_dataset_class


class _CpuImageProcessor:
    crop_size = {"height": 4, "width": 4}

    def preprocess(self, frames, *, return_tensors: str):
        if return_tensors != "pt" or len(frames) == 0:
            raise RuntimeError("unexpected inherited image processor call")
        return {"pixel_values": torch.zeros((len(frames), 3, 4, 4), dtype=torch.float32)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_report(path: Path, report: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-yaml", required=True, type=Path)
    parser.add_argument("--resolver", required=True, type=Path)
    parser.add_argument("--resolver-sha256", required=True)
    parser.add_argument("--resolver-config", required=True, type=Path)
    parser.add_argument("--resolver-config-sha256", required=True)
    parser.add_argument("--scratch-root", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    if args.report.exists():
        raise RuntimeError("refusing to replace an existing validation report")
    if _sha256(args.resolver) != args.resolver_sha256 or _sha256(args.resolver_config) != args.resolver_config_sha256:
        raise RuntimeError("requested resolver binding does not match its SHA-256")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.scratch_root.mkdir(parents=True, exist_ok=True)
    report = {"schema": "nc_rted_bounded_stage2_cpu_validation/v1", "status": "RUNNING", "cases": []}
    _write_report(args.report, report)
    stage2 = {
        "mode": "bounded", "module": str(args.resolver.resolve()), "module_sha256": args.resolver_sha256,
        "expected_resolver_sha256": args.resolver_sha256, "config": str(args.resolver_config.resolve()),
        "config_sha256": args.resolver_config_sha256, "accepted_status": "APPROVED_FOR_EXECUTION",
        "scratch_root": str(args.scratch_root.resolve()), "minimum_free_bytes": 20 * 1024 ** 3,
        "overhead_bytes": 16 * 1024 ** 2, "max_temporary_bytes": None,
    }
    cache = BoundedStage2Cache(stage2)
    root = Path(os.environ["NC_RTED_REACTVAU_ROOT"])
    sys.path.insert(0, str(root))
    from llava import conversation as conversation_lib
    from llava.train import train as train_module
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, model_max_length=8192)
    tokenizer.padding_side = "right"
    conversation_lib.default_conversation = conversation_lib.conv_templates["qwen_2"]
    data_args = train_module.DataArguments(data_path=str(args.dataset_yaml), lazy_preprocess=True,
        frames_upbound=64, frames_lowbound=4, local_num_frames=1, sample_type="dynamic_fps1", time_msg="short_online_v2")
    data_args.image_processor = _CpuImageProcessor()
    data_args.is_multimodal = True
    data_args.mm_use_im_start_end = False
    previous = os.environ.get("REACTVAU_STAGE2_CACHE_CONFIG")
    os.environ["REACTVAU_STAGE2_CACHE_CONFIG"] = str(args.resolver_config)
    try:
        dataset = _stage2_dataset_class(train_module.LazySupervisedDataset, cache)(str(args.dataset_yaml), tokenizer, data_args)
    finally:
        if previous is None:
            os.environ.pop("REACTVAU_STAGE2_CACHE_CONFIG", None)
        else:
            os.environ["REACTVAU_STAGE2_CACHE_CONFIG"] = previous
    indices = {row["id"]: index for index, row in enumerate(dataset.list_data_dict)}
    targets = (316, 27023, 11338)
    reader = OriginalSamplingAuditReader(dataset)
    original_acquire = cache.acquire
    leased_paths: list[dict] = []

    @contextmanager
    def observed_acquire(self, relative, request_index):
        with original_acquire(relative, request_index) as path:
            leased_paths.append({"relative": relative, "request_index": request_index, "path": str(path), "exists_during_lease": path.exists()})
            yield path

    cache.acquire = MethodType(observed_acquire, cache)
    try:
        for target in targets:
            item = {"id": target, "status": "RUNNING"}
            report["cases"].append(item)
            _write_report(args.report, report)
            try:
                read = reader.read(indices[target])
                audit = read.audit
                derived = "/videos/" not in audit.relative_video
                prepared = [json.loads(line) for line in cache.events_path.read_text().splitlines() if line]
                leases = [entry for entry in leased_paths if entry["relative"] == audit.relative_video]
                if len(leases) != 1 or not leases[0]["exists_during_lease"]:
                    raise RuntimeError("original Stage2 process_video did not use exactly one observed bounded lease")
                released = [event for event in prepared if event["event"] == "released" and event["relative"] == audit.relative_video]
                prepared_event = next((event for event in reversed(prepared) if event["event"] == "prepared" and event["relative"] == audit.relative_video), None)
                if derived and (len(released) != 1 or prepared_event is None or Path(prepared_event["output"]).exists()):
                    raise RuntimeError("successful derived bounded lease was not released")
                if not derived and Path(leases[0]["path"]) != Path(cache.requests.resolve(
                        audit.relative_video, leases[0]["request_index"])["source"]):
                    raise RuntimeError("direct source media was copied instead of leased")
                if any(right <= left for left, right in zip(audit.frame_indices, audit.frame_indices[1:])):
                    raise RuntimeError("original sampler returned duplicate or decreasing frame indices")
                item.update({"status": "PASS", "dataset_index": indices[target], "relative_video": audit.relative_video,
                             "derived": derived, "frame_indices": list(audit.frame_indices),
                             "sampled_frame_times": list(audit.sampled_frame_times), "fps": audit.fps,
                             "time_message": audit.time_message, "pg_scores": list(audit.aligned_pg_scores),
                             "lease": leases[0], "released_after_decode": bool(released) if derived else None})
            except BaseException as error:
                item.update({"status": "FAILED", "error": f"{type(error).__name__}: {error}"})
                report["status"] = "FAILED"
                _write_report(args.report, report)
                raise
            _write_report(args.report, report)
    except BaseException:
        raise
    report["status"] = "PASS"
    _write_report(args.report, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
