#!/usr/bin/env python3
"""Measure one preselected development prefix through fresh Fast + production VAD."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import importlib.util
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nc_rted.development_fast_runtime import bound_json, digest_file, validate_plan, verify_scorer_inputs, atomic_json
from nc_rted.production_cold import run_cold_prefix, validate_runtime_fast, ColdError
from nc_rted.storage_lock import open_lock_file


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ColdError("bound Fast implementation cannot be imported")
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    names = ("plan", "training-bindings-report", "frozen-fast-config", "prediction-runtime", "model-manifest")
    for name in names:
        parser.add_argument("--" + name, type=Path, required=True)
        parser.add_argument("--" + name + "-sha256", required=True)
    parser.add_argument("--group", choices=("R0", "A", "U", "S", "F"), required=True)
    parser.add_argument("--seed", type=int, choices=(17, 42, 2026))
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--deadline", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.parent.is_dir():
        parser.error("output must be absent inside an existing directory")
    if (args.group == "R0") != (args.seed is None):
        parser.error("R0 has no seed; trained groups require a formal seed")
    deadline = datetime.fromisoformat(args.deadline.replace("Z", "+00:00"))
    if deadline.tzinfo is None or deadline <= datetime.now(timezone.utc):
        parser.error("deadline must be a future timezone-aware date")
    lock = args.output.parent / (args.output.name + ".lock")
    with open_lock_file(lock, 20 << 30) as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.output.exists() or args.output.with_name(args.output.name + ".cold-cache").exists():
            parser.error("output or cold cache already exists; use a new versioned result")
        execute(args, names, deadline, parser)


def execute(args, names, deadline, parser):
    bindings = {name.replace("-", "_"): {"path": str(getattr(args, name.replace("-", "_")).resolve()),
                 "sha256": getattr(args, name.replace("-", "_") + "_sha256")} for name in names}
    documents = {name: bound_json(binding) for name, binding in bindings.items()}
    plan, config = documents["plan"], documents["frozen_fast_config"]
    media = validate_plan(plan, documents["training_bindings_report"])
    inputs = bound_json(plan["runtime_inputs"])
    selected = [row for row in inputs["records"] if row["sample_id"] == args.sample_id]
    if len(selected) != 1:
        parser.error("sample must be an exact sealed development prefix")
    record = selected[0]
    paths = verify_scorer_inputs(config, documents["training_bindings_report"])
    from nc_rted.prediction_runtime import load_prediction_runtime, _import_bound_runtime, _DefaultLoader
    from nc_rted.prediction_inputs import ModelTask, load_model_artifact
    from nc_rted.numerics import configure_deterministic_algorithms
    from nc_rted.detection_media import OpenCVFrames
    runtime = load_prediction_runtime(args.prediction_runtime, expected_sha256=args.prediction_runtime_sha256)
    validate_runtime_fast(runtime, paths, config, plan)
    task = ModelTask(args.group, args.seed)
    artifact = load_model_artifact(args.model_manifest, expected_sha256=args.model_manifest_sha256, task=task)
    if datetime.now(timezone.utc) >= deadline:
        raise ColdError("deadline reached during preflight; no models loaded")
    configure_deterministic_algorithms()
    _import_bound_runtime(runtime)
    model = _DefaultLoader(runtime, args.device).load(artifact)
    released = module("nc_rted_cold_released_fast", paths["released_precompute"])
    prompts = module("nc_rted_cold_released_prompt", paths["prompt_source"])
    # Slow language is already on CPU after the production loader. Fast is
    # constructed once before timing, then follows the accepted residency path.
    scorer = released.PaliGemmaScorer(paths["model_path"], paths["lora_path"], device=args.device,
                                     image_size=384, streamforest_weights_path=paths["streamforest_weights"],
                                     vision_feature_layer=-2, attn_implementation="sdpa")
    model.residency.detector = scorer
    model.residency._stage_fast_cpu()
    model.residency.stage_language()
    prompt = prompts.get_grid_prompt(add_special_tokens=False)
    result = run_cold_prefix(model=model, runtime=runtime,
                             media_row=media[(record["dataset"], record["key"])], record=record,
                             score_grid=lambda grid: model.residency.score_fast(runtime.document["fast"], 1, [grid], prompt),
                             make_grid=lambda frames: released.create_grid_image(frames, image_size=384),
                             cache_root=args.output.parent / (args.output.name + ".cold-cache"),
                             deadline=deadline, device=args.device, decoder_factory=OpenCVFrames)
    result["bindings"] = bindings
    source_root = Path(__file__).resolve().parents[1] / "src/nc_rted"
    result["implementation"] = {"cli": digest_file(__file__), **{name: digest_file(source_root / (name + ".py"))
        for name in ("production_cold", "development_fast_runtime", "prediction_runtime", "prediction_adapters",
                     "prediction_worker", "detection_provider", "detection_media", "media_observer",
                     "observation_cache", "detector", "features", "bridge", "model", "numerics")}}
    atomic_json(args.output, result)
    print(json.dumps({"output": str(args.output), "sha256": digest_file(args.output), "slow_calls": result["slow_calls"]}))


if __name__ == "__main__":
    main()
