#!/usr/bin/env python3
"""Run the sealed development Fast worklist, committing each completed medium."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import itertools
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nc_rted.development_fast_runtime import (FastPreparationError, bound_json, digest_file,
    execute_plan, validate_plan, verify_scorer_inputs)


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None: raise FastPreparationError("cannot import bound scorer component")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("plan", "frozen-fast-config", "training-bindings-report"):
        parser.add_argument("--" + name, type=Path, required=True)
        parser.add_argument("--" + name + "-sha256", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--deadline", required=True, help="ISO8601 UTC deadline; committed media are retained")
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): parser.error("refusing to overwrite final output")
    try:
        deadline = datetime.fromisoformat(args.deadline.replace("Z", "+00:00"))
        if deadline.tzinfo is None: raise FastPreparationError("deadline must include timezone")
        bindings = {name: {"path": str(getattr(args, name).resolve()), "sha256": getattr(args, name + "_sha256")}
                    for name in ("plan", "frozen_fast_config", "training_bindings_report")}
        documents = {name: bound_json(binding) for name, binding in bindings.items()}
        plan, config, inherited = documents["plan"], documents["frozen_fast_config"], documents["training_bindings_report"]
        media = validate_plan(plan, inherited)
        # Content verification precedes imports/model construction; no arbitrary CLI model hashes are trusted.
        paths = verify_scorer_inputs(config, inherited)
        binding = {"schema": "nc_rted_development_fast_execution/v1", "inputs": bindings,
                   "device": args.device, "batch_size": 1,
                   "implementation": {"cli": digest_file(__file__), "runtime": digest_file(Path(__file__).resolve().parents[1] / "src/nc_rted/development_fast_runtime.py")}}
        def factory():
            released = _module("nc_rted_released_fast", paths["released_precompute"])
            grid_module = _module("reactvau_fast_grid_stream", paths["grid_stream"])
            prompt_module = _module("nc_rted_released_prompt", paths["prompt_source"])
            scorer = released.PaliGemmaScorer(paths["model_path"], paths["lora_path"], device=args.device,
                image_size=384, streamforest_weights_path=paths["streamforest_weights"], vision_feature_layer=-2,
                attn_implementation="sdpa")
            prompt = prompt_module.get_grid_prompt(add_special_tokens=False)
            def score(source, maximum):
                values = []
                with grid_module.open_video_grids(source["media_path"], 384, 4, 4, released.create_grid_image) as (_, grids):
                    for grid_image in itertools.islice(grids, maximum + 1):
                        if datetime.now(timezone.utc) >= deadline:
                            raise FastPreparationError("development Fast deadline reached; media commits preserved")
                        values.extend(scorer.batch_score_grids([grid_image], prompt, batch_size=1))
                return values
            return score
        result = execute_plan(plan=plan, media=media, binding=binding, journal=args.journal,
                              output=args.output, scorer_factory=factory, deadline=deadline, minimum_free_bytes=20 * 1024**3 + 1_600_000_000)
        print(json.dumps({"output": str(args.output), "sha256": digest_file(args.output), "media": len(result["media"])}))
    except (FastPreparationError, KeyError, ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__": main()
