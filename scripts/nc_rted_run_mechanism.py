#!/usr/bin/env python3
"""Run actual checkpoint-time NC-RTED mechanism diagnostics on development prefixes."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nc_rted.mechanism_diagnostics import DiagnosticError
from nc_rted.mechanism_runtime import MechanismRuntimeError, run_checkpoint_diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-manifest", required=True)
    parser.add_argument("--runtime-manifest-sha256", required=True)
    parser.add_argument("--incremental-checkpoint")
    parser.add_argument("--runtime-inputs", required=True)
    parser.add_argument("--runtime-inputs-sha256", required=True)
    parser.add_argument("--fast-snapshot", required=True)
    parser.add_argument("--fast-snapshot-sha256", required=True)
    parser.add_argument("--prediction-runtime-manifest", required=True)
    parser.add_argument("--prediction-runtime-manifest-sha256", required=True)
    parser.add_argument("--diagnostic-training-teacher-store", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    try:
        report = run_checkpoint_diagnostics(
            args.output, runtime_manifest=args.runtime_manifest,
            runtime_manifest_sha256=args.runtime_manifest_sha256,
            runtime_inputs=args.runtime_inputs, runtime_inputs_sha256=args.runtime_inputs_sha256,
            fast_snapshot=args.fast_snapshot, fast_snapshot_sha256=args.fast_snapshot_sha256,
            prediction_runtime_manifest=args.prediction_runtime_manifest,
            prediction_runtime_manifest_sha256=args.prediction_runtime_manifest_sha256,
            diagnostic_training_teacher_store=args.diagnostic_training_teacher_store,
            checkpoint=args.incremental_checkpoint,
            generation_config={"do_sample": False, "num_beams": 1, "max_new_tokens": args.max_new_tokens},
        )
    except (MechanismRuntimeError, DiagnosticError) as error:
        parser.error(str(error))
    print(f"records={len(report['records'])} output={Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
