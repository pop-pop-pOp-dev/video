#!/usr/bin/env python3
"""Evaluate a complete frozen NC-RTED prediction matrix with official code."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nc_rted.evaluation import EvaluationError, evaluate_frozen_matrix, freeze_prediction_matrix


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--stores-root", required=True)
    parser.add_argument("--reactvau-root", required=True)
    parser.add_argument("--ucf-annotation", required=True)
    parser.add_argument("--xd-annotation", required=True)
    parser.add_argument("--xd-video-dir")
    parser.add_argument("--hivau-references", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-seed", required=True, type=int)
    args = parser.parse_args()
    try:
        frozen = freeze_prediction_matrix(plan_path=args.plan, plan_sha256=args.plan_sha256, stores_root=args.stores_root)
        evaluate_frozen_matrix(frozen=frozen, reactvau_root=args.reactvau_root, ucf_annotation=args.ucf_annotation,
                               xd_annotation=args.xd_annotation, hivau_references=args.hivau_references,
                               output_dir=args.output_dir, bootstrap_seed=args.bootstrap_seed, xd_video_dir=args.xd_video_dir)
    except EvaluationError as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
