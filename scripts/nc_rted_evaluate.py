#!/usr/bin/env python3
"""Evaluate a complete frozen NC-RTED prediction matrix with official code."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nc_rted.evaluation import EvaluationError, evaluate_frozen_matrix, freeze_prediction_matrix
from nc_rted.mechanism_posteval import PostEvalError, stratify_completed_matrix


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
    parser.add_argument("--mechanism-strata-output")
    parser.add_argument("--mechanism-decision-threshold", type=float)
    args = parser.parse_args()
    try:
        frozen = freeze_prediction_matrix(plan_path=args.plan, plan_sha256=args.plan_sha256, stores_root=args.stores_root)
        if args.mechanism_strata_output:
            if args.mechanism_decision_threshold is None:
                parser.error("--mechanism-strata-output requires --mechanism-decision-threshold")
            output = Path(args.mechanism_strata_output)
            if output.exists():
                raise EvaluationError("mechanism strata output must not already exist")
            from nc_rted.evaluation import _official_modules
            detect, _ = _official_modules(Path(args.reactvau_root).resolve())
            annotations = ({("ucf", key): value for key, value in detect.load_anno_txt(args.ucf_annotation, "ucf-crime").items()} |
                           {("xd", key): value for key, value in detect.load_anno_txt(
                               args.xd_annotation, "xd-violence",
                               video_dir=None if args.xd_video_dir is None else args.xd_video_dir).items()})
            strata = stratify_completed_matrix(frozen=frozen, annotations=annotations,
                                               make_labels=detect.make_gt_labels_from_anno,
                                               decision_threshold=args.mechanism_decision_threshold)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(strata, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        evaluate_frozen_matrix(frozen=frozen, reactvau_root=args.reactvau_root, ucf_annotation=args.ucf_annotation,
                               xd_annotation=args.xd_annotation, hivau_references=args.hivau_references,
                               output_dir=args.output_dir, bootstrap_seed=args.bootstrap_seed, xd_video_dir=args.xd_video_dir)
    except (EvaluationError, PostEvalError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
