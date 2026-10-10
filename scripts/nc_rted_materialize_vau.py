#!/usr/bin/env python3
"""Build or execute the label-free official VAU media materialization plan."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from nc_rted.vau_media import (MIN_FREE_BYTES, VAUMediaError, _atomic_json_new, build_plans,
                               estimate, materialize, plan_document, raw_frame_capacity, sha256_file,
                               validate_plan, _read_identity)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--ucf-ranges", type=Path, required=True)
    parser.add_argument("--xd-ranges", type=Path, required=True)
    parser.add_argument("--ucf-root", type=Path, required=True)
    parser.add_argument("--xd-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument("--plan-output", type=Path)
    parser.add_argument("--plan-path", type=Path)
    parser.add_argument("--plan-sha256")
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--hard-guard-gib", type=float, default=20.0)
    parser.add_argument("--max-artifact-gib", type=float, default=50.0)
    args = parser.parse_args()
    if args.project_root is not None and args.project_root.resolve() != ROOT.resolve():
        raise VAUMediaError("--project-root must be this approved project root")
    identities = _read_identity(args.identity)
    identity_sha256 = sha256_file(args.identity)
    plans = build_plans(identity_path=args.identity, ucf_ranges=args.ucf_ranges, xd_ranges=args.xd_ranges,
                        ucf_root=args.ucf_root, xd_root=args.xd_root, output_root=args.output_root)
    plan = estimate(plans)
    document = plan_document(plans, identity_path=args.identity, ucf_ranges=args.ucf_ranges, xd_ranges=args.xd_ranges)
    plan = document["summary"]
    if args.plan_output:
        output_sha = _atomic_json_new(args.plan_output, document)
        plan = {**plan, "plan_output": str(args.plan_output.resolve()), "plan_output_sha256": output_sha}
    if args.estimate_only:
        print(json.dumps(plan, sort_keys=True))
        return
    guard = int(args.hard_guard_gib * 1024**3)
    if guard < MIN_FREE_BYTES:
        raise VAUMediaError("--hard-guard-gib must be at least 20")
    if args.max_artifact_gib <= 0:
        raise VAUMediaError("--max-artifact-gib must be positive")
    if args.plan_path is None or args.plan_sha256 is None or args.project_root is None:
        raise VAUMediaError("materialization requires --plan-path, --plan-sha256, and --project-root")
    if document["summary"]["identity_sha256"] != identity_sha256:
        raise VAUMediaError("identity changed while rebuilding immutable materialization plan")
    cap = int(args.max_artifact_gib * 1024**3)
    validate_plan(args.plan_path, args.plan_sha256, document, project_root=args.project_root,
                  output_root=args.output_root, max_artifact_bytes=cap)
    catalog, journal = materialize(plans, output_root=args.output_root, hard_guard_bytes=guard,
                                   max_artifact_bytes=cap, parent_bindings=document["parents"])
    # Bind every question id without reading prompt/task/type.  It is intentionally
    # separate from the catalog for the later blind question roster join.
    by_video = {row["media_key"]: row for row in json.loads(catalog.read_text())["media"]}
    bindings = [{"id": row["id"], "video": row["video"], "media_path": by_video[row["video"]]["media_path"],
                 "media_sha256": by_video[row["video"]]["media_sha256"]}
                for row in identities]
    binding_path = args.output_root / "vau_question_media_bindings.json"
    binding_document = {"schema": "nc_rted_vau_question_media_bindings/v1", "bindings": bindings}
    if binding_path.exists():
        if json.loads(binding_path.read_text()) != binding_document:
            raise VAUMediaError("existing immutable question bindings differ from recovered catalog")
        binding_sha = sha256_file(binding_path)
    else:
        binding_sha = _atomic_json_new(binding_path, binding_document)
    print(json.dumps({"catalog": str(catalog), "catalog_sha256": sha256_file(catalog), "journal": str(journal),
                      "bindings": str(binding_path), "bindings_sha256": binding_sha, "plan": plan}, sort_keys=True))


if __name__ == "__main__":
    main()
