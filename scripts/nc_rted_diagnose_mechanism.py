#!/usr/bin/env python3
"""Prepare the legal fixed development-prefix input for checkpoint diagnostics."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nc_rted.mechanism_diagnostics import (write_development_manifest, write_development_media_catalog,
                                           write_development_runtime_inputs, write_development_fast_plan)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--media-catalog")
    parser.add_argument("--produce-media-catalog", action="store_true")
    parser.add_argument("--ucf-media-root")
    parser.add_argument("--xd-media-root")
    parser.add_argument("--produce-fast-plan", action="store_true")
    parser.add_argument("--runtime-inputs")
    parser.add_argument("--fast-identity-json")
    parser.add_argument("--protocols-json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--development-manifest")
    parser.add_argument("--source-splits", required=True)
    parser.add_argument("--ucf-train-database", required=True)
    parser.add_argument("--xd-train-database", required=True)
    parser.add_argument("--requested", type=int, default=256)
    parser.add_argument("--per-family-cap", type=int, default=4)
    args = parser.parse_args()
    if args.produce_fast_plan and (not args.runtime_inputs or not args.fast_identity_json or not args.protocols_json):
        parser.error("--produce-fast-plan requires --runtime-inputs, --fast-identity-json, and --protocols-json")
    if args.produce_media_catalog and (not args.development_manifest or not args.ucf_media_root or not args.xd_media_root):
        parser.error("--produce-media-catalog requires --development-manifest and both media roots")
    if not args.produce_media_catalog and bool(args.development_manifest) != bool(args.media_catalog):
        parser.error("--development-manifest and --media-catalog must be supplied together")
    document = (write_development_fast_plan(args.output, runtime_inputs=args.runtime_inputs,
        fast_identity=json.loads(Path(args.fast_identity_json).read_text()), protocols=json.loads(Path(args.protocols_json).read_text())) if args.produce_fast_plan else
        write_development_media_catalog(args.output, development_manifest=args.development_manifest,
        ucf_media_root=args.ucf_media_root, xd_media_root=args.xd_media_root) if args.produce_media_catalog else
        write_development_runtime_inputs(args.output, development_manifest=args.development_manifest,
        splits_path=args.source_splits, ucf_database_path=args.ucf_train_database,
        xd_database_path=args.xd_train_database, media_catalog=args.media_catalog)
        if args.development_manifest else write_development_manifest(args.output, splits_path=args.source_splits,
        ucf_database_path=args.ucf_train_database, xd_database_path=args.xd_train_database,
        requested=args.requested, per_family_cap=args.per_family_cap))
    status = document.get("status", document.get("schema", "CREATED"))
    print(f"{status} records={len(document.get('records', document.get('media', [])))} output={args.output}")


if __name__ == "__main__":
    main()
