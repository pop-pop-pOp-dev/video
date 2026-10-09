#!/usr/bin/env python3
"""Build an immutable NC-RTED teacher manifest from frozen train records."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nc_rted.teacher_pipeline import PipelineError, build_teachers, parse_frozen_records, publish_teacher_manifest
from nc_rted.teacher_store import TeacherStoreError, build_teachers_from_store


def main() -> None:
    parser = argparse.ArgumentParser()
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--store", type=Path, help="committed chunked teacher-observation store")
    inputs.add_argument("--input", type=Path, help="small JSON fixture only; not a production input")
    parser.add_argument("--frozen-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.store is not None:
            manifest = build_teachers_from_store(args.store)
            input_provenance = args.store / "index.json"
        else:
            records = parse_frozen_records(json.loads(args.input.read_text()))
            manifest = build_teachers(records)
            input_provenance = args.input
        publish_teacher_manifest(args.output, manifest, input_provenance, args.frozen_config)
    except (OSError, json.JSONDecodeError, PipelineError, TeacherStoreError) as error:
        raise SystemExit(f"teacher build refused: {error}")


if __name__ == "__main__":
    main()
