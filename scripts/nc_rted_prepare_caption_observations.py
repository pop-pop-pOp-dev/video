#!/usr/bin/env python3
"""Populate the bounded persistent caption-observation cache for one manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from nc_rted.production_runtime import assemble_caption_preparation, load_caption_preparation_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--sample-id", action="append")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest = load_caption_preparation_manifest(args.manifest, expected_sha256=args.manifest_sha256)
    if manifest.run["mode"] != "diagnostic":
        raise SystemExit("caption preparation requires a separately admitted diagnostic manifest")
    result = assemble_caption_preparation(manifest).prepare_caption_observations(args.sample_id)
    output = Path(args.output)
    if not output.is_absolute(): raise SystemExit("--output must be absolute")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"schema": "nc_rted_caption_observation_preparation/v1", "manifest_sha256": manifest.config_sha256,
                                  "result": result}, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
