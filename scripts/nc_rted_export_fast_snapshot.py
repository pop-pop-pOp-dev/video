#!/usr/bin/env python3
"""Export explicit training-media Fast score bindings without decoding media."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nc_rted.fast_snapshot import FastSnapshotError, build_snapshot, write_snapshot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast-cache", required=True)
    parser.add_argument("--fast-cache-sha256", required=True)
    parser.add_argument("--media-metadata", required=True)
    parser.add_argument("--media-metadata-sha256", required=True)
    parser.add_argument("--fast-identity-json", required=True, help="JSON mapping with checkpoint and implementation identities")
    parser.add_argument("--output")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if bool(args.output) == bool(args.dry_run):
        parser.error("supply exactly one of --output or --dry-run")
    try:
        identity = json.loads(args.fast_identity_json)
        document = build_snapshot(fast_cache=args.fast_cache, fast_cache_sha256=args.fast_cache_sha256,
                                  media_metadata=args.media_metadata, media_metadata_sha256=args.media_metadata_sha256,
                                  fast_identity=identity)
        summary = {"schema": document["schema"], "media_matched": len(document["media"]),
                   "fast_cache_sha256": args.fast_cache_sha256, "media_metadata_sha256": args.media_metadata_sha256}
        if args.dry_run:
            print(json.dumps({"status": "DRY_RUN_MATCHED", **summary}, sort_keys=True))
            return
        digest = write_snapshot(document, args.output)
        print(json.dumps({"status": "EXPORTED", "output": str(Path(args.output).resolve()), "snapshot_sha256": digest, **summary}, sort_keys=True))
    except (FastSnapshotError, json.JSONDecodeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
