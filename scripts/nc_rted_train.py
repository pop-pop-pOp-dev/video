#!/usr/bin/env python3
"""Run an explicitly bound NC-RTED diagnostic or formally admitted job."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nc_rted.production_runtime import ProductionRuntimeError, assemble, load_formal_admission, load_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--mode", choices=("diagnostic", "formal"), required=True)
    parser.add_argument("--admission", help="formal admission artifact, kept outside the hashed runtime config")
    parser.add_argument("--admission-sha256", help="SHA-256 for --admission")
    parser.add_argument("--dry-run", action="store_true", help="validate all bound assets without importing models")
    args = parser.parse_args()
    manifest = load_manifest(args.config, expected_sha256=args.config_sha256)
    if manifest.run["mode"] != args.mode:
        raise ProductionRuntimeError("CLI mode differs from hash-bound config mode")
    if args.mode == "formal" and (not args.admission or not args.admission_sha256):
        raise ProductionRuntimeError("formal mode requires --admission and --admission-sha256")
    if args.mode == "diagnostic" and (args.admission or args.admission_sha256):
        raise ProductionRuntimeError("diagnostic mode cannot receive formal admission")
    admission = load_formal_admission(args.admission, expected_sha256=args.admission_sha256) if args.mode == "formal" else None
    if args.dry_run:
        print(json.dumps({"status": "FILE_BINDINGS_PASS_SEMANTIC_NOT_RUN", "schema": manifest.document["schema"], "run_id": manifest.run["run_id"]}))
        return 0
    print(json.dumps(assemble(manifest, admission=admission).run(), sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ProductionRuntimeError as error:
        raise SystemExit(f"NC-RTED runtime rejected configuration: {error}")
