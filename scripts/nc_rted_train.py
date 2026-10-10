#!/usr/bin/env python3
"""Run an explicitly bound NC-RTED diagnostic or formally admitted job."""
import argparse
import hashlib
import json
import sys
from pathlib import Path


def _verify_captured_source_map(map_path: str, admission_path: str, admission_sha256: str,
                                config_path: str, config_sha256: str) -> None:
    """Fail before importing project modules unless the immutable closure matches admission."""
    try:
        admission_raw=Path(admission_path).read_bytes()
        if hashlib.sha256(admission_raw).hexdigest() != admission_sha256: raise ValueError
        config_raw=Path(config_path).read_bytes()
        if hashlib.sha256(config_raw).hexdigest() != config_sha256: raise ValueError
        admission=json.loads(admission_raw)
        config=json.loads(config_raw)
        descriptor=Path(map_path).resolve(); document=json.loads(descriptor.read_text())
        expected=admission["source_files"]; mapping=document["files"]
        if document.get("schema") != "nc_rted_captured_source_map/v1" or not isinstance(expected,dict) or not isinstance(mapping,dict) or set(mapping) != set(expected): raise ValueError
        root=descriptor.parent.resolve()
        for original, relative in mapping.items():
            candidate=(root/relative).resolve()
            if (not isinstance(original,str) or not isinstance(relative,str) or Path(relative).is_absolute() or ".." in Path(relative).parts or
                    root not in candidate.parents or not candidate.is_file() or candidate.is_symlink() or hashlib.sha256(candidate.read_bytes()).hexdigest() != expected[original]): raise ValueError
        runtime=document["runtime"]; inherited=config["inherited"]; stage2=config["stage2_cache"]
        if runtime != {"inherited_external_root":"inherited","inherited_source_manifest":"inherited-source-manifest.json","stage2_module":"stage2-module.py"}: raise ValueError
        manifest=root/runtime["inherited_source_manifest"]; files=json.loads(manifest.read_text())["files"]; external=root/runtime["inherited_external_root"]
        if hashlib.sha256(manifest.read_bytes()).hexdigest() != inherited["source_manifest_sha256"] or not isinstance(files,dict): raise ValueError
        for relative, digest in files.items():
            candidate=(external/relative).resolve()
            if (not isinstance(relative,str) or not isinstance(digest,str) or Path(relative).is_absolute() or ".." in Path(relative).parts or
                    external not in candidate.parents or not candidate.is_file() or candidate.is_symlink() or hashlib.sha256(candidate.read_bytes()).hexdigest() != digest): raise ValueError
        stage=root/runtime["stage2_module"]
        if not stage.is_file() or stage.is_symlink() or hashlib.sha256(stage.read_bytes()).hexdigest() != stage2["module_sha256"]: raise ValueError
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise RuntimeError("captured formal source map is invalid before project imports") from error


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--mode", choices=("diagnostic", "formal"), required=True)
    parser.add_argument("--admission", help="formal admission artifact, kept outside the hashed runtime config")
    parser.add_argument("--admission-sha256", help="SHA-256 for --admission")
    parser.add_argument("--captured-source-map", help="verified immutable source map for a formal queue launch")
    parser.add_argument("--dry-run", action="store_true", help="validate all bound assets without importing models")
    args = parser.parse_args()
    if args.mode == "formal" and (not args.admission or not args.admission_sha256):
        raise RuntimeError("formal mode requires --admission and --admission-sha256")
    if args.mode == "diagnostic" and (args.admission or args.admission_sha256):
        raise RuntimeError("diagnostic mode cannot receive formal admission")
    if args.mode == "diagnostic" and args.captured_source_map:
        raise RuntimeError("diagnostic mode cannot receive a captured source map")
    if args.captured_source_map:
        _verify_captured_source_map(args.captured_source_map, args.admission, args.admission_sha256, args.config, args.config_sha256)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from nc_rted.captured_sources import CapturedSourceError, load_captured_runtime, load_captured_sources
    from nc_rted.production_runtime import ProductionRuntimeError, assemble, load_formal_admission, load_manifest
    # Load once normally only to obtain the immutable document used to verify
    # snapshot runtime substitutions; the captured config bytes stay hash-bound.
    preliminary=load_manifest(args.config, expected_sha256=args.config_sha256) if not args.captured_source_map else None
    if args.captured_source_map:
        raw=json.loads(Path(args.config).read_text())
        captured_runtime=load_captured_runtime(args.captured_source_map, raw)
        manifest=load_manifest(args.config, expected_sha256=args.config_sha256, captured_runtime=captured_runtime)
    else:
        manifest=preliminary
    if manifest.run["mode"] != args.mode:
        raise ProductionRuntimeError("CLI mode differs from hash-bound config mode")
    admission = load_formal_admission(args.admission, expected_sha256=args.admission_sha256) if args.mode == "formal" else None
    try:
        captured_sources=load_captured_sources(args.captured_source_map, admission) if args.captured_source_map else None
    except CapturedSourceError as error:
        raise ProductionRuntimeError(str(error)) from error
    if args.dry_run:
        if args.mode == "formal":
            # This is the same pre-model source gate used by assembly; dry-run
            # must not turn immutable-capture validation into import-only smoke.
            from nc_rted.production_runtime import _checkpoint_identity, _validate_formal_admission_before_models
            from nc_rted.task_inputs import TrainingCatalog
            catalog_document=manifest.document["catalog"]
            catalog=TrainingCatalog.load(catalog_document["manifest_directory"], catalog_document["training_annotations"], expected_provenance_sha256=catalog_document["provenance_sha256"])
            _validate_formal_admission_before_models(admission, _checkpoint_identity(manifest,catalog), captured_sources=captured_sources)
        print(json.dumps({"status": "FILE_BINDINGS_PASS_SEMANTIC_NOT_RUN", "schema": manifest.document["schema"], "run_id": manifest.run["run_id"]}))
        return 0
    print(json.dumps(assemble(manifest, admission=admission, captured_sources=captured_sources).run(), sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError) as error:
        raise SystemExit(f"NC-RTED runtime rejected configuration: {error}")
