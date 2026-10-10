#!/usr/bin/env python3
"""Derive conservative source34 seed applicability from source33 evidence.

This CPU-only producer does not qualify, train, or enqueue work.  It creates
target runtime bindings for seeds 42 and 2026 and an immutable applicability
record consumed only by source34's resource attestation verifier.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path


GROUPS = ("A", "U", "S", "F")
ALLOWED = "src/nc_rted/resource_attestation.py"


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical(value): return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def bound(path, expected):
    path = Path(path).resolve()
    if not path.is_file() or sha(path) != expected: raise ValueError("hash-bound input differs")
    value = json.loads(path.read_text())
    if not isinstance(value, dict): raise ValueError("bound input is not an object")
    return path, value


def source_manifest(root):
    root = Path(root).resolve(); package = root / "src/nc_rted"
    files = {f"scripts/{name}": sha(root / "scripts" / name) for name in ("nc_rted_interleaved_gpu_diagnostic.py", "nc_rted_interleaved_formal.py", "nc_rted_qualify_interleaved_formal_bundle.py")}
    files.update({str(path.relative_to(root)): sha(path) for path in sorted(package.glob("*.py"))})
    return {"schema": "nc_rted_interleaved_source_manifest/v1", "code_sha256": hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest(), "files": files}


def normalized(document):
    value = json.loads(json.dumps(document))
    for key in ("run_id", "seed", "checkpoint_root", "progress_path"): value["run"].pop(key, None)
    for key in ("code_sha256", "runtime_sha256"): value["hashes"].pop(key, None)
    return value


def import_source34(root):
    root = Path(root).resolve()
    source = str(root / "src")
    if not (root / "src/nc_rted/production_runtime.py").is_file() or not (root / "src/nc_rted/resource_attestation.py").is_file():
        raise ValueError("source34 root is incomplete")
    if source not in sys.path: sys.path.insert(0, source)
    runtime = importlib.import_module("nc_rted.production_runtime")
    attestation = importlib.import_module("nc_rted.resource_attestation")
    if Path(runtime.__file__).resolve() != root / "src/nc_rted/production_runtime.py" or Path(attestation.__file__).resolve() != root / "src/nc_rted/resource_attestation.py":
        raise ValueError("source34 runtime imports differ from source root")
    return runtime, attestation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source33-root", required=True); parser.add_argument("--source34-root", required=True)
    parser.add_argument("--profile", required=True); parser.add_argument("--profile-sha256", required=True)
    parser.add_argument("--qualification", required=True); parser.add_argument("--qualification-sha256", required=True)
    parser.add_argument("--safety-multiplier", type=float, required=True); parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    profile_path, profile = bound(args.profile, args.profile_sha256); qualification_path, qualification = bound(args.qualification, args.qualification_sha256)
    runtime, attestation = import_source34(args.source34_root)
    if args.safety_multiplier < 1 or profile.get("status") != "NON_ADMITTED_FORMAL_PROFILE": raise ValueError("profile or safety multiplier differs")
    bundle_path, bundle = bound(profile["diagnostic_bundle"], profile["diagnostic_bundle_sha256"]); del bundle_path
    _, measured_source = bound(bundle["source_manifest"], bundle["source_manifest_sha256"])
    reconstructed_source33 = source_manifest(args.source33_root)
    if reconstructed_source33 != measured_source:
        raise ValueError("source33 root does not reproduce its measured source manifest")
    target_source = source_manifest(args.source34_root)
    changed = {name for name in set(measured_source["files"]) | set(target_source["files"]) if measured_source["files"].get(name) != target_source["files"].get(name)}
    if changed != {ALLOWED}: raise ValueError("source34 differs outside its resource consumer")
    if qualification.get("source_sha256") != measured_source["code_sha256"]: raise ValueError("qualification source differs")
    measured = profile["member_identities"]; workload = qualification.get("workload", {})
    if workload.get("member_identities") != measured or any(value.get("seed") != "17" for value in measured.values()): raise ValueError("qualification is not seed17 profile evidence")
    metrics = qualification.get("measurements", {})
    update, setup = metrics.get("seconds_per_bundle_update_upper_bound"), metrics.get("setup_checkpoint_seconds_upper_bound")
    if not isinstance(update, (int, float)) or not isinstance(setup, (int, float)) or update <= 0 or setup <= 0: raise ValueError("qualification timing is absent")
    output = Path(args.output_root).resolve()
    if output.exists(): raise ValueError("output root already exists")
    output.mkdir(parents=True)
    (output / "source34-manifest.json").write_bytes(canonical(target_source))
    unchanged = {name: digest for name, digest in measured_source["files"].items() if name != ALLOWED}
    for seed in (42, 2026):
        target, identities = {}, {}
        for group in GROUPS:
            _, original = bound(profile["members"][group]["runtime"], profile["members"][group]["runtime_sha256"])
            value = json.loads(json.dumps(original)); value["run"].update({"seed": seed, "run_id": f"formal:{seed}:{group}", "checkpoint_root": str(output / "runs" / str(seed) / group / "checkpoints"), "progress_path": str(output / "runs" / str(seed) / group / "progress.json")})
            value["hashes"]["code_sha256"] = target_source["code_sha256"]; value["hashes"]["runtime_sha256"] = target_source["code_sha256"]
            if normalized(original) != normalized(value): raise ValueError("target runtime changes recipe/data/provider/model/recovery")
            path = output / "members" / str(seed) / f"{group}.runtime.json"; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(canonical(value))
            manifest = runtime.load_manifest(path, expected_sha256=sha(path))
            identities[group] = attestation.formal_runtime_identity(manifest)
            if identities[group].get("seed") != str(seed) or identities[group].get("code_sha256") != target_source["code_sha256"]:
                raise ValueError("source34 target runtime identity differs")
        projected = (1000 * update + setup) * args.safety_multiplier
        document = {"schema": "nc_rted_source34_seed_applicability/v1", "status": "PASS_CONSERVATIVE_SEED_APPLICABILITY", "measured_qualification": {"path": str(qualification_path), "sha256": args.qualification_sha256}, "measured_identities": measured, "target_seed": seed, "target_identities": identities, "measured_runtimes": {group: {"path": profile["members"][group]["runtime"], "sha256": profile["members"][group]["runtime_sha256"]} for group in GROUPS}, "target_runtimes": {group: {"path": str(output / "members" / str(seed) / f"{group}.runtime.json"), "sha256": sha(output / "members" / str(seed) / f"{group}.runtime.json")} for group in GROUPS}, "source_transition": {"allowed_changed_files": [ALLOWED], "unchanged_files": unchanged, "measured_manifest": {"path": bundle["source_manifest"], "sha256": bundle["source_manifest_sha256"]}, "target_manifest": {"path": str(output / "source34-manifest.json"), "sha256": sha(output / "source34-manifest.json")}}, "invariants": {"samples": 8000, "updates": 1000, "accumulation": 8, "shared_preparation": "frozen_provider_only", "common_recovery": True, "sampler_rng_difference_explicit": True}, "projection": {"measured_seconds_per_bundle_update_upper_bound": update, "measured_setup_checkpoint_seconds_upper_bound": setup, "safety_multiplier": args.safety_multiplier, "projected_total_seconds_upper_bound": projected, "is_measured_target_timing": False}}
        (output / f"seed{seed}-applicability.json").write_bytes(canonical(document))
    print(json.dumps({"status": "DERIVED_NOT_MEASURED", "output_root": str(output), "source34_code_sha256": target_source["code_sha256"]}, sort_keys=True))


if __name__ == "__main__": main()
