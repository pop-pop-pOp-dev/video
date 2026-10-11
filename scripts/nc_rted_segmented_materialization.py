"""Immutable source39 successor runtime materialization for segmented formal work."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

GROUPS = ("A", "U", "S", "F")
ALLOWED_CHANGED = {
    "scripts/nc_rted_interleaved_formal.py",
    "scripts/nc_rted_queue.py",
    "src/nc_rted/queue.py",
    "src/nc_rted/resource_attestation.py",
}


class MaterializationError(ValueError):
    pass


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _bound(binding, label):
    if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
        raise MaterializationError(f"{label} binding differs")
    path = Path(binding["path"]).resolve()
    if not path.is_file() or digest(path) != binding["sha256"]:
        raise MaterializationError(f"{label} differs from its SHA-256")
    return path, json.loads(path.read_text())


def _manifest(root, baseline):
    root = Path(root).resolve()
    files = {}
    for relative in sorted(set(baseline) | {"scripts/nc_rted_queue.py"}):
        path = root / relative
        if not path.is_file():
            raise MaterializationError(f"successor source is missing {relative}")
        files[relative] = digest(path)
    if not files:
        raise MaterializationError("successor source closure is incomplete")
    return {"schema":"nc_rted_interleaved_source_manifest/v1", "files":files,
            "code_sha256":hashlib.sha256(json.dumps(files,sort_keys=True,separators=(",", ":")).encode()).hexdigest()}


def _write_tree(root, documents):
    if root.exists() or not root.parent.is_dir():
        raise MaterializationError("output root already exists or has no parent directory")
    stage = Path(tempfile.mkdtemp(prefix=f".{root.name}.", dir=root.parent))
    try:
        for relative, value in documents.items():
            target = stage / relative; target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(canonical(value))
        for directory, _, _ in os.walk(stage, topdown=False):
            fd=os.open(directory, os.O_DIRECTORY)
            try: os.fsync(fd)
            finally: os.close(fd)
        os.rename(stage, root)
        fd=os.open(root.parent, os.O_DIRECTORY)
        try: os.fsync(fd)
        finally: os.close(fd)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True); raise


def materialize_seed(*, source_root: Path, output_root: Path, qualification_binding: dict,
                     profile_binding: dict, gates_binding: dict, lease_id: str,
                     lease_expiry: int) -> dict:
    """Create a fail-closed successor closure; callers own queue and budget admission."""
    source_root, output_root = Path(source_root).resolve(), Path(output_root).resolve()
    qualification_path, qualification = _bound(qualification_binding, "qualification")
    profile_path, profile = _bound(profile_binding, "profile")
    gates_path, gates = _bound(gates_binding, "gates")
    if (profile.get("schema") != "nc_rted_interleaved_formal_qualification_profile/v1" or
            set(profile.get("members", ())) != set(GROUPS) or
            qualification.get("qualification_profile") != {"path":str(profile_path), "sha256":profile_binding["sha256"]}):
        raise MaterializationError("profile and qualification differ")
    checks, evidence = gates.get("engineering_checks"), gates.get("engineering_gate_evidence")
    if (gates.get("schema") != "nc_rted_source39_recovered_engineering_gate_bindings/v1" or
            gates.get("status") != "ENGINEERING_EVIDENCE_ONLY_NOT_RESOURCE_ADMISSION" or
            gates.get("qualification") != {"path":str(qualification_path), "sha256":qualification_binding["sha256"]} or
            gates.get("profile") != {"path":str(profile_path), "sha256":profile_binding["sha256"]} or
            checks != {str(number): "PASS" for number in range(1, 11)} or
            not isinstance(evidence, dict) or set(evidence) != set(checks) or
            qualification.get("status") != "PASS_GPU_KERNEL_AND_INHERITED_RUNTIME_IMPORTS"):
        raise MaterializationError("engineering gate or qualification status differs")
    diagnostic_path, diagnostic = _bound({"path":profile["diagnostic_bundle"], "sha256":profile["diagnostic_bundle_sha256"]}, "diagnostic bundle")
    qualified_path, qualified = _bound({"path":diagnostic["source_manifest"], "sha256":diagnostic["source_manifest_sha256"]}, "qualified source manifest")
    successor = _manifest(source_root, qualified["files"])
    changed = sorted(name for name in set(qualified["files"]) | set(successor["files"])
                     if qualified["files"].get(name) != successor["files"].get(name))
    if set(changed) != ALLOWED_CHANGED:
        raise MaterializationError("successor changes are not the exact four scheduler/resource paths")
    # Worker admission keys are canonical absolute paths, while the bundle
    # closure deliberately keeps relative names for source-tree comparison.
    source_files = {str((source_root / name).resolve()): value for name, value in successor["files"].items()}
    admitted_source_sha256 = hashlib.sha256(
        json.dumps(source_files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    environment = copy.deepcopy(qualification["runtime_environment"]["environment"])
    environment["PYTHONPATH"] = str(source_root / "src")
    interpreter = qualification["runtime_environment"]["interpreter"]
    runtimes, admissions, members = {}, {}, {}
    for group in GROUPS:
        member = profile["members"][group]
        runtime_path, runtime = _bound({"path":member.get("runtime"), "sha256":member.get("runtime_sha256")}, f"{group} qualified runtime")
        runtime = copy.deepcopy(runtime); runtime["hashes"]["code_sha256"] = admitted_source_sha256; runtime["hashes"]["runtime_sha256"] = admitted_source_sha256
        runtime["run"].update({"run_id":f"formal:17:{group}:source39-segmented-v1", "checkpoint_root":str(output_root/"runs"/group/"checkpoints"), "progress_path":str(output_root/"runs"/group/"progress.json")})
        runtimes[group] = runtime
    # Current source APIs compute the new checkpoint identities after final paths are fixed.
    import sys
    sys.path.insert(0, str(source_root / "src"))
    from nc_rted.production_runtime import _checkpoint_identity, load_manifest
    from nc_rted.task_inputs import TrainingCatalog
    identities = {}
    for group in GROUPS:
        temporary = output_root.parent / f".{output_root.name}.{group}.runtime.json"
        temporary.write_bytes(canonical(runtimes[group]))
        try:
            manifest=load_manifest(temporary, expected_sha256=digest(temporary)); catalog_doc=manifest.document["catalog"]
            identities[group]=_checkpoint_identity(manifest, TrainingCatalog.load(catalog_doc["manifest_directory"], catalog_doc["training_annotations"], expected_provenance_sha256=catalog_doc["provenance_sha256"]))
        finally: temporary.unlink(missing_ok=True)
    scientific=("group","seed","data_sha256","teacher_sha256","inherited_weights_sha256")
    qualified_identities=profile["member_identities"]
    if any({key:identities[g].get(key) for key in scientific}!={key:qualified_identities[g].get(key) for key in scientific} for g in GROUPS):
        raise MaterializationError("successor changed a qualified scientific identity")
    transition={"qualified_source_manifest":{"path":str(qualified_path),"sha256":diagnostic["source_manifest_sha256"]},"successor_source_manifest":{"path":str(output_root/"successor-source-manifest.json"),"sha256":hashlib.sha256(canonical(successor)).hexdigest()},"allowed_changed_files":changed,"admitted_source_root":str(source_root),"admitted_source_files_sha256":admitted_source_sha256}
    scope={"schema":"nc_rted_source39_segmented_allocation_scope/v1","status":"PASS","lease_id":lease_id,"lease_expires_utc_epoch":lease_expiry,"resumable_segments_only":True,"full_matrix_authorized":False,"qualification":{"path":str(qualification_path),"sha256":qualification_binding["sha256"]},"profile":{"path":str(profile_path),"sha256":profile_binding["sha256"]},"gates":{"path":str(gates_path),"sha256":gates_binding["sha256"]},"engineering_checks":checks,"engineering_gate_evidence":evidence}
    scope_binding={"path":str(output_root/"allocation-scope.json"),"sha256":hashlib.sha256(canonical(scope)).hexdigest()}
    for group in GROUPS:
        admissions[group]={"schema":"nc_rted_source39_segmented_formal_admission/v1","status":"PASS","formal_execution_allowed":True,"run_identity":identities[group],"source_files":source_files,"engineering_checks":checks,"gate_evidence":evidence,"source_transition":transition,"engineering_gates":{"path":str(gates_path),"sha256":gates_binding["sha256"]},"allocation_scope":scope_binding}
        members[group]={"runtime":str(output_root/"members"/group/"runtime.json"),"runtime_sha256":hashlib.sha256(canonical(runtimes[group])).hexdigest(),"admission":str(output_root/"members"/group/"admission.json"),"admission_sha256":hashlib.sha256(canonical(admissions[group])).hexdigest()}
    source_map={"schema":"nc_rted_captured_source_map/v1","files":{path:str(Path(path).relative_to(source_root)) for path in source_files},"runtime":{"inherited_external_root":"inherited","inherited_source_manifest":"inherited-source-manifest.json","stage2_module":"stage2-module.py"}}
    bundle={"schema":"nc_rted_interleaved_formal_bundle/v1","bundle_checkpoint_root":str(output_root/"bundle-checkpoints"),"members":members,"source_manifest":str(output_root/"successor-source-manifest.json"),"source_manifest_sha256":hashlib.sha256(canonical(successor)).hexdigest(),"captured_source_map":{"path":str(output_root/"source-map.json"),"sha256":hashlib.sha256(canonical(source_map)).hexdigest()}}
    _write_tree(output_root,{Path("successor-source-manifest.json"):successor,Path("source-map.json"):source_map,Path("bundle.json"):bundle,Path("allocation-scope.json"):scope,**{Path("members")/g/"runtime.json":runtimes[g] for g in GROUPS},**{Path("members")/g/"admission.json":admissions[g] for g in GROUPS}})
    return {"bundle":{"path":str(output_root/"bundle.json"),"sha256":hashlib.sha256(canonical(bundle)).hexdigest()},"member_identities":identities,"qualified_member_identities":qualified_identities,"source_transition":transition,"member_runs":{g:runtimes[g]["run"] for g in GROUPS},"source_sha256":admitted_source_sha256,"source_root":str(source_root),"qualification":{"path":str(qualification_path),"sha256":qualification_binding["sha256"]},"profile":{"path":str(profile_path),"sha256":profile_binding["sha256"]},"gates":{"path":str(gates_path),"sha256":gates_binding["sha256"]},"scope":scope_binding,"environment":environment,"interpreter":interpreter}
