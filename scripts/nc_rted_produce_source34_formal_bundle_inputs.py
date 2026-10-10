#!/usr/bin/env python3
"""Materialize source34 seed-42/2026 formal inputs from accepted applicability.

This external producer performs no qualification, model work, or queue worker
launch.  It is deliberately run with its controller CWD set to source34 so the
relative source closure retained by the source manifests remains hash-stable.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import shutil
import socket
import sys
import tempfile
import time
from pathlib import Path


GROUPS = ("A", "U", "S", "F")
PROJECT = Path("/root/autodl-tmp/lookaway-wm").resolve()
FORMAL_DEADLINE = 1_792_724_040
RENTAL_CUTOFF = 1_792_080_000


class ProducerError(ValueError):
    pass


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def bound_json(path_value: str, expected: str, name: str) -> tuple[Path, dict]:
    path = Path(path_value).resolve()
    if not path.is_file() or not isinstance(expected, str) or len(expected) != 64 or digest(path) != expected:
        raise ProducerError(f"{name} is absent or differs from its SHA-256")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ProducerError(f"{name} is invalid JSON") from error
    if not isinstance(value, dict):
        raise ProducerError(f"{name} must be an object")
    return path, value


def import_source34(root: Path):
    root = root.resolve()
    if not (root / "src/nc_rted/resource_attestation.py").is_file() or not (root / "scripts/nc_rted_queue.py").is_file():
        raise ProducerError("runtime-source-root is not a complete source34 consumer tree")
    source = str(root / "src")
    loaded = sys.modules.get("nc_rted")
    if loaded is not None and source not in [str(Path(item).resolve().parent) for item in getattr(loaded, "__path__", ())]:
        raise ProducerError("nc_rted was already imported from a different source tree")
    if source not in sys.path:
        sys.path.insert(0, source)
    try:
        modules = tuple(importlib.import_module(name) for name in (
            "nc_rted.resource_attestation", "nc_rted.formal_bundle", "nc_rted.production_runtime", "nc_rted.queue"))
    except ImportError as error:
        raise ProducerError("cannot import source34 consumer APIs") from error
    for module, relative in zip(modules, ("resource_attestation.py", "formal_bundle.py", "production_runtime.py", "queue.py")):
        if Path(module.__file__).resolve() != root / "src/nc_rted" / relative:
            raise ProducerError("consumer import did not resolve from runtime-source-root")
    return modules


def require_scope(scope: dict, *, applicability_path: Path, applicability_sha: str, qualification: dict,
                  target_seed: int, environment: dict, interpreter: dict, host: str, gpu_uuid: str,
                  physical_gpu: int, budget: int, reserve: int, now: float) -> dict:
    required = {"schema", "status", "authorization_id", "authorized_by", "applicability", "qualification", "target_seed",
                "host", "physical_gpu", "gpu_uuid", "project_volume", "runtime_environment", "lease_id",
                "lease_expires_utc_epoch", "max_budget_seconds", "min_free_bytes", "deadline_utc_epoch",
                "engineering_gate_evidence", "engineering_checks"}
    if set(scope) != required or scope.get("schema") != "nc_rted_source34_target_resource_scope/v1" or scope.get("status") != "AUTHORIZED":
        raise ProducerError("source34 authorization scope schema or status differs")
    if (scope.get("applicability") != {"path": str(applicability_path), "sha256": applicability_sha} or
            scope.get("qualification") != qualification or scope.get("target_seed") != target_seed or
            scope.get("host") != host or scope.get("physical_gpu") != physical_gpu or scope.get("gpu_uuid") != gpu_uuid or
            scope.get("project_volume") != str(PROJECT) or scope.get("runtime_environment") != {"interpreter": interpreter, "environment": environment}):
        raise ProducerError("source34 authorization scope does not bind this target workload")
    checks, gates = scope.get("engineering_checks"), scope.get("engineering_gate_evidence")
    if checks != {str(number): "PASS" for number in range(1, 11)} or not isinstance(gates, dict) or set(gates) != set(checks):
        raise ProducerError("source34 authorization scope lacks ten accepted engineering gates")
    for number, binding in gates.items():
        if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
            raise ProducerError(f"source34 gate {number} binding differs")
        bound_json(binding["path"], binding["sha256"], f"source34 gate {number} evidence")
    values = ("max_budget_seconds", "min_free_bytes", "deadline_utc_epoch", "lease_expires_utc_epoch")
    if any(type(scope.get(key)) not in (int, float) or not math.isfinite(scope[key]) or scope[key] <= 0 for key in values):
        raise ProducerError("source34 authorization scope numeric bounds are invalid")
    if (scope["max_budget_seconds"] < budget or scope["min_free_bytes"] > reserve or scope["deadline_utc_epoch"] < FORMAL_DEADLINE or
            scope["lease_expires_utc_epoch"] > RENTAL_CUTOFF or scope["lease_expires_utc_epoch"] <= now or now + budget > scope["lease_expires_utc_epoch"]):
        raise ProducerError("source34 authorization scope does not cover the requested resource contract")
    return gates


def write_tree(root: Path, documents: dict[Path, object]) -> None:
    if root.exists():
        raise ProducerError("output-root already exists")
    stage = Path(tempfile.mkdtemp(prefix=f".{root.name}.", dir=root.parent))
    try:
        for relative, document in documents.items():
            target = stage / relative; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(canonical(document))
        os.rename(stage, root)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def preflight_queue(queue_type, *, job_key: str, payload: dict, evidence: dict[str, tuple[str, str]], directory: Path) -> None:
    descriptor, value = tempfile.mkstemp(prefix=".source34-preflight-", suffix=".sqlite3", dir=directory); os.close(descriptor)
    database = Path(value)
    try:
        queue = queue_type(database)
        for name, (path, checksum) in evidence.items(): queue.add_evidence(name, path, checksum, accepted=True)
        queue.add_job(job_key, "formal_bundle", payload)
        with queue.connect() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE job_key=?", (job_key,)).fetchone()
            failure = queue._guard_failure(connection, row)
        if failure: raise ProducerError(f"source34 queue admission rejected: {failure}")
    finally:
        for path in (database, Path(f"{database}-wal"), Path(f"{database}-shm")): path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-source-root", required=True); parser.add_argument("--applicability", required=True); parser.add_argument("--applicability-sha256", required=True)
    parser.add_argument("--authorization-scope", required=True); parser.add_argument("--authorization-scope-sha256", required=True)
    parser.add_argument("--interpreter", required=True); parser.add_argument("--physical-gpu", type=int, required=True); parser.add_argument("--run-budget-seconds", type=int, required=True)
    parser.add_argument("--min-free-bytes", type=int, default=20 * 1024 ** 3); parser.add_argument("--queue-db", required=True); parser.add_argument("--queue-run-dir", required=True)
    parser.add_argument("--bundle-checkpoint-root", required=True); parser.add_argument("--output-root", required=True); parser.add_argument("--job-key", required=True)
    parser.add_argument("--materialize", action="store_true"); parser.add_argument("--enqueue", action="store_true")
    args = parser.parse_args()
    if args.enqueue and not args.materialize: parser.error("--enqueue requires --materialize")
    if args.run_budget_seconds <= 0 or args.min_free_bytes < 20 * 1024 ** 3: parser.error("resource bounds are invalid")
    attestation, formal_bundle, runtime, queue_type = import_source34(Path(args.runtime_source_root))
    os.chdir(Path(args.runtime_source_root).resolve())  # Required for the hash-bound relative source closure.
    applicability_path, applicability = bound_json(args.applicability, args.applicability_sha256, "source34 applicability")
    qualification_binding = applicability.get("measured_qualification", {})
    qualification_path, qualification = bound_json(qualification_binding.get("path"), qualification_binding.get("sha256"), "source33 qualification")
    target_seed = applicability.get("target_seed")
    if type(target_seed) is not int or target_seed not in {42, 2026}: raise ProducerError("applicability target seed differs")
    target_bindings = applicability.get("target_runtimes")
    if not isinstance(target_bindings, dict) or set(target_bindings) != set(GROUPS): raise ProducerError("applicability target runtimes differ")
    source_binding = applicability.get("source_transition", {}).get("target_manifest", {})
    source_path, source = bound_json(source_binding.get("path"), source_binding.get("sha256"), "source34 source manifest")
    files = source.get("files")
    if (source.get("schema") != "nc_rted_interleaved_source_manifest/v1" or not isinstance(files, dict) or
            source.get("code_sha256") != hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest() or
            any(not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts or not isinstance(value, str) or
                not (Path(args.runtime_source_root) / name).is_file() or digest(Path(args.runtime_source_root) / name) != value for name, value in files.items())):
        raise ProducerError("source34 source closure differs from applicability")
    scope_path, scope = bound_json(args.authorization_scope, args.authorization_scope_sha256, "source34 authorization scope")
    runtime_environment = scope.get("runtime_environment", {})
    if not isinstance(runtime_environment, dict) or set(runtime_environment) != {"interpreter", "environment"}: raise ProducerError("scope runtime environment differs")
    interpreter, environment = runtime_environment["interpreter"], runtime_environment["environment"]
    if not isinstance(interpreter, dict) or Path(interpreter.get("path", "")).resolve() != Path(args.interpreter).resolve(): raise ProducerError("requested interpreter differs from scope")
    host, current_uuid = socket.gethostname(), attestation.gpu_uuid(args.physical_gpu)
    gates = require_scope(scope, applicability_path=applicability_path, applicability_sha=args.applicability_sha256,
                          qualification={"path": str(qualification_path), "sha256": qualification_binding["sha256"]}, target_seed=target_seed,
                          environment=environment, interpreter=interpreter, host=host, gpu_uuid=current_uuid, physical_gpu=args.physical_gpu,
                          budget=args.run_budget_seconds, reserve=args.min_free_bytes, now=time.time())
    if shutil.disk_usage(PROJECT).free < args.min_free_bytes: raise ProducerError("project volume does not meet requested reserve")
    root, run_dir, common = Path(args.output_root).resolve(), Path(args.queue_run_dir).resolve(), Path(args.bundle_checkpoint_root).resolve()
    if any(not path.is_absolute() or PROJECT not in (path, *path.parents) for path in (root, run_dir, common)): raise ProducerError("output and execution roots must be under project volume")
    identities, members, admissions = {}, {}, {}
    for group in GROUPS:
        binding = target_bindings[group]
        target_path, _ = bound_json(binding.get("path"), binding.get("sha256"), f"target runtime {group}")
        manifest = runtime.load_manifest(target_path, expected_sha256=binding["sha256"])
        identity = attestation.formal_runtime_identity(manifest)
        if identity != applicability.get("target_identities", {}).get(group) or identity.get("seed") != str(target_seed): raise ProducerError("target runtime identity differs from applicability")
        identities[group] = identity
        admissions[group] = {"schema": "nc_rted_source34_formal_admission/v1", "status": "PASS", "formal_execution_allowed": True,
                             "engineering_checks": {key: "PASS" for key in gates}, "run_identity": identity, "source_files": files,
                             "gate_evidence": gates, "producer_scope": {"path": str(scope_path), "sha256": args.authorization_scope_sha256}}
        members[group] = {"runtime": str(target_path), "runtime_sha256": binding["sha256"], "admission": str(root / "admissions" / f"{group}.json"), "admission_sha256": ""}
    if any(identity.get("code_sha256") != source["code_sha256"] for identity in identities.values()): raise ProducerError("target source identity differs")
    if not args.materialize:
        print(json.dumps({"status": "INPUT_BINDINGS_ONLY_NOT_ADMITTED", "job_key": args.job_key, "target_seed": target_seed, "output_root": str(root)}, sort_keys=True)); return
    for group in GROUPS: members[group]["admission_sha256"] = hashlib.sha256(canonical(admissions[group])).hexdigest()
    source_map = {"schema": "nc_rted_captured_source_map/v1", "files": {name: name for name in files}, "runtime": {"inherited_external_root": "inherited", "inherited_source_manifest": "inherited-source-manifest.json", "stage2_module": "stage2-module.py"}}
    bundle = {"schema": "nc_rted_interleaved_formal_bundle/v1", "bundle_checkpoint_root": str(common), "members": members,
              "source_manifest": str(source_path), "source_manifest_sha256": source_binding["sha256"], "captured_source_map": {"path": str(root / "source-map.json"), "sha256": hashlib.sha256(canonical(source_map)).hexdigest()}}
    bundle_sha = hashlib.sha256(canonical(bundle)).hexdigest()
    runs = {group: runtime.load_manifest(members[group]["runtime"], expected_sha256=members[group]["runtime_sha256"]).run for group in GROUPS}
    checkpoints = {group: str((Path(runs[group]["checkpoint_root"]) / "final" / "manifest.json").resolve()) for group in GROUPS}
    outputs = [{"path": checkpoints[group], "artifact_type": "checkpoint", "semantic": "formal_training", "run_identity": identities[group]} for group in GROUPS]
    outputs.append({"path": "bundle-report.json", "artifact_type": "report", "semantic": "formal_bundle", "member_identities": identities, "final_checkpoints": checkpoints, "bundle_checkpoint_root": str(common)})
    command = [interpreter["path"], str(Path(args.runtime_source_root).resolve() / "scripts/nc_rted_interleaved_formal.py"), "--bundle", str(root / "bundle.json"), "--bundle-sha256", bundle_sha, "--captured-root", "{queue_capture}", "--output", "bundle-report.json", "--resume", "auto"]
    payload = {"physical_gpu": args.physical_gpu, "bundle_config": str(root / "bundle.json"), "bundle_config_sha256": bundle_sha, "member_identities": identities, "frozen_source_sha256": source["code_sha256"], "bundle_evidence": f"{args.job_key}:bundle", "resource_authorization_evidence": f"{args.job_key}:authorization", "data_volume": str(PROJECT), "min_free_bytes": args.min_free_bytes, "run_budget_seconds": args.run_budget_seconds, "deadline_utc_epoch": FORMAL_DEADLINE, "execution_environment": environment, "interpreter": interpreter, "run_dir": str(run_dir), "progress_path": str(Path(runs["A"]["progress_path"])), "checkpoint_roots": [str(Path(runs[group]["checkpoint_root"])) for group in GROUPS], "bundle_checkpoint_root": str(common), "expected_outputs": outputs, "command": command, "resource_attestation": str(root / "bundle-resource-attestation.json"), "resource_attestation_sha256": ""}
    authorization = {"schema": "nc_rted_resource_authorization/v1", "status": "PASS", "host": host, "gpu_uuid": current_uuid, "lease_id": scope["lease_id"], "project_volume": str(PROJECT), "max_budget_seconds": args.run_budget_seconds, "min_free_bytes": args.min_free_bytes, "deadline_utc_epoch": FORMAL_DEADLINE, "lease_expires_utc_epoch": scope["lease_expires_utc_epoch"], "source_scope": {"path": str(scope_path), "sha256": args.authorization_scope_sha256}}
    attestation_doc = {"schema": "nc_rted_formal_bundle_resource_attestation/v1", "status": "PASS", "binding": {"job_key": args.job_key, "bundle_config": payload["bundle_config"], "bundle_config_sha256": bundle_sha, "member_identities": identities, "frozen_source_sha256": source["code_sha256"], "execution_inputs": {key: payload[key] for key in ("command", "execution_environment", "interpreter", "run_dir", "progress_path", "checkpoint_roots", "bundle_checkpoint_root", "expected_outputs")}}, "execution": {"host": host, "physical_gpu": args.physical_gpu, "gpu_uuid": current_uuid, "lease_id": authorization["lease_id"], "lease_expires_utc_epoch": authorization["lease_expires_utc_epoch"], "environment": environment, "interpreter": interpreter}, "qualification": {"accepted_evidence_name": f"{args.job_key}:qualification", "path": str(qualification_path), "sha256": qualification_binding["sha256"]}, "authorization": {"path": str(root / "resource-authorization.json"), "sha256": hashlib.sha256(canonical(authorization)).hexdigest()}, "contract": {key: payload[key] for key in ("data_volume", "min_free_bytes", "run_budget_seconds", "deadline_utc_epoch")}, "source34_applicability": {"path": str(applicability_path), "sha256": args.applicability_sha256}}
    payload["resource_attestation_sha256"] = hashlib.sha256(canonical(attestation_doc)).hexdigest()
    documents = {Path("source-map.json"): source_map, Path("bundle.json"): bundle, Path("resource-authorization.json"): authorization, Path("bundle-resource-attestation.json"): attestation_doc, **{Path("admissions") / f"{group}.json": document for group, document in admissions.items()}}
    evidence = {payload["bundle_evidence"]: (payload["bundle_config"], bundle_sha), f"{args.job_key}:qualification": (str(qualification_path), qualification_binding["sha256"]), payload["resource_authorization_evidence"]: (str(root / "resource-authorization.json"), attestation_doc["authorization"]["sha256"])}
    write_tree(root, documents)
    try:
        formal_bundle.load_bundle(payload["bundle_config"], bundle_sha); formal_bundle.validate_members(bundle); attestation.verify_bundle_attestation(payload, args.job_key, evidence)
        preflight_queue(queue_type.JobQueue, job_key=args.job_key, payload=payload, evidence=evidence, directory=root.parent)
    except BaseException:
        shutil.rmtree(root, ignore_errors=True); raise
    if args.enqueue:
        queue = queue_type.JobQueue(args.queue_db)
        for name, (path, checksum) in evidence.items(): queue.add_evidence(name, path, checksum, accepted=True)
        queue.add_job(args.job_key, "formal_bundle", payload)
    print(json.dumps({"status": "ENQUEUED" if args.enqueue else "MATERIALIZED", "job_key": args.job_key, "target_seed": target_seed, "output_root": str(root)}, sort_keys=True))


if __name__ == "__main__":
    try: main()
    except ProducerError as error: raise SystemExit(f"source34 formal input producer refused: {error}") from error
