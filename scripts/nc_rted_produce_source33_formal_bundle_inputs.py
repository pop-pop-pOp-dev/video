#!/usr/bin/env python3
"""Produce seed-17 formal-bundle inputs from frozen source33 evidence.

This external tool never runs a model or the formal runner.  It imports only
the source33 consumer APIs selected by ``--runtime-source-root``.  Materialize
is explicit; dry-run is the default and leaves no output files behind.
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
RENTAL_CUTOFF = 1_792_080_000
FORMAL_DEADLINE = 1_792_724_040


class ProducerError(ValueError):
    pass


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def bound_json(value: str, expected: str, name: str) -> tuple[Path, dict]:
    path = Path(value).resolve()
    if (not path.is_absolute() or not path.is_file() or not isinstance(expected, str) or
            len(expected) != 64 or digest(path) != expected):
        raise ProducerError(f"{name} is absent or differs from its SHA-256")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ProducerError(f"{name} is invalid JSON") from error
    if not isinstance(document, dict):
        raise ProducerError(f"{name} must be an object")
    return path, document


def import_source33(root: Path):
    root = root.resolve()
    if not (root / "src/nc_rted/resource_attestation.py").is_file() or not (root / "scripts/nc_rted_queue.py").is_file():
        raise ProducerError("runtime-source-root is not a complete source33 consumer tree")
    source = str(root / "src")
    loaded = sys.modules.get("nc_rted")
    if loaded is not None and source not in [str(Path(item).resolve()) for item in getattr(loaded, "__path__", ())]:
        raise ProducerError("nc_rted was already imported from a different source tree")
    if source not in sys.path:
        sys.path.insert(0, source)
    try:
        attestation = importlib.import_module("nc_rted.resource_attestation")
        formal_bundle = importlib.import_module("nc_rted.formal_bundle")
        runtime = importlib.import_module("nc_rted.production_runtime")
        queue = importlib.import_module("nc_rted.queue")
    except ImportError as error:
        raise ProducerError("cannot import the frozen source33 consumer APIs") from error
    for module, relative in ((attestation, "resource_attestation.py"), (formal_bundle, "formal_bundle.py"),
                             (runtime, "production_runtime.py"), (queue, "queue.py")):
        if Path(module.__file__).resolve() != root / "src/nc_rted" / relative:
            raise ProducerError("consumer import did not resolve from runtime-source-root")
    return attestation, formal_bundle, runtime, queue


def require_scope(scope: dict, *, qualification_path: Path, qualification_sha: str, profile_path: Path,
                  profile_sha: str, runtime_environment: dict, physical_gpu: int, host: str, gpu_uuid: str,
                  run_budget: int, reserve: int, now: float) -> dict:
    required = {"schema", "status", "authorization_id", "authorized_by", "qualification", "profile",
                "host", "physical_gpu", "gpu_uuid", "project_volume", "runtime_environment", "lease_id",
                "lease_expires_utc_epoch", "max_budget_seconds", "min_free_bytes", "deadline_utc_epoch",
                "engineering_gate_evidence", "engineering_checks"}
    if set(scope) != required or scope.get("schema") != "nc_rted_source33_formal_resource_scope/v1" or scope.get("status") != "AUTHORIZED":
        raise ProducerError("authorization scope schema or status differs")
    if (not isinstance(scope.get("authorization_id"), str) or not scope["authorization_id"] or
            not isinstance(scope.get("authorized_by"), str) or not scope["authorized_by"] or
            scope.get("qualification") != {"path": str(qualification_path), "sha256": qualification_sha} or
            scope.get("profile") != {"path": str(profile_path), "sha256": profile_sha} or
            scope.get("host") != host or scope.get("physical_gpu") != physical_gpu or scope.get("gpu_uuid") != gpu_uuid or
            scope.get("project_volume") != str(PROJECT) or scope.get("runtime_environment") != runtime_environment):
        raise ProducerError("authorization scope does not bind this measured source33 workload")
    gates = scope.get("engineering_gate_evidence")
    checks = scope.get("engineering_checks")
    if not isinstance(checks, dict) or checks != {str(item): "PASS" for item in range(1, 11)}:
        raise ProducerError("authorization scope does not declare ten accepted engineering gate results")
    if not isinstance(gates, dict) or set(gates) != {str(item) for item in range(1, 11)}:
        raise ProducerError("authorization scope lacks ten explicit engineering gate evidence bindings")
    for number, binding in gates.items():
        if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
            raise ProducerError(f"engineering gate {number} binding differs")
        bound_json(binding["path"], binding["sha256"], f"engineering gate {number} evidence")
    values = ("max_budget_seconds", "min_free_bytes", "deadline_utc_epoch", "lease_expires_utc_epoch")
    if any(type(scope.get(key)) not in (int, float) or not math.isfinite(scope[key]) or scope[key] <= 0 for key in values):
        raise ProducerError("authorization scope numeric bounds are invalid")
    if (scope["max_budget_seconds"] < run_budget or scope["min_free_bytes"] > reserve or
            scope["deadline_utc_epoch"] < FORMAL_DEADLINE or scope["lease_expires_utc_epoch"] > RENTAL_CUTOFF or
            scope["lease_expires_utc_epoch"] <= now or now + run_budget > scope["lease_expires_utc_epoch"]):
        raise ProducerError("authorization scope does not cover the requested resource contract")
    return gates


def checked_profile(profile: dict, profile_path: Path, profile_sha: str, qualification: dict, runtime, source_root: Path) -> tuple[dict, dict, dict]:
    required_profile = {"schema", "status", "diagnostic_bundle", "diagnostic_bundle_sha256", "members",
                        "member_identities", "updates", "accumulation", "shared_preparation",
                        "common_recovery", "complete_long_input_coverage"}
    if (set(profile) != required_profile or
            profile.get("schema") != "nc_rted_interleaved_formal_qualification_profile/v1" or
            profile.get("status") != "NON_ADMITTED_FORMAL_PROFILE" or profile.get("updates") != 1000 or
            profile.get("accumulation") != 8 or profile.get("shared_preparation") != "frozen_provider_only" or
            profile.get("common_recovery") is not True or not isinstance(profile.get("complete_long_input_coverage"), dict)):
        raise ProducerError("qualification profile is not the non-admitted source33 profile")
    members = profile.get("members")
    identities = profile.get("member_identities")
    if not isinstance(members, dict) or not isinstance(identities, dict) or set(members) != set(GROUPS) or set(identities) != set(GROUPS):
        raise ProducerError("qualification profile must bind exactly A/U/S/F")
    bundle_path, bundle = bound_json(profile["diagnostic_bundle"], profile["diagnostic_bundle_sha256"], "diagnostic bundle")
    source_path, source = bound_json(bundle["source_manifest"], bundle["source_manifest_sha256"], "source manifest")
    if source_path.parent != bundle_path.parent and not source_path.is_file():
        raise ProducerError("diagnostic source manifest is unavailable")
    files = source.get("files")
    if source.get("schema") != "nc_rted_interleaved_source_manifest/v1" or not isinstance(files, dict):
        raise ProducerError("diagnostic source manifest differs")
    if any(not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts or
           not isinstance(value, str) or len(value) != 64 or not (source_root / relative).is_file() or
           digest(source_root / relative) != value for relative, value in files.items()):
        raise ProducerError("frozen source33 closure differs from the diagnostic manifest")
    for group in GROUPS:
        item = members[group]
        if not isinstance(item, dict) or set(item) != {"runtime", "runtime_sha256"}:
            raise ProducerError("profile member binding differs")
        manifest = runtime.load_manifest(item["runtime"], expected_sha256=item["runtime_sha256"])
        identity = runtime_module_identity(runtime, manifest)
        if manifest.run.get("mode") != "formal" or manifest.run.get("group") != group or identity != identities[group]:
            raise ProducerError("profile member runtime/identity differs")
    expected_workload = {"member_identities": identities, "updates": 1000, "kind": "formal_bundle",
                         "shared_preparation": "frozen_provider_only", "common_recovery": True}
    if (qualification.get("workload") != expected_workload or qualification.get("source_sha256") != source.get("code_sha256") or
            qualification.get("qualification_profile") != {"path": str(profile_path), "sha256": profile_sha}):
        raise ProducerError("qualification does not measure this exact profile workload")
    if any(identity.get("seed") != "17" for identity in identities.values()):
        raise ProducerError("source33 producer is intentionally limited to its measured seed-17 identities")
    return bundle, source, identities


def runtime_module_identity(runtime, manifest):
    catalog_doc = manifest.document["catalog"]
    catalog = importlib.import_module("nc_rted.task_inputs").TrainingCatalog.load(
        catalog_doc["manifest_directory"], catalog_doc["training_annotations"],
        expected_provenance_sha256=catalog_doc["provenance_sha256"],
    )
    return runtime._checkpoint_identity(manifest, catalog)


def write_tree(root: Path, documents: dict[Path, object]) -> None:
    if root.exists():
        raise ProducerError("output-root already exists")
    stage = Path(tempfile.mkdtemp(prefix=f".{root.name}.", dir=root.parent))
    try:
        for relative, document in documents.items():
            target = stage / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(canonical(document))
            with target.open("rb") as stream:
                os.fsync(stream.fileno())
        for directory, _, _ in os.walk(stage, topdown=False):
            descriptor = os.open(directory, os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        os.rename(stage, root)
        descriptor = os.open(root.parent, os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def materialize_and_validate(root: Path, documents: dict[Path, object], validate) -> None:
    """Publish a candidate only while the source33 consumers accept it.

    The consumers require the final absolute paths embedded in the documents,
    so they must run after the atomic rename.  A failed check removes only the
    tree this invocation created; an existing output root is rejected before
    staging in ``write_tree``.
    """
    write_tree(root, documents)
    try:
        validate()
    except BaseException:
        shutil.rmtree(root, ignore_errors=True)
        raise


def preflight_queue(queue_type, *, job_key: str, payload: dict, evidence: dict[str, tuple[str, str]], directory: Path) -> None:
    """Run source33's admission guard against an isolated copy of the job."""
    descriptor, value = tempfile.mkstemp(prefix=".formal-bundle-preflight-", suffix=".sqlite3", dir=directory)
    os.close(descriptor)
    database = Path(value)
    try:
        queue = queue_type(database)
        for name, (path, checksum) in evidence.items():
            queue.add_evidence(name, path, checksum, accepted=True)
        queue.add_job(job_key, "formal_bundle", payload)
        with queue.connect() as db:
            row = db.execute("SELECT * FROM jobs WHERE job_key=?", (job_key,)).fetchone()
            failure = queue._guard_failure(db, row)
        if failure:
            raise ProducerError(f"source33 queue admission rejected: {failure}")
    finally:
        for path in (database, Path(f"{database}-wal"), Path(f"{database}-shm")):
            path.unlink(missing_ok=True)


def enqueue_after_preflight(queue_type, *, queue_db: str, job_key: str, payload: dict,
                            evidence: dict[str, tuple[str, str]], directory: Path) -> None:
    preflight_queue(queue_type, job_key=job_key, payload=payload, evidence=evidence, directory=directory)
    queue = queue_type(queue_db)
    for name, (path, checksum) in evidence.items():
        queue.add_evidence(name, path, checksum, accepted=True)
    queue.add_job(job_key, "formal_bundle", payload)


def require_disjoint_roots(*paths: Path) -> None:
    resolved = [path.resolve() for path in paths]
    for index, path in enumerate(resolved):
        if any(path == other or path in other.parents or other in path.parents for other in resolved[index + 1:]):
            raise ProducerError("formal input and execution roots must be distinct and nonoverlapping")


def build_documents(args, modules):
    attestation, formal_bundle, runtime, _queue = modules
    profile_path, profile = bound_json(args.qualification_profile, args.qualification_profile_sha256, "qualification profile")
    qualification_path, qualification = bound_json(args.qualification_report, args.qualification_report_sha256, "qualification report")
    scope_path, scope = bound_json(args.authorization_scope, args.authorization_scope_sha256, "authorization scope")
    now = time.time()
    bundle, source, identities = checked_profile(profile, profile_path, args.qualification_profile_sha256, qualification, runtime, Path(args.runtime_source_root).resolve())
    runtime_environment = qualification.get("runtime_environment")
    if not isinstance(runtime_environment, dict) or set(runtime_environment) != {"interpreter", "environment"}:
        raise ProducerError("qualification runtime environment differs")
    interpreter, environment = runtime_environment["interpreter"], runtime_environment["environment"]
    if not isinstance(interpreter, dict) or Path(interpreter.get("path", "")).resolve() != Path(args.interpreter).resolve():
        raise ProducerError("requested interpreter differs from the qualified interpreter")
    host, current_uuid = socket.gethostname(), attestation.gpu_uuid(args.physical_gpu)
    if qualification.get("host") != host or qualification.get("gpu_uuid") != current_uuid:
        raise ProducerError("qualification host or physical GPU UUID differs")
    attestation.validate_environment(environment, PROJECT)
    if shutil.disk_usage(PROJECT).free < args.min_free_bytes:
        raise ProducerError("project volume does not meet the requested reserve")
    gates = require_scope(scope, qualification_path=qualification_path, qualification_sha=args.qualification_report_sha256,
                          profile_path=profile_path, profile_sha=args.qualification_profile_sha256,
                          runtime_environment=runtime_environment, physical_gpu=args.physical_gpu, host=host,
                          gpu_uuid=current_uuid, run_budget=args.run_budget_seconds, reserve=args.min_free_bytes, now=now)
    root = Path(args.output_root).resolve()
    run_dir = Path(args.queue_run_dir).resolve()
    common = Path(args.bundle_checkpoint_root).resolve()
    if any(not path.is_absolute() or PROJECT not in (path, *path.parents) for path in (root, run_dir, common)):
        raise ProducerError("output and execution roots must be canonical descendants of the project volume")
    require_disjoint_roots(root, run_dir, common)
    # The frozen runtime identity hashes the manifest's relative keys.  The
    # queue controller must therefore run from runtime-source-root while it
    # captures this relative-key closure; the captured map then redirects all
    # subsequent runner reads to its immutable copy.
    source_files = dict(source["files"])
    source_map = {"schema": "nc_rted_captured_source_map/v1", "files": {relative: relative for relative in source_files},
                  "runtime": {"inherited_external_root": "inherited", "inherited_source_manifest": "inherited-source-manifest.json", "stage2_module": "stage2-module.py"}}
    admissions, members = {}, {}
    for group in GROUPS:
        item = profile["members"][group]
        admission = {"schema": "nc_rted_source33_formal_admission/v1", "status": "PASS", "formal_execution_allowed": True,
                     "engineering_checks": {key: "PASS" for key in gates}, "run_identity": identities[group],
                     "source_files": source_files, "gate_evidence": gates,
                     "producer_scope": {"path": str(scope_path), "sha256": args.authorization_scope_sha256}}
        admissions[group] = admission
        members[group] = {"runtime": item["runtime"], "runtime_sha256": item["runtime_sha256"],
                          "admission": str(root / "admissions" / f"{group}.json"), "admission_sha256": ""}
    authorization = {"schema": "nc_rted_resource_authorization/v1", "status": "PASS", "host": host, "gpu_uuid": current_uuid,
                     "lease_id": scope["lease_id"], "project_volume": str(PROJECT), "max_budget_seconds": args.run_budget_seconds,
                     "min_free_bytes": args.min_free_bytes, "deadline_utc_epoch": FORMAL_DEADLINE,
                     "lease_expires_utc_epoch": scope["lease_expires_utc_epoch"],
                     "source_scope": {"path": str(scope_path), "sha256": args.authorization_scope_sha256}}
    return root, profile, bundle, source, identities, source_map, admissions, members, authorization, interpreter, environment, run_dir, common, qualification_path, scope_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-source-root", required=True)
    parser.add_argument("--qualification-profile", required=True); parser.add_argument("--qualification-profile-sha256", required=True)
    parser.add_argument("--qualification-report", required=True); parser.add_argument("--qualification-report-sha256", required=True)
    parser.add_argument("--authorization-scope", required=True); parser.add_argument("--authorization-scope-sha256", required=True)
    parser.add_argument("--interpreter", required=True); parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--run-budget-seconds", type=int, required=True); parser.add_argument("--min-free-bytes", type=int, default=20 * 1024 ** 3)
    parser.add_argument("--queue-db", required=True); parser.add_argument("--queue-run-dir", required=True); parser.add_argument("--bundle-checkpoint-root", required=True)
    parser.add_argument("--output-root", required=True); parser.add_argument("--job-key", default="formal-bundle-seed17-source33")
    parser.add_argument("--materialize", action="store_true"); parser.add_argument("--enqueue", action="store_true")
    args = parser.parse_args()
    if args.enqueue and not args.materialize:
        parser.error("--enqueue requires --materialize")
    if args.run_budget_seconds <= 0 or args.min_free_bytes < 20 * 1024 ** 3:
        parser.error("resource bounds are invalid")
    modules = import_source33(Path(args.runtime_source_root))
    # Frozen source33 captures relative formal source keys via Path(key), so
    # every preflight below must use the same source-root working directory as
    # the later queue controller.
    os.chdir(Path(args.runtime_source_root).resolve())
    (root, profile, diagnostic_bundle, source, identities, source_map, admissions, members, authorization, interpreter, environment,
     run_dir, common, qualification_path, scope_path) = build_documents(args, modules)
    if not args.materialize:
        print(json.dumps({"status": "INPUT_BINDINGS_ONLY_NOT_ADMITTED", "output_root": str(root), "job_key": args.job_key,
                          "profile": str(Path(args.qualification_profile).resolve()), "qualification": str(qualification_path),
                          "authorization_scope": str(scope_path), "member_identities": identities}, sort_keys=True))
        return
    # Hash admissions before constructing their bundle references.
    temporary = Path(tempfile.mkdtemp(prefix=".formal-input-hashes.", dir=root.parent))
    try:
        for group, document in admissions.items():
            target = temporary / "admissions" / f"{group}.json"; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(canonical(document))
            members[group]["admission_sha256"] = digest(target)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    bundle = {"schema": "nc_rted_interleaved_formal_bundle/v1", "bundle_checkpoint_root": str(common), "members": members,
              "source_manifest": diagnostic_bundle["source_manifest"], "source_manifest_sha256": diagnostic_bundle["source_manifest_sha256"],
              "captured_source_map": {"path": str(root / "source-map.json"), "sha256": ""}}
    source_map_sha = hashlib.sha256(canonical(source_map)).hexdigest(); bundle["captured_source_map"]["sha256"] = source_map_sha
    bundle_sha = hashlib.sha256(canonical(bundle)).hexdigest()
    member_runs = {group: json.loads(Path(profile["members"][group]["runtime"]).read_text(encoding="utf-8"))["run"] for group in GROUPS}
    require_disjoint_roots(root, run_dir, common,
                           *(Path(member_runs[group]["checkpoint_root"]).resolve() for group in GROUPS),
                           *(Path(member_runs[group]["progress_path"]).resolve() for group in GROUPS))
    checkpoints = {group: str((Path(member_runs[group]["checkpoint_root"]) / "final" / "manifest.json").resolve()) for group in GROUPS}
    outputs = [{"path": checkpoints[group], "artifact_type": "checkpoint", "semantic": "formal_training", "run_identity": identities[group]} for group in GROUPS]
    outputs.append({"path": "bundle-report.json", "artifact_type": "report", "semantic": "formal_bundle", "member_identities": identities,
                    "final_checkpoints": checkpoints, "bundle_checkpoint_root": str(common)})
    command = [interpreter["path"], str((Path(args.runtime_source_root).resolve() / "scripts/nc_rted_interleaved_formal.py")), "--bundle", str(root / "bundle.json"), "--bundle-sha256", bundle_sha,
               "--captured-root", "{queue_capture}", "--output", "bundle-report.json", "--resume", "auto"]
    payload = {"physical_gpu": args.physical_gpu, "bundle_config": str(root / "bundle.json"), "bundle_config_sha256": bundle_sha,
               "member_identities": identities, "frozen_source_sha256": source["code_sha256"], "bundle_evidence": f"{args.job_key}:bundle",
               "resource_authorization_evidence": f"{args.job_key}:authorization", "data_volume": str(PROJECT), "min_free_bytes": args.min_free_bytes,
               "run_budget_seconds": args.run_budget_seconds, "deadline_utc_epoch": FORMAL_DEADLINE, "execution_environment": environment,
               "interpreter": interpreter, "run_dir": str(run_dir), "progress_path": str(Path(member_runs["A"]["progress_path"]).resolve()),
               "checkpoint_roots": [str(Path(member_runs[group]["checkpoint_root"]).resolve()) for group in GROUPS],
               "bundle_checkpoint_root": str(common), "expected_outputs": outputs, "command": command,
               "resource_attestation": str(root / "bundle-resource-attestation.json"), "resource_attestation_sha256": ""}
    attestation_doc = {"schema": "nc_rted_formal_bundle_resource_attestation/v1", "status": "PASS",
                       "binding": {"job_key": args.job_key, "bundle_config": payload["bundle_config"], "bundle_config_sha256": bundle_sha,
                                   "member_identities": identities, "frozen_source_sha256": source["code_sha256"],
                                   "execution_inputs": {key: payload[key] for key in ("command", "execution_environment", "interpreter", "run_dir", "progress_path", "checkpoint_roots", "bundle_checkpoint_root", "expected_outputs")}},
                       "execution": {"host": socket.gethostname(), "physical_gpu": args.physical_gpu, "gpu_uuid": modules[0].gpu_uuid(args.physical_gpu),
                                     "lease_id": authorization["lease_id"], "lease_expires_utc_epoch": authorization["lease_expires_utc_epoch"], "environment": environment, "interpreter": interpreter},
                       "qualification": {"accepted_evidence_name": f"{args.job_key}:qualification", "path": str(qualification_path), "sha256": args.qualification_report_sha256},
                       "authorization": {"path": str(root / "resource-authorization.json"), "sha256": hashlib.sha256(canonical(authorization)).hexdigest()},
                       "contract": {key: payload[key] for key in ("data_volume", "min_free_bytes", "run_budget_seconds", "deadline_utc_epoch")}}
    payload["resource_attestation_sha256"] = hashlib.sha256(canonical(attestation_doc)).hexdigest()
    documents = {Path("source-map.json"): source_map, Path("bundle.json"): bundle, Path("resource-authorization.json"): authorization,
                 Path("bundle-resource-attestation.json"): attestation_doc, **{Path("admissions") / f"{group}.json": doc for group, doc in admissions.items()}}
    evidence = {payload["bundle_evidence"]: (payload["bundle_config"], bundle_sha), f"{args.job_key}:qualification": (str(qualification_path), args.qualification_report_sha256),
                payload["resource_authorization_evidence"]: (str(root / "resource-authorization.json"), attestation_doc["authorization"]["sha256"])}

    def validate_consumers() -> None:
        modules[1].load_bundle(payload["bundle_config"], payload["bundle_config_sha256"]); modules[1].validate_members(json.loads((root / "bundle.json").read_text()))
        modules[0].verify_bundle_attestation(payload, args.job_key, evidence)

    materialize_and_validate(root, documents, validate_consumers)
    if args.enqueue:
        enqueue_after_preflight(modules[3].JobQueue, queue_db=args.queue_db, job_key=args.job_key,
                                payload=payload, evidence=evidence, directory=root.parent)
    print(json.dumps({"status": "MATERIALIZED" if not args.enqueue else "ENQUEUED", "output_root": str(root), "bundle_sha256": bundle_sha,
                      "resource_attestation_sha256": payload["resource_attestation_sha256"], "job_key": args.job_key}, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except ProducerError as error:
        raise SystemExit(f"formal input producer refused: {error}") from error
