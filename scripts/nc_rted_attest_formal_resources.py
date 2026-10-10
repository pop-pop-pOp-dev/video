#!/usr/bin/env python3
"""Create a hash-bound, local-host formal resource attestation.

This command is intentionally local-only. Run it on the same GPU host that
runs ``nc_rted_queue.py``; it does not grant a remote SSH child permission.
The queue holds the matching local GPU flock for the child lifetime.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nc_rted.production_runtime import load_formal_admission, load_manifest
from nc_rted.resource_attestation import (FORMAL_DEADLINE, RENTAL_CUTOFF, SCHEMA,
                                          formal_runtime_identity, gpu_uuid, publish_attestation, sha256_file,
                                          PROJECT_VOLUME, validate_environment, verify_attestation)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a local-only formal resource attestation")
    parser.add_argument("--job-key", required=True)
    parser.add_argument("--runtime-config", required=True)
    parser.add_argument("--runtime-config-sha256", required=True)
    parser.add_argument("--formal-admission", required=True)
    parser.add_argument("--formal-admission-sha256", required=True)
    parser.add_argument("--qualification-report", required=True)
    parser.add_argument("--qualification-report-sha256", required=True)
    parser.add_argument("--resource-authorization", required=True)
    parser.add_argument("--resource-authorization-sha256", required=True)
    parser.add_argument("--accepted-evidence-name", required=True)
    parser.add_argument("--resource-authorization-evidence-name", required=True)
    parser.add_argument("--runtime-evidence-name", required=True)
    parser.add_argument("--formal-admission-evidence-name", required=True)
    parser.add_argument("--physical-gpu", required=True, type=int)
    parser.add_argument("--data-volume", required=True)
    parser.add_argument("--min-free-bytes", required=True, type=int)
    parser.add_argument("--run-budget-seconds", required=True, type=int)
    parser.add_argument("--deadline-utc-epoch", required=True, type=float)
    parser.add_argument("--lease-id", required=True)
    parser.add_argument("--lease-expires-utc-epoch", required=True, type=float)
    parser.add_argument("--execution-environment", required=True, help="canonical JSON allowlist")
    parser.add_argument("--interpreter", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--expected-outputs", required=True, help="canonical JSON formal output contract")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.min_free_bytes <= 0 or args.run_budget_seconds <= 0 or not args.lease_id:
        parser.error("resource contract values must be positive and lease-id nonempty")
    now = time.time()
    if args.deadline_utc_epoch <= now or args.lease_expires_utc_epoch <= now:
        parser.error("deadline and resource lease must be in the future")
    if args.deadline_utc_epoch != FORMAL_DEADLINE or args.lease_expires_utc_epoch > RENTAL_CUTOFF or now + args.run_budget_seconds > args.lease_expires_utc_epoch:
        parser.error("deadline, rental cutoff, or run budget is outside the accepted interval")
    try:
        environment=json.loads(args.execution_environment); expected_outputs=json.loads(args.expected_outputs)
    except ValueError as error:
        parser.error(f"execution inputs must be JSON: {error}")
    if (not isinstance(environment,dict) or
            any(not isinstance(key,str) or not isinstance(value,str) for key,value in environment.items()) or
            not isinstance(expected_outputs,list) or len(expected_outputs)!=1):
        parser.error("execution environment or formal output contract is invalid")
    interpreter=Path(args.interpreter)
    if not interpreter.is_absolute() or not interpreter.is_file(): parser.error("interpreter must be an existing absolute file")
    qualification = Path(args.qualification_report)
    if not qualification.is_absolute() or not qualification.is_file() or sha256_file(qualification) != args.qualification_report_sha256:
        parser.error("qualification report is absent or its SHA-256 differs")
    try:
        qualification_doc = json.loads(qualification.read_text())
    except (OSError, ValueError) as error:
        parser.error(f"qualification report is invalid JSON: {error}")
    if qualification_doc.get("status") != "PASS_GPU_KERNEL_AND_INHERITED_RUNTIME_IMPORTS":
        parser.error("qualification report does not establish the accepted runtime status")
    authorization=Path(args.resource_authorization)
    if not authorization.is_absolute() or not authorization.is_file() or sha256_file(authorization) != args.resource_authorization_sha256:
        parser.error("resource authorization is absent or its SHA-256 differs")
    try: authorization_doc=json.loads(authorization.read_text())
    except (OSError, ValueError) as error: parser.error(f"resource authorization is invalid JSON: {error}")
    manifest = load_manifest(args.runtime_config, expected_sha256=args.runtime_config_sha256)
    admission = load_formal_admission(args.formal_admission, expected_sha256=args.formal_admission_sha256)
    identity = formal_runtime_identity(manifest)
    files = admission.get("source_files")
    digest = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if manifest.run.get("mode") != "formal" or admission.get("run_identity") != identity or manifest.document["hashes"]["code_sha256"] != digest:
        parser.error("runtime, formal admission, and frozen source identity do not match")
    volume = Path(args.data_volume)
    if not volume.is_absolute() or shutil.disk_usage(volume).free < args.min_free_bytes:
        parser.error("data volume does not meet the requested free-space reserve")
    if (authorization_doc.get("schema") != "nc_rted_resource_authorization/v1" or authorization_doc.get("status") != "PASS" or
            authorization_doc.get("host") != socket.gethostname() or authorization_doc.get("gpu_uuid") != gpu_uuid(args.physical_gpu) or
            authorization_doc.get("lease_id") != args.lease_id or authorization_doc.get("project_volume") != str(volume.resolve()) or
            authorization_doc.get("max_budget_seconds", 0) < args.run_budget_seconds or authorization_doc.get("lease_expires_utc_epoch", 0) < args.lease_expires_utc_epoch):
        parser.error("resource authorization does not cover the requested contract")
    if str(volume) != str(PROJECT_VOLUME.resolve()):
        parser.error("data volume must be the canonical approved project volume")
    validate_environment(environment, volume)
    entry=str((Path(__file__).resolve().parent / "nc_rted_train.py").resolve())
    # Preserve the qualified venv launcher in argv.  Its resolved ELF target
    # is recorded separately so verification does not silently replace it.
    command=[str(interpreter),entry,"--config",str(Path(args.runtime_config).resolve()),"--config-sha256",args.runtime_config_sha256,"--mode","formal","--admission",str(Path(args.formal_admission).resolve()),"--admission-sha256",args.formal_admission_sha256]
    execution_inputs={"command":command,"execution_environment":environment,"interpreter":{"path":str(interpreter),"target_sha256":sha256_file(interpreter.resolve()),"launcher_sha256":sha256_file(interpreter)},"run_dir":str(Path(args.run_dir).resolve()),"progress_path":manifest.run["progress_path"],"checkpoint_root":manifest.run["checkpoint_root"],"expected_outputs":expected_outputs}
    document = {
        "schema": SCHEMA, "status": "PASS", "issued_at_utc_epoch": now,
        "binding": {"job_key": args.job_key, "runtime_config": str(Path(args.runtime_config).resolve()), "runtime_config_sha256": args.runtime_config_sha256,
                    "formal_admission": str(Path(args.formal_admission).resolve()), "formal_admission_sha256": args.formal_admission_sha256,
                    "run_identity": identity, "frozen_source_sha256": digest, "execution_inputs": execution_inputs},
        "execution": {"host": socket.gethostname(), "physical_gpu": args.physical_gpu, "gpu_uuid": gpu_uuid(args.physical_gpu),
                      "lease_id": args.lease_id, "lease_expires_utc_epoch": args.lease_expires_utc_epoch,
                      "environment": environment, "interpreter": execution_inputs["interpreter"]},
        "qualification": {"accepted_evidence_name": args.accepted_evidence_name, "path": str(qualification.resolve()),
                            "sha256": args.qualification_report_sha256, "status": qualification_doc["status"]},
        "authorization": {"path":str(authorization.resolve()),"sha256":args.resource_authorization_sha256},
        "contract": {"data_volume": str(volume.resolve()), "min_free_bytes": args.min_free_bytes,
                     "run_budget_seconds": args.run_budget_seconds, "deadline_utc_epoch": args.deadline_utc_epoch},
    }
    output = Path(args.output)
    def queue_payload(attestation_sha256):
        return {"physical_gpu": args.physical_gpu, "run_identity": identity,
                "runtime_config": str(Path(args.runtime_config).resolve()), "runtime_config_sha256": args.runtime_config_sha256,
                "formal_admission": str(Path(args.formal_admission).resolve()), "formal_admission_sha256": args.formal_admission_sha256,
                "frozen_source_sha256": digest, "runtime_evidence": args.runtime_evidence_name,
                "formal_admission_evidence": args.formal_admission_evidence_name, "data_volume": str(volume.resolve()),
                "resource_authorization_evidence": args.resource_authorization_evidence_name,
                "min_free_bytes": args.min_free_bytes, "run_budget_seconds": args.run_budget_seconds,
                "deadline_utc_epoch": args.deadline_utc_epoch, **execution_inputs,
                "resource_attestation": str(output.resolve()), "resource_attestation_sha256": attestation_sha256}
    evidence = {
        args.runtime_evidence_name: (str(Path(args.runtime_config).resolve()), args.runtime_config_sha256),
        args.formal_admission_evidence_name: (str(Path(args.formal_admission).resolve()), args.formal_admission_sha256),
        args.accepted_evidence_name: (str(qualification.resolve()), args.qualification_report_sha256),
        args.resource_authorization_evidence_name: (str(authorization.resolve()), args.resource_authorization_sha256),
    }
    if len(evidence) != 4:
        parser.error("runtime, admission, qualification and authorization evidence names must differ")
    verify_attestation(queue_payload(""), args.job_key, evidence, candidate_document=document)
    if output.is_file():
        try:
            prior=json.loads(output.read_text())
        except (OSError, ValueError) as error:
            parser.error(f"existing attestation is unreadable: {error}")
        requested={key:value for key,value in document.items() if key != "issued_at_utc_epoch"}
        completed={key:value for key,value in prior.items() if key != "issued_at_utc_epoch"}
        if completed != requested:
            parser.error("existing attestation conflicts with requested bindings")
        # Re-enter the locked publisher so a link-before-directory-sync crash
        # is recovered rather than merely observed as a completed file.
        published=publish_attestation(prior, output)
        print(json.dumps({"status": "ATTESTED", "output": str(output.resolve()), "sha256": published, "queue_payload": queue_payload(published), "reused": True}, sort_keys=True))
        return
    published = publish_attestation(document, output)
    print(json.dumps({"status": "ATTESTED", "output": str(output.resolve()), "sha256": published, "queue_payload": queue_payload(published)}, sort_keys=True))


if __name__ == "__main__":
    main()
