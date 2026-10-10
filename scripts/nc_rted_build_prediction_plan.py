#!/usr/bin/env python3
"""Publish immutable NC-RTED v2 prediction-plan candidates and finalized plans.

Candidate publication freezes the entire thirteen-task registry without
pretending that the formal admission exists.  Finalization only binds an
already-issued PASS admission to that exact candidate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nc_rted.prediction_inputs import (CANDIDATE_SCHEMA_V2, SCHEMA_V2, PredictionInputError, _no_supervision,
                                       canonical_json, sha256_file, validate_v2_prediction_registration)
from nc_rted.storage_lock import allocation_lock, ensure_directory


RESERVE = 20 * 1024**3
BASE_FIELDS = {"schema", "run_id", "identity_manifest", "identity_manifest_sha256", "model_tasks", "protocol", "output_root", "denominators", "bindings"}
BINDING_NAMES = {"runtime", "fast_snapshot", "source_manifest", "tokenizer", "embedded_vision_binding", "decoder", "implementation_manifest", "resource_admission"}


class BuildError(ValueError):
    pass


def _read(path: Path, *, name: str) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise BuildError(f"{name} is not valid JSON") from error


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish(path: Path, document: object) -> str:
    """Use the shared admission lock, reserve, and non-overwriting publication."""
    if not path.is_absolute():
        raise BuildError("output must be an absolute path")
    payload = canonical_json(document) + b"\n"
    try:
        with allocation_lock(path.parent):
            if os.path.lexists(path):
                raise BuildError(f"output already exists: {path}")
            ensure_directory(path.parent, RESERVE)
            if shutil.disk_usage(path.parent).free < RESERVE + len(payload) + 8192:
                raise BuildError("output publication would violate the 20 GiB reserve")
            descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".pending", dir=path.parent)
            try:
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.link(temporary, path)
                _fsync_directory(path.parent)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
    except BuildError:
        raise
    except (OSError, PredictionInputError) as error:
        raise BuildError(f"immutable publication failed: {path}") from error
    return hashlib.sha256(payload).hexdigest()


def _registration(document: object) -> tuple[dict, str, str]:
    if not isinstance(document, dict) or set(document) != BASE_FIELDS or document.get("schema") != SCHEMA_V2:
        raise BuildError("registration fields differ from the v2 contract")
    try:
        validated = validate_v2_prediction_registration(document)
    except PredictionInputError as error:
        raise BuildError(str(error)) from error
    return document, validated["matrix_id"], validated["execution_binding_sha256"]


def candidate(args: argparse.Namespace) -> None:
    registration_path = Path(args.registration).absolute()
    registration = _read(registration_path, name="registration")
    registration, matrix_id, binding = _registration(registration)
    candidate_doc = {"schema": CANDIDATE_SCHEMA_V2, "matrix_id": matrix_id, "registration": registration,
                     "prediction_execution_binding_sha256": binding}
    digest = _publish(Path(args.output).absolute(), candidate_doc)
    print(json.dumps({"status": "CANDIDATE_PUBLISHED", "candidate": str(Path(args.output).absolute()),
                      "candidate_sha256": digest, "matrix_id": matrix_id, "prediction_execution_binding_sha256": binding}, sort_keys=True))


def finalize(args: argparse.Namespace) -> None:
    candidate_path = Path(args.candidate).absolute()
    try:
        candidate_digest = sha256_file(candidate_path)
        candidate_doc = _read(candidate_path, name="prediction plan candidate")
    except OSError as error:
        raise BuildError("prediction plan candidate is missing") from error
    _no_supervision(candidate_doc)
    if not isinstance(candidate_doc, dict) or set(candidate_doc) != {"schema", "matrix_id", "registration", "prediction_execution_binding_sha256"} or candidate_doc.get("schema") != CANDIDATE_SCHEMA_V2:
        raise BuildError("prediction plan candidate fields differ")
    registration, matrix_id, binding = _registration(candidate_doc["registration"])
    if candidate_doc.get("matrix_id") != matrix_id or candidate_doc.get("prediction_execution_binding_sha256") != binding:
        raise BuildError("prediction plan candidate binding differs")
    admission_path = Path(args.formal_admission).absolute()
    try:
        admission_digest = sha256_file(admission_path)
        admission = _read(admission_path, name="formal admission")
    except OSError as error:
        raise BuildError("formal admission is missing") from error
    _no_supervision(admission)
    bindings = registration["bindings"]
    if (not isinstance(admission, dict) or admission.get("status") != "PASS" or admission.get("formal_execution_allowed") is not True or
            admission.get("embedded_vision_binding_sha256") != bindings["embedded_vision_binding_sha256"] or
            admission.get("prediction_execution_binding_sha256") != binding or
            admission.get("prediction_plan_candidate_sha256") != candidate_digest or
            admission.get("resource_admission_sha256") != bindings["resource_admission_sha256"]):
        raise BuildError("formal admission does not bind the candidate/resource execution inputs")
    plan = {**registration, "matrix_id": matrix_id, "candidate": {"path": str(candidate_path), "sha256": candidate_digest},
            "admission": {"formal_admission": str(admission_path), "formal_admission_sha256": admission_digest}}
    digest = _publish(Path(args.output).absolute(), plan)
    print(json.dumps({"status": "PLAN_PUBLISHED", "manifest": str(Path(args.output).absolute()), "manifest_sha256": digest,
                      "matrix_id": matrix_id, "candidate_sha256": candidate_digest}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    create = sub.add_parser("candidate")
    create.add_argument("--registration", required=True)
    create.add_argument("--output", required=True)
    finish = sub.add_parser("finalize")
    finish.add_argument("--candidate", required=True)
    finish.add_argument("--formal-admission", required=True)
    finish.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        if args.action == "candidate":
            candidate(args)
        else:
            finalize(args)
    except (BuildError, PredictionInputError, OSError, ValueError) as error:
        print(json.dumps({"status": "BLOCKED", "error": str(error)}, sort_keys=True), file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
