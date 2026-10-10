#!/usr/bin/env python3
"""Refresh only the accepted local source binding for the R0 CPU preflight.

One-shot publication: after an interrupted multi-file write, preserve that
directory as evidence and retry with a new output directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nc_rted.prediction_inputs import IMPLEMENTATION_SCHEMA, _IMPLEMENTATION_FILES, canonical_json, sha256_file
from nc_rted.storage_lock import allocation_lock, ensure_directory


PREFLIGHT_SCHEMA = "nc_rted_r0_blind_manifest_build/v2"
ACCEPTANCE_SCHEMA = "nc_rted_prediction_source_acceptance/v1"
RESERVE = 20 * 1024**3


class RefreshError(ValueError):
    pass


def _sha(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _commit(value: object) -> bool:
    return isinstance(value, str) and len(value) in {40, 64} and all(char in "0123456789abcdef" for char in value)


def _read_bound(path: str | Path, digest: str, *, name: str) -> tuple[Path, dict[str, Any]]:
    candidate = Path(path)
    if not candidate.is_absolute() or candidate.is_symlink() or not candidate.is_file() or not _sha(digest) or sha256_file(candidate) != digest:
        raise RefreshError(f"{name} differs from its accepted SHA-256")
    try:
        value = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise RefreshError(f"{name} is not valid JSON") from error
    if not isinstance(value, dict):
        raise RefreshError(f"{name} must be a JSON object")
    return candidate, value


def _reference(value: object, *, name: str) -> tuple[str, str]:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"} or not isinstance(value.get("path"), str) or not _sha(value.get("sha256")):
        raise RefreshError(f"{name} reference differs")
    _read_bound(value["path"], value["sha256"], name=name)
    return value["path"], value["sha256"]


def _verify_release(root: Path, commit: str, acceptance_path: str, acceptance_sha256: str) -> None:
    if root.resolve() != ROOT.resolve() or not _commit(commit):
        raise RefreshError("accepted source root or commit differs from this refresher")
    try:
        actual = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
        subprocess.run(["git", "-C", str(root), "diff", "--quiet", "HEAD"], check=True)
    except subprocess.SubprocessError as error:
        raise RefreshError("accepted source checkout is dirty or unavailable") from error
    if actual != commit:
        raise RefreshError("accepted source checkout commit differs")
    _, acceptance = _read_bound(acceptance_path, acceptance_sha256, name="operator source acceptance")
    required = {"schema", "status", "release_root", "release_commit", "accepted_components"}
    if (set(acceptance) != required or acceptance.get("schema") != ACCEPTANCE_SCHEMA or acceptance.get("status") != "PASS" or
            acceptance.get("release_root") != str(root.resolve()) or acceptance.get("release_commit") != commit or
            not isinstance(acceptance.get("accepted_components"), dict) or set(acceptance["accepted_components"]) != {"code01a", "code01b"} or
            not all(_sha(value) for value in acceptance["accepted_components"].values())):
        raise RefreshError("operator source acceptance does not cover CODE-01A and CODE-01B")


def _old_preflight(path: str, digest: str) -> dict[str, tuple[str, str]]:
    _, report = _read_bound(path, digest, name="prior R0 preflight")
    required = {"schema", "status", "gpu_launched", "runtime", "identity_manifest", "r0_model_manifest", "bindings", "formal_admission", "denominators", "missing_dependencies"}
    if (set(report) != required or report.get("schema") != PREFLIGHT_SCHEMA or report.get("status") != "PREFLIGHT_PASS_FORMAL_ADMISSION_REQUIRED" or report.get("gpu_launched") is not False or
            report.get("denominators") != {"ucf": 251, "xd": 800, "vau": 3339} or report.get("missing_dependencies") != ["formal admission for this exact runtime/identity/binding set"]):
        raise RefreshError("prior R0 preflight schema or status differs")
    bindings = report.get("bindings")
    if not isinstance(bindings, dict) or set(bindings) != {"prediction_source", "implementation", "decoder", "embedded_vision"}:
        raise RefreshError("prior R0 preflight bindings differ")
    result = {name: _reference(report[name], name=name) for name in ("runtime", "identity_manifest", "r0_model_manifest")}
    result |= {name: _reference(bindings[name], name=name) for name in ("prediction_source", "decoder", "embedded_vision")}
    formal_path, formal_hash = _reference(report["formal_admission"], name="prior formal admission")
    _, formal = _read_bound(formal_path, formal_hash, name="prior formal admission")
    if formal != {"status": "BLOCKED", "formal_execution_allowed": False, "requested_output_root": formal.get("requested_output_root"), "reason": "formal admission has not accepted this exact runtime, identity, and binding set"} or not isinstance(formal["requested_output_root"], str):
        raise RefreshError("prior formal admission is not the required blocked placeholder")
    return result


def _publish(path: Path, document: dict[str, Any]) -> str:
    if not path.is_absolute() or os.path.lexists(path):
        raise RefreshError("refresh output must be a new absolute path")
    payload = canonical_json(document) + b"\n"
    try:
        with allocation_lock(path.parent):
            if os.path.lexists(path):
                raise RefreshError("refresh output already exists")
            ensure_directory(path.parent, RESERVE)
            block = max(4096, os.statvfs(path.parent).f_frsize)
            required = ((len(payload) + block - 1) // block + 3) * block
            if shutil.disk_usage(path.parent).free < RESERVE + required:
                raise RefreshError("refresh publication would violate storage reserve")
            descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".pending", dir=path.parent)
            try:
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(payload); stream.flush(); os.fsync(stream.fileno())
                os.link(temporary, path)
                parent = os.open(path.parent, os.O_DIRECTORY)
                try: os.fsync(parent)
                finally: os.close(parent)
            finally:
                if os.path.exists(temporary): os.unlink(temporary)
    except OSError as error:
        raise RefreshError("refresh publication failed") from error
    return hashlib.sha256(payload).hexdigest()


def run(args: argparse.Namespace) -> dict[str, Any]:
    release_root = Path(args.accepted_release_root)
    _verify_release(release_root, args.accepted_release_commit, args.source_acceptance, args.source_acceptance_sha256)
    old = _old_preflight(args.old_preflight, args.old_preflight_sha256)
    output = Path(args.output)
    if not args.output_root or not Path(args.output_root).is_absolute():
        raise RefreshError("requested prediction output root must be absolute")
    candidates = {relative: release_root / relative for relative in _IMPLEMENTATION_FILES}
    if any(not candidate.is_file() or candidate.is_symlink() for candidate in candidates.values()):
        raise RefreshError("accepted release implementation file set is unavailable")
    files = {relative: sha256_file(candidate) for relative, candidate in candidates.items()}
    implementation_path = output.parent / "implementation_manifest.json"
    implementation_hash = _publish(implementation_path, {"schema": IMPLEMENTATION_SCHEMA, "root": str(release_root.resolve()), "files": files})
    source_binding_path = output.parent / "source_refresh_binding.json"
    source_binding_hash = _publish(source_binding_path, {"schema": "nc_rted_r0_source_refresh_binding/v1", "release_commit": args.accepted_release_commit,
                                                           "prior_preflight_sha256": args.old_preflight_sha256, "implementation_manifest_sha256": implementation_hash,
                                                           "reused": old})
    formal_path = output.parent / "formal_admission_required.json"
    formal_hash = _publish(formal_path, {"status": "BLOCKED", "formal_execution_allowed": False, "requested_output_root": args.output_root,
                                         "reason": "formal admission has not accepted this exact runtime, identity, and binding set"})
    report = {"schema": PREFLIGHT_SCHEMA, "status": "PREFLIGHT_PASS_FORMAL_ADMISSION_REQUIRED", "gpu_launched": False,
              "runtime": dict(path=old["runtime"][0], sha256=old["runtime"][1]), "identity_manifest": dict(path=old["identity_manifest"][0], sha256=old["identity_manifest"][1]),
              "r0_model_manifest": dict(path=old["r0_model_manifest"][0], sha256=old["r0_model_manifest"][1]),
              "bindings": {"prediction_source": dict(path=old["prediction_source"][0], sha256=old["prediction_source"][1]),
                           "implementation": {"path": str(implementation_path), "sha256": implementation_hash},
                           "decoder": dict(path=old["decoder"][0], sha256=old["decoder"][1]), "embedded_vision": dict(path=old["embedded_vision"][0], sha256=old["embedded_vision"][1])},
              "formal_admission": {"path": str(formal_path), "sha256": formal_hash}, "denominators": {"ucf": 251, "xd": 800, "vau": 3339},
              "missing_dependencies": ["formal admission for this exact runtime/identity/binding set"],
              "source_refresh_binding": {"path": str(source_binding_path), "sha256": source_binding_hash}}
    # The historical v2 consumer deliberately has an exact field schema. Keep
    # its report compatible; the auxiliary binding stays separately published.
    report.pop("source_refresh_binding")
    report_hash = _publish(output, report)
    return {"status": report["status"], "output": str(output), "report_sha256": report_hash, "gpu_launched": False, "formal_execution_allowed": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accepted-release-root", required=True)
    parser.add_argument("--accepted-release-commit", required=True)
    parser.add_argument("--source-acceptance", required=True)
    parser.add_argument("--source-acceptance-sha256", required=True)
    parser.add_argument("--old-preflight", required=True)
    parser.add_argument("--old-preflight-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(run(args), sort_keys=True))
    except (RefreshError, OSError, ValueError) as error:
        print(json.dumps({"status": "BLOCKED", "error": str(error), "gpu_launched": False, "formal_execution_allowed": False}, sort_keys=True), file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
