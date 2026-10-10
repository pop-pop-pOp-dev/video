"""Immutable input checks shared by the formal bundle queue and runner.

The checks here deliberately allocate no model.  They bind the four admitted
members before the queue snapshots their executable closure.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from .production_runtime import (_checkpoint_identity, _validate_formal_admission_before_models,
                                 load_formal_admission, load_manifest, preflight)
from .task_inputs import TrainingCatalog

GROUPS = ("A", "U", "S", "F")
SCHEMA = "nc_rted_interleaved_formal_bundle/v1"


class FormalBundleContractError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_bundle(path_value: object, expected_sha256: object) -> tuple[Path, dict]:
    if not isinstance(path_value, str) or not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise FormalBundleContractError("formal bundle needs an absolute path and SHA-256")
    path = Path(path_value)
    if not path.is_absolute() or not path.is_file() or sha256_file(path) != expected_sha256:
        raise FormalBundleContractError("formal bundle is absent or differs from its SHA-256")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise FormalBundleContractError("formal bundle is invalid JSON") from error
    required = {"schema", "bundle_checkpoint_root", "members", "source_manifest", "source_manifest_sha256", "captured_source_map"}
    if not isinstance(document, dict) or set(document) != required or document.get("schema") != SCHEMA:
        raise FormalBundleContractError("formal bundle schema differs")
    if not isinstance(document.get("members"), dict) or set(document["members"]) != set(GROUPS):
        raise FormalBundleContractError("formal bundle needs exactly A/U/S/F members")
    return path, document


def _normalised_runtime(document: dict) -> dict:
    result = copy.deepcopy(document)
    try:
        for field in ("run_id", "group", "checkpoint_root", "progress_path"):
            result["run"].pop(field)
    except (KeyError, AttributeError) as error:
        raise FormalBundleContractError("formal member runtime identity is incomplete") from error
    return result


def validate_members(bundle: dict) -> dict[str, dict]:
    """Check admissions and identities without assembling a runtime."""
    members: dict[str, dict] = {}
    reference = None
    for group in GROUPS:
        item = bundle["members"][group]
        required = {"runtime", "runtime_sha256", "admission", "admission_sha256"}
        if not isinstance(item, dict) or set(item) != required:
            raise FormalBundleContractError("formal bundle member fields differ")
        try:
            runtime = load_manifest(item["runtime"], expected_sha256=item["runtime_sha256"])
            admission = load_formal_admission(item["admission"], expected_sha256=item["admission_sha256"])
            preflight(runtime)
            catalog_doc = runtime.document["catalog"]
            catalog = TrainingCatalog.load(catalog_doc["manifest_directory"], catalog_doc["training_annotations"],
                                           expected_provenance_sha256=catalog_doc["provenance_sha256"])
            identity = _checkpoint_identity(runtime, catalog)
            _validate_formal_admission_before_models(admission, identity)
        except Exception as error:
            raise FormalBundleContractError(f"formal member {group} is not independently admitted: {error}") from error
        if runtime.run.get("mode") != "formal" or runtime.run.get("group") != group:
            raise FormalBundleContractError("formal bundle member group or mode differs")
        normalised = _normalised_runtime(runtime.document)
        if reference is None:
            reference = normalised
        elif normalised != reference:
            raise FormalBundleContractError("formal members differ outside storage labels")
        members[group] = {"runtime": runtime, "admission": admission, "identity": identity}
    if len({item["identity"]["code_sha256"] for item in members.values()}) != 1:
        raise FormalBundleContractError("formal members have different source identities")
    return members
