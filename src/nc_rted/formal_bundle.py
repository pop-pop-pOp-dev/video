"""Immutable input checks shared by the formal bundle queue and runner.

The checks here deliberately allocate no model.  They bind the four admitted
members before the queue snapshots their executable closure.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Mapping

from .production_runtime import (_checkpoint_identity, _validate_formal_admission_before_models,
                                 load_formal_admission, load_manifest, preflight)
from .task_inputs import TrainingCatalog

GROUPS = ("A", "U", "S", "F")
SCHEMA = "nc_rted_interleaved_formal_bundle/v1"
FROZEN_CONSUMER_SCHEMA = "nc_rted_frozen_sample_consumer/v1"
FROZEN_INPUT_SOURCE_BOOKKEEPING = {"runtime_source", "builder", "caption_cache_producer"}


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
    allowed = required | {"frozen_sample_consumer"}
    if not isinstance(document, dict) or set(document) not in (required, allowed) or document.get("schema") != SCHEMA:
        raise FormalBundleContractError("formal bundle schema differs")
    if not isinstance(document.get("members"), dict) or set(document["members"]) != set(GROUPS):
        raise FormalBundleContractError("formal bundle needs exactly A/U/S/F members")
    return path, document


def load_frozen_sample_consumer(bundle: dict) -> dict | None:
    """Read the immutable cache-consumer contract named by a new bundle."""
    binding = bundle.get("frozen_sample_consumer")
    if binding is None:
        return None
    if (not isinstance(binding, dict) or set(binding) != {"path", "sha256"} or
            not isinstance(binding["path"], str) or not Path(binding["path"]).is_absolute() or
            not isinstance(binding["sha256"], str) or len(binding["sha256"]) != 64):
        raise FormalBundleContractError("frozen sample consumer binding differs")
    path = Path(binding["path"])
    if not path.is_file() or sha256_file(path) != binding["sha256"]:
        raise FormalBundleContractError("frozen sample consumer configuration differs")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise FormalBundleContractError("frozen sample consumer configuration is invalid") from error
    required = {"schema", "root", "identity", "producer_implementation", "device", "wait_seconds", "poll_seconds", "prefetch_depth"}
    if (not isinstance(document, dict) or set(document) != required or document.get("schema") != FROZEN_CONSUMER_SCHEMA or
            not isinstance(document["root"], str) or not Path(document["root"]).is_absolute() or
            not isinstance(document["identity"], dict) or not document["identity"] or
            not isinstance(document["producer_implementation"], dict) or not document["producer_implementation"] or
            not isinstance(document["device"], str) or not document["device"] or
            not isinstance(document["wait_seconds"], (int, float)) or not 0 < document["wait_seconds"] <= 600 or
            not isinstance(document["poll_seconds"], (int, float)) or not 0 < document["poll_seconds"] <= 60 or
            type(document["prefetch_depth"]) is not int or not 1 <= document["prefetch_depth"] <= 8):
        raise FormalBundleContractError("frozen sample consumer contract differs")
    return document


def validate_frozen_sample_consumer(document: dict, members: Mapping[str, tuple[object, TrainingCatalog]]) -> None:
    """Tie immutable producer records to every admitted runtime member."""
    identity = document["identity"]
    required = {"schema", "catalog_sha256", "frozen_inputs", "implementation"}
    implementation = document["producer_implementation"]
    if (set(identity) != required or identity.get("schema") != "nc_rted_frozen_material_producer/v1" or
            identity.get("implementation") != implementation or
            not isinstance(identity.get("frozen_inputs"), dict) or
            any(not isinstance(name, str) or not isinstance(digest, str) or len(digest) != 64 or
                any(char not in "0123456789abcdef" for char in digest)
                for name, digest in implementation.items())):
        raise FormalBundleContractError("frozen producer identity differs")
    for runtime, catalog in members.values():
        expected_inputs = {key: value for key, value in runtime.document.items()
                           if key not in {"run", "hashes", "teacher"} | FROZEN_INPUT_SOURCE_BOOKKEEPING}
        producer_inputs = identity["frozen_inputs"]
        if (identity["catalog_sha256"] != catalog.identity or
                set(producer_inputs) != set(expected_inputs) | FROZEN_INPUT_SOURCE_BOOKKEEPING or
                {key: value for key, value in producer_inputs.items() if key not in FROZEN_INPUT_SOURCE_BOOKKEEPING} != expected_inputs):
            raise FormalBundleContractError("frozen producer does not bind the admitted runtime")


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
    consumer = load_frozen_sample_consumer(bundle)
    consumer_digest = bundle["frozen_sample_consumer"]["sha256"] if consumer is not None else None
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
        except Exception as error:
            raise FormalBundleContractError(f"formal member {group} is not independently admitted: {error}") from error
        if runtime.run.get("mode") != "formal" or runtime.run.get("group") != group:
            raise FormalBundleContractError("formal bundle member group or mode differs")
        normalised = _normalised_runtime(runtime.document)
        if reference is None:
            reference = normalised
        elif normalised != reference:
            raise FormalBundleContractError("formal members differ outside storage labels")
        members[group] = {"runtime": runtime, "catalog": catalog, "admission": admission}
    if consumer is not None:
        validate_frozen_sample_consumer(consumer, {group: (item["runtime"], item["catalog"])
                                                   for group, item in members.items()})
    for item in members.values():
        identity = _checkpoint_identity(item["runtime"], item["catalog"],
                                        frozen_sample_consumer_sha256=consumer_digest)
        try:
            _validate_formal_admission_before_models(item["admission"], identity)
        except Exception as error:
            raise FormalBundleContractError("formal member admission identity differs") from error
        item["identity"] = identity
    if len({item["identity"]["code_sha256"] for item in members.values()}) != 1:
        raise FormalBundleContractError("formal members have different source identities")
    return members
