"""Blind, hash-bound inputs for NC-RTED prediction.

This module deliberately has no dependency on the training catalog.  In
particular, it never accepts targets, references, labels, answers, or metric
paths.  A prediction plan can therefore be inspected before any model is
constructed without making official test supervision available to a worker.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import json
import time
from pathlib import Path
from typing import Any, Mapping


SCHEMA = "nc_rted_blind_prediction/v1"
SCHEMA_V2 = "nc_rted_blind_prediction/v2"
CANDIDATE_SCHEMA_V2 = "nc_rted_blind_prediction_candidate/v2"
RESOURCE_ADMISSION_SCHEMA_V1 = "nc_rted_prediction_resource_admission/v2"
RESOURCE_QUALIFICATION_SCHEMA_V1 = "nc_rted_prediction_resource_qualification/v1"
RESOURCE_ALLOCATION_SCHEMA_V1 = "nc_rted_prediction_resource_allocation/v1"
RESOURCE_MEASUREMENT_SCHEMA_V1 = "nc_rted_prediction_resource_measurement/v1"
RESOURCE_AUTHORIZATION_SCHEMA_V1 = "nc_rted_prediction_resource_authorization/v1"
RESOURCE_EVIDENCE_REGISTRY_SCHEMA_V1 = "nc_rted_prediction_accepted_evidence_registry/v1"
IMPLEMENTATION_SCHEMA = "nc_rted_blind_prediction_implementation/v1"
GROUPS = ("R0", "A", "U", "S", "F")
FORMAL_SEEDS = (17, 42, 2026)
OFFICIAL_DENOMINATORS = {"ucf": 251, "xd": 800, "vau": 3339}
_HEX = set("0123456789abcdef")
_FORBIDDEN = {"label", "labels", "answer", "answers", "metric", "metrics", "target", "targets",
              "ground_truth", "groundtruth", "reference", "references", "annotation", "annotations",
              "teacher", "teachers"}
_IMPLEMENTATION_FILES = frozenset({
    "scripts/nc_rted_predict.py",
    "scripts/nc_rted_prediction_runtime_probe.py",
    "src/nc_rted/__init__.py",
    "src/nc_rted/batches.py",
    "src/nc_rted/bridge.py",
    "src/nc_rted/detection_media.py",
    "src/nc_rted/detection_provider.py",
    "src/nc_rted/detector.py",
    "src/nc_rted/frozen_vision.py",
    "src/nc_rted/inherited_memory.py",
    "src/nc_rted/loading.py",
    "src/nc_rted/media_observer.py",
    "src/nc_rted/numerics.py",
    "src/nc_rted/observation.py",
    "src/nc_rted/observation_cache.py",
    "src/nc_rted/prediction_adapters.py",
    "src/nc_rted/prediction_inputs.py",
    "src/nc_rted/prediction_media.py",
    "src/nc_rted/prediction_runtime.py",
    "src/nc_rted/prediction_store.py",
    "src/nc_rted/prediction_worker.py",
    "src/nc_rted/production_runtime.py",
    "src/nc_rted/recovery.py",
    "src/nc_rted/task_inputs.py",
    "src/nc_rted/storage_lock.py",
    "src/nc_rted/model.py",
})


class PredictionInputError(ValueError):
    pass


def canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(ch in _HEX for ch in value)


def _no_supervision(value: object, *, path: str = "") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise PredictionInputError("prediction documents require string keys")
            normalized = key.lower().replace("-", "_").replace(" ", "_")
            if normalized in _FORBIDDEN:
                raise PredictionInputError(f"prediction input contains forbidden supervision key: {path}{key}")
            _no_supervision(child, path=f"{path}{key}.")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _no_supervision(child, path=f"{path}{index}.")


def _bound_path(path: object, digest: object, *, name: str, verified: dict[tuple[str, str], tuple[int, int, int]] | None = None) -> Path:
    if not isinstance(path, str) or not _sha(digest):
        raise PredictionInputError(f"{name} requires an absolute path and SHA-256")
    result = Path(path)
    if not result.is_absolute() or not result.is_file():
        raise PredictionInputError(f"{name} is missing or differs from its SHA-256")
    stamp = _stamp(result)
    key = (str(result), str(digest))
    if verified is None or verified.get(key) != stamp:
        if sha256_file(result) != digest:
            raise PredictionInputError(f"{name} is missing or differs from its SHA-256")
        if verified is not None:
            verified[key] = _stamp(result)
    return result


def _stamp(path: Path) -> tuple[int, int, int]:
    info = path.stat()
    return info.st_ino, info.st_size, info.st_mtime_ns


class MediaVerifier:
    """Hash each immutable medium once, then recheck on an inode/size/mtime change."""
    def __init__(self):
        self._verified: dict[tuple[str, str], tuple[int, int, int]] = {}

    def verify(self, path: str, digest: str) -> None:
        _bound_path(path, digest, name="prediction media", verified=self._verified)


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    if not root.is_dir() or root.is_symlink():
        raise PredictionInputError("model export must be a real directory")
    for child in sorted(root.rglob("*")):
        if child.is_symlink():
            raise PredictionInputError("model export cannot contain symlinks")
        if child.is_file():
            digest.update(str(child.relative_to(root)).encode("utf-8"))
            digest.update(b"\0")
            digest.update(sha256_file(child).encode("ascii"))
            digest.update(b"\n")
    return digest.hexdigest()


def _bound_artifact(path: object, digest: object, *, name: str) -> str:
    if not isinstance(path, str) or not _sha(digest):
        raise PredictionInputError(f"{name} requires an absolute path and SHA-256")
    result = Path(path)
    if not result.is_absolute() or not result.is_dir() or _tree_sha256(result) != digest:
        raise PredictionInputError(f"{name} is missing or differs from its SHA-256")
    return str(result)


def verify_implementation_manifest(path: str | Path, digest: str) -> Path:
    """Verify the exact local source set before inherited runtime imports."""
    manifest_path = _bound_path(str(Path(path).absolute()), digest, name="prediction implementation manifest")
    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise PredictionInputError("prediction implementation manifest is not valid JSON") from error
    _no_supervision(document)
    if not isinstance(document, dict) or set(document) != {"schema", "root", "files"} or document["schema"] != IMPLEMENTATION_SCHEMA:
        raise PredictionInputError("prediction implementation manifest schema differs")
    root = Path(document["root"])
    files = document["files"]
    executing_root = Path(__file__).resolve().parents[2]
    if (not root.is_absolute() or not root.is_dir() or root.is_symlink() or root.resolve() != executing_root or
            not isinstance(files, dict) or set(files) != _IMPLEMENTATION_FILES):
        raise PredictionInputError("prediction implementation manifest file set differs")
    resolved_root = root.resolve()
    for relative, file_digest in files.items():
        if not isinstance(relative, str) or not _sha(file_digest):
            raise PredictionInputError("prediction implementation manifest entry is invalid")
        candidate = (root / relative)
        try:
            candidate.resolve().relative_to(resolved_root)
        except ValueError as error:
            raise PredictionInputError("prediction implementation source escapes its root") from error
        if candidate.is_symlink():
            raise PredictionInputError("prediction implementation source cannot be a symlink")
        _bound_path(str(candidate), file_digest, name=f"prediction implementation source {relative}")
    implementation_modules = {}
    for relative in files:
        if not relative.startswith("src/nc_rted/"):
            continue
        suffix = relative.removeprefix("src/").removesuffix(".py").replace("/", ".")
        name = suffix.removesuffix(".__init__")
        implementation_modules[name] = relative
    for name, module in tuple(__import__("sys").modules.items()):
        if name not in implementation_modules:
            continue
        origin = getattr(module, "__file__", None)
        if not origin:
            raise PredictionInputError("loaded prediction module has no admitted source origin")
        candidate = Path(origin).resolve()
        try:
            relative = str(candidate.relative_to(resolved_root))
        except ValueError as error:
            raise PredictionInputError("loaded prediction module origin escapes admitted checkout") from error
        if relative != implementation_modules[name] or sha256_file(candidate) != files[relative]:
            raise PredictionInputError(f"loaded prediction module differs from admitted source: {relative}")
    return executing_root


@dataclass(frozen=True)
class VadRequest:
    dataset: str
    media_id: str
    media_path: str
    media_sha256: str

    @property
    def identity(self) -> str:
        return f"vad:{self.dataset}:{self.media_id}"


@dataclass(frozen=True)
class VauRequest:
    instruction_id: str
    ordinal: int
    media_path: str
    media_sha256: str
    question: str

    @property
    def identity(self) -> str:
        return f"vau:{self.instruction_id}"


@dataclass(frozen=True)
class ModelArtifact:
    manifest_sha256: str
    group: str
    seed: int | None
    checkpoint: str | None
    checkpoint_manifest_sha256: str | None
    checkpoint_state_sha256: str | None
    final_checkpoint_attestation_sha256: str | None
    accepted_training_provenance_sha256: str | None
    training_identity: Mapping[str, str] | None

    @property
    def evidence_enabled(self) -> bool:
        return self.group != "R0"

    @property
    def task_id(self) -> str:
        return self.group if self.seed is None else f"{self.group}:seed{self.seed}"


@dataclass(frozen=True)
class ModelTask:
    group: str
    seed: int | None
    state: str = "READY"
    model_manifest: str | None = None
    model_manifest_sha256: str | None = None
    pending_reason: str | None = None

    @property
    def task_id(self) -> str:
        return self.group if self.seed is None else f"{self.group}:seed{self.seed}"


@dataclass(frozen=True)
class PredictionPlan:
    schema: str
    run_id: str
    manifest_sha256: str
    vad: tuple[VadRequest, ...]
    vau: tuple[VauRequest, ...]
    models: tuple[ModelTask, ...]
    bindings: Mapping[str, str]
    binding_sha256: Mapping[str, str]
    protocol: Mapping[str, Any]
    output_root: Path
    matrix_id: str | None = None
    execution_scope_sha256: str | None = None
    resource_scope: Mapping[str, Any] | None = None

    def requests(self) -> tuple[VadRequest | VauRequest, ...]:
        return self.vad + self.vau

    def selected_model(self, group: str, seed: int | None) -> ModelTask:
        for artifact in self.models:
            if artifact.group == group and artifact.seed == seed:
                if artifact.state != "READY":
                    raise PredictionInputError(f"prediction task {artifact.task_id} remains PENDING_DEPENDENCY")
                return artifact
        raise PredictionInputError(f"unknown prediction model task {group}:{seed}")


def prediction_execution_binding_sha256(*, bindings: Mapping[str, str], protocol: Mapping[str, Any],
                                        identity_manifest_sha256: str, identities: Mapping[str, Any],
                                        model_registry: object | None = None) -> str:
    """Digest every admitted execution input, including ordered blind identities."""
    value = {
        "bindings": dict(bindings),
        "identity_manifest_sha256": identity_manifest_sha256,
        "identity_mapping": identities,
        "implementation_manifest_sha256": bindings["implementation_manifest_sha256"],
        "protocol": dict(protocol),
    }
    if model_registry is not None:
        value["model_registry"] = model_registry
    return hashlib.sha256(canonical_json(value)).hexdigest()


def prediction_matrix_id(*, run_id: str, bindings: Mapping[str, str], protocol: Mapping[str, Any],
                         identity_manifest_sha256: str, denominators: Mapping[str, int]) -> str:
    """Stable matrix identity shared by immutable per-model plan revisions."""
    tasks = [{"group": "R0", "seed": None}]
    tasks.extend({"group": group, "seed": seed} for group in GROUPS if group != "R0" for seed in FORMAL_SEEDS)
    value = {"run_id": run_id, "bindings": dict(bindings), "protocol": dict(protocol),
             "identity_manifest_sha256": identity_manifest_sha256, "denominators": dict(denominators), "tasks": tasks}
    return hashlib.sha256(canonical_json(value)).hexdigest()


def prediction_execution_scope_sha256(*, run_id: str, bindings: Mapping[str, str], protocol: Mapping[str, Any],
                                      identity_manifest_sha256: str, denominators: Mapping[str, int], output_root: str) -> str:
    """Scope resource admission without depending on its own evidence hash."""
    scoped = {key: value for key, value in bindings.items() if not key.startswith("resource_admission")}
    value = {"kind": "blind_prediction", "run_id": run_id, "bindings": scoped, "protocol": dict(protocol),
             "identity_manifest_sha256": identity_manifest_sha256, "denominators": dict(denominators), "output_root": output_root}
    return hashlib.sha256(canonical_json(value)).hexdigest()


def prediction_task_execution_binding_sha256(*, matrix_id: str, task: ModelArtifact) -> str:
    if not _sha(matrix_id):
        raise PredictionInputError("v2 task execution requires a matrix identity")
    value = {"matrix_id": matrix_id, "task": task.task_id, "model_manifest_sha256": task.manifest_sha256,
             "checkpoint_manifest_sha256": task.checkpoint_manifest_sha256, "checkpoint_state_sha256": task.checkpoint_state_sha256}
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _parse_vad(rows: object, counts: Mapping[str, int], verified: dict[tuple[str, str], tuple[int, int, int]]) -> tuple[VadRequest, ...]:
    if not isinstance(rows, list):
        raise PredictionInputError("identity_manifest.vad must be a list")
    output = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"dataset", "id", "media_path", "media_sha256"}:
            raise PredictionInputError("each VAD identity has exactly dataset/id/media_path/media_sha256")
        dataset, identity = row["dataset"], row["id"]
        if dataset not in {"ucf", "xd"} or not isinstance(identity, str) or not identity:
            raise PredictionInputError("invalid VAD dataset or identity")
        media = _bound_path(row["media_path"], row["media_sha256"], name="VAD media", verified=verified)
        output.append(VadRequest(dataset, identity, str(media), row["media_sha256"]))
    if len({item.identity for item in output}) != len(output):
        raise PredictionInputError("duplicate VAD identity")
    observed = {name: sum(item.dataset == name for item in output) for name in ("ucf", "xd")}
    if observed != {"ucf": counts["ucf"], "xd": counts["xd"]}:
        raise PredictionInputError(f"VAD denominator differs: {observed}")
    return tuple(sorted(output, key=lambda item: (item.dataset, item.media_id)))


def _parse_vau(rows: object, count: int, verified: dict[tuple[str, str], tuple[int, int, int]]) -> tuple[VauRequest, ...]:
    if not isinstance(rows, list) or len(rows) != count:
        raise PredictionInputError(f"VAU denominator differs: expected {count}")
    output = []
    for ordinal, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != {"id", "media_path", "media_sha256", "question"}:
            raise PredictionInputError("each VAU identity has exactly id/media_path/media_sha256/question")
        identity, question = row["id"], row["question"]
        if not isinstance(identity, str) or not identity or not isinstance(question, str) or not question:
            raise PredictionInputError("invalid VAU identity or question")
        media = _bound_path(row["media_path"], row["media_sha256"], name="VAU media", verified=verified)
        output.append(VauRequest(identity, ordinal, str(media), row["media_sha256"], question))
    if len({item.instruction_id for item in output}) != len(output):
        raise PredictionInputError("duplicate VAU instruction identity")
    return tuple(output)


def _expected_tasks(models: list[ModelTask]) -> tuple[ModelTask, ...]:
    expected = {("R0", None)} | {(group, seed) for group in GROUPS if group != "R0" for seed in FORMAL_SEEDS}
    if {(item.group, item.seed) for item in models} != expected:
        raise PredictionInputError("model tasks must be R0 plus A/U/S/F for seeds 17, 42, and 2026")
    return tuple(sorted(models, key=lambda item: (GROUPS.index(item.group), -1 if item.seed is None else item.seed)))


def _parse_models(rows: object) -> tuple[ModelTask, ...]:
    if not isinstance(rows, list) or len(rows) != 13:
        raise PredictionInputError("exactly 13 R0/A/U/S/F model tasks are required")
    models = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"group", "seed"}:
            raise PredictionInputError("model task fields differ from the blind prediction contract")
        group = row["group"]
        seed = row["seed"]
        if group not in GROUPS or (group == "R0" and seed is not None) or (group != "R0" and seed not in FORMAL_SEEDS):
            raise PredictionInputError("invalid model artifact")
        models.append(ModelTask(group, seed))
    return _expected_tasks(models)


def _parse_models_v2(rows: object) -> tuple[ModelTask, ...]:
    """Parse an immutable full registry while permitting untrained tasks to wait."""
    if not isinstance(rows, list) or len(rows) != 13:
        raise PredictionInputError("exactly 13 R0/A/U/S/F model tasks are required")
    models = []
    required = {"group", "seed", "state", "model_manifest", "model_manifest_sha256", "pending_reason"}
    for row in rows:
        if not isinstance(row, dict) or set(row) != required:
            raise PredictionInputError("v2 model registry fields differ from the blind prediction contract")
        group, seed, state = row["group"], row["seed"], row["state"]
        if group not in GROUPS or (group == "R0" and seed is not None) or (group != "R0" and seed not in FORMAL_SEEDS):
            raise PredictionInputError("invalid model registry task")
        manifest, digest, reason = row["model_manifest"], row["model_manifest_sha256"], row["pending_reason"]
        if state == "READY":
            if not isinstance(manifest, str) or not Path(manifest).is_absolute() or not _sha(digest) or reason is not None:
                raise PredictionInputError("READY registry task requires only a hash-bound model manifest")
        elif state == "PENDING_DEPENDENCY":
            if manifest is not None or digest is not None or reason != "FINAL_CHECKPOINT_ATTESTATION_AND_PROVENANCE_PENDING":
                raise PredictionInputError("PENDING_DEPENDENCY registry task cannot carry a model artifact")
        else:
            raise PredictionInputError("model registry state is invalid")
        models.append(ModelTask(group, seed, state, manifest, digest, reason))
    return _expected_tasks(models)


def load_model_artifact(path: str | Path, *, expected_sha256: str, task: ModelTask) -> ModelArtifact:
    if task.state != "READY":
        raise PredictionInputError(f"prediction task {task.task_id} remains PENDING_DEPENDENCY")
    resolved = str(Path(path).absolute())
    if task.model_manifest is not None and (resolved != task.model_manifest or expected_sha256 != task.model_manifest_sha256):
        raise PredictionInputError("selected model manifest differs from its v2 registry binding")
    artifact_path = _bound_path(str(Path(path).absolute()), expected_sha256, name="selected model manifest")
    try:
        row = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise PredictionInputError("selected model manifest is not valid JSON") from error
    _no_supervision(row)
    expected = {"group", "seed", "checkpoint", "checkpoint_manifest_sha256", "checkpoint_state_sha256",
                "final_checkpoint_attestation", "final_checkpoint_attestation_sha256", "accepted_training_provenance",
                "accepted_training_provenance_sha256"}
    if not isinstance(row, dict) or set(row) != expected or row["group"] != task.group or row["seed"] != task.seed:
        raise PredictionInputError("selected model manifest does not match the registered task")
    if task.group == "R0":
        if any(row[name] is not None for name in ("checkpoint", "checkpoint_manifest_sha256", "checkpoint_state_sha256",
                                                  "final_checkpoint_attestation", "final_checkpoint_attestation_sha256",
                                                  "accepted_training_provenance", "accepted_training_provenance_sha256")):
            raise PredictionInputError("R0 must use only the original final Stage2 export")
        checkpoint = manifest_hash = state_hash = attestation_hash = provenance_hash = identity = None
    else:
        checkpoint = row["checkpoint"]
        manifest_hash, state_hash = row["checkpoint_manifest_sha256"], row["checkpoint_state_sha256"]
        attestation, attestation_hash = row["final_checkpoint_attestation"], row["final_checkpoint_attestation_sha256"]
        provenance, provenance_hash = row["accepted_training_provenance"], row["accepted_training_provenance_sha256"]
        if (not isinstance(checkpoint, str) or not isinstance(attestation, str) or not isinstance(provenance, str) or
                not _sha(manifest_hash) or not _sha(state_hash) or not _sha(attestation_hash) or not _sha(provenance_hash)):
            raise PredictionInputError("A/U/S/F require an exact completed checkpoint binding")
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.is_absolute() or not checkpoint_path.is_dir():
            raise PredictionInputError("checkpoint directory is absent")
        _bound_path(str(checkpoint_path / "manifest.json"), manifest_hash, name="checkpoint manifest")
        _bound_path(str(checkpoint_path / "state.pt"), state_hash, name="checkpoint state")
        attestation_path = _bound_path(attestation, attestation_hash, name="final checkpoint attestation")
        provenance_path = _bound_path(provenance, provenance_hash, name="accepted training provenance")
        try:
            attestation_document = json.loads(attestation_path.read_text(encoding="utf-8"))
            provenance_document = json.loads(provenance_path.read_text(encoding="utf-8"))
            checkpoint_document = json.loads((checkpoint_path / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise PredictionInputError("final checkpoint attestation or manifest is invalid") from error
        _no_supervision(checkpoint_document)
        required_identity = {"run_id", "group", "seed", "code_sha256", "config_sha256", "data_sha256",
                             "teacher_sha256", "inherited_weights_sha256", "runtime_sha256"}
        required_attestation = {"schema", "status", "final_checkpoint_allowed", "checkpoint_manifest_sha256",
                                "checkpoint_state_sha256", "completed_updates", "training_identity"}
        required_provenance = {"schema", "status", "training_allowed", "group", "seed", "checkpoint_manifest_sha256",
                               "checkpoint_state_sha256", "training_identity"}
        identity = attestation_document.get("training_identity") if isinstance(attestation_document, dict) else None
        if (not isinstance(attestation_document, dict) or set(attestation_document) != required_attestation or
                attestation_document.get("schema") != "nc_rted_final_checkpoint_attestation/v1" or
                attestation_document.get("status") != "PASS" or attestation_document.get("final_checkpoint_allowed") is not True or
                attestation_document.get("checkpoint_manifest_sha256") != manifest_hash or
                attestation_document.get("checkpoint_state_sha256") != state_hash or attestation_document.get("completed_updates") != 1000 or
                not isinstance(identity, dict) or set(identity) != required_identity or identity.get("group") != task.group or
                identity.get("seed") != str(task.seed) or not all(isinstance(value, str) and value for value in identity.values()) or
                not all(_sha(identity[name]) for name in required_identity if name.endswith("sha256")) or
                not isinstance(provenance_document, dict) or set(provenance_document) != required_provenance or
                provenance_document.get("schema") != "nc_rted_accepted_training_provenance/v1" or provenance_document.get("status") != "PASS" or
                provenance_document.get("training_allowed") is not True or provenance_document.get("group") != task.group or
                provenance_document.get("seed") != task.seed or provenance_document.get("checkpoint_manifest_sha256") != manifest_hash or
                provenance_document.get("checkpoint_state_sha256") != state_hash or provenance_document.get("training_identity") != identity or
                not isinstance(checkpoint_document, dict) or checkpoint_document.get("schema") != "nc_rted_checkpoint_v2" or
                checkpoint_document.get("final") is not True or checkpoint_document.get("completed_updates") != 1000 or
                checkpoint_document.get("identity") != identity or checkpoint_document.get("payload_sha256") != state_hash):
            raise PredictionInputError("selected checkpoint is not an admitted 1000-update final checkpoint")
        checkpoint = str(checkpoint_path)
    return ModelArtifact(expected_sha256, task.group, task.seed, checkpoint, manifest_hash, state_hash, attestation_hash, provenance_hash, identity)


def _validate_protocol(protocol: object) -> dict:
    required = {"hivau", "vad_route", "vad_causal_smoothing", "vad_fast_fusion"}
    hivau_keys = {"target_fps", "query_interval", "paligemma_batch_size", "max_new_tokens", "task", "fast_prompt_context"}
    hivau = protocol.get("hivau") if isinstance(protocol, dict) else None
    if (not isinstance(protocol, dict) or not required.issubset(protocol) or not isinstance(hivau, dict) or set(hivau) != hivau_keys or
            any(type(hivau[key]) is not int or hivau[key] < 1 for key in ("target_fps", "query_interval", "paligemma_batch_size", "max_new_tokens")) or
            not isinstance(hivau["task"], str) or not hivau["task"] or hivau["fast_prompt_context"] != "none"):
        raise PredictionInputError("VAU must bind the inherited HIVAU generation protocol without Fast prompt injection")
    vad = protocol.get("vad")
    if (protocol["vad_route"] != "reactvau_detection" or protocol["vad_causal_smoothing"] != "online" or
            not isinstance(protocol["vad_fast_fusion"], str) or not isinstance(vad, dict) or
            set(vad) != {"target_fps", "query_interval", "batch_size"} or
            any(type(vad[key]) is not int or vad[key] < 1 for key in vad)):
        raise PredictionInputError("VAD sampling protocol is incomplete")
    return protocol


def _validate_resource_admission(path: str, digest: str, *, scope_sha256: str) -> dict[str, Any]:
    resource_path = _bound_path(path, digest, name="resource admission")
    try:
        resource = json.loads(resource_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise PredictionInputError("resource admission is not valid JSON") from error
    _no_supervision(resource)
    required = {"schema", "status", "formal_execution_allowed", "scope", "accepted_allocation", "accepted_measurement"}
    scope_fields = {"kind", "run_id", "execution_scope_sha256", "device", "physical_gpu_uuid", "data_volume", "min_free_bytes", "run_budget_seconds", "deadline_utc_epoch"}
    if not isinstance(resource, dict) or set(resource) != required or resource.get("schema") != RESOURCE_ADMISSION_SCHEMA_V1 or resource.get("status") != "PASS" or resource.get("formal_execution_allowed") is not True:
        raise PredictionInputError("resource admission is not an accepted scoped formal execution record")
    scope = resource["scope"]
    if (not isinstance(scope, dict) or set(scope) != scope_fields or scope.get("kind") != "blind_prediction" or not isinstance(scope.get("run_id"), str) or not scope["run_id"] or
            scope.get("execution_scope_sha256") != scope_sha256 or not isinstance(scope.get("device"), str) or not scope["device"].startswith("cuda:") or
            not isinstance(scope.get("physical_gpu_uuid"), str) or not scope["physical_gpu_uuid"] or not isinstance(scope.get("data_volume"), str) or not Path(scope["data_volume"]).is_absolute() or
            type(scope.get("min_free_bytes")) is not int or scope["min_free_bytes"] < 20 * 1024**3 or type(scope.get("run_budget_seconds")) is not int or scope["run_budget_seconds"] < 1 or
            type(scope.get("deadline_utc_epoch")) not in {int, float} or isinstance(scope["deadline_utc_epoch"], bool) or not math.isfinite(scope["deadline_utc_epoch"]) or scope["deadline_utc_epoch"] <= 0):
        raise PredictionInputError("resource admission scope differs from blind prediction execution")
    allocation_binding = resource["accepted_allocation"]
    measurement_binding = resource["accepted_measurement"]
    if (not isinstance(allocation_binding, dict) or set(allocation_binding) != {"path", "sha256"} or
            not isinstance(measurement_binding, dict) or set(measurement_binding) != {"path", "sha256"}):
        raise PredictionInputError("resource admission accepted allocation/measurement bindings are incomplete")
    allocation_path = _bound_path(allocation_binding["path"], allocation_binding["sha256"], name="accepted resource allocation")
    measurement_path = _bound_path(measurement_binding["path"], measurement_binding["sha256"], name="accepted resource measurement")
    try:
        allocation = json.loads(allocation_path.read_text(encoding="utf-8"))
        measurement = json.loads(measurement_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise PredictionInputError("resource allocation/measurement evidence is not valid JSON") from error
    _no_supervision(allocation); _no_supervision(measurement)
    allocation_fields = {"schema", "status", "kind", "execution_scope_sha256", "allocation_id", "host", "device", "physical_gpu_uuid", "lease_id", "authorization_sha256", "limits"}
    measurement_fields = {"schema", "status", "kind", "execution_scope_sha256", "allocation_sha256", "host", "physical_gpu_uuid", "workload", "measured_at_utc_epoch", "qualified_runtime_seconds", "peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes"}
    limit_fields = {"data_volume", "min_free_bytes", "run_budget_seconds", "deadline_utc_epoch"}
    if (not isinstance(allocation, dict) or set(allocation) != allocation_fields or
            allocation.get("schema") != RESOURCE_ALLOCATION_SCHEMA_V1 or allocation.get("status") != "ACCEPTED" or allocation.get("kind") != "blind_prediction" or
            allocation.get("execution_scope_sha256") != scope_sha256 or
            not all(isinstance(allocation.get(key), str) and allocation[key] for key in ("allocation_id", "host", "device", "physical_gpu_uuid", "lease_id")) or not _sha(allocation.get("authorization_sha256")) or
            allocation["device"] != scope["device"] or allocation["physical_gpu_uuid"] != scope["physical_gpu_uuid"] or
            not isinstance(allocation.get("limits"), dict) or set(allocation["limits"]) != limit_fields or allocation["limits"] != {key: scope[key] for key in limit_fields}):
        raise PredictionInputError("accepted resource allocation differs from admitted scope")
    if (not isinstance(measurement, dict) or set(measurement) != measurement_fields or
            measurement.get("schema") != RESOURCE_MEASUREMENT_SCHEMA_V1 or measurement.get("status") != "MEASURED" or measurement.get("kind") != "blind_prediction" or
            measurement.get("execution_scope_sha256") != scope_sha256 or measurement.get("allocation_sha256") != allocation_binding["sha256"] or measurement.get("host") != allocation["host"] or measurement.get("physical_gpu_uuid") != allocation["physical_gpu_uuid"] or
            measurement.get("workload") != {"kind": "blind_prediction", "vad_queries": 135050, "slow_triggers": 47458, "vau_requests": 3339} or
            type(measurement.get("measured_at_utc_epoch")) not in {int, float} or not math.isfinite(measurement["measured_at_utc_epoch"]) or measurement["measured_at_utc_epoch"] <= 0 or measurement["measured_at_utc_epoch"] > time.time() or measurement["measured_at_utc_epoch"] > scope["deadline_utc_epoch"] or
            type(measurement.get("qualified_runtime_seconds")) not in {int, float} or not math.isfinite(measurement["qualified_runtime_seconds"]) or measurement["qualified_runtime_seconds"] <= 0 or measurement["qualified_runtime_seconds"] > scope["run_budget_seconds"] or
            any(type(measurement.get(key)) is not int or measurement[key] < 0 for key in ("peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes")) or measurement["peak_cuda_reserved_bytes"] < measurement["peak_cuda_allocated_bytes"]):
        raise PredictionInputError("accepted resource measurement differs from accepted allocation")
    return dict(scope)


def validate_resource_execution_evidence(scope: Mapping[str, Any], *, registry_path: str, registry_sha256: str,
                                         host: str, physical_gpu_uuid: str, capacity_bytes: int) -> None:
    """Check operator-accepted allocation evidence outside the candidate graph."""
    registry_file = _bound_path(registry_path, registry_sha256, name="operator accepted evidence registry")
    try:
        registry = json.loads(registry_file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise PredictionInputError("operator accepted evidence registry is not valid JSON") from error
    required = {"schema", "accepted"}
    names = {"resource_authorization", "resource_allocation", "resource_measurement"}
    if not isinstance(registry, dict) or set(registry) != required or registry.get("schema") != RESOURCE_EVIDENCE_REGISTRY_SCHEMA_V1 or not isinstance(registry.get("accepted"), dict) or set(registry["accepted"]) != names:
        raise PredictionInputError("operator accepted evidence registry differs")
    bound = {name: _bound_path(registry["accepted"][name].get("path") if isinstance(registry["accepted"][name], dict) else None,
                               registry["accepted"][name].get("sha256") if isinstance(registry["accepted"][name], dict) else None,
                               name=f"accepted {name}") for name in names}
    try:
        authorization = json.loads(bound["resource_authorization"].read_text(encoding="utf-8"))
        allocation = json.loads(bound["resource_allocation"].read_text(encoding="utf-8"))
        measurement = json.loads(bound["resource_measurement"].read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise PredictionInputError("accepted resource evidence is not valid JSON") from error
    limits = {key: scope[key] for key in ("data_volume", "min_free_bytes", "run_budget_seconds", "deadline_utc_epoch")}
    auth_fields = {"schema", "status", "host", "physical_gpu_uuid", "lease_id", "lease_expires_utc_epoch", "project_volume", "max_budget_seconds", "min_free_bytes", "deadline_utc_epoch", "capacity_bytes"}
    if (not isinstance(authorization, dict) or set(authorization) != auth_fields or authorization.get("schema") != RESOURCE_AUTHORIZATION_SCHEMA_V1 or authorization.get("status") != "PASS" or
            authorization.get("host") != host or authorization.get("physical_gpu_uuid") != physical_gpu_uuid or not isinstance(authorization.get("lease_id"), str) or not authorization["lease_id"] or
            type(authorization.get("lease_expires_utc_epoch")) not in {int, float} or authorization["lease_expires_utc_epoch"] <= time.time() or authorization["lease_expires_utc_epoch"] < scope["deadline_utc_epoch"] or
            authorization.get("project_volume") != scope["data_volume"] or authorization.get("max_budget_seconds") != scope["run_budget_seconds"] or authorization.get("min_free_bytes") != scope["min_free_bytes"] or authorization.get("deadline_utc_epoch") != scope["deadline_utc_epoch"] or
            type(authorization.get("capacity_bytes")) is not int or authorization["capacity_bytes"] != capacity_bytes):
        raise PredictionInputError("accepted resource authorization does not cover this execution")
    allocation_hash = registry["accepted"]["resource_allocation"]["sha256"]
    authorization_hash = registry["accepted"]["resource_authorization"]["sha256"]
    if (not isinstance(allocation, dict) or allocation.get("schema") != RESOURCE_ALLOCATION_SCHEMA_V1 or allocation.get("status") != "ACCEPTED" or
            allocation.get("execution_scope_sha256") != scope["execution_scope_sha256"] or allocation.get("host") != host or allocation.get("physical_gpu_uuid") != physical_gpu_uuid or
            allocation.get("lease_id") != authorization["lease_id"] or allocation.get("authorization_sha256") != authorization_hash or allocation.get("limits") != limits):
        raise PredictionInputError("accepted allocation does not cover this execution")
    if (not isinstance(measurement, dict) or measurement.get("schema") != RESOURCE_MEASUREMENT_SCHEMA_V1 or measurement.get("status") != "MEASURED" or
            measurement.get("allocation_sha256") != allocation_hash or measurement.get("host") != host or measurement.get("physical_gpu_uuid") != physical_gpu_uuid or
            measurement.get("workload") != {"kind": "blind_prediction", "vad_queries": 135050, "slow_triggers": 47458, "vau_requests": 3339} or
            type(measurement.get("peak_cuda_allocated_bytes")) is not int or measurement["peak_cuda_allocated_bytes"] <= 0 or type(measurement.get("peak_cuda_reserved_bytes")) is not int or
            not (measurement["peak_cuda_allocated_bytes"] <= measurement["peak_cuda_reserved_bytes"] <= capacity_bytes) or measurement.get("qualified_runtime_seconds", 0) > scope["run_budget_seconds"]):
        raise PredictionInputError("accepted measurement does not qualify this execution")


def validate_v2_prediction_registration(document: object) -> dict:
    """Validate candidate inputs identically before publication and execution."""
    fields = {"schema", "run_id", "identity_manifest", "identity_manifest_sha256", "model_tasks", "protocol", "output_root", "denominators", "bindings"}
    _no_supervision(document)
    if not isinstance(document, dict) or set(document) != fields or document.get("schema") != SCHEMA_V2 or not isinstance(document.get("run_id"), str) or not document["run_id"]:
        raise PredictionInputError("v2 prediction registration fields differ from the contract")
    denominators = document["denominators"]
    test_registration = document["run_id"].startswith("test:")
    if (not isinstance(denominators, dict) or set(denominators) != set(OFFICIAL_DENOMINATORS) or
            any(type(denominators[key]) is not int or denominators[key] < 1 for key in denominators) or
            (not test_registration and dict(denominators) != OFFICIAL_DENOMINATORS)):
        raise PredictionInputError("v2 prediction registration requires the 251/800/3339 official denominators")
    output_root = Path(document["output_root"])
    if not output_root.is_absolute():
        raise PredictionInputError("output_root must be absolute")
    protocol = _validate_protocol(document["protocol"])
    identity_path = _bound_path(document["identity_manifest"], document["identity_manifest_sha256"], name="identity manifest")
    try: identities = json.loads(identity_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error: raise PredictionInputError("identity manifest is not valid JSON") from error
    _no_supervision(identities)
    if not isinstance(identities, dict) or set(identities) != {"vad", "vau"}:
        raise PredictionInputError("identity manifest must contain only VAD and VAU identities")
    names = {"runtime", "fast_snapshot", "source_manifest", "tokenizer", "embedded_vision_binding", "decoder", "implementation_manifest", "resource_admission"}
    bindings = document["bindings"]
    if not isinstance(bindings, dict) or set(bindings) != names | {f"{name}_sha256" for name in names}:
        raise PredictionInputError("v2 runtime/Fast/source/tokenizer/vision/decoder/resource bindings are incomplete")
    bound = {}
    for name in names - {"resource_admission"}:
        validator = _bound_artifact if name == "tokenizer" else _bound_path
        bound[name] = str(validator(bindings[name], bindings[f"{name}_sha256"], name=name))
    bound_hashes = {name: str(bindings[f"{name}_sha256"]) for name in names}
    scope = prediction_execution_scope_sha256(run_id=document["run_id"], bindings=bindings, protocol=protocol, identity_manifest_sha256=document["identity_manifest_sha256"], denominators=denominators, output_root=str(output_root))
    resource_scope = None
    if test_registration:
        # Test-only fixture compatibility; formal registrations always require
        # the scoped v2 record above.
        resource = _bound_path(bindings["resource_admission"], bindings["resource_admission_sha256"], name="resource admission")
        if json.loads(resource.read_text(encoding="utf-8")).get("schema") not in {"nc_rted_prediction_resource_admission/v1", RESOURCE_ADMISSION_SCHEMA_V1}:
            raise PredictionInputError("test resource admission schema differs")
    else:
        resource_scope = _validate_resource_admission(bindings["resource_admission"], bindings["resource_admission_sha256"], scope_sha256=scope)
    models = _parse_models_v2(document["model_tasks"])
    for task in models:
        if task.state == "READY": load_model_artifact(task.model_manifest, expected_sha256=task.model_manifest_sha256, task=task)
    verified: dict[tuple[str, str], tuple[int, int, int]] = {}
    vad, vau = _parse_vad(identities["vad"], denominators, verified), _parse_vau(identities["vau"], denominators["vau"], verified)
    return {"identities": identities, "bindings": bound, "binding_sha256": bound_hashes, "models": models, "vad": vad, "vau": vau, "protocol": protocol, "output_root": output_root,
            "matrix_id": prediction_matrix_id(run_id=document["run_id"], bindings=bindings, protocol=protocol, identity_manifest_sha256=document["identity_manifest_sha256"], denominators=denominators),
            "execution_scope_sha256": scope, "resource_scope": resource_scope, "execution_binding_sha256": prediction_execution_binding_sha256(bindings=bindings, protocol=protocol, identity_manifest_sha256=document["identity_manifest_sha256"], identities=identities, model_registry=document["model_tasks"])}


def load_prediction_plan(path: str | Path, *, expected_sha256: str | None = None) -> PredictionPlan:
    manifest_path = Path(path).absolute()
    actual = sha256_file(manifest_path)
    if expected_sha256 is not None and actual != expected_sha256:
        raise PredictionInputError("prediction manifest SHA-256 differs")
    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise PredictionInputError("prediction manifest is not valid JSON") from error
    _no_supervision(document)
    if not isinstance(document, dict) or document.get("schema") not in {SCHEMA, SCHEMA_V2}:
        raise PredictionInputError("unsupported blind prediction manifest")
    v2 = document["schema"] == SCHEMA_V2
    base_fields = {"schema", "run_id", "identity_manifest", "identity_manifest_sha256", "model_tasks", "protocol", "output_root", "denominators", "bindings"}
    allowed = base_fields | {"admission"}
    if v2:
        allowed |= {"matrix_id", "candidate"}
    if set(document) != allowed or not isinstance(document.get("run_id"), str) or not document["run_id"]:
        raise PredictionInputError("prediction manifest fields differ from the contract")
    registration = validate_v2_prediction_registration({name: document[name] for name in base_fields}) if v2 else None
    identity_path = _bound_path(document["identity_manifest"], document["identity_manifest_sha256"], name="identity manifest")
    try:
        identities = json.loads(identity_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise PredictionInputError("identity manifest is not valid JSON") from error
    _no_supervision(identities)
    if not isinstance(identities, dict) or set(identities) != {"vad", "vau"}:
        raise PredictionInputError("identity manifest must contain only VAD and VAU identities")
    bindings = document["bindings"]
    binding_names = {"runtime", "fast_snapshot", "source_manifest", "tokenizer", "embedded_vision_binding", "decoder", "implementation_manifest"}
    if v2:
        binding_names.add("resource_admission")
    if not isinstance(bindings, dict) or set(bindings) != binding_names | {f"{name}_sha256" for name in binding_names}:
        raise PredictionInputError("runtime/Fast/source/tokenizer/vision/decoder/resource bindings are incomplete")
    bound = {}
    for name in binding_names:
        validator = _bound_artifact if name == "tokenizer" else _bound_path
        bound[name] = str(validator(bindings[name], bindings[f"{name}_sha256"], name=name))
    bound_hashes = {name: str(bindings[f"{name}_sha256"]) for name in binding_names}
    if v2 and registration is None:
        try:
            resource_document = json.loads(Path(bound["resource_admission"]).read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise PredictionInputError("resource admission is not valid JSON") from error
        _no_supervision(resource_document)
        if (not isinstance(resource_document, dict) or resource_document.get("schema") != RESOURCE_ADMISSION_SCHEMA_V1 or
                resource_document.get("status") != "PASS" or resource_document.get("formal_execution_allowed") is not True):
            raise PredictionInputError("resource admission is not an accepted formal execution record")
    models = _parse_models_v2(document["model_tasks"]) if v2 else _parse_models(document["model_tasks"])
    admission = document["admission"]
    required_admission = {"formal_admission", "formal_admission_sha256"}
    if not isinstance(admission, dict) or set(admission) != required_admission:
        raise PredictionInputError("formal admission binding is incomplete")
    formal_admission = _bound_path(admission["formal_admission"], admission["formal_admission_sha256"], name="formal admission")
    try:
        admission_document = json.loads(formal_admission.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise PredictionInputError("formal admission is not valid JSON") from error
    _no_supervision(admission_document)
    registry = document["model_tasks"] if v2 else None
    admission_binding = prediction_execution_binding_sha256(bindings=bindings, protocol=document["protocol"],
                                                            identity_manifest_sha256=document["identity_manifest_sha256"], identities=identities,
                                                            model_registry=registry)
    if (not isinstance(admission_document, dict) or admission_document.get("status") != "PASS" or admission_document.get("formal_execution_allowed") is not True or
            admission_document.get("embedded_vision_binding_sha256") != bindings["embedded_vision_binding_sha256"] or
            admission_document.get("prediction_execution_binding_sha256") != admission_binding):
        raise PredictionInputError("formal admission does not attest the exact embedded vision binding")
    matrix_id = None
    if v2:
        candidate = document["candidate"]
        if not isinstance(candidate, dict) or set(candidate) != {"path", "sha256"}:
            raise PredictionInputError("v2 prediction candidate binding is incomplete")
        candidate_path = _bound_path(candidate["path"], candidate["sha256"], name="prediction plan candidate")
        try:
            candidate_document = json.loads(candidate_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise PredictionInputError("prediction plan candidate is not valid JSON") from error
        _no_supervision(candidate_document)
        candidate_registration = {name: document[name] for name in base_fields}
        if (not isinstance(candidate_document, dict) or set(candidate_document) != {"schema", "matrix_id", "registration", "prediction_execution_binding_sha256"} or
                candidate_document.get("schema") != CANDIDATE_SCHEMA_V2 or candidate_document.get("registration") != candidate_registration or
                candidate_document.get("prediction_execution_binding_sha256") != admission_binding):
            raise PredictionInputError("prediction plan candidate differs from the admitted registry")
        expected_matrix = prediction_matrix_id(run_id=document["run_id"], bindings=bindings, protocol=document["protocol"],
                                              identity_manifest_sha256=document["identity_manifest_sha256"], denominators=document["denominators"])
        matrix_id = document["matrix_id"]
        if matrix_id != expected_matrix or candidate_document.get("matrix_id") != matrix_id:
            raise PredictionInputError("prediction matrix identity differs from the immutable registration")
        if (admission_document.get("prediction_plan_candidate_sha256") != candidate["sha256"] or
                admission_document.get("resource_admission_sha256") != bindings["resource_admission_sha256"]):
            raise PredictionInputError("formal admission does not attest the candidate/resource binding")
    denominators = document["denominators"]
    if not isinstance(denominators, dict) or set(denominators) != set(OFFICIAL_DENOMINATORS) or any(type(denominators[k]) is not int or denominators[k] < 1 for k in denominators):
        raise PredictionInputError("invalid denominator contract")
    protocol = document["protocol"]
    required_protocol = {"hivau", "vad_route", "vad_causal_smoothing", "vad_fast_fusion"}
    hivau = protocol.get("hivau") if isinstance(protocol, dict) else None
    hivau_keys = {"target_fps", "query_interval", "paligemma_batch_size", "max_new_tokens", "task", "fast_prompt_context"}
    if not isinstance(protocol, dict) or not required_protocol.issubset(protocol) or not isinstance(hivau, dict) or set(hivau) != hivau_keys or any(type(hivau[key]) is not int or hivau[key] < 1 for key in ("target_fps", "query_interval", "paligemma_batch_size", "max_new_tokens")) or not isinstance(hivau["task"], str) or not hivau["task"] or hivau["fast_prompt_context"] != "none":
        raise PredictionInputError("VAU must bind the inherited HIVAU generation protocol without Fast prompt injection")
    if protocol["vad_route"] != "reactvau_detection" or protocol["vad_causal_smoothing"] != "online" or not isinstance(protocol["vad_fast_fusion"], str):
        raise PredictionInputError("VAD must bind the inherited detection/fusion/causal-smoothing route")
    vad = protocol.get("vad")
    if not isinstance(vad, dict) or set(vad) != {"target_fps", "query_interval", "batch_size"} or any(type(vad[key]) is not int or vad[key] < 1 for key in vad):
        raise PredictionInputError("VAD sampling protocol is incomplete")
    if dict(denominators) != OFFICIAL_DENOMINATORS and not document["run_id"].startswith("test:"):
        raise PredictionInputError("formal prediction requires the 251/800/3339 official denominators")
    output_root = Path(document["output_root"])
    if not output_root.is_absolute():
        raise PredictionInputError("output_root must be absolute")
    verified_media: dict[tuple[str, str], tuple[int, int, int]] = {}
    if registration is not None:
        return PredictionPlan(document["schema"], document["run_id"], actual, registration["vad"], registration["vau"], registration["models"],
                              registration["bindings"], registration["binding_sha256"], registration["protocol"], registration["output_root"],
                              matrix_id, registration["execution_scope_sha256"], registration["resource_scope"])
    return PredictionPlan(document["schema"], document["run_id"], actual, _parse_vad(identities["vad"], denominators, verified_media),
                          _parse_vau(identities["vau"], denominators["vau"], verified_media), models, bound, bound_hashes,
                          protocol, output_root, matrix_id)
