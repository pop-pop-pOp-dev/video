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
RESOURCE_PROJECTION_SCHEMA_V1 = "nc_rted_prediction_resource_projection/v1"
RESOURCE_APPLICABILITY_SCHEMA_V1 = "nc_rted_prediction_target_applicability/v1"
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
    "src/nc_rted/features.py",
    "src/nc_rted/tracking.py",
    "src/nc_rted/caption_provider.py",
    "src/nc_rted/caption_sampling.py",
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

    def execution_requests(self) -> tuple[VadRequest | VauRequest, ...]:
        """Schedule VAU work by bound physical medium without changing identities."""
        grouped: dict[tuple[str, str], list[VauRequest]] = {}
        for request in self.vau:
            grouped.setdefault((request.media_path, request.media_sha256), []).append(request)
        return self.vad + tuple(request for requests in grouped.values() for request in requests)

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


_FULL_PREDICTION_WORKLOAD = {"kind": "blind_prediction", "vad_queries": 135050,
                             "slow_triggers": 47458, "vau_requests": 3339}
_R0_OUTPUT_BUDGET_BYTES = 1_600_000_000
_R0_OUTPUT_GEOMETRY = {"vad_videos_per_model": 1051, "vad_total_frames_per_model": 3372608,
                       "vad_total_queries_per_model": 135050, "vau_requests_per_model": 3339,
                       "max_new_tokens": 512, "maximum_decoded_text_json_escaped_bytes": 393216}
_R0_TARGET_IMPLEMENTATION_DELTA_FILES = frozenset({"src/nc_rted/prediction_inputs.py", "src/nc_rted/prediction_worker.py"})


def _bound_json(path: object, digest: object, *, name: str) -> tuple[Path, dict[str, Any]]:
    bound = _bound_path(path, digest, name=name)
    try:
        document = json.loads(bound.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise PredictionInputError(f"{name} is not valid JSON") from error
    _no_supervision(document)
    if not isinstance(document, dict):
        raise PredictionInputError(f"{name} is not a JSON object")
    return bound, document


def _validate_projection_output_budget(assessment: Mapping[str, Any], output_budget_bytes: object) -> None:
    metadata = assessment.get("bound_metadata")
    payloads = assessment.get("default_factory_payloads")
    if (assessment.get("schema") != "nc_rted_gate02_prediction_output_budget_assessment/v1" or
            assessment.get("status") != "PRACTICAL_BOUND_DERIVED_CAP_NOT_IMPLEMENTED" or
            not isinstance(metadata, dict) or not isinstance(payloads, dict) or
            any(metadata.get(key) != value for key, value in _R0_OUTPUT_GEOMETRY.items()) or
            not isinstance(payloads.get("vad"), str) or not isinstance(payloads.get("vau"), str)):
        raise PredictionInputError("resource projection output assessment geometry differs")
    derived = (3339 * 393216 + 3339 * 512 * 7 + 3339 * 8192 + 3372608 * 33 +
               3372608 * 7 + 135050 * 512 + 1051 * 8192 + 16 * 1024**2)
    if output_budget_bytes != _R0_OUTPUT_BUDGET_BYTES or derived > output_budget_bytes:
        raise PredictionInputError("resource projection output reserve is not the derived default-factory envelope")


def _validate_representative_projection_evidence(*, inventory: Mapping[str, Any], reports: object,
                                                 preflight: Mapping[str, Any]) -> dict[str, int]:
    """Bind historical fixed representatives to their actual non-formal runtime inputs."""
    rows = inventory.get("rows")
    if (inventory.get("schema") != "nc_rted_fixed_prediction_measurement_inventory/v10" or
            inventory.get("status") != "FIXED_REPRESENTATIVES_COMPLETED_NOT_FORMAL_ADMISSION" or not isinstance(rows, list)):
        raise PredictionInputError("resource projection representative inventory differs")
    bindings = preflight.get("bindings")
    if (not isinstance(bindings, dict) or not isinstance(preflight.get("runtime"), dict) or
            not isinstance(preflight.get("identity_manifest"), dict) or not isinstance(preflight.get("r0_model_manifest"), dict) or
            not isinstance(bindings.get("prediction_source"), dict)):
        raise PredictionInputError("resource projection preflight bindings differ")
    expected = {"runtime": preflight["runtime"].get("sha256"), "identity": preflight["identity_manifest"].get("sha256"),
                "model": preflight["r0_model_manifest"].get("sha256"), "source": bindings["prediction_source"].get("sha256")}
    if not all(_sha(value) for value in expected.values()) or not isinstance(reports, list):
        raise PredictionInputError("resource projection representative runtime bindings differ")
    supplied = {(item.get("path"), item.get("sha256")) for item in reports if isinstance(item, dict) and set(item) == {"path", "sha256"}}
    accepted_rows = [row for row in rows if isinstance(row, dict) and row.get("status") == "PASS"]
    if len(accepted_rows) != 21 or {(row.get("report"), row.get("report_sha256")) for row in accepted_rows} != supplied:
        raise PredictionInputError("resource projection requires exactly the 21 accepted representative reports")
    peak_allocated = peak_reserved = 0
    for row in accepted_rows:
        _, acceptance = _bound_json(row.get("acceptance"), row.get("acceptance_sha256"), name="representative acceptance")
        _, probe = _bound_json(row["report"], row["report_sha256"], name="representative timing report")
        candidate = probe.get("candidate_result")
        diagnostic = candidate.get("diagnostic_admission") if isinstance(candidate, dict) else None
        if not isinstance(diagnostic, dict) or set(diagnostic) != {"path", "sha256"}:
            raise PredictionInputError("resource projection representative diagnostic admission differs")
        _, diagnostic_document = _bound_json(diagnostic["path"], diagnostic["sha256"], name="representative diagnostic admission")
        terminal = acceptance.get("terminal")
        if (acceptance.get("report") != row["report"] or acceptance.get("report_sha256") != row["report_sha256"] or
                not isinstance(terminal, dict) or terminal.get("status") != "TERMINAL_RUNTIME_PROBE" or terminal.get("candidate_status") != "PASS_RUNTIME_PROBE" or
                not isinstance(candidate, dict) or candidate.get("status") != "PASS_RUNTIME_PROBE" or candidate.get("identity") != row.get("identity") or
                candidate.get("device") != "cuda:0" or candidate.get("formal_prediction") is not False or candidate.get("prediction_store_written") is not False or
                diagnostic_document.get("schema") != "nc_rted_prediction_runtime_probe_admission/v1" or diagnostic_document.get("status") != "PASS" or
                diagnostic_document.get("device") != candidate.get("device") or diagnostic_document.get("output") != row["report"] or
                candidate.get("runtime", {}).get("sha256") != expected["runtime"] or candidate.get("identity_manifest", {}).get("sha256") != expected["identity"] or
                candidate.get("model_manifest", {}).get("sha256") != expected["model"] or candidate.get("source_manifest", {}).get("sha256") != expected["source"]):
            raise PredictionInputError("resource projection representative runtime/source/identity applicability differs")
        allocated = candidate.get("peak_cuda_allocated_bytes")
        reserved = candidate.get("peak_cuda_reserved_bytes")
        if type(allocated) is not int or type(reserved) is not int or allocated <= 0 or reserved < allocated:
            raise PredictionInputError("resource projection representative peak differs")
        peak_allocated, peak_reserved = max(peak_allocated, allocated), max(peak_reserved, reserved)
    return {"allocated": peak_allocated, "reserved": peak_reserved}


def _validate_projection_timing_inputs(*, timing_inputs: object, inventory_binding: Mapping[str, Any]) -> None:
    """Bind the central estimate to its inventory and source29 overhead acceptances."""
    if not isinstance(timing_inputs, dict) or set(timing_inputs) != {"inventory", "device_timeline", "overhead_acceptances"}:
        raise PredictionInputError("resource projection timing input bindings differ")
    inventory = timing_inputs["inventory"]
    if inventory != inventory_binding:
        raise PredictionInputError("resource projection timing inventory differs from representative inventory")
    device_binding = timing_inputs["device_timeline"]
    overhead = timing_inputs["overhead_acceptances"]
    if (not isinstance(device_binding, dict) or set(device_binding) != {"path", "sha256"} or
            not isinstance(overhead, list) or len(overhead) != 2):
        raise PredictionInputError("resource projection timing input bindings differ")
    _, device_timeline = _bound_json(device_binding.get("path"), device_binding.get("sha256"),
                                     name="resource projection device timeline")
    accepted_inputs = device_timeline.get("accepted_inputs")
    if (device_timeline.get("schema") != "nc_rted_gate02_source29_device_timeline/v1" or
            device_timeline.get("status") != "PLANNING_ESTIMATE_NOT_RESOURCE_ADMISSION" or
            not isinstance(accepted_inputs, dict) or
            not {"gate02_prediction_measurement_inventory_v10.json", "vau_residency_gpu_equivalence_v1.json",
                 "r0_gpu1_arson018_runtime_acceptance_v6.json"}.issubset(set(accepted_inputs.get("source_reports", []))) or
            not isinstance(accepted_inputs.get("vau_new_acceptance"), dict)):
        raise PredictionInputError("resource projection device timeline differs")
    expected = {"vau:869": accepted_inputs["vau_new_acceptance"].get("869"),
                "vau:3286": accepted_inputs["vau_new_acceptance"].get("3286")}
    supplied: dict[str, str] = {}
    for index, binding in enumerate(overhead):
        if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
            raise PredictionInputError("resource projection overhead acceptance binding differs")
        _, acceptance = _bound_json(binding["path"], binding["sha256"], name=f"resource projection overhead acceptance {index}")
        terminal = acceptance.get("terminal")
        identity = acceptance.get("identity")
        if (identity not in expected or identity in supplied or acceptance.get("status") != "PASS_FIXED_REPRESENTATIVE_MEASUREMENT" or
                not isinstance(terminal, dict) or terminal.get("status") != "TERMINAL_RUNTIME_PROBE" or
                terminal.get("candidate_status") != "PASS_RUNTIME_PROBE"):
            raise PredictionInputError("resource projection overhead acceptance differs")
        supplied[identity] = binding["sha256"]
    if supplied != expected or not all(_sha(value) for value in expected.values()):
        raise PredictionInputError("resource projection overhead acceptance does not match device timeline")


def _validate_target_applicability(binding: object, *, projection: Mapping[str, Any], scope: Mapping[str, Any],
                                   target_bindings: Mapping[str, Any] | None = None) -> None:
    """Connect historical representative measurements to the final execution target."""
    if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
        raise PredictionInputError("resource projection target applicability binding differs")
    _, record = _bound_json(binding["path"], binding["sha256"], name="resource projection target applicability")
    fields = {"schema", "status", "execution_scope_sha256", "host", "device", "physical_gpu_uuid", "historical_preflight",
              "historical_inventory", "target_preflight", "cache_preflight", "target_bindings", "resource_observation", "historical_launch",
              "cache_acceptance", "allowed_implementation_deltas", "zero_timing_speed_credit"}
    if (set(record) != fields or record.get("schema") != RESOURCE_APPLICABILITY_SCHEMA_V1 or
            record.get("status") != "PASS_CONSERVATIVE_TARGET_APPLICABILITY" or
            record.get("execution_scope_sha256") != scope["execution_scope_sha256"] or record.get("host") != projection["host"] or
            record.get("device") != scope["device"] or record.get("physical_gpu_uuid") != scope["physical_gpu_uuid"] or
            record.get("zero_timing_speed_credit") is not True or record.get("historical_preflight") != projection["source29_preflight"] or
            record.get("historical_inventory") != projection["representative_inventory"]):
        raise PredictionInputError("resource projection target applicability differs")
    target_preflight_ref, cache_preflight_ref = record["target_preflight"], record["cache_preflight"]
    if any(not isinstance(item, dict) or set(item) != {"path", "sha256"} for item in (target_preflight_ref, cache_preflight_ref)):
        raise PredictionInputError("resource projection target preflight binding differs")
    _, historical = _bound_json(projection["source29_preflight"]["path"], projection["source29_preflight"]["sha256"], name="historical preflight")
    _, target = _bound_json(target_preflight_ref["path"], target_preflight_ref["sha256"], name="target preflight")
    _, cache_preflight = _bound_json(cache_preflight_ref["path"], cache_preflight_ref["sha256"], name="cache gate preflight")
    equal = {"runtime_sha256": historical.get("runtime", {}).get("sha256"), "identity_manifest_sha256": historical.get("identity_manifest", {}).get("sha256"),
             "r0_model_manifest_sha256": historical.get("r0_model_manifest", {}).get("sha256"),
             "source_manifest_sha256": historical.get("bindings", {}).get("prediction_source", {}).get("sha256")}
    target_values = {"runtime_sha256": target.get("runtime", {}).get("sha256"), "identity_manifest_sha256": target.get("identity_manifest", {}).get("sha256"),
                     "r0_model_manifest_sha256": target.get("r0_model_manifest", {}).get("sha256"),
                     "source_manifest_sha256": target.get("bindings", {}).get("prediction_source", {}).get("sha256")}
    stated = record.get("target_bindings")
    if (not isinstance(stated, dict) or set(stated) != {"runtime_sha256", "fast_snapshot_sha256", "source_manifest_sha256", "tokenizer_sha256",
                                                        "embedded_vision_binding_sha256", "decoder_sha256", "implementation_manifest_sha256",
                                                        "identity_manifest_sha256", "r0_model_manifest_sha256"} or
            any(not _sha(value) for value in stated.values()) or target_values != equal or
            any(stated[key] != value for key, value in target_values.items()) or
            stated["implementation_manifest_sha256"] != target.get("bindings", {}).get("implementation", {}).get("sha256")):
        raise PredictionInputError("resource projection target bindings are not applicable to historical evidence")
    target_implementation = target.get("bindings", {}).get("implementation")
    cache_implementation = cache_preflight.get("bindings", {}).get("implementation")
    allowed = record.get("allowed_implementation_deltas")
    if (not isinstance(target_implementation, dict) or not isinstance(cache_implementation, dict) or
            not isinstance(allowed, list) or set(allowed) != _R0_TARGET_IMPLEMENTATION_DELTA_FILES):
        raise PredictionInputError("resource projection implementation applicability differs")
    _, target_implementation_document = _bound_json(target_implementation.get("path"), target_implementation.get("sha256"), name="target implementation manifest")
    _, cache_implementation_document = _bound_json(cache_implementation.get("path"), cache_implementation.get("sha256"), name="cache implementation manifest")
    target_files, cache_files = target_implementation_document.get("files"), cache_implementation_document.get("files")
    if not isinstance(target_files, dict) or not isinstance(cache_files, dict):
        raise PredictionInputError("resource projection implementation manifests differ")
    changed = {key for key in set(target_files) | set(cache_files) if target_files.get(key) != cache_files.get(key)}
    if changed != _R0_TARGET_IMPLEMENTATION_DELTA_FILES:
        raise PredictionInputError("resource projection target changes inference files beyond the accepted applicability delta")
    observation_ref = record["resource_observation"]
    launch_ref = record["historical_launch"]
    cache_ref = record["cache_acceptance"]
    if any(not isinstance(item, dict) or set(item) != {"path", "sha256"} for item in (observation_ref, launch_ref, cache_ref)):
        raise PredictionInputError("resource projection target physical evidence binding differs")
    _, observation = _bound_json(observation_ref["path"], observation_ref["sha256"], name="target resource observation")
    _, launch = _bound_json(launch_ref["path"], launch_ref["sha256"], name="historical physical launch")
    _, cache = _bound_json(cache_ref["path"], cache_ref["sha256"], name="source31 cache acceptance")
    _, cache_report = _bound_json(cache.get("report"), cache.get("report_sha256"), name="source31 cache runtime report")
    if (observation.get("schema") != "nc_rted_root_resource_observation/v1" or observation.get("status") != "OBSERVED_NOT_FORMAL_ADMISSION" or
            observation.get("host") != record["host"] or record["physical_gpu_uuid"] not in "\n".join(observation.get("gpu_inventory", [])) or
            launch.get("physical_gpu_uuid") != record["physical_gpu_uuid"] or launch.get("preflight_sha256") != projection["source29_preflight"]["sha256"] or
            cache.get("status") != "PASS_CACHE_EQUIVALENCE_GPU_GATE" or cache.get("formal_prediction") is not False or
            cache_report.get("candidate_result", {}).get("preflight_report", {}).get("sha256") != cache_preflight_ref["sha256"] or
            cache_report.get("candidate_result", {}).get("prediction_store_written") is not False or
            cache.get("peak_cuda_reserved_bytes", 0) > projection["peak_cuda_reserved_bytes"]):
        raise PredictionInputError("resource projection target physical/cache applicability differs")
    if target_bindings is not None and any(target_bindings.get(key) != value for key, value in stated.items()):
        raise PredictionInputError("target applicability does not bind the final prediction registration")


def _validate_resource_projection(projection: object, *, allocation_sha256: str, host: str,
                                  physical_gpu_uuid: str, scope: Mapping[str, Any]) -> None:
    """Validate a conservative estimate without relabelling it a full measurement."""
    fields = {"schema", "status", "projection_basis", "kind", "execution_scope_sha256", "allocation_sha256",
              "host", "physical_gpu_uuid", "workload", "representative_inventory", "representative_reports", "peak_evidence", "source29_preflight",
              "prediction_source_sha256", "projection_report", "timing_inputs", "target_applicability", "output_budget_assessment",
              "projected_runtime_seconds", "safety_margin_fraction", "runtime_budget_seconds",
              "peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes", "output_budget_bytes", "required_free_bytes"}
    if (not isinstance(projection, dict) or set(projection) != fields or
            projection.get("schema") != RESOURCE_PROJECTION_SCHEMA_V1 or
            projection.get("status") != "PROJECTED_FROM_REPRESENTATIVE_MEASUREMENTS" or
            projection.get("projection_basis") != "CONSERVATIVE_FULL_WORKLOAD_ESTIMATE" or
            projection.get("kind") != "blind_prediction" or
            projection.get("execution_scope_sha256") != scope["execution_scope_sha256"] or
            projection.get("allocation_sha256") != allocation_sha256 or projection.get("host") != host or
            projection.get("physical_gpu_uuid") != physical_gpu_uuid or
            projection.get("workload") != _FULL_PREDICTION_WORKLOAD):
        raise PredictionInputError("accepted resource projection differs from accepted allocation")
    inventory_binding = projection["representative_inventory"]
    if not isinstance(inventory_binding, dict) or set(inventory_binding) != {"path", "sha256"}:
        raise PredictionInputError("resource projection representative inventory binding differs")
    _, inventory = _bound_json(inventory_binding["path"], inventory_binding["sha256"], name="resource projection representative inventory")
    reports = projection["representative_reports"]
    if not isinstance(reports, list) or not reports:
        raise PredictionInputError("resource projection requires hash-bound representative reports")
    seen: set[tuple[str, str]] = set()
    for index, binding in enumerate(reports):
        if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
            raise PredictionInputError("resource projection representative report binding differs")
        report_path, report_hash = binding["path"], binding["sha256"]
        key = (report_path, report_hash)
        if not isinstance(report_path, str) or not _sha(report_hash) or key in seen:
            raise PredictionInputError("resource projection representative reports must be unique and hash-bound")
        seen.add(key)
        _bound_json(report_path, report_hash, name=f"representative timing report {index}")
    peak_binding = projection["peak_evidence"]
    if (not isinstance(peak_binding, dict) or set(peak_binding) != {"path", "sha256"} or
            (peak_binding["path"], peak_binding["sha256"]) not in seen):
        raise PredictionInputError("resource projection peak evidence must be an accepted representative report")
    evidence: dict[str, dict[str, Any]] = {}
    for name in ("source29_preflight", "projection_report", "output_budget_assessment"):
        binding = projection[name]
        if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
            raise PredictionInputError("resource projection evidence binding differs")
        _, evidence[name] = _bound_json(binding["path"], binding["sha256"], name=f"resource projection {name}")
    preflight = evidence["source29_preflight"]
    source = preflight.get("bindings")
    if (preflight.get("schema") != "nc_rted_r0_blind_manifest_build/v2" or
            preflight.get("status") != "PREFLIGHT_PASS_FORMAL_ADMISSION_REQUIRED" or preflight.get("gpu_launched") is not False or
            not isinstance(source, dict) or not isinstance(source.get("prediction_source"), dict) or
            source["prediction_source"].get("sha256") != projection.get("prediction_source_sha256") or
            not _sha(projection.get("prediction_source_sha256"))):
        raise PredictionInputError("resource projection source29 preflight/source binding differs")
    observed_peak = _validate_representative_projection_evidence(inventory=inventory, reports=reports,
                                                                  preflight=evidence["source29_preflight"])
    if (projection["peak_cuda_allocated_bytes"] < observed_peak["allocated"] or
            projection["peak_cuda_reserved_bytes"] < observed_peak["reserved"]):
        raise PredictionInputError("resource projection peak evidence does not support its capacity claim")
    timeline = evidence["projection_report"]
    central = timeline.get("central_per_model_seconds") if isinstance(timeline, dict) else None
    margin = timeline.get("safety_margin") if isinstance(timeline, dict) else None
    if (timeline.get("schema") != "nc_rted_gate02_source29_stratified_timeline/v1" or
            timeline.get("status") != "CENTRAL_PLANNING_ESTIMATE_NOT_ADMISSION_OR_GUARANTEE" or not isinstance(central, dict) or
            not isinstance(margin, dict) or central.get("total") != projection["projected_runtime_seconds"] or
            margin.get("fraction") != projection["safety_margin_fraction"]):
        raise PredictionInputError("resource projection report does not support its estimate")
    _validate_projection_timing_inputs(timing_inputs=projection["timing_inputs"], inventory_binding=inventory_binding)
    _validate_target_applicability(projection["target_applicability"], projection=projection, scope=scope)
    numeric = ("projected_runtime_seconds", "safety_margin_fraction", "runtime_budget_seconds")
    if (any(type(projection.get(key)) not in {int, float} or isinstance(projection[key], bool) or not math.isfinite(projection[key]) for key in numeric) or
            projection["projected_runtime_seconds"] <= 0 or projection["safety_margin_fraction"] <= 0 or
            projection["runtime_budget_seconds"] < projection["projected_runtime_seconds"] * (1 + projection["safety_margin_fraction"]) or
            projection["runtime_budget_seconds"] > scope["run_budget_seconds"]):
        raise PredictionInputError("resource projection runtime estimate or conservative budget differs")
    if (any(type(projection.get(key)) is not int or projection[key] <= 0 for key in
            ("peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes", "output_budget_bytes", "required_free_bytes")) or
            projection["peak_cuda_reserved_bytes"] < projection["peak_cuda_allocated_bytes"] or
            projection["required_free_bytes"] != scope["min_free_bytes"] or
            projection["required_free_bytes"] < 20 * 1024**3 + projection["output_budget_bytes"]):
        raise PredictionInputError("resource projection peak/output reserve differs from admitted scope")
    _validate_projection_output_budget(evidence["output_budget_assessment"], projection["output_budget_bytes"])


def _validate_resource_admission(path: str, digest: str, *, scope_sha256: str) -> dict[str, Any]:
    resource_path = _bound_path(path, digest, name="resource admission")
    try:
        resource = json.loads(resource_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise PredictionInputError("resource admission is not valid JSON") from error
    _no_supervision(resource)
    required = {"schema", "status", "formal_execution_allowed", "scope", "accepted_allocation"}
    scope_fields = {"kind", "run_id", "execution_scope_sha256", "device", "physical_gpu_uuid", "data_volume", "min_free_bytes", "run_budget_seconds", "deadline_utc_epoch"}
    if (not isinstance(resource, dict) or (set(resource) != required | {"accepted_measurement"} and set(resource) != required | {"accepted_projection"}) or
            resource.get("schema") != RESOURCE_ADMISSION_SCHEMA_V1 or resource.get("status") != "PASS" or resource.get("formal_execution_allowed") is not True):
        raise PredictionInputError("resource admission is not an accepted scoped formal execution record")
    scope = resource["scope"]
    if (not isinstance(scope, dict) or set(scope) != scope_fields or scope.get("kind") != "blind_prediction" or not isinstance(scope.get("run_id"), str) or not scope["run_id"] or
            scope.get("execution_scope_sha256") != scope_sha256 or not isinstance(scope.get("device"), str) or not scope["device"].startswith("cuda:") or
            not isinstance(scope.get("physical_gpu_uuid"), str) or not scope["physical_gpu_uuid"] or not isinstance(scope.get("data_volume"), str) or not Path(scope["data_volume"]).is_absolute() or
            type(scope.get("min_free_bytes")) is not int or scope["min_free_bytes"] < 20 * 1024**3 or type(scope.get("run_budget_seconds")) is not int or scope["run_budget_seconds"] < 1 or
            type(scope.get("deadline_utc_epoch")) not in {int, float} or isinstance(scope["deadline_utc_epoch"], bool) or not math.isfinite(scope["deadline_utc_epoch"]) or scope["deadline_utc_epoch"] <= 0):
        raise PredictionInputError("resource admission scope differs from blind prediction execution")
    allocation_binding = resource["accepted_allocation"]
    accepted_name = "accepted_measurement" if "accepted_measurement" in resource else "accepted_projection"
    evidence_binding = resource[accepted_name]
    if (not isinstance(allocation_binding, dict) or set(allocation_binding) != {"path", "sha256"} or
            not isinstance(evidence_binding, dict) or set(evidence_binding) != {"path", "sha256"}):
        raise PredictionInputError("resource admission accepted allocation/measurement bindings are incomplete")
    allocation_path = _bound_path(allocation_binding["path"], allocation_binding["sha256"], name="accepted resource allocation")
    evidence_path = _bound_path(evidence_binding["path"], evidence_binding["sha256"], name=f"accepted resource {accepted_name.removeprefix('accepted_')}")
    try:
        allocation = json.loads(allocation_path.read_text(encoding="utf-8"))
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise PredictionInputError("resource allocation/measurement evidence is not valid JSON") from error
    _no_supervision(allocation); _no_supervision(evidence)
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
    if accepted_name == "accepted_measurement":
        measurement = evidence
        if (not isinstance(measurement, dict) or set(measurement) != measurement_fields or
                measurement.get("schema") != RESOURCE_MEASUREMENT_SCHEMA_V1 or measurement.get("status") != "MEASURED" or measurement.get("kind") != "blind_prediction" or
                measurement.get("execution_scope_sha256") != scope_sha256 or measurement.get("allocation_sha256") != allocation_binding["sha256"] or measurement.get("host") != allocation["host"] or measurement.get("physical_gpu_uuid") != allocation["physical_gpu_uuid"] or
                measurement.get("workload") != _FULL_PREDICTION_WORKLOAD or
                type(measurement.get("measured_at_utc_epoch")) not in {int, float} or not math.isfinite(measurement["measured_at_utc_epoch"]) or measurement["measured_at_utc_epoch"] <= 0 or measurement["measured_at_utc_epoch"] > time.time() or measurement["measured_at_utc_epoch"] > scope["deadline_utc_epoch"] or
                type(measurement.get("qualified_runtime_seconds")) not in {int, float} or not math.isfinite(measurement["qualified_runtime_seconds"]) or measurement["qualified_runtime_seconds"] <= 0 or measurement["qualified_runtime_seconds"] > scope["run_budget_seconds"] or
                any(type(measurement.get(key)) is not int or measurement[key] < 0 for key in ("peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes")) or measurement["peak_cuda_reserved_bytes"] < measurement["peak_cuda_allocated_bytes"]):
            raise PredictionInputError("accepted resource measurement differs from accepted allocation")
    else:
        _validate_resource_projection(evidence, allocation_sha256=allocation_binding["sha256"], host=allocation["host"],
                                      physical_gpu_uuid=allocation["physical_gpu_uuid"], scope=scope)
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
    common_names = {"resource_authorization", "resource_allocation"}
    accepted = registry.get("accepted") if isinstance(registry, dict) else None
    if (not isinstance(registry, dict) or set(registry) != required or registry.get("schema") != RESOURCE_EVIDENCE_REGISTRY_SCHEMA_V1 or
            not isinstance(accepted, dict) or (set(accepted) != common_names | {"resource_measurement"} and set(accepted) != common_names | {"resource_projection"})):
        raise PredictionInputError("operator accepted evidence registry differs")
    evidence_name = "resource_measurement" if "resource_measurement" in accepted else "resource_projection"
    names = common_names | {evidence_name}
    bound = {name: _bound_path(registry["accepted"][name].get("path") if isinstance(registry["accepted"][name], dict) else None,
                               registry["accepted"][name].get("sha256") if isinstance(registry["accepted"][name], dict) else None,
                               name=f"accepted {name}") for name in names}
    try:
        authorization = json.loads(bound["resource_authorization"].read_text(encoding="utf-8"))
        allocation = json.loads(bound["resource_allocation"].read_text(encoding="utf-8"))
        evidence = json.loads(bound[evidence_name].read_text(encoding="utf-8"))
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
    if evidence_name == "resource_measurement":
        measurement = evidence
        if (not isinstance(measurement, dict) or measurement.get("schema") != RESOURCE_MEASUREMENT_SCHEMA_V1 or measurement.get("status") != "MEASURED" or
                measurement.get("allocation_sha256") != allocation_hash or measurement.get("host") != host or measurement.get("physical_gpu_uuid") != physical_gpu_uuid or
                measurement.get("workload") != _FULL_PREDICTION_WORKLOAD or
                type(measurement.get("peak_cuda_allocated_bytes")) is not int or measurement["peak_cuda_allocated_bytes"] <= 0 or type(measurement.get("peak_cuda_reserved_bytes")) is not int or
                not (measurement["peak_cuda_allocated_bytes"] <= measurement["peak_cuda_reserved_bytes"] <= capacity_bytes) or measurement.get("qualified_runtime_seconds", 0) > scope["run_budget_seconds"]):
            raise PredictionInputError("accepted measurement does not qualify this execution")
    else:
        _validate_resource_projection(evidence, allocation_sha256=allocation_hash, host=host,
                                      physical_gpu_uuid=physical_gpu_uuid, scope=scope)
        if evidence["peak_cuda_reserved_bytes"] > capacity_bytes:
            raise PredictionInputError("accepted projection exceeds this execution capacity")


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
    if resource_scope is not None:
        resource_document = json.loads(Path(bindings["resource_admission"]).read_text(encoding="utf-8"))
        projection_binding = resource_document.get("accepted_projection") if isinstance(resource_document, dict) else None
        if projection_binding is not None:
            if not isinstance(projection_binding, dict):
                raise PredictionInputError("resource admission projected target applicability differs")
            _, projection = _bound_json(projection_binding.get("path"), projection_binding.get("sha256"), name="accepted resource projection")
            r0 = next((task for task in models if task.task_id == "R0"), None)
            target = {f"{name}_sha256": bindings[f"{name}_sha256"] for name in names - {"resource_admission"}}
            target["identity_manifest_sha256"] = document["identity_manifest_sha256"]
            target["r0_model_manifest_sha256"] = r0.model_manifest_sha256 if r0 is not None else None
            _validate_target_applicability(projection.get("target_applicability"), projection=projection, scope=resource_scope,
                                           target_bindings=target)
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
