"""Blind, hash-bound inputs for NC-RTED prediction.

This module deliberately has no dependency on the training catalog.  In
particular, it never accepts targets, references, labels, answers, or metric
paths.  A prediction plan can therefore be inspected before any model is
constructed without making official test supervision available to a worker.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


SCHEMA = "nc_rted_blind_prediction/v1"
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

    @property
    def task_id(self) -> str:
        return self.group if self.seed is None else f"{self.group}:seed{self.seed}"


@dataclass(frozen=True)
class PredictionPlan:
    run_id: str
    manifest_sha256: str
    vad: tuple[VadRequest, ...]
    vau: tuple[VauRequest, ...]
    models: tuple[ModelTask, ...]
    bindings: Mapping[str, str]
    binding_sha256: Mapping[str, str]
    protocol: Mapping[str, Any]
    output_root: Path

    def requests(self) -> tuple[VadRequest | VauRequest, ...]:
        return self.vad + self.vau

    def selected_model(self, group: str, seed: int | None) -> ModelTask:
        for artifact in self.models:
            if artifact.group == group and artifact.seed == seed:
                return artifact
        raise PredictionInputError(f"unknown prediction model task {group}:{seed}")


def prediction_execution_binding_sha256(*, bindings: Mapping[str, str], protocol: Mapping[str, Any],
                                        identity_manifest_sha256: str, identities: Mapping[str, Any]) -> str:
    """Digest every admitted execution input, including ordered blind identities."""
    value = {
        "bindings": dict(bindings),
        "identity_manifest_sha256": identity_manifest_sha256,
        "identity_mapping": identities,
        "implementation_manifest_sha256": bindings["implementation_manifest_sha256"],
        "protocol": dict(protocol),
    }
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
    expected = {("R0", None)} | {(group, seed) for group in GROUPS if group != "R0" for seed in FORMAL_SEEDS}
    if {(item.group, item.seed) for item in models} != expected:
        raise PredictionInputError("model tasks must be R0 plus A/U/S/F for seeds 17, 42, and 2026")
    return tuple(sorted(models, key=lambda item: (GROUPS.index(item.group), -1 if item.seed is None else item.seed)))


def load_model_artifact(path: str | Path, *, expected_sha256: str, task: ModelTask) -> ModelArtifact:
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
    if not isinstance(document, dict) or document.get("schema") != SCHEMA:
        raise PredictionInputError("unsupported blind prediction manifest")
    allowed = {"schema", "run_id", "identity_manifest", "identity_manifest_sha256", "model_tasks", "protocol", "output_root", "denominators", "admission", "bindings"}
    if set(document) != allowed or not isinstance(document.get("run_id"), str) or not document["run_id"]:
        raise PredictionInputError("prediction manifest fields differ from the contract")
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
    if not isinstance(bindings, dict) or set(bindings) != binding_names | {f"{name}_sha256" for name in binding_names}:
        raise PredictionInputError("runtime/Fast/source/tokenizer/vision/decoder/implementation bindings are incomplete")
    bound = {}
    for name in binding_names:
        validator = _bound_artifact if name == "tokenizer" else _bound_path
        bound[name] = str(validator(bindings[name], bindings[f"{name}_sha256"], name=name))
    bound_hashes = {name: str(bindings[f"{name}_sha256"]) for name in binding_names}
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
    admission_binding = prediction_execution_binding_sha256(bindings=bindings, protocol=document["protocol"],
                                                            identity_manifest_sha256=document["identity_manifest_sha256"], identities=identities)
    if (not isinstance(admission_document, dict) or admission_document.get("status") != "PASS" or admission_document.get("formal_execution_allowed") is not True or
            admission_document.get("embedded_vision_binding_sha256") != bindings["embedded_vision_binding_sha256"] or
            admission_document.get("prediction_execution_binding_sha256") != admission_binding):
        raise PredictionInputError("formal admission does not attest the exact embedded vision binding")
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
    return PredictionPlan(document["run_id"], actual, _parse_vad(identities["vad"], denominators, verified_media), _parse_vau(identities["vau"], denominators["vau"], verified_media), _parse_models(document["model_tasks"]), bound, bound_hashes, protocol, output_root)
