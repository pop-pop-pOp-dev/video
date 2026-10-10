"""Concrete, label-free assembly for one NC-RTED blind prediction worker."""
from __future__ import annotations

from dataclasses import dataclass
import importlib
import importlib.abc
from importlib.machinery import PathFinder
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import torch

from .prediction_inputs import (PredictionInputError, _bound_artifact, _bound_path, _no_supervision, _sha, sha256_file,
                                verify_implementation_manifest)


SCHEMA = "nc_rted_prediction_runtime/v2"
_EXPORT_FILES = {"config.json", "adapter_config.json", "adapter_model.safetensors", "non_lora_trainables.bin"}
_REQUIRED_SOURCES = {"llava/train/train.py", "llava/conversation.py", "llava/model/multimodal_encoder/siglip_encoder.py",
                     "llava/model/multimodal_projector/memory_manager.py", "eval_utils/vad/eval_reactvau_detection.py",
                     "eval_utils/vad/detect_utils.py", "eval_utils/hivau/reactvau_inference.py", "llava/mm_utils.py",
                     "llava/constants.py", "vad/get_prompt.py"}
_VERIFIED_INHERITED_MODULES: dict[str, tuple[Path, str]] = {}
_INHERITED_PREFIXES = ("llava", "eval_utils", "vad")
_INHERITED_ALIASES = {"detect_utils": "eval_utils/vad/detect_utils.py"}


@dataclass(frozen=True)
class PredictionRuntime:
    path: Path
    sha256: str
    document: dict[str, Any]

    @property
    def inherited(self) -> dict[str, Any]:
        return self.document["inherited"]


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PredictionInputError(f"prediction runtime {name} is invalid")
    return value


def _source_manifest(inherited: dict[str, Any]) -> None:
    root = Path(inherited["external_root"])
    if not root.is_absolute() or not root.is_dir():
        raise PredictionInputError("ReactVAU source root is unavailable")
    manifest = _bound_path(inherited["source_manifest"], inherited["source_manifest_sha256"], name="ReactVAU source manifest")
    try:
        document = json.loads(manifest.read_text(encoding="utf-8"))
        _no_supervision(document)
        files = document["files"]
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise PredictionInputError("ReactVAU source manifest is invalid") from error
    if not isinstance(files, dict) or not files:
        raise PredictionInputError("ReactVAU source manifest is incomplete")
    if not _REQUIRED_SOURCES.issubset(files):
        raise PredictionInputError("ReactVAU source manifest omits prediction imports")
    for candidate in root.rglob("*"):
        if candidate.is_symlink():
            raise PredictionInputError("ReactVAU source tree cannot contain symlinks")
    executable = {str(path.relative_to(root)) for path in root.rglob("*.py") if path.is_file()}
    if set(files) != executable:
        raise PredictionInputError("ReactVAU source manifest must bind the complete executable Python source set")
    for relative, digest in files.items():
        if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise PredictionInputError("ReactVAU source manifest escapes its root")
        _bound_path(str(root / relative), digest, name="ReactVAU source")


def load_prediction_runtime(path: str | Path, *, expected_sha256: str) -> PredictionRuntime:
    root = _bound_path(str(Path(path).absolute()), expected_sha256, name="prediction runtime")
    try:
        doc = json.loads(root.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise PredictionInputError("prediction runtime is invalid JSON") from error
    _no_supervision(doc)
    required = {"schema", "inherited", "fast", "media", "vision", "detector", "protocols", "generation", "numerics", "cache"}
    if not isinstance(doc, dict) or set(doc) != required or doc.get("schema") != SCHEMA:
        raise PredictionInputError("prediction runtime schema differs")
    inherited = _mapping(doc["inherited"], "inherited")
    inherited_fields = {"external_root", "base_directory", "base_directory_sha256", "stage2_export", "stage2_export_sha256", "stage2_export_hashes", "tokenizer", "tokenizer_sha256", "source_manifest", "source_manifest_sha256"}
    if set(inherited) != inherited_fields:
        raise PredictionInputError("prediction inherited binding is incomplete")
    _bound_artifact(inherited["base_directory"], inherited["base_directory_sha256"], name="base model")
    _bound_artifact(inherited["stage2_export"], inherited["stage2_export_sha256"], name="original final Stage2 export")
    hashes = _mapping(inherited["stage2_export_hashes"], "stage2_export_hashes")
    if set(hashes) != _EXPORT_FILES or not all(_sha(value) for value in hashes.values()):
        raise PredictionInputError("original final Stage2 export hashes are incomplete")
    for name, digest in hashes.items():
        _bound_path(str(Path(inherited["stage2_export"]) / name), digest, name="original final Stage2 export file")
    _bound_artifact(inherited["tokenizer"], inherited["tokenizer_sha256"], name="tokenizer")
    _source_manifest(inherited)
    sections = {
        "fast": {"snapshot", "snapshot_sha256", "model", "model_sha256", "lora", "lora_sha256", "streamforest_weights", "streamforest_weights_sha256", "attn_implementation", "image_size", "vision_feature_layer"},
        "media": {"catalog", "catalog_sha256"},
        "vision": {"derived_snapshot", "derived_snapshot_sha256", "parent_export", "parent_export_sha256", "raw_snapshot", "raw_snapshot_sha256"},
        "detector": {"snapshot", "snapshot_sha256", "score_threshold"},
        "numerics": {"policy", "policy_sha256"},
    }
    for section, fields in sections.items():
        value = _mapping(doc[section], section)
        if set(value) != fields:
            raise PredictionInputError(f"prediction runtime {section} binding is incomplete")
    for section, name, directory in (("fast", "snapshot", False), ("fast", "model", True), ("fast", "lora", True), ("fast", "streamforest_weights", False), ("media", "catalog", False), ("vision", "derived_snapshot", True), ("vision", "raw_snapshot", True), ("vision", "parent_export", False), ("detector", "snapshot", True), ("numerics", "policy", False)):
        value = doc[section][name]
        digest = doc[section][f"{name}_sha256"]
        if value is None:
            if section != "fast" or name != "lora" or digest is not None:
                raise PredictionInputError(f"prediction runtime {section}.{name} is absent")
            continue
        if directory: _bound_artifact(value, digest, name=f"{section}.{name}")
        else: _bound_path(value, digest, name=f"{section}.{name}")
    if (doc["vision"]["parent_export"] != str(Path(inherited["stage2_export"]) / "non_lora_trainables.bin") or
            doc["vision"]["parent_export_sha256"] != hashes["non_lora_trainables.bin"]):
        raise PredictionInputError("derived vision parent is not the original final Stage2 export")
    if not isinstance(doc["protocols"], dict) or not isinstance(doc["generation"], dict) or doc["generation"].get("do_sample") is not False:
        raise PredictionInputError("prediction protocol or greedy generation binding is invalid")
    cache = _mapping(doc["cache"], "cache")
    if set(cache) != {"root", "max_bytes"} or not isinstance(cache["root"], str) or not Path(cache["root"]).is_absolute() or type(cache["max_bytes"]) is not int or cache["max_bytes"] < 1:
        raise PredictionInputError("prediction cache binding is invalid")
    return PredictionRuntime(root, expected_sha256, doc)


def _restore_trainable(bridge, artifact) -> dict[str, Any]:
    if artifact.group == "R0":
        if artifact.checkpoint is not None:
            raise PredictionInputError("R0 cannot restore an incremental checkpoint")
        return {"source": "original_final_stage2", "trainable": 0}
    directory = Path(artifact.checkpoint)
    manifest_path, state_path = directory / "manifest.json", directory / "state.pt"
    _bound_path(str(manifest_path), artifact.checkpoint_manifest_sha256, name="selected checkpoint manifest")
    _bound_path(str(state_path), artifact.checkpoint_state_sha256, name="selected checkpoint state")
    try: manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error: raise PredictionInputError("selected checkpoint manifest is invalid") from error
    identity = manifest.get("identity") if isinstance(manifest, dict) else None
    if not isinstance(identity, dict) or identity.get("group") != artifact.group or identity.get("seed") != str(artifact.seed):
        raise PredictionInputError("selected checkpoint group or seed differs from model task")
    from .recovery import RecoveryError, validate_checkpoint_payload
    try: state = validate_checkpoint_payload(state_path, manifest)
    except RecoveryError as error: raise PredictionInputError("selected checkpoint state is invalid") from error
    live = {name: value for name, value in bridge.named_parameters() if value.requires_grad}
    saved = state["trainable"]
    if set(live) != set(saved):
        raise PredictionInputError("checkpoint trainable keys differ from original Slow/evidence layout")
    with torch.no_grad():
        for name, parameter in live.items():
            value = saved[name]
            if value.shape != parameter.shape or value.dtype != parameter.dtype or not bool(torch.isfinite(value).all()):
                raise PredictionInputError(f"checkpoint trainable tensor differs: {name}")
            parameter.copy_(value.to(parameter.device))
    return {"source": str(directory), "trainable": len(saved), "state_sha256": artifact.checkpoint_state_sha256}


class _BoundInheritedImportGuard(importlib.abc.MetaPathFinder):
    """Admit inherited package sources before Python executes their code."""
    def __init__(self, root: str, manifest: dict[str, str]):
        self.root = Path(root).resolve()
        self.manifest = manifest

    def find_spec(self, fullname, path=None, target=None):
        if not (fullname in _INHERITED_ALIASES or any(fullname == prefix or fullname.startswith(prefix + ".") for prefix in _INHERITED_PREFIXES)):
            return None
        spec = PathFinder.find_spec(fullname, path)
        if spec is None:
            raise PredictionInputError("inherited import has no admitted source origin")
        if spec.origin is None:
            locations = list(spec.submodule_search_locations or ())
            expected_directory = self.root / fullname.replace(".", "/")
            if not locations or any(Path(location).resolve() != expected_directory for location in locations):
                raise PredictionInputError("inherited namespace escapes bound source root")
            for child in expected_directory.rglob("*.py"):
                relative = str(child.relative_to(self.root))
                if relative not in self.manifest or sha256_file(child) != self.manifest[relative]:
                    raise PredictionInputError("inherited namespace differs from complete source manifest")
            return spec
        if spec.origin in {"built-in", "frozen"}:
            raise PredictionInputError("inherited import has no admitted source origin")
        candidate = Path(spec.origin).resolve()
        try:
            relative = str(candidate.relative_to(self.root))
        except ValueError as error:
            raise PredictionInputError("inherited import origin escapes bound source root") from error
        expected = _INHERITED_ALIASES.get(fullname, fullname.replace(".", "/") + ".py")
        package_init = fullname.replace(".", "/") + "/__init__.py"
        expected_relative = package_init if package_init in self.manifest else expected
        if relative != expected_relative or relative not in self.manifest or sha256_file(candidate) != self.manifest[relative]:
            raise PredictionInputError("inherited import differs from complete source manifest")
        try:
            source = candidate.read_bytes()
        except OSError as error:
            raise PredictionInputError("inherited import source is unreadable") from error
        if hashlib.sha256(source).hexdigest() != self.manifest[relative]:
            raise PredictionInputError("inherited import source changed before execution")
        spec.loader = _VerifiedInheritedSourceLoader(fullname, str(candidate), source)
        return spec


class _VerifiedInheritedSourceLoader(importlib.abc.Loader):
    """Execute only source bytes validated by the inherited import guard."""
    def __init__(self, name: str, origin: str, source: bytes):
        self.name, self.origin, self.source = name, origin, source

    def create_module(self, spec):
        return None

    def exec_module(self, module) -> None:
        code = compile(self.source, self.origin, "exec", dont_inherit=True)
        _VERIFIED_INHERITED_MODULES[self.name] = (Path(self.origin).resolve(), hashlib.sha256(self.source).hexdigest())
        exec(code, module.__dict__)


def _install_bound_inherited_import_guard(root: str, manifest: dict[str, str]) -> _BoundInheritedImportGuard:
    resolved = Path(root).resolve()
    for finder in sys.meta_path:
        if isinstance(finder, _BoundInheritedImportGuard):
            if finder.root != resolved or finder.manifest != manifest:
                raise PredictionInputError("a different inherited import guard is already installed")
            return finder
    guard = _BoundInheritedImportGuard(root, manifest)
    sys.meta_path.insert(0, guard)
    return guard


def _import_bound_runtime(runtime: PredictionRuntime) -> None:
    root = runtime.inherited["external_root"]
    # train.py may initialize CUDA through DeepSpeed, therefore this is called
    # only after configure_deterministic_algorithms.
    manifest = json.loads(Path(runtime.inherited["source_manifest"]).read_text(encoding="utf-8"))["files"]
    # Cached parent packages control resolution of every child import. Audit all
    # of them before adding the runtime root or running an inherited initializer.
    _audit_bound_inherited_modules(root, manifest)
    vad_directory = str(Path(root) / "eval_utils" / "vad")
    if vad_directory not in sys.path: sys.path.insert(0, vad_directory)
    if root not in sys.path: sys.path.insert(0, root)
    guard = _install_bound_inherited_import_guard(root, manifest)
    try:
        for name in ("llava.train.train", "llava.conversation", "llava.mm_utils", "llava.constants",
                     "llava.model.multimodal_encoder.siglip_encoder", "llava.model.multimodal_projector.memory_manager",
                     "detect_utils", "vad.get_prompt", "eval_utils.vad.eval_reactvau_detection",
                     "eval_utils.vad.detect_utils", "eval_utils.hivau.reactvau_inference"):
            importlib.import_module(name)
        _audit_bound_inherited_modules(root, manifest)
    except BaseException:
        # A failed preflight must not leave a guard that changes later work.
        if guard in sys.meta_path:
            sys.meta_path.remove(guard)
        raise


def _audit_bound_inherited_modules(root: str, manifest: dict[str, str]) -> None:
    """Reject foreign cached inherited packages and audit every loaded origin."""
    bound_root = Path(root).resolve()
    for name, module in tuple(sys.modules.items()):
        origin = getattr(module, "__file__", None)
        if not (name in _INHERITED_ALIASES or any(name == prefix or name.startswith(prefix + ".") for prefix in _INHERITED_PREFIXES)):
            continue
        if not origin:
            locations = list(getattr(module, "__path__", ()) or ())
            expected_directory = bound_root / name.replace(".", "/")
            if not locations or any(Path(location).resolve() != expected_directory for location in locations):
                raise PredictionInputError("loaded inherited namespace has no bound source origin")
            for child in expected_directory.rglob("*.py"):
                relative = str(child.relative_to(bound_root))
                if relative not in manifest or sha256_file(child) != manifest[relative]:
                    raise PredictionInputError("loaded inherited namespace differs from complete source manifest")
            continue
        candidate = Path(origin).resolve()
        try:
            relative = str(candidate.relative_to(bound_root))
        except ValueError as error:
            raise PredictionInputError("loaded inherited module origin escapes bound source root") from error
        expected = _INHERITED_ALIASES.get(name, name.replace(".", "/") + ".py")
        package_init = name.replace(".", "/") + "/__init__.py"
        if package_init in manifest and relative != package_init:
            raise PredictionInputError("loaded inherited package origin differs from registered import name")
        if package_init not in manifest and relative != expected:
            raise PredictionInputError("loaded inherited module origin differs from registered import name")
        if relative not in manifest or sha256_file(candidate) != manifest[relative]:
            raise PredictionInputError("loaded inherited module differs from complete source manifest")
        admitted = _VERIFIED_INHERITED_MODULES.get(name)
        if admitted != (candidate, manifest[relative]):
            raise PredictionInputError("loaded inherited module lacks verified execution provenance")


def _reconcile_plan(plan, runtime: PredictionRuntime) -> None:
    """Reject a plan whose separately admitted bindings differ from execution."""
    # This occurs before any inherited evaluator/model import. The admission
    # document binds this manifest, while this check catches local source drift.
    verify_implementation_manifest(plan.bindings["implementation_manifest"],
                                   plan.binding_sha256["implementation_manifest"])
    doc, inherited = runtime.document, runtime.inherited
    required = {"fast_snapshot": (doc["fast"]["snapshot"], doc["fast"]["snapshot_sha256"]),
                "source_manifest": (inherited["source_manifest"], inherited["source_manifest_sha256"]),
                "tokenizer": (inherited["tokenizer"], inherited["tokenizer_sha256"])}
    for name, (path, digest) in required.items():
        if plan.bindings[name] != path or plan.binding_sha256[name] != digest:
            raise PredictionInputError(f"plan {name} binding differs from executed runtime")
    report_path = _bound_path(plan.bindings["embedded_vision_binding"], plan.binding_sha256["embedded_vision_binding"], name="embedded vision binding")
    try: report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error: raise PredictionInputError("embedded vision binding report is invalid") from error
    _no_supervision(report)
    if not isinstance(report, dict) or report.get("derived_snapshot_sha256") != doc["vision"]["derived_snapshot_sha256"] or report.get("parent_export_sha256") != doc["vision"]["parent_export_sha256"]:
        raise PredictionInputError("embedded vision binding does not attest executed derived vision")
    decoder_path = _bound_path(plan.bindings["decoder"], plan.binding_sha256["decoder"], name="decoder binding")
    try: decoder = json.loads(decoder_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error: raise PredictionInputError("decoder binding is invalid") from error
    if decoder != {"schema": "nc_rted_blind_decoder/v1", "implementation": "nc_rted.detection_media.OpenCVFrames"}:
        raise PredictionInputError("decoder binding differs from the executed OpenCV decoder")
    protocols = doc["protocols"]
    vad = protocols.get("vad_config")
    if (not isinstance(protocols.get("hivau"), dict) or protocols["hivau"] != plan.protocol["hivau"] or
            not isinstance(vad, dict) or {key: vad.get(key) for key in ("target_fps", "query_interval", "batch_size")} != plan.protocol["vad"] or
            vad.get("target_fps") != 4 or vad.get("query_interval") != 4 or vad.get("fusion") != plan.protocol["vad_fast_fusion"] or
            protocols["hivau"].get("target_fps") != 4 or protocols["hivau"].get("query_interval") != 4):
        raise PredictionInputError("runtime sampling protocol differs from blind prediction plan")


def validate_artifact_runtime(artifact, runtime: PredictionRuntime) -> None:
    """Require a trained task to attest the exact Stage2 export being executed."""
    if artifact.training_identity is not None and artifact.training_identity["inherited_weights_sha256"] != runtime.inherited["stage2_export_sha256"]:
        raise PredictionInputError("selected checkpoint inherited-weight provenance differs from executed Stage2 export")


class _DefaultLoader:
    def __init__(self, runtime: PredictionRuntime, device: str): self.runtime, self.device = runtime, device
    def load(self, artifact):
        from .bridge import EvidenceSlowBridge
        from .loading import load_inherited_slow
        from .features import CELL_FEATURE_DIM
        from .model import RelationTimeEvidence
        inherited = self.runtime.inherited
        validate_artifact_runtime(artifact, self.runtime)
        slow, report = load_inherited_slow(inherited["base_directory"], inherited["stage2_export"], train_lora=artifact.group != "R0", expected_hashes=inherited["stage2_export_hashes"], device=self.device)
        raw = slow.get_base_model() if hasattr(slow, "get_base_model") else slow
        evidence = RelationTimeEvidence(CELL_FEATURE_DIM, raw.config.hidden_size)
        if artifact.group == "R0":
            for parameter in evidence.parameters(): parameter.requires_grad_(False)
        bridge = EvidenceSlowBridge(slow, evidence)
        restored = _restore_trainable(bridge, artifact)
        bridge.eval(); bridge.prediction_evidence_enabled = artifact.group != "R0"
        model_task = getattr(artifact, "task_id", None)
        if not isinstance(model_task, str) or not model_task:
            model_task = f"{artifact.group}:{getattr(artifact, 'seed', None)}:{getattr(artifact, 'evidence_enabled', None)}"
        vad, vau, residency = _build_routes(self.runtime, bridge, device=self.device, model_task=model_task)
        from .prediction_adapters import BoundReactVAUModel
        residency.stage_language()
        return BoundReactVAUModel(artifact.group, artifact.seed, artifact.evidence_enabled, bridge, vad, vau, {"slow": report, "checkpoint": restored}, residency)


class _RunnerVadAdapter:
    def predict(self, request, model, *, protocol):
        if not isinstance(protocol, dict): raise PredictionInputError("VAD protocol differs before language activation")
        primary_error = None
        try:
            model.residency.activate_language()
            return model.vad_detector.detect(request)
        except BaseException as error:
            primary_error = error
            raise
        finally:
            try: model.residency.stage_language()
            except BaseException:
                if primary_error is None: raise


class _RunnerVauAdapter:
    def generate(self, request, model, *, protocol):
        primary_error = None
        try:
            model.residency.activate_language()
            return model.hivau_inference.generate(request)
        except BaseException as error:
            primary_error = error
            raise
        finally:
            try: model.residency.stage_language()
            except BaseException:
                if primary_error is None: raise


class _FullMediaObserver:
    def __init__(self, observer, media):
        from .prediction_media import FullBlindHivauReader
        self.observer, self.media = observer, FullBlindHivauReader._physical_catalog(media)
    def observe_full_media(self, *, media_path: str, media_sha256: str, sampled_frame_times, observed_seconds: float):
        item = self.media.get((media_path, media_sha256))
        if item is None: raise PredictionInputError("HIVAU medium is absent or ambiguous in the official catalog")
        from .batches import caption_block_endpoints, pack_observation_blocks
        start, blocks = 0.0, []
        for end in caption_block_endpoints(observed_seconds):
            with self.observer._open(item) as (path, verify):
                decoder = self.observer.decoder_factory(path)
                try: blocks.append(self.observer._block(decoder, item, start, end, verify))
                finally: decoder.close()
            start = end
        return pack_observation_blocks([block.features for block in blocks], task="caption")


def _build_routes(runtime: PredictionRuntime, bridge, *, device: str, model_task: str):
    from .detector import FrozenRTDetr, InheritedSigLipAdapter
    from .detection_media import OpenCVFrames
    from .detection_provider import DetectionProtocol
    from .media_observer import BoundMedia, CausalMediaObserver
    from .numerics import deterministic_policy
    from .observation_cache import FrozenFrameCache
    from .prediction_adapters import BlindDetectionRunner, BlindPromptTokenizer, BlindVauRunner
    from .prediction_media import FullBlindDetectionReader, FullBlindHivauReader
    doc = runtime.document
    policy = deterministic_policy()
    if Path(doc["numerics"]["policy"]).read_text(encoding="utf-8").strip() != policy.identity():
        raise PredictionInputError("runtime numerical policy identity differs")
    raw, tower = bridge.raw_slow, bridge.raw_slow.get_vision_tower()
    if not getattr(tower, "is_loaded", False): raise PredictionInputError("original final Stage2 tower was not retained")
    tower.to(device=device, dtype=next(raw.get_model().mm_projector.parameters()).dtype); tower.requires_grad_(False); tower.eval()
    vision = doc["vision"]
    siglip = InheritedSigLipAdapter(tower, Path(vision["derived_snapshot"]), expected_parent_export_sha256=vision["parent_export_sha256"], expected_parent_export=vision["parent_export"], expected_raw_config_sha256=sha256_file(Path(vision["raw_snapshot"]) / "config.json"), numerical_policy_identity=policy.identity())
    evidence_enabled = bool(getattr(bridge, "prediction_evidence_enabled", True))
    detector = (FrozenRTDetr(Path(doc["detector"]["snapshot"]), device=device, score_threshold=doc["detector"]["score_threshold"], numerical_policy_identity=policy.identity())
                if evidence_enabled else None)
    media_document = json.loads(Path(doc["media"]["catalog"]).read_text(encoding="utf-8")); _no_supervision(media_document)
    if not isinstance(media_document, dict) or set(media_document) != {"schema", "media"} or media_document["schema"] != "nc_rted_blind_media_catalog/v1":
        raise PredictionInputError("official blind media catalog schema differs")
    rows = media_document["media"]
    media = {}
    if not isinstance(rows, list): raise PredictionInputError("official media catalog is invalid")
    for row in rows:
        if not isinstance(row, dict) or set(row) != set(BoundMedia.__dataclass_fields__): raise PredictionInputError("blind media catalog row differs")
        item = BoundMedia(**row); item.validate()
        if (item.dataset, item.media_key) in media: raise PredictionInputError("official media catalog has duplicate identity")
        media[(item.dataset, item.media_key)] = item
    observer = (CausalMediaObserver(detector=detector, siglip=siglip, cache=FrozenFrameCache(doc["cache"]["root"], doc["cache"]["max_bytes"]), media_catalog=media, decoder_factory=OpenCVFrames)
                if evidence_enabled else None)
    snapshot = json.loads(Path(doc["fast"]["snapshot"]).read_text(encoding="utf-8")); _no_supervision(snapshot)
    fast_rows = snapshot.get("media") if isinstance(snapshot, dict) and set(snapshot) == {"schema", "media"} and snapshot.get("schema") == "nc_rted_blind_fast_snapshot/v1" else None
    if not isinstance(fast_rows, list): raise PredictionInputError("Fast full-query snapshot is invalid")
    protocols = {key: DetectionProtocol(**value) for key, value in doc["protocols"]["vad"].items()}
    if set(protocols) != {"ucf", "xd"}: raise PredictionInputError("Fast protocols must bind UCF and XD")
    for value in protocols.values(): value.validate()
    fast_identities = set()
    for row in fast_rows:
        required = {"dataset", "media_key", "media_path", "media_sha256", "fps", "frame_count", "height", "width", "target_fps", "query_interval", "queries"}
        if not isinstance(row, dict) or set(row) != required or (row["dataset"], row["media_key"]) not in media:
            raise PredictionInputError("blind Fast snapshot row differs from official catalog")
        identity = (row["dataset"], row["media_key"])
        if identity in fast_identities: raise PredictionInputError("blind Fast snapshot has duplicate identity")
        fast_identities.add(identity)
        bound = media[identity]
        if (row["media_path"], row["media_sha256"], row["fps"], row["frame_count"], row["height"], row["width"]) != (bound.media_path, bound.media_sha256, bound.fps, bound.frame_count, bound.height, bound.width):
            raise PredictionInputError("blind Fast snapshot and media catalog differ")
        for index, query in enumerate(row["queries"]):
            if (not isinstance(query, dict) or set(query) != {"index", "frame_indices", "fast_score"} or query["index"] != index or
                    not isinstance(query["frame_indices"], list) or not query["frame_indices"] or not all(type(frame) is int and frame >= 0 for frame in query["frame_indices"]) or
                    query["frame_indices"] != sorted(set(query["frame_indices"])) or not isinstance(query["fast_score"], (int, float)) or not math.isfinite(query["fast_score"]) or not 0 <= query["fast_score"] <= 1):
                raise PredictionInputError("blind Fast query schema differs")
    frozen = type("FrozenFast", (), {"rows": {(row["dataset"], row["media_key"]): row for row in fast_rows}, "protocols": protocols, "encode": siglip})()
    tokenizer, data_args, conversation, tokenizer_image_token, image_token, image_token_index = _tokenizer(runtime)
    prompt = BlindPromptTokenizer(tokenizer, data_args, conversation.conv_templates, tokenizer_image_token,
                                  image_token, image_token_index)
    vad_config = doc["protocols"]["vad_config"]
    from eval_utils.vad.detect_utils import OnlineSmoother
    runner = BlindDetectionRunner(bridge=bridge, reader=FullBlindDetectionReader(frozen), protocol=protocols["ucf"], observation_reader=observer, prompt_tokenizer=prompt, yes_token_ids=tuple(vad_config["yes_token_ids"]), no_token_ids=tuple(vad_config["no_token_ids"]), fusion=vad_config["fusion"], fusion_alpha=vad_config["fusion_alpha"], smoother=OnlineSmoother(alpha=vad_config["online_smooth_alpha"], beta=vad_config["online_smooth_beta"]))
    original = runner.detect
    def detect(request): runner.protocol = protocols[request.dataset]; return original(request)
    runner.detect = detect
    hivau = doc["protocols"]["hivau"]
    residency = _VauPhaseResidency(bridge, device)
    fast = _ResidentBatchedFast(doc["fast"], hivau["paligemma_batch_size"], residency)
    runtime_identity = getattr(runtime, "sha256", None)
    if not isinstance(runtime_identity, str) or not runtime_identity:
        runtime_identity = hashlib.sha256(json.dumps(doc, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")).hexdigest()
    cache_binding = hashlib.sha256(json.dumps({"runtime_sha256": runtime_identity, "model_task": model_task,
                                                "evidence_enabled": evidence_enabled, "numerics": doc["numerics"]},
                                               sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    reader = FullBlindHivauReader(slow=bridge.slow, fast_detector=fast, fast_prompt="", vision_encoder=siglip,
                                  observer=None if observer is None else _FullMediaObserver(observer, media),
                                  evidence_enabled=evidence_enabled, target_fps=hivau["target_fps"], query_interval=hivau["query_interval"], media_catalog=media,
                                  material_cache_binding=cache_binding)
    generation = dict(doc["generation"]); generation["max_new_tokens"] = hivau["max_new_tokens"]
    _audit_bound_inherited_modules(runtime.inherited["external_root"],
                                   json.loads(Path(runtime.inherited["source_manifest"]).read_text(encoding="utf-8"))["files"])
    return runner, BlindVauRunner(bridge=bridge, media_reader=reader, prompt_tokenizer=prompt, tokenizer=tokenizer, generation_config=generation), residency


def _tokenizer(runtime):
    from transformers import AutoTokenizer
    from llava import conversation as conversation_lib
    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from llava.mm_utils import tokenizer_image_token
    from llava.train import train as train_module
    from .production_runtime import _configure_inherited_data_args, _configure_inherited_tokenizer
    tokenizer = AutoTokenizer.from_pretrained(runtime.inherited["tokenizer"], local_files_only=True, model_max_length=8192); _configure_inherited_tokenizer(tokenizer)
    conversation_lib.default_conversation = conversation_lib.conv_templates["qwen_2"]
    data = train_module.DataArguments(data_path="", lazy_preprocess=True, frames_upbound=64, frames_lowbound=4, local_num_frames=1, sample_type="dynamic_fps1", time_msg="short_online_v2")
    _configure_inherited_data_args(data, type("Config", (), {"mm_use_im_start_end": False, "mm_use_im_patch_token": True})())
    data.is_multimodal = True
    return tokenizer, data, conversation_lib, tokenizer_image_token, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX


def _fast_detector(config, device):
    from eval_utils.vad.eval_reactvau_detection import PaliGemmaDetector
    return PaliGemmaDetector(model_path=config["model"], lora_path=config["lora"], device=device, attn_implementation=config["attn_implementation"], image_size=config["image_size"], streamforest_weights_path=config["streamforest_weights"], vision_feature_layer=config["vision_feature_layer"])


class _BatchedFast:
    def __init__(self, detector, batch_size: int): self.detector, self.image_size, self.batch_size = detector, detector.image_size, batch_size
    def batch_score_grids(self, grids, prompt):
        return [score for start in range(0, len(grids), self.batch_size)
                for score in self.detector.batch_score_grids(grids[start:start + self.batch_size], prompt)]


class _VauPhaseResidency:
    """Stage only Slow language/evidence tensors; guarded vision remains on CUDA."""
    def __init__(self, bridge, device: str):
        self.bridge, self.device, self.detector = bridge, device, None
        raw = bridge.raw_slow
        self.vision = raw.get_vision_tower()
        visual_ids = {id(value) for value in self.vision.parameters()} | {id(value) for value in self.vision.buffers()}
        self.language_parameters = [value for value in bridge.parameters() if id(value) not in visual_ids]
        self.language_buffers = []
        buffer_ids = set()
        for module in bridge.modules():
            for name, value in module._buffers.items():
                if value is not None and id(value) not in visual_ids and id(value) not in buffer_ids:
                    self.language_buffers.append((module, name)); buffer_ids.add(id(value))
        for name, value in list(bridge.named_parameters(remove_duplicate=False)) + list(bridge.named_buffers(remove_duplicate=False)):
            if id(value) in visual_ids and "vision_tower" not in name:
                raise PredictionInputError("guarded vision tensor is shared with a language owner")
        self.state = "language_cuda_visual_cuda_fast_absent"

    def _sync_and_empty(self) -> None:
        if str(self.device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()

    @staticmethod
    def _module_is_cpu(module) -> bool:
        parameters = getattr(module, "parameters", None)
        buffers = getattr(module, "buffers", None)
        if not callable(parameters) or not callable(buffers): return True
        devices = {str(value.device) for value in parameters()} | {str(value.device) for value in buffers()}
        return not devices or devices == {"cpu"}

    def _stage_fast_cpu(self) -> None:
        if self.detector is None: return
        self.detector.model.to("cpu")
        self.detector.device = torch.device("cpu")
        if not self._module_is_cpu(self.detector.model) or self.detector.device != torch.device("cpu"):
            raise PredictionInputError("Fast detector did not reach CPU residency")

    def _move_language(self, device: str) -> None:
        parameter_originals = [(value, value.data) for value in self.language_parameters]
        buffer_originals = [(module, name, module._buffers[name]) for module, name in self.language_buffers]
        try:
            with torch.no_grad():
                for value, _ in parameter_originals: value.data = value.data.to(device)
                for module, name, value in buffer_originals: module._buffers[name] = value.to(device) if value is not None else None
            if any(str(value.device) != self.device for value in self.vision.parameters()):
                raise PredictionInputError("guarded vision tower moved during language residency transition")
        except BaseException:
            try:
                with torch.no_grad():
                    for value, original in parameter_originals: value.data = original
                    for module, name, original in buffer_originals: module._buffers[name] = original
            except BaseException:
                # A later explicit stage_language call is the only recovery path.
                self.state = "residency_faulted"
            raise

    def activate_language(self) -> None:
        if self.state == "residency_faulted":
            raise PredictionInputError("VAU residency is faulted; stage language before reentry")
        self._move_language(self.device)
        self.bridge.eval()
        self.state = "language_cuda_visual_cuda_fast_cpu" if self.detector is not None else "language_cuda_visual_cuda_fast_absent"

    def stage_language(self) -> None:
        try:
            self._move_language("cpu")
            if self.state == "residency_faulted": self._stage_fast_cpu()
            self._sync_and_empty()
        except BaseException:
            self.state = "residency_faulted"
            raise
        self.state = "language_cpu_visual_cuda_fast_cpu" if self.detector is not None else "language_cpu_visual_cuda_fast_absent"

    def score_fast(self, config: dict[str, Any], batch_size: int, grids, prompt):
        self.stage_language()
        detector = self.detector
        primary_error = None
        try:
            if detector is None:
                detector = _fast_detector(config, self.device)
                self.detector = detector
            else:
                detector.model.to(self.device)
                detector.device = torch.device(self.device)
            result = _BatchedFast(detector, batch_size).batch_score_grids(grids, prompt)
            return result
        except BaseException as error:
            primary_error = error
            raise
        finally:
            if detector is not None:
                try:
                    self._stage_fast_cpu()
                except BaseException:
                    self.state = "residency_faulted"
                    if primary_error is None: raise
            try:
                self._sync_and_empty()
            except BaseException:
                self.state = "residency_faulted"
                if primary_error is None: raise
            if primary_error is None:
                self.activate_language()
            elif self.state != "residency_faulted":
                self.state = "language_cpu_visual_cuda_fast_cpu" if detector is not None else "language_cpu_visual_cuda_fast_absent"


class _ResidentBatchedFast:
    """Lazy Fast wrapper that preserves original batch geometry and score order."""
    def __init__(self, config: dict[str, Any], batch_size: int, residency: _VauPhaseResidency):
        if type(batch_size) is not int or batch_size < 1:
            raise PredictionInputError("HIVAU Fast batch size is invalid")
        self.config, self.batch_size, self.residency = dict(config), batch_size, residency
        self.image_size = self.config["image_size"]

    def batch_score_grids(self, grids, prompt):
        return self.residency.score_fast(self.config, self.batch_size, grids, prompt)


def default_factory(plan, model, *, device: str = "cuda:0") -> dict[str, Any]:
    runtime_hash = getattr(plan, "binding_sha256", {}).get("runtime")
    if runtime_hash is None: raise PredictionInputError("prediction plan lost its runtime hash binding")
    runtime = load_prediction_runtime(plan.bindings["runtime"], expected_sha256=runtime_hash)
    _reconcile_plan(plan, runtime)
    from .numerics import configure_deterministic_algorithms
    configure_deterministic_algorithms()
    _import_bound_runtime(runtime)
    return {"loader": _DefaultLoader(runtime, device), "vad": _RunnerVadAdapter(), "vau": _RunnerVauAdapter()}
