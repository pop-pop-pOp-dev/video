"""Assembly and provenance binding for checkpoint-time mechanism diagnostics.

This module composes the existing inherited loading path without creating an
optimizer or restoring process state.  It restores only hash-validated
trainable checkpoint tensors before replaying development-only prefixes.
"""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from typing import Any, Callable, Mapping

import torch

from .detection_media import OpenCVFrames
from .detection_provider import DetectionPrefix, DetectionQuery, FrozenDetectionProvider
from .mechanism_diagnostics import (DIAGNOSTIC_SCHEMA, RUNTIME_INPUT_SCHEMA,
                                    DiagnosticError, run_development_suite,
                                    sha256_file)
from .mechanism_background import BackgroundDiagnosticError, run_global_context_diagnostics
from .mechanism_spec9 import Spec9Error, recompute_geometry_checks
from .mechanism_posteval import ColdRunMeter
from .production_runtime import load_manifest


CHECKPOINT_RUNTIME_SCHEMA = "nc_rted_mechanism_checkpoint_runtime/v1"
CHECKPOINT_REPORT_SCHEMA = "nc_rted_mechanism_checkpoint_diagnostic/v1"
DEVELOPMENT_FAST_SCHEMA = "nc_rted_development_fast/v1"
VARIANTS = ("baseline", "branch_disable", "time_permute", "relation_permute",
            "time_permute_random", "relation_permute_random")


class MechanismRuntimeError(ValueError):
    pass


@dataclass(frozen=True)
class DevelopmentBindings:
    runtime_inputs: Path
    runtime_inputs_sha256: str
    development_manifest: Path
    development_manifest_sha256: str
    fast_snapshot: Path
    fast_snapshot_sha256: str
    fast_schema: str
    fast_identity: dict[str, Any]
    sample_ids: tuple[str, ...]


@dataclass(frozen=True)
class AssembledMechanismRuntime:
    runtime: Any
    reader: Callable
    observation_reader: Callable
    tokenizer: Any
    protocols: Mapping[str, Any]
    checkpoint: Path
    checkpoint_receipt: dict
    observer: Any


def _sha_bound_json(path: str | Path, expected_sha256: str, *, name: str) -> tuple[Path, dict]:
    candidate = Path(path).resolve()
    if (not candidate.is_file() or not isinstance(expected_sha256, str) or len(expected_sha256) != 64
            or sha256_file(candidate) != expected_sha256):
        raise MechanismRuntimeError(f"{name} SHA-256 differs")
    try:
        document = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise MechanismRuntimeError(f"{name} is not valid JSON") from error
    if not isinstance(document, dict):
        raise MechanismRuntimeError(f"{name} must be a JSON object")
    return candidate, document


def _bound_reference(value: Any, *, name: str) -> tuple[Path, str]:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise MechanismRuntimeError(f"{name} binding is invalid")
    path, digest = value["path"], value["sha256"]
    if not isinstance(path, str) or not isinstance(digest, str):
        raise MechanismRuntimeError(f"{name} binding is invalid")
    candidate = Path(path).resolve()
    if not candidate.is_file() or len(digest) != 64 or sha256_file(candidate) != digest:
        raise MechanismRuntimeError(f"{name} SHA-256 differs")
    return candidate, digest


def bind_development_inputs(*, runtime_inputs: str | Path, runtime_inputs_sha256: str,
                            fast_snapshot: str | Path, fast_snapshot_sha256: str,
                            expected_fast_identity: Mapping[str, Any]) -> DevelopmentBindings:
    """Verify that the executable prefixes and Fast rows are development-only."""
    inputs_path, inputs = _sha_bound_json(runtime_inputs, runtime_inputs_sha256, name="mechanism runtime inputs")
    if inputs.get("schema") != RUNTIME_INPUT_SCHEMA or inputs.get("scope") != "development allocation only; causal prefixes only":
        raise MechanismRuntimeError("mechanism runtime inputs are not a sealed development allocation")
    manifest_ref = inputs.get("inputs", {}).get("development_manifest")
    development_path, development_sha = _bound_reference(manifest_ref, name="development manifest")
    _, development = _sha_bound_json(development_path, development_sha, name="development manifest")
    if development.get("schema") != DIAGNOSTIC_SCHEMA:
        raise MechanismRuntimeError("development allocation manifest schema differs")
    source_inputs = development.get("inputs")
    if not isinstance(source_inputs, dict) or not source_inputs:
        raise MechanismRuntimeError("development allocation manifest has no source bindings")
    for name, reference in source_inputs.items():
        _bound_reference(reference, name=f"development source {name}")
    immutable = ("sample_id", "dataset", "key", "family", "query_index", "observed_seconds", "class", "scope")
    allocation: dict[str, dict] = {}
    for row in development.get("records", []):
        if (not isinstance(row, dict) or any(name not in row for name in immutable)
                or not isinstance(row["sample_id"], str) or row["sample_id"] in allocation):
            raise MechanismRuntimeError("development allocation records are invalid")
        allocation[row["sample_id"]] = row
    if not allocation:
        raise MechanismRuntimeError("development allocation manifest has no records")
    records = inputs.get("records")
    if not isinstance(records, list) or not records:
        raise MechanismRuntimeError("mechanism runtime inputs have no development prefixes")
    selected, sample_ids = set(), set()
    for row in records:
        if not isinstance(row, dict) or row.get("scope") != "vad_causal_latest_8s":
            raise MechanismRuntimeError("mechanism prefix scope differs")
        sample_id = row.get("sample_id")
        expected = allocation.get(sample_id)
        if expected is None or sample_id in sample_ids:
            raise MechanismRuntimeError("mechanism prefix is absent from the sealed development allocation")
        if any(row.get(name) != expected[name] for name in immutable):
            raise MechanismRuntimeError("mechanism prefix immutable fields differ from the sealed development allocation")
        if not isinstance(row.get("media_sha256"), str) or len(row["media_sha256"]) != 64:
            raise MechanismRuntimeError("mechanism prefix has no bound media")
        selected.add((row["dataset"], row["key"], row["media_sha256"]))
        sample_ids.add(sample_id)
    if sample_ids != set(allocation):
        raise MechanismRuntimeError("mechanism runtime inputs omit sealed development prefixes")

    fast_path, fast = _sha_bound_json(fast_snapshot, fast_snapshot_sha256, name="development Fast snapshot")
    identity = fast.get("fast_identity")
    if (fast.get("schema") != DEVELOPMENT_FAST_SCHEMA or not isinstance(identity, dict)
            or identity != dict(expected_fast_identity)):
        raise MechanismRuntimeError("development Fast snapshot identity differs from the runtime manifest")
    rows = fast.get("media")
    if not isinstance(rows, list) or not rows:
        raise MechanismRuntimeError("development Fast snapshot has no media rows")
    bound = set()
    for row in rows:
        if not isinstance(row, dict):
            raise MechanismRuntimeError("development Fast row is invalid")
        item = (row.get("dataset"), row.get("media_key"), row.get("media_sha256"))
        if item in bound or item not in selected:
            raise MechanismRuntimeError("development Fast snapshot does not exactly bind selected development media")
        bound.add(item)
    if bound != selected:
        raise MechanismRuntimeError("development Fast snapshot omits selected development media")
    return DevelopmentBindings(inputs_path, runtime_inputs_sha256, development_path, development_sha,
                               fast_path, fast_snapshot_sha256, fast["schema"], identity,
                               tuple(sorted(sample_ids)))


class DevelopmentPrefixReader:
    """Replay a hash-bound Fast prefix without relaxing the formal reader schema.

    The production reader correctly requires a score for every query of a
    training video.  This development diagnostic has a sealed terminal query,
    so it validates the exact available causal prefix and never reads a future
    frame or accepts a fabricated later score.
    """
    def __init__(self, snapshot: str | Path, *, snapshot_sha256: str, fast_identity: Mapping[str, Any],
                 protocols: Mapping[str, Any], encode: Callable, decoder_factory: Callable = OpenCVFrames):
        if sha256_file(snapshot) != snapshot_sha256:
            raise MechanismRuntimeError("development Fast snapshot hash differs")
        try:
            document = json.loads(Path(snapshot).read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise MechanismRuntimeError("development Fast snapshot is invalid JSON") from error
        if document.get("schema") != DEVELOPMENT_FAST_SCHEMA or document.get("fast_identity") != dict(fast_identity):
            raise MechanismRuntimeError("development Fast snapshot schema or identity differs")
        rows = document.get("media")
        if not isinstance(rows, list) or not rows:
            raise MechanismRuntimeError("development Fast snapshot has no media rows")
        self.rows = {}
        for row in rows:
            if not isinstance(row, dict):
                raise MechanismRuntimeError("development Fast media row is invalid")
            identity = (row.get("dataset"), row.get("media_key"))
            if identity in self.rows:
                raise MechanismRuntimeError("development Fast snapshot has duplicate media identities")
            required = {"dataset", "media_key", "media_path", "media_sha256", "fps", "frame_count", "height", "width",
                        "target_fps", "query_interval", "max_query_index", "queries"}
            if set(row) != required:
                raise MechanismRuntimeError("development Fast row schema differs")
            self.rows[identity] = row
        self.protocols, self.encode, self.decoder_factory = dict(protocols), encode, decoder_factory
        self.snapshot_sha256 = snapshot_sha256

    def __call__(self, dataset: str, media_key: str, target_query_index: int) -> DetectionPrefix:
        if type(target_query_index) is not int or target_query_index < 0:
            raise MechanismRuntimeError("development target query is invalid")
        row = self.rows.get((dataset, media_key))
        protocol = self.protocols.get(dataset)
        if row is None or protocol is None:
            raise MechanismRuntimeError("development source is absent from the Fast snapshot")
        protocol.validate()
        path = Path(row["media_path"])
        if not path.is_file() or sha256_file(path) != row["media_sha256"]:
            raise MechanismRuntimeError("development media content differs")
        fps, total, maximum, queries = row["fps"], row["frame_count"], row["max_query_index"], row["queries"]
        if (not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0 or type(total) is not int or total <= 0
                or type(maximum) is not int or maximum < 0 or not isinstance(queries, list)
                or row["target_fps"] != 4 or row["query_interval"] != 4):
            raise MechanismRuntimeError("development Fast geometry differs")
        interval, sampled = max(1, int(fps / 4)), (total + max(1, int(fps / 4)) - 1) // max(1, int(fps / 4))
        full_count = (sampled + 3) // 4
        if maximum >= full_count or len(queries) != maximum + 1 or target_query_index > maximum:
            raise MechanismRuntimeError("development Fast prefix exceeds its sealed causal maximum")
        if any(type(row[key]) is not int or row[key] <= 0 for key in ("height", "width")):
            raise MechanismRuntimeError("development media dimensions are invalid")
        for index in range(target_query_index + 1):
            item = queries[index]
            expected = list(range(index * 4 * interval, min((index + 1) * 4 * interval, total), interval))
            if (not isinstance(item, dict) or set(item) != {"index", "frame_indices", "fast_score"}
                    or item["index"] != index or item["frame_indices"] != expected
                    or not isinstance(item["fast_score"], (int, float)) or not math.isfinite(item["fast_score"])
                    or not 0 <= item["fast_score"] <= 1):
                raise MechanismRuntimeError("development Fast query grid differs from the causal protocol")

        def iterate():
            media = path.open("rb")
            decoder = None
            try:
                fcntl.flock(media.fileno(), fcntl.LOCK_SH)
                before = os.fstat(media.fileno())
                digest = hashlib.sha256()
                for part in iter(lambda: media.read(8 << 20), b""):
                    digest.update(part)
                expected_signature = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)

                def signature():
                    value = os.fstat(media.fileno())
                    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns

                if digest.hexdigest() != row["media_sha256"] or signature() != expected_signature:
                    raise MechanismRuntimeError("development media changed before decoding")
                media.seek(0)
                decoder = self.decoder_factory(Path(f"/proc/self/fd/{media.fileno()}"))
                if (abs(decoder.fps - fps) > 1e-6 or decoder.frame_count != total or decoder.height != row["height"]
                        or decoder.width != row["width"]):
                    raise MechanismRuntimeError("development decoder geometry differs")
                for index in range(target_query_index + 1):
                    item = queries[index]
                    if signature() != expected_signature:
                        raise MechanismRuntimeError("development media changed during decoding")
                    frames = [decoder.read(frame) for frame in item["frame_indices"]]
                    if signature() != expected_signature:
                        raise MechanismRuntimeError("development media changed during decoding")
                    last = self.encode([frames[-1]])
                    if not isinstance(last, torch.Tensor) or last.shape != (1, 729, 1152):
                        raise MechanismRuntimeError("inherited development SigLIP output differs")
                    dense = None
                    if index == target_query_index and protocol.rt_anomaly:
                        padded = frames + [frames[-1]] * (4 - len(frames))
                        dense = self.encode(padded)
                        if not isinstance(dense, torch.Tensor) or dense.shape != (4, 729, 1152):
                            raise MechanismRuntimeError("inherited development RT SigLIP output differs")
                    yield DetectionQuery(index, tuple(item["frame_indices"]),
                                         tuple(frame / fps for frame in item["frame_indices"]),
                                         float(item["fast_score"]), last[0].detach(),
                                         dense.detach() if dense is not None else None)
            finally:
                if decoder is not None:
                    decoder.close()
                media.close()
        return DetectionPrefix(iterate(), row["height"], row["width"])


def _assembled_detection_provider(runtime: Any) -> FrozenDetectionProvider:
    """Obtain the existing assembled provider without recreating its model stack.

    ``ProductionRuntime`` intentionally keeps this training-only provider out of
    its public surface.  The provider closure is the sole assembled owner of the
    frozen SigLIP encoder and causal observer, so validating and reusing it here
    prevents a diagnostic-only alternate loader.
    """
    worker = getattr(runtime, "worker", None)
    provider = getattr(worker, "provider", None)
    closure = getattr(provider, "__closure__", None)
    values = [] if closure is None else [cell.cell_contents for cell in closure]
    matches = [value for value in values if isinstance(value, FrozenDetectionProvider)]
    if len(matches) != 1:
        raise MechanismRuntimeError("assembled runtime does not expose exactly one frozen detection provider")
    result = matches[0]
    if not callable(result.reader) or not callable(result.observation_reader) or not isinstance(result.protocols, dict):
        raise MechanismRuntimeError("assembled detection provider is incomplete")
    return result


def _development_observer_media(bindings: DevelopmentBindings) -> dict:
    """Build a direct-lease observer catalog from sealed development media."""
    from .media_observer import BoundMedia
    _, fast = _sha_bound_json(bindings.fast_snapshot, bindings.fast_snapshot_sha256,
                               name="development Fast snapshot")
    inputs = json.loads(bindings.runtime_inputs.read_text(encoding="utf-8"))
    sources: dict[tuple[str, str], dict] = {}
    for record in inputs["records"]:
        identity = (record["dataset"], record["key"])
        current = sources.setdefault(identity, record)
        if (current["media_path"], current["media_sha256"]) != (record["media_path"], record["media_sha256"]):
            raise MechanismRuntimeError("sealed development prefixes disagree on source media")
    media = {}
    for row in fast["media"]:
        identity = (row.get("dataset"), row.get("media_key"))
        source = sources.get(identity)
        if source is None or (row.get("media_path"), row.get("media_sha256")) != (
                source.get("media_path"), source.get("media_sha256")):
            raise MechanismRuntimeError("development Fast media differs from sealed runtime input media")
        try:
            item = BoundMedia(row["dataset"], row["media_key"], row["media_path"], row["media_sha256"],
                              row["fps"], row["frame_count"], row["height"], row["width"])
            item.validate()
        except (KeyError, RuntimeError, TypeError, ValueError) as error:
            raise MechanismRuntimeError("development Fast media geometry is invalid") from error
        if identity in media:
            raise MechanismRuntimeError("development Fast snapshot has duplicate observer media")
        media[identity] = item
    if set(media) != set(sources):
        raise MechanismRuntimeError("development Fast snapshot omits observer media")
    return media


def bind_prediction_vad_tokens(*, prediction_runtime_manifest: str | Path,
                               prediction_runtime_manifest_sha256: str, manifest,
                               load_prediction_runtime: Callable | None = None) -> tuple[tuple[int, ...], tuple[int, ...], dict]:
    """Bind Yes/No variants from the admitted blind prediction runtime."""
    if load_prediction_runtime is None:
        from .prediction_runtime import load_prediction_runtime
    runtime = load_prediction_runtime(prediction_runtime_manifest,
                                      expected_sha256=prediction_runtime_manifest_sha256)
    document, formal = getattr(runtime, "document", None), getattr(manifest, "document", None)
    if not isinstance(document, dict) or not isinstance(formal, dict):
        raise MechanismRuntimeError("prediction VAD runtime is invalid")
    inherited, prediction_inherited = formal.get("inherited"), document.get("inherited")
    if (not isinstance(inherited, dict) or not isinstance(prediction_inherited, dict)
            or prediction_inherited.get("external_root") != inherited.get("external_root")
            or prediction_inherited.get("source_manifest") != inherited.get("source_manifest")
            or prediction_inherited.get("source_manifest_sha256") != inherited.get("source_manifest_sha256")
            or prediction_inherited.get("stage2_export_hashes") != inherited.get("export_hashes")):
        raise MechanismRuntimeError("prediction VAD runtime inherited identity differs from checkpoint runtime")
    vad = document.get("protocols", {}).get("vad_config")
    if not isinstance(vad, dict):
        raise MechanismRuntimeError("prediction VAD runtime has no VAD config")
    yes, no = vad.get("yes_token_ids"), vad.get("no_token_ids")
    if (not isinstance(yes, list) or not isinstance(no, list) or not yes or not no
            or not all(type(value) is int and value >= 0 for value in yes + no)):
        raise MechanismRuntimeError("prediction VAD runtime has invalid Yes/No token variants")
    return tuple(yes), tuple(no), {"path": str(Path(prediction_runtime_manifest).resolve()),
                                   "sha256": prediction_runtime_manifest_sha256}


def _diagnostic_assembly(manifest, *, checkpoint: Path, bindings: DevelopmentBindings):
    """Compose evaluation-only runtime pieces without formal admission/training state.

    This is intentionally separate from ``production_runtime.assemble``: that
    function admits a formal training job and creates an optimizer.  A frozen
    checkpoint diagnostic only needs the exact inherited loader, evidence
    module, checkpoint trainable tensors, and causal observation path.
    """
    from .production_runtime import (_assert_inherited_modules_bound, _checkpoint_identity,
                                     _configure_inherited_data_args, _configure_inherited_tokenizer,
                                     _configure_training_memory_mode, _import_bound_inherited_runtime,
                                     _protocols, _require_loaded_final_stage2_tower, preflight)
    from .task_inputs import InheritedTaskTokenizer, TrainingCatalog
    from .train_worker import build_training_bridge
    from .prediction_runtime import _restore_trainable
    preflight(manifest)
    doc, run, inherited = manifest.document, manifest.run, manifest.document["inherited"]
    catalog = TrainingCatalog.load(doc["catalog"]["manifest_directory"], doc["catalog"]["training_annotations"],
                                   expected_provenance_sha256=doc["catalog"]["provenance_sha256"])
    identity = _checkpoint_identity(manifest, catalog)
    if checkpoint.name != "final":
        raise MechanismRuntimeError("checkpoint diagnostic requires the immutable final checkpoint")
    receipt_path = checkpoint / "manifest.json"
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise MechanismRuntimeError("final checkpoint receipt is invalid") from error
    if (receipt.get("schema") != "nc_rted_checkpoint_v2" or receipt.get("identity") != identity
            or receipt.get("final") is not True or receipt.get("completed_updates") != 1000):
        raise MechanismRuntimeError("final checkpoint identity or completion differs")
    _assert_inherited_modules_bound(inherited)
    inherited_root = str(Path(inherited["external_root"]).resolve())
    if inherited_root not in sys.path:
        sys.path.insert(0, inherited_root)
    _import_bound_inherited_runtime(inherited)
    from .detector import FrozenRTDetr, InheritedSigLipAdapter
    from .media_observer import CausalMediaObserver, lease_verified_media
    from .observation_cache import FrozenFrameCache
    from .detection_media import OpenCVFrames
    from .detection_provider import FrozenDetectionProvider
    from .numerics import configure_deterministic_algorithms
    from transformers import AutoTokenizer
    from llava.train import train as train_module
    from llava import conversation as conversation_lib
    numerical = configure_deterministic_algorithms()
    bridge, _ = build_training_bridge(inherited["base_directory"], inherited["export_directory"],
                                      export_hashes=inherited["export_hashes"], seed=run["seed"], device=run["device"])
    artifact = SimpleNamespace(group=run["group"], seed=run["seed"], checkpoint=str(checkpoint),
                               checkpoint_manifest_sha256=sha256_file(checkpoint / "manifest.json"),
                               checkpoint_state_sha256=sha256_file(checkpoint / "state.pt"))
    restore = _restore_trainable(bridge, artifact)
    raw, slow = bridge.raw_slow, bridge.slow
    _configure_training_memory_mode(raw)
    tower = slow.get_vision_tower(); _require_loaded_final_stage2_tower(tower)
    dtype = next(raw.get_model().mm_projector.parameters()).dtype
    tower.to(device=run["device"], dtype=dtype); tower.requires_grad_(False); tower.eval()
    tokenizer = AutoTokenizer.from_pretrained(inherited["tokenizer_directory"], local_files_only=True, model_max_length=8192)
    _configure_inherited_tokenizer(tokenizer)
    conversation_lib.default_conversation = conversation_lib.conv_templates["qwen_2"]
    data_args = train_module.DataArguments(data_path=doc["catalog"]["dataset_yaml"], lazy_preprocess=True,
        frames_upbound=64, frames_lowbound=4, local_num_frames=1, sample_type="dynamic_fps1", time_msg="short_online_v2")
    _configure_inherited_data_args(data_args, raw.config); data_args.image_processor = tower.image_processor; data_args.is_multimodal = True
    media = _development_observer_media(bindings)
    detector = FrozenRTDetr(Path(doc["detector"]["snapshot"]), device=run["device"], score_threshold=doc["detector"]["score_threshold"], numerical_policy_identity=numerical.identity())
    siglip = InheritedSigLipAdapter(tower, Path(doc["detector"]["final_stage2_siglip_snapshot"]),
        expected_parent_export_sha256=inherited["export_hashes"]["non_lora_trainables.bin"],
        expected_parent_export=Path(inherited["export_directory"]) / "non_lora_trainables.bin",
        expected_raw_config_sha256=sha256_file(Path(doc["detector"]["siglip_snapshot"]) / "config.json"), numerical_policy_identity=numerical.identity())
    observer = CausalMediaObserver(detector=detector, siglip=siglip,
        cache=FrozenFrameCache(doc["media"]["observation_cache_root"], doc["media"]["observation_cache_max_bytes"]),
        media_catalog=media, lease_resolver=lease_verified_media, decoder_factory=OpenCVFrames)
    protocols = _protocols(doc["fast"]["protocols"])
    class EncoderReader:
        def __call__(self, *args):
            raise MechanismRuntimeError("diagnostic reader must replace the formal Fast snapshot")

        def encode(self, frames):
            return siglip(frames)

    base_reader = EncoderReader()
    detections = FrozenDetectionProvider(slow, catalog, protocols, reader=base_reader, observation_reader=observer)
    worker = SimpleNamespace(bridge=bridge, tokenizer=InheritedTaskTokenizer.from_original(tokenizer, data_args),
                             provider=lambda _: detections)
    bridge.eval()
    return SimpleNamespace(worker=worker), receipt, restore


def assemble_checkpoint_runtime(*, runtime_manifest: str | Path, runtime_manifest_sha256: str,
                                bindings: DevelopmentBindings, checkpoint: str | Path | None = None,
                                load_runtime: Callable = load_manifest, assemble_runtime: Callable | None = None) -> AssembledMechanismRuntime:
    """Load inherited + incremental weights and bind their real replay helpers."""
    manifest = load_runtime(runtime_manifest, expected_sha256=runtime_manifest_sha256)
    document = getattr(manifest, "document", None)
    if not isinstance(document, dict) or document.get("fast", {}).get("identity") != bindings.fast_identity:
        raise MechanismRuntimeError("runtime manifest Fast identity differs from development Fast snapshot")
    checkpoint = Path(checkpoint).resolve() if checkpoint is not None else Path(manifest.run["checkpoint_root"]).resolve() / "final"
    if not checkpoint.is_dir():
        raise MechanismRuntimeError("an existing final incremental checkpoint is required")
    runtime, receipt, _ = (_diagnostic_assembly(manifest, checkpoint=checkpoint, bindings=bindings) if assemble_runtime is None
                           else assemble_runtime(manifest, checkpoint=checkpoint))
    worker = getattr(runtime, "worker", None)
    if worker is None:
        raise MechanismRuntimeError("checkpoint diagnostic requires a full inherited runtime")
    selected, updates = checkpoint, 1000
    if (not isinstance(receipt, dict) or receipt.get("final") is not True
            or receipt.get("completed_updates") != updates or selected.name != "final"):
        raise MechanismRuntimeError("incremental checkpoint receipt differs from the assembled runtime")
    bridge, tokenizer = getattr(worker, "bridge", None), getattr(worker, "tokenizer", None)
    if bridge is None or tokenizer is None:
        raise MechanismRuntimeError("assembled runtime lacks inherited Slow execution")
    bridge.eval()
    provider = _assembled_detection_provider(runtime)
    reader = DevelopmentPrefixReader(bindings.fast_snapshot, snapshot_sha256=bindings.fast_snapshot_sha256,
                                     fast_identity=bindings.fast_identity, protocols=provider.protocols,
                                     encode=provider.reader.encode)
    observer = provider.observation_reader
    if not hasattr(observer, "cache"):
        observer = getattr(observer, "__self__", None)
    if observer is not None and hasattr(observer, "capture_raw_geometry"):
        observer.capture_raw_geometry = True
    return AssembledMechanismRuntime(runtime, reader, provider.observation_reader, tokenizer,
                                     provider.protocols, selected, receipt, observer)


def _validate_suite(document: Mapping[str, Any], *, expected_sample_ids: tuple[str, ...]) -> list[dict]:
    rows = document.get("records")
    if not isinstance(rows, list) or not rows:
        raise MechanismRuntimeError("mechanism suite produced no records")
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("sample_id"), str) or row.get("intervention") not in VARIANTS:
            raise MechanismRuntimeError("mechanism suite row identity differs")
        for name in ("original_task_loss", "detection_probability", "token_delta_frobenius_norm"):
            value = row.get(name)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise MechanismRuntimeError(f"mechanism suite row has invalid {name}")
        if not 0 <= float(row["detection_probability"]) <= 1 or not isinstance(row.get("no_candidate"), bool):
            raise MechanismRuntimeError("mechanism suite row has invalid probability/no-candidate state")
        if not isinstance(row.get("fixed_greedy_text"), str) or not isinstance(row.get("fixed_greedy_token_ids"), list):
            raise MechanismRuntimeError("mechanism suite row lacks real greedy decoding")
        grouped.setdefault(row["sample_id"], []).append(dict(row))
    if set(grouped) != set(expected_sample_ids):
        raise MechanismRuntimeError("mechanism suite sample set differs from sealed development prefixes")
    annotated = []
    for sample_id, entries in grouped.items():
        names = [entry["intervention"] for entry in entries]
        if len(entries) != len(VARIANTS) or set(names) != set(VARIANTS):
            raise MechanismRuntimeError(f"mechanism suite variants differ for {sample_id}")
        for entry in entries:
            name, norm = entry["intervention"], float(entry["token_delta_frobenius_norm"])
            entry["mechanism_noop_status"] = (
                "BASELINE" if name == "baseline" else "BRANCH_DISABLED" if name == "branch_disable"
                else "EMPTY_OR_INVARIANT_EVIDENCE" if norm == 0.0 else "INTERVENTION_APPLIED")
            annotated.append(entry)
    return sorted(annotated, key=lambda row: (row["sample_id"], VARIANTS.index(row["intervention"])))


def _write_new_json(path: Path, document: Mapping[str, Any]) -> None:
    if path.exists():
        raise MechanismRuntimeError("refusing to overwrite mechanism checkpoint report")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".pending", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(document, stream, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _safe_geometry_translation(tracking: Any) -> tuple[float, float] | None:
    """Choose a nonzero, in-bounds common box translation for a raw replay."""
    boxes = [item.box_xyxy for track in getattr(tracking, "tracks", ()) for item in track.observations]
    if not boxes:
        return None
    left, top = min(box[0] for box in boxes), min(box[1] for box in boxes)
    right, bottom = min(1.0 - box[2] for box in boxes), min(1.0 - box[3] for box in boxes)
    def component(negative: float, positive: float) -> float:
        if positive > 0:
            return min(0.01, positive / 2.0)
        if negative > 0:
            return -min(0.01, negative / 2.0)
        return 0.0
    result = component(left, right), component(top, bottom)
    return result if result != (0.0, 0.0) else None


def _geometry_diagnostics(*, records: Mapping[str, Mapping[str, Any]], observations: Mapping[str, Any]) -> list[dict]:
    """Reassemble raw, process-local observer inputs through the accepted assembler."""
    from .features import assemble_relation_features
    result = []
    for sample_id, record in sorted(records.items()):
        observed = observations.get(sample_id)
        tracking, frames = getattr(observed, "tracking", None), getattr(observed, "frozen_frames", ())
        if tracking is None or not isinstance(frames, tuple) or not frames:
            result.append({"sample_id": sample_id, "status": "RAW_OBSERVER_INPUTS_NOT_CAPTURED"})
            continue
        translation = _safe_geometry_translation(tracking)
        if translation is None:
            result.append({"sample_id": sample_id, "status": "NO_IN_BOUNDS_COMMON_TRANSLATION"})
            continue
        try:
            checks = recompute_geometry_checks(assembler=assemble_relation_features, tracking=tracking, frames=frames,
                                               query_s=float(record["observed_seconds"]), translation_xy=translation,
                                               trajectory_translation_xy=translation)
        except (Spec9Error, KeyError, TypeError, ValueError) as error:
            result.append({"sample_id": sample_id, "status": "GEOMETRY_RECOMPUTATION_FAILED", "detail": str(error)})
            continue
        result.append({"sample_id": sample_id, "status": "COMPLETED", "checks": checks})
    return result


def _cold_runtime_receipt(*, assembled: AssembledMechanismRuntime, records: Mapping[str, Mapping[str, Any]],
                          generation_config: Mapping[str, Any], yes_token_ids: tuple[int, ...], no_token_ids: tuple[int, ...], parent: Path) -> dict:
    """Run one sealed prefix with an isolated empty observation cache."""
    from .mechanism_diagnostics import development_detection_task, execute_detection_prefix
    from .observation_cache import FrozenFrameCache
    observer = assembled.observer
    if observer is None or not hasattr(observer, "cache"):
        return {"status": "OBSERVER_CACHE_UNAVAILABLE"}
    sample_id = sorted(records)[0]; task = development_detection_task(records[sample_id])
    protocol = assembled.protocols.get(task.dataset)
    if protocol is None: raise MechanismRuntimeError("cold diagnostic source has no protocol")
    meter = ColdRunMeter(); original_cache, original_meter = observer.cache, getattr(observer, "meter", None)
    with tempfile.TemporaryDirectory(prefix="nc-rted-cold-", dir=parent) as directory:
        cold_cache = FrozenFrameCache(Path(directory), original_cache.max_bytes,
                                      min_free_bytes=original_cache.min_free_bytes)
        def clear():
            for path in Path(directory).glob("*.pt"): path.unlink()
        observer.cache, observer.meter = cold_cache, meter
        try:
            _, receipt = meter.measure(lambda: execute_detection_prefix(
                bridge=assembled.runtime.worker.bridge, task=task, reader=assembled.reader,
                observation_reader=assembled.observation_reader, tokenizer=assembled.tokenizer, protocol=protocol,
                name="baseline", generation_config=generation_config, yes_token_ids=yes_token_ids,
                no_token_ids=no_token_ids, meter=meter), clear_application_caches=clear)
        finally:
            observer.cache, observer.meter = original_cache, original_meter
    return {"status": "COMPLETED", "sample_id": sample_id, "receipt": receipt,
            "scope": "one lexicographically first sealed prefix; empty isolated observation cache; resident models",
            "slow_call_scope": "three model invocations: supervised CE, final-token probability, greedy generation; not generated-token call count",
            "process_scope": "complete observer processing including frozen-frame cache, SigLIP encoding, tracking, and feature assembly; excludes Fast-prefix reader replay"}


def _development_teacher_targets(*, records: Mapping[str, Mapping[str, Any]], observations: Mapping[str, Any]) -> dict:
    """Bind real development observations for a future target/reference teacher run.

    The inherited teacher writer only accepts training allocations; this receipt
    deliberately does not coerce development targets into that schema.
    """
    targets = []
    for sample_id, record in sorted(records.items()):
        value = observations.get(sample_id)
        status = "OBSERVED" if value is not None else "OBSERVATION_NOT_EXECUTED"
        targets.append({"sample_id": sample_id, "dataset": record["dataset"], "key": record["key"],
                        "query_index": record["query_index"], "observed_seconds": record["observed_seconds"],
                        "class": record["class"], "status": status})
    return {"scope": "sealed development-prefix targets only; no training-teacher substitution",
            "status": "PENDING_LEGAL_DEVELOPMENT_TARGET_TEACHER_ADAPTER", "targets": targets}


def _bound_training_teacher_compacts(teacher: Mapping[str, Any], *, diagnostic_store: str | Path | None):
    """Load only the manifest-bound immutable training compact teacher store."""
    from .teacher_store import load_teacher_store
    artifact, digest = teacher.get("artifact"), teacher.get("sha256")
    manifest = Path(artifact) if isinstance(artifact, str) else None
    if manifest is None or not manifest.is_file() or not isinstance(digest, str) or sha256_file(manifest) != digest:
        raise MechanismRuntimeError("formal teacher manifest SHA-256 differs")
    try: document = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error: raise MechanismRuntimeError("formal teacher manifest is invalid") from error
    expected = document.get("provenance", {}).get("input_sha256")
    path = Path(diagnostic_store) if diagnostic_store is not None else None
    index = None if path is None else path / "index.json"
    if (not isinstance(expected, str) or len(expected) != 64 or index is None or not index.is_file()
            or sha256_file(index) != expected):
        raise MechanismRuntimeError("diagnostic training teacher store does not match formal teacher input provenance")
    try:
        return load_teacher_store(path)
    except Exception as error:
        raise MechanismRuntimeError("bound training teacher store cannot be read") from error


def _formal_training_teacher_summary(teacher: Mapping[str, Any], runner: Callable | None) -> dict:
    """Summarize the formal artifact in its declared JSON-or-store representation."""
    artifact, digest = teacher.get("artifact"), teacher.get("sha256")
    path = Path(artifact) if isinstance(artifact, str) else None
    if runner is not None:
        value = runner(str(path)) if path is not None else runner("")
        if not isinstance(value, dict): raise MechanismRuntimeError("verified teacher summary is invalid")
        return value
    if path is None or not isinstance(digest, str) or not path.exists() or sha256_file(path / "index.json" if path.is_dir() else path) != digest:
        raise MechanismRuntimeError("formal teacher artifact SHA-256 differs")
    if path.is_file():
        try: document = json.loads(path.read_text(encoding="utf-8")); rows = document.get("rows")
        except (OSError, ValueError) as error: raise MechanismRuntimeError("formal teacher artifact JSON is invalid") from error
        if not isinstance(rows, list): raise MechanismRuntimeError("formal teacher artifact has no rows")
        from .mechanism_spec9 import teacher_summary
        value = teacher_summary(rows)
    else:
        from .mechanism_spec9 import teacher_summary_from_store
        value = teacher_summary_from_store(str(path))
    if not isinstance(value, dict): raise MechanismRuntimeError("verified teacher summary is invalid")
    return value


def _bound_development_splits(bindings: DevelopmentBindings) -> dict[tuple[str, str], Mapping[str, Any]]:
    document = json.loads(bindings.runtime_inputs.read_text(encoding="utf-8"))
    reference = document.get("inputs", {}).get("source_splits")
    path, _ = _bound_reference(reference, name="development source splits")
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise MechanismRuntimeError("development source splits are invalid") from error
    if not isinstance(rows, list):
        raise MechanismRuntimeError("development source splits must be a list")
    result = {}
    for row in rows:
        key = (row.get("dataset"), row.get("key")) if isinstance(row, Mapping) else None
        if (not isinstance(key, tuple) or not all(isinstance(value, str) and value for value in key)
                or key in result):
            raise MechanismRuntimeError("development source splits have invalid identities")
        result[key] = row
    return result


def _development_teacher_summary(*, bindings: DevelopmentBindings, teacher: Mapping[str, Any],
                                 targets: list[Any], expected_sample_ids: tuple[str, ...], diagnostic_store: str | Path | None,
                                 records: Mapping[str, Mapping[str, Any]]) -> dict:
    if len(targets) != len(expected_sample_ids):
        return {"scope": "sealed development-prefix targets only; no training-teacher substitution",
                "status": "OBSERVATIONS_INCOMPLETE", "observed_targets": len(targets),
                "expected_targets": len(expected_sample_ids)}
    from .development_teacher import summarize_development_teacher_targets
    training = _bound_training_teacher_compacts(teacher, diagnostic_store=diagnostic_store)
    result = summarize_development_teacher_targets(development_targets=tuple(targets), training_compacts=training)
    from .mechanism_spec9 import teacher_summary
    source_by_window = {f"detection:{row['dataset']}:{row['key']}:{row['query_index']}": row["class"]
                        for row in records.values()}
    class_rows = {name: [] for name in ("normal", "anomalous")}; candidate_response = {}
    for row in result["rows"]:
        source = source_by_window.get(row.get("window_id"))
        if source not in class_rows:
            raise MechanismRuntimeError("development teacher row is not bound to a sealed target")
        class_rows[source].append(row)
        if row.get("aux_valid") is True:
            candidate_response.setdefault(len(row.get("relation_ids", [])), []).append(float(row["F_quality"]))
    by_class = {name: {"targets": len(items), "aux_valid": sum(item.get("aux_valid") is True for item in items)}
                for name, items in class_rows.items()}
    return {"scope": "sealed development-prefix targets with immutable training-only references/calibration",
            "status": "COMPLETED", "summary": teacher_summary(result["rows"]), "rows": result["rows"],
            "source_class_coverage": by_class,
            "normal_tail_response": [float(row["F_quality"]) for row in class_rows["normal"] if row.get("aux_valid") is True],
            "candidate_response_dependence": {str(count): {"count": len(values), "mean_F_quality": sum(values) / len(values)}
                                             for count, values in sorted(candidate_response.items())}}


def run_checkpoint_diagnostics(output: str | Path, *, runtime_manifest: str | Path,
                               runtime_manifest_sha256: str, runtime_inputs: str | Path,
                               runtime_inputs_sha256: str, fast_snapshot: str | Path,
                               fast_snapshot_sha256: str, checkpoint: str | Path | None = None,
                               generation_config: Mapping[str, Any], load_runtime: Callable = load_manifest,
                               assemble_runtime: Callable | None = None,
                               suite_runner: Callable = run_development_suite,
                               prediction_runtime_manifest: str | Path,
                               prediction_runtime_manifest_sha256: str,
                               load_prediction_runtime: Callable | None = None,
                               teacher_summary_runner: Callable | None = None,
                               diagnostic_training_teacher_store: str | Path | None = None) -> dict:
    """Execute the complete frozen checkpoint diagnostic and publish one report."""
    output = Path(output).resolve()
    if output.exists():
        raise MechanismRuntimeError("refusing to overwrite mechanism checkpoint report")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path, manifest_document = _sha_bound_json(runtime_manifest, runtime_manifest_sha256,
                                                        name="mechanism runtime manifest")
    manifest = load_runtime(manifest_path, expected_sha256=runtime_manifest_sha256)
    fast_identity = manifest_document.get("fast", {}).get("identity")
    if not isinstance(fast_identity, dict) or not fast_identity:
        raise MechanismRuntimeError("runtime manifest has no Fast identity")
    yes_token_ids, no_token_ids, prediction_binding = bind_prediction_vad_tokens(
        prediction_runtime_manifest=prediction_runtime_manifest,
        prediction_runtime_manifest_sha256=prediction_runtime_manifest_sha256, manifest=manifest,
        load_prediction_runtime=load_prediction_runtime)
    bindings = bind_development_inputs(runtime_inputs=runtime_inputs, runtime_inputs_sha256=runtime_inputs_sha256,
                                       fast_snapshot=fast_snapshot, fast_snapshot_sha256=fast_snapshot_sha256,
                                       expected_fast_identity=fast_identity)
    assembled = assemble_checkpoint_runtime(runtime_manifest=manifest_path,
                                            runtime_manifest_sha256=runtime_manifest_sha256,
                                            bindings=bindings, checkpoint=checkpoint,
                                            load_runtime=lambda *_args, **_kwargs: manifest, assemble_runtime=assemble_runtime)
    sealed_records = json.loads(bindings.runtime_inputs.read_text(encoding="utf-8"))["records"]
    record_by_sample = {record["sample_id"]: record for record in sealed_records}
    observed: dict[str, Any] = {}; geometry: list[dict] = []; development_targets: list[Any] = []
    development_splits = None

    def capture_observation(dataset: str, key: str, endpoint: float):
        nonlocal development_splits
        value = assembled.observation_reader(dataset, key, endpoint)
        matches = [record["sample_id"] for record in sealed_records
                   if (record["dataset"], record["key"], record["observed_seconds"]) == (dataset, key, endpoint)]
        if len(matches) != 1:
            raise MechanismRuntimeError("observed feature cannot be bound to exactly one sealed prefix")
        sample_id = matches[0]
        if sample_id not in observed:
            geometry.extend(_geometry_diagnostics(records={sample_id: record_by_sample[sample_id]}, observations={sample_id: value}))
            try:
                from .development_teacher_targets import development_feature_assembly_to_teacher_window
                if development_splits is None:
                    development_splits = _bound_development_splits(bindings)
                split = development_splits.get((dataset, key))
                if split is None:
                    raise MechanismRuntimeError("sealed development source split is absent")
                development_targets.append(development_feature_assembly_to_teacher_window(
                    value.features, value.relation_class_pairs, development_prefix=record_by_sample[sample_id],
                    development_source_split=split))
            except MechanismRuntimeError:
                raise
            except Exception as error:
                raise MechanismRuntimeError("development teacher target conversion failed") from error
            observed[sample_id] = replace(value, tracking=None, frozen_frames=())
        return value

    with tempfile.TemporaryDirectory(prefix="nc-rted-mechanism-", dir=output.parent) as temporary:
        suite_path = Path(temporary) / "suite.json"
        suite = suite_runner(suite_path, runtime_inputs=bindings.runtime_inputs, bridge=assembled.runtime.worker.bridge,
                             reader=assembled.reader, observation_reader=capture_observation,
                             tokenizer=assembled.tokenizer, protocols=assembled.protocols,
                             generation_config=dict(generation_config), yes_token_ids=tuple(yes_token_ids),
                             no_token_ids=tuple(no_token_ids))
    if not isinstance(suite, dict):
        raise MechanismRuntimeError("mechanism suite did not return a document")
    for sample_id in sorted(set(record_by_sample) - {row["sample_id"] for row in geometry}):
        geometry.append({"sample_id": sample_id, "status": "RAW_OBSERVER_INPUTS_NOT_CAPTURED"})
    rows = _validate_suite(suite, expected_sample_ids=bindings.sample_ids)
    baseline = {row["sample_id"]: row for row in rows if row["intervention"] == "baseline"}
    try:
        global_context = run_global_context_diagnostics(
            records=record_by_sample, observations=observed, baseline_rows=baseline,
            bridge=assembled.runtime.worker.bridge, reader=assembled.reader, tokenizer=assembled.tokenizer,
            protocols=assembled.protocols, generation_config=dict(generation_config),
            yes_token_ids=yes_token_ids, no_token_ids=no_token_ids)
    except BackgroundDiagnosticError as error:
        raise MechanismRuntimeError(str(error)) from error
    cold = _cold_runtime_receipt(assembled=assembled, records=record_by_sample,
                                 generation_config=dict(generation_config), yes_token_ids=yes_token_ids,
                                 no_token_ids=no_token_ids, parent=output.parent)
    teacher = manifest_document.get("teacher")
    if (not isinstance(teacher, dict) or not isinstance(teacher.get("artifact"), str)
            or not isinstance(teacher.get("sha256"), str) or len(teacher["sha256"]) != 64):
        raise MechanismRuntimeError("runtime manifest has no bound teacher store")
    teacher_summary = _formal_training_teacher_summary(teacher, teacher_summary_runner)
    development_teacher = _development_teacher_summary(bindings=bindings, teacher=teacher,
                                                         targets=development_targets,
                                                         expected_sample_ids=bindings.sample_ids,
                                                         diagnostic_store=diagnostic_training_teacher_store,
                                                         records=record_by_sample)
    result = {
        "schema": CHECKPOINT_REPORT_SCHEMA,
        "checkpoint_runtime": {"schema": CHECKPOINT_RUNTIME_SCHEMA,
                               "runtime_manifest": {"path": str(manifest_path), "sha256": runtime_manifest_sha256},
                               "incremental_checkpoint": str(assembled.checkpoint),
                               "checkpoint_receipt": assembled.checkpoint_receipt},
        "model_identity": {
            "group": manifest_document["run"]["group"], "seed": manifest_document["run"]["seed"],
            "code_sha256": manifest_document["hashes"]["code_sha256"],
            "runtime_sha256": manifest_document["hashes"]["runtime_sha256"],
            "inherited_weights_sha256": manifest_document["hashes"]["inherited_weights_sha256"],
            "detector_snapshot_sha256": manifest_document["detector"]["snapshot_sha256"],
            "siglip_snapshot_sha256": manifest_document["detector"]["siglip_snapshot_sha256"],
            "final_stage2_siglip_snapshot_sha256": manifest_document["detector"]["final_stage2_siglip_snapshot_sha256"]},
        "development_bindings": {
            "runtime_inputs": {"path": str(bindings.runtime_inputs), "sha256": bindings.runtime_inputs_sha256},
            "development_manifest": {"path": str(bindings.development_manifest), "sha256": bindings.development_manifest_sha256},
            "fast_snapshot": {"path": str(bindings.fast_snapshot), "sha256": bindings.fast_snapshot_sha256},
            "fast_identity": bindings.fast_identity},
        "generation_config": dict(generation_config),
        "prediction_vad_runtime": prediction_binding,
        "training_teacher_artifact_not_development_prefixes": {
            "artifact": {"path": teacher["artifact"], "sha256": teacher["sha256"]},
            "summary": teacher_summary,
            "scope": "training teacher artifact; does not substitute for fixed development-prefix teacher diagnostics",
        },
        "development_prefix_teacher_diagnostic": development_teacher,
        "records": rows,
        "background_perturbation": global_context,
        "geometry_invariance": geometry,
        "cold_runtime_cost": cold,
    }
    _write_new_json(output, result)
    return result
