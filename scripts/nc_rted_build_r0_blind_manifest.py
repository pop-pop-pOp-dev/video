#!/usr/bin/env python3
"""Assemble a hash-bound R0 blind-prediction candidate without launching CUDA.

The builder only binds label-free inputs and structurally preflights the
runtime on CPU.  Formal admission remains an explicit separate requirement.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nc_rted.prediction_inputs import IMPLEMENTATION_SCHEMA, canonical_json, sha256_file
from nc_rted.prediction_runtime import load_prediction_runtime
from nc_rted.storage_lock import allocation_lock, ensure_directory


VAD_COUNTS = {"ucf": 251, "xd": 800}
VAU_COUNT = 3339
VAU_MEDIA_COUNT = 1369
MEDIA_SCHEMA = "nc_rted_blind_media_catalog/v1"
FAST_SCHEMA = "nc_rted_blind_fast_snapshot/v1"
VAU_BINDING_SCHEMA = "nc_rted_vau_question_media_bindings/v1"
EXPORT_FILES = {"config.json", "adapter_config.json", "adapter_model.safetensors", "non_lora_trainables.bin"}
MEDIA_FIELDS = {"dataset", "media_key", "media_path", "media_sha256", "fps", "frame_count", "height", "width", "request_index"}
RESERVE = 20 * 1024**3


class BuildError(ValueError):
    pass


def _sha(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _read(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise BuildError(f"invalid JSON: {path}") from error


def _tree(path: Path) -> str:
    """Hash a real, nonempty artifact tree and reject symlinks."""
    digest, files = hashlib.sha256(), 0
    if not path.is_dir() or path.is_symlink():
        raise BuildError(f"required directory is absent or a symlink: {path}")
    for child in sorted(path.rglob("*")):
        if child.is_symlink():
            raise BuildError(f"bound directory contains symlink: {child}")
        if child.is_file():
            files += 1
            digest.update(str(child.relative_to(path)).encode("utf-8"))
            digest.update(b"\0")
            digest.update(sha256_file(child).encode("ascii"))
            digest.update(b"\n")
    if files == 0:
        raise BuildError(f"required directory is empty: {path}")
    return digest.hexdigest()


def _file(path: Path, name: str) -> tuple[str, str]:
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise BuildError(f"{name} is absent or not an absolute regular file")
    return str(path), sha256_file(path)


def _directory(path: Path, name: str) -> tuple[str, str]:
    if not path.is_absolute():
        raise BuildError(f"{name} is not absolute")
    return str(path), _tree(path)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ensure_durable_directory(path: Path) -> None:
    """Admit missing-directory metadata before creation, then durably sync it."""
    missing: list[Path] = []
    current = path
    while not current.exists():
        missing.append(current)
        if current.parent == current:
            raise BuildError(f"no existing output ancestor: {path}")
        current = current.parent
    if not current.is_dir() or current.is_symlink():
        raise BuildError(f"output ancestor is not a regular directory: {current}")
    try:
        ensure_directory(path, RESERVE)
    except OSError as error:
        raise BuildError("bound output directory creation would violate the 20 GiB reserve") from error
    for directory in reversed(missing):
        if not directory.is_dir() or directory.is_symlink():
            raise BuildError(f"output path is not a regular directory: {directory}")
        _fsync_directory(directory)
        _fsync_directory(directory.parent)


def _write(path: Path, value: object) -> str:
    """Atomically publish one immutable bound JSON artifact with disk reserve."""
    if not path.is_absolute():
        raise BuildError("bound output path is not absolute")
    if os.path.lexists(path):
        raise BuildError(f"bound output already exists: {path}")
    encoded = canonical_json(value) + b"\n"
    try:
        with allocation_lock(path.parent):
            if os.path.lexists(path):
                raise BuildError(f"bound output already exists: {path}")
            _ensure_durable_directory(path.parent)
            if shutil.disk_usage(path.parent).free < RESERVE + len(encoded) + 8192:
                raise BuildError("bound output publication would violate the 20 GiB reserve")
            descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".pending", dir=path.parent)
            try:
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(encoded)
                    stream.flush()
                    os.fsync(stream.fileno())
                try:
                    os.link(temporary, path)
                    _fsync_directory(path.parent)
                except FileExistsError as error:
                    raise BuildError(f"bound output already exists: {path}") from error
                except OSError as error:
                    raise BuildError(f"bound output publication failed: {path}") from error
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
    except BuildError:
        raise
    except OSError as error:
        raise BuildError(f"bound output allocation lock failed: {path}") from error
    return hashlib.sha256(encoded).hexdigest()


def _file_after_write(path: Path, value: object) -> tuple[str, str]:
    return str(path), _write(path, value)


def _clean_keys(value: object) -> None:
    forbidden = {"label", "labels", "answer", "answers", "target", "targets", "reference", "references", "metric", "metrics",
                 "teacher", "teachers", "annotation", "annotations", "ground_truth", "groundtruth"}
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str) or key.lower().replace("-", "_").replace(" ", "_") in forbidden:
                raise BuildError("blind builder input contains supervision metadata")
            _clean_keys(child)
    elif isinstance(value, list):
        for child in value:
            _clean_keys(child)


def _catalog(path: Path) -> list[dict[str, Any]]:
    value = _read(path)
    _clean_keys(value)
    if not isinstance(value, dict) or set(value) != {"schema", "media"} or value["schema"] != MEDIA_SCHEMA or not isinstance(value["media"], list):
        raise BuildError("media catalog schema differs")
    identities: set[tuple[str, str]] = set()
    rows: list[dict[str, Any]] = []
    for row in value["media"]:
        if not isinstance(row, dict) or set(row) != MEDIA_FIELDS:
            raise BuildError("media catalog row schema differs")
        dataset, key = row["dataset"], row["media_key"]
        if (not isinstance(dataset, str) or not dataset or not isinstance(key, str) or not key or not isinstance(row["media_path"], str) or
                not _sha(row["media_sha256"]) or not isinstance(row["fps"], (int, float)) or isinstance(row["fps"], bool) or
                not math.isfinite(float(row["fps"])) or float(row["fps"]) <= 0 or type(row["frame_count"]) is not int or row["frame_count"] < 1 or
                type(row["height"]) is not int or row["height"] < 1 or type(row["width"]) is not int or row["width"] < 1 or
                (row["request_index"] is not None and (type(row["request_index"]) is not int or row["request_index"] < 0))):
            raise BuildError("media catalog row has invalid identity or geometry")
        identity = (dataset, key)
        if identity in identities:
            raise BuildError("media catalog has duplicate identity")
        identities.add(identity)
        _, digest = _file(Path(row["media_path"]), "catalog media")
        if digest != row["media_sha256"]:
            raise BuildError("catalog media differs from its hash")
        rows.append(row)
    return rows


def _vau_rows(bindings_path: Path, roster_path: Path, catalog: list[dict[str, Any]], *, expected_count: int = VAU_COUNT, expected_media_count: int = VAU_MEDIA_COUNT) -> list[dict[str, str]]:
    bindings_doc, roster = _read(bindings_path), _read(roster_path)
    _clean_keys(bindings_doc)
    _clean_keys(roster)
    if (not isinstance(bindings_doc, dict) or set(bindings_doc) != {"schema", "bindings"} or bindings_doc["schema"] != VAU_BINDING_SCHEMA or
            not isinstance(bindings_doc["bindings"], list) or not isinstance(roster, list)):
        raise BuildError("VAU bindings or official question roster schema differs")
    by_id: dict[str, dict[str, Any]] = {}
    for row in bindings_doc["bindings"]:
        if not isinstance(row, dict) or set(row) != {"id", "video", "media_path", "media_sha256"} or type(row["id"]) is not int or row["id"] < 0 or not isinstance(row["video"], str) or not row["video"] or not isinstance(row["media_path"], str) or not _sha(row["media_sha256"]):
            raise BuildError("VAU binding row differs from the materializer contract")
        identity = str(row["id"])
        if identity in by_id:
            raise BuildError("duplicate VAU media binding")
        by_id[identity] = row
    output: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in roster:
        if not isinstance(row, dict) or set(row) != {"id", "prompt", "task", "type", "video"} or type(row["id"]) is not int or row["id"] < 0 or not isinstance(row["prompt"], str) or not row["prompt"] or not isinstance(row["task"], str) or not row["task"] or not isinstance(row["type"], str) or not row["type"] or not isinstance(row["video"], str) or not row["video"]:
            raise BuildError("official VAU question roster differs")
        identity = str(row["id"])
        if identity in seen or identity not in by_id:
            raise BuildError("VAU question/binding identity differs")
        bound = by_id[identity]
        if row["video"] != bound["video"]:
            raise BuildError("VAU question/binding source video differs")
        seen.add(identity)
        output.append({"id": identity, "media_path": bound["media_path"], "media_sha256": bound["media_sha256"], "question": row["prompt"]})
    if len(output) != expected_count or seen != set(by_id):
        raise BuildError(f"VAU denominator or binding coverage differs from {expected_count}")
    geometry = {(row["media_path"], row["media_sha256"]): row for row in catalog}
    if len(geometry) != expected_media_count or len(catalog) != expected_media_count:
        raise BuildError(f"VAU catalog denominator differs from {expected_media_count} materialized media")
    bound_media = {(row["media_path"], row["media_sha256"]) for row in output}
    if bound_media != set(geometry):
        raise BuildError("VAU bindings do not cover exactly the materialized media catalog")
    return output


def _fast_snapshot(path: Path, vad: list[dict[str, Any]]) -> None:
    document = _read(path)
    _clean_keys(document)
    if not isinstance(document, dict) or set(document) != {"schema", "media"} or document["schema"] != FAST_SCHEMA or not isinstance(document["media"], list):
        raise BuildError("Fast snapshot schema differs")
    expected = {(row["dataset"], row["media_key"]): row for row in vad}
    observed: set[tuple[str, str]] = set()
    fields = MEDIA_FIELDS - {"request_index"} | {"target_fps", "query_interval", "queries"}
    for row in document["media"]:
        if not isinstance(row, dict) or set(row) != fields:
            raise BuildError("Fast snapshot row schema differs")
        identity = (row["dataset"], row["media_key"])
        bound = expected.get(identity)
        if identity in observed or bound is None:
            raise BuildError("Fast snapshot identity differs from VAD catalog")
        observed.add(identity)
        if any(row[name] != bound[name] for name in ("media_path", "media_sha256", "fps", "frame_count", "height", "width")):
            raise BuildError("Fast snapshot media binding differs from VAD catalog")
        if row["target_fps"] != 4 or row["query_interval"] != 4 or not isinstance(row["queries"], list):
            raise BuildError("Fast snapshot uses a different inherited sampling schedule")
        interval = max(1, int(float(row["fps"]) / 4))
        expected_count = ((row["frame_count"] + interval - 1) // interval + 3) // 4
        if len(row["queries"]) != expected_count:
            raise BuildError("Fast snapshot does not cover every VAD query")
        for index, query in enumerate(row["queries"]):
            frames = list(range(index * 4 * interval, min((index + 1) * 4 * interval, row["frame_count"]), interval))
            if (not isinstance(query, dict) or set(query) != {"index", "frame_indices", "fast_score"} or query["index"] != index or query["frame_indices"] != frames or not isinstance(query["fast_score"], (int, float)) or isinstance(query["fast_score"], bool) or not math.isfinite(float(query["fast_score"])) or not 0 <= float(query["fast_score"]) <= 1):
                raise BuildError("Fast snapshot query differs from the inherited schedule")
    if observed != set(expected):
        raise BuildError("Fast snapshot does not cover the exact VAD catalog")


def _fast_config(path: Path) -> dict[str, Any]:
    config = _read(path)
    _clean_keys(config)
    required = {"model", "lora", "streamforest_weights", "attn_implementation", "image_size", "vision_feature_layer", "protocols", "generation", "cache"}
    if not isinstance(config, dict) or set(config) != required:
        raise BuildError("Fast config fields differ")
    if (not isinstance(config["model"], str) or not isinstance(config["streamforest_weights"], str) or (config["lora"] is not None and not isinstance(config["lora"], str)) or not isinstance(config["attn_implementation"], str) or not config["attn_implementation"] or type(config["image_size"]) is not int or config["image_size"] < 1 or type(config["vision_feature_layer"]) is not int):
        raise BuildError("Fast model binding fields are invalid")
    protocols = config["protocols"]
    if not isinstance(protocols, dict) or set(protocols) != {"vad", "vad_config", "hivau"}:
        raise BuildError("Fast protocol fields differ")
    vad_fields = {"question_template", "prompt_style", "time_message_style", "memory_enhancement", "rt_anomaly", "trigger_threshold", "pool_threshold", "scoring"}
    if not isinstance(protocols["vad"], dict) or set(protocols["vad"]) != set(VAD_COUNTS):
        raise BuildError("Fast VAD protocol must bind UCF and XD")
    for value in protocols["vad"].values():
        if not isinstance(value, dict) or set(value) != vad_fields or not isinstance(value["question_template"], str) or not value["question_template"] or value["prompt_style"] not in {"default", "skeptical", "neutral", "hivau"} or value["time_message_style"] not in {"short_online", "short_online_v2", "simple", "none"} or type(value["memory_enhancement"]) is not bool or type(value["rt_anomaly"]) is not bool or value["scoring"] != "yesno" or any(not isinstance(value[key], (int, float)) or isinstance(value[key], bool) or not math.isfinite(float(value[key])) or not 0 <= float(value[key]) <= 1 for key in ("trigger_threshold", "pool_threshold")):
            raise BuildError("Fast VAD protocol differs from the inherited contract")
    vad_config = protocols["vad_config"]
    expected_vad_config = {"yes_token_ids", "no_token_ids", "fusion", "fusion_alpha", "online_smooth_alpha", "online_smooth_beta", "target_fps", "query_interval", "batch_size"}
    if not isinstance(vad_config, dict) or set(vad_config) != expected_vad_config or vad_config["fusion"] not in {"replace", "weighted", "adaptive"} or any(not isinstance(vad_config[key], list) or not vad_config[key] or any(type(token) is not int or token < 0 for token in vad_config[key]) for key in ("yes_token_ids", "no_token_ids")) or any(not isinstance(vad_config[key], (int, float)) or isinstance(vad_config[key], bool) or not math.isfinite(float(vad_config[key])) or not 0 <= float(vad_config[key]) <= 1 for key in ("fusion_alpha", "online_smooth_alpha", "online_smooth_beta")) or vad_config["target_fps"] != 4 or vad_config["query_interval"] != 4 or type(vad_config["batch_size"]) is not int or vad_config["batch_size"] < 1:
        raise BuildError("Fast VAD evaluator configuration differs")
    hivau = protocols["hivau"]
    hivau_fields = {"target_fps", "query_interval", "paligemma_batch_size", "max_new_tokens", "task", "fast_prompt_context"}
    if not isinstance(hivau, dict) or set(hivau) != hivau_fields or any(type(hivau[key]) is not int or hivau[key] < 1 for key in ("target_fps", "query_interval", "paligemma_batch_size", "max_new_tokens")) or hivau["target_fps"] != 4 or hivau["query_interval"] != 4 or not isinstance(hivau["task"], str) or not hivau["task"] or hivau["fast_prompt_context"] != "none":
        raise BuildError("Fast HIVAU protocol differs from the blind contract")
    generation = config["generation"]
    if not isinstance(generation, dict) or generation.get("do_sample") is not False or generation.get("num_beams", 1) != 1 or generation.get("num_return_sequences", 1) != 1:
        raise BuildError("generation configuration is not greedy")
    forbidden_generation = {"penalty_alpha", "top_k", "top_p", "typical_p", "constraints", "force_words_ids", "prefix_allowed_tokens_fn", "assistant_model"}
    if forbidden_generation.intersection(generation):
        raise BuildError("generation configuration enables an unsupported decoder")
    cache = config["cache"]
    if not isinstance(cache, dict) or set(cache) != {"root", "max_bytes"} or not isinstance(cache["root"], str) or not Path(cache["root"]).is_absolute() or type(cache["max_bytes"]) is not int or cache["max_bytes"] < 1:
        raise BuildError("cache configuration differs")
    return config


def _caption_bindings(path: Path) -> dict[str, Any]:
    caption = _read(path)
    _clean_keys(caption)
    if not isinstance(caption, dict) or caption.get("schema") != "nc_rted_production_runtime/v1":
        raise BuildError("caption runtime schema differs")
    inherited, detector = caption.get("inherited"), caption.get("detector")
    required_inherited = {"external_root", "base_directory", "base_directory_sha256", "export_directory", "export_hashes", "tokenizer_directory", "tokenizer_sha256", "source_manifest", "source_manifest_sha256"}
    required_detector = {"final_stage2_siglip_snapshot", "final_stage2_siglip_snapshot_sha256", "siglip_snapshot", "siglip_snapshot_sha256", "snapshot", "snapshot_sha256", "score_threshold"}
    if not isinstance(inherited, dict) or not isinstance(detector, dict) or not required_inherited.issubset(inherited) or not required_detector.issubset(detector) or not isinstance(inherited["export_hashes"], dict) or set(inherited["export_hashes"]) != EXPORT_FILES or not all(_sha(value) for value in inherited["export_hashes"].values()):
        raise BuildError("caption runtime does not bind every original artifact")
    base, base_hash = _directory(Path(inherited["base_directory"]), "base model")
    if base_hash != inherited["base_directory_sha256"]:
        raise BuildError("base model differs from caption runtime hash")
    export_path, export_hash = _directory(Path(inherited["export_directory"]), "original final Stage2 export")
    for name, digest in inherited["export_hashes"].items():
        _, actual = _file(Path(export_path) / name, "original final Stage2 export file")
        if actual != digest:
            raise BuildError("original final Stage2 export file differs")
    source, source_hash = _file(Path(inherited["source_manifest"]), "ReactVAU source manifest")
    if source_hash != inherited["source_manifest_sha256"]:
        raise BuildError("ReactVAU source manifest differs from caption runtime hash")
    tokenizer, tokenizer_hash = _directory(Path(inherited["tokenizer_directory"]), "tokenizer")
    if tokenizer_hash != inherited["tokenizer_sha256"]:
        raise BuildError("tokenizer differs from caption runtime hash")
    derived, derived_hash = _directory(Path(detector["final_stage2_siglip_snapshot"]), "derived vision")
    raw, raw_hash = _directory(Path(detector["siglip_snapshot"]), "raw vision")
    detector_path, detector_hash = _directory(Path(detector["snapshot"]), "detector")
    for name, actual, expected in (("derived vision", derived_hash, detector["final_stage2_siglip_snapshot_sha256"]), ("raw vision", raw_hash, detector["siglip_snapshot_sha256"]), ("detector", detector_hash, detector["snapshot_sha256"])):
        if actual != expected:
            raise BuildError(f"{name} differs from caption runtime hash")
    root = Path(inherited["external_root"])
    if not root.is_absolute() or not root.is_dir() or root.is_symlink():
        raise BuildError("ReactVAU source root is unavailable")
    if not isinstance(detector["score_threshold"], (int, float)) or isinstance(detector["score_threshold"], bool) or not math.isfinite(float(detector["score_threshold"])) or not 0 <= float(detector["score_threshold"]) <= 1:
        raise BuildError("detector score threshold is invalid")
    return {"external_root": str(root), "base": (base, base_hash), "export": (export_path, export_hash), "export_hashes": inherited["export_hashes"], "source": (source, source_hash), "tokenizer": (tokenizer, tokenizer_hash), "derived": (derived, derived_hash), "raw": (raw, raw_hash), "detector": (detector_path, detector_hash), "score_threshold": detector["score_threshold"]}


def _prediction_source_manifest(output: Path, external_root: str) -> tuple[str, str]:
    """Bind every executable Python source without changing caption's manifest."""
    root = Path(external_root)
    files: dict[str, str] = {}
    for candidate in sorted(root.rglob("*.py")):
        if candidate.is_symlink() or not candidate.is_file():
            raise BuildError("ReactVAU prediction source tree contains an invalid Python path")
        files[str(candidate.relative_to(root))] = sha256_file(candidate)
    if not files:
        raise BuildError("ReactVAU prediction source tree has no executable Python files")
    return _file_after_write(output / "prediction_source_manifest.json", {
        "schema": "nc_rted_prediction_source_manifest/v1",
        "files": files,
    })


def _implementation_manifest(output: Path) -> tuple[str, str]:
    from nc_rted.prediction_inputs import _IMPLEMENTATION_FILES
    files = {relative: sha256_file(ROOT / relative) for relative in _IMPLEMENTATION_FILES}
    return _file_after_write(output / "implementation_manifest.json", {"schema": IMPLEMENTATION_SCHEMA, "root": str(ROOT), "files": files})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--caption-runtime", type=Path, required=True)
    parser.add_argument("--vad-catalog", type=Path, required=True)
    parser.add_argument("--fast-snapshot", type=Path, required=True)
    parser.add_argument("--vau-bindings", type=Path, required=True)
    parser.add_argument("--vau-questions", type=Path, required=True)
    parser.add_argument("--vau-catalog", type=Path, required=True)
    parser.add_argument("--fast-config", type=Path, required=True)
    parser.add_argument("--numerics-policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    report: dict[str, Any] = {"schema": "nc_rted_r0_blind_manifest_build/v2", "status": "BLOCKED", "gpu_launched": False, "missing_dependencies": []}
    required = {"caption_runtime": args.caption_runtime, "vad_catalog": args.vad_catalog, "fast_snapshot": args.fast_snapshot, "vau_bindings": args.vau_bindings, "vau_questions": args.vau_questions, "vau_catalog": args.vau_catalog, "fast_config": args.fast_config, "numerics_policy": args.numerics_policy}
    missing = [f"{name}: {path}" for name, path in required.items() if not path.is_absolute() or not path.is_file() or path.is_symlink()]
    if not args.output.is_absolute(): missing.append("output must be absolute")
    if not args.output_root.is_absolute(): missing.append("output_root must be absolute")
    if missing:
        report["missing_dependencies"] = missing
        if args.output.is_absolute(): _write(args.output, report)
        print(json.dumps({"status": report["status"], "output": str(args.output), "missing_dependencies": missing}, sort_keys=True))
        return 0
    try:
        caption = _caption_bindings(args.caption_runtime)
        vad = _catalog(args.vad_catalog)
        counts = {dataset: sum(row["dataset"] == dataset for row in vad) for dataset in VAD_COUNTS}
        if counts != VAD_COUNTS or any(row["dataset"] not in VAD_COUNTS for row in vad): raise BuildError("VAD catalog denominator differs from 251/800")
        vau_catalog = _catalog(args.vau_catalog)
        vau = _vau_rows(args.vau_bindings, args.vau_questions, vau_catalog)
        _fast_snapshot(args.fast_snapshot, vad)
        config = _fast_config(args.fast_config)
        fast_model, fast_model_hash = _directory(Path(config["model"]), "Fast model")
        weights, weights_hash = _file(Path(config["streamforest_weights"]), "Fast weights")
        lora, lora_hash = (None, None) if config["lora"] is None else _directory(Path(config["lora"]), "Fast LoRA")
        policy_path, policy_hash = _file(args.numerics_policy, "numerics policy")
        if not Path(policy_path).read_text(encoding="utf-8").strip(): raise BuildError("numerics policy is empty")
        merged_media = vad + vau_catalog
        if len(merged_media) != len({(row["dataset"], row["media_key"]) for row in merged_media}): raise BuildError("merged blind media catalog has duplicate identities")
        output_dir = args.output.parent
        prediction_source_path, prediction_source_hash = _prediction_source_manifest(output_dir, caption["external_root"])
        catalog_path, catalog_hash = _file_after_write(output_dir / "blind_media_catalog.json", {"schema": MEDIA_SCHEMA, "media": merged_media})
        identity_path, identity_hash = _file_after_write(output_dir / "blind_identity_manifest.json", {"vad": [{"dataset": row["dataset"], "id": row["media_key"], "media_path": row["media_path"], "media_sha256": row["media_sha256"]} for row in vad], "vau": vau})
        fast_path, fast_hash = _file(args.fast_snapshot, "Fast snapshot")
        runtime = {"schema": "nc_rted_prediction_runtime/v2", "inherited": {"external_root": caption["external_root"], "base_directory": caption["base"][0], "base_directory_sha256": caption["base"][1], "stage2_export": caption["export"][0], "stage2_export_sha256": caption["export"][1], "stage2_export_hashes": caption["export_hashes"], "tokenizer": caption["tokenizer"][0], "tokenizer_sha256": caption["tokenizer"][1], "source_manifest": prediction_source_path, "source_manifest_sha256": prediction_source_hash}, "fast": {"snapshot": fast_path, "snapshot_sha256": fast_hash, "model": fast_model, "model_sha256": fast_model_hash, "lora": lora, "lora_sha256": lora_hash, "streamforest_weights": weights, "streamforest_weights_sha256": weights_hash, "attn_implementation": config["attn_implementation"], "image_size": config["image_size"], "vision_feature_layer": config["vision_feature_layer"]}, "media": {"catalog": catalog_path, "catalog_sha256": catalog_hash}, "vision": {"derived_snapshot": caption["derived"][0], "derived_snapshot_sha256": caption["derived"][1], "parent_export": str(Path(caption["export"][0]) / "non_lora_trainables.bin"), "parent_export_sha256": caption["export_hashes"]["non_lora_trainables.bin"], "raw_snapshot": caption["raw"][0], "raw_snapshot_sha256": caption["raw"][1]}, "detector": {"snapshot": caption["detector"][0], "snapshot_sha256": caption["detector"][1], "score_threshold": caption["score_threshold"]}, "protocols": config["protocols"], "generation": config["generation"], "numerics": {"policy": policy_path, "policy_sha256": policy_hash}, "cache": config["cache"]}
        runtime_path, runtime_hash = _file_after_write(output_dir / "r0_prediction_runtime.json", runtime)
        load_prediction_runtime(runtime_path, expected_sha256=runtime_hash)
        implementation_path, implementation_hash = _implementation_manifest(output_dir)
        decoder_path, decoder_hash = _file_after_write(output_dir / "decoder_binding.json", {"schema": "nc_rted_blind_decoder/v1", "implementation": "nc_rted.detection_media.OpenCVFrames"})
        vision_path, vision_hash = _file_after_write(output_dir / "embedded_vision_binding.json", {"schema": "nc_rted_r0_embedded_vision_binding/v1", "derived_snapshot_sha256": caption["derived"][1], "parent_export_sha256": caption["export_hashes"]["non_lora_trainables.bin"], "raw_snapshot_sha256": caption["raw"][1]})
        model_path, model_hash = _file_after_write(output_dir / "model_R0.json", {"group": "R0", "seed": None, "checkpoint": None, "checkpoint_manifest_sha256": None, "checkpoint_state_sha256": None, "final_checkpoint_attestation": None, "final_checkpoint_attestation_sha256": None, "accepted_training_provenance": None, "accepted_training_provenance_sha256": None})
        admission_path, admission_hash = _file_after_write(output_dir / "formal_admission_required.json", {"status": "BLOCKED", "formal_execution_allowed": False, "requested_output_root": str(args.output_root), "reason": "formal admission has not accepted this exact runtime, identity, and binding set"})
        report = {"schema": "nc_rted_r0_blind_manifest_build/v2", "status": "PREFLIGHT_PASS_FORMAL_ADMISSION_REQUIRED", "gpu_launched": False, "runtime": {"path": runtime_path, "sha256": runtime_hash}, "identity_manifest": {"path": identity_path, "sha256": identity_hash}, "r0_model_manifest": {"path": model_path, "sha256": model_hash}, "bindings": {"prediction_source": {"path": prediction_source_path, "sha256": prediction_source_hash}, "implementation": {"path": implementation_path, "sha256": implementation_hash}, "decoder": {"path": decoder_path, "sha256": decoder_hash}, "embedded_vision": {"path": vision_path, "sha256": vision_hash}}, "formal_admission": {"path": admission_path, "sha256": admission_hash}, "denominators": {**VAD_COUNTS, "vau": VAU_COUNT}, "missing_dependencies": ["formal admission for this exact runtime/identity/binding set"]}
    except BuildError as error:
        report["missing_dependencies"] = [str(error)]
    except (KeyError, TypeError, OSError, UnicodeDecodeError) as error:
        report["missing_dependencies"] = [f"invalid required concrete binding: {error}"]
    _write(args.output, report)
    print(json.dumps({"status": report["status"], "output": str(args.output), "missing_dependencies": report["missing_dependencies"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
