#!/usr/bin/env python3
"""Build a sealed four-group manifest set for the interleaved GPU diagnostic.

This is deliberately a preparation tool.  It never invokes CUDA or marks a
formal run admissible; it only publishes manifests after every supplied input
has a completed, hash-bound provenance record.
"""
from __future__ import annotations

import argparse
import copy
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nc_rted.production_runtime import (ProductionRuntimeError, RuntimeManifest, _media_catalog,
                                        _validate_detection_bindings, preflight)
from nc_rted.storage_lock import allocation_lock, ensure_directory
from nc_rted.task_inputs import TaskInputError, TrainingCatalog, sha256_file
from nc_rted.train_worker import TeacherIndex


GROUPS = ("A", "U", "S", "F")
_HEX = set("0123456789abcdef")
_FORBIDDEN_FAST_MARKERS = ("blind_fast_snapshot", "official_blind", "official test", "validation")
_RESERVE_BYTES = 20 * 1024 ** 3
_RENAME_NOREPLACE = 1
_CAPTION_PRODUCER_FILES = (
    "src/nc_rted/batches.py", "src/nc_rted/detection_media.py", "src/nc_rted/detector.py",
    "src/nc_rted/features.py", "src/nc_rted/media_observer.py", "src/nc_rted/observation.py",
    "src/nc_rted/observation_cache.py", "src/nc_rted/tracking.py")
_PROTOCOL_FIELDS = {
    "question_template", "prompt_style", "time_message_style", "memory_enhancement",
    "rt_anomaly", "trigger_threshold", "pool_threshold", "scoring"}


class BuildError(ValueError):
    pass


def _sha(path: Path) -> str:
    return sha256_file(path)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _write_json(path: Path, document: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(document, stream, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_no_replace(source: Path, destination: Path) -> None:
    """Publish one directory without ever replacing an existing destination."""
    try:
        renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
    except AttributeError as error:
        raise BuildError("renameat2(RENAME_NOREPLACE) is required for manifest publication") from error
    renameat2.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
    renameat2.restype = ctypes.c_int
    if renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), _RENAME_NOREPLACE) != 0:
        code = ctypes.get_errno()
        if code == errno.EEXIST:
            raise BuildError("output root already exists")
        raise OSError(code, os.strerror(code), str(destination))


def _allocation_budget(documents: list[dict], block: int) -> int:
    """Reserve payload blocks plus conservative file and directory metadata."""
    serialized = [len(_canonical(document)) + 1 for document in documents]
    payload = sum(((size + block - 1) // block) * block for size in serialized)
    # Eight files and three newly populated directories (temporary, members, staging).
    return payload + (len(serialized) + 6) * block


def _published_identically(root: Path, report: dict) -> bool:
    try:
        published = json.loads((root / "manifest_set.json").read_text(encoding="utf-8"))
        if published != report:
            return False
        if _sha(Path(report["observer_catalog"])) != report["observer_catalog_sha256"]:
            return False
        for group, expected in report["members"].items():
            if _sha(root / "members" / f"{group}.runtime.json") != expected:
                return False
        return (_sha(Path(report["bundle_manifest"])) == report["bundle_manifest_sha256"] and
                _sha(Path(report["source_manifest"])) == report["source_manifest_sha256"])
    except (KeyError, OSError, ValueError):
        return False


def _validate_recovered_publication(root: Path, args: argparse.Namespace, catalog: TrainingCatalog) -> None:
    """Re-run consumer admission from published files without a staging allocation."""
    try:
        media_catalog = _media_catalog(root / "observer_media_catalog.json")
        _validate_detection_bindings(catalog, Path(args.training_fast), media_catalog)
        for group in GROUPS:
            member = root / "members" / f"{group}.runtime.json"
            document = json.loads(member.read_text(encoding="utf-8"))
            preflight(RuntimeManifest(member, _sha(member), document))
    except (ProductionRuntimeError, OSError, TypeError, ValueError) as error:
        raise BuildError("published runtime failed production preflight") from error


def _bound_json(value: str, expected: str, label: str) -> tuple[Path, dict]:
    if not isinstance(expected, str) or len(expected) != 64 or set(expected) - _HEX:
        raise BuildError(f"{label} needs a SHA-256")
    path = Path(value)
    if not path.is_absolute() or not path.is_file() or _sha(path) != expected:
        raise BuildError(f"{label} is absent or differs from its SHA-256")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BuildError(f"{label} is not valid JSON") from error
    if not isinstance(document, dict):
        raise BuildError(f"{label} must be a JSON object")
    return path, document


def _bound_file(value: str, expected: str, label: str) -> Path:
    if not isinstance(expected, str) or len(expected) != 64 or set(expected) - _HEX:
        raise BuildError(f"{label} needs a SHA-256")
    path = Path(value)
    if not path.is_absolute() or not path.is_file() or _sha(path) != expected:
        raise BuildError(f"{label} is absent or differs from its SHA-256")
    return path


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and not (set(value) - _HEX)


def _completion_record(path: str, expected: str, *, field: str, complete: str, label: str) -> tuple[Path, dict]:
    record_path, document = _bound_json(path, expected, label)
    if document.get(field) != complete:
        raise BuildError(f"{label} is not complete")
    return record_path, document


def _source_manifest(source_root: Path) -> dict:
    entries = ("nc_rted_interleaved_gpu_diagnostic.py", "nc_rted_interleaved_formal.py",
               "nc_rted_qualify_interleaved_formal_bundle.py")
    package = source_root / "src" / "nc_rted"
    if (not source_root.is_absolute() or not package.is_dir() or
            any(not (source_root / "scripts" / entry).is_file() for entry in entries)):
        raise BuildError("runtime source root lacks the interleaved harness or nc_rted package")
    files = {f"scripts/{entry}": _sha(source_root / "scripts" / entry) for entry in entries}
    for candidate in sorted(package.glob("*.py")):
        files[str(candidate.relative_to(source_root))] = _sha(candidate)
    if not files or len(files) < 2:
        raise BuildError("runtime source closure is incomplete")
    return {"schema": "nc_rted_interleaved_source_manifest/v1",
            "code_sha256": hashlib.sha256(_canonical(files)).hexdigest(), "files": files}


def _validate_recipe(path: str, expected: str) -> tuple[Path, dict]:
    recipe_path = Path(path)
    if not recipe_path.is_absolute() or not recipe_path.is_file() or _sha(recipe_path) != expected:
        raise BuildError("training recipe is absent or differs from its SHA-256")
    try:
        import yaml
        recipe = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))
    except (ImportError, OSError, ValueError) as error:
        raise BuildError("training recipe is invalid") from error
    if not isinstance(recipe, dict) or recipe.get("updates") != 1000 or recipe.get("accumulation") != 8:
        raise BuildError("training recipe must retain updates=1000 and accumulation=8")
    return recipe_path, recipe


def _validate_fast(args: argparse.Namespace) -> dict:
    path, expected = args.training_fast, args.training_fast_sha256
    fast_path, fast = _bound_json(path, expected, "training Fast snapshot")
    text = (str(fast_path) + "\n" + json.dumps(fast, sort_keys=True)).lower()
    if any(marker in text for marker in _FORBIDDEN_FAST_MARKERS):
        raise BuildError("official blind/validation Fast snapshots cannot be training inputs")
    identity = fast.get("fast_identity")
    if fast.get("schema") != "nc_rted_frozen_fast/v1" or not isinstance(identity, dict) or not identity.get("checkpoint") or not identity.get("implementation"):
        raise BuildError("training Fast snapshot has no complete frozen identity")
    report_path, report = _bound_json(args.fast_binding_report, args.fast_binding_report_sha256,
                                      "training Fast binding report")
    if (report.get("status") != "PASS_ALL_FIXED_TRAIN_PREFIX_BINDINGS_CANDIDATE" or
            report.get("snapshot") != str(fast_path) or report.get("snapshot_sha256") != expected or
            report.get("fixed_prefixes_verified") != 6000 or report.get("media") != 2413 or
            report.get("fast_identity") != identity or report.get("formal_acceptance") is not False):
        raise BuildError("training Fast binding report does not seal the exact legal 6000-prefix snapshot")
    return {"snapshot": str(fast_path), "snapshot_sha256": expected, "identity": copy.deepcopy(identity),
            "binding_report": str(report_path), "binding_report_sha256": args.fast_binding_report_sha256}


def _inherited_protocols(path: str, expected: str) -> dict:
    _, source = _bound_json(path, expected, "inherited protocol source")
    try:
        vad = source["protocols"]["vad"]
        raw = {"ucf-crime": vad["ucf"], "xd-violence": vad["xd"]}
    except (KeyError, TypeError) as error:
        raise BuildError("inherited protocol source does not contain the frozen VAD protocols") from error
    if any(not isinstance(value, dict) or set(value) != _PROTOCOL_FIELDS for value in raw.values()):
        raise BuildError("inherited protocol source has fields outside the frozen VAD protocol contract")
    # Only the two VAD protocol mappings are inherited.  In particular, no R0
    # Fast snapshot, scores, media, fusion, or token configuration is copied.
    return copy.deepcopy(raw)


def _validate_catalog(runtime: dict) -> TrainingCatalog:
    catalog = runtime.get("catalog")
    if not isinstance(catalog, dict):
        raise BuildError("caption runtime has no training catalog")
    try:
        loaded = TrainingCatalog.load(catalog["manifest_directory"], catalog["training_annotations"],
                                      expected_provenance_sha256=catalog["provenance_sha256"])
    except (KeyError, OSError, ValueError, TaskInputError) as error:
        raise BuildError("caption runtime does not bind the exact 6000 detection + 2000 caption catalog") from error
    counts = {kind: sum(task.task == kind for task in loaded.tasks.values()) for kind in ("detection", "caption")}
    if len(loaded.tasks) != 8000 or counts != {"detection": 6000, "caption": 2000}:
        raise BuildError("caption runtime does not bind the exact 6000 detection + 2000 caption catalog")
    return loaded


def _sealed_teacher(args: argparse.Namespace, catalog: TrainingCatalog) -> tuple[dict, dict]:
    _, teacher_status = _completion_record(
        args.teacher_status, args.teacher_status_sha256, field="status", complete="TEACHER_COMPLETED", label="teacher supervisor")
    artifact = Path(args.teacher_manifest)
    if not artifact.is_absolute() or not artifact.is_file() or _sha(artifact) != args.teacher_manifest_sha256:
        raise BuildError("teacher manifest is absent or differs from its SHA-256")
    if teacher_status.get("output") != str(artifact) or teacher_status.get("teacher_manifest_sha256") != args.teacher_manifest_sha256:
        raise BuildError("teacher supervisor does not seal the supplied teacher manifest")
    events_path = _bound_file(args.teacher_events, args.teacher_events_sha256, "teacher supervisor events")
    try:
        rows = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines() if line]
    except (OSError, ValueError) as error:
        raise BuildError("teacher supervisor events are invalid") from error
    running = [row for row in rows if isinstance(row, dict) and row.get("status") == "TEACHER_RUNNING"]
    completed = [row for row in rows if isinstance(row, dict) and row.get("status") == "TEACHER_COMPLETED"]
    if len(running) != 1 or len(completed) != 1:
        raise BuildError("teacher supervisor events lack one running and completed seal")
    running, completed = running[0], completed[0]
    try:
        document = json.loads(artifact.read_text(encoding="utf-8"))
        provenance = document["provenance"]
        input_sha = provenance["input_sha256"]
        config_sha = provenance["config_sha256"]
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise BuildError("teacher manifest lacks out-of-source provenance") from error
    observed_sha = running.get("observation_index_sha256")
    running_config_sha = running.get("config_sha256")
    status_config_sha = teacher_status.get("config_sha256")
    if not all(_is_sha256(value) for value in (input_sha, config_sha, observed_sha, running_config_sha, status_config_sha)):
        raise BuildError("teacher manifest and supervisor provenance need SHA-256 digests")
    if (input_sha != observed_sha or config_sha != status_config_sha or running_config_sha != config_sha or
            completed.get("output") != str(artifact) or
            completed.get("teacher_manifest_sha256") != args.teacher_manifest_sha256):
        raise BuildError("teacher manifest provenance is not sealed by supervisor completion evidence")
    try:
        TeacherIndex.load(artifact, catalog, expected_sha256=args.teacher_manifest_sha256)
    except (OSError, ValueError, TaskInputError) as error:
        raise BuildError("teacher manifest is incompatible with the fixed detection catalog") from error
    completion = {"status": str(Path(args.teacher_status)), "status_sha256": args.teacher_status_sha256,
                  "events": str(events_path), "events_sha256": args.teacher_events_sha256,
                  "observation_index_sha256": input_sha, "config_sha256": config_sha}
    return {"artifact": str(artifact), "sha256": args.teacher_manifest_sha256}, completion


def _completed_caption_runtime(args: argparse.Namespace, source: dict) -> dict:
    _, caption_status = _completion_record(
        args.caption_supervisor, args.caption_supervisor_sha256, field="state", complete="FULL_PREPARATION_COMPLETED", label="caption supervisor")
    caption_path, caption = _bound_json(args.caption_runtime, args.caption_runtime_sha256, "caption runtime")
    if caption.get("schema") != "nc_rted_production_runtime/v1" or caption_status.get("config_sha256") != args.caption_runtime_sha256:
        raise BuildError("caption runtime is not the completed preparation binding")
    if caption_status.get("formal_execution_allowed") is not False:
        raise BuildError("caption preparation supervisor must remain diagnostic-only")
    cache = caption.get("media", {}).get("caption_observation_cache")
    if not isinstance(cache, dict) or not isinstance(cache.get("root"), str) or not Path(cache["root"]).is_absolute():
        raise BuildError("caption runtime does not bind a completed caption cache")
    runtime_source = caption.get("runtime_source")
    if not isinstance(runtime_source, dict):
        raise BuildError("caption runtime does not bind its frozen producer source")
    _, producer_source = _bound_json(runtime_source.get("manifest"), runtime_source.get("manifest_sha256"),
                                     "caption producer source manifest")
    producer_files = producer_source.get("files")
    if not isinstance(producer_files, dict) or any(producer_files.get(name) != source["files"].get(name)
                                                   for name in _CAPTION_PRODUCER_FILES):
        raise BuildError("later runtime source changes the accepted caption-cache producer identity")
    return caption


def _detection_media_rows(fast: dict) -> list[dict]:
    rows = fast.get("media")
    if not isinstance(rows, list) or len(rows) != 2413:
        raise BuildError("training Fast snapshot does not contain exactly 2,413 media bindings")
    required = ("dataset", "media_key", "media_path", "media_sha256", "fps", "frame_count", "height", "width")
    output = []
    for row in rows:
        if not isinstance(row, dict) or any(name not in row for name in required):
            raise BuildError("training Fast snapshot has an incomplete media binding")
        output.append({name: row[name] for name in required} | {"aliases": []})
    return output


def _merged_media_catalog(caption: dict, fast: dict) -> dict:
    media = caption.get("media")
    if not isinstance(media, dict):
        raise BuildError("caption runtime has no media configuration")
    source_path = _bound_file(media.get("catalog"), media.get("catalog_sha256"), "caption runtime media catalog")
    try:
        caption_rows = json.loads(source_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise BuildError("caption runtime media catalog is invalid") from error
    if isinstance(caption_rows, dict):
        caption_rows = caption_rows.get("media")
    if not isinstance(caption_rows, list) or len(caption_rows) != 2000:
        raise BuildError("caption runtime does not bind exactly 2,000 caption media identities")
    merged = copy.deepcopy(caption_rows) + _detection_media_rows(fast)
    identities = set()
    for row in merged:
        if not isinstance(row, dict):
            raise BuildError("merged observer catalog has invalid media rows")
        aliases = row.get("aliases", [])
        if not isinstance(aliases, list):
            raise BuildError("merged observer catalog has invalid aliases")
        keys = [row.get("media_key"), *aliases]
        dataset = row.get("dataset")
        if not isinstance(dataset, str) or any(not isinstance(key, str) or not key or (dataset, key) in identities for key in keys):
            raise BuildError("caption and detection observer catalogs overlap or are invalid")
        identities.update((dataset, key) for key in keys)
    return {"schema": "nc_rted_interleaved_observer_catalog/v1", "media": merged}


def _compose_runtime(args: argparse.Namespace, source: dict) -> tuple[dict, TrainingCatalog, dict]:
    _validate_recipe(args.training_recipe, args.training_recipe_sha256)
    caption = _completed_caption_runtime(args, source)
    catalog = _validate_catalog(caption)
    runtime = copy.deepcopy(caption)
    runtime["fast"] = _validate_fast(args)
    runtime["fast"]["protocols"] = _inherited_protocols(args.protocol_source, args.protocol_source_sha256)
    runtime["teacher"], completion = _sealed_teacher(args, catalog)
    runtime["caption_cache_producer"] = copy.deepcopy(runtime.pop("runtime_source"))
    runtime["hashes"]["code_sha256"] = source["code_sha256"]
    runtime["hashes"]["runtime_sha256"] = source["code_sha256"]
    _, fast_snapshot = _bound_json(args.training_fast, args.training_fast_sha256, "training Fast snapshot")
    return runtime, catalog, {"catalog": _merged_media_catalog(caption, fast_snapshot), "teacher_completion": completion}


def build(args: argparse.Namespace) -> dict:
    output_root = Path(args.output_root)
    source_root = Path(args.runtime_source_root)
    if not output_root.is_absolute() or not output_root.parent.is_dir():
        raise BuildError("output root must be a new path below an existing directory")
    if args.seed not in {17, 42, 2026} or not isinstance(args.device, str) or not args.device:
        raise BuildError("seed or device is invalid")
    if not 1 <= args.diagnostic_updates <= 1000 or not 1 <= args.diagnostic_checkpoint_interval <= 50 or args.diagnostic_checkpoint_interval > args.diagnostic_updates:
        raise BuildError("diagnostic prefix/checkpoint interval is invalid")
    source = _source_manifest(source_root)
    runtime, catalog, auxiliary = _compose_runtime(args, source)

    temporary = None
    try:
      with allocation_lock(output_root.parent):
        staging = output_root.parent / ".nc_rted_interleaved_staging"
        ensure_directory(staging, _RESERVE_BYTES)
        # Calculate the exact serialized payload before allocating the staging
        # tree, then add block-rounded conservative metadata overhead.
        media_document = auxiliary["catalog"]
        source_name = "source_manifest.json"
        runtime["media"] = copy.deepcopy(runtime["media"])
        runtime["media"]["catalog"] = str(output_root / "observer_media_catalog.json")
        runtime["media"]["catalog_sha256"] = hashlib.sha256(_canonical(media_document) + b"\n").hexdigest()
        runtime["runtime_source"] = {"manifest": str(output_root / source_name),
                                     "manifest_sha256": hashlib.sha256(_canonical(source) + b"\n").hexdigest()}
        member_documents = {}
        member_hashes = {}
        for group in GROUPS:
            document = copy.deepcopy(runtime)
            document["run"] = {"run_id": f"diagnostic:interleaved:{args.seed}:{group}", "group": group,
                               "seed": args.seed, "device": args.device, "mode": "diagnostic",
                               "diagnostic_updates": args.diagnostic_updates,
                               "checkpoint_root": str(output_root / "runs" / group / "checkpoints"),
                               "progress_path": str(output_root / "runs" / group / "progress.json")}
            member_documents[group] = document
            member_hashes[group] = hashlib.sha256(_canonical(document) + b"\n").hexdigest()
        members = {group: {"manifest": str(output_root / "members" / f"{group}.runtime.json"),
                           "sha256": member_hashes[group]} for group in GROUPS}
        bundle = {"schema": "nc_rted_interleaved_gpu_harness/v1", "diagnostic_updates": args.diagnostic_updates,
                  "diagnostic_checkpoint_interval": args.diagnostic_checkpoint_interval,
                  "bundle_checkpoint_root": str(output_root / "bundle-checkpoints"), "members": members,
                  "source_manifest": str(output_root / source_name), "source_manifest_sha256": runtime["runtime_source"]["manifest_sha256"]}
        report = {"schema": "nc_rted_interleaved_diagnostic_manifest_set/v1", "purpose": "diagnostic_only",
                  "recipe": {"updates": 1000, "accumulation": 8}, "caption_runtime": args.caption_runtime,
                  "caption_runtime_sha256": args.caption_runtime_sha256, "training_fast": args.training_fast,
                  "training_fast_sha256": args.training_fast_sha256, "fast_binding_report": args.fast_binding_report,
                  "fast_binding_report_sha256": args.fast_binding_report_sha256, "protocol_source": args.protocol_source,
                  "protocol_source_sha256": args.protocol_source_sha256, "teacher_manifest": args.teacher_manifest,
                  "teacher_manifest_sha256": args.teacher_manifest_sha256,
                  "teacher_completion": auxiliary["teacher_completion"], "members": member_hashes,
                  "observer_catalog": str(output_root / "observer_media_catalog.json"),
                  "observer_catalog_sha256": runtime["media"]["catalog_sha256"],
                  "bundle_manifest": str(output_root / "interleaved_bundle.json"),
                  "bundle_manifest_sha256": hashlib.sha256(_canonical(bundle) + b"\n").hexdigest(),
                  "source_manifest": str(output_root / source_name),
                  "source_manifest_sha256": runtime["runtime_source"]["manifest_sha256"]}
        # A failed final directory sync can leave a complete, published artifact.
        # Validate and finish its durability before considering a new allocation.
        if output_root.exists():
            if not _published_identically(output_root, report):
                raise BuildError("output root already exists with different artifacts")
            _validate_recovered_publication(output_root, args, catalog)
            _fsync_directory(output_root)
            _fsync_directory(output_root.parent)
            _fsync_directory(staging)
            return report
        block = max(4096, os.statvfs(staging).f_frsize)
        budget = _allocation_budget([media_document, source, *member_documents.values(), bundle, report], block)
        if shutil.disk_usage(staging).free < _RESERVE_BYTES + budget:
            raise BuildError("manifest publication lacks reserve-plus-budget admission")
        temporary = Path(tempfile.mkdtemp(prefix="interleaved-manifests-", dir=staging))
        media_path = temporary / "observer_media_catalog.json"
        _write_json(media_path, media_document)
        source_path = temporary / "source_manifest.json"
        _write_json(source_path, source)
        shadow = copy.deepcopy(runtime)
        shadow["media"]["catalog"] = str(media_path)
        try:
            media_catalog = _media_catalog(media_path)
            _validate_detection_bindings(catalog, Path(args.training_fast), media_catalog)
            preflight(RuntimeManifest(Path(args.caption_runtime), hashlib.sha256(_canonical(shadow)).hexdigest(), shadow))
        except (ProductionRuntimeError, OSError, TypeError, ValueError) as error:
            raise BuildError("composed runtime failed production preflight") from error
        (temporary / "members").mkdir()
        for group in GROUPS:
            document = member_documents[group]
            target = temporary / "members" / f"{group}.runtime.json"
            _write_json(target, document)
        bundle_path = temporary / "interleaved_bundle.json"
        _write_json(bundle_path, bundle)
        _write_json(temporary / "manifest_set.json", report)
        # The serializations are already written, but retaining this admission
        # calculation makes any future document growth fail closed before publish.
        if shutil.disk_usage(staging).free < _RESERVE_BYTES:
            raise BuildError("manifest publication consumed the protected 20 GiB reserve")
        _fsync_directory(temporary / "members")
        _fsync_directory(temporary)
        _fsync_directory(staging)
        _publish_no_replace(temporary, output_root)
        temporary = None
        _fsync_directory(output_root)
        _fsync_directory(output_root.parent)
        _fsync_directory(staging)
        return report
    except BaseException:
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-fast", required=True)
    parser.add_argument("--training-fast-sha256", required=True)
    parser.add_argument("--fast-binding-report", required=True)
    parser.add_argument("--fast-binding-report-sha256", required=True)
    parser.add_argument("--protocol-source", required=True)
    parser.add_argument("--protocol-source-sha256", required=True)
    parser.add_argument("--teacher-manifest", required=True)
    parser.add_argument("--teacher-manifest-sha256", required=True)
    parser.add_argument("--teacher-status", required=True)
    parser.add_argument("--teacher-status-sha256", required=True)
    parser.add_argument("--teacher-events", required=True)
    parser.add_argument("--teacher-events-sha256", required=True)
    parser.add_argument("--caption-runtime", required=True)
    parser.add_argument("--caption-runtime-sha256", required=True)
    parser.add_argument("--caption-supervisor", required=True)
    parser.add_argument("--caption-supervisor-sha256", required=True)
    parser.add_argument("--training-recipe", required=True)
    parser.add_argument("--training-recipe-sha256", required=True)
    parser.add_argument("--runtime-source-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--diagnostic-updates", type=int, required=True)
    parser.add_argument("--diagnostic-checkpoint-interval", type=int, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(build(args), sort_keys=True))
    except BuildError as error:
        raise SystemExit(f"manifest builder refused: {error}")


if __name__ == "__main__":
    main()
