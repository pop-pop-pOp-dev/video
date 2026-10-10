"""Sealed development-only Fast execution with per-media atomic recovery."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Callable, Mapping


class FastPreparationError(ValueError):
    pass


def digest_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def bound_json(binding: Mapping) -> dict | list:
    path = Path(binding["path"])
    if digest_file(path) != binding["sha256"]:
        raise FastPreparationError(f"bound JSON changed: {path}")
    return json.loads(path.read_text())


def _unique(rows, key):
    result = {}
    for row in rows:
        identity = key(row)
        if identity in result:
            raise FastPreparationError("duplicate bound identity")
        result[identity] = row
    return result


def validate_plan(plan: dict, training_bindings: dict) -> dict:
    """Close the entire worklist against the fixed legal development inputs."""
    if plan.get("schema") != "nc_rted_mechanism_development_fast_plan/v1":
        raise FastPreparationError("development Fast plan schema differs")
    frozen = training_bindings["training_inputs"]["fast"]
    if plan.get("fast_identity") != frozen["identity"] or plan.get("protocols") != frozen["protocols"]:
        raise FastPreparationError("plan differs from the inherited Fast identity/protocols")
    runtime = bound_json(plan["runtime_inputs"])
    if runtime.get("schema") != "nc_rted_mechanism_development_runtime_inputs/v1":
        raise FastPreparationError("runtime input schema differs")
    documents = {key: bound_json(value) for key, value in runtime["inputs"].items()}
    development = documents["development_manifest"]
    if development.get("schema") != "nc_rted_mechanism_development_prefixes/v1":
        raise FastPreparationError("development manifest schema differs")
    for key, binding in development["inputs"].items():
        if runtime["inputs"].get(key) != binding:
            raise FastPreparationError("development input lineage differs")
    catalog = documents["media_catalog"]
    if catalog.get("development_manifest") != runtime["inputs"]["development_manifest"]:
        raise FastPreparationError("media catalog belongs to another development set")
    selected = _unique(development["records"], lambda row: row["sample_id"])
    actual = _unique(runtime["records"], lambda row: row["sample_id"])
    if not selected or set(selected) != set(actual):
        raise FastPreparationError("runtime must contain every sealed development prefix exactly once")
    allocations = _unique(documents["source_splits"], lambda row: (row["dataset"], row["key"]))
    media = _unique(catalog["media"], lambda row: (row["dataset"], row["media_key"]))
    databases = {"ucf-crime": documents["ucf_database"], "xd-violence": documents["xd_database"]}
    work = {}
    for sample_id, row in actual.items():
        original = selected[sample_id]
        if set(row) != set(original) | {"media_path", "media_sha256"} or any(row.get(k) != v for k, v in original.items()):
            raise FastPreparationError("sealed development row was changed")
        identity = (row["dataset"], row["key"])
        if allocations.get(identity, {}).get("allocation") != "development" or row.get("scope") != "vad_causal_latest_8s":
            raise FastPreparationError("source is outside development allocation")
        source = media.get(identity)
        if source is None or any(row[k] != source[k] for k in ("media_path", "media_sha256")):
            raise FastPreparationError("runtime media binding differs")
        item = databases[row["dataset"]].get(row["key"])
        fps, frames = float(source["fps"]), int(source["frame_count"])
        if item is None or not math.isfinite(fps) or fps <= 0 or frames < 1 or min(int(source["height"]), int(source["width"])) < 1:
            raise FastPreparationError("invalid bound media geometry")
        if int(item["n_frames"]) != frames or not math.isclose(float(item["fps"]), fps, rel_tol=1e-6):
            raise FastPreparationError("media geometry differs from legal training source")
        query = row["query_index"]
        if type(query) is not int or query < 0:
            raise FastPreparationError("invalid development query index")
        indices = list(range(query * 4 * max(1, int(fps / 4)), min((query + 1) * 4 * max(1, int(fps / 4)), frames), max(1, int(fps / 4))))
        if not indices or not math.isclose(float(row["observed_seconds"]), indices[-1] / fps, abs_tol=1e-6):
            raise FastPreparationError("query exceeds its sealed causal endpoint")
        prior = work.get(identity)
        candidate = {k: row[k] for k in ("dataset", "key", "media_path", "media_sha256")}
        candidate["max_query_index"] = max(query, prior["max_query_index"] if prior else query)
        work[identity] = candidate
    planned = _unique(plan["sources"], lambda row: (row["dataset"], row["key"]))
    if planned != work or set(media) != set(work):
        raise FastPreparationError("Fast plan source set or maximum query differs from sealed development inputs")
    return media


def verify_scorer_inputs(config: dict, training_bindings: dict) -> dict:
    """Verify actual load paths against the previously frozen Fast inventories."""
    identity = training_bindings["training_inputs"]["fast"]["identity"]
    checkpoint = identity["checkpoint"]
    if config["selected_json_sha256"] != checkpoint["selection_sha256"] or config["source_sha256"] != identity["implementation"]:
        raise FastPreparationError("original Fast configuration identity differs")
    if (config["image_size"], config["target_fps"], config["query_interval"], config["batch_size"], config["attn_implementation"]) != (384, 4, 4, 1, "sdpa"):
        raise FastPreparationError("unsupported inherited Fast scoring protocol")
    model = Path(config["model_path"])
    stage1 = Path(config["stage1_output"])
    selection = bound_json({"path": str(stage1 / "selected.json"), "sha256": checkpoint["selection_sha256"]})
    if selection.get("checkpoint") != config["selected_checkpoint"]:
        raise FastPreparationError("selected Stage1 checkpoint differs")
    relative_adapter = Path(selection["adapter"])
    if relative_adapter.is_absolute() or ".." in relative_adapter.parts:
        raise FastPreparationError("selected adapter path escapes Stage1 output")
    adapter = stage1 / relative_adapter
    for directory, inventory in [(model, config["model_inventory_sha256"]), (adapter, checkpoint["adapter_files"])]:
        for name, expected in inventory.items():
            if Path(name).is_absolute() or ".." in Path(name).parts or digest_file(directory / name) != expected:
                raise FastPreparationError("actual Fast model/adapter file differs from frozen identity")
    vision = Path(config["streamforest_vision_weights_path"])
    if digest_file(vision) != config["streamforest_vision_weights_sha256"]:
        raise FastPreparationError("actual frozen vision weights differ")
    paths = {}
    for key, suffix in [("released_precompute", "/precompute_pg_scores.py"), ("grid_stream", "/reactvau_fast_grid_stream.py"), ("prompt_source", "/get_prompt.py")]:
        matches = [(p, sha) for p, sha in config["source_sha256"].items() if p.endswith(suffix)]
        if len(matches) != 1 or digest_file(matches[0][0]) != matches[0][1]:
            raise FastPreparationError("actual Fast implementation differs")
        paths[key] = matches[0][0]
    if paths["released_precompute"] != config["released_precompute"]:
        raise FastPreparationError("released scorer path differs")
    return {**paths, "model_path": str(model), "lora_path": str(adapter), "streamforest_weights": str(vision)}


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FastPreparationError(f"refusing to overwrite committed artifact: {path}")
    fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        os.link(temporary, path)
        parent = os.open(path.parent, os.O_DIRECTORY)
        try: os.fsync(parent)
        finally: os.close(parent)
    finally:
        Path(temporary).unlink(missing_ok=True)


@contextmanager
def journal_lock(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".lock").open("a") as stream:
        try: fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc: raise FastPreparationError("development Fast journal already has a worker") from exc
        try: yield
        finally: fcntl.flock(stream, fcntl.LOCK_UN)


def _media_row(source, metadata, scores):
    maximum = source["max_query_index"]
    if not isinstance(scores, list) or len(scores) != maximum + 1 or any(type(x) not in (float, int) or not math.isfinite(x) or not 0 <= x <= 1 for x in scores):
        raise FastPreparationError("scorer omitted or corrupted bounded scores")
    fps, frames = float(metadata["fps"]), int(metadata["frame_count"])
    interval = max(1, int(fps / 4))
    queries = [{"index": i, "frame_indices": list(range(i * 4 * interval, min((i + 1) * 4 * interval, frames), interval)), "fast_score": float(score)} for i, score in enumerate(scores)]
    if any(not row["frame_indices"] for row in queries):
        raise FastPreparationError("query lies beyond actual media")
    return {**{k: metadata[k] for k in ("dataset", "media_key", "media_path", "media_sha256", "fps", "frame_count", "height", "width")}, "target_fps": 4, "query_interval": 4, "max_query_index": maximum, "queries": queries}


def execute_plan(*, plan: dict, media: dict, binding: dict, journal: Path, output: Path,
                 scorer_factory: Callable, deadline: datetime, minimum_free_bytes: int = 20 * 1024**3) -> dict:
    """Reuse verified media commits; instantiate the model only for missing work."""
    if output.exists(): raise FastPreparationError("final output already exists")
    if deadline.tzinfo is None: raise FastPreparationError("deadline must include timezone")
    scope = canonical_digest(binding)
    with journal_lock(journal):
        header = journal / "binding.json"
        if header.exists():
            if json.loads(header.read_text()) != binding: raise FastPreparationError("journal belongs to a different plan/scorer identity")
        else: atomic_json(header, binding)
        rows, scorer = [], None
        for source in plan["sources"]:
            if datetime.now(timezone.utc) >= deadline: raise FastPreparationError("development Fast deadline reached; committed media retained")
            if shutil.disk_usage(journal).free < minimum_free_bytes: raise FastPreparationError("disk reserve reached; committed media retained")
            key = (source["dataset"], source["key"])
            if digest_file(source["media_path"]) != source["media_sha256"]: raise FastPreparationError("bound development media changed")
            record_path = journal / (canonical_digest(list(key)) + ".json")
            if record_path.exists():
                record = json.loads(record_path.read_text()); row = record["media"]
                scores = [entry["fast_score"] for entry in row["queries"]]
                if record.get("scope") != scope or record.get("source") != source or record.get("media_sha256") != canonical_digest(row) or row != _media_row(source, media[key], scores):
                    raise FastPreparationError("committed development Fast row is corrupt or has a different scope")
            else:
                if scorer is None: scorer = scorer_factory()
                row = _media_row(source, media[key], scorer(source, source["max_query_index"]))
                atomic_json(record_path, {"scope": scope, "source": source, "media": row, "media_sha256": canonical_digest(row)})
            rows.append(row)
            print(json.dumps({"committed_media": len(rows), "total_media": len(plan["sources"]), "source": list(key)}), flush=True)
        result = {"schema": "nc_rted_development_fast/v1", "fast_identity": plan["fast_identity"], "media": rows}
        atomic_json(output, result)
        return result
