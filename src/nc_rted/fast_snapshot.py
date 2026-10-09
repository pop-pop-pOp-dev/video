"""Export hash-bound inherited Fast scores into the NC-RTED replay schema."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Mapping

from .task_inputs import TaskInputError, sha256_file


SCHEMA = "nc_rted_frozen_fast/v1"


class FastSnapshotError(TaskInputError):
    pass


def _read_json(path: str | Path, expected_sha256: str) -> object:
    source = Path(path)
    if not source.is_file() or sha256_file(source) != expected_sha256:
        raise FastSnapshotError("bound input is missing or its SHA-256 differs")
    return json.loads(source.read_text())


def _identity(value: Mapping) -> dict:
    if not isinstance(value, Mapping) or not value:
        raise FastSnapshotError("Fast identity must be a nonempty mapping")
    required = {"checkpoint", "implementation"}
    if not required.issubset(value) or any(not value[key] for key in required):
        raise FastSnapshotError("Fast identity must bind checkpoint and implementation")
    return dict(value)


def _metadata(document: object) -> list[dict]:
    rows = document.get("media") if isinstance(document, dict) else document
    if not isinstance(rows, list) or not rows:
        raise FastSnapshotError("media metadata must contain nonempty media rows")
    seen, result = set(), []
    for row in rows:
        if not isinstance(row, dict):
            raise FastSnapshotError("media metadata row is not a mapping")
        required = ("dataset", "media_key", "fast_cache_key", "media_path", "media_sha256", "fps", "frame_count", "height", "width")
        if any(key not in row for key in required):
            raise FastSnapshotError("media metadata lacks a required identity field")
        if not isinstance(row["fast_cache_key"], str) or not row["fast_cache_key"]:
            raise FastSnapshotError("explicit original Fast cache key is required")
        identity = (row["dataset"], row["media_key"])
        if identity in seen or not all(isinstance(value, str) and value for value in identity) or not isinstance(row["media_path"], str):
            raise FastSnapshotError("duplicate or invalid media identity")
        if (not isinstance(row["media_sha256"], str) or len(row["media_sha256"]) != 64 or
                not isinstance(row["fps"], (int, float)) or not math.isfinite(row["fps"]) or row["fps"] <= 0 or
                type(row["frame_count"]) is not int or row["frame_count"] <= 0 or
                any(type(row[key]) is not int or row[key] <= 0 for key in ("height", "width"))):
            raise FastSnapshotError("invalid bound media metadata")
        seen.add(identity)
        result.append(dict(row))
    return result


def _queries(entry: object, row: dict) -> list[dict]:
    if not isinstance(entry, dict):
        raise FastSnapshotError("Fast cache entry is not a mapping")
    scores, recorded_interval = entry.get("pg_scores"), entry.get("sample_interval")
    interval = max(1, int(float(row["fps"]) / 4))
    if type(recorded_interval) is not int or recorded_interval != interval:
        raise FastSnapshotError("Fast cache sample_interval is not evidenced by bound media FPS")
    if not isinstance(scores, list) or not scores:
        raise FastSnapshotError("Fast cache lacks complete per-query scores")
    sampled_count = (row["frame_count"] + interval - 1) // interval
    expected_queries = (sampled_count + 3) // 4
    if len(scores) != expected_queries:
        raise FastSnapshotError("Fast score count does not cover every bound original query")
    queries = []
    for index, score in enumerate(scores):
        if not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
            raise FastSnapshotError("Fast cache contains missing or invalid score")
        frames = list(range(index * 4 * interval, min((index + 1) * 4 * interval, row["frame_count"]), interval))
        if not frames:
            raise FastSnapshotError("empty derived Fast query")
        # Preserve JSON's original int/float score representation; do not round.
        queries.append({"index": index, "frame_indices": frames, "fast_score": score})
    return queries


def build_snapshot(*, fast_cache: str | Path, fast_cache_sha256: str,
                   media_metadata: str | Path, media_metadata_sha256: str,
                   fast_identity: Mapping) -> dict:
    """Build in memory only; neither hashes media nor opens/decodes a video."""
    cache = _read_json(fast_cache, fast_cache_sha256)
    if not isinstance(cache, dict):
        raise FastSnapshotError("Fast cache must be a key-to-entry mapping")
    media = []
    for row in _metadata(_read_json(media_metadata, media_metadata_sha256)):
        entry = cache.get(row["fast_cache_key"])
        if entry is None:
            raise FastSnapshotError("bound media has no Fast cache entry")
        media.append({"dataset": row["dataset"], "media_key": row["media_key"], "media_path": row["media_path"],
                      "media_sha256": row["media_sha256"], "fps": row["fps"], "frame_count": row["frame_count"],
                      "height": row["height"], "width": row["width"], "target_fps": 4, "query_interval": 4,
                      "fast_cache_key": row["fast_cache_key"], "queries": _queries(entry, row)})
    return {"schema": SCHEMA, "fast_identity": _identity(fast_identity),
            "bindings": {"fast_cache_sha256": fast_cache_sha256, "media_metadata_sha256": media_metadata_sha256},
            "media": media}


def write_snapshot(document: dict, output: str | Path, *, reserved_free_bytes: int = 20 * 1024**3) -> str:
    target = Path(output)
    if target.exists():
        raise FastSnapshotError("refusing to overwrite frozen Fast snapshot")
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode()
    if reserved_free_bytes < 0 or shutil.disk_usage(target.parent).free < reserved_free_bytes + len(encoded) + 8192:
        raise FastSnapshotError("Fast snapshot would violate disk reserve")
    descriptor, name = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        if shutil.disk_usage(target.parent).free < reserved_free_bytes:
            raise FastSnapshotError("Fast snapshot would violate disk reserve")
        try:
            os.link(name, target)  # atomic create-if-absent on this same filesystem
        except FileExistsError as error:
            raise FastSnapshotError("refusing to overwrite frozen Fast snapshot") from error
        os.unlink(name)
        directory = os.open(target.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(name):
            os.unlink(name)
    return hashlib.sha256(encoded).hexdigest()
