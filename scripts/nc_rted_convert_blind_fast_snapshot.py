#!/usr/bin/env python3
"""Convert a completed label-free blind Fast raw cache into the runtime snapshot."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile


SCHEMA = "nc_rted_blind_fast_snapshot/v1"
RAW_SCHEMA = "nc_rted_blind_fast_raw/v1"
CATALOG_SCHEMA = "nc_rted_blind_media_catalog/v1"
RESERVE = 20 * 1024**3


class ConversionError(ValueError):
    pass


def _digest(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> object:
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise ConversionError(f"bound input is absent: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ConversionError(f"invalid JSON: {path}") from error


def _validate_catalog(catalog: object) -> dict[tuple[str, str], dict]:
    if not isinstance(catalog, dict) or set(catalog) != {"schema", "media"} or catalog["schema"] != CATALOG_SCHEMA or not isinstance(catalog["media"], list):
        raise ConversionError("VAD catalog schema differs")
    rows = catalog["media"]
    if len(rows) != 1051:
        raise ConversionError("VAD catalog denominator differs")
    fields = {"dataset", "media_key", "media_path", "media_sha256", "fps", "frame_count", "height", "width", "request_index"}
    expected: dict[tuple[str, str], dict] = {}
    counts = {"ucf": 0, "xd": 0}
    for row in rows:
        if not isinstance(row, dict) or set(row) != fields:
            raise ConversionError("VAD catalog row schema differs")
        dataset, key = row["dataset"], row["media_key"]
        if (dataset not in counts or not isinstance(key, str) or not key or not isinstance(row["media_path"], str) or not Path(row["media_path"]).is_absolute() or
                not _digest(row["media_sha256"]) or not isinstance(row["fps"], (int, float)) or isinstance(row["fps"], bool) or not math.isfinite(float(row["fps"])) or float(row["fps"]) <= 0 or
                type(row["frame_count"]) is not int or row["frame_count"] < 1 or type(row["height"]) is not int or row["height"] < 1 or type(row["width"]) is not int or row["width"] < 1 or
                (row["request_index"] is not None and (type(row["request_index"]) is not int or row["request_index"] < 0))):
            raise ConversionError("VAD catalog row identity or geometry differs")
        identity = (dataset, key)
        if identity in expected:
            raise ConversionError("VAD catalog has duplicate identity")
        expected[identity] = row
        counts[dataset] += 1
    if len(expected) != 1051 or counts != {"ucf": 251, "xd": 800}:
        raise ConversionError("VAD catalog denominator differs")
    return expected


def _scorer_binding(config: object, raw_config_sha256: str) -> dict:
    if not isinstance(config, dict) or config.get("schema") != "nc_rted_blind_fast_raw_preparation/v4" or not _digest(raw_config_sha256):
        raise ConversionError("raw Fast configuration binding differs")
    required = {"catalog_path", "catalog_sha256", "selected_sha256", "source_sha256", "vision_weights_sha256"}
    if not required.issubset(config) or not isinstance(config["catalog_path"], str) or not _digest(config["catalog_sha256"]) or not _digest(config["selected_sha256"]) or not _digest(config["vision_weights_sha256"]):
        raise ConversionError("raw Fast scorer binding differs")
    source = config["source_sha256"]
    if not isinstance(source, dict) or not source:
        raise ConversionError("raw Fast scorer binding differs")
    for path, digest in source.items():
        if not isinstance(path, str) or not Path(path).is_absolute() or not _digest(digest):
            raise ConversionError("raw Fast scorer binding differs")
    return {"config_sha256": raw_config_sha256, "catalog_sha256": config["catalog_sha256"], "selected_sha256": config["selected_sha256"],
            "source_sha256": source, "vision_weights_sha256": config["vision_weights_sha256"]}


def _snapshot(raw_path: Path, catalog_path: Path, raw_config_path: Path, raw_config_sha256: str) -> dict:
    raw, catalog, config = _read(raw_path), _read(catalog_path), _read(raw_config_path)
    expected = _validate_catalog(catalog)
    binding = _scorer_binding(config, raw_config_sha256)
    if (_sha(raw_config_path) != raw_config_sha256 or config["catalog_path"] != str(catalog_path) or config["catalog_sha256"] != _sha(catalog_path)):
        raise ConversionError("raw Fast configuration binding differs")
    expected_binding = hashlib.sha256(json.dumps(binding, sort_keys=True).encode("utf-8")).hexdigest()
    if (not isinstance(raw, dict) or set(raw) != {"schema", "catalog_sha256", "scorer_binding_sha256", "rows"} or raw["schema"] != RAW_SCHEMA or
            raw["catalog_sha256"] != config["catalog_sha256"] or raw["scorer_binding_sha256"] != expected_binding or
            not isinstance(raw["rows"], list)):
        raise ConversionError("blind Fast raw cache binding differs")
    output, seen = [], set()
    fields = {"dataset", "media_key", "media_path", "media_sha256", "fps", "frame_count", "height", "width", "sample_interval", "queries"}
    for row in raw["rows"]:
        if not isinstance(row, dict) or set(row) != fields:
            raise ConversionError("blind Fast raw row schema differs")
        identity = (row["dataset"], row["media_key"])
        bound = expected.get(identity)
        if identity in seen or bound is None:
            raise ConversionError("blind Fast raw identity differs")
        seen.add(identity)
        if any(row[name] != bound.get(name) for name in ("media_path", "media_sha256", "fps", "frame_count", "height", "width")):
            raise ConversionError("blind Fast raw media binding differs")
        interval = max(1, int(float(row["fps"]) / 4))
        if row["sample_interval"] != interval or not isinstance(row["queries"], list):
            raise ConversionError("blind Fast raw sampling schedule differs")
        count = ((row["frame_count"] + interval - 1) // interval + 3) // 4
        if len(row["queries"]) != count:
            raise ConversionError("blind Fast raw query coverage differs")
        for index, query in enumerate(row["queries"]):
            frames = list(range(index * 4 * interval, min((index + 1) * 4 * interval, row["frame_count"]), interval))
            if (not isinstance(query, dict) or set(query) != {"index", "frame_indices", "fast_score"} or query["index"] != index or query["frame_indices"] != frames or
                    not isinstance(query["fast_score"], (int, float)) or isinstance(query["fast_score"], bool) or not math.isfinite(float(query["fast_score"])) or not 0 <= float(query["fast_score"]) <= 1):
                raise ConversionError("blind Fast raw query differs")
        output.append({key: row[key] for key in ("dataset", "media_key", "media_path", "media_sha256", "fps", "frame_count", "height", "width")}
                      | {"target_fps": 4, "query_interval": 4, "queries": row["queries"]})
    if seen != set(expected):
        raise ConversionError("blind Fast raw cache omits VAD media")
    return {"schema": SCHEMA, "media": output}


def _publish(path: Path, document: dict) -> str:
    if not path.is_absolute() or path.exists():
        raise ConversionError("output must be an absent absolute path")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    if os.statvfs(path.parent).f_bavail * os.statvfs(path.parent).f_frsize < RESERVE + len(encoded) + 8192:
        raise ConversionError("snapshot publication would violate the 20 GiB reserve")
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded); stream.flush(); os.fsync(stream.fileno())
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_DIRECTORY)
        try: os.fsync(directory)
        finally: os.close(directory)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)
    return hashlib.sha256(encoded).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--raw-config", type=Path, required=True)
    parser.add_argument("--raw-config-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        document = _snapshot(args.raw, args.catalog, args.raw_config, args.raw_config_sha256)
        digest = _publish(args.output, document)
    except ConversionError as error:
        parser.error(str(error))
    print(json.dumps({"status": "EXPORTED", "output": str(args.output), "sha256": digest, "media": len(document["media"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
