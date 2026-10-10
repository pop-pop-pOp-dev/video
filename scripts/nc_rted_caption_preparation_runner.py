#!/usr/bin/env python3
"""Resumable diagnostic runner for the bound 2,000-caption observation cache.

The accepted runtime lives in ``--runtime-source-root`` and is never edited by
this wrapper.  This wrapper first validates all local bindings, then either
runs one cold/warm parity smoke or resumes the fixed caption set one entry at a
time.  It intentionally has no training, teacher, or Fast-provider path.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any


GIB = 1024 ** 3
class RunnerError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def append_event(path: Path, value: dict[str, Any]) -> None:
    value = {"time_unix": time.time(), **value}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def path_values(value: object) -> list[Path]:
    result: list[Path] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"root", "path", "catalog", "module", "config", "snapshot", "artifact", "directory", "manifest",
                       "external_root", "source_manifest", "training_annotations", "provenance", "dataset_yaml", "caption_subset",
                       "pg_scores", "base_directory", "export_directory", "tokenizer_directory", "scratch_root", "checkpoint_root",
                       "progress_path", "observation_cache_root"} and isinstance(item, str) and item.startswith("/"):
                result.append(Path(item))
            result.extend(path_values(item))
    elif isinstance(value, list):
        for item in value:
            result.extend(path_values(item))
    return result


def validate_runtime_source(config: dict, runtime_root: Path) -> None:
    source = config.get("runtime_source")
    commit = source.get("source_commit") if isinstance(source, dict) else None
    if not isinstance(commit, str) or len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
        raise RunnerError("runtime source commit is invalid")
    manifest = Path(source.get("manifest", ""))
    if not manifest.is_file() or sha256(manifest) != source.get("manifest_sha256"):
        raise RunnerError("runtime source manifest differs")
    document = json.loads(manifest.read_text(encoding="utf-8"))
    if document.get("source_commit") != commit or document.get("snapshot_root") != str(runtime_root):
        raise RunnerError("runtime source manifest does not bind this snapshot root")
    for relative, expected in document.get("files", {}).items():
        path = runtime_root / relative
        if not path.is_file() or sha256(path) != expected:
            raise RunnerError(f"runtime source file differs: {relative}")


def validate_config(path: Path, expected: str, runtime_root: Path) -> dict:
    if not path.is_absolute() or not path.is_file() or sha256(path) != expected:
        raise RunnerError("runtime config is absent or hash differs")
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema") != "nc_rted_production_runtime/v1":
        raise RunnerError("unsupported runtime config schema")
    if "teacher" in config or "fast" in config:
        raise RunnerError("caption preparation config must not include teacher or Fast")
    caption_cache = config.get("media", {}).get("caption_observation_cache", {})
    if caption_cache.get("max_bytes") != 23 * GIB or caption_cache.get("minimum_free_bytes") != 20 * GIB or caption_cache.get("feature_dtype") != "bfloat16":
        raise RunnerError("caption cache resource binding differs")
    missing = []
    for value in path_values(config):
        # Output roots are intentionally created only after all input assets pass.
        if value.name in {"caption-observation-cache", "observation-frame-cache", "bounded-stage2-scratch", "diagnostic-checkpoints", "progress.json"}:
            continue
        if not value.exists(): missing.append(str(value))
    if missing:
        raise RunnerError("referenced paths are absent: " + ", ".join(sorted(set(missing))))
    validate_runtime_source(config, runtime_root)
    root = Path(caption_cache["root"])
    if root.is_symlink() or shutil.disk_usage(root.parent).free < 20 * GIB:
        raise RunnerError("caption cache cannot preserve its 20 GiB free-space reserve")
    return config


def tensor_equal(left, right) -> bool:
    import torch
    if left.dtype != right.dtype or tuple(left.shape) != tuple(right.shape): return False
    if left.dtype.is_floating_point or left.dtype.is_complex:
        return bool(torch.allclose(left, right, rtol=0, atol=0, equal_nan=True))
    return bool(torch.equal(left, right))


def materials_equal(left, right) -> bool:
    if (left.context.image_sizes != right.context.image_sizes or left.context.observed_seconds != right.context.observed_seconds
            or left.context.sampled_frame_times != right.context.sampled_frame_times or left.context.time_message != right.context.time_message):
        return False
    tensors = (left.context.visual_embeddings, right.context.visual_embeddings,
               left.context.images[0], right.context.images[0],
               left.observations.features, right.observations.features,
               left.observations.valid, right.observations.valid,
               left.observations.observed_times, right.observations.observed_times)
    return all(tensor_equal(tensors[index], tensors[index + 1]) for index in range(0, len(tensors), 2))


def install_entry_recorder(provider):
    observer = provider.observer
    original = observer._provenance
    state: dict[str, str] = {}
    def recorded(*args, **kwargs):
        provenance, media = original(*args, **kwargs)
        state["entry"] = str(observer.cache._path(observer.cache.key(provenance)))
        state["provenance"] = provenance
        return provenance, media
    observer._provenance = recorded
    return state


def load_completed(events: Path, config_sha: str, cache) -> set[str]:
    completed = set()
    if not events.exists(): return completed
    for line in events.read_text(encoding="utf-8").splitlines():
        try: event = json.loads(line)
        except ValueError: continue
        if event.get("event") == "caption_complete" and event.get("config_sha256") == config_sha:
            sample_id, entry, provenance, expected = (event.get("sample_id"), event.get("cache_entry"),
                                                       event.get("provenance"), event.get("cache_file_sha256"))
            if not (isinstance(sample_id, str) and isinstance(entry, str) and isinstance(provenance, dict)
                    and isinstance(expected, str) and Path(entry).is_file() and sha256(Path(entry)) == expected):
                continue
            try:
                if cache.get(provenance, feature_dtype=__import__("torch").bfloat16) is not None:
                    completed.add(sample_id)
            except Exception:
                continue
    return completed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--runtime-source-root", required=True)
    parser.add_argument("--journal-root", required=True)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--smoke-sample-id", default="caption:ucf-crime:2913")
    args = parser.parse_args()
    if sum((args.smoke_only, args.full, args.preflight_only)) != 1:
        raise SystemExit("choose exactly one of --preflight-only, --smoke-only, or --full")
    manifest, runtime_root, journal = Path(args.manifest), Path(args.runtime_source_root), Path(args.journal_root)
    if not (manifest.is_absolute() and runtime_root.is_absolute() and journal.is_absolute()):
        raise SystemExit("manifest, runtime source root, and journal root must be absolute")
    if str(runtime_root / "src") not in sys.path: sys.path.insert(0, str(runtime_root / "src"))
    config = validate_config(manifest, args.manifest_sha256, runtime_root)
    journal.mkdir(parents=True, exist_ok=True)
    events = journal / "events.jsonl"
    run = {"schema": "nc_rted_caption_preparation_runner/v1", "config": str(manifest),
           "config_sha256": args.manifest_sha256, "runtime_source_root": str(runtime_root),
           "source_commit": config["runtime_source"]["source_commit"], "runner_sha256": sha256(Path(__file__).resolve())}
    run_path = journal / "run.json"
    if run_path.exists() and json.loads(run_path.read_text(encoding="utf-8")) != run:
        raise RunnerError("journal is bound to a different runner/config/source")
    write_json(run_path, run)
    append_event(events, {"event": "preflight_pass", "config_sha256": args.manifest_sha256,
                          "free_bytes": shutil.disk_usage(Path(config["media"]["caption_observation_cache"]["root"]).parent).free})
    from nc_rted.production_runtime import assemble_caption_preparation, load_caption_preparation_manifest
    bound = load_caption_preparation_manifest(manifest, expected_sha256=args.manifest_sha256)
    if args.preflight_only:
        print(json.dumps({"status": "PREFLIGHT_PASS", "config_sha256": args.manifest_sha256,
                          "source_commit": config["runtime_source"]["source_commit"],
                          "caption_cache_max_bytes": config["media"]["caption_observation_cache"]["max_bytes"],
                          "minimum_free_bytes": config["media"]["caption_observation_cache"]["minimum_free_bytes"]}))
        return
    runtime = assemble_caption_preparation(bound)
    provider = runtime.caption_provider
    recorder = install_entry_recorder(provider)
    if args.smoke_only:
        entries = Path(config["media"]["caption_observation_cache"]["root"]) / "entries"
        if any(entries.glob("*.pt")):
            raise RunnerError("cold smoke requires an empty caption observation cache")
        started = time.monotonic(); cold = provider(args.smoke_sample_id); cold_seconds = time.monotonic() - started
        started = time.monotonic(); warm = provider(args.smoke_sample_id); warm_seconds = time.monotonic() - started
        if not materials_equal(cold, warm) or not recorder.get("entry") or not Path(recorder["entry"]).is_file():
            raise RunnerError("cold/warm provider result or persistent observation binding differs")
        append_event(events, {"event": "cold_warm_smoke_pass", "config_sha256": args.manifest_sha256,
                              "sample_id": args.smoke_sample_id, "cache_entry": recorder["entry"],
                              "cold_seconds": cold_seconds, "warm_seconds": warm_seconds,
                              "observed_seconds": cold.context.observed_seconds,
                              "blocks": cold.observations.features.shape[1]})
        print(json.dumps({"status": "SMOKE_PASS", "sample_id": args.smoke_sample_id, "cold_seconds": cold_seconds, "warm_seconds": warm_seconds}))
        return
    completed = load_completed(events, args.manifest_sha256, provider.observer.cache)
    expected = sorted(provider._index)
    for sequence, sample_id in enumerate(expected, 1):
        if sample_id in completed: continue
        append_event(events, {"event": "caption_started", "config_sha256": args.manifest_sha256, "sample_id": sample_id, "sequence": sequence})
        started = time.monotonic()
        try:
            material = provider(sample_id)
            entry = recorder.get("entry")
            provenance = recorder.get("provenance")
            if not entry or not isinstance(provenance, dict) or not Path(entry).is_file():
                raise RunnerError("caption observation cache entry was not committed")
            append_event(events, {"event": "caption_complete", "config_sha256": args.manifest_sha256, "sample_id": sample_id,
                                  "sequence": sequence, "cache_entry": entry, "cache_file_sha256": sha256(Path(entry)),
                                  "provenance": provenance, "seconds": time.monotonic() - started,
                                  "observed_seconds": material.context.observed_seconds, "blocks": material.observations.features.shape[1]})
        except BaseException as error:
            append_event(events, {"event": "caption_failure", "config_sha256": args.manifest_sha256, "sample_id": sample_id,
                                  "sequence": sequence, "seconds": time.monotonic() - started, "error": repr(error)})
            raise
    append_event(events, {"event": "full_complete", "config_sha256": args.manifest_sha256, "requested": len(expected)})
    print(json.dumps({"status": "FULL_PASS", "requested": len(expected)}))


if __name__ == "__main__":
    try: main()
    except RunnerError as error:
        print(f"caption preparation runner failed: {error}", file=sys.stderr)
        raise SystemExit(2)
