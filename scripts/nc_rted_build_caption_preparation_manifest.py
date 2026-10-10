#!/usr/bin/env python3
"""Build a hash-bound, caption-only observation-preparation manifest.

This tool is deliberately CPU-only.  It binds every logical caption path to
the frozen Stage2 resolver's raw source and original train-manifest index; it
does not decode video frames, construct a model, or start a preparation run.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import shutil
import sys
import tempfile
from types import SimpleNamespace
from typing import Any

from nc_rted.storage_lock import allocation_lock, ensure_directory


class BuildError(RuntimeError):
    pass


GIB = 1024 ** 3
MINIMUM_FREE_BYTES = 20 * GIB
PROBE_MAXIMUM_BYTES = 64 * 1024 ** 2
SAMPLING = {"local_num_frames": 1, "frames_upbound": 64, "frames_lowbound": 4,
            "sample_type": "dynamic_fps1", "time_msg": "short_online_v2",
            "model_max_length": 8192, "vision_chunk_size": 32, "projector": "original"}
EXPORT_HASHES = {
    "adapter_config.json": "7019ba36a10c61fc8663bd0bd1e914770358b765d46120113a5ed8eb8493c7cb",
    "adapter_model.safetensors": "69d757966addcd4abc8535c00097635e348bb7dd1dc55fe5b0e7a75e76a99c5d",
    "config.json": "12b1cd1e810c01d5bc5aac071f5fa99a33c42a7d67d8fc9083adbefe800a27b4",
    "non_lora_trainables.bin": "c1c65da4d24f0f3716c523e06c4342aea0868def178f20323c21f4e54d55421c",
}


def _absolute(value: str, name: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise BuildError(f"{name} must be absolute")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    if not root.is_dir() or root.is_symlink():
        raise BuildError(f"asset tree is unavailable or unsafe: {root}")
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise BuildError(f"asset tree contains a symlink: {path}")
        if path.is_file():
            digest.update(str(path.relative_to(root)).encode("utf-8"))
            digest.update(b"\0")
            digest.update(_sha256(path).encode("ascii"))
            digest.update(b"\n")
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> str:
    """Durably publish a new manifest without replacing an existing binding."""
    content = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with allocation_lock(path.parent):
        if path.exists():
            raise BuildError(f"refusing to overwrite published artifact: {path}")
        block = max(4096, os.statvfs(path.parent).f_frsize)
        allocation = ((len(content) + block - 1) // block) * block + 2 * block
        if shutil.disk_usage(path.parent).free < MINIMUM_FREE_BYTES + allocation:
            raise BuildError("manifest publication would violate the required free-space reserve")
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                # link(2) publishes atomically but cannot replace a competing file.
                os.link(temporary, path)
            except FileExistsError as error:
                raise BuildError(f"refusing to overwrite published artifact: {path}") from error
            except OSError as error:
                raise BuildError(f"could not publish manifest artifact: {path}") from error
            directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            temporary.unlink(missing_ok=True)
    return hashlib.sha256(content).hexdigest()


def _admit_probe_temporary(root: Path) -> None:
    block = max(4096, os.statvfs(root).f_frsize)
    if shutil.disk_usage(root).free < MINIMUM_FREE_BYTES + PROBE_MAXIMUM_BYTES + 2 * block:
        raise BuildError("segment-metadata probe would violate the required free-space reserve")


@contextmanager
def _frozen_probe_environment(serial, root: Path):
    """Constrain the frozen resolver's one-frame FPS writer probe."""
    ensure_directory(root, MINIMUM_FREE_BYTES)
    with allocation_lock(root):
        original_tempfile = serial.tempfile
        original_limit = resource.getrlimit(resource.RLIMIT_FSIZE)

        def guarded_mkstemp(*args, **kwargs):
            _admit_probe_temporary(root)
            kwargs["dir"] = str(root)
            return original_tempfile.mkstemp(*args, **kwargs)

        try:
            resource.setrlimit(resource.RLIMIT_FSIZE, (PROBE_MAXIMUM_BYTES, original_limit[1]))
            serial.tempfile = SimpleNamespace(mkstemp=guarded_mkstemp)
            yield
        finally:
            serial.tempfile = original_tempfile
            resource.setrlimit(resource.RLIMIT_FSIZE, original_limit)


def _expected_segment(serial, source: Path, segment: list[float], probe_root: Path) -> dict:
    with _frozen_probe_environment(serial, probe_root):
        return serial.expected_segment(source, segment)


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location("nc_rted_caption_stage2_serial", path)
    if spec is None or spec.loader is None:
        raise BuildError(f"cannot load frozen serial resolver: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _metadata(path: Path) -> tuple[float, int, int, int]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    if fps <= 0 or frames <= 0 or width <= 0 or height <= 0:
        raise BuildError(f"cannot read raw source metadata: {path}")
    return fps, frames, width, height


def _source_manifest(external_root: Path) -> dict[str, Any]:
    files: dict[str, str] = {}
    for directory in ("llava", "eval_utils", "vad"):
        root = external_root / directory
        if not root.is_dir():
            raise BuildError(f"ReactVAU source directory is absent: {root}")
        for path in sorted(root.rglob("*.py")):
            if path.is_symlink():
                raise BuildError(f"ReactVAU source path is a symlink: {path}")
            files[str(path.relative_to(external_root))] = _sha256(path)
    if not files:
        raise BuildError("ReactVAU source manifest would be empty")
    return {"schema": "nc_rted_reactvau_source_manifest/v1", "files": files}


def _runtime_source_manifest(snapshot_root: Path, expected_commit: str | None, github_commit: str | None) -> dict[str, Any]:
    snapshot = snapshot_root / "CODE_SNAPSHOT.json"
    if not snapshot.is_file():
        raise BuildError(f"published runtime snapshot is absent: {snapshot}")
    document = json.loads(snapshot.read_text(encoding="utf-8"))
    source_commit = document.get("source_commit")
    if (not isinstance(source_commit, str) or len(source_commit) != 40
            or any(char not in "0123456789abcdef" for char in source_commit)):
        raise BuildError("published runtime snapshot has an invalid source commit")
    if expected_commit is not None and source_commit != expected_commit:
        raise BuildError("published runtime snapshot does not bind the requested source commit")
    if github_commit is not None and (len(github_commit) != 40 or any(char not in "0123456789abcdef" for char in github_commit)):
        raise BuildError("requested GitHub commit is invalid")
    files = document.get("files")
    if isinstance(files, dict):
        files = [{"path": path, "sha256": digest} for path, digest in files.items()]
    if not isinstance(files, list):
        raise BuildError("published runtime snapshot has no file map")
    source: dict[str, str] = {}
    for item in files:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str) or not isinstance(item.get("sha256"), str):
            raise BuildError("published runtime snapshot has an invalid file entry")
        relative = item["path"]
        if not relative.startswith("src/nc_rted/"):
            continue
        path = snapshot_root / relative
        if not path.is_file() or _sha256(path) != item["sha256"]:
            raise BuildError(f"published runtime source differs: {relative}")
        source[relative] = item["sha256"]
    if not source:
        raise BuildError("published runtime snapshot has no nc_rted source")
    return {
        "schema": "nc_rted_runtime_source_manifest/v1",
        "source_commit": source_commit,
        "github_commit": github_commit,
        "snapshot_root": str(snapshot_root),
        "snapshot_sha256": _sha256(snapshot),
        "files": source,
    }


def _caption_media(captions_path: Path, stage2_config: dict, serial,
                   probe_root: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    captions = json.loads(captions_path.read_text(encoding="utf-8"))
    train = json.loads(Path(stage2_config["train_json"]).read_text(encoding="utf-8"))
    if not isinstance(captions, list) or len(captions) != 2000:
        raise BuildError("caption subset must contain exactly 2,000 rows")
    indices: dict[int | str, int] = {}
    for index, row in enumerate(train):
        identifier = row.get("id") if isinstance(row, dict) else None
        if identifier in indices:
            raise BuildError("frozen full train manifest has duplicate identities")
        indices[identifier] = index
    resolver = serial.FrozenTrainRequests(stage2_config)
    source_metadata: dict[Path, tuple[float, int, int, int]] = {}
    media: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    kinds: dict[str, int] = {"clips": 0, "events": 0, "videos": 0}
    for row in captions:
        if not isinstance(row, dict) or not isinstance(row.get("id"), (int, str)) or not isinstance(row.get("video"), str):
            raise BuildError("caption subset row lacks frozen identity")
        request_index = indices.get(row["id"])
        if request_index is None:
            raise BuildError(f"caption identity is absent from frozen full train manifest: {row['id']}")
        relative_video = row["video"]
        request = resolver.resolve(relative_video, request_index)
        source = Path(request["source"])
        if not source.is_file():
            raise BuildError(f"resolved raw source is absent: {source}")
        metadata = source_metadata.get(source)
        if metadata is None:
            metadata = _metadata(source)
            source_metadata[source] = metadata
        dataset = relative_video.split("/", 1)[0]
        identity = (dataset, relative_video)
        if identity in seen:
            raise BuildError(f"caption subset has duplicate logical media: {relative_video}")
        seen.add(identity)
        kind = request["kind"]
        if kind not in kinds:
            raise BuildError(f"unexpected frozen media kind: {kind}")
        kinds[kind] += 1
        if kind == "videos":
            fps, frame_count, width, height = metadata
        else:
            expected = _expected_segment(serial, source, request["segment"], probe_root)
            fps, frame_count = expected["fps"], expected["frames"]
            width, height = expected["width"], expected["height"]
        media.append({
            "dataset": dataset,
            "media_key": relative_video,
            "media_path": str(source),
            "media_sha256": request["source_sha256"],
            "fps": fps,
            "frame_count": frame_count,
            "height": height,
            "width": width,
            "request_index": request_index,
            "aliases": [],
        })
    if len(media) != 2000:
        raise BuildError("caption media catalog does not contain 2,000 rows")
    return media, {"unique_sources": len(source_metadata), **kinds}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--runtime-source-root", required=True,
                        help="published v12 snapshot, not the builder worktree")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run-id", default="diagnostic:caption-preparation-2000-v1")
    parser.add_argument("--source-commit", help="optional expected commit; validated against CODE_SNAPSHOT.json")
    parser.add_argument("--github-commit", help="optional verified 40-hex GitHub commit")
    args = parser.parse_args()

    project_root = _absolute(args.project_root, "project root").resolve()
    snapshot_root = _absolute(args.runtime_source_root, "runtime source root").resolve()
    output = _absolute(args.output_dir, "output directory")
    if not project_root.is_dir() or output.exists():
        raise BuildError("project root is absent or output directory already exists")
    try:
        output.resolve(strict=False).relative_to(project_root)
    except ValueError as error:
        raise BuildError("output directory must be below project root") from error

    subset = project_root / "artifacts/nc_rted/caption_subset_candidate_v1/train_captions2000.json"
    dataset_yaml = project_root / "artifacts/nc_rted/caption_subset_candidate_v1/dataset.yaml"
    pg_scores = project_root / "artifacts/reactvau/fast_selected_best6000_partial97140_recovery_v6_20261003/pg_scores_hivau_train.json"
    annotations = project_root / "data/reactvau/derived/instruction_available_97140_20260930/train.json"
    manifest_directory = project_root / "artifacts/nc_rted/manifest_candidate_causal_v6"
    provenance = manifest_directory / "provenance.json"
    external = project_root / "external/ReactVAU-paper"
    base = project_root / "models/reactvau/StreamForest-Qwen2-7B"
    export = project_root / "artifacts/reactvau/stage2_bf16_dual_ddp_full_zero3_cpuoffload_ranked_ninja_envfix_detached_resume_20261004"
    raw_siglip = project_root / "models/reactvau/siglip-so400m-patch14-384"
    final_siglip = project_root / "models/reactvau/nc_rted_final_stage2_siglip_v2"
    rtdetr = project_root / "models/nc_rted/rtdetr_r50vd_coco_o365/457857cec8ac28ddede40ecee9eed2beca321af8-v2"
    stage2_config = project_root / "configs/reactvau/STAGE2_EPHEMERAL_CACHE_FULL_ZERO3_EXECUTION_20261004.json"
    resolver_path = project_root / "scripts/reactvau_stage2_cache_serial.py"
    required = (subset, dataset_yaml, pg_scores, annotations, provenance, external, base, export,
                raw_siglip, final_siglip, rtdetr, stage2_config, resolver_path)
    absent = [str(path) for path in required if not path.exists()]
    if absent:
        raise BuildError("required input is absent: " + ", ".join(absent))
    if any(_sha256(export / name) != expected for name, expected in EXPORT_HASHES.items()):
        raise BuildError("known Stage2 export hashes differ")

    stage2 = json.loads(stage2_config.read_text(encoding="utf-8"))
    if stage2.get("status") != "APPROVED_FOR_EXECUTION":
        raise BuildError("frozen Stage2 cache is not approved")
    serial = _load_module(resolver_path)

    output.mkdir(parents=True)
    media, inventory = _caption_media(subset, stage2, serial, output / "segment-metadata-probes")
    catalog_sha = _write_json(output / "caption_media_catalog.json", {"schema": "nc_rted_caption_media_catalog/v1", "media": media})
    reactvau_sha = _write_json(output / "reactvau_source_manifest.json", _source_manifest(external))
    runtime_source = _runtime_source_manifest(snapshot_root, args.source_commit, args.github_commit)
    runtime_source_sha = _write_json(output / "runtime_source_manifest.json", runtime_source)

    source_manifest = output / "reactvau_source_manifest.json"
    bounded_scratch = output / "bounded-stage2-scratch"
    manifest: dict[str, Any] = {
        "schema": "nc_rted_production_runtime/v1",
        "run": {"run_id": args.run_id, "group": "A", "seed": 20261010, "device": args.device,
                "checkpoint_root": str(output / "diagnostic-checkpoints"), "progress_path": str(output / "progress.json"),
                "mode": "diagnostic", "diagnostic_updates": 1},
        "hashes": {"code_sha256": runtime_source_sha, "runtime_sha256": runtime_source_sha,
                   "inherited_weights_sha256": _sha256(export / "non_lora_trainables.bin")},
        "runtime_source": {"manifest": str(output / "runtime_source_manifest.json"), "manifest_sha256": runtime_source_sha,
                           "source_commit": runtime_source["source_commit"], "github_commit": runtime_source["github_commit"]},
        "sampling": SAMPLING,
        "catalog": {"manifest_directory": str(manifest_directory), "training_annotations": str(annotations),
                    "training_annotations_sha256": _sha256(annotations), "provenance": str(provenance),
                    "provenance_sha256": _sha256(provenance), "dataset_yaml": str(dataset_yaml),
                    "dataset_yaml_sha256": _sha256(dataset_yaml), "caption_subset": str(subset),
                    "caption_subset_sha256": _sha256(subset), "pg_scores": str(pg_scores), "pg_scores_sha256": _sha256(pg_scores)},
        "inherited": {"external_root": str(external), "source_manifest": str(source_manifest),
                      "source_manifest_sha256": reactvau_sha, "base_directory": str(base),
                      "base_directory_sha256": _tree_sha256(base), "export_directory": str(export),
                      "tokenizer_directory": str(base), "tokenizer_sha256": _tree_sha256(base),
                      "export_hashes": EXPORT_HASHES},
        "stage2_cache": {"mode": "bounded", "module": str(resolver_path), "module_sha256": _sha256(resolver_path),
                         "expected_resolver_sha256": _sha256(resolver_path), "config": str(stage2_config),
                         "config_sha256": _sha256(stage2_config), "accepted_status": "APPROVED_FOR_EXECUTION",
                         "scratch_root": str(bounded_scratch), "minimum_free_bytes": 20 * GIB,
                         "overhead_bytes": 16 * 1024 ** 2, "max_temporary_bytes": 4 * GIB,
                         "child_timeout_seconds": 600},
        "media": {"catalog": str(output / "caption_media_catalog.json"), "catalog_sha256": catalog_sha,
                  "observation_cache_root": str(output / "observation-frame-cache"), "observation_cache_max_bytes": 4 * GIB,
                  "caption_observation_cache": {"root": str(output / "caption-observation-cache"), "max_bytes": 23 * GIB,
                                                "minimum_free_bytes": 20 * GIB, "feature_dtype": "bfloat16"}},
        "detector": {"snapshot": str(rtdetr), "snapshot_sha256": _tree_sha256(rtdetr),
                     "siglip_snapshot": str(raw_siglip), "siglip_snapshot_sha256": _tree_sha256(raw_siglip),
                     "final_stage2_siglip_snapshot": str(final_siglip), "final_stage2_siglip_snapshot_sha256": _tree_sha256(final_siglip),
                     "score_threshold": 0.3},
        "builder": {"schema": "nc_rted_caption_preparation_builder/v1", "gpu_launched": False,
                    "provider_report": str(project_root / "reports/nc_rted/real_caption_gpu_pro6000_v6.json"),
                    "provider_report_sha256": _sha256(project_root / "reports/nc_rted/real_caption_gpu_pro6000_v6.json"),
                    "inventory": inventory},
    }
    runtime_sha = _write_json(output / "caption_preparation_runtime.json", manifest)
    report = {"schema": "nc_rted_caption_preparation_build_report/v1", "status": "BUILT_CPU_ONLY",
              "runtime_config": str(output / "caption_preparation_runtime.json"), "runtime_config_sha256": runtime_sha,
              "catalog": str(output / "caption_media_catalog.json"), "catalog_sha256": catalog_sha,
              "reactvau_source_manifest_sha256": reactvau_sha, "runtime_source_manifest_sha256": runtime_source_sha,
              "source_commit": runtime_source["source_commit"], "github_commit": runtime_source["github_commit"], "inventory": inventory,
              "caption_cache_max_bytes": 23 * GIB, "minimum_free_bytes": 20 * GIB, "gpu_launched": False}
    _write_json(output / "build_report.json", report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except BuildError as error:
        print(f"caption preparation manifest build failed: {error}", file=sys.stderr)
        raise SystemExit(2)
