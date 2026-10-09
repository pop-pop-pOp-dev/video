from pathlib import Path
import hashlib
import json
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nc_rted.fast_snapshot import FastSnapshotError, build_snapshot, write_snapshot


def _write(path, value):
    path.write_text(json.dumps(value))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inputs(tmp_path, *, interval=7, scores=(0.0408, 0.0185), fps=30.0, frames=50):
    cache = tmp_path / "pg_scores_hivau_train.json"
    metadata = tmp_path / "media.json"
    cache_hash = _write(cache, {"ucf-crime/clips/train/Abuse001_x264_E0C0.mp4": {"pg_scores": list(scores), "sample_interval": interval}})
    metadata_hash = _write(metadata, {"media": [{"dataset": "ucf-crime", "media_key": "Abuse001_x264", "fast_cache_key": "ucf-crime/clips/train/Abuse001_x264_E0C0.mp4",
                                                    "media_path": "/bound/train/Abuse001.mp4", "media_sha256": "a" * 64,
                                                    "fps": fps, "frame_count": frames, "height": 240, "width": 320}]})
    return cache, cache_hash, metadata, metadata_hash


def test_exports_actual_pg_cache_entry_shape_without_score_requantization(tmp_path):
    cache, cache_hash, metadata, metadata_hash = _inputs(tmp_path)
    document = build_snapshot(fast_cache=cache, fast_cache_sha256=cache_hash, media_metadata=metadata,
                              media_metadata_sha256=metadata_hash, fast_identity={"checkpoint": "sha256:checkpoint", "implementation": "sha256:code"})
    row = document["media"][0]
    assert document["schema"] == "nc_rted_frozen_fast/v1"
    assert row["queries"][0] == {"index": 0, "frame_indices": [0, 7, 14, 21], "fast_score": 0.0408}
    assert row["queries"][1]["frame_indices"] == [28, 35, 42, 49]
    assert row["queries"][1]["fast_score"] == 0.0185


def test_refuses_unevidenced_interval_or_incomplete_scores(tmp_path):
    cache, cache_hash, metadata, metadata_hash = _inputs(tmp_path, interval=6)
    with pytest.raises(FastSnapshotError, match="sample_interval"):
        build_snapshot(fast_cache=cache, fast_cache_sha256=cache_hash, media_metadata=metadata,
                       media_metadata_sha256=metadata_hash, fast_identity={"checkpoint": "x", "implementation": "y"})
    cache, cache_hash, metadata, metadata_hash = _inputs(tmp_path, scores=(.1,))
    with pytest.raises(FastSnapshotError, match="score count"):
        build_snapshot(fast_cache=cache, fast_cache_sha256=cache_hash, media_metadata=metadata,
                       media_metadata_sha256=metadata_hash, fast_identity={"checkpoint": "x", "implementation": "y"})


def test_atomic_writer_refuses_overwrite_and_cli_dry_run(tmp_path):
    cache, cache_hash, metadata, metadata_hash = _inputs(tmp_path)
    document = build_snapshot(fast_cache=cache, fast_cache_sha256=cache_hash, media_metadata=metadata,
                              media_metadata_sha256=metadata_hash, fast_identity={"checkpoint": "x", "implementation": "y"})
    target = tmp_path / "snapshot.json"
    digest = write_snapshot(document, target, reserved_free_bytes=0)
    assert digest == hashlib.sha256(target.read_bytes()).hexdigest()
    with pytest.raises(FastSnapshotError, match="overwrite"):
        write_snapshot(document, target, reserved_free_bytes=0)
    script = Path(__file__).resolve().parents[1] / "scripts" / "nc_rted_export_fast_snapshot.py"
    result = subprocess.run([sys.executable, str(script), "--fast-cache", str(cache), "--fast-cache-sha256", cache_hash,
                             "--media-metadata", str(metadata), "--media-metadata-sha256", metadata_hash,
                             "--fast-identity-json", '{"checkpoint":"x","implementation":"y"}', "--dry-run"],
                            env={**__import__("os").environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")},
                            check=True, text=True, capture_output=True)
    assert json.loads(result.stdout)["status"] == "DRY_RUN_MATCHED"


def test_local_training_pg_cache_has_the_bound_export_fields_when_available():
    path = Path("/root/autodl-tmp/lookaway-wm/artifacts/reactvau/fast_selected_best6000_partial97140_recovery_v6_20261003/pg_scores_hivau_train.json")
    if not path.is_file():
        pytest.skip("local training Fast artifact is unavailable")
    entry = next(iter(json.loads(path.read_text()).values()))
    assert isinstance(entry["sample_interval"], int) and entry["sample_interval"] > 0
    assert isinstance(entry["pg_scores"], list) and entry["pg_scores"]
    assert all(isinstance(score, (int, float)) and 0 <= score <= 1 for score in entry["pg_scores"])
