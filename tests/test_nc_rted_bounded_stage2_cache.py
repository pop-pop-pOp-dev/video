import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from nc_rted.bounded_stage2_cache import BoundedStage2Cache, BoundedStage2CacheError, _ValidatorTemporaryGuard


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _asset(root: Path, name: str, contents: bytes) -> tuple[Path, str]:
    path = root / name
    path.write_bytes(contents)
    return path, digest(path)


def bounded_stage2(tmp_path: Path) -> tuple[dict, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    source, source_sha = _asset(tmp_path, "source.mp4", b"raw-media")
    train, train_sha = _asset(tmp_path, "train.json", json.dumps([
        {"id": 1, "video": "ucf-crime/videos/train/sample.mp4"},
        {"id": 2, "video": "ucf-crime/events/train/sample_E0.mp4"},
    ]).encode())
    frozen = {"train_json": (train, train_sha)}
    for name in ("ucf_database", "xd_database", "identity_map", "official_splitter"):
        frozen[name] = _asset(tmp_path, f"{name}.json", name.encode())
    config = {"status": "APPROVED_FOR_EXECUTION", "source": str(source), "source_sha256": source_sha}
    for name, (path, value) in frozen.items():
        config[name] = str(path)
        config[f"{name}_sha256"] = value
    config_path, config_sha = _asset(tmp_path, "resolver.json", json.dumps(config, sort_keys=True).encode())
    resolver = tmp_path / "resolver.py"
    resolver.write_text('''
import json
from pathlib import Path
import tempfile
FROZEN = (("train_json", "train_json_sha256"), ("ucf_database", "ucf_database_sha256"), ("xd_database", "xd_database_sha256"), ("identity_map", "identity_map_sha256"), ("official_splitter", "official_splitter_sha256"))
class FrozenTrainRequests:
    def __init__(self, config): self.config = config
    def resolve(self, relative, index=None):
        rows = json.loads(Path(self.config["train_json"]).read_text())
        if type(index) is not int or index >= len(rows) or rows[index]["video"] != relative:
            raise ValueError("request index does not bind frozen path")
        return {"relative": relative, "kind": "videos" if "/videos/" in relative else "events", "source": Path(self.config["source"]), "source_sha256": self.config["source_sha256"], "segment": [0, 1], "indices": [index]}
def full_decode(path): return {"sha256": "x", "bytes": Path(path).stat().st_size, "frames": 1, "fps": 1.0}
def validate_segment_output(meta, path, source, segment): return None
''', encoding="utf-8")
    stage2 = {
        "module": str(resolver), "module_sha256": digest(resolver),
        "expected_resolver_sha256": digest(resolver), "config": str(config_path), "config_sha256": config_sha,
        "accepted_status": "APPROVED_FOR_EXECUTION", "scratch_root": str(tmp_path / "scratch"),
        "minimum_free_bytes": 20 * 1024 ** 3, "overhead_bytes": 1, "max_temporary_bytes": 1024,
    }
    return stage2, source


def test_direct_video_lease_uses_original_source_without_copying(tmp_path):
    stage2, source = bounded_stage2(tmp_path)
    cache = BoundedStage2Cache(stage2)
    annotation = {"id": 1, "video": "ucf-crime/videos/train/sample.mp4",
                  "_reactvau_relative_video": "ucf-crime/videos/train/sample.mp4"}
    index = cache.request_index_for(annotation)
    with cache.acquire(annotation["video"], index) as leased:
        assert leased == source
        assert leased.read_bytes() == b"raw-media"
    assert not list(Path(stage2["scratch_root"]).glob("*.mp4"))


def test_derived_lease_uses_original_validation_and_releases_success(tmp_path, monkeypatch):
    stage2, _ = bounded_stage2(tmp_path)
    cache = BoundedStage2Cache(stage2)
    monkeypatch.setattr(cache, "_run_child", lambda request, output, maximum: (
        output.write_bytes(b"derived-media") and SimpleNamespace(returncode=0)))
    annotation = {"id": 2, "video": "ucf-crime/events/train/sample_E0.mp4",
                  "_reactvau_relative_video": "ucf-crime/events/train/sample_E0.mp4"}
    index = cache.request_index_for(annotation)
    with cache.acquire(annotation["video"], index) as leased:
        assert leased.exists()
        assert leased.read_bytes() == b"derived-media"
    scratch = Path(stage2["scratch_root"])
    assert not list(scratch.glob("*.partial.mp4"))
    assert '"event": "prepared"' in cache.events_path.read_text()
    assert '"event": "released"' in cache.events_path.read_text()


def test_derived_consumer_failure_preserves_media_and_audit(tmp_path, monkeypatch):
    stage2, _ = bounded_stage2(tmp_path)
    cache = BoundedStage2Cache(stage2)
    monkeypatch.setattr(cache, "_run_child", lambda request, output, maximum: (
        output.write_bytes(b"derived-media") and SimpleNamespace(returncode=0)))
    with pytest.raises(ValueError, match="decoder failed"):
        with cache.acquire("ucf-crime/events/train/sample_E0.mp4", 1):
            raise ValueError("decoder failed")
    assert list(Path(stage2["scratch_root"]).glob("*.failed"))
    assert '"event": "consumer_failure"' in cache.events_path.read_text()


def test_rejects_request_mismatch_source_change_and_reserve_violation(tmp_path, monkeypatch):
    stage2, source = bounded_stage2(tmp_path)
    cache = BoundedStage2Cache(stage2)
    with pytest.raises(BoundedStage2CacheError, match="request index"):
        with cache.acquire("ucf-crime/videos/train/sample.mp4", 1):
            pass
    source.write_bytes(b"changed")
    with pytest.raises(BoundedStage2CacheError, match="source SHA"):
        with cache.acquire("ucf-crime/videos/train/sample.mp4", 0):
            pass
    stage2, _ = bounded_stage2(tmp_path / "second")
    cache = BoundedStage2Cache(stage2)
    monkeypatch.setattr(cache, "_available_bytes", lambda: cache.minimum_free_bytes + cache.overhead_bytes)
    with pytest.raises(BoundedStage2CacheError, match="reserve"):
        with cache.acquire("ucf-crime/events/train/sample_E0.mp4", 1):
            pass


def test_rejects_invalid_child_memory_and_restores_validator_module_on_admission_failure(tmp_path):
    stage2, _ = bounded_stage2(tmp_path)
    config_path = Path(stage2["config"])
    config = json.loads(config_path.read_text())
    config["child_memory_bytes"] = 0
    config_path.write_text(json.dumps(config))
    stage2["config_sha256"] = digest(config_path)
    with pytest.raises(BoundedStage2CacheError, match="child memory"):
        BoundedStage2Cache(stage2)
    original_tempfile = object()
    cache = SimpleNamespace(serial=SimpleNamespace(tempfile=original_tempfile),
                            _admit_temporary=lambda: (_ for _ in ()).throw(BoundedStage2CacheError("reserve")))
    with pytest.raises(BoundedStage2CacheError, match="reserve"):
        _ValidatorTemporaryGuard(cache).__enter__()
    assert cache.serial.tempfile is original_tempfile


def test_rejects_executable_mutation_after_construction_and_outside_scratch(tmp_path):
    stage2, _ = bounded_stage2(tmp_path)
    cache = BoundedStage2Cache(stage2)
    Path(stage2["module"]).write_text("changed")
    with pytest.raises(BoundedStage2CacheError, match="resolver SHA"):
        cache._run_child({}, tmp_path / "unused.mp4", 1)
    stage2, _ = bounded_stage2(tmp_path / "outside")
    stage2["scratch_root"] = "/root/autodl-tmp/nc-rted-caption-runtime/bounded-stage2-test"
    with pytest.raises(BoundedStage2CacheError, match="approved data volume"):
        BoundedStage2Cache(stage2)
