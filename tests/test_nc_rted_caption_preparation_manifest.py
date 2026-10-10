import importlib.util
from contextlib import contextmanager
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


def _builder_module():
    path = Path(__file__).resolve().parents[1] / "scripts/nc_rted_build_caption_preparation_manifest.py"
    spec = importlib.util.spec_from_file_location("caption_preparation_builder", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_manifest_publication_is_durable_and_never_overwrites(tmp_path):
    builder = _builder_module()
    catalog = tmp_path / "caption_media_catalog.json"
    digest = builder._write_json(catalog, {"catalog": "bound"})
    assert catalog.is_file()
    assert digest == builder._sha256(catalog)
    with pytest.raises(builder.BuildError, match="refusing to overwrite"):
        builder._write_json(catalog, {"catalog": "replacement"})
    assert catalog.read_text(encoding="utf-8") == '{\n  "catalog": "bound"\n}\n'


@pytest.mark.parametrize("name", ["caption_media_catalog.json", "caption_preparation_runtime.json"])
def test_interrupted_manifest_publication_leaves_no_final_artifact(tmp_path, monkeypatch, name):
    builder = _builder_module()
    destination = tmp_path / name
    monkeypatch.setattr(builder.os, "link", lambda source, target: (_ for _ in ()).throw(OSError("interrupted")))
    with pytest.raises(builder.BuildError, match="could not publish"):
        builder._write_json(destination, {"binding": name})
    assert not destination.exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_manifest_publication_checks_reserve_before_creating_temporary(tmp_path, monkeypatch):
    builder = _builder_module()
    monkeypatch.setattr(builder.shutil, "disk_usage", lambda path: SimpleNamespace(free=builder.MINIMUM_FREE_BYTES))
    monkeypatch.setattr(builder.tempfile, "mkstemp", lambda *args, **kwargs: pytest.fail("temporary created before reserve admission"))
    with pytest.raises(builder.BuildError, match="free-space reserve"):
        builder._write_json(tmp_path / "build_report.json", {"complete": True})


def test_manifest_publication_holds_allocation_lock_through_temporary_creation(tmp_path, monkeypatch):
    builder = _builder_module()
    state = {"locked": False}
    @contextmanager
    def locked(path):
        assert not state["locked"]
        state["locked"] = True
        try: yield
        finally: state["locked"] = False
    original = builder.tempfile.mkstemp
    def checked_mkstemp(*args, **kwargs):
        assert state["locked"]
        return original(*args, **kwargs)
    monkeypatch.setattr(builder, "allocation_lock", locked)
    monkeypatch.setattr(builder.tempfile, "mkstemp", checked_mkstemp)
    builder._write_json(tmp_path / "runtime_source_manifest.json", {"source": "bound"})
    assert not state["locked"]


def test_expected_segment_probe_is_reserve_bound_and_confined(tmp_path, monkeypatch):
    builder = _builder_module()
    source = tmp_path / "source.mp4"; source.write_bytes(b"source")
    probe_root = tmp_path / "approved-probes"
    class Serial:
        tempfile = __import__("tempfile")
        @staticmethod
        def expected_segment(source, segment):
            descriptor, name = Serial.tempfile.mkstemp(suffix=".mp4")
            os.close(descriptor)
            path = Path(name)
            try:
                assert path.parent == probe_root
                path.write_bytes(b"probe")
            finally:
                path.unlink(missing_ok=True)
            return {"frames": 1, "fps": 1., "width": 1, "height": 1}
    assert builder._expected_segment(Serial, source, [0., 1.], probe_root)["frames"] == 1
    assert not list(probe_root.glob("*.mp4"))
    monkeypatch.setattr(builder.shutil, "disk_usage", lambda path: SimpleNamespace(free=builder.MINIMUM_FREE_BYTES))
    with pytest.raises(builder.BuildError, match="probe would violate"):
        builder._expected_segment(Serial, source, [0., 1.], probe_root)
