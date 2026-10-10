from types import SimpleNamespace
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

import nc_rted.loading as loading
from nc_rted.prediction_runtime import _DefaultLoader


def test_default_loader_reaches_inherited_load_boundary_after_runtime_imports(monkeypatch):
    class LoadBoundary(Exception):
        pass

    captured = {}

    def stop_at_inherited_load(*args, **kwargs):
        captured["args"], captured["kwargs"] = args, kwargs
        raise LoadBoundary()

    monkeypatch.setattr(loading, "load_inherited_slow", stop_at_inherited_load)
    runtime = SimpleNamespace(inherited={
        "base_directory": "/base", "stage2_export": "/export",
        "stage2_export_hashes": {"config.json": "a"}, "stage2_export_sha256": "stage2",
    })
    artifact = SimpleNamespace(group="R0", training_identity=None)

    with pytest.raises(LoadBoundary):
        _DefaultLoader(runtime, "cuda:0").load(artifact)

    assert captured["args"] == ("/base", "/export")
    assert captured["kwargs"] == {
        "train_lora": False, "expected_hashes": {"config.json": "a"}, "device": "cuda:0",
    }


def test_verified_probe_bootstrap_reaches_loader_boundary_with_feature_dependencies(tmp_path):
    root = Path(__file__).resolve().parents[1]
    probe_path = root / "scripts" / "nc_rted_prediction_runtime_probe.py"
    spec = importlib.util.spec_from_file_location("verified_probe_manifest", probe_path)
    probe = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(probe)
    files = {relative: hashlib.sha256((root / relative).read_bytes()).hexdigest() for relative in probe._FILES}
    implementation = tmp_path / "implementation.json"
    implementation.write_text(json.dumps({"schema": "nc_rted_blind_prediction_implementation/v1", "root": str(root), "files": files}), encoding="utf-8")
    preflight = tmp_path / "preflight.json"
    preflight.write_text(json.dumps({"bindings": {"implementation": {"path": str(implementation), "sha256": hashlib.sha256(implementation.read_bytes()).hexdigest()}}}), encoding="utf-8")
    command = """
import importlib.util, json, sys
from pathlib import Path
from types import SimpleNamespace
root, preflight = map(Path, sys.argv[1:])
probe_path = root / 'scripts' / 'nc_rted_prediction_runtime_probe.py'
spec = importlib.util.spec_from_file_location('verified_probe', probe_path)
probe = importlib.util.module_from_spec(spec); spec.loader.exec_module(probe)
probe._bootstrap(str(preflight), __import__('hashlib').sha256(preflight.read_bytes()).hexdigest())
probe._api()
import nc_rted.loading as loading
from nc_rted.prediction_runtime import _DefaultLoader
class Boundary(Exception): pass
def stop(*args, **kwargs): raise Boundary()
loading.load_inherited_slow = stop
runtime = SimpleNamespace(inherited={'base_directory':'/base','stage2_export':'/export','stage2_export_hashes':{'config.json':'a'},'stage2_export_sha256':'stage2'})
artifact = SimpleNamespace(group='R0', training_identity=None)
try: _DefaultLoader(runtime, 'cuda:0').load(artifact)
except Boundary: print('LOADER_BOUNDARY')
else: raise AssertionError('loader did not reach boundary')
"""
    result = subprocess.run([sys.executable, "-c", command, str(root), str(preflight)], text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "LOADER_BOUNDARY"
