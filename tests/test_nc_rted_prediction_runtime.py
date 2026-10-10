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


def test_verified_probe_bootstrap_returns_r0_loader_after_route_construction(tmp_path):
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
    command = r"""
import hashlib, importlib.util, json, sys, types
from pathlib import Path
from types import SimpleNamespace
import torch
root, preflight, scratch = map(Path, sys.argv[1:])
spec = importlib.util.spec_from_file_location('verified_probe', root / 'scripts' / 'nc_rted_prediction_runtime_probe.py')
probe = importlib.util.module_from_spec(spec); spec.loader.exec_module(probe)
probe._bootstrap(str(preflight), hashlib.sha256(preflight.read_bytes()).hexdigest())
finder = next(item for item in sys.meta_path if item.__class__.__name__ == 'Finder')
original_find_spec = finder.__class__.find_spec
def traced_find_spec(self, fullname, path=None, target=None):
    try: return original_find_spec(self, fullname, path, target)
    except probe.ProbeError:
        print('GUARD_REJECTED:' + fullname, file=sys.stderr)
        raise
finder.__class__.find_spec = traced_find_spec
probe._api()
import nc_rted.detector as detector
import nc_rted.detection_provider as provider
import nc_rted.loading as loading
from nc_rted.numerics import deterministic_policy
import nc_rted.prediction_runtime as runtime
class Tower(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.config = SimpleNamespace(hidden_size=4, num_attention_heads=1); self.is_loaded = True
class Raw(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.config = SimpleNamespace(hidden_size=4); self.embedding = torch.nn.Embedding(8, 4); self.projector = torch.nn.Linear(4, 4); self.tower = Tower()
    def get_input_embeddings(self): return self.embedding
    def get_model(self): return SimpleNamespace(mm_projector=SimpleNamespace(parameters=self.projector.parameters, mlp=SimpleNamespace(parameters=self.projector.parameters)))
    def get_vision_tower(self): return self.tower
class Siglip:
    def __init__(self, *args, **kwargs): pass
    def __call__(self, values): return [torch.zeros(1, 4) for _ in values]
class Fast:
    image_size = 8
    def batch_score_grids(self, grids, prompt): return [0.0 for _ in grids]
class Protocol:
    def __init__(self, **values): self.__dict__.update(values)
    def validate(self): pass
detector.InheritedSigLipAdapter = Siglip
provider.DetectionProtocol = Protocol
loading.load_inherited_slow = lambda *args, **kwargs: (Raw(), {'mocked_inherited_load': True})
runtime._fast_detector = lambda *args, **kwargs: Fast()
runtime._tokenizer = lambda value: (SimpleNamespace(), SimpleNamespace(), SimpleNamespace(conv_templates={}), lambda *args, **kwargs: None, '<image>', -200)
runtime._audit_bound_inherited_modules = lambda *args, **kwargs: None
eval_utils = types.ModuleType('eval_utils'); vad = types.ModuleType('eval_utils.vad'); detect_utils = types.ModuleType('eval_utils.vad.detect_utils')
class Smoother:
    def __init__(self, **kwargs): pass
detect_utils.OnlineSmoother = Smoother
sys.modules.update({'eval_utils': eval_utils, 'eval_utils.vad': vad, 'eval_utils.vad.detect_utils': detect_utils})
raw_snapshot = scratch / 'raw'; raw_snapshot.mkdir(); (raw_snapshot / 'config.json').write_text('{}', encoding='utf-8')
policy = scratch / 'policy.txt'; policy.write_text(deterministic_policy().identity(), encoding='utf-8')
catalog = scratch / 'catalog.json'; catalog.write_text(json.dumps({'schema': 'nc_rted_blind_media_catalog/v1', 'media': []}), encoding='utf-8')
fast_snapshot = scratch / 'fast.json'; fast_snapshot.write_text(json.dumps({'schema': 'nc_rted_blind_fast_snapshot/v1', 'media': []}), encoding='utf-8')
source_manifest = scratch / 'source.json'; source_manifest.write_text(json.dumps({'files': {}}), encoding='utf-8')
document = {'numerics': {'policy': str(policy)}, 'vision': {'derived_snapshot': str(scratch), 'parent_export_sha256': 'parent', 'parent_export': str(scratch / 'parent.bin'), 'raw_snapshot': str(raw_snapshot)}, 'detector': {'snapshot': str(scratch), 'score_threshold': 0.5}, 'media': {'catalog': str(catalog)}, 'cache': {'root': str(scratch / 'cache'), 'max_bytes': 1}, 'fast': {'snapshot': str(fast_snapshot), 'model': 'model', 'lora': None, 'attn_implementation': 'eager', 'image_size': 8, 'streamforest_weights': 'weights', 'vision_feature_layer': -1}, 'protocols': {'vad': {'ucf': {}, 'xd': {}}, 'vad_config': {'yes_token_ids': [1], 'no_token_ids': [2], 'fusion': 'replace', 'fusion_alpha': 0.0, 'online_smooth_alpha': 0.1, 'online_smooth_beta': 0.1}, 'hivau': {'target_fps': 4, 'query_interval': 4, 'paligemma_batch_size': 1, 'max_new_tokens': 1}}, 'generation': {'do_sample': False}, 'inherited': {'base_directory': '/base', 'stage2_export': '/export', 'stage2_export_hashes': {'config.json': 'a'}, 'external_root': '/external', 'source_manifest': str(scratch / 'source.json')}}
model = runtime._DefaultLoader(SimpleNamespace(document=document, inherited=document['inherited']), 'cpu').load(SimpleNamespace(group='R0', training_identity=None, checkpoint=None, seed=None, evidence_enabled=False))
assert model.group == 'R0' and model.vad_detector is not None and model.hivau_inference is not None
print('LOADER_RETURNED')
"""
    result = subprocess.run([sys.executable, "-c", command, str(root), str(preflight), str(tmp_path)], text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "LOADER_RETURNED"
