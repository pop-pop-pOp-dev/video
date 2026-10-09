"""Regression coverage for FrozenRTDetr's source-to-loaded-state binding."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nc_rted.detector import DetectorError, FrozenRTDetr, RTDETR_PROVENANCE_FILE


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_staged_copy_reserves_block_allocation_before_creating_file(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import nc_rted.detector as module
    source=tmp_path/'source';source.write_bytes(b'verified bytes')
    destination=tmp_path/'destination'
    monkeypatch.setattr(module.shutil,'disk_usage',lambda path:SimpleNamespace(free=(20<<30)+1))
    with pytest.raises(module.DetectorError,match='free-space reserve'):
        module._copy_verified(source,destination,_sha256(source))
    assert not destination.exists()


def _snapshot(path: Path, value: float) -> None:
    from safetensors.torch import save_file

    path.mkdir()
    config = {
        "model_type": "rt_detr",
        "num_labels": 80,
        "id2label": {"0": "person"},
        "backbone_config": {"depths": [3, 4, 6, 3]},
    }
    processor = {"size": {"height": 640, "width": 640}, "do_resize": True}
    (path / "config.json").write_text(json.dumps(config))
    (path / "preprocessor_config.json").write_text(json.dumps(processor))
    save_file({"linear.weight": torch.tensor([[value]], dtype=torch.float32)}, str(path / "model.safetensors"))
    files = {name: _sha256(path / name) for name in ("config.json", "preprocessor_config.json", "model.safetensors")}
    (path / RTDETR_PROVENANCE_FILE).write_text(json.dumps({
        "model_id": "PekingU/rtdetr_r50vd_coco_o365",
        "architecture": "rtdetr_r50vd",
        "num_labels": 80,
        "person_class_id": 0,
        "files": files,
    }))


def _install_fake_transformers(monkeypatch, *, swap_source: Path | None = None,
                               replacement_weight: float | None = None, loaded_weight: float | None = None):
    from safetensors import safe_open
    from safetensors.torch import save_file

    class FakeConfig:
        def __init__(self, document):
            self.document = document
            self.model_type = document["model_type"]
            self.num_labels = document["num_labels"]
            self.id2label = {int(key): value for key, value in document["id2label"].items()}
            self.backbone_config = document["backbone_config"]

        @classmethod
        def from_dict(cls, document):
            return cls(document)

        def to_dict(self):
            return self.document

    class FakeProcessor:
        def __init__(self, document):
            self.document = document

        @classmethod
        def from_pretrained(cls, path, local_files_only):
            result = cls(json.loads((Path(path) / "preprocessor_config.json").read_text()))
            if swap_source is not None:
                replacement = swap_source / ".replacement.safetensors"
                save_file({"linear.weight": torch.tensor([[replacement_weight]], dtype=torch.float32)}, str(replacement))
                replacement.replace(swap_source / "model.safetensors")
            return result

        @classmethod
        def from_dict(cls, document):
            return cls(document)

        def to_dict(self):
            return self.document

    class FakeModel(torch.nn.Module):
        def __init__(self, document, state):
            super().__init__()
            self.config = FakeConfig(document)
            self.linear = torch.nn.Linear(1, 1, bias=False)
            self.load_state_dict(state, strict=True)

        @classmethod
        def from_pretrained(cls, path, local_files_only, use_safetensors):
            assert local_files_only and use_safetensors
            root = Path(path)
            with safe_open(str(root / "model.safetensors"), framework="pt", device="cpu") as source:
                state = {name: source.get_tensor(name) for name in source.keys()}
            if loaded_weight is not None:
                state["linear.weight"] = torch.tensor([[loaded_weight]], dtype=torch.float32)
            return cls(json.loads((root / "config.json").read_text()), state)

    module = types.ModuleType("transformers")
    module.__version__ = "test"
    module.RTDetrConfig = FakeConfig
    module.RTDetrImageProcessor = FakeProcessor
    module.RTDetrForObjectDetection = FakeModel
    monkeypatch.setitem(sys.modules, "transformers", module)


@pytest.mark.skipif(importlib.util.find_spec("safetensors") is None, reason="requires safetensors")
def test_frozen_rtdetr_accepts_verified_staged_model(tmp_path, monkeypatch):
    snapshot = tmp_path / "snapshot"
    _snapshot(snapshot, 1.25)
    _install_fake_transformers(monkeypatch)

    detector = FrozenRTDetr(snapshot)

    assert detector.model.linear.weight.detach().item() == 1.25
    assert detector.identity()["files"]["model.safetensors"] == _sha256(snapshot / "model.safetensors")


@pytest.mark.skipif(importlib.util.find_spec("safetensors") is None, reason="requires safetensors")
def test_frozen_rtdetr_replacement_after_audit_cannot_change_loaded_state(tmp_path, monkeypatch):
    snapshot = tmp_path / "snapshot"
    _snapshot(snapshot, 1.25)
    expected_weight_hash = _sha256(snapshot / "model.safetensors")
    _install_fake_transformers(monkeypatch, swap_source=snapshot, replacement_weight=9.5)

    detector = FrozenRTDetr(snapshot)

    assert detector.model.linear.weight.detach().item() == 1.25
    assert _sha256(snapshot / "model.safetensors") != expected_weight_hash
    assert detector.identity()["files"]["model.safetensors"] == expected_weight_hash


@pytest.mark.skipif(importlib.util.find_spec("safetensors") is None, reason="requires safetensors")
def test_frozen_rtdetr_rejects_model_with_wrong_loaded_weight(tmp_path, monkeypatch):
    snapshot = tmp_path / "snapshot"
    _snapshot(snapshot, 1.25)
    _install_fake_transformers(monkeypatch, loaded_weight=9.5)

    with pytest.raises(DetectorError, match="weight differs"):
        FrozenRTDetr(snapshot)
