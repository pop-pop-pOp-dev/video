"""Derived final-Stage2 vision provenance regressions."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nc_rted.detector import DetectorError
from nc_rted.frozen_vision import (DEFAULT_SOURCE_KEY_PREFIX, SCHEMA,
                                   bind_derived_final_stage2_vision,
                                   verify_loaded_derived_vision)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_sha(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file():
            digest.update(str(path.relative_to(root)).encode()); digest.update(b"\0")
            digest.update(_sha(path).encode()); digest.update(b"\n")
    return digest.hexdigest()


@pytest.mark.skipif(__import__("importlib").util.find_spec("safetensors") is None, reason="requires safetensors")
def test_derived_final_stage2_binding_accepts_exact_loaded_tensors_and_rejects_drift(tmp_path):
    from safetensors.torch import save_file
    root = tmp_path / "derived"; root.mkdir()
    (root / "config.json").write_text('{"model_type":"siglip"}')
    tensors = {f"vision_model.tensor.{index}": torch.tensor([index], dtype=torch.bfloat16) for index in range(421)}
    save_file(tensors, str(root / "model.safetensors"), metadata={"format": "pt"})
    parent = "a" * 64
    provenance = {"schema": SCHEMA, "source_export_sha256": parent,
                  "source_key_prefix": DEFAULT_SOURCE_KEY_PREFIX, "tensor_count": 421,
                  "tensor_dtype": "bfloat16", "raw_config_sha256": _sha(root / "config.json"),
                  "files": {"config.json": _sha(root / "config.json"), "model.safetensors": _sha(root / "model.safetensors")},
                  "source_key_map": {name: DEFAULT_SOURCE_KEY_PREFIX + name for name in tensors}}
    (root / "nc_rted_provenance.json").write_text(json.dumps(provenance))
    parent_export = tmp_path / "parent.bin"
    torch.save({DEFAULT_SOURCE_KEY_PREFIX + name: value for name, value in tensors.items()}, parent_export)
    parent = _sha(parent_export); provenance["source_export_sha256"] = parent
    (root / "nc_rted_provenance.json").write_text(json.dumps(provenance))
    binding = bind_derived_final_stage2_vision(root, expected_parent_export_sha256=parent, expected_parent_export=parent_export,
                                                expected_raw_config_sha256=_sha(root / "config.json"))
    verify_loaded_derived_vision(binding, tensors)
    # The binding retains the verified source inode. Replacing its pathname
    # after construction cannot alter the later live-tower comparison.
    replacement = root / "replacement.safetensors"
    save_file({name: torch.zeros_like(value) for name, value in tensors.items()}, str(replacement), metadata={"format": "pt"})
    replacement.replace(root / "model.safetensors")
    verify_loaded_derived_vision(binding, tensors)
    changed = dict(tensors); changed["vision_model.tensor.0"] = torch.tensor([99], dtype=torch.bfloat16)
    with pytest.raises(DetectorError, match="loaded SigLip weight differs"):
        verify_loaded_derived_vision(binding, changed)


@pytest.mark.skipif(__import__("importlib").util.find_spec("safetensors") is None, reason="requires safetensors")
def test_derived_final_stage2_binding_rejects_parent_config_and_tensor_map_changes(tmp_path):
    from safetensors.torch import save_file
    root = tmp_path / "derived"; root.mkdir()
    (root / "config.json").write_text('{"model_type":"siglip"}')
    tensor = {f"vision_model.x.{index}": torch.ones(1, dtype=torch.bfloat16) for index in range(421)}
    save_file(tensor, str(root / "model.safetensors"), metadata={"format": "pt"})
    files = {"config.json": _sha(root / "config.json"), "model.safetensors": _sha(root / "model.safetensors")}
    parent_export = tmp_path / "parent.bin"
    torch.save({DEFAULT_SOURCE_KEY_PREFIX + name: value for name, value in tensor.items()}, parent_export)
    parent_sha = _sha(parent_export)
    base = {"schema": SCHEMA, "source_export_sha256": parent_sha, "source_key_prefix": DEFAULT_SOURCE_KEY_PREFIX,
            "tensor_count": 421, "tensor_dtype": "bfloat16", "raw_config_sha256": files["config.json"],
            "files": files, "source_key_map": {name: DEFAULT_SOURCE_KEY_PREFIX + name for name in tensor}}
    provenance = root / "nc_rted_provenance.json"
    provenance.write_text(json.dumps(base))
    with pytest.raises(DetectorError, match="parent export differs"):
        bind_derived_final_stage2_vision(root, expected_parent_export_sha256="b" * 64, expected_parent_export=parent_export)
    with pytest.raises(DetectorError, match="expected raw config differs"):
        bind_derived_final_stage2_vision(root, expected_parent_export_sha256=parent_sha, expected_parent_export=parent_export,
                                         expected_raw_config_sha256="c" * 64)
    broken = dict(base); broken["source_key_map"] = {name: "wrong." + name for name in tensor}
    provenance.write_text(json.dumps(broken))
    with pytest.raises(DetectorError, match="tensor map differs"):
        bind_derived_final_stage2_vision(root, expected_parent_export_sha256=parent_sha, expected_parent_export=parent_export)
    # A hash-matched but value-different parent cannot be justified by a
    # provenance claim: each derived tensor must still match its parent key.
    provenance.write_text(json.dumps(base))
    wrong_parent = {DEFAULT_SOURCE_KEY_PREFIX + name: value.clone() for name, value in tensor.items()}
    wrong_parent[DEFAULT_SOURCE_KEY_PREFIX + "vision_model.x.0"] = torch.zeros(1, dtype=torch.bfloat16)
    torch.save(wrong_parent, parent_export)
    wrong_sha = _sha(parent_export); base["source_export_sha256"] = wrong_sha
    provenance.write_text(json.dumps(base))
    with pytest.raises(DetectorError, match="tensor differs from parent export"):
        bind_derived_final_stage2_vision(root, expected_parent_export_sha256=wrong_sha, expected_parent_export=parent_export)


@pytest.mark.skipif(__import__("importlib").util.find_spec("safetensors") is None, reason="requires safetensors")
def test_exporter_writes_exact_parent_prefixed_bf16_snapshot_without_overwrite(tmp_path, monkeypatch):
    from safetensors.torch import save_file
    script = Path(__file__).resolve().parents[1] / "scripts" / "nc_rted_export_frozen_vision.py"
    spec = importlib.util.spec_from_file_location("vision_export", script)
    module = importlib.util.module_from_spec(spec); assert spec.loader is not None; spec.loader.exec_module(module)
    monkeypatch.setattr(module.shutil, "disk_usage", lambda _: type("Disk", (), {"free": 21 * 1024 ** 3})())
    raw = tmp_path / "raw"; raw.mkdir(); (raw / "config.json").write_text('{"model_type":"siglip"}')
    retained = {f"vision_model.tensor.{index}": torch.tensor([index], dtype=torch.bfloat16) for index in range(421)}
    save_file(retained, str(raw / "model.safetensors"), metadata={"format": "pt"})
    parent = tmp_path / "non_lora_trainables.bin"
    torch.save({DEFAULT_SOURCE_KEY_PREFIX + name: value for name, value in retained.items()}, parent)
    output = tmp_path / "derived"
    report = module.export(source_export=parent, source_export_sha256=_sha(parent), raw_snapshot=raw,
                           raw_snapshot_sha256=_tree_sha(raw), output=output)
    assert report["tensor_count"] == 421
    provenance = json.loads((output / "nc_rted_provenance.json").read_text())
    assert provenance["tensor_count"] == 421
    assert provenance["files"]["config.json"] == _sha(raw / "config.json")
    binding = bind_derived_final_stage2_vision(output, expected_parent_export_sha256=_sha(parent), expected_parent_export=parent)
    verify_loaded_derived_vision(binding, retained)
    with pytest.raises(RuntimeError, match="already exists"):
        module.export(source_export=parent, source_export_sha256=_sha(parent), raw_snapshot=raw,
                      raw_snapshot_sha256=_tree_sha(raw), output=output)
    staged, competing = tmp_path / "staged", tmp_path / "competing"
    staged.mkdir(); competing.mkdir()
    with pytest.raises(RuntimeError, match="already exists"):
        module._rename_noreplace(staged, competing)
