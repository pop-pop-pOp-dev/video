"""Effective-config binding regressions for the inherited SigLIP adapter."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nc_rted.detector import DetectorError, InheritedSigLipAdapter


@pytest.mark.skipif(importlib.util.find_spec("safetensors") is None or not os.environ.get("NC_RTED_REACTVAU_ROOT"),
                    reason="requires the isolated ReactVAU environment and source root")
def test_siglip_rejects_effective_config_loaded_from_b_after_a_is_restored(tmp_path):
    """A config-only source swap must not be masked by restoring its bytes."""
    from safetensors.torch import save_file

    root = Path(os.environ["NC_RTED_REACTVAU_ROOT"])
    sys.path.insert(0, str(root))
    from llava.model.multimodal_encoder.siglip_encoder import (
        SigLipImageProcessor,
        SigLipVisionConfig,
        SigLipVisionTower,
    )

    snapshot = tmp_path / "siglip"
    snapshot.mkdir()
    config_a = {
        "model_type": "siglip",
        "vision_config": {
            "model_type": "siglip_vision_model",
            "image_size": 384,
            "patch_size": 14,
            "hidden_size": 1152,
            "intermediate_size": 4304,
            "num_attention_heads": 16,
            "num_hidden_layers": 27,
        },
    }
    config_b = json.loads(json.dumps(config_a))
    config_b["vision_config"]["layer_norm_eps"] = 1e-3
    (snapshot / "config.json").write_text(json.dumps(config_a))
    expected = torch.tensor([1.25])
    save_file({
        "vision_model.embeddings.weight": expected,
        "vision_model.encoder.layers.26.deleted": torch.tensor([0.]),
        "vision_model.head.removed": torch.tensor([0.]),
    }, str(snapshot / "model.safetensors"))

    # This is the real ReactVAU config loader used by SigLipVisionModel's
    # from_pretrained path. The loaded configuration B changes behavior without
    # changing any saved tensor shape or value.
    (snapshot / "config.json").write_text(json.dumps(config_b))
    loaded_b = SigLipVisionConfig.from_pretrained(snapshot)
    assert loaded_b.layer_norm_eps == pytest.approx(1e-3)
    (snapshot / "config.json").write_text(json.dumps(config_a))

    class Inner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = loaded_b
            self.vision_model = torch.nn.Module()
            self.vision_model.embeddings = torch.nn.Module()
            self.vision_model.embeddings.register_parameter("weight", torch.nn.Parameter(expected.clone()))
            self.vision_model.encoder = torch.nn.Module()
            self.vision_model.encoder.layers = torch.nn.ModuleList([torch.nn.Identity() for _ in range(26)])
            self.vision_model.head = torch.nn.Identity()

    tower = SigLipVisionTower.__new__(SigLipVisionTower)
    torch.nn.Module.__init__(tower)
    tower.is_loaded = True
    tower.vision_tower_name = str(snapshot)
    tower.vision_tower = Inner()
    tower.image_processor = SigLipImageProcessor()
    tower.eval()
    tower.vision_tower.eval()
    tower.vision_tower.requires_grad_(False)

    with pytest.raises(DetectorError, match="effective vision configuration differs"):
        InheritedSigLipAdapter(tower, snapshot)
