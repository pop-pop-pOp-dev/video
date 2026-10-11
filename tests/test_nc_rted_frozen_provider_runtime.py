import gc
from types import SimpleNamespace
import weakref

import pytest
import torch
from torch import nn

from nc_rted.frozen_provider_runtime import FrozenVisualModel
from nc_rted.inherited_memory import caption_memory_from_patches


class Projector(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Linear(1152, 4)

    def forward(self, patches, *, local_num_frames, pg_scores):
        assert local_num_frames == 1
        assert len(pg_scores) == len(patches)
        return self.mlp(patches.mean(dim=1)).unsqueeze(0)


class Original(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(mm_local_num_frames=1)
        self.model = nn.Module()
        self.model.mm_projector = Projector()
        self.model.language_layers = nn.Linear(64, 64)
        self.tower = nn.Linear(8, 8)
        self.requires_grad_(False)

    def get_model(self):
        return self.model

    def get_vision_tower(self):
        return self.tower


def test_visual_facade_preserves_original_memory_and_releases_language_layers():
    original = Original()
    language = weakref.ref(original.model.language_layers)
    tower, projector = original.tower, original.model.mm_projector
    patches = torch.ones(2, 729, 1152)
    reference = caption_memory_from_patches(original, patches, [.1, .8])
    visual = FrozenVisualModel(original)
    assert visual.get_model().mm_projector is projector
    assert visual.get_vision_tower() is tower
    assert torch.equal(caption_memory_from_patches(visual, patches, [.1, .8]), reference)
    del original
    gc.collect()
    assert language() is None
    assert set(map(id, visual.parameters())) == set(map(id, [*tower.parameters(), *projector.parameters()]))


def test_visual_facade_refuses_trainable_projector():
    original = Original()
    original.model.mm_projector.mlp.weight.requires_grad_(True)
    with pytest.raises(ValueError, match="trainable visual"):
        FrozenVisualModel(original)
