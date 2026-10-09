import pytest
import torch
from torch import nn

from nc_rted.loading import (
    CheckpointError, configure_slow_trainability, load_non_lora_exact,
    normalize_non_lora, require_tensor_subset,
)


def test_saved_projector_aliases_require_exact_agreement():
    x = torch.tensor([1., 2.], dtype=torch.bfloat16)
    source = {"base_model.model.model.mm_projector.weight": x,
              "model.mm_projector.weight": x.clone()}
    assert set(normalize_non_lora(source)) == {"model.mm_projector.weight"}
    source["model.mm_projector.weight"][0] = 3
    with pytest.raises(CheckpointError, match="conflicting"):
        normalize_non_lora(source)


def test_unknown_or_misshaped_saved_weights_fail_before_any_mutation():
    model = nn.Linear(2, 3, bias=False).to(torch.bfloat16)
    before = model.weight.detach().clone()
    for saved in ({"weight": torch.zeros(2, 3, dtype=torch.bfloat16)},
                  {"weight": torch.zeros_like(before), "missing": torch.zeros(1)}):
        with pytest.raises(CheckpointError):
            load_non_lora_exact(model, saved)
        assert torch.equal(model.weight, before)
    report = load_non_lora_exact(model, {"weight": torch.ones_like(before)})
    assert report["unique_model_keys"] == 1 and torch.equal(model.weight, torch.ones_like(before))


def test_reload_never_silently_casts_or_accepts_nonfinite_saved_tensors():
    with pytest.raises(CheckpointError, match="dtype"):
        require_tensor_subset({"w": torch.ones(2)}, {"w": torch.ones(2, dtype=torch.bfloat16)})
    with pytest.raises(CheckpointError, match="nonfinite"):
        require_tensor_subset({"w": torch.tensor([float("nan")])}, {"w": torch.ones(1)})


def test_only_exact_inherited_lora_allowlist_can_train():
    model = nn.Module()
    model.layer = nn.Module()
    model.layer.lora_A = nn.ModuleDict({"default": nn.Linear(2, 1, bias=False)})
    model.layer.lora_B = nn.ModuleDict({"default": nn.Linear(1, 2, bias=False)})
    model.projector = nn.Linear(2, 2)
    saved = {"layer.lora_A.weight", "layer.lora_B.weight"}
    trainable = configure_slow_trainability(model, saved, train_lora=True)
    assert set(trainable) == {"layer.lora_A.default.weight", "layer.lora_B.default.weight"}
    assert not model.projector.weight.requires_grad
    assert configure_slow_trainability(model, saved, train_lora=False) == []
    with pytest.raises(CheckpointError, match="allowlist"):
        configure_slow_trainability(model, {"layer.lora_A.weight"}, train_lora=True)
