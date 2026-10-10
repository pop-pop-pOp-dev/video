from types import SimpleNamespace
import pytest
import torch
from nc_rted.detector import DetectorError
from nc_rted.frozen_vision import ExactDerivedVisionVerifier, verify_loaded_derived_vision


def sample():
    weights = {"b": torch.arange(12, dtype=torch.bfloat16).reshape(3, 4),
               "a": torch.tensor([0.0, -0.0, -1.0, 2.0], dtype=torch.bfloat16)}
    binding = SimpleNamespace(weights=weights, source_key_map={name: name for name in weights})
    return binding, {name: tensor.clone() for name, tensor in weights.items()}


def test_reference_and_live_values_use_same_exact_predicate():
    binding, state = sample()
    verifier = ExactDerivedVisionVerifier(binding)
    verify_loaded_derived_vision(binding, state)
    verifier.verify(state)
    version = state["b"]._version
    state["b"].data[1, 2] += 1
    assert state["b"]._version == version
    for check in (lambda: verify_loaded_derived_vision(binding, state), lambda: verifier.verify(state)):
        with pytest.raises(DetectorError, match="differs"):
            check()
    state["b"].data.copy_(binding.weights["b"])
    verifier.verify(state)


@pytest.mark.parametrize("drift", ["key", "shape", "dtype", "nan", "inf"])
def test_rejects_incompatible_or_nonfinite_live_values(drift):
    binding, state = sample()
    verifier = ExactDerivedVisionVerifier(binding)
    if drift == "key": state["c"] = state.pop("a")
    elif drift == "shape": state["b"] = state["b"].reshape(4, 3)
    elif drift == "dtype": state["b"] = state["b"].float()
    elif drift == "nan": state["b"][0, 0] = float("nan")
    else: state["b"][0, 0] = float("inf")
    with pytest.raises(DetectorError): verifier.verify(state)


def test_private_reference_cannot_be_changed_by_live_or_input_alias():
    binding, state = sample()
    verifier = ExactDerivedVisionVerifier(binding)
    binding.weights["a"].data[0] += 1
    verifier.verify(state)
    state["a"].data.copy_(binding.weights["a"])
    with pytest.raises(DetectorError): verifier.verify(state)


def test_key_order_and_noncontiguous_layout_do_not_change_values():
    binding, state = sample()
    verifier = ExactDerivedVisionVerifier(binding)
    state = {name: value.t().contiguous().t() if value.ndim == 2 else value
             for name, value in reversed(list(state.items()))}
    verifier.verify(state)
