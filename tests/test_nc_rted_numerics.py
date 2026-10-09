import os

import pytest

from nc_rted import numerics


def test_policy_identity_is_stable_and_hashable():
    first, second = numerics.deterministic_policy(), numerics.deterministic_policy()
    assert first == second
    assert len(first.identity()) == 64
    assert first.cublas_workspace_config == ":4096:8"
    assert first.deterministic_algorithms and not first.warn_only


def test_configure_sets_required_environment_and_strict_algorithms(monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setattr(numerics.torch.cuda, "is_initialized", lambda: False)
    calls = []
    monkeypatch.setattr(numerics.torch, "use_deterministic_algorithms", lambda enabled, *, warn_only: calls.append((enabled, warn_only)))
    monkeypatch.setattr(numerics.torch, "are_deterministic_algorithms_enabled", lambda: True)
    monkeypatch.setattr(numerics.torch, "is_deterministic_algorithms_warn_only_enabled", lambda: False)
    policy = numerics.configure_deterministic_algorithms()
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert calls == [(True, False)]
    assert policy == numerics.deterministic_policy()


def test_configure_fails_before_cuda_or_on_workspace_mismatch(monkeypatch):
    monkeypatch.setattr(numerics.torch.cuda, "is_initialized", lambda: True)
    with pytest.raises(numerics.NumericalPolicyError, match="before CUDA"):
        numerics.configure_deterministic_algorithms()

    monkeypatch.setattr(numerics.torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    with pytest.raises(numerics.NumericalPolicyError, match="differs"):
        numerics.configure_deterministic_algorithms()


def test_configure_fails_when_torch_cannot_verify_strict_policy(monkeypatch):
    monkeypatch.setattr(numerics.torch.cuda, "is_initialized", lambda: False)
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setattr(numerics.torch, "use_deterministic_algorithms", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(numerics.torch, "are_deterministic_algorithms_enabled", lambda: False)
    monkeypatch.setattr(numerics.torch, "is_deterministic_algorithms_warn_only_enabled", lambda: False)
    with pytest.raises(numerics.NumericalPolicyError, match="not enabled"):
        numerics.configure_deterministic_algorithms()

    monkeypatch.setattr(numerics.torch, "are_deterministic_algorithms_enabled", lambda: True)
    monkeypatch.setattr(numerics.torch, "is_deterministic_algorithms_warn_only_enabled", lambda: True)
    with pytest.raises(numerics.NumericalPolicyError, match="must fail"):
        numerics.configure_deterministic_algorithms()
