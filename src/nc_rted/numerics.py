"""Fail-closed numerical policy for reproducible inherited memory execution."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os

import torch


NUMERICAL_POLICY_SCHEMA = "nc_rted_deterministic_algorithms/v1"
_CUBLAS_WORKSPACE_CONFIG = ":4096:8"


class NumericalPolicyError(RuntimeError):
    """Raised when the required deterministic policy cannot be established."""


@dataclass(frozen=True)
class NumericalPolicy:
    """The explicit process-wide settings required before CUDA initialization."""
    schema: str
    cublas_workspace_config: str
    deterministic_algorithms: bool
    warn_only: bool

    def identity(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("ascii")).hexdigest()


def deterministic_policy() -> NumericalPolicy:
    """Return the fixed policy identity without modifying process state."""
    return NumericalPolicy(NUMERICAL_POLICY_SCHEMA, _CUBLAS_WORKSPACE_CONFIG, True, False)


def configure_deterministic_algorithms() -> NumericalPolicy:
    """Set and verify the fixed policy before the first CUDA interaction.

    This does not alter model formulas, dtype, input order, or batching.
    PyTorch may select different kernels under deterministic mode, so all
    comparison groups and caches must bind the returned policy identity.  It is
    deliberately process-wide because PyTorch deterministic-algorithm selection
    and cuBLAS workspace configuration are process-wide controls.
    """
    if torch.cuda.is_initialized():
        raise NumericalPolicyError("deterministic policy must be configured before CUDA initialization")
    existing = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if existing is not None and existing != _CUBLAS_WORKSPACE_CONFIG:
        raise NumericalPolicyError("CUBLAS_WORKSPACE_CONFIG differs from required deterministic policy")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = _CUBLAS_WORKSPACE_CONFIG
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
    except Exception as error:
        raise NumericalPolicyError("unable to enable deterministic algorithms") from error
    if not torch.are_deterministic_algorithms_enabled():
        raise NumericalPolicyError("deterministic algorithms are not enabled")
    if torch.is_deterministic_algorithms_warn_only_enabled():
        raise NumericalPolicyError("deterministic algorithms must fail instead of warn")
    return deterministic_policy()
