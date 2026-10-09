"""Audited-key loading of inherited Stage2 tensors without merging its LoRA.

Key normalization preserves the previously verified rules in
``scripts/reactvau_stage2_cpu_reload_validate.py``. That executable is deliberately
not imported: its top-level CUDA visibility mutation is inappropriate here.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import hashlib
import torch
from torch import Tensor, nn


class CheckpointError(RuntimeError):
    pass


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_non_lora(state: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Normalize each saved source independently, rejecting conflicting aliases."""
    if not state:
        raise CheckpointError("empty non-LoRA export")
    result = {}
    for source, value in state.items():
        if not isinstance(source, str) or not isinstance(value, Tensor):
            raise CheckpointError("non-LoRA export must map string keys to tensors")
        key = source.removeprefix("base_model.model.")
        key = key.replace(".base_layer.", ".").replace(".modules_to_save.default.", ".")
        if key in result and not torch.equal(result[key], value):
            raise CheckpointError(f"conflicting saved aliases for {key}")
        result[key] = value
    return result


def require_tensor_subset(saved: Mapping[str, Tensor], actual: Mapping[str, Tensor],
                          *, compare_values: bool = True) -> None:
    """Every saved tensor must exist and match; unrelated base tensors are allowed."""
    for key, expected in saved.items():
        if key not in actual:
            raise CheckpointError(f"missing loaded key: {key}")
        observed = actual[key]
        if observed.shape != expected.shape:
            raise CheckpointError(f"shape mismatch: {key}")
        if observed.dtype != expected.dtype:
            raise CheckpointError(f"dtype mismatch: {key}: {observed.dtype} vs {expected.dtype}")
        if expected.is_floating_point() and not bool(torch.isfinite(expected).all()):
            raise CheckpointError(f"nonfinite saved tensor: {key}")
        if compare_values and not torch.equal(expected.detach().cpu(), observed.detach().cpu()):
            raise CheckpointError(f"loaded tensor differs: {key}")


def load_non_lora_exact(model: nn.Module, saved: Mapping[str, Tensor]) -> dict:
    """Preflight every key/shape/dtype before mutating any inherited parameter."""
    normalized = normalize_non_lora(saved)
    require_tensor_subset(normalized, model.state_dict(), compare_values=False)
    incompatible = model.load_state_dict(normalized, strict=False)
    if incompatible.unexpected_keys or set(normalized).intersection(incompatible.missing_keys):
        raise CheckpointError("saved non-LoRA tensors were silently omitted")
    require_tensor_subset(normalized, model.state_dict())
    return {"saved_tensors": len(saved), "unique_model_keys": len(normalized),
            "saved_projector_tensors": sum("mm_projector" in k for k in saved),
            "base_only_missing_keys": sorted(incompatible.missing_keys)}


def lora_model_key(saved_key: str) -> str:
    for leaf in ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B"):
        token = f".{leaf}."
        if token in saved_key:
            return saved_key.replace(token, f".{leaf}.default.")
    raise CheckpointError(f"unsupported non-LoRA tensor in adapter export: {saved_key}")


def configure_slow_trainability(model: nn.Module, saved_lora_keys: set[str],
                               *, train_lora: bool) -> list[str]:
    """Exact allowlist: only inherited saved LoRA tensors can train inside Slow."""
    expected = {lora_model_key(key) for key in saved_lora_keys}
    parameters = dict(model.named_parameters())
    observed = {key for key in parameters if any(f".{leaf}." in key for leaf in
                ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B"))}
    if expected != observed:
        raise CheckpointError(f"LoRA allowlist differs: missing={sorted(expected-observed)}, "
                              f"unexpected={sorted(observed-expected)}")
    for name, parameter in parameters.items():
        parameter.requires_grad_(train_lora and name in expected)
    return sorted(name for name, parameter in parameters.items() if parameter.requires_grad)


def load_inherited_slow(base_directory: str | Path, export_directory: str | Path,
                         *, train_lora: bool, expected_hashes: Mapping[str, str],
                         device: str = "cpu"):
    """Load the real inherited model; caller supplies pinned llava on sys.path.

    Hash keys are export filenames. All base shards must separately be bound by
    the original-file manifest before admitting a formal job. Loading on CPU
    first permits exact value checks before any device-specific execution.
    """
    base, export = Path(base_directory).resolve(), Path(export_directory).resolve()
    required = ("config.json", "adapter_config.json", "adapter_model.safetensors", "non_lora_trainables.bin")
    if not base.is_dir() or not export.is_dir():
        raise CheckpointError("base and export must be existing local directories")
    if set(expected_hashes) != set(required):
        raise CheckpointError("explicit hashes for all four inherited export files are required")
    for name in required:
        if not (export / name).is_file() or file_sha256(export / name) != expected_hashes[name]:
            raise CheckpointError(f"original export identity mismatch: {name}")
    from llava.model.language_model.llava_qwen import LlavaQwenConfig, LlavaQwenForCausalLM
    from peft import PeftModel
    from safetensors.torch import load_file

    config = LlavaQwenConfig.from_pretrained(str(export), local_files_only=True)
    if not Path(str(getattr(config, "mm_vision_tower", ""))).is_dir():
        raise CheckpointError("inherited config must bind an existing local vision tower")
    model = LlavaQwenForCausalLM.from_pretrained(
        str(base), config=config, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        attn_implementation="sdpa", local_files_only=True,
    )
    raw = torch.load(export / "non_lora_trainables.bin", map_location="cpu", weights_only=True)
    if not isinstance(raw, Mapping):
        raise CheckpointError("invalid non-LoRA tensor mapping")
    non_lora_report = load_non_lora_exact(model, raw)
    if non_lora_report["saved_projector_tensors"] == 0:
        raise CheckpointError("trained projector is missing")
    del raw
    # is_trainable=False avoids PEFT silently promoting the saved BF16 tensors;
    # the exact allowlist below enables gradients explicitly after reload checks.
    model = PeftModel.from_pretrained(model, str(export), is_trainable=False,
                                     autocast_adapter_dtype=False, local_files_only=True)
    saved = load_file(str(export / "adapter_model.safetensors"), device="cpu")
    if not saved:
        raise CheckpointError("empty inherited LoRA")
    mapped = {lora_model_key(k): v for k, v in saved.items()}
    require_tensor_subset(mapped, model.state_dict())
    trainable = configure_slow_trainability(model, set(saved), train_lora=train_lora)
    report = {"status": "PASS_TENSOR_RELOAD", "non_lora": non_lora_report,
              "lora_tensors": len(saved), "trainable_slow_keys": trainable,
              "merged": False, "dtype": "bfloat16", "export_hashes": dict(expected_hashes),
              "base_directory": str(base), "export_directory": str(export),
              "capacity_acceptance": False}
    del saved, mapped
    model.to(device)
    model.train(train_lora)
    return model, report
