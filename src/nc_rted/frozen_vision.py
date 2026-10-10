"""Binding for the final-Stage2 SigLIP tensors retained in a derived snapshot."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path

import torch

from .detector import DetectorError, sha256_file


SCHEMA = "nc_rted_final_stage2_vision/v1"
DEFAULT_SOURCE_KEY_PREFIX = "base_model.model.model.vision_tower.vision_tower."


@dataclass(frozen=True)
class DerivedVisionBinding:
    snapshot: Path
    config_sha256: str
    weights_sha256: str
    parent_export_sha256: str
    source_key_prefix: str
    source_key_map: dict[str, str]
    config_bytes: bytes
    weights: dict[str, torch.Tensor]


def _read_json(path: Path, message: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise DetectorError(message) from error
    if not isinstance(value, dict):
        raise DetectorError(message)
    return value


def bind_derived_final_stage2_vision(snapshot: str | Path, *, expected_parent_export_sha256: str,
                                     expected_parent_export: str | Path,
                                     expected_parent_key_prefix: str = DEFAULT_SOURCE_KEY_PREFIX,
                                     expected_raw_config_sha256: str | None = None) -> DerivedVisionBinding:
    """Validate a derived final-Stage2 vision asset without trusting its path.

    The adapter accepts a different path from ``tower.vision_tower_name`` only
    after this binding proves that every retained snapshot tensor originated in
    the exact final inherited non-LoRA export selected by the runtime.
    """
    root = Path(snapshot).resolve()
    config, weights, provenance_path = root / "config.json", root / "model.safetensors", root / "nc_rted_provenance.json"
    if not root.is_dir() or any(not path.is_file() for path in (config, weights, provenance_path)):
        raise DetectorError("derived final-Stage2 SigLip snapshot is incomplete")
    provenance = _read_json(provenance_path, "derived final-Stage2 SigLip provenance is invalid")
    if provenance.get("schema") != SCHEMA:
        raise DetectorError("derived final-Stage2 SigLip provenance schema differs")
    if provenance.get("source_export_sha256") != expected_parent_export_sha256:
        raise DetectorError("derived final-Stage2 SigLip parent export differs")
    if provenance.get("source_key_prefix") != expected_parent_key_prefix:
        raise DetectorError("derived final-Stage2 SigLip source key prefix differs")
    files = provenance.get("files")
    if not isinstance(files, dict) or set(files) != {"config.json", "model.safetensors"}:
        raise DetectorError("derived final-Stage2 SigLip provenance file map is incomplete")
    if any(not isinstance(value, str) or len(value) != 64 for value in files.values()):
        raise DetectorError("derived final-Stage2 SigLip provenance hashes are invalid")
    config_bytes, weights_bytes = config.read_bytes(), weights.read_bytes()
    if hashlib.sha256(config_bytes).hexdigest() != files["config.json"] or hashlib.sha256(weights_bytes).hexdigest() != files["model.safetensors"]:
        raise DetectorError("derived final-Stage2 SigLip snapshot differs from provenance")
    raw_config = provenance.get("raw_config_sha256")
    if not isinstance(raw_config, str) or len(raw_config) != 64 or raw_config != files["config.json"]:
        raise DetectorError("derived final-Stage2 SigLip raw config binding differs")
    if expected_raw_config_sha256 is not None and raw_config != expected_raw_config_sha256:
        raise DetectorError("derived final-Stage2 SigLip expected raw config differs")
    mapping = provenance.get("source_key_map")
    if not isinstance(mapping, dict) or not mapping:
        raise DetectorError("derived final-Stage2 SigLip tensor map is missing")
    if provenance.get("tensor_count") != 421 or len(mapping) != 421 or provenance.get("tensor_dtype") != "bfloat16":
        raise DetectorError("derived final-Stage2 SigLip tensor inventory differs")
    if any(not isinstance(name, str) or not isinstance(source, str) or source != expected_parent_key_prefix + name
           for name, source in mapping.items()):
        raise DetectorError("derived final-Stage2 SigLip tensor map differs from parent prefix")
    try:
        from safetensors.torch import load
    except ImportError as error:
        raise DetectorError("safetensors is required to verify the derived SigLip binding") from error
    parent_path = Path(expected_parent_export)
    if not parent_path.is_file():
        raise DetectorError("derived final-Stage2 SigLip parent export is absent")
    parent_bytes = parent_path.read_bytes()
    if hashlib.sha256(parent_bytes).hexdigest() != expected_parent_export_sha256:
        raise DetectorError("derived final-Stage2 SigLip parent export bytes differ")
    try:
        parent = torch.load(io.BytesIO(parent_bytes), map_location="cpu", weights_only=True)
    except (RuntimeError, ValueError, TypeError) as error:
        raise DetectorError("derived final-Stage2 SigLip parent export cannot be loaded") from error
    if not isinstance(parent, dict):
        raise DetectorError("derived final-Stage2 SigLip parent export is not a tensor map")
    parsed = load(weights_bytes)
    names = set(parsed)
    if names != set(mapping):
        raise DetectorError("derived final-Stage2 SigLip tensor names differ from provenance")
    for name in sorted(names):
        tensor = parsed[name]
        if str(tensor.dtype) != "torch.bfloat16" or not tensor.isfinite().all().item():
            raise DetectorError("derived final-Stage2 SigLip tensor dtype or values differ")
        original = parent.get(mapping[name])
        if (not isinstance(original, torch.Tensor) or original.shape != tensor.shape or
                original.dtype != tensor.dtype or not torch.equal(original.detach().cpu(), tensor)):
            raise DetectorError(f"derived final-Stage2 SigLip tensor differs from parent export: {name}")
    parent_names = {name for name in parent if isinstance(name, str) and name.startswith(expected_parent_key_prefix)}
    if parent_names != set(mapping.values()):
        raise DetectorError("derived final-Stage2 SigLip parent vision inventory differs")
    return DerivedVisionBinding(root, files["config.json"], files["model.safetensors"],
                                expected_parent_export_sha256, expected_parent_key_prefix, dict(mapping), config_bytes, parsed)


def verify_loaded_derived_vision(binding: DerivedVisionBinding, state: dict) -> None:
    """Compare every derived tensor to the already-loaded inherited tower."""
    if set(state) != set(binding.source_key_map):
        raise DetectorError("loaded SigLip state keys differ from derived final-Stage2 binding")
    for name in sorted(binding.source_key_map):
        expected, loaded = binding.weights[name], state[name].detach().cpu()
        if (expected.shape != loaded.shape or expected.dtype != loaded.dtype or
                not torch.equal(expected, loaded)):
            raise DetectorError(f"loaded SigLip weight differs from derived final-Stage2 binding: {name}")



class ExactDerivedVisionVerifier:
    """Keep a detached exact reference on the live tower's device.

    Initial parent/export validation remains mandatory. Each check compares all
    values in the original dtype, including mutations through ``Tensor.data``
    that do not increment the tensor version. Only the frozen reference is
    retained; the concatenated live values are temporary. There is no reduced
    checksum, quantization, model output, or trainable-state cache.
    """
    def __init__(self, binding: DerivedVisionBinding):
        self.binding = binding
        self.names = tuple(sorted(binding.source_key_map))
        if not self.names or set(binding.weights) != set(self.names):
            raise DetectorError("derived vision reference inventory differs")
        self.layout = tuple((name, tuple(binding.weights[name].shape), binding.weights[name].dtype)
                            for name in self.names)
        dtypes = {dtype for _, _, dtype in self.layout}
        if len(dtypes) != 1:
            raise DetectorError("derived vision reference must retain one original dtype")
        # This private snapshot does not alias the supplied reference tensors.
        with torch.inference_mode(False), torch.no_grad():
            self._cpu_reference = torch.cat([binding.weights[name].detach().cpu().reshape(-1)
                                            for name in self.names]).clone()
        self._device_reference = None
        self._device = None

    @torch.no_grad()
    def verify(self, state: dict) -> None:
        if set(state) != set(self.names):
            raise DetectorError("loaded SigLip state keys differ from derived final-Stage2 binding")
        devices = set()
        values = []
        for name, shape, dtype in self.layout:
            value = state[name]
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or value.dtype != dtype:
                raise DetectorError(f"loaded SigLip weight differs from bound shape or dtype: {name}")
            devices.add(value.device)
            values.append(value.detach().reshape(-1))
        if len(devices) != 1:
            raise DetectorError("loaded SigLip weights span multiple devices")
        device = devices.pop()
        if device.type not in {"cpu", "cuda"}:
            raise DetectorError("unsupported exact SigLip comparison device")
        if self._device != device or self._device_reference is None:
            # Retain at most one device copy; device changes cannot reuse a
            # reference from another GPU. CPU checks use the private CPU copy.
            self._device_reference = (self._cpu_reference if device.type == "cpu"
                                      else self._cpu_reference.to(device=device))
            self._device = device
        live = torch.cat(values)
        if not torch.equal(live, self._device_reference):
            raise DetectorError("loaded SigLip weight differs from bound exact reference")


def provenance_digest(binding: DerivedVisionBinding) -> str:
    """Stable identity for callers that need one provenance field."""
    payload = json.dumps({"snapshot": str(binding.snapshot), "config_sha256": binding.config_sha256,
                          "weights_sha256": binding.weights_sha256,
                          "parent_export_sha256": binding.parent_export_sha256,
                          "source_key_prefix": binding.source_key_prefix,
                          "source_key_map": binding.source_key_map}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
