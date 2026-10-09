#!/usr/bin/env python3
"""Export the final Stage2 retained SigLIP tensors as a provenance-bound snapshot."""
from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from nc_rted.frozen_vision import DEFAULT_SOURCE_KEY_PREFIX, SCHEMA
from nc_rted.detector import sha256_file
from nc_rted.storage_lock import allocation_lock

_RESERVE = 20 * 1024 ** 3
_OVERHEAD_BYTES = 8 * 1024 ** 2


def _tree_sha256(root: Path) -> str:
    return _tree_digest(_file_hashes(root))


def _file_hashes(root: Path) -> dict[str, str]:
    return {str(path.relative_to(root)): sha256_file(path) for path in sorted(root.rglob("*")) if path.is_file()}


def _tree_digest(files: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for name, file_hash in sorted(files.items()):
        digest.update(name.encode()); digest.update(b"\0")
        digest.update(file_hash.encode()); digest.update(b"\n")
    return digest.hexdigest()


def _rename_noreplace(source: Path, destination: Path) -> None:
    renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("atomic no-replace publication is unavailable")
    result = renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1)
    if result != 0:
        failure = ctypes.get_errno()
        if failure == errno.EEXIST:
            raise RuntimeError("output snapshot already exists")
        raise OSError(failure, os.strerror(failure), destination)


def export(*, source_export: Path, source_export_sha256: str, raw_snapshot: Path,
           raw_snapshot_sha256: str, output: Path, source_key_prefix: str = DEFAULT_SOURCE_KEY_PREFIX) -> dict:
    source_export, raw_snapshot, output = source_export.resolve(), raw_snapshot.resolve(), output.resolve()
    config, raw_weights = raw_snapshot / "config.json", raw_snapshot / "model.safetensors"
    if not source_export.is_file() or not raw_snapshot.is_dir() or not config.is_file() or not raw_weights.is_file():
        raise RuntimeError("source export or raw SigLip snapshot identity differs")
    expected_raw = _file_hashes(raw_snapshot)
    if _tree_digest(expected_raw) != raw_snapshot_sha256:
        raise RuntimeError("source export or raw SigLip snapshot identity differs")
    # Deserialize the same bytes just hashed; passing the mutable pathname to
    # torch.load would leave a replacement interval after identity validation.
    source_bytes = source_export.read_bytes()
    if hashlib.sha256(source_bytes).hexdigest() != source_export_sha256:
        raise RuntimeError("source export or raw SigLip snapshot identity differs")
    state = torch.load(io.BytesIO(source_bytes), map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise RuntimeError("final Stage2 non-LoRA export is not a tensor mapping")
    retained = {name.removeprefix(source_key_prefix): value for name, value in state.items()
                if isinstance(name, str) and name.startswith(source_key_prefix)}
    if len(retained) != 421 or len(retained) != sum(name.startswith(source_key_prefix) for name in state):
        raise RuntimeError("final Stage2 export has invalid vision tensor keys")
    if any(not isinstance(value, torch.Tensor) or value.dtype != torch.bfloat16 or not bool(torch.isfinite(value).all())
           for value in retained.values()):
        raise RuntimeError("final Stage2 vision tensors must be finite BF16")
    # safetensors needs a path, so hold the consumed raw bytes in an anonymous
    # file descriptor rather than reopening the mutable source pathname.
    raw_bytes = raw_weights.read_bytes()
    config_bytes = config.read_bytes()
    if (hashlib.sha256(raw_bytes).hexdigest() != expected_raw.get("model.safetensors") or
            hashlib.sha256(config_bytes).hexdigest() != expected_raw.get("config.json")):
        raise RuntimeError("raw SigLip snapshot changed while capturing export inputs")
    raw_fd = os.memfd_create("nc-rted-raw-siglip", flags=0)
    try:
        offset = 0
        while offset < len(raw_bytes):
            offset += os.write(raw_fd, raw_bytes[offset:])
        os.lseek(raw_fd, 0, os.SEEK_SET)
        raw_path = f"/proc/self/fd/{raw_fd}"
        with safe_open(raw_path, framework="pt", device="cpu") as raw:
            expected = {name for name in raw.keys() if name.startswith("vision_model.") and
                        not name.startswith("vision_model.encoder.layers.26.") and not name.startswith("vision_model.head.")}
            if len(expected) != 421 or set(retained) != expected:
                raise RuntimeError("final Stage2 retained vision keys differ from raw deleted-layer binding")
            for name, value in retained.items():
                source = raw.get_tensor(name)
                if source.shape != value.shape:
                    raise RuntimeError(f"final Stage2 vision tensor shape differs: {name}")
    finally:
        os.close(raw_fd)
    parent = output.parent
    with allocation_lock(parent):
        # Check while holding the same coordinator used for publication; an
        # earlier pre-lock check would let two exporters overwrite each other.
        estimated = sum(value.numel() * value.element_size() for value in retained.values()) + len(config.read_bytes()) + _OVERHEAD_BYTES
        if shutil.disk_usage(parent).free <= _RESERVE + estimated:
            raise RuntimeError("vision export would violate the 20 GiB free-space reserve")
        temporary = Path(tempfile.mkdtemp(prefix=".nc-rted-vision-", dir=parent))
        try:
            os.chmod(temporary, 0o700)
            # Preserve the config bytes consumed after the initial tree check;
            # the final tree recheck detects any source-path replacement.
            (temporary / "config.json").write_bytes(config_bytes)
            save_file({name: value.contiguous() for name, value in sorted(retained.items())}, str(temporary / "model.safetensors"), metadata={"format": "pt"})
            files = {name: sha256_file(temporary / name) for name in ("config.json", "model.safetensors")}
            provenance = {"schema": SCHEMA, "source_export": str(source_export), "source_export_sha256": source_export_sha256,
                          "source_key_prefix": source_key_prefix, "tensor_count": len(retained), "tensor_dtype": "bfloat16",
                          "raw_config_path": str(config), "raw_config_sha256": files["config.json"], "files": files,
                          "source_key_map": {name: source_key_prefix + name for name in sorted(retained)}}
            (temporary / "nc_rted_provenance.json").write_text(json.dumps(provenance, sort_keys=True, indent=2) + "\n")
            for path in (temporary / "config.json", temporary / "model.safetensors", temporary / "nc_rted_provenance.json"):
                with path.open("rb") as handle:
                    os.fsync(handle.fileno())
            if _file_hashes(raw_snapshot) != expected_raw:
                raise RuntimeError("raw SigLip snapshot changed during export")
            if shutil.disk_usage(parent).free <= _RESERVE:
                raise RuntimeError("vision export would violate the 20 GiB free-space reserve")
            _rename_noreplace(temporary, output)
            directory = os.open(parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
    return {"snapshot": str(output), "tensor_count": len(retained), "files": files,
            "provenance_sha256": sha256_file(output / "nc_rted_provenance.json")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-export", type=Path, required=True)
    parser.add_argument("--source-export-sha256", required=True)
    parser.add_argument("--raw-snapshot", type=Path, required=True)
    parser.add_argument("--raw-snapshot-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-key-prefix", default=DEFAULT_SOURCE_KEY_PREFIX)
    print(json.dumps(export(**vars(parser.parse_args())), sort_keys=True))


if __name__ == "__main__":
    main()
