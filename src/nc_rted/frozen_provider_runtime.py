"""Assemble original frozen visual providers without retaining the language model.

The inherited loader still checks every original weight. No language-model
forward is run: after assembly only its exact vision tower/projector are kept.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import gc
import hashlib
import json
import os
from pathlib import Path
import tempfile

import torch
from torch import nn

from . import production_runtime as production
from .detection_media import StreamingDetectionReader
from .detection_provider import FrozenDetectionProvider


PROJECT = Path("/root/autodl-tmp/lookaway-wm")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


class FrozenVisualModel(nn.Module):
    """Only the original visual objects required by inherited_memory remain."""

    def __init__(self, original):
        super().__init__()
        raw = original.get_base_model() if hasattr(original, "get_base_model") else original
        self.config = copy.deepcopy(raw.config)
        self.visual = nn.Module()
        self.visual.mm_projector = raw.get_model().mm_projector
        self.vision_tower = original.get_vision_tower()
        if any(parameter.requires_grad for parameter in self.parameters()):
            raise ValueError("frozen producer received trainable visual weights")
        self.eval()

    def get_model(self):
        return self.visual

    def get_vision_tower(self):
        return self.vision_tower


@dataclass
class FrozenProviderRuntime:
    catalog: object
    producer_identity: dict
    provider: object
    visual_model: FrozenVisualModel
    execution: dict


def _write_runtime(document):
    content = _canonical(document)
    root = PROJECT / ".cache" / "frozen-material-runtime"
    root.mkdir(parents=True, exist_ok=True)
    target = root / (hashlib.sha256(content).hexdigest() + ".json")
    if target.exists():
        if target.read_bytes() != content:
            raise ValueError("frozen producer runtime collision")
        return target
    descriptor, temporary = tempfile.mkstemp(prefix=".runtime-", dir=root)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError:
            if target.read_bytes() != content:
                raise ValueError("concurrent frozen producer runtime differs")
        directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return target


def assemble_frozen_provider(runtime_path, runtime_sha256, device):
    """Use the original caption/detection computation on an explicit CUDA device."""
    path = Path(runtime_path).resolve()
    if _sha(path) != runtime_sha256:
        raise ValueError("source runtime hash differs")
    original = json.loads(path.read_text())
    if not isinstance(original, dict) or torch.device(device).type != "cuda":
        raise ValueError("visual material production requires bound runtime and CUDA")
    document = copy.deepcopy(original)
    document["run"].update(mode="diagnostic", run_id="diagnostic:frozen-material-provider",
                           diagnostic_updates=1, device=str(torch.device(device)))
    derived = _write_runtime(document)
    # assemble_caption_preparation itself performs the full existing preflight.
    # Avoid calling load_manifest first, which would duplicate that same scan.
    manifest = production.RuntimeManifest(derived, _sha(derived), document)
    assembled = production.assemble_caption_preparation(manifest)
    captions = assembled.caption_provider
    visual = FrozenVisualModel(captions.model)
    captions.model = visual
    # No optimizer, teacher, bridge, or language-model forward is constructed.
    del assembled
    gc.collect()
    torch.cuda.empty_cache()
    if any(parameter.device != torch.device(device) for parameter in visual.parameters()):
        raise ValueError("frozen visual model is not entirely on the requested GPU")
    observer = captions.observer
    protocols = production._protocols(document["fast"]["protocols"])
    reader = StreamingDetectionReader(
        document["fast"]["snapshot"], snapshot_sha256=document["fast"]["snapshot_sha256"],
        fast_identity=document["fast"]["identity"], protocols=protocols,
        encode=observer.observer.siglip)
    detections = FrozenDetectionProvider(visual, captions.catalog, protocols,
                                         reader=reader, observation_reader=observer)
    module_names = (
        "frozen_provider_runtime", "production_runtime", "loading", "caption_provider",
        "caption_sampling", "caption_observation_cache", "detection_provider",
        "detection_media", "inherited_memory", "media_observer", "detector",
        "batches", "features", "tracking", "observation", "numerics",
    )
    sources = {}
    for name in module_names:
        source = Path(__file__).with_name(name + ".py")
        if not source.is_file():
            raise ValueError("missing frozen provider implementation: " + name)
        sources[name] = _sha(source)
    frozen_inputs = {key: value for key, value in original.items()
                     if key not in {"run", "hashes", "teacher"}}
    identity = {
        "schema": "nc_rted_frozen_material_producer/v1",
        "catalog_sha256": captions.catalog.identity,
        "frozen_inputs": frozen_inputs,
        "implementation": sources,
    }

    @torch.no_grad()
    def provider(sample_id):
        visual.eval()
        if captions.catalog.tasks[sample_id].task == "caption":
            return captions(sample_id)
        return detections(sample_id)

    execution = {"device": str(torch.device(device)), "torch": torch.__version__,
                 "cuda": torch.version.cuda, "device_name": torch.cuda.get_device_name(device),
                 "source_runtime": str(path), "source_runtime_sha256": runtime_sha256,
                 "derived_runtime": str(derived), "language_model_retained": False,
                 "formal_training": False}
    return FrozenProviderRuntime(captions.catalog, identity, provider, visual, execution)
