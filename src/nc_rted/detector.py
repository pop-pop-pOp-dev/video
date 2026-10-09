"""Frozen RT-DETR-R50 COCO binding for NC-RTED observations."""
from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager, nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Protocol

import torch
from PIL import Image
import numpy as np
from .observation_cache import cache_key
from .storage_lock import allocation_lock

COCO_PERSON_CLASS = 0
RTDETR_MODEL_ID = "PekingU/rtdetr_r50vd_coco_o365"
RTDETR_PROVENANCE_FILE = "nc_rted_provenance.json"
_STAGING_RESERVE_BYTES = 20 * 1024 ** 3


class DetectorError(RuntimeError): pass


@dataclass(frozen=True)
class Detection:
    box_xyxy: tuple[float, float, float, float]
    class_id: int
    confidence: float


@dataclass(frozen=True)
class CausalWindowObservation:
    """Feature window plus audit-only relation identities and COCO class pairs."""

    features: object
    relation_class_pairs: tuple[tuple[int, int], ...]
    relation_ids: tuple[str, ...]


def _normalized_box(box: torch.Tensor, width: int, height: int) -> tuple[float, float, float, float] | None:
    """Clip RT-DETR's unconstrained xyxy output to the actual image extent."""
    values = box.detach().float().cpu().numpy()
    if values.shape != (4,) or not np.isfinite(values).all():
        raise DetectorError("RT-DETR emitted a nonfinite bounding box")
    left, top, right, bottom = values.tolist()
    left, right = np.clip((left, right), 0.0, float(width))
    top, bottom = np.clip((top, bottom), 0.0, float(height))
    if right <= left or bottom <= top:
        return None
    return (float(left / width), float(top / height), float(right / width), float(bottom / height))


def sha256_file(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b""): digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _copy_verified(source: Path, destination: Path, expected_sha256: str) -> None:
    """Copy one source snapshot file while binding the staged bytes to provenance."""
    content=source.read_bytes()
    if hashlib.sha256(content).hexdigest()!=expected_sha256:
        raise DetectorError(f"RT-DETR source changed before staging: {source.name}")
    with allocation_lock(destination.parent):
        block=max(4096,os.statvfs(destination.parent).f_frsize)
        allocation=((len(content)+block-1)//block)*block+2*block
        if shutil.disk_usage(destination.parent).free < _STAGING_RESERVE_BYTES+allocation:
            raise DetectorError('RT-DETR staging would violate the 20 GiB free-space reserve')
        with destination.open("xb") as writer:
            writer.write(content)
            writer.flush()
            os.fsync(writer.fileno())
        if sha256_file(destination) != expected_sha256:
            raise DetectorError(f"RT-DETR source changed while staging: {source.name}")


@contextmanager
def _bound_rtdetr_snapshot(snapshot: Path, expected_files: dict[str, str]):
    """Stage immutable, provenance-verified RT-DETR load inputs privately.

    Transformers opens paths itself, so auditing a source path and subsequently
    giving that path to ``from_pretrained`` leaves a replacement interval.  The
    staged directory is instead the sole path passed to Transformers.
    """
    required = {"config.json", "preprocessor_config.json", "model.safetensors"}
    if not required.issubset(expected_files) or any(not isinstance(value, str) or len(value) != 64
                                                     for value in expected_files.values()):
        raise DetectorError("RT-DETR provenance lacks supported config, processor, or safetensors hashes")
    model_files = {name for name in expected_files if name.endswith((".safetensors", ".bin", ".index.json"))}
    if model_files != {"model.safetensors"}:
        raise DetectorError("RT-DETR binding supports only one provenance-listed model.safetensors file")
    root = Path(tempfile.gettempdir())
    # Hold one filesystem allocation lock through model staging and release
    # after construction removes the private snapshot. Other model loads wait;
    # cache/journal writes use this same coordinator.
    with allocation_lock(root):
        staged_bytes = sum((snapshot / name).stat().st_size for name in required)
        block=max(4096,os.statvfs(root).f_frsize)
        if shutil.disk_usage(root).free <= _STAGING_RESERVE_BYTES + staged_bytes + 8*block:
            raise DetectorError("RT-DETR staging would violate the 20 GiB free-space reserve")
        with tempfile.TemporaryDirectory(prefix="nc-rted-rtdetr-", dir=root) as temporary:
            staged = Path(temporary)
            os.chmod(staged, 0o700)
            for name in sorted(required):
                _copy_verified(snapshot / name, staged / name, expected_files[name])
            yield staged


def _assert_staged_files(staged: Path, expected_files: dict[str, str]) -> None:
    for name in ("config.json", "preprocessor_config.json", "model.safetensors"):
        if sha256_file(staged / name) != expected_files[name]:
            raise DetectorError(f"staged RT-DETR input changed: {name}")


def _assert_rtdetr_processor_binding(processor, expected_preprocessor: dict, processor_class) -> None:
    expected = processor_class.from_dict(expected_preprocessor).to_dict()
    if _canonical_json(processor.to_dict()) != _canonical_json(expected):
        raise DetectorError("loaded RT-DETR processor differs from the verified preprocessor configuration")


def _assert_rtdetr_config_binding(config, expected_config: dict, config_class) -> None:
    expected = config_class.from_dict(expected_config).to_dict()
    actual = config.to_dict()
    # from_pretrained records its local load directory here, and may infer a
    # runtime dtype from the staged weights. Neither field is source-config
    # semantics; the exact tensor binding below verifies the latter separately.
    for value in (expected, actual):
        value.pop("_name_or_path", None)
        value.pop("torch_dtype", None)
        value.pop("_attn_implementation_autoset", None)
    if _canonical_json(actual) != _canonical_json(expected):
        raise DetectorError("loaded RT-DETR configuration differs from the verified configuration")


def _assert_rtdetr_weight_binding(model, weights: Path) -> None:
    try:
        from safetensors import safe_open
    except ImportError as error:
        raise DetectorError("safetensors is required to verify the RT-DETR weight binding") from error
    actual = model.state_dict()
    with safe_open(str(weights), framework="pt", device="cpu") as source:
        expected = set(source.keys())
        actual_keys = set(actual)
        if expected - actual_keys:
            raise DetectorError("loaded RT-DETR state keys differ from the verified safetensors snapshot")
        # Transformers 4.47 exposes RT-DETR decoder heads a second time at the
        # model root. They are aliases of the serialized decoder tensors, not
        # independent learned state. Require every nonserialized state key to
        # be exactly one such same-storage alias; a new buffer or parameter is
        # not accepted merely because the serialized subset matches.
        for name in sorted(actual_keys - expected):
            value = actual[name]
            aliases = [candidate for candidate in expected
                       if actual[candidate].shape == value.shape
                       and actual[candidate].dtype == value.dtype
                       and actual[candidate].data_ptr() == value.data_ptr()]
            if len(aliases) != 1:
                raise DetectorError("loaded RT-DETR state includes an unbound nonserialized tensor")
        for name in sorted(expected):
            source_value = source.get_tensor(name)
            loaded = actual[name].detach().cpu()
            if source_value.shape != loaded.shape or source_value.dtype != loaded.dtype or not torch.equal(source_value, loaded):
                raise DetectorError(f"loaded RT-DETR weight differs from verified safetensors snapshot: {name}")


class FrozenRTDetr:
    """Explicit local snapshot only; uses Transformers RT-DETR postprocessing."""
    def __init__(self, snapshot: Path, device: str="cpu", score_threshold: float=.3,
                 numerical_policy_identity: str | None = None):
        if not 0 <= score_threshold <= 1: raise DetectorError("invalid score threshold")
        self.snapshot=Path(snapshot)
        required=("config.json","preprocessor_config.json",RTDETR_PROVENANCE_FILE)
        if not self.snapshot.is_dir() or any(not (self.snapshot/name).is_file() for name in required):
            raise DetectorError("RT-DETR requires a complete explicit local snapshot")
        provenance=json.loads((self.snapshot/RTDETR_PROVENANCE_FILE).read_text())
        if not {"model_id":RTDETR_MODEL_ID,"architecture":"rtdetr_r50vd","num_labels":80,"person_class_id":COCO_PERSON_CLASS}.items() <= provenance.items():
            raise DetectorError("RT-DETR snapshot provenance is not the pinned R50VD COCO binding")
        expected_files=provenance.get("files")
        actual_files={str(path.relative_to(self.snapshot)):sha256_file(path) for path in sorted(self.snapshot.rglob("*")) if path.is_file() and path.name != RTDETR_PROVENANCE_FILE}
        if not isinstance(expected_files,dict) or actual_files != expected_files:
            raise DetectorError("RT-DETR snapshot files do not match its pinned provenance")
        import inspect
        import transformers
        from transformers import RTDetrConfig, RTDetrForObjectDetection, RTDetrImageProcessor
        # Never give Transformers the mutable source directory.  It may open
        # config, processor and weights at different times, so stage every load
        # input before the first library call and bind the constructed objects
        # to those staged bytes afterwards.
        with _bound_rtdetr_snapshot(self.snapshot, expected_files) as staged:
            expected_config = json.loads((staged / "config.json").read_text())
            expected_preprocessor = json.loads((staged / "preprocessor_config.json").read_text())
            self.processor = RTDetrImageProcessor.from_pretrained(staged, local_files_only=True)
            self.model = RTDetrForObjectDetection.from_pretrained(
                staged, local_files_only=True, use_safetensors=True
            ).to(device).eval()
            _assert_staged_files(staged, expected_files)
            _assert_rtdetr_processor_binding(self.processor, expected_preprocessor, RTDetrImageProcessor)
            _assert_rtdetr_config_binding(self.model.config, expected_config, RTDetrConfig)
            _assert_rtdetr_weight_binding(self.model, staged / "model.safetensors")
            # The safetensors reader above reopens the staged path; check it
            # again before allowing the temporary binding to disappear.
            _assert_staged_files(staged, expected_files)
        labels=getattr(self.model.config,"id2label",{})
        if self.model.config.model_type != "rt_detr" or self.model.config.num_labels != 80 or str(labels.get(0," ")).lower() != "person":
            raise DetectorError("snapshot configuration is not 80-class COCO RT-DETR with person=0")
        backbone = getattr(self.model.config, "backbone_config", None)
        depths = backbone.get("depths") if isinstance(backbone, dict) else getattr(backbone, "depths", None)
        if tuple(depths or ()) != (3, 4, 6, 3):
            raise DetectorError("snapshot configuration is not an RT-DETR R50 backbone")
        for parameter in self.model.parameters(): parameter.requires_grad_(False)
        self.device=torch.device(device); self.score_threshold=score_threshold
        implementation = {
            "adapter_sha256": sha256_file(Path(__file__)),
            "transformers_version": transformers.__version__,
            "torch_version": str(torch.__version__),
            "cuda_version": torch.version.cuda,
            "device": str(self.device),
            "dtype": str(next(self.model.parameters()).dtype),
            "cuda_device_name": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else None,
            "model_source_sha256": sha256_file(Path(inspect.getfile(RTDetrForObjectDetection))),
            "processor_source_sha256": sha256_file(Path(inspect.getfile(RTDetrImageProcessor))),
        }
        self._identity={"model_id":RTDETR_MODEL_ID,"snapshot":str(self.snapshot.resolve()),"files":actual_files,"provenance":provenance,"preprocess":self.processor.to_dict(),"score_threshold":self.score_threshold,"implementation":implementation,"numerical_policy_identity":numerical_policy_identity}

    def identity(self) -> dict:
        return self._identity

    @torch.inference_mode()
    def detect(self, image: Image.Image) -> tuple[Detection, ...]:
        inputs=self.processor(images=image,return_tensors="pt").to(self.device)
        outputs=self.model(**inputs)
        width,height=image.size
        result=self.processor.post_process_object_detection(outputs,target_sizes=torch.tensor([[height,width]],device=self.device),threshold=self.score_threshold)[0]
        detections=[]
        for box,score,label in zip(result["boxes"].cpu(),result["scores"].cpu(),result["labels"].cpu()):
            normalized = _normalized_box(box, width, height)
            if normalized is None:
                continue
            detections.append(Detection(normalized,int(label),float(score)))
        return tuple(detections[:8])


def _normalized_siglip_vision_config(config) -> dict:
    if not hasattr(config, "to_dict") or not callable(config.to_dict):
        raise DetectorError("loaded SigLip tower lacks an effective vision configuration")
    value = dict(config.to_dict())
    # ``from_pretrained`` may annotate the local model path and set its private
    # attention-backend initialization marker. Neither value changes the
    # serialized vision operator; every actual vision-config setting stays
    # bound, including layer_norm_eps and dropout.
    value.pop("_name_or_path", None)
    value.pop("_attn_implementation_autoset", None)
    return value


def _assert_siglip_effective_vision_config(inner, expected_vision: dict) -> dict:
    actual_config = getattr(inner, "config", None)
    config_class = type(actual_config)
    factory = getattr(config_class, "from_dict", None)
    if not isinstance(expected_vision, dict) or not callable(factory):
        raise DetectorError("loaded SigLip tower lacks a usable effective vision configuration")
    expected = _normalized_siglip_vision_config(factory(dict(expected_vision)))
    actual = _normalized_siglip_vision_config(actual_config)
    if _canonical_json(actual) != _canonical_json(expected):
        raise DetectorError("loaded SigLip effective vision configuration differs from verified snapshot")
    return expected


class InheritedSigLipAdapter:
    """Adapter over an already-loaded ReactVAU ``SigLipVisionTower`` instance."""
    def __init__(self, tower, snapshot: Path, *, expected_parent_export_sha256: str | None = None,
                 expected_parent_export: str | Path | None = None,
                 expected_parent_key_prefix: str = "base_model.model.model.vision_tower.vision_tower.",
                 expected_raw_config_sha256: str | None = None,
                 numerical_policy_identity: str | None = None):
        import inspect
        if (tower.__class__.__module__ != "llava.model.multimodal_encoder.siglip_encoder" or
                tower.__class__.__name__ != "SigLipVisionTower" or not callable(tower) or
                not getattr(tower, "is_loaded", False) or not hasattr(tower, "vision_tower")):
            raise DetectorError("expected an already-loaded original SigLipVisionTower")
        self.tower, self.snapshot = tower, Path(snapshot).resolve()
        if not self.snapshot.is_dir() or not (self.snapshot/"config.json").is_file() or not (self.snapshot/"model.safetensors").is_file():
            raise DetectorError("missing inherited SigLip snapshot")
        loaded_path = Path(str(tower.vision_tower_name)).resolve()
        self._derived_binding = None
        derived_requested = expected_parent_export_sha256 is not None or expected_parent_export is not None
        if derived_requested:
            if expected_parent_export_sha256 is None or expected_parent_export is None:
                raise DetectorError("derived SigLip binding requires both parent path and SHA-256")
            from .frozen_vision import bind_derived_final_stage2_vision
            self._derived_binding = bind_derived_final_stage2_vision(
                self.snapshot, expected_parent_export_sha256=expected_parent_export_sha256,
                expected_parent_export=expected_parent_export,
                expected_parent_key_prefix=expected_parent_key_prefix,
                expected_raw_config_sha256=expected_raw_config_sha256)
        elif loaded_path != self.snapshot:
            if expected_parent_export_sha256 is None:
                raise DetectorError("loaded SigLip tower path differs from bound snapshot")
        source = Path(inspect.getsourcefile(tower.__class__) or "")
        if not source.is_file() or source.name != "siglip_encoder.py":
            raise DetectorError("SigLip tower source is not the inherited encoder")
        self._source_sha256 = sha256_file(source)
        config_bytes = (self._derived_binding.config_bytes if self._derived_binding is not None
                        else (self.snapshot / "config.json").read_bytes())
        self._config_sha256 = hashlib.sha256(config_bytes).hexdigest()
        self._weights_sha256 = (self._derived_binding.weights_sha256 if self._derived_binding is not None
                                else sha256_file(self.snapshot/"model.safetensors"))
        self._weights_signature = (None if self._derived_binding is not None
                                   else self._signature(self.snapshot/"model.safetensors"))
        config=json.loads(config_bytes)
        vision=config.get("vision_config",{})
        if config.get("model_type")!="siglip" or (vision.get("image_size"),vision.get("patch_size"),vision.get("hidden_size"),vision.get("num_hidden_layers")) != (384,14,1152,27):
            raise DetectorError("snapshot is not Google SigLIP SO400M patch14-384")
        processor = tower.image_processor
        if (processor.__class__.__module__ != "llava.model.multimodal_encoder.siglip_encoder" or
                processor.__class__.__name__ != "SigLipImageProcessor" or tuple(processor.size) != (384,384) or
                tuple(processor.image_mean) != (.5,.5,.5) or tuple(processor.image_std) != (.5,.5,.5) or
                getattr(processor.resample, "name", None) != "BICUBIC" or processor.rescale_factor != 1/255):
            raise DetectorError("loaded SigLip processor differs from inherited 384 bicubic processor")
        inner = tower.vision_tower
        layers = getattr(getattr(inner, "vision_model", None), "encoder", None)
        if (not hasattr(layers, "layers") or len(layers.layers) != 26 or
                inner.vision_model.head.__class__.__name__ != "Identity"):
            raise DetectorError("loaded SigLip tower did not remove only final encoder layer and head")
        # The outer ReactVAU wrapper keeps a default config even after loading.
        # Bind the inner model config that created its layer norms, activations,
        # and attention behavior, then separately account for the intended
        # final-layer/head removal above.
        self._effective_vision_config = _assert_siglip_effective_vision_config(inner, vision)
        self._processor_binding = self._processor_signature()
        if self._processor_binding[-2:] != ("channels_first", (384,384)):
            raise DetectorError("loaded SigLip channel/crop configuration differs")
        self._verify_weights()
        self._freeze_eval()
        self._tower_state_signature = self._state_signature()
        self._identity = {"repo":"google/siglip-so400m-patch14-384","snapshot":str(self.snapshot),"config_sha256":self._config_sha256,"weights_sha256":self._weights_sha256,"tower_source_sha256":self._source_sha256,"feature_layer":"hidden_states[-1] after deleted layer 26","processor":"ReactVAU SigLipImageProcessor direct bicubic resize 384","dtype":str(tower.dtype),"device":str(tower.device),"encoding_impl":"ReactVAU SigLipVisionTower.forward(chunk_size)","torch_version":str(torch.__version__),"cuda_version":torch.version.cuda,
                          "processor_config":self._processor_binding,
                          "cuda_device_name":torch.cuda.get_device_name(tower.device) if tower.device.type == "cuda" else None,
                          "numerical_policy_identity": numerical_policy_identity}
        if self._derived_binding is not None:
            from .frozen_vision import provenance_digest
            self._identity["derived_final_stage2_vision"] = {
                "parent_export_sha256": self._derived_binding.parent_export_sha256,
                "source_key_prefix": self._derived_binding.source_key_prefix,
                "tensor_count": len(self._derived_binding.source_key_map),
                "provenance_digest": provenance_digest(self._derived_binding),
                "loaded_tower_path": str(loaded_path),
            }

    @staticmethod
    def _signature(path: Path):
        status=path.stat()
        return (status.st_dev,status.st_ino,status.st_size,status.st_mtime_ns,status.st_ctime_ns)

    def _processor_signature(self):
        processor = self.tower.image_processor
        return (processor.__class__.__module__, processor.__class__.__name__,
                tuple(processor.size), tuple(processor.image_mean), tuple(processor.image_std),
                getattr(processor.resample, "name", None), processor.rescale_factor,
                getattr(processor.data_format, "value", processor.data_format),
                (processor.crop_size.get("height"),processor.crop_size.get("width")))

    def _state_signature(self):
        return tuple((name, tuple(value.shape), str(value.dtype), str(value.device), value.data_ptr(), value._version)
                     for name,value in self.tower.vision_tower.state_dict().items())

    def _freeze_eval(self):
        self.tower.eval(); self.tower.vision_tower.eval()
        if self.tower.training or self.tower.vision_tower.training or any(parameter.requires_grad for parameter in self.tower.vision_tower.parameters()):
            raise DetectorError("loaded SigLip tower must remain frozen and in eval mode")

    def _verify_weights(self):
        if self._derived_binding is not None:
            from .frozen_vision import verify_loaded_derived_vision
            verify_loaded_derived_vision(self._derived_binding, self.tower.vision_tower.state_dict())
            return
        try:
            from safetensors import safe_open
        except ImportError as error:
            raise DetectorError("safetensors is required to verify the inherited SigLip binding") from error
        actual=self.tower.vision_tower.state_dict()
        with safe_open(str(self.snapshot/"model.safetensors"), framework="pt", device="cpu") as source:
            expected={name for name in source.keys() if name.startswith("vision_model.") and
                      not name.startswith("vision_model.encoder.layers.26.") and not name.startswith("vision_model.head.")}
            if set(actual) != expected:
                raise DetectorError("loaded SigLip state keys differ from inherited deleted-layer binding")
            for name in sorted(expected):
                source_value=source.get_tensor(name)
                loaded=actual[name].detach().cpu()
                # ReactVAU may load BF16/FP16 source weights. Compare against the
                # source rounded into the loaded dtype, not an FP32 approximation.
                if source_value.shape != loaded.shape or not torch.equal(source_value.to(dtype=loaded.dtype),loaded):
                    raise DetectorError(f"loaded SigLip weight differs from bound snapshot: {name}")

    def _assert_intact(self):
        if not self.tower.is_loaded:
            raise DetectorError("bound SigLip snapshot changed after adapter construction")
        if (self._derived_binding is None and
                (sha256_file(self.snapshot/"config.json") != self._config_sha256 or
                 self._signature(self.snapshot/"model.safetensors") != self._weights_signature)):
            raise DetectorError("bound SigLip snapshot changed after adapter construction")
        self._freeze_eval()
        if self._processor_signature() != self._processor_binding:
            raise DetectorError("loaded SigLip preprocessing changed after adapter construction")
        if (len(self.tower.vision_tower.vision_model.encoder.layers) != 26 or
                not isinstance(self.tower.vision_tower.vision_model.head, torch.nn.Identity)):
            raise DetectorError("loaded SigLip feature extraction changed after adapter construction")
        if _assert_siglip_effective_vision_config(self.tower.vision_tower,
                                                   self._effective_vision_config) != self._effective_vision_config:
            raise DetectorError("loaded SigLip effective vision configuration changed after adapter construction")
        if self._state_signature() != self._tower_state_signature:
            raise DetectorError("loaded SigLip tower changed after adapter construction")
        if self._derived_binding is not None:
            from .frozen_vision import verify_loaded_derived_vision
            verify_loaded_derived_vision(self._derived_binding, self.tower.vision_tower.state_dict())
    @torch.inference_mode()
    def __call__(self, images: list[Image.Image]) -> torch.Tensor:
        self._assert_intact()
        pixels=self.tower.image_processor.preprocess(images,return_tensors="pt")["pixel_values"]
        pixels=pixels.to(device=self.tower.device,dtype=self.tower.dtype)
        features=self.tower(pixels)
        if isinstance(features,list): features=torch.cat(features,dim=0)
        if not isinstance(features,torch.Tensor) or features.ndim != 3 or features.shape[1:] != (729,1152): raise DetectorError("inherited SigLip tower did not return final [N,729,1152] patches")
        return features.detach()
    def identity(self):
        self._assert_intact()  # Validate even when every frame will hit the cache.
        return dict(self._identity)


def observe_causal_window(images: list[Image.Image], timestamps: list[float], query_s: float,
                          detector: FrozenRTDetr, siglip_encode, cache, media_hash: str,
                          siglip_identity: dict, *, window_start_s: float | None = None) -> CausalWindowObservation:
    """Connect frozen detector/SigLIP outputs to the existing causal tracker/features.

    ``siglip_encode`` receives uncropped RGB images and returns [N,729,1152] final
    hidden-state patch features using the inherited processor.  It is deliberately
    injected so the accepted ReactVAU adapter owns model loading.
    """
    from .observation import causal_frame_indices, pool_patch_regions
    from .tracking import Detection as TrackDetection, FrameObservations, causal_tracks
    from .features import FrozenFrameFeatures, assemble_relation_features
    pts=np.asarray(timestamps,dtype=np.float64)
    if window_start_s is None:
        indexes=causal_frame_indices(pts,query_s)
    else:
        if (not math.isfinite(window_start_s) or window_start_s < 0 or window_start_s >= query_s
                or query_s - window_start_s > 8.0):
            raise DetectorError("window start must define a nonempty causal interval of at most eight seconds")
        first_tick=math.floor(window_start_s * 2.0) + 1
        last_tick=math.floor(query_s * 2.0)
        ticks=np.arange(first_tick,last_tick+1,dtype=np.float64)/2.0
        indexes=np.searchsorted(pts,ticks,side="right")-1
        indexes=indexes[indexes >= 0]
        indexes=np.unique(indexes[(pts[indexes] > window_start_s) & (pts[indexes] <= query_s)]).astype(np.int64)
        if indexes.size > 16:
            raise DetectorError("causal window exceeds 2 FPS eight-second observation cap")
    selected=[images[index] for index in indexes]
    detector_identity=detector.identity()
    # The cache stores individual frames. GPU BF16 kernels can change their
    # rounding with batch shape, so a frame's value must not depend on how many
    # other frames missed the cache. Bind and enforce singleton encoding here;
    # the inherited adapter's original task-specific batching remains unchanged.
    siglip_identity = {"encoder": siglip_identity,
                       "observation_encoding": "singleton-frame/v1",
                       "observation_source_sha256": sha256_file(Path(__file__))}
    keys = [cache_key(media_hash, float(pts[index]), detector_identity, siglip_identity) for index in indexes]
    population = cache.population(keys) if hasattr(cache, "population") else nullcontext()
    with population:
        missing=[]; patches={}; detections={}
        for local,index in enumerate(indexes):
            key=cache_key(media_hash,float(pts[index]),detector_identity,siglip_identity)
            value=cache.get(key)
            if value is None: missing.append((local,key))
            else:
                patches[local]=value['patches']
                saved=value.get('detections')
                if saved is not None: detections[local]=tuple(Detection(tuple(row['box_xyxy']),int(row['class_id']),float(row['confidence'])) for row in saved)
        for local, key in missing:
            encoded = siglip_encode([selected[local]])
            if (not isinstance(encoded, torch.Tensor) or encoded.shape != (1,729,1152)
                    or not encoded.is_floating_point() or not bool(torch.isfinite(encoded).all())):
                raise DetectorError("SigLIP adapter must return finite [1,729,1152] frozen patches")
            # Match FrozenFrameCache's canonical layout before any ROI/global
            # reduction. Original SigLIP may return column-major patch views;
            # changing strides only on cache hits changes floating reductions.
            patches[local] = encoded[0].detach().cpu().contiguous().clone()
        tracked=[]; frozen=[]
        for local,index in enumerate(indexes):
            patch=patches[local]
            frame_detections=detections.get(local)
            if frame_detections is None:
                frame_detections=detector.detect(selected[local])
                detections[local]=frame_detections
                key=cache_key(media_hash,float(pts[index]),detector_identity,siglip_identity)
                cache.put(key,{'patches':patch,'detections':[{'box_xyxy':item.box_xyxy,'class_id':item.class_id,'confidence':item.confidence} for item in frame_detections]})
            boxes=torch.tensor([item.box_xyxy for item in frame_detections],dtype=torch.float32)
            if len(frame_detections):
                appearance,valid=pool_patch_regions(patch,boxes)
                values=tuple(TrackDetection(item.box_xyxy,item.class_id,item.confidence,tuple(appearance[number].float().tolist())) for number,item in enumerate(frame_detections) if bool(valid[number]))
            else: values=()
            tracked.append(FrameObservations(float(pts[index]),values))
            frozen.append(FrozenFrameFeatures(float(pts[index]),patch))
        tracking=causal_tracks(tracked)
        features=assemble_relation_features(tracking,tuple(frozen),query_s,window_start_s=window_start_s)
        tracks={track.local_index: track for track in tracking.tracks}
        relation_ids=tuple(f"{relation.first_track}:{relation.second_track}" for relation in features.relations)
        try:
            class_pairs=tuple((tracks[relation.first_track].class_id, tracks[relation.second_track].class_id)
                              for relation in features.relations)
        except KeyError as error:
            raise DetectorError("assembled relation is absent from its tracking result") from error
        return CausalWindowObservation(features, class_pairs, relation_ids)


def build_causal_window(images: list[Image.Image], timestamps: list[float], query_s: float,
                        detector: FrozenRTDetr, siglip_encode, cache, media_hash: str,
                        siglip_identity: dict, *, window_start_s: float | None = None):
    """Compatibility wrapper returning only the existing feature assembly."""
    return observe_causal_window(images,timestamps,query_s,detector,siglip_encode,cache,media_hash,
                                 siglip_identity,window_start_s=window_start_s).features
