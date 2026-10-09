"""Bounded adapter from the inherited Stage2 caption path to NC-RTED samples."""
from __future__ import annotations

import copy
from dataclasses import dataclass
import math
from typing import Protocol, TYPE_CHECKING

import torch

from .caption_sampling import OriginalSamplingAudit, OriginalSamplingAuditReader
from .bridge import ObservationBatch
from .batches import caption_block_endpoints
from .inherited_memory import caption_memory_from_patches
from .task_inputs import FrozenTaskContext, TaskInputError, TrainingCatalog
if TYPE_CHECKING:
    from .train_worker import SampleMaterial


class CaptionProviderError(TaskInputError):
    pass


@dataclass(frozen=True)
class CaptionObservationAudit:
    """Output from the accepted causal detector adapter for one caption media item."""
    observations: ObservationBatch
    original_sampled_frame_times: tuple[float, ...]
    original_observed_seconds: float
    original_time_message: str
    detector_identity: dict


class CaptionCausalObserver(Protocol):
    """The real adapter owns cache-resolver decode and calls observe_causal_window."""
    ready: bool

    def observe_causal_window(self, *, sample_id: str,
                              annotation: dict, sampling: OriginalSamplingAudit) -> CaptionObservationAudit: ...


DECODE_FIELDS = ("start", "end", "fps", "video_read_type")
DECODE_ARGUMENTS = ("local_num_frames", "frames_upbound", "frames_lowbound", "sample_type",
                    "time_msg", "frame_aspect_ratio", "frame_grid_pinpoints", "max_num_pixels")


def _validate_decode_fields(annotation: dict, instruction: dict) -> None:
    for key in DECODE_FIELDS:
        # The inherited dataset explicitly inserts its configured reader type.
        # The adopted caption path is decord when the original row has no override.
        expected = instruction.get(key, "decord") if key == "video_read_type" else instruction.get(key)
        actual = annotation.get(key, "decord") if key == "video_read_type" else annotation.get(key)
        if actual != expected or (key != "video_read_type" and (key in annotation) != (key in instruction)):
            raise CaptionProviderError(f"inherited decode-controlling annotation changed: {key}")


def _decode_arguments(dataset) -> dict:
    args = getattr(dataset, "data_args", None)
    return copy.deepcopy({key: getattr(args, key) for key in DECODE_ARGUMENTS if hasattr(args, key)})


def _ordinary_observations(observed: ObservationBatch) -> ObservationBatch:
    values = (observed.features, observed.valid, observed.observed_times)
    if any(not isinstance(value, torch.Tensor) or value.requires_grad for value in values):
        raise CaptionProviderError("caption observations must be frozen tensors")
    with torch.inference_mode(False), torch.no_grad():
        return ObservationBatch(*(value.clone().detach() for value in values))


def _caption_index(dataset, catalog: TrainingCatalog) -> dict[str, int]:
    rows = getattr(dataset, "list_data_dict", None)
    if not isinstance(rows, list):
        raise CaptionProviderError("expected inherited LazySupervisedDataset list_data_dict")
    expected = {}
    for task in catalog.tasks.values():
        if task.task != "caption" or task.instruction is None:
            continue
        original = task.instruction
        identity = (original.get("id"), original.get("video"))
        if (type(identity[0]) not in {int, str} or isinstance(identity[0], str) and not identity[0]
                or not isinstance(identity[1], str) or not identity[1] or identity in expected):
            raise CaptionProviderError("fixed caption instructions have invalid or duplicate identities")
        expected[identity] = task
    index: dict[str, int] = {}
    for position, row in enumerate(rows):
        if not isinstance(row, dict):
            raise CaptionProviderError("inherited dataset annotation is not a mapping")
        task = expected.get((row.get("id"), row.get("_reactvau_relative_video")))
        if task is None:
            continue
        original = task.instruction
        if any(row.get(name) != original.get(name) for name in ("id", "task", "type", "conversations")):
            raise CaptionProviderError("inherited caption annotation differs from fixed instruction identity")
        _validate_decode_fields(row, original)
        if not isinstance(row.get("_reactvau_relative_video"), str):
            raise CaptionProviderError("inherited caption annotation lost frozen media binding")
        if task.sample_id in index:
            raise CaptionProviderError("caption task maps to multiple inherited dataset rows")
        index[task.sample_id] = position
    if set(index) != {task.sample_id for task in expected.values()}:
        raise CaptionProviderError("inherited dataset does not exactly cover fixed training caption identities")
    return index


def _extract_original_pixels(sample: dict) -> tuple[torch.Tensor, tuple[int, int]]:
    image = sample.get("image")
    if not isinstance(image, list) or len(image) != 1:
        raise CaptionProviderError("inherited caption sample must retain exactly one video image entry")
    pixels, image_size, modality = image[0]
    if modality != "video" or not isinstance(pixels, torch.Tensor) or pixels.ndim != 4:
        raise CaptionProviderError("inherited caption sample lost original video preprocessing")
    if not (isinstance(image_size, tuple) and len(image_size) == 2 and all(isinstance(item, int) and item > 0 for item in image_size)):
        raise CaptionProviderError("inherited caption sample has invalid image size")
    if pixels.requires_grad or not bool(torch.isfinite(pixels).all()):
        raise CaptionProviderError("inherited caption pixels must be finite and frozen")
    return pixels, image_size


class Stage2CaptionProvider:
    """Use original Stage2 decode/PG/preprocess, then add audited frozen evidence.

    This class accepts only a catalog sample ID at call time. It never receives a
    target answer or a teacher row, and it never creates an alternate prompt,
    media sampler, PG score, or detector output.
    """
    def __init__(self, *, catalog: TrainingCatalog, dataset, model, vision_tower,
                 observer: CaptionCausalObserver):
        self.catalog, self.dataset = catalog, dataset
        self.model, self.vision_tower, self.observer = model, vision_tower, observer
        self._index = _caption_index(dataset, catalog)
        self._bound_media = {index: dataset.list_data_dict[index]["video"] for index in self._index.values()}
        self.sampling_reader = OriginalSamplingAuditReader(dataset)
        self._decode_config = _decode_arguments(dataset)

    @torch.no_grad()
    def __call__(self, sample_id: str) -> SampleMaterial:
        from .train_worker import SampleMaterial
        task = self.catalog.tasks.get(sample_id)
        if task is None or task.task != "caption" or task.instruction is None:
            raise CaptionProviderError("caption provider accepts only a fixed caption sample ID")
        if not getattr(self.observer, "ready", False):
            raise CaptionProviderError("causal caption observation is unavailable without an accepted RT-DETR binding")
        position = self._index[sample_id]
        annotation = self.dataset.list_data_dict[position]
        original = task.instruction
        if any(annotation.get(name) != original.get(name) for name in ("id", "task", "type", "conversations")):
            raise CaptionProviderError("inherited caption dataset identity drifted")
        _validate_decode_fields(annotation, original)
        if _decode_arguments(self.dataset) != self._decode_config:
            raise CaptionProviderError("inherited caption decode configuration changed")
        # Direct _get_item avoids upstream __getitem__ retries that substitute a
        # later sample. It invokes frozen cache resolution, original video decode,
        # image preprocessing, original prompt preprocessing, and PG alignment.
        if (annotation.get("_reactvau_relative_video") != original["video"]
                or annotation.get("video") != self._bound_media[position]):
            raise CaptionProviderError("inherited caption media binding drifted")
        sampling_read = self.sampling_reader.read(position)
        sample = sampling_read.sample
        if sample.get("id") != annotation.get("id"):
            raise CaptionProviderError("inherited caption sample ID drifted")
        pixels, image_size = _extract_original_pixels(sample)
        scores = sample.get("pg_scores")
        if not isinstance(scores, list) or len(scores) != len(pixels) or any(not math.isfinite(float(score)) or not 0 <= float(score) <= 1 for score in scores):
            raise CaptionProviderError("original per-frame PG alignment is missing or invalid")
        # The inherited trainer moves floating image inputs to its model dtype
        # before forward. SigLIP casts its result back to the input pixel dtype.
        # Reproduce that boundary so CPU float32 decode cannot alter BF16 memory.
        raw = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
        parameter = next(raw.get_model().mm_projector.mlp.parameters())
        pixels = pixels.to(device=parameter.device, dtype=parameter.dtype)
        patches = self.vision_tower(pixels, chunk_size=32)
        if not isinstance(patches, torch.Tensor) or patches.shape != (len(pixels), 729, 1152) or patches.requires_grad:
            raise CaptionProviderError("inherited vision tower did not return frozen original patches")
        memory = caption_memory_from_patches(self.model, patches, [float(score) for score in scores])
        # The inherited task preprocessor needs conversations; the frozen
        # evidence producer receives media/decode metadata only, never answers.
        media_metadata = {key: copy.deepcopy(annotation[key]) for key in
                          ("id", "video", "_reactvau_relative_video", *DECODE_FIELDS) if key in annotation}
        audit = self.observer.observe_causal_window(sample_id=sample_id, annotation=media_metadata, sampling=sampling_read.audit)
        if not isinstance(audit, CaptionObservationAudit) or not audit.detector_identity:
            raise CaptionProviderError("causal observer did not produce an auditable detector result")
        if (audit.original_sampled_frame_times != sampling_read.audit.sampled_frame_times
                or audit.original_time_message != sampling_read.audit.time_message):
            raise CaptionProviderError("causal observer changed the original caption sampling/prompt")
        if not math.isfinite(audit.original_observed_seconds) or audit.original_observed_seconds <= 0:
            raise CaptionProviderError("causal observer has no stable observed end")
        if (not audit.original_sampled_frame_times or any(not math.isfinite(value) or value < 0 or value > audit.original_observed_seconds + 1e-6
                                                          for value in audit.original_sampled_frame_times) or any(
                right <= left for left, right in zip(audit.original_sampled_frame_times, audit.original_sampled_frame_times[1:]))):
            raise CaptionProviderError("original caption sampling audit is invalid")
        if len(audit.original_sampled_frame_times) != len(pixels):
            raise CaptionProviderError("original caption sampling audit differs from decoded frame count")
        expected_blocks = caption_block_endpoints(audit.original_observed_seconds)
        if audit.observations.features.shape[1] != len(expected_blocks):
            raise CaptionProviderError("caption relation evidence must cover every 8-second block including tail")
        context = FrozenTaskContext(memory, [pixels], [image_size], audit.original_observed_seconds,
                                    audit.original_sampled_frame_times, audit.original_time_message)
        return SampleMaterial(context=context, observations=_ordinary_observations(audit.observations))
