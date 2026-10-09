"""Read the inherited Stage2 caption sampling audit without changing its decode path."""
from __future__ import annotations

from dataclasses import dataclass
import math
from types import MethodType


class CaptionSamplingError(RuntimeError):
    pass


@dataclass(frozen=True)
class OriginalSamplingAudit:
    annotation_id: int | str
    video: str
    relative_video: str
    process_video_argument: str
    frame_indices: tuple[int, ...]
    fps: float
    sampled_frame_times: tuple[float, ...]
    time_message: str
    aligned_pg_scores: tuple[float, ...]


@dataclass(frozen=True)
class SamplingRead:
    """The original `_get_item` result and audit from its one video decode call."""
    sample: dict
    audit: OriginalSamplingAudit


class OriginalSamplingAuditReader:
    """Capture the existing `process_video` result during fail-closed `_get_item`.

    This is intentionally an instance adapter rather than a replacement decoder.
    The temporarily installed method delegates to the original bound method with
    exactly the supplied arguments, which preserves the inherited Stage2 cache
    resolver and all original sampling decisions.
    """
    def __init__(self, dataset):
        if not isinstance(getattr(dataset, "list_data_dict", None), list) or not callable(getattr(dataset, "_get_item", None)):
            raise CaptionSamplingError("expected an inherited Stage2 LazySupervisedDataset instance")
        self.dataset = dataset
        self._active = False

    def read(self, index: int) -> SamplingRead:
        if type(index) is not int or not 0 <= index < len(self.dataset.list_data_dict):
            raise CaptionSamplingError("invalid inherited dataset index")
        if self._active:
            raise CaptionSamplingError("sampling audit reader is not reentrant")
        annotation = self.dataset.list_data_dict[index]
        if (not isinstance(annotation, dict) or type(annotation.get("id")) not in {int, str}
                or not all(isinstance(annotation.get(key), str) and annotation[key] for key in ("video", "_reactvau_relative_video"))):
            raise CaptionSamplingError("annotation lacks frozen caption/media identity")
        original_process_video = self.dataset.process_video
        captured = []

        def capture(_dataset, video_file, data_anno, data_args):
            if data_anno is not annotation:
                raise CaptionSamplingError("inherited decoder received a different annotation")
            result = original_process_video(video_file, data_anno, data_args)
            if not isinstance(result, tuple) or len(result) != 4:
                raise CaptionSamplingError("inherited process_video returned an unsupported result")
            captured.append((video_file, result))
            return result

        self._active = True
        self.dataset.process_video = MethodType(capture, self.dataset)
        try:
            # Do not call __getitem__: released upstream retry logic can return a
            # later sample after a failure. _get_item preserves the requested index.
            sample = self.dataset._get_item(index)
        finally:
            self.dataset.process_video = original_process_video
            self._active = False
        if len(captured) != 1 or not isinstance(sample, dict) or sample.get("id") != annotation["id"]:
            raise CaptionSamplingError("caption audit did not retain exactly one requested inherited sample")
        video_file, (_, time_message, frame_indices, fps) = captured[0]
        if not isinstance(video_file, str) or not isinstance(time_message, str) or not math.isfinite(float(fps)) or float(fps) <= 0:
            raise CaptionSamplingError("invalid inherited video sampling result")
        from numbers import Integral
        if any(isinstance(value, bool) or not isinstance(value, Integral) or value < 0 for value in frame_indices):
            raise CaptionSamplingError("inherited frame indices must be nonnegative integers")
        indices = tuple(int(value) for value in frame_indices)
        times = tuple(value / float(fps) for value in indices)
        if not indices or any(right <= left for left, right in zip(indices, indices[1:])):
            raise CaptionSamplingError("inherited frame indices must be strictly increasing")
        scores = sample.get("pg_scores")
        if not isinstance(scores, list) or len(scores) != len(indices) or any(not math.isfinite(float(score)) or not 0 <= float(score) <= 1 for score in scores):
            raise CaptionSamplingError("inherited per-frame PG alignment is missing or invalid")
        return SamplingRead(sample, OriginalSamplingAudit(
            annotation["id"], annotation["video"], annotation["_reactvau_relative_video"],
            video_file, indices, float(fps), times, time_message,
            tuple(float(score) for score in scores),
        ))
