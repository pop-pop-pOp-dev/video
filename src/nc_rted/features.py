"""Causal relation-window features assembled from tracking snapshots.

All visual inputs are caller-supplied frozen SigLIP patch features.  This module
does not decode frames, run a detector, use an ID feature, or read any future
frame.  Missing relation/time cells remain masked and NaN-valued.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

import numpy as np
import torch
from torch import Tensor

from .observation import background_feature, pool_patch_regions, relative_geometry
from .tracking import COCO_PERSON_CLASS, MAX_TRACK_PAIRS, CausalTrack, TrackObservation, TrackingResult, TrackingStatus


SIGLIP_FEATURE_DIM = 1152
COCO_CLASS_COUNT = 80
GEOMETRY_DIM = 5
VELOCITY_DIM = 2
CELL_FEATURE_DIM = 4 * SIGLIP_FEATURE_DIM + 2 * GEOMETRY_DIM + VELOCITY_DIM
PROCESS_FEATURE_DIM = 3 * SIGLIP_FEATURE_DIM + 3 * GEOMETRY_DIM + VELOCITY_DIM


def _slices(blocks: tuple[tuple[str, int], ...]) -> dict[str, slice]:
    start = 0
    result = {}
    for name, width in blocks:
        result[name] = slice(start, start + width)
        start += width
    return result


STUDENT_BLOCK_SLICES = _slices((
    ("local_first", SIGLIP_FEATURE_DIM), ("local_second", SIGLIP_FEATURE_DIM),
    ("joint", SIGLIP_FEATURE_DIM), ("global", SIGLIP_FEATURE_DIM),
    ("current_geometry", GEOMETRY_DIM), ("geometry_change", GEOMETRY_DIM),
    ("relative_velocity", VELOCITY_DIM),
))
PROCESS_BLOCK_SLICES = _slices((
    ("initial_geometry", GEOMETRY_DIM), ("current_geometry", GEOMETRY_DIM),
    ("geometry_change", GEOMETRY_DIM), ("relative_velocity", VELOCITY_DIM),
    ("local_first_change", SIGLIP_FEATURE_DIM), ("local_second_change", SIGLIP_FEATURE_DIM),
    ("joint_change", SIGLIP_FEATURE_DIM),
))


class FeatureStatus(str, Enum):
    OK = "ok"
    TRACKING_FAILURE = "tracking_failure"
    NO_RELATION_PAIRS = "no_relation_pairs"
    INVALID_INPUT = "invalid_input"


@dataclass(frozen=True)
class FrozenFrameFeatures:
    timestamp_s: float
    patches: Tensor


@dataclass(frozen=True)
class RelationWindowFeatures:
    first_track: int
    second_track: int
    cell_mask: np.ndarray  # [4] bool
    feature_valid: np.ndarray  # [4] bool; explicit bridge validity mask
    observed_times_s: np.ndarray  # [4], actual representative times; NaN when missing
    cell_right_boundaries_s: np.ndarray  # [4], fixed grid coordinates, never observations
    student_cells: np.ndarray  # [4, CELL_FEATURE_DIM], NaN where mask is false
    process_cells: np.ndarray  # [4, PROCESS_FEATURE_DIM], NaN where mask is false
    static_background: np.ndarray  # [SIGLIP_FEATURE_DIM], NaN when unavailable
    background_valid: bool
    static_class_composition: np.ndarray  # [COCO_CLASS_COUNT]
    initial_geometry: np.ndarray  # [GEOMETRY_DIM]


@dataclass(frozen=True)
class FeatureAssemblyResult:
    status: FeatureStatus
    relations: tuple[RelationWindowFeatures, ...]
    invalid_reason: str | None = None


def _as_numpy(value: Tensor) -> np.ndarray:
    return value.detach().float().cpu().numpy().astype(np.float32, copy=False)


def _validate_patch_frame(frame: FrozenFrameFeatures, window_start_s: float, query_s: float) -> None:
    if not math.isfinite(frame.timestamp_s) or frame.timestamp_s <= window_start_s or frame.timestamp_s > query_s:
        raise ValueError("frozen frame is outside the causal observation window")
    if frame.patches.ndim != 2 or frame.patches.shape != (729, SIGLIP_FEATURE_DIM):
        raise ValueError("expected frozen SigLIP patch features [729,1152]")
    if not frame.patches.is_floating_point() or not bool(torch.isfinite(frame.patches).all()):
        raise ValueError("frozen SigLIP patch features must be finite floating values")


def _track_by_index(tracks: tuple[CausalTrack, ...]) -> dict[int, CausalTrack]:
    result = {track.local_index: track for track in tracks}
    if len(result) != len(tracks) or sorted(result) != list(range(len(tracks))):
        raise ValueError("track indexes must be unique contiguous local indexes")
    return result


def _shared_observations(first: CausalTrack, second: CausalTrack) -> list[tuple[TrackObservation, TrackObservation]]:
    first_by_frame = {item.frame_index: item for item in first.observations}
    second_by_frame = {item.frame_index: item for item in second.observations}
    return [(first_by_frame[index], second_by_frame[index]) for index in sorted(first_by_frame.keys() & second_by_frame.keys())]


def _historical_relation_pairs(tracks: tuple[CausalTrack, ...]) -> list[tuple[CausalTrack, CausalTrack, list[tuple[TrackObservation, TrackObservation]]]]:
    """Pairs observed anywhere in the window, including pairs absent at query time."""
    selected = []
    for first_position, first in enumerate(tracks):
        for second in tracks[first_position + 1:]:
            if first.class_id != COCO_PERSON_CLASS and second.class_id != COCO_PERSON_CLASS:
                continue
            shared = _shared_observations(first, second)
            if shared:
                selected.append((first, second, shared))
    selected.sort(key=lambda item: (item[2][0][0].frame_index, item[0].local_index, item[1].local_index))
    return selected[:MAX_TRACK_PAIRS]


def _union_box(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> Tensor:
    return torch.tensor([[min(first[0], second[0]), min(first[1], second[1]),
                          max(first[2], second[2]), max(first[3], second[3])]], dtype=torch.float32)


def _class_composition(tracks: tuple[CausalTrack, ...], frame_index: int) -> np.ndarray:
    composition = np.zeros(COCO_CLASS_COUNT, dtype=np.float32)
    for track in tracks:
        if any(item.frame_index == frame_index for item in track.observations):
            if not 0 <= track.class_id < COCO_CLASS_COUNT:
                raise ValueError("COCO class must be in [0,80)")
            composition[track.class_id] += 1.0
    total = float(composition.sum())
    return composition / total if total else composition


def _frame_boxes(tracks: tuple[CausalTrack, ...], frame_index: int) -> Tensor:
    boxes = [item.box_xyxy for track in tracks for item in track.observations if item.frame_index == frame_index]
    return torch.tensor(boxes, dtype=torch.float32) if boxes else torch.empty((0, 4), dtype=torch.float32)


def _pooled_features(patches: Tensor, first: TrackObservation, second: TrackObservation) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    device = patches.device
    boxes = torch.tensor([first.box_xyxy, second.box_xyxy], dtype=torch.float32, device=device)
    local, valid = pool_patch_regions(patches, boxes)
    joint, joint_valid = pool_patch_regions(patches, _union_box(first.box_xyxy, second.box_xyxy).to(device))
    if not bool(valid.all() and joint_valid.all()):
        raise ValueError("tracked boxes must remain valid ROI observations")
    geometry = _as_numpy(relative_geometry(boxes[0], boxes[1]))
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0
    return _as_numpy(local[0]), _as_numpy(local[1]), _as_numpy(joint[0]), _as_numpy(centers[1] - centers[0]), geometry


def causal_cell_right_boundaries(query_s: float, window_start_s: float | None = None) -> np.ndarray:
    """Return fixed grid coordinates, not assertions of observed frame times."""
    if not math.isfinite(query_s) or query_s < 0:
        raise ValueError("query timestamp must be finite and nonnegative")
    start = query_s - 8.0 if window_start_s is None else window_start_s
    if not math.isfinite(start) or start < 0 or not 0 < query_s - start <= 8.0:
        raise ValueError("window start must be finite, nonnegative, and within eight seconds of query")
    return np.minimum(start + np.arange(1, 5, dtype=np.float32) * 2.0, query_s).astype(np.float32)


def assemble_relation_features(tracking: TrackingResult, frames: tuple[FrozenFrameFeatures, ...], query_s: float,
                               window_start_s: float | None = None) -> FeatureAssemblyResult:
    """Build fixed relation features for the complete causal eight-second window.

    Callers must supply every decoded observation selected for this window.  The
    function does not select a last/high-score block: full descriptions must
    invoke it for every causal eight-second block before their fixed aggregation.
    """
    try:
        if tracking.status in {TrackingStatus.INVALID_INPUT, TrackingStatus.EMPTY_OBSERVATIONS}:
            return FeatureAssemblyResult(FeatureStatus.TRACKING_FAILURE, (), tracking.status.value)
        if not math.isfinite(query_s) or query_s < 0:
            raise ValueError("query timestamp must be finite and nonnegative")
        window_start_s = query_s - 8.0 if window_start_s is None else window_start_s
        if not math.isfinite(window_start_s) or window_start_s < 0 or not 0 < query_s - window_start_s <= 8.0:
            raise ValueError("window start must be finite, nonnegative, and within eight seconds of query")
        if not frames:
            raise ValueError("at least one frozen causal frame is required")
        for frame in frames:
            _validate_patch_frame(frame, window_start_s, query_s)
        timestamps = np.asarray([frame.timestamp_s for frame in frames], dtype=np.float64)
        if np.any(np.diff(timestamps) <= 0):
            raise ValueError("frozen frame timestamps must be strictly increasing")
        tracks = _track_by_index(tracking.tracks)
        pairs = _historical_relation_pairs(tuple(tracks.values()))
        if not pairs:
            return FeatureAssemblyResult(FeatureStatus.NO_RELATION_PAIRS, ())

        relations = []
        cell_right_boundaries = causal_cell_right_boundaries(query_s, window_start_s)
        for first, second, shared in pairs:
            student = np.full((4, CELL_FEATURE_DIM), np.nan, dtype=np.float32)
            process = np.full((4, PROCESS_FEATURE_DIM), np.nan, dtype=np.float32)
            mask = np.zeros(4, dtype=bool)
            observed_times = np.full(4, np.nan, dtype=np.float32)
            # The anchor is the earliest actual shared observation, independent
            # of the representative selected later for a two-second cell.
            anchor_first, anchor_second = shared[0]
            if anchor_first.frame_index >= len(frames):
                raise ValueError("track references a missing frozen frame")
            anchor_frame = frames[anchor_first.frame_index]
            if (abs(anchor_frame.timestamp_s - anchor_first.timestamp_s) > 1e-9
                    or abs(anchor_frame.timestamp_s - anchor_second.timestamp_s) > 1e-9):
                raise ValueError("tracking and frozen-frame timestamps disagree")
            anchor_local_first, anchor_local_second, anchor_joint, anchor_centers, anchor_geometry = _pooled_features(
                anchor_frame.patches, anchor_first, anchor_second,
            )
            boxes = _frame_boxes(tuple(tracks.values()), anchor_first.frame_index).to(anchor_frame.patches.device)
            background, background_valid = background_feature(anchor_frame.patches, boxes)
            static_background = _as_numpy(background) if background_valid else np.full(SIGLIP_FEATURE_DIM, np.nan, dtype=np.float32)
            class_composition = _class_composition(tuple(tracks.values()), anchor_first.frame_index)
            # Later real observations win within each cell; missing cells retain NaN.
            selected: dict[int, tuple[TrackObservation, TrackObservation]] = {}
            for first_observation, second_observation in shared:
                if first_observation.frame_index >= len(frames):
                    raise ValueError("track references a missing frozen frame")
                frame = frames[first_observation.frame_index]
                if abs(frame.timestamp_s - first_observation.timestamp_s) > 1e-9 or abs(frame.timestamp_s - second_observation.timestamp_s) > 1e-9:
                    raise ValueError("tracking and frozen-frame timestamps disagree")
                cell = min(int(math.floor((frame.timestamp_s - window_start_s) / 2.0)), 3)
                selected[cell] = (first_observation, second_observation)
            for cell in sorted(selected):
                first_observation, second_observation = selected[cell]
                frame = frames[first_observation.frame_index]
                local_first, local_second, joint, centers_delta, geometry = _pooled_features(
                    frame.patches, first_observation, second_observation,
                )
                global_feature = _as_numpy(frame.patches.mean(0))
                visual_change = np.concatenate((local_first - anchor_local_first, local_second - anchor_local_second,
                                                joint - anchor_joint))
                geometry_change = geometry - anchor_geometry
                elapsed = first_observation.timestamp_s - anchor_first.timestamp_s
                if elapsed < 0:
                    raise ValueError("shared observations are not forward-causal")
                velocity = np.zeros(VELOCITY_DIM, dtype=np.float32) if elapsed == 0 else (centers_delta - anchor_centers) / elapsed
                student[cell] = np.concatenate((local_first, local_second, joint, global_feature,
                                                geometry, geometry_change, velocity))
                process[cell] = np.concatenate((anchor_geometry, geometry, geometry_change, velocity, visual_change))
                mask[cell] = True
                observed_times[cell] = first_observation.timestamp_s
            relations.append(RelationWindowFeatures(
                first.local_index, second.local_index, mask, mask.copy(), observed_times, cell_right_boundaries.copy(),
                student, process, static_background, background_valid, class_composition, anchor_geometry,
            ))
        return FeatureAssemblyResult(FeatureStatus.OK, tuple(relations))
    except (TypeError, ValueError, IndexError) as error:
        return FeatureAssemblyResult(FeatureStatus.INVALID_INPUT, (), str(error))
