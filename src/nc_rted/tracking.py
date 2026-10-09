"""Fixed causal instance tracking for NC-RTED observations.

The tracker consumes detections and frozen appearance vectors already computed for
causal frames.  It neither runs a detector nor stores source or global identity
features.  Association is strictly forward in timestamp order.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Iterable

import numpy as np


COCO_PERSON_CLASS = 0
MAX_DETECTIONS_PER_FRAME = 8
MAX_TRACK_PAIRS = 16

# These values are part of the pre-freeze tracker contract.  Change only through
# a new frozen configuration, never as a result of a test-set measurement.
MIN_IOU = 0.30
MIN_COSINE_SIMILARITY = 0.70
MAX_ASSOCIATION_GAP_SECONDS = 1.25


class TrackingStatus(str, Enum):
    OK = "ok"
    EMPTY_OBSERVATIONS = "empty_observations"
    NO_VALID_DETECTIONS = "no_valid_detections"
    NO_PERSON_ENDPOINT = "no_person_endpoint"
    INVALID_INPUT = "invalid_input"


@dataclass(frozen=True)
class Detection:
    """One detector result in normalized ``xyxy`` coordinates.

    ``appearance`` must be a finite nonzero frozen feature vector.  It is used
    transiently for association and is never emitted as an ID-like feature.
    """

    box_xyxy: tuple[float, float, float, float]
    class_id: int
    confidence: float
    appearance: tuple[float, ...]


@dataclass(frozen=True)
class FrameObservations:
    timestamp_s: float
    detections: tuple[Detection, ...]


@dataclass(frozen=True)
class TrackObservation:
    frame_index: int
    timestamp_s: float
    box_xyxy: tuple[float, float, float, float]
    class_id: int
    confidence: float


@dataclass(frozen=True)
class CausalTrack:
    local_index: int
    class_id: int
    observations: tuple[TrackObservation, ...]


@dataclass(frozen=True)
class TrackPair:
    """An unordered local pair, represented with increasing track indices."""

    first_track: int
    second_track: int


@dataclass(frozen=True)
class TrackingResult:
    status: TrackingStatus
    tracks: tuple[CausalTrack, ...]
    pairs: tuple[TrackPair, ...]
    invalid_reason: str | None = None


@dataclass
class _MutableTrack:
    local_index: int
    class_id: int
    observations: list[TrackObservation]
    last_appearance: np.ndarray


def _box_array(box: tuple[float, float, float, float]) -> np.ndarray:
    value = np.asarray(box, dtype=np.float64)
    if value.shape != (4,) or not np.isfinite(value).all():
        raise ValueError("box must contain four finite xyxy coordinates")
    if (value < 0).any() or (value > 1).any() or value[2] <= value[0] or value[3] <= value[1]:
        raise ValueError("box must be normalized and have positive area")
    return value


def _appearance_array(appearance: tuple[float, ...]) -> np.ndarray:
    value = np.asarray(appearance, dtype=np.float64)
    if value.ndim != 1 or value.size == 0 or not np.isfinite(value).all():
        raise ValueError("appearance must be a nonempty finite vector")
    if float(np.linalg.norm(value)) == 0.0:
        raise ValueError("appearance must have nonzero norm")
    return value


def _validate_detection(detection: Detection) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(detection.class_id, int) or detection.class_id < 0:
        raise ValueError("class_id must be a nonnegative integer")
    if not math.isfinite(detection.confidence) or not 0.0 <= detection.confidence <= 1.0:
        raise ValueError("confidence must be finite in [0, 1]")
    return _box_array(detection.box_xyxy), _appearance_array(detection.appearance)


def _iou(first: np.ndarray, second: np.ndarray) -> float:
    top_left = np.maximum(first[:2], second[:2])
    bottom_right = np.minimum(first[2:], second[2:])
    intersection = np.prod(np.maximum(bottom_right - top_left, 0.0))
    union = np.prod(first[2:] - first[:2]) + np.prod(second[2:] - second[:2]) - intersection
    return float(intersection / union)


def _cosine(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.dot(first, second) / (np.linalg.norm(first) * np.linalg.norm(second)))


def _detection_order(detection: Detection, original_index: int) -> tuple:
    # A stable total order makes the per-frame cap independent of input order.
    return (-detection.confidence, detection.class_id, detection.box_xyxy, detection.appearance, original_index)


def _frame_detections(frame: FrameObservations) -> list[tuple[Detection, np.ndarray, np.ndarray]]:
    checked = []
    for original_index, detection in enumerate(frame.detections):
        box, appearance = _validate_detection(detection)
        checked.append((detection, box, appearance, original_index))
    checked.sort(key=lambda item: _detection_order(item[0], item[3]))
    return [(detection, box, appearance) for detection, box, appearance, _ in checked[:MAX_DETECTIONS_PER_FRAME]]


def causal_tracks(frames: Iterable[FrameObservations]) -> TrackingResult:
    """Track supplied causal frames and select at most sixteen current pairs.

    Matching needs same COCO class, a gap no larger than
    ``MAX_ASSOCIATION_GAP_SECONDS``, IoU >= ``MIN_IOU``, and cosine >=
    ``MIN_COSINE_SIMILARITY``.  Candidate matches are greedily accepted in
    descending ``(IoU + cosine)`` order; ties use local track index and the
    deterministic capped frame-detection order.  No future frame can alter a
    previous association, and no missing observation is interpolated.
    """
    try:
        items = tuple(frames)
        if not items:
            return TrackingResult(TrackingStatus.EMPTY_OBSERVATIONS, (), ())
        timestamps = np.asarray([item.timestamp_s for item in items], dtype=np.float64)
        if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
            raise ValueError("frame timestamps must be finite and strictly increasing")

        tracks: list[_MutableTrack] = []
        valid_detections = 0
        for frame_index, frame in enumerate(items):
            detections = _frame_detections(frame)
            valid_detections += len(detections)
            candidates: list[tuple[float, int, int]] = []
            for detection_index, (detection, box, appearance) in enumerate(detections):
                for track in tracks:
                    previous = track.observations[-1]
                    gap = frame.timestamp_s - previous.timestamp_s
                    if track.class_id != detection.class_id or gap <= 0 or gap > MAX_ASSOCIATION_GAP_SECONDS:
                        continue
                    iou = _iou(_box_array(previous.box_xyxy), box)
                    cosine = _cosine(track.last_appearance, appearance)
                    if iou >= MIN_IOU and cosine >= MIN_COSINE_SIMILARITY:
                        candidates.append((-(iou + cosine), track.local_index, detection_index))
            candidates.sort()
            assigned_tracks: set[int] = set()
            assigned_detections: set[int] = set()
            for _, track_index, detection_index in candidates:
                if track_index in assigned_tracks or detection_index in assigned_detections:
                    continue
                detection, box, appearance = detections[detection_index]
                track = tracks[track_index]
                track.observations.append(TrackObservation(
                    frame_index, frame.timestamp_s, detection.box_xyxy, detection.class_id, detection.confidence,
                ))
                track.last_appearance = appearance
                assigned_tracks.add(track_index)
                assigned_detections.add(detection_index)
            for detection_index, (detection, _, appearance) in enumerate(detections):
                if detection_index not in assigned_detections:
                    tracks.append(_MutableTrack(
                        len(tracks), detection.class_id,
                        [TrackObservation(frame_index, frame.timestamp_s, detection.box_xyxy,
                                          detection.class_id, detection.confidence)], appearance,
                    ))
        if not valid_detections:
            return TrackingResult(TrackingStatus.NO_VALID_DETECTIONS, (), ())
        frozen_tracks = tuple(CausalTrack(track.local_index, track.class_id, tuple(track.observations)) for track in tracks)
        endpoints = [track for track in frozen_tracks if track.observations[-1].frame_index == len(items) - 1]
        person_endpoints = [track for track in endpoints if track.class_id == COCO_PERSON_CLASS]
        if not person_endpoints:
            return TrackingResult(TrackingStatus.NO_PERSON_ENDPOINT, frozen_tracks, ())
        # Pairs are unordered.  Prefer pairs most recently co-observed, then the
        # strongest endpoint-confidence sum, then local indexes as the final tie.
        pairs: list[tuple[tuple, TrackPair]] = []
        for position, first in enumerate(endpoints):
            for second in endpoints[position + 1:]:
                if first.class_id != COCO_PERSON_CLASS and second.class_id != COCO_PERSON_CLASS:
                    continue
                pair = TrackPair(min(first.local_index, second.local_index), max(first.local_index, second.local_index))
                last_frame = max(first.observations[-1].frame_index, second.observations[-1].frame_index)
                confidence = first.observations[-1].confidence + second.observations[-1].confidence
                pairs.append(((-last_frame, -confidence, pair.first_track, pair.second_track), pair))
        pairs.sort(key=lambda item: item[0])
        return TrackingResult(TrackingStatus.OK, frozen_tracks, tuple(pair for _, pair in pairs[:MAX_TRACK_PAIRS]))
    except (TypeError, ValueError) as error:
        return TrackingResult(TrackingStatus.INVALID_INPUT, (), (), str(error))
