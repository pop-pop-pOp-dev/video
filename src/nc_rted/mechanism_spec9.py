"""Bounded, post-freeze consumers for NC-RTED Section 9 diagnostics."""
from __future__ import annotations

from collections import Counter
import math
from typing import Callable, Mapping, Sequence

class Spec9Error(ValueError): pass


def teacher_summary(rows: Sequence[Mapping]) -> dict:
    """Summarize permitted teacher records without asserting relation truth."""
    total = valid = 0; quality = []; relation = Counter(); time = Counter(); candidates = Counter(); rejection = Counter()
    for row in rows:
        total += 1
        if row.get("aux_valid") is not True:
            rejection[str(row.get("rejection", "missing"))] += 1; continue
        mask, values, ids = row.get("mask"), row.get("F_positions"), row.get("relation_ids")
        if not isinstance(mask, list) or not isinstance(values, list) or not isinstance(ids, list) or len(mask) != len(values):
            raise Spec9Error("teacher record shape differs")
        q = row.get("F_quality")
        if not isinstance(q, (int, float)) or not math.isfinite(q): raise Spec9Error("teacher quality differs")
        valid += 1; quality.append(float(q)); candidates[len(ids)] += 1
        for index, (supported, mass) in enumerate(zip(mask, values)):
            if supported:
                relation[index // 4] += float(mass); time[index % 4] += float(mass)
    return {"records": total, "aux_valid": valid, "coverage": valid / total if total else 0.,
            "quality": {"mean": sum(quality) / len(quality) if quality else 0., "values": quality},
            "relation_marginal": dict(relation), "time_marginal": dict(time),
            "candidate_count": dict(candidates), "rejection": dict(rejection)}


def teacher_summary_from_store(path: str) -> dict:
    """Consume the committed teacher store through its verified reader."""
    from .teacher_store import build_teachers_from_store
    document = build_teachers_from_store(path)
    if not isinstance(document, dict) or not isinstance(document.get("rows"), list):
        raise Spec9Error("verified teacher store did not produce teacher rows")
    return teacher_summary(document["rows"])


def recompute_geometry_checks(*, assembler: Callable, tracking, frames, query_s: float,
                              translation_xy: tuple[float, float],
                              trajectory_translation_xy: tuple[float, float] | None = None) -> dict:
    """Run raw trajectory perturbations through the accepted feature producer.

    This intentionally compares only geometry and velocity blocks.  Translating
    boxes changes ROI pooling locations and therefore does not establish an
    invariant for visual patch features.
    """
    import numpy as np
    from .features import PROCESS_BLOCK_SLICES, STUDENT_BLOCK_SLICES
    from .tracking import CausalTrack, TrackObservation, TrackingResult
    dx, dy = translation_xy
    if not all(math.isfinite(value) for value in translation_xy): raise Spec9Error("translation must be finite")
    trajectory_dx, trajectory_dy = translation_xy if trajectory_translation_xy is None else trajectory_translation_xy
    if not all(math.isfinite(value) for value in (trajectory_dx, trajectory_dy)):
        raise Spec9Error("trajectory translation must be finite")
    window_start_s = max(0.0, query_s - 8.0)
    duration = query_s - window_start_s
    if duration <= 0:
        raise Spec9Error("geometry window duration is invalid")
    def move(box, timestamp_s, *, trajectory=False):
        scale = (timestamp_s - window_start_s) / duration if trajectory else 1.0
        shift_x, shift_y = (trajectory_dx, trajectory_dy) if trajectory else (dx, dy)
        x1, y1, x2, y2 = box
        result = x1 + shift_x * scale, y1 + shift_y * scale, x2 + shift_x * scale, y2 + shift_y * scale
        if min(result) < 0 or max(result) > 1: raise Spec9Error("translation leaves normalized image domain")
        return result
    def rebuild(track, boxes):
        return CausalTrack(track.local_index, track.class_id,
            tuple(TrackObservation(item.frame_index, item.timestamp_s, box, item.class_id, item.confidence)
                  for item, box in zip(track.observations, boxes)))
    moved_tracks = tuple(rebuild(track, tuple(move(item.box_xyxy, item.timestamp_s) for item in track.observations))
                         for track in tracking.tracks)
    moved = TrackingResult(tracking.status, moved_tracks, tracking.pairs, tracking.invalid_reason)
    trajectory_tracks = tuple(rebuild(track, tuple(move(item.box_xyxy, item.timestamp_s, trajectory=True)
                                                   for item in track.observations)) for track in tracking.tracks)
    trajectory = TrackingResult(tracking.status, trajectory_tracks, tracking.pairs, tracking.invalid_reason)
    # Keep timestamps/frame order fixed but reverse each observed box sequence.
    # This is a trajectory reversal at the feature producer boundary.
    reversed_tracks = tuple(rebuild(track, tuple(item.box_xyxy for item in reversed(track.observations)))
                            for track in tracking.tracks)
    reversed_tracking = TrackingResult(tracking.status, reversed_tracks, tracking.pairs, tracking.invalid_reason)
    original, translated, trajectory_result, reversed_result = (assembler(tracking, frames, query_s, window_start_s=window_start_s),
                                                                  assembler(moved, frames, query_s, window_start_s=window_start_s),
                                                                  assembler(trajectory, frames, query_s, window_start_s=window_start_s),
                                                                  assembler(reversed_tracking, frames, query_s, window_start_s=window_start_s))
    if not original.relations or getattr(original.status, "value", original.status) != "ok":
        raise Spec9Error("geometry source has no comparable assembled relation")
    if original.status != translated.status or len(original.relations) != len(translated.relations):
        raise Spec9Error("translation changed feature assembly status")
    if original.status != trajectory_result.status or len(original.relations) != len(trajectory_result.relations):
        raise Spec9Error("time-varying translation changed feature assembly status")
    if original.status != reversed_result.status or len(original.relations) != len(reversed_result.relations):
        raise Spec9Error("trajectory reversal changed feature assembly status")
    if any(not np.array_equal(first.feature_valid, second.feature_valid)
           for first, second in zip(original.relations, translated.relations)):
        raise Spec9Error("translation changed feature support")
    if any(not np.array_equal(first.feature_valid, second.feature_valid)
           for first, second in zip(original.relations, trajectory_result.relations)):
        raise Spec9Error("time-varying translation changed feature support")
    geometry = (STUDENT_BLOCK_SLICES["current_geometry"], STUDENT_BLOCK_SLICES["geometry_change"],
                PROCESS_BLOCK_SLICES["initial_geometry"], PROCESS_BLOCK_SLICES["current_geometry"],
                PROCESS_BLOCK_SLICES["geometry_change"])
    velocity = (STUDENT_BLOCK_SLICES["relative_velocity"], PROCESS_BLOCK_SLICES["relative_velocity"])
    def equal_blocks(left, right, blocks):
        return all(np.allclose(left.student_cells[:, block], right.student_cells[:, block], atol=1e-6, rtol=1e-6, equal_nan=True)
                   if block in geometry[:2] or block == STUDENT_BLOCK_SLICES["relative_velocity"]
                   else np.allclose(left.process_cells[:, block], right.process_cells[:, block], atol=1e-6, rtol=1e-6, equal_nan=True)
                   for block in blocks)
    translated_geometry = all(equal_blocks(first, second, geometry) for first, second in zip(original.relations, translated.relations))
    translated_velocity = all(equal_blocks(first, second, velocity) for first, second in zip(original.relations, translated.relations))
    trajectory_geometry = all(equal_blocks(first, second, geometry) for first, second in zip(original.relations, trajectory_result.relations))
    trajectory_velocity = all(equal_blocks(first, second, velocity) for first, second in zip(original.relations, trajectory_result.relations))
    reversal_changes_velocity = any(
        not np.allclose(first.process_cells[:, PROCESS_BLOCK_SLICES["relative_velocity"]],
                        second.process_cells[:, PROCESS_BLOCK_SLICES["relative_velocity"]], equal_nan=True)
        for first, second in zip(original.relations, reversed_result.relations)
    )
    return {"raw_coordinate_translation_preserves_relative_geometry": translated_geometry,
            "overall_translation_preserves_relative_velocity": translated_velocity,
            "common_time_varying_translation_preserves_relative_geometry": trajectory_geometry,
            "common_time_varying_translation_preserves_relative_velocity": trajectory_velocity,
            "trajectory_reversal_changes_signed_motion": reversal_changes_velocity,
            "scope": "raw normalized boxes recomputed through accepted feature assembler"}


def validate_cold_receipt(receipt: Mapping) -> dict:
    required = {"cold_cache", "elapsed_seconds", "peak_gpu_bytes", "detector_seconds", "process_seconds", "slow_calls"}
    if set(receipt) != required or receipt["cold_cache"] is not True:
        raise Spec9Error("cold end-to-end receipt differs")
    for key in required - {"cold_cache"}:
        value = receipt[key]
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0: raise Spec9Error("cold receipt value differs")
    return dict(receipt)
