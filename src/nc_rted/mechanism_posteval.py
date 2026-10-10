"""Post-freeze Section 9 stratification and cold-runtime receipts."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
import hashlib
import time
from typing import Any, Callable, Mapping, Sequence
from pathlib import Path

import numpy as np

import torch


class PostEvalError(ValueError): pass


REQUIRED_TASKS = {"R0"} | {f"{group}:seed{seed}" for group in ("A", "U", "S", "F") for seed in (17, 42, 2026)}
STRATA = ("fast_trigger", "short_event", "long_video", "small_object", "no_candidate", "association_failure", "reference_insufficient")
SHORT_EVENT_SECONDS = 8.0
LONG_VIDEO_SECONDS = 600.0
_MEDIA_FPS: dict[tuple[str, str], float | None] = {}


def _summarize_frozen_rows(records: Sequence[Mapping]) -> dict:
    """Group rows admitted by ``stratify_completed_matrix`` only."""
    values = {name: defaultdict(list) for name in STRATA}
    unsupported = {name: 0 for name in STRATA}
    for row in records:
        if not isinstance(row, Mapping) or row.get("model_task") not in REQUIRED_TASKS:
            raise PostEvalError("prediction record is not bound to the frozen thirteen-task matrix")
        identity = row.get("identity")
        if not isinstance(identity, str): raise PostEvalError("prediction record lacks identity")
        metadata = row.get("diagnostic_metadata")
        if not isinstance(metadata, Mapping):
            for name in STRATA: unsupported[name] += 1
            continue
        for name in STRATA:
            value = metadata.get(name)
            if type(value) is bool:
                values[name][str(value).lower()].append({"model_task": row["model_task"], "identity": identity})
            elif value is None: unsupported[name] += 1
            else: raise PostEvalError(f"diagnostic metadata {name} must be bool or null")
    return {"schema": "nc_rted_mechanism_posteval_strata/v2", "frozen_tasks": sorted(REQUIRED_TASKS),
            "strata": {name: {key: sorted(items, key=lambda item: (item["model_task"], item["identity"]))
                               for key, items in cells.items()} for name, cells in values.items()},
            "unsupported": unsupported}


def _outcome_cell() -> dict:
    return {"normal_denominator": 0, "anomalous_denominator": 0,
            "normal_false_positive": 0, "anomalous_recalled": 0}


def _accumulate(cells: dict, key: tuple[str, str, str, str], *, label: bool, positive: bool) -> None:
    cell = cells.setdefault(key, _outcome_cell())
    if label:
        cell["anomalous_denominator"] += 1
        cell["anomalous_recalled"] += int(positive)
    else:
        cell["normal_denominator"] += 1
        cell["normal_false_positive"] += int(positive)


def _render(cells: Mapping[tuple[str, str, str, str], dict]) -> list[dict]:
    return [{"model_task": task, "dataset": dataset, "stratum": stratum, "membership": membership, **value}
            for (task, dataset, stratum, membership), value in sorted(cells.items())]


def _annotation_categories(annotation: Mapping[str, Any]) -> tuple[str, ...]:
    """Use only categories explicitly supplied by the evaluator-opened annotation."""
    value = annotation.get("label")
    values = [value] if isinstance(value, str) else value if isinstance(value, list) else []
    if not all(isinstance(item, str) and item for item in values):
        return ()
    return tuple(sorted(set(values)))


def _bound_media_fps(request: Any) -> float | None:
    path, digest = getattr(request, "media_path", None), getattr(request, "media_sha256", None)
    if not isinstance(path, str) or not isinstance(digest, str) or len(digest) != 64:
        return None
    key = path, digest
    if key in _MEDIA_FPS:
        return _MEDIA_FPS[key]
    try:
        hasher = hashlib.sha256()
        with Path(path).open("rb") as stream:
            for block in iter(lambda: stream.read(8 << 20), b""):
                hasher.update(block)
        if hasher.hexdigest() != digest:
            _MEDIA_FPS[key] = None; return None
        import cv2
        capture = cv2.VideoCapture(path); fps = float(capture.get(cv2.CAP_PROP_FPS)); capture.release()
        _MEDIA_FPS[key] = fps if math.isfinite(fps) and fps > 0 else None
        return _MEDIA_FPS[key]
    except (OSError, ImportError):
        _MEDIA_FPS[key] = None
        return None


def _annotation_temporal_strata(annotation: Mapping[str, Any], total_frames: Any, request: Any) -> dict[str, bool | None]:
    """Apply fixed descriptive duration definitions to evaluator-opened metadata."""
    fps = annotation.get("fps")
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(float(fps)) or float(fps) <= 0:
        fps = _bound_media_fps(request)
    intervals = annotation.get("intervals_raw")
    if (isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(float(fps))
            or float(fps) <= 0 or type(total_frames) is not int or total_frames < 1):
        return {"short_event": None, "long_video": None}
    long_video = total_frames / float(fps) >= LONG_VIDEO_SECONDS
    if not isinstance(intervals, list):
        return {"short_event": None, "long_video": long_video}
    durations = []
    for interval in intervals:
        if (not isinstance(interval, (list, tuple)) or len(interval) != 2 or any(type(value) is not int or value < 0 for value in interval)
                or interval[1] < interval[0]):
            return {"short_event": None, "long_video": long_video}
        durations.append((interval[1] - interval[0] + 1) / float(fps))
    return {"short_event": any(duration <= SHORT_EVENT_SECONDS for duration in durations), "long_video": long_video}


def stratify_completed_matrix(*, frozen: Any, annotations: Mapping[str, Any], make_labels: Callable,
                              decision_threshold: float) -> dict:
    """Read actual frozen stores after the evaluator's full-matrix gate.

    Labels are supplied only by the already-opened official evaluator path.  The
    optional observer fields are taken verbatim from persisted query metadata;
    unavailable fields remain unsupported.
    """
    tasks = getattr(frozen, "tasks", None)
    plan = getattr(frozen, "plan", None)
    if not isinstance(tasks, Mapping) or set(tasks) != REQUIRED_TASKS or plan is None:
        raise PostEvalError("completed evaluator matrix is required before post-evaluation strata")
    if isinstance(decision_threshold, bool) or not isinstance(decision_threshold, (int, float)) or not 0 <= float(decision_threshold) <= 1:
        raise PostEvalError("post-evaluation decision threshold is invalid")
    rows, video_outcomes, triggered_query_outcomes, category_outcomes = [], {}, {}, {}
    category_unsupported = 0
    for task_id, task in sorted(tasks.items()):
        for request in plan.vad:
            record = task.records.get(request.identity)
            if not isinstance(record, Mapping) or not isinstance(record.get("payload"), Mapping):
                raise PostEvalError("frozen VAD record is absent")
            payload = record["payload"]
            annotation = annotations.get((request.dataset, request.media_id))
            if not isinstance(annotation, Mapping):
                raise PostEvalError("official VAD annotation is absent after freeze")
            labels = np.asarray(make_labels(annotation, payload.get("total_frames")))
            scores = payload.get("causal_smoothed_scores")
            queries = payload.get("queries")
            if (labels.ndim != 1 or not np.isin(labels, (0, 1)).all() or not isinstance(scores, list)
                    or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value))
                           or not 0 <= float(value) <= 1 for value in scores)
                    or len(labels) != len(scores) or not isinstance(queries, list)):
                raise PostEvalError("frozen VAD payload differs from evaluator geometry")
            metadata = {name: None for name in STRATA}
            metadata.update(_annotation_temporal_strata(annotation, payload.get("total_frames"), request))
            for name in STRATA:
                if name in {"small_object", "no_candidate", "association_failure", "reference_insufficient"}:
                    continue
                values = [query.get("diagnostic_metadata", {}).get(name) for query in queries
                          if isinstance(query, Mapping) and isinstance(query.get("diagnostic_metadata"), Mapping)
                          and query["diagnostic_metadata"].get(name) is not None]
                if values and all(type(value) is bool for value in values) and len(values) == len(queries):
                    metadata[name] = any(values)
            # The Fast trigger is a persisted query fact, available even when no
            # observer metadata exists.
            triggered = [query.get("triggered") for query in queries if isinstance(query, Mapping)]
            if len(triggered) == len(queries) and all(type(value) is bool for value in triggered):
                metadata["fast_trigger"] = any(triggered)
            video_label = bool(np.any(labels == 1))
            maximum = max(float(value) for value in scores)
            categories = _annotation_categories(annotation)
            if not categories:
                category_unsupported += 1
            for category in categories:
                _accumulate(category_outcomes, (task_id, request.dataset, "category", category), label=video_label,
                            positive=maximum >= float(decision_threshold))
            memberships = [("all", "all")] + [(name, str(value).lower()) for name, value in metadata.items()
                                                  if type(value) is bool]
            for stratum, membership in memberships:
                _accumulate(video_outcomes, (task_id, request.dataset, stratum, membership), label=video_label,
                            positive=maximum >= float(decision_threshold))
            observed_metadata = {name: 0 for name in ("small_object", "no_candidate", "association_failure", "reference_insufficient")}
            for query in queries:
                if not isinstance(query, Mapping) or type(query.get("triggered")) is not bool:
                    raise PostEvalError("frozen VAD query lacks inherited Fast trigger state")
                if not query["triggered"]:
                    continue
                indices, score = query.get("frame_indices"), query.get("final_score")
                if (not isinstance(indices, list) or not indices or any(type(index) is not int or index < 0 or index >= len(labels) for index in indices)
                        or isinstance(score, bool) or not isinstance(score, (int, float))
                        or not math.isfinite(float(score)) or not 0 <= float(score) <= 1):
                    raise PostEvalError("triggered VAD query geometry differs")
                query_label = bool(np.any(labels[np.asarray(indices)] == 1))
                query_memberships = [("all", "all"), ("fast_trigger", "true")]
                for name in ("short_event", "long_video"):
                    value = metadata[name]
                    if type(value) is bool:
                        query_memberships.append((name, str(value).lower()))
                for category in categories:
                    query_memberships.append(("category", category))
                details = query.get("diagnostic_metadata")
                if isinstance(details, Mapping):
                    for name in observed_metadata:
                        value = details.get(name)
                        if type(value) is bool:
                            observed_metadata[name] += 1
                            query_memberships.append((name, str(value).lower()))
                for stratum, membership in query_memberships:
                    _accumulate(triggered_query_outcomes, (task_id, request.dataset, stratum, membership),
                                label=query_label, positive=float(score) >= float(decision_threshold))
            rows.append({"model_task": task_id, "identity": request.identity,
                         "diagnostic_metadata": metadata,
                         "truth": "anomalous" if video_label else "normal",
                         "scores": scores})
    report = _summarize_frozen_rows(rows)
    report["decision_threshold"] = float(decision_threshold)
    report["decision_threshold_source"] = "CLI descriptive sensitivity setting; not fit from official labels or a frozen classification rule"
    report["outcome_scope"] = {
        "video_strata_membership": "temporal strata and explicit annotation categories are video-level metadata",
        "triggered_query_membership": "temporal strata and explicit annotation categories are reused for triggered queries; observer strata are query-level",
        "triggered_query_label": "anomalous when any persisted query frame_indices has an official anomalous frame label",
    }
    report["temporal_strata_definitions"] = {"short_event": {"rule": "any inclusive intervals_raw duration <= 8 seconds",
                                                                   "seconds": SHORT_EVENT_SECONDS},
                                            "long_video": {"rule": "total_frames / verified annotation or plan-bound media fps >= 600 seconds",
                                                           "seconds": LONG_VIDEO_SECONDS}}
    report["video_outcomes"] = _render(video_outcomes)
    report["fast_triggered_query_outcomes"] = _render(triggered_query_outcomes)
    report["triggered_query_metadata_scope"] = "observer strata are measured only on triggered queries with persisted boolean metadata; null or nontriggered queries are not negative evidence"
    report["category_video_outcomes"] = _render(category_outcomes)
    report["category_unsupported"] = category_unsupported
    return report


@dataclass
class ColdRunMeter:
    """Measure an explicitly cache-cleared operation and its constituent work."""
    detector_calls: int = 0
    process_calls: int = 0
    slow_calls: int = 0
    detector_seconds: float = 0.0
    process_seconds: float = 0.0

    def detector(self, operation: Callable[[], object]) -> object:
        if torch.cuda.is_available(): torch.cuda.synchronize()
        started = time.perf_counter()
        try:
            return operation()
        finally:
            if torch.cuda.is_available(): torch.cuda.synchronize()
            self.detector_calls += 1
            self.detector_seconds += time.perf_counter() - started

    def process(self, operation: Callable[[], object]) -> object:
        if torch.cuda.is_available(): torch.cuda.synchronize()
        started = time.perf_counter()
        try:
            return operation()
        finally:
            if torch.cuda.is_available(): torch.cuda.synchronize()
            self.process_calls += 1
            self.process_seconds += time.perf_counter() - started

    def slow(self, operation: Callable[[], object]) -> object:
        self.slow_calls += 1
        return operation()

    def measure(self, operation: Callable[[], object], *, clear_application_caches: Callable[[], None]) -> tuple[object, dict]:
        """Clear the supplied application's caches immediately before timing."""
        if not callable(clear_application_caches):
            raise PostEvalError("cold measurement requires an application cache reset")
        clear_application_caches()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        started = time.perf_counter()
        result = operation()
        if torch.cuda.is_available(): torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        return result, {"cold_cache": True, "elapsed_seconds": elapsed,
                        "peak_gpu_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
                        "detector_seconds": self.detector_seconds, "process_seconds": self.process_seconds,
                        "slow_calls": self.slow_calls}


def measure_production_vad_cold(*, runner: Any, request: Any, observer: Any, cache_factory: Callable[[], Any]) -> tuple[dict, dict]:
    """Measure the accepted production streaming detect/trigger/fusion route.

    Models stay resident; only the observer cache is isolated and empty.  The
    runner must be the production ``BlindVadRunner`` (or equivalent with its
    ``detect`` method), never the mechanism CE/generation executor.
    """
    if not callable(getattr(runner, "detect", None)) or not hasattr(observer, "cache") or not callable(cache_factory):
        raise PostEvalError("production cold measurement requires production VAD runner and observer cache")
    meter, original_cache, original_meter = ColdRunMeter(), observer.cache, getattr(observer, "meter", None)
    cold_cache = cache_factory()
    if cold_cache is original_cache:
        raise PostEvalError("production cold measurement requires an isolated cache")
    counters = {"slow_calls": 0}
    from contextlib import nullcontext
    try:
        from .prediction_worker import count_slow_execution
        model = type("ProductionModel", (), {"bridge": runner.bridge})()
        slow_counter = count_slow_execution(model, counters)
    except (AttributeError, ImportError):
        slow_counter = nullcontext()
    observer.cache, observer.meter = cold_cache, meter
    try:
        with slow_counter:
            payload, receipt = meter.measure(lambda: runner.detect(request), clear_application_caches=lambda: None)
    finally:
        observer.cache, observer.meter = original_cache, original_meter
    receipt["slow_calls"] = counters["slow_calls"]
    return payload, {**receipt, "scope": "production streaming Fast trigger/fusion route; resident models; isolated empty observer cache"}
