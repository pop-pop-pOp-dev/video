"""Adapter-driven blind prediction execution.

Real adapters are intentionally injected after preflight: importing a training
tokenizer or a test evaluator into this module would violate the blind boundary.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import errno
from functools import wraps
import math
from numbers import Real
from typing import Any, Mapping, Protocol

from .prediction_inputs import (MediaVerifier, ModelArtifact, PredictionInputError, PredictionPlan, VadRequest, VauRequest,
                                prediction_task_execution_binding_sha256)
from .prediction_store import PredictionPublicationError, PredictionStore, PredictionStoreError


class PredictionExecutionError(RuntimeError):
    pass


_DIAGNOSTIC_METADATA_FIELDS = ("small_object", "no_candidate", "association_failure", "reference_insufficient")


def _validate_diagnostic_metadata(value: object) -> None:
    if value is None:
        return
    if not isinstance(value, dict) or set(value) != set(_DIAGNOSTIC_METADATA_FIELDS):
        raise PredictionExecutionError("VAD diagnostic metadata has an invalid schema")
    if any(item is not None and type(item) is not bool for item in value.values()):
        raise PredictionExecutionError("VAD diagnostic metadata must contain booleans or null")
    if value["reference_insufficient"] is not None:
        raise PredictionExecutionError("reference insufficiency is unsupported at prediction deployment")


class BoundModel(Protocol):
    group: str
    evidence_enabled: bool


class ModelLoader(Protocol):
    def load(self, artifact: ModelArtifact) -> BoundModel: ...


class VadAdapter(Protocol):
    def predict(self, request: VadRequest, model: BoundModel, *, protocol: dict[str, Any]) -> dict[str, Any]: ...


class VauAdapter(Protocol):
    def generate(self, request: VauRequest, model: BoundModel, *, protocol: dict[str, Any]) -> dict[str, Any]: ...


def validate_loaded_model_identity(model: BoundModel, artifact: ModelArtifact) -> BoundModel:
    """The identity contract used by both formal and diagnostic execution."""
    if (getattr(model, "group", None) != artifact.group or
            bool(getattr(model, "evidence_enabled", None)) != artifact.evidence_enabled or
            getattr(model, "seed", None) != artifact.seed):
        raise PredictionExecutionError("loaded model does not match its declared task identity")
    return model


def _bound_vad_geometry(expected_media: object) -> tuple[int, int, list[Mapping[str, object]]]:
    if not isinstance(expected_media, Mapping):
        raise PredictionExecutionError("bound Fast media geometry is invalid")
    total, fps, target_fps, query_interval, queries = (expected_media.get("frame_count"), expected_media.get("fps"),
                                                        expected_media.get("target_fps"), expected_media.get("query_interval"),
                                                        expected_media.get("queries"))
    if (type(total) is not int or total < 1 or isinstance(fps, bool) or not isinstance(fps, Real) or
            not math.isfinite(float(fps)) or float(fps) <= 0 or type(target_fps) is not int or target_fps < 1 or
            type(query_interval) is not int or query_interval < 1 or not isinstance(queries, list) or not queries):
        raise PredictionExecutionError("bound Fast media geometry is invalid")
    interval = max(1, int(float(fps) / target_fps))
    sampled = list(range(0, total, interval))
    expected_groups = [sampled[offset:offset + query_interval] for offset in range(0, len(sampled), query_interval)]
    if len(queries) != len(expected_groups):
        raise PredictionExecutionError("bound Fast query count differs from the sampling timeline")
    for index, (query, expected_frames) in enumerate(zip(queries, expected_groups)):
        frames = query.get("frame_indices") if isinstance(query, Mapping) else None
        score = query.get("fast_score") if isinstance(query, Mapping) else None
        if (not isinstance(query, Mapping) or query.get("index") != index or not isinstance(frames, list) or not frames or
                any(type(frame) is not int or frame < 0 or frame >= total for frame in frames) or
                frames != expected_frames):
            raise PredictionExecutionError("bound Fast query geometry is invalid")
        PredictionWorker._probability(score, name="bound_fast_score")
    return total, interval, queries


def validate_vad_payload(payload: object, *, expected_media: object = None,
                         trigger_threshold: object = None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise PredictionExecutionError("VAD adapter omitted structured output")
    queries, causal, total = payload.get("queries"), payload.get("causal_smoothed_scores"), payload.get("total_frames")
    if not isinstance(queries, list) or not queries or type(total) is not int or total < 1:
        raise PredictionExecutionError("VAD adapter omitted complete query/original-frame geometry")
    if not isinstance(causal, list) or len(causal) != total:
        raise PredictionExecutionError("VAD causal scores do not cover every original frame")
    for index, query in enumerate(queries):
        if not isinstance(query, dict) or query.get("query_index") != index:
            raise PredictionExecutionError("VAD query indices are incomplete or unordered")
        frames = query.get("frame_indices")
        if (not isinstance(frames, list) or not frames or any(type(frame) is not int or frame < 0 or frame >= total for frame in frames) or frames != sorted(set(frames))):
            raise PredictionExecutionError("VAD query frame coverage is invalid")
        for name in ("fast_score", "final_score"):
            PredictionWorker._probability(query.get(name), name=name)
        for name in ("slow_score", "fused_score"):
            if query.get(name) is not None:
                PredictionWorker._probability(query[name], name=name)
        _validate_diagnostic_metadata(query.get("diagnostic_metadata"))
    for value in causal:
        PredictionWorker._probability(value, name="causal_smoothed_score")
    if expected_media is not None:
        total, interval, bound_queries = _bound_vad_geometry(expected_media)
        PredictionWorker._probability(trigger_threshold, name="trigger_threshold")
        if payload.get("total_frames") != total or payload.get("sample_interval") != interval:
            raise PredictionExecutionError("VAD payload geometry differs from bound Fast media")
        if len(queries) != len(bound_queries):
            raise PredictionExecutionError("VAD payload omits bound Fast queries")
        for query, bound in zip(queries, bound_queries):
            fast = bound["fast_score"]
            expected_trigger = float(fast) >= float(trigger_threshold)
            if query["frame_indices"] != bound["frame_indices"] or query["fast_score"] != fast:
                raise PredictionExecutionError("VAD payload query differs from bound Fast media")
            if type(query.get("triggered")) is not bool or query["triggered"] != expected_trigger:
                raise PredictionExecutionError("VAD payload trigger differs from bound Fast score")
            if (query.get("slow_score") is not None) != expected_trigger:
                raise PredictionExecutionError("VAD payload Slow score differs from bound trigger")
    return payload


@contextmanager
def count_slow_execution(loaded_model: BoundModel, counters: dict[str, int], *, on_progress=None):
    """Count returned raw Slow forwards, including PEFT's direct-forward route.

    Generation calls Qwen2ForCausalLM.generate on bridge.raw_slow. Hooking only
    the PEFT wrapper misses those calls; wrapping the raw forward covers both
    routes without changing its signature or outputs.
    """
    try:
        import torch
        bridge = loaded_model.bridge
        slow = bridge.raw_slow if hasattr(bridge, "raw_slow") else bridge.slow
    except (AttributeError, ImportError) as error:
        raise PredictionExecutionError("loaded model has no actual Slow module") from error
    if not isinstance(slow, torch.nn.Module):
        raise PredictionExecutionError("loaded model Slow path is not a torch module")
    counters.setdefault("completed_slow_forwards", 0)
    if type(counters["completed_slow_forwards"]) is not int or counters["completed_slow_forwards"] < 0:
        raise ValueError("completed_slow_forwards counter is invalid")
    original = slow.forward
    missing = object()
    prior = vars(slow).get("forward", missing)

    @wraps(original)
    def observed(*args, **kwargs):
        result = original(*args, **kwargs)
        counters["completed_slow_forwards"] += 1
        if on_progress is not None:
            on_progress()
        return result

    slow.forward = observed
    try:
        yield slow
    finally:
        if prior is missing:
            del slow.forward
        else:
            slow.forward = prior


@contextmanager
def count_generation_execution(loaded_model: BoundModel, counters: dict[str, int]):
    """Count completed public generation calls independently of Slow forwards."""
    try:
        runtime = loaded_model.hivau_inference  # type: ignore[attr-defined]
        original = runtime.generate
    except AttributeError as error:
        raise PredictionExecutionError("loaded model has no actual generation runtime") from error
    if not callable(original):
        raise PredictionExecutionError("loaded model generation API is invalid")
    if not isinstance(counters, dict):
        raise TypeError("execution counters must be a dictionary")
    counters.setdefault("completed_generation_calls", 0)
    if type(counters["completed_generation_calls"]) is not int or counters["completed_generation_calls"] < 0:
        raise ValueError("completed_generation_calls counter is invalid")

    def counted_generate(*args, **kwargs):
        result = original(*args, **kwargs)
        counters["completed_generation_calls"] += 1
        return result

    runtime.generate = counted_generate
    try:
        yield runtime
    finally:
        runtime.generate = original


def validate_vau_payload(payload: object, *, max_new_tokens: int) -> dict[str, Any]:
    if (not isinstance(payload, dict) or not isinstance(payload.get("text"), str) or
            not isinstance(payload.get("token_ids"), list) or any(type(token) is not int for token in payload["token_ids"]) or
            len(payload["token_ids"]) > max_new_tokens):
        raise PredictionExecutionError("VAU adapter omitted complete bounded generation output")
    return payload


class PredictionWorker:
    def __init__(self, plan: PredictionPlan, store: PredictionStore, *, loader: ModelLoader, vad: VadAdapter, vau: VauAdapter,
                 model: ModelArtifact, max_attempts: int = 3):
        if max_attempts != 3:
            raise ValueError("prediction retry limit is fixed at three retries")
        if (store.model_task != model.task_id or store.run_id != plan.run_id or store.manifest_sha256 != plan.manifest_sha256 or
                store.model_binding_sha256 != model.manifest_sha256):
            raise PredictionExecutionError("prediction store/model task identity differs")
        if plan.matrix_id is not None:
            binding = prediction_task_execution_binding_sha256(matrix_id=plan.matrix_id, task=model)
            if store.matrix_id != plan.matrix_id or store.task_execution_binding_sha256 != binding:
                raise PredictionExecutionError("prediction store cannot reuse a different v2 task execution")
        self.plan, self.store, self.loader, self.vad, self.vau, self.model, self.max_retries = plan, store, loader, vad, vau, model, max_attempts
        self.media = MediaVerifier()

    def _model(self) -> BoundModel:
        return validate_loaded_model_identity(self.loader.load(self.model), self.model)

    @staticmethod
    def _retryable(error: BaseException) -> bool:
        """Only transient local I/O receives an automatic retry lease."""
        if isinstance(error, (PredictionInputError, PredictionExecutionError, PredictionStoreError, ValueError)):
            return False
        # Do not retry permission, read-only filesystem, capacity, or unknown
        # I/O errors. Those need an explicit operator repair/protective block.
        transient = {errno.EAGAIN, errno.EINTR, errno.ETIMEDOUT, errno.ECONNABORTED,
                     errno.ECONNRESET, errno.ENETDOWN, errno.ENETUNREACH, errno.EHOSTUNREACH}
        return isinstance(error, OSError) and error.errno in transient

    @staticmethod
    def _retry_deadline(attempt: int) -> datetime:
        delays = (5, 20, 60)
        if not 1 <= attempt <= len(delays):
            raise PredictionExecutionError("retry attempt exceeds fixed backoff schedule")
        return datetime.now(timezone.utc) + timedelta(minutes=delays[attempt - 1])

    @staticmethod
    def _probability(value: object, *, name: str) -> None:
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
            raise PredictionExecutionError(f"VAD {name} must be a finite probability")

    def _validate_vad_payload(self, payload: object) -> dict[str, Any]:
        return validate_vad_payload(payload)

    def run(self) -> dict[str, int]:
        with self.store.lease():
            return self._run_locked()

    def _run_locked(self) -> dict[str, int]:
        model = self._model()
        completed = failed = skipped = 0
        for request in self.plan.execution_requests():
            if not self.store.should_run(identity=request.identity, max_retries=self.max_retries):
                skipped += 1
                continue
            previous = self.store.get(identity=request.identity)
            attempt = 1 if previous is None else previous["attempt"] + 1
            provenance = {"model_group": self.model.group, "model_seed": self.model.seed, "model_task": self.model.task_id, "model_evidence_enabled": model.evidence_enabled,
                          "manifest_sha256": self.plan.manifest_sha256, "model_manifest_sha256": self.model.manifest_sha256,
                          "checkpoint_manifest_sha256": self.model.checkpoint_manifest_sha256, "checkpoint_state_sha256": self.model.checkpoint_state_sha256,
                          "bindings": dict(self.plan.bindings), "protocol": dict(self.plan.protocol), "matrix_id": self.plan.matrix_id,
                          "task_execution_binding_sha256": self.store.task_execution_binding_sha256,
                          "stage": "vad" if isinstance(request, VadRequest) else "vau"}
            try:
                self.media.verify(request.media_path, request.media_sha256)
                if isinstance(request, VadRequest):
                    payload = self._validate_vad_payload(self.vad.predict(request, model, protocol=dict(self.plan.protocol)))
                else:
                    payload = validate_vau_payload(self.vau.generate(request, model, protocol=dict(self.plan.protocol)),
                                                   max_new_tokens=self.plan.protocol["hivau"]["max_new_tokens"])
            except Exception as error:
                retryable = self._retryable(error) and attempt <= self.max_retries
                self.store.publish(identity=request.identity, attempt=attempt, status="failure", provenance=provenance,
                                   error=error, retryable=retryable,
                                   retry_not_before=self._retry_deadline(attempt) if retryable else None)
                failed += 1
            else:
                try:
                    self.store.publish(identity=request.identity, attempt=attempt, status="success", provenance=provenance, payload=payload)
                except PredictionPublicationError as error:
                    # The result file is durable. Do not replace it with a failure
                    # record merely because publishing the resume index failed.
                    raise PredictionExecutionError("prediction result committed but index publication failed") from error
                completed += 1
        succeeded = technical_failures = missing = 0
        for request in self.plan.requests():
            record = self.store.get(identity=request.identity)
            if record is None:
                missing += 1
            elif record["status"] == "success":
                succeeded += 1
            else:
                technical_failures += 1
        return {"completed": completed, "failed": failed, "skipped": skipped, "expected": len(self.plan.requests()),
                "succeeded": succeeded, "technical_failures": technical_failures, "missing": missing}
