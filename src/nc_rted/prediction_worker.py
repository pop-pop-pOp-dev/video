"""Adapter-driven blind prediction execution.

Real adapters are intentionally injected after preflight: importing a training
tokenizer or a test evaluator into this module would violate the blind boundary.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import errno
import math
from numbers import Real
from typing import Any, Protocol

from .prediction_inputs import MediaVerifier, ModelArtifact, PredictionInputError, PredictionPlan, VadRequest, VauRequest
from .prediction_store import PredictionPublicationError, PredictionStore, PredictionStoreError


class PredictionExecutionError(RuntimeError):
    pass


class BoundModel(Protocol):
    group: str
    evidence_enabled: bool


class ModelLoader(Protocol):
    def load(self, artifact: ModelArtifact) -> BoundModel: ...


class VadAdapter(Protocol):
    def predict(self, request: VadRequest, model: BoundModel, *, protocol: dict[str, Any]) -> dict[str, Any]: ...


class VauAdapter(Protocol):
    def generate(self, request: VauRequest, model: BoundModel, *, protocol: dict[str, Any]) -> dict[str, Any]: ...


class PredictionWorker:
    def __init__(self, plan: PredictionPlan, store: PredictionStore, *, loader: ModelLoader, vad: VadAdapter, vau: VauAdapter,
                 model: ModelArtifact, max_attempts: int = 3):
        if max_attempts != 3:
            raise ValueError("prediction retry limit is fixed at three retries")
        if (store.model_task != model.task_id or store.run_id != plan.run_id or store.manifest_sha256 != plan.manifest_sha256 or
                store.model_binding_sha256 != model.manifest_sha256):
            raise PredictionExecutionError("prediction store/model task identity differs")
        self.plan, self.store, self.loader, self.vad, self.vau, self.model, self.max_retries = plan, store, loader, vad, vau, model, max_attempts
        self.media = MediaVerifier()

    def _model(self) -> BoundModel:
        model = self.loader.load(self.model)
        if getattr(model, "group", None) != self.model.group or bool(getattr(model, "evidence_enabled", None)) != self.model.evidence_enabled:
            raise PredictionExecutionError("loaded model does not match its group/evidence identity")
        if getattr(model, "seed", None) != self.model.seed:
            raise PredictionExecutionError("loaded model does not match its seed identity")
        return model

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
            if (not isinstance(frames, list) or not frames or any(type(frame) is not int or frame < 0 or frame >= total for frame in frames) or
                    frames != sorted(set(frames))):
                raise PredictionExecutionError("VAD query frame coverage is invalid")
            for name in ("fast_score", "final_score"):
                self._probability(query.get(name), name=name)
            for name in ("slow_score", "fused_score"):
                if query.get(name) is not None:
                    self._probability(query[name], name=name)
        for value in causal:
            self._probability(value, name="causal_smoothed_score")
        return payload

    def run(self) -> dict[str, int]:
        with self.store.lease():
            return self._run_locked()

    def _run_locked(self) -> dict[str, int]:
        model = self._model()
        completed = failed = skipped = 0
        for request in self.plan.requests():
            if not self.store.should_run(identity=request.identity, max_retries=self.max_retries):
                skipped += 1
                continue
            previous = self.store.get(identity=request.identity)
            attempt = 1 if previous is None else previous["attempt"] + 1
            provenance = {"model_group": self.model.group, "model_seed": self.model.seed, "model_task": self.model.task_id, "model_evidence_enabled": model.evidence_enabled,
                          "manifest_sha256": self.plan.manifest_sha256, "model_manifest_sha256": self.model.manifest_sha256,
                          "checkpoint_manifest_sha256": self.model.checkpoint_manifest_sha256, "checkpoint_state_sha256": self.model.checkpoint_state_sha256,
                          "bindings": dict(self.plan.bindings), "protocol": dict(self.plan.protocol),
                          "stage": "vad" if isinstance(request, VadRequest) else "vau"}
            try:
                self.media.verify(request.media_path, request.media_sha256)
                if isinstance(request, VadRequest):
                    payload = self._validate_vad_payload(self.vad.predict(request, model, protocol=dict(self.plan.protocol)))
                else:
                    payload = self.vau.generate(request, model, protocol=dict(self.plan.protocol))
                    if not isinstance(payload, dict) or "text" not in payload or "token_ids" not in payload:
                        raise PredictionExecutionError("VAU adapter omitted full text/token output")
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
