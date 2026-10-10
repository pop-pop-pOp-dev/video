"""Persistent, hash-bound reuse of frozen caption relation observations."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import pickle
import tempfile
from typing import Callable

import torch

from .bridge import ObservationBatch
from .caption_provider import CaptionObservationAudit
from .caption_sampling import OriginalSamplingAudit
from .storage_lock import allocation_lock, ensure_directory, open_lock_file


CACHE_SCHEMA = "nc_rted_caption_observation_cache/v1"
LAYOUT_SCHEMA = "nc_rted_caption_observation_layout/v1"


class CaptionObservationCacheError(RuntimeError):
    pass


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise CaptionObservationCacheError("caption observation provenance is not canonical JSON") from error


def _digest(value: object) -> str:
    digest = hashlib.sha256()

    def field(data: bytes) -> None:
        digest.update(str(len(data)).encode("ascii") + b":" + data)

    def visit(item: object) -> None:
        if isinstance(item, torch.Tensor):
            if item.device.type != "cpu" or item.requires_grad or not item.is_contiguous():
                raise CaptionObservationCacheError("cache tensors must be detached contiguous CPU tensors")
            digest.update(b"T")
            field(_canonical([str(item.dtype), list(item.shape)]))
            field(item.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            digest.update(b"D")
            for key in sorted(item):
                if not isinstance(key, str):
                    raise CaptionObservationCacheError("cache mapping keys must be strings")
                field(key.encode("utf-8")); visit(item[key])
        elif isinstance(item, list):
            digest.update(b"L")
            for part in item: visit(part)
        else:
            digest.update(b"J"); field(_canonical(item))
    visit(value)
    return digest.hexdigest()


def _observation_payload(audit: CaptionObservationAudit, *, feature_dtype: torch.dtype | None = None) -> dict:
    observation = audit.observations
    values = {"features": observation.features, "valid": observation.valid, "observed_times": observation.observed_times}
    if (not isinstance(observation, ObservationBatch) or any(not isinstance(value, torch.Tensor) for value in values.values())
            or any(value.requires_grad for value in values.values())):
        raise CaptionObservationCacheError("only frozen observation tensors may enter the persistent cache")
    if (values["features"].ndim != 5 or values["valid"].ndim != 4 or values["observed_times"].ndim != 4
            or values["features"].dtype not in {torch.bfloat16, torch.float32} or values["valid"].dtype != torch.bool
            or values["observed_times"].dtype != torch.float32 or values["features"].shape != values["valid"].shape + (values["features"].shape[-1],)
            or values["observed_times"].shape != values["valid"].shape
            or values["features"].shape[0] != 1 or values["features"].shape[3] != 4
            or feature_dtype is not None and values["features"].dtype != feature_dtype):
        raise CaptionObservationCacheError("caption observation tensor layout is invalid")
    copied = {key: value.detach().cpu().contiguous().clone() for key, value in values.items()}
    return {"observations": copied, "sampled_frame_times": list(audit.original_sampled_frame_times),
            "observed_seconds": audit.original_observed_seconds, "time_message": audit.original_time_message,
            "detector_identity": audit.detector_identity}


def _audit(payload: dict, *, feature_dtype: torch.dtype | None = None) -> CaptionObservationAudit:
    try:
        values = payload["observations"]
        observation = ObservationBatch(values["features"], values["valid"], values["observed_times"])
        audit = CaptionObservationAudit(observation, tuple(payload["sampled_frame_times"]), payload["observed_seconds"],
                                        payload["time_message"], payload["detector_identity"])
        _observation_payload(audit, feature_dtype=feature_dtype)
        return audit
    except (KeyError, TypeError, CaptionObservationCacheError) as error:
        raise CaptionObservationCacheError("cached caption observation payload is invalid") from error


class CaptionObservationCache:
    """Atomic, non-evicting compact store capped below the approved media budget."""
    def __init__(self, root: str | Path, max_bytes: int, *, min_free_bytes: int = 20 << 30):
        if type(max_bytes) is not int or max_bytes <= 0 or type(min_free_bytes) is not int or min_free_bytes < 20 << 30:
            raise CaptionObservationCacheError("caption observation cache resource limits are invalid")
        self.root, self.max_bytes, self.min_free_bytes = Path(root), max_bytes, min_free_bytes
        ensure_directory(self.root / "entries", self.min_free_bytes)
        ensure_directory(self.root / "media-index", self.min_free_bytes)

    @staticmethod
    def key(provenance: dict) -> str:
        return hashlib.sha256(_canonical({"schema": CACHE_SCHEMA, "provenance": provenance})).hexdigest()

    def _path(self, key: str) -> Path:
        if not isinstance(key, str) or len(key) != 64 or set(key) - set("0123456789abcdef"):
            raise CaptionObservationCacheError("caption observation cache key is invalid")
        return self.root / "entries" / f"{key}.pt"

    def _lock(self):
        return open_lock_file(self.root / ".caption-observations.lock", self.min_free_bytes)

    @contextmanager
    def _locked(self):
        # Every transition that can allocate follows this order. The lock-file
        # opener may briefly create metadata under allocation admission before
        # flock acquisition, but it never holds that admission while waiting.
        with self._lock() as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            yield

    @contextmanager
    def _locked_allocation(self):
        with self._locked():
            with allocation_lock(self.root):
                yield

    def _invalidate_locked(self, path: Path, key: str, reason: str) -> None:
        path.unlink(missing_ok=True)
        with allocation_lock(self.root):
            with (self.root / "invalidations.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"key": key, "reason": reason}, sort_keys=True) + "\n")
                handle.flush(); os.fsync(handle.fileno())

    def get(self, provenance: dict, *, feature_dtype: torch.dtype | None = None) -> CaptionObservationAudit | None:
        key, path = self.key(provenance), self._path(self.key(provenance))
        with self._locked():
            if not path.is_file(): return None
            try:
                record = torch.load(path, map_location="cpu", weights_only=True)
                if (not isinstance(record, dict) or record.get("schema") != CACHE_SCHEMA or record.get("key") != key
                        or record.get("provenance") != provenance or record.get("sha256") != _digest(record["payload"])):
                    raise CaptionObservationCacheError("cached caption observation identity or checksum differs")
                return _audit(record["payload"], feature_dtype=feature_dtype)
            except (OSError, KeyError, ValueError, RuntimeError, EOFError, pickle.UnpicklingError, CaptionObservationCacheError):
                self._invalidate_locked(path, key, "invalid_caption_observation")
                return None

    def _reserve(self, additional_bytes: int = 0) -> None:
        import shutil
        if shutil.disk_usage(self.root).free < self.min_free_bytes + additional_bytes:
            raise CaptionObservationCacheError("caption observation cache must preserve the 20 GiB reserve")

    def _total_bytes(self) -> int:
        return sum(path.stat().st_size for path in (self.root / "entries").glob("*.pt"))

    def put(self, provenance: dict, audit: CaptionObservationAudit, *, feature_dtype: torch.dtype | None = None) -> None:
        key, path = self.key(provenance), self._path(self.key(provenance))
        payload = _observation_payload(audit, feature_dtype=feature_dtype)
        record = {"schema": CACHE_SCHEMA, "key": key, "provenance": provenance, "payload": payload,
                  "sha256": _digest(payload)}
        serialized = io.BytesIO(); torch.save(record, serialized); content = serialized.getvalue()
        with self._locked_allocation():
            if path.is_file():
                try:
                    existing = torch.load(path, map_location="cpu", weights_only=True)
                    if (not isinstance(existing, dict) or existing.get("schema") != CACHE_SCHEMA or existing.get("key") != key
                            or existing.get("provenance") != provenance or existing.get("sha256") != _digest(existing["payload"])):
                        raise CaptionObservationCacheError("concurrent cached observation is invalid")
                    _audit(existing["payload"], feature_dtype=feature_dtype)
                    return
                except (OSError, KeyError, ValueError, RuntimeError, EOFError, pickle.UnpicklingError, CaptionObservationCacheError):
                    self._invalidate_locked(path, key, "invalid_caption_observation")
            block = max(4096, os.statvfs(self.root).f_frsize)
            allocation = ((len(content) + block - 1) // block) * block + 2 * block
            if self._total_bytes() + allocation > self.max_bytes:
                raise CaptionObservationCacheError("caption observation cache upper bound would be exceeded")
            self._reserve(allocation)
            descriptor, temporary_name = tempfile.mkstemp(prefix=".caption-", suffix=".tmp", dir=path.parent)
            temporary = Path(temporary_name)
            try:
                with os.fdopen(descriptor, "wb") as output:
                    output.write(content); output.flush(); os.fsync(output.fileno())
                self._reserve(); os.replace(temporary, path)
                directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
                try: os.fsync(directory)
                finally: os.close(directory)
            finally:
                temporary.unlink(missing_ok=True)

    def record_media(self, provenance: dict, media: dict) -> None:
        """Keep a compact immutable logical-media row, never the media bytes."""
        key = self.key({"media": provenance})
        path = self.root / "media-index" / f"{key}.json"
        record = {"schema": CACHE_SCHEMA, "provenance": provenance, "media": media}
        content = _canonical(record)
        with self._locked_allocation():
            if path.exists():
                if path.read_bytes() != content:
                    raise CaptionObservationCacheError("logical caption media metadata changed")
                return
            self._reserve(2 * max(4096, os.statvfs(self.root).f_frsize))
            descriptor, temporary_name = tempfile.mkstemp(prefix=".media-", suffix=".tmp", dir=path.parent)
            temporary = Path(temporary_name)
            try:
                with os.fdopen(descriptor, "wb") as output:
                    output.write(content); output.flush(); os.fsync(output.fileno())
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)


class ReusedCaptionObserver:
    """Cache wrapper that keeps the original observer as the only cold producer."""
    def __init__(self, observer, cache: CaptionObservationCache, *, resolver_provenance: Callable,
                 feature_dtype: str):
        if feature_dtype != "bfloat16":
            raise CaptionObservationCacheError("caption observation cache requires the fixed BF16 feature layout")
        self.observer, self.cache, self.resolver_provenance = observer, cache, resolver_provenance
        self.feature_dtype = feature_dtype
        self.ready = observer.ready

    def detection(self, dataset: str, media_key: str, query_s: float):
        return self.observer.detection(dataset, media_key, query_s)

    def __call__(self, dataset: str, media_key: str, query_s: float):
        return self.observer(dataset, media_key, query_s)

    @staticmethod
    def _sampling(sampling: OriginalSamplingAudit) -> dict:
        return {"annotation_id": sampling.annotation_id, "video": sampling.video, "relative_video": sampling.relative_video,
                "process_video_argument": sampling.process_video_argument, "frame_indices": list(sampling.frame_indices),
                "fps": sampling.fps, "sampled_frame_times": list(sampling.sampled_frame_times),
                "time_message": sampling.time_message, "aligned_pg_scores": list(sampling.aligned_pg_scores)}

    def _provenance(self, sample_id: str, annotation: dict, sampling: OriginalSamplingAudit) -> tuple[dict, dict]:
        parts = sample_id.split(":", 2)
        if len(parts) != 3 or annotation.get("_reactvau_relative_video") != sampling.relative_video:
            raise CaptionObservationCacheError("caption observation request identity is invalid")
        media = self.observer._bound(parts[1], sampling.relative_video)
        media_record = {key: getattr(media, key) for key in media.__dataclass_fields__}
        implementation = getattr(self.observer, "caption_observation_implementation_identity", None)
        if not callable(implementation):
            raise CaptionObservationCacheError("caption observer lacks a frozen implementation identity")
        provenance = {"schema": LAYOUT_SCHEMA, "sample_id": sample_id,
                      "annotation": {key: annotation.get(key) for key in ("id", "video", "_reactvau_relative_video", "start", "end", "fps", "video_read_type") if key in annotation},
                      "sampling": self._sampling(sampling), "media": media_record,
                      "resolver": self.resolver_provenance(media), "detector": self.observer.detector.identity(),
                      "siglip": self.observer.siglip.identity(), "implementation": implementation(),
                      "layout": {"features_dtype": self.feature_dtype, "valid_dtype": "bool", "times_dtype": "float32",
                                 "shape": "[1,caption_blocks,candidates,4,cell_feature_dim]", "padding": "NaN"}}
        return provenance, media_record

    def observe_causal_window(self, *, sample_id: str, annotation: dict, sampling: OriginalSamplingAudit) -> CaptionObservationAudit:
        provenance, media_record = self._provenance(sample_id, annotation, sampling)
        cached = self.cache.get(provenance, feature_dtype=torch.bfloat16)
        if cached is not None:
            if (cached.original_sampled_frame_times != sampling.sampled_frame_times or cached.original_time_message != sampling.time_message
                    or cached.detector_identity != self.observer.detector.identity()):
                raise CaptionObservationCacheError("cached caption observation audit differs from the frozen request")
            return cached
        audit = self.observer.observe_causal_window(sample_id=sample_id, annotation=annotation, sampling=sampling)
        if (audit.original_sampled_frame_times != sampling.sampled_frame_times or audit.original_time_message != sampling.time_message):
            raise CaptionObservationCacheError("cold caption observation changed original sampling")
        self.cache.record_media(provenance["resolver"], media_record)
        self.cache.put(provenance, audit, feature_dtype=torch.bfloat16)
        return audit
