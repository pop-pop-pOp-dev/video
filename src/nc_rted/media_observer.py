"""Leased-media causal relation observations for detection and caption samples."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
import fcntl
import hashlib
import math
import os
from pathlib import Path
from typing import Callable, ContextManager, Mapping, Protocol

from PIL import Image

from .batches import caption_block_endpoints, pack_observation_blocks
from .caption_provider import CaptionObservationAudit
from .caption_sampling import OriginalSamplingAudit
from .detector import CausalWindowObservation, observe_causal_window
from .features import FeatureAssemblyResult, FeatureStatus
from .observation_cache import FrozenFrameCache
from .task_inputs import TaskInputError


class MediaObserverError(TaskInputError):
    pass


@dataclass(frozen=True)
class BoundMedia:
    dataset: str
    media_key: str
    media_path: str
    media_sha256: str
    fps: float
    frame_count: int
    height: int
    width: int
    request_index: int | None = None

    def validate(self) -> None:
        if (not self.dataset or not self.media_key or not self.media_path
                or not isinstance(self.media_sha256, str) or len(self.media_sha256) != 64
                or any(char not in "0123456789abcdef" for char in self.media_sha256)
                or not isinstance(self.fps, (int, float)) or isinstance(self.fps, bool)
                or not math.isfinite(self.fps) or self.fps <= 0
                or any(type(value) is not int or value <= 0 for value in (self.frame_count, self.height, self.width))):
            raise MediaObserverError("invalid bound media identity or metadata")

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.fps


class FrameDecoder(Protocol):
    fps: float
    frame_count: int
    height: int
    width: int
    def read(self, index: int) -> Image.Image: ...
    def close(self) -> None: ...


@contextmanager
def lease_verified_media(media: BoundMedia, *, verifier_sink: list | None = None):
    """Yield a verified, shared-locked fd path for the exact hashed inode."""
    path = Path(media.media_path)
    if not path.is_file():
        raise MediaObserverError("bound media is absent before observation")
    handle = path.open("rb")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
        before = os.fstat(handle.fileno())
        signature = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        digest = hashlib.sha256()
        for part in iter(lambda: handle.read(8 << 20), b""):
            digest.update(part)
        if digest.hexdigest() != media.media_sha256:
            raise MediaObserverError("bound media content differs before observation")
        handle.seek(0)
        current = os.fstat(handle.fileno())
        if signature != (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns, current.st_ctime_ns):
            raise MediaObserverError("bound media changed during verification")
        def verify_content():
            status = os.fstat(handle.fileno())
            if signature != (status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns, status.st_ctime_ns):
                raise MediaObserverError("bound media changed before frozen cache publication")
        if verifier_sink is not None:
            verifier_sink.append(verify_content)
        failure = None
        try:
            yield Path(f"/proc/self/fd/{handle.fileno()}")
        except BaseException as error:
            failure = error
            raise
        finally:
            after = os.fstat(handle.fileno())
            if signature != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                if failure is None:
                    raise MediaObserverError("bound media changed while leased")
                # Retain the original decoding/inference failure and traceback.
                # Python 3.10 has no add_note; preserve an inspectable attribute.
                failure.media_integrity_failure = "bound media changed while leased"
                if hasattr(failure, "add_note"):
                    failure.add_note(failure.media_integrity_failure)
    finally:
        handle.close()


# Backward-compatible spelling for early adapters. New runtime code should use
# lease_verified_media so its same-inode property is explicit.
direct_file_lease = lease_verified_media


def _empty() -> CausalWindowObservation:
    return CausalWindowObservation(FeatureAssemblyResult(FeatureStatus.NO_RELATION_PAIRS, ()), (), ())


class CausalMediaObserver:
    """Concrete 2FPS observer; no detector binding means explicit non-readiness.

    The caller gives a metadata registry and a leased-file callback. Frame RGB is
    retained only for the current block and frozen observations are reused by the
    bounded cache across overlapping calls.
    """
    def __init__(self, *, detector=None, siglip=None, cache: FrozenFrameCache | None = None,
                 media_catalog: Mapping[tuple[str, str], BoundMedia],
                 lease_resolver: Callable[[BoundMedia], ContextManager[Path]] = lease_verified_media,
                 decoder_factory: Callable[[Path], FrameDecoder] | None = None,
                 observe: Callable = observe_causal_window):
        self.media = dict(media_catalog)
        if not self.media:
            raise MediaObserverError("media observer needs bound media")
        for identity, item in self.media.items():
            if identity != (item.dataset, item.media_key):
                raise MediaObserverError("media registry key differs from bound identity")
            item.validate()
        self.detector, self.siglip, self.cache = detector, siglip, cache
        self.lease, self.decoder_factory, self.observe = lease_resolver, decoder_factory, observe
        self.ready = detector is not None and siglip is not None and cache is not None and decoder_factory is not None

    def _bound(self, dataset: str, media_key: str) -> BoundMedia:
        try:
            return self.media[(dataset, media_key)]
        except KeyError as error:
            raise MediaObserverError("media identity is absent from the bound observer registry") from error

    @contextmanager
    def _open(self, media: BoundMedia):
        if not self.ready:
            raise MediaObserverError("causal observation is unavailable without accepted RT-DETR/SigLIP bindings")
        verifiers = []
        if self.lease is lease_verified_media:
            with lease_verified_media(media, verifier_sink=verifiers) as path:
                yield path, verifiers[0]
        else:
            with self.lease(media) as path:
                # Preserve the exact baseline checked by the hashing lease;
                # never establish a new baseline after that check.
                with lease_verified_media(replace(media, media_path=str(path)), verifier_sink=verifiers) as verified:
                    yield verified, verifiers[0]

    def _validate_decoder(self, decoder: FrameDecoder, media: BoundMedia) -> None:
        if (not math.isfinite(float(decoder.fps)) or float(decoder.fps) <= 0
                or any(type(value) is not int or value <= 0 for value in (decoder.frame_count, decoder.height, decoder.width))
                or abs(float(decoder.fps) - media.fps) > 1e-6 or decoder.frame_count != media.frame_count
                or decoder.height != media.height or decoder.width != media.width):
            raise MediaObserverError("leased media decoder metadata differs from bound identity")

    @staticmethod
    def _indices(media: BoundMedia, start_s: float, end_s: float) -> list[int]:
        if not (math.isfinite(start_s) and math.isfinite(end_s) and 0 <= start_s < end_s <= media.duration_s + 1e-9):
            raise MediaObserverError("invalid causal media interval")
        first_tick, last_tick = math.floor(start_s * 2) + 1, math.floor(end_s * 2)
        indices = []
        for tick in range(first_tick, last_tick + 1):
            index = min(int(math.floor((tick / 2) * media.fps)), media.frame_count - 1)
            timestamp = index / media.fps
            if start_s < timestamp <= end_s and (not indices or indices[-1] != index):
                indices.append(index)
        if len(indices) > 16:
            raise MediaObserverError("global 2FPS grid exceeded an eight-second block cap")
        return indices

    def _block(self, decoder: FrameDecoder, media: BoundMedia, start_s: float, end_s: float,
               verify_content: Callable[[], None]) -> CausalWindowObservation:
        indices = self._indices(media, start_s, end_s)
        if not indices:
            return _empty()
        verify_content()
        images = [decoder.read(index) for index in indices]
        # Observe/cache only after all RGB reads are verified against the inode
        # signature pinned before decode. A later change cannot alter these RGBs.
        verify_content()
        timestamps = [index / media.fps for index in indices]
        # The block list and RGB images are local and released after this call.
        return self.observe(images, timestamps, end_s, self.detector, self.siglip, self.cache,
                            media.media_sha256, self.siglip.identity(), window_start_s=start_s)

    def detection(self, dataset: str, media_key: str, query_s: float) -> CausalWindowObservation:
        media = self._bound(dataset, media_key)
        if not math.isfinite(query_s) or query_s <= 0 or query_s > media.duration_s + 1e-9:
            raise MediaObserverError("detection query lies outside bound media duration")
        start_s = max(0.0, query_s - 8.0)
        with self._open(media) as (path, verify_content):
            decoder = self.decoder_factory(path)
            try:
                self._validate_decoder(decoder, media)
                return self._block(decoder, media, start_s, query_s, verify_content)
            finally:
                decoder.close()

    def __call__(self, dataset: str, media_key: str, query_s: float) -> CausalWindowObservation:
        return self.detection(dataset, media_key, query_s)

    def observe_causal_window(self, *, sample_id: str, annotation: dict,
                              sampling: OriginalSamplingAudit) -> CaptionObservationAudit:
        if not isinstance(annotation, dict) or not isinstance(sample_id, str) or not sample_id.startswith("caption:"):
            raise MediaObserverError("caption observer requires a fixed caption media identity")
        # This concrete path observes already materialized full-media clips.
        # An upstream start/end override changes the task's allowed scope and
        # needs an explicitly range-aware provider; never silently read outside it.
        if "start" in annotation or "end" in annotation or annotation.get("video_read_type", "decord") != "decord":
            raise MediaObserverError("caption range/reader override requires a matching scoped media binding")
        parts = sample_id.split(":", 2)
        if len(parts) != 3 or annotation.get("id") != sampling.annotation_id or annotation.get("_reactvau_relative_video") != sampling.relative_video:
            raise MediaObserverError("caption observer identity differs from original sampling audit")
        media = self._bound(parts[1], sampling.relative_video)
        if annotation.get("video") != media.media_path and annotation.get("video") != sampling.video:
            raise MediaObserverError("caption annotation media path differs from bound identity")
        # Full duration is metadata-derived, never inferred from the final original
        # caption sample. It defines all left-aligned 8-second blocks and tail.
        endpoints = caption_block_endpoints(media.duration_s)
        blocks = []
        with self._open(media) as (path, verify_content):
            decoder = self.decoder_factory(path)
            try:
                self._validate_decoder(decoder, media)
                start_s = 0.0
                for end_s in endpoints:
                    blocks.append(self._block(decoder, media, start_s, end_s, verify_content))
                    start_s = end_s
            finally:
                decoder.close()
        observation = pack_observation_blocks([block.features for block in blocks], task="caption")
        return CaptionObservationAudit(observation, sampling.sampled_frame_times, media.duration_s,
                                       sampling.time_message, self.detector.identity())
