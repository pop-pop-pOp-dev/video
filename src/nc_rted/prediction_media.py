"""Full-stream, label-free VAD media replay for blind prediction."""
from __future__ import annotations

from pathlib import Path
import fcntl, hashlib, os
from collections import OrderedDict
from dataclasses import dataclass
import math
import torch

from .detection_media import OpenCVFrames
from .detection_provider import DetectionPrefix, DetectionQuery
from .prediction_worker import PredictionExecutionError
from .task_inputs import sha256_file
from .batches import caption_block_endpoints
from .inherited_memory import detection_memory_from_stream
from .media_observer import BoundMedia, lease_verified_media


@dataclass(frozen=True)
class BlindHivauMaterial:
    visual_embeddings: torch.Tensor
    images: list[torch.Tensor]
    image_sizes: list[tuple[int, int]]
    observed_seconds: float
    sampled_frame_times: tuple[float, ...]
    time_message: str
    observations: object
    fast_scores: tuple[float, ...]


@dataclass(frozen=True)
class _CachedHivauMaterial:
    """CPU-only immutable snapshot; every cache return gets fresh tensors."""
    visual_embeddings: torch.Tensor
    images: tuple[torch.Tensor, ...]
    image_sizes: tuple[tuple[int, int], ...]
    observed_seconds: float
    sampled_frame_times: tuple[float, ...]
    time_message: str
    observation_features: torch.Tensor | None
    observation_valid: torch.Tensor | None
    observation_times: torch.Tensor | None
    fast_scores: tuple[float, ...]
    bytes: int


class _BoundedHivauMaterialCache:
    """Per-reader LRU whose entries are detached CPU snapshots, never results."""
    def __init__(self, max_bytes: int, max_entries: int):
        if type(max_bytes) is not int or max_bytes < 1 or type(max_entries) is not int or max_entries < 1:
            raise PredictionExecutionError("HIVAU material cache bounds are invalid")
        self.max_bytes, self.max_entries = max_bytes, max_entries
        self.entries: OrderedDict[tuple, _CachedHivauMaterial] = OrderedDict()
        self.bytes = 0

    @staticmethod
    def _tensor_snapshot(value: torch.Tensor) -> torch.Tensor:
        return value.detach().to("cpu").clone()

    @staticmethod
    def _tensor_bytes(value: torch.Tensor | None) -> int:
        return 0 if value is None else value.numel() * value.element_size()

    def freeze(self, material: BlindHivauMaterial) -> _CachedHivauMaterial | None:
        observations = material.observations
        if observations is not None:
            # Runtime observations are ObservationBatch. Refuse unknown mutable
            # objects rather than retaining an alias with uncertain semantics.
            fields = tuple(getattr(observations, name, None) for name in ("features", "valid", "observed_times"))
            if not all(isinstance(value, torch.Tensor) for value in fields):
                return None
            observation_features, observation_valid, observation_times = fields
        else:
            observation_features = observation_valid = observation_times = None
        visual = self._tensor_snapshot(material.visual_embeddings)
        images = tuple(self._tensor_snapshot(image) for image in material.images)
        frozen = _CachedHivauMaterial(visual, images, tuple(material.image_sizes), material.observed_seconds,
                                      tuple(material.sampled_frame_times), material.time_message,
                                      None if observation_features is None else self._tensor_snapshot(observation_features),
                                      None if observation_valid is None else self._tensor_snapshot(observation_valid),
                                      None if observation_times is None else self._tensor_snapshot(observation_times),
                                      tuple(material.fast_scores), 0)
        size = self._tensor_bytes(frozen.visual_embeddings) + sum(self._tensor_bytes(image) for image in frozen.images)
        size += self._tensor_bytes(frozen.observation_features) + self._tensor_bytes(frozen.observation_valid) + self._tensor_bytes(frozen.observation_times)
        return _CachedHivauMaterial(frozen.visual_embeddings, frozen.images, frozen.image_sizes,
                                    frozen.observed_seconds, frozen.sampled_frame_times, frozen.time_message,
                                    frozen.observation_features, frozen.observation_valid, frozen.observation_times,
                                    frozen.fast_scores, size)

    def get(self, key: tuple, *, device: torch.device) -> BlindHivauMaterial | None:
        frozen = self.entries.pop(key, None)
        if frozen is None:
            return None
        self.entries[key] = frozen
        observations = None
        if frozen.observation_features is not None:
            from .bridge import ObservationBatch
            observations = ObservationBatch(frozen.observation_features.to(device).clone(),
                                            frozen.observation_valid.to(device).clone(),
                                            frozen.observation_times.to(device).clone())
        return BlindHivauMaterial(frozen.visual_embeddings.to(device).clone(),
                                  [image.to(device).clone() for image in frozen.images], list(frozen.image_sizes),
                                  frozen.observed_seconds, frozen.sampled_frame_times, frozen.time_message,
                                  observations, frozen.fast_scores)

    def put(self, key: tuple, material: BlindHivauMaterial) -> None:
        frozen = self.freeze(material)
        if frozen is None or frozen.bytes > self.max_bytes:
            return
        prior = self.entries.pop(key, None)
        if prior is not None:
            self.bytes -= prior.bytes
        self.entries[key] = frozen
        self.bytes += frozen.bytes
        while len(self.entries) > self.max_entries or self.bytes > self.max_bytes:
            _, evicted = self.entries.popitem(last=False)
            self.bytes -= evicted.bytes

    def clear(self) -> None:
        self.entries.clear()
        self.bytes = 0


class FullBlindDetectionReader:
    """Decode each bound Fast query once and retain RT patches for every query."""
    def __init__(self, frozen_reader, *, decoder_factory=OpenCVFrames):
        self.rows, self.protocols, self.encode = frozen_reader.rows, frozen_reader.protocols, frozen_reader.encode
        self.decoder_factory = decoder_factory

    def __call__(self, dataset: str, media_key: str, *, expected_media_path: str | None = None,
                 expected_media_sha256: str | None = None) -> DetectionPrefix:
        row = self.rows.get((dataset, media_key))
        if row is None: raise PredictionExecutionError("blind VAD identity absent from frozen Fast snapshot")
        protocol = self.protocols[dataset]; protocol.validate()
        path = Path(row["media_path"])
        if ((expected_media_path is not None and str(path) != expected_media_path) or
                (expected_media_sha256 is not None and row["media_sha256"] != expected_media_sha256)):
            raise PredictionExecutionError("Fast snapshot media differs from selected blind request")
        if not path.is_file() or sha256_file(path) != row["media_sha256"]:
            raise PredictionExecutionError("blind VAD media differs from frozen Fast binding")
        fps, total = float(row["fps"]), int(row["frame_count"])
        if not math.isfinite(fps) or fps <= 0 or total < 1:
            raise PredictionExecutionError("frozen Fast media geometry is invalid")
        interval = max(1, int(fps / 4)); queries = row["queries"]
        if row.get("target_fps") != 4 or row.get("query_interval") != 4 or len(queries) != ((total + interval - 1) // interval + 3) // 4:
            raise PredictionExecutionError("frozen Fast VAD schedule differs from inherited route")
        def iterate():
            handle = path.open("rb"); decoder = None
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
                before = os.fstat(handle.fileno())
                digest = hashlib.sha256()
                for block in iter(lambda: handle.read(8 << 20), b""): digest.update(block)
                signature = lambda: (os.fstat(handle.fileno()).st_dev, os.fstat(handle.fileno()).st_ino, os.fstat(handle.fileno()).st_size, os.fstat(handle.fileno()).st_mtime_ns)
                expected_signature = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
                if digest.hexdigest() != row["media_sha256"] or signature() != expected_signature:
                    raise PredictionExecutionError("blind VAD media changed before decode")
                decoder = self.decoder_factory(Path(f"/proc/self/fd/{handle.fileno()}"))
                if (not math.isfinite(float(decoder.fps)) or float(decoder.fps) <= 0 or decoder.frame_count != total or
                        abs(float(decoder.fps) - fps) > 1e-6 or decoder.height != row["height"] or decoder.width != row["width"]):
                    raise PredictionExecutionError("Fast snapshot geometry differs from decoded medium")
                for index, item in enumerate(queries):
                    frames_i = list(range(index * 4 * interval, min((index + 1) * 4 * interval, total), interval))
                    if item.get("index") != index or item.get("frame_indices") != frames_i:
                        raise PredictionExecutionError("frozen Fast VAD frame group differs from inherited route")
                    frames = [decoder.read(frame) for frame in frames_i]
                    if signature() != expected_signature: raise PredictionExecutionError("blind VAD media changed during decode")
                    last = self.encode([frames[-1]])
                    score = float(item["fast_score"])
                    dense = self.encode(frames + [frames[-1]] * (4 - len(frames))) if protocol.rt_anomaly and score >= protocol.trigger_threshold else None
                    yield DetectionQuery(index, tuple(frames_i), tuple(frame / fps for frame in frames_i), score, last[0].detach(), None if dense is None else dense.detach())
            finally:
                if decoder is not None: decoder.close()
                handle.close()
        return DetectionPrefix(iterate(), int(row["height"]), int(row["width"]), total, interval)


class FullBlindHivauReader:
    """Identity-only implementation of the released HIVAU media/memory loop.

    ``fast_detector`` is the bound PaliGemma detector and ``vision_encoder`` is
    the final embedded Slow tower adapter. ``observer`` receives only the media
    identity plus sampled timeline and returns frozen relation observations for
    every 8-second block; it cannot receive a question or an answer.
    """
    def __init__(self, *, slow, fast_detector, fast_prompt: str, vision_encoder, observer, target_fps: int = 4, query_interval: int = 4,
                 decoder_factory=OpenCVFrames, memory_factory=None, grid_builder=None, time_message_style: str = "short_online_v2", media_catalog=None,
                 evidence_enabled: bool = True, material_cache_binding: str = "reader-local-v1",
                 material_cache_max_bytes: int = 512 << 20, material_cache_max_entries: int = 128):
        if target_fps < 1 or query_interval != 4:
            raise PredictionExecutionError("HIVAU requires a positive target FPS and four-frame queries")
        self.slow, self.fast_detector, self.fast_prompt, self.vision_encoder, self.observer = slow, fast_detector, fast_prompt, vision_encoder, observer
        self.target_fps, self.query_interval, self.decoder_factory, self.time_message_style = target_fps, query_interval, decoder_factory, time_message_style
        self.memory_factory, self.grid_builder = memory_factory, grid_builder
        self.media_catalog = None if media_catalog is None else self._physical_catalog(media_catalog)
        self.evidence_enabled = evidence_enabled
        if not isinstance(material_cache_binding, str) or not material_cache_binding:
            raise PredictionExecutionError("HIVAU material cache binding is invalid")
        self.material_cache_binding = material_cache_binding
        self.material_cache = _BoundedHivauMaterialCache(material_cache_max_bytes, material_cache_max_entries)

    @staticmethod
    def _physical_catalog(media_catalog):
        """Collapse only catalog aliases with the same bound physical medium."""
        items = media_catalog.values() if hasattr(media_catalog, "values") else media_catalog
        result = {}
        for item in items:
            if not isinstance(item, BoundMedia):
                raise PredictionExecutionError("HIVAU catalog contains an invalid media binding")
            key = (item.media_path, item.media_sha256)
            physical = (item.dataset, item.media_path, item.media_sha256, item.fps,
                        item.frame_count, item.height, item.width)
            prior = result.get(key)
            if prior is not None:
                prior_physical = (prior.dataset, prior.media_path, prior.media_sha256, prior.fps,
                                  prior.frame_count, prior.height, prior.width)
                if prior_physical != physical:
                    raise PredictionExecutionError("HIVAU catalog aliases disagree on bound media geometry")
                result[key] = min((prior, item), key=lambda value: (value.dataset, value.media_key,
                                                                      -1 if value.request_index is None else value.request_index))
            else:
                result[key] = item
        return result

    def read(self, *, media_path: str, media_sha256: str) -> BlindHivauMaterial:
        path = Path(media_path)
        if not path.is_file() or sha256_file(path) != media_sha256:
            raise PredictionExecutionError("HIVAU media differs from its identity binding")
        # Keep the same verified inode open throughout Fast/Slow memory replay.
        # The observation route independently leases and verifies this immutable
        # catalog medium before publishing frozen blocks.
        media = None if self.media_catalog is None else self.media_catalog.get((str(path), media_sha256))
        if self.media_catalog is not None and media is None:
            raise PredictionExecutionError("HIVAU media is absent or ambiguous in the official catalog")
        media = media if media is not None else BoundMedia("hivau", path.name, str(path), media_sha256, 1., 1, 1, 1)
        key = (self.material_cache_binding, str(path), media_sha256, media.fps, media.frame_count, media.height, media.width,
               self.target_fps, self.query_interval, self.evidence_enabled)
        parameter = next((self.slow.get_base_model() if hasattr(self.slow, "get_base_model") else self.slow).get_model().mm_projector.mlp.parameters())
        cached = self.material_cache.get(key, device=parameter.device)
        if cached is not None:
            return cached
        material = None
        with lease_verified_media(media) as leased:
            decoder = self.decoder_factory(leased)
            try:
                fps, total = float(decoder.fps), int(decoder.frame_count)
                if not math.isfinite(fps) or fps <= 0 or total < 1 or int(decoder.height) < 1 or int(decoder.width) < 1: raise PredictionExecutionError("invalid HIVAU media geometry")
                if self.media_catalog is not None and (abs(fps - media.fps) > 1e-6 or total != media.frame_count or int(decoder.height) != media.height or int(decoder.width) != media.width):
                    raise PredictionExecutionError("HIVAU decoder geometry differs from official catalog")
                interval = max(1, int(fps / self.target_fps)); sampled = list(range(0, total, interval))
                groups = [sampled[index:index + self.query_interval] for index in range(0, len(sampled), self.query_interval)]
                frames = [[decoder.read(frame) for frame in group] for group in groups]
                if self.grid_builder is None:
                    from eval_utils.hivau.reactvau_inference import create_grid_image
                    grid_builder = create_grid_image
                else: grid_builder = self.grid_builder
                grids = [grid_builder(list(group), image_size=self.fast_detector.image_size) for group in frames]
                scores = self.fast_detector.batch_score_grids(grids, self.fast_prompt)
                if len(scores) != len(groups): raise PredictionExecutionError("HIVAU Fast output count differs from query groups")
                if self.memory_factory is None:
                    from llava.model.multimodal_projector.memory_manager import MemoryManager
                    memory_factory = MemoryManager
                else: memory_factory = self.memory_factory
                memory = memory_factory(self.slow.get_vision_tower().config.hidden_size, self.slow.get_vision_tower().config.num_attention_heads,
                                   st_memory_windows=[1, 12], st_memory_tokens=[729, 128], event_split_window=4,
                                   long_memory_tokens_per_frame=64, long_memory_tokens_quota=2048, anomaly_pool_max_size=0,
                                   anomaly_pool_tokens=128, anomaly_pool_protect_recent=2)
                for group, score in zip(frames, scores):
                    features = self.vision_encoder([group[-1]])[0]
                    memory.update_with_anomaly_score(features, anomaly_score=float(score))
                visual = detection_memory_from_stream(self.slow, memory)
                times = tuple(frame / fps for frame in sampled); end = total / fps
                observed = None
                if self.evidence_enabled:
                    if self.observer is None:
                        raise PredictionExecutionError("HIVAU evidence route has no observation reader")
                    observed = self.observer.observe_full_media(media_path=str(path), media_sha256=media_sha256,
                                                                sampled_frame_times=times, observed_seconds=end)
                    if observed.features.shape[1] != len(caption_block_endpoints(end)):
                        raise PredictionExecutionError("HIVAU observation blocks omit part of the media duration")
                image = torch.zeros(1, 3, int(decoder.height), int(decoder.width), device=parameter.device, dtype=parameter.dtype)
                message = f"\nThe video contains {len(groups)} frames sampled from the past {end:.1f} seconds ago (0.0s of the entire video) up to the present moment ({end:.1f}s of the entire video). "
                material = BlindHivauMaterial(visual, [image], [(int(decoder.height), int(decoder.width))], end, times, message, observed, tuple(map(float, scores)))
            finally: decoder.close()
        # Both decoder.close() and the verified-media lease's final integrity
        # check succeeded. A late failure must never become a cache hit.
        if material is None:
            raise PredictionExecutionError("HIVAU reader completed without material")
        self.material_cache.put(key, material)
        return material
