"""Full-stream, label-free VAD media replay for blind prediction."""
from __future__ import annotations

from pathlib import Path
import fcntl, hashlib, os
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
                 evidence_enabled: bool = True):
        if target_fps < 1 or query_interval != 4:
            raise PredictionExecutionError("HIVAU requires a positive target FPS and four-frame queries")
        self.slow, self.fast_detector, self.fast_prompt, self.vision_encoder, self.observer = slow, fast_detector, fast_prompt, vision_encoder, observer
        self.target_fps, self.query_interval, self.decoder_factory, self.time_message_style = target_fps, query_interval, decoder_factory, time_message_style
        self.memory_factory, self.grid_builder = memory_factory, grid_builder
        self.media_catalog = None if media_catalog is None else tuple(media_catalog.values() if hasattr(media_catalog, "values") else media_catalog)
        self.evidence_enabled = evidence_enabled

    def read(self, *, media_path: str, media_sha256: str) -> BlindHivauMaterial:
        path = Path(media_path)
        if not path.is_file() or sha256_file(path) != media_sha256:
            raise PredictionExecutionError("HIVAU media differs from its identity binding")
        # Keep the same verified inode open throughout Fast/Slow memory replay.
        # The observation route independently leases and verifies this immutable
        # catalog medium before publishing frozen blocks.
        matches = [] if self.media_catalog is None else [item for item in self.media_catalog if item.media_path == str(path) and item.media_sha256 == media_sha256]
        if self.media_catalog is not None and len(matches) != 1:
            raise PredictionExecutionError("HIVAU media is absent or ambiguous in the official catalog")
        media = matches[0] if matches else BoundMedia("hivau", path.name, str(path), media_sha256, 1., 1, 1, 1)
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
                parameter = next((self.slow.get_base_model() if hasattr(self.slow, "get_base_model") else self.slow).get_model().mm_projector.mlp.parameters())
                image = torch.zeros(1, 3, int(decoder.height), int(decoder.width), device=parameter.device, dtype=parameter.dtype)
                message = f"\nThe video contains {len(groups)} frames sampled from the past {end:.1f} seconds ago (0.0s of the entire video) up to the present moment ({end:.1f}s of the entire video). "
                return BlindHivauMaterial(visual, [image], [(int(decoder.height), int(decoder.width))], end, times, message, observed, tuple(map(float, scores)))
            finally: decoder.close()
