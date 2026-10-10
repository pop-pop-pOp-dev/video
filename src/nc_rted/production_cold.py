"""Fresh development Fast scoring through the actual production VAD route.

The application cache starts empty; model loading and OS page-cache eviction
are outside this measurement. CPU/GPU phase transfers are inside its timing.
"""
from __future__ import annotations

from contextlib import contextmanager, ExitStack
from dataclasses import replace
from datetime import datetime, timezone
from functools import wraps
import math
from pathlib import Path
import shutil
import time

import torch

from .detection_provider import DetectionPrefix, DetectionQuery, DetectionProtocol
from .media_observer import BoundMedia, CausalMediaObserver, verified_path_lease
from .observation_cache import FrozenFrameCache
from .prediction_inputs import VadRequest
from .prediction_worker import count_slow_execution


ALIASES = {"ucf-crime": "ucf", "xd-violence": "xd"}
MIN_FREE = 20 << 30


class ColdError(ValueError):
    pass


class CostMeter:
    def __init__(self, device):
        self.device = torch.device(device)
        self.seconds, self.calls = {}, {}

    def sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def call(self, name, operation):
        self.sync()
        started = time.perf_counter()
        try:
            result = operation()
            self.calls[name] = self.calls.get(name, 0) + 1
            return result
        finally:
            self.sync()
            self.seconds[name] = self.seconds.get(name, 0.0) + time.perf_counter() - started

    def detector(self, operation):
        return self.call("detector", operation)

    def process(self, operation):
        return self.call("observation_pipeline_including_detector", operation)

    @contextmanager
    def forward(self, module, name):
        missing = object()
        prior = vars(module).get("forward", missing)
        original = module.forward

        @wraps(original)
        def measured(*args, **kwargs):
            return self.call(name, lambda: original(*args, **kwargs))

        module.forward = measured
        try:
            yield
        finally:
            if prior is missing:
                del module.forward
            else:
                module.forward = prior


class LivePrefixReader:
    """One sealed causal prefix, sequentially decoded and freshly Fast-scored."""
    def __init__(self, *, media, record, protocol, decoder_factory, encode,
                 score_grid, make_grid, meter, deadline):
        media.validate()
        protocol.validate()
        self.media, self.record, self.protocol = media, record, protocol
        self.decoder_factory, self.encode = decoder_factory, encode
        self.score_grid, self.make_grid, self.meter, self.deadline = score_grid, make_grid, meter, deadline
        self.used = False
        self.interval = max(1, int(media.fps / 4))
        end_query = record["query_index"]
        if type(end_query) is not int or end_query < 0:
            raise ColdError("invalid sealed query endpoint")
        last_indices = self.indices(end_query)
        if (not last_indices or not math.isclose(last_indices[-1] / media.fps,
                                               record["observed_seconds"], abs_tol=1e-6)):
            raise ColdError("sealed prefix endpoint differs from actual sampling")
        # Do not expand predictions into any unobserved original frames.
        self.observed_frame_count = last_indices[-1] + 1

    def indices(self, index):
        return list(range(index * 4 * self.interval,
                          min((index + 1) * 4 * self.interval, self.media.frame_count), self.interval))

    def guard(self):
        if datetime.now(timezone.utc) >= self.deadline:
            raise ColdError("cold diagnostic deadline reached")
        if shutil.disk_usage(Path(self.media.media_path).parent).free < MIN_FREE:
            raise ColdError("cold diagnostic must preserve 20 GiB free")

    def __call__(self, dataset, media_key, *, expected_media_path, expected_media_sha256):
        media = self.media
        if self.used:
            raise ColdError("cold reader is single-use; create a fresh application cache")
        if (dataset, media_key, expected_media_path, expected_media_sha256) != (
                media.dataset, media.media_key, media.media_path, media.media_sha256):
            raise ColdError("request differs from sealed development medium")
        self.used = True

        def iterate():
            self.guard()
            with verified_path_lease(Path(media.media_path), media.media_sha256) as lease:
                decoder = self.decoder_factory(lease.path)
                try:
                    if (decoder.frame_count != media.frame_count or decoder.height != media.height
                            or decoder.width != media.width or not math.isfinite(decoder.fps)
                            or abs(decoder.fps - media.fps) > 1e-6):
                        raise ColdError("decoded geometry differs from sealed development media")
                    for index in range(self.record["query_index"] + 1):
                        self.guard()
                        indices = self.indices(index)
                        lease.verify_content()
                        frames = self.meter.call("decode", lambda: [decoder.read(i) for i in indices])
                        lease.verify_content()
                        # The released grid constructor pads its input list in-place.
                        grid = self.make_grid(list(frames))
                        values = self.meter.call("fast_including_phase_transfers", lambda: self.score_grid(grid))
                        if len(values) != 1 or not math.isfinite(float(values[0])) or not 0 <= float(values[0]) <= 1:
                            raise ColdError("fresh Fast scorer returned an invalid probability")
                        score = float(values[0])
                        last = self.meter.call("memory_vision", lambda: self.encode([frames[-1]]))
                        dense = None
                        if self.protocol.rt_anomaly and score >= self.protocol.trigger_threshold:
                            dense = self.meter.call("memory_vision", lambda: self.encode(frames + [frames[-1]] * (4-len(frames))))
                        lease.verify_content()
                        yield DetectionQuery(index, tuple(indices), tuple(i/media.fps for i in indices),
                                             score, last[0].detach(), None if dense is None else dense.detach())
                finally:
                    decoder.close()

        return DetectionPrefix(iterate(), media.height, media.width, self.observed_frame_count, self.interval)


def validate_runtime_fast(runtime, paths, config, plan):
    """The fresh scorer must match the production Fast and trigger configuration."""
    fast = runtime.document["fast"]
    for key, path_key in (("model", "model_path"), ("lora", "lora_path"), ("streamforest_weights", "streamforest_weights")):
        if Path(fast[key]).resolve() != Path(paths[path_key]).resolve():
            raise ColdError("production Fast paths differ from frozen training Fast")
    if (fast["image_size"], fast["vision_feature_layer"], fast["attn_implementation"]) != (384, -2, "sdpa"):
        raise ColdError("production Fast settings differ")
    for dataset, alias in ALIASES.items():
        if runtime.document["protocols"]["vad"][alias] != plan["protocols"][dataset]:
            raise ColdError("development and production trigger/memory protocols differ")


def run_cold_prefix(*, model, runtime, media_row, record, score_grid, make_grid,
                    cache_root, deadline, device, decoder_factory):
    """Use the existing production runner with fresh Fast and observation inputs."""
    from .prediction_adapters import BlindDetectionRunner
    runner = model.vad_detector
    if not isinstance(runner, BlindDetectionRunner):
        raise ColdError("measurement requires the actual production BlindDetectionRunner")
    alias = ALIASES[record["dataset"]]
    media = BoundMedia(**{k: media_row[k] for k in BoundMedia.__dataclass_fields__ if k in media_row})
    media = replace(media, dataset=alias)
    protocol = DetectionProtocol(**runtime.document["protocols"]["vad"][alias])
    if cache_root.exists():
        raise ColdError("cold observation cache must be new")
    if shutil.disk_usage(cache_root.parent).free < MIN_FREE + (1 << 20):
        raise ColdError("insufficient cold measurement reserve")
    cache_root.mkdir()
    meter = CostMeter(device)
    encode = model.hivau_inference.media_reader.vision_encoder
    reader = LivePrefixReader(media=media, record=record, protocol=protocol,
                              decoder_factory=decoder_factory, encode=encode,
                              score_grid=score_grid, make_grid=make_grid, meter=meter, deadline=deadline)
    original_reader, original_observer = runner.reader, runner.observation_reader
    if model.evidence_enabled:
        if not isinstance(original_observer, CausalMediaObserver):
            raise ColdError("augmented model lacks a concrete causal observer")
        observer = CausalMediaObserver(detector=original_observer.detector, siglip=original_observer.siglip,
                                      cache=FrozenFrameCache(cache_root, 256 << 20, min_free_bytes=MIN_FREE),
                                      media_catalog={(alias, media.media_key): media}, decoder_factory=decoder_factory,
                                      meter=meter)
    else:
        observer = None
    runner.reader, runner.observation_reader = reader, observer
    counters = {}
    request = VadRequest(alias, media.media_key, media.media_path, media.media_sha256)
    try:
        with ExitStack() as stack:
            stack.enter_context(count_slow_execution(model, counters))
            stack.enter_context(meter.forward(model.bridge.raw_slow, "slow_forward"))
            stack.enter_context(meter.forward(model.bridge.evidence, "process_encoder_and_tokens"))
            stack.enter_context(torch.no_grad())
            reader.guard()
            meter.sync()
            if meter.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(meter.device)
            started = time.perf_counter()
            # Replay captures projector device/dtype in its constructor, before
            # asking the reader for its first Fast query.
            meter.call("initial_language_transfer", model.residency.activate_language)
            payload = runner.detect(request)
            meter.sync()
            elapsed = time.perf_counter() - started
            peak = int(torch.cuda.max_memory_allocated(meter.device)) if meter.device.type == "cuda" else 0
            reserved = int(torch.cuda.max_memory_reserved(meter.device)) if meter.device.type == "cuda" else 0
        expected_calls = sum(q["slow_score"] is not None for q in payload["queries"])
        if counters["completed_slow_forwards"] != expected_calls:
            raise ColdError("actual Slow forward count differs from production outputs")
        return {"schema": "nc_rted_production_cold_prefix/v1", "sample_id": record["sample_id"],
                "model_task": model.group if model.seed is None else f"{model.group}:seed{model.seed}",
                "scope": "sealed development prefix; fresh Fast; empty application observation cache; resident models; OS page cache uncontrolled",
                "residency_policy": "accepted Fast/Slow language CPU-GPU phase offload; all timed transfers included",
                "component_timing": "nested inclusive timings; do not sum components",
                "elapsed_seconds": elapsed, "peak_gpu_allocated_bytes": peak, "peak_gpu_reserved_bytes": reserved,
                "slow_calls": counters["completed_slow_forwards"], "component_seconds": meter.seconds,
                "component_completed_calls": meter.calls, "cache_root": str(cache_root),
                "payload": payload}
    finally:
        runner.reader, runner.observation_reader = original_reader, original_observer
