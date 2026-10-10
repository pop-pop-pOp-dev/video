import hashlib
import multiprocessing
import os
import stat

import pytest
import torch

from nc_rted.bridge import ObservationBatch
from nc_rted.caption_observation_cache import (CaptionObservationCache, CaptionObservationCacheError,
                                                ReusedCaptionObserver)
from nc_rted.caption_provider import CaptionObservationAudit
from nc_rted.caption_sampling import OriginalSamplingAudit
from nc_rted.media_observer import BoundMedia


class _Identity:
    def __init__(self, name): self.name = name
    def identity(self): return {"name": self.name, "numerical_policy_identity": "fixed"}


class _Observer:
    ready = True
    def __init__(self):
        self.calls = 0
        self.detector, self.siglip = _Identity("detector"), _Identity("siglip")
        self.media = {("ucf-crime", "captions/clip.mp4"): BoundMedia(
            "ucf-crime", "captions/clip.mp4", "/logical/derived-clip.mp4", "a" * 64, 4., 36, 8, 12, 17)}
    def _bound(self, dataset, media_key): return self.media[(dataset, media_key)]
    def caption_observation_implementation_identity(self): return {"producer": "fixed-v1"}
    def detection(self, dataset, media_key, query_s): return (dataset, media_key, query_s)
    def __call__(self, dataset, media_key, query_s): return self.detection(dataset, media_key, query_s)
    def observe_causal_window(self, *, sample_id, annotation, sampling):
        self.calls += 1
        features = torch.full((1, 2, 1, 4, 7), float("nan"), dtype=torch.bfloat16)
        valid = torch.zeros((1, 2, 1, 4), dtype=torch.bool); valid[0, 0, 0, 0] = True
        times = torch.full((1, 2, 1, 4), float("nan"), dtype=torch.float32); times[0, 0, 0, 0] = .5
        features[0, 0, 0, 0] = torch.arange(7, dtype=torch.bfloat16)
        return CaptionObservationAudit(ObservationBatch(features, valid, times), sampling.sampled_frame_times,
                                       9., sampling.time_message, self.detector.identity())


def _sampling():
    return OriginalSamplingAudit("caption-1", "original.mp4", "captions/clip.mp4", "leased.mp4", (0, 8, 20),
                                 4., (0., 2., 5.), "original sampler", (.1, .2, .3))


def _wrapped(root, observer=None, source="b"):
    observer = _Observer() if observer is None else observer
    cache = CaptionObservationCache(root, 1 << 20, min_free_bytes=20 << 30)
    wrapped = ReusedCaptionObserver(observer, cache, resolver_provenance=lambda media: {
        "mode": "bounded", "source_sha256": source * 64, "segment": [1., 10.], "request_index": media.request_index},
        feature_dtype="bfloat16")
    return wrapped, cache, observer


def _observe(wrapped):
    return wrapped.observe_causal_window(sample_id="caption:ucf-crime:caption-1",
        annotation={"id": "caption-1", "video": "original.mp4", "_reactvau_relative_video": "captions/clip.mp4"}, sampling=_sampling())


def _race_audit():
    values = ObservationBatch(torch.zeros((1, 1, 0, 4, 7), dtype=torch.bfloat16),
                              torch.zeros((1, 1, 0, 4), dtype=torch.bool),
                              torch.empty((1, 1, 0, 4), dtype=torch.float32))
    return CaptionObservationAudit(values, (0.,), 1., "race", {"identity": "race"})


def _race_reader(root, provenance, ready, start, result):
    try:
        cache = CaptionObservationCache(root, 1 << 20, min_free_bytes=20 << 30)
        ready.set(); start.wait(10); cache.get(provenance, feature_dtype=torch.bfloat16); result.put("reader")
    except BaseException as error:
        result.put(type(error).__name__)


def _race_publisher(root, provenance, ready, start, result):
    try:
        cache = CaptionObservationCache(root, 1 << 20, min_free_bytes=20 << 30)
        ready.set(); start.wait(10); cache.put(provenance, _race_audit(), feature_dtype=torch.bfloat16); result.put("publisher")
    except BaseException as error:
        result.put(type(error).__name__)


def test_caption_observation_cache_cold_warm_equality_and_restart_resume(tmp_path):
    wrapped, _, observer = _wrapped(tmp_path)
    cold = _observe(wrapped); warm = _observe(wrapped)
    assert observer.calls == 1
    torch.testing.assert_close(cold.observations.features, warm.observations.features, equal_nan=True)
    torch.testing.assert_close(cold.observations.observed_times, warm.observations.observed_times, equal_nan=True)
    assert torch.equal(cold.observations.valid, warm.observations.valid)
    resumed, _, fresh = _wrapped(tmp_path)
    assert _observe(resumed).original_observed_seconds == 9.
    assert fresh.calls == 0
    assert resumed("ucf-crime", "captions/clip.mp4", 2.) == ("ucf-crime", "captions/clip.mp4", 2.)
    indexes = list((tmp_path / "media-index").glob("*.json"))
    assert len(indexes) == 1
    assert b"derived-clip.mp4" in indexes[0].read_bytes()
    assert not list(tmp_path.rglob("*.mp4"))


def test_caption_observation_cache_invalidates_when_bound_source_or_sampling_changes(tmp_path):
    first, _, observer = _wrapped(tmp_path, source="b")
    _observe(first)
    changed_source, _, same_observer = _wrapped(tmp_path, observer=observer, source="c")
    _observe(changed_source)
    assert same_observer.calls == 2
    changed_sampling = OriginalSamplingAudit("caption-1", "original.mp4", "captions/clip.mp4", "leased.mp4", (0, 12),
                                             4., (0., 3.), "original sampler", (.1, .2))
    changed_source.observe_causal_window(sample_id="caption:ucf-crime:caption-1",
        annotation={"id": "caption-1", "video": "original.mp4", "_reactvau_relative_video": "captions/clip.mp4"}, sampling=changed_sampling)
    assert same_observer.calls == 3


def test_caption_observation_cache_invalidates_when_the_frozen_producer_changes(tmp_path):
    first, _, observer = _wrapped(tmp_path)
    _observe(first)
    observer.caption_observation_implementation_identity = lambda: {"producer": "fixed-v2"}
    changed, _, _ = _wrapped(tmp_path, observer=observer)
    _observe(changed)
    assert observer.calls == 2


def test_caption_observation_cache_discards_corruption_and_never_accepts_trainable_values(tmp_path):
    wrapped, cache, observer = _wrapped(tmp_path)
    _observe(wrapped)
    provenance, _ = wrapped._provenance("caption:ucf-crime:caption-1",
        {"id": "caption-1", "video": "original.mp4", "_reactvau_relative_video": "captions/clip.mp4"}, _sampling())
    cache._path(cache.key(provenance)).write_bytes(b"corrupt")
    assert _observe(wrapped).original_observed_seconds == 9.
    assert observer.calls == 2
    trainable = CaptionObservationAudit(ObservationBatch(torch.ones((1, 1, 0, 4, 7), requires_grad=True),
        torch.zeros((1, 1, 0, 4), dtype=torch.bool), torch.empty((1, 1, 0, 4))), (0.,), 1., "t", {"x": 1})
    with pytest.raises(CaptionObservationCacheError, match="frozen observation"):
        cache.put({"different": True}, trainable)


def test_caption_observation_cache_corrupt_reader_and_publisher_do_not_deadlock(tmp_path):
    provenance = {"race": "same-entry"}
    cache = CaptionObservationCache(tmp_path, 1 << 20, min_free_bytes=20 << 30)
    cache._path(cache.key(provenance)).write_bytes(b"corrupt")
    context = multiprocessing.get_context("fork")
    reader_ready, writer_ready, start, result = context.Event(), context.Event(), context.Event(), context.Queue()
    reader = context.Process(target=_race_reader, args=(str(tmp_path), provenance, reader_ready, start, result))
    writer = context.Process(target=_race_publisher, args=(str(tmp_path), provenance, writer_ready, start, result))
    reader.start(); writer.start()
    assert reader_ready.wait(10) and writer_ready.wait(10)
    start.set(); reader.join(10); writer.join(10)
    assert reader.exitcode == 0 and writer.exitcode == 0
    assert {result.get(timeout=2), result.get(timeout=2)} == {"reader", "publisher"}
    assert cache.get(provenance, feature_dtype=torch.bfloat16) is not None


def test_media_index_publication_fsyncs_its_directory_and_propagates_failure(tmp_path, monkeypatch):
    import nc_rted.caption_observation_cache as module
    cache = CaptionObservationCache(tmp_path, 1 << 20, min_free_bytes=20 << 30)
    original_fsync = os.fsync
    first_provenance, first_media = {"media": "first"}, {"path": "first"}
    key = cache.key({"media": first_provenance})
    destination = tmp_path / "media-index" / f"{key}.json"
    failures = []
    def failing_fsync(descriptor):
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            failures.append(descriptor)
            raise OSError("directory fsync failed")
        return original_fsync(descriptor)
    monkeypatch.setattr(module.os, "fsync", failing_fsync)
    with pytest.raises(OSError, match="directory fsync failed"):
        cache.record_media(first_provenance, first_media)
    assert failures and destination.is_file()
    synchronized = []
    def recording_fsync(descriptor):
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            synchronized.append(descriptor)
        return original_fsync(descriptor)
    monkeypatch.setattr(module.os, "fsync", recording_fsync)
    cache.record_media(first_provenance, first_media)
    assert synchronized
