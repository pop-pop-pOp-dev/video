from contextlib import contextmanager
from pathlib import Path
import hashlib
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nc_rted.caption_sampling import OriginalSamplingAudit
from nc_rted.features import FeatureAssemblyResult, FeatureStatus
from nc_rted.media_observer import (BoundMedia, CausalMediaObserver, MediaObserverError,
                                    lease_verified_media, verified_path_lease)
from nc_rted.detector import CausalWindowObservation


class _Detector:
    def identity(self):
        return {"detector": "accepted"}


class _SigLip:
    def identity(self):
        return {"siglip": "accepted"}


class _Decoder:
    def __init__(self, path, reads, frame_count):
        self.fps, self.frame_count, self.height, self.width = 4., frame_count, 8, 12
        self.reads = reads
        self.closed = False

    def read(self, index):
        self.reads.append(index)
        return object()

    def close(self):
        self.closed = True


def _observer(tmp_path, *, duration_frames=37):
    media_path = tmp_path / "media.mp4"
    media_path.write_bytes(b"bound")
    digest = hashlib.sha256(media_path.read_bytes()).hexdigest()
    media = BoundMedia("ucf-crime", "frozen/clip.mp4", str(media_path), digest, 4., duration_frames, 8, 12, 11)
    reads, calls = [], []
    @contextmanager
    def lease(bound):
        assert bound.request_index == 11
        yield media_path
    def factory(path):
        return _Decoder(path, reads, duration_frames)
    def observe(images, timestamps, query, detector, siglip, cache, media_hash, siglip_identity, *, window_start_s):
        calls.append((tuple(timestamps), query, window_start_s))
        return CausalWindowObservation(FeatureAssemblyResult(FeatureStatus.NO_RELATION_PAIRS, ()), (), ())
    instance = CausalMediaObserver(detector=_Detector(), siglip=_SigLip(), cache=object(),
                                   media_catalog={("ucf-crime", "frozen/clip.mp4"): media}, lease_resolver=lease,
                                   decoder_factory=factory, observe=observe)
    return instance, reads, calls


def test_caption_observation_implementation_identity_hashes_real_producers():
    identity = CausalMediaObserver.caption_observation_implementation_identity()
    assert set(identity) == {"batches", "detection_media", "detector", "features", "media_observer",
                             "observation", "observation_cache", "tracking"}
    assert all(isinstance(digest, str) and len(digest) == 64 for digest in identity.values())


def test_detection_reads_only_global_2fps_frames_at_or_before_query(tmp_path):
    observer, reads, calls = _observer(tmp_path)
    observer("ucf-crime", "frozen/clip.mp4", 3.)
    assert reads == [2, 4, 6, 8, 10, 12]
    assert calls == [((.5, 1., 1.5, 2., 2.5, 3.), 3., 0.)]
    assert max(reads) / 4 <= 3.


def test_caption_covers_tail_and_represents_subhalfsecond_tail_as_empty_mask(tmp_path):
    observer, reads, calls = _observer(tmp_path, duration_frames=33)  # 8.25 seconds
    sampling = OriginalSamplingAudit("id", "original.mp4", "frozen/clip.mp4", "leased.mp4", (0, 4), 4., (0., 1.), "original", (.1, .2))
    audit = observer.observe_causal_window(sample_id="caption:ucf-crime:id", annotation={"id": "id", "video": "original.mp4", "_reactvau_relative_video": "frozen/clip.mp4"}, sampling=sampling)
    assert audit.original_observed_seconds == 8.25
    assert audit.observations.features.shape[1] == 2
    assert calls == [((.5, 1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5., 5.5, 6., 6.5, 7., 7.5, 8.), 8., 0.)]
    assert reads[-1] == 32
    assert audit.observations.valid.any().item() is False


def test_unbound_detector_refuses_before_media_lease(tmp_path):
    path = tmp_path / "media.mp4"
    path.write_bytes(b"bound")
    media = BoundMedia("ucf-crime", "frozen/clip.mp4", str(path), hashlib.sha256(b"bound").hexdigest(), 4., 4, 8, 8)
    opened = []
    @contextmanager
    def lease(bound):
        opened.append(True)
        yield path
    observer = CausalMediaObserver(media_catalog={("ucf-crime", "frozen/clip.mp4"): media}, lease_resolver=lease)
    assert observer.ready is False
    with pytest.raises(MediaObserverError, match="RT-DETR"):
        observer("ucf-crime", "frozen/clip.mp4", .5)
    assert not opened


def test_verified_lease_yields_same_open_inode_path(tmp_path):
    path = tmp_path / "media.mp4"
    path.write_bytes(b"bound")
    media = BoundMedia("ucf-crime", "frozen/clip.mp4", str(path), hashlib.sha256(b"bound").hexdigest(), 4., 4, 8, 8)
    with lease_verified_media(media) as leased:
        assert str(leased).startswith("/proc/self/fd/")
        assert leased.read_bytes() == b"bound"


def test_custom_cache_lease_cannot_swap_bytes_after_verification(tmp_path):
    observer, _, _ = _observer(tmp_path)
    media = next(iter(observer.media.values()))
    original_factory = observer.decoder_factory
    def replace_name(path):
        replacement = tmp_path / "replacement.mp4"
        replacement.write_bytes(b"different future file")
        replacement.replace(media.media_path)
        assert path.read_bytes() == b"bound"
        return original_factory(path)
    observer.decoder_factory = replace_name
    try:
        observer.detection(media.dataset, media.media_key, 3.)
    except MediaObserverError as error:
        # Unlinking may change ctime on some filesystems. Conservative rejection
        # is allowed; the decoder above still must see the original bytes.
        assert "changed" in str(error)


def test_verified_custom_lease_accepts_derived_segment_with_its_own_geometry(tmp_path):
    raw = tmp_path / "raw.mp4"; raw.write_bytes(b"raw-source")
    segment = tmp_path / "segment.mp4"; segment.write_bytes(b"derived-segment")
    media = BoundMedia("ucf-crime", "events/clip.mp4", str(raw), hashlib.sha256(raw.read_bytes()).hexdigest(), 4., 8, 8, 12, 11)
    reads, calls = [], []
    @contextmanager
    def lease(_):
        with verified_path_lease(segment, hashlib.sha256(segment.read_bytes()).hexdigest(),
                                 source_sha256=hashlib.sha256(raw.read_bytes()).hexdigest()) as verified:
            yield verified
    observer = CausalMediaObserver(detector=_Detector(), siglip=_SigLip(), cache=object(),
                                   media_catalog={("ucf-crime", media.media_key): media}, lease_resolver=lease,
                                   decoder_factory=lambda path: _Decoder(path, reads, media.frame_count),
                                   observe=lambda *args, **kwargs: calls.append(args) or CausalWindowObservation(FeatureAssemblyResult(FeatureStatus.NO_RELATION_PAIRS, ()), (), ()))
    assert observer.detection(media.dataset, media.media_key, 1.).features.status == FeatureStatus.NO_RELATION_PAIRS
    assert reads and calls


def test_same_source_derived_segments_have_distinct_frame_cache_identities(tmp_path):
    import torch
    from PIL import Image
    from nc_rted.detector import observe_causal_window
    from nc_rted.observation_cache import FrozenFrameCache

    raw = tmp_path / "raw.mp4"; raw.write_bytes(b"raw-source")
    first = tmp_path / "first-segment.mp4"; first.write_bytes(b"first-segment")
    second = tmp_path / "second-segment.mp4"; second.write_bytes(b"second-segment")
    raw_sha256 = hashlib.sha256(raw.read_bytes()).hexdigest()
    media = {
        ("ucf-crime", "events/train/source_E0.mp4"): BoundMedia("ucf-crime", "events/train/source_E0.mp4", str(raw), raw_sha256, 4., 4, 8, 12, 1),
        ("ucf-crime", "events/train/source_E1.mp4"): BoundMedia("ucf-crime", "events/train/source_E1.mp4", str(raw), raw_sha256, 4., 4, 8, 12, 2),
    }
    segments = {"events/train/source_E0.mp4": first, "events/train/source_E1.mp4": second}
    class Detector(_Detector):
        calls = 0
        def detect(self, image):
            self.calls += 1
            return ()
    class SigLip(_SigLip):
        calls = 0
        def __call__(self, images):
            self.calls += 1
            return torch.ones(len(images), 729, 1152)
    @contextmanager
    def lease(bound):
        segment = segments[bound.media_key]
        with verified_path_lease(segment, hashlib.sha256(segment.read_bytes()).hexdigest(),
                                 source_sha256=raw_sha256) as verified:
            yield verified
    class Decoder(_Decoder):
        def read(self, index):
            self.reads.append(index)
            return Image.new("RGB", (12, 8), color=(index, 0, 0))
    detector, siglip, reads = Detector(), SigLip(), []
    observer = CausalMediaObserver(detector=detector, siglip=siglip,
                                   cache=FrozenFrameCache(tmp_path / "frame-cache", 64 << 20, min_free_bytes=0),
                                   media_catalog=media, lease_resolver=lease,
                                   decoder_factory=lambda path: Decoder(path, reads, 4), observe=observe_causal_window)
    observer.detection("ucf-crime", "events/train/source_E0.mp4", .5)
    observer.detection("ucf-crime", "events/train/source_E1.mp4", .5)
    assert detector.calls == siglip.calls == 2


def test_inplace_mutation_is_reported_when_verified_lease_releases(tmp_path):
    observer, _, _ = _observer(tmp_path)
    media = next(iter(observer.media.values()))
    with pytest.raises(MediaObserverError, match="changed while leased"):
        with lease_verified_media(media):
            Path(media.media_path).write_bytes(b"modified")


def test_future_extension_preserves_real_features_and_overlaps_reuse_cache(tmp_path):
    import numpy as np
    import torch
    from PIL import Image
    from nc_rted.detector import Detection, observe_causal_window
    from nc_rted.observation_cache import FrozenFrameCache

    class Detector(_Detector):
        calls = 0
        def detect(self, image):
            self.calls += 1
            return (Detection((.1, .1, .4, .8), 0, .9),
                    Detection((.5, .2, .9, .8), 2, .8))
    class SigLip(_SigLip):
        frames = 0
        def __call__(self, images):
            self.frames += len(images)
            # Per-frame RGB differences must survive to the feature pipeline.
            values = torch.tensor([image.getpixel((0, 0))[0] + 1 for image in images], dtype=torch.float32)
            return values[:, None, None].expand(-1, 729, 1152).contiguous()
    class Decoder(_Decoder):
        def read(self, index):
            self.reads.append(index)
            return Image.new("RGB", (12, 8), color=(index, 0, 0))

    observer, reads, _ = _observer(tmp_path, duration_frames=16)
    detector, siglip = Detector(), SigLip()
    observer.detector, observer.siglip = detector, siglip
    observer.cache = FrozenFrameCache(tmp_path / "frame-cache", max_bytes=64 << 20, min_free_bytes=0)
    observer.observe = observe_causal_window
    observer.decoder_factory = lambda path: Decoder(path, reads, 16)
    first = observer.detection("ucf-crime", "frozen/clip.mp4", 3.)
    observer.detection("ucf-crime", "frozen/clip.mp4", 3.5)
    assert detector.calls == siglip.frames == 7  # Six shared ticks, one new tick.
    assert max(reads) == 14
    assert first.features.relations

    # A different complete media identity with appended future frames must have
    # exactly the same observed feature values for the original query.
    extended_path = tmp_path / "extended.mp4"
    extended_path.write_bytes(b"bound plus future")
    extended = BoundMedia("ucf-crime", "extended.mp4", str(extended_path),
                          hashlib.sha256(extended_path.read_bytes()).hexdigest(), 4., 80, 8, 12)
    extended_reads = []
    extended_observer = CausalMediaObserver(
        detector=detector, siglip=siglip, cache=observer.cache,
        media_catalog={(extended.dataset, extended.media_key): extended},
        decoder_factory=lambda path: Decoder(path, extended_reads, 80))
    later = extended_observer.detection(extended.dataset, extended.media_key, 3.)
    assert extended_reads == [2, 4, 6, 8, 10, 12]
    assert later.relation_ids == first.relation_ids
    for left, right in zip(first.features.relations, later.features.relations):
        np.testing.assert_array_equal(left.student_cells, right.student_cells)
        np.testing.assert_array_equal(left.cell_mask, right.cell_mask)
        np.testing.assert_array_equal(left.observed_times_s, right.observed_times_s)


def test_modified_rgb_reads_never_reach_observation_cache(tmp_path):
    observer, reads, calls = _observer(tmp_path)
    media = next(iter(observer.media.values()))
    class MutatingDecoder(_Decoder):
        def read(self, index):
            Path(media.media_path).write_bytes(b"changed while decoding")
            return super().read(index)
    observer.decoder_factory = lambda path: MutatingDecoder(path, reads, media.frame_count)
    with pytest.raises(MediaObserverError, match="cache publication"):
        observer.detection(media.dataset, media.media_key, 3.)
    assert not calls


def test_caption_range_override_is_rejected_before_decoding(tmp_path):
    observer, reads, calls = _observer(tmp_path)
    sampling = OriginalSamplingAudit("id", "original.mp4", "frozen/clip.mp4", "leased.mp4", (0, 4), 4., (0., 1.), "original", (.1, .2))
    with pytest.raises(MediaObserverError, match="range/reader override"):
        observer.observe_causal_window(sample_id="caption:ucf-crime:id", annotation={"id":"id", "video":"original.mp4", "_reactvau_relative_video":"frozen/clip.mp4", "start":0, "end":1}, sampling=sampling)
    assert not reads and not calls


def test_empty_detections_pack_as_explicit_bypass_not_technical_failure():
    import torch
    from PIL import Image
    from nc_rted.detector import observe_causal_window
    from nc_rted.batches import pack_observation_blocks
    class Detector(_Detector):
        def detect(self, image): return ()
    class Cache:
        def get(self, key): return None
        def put(self, key, value): pass
    class SigLip(_SigLip):
        def __call__(self, images): return torch.zeros(len(images), 729, 1152)
    block = observe_causal_window([Image.new("RGB", (12, 8))], [.5], .5,
                                  Detector(), SigLip(), Cache(), "a"*64, {}, window_start_s=0.)
    assert block.features.status == FeatureStatus.NO_RELATION_PAIRS
    batch = pack_observation_blocks([block.features], task="detection")
    assert not batch.valid.any() and batch.features.shape[2] == 0


@pytest.mark.parametrize("fps", [float("nan"), float("inf"), 0.])
def test_invalid_decoded_clock_is_rejected_before_observation(tmp_path, fps):
    observer, reads, calls = _observer(tmp_path)
    original_factory = observer.decoder_factory
    def factory(path):
        decoder = original_factory(path); decoder.fps = fps; return decoder
    observer.decoder_factory = factory
    with pytest.raises(MediaObserverError, match="decoder metadata"):
        observer.detection("ucf-crime", "frozen/clip.mp4", 3.)
    assert not reads and not calls


def test_lease_checks_integrity_even_when_decoder_raises(tmp_path):
    observer, _, _ = _observer(tmp_path)
    media = next(iter(observer.media.values()))
    error = RuntimeError("original decode error")
    with pytest.raises(RuntimeError) as captured:
        with lease_verified_media(media):
            Path(media.media_path).write_bytes(b"changed")
            raise error
    assert captured.value is error
    assert error.media_integrity_failure == "bound media changed while leased"


def test_mutation_restore_retry_does_not_reuse_poisoned_frame_cache(tmp_path):
    import torch
    from PIL import Image
    from nc_rted.detector import observe_causal_window
    from nc_rted.observation_cache import FrozenFrameCache
    observer, reads, _ = _observer(tmp_path)
    media = next(iter(observer.media.values()))
    observer.observe = observe_causal_window
    observer.cache = FrozenFrameCache(tmp_path / "cache", 1 << 20, min_free_bytes=0)
    class Detector(_Detector):
        def detect(self, image): return ()
    class SigLip(_SigLip):
        calls = 0
        def __call__(self, images):
            self.calls += 1
            return torch.ones(len(images), 729, 1152)
    siglip = SigLip(); observer.siglip = siglip; observer.detector = Detector()
    class Decoder(_Decoder):
        mutate = True
        def read(self, index):
            if self.mutate: Path(media.media_path).write_bytes(b"corrupted frames")
            return Image.new("RGB", (12, 8))
    observer.decoder_factory = lambda path: Decoder(path, reads, media.frame_count)
    with pytest.raises(MediaObserverError): observer.detection(media.dataset, media.media_key, .5)
    assert siglip.calls == 0 and not list(observer.cache.root.glob("*.pt"))
    Path(media.media_path).write_bytes(b"bound")
    Decoder.mutate = False
    assert observer.detection(media.dataset, media.media_key, .5).features.status == FeatureStatus.NO_RELATION_PAIRS
    assert siglip.calls == 1


def test_mutation_between_hash_and_decoder_uses_original_signature(tmp_path):
    observer, reads, calls = _observer(tmp_path)
    media = next(iter(observer.media.values()))
    original_open = observer._open
    @contextmanager
    def mutate_after_hash(item):
        with original_open(item) as verified:
            Path(item.media_path).write_bytes(b"changed after hash baseline")
            yield verified
    observer._open = mutate_after_hash
    with pytest.raises(MediaObserverError, match="cache publication"):
        observer.detection(media.dataset, media.media_key, 3.)
    assert not reads and not calls
    Path(media.media_path).write_bytes(b"bound")
    observer._open = original_open
    observer.detection(media.dataset, media.media_key, 3.)
    assert len(calls) == 1 and reads == [2,4,6,8,10,12]
