import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from nc_rted.detection_media import OpenCVFrames, StreamingDetectionReader
from nc_rted.detection_provider import DetectionProtocol
from nc_rted.task_inputs import TaskInputError, sha256_file


def setup(tmp_path, *, total=10, rt=True):
    media = tmp_path / "media.mp4"
    media.write_bytes(b"test media identity")
    row = dict(dataset="ucf", media_key="video", media_path=str(media), media_sha256=sha256_file(media),
               fps=4., frame_count=total, height=8, width=12, target_fps=4, query_interval=4,
               queries=[dict(index=i, frame_indices=list(range(i*4, min((i+1)*4, total))), fast_score=.7)
                        for i in range((total+3)//4)])
    document = dict(schema="nc_rted_frozen_fast/v1", fast_identity={"weights_sha256":"frozen"}, media=[row])
    snapshot = tmp_path / "fast.json"
    snapshot.write_text(json.dumps(document))
    calls, encodings = [], []
    class Decoder:
        fps=4.; frame_count=total; height=8; width=12
        def __init__(self, path): calls.append("open")
        def read(self, index): calls.append(index); return index
        def close(self): calls.append("close")
    def encode(frames):
        encodings.append(tuple(frames))
        return torch.stack([torch.full((729,1152), float(frame)) for frame in frames])
    protocol = DetectionProtocol("question", "neutral", "none", True, rt, .5, .6)
    def build():
        return StreamingDetectionReader(snapshot, snapshot_sha256=sha256_file(snapshot),
            fast_identity=document["fast_identity"], protocols={"ucf":protocol}, encode=encode, decoder_factory=Decoder)
    return build, document, snapshot, calls, encodings


def test_opencv_frames_uses_exact_monotone_grab_retrieve_and_backward_seek_fallback():
    class Capture:
        def __init__(self): self.position, self.calls = 0, []
        def grab(self):
            self.calls.append(("grab", self.position)); self.position += 1; return True
        def retrieve(self):
            self.calls.append(("retrieve", self.position - 1))
            return True, np.full((1, 1, 3), self.position - 1, dtype=np.uint8)
        def set(self, key, index): self.calls.append(("set", key, index)); self.position = index
        def read(self):
            self.calls.append(("read", self.position)); value = self.position; self.position += 1
            return True, np.full((1, 1, 3), value, dtype=np.uint8)

    capture = Capture()
    decoder = object.__new__(OpenCVFrames)
    decoder.capture, decoder._next_index = capture, 0
    decoder.cv2 = type("CV2", (), {"CAP_PROP_POS_FRAMES": 1, "COLOR_BGR2RGB": 2,
                                    "cvtColor": staticmethod(lambda frame, code: frame)})()
    assert [np.asarray(decoder.read(index))[0, 0, 0] for index in (0, 1, 3, 2)] == [0, 1, 3, 2]
    assert capture.calls == [("grab", 0), ("retrieve", 0), ("grab", 1), ("retrieve", 1),
                             ("grab", 2), ("grab", 3), ("retrieve", 3), ("set", 1, 2), ("read", 2)]


def test_decoder_is_lazy_and_never_decodes_after_target(tmp_path):
    build, _, _, calls, encodings = setup(tmp_path)
    prefix = build()("ucf", "video", 1)
    assert calls == []
    queries = list(prefix.queries)
    assert calls == ["open", *range(8), "close"]
    assert encodings == [(3,), (7,), (4,5,6,7)]
    assert queries[-1].frame_times_s == (1.,1.25,1.5,1.75)


def test_tail_padding_uses_only_observed_last_frame(tmp_path):
    build, _, _, calls, encodings = setup(tmp_path)
    queries = list(build()("ucf", "video", 2).queries)
    assert calls == ["open", *range(10), "close"]
    assert encodings[-2:] == [(9,), (8,9,9,9)]
    assert queries[-1].frame_indices == (8,9)
    assert queries[-1].dense_patches.shape == (4,729,1152)


def test_early_close_releases_decoder(tmp_path):
    build, _, _, calls, _ = setup(tmp_path)
    queries = build()("ucf", "video", 2).queries
    next(queries)
    queries.close()
    assert calls == ["open",0,1,2,3,"close"]


def test_changed_media_and_misaligned_scores_fail_before_decode(tmp_path):
    build, document, snapshot, calls, _ = setup(tmp_path)
    document["media"][0]["queries"][0]["frame_indices"] = [0,1,2,4]
    snapshot.write_text(json.dumps(document))
    with pytest.raises(TaskInputError, match="frame groups"):
        build()("ucf", "video", 1)
    assert calls == []
    (tmp_path / "media.mp4").write_bytes(b"changed")
    with pytest.raises(TaskInputError, match="content changed"):
        build()("ucf", "video", 1)


def test_future_score_changes_do_not_change_selected_prefix(tmp_path):
    build, document, snapshot, _, _ = setup(tmp_path)
    first = list(build()("ucf", "video", 0).queries)
    document["media"][0]["queries"][2]["fast_score"] = .1
    snapshot.write_text(json.dumps(document))
    second = list(build()("ucf", "video", 0).queries)
    assert first[0].frame_times_s == second[0].frame_times_s
    assert first[0].fast_score == second[0].fast_score
    assert torch.equal(first[0].last_frame_patches, second[0].last_frame_patches)
    assert torch.equal(first[0].dense_patches, second[0].dense_patches)


def test_media_replaced_after_lazy_prefix_creation_is_rejected(tmp_path):
    build, _, _, calls, _ = setup(tmp_path)
    prefix = build()("ucf", "video", 0)
    replacement = tmp_path / "replacement.mp4"
    replacement.write_bytes(b"same metadata different pixels")
    replacement.replace(tmp_path / "media.mp4")
    with pytest.raises(TaskInputError, match="content changed before decode"):
        list(prefix.queries)
    assert calls == []


def test_decoder_opens_verified_inode_even_if_pathname_is_replaced(tmp_path):
    from pathlib import Path
    build, _, _, calls, _ = setup(tmp_path)
    reader = build()
    previous = reader.decoder_factory
    def factory(leased):
        original_bytes = Path(leased).read_bytes()
        replacement = tmp_path / "replacement.mp4"
        replacement.write_bytes(b"changed content")
        replacement.replace(tmp_path / "media.mp4")
        assert Path(leased).read_bytes() == original_bytes
        return previous(leased)
    reader.decoder_factory = factory
    # Filesystems may change the old inode's ctime on unlink. Either safe
    # refusal or continuing on the verified inode is valid; replacement bytes
    # must never enter the decoder (the factory checks this above).
    try:
        result = list(reader("ucf","video",0).queries)
    except TaskInputError as error:
        assert "modified during prefix" in str(error)
    else:
        assert len(result) == 1
    assert calls[0] == "open" and calls[-1] == "close"


def test_in_place_media_mutation_is_rejected_before_encoding(tmp_path):
    import os
    from pathlib import Path
    build, _, _, calls, encodings = setup(tmp_path)
    reader = build(); previous = reader.decoder_factory
    def factory(leased):
        original = Path(leased).stat()
        (tmp_path / "media.mp4").write_bytes(b"changed in place")
        os.utime(tmp_path / "media.mp4", ns=(original.st_atime_ns, original.st_mtime_ns+1000000000))
        return previous(leased)
    reader.decoder_factory = factory
    with pytest.raises(TaskInputError, match="modified during prefix"):
        list(reader("ucf","video",0).queries)
    assert calls == ["open","close"] and encodings == []
