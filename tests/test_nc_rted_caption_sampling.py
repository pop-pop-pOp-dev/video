from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nc_rted.caption_sampling import CaptionSamplingError, OriginalSamplingAuditReader


class _Dataset:
    def __init__(self):
        self.list_data_dict = [{"id": "cap-1", "video": "clip.mp4", "_reactvau_relative_video": "frozen/clip.mp4"}]
        self.calls = []

    def __getitem__(self, index):
        raise AssertionError("reader must not use retrying __getitem__")

    def process_video(self, video_file, annotation, data_args):
        self.calls.append((video_file, annotation, data_args))
        return [object(), object(), object()], "original message", [0, 4, 9], 4.0

    def _get_item(self, index):
        annotation = self.list_data_dict[index]
        self.process_video("resolver-leased-path.mp4", annotation, {"original": "args"})
        return {"id": annotation["id"], "pg_scores": [.1, .2, .3]}


def test_reader_captures_original_process_video_result_and_restores_method():
    dataset = _Dataset()
    original = dataset.process_video
    result = OriginalSamplingAuditReader(dataset).read(0)
    assert dataset.process_video == original
    assert len(dataset.calls) == 1
    assert result.audit.process_video_argument == "resolver-leased-path.mp4"
    assert result.audit.frame_indices == (0, 4, 9)
    assert result.audit.sampled_frame_times == (0., 1., 2.25)
    assert result.audit.time_message == "original message"
    assert result.audit.aligned_pg_scores == (.1, .2, .3)


def test_reader_rejects_pg_alignment_that_does_not_match_original_frames():
    dataset = _Dataset()
    def invalid_item(index):
        annotation = dataset.list_data_dict[index]
        dataset.process_video("resolver-leased-path.mp4", annotation, {"original": "args"})
        return {"id": "cap-1", "pg_scores": [.1]}
    dataset._get_item = invalid_item
    with pytest.raises(CaptionSamplingError, match="PG alignment"):
        OriginalSamplingAuditReader(dataset).read(0)


def test_original_integer_id_and_keyword_process_video_call():
    dataset = _Dataset(); dataset.list_data_dict[0]["id"] = 38
    def item(index):
        annotation = dataset.list_data_dict[index]
        dataset.process_video("bound.mp4", data_anno=annotation, data_args={})
        return {"id": 38, "pg_scores": [.1, .2, .3]}
    dataset._get_item = item
    result = OriginalSamplingAuditReader(dataset).read(0)
    assert result.audit.annotation_id == 38
