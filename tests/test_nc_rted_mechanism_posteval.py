import hashlib
import sys
from types import SimpleNamespace

import numpy as np

import nc_rted.mechanism_posteval as posteval
from nc_rted.mechanism_posteval import ColdRunMeter, REQUIRED_TASKS, _bound_media_fps, _summarize_frozen_rows, stratify_completed_matrix


def test_posteval_strata_preserves_explicit_metadata_only():
    row = {"model_task": "R0", "identity": "vad:dev", "diagnostic_metadata":
           {"fast_trigger": True, "short_event": None, "long_video": False, "small_object": None,
            "no_candidate": False, "association_failure": False, "reference_insufficient": True}}
    report = _summarize_frozen_rows(records=[row])
    assert report["strata"]["fast_trigger"]["true"] == [{"model_task": "R0", "identity": "vad:dev"}]
    assert report["unsupported"]["short_event"] == 1


def test_cold_meter_records_actual_operation_without_cached_label():
    meter = ColdRunMeter()
    cleared = []
    result, receipt = meter.measure(lambda: meter.slow(lambda: "done"), clear_application_caches=lambda: cleared.append(True))
    assert result == "done" and cleared == [True] and receipt["slow_calls"] == 1 and receipt["elapsed_seconds"] >= 0


def test_bound_media_fps_uses_sha_verified_request_media_and_caches(tmp_path, monkeypatch):
    media = tmp_path / "bound.mp4"
    media.write_bytes(b"bound-media-fixture")
    request = SimpleNamespace(media_path=str(media), media_sha256=hashlib.sha256(media.read_bytes()).hexdigest())
    calls = []

    class Capture:
        def __init__(self, path):
            calls.append(path)

        def get(self, property_id):
            assert property_id == 11
            return 29.97

        def release(self):
            return None

    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(VideoCapture=Capture, CAP_PROP_FPS=11))
    posteval._MEDIA_FPS.clear()
    assert _bound_media_fps(request) == 29.97
    assert _bound_media_fps(request) == 29.97
    assert calls == [str(media)]


def test_completed_matrix_uses_dataset_scoped_numpy_official_labels():
    request = SimpleNamespace(dataset="ucf", media_id="shared", identity="vad:ucf:shared")
    payload = {"total_frames": 2, "causal_smoothed_scores": [.2, .8],
               "queries": [{"triggered": True, "frame_indices": [0], "final_score": .2,
                            "diagnostic_metadata": {"no_candidate": True}},
                           {"triggered": True, "frame_indices": [1], "final_score": .8,
                            "diagnostic_metadata": {"no_candidate": None}}]}
    tasks = {task: SimpleNamespace(records={request.identity: {"payload": payload}}) for task in REQUIRED_TASKS}
    frozen = SimpleNamespace(tasks=tasks, plan=SimpleNamespace(vad=(request,), protocol={}))
    report = stratify_completed_matrix(frozen=frozen, annotations={("ucf", "shared"): {"label": ["Abuse"], "truth": 1,
                                                                    "fps": 1., "intervals_raw": [[0, 2]]}},
                                       make_labels=lambda annotation, count: np.asarray([annotation["truth"]] * count),
                                       decision_threshold=.5)
    row = next(item for item in report["video_outcomes"] if item["model_task"] == "F:seed17" and item["stratum"] == "all")
    assert row["anomalous_recalled"] == 1 and row["anomalous_denominator"] == 1
    query = next(item for item in report["fast_triggered_query_outcomes"] if item["model_task"] == "F:seed17" and item["stratum"] == "all")
    assert query["anomalous_recalled"] == 1
    assert report["decision_threshold_source"].startswith("CLI descriptive sensitivity")
    assert report["outcome_scope"]["video_strata_membership"].startswith("temporal")
    assert report["category_video_outcomes"][0]["membership"] == "Abuse"
    assert report["temporal_strata_definitions"]["short_event"]["seconds"] == 8.0
    assert report["strata"]["short_event"]["true"][0]["model_task"] == "A:seed17"
    observed = next(item for item in report["fast_triggered_query_outcomes"] if item["stratum"] == "no_candidate")
    assert observed["anomalous_denominator"] == 1
    temporal = next(item for item in report["fast_triggered_query_outcomes"] if item["stratum"] == "short_event")
    category = next(item for item in report["fast_triggered_query_outcomes"] if item["stratum"] == "category")
    assert temporal["membership"] == "true" and temporal["anomalous_denominator"] == 2
    assert category["membership"] == "Abuse" and category["anomalous_denominator"] == 2
    assert report["strata"]["fast_trigger"]["true"][0]["model_task"] == "A:seed17"
