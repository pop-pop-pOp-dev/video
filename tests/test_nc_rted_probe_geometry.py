from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch

from nc_rted.prediction_worker import (PredictionExecutionError, count_generation_execution,
                                       count_slow_execution, validate_vad_payload)


def _media() -> dict:
    return {"frame_count": 12, "fps": 8.0, "target_fps": 4, "query_interval": 4,
            "queries": [{"index": 0, "frame_indices": [0, 2, 4, 6], "fast_score": 0.2},
                        {"index": 1, "frame_indices": [8, 10], "fast_score": 0.8}]}


def _payload() -> dict:
    return {"total_frames": 12, "sample_interval": 2, "causal_smoothed_scores": [0.2] * 12,
            "queries": [{"query_index": 0, "frame_indices": [0, 2, 4, 6], "fast_score": 0.2,
                         "final_score": 0.2, "triggered": False, "slow_score": None, "fused_score": None},
                        {"query_index": 1, "frame_indices": [8, 10], "fast_score": 0.8,
                         "final_score": 0.7, "triggered": True, "slow_score": 0.6, "fused_score": 0.7}]}


def test_strict_vad_geometry_accepts_the_complete_bound_timeline():
    assert validate_vad_payload(_payload())["total_frames"] == 12
    assert validate_vad_payload(_payload(), expected_media=_media(), trigger_threshold=0.5)["total_frames"] == 12


@pytest.mark.parametrize("change,error", [
    (lambda value: (value.update(total_frames=11), value.update(causal_smoothed_scores=value["causal_smoothed_scores"][:11])), "geometry differs"),
    (lambda value: value["queries"].pop(), "omits bound Fast queries"),
    (lambda value: value["queries"][1].update(fast_score=0.7), "query differs"),
    (lambda value: value["queries"][1].update(triggered=False, slow_score=None), "trigger differs"),
])
def test_strict_vad_geometry_rejects_truncation_or_fast_trigger_changes(change, error):
    payload = copy.deepcopy(_payload())
    change(payload)
    with pytest.raises(PredictionExecutionError, match=error):
        validate_vad_payload(payload, expected_media=_media(), trigger_threshold=0.5)


def test_strict_vad_geometry_rejects_jointly_corrupted_fast_and_payload_timeline():
    media = _media()
    payload = _payload()
    media["queries"] = [
        {"index": 0, "frame_indices": [1, 3, 5, 7], "fast_score": 0.2},
        {"index": 1, "frame_indices": [9, 11], "fast_score": 0.8},
    ]
    for row, bound in zip(payload["queries"], media["queries"]):
        row["frame_indices"] = bound["frame_indices"]
    with pytest.raises(PredictionExecutionError, match="bound Fast query geometry"):
        validate_vad_payload(payload, expected_media=media, trigger_threshold=0.5)


class _Slow(torch.nn.Module):
    def forward(self, value):
        return value + 1


class _GenerationRuntime:
    def __init__(self, slow):
        self.slow = slow

    def generate(self, value):
        for _ in range(3):
            value = self.slow(value)
        return value


def test_execution_counters_distinguish_real_slow_forwards_from_generation_calls():
    slow = _Slow()
    loaded = SimpleNamespace(bridge=SimpleNamespace(slow=slow), hivau_inference=_GenerationRuntime(slow))
    counters = {}
    with count_slow_execution(loaded, counters), count_generation_execution(loaded, counters):
        assert loaded.hivau_inference.generate(torch.tensor(1)).item() == 4
        assert slow(torch.tensor(2)).item() == 3
    assert counters == {"completed_slow_forwards": 4, "completed_generation_calls": 1}
    loaded.hivau_inference.generate(torch.tensor(0))
    assert counters == {"completed_slow_forwards": 4, "completed_generation_calls": 1}
