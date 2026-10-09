from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from nc_rted.detection_provider import (DetectionMemoryReplay, DetectionPrefix, DetectionProtocol,
    DetectionQuery, FrozenDetectionProvider)
from nc_rted.task_inputs import TaskInputError, TrainingTask


class FakeMemory:
    def __init__(self, *args, **kwargs):
        self.events, self.pool, self.frames = [], [], []
        self.fail_pool = False

    def update(self, value):
        self.events.append("update")
        self.frames.append(value[:1])

    def update_with_anomaly_score(self, value, anomaly_score):
        self.events.append(("scored_update", anomaly_score))
        self.frames.append(value[:1])

    def get_memory_tokens(self, rt_anomaly_tokens=None):
        self.events.append(("read", len(self.pool), rt_anomaly_tokens is not None))
        values = self.frames + self.pool
        if rt_anomaly_tokens is not None:
            values += [rt_anomaly_tokens.reshape(-1, 1152)[:1]]
        return torch.cat(values).unsqueeze(0)

    def update_anomaly_pool(self, value, score):
        self.events.append(("pool", score))
        if self.fail_pool:
            raise RuntimeError("partial update")
        self.pool.append(value[:1])

    def get_anomaly_context(self):
        return {"context_str": f"{len(self.pool)} past events"}


def model():
    projector = nn.Module()
    projector.mlp = nn.Linear(1152, 8, bias=False)
    projector.requires_grad_(False)
    return SimpleNamespace(get_model=lambda: SimpleNamespace(mm_projector=projector))


def protocol(**kwargs):
    return replace(DetectionProtocol("Score {score_pct}; {anomaly_context}", "skeptical", "simple",
                                    True, False, .5, .6), **kwargs)


def query(index, score=.8, count=4, dense=False):
    start = index * 4
    return DetectionQuery(index, tuple(range(start, start + count)),
        tuple(i / 4 for i in range(start, start + count)), score,
        torch.full((729, 1152), float(index + 1)),
        torch.full((4, 729, 1152), float(index + 1)) if dense else None)


def formatter(style, end, count):
    return f"{style}:{end}:{count}"


def replay(**kwargs):
    return DetectionMemoryReplay(model(), protocol(**kwargs), memory_factory=FakeMemory,
                                 time_formatter=formatter)


def test_original_memory_order_and_pg_pool_independent_of_slow():
    r = replay()
    r.step(query(0))
    context, question = r.step(query(1), capture=True, image_height=8, image_width=12)
    assert r.memory.events == [("scored_update", .8), ("pool", .8),
                               ("scored_update", .8), ("read", 1, False), ("pool", .8)]
    assert question == "Score 80; 1 past events"
    assert context.sampled_frame_times == (.75, 1.75)
    assert context.time_message == "simple:1.75:2"
    assert context.image_sizes == [(8, 12)]
    assert context.visual_embeddings.shape == (1, 3, 8)
    assert not context.visual_embeddings.requires_grad
    assert not torch.is_inference(context.visual_embeddings)


def test_nontriggered_pool_is_retained_and_disabled_enhancement_does_not_write_pool():
    r = replay(trigger_threshold=.9, pool_threshold=.6)
    r.step(query(0, score=.7))
    assert r.memory.events[-1] == ("pool", .7)
    r = replay(memory_enhancement=False, prompt_style="neutral", question_template="Unmodified")
    context, question = r.step(query(0), capture=True)
    assert r.memory.events == ["update", ("read", 0, False)]
    assert question == "Unmodified"


def test_eos_dense_rt_uses_original_padding_without_inventing_timestamps():
    r = replay(rt_anomaly=True)
    r.step(query(0))
    context, _ = r.step(query(1, count=2, dense=True), capture=True)
    assert context.observed_seconds == 1.25
    assert context.sampled_frame_times == (.75, 1.25)
    assert ("read", 1, True) in r.memory.events
    r = replay(rt_anomaly=True)
    with pytest.raises(TaskInputError, match="four-frame"):
        r.step(query(0), capture=True)
    assert r.memory.events == []


@pytest.mark.parametrize("mutation", ["index", "frame_order", "time_order", "score", "dtype", "grad"])
def test_malformed_queries_cannot_mutate_memory(mutation):
    r, q = replay(), query(0)
    if mutation == "index": q = replace(q, index=1)
    if mutation == "frame_order": q = replace(q, frame_indices=(0, 1, 1, 3))
    if mutation == "time_order": q = replace(q, frame_times_s=(0., .5, .25, .75))
    if mutation == "score": q = replace(q, fast_score=float("nan"))
    if mutation == "dtype": q = replace(q, last_frame_patches=q.last_frame_patches.bfloat16())
    if mutation == "grad": q = replace(q, last_frame_patches=q.last_frame_patches.requires_grad_())
    with pytest.raises(TaskInputError):
        r.step(q, capture=True)
    assert r.memory.events == []


def test_partial_replay_failure_requires_reconstruction():
    r = replay()
    r.memory.fail_pool = True
    with pytest.raises(RuntimeError, match="partial"):
        r.step(query(0))
    with pytest.raises(TaskInputError, match="reconstructed"):
        r.step(query(0))


def test_provider_never_requests_a_future_query_and_checks_endpoint(monkeypatch):
    from nc_rted import detection_provider as module
    task = TrainingTask("sample", "detection", "ucf", "family", "media", 1.75, 1, None, 1)
    catalog = SimpleNamespace(tasks={"sample": task})
    visits = []
    def queries():
        for i in range(2):
            visits.append(i)
            yield query(i)
        raise AssertionError("future frames requested")
    reader = lambda dataset, key, target: DetectionPrefix(queries(), 8, 12)
    observation_reader = lambda *args: SimpleNamespace(features="features", relation_ids=("0:1",))
    monkeypatch.setattr(module, "pack_observation_blocks", lambda *args, **kwargs: "observation")
    p = FrozenDetectionProvider(model(), catalog, {"ucf": protocol()}, reader, observation_reader,
                                memory_factory=FakeMemory, time_formatter=formatter)
    material = p("sample")
    assert visits == [0, 1] and material.relation_ids == ("0:1",)
    assert material.context.observed_seconds == 1.75
    catalog.tasks["sample"] = replace(task, observed_seconds=1.5)
    with pytest.raises(TaskInputError, match="endpoint"):
        p("sample")
