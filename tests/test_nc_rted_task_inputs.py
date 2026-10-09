from types import SimpleNamespace
import copy

import pytest
import torch

from nc_rted.bridge import ObservationBatch
from nc_rted.features import CELL_FEATURE_DIM
from nc_rted.task_inputs import (FrozenTaskContext, InheritedTaskTokenizer, TaskInputError,
                                TrainingTask, teacher_batch, validate_observation_scope)


def context(end=8.):
    return FrozenTaskContext(torch.zeros(1, 2, 8), [torch.zeros(1, 3, 2, 2)], [(2, 2)],
                             end, (0., end), "original time message")


def observations(blocks=1, relations=2):
    mask = torch.ones(1, blocks, relations, 4, dtype=torch.bool)
    times = torch.tensor([1., 3., 5., 7.]).expand(1, blocks, relations, 4).clone()
    for block in range(blocks):
        times[:, block] += block * 8
    return ObservationBatch(torch.zeros(1, blocks, relations, 4, CELL_FEATURE_DIM), mask, times)


def row():
    return dict(window_id="x", aux_valid=True, relation_ids=["b", "a"],
                mask=[True] * 8 + [False] * 56,
                F_positions=[.2] * 4 + [.05] * 4 + [0.] * 56,
                F_quality=.8, S_quality=.8, U_quality=.4)


def test_teacher_reorders_audit_relations_without_altering_observation_support():
    observed = observations()
    before = observed.valid.clone()
    result = teacher_batch(row(), sample_id="x", group="F", relation_ids=("a", "b"), observations=observed)
    assert torch.equal(observed.valid, before)
    assert torch.allclose(result.position[0, 0, 0], torch.full((4,), .05))
    assert torch.allclose(result.position[0, 0, 1], torch.full((4,), .2))
    assert result.eligible.item() and result.quality.item() == pytest.approx(.8)
    u = teacher_batch(row(), sample_id="x", group="U", relation_ids=("a", "b"), observations=observed)
    assert torch.equal(u.position, result.position)
    assert u.quality.item() == pytest.approx(.4)


def test_teacher_rejection_is_distinct_from_valid_empty_evidence():
    observed = observations()
    rejected = teacher_batch(dict(window_id="x", aux_valid=False, rejection="insufficient_sources"),
                             sample_id="x", group="S", relation_ids=("a", "b"), observations=observed)
    empty = row(); empty.update(F_quality=0., S_quality=0., U_quality=0.)
    valid_empty = teacher_batch(empty, sample_id="x", group="F", relation_ids=("a", "b"), observations=observed)
    assert not rejected.eligible.item() and valid_empty.eligible.item()
    assert rejected.quality.item() == valid_empty.quality.item() == 0


@pytest.mark.parametrize("mutation", ["identity", "relations", "outside_support", "sf_differs", "phantom", "nan"])
def test_teacher_misalignment_fails_before_loss(mutation):
    target, observed = row(), observations()
    if mutation == "identity": target["window_id"] = "other"
    if mutation == "relations": target["relation_ids"] = ["a", "other"]
    if mutation == "outside_support": observed.valid[0, 0, 0, 0] = False
    if mutation == "sf_differs": target["S_quality"] = .7
    if mutation == "phantom": target["mask"][8] = True
    if mutation == "nan": target["F_positions"][0] = float("nan")
    with pytest.raises(TaskInputError):
        teacher_batch(target, sample_id="x", group="F", relation_ids=("a", "b"), observations=observed)


def test_full_caption_tail_is_required_and_future_observations_rejected():
    observed = observations(blocks=3)
    observed.observed_times[:, 2] = torch.tensor([16.1, 16.3, 16.7, 17.2])
    validate_observation_scope("caption", context(17.25), observed)
    with pytest.raises(TaskInputError, match="omit or duplicate"):
        validate_observation_scope("caption", context(17.25), observations(blocks=2))
    observed.observed_times[0, 2, 0, 3] = 17.3
    with pytest.raises(TaskInputError, match="causal block"):
        validate_observation_scope("caption", context(17.25), observed)
    with pytest.raises(TaskInputError, match="future"):
        FrozenTaskContext(context().visual_embeddings, context().images, context().image_sizes,
                          8., (0., 9.), "").validate()


def test_caption_passes_exact_original_answer_to_original_preprocessor():
    calls = []
    def multimodal(sources, data_args, msg):
        calls.append((copy.deepcopy(sources), msg))
        return sources
    def qwen(sources, tokenizer, has_image):
        assert has_image
        return dict(input_ids=torch.tensor([[1, -200, 2]]), labels=torch.tensor([[-100, -100, 2]]))
    instruction = dict(conversations=[dict(from_="unused")])
    instruction["conversations"] = [{"from": "human", "value": "<video>\nDescribe."},
                                   {"from": "gpt", "value": "Unmodified original caption."}]
    task = TrainingTask("caption:x:1", "caption", "x", "family", "video", None, None, instruction)
    before = copy.deepcopy(instruction)
    encoder = InheritedTaskTokenizer(None, SimpleNamespace(), multimodal, qwen)
    encoded = encoder.encode(task, context())
    assert calls == [([[{"from": "human", "value": "<image>\nDescribe."},
                       {"from": "gpt", "value": "Unmodified original caption."}]], "original time message")]
    assert task.instruction == before
    assert torch.equal(encoded.labels, torch.tensor([[-100, -100, 2]]))


def test_detection_endpoint_and_prompt_must_be_bound():
    encoder = InheritedTaskTokenizer(None, None, None, None)
    task = TrainingTask("x", "detection", "x", "family", "video", 7., 1, None)
    with pytest.raises(TaskInputError, match="endpoint"):
        encoder.encode(task, context(), detection_question="Question")
    task = TrainingTask("x", "detection", "x", "family", "video", 8., 1, None)
    with pytest.raises(TaskInputError, match="yes/no"):
        encoder.encode(task, context(), detection_question="Question", detection_scoring="cot")
