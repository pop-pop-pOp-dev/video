import json

import torch

from types import SimpleNamespace

from nc_rted.bridge import EvidenceSlowBridge, ObservationBatch, SlowInputs
from nc_rted.mechanism_diagnostics import (equal_norm_random_delta, evidence_with_position,
                                           prepare_intervened, permute_position,
                                           select_development_prefixes, write_development_manifest,
                                           write_development_runtime_inputs, development_detection_task,
                                           write_development_fast_plan, score_development_fast)
from nc_rted.model import RelationTimeEvidence


def _inputs():
    torch.manual_seed(4)
    features = torch.randn(1, 2, 3, 4, 5)
    valid = torch.tensor([[[[1, 1, 0, 0], [1, 1, 1, 0], [0, 1, 1, 1]],
                           [[1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 0]]]], dtype=torch.bool)
    times = torch.arange(8, dtype=torch.float32).reshape(1, 2, 4)
    return features, valid, times


def test_resampler_baseline_matches_model_and_permutations_preserve_support():
    model = RelationTimeEvidence(5, 7, width=12, heads=3).eval()
    features, valid, times = _inputs()
    direct = model(features, valid, times)
    baseline = evidence_with_position(model, features, valid, times)
    assert torch.allclose(direct.position, baseline.position)
    assert torch.allclose(direct.evidence_tokens, baseline.evidence_tokens)
    for axis in ("time", "relation"):
        changed = evidence_with_position(model, features, valid, times, axis=axis)
        assert torch.equal(changed.position[~valid], torch.zeros_like(changed.position[~valid]))
        assert torch.allclose(torch.sort(changed.position[valid]).values, torch.sort(direct.position[valid]).values)
        assert not torch.allclose(changed.evidence_tokens, direct.evidence_tokens)


def test_equal_norm_random_control_is_deterministic_and_zero_is_identity():
    delta = torch.tensor([[[1.0, -2.0], [3.0, 4.0]]])
    first, second = equal_norm_random_delta(delta), equal_norm_random_delta(delta)
    assert torch.equal(first, second)
    assert torch.allclose(torch.linalg.vector_norm(first.float()), torch.linalg.vector_norm(delta.float()))
    assert torch.equal(equal_norm_random_delta(torch.zeros_like(delta)), torch.zeros_like(delta))
    assert not torch.equal(equal_norm_random_delta(delta, seed=1), equal_norm_random_delta(delta, seed=2))


def test_no_candidate_permutation_is_the_model_bypass():
    model = RelationTimeEvidence(5, 7, width=12, heads=3).eval()
    features = torch.empty(1, 1, 0, 4, 5)
    valid = torch.empty(1, 1, 0, 4, dtype=torch.bool)
    times = torch.empty(1, 1, 4)
    direct = model(features, valid, times)
    changed = evidence_with_position(model, features, valid, times, axis="time")
    assert torch.equal(direct.evidence_tokens, changed.evidence_tokens)


def test_bounded_fast_snapshot_requests_only_selected_queries(tmp_path):
    import cv2
    path = str(tmp_path / "sample.mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 4, (8, 8))
    for _ in range(20): writer.write(torch.zeros(8, 8, 3, dtype=torch.uint8).numpy())
    writer.release()
    import hashlib
    runtime = tmp_path / "runtime.json"; runtime.write_text("{}")
    digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
    plan = {"schema": "nc_rted_mechanism_development_fast_plan/v1", "fast_identity": {"frozen": "x"},
            "runtime_inputs": {"path": str(runtime), "sha256": hashlib.sha256(runtime.read_bytes()).hexdigest()}, "sources":
            [{"dataset": "ucf-crime", "key": "sample", "media_path": path, "media_sha256": digest, "max_query_index": 1}]}
    seen = []
    output = score_development_fast(plan, score_source=lambda source, maximum: seen.append(maximum) or [.1] * (maximum + 1))
    assert seen == [1] and [q["index"] for q in output["media"][0]["queries"]] == [0, 1]


def _databases():
    rows = {}
    for dataset in ("ucf-crime", "xd-violence"):
        rows[dataset] = {}
        for label, offset in (([], 0), ([[0.0, 64.0]], 1)):
            key = f"{dataset}-{offset}"
            rows[dataset][key] = {"label": label, "events": label, "fps": 4, "n_frames": 256}
    return rows


def test_development_selection_is_fixed_stratified_and_manifest_is_hashed(tmp_path):
    databases = _databases()
    splits = [{"dataset": dataset, "key": key, "allocation": "development"}
              for dataset, entries in databases.items() for key in entries]
    selected, status = select_development_prefixes(databases, splits, requested=16, per_family_cap=4)
    assert status == "FIXED_SOURCE_AND_CLASS_STRATIFIED_256"
    assert len(selected) == 16
    assert {row["dataset"] for row in selected} == {"ucf-crime", "xd-violence"}
    assert {row["class"] for row in selected} == {"normal", "anomalous"}
    source = tmp_path / "source_splits.json"; ucf = tmp_path / "ucf.json"; xd = tmp_path / "xd.json"; output = tmp_path / "manifest.json"
    source.write_text(json.dumps(splits)); ucf.write_text(json.dumps(databases["ucf-crime"])); xd.write_text(json.dumps(databases["xd-violence"]))
    document = write_development_manifest(output, splits_path=source, ucf_database_path=ucf, xd_database_path=xd, requested=16)
    assert document["status"] == status and len(json.loads(output.read_text())["records"]) == 16


def test_runtime_inputs_bind_only_development_prefixes_to_media(tmp_path):
    databases = _databases()
    splits = [{"dataset": dataset, "key": key, "allocation": "development"}
              for dataset, entries in databases.items() for key in entries]
    source, ucf, xd = tmp_path / "splits.json", tmp_path / "ucf.json", tmp_path / "xd.json"
    development, catalog, output = tmp_path / "development.json", tmp_path / "media.json", tmp_path / "runtime.json"
    source.write_text(json.dumps(splits)); ucf.write_text(json.dumps(databases["ucf-crime"])); xd.write_text(json.dumps(databases["xd-violence"]))
    write_development_manifest(development, splits_path=source, ucf_database_path=ucf, xd_database_path=xd, requested=16)
    catalog.write_text(json.dumps({"media": [{"dataset": dataset, "media_key": key, "media_path": f"/{key}.mp4", "media_sha256": "a" * 64}
                                               for dataset, entries in databases.items() for key in entries]}))
    bound = write_development_runtime_inputs(output, development_manifest=development, splits_path=source,
        ucf_database_path=ucf, xd_database_path=xd, media_catalog=catalog)
    assert bound["schema"] == "nc_rted_mechanism_development_runtime_inputs/v1"
    assert len(bound["records"]) == 16 and all(row["media_path"].endswith(".mp4") for row in bound["records"])
    task = development_detection_task(bound["records"][0])
    assert task.task == "detection" and task.label == int(bound["records"][0]["class"] == "anomalous")
    plan_path = tmp_path / "fast-plan.json"
    plan = write_development_fast_plan(plan_path, runtime_inputs=output, fast_identity={"frozen": "identity"},
        protocols={"ucf-crime": {"question": "q"}, "xd-violence": {"question": "q"}})
    assert plan["schema"] == "nc_rted_mechanism_development_fast_plan/v1"
    assert len(plan["sources"]) == 4 and all(row["max_query_index"] >= 0 for row in plan["sources"])


class _RawSlow(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(32, 7)
        self.config = SimpleNamespace(mm_patch_merge_type="flat", frame_aspect_ratio="square",
                                      tokenizer_model_max_length=64)

    def get_input_embeddings(self):
        return self.embedding

    def prepare_inputs_labels_for_LLM(self, input_ids, position_ids, attention_mask, past_key_values,
                                      labels, images, visual_embeddings, modalities, *, image_sizes):
        visual = visual_embeddings[0]
        text = self.embedding(input_ids.clamp_min(0))
        embeds = torch.cat((text[:, :1], visual, text[:, 2:]), dim=1)
        return None, position_ids, attention_mask, past_key_values, embeds, labels


def test_intervened_tokens_reach_inherited_slow_preparation():
    torch.manual_seed(8)
    bridge = EvidenceSlowBridge(_RawSlow(), RelationTimeEvidence(5, 7, width=12, heads=3)).eval()
    features, valid, times = _inputs()
    inputs = SlowInputs(torch.tensor([[1, -200, 2]]), torch.randn(1, 2, 7),
                        [torch.zeros(1)], [(1, 1)], attention_mask=torch.ones(1, 3, dtype=torch.bool))
    observations = ObservationBatch(features, valid, times)
    disabled, disabled_result = prepare_intervened(bridge, inputs, observations, name="branch_disable")
    changed, changed_result = prepare_intervened(bridge, inputs, observations, name="time_permute")
    random, random_result = prepare_intervened(bridge, inputs, observations, name="equal_norm_random",
                                                paired_structured_name="time_permute")
    assert disabled_result.token_delta_norm == 0.0
    assert changed.arguments["inputs_embeds"].shape[1] == disabled.arguments["inputs_embeds"].shape[1] + 16
    assert changed_result.token_delta_norm > 0
    assert random_result.token_delta_norm == changed_result.token_delta_norm
    assert not torch.allclose(changed.arguments["inputs_embeds"], random.arguments["inputs_embeds"])
