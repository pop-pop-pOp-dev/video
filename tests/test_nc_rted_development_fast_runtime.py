from datetime import datetime, timedelta, timezone
import copy
import json
from pathlib import Path
import pytest
from nc_rted.development_fast_runtime import (FastPreparationError, atomic_json, canonical_digest,
    digest_file, execute_plan, validate_plan, verify_scorer_inputs)


def _json(root, name, value):
    path = root / name
    path.write_text(json.dumps(value))
    return {"path": str(path), "sha256": digest_file(path)}


def _case(root):
    rows, catalog, split, database = [], [], [], {}
    for index in range(2):
        key = "normal" + str(index)
        video = root / (key + ".mp4"); video.write_bytes(key.encode())
        row = {"sample_id": key, "dataset": "ucf-crime", "key": key, "family": key,
               "query_index": index, "observed_seconds": (index * 4 + 3) / 4,
               "class": "normal", "scope": "vad_causal_latest_8s"}
        rows.append(row)
        catalog.append({"dataset": "ucf-crime", "media_key": key, "media_path": str(video),
                        "media_sha256": digest_file(video), "fps": 4., "frame_count": 20, "height": 8, "width": 8})
        split.append({"dataset": "ucf-crime", "key": key, "allocation": "development"})
        database[key] = {"fps": 4., "n_frames": 20}
    inputs = {"source_splits": _json(root, "split.json", split),
              "ucf_database": _json(root, "ucf.json", database), "xd_database": _json(root, "xd.json", {})}
    dev = _json(root, "dev.json", {"schema": "nc_rted_mechanism_development_prefixes/v1", "inputs": inputs, "records": rows})
    cat = _json(root, "media.json", {"development_manifest": dev, "media": catalog})
    runtime_rows = [{**row, **{k: item[k] for k in ("media_path", "media_sha256")}} for row, item in zip(rows, catalog)]
    runtime = _json(root, "runtime.json", {"schema": "nc_rted_mechanism_development_runtime_inputs/v1",
        "inputs": {**inputs, "development_manifest": dev, "media_catalog": cat}, "records": runtime_rows})
    inherited = {"training_inputs": {"fast": {"identity": {"frozen": "same"}, "protocols": {"ucf-crime": {}, "xd-violence": {}}}}}
    plan = {"schema": "nc_rted_mechanism_development_fast_plan/v1", "runtime_inputs": runtime,
            "fast_identity": inherited["training_inputs"]["fast"]["identity"], "protocols": inherited["training_inputs"]["fast"]["protocols"],
            "sources": [{**{k: row[k] for k in ("dataset", "key", "media_path", "media_sha256")}, "max_query_index": row["query_index"]} for row in runtime_rows]}
    return plan, inherited


@pytest.mark.parametrize("change", ["subset", "future", "identity", "duplicate", "endpoint"])
def test_rejects_scope_and_frozen_identity_changes_before_scorer(tmp_path, change):
    plan, inherited = _case(tmp_path)
    if change == "subset": plan["sources"].pop()
    elif change == "future": plan["sources"][0]["max_query_index"] += 1
    elif change == "identity": plan["fast_identity"] = {"different": True}
    elif change == "duplicate": plan["sources"].append(plan["sources"][0])
    else:
        runtime = json.loads(Path(plan["runtime_inputs"]["path"]).read_text())
        runtime["records"][0]["observed_seconds"] = 9
        plan["runtime_inputs"] = _json(tmp_path, "runtime.json", runtime)
    with pytest.raises(FastPreparationError): validate_plan(plan, inherited)


def test_interruption_reuses_committed_media_and_rejects_corruption(tmp_path):
    plan, inherited = _case(tmp_path); media = validate_plan(plan, inherited)
    journal, output = tmp_path / "journal", tmp_path / "output.json"
    binding = {"plan": canonical_digest(plan), "scorer": "frozen"}
    calls = []
    def interrupted(source, maximum):
        calls.append(source["key"])
        if len(calls) == 2: raise RuntimeError("simulated interruption")
        return [.2] * (maximum + 1)
    kwargs = dict(plan=plan, media=media, binding=binding, journal=journal, output=output,
                  deadline=datetime.now(timezone.utc)+timedelta(minutes=1), minimum_free_bytes=0)
    with pytest.raises(RuntimeError, match="interruption"):
        execute_plan(**kwargs, scorer_factory=lambda: interrupted)
    assert not output.exists()
    resumed = []
    def scoring(source, maximum):
        resumed.append(source["key"]); return [.3] * (maximum + 1)
    result = execute_plan(**kwargs, scorer_factory=lambda: scoring)
    assert resumed == ["normal1"]
    assert len(result["media"]) == 2 and result["media"][0]["queries"][0]["fast_score"] == .2
    assert result["fast_identity"] == plan["fast_identity"]
    kwargs["output"] = tmp_path / "second.json"
    bad = journal / (canonical_digest(["ucf-crime", "normal0"]) + ".json")
    record = json.loads(bad.read_text()); record["media"]["queries"][0]["fast_score"] = .9
    bad.write_text(json.dumps(record))
    with pytest.raises(FastPreparationError, match="corrupt"):
        execute_plan(**kwargs, scorer_factory=lambda: pytest.fail("model must not load for corrupt record"))


def test_actual_model_inventory_rejects_different_adapter(tmp_path):
    model, adapter = tmp_path / "model", tmp_path / "stage1" / "selected" / "adapter"
    model.mkdir(); adapter.mkdir(parents=True)
    (model / "config.json").write_text("model"); (adapter / "adapter.bin").write_text("trained")
    vision = tmp_path / "vision.bin"; vision.write_text("vision")
    source = {}
    for name in ("precompute_pg_scores.py", "reactvau_fast_grid_stream.py", "get_prompt.py"):
        p = tmp_path / name; p.write_text("# frozen"); source[str(p)] = digest_file(p)
    selection = _json(adapter.parent.parent, "selected.json", {"checkpoint": "checkpoint-best", "adapter": "selected/adapter"})
    config = {"selected_json_sha256": selection["sha256"], "source_sha256": source, "image_size": 384,
        "target_fps": 4, "query_interval": 4, "batch_size": 1, "attn_implementation": "sdpa",
        "model_path": str(model), "stage1_output": str(adapter.parent.parent), "selected_checkpoint": "checkpoint-best",
        "model_inventory_sha256": {"config.json": digest_file(model / "config.json")},
        "streamforest_vision_weights_path": str(vision), "streamforest_vision_weights_sha256": digest_file(vision),
        "released_precompute": str(tmp_path / "precompute_pg_scores.py")}
    inherited = {"training_inputs": {"fast": {"identity": {"checkpoint": {"selection_sha256": selection["sha256"],
        "adapter_files": {"adapter.bin": digest_file(adapter / "adapter.bin")}}, "implementation": source}}}}
    assert verify_scorer_inputs(config, inherited)["lora_path"] == str(adapter)
    (adapter / "adapter.bin").write_text("other")
    with pytest.raises(FastPreparationError, match="model/adapter"): verify_scorer_inputs(config, inherited)


def test_atomic_commit_never_overwrites_existing_record(tmp_path):
    path = tmp_path / "record.json"; atomic_json(path, {"first": True})
    with pytest.raises(FastPreparationError): atomic_json(path, {"first": False})
    assert json.loads(path.read_text()) == {"first": True}
