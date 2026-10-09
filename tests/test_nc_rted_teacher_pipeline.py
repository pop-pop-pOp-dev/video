import json
from pathlib import Path

import numpy as np
import pytest

from nc_rted.features import PROCESS_FEATURE_DIM
from nc_rted import teacher_pipeline
from nc_rted.teacher_pipeline import (PipelineError, build_teachers, detection_window_id,
                                      parse_frozen_records, publish_teacher_manifest, relation_ids)


def raw_window(name, fold, *, normal, family=None, alias=None):
    process = np.zeros((4, PROCESS_FEATURE_DIM), dtype=float).tolist()
    return {
        "dataset": "ucf-crime", "window_id": name, "source_family": family or name + "-family",
        "content_alias": alias or name + "-alias", "fold": fold, "normal_permitted": normal,
        "background": [1.] + [0.] * 1151, "background_valid": True,
        "class_composition": [.5, .5] + [0.] * 78,
        "pairs": [{"pair_id": name + "-pair", "class_pair": [0, 2], "initial_geometry": [.2, 0, 0, 0, 0],
                   "candidate_pair_count": 1, "valid_cells": [True] * 4, "process_cells": process}],
    }


def document(windows):
    return {"schema": "nc_rted_frozen_feature_records/v1", "split": "train", "windows": windows}


def test_parser_is_train_only_and_preserves_masked_nan_policy():
    row = raw_window("w", 0, normal=True)
    row["pairs"][0]["valid_cells"] = [True, False, False, False]
    row["pairs"][0]["process_cells"][1] = [None] * PROCESS_FEATURE_DIM
    row["pairs"][0]["process_cells"][2] = [None] * PROCESS_FEATURE_DIM
    row["pairs"][0]["process_cells"][3] = [None] * PROCESS_FEATURE_DIM
    parsed = parse_frozen_records(document([row]))
    assert np.isnan(parsed[0].pairs[0].process_cells[1]).all()
    with pytest.raises(PipelineError, match="train-only"):
        parse_frozen_records({**document([raw_window("x", 0, normal=True)]), "split": "test"})


def test_crossfit_pipeline_emits_explicit_aux_rejection_when_c_support_is_missing():
    rows = [raw_window("q", 0, normal=False)] + [raw_window(f"r{i}", i + 2, normal=True) for i in range(3)]
    result = build_teachers(parse_frozen_records(document(rows)))
    q = next(row for row in result["rows"] if row["window_id"] == "q")
    assert not q["aux_valid"]
    assert q["F_quality"] is None and q["joint"] is None
    assert "64..128" in q["rejection"]


def test_fold_alias_or_family_crossing_fails_before_teacher_construction():
    rows = [raw_window("a", 0, normal=True, family="shared"), raw_window("b", 1, normal=True, family="shared")]
    with pytest.raises(PipelineError, match="crosses teacher folds"):
        build_teachers(parse_frozen_records(document(rows)))


def test_publish_is_immutable_and_records_input_config_hashes(tmp_path):
    input_path, config_path, output = tmp_path / "records.json", tmp_path / "frozen.yaml", tmp_path / "teacher"
    input_path.write_text(json.dumps(document([raw_window("q", 0, normal=False)])))
    config_path.write_text("frozen: true\n")
    publish_teacher_manifest(output, {"schema": "nc_rted_teacher_pipeline/v1", "rows": []}, input_path, config_path)
    payload = json.loads((output / "teacher_manifest.json").read_text())
    assert set(payload["provenance"]) == {"input_sha256", "config_sha256"}
    with pytest.raises(PipelineError, match="overwrite"):
        publish_teacher_manifest(output, {"schema": "nc_rted_teacher_pipeline/v1", "rows": []}, input_path, config_path)


def test_relation_ids_map_actual_pair_order_to_fixed_four_cell_slots():
    row = raw_window("q", 0, normal=True)
    second = dict(row["pairs"][0]); second["pair_id"] = "q-pair-2"
    row["pairs"].append(second)
    parsed = parse_frozen_records(document([row]))[0]
    assert relation_ids(parsed) == ["q-pair", "q-pair-2"]
    assert detection_window_id("ucf-crime", "Abuse001_x264", 7) == "detection:ucf-crime:Abuse001_x264:7"


def test_empty_pair_window_is_an_explicit_rejection_not_a_parse_failure():
    row = raw_window("empty", 0, normal=False)
    row["pairs"] = []
    parsed = parse_frozen_records(document([row]))
    result = build_teachers(parsed)
    rejected = result["rows"][0]
    assert not rejected["aux_valid"]
    assert rejected["F_quality"] is None


@pytest.mark.parametrize("field,value", [("normal_permitted", "false"), ("background_valid", 0)])
def test_parser_requires_actual_boolean_record_flags(field, value):
    row = raw_window("typed", 0, normal=True)
    row[field] = value
    with pytest.raises(PipelineError, match=field):
        parse_frozen_records(document([row]))


def test_unmatched_pair_is_masked_while_other_pairs_remain_usable(monkeypatch):
    row = raw_window("q", 0, normal=True)
    matched = dict(row["pairs"][0])
    matched["pair_id"] = "usable"
    unmatched = dict(row["pairs"][0])
    unmatched["pair_id"] = "missing"
    row["pairs"] = [matched, unmatched]
    window = parse_frozen_records(document([row]))[0]

    def pair_distances(pair, _references, _scales):
        if pair.pair_id == "missing":
            raise PipelineError("no compatible pair")
        return np.ones((3, 4), dtype=float)

    monkeypatch.setattr(teacher_pipeline, "_pair_distances", pair_distances)
    support = {}
    distances, valid = teacher_pipeline._window_teacher_distances(window, (), {}, support)
    assert valid[:4].all()
    assert not valid[4:8].any()
    assert np.isnan(distances[:, 4:8]).all()
    assert support["supported_pair_count"] == 1
    assert support["rejected_pairs"] == [{"pair_id": "missing", "reason": "no compatible pair"}]


def test_static_filter_excludes_missing_background_from_r_support():
    usable = raw_window("usable", 2, normal=True)
    missing = raw_window("missing", 3, normal=True)
    missing["background_valid"] = False
    parsed = parse_frozen_records(document([usable, missing]))
    eligible, excluded = teacher_pipeline._static_eligible_records(parsed)
    assert [item.window_id for item in eligible] == ["usable"]
    assert excluded == [{"window_id": "missing", "reason": "missing reliable window background"}]
