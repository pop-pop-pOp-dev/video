import hashlib
import json

import numpy as np

from nc_rted.features import PROCESS_FEATURE_DIM
from nc_rted.observation_extraction import ObservationJournal
from nc_rted.observation_seed import seed_committed_prefix
from nc_rted.teacher_records import CompactTeacherWindow


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compact(window_id):
    cells = np.full((4, PROCESS_FEATURE_DIM), np.nan, dtype=np.float32)
    cells[:2] = 1.0
    return CompactTeacherWindow({
        "dataset": "ucf-crime", "window_id": window_id, "source_family": "family", "content_alias": "alias",
        "fold": 0, "normal_permitted": True, "background": np.r_[np.float32(1), np.zeros(1151, dtype=np.float32)],
        "background_valid": True, "class_composition": np.r_[np.float32(1), np.zeros(79, dtype=np.float32)],
        "pairs": [{"pair_id": "0:1", "class_pair": [0, 1], "initial_geometry": np.zeros(5, dtype=np.float32),
                   "candidate_pair_count": 1, "valid_cells": np.array([True, True, False, False]), "process_cells": cells}],
    }, ("0:1",))


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")))
    return digest(path)


def prepared(tmp_path):
    ids = ("detection:ucf-crime:a:1", "detection:ucf-crime:b:1")
    old_cfg = {"schema": "nc_rted_observation_extraction/v2", "output": str(tmp_path / "old"),
               "frame_cache": str(tmp_path / "old-cache"), "cpu_threads": 50,
               "code_sha256": {"src/nc_rted/detector.py": "old-detector", "src/nc_rted/frozen_vision.py": "old-vision", "other": "same"}}
    old_cfg_path = tmp_path / "old.json"; old_sha = write_json(old_cfg_path, old_cfg)
    source = ObservationJournal(tmp_path / "old", {"config_sha256": old_sha}, ids, reserved_free_bytes=0)
    with source.writer():
        source.put(compact(ids[0]))
    probe_cfg = {**old_cfg, "output": str(tmp_path / "probe"), "frame_cache": str(tmp_path / "probe-cache"), "cpu_threads": 8,
                 "code_sha256": {"src/nc_rted/detector.py": "new-detector", "src/nc_rted/frozen_vision.py": "new-vision", "other": "same"}}
    probe_cfg_path = tmp_path / "probe.json"; probe_cfg_sha = write_json(probe_cfg_path, probe_cfg)
    probe = ObservationJournal(tmp_path / "probe", {"config_sha256": probe_cfg_sha, "cpu_threads": 8}, ids, reserved_free_bytes=0)
    with probe.writer():
        probe.put(compact(ids[0]))
    source_run_sha, probe_run_sha = digest(tmp_path / "old" / "run.json"), digest(tmp_path / "probe" / "run.json")
    report = tmp_path / "equivalence.json"
    report_value = {"status": "PASS_EQUIVALENCE", "three_windows": [
        {"window_id": ids[0], "payload_equal": True, "index_entry_equal": True} for _ in range(3)],
        "source_config": {"path": str(old_cfg_path), "sha256": old_sha},
        "source_run": {"path": str(tmp_path / "old" / "run.json"), "sha256": source_run_sha},
        "probe_config": {"path": str(probe_cfg_path), "sha256": probe_cfg_sha},
        "probe_run": {"path": str(tmp_path / "probe" / "run.json"), "sha256": probe_run_sha}}
    report_sha = write_json(report, report_value)
    new_cfg = {**old_cfg, "output": str(tmp_path / "new"), "frame_cache": str(tmp_path / "new-cache"), "cpu_threads": 8,
               "code_sha256": {"src/nc_rted/detector.py": "new-detector", "src/nc_rted/frozen_vision.py": "new-vision", "other": "same"},
               "resume_parent": {"schema": "nc_rted_equivalent_observation_parent/v1",
                   "config": {"path": str(old_cfg_path), "sha256": old_sha},
                   "run": {"path": str(tmp_path / "old" / "run.json"), "sha256": source_run_sha},
                   "equivalence_report": {"path": str(report), "sha256": report_sha},
                   "probe_config": {"path": str(probe_cfg_path), "sha256": probe_cfg_sha},
                   "probe_run": {"path": str(tmp_path / "probe" / "run.json"), "sha256": probe_run_sha},
                   "reuse_policy": "validated immutable committed window records; preserve parent; compute missing fixed-selection windows only",
                   "source_change_scope": ["src/nc_rted/detector.py", "src/nc_rted/frozen_vision.py"]}}
    new_cfg_path = tmp_path / "new.json"; new_sha = write_json(new_cfg_path, new_cfg)
    return dict(source=tmp_path / "old", old_cfg=old_cfg_path, old_sha=old_sha, new_cfg=new_cfg_path, new_sha=new_sha,
                report=report, report_sha=report_sha, probe=tmp_path / "probe", probe_sha=probe_cfg_sha, ids=ids)


def seed(values):
    return seed_committed_prefix(source_root=values["source"], source_config=values["old_cfg"],
        source_config_sha256=values["old_sha"], target_config=values["new_cfg"], target_config_sha256=values["new_sha"],
        equivalence_report=values["report"], equivalence_sha256=values["report_sha"], probe_output=values["probe"],
        probe_config_sha256=values["probe_sha"], reserved_free_bytes=0)


def test_seed_skips_uncommitted_source_window(tmp_path):
    values = prepared(tmp_path)
    receipt = seed(values)
    assert receipt["status"] == "PARTIAL_RESUMABLE_OBSERVATIONS"
    assert receipt["copied_window_ids"] == [values["ids"][0]]
    destination = tmp_path / "new" / "windows"
    assert len(list(destination.iterdir())) == 1


def test_seed_retry_preserves_source_and_never_overwrites_window(tmp_path):
    values = prepared(tmp_path)
    source_run = values["source"] / "run.json"
    before = digest(source_run)
    first = seed(values)
    source_file = next((values["source"] / "windows").rglob("*.npz"))
    target_file = next((tmp_path / "new" / "windows").rglob("*.npz"))
    second = seed(values)
    assert digest(source_run) == before
    assert source_file.stat().st_ino == target_file.stat().st_ino
    assert first["copied_window_ids"] == [values["ids"][0]] and second["copied_window_ids"] == []
    assert second["reused_window_ids"] == [values["ids"][0]]
