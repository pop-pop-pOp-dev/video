import numpy as np
import pytest

from nc_rted.features import PROCESS_FEATURE_DIM
from nc_rted.teacher_records import CompactTeacherWindow
from nc_rted import teacher_store
from nc_rted.teacher_store import (TeacherStoreError, build_teachers_from_store,
                                    load_teacher_store, write_teacher_store)


def accepted(name="detection:ucf-crime:key:1"):
    process = np.full((4, PROCESS_FEATURE_DIM), np.nan, dtype=np.float32)
    process[:2] = 1.25
    record = {
        "dataset": "ucf-crime", "window_id": name, "source_family": "family-a", "content_alias": "alias-a",
        "fold": 0, "normal_permitted": True, "background": np.r_[np.float32(1), np.zeros(1151, dtype=np.float32)],
        "background_valid": True, "class_composition": np.r_[np.float32(1), np.zeros(79, dtype=np.float32)],
        "pairs": [{"pair_id": "0:1", "class_pair": [0, 2], "initial_geometry": np.zeros(5, dtype=np.float32),
                   "candidate_pair_count": 1, "valid_cells": np.array([True, True, False, False]),
                   "process_cells": process}],
    }
    return CompactTeacherWindow(record, ("0:1",))


def rejected(name="detection:ucf-crime:key:2"):
    record = {"window_id": name, "dataset": "ucf-crime", "aux_valid": False, "rejection": "tracking failure"}
    return CompactTeacherWindow(record, (), {"reason": "tracking failure"})


def test_chunked_store_round_trips_dtype_masked_nan_and_rejection(tmp_path):
    destination = tmp_path / "teacher-store"
    write_teacher_store(destination, (accepted(), rejected()), reserved_free_bytes=0)
    loaded = {item.record["window_id"]: item for item in load_teacher_store(destination)}
    restored = loaded["detection:ucf-crime:key:1"].record
    assert restored["pairs"][0]["process_cells"].dtype == np.float32
    assert np.isnan(restored["pairs"][0]["process_cells"][2:]).all()
    assert loaded["detection:ucf-crime:key:2"].record["aux_valid"] is False


def test_uncommitted_directory_is_not_a_visible_store(tmp_path):
    destination = tmp_path / "teacher-store"
    (tmp_path / ".teacher-store.tmp-interrupted").mkdir()
    with pytest.raises(TeacherStoreError, match="absent or not committed"):
        load_teacher_store(destination)
    assert not destination.exists()


def test_store_detects_payload_tampering(tmp_path):
    destination = tmp_path / "teacher-store"
    write_teacher_store(destination, (accepted(),), reserved_free_bytes=0)
    payload = next((destination / "chunks").iterdir())
    payload.write_bytes(payload.read_bytes() + b"tampered")
    with pytest.raises(TeacherStoreError, match="payload hash mismatch"):
        load_teacher_store(destination)


def test_all_rejections_are_preserved_and_build_without_payloads(tmp_path):
    destination = tmp_path / "teacher-store"
    item = rejected()
    write_teacher_store(destination, (item,), reserved_free_bytes=0)
    output = build_teachers_from_store(destination)
    assert output["rows"] == [item.record]


def test_reserve_is_checked_after_each_chunk_and_failed_stage_is_not_published(tmp_path, monkeypatch):
    calls = []

    def reserve(_directory, _bytes):
        calls.append(1)
        if len(calls) == 3:  # initial, before payload, then post-payload
            raise TeacherStoreError("teacher store would violate the 20 GiB free-space reserve")

    monkeypatch.setattr(teacher_store, "_require_reserve", reserve)
    destination = tmp_path / "teacher-store"
    with pytest.raises(TeacherStoreError, match="reserve"):
        write_teacher_store(destination, (accepted(),), reserved_free_bytes=0)
    assert len(calls) == 3
    assert not destination.exists()


def test_accepted_zero_pair_payload_round_trip(tmp_path):
    source = accepted()
    empty = CompactTeacherWindow({**source.record, "pairs": []}, ())
    output = tmp_path / "empty-pairs"
    write_teacher_store(output, (empty,), reserved_free_bytes=0)
    loaded = load_teacher_store(output)
    assert loaded[0].record["pairs"] == [] and loaded[0].relation_ids == ()


def test_rejection_only_metadata_reserve_checked_before_publish(tmp_path, monkeypatch):
    calls = []
    def reserve(_directory, threshold):
        calls.append(threshold)
        if len(calls) == 2:
            assert threshold >= 16384
            raise TeacherStoreError("metadata reserve")
    monkeypatch.setattr(teacher_store, "_require_reserve", reserve)
    output = tmp_path / "rejections"
    with pytest.raises(TeacherStoreError, match="metadata reserve"):
        write_teacher_store(output, (rejected(),), reserved_free_bytes=0)
    assert not output.exists()


def test_rejection_only_final_reserve_prevents_publish(tmp_path, monkeypatch):
    calls = []
    def reserve(_directory, threshold):
        calls.append(threshold)
        if len(calls) == 3:
            raise TeacherStoreError("final reserve")
    monkeypatch.setattr(teacher_store, "_require_reserve", reserve)
    output = tmp_path / "rejections"
    with pytest.raises(TeacherStoreError, match="final reserve"):
        write_teacher_store(output, (rejected(),), reserved_free_bytes=0)
    assert not output.exists()
