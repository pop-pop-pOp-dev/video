import json
import hashlib
from pathlib import Path

import pytest

from nc_rted import vau_media as media


def test_project_ranges_exposes_only_media_ranges(tmp_path):
    source = tmp_path / "ranges.json"
    source.write_text(json.dumps({"P": {"label": ["sealed"], "events": [[1, 3]],
                                           "clips": [[[1, 2], [2, 3]]],
                                           "events_summary": ["sealed text"]}}))
    assert media.project_ranges(source) == {"P": {"events": [(1.0, 3.0)],
                                                    "clips": [[(1.0, 2.0), (2.0, 3.0)]]}}


def test_identity_requires_the_full_official_question_denominator(tmp_path):
    path = tmp_path / "identity.json"
    path.write_text(json.dumps([{"id": 0, "video": "ucf-crime/videos/test/P.mp4"}]))
    with pytest.raises(media.VAUMediaError, match="3339"):
        media._read_identity(path)


@pytest.mark.parametrize(("video", "expected"), [
    ("ucf-crime/videos/test/Abuse028_x264.mp4", ("ucf-crime", "Abuse028_x264", None, None)),
    ("ucf-crime/events/test/Abuse028_x264_E0.mp4", ("ucf-crime", "Abuse028_x264", 0, None)),
    ("xd-violence/clips/test/v=xx__#1_label_A_E12C4.mp4", ("xd-violence", "v=xx__#1_label_A", 12, 4)),
])
def test_official_path_parser_binds_only_filename_range_identity(video, expected):
    assert media._parts(video) == expected


def test_materializer_is_resumable_and_catalog_rows_have_runtime_keys_only(tmp_path, monkeypatch):
    parent = tmp_path / "parent.mp4"
    parent.write_bytes(b"immutable parent bytes")
    geometry = media.Geometry(30.0, 12, 8, 16)
    monkeypatch.setattr(media, "_probe", lambda path: geometry)
    monkeypatch.setattr(media, "_free_bytes", lambda path: media.MIN_FREE_BYTES)
    plans = [media.MediaPlan("ucf", f"ucf-crime/videos/test/P{i}.mp4", "P", parent, None, None)
             for i in range(1369)]
    output = tmp_path / "out"
    catalog, journal = media.materialize(plans, output_root=output)
    document = json.loads(catalog.read_text())
    expected_keys = {"dataset", "media_key", "media_path", "media_sha256", "fps", "frame_count", "height", "width", "request_index"}
    assert document["schema"] == media.CATALOG_SCHEMA
    assert len(document["media"]) == 1369
    assert all(set(row) == expected_keys for row in document["media"])
    assert len(journal.read_text().splitlines()) == 2738
    # A second invocation validates every recovered hash/geometry and must not
    # overwrite either immutable artifact.
    second, _ = media.materialize(plans, output_root=output)
    assert second == catalog
    assert len(journal.read_text().splitlines()) == 2738


def test_interrupted_journal_tail_is_preserved_and_repaired(tmp_path):
    journal = tmp_path / "journal.jsonl"
    journal.write_bytes(b'{"schema":"nc_rted_vau_materialization_journal/v1","status":"complete"')
    assert media._completed(journal) == {}
    assert journal.read_bytes() == b""
    preserved = list(tmp_path.glob("journal.jsonl.interrupted-*"))
    assert len(preserved) == 1
    assert preserved[0].read_bytes() == b'{"schema":"nc_rted_vau_materialization_journal/v1","status":"complete"'


def test_completed_record_integrity_failure_never_regenerates(tmp_path, monkeypatch):
    parent = tmp_path / "parent.mp4"
    parent.write_bytes(b"immutable parent bytes")
    geometry = media.Geometry(30.0, 12, 8, 16)
    monkeypatch.setattr(media, "_probe", lambda path: geometry)
    monkeypatch.setattr(media, "_free_bytes", lambda path: media.MIN_FREE_BYTES)
    plans = [media.MediaPlan("ucf", f"ucf-crime/videos/test/P{i}.mp4", "P", parent, None, None)
             for i in range(1369)]
    output = tmp_path / "out"
    output.mkdir()
    bad = {"schema": media.JOURNAL_SCHEMA, "status": "complete", "video": plans[0].video,
           "parent_path": str(parent), "parent_sha256": media.sha256_file(parent), "range_s": None,
           "media_path": str(parent), "media_sha256": "0" * 64,
           "geometry": {"fps": 30.0, "frame_count": 12, "height": 8, "width": 16}}
    (output / "journal.jsonl").write_text(json.dumps(bad) + "\n")
    with pytest.raises(media.VAUMediaError, match="child hash"):
        media.materialize(plans, output_root=output)
    assert len((output / "journal.jsonl").read_text().splitlines()) == 1


def test_direct_parent_mutation_after_start_never_rebinds_catalog(tmp_path, monkeypatch):
    parent = tmp_path / "parent.mp4"; parent.write_bytes(b"admitted parent")
    geometry = media.Geometry(30.0, 12, 8, 16)
    monkeypatch.setattr(media, "_probe", lambda path: geometry)
    monkeypatch.setattr(media, "_free_bytes", lambda path: media.MIN_FREE_BYTES)
    plans = [media.MediaPlan("ucf", f"ucf-crime/videos/test/P{i}.mp4", "P", parent, None, None)
             for i in range(1369)]
    original = media._record
    changed = False
    def mutate_after_start(journal, event):
        nonlocal changed
        original(journal, event)
        if event["status"] == "started" and not changed:
            parent.write_bytes(b"replacement parent"); changed = True
    monkeypatch.setattr(media, "_record", mutate_after_start)
    with pytest.raises(media.VAUMediaError, match="parent changed"):
        media.materialize(plans, output_root=tmp_path / "out")


def test_full_decode_rejects_a_truncated_actual_child(tmp_path):
    cv2 = pytest.importorskip("cv2")
    pytest.importorskip("decord")
    import numpy as np
    parent = tmp_path / "parent.mp4"
    writer = cv2.VideoWriter(str(parent), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (32, 32))
    assert writer.isOpened()
    for index in range(8):
        writer.write(np.full((32, 32, 3), index * 20, dtype=np.uint8))
    writer.release()
    import decord
    child = tmp_path / "child.mp4"
    media._split_official_chunked(decord.VideoReader(str(parent)), (0.2, 1.0), child)
    expected = media.Geometry(5.0, 4, 32, 32)
    assert media._verify_child_decode(child, expected) == expected
    child.write_bytes(child.read_bytes()[:child.stat().st_size // 2])
    with pytest.raises(media.VAUMediaError):
        media._verify_child_decode(child, expected)


def test_plan_admission_requires_hash_root_and_50_gib_cap(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    plan = project / "plan.json"
    document = {"schema": "nc_rted_vau_materialization_plan/v1", "parents": [], "media": [], "summary": {}}
    plan.write_text(json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n")
    digest = hashlib.sha256(plan.read_bytes()).hexdigest()
    with pytest.raises(media.VAUMediaError, match="SHA-256"):
        media.validate_plan(plan, "0" * 64, document, project_root=project, output_root=project / "output", max_artifact_bytes=1)
    with pytest.raises(media.VAUMediaError, match="escapes"):
        media.validate_plan(plan, digest, document, project_root=project, output_root=tmp_path / "outside", max_artifact_bytes=1)
    with pytest.raises(media.VAUMediaError, match="differs"):
        media.validate_plan(plan, digest, {**document, "summary": {"different": True}}, project_root=project,
                            output_root=project / "output", max_artifact_bytes=1)
    with pytest.raises(media.VAUMediaError, match="50 GiB"):
        media.validate_plan(plan, digest, document, project_root=project, output_root=project / "output", max_artifact_bytes=50 * 1024**3 + 1)
