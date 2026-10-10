from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "nc_rted_build_interleaved_diagnostic_manifests.py"
spec = importlib.util.spec_from_file_location("interleaved_manifest_builder", SCRIPT)
builder = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(builder)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value) -> str:
    if isinstance(value, (dict, list)):
        path.write_text(json.dumps(value), encoding="utf-8")
    else:
        path.write_text(value, encoding="utf-8")
    return digest(path)


def catalog(root: Path) -> tuple[Path, Path, str, Path, str]:
    manifest = root / "catalog"; manifest.mkdir()
    annotations = []
    captions = []
    for index in range(2000):
        row = {"id": index, "video": f"clips/{index}.mp4", "task": "caption", "type": "clip",
               "conversations": [{"from": "human", "value": "<video> describe"}, {"from": "gpt", "value": "answer"}]}
        annotations.append(row)
        captions.append({**row, "dataset": "ucf-crime", "parent_key": "captions"})
    annotation = root / "annotations.json"; annotation_sha = write(annotation, annotations)
    detections = []
    splits = [{"family": "ucf-crime:captions", "allocation": "train"}]
    for dataset in ("ucf-crime", "xd-violence"):
        for label, name in (("normal", "normal"), ("anomalous", "anomalous")):
            family = f"{dataset}:{name}"; splits.append({"family": family, "allocation": "train"})
            for index in range(1500):
                detections.append({"family": family, "dataset": dataset, "class": label, "key": name,
                                   "observed_seconds": 1.0, "query_index": index})
    outputs = {}
    for name, value in (("source_splits.json", splits), ("train8000_captions.json", captions),
                        ("train8000_detection_prefixes.json", detections)):
        outputs[name] = write(manifest / name, value)
    provenance = {"schema": "nc_rted_manifest_provenance/v1", "inputs": {str(annotation.resolve()): annotation_sha},
                  "outputs": outputs}
    provenance_sha = write(manifest / "provenance.json", provenance)
    return manifest, annotation, provenance_sha, manifest / "train8000_captions.json", digest(manifest / "train8000_captions.json")


def inputs(tmp_path: Path) -> tuple[argparse.Namespace, dict]:
    manifest, annotations, provenance_sha, captions, captions_sha = catalog(tmp_path)
    identity = {"checkpoint": {"selection_sha256": "1" * 64}, "implementation": {"module": "2" * 64}}
    def media(dataset: str, key: str, index: int, *, queries=None):
        return {"dataset": dataset, "media_key": key, "media_path": str(tmp_path / f"{dataset}-{index}.mp4"),
                "media_sha256": f"{index:064x}", "fps": 1.0, "frame_count": 2, "height": 2, "width": 2,
                "queries": [] if queries is None else queries}
    used = [media(dataset, name, index, queries=[{"index": query, "frame_indices": [1]} for query in range(1500)])
            for index, (dataset, name) in enumerate((("ucf-crime", "normal"), ("ucf-crime", "anomalous"),
                                                      ("xd-violence", "normal"), ("xd-violence", "anomalous")), 1)]
    fast_media = used + [media("ucf-crime", f"unused-{index}", index + 10) for index in range(2409)]
    fast = tmp_path / "training-fast.json"
    fast_sha = write(fast, {"schema": "nc_rted_frozen_fast/v1", "fast_identity": identity, "media": fast_media})
    fast_report = tmp_path / "training-fast-report.json"
    fast_report_sha = write(fast_report, {"status": "PASS_ALL_FIXED_TRAIN_PREFIX_BINDINGS_CANDIDATE",
                                          "snapshot": str(fast), "snapshot_sha256": fast_sha,
                                          "fixed_prefixes_verified": 6000, "media": 2413,
                                          "fast_identity": identity, "formal_acceptance": False})
    protocol = {"question_template": "Is there an anomaly?", "prompt_style": "skeptical",
                "time_message_style": "short_online_v2", "memory_enhancement": False,
                "rt_anomaly": False, "trigger_threshold": 0.5, "pool_threshold": 0.6, "scoring": "yesno"}
    protocol_source = tmp_path / "r0-blind-protocol-source.json"
    protocol_source_sha = write(protocol_source, {"protocols": {"vad": {"ucf": protocol, "xd": protocol},
                                                                 "vad_config": {"fusion": "adaptive"}},
                                                   "blind_fast_snapshot": "/forbidden/blind_fast_snapshot_v2.json"})
    teacher_rows = [{"window_id": f"detection:{dataset}:{name}:{index}", "dataset": dataset, "aux_valid": False,
                     "rejection": "fixture"}
                    for dataset in ("ucf-crime", "xd-violence") for name in ("normal", "anomalous")
                    for index in range(1500)]
    observation_sha, teacher_config_sha = "3" * 64, "4" * 64
    teacher = tmp_path / "teacher_manifest.json"
    teacher_sha = write(teacher, {"schema": "nc_rted_teacher_pipeline/v1", "rows": teacher_rows,
                                  "provenance": {"input_sha256": observation_sha, "config_sha256": teacher_config_sha}})
    cache = str(tmp_path / "caption-cache")
    source_root = Path("/root/autodl-tmp/lookaway-wm")
    producer_files = {name: digest(source_root / name) for name in builder._CAPTION_PRODUCER_FILES}
    producer = tmp_path / "caption-producer-source.json"
    producer_sha = write(producer, {"schema": "nc_rted_runtime_source_manifest/v1", "files": producer_files})
    caption_media = [media("ucf-crime", f"clips/{index}.mp4", index + 3000) for index in range(2000)]
    caption_media_path = tmp_path / "caption-media.json"; caption_media_sha = write(caption_media_path, {"media": caption_media})
    document = {"schema": "nc_rted_production_runtime/v1", "run": {"mode": "diagnostic"},
                "hashes": {"code_sha256": "1" * 64, "runtime_sha256": "2" * 64},
                "media": {"catalog": str(caption_media_path), "catalog_sha256": caption_media_sha,
                          "caption_observation_cache": {"root": cache}},
                "runtime_source": {"manifest": str(producer), "manifest_sha256": producer_sha},
                "catalog": {"manifest_directory": str(manifest), "training_annotations": str(annotations),
                            "provenance_sha256": provenance_sha, "caption_subset": str(captions),
                            "caption_subset_sha256": captions_sha}}
    caption = tmp_path / "caption-runtime.json"; caption_sha = write(caption, document)
    teacher_status = tmp_path / "teacher-status.json"
    teacher_status_sha = write(teacher_status, {"status": "TEACHER_COMPLETED", "output": str(teacher),
                                                 "teacher_manifest_sha256": teacher_sha, "config_sha256": teacher_config_sha})
    teacher_events = tmp_path / "teacher-events.jsonl"
    teacher_events_sha = write(teacher_events, "\n".join((json.dumps({"status": "TEACHER_RUNNING", "config_sha256": teacher_config_sha,
                                                                          "observation_index_sha256": observation_sha}),
                                                            json.dumps({"status": "TEACHER_COMPLETED", "output": str(teacher),
                                                                        "teacher_manifest_sha256": teacher_sha}))) + "\n")
    caption_status = tmp_path / "caption-supervisor.json"
    caption_status_sha = write(caption_status, {"state": "FULL_PREPARATION_COMPLETED", "config_sha256": caption_sha,
                                                 "formal_execution_allowed": False})
    recipe = ROOT / "configs" / "nc_rted" / "training.yaml"
    return argparse.Namespace(training_fast=str(fast), training_fast_sha256=fast_sha,
                              fast_binding_report=str(fast_report), fast_binding_report_sha256=fast_report_sha,
                              protocol_source=str(protocol_source), protocol_source_sha256=protocol_source_sha,
                              teacher_manifest=str(teacher), teacher_manifest_sha256=teacher_sha,
                              teacher_status=str(teacher_status), teacher_status_sha256=teacher_status_sha,
                              teacher_events=str(teacher_events), teacher_events_sha256=teacher_events_sha,
                              caption_runtime=str(caption), caption_runtime_sha256=caption_sha,
                              caption_supervisor=str(caption_status), caption_supervisor_sha256=caption_status_sha,
                              training_recipe=str(recipe), training_recipe_sha256=digest(recipe),
                              runtime_source_root=str(source_root), output_root=str(tmp_path / "published"), seed=17, device="cuda:0",
                              diagnostic_updates=4, diagnostic_checkpoint_interval=2), document


def test_builder_composes_and_publishes_four_equivalent_diagnostic_manifests(tmp_path, monkeypatch):
    args, document = inputs(tmp_path)
    preflight = []
    monkeypatch.setattr(builder, "preflight", lambda manifest: preflight.append(manifest.document))
    report = builder.build(args)
    output = Path(args.output_root)
    assert report["purpose"] == "diagnostic_only"
    assert set(report["members"]) == {"A", "U", "S", "F"}
    bundle = json.loads((output / "interleaved_bundle.json").read_text())
    assert bundle["diagnostic_updates"] == 4
    assert set(bundle["members"]) == {"A", "U", "S", "F"}
    docs = {group: json.loads((output / "members" / f"{group}.runtime.json").read_text()) for group in "AUSF"}
    assert {doc["run"]["run_id"] for doc in docs.values()} == {f"diagnostic:interleaved:17:{group}" for group in "AUSF"}
    normal = []
    for doc in docs.values():
        clone = json.loads(json.dumps(doc))
        for field in ("run_id", "group", "checkpoint_root", "progress_path"):
            clone["run"].pop(field)
        normal.append(clone)
    assert normal.count(normal[0]) == 4
    source = json.loads((output / "source_manifest.json").read_text())
    assert "scripts/nc_rted_interleaved_gpu_diagnostic.py" in source["files"]
    assert source["code_sha256"] == docs["A"]["hashes"]["code_sha256"]
    assert preflight and preflight[0]["fast"]["identity"] == json.loads(Path(args.training_fast).read_text())["fast_identity"]
    assert preflight[0]["teacher"] == {"artifact": args.teacher_manifest, "sha256": args.teacher_manifest_sha256}
    assert preflight[0]["caption_cache_producer"]["manifest"] == document["runtime_source"]["manifest"]
    assert preflight[0]["runtime_source"]["manifest"] == str(output / "source_manifest.json")
    assert set(preflight[0]["fast"]["protocols"]) == {"ucf-crime", "xd-violence"}
    assert "blind_fast_snapshot" not in json.dumps(preflight[0], sort_keys=True)


def test_builder_rejects_blind_fast_before_creating_output(tmp_path):
    args, _ = inputs(tmp_path)
    blind = tmp_path / "blind_fast_snapshot_v2.json"
    blind_sha = write(blind, {"schema": "nc_rted_frozen_fast/v1", "fast_identity": {"checkpoint": "x", "implementation": "y"}})
    args.training_fast, args.training_fast_sha256 = str(blind), blind_sha
    with pytest.raises(builder.BuildError, match="blind"):
        builder.build(args)
    assert not Path(args.output_root).exists()


def test_builder_rejects_incomplete_caption_before_creating_output(tmp_path):
    args, _ = inputs(tmp_path)
    status = Path(args.caption_supervisor)
    args.caption_supervisor_sha256 = write(status, {"state": "STOPPED_ON_FAILURE", "config_sha256": args.caption_runtime_sha256,
                                                     "formal_execution_allowed": False})
    with pytest.raises(builder.BuildError, match="caption supervisor is not complete"):
        builder.build(args)
    assert not Path(args.output_root).exists()


def test_builder_rejects_incomplete_teacher_before_creating_output(tmp_path):
    args, _ = inputs(tmp_path)
    args.teacher_status_sha256 = write(Path(args.teacher_status), {"status": "WAITING_COMPLETE_OBSERVATION_SEAL"})
    with pytest.raises(builder.BuildError, match="teacher supervisor is not complete"):
        builder.build(args)
    assert not Path(args.output_root).exists()


def test_builder_rejects_protocol_source_hash_mismatch_before_creating_output(tmp_path):
    args, _ = inputs(tmp_path)
    args.protocol_source_sha256 = "0" * 64
    with pytest.raises(builder.BuildError, match="inherited protocol source is absent"):
        builder.build(args)
    assert not Path(args.output_root).exists()


def test_builder_preserves_no_output_when_composed_preflight_fails(tmp_path, monkeypatch):
    args, _ = inputs(tmp_path)
    def reject_preflight(_manifest):
        raise builder.ProductionRuntimeError("missing asset")

    monkeypatch.setattr(builder, "preflight", reject_preflight)
    with pytest.raises(builder.BuildError, match="composed runtime failed production preflight"):
        builder.build(args)
    assert not Path(args.output_root).exists()


def test_builder_rejects_runtime_that_changes_caption_cache_producer(tmp_path):
    args, _ = inputs(tmp_path)
    caption = Path(args.caption_runtime); caption_doc = json.loads(caption.read_text())
    source = Path(caption_doc["runtime_source"]["manifest"])
    producer = json.loads(source.read_text())
    producer["files"]["src/nc_rted/observation.py"] = "0" * 64
    caption_doc["runtime_source"]["manifest_sha256"] = write(source, producer)
    args.caption_runtime_sha256 = write(caption, caption_doc)
    args.caption_supervisor_sha256 = write(Path(args.caption_supervisor), {
        "state": "FULL_PREPARATION_COMPLETED", "config_sha256": args.caption_runtime_sha256,
        "formal_execution_allowed": False})
    with pytest.raises(builder.BuildError, match="caption-cache producer identity"):
        builder.build(args)
    assert not Path(args.output_root).exists()


def test_builder_rejects_tampered_caption_catalog_before_rebinding(tmp_path):
    args, document = inputs(tmp_path)
    media = Path(document["media"]["catalog"])
    catalog_document = json.loads(media.read_text())
    catalog_document["media"][0]["width"] = 3
    write(media, catalog_document)
    with pytest.raises(builder.BuildError, match="caption runtime media catalog is absent"):
        builder.build(args)
    assert not Path(args.output_root).exists()


def test_builder_rejects_missing_detection_media_before_publication(tmp_path, monkeypatch):
    args, _ = inputs(tmp_path)
    fast = Path(args.training_fast)
    fast_document = json.loads(fast.read_text())
    fast_document["media"][0]["media_key"] = "not-the-selected-normal-key"
    args.training_fast_sha256 = write(fast, fast_document)
    report = Path(args.fast_binding_report)
    report_document = json.loads(report.read_text())
    report_document["snapshot_sha256"] = args.training_fast_sha256
    args.fast_binding_report_sha256 = write(report, report_document)
    monkeypatch.setattr(builder, "preflight", lambda _manifest: None)
    with pytest.raises(builder.BuildError, match="composed runtime failed production preflight"):
        builder.build(args)
    assert not Path(args.output_root).exists()


def test_builder_rejects_incomplete_teacher_artifact_before_publication(tmp_path):
    args, _ = inputs(tmp_path)
    teacher = Path(args.teacher_manifest)
    manifest = json.loads(teacher.read_text())
    manifest["rows"].pop()
    args.teacher_manifest_sha256 = write(teacher, manifest)
    status = Path(args.teacher_status)
    status_document = json.loads(status.read_text())
    status_document["teacher_manifest_sha256"] = args.teacher_manifest_sha256
    args.teacher_status_sha256 = write(status, status_document)
    events = Path(args.teacher_events)
    rows = [json.loads(line) for line in events.read_text().splitlines()]
    rows[-1]["teacher_manifest_sha256"] = args.teacher_manifest_sha256
    args.teacher_events_sha256 = write(events, "\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(builder.BuildError, match="teacher manifest is incompatible"):
        builder.build(args)
    assert not Path(args.output_root).exists()


@pytest.mark.parametrize("field,value", [("input_sha256", "missing"), ("input_sha256", None), ("input_sha256", "not-a-digest"),
                                           ("config_sha256", None), ("config_sha256", "not-a-digest")])
def test_builder_rejects_null_or_malformed_teacher_provenance(tmp_path, field, value):
    args, _ = inputs(tmp_path)
    teacher = Path(args.teacher_manifest)
    manifest = json.loads(teacher.read_text())
    if value == "missing":
        manifest["provenance"].pop(field)
    else:
        manifest["provenance"][field] = value
    args.teacher_manifest_sha256 = write(teacher, manifest)
    status = Path(args.teacher_status)
    status_document = json.loads(status.read_text())
    status_document["teacher_manifest_sha256"] = args.teacher_manifest_sha256
    args.teacher_status_sha256 = write(status, status_document)
    events = Path(args.teacher_events)
    rows = [json.loads(line) for line in events.read_text().splitlines()]
    rows[-1]["teacher_manifest_sha256"] = args.teacher_manifest_sha256
    args.teacher_events_sha256 = write(events, "\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(builder.BuildError, match="(lacks out-of-source|provenance need SHA-256)"):
        builder.build(args)
    assert not Path(args.output_root).exists()


def test_builder_rejects_teacher_with_incompatible_schema_via_consumer(tmp_path):
    args, _ = inputs(tmp_path)
    teacher = Path(args.teacher_manifest)
    manifest = json.loads(teacher.read_text())
    manifest["schema"] = "nc_rted_teacher_pipeline/v999"
    args.teacher_manifest_sha256 = write(teacher, manifest)
    status = Path(args.teacher_status)
    status_document = json.loads(status.read_text())
    status_document["teacher_manifest_sha256"] = args.teacher_manifest_sha256
    args.teacher_status_sha256 = write(status, status_document)
    events = Path(args.teacher_events)
    rows = [json.loads(line) for line in events.read_text().splitlines()]
    rows[-1]["teacher_manifest_sha256"] = args.teacher_manifest_sha256
    args.teacher_events_sha256 = write(events, "\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(builder.BuildError, match="teacher manifest is incompatible"):
        builder.build(args)
    assert not Path(args.output_root).exists()


def test_builder_rejects_near_reserve_before_staging_allocation(tmp_path, monkeypatch):
    args, _ = inputs(tmp_path)
    (tmp_path / ".nc_rted_interleaved_staging").mkdir()
    monkeypatch.setattr(builder.shutil, "disk_usage", lambda _path: SimpleNamespace(
        free=builder._RESERVE_BYTES + 4096))
    with pytest.raises(builder.BuildError, match="reserve-plus-budget"):
        builder.build(args)
    assert not Path(args.output_root).exists()


def test_builder_recovers_identical_publication_after_post_publish_sync_failure(tmp_path, monkeypatch):
    args, _ = inputs(tmp_path)
    monkeypatch.setattr(builder, "preflight", lambda _manifest: None)
    original = builder._fsync_directory

    def fail_parent(path):
        if path == tmp_path:
            raise OSError("injected post-publication sync failure")
        original(path)

    monkeypatch.setattr(builder, "_fsync_directory", fail_parent)
    with pytest.raises(OSError, match="post-publication"):
        builder.build(args)
    assert Path(args.output_root).is_dir()
    monkeypatch.setattr(builder, "_fsync_directory", original)
    monkeypatch.setattr(builder.shutil, "disk_usage", lambda _path: SimpleNamespace(
        free=builder._RESERVE_BYTES + 4096))
    retry = builder.build(args)
    assert retry["bundle_manifest"] == str(Path(args.output_root) / "interleaved_bundle.json")


def test_builder_recovery_rejects_published_preflight_failure_without_allocation(tmp_path, monkeypatch):
    args, _ = inputs(tmp_path)
    monkeypatch.setattr(builder, "preflight", lambda _manifest: None)
    builder.build(args)
    monkeypatch.setattr(builder, "preflight", lambda _manifest: (_ for _ in ()).throw(
        builder.ProductionRuntimeError("external dependency unavailable")))
    monkeypatch.setattr(builder.shutil, "disk_usage", lambda _path: SimpleNamespace(
        free=builder._RESERVE_BYTES + 4096))
    with pytest.raises(builder.BuildError, match="published runtime failed production preflight"):
        builder.build(args)
    assert Path(args.output_root).is_dir()
    staging = tmp_path / ".nc_rted_interleaved_staging"
    assert not list(staging.glob("interleaved-manifests-*"))


def test_builder_retries_final_staging_sync_before_return(tmp_path, monkeypatch):
    args, _ = inputs(tmp_path)
    monkeypatch.setattr(builder, "preflight", lambda _manifest: None)
    original = builder._fsync_directory
    staging = tmp_path / ".nc_rted_interleaved_staging"
    staging_syncs = 0

    def fail_final_staging(path):
        nonlocal staging_syncs
        if path == staging:
            staging_syncs += 1
            if staging_syncs == 2:
                raise OSError("injected final staging sync failure")
        original(path)

    monkeypatch.setattr(builder, "_fsync_directory", fail_final_staging)
    with pytest.raises(OSError, match="final staging"):
        builder.build(args)
    assert Path(args.output_root).is_dir()
    calls = []

    def record_sync(path):
        calls.append(path)
        original(path)

    monkeypatch.setattr(builder, "_fsync_directory", record_sync)
    builder.build(args)
    assert calls == [Path(args.output_root), tmp_path, staging]


def test_builder_preserves_existing_different_destination(tmp_path, monkeypatch):
    args, _ = inputs(tmp_path)
    monkeypatch.setattr(builder, "preflight", lambda _manifest: None)
    builder.build(args)
    args.seed = 42
    with pytest.raises(builder.BuildError, match="different artifacts"):
        builder.build(args)
