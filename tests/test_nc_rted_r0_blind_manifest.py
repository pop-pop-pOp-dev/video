import hashlib
import importlib.util
import json
from contextlib import contextmanager
import multiprocessing
from pathlib import Path
import queue

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "nc_rted_build_r0_blind_manifest.py"
SPEC = importlib.util.spec_from_file_location("r0_blind_manifest_builder", SCRIPT)
builder = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(builder)
CONVERTER = Path(__file__).resolve().parents[1] / "scripts" / "nc_rted_convert_blind_fast_snapshot.py"
CONVERTER_SPEC = importlib.util.spec_from_file_location("blind_fast_converter", CONVERTER)
converter = importlib.util.module_from_spec(CONVERTER_SPEC)
assert CONVERTER_SPEC.loader is not None
CONVERTER_SPEC.loader.exec_module(converter)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _media(path: Path, *, dataset="ucf", key="item") -> dict:
    path.write_bytes(b"immutable media")
    return {"dataset": dataset, "media_key": key, "media_path": str(path), "media_sha256": _digest(path),
            "fps": 4.0, "frame_count": 5, "height": 2, "width": 3, "request_index": None}


def _catalog(path: Path, rows: list[dict]) -> None:
    path.write_text(json.dumps({"schema": builder.MEDIA_SCHEMA, "media": rows}))


def _converter_documents(tmp_path: Path, rows: list[dict], config: dict | None = None) -> tuple[Path, Path, str, Path]:
    catalog = tmp_path / "catalog.json"
    _catalog(catalog, rows)
    config = {"schema": "nc_rted_blind_fast_raw_preparation/v4", "catalog_path": str(catalog), "catalog_sha256": _digest(catalog),
              "selected_sha256": "b" * 64, "source_sha256": {"/source.py": "c" * 64}, "vision_weights_sha256": "d" * 64} if config is None else config
    config["catalog_path"], config["catalog_sha256"] = str(catalog), _digest(catalog)
    config_path = tmp_path / "raw-config.json"
    config_path.write_text(json.dumps(config))
    config_hash = _digest(config_path)
    binding = {"config_sha256": config_hash, "catalog_sha256": config["catalog_sha256"], "selected_sha256": config.get("selected_sha256"),
               "source_sha256": config.get("source_sha256"), "vision_weights_sha256": config.get("vision_weights_sha256")}
    raw_rows = [{key: row[key] for key in ("dataset", "media_key", "media_path", "media_sha256", "fps", "frame_count", "height", "width")}
                | {"sample_interval": 1, "queries": [{"index": 0, "frame_indices": [0], "fast_score": 0.1}] if row["frame_count"] else []} for row in rows]
    raw = tmp_path / "raw.json"
    raw.write_text(json.dumps({"schema": converter.RAW_SCHEMA, "catalog_sha256": config["catalog_sha256"],
                               "scorer_binding_sha256": hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest(), "rows": raw_rows}))
    return raw, catalog, config_hash, config_path


def _ample_output_space(monkeypatch) -> None:
    monkeypatch.setattr(builder.shutil, "disk_usage", lambda _: type("Usage", (), {"free": builder.RESERVE + 1024**2})())


def _competing_bound_writer(output: str, results) -> None:
    try:
        builder._write(Path(output), {"payload": "x"})
        results.put("published")
    except builder.BuildError as error:
        results.put(str(error))


def test_empty_artifact_directory_cannot_bind_a_model(tmp_path):
    with pytest.raises(builder.BuildError, match="empty"):
        builder._tree(tmp_path / "empty")
    (tmp_path / "empty").mkdir()
    with pytest.raises(builder.BuildError, match="empty"):
        builder._tree(tmp_path / "empty")


def test_caption_binding_blocks_when_stage2_export_file_is_absent(tmp_path):
    def artifact(name: str) -> tuple[Path, str]:
        directory = tmp_path / name
        directory.mkdir()
        (directory / "bound").write_bytes(name.encode())
        return directory, builder._tree(directory)

    base, base_hash = artifact("base")
    export, _ = artifact("export")
    tokenizer, tokenizer_hash = artifact("tokenizer")
    derived, derived_hash = artifact("derived")
    raw, raw_hash = artifact("raw")
    detector, detector_hash = artifact("detector")
    source = tmp_path / "source.json"
    source.write_text("{}")
    external = tmp_path / "external"
    external.mkdir()
    caption = {"schema": "nc_rted_production_runtime/v1", "inherited": {
        "external_root": str(external), "base_directory": str(base), "base_directory_sha256": base_hash,
        "export_directory": str(export), "export_hashes": {name: "0" * 64 for name in builder.EXPORT_FILES},
        "tokenizer_directory": str(tokenizer), "tokenizer_sha256": tokenizer_hash,
        "source_manifest": str(source), "source_manifest_sha256": _digest(source),
    }, "detector": {
        "final_stage2_siglip_snapshot": str(derived), "final_stage2_siglip_snapshot_sha256": derived_hash,
        "siglip_snapshot": str(raw), "siglip_snapshot_sha256": raw_hash,
        "snapshot": str(detector), "snapshot_sha256": detector_hash, "score_threshold": 0.3,
    }}
    path = tmp_path / "caption.json"
    path.write_text(json.dumps(caption))
    with pytest.raises(builder.BuildError, match="Stage2 export file"):
        builder._caption_bindings(path)


def test_materializer_bindings_are_joined_losslessly_with_integer_ids(tmp_path):
    media = _media(tmp_path / "media.mp4")
    catalog = tmp_path / "catalog.json"
    _catalog(catalog, [media])
    bindings = tmp_path / "bindings.json"
    bindings.write_text(json.dumps({"schema": builder.VAU_BINDING_SCHEMA, "bindings": [{
        "id": 7, "video": "official/video.mp4", "media_path": media["media_path"], "media_sha256": media["media_sha256"],
    }]}))
    roster = tmp_path / "roster.json"
    roster.write_text(json.dumps([{"id": 7, "prompt": "Describe the clip.", "task": "description", "type": "video", "video": "official/video.mp4"}]))
    rows = builder._vau_rows(bindings, roster, builder._catalog(catalog), expected_count=1, expected_media_count=1)
    assert rows == [{"id": "7", "media_path": media["media_path"], "media_sha256": media["media_sha256"], "question": "Describe the clip."}]


def test_vau_binding_cannot_substitute_missing_catalog_geometry(tmp_path):
    media = _media(tmp_path / "media.mp4")
    catalog = tmp_path / "catalog.json"
    _catalog(catalog, [media])
    bindings = tmp_path / "bindings.json"
    bindings.write_text(json.dumps({"schema": builder.VAU_BINDING_SCHEMA, "bindings": [{
        "id": 1, "video": "official/video.mp4", "media_path": str(tmp_path / "other.mp4"), "media_sha256": "a" * 64,
    }]}))
    roster = tmp_path / "roster.json"
    roster.write_text(json.dumps([{"id": 1, "prompt": "Describe the clip.", "task": "description", "type": "video", "video": "official/video.mp4"}]))
    with pytest.raises(builder.BuildError, match="materialized media catalog"):
        builder._vau_rows(bindings, roster, builder._catalog(catalog), expected_count=1, expected_media_count=1)


def test_fast_snapshot_must_cover_each_inherited_query(tmp_path):
    media = _media(tmp_path / "media.mp4")
    snapshot = tmp_path / "fast.json"
    row = {key: value for key, value in media.items() if key != "request_index"}
    row.update({"target_fps": 4, "query_interval": 4, "queries": [{"index": 0, "frame_indices": [0, 1, 2, 3], "fast_score": 0.1}]})
    snapshot.write_text(json.dumps({"schema": builder.FAST_SCHEMA, "media": [row]}))
    with pytest.raises(builder.BuildError, match="does not cover every VAD query"):
        builder._fast_snapshot(snapshot, [media])


def test_fast_config_binds_vad_sampling_fields_required_by_runtime(tmp_path):
    vad = {"question_template": "Is this anomalous?", "prompt_style": "default", "time_message_style": "none",
           "memory_enhancement": True, "rt_anomaly": True, "trigger_threshold": 0.3, "pool_threshold": 0.2, "scoring": "yesno"}
    config = {"model": "/models/fast", "lora": None, "streamforest_weights": "/models/weights.bin", "attn_implementation": "sdpa",
              "image_size": 384, "vision_feature_layer": -2,
              "protocols": {"vad": {"ucf": vad, "xd": dict(vad)}, "vad_config": {
                  "yes_token_ids": [1], "no_token_ids": [2], "fusion": "replace", "fusion_alpha": 0.5,
                  "online_smooth_alpha": 0.5, "online_smooth_beta": 0.5, "target_fps": 4, "query_interval": 4, "batch_size": 1,
              }, "hivau": {"target_fps": 4, "query_interval": 4, "paligemma_batch_size": 1, "max_new_tokens": 32,
                           "task": "description", "fast_prompt_context": "none"}},
              "generation": {"do_sample": False, "num_beams": 1, "num_return_sequences": 1},
              "cache": {"root": "/cache", "max_bytes": 1}}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    assert builder._fast_config(path)["protocols"]["vad_config"]["batch_size"] == 1
    config["protocols"]["vad_config"]["target_fps"] = 2
    path.write_text(json.dumps(config))
    with pytest.raises(builder.BuildError, match="VAD evaluator"):
        builder._fast_config(path)


def test_prediction_source_manifest_binds_every_python_file_separately(tmp_path, monkeypatch):
    _ample_output_space(monkeypatch)
    root = tmp_path / "reactvau"
    (root / "scripts" / "precompute").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "scripts" / "precompute" / "score.py").write_text("print('score')\n")
    (root / "tests" / "test_runtime.py").write_text("print('test')\n")
    manifest, digest = builder._prediction_source_manifest(tmp_path / "output", str(root))
    document = json.loads(Path(manifest).read_text())
    assert digest == _digest(Path(manifest))
    assert set(document["files"]) == {"scripts/precompute/score.py", "tests/test_runtime.py"}


def test_bound_writer_preserves_existing_output(tmp_path):
    output = tmp_path / "output.json"
    output.write_bytes(b"original")
    with pytest.raises(builder.BuildError, match="already exists"):
        builder._write(output, {"new": True})
    assert output.read_bytes() == b"original"


def test_bound_writer_rejects_insufficient_space(tmp_path, monkeypatch):
    output = tmp_path / "output.json"
    monkeypatch.setattr(builder.shutil, "disk_usage", lambda _: type("Usage", (), {"free": builder.RESERVE})())
    with pytest.raises(builder.BuildError, match="20 GiB reserve"):
        builder._write(output, {"new": True})
    assert not output.exists()


def test_bound_writer_rejects_nested_directory_before_creation_at_reserve_boundary(tmp_path, monkeypatch):
    # Create the reusable coordination inode while sufficient headroom exists.
    with builder.allocation_lock(tmp_path):
        pass
    monkeypatch.setattr(builder.shutil, "disk_usage", lambda _: type("Usage", (), {"free": builder.RESERVE})())
    output = tmp_path / "nested" / "deeper" / "output.json"
    with pytest.raises(builder.BuildError, match="directory creation would violate"):
        builder._write(output, {"new": True})
    assert not output.parent.exists()
    assert not (tmp_path / "nested").exists()


def test_competing_bound_writers_preserve_reserve_under_real_allocation_lock(tmp_path, monkeypatch):
    # Materialize the lock inode before reducing the simulated shared headroom.
    with builder.allocation_lock(tmp_path):
        pass
    first, second = tmp_path / "first.json", tmp_path / "second.json"
    context = multiprocessing.get_context("fork")
    ready, release, results = context.Event(), context.Event(), context.Queue()
    real_link = builder.os.link

    def delayed_first_link(source, target):
        if Path(target) == first:
            ready.set()
            if not release.wait(20):
                raise OSError("first writer was not released")
        return real_link(source, target)

    def shared_headroom(_):
        free = builder.RESERVE + 1024**2 if not first.exists() else builder.RESERVE
        return type("Usage", (), {"free": free})()

    monkeypatch.setattr(builder.os, "link", delayed_first_link)
    monkeypatch.setattr(builder.shutil, "disk_usage", shared_headroom)
    children = [context.Process(target=_competing_bound_writer, args=(str(path), results)) for path in (first, second)]
    try:
        children[0].start()
        assert ready.wait(20), "first writer did not reach publication"
        children[1].start()
        with pytest.raises(queue.Empty):
            results.get(timeout=.3)
        release.set()
        outcomes = [results.get(timeout=20) for _ in children]
        assert outcomes.count("published") == 1
        assert sum("20 GiB reserve" in outcome for outcome in outcomes) == 1
        assert first.exists() and not second.exists()
        for child in children:
            child.join(20)
            assert child.exitcode == 0
    finally:
        release.set()
        for child in children:
            if child.is_alive():
                child.terminate()
            child.join(5)


def test_bound_writer_uses_shared_allocation_lock(tmp_path, monkeypatch):
    _ample_output_space(monkeypatch)
    calls = []

    @contextmanager
    def locked(path):
        calls.append(Path(path))
        yield

    monkeypatch.setattr(builder, "allocation_lock", locked)
    output = tmp_path / "output.json"
    builder._write(output, {"new": True})
    assert calls == [output.parent]


def test_bound_writer_cleans_interrupted_publication(tmp_path, monkeypatch):
    _ample_output_space(monkeypatch)
    output = tmp_path / "output.json"

    def interrupted_link(_source, _target):
        raise OSError("simulated interruption")

    monkeypatch.setattr(builder.os, "link", interrupted_link)
    with pytest.raises(builder.BuildError, match="publication failed"):
        builder._write(output, {"new": True})
    assert not output.exists()
    assert not list(tmp_path.glob(".output.json.*.pending"))


def test_blind_fast_converter_requires_the_raw_config_binding_and_full_grid(tmp_path):
    catalog_rows = []
    for index in range(1051):
        dataset = "ucf" if index < 251 else "xd"
        catalog_rows.append({"dataset": dataset, "media_key": str(index), "media_path": f"/media/{index}.mp4", "media_sha256": f"{index:064x}",
                             "fps": 4.0, "frame_count": 1, "height": 2, "width": 3, "request_index": None})
    raw, catalog, config_hash, config_path = _converter_documents(tmp_path, catalog_rows)
    snapshot = converter._snapshot(raw, catalog, config_path, config_hash)
    assert snapshot["schema"] == converter.SCHEMA and len(snapshot["media"]) == 1051
    raw_document = json.loads(raw.read_text())
    raw_document["rows"][0]["queries"][0]["frame_indices"] = [1]
    raw.write_text(json.dumps(raw_document))
    with pytest.raises(converter.ConversionError, match="query differs"):
        converter._snapshot(raw, catalog, config_path, config_hash)


def test_blind_fast_converter_rejects_duplicate_foreign_and_zero_geometry_catalogs(tmp_path):
    catalog_rows = [{"dataset": "ucf" if index < 251 else "xd", "media_key": str(index), "media_path": f"/media/{index}.mp4", "media_sha256": f"{index:064x}",
                     "fps": 4.0, "frame_count": 1, "height": 2, "width": 3, "request_index": None} for index in range(1051)]
    duplicate_foreign = [dict(row) for row in catalog_rows]
    duplicate_foreign[250] = dict(duplicate_foreign[0])
    duplicate_foreign[-1] = dict(duplicate_foreign[-1], dataset="foreign")
    raw, catalog, config_hash, config_path = _converter_documents(tmp_path, duplicate_foreign)
    with pytest.raises(converter.ConversionError, match="duplicate identity"):
        converter._snapshot(raw, catalog, config_path, config_hash)
    zero_geometry = [dict(row) for row in catalog_rows]
    zero_geometry[0]["frame_count"] = 0
    raw, catalog, config_hash, config_path = _converter_documents(tmp_path, zero_geometry)
    with pytest.raises(converter.ConversionError, match="identity or geometry"):
        converter._snapshot(raw, catalog, config_path, config_hash)


def test_blind_fast_converter_rejects_missing_or_mismatched_scorer_binding(tmp_path):
    rows = [{"dataset": "ucf" if index < 251 else "xd", "media_key": str(index), "media_path": f"/media/{index}.mp4", "media_sha256": f"{index:064x}",
             "fps": 4.0, "frame_count": 1, "height": 2, "width": 3, "request_index": None} for index in range(1051)]
    missing = {"schema": "nc_rted_blind_fast_raw_preparation/v4", "selected_sha256": "b" * 64, "source_sha256": None, "vision_weights_sha256": "d" * 64}
    raw, catalog, config_hash, config_path = _converter_documents(tmp_path, rows, missing)
    with pytest.raises(converter.ConversionError, match="scorer binding"):
        converter._snapshot(raw, catalog, config_path, config_hash)
    raw, catalog, config_hash, config_path = _converter_documents(tmp_path, rows)
    raw_document = json.loads(raw.read_text())
    raw_document["scorer_binding_sha256"] = "0" * 64
    raw.write_text(json.dumps(raw_document))
    with pytest.raises(converter.ConversionError, match="raw cache binding"):
        converter._snapshot(raw, catalog, config_path, config_hash)


def test_missing_inputs_write_a_blocked_report_without_fake_assets(tmp_path, monkeypatch):
    _ample_output_space(monkeypatch)
    output = tmp_path / "report.json"
    code = builder.main(["--caption-runtime", str(tmp_path / "caption.json"), "--vad-catalog", str(tmp_path / "vad.json"),
                         "--fast-snapshot", str(tmp_path / "fast.json"), "--vau-bindings", str(tmp_path / "bindings.json"),
                         "--vau-questions", str(tmp_path / "questions.json"), "--vau-catalog", str(tmp_path / "vau.json"),
                         "--fast-config", str(tmp_path / "config.json"), "--numerics-policy", str(tmp_path / "numerics.txt"),
                         "--output", str(output), "--output-root", str(tmp_path / "predictions")])
    assert code == 0
    report = json.loads(output.read_text())
    assert report["status"] == "BLOCKED"
    assert "fast_snapshot:" in "\n".join(report["missing_dependencies"])
