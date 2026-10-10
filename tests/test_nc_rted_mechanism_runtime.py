import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from nc_rted.detection_provider import DetectionProtocol, FrozenDetectionProvider
from nc_rted.mechanism_runtime import (DevelopmentPrefixReader, MechanismRuntimeError,
                                       bind_development_inputs, run_checkpoint_diagnostics)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, document: dict) -> str:
    path.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")
    return _sha(path)


def _bindings(tmp_path: Path):
    source_refs = {}
    for name in ("source_splits", "ucf_database", "xd_database"):
        path = tmp_path / f"{name}.json"
        contents = [] if name == "source_splits" else {"name": name}
        source_refs[name] = {"path": str(path), "sha256": _write(path, contents)}
    development_path = tmp_path / "development.json"
    development_sha = _write(development_path, {
        "schema": "nc_rted_mechanism_development_prefixes/v1", "inputs": source_refs,
        "records": [{"sample_id": "development:ucf-crime:source:0", "dataset": "ucf-crime", "key": "source",
                     "family": "source", "query_index": 0, "observed_seconds": .75,
                     "class": "normal", "scope": "vad_causal_latest_8s"}],
    })
    media_path = tmp_path / "source.mp4"
    media_path.write_bytes(b"test media")
    media_sha = _sha(media_path)
    runtime_inputs = tmp_path / "runtime-inputs.json"
    runtime_sha = _write(runtime_inputs, {
        "schema": "nc_rted_mechanism_development_runtime_inputs/v1",
        "scope": "development allocation only; causal prefixes only",
        "inputs": {"development_manifest": {"path": str(development_path), "sha256": development_sha}},
        "records": [{"sample_id": "development:ucf-crime:source:0", "dataset": "ucf-crime", "key": "source",
                     "family": "source", "query_index": 0, "observed_seconds": .75,
                     "class": "normal", "scope": "vad_causal_latest_8s", "media_path": str(media_path),
                     "media_sha256": media_sha}],
    })
    identity = {"checkpoint": "stage1", "implementation": "exact"}
    fast_path = tmp_path / "fast.json"
    fast_sha = _write(fast_path, {
        "schema": "nc_rted_development_fast/v1", "fast_identity": identity,
        "media": [{"dataset": "ucf-crime", "media_key": "source", "media_path": str(media_path),
                   "media_sha256": media_sha, "fps": 4, "frame_count": 8, "height": 8, "width": 8,
                   "target_fps": 4, "query_interval": 4, "max_query_index": 0,
                   "queries": [{"index": 0, "frame_indices": [0, 1, 2, 3], "fast_score": .1}]}],
    })
    manifest = tmp_path / "runtime.json"
    manifest_sha = _write(manifest, {
        "fast": {"identity": identity}, "run": {"group": "F", "seed": 42,
                                               "checkpoint_root": str(tmp_path / "checkpoint-root")},
        "inherited": {"external_root": "/bound/reactvau", "source_manifest": "/bound/sources.json",
                      "source_manifest_sha256": "f" * 64, "export_hashes": {"non_lora_trainables.bin": "e" * 64}},
        "teacher": {"artifact": "/bound/teacher", "sha256": "a" * 64},
        "hashes": {"code_sha256": "a" * 64, "runtime_sha256": "b" * 64,
                   "inherited_weights_sha256": "c" * 64},
        "detector": {"snapshot_sha256": "d" * 64, "siglip_snapshot_sha256": "e" * 64,
                     "final_stage2_siglip_snapshot_sha256": "f" * 64},
    })
    prediction = tmp_path / "prediction-runtime.json"
    prediction_sha = _write(prediction, {"protocols": {"vad_config": {"yes_token_ids": [11, 12], "no_token_ids": [13]}},
                                         "inherited": json.loads(manifest.read_text())["inherited"]})
    return SimpleNamespace(runtime_inputs=runtime_inputs, runtime_sha=runtime_sha, fast_path=fast_path,
                           fast_sha=fast_sha, manifest=manifest, manifest_sha=manifest_sha,
                           prediction=prediction, prediction_sha=prediction_sha)


def _prediction_runtime(files):
    document = json.loads(files.prediction.read_text())
    document["inherited"]["stage2_export_hashes"] = document["inherited"].pop("export_hashes")
    return SimpleNamespace(document=document)


def _assembled_runtime(tmp_path: Path):
    identity = {"run_id": "formal:42:F", "group": "F", "seed": "42"}
    final = tmp_path / "checkpoint-root" / "final"
    final.mkdir(parents=True)

    class Store:
        def __init__(self):
            self.identity = identity
            self.restored = None

        def latest(self):
            return final

    class Reader:
        def __call__(self, *args):
            raise AssertionError("the synthetic suite must not decode media")

        @staticmethod
        def encode(frames):
            return frames

    detection = FrozenDetectionProvider(object(), object(), {"ucf-crime": object()}, reader=Reader(),
                                        observation_reader=lambda *args: None)

    def provider(sample_id):
        return detection

    class Bridge:
        def __init__(self):
            self.evaluated = False

        def eval(self):
            self.evaluated = True
            return self

    worker = SimpleNamespace(store=Store(), trainer=SimpleNamespace(recipe=SimpleNamespace(updates=1000)),
                             bridge=Bridge(), tokenizer=object(), provider=provider)
    return SimpleNamespace(worker=worker), detection


def _suite(output, **kwargs):
    assert kwargs["reader"].snapshot_sha256
    rows = []
    for name in ("baseline", "branch_disable", "time_permute", "relation_permute",
                 "time_permute_random", "relation_permute_random"):
        rows.append({"sample_id": "development:ucf-crime:source:0", "intervention": name,
                     "original_task_loss": .5, "detection_probability": .25,
                     "fixed_greedy_token_ids": [1, 2], "fixed_greedy_text": "No", "no_candidate": False,
                     "token_delta_frobenius_norm": 0.0 if name in {"baseline", "branch_disable"} else 1.0})
    return {"records": rows}


def test_runner_binds_real_shaped_inputs_loads_final_trainables_and_writes_six_variant_report(tmp_path):
    files = _bindings(tmp_path)
    runtime, _ = _assembled_runtime(tmp_path)
    manifest = SimpleNamespace(document=json.loads(files.manifest.read_text()),
                               run=json.loads(files.manifest.read_text())["run"])

    def assembled(value, *, checkpoint):
        assert value is manifest
        assert checkpoint.name == "final"
        return runtime, {"identity": runtime.worker.store.identity, "final": True,
                         "completed_updates": 1000, "payload_sha256": "9" * 64}, {"trainable": 3}

    report = run_checkpoint_diagnostics(
        tmp_path / "report.json", runtime_manifest=files.manifest, runtime_manifest_sha256=files.manifest_sha,
        runtime_inputs=files.runtime_inputs, runtime_inputs_sha256=files.runtime_sha,
        fast_snapshot=files.fast_path, fast_snapshot_sha256=files.fast_sha,
        prediction_runtime_manifest=files.prediction, prediction_runtime_manifest_sha256=files.prediction_sha,
        generation_config={"do_sample": False, "num_beams": 1, "max_new_tokens": 8},
        load_runtime=lambda *args, **kwargs: manifest, load_prediction_runtime=lambda *args, **kwargs: _prediction_runtime(files),
        assemble_runtime=assembled, suite_runner=_suite, teacher_summary_runner=lambda _: {"records": 1},
    )
    assert runtime.worker.bridge.evaluated
    assert len(report["records"]) == 6
    assert report["records"][0]["mechanism_noop_status"] == "BASELINE"
    assert report["records"][1]["mechanism_noop_status"] == "BRANCH_DISABLED"
    assert report["model_identity"]["inherited_weights_sha256"] == "c" * 64
    assert report["geometry_invariance"] == [{"sample_id": "development:ucf-crime:source:0",
                                                "status": "RAW_OBSERVER_INPUTS_NOT_CAPTURED"}]
    assert json.loads((tmp_path / "report.json").read_text())["checkpoint_runtime"]["checkpoint_receipt"]["final"] is True


def test_bad_development_fast_binding_fails_before_model_assembly(tmp_path):
    files = _bindings(tmp_path)
    fast = json.loads(files.fast_path.read_text())
    fast["media"][0]["media_sha256"] = "0" * 64
    fast_sha = _write(files.fast_path, fast)
    assembled = False

    def unexpected_assembly(manifest):
        nonlocal assembled
        assembled = True
        raise AssertionError("bad bindings must fail before model assembly")

    with pytest.raises(MechanismRuntimeError, match="development Fast snapshot"):
        run_checkpoint_diagnostics(
            tmp_path / "report.json", runtime_manifest=files.manifest, runtime_manifest_sha256=files.manifest_sha,
            runtime_inputs=files.runtime_inputs, runtime_inputs_sha256=files.runtime_sha,
            fast_snapshot=files.fast_path, fast_snapshot_sha256=fast_sha,
            prediction_runtime_manifest=files.prediction, prediction_runtime_manifest_sha256=files.prediction_sha,
            generation_config={"do_sample": False, "num_beams": 1, "max_new_tokens": 8},
            load_runtime=lambda *args, **kwargs: SimpleNamespace(document=json.loads(files.manifest.read_text())),
            load_prediction_runtime=lambda *args, **kwargs: _prediction_runtime(files),
            assemble_runtime=unexpected_assembly, suite_runner=_suite, teacher_summary_runner=lambda _: {"records": 1},
        )
    assert not assembled


@pytest.mark.parametrize(("field", "value", "message"), [
    ("class", "anomalous", "immutable fields differ"),
    ("observed_seconds", 1.0, "immutable fields differ"),
])
def test_runtime_inputs_reject_altered_sealed_development_fields(tmp_path, field, value, message):
    files = _bindings(tmp_path)
    document = json.loads(files.runtime_inputs.read_text())
    document["records"][0][field] = value
    runtime_sha = _write(files.runtime_inputs, document)
    with pytest.raises(MechanismRuntimeError, match=message):
        bind_development_inputs(runtime_inputs=files.runtime_inputs, runtime_inputs_sha256=runtime_sha,
                                fast_snapshot=files.fast_path, fast_snapshot_sha256=files.fast_sha,
                                expected_fast_identity=json.loads(files.fast_path.read_text())["fast_identity"])


def test_runtime_inputs_reject_subset_of_sealed_development_prefixes(tmp_path):
    files = _bindings(tmp_path)
    development = json.loads((tmp_path / "development.json").read_text())
    extra = dict(development["records"][0])
    extra["sample_id"] = "development:ucf-crime:source:1"
    extra["query_index"] = 1
    extra["observed_seconds"] = 1.75
    development["records"].append(extra)
    development_sha = _write(tmp_path / "development.json", development)
    runtime_inputs = json.loads(files.runtime_inputs.read_text())
    runtime_inputs["inputs"]["development_manifest"]["sha256"] = development_sha
    runtime_sha = _write(files.runtime_inputs, runtime_inputs)
    with pytest.raises(MechanismRuntimeError, match="omit sealed development prefixes"):
        bind_development_inputs(runtime_inputs=files.runtime_inputs, runtime_inputs_sha256=runtime_sha,
                                fast_snapshot=files.fast_path, fast_snapshot_sha256=files.fast_sha,
                                expected_fast_identity=json.loads(files.fast_path.read_text())["fast_identity"])


def test_suite_rejects_missing_sealed_development_prefix(tmp_path):
    files = _bindings(tmp_path)
    runtime, _ = _assembled_runtime(tmp_path)
    manifest_document = json.loads(files.manifest.read_text())
    manifest = SimpleNamespace(document=manifest_document, run=manifest_document["run"])

    def incomplete(output, **kwargs):
        return {"records": _suite(output, **kwargs)["records"][:-1]}

    def assembled(value, *, checkpoint):
        return runtime, {"identity": runtime.worker.store.identity, "final": True,
                         "completed_updates": 1000}, {"trainable": 3}

    with pytest.raises(MechanismRuntimeError, match="variants differ"):
        run_checkpoint_diagnostics(
            tmp_path / "report.json", runtime_manifest=files.manifest, runtime_manifest_sha256=files.manifest_sha,
            runtime_inputs=files.runtime_inputs, runtime_inputs_sha256=files.runtime_sha,
            fast_snapshot=files.fast_path, fast_snapshot_sha256=files.fast_sha,
            prediction_runtime_manifest=files.prediction, prediction_runtime_manifest_sha256=files.prediction_sha,
            generation_config={"do_sample": False, "num_beams": 1, "max_new_tokens": 8},
            load_runtime=lambda *args, **kwargs: manifest, load_prediction_runtime=lambda *args, **kwargs: _prediction_runtime(files),
            assemble_runtime=assembled, suite_runner=incomplete, teacher_summary_runner=lambda _: {"records": 1})


def test_development_reader_rejects_future_or_noncausal_bounded_queries_before_decode(tmp_path):
    files = _bindings(tmp_path)
    identity = json.loads(files.fast_path.read_text())["fast_identity"]
    protocol = DetectionProtocol("Is there violence?", "default", "short_online_v2", False, False, .5, .5)
    reader = DevelopmentPrefixReader(files.fast_path, snapshot_sha256=files.fast_sha, fast_identity=identity,
                                     protocols={"ucf-crime": protocol}, encode=lambda frames: frames)
    with pytest.raises(MechanismRuntimeError, match="sealed causal maximum"):
        reader("ucf-crime", "source", 1)
    document = json.loads(files.fast_path.read_text())
    document["media"][0]["queries"][0]["frame_indices"] = [1, 2, 3, 4]
    bad_sha = _write(files.fast_path, document)
    reader = DevelopmentPrefixReader(files.fast_path, snapshot_sha256=bad_sha, fast_identity=identity,
                                     protocols={"ucf-crime": protocol}, encode=lambda frames: frames)
    with pytest.raises(MechanismRuntimeError, match="query grid"):
        reader("ucf-crime", "source", 0)
