import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("segmented_materialization", ROOT / "scripts" / "nc_rted_segmented_materialization.py")
materialization = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(materialization)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(materialization.canonical(value))
    return {"path": str(path), "sha256": _sha(path)}


def _runtime(group):
    scientific = {"group": "shared", "seed": "17", "data_sha256": "data", "teacher_sha256": "teacher",
                  "inherited_weights_sha256": "weights"}
    return {"run": {"run_id": f"qualified-{group}", "checkpoint_root": "/old/checkpoints",
                     "progress_path": "/old/progress"},
            "hashes": {"code_sha256": "old-code", "runtime_sha256": "old-runtime"},
            "catalog": {"manifest_directory": "/catalog", "training_annotations": "/annotations",
                        "provenance_sha256": "provenance"}, "identity": scientific}


def _install_identity_stubs(monkeypatch):
    production = types.ModuleType("nc_rted.production_runtime")
    production.load_manifest = lambda path, expected_sha256: types.SimpleNamespace(document=json.loads(Path(path).read_text()))
    production._checkpoint_identity = lambda manifest, _catalog, frozen_sample_consumer_sha256=None: {
        **manifest.document["identity"], "run_id": manifest.document["run"]["run_id"],
        "code_sha256": manifest.document["hashes"]["code_sha256"], "config_sha256": _shaifest(manifest),
        "runtime_sha256": manifest.document["hashes"]["runtime_sha256"],
        **({"frozen_sample_consumer_sha256": frozen_sample_consumer_sha256}
           if frozen_sample_consumer_sha256 is not None else {}),
    }
    task_inputs = types.ModuleType("nc_rted.task_inputs")
    task_inputs.TrainingCatalog = types.SimpleNamespace(load=lambda *args, **kwargs: types.SimpleNamespace(identity="catalog"))
    monkeypatch.setitem(sys.modules, "nc_rted.production_runtime", production)
    monkeypatch.setitem(sys.modules, "nc_rted.task_inputs", task_inputs)


def _shaifest(manifest):
    return hashlib.sha256(json.dumps(manifest.document, sort_keys=True).encode()).hexdigest()


def _fixture(tmp_path):
    source = tmp_path / "successor"
    qualified_files = {}
    files = {
        "scripts/nc_rted_interleaved_formal.py": ("old runner", "new runner"),
        "src/nc_rted/queue.py": ("old queue", "new queue"),
        "src/nc_rted/resource_attestation.py": ("old attestation", "new attestation"),
        "src/nc_rted/formal_bundle.py": ("same bundle", "same bundle"),
        "src/nc_rted/unchanged.py": ("same", "same"),
    }
    for name, (old, new) in files.items():
        target = source / name; target.parent.mkdir(parents=True, exist_ok=True); target.write_text(new)
        qualified_files[name] = hashlib.sha256(old.encode()).hexdigest()
    queue_script = source / "scripts/nc_rted_queue.py"; queue_script.write_text("new queue script")
    qualified = {"schema": "nc_rted_interleaved_source_manifest/v1", "files": qualified_files,
                 "code_sha256": "qualified-code"}
    qualified_path = tmp_path / "qualified-source.json"; qualified_binding = _write_json(qualified_path, qualified)
    diagnostic_path = tmp_path / "diagnostic.json"
    diagnostic_binding = _write_json(diagnostic_path, {"source_manifest": str(qualified_path),
                                                        "source_manifest_sha256": qualified_binding["sha256"]})
    members = {}; identities = {}
    for group in materialization.GROUPS:
        runtime_path = tmp_path / f"{group}.runtime.json"; runtime_binding = _write_json(runtime_path, _runtime(group))
        members[group] = {"runtime": runtime_binding["path"], "runtime_sha256": runtime_binding["sha256"]}
        identities[group] = _runtime(group)["identity"]
    profile_path = tmp_path / "profile.json"
    profile_binding = _write_json(profile_path, {"schema": "nc_rted_interleaved_formal_qualification_profile/v1",
                                                  "diagnostic_bundle": str(diagnostic_path),
                                                  "diagnostic_bundle_sha256": diagnostic_binding["sha256"],
                                                  "members": members, "member_identities": identities})
    qualification_path = tmp_path / "qualification.json"
    qualification_binding = _write_json(qualification_path, {"status": "PASS_GPU_KERNEL_AND_INHERITED_RUNTIME_IMPORTS",
                                                               "qualification_profile": profile_binding,
                                                               "runtime_environment": {"environment": {"PYTHONPATH": "/old"},
                                                                                       "interpreter": {"path": "/python"}}})
    gates_binding = _write_json(tmp_path / "gates.json", {
        "schema": "nc_rted_source39_recovered_engineering_gate_bindings/v1",
        "status": "ENGINEERING_EVIDENCE_ONLY_NOT_RESOURCE_ADMISSION",
        "qualification": qualification_binding, "profile": profile_binding,
        "engineering_checks": {str(number): "PASS" for number in range(1, 11)},
        "engineering_gate_evidence": {str(number): {"path": f"/gate/{number}", "sha256": "0" * 64} for number in range(1, 11)},
    })
    return source, qualification_binding, profile_binding, gates_binding, identities


def test_materialize_successor_preserves_scientific_identities_and_return_contract(tmp_path, monkeypatch):
    _install_identity_stubs(monkeypatch)
    source, qualification, profile, gates, qualified = _fixture(tmp_path)
    output = tmp_path / "output"
    result = materialization.materialize_seed(source_root=source, output_root=output, qualification_binding=qualification,
                                              profile_binding=profile, gates_binding=gates, lease_id="lease", lease_expiry=99)
    assert set(result) == {"bundle", "member_identities", "qualified_member_identities", "source_transition", "member_runs",
                           "source_sha256", "source_root", "qualification", "profile", "gates", "scope", "environment", "interpreter"}
    assert result["qualified_member_identities"] == qualified
    assert result["environment"]["PYTHONPATH"] == str(source / "src")
    assert set(result["member_identities"]) == set(materialization.GROUPS)
    for group in materialization.GROUPS:
        for field in ("group", "seed", "data_sha256", "teacher_sha256", "inherited_weights_sha256"):
            assert result["member_identities"][group][field] == qualified[group][field]
        assert result["member_runs"][group]["checkpoint_root"].startswith(str(output))
    assert set(result["source_transition"]["allowed_changed_files"]) == materialization.ALLOWED_CHANGED
    absolute_sources = {str((source / name).resolve()): value
                        for name, value in json.loads((output / "successor-source-manifest.json").read_text())["files"].items()}
    admitted_source = hashlib.sha256(json.dumps(absolute_sources, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert result["source_sha256"] == admitted_source
    assert result["source_transition"]["admitted_source_root"] == str(source)
    assert result["source_transition"]["admitted_source_files_sha256"] == admitted_source
    assert admitted_source != json.loads((output / "successor-source-manifest.json").read_text())["code_sha256"]
    for group in materialization.GROUPS:
        runtime = json.loads((output / "members" / group / "runtime.json").read_text())
        admission = json.loads((output / "members" / group / "admission.json").read_text())
        assert runtime["hashes"]["code_sha256"] == admitted_source
        assert admission["source_files"] == absolute_sources
    assert Path(result["bundle"]["path"]).is_file()
    with pytest.raises(materialization.MaterializationError, match="output root already exists"):
        materialization.materialize_seed(source_root=source, output_root=output, qualification_binding=qualification,
                                         profile_binding=profile, gates_binding=gates, lease_id="lease", lease_expiry=99)


def test_materialize_rejects_source_closure_extra_change(tmp_path, monkeypatch):
    _install_identity_stubs(monkeypatch)
    source, qualification, profile, gates, _ = _fixture(tmp_path)
    (source / "src/nc_rted/unchanged.py").write_text("changed too")
    with pytest.raises(materialization.MaterializationError, match="exact four"):
        materialization.materialize_seed(source_root=source, output_root=tmp_path / "output", qualification_binding=qualification,
                                         profile_binding=profile, gates_binding=gates, lease_id="lease", lease_expiry=99)


def test_materialize_binds_the_frozen_consumer_and_its_extended_source_closure(tmp_path, monkeypatch):
    _install_identity_stubs(monkeypatch)
    source, qualification, profile, gates, _ = _fixture(tmp_path)
    for name in materialization.FROZEN_CONSUMER_CHANGED - materialization.ALLOWED_CHANGED:
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("consumer successor")
    # formal_bundle is already in the qualified closure, so make its successor
    # byte distinct as well as adding the new consumer-only modules.
    (source / "src/nc_rted/formal_bundle.py").write_text("consumer successor changed")
    frozen_inputs = {key: value for key, value in _runtime("A").items()
                     if key not in {"run", "hashes", "teacher"} | materialization.FROZEN_INPUT_SOURCE_BOOKKEEPING}
    frozen_inputs.update({"runtime_source": {"commit": "old"}, "builder": {"commit": "old"},
                          "caption_cache_producer": {"commit": "old"}})
    implementation = {"frozen_provider_runtime": "a" * 64}
    config = {"schema": "nc_rted_frozen_sample_consumer/v1", "root": str(tmp_path / "producer-store"),
              "identity": {"schema": "nc_rted_frozen_material_producer/v1", "catalog_sha256": "catalog",
                           "frozen_inputs": frozen_inputs, "implementation": implementation},
              "producer_implementation": implementation, "device": "cuda:0", "wait_seconds": 120,
              "poll_seconds": 1, "prefetch_depth": 8}
    binding = _write_json(tmp_path / "consumer.json", config)
    result = materialization.materialize_seed(source_root=source, output_root=tmp_path / "output",
                                              qualification_binding=qualification, profile_binding=profile,
                                              gates_binding=gates, lease_id="lease", lease_expiry=99,
                                              frozen_sample_consumer_binding=binding)
    assert result["frozen_sample_consumer"] == binding
    bundle = json.loads(Path(result["bundle"]["path"]).read_text())
    assert bundle["frozen_sample_consumer"] == binding
    assert all(identity["frozen_sample_consumer_sha256"] == binding["sha256"]
               for identity in result["member_identities"].values())
