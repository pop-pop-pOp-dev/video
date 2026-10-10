import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("formal_qualification", ROOT / "scripts" / "nc_rted_qualify_interleaved_formal_bundle.py")
qualification = importlib.util.module_from_spec(spec); spec.loader.exec_module(qualification)
diagnostic_spec = importlib.util.spec_from_file_location("interleaved_diagnostic", ROOT / "scripts" / "nc_rted_interleaved_gpu_diagnostic.py")
diagnostic = importlib.util.module_from_spec(diagnostic_spec); diagnostic_spec.loader.exec_module(diagnostic)


def _sha(path):
    return qualification._sha(Path(path))


def _binding(path):
    return {"path": str(path), "sha256": _sha(path)}


def _longest_reports(tmp_path):
    reports = {}
    for name, status in (("caption", "PASS_REAL_LONGEST_CAPTION_PROVIDER_FORWARD_BACKWARD"),
                         ("detection_xd", "PASS_REAL_LONGEST_DETECTION_PREFIX_FORWARD_BACKWARD"),
                         ("detection_ucf", "PASS_REAL_LONGEST_DETECTION_PREFIX_FORWARD_BACKWARD")):
        report = tmp_path / f"{name}.json"; report.write_text(json.dumps({"status": status, "selected": {"id": name}}))
        reports[name] = _binding(report)
    return reports


def test_prepare_creates_non_admitted_formal_profiles_from_diagnostic_members(tmp_path, monkeypatch):
    members = {}
    for group in "AUSF":
        path = tmp_path / f"{group}.json"
        path.write_text(json.dumps({"run": {"run_id": f"diagnostic:interleaved:17:{group}", "group": group, "seed": 17, "mode": "diagnostic", "diagnostic_updates": 4, "checkpoint_root": "/old/checkpoints", "progress_path": "/old/progress"}}))
        members[group] = {"manifest": str(path), "sha256": qualification._sha(path)}
    source = tmp_path / "source.json"; source.write_text("{}")
    bundle = {"schema": "nc_rted_interleaved_gpu_harness/v1", "diagnostic_updates": 4, "diagnostic_checkpoint_interval": 4, "bundle_checkpoint_root": "/old/bundle", "members": members, "source_manifest": str(source), "source_manifest_sha256": qualification._sha(source)}
    bundle_path = tmp_path / "bundle.json"; bundle_path.write_text(json.dumps(bundle))
    def fake_load(path, *, expected_sha256):
        return SimpleNamespace(document=json.loads(Path(path).read_text()))
    monkeypatch.setattr(qualification, "load_manifest", fake_load)
    monkeypatch.setattr(qualification, "formal_runtime_identity", lambda manifest: {"group": manifest.document["run"]["group"], "code_sha256": "a" * 64})
    monkeypatch.setattr(qualification, "_verify_common_source_closure", lambda bundle, source: None)
    monkeypatch.setattr(qualification, "_source_files", lambda *args: {})
    reports = _longest_reports(tmp_path)
    candidates = tmp_path / "caption-candidates.json"; candidates.write_text("{}")
    coverage = {"coverage": "derived"}
    monkeypatch.setattr(qualification, "_complete_long_input_coverage", lambda *args: coverage)
    args = SimpleNamespace(diagnostic_bundle=str(bundle_path), diagnostic_bundle_sha256=qualification._sha(bundle_path),
                           caption_longest_report=reports["caption"]["path"], caption_longest_report_sha256=reports["caption"]["sha256"],
                           caption_longest_source_root=str(tmp_path),
                           caption_longest_candidates=str(candidates), caption_longest_candidates_sha256=_sha(candidates),
                           detection_xd_longest_report=reports["detection_xd"]["path"], detection_xd_longest_report_sha256=reports["detection_xd"]["sha256"],
                           detection_xd_longest_source_root=str(tmp_path),
                           detection_ucf_longest_report=reports["detection_ucf"]["path"], detection_ucf_longest_report_sha256=reports["detection_ucf"]["sha256"],
                           detection_ucf_longest_source_root=str(tmp_path),
                           output_root=str(tmp_path / "profile"))
    qualification.prepare(args)
    profile = json.loads((tmp_path / "profile" / "qualification_profile.json").read_text())
    assert profile["status"] == "NON_ADMITTED_FORMAL_PROFILE"
    for group, member in profile["members"].items():
        runtime = json.loads(Path(member["runtime"]).read_text())
        assert runtime["run"]["mode"] == "formal"
        assert "diagnostic_updates" not in runtime["run"]
        assert runtime["run"]["run_id"].startswith("qualification:formal:")
    assert "admission" not in json.dumps(profile)
    assert profile["complete_long_input_coverage"] == coverage


def test_formal_document_refuses_group_or_mode_drift(tmp_path):
    base = {"run": {"run_id": "diagnostic:interleaved:17:A", "group": "A", "seed": 17, "mode": "diagnostic", "diagnostic_updates": 4}}
    assert qualification._formal_document(base, "A", tmp_path)["run"]["mode"] == "formal"
    base["run"]["group"] = "U"
    try:
        qualification._formal_document(base, "A", tmp_path)
    except ValueError:
        return
    raise AssertionError("group drift was accepted")


def test_execution_binding_preserves_launcher_and_sets_uuid(monkeypatch, tmp_path):
    target = tmp_path / "python-target"; target.write_bytes(b"target")
    launcher = tmp_path / "python"; launcher.symlink_to(target)
    environment = {"controlled": "environment"}
    monkeypatch.setattr(qualification, "validate_environment", lambda value, volume: value == environment)
    monkeypatch.setattr(qualification, "gpu_uuid", lambda index: "GPU-bound" if index == 3 else None)
    monkeypatch.setattr(qualification, "sha256_file", lambda path: "digest-" + path.name)
    interpreter, child, uuid = qualification._execution_binding(str(launcher), environment, 3)
    assert interpreter == {"path": str(launcher), "launcher_sha256": "digest-python", "target_sha256": "digest-python-target"}
    assert child == {"controlled": "environment", "CUDA_VISIBLE_DEVICES": "GPU-bound"}
    assert uuid == "GPU-bound"


def _driver_inventory(monkeypatch, text):
    monkeypatch.setattr(diagnostic.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout=text))


def _one_logical_cuda(monkeypatch):
    monkeypatch.setattr(diagnostic.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(diagnostic.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(diagnostic.torch.cuda, "current_device", lambda: 0)


def test_diagnostic_cuda_binding_resolves_exact_and_prefix_uuid_visibility(monkeypatch):
    _driver_inventory(monkeypatch, "0, GPU-first\n1, GPU-second\n")
    _one_logical_cuda(monkeypatch)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-first")
    assert diagnostic._cuda_binding() == {"cuda_visible_devices": "GPU-first", "gpu_uuid": "GPU-first"}
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-sec")
    assert diagnostic._cuda_binding() == {"cuda_visible_devices": "GPU-second", "gpu_uuid": "GPU-second"}


def test_diagnostic_cuda_binding_rejects_unresolved_or_ambiguous_visibility(monkeypatch):
    _one_logical_cuda(monkeypatch)
    _driver_inventory(monkeypatch, "0, GPU-alpha\n1, GPU-alpine\n")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(RuntimeError, match="UUID, not an ordinal"):
        diagnostic._cuda_binding()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-missing")
    with pytest.raises(RuntimeError, match="does not resolve"):
        diagnostic._cuda_binding()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-alp")
    with pytest.raises(RuntimeError, match="does not resolve"):
        diagnostic._cuda_binding()


def test_diagnostic_cuda_binding_rejects_nonunique_cuda_logical_device(monkeypatch):
    _driver_inventory(monkeypatch, "0, GPU-bound\n")
    _one_logical_cuda(monkeypatch)
    monkeypatch.setattr(diagnostic.torch.cuda, "device_count", lambda: 2)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-bound")
    with pytest.raises(RuntimeError, match="exactly one CUDA logical device"):
        diagnostic._cuda_binding()


def test_diagnostic_main_uses_a_fresh_binding_probe_before_parent_assembly(monkeypatch, tmp_path):
    events = []
    monkeypatch.setattr(diagnostic, "_load_bundle_manifest", lambda *args: {})
    monkeypatch.setattr(diagnostic, "_load_members", lambda *args: {})
    monkeypatch.setattr(diagnostic, "_verify_source_manifest", lambda *args: events.append("source_verified"))
    binding = {"cuda_visible_devices": "GPU-bound", "gpu_uuid": "GPU-bound"}
    monkeypatch.setattr(diagnostic, "_binding_probe", lambda args: events.append("probe") or binding)
    def run_parent(args, harness, members, observed_binding):
        events.append("parent")
        assert observed_binding == binding
        return {"status": "BUNDLED_DIAGNOSTIC_COMPLETE"}
    monkeypatch.setattr(diagnostic, "_run_parent", run_parent)
    monkeypatch.setattr(diagnostic.sys, "argv", ["diagnostic", "--manifest", "manifest", "--manifest-sha256", "a" * 64,
                                                   "--runtime-source-root", str(diagnostic.ROOT), "--output", str(tmp_path / "out.json"),
                                                   "--mode", "run"])
    diagnostic.main()
    assert events == ["source_verified", "probe", "parent"]


def test_binding_probe_exits_before_loading_bundle_members_or_sources(monkeypatch, tmp_path):
    binding = {"cuda_visible_devices": "GPU-bound", "gpu_uuid": "GPU-bound"}
    monkeypatch.setattr(diagnostic, "_cuda_binding", lambda: binding)
    for name in ("_load_bundle_manifest", "_load_members", "_verify_source_manifest"):
        monkeypatch.setattr(diagnostic, name, lambda *args, **kwargs: pytest.fail("binding probe loaded bundle inputs"))
    output = tmp_path / "probe.json"
    monkeypatch.setattr(diagnostic.sys, "argv", ["diagnostic", "--manifest", "manifest", "--manifest-sha256", "a" * 64,
                                                   "--runtime-source-root", str(diagnostic.ROOT), "--output", str(output),
                                                   "--mode", "binding-probe"])
    diagnostic.main()
    assert json.loads(output.read_text()) == {"status": "CUDA_BINDING_PROBE_COMPLETE", "cuda_binding": binding}


def test_parent_defers_actual_cuda_binding_until_after_production_construction(monkeypatch):
    import torch
    events = []
    binding = {"cuda_visible_devices": "GPU-bound", "gpu_uuid": "GPU-bound"}

    def construct(*args, **kwargs):
        events.append("construct")
        assert not torch.cuda.is_initialized()
        return "models", "workers"

    monkeypatch.setattr(diagnostic, "_construct_workers", construct)
    monkeypatch.setattr(diagnostic, "_cuda_binding", lambda: events.append("bind") or binding)
    assert diagnostic._verified_parent_binding({}, {}, binding) == ("models", "workers", binding)
    assert events == ["construct", "bind"]


def test_diagnostic_prefix_requires_actual_derived_fixed_material_set():
    coverage = {"diagnostic_prefix": {"updates": 4, "accumulation": 8, "seed": 17,
                                        "sample_ids": ["a", "b"], "order_sha256": "c" * 64}}
    observed = {"a": "a" * 64, "b": "b" * 64}
    assert qualification._verify_diagnostic_prefix(coverage, {"material_digests": observed}) == {
        "diagnostic_prefix": coverage["diagnostic_prefix"], "material_digests": observed}
    with pytest.raises(ValueError, match="bound fixed input prefix"):
        qualification._verify_diagnostic_prefix(coverage, {"material_digests": {"a": "a" * 64}})


def test_longest_report_requires_catalog_selection_and_unchanged_provider_lineage(tmp_path):
    source_files = {path: "a" * 64 for path in qualification.LONGEST_SOURCE_SCOPE["detection_xd"]}
    report = tmp_path / "longest.json"
    report.write_text(json.dumps({"status": qualification.LONGEST_STATUS["detection_xd"],
                                  "selected": {"dataset": "xd-violence", "key": "video", "query_index": 9},
                                  "source_sha256": source_files,
                                  "diagnostic_protocol": {"scoring": "yesno"}}))
    task = SimpleNamespace(task="detection", dataset="xd-violence", media_key="video", query_index=9)
    catalog = SimpleNamespace(tasks={"detection:xd-violence:video:9": task})
    runtime = {"fast": {"protocols": {"xd-violence": {"scoring": "yesno"}}}}
    result = qualification._component_longest("detection_xd", str(report), _sha(report), catalog, source_files, runtime, None, None)
    assert result["selected_catalog_sample"] == "detection:xd-violence:video:9"
    source_files["src/nc_rted/detection_provider.py"] = "b" * 64
    with pytest.raises(ValueError, match="unchanged provider/training lineage"):
        qualification._component_longest("detection_xd", str(report), _sha(report), catalog, source_files, runtime, None, None)


def test_catalog_prefix_is_derived_from_bound_catalog_and_seed(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest"; manifest.mkdir()
    annotations = tmp_path / "train.json"; annotations.write_text("[]")
    provenance = manifest / "provenance.json"; provenance.write_text("{}")
    tasks = {"one": object(), "two": object(), "three": object(), "four": object(),
             "five": object(), "six": object(), "seven": object(), "eight": object()}
    catalog = SimpleNamespace(tasks=tasks, identity="catalog-id")
    monkeypatch.setattr(qualification.TrainingCatalog, "load", lambda *args, **kwargs: catalog)
    runtime = {"catalog": {"manifest_directory": str(manifest), "training_annotations": str(annotations),
                           "training_annotations_sha256": _sha(annotations), "provenance": str(provenance),
                           "provenance_sha256": _sha(provenance)},
               "run": {"diagnostic_updates": 1, "seed": 17}}
    binding, loaded, prefix = qualification._catalog_and_prefix(runtime)
    assert binding["identity"] == "catalog-id" and loaded is catalog
    assert len(prefix["sample_ids"]) == 8
    assert prefix["order_sha256"] == qualification.order_sha256(prefix["sample_ids"])


def test_caption_candidates_map_historical_diagnostic_selection_into_full_subset(tmp_path):
    full_subset = tmp_path / "captions2000.json"
    full_subset.write_text(json.dumps([{"id": 7, "video": "xd/videos/a.mp4"}]))
    runtime = {"catalog": {"caption_subset": str(full_subset), "caption_subset_sha256": _sha(full_subset)}}
    report = {"diagnostic_subset_sha256": "d" * 64,
              "selected": {"instruction_id": 7, "relative_video": "xd/videos/a.mp4",
                           "media": {"dataset": "xd-violence", "media_path": "/data/a.mp4", "media_sha256": "m" * 64}}}
    candidates = tmp_path / "candidates.json"
    candidates.write_text(json.dumps({"schema": "nc_rted_caption_longest_candidates/v2",
                                     "caption_subset_sha256": _sha(full_subset),
                                     "global_longest_all_2000": {"id": 7, "dataset": "xd-violence",
                                                                  "relative_video": "xd/videos/a.mp4", "source": "/data/a.mp4",
                                                                  "source_sha256": "m" * 64}}))
    result = qualification._caption_candidate_binding(str(candidates), _sha(candidates), report, runtime)
    assert result["diagnostic_subset_sha256"] == "d" * 64
    assert result["full_caption_subset"] == _binding(full_subset)
    mismatched_runtime = {"catalog": {"caption_subset": str(full_subset), "caption_subset_sha256": "f" * 64}}
    with pytest.raises(ValueError):
        qualification._caption_candidate_binding(str(candidates), _sha(candidates), report, mismatched_runtime)
    report["selected"]["media"]["media_sha256"] = "x" * 64
    with pytest.raises(ValueError, match="does not map"):
        qualification._caption_candidate_binding(str(candidates), _sha(candidates), report, runtime)


def test_historical_longest_source_transitions_are_scoped_to_material_callables():
    reports = Path("/root/autodl-tmp/lookaway-wm/reports/nc_rted")
    cases = (("caption", "real_caption_gpu_longest_pro6000_v2.json", "nc_rted_publication_candidate_v10"),
             ("detection_xd", "real_slow_gpu_longest_xd-violence_pro6000_v2.json", "nc_rted_publication_candidate_v10"),
             ("detection_ucf", "real_slow_gpu_longest_ucf-crime_pro6000_v3.json", "nc_rted_publication_candidate_v12"))
    for name, report_name, root_name in cases:
        report = json.loads((reports / report_name).read_text())
        source_files = {relative: qualification._sha(qualification.ROOT / relative)
                        for relative in qualification.LONGEST_SOURCE_SCOPE[name]}
        lineage = qualification._source_lineage(name, report["source_sha256"], source_files,
                                                f"/root/autodl-tmp/lookaway-wm/.cache/{root_name}")
        assert lineage["mode"] == "historical_source_and_scoped_callable_compatibility"


def test_profile_identity_rejects_missing_member_hash_and_derivation_drift(tmp_path, monkeypatch):
    root = tmp_path / "profile"; (root / "members").mkdir(parents=True)
    diagnostic_members, profiles = {}, {}
    for group in qualification.GROUPS:
        original = {"run": {"run_id": f"diagnostic:17:{group}", "group": group, "seed": 17,
                            "mode": "diagnostic", "diagnostic_updates": 4}}
        origin = tmp_path / f"{group}.origin.json"; origin.write_text(json.dumps(original))
        target = root / "members" / f"{group}.runtime.json"
        target.write_text(json.dumps(qualification._formal_document(original, group, root)))
        diagnostic_members[group] = {"manifest": str(origin), "sha256": _sha(origin)}
        profiles[group] = {"runtime": str(target), "runtime_sha256": _sha(target)}
    bundle_path = tmp_path / "bundle.json"; bundle_path.write_text("bundle")
    bundle = {"members": diagnostic_members}
    profile = {"schema": qualification.PROFILE_SCHEMA, "status": "NON_ADMITTED_FORMAL_PROFILE",
               "diagnostic_bundle": str(bundle_path), "diagnostic_bundle_sha256": _sha(bundle_path), "members": profiles,
               "member_identities": {group: {"group": group, "code_sha256": "a" * 64} for group in qualification.GROUPS},
               "updates": 1000, "accumulation": 8, "shared_preparation": "frozen_provider_only", "common_recovery": True,
               "complete_long_input_coverage": {}}
    monkeypatch.setattr(qualification, "load_manifest", lambda path, *, expected_sha256: SimpleNamespace(document=json.loads(Path(path).read_text())))
    monkeypatch.setattr(qualification, "preflight", lambda manifest: None)
    monkeypatch.setattr(qualification, "formal_runtime_identity", lambda manifest: {"group": manifest.document["run"]["group"], "code_sha256": "a" * 64})
    assert qualification._profile_identities(profile, root / "qualification_profile.json", bundle, bundle_path)["A"]["group"] == "A"
    (root / "members" / "A.runtime.json").unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        qualification._profile_identities(profile, root / "qualification_profile.json", bundle, bundle_path)
    original = {"run": {"run_id": "diagnostic:17:A", "group": "A", "seed": 17,
                        "mode": "diagnostic", "diagnostic_updates": 4}}
    target = root / "members" / "A.runtime.json"; target.write_text(json.dumps(qualification._formal_document(original, "A", root)))
    profile["members"]["A"]["runtime_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="member file"):
        qualification._profile_identities(profile, root / "qualification_profile.json", bundle, bundle_path)
    profile["members"]["A"]["runtime_sha256"] = _sha(target)
    altered = json.loads(target.read_text()); altered["run"]["seed"] = 99; target.write_text(json.dumps(altered))
    profile["members"]["A"]["runtime_sha256"] = _sha(target)
    with pytest.raises(ValueError, match="not derived"):
        qualification._profile_identities(profile, root / "qualification_profile.json", bundle, bundle_path)
    target.write_text(json.dumps(qualification._formal_document(original, "A", root)))
    profile["members"]["A"]["runtime_sha256"] = _sha(target)
    profile["member_identities"]["A"] = {"group": "A", "code_sha256": "c" * 64}
    with pytest.raises(ValueError, match="identity"):
        qualification._profile_identities(profile, root / "qualification_profile.json", bundle, bundle_path)
    profile["member_identities"]["A"] = {"group": "A", "code_sha256": "a" * 64}
    changed_origin = json.loads((tmp_path / "A.origin.json").read_text()); changed_origin["run"]["seed"] = 98
    (tmp_path / "A.origin.json").write_text(json.dumps(changed_origin)); bundle["members"]["A"]["sha256"] = _sha(tmp_path / "A.origin.json")
    with pytest.raises(ValueError, match="not derived"):
        qualification._profile_identities(profile, root / "qualification_profile.json", bundle, bundle_path)


def test_common_source_closure_is_accepted_by_formal_verifier(tmp_path):
    builder_spec = importlib.util.spec_from_file_location("diagnostic_builder", ROOT / "scripts" / "nc_rted_build_interleaved_diagnostic_manifests.py")
    builder = importlib.util.module_from_spec(builder_spec); builder_spec.loader.exec_module(builder)
    formal_spec = importlib.util.spec_from_file_location("formal_runner", ROOT / "scripts" / "nc_rted_interleaved_formal.py")
    formal = importlib.util.module_from_spec(formal_spec); formal_spec.loader.exec_module(formal)
    source = builder._source_manifest(ROOT)
    assert {"scripts/nc_rted_interleaved_gpu_diagnostic.py", "scripts/nc_rted_interleaved_formal.py",
            "scripts/nc_rted_qualify_interleaved_formal_bundle.py"}.issubset(source["files"])
    path = tmp_path / "bundle-source-manifest.json"; path.write_text(json.dumps(source))
    captured = tmp_path / "captured"; captured.mkdir(); (captured / "bundle-source-manifest.json").write_text(path.read_text())
    formal._verify_source_manifest({"source_manifest_sha256": _sha(path)}, captured)
    qualification._verify_common_source_closure({"source_manifest": str(path), "source_manifest_sha256": _sha(path)}, source["code_sha256"])


def test_qualify_uses_only_controlled_environment_and_rejects_device_mismatch(tmp_path, monkeypatch):
    profile_path = tmp_path / "profile.json"; bundle_path = tmp_path / "bundle.json"; bundle_path.write_text("bundle")
    bound = {"diagnostic_prefix": {"updates": 4, "accumulation": 8, "seed": 17,
                                    "sample_ids": ["long-sample"], "order_sha256": "d" * 64}}
    profile = {"schema": qualification.PROFILE_SCHEMA, "status": "NON_ADMITTED_FORMAL_PROFILE",
               "diagnostic_bundle": str(bundle_path), "diagnostic_bundle_sha256": _sha(bundle_path),
               "complete_long_input_coverage": bound,
               "members": {"A": {"runtime": str(tmp_path / "unused.json"), "runtime_sha256": "a" * 64}}}
    profile_path.write_text(json.dumps(profile))
    launcher = tmp_path / "venv-python"; launcher.write_bytes(b"launcher")
    raw = tmp_path / "raw.json"; serial = tmp_path / "serial.json"
    serial.write_text(json.dumps({"cuda_binding": {"cuda_visible_devices": "GPU-bound", "gpu_uuid": "GPU-bound"}}))
    resumed = {"status": "BUNDLED_DIAGNOSTIC_COMPLETE", "serial_comparison": "exact_state_digest_match",
               "completed": {group: 1 for group in qualification.GROUPS}, "material_digests": {"long-sample": "b" * 64},
               "cuda_binding": {"cuda_visible_devices": "GPU-bound", "gpu_uuid": "GPU-bound"},
               "peak_memory_bytes": 1, "peak_reserved_memory_bytes": 2, "elapsed_seconds": 3}
    calls = []
    def run(command, **kwargs):
        calls.append((command, kwargs)); raw.write_text(json.dumps({"status": "BUNDLED_DIAGNOSTIC_COMPLETE", "serial_reference": str(serial), "resumed": resumed}))
    identities = {group: {"group": group, "code_sha256": "a" * 64} for group in qualification.GROUPS}
    origin = tmp_path / "origin.json"; origin.write_text(json.dumps({"run": {"diagnostic_updates": 4, "seed": 17}}))
    monkeypatch.setattr(qualification, "_diagnostic_bundle", lambda *args: {"members": {"A": {"manifest": str(origin), "sha256": _sha(origin)}}})
    monkeypatch.setattr(qualification, "_profile_identities", lambda *args: identities)
    monkeypatch.setattr(qualification, "load_manifest", lambda path, *, expected_sha256: SimpleNamespace(document=json.loads(Path(path).read_text())))
    monkeypatch.setattr(qualification, "_verify_common_source_closure", lambda *args: None)
    monkeypatch.setattr(qualification, "_source_files", lambda *args: {})
    monkeypatch.setattr(qualification, "_complete_long_input_coverage", lambda *args: bound)
    monkeypatch.setattr(qualification, "validate_environment", lambda *args: None)
    monkeypatch.setattr(qualification, "gpu_uuid", lambda index: "GPU-bound")
    monkeypatch.setattr(qualification, "sha256_file", lambda path: "a" * 64)
    monkeypatch.setattr(qualification, "python_runtime_probe", lambda *args: {"runtime": "bound"})
    monkeypatch.setattr(qualification, "gpu_memory_bytes", lambda index: 4)
    monkeypatch.setattr(qualification.subprocess, "run", run)
    args = SimpleNamespace(profile=str(profile_path), profile_sha256=_sha(profile_path), runtime_source_root=str(ROOT), raw_output=str(raw),
                           physical_gpu=1, interpreter=str(launcher), execution_environment=json.dumps({"ONLY": "claimed"}),
                           validity_seconds=10, fault_after_group="U", output=str(tmp_path / "qualification.json"))
    qualification.qualify(args)
    assert calls[0][0][0] == str(launcher)
    assert calls[0][1]["env"] == {"ONLY": "claimed", "CUDA_VISIBLE_DEVICES": "GPU-bound"}
    resumed["cuda_binding"]["gpu_uuid"] = "GPU-other"
    with pytest.raises(ValueError, match="CUDA device"):
        qualification.qualify(args)
