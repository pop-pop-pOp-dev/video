import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nc_rted import resource_attestation as attestation
from nc_rted.production_runtime import load_manifest
from test_nc_rted_queue import bundle_fixture


GROUPS = ("A", "U", "S", "F")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value), encoding="utf-8")
    return digest(path)


def code_sha(files):
    return hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def source34_bundle(tmp_path, monkeypatch, *, seed=42):
    """Build a source33 measurement and source34 payload through real consumers."""
    payload, qualification, _authorization, attestation_path, evidence, _ = bundle_fixture(tmp_path, monkeypatch)
    repo = Path(__file__).resolve().parents[1]
    files = {str(path.relative_to(repo)): digest(path) for path in sorted((repo / "src" / "nc_rted").glob("*.py"))}
    files.update({f"scripts/{name}": digest(repo / "scripts" / name) for name in (
        "nc_rted_interleaved_gpu_diagnostic.py", "nc_rted_interleaved_formal.py",
        "nc_rted_qualify_interleaved_formal_bundle.py",
    )})
    target_code = code_sha(files)
    measured_files = dict(files)
    measured_files["src/nc_rted/resource_attestation.py"] = "0" * 64
    measured_code = code_sha(measured_files)
    target_manifest, measured_manifest = tmp_path / "source34-manifest.json", tmp_path / "source33-manifest.json"
    write(target_manifest, {"schema": "nc_rted_interleaved_source_manifest/v1", "files": files, "code_sha256": target_code})
    write(measured_manifest, {"schema": "nc_rted_interleaved_source_manifest/v1", "files": measured_files, "code_sha256": measured_code})

    source33_root = tmp_path / "source33" / "src" / "nc_rted"
    source33_root.mkdir(parents=True)
    (source33_root / "production_runtime.py").write_bytes((repo / "src" / "nc_rted" / "production_runtime.py").read_bytes())
    target_environment = payload["execution_environment"]
    measured_environment = dict(target_environment, PYTHONPATH=str(source33_root.parent))
    monkeypatch.setattr(attestation, "python_runtime_probe", lambda _interpreter, environment: {
        "project_module": str(Path(environment["PYTHONPATH"]) / "nc_rted" / "production_runtime.py"), "stable": "probe",
    })

    bundle_path = Path(payload["bundle_config"])
    bundle = json.loads(bundle_path.read_text())
    identities, target_runtime_bindings = {}, {}
    target_runs = attestation.PROJECT_VOLUME / ".cache" / "nc_rted_queue_tests" / tmp_path.name / "source34-runs"
    for group in GROUPS:
        member = bundle["members"][group]
        runtime_path = Path(member["runtime"])
        runtime = json.loads(runtime_path.read_text())
        runtime["run"].update({"run_id": f"source34:{seed}:{group}", "seed": seed,
                               "checkpoint_root": str(target_runs / group / "checkpoints"),
                               "progress_path": str(target_runs / group / "progress.json")})
        runtime["hashes"].update({"code_sha256": target_code, "runtime_sha256": target_code})
        write(runtime_path, runtime)
        member["runtime_sha256"] = digest(runtime_path)
        identity = attestation.formal_runtime_identity(load_manifest(runtime_path, expected_sha256=member["runtime_sha256"]))
        identities[group] = identity
        admission_path = Path(member["admission"])
        admission = json.loads(admission_path.read_text())
        admission.update({"run_identity": identity, "source_files": files})
        member["admission_sha256"] = write(admission_path, admission)
        target_runtime_bindings[group] = {"path": str(runtime_path), "sha256": member["runtime_sha256"]}

    measured_root = tmp_path / "measured-runtimes"
    measured_root.mkdir()
    measured_identities, measured_runtime_bindings = {}, {}
    for group in GROUPS:
        target_runtime = json.loads(Path(bundle["members"][group]["runtime"]).read_text())
        measured_runtime = json.loads(json.dumps(target_runtime))
        measured_runtime["run"].update({"run_id": f"source33:17:{group}", "seed": 17,
                                         "checkpoint_root": str(measured_root / group / "checkpoints"),
                                         "progress_path": str(measured_root / group / "progress.json")})
        measured_runtime["hashes"].update({"code_sha256": measured_code, "runtime_sha256": measured_code})
        path = measured_root / f"{group}.runtime.json"
        sha = write(path, measured_runtime)
        measured_identities[group] = attestation.formal_runtime_identity(load_manifest(path, expected_sha256=sha))
        measured_runtime_bindings[group] = {"path": str(path), "sha256": sha}

    bundle["source_manifest"] = str(target_manifest)
    bundle["source_manifest_sha256"] = digest(target_manifest)
    write(bundle_path, bundle)
    payload.update({"bundle_config_sha256": digest(bundle_path), "member_identities": identities, "frozen_source_sha256": target_code})
    payload["checkpoint_roots"] = [json.loads(Path(bundle["members"][group]["runtime"]).read_text())["run"]["checkpoint_root"] for group in GROUPS]
    payload["progress_path"] = json.loads(Path(bundle["members"]["A"]["runtime"]).read_text())["run"]["progress_path"]
    checkpoints = {group: str((Path(payload["checkpoint_roots"][index]) / "final" / "manifest.json").resolve()) for index, group in enumerate(GROUPS)}
    payload["expected_outputs"] = [
        {"path": checkpoints[group], "artifact_type": "checkpoint", "semantic": "formal_training", "run_identity": identities[group]}
        for group in GROUPS
    ] + [{"path": "bundle-report.json", "artifact_type": "report", "semantic": "formal_bundle",
          "member_identities": identities, "final_checkpoints": checkpoints,
          "bundle_checkpoint_root": str(Path(bundle["bundle_checkpoint_root"]).resolve())}]
    payload["command"][payload["command"].index("--bundle-sha256") + 1] = payload["bundle_config_sha256"]

    profile = tmp_path / "source33-profile.json"
    profile_document = {"schema": "nc_rted_interleaved_formal_qualification_profile/v1", "status": "NON_ADMITTED_FORMAL_PROFILE",
                        "members": {group: {"runtime": measured_runtime_bindings[group]["path"], "runtime_sha256": measured_runtime_bindings[group]["sha256"]} for group in GROUPS},
                        "member_identities": measured_identities, "updates": 1000, "accumulation": 8,
                        "shared_preparation": "frozen_provider_only", "common_recovery": True}
    profile_sha = write(profile, profile_document)
    qualification_document = json.loads(Path(qualification).read_text())
    qualification_document.update({"source_sha256": measured_code,
                                    "workload": {"member_identities": measured_identities, "updates": 1000, "kind": "formal_bundle",
                                                 "shared_preparation": "frozen_provider_only", "common_recovery": True},
                                    "runtime_environment": {"interpreter": payload["interpreter"], "environment": measured_environment},
                                    "qualification_profile": {"path": str(profile), "sha256": profile_sha}})
    qualification_document["measurements"].update({"seconds_per_bundle_update_upper_bound": 0.02,
                                                     "setup_checkpoint_seconds_upper_bound": 1.0,
                                                     "local_import_probe": {"status": "PASS", "interpreter": payload["interpreter"],
                                                                            "environment": measured_environment, "duration_seconds": 0.01,
                                                                            "runtime_identity": attestation.python_runtime_probe(payload["interpreter"], measured_environment)}})
    qualification_sha = write(qualification, qualification_document)

    applicability_path = tmp_path / "source34-applicability.json"
    applicability = {"schema": attestation.SOURCE34_APPLICABILITY_SCHEMA, "status": "PASS_CONSERVATIVE_SEED_APPLICABILITY",
                     "measured_qualification": {"path": str(qualification), "sha256": qualification_sha},
                     "measured_identities": measured_identities, "target_seed": seed, "target_identities": identities,
                     "measured_runtimes": measured_runtime_bindings, "target_runtimes": target_runtime_bindings,
                     "source_transition": {"allowed_changed_files": ["src/nc_rted/resource_attestation.py"],
                                           "unchanged_files": {name: value for name, value in measured_files.items() if name != "src/nc_rted/resource_attestation.py"},
                                           "measured_manifest": {"path": str(measured_manifest), "sha256": digest(measured_manifest)},
                                           "target_manifest": {"path": str(target_manifest), "sha256": digest(target_manifest)}},
                     "invariants": {"samples": 8000, "updates": 1000, "accumulation": 8, "shared_preparation": "frozen_provider_only",
                                    "common_recovery": True, "sampler_rng_difference_explicit": True},
                     "projection": {"measured_seconds_per_bundle_update_upper_bound": 0.02, "measured_setup_checkpoint_seconds_upper_bound": 1.0,
                                    "safety_multiplier": 2.0, "projected_total_seconds_upper_bound": 42.0, "is_measured_target_timing": False}}
    applicability_sha = write(applicability_path, applicability)

    attestation_document = json.loads(attestation_path.read_text())
    attestation_document["binding"].update({"bundle_config_sha256": payload["bundle_config_sha256"], "member_identities": identities,
                                              "frozen_source_sha256": target_code,
                                              "execution_inputs": {key: payload[key] for key in ("command", "execution_environment", "interpreter", "run_dir", "progress_path", "checkpoint_roots", "bundle_checkpoint_root", "expected_outputs")}})
    attestation_document["qualification"]["sha256"] = qualification_sha
    attestation_document["source34_applicability"] = {"path": str(applicability_path), "sha256": applicability_sha}
    write(attestation_path, attestation_document)
    payload["resource_attestation_sha256"] = digest(attestation_path)
    evidence["bundle"] = (str(bundle_path), payload["bundle_config_sha256"])
    evidence["qualification"] = (str(qualification), qualification_sha)
    return payload, evidence, applicability_path, attestation_path


def test_source34_bundle_attestation_accepts_proven_seed_relocation(tmp_path, monkeypatch):
    payload, evidence, _applicability, _attestation = source34_bundle(tmp_path, monkeypatch)
    attestation.verify_bundle_attestation(payload, "bundle", evidence)


@pytest.mark.parametrize("mutate", (
    lambda document: document["source_transition"].update(allowed_changed_files=["src/nc_rted/training.py"]),
    lambda document: document["target_runtimes"]["A"].update(sha256="0" * 64),
    lambda document: document["projection"].update(measured_seconds_per_bundle_update_upper_bound=0.03),
    lambda document: document["projection"].update(projected_total_seconds_upper_bound=61.0),
))
def test_source34_bundle_attestation_rejects_unproven_transfer(tmp_path, monkeypatch, mutate):
    payload, evidence, applicability_path, attestation_path = source34_bundle(tmp_path, monkeypatch)
    document = json.loads(applicability_path.read_text())
    mutate(document)
    write(applicability_path, document)
    attestation_document = json.loads(attestation_path.read_text())
    attestation_document["source34_applicability"]["sha256"] = digest(applicability_path)
    write(attestation_path, attestation_document)
    payload["resource_attestation_sha256"] = digest(attestation_path)
    with pytest.raises(attestation.ResourceAttestationError):
        attestation.verify_bundle_attestation(payload, "bundle", evidence)


def test_source34_bundle_attestation_rejects_non_pythonpath_probe_change(tmp_path, monkeypatch):
    payload, evidence, _applicability, _attestation = source34_bundle(tmp_path, monkeypatch)
    monkeypatch.setattr(attestation, "python_runtime_probe", lambda _interpreter, environment: {
        "project_module": str(Path(environment["PYTHONPATH"]) / "nc_rted" / "production_runtime.py"), "stable": "changed",
    })
    with pytest.raises(attestation.ResourceAttestationError):
        attestation.verify_bundle_attestation(payload, "bundle", evidence)


def test_source34_bundle_attestation_binds_measured_probe_environment(tmp_path, monkeypatch):
    payload, evidence, applicability_path, attestation_path = source34_bundle(tmp_path, monkeypatch)
    attestation_document = json.loads(attestation_path.read_text())
    qualification_path = Path(attestation_document["qualification"]["path"])
    qualification = json.loads(qualification_path.read_text())
    qualification["measurements"]["local_import_probe"]["environment"] = {"PYTHONPATH": "/wrong/source33"}
    qualification_sha = write(qualification_path, qualification)
    applicability = json.loads(applicability_path.read_text())
    applicability["measured_qualification"]["sha256"] = qualification_sha
    applicability_sha = write(applicability_path, applicability)
    attestation_document["qualification"]["sha256"] = qualification_sha
    attestation_document["source34_applicability"]["sha256"] = applicability_sha
    write(attestation_path, attestation_document)
    payload["resource_attestation_sha256"] = digest(attestation_path)
    evidence["qualification"] = (str(qualification_path), qualification_sha)
    with pytest.raises(attestation.ResourceAttestationError):
        attestation.verify_bundle_attestation(payload, "bundle", evidence)
