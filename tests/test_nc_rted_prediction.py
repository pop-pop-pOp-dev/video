from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
import errno
import hashlib
import importlib._bootstrap_external
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
import types

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nc_rted.prediction_inputs import (CANDIDATE_SCHEMA_V2, IMPLEMENTATION_SCHEMA, RESOURCE_ADMISSION_SCHEMA_V1, RESOURCE_ALLOCATION_SCHEMA_V1, RESOURCE_MEASUREMENT_SCHEMA_V1, SCHEMA_V2,
                                       _IMPLEMENTATION_FILES, ModelArtifact, PredictionInputError, VadRequest, VauRequest, _bound_path, canonical_json,
                                       load_model_artifact, load_prediction_plan, prediction_execution_binding_sha256, prediction_matrix_id,
                                       verify_implementation_manifest)
import nc_rted.prediction_store as prediction_store
import nc_rted.prediction_inputs as prediction_inputs
from nc_rted.prediction_store import PredictionPublicationError, PredictionStore, PredictionStoreError
from nc_rted.prediction_worker import PredictionExecutionError, PredictionWorker
from nc_rted.prediction_media import FullBlindDetectionReader
from nc_rted.prediction_media import FullBlindHivauReader
from nc_rted.detection_provider import DetectionProtocol
from nc_rted.prediction_runtime import (PredictionRuntime, _VERIFIED_INHERITED_MODULES, _audit_bound_inherited_modules, _import_bound_runtime,
                                        _restore_trainable, load_prediction_runtime, validate_artifact_runtime)
from nc_rted.prediction_adapters import BlindDetectionRunner, BlindPromptTokenizer


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree(path: Path) -> str:
    digest = hashlib.sha256()
    for child in sorted(path.rglob("*")):
        if child.is_file():
            digest.update(str(child.relative_to(path)).encode()); digest.update(b"\0")
            digest.update(_digest(child).encode()); digest.update(b"\n")
    return digest.hexdigest()


def _write(path: Path, value: object) -> str:
    path.write_text(json.dumps(value))
    return _digest(path)


def _fixture(tmp_path, *, forbidden=False):
    tmp_path.mkdir(parents=True, exist_ok=True)
    media = []
    for name in ("ucf", "xd", "vau"):
        path = tmp_path / f"{name}.mp4"; path.write_bytes(name.encode()); media.append(path)
    tasks = [("R0", None)] + [(group, seed) for group in ("A", "U", "S", "F") for seed in (17, 42, 2026)]
    bindings = {}
    for name in ("runtime", "fast_snapshot", "source_manifest", "tokenizer", "embedded_vision_binding", "decoder", "implementation_manifest"):
        if name == "tokenizer":
            item = tmp_path / name; item.mkdir(); (item / "tokenizer.json").write_text(name)
            bindings[name], bindings[f"{name}_sha256"] = str(item), _tree(item)
        else:
            item = tmp_path / f"{name}.json"; item.write_text(name)
            bindings[name], bindings[f"{name}_sha256"] = str(item), _digest(item)
    identities = {"vad": [
        {"dataset": "ucf", "id": "u1", "media_path": str(media[0]), "media_sha256": _digest(media[0])},
        {"dataset": "xd", "id": "x1", "media_path": str(media[1]), "media_sha256": _digest(media[1])},
    ], "vau": [{"id": "q1", "media_path": str(media[2]), "media_sha256": _digest(media[2]), "question": "Describe this clip exactly."}]}
    if forbidden:
        identities["vau"][0]["answer"] = "forbidden"
    identity_path = tmp_path / "identities.json"; identity_digest = _write(identity_path, identities)
    model_tasks, model_manifests = [], {}
    for group, seed in tasks:
        task = {"group": group, "seed": seed}
        model_tasks.append(task)
        if group == "R0":
            artifact = {**task, "checkpoint": None, "checkpoint_manifest_sha256": None, "checkpoint_state_sha256": None,
                        "final_checkpoint_attestation": None, "final_checkpoint_attestation_sha256": None,
                        "accepted_training_provenance": None, "accepted_training_provenance_sha256": None}
        else:
            checkpoint = tmp_path / f"checkpoint-{group}-{seed}"; checkpoint.mkdir()
            checkpoint_state = checkpoint / "state.pt"; checkpoint_state.write_bytes(f"{group}-{seed}".encode())
            identity = {"run_id": "run", "group": group, "seed": str(seed), "code_sha256": "1" * 64, "config_sha256": "2" * 64,
                        "data_sha256": "3" * 64, "teacher_sha256": "4" * 64, "inherited_weights_sha256": "5" * 64, "runtime_sha256": "6" * 64}
            checkpoint_manifest = checkpoint / "manifest.json"
            _write(checkpoint_manifest, {"schema": "nc_rted_checkpoint_v2", "identity": identity, "final": True,
                                         "completed_updates": 1000, "payload_sha256": _digest(checkpoint_state)})
            attestation = tmp_path / f"final-{group}-{seed}.json"
            _write(attestation, {"schema": "nc_rted_final_checkpoint_attestation/v1", "status": "PASS", "final_checkpoint_allowed": True,
                                 "checkpoint_manifest_sha256": _digest(checkpoint_manifest), "checkpoint_state_sha256": _digest(checkpoint_state),
                                 "completed_updates": 1000, "training_identity": identity})
            provenance = tmp_path / f"accepted-{group}-{seed}.json"
            _write(provenance, {"schema": "nc_rted_accepted_training_provenance/v1", "status": "PASS", "training_allowed": True,
                                "group": group, "seed": seed, "checkpoint_manifest_sha256": _digest(checkpoint_manifest),
                                "checkpoint_state_sha256": _digest(checkpoint_state), "training_identity": identity})
            artifact = {**task, "checkpoint": str(checkpoint), "checkpoint_manifest_sha256": _digest(checkpoint_manifest),
                        "checkpoint_state_sha256": _digest(checkpoint_state), "final_checkpoint_attestation": str(attestation),
                        "final_checkpoint_attestation_sha256": _digest(attestation), "accepted_training_provenance": str(provenance),
                        "accepted_training_provenance_sha256": _digest(provenance)}
        path = tmp_path / f"model-{group}-{seed}.json"; _write(path, artifact)
        model_manifests[(group, seed)] = (path, _digest(path))
    protocol = {"hivau": {"target_fps": 4, "query_interval": 4, "paligemma_batch_size": 32,
                            "max_new_tokens": 512, "task": "description", "fast_prompt_context": "none"},
                "vad": {"target_fps": 4, "query_interval": 4, "batch_size": 1},
                "vad_route": "reactvau_detection", "vad_causal_smoothing": "online", "vad_fast_fusion": "released"}
    admission_path = tmp_path / "admission.json"
    _write(admission_path, {"status": "PASS", "formal_execution_allowed": True,
                            "embedded_vision_binding_sha256": bindings["embedded_vision_binding_sha256"],
                            "prediction_execution_binding_sha256": prediction_execution_binding_sha256(
                                bindings=bindings, protocol=protocol, identity_manifest_sha256=identity_digest, identities=identities)})
    manifest = {"schema": "nc_rted_blind_prediction/v1", "run_id": "test:blind-test", "identity_manifest": str(identity_path),
                "identity_manifest_sha256": identity_digest, "model_tasks": model_tasks,
                "protocol": protocol,
                "output_root": str(tmp_path / "output"), "denominators": {"ucf": 1, "xd": 1, "vau": 1},
                "admission": {"formal_admission": str(admission_path), "formal_admission_sha256": _digest(admission_path)}, "bindings": bindings}
    manifest_path = tmp_path / "manifest.json"; digest = _write(manifest_path, manifest)
    return manifest_path, digest, model_manifests


def _v2_fixture(tmp_path, *, ready=(("R0", None),)):
    _, _, model_manifests = _fixture(tmp_path)
    v1 = json.loads((tmp_path / "manifest.json").read_text())
    resource = tmp_path / "resource.json"
    resource_digest = _write(resource, {"schema": RESOURCE_ADMISSION_SCHEMA_V1, "status": "PASS", "formal_execution_allowed": True})
    bindings = {**v1["bindings"], "resource_admission": str(resource), "resource_admission_sha256": resource_digest}
    ready = set(ready)
    registry = []
    for group, seed in [("R0", None)] + [(group, seed) for group in ("A", "U", "S", "F") for seed in (17, 42, 2026)]:
        if (group, seed) in ready:
            path, digest = model_manifests[(group, seed)]
            registry.append({"group": group, "seed": seed, "state": "READY", "model_manifest": str(path),
                             "model_manifest_sha256": digest, "pending_reason": None})
        else:
            registry.append({"group": group, "seed": seed, "state": "PENDING_DEPENDENCY", "model_manifest": None,
                             "model_manifest_sha256": None, "pending_reason": "FINAL_CHECKPOINT_ATTESTATION_AND_PROVENANCE_PENDING"})
    registration = {"schema": SCHEMA_V2, "run_id": v1["run_id"], "identity_manifest": v1["identity_manifest"],
                    "identity_manifest_sha256": v1["identity_manifest_sha256"], "model_tasks": registry, "protocol": v1["protocol"],
                    "output_root": v1["output_root"], "denominators": v1["denominators"], "bindings": bindings}
    identities = json.loads(Path(v1["identity_manifest"]).read_text())
    matrix_id = prediction_matrix_id(run_id=registration["run_id"], bindings=bindings, protocol=registration["protocol"],
                                    identity_manifest_sha256=registration["identity_manifest_sha256"], denominators=registration["denominators"])
    binding = prediction_execution_binding_sha256(bindings=bindings, protocol=registration["protocol"],
                                                  identity_manifest_sha256=registration["identity_manifest_sha256"], identities=identities,
                                                  model_registry=registry)
    candidate = tmp_path / "candidate.json"
    candidate_digest = _write(candidate, {"schema": CANDIDATE_SCHEMA_V2, "matrix_id": matrix_id, "registration": registration,
                                          "prediction_execution_binding_sha256": binding})
    admission = tmp_path / "admission-v2.json"
    admission_digest = _write(admission, {"status": "PASS", "formal_execution_allowed": True,
                                          "embedded_vision_binding_sha256": bindings["embedded_vision_binding_sha256"],
                                          "prediction_execution_binding_sha256": binding,
                                          "prediction_plan_candidate_sha256": candidate_digest,
                                          "resource_admission_sha256": resource_digest})
    plan = tmp_path / "plan-v2.json"
    plan_digest = _write(plan, {**registration, "matrix_id": matrix_id, "candidate": {"path": str(candidate), "sha256": candidate_digest},
                                "admission": {"formal_admission": str(admission), "formal_admission_sha256": admission_digest}})
    return plan, plan_digest, model_manifests, registration


def test_plan_rejects_supervision_and_preserves_verbatim_question(tmp_path):
    manifest, digest, model_manifests = _fixture(tmp_path / "provenance")
    plan = load_prediction_plan(manifest, expected_sha256=digest)
    assert plan.vau[0].question == "Describe this clip exactly."
    assert len(plan.models) == 13
    assert plan.binding_sha256["runtime"] == json.loads(manifest.read_text())["bindings"]["runtime_sha256"]
    r0 = plan.selected_model("R0", None)
    artifact_path, artifact_hash = model_manifests[("R0", None)]
    artifact = load_model_artifact(artifact_path, expected_sha256=artifact_hash, task=r0)
    assert artifact.evidence_enabled is False
    # Keep the binding valid to prove the forbidden key itself is rejected.
    identity = Path(json.loads(manifest.read_text())["identity_manifest"])
    row = json.loads(identity.read_text()); row["vau"][0]["answer"] = "no"; identity.write_text(json.dumps(row))
    document = json.loads(manifest.read_text()); document["identity_manifest_sha256"] = _digest(identity); manifest.write_text(json.dumps(document))
    with pytest.raises(PredictionInputError):
        load_prediction_plan(manifest)


def test_r0_rejects_a_checkpoint_and_trained_groups_require_one(tmp_path):
    manifest, digest, model_manifests = _fixture(tmp_path)
    plan = load_prediction_plan(manifest, expected_sha256=digest)
    r0_path, r0_hash = model_manifests[("R0", None)]
    r0 = json.loads(r0_path.read_text())
    r0["checkpoint"] = str(tmp_path)
    r0_path.write_text(json.dumps(r0))
    with pytest.raises(PredictionInputError, match="R0"):
        load_model_artifact(r0_path, expected_sha256=_digest(r0_path), task=plan.selected_model("R0", None))
    trained_path, _ = model_manifests[("F", 2026)]
    trained = json.loads(trained_path.read_text())
    trained["checkpoint"] = None
    trained_path.write_text(json.dumps(trained))
    with pytest.raises(PredictionInputError, match="checkpoint"):
        load_model_artifact(trained_path, expected_sha256=_digest(trained_path), task=plan.selected_model("F", 2026))


def test_v2_registry_allows_r0_while_untrained_tasks_remain_explicitly_pending(tmp_path):
    manifest, digest, model_manifests, _ = _v2_fixture(tmp_path)
    plan = load_prediction_plan(manifest, expected_sha256=digest)
    assert plan.schema == SCHEMA_V2 and plan.matrix_id is not None and len(plan.models) == 13
    r0_path, r0_digest = model_manifests[("R0", None)]
    assert load_model_artifact(r0_path, expected_sha256=r0_digest, task=plan.selected_model("R0", None)).task_id == "R0"
    with pytest.raises(PredictionInputError, match="PENDING_DEPENDENCY"):
        plan.selected_model("A", 17)


def test_v2_registry_and_candidate_cannot_be_rebound_after_r0_prediction(tmp_path):
    manifest, _, _, _ = _v2_fixture(tmp_path)
    document = json.loads(manifest.read_text())
    document["model_tasks"][1]["state"] = "READY"
    document["model_tasks"][1]["model_manifest"] = str(tmp_path / "invented.json")
    document["model_tasks"][1]["model_manifest_sha256"] = "a" * 64
    document["model_tasks"][1]["pending_reason"] = None
    _write(manifest, document)
    with pytest.raises(PredictionInputError, match="selected model manifest"):
        load_prediction_plan(manifest)


def test_v2_store_reopens_unchanged_task_across_plan_revision(tmp_path):
    """A new registry revision must preserve immutable R0 records and skip them."""
    from nc_rted.prediction_inputs import prediction_task_execution_binding_sha256
    matrix, task = "a" * 64, "b" * 64
    first = PredictionStore(tmp_path / "r0", run_id="run", manifest_sha256="c" * 64, model_task="R0",
                            model_binding_sha256="d" * 64, matrix_id=matrix, task_execution_binding_sha256=task)
    first.publish(identity="vad:ucf:one", attempt=1, status="success",
                  provenance={"matrix_id": matrix, "task_execution_binding_sha256": task}, payload={"queries": []})
    reopened = PredictionStore(first.root, run_id="run", manifest_sha256="e" * 64, model_task="R0",
                               model_binding_sha256="d" * 64, matrix_id=matrix, task_execution_binding_sha256=task)
    record = reopened.get(identity="vad:ucf:one")
    assert record["manifest_sha256"] == "c" * 64
    assert reopened.should_run(identity="vad:ucf:one", max_retries=3) is False


def test_v2_worker_skips_preserved_r0_results_after_plan_revision(tmp_path):
    from nc_rted.prediction_inputs import prediction_task_execution_binding_sha256
    manifest, digest, models, _ = _v2_fixture(tmp_path)
    plan = load_prediction_plan(manifest, expected_sha256=digest)
    path, bound = models[("R0", None)]
    artifact = load_model_artifact(path, expected_sha256=bound, task=plan.selected_model("R0", None))
    task_binding = prediction_task_execution_binding_sha256(matrix_id=plan.matrix_id, task=artifact)
    store = PredictionStore(plan.output_root / "R0", run_id=plan.run_id, manifest_sha256=plan.manifest_sha256,
                            model_task="R0", model_binding_sha256=artifact.manifest_sha256, matrix_id=plan.matrix_id,
                            task_execution_binding_sha256=task_binding)
    assert PredictionWorker(plan, store, loader=_Loader(), vad=_Vad(), vau=_Vau(), model=artifact).run()["succeeded"] == 3
    revised = replace(plan, manifest_sha256="f" * 64)
    reopened = PredictionStore(store.root, run_id=revised.run_id, manifest_sha256=revised.manifest_sha256,
                               model_task="R0", model_binding_sha256=artifact.manifest_sha256, matrix_id=revised.matrix_id,
                               task_execution_binding_sha256=task_binding)
    outcome = PredictionWorker(revised, reopened, loader=_Loader(), vad=_Vad(), vau=_Vau(), model=artifact).run()
    assert outcome["completed"] == 0 and outcome["skipped"] == 3


def test_v2_actual_registry_promotion_preserves_partial_r0_resume(tmp_path):
    """Promoting A:17 creates a fresh admitted plan without rerunning R0."""
    manifest, digest, models, _ = _v2_fixture(tmp_path)
    first = load_prediction_plan(manifest, expected_sha256=digest)
    r0_path, r0_hash = models[("R0", None)]
    artifact = load_model_artifact(r0_path, expected_sha256=r0_hash, task=first.selected_model("R0", None))
    task_hash = prediction_inputs.prediction_task_execution_binding_sha256(matrix_id=first.matrix_id, task=artifact)
    store = PredictionStore(first.output_root / "R0", run_id=first.run_id, manifest_sha256=first.manifest_sha256,
                            model_task="R0", model_binding_sha256=artifact.manifest_sha256, matrix_id=first.matrix_id,
                            task_execution_binding_sha256=task_hash)
    assert PredictionWorker(first, store, loader=_Loader(), vad=_Vad(), vau=_Vau(), model=artifact).run()["succeeded"] == 3
    revised = json.loads(manifest.read_text())
    promoted = next(row for row in revised["model_tasks"] if row["group"] == "A" and row["seed"] == 17)
    promoted.update(state="READY", model_manifest=str(models[("A", 17)][0]), model_manifest_sha256=models[("A", 17)][1], pending_reason=None)
    registration = {key: revised[key] for key in ("schema", "run_id", "identity_manifest", "identity_manifest_sha256", "model_tasks", "protocol", "output_root", "denominators", "bindings")}
    identities = json.loads(Path(registration["identity_manifest"]).read_text())
    binding = prediction_execution_binding_sha256(bindings=registration["bindings"], protocol=registration["protocol"],
                                                  identity_manifest_sha256=registration["identity_manifest_sha256"], identities=identities,
                                                  model_registry=registration["model_tasks"])
    candidate = tmp_path / "promoted-candidate.json"
    candidate_hash = _write(candidate, {"schema": CANDIDATE_SCHEMA_V2, "matrix_id": first.matrix_id, "registration": registration,
                                        "prediction_execution_binding_sha256": binding})
    admission = tmp_path / "promoted-admission.json"
    _write(admission, {"status": "PASS", "formal_execution_allowed": True,
                       "embedded_vision_binding_sha256": registration["bindings"]["embedded_vision_binding_sha256"],
                       "prediction_execution_binding_sha256": binding, "prediction_plan_candidate_sha256": candidate_hash,
                       "resource_admission_sha256": registration["bindings"]["resource_admission_sha256"]})
    revised.update(candidate={"path": str(candidate), "sha256": candidate_hash},
                   admission={"formal_admission": str(admission), "formal_admission_sha256": _digest(admission)})
    promoted_path = tmp_path / "promoted-plan.json"; promoted_hash = _write(promoted_path, revised)
    resumed = load_prediction_plan(promoted_path, expected_sha256=promoted_hash)
    reopened = PredictionStore(store.root, run_id=resumed.run_id, manifest_sha256=resumed.manifest_sha256,
                               model_task="R0", model_binding_sha256=artifact.manifest_sha256, matrix_id=resumed.matrix_id,
                               task_execution_binding_sha256=task_hash)
    result = PredictionWorker(resumed, reopened, loader=_Loader(), vad=_Vad(), vau=_Vau(), model=artifact).run()
    assert resumed.selected_model("A", 17).state == "READY" and result["completed"] == 0 and result["skipped"] == 3


def test_v2_plan_builder_publishes_candidate_then_requires_bound_pass_admission(tmp_path):
    _, _, _, registration = _v2_fixture(tmp_path / "fixture")
    registration_path = tmp_path / "registration.json"; _write(registration_path, registration)
    root = Path(__file__).resolve().parents[1]
    candidate = tmp_path / "candidate.json"
    result = subprocess.run([sys.executable, str(root / "scripts/nc_rted_build_prediction_plan.py"), "candidate",
                             "--registration", str(registration_path), "--output", str(candidate)], text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    candidate_output = json.loads(result.stdout)
    admission = tmp_path / "admission.json"
    _write(admission, {"status": "PASS", "formal_execution_allowed": True,
                       "embedded_vision_binding_sha256": registration["bindings"]["embedded_vision_binding_sha256"],
                       "prediction_execution_binding_sha256": candidate_output["prediction_execution_binding_sha256"],
                       "prediction_plan_candidate_sha256": candidate_output["candidate_sha256"],
                       "resource_admission_sha256": registration["bindings"]["resource_admission_sha256"]})
    plan = tmp_path / "plan.json"
    result = subprocess.run([sys.executable, str(root / "scripts/nc_rted_build_prediction_plan.py"), "finalize",
                             "--candidate", str(candidate), "--formal-admission", str(admission), "--output", str(plan)],
                            text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    loaded = load_prediction_plan(plan, expected_sha256=json.loads(result.stdout)["manifest_sha256"])
    assert loaded.matrix_id == candidate_output["matrix_id"]


def test_v2_builder_rejects_incomplete_formal_denominators_and_bare_resource(tmp_path):
    _, _, _, registration = _v2_fixture(tmp_path)
    root = Path(__file__).resolve().parents[1]
    registration["run_id"] = "formal:blind"
    registration_path = tmp_path / "registration.json"; _write(registration_path, registration)
    command = [sys.executable, str(root / "scripts/nc_rted_build_prediction_plan.py"), "candidate",
               "--registration", str(registration_path), "--output", str(tmp_path / "candidate.json")]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    assert result.returncode == 2 and "251/800/3339" in result.stderr
    registration["denominators"] = {"ucf": 251, "xd": 800, "vau": 3339}
    _write(registration_path, registration)
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    assert result.returncode == 2 and "resource admission" in result.stderr
    registration["denominators"] = {"ucf": 251.0, "xd": 800, "vau": 3339}
    _write(registration_path, registration)
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    assert result.returncode == 2 and "251/800/3339" in result.stderr


def test_resource_qualification_rejects_limit_drift(tmp_path):
    scope = {"kind": "blind_prediction", "run_id": "formal:blind", "execution_scope_sha256": "a" * 64,
             "device": "cuda:1", "physical_gpu_uuid": "GPU-1", "data_volume": str(tmp_path),
             "min_free_bytes": 20 * 1024**3, "run_budget_seconds": 60, "deadline_utc_epoch": time.time() + 600}
    allocation = {"schema": RESOURCE_ALLOCATION_SCHEMA_V1, "status": "ACCEPTED", "kind": "blind_prediction",
                  "execution_scope_sha256": scope["execution_scope_sha256"], "allocation_id": "alloc", "host": "host",
                  "device": "cuda:1", "physical_gpu_uuid": "GPU-1", "lease_id": "lease", "authorization_sha256": "c" * 64,
                  "limits": {key: scope[key] for key in ("data_volume", "min_free_bytes", "run_budget_seconds", "deadline_utc_epoch")}}
    allocation_path = tmp_path / "allocation.json"; allocation_hash = _write(allocation_path, allocation)
    measurement = {"schema": RESOURCE_MEASUREMENT_SCHEMA_V1, "status": "MEASURED", "kind": "blind_prediction",
                   "execution_scope_sha256": scope["execution_scope_sha256"], "allocation_sha256": allocation_hash, "host": "host", "physical_gpu_uuid": "GPU-1",
                   "workload": {"kind": "blind_prediction", "vad_queries": 135050, "slow_triggers": 47458, "vau_requests": 3339},
                   "measured_at_utc_epoch": time.time(), "qualified_runtime_seconds": 1.0,
                   "peak_cuda_allocated_bytes": 1, "peak_cuda_reserved_bytes": 2}
    measurement_path = tmp_path / "measurement.json"; measurement_hash = _write(measurement_path, measurement)
    admission = tmp_path / "admission.json"
    _write(admission, {"schema": RESOURCE_ADMISSION_SCHEMA_V1, "status": "PASS", "formal_execution_allowed": True,
                       "scope": scope, "accepted_allocation": {"path": str(allocation_path), "sha256": allocation_hash},
                       "accepted_measurement": {"path": str(measurement_path), "sha256": measurement_hash}})
    assert prediction_inputs._validate_resource_admission(str(admission), _digest(admission), scope_sha256="a" * 64)["device"] == "cuda:1"
    measurement["peak_cuda_reserved_bytes"] = 0
    measurement_hash = _write(measurement_path, measurement)
    admission_document = json.loads(admission.read_text()); admission_document["accepted_measurement"]["sha256"] = measurement_hash; _write(admission, admission_document)
    with pytest.raises(PredictionInputError, match="accepted resource measurement"):
        prediction_inputs._validate_resource_admission(str(admission), _digest(admission), scope_sha256="a" * 64)


def test_resource_execution_requires_operator_registry_and_substantive_evidence(tmp_path):
    scope = {"kind": "blind_prediction", "run_id": "formal:blind", "execution_scope_sha256": "a" * 64, "device": "cuda:0",
             "physical_gpu_uuid": "GPU-1", "data_volume": str(tmp_path), "min_free_bytes": 20 * 1024**3,
             "run_budget_seconds": 60, "deadline_utc_epoch": time.time() + 600}
    limits = {key: scope[key] for key in ("data_volume", "min_free_bytes", "run_budget_seconds", "deadline_utc_epoch")}
    authorization = {"schema": prediction_inputs.RESOURCE_AUTHORIZATION_SCHEMA_V1, "status": "PASS", "host": "host", "physical_gpu_uuid": "GPU-1", "lease_id": "lease",
                     "lease_expires_utc_epoch": time.time() + 700, "project_volume": str(tmp_path), "max_budget_seconds": 60, "min_free_bytes": 20 * 1024**3,
                     "deadline_utc_epoch": scope["deadline_utc_epoch"], "capacity_bytes": 100}
    authorization_path = tmp_path / "authorization.json"; authorization_hash = _write(authorization_path, authorization)
    allocation = {"schema": RESOURCE_ALLOCATION_SCHEMA_V1, "status": "ACCEPTED", "kind": "blind_prediction", "execution_scope_sha256": "a" * 64,
                  "allocation_id": "alloc", "host": "host", "device": "cuda:0", "physical_gpu_uuid": "GPU-1", "lease_id": "lease", "authorization_sha256": authorization_hash, "limits": limits}
    allocation_path = tmp_path / "allocation.json"; allocation_hash = _write(allocation_path, allocation)
    measurement = {"schema": RESOURCE_MEASUREMENT_SCHEMA_V1, "status": "MEASURED", "kind": "blind_prediction", "execution_scope_sha256": "a" * 64,
                   "allocation_sha256": allocation_hash, "host": "host", "physical_gpu_uuid": "GPU-1", "workload": {"kind": "blind_prediction", "vad_queries": 135050, "slow_triggers": 47458, "vau_requests": 3339},
                   "measured_at_utc_epoch": time.time(), "qualified_runtime_seconds": 1.0, "peak_cuda_allocated_bytes": 1, "peak_cuda_reserved_bytes": 2}
    measurement_path = tmp_path / "measurement.json"; measurement_hash = _write(measurement_path, measurement)
    registry = {"schema": prediction_inputs.RESOURCE_EVIDENCE_REGISTRY_SCHEMA_V1, "accepted": {"resource_authorization": {"path": str(authorization_path), "sha256": authorization_hash},
                "resource_allocation": {"path": str(allocation_path), "sha256": allocation_hash}, "resource_measurement": {"path": str(measurement_path), "sha256": measurement_hash}}}
    registry_path = tmp_path / "operator-registry.json"; registry_hash = _write(registry_path, registry)
    prediction_inputs.validate_resource_execution_evidence(scope, registry_path=str(registry_path), registry_sha256=registry_hash, host="host", physical_gpu_uuid="GPU-1", capacity_bytes=100)
    # Rehashing self-declared records does not add them to the operator registry.
    measurement["peak_cuda_allocated_bytes"] = 0; forged_hash = _write(measurement_path, measurement)
    with pytest.raises(PredictionInputError, match="accepted resource_measurement"):
        prediction_inputs.validate_resource_execution_evidence(scope, registry_path=str(registry_path), registry_sha256=registry_hash, host="host", physical_gpu_uuid="GPU-1", capacity_bytes=100)
    assert forged_hash != measurement_hash


def test_execution_resource_rejects_device_uuid_and_expiry(monkeypatch, tmp_path):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("prediction_entrypoint", root / "scripts/nc_rted_predict.py")
    entrypoint = importlib.util.module_from_spec(spec); assert spec.loader is not None; spec.loader.exec_module(entrypoint)
    output = entrypoint._APPROVED_DATA_VOLUME / ".cache" / "prediction-resource-test"
    scope = {"device": "cuda:1", "physical_gpu_uuid": "GPU-1", "data_volume": str(entrypoint._APPROVED_DATA_VOLUME),
             "min_free_bytes": 1, "run_budget_seconds": 60, "deadline_utc_epoch": time.time() + 60}
    plan = SimpleNamespace(resource_scope=scope, output_root=output)
    monkeypatch.setattr(entrypoint, "_physical_gpu_uuid", lambda device, **kwargs: "GPU-1")
    with pytest.raises(ValueError, match="requested device"):
        entrypoint._admit_execution_resource(plan, "cuda:0")
    scope["deadline_utc_epoch"] = time.time() - 1
    with pytest.raises(ValueError, match="expired"):
        entrypoint._admit_execution_resource(plan, "cuda:1")
    scope["deadline_utc_epoch"] = time.time() + 60
    monkeypatch.setattr(entrypoint, "_physical_gpu_uuid", lambda device, **kwargs: "GPU-other")
    with pytest.raises(ValueError, match="physical GPU"):
        entrypoint._admit_execution_resource(plan, "cuda:1")


def _prediction_entrypoint():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("prediction_entrypoint_resource", root / "scripts/nc_rted_predict.py")
    entrypoint = importlib.util.module_from_spec(spec); assert spec.loader is not None; spec.loader.exec_module(entrypoint)
    return entrypoint


def test_admitted_gpu_uuid_resolves_masked_and_reordered_visible_devices():
    entrypoint = _prediction_entrypoint()
    inventory = "0, GPU-zero\n1, GPU-one\n"
    assert entrypoint._physical_gpu_uuid("cuda:0", visible="1,0", inventory=inventory) == "GPU-one"
    assert entrypoint._physical_gpu_uuid("cuda:1", visible="1,0", inventory=inventory) == "GPU-zero"
    assert entrypoint._physical_gpu_uuid("cuda:0", visible="GPU-one", inventory=inventory) == "GPU-one"


def test_resource_admission_rechecks_expiry_after_uuid_lookup(monkeypatch):
    entrypoint = _prediction_entrypoint()
    scope = {"device": "cuda:0", "physical_gpu_uuid": "GPU-1", "data_volume": str(entrypoint._APPROVED_DATA_VOLUME),
             "min_free_bytes": 1, "run_budget_seconds": 60, "deadline_utc_epoch": time.time() + .02}
    def delayed_uuid(device, **kwargs):
        time.sleep(.03)
        return "GPU-1"
    monkeypatch.setattr(entrypoint, "_physical_gpu_uuid", delayed_uuid)
    with pytest.raises(ValueError, match="deadline has expired"):
        entrypoint._admit_execution_resource(SimpleNamespace(resource_scope=scope,
                                                             output_root=entrypoint._APPROVED_DATA_VOLUME / ".cache" / "prediction-resource-test"), "cuda:0")


@pytest.mark.parametrize("stage", ("model_initialization", "prediction"))
def test_watchdog_expires_during_model_initialization_or_prediction(stage):
    entrypoint = _prediction_entrypoint()
    def native_operation():
        # This simulates a native/CUDA call: ordinary Python exceptions are
        # caught internally, so only the parent watchdog can enforce expiry.
        try:
            while True:
                time.sleep(.05)
        except ValueError:
            return {"stage": stage, "escaped": True}
    with pytest.raises(ValueError, match="runtime budget expired"):
        entrypoint._watchdog(time.time() + .15, native_operation)


def test_watchdog_kills_native_style_work_that_catches_valueerror():
    entrypoint = _prediction_entrypoint()
    def catches_valueerror():
        try:
            while True:
                time.sleep(.05)
        except ValueError:
            return {"escaped": True}
    with pytest.raises(ValueError, match="runtime budget expired"):
        entrypoint._watchdog(time.time() + .15, catches_valueerror)


def test_watchdog_reaps_descendant_that_keeps_result_pipe_open():
    entrypoint = _prediction_entrypoint()
    def leaves_descendant():
        if os.fork() == 0:
            time.sleep(5)
            os._exit(0)
        return {"done": True}
    started = time.monotonic()
    assert entrypoint._watchdog(time.time() + .5, leaves_descendant) == {"done": True}
    assert time.monotonic() - started < 1


def test_budget_control_paths_reject_symlinks_without_touching_target(tmp_path):
    entrypoint = _prediction_entrypoint()
    volume, outside = tmp_path / "volume", tmp_path / "outside"
    volume.mkdir(); outside.write_text("preserve")
    root = volume / "out"; root.mkdir()
    (root / ".resource-budget.pending").symlink_to(outside)
    with pytest.raises(ValueError, match="control path"):
        entrypoint._write_budget_ledger(root / ".resource-budget.pending", {"x": 1}, volume, 1)
    assert outside.read_text() == "preserve"
    (root / ".resource-budget.lock").symlink_to(outside)
    plan = SimpleNamespace(resource_scope={"data_volume": str(volume), "min_free_bytes": 1, "run_budget_seconds": 1}, output_root=root,
                           execution_scope_sha256="a" * 64, run_id="run")
    with pytest.raises(ValueError, match="control path"):
        with entrypoint._cumulative_budget(plan, time.time() + 1): pass
    assert outside.read_text() == "preserve"


def test_cumulative_budget_debits_an_abandoned_reservation(tmp_path):
    entrypoint = _prediction_entrypoint()
    scope = {"data_volume": str(tmp_path), "min_free_bytes": 1, "run_budget_seconds": 5}
    plan = SimpleNamespace(resource_scope=scope, output_root=tmp_path / "out", execution_scope_sha256="a" * 64, run_id="run")
    plan.output_root.mkdir()
    (plan.output_root / ".resource-budget.json").write_text(json.dumps({"schema": "nc_rted_prediction_budget/v1", "scope": "a" * 64,
        "run_id": "run", "consumed_seconds": 0.0, "active": {"reserved_seconds": 5.0, "started_utc_epoch": time.time()}}))
    with pytest.raises(ValueError, match="cumulative"):
        with entrypoint._cumulative_budget(plan, time.time() + 30):
            pass


def test_task_output_and_store_reject_symlink_escape_and_reserve_depletion(tmp_path, monkeypatch):
    entrypoint = _prediction_entrypoint()
    volume, outside = tmp_path / "volume", tmp_path / "outside"
    volume.mkdir(); outside.mkdir()
    root = volume / "predictions"; root.mkdir(); (root / "R0").symlink_to(outside, target_is_directory=True)
    scope = {"data_volume": str(volume), "min_free_bytes": 1}
    with pytest.raises(ValueError, match="cannot be a symlink"):
        entrypoint._task_output_root(SimpleNamespace(output_root=root), "R0", scope)
    store = PredictionStore(volume / "safe", run_id="run", manifest_sha256="a" * 64, model_task="R0", model_binding_sha256="b" * 64,
                            storage_volume=volume, min_free_bytes=1)
    monkeypatch.setattr(prediction_store.shutil, "disk_usage", lambda path: SimpleNamespace(free=1))
    with pytest.raises(PredictionStoreError, match="storage reserve"):
        store.publish(identity="vad:x", attempt=1, status="success", provenance={}, payload={})


def test_task_output_rejects_different_mount(monkeypatch, tmp_path):
    entrypoint = _prediction_entrypoint()
    volume = tmp_path / "volume"; root = volume / "out"; root.mkdir(parents=True)
    monkeypatch.setattr(entrypoint, "_mount_identity", lambda path: (1, "/admitted") if Path(path).resolve() == volume.resolve() else (2, "/redirected"))
    with pytest.raises(ValueError, match="filesystem or mount"):
        entrypoint._task_output_root(SimpleNamespace(output_root=root), "R0", {"data_volume": str(volume), "min_free_bytes": 1})


def test_runtime_probe_only_selects_one_hash_bound_blind_identity(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("runtime_probe", root / "scripts/nc_rted_prediction_runtime_probe.py")
    probe = importlib.util.module_from_spec(spec); assert spec.loader is not None; spec.loader.exec_module(probe)
    monkeypatch.setattr(probe, "VadRequest", VadRequest, raising=False)
    monkeypatch.setattr(probe, "VauRequest", VauRequest, raising=False)
    monkeypatch.setattr(probe, "_bound_path", _bound_path, raising=False)
    medium = tmp_path / "clip.mp4"; medium.write_bytes(b"blind")
    identity = {"vad": [{"dataset": "ucf", "id": "one", "media_path": str(medium), "media_sha256": _digest(medium)}],
                "vau": [{"id": "two", "media_path": str(medium), "media_sha256": _digest(medium), "question": "Describe the clip."}]}
    request = probe._request(identity, "vad:ucf:one", "vad")
    assert request.identity == "vad:ucf:one"
    assert probe._protocol({"protocols": {"hivau": {"target_fps": 4}, "vad_config": {"target_fps": 4, "query_interval": 4, "batch_size": 1, "fusion": "adaptive"}}}) == {
        "hivau": {"target_fps": 4}, "vad": {"target_fps": 4, "query_interval": 4, "batch_size": 1},
        "vad_route": "reactvau_detection", "vad_causal_smoothing": "online", "vad_fast_fusion": "adaptive"}
    with pytest.raises(probe.ProbeError, match="IDENTITY_INVALID"):
        probe._request(identity, "vau:two", "vad")

def test_trained_model_rejects_nonfinal_or_wrong_provenance_attestation(tmp_path):
    manifest, digest, model_manifests = _fixture(tmp_path)
    plan = load_prediction_plan(manifest, expected_sha256=digest)
    path, _ = model_manifests[("F", 2026)]
    artifact = json.loads(path.read_text())
    attestation = Path(artifact["final_checkpoint_attestation"])
    document = json.loads(attestation.read_text()); document["completed_updates"] = 950
    _write(attestation, document); artifact["final_checkpoint_attestation_sha256"] = _digest(attestation); _write(path, artifact)
    with pytest.raises(PredictionInputError, match="1000-update final"):
        load_model_artifact(path, expected_sha256=_digest(path), task=plan.selected_model("F", 2026))
    manifest, digest, model_manifests = _fixture(tmp_path / "checkpoint-supervision")
    plan = load_prediction_plan(manifest, expected_sha256=digest)
    path, _ = model_manifests[("F", 2026)]
    artifact = json.loads(path.read_text()); checkpoint = Path(artifact["checkpoint"]) / "manifest.json"
    document = json.loads(checkpoint.read_text()); document["nested"] = {"labels": ["forbidden"]}
    _write(checkpoint, document)
    artifact["checkpoint_manifest_sha256"] = _digest(checkpoint)
    attestation = Path(artifact["final_checkpoint_attestation"]); attested = json.loads(attestation.read_text())
    attested["checkpoint_manifest_sha256"] = _digest(checkpoint); _write(attestation, attested)
    provenance = Path(artifact["accepted_training_provenance"]); accepted = json.loads(provenance.read_text())
    accepted["checkpoint_manifest_sha256"] = _digest(checkpoint); _write(provenance, accepted)
    artifact["final_checkpoint_attestation_sha256"] = _digest(attestation)
    artifact["accepted_training_provenance_sha256"] = _digest(provenance); _write(path, artifact)
    with pytest.raises(PredictionInputError, match="forbidden supervision"):
        load_model_artifact(path, expected_sha256=_digest(path), task=plan.selected_model("F", 2026))
    manifest, digest, model_manifests = _fixture(tmp_path / "provenance")
    plan = load_prediction_plan(manifest, expected_sha256=digest)
    path, _ = model_manifests[("F", 2026)]
    artifact = json.loads(path.read_text())
    provenance = Path(artifact["accepted_training_provenance"])
    document = json.loads(provenance.read_text()); document["training_identity"]["code_sha256"] = "f" * 64
    _write(provenance, document); artifact["accepted_training_provenance_sha256"] = _digest(provenance); _write(path, artifact)
    with pytest.raises(PredictionInputError, match="1000-update final"):
        load_model_artifact(path, expected_sha256=_digest(path), task=plan.selected_model("F", 2026))


def test_plan_requires_formal_admission_for_exact_vision_binding(tmp_path):
    manifest, _, _ = _fixture(tmp_path)
    document = json.loads(manifest.read_text())
    admission = Path(document["admission"]["formal_admission"])
    _write(admission, {"status": "PASS", "formal_execution_allowed": True,
                       "embedded_vision_binding_sha256": "0" * 64})
    document["admission"]["formal_admission_sha256"] = _digest(admission)
    manifest.write_text(json.dumps(document))
    with pytest.raises(PredictionInputError, match="embedded vision"):
        load_prediction_plan(manifest)


def test_plan_rejects_supervision_in_formal_admission(tmp_path):
    manifest, _, _ = _fixture(tmp_path)
    document = json.loads(manifest.read_text())
    admission = Path(document["admission"]["formal_admission"])
    row = json.loads(admission.read_text()); row["nested"] = {"answer": "forbidden"}
    _write(admission, row); document["admission"]["formal_admission_sha256"] = _digest(admission); manifest.write_text(json.dumps(document))
    with pytest.raises(PredictionInputError, match="forbidden supervision"):
        load_prediction_plan(manifest)


def test_plan_admission_binds_ordered_identity_manifest(tmp_path):
    manifest, _, _ = _fixture(tmp_path)
    document = json.loads(manifest.read_text())
    identity = Path(document["identity_manifest"])
    rows = json.loads(identity.read_text())
    rows["vad"].reverse()
    _write(identity, rows)
    document["identity_manifest_sha256"] = _digest(identity)
    manifest.write_text(json.dumps(document))
    with pytest.raises(PredictionInputError, match="formal admission"):
        load_prediction_plan(manifest)


def test_implementation_manifest_attests_exact_current_prediction_sources(tmp_path):
    root = Path(__file__).resolve().parents[1]
    implementation = {"schema": IMPLEMENTATION_SCHEMA, "root": str(root),
                      "files": {relative: _digest(root / relative) for relative in _IMPLEMENTATION_FILES}}
    path = tmp_path / "implementation.json"
    digest = _write(path, implementation)
    verify_implementation_manifest(path, digest)
    implementation["files"]["src/nc_rted/prediction_worker.py"] = "0" * 64
    digest = _write(path, implementation)
    with pytest.raises(PredictionInputError, match="prediction implementation source"):
        verify_implementation_manifest(path, digest)
    other_root = tmp_path / "other-checkout"; other_root.mkdir()
    implementation["files"]["src/nc_rted/prediction_worker.py"] = _digest(root / "src/nc_rted/prediction_worker.py")
    implementation["root"] = str(other_root)
    digest = _write(path, implementation)
    with pytest.raises(PredictionInputError, match="file set differs"):
        verify_implementation_manifest(path, digest)
    with pytest.raises(ValueError):
        canonical_json({"not_a_number": float("nan")})


def test_implementation_manifest_rejects_foreign_cached_prediction_module(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    implementation = {"schema": IMPLEMENTATION_SCHEMA, "root": str(root),
                      "files": {relative: _digest(root / relative) for relative in _IMPLEMENTATION_FILES}}
    path = tmp_path / "implementation.json"; digest = _write(path, implementation)
    foreign = tmp_path / "foreign.py"; foreign.write_text("# foreign\n")
    module = types.ModuleType("foreign_worker"); module.__file__ = str(foreign)
    monkeypatch.setitem(sys.modules, "nc_rted.prediction_worker", module)
    with pytest.raises(PredictionInputError, match="escapes admitted|differs from admitted source"):
        verify_implementation_manifest(path, digest)


def test_inherited_import_rejects_undeclared_transitive_source_before_execution(tmp_path, monkeypatch):
    root = tmp_path / "external"
    sentinel = tmp_path / "foreign-import-ran"
    original_modules = {name for name in sys.modules if name == "llava" or name.startswith("llava.") or name == "eval_utils" or name.startswith("eval_utils.")}
    monkeypatch.setattr(sys, "path", list(sys.path))
    sources = {
        "llava/__init__.py": "",
        "llava/train/__init__.py": "",
        "llava/train/train.py": "from . import foreign\n",
        "llava/train/foreign.py": f"from pathlib import Path\nPath({str(sentinel)!r}).write_text('ran')\n",
    }
    files = {}
    for relative, text in sources.items():
        path = root / relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(text); files[relative] = _digest(path)
    # The foreign child is deliberately present on disk but absent from the
    # admitted manifest. It must fail during resolution, before its body runs.
    files.pop("llava/train/foreign.py")
    source_manifest = tmp_path / "source-manifest.json"; _write(source_manifest, {"files": files})
    for name in tuple(sys.modules):
        if name == "llava" or name.startswith("llava.") or name == "eval_utils" or name.startswith("eval_utils."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    runtime = PredictionRuntime(tmp_path / "runtime.json", "a" * 64,
                                {"inherited": {"external_root": str(root), "source_manifest": str(source_manifest)}})
    with pytest.raises(PredictionInputError, match="complete source manifest"):
        _import_bound_runtime(runtime)
    assert not sentinel.exists()
    for name in tuple(sys.modules):
        if (name == "llava" or name.startswith("llava.") or name == "eval_utils" or name.startswith("eval_utils.")) and name not in original_modules:
            sys.modules.pop(name)


def test_inherited_import_executes_verified_source_not_cached_bytecode(tmp_path, monkeypatch):
    root = tmp_path / "external"; good, bad = tmp_path / "good", tmp_path / "bad"
    sources = {"llava/__init__.py": "", "llava/train/__init__.py": "",
               "llava/train/train.py": f"from pathlib import Path\nPath({str(good)!r}).write_text('source')\n"}
    files = {}
    for relative, text in sources.items():
        path = root / relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(text); files[relative] = _digest(path)
    source = root / "llava/train/train.py"
    malicious = compile(f"from pathlib import Path\nPath({str(bad)!r}).write_text('bytecode')\n", str(source), "exec")
    pyc = importlib._bootstrap_external._code_to_timestamp_pyc(malicious, int(source.stat().st_mtime), source.stat().st_size)
    cached = source.parent / "__pycache__" / f"train.{sys.implementation.cache_tag}.pyc"; cached.parent.mkdir(); cached.write_bytes(pyc)
    source_manifest = tmp_path / "source-manifest.json"; _write(source_manifest, {"files": files})
    original_modules = {name for name in sys.modules if name == "llava" or name.startswith("llava.") or name == "eval_utils" or name.startswith("eval_utils.")}
    monkeypatch.setattr(sys, "path", list(sys.path))
    for name in original_modules: monkeypatch.delitem(sys.modules, name, raising=False)
    runtime = PredictionRuntime(tmp_path / "runtime.json", "a" * 64,
                                {"inherited": {"external_root": str(root), "source_manifest": str(source_manifest)}})
    with pytest.raises(PredictionInputError):
        _import_bound_runtime(runtime)
    assert good.read_text() == "source" and not bad.exists()
    for name in tuple(sys.modules):
        if (name == "llava" or name.startswith("llava.") or name == "eval_utils" or name.startswith("eval_utils.")) and name not in original_modules:
            sys.modules.pop(name)


def test_inherited_import_rejects_preloaded_module_without_verified_execution(tmp_path, monkeypatch):
    root = tmp_path / "external"; source = root / "llava" / "__init__.py"; source.parent.mkdir(parents=True); source.write_text("")
    module = types.ModuleType("llava"); module.__file__ = str(source)
    monkeypatch.setitem(sys.modules, "llava", module)
    runtime = PredictionRuntime(tmp_path / "runtime.json", "a" * 64,
                                {"inherited": {"external_root": str(root), "source_manifest": str(tmp_path / "unused.json")}})
    with pytest.raises(PredictionInputError, match="verified execution provenance"):
        _audit_bound_inherited_modules(str(root), {"llava/__init__.py": _digest(source)})


def test_inherited_import_rejects_foreign_detect_utils_alias(tmp_path, monkeypatch):
    root = tmp_path / "external"; source = root / "eval_utils" / "vad" / "detect_utils.py"
    source.parent.mkdir(parents=True); source.write_text("BOUND = True\n")
    foreign = tmp_path / "foreign_detect_utils.py"; foreign.write_text("SENTINEL = True\n")
    module = types.ModuleType("detect_utils"); module.__file__ = str(foreign)
    monkeypatch.setitem(sys.modules, "detect_utils", module)
    with pytest.raises(PredictionInputError, match="escapes bound source root"):
        _audit_bound_inherited_modules(str(root), {"eval_utils/vad/detect_utils.py": _digest(source)})


def test_blind_prompt_tokenizer_preserves_open_assistant_turn_and_time_message():
    class Conversation:
        roles = ("user", "assistant")
        def __init__(self): self.messages = []
        def copy(self): return Conversation()
        def append_message(self, role, content): self.messages.append((role, content))
        def get_prompt(self): return repr(self.messages)
    received = {}
    def tokenize(prompt, tokenizer, image_index, return_tensors):
        received.update(prompt=prompt, tokenizer=tokenizer, image_index=image_index, return_tensors=return_tensors)
        return torch.tensor([1, image_index, 2])
    tokenizer = SimpleNamespace(pad_token_id=0)
    data_args = SimpleNamespace(is_multimodal=True)
    prompt = BlindPromptTokenizer(tokenizer, data_args, {"qwen_2": Conversation()}, tokenize, "<image>", -200)
    context = SimpleNamespace(time_message="The video has been observed for 4 seconds. ",
                              visual_embeddings=torch.zeros(1, 1), images="frames", image_sizes="sizes")
    inputs = prompt.encode(question="What happened?", context=context)
    assert received == {"prompt": "[('user', '<image>\\nThe video has been observed for 4 seconds. What happened?'), ('assistant', None)]",
                        "tokenizer": tokenizer, "image_index": -200, "return_tensors": "pt"}
    assert inputs.input_ids.tolist() == [[1, -200, 2]]
    assert inputs.attention_mask.tolist() == [[1, 1, 1]]


def test_store_atomic_immutable_and_retryable_failure(tmp_path):
    store = PredictionStore(tmp_path / "out", run_id="run", manifest_sha256="a" * 64, model_task="A:seed17", model_binding_sha256="b" * 64)
    err = RuntimeError("transient")
    deadline = datetime.now(timezone.utc) + timedelta(minutes=5)
    failed = store.publish(identity="vau:q", attempt=1, status="failure", provenance={"stage": "vau"}, error=err,
                           retryable=True, retry_not_before=deadline)
    assert failed["failure"]["exception_class"] == "RuntimeError"
    assert not store.should_run(identity="vau:q", max_retries=3, now=deadline - timedelta(seconds=1))
    assert store.should_run(identity="vau:q", max_retries=3, now=deadline + timedelta(seconds=1))
    good = store.publish(identity="vau:q", attempt=2, status="success", provenance={}, payload={"text": "all", "token_ids": [1]})
    assert good["status"] == "success" and not store.should_run(identity="vau:q", max_retries=3)
    with pytest.raises(PredictionStoreError):
        store.publish(identity="vau:q", attempt=3, status="success", provenance={}, payload={})
    with pytest.raises(PredictionStoreError, match="index schema"):
        PredictionStore(tmp_path / "out", run_id="run", manifest_sha256="a" * 64, model_task="A:seed17", model_binding_sha256="c" * 64)


def test_store_recovers_committed_record_after_index_publication_failure(tmp_path, monkeypatch):
    store = PredictionStore(tmp_path / "out", run_id="run", manifest_sha256="a" * 64, model_task="A:seed17", model_binding_sha256="b" * 64)
    original = prediction_store._write_atomic
    def fail_index(path, value):
        if path == store.index_path:
            raise OSError(errno.EIO, "index unavailable")
        return original(path, value)
    monkeypatch.setattr(prediction_store, "_write_atomic", fail_index)
    with pytest.raises(PredictionPublicationError, match="record committed"):
        store.publish(identity="vau:q", attempt=1, status="success", provenance={}, payload={"text": "ok", "token_ids": [1]})
    committed = store.records / f"{store.key(identity='vau:q')}.attempt1.json"
    before = committed.read_bytes()
    monkeypatch.setattr(prediction_store, "_write_atomic", original)
    recovered = PredictionStore(store.root, run_id="run", manifest_sha256="a" * 64, model_task="A:seed17", model_binding_sha256="b" * 64)
    assert recovered.get(identity="vau:q")["status"] == "success"
    assert not recovered.should_run(identity="vau:q", max_retries=3)
    with pytest.raises(PredictionStoreError, match="immutable"):
        recovered.publish(identity="vau:q", attempt=1, status="failure", provenance={}, error=RuntimeError("replacement"))
    assert committed.read_bytes() == before


def test_store_blocks_missing_or_changed_indexed_success_on_restart(tmp_path):
    store = PredictionStore(tmp_path / "missing", run_id="run", manifest_sha256="a" * 64, model_task="A:seed17", model_binding_sha256="b" * 64)
    store.publish(identity="vau:q", attempt=1, status="success", provenance={}, payload={"text": "ok", "token_ids": [1]})
    record_path = store.records / f"{store.key(identity='vau:q')}.attempt1.json"; record_path.unlink()
    with pytest.raises(PredictionStoreError, match="unreadable"):
        PredictionStore(store.root, run_id="run", manifest_sha256="a" * 64, model_task="A:seed17", model_binding_sha256="b" * 64)
    changed = PredictionStore(tmp_path / "changed", run_id="run", manifest_sha256="a" * 64, model_task="A:seed17", model_binding_sha256="b" * 64)
    changed.publish(identity="vau:q", attempt=1, status="success", provenance={}, payload={"text": "ok", "token_ids": [1]})
    record_path = changed.records / f"{changed.key(identity='vau:q')}.attempt1.json"
    record = json.loads(record_path.read_text()); record["payload"]["text"] = "changed"
    record["record_sha256"] = hashlib.sha256(canonical_json({key: value for key, value in record.items() if key != "record_sha256"})).hexdigest()
    record_path.write_text(json.dumps(record))
    with pytest.raises(PredictionStoreError, match="index record differs"):
        PredictionStore(changed.root, run_id="run", manifest_sha256="a" * 64, model_task="A:seed17", model_binding_sha256="b" * 64)


def test_store_blocks_noncanonical_indexed_success_on_restart(tmp_path):
    store = PredictionStore(tmp_path / "out", run_id="run", manifest_sha256="a" * 64, model_task="A:seed17", model_binding_sha256="b" * 64)
    store.publish(identity="vau:q", attempt=1, status="success", provenance={}, payload={"text": "ok", "token_ids": [1]})
    index = json.loads(store.index_path.read_text()); key = store.key(identity="vau:q")
    original = store.root / index["records"][key]["path"]; moved = store.root / "saved.json"; original.rename(moved)
    index["records"][key]["path"] = "saved.json"; store.index_path.write_text(json.dumps(index)); before = store.index_path.read_bytes()
    with pytest.raises(PredictionStoreError, match="canonical attempt"):
        PredictionStore(store.root, run_id="run", manifest_sha256="a" * 64, model_task="A:seed17", model_binding_sha256="b" * 64)
    assert store.index_path.read_bytes() == before


def test_worker_does_not_replace_success_when_index_publication_fails(tmp_path, monkeypatch):
    manifest, digest, model_manifests = _fixture(tmp_path)
    plan = load_prediction_plan(manifest, expected_sha256=digest)
    artifact_path, artifact_hash = model_manifests[("A", 17)]
    selected = load_model_artifact(artifact_path, expected_sha256=artifact_hash, task=plan.selected_model("A", 17))
    store = PredictionStore(plan.output_root / selected.task_id, run_id=plan.run_id, manifest_sha256=plan.manifest_sha256,
                            model_task=selected.task_id, model_binding_sha256=selected.manifest_sha256)
    original = prediction_store._write_atomic
    def fail_index(path, value):
        if path == store.index_path:
            raise OSError(errno.EIO, "index unavailable")
        return original(path, value)
    monkeypatch.setattr(prediction_store, "_write_atomic", fail_index)
    with pytest.raises(PredictionExecutionError, match="result committed"):
        PredictionWorker(plan, store, loader=_Loader(), vad=_Vad(), vau=_Vau(), model=selected).run()
    committed = store.records / f"{store.key(identity='vad:ucf:u1')}.attempt1.json"
    assert json.loads(committed.read_text())["status"] == "success"
    monkeypatch.setattr(prediction_store, "_write_atomic", original)
    recovered = PredictionStore(store.root, run_id=plan.run_id, manifest_sha256=plan.manifest_sha256,
                                model_task=selected.task_id, model_binding_sha256=selected.manifest_sha256)
    assert recovered.get(identity="vad:ucf:u1")["status"] == "success"


def test_store_rejects_record_with_rehashed_wrong_identity(tmp_path):
    store = PredictionStore(tmp_path / "out", run_id="run", manifest_sha256="a" * 64, model_task="A:seed17", model_binding_sha256="b" * 64)
    store.publish(identity="vau:q", attempt=1, status="success", provenance={}, payload={"text": "ok", "token_ids": [1]})
    index = json.loads(store.index_path.read_text())
    record_path = store.root / index["records"][store.key(identity="vau:q")]["path"]
    record = json.loads(record_path.read_text())
    record["identity"] = "vau:other"
    record["record_sha256"] = hashlib.sha256(canonical_json({key: value for key, value in record.items() if key != "record_sha256"})).hexdigest()
    record_path.write_text(json.dumps(record))
    index["records"][store.key(identity="vau:q")]["record_sha256"] = record["record_sha256"]
    store.index_path.write_text(json.dumps(index))
    with pytest.raises(PredictionStoreError, match="identity differs"):
        store.get(identity="vau:q")


def test_retry_classification_and_persisted_fixed_deadlines(tmp_path):
    assert PredictionWorker._retryable(OSError(errno.ETIMEDOUT, "timed out")) is True
    for code in (errno.EACCES, errno.EROFS, errno.ENOSPC, errno.EIO):
        assert PredictionWorker._retryable(OSError(code, "permanent")) is False
    store = PredictionStore(tmp_path / "out", run_id="run", manifest_sha256="a" * 64, model_task="A:seed17", model_binding_sha256="b" * 64)
    origin = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for attempt, delay in enumerate((5, 20, 60), start=1):
        deadline = origin + timedelta(minutes=delay)
        store.publish(identity="vad:v", attempt=attempt, status="failure", provenance={}, error=OSError(errno.ETIMEDOUT, "timed out"),
                      retryable=True, retry_not_before=deadline)
        assert not store.should_run(identity="vad:v", max_retries=3, now=deadline - timedelta(microseconds=1))
        assert store.should_run(identity="vad:v", max_retries=3, now=deadline)
    store.publish(identity="vad:v", attempt=4, status="failure", provenance={}, error=OSError(errno.ETIMEDOUT, "timed out"), retryable=False)
    assert not store.should_run(identity="vad:v", max_retries=3, now=origin + timedelta(days=1))


@dataclass
class _Model:
    group: str
    seed: int | None
    evidence_enabled: bool


class _Loader:
    def __init__(self): self.models = []
    def load(self, artifact):
        model = _Model(artifact.group, artifact.seed, artifact.evidence_enabled); self.models.append(model); return model


class _Vad:
    def predict(self, request, model, *, protocol):
        return {"queries": [{"query_index": 0, "frame_indices": [0], "fast_score": .1, "final_score": .1,
                             "triggered": False, "slow_score": None, "fused_score": None}],
                "causal_smoothed_scores": [.1], "total_frames": 1}


class _Vau:
    def __init__(self): self.questions = []
    def generate(self, request, model, *, protocol):
        assert protocol["hivau"]["target_fps"] == 4 and protocol["hivau"]["query_interval"] == 4
        self.questions.append(request.question)
        return {"text": request.question, "token_ids": [1, 2, 3]}


def test_worker_loads_one_selected_model_and_never_changes_question(tmp_path):
    manifest, digest, model_manifests = _fixture(tmp_path)
    plan = load_prediction_plan(manifest, expected_sha256=digest)
    artifact_path, artifact_hash = model_manifests[("A", 17)]
    selected = load_model_artifact(artifact_path, expected_sha256=artifact_hash, task=plan.selected_model("A", 17))
    store = PredictionStore(plan.output_root / selected.task_id, run_id=plan.run_id, manifest_sha256=plan.manifest_sha256, model_task=selected.task_id, model_binding_sha256=selected.manifest_sha256)
    loader, vau = _Loader(), _Vau()
    outcome = PredictionWorker(plan, store, loader=loader, vad=_Vad(), vau=vau, model=selected).run()
    assert len(loader.models) == 1 and loader.models[0].group == "A" and loader.models[0].seed == 17
    assert vau.questions == ["Describe this clip exactly."]
    assert outcome == {"completed": 3, "failed": 0, "skipped": 0, "expected": 3,
                       "succeeded": 3, "technical_failures": 0, "missing": 0}


def test_separate_model_tasks_receive_independent_slow_bound_objects(tmp_path):
    manifest, digest, model_manifests = _fixture(tmp_path)
    plan = load_prediction_plan(manifest, expected_sha256=digest)
    loader = _Loader()
    for seed in (17, 42):
        artifact_path, artifact_hash = model_manifests[("A", seed)]
        selected = load_model_artifact(artifact_path, expected_sha256=artifact_hash, task=plan.selected_model("A", seed))
        store = PredictionStore(plan.output_root / selected.task_id, run_id=plan.run_id, manifest_sha256=plan.manifest_sha256,
                                model_task=selected.task_id, model_binding_sha256=selected.manifest_sha256)
        assert PredictionWorker(plan, store, loader=loader, vad=_Vad(), vau=_Vau(), model=selected).run()["succeeded"] == 3
    assert len(loader.models) == 2 and loader.models[0] is not loader.models[1]


def test_test_run_entrypoint_completes_all_bound_rows(tmp_path):
    manifest, digest, model_manifests = _fixture(tmp_path)
    artifact_path, artifact_hash = model_manifests[("A", 17)]
    adapter = tmp_path / "entrypoint_adapter.py"
    adapter.write_text('''
class Model:
    def __init__(self, artifact):
        self.group, self.seed, self.evidence_enabled = artifact.group, artifact.seed, artifact.evidence_enabled
class Loader:
    def load(self, artifact): return Model(artifact)
class Vad:
    def predict(self, request, model, *, protocol):
        return {"queries": [{"query_index": 0, "frame_indices": [0], "fast_score": .2, "final_score": .2, "triggered": False, "slow_score": None, "fused_score": None}], "causal_smoothed_scores": [.2], "total_frames": 1}
class Vau:
    def generate(self, request, model, *, protocol): return {"text": request.question, "token_ids": [1]}
def factory(plan, model): return {"loader": Loader(), "vad": Vad(), "vau": Vau()}
''')
    environment = {**os.environ, "PYTHONPATH": f"{tmp_path}:{Path(__file__).resolve().parents[1] / 'src'}"}
    result = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / "scripts" / "nc_rted_predict.py"), "run",
                             "--manifest", str(manifest), "--manifest-sha256", digest, "--group", "A", "--seed", "17",
                             "--model-manifest", str(artifact_path), "--model-manifest-sha256", artifact_hash,
                             "--adapter-factory", "entrypoint_adapter:factory"], text=True, capture_output=True, env=environment, check=False)
    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout)
    assert output["status"] == "COMPLETE" and output["succeeded"] == 3


def test_default_cli_bootstrap_executes_verified_implementation_source_not_pyc(tmp_path):
    manifest, _, _ = _fixture(tmp_path)
    root = Path(__file__).resolve().parents[1]
    implementation = {"schema": IMPLEMENTATION_SCHEMA, "root": str(root),
                      "files": {relative: _digest(root / relative) for relative in _IMPLEMENTATION_FILES}}
    implementation_path = tmp_path / "implementation.json"; implementation_digest = _write(implementation_path, implementation)
    document = json.loads(manifest.read_text()); bindings = document["bindings"]
    bindings["implementation_manifest"] = str(implementation_path)
    bindings["implementation_manifest_sha256"] = implementation_digest
    identities = json.loads(Path(document["identity_manifest"]).read_text())
    admission = Path(document["admission"]["formal_admission"])
    _write(admission, {"status": "PASS", "formal_execution_allowed": True,
                       "embedded_vision_binding_sha256": bindings["embedded_vision_binding_sha256"],
                       "prediction_execution_binding_sha256": prediction_execution_binding_sha256(
                           bindings=bindings, protocol=document["protocol"],
                           identity_manifest_sha256=document["identity_manifest_sha256"], identities=identities)})
    document["admission"]["formal_admission_sha256"] = _digest(admission); manifest_digest = _write(manifest, document)
    worker = root / "src/nc_rted/prediction_worker.py"; sentinel = tmp_path / "malicious-pyc-ran"
    cached = Path(importlib.util.cache_from_source(str(worker))); previous = cached.read_bytes() if cached.exists() else None
    malicious = compile(f"from pathlib import Path\nPath({str(sentinel)!r}).write_text('bad')\n", str(worker), "exec")
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_bytes(importlib._bootstrap_external._code_to_timestamp_pyc(malicious, int(worker.stat().st_mtime), worker.stat().st_size))
    try:
        environment = {**os.environ, "PYTHONPATH": str(root / "src")}
        result = subprocess.run([sys.executable, str(root / "scripts/nc_rted_predict.py"), "plan", "--manifest", str(manifest),
                                 "--manifest-sha256", manifest_digest], text=True, capture_output=True, env=environment, check=False)
    finally:
        if previous is None: cached.unlink(missing_ok=True)
        else: cached.write_bytes(previous)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "PLAN_VALID" and not sentinel.exists()


def test_entrypoint_missing_manifest_is_structured_blocked(tmp_path):
    result = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / "scripts" / "nc_rted_predict.py"), "plan",
                             "--manifest", str(tmp_path / "missing.json")], text=True, capture_output=True, check=False)
    assert result.returncode == 2
    assert json.loads(result.stderr)["status"] == "BLOCKED"


class _FailSecondVad(_Vad):
    def __init__(self): self.calls = 0
    def predict(self, request, model, *, protocol):
        self.calls += 1
        if self.calls == 2: raise RuntimeError("decoder failure")
        return super().predict(request, model, protocol=protocol)


def test_multirow_failure_is_persisted_and_cannot_be_complete(tmp_path):
    manifest, digest, model_manifests = _fixture(tmp_path)
    plan = load_prediction_plan(manifest, expected_sha256=digest)
    artifact_path, artifact_hash = model_manifests[("A", 17)]
    selected = load_model_artifact(artifact_path, expected_sha256=artifact_hash, task=plan.selected_model("A", 17))
    store = PredictionStore(plan.output_root / selected.task_id, run_id=plan.run_id, manifest_sha256=plan.manifest_sha256, model_task=selected.task_id, model_binding_sha256=selected.manifest_sha256)
    loader = _Loader()
    outcome = PredictionWorker(plan, store, loader=loader, vad=_FailSecondVad(), vau=_Vau(), model=selected).run()
    assert len(loader.models) == 1
    assert outcome["succeeded"] == 2 and outcome["technical_failures"] == 1 and outcome["succeeded"] != outcome["expected"]
    assert store.get(identity="vad:xd:x1")["status"] == "failure"


class _InvalidVad(_Vad):
    def predict(self, request, model, *, protocol):
        result = super().predict(request, model, protocol=protocol)
        if request.dataset == "ucf":
            result["causal_smoothed_scores"] = [float("nan")]
        return result


def test_vad_numerical_failure_is_persisted_without_automatic_retry(tmp_path):
    manifest, digest, model_manifests = _fixture(tmp_path)
    plan = load_prediction_plan(manifest, expected_sha256=digest)
    artifact_path, artifact_hash = model_manifests[("A", 17)]
    selected = load_model_artifact(artifact_path, expected_sha256=artifact_hash, task=plan.selected_model("A", 17))
    store = PredictionStore(plan.output_root / selected.task_id, run_id=plan.run_id, manifest_sha256=plan.manifest_sha256,
                            model_task=selected.task_id, model_binding_sha256=selected.manifest_sha256)
    outcome = PredictionWorker(plan, store, loader=_Loader(), vad=_InvalidVad(), vau=_Vau(), model=selected).run()
    record = store.get(identity="vad:ucf:u1")
    assert outcome["technical_failures"] == 1
    assert record["retryable"] is False
    assert not store.should_run(identity="vad:ucf:u1", max_retries=3)


def test_blind_detection_runner_matches_released_query_then_frame_smoothing(monkeypatch):
    class Replay:
        instances = []
        def __init__(self, *args, **kwargs): self.instances.append(self)
        def step(self, query, *, capture, image_height, image_width): return None
    monkeypatch.setattr("nc_rted.prediction_adapters.DetectionMemoryReplay", Replay)
    class Smoother:
        def __init__(self): self.calls = []; self.previous = 0.0
        def reset(self): self.calls.clear(); self.previous = 0.0
        def step(self, score):
            self.calls.append(score)
            self.previous = .5 * self.previous + .5 * score
            return self.previous
    query = lambda index, score: SimpleNamespace(index=index, frame_indices=(index * 12,), fast_score=score,
                                                  dense_patches=None, end_seconds=float(index + 1))
    prefix = SimpleNamespace(queries=[query(0, .1), query(1, .4)], image_height=2, image_width=3,
                             frame_count=19, sample_interval=3)
    smoother = Smoother()
    runner = BlindDetectionRunner(bridge=SimpleNamespace(slow=object()), reader=lambda *args, **kwargs: prefix,
                                  protocol=DetectionProtocol("Q", "default", "none", False, True, .5, .6),
                                  observation_reader=None, prompt_tokenizer=None, yes_token_ids=(1,), no_token_ids=(2,),
                                  fusion="replace", fusion_alpha=.5, smoother=smoother)
    result = runner.detect(SimpleNamespace(dataset="ucf", media_id="m", media_path="/bound", media_sha256="a" * 64))
    # Reference evaluator: smooth one query score, expand each to four sampled
    # frames, then repeat each sampled value to original-frame resolution.
    reference = [value for query_score in (.05, .225) for value in [query_score] * 4]
    reference = [value for sampled_score in reference for value in [sampled_score] * 3][:19]
    assert result["causal_smoothed_scores"] == reference
    assert result["total_frames"] == 19
    assert smoother.calls == [.1, .4]
    runner.detect(SimpleNamespace(dataset="ucf", media_id="next", media_path="/bound-next", media_sha256="b" * 64))
    assert len(Replay.instances) == 2 and Replay.instances[0] is not Replay.instances[1]


def test_vad_slow_forward_runs_with_gradients_disabled():
    class Bridge:
        prediction_evidence_enabled = False
        def prepare(self, inputs, observations, *, enabled):
            assert not torch.is_grad_enabled() and observations is None and enabled is False
            return SimpleNamespace(arguments={})
        class slow:
            @staticmethod
            def __call__(**kwargs):
                raise AssertionError("unreachable")
    bridge = Bridge()
    bridge.slow = lambda **kwargs: SimpleNamespace(logits=torch.tensor([[[0.0, 1.0, 2.0]]], requires_grad=True))
    runner = BlindDetectionRunner(bridge=bridge, reader=None, protocol=None, observation_reader=None, prompt_tokenizer=None,
                                  yes_token_ids=(2,), no_token_ids=(1,), fusion="replace", fusion_alpha=.5,
                                  smoother=None)
    assert 0 < runner._slow_probability(object(), None, enabled=False) < 1


def test_incremental_checkpoint_restores_only_selected_trainable_mapping(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint"; checkpoint.mkdir()
    manifest = checkpoint / "manifest.json"; manifest.write_text(json.dumps({"identity": {"group": "A", "seed": "17"}}))
    state = checkpoint / "state.pt"; state.write_bytes(b"bound incremental state")
    artifact = ModelArtifact("a" * 64, "A", 17, str(checkpoint), _digest(manifest), _digest(state), "b" * 64, "c" * 64, None)
    lora = torch.nn.Parameter(torch.zeros(2)); evidence = torch.nn.Parameter(torch.zeros(1))
    bridge = SimpleNamespace(named_parameters=lambda: [("lora", lora), ("evidence", evidence)])
    payload = {"trainable": {"lora": torch.tensor([1.0, 2.0]), "evidence": torch.tensor([3.0])}}
    monkeypatch.setattr("nc_rted.recovery.validate_checkpoint_payload", lambda path, document: payload)
    report = _restore_trainable(bridge, artifact)
    assert report["trainable"] == 2
    assert torch.equal(lora, payload["trainable"]["lora"]) and torch.equal(evidence, payload["trainable"]["evidence"])


def test_preflight_runtime_rejects_checkpoint_for_another_stage2_export():
    identity = {"inherited_weights_sha256": "a" * 64}
    artifact = SimpleNamespace(training_identity=identity)
    runtime = SimpleNamespace(inherited={"stage2_export_sha256": "b" * 64})
    with pytest.raises(PredictionInputError, match="inherited-weight provenance"):
        validate_artifact_runtime(artifact, runtime)


def test_inherited_module_audit_rejects_foreign_cached_dependency(monkeypatch, tmp_path):
    foreign = tmp_path / "foreign.py"; foreign.write_text("# foreign\n")
    module = types.ModuleType("llava.mm_utils"); module.__file__ = str(foreign)
    monkeypatch.setitem(sys.modules, "llava.mm_utils", module)
    with pytest.raises(PredictionInputError, match="escapes bound|differs from registered import name"):
        _audit_bound_inherited_modules(str(Path(__file__).resolve().parents[1]), {})


def test_full_blind_reader_only_builds_rt_for_trigger_and_pads_eos(tmp_path):
    media = tmp_path / "video.mp4"; media.write_bytes(b"bound")
    protocol = DetectionProtocol("Q {score_pct}", "default", "none", False, True, .5, .6)
    class Reader: rows = {("ucf", "m"): {"media_path": str(media), "media_sha256": _digest(media), "fps": 10., "frame_count": 13,
                                              "target_fps": 4, "query_interval": 4, "height": 2, "width": 3,
                                              "queries": [{"index": 0, "frame_indices": [0, 2, 4, 6], "fast_score": .1},
                                                          {"index": 1, "frame_indices": [8, 10, 12], "fast_score": .9}]}}
    Reader.protocols = {"ucf": protocol}
    calls = []
    def encode(frames): calls.append(len(frames)); return torch.zeros((len(frames), 729, 1152))
    Reader.encode = staticmethod(encode)
    class Decoder:
        fps, frame_count, height, width = 10., 13, 2, 3
        def __init__(self, path): pass
        def read(self, index): return index
        def close(self): pass
    prefix = FullBlindDetectionReader(Reader, decoder_factory=Decoder)("ucf", "m")
    queries = list(prefix.queries)
    assert [query.dense_patches is not None for query in queries] == [False, True]
    assert queries[1].dense_patches.shape[0] == 4
    assert calls == [1, 1, 4]


def test_hivau_reader_preserves_full_timeline_tail_and_blocks(tmp_path, monkeypatch):
    media = tmp_path / "clip.mp4"; media.write_bytes(b"clip")
    class Decoder:
        fps, frame_count, height, width = 7., 61, 2, 3
        def __init__(self, path): pass
        def read(self, index): return index
        def close(self): pass
    class Fast:
        image_size = 384
        def batch_score_grids(self, grids, prompt): return [.2] * len(grids)
    class Tower: config = type("C", (), {"hidden_size": 1, "num_attention_heads": 1})()
    parameter = torch.nn.Parameter(torch.zeros(1))
    class Slow:
        def get_vision_tower(self): return Tower()
        def get_model(self): return type("M", (), {"mm_projector": type("P", (), {"mlp": torch.nn.Linear(1, 1)})()})()
    class Memory:
        created = 0
        def __init__(self, *args, **kwargs):
            type(self).created += 1
            self.items=[]
        def update_with_anomaly_score(self, feature, anomaly_score): self.items.append((feature, anomaly_score))
    class Observed: features = torch.zeros((1, 2, 1, 4, 384))
    seen = {}
    class Observer:
        def observe_full_media(self, **kwargs): seen.update(kwargs); return Observed()
    monkeypatch.setattr("nc_rted.prediction_media.detection_memory_from_stream", lambda slow, memory: torch.zeros((1, 2, 1)))
    reader = FullBlindHivauReader(slow=Slow(), fast_detector=Fast(), fast_prompt="pg", vision_encoder=lambda frames: torch.zeros((len(frames),729,1152)), observer=Observer(), target_fps=4, decoder_factory=Decoder, memory_factory=Memory, grid_builder=lambda frames, image_size: frames)
    material = reader.read(media_path=str(media), media_sha256=_digest(media))
    reader.read(media_path=str(media), media_sha256=_digest(media))
    assert material.sampled_frame_times[-1] == 60 / 7 and len(material.fast_scores) == 16
    assert seen["observed_seconds"] == 61 / 7 and "16 frames" in material.time_message
    assert Memory.created == 2


def test_hivau_r0_bypasses_evidence_observer(tmp_path, monkeypatch):
    media = tmp_path / "clip.mp4"; media.write_bytes(b"clip")
    class Decoder:
        fps, frame_count, height, width = 4., 4, 2, 3
        def __init__(self, path): pass
        def read(self, index): return index
        def close(self): pass
    class Fast:
        image_size = 384
        def batch_score_grids(self, grids, prompt): return [.2] * len(grids)
    class Tower: config = type("C", (), {"hidden_size": 1, "num_attention_heads": 1})()
    class Slow:
        def get_vision_tower(self): return Tower()
        def get_model(self): return type("M", (), {"mm_projector": type("P", (), {"mlp": torch.nn.Linear(1, 1)})()})()
    class Memory:
        def __init__(self, *args, **kwargs): pass
        def update_with_anomaly_score(self, *args, **kwargs): pass
    class Observer:
        def observe_full_media(self, **kwargs): raise AssertionError("R0 must not build evidence observations")
    monkeypatch.setattr("nc_rted.prediction_media.detection_memory_from_stream", lambda slow, memory: torch.zeros((1, 2, 1)))
    reader = FullBlindHivauReader(slow=Slow(), fast_detector=Fast(), fast_prompt="", vision_encoder=lambda frames: torch.zeros((len(frames), 729, 1152)),
                                  observer=Observer(), evidence_enabled=False, decoder_factory=Decoder, memory_factory=Memory,
                                  grid_builder=lambda frames, image_size: frames)
    material = reader.read(media_path=str(media), media_sha256=_digest(media))
    assert material.observations is None


def test_blind_runtime_rejects_supervision_before_loading_assets(tmp_path):
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps({"schema": "nc_rted_prediction_runtime/v1", "label": "forbidden"}))
    with pytest.raises(PredictionInputError, match="forbidden supervision"):
        load_prediction_runtime(path, expected_sha256=_digest(path))


@pytest.mark.parametrize("interrupted", (False, True))
def test_watchdog_cleans_child_before_setsid(monkeypatch, tmp_path, interrupted):
    entrypoint = _prediction_entrypoint()
    child_pid = tmp_path / "child.pid"
    def delayed_setsid():
        child_pid.write_text(str(os.getpid()))
        time.sleep(10)
    monkeypatch.setattr(entrypoint.os, "setsid", delayed_setsid)
    if interrupted:
        def interrupt(*args):
            raise KeyboardInterrupt()
        monkeypatch.setattr(entrypoint.select, "select", interrupt)
    started = time.monotonic()
    with pytest.raises(KeyboardInterrupt if interrupted else ValueError):
        entrypoint._watchdog(time.time() + .1, lambda: {"unexpected": True})
    assert time.monotonic() - started < 2
    if child_pid.exists():
        with pytest.raises(ProcessLookupError):
            os.kill(int(child_pid.read_text()), 0)


@pytest.mark.parametrize("kind", ("fifo", "symlink", "large"))
def test_budget_ledger_rejects_invalid_inode_before_read(tmp_path, kind):
    entrypoint = _prediction_entrypoint()
    root = tmp_path / "output"; root.mkdir()
    ledger = root / ".resource-budget.json"
    outside = tmp_path / "outside"; outside.write_text("preserve")
    if kind == "fifo": os.mkfifo(ledger)
    elif kind == "symlink": ledger.symlink_to(outside)
    else: ledger.write_bytes(b" " * 65537)
    plan = SimpleNamespace(resource_scope={"data_volume": str(tmp_path), "min_free_bytes": 1, "run_budget_seconds": 2},
                           output_root=root, execution_scope_sha256="a" * 64, run_id="run")
    with pytest.raises(ValueError, match="control path|too large"):
        with entrypoint._cumulative_budget(plan, time.time() + 2):
            pytest.fail("invalid ledger admitted")
    assert outside.read_text() == "preserve"


@pytest.mark.parametrize("kind", ("fifo", "symlink", "dangling"))
def test_worker_lock_rejects_redirection_before_creation(tmp_path, kind):
    root = tmp_path / "output"; root.mkdir()
    outside = tmp_path / "outside"
    lock = root / ".worker.lock"
    if kind == "fifo": os.mkfifo(lock)
    else:
        if kind == "symlink": outside.write_text("preserve")
        lock.symlink_to(outside)
    with pytest.raises(PredictionStoreError, match="worker lock"):
        PredictionStore(root, run_id="run", manifest_sha256="a" * 64, model_task="R0", model_binding_sha256="b" * 64,
                        storage_volume=tmp_path, min_free_bytes=1)
    assert outside.read_text() == "preserve" if kind == "symlink" else not outside.exists()


def test_budget_keeps_full_charge_when_child_reap_is_unconfirmed(tmp_path):
    entrypoint = _prediction_entrypoint()
    plan = SimpleNamespace(resource_scope={"data_volume": str(tmp_path), "min_free_bytes": 1, "run_budget_seconds": 10},
                           output_root=tmp_path / "out", execution_scope_sha256="a" * 64, run_id="run")
    with pytest.raises(entrypoint.WorkerCleanupIncomplete):
        with entrypoint._cumulative_budget(plan, time.time() + 5):
            raise entrypoint.WorkerCleanupIncomplete("test")
    assert json.loads((plan.output_root / ".resource-budget.json").read_text())["consumed_seconds"] > 4


@pytest.mark.parametrize("failure", (KeyboardInterrupt, OSError))
def test_watchdog_setup_failure_cleans_direct_child(monkeypatch, failure):
    entrypoint = _prediction_entrypoint()
    fork = entrypoint.os.fork
    created = []
    def recording_fork():
        pid = fork()
        if pid: created.append(pid)
        return pid
    def fail_setup(*args):
        raise failure("injected setup interruption")
    monkeypatch.setattr(entrypoint.os, "fork", recording_fork)
    monkeypatch.setattr(entrypoint.os, "set_blocking", fail_setup)
    with pytest.raises(failure):
        entrypoint._watchdog(time.time() + 2, lambda: time.sleep(10))
    assert len(created) == 1
    with pytest.raises(ProcessLookupError):
        os.kill(created[0], 0)
