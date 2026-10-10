import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "nc_rted_produce_source34_formal_bundle_inputs.py"
SPEC = importlib.util.spec_from_file_location("source34_formal_bundle_input_producer", SCRIPT)
producer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = producer
SPEC.loader.exec_module(producer)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nc_rted import resource_attestation as attestation
from test_nc_rted_source34_seed_applicability import digest, source34_bundle, write


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_source34_scope_requires_applicability_seed_and_all_gate_evidence(tmp_path):
    applicability, qualification = tmp_path / "applicability.json", tmp_path / "qualification.json"
    applicability_sha, qualification_sha = write_json(applicability, {}), write_json(qualification, {})
    gates = {}
    for number in range(1, 11):
        path = tmp_path / f"gate-{number}.json"
        gates[str(number)] = {"path": str(path), "sha256": write_json(path, {"gate": number})}
    interpreter, environment = {"path": "/bin/python"}, {"PYTHONPATH": "/source34/src"}
    scope = {"schema": "nc_rted_source34_target_resource_scope/v1", "status": "AUTHORIZED", "authorization_id": "scope",
             "authorized_by": "owner", "applicability": {"path": str(applicability), "sha256": applicability_sha},
             "qualification": {"path": str(qualification), "sha256": qualification_sha}, "target_seed": 42, "host": "host",
             "physical_gpu": 0, "gpu_uuid": "gpu", "project_volume": str(producer.PROJECT),
             "runtime_environment": {"interpreter": interpreter, "environment": environment}, "lease_id": "lease",
             "lease_expires_utc_epoch": producer.RENTAL_CUTOFF, "max_budget_seconds": 60, "min_free_bytes": 30,
             "deadline_utc_epoch": producer.FORMAL_DEADLINE, "engineering_gate_evidence": gates,
             "engineering_checks": {str(number): "PASS" for number in range(1, 11)}}
    arguments = dict(applicability_path=applicability, applicability_sha=applicability_sha,
                     qualification={"path": str(qualification), "sha256": qualification_sha}, target_seed=42,
                     environment=environment, interpreter=interpreter, host="host", gpu_uuid="gpu", physical_gpu=0,
                     budget=60, reserve=30, now=1)
    assert producer.require_scope(scope, **arguments) == gates
    broken = dict(scope); broken["target_seed"] = 17
    with pytest.raises(producer.ProducerError, match="does not bind"):
        producer.require_scope(broken, **arguments)
    broken = dict(scope); broken["engineering_gate_evidence"] = dict(gates); broken["engineering_gate_evidence"].pop("10")
    with pytest.raises(producer.ProducerError, match="ten accepted"):
        producer.require_scope(broken, **arguments)


def test_materializer_runs_real_source34_consumers_and_disposable_queue(tmp_path, monkeypatch):
    payload, _evidence, applicability_path, attestation_path = source34_bundle(tmp_path, monkeypatch)
    applicability = json.loads(applicability_path.read_text())
    qualification = applicability["measured_qualification"]
    gates = {}
    for number in range(1, 11):
        path = tmp_path / f"gate-{number}.json"
        gates[str(number)] = {"path": str(path), "sha256": write(path, {"gate": number})}
    scope = tmp_path / "scope.json"
    environment = payload["execution_environment"]
    scope_document = {"schema": "nc_rted_source34_target_resource_scope/v1", "status": "AUTHORIZED",
                      "authorization_id": "source34-test", "authorized_by": "test", "applicability": {"path": str(applicability_path), "sha256": digest(applicability_path)},
                      "qualification": qualification, "target_seed": 42, "host": __import__("socket").gethostname(),
                      "physical_gpu": 0, "gpu_uuid": "GPU-valid", "project_volume": str(producer.PROJECT),
                      "runtime_environment": {"interpreter": payload["interpreter"], "environment": environment}, "lease_id": "lease",
                      "lease_expires_utc_epoch": attestation.RENTAL_CUTOFF, "max_budget_seconds": 60,
                      "min_free_bytes": 20 * 1024 ** 3, "deadline_utc_epoch": attestation.FORMAL_DEADLINE,
                      "engineering_gate_evidence": gates, "engineering_checks": {str(number): "PASS" for number in range(1, 11)}}
    scope_sha = write(scope, scope_document)
    root = producer.PROJECT / ".cache" / "nc_rted_queue_tests" / tmp_path.name / "materialized-source34"
    run_dir = producer.PROJECT / ".cache" / "nc_rted_queue_tests" / tmp_path.name / "materialized-source34-runs"
    common = producer.PROJECT / ".cache" / "nc_rted_queue_tests" / tmp_path.name / "materialized-source34-common"
    queue_db = producer.PROJECT / ".cache" / "nc_rted_queue_tests" / tmp_path.name / "materialized-source34.sqlite3"
    for path in (root, run_dir, common, queue_db):
        if path.is_dir(): __import__("shutil").rmtree(path)
        else: path.unlink(missing_ok=True)
    captured = {}
    original_preflight = producer.preflight_queue
    def inspect_preflight(*args, **kwargs):
        captured["payload"] = kwargs["payload"]
        captured["evidence"] = kwargs["evidence"]
        return original_preflight(*args, **kwargs)
    monkeypatch.setattr(producer, "preflight_queue", inspect_preflight)
    repo = Path(__file__).resolve().parents[1]
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--runtime-source-root", str(repo), "--applicability", str(applicability_path),
                                        "--applicability-sha256", digest(applicability_path), "--authorization-scope", str(scope),
                                        "--authorization-scope-sha256", scope_sha, "--interpreter", payload["interpreter"]["path"],
                                        "--physical-gpu", "0", "--run-budget-seconds", "60", "--queue-db", str(queue_db),
                                        "--queue-run-dir", str(run_dir), "--bundle-checkpoint-root", str(common), "--output-root", str(root),
                                        "--job-key", "source34-test", "--materialize"])
    monkeypatch.chdir(repo)
    producer.main()
    assert (root / "bundle.json").is_file() and (root / "bundle-resource-attestation.json").is_file()
    assert captured["payload"]["member_identities"]["A"]["seed"] == "42"
    producer.import_source34(repo)[0].verify_bundle_attestation(captured["payload"], "source34-test", captured["evidence"])
    __import__("shutil").rmtree(root.parent)
