import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "nc_rted_produce_source33_formal_bundle_inputs.py"
SPEC = importlib.util.spec_from_file_location("source33_formal_bundle_input_producer", SCRIPT)
producer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = producer
SPEC.loader.exec_module(producer)


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scope_for(tmp_path):
    qualification = tmp_path / "qualification.json"
    profile = tmp_path / "profile.json"
    qualification_sha = write_json(qualification, {"kind": "qualification"})
    profile_sha = write_json(profile, {"kind": "profile"})
    gates = {}
    for number in range(1, 11):
        path = tmp_path / f"gate-{number}.json"
        gates[str(number)] = {"path": str(path), "sha256": write_json(path, {"gate": number})}
    scope = {
        "schema": "nc_rted_source33_formal_resource_scope/v1", "status": "AUTHORIZED",
        "authorization_id": "authorization", "authorized_by": "owner",
        "qualification": {"path": str(qualification), "sha256": qualification_sha},
        "profile": {"path": str(profile), "sha256": profile_sha},
        "host": "host", "physical_gpu": 0, "gpu_uuid": "gpu", "project_volume": str(producer.PROJECT),
        "runtime_environment": {"interpreter": "python", "environment": "environment"},
        "lease_id": "lease", "lease_expires_utc_epoch": producer.RENTAL_CUTOFF,
        "max_budget_seconds": 60, "min_free_bytes": 30, "deadline_utc_epoch": producer.FORMAL_DEADLINE,
        "engineering_gate_evidence": gates, "engineering_checks": {str(number): "PASS" for number in range(1, 11)},
    }
    return scope, qualification, qualification_sha, profile, profile_sha


def test_scope_rejects_missing_or_mismatched_gate_evidence(tmp_path):
    scope, qualification, qualification_sha, profile, profile_sha = scope_for(tmp_path)
    arguments = dict(qualification_path=qualification, qualification_sha=qualification_sha, profile_path=profile,
                     profile_sha=profile_sha, runtime_environment={"interpreter": "python", "environment": "environment"},
                     physical_gpu=0, host="host", gpu_uuid="gpu", run_budget=60, reserve=30, now=1)
    broken = dict(scope)
    broken["engineering_gate_evidence"] = dict(scope["engineering_gate_evidence"])
    broken["engineering_gate_evidence"].pop("10")
    with pytest.raises(producer.ProducerError, match="ten explicit"):
        producer.require_scope(broken, **arguments)
    broken = dict(scope)
    broken["engineering_gate_evidence"] = dict(scope["engineering_gate_evidence"])
    broken["engineering_gate_evidence"]["1"] = dict(broken["engineering_gate_evidence"]["1"])
    broken["engineering_gate_evidence"]["1"]["sha256"] = "0" * 64
    with pytest.raises(producer.ProducerError, match="engineering gate 1"):
        producer.require_scope(broken, **arguments)
    broken = dict(scope)
    broken["deadline_utc_epoch"] = float("nan")
    with pytest.raises(producer.ProducerError, match="numeric bounds"):
        producer.require_scope(broken, **arguments)
    broken = dict(scope)
    broken["engineering_checks"] = dict(scope["engineering_checks"])
    broken["engineering_checks"]["4"] = "FAIL"
    with pytest.raises(producer.ProducerError, match="accepted engineering"):
        producer.require_scope(broken, **arguments)


def test_profile_rejects_non_seed17_identity(tmp_path, monkeypatch):
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "member.py").write_text("pass\n", encoding="utf-8")
    source = {"schema": "nc_rted_interleaved_source_manifest/v1", "files": {"member.py": producer.digest(source_root / "member.py")}, "code_sha256": "source"}
    identity = {"seed": "42", "code_sha256": "source"}
    identities = {group: identity for group in producer.GROUPS}
    profile = {"schema": "nc_rted_interleaved_formal_qualification_profile/v1", "status": "NON_ADMITTED_FORMAL_PROFILE",
               "members": {group: {"runtime": group, "runtime_sha256": "x"} for group in producer.GROUPS},
               "member_identities": identities, "diagnostic_bundle": "bundle", "diagnostic_bundle_sha256": "b",
               "updates": 1000, "accumulation": 8, "shared_preparation": "frozen_provider_only",
               "common_recovery": True, "complete_long_input_coverage": {}}
    bundle = {"source_manifest": "source", "source_manifest_sha256": "s"}
    qualification = {"workload": {"member_identities": identities, "updates": 1000, "kind": "formal_bundle",
                                  "shared_preparation": "frozen_provider_only", "common_recovery": True}, "source_sha256": "source",
                     "qualification_profile": {"path": str(tmp_path / "profile.json"), "sha256": "p"}}
    monkeypatch.setattr(producer, "bound_json", lambda path, _sha, _name: (Path(path), bundle if path == "bundle" else source))
    monkeypatch.setattr(producer, "runtime_module_identity", lambda _runtime, _manifest: identity)

    class Runtime:
        def load_manifest(self, path, expected_sha256):
            return SimpleNamespace(run={"mode": "formal", "group": path}, document={})

    with pytest.raises(producer.ProducerError, match="seed-17"):
        producer.checked_profile(profile, tmp_path / "profile.json", "p", qualification, Runtime(), source_root)


def test_dry_run_never_creates_output_root(tmp_path, monkeypatch, capsys):
    root = tmp_path / "output"
    qualification = tmp_path / "qualification.json"
    scope = tmp_path / "scope.json"
    identities = {group: {"seed": "17"} for group in producer.GROUPS}
    result = (root, {}, {}, {}, identities, {}, {}, {}, {}, {}, {}, tmp_path / "run", tmp_path / "common", qualification, scope)
    monkeypatch.setattr(producer, "import_source33", lambda _root: object())
    monkeypatch.setattr(producer, "build_documents", lambda _args, _modules: result)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--runtime-source-root", str(tmp_path), "--qualification-profile", "p",
                                        "--qualification-profile-sha256", "p", "--qualification-report", "q",
                                        "--qualification-report-sha256", "q", "--authorization-scope", "s",
                                        "--authorization-scope-sha256", "s", "--interpreter", "python", "--physical-gpu", "0",
                                        "--run-budget-seconds", "1", "--queue-db", str(tmp_path / "queue.sqlite"),
                                        "--queue-run-dir", str(tmp_path / "run"), "--bundle-checkpoint-root", str(tmp_path / "common"),
                                        "--output-root", str(root)])
    producer.main()
    assert not root.exists()
    assert json.loads(capsys.readouterr().out)["status"] == "INPUT_BINDINGS_ONLY_NOT_ADMITTED"


def test_admission_source_keys_preserve_frozen_relative_identity(tmp_path):
    source_root = tmp_path / "source"
    source_file = source_root / "src" / "nc_rted" / "module.py"
    source_file.parent.mkdir(parents=True)
    source_file.write_text("pass\n", encoding="utf-8")
    relative = "src/nc_rted/module.py"
    files = {relative: producer.digest(source_file)}
    qualified = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    absolute = {str(source_file): files[relative]}
    admitted = hashlib.sha256(json.dumps(absolute, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert qualified != admitted
    assert {relative: relative} == {name: name for name in files}


def test_failed_consumer_validation_removes_new_output_tree(tmp_path):
    root = tmp_path / "output"
    with pytest.raises(RuntimeError, match="consumer"):
        producer.materialize_and_validate(root, {Path("document.json"): {"value": 1}}, lambda: (_ for _ in ()).throw(RuntimeError("consumer rejected")))
    assert not root.exists()


def test_rejects_overlapping_publication_and_execution_roots(tmp_path):
    with pytest.raises(producer.ProducerError, match="nonoverlapping"):
        producer.require_disjoint_roots(tmp_path / "inputs", tmp_path / "inputs" / "run")


def test_queue_preflight_rejects_before_real_queue_insertion(tmp_path):
    evidence_path = tmp_path / "evidence.json"
    checksum = write_json(evidence_path, {"evidence": True})
    instances = []

    class Connection:
        def __enter__(self): return self
        def __exit__(self, *_args): return False
        def execute(self, *_args): return self
        def fetchone(self): return object()

    class RejectingQueue:
        def __init__(self, path): self.path = Path(path); instances.append(self)
        def add_evidence(self, *_args, **_kwargs): pass
        def add_job(self, *_args, **_kwargs): pass
        def connect(self): return Connection()
        def _guard_failure(self, _db, _row): return "rejected by source33"

    queue_db = tmp_path / "real.sqlite"
    with pytest.raises(producer.ProducerError, match="rejected by source33"):
        producer.enqueue_after_preflight(RejectingQueue, queue_db=str(queue_db), job_key="job", payload={},
                                         evidence={"evidence": (str(evidence_path), checksum)}, directory=tmp_path)
    assert len(instances) == 1
    assert instances[0].path != queue_db
    assert not queue_db.exists()
