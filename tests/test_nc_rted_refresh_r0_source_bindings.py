import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "nc_rted_refresh_r0_source_bindings.py"
SPEC = importlib.util.spec_from_file_location("r0_source_refresh", SCRIPT)
refresh = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(refresh)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: dict) -> str:
    path.write_text(json.dumps(value))
    return digest(path)


def bound(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": digest(path)}


def old_preflight(tmp_path: Path, *, runtime_hash: str | None = None) -> tuple[Path, str]:
    item = tmp_path / "item.json"; item.write_text("{}")
    reference = bound(item)
    if runtime_hash is not None:
        reference = {**reference, "sha256": runtime_hash}
    formal = tmp_path / "formal.json"
    formal_hash = write(formal, {"status": "BLOCKED", "formal_execution_allowed": False,
                                 "requested_output_root": str(tmp_path / "predictions"),
                                 "reason": "formal admission has not accepted this exact runtime, identity, and binding set"})
    report = {"schema": refresh.PREFLIGHT_SCHEMA, "status": "PREFLIGHT_PASS_FORMAL_ADMISSION_REQUIRED", "gpu_launched": False,
              "runtime": reference, "identity_manifest": bound(item), "r0_model_manifest": bound(item),
              "bindings": {"prediction_source": bound(item), "implementation": bound(item), "decoder": bound(item), "embedded_vision": bound(item)},
              "formal_admission": {"path": str(formal), "sha256": formal_hash}, "denominators": {"ucf": 251, "xd": 800, "vau": 3339},
              "missing_dependencies": ["formal admission for this exact runtime/identity/binding set"]}
    path = tmp_path / "preflight.json"
    return path, write(path, report)


def test_old_preflight_rejects_nonblocked_admission(tmp_path):
    item = tmp_path / "item.json"; item.write_text("{}")
    reference = {"path": str(item), "sha256": digest(item)}
    formal = tmp_path / "formal.json"; formal_hash = write(formal, {"status": "PASS", "formal_execution_allowed": True})
    report = {"schema": refresh.PREFLIGHT_SCHEMA, "status": "PREFLIGHT_PASS_FORMAL_ADMISSION_REQUIRED", "gpu_launched": False,
              "runtime": reference, "identity_manifest": reference, "r0_model_manifest": reference,
              "bindings": {"prediction_source": reference, "implementation": reference, "decoder": reference, "embedded_vision": reference},
              "formal_admission": {"path": str(formal), "sha256": formal_hash}, "denominators": {"ucf": 251, "xd": 800, "vau": 3339},
              "missing_dependencies": ["formal admission for this exact runtime/identity/binding set"]}
    path = tmp_path / "preflight.json"; report_hash = write(path, report)
    with pytest.raises(refresh.RefreshError, match="blocked placeholder"):
        refresh._old_preflight(str(path), report_hash)


def test_old_preflight_rejects_changed_reused_evidence(tmp_path):
    path, report_hash = old_preflight(tmp_path, runtime_hash="0" * 64)
    with pytest.raises(refresh.RefreshError, match="runtime differs"):
        refresh._old_preflight(str(path), report_hash)


def test_release_acceptance_requires_both_code01_components(tmp_path, monkeypatch):
    release = tmp_path / "release"; release.mkdir()
    commit = "a" * 40
    acceptance = tmp_path / "acceptance.json"
    acceptance_hash = write(acceptance, {"schema": refresh.ACCEPTANCE_SCHEMA, "status": "PASS", "release_root": str(release), "release_commit": commit,
                                          "accepted_components": {"code01a": "b" * 64}})
    monkeypatch.setattr(refresh, "ROOT", release)
    monkeypatch.setattr(refresh.subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, stdout=commit + "\n"))
    with pytest.raises(refresh.RefreshError, match="CODE-01A and CODE-01B"):
        refresh._verify_release(release, commit, str(acceptance), acceptance_hash)


def test_publish_never_overwrites_existing_output(tmp_path):
    path = tmp_path / "already.json"
    path.write_text('{"historical":true}\n')
    with pytest.raises(refresh.RefreshError, match="new absolute path"):
        refresh._publish(path, {"replacement": True})
    assert path.read_text() == '{"historical":true}\n'


def test_run_refreshes_only_source_binding_and_small_json(tmp_path):
    commit = subprocess.run(["git", "-C", str(refresh.ROOT), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    acceptance = tmp_path / "acceptance.json"
    acceptance_hash = write(acceptance, {"schema": refresh.ACCEPTANCE_SCHEMA, "status": "PASS", "release_root": str(refresh.ROOT.resolve()),
                                         "release_commit": commit, "accepted_components": {"code01a": "a" * 64, "code01b": "b" * 64}})
    prior, prior_hash = old_preflight(tmp_path)
    output = tmp_path / "output" / "preflight.json"
    result = refresh.run(type("Args", (), {"accepted_release_root": str(refresh.ROOT), "accepted_release_commit": commit,
                                             "source_acceptance": str(acceptance), "source_acceptance_sha256": acceptance_hash,
                                             "old_preflight": str(prior), "old_preflight_sha256": prior_hash,
                                             "output": str(output), "output_root": str(tmp_path / "prediction-output")})())
    report = json.loads(output.read_text())
    implementation = output.parent / "implementation_manifest.json"
    source_binding = output.parent / "source_refresh_binding.json"
    admission = output.parent / "formal_admission_required.json"
    assert result["status"] == "PREFLIGHT_PASS_FORMAL_ADMISSION_REQUIRED"
    assert set(report) == {"schema", "status", "gpu_launched", "runtime", "identity_manifest", "r0_model_manifest", "bindings", "formal_admission", "denominators", "missing_dependencies"}
    assert report["bindings"]["implementation"] == bound(implementation)
    assert report["formal_admission"] == bound(admission)
    assert json.loads(implementation.read_text())["files"].keys() == refresh._IMPLEMENTATION_FILES
    assert json.loads(source_binding.read_text())["reused"]["runtime"][0] == str(tmp_path / "item.json")
    assert json.loads(admission.read_text())["formal_execution_allowed"] is False


def test_release_rejects_staged_source_drift(tmp_path, monkeypatch):
    release = tmp_path / "release"; release.mkdir()
    def git(*args):
        return subprocess.run(["git", "-C", str(release), *args], check=True, capture_output=True, text=True).stdout.strip()
    git("init", "-q")
    source = release / "source.py"; source.write_text("accepted = True\n")
    git("add", "source.py")
    git("-c", "user.name=NC RTED Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "fixture")
    commit = git("rev-parse", "HEAD")
    source.write_text("accepted = False\n"); git("add", "source.py")
    monkeypatch.setattr(refresh, "ROOT", release)
    with pytest.raises(refresh.RefreshError, match="dirty"):
        refresh._verify_release(release, commit, str(tmp_path / "not-read.json"), "0" * 64)


def test_publication_existing_directory_reserves_payload_and_metadata(tmp_path, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(refresh.shutil, "disk_usage", lambda path: SimpleNamespace(free=refresh.RESERVE + 4096))
    target = tmp_path / "report.json"
    with pytest.raises(refresh.RefreshError, match="storage reserve"):
        refresh._publish(target, {"status": "BLOCKED"})
    assert not target.exists()
    assert not list(tmp_path.glob("*.pending"))
