import time
import json
import csv
import io
import hashlib
import base64
import venv
from pathlib import Path

import pytest

from nc_rted import resource_attestation as contract


def environment(root):
    (root / "tmpdir").mkdir(parents=True, exist_ok=True)
    return {**{key: str(root / key.lower()) for key in contract.CACHE_KEYS},
            "PYTHONPATH": str(Path(contract.__file__).resolve().parents[1]),
            "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1"}


def qualification():
    now = time.time()
    runtime = {"executable": "/qualified/python", "packages": {"torch": "qualified"}}
    interpreter = {"path": "/qualified/python"}
    env = {"qualified": "environment"}
    identity = {"group": "F", "seed": 17}
    document = {
        "schema": "nc_rted_runtime_qualification/v2",
        "status": "PASS_GPU_KERNEL_AND_INHERITED_RUNTIME_IMPORTS",
        "host": "qualified-host", "gpu_uuid": "qualified-gpu", "source_sha256": "s" * 64,
        "runtime_identity": identity,
        "runtime_environment": {"interpreter": interpreter, "environment": env},
        "workload": {"run_identity": identity, "updates": 1000, "kind": "formal_train"},
        "valid_from_utc_epoch": now - 1, "valid_until_utc_epoch": now + 100,
        "resource_envelope": {"device_memory_bytes": 32, "required_memory_bytes": 24},
        "measurements": {"measured_at_utc_epoch": now, "peak_cuda_allocated_bytes": 16,
                         "peak_cuda_reserved_bytes": 20, "seconds_per_update_upper_bound": .02,
                         "setup_checkpoint_seconds_upper_bound": 1,
                         "forward_backward_completed": True, "complete_long_input": True,
                         "optimizer_updates": 1,
                         "local_import_probe": {"status": "PASS", "interpreter": interpreter,
                                                "environment": env, "duration_seconds": .1,
                                                "runtime_identity": runtime}},
    }
    arguments = dict(identity=identity, source_sha256="s" * 64, host="qualified-host",
                     device_uuid="qualified-gpu", interpreter=interpreter, environment=env,
                     budget=60, now=now, actual_runtime=runtime)
    return document, arguments


@pytest.mark.parametrize("mutation", [
    lambda d: d["measurements"].update(measured_at_utc_epoch=time.time() + 30),
    lambda d: d["measurements"].update(measured_at_utc_epoch=1),
    lambda d: d["measurements"].update(peak_cuda_reserved_bytes=8),
    lambda d: d["resource_envelope"].update(device_memory_bytes=10),
    lambda d: d["measurements"].update(seconds_per_update_upper_bound=1),
    lambda d: d["measurements"].update(forward_backward_completed=False),
    lambda d: d["measurements"].update(complete_long_input=False),
    lambda d: d["measurements"].update(optimizer_updates=True),
    lambda d: d["measurements"]["local_import_probe"].update(duration_seconds=float("nan")),
    lambda d: d["measurements"]["local_import_probe"].update(runtime_identity={"different": "packages"}),
    lambda d: d["workload"].update(updates=1),
])
def test_qualification_rejects_unmeasured_or_inapplicable_capacity(mutation):
    document, arguments = qualification()
    contract.validate_qualification(document, **arguments)
    mutation(document)
    with pytest.raises(contract.ResourceAttestationError):
        contract.validate_qualification(document, **arguments)


def test_environment_requires_explicit_caches_without_path_aliases(tmp_path):
    env = environment(tmp_path)
    contract.validate_environment(env, contract.PROJECT_VOLUME)
    for key in contract.CACHE_KEYS:
        missing = dict(env)
        missing.pop(key)
        with pytest.raises(contract.ResourceAttestationError):
            contract.validate_environment(missing, contract.PROJECT_VOLUME)
    alias = tmp_path / "alias"
    alias.symlink_to(tmp_path, target_is_directory=True)
    env["TMPDIR"] = str(alias / "temporary")
    with pytest.raises(contract.ResourceAttestationError, match="canonical"):
        contract.validate_environment(env, contract.PROJECT_VOLUME)
    env["TMPDIR"] = str(tmp_path / "absent")
    with pytest.raises(contract.ResourceAttestationError, match="not usable"):
        contract.validate_environment(env, contract.PROJECT_VOLUME)


def test_local_probe_rejects_executable_wrappers_before_start(tmp_path):
    wrapper = tmp_path / "python-wrapper"
    wrapper.write_text("#!/bin/sh\nexit 0\n")
    wrapper.chmod(0o700)
    with pytest.raises(contract.ResourceAttestationError, match="wrapper"):
        contract.python_runtime_probe({"path": str(wrapper)}, {})


def test_attestation_recovery_syncs_existing_final_and_preserves_conflicting_temp(tmp_path, monkeypatch):
    monkeypatch.setattr(contract, "MIN_FREE_BYTES", 0)
    target = tmp_path / "ancestry" / "nested" / "resource.json"
    document = {"schema": "test", "value": 1}
    contract.publish_attestation(document, target)
    identical = target.parent / "resource.json.interrupted.tmp"
    identical.write_bytes(target.read_bytes())
    conflicting = target.parent / "resource.json.conflicting.tmp"
    conflicting.write_bytes(b"other preserved evidence")
    synced = []
    original = contract.os.fsync
    monkeypatch.setattr(contract.os, "fsync", lambda fd: (synced.append(contract.os.fstat(fd).st_ino), original(fd))[-1])
    contract.publish_attestation(document, target)
    assert not identical.exists()
    assert conflicting.read_bytes() == b"other preserved evidence"
    assert target.stat().st_ino in synced
    assert target.parent.stat().st_ino in synced
    assert tmp_path.stat().st_ino in synced
    assert contract.PROJECT_VOLUME.stat().st_ino in synced


def test_contract_stat_eio_is_normalized_as_protective_failure(tmp_path, monkeypatch):
    path = tmp_path / "contract.json"
    original = Path.is_file
    def failing(candidate):
        if candidate == path:
            raise OSError(5, "injected storage failure")
        return original(candidate)
    monkeypatch.setattr(Path, "is_file", failing)
    with pytest.raises(contract.ResourceAttestationError):
        contract.resource_lease_expiry({"resource_attestation": str(path), "resource_attestation_sha256": "0" * 64})


def test_runtime_fingerprint_detects_dependency_and_startup_bytes(tmp_path):
    # Disposable pure-Python packages exercise real RECORD-content hashing
    # without modifying the actual environment or initializing a GPU.
    root = tmp_path / "packages"
    root.mkdir()
    for name in ("torch", "transformers", "peft", "accelerate", "safetensors", "tokenizers", "numpy"):
        module = root / (name + ".py")
        module.write_text("value = 1\n")
        metadata = root / (name + "-1.0.dist-info")
        metadata.mkdir()
        (metadata / "METADATA").write_text("Name: " + name + "\nVersion: 1.0\n")
        row = io.StringIO()
        csv.writer(row).writerow([module.name, "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(module.read_bytes()).digest()).rstrip(b"=").decode(), module.stat().st_size])
        (metadata / "RECORD").write_text(row.getvalue())
    project = root / "nc_rted"
    project.mkdir()
    (project / "__init__.py").touch()
    (project / "production_runtime.py").touch()
    virtual = tmp_path / "venv"
    venv.EnvBuilder(with_pip=False).create(virtual)
    interpreter = {"path": str(virtual / "bin" / "python")}
    env = environment(tmp_path)
    env["PYTHONPATH"] = str(root)
    first = contract.python_runtime_probe(interpreter, env)
    (root / "torch.py").write_text("value = 2\n")
    second = contract.python_runtime_probe(interpreter, env)
    assert first["packages"]["torch"]["metadata"] == second["packages"]["torch"]["metadata"]
    assert first["packages"]["torch"]["installed_files_sha256"] != second["packages"]["torch"]["installed_files_sha256"]
    config = virtual / "pyvenv.cfg"
    config.write_text(config.read_text() + "# changed qualified startup\n")
    third = contract.python_runtime_probe(interpreter, env)
    assert second["startup"] != third["startup"]


def test_real_controlled_environment_probe_preserves_venv_and_imports(tmp_path):
    interpreter = Path("/root/autodl-tmp/lookaway-wm/.venv-reactvau/bin/python")
    env = environment(tmp_path)
    for key in contract.CACHE_KEYS:
        Path(env[key]).mkdir(exist_ok=True)
    measured = contract.python_runtime_probe({"path": str(interpreter)}, env)
    assert measured["executable"] == str(interpreter)
    assert measured["prefix"] == str(interpreter.parent.parent)
    assert measured["prefix"] != measured["base_prefix"]
    assert measured["project_module"] == str(Path(contract.__file__).with_name("production_runtime.py"))
    assert set(measured["packages"]) == {"torch", "transformers", "peft", "accelerate", "safetensors", "tokenizers", "numpy"}
    assert all(value["metadata"]["METADATA"] for value in measured["packages"].values())
