"""Fail-closed local-host resource admission for formal queue work."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import subprocess
import tempfile
import time
import math
import re
from pathlib import Path

from .production_runtime import (_validate_formal_admission_before_models,
                                 _checkpoint_identity, load_formal_admission, load_manifest)
from .task_inputs import TrainingCatalog
from .storage_lock import allocation_lock, ensure_directory


SCHEMA = "nc_rted_formal_resource_attestation/v1"


class ResourceAttestationError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gpu_uuid(index: int) -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader,nounits", "-i", str(index)],
            text=True, capture_output=True, check=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ResourceAttestationError("bound CUDA device cannot be queried on this host") from error
    value = result.stdout.strip()
    if not value or "\n" in value:
        raise ResourceAttestationError("bound CUDA device has no unique UUID")
    return value


def gpu_memory_bytes(index: int) -> int:
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=memory.total",
                                 "--format=csv,noheader,nounits", "-i", str(index)],
                                text=True, capture_output=True, check=True, timeout=10)
        value = int(result.stdout.strip()) * 1024 ** 2
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        raise ResourceAttestationError("current GPU memory cannot be measured") from error
    if value <= 0:
        raise ResourceAttestationError("current GPU memory is invalid")
    return value


def _bound_file(path_value: object, digest: object, name: str) -> tuple[Path, dict]:
    if not isinstance(path_value, str) or not isinstance(digest, str) or len(digest) != 64:
        raise ResourceAttestationError(f"{name} needs an absolute path and SHA-256")
    path = Path(path_value)
    try:
        if not path.is_absolute() or not path.is_file():
            raise ResourceAttestationError(f"{name} is absent or its SHA-256 differs")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ResourceAttestationError(f"{name} is absent or its SHA-256 differs")
        value = json.loads(raw)
    except (OSError, ValueError) as error:
        raise ResourceAttestationError(f"{name} is invalid JSON") from error
    if not isinstance(value, dict):
        raise ResourceAttestationError(f"{name} is not an object")
    return path, value


MIN_FREE_BYTES = 20 * 1024 ** 3
FORMAL_DEADLINE = 1792724040  # 2026-10-23T02:54:00Z
RENTAL_CUTOFF = 1792080000  # 2026-10-15T16:00:00Z
PROJECT_VOLUME = Path("/root/autodl-tmp/lookaway-wm")
CACHE_KEYS = {"TMPDIR", "XDG_CACHE_HOME", "HF_HUB_CACHE", "HF_XET_CACHE", "HF_ASSETS_CACHE",
              "HF_DATASETS_CACHE", "TORCH_HOME", "TORCH_EXTENSIONS_DIR",
              "TRITON_CACHE_DIR", "CUDA_CACHE_PATH"}
ENVIRONMENT_KEYS = CACHE_KEYS | {"PYTHONPATH", "HF_HOME", "TRANSFORMERS_CACHE",
                               "PYTHONNOUSERSITE", "PYTHONDONTWRITEBYTECODE"}
QUALIFICATION_MAX_AGE = 7 * 24 * 3600


def publish_attestation(document: dict, output: str | Path) -> str:
    """Durably publish once without exposing a partial or replacing evidence."""
    target = _contained(PROJECT_VOLUME.resolve(), str(output), "attestation output")
    encoded = (json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    ensure_directory(target.parent, MIN_FREE_BYTES)
    with allocation_lock(target.parent):
        # A prior crash may have left any newly created ancestor unsynced.
        # Recovery must finish this even when the final file already exists.
        _sync_ancestry(target.parent, PROJECT_VOLUME.resolve())
        # Recovery runs even if link() completed before an interrupted fsync.
        for stale in target.parent.glob(target.name + ".*.tmp"):
            if stale.is_file() and not stale.is_symlink() and stale.read_bytes() == encoded:
                stale.unlink()
        fd = os.open(target.parent, os.O_DIRECTORY)
        try: os.fsync(fd)
        finally: os.close(fd)
        if target.is_file():
            if sha256_file(target) == hashlib.sha256(encoded).hexdigest():
                with target.open("rb") as stream: os.fsync(stream.fileno())
                fd = os.open(target.parent, os.O_DIRECTORY)
                try: os.fsync(fd)
                finally: os.close(fd)
                return sha256_file(target)
            raise ResourceAttestationError("conflicting attestation output already exists")
        block = max(4096, os.statvfs(target.parent).f_frsize)
        if shutil.disk_usage(target.parent).free < MIN_FREE_BYTES + len(encoded) + 2 * block:
            raise ResourceAttestationError("attestation publication would violate disk reserve")
        descriptor, temporary = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=target.parent)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(encoded); stream.flush(); os.fsync(stream.fileno())
            if shutil.disk_usage(target.parent).free < MIN_FREE_BYTES:
                raise ResourceAttestationError("attestation publication would violate disk reserve")
            try:
                os.link(temporary, target)
            except FileExistsError:
                if not target.is_file() or sha256_file(target) != hashlib.sha256(encoded).hexdigest():
                    raise ResourceAttestationError("conflicting attestation output already exists")
            fd = os.open(target.parent, os.O_DIRECTORY)
            try: os.fsync(fd)
            finally: os.close(fd)
            _sync_ancestry(target.parent, PROJECT_VOLUME.resolve())
        finally:
            if Path(temporary).unlink(missing_ok=True) is None:
                fd = os.open(target.parent, os.O_DIRECTORY)
                try: os.fsync(fd)
                finally: os.close(fd)
    return hashlib.sha256(encoded).hexdigest()


def _sync_ancestry(path: Path, volume: Path) -> None:
    _contained(volume, str(path), "publication directory")
    while True:
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        if path == volume:
            return
        path = path.parent


def _accepted(name: object, path: Path, digest: str, evidence: dict[str, tuple[str, str]], label: str) -> None:
    if not isinstance(name, str) or evidence.get(name) != (str(path), digest):
        raise ResourceAttestationError(f"{label} is not accepted exact evidence")


def _finite_positive(value: object, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ResourceAttestationError(f"{label} must be finite and positive")
    return float(value)


def _contained(volume: Path, value: object, label: str) -> Path:
    if not isinstance(value, str):
        raise ResourceAttestationError(f"{label} is absent")
    original = Path(value)
    path = original.resolve()
    if not original.is_absolute() or ".." in original.parts or original != path:
        raise ResourceAttestationError(f"{label} is not a canonical absolute path")
    if path != volume and volume not in path.parents:
        raise ResourceAttestationError(f"{label} escapes the approved data volume")
    ancestor = path
    while not ancestor.exists():
        ancestor = ancestor.parent
    if ancestor.stat().st_dev != volume.stat().st_dev:
        raise ResourceAttestationError(f"{label} is on an unapproved filesystem")
    return path


def validate_environment(environment: object, volume: Path) -> None:
    if (not isinstance(environment, dict) or set(environment) - ENVIRONMENT_KEYS or
            not CACHE_KEYS.issubset(environment) or
            any(not isinstance(value, str) or not value for value in environment.values()) or
            environment.get("PYTHONNOUSERSITE") != "1" or
            environment.get("PYTHONDONTWRITEBYTECODE") != "1"):
        raise ResourceAttestationError("formal environment must bind caches and disable user startup/bytecode")
    for key in CACHE_KEYS | ({"TRANSFORMERS_CACHE"} & environment.keys()):
        _contained(volume, environment[key], key)
    temporary = Path(environment["TMPDIR"])
    if not temporary.is_dir() or not os.access(temporary, os.W_OK | os.X_OK):
        raise ResourceAttestationError("qualified temporary directory is not usable")
    # HF_HOME is the existing credential/config root; actual HF data caches
    # above are mandatory and remain on the project volume.
    if environment.get("PYTHONPATH") != str(Path(__file__).resolve().parents[1]):
        raise ResourceAttestationError("formal PYTHONPATH is not the admitted source root")


def python_runtime_probe(interpreter: dict, environment: dict) -> dict:
    """Measure the exact local interpreter and import startup without CUDA work."""
    path = Path(interpreter["path"])
    with path.open("rb") as stream:
        if stream.read(4) != b"\x7fELF":
            raise ResourceAttestationError("qualified interpreter must be a local Python ELF, not a wrapper")
    program = '''import base64, hashlib, importlib, importlib.metadata, json, pathlib, site, sys, tempfile
packages = {}
for name in ("torch", "transformers", "peft", "accelerate", "safetensors", "tokenizers", "numpy"):
    module = importlib.import_module(name)
    distribution = importlib.metadata.distribution(name)
    metadata = {}
    for filename in ("METADATA", "RECORD"):
        value = distribution.read_text(filename)
        metadata[filename] = hashlib.sha256(value.encode()).hexdigest() if value is not None else None
    files = {}
    for relative in distribution.files or ():
        expected = relative.hash
        if expected is None:
            continue
        filename = pathlib.Path(distribution.locate_file(relative)).resolve()
        digest = hashlib.new(expected.mode)
        with filename.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 << 20), b""):
                digest.update(chunk)
        files[str(relative)] = {"algorithm": expected.mode, "digest": digest.hexdigest(),
                               "record_digest": expected.value}
    if not files:
        raise RuntimeError("dependency has no verifiable installed file records")
    packages[name] = {"version": distribution.version, "module": str(pathlib.Path(module.__file__).resolve()), "metadata": metadata,
                      "installed_files_sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()}
project = importlib.import_module("nc_rted.production_runtime")
startup = {}
startup_paths = [pathlib.Path(sys.prefix) / "pyvenv.cfg"]
for directory in site.getsitepackages():
    startup_paths.extend(pathlib.Path(directory).glob("*.pth"))
for name in ("site", "sitecustomize", "usercustomize"):
    module = sys.modules.get(name)
    if module is not None and getattr(module, "__file__", None):
        startup_paths.append(pathlib.Path(module.__file__))
for filename in startup_paths:
    if filename.is_file():
        startup[str(filename.resolve())] = hashlib.sha256(filename.read_bytes()).hexdigest()
print(json.dumps({"executable": sys.executable, "prefix": sys.prefix, "base_prefix": sys.base_prefix,
 "version": sys.version, "project_module": str(pathlib.Path(project.__file__).resolve()), "packages": packages,
 "startup": startup, "temporary_directory": tempfile.gettempdir()}, sort_keys=True))
'''
    try:
        result = subprocess.run([str(path), "-c", program], env=environment,
                                cwd=str(PROJECT_VOLUME), capture_output=True, text=True,
                                check=True, timeout=60)
        measured = json.loads(result.stdout)
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        raise ResourceAttestationError("qualified controlled-environment local import probe failed") from error
    if measured.get("executable") != str(path):
        raise ResourceAttestationError("interpreter invocation differs from qualified Python")
    if measured.get("temporary_directory") != environment.get("TMPDIR"):
        raise ResourceAttestationError("effective temporary directory differs from qualification")
    return measured


def validate_qualification(document: dict, *, identity: dict, source_sha256: str,
                           host: str, device_uuid: str, interpreter: dict, environment: dict,
                           budget: float, now: float, actual_runtime: dict | None = None) -> None:
    """One substantive contract shared by publisher and queue admission.

    Authority is the queue's exact accepted-evidence registration, not PASS
    strings. GPU measurements must be supplied by that accepted qualification;
    the local startup is independently executed and compared at admission.
    """
    if (document.get("schema") != "nc_rted_runtime_qualification/v2" or
            document.get("status") != "PASS_GPU_KERNEL_AND_INHERITED_RUNTIME_IMPORTS" or
            document.get("host") != host or document.get("gpu_uuid") != device_uuid or
            document.get("runtime_identity") != identity or document.get("source_sha256") != source_sha256 or
            document.get("workload") != {"run_identity": identity, "updates": 1000, "kind": "formal_train"} or
            document.get("runtime_environment") != {"interpreter": interpreter, "environment": environment}):
        raise ResourceAttestationError("qualification workload/runtime/host/device differs")
    start = _finite_positive(document.get("valid_from_utc_epoch"), "qualification start")
    end = _finite_positive(document.get("valid_until_utc_epoch"), "qualification expiry")
    measurements = document.get("measurements", {})
    measured_at = _finite_positive(measurements.get("measured_at_utc_epoch"), "measurement time")
    if not (start <= measured_at <= now < end and now - measured_at <= QUALIFICATION_MAX_AGE and now + budget <= end):
        raise ResourceAttestationError("qualification is stale, future dated, or expires during the run")
    envelope = document.get("resource_envelope", {})
    available = _finite_positive(envelope.get("device_memory_bytes"), "qualified device memory")
    required = _finite_positive(envelope.get("required_memory_bytes"), "qualified required memory")
    allocated = _finite_positive(measurements.get("peak_cuda_allocated_bytes"), "measured allocation")
    reserved = _finite_positive(measurements.get("peak_cuda_reserved_bytes"), "measured reservation")
    update_seconds = _finite_positive(measurements.get("seconds_per_update_upper_bound"), "measured update bound")
    overhead = _finite_positive(measurements.get("setup_checkpoint_seconds_upper_bound"), "measured overhead bound")
    if not (allocated <= reserved <= required <= available) or 1000 * update_seconds + overhead > budget:
        raise ResourceAttestationError("measured memory/time envelope does not fit allocation")
    if (measurements.get("forward_backward_completed") is not True or
            measurements.get("complete_long_input") is not True or
            type(measurements.get("optimizer_updates")) is not int or measurements["optimizer_updates"] < 1):
        raise ResourceAttestationError("qualification lacks actual complete workload measurements")
    probe = measurements.get("local_import_probe", {})
    if (probe.get("status") != "PASS" or probe.get("interpreter") != interpreter or
            probe.get("environment") != environment):
        raise ResourceAttestationError("qualification lacks its bound local import probe")
    _finite_positive(probe.get("duration_seconds"), "local import duration")
    actual_runtime = python_runtime_probe(interpreter, environment) if actual_runtime is None else actual_runtime
    if probe.get("runtime_identity") != actual_runtime:
        raise ResourceAttestationError("qualified dependencies/startup differ from actual interpreter imports")


def formal_runtime_identity(runtime) -> dict:
    """Derive the same checkpoint identity the formal worker admits."""
    catalog_document = runtime.document["catalog"]
    catalog = TrainingCatalog.load(catalog_document["manifest_directory"], catalog_document["training_annotations"],
                                   expected_provenance_sha256=catalog_document["provenance_sha256"])
    return _checkpoint_identity(runtime, catalog)


def verify_attestation(payload: dict, job_key: str, accepted_evidence: dict[str, tuple[str, str]], reservation: dict | None = None,
                       *, candidate_document: dict | None = None) -> None:
    """Validate the local, hash-bound resource contract before a queue claim."""
    if candidate_document is None:
        _, document = _bound_file(payload.get("resource_attestation"), payload.get("resource_attestation_sha256"), "resource attestation")
    else:
        # Publisher validates before allocation; runtime queue callers always
        # use the hash-bound file route above.
        document = candidate_document
    binding = document.get("binding")
    execution = document.get("execution")
    contract = document.get("contract")
    qualification = document.get("qualification")
    authorization = document.get("authorization")
    if (document.get("schema") != SCHEMA or document.get("status") != "PASS" or not isinstance(binding, dict) or
            not isinstance(execution, dict) or not isinstance(contract, dict) or not isinstance(qualification, dict) or not isinstance(authorization, dict)):
        raise ResourceAttestationError("resource attestation schema or status is invalid")
    required = ("runtime_config", "runtime_config_sha256", "formal_admission", "formal_admission_sha256", "run_identity", "frozen_source_sha256")
    if any(binding.get(key) != payload.get(key) for key in required) or binding.get("job_key") != job_key:
        raise ResourceAttestationError("resource attestation does not bind this formal payload")
    runtime_path, _ = _bound_file(binding["runtime_config"], binding["runtime_config_sha256"], "runtime config")
    admission_path, _ = _bound_file(binding["formal_admission"], binding["formal_admission_sha256"], "formal admission")
    _accepted(payload.get("runtime_evidence"), runtime_path, binding["runtime_config_sha256"], accepted_evidence, "runtime config")
    _accepted(payload.get("formal_admission_evidence"), admission_path, binding["formal_admission_sha256"], accepted_evidence, "formal admission")
    try:
        runtime = load_manifest(runtime_path, expected_sha256=binding["runtime_config_sha256"])
        admission = load_formal_admission(admission_path, expected_sha256=binding["formal_admission_sha256"])
        identity = formal_runtime_identity(runtime)
        _validate_formal_admission_before_models(admission, identity)
    except Exception as error:
        raise ResourceAttestationError(f"production formal runtime/admission validation failed: {error}") from error
    if binding["run_identity"] != identity or runtime.document["hashes"]["code_sha256"] != binding["frozen_source_sha256"]:
        raise ResourceAttestationError("runtime config is not bound to the attested run/source")
    if payload.get("progress_path") != runtime.run["progress_path"] or payload.get("checkpoint_root") != runtime.run["checkpoint_root"]:
        raise ResourceAttestationError("formal paths differ from the admitted runtime destination")
    source_files = admission["source_files"]
    for entry in (Path(__file__).resolve().parents[2] / "scripts" / "nc_rted_train.py", Path(__file__).resolve().parents[2] / "scripts" / "nc_rted_queue.py"):
        expected = source_files.get(str(entry))
        if not isinstance(expected, str) or sha256_file(entry) != expected:
            raise ResourceAttestationError("formal admission omits or changes an executed entry point")
    if execution.get("host") != socket.gethostname() or execution.get("physical_gpu") != payload.get("physical_gpu"):
        raise ResourceAttestationError("resource attestation is for a different execution host/device")
    volume = Path(contract.get("data_volume", "")).resolve()
    if contract.get("data_volume") != str(PROJECT_VOLUME.resolve()) or volume != PROJECT_VOLUME.resolve():
        raise ResourceAttestationError("resource contract must use the canonical approved project volume")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", job_key):
        raise ResourceAttestationError("formal job key is not a safe single path component")
    environment = payload.get("execution_environment")
    validate_environment(environment, volume)
    if execution.get("environment") != environment:
        raise ResourceAttestationError("formal environment differs from attestation")
    interpreter = execution.get("interpreter")
    if (not isinstance(interpreter, dict) or interpreter != payload.get("interpreter") or
            not isinstance(interpreter.get("path"), str) or not Path(interpreter["path"]).is_absolute() or
            sha256_file(Path(interpreter["path"])) != interpreter.get("launcher_sha256") or
            sha256_file(Path(interpreter["path"]).resolve()) != interpreter.get("target_sha256")):
        raise ResourceAttestationError("formal interpreter/runtime identity is not attested")
    if execution.get("gpu_uuid") != gpu_uuid(int(payload["physical_gpu"])):
        raise ResourceAttestationError("attested GPU UUID is not the current execution device")
    now = time.time()
    lease_expiry = _finite_positive(execution.get("lease_expires_utc_epoch"), "resource lease expiry")
    if not isinstance(execution.get("lease_id"), str) or not execution["lease_id"] or lease_expiry <= now:
        raise ResourceAttestationError("resource lease is absent or expired")
    authorization_path, authorization_doc = _bound_file(authorization.get("path"), authorization.get("sha256"), "resource authorization")
    _accepted(payload.get("resource_authorization_evidence"), authorization_path, authorization.get("sha256"), accepted_evidence, "resource authorization")
    if (authorization_doc.get("schema") != "nc_rted_resource_authorization/v1" or authorization_doc.get("status") != "PASS" or
            authorization_doc.get("host") != execution["host"] or authorization_doc.get("gpu_uuid") != execution["gpu_uuid"] or
            authorization_doc.get("lease_id") != execution["lease_id"] or authorization_doc.get("project_volume") != contract.get("data_volume") or
            _finite_positive(authorization_doc.get("max_budget_seconds"), "authorized budget") < float(contract.get("run_budget_seconds", 0)) or
            _finite_positive(authorization_doc.get("min_free_bytes"), "authorized reserve") > float(contract.get("min_free_bytes", 0)) or
            _finite_positive(authorization_doc.get("deadline_utc_epoch"), "authorized deadline") < float(contract.get("deadline_utc_epoch", 0)) or
            _finite_positive(authorization_doc.get("lease_expires_utc_epoch"), "authorized lease") < lease_expiry):
        raise ResourceAttestationError("resource authorization does not cover the admitted contract")
    name = qualification.get("accepted_evidence_name")
    expected = accepted_evidence.get(name) if isinstance(name, str) else None
    qualification_path, qualification_doc = _bound_file(qualification.get("path"), qualification.get("sha256"), "qualification report")
    if (not expected or qualification.get("path") != expected[0] or qualification.get("sha256") != expected[1] or
            qualification.get("status") != "PASS_GPU_KERNEL_AND_INHERITED_RUNTIME_IMPORTS"):
        raise ResourceAttestationError("qualification is not an accepted registered runtime report")
    validate_qualification(qualification_doc, identity=identity, source_sha256=binding["frozen_source_sha256"],
                           host=execution["host"], device_uuid=execution["gpu_uuid"], interpreter=interpreter,
                           environment=environment, budget=_finite_positive(payload.get("run_budget_seconds"), "run budget"), now=now)
    if qualification_doc["resource_envelope"]["device_memory_bytes"] != gpu_memory_bytes(int(payload["physical_gpu"])):
        raise ResourceAttestationError("measured device capacity differs from the qualified device")
    if (contract.get("data_volume") != payload.get("data_volume") or contract.get("min_free_bytes") != payload.get("min_free_bytes") or
            contract.get("run_budget_seconds") != payload.get("run_budget_seconds") or contract.get("deadline_utc_epoch") != payload.get("deadline_utc_epoch")):
        raise ResourceAttestationError("resource contract differs from the formal payload")
    reserve = _finite_positive(contract.get("min_free_bytes"), "disk reserve")
    budget = _finite_positive(contract.get("run_budget_seconds"), "run budget")
    deadline = _finite_positive(contract.get("deadline_utc_epoch"), "deadline")
    if (reserve < MIN_FREE_BYTES or deadline != FORMAL_DEADLINE or deadline <= now or
            lease_expiry > RENTAL_CUTOFF or now + budget > lease_expiry or now + budget > deadline):
        raise ResourceAttestationError("resource contract deadline is absent or expired")
    if shutil.disk_usage(contract["data_volume"]).free < reserve:
        raise ResourceAttestationError("resource contract disk reserve is no longer available")
    volume = Path(contract["data_volume"]).resolve()
    for key in ("run_dir", "progress_path", "checkpoint_root"):
        _contained(volume, payload.get(key), key)
    outputs = payload.get("expected_outputs")
    admitted_checkpoint = str((Path(payload["checkpoint_root"]) / "final" / "manifest.json").resolve())
    if (not isinstance(outputs, list) or len(outputs) != 1 or outputs[0].get("path") != admitted_checkpoint or
            outputs[0].get("artifact_type") != "checkpoint" or outputs[0].get("semantic") != "formal_training" or
            outputs[0].get("run_identity") != binding["run_identity"]):
        raise ResourceAttestationError("formal output contract is not the admitted final checkpoint")
    if binding.get("execution_inputs") != {key: payload.get(key) for key in ("command", "execution_environment", "interpreter", "run_dir", "progress_path", "checkpoint_root", "expected_outputs")}:
        raise ResourceAttestationError("formal execution/output contract differs from attestation")
    if reservation is not None and (reservation.get("lease_id") != execution["lease_id"] or reservation.get("physical_gpu") != execution["physical_gpu"] or reservation.get("host") != execution["host"]):
        raise ResourceAttestationError("attestation reservation is not held by this attempt")


def resource_lease_expiry(payload: dict) -> float:
    """Cheap integrity check of the compact resource contract during supervision."""
    _, document = _bound_file(payload.get("resource_attestation"), payload.get("resource_attestation_sha256"), "resource attestation")
    execution = document.get("execution")
    if not isinstance(execution, dict):
        raise ResourceAttestationError("resource attestation execution contract is invalid")
    return _finite_positive(execution.get("lease_expires_utc_epoch"), "resource lease expiry")


def lease_expired(payload: dict) -> bool:
    try: return resource_lease_expiry(payload) <= time.time()
    except (ResourceAttestationError, OSError, ValueError, TypeError): return True
