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
BUNDLE_SCHEMA = "nc_rted_formal_bundle_resource_attestation/v1"


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
FORMAL_PATH = "/root/autodl-tmp/lookaway-wm/.venv-reactvau/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
ENVIRONMENT_KEYS = CACHE_KEYS | {"PYTHONPATH", "HF_HOME", "TRANSFORMERS_CACHE",
                               "PYTHONNOUSERSITE", "PYTHONDONTWRITEBYTECODE", "PATH"}
QUALIFICATION_MAX_AGE = 7 * 24 * 3600
SOURCE34_APPLICABILITY_SCHEMA = "nc_rted_source34_seed_applicability/v1"
GROUPS = ("A", "U", "S", "F")


def normalized_runtime_for_seed(document: object) -> dict:
    """Return the immutable runtime contract after removing seed relocation."""
    if not isinstance(document, dict) or not isinstance(document.get("run"), dict) or not isinstance(document.get("hashes"), dict):
        raise ResourceAttestationError("source34 runtime document is incomplete")
    value = json.loads(json.dumps(document))
    for key in ("run_id", "seed", "checkpoint_root", "progress_path"):
        value["run"].pop(key, None)
    for key in ("code_sha256", "runtime_sha256"):
        value["hashes"].pop(key, None)
    return value


def _manifest_code_sha256(manifest: dict, name: str) -> tuple[dict[str, str], str]:
    files = manifest.get("files")
    if (manifest.get("schema") != "nc_rted_interleaved_source_manifest/v1" or not isinstance(files, dict) or
            not files or any(not isinstance(key, str) or Path(key).is_absolute() or ".." in Path(key).parts or
                             not isinstance(value, str) or len(value) != 64 for key, value in files.items())):
        raise ResourceAttestationError(f"{name} differs")
    code = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if manifest.get("code_sha256") != code:
        raise ResourceAttestationError(f"{name} code identity differs")
    return files, code


def _verify_source34_applicability(document: object, qualification_document: dict, qualification_path: Path,
                                    qualification_sha256: str, identities: dict, budget: float,
                                    target_runtimes: dict[str, tuple[dict, dict]] | None = None) -> dict | None:
    """Accept a conservative source33 measurement projection for seed 42/2026 only."""
    if document is None:
        return None
    if not isinstance(document, dict) or set(document) != {"schema", "status", "measured_qualification", "measured_identities", "target_seed", "target_identities", "source_transition", "invariants", "projection", "measured_runtimes", "target_runtimes"}:
        raise ResourceAttestationError("source34 applicability schema differs")
    if document["schema"] != SOURCE34_APPLICABILITY_SCHEMA or document["status"] != "PASS_CONSERVATIVE_SEED_APPLICABILITY":
        raise ResourceAttestationError("source34 applicability is not accepted")
    if document["measured_qualification"] != {"path": str(qualification_path), "sha256": qualification_sha256}:
        raise ResourceAttestationError("source34 applicability binds a different source33 measurement")
    if type(document["target_seed"]) is not int or document["target_seed"] not in {42, 2026} or document["target_identities"] != identities:
        raise ResourceAttestationError("source34 applicability target identities differ")
    if set(document["measured_identities"]) != {"A", "U", "S", "F"} or any(item.get("seed") != "17" for item in document["measured_identities"].values()):
        raise ResourceAttestationError("source34 applicability measured seed differs")
    profile_binding = qualification_document.get("qualification_profile")
    if not isinstance(profile_binding, dict) or set(profile_binding) != {"path", "sha256"}:
        raise ResourceAttestationError("source33 qualification profile binding differs")
    _, profile = _bound_file(profile_binding["path"], profile_binding["sha256"], "source33 qualification profile")
    if (profile.get("schema") != "nc_rted_interleaved_formal_qualification_profile/v1" or
            profile.get("status") != "NON_ADMITTED_FORMAL_PROFILE" or
            profile.get("member_identities") != document["measured_identities"] or
            profile.get("updates") != 1000 or profile.get("accumulation") != 8 or
            profile.get("shared_preparation") != "frozen_provider_only" or profile.get("common_recovery") is not True):
        raise ResourceAttestationError("source33 qualification profile invariants differ")
    if set(document["measured_runtimes"]) != set(GROUPS) or set(document["target_runtimes"]) != set(GROUPS):
        raise ResourceAttestationError("source34 applicability runtime bindings differ")
    for group in GROUPS:
        _, measured_runtime = _bound_file(document["measured_runtimes"][group]["path"], document["measured_runtimes"][group]["sha256"], "measured runtime")
        expected_measured = {"path": profile["members"][group].get("runtime"), "sha256": profile["members"][group].get("runtime_sha256")}
        actual_target = None if target_runtimes is None else target_runtimes.get(group)
        if (document["measured_runtimes"][group] != expected_measured or not isinstance(actual_target, tuple) or
                document["target_runtimes"][group] != actual_target[1] or
                normalized_runtime_for_seed(measured_runtime) != normalized_runtime_for_seed(actual_target[0])):
            raise ResourceAttestationError("source34 applicability runtime normalization differs")
    if any(item.get("seed") != str(document["target_seed"]) for item in identities.values()):
        raise ResourceAttestationError("source34 applicability target seed differs")
    transition = document["source_transition"]
    if (not isinstance(transition, dict) or set(transition) != {"allowed_changed_files", "unchanged_files", "measured_manifest", "target_manifest"} or
            transition["allowed_changed_files"] != ["src/nc_rted/resource_attestation.py"] or
            not isinstance(transition["unchanged_files"], dict) or not transition["unchanged_files"]):
        raise ResourceAttestationError("source34 applicability source transition differs")
    _, measured_manifest = _bound_file(transition["measured_manifest"].get("path"), transition["measured_manifest"].get("sha256"), "source33 manifest")
    _, target_manifest = _bound_file(transition["target_manifest"].get("path"), transition["target_manifest"].get("sha256"), "source34 manifest")
    measured_files, measured_code = _manifest_code_sha256(measured_manifest, "source33 manifest")
    target_files, target_code = _manifest_code_sha256(target_manifest, "source34 manifest")
    changed = {name for name in set(measured_files or ()) | set(target_files or ()) if (measured_files or {}).get(name) != (target_files or {}).get(name)}
    if changed != {"src/nc_rted/resource_attestation.py"} or transition["unchanged_files"] != {name: value for name, value in measured_files.items() if name != "src/nc_rted/resource_attestation.py"}:
        raise ResourceAttestationError("source34 applicability source bytes differ")
    if (qualification_document.get("source_sha256") != measured_code or
            any(value.get("code_sha256") != measured_code for value in document["measured_identities"].values()) or
            any(value.get("code_sha256") != target_code for value in identities.values())):
        raise ResourceAttestationError("source34 applicability source identities differ")
    invariants = document["invariants"]
    if invariants != {"samples": 8000, "updates": 1000, "accumulation": 8, "shared_preparation": "frozen_provider_only", "common_recovery": True, "sampler_rng_difference_explicit": True}:
        raise ResourceAttestationError("source34 applicability runtime invariants differ")
    projection = document["projection"]
    if (not isinstance(projection, dict) or set(projection) != {"measured_seconds_per_bundle_update_upper_bound", "measured_setup_checkpoint_seconds_upper_bound", "safety_multiplier", "projected_total_seconds_upper_bound", "is_measured_target_timing"} or
            projection.get("is_measured_target_timing") is not False or
            projection.get("measured_seconds_per_bundle_update_upper_bound") != qualification_document.get("measurements", {}).get("seconds_per_bundle_update_upper_bound") or
            projection.get("measured_setup_checkpoint_seconds_upper_bound") != qualification_document.get("measurements", {}).get("setup_checkpoint_seconds_upper_bound") or
            _finite_positive(projection.get("safety_multiplier"), "source34 safety multiplier") < 1 or
            not math.isclose((1000 * _finite_positive(projection.get("measured_seconds_per_bundle_update_upper_bound"), "source33 measured update bound") +
                              _finite_positive(projection.get("measured_setup_checkpoint_seconds_upper_bound"), "source33 measured setup bound")) * projection["safety_multiplier"],
                             _finite_positive(projection.get("projected_total_seconds_upper_bound"), "source34 projected target bound"), rel_tol=0, abs_tol=1e-6) or
            projection["projected_total_seconds_upper_bound"] > budget):
        raise ResourceAttestationError("source34 applicability projected budget differs")
    return document


def _source34_probe_matches(measured: object, actual: object, measured_environment: dict, target_environment: dict,
                            measured_manifest: dict, target_manifest: dict) -> bool:
    """Permit only the proven source-root relocation in the import probe."""
    if not isinstance(measured, dict) or not isinstance(actual, dict):
        return False
    if not isinstance(measured, dict) or not isinstance(actual, dict):
        return False
    measured_module, target_module = measured.get("project_module"), actual.get("project_module")
    if not isinstance(measured_module, str) or not isinstance(target_module, str):
        return False
    relative = Path("nc_rted/production_runtime.py")
    measured_root = Path(measured_environment["PYTHONPATH"]).resolve()
    target_root = Path(target_environment["PYTHONPATH"]).resolve()
    if (Path(measured_module).resolve() != measured_root / relative or Path(target_module).resolve() != target_root / relative or
            measured_manifest.get("files", {}).get("src/nc_rted/production_runtime.py") != target_manifest.get("files", {}).get("src/nc_rted/production_runtime.py")):
        return False
    candidate = json.loads(json.dumps(measured))
    candidate["project_module"] = target_module
    return candidate == actual


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
    if environment.get("PATH") != FORMAL_PATH:
        raise ResourceAttestationError("formal PATH is not the admitted deterministic path")


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


def verify_bundle_attestation(payload: dict, job_key: str, accepted_evidence: dict[str, tuple[str, str]],
                              reservation: dict | None = None) -> None:
    """Validate an attestation for four admitted groups sharing preparation only."""
    from .formal_bundle import FormalBundleContractError, load_bundle, validate_members
    _, document = _bound_file(payload.get("resource_attestation"), payload.get("resource_attestation_sha256"),
                              "formal bundle resource attestation")
    binding, execution, contract = document.get("binding"), document.get("execution"), document.get("contract")
    qualification, authorization = document.get("qualification"), document.get("authorization")
    if (document.get("schema") != BUNDLE_SCHEMA or document.get("status") != "PASS" or
            not all(isinstance(value, dict) for value in (binding, execution, contract, qualification, authorization))):
        raise ResourceAttestationError("formal bundle resource attestation schema or status is invalid")
    required = ("bundle_config", "bundle_config_sha256", "member_identities", "frozen_source_sha256")
    if binding.get("job_key") != job_key or any(binding.get(key) != payload.get(key) for key in required):
        raise ResourceAttestationError("resource attestation does not bind this formal bundle payload")
    try:
        bundle_path, bundle = load_bundle(binding["bundle_config"], binding["bundle_config_sha256"])
        members = validate_members(bundle)
    except FormalBundleContractError as error:
        raise ResourceAttestationError(str(error)) from error
    _accepted(payload.get("bundle_evidence"), bundle_path, binding["bundle_config_sha256"], accepted_evidence, "formal bundle")
    identities = {group: value["identity"] for group, value in members.items()}
    if binding["member_identities"] != identities or payload.get("member_identities") != identities:
        raise ResourceAttestationError("formal bundle member identities differ")
    checkpoint_roots = [str(Path(members[group]["runtime"].run["checkpoint_root"]).resolve())
                        for group in ("A", "U", "S", "F")]
    checkpoints = {group: str((Path(members[group]["runtime"].run["checkpoint_root"]) / "final" /
                               "manifest.json").resolve()) for group in ("A", "U", "S", "F")}
    outputs = payload.get("expected_outputs")
    expected_outputs = [
        {"path": checkpoints[group], "artifact_type": "checkpoint", "semantic": "formal_training",
         "run_identity": identities[group]}
        for group in ("A", "U", "S", "F")
    ]
    expected_outputs.append({"path": "bundle-report.json", "artifact_type": "report", "semantic": "formal_bundle",
                             "member_identities": identities, "final_checkpoints": checkpoints,
                             "bundle_checkpoint_root": str(Path(bundle["bundle_checkpoint_root"]).resolve())})
    if outputs != expected_outputs or payload.get("checkpoint_roots") != checkpoint_roots:
        raise ResourceAttestationError("formal bundle output contract differs from admitted members")
    source = next(iter(identities.values()))["code_sha256"]
    if binding["frozen_source_sha256"] != source:
        raise ResourceAttestationError("formal bundle source identity differs")
    volume = PROJECT_VOLUME.resolve()
    roots = [Path(payload.get("run_dir", "")), Path(bundle["bundle_checkpoint_root"]),
             *(Path(members[group]["runtime"].run["checkpoint_root"]) for group in ("A", "U", "S", "F")),
             *(Path(members[group]["runtime"].run["progress_path"]) for group in ("A", "U", "S", "F"))]
    canonical_roots = [_contained(volume, str(path), "formal bundle destination") for path in roots]
    if len(set(canonical_roots)) != len(canonical_roots):
        raise ResourceAttestationError("formal bundle destinations alias")
    for index, path in enumerate(canonical_roots):
        if any(path in other.parents or other in path.parents for other in canonical_roots[index + 1:]):
            raise ResourceAttestationError("formal bundle destinations overlap")
    if (payload.get("bundle_checkpoint_root") != str(canonical_roots[1]) or
            payload.get("progress_path") not in {str(path) for path in canonical_roots[6:]}):
        raise ResourceAttestationError("formal bundle queue destinations differ")
    execution_inputs = {key: payload.get(key) for key in ("command", "execution_environment", "interpreter",
                                                            "run_dir", "progress_path", "checkpoint_roots",
                                                            "bundle_checkpoint_root", "expected_outputs")}
    if binding.get("execution_inputs") != execution_inputs:
        raise ResourceAttestationError("formal bundle execution/output contract differs from attestation")
    if execution.get("host") != socket.gethostname() or execution.get("physical_gpu") != payload.get("physical_gpu"):
        raise ResourceAttestationError("formal bundle attestation is for a different execution host/device")
    environment, interpreter = payload.get("execution_environment"), payload.get("interpreter")
    validate_environment(environment, volume)
    if execution.get("environment") != environment or execution.get("interpreter") != interpreter:
        raise ResourceAttestationError("formal bundle execution environment differs")
    if (not isinstance(interpreter, dict) or not isinstance(interpreter.get("path"), str) or
            sha256_file(Path(interpreter["path"])) != interpreter.get("launcher_sha256") or
            sha256_file(Path(interpreter["path"]).resolve()) != interpreter.get("target_sha256")):
        raise ResourceAttestationError("formal bundle interpreter is not attested")
    if execution.get("gpu_uuid") != gpu_uuid(int(payload["physical_gpu"])):
        raise ResourceAttestationError("formal bundle GPU differs")
    now = time.time(); budget = _finite_positive(payload.get("run_budget_seconds"), "run budget")
    lease_expiry = _finite_positive(execution.get("lease_expires_utc_epoch"), "resource lease expiry")
    if not isinstance(execution.get("lease_id"), str) or not execution["lease_id"] or lease_expiry <= now:
        raise ResourceAttestationError("formal bundle lease is absent or expired")
    authorization_path, authorization_doc = _bound_file(authorization.get("path"), authorization.get("sha256"), "resource authorization")
    _accepted(payload.get("resource_authorization_evidence"), authorization_path, authorization.get("sha256"), accepted_evidence, "resource authorization")
    if (authorization_doc.get("schema") != "nc_rted_resource_authorization/v1" or authorization_doc.get("status") != "PASS" or
            authorization_doc.get("host") != execution["host"] or authorization_doc.get("gpu_uuid") != execution["gpu_uuid"] or
            authorization_doc.get("lease_id") != execution["lease_id"] or
            authorization_doc.get("project_volume") != contract.get("data_volume") or
            _finite_positive(authorization_doc.get("max_budget_seconds"), "authorized budget") < budget or
            _finite_positive(authorization_doc.get("min_free_bytes"), "authorized reserve") > float(contract.get("min_free_bytes", 0)) or
            _finite_positive(authorization_doc.get("deadline_utc_epoch"), "authorized deadline") < float(contract.get("deadline_utc_epoch", 0)) or
            _finite_positive(authorization_doc.get("lease_expires_utc_epoch"), "authorized lease") < lease_expiry):
        raise ResourceAttestationError("resource authorization differs from the formal bundle")
    name = qualification.get("accepted_evidence_name")
    expected = accepted_evidence.get(name) if isinstance(name, str) else None
    qualification_path, qualification_doc = _bound_file(qualification.get("path"), qualification.get("sha256"), "bundle qualification report")
    if not expected or (qualification.get("path"), qualification.get("sha256")) != expected:
        raise ResourceAttestationError("bundle qualification is not accepted evidence")
    applicability = None
    if document.get("source34_applicability") is not None:
        binding = document["source34_applicability"]
        if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
            raise ResourceAttestationError("source34 applicability binding differs")
        _, applicability_doc = _bound_file(binding["path"], binding["sha256"], "source34 applicability")
        applicability = _verify_source34_applicability(
            applicability_doc, qualification_doc, qualification_path, qualification["sha256"], identities, budget,
            {group: (members[group]["runtime"].document,
                     {"path": bundle["members"][group]["runtime"], "sha256": bundle["members"][group]["runtime_sha256"]})
             for group in GROUPS},
        )
    if applicability is not None:
        qualified_environment = qualification_doc.get("runtime_environment", {}).get("environment")
        if (not isinstance(qualified_environment, dict) or qualified_environment.get("PYTHONPATH") == environment.get("PYTHONPATH") or
                {key: value for key, value in qualified_environment.items() if key != "PYTHONPATH"} != {key: value for key, value in environment.items() if key != "PYTHONPATH"} or
                qualification_doc.get("runtime_environment", {}).get("interpreter") != interpreter):
            raise ResourceAttestationError("source34 environment relocation differs outside PYTHONPATH")
    workload = {"member_identities": identities, "updates": 1000, "kind": "formal_bundle",
                "shared_preparation": "frozen_provider_only", "common_recovery": True}
    measurements, envelope = qualification_doc.get("measurements", {}), qualification_doc.get("resource_envelope", {})
    if applicability is not None and (applicability["projection"]["measured_seconds_per_bundle_update_upper_bound"] != measurements.get("seconds_per_bundle_update_upper_bound") or applicability["projection"]["measured_setup_checkpoint_seconds_upper_bound"] != measurements.get("setup_checkpoint_seconds_upper_bound")):
        raise ResourceAttestationError("source34 projection does not preserve measured source33 timing")
    if (qualification_doc.get("schema") != "nc_rted_runtime_qualification/v2" or qualification_doc.get("status") != "PASS_GPU_KERNEL_AND_INHERITED_RUNTIME_IMPORTS" or
            qualification_doc.get("host") != execution["host"] or qualification_doc.get("gpu_uuid") != execution["gpu_uuid"] or
            qualification_doc.get("source_sha256") != (source if applicability is None else applicability["measured_identities"]["A"]["code_sha256"]) or
            qualification_doc.get("workload") != (workload if applicability is None else {"member_identities": applicability["measured_identities"], "updates": 1000, "kind": "formal_bundle", "shared_preparation": "frozen_provider_only", "common_recovery": True}) or
            (applicability is None and qualification_doc.get("runtime_environment") != {"interpreter": interpreter, "environment": environment}) or
            measurements.get("forward_backward_completed") is not True or measurements.get("complete_long_input") is not True or
            measurements.get("common_boundary_recovery_completed") is not True or
            type(measurements.get("optimizer_updates")) is not int or measurements["optimizer_updates"] < 1):
        raise ResourceAttestationError("bundle qualification does not measure this formal workload")
    start = _finite_positive(qualification_doc.get("valid_from_utc_epoch"), "qualification start")
    end = _finite_positive(qualification_doc.get("valid_until_utc_epoch"), "qualification expiry")
    measured_at = _finite_positive(measurements.get("measured_at_utc_epoch"), "measurement time")
    if not (start <= measured_at <= now < end and now - measured_at <= QUALIFICATION_MAX_AGE and now + budget <= end):
        raise ResourceAttestationError("bundle qualification is stale, future dated, or expires during the run")
    probe = measurements.get("local_import_probe", {})
    actual_probe = python_runtime_probe(interpreter, environment)
    probe_matches = probe.get("runtime_identity") == actual_probe
    if applicability is not None:
        transition = applicability["source_transition"]
        _, measured_manifest = _bound_file(transition["measured_manifest"]["path"], transition["measured_manifest"]["sha256"], "source33 manifest")
        _, target_manifest = _bound_file(transition["target_manifest"]["path"], transition["target_manifest"]["sha256"], "source34 manifest")
        probe_matches = _source34_probe_matches(probe.get("runtime_identity"), actual_probe, qualified_environment, environment,
                                                measured_manifest, target_manifest)
    if (probe.get("status") != "PASS" or probe.get("interpreter") != interpreter or
            (applicability is None and probe.get("environment") != environment) or
            (applicability is not None and probe.get("environment") != qualified_environment) or not probe_matches):
        raise ResourceAttestationError("bundle qualification local imports differ")
    available = _finite_positive(envelope.get("device_memory_bytes"), "qualified device memory")
    required_memory = _finite_positive(envelope.get("required_memory_bytes"), "qualified required memory")
    allocated = _finite_positive(measurements.get("peak_cuda_allocated_bytes"), "measured allocation")
    reserved = _finite_positive(measurements.get("peak_cuda_reserved_bytes"), "measured reservation")
    updates = _finite_positive(measurements.get("seconds_per_bundle_update_upper_bound"), "bundle update bound")
    overhead = _finite_positive(measurements.get("setup_checkpoint_seconds_upper_bound"), "measured overhead")
    projected = applicability["projection"]["projected_total_seconds_upper_bound"] if applicability else 1000 * updates + overhead
    if not (allocated <= reserved <= required_memory <= available) or projected > budget or available != gpu_memory_bytes(int(payload["physical_gpu"])):
        raise ResourceAttestationError("bundle measured memory/time envelope does not fit allocation")
    if contract != {key: payload.get(key) for key in ("data_volume", "min_free_bytes", "run_budget_seconds", "deadline_utc_epoch")}:
        raise ResourceAttestationError("formal bundle resource contract differs")
    if (contract.get("data_volume") != str(volume) or _finite_positive(contract.get("min_free_bytes"), "disk reserve") < MIN_FREE_BYTES or
            _finite_positive(contract.get("deadline_utc_epoch"), "deadline") != FORMAL_DEADLINE or now + budget > lease_expiry or lease_expiry > RENTAL_CUTOFF or
            shutil.disk_usage(volume).free < contract["min_free_bytes"]):
        raise ResourceAttestationError("formal bundle resource contract is not currently viable")
    if reservation is not None and (reservation.get("lease_id") != execution["lease_id"] or reservation.get("physical_gpu") != execution["physical_gpu"] or reservation.get("host") != execution["host"]):
        raise ResourceAttestationError("formal bundle reservation is not held by this attempt")


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
