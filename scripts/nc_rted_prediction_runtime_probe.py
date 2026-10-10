#!/usr/bin/env python3
"""Bound diagnostic in a disposable child; the parent owns deadline and report.

The parent imports only the standard library. Its POSIX allocation lock uses
storage_lock.py's filesystem-root inode, without importing unverified project
code. No parent CUDA context exists when the diagnostic child is forked.
"""
from __future__ import annotations

import argparse
import errno
import fcntl
import hashlib
import importlib.abc
import importlib.machinery
import json
import math
import os
from pathlib import Path
import re
import select
import shutil
import signal
import sys
import tempfile
import time
from contextlib import contextmanager
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
_EXECUTED_CODE = sys._getframe().f_code
PROJECT_VOLUME = Path("/root/autodl-tmp/lookaway-wm")
RESERVE = 20 * 1024 ** 3
REPORT_CAPACITY = 1024 ** 2
FINALIZATION_SECONDS = 5
ADMISSION_PARSE_SECONDS = 5
MAX_BINDING_PATH_BYTES = 4096
MAX_REPORT_INTEGER = (1 << 63) - 1
MAX_IPC_REPORT_BYTES = REPORT_CAPACITY // 2
PREFLIGHT_SCHEMA = "nc_rted_r0_blind_manifest_build/v2"
ADMISSION_SCHEMA = "nc_rted_prediction_runtime_probe_admission/v1"
_CODES = frozenset({"BOOTSTRAP_INVALID", "ADMISSION_INVALID", "OUTPUT_UNAVAILABLE",
                    "CUDA_UNAVAILABLE", "PREFLIGHT_INVALID", "SOURCE_MISMATCH",
                    "IDENTITY_INVALID", "PAYLOAD_INVALID", "MODEL_IDENTITY_INVALID",
                    "DEADLINE_EXCEEDED", "RUNTIME_FAILURE", "MEASUREMENT_FAILED"})
_REPORT_FIELDS = frozenset({
    "status", "stage", "started_at_utc_epoch", "deadline_utc_epoch", "device", "diagnostic_admission",
    "probe_script_sha256", "peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes", "formal_prediction",
    "prediction_store_written", "preflight_report", "runtime", "source_manifest", "identity_manifest",
    "model_manifest", "embedded_vision_binding", "decoder_binding", "implementation_binding", "bindings",
    "identity", "counters", "summary", "error_code", "peak_measurement_incomplete", "elapsed_seconds",
    "publication_deadline_utc_epoch", "child_pid",
})

class ProbeError(ValueError):
    def __init__(self, code):
        self.code = code if code in _CODES else "RUNTIME_FAILURE"
        super().__init__(self.code)

def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

def _bound(path, digest, code, *, max_bytes=None):
    path = Path(path)
    if (not path.is_absolute() or not path.is_file() or not isinstance(digest, str) or
            not re.fullmatch(r"[0-9a-f]{64}", digest)):
        raise ProbeError(code)
    with path.open("rb") as stream:
        raw = stream.read() if max_bytes is None else stream.read(max_bytes + 1)
    if max_bytes is not None and len(raw) > max_bytes:
        raise ProbeError(code)
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ProbeError(code)
    try:
        return path, json.loads(raw)
    except (ValueError, UnicodeError) as error:
        raise ProbeError(code) from error

_FILES = frozenset({'src/nc_rted/prediction_media.py', 'src/nc_rted/bridge.py', 'src/nc_rted/__init__.py', 'src/nc_rted/detection_provider.py', 'src/nc_rted/prediction_worker.py', 'src/nc_rted/task_inputs.py', 'src/nc_rted/observation.py', 'scripts/nc_rted_prediction_runtime_probe.py', 'scripts/nc_rted_predict.py', 'src/nc_rted/frozen_vision.py', 'src/nc_rted/prediction_inputs.py', 'src/nc_rted/detector.py', 'src/nc_rted/inherited_memory.py', 'src/nc_rted/loading.py', 'src/nc_rted/numerics.py', 'src/nc_rted/model.py', 'src/nc_rted/batches.py', 'src/nc_rted/detection_media.py', 'src/nc_rted/prediction_adapters.py', 'src/nc_rted/recovery.py', 'src/nc_rted/prediction_runtime.py', 'src/nc_rted/production_runtime.py', 'src/nc_rted/prediction_store.py', 'src/nc_rted/storage_lock.py', 'src/nc_rted/observation_cache.py', 'src/nc_rted/media_observer.py'})

def _bootstrap(path,digest):
 _,pre=_bound(path,digest,"BOOTSTRAP_INVALID")
 try:ref=pre["bindings"]["implementation"];_,doc=_bound(ref["path"],ref["sha256"],"BOOTSTRAP_INVALID");files=doc["files"]
 except (KeyError,TypeError,ProbeError) as e:raise ProbeError("BOOTSTRAP_INVALID") from e
 probe=ROOT/"scripts/nc_rted_prediction_runtime_probe.py"
 if not isinstance(doc,dict) or doc.get("schema")!="nc_rted_blind_prediction_implementation/v1" or Path(doc.get("root","")).resolve()!=ROOT or not isinstance(files,dict) or set(files)!=_FILES or _sha(probe)!=files.get("scripts/nc_rted_prediction_runtime_probe.py"):raise ProbeError("BOOTSTRAP_INVALID")
 captured={}
 for rel,h in files.items():
  p=ROOT/rel
  if not isinstance(h,str) or len(h)!=64 or p.is_symlink() or not p.is_file():raise ProbeError("BOOTSTRAP_INVALID")
  raw=p.read_bytes()
  if hashlib.sha256(raw).hexdigest()!=h:raise ProbeError("BOOTSTRAP_INVALID")
  captured[p.resolve()]=raw
 if compile(captured[probe.resolve()],_EXECUTED_CODE.co_filename,"exec",dont_inherit=True,optimize=sys.flags.optimize)!=_EXECUTED_CODE:raise ProbeError("BOOTSTRAP_INVALID")
 modules=set();packages=set()
 for rel in files:
  if rel.startswith("src/nc_rted/"):
   parts=list(Path(rel).relative_to("src").with_suffix("").parts)
   if parts[-1]=="__init__":parts=parts[:-1]
   if parts:modules.add(".".join(parts))
   packages.update(".".join(parts[:i]) for i in range(1,len(parts)))
 def owns(name):return name in modules or name in packages or name.startswith("nc_rted.")
 if any(owns(n) for n in sys.modules):raise ProbeError("BOOTSTRAP_INVALID")
 class Loader(importlib.machinery.SourceFileLoader):
  def get_code(self,fullname):return compile(captured[Path(self.path).resolve()],self.path,"exec",dont_inherit=True)
 class Finder(importlib.abc.MetaPathFinder):
  def find_spec(self,fullname,path=None,target=None):
   if not owns(fullname):return None
   spec=importlib.machinery.PathFinder.find_spec(fullname,path)
   if spec is None or not spec.origin or spec.origin in {"built-in","frozen"} or Path(spec.origin).resolve() not in captured:raise ProbeError("BOOTSTRAP_INVALID")
   spec.loader=Loader(fullname,spec.origin);return spec
 sys.path.insert(0,str(ROOT/"src"));sys.meta_path.insert(0,Finder())

def _api():
    global ModelTask, VadRequest, VauRequest, _bound_artifact, _bound_path, _no_supervision
    global PredictionInputError, PredictionExecutionError
    global load_model_artifact, verify_implementation_manifest, default_factory
    global validate_loaded_model_identity, validate_vad_payload, validate_vau_payload
    global count_slow_execution, count_generation_execution
    from nc_rted.prediction_inputs import (ModelTask, VadRequest, VauRequest, PredictionInputError, _bound_artifact, _bound_path,
        _no_supervision, load_model_artifact, verify_implementation_manifest)
    from nc_rted.prediction_runtime import default_factory
    from nc_rted.prediction_worker import (PredictionExecutionError, validate_loaded_model_identity,
        validate_vad_payload, validate_vau_payload, count_slow_execution, count_generation_execution)


def _sync(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def _allocation(path, *, deadline):
    # Same filesystem-root POSIX record lock as nc_rted.storage_lock. The
    # standard-library-only supervisor never opens that inode elsewhere.
    root = path
    while not root.exists():
        root = root.parent
    device = root.stat().st_dev
    while root.parent != root and root.parent.stat().st_dev == device:
        root = root.parent
    lock = root / ".nc_rted_allocation.lock"
    flags = os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC
    try:
        descriptor = os.open(lock, flags)
    except FileNotFoundError:
        if shutil.disk_usage(root).free < RESERVE + 8192:
            raise ProbeError("OUTPUT_UNAVAILABLE")
        try:
            descriptor = os.open(lock, flags | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            descriptor = os.open(lock, flags)
    locked = False
    try:
        while True:
            try:
                fcntl.lockf(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                locked = True
                break
            except OSError as error:
                if error.errno not in {errno.EACCES, errno.EAGAIN}:
                    raise
                remaining = deadline - time.time()
                if remaining <= 0:
                    raise ProbeError("DEADLINE_EXCEEDED") from error
                time.sleep(min(0.05, remaining))
        if time.time() >= deadline:
            raise ProbeError("DEADLINE_EXCEEDED")
        yield
    finally:
        if locked:
            fcntl.lockf(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _canonical_output(value):
    path = Path(value)
    approved = PROJECT_VOLUME.resolve(strict=True)
    if not path.is_absolute() or ".." in path.parts:
        raise ProbeError("OUTPUT_UNAVAILABLE")
    # Reject aliases before creation, including dangling destination symlinks.
    if path != path.resolve() or os.path.lexists(path):
        raise ProbeError("OUTPUT_UNAVAILABLE")
    try:
        path.relative_to(approved)
    except ValueError as error:
        raise ProbeError("OUTPUT_UNAVAILABLE") from error
    ancestor = path.parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    if ancestor.stat().st_dev != approved.stat().st_dev:
        raise ProbeError("OUTPUT_UNAVAILABLE")
    return path, ancestor


def _read_admission(args, *, started=None):
    started = time.time() if started is None else started
    _, admission = _bound(args.diagnostic_admission, args.diagnostic_admission_sha256,
                          "ADMISSION_INVALID", max_bytes=65536)
    fields = {"schema", "status", "kind", "device", "data_volume", "min_free_bytes",
              "run_budget_seconds", "deadline_utc_epoch", "output", "preflight_report_sha256"}
    if (not isinstance(admission, dict) or set(admission) != fields or
            admission.get("schema") != ADMISSION_SCHEMA or admission.get("status") != "PASS" or
            args.kind not in {"vad", "vau"} or admission.get("kind") != args.kind or
            admission.get("device") != args.device or not re.fullmatch(r"cuda:[0-9]+", args.device) or
            admission.get("output") != args.output or
            admission.get("data_volume") != str(PROJECT_VOLUME.resolve(strict=True)) or
            admission.get("preflight_report_sha256") != args.preflight_report_sha256 or
            type(admission.get("min_free_bytes")) is not int or admission["min_free_bytes"] < RESERVE or
            type(admission.get("run_budget_seconds")) is not int or admission["run_budget_seconds"] < 1 or
            type(admission.get("deadline_utc_epoch")) not in {int, float} or
            not math.isfinite(admission["deadline_utc_epoch"])):
        raise ProbeError("ADMISSION_INVALID")
    publication_deadline = min(started + admission["run_budget_seconds"], admission["deadline_utc_epoch"])
    execution_deadline = publication_deadline - FINALIZATION_SECONDS
    if execution_deadline <= time.time():
        raise ProbeError("DEADLINE_EXCEEDED")
    return dict(admission), execution_deadline, publication_deadline


def _read_admission_operation(args, started):
    return _read_admission(args, started=started)


def _hash_script_operation(path):
    return _sha(path)


def _reserve_slot(args, admission, *, deadline):
    path, ancestor = _canonical_output(args.output)
    with _allocation(ancestor, deadline=deadline):
        if time.time() >= deadline:
            raise ProbeError("DEADLINE_EXCEEDED")
        path, ancestor = _canonical_output(args.output)
        missing = len(path.parent.relative_to(ancestor).parts)
        block = max(4096, os.statvfs(ancestor).f_frsize)
        reserve = admission["min_free_bytes"]
        if shutil.disk_usage(ancestor).free < reserve + REPORT_CAPACITY + (2 * missing + 8) * block:
            raise ProbeError("OUTPUT_UNAVAILABLE")
        current = ancestor
        for part in path.parent.relative_to(ancestor).parts:
            current = current / part
            current.mkdir()
            _sync(current)
            _sync(current.parent)
        descriptor, temporary = tempfile.mkstemp(prefix="." + path.name + ".reserved-", dir=path.parent)
        try:
            os.posix_fallocate(descriptor, 0, REPORT_CAPACITY)
            os.fsync(descriptor)
            _sync(path.parent)
            status = os.fstat(descriptor)
        except BaseException:
            os.unlink(temporary)
            _sync(path.parent)
            raise
        finally:
            os.close(descriptor)
    return {"temporary": str(temporary), "target": str(path), "device": status.st_dev,
            "inode": status.st_ino, "capacity": REPORT_CAPACITY}


def _admit(args, *, started=None):
    admission, execution_deadline, publication_deadline = _read_admission(args, started=started)
    slot = _reserve_slot(args, admission, deadline=execution_deadline)
    admission["execution_deadline_utc_epoch"] = execution_deadline
    return admission, publication_deadline, slot


def _reserve_slot_operation(args, admission, deadline):
    return _reserve_slot(args, admission, deadline=deadline)


def _publish(slot, value, *, reserve, deadline):
    report = {"status": "PENDING_TERMINAL_CONFIRMATION", "candidate_result": _snapshot(value)}
    data = json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"
    if len(data) > REPORT_CAPACITY:
        raise ProbeError("OUTPUT_UNAVAILABLE")
    try:
        temporary = Path(slot["temporary"])
        target = Path(slot["target"])
        if (not temporary.is_absolute() or not target.is_absolute() or slot.get("capacity") != REPORT_CAPACITY):
            raise ValueError()
        descriptor = os.open(temporary, os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC)
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise ProbeError("OUTPUT_UNAVAILABLE") from error
    try:
        with _allocation(target.parent, deadline=deadline):
            if time.time() >= deadline:
                raise ProbeError("DEADLINE_EXCEEDED")
            _canonical_output(str(target))
            status = os.fstat(descriptor)
            if (status.st_dev != slot["device"] or status.st_ino != slot["inode"] or
                    status.st_size != REPORT_CAPACITY):
                raise ProbeError("OUTPUT_UNAVAILABLE")
            block = max(4096, os.statvfs(target.parent).f_frsize)
            before_blocks = status.st_blocks * 512
            # Reclaim the unused reservation first and then admit link metadata
            # against actual free space, rather than an inferred logical size.
            os.ftruncate(descriptor, len(data))
            os.fsync(descriptor)
            after_blocks = os.fstat(descriptor).st_blocks * 512
            if after_blocks > before_blocks or shutil.disk_usage(target.parent).free < reserve + block:
                raise ProbeError("OUTPUT_UNAVAILABLE")
            os.lseek(descriptor, 0, os.SEEK_SET)
            offset = 0
            while offset < len(data):
                offset += os.write(descriptor, data[offset:])
            os.fsync(descriptor)
            if time.time() >= deadline:
                raise ProbeError("DEADLINE_EXCEEDED")
            os.link(temporary, target)
            _sync(target.parent)
            temporary.unlink()
            _sync(target.parent)
        return hashlib.sha256(data).hexdigest()
    finally:
        os.close(descriptor)


def _publish_operation(slot, value, reserve, deadline):
    return _publish(slot, value, reserve=reserve, deadline=deadline)


def _protocol(runtime):
    try:
        protocols = runtime["protocols"]
        vad = protocols["vad_config"]
        return {"hivau": protocols["hivau"],
                "vad": {key: vad[key] for key in ("target_fps", "query_interval", "batch_size")},
                "vad_route": "reactvau_detection", "vad_causal_smoothing": "online",
                "vad_fast_fusion": vad["fusion"]}
    except (KeyError, TypeError) as error:
        raise ProbeError("PREFLIGHT_INVALID") from error


def _typed_path_hash(path, digest, *, code):
    if (not isinstance(path, str) or "\0" in path or len(path.encode("utf-8")) > MAX_BINDING_PATH_BYTES or
            not Path(path).is_absolute() or ".." in Path(path).parts or
            not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)):
        raise ProbeError(code)
    return path, digest


def _runtime_bindings(document, *, runtime, runtime_sha256, source, source_sha256, resolved, bindings):
    """Extract only typed, blind-safe references for IPC and factory assembly."""
    try:
        _no_supervision(document)
        if not isinstance(document, dict):
            raise ValueError()
        inherited, fast = document["inherited"], document["fast"]
        if not isinstance(inherited, dict) or not isinstance(fast, dict):
            raise ValueError()
        source_path, source_digest = _typed_path_hash(inherited["source_manifest"], inherited["source_manifest_sha256"], code="SOURCE_MISMATCH")
        tokenizer, tokenizer_sha256 = _typed_path_hash(inherited["tokenizer"], inherited["tokenizer_sha256"], code="PREFLIGHT_INVALID")
        fast_path, fast_sha256 = _typed_path_hash(fast["snapshot"], fast["snapshot_sha256"], code="PREFLIGHT_INVALID")
        if source_path != str(source) or source_digest != source_sha256:
            raise ProbeError("SOURCE_MISMATCH")
        verified_tokenizer = _bound_artifact(tokenizer, tokenizer_sha256, name="probe tokenizer")
        verified_fast, fast_document = _bound(fast_path, fast_sha256, "PREFLIGHT_INVALID")
        _no_supervision(fast_document)
        result = {
            "runtime": str(runtime), "runtime_sha256": runtime_sha256,
            "fast_snapshot": str(verified_fast), "fast_snapshot_sha256": fast_sha256,
            "source_manifest": str(source), "source_manifest_sha256": source_sha256,
            "tokenizer": str(verified_tokenizer), "tokenizer_sha256": tokenizer_sha256,
            "embedded_vision_binding": str(resolved["embedded_vision"]), "embedded_vision_binding_sha256": bindings["embedded_vision"]["sha256"],
            "decoder": str(resolved["decoder"]), "decoder_sha256": bindings["decoder"]["sha256"],
            "implementation_manifest": str(resolved["implementation"]), "implementation_manifest_sha256": bindings["implementation"]["sha256"],
        }
        return result, fast_document
    except ProbeError:
        raise
    except Exception as error:
        raise ProbeError("PREFLIGHT_INVALID") from error


def _request(identities, identity, kind):
    try:
        _no_supervision(identities)
        if not isinstance(identities, dict) or set(identities) != {"vad", "vau"}:
            raise ValueError()
        keys = {"vad": {"dataset", "id", "media_path", "media_sha256"},
                "vau": {"id", "media_path", "media_sha256", "question"}}
        seen = set()
        selected = None
        for route in ("vad", "vau"):
            if not isinstance(identities[route], list):
                raise ValueError()
            for ordinal, row in enumerate(identities[route]):
                if not isinstance(row, dict) or set(row) != keys[route]:
                    raise ValueError()
                if (not isinstance(row["id"], str) or not row["id"] or
                        not isinstance(row["media_path"], str) or not Path(row["media_path"]).is_absolute() or
                        not isinstance(row["media_sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", row["media_sha256"])):
                    raise ValueError()
                if route == "vad":
                    if row["dataset"] not in {"ucf", "xd"}:
                        raise ValueError()
                    key = "vad:" + row["dataset"] + ":" + row["id"]
                else:
                    if not isinstance(row["question"], str) or not row["question"]:
                        raise ValueError()
                    key = "vau:" + row["id"]
                if key in seen:
                    raise ValueError()
                seen.add(key)
                if key == identity and route == kind:
                    selected = (row, ordinal)
        if selected is None:
            raise ValueError()
        row, ordinal = selected
        media = str(_bound_path(row["media_path"], row["media_sha256"], name="probe selected media"))
        if kind == "vad":
            return VadRequest(row["dataset"], row["id"], media, row["media_sha256"])
        return VauRequest(row["id"], ordinal, media, row["media_sha256"], row["question"])
    except Exception as error:
        raise ProbeError("IDENTITY_INVALID") from error


def _peaks(context):
    torch = context.get("_torch")
    if torch is not None and context.get("cuda_started"):
        device = context["device"]
        context["peak_cuda_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
        context["peak_cuda_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))


def _snapshot(context):
    # All values below are populated explicitly; model exception/payload text
    # is never copied into this channel, even when validation fails.
    def bounded(value, *, depth=0):
        if depth > 6:
            raise ValueError()
        if value is None or isinstance(value, bool):
            return value
        if type(value) is int:
            if not -MAX_REPORT_INTEGER <= value <= MAX_REPORT_INTEGER:
                raise ValueError()
            return value
        if type(value) is float:
            if not math.isfinite(value):
                raise ValueError()
            return value
        if isinstance(value, str):
            if "\0" in value or len(value.encode("utf-8")) > MAX_BINDING_PATH_BYTES:
                raise ValueError()
            return value
        if isinstance(value, list):
            if len(value) > 128:
                raise ValueError()
            return [bounded(item, depth=depth + 1) for item in value]
        if isinstance(value, dict):
            if len(value) > 64:
                raise ValueError()
            return {bounded(key, depth=depth + 1): bounded(item, depth=depth + 1)
                    for key, item in value.items() if isinstance(key, str) and len(key) <= 128}
        raise ValueError()

    result = {}
    for key in _REPORT_FIELDS:
        if key in context:
            try:
                result[key] = bounded(context[key])
            except (TypeError, UnicodeError, ValueError):
                continue
    raw = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(raw) > MAX_IPC_REPORT_BYTES:
        sanitized = result
        result = {"status": "FAILED_RUNTIME_PROBE", "stage": "report_sanitization", "error_code": "RUNTIME_FAILURE"}
        for key in ("started_at_utc_epoch", "deadline_utc_epoch", "publication_deadline_utc_epoch"):
            value = sanitized.get(key)
            if (type(value) in {int, float} and math.isfinite(value) and
                    abs(float(value)) <= MAX_REPORT_INTEGER):
                result[key] = value
        raw = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        if len(raw) > MAX_IPC_REPORT_BYTES:
            raise ProbeError("RUNTIME_FAILURE")
    return result


def _stage(context, stage):
    context["stage"] = stage
    context["elapsed_seconds"] = {"total": time.time() - context["started_at_utc_epoch"]}
    sender = context.get("_send")
    if sender:
        sender(_snapshot(context))


def _remaining(context):
    if time.time() >= context["deadline_utc_epoch"]:
        raise ProbeError("DEADLINE_EXCEEDED")


def run(args, context):
    _stage(context, "preflight")
    report, preflight = _bound(args.preflight_report, args.preflight_report_sha256, "PREFLIGHT_INVALID")
    try:
        _no_supervision(preflight)
    except Exception as error:
        raise ProbeError("PREFLIGHT_INVALID") from error
    if (not isinstance(preflight, dict) or preflight.get("schema") != PREFLIGHT_SCHEMA or
            preflight.get("status") != "PREFLIGHT_PASS_FORMAL_ADMISSION_REQUIRED" or
            preflight.get("gpu_launched") is not False):
        raise ProbeError("PREFLIGHT_INVALID")
    context["preflight_report"] = {"path": str(report), "sha256": args.preflight_report_sha256}
    try:
        rref, iref, mref, bindings = (preflight[key] for key in ("runtime", "identity_manifest", "r0_model_manifest", "bindings"))
        sref = bindings["prediction_source"]
        def record_bound(ref, name, code):
            if not isinstance(ref, dict) or set(ref) != {"path", "sha256"}:
                raise ProbeError(code)
            path, value = _bound(ref["path"], ref["sha256"], code)
            context[name] = {"path": str(path), "sha256": ref["sha256"]}
            _stage(context, "preflight")
            return path, value

        runtime, document = record_bound(rref, "runtime", "PREFLIGHT_INVALID")
        source, _ = record_bound(sref, "source_manifest", "SOURCE_MISMATCH")
        _, identities = record_bound(iref, "identity_manifest", "IDENTITY_INVALID")
        model, _ = record_bound(mref, "model_manifest", "MODEL_IDENTITY_INVALID")
        resolved = {}
        for key in ("embedded_vision", "decoder", "implementation"):
            resolved[key] = record_bound(bindings[key], key + "_binding", "PREFLIGHT_INVALID")[0]
    except (KeyError, TypeError) as error:
        raise ProbeError("PREFLIGHT_INVALID") from error
    verify_implementation_manifest(bindings["implementation"]["path"], bindings["implementation"]["sha256"])
    _stage(context, "request_validation")
    request = _request(identities, args.identity, args.kind)
    context["identity"] = request.identity
    try:
        protocol = _protocol(document)
    except ProbeError:
        raise
    except Exception as error:
        raise ProbeError("PREFLIGHT_INVALID") from error
    try:
        artifact = load_model_artifact(model, expected_sha256=mref["sha256"], task=ModelTask("R0", None))
    except Exception as error:
        raise ProbeError("MODEL_IDENTITY_INVALID") from error
    bound, fast = _runtime_bindings(document, runtime=runtime, runtime_sha256=rref["sha256"], source=source,
                                    source_sha256=sref["sha256"], resolved=resolved, bindings=bindings)
    context["bindings"] = bound
    expected_media = None
    threshold = None
    if args.kind == "vad":
        try:
            if (not isinstance(fast, dict) or fast.get("schema") != "nc_rted_blind_fast_snapshot/v1" or
                    set(fast) != {"schema", "media"} or not isinstance(fast["media"], list)):
                raise ValueError()
            rows = [row for row in fast["media"] if isinstance(row, dict) and row.get("dataset") == request.dataset and row.get("media_key") == request.media_id]
            if len(rows) != 1 or rows[0].get("media_path") != request.media_path or rows[0].get("media_sha256") != request.media_sha256:
                raise ProbeError("IDENTITY_INVALID")
            expected_media = rows[0]
            if any(expected_media[key] != protocol["vad"][key] for key in ("target_fps", "query_interval")):
                raise ValueError()
            threshold = document["protocols"]["vad"][request.dataset]["trigger_threshold"]
        except ProbeError:
            raise
        except Exception as error:
            raise ProbeError("PREFLIGHT_INVALID") from error
    _stage(context, "cuda_initialization")
    _remaining(context)
    import torch
    if not torch.cuda.is_available():
        raise ProbeError("CUDA_UNAVAILABLE")
    _remaining(context)
    context["_torch"] = torch
    context["cuda_started"] = True
    torch.cuda.set_device(args.device)
    torch.cuda.reset_peak_memory_stats(args.device)
    torch.cuda.synchronize(args.device)
    _stage(context, "factory")
    factory = default_factory(SimpleNamespace(bindings=bound, binding_sha256={key[:-7]: value for key, value in bound.items() if key.endswith("_sha256")}, protocol=protocol), artifact, device=args.device)
    torch.cuda.synchronize(args.device)
    _peaks(context)
    _stage(context, "model_loading")
    loaded = factory["loader"].load(artifact)
    try:
        validate_loaded_model_identity(loaded, artifact)
    except Exception as error:
        raise ProbeError("MODEL_IDENTITY_INVALID") from error
    torch.cuda.synchronize(args.device)
    _peaks(context)
    _remaining(context)
    context["counters"] = {"completed_slow_forwards": 0, "completed_generation_calls": 0}
    _stage(context, args.kind + "_inference")
    def forward_progress():
        _peaks(context)
        _stage(context, context["stage"])

    try:
        with count_slow_execution(loaded, context["counters"], on_progress=forward_progress), count_generation_execution(loaded, context["counters"]):
            if args.kind == "vad":
                payload = factory["vad"].predict(request, loaded, protocol=protocol)
            else:
                payload = factory["vau"].generate(request, loaded, protocol=protocol)
    except (PredictionInputError, PredictionExecutionError, ValueError, TypeError, KeyError) as error:
        raise ProbeError("PAYLOAD_INVALID") from error
    torch.cuda.synchronize(args.device)
    _peaks(context)
    _stage(context, "payload_validation")
    try:
        if args.kind == "vad":
            payload = validate_vad_payload(payload, expected_media=expected_media, trigger_threshold=threshold)
            summary = {"kind": "vad", "query_count": len(payload["queries"]), "total_frames": payload["total_frames"],
                       "validated_trigger_count": sum(query["triggered"] for query in payload["queries"])}
            if context["counters"]["completed_slow_forwards"] != summary["validated_trigger_count"]:
                raise ValueError()
        else:
            payload = validate_vau_payload(payload, max_new_tokens=protocol["hivau"]["max_new_tokens"])
            summary = {"kind": "vau", "token_count": len(payload["token_ids"])}
    except Exception as error:
        raise ProbeError("PAYLOAD_INVALID") from error
    summary.update(context["counters"])
    summary["slow_forward_qualified"] = context["counters"]["completed_slow_forwards"] > 0
    context["summary"] = summary
    _remaining(context)
    if context.get("peak_cuda_allocated_bytes") is None or context.get("peak_cuda_reserved_bytes") is None:
        raise ProbeError("MEASUREMENT_FAILED")
    context["status"] = "PASS_RUNTIME_PROBE"
    _stage(context, "completed")
    return _snapshot(context)


def _worker(args, context, connection):
    os.setsid()
    # Redirect native/model output too; only the explicit sanitized IPC channel
    # is forwarded. A failure must not leak generated text through stderr.
    sink = os.open(os.devnull, os.O_WRONLY)
    os.dup2(sink, 1)
    os.dup2(sink, 2)
    os.close(sink)
    context["_send"] = connection.send
    try:
        _stage(context, "bootstrap")
        _remaining(context)
        _bootstrap(args.preflight_report, args.preflight_report_sha256)
        _stage(context, "imports")
        _api()
        run(args, context)
    except ProbeError as error:
        context.update(status="INCOMPLETE_RUNTIME_PROBE", error_code=error.code)
    except Exception:
        context.update(status="FAILED_RUNTIME_PROBE", error_code="RUNTIME_FAILURE")
    finally:
        try:
            _peaks(context)
        except Exception:
            context["peak_measurement_incomplete"] = True
            if context.get("status") == "PASS_RUNTIME_PROBE":
                context.update(status="FAILED_RUNTIME_PROBE", error_code="MEASUREMENT_FAILED")
        context["elapsed_seconds"] = {"total": time.time() - context["started_at_utc_epoch"]}
        connection.send(_snapshot(context))
        connection.close()


class _JsonSender:
    def __init__(self, descriptor):
        self.descriptor = descriptor

    def send(self, value):
        try:
            data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        except (TypeError, UnicodeError, ValueError):
            data = b'{"error_code":"RUNTIME_FAILURE","status":"FAILED_RUNTIME_PROBE"}'
        if len(data) > MAX_IPC_REPORT_BYTES:
            data = json.dumps({"status": "FAILED_RUNTIME_PROBE", "error_code": "RUNTIME_FAILURE"}, separators=(",", ":")).encode("utf-8")
        os.write(self.descriptor, data + b"\n")

    def close(self):
        pass


def _fork_target(target, arguments):
    receive, send = os.pipe()
    child = os.fork()
    if child == 0:
        os.close(receive)
        sender = _JsonSender(send)
        try:
            target(*arguments, sender)
        finally:
            os.close(send)
            os._exit(0)
    os.close(send)
    os.set_blocking(receive, False)
    return child, receive


def _terminate(child):
    try:
        if os.getpgid(child) == child:
            os.killpg(child, signal.SIGKILL)
        else:
            os.kill(child, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _collect(child, receive, *, deadline):
    messages, buffer, timed_out, exit_status = [], b"", False, None
    try:
        while True:
            if exit_status is None:
                pid, status = os.waitpid(child, os.WNOHANG)
                if pid:
                    exit_status = status
            now = time.time()
            if exit_status is None and now >= deadline:
                timed_out = True
                _terminate(child)
                # Never block in waitpid: an uninterruptible filesystem child
                # must not hold the coordinator or Python exit handler hostage.
                reap_until = time.monotonic() + .1
                while time.monotonic() < reap_until:
                    pid, status = os.waitpid(child, os.WNOHANG)
                    if pid:
                        exit_status = status
                        break
                    time.sleep(.005)
                break
            wait = 0 if exit_status is not None else max(0, min(.05, deadline - now))
            readable, _, _ = select.select([receive], [], [], wait)
            if exit_status is not None and not readable:
                break
            if readable:
                try:
                    chunk = os.read(receive, 65536)
                except BlockingIOError:
                    chunk = b""
                if chunk:
                    buffer += chunk
                    while b"\n" in buffer:
                        line, buffer = buffer.split(b"\n", 1)
                        try:
                            messages.append(json.loads(line))
                        except (UnicodeError, ValueError):
                            pass
                elif exit_status is not None:
                    break
    finally:
        os.close(receive)
    return messages, timed_out, exit_status


def _supervise(args, context, *, target=_worker):
    child, receive = _fork_target(target, (args, dict(context)))
    messages, timed_out, exit_status = _collect(child, receive, deadline=context["deadline_utc_epoch"])
    last = messages[-1] if messages else dict(context)
    if timed_out:
        last.update(status="INCOMPLETE_RUNTIME_PROBE", error_code="DEADLINE_EXCEEDED", peak_measurement_incomplete=True)
    elif exit_status is None or not os.WIFEXITED(exit_status) or os.WEXITSTATUS(exit_status) != 0 or last.get("status") not in {"PASS_RUNTIME_PROBE", "FAILED_RUNTIME_PROBE", "INCOMPLETE_RUNTIME_PROBE"}:
        last.update(status="FAILED_RUNTIME_PROBE", error_code="RUNTIME_FAILURE", peak_measurement_incomplete=True)
    last["elapsed_seconds"] = {"total": time.time() - context["started_at_utc_epoch"]}
    return _snapshot(last)


def _operation_worker(target, arguments, connection):
    """Run potentially blocking parent filesystem work in a killable child."""
    os.setsid()
    try:
        connection.send({"ok": True, "value": target(*arguments)})
    except ProbeError as error:
        connection.send({"ok": False, "code": error.code})
    except Exception:
        connection.send({"ok": False, "code": "RUNTIME_FAILURE"})
    finally:
        connection.close()


def _supervise_operation(target, arguments, *, deadline):
    if time.time() >= deadline:
        raise ProbeError("DEADLINE_EXCEEDED")
    child, receive = _fork_target(_operation_worker, (target, arguments))
    messages, timed_out, exit_status = _collect(child, receive, deadline=deadline)
    message = messages[-1] if messages else None
    if timed_out or exit_status is None or time.time() >= deadline:
        raise ProbeError("DEADLINE_EXCEEDED")
    if not isinstance(message, dict) or message.get("ok") is not True:
        raise ProbeError(message.get("code") if isinstance(message, dict) else "RUNTIME_FAILURE")
    return message["value"]


def main():
    started = time.time()
    parser = argparse.ArgumentParser()
    for field in ("preflight-report", "preflight-report-sha256", "diagnostic-admission",
                  "diagnostic-admission-sha256", "identity", "device", "output"):
        parser.add_argument("--" + field, required=True)
    parser.add_argument("--kind", required=True, choices=("vad", "vau"))
    args = parser.parse_args()
    result = None
    try:
        admission, execution_deadline, publication_deadline = _supervise_operation(
            _read_admission_operation, (args, started), deadline=started + ADMISSION_PARSE_SECONDS)
        slot = _supervise_operation(_reserve_slot_operation, (args, admission, execution_deadline), deadline=execution_deadline)
        context = {"status": "RUNNING", "stage": "admitted", "started_at_utc_epoch": started,
                   "deadline_utc_epoch": execution_deadline, "publication_deadline_utc_epoch": publication_deadline,
                   "device": args.device,
                   "diagnostic_admission": {"path": args.diagnostic_admission, "sha256": args.diagnostic_admission_sha256},
                   "peak_cuda_allocated_bytes": None,
                   "peak_cuda_reserved_bytes": None, "formal_prediction": False, "prediction_store_written": False}
        try:
            context["probe_script_sha256"] = _supervise_operation(_hash_script_operation, (__file__,), deadline=execution_deadline)
        except ProbeError as error:
            result = {**context, "status": "INCOMPLETE_RUNTIME_PROBE", "error_code": error.code}
        if result is None and time.time() >= execution_deadline:
            result = {**context, "status": "INCOMPLETE_RUNTIME_PROBE", "error_code": "DEADLINE_EXCEEDED"}
        elif result is None:
            try:
                result = _supervise(args, context)
            except Exception:
                result = {**context, "status": "FAILED_RUNTIME_PROBE", "error_code": "RUNTIME_FAILURE"}
        digest = _supervise_operation(_publish_operation,
                                      (slot, _snapshot(result), admission["min_free_bytes"], publication_deadline),
                                      deadline=publication_deadline)
    except Exception as error:
        code = error.code if isinstance(error, ProbeError) else "RUNTIME_FAILURE"
        print(json.dumps({"status": "BLOCKED", "error_code": code}), file=sys.stderr)
        return 2
    print(json.dumps({"status": "TERMINAL_RUNTIME_PROBE", "candidate_status": result["status"],
                      "candidate_error_code": result.get("error_code"), "report_sha256": digest}, sort_keys=True))
    return 0 if result["status"] == "PASS_RUNTIME_PROBE" else 3


if __name__ == "__main__":
    raise SystemExit(main())
