#!/usr/bin/env python3
"""Blind prediction entry point; model-specific adapters are supplied explicitly."""
from __future__ import annotations

import argparse, fcntl, hashlib, importlib, importlib.abc, importlib.machinery, json, math, os, select, shutil, signal, stat, subprocess, time, uuid
from contextlib import contextmanager
from pathlib import Path
import sys

_EXECUTED_DRIVER_CODE = sys._getframe().f_code
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class BootstrapError(RuntimeError):
    pass


class WorkerCleanupIncomplete(RuntimeError):
    """An unreaped worker must retain its full resource-budget charge."""
    retain_full_budget = True


PredictionInputError = BootstrapError
_APPROVED_DATA_VOLUME = Path("/root/autodl-tmp/lookaway-wm").resolve()


# This mirrors prediction_inputs._IMPLEMENTATION_FILES without importing that
# module. The driver must establish source integrity before it imports it.
_IMPLEMENTATION_FILES = frozenset({
    "scripts/nc_rted_predict.py", "src/nc_rted/__init__.py", "src/nc_rted/batches.py", "src/nc_rted/bridge.py",
    "scripts/nc_rted_prediction_runtime_probe.py",
    "src/nc_rted/detection_media.py", "src/nc_rted/detection_provider.py", "src/nc_rted/detector.py",
    "src/nc_rted/frozen_vision.py", "src/nc_rted/inherited_memory.py", "src/nc_rted/loading.py",
    "src/nc_rted/media_observer.py", "src/nc_rted/numerics.py", "src/nc_rted/observation.py",
    "src/nc_rted/observation_cache.py", "src/nc_rted/prediction_adapters.py", "src/nc_rted/prediction_inputs.py",
    "src/nc_rted/prediction_media.py", "src/nc_rted/prediction_runtime.py", "src/nc_rted/prediction_store.py",
    "src/nc_rted/prediction_worker.py", "src/nc_rted/production_runtime.py", "src/nc_rted/recovery.py",
    "src/nc_rted/task_inputs.py", "src/nc_rted/storage_lock.py", "src/nc_rted/model.py",
})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _install_verified_implementation_imports(manifest_path: str, manifest_sha256: str | None) -> None:
    """Bind the driver and every nc_rted import before Python can read pyc."""
    path = Path(manifest_path)
    raw = path.read_bytes()
    if manifest_sha256 is not None and hashlib.sha256(raw).hexdigest() != manifest_sha256:
        raise BootstrapError("prediction manifest SHA-256 differs")
    try:
        plan = json.loads(raw)
        binding = plan["bindings"]
        implementation_path = Path(binding["implementation_manifest"])
        implementation_sha256 = binding["implementation_manifest_sha256"]
        document_raw = implementation_path.read_bytes()
        document = json.loads(document_raw)
    except (KeyError, OSError, ValueError, TypeError) as error:
        raise BootstrapError("prediction implementation bootstrap is invalid") from error
    if hashlib.sha256(document_raw).hexdigest() != implementation_sha256:
        raise BootstrapError("prediction implementation manifest SHA-256 differs")
    files = document.get("files") if isinstance(document, dict) else None
    if (not isinstance(document, dict) or document.get("schema") != "nc_rted_blind_prediction_implementation/v1" or
            Path(document.get("root", "")).resolve() != ROOT or not isinstance(files, dict) or set(files) != _IMPLEMENTATION_FILES):
        raise BootstrapError("prediction implementation manifest file set differs")
    captured = {}
    for relative, digest in files.items():
        candidate = ROOT / relative
        if (not isinstance(digest, str) or len(digest) != 64 or candidate.is_symlink() or not candidate.is_file() or
                _sha256(candidate) != digest):
            raise BootstrapError("prediction implementation source differs from admitted manifest")
        captured[candidate.resolve()] = candidate.read_bytes()
    driver = ROOT / "scripts/nc_rted_predict.py"
    if (hashlib.sha256(captured[driver.resolve()]).hexdigest() != files["scripts/nc_rted_predict.py"] or
            compile(captured[driver.resolve()], _EXECUTED_DRIVER_CODE.co_filename, "exec", dont_inherit=True,
                    optimize=sys.flags.optimize) != _EXECUTED_DRIVER_CODE):
        raise BootstrapError("executing prediction driver differs from verified source bytes")
    modules, packages = set(), set()
    for relative in files:
        if not relative.startswith("src/nc_rted/"):
            continue
        parts = list(Path(relative).relative_to("src").with_suffix("").parts)
        if parts[-1] == "__init__": parts = parts[:-1]
        name = ".".join(parts)
        if name: modules.add(name)
        for depth in range(1, len(parts)):
            packages.add(".".join(parts[:depth]))
    def owns(name: str) -> bool:
        return name in modules or name in packages or name.startswith("nc_rted.")
    if any(owns(name) for name in sys.modules):
        raise BootstrapError("prediction implementation was imported before verified source admission")
    class Loader(importlib.machinery.SourceFileLoader):
        def get_code(self, fullname):
            return compile(captured[Path(self.path).resolve()], self.path, "exec", dont_inherit=True)
    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if not owns(fullname): return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec is None or not spec.origin or spec.origin in {"built-in", "frozen"}:
                raise BootstrapError("unbound prediction implementation import: " + fullname)
            origin = Path(spec.origin).resolve()
            if origin not in captured or not isinstance(spec.loader, importlib.machinery.SourceFileLoader):
                raise BootstrapError("unbound prediction implementation import: " + fullname)
            spec.loader = Loader(fullname, str(origin))
            return spec
    sys.meta_path.insert(0, Finder())


def _load_prediction_api() -> None:
    global PredictionInputError, load_model_artifact, load_prediction_plan, prediction_task_execution_binding_sha256, verify_implementation_manifest, validate_resource_execution_evidence, PredictionStore, PredictionWorker
    from nc_rted.prediction_inputs import PredictionInputError, load_model_artifact, load_prediction_plan, prediction_task_execution_binding_sha256, verify_implementation_manifest, validate_resource_execution_evidence
    from nc_rted.prediction_store import PredictionStore
    from nc_rted.prediction_worker import PredictionWorker


def _factory(spec: str | None, plan, model, *, device: str):
    if spec is None:
        from nc_rted.prediction_runtime import default_factory
        return _unpack_factory(default_factory(plan, model, device=device))
    if ":" not in spec:
        raise ValueError("adapter factory must be module:callable")
    module, name = spec.split(":", 1)
    result = getattr(importlib.import_module(module), name)(plan, model)
    return _unpack_factory(result)


def _unpack_factory(result):
    if not isinstance(result, dict) or set(result) != {"loader", "vad", "vau"}:
        raise ValueError("adapter factory must return loader/vad/vau")
    return result["loader"], result["vad"], result["vau"]


def _physical_gpu_uuid(device: str, *, visible: str | None = None, inventory: str | None = None, timeout: float = 10.0) -> str:
    """Resolve a CUDA logical ordinal through CUDA_VISIBLE_DEVICES."""
    try:
        logical = int(device.split(":", 1)[1])
        if inventory is None:
            inventory = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], check=True,
                                       capture_output=True, text=True, timeout=timeout).stdout
        rows = {index.strip(): uuid.strip() for line in inventory.splitlines() if "," in line for index, uuid in [line.split(",", 1)]}
        selection = os.environ.get("CUDA_VISIBLE_DEVICES") if visible is None else visible
        tokens = list(rows) if not selection else [item.strip() for item in selection.split(",")]
        token = tokens[logical]
        uuid = rows.get(token) or next((value for value in rows.values() if value == token or value.startswith(token)), "")
    except (IndexError, ValueError, OSError, subprocess.SubprocessError) as error:
        raise ValueError("admitted CUDA device UUID is unavailable") from error
    if not uuid or "\n" in uuid:
        raise ValueError("admitted CUDA device UUID is invalid")
    return uuid


def _watchdog(deadline: float | None, operation):
    """Run model work in a killable child so native calls cannot evade expiry."""
    if deadline is None:
        return operation()
    if time.time() >= deadline:
        raise ValueError("resource admission deadline has expired")
    reader, writer = os.pipe()
    pid = os.fork()
    if pid == 0:
        try:
            os.close(reader); os.setsid()
            payload = {"ok": True, "value": operation()}
        except BaseException as error:
            payload = {"ok": False, "error": type(error).__name__}
        with os.fdopen(writer, "wb") as stream:
            stream.write(json.dumps(payload).encode())
            stream.flush()
        os._exit(0)
    data = b""; child_reaped = False; eof = False; writer_open = True
    def cleanup():
        nonlocal child_reaped
        try: os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            # setsid may not have run yet. The unreaped direct child still owns
            # this PID, so the fallback cannot target a reused process.
            if not child_reaped:
                try: os.kill(pid, signal.SIGKILL)
                except ProcessLookupError: pass
        if not child_reaped:
            cutoff = time.monotonic() + 1.0
            while True:
                try:
                    waited, _ = os.waitpid(pid, os.WNOHANG)
                    if waited:
                        child_reaped = True
                        return
                except InterruptedError: continue
                except ChildProcessError:
                    child_reaped = True; return
                if time.monotonic() >= cutoff:
                    raise WorkerCleanupIncomplete("prediction worker termination remains unconfirmed")
                time.sleep(.01)
    def drain():
        nonlocal data, eof
        while True:
            try: chunk = os.read(reader, 65536)
            except BlockingIOError: return
            if not chunk:
                eof = True; return
            data += chunk
    try:
        os.close(writer)
        writer_open = False
        os.set_blocking(reader, False)
        while True:
            drain()
            if not child_reaped:
                done, _ = os.waitpid(pid, os.WNOHANG)
                if done:
                    child_reaped = True
                    # A descendant can retain the writer after the direct child
                    # exits. Kill the worker session before accepting payload.
                    cleanup()
            if child_reaped and eof: break
            if time.time() >= deadline:
                cleanup()
                raise ValueError("admitted prediction runtime budget expired")
            remaining = deadline - time.time()
            if remaining <= 0:
                cleanup()
                raise ValueError("admitted prediction runtime budget expired")
            select.select([reader], [], [], min(.1, remaining))
        payload = json.loads(data.decode())
        if not payload.get("ok"): raise ValueError("prediction child failed")
        return payload["value"]
    except BaseException:
        cleanup()
        raise
    finally:
        os.close(reader)
        if writer_open:
            try: os.close(writer)
            except OSError: pass


def _gpu_memory_bytes(uuid: str, *, timeout: float = 10.0) -> int:
    result = subprocess.run(["nvidia-smi", "--query-gpu=uuid,memory.total", "--format=csv,noheader"], check=True, capture_output=True, text=True, timeout=timeout)
    for line in result.stdout.splitlines():
        candidate, memory = [item.strip() for item in line.split(",", 1)]
        if candidate == uuid:
            return int(memory.split()[0]) * 1024**2
    raise ValueError("admitted CUDA capacity is unavailable")


def _mount_identity(path: Path) -> tuple[int, str]:
    target = path.resolve()
    while not target.exists(): target = target.parent
    best = None
    try:
        for line in Path("/proc/self/mountinfo").read_text().splitlines():
            fields = line.split()
            if len(fields) > 5:
                point = Path(fields[4].replace("\\040", " "))
                try:
                    target.relative_to(point)
                    if best is None or len(point.parts) > len(best[1].parts): best = (int(fields[0]), point)
                except ValueError: pass
    except (OSError, ValueError):
        raise ValueError("Linux mount identity is unavailable")
    if best is None: raise ValueError("filesystem mount identity is unavailable")
    return target.stat().st_dev, str(best[1])


def _same_admitted_filesystem(path: Path, volume: Path) -> None:
    if _mount_identity(path) != _mount_identity(volume):
        raise ValueError("prediction output uses a different filesystem or mount")


def _admit_execution_resource(plan, device: str, *, evidence_registry: tuple[str, str] | None = None) -> tuple[float | None, str]:
    """Reconcile the v2 resource admission before factory/CUDA initialization."""
    scope = plan.resource_scope
    if scope is None:
        return None, device
    if device != scope["device"]:
        raise ValueError("requested device differs from admitted resource device")
    volume = Path(scope["data_volume"]).resolve()
    if volume != _APPROVED_DATA_VOLUME or not volume.is_dir():
        raise ValueError("resource admission does not use the approved data volume")
    output_root = Path(plan.output_root)
    if output_root.is_symlink():
        raise ValueError("prediction output root cannot be a symlink")
    try:
        output_root.resolve().relative_to(volume)
    except ValueError as error:
        raise ValueError("prediction output is outside the admitted data volume") from error
    if shutil.disk_usage(volume).free < scope["min_free_bytes"]:
        raise ValueError("admitted data-volume reserve is unavailable")
    deadline = float(scope["deadline_utc_epoch"])
    if not math.isfinite(deadline) or deadline <= time.time():
        raise ValueError("resource admission deadline has expired")
    deadline = min(deadline, time.time() + scope["run_budget_seconds"])
    remaining = deadline - time.time()
    if remaining <= 0:
        raise ValueError("resource admission deadline has expired")
    if _physical_gpu_uuid(device, timeout=min(10.0, remaining)) != scope["physical_gpu_uuid"]:
        raise ValueError("requested device physical GPU differs from admission")
    if time.time() >= deadline:
        raise ValueError("resource admission deadline has expired")
    # CUDA accepts a UUID in CUDA_VISIBLE_DEVICES.  Resetting visibility before
    # any model import makes cuda:0 the admitted physical device regardless of
    # an inherited ordinal mask or ordering.
    os.environ["CUDA_VISIBLE_DEVICES"] = scope["physical_gpu_uuid"]
    if _physical_gpu_uuid("cuda:0", visible=scope["physical_gpu_uuid"]) != scope["physical_gpu_uuid"] or time.time() >= deadline:
        raise ValueError("admitted CUDA UUID route is unavailable")
    if evidence_registry is None:
        raise ValueError("formal execution requires an operator accepted evidence registry")
    validate_resource_execution_evidence(scope, registry_path=evidence_registry[0], registry_sha256=evidence_registry[1],
                                         host=os.uname().nodename, physical_gpu_uuid=scope["physical_gpu_uuid"],
                                         capacity_bytes=_gpu_memory_bytes(scope["physical_gpu_uuid"], timeout=min(10.0, max(.001, deadline - time.time()))))
    return deadline, "cuda:0"


def _task_output_root(plan, task_id: str, scope: dict | None) -> Path:
    raw_root = Path(plan.output_root)
    if raw_root.is_symlink(): raise ValueError("prediction output root cannot be a symlink")
    root = raw_root.resolve()
    task = root / task_id
    if scope is None: return task
    volume = Path(scope["data_volume"]).resolve()
    _same_admitted_filesystem(root, volume)
    if task.exists() and task.is_symlink(): raise ValueError("prediction task output cannot be a symlink")
    try: task.resolve().relative_to(volume)
    except ValueError as error: raise ValueError("prediction task output escapes admitted volume") from error
    _same_admitted_filesystem(task, volume)
    if shutil.disk_usage(volume).free <= scope["min_free_bytes"]: raise ValueError("admitted reserve leaves no prediction publication capacity")
    return task


@contextmanager
def _cumulative_budget(plan, deadline: float | None):
    if deadline is None:
        yield None; return
    scope = plan.resource_scope; root = Path(plan.output_root)
    if root.is_symlink(): raise ValueError("prediction output root cannot be a symlink")
    volume = Path(scope["data_volume"]).resolve()
    _same_admitted_filesystem(root, volume)
    from nc_rted.storage_lock import allocation_lock, ensure_directory
    with allocation_lock(volume):
        ensure_directory(root, scope["min_free_bytes"])
        if shutil.disk_usage(volume).free < scope["min_free_bytes"]:
            raise ValueError("admitted data-volume reserve is unavailable")
    lock = root / ".resource-budget.lock"; ledger = root / ".resource-budget.json"
    with _control_file(lock, volume, scope["min_free_bytes"]) as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            value = _read_budget_ledger(ledger, volume) if os.path.lexists(ledger) else {"schema": "nc_rted_prediction_budget/v1", "scope": plan.execution_scope_sha256, "run_id": plan.run_id, "consumed_seconds": 0.0, "active": None}
            if (value.get("schema") != "nc_rted_prediction_budget/v1" or value.get("scope") != plan.execution_scope_sha256 or value.get("run_id") != plan.run_id or
                    type(value.get("consumed_seconds")) not in {int, float} or not math.isfinite(value["consumed_seconds"]) or value["consumed_seconds"] < 0):
                raise ValueError("resource budget ledger differs from admitted scope")
            active = value.get("active")
            if active is not None:
                if (not isinstance(active, dict) or set(active) != {"reserved_seconds", "started_utc_epoch"} or
                        type(active["reserved_seconds"]) not in {int, float} or not math.isfinite(active["reserved_seconds"]) or active["reserved_seconds"] <= 0 or
                        type(active["started_utc_epoch"]) not in {int, float} or not math.isfinite(active["started_utc_epoch"]) or active["started_utc_epoch"] <= 0):
                    raise ValueError("resource budget ledger active reservation is invalid")
                # A process that died after reserving capacity cannot replenish
                # the run budget on its retry.
                value["consumed_seconds"] += active["reserved_seconds"]
                value["active"] = None
                _write_budget_ledger(ledger, value, volume, scope["min_free_bytes"])
            remaining = scope["run_budget_seconds"] - float(value["consumed_seconds"])
            bounded = min(deadline, time.time() + remaining)
            if remaining <= 0 or bounded <= time.time(): raise ValueError("admitted cumulative prediction budget is exhausted")
            grant = bounded - time.time()
            value["active"] = {"reserved_seconds": grant, "started_utc_epoch": time.time()}
            _write_budget_ledger(ledger, value, volume, scope["min_free_bytes"])
            started = time.monotonic()
            try: yield bounded
            finally:
                value["consumed_seconds"] += (grant if getattr(sys.exc_info()[1], "retain_full_budget", False)
                                               else min(grant, time.monotonic() - started))
                value["active"] = None
                _write_budget_ledger(ledger, value, volume, scope["min_free_bytes"])
        finally: fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _write_budget_ledger(path: Path, value: dict, volume: Path, reserve: int) -> None:
    payload = json.dumps(value, sort_keys=True).encode()
    from nc_rted.storage_lock import allocation_lock
    with allocation_lock(volume):
        if shutil.disk_usage(volume).free < reserve + len(payload) + 8192:
            raise ValueError("prediction budget ledger would violate the admitted storage reserve")
        _control_path(path, volume)
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.pending")
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_DIRECTORY)
        try: os.fsync(descriptor)
        finally: os.close(descriptor)


def _control_path(path: Path, volume: Path) -> None:
    _same_admitted_filesystem(path.parent, volume)
    try: info = path.lstat()
    except FileNotFoundError: return
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_dev != volume.stat().st_dev:
        raise ValueError("prediction control path is not a regular admitted-volume file")


def _read_budget_ledger(path: Path, volume: Path) -> dict:
    _control_path(path, volume)
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or
                info.st_dev != volume.stat().st_dev or info.st_size > 65536):
            raise ValueError("prediction control path is not a bounded regular admitted-volume file")
        raw = stream.read(65537)
        if len(raw) > 65536:
            raise ValueError("prediction budget ledger is too large")
    return json.loads(raw)


@contextmanager
def _control_file(path: Path, volume: Path, reserve: int):
    from nc_rted.storage_lock import allocation_lock
    with allocation_lock(volume):
        _control_path(path, volume)
        if shutil.disk_usage(volume).free < reserve + 8192:
            raise ValueError("prediction control file would violate the admitted storage reserve")
        flags = os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
        try: descriptor = os.open(path, flags | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError: descriptor = os.open(path, flags)
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_dev != volume.stat().st_dev:
            os.close(descriptor)
            raise ValueError("prediction control path is not a regular admitted-volume file")
    try:
        with os.fdopen(descriptor, "a+") as handle: yield handle
    finally: pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "preflight", "run"))
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--adapter-factory", help="optional bound inference adapter factory module:callable")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--group", choices=("R0", "A", "U", "S", "F"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--model-manifest", help="selected task's complete artifact manifest")
    parser.add_argument("--model-manifest-sha256")
    parser.add_argument("--accepted-evidence-registry")
    parser.add_argument("--accepted-evidence-registry-sha256")
    args = parser.parse_args()
    try:
        if args.adapter_factory is None:
            _install_verified_implementation_imports(args.manifest, args.manifest_sha256)
        _load_prediction_api()
        plan = load_prediction_plan(args.manifest, expected_sha256=args.manifest_sha256)
        if args.adapter_factory is not None and not plan.run_id.startswith("test:"):
            raise ValueError("formal blind prediction requires the admitted default factory")
        result = {"run_id": plan.run_id, "manifest_sha256": plan.manifest_sha256,
                  "vad": len(plan.vad), "vau": len(plan.vau), "model_tasks": [item.task_id for item in plan.models],
                  "status": "PLAN_VALID"}
        if args.action == "plan":
            print(json.dumps(result, sort_keys=True)); return
        if args.adapter_factory is None:
            implementation_root = verify_implementation_manifest(plan.bindings["implementation_manifest"],
                                                                  plan.binding_sha256["implementation_manifest"])
            if Path(__file__).resolve() != implementation_root / "scripts" / "nc_rted_predict.py":
                raise PredictionInputError("prediction entrypoint differs from admitted implementation checkout")
        if args.group is None or (args.group == "R0" and args.seed is not None) or (args.group != "R0" and args.seed is None):
            raise ValueError("preflight/run require R0 without seed or A/U/S/F with one formal seed")
        model = plan.selected_model(args.group, args.seed)
        if args.model_manifest is None or args.model_manifest_sha256 is None:
            raise ValueError("preflight/run require the selected task's hash-bound model manifest")
        artifact = load_model_artifact(args.model_manifest, expected_sha256=args.model_manifest_sha256, task=model)
        result["model_task"] = artifact.task_id
        if args.action == "preflight":
            # Default preflight checks the selected immutable bindings without
            # importing inherited modules, which may initialize CUDA.
            if args.adapter_factory is None:
                from nc_rted.prediction_runtime import _reconcile_plan, load_prediction_runtime, validate_artifact_runtime
                runtime = load_prediction_runtime(plan.bindings["runtime"], expected_sha256=plan.binding_sha256["runtime"])
                _reconcile_plan(plan, runtime)
                validate_artifact_runtime(artifact, runtime)
            else:
                _factory(args.adapter_factory, plan, artifact, device=args.device)
            result["status"] = "PREFLIGHT_PASS"
            print(json.dumps(result, sort_keys=True)); return
        if (args.accepted_evidence_registry is None) != (args.accepted_evidence_registry_sha256 is None):
            raise ValueError("accepted evidence registry requires both path and SHA-256")
        deadline, execution_device = _admit_execution_resource(plan, args.device, evidence_registry=(args.accepted_evidence_registry, args.accepted_evidence_registry_sha256) if args.accepted_evidence_registry else None)
        def operation():
            loader, vad, vau = _factory(args.adapter_factory, plan, artifact, device=execution_device)
            store = PredictionStore(_task_output_root(plan, artifact.task_id, plan.resource_scope), run_id=plan.run_id, manifest_sha256=plan.manifest_sha256,
                                    model_task=artifact.task_id, model_binding_sha256=artifact.manifest_sha256, matrix_id=plan.matrix_id,
                                    task_execution_binding_sha256=(prediction_task_execution_binding_sha256(matrix_id=plan.matrix_id, task=artifact)
                                                                     if plan.matrix_id is not None else None),
                                    storage_volume=(plan.resource_scope or {}).get("data_volume"),
                                    min_free_bytes=(plan.resource_scope or {}).get("min_free_bytes", 0))
            return PredictionWorker(plan, store, loader=loader, vad=vad, vau=vau, model=artifact).run()
        with _cumulative_budget(plan, deadline) as bounded_deadline:
            result.update(_watchdog(bounded_deadline, operation))
        result["status"] = "COMPLETE" if result["succeeded"] == result["expected"] and result["technical_failures"] == 0 and result["missing"] == 0 else "INCOMPLETE_TECHNICAL_FAILURES"
        print(json.dumps(result, sort_keys=True))
        if result["status"] != "COMPLETE":
            raise SystemExit(3)
    except (PredictionInputError, OSError, ValueError, RuntimeError) as error:
        print(json.dumps({"status": "BLOCKED", "error": str(error)}, sort_keys=True), file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
