#!/usr/bin/env python3
"""Blind prediction entry point; model-specific adapters are supplied explicitly."""
from __future__ import annotations

import argparse, hashlib, importlib, importlib.abc, importlib.machinery, json
from pathlib import Path
import sys

_EXECUTED_DRIVER_CODE = sys._getframe().f_code
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class BootstrapError(RuntimeError):
    pass


PredictionInputError = BootstrapError


# This mirrors prediction_inputs._IMPLEMENTATION_FILES without importing that
# module. The driver must establish source integrity before it imports it.
_IMPLEMENTATION_FILES = frozenset({
    "scripts/nc_rted_predict.py", "src/nc_rted/__init__.py", "src/nc_rted/batches.py", "src/nc_rted/bridge.py",
    "src/nc_rted/detection_media.py", "src/nc_rted/detection_provider.py", "src/nc_rted/detector.py",
    "src/nc_rted/frozen_vision.py", "src/nc_rted/inherited_memory.py", "src/nc_rted/loading.py",
    "src/nc_rted/media_observer.py", "src/nc_rted/numerics.py", "src/nc_rted/observation.py",
    "src/nc_rted/observation_cache.py", "src/nc_rted/prediction_adapters.py", "src/nc_rted/prediction_inputs.py",
    "src/nc_rted/prediction_media.py", "src/nc_rted/prediction_runtime.py", "src/nc_rted/prediction_store.py",
    "src/nc_rted/prediction_worker.py", "src/nc_rted/production_runtime.py", "src/nc_rted/recovery.py",
    "src/nc_rted/task_inputs.py", "src/nc_rted/model.py",
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
    global PredictionInputError, load_model_artifact, load_prediction_plan, verify_implementation_manifest, PredictionStore, PredictionWorker
    from nc_rted.prediction_inputs import PredictionInputError, load_model_artifact, load_prediction_plan, verify_implementation_manifest
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
        loader, vad, vau = _factory(args.adapter_factory, plan, artifact, device=args.device)
        store = PredictionStore(plan.output_root / artifact.task_id, run_id=plan.run_id, manifest_sha256=plan.manifest_sha256,
                                model_task=artifact.task_id, model_binding_sha256=artifact.manifest_sha256)
        result.update(PredictionWorker(plan, store, loader=loader, vad=vad, vau=vau, model=artifact).run())
        result["status"] = "COMPLETE" if result["succeeded"] == result["expected"] and result["technical_failures"] == 0 and result["missing"] == 0 else "INCOMPLETE_TECHNICAL_FAILURES"
        print(json.dumps(result, sort_keys=True))
        if result["status"] != "COMPLETE":
            raise SystemExit(3)
    except (PredictionInputError, OSError, ValueError, RuntimeError) as error:
        print(json.dumps({"status": "BLOCKED", "error": str(error)}, sort_keys=True), file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
