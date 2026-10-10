#!/usr/bin/env python3
"""Root-scheduled real-model A/U/S/F interleaving admission harness.

The harness loads the accepted inherited Slow stack once and creates four
independent trainable/optimizer states.  It does not call the single-group
``ProductionRuntime.run`` method.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any
import uuid
from dataclasses import fields, is_dataclass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from nc_rted.interleaved_training import Bundle, SameSeedBundleWorker, clone_with_shared_frozen
from nc_rted.production_runtime import _checkpoint_identity, assemble, load_manifest, preflight
from nc_rted.recovery import CheckpointStore, capture_rng, restore_rng
from nc_rted.train_worker import TrainingWorker
from nc_rted.training import Recipe

SCHEMA = "nc_rted_interleaved_gpu_harness/v1"
GROUPS = ("A", "U", "S", "F")


class InjectedInterruption(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode("utf-8")
    if path.exists():
        if path.read_bytes() != encoded:
            raise FileExistsError(f"immutable diagnostic evidence already differs: {path}")
        _sync_evidence(path)
        return
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("xb") as stream:
        stream.write(encoded); stream.flush(); os.fsync(stream.fileno())
    try:
        # link(2) is an atomic no-overwrite publication primitive on this
        # filesystem; retain an existing identical report but never replace it.
        os.link(temporary, path)
    except FileExistsError:
        if path.read_bytes() != encoded:
            raise FileExistsError(f"immutable diagnostic evidence already differs: {path}")
    finally:
        temporary.unlink(missing_ok=True)
    _sync_evidence(path)


def _sync_evidence(path: Path) -> None:
    """Finish durability for both a new link and an identical retry."""
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _load_bundle_manifest(path: str | Path, expected_sha256: str) -> dict:
    path = Path(path).absolute()
    if not path.is_file() or _sha256(path) != expected_sha256:
        raise ValueError("bundle manifest is absent or its SHA-256 differs")
    document = json.loads(path.read_text(encoding="utf-8"))
    required = {"schema", "diagnostic_updates", "diagnostic_checkpoint_interval", "bundle_checkpoint_root", "members",
                "source_manifest", "source_manifest_sha256"}
    if not isinstance(document, dict) or set(document) != required or document["schema"] != SCHEMA:
        raise ValueError("invalid interleaved GPU harness manifest")
    if type(document["diagnostic_updates"]) is not int or not 1 <= document["diagnostic_updates"] <= 1000:
        raise ValueError("diagnostic update prefix is invalid")
    if type(document["diagnostic_checkpoint_interval"]) is not int or not 1 <= document["diagnostic_checkpoint_interval"] <= 50:
        raise ValueError("diagnostic checkpoint interval is invalid")
    if document["diagnostic_checkpoint_interval"] > document["diagnostic_updates"]:
        raise ValueError("diagnostic prefix must reach a checkpoint boundary")
    if not isinstance(document["bundle_checkpoint_root"], str) or not Path(document["bundle_checkpoint_root"]).is_absolute():
        raise ValueError("bundle checkpoint root must be absolute")
    if (not isinstance(document["source_manifest"], str) or not Path(document["source_manifest"]).is_absolute() or
            not isinstance(document["source_manifest_sha256"], str) or len(document["source_manifest_sha256"]) != 64):
        raise ValueError("source manifest binding is invalid")
    members = document["members"]
    if not isinstance(members, dict) or set(members) != set(GROUPS):
        raise ValueError("exactly A/U/S/F member manifests are required")
    for group, member in members.items():
        if not isinstance(member, dict) or set(member) != {"manifest", "sha256"}:
            raise ValueError(f"invalid {group} member manifest")
        if not isinstance(member["manifest"], str) or not Path(member["manifest"]).is_absolute():
            raise ValueError(f"invalid {group} manifest path")
        if not isinstance(member["sha256"], str) or len(member["sha256"]) != 64:
            raise ValueError(f"invalid {group} manifest SHA-256")
    return document


def _source_identity(files: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _verify_source_manifest(harness: dict, members: dict, *, root: Path = ROOT) -> None:
    manifest = Path(harness["source_manifest"])
    if not manifest.is_file() or _sha256(manifest) != harness["source_manifest_sha256"]:
        raise ValueError("bound source manifest differs")
    document = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or set(document) != {"schema", "code_sha256", "files"} or document["schema"] != "nc_rted_interleaved_source_manifest/v1":
        raise ValueError("source manifest schema is invalid")
    files = document["files"]
    required = {"scripts/nc_rted_interleaved_gpu_diagnostic.py"}
    required.update(f"src/nc_rted/{path.name}" for path in (root / "src" / "nc_rted").glob("*.py"))
    if not isinstance(files, dict) or not required.issubset(files) or document["code_sha256"] != _source_identity(files):
        raise ValueError("source manifest coverage or identity is invalid")
    for relative, expected in files.items():
        candidate = root / relative
        if (not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts or
                not isinstance(expected, str) or len(expected) != 64 or not candidate.is_file() or _sha256(candidate) != expected):
            raise ValueError("accepted runtime source differs")
    if any(member.document["hashes"]["code_sha256"] != document["code_sha256"] for member in members.values()):
        raise ValueError("member code identity differs from accepted source manifest")


def _normalised_runtime(document: dict) -> dict:
    result = copy.deepcopy(document)
    for key in ("run_id", "group", "checkpoint_root", "progress_path"):
        result["run"].pop(key)
    return result


def _load_members(bundle_manifest: dict) -> dict:
    members = {group: load_manifest(spec["manifest"], expected_sha256=spec["sha256"])
               for group, spec in bundle_manifest["members"].items()}
    reference = _normalised_runtime(members["A"].document)
    for group, manifest in members.items():
        if manifest.run["mode"] != "diagnostic" or manifest.run["group"] != group:
            raise ValueError(f"{group} is not its diagnostic runtime manifest")
        if manifest.run.get("diagnostic_updates") != bundle_manifest["diagnostic_updates"]:
            raise ValueError("member diagnostic prefix differs from bundle manifest")
        if _normalised_runtime(manifest.document) != reference:
            raise ValueError("member runtime inputs differ outside storage labels")
        preflight(manifest)
    return members


def _recipe(interval: int) -> Recipe:
    # The 1000-update, accumulation-8 kernel is unchanged.  This only shortens
    # diagnostic checkpoint cadence to make bounded interruption testable.
    return Recipe(save_interval=interval)


def _digest_value(value: Any, digest) -> None:
    def frame(tag: str, payload: bytes) -> None:
        digest.update(tag.encode("ascii")); digest.update(len(payload).to_bytes(8, "big")); digest.update(payload)
    frame("type", f"{type(value).__module__}.{type(value).__qualname__}".encode())
    if isinstance(value, torch.Tensor):
        frame("dtype", str(value.dtype).encode()); frame("shape", repr(tuple(value.shape)).encode()); frame("device", str(value.device).encode())
        flattened = value.detach().contiguous().cpu().reshape(-1)
        frame("tensor", flattened.view(torch.uint8).numpy().tobytes()); return
    if is_dataclass(value):
        frame("field_count", len(fields(value)).to_bytes(8, "big"))
        for field in fields(value): frame("field", field.name.encode()); _digest_value(getattr(value, field.name), digest)
        return
    if isinstance(value, dict):
        frame("mapping_length", len(value).to_bytes(8, "big"))
        for key in sorted(value, key=repr): _digest_value(key, digest); _digest_value(value[key], digest)
        return
    if isinstance(value, (tuple, list)):
        frame("sequence_length", len(value).to_bytes(8, "big"))
        for item in value: _digest_value(item, digest)
        return
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        frame("scalar", repr(value).encode()); return
    if hasattr(value, "__dict__"):
        _digest_value(vars(value), digest); return
    raise ValueError(f"cannot digest prepared material {type(value)!r}")


def _material_digest(value: Any) -> str:
    digest = hashlib.sha256()
    _digest_value(value, digest)
    return digest.hexdigest()


class _DigestingProvider:
    """Retain content identities, never prepared tensors, across updates."""
    def __init__(self, provider):
        self.provider, self.digests = provider, {}

    def __call__(self, sample_id: str):
        value = self.provider(sample_id)
        digest = _material_digest(value)
        prior = self.digests.setdefault(sample_id, digest)
        if prior != digest: raise RuntimeError("prepared material changed for the same sample")
        return value


def _trainer_state_digest(model: torch.nn.Module, trainer, rng: dict) -> str:
    digest = hashlib.sha256()
    _digest_value(model.state_dict(), digest)
    _digest_value(trainer.master_parameters, digest)
    _digest_value(trainer.optimizer.state_dict(), digest)
    _digest_value(trainer.scheduler.state_dict(), digest)
    _digest_value(trainer.metadata(), digest)
    _digest_value(rng, digest)
    return digest.hexdigest()


def _assembly_manifest(manifest, serial_root: Path | None):
    """Give factory-only assembly an isolated checkpoint namespace."""
    if serial_root is None: return manifest
    document = copy.deepcopy(manifest.document)
    document["run"]["checkpoint_root"] = str(serial_root / "factory-A")
    return type(manifest)(manifest.path, manifest.config_sha256, document)


def _construct_workers(members: dict, harness: dict, *, store_root_override: Path | None = None):
    """Use the production factory once; clones retain only shared frozen base."""
    # ``assemble`` creates its own TrainingWorker/CheckpointStore.  Give that
    # transient factory worker an isolated serial root so it cannot leave a
    # .store.lock in the fault/recovery namespace.
    runtime = assemble(_assembly_manifest(members["A"], store_root_override))
    exemplar = runtime.worker
    frozen = [parameter for parameter in exemplar.bridge.parameters() if not parameter.requires_grad]
    models = {"A": exemplar.bridge}
    for group in GROUPS[1:]:
        models[group] = clone_with_shared_frozen(exemplar.bridge, frozen)
    workers = {}
    for group in GROUPS:
        manifest = members[group]
        root = store_root_override / group if store_root_override is not None else Path(manifest.run["checkpoint_root"])
        workers[group] = TrainingWorker(
            models[group], exemplar.catalog, exemplar.teachers, exemplar.tokenizer, exemplar.provider,
            CheckpointStore(root, _checkpoint_identity(manifest, exemplar.catalog)),
            group=group, seed=manifest.run["seed"], recipe=_recipe(harness["diagnostic_checkpoint_interval"]))
    return models, workers


def _physical_gpu_inventory() -> dict[str, str]:
    """Return physical GPU UUIDs from the driver inventory."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError("CUDA driver inventory is unavailable") from error
    rows: dict[str, str] = {}
    try:
        for line in result.stdout.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != 2 or not fields[0].isdigit() or not fields[1] or "\n" in fields[1]:
                raise ValueError
            if fields[0] in rows or fields[1] in rows.values():
                raise ValueError
            rows[fields[0]] = fields[1]
    except ValueError as error:
        raise RuntimeError("CUDA driver inventory is invalid") from error
    if not rows:
        raise RuntimeError("CUDA driver inventory is empty")
    return rows


def _visible_gpu_identity(visible: str) -> str:
    """Resolve one CUDA_VISIBLE_DEVICES token to a unique physical GPU."""
    token = visible.strip()
    if not token or "," in visible:
        raise RuntimeError("diagnostic requires exactly one CUDA-visible device")
    if token.isdigit():
        raise RuntimeError("diagnostic requires a CUDA-visible GPU UUID, not an ordinal")
    rows = _physical_gpu_inventory()
    matches = [gpu_uuid for gpu_uuid in rows.values() if gpu_uuid == token or gpu_uuid.startswith(token)]
    if len(matches) != 1:
        raise RuntimeError("CUDA_VISIBLE_DEVICES does not resolve to one physical GPU")
    return matches[0]


def _cuda_binding() -> dict:
    """Resolve and validate the one real GPU after deterministic setup."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    gpu_uuid = _visible_gpu_identity(visible)
    if not torch.cuda.is_available():
        raise RuntimeError("diagnostic requires an available CUDA device")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("diagnostic requires exactly one CUDA logical device")
    if torch.cuda.current_device() != 0:
        raise RuntimeError("diagnostic requires CUDA logical device 0")
    # Report the resolved physical UUID in both existing schema fields.  The
    # launcher sets CUDA_VISIBLE_DEVICES to this UUID, and a unique prefix is
    # accepted only after driver inventory resolution.
    return {"cuda_visible_devices": gpu_uuid, "gpu_uuid": gpu_uuid}


def _verified_parent_binding(members: dict, harness: dict, expected: dict, *,
                             store_root_override: Path | None = None):
    """Assemble first so production numerical policy precedes parent CUDA init."""
    models, workers = _construct_workers(members, harness, store_root_override=store_root_override)
    observed = _cuda_binding()
    if observed != expected:
        raise RuntimeError("parent diagnostic GPU binding differs from fresh binding probe")
    return models, workers, observed


def _serial_reference(models: dict, workers: dict, updates: int, cuda_binding: dict) -> dict:
    initial_rng = capture_rng()
    provider = _DigestingProvider(workers["A"].material_for_sample)
    groups = {}
    for group in GROUPS:
        restore_rng(initial_rng)
        reports = []
        trainer = workers[group].trainer
        trainer.run(lambda sample_id, group=group: workers[group].loss_for_material(sample_id, provider(sample_id)),
                    progress=reports.append, stop_after=updates)
        groups[group] = dict(state_digest=_trainer_state_digest(models[group], trainer, capture_rng()), reports=reports,
                             cursor=trainer.cursor, completed_updates=trainer.completed_updates)
    return dict(groups=groups, material_digests=provider.digests, cuda_binding=cuda_binding)


def _empty_roots(members: dict, harness: dict) -> None:
    roots = [Path(members[group].run["checkpoint_root"]) for group in GROUPS] + [Path(harness["bundle_checkpoint_root"])]
    for root in roots:
        if root.exists() and any(root.iterdir()):
            raise RuntimeError(f"diagnostic checkpoint root is not empty: {root}")


def _bundle_run(members: dict, harness: dict, expected_cuda_binding: dict, *, fail_group: str | None = None,
                resume: bool = False) -> dict:
    # Start before model construction so the reported peak includes inherited
    # loading, clone construction, provider preparation, and all updates. Each
    # bundle mode is a fresh process, so its CUDA peak begins empty without a
    # post-assembly reset that would discard the construction peak.
    models, workers, cuda_binding = _verified_parent_binding(members, harness, expected_cuda_binding)
    updates = harness["diagnostic_updates"]
    initial = {group: {name: parameter.detach().cpu().clone() for name, parameter in models[group].named_parameters()
                       if parameter.requires_grad} for group in GROUPS}
    initial_rng = capture_rng()
    def interrupt(group: str, update: int, final: bool) -> None:
        if fail_group == group:
            raise InjectedInterruption(f"interrupted after {group} publication at update {update}")
    reports = {group: [] for group in GROUPS}
    provider = _DigestingProvider(workers["A"].material_for_sample)
    bundles = {group: Bundle(group, workers[group].trainer,
                             lambda sample_id, material, group=group: workers[group].loss_for_material(sample_id, material),
                             store=workers[group].store, progress=reports[group].append,
                             rng=copy.deepcopy(initial_rng)) for group in GROUPS}
    worker = SameSeedBundleWorker(bundles, provider,
                                  bundle_checkpoint_root=harness["bundle_checkpoint_root"],
                                  after_checkpoint_publication=interrupt if fail_group else None)
    if resume:
        worker.restore()
    started = time.perf_counter(); worker.run(stop_after=updates); elapsed = time.perf_counter() - started
    state = {group: _trainer_state_digest(models[group], workers[group].trainer, bundles[group].rng) for group in GROUPS}
    gradient_and_update = {}
    for group in GROUPS:
        changes = {"new": 0., "lora": 0.}
        for name, parameter in models[group].named_parameters():
            if not parameter.requires_grad: continue
            bucket = "new" if name.startswith("evidence.") else "lora"
            changes[bucket] += float((parameter.detach().cpu() - initial[group][name]).abs().sum())
        summary = reports[group][-1]["gradient_summary"]
        if not all(changes[name] > 0 and summary[name]["nonzero"] > 0 for name in ("new", "lora")):
            raise RuntimeError("diagnostic needs nonzero LoRA/new-module gradients and updates")
        gradient_and_update[group] = dict(update_l1=changes, gradient_summary=summary)
    return dict(status="BUNDLED_DIAGNOSTIC_COMPLETE", elapsed_seconds=elapsed,
                peak_memory_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
                peak_reserved_memory_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0,
                completed={group: item.trainer.completed_updates for group, item in workers.items()},
                cursor={group: item.trainer.cursor for group, item in workers.items()}, state_digest=state,
                reports=reports, gradient_and_update=gradient_and_update,
                material_digests=provider.digests, cuda_binding=cuda_binding,
                diagnostic_recipe=dict(updates=1000, accumulation=8,
                                       checkpoint_interval=harness["diagnostic_checkpoint_interval"]))


def _release(*objects) -> None:
    del objects
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


def _child(args, mode: str, output: Path, extra: list[str]) -> subprocess.CompletedProcess:
    command = [sys.executable, str(Path(__file__).absolute()), "--manifest", args.manifest,
               "--manifest-sha256", args.manifest_sha256, "--runtime-source-root", args.runtime_source_root,
               "--output", str(output), "--mode", mode, *extra]
    return subprocess.run(command, check=False)


def _binding_probe(args) -> dict:
    """Get CUDA runtime evidence in a disposable process before parent assembly."""
    output = Path(args.output).with_suffix(".binding-probe.json")
    if _child(args, "binding-probe", output, []).returncode:
        raise RuntimeError("fresh CUDA binding probe failed")
    document = json.loads(output.read_text())
    binding = document.get("cuda_binding")
    if (document.get("status") != "CUDA_BINDING_PROBE_COMPLETE" or not isinstance(binding, dict)
            or set(binding) != {"cuda_visible_devices", "gpu_uuid"}
            or any(not isinstance(value, str) or not value for value in binding.values())):
        raise RuntimeError("fresh CUDA binding probe emitted invalid evidence")
    return binding


def _run_parent(args, harness: dict, members: dict, expected_cuda_binding: dict) -> dict:
    # Parent computes the four serial oracles, then releases all CUDA memory
    # before fault/resume each construct their own independent process.
    models, workers, cuda_binding = _verified_parent_binding(
        members, harness, expected_cuda_binding, store_root_override=Path(args.output).with_suffix(".serial-stores"))
    serial = _serial_reference(models, workers, harness["diagnostic_updates"], cuda_binding)
    serial_path = Path(args.output).with_suffix(".serial.json")
    _write_json(serial_path, serial)
    models = workers = None
    _release()
    fault_path = Path(args.output).with_suffix(".fault.json")
    if _child(args, "fault", fault_path, ["--fault-after-group", args.fault_after_group]).returncode != 75:
        raise RuntimeError("fault process did not exit at the requested checkpoint publication")
    resume_path = Path(args.output).with_suffix(".resume.json")
    if _child(args, "resume", resume_path, ["--serial-reference", str(serial_path)]).returncode:
        raise RuntimeError("fresh-process recovery failed")
    fault = json.loads(fault_path.read_text())
    resumed = json.loads(resume_path.read_text())
    if fault.get("cuda_binding") != cuda_binding or resumed.get("cuda_binding") != cuda_binding:
        raise RuntimeError("fresh-process diagnostic GPU binding differs from serial binding")
    return dict(status="BUNDLED_DIAGNOSTIC_COMPLETE", serial_reference=str(serial_path),
                fault_report=str(fault_path), cuda_binding=cuda_binding, resumed=resumed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--runtime-source-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=("prepare", "binding-probe", "run", "fault", "resume"), required=True)
    parser.add_argument("--fault-after-group", choices=GROUPS, default="U")
    parser.add_argument("--serial-reference")
    args = parser.parse_args()
    if Path(args.runtime_source_root).resolve() != ROOT.resolve():
        raise SystemExit("runtime source root must match this immutable harness checkout")
    output = Path(args.output)
    if args.mode == "binding-probe":
        # This child exists solely to obtain disposable driver/Torch metadata.
        # It must not repeat the parent's sealed bundle/member/media validation.
        _write_json(output, dict(status="CUDA_BINDING_PROBE_COMPLETE", cuda_binding=_cuda_binding()))
        return
    harness = _load_bundle_manifest(args.manifest, args.manifest_sha256)
    members = _load_members(harness)
    _verify_source_manifest(harness, members)
    if args.mode == "prepare":
        result = dict(status="PREFLIGHT_COMPLETE", schema=SCHEMA,
                      members={group: str(member.path) for group, member in members.items()},
                      diagnostic_updates=harness["diagnostic_updates"], recipe=dict(updates=1000, accumulation=8))
    elif args.mode == "run":
        result = _run_parent(args, harness, members, _binding_probe(args))
    elif args.mode == "fault":
        cuda_binding = _binding_probe(args)
        _empty_roots(members, harness)
        try:
            _bundle_run(members, harness, cuda_binding, fail_group=args.fault_after_group)
        except InjectedInterruption as error:
            _write_json(output, dict(status="INJECTED_INTERRUPT", detail=str(error), cuda_binding=cuda_binding))
            raise SystemExit(75)
        raise RuntimeError("fault mode completed without interruption")
    else:
        if not args.serial_reference: raise SystemExit("resume mode requires --serial-reference")
        cuda_binding = _binding_probe(args)
        serial = json.loads(Path(args.serial_reference).read_text())
        if serial.get("cuda_binding") != cuda_binding:
            raise RuntimeError("fresh-process GPU binding differs from serial binding")
        result = _bundle_run(members, harness, cuda_binding, resume=True)
        if result["state_digest"] != {group: serial["groups"][group]["state_digest"] for group in GROUPS}:
            raise RuntimeError("fresh-process bundled replay differs from serial reference")
        if result["reports"] != {group: serial["groups"][group]["reports"] for group in GROUPS}:
            raise RuntimeError("fresh-process bundled forward/loss/gradient reports differ from serial reference")
        if result["material_digests"] != serial["material_digests"]:
            raise RuntimeError("fresh-process prepared prefix differs from serial reference")
        result["serial_comparison"] = "exact_state_digest_match"
    _write_json(output, result)


if __name__ == "__main__":
    main()
