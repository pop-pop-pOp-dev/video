#!/usr/bin/env python3
"""Run an already-admitted same-seed A/U/S/F formal bundle on one device.

The bundle shares only frozen provider material.  Every group keeps its own
trainable parameters, master parameters, optimizer, scheduler, RNG, checkpoint
store, progress stream, and formal admission.  Queue/resource admission is an
outer gate; this entry point deliberately refuses missing formal admissions.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nc_rted.captured_sources import CapturedSourceError, load_captured_runtime, load_captured_sources
from nc_rted.interleaved_training import Bundle, SameSeedBundleWorker, clone_with_shared_frozen
from nc_rted.production_runtime import (_checkpoint_identity, assemble, load_formal_admission,
                                        load_manifest, preflight)
from nc_rted.recovery import CheckpointStore, capture_rng
from nc_rted.train_worker import TrainingWorker, publish_progress

GROUPS = ("A", "U", "S", "F")
SCHEMA = "nc_rted_interleaved_formal_bundle/v1"


class FormalBundleError(ValueError):
    pass


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_identity(files: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _load_bundle(path: str, expected_sha256: str) -> dict:
    bundle_path = Path(path).absolute()
    if not bundle_path.is_file() or _sha(bundle_path) != expected_sha256:
        raise FormalBundleError("formal bundle manifest is absent or differs from its SHA-256")
    try:
        document = json.loads(bundle_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise FormalBundleError("formal bundle manifest is invalid JSON") from error
    required = {"schema", "bundle_checkpoint_root", "members", "source_manifest", "source_manifest_sha256", "captured_source_map"}
    if not isinstance(document, dict) or set(document) != required or document.get("schema") != SCHEMA:
        raise FormalBundleError("formal bundle manifest fields differ")
    for key in ("bundle_checkpoint_root", "source_manifest"):
        if not isinstance(document[key], str) or not Path(document[key]).is_absolute():
            raise FormalBundleError("formal bundle has a relative runtime path")
    if not isinstance(document["source_manifest_sha256"], str) or len(document["source_manifest_sha256"]) != 64:
        raise FormalBundleError("formal bundle source binding is invalid")
    source_map = document["captured_source_map"]
    if not isinstance(source_map, dict) or set(source_map) != {"path", "sha256"} or not isinstance(source_map["path"], str) or not Path(source_map["path"]).is_absolute() or not isinstance(source_map["sha256"], str) or len(source_map["sha256"]) != 64:
        raise FormalBundleError("formal bundle captured source map is invalid")
    members = document["members"]
    if not isinstance(members, dict) or set(members) != set(GROUPS):
        raise FormalBundleError("formal bundle needs exactly A/U/S/F members")
    for member in members.values():
        if not isinstance(member, dict) or set(member) != {"runtime", "runtime_sha256", "admission", "admission_sha256"}:
            raise FormalBundleError("formal bundle member fields differ")
        for path_key, hash_key in (("runtime", "runtime_sha256"), ("admission", "admission_sha256")):
            if not isinstance(member[path_key], str) or not Path(member[path_key]).is_absolute() or not isinstance(member[hash_key], str) or len(member[hash_key]) != 64:
                raise FormalBundleError("formal bundle member binding is invalid")
    return document


def _verify_source_manifest(bundle: dict, captured_root: Path) -> None:
    """Verify the source list from the queue's immutable input capture."""
    path = captured_root / "bundle-source-manifest.json"
    if not path.is_file() or _sha(path) != bundle["source_manifest_sha256"]:
        raise FormalBundleError("formal bundle source manifest differs")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        files = document["files"]
    except (OSError, KeyError, ValueError, TypeError) as error:
        raise FormalBundleError("formal bundle source manifest is invalid") from error
    required = {"scripts/nc_rted_interleaved_formal.py"}
    required.update(f"src/nc_rted/{item.name}" for item in (ROOT / "src" / "nc_rted").glob("*.py"))
    if (document.get("schema") != "nc_rted_interleaved_source_manifest/v1" or not isinstance(files, dict) or
            not required.issubset(files) or document.get("code_sha256") != _source_identity(files)):
        raise FormalBundleError("formal bundle source coverage or identity is invalid")
    for relative, expected in files.items():
        candidate = ROOT / relative
        if (not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts or
                not isinstance(expected, str) or len(expected) != 64 or not candidate.is_file() or _sha(candidate) != expected):
            raise FormalBundleError("formal bundle source differs")


def _normalised_runtime(document: dict) -> dict:
    result = copy.deepcopy(document)
    result["run"].pop("run_id")
    result["run"].pop("group")
    result["run"].pop("checkpoint_root")
    result["run"].pop("progress_path")
    return result


def _load_members(bundle: dict, captured_root: Path) -> tuple[dict, dict, dict]:
    source_map = captured_root / "source-map.json"
    if not source_map.is_file() or _sha(source_map) != bundle["captured_source_map"]["sha256"]:
        raise FormalBundleError("formal bundle captured source map differs")
    members, admissions, captured = {}, {}, None
    reference = None
    for group in GROUPS:
        item = bundle["members"][group]
        member_root = captured_root / "members" / group
        admission_path = member_root / "admission.json"
        runtime_path = member_root / "runtime.json"
        admission = load_formal_admission(admission_path, expected_sha256=item["admission_sha256"])
        try:
            group_captured = load_captured_sources(source_map, admission)
            captured_runtime = load_captured_runtime(source_map, json.loads(runtime_path.read_text(encoding="utf-8")))
        except (CapturedSourceError, OSError, ValueError) as error:
            raise FormalBundleError("formal bundle captured source closure is invalid") from error
        manifest = load_manifest(runtime_path, expected_sha256=item["runtime_sha256"], captured_runtime=captured_runtime)
        if manifest.run["mode"] != "formal" or manifest.run["group"] != group:
            raise FormalBundleError("formal bundle member is not its named formal run")
        preflight(manifest)
        if reference is None:
            reference = _normalised_runtime(manifest.document)
            captured = group_captured
        elif _normalised_runtime(manifest.document) != reference:
            raise FormalBundleError("formal bundle members differ outside run storage labels")
        if group_captured != captured:
            raise FormalBundleError("formal bundle members do not share one captured source closure")
        members[group], admissions[group] = manifest, admission
    code_sha = members["A"].document["hashes"]["code_sha256"]
    if any(member.document["hashes"]["code_sha256"] != code_sha for member in members.values()):
        raise FormalBundleError("formal bundle member source identities differ")
    return members, admissions, captured


def _empty_or_recoverable(bundle: dict, members: dict, *, resume: bool | None) -> bool:
    roots = [Path(bundle["bundle_checkpoint_root"]), *(Path(members[group].run["checkpoint_root"]) for group in GROUPS)]
    populated = [root for root in roots if root.exists() and any(root.iterdir())]
    if resume is True:
        if not populated:
            raise FormalBundleError("formal bundle resume has no checkpoint evidence")
    elif resume is False and populated:
        raise FormalBundleError("formal bundle run refuses existing checkpoints; use resume")
    return bool(populated)


def _construct(members: dict, admissions: dict, captured: dict) -> tuple[dict, dict]:
    runtime = assemble(members["A"], admission=admissions["A"], captured_sources=captured)
    exemplar = runtime.worker
    frozen = [parameter for parameter in exemplar.bridge.parameters() if not parameter.requires_grad]
    models = {"A": exemplar.bridge}
    for group in GROUPS[1:]:
        models[group] = clone_with_shared_frozen(exemplar.bridge, frozen)
    workers = {"A": exemplar}
    for group in GROUPS[1:]:
        manifest = members[group]
        workers[group] = TrainingWorker(models[group], exemplar.catalog, exemplar.teachers, exemplar.tokenizer, exemplar.provider,
                                        CheckpointStore(manifest.run["checkpoint_root"], _checkpoint_identity(manifest, exemplar.catalog)),
                                        group=group, seed=manifest.run["seed"], captured_sources=captured)
    for group in GROUPS:
        workers[group]._verify_formal_admission(admissions[group])
    return models, workers


def _write(path: Path, document: dict) -> None:
    payload = (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode()
    if path.exists():
        if path.read_bytes() != payload:
            raise FormalBundleError("formal bundle result already differs")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError:
        if path.read_bytes() != payload:
            raise
    finally:
        temporary.unlink(missing_ok=True)


def _publish_completion(report: Path, result: dict) -> None:
    """Publish the queue's attempt-bound completion after every output exists."""
    required = ("NC_RTED_PRODUCER_COMPLETION", "NC_RTED_JOB_KEY", "NC_RTED_LEASE_TOKEN", "NC_RTED_INPUT_HASH")
    if any(not os.environ.get(key) for key in required):
        raise FormalBundleError("formal bundle requires a queue attempt contract")
    artifacts = []
    if result["status"] == "FORMAL_BUNDLE_COMPLETE":
        for group in GROUPS:
            path = Path(result["final_checkpoints"][group])
            if not path.is_file():
                raise FormalBundleError("formal bundle final checkpoint is missing")
            artifacts.append({"path": str(path.resolve()), "checksum": _sha(path)})
    elif result["status"] == "FORMAL_BUNDLE_SEGMENT_PAUSED":
        path = Path(result["bundle_checkpoint"])
        if not path.is_file() or _sha(path) != result["bundle_checkpoint_sha256"]:
            raise FormalBundleError("formal bundle segment boundary is missing or differs")
    else:
        raise FormalBundleError("formal bundle result status is invalid")
    artifacts.append({"path": str(report.resolve()), "checksum": _sha(report)})
    _write(Path(os.environ["NC_RTED_PRODUCER_COMPLETION"]), {
        "job_key": os.environ["NC_RTED_JOB_KEY"], "lease_token": os.environ["NC_RTED_LEASE_TOKEN"],
        "input_hash": os.environ["NC_RTED_INPUT_HASH"], "artifacts": artifacts,
    })


def _queue_progress(update: int, final: bool, boundary: Path, identities: dict) -> None:
    required = ("NC_RTED_PROGRESS_ROOT", "NC_RTED_PROGRESS_PATH", "NC_RTED_JOB_KEY", "NC_RTED_LEASE_TOKEN")
    if any(not os.environ.get(key) for key in required):
        raise FormalBundleError("formal bundle progress requires a queue attempt contract")
    root = Path(os.environ["NC_RTED_PROGRESS_ROOT"])
    receipt = root / f"bundle-progress-{update:06d}.json"
    _write(receipt, {"schema": "nc_rted_formal_bundle_progress/v1", "job_key": os.environ["NC_RTED_JOB_KEY"],
                     "lease_token": os.environ["NC_RTED_LEASE_TOKEN"], "counter": update, "final": final,
                     "member_identities": identities, "bundle_checkpoint": str(boundary.resolve()),
                     "bundle_checkpoint_sha256": _sha(boundary)})
    commit = root / f"bundle-progress-commit-{update:06d}.json"
    _write(commit, {"schema": "nc_rted_progress_commit_v1", "job_key": os.environ["NC_RTED_JOB_KEY"],
                    "lease_token": os.environ["NC_RTED_LEASE_TOKEN"], "counter": update,
                    "transaction_type": "formal_bundle", "artifact_path": str(receipt.resolve()),
                    "artifact_sha256": _sha(receipt)})
    publish_progress(os.environ["NC_RTED_PROGRESS_PATH"], {"update": update,
                     "job_key": os.environ["NC_RTED_JOB_KEY"], "lease_token": os.environ["NC_RTED_LEASE_TOKEN"],
                     "committed_path": str(commit.resolve())})


def _captured_root(path: str) -> Path:
    root = Path(path).absolute()
    required = {"bundle.json", "bundle-source-manifest.json", "source-map.json"}
    if not root.is_dir() or any(not (root / name).is_file() for name in required):
        raise FormalBundleError("formal bundle needs a complete queue input capture")
    for candidate in [root, *root.rglob("*")]:
        if candidate.stat().st_mode & 0o222:
            raise FormalBundleError("formal bundle input capture is writable")
    return root


def run(bundle: dict, *, captured_root: Path, resume: bool | None,
        start_update: int = 0, stop_update: int = 1000) -> dict:
    if type(start_update) is not int or type(stop_update) is not int or not 0 <= start_update < stop_update <= 1000:
        raise FormalBundleError("formal bundle segment updates are invalid")
    _verify_source_manifest(bundle, captured_root)
    members, admissions, captured = _load_members(bundle, captured_root)
    should_restore = _empty_or_recoverable(bundle, members, resume=resume)
    models, workers = _construct(members, admissions, captured)
    initial_rng = capture_rng()
    bundles = {
        group: Bundle(group, workers[group].trainer,
                      lambda sample_id, material, group=group: workers[group].loss_for_material(sample_id, material),
                      store=workers[group].store,
                      progress=None,
                      rng=copy.deepcopy(initial_rng))
        for group in GROUPS
    }
    identities = {group: dict(workers[group].store.identity) for group in GROUPS}
    worker = SameSeedBundleWorker(bundles, workers["A"].material_for_sample,
                                  bundle_checkpoint_root=bundle["bundle_checkpoint_root"],
                                  after_bundle_checkpoint=lambda update, final, boundary:
                                  _queue_progress(update, final, boundary, identities))
    if start_update > 0 and not should_restore:
        raise FormalBundleError("resumed formal bundle segment requires a committed checkpoint")
    if should_restore:
        worker.restore()
    restored = {group: item.trainer.completed_updates for group, item in workers.items()}
    if len(set(restored.values())) != 1:
        raise FormalBundleError("formal bundle restore diverges across groups")
    resumed_from = next(iter(restored.values()))
    if not start_update <= resumed_from <= stop_update:
        raise FormalBundleError("formal bundle restore is outside the admitted segment interval")
    interval = workers["A"].trainer.recipe.save_interval
    if stop_update != 1000 and stop_update % interval:
        raise FormalBundleError("formal bundle segment stop must be a common checkpoint boundary")
    worker.run(stop_after=stop_update)
    if any(item.trainer.completed_updates != stop_update for item in workers.values()):
        raise FormalBundleError("formal bundle did not stop at the requested absolute update")
    boundary_name = "final.json" if stop_update == 1000 else f"update_{stop_update:06d}.json"
    boundary = Path(bundle["bundle_checkpoint_root"]) / "commits" / boundary_name
    if not boundary.is_file():
        raise FormalBundleError("formal bundle did not durably publish the requested common boundary")
    if stop_update < 1000:
        return {"status": "FORMAL_BUNDLE_SEGMENT_PAUSED", "complete": False,
                "start_update": start_update, "resumed_from_update": resumed_from,
                "stop_update": stop_update, "total_updates": 1000,
                "members": {group: dict(workers[group].store.identity) for group in GROUPS},
                "bundle_checkpoint_root": bundle["bundle_checkpoint_root"],
                "bundle_checkpoint": str(boundary.resolve()), "bundle_checkpoint_sha256": _sha(boundary),
                "shared_material": "frozen_provider_only"}
    return {"status": "FORMAL_BUNDLE_COMPLETE", "members": {group: dict(workers[group].store.identity) for group in GROUPS},
            "final_checkpoints": {group: str((Path(members[group].run["checkpoint_root"]) / "final" / "manifest.json").resolve()) for group in GROUPS},
            "bundle_checkpoint_root": bundle["bundle_checkpoint_root"], "shared_material": "frozen_provider_only"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--bundle-sha256", required=True)
    parser.add_argument("--captured-root", required=True,
                        help="queue-created immutable input directory")
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", choices=("required", "auto"),
                        help="queue retries use auto; direct invocations must name required")
    parser.add_argument("--start-update", type=int, default=0)
    parser.add_argument("--stop-update", type=int, default=1000)
    args = parser.parse_args()
    try:
        captured_root = _captured_root(args.captured_root)
        bundle_path = captured_root / "bundle.json"
        if Path(args.bundle).absolute() != bundle_path:
            raise FormalBundleError("formal bundle must be read from the queue input capture")
        bundle = _load_bundle(bundle_path, args.bundle_sha256)
        if args.start_update == 0 and args.resume not in (None, "auto"):
            raise FormalBundleError("initial formal bundle segment only permits an automatic durable retry")
        if args.start_update > 0 and args.resume not in ("required", "auto"):
            raise FormalBundleError("resumed formal bundle segment requires --resume required")
        resume = {None: False, "required": True, "auto": None}[args.resume]
        result = run(bundle, captured_root=captured_root, resume=resume,
                     start_update=args.start_update, stop_update=args.stop_update)
        report = Path(args.output).absolute()
        _write(report, result)
        _publish_completion(report, result)
    except (FormalBundleError, ValueError, OSError) as error:
        raise SystemExit(f"formal bundle refused: {error}") from error


if __name__ == "__main__":
    main()
