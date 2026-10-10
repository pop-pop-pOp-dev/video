"""Atomic incremental checkpoints at completed optimizer-update boundaries.

Only trainable tensors are saved; inherited frozen weights are bound by hashes.
Each directory is immutable and self-verifying. A crash after directory rename
but before the pointer update is recovered by scanning committed directories.
"""
from __future__ import annotations

from dataclasses import asdict
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import uuid

import numpy as np
import torch

from .training import IncrementalTrainer


class RecoveryError(RuntimeError):
    pass

class CheckpointReadUncertain(RecoveryError):
    pass


def validate_checkpoint_payload(path: Path, manifest: dict) -> dict:
    """Non-mutating structural validation for a published checkpoint payload."""
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except OSError as exc:
        raise CheckpointReadUncertain("checkpoint payload could not be read") from exc
    except Exception as exc:
        raise RecoveryError("checkpoint payload cannot be safely deserialized") from exc
    required = {"trainable", "optimizer_master", "optimizer", "scheduler", "training", "rng"}
    if not isinstance(state, dict) or set(state) != required:
        raise RecoveryError("checkpoint payload structure is invalid")
    info = state["training"]
    metadata={"seed","recipe","order","completed_updates","cursor"}
    derived_order=hashlib.sha256(json.dumps(info.get("order", []) if isinstance(info,dict) else [], ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    if (not isinstance(info, dict) or not metadata <= set(info) or not isinstance(info.get("seed"), int) or
            not isinstance(info.get("recipe"), dict) or not info["recipe"] or not isinstance(info.get("order"), list) or not info["order"] or not all(isinstance(item,str) and item for item in info["order"]) or
            isinstance(info.get("completed_updates"),bool) or not isinstance(info.get("completed_updates"),int) or info["completed_updates"] < 1 or
            isinstance(info.get("cursor"),bool) or not isinstance(info.get("cursor"),int) or
            isinstance(info["recipe"].get("accumulation"),bool) or not isinstance(info["recipe"].get("accumulation"),int) or info["recipe"]["accumulation"] < 1 or
            info["cursor"] != info["completed_updates"] * info["recipe"]["accumulation"] or
            isinstance(manifest.get("completed_updates"),bool) or not isinstance(manifest.get("completed_updates"),int) or manifest["completed_updates"] < 1 or
            info.get("completed_updates") != manifest.get("completed_updates") or info.get("cursor") != manifest.get("cursor") or
            not isinstance(info.get("order_sha256"),str) or info["order_sha256"] != derived_order or
            ("order_sha256" in manifest and info.get("order_sha256") != manifest["order_sha256"]) or
            (isinstance(manifest.get("identity"),dict) and str(info["seed"]) != manifest["identity"].get("seed"))):
        raise RecoveryError("checkpoint payload/manifest progress mismatch")
    if not isinstance(state["trainable"], dict) or not state["trainable"] or set(state["trainable"]) != set(state["optimizer_master"]):
        raise RecoveryError("checkpoint trainable/master structure is invalid")
    optimizer=state["optimizer"]
    if (not isinstance(optimizer, dict) or not isinstance(optimizer.get("state"), dict) or not optimizer["state"] or not isinstance(optimizer.get("param_groups"), list) or not optimizer["param_groups"] or any(not isinstance(group,dict) or not isinstance(group.get("params"),list) or not group["params"] or any(isinstance(param,bool) or not isinstance(param,int) for param in group["params"]) for group in optimizer["param_groups"]) or
            not isinstance(state["scheduler"], dict) or state["scheduler"].get("last_epoch") != manifest.get("completed_updates") or
            not isinstance(state["rng"], dict) or not {"python","torch","numpy","cuda"} <= set(state["rng"])):
        raise RecoveryError("checkpoint optimizer/scheduler/RNG structure is invalid")
    for param_id, entry in optimizer["state"].items():
        if isinstance(param_id, bool) or not isinstance(param_id, int) or not isinstance(entry, dict):
                raise RecoveryError("checkpoint optimizer moment structure is invalid")
        if set(entry) != {"step", "exp_avg", "exp_avg_sq"} or not isinstance(entry["step"], torch.Tensor) or entry["step"].numel() != 1:
            raise RecoveryError("checkpoint optimizer moment structure is invalid")
    parameter_names=[]; parameter_ids=[]
    for group in optimizer["param_groups"]:
        names=group.get("parameter_names")
        required_group={"params","name","parameter_names","initial_lr","lr","weight_decay","betas","eps","amsgrad"}
        if (not required_group <= set(group) or not isinstance(names,list) or len(names) != len(group["params"]) or not all(isinstance(name,str) and name for name in names) or not isinstance(group["name"],str) or not isinstance(group["initial_lr"],(int,float)) or not isinstance(group["lr"],(int,float)) or not math.isfinite(group["lr"]) or group["lr"] < 0 or not isinstance(group["weight_decay"],(int,float)) or not isinstance(group["betas"],tuple) or len(group["betas"]) != 2 or not all(isinstance(value,(int,float)) for value in group["betas"]) or not isinstance(group["eps"],(int,float)) or not isinstance(group["amsgrad"],bool)):
            raise RecoveryError("checkpoint optimizer parameter mapping is invalid")
        parameter_names.extend(names); parameter_ids.extend(group["params"])
    if len(set(parameter_names)) != len(parameter_names) or len(set(parameter_ids)) != len(parameter_ids) or set(parameter_names) != set(state["optimizer_master"]) or set(parameter_ids) != set(optimizer["state"]):
        raise RecoveryError("checkpoint optimizer parameter mapping is invalid")
    for name, param_id in zip(parameter_names, parameter_ids):
        entry=optimizer["state"][param_id]; master=state["optimizer_master"][name]
        if (not isinstance(entry["exp_avg"],torch.Tensor) or not isinstance(entry["exp_avg_sq"],torch.Tensor) or
                entry["exp_avg"].shape != master.shape or entry["exp_avg_sq"].shape != master.shape or
                entry["exp_avg"].dtype != torch.float32 or entry["exp_avg_sq"].dtype != torch.float32 or
                not bool(torch.isfinite(entry["exp_avg"]).all()) or not bool(torch.isfinite(entry["exp_avg_sq"]).all()) or
                bool((entry["exp_avg_sq"] < 0).any())):
            raise RecoveryError("checkpoint optimizer moment/master mismatch")
        if (not bool(torch.isfinite(entry["step"]).all()) or float(entry["step"].item()) < 1 or float(entry["step"].item()) > manifest["completed_updates"] or not float(entry["step"].item()).is_integer()):
            raise RecoveryError("checkpoint optimizer step is invalid")
        if (not isinstance(entry["exp_avg"], torch.Tensor) or not isinstance(entry["exp_avg_sq"], torch.Tensor) or
                entry["exp_avg"].shape != entry["exp_avg_sq"].shape or not entry["exp_avg"].numel()):
            raise RecoveryError("checkpoint optimizer moment structure is invalid")
    rng=state["rng"]; numpy_rng=rng["numpy"]
    if (not isinstance(rng["python"], tuple) or not isinstance(rng["torch"], torch.Tensor) or rng["torch"].dtype != torch.uint8 or rng["torch"].ndim != 1 or
            not isinstance(numpy_rng, dict) or set(numpy_rng) != {"algorithm","keys","position","has_gauss","cached_gauss"} or not isinstance(numpy_rng["algorithm"],str) or
            not isinstance(numpy_rng["keys"],torch.Tensor) or numpy_rng["keys"].dtype != torch.int64 or numpy_rng["keys"].ndim != 1 or not numpy_rng["keys"].numel() or
            any(isinstance(numpy_rng[key],bool) or not isinstance(numpy_rng[key],int) for key in ("position","has_gauss")) or
            not isinstance(numpy_rng["cached_gauss"],float) or not isinstance(rng["cuda"],list) or any(not isinstance(item,torch.Tensor) or item.dtype != torch.uint8 or item.ndim != 1 or not item.numel() for item in rng["cuda"])):
        raise RecoveryError("checkpoint RNG structure is invalid")
    # Validate restoratability with isolated generators before any caller can
    # mutate a trainer or the process-global RNG state.
    try:
        random.Random().setstate(rng["python"])
        torch.Generator(device="cpu").set_state(rng["torch"])
        np.random.RandomState().set_state((numpy_rng["algorithm"], numpy_rng["keys"].numpy().astype(np.uint32),
                                           numpy_rng["position"], numpy_rng["has_gauss"], numpy_rng["cached_gauss"]))
    except (TypeError, ValueError, RuntimeError, OverflowError) as exc:
        raise RecoveryError("checkpoint RNG state is not restorable") from exc
    for name, value in state["trainable"].items():
        master = state["optimizer_master"][name]
        if (not isinstance(value, torch.Tensor) or not isinstance(master, torch.Tensor) or value.shape != master.shape or master.dtype != torch.float32 or not bool(torch.isfinite(value).all()) or not bool(torch.isfinite(master).all()) or not torch.equal(master.to(value.dtype),value)):
            raise RecoveryError("checkpoint trainable tensor structure is invalid")
    return state


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def capture_rng() -> dict:
    numpy_state = np.random.get_state()
    return dict(python=random.getstate(), torch=torch.get_rng_state(),
                numpy=dict(algorithm=numpy_state[0], keys=torch.from_numpy(numpy_state[1].astype(np.int64)),
                           position=numpy_state[2], has_gauss=numpy_state[3], cached_gauss=numpy_state[4]),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state: dict) -> None:
    expected_cuda = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if len(state["cuda"]) != expected_cuda:
        raise RecoveryError("CUDA RNG topology differs; cross-device resume requires separate acceptance")
    try:
        for index, saved in enumerate(state["cuda"]):
            torch.Generator(device=f"cuda:{index}").set_state(saved)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise RecoveryError("CUDA RNG state is not restorable") from exc
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n["algorithm"], n["keys"].numpy().astype(np.uint32), n["position"],
                         n["has_gauss"], n["cached_gauss"]))
    torch.set_rng_state(state["torch"])
    if expected_cuda:
        torch.cuda.set_rng_state_all(state["cuda"])


class CheckpointStore:
    def __init__(self, root: str | Path, identity: dict[str, str], *, min_free_bytes: int = 20 << 30):
        required = {"run_id", "group", "seed", "code_sha256", "config_sha256", "data_sha256",
                    "teacher_sha256", "inherited_weights_sha256", "runtime_sha256"}
        if set(identity) != required or not all(isinstance(v, str) and v for v in identity.values()):
            raise RecoveryError("checkpoint identity is incomplete")
        if identity["group"] not in {"A", "U", "S", "F"}:
            raise RecoveryError("invalid training group")
        for key, value in identity.items():
            if key.endswith("sha256") and (len(value) != 64 or any(c not in "0123456789abcdef" for c in value)):
                raise RecoveryError(f"invalid identity hash: {key}")
        self.root = Path(root).absolute()
        if self.root.is_symlink():
            raise RecoveryError("checkpoint root may not be a symlink")
        self.root.mkdir(parents=True, exist_ok=True)
        self.identity = dict(identity)
        self.min_free_bytes = min_free_bytes
        if min_free_bytes < 0:
            raise RecoveryError("negative disk reserve")
        self.reconcile()

    @contextmanager
    def _lock(self):
        with (self.root / ".store.lock").open("a+") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    def reconcile(self) -> list[dict]:
        """Reclaim this store's interrupted writes only after acquiring its writer lock.

        Ownership markers are durable before any large payload write. Unknown
        artifacts remain untouched and count against the ordinary free-space guard.
        Failure metadata is persisted before reclaiming replaceable partial data.
        """
        with self._lock():
            return self._reconcile_locked()

    def _reconcile_locked(self) -> list[dict]:
        reports = []
        for directory in sorted(self.root.iterdir()):
            if not re.fullmatch(r"\.(pending_[0-9a-f]{32}|retired_update_[0-9]+_[0-9a-f]{32})", directory.name):
                continue
            if directory.is_symlink() or not directory.is_dir():
                continue
            marker = directory / "owner.json"
            if not marker.is_file() or marker.is_symlink():
                continue
            try:
                owner = json.loads(marker.read_text())
            except (ValueError, OSError):
                continue
            if owner.get("schema") != "nc_rted_checkpoint_owner_v1" or owner.get("identity") != self.identity:
                continue
            files = list(directory.iterdir())
            if any(f.name not in {"owner.json", "manifest.json", "state.pt"} or f.is_symlink() or not f.is_file() for f in files):
                continue
            record = dict(action="reclaim_interrupted_checkpoint", directory=directory.name,
                          identity=self.identity, files={f.name: f.stat().st_size for f in files},
                          owner_sha256=_sha(marker))
            with (self.root / "reconciliation.jsonl").open("a") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            # Keep the durable ownership marker until all sizeable data is gone.
            for name in ("state.pt", "manifest.json", "owner.json"):
                (directory / name).unlink(missing_ok=True)
            directory.rmdir()
            _sync_dir(self.root)
            reports.append(record)
        return reports

    def _validate(self, directory: Path) -> dict:
        if directory.parent != self.root or directory.is_symlink():
            raise RecoveryError("checkpoint path escapes store")
        manifest = directory / "manifest.json"
        payload = directory / "state.pt"
        if manifest.is_symlink() or payload.is_symlink() or not manifest.is_file() or not payload.is_file():
            raise RecoveryError(f"incomplete checkpoint: {directory.name}")
        document = json.loads(manifest.read_text())
        if document.get("schema") != "nc_rted_checkpoint_v2" or document.get("identity") != self.identity:
            raise RecoveryError("checkpoint identity/schema mismatch")
        if (directory.name == "final") != bool(document.get("final")):
            raise RecoveryError("checkpoint directory/final marker mismatch")
        if directory.name.startswith("update_") and int(directory.name[7:]) != document.get("completed_updates"):
            raise RecoveryError("checkpoint directory/update counter mismatch")
        if document.get("payload_bytes") != payload.stat().st_size or document.get("payload_sha256") != _sha(payload):
            raise RecoveryError("checkpoint payload corruption")
        return document

    def committed(self) -> list[Path]:
        result = []
        for directory in self.root.iterdir():
            if directory.name.startswith("update_") and directory.name[7:].isdigit():
                self._validate(directory)
                result.append(directory)
        return sorted(result, key=lambda p: int(p.name[7:]))

    def save(self, trainer: IncrementalTrainer, *, final: bool = False) -> Path:
        with self._lock():
            self._reconcile_locked()
            return self._save_locked(trainer, final=final)

    def save_or_verify(self, trainer: IncrementalTrainer, *, final: bool = False) -> Path:
        """Publish a boundary once, or prove an immutable replay is identical.

        Bundled recovery can replay an update that a process published just
        before it died, but before the bundle-wide commit became durable.  That
        checkpoint remains evidence and must never be replaced.  The replay is
        allowed to reuse it only when every state component, including RNG,
        equals the state it would publish now.
        """
        with self._lock():
            self._reconcile_locked()
            target = self.root / ("final" if final else f"update_{trainer.completed_updates:06d}")
            if not target.exists():
                return self._save_locked(trainer, final=final)
            manifest = self._validate(target)
            state = validate_checkpoint_payload(target / "state.pt", manifest)
            expected = dict(
                trainable={name: parameter.detach().cpu() for name, parameter in trainer.model.named_parameters()
                           if parameter.requires_grad},
                optimizer_master={name: parameter.detach().cpu() for name, parameter in trainer.master_parameters.items()},
                optimizer=trainer.optimizer.state_dict(), scheduler=trainer.scheduler.state_dict(),
                training=trainer.metadata(), rng=capture_rng())
            if not _same_checkpoint_value(state, expected):
                raise RecoveryError("immutable checkpoint differs from replay state")
            return target

    def _save_locked(self, trainer: IncrementalTrainer, *, final: bool) -> Path:
        if str(trainer.seed) != self.identity["seed"]:
            raise RecoveryError("seed does not match run identity")
        update = trainer.completed_updates
        if update < 1 or final != (update == trainer.recipe.updates):
            raise RecoveryError("invalid checkpoint completion state")
        if not final and update % trainer.recipe.save_interval:
            raise RecoveryError("checkpoint must be at a configured recovery boundary")
        if (not trainer.at_update_boundary or any(p.grad is not None for p in trainer.model.parameters()) or
                any(p.grad is not None for p in trainer.master_parameters.values())):
            raise RecoveryError("partial accumulation/uncleared gradients cannot be checkpointed")
        target = self.root / ("final" if final else f"update_{update:06d}")
        if target.exists():
            raise RecoveryError("refusing to overwrite an immutable committed checkpoint")
        parameters = {n: p.detach().cpu().clone() for n, p in trainer.model.named_parameters() if p.requires_grad}
        masters = {n: p.detach().cpu().clone() for n, p in trainer.master_parameters.items()}
        state = dict(trainable=parameters, optimizer_master=masters, optimizer=trainer.optimizer.state_dict(),
                     scheduler=trainer.scheduler.state_dict(), training=trainer.metadata(), rng=capture_rng())
        # Conservative bound includes optimizer states and serialization overhead.
        def tensor_bytes(value):
            if isinstance(value, torch.Tensor):
                return value.numel() * value.element_size()
            if isinstance(value, dict):
                return sum(tensor_bytes(v) for v in value.values())
            if isinstance(value, (list, tuple)):
                return sum(tensor_bytes(v) for v in value)
            return 0
        required = tensor_bytes(state) + (16 << 20)
        if shutil.disk_usage(self.root).free < self.min_free_bytes + required:
            raise RecoveryError("disk reserve would be breached by checkpoint publication")
        temporary = self.root / f".pending_{uuid.uuid4().hex}"
        temporary.mkdir()
        _write_json(temporary / "owner.json", dict(schema="nc_rted_checkpoint_owner_v1", identity=self.identity))
        _sync_dir(temporary)
        payload = temporary / "state.pt"
        with payload.open("xb") as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        manifest = dict(schema="nc_rted_checkpoint_v2", identity=self.identity, final=final,
                        completed_updates=update, cursor=trainer.cursor,
                        order_sha256=trainer.metadata()["order_sha256"],
                        payload_bytes=payload.stat().st_size, payload_sha256=_sha(payload))
        _write_json(temporary / "manifest.json", manifest)
        _sync_dir(temporary)
        os.replace(temporary, target)
        _sync_dir(self.root)
        # latest is advisory; recovery can reconcile publication before this step.
        pointer = self.root / f".latest_{uuid.uuid4().hex}.json"
        _write_json(pointer, dict(directory=target.name, manifest_sha256=_sha(target / "manifest.json")))
        os.replace(pointer, self.root / "latest.json")
        _sync_dir(self.root)
        self._prune_recovery()
        return target

    def _prune_recovery(self) -> None:
        for directory in self.committed()[:-2]:
            # Only this store's verified two-file recovery directories can rotate.
            if {p.name for p in directory.iterdir()} != {"state.pt", "manifest.json", "owner.json"}:
                raise RecoveryError("unexpected file in recovery directory; refusing deletion")
            retired = self.root / f".retired_{directory.name}_{uuid.uuid4().hex}"
            os.replace(directory, retired)
            _sync_dir(self.root)
            (retired / "state.pt").unlink()
            (retired / "manifest.json").unlink()
            (retired / "owner.json").unlink()
            retired.rmdir()
        _sync_dir(self.root)

    def latest(self) -> Path | None:
        final = self.root / "final"
        if final.exists():
            self._validate(final)
            return final
        committed = self.committed()
        return committed[-1] if committed else None

    def restore(self, trainer: IncrementalTrainer, directory: Path | None = None) -> dict:
        with self._lock():
            return self._restore_locked(trainer, directory)

    def _restore_locked(self, trainer: IncrementalTrainer, directory: Path | None) -> dict:
        state, manifest = self._prepare_restore_locked(trainer, directory)
        self._apply_prepared_restore(trainer, state)
        return manifest

    def prepare_restore(self, trainer: IncrementalTrainer, directory: Path | None = None) -> tuple[dict, dict]:
        """Validate a restore without changing a trainer or process RNG."""
        with self._lock():
            return self._prepare_restore_locked(trainer, directory)

    def _prepare_restore_locked(self, trainer: IncrementalTrainer, directory: Path | None) -> tuple[dict, dict]:
        path = self.latest() if directory is None else Path(directory).absolute()
        if path is None:
            raise RecoveryError("no committed checkpoint")
        manifest = self._validate(path)
        state = validate_checkpoint_payload(path / "state.pt", manifest)
        info = state["training"]
        if (str(trainer.seed) != self.identity["seed"] or info["seed"] != trainer.seed or
                info["recipe"] != asdict(trainer.recipe) or info["order"] != trainer.order):
            raise RecoveryError("recipe, seed, or complete sample order changed")
        update = info["completed_updates"]
        if type(update) is not int or not 0 < update <= trainer.recipe.updates or info["cursor"] != update * trainer.recipe.accumulation:
            raise RecoveryError("invalid update/cursor state")
        if (manifest["completed_updates"] != update or manifest["cursor"] != info["cursor"] or
                manifest["order_sha256"] != trainer.metadata()["order_sha256"] or
                manifest["final"] != (update == trainer.recipe.updates)):
            raise RecoveryError("manifest and optimizer-boundary metadata disagree")
        parameters = {n: p for n, p in trainer.model.named_parameters() if p.requires_grad}
        saved = state["trainable"]
        masters = state["optimizer_master"]
        if set(parameters) != set(saved) or set(masters) != set(parameters):
            raise RecoveryError("trainable parameter keys changed")
        for name, parameter in parameters.items():
            value = saved[name]
            if parameter.shape != value.shape or parameter.dtype != value.dtype or not bool(torch.isfinite(value).all()):
                raise RecoveryError(f"saved trainable tensor invalid: {name}")
            master = masters[name]
            if (master.shape != value.shape or master.dtype != torch.float32 or
                    not bool(torch.isfinite(master).all()) or not torch.equal(master.to(value.dtype), value)):
                raise RecoveryError(f"saved FP32 master is inconsistent: {name}")
        optimizer = state["optimizer"]
        live_groups = trainer.optimizer.state_dict()["param_groups"]
        if len(optimizer["param_groups"]) != len(live_groups):
            raise RecoveryError("optimizer group count differs")
        for saved_group, live in zip(optimizer["param_groups"], live_groups):
            for key in ("params", "name", "parameter_names", "initial_lr", "weight_decay", "betas", "eps", "amsgrad"):
                if saved_group.get(key) != live.get(key):
                    raise RecoveryError(f"optimizer parameter-group contract changed: {key}")
        index_to_name = {}
        for group in optimizer["param_groups"]:
            index_to_name.update(zip(group["params"], group["parameter_names"]))
        if len(index_to_name) != len(parameters) or not set(optimizer["state"]) <= set(index_to_name):
            raise RecoveryError("invalid optimizer state ownership")
        for index, entry in optimizer["state"].items():
            if set(entry) != {"step", "exp_avg", "exp_avg_sq"}:
                raise RecoveryError("missing or unexpected Adam state fields")
            expected = masters[index_to_name[index]]
            for key in ("exp_avg", "exp_avg_sq"):
                value = entry[key]
                if (not isinstance(value, torch.Tensor) or value.shape != expected.shape or
                        value.dtype != torch.float32 or not bool(torch.isfinite(value).all())):
                    raise RecoveryError("invalid optimizer moment shape/dtype/value")
            step = entry["step"]
            if (not isinstance(step, torch.Tensor) or step.numel() != 1 or
                    not bool(torch.isfinite(step)) or not 1 <= float(step) <= update or
                    float(step) != int(float(step))):
                raise RecoveryError("invalid Adam step counter")
        if state["scheduler"]["last_epoch"] != update:
            raise RecoveryError("scheduler update differs")
        cuda_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if len(state["rng"]["cuda"]) != cuda_count:
            raise RecoveryError("CUDA RNG topology differs")
        try:
            for index, saved_cuda in enumerate(state["rng"]["cuda"]):
                torch.Generator(device=f"cuda:{index}").set_state(saved_cuda)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise RecoveryError("CUDA RNG state is not restorable") from exc
        return state, manifest

    @staticmethod
    def _apply_prepared_restore(trainer: IncrementalTrainer, state: dict) -> None:
        """Apply a state previously accepted by :meth:`prepare_restore`."""
        parameters = {n: p for n, p in trainer.model.named_parameters() if p.requires_grad}
        saved = state["trainable"]
        masters = state["optimizer_master"]
        update = state["training"]["completed_updates"]
        # All key/shape/identity checks precede parameter mutation.
        with torch.no_grad():
            for name, parameter in parameters.items():
                parameter.copy_(saved[name].to(parameter.device))
                trainer.master_parameters[name].copy_(masters[name].to(parameter.device))
        trainer.optimizer.load_state_dict(state["optimizer"])
        trainer.scheduler.load_state_dict(state["scheduler"])
        trainer.completed_updates = update
        trainer.model.zero_grad(set_to_none=True)
        trainer.optimizer.zero_grad(set_to_none=True)
        restore_rng(state["rng"])
        trainer.at_update_boundary = True


def _same_checkpoint_value(left, right) -> bool:
    """Exact recursive comparison without device-dependent tensor identity."""
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return left.shape == right.shape and left.dtype == right.dtype and torch.equal(left.cpu(), right.detach().cpu())
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_same_checkpoint_value(left[key], right[key]) for key in left)
    if isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
        return len(left) == len(right) and all(_same_checkpoint_value(a, b) for a, b in zip(left, right))
    return left == right
