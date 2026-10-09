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
        path = self.latest() if directory is None else Path(directory).absolute()
        if path is None:
            raise RecoveryError("no committed checkpoint")
        manifest = self._validate(path)
        state = torch.load(path / "state.pt", map_location="cpu", weights_only=True)
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
        return manifest
