"""Same-seed A/U/S/F coordinator over independent trainer bundles."""
from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
import copy
import hashlib
import json
import os
from pathlib import Path
import uuid
from typing import Callable, Mapping, Any
import torch

from .recovery import CheckpointStore, RecoveryError, capture_rng, restore_rng


@dataclass
class Bundle:
    group: str
    trainer: Any
    loss_for_material: Callable[[str, Any], Any]
    store: Any | None = None
    progress: Callable[[dict], None] | None = None
    rng: dict | None = None


class BundleCheckpointStore:
    """Durable all-group checkpoint boundaries over immutable group stores.

    Per-group checkpoints may become visible while a process is dying.  They
    deliberately remain immutable forensic evidence; only this small manifest
    makes a boundary recoverable.  Therefore a restore never mixes the latest
    file from each group.
    """
    schema = "nc_rted_bundle_checkpoint_v1"

    def __init__(self, root: str | Path, bundles: Mapping[str, Bundle]):
        self.root = Path(root).absolute()
        self.root.mkdir(parents=True, exist_ok=True)
        self.commits = self.root / "commits"
        self.commits.mkdir(exist_ok=True)
        self._bundles = dict(bundles)
        identities = {group: dict(bundle.store.identity) for group, bundle in bundles.items()
                      if bundle.store is not None}
        if set(identities) != {"A", "U", "S", "F"}:
            raise ValueError("bundle commits require four checkpoint stores")
        shared_keys = {"seed", "code_sha256", "data_sha256", "teacher_sha256",
                       "inherited_weights_sha256", "runtime_sha256"}
        reference = {key: identities["A"][key] for key in shared_keys}
        if any({key: identity[key] for key in shared_keys} != reference for identity in identities.values()):
            raise ValueError("bundle scientific input identities differ")
        # Each group preserves its own run ID and configuration SHA.  The
        # bundle binds the four complete identities rather than pretending
        # their storage labels are scientifically shared.
        self.scientific_identity = reference
        self.member_identities = identities

    @staticmethod
    def _name(update: int, final: bool) -> str:
        return "final.json" if final else f"update_{update:06d}.json"

    def _path(self, update: int, final: bool) -> Path:
        return self.commits / self._name(update, final)

    def commit(self, update: int, *, final: bool) -> Path:
        records = {}
        for group in ("A", "U", "S", "F"):
            store = self._bundles[group].store
            assert store is not None
            name = "final" if final else f"update_{update:06d}"
            path = store.root / name
            manifest = store._validate(path)
            if manifest["completed_updates"] != update or bool(manifest["final"]) != final:
                raise RecoveryError("group checkpoint does not match bundle boundary")
            records[group] = dict(directory=name, manifest_sha256=_sha256(path / "manifest.json"))
        document = dict(schema=self.schema, scientific_identity=self.scientific_identity,
                        members=self.member_identities, completed_updates=update, final=final,
                        checkpoints=records)
        target = self._path(update, final)
        if target.exists():
            if json.loads(target.read_text()) != document:
                raise RecoveryError("immutable bundle checkpoint differs")
            return target
        temporary = self.commits / f".pending_{uuid.uuid4().hex}"
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(document, stream, sort_keys=True, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        _sync_directory(self.commits)
        return target

    def latest(self) -> dict | None:
        candidates = []
        for path in self.commits.glob("*.json"):
            try:
                document = json.loads(path.read_text())
            except (OSError, ValueError) as error:
                raise RecoveryError("bundle checkpoint manifest is unreadable") from error
            if (document.get("schema") != self.schema or
                    document.get("scientific_identity") != self.scientific_identity or
                    document.get("members") != self.member_identities):
                raise RecoveryError("bundle checkpoint identity/schema mismatch")
            update = document.get("completed_updates")
            final = document.get("final")
            if type(update) is not int or update < 1 or type(final) is not bool or path.name != self._name(update, final):
                raise RecoveryError("bundle checkpoint boundary is invalid")
            checkpoints = document.get("checkpoints")
            if not isinstance(checkpoints, dict) or set(checkpoints) != {"A", "U", "S", "F"}:
                raise RecoveryError("bundle checkpoint group set is invalid")
            candidates.append(document)
        return max(candidates, key=lambda item: item["completed_updates"]) if candidates else None

    def prepare_restore(self) -> dict | None:
        """Prevalidate all group payloads before any trainer/RNG can change."""
        boundary = self.latest()
        if boundary is None:
            return None
        prepared = {}
        for group in ("A", "U", "S", "F"):
            record = boundary["checkpoints"][group]
            if not isinstance(record, dict) or set(record) != {"directory", "manifest_sha256"}:
                raise RecoveryError("bundle checkpoint record is invalid")
            store = self._bundles[group].store
            assert store is not None
            path = store.root / record["directory"]
            if _sha256(path / "manifest.json") != record["manifest_sha256"]:
                raise RecoveryError("bundle checkpoint manifest digest differs")
            state, manifest = store.prepare_restore(self._bundles[group].trainer, path)
            if manifest["completed_updates"] != boundary["completed_updates"]:
                raise RecoveryError("bundle checkpoint update differs")
            prepared[group] = (state, manifest)
        return prepared


def clone_with_shared_frozen(baseline, frozen_parameters):
    """Copy trainables while preserving read-only parameter object identity."""
    frozen_parameters = tuple(frozen_parameters)
    if any(parameter.requires_grad for parameter in frozen_parameters):
        raise ValueError("shared frozen parameters must not require gradients")
    clone = copy.deepcopy(baseline, memo={id(parameter): parameter for parameter in frozen_parameters})
    source = {name: parameter for name, parameter in baseline.named_parameters() if parameter.requires_grad}
    copied = {name: parameter for name, parameter in clone.named_parameters() if parameter.requires_grad}
    if source.keys() != copied.keys() or any(_storage_key(source[name]) == _storage_key(copied[name]) for name in source):
        raise ValueError("cloned trainable parameters share source storage")
    return clone


class SameSeedBundleWorker:
    def __init__(self, bundles: Mapping[str, Bundle], frozen_provider: Callable[[str], Any], *,
                 bundle_checkpoint_root: str | Path | None = None,
                 after_checkpoint_publication: Callable[[str, int, bool], None] | None = None):
        if set(bundles) != {"A", "U", "S", "F"}: raise ValueError("bundles must be A/U/S/F")
        self.bundles, self.frozen_provider = dict(bundles), frozen_provider
        reference = self.bundles["A"].trainer
        if any(bundle.trainer.seed != reference.seed or bundle.trainer.order != reference.order or
               bundle.trainer.recipe != reference.recipe for bundle in self.bundles.values()):
            raise ValueError("same-seed bundles require identical seed, order, and recipe")
        for bundle in self.bundles.values():
            bundle.rng = capture_rng() if bundle.rng is None else bundle.rng
        if len({id(bundle.trainer) for bundle in self.bundles.values()}) != 4:
            raise ValueError("each bundle requires an independent trainer")
        if len({id(bundle.trainer.optimizer) for bundle in self.bundles.values()}) != 4:
            raise ValueError("each bundle requires an independent optimizer")
        if any(bundle.store is None for bundle in self.bundles.values()) and any(bundle.store is not None for bundle in self.bundles.values()):
            raise ValueError("bundles either all use checkpoint stores or none do")
        seen=set(); storage=set()
        for group,bundle in self.bundles.items():
            if bundle.group != group: raise ValueError("bundle group key mismatch")
            owned=[*bundle.trainer.forward_parameters.values(),*bundle.trainer.master_parameters.values()]
            owned.extend(parameter for state in bundle.trainer.optimizer.state.values() for parameter in state.values() if isinstance(parameter,torch.Tensor))
            identities={id(parameter) for parameter in owned}
            if seen & identities: raise ValueError("bundle trainable or optimizer storage overlaps")
            seen.update(identities)
            locations = {_storage_key(parameter) for parameter in owned}
            if storage & locations: raise ValueError("bundle trainable or optimizer tensor storage overlaps")
            storage.update(locations)
            if bundle.store is not None and bundle.store.identity["group"] != group:
                raise ValueError("checkpoint store group mismatch")
            reference = self.bundles["A"].trainer.forward_parameters
            if bundle.trainer.forward_parameters.keys() != reference.keys() or any(
                    not torch.equal(bundle.trainer.forward_parameters[name].detach().cpu(), parameter.detach().cpu())
                    for name, parameter in reference.items()):
                raise ValueError("same-seed bundles need equal initial trainable values")
        self.after_checkpoint_publication = after_checkpoint_publication
        self.bundle_store = (BundleCheckpointStore(bundle_checkpoint_root, self.bundles)
                             if bundle_checkpoint_root is not None else None)
        if any(bundle.store is not None for bundle in self.bundles.values()) and self.bundle_store is None:
            raise ValueError("checkpointed bundles require a common bundle checkpoint root")

    @staticmethod
    def _prepared(value):
        """Return an exact immutable structural fingerprint of prepared data."""
        if isinstance(value,torch.Tensor):
            if value.requires_grad or value.grad_fn is not None: raise ValueError("shared prepared material must be detached")
            return ("tensor", id(value), value._version, str(value.dtype), tuple(value.shape), str(value.device))
        if is_dataclass(value):
            return ("dataclass", type(value).__module__, type(value).__qualname__,
                    tuple((field.name, SameSeedBundleWorker._prepared(getattr(value, field.name))) for field in fields(value)))
        if isinstance(value,Mapping):
            return ("mapping", type(value).__module__, type(value).__qualname__,
                    tuple((SameSeedBundleWorker._prepared(key), SameSeedBundleWorker._prepared(item)) for key, item in value.items()))
        if isinstance(value,(tuple,list)):
            return (type(value).__name__, tuple(SameSeedBundleWorker._prepared(item) for item in value))
        if value is None or isinstance(value, (bool, int, float, str, bytes)):
            return (type(value).__name__, value)
        if hasattr(value, "__dict__"):
            return ("object", type(value).__module__, type(value).__qualname__,
                    SameSeedBundleWorker._prepared(vars(value)))
        raise ValueError(f"unsupported shared prepared material: {type(value)!r}")

    def restore(self) -> None:
        """Restore every bundle from one common committed update boundary."""
        if self.bundle_store is None: raise ValueError("all bundles require a common bundle checkpoint store for replay")
        prepared = self.bundle_store.prepare_restore()
        if prepared is None:
            if all(bundle.trainer.completed_updates == 0 for bundle in self.bundles.values()): return
            raise RuntimeError("no committed bundle checkpoint boundary")
        for group in ("A", "U", "S", "F"):
            bundle = self.bundles[group]
            state, _ = prepared[group]
            CheckpointStore._apply_prepared_restore(bundle.trainer, state)
            bundle.rng = copy.deepcopy(state["rng"])

    def run(self, *, stop_after: int | None = None) -> None:
        target = self.bundles["A"].trainer.recipe.updates if stop_after is None else stop_after
        while self.bundles["A"].trainer.completed_updates < target:
            cursor = self.bundles["A"].trainer.cursor
            order = self.bundles["A"].trainer.order[cursor:cursor + self.bundles["A"].trainer.recipe.accumulation]
            materials = {sample_id: self.frozen_provider(sample_id) for sample_id in order}
            prepared = {sample_id: self._prepared(material) for sample_id, material in materials.items()}
            reports = {}
            for group in ("A", "U", "S", "F"):
                bundle = self.bundles[group]; trainer = bundle.trainer
                if trainer.cursor != cursor: raise RuntimeError("bundle cursor diverged")
                restore_rng(bundle.rng)
                report = trainer.step(lambda sample_id, b=bundle: b.loss_for_material(sample_id, materials[sample_id]))
                bundle.rng = capture_rng()
                if any(self._prepared(materials[sample_id]) != fingerprint
                       for sample_id, fingerprint in prepared.items()):
                    raise RuntimeError("a bundle mutated shared prepared material")
                reports[group] = report
                if bundle.store and (trainer.completed_updates % trainer.recipe.save_interval == 0 or trainer.completed_updates == trainer.recipe.updates):
                    final = trainer.completed_updates == trainer.recipe.updates
                    bundle.store.save_or_verify(trainer, final=final)
                    if self.after_checkpoint_publication: self.after_checkpoint_publication(group, trainer.completed_updates, final)
            if self.bundle_store and (self.bundles["A"].trainer.completed_updates % self.bundles["A"].trainer.recipe.save_interval == 0 or self.bundles["A"].trainer.completed_updates == self.bundles["A"].trainer.recipe.updates):
                update = self.bundles["A"].trainer.completed_updates
                self.bundle_store.commit(update, final=update == self.bundles["A"].trainer.recipe.updates)
            for group, report in reports.items():
                if self.bundles[group].progress: self.bundles[group].progress(report)
            # Release the eight-sample working set before the provider starts
            # constructing the next update's inputs.
            materials = None
            prepared = None


def _storage_key(tensor: torch.Tensor) -> tuple[str, int]:
    return (str(tensor.device), tensor.untyped_storage().data_ptr())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
