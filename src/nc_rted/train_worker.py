"""Incremental worker orchestration; scheduling and media decoding stay separate.

The provider must implement the pinned inherited media paths and provide only
frozen observations/memory. This worker owns labels, teacher alignment, updates,
progress and recovery. Merely constructing it does not admit a formal job.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Protocol

import torch

from .bridge import EvidenceSlowBridge, ObservationBatch
from .features import CELL_FEATURE_DIM
from .loading import load_inherited_slow
from .model import RelationTimeEvidence
from .recovery import CheckpointStore
from .task_inputs import (FrozenTaskContext, InheritedTaskTokenizer, TaskInputError,
                          TrainingCatalog, TrainingTask, sha256_file, teacher_batch,
                          validate_observation_scope)
from .training import IncrementalTrainer, Recipe, publish_progress, seed_run


@dataclass(frozen=True)
class SampleMaterial:
    context: FrozenTaskContext
    observations: ObservationBatch
    relation_ids: tuple[str, ...] = ()
    detection_question: str | None = None
    detection_scoring: str = "yesno"


class FrozenSampleProvider(Protocol):
    """No teacher/answers are passed into the public frozen-cache producer."""
    def __call__(self, sample_id: str) -> SampleMaterial: ...


def build_training_bridge(base_directory: str | Path, export_directory: str | Path,
                          *, export_hashes: dict[str, str], seed: int, device: str):
    slow, load_report = load_inherited_slow(base_directory, export_directory,
                                           train_lora=True, expected_hashes=export_hashes, device=device)
    # Reset after pretrained loading as that loader may consume random numbers.
    # All groups with the same seed therefore start with identical new parameters.
    seed_run(seed)
    raw = slow.get_base_model()
    evidence = RelationTimeEvidence(CELL_FEATURE_DIM, raw.config.hidden_size)
    bridge = EvidenceSlowBridge(slow, evidence)
    return bridge, load_report


class TeacherIndex:
    def __init__(self, document: dict, catalog: TrainingCatalog, *, identity: str | None = None):
        self.identity = identity
        if document.get("schema") != "nc_rted_teacher_pipeline/v1":
            raise TaskInputError("unsupported teacher manifest")
        rows = document.get("rows")
        if not isinstance(rows, list):
            raise TaskInputError("missing teacher rows")
        self.rows = {row["window_id"]: row for row in rows}
        expected = {key for key, task in catalog.tasks.items() if task.task == "detection"}
        if len(self.rows) != len(rows) or set(self.rows) != expected:
            raise TaskInputError("teacher must cover every fixed detection window exactly once, including rejections")
        for key, row in self.rows.items():
            if row.get("dataset") != catalog.tasks[key].dataset or type(row.get("aux_valid")) is not bool:
                raise TaskInputError("teacher dataset/support metadata mismatch")
        # Histograms use Python doubles from the same frozen artifact. These are
        # teacher-only checks; no deployment labels or reference IDs reach Slow.
        for dataset in {task.dataset for task in catalog.tasks.values()}:
            eligible = [row for row in rows if row["dataset"] == dataset and row["aux_valid"]]
            if sorted(row["U_quality"] for row in eligible) != sorted(row["F_quality"] for row in eligible):
                raise TaskInputError("U/F quality histogram differs")
            if any(row["S_quality"] != row["F_quality"] for row in eligible):
                raise TaskInputError("S/F quality differs")

    @classmethod
    def load(cls, path: str | Path, catalog: TrainingCatalog, *, expected_sha256: str):
        if sha256_file(path) != expected_sha256:
            raise TaskInputError("teacher artifact identity changed")
        return cls(json.loads(Path(path).read_text()), catalog, identity=expected_sha256)


class TrainingWorker:
    def __init__(self, bridge: EvidenceSlowBridge, catalog: TrainingCatalog, teachers: TeacherIndex,
                 tokenizer: InheritedTaskTokenizer, provider: FrozenSampleProvider,
                 store: CheckpointStore, *, group: str, seed: int, recipe: Recipe = Recipe()):
        if group not in {"A", "U", "S", "F"} or str(seed) != store.identity["seed"] or group != store.identity["group"]:
            raise TaskInputError("worker and checkpoint run identity differ")
        if catalog.identity != store.identity["data_sha256"]:
            raise TaskInputError("worker catalog and checkpoint identity differ")
        if teachers.identity != store.identity["teacher_sha256"]:
            raise TaskInputError("worker teacher and checkpoint identity differ")
        self.bridge, self.catalog, self.teachers = bridge, catalog, teachers
        self.tokenizer, self.provider, self.store = tokenizer, provider, store
        self.group, self.seed = group, seed
        self.trainer = IncrementalTrainer(bridge, list(catalog.tasks), seed, recipe)

    def loss_for_sample(self, sample_id: str) -> torch.Tensor:
        return self.loss_for_material(sample_id, self.material_for_sample(sample_id))

    def material_for_sample(self, sample_id: str) -> SampleMaterial:
        """Obtain one frozen provider result for all same-seed consumers."""
        task: TrainingTask = self.catalog.tasks[sample_id]
        material = self.provider(sample_id)
        validate_observation_scope(task.task, material.context, material.observations)
        return material

    def loss_for_material(self, sample_id: str, material: SampleMaterial) -> torch.Tensor:
        """Consume a prevalidated frozen material without recalling the provider."""
        task: TrainingTask = self.catalog.tasks[sample_id]
        validate_observation_scope(task.task, material.context, material.observations)
        inputs = self.tokenizer.encode(task, material.context, detection_question=material.detection_question,
                                       detection_scoring=material.detection_scoring)
        teacher = None
        if task.task == "detection":
            teacher = teacher_batch(self.teachers.rows[sample_id], sample_id=sample_id, group=self.group,
                                    relation_ids=material.relation_ids, observations=material.observations)
        # Teachers are never passed into caption forward. A sees identical public
        # inputs; its auxiliary loss is zero in the shared bridge implementation.
        return self.bridge(inputs, material.observations, task=task.task, group=self.group, teacher=teacher).loss

    def run(self, *, progress_path: str | Path, admission: dict | None = None,
            diagnostic_updates: int | None = None) -> dict:
        """Diagnostic mode cannot mark a formal run complete; formal needs full acceptance.

        Queue/device/disk/lease admission remains a separate mandatory outer gate.
        This source gate prevents accidentally treating CPU interface checks as
        acceptance of all ten engineering requirements.
        """
        diagnostic_identity = self.store.identity["run_id"].startswith("diagnostic:")
        if diagnostic_identity != (diagnostic_updates is not None):
            raise TaskInputError("diagnostic and formal runs require separate checkpoint identities")
        if diagnostic_updates is None:
            self._verify_formal_admission(admission)
        elif type(diagnostic_updates) is not int or not 1 <= diagnostic_updates <= self.trainer.recipe.updates:
            raise TaskInputError("invalid diagnostic update count")
        if self.store.latest() is not None:
            self.store.restore(self.trainer)
        self.trainer.run(self.loss_for_sample, save=self.store.save,
                         progress=lambda report: publish_progress(progress_path, report),
                         stop_after=diagnostic_updates)
        complete = diagnostic_updates is None and self.trainer.completed_updates == 1000
        return dict(status="FORMAL_TRAINING_COMPLETE" if complete else "DIAGNOSTIC_ONLY",
                    completed_updates=self.trainer.completed_updates, cursor=self.trainer.cursor,
                    run_identity=dict(self.store.identity), formal_result=complete)

    def _verify_formal_admission(self, admission: dict | None) -> None:
        if self.trainer.recipe != Recipe() or self.seed not in {17, 42, 2026} or len(self.catalog.tasks) != 8000:
            raise TaskInputError("formal worker must retain the complete fixed recipe")
        if not admission or admission.get("status") != "PASS" or admission.get("formal_execution_allowed") is not True:
            raise TaskInputError("formal execution requires an accepted frozen implementation")
        if admission.get("run_identity") != self.store.identity:
            raise TaskInputError("formal admission is not bound to this exact run")
        checks = admission.get("engineering_checks")
        if not isinstance(checks, dict) or set(checks) != {str(i) for i in range(1, 11)} or any(value != "PASS" for value in checks.values()):
            raise TaskInputError("all ten specification acceptance checks must pass")
        files = admission.get("source_files")
        if not isinstance(files, dict) or not files:
            raise TaskInputError("formal admission has no source-file binding")
        source_digest = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if source_digest != self.store.identity["code_sha256"]:
            raise TaskInputError("source manifest and run identity differ")
        root = Path(__file__).parent
        required = {str(path.resolve()) for path in root.glob("*.py")}
        if not required.issubset(files):
            raise TaskInputError("formal admission omits NC-RTED runtime source")
        for name, digest in files.items():
            if sha256_file(name) != digest:
                raise TaskInputError(f"accepted implementation changed: {name}")
