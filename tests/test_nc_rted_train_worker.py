from types import SimpleNamespace

import pytest

from nc_rted.task_inputs import TaskInputError, TrainingCatalog, TrainingTask
from nc_rted.train_worker import TeacherIndex, TrainingWorker
from nc_rted.training import Recipe


def catalog():
    return TrainingCatalog([TrainingTask("a", "detection", "ucf-crime", "family", "media", 8., 0, None),
                            TrainingTask("b", "detection", "ucf-crime", "other", "media2", 8., 1, None)], "d" * 64)


def teachers():
    return {"schema": "nc_rted_teacher_pipeline/v1", "rows": [
        dict(window_id="a", dataset="ucf-crime", aux_valid=True, F_quality=.2, S_quality=.2, U_quality=.8),
        dict(window_id="b", dataset="ucf-crime", aux_valid=True, F_quality=.8, S_quality=.8, U_quality=.2)]}


def test_teacher_index_requires_full_denominator_and_histogram():
    assert len(TeacherIndex(teachers(), catalog()).rows) == 2
    missing = teachers(); missing["rows"].pop()
    with pytest.raises(TaskInputError, match="every fixed detection"):
        TeacherIndex(missing, catalog())
    mismatch = teachers(); mismatch["rows"][0]["U_quality"] = .3
    with pytest.raises(TaskInputError, match="histogram"):
        TeacherIndex(mismatch, catalog())
    mismatch = teachers(); mismatch["rows"][0]["S_quality"] = .3
    with pytest.raises(TaskInputError, match="S/F"):
        TeacherIndex(mismatch, catalog())


def test_diagnostic_checkpoint_cannot_be_promoted_to_formal():
    worker = object.__new__(TrainingWorker)
    worker.store = SimpleNamespace(identity={"run_id": "diagnostic:probe"})
    with pytest.raises(TaskInputError, match="separate checkpoint"):
        worker.run(progress_path="unused")
    worker.store = SimpleNamespace(identity={"run_id": "F_seed17"})
    with pytest.raises(TaskInputError, match="separate checkpoint"):
        worker.run(progress_path="unused", diagnostic_updates=2)


def test_missing_formal_acceptance_blocks_before_reading_provider():
    worker = object.__new__(TrainingWorker)
    worker.store = SimpleNamespace(identity={"run_id": "F_seed17"})
    worker.trainer = SimpleNamespace(recipe=Recipe())
    worker.seed = 17
    worker.catalog = SimpleNamespace(tasks=[None] * 8000)
    with pytest.raises(TaskInputError, match="accepted frozen"):
        worker.run(progress_path="unused")
