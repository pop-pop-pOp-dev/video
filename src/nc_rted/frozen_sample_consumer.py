"""Bounded CPU-record consumption for GPU-produced training material."""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from dataclasses import replace
import time
from typing import Iterable, Mapping

import torch

from .bridge import ObservationBatch
from .frozen_sample_store import FrozenSampleStore, FrozenSampleStoreError
from .task_inputs import FrozenTaskContext, TrainingTask, validate_observation_scope
from .train_worker import SampleMaterial


class FrozenSampleConsumerError(RuntimeError):
    pass


class FrozenSampleConsumer:
    """Read only immutable producer material and move it at the update boundary.

    Prefetch has one CPU-only worker and a fixed number of in-memory futures.
    It cannot invoke an inherited media provider, so a producer lag is surfaced
    as a bounded failure instead of silently changing the training computation.
    """

    def __init__(self, tasks: Mapping[str, TrainingTask], *, root: str, identity: dict,
                 device: str | torch.device, wait_seconds: float = 120.0,
                 poll_seconds: float = 0.25, prefetch_depth: int = 8):
        if not isinstance(tasks, Mapping) or not tasks:
            raise FrozenSampleConsumerError("consumer requires a nonempty task mapping")
        if (not isinstance(wait_seconds, (int, float)) or wait_seconds <= 0 or
                not isinstance(poll_seconds, (int, float)) or poll_seconds <= 0 or
                type(prefetch_depth) is not int or not 1 <= prefetch_depth <= 8):
            raise FrozenSampleConsumerError("consumer bounds are invalid")
        self.tasks = dict(tasks)
        self.store = FrozenSampleStore(root, identity=identity)
        self.device = torch.device(device)
        self.wait_seconds, self.poll_seconds, self.prefetch_depth = float(wait_seconds), float(poll_seconds), prefetch_depth
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="frozen-sample-prefetch")
        self._pending: dict[str, Future[SampleMaterial]] = {}

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)

    def prefetch(self, sample_ids: Iterable[str]) -> None:
        for sample_id in sample_ids:
            if len(self._pending) >= self.prefetch_depth:
                return
            if sample_id not in self.tasks:
                raise FrozenSampleConsumerError("prefetch names a task outside the frozen catalog")
            if sample_id not in self._pending:
                self._pending[sample_id] = self._executor.submit(self._wait_for_cpu, self.tasks[sample_id])

    def _wait_for_cpu(self, task: TrainingTask) -> SampleMaterial:
        deadline = time.monotonic() + self.wait_seconds
        while True:
            try:
                material = self.store.get(task)
            except FrozenSampleStoreError as error:
                raise FrozenSampleConsumerError(f"frozen producer record is invalid for {task.sample_id}") from error
            if material is not None:
                return material
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise FrozenSampleConsumerError(f"timed out waiting for frozen producer record: {task.sample_id}")
            time.sleep(min(self.poll_seconds, remaining))

    @staticmethod
    def _cpu_material(task: TrainingTask, material: SampleMaterial) -> None:
        if not isinstance(material, SampleMaterial):
            raise FrozenSampleConsumerError("frozen store returned another material type")
        tensors = (material.context.visual_embeddings, *material.context.images,
                   material.observations.features, material.observations.valid, material.observations.observed_times)
        if any(not isinstance(value, torch.Tensor) or value.device.type != "cpu" or value.requires_grad for value in tensors):
            raise FrozenSampleConsumerError("frozen store material must remain detached on CPU")
        validate_observation_scope(task.task, material.context, material.observations)

    def _to_device(self, task: TrainingTask, material: SampleMaterial) -> SampleMaterial:
        self._cpu_material(task, material)
        move = lambda value: value.to(self.device, non_blocking=True)
        context = FrozenTaskContext(move(material.context.visual_embeddings), [move(image) for image in material.context.images],
                                    material.context.image_sizes, material.context.observed_seconds,
                                    material.context.sampled_frame_times, material.context.time_message)
        observations = ObservationBatch(move(material.observations.features), move(material.observations.valid),
                                        move(material.observations.observed_times))
        result = replace(material, context=context, observations=observations)
        tensors = (result.context.visual_embeddings, *result.context.images, result.observations.features,
                   result.observations.valid, result.observations.observed_times)
        if any(value.device != self.device for value in tensors):
            raise FrozenSampleConsumerError("frozen material device transfer differs from the admitted device")
        validate_observation_scope(task.task, result.context, result.observations)
        return result

    def __call__(self, sample_id: str) -> SampleMaterial:
        task = self.tasks.get(sample_id)
        if task is None:
            raise FrozenSampleConsumerError("consumer names a task outside the frozen catalog")
        future = self._pending.pop(sample_id, None)
        if future is None:
            future = self._executor.submit(self._wait_for_cpu, task)
        try:
            material = future.result(timeout=self.wait_seconds)
        except TimeoutError as error:
            raise FrozenSampleConsumerError(f"timed out waiting for frozen producer record: {sample_id}") from error
        return self._to_device(task, material)
