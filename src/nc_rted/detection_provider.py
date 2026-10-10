"""Causal replay of the original PG-driven detection memory for training prefixes.

The original evaluator inserts PG scores into its pool (the nearby comment
mentions fused scores, but the executed argument is PG). Therefore this memory
can be replayed from frozen observations without running or caching Slow.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from types import SimpleNamespace
from typing import Callable, Iterable, Protocol, TYPE_CHECKING

import torch
from torch import Tensor

from .batches import pack_observation_blocks
from .inherited_memory import detection_memory_from_stream
from .task_inputs import FrozenTaskContext, TaskInputError, TrainingCatalog
if TYPE_CHECKING:
    from .train_worker import SampleMaterial


@dataclass(frozen=True)
class DetectionProtocol:
    question_template: str
    prompt_style: str
    time_message_style: str
    memory_enhancement: bool
    rt_anomaly: bool
    trigger_threshold: float
    pool_threshold: float
    scoring: str = "yesno"

    def validate(self) -> None:
        if not self.question_template or self.prompt_style not in {"default", "skeptical", "neutral", "hivau"} or self.scoring != "yesno":
            raise TaskInputError("training prefix provider requires the frozen inherited yes/no protocol")
        if self.time_message_style not in {"short_online", "short_online_v2", "simple", "none"}:
            raise TaskInputError("unsupported inherited detection time message")
        if type(self.memory_enhancement) is not bool or type(self.rt_anomaly) is not bool:
            raise TaskInputError("memory flags must be explicit booleans")
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in (self.trigger_threshold, self.pool_threshold)):
            raise TaskInputError("invalid fixed Fast thresholds")


@dataclass(frozen=True)
class DetectionQuery:
    index: int
    frame_indices: tuple[int, ...]
    frame_times_s: tuple[float, ...]
    fast_score: float
    last_frame_patches: Tensor
    dense_patches: Tensor | None = None

    @property
    def end_seconds(self) -> float:
        return self.frame_times_s[-1]


@dataclass(frozen=True)
class DetectionPrefix:
    queries: Iterable[DetectionQuery]
    image_height: int
    image_width: int
    frame_count: int | None = None
    sample_interval: int | None = None


class FrozenPrefixReader(Protocol):
    def __call__(self, dataset: str, media_key: str, target_query_index: int) -> DetectionPrefix: ...


def original_detection_time_message(style: str, current_time: float, count: int) -> str:
    from eval_utils.vad.eval_reactvau_detection import StreamForestReasoner
    return StreamForestReasoner._generate_time_msg(SimpleNamespace(time_msg_style=style), current_time, count)


class DetectionMemoryReplay:
    def __init__(self, slow, protocol: DetectionProtocol, *, memory_factory=None,
                 time_formatter: Callable = original_detection_time_message):
        protocol.validate()
        self.slow, self.protocol, self.time_formatter = slow, protocol, time_formatter
        raw = slow.get_base_model() if hasattr(slow, "get_base_model") else slow
        parameter = next(raw.get_model().mm_projector.mlp.parameters())
        self.device, self.dtype = parameter.device, parameter.dtype
        if memory_factory is None:
            from llava.model.multimodal_projector.memory_manager import MemoryManager
            memory_factory = MemoryManager
        self.memory = memory_factory(1152, 16, st_memory_windows=[1, 12], st_memory_tokens=[729, 128],
                                     event_split_window=4, long_memory_tokens_per_frame=64,
                                     long_memory_tokens_quota=2048,
                                     anomaly_pool_max_size=8 if protocol.memory_enhancement else 0,
                                     anomaly_pool_tokens=128, anomaly_pool_protect_recent=2)
        self.times: list[float] = []
        self.last_original_index = -1
        self.failed = False

    def _validate_query(self, query: DetectionQuery, capture: bool) -> None:
        if self.failed:
            raise TaskInputError("failed memory replay must be reconstructed from its frozen prefix")
        if type(query.index) is not int or query.index != len(self.times):
            raise TaskInputError("original queries must replay exactly once from zero, in order")
        if not 1 <= len(query.frame_indices) <= 4 or len(query.frame_indices) != len(query.frame_times_s):
            raise TaskInputError("an original PG query has one to four actual sampled frames")
        if any(type(index) is not int or index < 0 for index in query.frame_indices):
            raise TaskInputError("invalid original frame indices")
        if query.frame_indices[0] <= self.last_original_index or any(b <= a for a, b in zip(query.frame_indices, query.frame_indices[1:])):
            raise TaskInputError("original query frames overlap or run backwards")
        if any(not math.isfinite(value) or value < 0 for value in query.frame_times_s) or any(
                b <= a for a, b in zip(query.frame_times_s, query.frame_times_s[1:])):
            raise TaskInputError("invalid actual original query timestamps")
        if self.times and query.frame_times_s[0] <= self.times[-1]:
            raise TaskInputError("future or duplicate query ordering")
        if not math.isfinite(query.fast_score) or not 0 <= query.fast_score <= 1:
            raise TaskInputError("frozen Fast probability missing or invalid")
        patches = query.last_frame_patches
        if patches.shape != (729, 1152) or patches.requires_grad or patches.dtype != self.dtype or patches.device != self.device or not bool(torch.isfinite(patches).all()):
            raise TaskInputError("original last-frame patches must retain frozen dtype/device/shape")
        if capture and self.protocol.rt_anomaly:
            dense = query.dense_patches
            # Original create_grid_image pads its frame_group list in-place to 4;
            # the optional dense RT path therefore also sees four frames at EOS.
            if dense is None or dense.shape != (4, 729, 1152) or dense.requires_grad or dense.dtype != self.dtype or dense.device != self.device or not bool(torch.isfinite(dense).all()):
                raise TaskInputError("target RT memory needs the original four-frame padded batch")

    @torch.no_grad()
    def step(self, query: DetectionQuery, *, capture: bool = False,
             image_height: int = 1, image_width: int = 1) -> tuple[FrozenTaskContext, str] | None:
        self._validate_query(query, capture)
        if min(image_height, image_width) < 1:
            raise TaskInputError("invalid original media dimensions")
        self.failed = True  # Any partial mutation requires a complete prefix replay.
        if self.protocol.memory_enhancement:
            self.memory.update_with_anomaly_score(query.last_frame_patches, anomaly_score=query.fast_score)
        else:
            self.memory.update(query.last_frame_patches)
        times = (*self.times, query.end_seconds)
        result = None
        if capture:
            # Capture before current-query pool insertion, as in the original SF call.
            visual = detection_memory_from_stream(self.slow, self.memory,
                rt_anomaly_tokens=query.dense_patches if self.protocol.rt_anomaly else None)
            style = self.protocol.prompt_style
            if style in {"neutral", "hivau"}:
                question = self.protocol.question_template
            elif style == "skeptical":
                context = self.memory.get_anomaly_context()["context_str"] if self.protocol.memory_enhancement else ""
                question = self.protocol.question_template.format(score_pct=int(round(query.fast_score * 100)), anomaly_context=context)
            else:
                question = self.protocol.question_template.format(score_pct=int(round(query.fast_score * 100)))
            time_message = self.time_formatter(self.protocol.time_message_style, query.end_seconds, len(times))
            dummy = torch.zeros(1, 3, image_height, image_width, device=self.device, dtype=self.dtype)
            result = (FrozenTaskContext(visual, [dummy], [(image_height, image_width)],
                                        query.end_seconds, times, time_message), question)
        # The original triggered and non-triggered branches both insert PG here.
        if self.protocol.memory_enhancement and query.fast_score >= self.protocol.pool_threshold:
            self.memory.update_anomaly_pool(query.last_frame_patches, query.fast_score)
        self.times.append(query.end_seconds)
        self.last_original_index = query.frame_indices[-1]
        self.failed = False
        return result


class FrozenDetectionProvider:
    """Assemble exactly one fixed prefix; reader never receives labels or teacher values."""
    def __init__(self, slow, catalog: TrainingCatalog, protocols: dict[str, DetectionProtocol],
                 reader: FrozenPrefixReader, observation_reader: Callable,
                 *, memory_factory=None, time_formatter: Callable = original_detection_time_message):
        self.slow, self.catalog, self.protocols = slow, catalog, protocols
        self.reader, self.observation_reader = reader, observation_reader
        self.memory_factory, self.time_formatter = memory_factory, time_formatter

    def __call__(self, sample_id: str) -> SampleMaterial:
        from .train_worker import SampleMaterial
        task = self.catalog.tasks[sample_id]
        if task.task != "detection" or task.query_index is None:
            raise TaskInputError("detection provider received another task")
        protocol = self.protocols[task.dataset]
        replay = DetectionMemoryReplay(self.slow, protocol, memory_factory=self.memory_factory,
                                       time_formatter=self.time_formatter)
        prefix = self.reader(task.dataset, task.media_key, task.query_index)
        captured = None
        queries = iter(prefix.queries)
        try:
            for query in queries:
                if query.index > task.query_index:
                    raise TaskInputError("prefix reader supplied future queries")
                target = query.index == task.query_index
                if target and abs(query.end_seconds - task.observed_seconds) > 1e-6:
                    raise TaskInputError("actual target endpoint differs from the frozen label endpoint")
                captured = replay.step(query, capture=target,
                                       image_height=prefix.image_height, image_width=prefix.image_width)
                if target:
                    break  # Never request later frames from a lazy reader.
        finally:
            close = getattr(queries, "close", None)
            if close is not None:
                close()
        if captured is None:
            raise TaskInputError("prefix reader omitted the selected query")
        context, question = captured
        observed = self.observation_reader(task.dataset, task.media_key, context.observed_seconds)
        observations = pack_observation_blocks([observed.features], task="detection",
                                               dtype=context.visual_embeddings.dtype, device=context.visual_embeddings.device)
        return SampleMaterial(context, observations, tuple(observed.relation_ids), question, protocol.scoring)
