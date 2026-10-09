"""Training-only task catalog and audited boundary into the inherited Slow task.

Reference identities live here for validation, never in student tensors. Frozen
visual memories are supplied by the original caption/detection runtime; this
module does not change its sampling, memory, prompt selection, or Fast scores.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Callable, Mapping

import torch
from torch import Tensor

from .batches import caption_block_endpoints
from .bridge import ObservationBatch, SlowInputs, TeacherBatch
from .features import CELL_FEATURE_DIM


class TaskInputError(ValueError):
    pass


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(8 << 20), b""):
            digest.update(part)
    return digest.hexdigest()


def detection_window_id(row: Mapping) -> str:
    return f"detection:{row['dataset']}:{row['key']}:{row['query_index']}"


@dataclass(frozen=True)
class TrainingTask:
    sample_id: str
    task: str
    dataset: str
    family: str
    media_key: str
    observed_seconds: float | None
    label: int | None
    instruction: dict | None
    query_index: int | None = None


class TrainingCatalog:
    """Read only fixed train manifests and hash-bound existing train instructions."""
    def __init__(self, tasks: list[TrainingTask], identity: str):
        self.tasks = {task.sample_id: task for task in tasks}
        if not tasks or len(self.tasks) != len(tasks):
            raise TaskInputError("empty or duplicate training sample identities")
        self.identity = identity

    @classmethod
    def load(cls, manifest_directory: str | Path, training_annotations: str | Path,
             *, expected_provenance_sha256: str):
        root, annotations = Path(manifest_directory), Path(training_annotations)
        provenance_path = root / "provenance.json"
        if sha256_file(provenance_path) != expected_provenance_sha256:
            raise TaskInputError("training manifest provenance changed")
        provenance = json.loads(provenance_path.read_text())
        if provenance.get("schema") != "nc_rted_manifest_provenance/v1":
            raise TaskInputError("unsupported manifest provenance")
        # Never open other inputs named in provenance (which also lists official
        # identity-only test preparation). Only the explicitly supplied train file.
        expected = provenance["inputs"].get(str(annotations.resolve()))
        if expected is None or sha256_file(annotations) != expected:
            raise TaskInputError("training annotations are not bound by provenance")
        selected = {}
        for name in ("source_splits.json", "train8000_captions.json", "train8000_detection_prefixes.json"):
            if sha256_file(root / name) != provenance["outputs"].get(name):
                raise TaskInputError(f"training manifest changed: {name}")
            selected[name] = json.loads((root / name).read_text())
        splits = {}
        for row in selected["source_splits.json"]:
            family = row["family"]
            if family in splits and splits[family] != row["allocation"]:
                raise TaskInputError("source family crosses allocations")
            splits[family] = row["allocation"]
        instructions = {}
        for row in json.loads(annotations.read_text()):
            key = (row["id"], row["video"])
            if key in instructions:
                raise TaskInputError("duplicate original training instruction")
            instructions[key] = row
        tasks = []
        balance = {}
        for row in selected["train8000_detection_prefixes.json"]:
            family, dataset = row["family"], row["dataset"]
            if splits.get(family) != "train" or dataset not in {"ucf-crime", "xd-violence"}:
                raise TaskInputError("detection sample outside incremental training allocation")
            if row["class"] not in {"normal", "anomalous"}:
                raise TaskInputError("unknown manifest detection class")
            endpoint = float(row["observed_seconds"])
            if not math.isfinite(endpoint) or endpoint < 0 or type(row["query_index"]) is not int or row["query_index"] < 0:
                raise TaskInputError("invalid causal detection endpoint")
            label = int(row["class"] == "anomalous")
            balance[(dataset, label)] = balance.get((dataset, label), 0) + 1
            tasks.append(TrainingTask(detection_window_id(row), "detection", dataset, family,
                                      row["key"], endpoint, label, None, row["query_index"]))
        for row in selected["train8000_captions.json"]:
            parent_family = row["parent_key"].split("__#", 1)[0] if row["dataset"] == "xd-violence" else row["parent_key"]
            family = f"{row['dataset']}:{parent_family}"
            original = instructions.get((row["id"], row["video"]))
            if splits.get(family) != "train" or original is None:
                raise TaskInputError("caption sample outside bound training sources")
            if row["task"] not in {"caption", "description"} or row["type"] not in {"clip", "event", "video"}:
                raise TaskInputError("caption matrix includes an illegal task")
            if any(original[k] != row[k] for k in ("task", "type", "video", "id")):
                raise TaskInputError("caption instruction does not match fixed identity")
            conversations = original["conversations"]
            if not conversations or not any(turn.get("from") == "gpt" and turn.get("value") for turn in conversations):
                raise TaskInputError("caption lacks an original supervised answer")
            tasks.append(TrainingTask(f"caption:{row['dataset']}:{row['id']}", "caption", row["dataset"],
                                      family, row["video"], None, None, copy.deepcopy(original)))
        if len(tasks) != 8000 or len(selected["train8000_captions.json"]) != 2000 or balance != {
                (dataset, label): 1500 for dataset in ("ucf-crime", "xd-violence") for label in (0, 1)}:
            raise TaskInputError("fixed 6000 balanced detection + 2000 caption denominator differs")
        return cls(tasks, expected_provenance_sha256)


@dataclass(frozen=True)
class FrozenTaskContext:
    """Already built by the pinned original runtime, at the actual observed end."""
    visual_embeddings: Tensor
    images: list[Tensor]
    image_sizes: list[tuple[int, int]]
    observed_seconds: float
    sampled_frame_times: tuple[float, ...]
    time_message: str

    def validate(self) -> None:
        t = self.observed_seconds
        if not math.isfinite(t) or t < 0 or not self.sampled_frame_times:
            raise TaskInputError("missing observed prefix context")
        times = self.sampled_frame_times
        if any(not math.isfinite(x) or x < 0 or x > t + 1e-6 for x in times) or any(
                right <= left for left, right in zip(times, times[1:])):
            raise TaskInputError("original visual memory contains future or nonchronological observations")
        if self.visual_embeddings.requires_grad or self.visual_embeddings.ndim != 3 or self.visual_embeddings.shape[0] != 1:
            raise TaskInputError("original visual tokens must be frozen microbatch-one embeddings")
        if not bool(torch.isfinite(self.visual_embeddings).all()):
            raise TaskInputError("nonfinite original visual memory")
        if len(self.images) != 1 or len(self.image_sizes) != 1:
            raise TaskInputError("exactly one inherited video metadata entry is required")


def validate_observation_scope(task: str, context: FrozenTaskContext, observations: ObservationBatch) -> None:
    context.validate()
    if task not in {"detection", "caption"}:
        raise TaskInputError("unknown task")
    features, valid, times = observations.features, observations.valid, observations.observed_times
    if features.ndim != 5 or features.shape[0] != 1 or features.shape[2] > 16 or features.shape[3:] != (4, CELL_FEATURE_DIM):
        raise TaskInputError("invalid production observation shape")
    if valid.dtype != torch.bool or valid.shape != features.shape[:-1] or times.shape != valid.shape or times.dtype != torch.float32:
        raise TaskInputError("production masks/timestamps require exact shapes and FP32 time")
    if not bool(torch.isfinite(features[valid]).all()) or not bool(torch.isfinite(times[valid]).all()):
        raise TaskInputError("nonfinite observed evidence")
    ends = (context.observed_seconds,) if task == "detection" else caption_block_endpoints(context.observed_seconds)
    if features.shape[1] != len(ends):
        raise TaskInputError("observation blocks omit or duplicate the full observed range")
    for block, end in enumerate(ends):
        start = max(0., end - 8.) if task == "detection" else block * 8.
        actual = times[0, block][valid[0, block]]
        # The first media frame at zero is legal; other blocks use (start,end].
        if bool(((actual < start) | (actual > end + 1e-6)).any()) or (start > 0 and bool((actual <= start).any())):
            raise TaskInputError("evidence timestamp outside its causal block")
        for relation in range(valid.shape[2]):
            observed = times[0, block, relation][valid[0, block, relation]]
            if bool((observed[1:] <= observed[:-1]).any()):
                raise TaskInputError("relation times must be strictly increasing")


class InheritedTaskTokenizer:
    """Calls the original Stage2 preprocessing functions without replacing its CE masks.

    The caller pins source/tokenizer hashes. Functions are injected to avoid
    importing the original trainer's CLI/runtime side effects in worker imports.
    Production construction uses ``from_original``; no tokenizer is reimplemented.
    """
    def __init__(self, tokenizer, data_args, preprocess_multimodal: Callable,
                 preprocess_qwen: Callable):
        self.tokenizer, self.data_args = tokenizer, data_args
        self.multimodal, self.qwen = preprocess_multimodal, preprocess_qwen

    @classmethod
    def from_original(cls, tokenizer, data_args):
        from llava.train.train import preprocess_multimodal, preprocess_qwen
        return cls(tokenizer, data_args, preprocess_multimodal, preprocess_qwen)

    def encode(self, task: TrainingTask, context: FrozenTaskContext, *, detection_question: str | None = None,
               detection_scoring: str = "yesno") -> SlowInputs:
        context.validate()
        if task.task == "caption":
            if task.instruction is None:
                raise TaskInputError("caption requires its original training instruction")
            conversations = copy.deepcopy(task.instruction["conversations"])
        elif task.task == "detection":
            if task.label not in (0, 1) or task.observed_seconds is None or abs(task.observed_seconds - context.observed_seconds) > 1e-6:
                raise TaskInputError("detection target does not match frozen causal endpoint")
            if not detection_question or detection_scoring != "yesno":
                raise TaskInputError("binary prefix targets require the inherited yes/no question, not a rating/CoT target")
            conversations = [{"from": "human", "value": "<image>\n" + detection_question},
                             {"from": "gpt", "value": "Yes" if task.label else "No"}]
        else:
            raise TaskInputError("unknown task")
        for turn in conversations:
            turn["value"] = turn["value"].replace("<video>", "<image>")
        sources = self.multimodal([conversations], self.data_args, msg=context.time_message)
        encoded = self.qwen(sources, self.tokenizer, has_image=True)
        ids, labels = encoded["input_ids"], encoded["labels"]
        if ids.ndim != 2 or ids.shape[0] != 1 or labels.shape != ids.shape or int((ids == -200).sum()) != 1:
            raise TaskInputError("inherited tokenization must preserve one visual insertion and every target")
        if not bool((labels != -100).any()):
            raise TaskInputError("empty supervised task")
        device = context.visual_embeddings.device
        return SlowInputs(ids.to(device), context.visual_embeddings, context.images, context.image_sizes,
                          labels=labels.to(device), attention_mask=torch.ones_like(ids, device=device))


def teacher_batch(row: Mapping, *, sample_id: str, group: str, relation_ids: tuple[str, ...],
                  observations: ObservationBatch) -> TeacherBatch:
    """Align teacher audit IDs to observation order, leaving student masks untouched."""
    if group not in {"A", "U", "S", "F"} or row.get("window_id") != sample_id:
        raise TaskInputError("teacher identity or group mismatch")
    shape = observations.valid.shape
    if shape[:2] != (1, 1) or shape[2] != len(relation_ids) or len(set(relation_ids)) != len(relation_ids):
        raise TaskInputError("teacher requires one detection window with unique audit relation IDs")
    quality = torch.zeros((1, 1), dtype=torch.float32)
    positions = torch.zeros(shape, dtype=torch.float32)
    eligible = torch.zeros((1, 1), dtype=torch.bool)
    if row.get("aux_valid") is not True:
        if row.get("aux_valid") is not False or not row.get("rejection"):
            raise TaskInputError("rejected teacher must record an explicit reason")
        return TeacherBatch(quality, positions, eligible)
    stored_ids = row["relation_ids"]
    if len(stored_ids) != len(set(stored_ids)) or set(stored_ids) != set(relation_ids):
        raise TaskInputError("teacher relation set differs from frozen observations")
    mask = torch.as_tensor(row["mask"], dtype=torch.bool)
    values = torch.as_tensor(row["F_positions"], dtype=torch.float32)
    if mask.shape != (64,) or values.shape != (64,) or bool(mask[len(stored_ids) * 4:].any()):
        raise TaskInputError("teacher must use padded 16 by 4 support without phantom relations")
    if not bool(torch.isfinite(values).all()) or bool((values < 0).any()) or bool((values[~mask] != 0).any()):
        raise TaskInputError("invalid teacher position distribution")
    if abs(float(values.sum()) - 1.) > 1e-5 or not bool(mask.any()):
        raise TaskInputError("valid teacher needs normalized nonempty supported locations")
    indices = {key: index for index, key in enumerate(stored_ids)}
    for index, key in enumerate(relation_ids):
        offset = indices[key] * 4
        support = mask[offset:offset + 4]
        if bool((support & ~observations.valid[0, 0, index].cpu()).any()):
            raise TaskInputError("teacher places support on an unobserved cell")
        positions[0, 0, index] = values[offset:offset + 4]
    f, s, u = (float(row[key]) for key in ("F_quality", "S_quality", "U_quality"))
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in (f, s, u)) or f != s:
        raise TaskInputError("teacher qualities must be legal and S/F identical")
    quality[0, 0] = u if group == "U" else f
    eligible[0, 0] = True
    return TeacherBatch(quality, positions, eligible)
