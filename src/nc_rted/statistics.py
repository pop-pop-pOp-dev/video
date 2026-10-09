"""Predeclared source-family paired statistics for blind NC-RTED predictions."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np


class StatisticsError(ValueError):
    pass


@dataclass(frozen=True)
class Prediction:
    sample_id: str
    source_id: str
    dataset: str
    label: int
    score: float


@dataclass(frozen=True)
class Comparison:
    name: str
    dataset: str
    metric: str
    candidate: str
    baseline: str


@dataclass(frozen=True)
class ComparisonResult:
    name: str
    dataset: str
    metric: str
    effect: float
    ci95: tuple[float, float]
    raw_p_value: float
    holm_p_value: float
    per_seed_effect: dict[int, float]
    seed_mean: float
    seed_sd: float
    bootstrap_repeats: int
    valid_bootstrap_repeats: int
    degenerate_bootstrap_repeats: int
    inference_status: str


def _validate_records(records: Sequence[Prediction]) -> dict[str, tuple[str, str, int]]:
    identities: dict[str, tuple[str, str, int]] = {}
    for row in records:
        if not row.sample_id or not row.source_id or not row.dataset:
            raise StatisticsError("sample, source, and dataset identities must be nonempty")
        if row.label not in (0, 1) or not math.isfinite(row.score):
            raise StatisticsError("labels must be binary and scores finite")
        identity = (row.source_id, row.dataset, row.label)
        if row.sample_id in identities:
            raise StatisticsError("duplicate sample identity")
        identities[row.sample_id] = identity
    if not identities:
        raise StatisticsError("prediction set is empty")
    return identities


def validate_prediction_matrix(predictions: Mapping[str, Mapping[int, Sequence[Prediction]]],
                               groups: Sequence[str] = ("A", "U", "S", "F"),
                               seeds: Sequence[int] = (17, 42, 2026)) -> None:
    """Require a complete, identical sample/source/label matrix before scoring."""
    expected: dict[str, tuple[str, str, int]] | None = None
    for group in groups:
        if group not in predictions:
            raise StatisticsError(f"missing group {group}")
        for seed in seeds:
            if seed not in predictions[group]:
                raise StatisticsError(f"missing seed {seed} for group {group}")
            actual = _validate_records(predictions[group][seed])
            if expected is None:
                expected = actual
            elif actual != expected:
                raise StatisticsError("groups/seeds must retain identical sample/source/label identities")
    assert expected is not None
    for dataset in {dataset for _, dataset, _ in expected.values()}:
        labels = {label for _, row_dataset, label in expected.values() if row_dataset == dataset}
        if labels != {0, 1}:
            raise StatisticsError(f"dataset {dataset} does not contain both classes")


@dataclass(frozen=True)
class _PreparedScores:
    labels: np.ndarray
    source_index: np.ndarray
    source_sizes: np.ndarray
    group_ends: np.ndarray

    @classmethod
    def from_records(cls, records: Sequence[Prediction]) -> "_PreparedScores":
        _validate_records(records)
        source_ids = sorted({row.source_id for row in records})
        source_index = {source: index for index, source in enumerate(source_ids)}
        order = np.argsort(np.asarray([-row.score for row in records]), kind="mergesort")
        labels = np.asarray([records[index].label for index in order], dtype=np.int8)
        sources = np.asarray([source_index[records[index].source_id] for index in order], dtype=np.int64)
        scores = np.asarray([records[index].score for index in order], dtype=np.float64)
        ends = np.r_[np.flatnonzero(scores[:-1] != scores[1:]), len(scores) - 1]
        sizes = np.bincount(sources, minlength=len(source_ids)).astype(np.float64)
        return cls(labels, sources, sizes, ends)

    def metric(self, multiplicities: np.ndarray, name: str) -> float:
        if multiplicities.shape != self.source_sizes.shape:
            raise StatisticsError("source bootstrap identity mismatch")
        weights = multiplicities[self.source_index].astype(np.float64, copy=False)
        positive = float(weights[self.labels == 1].sum())
        negative = float(weights[self.labels == 0].sum())
        if positive <= 0 or negative <= 0:
            raise StatisticsError("metric is undefined without both weighted classes")
        group_positive = np.add.reduceat(weights * (self.labels == 1), np.r_[0, self.group_ends[:-1] + 1])
        group_negative = np.add.reduceat(weights * (self.labels == 0), np.r_[0, self.group_ends[:-1] + 1])
        if name == "auroc":
            prior_negative = np.r_[0., np.cumsum(group_negative)[:-1]]
            return float(np.sum(group_positive * (negative - prior_negative - .5 * group_negative)) / (positive * negative))
        if name == "ap":
            cumulative_positive = np.cumsum(group_positive)
            cumulative_total = cumulative_positive + np.cumsum(group_negative)
            precision = np.divide(cumulative_positive, cumulative_total, out=np.zeros_like(cumulative_positive), where=cumulative_total > 0)
            return float(np.sum(group_positive / positive * precision))
        raise StatisticsError("metric must be auroc or ap")


def source_weighted_auroc(records: Sequence[Prediction], sample_weight: Sequence[float] | None = None) -> float:
    """Tie-correct AUROC, matching sklearn for supplied positive sample weights."""
    return _metric_records(records, sample_weight, "auroc")


def source_weighted_average_precision(records: Sequence[Prediction], sample_weight: Sequence[float] | None = None) -> float:
    """Tie-correct non-interpolated AP, matching sklearn for supplied weights."""
    return _metric_records(records, sample_weight, "ap")


def _metric_records(records: Sequence[Prediction], sample_weight: Sequence[float] | None, metric: str) -> float:
    _validate_records(records)
    weights = np.ones(len(records), dtype=np.float64) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
    if weights.shape != (len(records),) or np.any(weights < 0) or not np.all(np.isfinite(weights)):
        raise StatisticsError("sample weights must be finite, nonnegative, and aligned")
    order = np.argsort(np.asarray([-row.score for row in records]), kind="mergesort")
    labels, weights = np.asarray([records[index].label for index in order]), weights[order]
    scores = np.asarray([records[index].score for index in order])
    ends = np.r_[np.flatnonzero(scores[:-1] != scores[1:]), len(scores) - 1]
    positive, negative = weights[labels == 1].sum(), weights[labels == 0].sum()
    if positive <= 0 or negative <= 0:
        raise StatisticsError("metric is undefined without both weighted classes")
    starts = np.r_[0, ends[:-1] + 1]
    p = np.add.reduceat(weights * (labels == 1), starts)
    n = np.add.reduceat(weights * (labels == 0), starts)
    if metric == "auroc":
        return float(np.sum(p * (negative - np.r_[0., np.cumsum(n)[:-1]] - .5 * n)) / (positive * negative))
    cp, ct = np.cumsum(p), np.cumsum(p + n)
    precision = np.divide(cp, ct, out=np.zeros_like(cp), where=ct > 0)
    return float(np.sum(p / positive * precision))


def holm_adjust(raw_p_values: Mapping[str, float]) -> dict[str, float]:
    if not raw_p_values:
        raise StatisticsError("Holm adjustment needs at least one p value")
    if any(not 0 <= value <= 1 for value in raw_p_values.values()):
        raise StatisticsError("p values must be in [0, 1]")
    ordered = sorted(raw_p_values, key=lambda name: (raw_p_values[name], name))
    total, running, result = len(ordered), 0., {}
    for rank, name in enumerate(ordered):
        running = max(running, min(1., (total - rank) * raw_p_values[name]))
        result[name] = running
    return result


def evaluate_six_primary(predictions: Mapping[str, Mapping[int, Sequence[Prediction]]], *,
                         bootstrap_seed: int, bootstrap_repeats: int = 10_000,
                         seeds: Sequence[int] = (17, 42, 2026)) -> dict[str, ComparisonResult]:
    """Evaluate the six preregistered F-A/F-U/F-S x UCF/XD comparisons."""
    if bootstrap_repeats < 1:
        raise StatisticsError("bootstrap_repeats must be positive")
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise StatisticsError("formal statistics require exactly three distinct training seeds")
    validate_prediction_matrix(predictions, seeds=seeds)
    comparisons = tuple(Comparison(f"F-{base}:{dataset}", dataset, metric, "F", base)
                        for dataset, metric in (("ucf-crime", "auroc"), ("xd-violence", "ap"))
                        for base in ("A", "U", "S"))
    prepared = {(group, seed, dataset): _PreparedScores.from_records(
        [row for row in predictions[group][seed] if row.dataset == dataset])
        for group in ("A", "U", "S", "F") for seed in seeds
        for dataset in ("ucf-crime", "xd-violence")}
    source_counts = {dataset: prepared[("A", seeds[0], dataset)].source_sizes.size for dataset in ("ucf-crime", "xd-violence")}
    for dataset, source_count in source_counts.items():
        if any(prepared[(group, seed, dataset)].source_sizes.size != source_count for group in ("A", "U", "S", "F") for seed in seeds):
            raise StatisticsError("source identities are inconsistent within a dataset")
    per_seed = {comparison.name: {seed: prepared[(comparison.candidate, seed, comparison.dataset)].metric(np.ones(source_counts[comparison.dataset], dtype=np.int64), comparison.metric) - prepared[(comparison.baseline, seed, comparison.dataset)].metric(np.ones(source_counts[comparison.dataset], dtype=np.int64), comparison.metric) for seed in seeds} for comparison in comparisons}
    effect = {name: float(np.mean(list(values.values()))) for name, values in per_seed.items()}
    draws: dict[str, list[float]] = {comparison.name: [] for comparison in comparisons}
    degenerate = {comparison.name: 0 for comparison in comparisons}
    rng = np.random.default_rng(bootstrap_seed)
    for _ in range(bootstrap_repeats):
        multiplicities = {dataset: np.bincount(rng.integers(source_count, size=source_count), minlength=source_count)
                          for dataset, source_count in source_counts.items()}
        for comparison in comparisons:
            try:
                draw = multiplicities[comparison.dataset]
                difference = [prepared[(comparison.candidate, seed, comparison.dataset)].metric(draw, comparison.metric) - prepared[(comparison.baseline, seed, comparison.dataset)].metric(draw, comparison.metric) for seed in seeds]
            except StatisticsError:
                degenerate[comparison.name] += 1
            else:
                draws[comparison.name].append(float(np.mean(difference)))
    raw_p = {}
    for comparison in comparisons:
        values = np.asarray(draws[comparison.name])
        if not len(values):
            raise StatisticsError(f"every bootstrap draw was degenerate for {comparison.name}")
        raw_p[comparison.name] = float((1 + np.count_nonzero(np.abs(values - effect[comparison.name]) >= abs(effect[comparison.name])) + degenerate[comparison.name]) / (bootstrap_repeats + 1))
    adjusted = holm_adjust(raw_p)
    return {comparison.name: ComparisonResult(comparison.name, comparison.dataset, comparison.metric,
            effect[comparison.name], tuple(float(value) for value in np.quantile(draws[comparison.name], (.025, .975))), raw_p[comparison.name], adjusted[comparison.name],
            per_seed[comparison.name], effect[comparison.name], float(np.std(list(per_seed[comparison.name].values()), ddof=1)), bootstrap_repeats,
            len(draws[comparison.name]), degenerate[comparison.name],
            "INFERENCE_UNCONFIRMED_DEGENERATE_RESAMPLES" if degenerate[comparison.name] else "INFERENCE_READY") for comparison in comparisons}
