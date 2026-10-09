from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nc_rted.statistics import (
    _PreparedScores,
    Prediction,
    StatisticsError,
    evaluate_six_primary,
    holm_adjust,
    source_weighted_auroc,
    source_weighted_average_precision,
    validate_prediction_matrix,
)


def _records(scores):
    return [
        Prediction("s0", "family-positive-0", "ucf-crime", 1, scores[0]),
        Prediction("s1", "family-negative-0", "ucf-crime", 0, scores[1]),
        Prediction("s2", "family-positive-1", "ucf-crime", 1, scores[2]),
        Prediction("s3", "family-negative-1", "ucf-crime", 0, scores[3]),
    ]


def test_tie_correct_metrics_match_sklearn_with_unit_and_random_weights():
    metrics = pytest.importorskip("sklearn.metrics")
    records = _records((.8, .8, .2, .1))
    labels = [row.label for row in records]
    scores = [row.score for row in records]
    for weights in (np.ones(4), np.asarray((.1, 2.5, 3., .7))):
        assert source_weighted_auroc(records, weights) == pytest.approx(metrics.roc_auc_score(labels, scores, sample_weight=weights))
        assert source_weighted_average_precision(records, weights) == pytest.approx(metrics.average_precision_score(labels, scores, sample_weight=weights))
    source_weights = np.asarray((1, 1, 1, 1), dtype=float)
    assert source_weighted_auroc(records) == pytest.approx(metrics.roc_auc_score(labels, scores, sample_weight=source_weights))


def test_full_sample_metric_and_source_multiplicity_match_sklearn_for_unequal_sources():
    metrics = pytest.importorskip("sklearn.metrics")
    records = [
        Prediction("a", "short", "ucf-crime", 1, .9),
        Prediction("b", "long", "ucf-crime", 0, .8),
        Prediction("c", "long", "ucf-crime", 1, .7),
        Prediction("d", "long", "ucf-crime", 0, .1),
    ]
    labels, scores = [row.label for row in records], [row.score for row in records]
    prepared = _PreparedScores.from_records(records)
    assert prepared.metric(np.ones(2), "auroc") == pytest.approx(metrics.roc_auc_score(labels, scores))
    assert prepared.metric(np.ones(2), "ap") == pytest.approx(metrics.average_precision_score(labels, scores))
    weights = np.asarray((2., 3., 3., 3.))
    assert prepared.metric(np.asarray((3, 2)), "auroc") == pytest.approx(metrics.roc_auc_score(labels, scores, sample_weight=weights))
    assert prepared.metric(np.asarray((3, 2)), "ap") == pytest.approx(metrics.average_precision_score(labels, scores, sample_weight=weights))


def _matrix():
    data = {}
    for group in ("A", "U", "S", "F"):
        data[group] = {}
        for seed in (17, 42, 2026):
            ucf = _records((.9, .1, .8, .2) if group == "F" else (.4, .8, .7, .2))
            xd = [Prediction("xd-" + row.sample_id, row.source_id, "xd-violence", row.label, row.score) for row in ucf]
            data[group][seed] = ucf + xd
    return data


def test_six_primary_statistics_are_reproducible_and_report_all_contract_fields():
    first = evaluate_six_primary(_matrix(), bootstrap_seed=1234, bootstrap_repeats=80)
    second = evaluate_six_primary(_matrix(), bootstrap_seed=1234, bootstrap_repeats=80)
    assert set(first) == {"F-A:ucf-crime", "F-U:ucf-crime", "F-S:ucf-crime", "F-A:xd-violence", "F-U:xd-violence", "F-S:xd-violence"}
    assert first == second
    result = first["F-A:ucf-crime"]
    assert result.effect > 0
    assert set(result.per_seed_effect) == {17, 42, 2026}
    assert result.valid_bootstrap_repeats + result.degenerate_bootstrap_repeats == 80
    assert 0 <= result.raw_p_value <= result.holm_p_value <= 1


def test_validation_rejects_identity_drift_nonfinite_scores_and_missing_class():
    matrix = _matrix()
    drift = _matrix()
    drift["F"][17][0] = Prediction("other", "family-0", "ucf-crime", 1, .9)
    with pytest.raises(StatisticsError, match="identical"):
        validate_prediction_matrix(drift)
    invalid = _matrix()
    invalid["A"][17][0] = Prediction("s0", "family-0", "ucf-crime", 1, float("nan"))
    with pytest.raises(StatisticsError, match="finite"):
        validate_prediction_matrix(invalid)
    only_positive = {group: {seed: [Prediction("x", "source", "ucf-crime", 1, .5)] for seed in (17, 42, 2026)} for group in ("A", "U", "S", "F")}
    with pytest.raises(StatisticsError, match="both classes"):
        validate_prediction_matrix(only_positive)


def test_degenerate_draws_are_counted_and_holm_is_step_down_monotone():
    result = evaluate_six_primary(_matrix(), bootstrap_seed=1, bootstrap_repeats=100)
    assert all(item.degenerate_bootstrap_repeats > 0 for item in result.values())
    assert all(item.inference_status == "INFERENCE_UNCONFIRMED_DEGENERATE_RESAMPLES" for item in result.values())
    adjusted = holm_adjust({"a": .01, "b": .03, "c": .04})
    assert adjusted == {"a": .03, "b": .06, "c": .06}
