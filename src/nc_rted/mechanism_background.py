"""Actual global-context feature interventions for Section 9 diagnostics."""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

import numpy as np

from .detector import CausalWindowObservation
from .features import FeatureAssemblyResult, STUDENT_BLOCK_SLICES


BACKGROUND_SCHEMA = "nc_rted_global_context_feature_diagnostic/v1"
GLOBAL = STUDENT_BLOCK_SLICES["global"]


class BackgroundDiagnosticError(ValueError):
    pass


def _result(value: object) -> FeatureAssemblyResult:
    if not isinstance(value, CausalWindowObservation) or not isinstance(value.features, FeatureAssemblyResult):
        raise BackgroundDiagnosticError("global-context diagnostic requires assembled frozen features")
    return value.features


def _replace_global(observation: CausalWindowObservation, *, donor: CausalWindowObservation | None) -> CausalWindowObservation:
    source = _result(observation)
    donor_result = None if donor is None else _result(donor)
    if donor_result is not None and len(source.relations) != len(donor_result.relations):
        raise BackgroundDiagnosticError("matched global-context donor relation layout differs")
    relations = []
    for index, relation in enumerate(source.relations):
        values, valid = np.asarray(relation.student_cells).copy(), np.asarray(relation.feature_valid, dtype=bool)
        if donor_result is None:
            values[valid, GLOBAL] = 0
        else:
            donor_relation = donor_result.relations[index]
            if not np.array_equal(valid, np.asarray(donor_relation.feature_valid, dtype=bool)):
                raise BackgroundDiagnosticError("matched global-context donor validity differs")
            donor_values = np.asarray(donor_relation.student_cells)
            if not np.isfinite(donor_values[valid, GLOBAL]).all():
                raise BackgroundDiagnosticError("matched global-context donor is nonfinite")
            values[valid, GLOBAL] = donor_values[valid, GLOBAL]
        relations.append(replace(relation, student_cells=values))
    return replace(observation, features=replace(source, relations=tuple(relations)))


def _compatible_donor(source: CausalWindowObservation, candidate: CausalWindowObservation) -> bool:
    """Require the exact relation/time support consumed by the replacement."""
    source_result, candidate_result = _result(source), _result(candidate)
    return (len(source_result.relations) == len(candidate_result.relations)
            and all(np.array_equal(first.feature_valid, second.feature_valid)
                    for first, second in zip(source_result.relations, candidate_result.relations)))


def run_global_context_diagnostics(*, records: Mapping[str, Mapping[str, Any]],
                                   observations: Mapping[str, CausalWindowObservation], baseline_rows: Mapping[str, Mapping[str, Any]],
                                   bridge, reader, tokenizer, protocols: Mapping[str, Any],
                                   generation_config: Mapping[str, Any], yes_token_ids: tuple[int, ...],
                                   no_token_ids: tuple[int, ...]) -> dict:
    """Run altered global-context features through the existing real prefix executor."""
    from .mechanism_diagnostics import development_detection_task, execute_detection_prefix
    rows = []
    for sample_id in sorted(records):
        baseline, observation = baseline_rows[sample_id], observations.get(sample_id)
        common = {"sample_id": sample_id, "feature_layer": "student_cells.global",
                  "scope": "whole-frame global-context feature; not a pixel or pure-background intervention"}
        if observation is None or not _result(observation).relations:
            rows.append({**common, "intervention": "baseline", "execution_status": "UNAVAILABLE",
                         "baseline_slow_output": baseline})
            continue
        task, protocol = development_detection_task(records[sample_id]), protocols.get(records[sample_id]["dataset"])
        if protocol is None:
            raise BackgroundDiagnosticError("global-context diagnostic source has no protocol")
        rows.append({**common, "intervention": "baseline", "execution_status": "EXECUTED",
                     "baseline_slow_output": baseline})
        donor = next((observations[other] for other in sorted(observations)
                      if other != sample_id and records[other]["family"] != records[sample_id]["family"]
                      and _compatible_donor(observation, observations[other])), None)
        for name, selected_donor in (("zero_global_context", None), ("matched_global_context_replacement", donor)):
            if name.startswith("matched") and selected_donor is None:
                rows.append({**common, "intervention": name, "execution_status": "NO_COMPATIBLE_CROSS_FAMILY_DONOR"})
                continue
            altered = _replace_global(observation, donor=selected_donor)
            def altered_reader(dataset, key, endpoint):
                if (dataset, key, endpoint) != (task.dataset, task.media_key, task.observed_seconds):
                    raise BackgroundDiagnosticError("global-context observation identity differs")
                return altered
            row = execute_detection_prefix(bridge=bridge, task=task, reader=reader, observation_reader=altered_reader,
                                           tokenizer=tokenizer, protocol=protocol, name="baseline",
                                           generation_config=generation_config, yes_token_ids=yes_token_ids,
                                           no_token_ids=no_token_ids)
            row["intervention"] = name
            row.update(common)
            row["execution_status"] = "EXECUTED"
            rows.append(row)
    return {"schema": BACKGROUND_SCHEMA, "feature_layer": "student_cells.global",
            "scope": "whole-frame global-context feature only; no pixel/background/label-flip claim", "records": rows}
