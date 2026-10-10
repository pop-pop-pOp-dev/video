#!/usr/bin/env python3
"""Prepare or publish an R0 representative-measurement resource projection.

The ``inspect`` action is CPU-only and writes no execution admission.  The
``publish`` action needs an already-PASS operator authorization: it does not
invent a host, UUID, lease, capacity, or full-workload measurement.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nc_rted.prediction_inputs import (RESOURCE_ADMISSION_SCHEMA_V1, RESOURCE_ALLOCATION_SCHEMA_V1,
                                       RESOURCE_AUTHORIZATION_SCHEMA_V1, RESOURCE_EVIDENCE_REGISTRY_SCHEMA_V1,
                                       RESOURCE_PROJECTION_SCHEMA_V1, PredictionInputError, _validate_projection_output_budget,
                                       _validate_projection_timing_inputs, _validate_representative_projection_evidence, _validate_target_applicability,
                                       canonical_json, sha256_file)


WORKLOAD = {"kind": "blind_prediction", "vad_queries": 135050, "slow_triggers": 47458, "vau_requests": 3339}
OUTPUT_BUDGET_BYTES = 1_600_000_000
CENTRAL_SECONDS = 201726
MARGIN = .30
RUNTIME_BUDGET_SECONDS = 262260


class PublishError(ValueError):
    pass


def _read(path: str | Path, digest: str, name: str) -> tuple[Path, dict[str, Any]]:
    candidate = Path(path)
    if not candidate.is_absolute() or not candidate.is_file() or sha256_file(candidate) != digest:
        raise PublishError(f"{name} differs from its SHA-256")
    try:
        value = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise PublishError(f"{name} is not valid JSON") from error
    if not isinstance(value, dict):
        raise PublishError(f"{name} is not a JSON object")
    return candidate, value


def _reference(path: str, digest: str) -> dict[str, str]:
    return {"path": str(Path(path).resolve()), "sha256": digest}


def _write_new(path: str, document: dict[str, Any]) -> str:
    output = Path(path)
    if not output.is_absolute() or output.exists():
        raise PublishError("output must be a new absolute path")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json(document) + b"\n"
    output.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _representatives(inventory_path: str, inventory_sha: str) -> tuple[dict[str, Any], dict[str, str], list[dict[str, str]]]:
    inventory_file, inventory = _read(inventory_path, inventory_sha, "representative inventory")
    if (inventory.get("schema") != "nc_rted_fixed_prediction_measurement_inventory/v10" or
            inventory.get("status") != "FIXED_REPRESENTATIVES_COMPLETED_NOT_FORMAL_ADMISSION" or not isinstance(inventory.get("rows"), list)):
        raise PublishError("representative inventory is not the accepted fixed-measurement inventory")
    reports = []
    for row in inventory["rows"]:
        if not isinstance(row, dict) or row.get("status") != "PASS":
            continue
        report, report_sha = row.get("report"), row.get("report_sha256")
        acceptance, acceptance_sha = row.get("acceptance"), row.get("acceptance_sha256")
        if not all(isinstance(value, str) for value in (report, report_sha, acceptance, acceptance_sha)):
            raise PublishError("accepted representative is missing report/acceptance provenance")
        _, accepted = _read(acceptance, acceptance_sha, "representative acceptance")
        _, probe = _read(report, report_sha, "representative probe")
        terminal = accepted.get("terminal")
        candidate = probe.get("candidate_result")
        if (not isinstance(terminal, dict) or terminal.get("status") != "TERMINAL_RUNTIME_PROBE" or
                terminal.get("candidate_status") != "PASS_RUNTIME_PROBE" or terminal.get("report_sha256") != report_sha or
                accepted.get("report") != report or accepted.get("report_sha256") != report_sha or
                not isinstance(candidate, dict) or candidate.get("status") != "PASS_RUNTIME_PROBE" or
                candidate.get("formal_prediction") is not False or candidate.get("prediction_store_written") is not False):
            raise PublishError("representative acceptance/probe does not establish a non-formal PASS measurement")
        reports.append(_reference(report, report_sha))
    if len(reports) != 21:
        raise PublishError("accepted inventory must contribute exactly 21 fixed representatives")
    return inventory, _reference(str(inventory_file), inventory_sha), reports


def _inputs(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, str]], dict[str, int], dict[str, str]]:
    inventory_doc, inventory, reports = _representatives(args.inventory, args.inventory_sha256)
    preflight_file, preflight = _read(args.preflight, args.preflight_sha256, "source29 preflight")
    if (preflight.get("schema") != "nc_rted_r0_blind_manifest_build/v2" or preflight.get("gpu_launched") is not False or
            preflight.get("status") != "PREFLIGHT_PASS_FORMAL_ADMISSION_REQUIRED" or not isinstance(preflight.get("bindings"), dict)):
        raise PublishError("source29 preflight differs")
    timeline_file, timeline = _read(args.timeline, args.timeline_sha256, "source29 timing projection")
    if (timeline.get("schema") != "nc_rted_gate02_source29_stratified_timeline/v1" or
            timeline.get("status") != "CENTRAL_PLANNING_ESTIMATE_NOT_ADMISSION_OR_GUARANTEE" or
            timeline.get("central_per_model_seconds", {}).get("total") != CENTRAL_SECONDS or
            timeline.get("safety_margin", {}).get("fraction") != MARGIN):
        raise PublishError("timing projection does not support the fixed central estimate/margin")
    budget_file, budget = _read(args.output_budget_assessment, args.output_budget_assessment_sha256, "output budget assessment")
    try: _validate_projection_output_budget(budget, OUTPUT_BUDGET_BYTES)
    except PredictionInputError as error: raise PublishError(str(error)) from error
    device_timeline_file, _ = _read(args.device_timeline, args.device_timeline_sha256, "source29 device timeline")
    overhead_869_file, _ = _read(args.vau869_overhead_acceptance, args.vau869_overhead_acceptance_sha256,
                                 "source29 VAU869 overhead acceptance")
    overhead_3286_file, _ = _read(args.vau3286_overhead_acceptance, args.vau3286_overhead_acceptance_sha256,
                                  "source29 VAU3286 overhead acceptance")
    timing_inputs = {"inventory": inventory, "device_timeline": _reference(str(device_timeline_file), args.device_timeline_sha256),
                     "overhead_acceptances": [_reference(str(overhead_869_file), args.vau869_overhead_acceptance_sha256),
                                               _reference(str(overhead_3286_file), args.vau3286_overhead_acceptance_sha256)]}
    try: _validate_projection_timing_inputs(timing_inputs=timing_inputs, inventory_binding=inventory)
    except PredictionInputError as error: raise PublishError(str(error)) from error
    bindings = preflight["bindings"]
    if not isinstance(bindings.get("prediction_source"), dict):
        raise PublishError("source29 preflight prediction-source binding differs")
    try: peaks = _validate_representative_projection_evidence(inventory=inventory_doc, reports=reports, preflight=preflight)
    except PredictionInputError as error: raise PublishError(str(error)) from error
    peak_report = max(reports, key=lambda binding: json.loads(Path(binding["path"]).read_text())["candidate_result"]["peak_cuda_reserved_bytes"])
    evidence = {"representative_inventory": inventory, "representative_reports": reports,
                "peak": {**peaks, "report": peak_report},
                "source29_preflight": _reference(str(preflight_file), args.preflight_sha256),
                "prediction_source_sha256": bindings["prediction_source"].get("sha256"),
                "projection_report": _reference(str(timeline_file), args.timeline_sha256),
                "timing_inputs": timing_inputs,
                "output_budget_assessment": _reference(str(budget_file), args.output_budget_assessment_sha256)}
    return preflight, reports, evidence["peak"], evidence


def inspect(args: argparse.Namespace) -> None:
    _, reports, peak, evidence = _inputs(args)
    document = {"schema": "nc_rted_r0_projection_inputs/v1", "status": "PREPARED_NOT_RESOURCE_ADMISSION",
                "workload": WORKLOAD, "representative_count": len(reports), "representative_inventory": evidence["representative_inventory"],
                "representative_reports": reports, "peak_evidence": peak["report"], "peak_cuda_allocated_bytes": peak["allocated"],
                "peak_cuda_reserved_bytes": peak["reserved"], "source29_preflight": evidence["source29_preflight"],
                "projection_report": evidence["projection_report"], "projected_runtime_seconds": CENTRAL_SECONDS,
                "timing_inputs": evidence["timing_inputs"],
                "safety_margin_fraction": MARGIN, "runtime_budget_seconds": RUNTIME_BUDGET_SECONDS,
                "output_budget_assessment": evidence["output_budget_assessment"], "output_budget_bytes": OUTPUT_BUDGET_BYTES,
                "formal_execution_allowed": False}
    digest = _write_new(args.output, document)
    print(json.dumps({"status": document["status"], "output": str(Path(args.output).resolve()), "sha256": digest}, sort_keys=True))


def publish(args: argparse.Namespace) -> None:
    if not all((args.authorization, args.authorization_sha256, args.scope_json, args.projection_output,
                args.allocation_output, args.admission_output, args.registry_output)):
        raise PublishError("publish requires authorization, scope, and all four output paths")
    if not isinstance(args.allocation_id, str) or not args.allocation_id:
        raise PublishError("publish requires a nonempty allocation id")
    _, _, peak, evidence = _inputs(args)
    try: scope = json.loads(args.scope_json)
    except ValueError as error: raise PublishError("scope JSON is invalid") from error
    required_scope = {"kind", "run_id", "execution_scope_sha256", "device", "physical_gpu_uuid", "data_volume", "min_free_bytes", "run_budget_seconds", "deadline_utc_epoch"}
    if (not isinstance(scope, dict) or set(scope) != required_scope or scope.get("kind") != "blind_prediction" or
            scope.get("workload") is not None or scope.get("run_budget_seconds") != RUNTIME_BUDGET_SECONDS or
            scope.get("min_free_bytes") != 20 * 1024**3 + OUTPUT_BUDGET_BYTES):
        raise PublishError("scope must use the fixed R0 runtime and output reserve")
    applicability_file, _ = _read(args.target_applicability, args.target_applicability_sha256, "target applicability")
    authorization_file, authorization = _read(args.authorization, args.authorization_sha256, "operator authorization")
    if (authorization.get("schema") != RESOURCE_AUTHORIZATION_SCHEMA_V1 or authorization.get("status") != "PASS" or
            authorization.get("host") is None or authorization.get("physical_gpu_uuid") != scope["physical_gpu_uuid"] or
            authorization.get("project_volume") != scope["data_volume"] or authorization.get("max_budget_seconds") != scope["run_budget_seconds"] or
            authorization.get("min_free_bytes") != scope["min_free_bytes"] or authorization.get("deadline_utc_epoch") != scope["deadline_utc_epoch"]):
        raise PublishError("operator authorization does not cover the projection scope")
    allocation = {"schema": RESOURCE_ALLOCATION_SCHEMA_V1, "status": "ACCEPTED", "kind": "blind_prediction",
                  "execution_scope_sha256": scope["execution_scope_sha256"], "allocation_id": args.allocation_id,
                  "host": authorization["host"], "device": scope["device"], "physical_gpu_uuid": scope["physical_gpu_uuid"],
                  "lease_id": authorization["lease_id"], "authorization_sha256": args.authorization_sha256,
                  "limits": {key: scope[key] for key in ("data_volume", "min_free_bytes", "run_budget_seconds", "deadline_utc_epoch")}}
    allocation_hash = _write_new(args.allocation_output, allocation)
    projection = {"schema": RESOURCE_PROJECTION_SCHEMA_V1, "status": "PROJECTED_FROM_REPRESENTATIVE_MEASUREMENTS",
                  "projection_basis": "CONSERVATIVE_FULL_WORKLOAD_ESTIMATE", "kind": "blind_prediction",
                  "execution_scope_sha256": scope["execution_scope_sha256"], "allocation_sha256": allocation_hash,
                  "host": authorization["host"], "physical_gpu_uuid": scope["physical_gpu_uuid"], "workload": WORKLOAD,
                  "representative_inventory": evidence["representative_inventory"], "representative_reports": evidence["representative_reports"],
                  "peak_evidence": peak["report"], "source29_preflight": evidence["source29_preflight"],
                  "prediction_source_sha256": evidence["prediction_source_sha256"], "projection_report": evidence["projection_report"],
                  "timing_inputs": evidence["timing_inputs"],
                  "target_applicability": _reference(str(applicability_file), args.target_applicability_sha256),
                  "output_budget_assessment": evidence["output_budget_assessment"], "projected_runtime_seconds": CENTRAL_SECONDS,
                  "safety_margin_fraction": MARGIN, "runtime_budget_seconds": RUNTIME_BUDGET_SECONDS,
                  "peak_cuda_allocated_bytes": peak["allocated"], "peak_cuda_reserved_bytes": peak["reserved"],
                  "output_budget_bytes": OUTPUT_BUDGET_BYTES, "required_free_bytes": scope["min_free_bytes"]}
    try: _validate_target_applicability(projection["target_applicability"], projection=projection, scope=scope)
    except PredictionInputError as error: raise PublishError(str(error)) from error
    projection_hash = _write_new(args.projection_output, projection)
    admission = {"schema": RESOURCE_ADMISSION_SCHEMA_V1, "status": "PASS", "formal_execution_allowed": True, "scope": scope,
                 "accepted_allocation": _reference(args.allocation_output, allocation_hash),
                 "accepted_projection": _reference(args.projection_output, projection_hash)}
    admission_hash = _write_new(args.admission_output, admission)
    registry = {"schema": RESOURCE_EVIDENCE_REGISTRY_SCHEMA_V1, "accepted": {
        "resource_authorization": _reference(str(authorization_file), args.authorization_sha256),
        "resource_allocation": _reference(args.allocation_output, allocation_hash),
        "resource_projection": _reference(args.projection_output, projection_hash)}}
    registry_hash = _write_new(args.registry_output, registry)
    print(json.dumps({"status": "PUBLISHED_FROM_OPERATOR_AUTHORIZATION", "projection": _reference(args.projection_output, projection_hash),
                      "resource_admission": _reference(args.admission_output, admission_hash),
                      "accepted_evidence_registry": _reference(args.registry_output, registry_hash)}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", required=True); parser.add_argument("--inventory-sha256", required=True)
    parser.add_argument("--preflight", required=True); parser.add_argument("--preflight-sha256", required=True)
    parser.add_argument("--timeline", required=True); parser.add_argument("--timeline-sha256", required=True)
    parser.add_argument("--device-timeline", required=True); parser.add_argument("--device-timeline-sha256", required=True)
    parser.add_argument("--vau869-overhead-acceptance", required=True); parser.add_argument("--vau869-overhead-acceptance-sha256", required=True)
    parser.add_argument("--vau3286-overhead-acceptance", required=True); parser.add_argument("--vau3286-overhead-acceptance-sha256", required=True)
    parser.add_argument("--output-budget-assessment", required=True); parser.add_argument("--output-budget-assessment-sha256", required=True)
    parser.add_argument("--action", choices=("inspect", "publish"), default="inspect")
    parser.add_argument("--output")
    parser.add_argument("--authorization"); parser.add_argument("--authorization-sha256")
    parser.add_argument("--target-applicability"); parser.add_argument("--target-applicability-sha256")
    parser.add_argument("--scope-json"); parser.add_argument("--allocation-id")
    parser.add_argument("--allocation-output"); parser.add_argument("--projection-output")
    parser.add_argument("--admission-output"); parser.add_argument("--registry-output")
    args = parser.parse_args()
    try:
        if args.action == "inspect":
            if not args.output: raise PublishError("inspect requires --output")
            inspect(args)
        else: publish(args)
    except (PublishError, OSError, ValueError) as error:
        print(json.dumps({"status": "BLOCKED", "error": str(error), "formal_execution_allowed": False}, sort_keys=True), file=sys.stderr); raise SystemExit(2)


if __name__ == "__main__": main()
