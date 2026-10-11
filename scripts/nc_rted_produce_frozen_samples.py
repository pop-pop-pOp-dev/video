#!/usr/bin/env python3
"""Publish immutable frozen SampleMaterial records from a supplied GPU provider."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from nc_rted.frozen_sample_cache import FrozenSampleStore
from nc_rted.frozen_provider_runtime import assemble_frozen_provider


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--runtime-sha256", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-bytes", type=int, default=80 << 30)
    parser.add_argument("--min-free-bytes", type=int, default=20 << 30)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--sample-id", action="append")
    source.add_argument("--sample-list", type=Path,
                        help="JSON array of fixed catalog sample IDs, in execution order")
    parser.add_argument("--execution-report", type=Path, required=True)
    args = parser.parse_args()
    if args.sample_list is not None:
        try:
            sample_ids = json.loads(args.sample_list.read_text())
        except (OSError, json.JSONDecodeError) as error:
            parser.error(f"cannot load sample list: {error}")
        if not isinstance(sample_ids, list) or not all(isinstance(item, str) for item in sample_ids):
            parser.error("sample list must be a JSON array of strings")
    else:
        sample_ids = args.sample_id
    runtime = assemble_frozen_provider(args.runtime, args.runtime_sha256, args.device)
    catalog, identity, provider = runtime.catalog, runtime.producer_identity, runtime.provider
    if not isinstance(identity, dict) or identity.get("catalog_sha256") != catalog.identity:
        parser.error("frozen provider identity does not bind this catalog")
    if not isinstance(getattr(runtime, "execution", None), dict):
        parser.error("frozen provider must expose execution metadata")
    if not sample_ids or len(set(sample_ids)) != len(sample_ids) or any(sample_id not in catalog.tasks for sample_id in sample_ids):
        parser.error("sample IDs must be unique fixed catalog IDs")
    if not callable(provider):
        parser.error("frozen provider is not callable")
    store = FrozenSampleStore(args.output_root, identity=identity, max_bytes=args.max_bytes,
                              min_free_bytes=args.min_free_bytes)
    records = []
    for sample_id in sample_ids:
        started = time.monotonic()
        task = catalog.tasks[sample_id]
        reused = store.get(task)
        if reused is None:
            record, status = store.put(task, provider(sample_id)), "READY"
        else:
            record, status = store._record(task), "REUSED"
        del reused
        records.append(record)
        print(json.dumps({"schema": "nc_rted_frozen_sample_progress/v1", "sample_id": sample_id,
                          "status": status, "elapsed_seconds": time.monotonic() - started,
                          "bytes": record["bytes"]}, sort_keys=True), flush=True)
    report = {"schema": "nc_rted_frozen_sample_execution/v1", "identity": identity,
              "execution": runtime.execution, "sample_ids": sample_ids, "records": records}
    args.execution_report.parent.mkdir(parents=True, exist_ok=True)
    FrozenSampleStore._write_once(args.execution_report, json.dumps(
        report, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n")
    print(json.dumps({"schema": "nc_rted_frozen_sample_production/v1", "identity": identity,
                      "execution_report": str(args.execution_report), "records": records}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
