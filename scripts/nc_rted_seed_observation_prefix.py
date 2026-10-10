#!/usr/bin/env python3
"""Seed a v25 observation journal from committed v24 windows; never runs models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nc_rted.observation_seed import seed_committed_prefix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--source-config-sha256", required=True)
    parser.add_argument("--target-config", type=Path, required=True)
    parser.add_argument("--target-config-sha256", required=True)
    parser.add_argument("--equivalence-report", type=Path, required=True)
    parser.add_argument("--equivalence-report-sha256", required=True)
    parser.add_argument("--probe-output", type=Path, required=True)
    parser.add_argument("--probe-config-sha256", required=True)
    args = parser.parse_args()
    values = vars(args)
    values["equivalence_sha256"] = values.pop("equivalence_report_sha256")
    print(json.dumps(seed_committed_prefix(**values), sort_keys=True))


if __name__ == "__main__":
    main()
