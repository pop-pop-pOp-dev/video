#!/usr/bin/env python3
"""Launch the corrected source39 segmented controller on the remote host once."""
from __future__ import annotations

import subprocess
import textwrap

from nc_rted_launch_source39_segmented_formal_v1 import CONTROL_PATH, REMOTE_HOST, REMOTE_PORT, REMOTE_LAUNCHER


SOURCE_V1 = ".cache/nc_rted_segmented_training_source_v1"
SOURCE_V2 = ".cache/nc_rted_segmented_training_source_v2"
OUTPUT_V1 = "source39_segmented_seed17_v1"
OUTPUT_V2 = "source39_segmented_seed17_v2"
LOG_V1 = "nc_rted_source39_seed17_segmented_formal_v1.stdout.log"
LOG_V2 = "nc_rted_source39_seed17_segmented_formal_v2.stdout.log"
RECEIPT_V1 = "source39_seed17_segmented_formal_v1.launch.json"
RECEIPT_V2 = "source39_seed17_segmented_formal_v2.launch.json"
EXPECTED_CONTROLLER_SHA256 = "b116d6a089396c134e3b8ddd8b257903a5311f9637b6378acc92a44e2f8f5658"


def main() -> None:
    launcher = REMOTE_LAUNCHER.replace(SOURCE_V1, SOURCE_V2).replace(OUTPUT_V1, OUTPUT_V2)
    launcher = launcher.replace(LOG_V1, LOG_V2).replace(RECEIPT_V1, RECEIPT_V2)
    if EXPECTED_CONTROLLER_SHA256 not in launcher:
        raise RuntimeError("v2 launch template lost the accepted controller pin")
    command = [
        "ssh", "-p", REMOTE_PORT, "-o", "BatchMode=yes", "-o", f"ControlPath={CONTROL_PATH}",
        REMOTE_HOST, "/bin/bash", "-s",
    ]
    subprocess.run(command, input=textwrap.dedent(launcher), text=True, check=True)


if __name__ == "__main__":
    main()
