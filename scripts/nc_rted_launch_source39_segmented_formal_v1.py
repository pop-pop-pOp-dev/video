#!/usr/bin/env python3
"""Launch the accepted source39 segmented controller on the remote host once."""
from __future__ import annotations

import argparse
import subprocess
import textwrap


REMOTE_HOST = "root@connect.weste.seetacloud.com"
REMOTE_PORT = "35548"
CONTROL_PATH = "/root/autodl-tmp/lookaway-wm/.cache/nc_rted_pro6000_ssh.sock"


REMOTE_LAUNCHER = r'''
set -euo pipefail
source /root/autodl-tmp/lookaway-wm/configs/reactvau/download_environment.sh
/root/autodl-tmp/lookaway-wm/.venv-reactvau/bin/python - <<'PY'
import fcntl
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path("/root/autodl-tmp/lookaway-wm")
SOURCE = PROJECT / ".cache/nc_rted_segmented_training_source_v1"
OUTPUT = PROJECT / "artifacts/nc_rted/source39_segmented_seed17_v1"
LOG = PROJECT / "logs/nc_rted_source39_seed17_segmented_formal_v1.stdout.log"
RECEIPT = PROJECT / "reports/nc_rted/source39_seed17_segmented_formal_v1.launch.json"
LOCK = RECEIPT.with_suffix(".launch.lock")
PYTHON = PROJECT / ".venv-reactvau/bin/python"
CONTROLLER = SOURCE / "scripts/nc_rted_source39_segmented_formal_v1.py"
EXPECTED_CONTROLLER_SHA256 = "b116d6a089396c134e3b8ddd8b257903a5311f9637b6378acc92a44e2f8f5658"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def process_start_ticks(pid):
    return int((Path("/proc") / str(pid) / "stat").read_text().split()[21])


def atomic_once(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    return {"path": str(path), "sha256": hashlib.sha256(payload).hexdigest()}


def matching_controller_process():
    needle = str(CONTROLLER).encode()
    output = str(OUTPUT).encode()
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            command = (proc / "cmdline").read_bytes()
        except OSError:
            continue
        if needle in command or output in command:
            return int(proc.name), command.replace(b"\0", b" ").decode("utf-8", "replace")
    return None


def compute_process_observation():
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader,nounits"],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    return {
        "observed_utc": datetime.now(timezone.utc).isoformat(),
        "command": result.args,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


for required in (PYTHON, CONTROLLER, PROJECT / "scripts/run_logged.py"):
    if not required.is_file():
        raise RuntimeError(f"required launch path is absent: {required}")
if digest(CONTROLLER) != EXPECTED_CONTROLLER_SHA256:
    raise RuntimeError("accepted segmented controller bytes differ")

command = [
    str(PYTHON), str(PROJECT / "scripts/run_logged.py"), "--name", "source39-seed17-segmented-formal-v1", "--",
    str(PYTHON), str(CONTROLLER), "--runtime-source-root", str(SOURCE), "--output-root", str(OUTPUT),
    "--lease-id", "pro6000-user-seven-day-rental-20261009", "--lease-expires-utc", "1792080000", "--run",
]
LOCK.parent.mkdir(parents=True, exist_ok=True)
with LOCK.open("a+") as lock:
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    existing = matching_controller_process()
    if RECEIPT.exists() or OUTPUT.exists() or LOG.exists() or existing is not None:
        raise RuntimeError(f"refusing duplicate segmented launch: receipt={RECEIPT.exists()} output={OUTPUT.exists()} log={LOG.exists()} process={existing}")
    observation = compute_process_observation()
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("xb") as stream:
        process = subprocess.Popen(
            command, cwd=SOURCE, env={**os.environ, "PYTHONPATH": str(SOURCE / "src")},
            stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
        )
    receipt = atomic_once(RECEIPT, {
        "schema": "nc_rted_detached_launch/v1",
        "status": "LAUNCHED_DETACHED",
        "utc": datetime.now(timezone.utc).isoformat(),
        "logger_pid": process.pid,
        "logger_start_ticks": process_start_ticks(process.pid),
        "command": command,
        "project": str(PROJECT),
        "source": str(SOURCE),
        "controller": str(CONTROLLER),
        "controller_sha256": EXPECTED_CONTROLLER_SHA256,
        "output_root": str(OUTPUT),
        "stdout_log": str(LOG),
        "queue_phase": "CONTROLLER_STARTED_NO_DURABLE_SEGMENT_YET",
        "compute_process_availability_observation": observation,
    })
print(json.dumps({"status": "LAUNCHED_DETACHED", "receipt": receipt, "logger_pid": process.pid, "stdout_log": str(LOG)}, sort_keys=True))
PY
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print-command", action="store_true", help="Print the exact remote command without connecting.")
    args = parser.parse_args()
    command = [
        "ssh", "-p", REMOTE_PORT, "-o", "BatchMode=yes", "-o", f"ControlPath={CONTROL_PATH}",
        REMOTE_HOST, "/bin/bash", "-s",
    ]
    if args.print_command:
        print(" ".join(command))
        print(textwrap.dedent(REMOTE_LAUNCHER).strip())
        return
    subprocess.run(command, input=textwrap.dedent(REMOTE_LAUNCHER), text=True, check=True)


if __name__ == "__main__":
    main()
