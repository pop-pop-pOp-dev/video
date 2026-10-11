#!/usr/bin/env python3
"""Produce one fixed <=800-ID window and transport each immutable record safely."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import time

from nc_rted.frozen_sample_cache import FrozenSampleCacheError, FrozenSampleStore, _record_key, canonical
from nc_rted.frozen_sample_transport import (
    FrozenSampleTransportError,
    cache_status,
    evict_local_after_ack,
    install_ack,
    task_document,
    wait_for_capacity,
)
from nc_rted.frozen_provider_runtime import assemble_frozen_provider
from nc_rted.storage_lock import ensure_directory


class RemoteReceiver:
    """Minimal SSH/SCP adapter; record publication remains receiver-owned."""

    def __init__(self, args: argparse.Namespace):
        self.host, self.port = args.remote_host, args.remote_port
        self.control_path = args.ssh_control_path
        self.python, self.receiver, self.root = args.remote_python, args.remote_receiver, args.remote_root
        self.pythonpath = args.remote_pythonpath
        remote = shlex.join(["env", f"PYTHONPATH={self.pythonpath}", self.python, self.receiver, "serve"])
        self.process = subprocess.Popen(["ssh", "-p", str(self.port), "-o", f"ControlPath={self.control_path}",
                                         self.host, remote], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        text=True, bufsize=1)

    def close(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
        if self.process.poll() is None:
            self.process.terminate()
        self.process.wait(timeout=10)

    def _request(self, request: dict) -> dict:
        if self.process.stdin is None or self.process.stdout is None or self.process.poll() is not None:
            raise FrozenSampleTransportError("persistent remote receiver is not running")
        self.process.stdin.write(json.dumps(request, sort_keys=True, separators=(",", ":")) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise FrozenSampleTransportError("persistent remote receiver closed without a response")
        try:
            response = json.loads(line)
        except json.JSONDecodeError as error:
            raise FrozenSampleTransportError("persistent remote receiver returned invalid JSON") from error
        if not isinstance(response, dict) or "error" in response:
            raise FrozenSampleTransportError(f"remote receiver rejected request: {response.get('error')}")
        return response

    def _scp(self, source: Path, destination: str) -> None:
        subprocess.run(["scp", "-P", str(self.port), "-o", f"ControlPath={self.control_path}",
                        str(source), f"{self.host}:{destination}"], check=True)

    def status(self, max_bytes: int, min_free_bytes: int, next_bytes: int) -> dict:
        return self._request({"command": "status", "root": self.root, "max_bytes": max_bytes,
                              "min_free_bytes": min_free_bytes, "next_bytes": next_bytes})

    def send(self, *, tensor: Path, binding: Path, task: object, max_bytes: int,
             min_free_bytes: int, local_ack: Path) -> dict:
        binding_content = binding.read_bytes()
        record_key = hashlib.sha256(task.sample_id.encode()).hexdigest()
        nonce = hashlib.sha256(binding_content).hexdigest()
        incoming = PurePosixPath(self.root) / ".incoming"
        tensor_stage = incoming / f"{record_key}.{nonce}.pt.part"
        binding_stage = incoming / f"{record_key}.{nonce}.json.part"
        # Tensor arrives first and remains unpublished.  The binding is only a
        # staging companion; the receiver writes the final JSON publish point.
        self._scp(tensor, str(tensor_stage))
        self._scp(binding, str(binding_stage))
        receipt = self._request({"command": "receive", "root": self.root,
                                 "staged_tensor": str(tensor_stage), "staged_binding": str(binding_stage),
                                 "task": task_document(task), "max_bytes": max_bytes,
                                 "min_free_bytes": min_free_bytes})
        try:
            FrozenSampleStore._write_once(local_ack, canonical(receipt.pop("ack")))
        except (KeyError, FrozenSampleCacheError) as error:
            raise FrozenSampleTransportError("remote receiver did not return an immutable ACK") from error
        return receipt


def _sample_ids(path: Path) -> list[str]:
    try:
        values = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise FrozenSampleTransportError(f"cannot load fixed sample window: {error}") from error
    if (not isinstance(values, list) or not 1 <= len(values) <= 800 or
            not all(isinstance(value, str) and value for value in values) or len(set(values)) != len(values)):
        raise FrozenSampleTransportError("sample window must be an ordered unique list of at most 800 IDs")
    return values


def _runtime_is_training_runtime(path: Path) -> None:
    try:
        document = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise FrozenSampleTransportError(f"cannot read producer runtime: {error}") from error
    if not isinstance(document, dict) or not isinstance(document.get("fast"), dict) or not document["fast"].get("protocols"):
        raise FrozenSampleTransportError("producer runtime must bind the actual training detector protocols")


TRAINING_RUNTIME = Path("/root/autodl-tmp/lookaway-wm/artifacts/nc_rted/accel02_bindings/"
                        "source39_seed17_member_A_runtime.json")
TRAINING_RUNTIME_SHA256 = "3dd6a96d4550a7ba4ab94b6bf62e3451b0f7a00b4dc0e0ed4f2200e0ea01c991"
ACK_RESERVATION_BYTES = 4096


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--runtime-sha256", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--sample-list", type=Path, required=True,
                        help="fixed first-800 sampler IDs in exact training order")
    parser.add_argument("--local-root", type=Path, required=True)
    parser.add_argument("--local-max-bytes", type=int, default=1 << 30)
    parser.add_argument("--local-min-free-bytes", type=int, default=23_074_836_480)
    parser.add_argument("--remote-host", required=True)
    parser.add_argument("--remote-port", type=int, default=35548)
    parser.add_argument("--ssh-control-path", required=True)
    parser.add_argument("--remote-python", required=True)
    parser.add_argument("--remote-pythonpath", required=True,
                        help="reviewed remote source root containing nc_rted.frozen_sample_transport")
    parser.add_argument("--remote-receiver", required=True)
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--remote-max-bytes", type=int, default=80 << 30)
    parser.add_argument("--remote-min-free-bytes", type=int, default=20 << 30)
    parser.add_argument("--max-record-bytes", type=int, required=True,
                        help="strict capacity reservation before each provider call")
    parser.add_argument("--capacity-wait-seconds", type=float, default=300.0)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--evict-after-ack", action="store_true",
                        help="delete only the local recomputable record after exact remote ACK verification")
    parser.add_argument("--execution-report", type=Path, required=True)
    args = parser.parse_args()
    if args.local_max_bytes != 1 << 30 or args.remote_max_bytes > 80 << 30:
        parser.error("transport cache limits are fixed at local <=1GiB and remote <=80GiB")
    if args.local_min_free_bytes < 23_074_836_480 or args.remote_min_free_bytes < 20 << 30:
        parser.error("transport free-space reserves are below the admitted floors")
    if not 0 < args.max_record_bytes <= args.local_max_bytes - ACK_RESERVATION_BYTES:
        parser.error("max record reservation and immutable ACK must fit the 1GiB local cache")
    if args.remote_max_bytes < args.max_record_bytes + ACK_RESERVATION_BYTES:
        parser.error("remote cache cannot admit one reserved record and ACK")
    ids = _sample_ids(args.sample_list)
    if args.runtime.resolve() != TRAINING_RUNTIME.resolve() or args.runtime_sha256 != TRAINING_RUNTIME_SHA256:
        parser.error("frozen transport is bound only to source39_seed17_member_A_runtime.json")
    _runtime_is_training_runtime(args.runtime)
    runtime = assemble_frozen_provider(args.runtime, args.runtime_sha256, args.device)
    catalog, identity, provider = runtime.catalog, runtime.producer_identity, runtime.provider
    if not isinstance(identity, dict) or identity.get("catalog_sha256") != catalog.identity or not callable(provider):
        parser.error("frozen runtime identity or provider differs")
    if any(sample_id not in catalog.tasks for sample_id in ids):
        parser.error("fixed window contains a sample outside the bound training catalog")
    local = FrozenSampleStore(args.local_root, identity=identity, max_bytes=args.local_max_bytes,
                              min_free_bytes=args.local_min_free_bytes)
    remote = RemoteReceiver(args)
    results = []
    try:
        for sample_id in ids:
            task = catalog.tasks[sample_id]
            try:
                material = local.get(task)
            except FrozenSampleCacheError as error:
                raise FrozenSampleTransportError(f"local frozen record is invalid: {sample_id}") from error
            if material is None:
                # Both reservations occur before provider(sample_id), so a slow
                # consumer blocks this window rather than producing into a full cache.
                wait_for_capacity(lambda: cache_status(args.local_root, max_bytes=args.local_max_bytes,
                                                        min_free_bytes=args.local_min_free_bytes,
                                                        next_bytes=args.max_record_bytes + ACK_RESERVATION_BYTES),
                                  timeout_seconds=args.capacity_wait_seconds, poll_seconds=args.poll_seconds,
                                  where="local frozen cache")
                wait_for_capacity(lambda: remote.status(args.remote_max_bytes, args.remote_min_free_bytes,
                                                         args.max_record_bytes + ACK_RESERVATION_BYTES),
                                  timeout_seconds=args.capacity_wait_seconds, poll_seconds=args.poll_seconds,
                                  where="remote frozen cache")
                started = time.monotonic()
                record = local.put(task, provider(sample_id))
                produced = True
                elapsed = time.monotonic() - started
            else:
                record, produced, elapsed = local._record(task), False, 0.0
            del material
            tensor, binding = local.paths(sample_id)
            if record["bytes"] > args.max_record_bytes:
                raise FrozenSampleTransportError("produced record exceeds its pre-provider capacity reservation")
            wait_for_capacity(lambda: remote.status(args.remote_max_bytes, args.remote_min_free_bytes,
                                                     record["bytes"] + ACK_RESERVATION_BYTES),
                              timeout_seconds=args.capacity_wait_seconds, poll_seconds=args.poll_seconds,
                              where="remote frozen cache")
            ack_temporary = args.local_root / ".incoming" / f"{_record_key(sample_id)}.ack.part"
            ensure_directory(ack_temporary.parent, args.local_min_free_bytes)
            receipt = remote.send(tensor=tensor, binding=binding, task=task, max_bytes=args.remote_max_bytes,
                                  min_free_bytes=args.remote_min_free_bytes, local_ack=ack_temporary)
            installed_ack = install_ack(root=args.local_root, task=task, identity=identity, ack=ack_temporary,
                                        min_free_bytes=args.local_min_free_bytes)
            ack_temporary.unlink(missing_ok=True)
            if args.evict_after_ack:
                evict_local_after_ack(root=args.local_root, task=task, identity=identity, ack=installed_ack)
            result = {"sample_id": sample_id, "produced": produced, "producer_seconds": elapsed,
                      "local_record": record, "remote_receipt": receipt,
                      "local_ack": str(installed_ack), "local_evicted": args.evict_after_ack}
            results.append(result)
            print(json.dumps({"schema": "nc_rted_frozen_sample_transport_progress/v1", **result}, sort_keys=True),
                  flush=True)
    finally:
        remote.close()
    report = {"schema": "nc_rted_frozen_sample_transport_execution/v1", "identity": identity,
              "runtime": runtime.execution, "sample_ids": ids, "results": results,
              "remote_records": "retained"}
    args.execution_report.parent.mkdir(parents=True, exist_ok=True)
    FrozenSampleStore._write_once(args.execution_report, canonical(report))
    print(json.dumps({"schema": "nc_rted_frozen_sample_transport_complete/v1",
                      "execution_report": str(args.execution_report), "samples": len(results)}, sort_keys=True))


if __name__ == "__main__":
    main()
