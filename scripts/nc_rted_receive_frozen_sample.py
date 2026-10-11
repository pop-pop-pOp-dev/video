#!/usr/bin/env python3
"""Remote receiver for bounded immutable frozen-sample transport."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from nc_rted.frozen_sample_transport import (
    cache_status,
    prepare_transport_root,
    receive_staged,
    task_from_document,
)


def _limits(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max-bytes", type=int, default=80 << 30)
    parser.add_argument("--min-free-bytes", type=int, default=20 << 30)


def _serve() -> None:
    """Keep one model-free receiver resident for an ordered transport window."""
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if not isinstance(request, dict) or request.get("command") not in {"status", "receive"}:
                raise ValueError("unsupported receiver request")
            if request["command"] == "status":
                root = Path(request["root"])
                prepare_transport_root(root, min_free_bytes=request["min_free_bytes"])
                response = cache_status(root, max_bytes=request["max_bytes"],
                                        min_free_bytes=request["min_free_bytes"], next_bytes=request["next_bytes"])
            else:
                task = task_from_document(request["task"])
                response = receive_staged(root=request["root"], staged_tensor=request["staged_tensor"],
                                          staged_binding=request["staged_binding"], task=task,
                                          max_bytes=request["max_bytes"], min_free_bytes=request["min_free_bytes"])
                response["ack"] = json.loads(Path(response["ack_path"]).read_text())
        except (KeyError, TypeError, ValueError, OSError, FrozenSampleTransportError) as error:
            response = {"error": str(error)}
        print(json.dumps(response, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    status = commands.add_parser("status")
    status.add_argument("--root", type=Path, required=True)
    status.add_argument("--next-bytes", type=int, required=True)
    _limits(status)
    receive = commands.add_parser("receive")
    receive.add_argument("--root", type=Path, required=True)
    receive.add_argument("--staged-tensor", type=Path, required=True)
    receive.add_argument("--staged-binding", type=Path, required=True)
    receive.add_argument("--task-json", required=True)
    _limits(receive)
    commands.add_parser("serve")
    args = parser.parse_args()
    if args.command == "serve":
        _serve()
    elif args.command == "status":
        prepare_transport_root(args.root, min_free_bytes=args.min_free_bytes)
        print(json.dumps(cache_status(args.root, max_bytes=args.max_bytes, min_free_bytes=args.min_free_bytes,
                                      next_bytes=args.next_bytes), sort_keys=True))
    elif args.command == "receive":
        try:
            task = task_from_document(json.loads(args.task_json))
        except (json.JSONDecodeError, TypeError) as error:
            parser.error(f"invalid task JSON: {error}")
        print(json.dumps(receive_staged(root=args.root, staged_tensor=args.staged_tensor,
                                         staged_binding=args.staged_binding, task=task,
                                         max_bytes=args.max_bytes, min_free_bytes=args.min_free_bytes), sort_keys=True))


if __name__ == "__main__":
    main()
