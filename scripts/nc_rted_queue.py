#!/usr/bin/env python3
"""Standalone NC-RTED queue controller; never receives test labels or metrics."""
import argparse, contextlib, fcntl, hashlib, json, os, signal, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nc_rted.queue import JobQueue, QueueError

def proc_starttime(pid):
    try: return Path(f"/proc/{pid}/stat").read_text().split()[21]
    except OSError: return None

def terminate_group(process, timeout=30):
    """Hard limits retain ownership until the detached process group is dead."""
    if process.poll() is not None: return True
    os.killpg(process.pid, signal.SIGTERM)
    deadline=time.monotonic()+timeout
    while process.poll() is None and time.monotonic()<deadline: time.sleep(.2)
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGKILL)
        deadline=time.monotonic()+5
        while process.poll() is None and time.monotonic()<deadline: time.sleep(.1)
    return process.poll() is not None

def observe_progress(queue, job, path):
    if not path or not Path(path).is_file(): return
    try:
        counter=json.loads(Path(path).read_text())["counter"]
        if not isinstance(counter,int) or counter < 0: raise ValueError
        queue.record_progress(job["job_key"],job["lease_token"],counter)
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return

@contextlib.contextmanager
def gpu_lock(gpu):
    """One local worker owns a CUDA device until its detached child exits."""
    if gpu is None:
        yield None; return
    path = Path("/tmp") / f"nc_rted_gpu_{os.uname().nodename}_{gpu}.lock"
    with path.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield handle
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

def worker(queue, owner, once):
    while True:
        queue.recover_expired(); job = queue.claim(owner)
        if not job:
            if once: return 0
            time.sleep(60); continue
        payload = json.loads(job["payload"]); command = payload["command"]
        run_dir = Path(payload.get("run_dir", "state/nc_rted_runs")) / job["job_key"]
        run_dir.mkdir(parents=True, exist_ok=True); temporary = run_dir / "result.tmp"; final = run_dir / "result.json"
        process = None
        try:
            with gpu_lock(payload.get("physical_gpu")) as lock_handle:
                environment=dict(os.environ); environment["CUDA_VISIBLE_DEVICES"]=str(payload["physical_gpu"])
                process = subprocess.Popen(command, cwd=run_dir, start_new_session=True, env=environment, pass_fds=(() if lock_handle is None else (lock_handle.fileno(),)))
                queue.heartbeat(job["job_key"], job["lease_token"], pid=process.pid, process_starttime=proc_starttime(process.pid))
                queue.start_attempt_journal(job["job_key"], job["lease_token"], process.pid, proc_starttime(process.pid), str(payload["physical_gpu"]), run_dir / f"attempt-{job['attempts']}.json")
                while process.poll() is None:
                    time.sleep(5); observe_progress(queue,job,payload.get("progress_path")); queue.runtime_guard(job["job_key"], job["lease_token"]); queue.heartbeat(job["job_key"], job["lease_token"])
                if process.returncode: raise RuntimeError(f"command exited {process.returncode}")
                artifacts=[]
                for expected in payload["expected_outputs"]:
                    artifact=Path(expected["path"])
                    if not artifact.is_file(): raise RuntimeError(f"missing declared output {artifact}")
                    digest=hashlib.sha256(artifact.read_bytes()).hexdigest()
                    if expected.get("checksum") and digest != expected["checksum"]: raise RuntimeError(f"output checksum mismatch {artifact}")
                    artifacts.append({"path":str(artifact),"checksum":digest})
                queue.validate_artifacts(payload["expected_outputs"], artifacts)
                temporary.write_text(json.dumps({"job_key": job["job_key"], "artifacts":artifacts, "completed_at": time.time()}) + "\n")
                queue.commit(job["job_key"], job["lease_token"], temporary, final)
        except Exception as exc:
            protective=any(marker in str(exc).lower() for marker in ("deadline", "disk hard", "budget hard", "nan", "integrity", "leakage"))
            if 'process' in locals() and process.poll() is None and protective:
                if not terminate_group(process):
                    queue.protected_live(job["job_key"],job["lease_token"],f"protective failure; child still live: {exc}")
                    if once: return 1
                    continue
            queue.fail(job["job_key"], job["lease_token"], str(exc))
        if once: return 0

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--db", default="state/nc_rted/tasks.sqlite3")
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("init"); sub.add_parser("status"); sub.add_parser("recover"); sub.add_parser("reconcile")
    ev = sub.add_parser("evidence"); ev.add_argument("name"); ev.add_argument("path"); ev.add_argument("--checksum")
    ev.add_argument("--schema", default=""); ev.add_argument("--input-code-hash", default=""); ev.add_argument("--accepted", action="store_true")
    gate = sub.add_parser("complete-gate"); gate.add_argument("job_key")
    wk = sub.add_parser("worker"); wk.add_argument("--owner", default=f"{os.uname().nodename}:{os.getpid()}"); wk.add_argument("--once", action="store_true")
    args = parser.parse_args(); queue = JobQueue(args.db)
    if args.action == "init": queue.register_matrix(); print(json.dumps({"registered": 28, "formal_runs": 12, "formal_jobs_claimable": False}))
    elif args.action == "status": print(json.dumps(queue.status(), indent=2))
    elif args.action == "recover": print(queue.recover_expired())
    elif args.action == "reconcile": print(queue.reconcile())
    elif args.action == "evidence": queue.add_evidence(args.name, args.path, args.checksum, args.schema, args.input_code_hash, args.accepted)
    elif args.action == "complete-gate": queue.complete_gate(args.job_key)
    else: return worker(queue, args.owner, args.once)
if __name__ == "__main__":
    try: raise SystemExit(main())
    except QueueError as exc: raise SystemExit(f"queue error: {exc}")
