#!/usr/bin/env python3
"""Standalone NC-RTED queue controller; never receives test labels or metrics."""
import argparse, contextlib, hashlib, json, os, signal, sqlite3, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nc_rted.queue import ArtifactReadUncertain, HardLimit, JobQueue, QueueError
from nc_rted.recovery import CheckpointReadUncertain, RecoveryError
from nc_rted.worker_runtime import acquire_supervisor_lock, attempt_state, gpu_lock, group_live, group_members, group_state, process_live, process_starttime, write_journal

class ConclusiveOutputFailure(QueueError):
    """The exited producer's publication is present but invalid."""

class PublicationUncertain(QueueError):
    """I/O or atomic-commit failure; retain the attempt lease for retry."""

def poll_seconds(payload):
    value = payload.get("poll_seconds", 5)
    return min(5, max(.05, float(value))) if isinstance(value, (int, float)) else 5

def controller_may_live(owner):
    """Default owners include host:pid; do not steal from a live controller."""
    try:
        parts=owner.split(":")
        if len(parts) not in {2,3} or (len(parts)==3 and parts[1] != "recover"): return True
        host, raw_pid = parts[0], parts[-1]; pid = int(raw_pid)
        if host != os.uname().nodename: return True
        os.kill(pid, 0)
        return True
    except (ValueError, ProcessLookupError):
        return False

def terminate_group(pid, starttime, known_members=None, timeout=30):
    """Signal only identity-verified members; unknown group state is retained."""
    if attempt_state(pid, starttime, os.uname().nodename) not in {"live", "group_live"}:
        return False
    def signal_verified(signum):
        members=group_members(pid)
        if members is None or (known_members is not None and not set(members.items()) <= set(known_members.items())):
            return None
        if not members: return {}
        descriptors=[]
        try:
            for member, member_start in members.items():
                descriptor=os.pidfd_open(member)
                if process_starttime(member) != member_start:
                    return None
                descriptors.append(descriptor)
            for descriptor in descriptors: signal.pidfd_send_signal(descriptor, signum)
        except (AttributeError, OSError):
            return None
        finally:
            for descriptor in descriptors: os.close(descriptor)
        return members
    if signal_verified(signal.SIGTERM) is None: return False
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        members=group_members(pid)
        if members == {}: return True
        if members is None or (known_members is not None and not set(members.items()) <= set(known_members.items())): return False
        time.sleep(.2)
    if signal_verified(signal.SIGKILL) is None: return False
    deadline=time.monotonic()+5
    while time.monotonic()<deadline:
        members=group_members(pid)
        if members == {}: return True
        if members is None or (known_members is not None and not set(members.items()) <= set(known_members.items())): return False
        time.sleep(.1)
    return False

def observe_progress(queue, job, path, owner=None):
    if not path or not Path(path).is_file(): return
    try:
        record=json.loads(Path(path).read_text())
        if not isinstance(record,dict): raise ValueError
        counter=record["counter"]
        if (not isinstance(counter,int) or counter < 0 or record.get("job_key") != job["job_key"] or
                record.get("lease_token") != job["lease_token"]): raise ValueError
        evidence=Path(record.get("committed_path", ""))
        root=attempt_directory(json.loads(job["payload"]),job).resolve()
        if not evidence.is_file() or root not in evidence.resolve().parents:
            raise ValueError
        if evidence.resolve() == Path(path).resolve(): raise ValueError
        committed=json.loads(evidence.read_text())
        if not isinstance(committed,dict): raise ValueError
        if (committed.get("schema") != "nc_rted_progress_commit_v1" or committed.get("job_key") != job["job_key"] or
                committed.get("lease_token") != job["lease_token"] or committed.get("counter") != counter):
            raise ValueError
        artifact=Path(committed.get("artifact_path", ""))
        digest=committed.get("artifact_sha256")
        if (not artifact.is_file() or artifact.resolve() == Path(path).resolve() or root not in artifact.resolve().parents or not isinstance(digest,str) or len(digest)!=64 or
                hashlib.sha256(artifact.read_bytes()).hexdigest() != digest): raise ValueError
        transaction=committed.get("transaction_type")
        artifact_document=json.loads(artifact.read_text())
        if not isinstance(artifact_document,dict): raise ValueError
        if transaction == "checkpoint":
            payload=artifact.parent / "state.pt"
            expected=json.loads(job["payload"]).get("run_identity")
            if (artifact_document.get("schema") != "nc_rted_checkpoint_v2" or artifact_document.get("completed_updates") != counter or
                    not isinstance(expected,dict) or artifact_document.get("identity") != expected or not payload.is_file() or payload.resolve() == Path(path).resolve() or artifact_document.get("payload_bytes") != payload.stat().st_size or artifact_document.get("payload_sha256") != hashlib.sha256(payload.read_bytes()).hexdigest()): raise ValueError
            from nc_rted.recovery import validate_checkpoint_payload
            validate_checkpoint_payload(payload, artifact_document)
        elif transaction == "media":
            ids=artifact_document.get("committed_ids")
            results=artifact_document.get("results")
            if (artifact.resolve() == Path(path).resolve() or artifact_document.get("schema") != "nc_rted_media_commit_v1" or artifact_document.get("job_key") != job["job_key"] or artifact_document.get("lease_token") != job["lease_token"] or artifact_document.get("input_hash") != job.get("input_hash") or not isinstance(ids,list) or not ids or len(ids) != counter or len(set(ids)) != len(ids) or not isinstance(results,list) or len(results) != counter): raise ValueError
            result_ids=[]
            for result in results:
                if not isinstance(result,dict) or not isinstance(result.get("id"),str) or not result["id"]: raise ValueError
                candidate=Path(result.get("path", "")); checksum=result.get("sha256")
                if (not candidate.is_file() or root not in candidate.resolve().parents or candidate.resolve() == Path(path).resolve() or not isinstance(checksum,str) or len(checksum)!=64 or hashlib.sha256(candidate.read_bytes()).hexdigest()!=checksum): raise ValueError
                result_ids.append(result["id"])
            if result_ids != ids or len(set(result_ids)) != len(result_ids): raise ValueError
        else: raise ValueError
        queue.record_progress(job["job_key"],job["lease_token"],counter,owner=owner)
    except (OSError, ValueError, KeyError, TypeError, AttributeError, json.JSONDecodeError, RecoveryError):
        return

def artifact_records(payload):
    records=[]
    for expected in payload["expected_outputs"]:
        artifact=Path(expected["path"])
        if not artifact.is_file(): raise RuntimeError(f"missing declared output {artifact}")
        records.append({"path":str(artifact),"checksum":hashlib.sha256(artifact.read_bytes()).hexdigest()})
    JobQueue.validate_artifacts(payload["expected_outputs"], records)
    return records

def attempt_directory(payload, job):
    return Path(payload["run_dir"]) / job["job_key"] / f"attempt-{job['attempts']}-{job['lease_token']}"

def output_records(payload, job):
    root=attempt_directory(payload,job).resolve(); records=[]
    for expected in payload["expected_outputs"]:
        try: raw=Path(expected["path"])
        except (OSError, TypeError) as exc: raise PublicationUncertain("cannot resolve declared output path") from exc
        if raw.is_absolute(): raise ConclusiveOutputFailure("attempt outputs must use relative paths")
        try:
            artifact=(root/raw).resolve()
            if root not in artifact.parents or not artifact.is_file(): raise ConclusiveOutputFailure("missing attempt-local declared output")
            if artifact.is_symlink(): raise ConclusiveOutputFailure("attempt output cannot be a symlink")
            with artifact.open("rb") as handle: os.fsync(handle.fileno())
            records.append({"path":str(artifact),"checksum":hashlib.sha256(artifact.read_bytes()).hexdigest()})
        except OSError as exc: raise PublicationUncertain("cannot read declared output") from exc
    try:
        descriptor=os.open(root,os.O_DIRECTORY); os.fsync(descriptor); os.close(descriptor)
    except OSError as exc: raise PublicationUncertain("cannot sync attempt output directory") from exc
    translated=[{**expected,"path":str((root/Path(expected['path'])).resolve())} for expected in payload["expected_outputs"]]
    try: JobQueue.validate_artifacts(translated,records)
    except (ArtifactReadUncertain, CheckpointReadUncertain) as exc: raise PublicationUncertain(str(exc)) from exc
    except OSError as exc: raise PublicationUncertain("cannot inspect declared output metadata") from exc
    except QueueError as exc: raise ConclusiveOutputFailure(str(exc)) from exc
    return records

def sync_attempt_publication(run_dir, records, completion):
    """Durably retain all declared files and their ancestry before SQL success."""
    root=Path(run_dir).resolve()
    paths=[Path(record["path"]) for record in records] + [Path(completion)]
    for path in paths:
        with path.open("rb") as handle: os.fsync(handle.fileno())
        parent=path.parent.resolve()
        while True:
            descriptor=os.open(parent,os.O_DIRECTORY)
            try: os.fsync(descriptor)
            finally: os.close(descriptor)
            if parent == root: break
            if root not in parent.parents: raise QueueError("publication path escaped attempt root")
            parent=parent.parent
    # Make the attempt directory itself reachable from its job/run parents.
    parent=root.parent
    # The worker may have created run_dir and arbitrary missing parents. Sync
    # all ancestor entries through the filesystem root before SQL success.
    while True:
        descriptor=os.open(parent,os.O_DIRECTORY)
        try: os.fsync(descriptor)
        finally: os.close(descriptor)
        if parent == parent.parent: break
        parent=parent.parent

def commit_outputs(queue, job, payload, owner=None):
    if job.get("kind") == "prediction" and not any(item.get("artifact_type") == "prediction" and item.get("semantic") == "prediction" for item in payload.get("expected_outputs", [])):
        raise ConclusiveOutputFailure("prediction job lacks required prediction provenance contract")
    run_dir=attempt_directory(payload, job)
    try: run_dir.mkdir(parents=True,exist_ok=True)
    except OSError as exc: raise PublicationUncertain("cannot create attempt publication directory") from exc
    temporary, final=run_dir / "result.tmp", run_dir / "result.json"
    records=output_records(payload,job)
    completion=run_dir / "producer_completion.json"
    try: produced=json.loads(completion.read_text())
    except FileNotFoundError as exc: raise ConclusiveOutputFailure("missing producer completion") from exc
    except OSError as exc: raise PublicationUncertain("cannot read producer completion") from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc: raise ConclusiveOutputFailure("missing valid attempt-bound producer completion") from exc
    if not isinstance(produced,dict): raise ConclusiveOutputFailure("producer completion is not an object")
    if (produced.get("job_key") != job["job_key"] or produced.get("lease_token") != job["lease_token"] or
            produced.get("input_hash") != job.get("input_hash") or produced.get("artifacts") != records):
        raise ConclusiveOutputFailure("producer completion does not bind this attempt and inputs")
    try:
        with completion.open("rb") as handle: os.fsync(handle.fileno())
    except OSError as exc: raise PublicationUncertain("cannot sync producer completion") from exc
    # Formal checkpoint has an implicit payload that must be durable too.
    payload_records=list(records)
    for record in records:
        if Path(record["path"]).name == "manifest.json":
            state=Path(record["path"]).parent / "state.pt"
            if state.is_file():
                try: payload_records.append({"path":str(state),"checksum":hashlib.sha256(state.read_bytes()).hexdigest()})
                except OSError as exc: raise PublicationUncertain("cannot read checkpoint payload for publication") from exc
    try: sync_attempt_publication(run_dir, payload_records, completion)
    except OSError as exc: raise PublicationUncertain("cannot durably sync attempt publication") from exc
    durable = temporary if temporary.exists() else final if final.exists() else None
    if durable is not None:
        try: published=json.loads(durable.read_text())
        except OSError as exc: raise PublicationUncertain("cannot read durable result record") from exc
        except (json.JSONDecodeError, UnicodeDecodeError) as exc: raise ConclusiveOutputFailure("invalid durable result record") from exc
        if not isinstance(published,dict): raise ConclusiveOutputFailure("durable result record is not an object")
        if published.get("job_key") != job["job_key"] or published.get("lease_token") != job["lease_token"] or published.get("artifacts") != records:
            raise ConclusiveOutputFailure("durable result record does not match current attempt")
    else:
        try: write_journal(temporary, {"job_key":job["job_key"],"lease_token":job["lease_token"],"artifacts":records,"completed_at":time.time()})
        except OSError as exc: raise PublicationUncertain("cannot write durable result record") from exc
    try: queue.commit(job["job_key"],job["lease_token"],temporary,final,owner=owner)
    except HardLimit:
        raise
    except (OSError, sqlite3.Error, RuntimeError, QueueError) as exc: raise PublicationUncertain("atomic output publication failed") from exc

def durable_publication_exists(payload, job):
    run=attempt_directory(payload,job)
    return any((run/name).is_file() for name in ("producer_completion.json", "result.tmp", "result.json"))

def progress_timeout(payload):
    p99 = payload.get("progress_p99_seconds")
    if not isinstance(p99, (int, float)) or p99 <= 0: return 1800
    return max(1800, 5 * p99)

def monitor(queue, job, process, payload, owner):
    """Heartbeat proves ownership; progress timeout catches a stalled live child."""
    while process.poll() is None:
        members=group_members(process.pid)
        if members is None: raise QueueError("process group observation unknown")
        process.known_members.update(members)
        if getattr(process,"journal_path",None):
            queue.update_attempt_journal(job["job_key"],job["lease_token"],process.pid,job["process_starttime"],str(payload.get("physical_gpu")),process.journal_path,member_identities=process.known_members)
        time.sleep(poll_seconds(payload))
        observe_progress(queue, job, payload.get("progress_path"), owner); queue.runtime_guard(job["job_key"], job["lease_token"])
        enforce_progress_timeout(queue, job, payload)
        queue.heartbeat(job["job_key"], job["lease_token"], pid=process.pid, process_starttime=process_starttime(process.pid), owner=owner)

def enforce_progress_timeout(queue, job, payload):
    row = next(row for row in queue.status() if row["job_key"] == job["job_key"])
    last_progress = row.get("progress_at") or queue.attempt_started_at(job["job_key"], job["lease_token"])
    if last_progress and time.time() - last_progress > progress_timeout(payload):
        # A stale counter alone is not permission to terminate normal long work.
        queue.protected_live(job["job_key"], job["lease_token"], "suspected progress stall; verify long-input work before termination", owner=job.get("lease_owner"))

def adopt_live(queue, owner):
    """Reattach after controller death; never launch a second same-attempt child."""
    host = owner.split(":", 1)[0]
    for job in queue.running_attempts():
        if job.get("lease_owner", "").split(":", 1)[0] != host: continue
        state=attempt_state(job.get("pid"), job.get("process_starttime"), host)
        if state not in {"live", "group_live"}: continue
        try: lock=acquire_supervisor_lock(attempt_directory(json.loads(job["payload"]), job))
        except OSError: continue
        if lock is False: continue
        queue.take_supervision(job["job_key"], job["lease_token"], owner)
        job=next(candidate for candidate in queue.running_attempts() if candidate["job_key"] == job["job_key"] and candidate["lease_token"] == job["lease_token"])
        job["lease_owner"] = owner
        state=attempt_state(job.get("pid"), job.get("process_starttime"), host)
        if state not in {"live", "group_live"}: lock.close(); continue
        try: known={int(k):str(v) for k,v in json.loads(Path(job["journal_path"]).read_text()).get("member_identities",{}).items()}
        except (OSError, ValueError, TypeError): known={}
        if state == "group_live" and not known: lock.close(); continue
        leader_before=process_starttime(job["pid"]) if state == "live" else None
        current=group_members(job["pid"])
        # A verified live leader authenticates newly spawned descendants; once
        # it is gone, only the prior durable snapshot can establish continuity.
        if (current is None or (state == "live" and leader_before != job["process_starttime"]) or
                (state == "live" and process_starttime(job["pid"]) != leader_before) or
                (state == "group_live" and not set(current.items()) <= set(known.items()))): lock.close(); continue
        known.update(current)
        payload=json.loads(job["payload"])
        queue.update_attempt_journal(job["job_key"],job["lease_token"],job["pid"],job["process_starttime"],str(payload.get("physical_gpu")),job["journal_path"],member_identities=known)
        process = type("Adopted", (), {"pid":job["pid"], "group_only":state == "group_live", "known_members":known, "journal_path":Path(job["journal_path"]), "lock":lock, "poll":lambda self: None})()
        # The controller cannot obtain a child's return code after adoption; its
        # durable output contract decides success once the PID exits.
        return job, payload, process
    return None

def reconcile_exited_attempts(queue, owner):
    """Finish a durable child outcome after its supervising controller died."""
    host=owner.split(":",1)[0]
    for job in queue.running_attempts():
        if job.get("lease_owner", "").split(":",1)[0] != host: continue
        if job.get("state") == "LAUNCHING":
            try: lock=acquire_supervisor_lock(attempt_directory(json.loads(job["payload"]),job))
            except OSError: continue
            if lock is False: continue
            queue.take_supervision(job["job_key"],job["lease_token"],owner); job["lease_owner"]=owner
            queue.protected_live(job["job_key"], job["lease_token"], "launch intent has no durable child identity", owner=owner)
            lock.close()
            continue
        state = attempt_state(job.get("pid"), job.get("process_starttime"), host)
        if state == "live": continue
        if state == "group_live":
            continue
        try: lock=acquire_supervisor_lock(attempt_directory(json.loads(job["payload"]),job))
        except OSError: continue
        if lock is False: continue
        queue.take_supervision(job["job_key"],job["lease_token"],owner); job["lease_owner"]=owner
        # The job row is policy authority. Re-read it after ownership transfer
        # so a stop written by a prior controller cannot be bypassed by this
        # controller's pre-lock snapshot.
        job=next(candidate for candidate in queue.running_attempts() if candidate["job_key"] == job["job_key"] and candidate["lease_token"] == job["lease_token"])
        if state in {"foreign", "unknown", "identity_mismatch"}:
            queue.protected_live(job["job_key"], job["lease_token"], "PID identity mismatch; child outcome unknown", owner=owner)
            lock.close()
            continue
        if job.get("protective_stop_code"):
            queue.fail(job["job_key"],job["lease_token"],f"hard limit: {job['protective_stop_code']}",owner=owner,code=job["protective_stop_code"])
            lock.close(); continue
        payload=json.loads(job["payload"])
        try:
            commit_outputs(queue,job,payload,owner)
        except HardLimit as error:
            queue.fail(job["job_key"],job["lease_token"],f"hard limit: {error.code}",owner=owner,code=error.code)
        except ConclusiveOutputFailure as error:
                queue.fail(job["job_key"],job["lease_token"],f"exited child reconciliation: {error}",owner=owner)
        except PublicationUncertain as error:
            queue.protected_live(job["job_key"],job["lease_token"],f"reconciliation publication uncertain: {error}",owner=owner)
        except OSError as error:
            queue.protected_live(job["job_key"],job["lease_token"],f"reconciliation publication uncertain: {error}",owner=owner)
        finally:
            lock.close()

def worker(queue, owner, once):
    while True:
        reconcile_exited_attempts(queue,owner); queue.recover_expired(); adopted=adopt_live(queue, owner)
        if adopted:
            job,payload,process=adopted
            try:
                if job.get("protective_stop_code"):
                    code=job["protective_stop_code"]
                    if not terminate_group(process.pid, job["process_starttime"], process.known_members):
                        queue.protected_live(job["job_key"],job["lease_token"],f"protective stop retained pending verified exit: {code}",owner=owner)
                        return 1
                    queue.fail(job["job_key"],job["lease_token"],f"hard limit: {code}",owner=owner,code=code)
                    if once: return 1
                    continue
                while group_state(process.pid) == "live":
                    members=group_members(process.pid)
                    if members is None: raise QueueError("process group observation unknown")
                    # While the leader identity remains verified, descendants
                    # become part of this attempt's continuity snapshot.
                    if process_live(process.pid, job["process_starttime"], owner.split(":",1)[0]):
                        process.known_members.update(members)
                        queue.update_attempt_journal(job["job_key"],job["lease_token"],process.pid,job["process_starttime"],str(payload.get("physical_gpu")),process.journal_path,member_identities=process.known_members)
                    time.sleep(poll_seconds(payload)); observe_progress(queue,job,payload.get("progress_path")); queue.runtime_guard(job["job_key"], job["lease_token"]); enforce_progress_timeout(queue, job, payload); queue.heartbeat(job["job_key"],job["lease_token"],pid=process.pid,process_starttime=job["process_starttime"],owner=owner)
            except Exception as exc:
                hard = isinstance(exc, HardLimit)
                if hard: queue.record_protective_stop(job["job_key"],job["lease_token"],exc.code,owner=owner)
                if not hard or not terminate_group(process.pid, job["process_starttime"], getattr(process,"known_members",None)):
                    queue.protected_live(job["job_key"], job["lease_token"], f"protective adopted-child failure: {exc}", owner=owner)
                    # A supervisor must restart us; never signal success while
                    # leaving a child that requires continued supervision.
                    return 1
                queue.fail(job["job_key"], job["lease_token"], str(exc),owner=owner,code=exc.code if isinstance(exc,HardLimit) else "transient")
                if once: return 1
                continue
            process.lock.close()
            reconcile_exited_attempts(queue,owner)
            if group_state(process.pid) != "gone":
                return 1
            if once: return 0
            continue
        job = queue.claim(owner)
        if not job:
            if once: return 0
            time.sleep(60); continue
        payload = json.loads(job["payload"]); command = payload["command"]
        if "run_dir" not in payload: raise QueueError("worker payload requires data-volume run_dir")
        run_dir = attempt_directory(payload, job)
        run_dir.mkdir(parents=True, exist_ok=True); temporary = run_dir / "result.tmp"; final = run_dir / "result.json"
        process = None; supervisor = None; journal_path = run_dir / f"attempt-{job['attempts']}.json"
        try:
            supervisor=acquire_supervisor_lock(run_dir)
            if supervisor is False: raise QueueError("attempt supervisor busy")
            with gpu_lock(payload.get("data_volume", queue.path.parent), payload.get("physical_gpu")) as lock_handle:
                if lock_handle is False:
                    queue.release_unstarted(job["job_key"], job["lease_token"], "GPU lock busy")
                    if once: return 0
                    continue
                # This durable intent closes the crash window before Popen. A
                # restarted worker protects it rather than risking a duplicate.
                queue.start_attempt_journal(job["job_key"], job["lease_token"], None, None, str(payload.get("physical_gpu")), journal_path, state="LAUNCHING")
                environment=dict(os.environ); environment["CUDA_VISIBLE_DEVICES"]=str(payload["physical_gpu"])
                environment["NC_RTED_PRODUCER_COMPLETION"]=str(run_dir / "producer_completion.json")
                environment["NC_RTED_JOB_KEY"]=job["job_key"]; environment["NC_RTED_LEASE_TOKEN"]=job["lease_token"]; environment["NC_RTED_INPUT_HASH"]=job["input_hash"]
                process = subprocess.Popen(command, cwd=run_dir, start_new_session=True, env=environment, pass_fds=(() if lock_handle is None else (lock_handle.fileno(),)))
                starttime=process_starttime(process.pid)
                if starttime is None: raise RuntimeError("child PID disappeared before identity capture")
                job["process_starttime"] = starttime
                process.known_members=group_members(process.pid)
                process.journal_path=journal_path
                queue.heartbeat(job["job_key"], job["lease_token"], pid=process.pid, process_starttime=starttime, owner=owner)
                queue.update_attempt_journal(job["job_key"], job["lease_token"], process.pid, starttime, str(payload.get("physical_gpu")), journal_path, member_identities=process.known_members)
                monitor(queue,job,process,payload,owner)
                if process.returncode: raise RuntimeError(f"command exited {process.returncode}")
                state=group_state(process.pid)
                if state != "gone":
                    raise QueueError("process group remains live" if state == "live" else "process group observation unknown")
                commit_outputs(queue,job,payload,owner)
        except Exception as exc:
            protective=isinstance(exc, HardLimit)
            if protective:
                # Persist the outcome before signalling so recovery cannot
                # promote a completion produced after the hard limit was seen.
                queue.record_protective_stop(job["job_key"],job["lease_token"],exc.code,owner=owner)
            if process is not None and group_state(process.pid) != "gone":
                if not protective or not terminate_group(process.pid, job.get("process_starttime"), getattr(process,"known_members",None)):
                    queue.protected_live(job["job_key"],job["lease_token"],f"child/process group retained after worker error: {exc}",owner=owner)
                    return 1
            if protective:
                queue.fail(job["job_key"], job["lease_token"], str(exc),owner=owner,code=exc.code)
                return 1
            if durable_publication_exists(payload,job):
                queue.protected_live(job["job_key"],job["lease_token"],f"durable completion requires reconciliation: {exc}",owner=owner)
                return 1
            queue.fail(job["job_key"], job["lease_token"], str(exc),owner=owner,code=exc.code if isinstance(exc,HardLimit) else "transient")
        finally:
            if supervisor not in (None, False): supervisor.close()
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
    elif args.action == "recover":
        reconcile_exited_attempts(queue, f"{os.uname().nodename}:recover:{os.getpid()}")
        print(queue.recover_expired())
    elif args.action == "reconcile": print(queue.reconcile())
    elif args.action == "evidence": queue.add_evidence(args.name, args.path, args.checksum, args.schema, args.input_code_hash, args.accepted)
    elif args.action == "complete-gate": queue.complete_gate(args.job_key)
    else: return worker(queue, args.owner, args.once)
if __name__ == "__main__":
    try: raise SystemExit(main())
    except QueueError as exc: raise SystemExit(f"queue error: {exc}")
