"""Small, transactional control plane for accepted NC-RTED work only."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import stat
import sqlite3
import time
import uuid
import sys
from contextlib import contextmanager
from pathlib import Path

from .resource_attestation import ResourceAttestationError, resource_lease_expiry, verify_attestation
from .worker_runtime import HeldGpuLock

PENDING, RUNNING, RETRY_WAIT, SUCCEEDED, BLOCKED = "PENDING", "RUNNING", "RETRY_WAIT", "SUCCEEDED", "BLOCKED"
RETRY_DELAYS = (300, 1200, 3600)
FORMAL_GROUPS = ("A", "U", "S", "F")
SEEDS = (17, 42, 2026)
FORMAL_DEADLINE = 1792724040  # 2026-10-23T02:54:00Z


def canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class QueueError(RuntimeError): pass
class ArtifactReadUncertain(QueueError): pass
class LeaseLost(QueueError): pass
class HardLimit(QueueError):
    """A structured, evidence-backed condition allowed to stop a child."""
    def __init__(self, code: str): self.code=code; super().__init__(code)


def after_attempt_journal_write():
    """Test seam for the file-before-SQL crash boundary."""


def _canonical_gpu_lock(payload: dict) -> Path:
    return (Path(payload["data_volume"]) / ".nc_rted_locks" /
            f"gpu_{socket.gethostname()}_{payload['physical_gpu']}.lock")


def _held_flock(lock_stat: os.stat_result, pid: int) -> bool:
    """Prove a particular process fd holds an exclusive flock for this inode."""
    try:
        for entry in Path(f"/proc/{pid}/fd").iterdir():
            if _fd_holds_flock(lock_stat, pid, entry.name): return True
    except (OSError, ValueError):
        return False
    return False


def _fd_holds_flock(lock_stat: os.stat_result, pid: int, descriptor: int | str) -> bool:
    """Prove this exact descriptor, rather than another same-inode fd, holds flock."""
    try:
        entry=Path(f"/proc/{pid}/fd/{descriptor}")
        current=os.stat(entry)
        if current.st_dev != lock_stat.st_dev or current.st_ino != lock_stat.st_ino: return False
        info=(Path(f"/proc/{pid}/fdinfo/{descriptor}")).read_text().splitlines()
        return any({"FLOCK", "ADVISORY", "WRITE"}.issubset(line.split()) for line in info)
    except (OSError, ValueError):
        return False


def _held_flock_for_current_process(lock_stat: os.stat_result) -> bool:
    return _held_flock(lock_stat, os.getpid())


class JobQueue:
    def __init__(self, database: str | Path):
        self.path = Path(database)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def connect(self):
        db = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA busy_timeout=30000")
        return db

    def initialize(self):
        with self.connect() as db:
            db.executescript("""
            CREATE TABLE IF NOT EXISTS jobs (
              id INTEGER PRIMARY KEY, job_key TEXT UNIQUE NOT NULL, kind TEXT NOT NULL,
              input_hash TEXT NOT NULL, payload TEXT NOT NULL, status TEXT NOT NULL,
              attempts INTEGER NOT NULL DEFAULT 0, max_attempts INTEGER NOT NULL DEFAULT 3,
              not_before REAL NOT NULL DEFAULT 0, lease_owner TEXT, lease_token TEXT,
              lease_expires REAL, heartbeat REAL, pid INTEGER, process_starttime TEXT,
              output_path TEXT, output_checksum TEXT, failure TEXT, protective_stop_code TEXT,
              progress_at REAL, progress_counter INTEGER,
              created_at REAL NOT NULL, updated_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS dependencies (job_id INTEGER NOT NULL REFERENCES jobs(id), depends_on INTEGER NOT NULL REFERENCES jobs(id), PRIMARY KEY(job_id, depends_on));
            CREATE TABLE IF NOT EXISTS evidence (name TEXT PRIMARY KEY, checksum TEXT NOT NULL, path TEXT NOT NULL, schema_name TEXT NOT NULL DEFAULT '', input_code_hash TEXT NOT NULL DEFAULT '', accepted INTEGER NOT NULL DEFAULT 0, verified_at REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS attempts (id INTEGER PRIMARY KEY, job_id INTEGER NOT NULL REFERENCES jobs(id), number INTEGER NOT NULL, owner TEXT NOT NULL, lease_token TEXT NOT NULL, started_at REAL NOT NULL, ended_at REAL, outcome TEXT, detail TEXT);
            CREATE TABLE IF NOT EXISTS attempt_journal (lease_token TEXT PRIMARY KEY, job_key TEXT NOT NULL, pid INTEGER, process_starttime TEXT, gpu_identity TEXT, journal_path TEXT NOT NULL, state TEXT NOT NULL, updated_at REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS reservations (lease_token TEXT PRIMARY KEY, job_key TEXT NOT NULL, lease_id TEXT NOT NULL, host TEXT NOT NULL, physical_gpu INTEGER NOT NULL, lock_device INTEGER NOT NULL, lock_inode INTEGER NOT NULL, acquired_at REAL NOT NULL, authorized_end REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS terminal_evidence (lease_token TEXT PRIMARY KEY, job_key TEXT NOT NULL, pid INTEGER NOT NULL, process_starttime TEXT NOT NULL, observed_at REAL NOT NULL);
            """)
            for statement in ("ALTER TABLE evidence ADD COLUMN schema_name TEXT NOT NULL DEFAULT ''", "ALTER TABLE evidence ADD COLUMN input_code_hash TEXT NOT NULL DEFAULT ''", "ALTER TABLE evidence ADD COLUMN accepted INTEGER NOT NULL DEFAULT 0"):
                try: db.execute(statement)
                except sqlite3.OperationalError: pass
            for statement in ("ALTER TABLE jobs ADD COLUMN progress_at REAL", "ALTER TABLE jobs ADD COLUMN progress_counter INTEGER", "ALTER TABLE jobs ADD COLUMN protective_stop_code TEXT"):
                try: db.execute(statement)
                except sqlite3.OperationalError: pass
            reservation_columns={row["name"] for row in db.execute("PRAGMA table_info(reservations)")}
            required_reservations={"lease_token","job_key","lease_id","host","physical_gpu","lock_device","lock_inode","acquired_at","authorized_end"}
            if reservation_columns != required_reservations:
                # Old rows lack the held-lock inode and fixed end. Preserve
                # them for forensic recovery, but never reconstruct authority.
                legacy=f"reservations_legacy_unverified_{int(time.time() * 1_000_000)}"
                db.execute(f'ALTER TABLE reservations RENAME TO "{legacy}"')
                db.execute("""CREATE TABLE reservations (
                  lease_token TEXT PRIMARY KEY, job_key TEXT NOT NULL, lease_id TEXT NOT NULL,
                  host TEXT NOT NULL, physical_gpu INTEGER NOT NULL, lock_device INTEGER NOT NULL,
                  lock_inode INTEGER NOT NULL, acquired_at REAL NOT NULL, authorized_end REAL NOT NULL
                )""")

    @contextmanager
    def transaction(self, db):
        db.execute("BEGIN IMMEDIATE")
        try:
            yield
        except Exception:
            db.execute("ROLLBACK"); raise
        else: db.execute("COMMIT")

    def add_evidence(self, name: str, path: str | Path, checksum: str | None = None, schema_name="", input_code_hash="", accepted=False):
        path = Path(path)
        if not path.is_file(): raise QueueError(f"evidence does not exist: {path}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if checksum and checksum != actual: raise QueueError("evidence checksum mismatch")
        with self.connect() as db:
            db.execute("INSERT OR REPLACE INTO evidence(name,checksum,path,schema_name,input_code_hash,accepted,verified_at) VALUES(?,?,?,?,?,?,?)", (name, actual, str(path.resolve()), schema_name, input_code_hash, int(accepted), time.time()))

    def add_job(self, job_key, kind, payload, dependencies=(), max_attempts=4):
        if max_attempts != 4: raise QueueError("NC-RTED permits one initial attempt plus three retries")
        if payload.get("uses_test_metrics") or payload.get("uses_test_labels"):
            raise QueueError("training/control workers cannot register test metric or label access")
        now = time.time(); encoded = json.dumps(payload, sort_keys=True)
        with self.connect() as db, self.transaction(db):
            existing = db.execute("SELECT input_hash,kind FROM jobs WHERE job_key=?", (job_key,)).fetchone()
            if existing:
                if existing["input_hash"] != canonical_hash(payload) or existing["kind"] != kind: raise QueueError(f"conflicting existing job {job_key}")
                return
            db.execute("INSERT INTO jobs(job_key,kind,input_hash,payload,status,max_attempts,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?)", (job_key, kind, canonical_hash(payload), encoded, PENDING, max_attempts, now, now))
            job_id = db.execute("SELECT id FROM jobs WHERE job_key=?", (job_key,)).fetchone()[0]
            for dep in dependencies:
                row = db.execute("SELECT id FROM jobs WHERE job_key=?", (dep,)).fetchone()
                if row is None: raise QueueError(f"unknown dependency {dep}")
                db.execute("INSERT INTO dependencies VALUES(?,?)", (job_id, row[0]))

    def _guard_failure(self, db, row):
        payload = json.loads(row["payload"])
        requirements = payload.get("evidence", [])
        for name in requirements:
            evidence = db.execute("SELECT * FROM evidence WHERE name=?", (name,)).fetchone()
            if evidence is None or not evidence["accepted"]:
                return f"missing verified evidence: {name}"
            path = Path(evidence["path"])
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != evidence["checksum"]: return f"evidence checksum changed: {name}"
        binding = payload.get("required_input_code_hash")
        if binding and any(db.execute("SELECT input_code_hash FROM evidence WHERE name=?", (name,)).fetchone()[0] != binding for name in requirements): return "evidence code-hash binding mismatch"
        min_free = int(payload.get("min_free_bytes", 0))
        volume=Path(payload.get("data_volume",self.path.parent))
        if min_free and shutil.disk_usage(volume).free < min_free:
            return "disk budget guard: less than required free space"
        deadline = payload.get("deadline_utc_epoch")
        if deadline and time.time() >= deadline: return "deadline guard reached"
        if not payload.get("command") and row["kind"] not in {"manual_gate", "audit"}:
            return "no audited concrete executable command"
        if row["kind"] not in {"manual_gate", "audit"} and not payload.get("expected_outputs"):
            return "no declared output validator/checksum contract"
        if row["kind"] == "prediction":
            contracts=payload.get("expected_outputs", [])
            if not any(item.get("artifact_type") == "prediction" and item.get("semantic") == "prediction" for item in contracts):
                return "prediction job lacks required prediction provenance contract"
        if row["kind"] not in {"manual_gate", "audit"} and payload.get("physical_gpu") is None:
            return "no physical CUDA device binding"
        if row["kind"] == "formal_train" and payload.get("command"):
            entry = str((Path(__file__).resolve().parents[2] / "scripts" / "nc_rted_train.py"))
            interpreter = payload.get("interpreter", {})
            required_command = [interpreter.get("path"), entry, "--config", payload.get("runtime_config"), "--config-sha256", payload.get("runtime_config_sha256"), "--mode", "formal", "--admission", payload.get("formal_admission"), "--admission-sha256", payload.get("formal_admission_sha256")]
            if payload.get("command") != required_command:
                return "formal training command is not the fixed local nc_rted_train entrypoint"
            outputs = payload.get("expected_outputs")
            if (not isinstance(outputs, list) or len(outputs) != 1 or outputs[0].get("artifact_type") != "checkpoint" or
                    outputs[0].get("semantic") != "formal_training" or outputs[0].get("run_identity") != payload.get("run_identity")):
                return "formal training requires one exact final checkpoint contract"
            evidence = {
                item["name"]: (item["path"], item["checksum"])
                for item in db.execute("SELECT name,path,checksum FROM evidence WHERE accepted=1")
            }
            try:
                verify_attestation(payload, row["job_key"], evidence)
            except (ResourceAttestationError, OSError, TypeError, ValueError) as error:
                return f"formal resource attestation rejected: {error}"
        return None

    def claim(self, owner: str, lease_seconds=120):
        now = time.time()
        with self.connect() as db, self.transaction(db):
            db.execute("UPDATE jobs SET status=?, updated_at=? WHERE status=? AND not_before<=?", (PENDING, now, RETRY_WAIT, now))
            rows = db.execute("SELECT * FROM jobs WHERE status=? AND not_before<=? AND kind NOT IN ('manual_gate','audit') ORDER BY id", (PENDING, now)).fetchall()
            for row in rows:
                requested_gpu = json.loads(row["payload"]).get("physical_gpu")
                if requested_gpu is not None:
                    active = db.execute("SELECT * FROM jobs WHERE status=? AND id!=?", (RUNNING, row["id"])).fetchall()
                    if any(json.loads(other["payload"]).get("physical_gpu") == requested_gpu and self._process_may_live(other) for other in active):
                        continue
                pending_dep = db.execute("SELECT 1 FROM dependencies d JOIN jobs p ON p.id=d.depends_on WHERE d.job_id=? AND p.status!=?", (row["id"], SUCCEEDED)).fetchone()
                if pending_dep: continue
                failure = self._guard_failure(db, row)
                if failure:
                    db.execute("UPDATE jobs SET status=?,failure=?,updated_at=? WHERE id=?", (BLOCKED, failure, now, row["id"])); continue
                token = uuid.uuid4().hex
                changed = db.execute("UPDATE jobs SET status=?,attempts=attempts+1,lease_owner=?,lease_token=?,lease_expires=?,heartbeat=?,updated_at=? WHERE id=? AND status=?", (RUNNING, owner, token, now + lease_seconds, now, now, row["id"], PENDING)).rowcount
                if changed:
                    db.execute("INSERT INTO attempts(job_id,number,owner,lease_token,started_at) VALUES(?,?,?,?,?)", (row["id"], row["attempts"] + 1, owner, token, now))
                    result = dict(row); result.update(status=RUNNING, lease_token=token, attempts=row["attempts"] + 1); return result
        return None

    def heartbeat(self, job_key, token, lease_seconds=120, pid=None, process_starttime=None, owner=None):
        now = time.time()
        with self.connect() as db:
            query="UPDATE jobs SET heartbeat=?,lease_expires=?,pid=COALESCE(?,pid),process_starttime=COALESCE(?,process_starttime),updated_at=? WHERE job_key=? AND status=? AND lease_token=?"
            args=[now, now + lease_seconds, pid, process_starttime, now, job_key, RUNNING, token]
            if owner is not None: query += " AND lease_owner=?"; args.append(owner)
            changed = db.execute(query, args).rowcount
        if not changed: raise LeaseLost(job_key)

    def take_supervision(self, job_key, token, owner, lease_seconds=120):
        """Fence a displaced controller by atomically replacing lease_owner."""
        now=time.time()
        with self.connect() as db:
            changed=db.execute("UPDATE jobs SET lease_owner=?,heartbeat=?,lease_expires=?,updated_at=? WHERE job_key=? AND status=? AND lease_token=?",(owner,now,now+lease_seconds,now,job_key,RUNNING,token)).rowcount
        if not changed: raise LeaseLost(job_key)

    def record_progress(self, job_key, token, counter: int, owner=None):
        """Progress is separate from liveness; callers must report real checkpoints/media commits."""
        now=time.time()
        with self.connect() as db, self.transaction(db):
            query="SELECT progress_counter FROM jobs WHERE job_key=? AND status=? AND lease_token=?"; args=[job_key,RUNNING,token]
            if owner is not None: query += " AND lease_owner=?"; args.append(owner)
            row=db.execute(query,args).fetchone()
            if row is None: raise LeaseLost(job_key)
            if row["progress_counter"] is not None and counter <= row["progress_counter"]: return False
            db.execute("UPDATE jobs SET progress_at=?,progress_counter=?,updated_at=? WHERE job_key=? AND status=? AND lease_token=? AND (progress_counter IS NULL OR progress_counter<?)",(now,counter,now,job_key,RUNNING,token,counter))
        return True

    def protected_live(self, job_key, token, detail, owner=None):
        """Leave an uncertain/live child reserved; it must never be retried."""
        with self.connect() as db:
            query="UPDATE jobs SET failure=?,updated_at=? WHERE job_key=? AND status=? AND lease_token=?"; args=[detail,time.time(),job_key,RUNNING,token]
            if owner is not None: query += " AND lease_owner=?"; args.append(owner)
            changed=db.execute(query,args).rowcount
        if not changed: raise LeaseLost(job_key)

    def record_protective_stop(self, job_key, token, code, owner=None):
        with self.connect() as db:
            query="UPDATE jobs SET protective_stop_code=?,updated_at=? WHERE job_key=? AND status=? AND lease_token=?"; args=[code,time.time(),job_key,RUNNING,token]
            if owner is not None: query += " AND lease_owner=?"; args.append(owner)
            changed=db.execute(query,args).rowcount
        if not changed: raise LeaseLost(job_key)

    def start_attempt_journal(self, job_key, token, pid, process_starttime, gpu_identity, journal_path, state="RUNNING", member_identities=None):
        """Persist intent before launch, then replace it with the observed PID."""
        record={"job_key":job_key,"lease_token":token,"pid":pid,"process_starttime":process_starttime,"gpu_identity":gpu_identity,"state":state,"member_identities":member_identities or {},"updated_at":time.time()}
        from .worker_runtime import write_journal
        path=Path(journal_path); write_journal(path,record)
        after_attempt_journal_write()
        with self.connect() as db: db.execute("INSERT OR REPLACE INTO attempt_journal VALUES(?,?,?,?,?,?,?,?)",(token,job_key,pid,process_starttime,gpu_identity,str(path),state,time.time()))

    def update_attempt_journal(self, job_key, token, pid, process_starttime, gpu_identity, journal_path, state="RUNNING", member_identities=None):
        self.start_attempt_journal(job_key, token, pid, process_starttime, gpu_identity, journal_path, state, member_identities)

    def repair_launching_journal(self, job_key, token, attempt_number, journal_path):
        """Adopt a durable post-launch identity left before its SQL replacement."""
        expected_path=Path(journal_path).resolve()
        try:
            record=json.loads(expected_path.read_text())
            if not isinstance(record,dict): return False
            pid=record.get("pid"); starttime=record.get("process_starttime")
            members=record.get("member_identities")
            if (record.get("job_key") != job_key or
                    record.get("lease_token") != token or record.get("state") != RUNNING or
                    not isinstance(pid,int) or pid <= 0 or not isinstance(starttime,str) or not starttime or
                    not isinstance(members,dict) or members.get(str(pid)) != starttime or
                    any(not isinstance(member,str) or not member.isdigit() or not isinstance(member_start,str) or not member_start for member,member_start in members.items())):
                return False
            normalized_members={int(member): str(member_start) for member,member_start in members.items()}
            if any(member <= 0 or not member_start for member,member_start in normalized_members.items()):
                return False
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return False
        now=time.time()
        with self.connect() as db, self.transaction(db):
            row=db.execute("SELECT j.*,q.id AS job_id,q.status,q.lease_token,q.attempts FROM attempt_journal j JOIN jobs q ON q.job_key=j.job_key AND q.lease_token=j.lease_token WHERE q.job_key=? AND q.lease_token=?",(job_key,token)).fetchone()
            if (not row or row["status"] != RUNNING or row["state"] != "LAUNCHING" or
                    row["attempts"] != attempt_number or Path(row["journal_path"]).resolve() != expected_path or
                    not db.execute("SELECT 1 FROM attempts WHERE job_id=? AND number=? AND lease_token=?",(row["job_id"],attempt_number,token)).fetchone()):
                return False
            db.execute("UPDATE attempt_journal SET pid=?,process_starttime=?,state=?,updated_at=? WHERE lease_token=?",(pid,starttime,RUNNING,now,token))
            db.execute("UPDATE jobs SET pid=?,process_starttime=?,updated_at=? WHERE id=? AND status=? AND lease_token=?",(pid,starttime,now,row["job_id"],RUNNING,token))
        return True

    def release_unstarted(self, job_key, token, detail):
        """Undo a resource-preflight claim before a child or journal exists."""
        now = time.time()
        with self.connect() as db, self.transaction(db):
            row = db.execute("SELECT * FROM jobs WHERE job_key=?", (job_key,)).fetchone()
            if not row or row["status"] != RUNNING or row["lease_token"] != token or row["pid"] is not None:
                raise LeaseLost(job_key)
            journal = db.execute("SELECT 1 FROM attempt_journal WHERE lease_token=?", (token,)).fetchone()
            if journal: raise QueueError("cannot release a journaled attempt")
            db.execute("DELETE FROM attempts WHERE job_id=? AND lease_token=?", (row["id"], token))
            db.execute("UPDATE jobs SET status=?,attempts=attempts-1,lease_owner=NULL,lease_token=NULL,lease_expires=NULL,failure=?,updated_at=? WHERE id=?", (PENDING, detail, now, row["id"]))

    def launch_guard(self, job_key, token, lock_handle=None):
        """Recheck immutable formal admission after the local GPU flock is held."""
        with self.connect() as db:
            row = db.execute("SELECT * FROM jobs WHERE job_key=? AND lease_token=? AND status=?", (job_key, token, RUNNING)).fetchone()
            if not row:
                raise LeaseLost(job_key)
            reservation=None
            if row["kind"] == "formal_train":
                reservation=db.execute("SELECT lease_id,host,physical_gpu,lock_device,lock_inode,authorized_end FROM reservations WHERE lease_token=? AND job_key=?", (token,job_key)).fetchone()
                if reservation is None: raise QueueError("formal reservation is not bound to claimed attempt")
            failure = self._guard_failure(db, row)
            if failure:
                raise QueueError(failure)
            if row["kind"] == "formal_train":
                if lock_handle in (None, False): raise QueueError("formal launch requires the held device lock")
                try:
                    current=os.fstat(lock_handle.fileno())
                    canonical=_canonical_gpu_lock(json.loads(row["payload"])).stat()
                except (AttributeError, OSError) as error: raise QueueError("formal launch lock is unreadable") from error
                if (current.st_dev != reservation["lock_device"] or current.st_ino != reservation["lock_inode"] or
                        canonical.st_dev != reservation["lock_device"] or canonical.st_ino != reservation["lock_inode"] or
                        not isinstance(lock_handle, HeldGpuLock) or not _fd_holds_flock(current, os.getpid(), lock_handle.fileno())):
                    raise QueueError("formal launch lost the bound device lock")
                try: verify_attestation(json.loads(row["payload"]), job_key, {item["name"]:(item["path"],item["checksum"]) for item in db.execute("SELECT name,path,checksum FROM evidence WHERE accepted=1")}, dict(reservation))
                except (ResourceAttestationError, OSError, ValueError, TypeError) as error: raise QueueError(f"formal reservation rejected: {error}") from error

    def recovered_lock_guard(self, job_key, token, pid):
        """A recovered formal child must still hold the bound inherited flock."""
        with self.connect() as db:
            row=db.execute("SELECT payload FROM jobs WHERE job_key=? AND lease_token=? AND status=?",(job_key,token,RUNNING)).fetchone()
            reservation=db.execute("SELECT lock_device,lock_inode FROM reservations WHERE lease_token=? AND job_key=?",(token,job_key)).fetchone()
            try:
                canonical=_canonical_gpu_lock(json.loads(row["payload"])).stat() if row else None
            except (OSError, ValueError, TypeError):
                canonical=None
            if (reservation is None or canonical is None or canonical.st_dev != reservation["lock_device"] or
                    canonical.st_ino != reservation["lock_inode"] or
                    not _held_flock(type("LockStat",(),{"st_dev":reservation["lock_device"],"st_ino":reservation["lock_inode"]})(),pid)):
                raise QueueError("recovered formal child lacks the bound device lock")

    def bind_reservation(self, job_key, token, lock_handle):
        """Record the actual held canonical device lock for this attempt."""
        if lock_handle in (None, False):
            raise QueueError("formal reservation requires a held device lock")
        try:
            lock_stat=os.fstat(lock_handle.fileno())
        except (AttributeError, OSError) as error:
            raise QueueError("formal reservation lock is unreadable") from error
        with self.connect() as db, self.transaction(db):
            row=db.execute("SELECT * FROM jobs WHERE job_key=? AND lease_token=? AND status=?",(job_key,token,RUNNING)).fetchone()
            if not row: raise LeaseLost(job_key)
            payload=json.loads(row["payload"])
            if row["kind"] != "formal_train": return
            canonical_lock=_canonical_gpu_lock(payload)
            try:
                expected_stat=canonical_lock.stat()
            except OSError as error:
                raise QueueError("formal reservation canonical device lock is unavailable") from error
            if (not isinstance(lock_handle, HeldGpuLock) or lock_handle.path != canonical_lock or not stat.S_ISREG(lock_stat.st_mode) or lock_stat.st_dev != expected_stat.st_dev or
                    lock_stat.st_ino != expected_stat.st_ino or not _held_flock_for_current_process(lock_stat)):
                raise QueueError("formal reservation requires the held canonical device lock")
            path, document = __import__("nc_rted.resource_attestation", fromlist=["_bound_file"])._bound_file(payload.get("resource_attestation"), payload.get("resource_attestation_sha256"), "resource attestation")
            del path
            execution=document.get("execution", {})
            started=db.execute("SELECT started_at FROM attempts WHERE job_id=? AND lease_token=?",(row["id"],token)).fetchone()
            if started is None: raise LeaseLost(job_key)
            end=min(float(execution.get("lease_expires_utc_epoch",0)), float(payload.get("deadline_utc_epoch",0)), started["started_at"] + float(payload.get("run_budget_seconds",0)))
            if end <= time.time(): raise QueueError("formal reservation interval is already expired")
            db.execute("INSERT OR REPLACE INTO reservations VALUES(?,?,?,?,?,?,?,?,?)",(token,job_key,execution.get("lease_id"),execution.get("host"),execution.get("physical_gpu"),lock_stat.st_dev,lock_stat.st_ino,time.time(),end))

    def runtime_guard(self, job_key, token):
        """Check hard limits while a child runs; progress timeout is intentionally external."""
        with self.connect() as db:
            row=db.execute("SELECT * FROM jobs WHERE job_key=? AND lease_token=? AND status=?",(job_key,token,RUNNING)).fetchone()
            if not row: raise LeaseLost(job_key)
            payload=json.loads(row["payload"]); now=time.time()
            if payload.get("deadline_utc_epoch") and now >= payload["deadline_utc_epoch"]: raise HardLimit("deadline")
            volume=Path(payload.get("data_volume",self.path.parent))
            try:
                if payload.get("min_free_bytes") and shutil.disk_usage(volume).free < int(payload["min_free_bytes"]): raise HardLimit("disk")
            except OSError as error: raise HardLimit("resource_integrity") from error
            attempt=db.execute("SELECT started_at FROM attempts WHERE job_id=? AND lease_token=?",(row["id"],token)).fetchone()
            if payload.get("run_budget_seconds") and attempt and now-attempt["started_at"] > payload["run_budget_seconds"]: raise HardLimit("budget")
            if row["kind"] == "formal_train":
                reservation=db.execute("SELECT authorized_end FROM reservations WHERE lease_token=? AND job_key=?",(token,job_key)).fetchone()
                if reservation is None: raise HardLimit("resource_integrity")
                if now >= reservation["authorized_end"]: raise HardLimit("rental_lease")
                try:
                    if now >= resource_lease_expiry(payload): raise HardLimit("rental_lease")
                except ResourceAttestationError as error: raise HardLimit("resource_integrity") from error

    def record_terminal_evidence(self, job_key, token, pid, process_starttime):
        """Durably record the supervisor's own observation of a dead attempt group."""
        from .worker_runtime import attempt_state, group_state
        if attempt_state(pid, process_starttime) != "gone" or group_state(pid) != "gone":
            raise QueueError("formal terminal evidence requires an observed dead attempt group")
        observed=time.time()
        with self.connect() as db, self.transaction(db):
            row=db.execute("SELECT * FROM jobs WHERE job_key=? AND lease_token=? AND status=?",(job_key,token,RUNNING)).fetchone()
            reservation=db.execute("SELECT authorized_end FROM reservations WHERE lease_token=? AND job_key=?",(token,job_key)).fetchone()
            if not row: raise LeaseLost(job_key)
            if not reservation or observed > reservation["authorized_end"]: raise HardLimit("rental_lease")
            db.execute("INSERT OR REPLACE INTO terminal_evidence VALUES(?,?,?,?,?)",(token,job_key,pid,str(process_starttime),observed))

    def completion_guard(self, job_key, token):
        """Accept only an attempt completion observed within its fixed reservation."""
        with self.connect() as db:
            row=db.execute("SELECT * FROM jobs WHERE job_key=? AND lease_token=? AND status=?",(job_key,token,RUNNING)).fetchone()
            if not row: raise LeaseLost(job_key)
            if row["kind"] != "formal_train": return
            reservation=db.execute("SELECT lease_id,host,physical_gpu,lock_device,lock_inode,authorized_end FROM reservations WHERE lease_token=? AND job_key=?",(token,job_key)).fetchone()
            if reservation is None: raise HardLimit("resource_integrity")
            payload=json.loads(row["payload"])
            try:
                _, attestation=__import__("nc_rted.resource_attestation", fromlist=["_bound_file"])._bound_file(payload["resource_attestation"],payload["resource_attestation_sha256"],"resource attestation")
                execution=attestation.get("execution", {})
                contract=attestation.get("contract", {})
                if (execution.get("lease_id") != reservation["lease_id"] or execution.get("host") != reservation["host"] or
                        execution.get("physical_gpu") != reservation["physical_gpu"] or
                        contract != {key:payload.get(key) for key in ("data_volume","min_free_bytes","run_budget_seconds","deadline_utc_epoch")}):
                    raise ResourceAttestationError("completion contract differs from admitted reservation")
            except (ResourceAttestationError, OSError, ValueError, TypeError) as error:
                raise HardLimit("resource_integrity") from error
            started=db.execute("SELECT started_at FROM attempts WHERE job_id=? AND lease_token=?",(row["id"],token)).fetchone()
            terminal=db.execute("SELECT pid,process_starttime,observed_at FROM terminal_evidence WHERE lease_token=? AND job_key=?",(token,job_key)).fetchone()
            if started is None: raise HardLimit("resource_integrity")
            if terminal is None or terminal["observed_at"] < started["started_at"]: raise HardLimit("resource_integrity")
            expiry=attestation["execution"]["lease_expires_utc_epoch"]
            if terminal["observed_at"] > started["started_at"] + payload["run_budget_seconds"]:
                raise HardLimit("budget")
            if started["started_at"] > expiry or terminal["observed_at"] > expiry or terminal["observed_at"] > reservation["authorized_end"]:
                raise HardLimit("rental_lease")

    def commit(self, job_key, token, temporary_output, output_path, owner=None):
        temporary_output, output_path = Path(temporary_output), Path(output_path)
        if not temporary_output.is_file() and not output_path.is_file(): raise QueueError("missing temporary output")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        source = temporary_output if temporary_output.is_file() else output_path
        checksum = hashlib.sha256(source.read_bytes()).hexdigest()
        with source.open("rb") as handle: os.fsync(handle.fileno())
        now = time.time()
        with self.connect() as db, self.transaction(db):
            row = db.execute("SELECT * FROM jobs WHERE job_key=?", (job_key,)).fetchone()
            if not row or row["status"] != RUNNING or row["lease_token"] != token or (owner is not None and row["lease_owner"] != owner): raise LeaseLost(job_key)
            if row["protective_stop_code"]:
                raise HardLimit(row["protective_stop_code"])
            if temporary_output.is_file(): os.replace(temporary_output, output_path)
            directory_fd = os.open(output_path.parent, os.O_DIRECTORY); os.fsync(directory_fd); os.close(directory_fd)
            db.execute("UPDATE jobs SET status=?,output_path=?,output_checksum=?,lease_owner=NULL,lease_token=NULL,lease_expires=NULL,updated_at=? WHERE id=?", (SUCCEEDED, str(output_path), checksum, now, row["id"]))
            db.execute("UPDATE attempts SET ended_at=?,outcome=? WHERE job_id=? AND lease_token=?", (now, SUCCEEDED, row["id"], token))
            db.execute("UPDATE attempt_journal SET state=?,updated_at=? WHERE lease_token=?", (SUCCEEDED,now,token))

    @staticmethod
    def validate_artifacts(expected_outputs, artifacts):
        """Validate a job-declared artifact contract; wrapper exit status is insufficient."""
        if len(expected_outputs) != len(artifacts): raise QueueError("artifact count mismatch")
        for expected, actual in zip(expected_outputs, artifacts):
            if expected.get("artifact_type") not in {"checkpoint", "prediction", "report", "smoke"}: raise QueueError("unknown artifact type")
            path=Path(expected["path"])
            if str(path) != actual.get("path") or not path.is_file(): raise QueueError("artifact path mismatch")
            if expected.get("checksum") and expected["checksum"] != actual.get("checksum"): raise QueueError("artifact checksum mismatch")
            fields=expected.get("required_json_keys", []); semantic=expected.get("semantic")
            if expected.get("artifact_type") == "prediction" and semantic != "prediction":
                raise QueueError("prediction artifact requires semantic provenance contract")
            if fields or semantic:
                try: document=json.loads(path.read_text())
                except OSError as exc: raise ArtifactReadUncertain("required JSON artifact unreadable") from exc
                except (json.JSONDecodeError, UnicodeDecodeError) as exc: raise QueueError("required JSON artifact invalid") from exc
                if not isinstance(document,dict): raise QueueError("required JSON artifact must be an object")
                if not all(field in document for field in fields): raise QueueError("artifact schema keys missing")
            if semantic == "formal_training":
                checkpoint = document.get("schema") == "nc_rted_checkpoint_v2"
                if not checkpoint or not (document.get("final") is True and document.get("completed_updates") == 1000 and document.get("identity") == expected.get("run_identity")):
                    raise QueueError("formal training artifact identity/update contract failed")
                payload = path.parent / "state.pt"
                root=path.parent.resolve()
                try:
                    valid_payload = (not payload.is_symlink() and payload.is_file() and payload.resolve().parent == root and
                                     document.get("payload_bytes") == payload.stat().st_size and
                                     document.get("payload_sha256") == hashlib.sha256(payload.read_bytes()).hexdigest())
                except OSError as exc:
                    raise ArtifactReadUncertain("formal checkpoint payload unreadable") from exc
                if (path.is_symlink() or path.name != "manifest.json" or path.parent.name != "final" or not valid_payload):
                    raise QueueError("formal checkpoint payload contract failed")
                try:
                    from .recovery import CheckpointReadUncertain, validate_checkpoint_payload
                    validate_checkpoint_payload(payload, document)
                except OSError as exc:
                    raise ArtifactReadUncertain("formal checkpoint payload unreadable") from exc
                except CheckpointReadUncertain as exc:
                    raise ArtifactReadUncertain("formal checkpoint payload unreadable") from exc
                except Exception as exc:
                    raise QueueError("formal checkpoint payload structure failed") from exc
            elif semantic == "prediction":
                ids, denominator=document.get("prediction_ids"), expected.get("denominator")
                if (not isinstance(ids,list) or not all(isinstance(item,str) and item for item in ids) or not isinstance(denominator,int) or len(ids) != denominator or len(set(ids)) != denominator):
                    raise QueueError("prediction artifact ID denominator contract failed")
                official = expected.get("official_ids")
                bindings=(expected.get("model_hash"), expected.get("input_hash"), document.get("model_hash"), document.get("input_hash"))
                valid_hash=lambda value: isinstance(value,str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
                if (not all(valid_hash(value) for value in bindings) or
                        not isinstance(official, list) or not all(isinstance(item,str) and item for item in official) or len(set(official)) != len(official) or set(ids) != set(official) or
                        bindings[0] != bindings[2] or bindings[1] != bindings[3]):
                    raise QueueError("prediction artifact provenance/official-ID contract failed")
            elif semantic is not None: raise QueueError("unknown semantic artifact contract")

    def running_attempts(self):
        with self.connect() as db:
            return [dict(row) for row in db.execute("SELECT j.*,q.payload,q.input_hash,q.lease_owner,q.status,q.attempts,q.kind,q.failure,q.protective_stop_code FROM attempt_journal j JOIN jobs q ON q.job_key=j.job_key AND q.lease_token=j.lease_token WHERE q.status=? AND j.state IN ('RUNNING','LAUNCHING')",(RUNNING,))]

    def attempt_started_at(self, job_key, token):
        with self.connect() as db:
            row=db.execute("SELECT a.started_at FROM attempts a JOIN jobs j ON j.id=a.job_id WHERE j.job_key=? AND a.lease_token=?",(job_key,token)).fetchone()
        return row["started_at"] if row else None

    def fail(self, job_key, token, detail, owner=None, code="transient"):
        now = time.time()
        with self.connect() as db, self.transaction(db):
            row = db.execute("SELECT * FROM jobs WHERE job_key=?", (job_key,)).fetchone()
            if not row or row["status"] != RUNNING or row["lease_token"] != token or (owner is not None and row["lease_owner"] != owner): raise LeaseLost(job_key)
            code = row["protective_stop_code"] or code
            protective = code in {"deadline", "disk", "budget", "nan", "integrity", "resource_integrity", "rental_lease", "leakage", "weight_mismatch"}
            if protective or row["attempts"] >= row["max_attempts"]: status, next_time = BLOCKED, now
            else: status, next_time = RETRY_WAIT, now + RETRY_DELAYS[row["attempts"] - 1]
            db.execute("UPDATE jobs SET status=?,not_before=?,failure=?,lease_owner=NULL,lease_token=NULL,lease_expires=NULL,pid=NULL,process_starttime=NULL,progress_at=NULL,progress_counter=NULL,updated_at=? WHERE id=?", (status, next_time, detail, now, row["id"]))
            db.execute("UPDATE attempts SET ended_at=?,outcome=?,detail=? WHERE job_id=? AND lease_token=?", (now, status, detail, row["id"], token))
            db.execute("UPDATE attempt_journal SET state=?,updated_at=? WHERE lease_token=?", (status,now,token))

    def recover_expired(self):
        now = time.time(); changed = 0
        with self.connect() as db, self.transaction(db):
            rows = db.execute("SELECT * FROM jobs WHERE status=? AND lease_expires<?", (RUNNING, now)).fetchall()
            for row in rows:
                if self._process_may_live(row): continue
                journal=db.execute("SELECT journal_path FROM attempt_journal WHERE lease_token=? AND state IN ('RUNNING','LAUNCHING')",(row["lease_token"],)).fetchone()
                # Preserve a durable completion record for the worker's
                # reconciliation path; expiry must not discard its lease.
                if journal and any((Path(journal["journal_path"]).parent / name).is_file() for name in ("producer_completion.json", "result.tmp", "result.json")):
                    continue
                # A hard-limit decision is durable policy, not controller
                # diagnostics.  Once the child identity is conclusively gone,
                # it must win over ordinary lease-expiry retry handling.
                stop = row["protective_stop_code"]
                status = BLOCKED if stop or row["attempts"] >= row["max_attempts"] else RETRY_WAIT
                delay = 0 if status == BLOCKED else RETRY_DELAYS[row["attempts"] - 1]
                detail = f"hard limit: {stop}" if stop else "lease expired; process left untouched"
                db.execute("UPDATE jobs SET status=?,not_before=?,failure=?,lease_owner=NULL,lease_token=NULL,lease_expires=NULL,pid=NULL,process_starttime=NULL,progress_at=NULL,progress_counter=NULL,updated_at=? WHERE id=?", (status, now + delay, detail, now, row["id"]))
                db.execute("UPDATE attempts SET ended_at=?,outcome=?,detail=? WHERE job_id=? AND lease_token=?",(now,status,detail,row["id"],row["lease_token"]))
                db.execute("UPDATE attempt_journal SET state=?,updated_at=? WHERE lease_token=?",(status,now,row["lease_token"]))
                changed += 1
        return changed

    @staticmethod
    def _process_may_live(row):
        from .worker_runtime import attempt_state
        host = (row["lease_owner"] or "").split(":", 1)[0]
        # Foreign ownership, prelaunch intent, PID reuse, and descendants must
        # remain reserved. Only a verified local gone/zombie-only group expires.
        return attempt_state(row["pid"], row["process_starttime"], host) not in {"gone"}

    def complete_gate(self, job_key):
        with self.connect() as db, self.transaction(db):
            row = db.execute("SELECT * FROM jobs WHERE job_key=?", (job_key,)).fetchone()
            if not row or row["kind"] not in {"manual_gate", "audit"}: raise QueueError("not a manual gate")
            if db.execute("SELECT 1 FROM dependencies d JOIN jobs p ON p.id=d.depends_on WHERE d.job_id=? AND p.status!=?", (row["id"], SUCCEEDED)).fetchone(): raise QueueError("gate dependencies are incomplete")
            failure = self._guard_failure(db, row)
            if failure: raise QueueError(failure)
            db.execute("UPDATE jobs SET status=?,output_checksum=?,updated_at=? WHERE id=?", (SUCCEEDED, canonical_hash({"job":job_key,"evidence":json.loads(row["payload"]).get("evidence",[])}), time.time(), row["id"]))

    def reconcile(self):
        """Re-open only guard-blocked jobs after their evidence becomes available."""
        with self.connect() as db, self.transaction(db):
            rows = db.execute("SELECT * FROM jobs WHERE status=?", (BLOCKED,)).fetchall(); changed = 0
            for row in rows:
                if row["failure"] and (row["failure"].startswith("missing verified evidence") or row["failure"].startswith("no audited")) and not self._guard_failure(db, row):
                    db.execute("UPDATE jobs SET status=?,failure=NULL,updated_at=? WHERE id=?", (PENDING, time.time(), row["id"])); changed += 1
        return changed

    def status(self):
        with self.connect() as db:
            return [dict(row) for row in db.execute("SELECT job_key,kind,status,attempts,output_checksum,failure,pid,process_starttime,progress_at,progress_counter FROM jobs ORDER BY id")]

    def register_matrix(self):
        self.add_job("prepare.resource_preflight", "audit", {"evidence": ["resource_feasibility"]})
        self.add_job("prepare.freeze_manifest", "manual_gate", {"evidence": ["frozen_manifest", "weight_environment_hashes"]}, ["prepare.resource_preflight"])
        self.add_job("prepare.executable_audit", "manual_gate", {"evidence": ["audited_executable"], "min_free_bytes": 20 * 1024**3}, ["prepare.freeze_manifest"])
        base = {"evidence":["frozen_manifest","audited_executable"],"command":None,"expected_outputs":None,"deadline_utc_epoch":FORMAL_DEADLINE,"device_binding":"PENDING_RESOURCE_AUDIT","run_budget_seconds":None}
        self.add_job("R0.blind_prediction", "prediction", base, ["prepare.executable_audit"])
        for group in FORMAL_GROUPS:
            for seed in SEEDS:
                key = f"{group}.seed{seed}.train"
                self.add_job(key, "formal_train", {**base,"group":group,"seed":seed,"evidence":["frozen_manifest","teacher_manifest","audited_executable"],"min_free_bytes":20*1024**3}, ["prepare.executable_audit"])
                self.add_job(f"{group}.seed{seed}.blind_prediction", "prediction", {**base,"group":group,"seed":seed}, [key])
