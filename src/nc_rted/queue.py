"""Small, transactional control plane for accepted NC-RTED work only."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import time
import uuid
import socket
import signal
from contextlib import contextmanager
from pathlib import Path

PENDING, RUNNING, RETRY_WAIT, SUCCEEDED, BLOCKED = "PENDING", "RUNNING", "RETRY_WAIT", "SUCCEEDED", "BLOCKED"
RETRY_DELAYS = (300, 1200, 3600)
FORMAL_GROUPS = ("A", "U", "S", "F")
SEEDS = (17, 42, 2026)
FORMAL_DEADLINE = 1792724040  # 2026-10-23T02:54:00Z


def canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class QueueError(RuntimeError): pass
class LeaseLost(QueueError): pass


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
              output_path TEXT, output_checksum TEXT, failure TEXT,
              progress_at REAL, progress_counter INTEGER,
              created_at REAL NOT NULL, updated_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS dependencies (job_id INTEGER NOT NULL REFERENCES jobs(id), depends_on INTEGER NOT NULL REFERENCES jobs(id), PRIMARY KEY(job_id, depends_on));
            CREATE TABLE IF NOT EXISTS evidence (name TEXT PRIMARY KEY, checksum TEXT NOT NULL, path TEXT NOT NULL, schema_name TEXT NOT NULL DEFAULT '', input_code_hash TEXT NOT NULL DEFAULT '', accepted INTEGER NOT NULL DEFAULT 0, verified_at REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS attempts (id INTEGER PRIMARY KEY, job_id INTEGER NOT NULL REFERENCES jobs(id), number INTEGER NOT NULL, owner TEXT NOT NULL, lease_token TEXT NOT NULL, started_at REAL NOT NULL, ended_at REAL, outcome TEXT, detail TEXT);
            CREATE TABLE IF NOT EXISTS attempt_journal (lease_token TEXT PRIMARY KEY, job_key TEXT NOT NULL, pid INTEGER, process_starttime TEXT, gpu_identity TEXT, journal_path TEXT NOT NULL, state TEXT NOT NULL, updated_at REAL NOT NULL);
            """)
            for statement in ("ALTER TABLE evidence ADD COLUMN schema_name TEXT NOT NULL DEFAULT ''", "ALTER TABLE evidence ADD COLUMN input_code_hash TEXT NOT NULL DEFAULT ''", "ALTER TABLE evidence ADD COLUMN accepted INTEGER NOT NULL DEFAULT 0"):
                try: db.execute(statement)
                except sqlite3.OperationalError: pass
            for statement in ("ALTER TABLE jobs ADD COLUMN progress_at REAL", "ALTER TABLE jobs ADD COLUMN progress_counter INTEGER"):
                try: db.execute(statement)
                except sqlite3.OperationalError: pass

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
        if min_free and shutil.disk_usage(self.path.parent).free < min_free:
            return "disk budget guard: less than required free space"
        deadline = payload.get("deadline_utc_epoch")
        if deadline and time.time() >= deadline: return "deadline guard reached"
        if not payload.get("command") and row["kind"] not in {"manual_gate", "audit"}:
            return "no audited concrete executable command"
        if row["kind"] not in {"manual_gate", "audit"} and not payload.get("expected_outputs"):
            return "no declared output validator/checksum contract"
        if row["kind"] not in {"manual_gate", "audit"} and payload.get("physical_gpu") is None:
            return "no physical CUDA device binding"
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

    def heartbeat(self, job_key, token, lease_seconds=120, pid=None, process_starttime=None):
        now = time.time()
        with self.connect() as db:
            changed = db.execute("UPDATE jobs SET heartbeat=?,lease_expires=?,pid=COALESCE(?,pid),process_starttime=COALESCE(?,process_starttime),updated_at=? WHERE job_key=? AND status=? AND lease_token=?", (now, now + lease_seconds, pid, process_starttime, now, job_key, RUNNING, token)).rowcount
        if not changed: raise LeaseLost(job_key)

    def record_progress(self, job_key, token, counter: int):
        """Progress is separate from liveness; callers must report real checkpoints/media commits."""
        now=time.time()
        with self.connect() as db:
            previous=db.execute("SELECT progress_counter FROM jobs WHERE job_key=? AND status=? AND lease_token=?",(job_key,RUNNING,token)).fetchone()
            if previous is None: raise LeaseLost(job_key)
            if previous["progress_counter"] is not None and counter <= previous["progress_counter"]: return False
            changed=db.execute("UPDATE jobs SET progress_at=?,progress_counter=?,updated_at=? WHERE job_key=? AND status=? AND lease_token=?",(now,counter,now,job_key,RUNNING,token)).rowcount
        if not changed: raise LeaseLost(job_key)
        return True

    def protected_live(self, job_key, token, detail):
        """Leave an uncertain/live child reserved; it must never be retried."""
        with self.connect() as db:
            changed=db.execute("UPDATE jobs SET failure=?,updated_at=? WHERE job_key=? AND status=? AND lease_token=?",(detail,time.time(),job_key,RUNNING,token)).rowcount
        if not changed: raise LeaseLost(job_key)

    def start_attempt_journal(self, job_key, token, pid, process_starttime, gpu_identity, journal_path):
        path=Path(journal_path); path.parent.mkdir(parents=True,exist_ok=True)
        record={"job_key":job_key,"lease_token":token,"pid":pid,"process_starttime":process_starttime,"gpu_identity":gpu_identity,"state":"RUNNING","updated_at":time.time()}
        path.write_text(json.dumps(record,sort_keys=True)+"\n")
        with path.open("rb") as handle: os.fsync(handle.fileno())
        with self.connect() as db: db.execute("INSERT OR REPLACE INTO attempt_journal VALUES(?,?,?,?,?,?,?,?)",(token,job_key,pid,process_starttime,gpu_identity,str(path),"RUNNING",time.time()))

    def runtime_guard(self, job_key, token):
        """Check hard limits while a child runs; progress timeout is intentionally external."""
        with self.connect() as db:
            row=db.execute("SELECT * FROM jobs WHERE job_key=? AND lease_token=? AND status=?",(job_key,token,RUNNING)).fetchone()
            if not row: raise LeaseLost(job_key)
            payload=json.loads(row["payload"]); now=time.time()
            if payload.get("deadline_utc_epoch") and now >= payload["deadline_utc_epoch"]: raise QueueError("deadline hard limit")
            volume=Path(payload.get("data_volume",self.path.parent))
            if payload.get("min_free_bytes") and shutil.disk_usage(volume).free < int(payload["min_free_bytes"]): raise QueueError("disk hard limit")
            attempt=db.execute("SELECT started_at FROM attempts WHERE job_id=? AND lease_token=?",(row["id"],token)).fetchone()
            if payload.get("run_budget_seconds") and attempt and now-attempt["started_at"] > payload["run_budget_seconds"]: raise QueueError("budget hard limit")

    def commit(self, job_key, token, temporary_output, output_path):
        temporary_output, output_path = Path(temporary_output), Path(output_path)
        if not temporary_output.is_file(): raise QueueError("missing temporary output")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        checksum = hashlib.sha256(temporary_output.read_bytes()).hexdigest()
        with temporary_output.open("rb") as handle: os.fsync(handle.fileno())
        now = time.time()
        with self.connect() as db, self.transaction(db):
            row = db.execute("SELECT * FROM jobs WHERE job_key=?", (job_key,)).fetchone()
            if not row or row["status"] != RUNNING or row["lease_token"] != token: raise LeaseLost(job_key)
            os.replace(temporary_output, output_path)
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
            fields=expected.get("required_json_keys", [])
            if fields:
                try: document=json.loads(path.read_text())
                except (OSError,json.JSONDecodeError) as exc: raise QueueError("required JSON artifact invalid") from exc
                if not all(field in document for field in fields): raise QueueError("artifact schema keys missing")

    def fail(self, job_key, token, detail):
        now = time.time()
        with self.connect() as db, self.transaction(db):
            row = db.execute("SELECT * FROM jobs WHERE job_key=?", (job_key,)).fetchone()
            if not row or row["status"] != RUNNING or row["lease_token"] != token: raise LeaseLost(job_key)
            protective = any(marker in detail.lower() for marker in ("nan", "integrity", "leakage", "weight mismatch", "budget", "disk hard limit"))
            if protective or row["attempts"] >= row["max_attempts"]: status, next_time = BLOCKED, now
            else: status, next_time = RETRY_WAIT, now + RETRY_DELAYS[row["attempts"] - 1]
            db.execute("UPDATE jobs SET status=?,not_before=?,failure=?,lease_owner=NULL,lease_token=NULL,lease_expires=NULL,updated_at=? WHERE id=?", (status, next_time, detail, now, row["id"]))
            db.execute("UPDATE attempts SET ended_at=?,outcome=?,detail=? WHERE job_id=? AND lease_token=?", (now, status, detail, row["id"], token))
            db.execute("UPDATE attempt_journal SET state=?,updated_at=? WHERE lease_token=?", (status,now,token))

    def recover_expired(self):
        now = time.time(); changed = 0
        with self.connect() as db, self.transaction(db):
            rows = db.execute("SELECT * FROM jobs WHERE status=? AND lease_expires<?", (RUNNING, now)).fetchall()
            for row in rows:
                if self._process_may_live(row): continue
                status = BLOCKED if row["attempts"] >= row["max_attempts"] else RETRY_WAIT
                delay = 0 if status == BLOCKED else RETRY_DELAYS[row["attempts"] - 1]
                db.execute("UPDATE jobs SET status=?,not_before=?,failure=?,lease_owner=NULL,lease_token=NULL,lease_expires=NULL,updated_at=? WHERE id=?", (status, now + delay, "lease expired; process left untouched", now, row["id"]))
                changed += 1
        return changed

    @staticmethod
    def _process_may_live(row):
        if not row["pid"]: return True
        host = (row["lease_owner"] or "").split(":", 1)[0]
        if host and host != socket.gethostname(): return True
        try:
            fields=Path(f"/proc/{row['pid']}/stat").read_text().split()
            return fields[2] != "Z" and fields[21] == str(row["process_starttime"])
        except OSError: return False

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
            return [dict(row) for row in db.execute("SELECT job_key,kind,status,attempts,output_checksum,failure,pid,process_starttime FROM jobs ORDER BY id")]

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
