import json
import threading
import time
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nc_rted.queue import BLOCKED, RETRY_WAIT, RUNNING, SUCCEEDED, JobQueue, LeaseLost


def add_ready(queue, key="job", **payload):
    command = payload.pop("command", ["/bin/true"])
    queue.add_job(key, "smoke", {"command": command, "physical_gpu": 0, "expected_outputs": [{"path": "/dev/null"}], **payload})


def test_concurrent_claim_has_single_lease(tmp_path):
    queue = JobQueue(tmp_path / "queue.sqlite"); add_ready(queue)
    barrier = threading.Barrier(3); results = []
    def claim(owner):
        barrier.wait(); results.append(queue.claim(owner))
    threads = [threading.Thread(target=claim, args=(f"owner-{i}",)) for i in range(2)]
    [t.start() for t in threads]; barrier.wait(); [t.join() for t in threads]
    claimed = [result for result in results if result]
    assert len(claimed) == 1 and claimed[0]["status"] == RUNNING


def test_stale_owner_cannot_commit(tmp_path):
    queue = JobQueue(tmp_path / "queue.sqlite"); add_ready(queue)
    import socket
    first = queue.claim(f"{socket.gethostname()}:first", lease_seconds=0.001)
    queue.heartbeat("job", first["lease_token"], lease_seconds=.001, pid=999999, process_starttime="x")
    time.sleep(.01); assert queue.recover_expired() == 1
    with queue.connect() as db:
        db.execute("UPDATE jobs SET not_before=0 WHERE job_key='job'")
    second = queue.claim("second")
    temporary = tmp_path / "output.tmp"; temporary.write_text("second")
    with pytest.raises(LeaseLost): queue.commit("job", first["lease_token"], temporary, tmp_path / "out")
    queue.commit("job", second["lease_token"], temporary, tmp_path / "out")
    assert queue.status()[0]["status"] == SUCCEEDED


def test_recovery_is_idempotent_and_never_kills_process(tmp_path):
    queue = JobQueue(tmp_path / "queue.sqlite"); add_ready(queue)
    import socket
    job = queue.claim(f"{socket.gethostname()}:worker", lease_seconds=0.001); queue.heartbeat("job", job["lease_token"], lease_seconds=0.001, pid=999999, process_starttime="x")
    time.sleep(.01); assert queue.recover_expired() == 1; assert queue.recover_expired() == 0
    row = queue.status()[0]; assert row["status"] == RETRY_WAIT and row["pid"] == 999999


def test_live_local_child_is_never_retried_after_observation_expiry(tmp_path):
    queue=JobQueue(tmp_path/'queue.sqlite'); add_ready(queue)
    import os, socket
    job=queue.claim(f'{socket.gethostname()}:worker',lease_seconds=.001)
    queue.heartbeat('job',job['lease_token'],lease_seconds=.001,pid=os.getpid(),process_starttime=Path('/proc/self/stat').read_text().split()[21])
    time.sleep(.01); assert queue.recover_expired()==0 and queue.status()[0]['status']==RUNNING

def test_live_gpu_reservation_blocks_a_different_pending_job(tmp_path):
    queue=JobQueue(tmp_path/'queue.sqlite'); add_ready(queue,'first',physical_gpu=0); add_ready(queue,'second',physical_gpu=0)
    import os, socket
    first=queue.claim(f'{socket.gethostname()}:worker')
    queue.heartbeat('first',first['lease_token'],pid=os.getpid(),process_starttime=Path('/proc/self/stat').read_text().split()[21])
    assert queue.claim('another') is None


def test_evidence_rehash_requires_explicit_acceptance(tmp_path):
    queue=JobQueue(tmp_path/'queue.sqlite'); evidence=tmp_path/'manifest'; evidence.write_text('a')
    queue.add_evidence('manifest',evidence); queue.add_job('job','smoke',{'command':['true'],'expected_outputs':[{'path':'/dev/null'}],'evidence':['manifest']})
    assert queue.claim('worker') is None
    queue.add_evidence('manifest',evidence,accepted=True); evidence.write_text('changed')
    queue.reconcile(); assert queue.claim('worker') is None

def test_artifact_contract_rejects_wrapper_only_or_missing_schema(tmp_path):
    queue=JobQueue(tmp_path/'queue.sqlite'); artifact=tmp_path/'artifact.json'; artifact.write_text('{"ok":true}')
    checksum=__import__('hashlib').sha256(artifact.read_bytes()).hexdigest()
    queue.validate_artifacts([{'path':str(artifact),'artifact_type':'report','checksum':checksum,'required_json_keys':['ok']}],[{'path':str(artifact),'checksum':checksum}])
    with pytest.raises(Exception): queue.validate_artifacts([{'path':str(artifact),'artifact_type':'report','required_json_keys':['coverage']}],[{'path':str(artifact),'checksum':checksum}])

def test_progress_counter_is_monotonic_and_separate_from_heartbeat(tmp_path):
    queue=JobQueue(tmp_path/'queue.sqlite'); add_ready(queue); job=queue.claim('worker')
    assert queue.record_progress('job',job['lease_token'],1)
    assert not queue.record_progress('job',job['lease_token'],1)
    assert queue.record_progress('job',job['lease_token'],2)


def test_failure_retry_disk_and_deadline_guards(tmp_path):
    queue = JobQueue(tmp_path / "queue.sqlite"); add_ready(queue)
    job = queue.claim("worker"); queue.fail("job", job["lease_token"], "temporary io")
    assert queue.status()[0]["status"] == RETRY_WAIT
    guarded = JobQueue(tmp_path / "guard.sqlite")
    add_ready(guarded, "disk", min_free_bytes=10**30); add_ready(guarded, "late", deadline_utc_epoch=time.time() - 1)
    assert guarded.claim("worker") is None
    rows = {row["job_key"]: row for row in guarded.status()}
    assert rows["disk"]["status"] == BLOCKED and "disk budget" in rows["disk"]["failure"]
    assert rows["late"]["status"] == BLOCKED and "deadline" in rows["late"]["failure"]

def test_three_retries_use_all_three_backoffs_and_integrity_blocks(tmp_path):
    queue=JobQueue(tmp_path/'queue.sqlite'); add_ready(queue); job=queue.claim('worker')
    queue.fail('job',job['lease_token'],'temporary io')
    with queue.connect() as db: db.execute("UPDATE jobs SET not_before=0 WHERE job_key='job'")
    for _ in range(2):
        job=queue.claim('worker'); queue.fail('job',job['lease_token'],'temporary io')
        with queue.connect() as db: db.execute("UPDATE jobs SET not_before=0 WHERE job_key='job'")
    job=queue.claim('worker'); queue.fail('job',job['lease_token'],'temporary io')
    assert queue.status()[0]['status']==BLOCKED
    queue=JobQueue(tmp_path/'integrity.sqlite'); add_ready(queue); job=queue.claim('worker'); queue.fail('job',job['lease_token'],'NaN gradient')
    assert queue.status()[0]['status']==BLOCKED


def test_matrix_is_registered_but_formal_work_is_not_claimable(tmp_path):
    queue = JobQueue(tmp_path / "queue.sqlite"); queue.register_matrix()
    rows = queue.status(); assert len(rows) == 28
    assert sum(row["kind"] == "formal_train" for row in rows) == 12
    assert queue.claim("worker") is None
