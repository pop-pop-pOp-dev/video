import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import nc_rted.queue as queue_module
from nc_rted.queue import BLOCKED, RETRY_WAIT, RUNNING, SUCCEEDED, JobQueue, QueueError
from nc_rted.worker_runtime import process_starttime

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "nc_rted_queue.py"
spec = importlib.util.spec_from_file_location("nc_rted_queue_script", SCRIPT)
worker_script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker_script)


def payload(tmp_path, output, command):
    return {
        "command": command,
        "physical_gpu": 0,
        "data_volume": str(tmp_path),
        "run_dir": str(tmp_path / "runs"),
        "expected_outputs": [{"path": output.name, "artifact_type": "smoke"}],
        "poll_seconds": 0.05,
    }


def wait_for_pid(queue, key="job", timeout=5):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        row = queue.status()[0]
        if row["pid"]:
            return row
        time.sleep(.02)
    raise AssertionError("worker did not publish child identity")


def cli_worker(database, once=True):
    command = [sys.executable, str(SCRIPT), "--db", str(database), "worker"]
    if once:
        command.append("--once")
    return subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def test_controller_death_adopts_same_child_and_commits_once(tmp_path):
    db = tmp_path / "queue.sqlite"; queue = JobQueue(db); output = tmp_path / "output.json"
    code = "import pathlib,time,os,json,hashlib; time.sleep(.45); p=pathlib.Path('output.json'); p.write_text('{}'); r={'job_key':os.environ['NC_RTED_JOB_KEY'],'lease_token':os.environ['NC_RTED_LEASE_TOKEN'],'input_hash':os.environ['NC_RTED_INPUT_HASH'],'artifacts':[{'path':str(p.resolve()),'checksum':hashlib.sha256(p.read_bytes()).hexdigest()}]}; pathlib.Path(os.environ['NC_RTED_PRODUCER_COMPLETION']).write_text(json.dumps(r))"
    queue.add_job("job", "smoke", payload(tmp_path, output, [sys.executable, "-c", code]))
    controller = cli_worker(db)
    row = wait_for_pid(queue); child_pid, token = row["pid"], None
    os.kill(controller.pid, signal.SIGKILL); controller.wait(timeout=5)
    assert Path(f"/proc/{child_pid}").exists()
    successor = cli_worker(db); code = successor.wait(timeout=8)
    stdout, stderr = successor.communicate()
    assert code == 0, stderr or stdout
    assert queue.status()[0]["status"] == SUCCEEDED
    assert queue.status()[0]["attempts"] == 1
    assert any((tmp_path / "runs" / "job").glob("attempt-*/result.json"))


def test_post_journal_crash_repairs_and_adopts_live_child_without_relaunch(tmp_path, monkeypatch):
    queue=JobQueue(tmp_path / "queue.sqlite"); output=tmp_path / "output.json"
    queue.add_job("job", "smoke", payload(tmp_path, output, ["/bin/true"]))
    job=queue.claim(f"{socket.gethostname()}:dead")
    run=worker_script.attempt_directory(json.loads(job["payload"]),job); run.mkdir(parents=True)
    journal=run / f"attempt-{job['attempts']}.json"
    queue.start_attempt_journal("job",job["lease_token"],None,None,"0",journal,state="LAUNCHING")
    child=subprocess.Popen([sys.executable,"-c","import time; time.sleep(5)"],start_new_session=True)
    try:
        start=process_starttime(child.pid)
        monkeypatch.setattr(queue_module,"after_attempt_journal_write",lambda: (_ for _ in ()).throw(RuntimeError("injected journal/SQL crash")))
        with pytest.raises(RuntimeError,match="injected"):
            queue.update_attempt_journal("job",job["lease_token"],child.pid,start,"0",journal,member_identities={child.pid:start})
        monkeypatch.setattr(queue_module,"after_attempt_journal_write",lambda: None)
        assert queue.status()[0]["pid"] is None and queue.running_attempts()[0]["state"] == "LAUNCHING"
        adopted=worker_script.adopt_live(queue,f"{socket.gethostname()}:replacement")
        assert adopted is not None and child.poll() is None
        repaired=queue.running_attempts()[0]
        assert repaired["state"] == "RUNNING" and repaired["pid"] == child.pid
        assert queue.claim("another-controller") is None
        adopted[2].lock.close()
    finally:
        child.terminate(); child.wait(timeout=5)


def test_post_journal_crash_reconciles_exited_child_completion(tmp_path, monkeypatch):
    queue=JobQueue(tmp_path / "queue.sqlite"); output=tmp_path / "artifact"
    queue.add_job("job", "smoke", payload(tmp_path, output, ["/bin/true"]))
    job=queue.claim(f"{socket.gethostname()}:dead")
    run=worker_script.attempt_directory(json.loads(job["payload"]),job); run.mkdir(parents=True)
    journal=run / f"attempt-{job['attempts']}.json"; queue.start_attempt_journal("job",job["lease_token"],None,None,"0",journal,state="LAUNCHING")
    artifact=run / "artifact"; artifact.write_text("ok")
    records=[{"path":str(artifact),"checksum":hashlib.sha256(artifact.read_bytes()).hexdigest()}]
    (run / "producer_completion.json").write_text(json.dumps({"job_key":"job","lease_token":job["lease_token"],"input_hash":job["input_hash"],"artifacts":records}))
    (run / "result.tmp").write_text(json.dumps({"job_key":"job","lease_token":job["lease_token"],"artifacts":records}))
    child=subprocess.Popen(["/bin/true"],start_new_session=True); start=process_starttime(child.pid); child.wait(timeout=5)
    monkeypatch.setattr(queue_module,"after_attempt_journal_write",lambda: (_ for _ in ()).throw(RuntimeError("injected journal/SQL crash")))
    with pytest.raises(RuntimeError,match="injected"):
        queue.update_attempt_journal("job",job["lease_token"],child.pid,start,"0",journal,member_identities={child.pid:start})
    monkeypatch.setattr(queue_module,"after_attempt_journal_write",lambda: None)
    worker_script.reconcile_exited_attempts(queue,f"{socket.gethostname()}:replacement")
    assert queue.status()[0]["status"] == SUCCEEDED and queue.status()[0]["attempts"] == 1


@pytest.mark.parametrize("record",[
    {"job_key":"other","lease_token":"wrong","pid":999999,"process_starttime":"1","gpu_identity":"0","state":"RUNNING","member_identities":{"999999":"1"}},
    {"job_key":"job","lease_token":None,"pid":999999,"process_starttime":"1","gpu_identity":"0","state":"RUNNING","member_identities":{"999999":"wrong"}},
])
def test_launch_repair_rejects_foreign_or_invalid_member_journal(tmp_path, record):
    queue=JobQueue(tmp_path / "queue.sqlite"); output=tmp_path / "artifact"
    queue.add_job("job", "smoke", payload(tmp_path, output, ["/bin/true"]))
    job=queue.claim(f"{socket.gethostname()}:dead")
    run=worker_script.attempt_directory(json.loads(job["payload"]),job); run.mkdir(parents=True)
    journal=run / f"attempt-{job['attempts']}.json"; queue.start_attempt_journal("job",job["lease_token"],None,None,"0",journal,state="LAUNCHING")
    record["lease_token"] = record["lease_token"] or job["lease_token"]
    from nc_rted.worker_runtime import write_journal
    write_journal(journal,record)
    worker_script.reconcile_exited_attempts(queue,f"{socket.gethostname()}:replacement")
    row=queue.status()[0]
    assert row["status"] == RUNNING and row["pid"] is None and "launch intent" in row["failure"]


def test_launch_repair_rejects_journal_from_another_path(tmp_path):
    queue=JobQueue(tmp_path / "queue.sqlite"); output=tmp_path / "artifact"
    queue.add_job("job", "smoke", payload(tmp_path, output, ["/bin/true"]))
    job=queue.claim(f"{socket.gethostname()}:dead")
    run=worker_script.attempt_directory(json.loads(job["payload"]),job); run.mkdir(parents=True)
    journal=run / f"attempt-{job['attempts']}.json"; queue.start_attempt_journal("job",job["lease_token"],None,None,"0",journal,state="LAUNCHING")
    foreign=tmp_path / "foreign-attempt.json"
    from nc_rted.worker_runtime import write_journal
    write_journal(foreign,{"job_key":"job","lease_token":job["lease_token"],"pid":999999,"process_starttime":"1","gpu_identity":"0","state":"RUNNING","member_identities":{"999999":"1"}})
    with queue.connect() as db: db.execute("UPDATE attempt_journal SET journal_path=? WHERE lease_token=?",(str(foreign),job["lease_token"]))
    worker_script.reconcile_exited_attempts(queue,f"{socket.gethostname()}:replacement")
    row=queue.status()[0]
    assert row["status"] == RUNNING and row["pid"] is None and "launch intent" in row["failure"]


@pytest.mark.parametrize("contents", ["[]", "null"])
def test_launch_repair_rejects_non_object_journal(tmp_path, contents):
    queue=JobQueue(tmp_path / "queue.sqlite"); output=tmp_path / "artifact"
    queue.add_job("job", "smoke", payload(tmp_path, output, ["/bin/true"]))
    job=queue.claim(f"{socket.gethostname()}:dead")
    run=worker_script.attempt_directory(json.loads(job["payload"]),job); run.mkdir(parents=True)
    journal=run / f"attempt-{job['attempts']}.json"; queue.start_attempt_journal("job",job["lease_token"],None,None,"0",journal,state="LAUNCHING")
    journal.write_text(contents)
    assert not queue.repair_launching_journal("job",job["lease_token"],job["attempts"],journal)


def test_adoption_does_not_repair_foreign_host_launch(tmp_path):
    queue=JobQueue(tmp_path / "queue.sqlite"); output=tmp_path / "artifact"
    queue.add_job("job", "smoke", payload(tmp_path, output, ["/bin/true"]))
    job=queue.claim("remote-host:dead")
    run=worker_script.attempt_directory(json.loads(job["payload"]),job); run.mkdir(parents=True)
    journal=run / f"attempt-{job['attempts']}.json"; queue.start_attempt_journal("job",job["lease_token"],None,None,"0",journal,state="LAUNCHING")
    from nc_rted.worker_runtime import write_journal
    write_journal(journal,{"job_key":"job","lease_token":job["lease_token"],"pid":999999,"process_starttime":"1","gpu_identity":"0","state":"RUNNING","member_identities":{"999999":"1"}})
    assert worker_script.adopt_live(queue,f"{socket.gethostname()}:replacement") is None
    assert queue.running_attempts()[0]["state"] == "LAUNCHING" and queue.status()[0]["pid"] is None


def test_restart_commits_existing_result_temp_without_new_attempt(tmp_path):
    db = tmp_path / "queue.sqlite"; queue = JobQueue(db); output = tmp_path / "artifact.json"; output.write_text("{}")
    queue.add_job("job", "smoke", payload(tmp_path, output, ["/bin/false"]))
    owner = f"{socket.gethostname()}:old"; job = queue.claim(owner)
    run = worker_script.attempt_directory(json.loads(job["payload"]), job); run.mkdir(parents=True)
    artifact=run/'artifact.json'; artifact.write_text('{}'); records = [{"path": str(artifact), "checksum": hashlib.sha256(artifact.read_bytes()).hexdigest()}]
    (run / "result.tmp").write_text(json.dumps({"job_key": "job", "lease_token": job["lease_token"], "artifacts": records}) + "\n")
    (run / "producer_completion.json").write_text(json.dumps({"job_key":"job","lease_token":job["lease_token"],"input_hash":job["input_hash"],"artifacts":records}))
    queue.start_attempt_journal("job", job["lease_token"], 999999, "0", "0", run / "attempt-1.json")
    worker_script.reconcile_exited_attempts(queue, f"{socket.gethostname()}:new")
    assert queue.status()[0]["status"] == SUCCEEDED


def test_standalone_recover_reconciles_durable_result_before_expiry(tmp_path):
    db=tmp_path/'queue.sqlite'; queue=JobQueue(db); output=tmp_path/'artifact'; output.write_text('{}')
    queue.add_job('job','smoke',payload(tmp_path,output,['/bin/false']))
    job=queue.claim(f'{socket.gethostname()}:old', lease_seconds=.001)
    run=worker_script.attempt_directory(json.loads(job['payload']),job); run.mkdir(parents=True)
    artifact=run/'artifact'; artifact.write_text('{}'); records=[{'path':str(artifact),'checksum':hashlib.sha256(artifact.read_bytes()).hexdigest()}]
    (run/'producer_completion.json').write_text(json.dumps({'job_key':'job','lease_token':job['lease_token'],'input_hash':job['input_hash'],'artifacts':records}))
    (run/'result.tmp').write_text(json.dumps({'job_key':'job','lease_token':job['lease_token'],'artifacts':records}))
    queue.start_attempt_journal('job',job['lease_token'],999999,'0','0',run/'attempt.json')
    time.sleep(.01)
    result=subprocess.run([sys.executable,str(SCRIPT),'--db',str(db),'recover'],cwd=ROOT,capture_output=True,text=True,timeout=5)
    assert result.returncode == 0 and queue.status()[0]['status'] == SUCCEEDED
    assert queue.status()[0]["attempts"] == 1
    worker_script.reconcile_exited_attempts(queue, f"{socket.gethostname()}:new")
    assert queue.status()[0]["status"] == SUCCEEDED


def test_spawn_failure_retries_without_polling_none(tmp_path):
    db = tmp_path / "queue.sqlite"; queue = JobQueue(db); output = tmp_path / "none"
    queue.add_job("job", "smoke", payload(tmp_path, output, [str(tmp_path / "does-not-exist")]))
    result = cli_worker(db); assert result.wait(timeout=5) == 0
    assert queue.status()[0]["status"] == RETRY_WAIT


def test_retry_cannot_complete_from_previous_attempt_artifact(tmp_path):
    db=tmp_path/'queue.sqlite'; queue=JobQueue(db); output=tmp_path/'artifact'; output.write_text('old')
    queue.add_job('job','smoke',payload(tmp_path,output,['/bin/true']))
    first=queue.claim(f'{socket.gethostname()}:old')
    run=worker_script.attempt_directory(json.loads(first['payload']),first); run.mkdir(parents=True)
    records=[{'path':str(output),'checksum':hashlib.sha256(output.read_bytes()).hexdigest()}]
    (run/'producer_completion.json').write_text(json.dumps({'job_key':'job','lease_token':first['lease_token'],'input_hash':first['input_hash'],'artifacts':records}))
    queue.fail('job',first['lease_token'],'transient failure')
    with queue.connect() as sql: sql.execute("UPDATE jobs SET not_before=0 WHERE job_key='job'")
    assert cli_worker(db).wait(timeout=5) == 0
    assert queue.status()[0]['status'] == RETRY_WAIT


def test_foreign_and_identity_mismatch_attempts_are_protected(tmp_path):
    output = tmp_path / "artifact"; output.write_text("ok")
    queue = JobQueue(tmp_path / "queue.sqlite"); queue.add_job("job", "smoke", payload(tmp_path, output, ["/bin/true"]))
    job = queue.claim(f"{socket.gethostname()}:old")
    queue.start_attempt_journal("job", job["lease_token"], os.getpid(), "wrong-start", "0", tmp_path / "attempt.json")
    worker_script.reconcile_exited_attempts(queue, f"{socket.gethostname()}:new")
    row = queue.status()[0]; assert row["status"] == RUNNING and "identity mismatch" in row["failure"]

    foreign = JobQueue(tmp_path / "foreign.sqlite"); foreign.add_job("job", "smoke", payload(tmp_path, output, ["/bin/true"]))
    job = foreign.claim("other-host:old")
    foreign.start_attempt_journal("job", job["lease_token"], 999999, "0", "0", tmp_path / "foreign.json")
    worker_script.reconcile_exited_attempts(foreign, f"{socket.gethostname()}:new")
    assert foreign.status()[0]["status"] == RUNNING


def test_progress_timeout_uses_p99_floor():
    assert worker_script.progress_timeout({}) == 1800
    assert worker_script.progress_timeout({"progress_p99_seconds": 500}) == 2500


def test_expired_lease_with_surviving_child_or_pid_mismatch_stays_reserved(tmp_path):
    output = tmp_path / "out"; output.write_text("ok")
    queue = JobQueue(tmp_path / "queue.sqlite"); queue.add_job("job", "smoke", payload(tmp_path, output, ["/bin/true"]))
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)"], start_new_session=True)
    try:
        job = queue.claim(f"{socket.gethostname()}:old", lease_seconds=.001)
        queue.heartbeat("job", job["lease_token"], lease_seconds=.001, pid=child.pid, process_starttime=process_starttime(child.pid))
        queue.start_attempt_journal("job", job["lease_token"], child.pid, process_starttime(child.pid), "0", tmp_path / "attempt.json")
        time.sleep(.01); assert queue.recover_expired() == 0
        assert queue.status()[0]["status"] == RUNNING
        with queue.connect() as db: db.execute("UPDATE jobs SET process_starttime='wrong', lease_expires=0 WHERE job_key='job'")
        assert queue.recover_expired() == 0
        assert queue.status()[0]["status"] == RUNNING
    finally:
        child.terminate(); child.wait(timeout=5)


def test_leader_exit_with_live_descendant_remains_protected(tmp_path):
    db = tmp_path / "queue.sqlite"; queue = JobQueue(db); output = tmp_path / "out"
    code = "import subprocess,sys; subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(5)'])"
    queue.add_job("job", "smoke", payload(tmp_path, output, [sys.executable, "-c", code]))
    controller = cli_worker(db); assert controller.wait(timeout=5) == 1
    row = queue.status()[0]; assert row["status"] == RUNNING and "group retained" in row["failure"]
    os.killpg(row["pid"], signal.SIGKILL)


def test_zombie_leader_reconciles_after_group_is_gone(tmp_path):
    output = tmp_path / "out"; output.write_text("ok")
    queue = JobQueue(tmp_path / "queue.sqlite"); queue.add_job("job", "smoke", payload(tmp_path, output, ["/bin/true"]))
    child = subprocess.Popen(["/bin/true"], start_new_session=True)
    time.sleep(.05)
    try:
        job = queue.claim(f"{socket.gethostname()}:old")
        queue.heartbeat("job", job["lease_token"], pid=child.pid, process_starttime=process_starttime(child.pid))
        run=worker_script.attempt_directory(json.loads(job["payload"]),job); run.mkdir(parents=True)
        artifact=run/'out'; artifact.write_text('ok'); records=[{"path":str(artifact),"checksum":hashlib.sha256(artifact.read_bytes()).hexdigest()}]
        (run/'producer_completion.json').write_text(json.dumps({'job_key':'job','lease_token':job['lease_token'],'input_hash':job['input_hash'],'artifacts':records}))
        queue.start_attempt_journal("job", job["lease_token"], child.pid, process_starttime(child.pid), "0", run / "attempt.json")
        worker_script.reconcile_exited_attempts(queue, f"{socket.gethostname()}:new")
        assert queue.status()[0]["status"] == SUCCEEDED
    finally:
        child.wait(timeout=5)


def test_adopted_transient_queue_error_does_not_kill_child(tmp_path, monkeypatch):
    output = tmp_path / "out"; output.write_text("ok")
    queue = JobQueue(tmp_path / "queue.sqlite"); queue.add_job("job", "smoke", payload(tmp_path, output, ["/bin/true"]))
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)"], start_new_session=True)
    try:
        job = queue.claim(f"{socket.gethostname()}:old")
        start = process_starttime(child.pid)
        queue.heartbeat("job", job["lease_token"], pid=child.pid, process_starttime=start)
        queue.start_attempt_journal("job", job["lease_token"], child.pid, start, "0", tmp_path / "attempt.json")
        monkeypatch.setattr(queue, "runtime_guard", lambda *_: (_ for _ in ()).throw(RuntimeError("transient database error")))
        assert worker_script.worker(queue, f"{socket.gethostname()}:new", once=True) == 1
        assert child.poll() is None and queue.status()[0]["status"] == RUNNING
    finally:
        child.terminate(); child.wait(timeout=5)


def test_transient_error_text_never_authorizes_group_kill(tmp_path, monkeypatch):
    output = tmp_path / "out"; output.write_text("ok")
    queue = JobQueue(tmp_path / "queue.sqlite"); queue.add_job("job", "smoke", payload(tmp_path, output, ["/bin/true"]))
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)"], start_new_session=True)
    try:
        job=queue.claim(f"{socket.gethostname()}:old"); start=process_starttime(child.pid)
        queue.heartbeat("job",job["lease_token"],pid=child.pid,process_starttime=start)
        queue.start_attempt_journal("job",job["lease_token"],child.pid,start,"0",tmp_path/'attempt.json')
        monkeypatch.setattr(queue, "runtime_guard", lambda *_: (_ for _ in ()).throw(RuntimeError("banana file error")))
        assert worker_script.worker(queue, f"{socket.gethostname()}:new", once=True) == 1
        assert child.poll() is None
    finally:
        child.terminate(); child.wait(timeout=5)


def test_progress_requires_attempt_local_committed_evidence(tmp_path):
    output=tmp_path/'out'; queue=JobQueue(tmp_path/'queue.sqlite'); queue.add_job('job','smoke',payload(tmp_path,output,['/bin/true']))
    job=queue.claim('owner'); run=worker_script.attempt_directory(json.loads(job['payload']),job); run.mkdir(parents=True)
    progress=run/'progress.json'; progress.write_text(json.dumps({'job_key':'job','lease_token':job['lease_token'],'counter':5,'committed_path':str(run/'missing.json')}))
    worker_script.observe_progress(queue,job,str(progress))
    assert queue.status()[0]['progress_counter'] is None
    evidence=run/'checkpoint.json'; evidence.write_text(json.dumps({'job_key':'job','lease_token':job['lease_token'],'counter':5}))
    artifact=run/'media.json'
    results=[]
    for identifier in ['a','b','c','d','e']:
        result=run/f'{identifier}.json'; result.write_text(identifier)
        results.append({'id':identifier,'path':str(result),'sha256':hashlib.sha256(result.read_bytes()).hexdigest()})
    artifact.write_text(json.dumps({'schema':'nc_rted_media_commit_v1','job_key':'job','lease_token':job['lease_token'],'input_hash':job['input_hash'],'committed_ids':['a','b','c','d','e'],'results':results}))
    evidence.write_text(json.dumps({'schema':'nc_rted_progress_commit_v1','transaction_type':'media','job_key':'job','lease_token':job['lease_token'],'counter':5,'artifact_path':str(artifact),'artifact_sha256':hashlib.sha256(artifact.read_bytes()).hexdigest()}))
    progress.write_text(json.dumps({'job_key':'job','lease_token':job['lease_token'],'counter':5,'committed_path':str(evidence)}))
    worker_script.observe_progress(queue,job,str(progress))
    assert queue.status()[0]['progress_counter'] == 5


def test_media_progress_without_committed_result_records_is_rejected(tmp_path):
    output=tmp_path/'out'; queue=JobQueue(tmp_path/'queue.sqlite'); queue.add_job('job','smoke',payload(tmp_path,output,['/bin/true']))
    job=queue.claim('owner'); run=worker_script.attempt_directory(json.loads(job['payload']),job); run.mkdir(parents=True)
    artifact=run/'media.json'; artifact.write_text(json.dumps({'schema':'nc_rted_media_commit_v1','job_key':'job','lease_token':job['lease_token'],'input_hash':job['input_hash'],'committed_ids':['a']}))
    commit=run/'commit.json'; commit.write_text(json.dumps({'schema':'nc_rted_progress_commit_v1','transaction_type':'media','job_key':'job','lease_token':job['lease_token'],'counter':1,'artifact_path':str(artifact),'artifact_sha256':hashlib.sha256(artifact.read_bytes()).hexdigest()}))
    progress=run/'progress.json'; progress.write_text(json.dumps({'job_key':'job','lease_token':job['lease_token'],'counter':1,'committed_path':str(commit)}))
    worker_script.observe_progress(queue,job,str(progress))
    assert queue.status()[0]['progress_counter'] is None


def test_producer_completion_alone_preserves_expired_attempt(tmp_path):
    output=tmp_path/'out'; queue=JobQueue(tmp_path/'queue.sqlite'); queue.add_job('job','smoke',payload(tmp_path,output,['/bin/true']))
    job=queue.claim(f'{socket.gethostname()}:old',lease_seconds=.001); run=worker_script.attempt_directory(json.loads(job['payload']),job); run.mkdir(parents=True)
    (run/'producer_completion.json').write_text('{}')
    queue.start_attempt_journal('job',job['lease_token'],999999,'0','0',run/'attempt.json')
    time.sleep(.01); assert queue.recover_expired() == 0 and queue.status()[0]['status'] == RUNNING


def test_adopted_leader_exit_with_descendant_exits_for_restart(tmp_path):
    output=tmp_path/'out'; queue=JobQueue(tmp_path/'queue.sqlite'); queue.add_job('job','smoke',payload(tmp_path,output,['/bin/true']))
    leader=subprocess.Popen([sys.executable,'-c',"import subprocess,sys,time; subprocess.Popen([sys.executable,'-c','import time; time.sleep(5)']); time.sleep(.3)"],start_new_session=True)
    start=process_starttime(leader.pid)
    try:
        job=queue.claim(f'{socket.gethostname()}:dead'); queue.heartbeat('job',job['lease_token'],pid=leader.pid,process_starttime=start)
        run=worker_script.attempt_directory(json.loads(job['payload']),job); run.mkdir(parents=True)
        queue.start_attempt_journal('job',job['lease_token'],leader.pid,start,'0',run/'attempt.json')
        started=time.monotonic(); assert worker_script.worker(queue,f'{socket.gethostname()}:{os.getpid()}',once=True) == 0
        assert time.monotonic()-started >= .25
        assert queue.status()[0]['status'] == RETRY_WAIT
    finally:
        with __import__('contextlib').suppress(ProcessLookupError): os.killpg(leader.pid,signal.SIGKILL)
        leader.wait(timeout=5)


def test_competing_controller_cannot_acquire_attempt_supervision_lock(tmp_path):
    output=tmp_path/'out'; queue=JobQueue(tmp_path/'queue.sqlite'); queue.add_job('job','smoke',payload(tmp_path,output,['/bin/true']))
    child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(5)'],start_new_session=True)
    try:
        job=queue.claim(f'{socket.gethostname()}:dead'); start=process_starttime(child.pid)
        queue.heartbeat('job',job['lease_token'],pid=child.pid,process_starttime=start)
        run=worker_script.attempt_directory(json.loads(job['payload']),job); run.mkdir(parents=True)
        queue.start_attempt_journal('job',job['lease_token'],child.pid,start,'0',run/'attempt.json')
        first=worker_script.adopt_live(queue,f'{socket.gethostname()}:{os.getpid()}')
        assert first is not None
        assert worker_script.adopt_live(queue,f'{socket.gethostname()}:other-dead') is None
        assert queue.status()[0]['lease_owner'] if 'lease_owner' in queue.status()[0] else True
        assert child.poll() is None
        first[2].lock.close()
    finally:
        child.terminate(); child.wait(timeout=5)


def test_commit_exception_after_result_publication_remains_recoverable(tmp_path, monkeypatch):
    db=tmp_path/'queue.sqlite'; queue=JobQueue(db); output=tmp_path/'out'
    code="import pathlib,os,json,hashlib; p=pathlib.Path('out'); p.write_text('ok'); pathlib.Path(os.environ['NC_RTED_PRODUCER_COMPLETION']).write_text(json.dumps({'job_key':os.environ['NC_RTED_JOB_KEY'],'lease_token':os.environ['NC_RTED_LEASE_TOKEN'],'input_hash':os.environ['NC_RTED_INPUT_HASH'],'artifacts':[{'path':str(p.resolve()),'checksum':hashlib.sha256(p.read_bytes()).hexdigest()}]}))"
    queue.add_job('job','smoke',payload(tmp_path,output,[sys.executable,'-c',code]))
    original=queue.commit; calls={'n':0}
    def fail_once(*args,**kwargs):
        calls['n']+=1
        if calls['n']==1:
            temporary, final=Path(args[2]),Path(args[3]); os.replace(temporary,final)
            raise RuntimeError('injected post-rename SQL error')
        return original(*args,**kwargs)
    monkeypatch.setattr(queue,'commit',fail_once)
    assert worker_script.worker(queue,f'{socket.gethostname()}:{os.getpid()}',once=True) == 1
    assert queue.status()[0]['status'] == RUNNING
    monkeypatch.setattr(queue,'commit',original)
    worker_script.reconcile_exited_attempts(queue,f'{socket.gethostname()}:{os.getpid()}')
    assert queue.status()[0]['status'] == SUCCEEDED


def test_reconciliation_commit_error_retains_lease_for_retry(tmp_path, monkeypatch):
    queue=JobQueue(tmp_path/'queue.sqlite'); output=tmp_path/'out'; queue.add_job('job','smoke',payload(tmp_path,output,['/bin/true']))
    job=queue.claim(f'{socket.gethostname()}:dead'); run=worker_script.attempt_directory(json.loads(job['payload']),job); run.mkdir(parents=True)
    artifact=run/'out'; artifact.write_text('ok'); records=[{'path':str(artifact),'checksum':hashlib.sha256(artifact.read_bytes()).hexdigest()}]
    (run/'producer_completion.json').write_text(json.dumps({'job_key':'job','lease_token':job['lease_token'],'input_hash':job['input_hash'],'artifacts':records}))
    queue.start_attempt_journal('job',job['lease_token'],999999,'0','0',run/'attempt.json')
    original=queue.commit; monkeypatch.setattr(queue,'commit',lambda *a,**k: (_ for _ in ()).throw(RuntimeError('temporary commit read error')))
    owner=f'{socket.gethostname()}:{os.getpid()}'; worker_script.reconcile_exited_attempts(queue,owner)
    assert queue.status()[0]['status'] == RUNNING
    monkeypatch.setattr(queue,'commit',original); worker_script.reconcile_exited_attempts(queue,owner)
    assert queue.status()[0]['status'] == SUCCEEDED


def test_gone_attempt_without_outputs_enters_retry(tmp_path):
    queue=JobQueue(tmp_path/'queue.sqlite'); output=tmp_path/'out'; queue.add_job('job','smoke',payload(tmp_path,output,['/bin/true']))
    job=queue.claim(f'{socket.gethostname()}:dead'); run=worker_script.attempt_directory(json.loads(job['payload']),job); run.mkdir(parents=True)
    queue.start_attempt_journal('job',job['lease_token'],999999,'0','0',run/'attempt.json')
    worker_script.reconcile_exited_attempts(queue,f'{socket.gethostname()}:{os.getpid()}')
    assert queue.status()[0]['status'] == RETRY_WAIT


def test_hard_limit_marker_blocks_reconciliation_success(tmp_path):
    queue=JobQueue(tmp_path/'queue.sqlite'); output=tmp_path/'out'; queue.add_job('job','smoke',payload(tmp_path,output,['/bin/true']))
    job=queue.claim(f'{socket.gethostname()}:dead'); run=worker_script.attempt_directory(json.loads(job['payload']),job); run.mkdir(parents=True)
    artifact=run/'out'; artifact.write_text('ok'); records=[{'path':str(artifact),'checksum':hashlib.sha256(artifact.read_bytes()).hexdigest()}]
    (run/'producer_completion.json').write_text(json.dumps({'job_key':'job','lease_token':job['lease_token'],'input_hash':job['input_hash'],'artifacts':records}))
    queue.start_attempt_journal('job',job['lease_token'],999999,'0','0',run/'attempt.json')
    queue.record_protective_stop('job',job['lease_token'],'deadline')
    worker_script.reconcile_exited_attempts(queue,f'{socket.gethostname()}:{os.getpid()}')
    assert queue.status()[0]['status'] == BLOCKED


def test_recovery_owner_can_be_replaced_after_its_process_is_gone(tmp_path):
    output=tmp_path/'out'; queue=JobQueue(tmp_path/'queue.sqlite'); queue.add_job('job','smoke',payload(tmp_path,output,['/bin/true']))
    job=queue.claim(f'{socket.gethostname()}:recover:999999');
    assert not worker_script.controller_may_live(f'{socket.gethostname()}:recover:999999')
