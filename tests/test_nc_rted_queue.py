import hashlib
import json
import os
import threading
import time
from pathlib import Path
import sys
import subprocess
import importlib.util

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nc_rted.queue import BLOCKED, RETRY_WAIT, RUNNING, SUCCEEDED, JobQueue, LeaseLost
from nc_rted import resource_attestation
from nc_rted.production_runtime import load_manifest
from nc_rted.recovery import CheckpointStore
from nc_rted.worker_runtime import gpu_lock
import nc_rted.queue as queue_module

_QUEUE_SCRIPT=Path(__file__).resolve().parents[1]/"scripts"/"nc_rted_queue.py"
_QUEUE_SPEC=importlib.util.spec_from_file_location("nc_rted_queue_script",_QUEUE_SCRIPT)
queue_script=importlib.util.module_from_spec(_QUEUE_SPEC); _QUEUE_SPEC.loader.exec_module(queue_script)


def _real_catalog_fixture(root, document):
    """Small-on-disk but schema-real fixed 6,000 + 2,000 training catalog."""
    annotations=[]; captions=[]; detections=[]
    for dataset in ("ucf-crime", "xd-violence"):
        for index in range(3000):
            detections.append({"family":f"{dataset}:family","dataset":dataset,"class":"anomalous" if index < 1500 else "normal",
                               "observed_seconds":0.0,"query_index":index,"key":f"{dataset}-{index}"})
    for index in range(2000):
        item={"id":f"caption-{index}","video":f"caption-video-{index}","task":"caption","type":"clip",
              "conversations":[{"from":"human","value":"<video>"},{"from":"gpt","value":"description"}]}
        annotations.append(item); captions.append({**{key:item[key] for key in ("id","video","task","type")},"dataset":"ucf-crime","parent_key":"family"})
    paths={"source_splits.json":[{"family":"ucf-crime:family","allocation":"train"},{"family":"xd-violence:family","allocation":"train"}],
           "train8000_captions.json":captions,"train8000_detection_prefixes.json":detections}
    annotation_path=Path(document["catalog"]["training_annotations"]); annotation_path.write_text(json.dumps(annotations))
    output_hashes={}
    for name,value in paths.items():
        path=root/name; path.write_text(json.dumps(value)); output_hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    provenance={"schema":"nc_rted_manifest_provenance/v1","inputs":{str(annotation_path.resolve()):hashlib.sha256(annotation_path.read_bytes()).hexdigest()},"outputs":output_hashes}
    provenance_path=root/"provenance.json"; provenance_path.write_text(json.dumps(provenance))
    document["catalog"]["training_annotations_sha256"]=hashlib.sha256(annotation_path.read_bytes()).hexdigest()
    document["catalog"]["provenance_sha256"]=hashlib.sha256(provenance_path.read_bytes()).hexdigest()


def formal_fixture(tmp_path, monkeypatch, *, real_probe=False):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_nc_rted_production_runtime import config
    runtime, _ = config(tmp_path, mode="formal")
    document = json.loads(runtime.read_text())
    volume = resource_attestation.PROJECT_VOLUME.resolve()
    run_root = volume / ".cache" / "nc_rted_queue_tests" / tmp_path.name
    run_root.mkdir(parents=True, exist_ok=True)
    document["run"].update({"checkpoint_root":str(run_root/"checkpoints"), "progress_path":str(run_root/"progress.json")})
    _real_catalog_fixture(tmp_path, document)
    source_root = Path(__file__).resolve().parents[1] / "src" / "nc_rted"
    source_files = {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_root.glob("*.py")}
    for name in ("nc_rted_train.py", "nc_rted_queue.py"):
        path=Path(__file__).resolve().parents[1]/"scripts"/name; source_files[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
    source = hashlib.sha256(json.dumps(source_files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    document["hashes"]["code_sha256"] = source
    runtime.write_text(json.dumps(document))
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    manifest=load_manifest(runtime, expected_sha256=digest(runtime))
    identity=resource_attestation.formal_runtime_identity(manifest)
    admission = tmp_path / "admission.json"
    admission.write_text(json.dumps({"status":"PASS","formal_execution_allowed":True,"engineering_checks":{str(i):"PASS" for i in range(1,11)},"source_files":source_files,"run_identity":identity}))
    monkeypatch.setattr(resource_attestation, "gpu_uuid", lambda index: "GPU-valid")
    monkeypatch.setattr(resource_attestation, "gpu_memory_bytes", lambda index: 32768 * 1024 ** 2)
    authorization=tmp_path/"authorization.json"
    authorization.write_text(json.dumps({"schema":"nc_rted_resource_authorization/v1","status":"PASS","host":__import__("socket").gethostname(),"gpu_uuid":"GPU-valid","lease_id":"lease","project_volume":str(volume),"max_budget_seconds":120,"min_free_bytes":20*1024**3,"deadline_utc_epoch":resource_attestation.FORMAL_DEADLINE,"lease_expires_utc_epoch":1792079990.0}))
    project_python=Path("/root/autodl-tmp/lookaway-wm/.venv-reactvau/bin/python")
    interpreter={"path":str(project_python),"launcher_sha256":digest(project_python),"target_sha256":digest(project_python.resolve())}
    cache_root=run_root/"cache"; cache_root.mkdir(parents=True, exist_ok=True)
    environment={key:str(cache_root/key.lower()) for key in resource_attestation.CACHE_KEYS}
    for value in environment.values(): Path(value).mkdir(parents=True, exist_ok=True)
    environment.update({"PYTHONPATH":str(Path(__file__).resolve().parents[1]/"src"),"PYTHONNOUSERSITE":"1","PYTHONDONTWRITEBYTECODE":"1"})
    probe_runtime={"fixture":"controlled-import-probe"}
    if real_probe:
        probe_runtime=resource_attestation.python_runtime_probe(interpreter, environment)
    else:
        monkeypatch.setattr(resource_attestation, "python_runtime_probe", lambda _interpreter, _environment: probe_runtime)
    now=time.time()
    qualification = tmp_path / "qualification.json"
    qualification.write_text(json.dumps({"schema":"nc_rted_runtime_qualification/v2","status":"PASS_GPU_KERNEL_AND_INHERITED_RUNTIME_IMPORTS","host":__import__("socket").gethostname(),"gpu_uuid":"GPU-valid","runtime_identity":identity,"source_sha256":source,"workload":{"run_identity":identity,"updates":1000,"kind":"formal_train"},"runtime_environment":{"interpreter":interpreter,"environment":environment},"resource_envelope":{"device_memory_bytes":32768*1024**2,"required_memory_bytes":1024},"measurements":{"measured_at_utc_epoch":now,"peak_cuda_allocated_bytes":512,"peak_cuda_reserved_bytes":768,"seconds_per_update_upper_bound":0.02,"setup_checkpoint_seconds_upper_bound":1,"forward_backward_completed":True,"complete_long_input":True,"optimizer_updates":1,"local_import_probe":{"status":"PASS","interpreter":interpreter,"environment":environment,"duration_seconds":0.01,"runtime_identity":probe_runtime}},"valid_from_utc_epoch":now-1,"valid_until_utc_epoch":1792080000.0}))
    payload = {"physical_gpu":0,"run_identity":identity,"runtime_config":str(runtime),"runtime_config_sha256":digest(runtime),"formal_admission":str(admission),"formal_admission_sha256":digest(admission),"frozen_source_sha256":source,"runtime_evidence":"runtime","formal_admission_evidence":"admission","resource_authorization_evidence":"authorization","_authorization_path":str(authorization),"data_volume":str(volume),"min_free_bytes":20*1024**3,"run_budget_seconds":60,"deadline_utc_epoch":resource_attestation.FORMAL_DEADLINE,"execution_environment":environment,"interpreter":interpreter,"run_dir":str(run_root/"runs"),"progress_path":document["run"]["progress_path"],"checkpoint_root":document["run"]["checkpoint_root"]}
    command=[interpreter["path"],str((Path(__file__).resolve().parents[1]/"scripts"/"nc_rted_train.py")),"--config",str(runtime),"--config-sha256",digest(runtime),"--mode","formal","--admission",str(admission),"--admission-sha256",digest(admission)]
    checkpoint={"path":str((Path(document["run"]["checkpoint_root"])/"final"/"manifest.json").resolve()),"artifact_type":"checkpoint","semantic":"formal_training","run_identity":identity}
    payload.update({"command":command,"expected_outputs":[checkpoint]})
    inputs={key:payload[key] for key in ("command","execution_environment","interpreter","run_dir","progress_path","checkpoint_root","expected_outputs")}
    att={"schema":"nc_rted_formal_resource_attestation/v1","status":"PASS","binding":{"job_key":"formal",**{k:payload[k] for k in ("runtime_config","runtime_config_sha256","formal_admission","formal_admission_sha256","run_identity","frozen_source_sha256")},"execution_inputs":inputs},"execution":{"host":__import__("socket").gethostname(),"physical_gpu":0,"gpu_uuid":"GPU-valid","runtime":"runtime-v1","environment":environment,"interpreter":interpreter,"lease_id":"lease","lease_expires_utc_epoch":1792079990.0},"qualification":{"accepted_evidence_name":"qualification","path":str(qualification),"sha256":digest(qualification),"status":"PASS_GPU_KERNEL_AND_INHERITED_RUNTIME_IMPORTS"},"authorization":{"path":str(authorization),"sha256":digest(authorization)},"contract":{k:payload[k] for k in ("data_volume","min_free_bytes","run_budget_seconds","deadline_utc_epoch")}}
    attestation=tmp_path/"attestation.json"; attestation.write_text(json.dumps(att)); payload.update({"resource_attestation":str(attestation),"resource_attestation_sha256":digest(attestation)})
    return payload, qualification, runtime, admission, attestation, digest


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


def test_legacy_reservations_migrate_without_reconstructing_authority(tmp_path):
    database=tmp_path/"queue.sqlite"
    connection=__import__("sqlite3").connect(database)
    connection.execute("CREATE TABLE reservations (lease_token TEXT PRIMARY KEY, job_key TEXT NOT NULL, lease_id TEXT NOT NULL, host TEXT NOT NULL, physical_gpu INTEGER NOT NULL, acquired_at REAL NOT NULL)")
    connection.execute("INSERT INTO reservations VALUES ('legacy','formal','lease','host',0,1)")
    connection.commit(); connection.close()
    JobQueue(database)
    connection=__import__("sqlite3").connect(database)
    columns={row[1] for row in connection.execute("PRAGMA table_info(reservations)")}
    legacy=connection.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'reservations_legacy_unverified_%'").fetchone()[0]
    assert {"lock_device","lock_inode","authorized_end"}.issubset(columns)
    assert connection.execute(f'SELECT lease_token FROM "{legacy}"').fetchone()[0] == "legacy"
    assert connection.execute("SELECT * FROM reservations").fetchone() is None
    connection.close()

def test_supervision_transfer_fences_displaced_controller(tmp_path):
    queue=JobQueue(tmp_path/'queue.sqlite'); add_ready(queue)
    job=queue.claim('first'); queue.take_supervision('job',job['lease_token'],'second')
    with pytest.raises(LeaseLost): queue.heartbeat('job',job['lease_token'],owner='first')
    queue.heartbeat('job',job['lease_token'],owner='second')

def test_protective_stop_is_dedicated_from_mutable_diagnostic_text(tmp_path):
    queue=JobQueue(tmp_path/'queue.sqlite'); add_ready(queue); job=queue.claim('owner')
    queue.record_protective_stop('job',job['lease_token'],'deadline')
    queue.protected_live('job',job['lease_token'],'later diagnostic')
    with queue.connect() as db:
        row=db.execute("SELECT protective_stop_code,failure FROM jobs WHERE job_key='job'").fetchone()
    assert row['protective_stop_code']=='deadline' and row['failure']=='later diagnostic'


def test_expired_protective_stop_blocks_only_after_identity_is_gone(tmp_path):
    import socket
    queue=JobQueue(tmp_path/'queue.sqlite'); add_ready(queue); job=queue.claim(f'{socket.gethostname()}:owner',lease_seconds=.001)
    queue.heartbeat('job',job['lease_token'],lease_seconds=.001,pid=999999,process_starttime='0')
    queue.record_protective_stop('job',job['lease_token'],'deadline')
    time.sleep(.01); assert queue.recover_expired()==1
    assert queue.status()[0]['status']==BLOCKED
    queue=JobQueue(tmp_path/'uncertain.sqlite'); add_ready(queue); job=queue.claim(f'{socket.gethostname()}:owner',lease_seconds=.001)
    queue.heartbeat('job',job['lease_token'],lease_seconds=.001,pid=os.getpid(),process_starttime='wrong')
    queue.record_protective_stop('job',job['lease_token'],'deadline')
    time.sleep(.01); assert queue.recover_expired()==0
    assert queue.status()[0]['status']==RUNNING


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
    row = queue.status()[0]; assert row["status"] == RETRY_WAIT and row["pid"] is None


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
    queue=JobQueue(tmp_path/'integrity.sqlite'); add_ready(queue); job=queue.claim('worker'); queue.fail('job',job['lease_token'],'NaN gradient',code='nan')
    assert queue.status()[0]['status']==BLOCKED


def test_matrix_is_registered_but_formal_work_is_not_claimable(tmp_path):
    queue = JobQueue(tmp_path / "queue.sqlite"); queue.register_matrix()
    rows = queue.status(); assert len(rows) == 28
    assert sum(row["kind"] == "formal_train" for row in rows) == 12
    assert queue.claim("worker") is None

def test_formal_command_is_closed_without_attested_resource_wiring(tmp_path):
    queue=JobQueue(tmp_path/'queue.sqlite')
    queue.add_job('formal','formal_train',{'command':['true'],'physical_gpu':0,'expected_outputs':[{'path':'/dev/null'}], 'resource_qualification':{'accepted':True,'gpu_uuid':'untrusted'}})
    assert queue.claim('worker') is None
    assert 'fixed local nc_rted_train' in queue.status()[0]['failure']


def test_formal_claim_requires_hash_bound_local_resource_attestation(tmp_path, monkeypatch):
    monkeypatch.setattr(resource_attestation, "gpu_uuid", lambda index: "GPU-accepted")
    runtime = tmp_path / "runtime.json"
    identity = {"run_id":"A-17","group":"A","seed":17,"device":"cuda:0","checkpoint_root":"/x","progress_path":"/y","mode":"formal"}
    source_files = {"/tmp/source.py": "a" * 64}
    source = hashlib.sha256(json.dumps(source_files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    runtime.write_text(json.dumps({"run": identity, "hashes": {"code_sha256": source}}))
    admission = tmp_path / "admission.json"; admission.write_text(json.dumps({"status":"PASS","formal_execution_allowed":True,"run_identity":identity,"source_files":source_files}))
    qualification = tmp_path / "qualification.json"; qualification.write_text(json.dumps({"status":"PASS_GPU_KERNEL_AND_INHERITED_RUNTIME_IMPORTS"}))
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    queue = JobQueue(tmp_path / "queue.sqlite"); queue.add_evidence("qualified_runtime", qualification, accepted=True)
    payload = {"command":["true"],"physical_gpu":0,"expected_outputs":[{"path":"/dev/null"}],"run_identity":identity,"runtime_config":str(runtime),"runtime_config_sha256":digest(runtime),"formal_admission":str(admission),"formal_admission_sha256":digest(admission),"frozen_source_sha256":source,"data_volume":str(tmp_path),"min_free_bytes":1,"run_budget_seconds":60,"deadline_utc_epoch":time.time()+60}
    attestation = tmp_path / "attestation.json"
    attestation.write_text(json.dumps({"schema":"nc_rted_formal_resource_attestation/v1","status":"PASS","binding":{"job_key":"formal", **{key:payload[key] for key in ("runtime_config","runtime_config_sha256","formal_admission","formal_admission_sha256","run_identity","frozen_source_sha256")}},"execution":{"host":__import__("socket").gethostname(),"physical_gpu":0,"gpu_uuid":"GPU-accepted","lease_id":"local-lease","lease_expires_utc_epoch":time.time()+60},"qualification":{"accepted_evidence_name":"qualified_runtime","path":str(qualification),"sha256":digest(qualification),"status":"PASS_GPU_KERNEL_AND_INHERITED_RUNTIME_IMPORTS"},"contract":{key:payload[key] for key in ("data_volume","min_free_bytes","run_budget_seconds","deadline_utc_epoch")}}))
    queue.add_job("formal", "formal_train", {**payload,"resource_attestation":str(attestation),"resource_attestation_sha256":digest(attestation)})
    # This intentionally incomplete synthetic fixture must never stand in for
    # production preflight/admission/source evidence.
    assert queue.claim("worker") is None

    # A stale lease, foreign GPU, or an arbitrary PASS qualification cannot pass.
    for change in (("lease_expires_utc_epoch", 0), ("gpu_uuid", "GPU-other"), ("qualification_status", "PASS")):
        document=json.loads(attestation.read_text())
        if change[0] == "qualification_status": document["qualification"]["status"] = change[1]
        else: document["execution"][change[0]] = change[1]
        attestation.write_text(json.dumps(document)); queue2=JobQueue(tmp_path / ("q-"+change[0]+".sqlite")); queue2.add_evidence("qualified_runtime", qualification, accepted=True)
        queue2.add_job("formal", "formal_train", {**payload,"resource_attestation":str(attestation),"resource_attestation_sha256":digest(attestation)})
        assert queue2.claim("worker") is None


def test_attestation_publication_is_atomic_idempotent_and_preserves_conflict(tmp_path, monkeypatch):
    monkeypatch.setattr(resource_attestation, "MIN_FREE_BYTES", 0)
    target = tmp_path / "attest" / "resource.json"
    value = {"schema": "test", "value": 1}
    first = resource_attestation.publish_attestation(value, target)
    assert target.is_file() and resource_attestation.publish_attestation(value, target) == first
    with pytest.raises(resource_attestation.ResourceAttestationError, match="conflicting"):
        resource_attestation.publish_attestation({"schema": "test", "value": 2}, target)


def test_formal_claim_and_post_lock_guard_accept_real_production_fixture(tmp_path, monkeypatch):
    payload, qualification, runtime, admission, _attestation, _digest = formal_fixture(tmp_path, monkeypatch)
    queue = JobQueue(tmp_path / "queue.sqlite")
    queue.add_evidence("runtime", runtime, accepted=True)
    queue.add_evidence("admission", admission, accepted=True)
    queue.add_evidence("qualification", qualification, accepted=True)
    queue.add_evidence("authorization", payload["_authorization_path"], accepted=True)
    queue.add_job("formal", "formal_train", payload)
    job = queue.claim("worker")
    assert job is not None, queue.status()
    with gpu_lock(payload["data_volume"], 0) as lock:
        queue.bind_reservation("formal", job["lease_token"], lock)
        queue.launch_guard("formal", job["lease_token"], lock)


def test_formal_launch_requires_the_held_attempt_reservation(tmp_path, monkeypatch):
    payload, qualification, runtime, admission, _attestation, _digest = formal_fixture(tmp_path, monkeypatch)
    queue=JobQueue(tmp_path/"queue.sqlite")
    for name,path in (("runtime",runtime),("admission",admission),("qualification",qualification),("authorization",payload["_authorization_path"])): queue.add_evidence(name,path,accepted=True)
    queue.add_job("formal","formal_train",payload); job=queue.claim("worker"); assert job is not None, queue.status()
    with pytest.raises(Exception, match="reservation"):
        queue.launch_guard("formal",job["lease_token"],None)
    with pytest.raises(Exception, match="held device lock"):
        queue.bind_reservation("formal",job["lease_token"],None)


def test_formal_launch_rechecks_the_bound_open_lock_description(tmp_path, monkeypatch):
    payload, qualification, runtime, admission, _attestation, _digest = formal_fixture(tmp_path, monkeypatch)
    queue=JobQueue(tmp_path/"queue.sqlite")
    for name,path in (("runtime",runtime),("admission",admission),("qualification",qualification),("authorization",payload["_authorization_path"])): queue.add_evidence(name,path,accepted=True)
    queue.add_job("formal","formal_train",payload); job=queue.claim("worker"); assert job is not None
    with gpu_lock(payload["data_volume"],0) as lock:
        queue.bind_reservation("formal",job["lease_token"],lock)
        __import__("fcntl").flock(lock.fileno(),__import__("fcntl").LOCK_UN)
        with pytest.raises(Exception,match="lost the bound device lock"):
            queue.launch_guard("formal",job["lease_token"],lock)


def test_formal_launch_requires_the_same_held_lock_descriptor(tmp_path, monkeypatch):
    payload, qualification, runtime, admission, _attestation, _digest = formal_fixture(tmp_path, monkeypatch)
    queue=JobQueue(tmp_path/"queue.sqlite")
    for name,path in (("runtime",runtime),("admission",admission),("qualification",qualification),("authorization",payload["_authorization_path"])): queue.add_evidence(name,path,accepted=True)
    queue.add_job("formal","formal_train",payload); job=queue.claim("worker"); assert job is not None
    with gpu_lock(payload["data_volume"],0) as lock:
        queue.bind_reservation("formal",job["lease_token"],lock)
        fcntl=__import__("fcntl"); fcntl.flock(lock.fileno(),fcntl.LOCK_UN)
        with lock.path.open("a+") as other:
            fcntl.flock(other.fileno(),fcntl.LOCK_EX | fcntl.LOCK_NB)
            with pytest.raises(Exception,match="lost the bound device lock"):
                queue.launch_guard("formal",job["lease_token"],lock)


def test_formal_launch_rejects_replaced_canonical_lock_path(tmp_path, monkeypatch):
    payload, qualification, runtime, admission, _attestation, _digest = formal_fixture(tmp_path, monkeypatch)
    queue=JobQueue(tmp_path/"queue.sqlite")
    for name,path in (("runtime",runtime),("admission",admission),("qualification",qualification),("authorization",payload["_authorization_path"])): queue.add_evidence(name,path,accepted=True)
    queue.add_job("formal","formal_train",payload); job=queue.claim("worker"); assert job is not None
    with gpu_lock(payload["data_volume"],0) as lock:
        queue.bind_reservation("formal",job["lease_token"],lock)
        canonical=lock.path; canonical.unlink(); canonical.touch()
        with pytest.raises(Exception,match="lost the bound device lock"):
            queue.launch_guard("formal",job["lease_token"],lock)


def test_recovered_formal_child_proves_inherited_lock_fd(tmp_path, monkeypatch):
    payload, qualification, runtime, admission, _attestation, _digest = formal_fixture(tmp_path, monkeypatch)
    queue=JobQueue(tmp_path/"queue.sqlite")
    for name,path in (("runtime",runtime),("admission",admission),("qualification",qualification),("authorization",payload["_authorization_path"])): queue.add_evidence(name,path,accepted=True)
    queue.add_job("formal","formal_train",payload); job=queue.claim("worker"); assert job is not None
    with gpu_lock(payload["data_volume"],0) as lock:
        queue.bind_reservation("formal",job["lease_token"],lock)
        child=subprocess.Popen([payload["interpreter"]["path"],"-c","import time; time.sleep(0.5)"],pass_fds=(lock.fileno(),))
        lock.close()
        queue.recovered_lock_guard("formal",job["lease_token"],child.pid)
        child.wait()
    with pytest.raises(Exception,match="recovered formal child"):
        queue.recovered_lock_guard("formal",job["lease_token"],child.pid)


def test_formal_reservation_requires_canonical_held_gpu_flock(tmp_path, monkeypatch):
    payload, qualification, runtime, admission, _attestation, _digest = formal_fixture(tmp_path, monkeypatch)
    queue=JobQueue(tmp_path/"queue.sqlite")
    for name,path in (("runtime",runtime),("admission",admission),("qualification",qualification),("authorization",payload["_authorization_path"])): queue.add_evidence(name,path,accepted=True)
    queue.add_job("formal","formal_train",payload); job=queue.claim("worker"); assert job is not None, queue.status()
    unrelated=tmp_path/"unrelated.lock"; unrelated.touch()
    with unrelated.open("a+") as handle:
        with pytest.raises(Exception,match="canonical device lock"):
            queue.bind_reservation("formal",job["lease_token"],handle)
    with gpu_lock(payload["data_volume"], 1) as wrong_device:
        with pytest.raises(Exception,match="canonical device lock"):
            queue.bind_reservation("formal",job["lease_token"],wrong_device)
    with gpu_lock(payload["data_volume"], 0):
        pass
    canonical=Path(payload["data_volume"])/".nc_rted_locks"/f"gpu_{__import__('socket').gethostname()}_0.lock"
    with gpu_lock(payload["data_volume"], 0) as held:
        with canonical.open("a+") as duplicate:
            with pytest.raises(Exception,match="canonical device lock"):
                queue.bind_reservation("formal",job["lease_token"],duplicate)
    with canonical.open("a+") as released:
        with pytest.raises(Exception,match="canonical device lock"):
            queue.bind_reservation("formal",job["lease_token"],released)


def test_formal_completion_uses_fixed_reservation_not_fresh_preflight(tmp_path, monkeypatch):
    payload, qualification, runtime, admission, _attestation, _digest = formal_fixture(tmp_path, monkeypatch)
    queue=JobQueue(tmp_path/"queue.sqlite")
    for name,path in (("runtime",runtime),("admission",admission),("qualification",qualification),("authorization",payload["_authorization_path"])): queue.add_evidence(name,path,accepted=True)
    queue.add_job("formal","formal_train",payload); job=queue.claim("worker"); assert job is not None, queue.status()
    with gpu_lock(payload["data_volume"], 0) as lock:
        queue.bind_reservation("formal",job["lease_token"],lock)
        completed=time.time()
        with queue.connect() as db: db.execute("UPDATE reservations SET authorized_end=? WHERE lease_token=?",(completed+.01,job["lease_token"]))
        with pytest.raises(Exception) as error: queue.completion_guard("formal",job["lease_token"])
        assert getattr(error.value,"code",None) == "resource_integrity"
        with queue.connect() as db: db.execute("INSERT INTO terminal_evidence VALUES(?,?,?,?,?)",(job["lease_token"],"formal",1,"start",completed))
        queue.completion_guard("formal",job["lease_token"])
        with queue.connect() as db: db.execute("UPDATE terminal_evidence SET observed_at=? WHERE lease_token=?",(completed+.02,job["lease_token"]))
        with pytest.raises(Exception) as error: queue.completion_guard("formal",job["lease_token"])
        assert getattr(error.value,"code",None) == "rental_lease"
        started=queue.attempt_started_at("formal",job["lease_token"])
        with queue.connect() as db: db.execute("UPDATE terminal_evidence SET observed_at=? WHERE lease_token=?",(started+payload["run_budget_seconds"]+.01,job["lease_token"]))
        with pytest.raises(Exception) as error: queue.completion_guard("formal",job["lease_token"])
        assert getattr(error.value,"code",None) == "budget"


def test_terminal_evidence_requires_parent_observed_dead_group(tmp_path):
    queue=JobQueue(tmp_path/"queue.sqlite"); add_ready(queue); job=queue.claim("worker")
    child=subprocess.Popen(["/bin/sleep","0.05"]); start=queue_script.process_starttime(child.pid); child.wait()
    with queue.connect() as db:
        db.execute("INSERT INTO reservations VALUES(?,?,?,?,?,?,?,?,?)",(job["lease_token"],"job","lease",__import__("socket").gethostname(),0,1,1,time.time(),time.time()+60))
    queue.record_terminal_evidence("job",job["lease_token"],child.pid,start)
    with queue.connect() as db:
        evidence=db.execute("SELECT * FROM terminal_evidence WHERE lease_token=?",(job["lease_token"],)).fetchone()
    assert evidence["pid"] == child.pid and evidence["observed_at"] <= time.time()


def test_formal_completion_resource_read_failure_is_protective(tmp_path, monkeypatch):
    payload, qualification, runtime, admission, _attestation, _digest = formal_fixture(tmp_path, monkeypatch)
    queue=JobQueue(tmp_path/"queue.sqlite")
    for name,path in (("runtime",runtime),("admission",admission),("qualification",qualification),("authorization",payload["_authorization_path"])): queue.add_evidence(name,path,accepted=True)
    queue.add_job("formal","formal_train",payload); job=queue.claim("worker"); assert job is not None, queue.status()
    with gpu_lock(payload["data_volume"], 0) as lock:
        queue.bind_reservation("formal",job["lease_token"],lock)
        monkeypatch.setattr(resource_attestation, "_bound_file", lambda *_args: (_ for _ in ()).throw(OSError("I/O")))
        with pytest.raises(Exception) as error: queue.completion_guard("formal",job["lease_token"])
        assert getattr(error.value,"code",None) == "resource_integrity"


def test_formal_output_completion_requires_supervisor_terminal_evidence(tmp_path, monkeypatch):
    payload, _qualification, _runtime, _admission, _attestation, _digest = formal_fixture(tmp_path, monkeypatch)
    job={"job_key":"formal","lease_token":"token","input_hash":"hash","attempts":1,"kind":"formal_train"}
    run=queue_script.attempt_directory(payload,job); run.mkdir(parents=True,exist_ok=True)
    manifest=Path(payload["expected_outputs"][0]["path"]); manifest.parent.mkdir(parents=True,exist_ok=True); manifest.write_text("{}")
    completion=run/"producer_completion.json"
    completion.write_text(json.dumps({"job_key":"formal","lease_token":"token","input_hash":"hash","artifacts":[],"completed_at":time.time()-10}))
    monkeypatch.setattr(queue_script,"output_records",lambda *_args: [])
    with pytest.raises(Exception,match="completion_guard"):
        queue_script.commit_outputs(object(),job,payload)


def test_runtime_guard_uses_fixed_attempt_end_not_remaining_original_budget(tmp_path, monkeypatch):
    payload, qualification, runtime, admission, _attestation, _digest = formal_fixture(tmp_path, monkeypatch)
    payload["run_budget_seconds"]=60
    queue=JobQueue(tmp_path/"queue.sqlite")
    for name,path in (("runtime",runtime),("admission",admission),("qualification",qualification),("authorization",payload["_authorization_path"])): queue.add_evidence(name,path,accepted=True)
    queue.add_job("formal","formal_train",payload); job=queue.claim("worker")
    with gpu_lock(payload["data_volume"],0) as lock:
        queue.bind_reservation("formal",job["lease_token"],lock)
        # The original budget no longer has to fit again at every polling tick.
        with queue.connect() as db: db.execute("UPDATE reservations SET authorized_end=? WHERE lease_token=?",(time.time()+5,job["lease_token"]))
        queue.runtime_guard("formal",job["lease_token"])


def test_runtime_guard_checks_authorized_end_before_attestation_read(tmp_path, monkeypatch):
    payload, qualification, runtime, admission, _attestation, _digest = formal_fixture(tmp_path, monkeypatch)
    queue=JobQueue(tmp_path/"queue.sqlite")
    for name,path in (("runtime",runtime),("admission",admission),("qualification",qualification),("authorization",payload["_authorization_path"])): queue.add_evidence(name,path,accepted=True)
    queue.add_job("formal","formal_train",payload); job=queue.claim("worker"); assert job is not None
    with gpu_lock(payload["data_volume"],0) as lock:
        queue.bind_reservation("formal",job["lease_token"],lock)
        with queue.connect() as db: db.execute("UPDATE reservations SET authorized_end=? WHERE lease_token=?",(time.time()-1,job["lease_token"]))
        monkeypatch.setattr(queue_module,"resource_lease_expiry",lambda _payload: pytest.fail("expired fixed end must stop before resource read"))
        with pytest.raises(Exception) as error: queue.runtime_guard("formal",job["lease_token"])
        assert getattr(error.value,"code",None) == "rental_lease"

def test_formal_launch_captures_validated_source_and_inputs(tmp_path, monkeypatch):
    payload, _qualification, _runtime, _admission, _attestation, _digest = formal_fixture(tmp_path, monkeypatch)
    command, environment, cuda_device=queue_script.capture_formal_inputs(payload,tmp_path/"attempt")
    captured=tmp_path/"attempt"/"immutable-inputs"
    assert command[1] == str(captured/"scripts"/"nc_rted_train.py")
    assert command[command.index("--config")+1] == str(captured/"runtime.json")
    assert environment["PYTHONPATH"] == str(captured/"src")
    assert cuda_device == "GPU-valid"
    assert all(not (path.stat().st_mode & 0o222) for path in [captured,*captured.rglob("*")])
    Path(payload["runtime_config"]).write_text("changed")
    assert hashlib.sha256((captured/"runtime.json").read_bytes()).hexdigest() == payload["runtime_config_sha256"]
    imported=subprocess.run([payload["interpreter"]["path"],"-c","import nc_rted.resource_attestation as m; print(m.__file__)"],text=True,capture_output=True,check=True,env=environment)
    assert str(captured/"src") in imported.stdout


def test_formal_worker_passes_only_captured_launch_environment_to_popen(tmp_path, monkeypatch):
    payload, qualification, runtime, admission, _attestation, _digest = formal_fixture(tmp_path, monkeypatch)
    queue=JobQueue(tmp_path/"queue.sqlite")
    for name,path in (("runtime",runtime),("admission",admission),("qualification",qualification),("authorization",payload["_authorization_path"])): queue.add_evidence(name,path,accepted=True)
    queue.add_job("formal","formal_train",payload)
    observed={}
    def stop_at_popen(command, **kwargs):
        root=Path(kwargs["cwd"])/"immutable-inputs"
        contract=json.loads((root/"launch-contract.json").read_text())
        expected=dict(contract["environment"])
        expected["PYTHONPATH"]=str(root/"src")
        expected["CUDA_VISIBLE_DEVICES"]=contract["cuda_visible_devices"]
        expected.update({
            "NC_RTED_PRODUCER_COMPLETION":str(Path(kwargs["cwd"])/"producer_completion.json"),
            "NC_RTED_JOB_KEY":"formal",
            "NC_RTED_LEASE_TOKEN":kwargs["env"]["NC_RTED_LEASE_TOKEN"],
            "NC_RTED_INPUT_HASH":kwargs["env"]["NC_RTED_INPUT_HASH"],
        })
        assert command == contract["command"]
        assert kwargs["env"] == expected
        observed["called"]=True
        raise RuntimeError("stop after controlled Popen inspection")
    monkeypatch.setattr(queue_script.subprocess,"Popen",stop_at_popen)
    assert queue_script.worker(queue,"worker",once=True) == 0
    assert observed == {"called":True}


def test_formal_checkpoint_destination_rejects_symlink_before_worker_launch(tmp_path, monkeypatch):
    payload, qualification, runtime, admission, _attestation, _digest = formal_fixture(tmp_path, monkeypatch)
    checkpoint=Path(payload["checkpoint_root"]); checkpoint.mkdir(parents=True,exist_ok=True)
    (checkpoint/"final").unlink(missing_ok=True)
    (checkpoint/"final").symlink_to(tmp_path,target_is_directory=True)
    queue=JobQueue(tmp_path/"queue.sqlite")
    for name,path in (("runtime",runtime),("admission",admission),("qualification",qualification),("authorization",payload["_authorization_path"])): queue.add_evidence(name,path,accepted=True)
    queue.add_job("formal","formal_train",payload)
    monkeypatch.setattr(queue_script.subprocess,"Popen",lambda *_args,**_kwargs: pytest.fail("Popen must not receive a symlinked checkpoint destination"))
    assert queue_script.worker(queue,"worker",once=True) == 0
    assert queue.status()[0]["status"] == BLOCKED


def test_formal_checkpoint_setup_keeps_final_absent_for_store_publication(tmp_path, monkeypatch):
    payload, _qualification, _runtime, _admission, _attestation, _digest = formal_fixture(tmp_path, monkeypatch)
    root=queue_script.ensure_checkpoint_directory(payload)
    assert root == Path(payload["checkpoint_root"]) and not (root/"final").exists()
    assert CheckpointStore(root,payload["run_identity"],min_free_bytes=0).latest() is None


def test_formal_destination_rejects_component_on_other_filesystem(tmp_path, monkeypatch):
    payload, _qualification, _runtime, _admission, _attestation, _digest = formal_fixture(tmp_path, monkeypatch)
    mounted=Path(payload["data_volume"])/".cache"/"nc_rted_queue_tests"/tmp_path.name/"mounted"
    mounted.mkdir(parents=True,exist_ok=True)
    original_stat=Path.stat
    def foreign_device(path, *args, **kwargs):
        result=original_stat(path,*args,**kwargs)
        if path == mounted:
            values=list(result); values[2]=result.st_dev+1
            return os.stat_result(values)
        return result
    monkeypatch.setattr(queue_script.Path,"stat",foreign_device)
    with pytest.raises(Exception,match="unapproved filesystem"):
        queue_script.ensure_formal_directory(payload,mounted/"child","formal destination")


def test_formal_reconciliation_rejects_early_or_backdated_producer_receipt_without_terminal_evidence(tmp_path, monkeypatch):
    payload, qualification, runtime, admission, _attestation, _digest = formal_fixture(tmp_path, monkeypatch)
    queue=JobQueue(tmp_path/"queue.sqlite")
    for name,path in (("runtime",runtime),("admission",admission),("qualification",qualification),("authorization",payload["_authorization_path"])): queue.add_evidence(name,path,accepted=True)
    queue.add_job("formal","formal_train",payload); job=queue.claim(f"{__import__('socket').gethostname()}:dead"); assert job is not None
    with gpu_lock(payload["data_volume"],0) as lock:
        queue.bind_reservation("formal",job["lease_token"],lock)
    run=queue_script.ensure_attempt_directory(payload,job)
    final=Path(payload["expected_outputs"][0]["path"]); final.parent.mkdir(parents=True,exist_ok=True)
    final.write_text("{}")
    records=[{"path":str(final),"checksum":hashlib.sha256(final.read_bytes()).hexdigest()}]
    monkeypatch.setattr(queue_script,"output_records",lambda *_args: records)
    (run/"producer_completion.json").write_text(json.dumps({"job_key":"formal","lease_token":job["lease_token"],"input_hash":job["input_hash"],"artifacts":records,"completed_at":1}))
    child=subprocess.Popen([sys.executable,"-c","import time; time.sleep(.15)"],start_new_session=True)
    start=queue_script.process_starttime(child.pid)
    queue.heartbeat("formal",job["lease_token"],pid=child.pid,process_starttime=start,owner=job["lease_owner"])
    queue.start_attempt_journal("formal",job["lease_token"],child.pid,start,"0",run/"attempt.json")
    queue_script.reconcile_exited_attempts(queue,f"{__import__('socket').gethostname()}:replacement")
    assert queue.status()[0]["status"] == RUNNING
    child.wait(timeout=5)
    queue_script.reconcile_exited_attempts(queue,f"{__import__('socket').gethostname()}:replacement")
    row=queue.status()[0]
    assert row["status"] == BLOCKED and "resource_integrity" in row["failure"]


def test_formal_running_worker_turns_resource_eio_into_protective_stop(tmp_path, monkeypatch):
    payload, qualification, runtime, admission, _attestation, _digest = formal_fixture(tmp_path, monkeypatch)
    payload["poll_seconds"]=.01
    queue=JobQueue(tmp_path/"queue.sqlite")
    for name,path in (("runtime",runtime),("admission",admission),("qualification",qualification),("authorization",payload["_authorization_path"])): queue.add_evidence(name,path,accepted=True)
    queue.add_job("formal","formal_train",payload)
    state={"launched":False}
    original_popen=queue_script.subprocess.Popen
    def launch(*args, **kwargs):
        process=original_popen(*args, **kwargs)
        state["launched"]=True
        return process
    original_read=Path.read_bytes
    attestation_path=Path(payload["resource_attestation"])
    def eio_after_launch(path, *args, **kwargs):
        if state["launched"] and path == attestation_path: raise OSError("injected resource EIO")
        return original_read(path, *args, **kwargs)
    monkeypatch.setattr(queue_script,"capture_formal_inputs",lambda _payload,_run: ([sys.executable,"-c","import time; time.sleep(5)"],dict(payload["execution_environment"]),"GPU-valid"))
    monkeypatch.setattr(queue_script.subprocess,"Popen",launch)
    monkeypatch.setattr(resource_attestation.Path,"read_bytes",eio_after_launch)
    assert queue_script.worker(queue,"worker",once=True) == 1
    row=queue.status()[0]
    with queue.connect() as db:
        code=db.execute("SELECT protective_stop_code FROM jobs WHERE job_key='formal'").fetchone()["protective_stop_code"]
    assert row["status"] == BLOCKED and code == "resource_integrity"


def test_captured_formal_source_map_drives_pre_model_and_worker_admission(tmp_path, monkeypatch):
    payload, _qualification, _runtime, _admission, _attestation, _digest = formal_fixture(tmp_path, monkeypatch)
    command, environment, _cuda_device=queue_script.capture_formal_inputs(payload,tmp_path/"attempt")
    captured=tmp_path/"attempt"/"immutable-inputs"
    original=Path(__file__).resolve().parents[1]/"src"/"nc_rted"/"production_runtime.py"
    original_bytes=original.read_bytes()
    program='''import json, sys, types
from nc_rted.captured_sources import load_captured_sources
from nc_rted.production_runtime import _validate_formal_admission_before_models
from nc_rted.train_worker import TrainingWorker
from nc_rted.training import Recipe
admission=json.loads(open(sys.argv[1]).read()); identity=json.loads(sys.argv[2]); sources=load_captured_sources(sys.argv[3], admission)
_validate_formal_admission_before_models(admission, identity, captured_sources=sources)
worker=object.__new__(TrainingWorker); worker.store=types.SimpleNamespace(identity=identity); worker.trainer=types.SimpleNamespace(recipe=Recipe()); worker.seed=int(identity["seed"]); worker.catalog=types.SimpleNamespace(tasks=dict.fromkeys(range(8000))); worker.captured_sources=sources
worker._verify_formal_admission(admission)
print("CAPTURED_ADMISSION_PASS")
'''
    try:
        original.write_bytes(original_bytes+b"\n# original changed after capture\n")
        checked=subprocess.run([payload["interpreter"]["path"],"-c",program,str(captured/"admission.json"),json.dumps(payload["run_identity"]),str(captured/"source-map.json")],text=True,capture_output=True,env=environment,check=True)
        assert "CAPTURED_ADMISSION_PASS" in checked.stdout
    finally:
        original.write_bytes(original_bytes)
    captured_runtime=captured/"src"/"nc_rted"/"production_runtime.py"
    captured_bytes=captured_runtime.read_bytes(); captured_runtime.write_bytes(captured_bytes+b"\n# captured tamper\n")
    rejected=subprocess.run([payload["interpreter"]["path"],"-c",program,str(captured/"admission.json"),json.dumps(payload["run_identity"]),str(captured/"source-map.json")],text=True,capture_output=True,env=environment)
    assert rejected.returncode != 0 and "captured source differs" in rejected.stderr


def test_captured_runtime_sources_drive_preflight_after_original_mutation(tmp_path, monkeypatch):
    payload, _qualification, _runtime, _admission, _attestation, _digest = formal_fixture(tmp_path, monkeypatch)
    command, environment, _cuda_device=queue_script.capture_formal_inputs(payload,tmp_path/"attempt")
    captured=tmp_path/"attempt"/"immutable-inputs"
    runtime_document=json.loads(Path(payload["runtime_config"]).read_text())
    inherited=Path(runtime_document["inherited"]["external_root"])/"llava"/"train"/"train.py"
    stage=Path(runtime_document["stage2_cache"]["module"])
    inherited_bytes,stage_bytes=inherited.read_bytes(),stage.read_bytes()
    program='''import json, sys
from nc_rted.captured_sources import load_captured_runtime
from nc_rted.production_runtime import load_manifest
raw=json.loads(open(sys.argv[1]).read()); mapping=load_captured_runtime(sys.argv[2],raw)
load_manifest(sys.argv[1], expected_sha256=sys.argv[3], captured_runtime=mapping)
print("CAPTURED_PREFLIGHT_PASS")
'''
    try:
        inherited.write_bytes(inherited_bytes+b"\n# original inherited source changed\n")
        stage.write_bytes(stage_bytes+b"\n# original resolver source changed\n")
        cli=subprocess.run([*command,"--dry-run"],text=True,capture_output=True,env=environment,check=True)
        assert "FILE_BINDINGS_PASS_SEMANTIC_NOT_RUN" in cli.stdout
        checked=subprocess.run([payload["interpreter"]["path"],"-c",program,str(captured/"runtime.json"),str(captured/"source-map.json"),payload["runtime_config_sha256"]],text=True,capture_output=True,env=environment,check=True)
        assert "CAPTURED_PREFLIGHT_PASS" in checked.stdout
    finally:
        inherited.write_bytes(inherited_bytes); stage.write_bytes(stage_bytes)
    (captured/"inherited"/"llava"/"train"/"train.py").write_bytes(inherited_bytes+b"\n# captured inherited tamper\n")
    rejected=subprocess.run([payload["interpreter"]["path"],"-c",program,str(captured/"runtime.json"),str(captured/"source-map.json"),payload["runtime_config_sha256"]],text=True,capture_output=True,env=environment)
    assert rejected.returncode != 0 and "captured runtime source map is invalid" in rejected.stderr

def test_formal_capture_rejects_source_replaced_after_admission(tmp_path, monkeypatch):
    payload, _qualification, _runtime, _admission, _attestation, _digest = formal_fixture(tmp_path, monkeypatch)
    source=Path(__file__).resolve().parents[1]/"src"/"nc_rted"/"resource_attestation.py"; original=source.read_bytes()
    try:
        source.write_bytes(original+b"\n# replacement after admission\n")
        with pytest.raises(Exception,match="captured formal source"):
            queue_script.capture_formal_inputs(payload,tmp_path/"attempt")
    finally:
        source.write_bytes(original)


def test_formal_capture_reuses_only_the_verified_existing_closure(tmp_path, monkeypatch):
    payload, _qualification, _runtime, _admission, _attestation, _digest = formal_fixture(tmp_path, monkeypatch)
    attempt=tmp_path/"attempt"
    first_command, first_environment, first_gpu=queue_script.capture_formal_inputs(payload,attempt)
    second_command, second_environment, second_gpu=queue_script.capture_formal_inputs(payload,attempt)
    assert second_command == first_command and second_environment == first_environment and second_gpu == first_gpu
    captured=attempt/"immutable-inputs"/"runtime.json"
    captured.write_text("corrupt")
    with pytest.raises(Exception,match="captured formal configuration"):
        queue_script.capture_formal_inputs(payload,attempt)


def test_formal_capture_block_rounds_reserve_and_syncs_recovery(tmp_path, monkeypatch):
    payload, _qualification, _runtime, _admission, _attestation, _digest = formal_fixture(tmp_path, monkeypatch)
    attempt=tmp_path/"attempt"
    source_files=json.loads(Path(payload["formal_admission"]).read_text())["source_files"]
    old_size=sum(Path(path).stat().st_size for path in source_files)+Path(payload["runtime_config"]).stat().st_size+Path(payload["formal_admission"]).stat().st_size
    usage=__import__("collections").namedtuple("usage","total used free")
    monkeypatch.setattr(queue_script.shutil,"disk_usage",lambda _path: usage(0,0,payload["min_free_bytes"]+old_size+8192))
    with pytest.raises(Exception,match="disk reserve"):
        queue_script.capture_formal_inputs(payload,attempt)
    monkeypatch.undo()
    queue_script.capture_formal_inputs(payload,attempt)
    root=attempt/"immutable-inputs"; root_inode=root.stat().st_ino; synced=[]
    original=queue_script.os.fsync
    def remember(fd):
        synced.append(os.fstat(fd).st_ino); return original(fd)
    monkeypatch.setattr(queue_script.os,"fsync",remember)
    queue_script.capture_formal_inputs(payload,attempt)
    assert root_inode in synced


def test_formal_attempt_directory_rejects_job_symlink_escape(tmp_path, monkeypatch):
    payload, _qualification, _runtime, _admission, _attestation, _digest = formal_fixture(tmp_path, monkeypatch)
    payload["run_dir"]=str(Path(payload["data_volume"])/".cache"/"nc_rted_queue_tests"/tmp_path.name/f"symlink-{time.time_ns()}")
    run=Path(payload["run_dir"]); run.mkdir(parents=True,exist_ok=True)
    (run/"formal").symlink_to(tmp_path, target_is_directory=True)
    job={"job_key":"formal","attempts":1,"lease_token":"token","kind":"formal_train"}
    with pytest.raises(Exception,match="symlink"):
        queue_script.ensure_attempt_directory(payload,job)


def test_attestation_cli_reuses_identical_completed_request(tmp_path, monkeypatch):
    payload, qualification, runtime, admission, _attestation, digest = formal_fixture(tmp_path, monkeypatch, real_probe=True)
    fake_bin=tmp_path/"bin"; fake_bin.mkdir(); smi=fake_bin/"nvidia-smi"
    smi.write_text("#!/bin/sh\ncase \"$*\" in *memory.total*) printf '32768\\n' ;; *) printf 'GPU-valid\\n' ;; esac\n"); smi.chmod(0o755)
    output=tmp_path/"attested.json"
    command=["/root/autodl-tmp/lookaway-wm/.venv-reactvau/bin/python",str(Path(__file__).resolve().parents[1]/"scripts"/"nc_rted_attest_formal_resources.py"),
             "--job-key","formal","--runtime-config",str(runtime),"--runtime-config-sha256",digest(runtime),
             "--formal-admission",str(admission),"--formal-admission-sha256",digest(admission),
             "--qualification-report",str(qualification),"--qualification-report-sha256",digest(qualification),
             "--resource-authorization",str(payload["_authorization_path"]),"--resource-authorization-sha256",digest(Path(payload["_authorization_path"])),
             "--accepted-evidence-name","qualification","--resource-authorization-evidence-name","authorization","--runtime-evidence-name","runtime","--formal-admission-evidence-name","admission","--physical-gpu","0","--data-volume",payload["data_volume"],
             "--min-free-bytes",str(20*1024**3),"--run-budget-seconds","60","--deadline-utc-epoch",str(payload["deadline_utc_epoch"]),
             "--lease-id","lease","--lease-expires-utc-epoch","1792079990","--execution-environment",json.dumps(payload["execution_environment"]),
             "--interpreter",payload["interpreter"]["path"],"--run-dir",payload["run_dir"],"--expected-outputs",json.dumps(payload["expected_outputs"]),"--output",str(output)]
    environment={**payload["execution_environment"],"PATH":str(fake_bin)+":"+os.environ.get("PATH","")}
    first=json.loads(subprocess.run(command,check=True,text=True,capture_output=True,env=environment).stdout)
    second=json.loads(subprocess.run(command,check=True,text=True,capture_output=True,env=environment).stdout)
    assert first["sha256"] == second["sha256"] and second["reused"] is True
    queue=JobQueue(tmp_path/"queue.sqlite")
    for name,path in (("runtime",runtime),("admission",admission),("qualification",qualification),("authorization",payload["_authorization_path"])): queue.add_evidence(name,path,accepted=True)
    queue.add_job("formal","formal_train",first["queue_payload"]); job=queue.claim("worker")
    assert job is not None, queue.status()
    with gpu_lock(payload["data_volume"], 0) as lock:
        queue.bind_reservation("formal",job["lease_token"],lock); queue.launch_guard("formal",job["lease_token"],lock)


def test_corrupt_attestation_becomes_a_protective_integrity_stop(tmp_path, monkeypatch):
    payload, qualification, runtime, admission, attestation, _digest = formal_fixture(tmp_path, monkeypatch)
    queue=JobQueue(tmp_path/"queue.sqlite")
    for name,path in (("runtime",runtime),("admission",admission),("qualification",qualification),("authorization",payload["_authorization_path"])): queue.add_evidence(name,path,accepted=True)
    queue.add_job("formal","formal_train",payload); job=queue.claim("worker")
    with gpu_lock(payload["data_volume"], 0) as lock:
        queue.bind_reservation("formal",job["lease_token"],lock)
        attestation.unlink()
        with pytest.raises(Exception) as error: queue.runtime_guard("formal",job["lease_token"])
        assert getattr(error.value,"code",None) == "resource_integrity"


def test_entrypoint_drift_rejects_an_otherwise_accepted_formal_payload(tmp_path, monkeypatch):
    payload, qualification, runtime, admission, _attestation, _digest = formal_fixture(tmp_path, monkeypatch)
    entry=Path(__file__).resolve().parents[1]/"scripts"/"nc_rted_train.py"; original=entry.read_bytes()
    try:
        entry.write_bytes(original+b"\n# focused drift probe\n")
        queue=JobQueue(tmp_path/"queue.sqlite")
        for name,path in (("runtime",runtime),("admission",admission),("qualification",qualification),("authorization",payload["_authorization_path"])): queue.add_evidence(name,path,accepted=True)
        queue.add_job("formal","formal_train",payload)
        assert queue.claim("worker") is None
    finally:
        entry.write_bytes(original)


def test_substantive_qualification_deadline_and_output_mutations_reject_independently(tmp_path, monkeypatch):
    for mode in ("qualification", "deadline", "output"):
        case=tmp_path/mode; case.mkdir()
        payload, qualification, runtime, admission, attestation, digest = formal_fixture(case, monkeypatch)
        document=json.loads(attestation.read_text())
        if mode == "qualification":
            report=json.loads(qualification.read_text()); report["resource_envelope"]={}; qualification.write_text(json.dumps(report))
            document["qualification"]["sha256"]=digest(qualification)
        elif mode == "deadline":
            payload["deadline_utc_epoch"]=resource_attestation.RENTAL_CUTOFF
            document["contract"]["deadline_utc_epoch"]=resource_attestation.RENTAL_CUTOFF
        else:
            payload["expected_outputs"]=[{**payload["expected_outputs"][0],"path":"smoke.json"}]
            document["binding"]["execution_inputs"]["expected_outputs"]=payload["expected_outputs"]
        attestation.write_text(json.dumps(document)); payload["resource_attestation_sha256"]=digest(attestation)
        queue=JobQueue(case/"queue.sqlite")
        for name,path in (("runtime",runtime),("admission",admission),("qualification",qualification),("authorization",payload["_authorization_path"])): queue.add_evidence(name,path,accepted=True)
        queue.add_job("formal","formal_train",payload)
        assert queue.claim("worker") is None, mode


@pytest.mark.parametrize("field,value", [("gpu_uuid", "GPU-wrong"), ("lease_expires_utc_epoch", 1)])
def test_formal_fixture_mutations_reject_independently(tmp_path, monkeypatch, field, value):
    payload, qualification, runtime, admission, attestation, digest = formal_fixture(tmp_path, monkeypatch)
    document=json.loads(attestation.read_text()); document["execution"][field]=value; attestation.write_text(json.dumps(document)); payload["resource_attestation_sha256"]=digest(attestation)
    queue=JobQueue(tmp_path/"queue.sqlite")
    for name,path in (("runtime",runtime),("admission",admission),("qualification",qualification),("authorization",payload["_authorization_path"])): queue.add_evidence(name,path,accepted=True)
    queue.add_job("formal","formal_train",payload)
    assert queue.claim("worker") is None

def test_semantic_artifact_contracts_require_final_identity_and_prediction_denominator(tmp_path):
    train=tmp_path/'train.json'; train.write_text(json.dumps({'status':'FORMAL_TRAINING_COMPLETE','completed_updates':1000,'run_identity':{'run':'x'}}))
    queue=JobQueue(tmp_path/'queue.sqlite'); checksum=__import__('hashlib').sha256(train.read_bytes()).hexdigest()
    with pytest.raises(Exception): queue.validate_artifacts([{'path':str(train),'artifact_type':'report','checksum':checksum,'semantic':'formal_training','run_identity':{'run':'x'}}],[{'path':str(train),'checksum':checksum}])
    final=tmp_path/'final'; final.mkdir(); state=final/'state.pt'; state.write_bytes(b'checkpoint')
    manifest=final/'manifest.json'; manifest.write_text(json.dumps({'schema':'nc_rted_checkpoint_v2','identity':{'run':'x'},'final':True,'completed_updates':1000,'cursor':1,'payload_bytes':state.stat().st_size,'payload_sha256':__import__('hashlib').sha256(state.read_bytes()).hexdigest()}))
    checksum=__import__('hashlib').sha256(manifest.read_bytes()).hexdigest()
    with pytest.raises(Exception): queue.validate_artifacts([{'path':str(manifest),'artifact_type':'checkpoint','checksum':checksum,'semantic':'formal_training','run_identity':{'run':'x'}}],[{'path':str(manifest),'checksum':checksum}])
    model='a'*64; input_hash='b'*64
    prediction=tmp_path/'prediction.json'; prediction.write_text(json.dumps({'prediction_ids':['a','b'],'model_hash':model,'input_hash':input_hash})); checksum=__import__('hashlib').sha256(prediction.read_bytes()).hexdigest()
    contract={'path':str(prediction),'artifact_type':'prediction','checksum':checksum,'semantic':'prediction','denominator':2,'official_ids':['a','b'],'model_hash':model,'input_hash':input_hash}
    queue.validate_artifacts([contract],[{'path':str(prediction),'checksum':checksum}])
    with pytest.raises(Exception): queue.validate_artifacts([{**contract,'model_hash':''}],[{'path':str(prediction),'checksum':checksum}])
    with pytest.raises(Exception): queue.validate_artifacts([{'path':str(prediction),'artifact_type':'prediction','semantic':'prediction','denominator':3}],[{'path':str(prediction),'checksum':checksum}])
