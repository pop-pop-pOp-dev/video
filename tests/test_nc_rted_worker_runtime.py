import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nc_rted.worker_runtime import group_members, gpu_lock, process_live, process_starttime, write_journal


def test_pid_starttime_rejects_stale_identity_and_accepts_live_child():
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(2)"])
    try:
        start = process_starttime(child.pid)
        assert process_live(child.pid, start)
        assert not process_live(child.pid, "not-the-starttime")
    finally:
        child.terminate(); child.wait(timeout=5)
    assert not process_live(child.pid, start)


def test_journal_is_atomic_and_gpu_lock_uses_declared_volume(tmp_path):
    journal = tmp_path / "runs" / "attempt.json"
    write_journal(journal, {"pid": 1, "state": "RUNNING"})
    assert journal.is_file() and not journal.with_suffix(".json.tmp").exists()
    with gpu_lock(tmp_path, 0) as first:
        assert first is not False
    assert (tmp_path / ".nc_rted_locks").is_dir()


def test_owned_stale_journal_temp_is_recovered_before_next_write(tmp_path):
    journal = tmp_path / "attempt.json"
    stale = journal.with_suffix(".json.tmp")
    stale.write_text('{"job_key":"job","lease_token":"lease"}\n')
    write_journal(journal, {"job_key": "job", "lease_token": "lease", "state": "RUNNING"})
    assert not stale.exists()
    assert '"state": "RUNNING"' in journal.read_text()


def test_group_member_snapshot_has_leader_identity_for_continuity():
    child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(2)'],start_new_session=True)
    try:
        members=group_members(child.pid)
        assert members is not None and members[child.pid] == process_starttime(child.pid)
    finally:
        child.terminate(); child.wait(timeout=5)
