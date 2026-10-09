"""Linux process-record-lock behavior for cooperating storage allocations."""
from __future__ import annotations

import hashlib
import multiprocessing
import os
from pathlib import Path
import queue
import select
import signal
import sys
import threading
import time

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _copy_process(destination, first, ready, release, attempted, results):
    from nc_rted.storage_lock import allocation_lock
    root = Path(destination).parent.parent
    attempted.set()
    with allocation_lock(destination):
        if first:
            ready.set()
            if not release.wait(20):
                raise RuntimeError("test failed to release first writer")
        # This marker models one payload of available headroom. The second
        # writer may only inspect it after the first commits under admission.
        if (root / "one-payload-spent").exists():
            results.put("reserve_rejected")
        else:
            Path(destination).write_bytes(b"x" * (1 << 20))
            (root / "one-payload-spent").write_text("spent")
            results.put("copied")


def test_competing_preparations_cannot_both_spend_one_payload_headroom(tmp_path):
    for name in ("output-a", "output-b"):
        (tmp_path / name).mkdir()
    context = multiprocessing.get_context("fork")
    ready, release = context.Event(), context.Event()
    attempts = [context.Event(), context.Event()]
    results = context.Queue()
    children = [context.Process(target=_copy_process, args=(str(tmp_path / name / "payload"), first,
                ready, release, attempted, results)) for name, first, attempted in
                zip(("output-a", "output-b"), (True, False), attempts)]
    try:
        children[0].start()
        assert ready.wait(20), "first writer did not enter admission"
        children[1].start()
        assert attempts[1].wait(20), "second writer did not attempt allocation"
        with pytest.raises(queue.Empty):
            results.get(timeout=.3)
        release.set()
        assert sorted(results.get(timeout=20) for _ in children) == ["copied", "reserve_rejected"]
        for child in children:
            child.join(20)
            assert child.exitcode == 0
        assert sum(item.stat().st_size for item in tmp_path.glob("output-*/payload")) == 1 << 20
    finally:
        release.set()
        for child in children:
            if child.is_alive():
                child.terminate()
            child.join(5)


def test_filesystem_lock_is_reentrant_across_component_directories(tmp_path):
    from nc_rted.storage_lock import allocation_lock, filesystem_root

    assert filesystem_root(tmp_path / "cache") == filesystem_root(tmp_path / "journal")
    with allocation_lock(tmp_path / "cache"):
        with allocation_lock(tmp_path / "journal"):
            pass


def test_forked_context_does_not_unlock_parent_and_fresh_child_waits(tmp_path):
    from nc_rted.storage_lock import allocation_lock

    parent_read, child_write = os.pipe()
    child = None
    try:
        with allocation_lock(tmp_path):
            child = os.fork()
            if child == 0:
                try:
                    os.close(parent_read)
                    # Leave the inherited context first. It must not close or
                    # unlock any inherited coordination descriptor.
                    pass
                finally:
                    # The context manager's finally runs after this branch.
                    pass
            else:
                os.close(child_write)
                child_write = None
                assert not select.select([parent_read], [], [], .3)[0]
        if child == 0:
            with allocation_lock(tmp_path):
                os.write(child_write, b"A")
            os._exit(0)
        assert select.select([parent_read], [], [], 10)[0]
        assert os.read(parent_read, 1) == b"A"
        status = _wait_child(child)
        child = None
        assert os.waitstatus_to_exitcode(status) == 0
    finally:
        if child == 0:
            os._exit(4)
        if child is not None:
            try:
                os.kill(child, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                os.waitpid(child, 0)
            except ChildProcessError:
                pass
        os.close(parent_read)
        if child_write is not None:
            os.close(child_write)


def test_fork_during_other_thread_fd_initialization_has_fresh_state(tmp_path, monkeypatch):
    import nc_rted.storage_lock as module

    # Previous tests legitimately cached a process-lifetime descriptor. Drop
    # only the test registry reference (never close its descriptor) so this
    # test can control a new state initialization transaction.
    module._PROCESS_STATES.clear()
    real_open = module._open_coordination_file
    entered, release = threading.Event(), threading.Event()

    def delayed_open(root):
        entered.set()
        assert release.wait(10)
        return real_open(root)

    monkeypatch.setattr(module, "_open_coordination_file", delayed_open)
    worker = threading.Thread(target=lambda: _hold_once(module, tmp_path))
    child = None
    try:
        worker.start()
        assert entered.wait(10)
        child = os.fork()
        if child == 0:
            try:
                module._open_coordination_file = real_open
                with module.allocation_lock(tmp_path):
                    pass
                os._exit(0)
            except BaseException:
                os._exit(3)
        status = _wait_child(child)
        child = None
        assert os.waitstatus_to_exitcode(status) == 0
    finally:
        release.set()
        worker.join(10)
        if child is not None:
            try:
                os.kill(child, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                os.waitpid(child, 0)
            except ChildProcessError:
                pass


def _hold_once(module, path):
    with module.allocation_lock(path):
        pass


def test_keyboard_interrupt_during_acquire_and_unlock_leaves_no_record_lock(tmp_path, monkeypatch):
    import nc_rted.storage_lock as module

    real_lockf = module.fcntl.lockf
    acquire_calls = 0

    def interrupted_acquire(fd, operation):
        nonlocal acquire_calls
        if operation == module.fcntl.LOCK_EX and acquire_calls == 0:
            acquire_calls += 1
            real_lockf(fd, operation)
            raise KeyboardInterrupt()
        return real_lockf(fd, operation)

    monkeypatch.setattr(module.fcntl, "lockf", interrupted_acquire)
    with pytest.raises(KeyboardInterrupt):
        with module.allocation_lock(tmp_path):
            pass
    _assert_fresh_process_can_lock(tmp_path)

    unlock_calls = 0

    def interrupted_unlock(fd, operation):
        nonlocal unlock_calls
        if operation == module.fcntl.LOCK_UN and unlock_calls == 0:
            unlock_calls += 1
            real_lockf(fd, operation)
            raise KeyboardInterrupt()
        return real_lockf(fd, operation)

    monkeypatch.setattr(module.fcntl, "lockf", interrupted_unlock)
    with pytest.raises(KeyboardInterrupt):
        with module.allocation_lock(tmp_path):
            pass
    _assert_fresh_process_can_lock(tmp_path)


@pytest.mark.parametrize("phase", ("pid_check", "unlock_entry"))
def test_real_sigint_is_deferred_through_cleanup_boundaries(tmp_path, monkeypatch, phase):
    import nc_rted.storage_lock as module

    real_pid = os.getpid()
    sent = {"value": False}
    armed = {"value": False}
    if phase == "pid_check":
        original = module.os.getpid

        def signal_on_cleanup_pid():
            if armed["value"] and not sent["value"]:
                sent["value"] = True
                os.kill(real_pid, signal.SIGINT)
            return original()

        monkeypatch.setattr(module.os, "getpid", signal_on_cleanup_pid)
    else:
        original = module._unlock_after_interrupt

        def signal_on_unlock_entry(fd):
            if armed["value"] and not sent["value"]:
                sent["value"] = True
                os.kill(real_pid, signal.SIGINT)
            return original(fd)

        monkeypatch.setattr(module, "_unlock_after_interrupt", signal_on_unlock_entry)

    body_reached = False
    with pytest.raises(KeyboardInterrupt):
        with module.allocation_lock(tmp_path):
            body_reached = True
            armed["value"] = True
    assert body_reached and sent["value"]
    _assert_fresh_process_can_lock(tmp_path)


def _assert_fresh_process_can_lock(path):
    from nc_rted.storage_lock import allocation_lock

    child = os.fork()
    if child == 0:
        try:
            with allocation_lock(path):
                pass
            os._exit(0)
        except BaseException:
            os._exit(3)
    status = _wait_child(child)
    assert os.waitstatus_to_exitcode(status) == 0


def _wait_child(child, timeout=10):
    reaped = False
    try:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            waited, status = os.waitpid(child, os.WNOHANG)
            if waited == child:
                reaped = True
                return status
            time.sleep(.01)
        pytest.fail("child did not exit before allocation-lock test deadline")
    finally:
        if not reaped:
            try:
                os.kill(child, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                os.waitpid(child, 0)
            except ChildProcessError:
                pass


@pytest.mark.parametrize("child_signal", (False, True))
def test_child_signal_is_delivered_after_inherited_context_cleanup(tmp_path, monkeypatch, child_signal):
    import nc_rted.storage_lock as module

    received = []
    previous = signal.signal(signal.SIGINT, lambda signum, frame: received.append(os.getpid()))
    parent_pid = os.getpid()
    child = None
    try:
        with module.allocation_lock(tmp_path):
            # This pending parent signal must never be replayed in the child.
            os.kill(parent_pid, signal.SIGINT)
            assert not received
            child = os.fork()
            if child == 0:
                try:
                    # Acquire fresh child state inside the inherited context;
                    # parent releases its lock independently below.
                    with module.allocation_lock(tmp_path):
                        if child_signal:
                            os.kill(os.getpid(), signal.SIGINT)
                        assert not received
                    # Child deferral forwards into the inherited handler.
                    assert not received
                except BaseException:
                    os._exit(3)
        if child == 0:
            expected = [os.getpid()] if child_signal else []
            os._exit(0 if received == expected else 4)
        assert received == [parent_pid]
        status = _wait_child(child)
        child = None
        assert os.waitstatus_to_exitcode(status) == 0
        _assert_fresh_process_can_lock(tmp_path)
    finally:
        if child == 0:
            os._exit(5)
        if child is not None:
            try:
                os.kill(child, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                os.waitpid(child, 0)
            except ChildProcessError:
                pass
        signal.signal(signal.SIGINT, previous)


@pytest.mark.parametrize("interrupt", (False, True))
def test_wait_child_reaps_blocked_child_on_timeout_or_parent_interrupt(tmp_path, monkeypatch, interrupt):
    from nc_rted.storage_lock import allocation_lock

    child = None
    try:
        with allocation_lock(tmp_path):
            child = os.fork()
            if child == 0:
                try:
                    with allocation_lock(tmp_path):
                        pass
                    os._exit(0)
                except BaseException:
                    os._exit(3)
            # Raise directly: the outer allocation intentionally defers real
            # SIGINT. Here the tested boundary is exceptional parent polling.
            if interrupt:
                def interrupted_sleep(seconds):
                    raise KeyboardInterrupt()
                monkeypatch.setattr(time, "sleep", interrupted_sleep)
            expected = KeyboardInterrupt if interrupt else pytest.fail.Exception
            with pytest.raises(expected):
                _wait_child(child, timeout=.05)
            with pytest.raises(ChildProcessError):
                os.waitpid(child, os.WNOHANG)
            child = None
    finally:
        if child == 0:
            os._exit(5)
        if child is not None:
            try:
                os.kill(child, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                os.waitpid(child, 0)
            except ChildProcessError:
                pass
