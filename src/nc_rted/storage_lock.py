"""Serialize cooperating allocations with POSIX process record locks on Linux.

Coordination descriptors intentionally remain open for the process lifetime.
Closing any descriptor for an inode releases that process's POSIX record locks,
so neither normal cleanup nor a forked child may close a cached descriptor.
The operating system reclaims these small per-filesystem descriptors at exit.
"""
from __future__ import annotations

from contextlib import contextmanager
import errno
import fcntl
import os
from pathlib import Path
import shutil
import signal
import threading


_COORDINATION_NAME = ".nc_rted_allocation.lock"
_COORDINATION_RESERVE = 20 * 1024 ** 3
_PROCESS_STATES: dict[tuple[int, str], "_ProcessState"] = {}


class _DeferredSigint:
    """Delay Ctrl-C across one main-thread outer allocation transition."""
    def __init__(self) -> None:
        self.pid = os.getpid()
        self.pending: tuple[int, int] | None = None
        self.previous = signal.signal(signal.SIGINT, self._record)

    def _record(self, signum, frame) -> None:
        # Tag actual reception/forwarding in the receiving process. A nested
        # child deferral may forward here before the inherited context exits.
        # finish() still suppresses pending parent signals copied by fork.
        self.pending = (os.getpid(), signum)

    def finish(self) -> None:
        signal.signal(signal.SIGINT, self.previous)
        pending = self.pending
        if pending is None or pending[0] != os.getpid():
            return
        _, signum = pending
        if self.previous is signal.SIG_IGN:
            return
        if self.previous is signal.SIG_DFL:
            signal.default_int_handler(signum, None)
        else:
            self.previous(signum, None)


def _main_thread_deferral() -> _DeferredSigint | None:
    if threading.current_thread() is threading.main_thread():
        return _DeferredSigint()
    return None


def _process_identity() -> tuple[int, str]:
    """PID plus Linux starttime prevents PID-reuse and stale-fork confusion."""
    try:
        # /proc/<pid>/stat field 22 is starttime; comm may contain spaces.
        fields = Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()
        return os.getpid(), fields[19]
    except (OSError, IndexError) as error:
        raise OSError("Linux /proc/self/stat is required for allocation locking") from error


class _FilesystemState:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.fd: int | None = None


class _ProcessState:
    def __init__(self) -> None:
        self.registry_mutex = threading.Lock()
        self.filesystems: dict[tuple[int, int], _FilesystemState] = {}

    def filesystem(self, key: tuple[int, int]) -> _FilesystemState:
        with self.registry_mutex:
            return self.filesystems.setdefault(key, _FilesystemState())


def _state_for_current_process() -> tuple[tuple[int, str], _ProcessState]:
    identity = _process_identity()
    # CPython dict.setdefault is atomic under the GIL. A global mutex could be
    # inherited locked after a multithreaded fork, so do not introduce one.
    return identity, _PROCESS_STATES.setdefault(identity, _ProcessState())


def existing_parent(path: Path) -> Path:
    path = Path(path).resolve()
    while not path.exists():
        path = path.parent
    return path if path.is_dir() else path.parent


def filesystem_root(path: Path) -> Path:
    root = existing_parent(path)
    device = root.stat().st_dev
    while root.parent != root and root.parent.stat().st_dev == device:
        root = root.parent
    return root


def _open_coordination_file(root: Path) -> int:
    """Open one fixed root-local inode. It is never unlinked or closed here."""
    path = root / _COORDINATION_NAME
    flags = os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        # Existing coordination costs no new allocation, even at low headroom.
        return os.open(path, flags)
    except FileNotFoundError:
        pass
    try:
        # Metadata is charged only to the winner of atomic O_EXCL creation.
        block = max(4096, os.statvfs(root).f_frsize)
        if shutil.disk_usage(root).free < _COORDINATION_RESERVE + 2 * block:
            raise OSError("disk hard limit: allocation coordination creation must preserve reserve")
        return os.open(path, flags | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return os.open(path, flags, 0o600)
    except OSError as error:
        if error.errno == errno.EEXIST:
            return os.open(path, flags, 0o600)
        raise


def _coordination_fd(state: _FilesystemState, root: Path) -> int:
    if state.fd is None:
        state.fd = _open_coordination_file(root)
    return state.fd


def _unlock_after_interrupt(fd: int) -> None:
    """Complete idempotent unlock before delivering a pending SIGINT."""
    interrupted = None
    while True:
        try:
            fcntl.lockf(fd, fcntl.LOCK_UN)
            break
        except KeyboardInterrupt as error:
            if interrupted is None:
                interrupted = error
            continue
    if interrupted is not None:
        raise interrupted


@contextmanager
def allocation_lock(path):
    """Hold shared allocation admission through the caller's allocation.

    The outermost entry in one thread obtains a POSIX record lock. Nested calls
    are recognized by native RLock ownership rather than an interruptible depth
    counter. POSIX record locks are not inherited across fork; a child receives
    a new process-identity state and ignores inherited parent state.
    """
    root = filesystem_root(path)
    root_stat = root.stat()
    identity, process = _state_for_current_process()
    state = process.filesystem((root_stat.st_dev, root_stat.st_ino))
    was_owned = state.lock._is_owned()
    deferred = None if was_owned else _main_thread_deferral()
    try:
        with state.lock:
            if was_owned:
                yield
            else:
                fd = _coordination_fd(state, root)
                try:
                    fcntl.lockf(fd, fcntl.LOCK_EX)
                    yield
                finally:
                    # A parent context copied by fork must not unlock a new
                    # child process-state lock during inherited cleanup. The
                    # main-thread deferred handler covers this PID check and
                    # helper entry; non-main threads retain lockf cleanup.
                    if identity[0] == os.getpid():
                        _unlock_after_interrupt(fd)
    finally:
        # Restore and replay only after kernel unlock and native RLock release.
        if deferred is not None:
            deferred.finish()


def ensure_directory(path, reserve):
    """Create missing directories under the same allocation admission."""
    path = Path(path)
    with allocation_lock(path):
        ancestor = existing_parent(path)
        missing = len(path.resolve().relative_to(ancestor).parts)
        if not missing:
            return
        block = max(4096, os.statvfs(ancestor).f_frsize)
        if shutil.disk_usage(ancestor).free < reserve + (2 * missing + 2) * block:
            raise OSError("disk hard limit: directory creation must preserve reserve")
        path.mkdir(parents=True, exist_ok=True)
        # Directory entries are part of the durable publication contract.
        current = path.resolve()
        while True:
            fd = os.open(current, os.O_DIRECTORY)
            try: os.fsync(fd)
            finally: os.close(fd)
            if current == ancestor:
                break
            current = current.parent


def open_lock_file(path, reserve):
    """Allocate lock-file directory metadata before waiting on its own flock."""
    path = Path(path)
    with allocation_lock(path):
        if not path.exists():
            block = max(4096, os.statvfs(path.parent).f_frsize)
            if shutil.disk_usage(path.parent).free < reserve + 2 * block:
                raise OSError("disk hard limit: lock-file creation must preserve reserve")
        return path.open("a+b")
