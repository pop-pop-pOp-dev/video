"""Durable local child-process helpers for NC-RTED queue workers."""
from __future__ import annotations
from contextlib import contextmanager
import fcntl, json, os, socket, uuid
from pathlib import Path

def process_stat(pid: int) -> list[str] | None:
    """Parse procfs stat after its final ')' (comm may contain spaces/parentheses)."""
    try:
        raw = Path(f"/proc/{pid}/stat").read_text()
        close = raw.rfind(")")
        if close < 0: return None
        return [raw[:raw.find(" ")], raw[close + 2:close + 3], *raw[close + 4:].split()]
    except FileNotFoundError: return None
    except OSError: return ["?", "?"]

def process_starttime(pid: int) -> str | None:
    fields = process_stat(pid)
    # `comm` is collapsed from the kernel's two stat fields to one here.
    return fields[20] if fields and len(fields) > 20 else None

def process_live(pid: int | None, starttime: str | None, host: str | None = None) -> bool:
    if not pid or not starttime or (host and host != socket.gethostname()): return False
    fields = process_stat(pid)
    return bool(fields and len(fields) > 20 and fields[20] == str(starttime) and fields[1] != "Z")

def group_live(pgid: int | None) -> bool:
    """Return whether any non-zombie member remains in a detached group."""
    if not pgid:
        return False
    for entry in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = process_stat(int(entry.parent.name))
            if fields and len(fields) > 3 and fields[0] != "?" and fields[1] != "Z" and int(fields[3]) == int(pgid):
                return True
        except (OSError, ValueError, IndexError):
            continue
    return False

def group_state(pgid: int | None) -> str:
    """Return live/gone/unknown; never turn unreadable procfs into gone."""
    if not pgid: return "gone"
    unknown = False; zombie_member = False
    try: entries=list(Path("/proc").glob("[0-9]*/stat"))
    except OSError: return "unknown"
    for entry in entries:
        fields=process_stat(int(entry.parent.name))
        if fields is None:
            # A process can disappear during scan; only an existing unreadable
            # stat is uncertainty. The directory check distinguishes them.
            if entry.exists(): unknown=True
            continue
        if len(fields) <= 3 or fields[0] == "?": unknown=True; continue
        try:
            if int(fields[3]) == int(pgid):
                if fields[1] != "Z": return "live"
                zombie_member = True
        except ValueError: unknown=True
    if unknown: return "unknown"
    # Procfs enumeration is advisory. Ask the kernel whether this process group
    # still exists before allowing any caller to release its reservation.
    try: os.killpg(int(pgid), 0)
    except ProcessLookupError: return "gone"
    except PermissionError: return "unknown"
    # A zombie leader cannot create descendants. The supported worker contract
    # disallows background descendants after its leader exits; a sweep that saw
    # only zombies is therefore a completed group, while any live member above
    # retained the reservation.
    return "gone" if zombie_member else "live"

def group_members(pgid: int | None) -> dict[int, str] | None:
    """Snapshot non-zombie member PID/starttimes, or None on uncertainty."""
    if not pgid: return {}
    result={}
    try: entries=list(Path("/proc").glob("[0-9]*/stat"))
    except OSError: return None
    for entry in entries:
        fields=process_stat(int(entry.parent.name))
        if fields is None:
            if entry.exists(): return None
            continue
        if len(fields)<=20 or fields[0] == "?": return None
        try:
            if int(fields[3]) == int(pgid) and fields[1] != "Z": result[int(entry.parent.name)]=fields[20]
        except ValueError: return None
    return result

@contextmanager
def supervisor_lock(directory: str | Path):
    """Non-inherited exclusive controller lock for one attempt lifecycle."""
    directory=Path(directory); directory.mkdir(parents=True, exist_ok=True)
    path=directory / ".supervisor.lock"
    with path.open("a+") as handle:
        try: fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: yield False; return
        try: yield handle
        finally: fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

def acquire_supervisor_lock(directory: str | Path):
    """Return a held, non-inherited lock handle or False."""
    directory=Path(directory); directory.mkdir(parents=True, exist_ok=True)
    handle=(directory / ".supervisor.lock").open("a+")
    try: fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError: handle.close(); return False
    return handle

def attempt_state(pid: int | None, starttime: str | None, host: str | None = None) -> str:
    """Classify an attempt without mistaking PID reuse or zombies for exit."""
    if host and host != socket.gethostname():
        return "foreign"
    if not pid or not starttime:
        return "unknown"
    fields = process_stat(pid)
    if fields is None:
        group=group_state(pid); return "group_live" if group == "live" else "gone" if group == "gone" else "unknown"
    if len(fields) <= 20 or fields[0] == "?":
        return "unknown"
    if fields[20] != str(starttime):
        return "identity_mismatch"
    if fields[1] == "Z":
        group=group_state(pid); return "group_live" if group == "live" else "gone" if group == "gone" else "unknown"
    return "live"

def write_journal(path: str | Path, record: dict) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{uuid.uuid4().hex}")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n"); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)
    for stale in path.parent.glob(path.name + ".tmp*"):
        if stale != temporary:
            stale.unlink(missing_ok=True)
    descriptor = os.open(path.parent, os.O_DIRECTORY)
    try: os.fsync(descriptor)
    finally: os.close(descriptor)

@contextmanager
def gpu_lock(data_volume: str | Path, gpu: int | None):
    if gpu is None:
        yield None; return
    directory = Path(data_volume) / ".nc_rted_locks"; directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"gpu_{socket.gethostname()}_{gpu}.lock"
    with path.open("a+") as handle:
        try: fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: yield False; return
        # Do not explicitly unlock: the child inherits this file description.
        # Closing the controller's descriptor releases it only after the child
        # has also exited, while an explicit LOCK_UN would free a live child.
        yield handle
