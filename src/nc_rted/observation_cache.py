"""Process-safe, bounded cache of compact frozen frame observations."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import pickle
import shutil
import tempfile
import torch
from .storage_lock import allocation_lock, ensure_directory, open_lock_file

CACHE_SCHEMA = "nc_rted_frozen_frame_cache/v2"


def cache_key(media_hash: str, timestamp_s: float, detector_identity: dict, siglip_identity: dict) -> str:
    payload = {"schema": CACHE_SCHEMA, "media": media_hash, "timestamp": format(timestamp_s, ".9f"),
               "detector": detector_identity, "siglip": siglip_identity}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _compact(value):
    if isinstance(value, torch.Tensor):
        if value.requires_grad: raise ValueError("trainable tensors cannot enter the frozen frame cache")
        return value.detach().cpu().contiguous().clone()
    if isinstance(value, dict): return {key: _compact(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)): return [_compact(item) for item in value]
    if value is None or type(value) in {str, int, float, bool}: return value
    raise ValueError("unsupported frozen cache payload")


def _digest(value):
    digest = hashlib.sha256()
    def field(data):
        digest.update(str(len(data)).encode() + b":" + data)
    def visit(item):
        if isinstance(item, torch.Tensor):
            digest.update(b"T")
            field(json.dumps([str(item.dtype), list(item.shape)]).encode())
            field(item.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            digest.update(b"D"); field(str(len(item)).encode())
            for key in sorted(item):
                field(json.dumps(key).encode()); visit(item[key])
        elif isinstance(item, list):
            digest.update(b"L"); field(str(len(item)).encode())
            for part in item: visit(part)
        else:
            digest.update(b"J"); field(json.dumps(item, allow_nan=False).encode())
    visit(value)
    return digest.hexdigest()


class FrozenFrameCache:
    def __init__(self, root: Path, max_bytes: int, *, min_free_bytes: int = 20 << 30):
        if type(max_bytes) is not int or max_bytes <= 0 or min_free_bytes < 0:
            raise ValueError("invalid frame-cache resource limits")
        self.root, self.max_bytes, self.min_free_bytes = Path(root), max_bytes, min_free_bytes
        ensure_directory(self.root / "locks", self.min_free_bytes)

    def path(self, key):
        if not isinstance(key, str) or len(key) != 64 or any(char not in "0123456789abcdef" for char in key):
            raise ValueError("invalid content-addressed cache key")
        return self.root / f"{key}.pt"

    @contextmanager
    def _lock(self):
        with open_lock_file(self.root / ".cache.lock", self.min_free_bytes) as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            yield

    @contextmanager
    def population(self, keys):
        # Fixed lock stripes bound filesystem overhead even for millions of
        # frames. Every worker acquires unique stripes in the same order.
        handles = []
        try:
            stripes = sorted({int(self.path(key).stem[:4], 16) % 256 for key in keys})
            for stripe in stripes:
                handle = open_lock_file(self.root / "locks" / f"{stripe:03d}.lock", self.min_free_bytes)
                handles.append(handle); fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            for handle in reversed(handles): handle.close()

    def get(self, key):
        path = self.path(key)
        with self._lock():
            if not path.is_file(): return None
            try:
                value = torch.load(path, map_location="cpu", weights_only=True)
                if (not isinstance(value, dict) or value.get("schema") != CACHE_SCHEMA or value.get("key") != key
                        or value.get("sha256") != _digest(value["payload"])):
                    raise ValueError("cache identity or content checksum mismatch")
                os.utime(path, None)
                return value["payload"]
            except (OSError, ValueError, KeyError, EOFError, RuntimeError, pickle.UnpicklingError):
                # Corrupt/replaced cache products are recomputable, never truth.
                path.unlink(missing_ok=True)
                with allocation_lock(self.root):
                    self._reserve(3 * max(4096, os.statvfs(self.root).f_frsize))
                    with (self.root / "invalidations.jsonl").open("a") as log:
                        log.write(json.dumps({"key":key,"reason":"invalid_frame_cache"}) + "\n")
                return None

    def _reserve(self,additional_bytes=0):
        if shutil.disk_usage(self.root).free < self.min_free_bytes+additional_bytes:
            raise OSError("disk hard limit: frame cache must preserve reserve")

    def put(self, key, value):
        target = self.path(key)
        payload = _compact(value)
        record = {"schema":CACHE_SCHEMA,"key":key,"payload":payload,"sha256":_digest(payload)}
        serialized=io.BytesIO()
        torch.save(record,serialized)
        content=serialized.getvalue()
        with self._lock(), allocation_lock(self.root):
            block=max(4096,os.statvfs(self.root).f_frsize)
            allocation=((len(content)+block-1)//block)*block+2*block
            self._reserve(allocation)
            descriptor, name = tempfile.mkstemp(prefix=".frame-", suffix=".tmp", dir=self.root)
            temporary = Path(name)
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(content); handle.flush(); os.fsync(handle.fileno())
                self._reserve()
                os.replace(temporary, target)
                self._evict_locked()
                directory_fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
                try: os.fsync(directory_fd)
                finally: os.close(directory_fd)
            finally:
                temporary.unlink(missing_ok=True)

    def _evict_locked(self):
        files = sorted(self.root.glob("*.pt"), key=lambda path: path.stat().st_atime_ns)
        total = sum(path.stat().st_size for path in files)
        for path in files:
            if total <= self.max_bytes: break
            size = path.stat().st_size; path.unlink(); total -= size

    def evict(self):
        with self._lock(): self._evict_locked()
