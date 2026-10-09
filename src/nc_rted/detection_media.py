"""Bounded original 4-FPS decoding with hash-bound frozen Fast query scores.

No labels, teacher targets or Slow output are consumed. A snapshot holds only
media identity, the inherited sampling protocol and frozen per-query scores.
"""
from __future__ import annotations

import fcntl
import hashlib
import os
import json
import math
from pathlib import Path
from typing import Callable

import torch

from .detection_provider import DetectionPrefix, DetectionProtocol, DetectionQuery
from .task_inputs import TaskInputError, sha256_file


class OpenCVFrames:
    def __init__(self, path: Path):
        import cv2
        self.cv2 = cv2
        self.capture = cv2.VideoCapture(str(path))
        if not self.capture.isOpened():
            self.capture.release()
            raise TaskInputError("cannot open bound training media")
        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        self.frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))

    def read(self, index: int):
        from PIL import Image
        self.capture.set(self.cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self.capture.read()
        if not ok:
            raise TaskInputError(f"bound original frame {index} could not be decoded")
        return Image.fromarray(self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB))

    def close(self):
        self.capture.release()


class StreamingDetectionReader:
    """Exact inherited grid groups, retaining at most four RGB frames per query.

    ``encode`` is the inherited SigLIP adapter. Last-frame encoding is a separate
    call as in the original evaluator; optional RT encoding is a four-frame batch.
    Inputs are read on iteration, and the decoder is closed on early termination.
    """
    def __init__(self, snapshot: str | Path, *, snapshot_sha256: str,
                 fast_identity: dict, protocols: dict[str, DetectionProtocol],
                 encode: Callable, decoder_factory=OpenCVFrames):
        if sha256_file(snapshot) != snapshot_sha256:
            raise TaskInputError("frozen Fast snapshot hash differs")
        document = json.loads(Path(snapshot).read_text())
        if document.get("schema") != "nc_rted_frozen_fast/v1" or document.get("fast_identity") != fast_identity or not fast_identity:
            raise TaskInputError("unbound frozen Fast implementation/checkpoint identity")
        self.rows = {}
        for row in document.get("media", []):
            identity = (row["dataset"], row["media_key"])
            if identity in self.rows:
                raise TaskInputError("duplicate frozen media identity")
            self.rows[identity] = row
        if not self.rows:
            raise TaskInputError("empty frozen Fast snapshot")
        self.protocols, self.encode, self.decoder_factory = protocols, encode, decoder_factory
        self.snapshot_sha256 = snapshot_sha256

    def __call__(self, dataset: str, media_key: str, target_query_index: int) -> DetectionPrefix:
        if type(target_query_index) is not int or target_query_index < 0:
            raise TaskInputError("invalid selected query")
        row = self.rows.get((dataset, media_key))
        if row is None:
            raise TaskInputError("selected media absent from frozen Fast snapshot")
        protocol = self.protocols[dataset]
        protocol.validate()
        path = Path(row["media_path"])
        # Read-only file hashing does not decode or infer future frames. The
        # snapshot is bound once; current media content is checked on each replay.
        if not path.is_file() or sha256_file(path) != row["media_sha256"]:
            raise TaskInputError("bound training media content changed")
        fps, total = row["fps"], row["frame_count"]
        if not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0 or type(total) is not int or total <= 0:
            raise TaskInputError("invalid original media metadata")
        interval = max(1, int(fps / 4))
        sampled_count = (total + interval - 1) // interval
        query_count = (sampled_count + 3) // 4
        queries = row["queries"]
        if row.get("target_fps") != 4 or row.get("query_interval") != 4 or len(queries) != query_count or target_query_index >= len(queries):
            raise TaskInputError("frozen Fast sampling/count differs from original four-frame protocol")
        if any(type(row.get(key)) is not int or row[key] <= 0 for key in ("height", "width")):
            raise TaskInputError("invalid original image dimensions")
        # Validate the selected causal prefix before opening the decoder.
        for index in range(target_query_index + 1):
            item = queries[index]
            expected = list(range(index * 4 * interval, min((index + 1) * 4 * interval, total), interval))
            if item.get("index") != index or item.get("frame_indices") != expected:
                raise TaskInputError("frozen Fast frame groups differ from original sampler")
            score = item.get("fast_score")
            if not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
                raise TaskInputError("missing or invalid frozen Fast score")

        def iterate():
            # Hash and decode the same open inode. /proc/self/fd pins that inode
            # even if another process replaces the pathname after verification.
            # A shared lease also excludes cooperative cache writers/eviction.
            media = path.open("rb")
            decoder = None
            try:
                fcntl.flock(media.fileno(), fcntl.LOCK_SH)
                before = os.fstat(media.fileno())
                digest = hashlib.sha256()
                for part in iter(lambda: media.read(8 << 20), b""):
                    digest.update(part)
                def signature():
                    value = os.fstat(media.fileno())
                    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)
                expected_signature = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
                if digest.hexdigest() != row["media_sha256"] or signature() != expected_signature:
                    raise TaskInputError("bound training media content changed before decode")
                media.seek(0)
                decoder = self.decoder_factory(Path(f"/proc/self/fd/{media.fileno()}"))
                if (abs(decoder.fps - fps) > 1e-6 or decoder.frame_count != total or
                        decoder.height != row["height"] or decoder.width != row["width"]):
                    raise TaskInputError("decoded metadata differs from frozen Fast source")
                for index in range(target_query_index + 1):
                    item = queries[index]
                    indices = tuple(item["frame_indices"])
                    if signature() != expected_signature:
                        raise TaskInputError("leased media was modified during prefix decoding")
                    frames = [decoder.read(frame) for frame in indices]
                    if signature() != expected_signature:
                        raise TaskInputError("leased media was modified during prefix decoding")
                    last = self.encode([frames[-1]])
                    if not isinstance(last, torch.Tensor) or last.shape != (1, 729, 1152):
                        raise TaskInputError("inherited last-frame SigLIP output shape differs")
                    dense = None
                    if index == target_query_index and protocol.rt_anomaly:
                        # Original grid creation pads the list in place. Repeat
                        # only the last already observed frame, never a future one.
                        padded = frames + [frames[-1]] * (4 - len(frames))
                        dense = self.encode(padded)
                    yield DetectionQuery(index, indices, tuple(frame / fps for frame in indices),
                                         float(item["fast_score"]), last[0].detach(),
                                         dense.detach() if dense is not None else None)
            finally:
                if decoder is not None:
                    decoder.close()
                media.close()
        return DetectionPrefix(iterate(), row["height"], row["width"])
