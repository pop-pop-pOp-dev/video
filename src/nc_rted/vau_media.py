"""Label-free materialization of the official VAU test media identities.

This module intentionally projects only media identity and ``events``/``clips``
time ranges from the public annotation databases.  It must never be given the
other annotation fields, in particular labels or textual answers.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Iterable


CATALOG_SCHEMA = "nc_rted_blind_media_catalog/v1"
BINDINGS_SCHEMA = "nc_rted_vau_media_bindings/v1"
JOURNAL_SCHEMA = "nc_rted_vau_materialization_journal/v1"
MIN_FREE_BYTES = 20 * 1024**3
MAX_ARTIFACT_BYTES = 50 * 1024**3
_DERIVED = re.compile(r"^(?P<parent>.+)_E(?P<event>[0-9]+)(?:C(?P<clip>[0-9]+))?$")
MAX_BATCH_BYTES = 128 * 1024 * 1024


class VAUMediaError(RuntimeError):
    pass


@dataclass(frozen=True)
class Geometry:
    fps: float
    frame_count: int
    height: int
    width: int

    def validate(self) -> None:
        if (not math.isfinite(self.fps) or self.fps <= 0 or
                any(type(x) is not int or x <= 0 for x in (self.frame_count, self.height, self.width))):
            raise VAUMediaError("invalid decoded geometry")


@dataclass(frozen=True)
class MediaPlan:
    dataset: str
    video: str
    parent_key: str
    parent_path: Path
    range_s: tuple[float, float] | None
    output_path: Path | None

    @property
    def derived(self) -> bool:
        return self.range_s is not None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("utf-8")


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_json_new(path: Path, value: Any) -> str:
    if path.exists():
        raise VAUMediaError(f"refusing to overwrite immutable output {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _json_bytes(value)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_dir(path.parent)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return hashlib.sha256(payload).hexdigest()


def _atomic_bytes_new(path: Path, payload: bytes) -> str:
    """Publish recovery evidence before the damaged journal is truncated."""
    if path.exists():
        if path.read_bytes() != payload:
            raise VAUMediaError("immutable recovery evidence differs")
        return hashlib.sha256(payload).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_dir(path.parent)
    except BaseException:
        try: os.unlink(temporary)
        except FileNotFoundError: pass
        raise
    return hashlib.sha256(payload).hexdigest()


@contextmanager
def output_lock(root: Path):
    import fcntl
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".materialize.lock"
    with lock.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield


def _free_bytes(path: Path) -> int:
    return shutil.disk_usage(path).free


def _require_space(root: Path, hard_guard_bytes: int) -> None:
    if hard_guard_bytes < MIN_FREE_BYTES:
        raise VAUMediaError("hard guard may not be below 20 GiB")
    if _free_bytes(root) < hard_guard_bytes:
        raise VAUMediaError("free space is below configured hard guard")


def _read_identity(path: Path) -> list[dict[str, Any]]:
    try:
        raw = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise VAUMediaError("identity JSON is unreadable") from error
    if not isinstance(raw, list):
        raise VAUMediaError("identity JSON must be a list")
    rows = []
    for row in raw:
        if not isinstance(row, dict) or set(row).isdisjoint({"id", "video"}):
            raise VAUMediaError("identity row lacks id/video")
        identifier, video = row.get("id"), row.get("video")
        if not isinstance(identifier, (str, int)) or not isinstance(video, str) or not video.endswith(".mp4"):
            raise VAUMediaError("invalid identity id/video")
        # Explicit projection: prompt/task/type and every other field are discarded.
        rows.append({"id": identifier, "video": video})
    if len({str(row["id"]) for row in rows}) != len(rows):
        raise VAUMediaError("duplicate question id")
    if len(rows) != 3339:
        raise VAUMediaError(f"official VAU question denominator is {len(rows)}, expected 3339")
    return rows


def project_ranges(path: Path) -> dict[str, dict[str, Any]]:
    """Read only filename/events/clips from one official test annotation DB."""
    try:
        raw = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise VAUMediaError(f"cannot read annotation DB {path}") from error
    if not isinstance(raw, dict):
        raise VAUMediaError("annotation DB must map filename to media definitions")
    result = {}
    for filename, value in raw.items():
        if not isinstance(filename, str) or not isinstance(value, dict):
            raise VAUMediaError("invalid annotation media definition")
        events, clips = value.get("events", []), value.get("clips", [])
        if not isinstance(events, list) or not isinstance(clips, list) or len(events) != len(clips):
            raise VAUMediaError(f"invalid ranges for {filename}")
        clean_events, clean_clips = [], []
        for event, event_clips in zip(events, clips):
            clean_events.append(_range(event, filename))
            if not isinstance(event_clips, list):
                raise VAUMediaError(f"invalid clips for {filename}")
            clean_clips.append([_range(item, filename) for item in event_clips])
        result[filename] = {"events": clean_events, "clips": clean_clips}
    return result


def _range(value: object, name: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise VAUMediaError(f"invalid range for {name}")
    start, end = float(value[0]), float(value[1])
    if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
        raise VAUMediaError(f"invalid range endpoints for {name}")
    return start, end


def _parts(video: str) -> tuple[str, str, int | None, int | None]:
    pieces = video.split("/")
    if len(pieces) != 4 or pieces[1] not in {"videos", "events", "clips"} or pieces[2] != "test":
        raise VAUMediaError(f"unsupported official VAU path {video}")
    stem = Path(pieces[-1]).stem
    match = _DERIVED.fullmatch(stem)
    if pieces[1] == "videos":
        if match is not None:
            raise VAUMediaError("direct-video identity has derived suffix")
        return pieces[0], stem, None, None
    if match is None:
        raise VAUMediaError(f"derived official VAU filename is invalid {video}")
    parent, event, clip = match.group("parent"), match.group("event"), match.group("clip")
    if pieces[1] == "events" and clip is not None:
        raise VAUMediaError("event identity has a clip suffix")
    if pieces[1] == "clips" and clip is None:
        raise VAUMediaError("clip identity has no clip index")
    return pieces[0], parent, None if event is None else int(event), None if clip is None else int(clip)


def build_plans(*, identity_path: Path, ucf_ranges: Path, xd_ranges: Path,
                ucf_root: Path, xd_root: Path, output_root: Path) -> list[MediaPlan]:
    identities = _read_identity(identity_path)
    ranges = {"ucf-crime": project_ranges(ucf_ranges), "xd-violence": project_ranges(xd_ranges)}
    raw_roots = {"ucf-crime": ucf_root, "xd-violence": xd_root}
    unique = sorted({row["video"] for row in identities})
    wanted = {"ucf-crime": set(), "xd-violence": set()}
    parsed = {}
    for video in unique:
        dataset_name, parent, event, clip = _parts(video)
        if dataset_name not in wanted:
            raise VAUMediaError(f"unsupported official VAU dataset {dataset_name}")
        wanted[dataset_name].add(parent)
        parsed[video] = (dataset_name, parent, event, clip)
    # Index each raw tree once.  A per-identity rglob turns a bounded planning
    # pass into millions of redundant directory walks on the official corpus.
    resolved = {}
    for dataset_name, root in raw_roots.items():
        matches = {key: [] for key in wanted[dataset_name]}
        for candidate in root.rglob("*.mp4"):
            if candidate.stem in matches:
                matches[candidate.stem].append(candidate.resolve())
        for parent, candidates in matches.items():
            if len(candidates) != 1:
                raise VAUMediaError(f"raw parent resolution is not unique for {dataset_name}:{parent}")
            resolved[(dataset_name, parent)] = candidates[0]
    plans = []
    for video in unique:
        dataset_name, parent, event, clip = parsed[video]
        if dataset_name not in ranges or parent not in ranges[dataset_name]:
            raise VAUMediaError(f"range projection has no parent for {video}")
        selected_range = None
        if event is not None:
            events, clips = ranges[dataset_name][parent]["events"], ranges[dataset_name][parent]["clips"]
            if event >= len(events):
                raise VAUMediaError(f"event index outside projected ranges for {video}")
            selected_range = clips[event][clip] if clip is not None else events[event]
        safe_dataset = "ucf" if dataset_name == "ucf-crime" else "xd"
        derived_path = None if selected_range is None else output_root / "media" / safe_dataset / Path(video).name
        plans.append(MediaPlan(safe_dataset, video, parent, resolved[(dataset_name, parent)], selected_range, derived_path))
    if len(plans) != 1369:
        raise VAUMediaError(f"official VAU media denominator is {len(plans)}, expected 1369")
    if sum(plan.derived for plan in plans) != 1219:
        raise VAUMediaError("official VAU derived denominator is not 1219")
    return plans


def _probe(path: Path) -> Geometry:
    try:
        import decord
    except ImportError as error:
        raise VAUMediaError("decord is required for official splitter-compatible materialization") from error
    try:
        reader = decord.VideoReader(str(path))
    except Exception as error:
        raise VAUMediaError("decoded child cannot be opened") from error
    try:
        if len(reader) < 1:
            raise VAUMediaError(f"empty decoded video {path}")
        frame = reader[0].asnumpy()
        geometry = Geometry(float(reader.get_avg_fps()), len(reader), int(frame.shape[0]), int(frame.shape[1]))
        geometry.validate()
        return geometry
    finally:
        del reader


def _expected_child_geometry(plan: MediaPlan, parent: Geometry) -> Geometry:
    if not plan.derived:
        return parent
    start, end = int(plan.range_s[0] * parent.fps), min(int(plan.range_s[1] * parent.fps), parent.frame_count - 1)
    if end <= start:
        raise VAUMediaError(f"official splitter would produce no frames for {plan.video}")
    return Geometry(parent.fps, end - start, parent.height, parent.width)


def _verify_child_decode(path: Path, expected: Geometry) -> Geometry:
    """Decode every child frame before accepting its bound geometry."""
    try:
        import decord
    except ImportError as error:
        raise VAUMediaError("decord is required for full child verification") from error
    try:
        reader = decord.VideoReader(str(path))
    except Exception as error:
        raise VAUMediaError("decoded child cannot be opened") from error
    try:
        actual = Geometry(float(reader.get_avg_fps()), len(reader), 0, 0)
        if not math.isclose(actual.fps, expected.fps, rel_tol=0.0, abs_tol=1e-6) or actual.frame_count != expected.frame_count:
            raise VAUMediaError("decoded child fps/frame count differs from official range")
        for offset in range(0, actual.frame_count, max(1, min(512, actual.frame_count))):
            try:
                frames = reader.get_batch(range(offset, min(offset + 512, actual.frame_count))).asnumpy()
            except Exception as error:
                raise VAUMediaError("decoded child frame stream is unreadable") from error
            if frames.ndim != 4 or frames.shape[0] != min(512, actual.frame_count - offset) or frames.shape[-1] != 3:
                raise VAUMediaError("decoded child frame stream is incomplete")
            height, width = int(frames.shape[1]), int(frames.shape[2])
            if height != expected.height or width != expected.width:
                raise VAUMediaError("decoded child geometry differs from parent range")
        return Geometry(actual.fps, actual.frame_count, expected.height, expected.width)
    finally:
        del reader


def _split_official_chunked(reader, segment: tuple[float, float], save_path: Path,
                            before_batch=None) -> None:
    """Official split_video.py semantics with bounded Decord batches.

    The source splitter uses ``range(int(start * fps), min(int(end * fps),
    len(reader) - 1))`` and OpenCV ``mp4v``.  The initial frame is decoded once
    only to determine a safe batch size; all indices remain exactly right-open.
    """
    try:
        import cv2
    except ImportError as error:
        raise VAUMediaError("OpenCV is required for materialization") from error
    fps = float(reader.get_avg_fps())
    start = int(segment[0] * fps)
    end = min(int(segment[1] * fps), len(reader) - 1)
    if end <= start:
        raise VAUMediaError("official splitter would produce no frames")
    first = reader.get_batch([start]).asnumpy()
    if first.ndim != 4 or first.shape[0] != 1 or first.shape[-1] != 3:
        raise VAUMediaError("decoded RGB frame geometry is invalid")
    height, width = int(first.shape[1]), int(first.shape[2])
    bytes_per_frame = int(first[0].nbytes)
    if bytes_per_frame <= 0:
        raise VAUMediaError("decoded frame has no bytes")
    batch = max(1, MAX_BATCH_BYTES // bytes_per_frame)
    writer = cv2.VideoWriter(str(save_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise VAUMediaError(f"could not open video writer {save_path}")
    try:
        if before_batch is not None:
            before_batch(1)
        writer.write(cv2.cvtColor(first[0], cv2.COLOR_RGB2BGR))
        for offset in range(start + 1, end, batch):
            stop = min(offset + batch, end)
            if before_batch is not None:
                before_batch(stop - offset)
            frames = reader.get_batch(range(offset, stop)).asnumpy()
            for frame in frames:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def _record(journal: Path, event: dict[str, Any]) -> None:
    _repair_interrupted_tail(journal)
    event = {"schema": JOURNAL_SCHEMA, **event}
    with journal.open("ab") as handle:
        handle.write(_json_bytes(event))
        handle.flush()
        os.fsync(handle.fileno())


def _repair_interrupted_tail(journal: Path) -> None:
    """Discard only an unterminated final write, never a committed journal row."""
    if not journal.exists():
        return
    raw = journal.read_bytes()
    if not raw or raw.endswith(b"\n"):
        return
    prefix, _, tail = raw.rpartition(b"\n")
    try:
        json.loads(tail)
    except ValueError:
        # A journal write becomes committed only after newline + fsync.  Retain
        # interrupted bytes as immutable evidence before resuming the append log.
        recovery = journal.with_name(f"{journal.name}.interrupted-{hashlib.sha256(tail).hexdigest()}.bin")
        _atomic_bytes_new(recovery, tail)
        with journal.open("r+b") as handle:
            handle.truncate(len(prefix) + (1 if prefix else 0))
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_dir(journal.parent)
        return
    # A complete JSON row without the commit newline is also uncommitted.
    recovery = journal.with_name(f"{journal.name}.interrupted-{hashlib.sha256(tail).hexdigest()}.bin")
    _atomic_bytes_new(recovery, tail)
    with journal.open("r+b") as handle:
        handle.truncate(len(prefix) + (1 if prefix else 0))
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_dir(journal.parent)


def _completed(journal: Path) -> dict[str, dict[str, Any]]:
    result = {}
    if not journal.exists():
        return result
    _repair_interrupted_tail(journal)
    for line in journal.read_text().splitlines():
        try:
            row = json.loads(line)
        except ValueError as error:
            raise VAUMediaError("journal is corrupt") from error
        if row.get("schema") != JOURNAL_SCHEMA:
            raise VAUMediaError("journal schema differs")
        if row.get("status") == "complete":
            result[row["video"]] = row
    return result


def _artifact_bytes(root: Path) -> int:
    media = root / "media"
    return sum(path.stat().st_size for path in media.rglob("*") if path.is_file()) if media.exists() else 0


def _child_raw_rgb_bytes(plan: MediaPlan, geometry: Geometry) -> int:
    if not plan.derived:
        return 0
    start, end = int(plan.range_s[0] * geometry.fps), min(int(plan.range_s[1] * geometry.fps), geometry.frame_count - 1)
    if end <= start:
        raise VAUMediaError(f"official splitter would produce no frames for {plan.video}")
    return (end - start) * geometry.height * geometry.width * 3


def _row(plan: MediaPlan, path: Path, digest: str, geometry: Geometry) -> dict[str, Any]:
    return {"dataset": plan.dataset, "media_key": plan.video, "media_path": str(path.resolve()),
            "media_sha256": digest, "fps": geometry.fps, "frame_count": geometry.frame_count,
            "height": geometry.height, "width": geometry.width, "request_index": None}


def _validate_record(plan: MediaPlan, record: dict[str, Any], parent_sha: str,
                     parent_geometry: Geometry) -> dict[str, Any]:
    path = plan.output_path if plan.derived else plan.parent_path
    if (record.get("parent_sha256") != parent_sha or record.get("range_s") != (None if plan.range_s is None else list(plan.range_s))
            or record.get("media_path") != str(path.resolve()) or not path.is_file()):
        raise VAUMediaError("completed record binding differs from current immutable plan")
    digest = sha256_file(path)
    if digest != record.get("media_sha256"):
        raise VAUMediaError("completed record child hash differs")
    geometry = _verify_child_decode(path, _expected_child_geometry(plan, parent_geometry)) if plan.derived else _probe(path)
    expected = record.get("geometry")
    if expected != {"fps": geometry.fps, "frame_count": geometry.frame_count, "height": geometry.height, "width": geometry.width}:
        raise VAUMediaError("completed record decoded geometry differs")
    return _row(plan, path, digest, geometry)


def estimate(plans: Iterable[MediaPlan]) -> dict[str, float | int]:
    """Conservative source-byte estimate before running an expensive derivation."""
    plans = list(plans)
    total_duration = 0.0
    derived_duration = 0.0
    estimated_bytes = 0.0
    geometries: dict[Path, Geometry] = {}
    for plan in plans:
        geometry = geometries.get(plan.parent_path)
        if geometry is None:
            geometry = _probe(plan.parent_path)
            geometries[plan.parent_path] = geometry
        duration = geometry.frame_count / geometry.fps
        total_duration += duration if not plan.derived else plan.range_s[1] - plan.range_s[0]
        if plan.derived:
            span = plan.range_s[1] - plan.range_s[0]
            derived_duration += span
            estimated_bytes += plan.parent_path.stat().st_size * span / duration
    return {"unique_media": len(plans), "derived_media": sum(1 for plan in plans if plan.derived),
            "total_selected_duration_s": total_duration, "derived_duration_s": derived_duration,
            "source_bitrate_scaled_estimate_bytes": int(math.ceil(estimated_bytes))}


def raw_frame_capacity(plans: Iterable[MediaPlan]) -> dict[str, Any]:
    """Strict RGB-frame ceiling for new child bytes before codec compression.

    This is a storage reservation bound, not a prediction of mp4v compression.
    The derived writer receives exactly these right-open frames as BGR/RGB arrays.
    """
    geometry: dict[Path, Geometry] = {}
    total = {"children": 0, "parent_originals": set(), "frames": 0, "raw_rgb_bytes": 0,
             "max_child_raw_rgb_bytes": 0, "max_child_video": ""}
    by_dataset: dict[str, dict[str, Any]] = {}
    for plan in plans:
        if not plan.derived:
            continue
        item = geometry.get(plan.parent_path)
        if item is None:
            item = _probe(plan.parent_path)
            geometry[plan.parent_path] = item
        start, end = int(plan.range_s[0] * item.fps), min(int(plan.range_s[1] * item.fps), item.frame_count - 1)
        frames = end - start
        if frames <= 0:
            raise VAUMediaError(f"official splitter would produce no frames for {plan.video}")
        raw = frames * item.height * item.width * 3
        bucket = by_dataset.setdefault(plan.dataset, {"children": 0, "parent_originals": set(), "frames": 0,
                                                       "raw_rgb_bytes": 0, "max_child_raw_rgb_bytes": 0,
                                                       "max_child_video": ""})
        for target in (bucket, total):
            target["children"] += 1
            target["parent_originals"].add(str(plan.parent_path))
            target["frames"] += frames
            target["raw_rgb_bytes"] += raw
            if raw > target["max_child_raw_rgb_bytes"]:
                target["max_child_raw_rgb_bytes"] = raw
                target["max_child_video"] = plan.video
    def public(value: dict[str, Any]) -> dict[str, Any]:
        value = dict(value)
        value["parent_originals"] = len(value["parent_originals"])
        value["raw_rgb_gib"] = value["raw_rgb_bytes"] / 1024**3
        value["max_child_raw_rgb_gib"] = value["max_child_raw_rgb_bytes"] / 1024**3
        return value
    return {"by_dataset": {key: public(value) for key, value in sorted(by_dataset.items())}, "total": public(total)}


def source_parent_inventory(plans: Iterable[MediaPlan]) -> list[dict[str, Any]]:
    """Hash and describe each required raw parent before any output is written."""
    unique = {}
    for plan in plans:
        previous = unique.setdefault(plan.parent_path, plan)
        if previous.dataset != plan.dataset or previous.parent_key != plan.parent_key:
            raise VAUMediaError("parent path has ambiguous official identity")
    rows = []
    for path, plan in sorted(unique.items(), key=lambda item: (item[1].dataset, item[1].parent_key)):
        geometry = _probe(path)
        rows.append({"dataset": plan.dataset, "parent_key": plan.parent_key, "parent_path": str(path),
                     "parent_sha256": sha256_file(path), "bytes": path.stat().st_size,
                     "fps": geometry.fps, "frame_count": geometry.frame_count,
                     "height": geometry.height, "width": geometry.width})
    if len(rows) != 150:
        raise VAUMediaError(f"official VAU parent denominator is {len(rows)}, expected 150")
    return rows


def plan_document(plans: list[MediaPlan], *, identity_path: Path, ucf_ranges: Path, xd_ranges: Path) -> dict[str, Any]:
    """Canonical label-free plan binding used by the materializer admission gate."""
    summary = estimate(plans)
    summary.update({"identity_path": str(identity_path.resolve()), "identity_sha256": sha256_file(identity_path),
                    "range_sources": {"ucf": str(ucf_ranges.resolve()), "xd": str(xd_ranges.resolve())},
                    "raw_frame_capacity": raw_frame_capacity(plans)})
    return {"schema": "nc_rted_vau_materialization_plan/v1", "summary": summary,
            "parents": source_parent_inventory(plans),
            "media": [{"dataset": item.dataset, "video": item.video, "parent_key": item.parent_key,
                       "parent_path": str(item.parent_path), "range_s": None if item.range_s is None else list(item.range_s),
                       "output_path": None if item.output_path is None else str(item.output_path)} for item in plans]}


def validate_plan(path: Path, expected_sha256: str, expected: dict[str, Any], *, project_root: Path,
                  output_root: Path, max_artifact_bytes: int) -> None:
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256) or sha256_file(path) != expected_sha256:
        raise VAUMediaError("immutable plan SHA-256 differs")
    try:
        actual = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise VAUMediaError("immutable plan is unreadable") from error
    if actual != expected:
        raise VAUMediaError("immutable plan differs from rebuilt official identity/range/source binding")
    project_root = project_root.resolve()
    if project_root == Path(project_root.anchor):
        raise VAUMediaError("project root must be an approved project directory")
    for candidate, description in ((path, "plan"), (output_root, "output root")):
        try: candidate.resolve().relative_to(project_root)
        except ValueError as error: raise VAUMediaError(f"{description} escapes project root") from error
    for item in actual.get("media", []):
        output = item.get("output_path") if isinstance(item, dict) else None
        if output is None: continue
        try: Path(output).resolve().relative_to(output_root.resolve())
        except ValueError as error: raise VAUMediaError("planned child escapes output root") from error
    if max_artifact_bytes <= 0 or max_artifact_bytes > MAX_ARTIFACT_BYTES:
        raise VAUMediaError("artifact cap must be positive and at most 50 GiB")


def _parent_bindings(rows: Iterable[dict[str, Any]]) -> dict[Path, tuple[str, Geometry]]:
    bindings = {}
    for row in rows:
        if not isinstance(row, dict): raise VAUMediaError("parent binding is invalid")
        try:
            path = Path(row["parent_path"]).resolve()
            digest = row["parent_sha256"]
            geometry = Geometry(float(row["fps"]), int(row["frame_count"]), int(row["height"]), int(row["width"]))
        except (KeyError, TypeError, ValueError) as error:
            raise VAUMediaError("parent binding is incomplete") from error
        if not re.fullmatch(r"[0-9a-f]{64}", digest) or path in bindings:
            raise VAUMediaError("parent binding is ambiguous")
        geometry.validate(); bindings[path] = (digest, geometry)
    return bindings


def _live_parent_bindings(plans: Iterable[MediaPlan]) -> list[dict[str, Any]]:
    """Compatibility binding for direct library callers; formal CLI passes the plan rows."""
    unique = {}
    for plan in plans:
        unique.setdefault(plan.parent_path.resolve(), plan)
    rows = []
    for path, plan in unique.items():
        geometry = _probe(path)
        rows.append({"parent_path": str(path), "parent_sha256": sha256_file(path),
                     "fps": geometry.fps, "frame_count": geometry.frame_count,
                     "height": geometry.height, "width": geometry.width})
    return rows


def _source_state(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def materialize(plans: list[MediaPlan], *, output_root: Path, hard_guard_bytes: int = MIN_FREE_BYTES,
                max_artifact_bytes: int = MAX_ARTIFACT_BYTES,
                parent_bindings: Iterable[dict[str, Any]] | None = None) -> tuple[Path, Path]:
    """Materialize all plans and atomically publish catalog and id bindings."""
    output_root = output_root.resolve()
    if max_artifact_bytes <= 0 or max_artifact_bytes > MAX_ARTIFACT_BYTES:
        raise VAUMediaError("artifact cap must be positive and at most 50 GiB")
    bindings = _parent_bindings(parent_bindings if parent_bindings is not None else _live_parent_bindings(plans))
    for plan in plans:
        if plan.derived:
            assert plan.output_path is not None
            try: plan.output_path.resolve().relative_to(output_root)
            except ValueError as error: raise VAUMediaError("planned child escapes output root") from error
    journal = output_root / "journal.jsonl"
    with output_lock(output_root):
        _require_space(output_root, hard_guard_bytes)
        completed = _completed(journal)
        rows = []
        parent_hashes: dict[Path, str] = {}
        parent_geometry: dict[Path, Geometry] = {}
        parent_states: dict[Path, tuple[int, int, int, int]] = {}
        for plan in plans:
            parent_path = plan.parent_path.resolve()
            expected_parent = bindings.get(parent_path)
            if expected_parent is None:
                raise VAUMediaError("plan parent is absent from immutable parent bindings")
            parent_sha = parent_hashes.get(parent_path)
            if parent_sha is None:
                state = _source_state(parent_path)
                parent_sha = sha256_file(parent_path)
                if _source_state(parent_path) != state or parent_sha != expected_parent[0]:
                    raise VAUMediaError("parent changed from immutable admitted binding")
                parent_hashes[parent_path] = parent_sha
                parent_states[parent_path] = state
            source_geometry = parent_geometry.get(parent_path)
            if source_geometry is None:
                state = _source_state(parent_path)
                source_geometry = _probe(parent_path)
                if _source_state(parent_path) != state or source_geometry != expected_parent[1]:
                    raise VAUMediaError("parent geometry changed from immutable admitted binding")
                parent_geometry[parent_path] = source_geometry
            prior = completed.get(plan.video)
            if prior:
                rows.append(_validate_record(plan, prior, parent_sha, source_geometry))
                continue
            _record(journal, {"status": "started", "video": plan.video, "parent_path": str(plan.parent_path),
                              "parent_sha256": parent_sha, "range_s": None if plan.range_s is None else list(plan.range_s)})
            try:
                if plan.derived:
                    _require_space(output_root, hard_guard_bytes)
                    assert plan.output_path is not None
                    child_reservation = _child_raw_rgb_bytes(plan, source_geometry)
                    if _artifact_bytes(output_root) + child_reservation > max_artifact_bytes:
                        raise VAUMediaError("artifact cap would be exceeded by worst-case in-flight child")
                    _require_space(output_root, hard_guard_bytes + child_reservation)
                    plan.output_path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = plan.output_path.with_name("." + plan.output_path.stem + ".partial.mp4")
                    if temporary.exists():
                        raise VAUMediaError("interrupted child is preserved for inspection")
                    if plan.output_path.exists():
                        raise VAUMediaError("unrecorded child is preserved for inspection")
                    try:
                        import decord
                    except ImportError as error:
                        raise VAUMediaError("decord is required for materialization") from error
                    reader = decord.VideoReader(str(plan.parent_path))
                    try:
                        source_state = _source_state(parent_path)
                        def before_write(frame_count=0):
                            if _source_state(parent_path) != source_state:
                                raise VAUMediaError("parent changed during child decoding")
                            # `_artifact_bytes` includes the in-progress partial below
                            # media/, so add only the next decoded frame batch.
                            next_raw_bytes = frame_count * source_geometry.height * source_geometry.width * 3
                            if _artifact_bytes(output_root) + next_raw_bytes > max_artifact_bytes:
                                raise VAUMediaError("artifact cap reached during child writing")
                        _split_official_chunked(reader, plan.range_s, temporary,
                                                before_batch=lambda count: (_require_space(output_root, hard_guard_bytes), before_write(count)))
                    finally:
                        del reader
                    if _source_state(parent_path) != source_state or sha256_file(parent_path) != parent_sha:
                        raise VAUMediaError("parent changed during child decoding")
                    with temporary.open("rb") as handle:
                        os.fsync(handle.fileno())
                    os.replace(temporary, plan.output_path)
                    _fsync_dir(plan.output_path.parent)
                    if _artifact_bytes(output_root) > max_artifact_bytes:
                        raise VAUMediaError("artifact cap exceeded after child commit")
                    path = plan.output_path
                else:
                    path = plan.parent_path
                digest = sha256_file(path)
                if not plan.derived:
                    if (_source_state(parent_path) != parent_states[parent_path] or digest != parent_sha or
                            _probe(parent_path) != source_geometry):
                        raise VAUMediaError("parent changed from immutable admitted binding")
                geometry = _verify_child_decode(path, _expected_child_geometry(plan, source_geometry)) if plan.derived else source_geometry
                row = _row(plan, path, digest, geometry)
                _record(journal, {"status": "complete", "video": plan.video, "parent_path": str(plan.parent_path),
                                  "parent_sha256": parent_sha, "range_s": None if plan.range_s is None else list(plan.range_s),
                                  "media_path": row["media_path"], "media_sha256": digest,
                                  "geometry": {key: row[key] for key in ("fps", "frame_count", "height", "width")}})
                rows.append(row)
            except BaseException as error:
                _record(journal, {"status": "failed", "video": plan.video, "parent_path": str(plan.parent_path),
                                  "parent_sha256": parent_sha, "range_s": None if plan.range_s is None else list(plan.range_s),
                                  "error": f"{type(error).__name__}: {error}"})
                raise
        if len(rows) != 1369 or len({(row["dataset"], row["media_key"]) for row in rows}) != 1369:
            raise VAUMediaError("catalog denominator or media identity uniqueness failed")
        catalog = output_root / "vau_media_catalog.json"
        document = {"schema": CATALOG_SCHEMA, "media": sorted(rows, key=lambda row: (row["dataset"], row["media_key"]))}
        if catalog.exists():
            try:
                existing = json.loads(catalog.read_text())
            except ValueError as error:
                raise VAUMediaError("existing immutable catalog is invalid") from error
            if existing != document:
                raise VAUMediaError("existing immutable catalog differs from recovered journal")
        else:
            _atomic_json_new(catalog, document)
        return catalog, output_root / "journal.jsonl"
