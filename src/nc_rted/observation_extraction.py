"""Resumable training-only observation preparation; no Slow, answers or test labels."""
from __future__ import annotations
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from .storage_lock import allocation_lock, ensure_directory, open_lock_file
import tempfile

from .media_observer import BoundMedia
from .task_inputs import detection_window_id
from .teacher_records import TeacherSourceTruth, feature_assembly_to_teacher_window
from .teacher_store import (STORE_SCHEMA, RESERVED_FREE_BYTES, load_teacher_store,
                            write_teacher_store, _load_index)

class ExtractionError(ValueError):
    pass


def canonical_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def atomic_json(path: Path, value: dict, *, reserved_free_bytes=RESERVED_FREE_BYTES):
    content=canonical_bytes(value)
    with allocation_lock(path.parent):
        block=max(4096,os.statvfs(path.parent).f_frsize)
        allocation=((len(content)+block-1)//block)*block+2*block
        if shutil.disk_usage(path.parent).free < reserved_free_bytes+allocation:
            raise ExtractionError('disk hard limit: metadata publication preserves free-space reserve')
        descriptor, temporary = tempfile.mkstemp(prefix=".publish-", dir=path.parent)
        try:
            with os.fdopen(descriptor,"wb") as handle:
                handle.write(content); handle.flush(); os.fsync(handle.fileno())
            os.replace(temporary,path)
            descriptor = os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
            try: os.fsync(descriptor)
            finally: os.close(descriptor)
        finally:
            Path(temporary).unlink(missing_ok=True)

def bound_bytes(binding: dict):
    if not isinstance(binding,dict) or set(binding) != {"path","sha256"}:
        raise ExtractionError("file binding requires exact path and SHA256")
    path=Path(binding["path"])
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != binding["sha256"]:
        raise ExtractionError("bound preparation input changed: " + str(path))
    return raw


def bound_json(binding: dict):
    return json.loads(bound_bytes(binding))


def load_detection_inputs(manifest_directory: Path, provenance_sha256: str,
                          media_binding: dict, pts_binding: dict):
    root=Path(manifest_directory)
    provenance=bound_json({"path":str(root/"provenance.json"),"sha256":provenance_sha256})
    if provenance.get("schema") != "nc_rted_manifest_provenance/v1" or provenance.get("test_answers_read") is not False:
        raise ExtractionError("unsupported or test-contaminated manifest provenance")
    # Deliberately read only training split/prefix outputs. Other provenance
    # inputs include test identity preparation and are never opened here.
    selected={name:bound_json({"path":str(root/name),"sha256":provenance["outputs"][name]})
              for name in ("source_splits.json","train8000_detection_prefixes.json")}
    splits={}
    for row in selected["source_splits.json"]:
        key=(row["dataset"],row["key"])
        if key in splits: raise ExtractionError("duplicate source split identity")
        splits[key]=row
    metadata=bound_json(media_binding)
    catalog={}
    fields=("dataset","media_key","media_path","media_sha256","fps","frame_count","height","width")
    for row in metadata["media"]:
        item=BoundMedia(**{name:row[name] for name in fields});item.validate()
        key=(item.dataset,item.media_key)
        if key in catalog: raise ExtractionError("duplicate observation media identity")
        catalog[key]=item
    pts=bound_json(pts_binding)
    if (pts.get("status") != "ALL_MEDIA_CFR_RELATIVE_PTS_MATCH" or pts.get("failures") != []
            or pts.get("input_sha256") != media_binding["sha256"]
            or pts.get("media_count") != len(catalog) or pts.get("passed") != len(catalog)):
        raise ExtractionError("all-media PTS evidence does not bind observation metadata")
    timing={}
    timing_bytes=bound_bytes({"path":pts["rows_path"],"sha256":pts.get("rows_sha256")})
    for line in timing_bytes.splitlines():
        row=json.loads(line);key=(row["dataset"],row["media_key"])
        if key in timing or key not in catalog: raise ExtractionError("duplicate or foreign PTS media")
        media=catalog[key]
        if (row.get("status") != "CFR_RELATIVE_PTS_MATCH" or row.get("media_sha256") != media.media_sha256
                or row.get("decoded_frames") != media.frame_count or row.get("nominal_frame_count") != media.frame_count
                or row.get("nonmonotonic") is not False or row.get("fps") != media.fps
                or not isinstance(row.get("max_relative_grid_error_seconds"),(int,float))
                or not math.isfinite(row["max_relative_grid_error_seconds"])
                or not 0 <= row["max_relative_grid_error_seconds"] <= 1e-6):
            raise ExtractionError("unaccepted individual media timing")
        timing[key]=row
    if set(timing)!=set(catalog): raise ExtractionError("incomplete all-media timing rows")
    truths=[];balance={};used=set();ids=set()
    for row in selected["train8000_detection_prefixes.json"]:
        key=(row["dataset"],row["key"])
        if key not in splits or key not in catalog: raise ExtractionError("selected training source missing")
        truth=TeacherSourceTruth.from_manifest_rows(row,splits[key])
        if not 0 < truth.observed_seconds <= catalog[key].duration_s:
            raise ExtractionError("selected prefix outside media duration")
        if row.get("class") not in {"normal","anomalous"}: raise ExtractionError("illegal training label")
        identity=detection_window_id(row)
        if identity in ids: raise ExtractionError("duplicate selected detection prefix")
        ids.add(identity);truths.append(truth);used.add(key)
        cell=(truth.dataset,row["class"]);balance[cell]=balance.get(cell,0)+1
    if len(truths)!=6000 or balance!={(dataset,label):1500 for dataset in ("ucf-crime","xd-violence") for label in ("normal","anomalous")}:
        raise ExtractionError("fixed balanced 6000 detection denominator differs")
    if used!=set(catalog): raise ExtractionError("media catalog is not exactly the selected training media")
    truths.sort(key=lambda item:(item.dataset,item.key,item.observed_seconds,item.query_index))
    return tuple(truths),catalog


@contextmanager
def observation_run_admission(root, *, reserved_free_bytes=RESERVED_FREE_BYTES):
    """Exclude duplicate output writers before either starts loading models."""
    root = Path(root)
    ensure_directory(root, reserved_free_bytes)
    with open_lock_file(root / ".writer.lock", reserved_free_bytes) as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ExtractionError("observation writer already active") from error
        yield (root.resolve(), os.getpid(), handle)


class ObservationJournal:
    """Each window is an immutable committed store; root commit seals exact IDs.

    A terminated writer can leave uncommitted staging directories. They are
    preserved and never counted. A resumed run revalidates existing windows.
    No full tensor store is copied when the final index is published.
    """
    def __init__(self, root: Path, binding: dict, window_ids: tuple[str,...],
                 *, reserved_free_bytes=RESERVED_FREE_BYTES):
        self.root=Path(root);self.binding=binding;self.ids=tuple(sorted(window_ids))
        if not self.ids or len(set(self.ids))!=len(self.ids) or any(not isinstance(x,str) or not x for x in self.ids):
            raise ExtractionError("journal requires unique nonempty window IDs")
        if reserved_free_bytes<0: raise ExtractionError("negative free-space reserve")
        self.reserve=reserved_free_bytes
        self._writer_pid=None

    def _run_bytes(self):
        return canonical_bytes({"schema":"nc_rted_observation_journal/v1",
                                "binding":self.binding,"window_ids":self.ids})

    def _require_writer(self):
        if self._writer_pid != os.getpid():
            raise ExtractionError("observation mutation requires a validated writer context")
        path=self.root/"run.json"
        if not path.is_file() or path.read_bytes()!=self._run_bytes():
            raise ExtractionError("observation run binding changed during writing")

    @contextmanager
    def writer(self, *, admission=None):
        if admission is None:
            with observation_run_admission(self.root, reserved_free_bytes=self.reserve) as owned:
                with self.writer(admission=owned):
                    yield self
            return
        root, pid, handle = admission
        if root != self.root.resolve() or pid != os.getpid() or handle.closed:
            raise ExtractionError("invalid observation writer admission")
        expected={"schema":"nc_rted_observation_journal/v1","binding":self.binding,"window_ids":self.ids}
        path=self.root/"run.json"
        if path.exists():
            if path.read_bytes()!=canonical_bytes(expected):raise ExtractionError("observation run binding or selected identities changed")
        else:
            if any(item.name != ".writer.lock" for item in self.root.iterdir()):
                raise ExtractionError("orphaned observation artifacts have no run binding")
            atomic_json(path,expected,reserved_free_bytes=self.reserve)
        ensure_directory(self.root/"windows", self.reserve)
        self._writer_pid=os.getpid()
        try:yield self
        finally:self._writer_pid=None

    def _reserve(self,additional=0):
        if shutil.disk_usage(self.root).free < self.reserve+additional:
            raise ExtractionError("disk hard limit: observation preparation preserves free-space reserve")

    def _window_path(self,window_id):
        if window_id not in self.ids:raise ExtractionError("window outside fixed selection")
        return self.root/"windows"/hashlib.sha256(window_id.encode()).hexdigest()

    def read(self,window_id):
        path=self._window_path(window_id)
        if not path.exists():return None
        values=load_teacher_store(path)
        if len(values)!=1 or values[0].record.get("window_id")!=window_id:
            raise ExtractionError("committed observation identity differs")
        self._validate_rejection(values[0])
        return values[0]

    @staticmethod
    def _validate_rejection(compact):
        if compact.rejection is not None and compact.record.get("rejection") not in {
                "no relation pairs in assembled observation", "missing reliable assembled background"}:
            raise ExtractionError("technical observation failure: "+compact.record["window_id"])

    def put(self,compact):
        self._require_writer()
        self._validate_rejection(compact)
        identity=compact.record["window_id"]
        if self.read(identity) is not None:raise ExtractionError("observation already committed")
        # Maximum uncompressed 16x4x3473 float64 process cells plus static arrays,
        # zip framing and metadata fit below 3MiB. This prevents writing across
        # the floor before the underlying store's post-write reserve check.
        with allocation_lock(self.root):
            self._reserve(3<<20)
            write_teacher_store(self._window_path(identity),(compact,),reserved_free_bytes=self.reserve)

    def finish(self):
        self._require_writer()
        entries=[]
        for identity in self.ids:
            value=self.read(identity)
            if value is None:raise ExtractionError("cannot seal an incomplete observation set")
            window_path=self._window_path(identity)
            entry=dict(_load_index(window_path)["entries"][0])
            if entry["kind"]=="record":entry["path"]=str(window_path.relative_to(self.root)/entry["path"])
            entries.append(entry)
        index={"schema":STORE_SCHEMA,"entries":entries,"observation_binding":self.binding,
               "selected_window_ids_sha256":hashlib.sha256(canonical_bytes(self.ids)).hexdigest()}
        commit={"schema":STORE_SCHEMA,"index_sha256":hashlib.sha256(canonical_bytes(index)).hexdigest()}
        required=len(canonical_bytes(index))+len(canonical_bytes(commit))+(1<<20)
        self._reserve(required)
        if (self.root/"commit.json").exists():
            if (self.root/"index.json").read_bytes()!=canonical_bytes(index) or (self.root/"commit.json").read_bytes()!=canonical_bytes(commit):
                raise ExtractionError("sealed observation store changed")
        else:
            atomic_json(self.root/"index.json",index,reserved_free_bytes=self.reserve)
            atomic_json(self.root/"commit.json",commit,reserved_free_bytes=self.reserve)
        return commit


def extract_windows(truths, observer, journal: ObservationJournal, *, max_new_windows=None, admission=None,
                    before_finish=None):
    if max_new_windows is not None and (type(max_new_windows) is not int or max_new_windows<1):
        raise ExtractionError("max_new_windows must be a positive preparation bound")
    if before_finish is not None and not callable(before_finish):
        raise ExtractionError("before_finish must be callable")
    requested=tuple(f"detection:{t.dataset}:{t.key}:{t.query_index}" for t in truths)
    if tuple(sorted(requested)) != journal.ids:
        raise ExtractionError("extraction truth set differs from journal denominator")
    computed=0;reused=0;rejections={}
    with journal.writer(admission=admission):
        for truth in truths:
            window_id=f"detection:{truth.dataset}:{truth.key}:{truth.query_index}"
            value=journal.read(window_id)
            if value is not None:reused+=1
            else:
                if max_new_windows is not None and computed>=max_new_windows:break
                # Observer receives only media identity/time, never normal truth.
                observed=observer.detection(truth.dataset,truth.key,truth.observed_seconds)
                value=feature_assembly_to_teacher_window(observed.features,truth,observed.relation_class_pairs)
                # Journal insertion, reuse and sealing share the same rejection
                # policy; technical failures cannot enter a completed store.
                journal.put(value);computed+=1
            if value.rejection:
                reason=value.record["rejection"];rejections[reason]=rejections.get(reason,0)+1
            atomic_json(journal.root/"progress.json",{"computed_this_pass":computed,"reused_this_pass":reused,
                         "completed_this_pass":computed+reused,"total":len(journal.ids),"last_window":window_id},
                         reserved_free_bytes=journal.reserve)
        complete=computed+reused==len(journal.ids)
        if complete:
            if before_finish is not None:before_finish()
            journal.finish()
        result={"status":"COMPLETE_OBSERVATIONS_NOT_TEACHERS" if complete else "PARTIAL_RESUMABLE_OBSERVATIONS",
                "computed":computed,"reused":reused,"total":len(journal.ids),"rejections_this_pass":rejections}
        atomic_json(journal.root/"last_pass.json",result,reserved_free_bytes=journal.reserve)
        return result
