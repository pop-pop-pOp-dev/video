import hashlib
import importlib.util
import inspect
import json
import os
import time
import errno
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from nc_rted.numerics import configure_deterministic_algorithms, deterministic_policy
from nc_rted.prediction_worker import count_slow_execution


def _probe():
    path = Path(__file__).resolve().parents[1] / "scripts/nc_rted_prediction_runtime_probe.py"
    spec = importlib.util.spec_from_file_location("probe_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, allow_nan=True), encoding="utf-8")
    return _sha(path)


def _tree_sha(path):
    digest = hashlib.sha256()
    for child in sorted(Path(path).rglob("*")):
        if child.is_file():
            digest.update(str(child.relative_to(path)).encode("utf-8"))
            digest.update(b"\0")
            digest.update(_sha(child).encode("ascii"))
            digest.update(b"\n")
    return digest.hexdigest()


def _admission(probe, path, output, **overrides):
    document = {
        "schema": probe.ADMISSION_SCHEMA, "status": "PASS", "kind": "vad", "device": "cuda:0",
        "data_volume": str(probe.PROJECT_VOLUME.resolve(strict=True)), "min_free_bytes": probe.RESERVE,
        "run_budget_seconds": 10, "deadline_utc_epoch": time.time() + 60, "output": str(output),
        "preflight_report_sha256": "p" * 64,
    }
    document.update(overrides)
    digest = _write_json(path, document)
    return SimpleNamespace(diagnostic_admission=str(path), diagnostic_admission_sha256=digest,
                           kind=document["kind"], device=document["device"], output=str(output),
                           preflight_report_sha256=document["preflight_report_sha256"])


def _native_sleep_child(_args, _context, connection):
    os.setsid()
    connection.send({"child_pid": os.getpid(), "stage": "native_sleep"})
    time.sleep(30)


class _RawSlow(torch.nn.Module):
    def forward(self, values, scale=1):
        return values * scale


class _DirectWrapper(torch.nn.Module):
    def __init__(self, raw):
        super().__init__()
        self.raw = raw

    def forward(self, values, scale=1):
        return self.raw.forward(values, scale=scale)


class _GenerationLoop:
    def __init__(self, raw, *, fail=False):
        self.raw, self.fail = raw, fail

    def generate(self):
        self.raw.forward(torch.tensor([1.0]))
        if self.fail:
            raise RuntimeError("generation route failed after its raw forward")
        return {"text": "ok", "token_ids": [4, 5]}


def _loaded_model(*, generation_fails=False):
    raw = _RawSlow()
    return SimpleNamespace(group="R0", seed=None, evidence_enabled=False,
                           bridge=SimpleNamespace(raw_slow=raw, slow=_DirectWrapper(raw)),
                           hivau_inference=_GenerationLoop(raw, fail=generation_fails))


def _cuda(monkeypatch, *, allocated=11, reserved=22):
    calls = []
    state = {"selected": False}

    def set_device(device):
        state["selected"] = True
        calls.append(("set_device", device))

    def reset(device):
        assert state["selected"], "peak reset must follow CUDA device selection"
        calls.append(("reset", device))

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: state["selected"])
    monkeypatch.setattr(torch.cuda, "set_device", set_device)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", reset)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: calls.append(("sync", device)))
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: calls.append(("allocated", device)) or allocated)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda device: calls.append(("reserved", device)) or reserved)
    return calls


def _install_success_environment(monkeypatch, tmp_path, kind):
    probe = _probe()
    probe._api()
    media = tmp_path / "media.bin"
    media.write_bytes(b"probe-media")
    media_sha = _sha(media)
    source = tmp_path / "source.json"
    source.write_text("{}", encoding="utf-8")
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer.json").write_text("{}", encoding="utf-8")
    query_rows = [
        {"index": 0, "frame_indices": [0, 2], "fast_score": 0.4},
        {"index": 1, "frame_indices": [4, 6], "fast_score": 0.7},
    ]
    expected_media = {"dataset": "ucf", "media_key": "clip-1", "media_path": str(media), "media_sha256": media_sha,
                      "frame_count": 8, "fps": 8.0, "target_fps": 4, "query_interval": 2, "queries": query_rows}
    identities = {
        "vad": [{"dataset": "ucf", "id": "clip-1", "media_path": str(media), "media_sha256": media_sha}],
        "vau": [{"id": "ask-1", "media_path": str(media), "media_sha256": media_sha, "question": "What happens?"}],
    }
    document = {
        "inherited": {"source_manifest": str(source), "source_manifest_sha256": "a" * 64,
                      "tokenizer": str(tokenizer), "tokenizer_sha256": _tree_sha(tokenizer)},
        "fast": {"snapshot": str(tmp_path / "fast.json"), "snapshot_sha256": "c" * 64},
        "protocols": {"hivau": {"max_new_tokens": 4},
                      "vad_config": {"target_fps": 4, "query_interval": 2, "batch_size": 1, "fusion": "fixed"},
                      "vad": {"ucf": {"trigger_threshold": 0.5}, "xd": {"trigger_threshold": 0.5}}},
    }
    bindings = {
        "prediction_source": {"path": str(source), "sha256": "a" * 64},
        "implementation": {"path": "implementation", "sha256": "d" * 64},
        "embedded_vision": {"path": "vision", "sha256": "e" * 64},
        "decoder": {"path": "decoder", "sha256": "f" * 64},
    }
    preflight = {"schema": probe.PREFLIGHT_SCHEMA, "status": "PREFLIGHT_PASS_FORMAL_ADMISSION_REQUIRED", "gpu_launched": False,
                 "runtime": {"path": "runtime", "sha256": "1" * 64},
                 "identity_manifest": {"path": "identities", "sha256": "2" * 64},
                 "r0_model_manifest": {"path": "model", "sha256": "3" * 64}, "bindings": bindings}
    docs = {
        "preflight": (tmp_path / "preflight.json", preflight), "runtime": (tmp_path / "runtime.json", document),
        str(source): (source, {}), "identities": (tmp_path / "identities.json", identities),
        "model": (tmp_path / "model.json", {}),
        str(tmp_path / "fast.json"): (tmp_path / "fast.json", {"schema": "nc_rted_blind_fast_snapshot/v1", "media": [expected_media]}),
        "implementation": (tmp_path / "implementation.json", {}), "vision": (tmp_path / "vision.json", {}),
        "decoder": (tmp_path / "decoder.json", {}),
    }
    monkeypatch.setattr(probe, "_bound", lambda path, digest, code: docs[path])
    monkeypatch.setattr(probe, "verify_implementation_manifest", lambda *args: None)
    model = _loaded_model()
    monkeypatch.setattr(probe, "load_model_artifact", lambda *args, **kwargs: model)
    def vad(_request, loaded, **_kwargs):
        return {"queries": [
            {"query_index": 0, "frame_indices": [0, 2], "fast_score": 0.4, "final_score": 0.4, "slow_score": None, "fused_score": None, "triggered": False},
            {"query_index": 1, "frame_indices": [4, 6], "fast_score": 0.7, "final_score": 0.8, "slow_score": float(loaded.bridge.raw_slow.forward(torch.tensor([1.0])).item()), "fused_score": 0.8, "triggered": True},
        ], "causal_smoothed_scores": [0.4] * 8, "total_frames": 8, "sample_interval": 2}
    monkeypatch.setattr(probe, "default_factory", lambda *args, **kwargs: {
        "loader": SimpleNamespace(load=lambda artifact: model), "vad": SimpleNamespace(predict=vad),
        "vau": SimpleNamespace(generate=lambda request, loaded, **kwargs: loaded.hivau_inference.generate()),
    })
    args = SimpleNamespace(preflight_report="preflight", preflight_report_sha256="p" * 64, kind=kind,
                           identity="vad:ucf:clip-1" if kind == "vad" else "vau:ask-1", device="cuda:3")
    context = {"status": "RUNNING", "stage": "admitted", "started_at_utc_epoch": time.time(),
               "deadline_utc_epoch": time.time() + 30, "device": "cuda:3",
               "peak_cuda_allocated_bytes": None, "peak_cuda_reserved_bytes": None}
    return probe, args, context, docs


def test_supervisor_kills_a_separate_blocked_native_child():
    probe = _probe()
    started = time.time()
    result = probe._supervise(SimpleNamespace(), {"started_at_utc_epoch": started, "deadline_utc_epoch": started + 0.15}, target=_native_sleep_child)
    assert result["status"] == "INCOMPLETE_RUNTIME_PROBE"
    assert result["error_code"] == "DEADLINE_EXCEEDED"
    assert result["child_pid"] != os.getpid()
    with pytest.raises(ProcessLookupError):
        os.kill(result["child_pid"], 0)


def test_supervisor_deadline_applies_during_bootstrap_before_cuda(monkeypatch, tmp_path):
    probe = _probe()
    marker = tmp_path / "cuda-was-reached"
    monkeypatch.setattr(probe, "_bootstrap", lambda *args: time.sleep(2))
    monkeypatch.setattr(probe, "_api", lambda: None)
    monkeypatch.setattr(probe, "run", lambda *args: marker.write_text("unexpected", encoding="utf-8"))
    started = time.time()
    result = probe._supervise(SimpleNamespace(preflight_report="irrelevant", preflight_report_sha256="x" * 64), {
        "status": "RUNNING", "stage": "admitted", "started_at_utc_epoch": started,
        "deadline_utc_epoch": started + 0.15, "device": "cuda:0"})
    assert result["error_code"] == "DEADLINE_EXCEEDED"
    assert not marker.exists()


def test_admission_requires_finite_future_deadline(tmp_path):
    probe = _probe()
    output = tmp_path / "report.json"
    args = _admission(probe, tmp_path / "admission.json", output, deadline_utc_epoch=float("inf"))
    with pytest.raises(probe.ProbeError, match="ADMISSION_INVALID"):
        probe._admit(args)
    args = _admission(probe, tmp_path / "expired.json", output, deadline_utc_epoch=time.time() - 1)
    with pytest.raises(probe.ProbeError, match="DEADLINE_EXCEEDED"):
        probe._admit(args)
    assert not output.exists()


def test_admission_rejects_oversized_document_before_path_handling(monkeypatch, tmp_path):
    probe = _probe()
    output = tmp_path / ("x" * 70_000)
    args = _admission(probe, tmp_path / "oversized-admission.json", output)
    monkeypatch.setattr(probe, "_canonical_output", lambda value: pytest.fail("oversized admission reached output handling"))
    with pytest.raises(probe.ProbeError, match="ADMISSION_INVALID"):
        probe._admit(args)


def test_canonical_output_rejects_aliases_and_parent_escapes_without_writes(tmp_path):
    probe = _probe()
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    dangling = tmp_path / "dangling"
    dangling.symlink_to(tmp_path / "missing", target_is_directory=True)
    for value in (str(tmp_path / "nested" / ".." / "report.json"), str(alias / "report.json"), str(dangling / "report.json")):
        with pytest.raises(probe.ProbeError, match="OUTPUT_UNAVAILABLE"):
            probe._canonical_output(value)
    assert not (real / "report.json").exists()
    assert not (tmp_path / "missing" / "report.json").exists()


def test_publication_reclaims_preallocated_slot_at_exact_reserve(monkeypatch, tmp_path):
    probe = _probe()
    monkeypatch.setattr(probe.shutil, "disk_usage", lambda path: SimpleNamespace(free=probe.RESERVE + probe.REPORT_CAPACITY + 1_000_000))
    output = tmp_path / "reserved" / "report.json"
    args = _admission(probe, tmp_path / "admission.json", output)
    assert not output.exists()
    admission, _deadline, slot = probe._admit(args)
    try:
        temporary, target = Path(slot["temporary"]), Path(slot["target"])
        assert target == output and not target.exists() and temporary.exists()
        assert temporary.stat().st_size == probe.REPORT_CAPACITY
        block = max(4096, os.statvfs(target.parent).f_frsize)
        monkeypatch.setattr(probe.shutil, "disk_usage", lambda path: SimpleNamespace(free=probe.RESERVE + block))
        digest = probe._publish(slot, {"status": "INCOMPLETE_RUNTIME_PROBE", "error_code": "CUDA_UNAVAILABLE"}, reserve=admission["min_free_bytes"], deadline=_deadline)
        assert _sha(output) == digest
        assert json.loads(output.read_text(encoding="utf-8"))["candidate_result"]["error_code"] == "CUDA_UNAVAILABLE"
    finally:
        if temporary.exists():
            temporary.unlink()


def test_publication_rejects_when_truncation_did_not_release_allocated_blocks(monkeypatch, tmp_path):
    probe = _probe()
    monkeypatch.setattr(probe.shutil, "disk_usage", lambda path: SimpleNamespace(free=probe.RESERVE + probe.REPORT_CAPACITY + 1_000_000))
    admission, deadline, slot = probe._admit(_admission(probe, tmp_path / "admission.json", tmp_path / "report.json"))
    original_fstat = probe.os.fstat
    calls = []

    def retained_blocks(descriptor):
        actual = original_fstat(descriptor)
        calls.append(actual)
        return SimpleNamespace(st_dev=actual.st_dev, st_ino=actual.st_ino, st_size=actual.st_size,
                               st_blocks=100 + len(calls))

    monkeypatch.setattr(probe.os, "fstat", retained_blocks)
    monkeypatch.setattr(probe.shutil, "disk_usage", lambda path: SimpleNamespace(free=probe.RESERVE + 4096))
    try:
        with pytest.raises(probe.ProbeError, match="OUTPUT_UNAVAILABLE"):
            probe._publish(slot, {"status": "INCOMPLETE_RUNTIME_PROBE"}, reserve=admission["min_free_bytes"], deadline=deadline)
        assert not Path(slot["target"]).exists()
    finally:
        temporary = Path(slot["temporary"])
        if temporary.exists():
            temporary.unlink()


def test_allocation_lock_contention_expires_before_admission_can_proceed(monkeypatch, tmp_path):
    probe = _probe()
    operations = []

    def contended_lockf(_descriptor, operation):
        operations.append(operation)
        if operation & probe.fcntl.LOCK_NB:
            raise OSError(errno.EAGAIN, "allocation lock is busy")

    monkeypatch.setattr(probe.fcntl, "lockf", contended_lockf)
    deadline = time.time() + 0.02
    with pytest.raises(probe.ProbeError, match="DEADLINE_EXCEEDED"):
        with probe._allocation(tmp_path, deadline=deadline):
            pytest.fail("contended allocation acquired a lock")
    assert operations and all(operation & probe.fcntl.LOCK_NB for operation in operations)


def test_publication_lock_contention_expires_without_creating_the_report(monkeypatch, tmp_path):
    probe = _probe()
    monkeypatch.setattr(probe.shutil, "disk_usage", lambda path: SimpleNamespace(free=probe.RESERVE + probe.REPORT_CAPACITY + 1_000_000))
    output = tmp_path / "reserved" / "report.json"
    admission, _deadline, slot = probe._admit(_admission(probe, tmp_path / "admission.json", output))
    temporary, target = Path(slot["temporary"]), Path(slot["target"])

    def contended_lockf(_descriptor, operation):
        if operation & probe.fcntl.LOCK_NB:
            raise OSError(errno.EAGAIN, "allocation lock is busy")

    monkeypatch.setattr(probe.fcntl, "lockf", contended_lockf)
    try:
        with pytest.raises(probe.ProbeError, match="DEADLINE_EXCEEDED"):
            probe._publish(slot, {"status": "INCOMPLETE_RUNTIME_PROBE"}, reserve=admission["min_free_bytes"], deadline=time.time() + 0.02)
        assert not target.exists()
        assert temporary.exists()
    finally:
        if temporary.exists():
            temporary.unlink()


def test_request_requires_exact_routes_identity_and_non_supervisory_keys(tmp_path):
    probe = _probe()
    probe._api()
    media = tmp_path / "media.bin"
    media.write_bytes(b"request-media")
    digest = _sha(media)
    identities = {
        "vad": [{"dataset": "ucf", "id": "clip", "media_path": str(media), "media_sha256": digest}],
        "vau": [{"id": "prompt", "media_path": str(media), "media_sha256": digest, "question": "Describe it."}],
    }
    assert probe._request(identities, "vad:ucf:clip", "vad").identity == "vad:ucf:clip"
    assert probe._request(identities, "vau:prompt", "vau").identity == "vau:prompt"
    forbidden = json.loads(json.dumps(identities))
    forbidden["vad"][0]["label"] = "forbidden"
    with pytest.raises(probe.ProbeError, match="IDENTITY_INVALID"):
        probe._request(forbidden, "vad:ucf:clip", "vad")
    with pytest.raises(probe.ProbeError, match="IDENTITY_INVALID"):
        probe._request(identities, "vau:prompt", "vad")


def test_preflight_rejects_reference_metadata_before_reading_the_reference(monkeypatch, tmp_path):
    probe = _probe()
    probe._api()
    preflight = {
        "schema": probe.PREFLIGHT_SCHEMA, "status": "PREFLIGHT_PASS_FORMAL_ADMISSION_REQUIRED", "gpu_launched": False,
        "runtime": {"path": "runtime", "sha256": "r" * 64, "untrusted": "extra"},
        "identity_manifest": {"path": "identities", "sha256": "i" * 64},
        "r0_model_manifest": {"path": "model", "sha256": "m" * 64},
        "bindings": {"prediction_source": {"path": "source", "sha256": "s" * 64}},
    }
    calls = []
    monkeypatch.setattr(probe, "_bound", lambda path, digest, code: calls.append(path) or (tmp_path / path, preflight))
    args = SimpleNamespace(preflight_report="preflight", preflight_report_sha256="p" * 64, kind="vad", identity="vad:ucf:clip", device="cuda:0")
    context = {"started_at_utc_epoch": time.time(), "deadline_utc_epoch": time.time() + 30}
    with pytest.raises(probe.ProbeError, match="PREFLIGHT_INVALID"):
        probe.run(args, context)
    assert calls == ["preflight"]
    assert "runtime" not in context


def test_partial_preflight_failure_retains_only_prior_verified_bindings(monkeypatch, tmp_path):
    probe = _probe()
    probe._api()
    runtime = tmp_path / "runtime.json"
    source = tmp_path / "source.json"
    preflight = {
        "schema": probe.PREFLIGHT_SCHEMA, "status": "PREFLIGHT_PASS_FORMAL_ADMISSION_REQUIRED", "gpu_launched": False,
        "runtime": {"path": "runtime", "sha256": "r" * 64},
        "identity_manifest": {"path": "identities", "sha256": "i" * 64},
        "r0_model_manifest": {"path": "model", "sha256": "m" * 64},
        "bindings": {"prediction_source": {"path": "source", "sha256": "s" * 64}},
    }
    document = {"inherited": {"source_manifest": str(source), "source_manifest_sha256": "s" * 64}}
    def bound(path, digest, code):
        if path == "preflight":
            return tmp_path / "preflight.json", preflight
        if path == "runtime":
            return runtime, document
        assert path == "source"
        raise probe.ProbeError("SOURCE_MISMATCH")
    monkeypatch.setattr(probe, "_bound", bound)
    args = SimpleNamespace(preflight_report="preflight", preflight_report_sha256="p" * 64, kind="vad", identity="vad:ucf:clip", device="cuda:0")
    context = {"started_at_utc_epoch": time.time(), "deadline_utc_epoch": time.time() + 30}
    with pytest.raises(probe.ProbeError, match="SOURCE_MISMATCH"):
        probe.run(args, context)
    assert context["runtime"] == {"path": str(runtime), "sha256": "r" * 64}
    assert "source_manifest" not in context and "identity_manifest" not in context and "model_manifest" not in context


def test_nondefault_device_peaks_and_worker_failure_are_sanitized(monkeypatch):
    probe = _probe()
    calls = []
    context = {"device": "cuda:3", "cuda_started": True, "_torch": SimpleNamespace(cuda=SimpleNamespace(
        max_memory_allocated=lambda device: calls.append(("allocated", device)) or 17,
        max_memory_reserved=lambda device: calls.append(("reserved", device)) or 29))}
    probe._peaks(context)
    assert calls == [("allocated", "cuda:3"), ("reserved", "cuda:3")]
    monkeypatch.setattr(probe, "_bootstrap", lambda *args: None)
    monkeypatch.setattr(probe, "_api", lambda: None)
    monkeypatch.setattr(probe, "run", lambda *args: (_ for _ in ()).throw(RuntimeError("secret generated answer")))
    started = time.time()
    result = probe._supervise(SimpleNamespace(preflight_report="unused", preflight_report_sha256="x" * 64), {
        "status": "RUNNING", "stage": "admitted", "started_at_utc_epoch": started,
        "deadline_utc_epoch": started + 10, "device": "cuda:3", "cuda_started": True,
        "_torch": context["_torch"], "peak_cuda_allocated_bytes": None, "peak_cuda_reserved_bytes": None})
    assert result["status"] == "FAILED_RUNTIME_PROBE" and result["error_code"] == "RUNTIME_FAILURE"
    assert result["peak_cuda_allocated_bytes"] == 17 and result["peak_cuda_reserved_bytes"] == 29
    assert "secret generated answer" not in json.dumps(result)


def test_worker_maps_preflight_supervision_to_preflight_invalid_and_omits_untyped_context(monkeypatch, tmp_path):
    probe, args, context, docs = _install_success_environment(monkeypatch, tmp_path, "vad")
    docs["preflight"][1]["answer"] = "secret generated answer"
    context["secret"] = "secret generated answer"
    monkeypatch.setattr(probe, "_bootstrap", lambda *args: None)
    monkeypatch.setattr(probe, "_api", lambda: None)
    result = probe._supervise(args, context)
    rendered = json.dumps(result)
    assert result["status"] == "INCOMPLETE_RUNTIME_PROBE"
    assert result["error_code"] == "PREFLIGHT_INVALID"
    assert "secret generated answer" not in rendered
    assert "secret" not in result


def test_tokenizer_binding_rejects_nul_sentinel_before_context_or_worker_ipc(monkeypatch, tmp_path):
    probe, args, context, docs = _install_success_environment(monkeypatch, tmp_path, "vad")
    sentinel = "tokenizer-sentinel"
    docs["runtime"][1]["inherited"]["tokenizer"] = "/tmp/" + sentinel + "\0suffix"
    with pytest.raises(probe.ProbeError, match="PREFLIGHT_INVALID"):
        probe.run(args, context)
    assert sentinel not in json.dumps(probe._snapshot(context))
    monkeypatch.setattr(probe, "_bootstrap", lambda *args: None)
    monkeypatch.setattr(probe, "_api", lambda: None)
    result = probe._supervise(args, context)
    assert result["error_code"] == "PREFLIGHT_INVALID"
    assert sentinel not in json.dumps(result)


def test_worker_ipc_snapshot_drops_values_that_exceed_the_serialized_report_limit():
    probe = _probe()

    def oversized_snapshot(_args, context, connection):
        context.update(status="INCOMPLETE_RUNTIME_PROBE", error_code="PAYLOAD_INVALID",
                       summary={"sentinel": "x" * probe.REPORT_CAPACITY})
        connection.send(probe._snapshot(context))

    started = time.time()
    result = probe._supervise(SimpleNamespace(), {"status": "RUNNING", "stage": "admitted",
        "started_at_utc_epoch": started, "deadline_utc_epoch": started + 5, "device": "cuda:0"},
        target=oversized_snapshot)
    assert "summary" not in result
    assert len(json.dumps(result, sort_keys=True, separators=(",", ":")).encode("utf-8")) < probe.REPORT_CAPACITY


def test_snapshot_fallback_bounds_aggregate_values_and_oversized_integers():
    probe = _probe()
    aggregate = {str(index): ["x" * probe.MAX_BINDING_PATH_BYTES for _ in range(128)] for index in range(64)}
    result = probe._snapshot({"status": "PASS_RUNTIME_PROBE", "elapsed_seconds": aggregate,
                              "counters": {"oversized": 1 << 10000}})
    encoded = json.dumps(result, sort_keys=True, separators=(",", ":")).encode("utf-8")
    assert result == {"status": "FAILED_RUNTIME_PROBE", "stage": "report_sanitization", "error_code": "RUNTIME_FAILURE"}
    assert len(encoded) < probe.MAX_IPC_REPORT_BYTES


def test_supervised_reservation_times_out_during_stalled_directory_sync(monkeypatch, tmp_path):
    probe = _probe()
    monkeypatch.setattr(probe.shutil, "disk_usage", lambda path: SimpleNamespace(free=probe.RESERVE + probe.REPORT_CAPACITY + 1_000_000))
    args = _admission(probe, tmp_path / "admission.json", tmp_path / "report.json")
    admission, _execution, _publication = probe._read_admission(args)
    monkeypatch.setattr(probe, "_sync", lambda path: time.sleep(30))
    deadline = time.time() + 0.1
    with pytest.raises(probe.ProbeError, match="DEADLINE_EXCEEDED"):
        probe._supervise_operation(probe._reserve_slot_operation, (args, admission, deadline), deadline=deadline)
    assert not Path(args.output).exists()


@pytest.mark.parametrize("sync_phase", (1, 2))
def test_post_link_sync_timeout_leaves_only_pending_report(monkeypatch, tmp_path, sync_phase):
    probe = _probe()
    monkeypatch.setattr(probe.shutil, "disk_usage", lambda path: SimpleNamespace(free=probe.RESERVE + probe.REPORT_CAPACITY + 1_000_000))
    admission, _deadline, slot = probe._admit(_admission(probe, tmp_path / "admission.json", tmp_path / "report.json"))
    original_sync = probe._sync
    calls = []

    def stall_post_link(path):
        calls.append(path)
        if len(calls) == sync_phase:
            time.sleep(30)
        return original_sync(path)

    monkeypatch.setattr(probe, "_sync", stall_post_link)
    deadline = time.time() + 0.1
    try:
        with pytest.raises(probe.ProbeError, match="DEADLINE_EXCEEDED"):
            probe._supervise_operation(probe._publish_operation,
                (slot, {"status": "PASS_RUNTIME_PROBE"}, admission["min_free_bytes"], deadline), deadline=deadline)
        target = Path(slot["target"])
        assert target.exists()
        report = json.loads(target.read_text(encoding="utf-8"))
        assert report["status"] == "PENDING_TERMINAL_CONFIRMATION"
        assert report["candidate_result"]["status"] == "PASS_RUNTIME_PROBE"
    finally:
        for path in (Path(slot["target"]), Path(slot["temporary"])):
            if path.exists():
                path.unlink()


def test_unreapable_operation_does_not_extend_supervisor_return(monkeypatch):
    probe = _probe()

    def stalled():
        time.sleep(.5)

    child, receive = probe._fork_target(lambda sender: stalled(), ())
    monkeypatch.setattr(probe, "_terminate", lambda _child: None)
    started = time.monotonic()
    _messages, timed_out, status = probe._collect(child, receive, deadline=time.time() + .05)
    assert timed_out and status is None and time.monotonic() - started < .25
    os.kill(child, probe.signal.SIGKILL)
    os.waitpid(child, 0)


def test_main_publishes_sanitized_child_timeout_during_reserved_finalization(monkeypatch, tmp_path, capsys):
    probe = _probe()
    monkeypatch.setattr(probe, "PROJECT_VOLUME", tmp_path)
    monkeypatch.setattr(probe, "RESERVE", 0)
    monkeypatch.setattr(probe, "FINALIZATION_SECONDS", 0.5)
    output = tmp_path / "report.json"
    args = _admission(probe, tmp_path / "admission.json", output, min_free_bytes=0,
                      run_budget_seconds=2, deadline_utc_epoch=time.time() + 10)

    def timed_out(_args, context):
        time.sleep(max(0, context["deadline_utc_epoch"] - time.time()) + 0.01)
        return {**context, "status": "INCOMPLETE_RUNTIME_PROBE", "error_code": "DEADLINE_EXCEEDED"}

    monkeypatch.setattr(probe, "_supervise", timed_out)
    monkeypatch.setattr(probe.sys, "argv", ["probe", "--preflight-report", str(tmp_path / "preflight.json"),
                        "--preflight-report-sha256", "p" * 64, "--diagnostic-admission", args.diagnostic_admission,
                        "--diagnostic-admission-sha256", args.diagnostic_admission_sha256, "--identity", "vad:ucf:clip",
                        "--device", "cuda:0", "--output", str(output), "--kind", "vad"])
    assert probe.main() == 3
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "PENDING_TERMINAL_CONFIRMATION"
    assert report["candidate_result"]["error_code"] == "DEADLINE_EXCEEDED"
    assert report["candidate_result"]["publication_deadline_utc_epoch"] > report["candidate_result"]["deadline_utc_epoch"]
    terminal = json.loads(capsys.readouterr().out)
    assert terminal["status"] == "TERMINAL_RUNTIME_PROBE"
    assert terminal["candidate_error_code"] == "DEADLINE_EXCEEDED"


def test_main_supervises_script_hash_and_preserves_original_admission_start(monkeypatch, tmp_path, capsys):
    probe = _probe()
    monkeypatch.setattr(probe, "PROJECT_VOLUME", tmp_path)
    monkeypatch.setattr(probe, "RESERVE", 0)
    monkeypatch.setattr(probe, "FINALIZATION_SECONDS", .5)
    output = tmp_path / "report.json"
    args = _admission(probe, tmp_path / "admission.json", output, min_free_bytes=0,
                      run_budget_seconds=2, deadline_utc_epoch=time.time() + 10)
    original_hash = probe._hash_script_operation
    original_admission = probe._read_admission_operation

    def delayed_admission(*operation_args):
        time.sleep(.25)
        return original_admission(*operation_args)

    def stalled_hash(path):
        time.sleep(30)
        return original_hash(path)

    monkeypatch.setattr(probe, "_hash_script_operation", stalled_hash)
    monkeypatch.setattr(probe, "_read_admission_operation", delayed_admission)
    monkeypatch.setattr(probe.sys, "argv", ["probe", "--preflight-report", "unused", "--preflight-report-sha256", "p" * 64,
                        "--diagnostic-admission", args.diagnostic_admission, "--diagnostic-admission-sha256", args.diagnostic_admission_sha256,
                        "--identity", "vad:ucf:clip", "--device", "cuda:0", "--output", str(output), "--kind", "vad"])
    before = time.time()
    assert probe.main() == 3
    candidate = json.loads(output.read_text(encoding="utf-8"))["candidate_result"]
    assert candidate["error_code"] == "DEADLINE_EXCEEDED"
    assert candidate["publication_deadline_utc_epoch"] <= before + 2.1
    assert json.loads(capsys.readouterr().out)["status"] == "TERMINAL_RUNTIME_PROBE"


def test_run_maps_malformed_fast_document_to_preflight_invalid(monkeypatch, tmp_path):
    probe, args, context, docs = _install_success_environment(monkeypatch, tmp_path, "vad")
    docs["runtime"][1]["fast"] = {"snapshot": {"answer": "secret"}, "snapshot_sha256": "c" * 64}
    with pytest.raises(probe.ProbeError, match="PREFLIGHT_INVALID"):
        probe.run(args, context)
    assert "secret" not in json.dumps(probe._snapshot(context))


def test_run_maps_adapter_structured_output_error_to_payload_invalid(monkeypatch, tmp_path):
    probe, args, context, _docs = _install_success_environment(monkeypatch, tmp_path, "vad")
    _cuda(monkeypatch)
    original_factory = probe.default_factory

    def invalid_factory(*factory_args, **factory_kwargs):
        factory = original_factory(*factory_args, **factory_kwargs)
        factory["vad"] = SimpleNamespace(predict=lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("bad adapter output")))
        return factory

    monkeypatch.setattr(probe, "default_factory", invalid_factory)
    with pytest.raises(probe.ProbeError, match="PAYLOAD_INVALID"):
        probe.run(args, context)


def test_factory_failure_records_peaks_when_factory_initialized_cuda(monkeypatch, tmp_path):
    probe, args, context, _docs = _install_success_environment(monkeypatch, tmp_path, "vad")
    _cuda(monkeypatch, allocated=31, reserved=47)

    def initialized_factory(*_args, **_kwargs):
        torch.cuda.set_device("cuda:3")
        raise RuntimeError("factory failed after CUDA initialization")

    monkeypatch.setattr(probe, "default_factory", initialized_factory)
    with pytest.raises(RuntimeError, match="factory failed"):
        probe.run(args, context)
    assert context["cuda_started"] is True
    assert context["peak_cuda_allocated_bytes"] == 31
    assert context["peak_cuda_reserved_bytes"] == 47


def test_factory_peak_survives_reset_before_lower_loader_peak(monkeypatch, tmp_path):
    probe, args, context, _docs = _install_success_environment(monkeypatch, tmp_path, "vad")
    state = {"selected": False, "factory": True}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: state["selected"])
    monkeypatch.setattr(torch.cuda, "set_device", lambda _device: state.update(selected=True))
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda _device: state.update(factory=False))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _device: 101 if state["factory"] else 11)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda _device: 151 if state["factory"] else 17)
    factory = probe.default_factory

    def initialized_factory(*factory_args, **factory_kwargs):
        torch.cuda.set_device("cuda:3")
        return factory(*factory_args, **factory_kwargs)

    monkeypatch.setattr(probe, "default_factory", initialized_factory)
    result = probe.run(args, context)
    assert result["peak_cuda_allocated_bytes"] == 101
    assert result["peak_cuda_reserved_bytes"] == 151


def test_post_factory_deadline_retains_factory_peak(monkeypatch, tmp_path):
    probe, args, context, _docs = _install_success_environment(monkeypatch, tmp_path, "vad")
    state = {"selected": False}
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: state["selected"])
    monkeypatch.setattr(torch.cuda, "set_device", lambda _device: state.update(selected=True))
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _device: 73)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda _device: 89)
    factory = probe.default_factory

    def initialized_factory(*factory_args, **factory_kwargs):
        torch.cuda.set_device("cuda:3")
        context["deadline_utc_epoch"] = time.time() - 1
        return factory(*factory_args, **factory_kwargs)

    monkeypatch.setattr(probe, "default_factory", initialized_factory)
    with pytest.raises(probe.ProbeError, match="DEADLINE_EXCEEDED"):
        probe.run(args, context)
    assert context["peak_cuda_allocated_bytes"] == 73
    assert context["peak_cuda_reserved_bytes"] == 89


@pytest.mark.parametrize("kind", ("vad", "vau"))
def test_successful_routes_use_real_request_geometry_validators_and_counters(monkeypatch, tmp_path, kind):
    probe, args, context, _docs = _install_success_environment(monkeypatch, tmp_path, kind)
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    calls = _cuda(monkeypatch)
    factory = probe.default_factory

    def policy_configuring_factory(*factory_args, **factory_kwargs):
        configure_deterministic_algorithms()
        return factory(*factory_args, **factory_kwargs)

    monkeypatch.setattr(probe, "default_factory", policy_configuring_factory)
    result = probe.run(args, context)
    assert result["status"] == "PASS_RUNTIME_PROBE"
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == deterministic_policy().cublas_workspace_config
    assert torch.are_deterministic_algorithms_enabled()
    assert result["summary"]["kind"] == kind
    assert result["peak_cuda_allocated_bytes"] == 11
    assert result["peak_cuda_reserved_bytes"] == 22
    assert set(result["phase_seconds"]) == {"factory", "cuda_initialization", "model_loading", "inference"}
    assert all(value >= 0 for value in result["phase_seconds"].values())
    assert calls[:3] == [("set_device", "cuda:3"), ("reset", "cuda:3"), ("sync", "cuda:3")]
    assert ("allocated", "cuda:3") in calls and ("reserved", "cuda:3") in calls
    if kind == "vad":
        assert result["summary"]["validated_trigger_count"] == 1
        assert result["summary"]["completed_slow_forwards"] == 1
        assert result["summary"]["completed_generation_calls"] == 0
    else:
        assert result["summary"]["completed_slow_forwards"] == 1
        assert result["summary"]["completed_generation_calls"] == 1


def test_raw_slow_counter_observes_direct_forward_generation_and_restores_signature():
    loaded = _loaded_model(generation_fails=True)
    raw = loaded.bridge.raw_slow
    original_signature = inspect.signature(raw.forward)
    counters = {"completed_slow_forwards": 0}
    with pytest.raises(RuntimeError, match="generation route failed"):
        with count_slow_execution(loaded, counters) as observed:
            assert observed is raw
            assert inspect.signature(raw.forward) == original_signature
            loaded.bridge.slow(torch.tensor([2.0]))
            loaded.hivau_inference.generate()
    assert counters["completed_slow_forwards"] == 2
    assert "forward" not in vars(raw)
    assert inspect.signature(raw.forward) == original_signature


def test_snapshot_aggregate_fallback_omits_integer_that_cannot_be_converted_to_float():
    probe = _probe()
    aggregate = {str(i): ["x" * 4096 for _ in range(128)] for i in range(2)}
    value = probe._snapshot({"status": "PASS_RUNTIME_PROBE", "elapsed_seconds": aggregate,
                             "started_at_utc_epoch": 1 << 10000})
    assert value["status"] == "FAILED_RUNTIME_PROBE"
    assert "started_at_utc_epoch" not in value
    assert len(json.dumps(value).encode()) < probe.MAX_IPC_REPORT_BYTES


def test_collect_parses_short_reads_after_reaping_child(monkeypatch):
    probe = _probe()
    expected = {"status": "PASS_RUNTIME_PROBE", "summary": {"queries": 31}}
    def completed(connection):
        connection.send(expected)
    child, receive = probe._fork_target(completed, ())
    os.waitid(os.P_PID, child, os.WEXITED | os.WNOWAIT)
    original_read = os.read
    monkeypatch.setattr(probe.os, "read", lambda descriptor, length: original_read(descriptor, min(length, 16)))
    messages, timed_out, exit_status = probe._collect(child, receive, deadline=time.time() + 2)
    assert messages == [expected]
    assert timed_out is False
    assert os.WIFEXITED(exit_status) and os.WEXITSTATUS(exit_status) == 0


def test_collect_does_not_wait_for_descendant_pipe_after_child_exit():
    probe = _probe()
    def completed(connection):
        if os.fork() == 0:
            time.sleep(.5)
            os._exit(0)
        connection.send({"status": "PASS_RUNTIME_PROBE"})
    child, receive = probe._fork_target(completed, ())
    os.waitid(os.P_PID, child, os.WEXITED | os.WNOWAIT)
    started = time.monotonic()
    messages, timed_out, status = probe._collect(child, receive, deadline=time.time() + .05)
    assert time.monotonic() - started < .3
    assert messages == [{"status": "PASS_RUNTIME_PROBE"}]
    assert timed_out is False and os.WIFEXITED(status)
