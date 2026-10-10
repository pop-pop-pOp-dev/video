#!/usr/bin/env python3
"""Prepare and measure a non-admitted A/U/S/F formal-bundle qualification."""
from __future__ import annotations

import argparse, ast, copy, hashlib, json, os, socket, subprocess, sys, time, uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nc_rted.production_runtime import load_manifest, preflight
from nc_rted.resource_attestation import (PROJECT_VOLUME, formal_runtime_identity, gpu_memory_bytes,
                                          gpu_uuid, python_runtime_probe, sha256_file,
                                          validate_environment)
from nc_rted.task_inputs import TrainingCatalog, detection_window_id
from nc_rted.training import order_sha256, sample_order

GROUPS = ("A", "U", "S", "F")
PROFILE_SCHEMA = "nc_rted_interleaved_formal_qualification_profile/v1"
QUALIFICATION_SCHEMA = "nc_rted_runtime_qualification/v2"
LONG_INPUT_SCHEMA = "nc_rted_complete_long_input_coverage/v2"

# These are the implementation files that turn an accepted selected input into
# provider material.  An old component report can be reused only when its
# relevant production lineage is byte-for-byte the lineage in the bound source
# closure.  The report is evidence of its selected longest input; it is never
# presented as evidence that the bounded diagnostic consumed that input.
LONGEST_SOURCE_SCOPE = {
    "caption": {"src/nc_rted/caption_provider.py", "src/nc_rted/caption_sampling.py",
                "src/nc_rted/task_inputs.py", "src/nc_rted/training.py",
                "src/nc_rted/production_runtime.py"},
    "detection_xd": {"src/nc_rted/detection_provider.py", "src/nc_rted/detection_media.py",
                     "src/nc_rted/task_inputs.py", "src/nc_rted/training.py",
                     "src/nc_rted/production_runtime.py"},
    "detection_ucf": {"src/nc_rted/detection_provider.py", "src/nc_rted/detection_media.py",
                      "src/nc_rted/task_inputs.py", "src/nc_rted/training.py",
                      "src/nc_rted/production_runtime.py"},
}

LONGEST_STATUS = {
    "caption": "PASS_REAL_LONGEST_CAPTION_PROVIDER_FORWARD_BACKWARD",
    "detection_xd": "PASS_REAL_LONGEST_DETECTION_PREFIX_FORWARD_BACKWARD",
    "detection_ucf": "PASS_REAL_LONGEST_DETECTION_PREFIX_FORWARD_BACKWARD",
}

# The accepted longest reports predate the captured diagnostic closure.  These
# are the narrow callable paths that construct the selected material.  We
# compare their ASTs against a source tree whose file hashes match the report;
# the one decoder transition is checked structurally because it deliberately
# replaces seek/read with the accepted monotone grab/retrieve implementation.
LINEAGE_CALLABLES = {
    "caption": {
        "src/nc_rted/caption_provider.py": ("Stage2CaptionProvider.__call__",),
        "src/nc_rted/training.py": ("sample_order", "order_sha256"),
        "src/nc_rted/production_runtime.py": ("_protocols", "_validate_caption_subset"),
    },
    "detection_xd": {
        "src/nc_rted/detection_provider.py": ("FrozenDetectionProvider.__call__", "original_detection_time_message"),
        "src/nc_rted/training.py": ("sample_order", "order_sha256"),
        "src/nc_rted/production_runtime.py": ("_protocols", "_validate_detection_bindings"),
    },
    "detection_ucf": {
        "src/nc_rted/detection_provider.py": ("FrozenDetectionProvider.__call__", "original_detection_time_message"),
        "src/nc_rted/training.py": ("sample_order", "order_sha256"),
        "src/nc_rted/production_runtime.py": ("_protocols", "_validate_detection_bindings"),
    },
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_once(path: Path, document: dict) -> str:
    payload = (json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload: raise ValueError("immutable output already differs")
        return _sha(path)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_bytes(payload)
    os.link(temporary, path); temporary.unlink(missing_ok=True)
    return _sha(path)


def _diagnostic_bundle(path: Path, digest: str) -> dict:
    if not path.is_absolute() or _sha(path) != digest: raise ValueError("diagnostic bundle differs")
    doc = json.loads(path.read_text())
    required = {"schema", "diagnostic_updates", "diagnostic_checkpoint_interval", "bundle_checkpoint_root", "members", "source_manifest", "source_manifest_sha256"}
    if set(doc) != required or doc["schema"] != "nc_rted_interleaved_gpu_harness/v1" or set(doc["members"]) != set(GROUPS):
        raise ValueError("diagnostic bundle schema differs")
    return doc


def _formal_document(document: dict, group: str, root: Path) -> dict:
    result = copy.deepcopy(document); run = result["run"]
    if run.get("mode") != "diagnostic" or run.get("group") != group: raise ValueError("diagnostic member differs")
    run.update({"run_id": f"qualification:formal:{run['seed']}:{group}", "mode": "formal",
                "checkpoint_root": str(root / "runs" / group / "checkpoints"),
                "progress_path": str(root / "runs" / group / "progress.json")})
    run.pop("diagnostic_updates", None)
    return result


def _source_identity(files: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _verify_common_source_closure(bundle: dict, source_sha256: str) -> None:
    path = Path(bundle["source_manifest"])
    if not path.is_file() or _sha(path) != bundle["source_manifest_sha256"]:
        raise ValueError("diagnostic source manifest differs")
    document = json.loads(path.read_text())
    files = document.get("files") if isinstance(document, dict) else None
    required = {"scripts/nc_rted_interleaved_gpu_diagnostic.py",
                "scripts/nc_rted_interleaved_formal.py",
                "scripts/nc_rted_qualify_interleaved_formal_bundle.py"}
    required.update(f"src/nc_rted/{item.name}" for item in (ROOT / "src" / "nc_rted").glob("*.py"))
    if (not isinstance(files, dict) or not required.issubset(files) or
            document.get("schema") != "nc_rted_interleaved_source_manifest/v1" or
            document.get("code_sha256") != _source_identity(files) or source_sha256 != document.get("code_sha256")):
        raise ValueError("diagnostic source closure is not usable by the formal runner")
    for relative, digest in files.items():
        candidate = ROOT / relative
        if (not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts or
                not isinstance(digest, str) or len(digest) != 64 or not candidate.is_file() or
                _sha(candidate) != digest):
            raise ValueError("accepted source closure differs")


def _source_files(bundle: dict, source_sha256: str) -> dict[str, str]:
    """Return the verified source closure used by the diagnostic runtime."""
    _verify_common_source_closure(bundle, source_sha256)
    return json.loads(Path(bundle["source_manifest"]).read_text())["files"]


def _execution_binding(interpreter_value: str, environment: dict, physical_gpu: int) -> tuple[dict, dict, str]:
    launcher = Path(interpreter_value).absolute()
    if not launcher.is_file():
        raise ValueError("interpreter is absent")
    validate_environment(environment, PROJECT_VOLUME)
    device_uuid = gpu_uuid(physical_gpu)
    interpreter = {"path": str(launcher), "launcher_sha256": sha256_file(launcher),
                   "target_sha256": sha256_file(launcher.resolve())}
    # The formal environment is the attested allowlist. CUDA visibility is a
    # separate hardware binding and is never inherited from this controller.
    child_environment = dict(environment, CUDA_VISIBLE_DEVICES=device_uuid)
    return interpreter, child_environment, device_uuid


def _profile_identities(profile: dict, profile_path: Path, bundle: dict, bundle_path: Path) -> dict:
    required = {"schema", "status", "diagnostic_bundle", "diagnostic_bundle_sha256", "members",
                "member_identities", "updates", "accumulation", "shared_preparation", "common_recovery",
                "complete_long_input_coverage"}
    if (set(profile) != required or profile.get("updates") != 1000 or profile.get("accumulation") != 8 or
            profile.get("shared_preparation") != "frozen_provider_only" or profile.get("common_recovery") is not True or
            not isinstance(profile.get("diagnostic_bundle"), str) or
            Path(profile["diagnostic_bundle"]) != bundle_path):
        raise ValueError("qualification profile structure differs")
    if set(profile.get("members", {})) != set(GROUPS) or set(profile.get("member_identities", {})) != set(GROUPS):
        raise ValueError("qualification profile members differ")
    root = profile_path.parent; identities = {}
    for group in GROUPS:
        item = profile["members"][group]
        if set(item) != {"runtime", "runtime_sha256"}: raise ValueError("profile member binding differs")
        path = Path(item["runtime"])
        if path != root / "members" / f"{group}.runtime.json" or _sha(path) != item["runtime_sha256"]:
            raise ValueError("profile member file differs")
        diagnostic = bundle["members"][group]
        origin = load_manifest(diagnostic["manifest"], expected_sha256=diagnostic["sha256"])
        preflight(origin)
        expected = _formal_document(origin.document, group, root)
        if json.loads(path.read_text()) != expected: raise ValueError("profile is not derived from diagnostic member")
        manifest = load_manifest(path, expected_sha256=item["runtime_sha256"])
        preflight(manifest)
        identity = formal_runtime_identity(manifest)
        if profile["member_identities"][group] != identity: raise ValueError("profile member identity differs")
        identities[group] = identity
    return identities


def _binding(path_value: str, digest: str) -> dict:
    path = Path(path_value).absolute()
    if (not path.is_file() or not isinstance(digest, str) or len(digest) != 64 or
            _sha(path) != digest):
        raise ValueError("bound evidence differs")
    return {"path": str(path), "sha256": digest}


def _catalog_and_prefix(runtime_document: dict) -> tuple[dict, TrainingCatalog, dict]:
    catalog_document = runtime_document.get("catalog")
    run = runtime_document.get("run")
    if not isinstance(catalog_document, dict) or not isinstance(run, dict):
        raise ValueError("runtime lacks catalog or run binding")
    required = {"manifest_directory", "training_annotations", "training_annotations_sha256",
                "provenance", "provenance_sha256"}
    if not required.issubset(catalog_document): raise ValueError("runtime catalog binding differs")
    for path_key, hash_key in (("training_annotations", "training_annotations_sha256"),
                               ("provenance", "provenance_sha256")):
        _binding(catalog_document[path_key], catalog_document[hash_key])
    catalog = TrainingCatalog.load(catalog_document["manifest_directory"], catalog_document["training_annotations"],
                                   expected_provenance_sha256=catalog_document["provenance_sha256"])
    updates = run.get("diagnostic_updates")
    seed = run.get("seed")
    if type(updates) is not int or updates < 1 or type(seed) is not int:
        raise ValueError("runtime diagnostic prefix binding differs")
    prefix = sample_order(list(catalog.tasks), seed)[:updates * 8]
    if len(prefix) != updates * 8 or len(prefix) != len(set(prefix)):
        raise ValueError("runtime cannot derive its fixed diagnostic prefix")
    catalog_binding = {key: catalog_document[key] for key in sorted(required)}
    catalog_binding["identity"] = catalog.identity
    return catalog_binding, catalog, {"updates": updates, "accumulation": 8, "seed": seed,
                                       "sample_ids": prefix, "order_sha256": order_sha256(prefix)}


def _selected_catalog_sample(name: str, selected: dict, catalog: TrainingCatalog) -> str:
    if name == "caption":
        if not isinstance(selected.get("instruction_id"), int) or not isinstance(selected.get("media"), dict):
            raise ValueError("caption-longest selection lacks its catalog identity")
        sample_id = f"caption:{selected['media'].get('dataset')}:{selected['instruction_id']}"
        task = catalog.tasks.get(sample_id)
        if (task is None or task.task != "caption" or task.instruction is None or
                task.instruction.get("id") != selected["instruction_id"] or
                task.media_key != selected.get("relative_video")):
            raise ValueError("caption-longest selection is not in the bound catalog")
        return sample_id
    if (selected.get("dataset") != ("xd-violence" if name == "detection_xd" else "ucf-crime") or
            not isinstance(selected.get("key"), str) or type(selected.get("query_index")) is not int):
        raise ValueError("detection-longest selection lacks its catalog identity")
    sample_id = f"detection:{selected['dataset']}:{selected['key']}:{selected['query_index']}"
    task = catalog.tasks.get(sample_id)
    if task is None or task.task != "detection" or detection_window_id({"dataset": task.dataset, "key": task.media_key, "query_index": task.query_index}) != sample_id:
        raise ValueError("detection-longest selection is not in the bound catalog")
    return sample_id


def _callable_ast(path: Path, qualified_name: str) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = tree
    for part in qualified_name.split("."):
        matches = [item for item in getattr(node, "body", [])
                   if isinstance(item, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == part]
        if len(matches) != 1: raise ValueError("historical source lacks required material callable")
        node = matches[0]
    return hashlib.sha256(ast.dump(node, annotate_fields=True, include_attributes=False).encode()).hexdigest()


def _monotone_decode_transition(historical: Path, current: Path) -> dict:
    """Recognize the reviewed seek/read to monotone grab/retrieve replacement."""
    before, after = historical.read_text(encoding="utf-8"), current.read_text(encoding="utf-8")
    if ("CAP_PROP_POS_FRAMES" not in before or "capture.read()" not in before or
            "if index < self._next_index:" not in after or "capture.grab()" not in after or
            "capture.retrieve()" not in after or "CAP_PROP_POS_FRAMES" not in after):
        raise ValueError("detection decode transition is not the accepted monotone equivalent")
    return {"kind": "reviewed_monotone_opencv_decode", "historical_sha256": _sha(historical),
            "current_sha256": _sha(current)}


def _source_lineage(name: str, report_sources: dict, source_files: dict[str, str], historical_root: str | None) -> dict:
    scope = LONGEST_SOURCE_SCOPE[name]
    if not scope.issubset(report_sources) or not scope.issubset(source_files):
        raise ValueError("component-longest report source scope is incomplete")
    mismatches = {relative for relative in scope if report_sources[relative] != source_files[relative]}
    if not mismatches:
        return {"mode": "exact_source_hashes", "files": {path: source_files[path] for path in sorted(scope)}}
    if not historical_root:
        raise ValueError("component-longest report has no unchanged provider/training lineage")
    root = Path(historical_root).absolute()
    if not root.is_dir(): raise ValueError("historical longest source root is absent")
    transitions = {}
    for relative in sorted(scope):
        historical, current = root / relative, ROOT / relative
        if not historical.is_file() or _sha(historical) != report_sources[relative]:
            raise ValueError("historical longest source does not match its accepted report")
        if _sha(current) != source_files[relative]:
            raise ValueError("diagnostic source closure differs from checked current source")
        if relative not in mismatches:
            transitions[relative] = {"kind": "exact_source_hash", "sha256": source_files[relative]}
        elif relative == "src/nc_rted/detection_media.py":
            transitions[relative] = _monotone_decode_transition(historical, current)
        else:
            callables = LINEAGE_CALLABLES[name].get(relative)
            if not callables:
                raise ValueError("changed longest source has no scoped compatibility proof")
            pairs = {callable_name: {"historical_ast_sha256": _callable_ast(historical, callable_name),
                                     "current_ast_sha256": _callable_ast(current, callable_name)}
                     for callable_name in callables}
            if any(pair["historical_ast_sha256"] != pair["current_ast_sha256"] for pair in pairs.values()):
                raise ValueError("changed longest source alters a selected-material callable")
            transitions[relative] = {"kind": "exact_selected_callable_ast", "callables": pairs,
                                     "historical_sha256": _sha(historical), "current_sha256": _sha(current)}
    return {"mode": "historical_source_and_scoped_callable_compatibility", "historical_root": str(root),
            "files": transitions}


def _caption_candidate_binding(path_value: str, digest: str, record: dict, runtime_document: dict) -> dict:
    binding = _binding(path_value, digest)
    document = json.loads(Path(binding["path"]).read_text())
    selected = record.get("selected", {})
    longest = document.get("global_longest_all_2000") if isinstance(document, dict) else None
    catalog = runtime_document.get("catalog", {})
    media = selected.get("media") if isinstance(selected, dict) else None
    if (not isinstance(catalog, dict) or not isinstance(catalog.get("caption_subset"), str) or
            not isinstance(catalog.get("caption_subset_sha256"), str)):
        raise ValueError("caption-longest runtime lacks a fixed full caption subset binding")
    subset_binding = _binding(catalog["caption_subset"], catalog["caption_subset_sha256"])
    try:
        subset_rows = json.loads(Path(subset_binding["path"]).read_text())
    except (OSError, ValueError) as error:
        raise ValueError("caption-longest full caption subset is not valid JSON") from error
    subset_matches = [row for row in subset_rows if isinstance(row, dict) and
                      row.get("id") == selected.get("instruction_id") and
                      row.get("video") == selected.get("relative_video")] if isinstance(subset_rows, list) else []
    if (document.get("schema") != "nc_rted_caption_longest_candidates/v2" or
            document.get("caption_subset_sha256") != catalog.get("caption_subset_sha256") or
            not isinstance(longest, dict) or not isinstance(media, dict) or
            len(subset_matches) != 1 or
            longest.get("id") != selected.get("instruction_id") or
            longest.get("dataset") != media.get("dataset") or
            longest.get("relative_video") != selected.get("relative_video") or
            longest.get("source") != media.get("media_path") or
            longest.get("source_sha256") != media.get("media_sha256")):
        raise ValueError("caption-longest candidate record does not map selected input into the full caption subset")
    diagnostic_subset = record.get("diagnostic_subset_sha256")
    if not isinstance(diagnostic_subset, str) or len(diagnostic_subset) != 64:
        raise ValueError("caption-longest report lacks its diagnostic subset binding")
    return {**binding, "full_caption_subset": subset_binding,
            "diagnostic_subset_sha256": diagnostic_subset,
            "selected": {key: longest[key] for key in ("id", "dataset", "relative_video", "source", "source_sha256")}}


def _component_policy(name: str, record: dict, runtime_document: dict, caption_candidates: dict | None) -> dict:
    if name == "caption":
        catalog, sampling, observed = runtime_document.get("catalog", {}), runtime_document.get("sampling", {}), record.get("original_sampling")
        if (not isinstance(observed, dict) or record.get("pg_sha256") != catalog.get("pg_scores_sha256") or
                sampling.get("local_num_frames") != 1 or sampling.get("frames_upbound") != 64 or
                sampling.get("frames_lowbound") != 4 or sampling.get("sample_type") != "dynamic_fps1" or
                sampling.get("time_msg") != "short_online_v2" or len(observed.get("times", [])) != 64 or
                observed.get("time_message", "").find("64 frames") < 0):
            raise ValueError("caption-longest report input policy differs from the bound runtime")
        if not isinstance(caption_candidates, dict): raise ValueError("caption-longest candidates are absent")
        return {"kind": "caption_original_sampling", "caption_longest_candidates": _caption_candidate_binding(
                    caption_candidates.get("path"), caption_candidates.get("sha256"), record, runtime_document),
                "pg_sha256": record["pg_sha256"], "original_sampling": observed,
                "runtime_sampling": sampling}
    protocol = runtime_document.get("fast", {}).get("protocols", {}).get("xd-violence" if name == "detection_xd" else "ucf-crime")
    if not isinstance(protocol, dict) or record.get("diagnostic_protocol") != protocol:
        raise ValueError("detection-longest report protocol differs from the bound runtime")
    return {"kind": "detection_protocol", "dataset": "xd-violence" if name == "detection_xd" else "ucf-crime",
            "diagnostic_protocol": protocol}


def _component_longest(name: str, path_value: str, digest: str, catalog: TrainingCatalog,
                       source_files: dict[str, str], runtime_document: dict, historical_root: str | None,
                       caption_candidates: dict | None) -> dict:
    binding = _binding(path_value, digest)
    record = json.loads(Path(binding["path"]).read_text())
    selected = record.get("selected") if isinstance(record, dict) else None
    report_sources = record.get("source_sha256") if isinstance(record, dict) else None
    if (record.get("status") != LONGEST_STATUS[name] or not isinstance(selected, dict) or
            not isinstance(report_sources, dict)):
        raise ValueError("component-longest evidence is incomplete")
    return {**binding, "selected_catalog_sample": _selected_catalog_sample(name, selected, catalog),
            "source_lineage": _source_lineage(name, report_sources, source_files, historical_root),
            "input_policy": _component_policy(name, record, runtime_document, caption_candidates)}


def _complete_long_input_coverage(runtime_document: dict, source_files: dict[str, str], bindings: dict[str, dict],
                                  historical_roots: dict[str, str | None], caption_candidates: dict) -> dict:
    catalog_binding, catalog, prefix = _catalog_and_prefix(runtime_document)
    if not isinstance(bindings, dict) or set(bindings) != set(LONGEST_STATUS):
        raise ValueError("complete-long-input coverage lacks all component-longest reports")
    if not isinstance(historical_roots, dict) or set(historical_roots) != set(LONGEST_STATUS):
        raise ValueError("complete-long-input coverage lacks historical source roots")
    components = {name: _component_longest(name, bindings[name].get("path"), bindings[name].get("sha256"),
                                            catalog, source_files, runtime_document, historical_roots[name], caption_candidates if name == "caption" else None)
                  for name in LONGEST_STATUS}
    return {"schema": LONG_INPUT_SCHEMA, "status": "BOUND_LONGEST_AND_DIAGNOSTIC_PREFIX",
            "training_catalog": catalog_binding,
            "input_policy": {"sampling": runtime_document.get("sampling"), "fast_protocols": runtime_document.get("fast", {}).get("protocols")},
            "component_longest_evidence": components, "diagnostic_prefix": prefix}


def _verify_diagnostic_prefix(coverage: dict, resumed: dict) -> dict:
    prefix = coverage["diagnostic_prefix"]
    expected = prefix["sample_ids"]
    observed = resumed.get("material_digests")
    if (not isinstance(observed, dict) or set(observed) != set(expected) or
            any(not isinstance(value, str) or len(value) != 64 for value in observed.values())):
        raise ValueError("diagnostic did not consume its bound fixed input prefix")
    return {"diagnostic_prefix": prefix, "material_digests": observed}


def _verify_diagnostic_device(raw: dict, resumed: dict, expected_uuid: str) -> None:
    serial_path = raw.get("serial_reference")
    if not isinstance(serial_path, str) or not Path(serial_path).is_file():
        raise ValueError("diagnostic lacks its serial device evidence")
    serial = json.loads(Path(serial_path).read_text())
    expected = {"cuda_visible_devices": expected_uuid, "gpu_uuid": expected_uuid}
    if serial.get("cuda_binding") != expected or resumed.get("cuda_binding") != expected:
        raise ValueError("diagnostic CUDA device differs from qualification device")


def prepare(args) -> None:
    bundle_path = Path(args.diagnostic_bundle).absolute(); bundle = _diagnostic_bundle(bundle_path, args.diagnostic_bundle_sha256)
    root = Path(args.output_root).absolute()
    if root.exists(): raise ValueError("qualification profile root must be new")
    members, identities, diagnostic_documents = {}, {}, {}
    for group in GROUPS:
        item = bundle["members"][group]
        runtime = load_manifest(item["manifest"], expected_sha256=item["sha256"])
        diagnostic_documents[group] = runtime.document
        doc = _formal_document(runtime.document, group, root)
        target = root / "members" / f"{group}.runtime.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        encoded = (json.dumps(doc, sort_keys=True, separators=(",", ":")) + "\n").encode(); target.write_bytes(encoded)
        manifest = load_manifest(target, expected_sha256=_sha(target))
        identities[group] = formal_runtime_identity(manifest)
        members[group] = {"runtime": str(target), "runtime_sha256": _sha(target)}
    source_sha256 = identities["A"].get("code_sha256")
    if (not isinstance(source_sha256, str) or len(source_sha256) != 64 or
            any(identity.get("code_sha256") != source_sha256 for identity in identities.values())):
        raise ValueError("formal members do not share one source identity")
    source_files = _source_files(bundle, source_sha256)
    long_input = _complete_long_input_coverage(
        diagnostic_documents["A"],
        source_files,
        {"caption": _binding(args.caption_longest_report, args.caption_longest_report_sha256),
         "detection_xd": _binding(args.detection_xd_longest_report, args.detection_xd_longest_report_sha256),
         "detection_ucf": _binding(args.detection_ucf_longest_report, args.detection_ucf_longest_report_sha256)},
        {"caption": args.caption_longest_source_root,
         "detection_xd": args.detection_xd_longest_source_root,
         "detection_ucf": args.detection_ucf_longest_source_root},
        _binding(args.caption_longest_candidates, args.caption_longest_candidates_sha256))
    profile = {"schema": PROFILE_SCHEMA, "status": "NON_ADMITTED_FORMAL_PROFILE", "diagnostic_bundle": str(bundle_path),
               "diagnostic_bundle_sha256": args.diagnostic_bundle_sha256, "members": members,
               "member_identities": identities, "updates": 1000, "accumulation": 8,
               "shared_preparation": "frozen_provider_only", "common_recovery": True,
               "complete_long_input_coverage": long_input}
    _write_once(root / "qualification_profile.json", profile)
    print(json.dumps({"status": profile["status"], "profile": str(root / "qualification_profile.json"), "sha256": _sha(root / "qualification_profile.json")}, sort_keys=True))


def qualify(args) -> None:
    profile_path = Path(args.profile).absolute(); profile = json.loads(profile_path.read_text())
    if _sha(profile_path) != args.profile_sha256 or profile.get("schema") != PROFILE_SCHEMA or profile.get("status") != "NON_ADMITTED_FORMAL_PROFILE": raise ValueError("qualification profile differs")
    bundle = Path(profile["diagnostic_bundle"]); bundle_doc = _diagnostic_bundle(bundle, profile["diagnostic_bundle_sha256"])
    if Path(args.runtime_source_root).resolve() != ROOT.resolve(): raise ValueError("runtime source root differs")
    environment = json.loads(args.execution_environment)
    if not isinstance(environment, dict): raise ValueError("execution environment is invalid")
    interpreter, child_environment, expected_uuid = _execution_binding(args.interpreter, environment, args.physical_gpu)
    identities = _profile_identities(profile, profile_path, bundle_doc, bundle)
    source_sha256 = identities["A"].get("code_sha256")
    if (not isinstance(source_sha256, str) or
            any(identity.get("code_sha256") != source_sha256 for identity in identities.values())):
        raise ValueError("profile members do not share one source identity")
    source_files = _source_files(bundle_doc, source_sha256)
    long_input = profile.get("complete_long_input_coverage")
    runtime_document = load_manifest(bundle_doc["members"]["A"]["manifest"],
                                     expected_sha256=bundle_doc["members"]["A"]["sha256"]).document
    if not isinstance(long_input, dict): raise ValueError("qualification profile complete-long-input coverage differs")
    bindings = {name: {key: long_input.get("component_longest_evidence", {}).get(name, {}).get(key)
                       for key in ("path", "sha256")} for name in LONGEST_STATUS}
    historical_roots = {name: long_input.get("component_longest_evidence", {}).get(name, {}).get("source_lineage", {}).get("historical_root")
                        for name in LONGEST_STATUS}
    caption_candidates = long_input.get("component_longest_evidence", {}).get("caption", {}).get("input_policy", {}).get("caption_longest_candidates")
    if long_input != _complete_long_input_coverage(runtime_document, source_files, bindings, historical_roots, caption_candidates):
        raise ValueError("qualification profile complete-long-input coverage differs")
    raw = Path(args.raw_output).absolute()
    command = [interpreter["path"], str(ROOT / "scripts" / "nc_rted_interleaved_gpu_diagnostic.py"), "--manifest", str(bundle), "--manifest-sha256", profile["diagnostic_bundle_sha256"], "--runtime-source-root", str(ROOT), "--output", str(raw), "--mode", "run", "--fault-after-group", args.fault_after_group]
    started = time.time(); subprocess.run(command, check=True, env=child_environment, cwd=str(PROJECT_VOLUME)); finished = time.time()
    raw_doc = json.loads(raw.read_text()); resumed = raw_doc.get("resumed", {})
    if raw_doc.get("status") != "BUNDLED_DIAGNOSTIC_COMPLETE" or resumed.get("status") != "BUNDLED_DIAGNOSTIC_COMPLETE" or resumed.get("serial_comparison") != "exact_state_digest_match": raise ValueError("diagnostic recovery is incomplete")
    if set(resumed.get("completed", {})) != set(GROUPS) or any(value < 1 for value in resumed["completed"].values()): raise ValueError("no completed bundle update")
    _verify_diagnostic_device(raw_doc, resumed, expected_uuid)
    diagnostic_prefix_coverage = _verify_diagnostic_prefix(long_input, resumed)
    probe_started = time.monotonic(); runtime = python_runtime_probe(interpreter, environment); probe_seconds = time.monotonic() - probe_started
    if gpu_uuid(args.physical_gpu) != expected_uuid: raise ValueError("GPU UUID changed during qualification")
    measurements = {"measured_at_utc_epoch": finished, "forward_backward_completed": True, "complete_long_input": True,
                    "common_boundary_recovery_completed": True, "optimizer_updates": min(resumed["completed"].values()),
                    "peak_cuda_allocated_bytes": resumed["peak_memory_bytes"], "peak_cuda_reserved_bytes": resumed["peak_reserved_memory_bytes"],
                    "seconds_per_bundle_update_upper_bound": resumed["elapsed_seconds"], "setup_checkpoint_seconds_upper_bound": finished - started,
                    "local_import_probe": {"status": "PASS", "interpreter": interpreter, "environment": environment, "duration_seconds": probe_seconds, "runtime_identity": runtime},
                    "complete_long_input_coverage": long_input,
                    "observed_diagnostic_prefix_coverage": diagnostic_prefix_coverage,
                    "raw_diagnostic": {"path": str(raw), "sha256": _sha(raw), "fault": raw.with_suffix(".fault.json").as_posix(), "resume": raw.with_suffix(".resume.json").as_posix(), "command": command, "cuda_visible_devices": expected_uuid}}
    document = {"schema": QUALIFICATION_SCHEMA, "status": "PASS_GPU_KERNEL_AND_INHERITED_RUNTIME_IMPORTS", "host": socket.gethostname(), "gpu_uuid": expected_uuid,
                "source_sha256": source_sha256, "workload": {"member_identities": identities, "updates": 1000, "kind": "formal_bundle", "shared_preparation": "frozen_provider_only", "common_recovery": True},
                "runtime_environment": {"interpreter": interpreter, "environment": environment}, "resource_envelope": {"device_memory_bytes": gpu_memory_bytes(args.physical_gpu), "required_memory_bytes": resumed["peak_reserved_memory_bytes"]}, "measurements": measurements,
                "valid_from_utc_epoch": finished, "valid_until_utc_epoch": finished + args.validity_seconds, "qualification_profile": {"path": str(profile_path), "sha256": args.profile_sha256}}
    _write_once(Path(args.output).absolute(), document); print(json.dumps({"status": document["status"], "output": str(Path(args.output).absolute()), "sha256": _sha(Path(args.output).absolute())}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="mode", required=True)
    prepare_parser = sub.add_parser("prepare"); prepare_parser.add_argument("--diagnostic-bundle", required=True); prepare_parser.add_argument("--diagnostic-bundle-sha256", required=True); prepare_parser.add_argument("--caption-longest-report", required=True); prepare_parser.add_argument("--caption-longest-report-sha256", required=True); prepare_parser.add_argument("--caption-longest-source-root", required=True); prepare_parser.add_argument("--caption-longest-candidates", required=True); prepare_parser.add_argument("--caption-longest-candidates-sha256", required=True); prepare_parser.add_argument("--detection-xd-longest-report", required=True); prepare_parser.add_argument("--detection-xd-longest-report-sha256", required=True); prepare_parser.add_argument("--detection-xd-longest-source-root", required=True); prepare_parser.add_argument("--detection-ucf-longest-report", required=True); prepare_parser.add_argument("--detection-ucf-longest-report-sha256", required=True); prepare_parser.add_argument("--detection-ucf-longest-source-root", required=True); prepare_parser.add_argument("--output-root", required=True); prepare_parser.set_defaults(action=prepare)
    qualification = sub.add_parser("qualify"); qualification.add_argument("--profile", required=True); qualification.add_argument("--profile-sha256", required=True); qualification.add_argument("--runtime-source-root", required=True); qualification.add_argument("--raw-output", required=True); qualification.add_argument("--physical-gpu", type=int, required=True); qualification.add_argument("--interpreter", required=True); qualification.add_argument("--execution-environment", required=True); qualification.add_argument("--validity-seconds", type=float, required=True); qualification.add_argument("--fault-after-group", choices=GROUPS, default="U"); qualification.add_argument("--output", required=True); qualification.set_defaults(action=qualify)
    args = parser.parse_args();
    if getattr(args, "validity_seconds", 1) <= 0: parser.error("validity must be positive")
    args.action(args)


if __name__ == "__main__": main()
