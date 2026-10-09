"""Concrete, fail-closed composition for an NC-RTED training run.

The JSON manifest is deliberately an input binding, not a convenience settings
file.  ``preflight`` performs only file/hash/schema checks.  ``assemble`` is
the first point at which inherited Python, CUDA, or model weights may load.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Iterator, Mapping

from .task_inputs import TaskInputError, TrainingCatalog, InheritedTaskTokenizer, sha256_file


SCHEMA = "nc_rted_production_runtime/v1"
_HASH = "0123456789abcdef"
_SAMPLING = {"local_num_frames": 1, "frames_upbound": 64, "frames_lowbound": 4,
             "sample_type": "dynamic_fps1", "time_msg": "short_online_v2",
             "model_max_length": 8192, "vision_chunk_size": 32,
             "projector": "original"}
_REQUIRED_EXPORTS = {"config.json", "adapter_config.json", "adapter_model.safetensors", "non_lora_trainables.bin"}
_REQUIRED_REACTVAU_SOURCES = {"llava/train/train.py", "llava/train/reactvau_stage2_cache_adapter.py",
                              "llava/model/llava_arch.py", "llava/model/multimodal_encoder/siglip_encoder.py",
                              "llava/model/multimodal_projector/memory_manager.py",
                              "llava/model/language_model/llava_qwen.py", "eval_utils/vad/eval_reactvau_detection.py",
                              "eval_utils/vad/detect_utils.py", "vad/get_prompt.py"}


class ProductionRuntimeError(TaskInputError):
    pass


def _is_sha(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in _HASH for char in value)


def _mapping(value: object, name: str) -> dict:
    if not isinstance(value, dict):
        raise ProductionRuntimeError(f"{name} must be an object")
    return value


def _bound_file(value: object, expected: object, name: str) -> Path:
    if not isinstance(value, str) or not _is_sha(expected):
        raise ProductionRuntimeError(f"{name} needs an absolute path and SHA-256")
    path = Path(value)
    if not path.is_absolute() or not path.is_file() or sha256_file(path) != expected:
        raise ProductionRuntimeError(f"{name} is absent or its SHA-256 differs")
    return path


def _tree_sha256(root: Path) -> str:
    """Stable digest for a local model snapshot without trusting a directory mtime."""
    digest = hashlib.sha256()
    if not root.is_dir() or root.is_symlink():
        raise ProductionRuntimeError("bound snapshot is not a real directory")
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not path.is_file():
            if path.is_symlink():
                raise ProductionRuntimeError("bound snapshot contains a symlink")
            continue
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _bound_tree(value: object, expected: object, name: str) -> Path:
    if not isinstance(value, str) or not _is_sha(expected):
        raise ProductionRuntimeError(f"{name} needs an absolute path and SHA-256")
    path = Path(value)
    if not path.is_absolute() or _tree_sha256(path) != expected:
        raise ProductionRuntimeError(f"{name} is absent or its tree SHA-256 differs")
    return path


@dataclass(frozen=True)
class RuntimeManifest:
    path: Path
    config_sha256: str
    document: dict

    @property
    def run(self) -> dict:
        return self.document["run"]


def load_manifest(path: str | Path, *, expected_sha256: str) -> RuntimeManifest:
    config = _bound_file(str(Path(path).absolute()), expected_sha256, "runtime config")
    try:
        document = json.loads(config.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ProductionRuntimeError("runtime config is not valid JSON") from error
    manifest = RuntimeManifest(config, expected_sha256, _mapping(document, "runtime config"))
    preflight(manifest)
    return manifest


def load_formal_admission(path: str | Path, *, expected_sha256: str) -> dict:
    path = _bound_file(str(Path(path).absolute()), expected_sha256, "formal admission")
    try: admission = _mapping(json.loads(path.read_text()), "formal admission")
    except (OSError, ValueError) as error: raise ProductionRuntimeError("formal admission is invalid JSON") from error
    checks = admission.get("engineering_checks")
    if (admission.get("status") != "PASS" or admission.get("formal_execution_allowed") is not True or
            not isinstance(checks, dict) or set(checks) != {str(index) for index in range(1, 11)} or
            any(item != "PASS" for item in checks.values()) or not isinstance(admission.get("source_files"), dict)):
        raise ProductionRuntimeError("formal admission does not contain all ten accepted gates")
    return admission


def preflight(manifest: RuntimeManifest) -> None:
    """Validate every declared asset before importing Transformers or ReactVAU."""
    doc = manifest.document
    if doc.get("schema") != SCHEMA:
        raise ProductionRuntimeError("unsupported production runtime schema")
    run, hashes = _mapping(doc.get("run"), "run"), _mapping(doc.get("hashes"), "hashes")
    required_run = {"run_id", "group", "seed", "device", "checkpoint_root", "progress_path", "mode"}
    if not required_run.issubset(run) or run.get("group") not in {"A", "U", "S", "F"}:
        raise ProductionRuntimeError("run identity is incomplete")
    if run.get("mode") not in {"diagnostic", "formal"} or not isinstance(run.get("seed"), int):
        raise ProductionRuntimeError("run mode or seed is invalid")
    if run["mode"] == "diagnostic":
        if not str(run["run_id"]).startswith("diagnostic:") or type(run.get("diagnostic_updates")) is not int or run["diagnostic_updates"] < 1:
            raise ProductionRuntimeError("diagnostic runs need diagnostic identity and update count")
    elif str(run["run_id"]).startswith("diagnostic:"):
        raise ProductionRuntimeError("formal runs need a non-diagnostic identity")
    for key in ("code_sha256", "runtime_sha256", "inherited_weights_sha256"):
        if not _is_sha(hashes.get(key)):
            raise ProductionRuntimeError(f"hashes.{key} is invalid")
    sampling = _mapping(doc.get("sampling"), "sampling")
    if sampling != _SAMPLING:
        raise ProductionRuntimeError("inherited sampling/projector contract differs")
    catalog = _mapping(doc.get("catalog"), "catalog")
    _bound_file(catalog.get("training_annotations"), catalog.get("training_annotations_sha256"), "training annotations")
    _bound_file(catalog.get("provenance"), catalog.get("provenance_sha256"), "manifest provenance")
    dataset_yaml = _bound_file(catalog.get("dataset_yaml"), catalog.get("dataset_yaml_sha256"), "Stage2 dataset YAML")
    subset = _bound_file(catalog.get("caption_subset"), catalog.get("caption_subset_sha256"), "fixed caption subset")
    pg_scores = _bound_file(catalog.get("pg_scores"), catalog.get("pg_scores_sha256"), "original PG scores")
    # The original LazySupervisedDataset only loads a YAML dataset declaration;
    # do not silently pass its JSON instruction input as a replacement.
    try:
        import yaml
        dataset_document = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
        datasets = dataset_document.get("datasets") if isinstance(dataset_document, dict) else None
        paths = [entry.get("json_path") for entry in datasets] if isinstance(datasets, list) else []
    except (ImportError, OSError, ValueError, TypeError, AttributeError) as error:
        raise ProductionRuntimeError("Stage2 dataset YAML is invalid") from error
    if paths != [str(subset)] or dataset_document.get("pg_scores_path") != str(pg_scores):
        raise ProductionRuntimeError("Stage2 dataset YAML does not bind fixed captions and PG scores")
    if not isinstance(catalog.get("manifest_directory"), str) or not Path(catalog["manifest_directory"]).is_absolute():
        raise ProductionRuntimeError("manifest directory must be absolute")
    inherited = _mapping(doc.get("inherited"), "inherited")
    if not isinstance(inherited.get("external_root"), str) or not Path(inherited["external_root"]).is_absolute():
        raise ProductionRuntimeError("ReactVAU source root must be absolute")
    source_manifest = _bound_file(inherited.get("source_manifest"), inherited.get("source_manifest_sha256"), "ReactVAU source manifest")
    try:
        source_files = _mapping(json.loads(source_manifest.read_text()), "ReactVAU source manifest").get("files")
        if not isinstance(source_files, dict) or not source_files:
            raise ProductionRuntimeError("ReactVAU source manifest has no files")
        for relative, expected in source_files.items():
            if Path(relative).is_absolute() or ".." in Path(relative).parts:
                raise ProductionRuntimeError("ReactVAU source manifest path escapes root")
            _bound_file(str(Path(inherited["external_root"]) / relative), expected, "ReactVAU source file")
    except (OSError, ValueError, TypeError) as error:
        raise ProductionRuntimeError("ReactVAU source manifest is invalid") from error
    _validate_source_coverage(inherited)
    _bound_tree(inherited.get("base_directory"), inherited.get("base_directory_sha256"), "inherited base model")
    if not isinstance(inherited.get("export_directory"), str) or not Path(inherited["export_directory"]).is_absolute():
        raise ProductionRuntimeError("inherited export directory must be absolute")
    _bound_tree(inherited.get("tokenizer_directory"), inherited.get("tokenizer_sha256"), "inherited tokenizer")
    exports = _mapping(inherited.get("export_hashes"), "inherited.export_hashes")
    if set(exports) != _REQUIRED_EXPORTS or not all(_is_sha(value) for value in exports.values()):
        raise ProductionRuntimeError("inherited export hashes are incomplete")
    for relative, expected in exports.items():
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ProductionRuntimeError("inherited export file escapes export directory")
        _bound_file(str(Path(inherited["export_directory"]) / relative), expected, "inherited export file")
    stage2 = _mapping(doc.get("stage2_cache"), "stage2_cache")
    mode = stage2.get("mode", "resolver")
    if mode == "resolver":
        _bound_file(stage2.get("module"), stage2.get("module_sha256"), "Stage2 resolver module")
        config = _bound_file(stage2.get("config"), stage2.get("config_sha256"), "Stage2 cache config")
        if stage2.get("accepted_status") != "APPROVED_FOR_EXECUTION":
            raise ProductionRuntimeError("Stage2 cache is not approved for execution")
        try: status = json.loads(config.read_text()).get("status")
        except (OSError, ValueError) as error: raise ProductionRuntimeError("Stage2 cache config is invalid") from error
        if status != stage2["accepted_status"]: raise ProductionRuntimeError("Stage2 cache status is not accepted")
    elif mode != "materialized":
        raise ProductionRuntimeError("Stage2 cache mode is unsupported")
    fast = _mapping(doc.get("fast"), "fast")
    _bound_file(fast.get("snapshot"), fast.get("snapshot_sha256"), "Fast snapshot")
    identity = _mapping(fast.get("identity"), "Fast identity")
    if not identity.get("checkpoint") or not identity.get("implementation"):
        raise ProductionRuntimeError("Fast identity is incomplete")
    _protocols(fast.get("protocols"))
    media = _mapping(doc.get("media"), "media")
    _bound_file(media.get("catalog"), media.get("catalog_sha256"), "media catalog")
    if type(media.get("observation_cache_max_bytes")) is not int or media["observation_cache_max_bytes"] < 1:
        raise ProductionRuntimeError("observation cache bound is invalid")
    if not isinstance(media.get("observation_cache_root"), str) or not Path(media["observation_cache_root"]).is_absolute():
        raise ProductionRuntimeError("observation cache root must be absolute")
    _media_catalog(Path(media["catalog"]))
    detector = _mapping(doc.get("detector"), "detector")
    _bound_tree(detector.get("snapshot"), detector.get("snapshot_sha256"), "RT-DETR snapshot")
    _bound_tree(detector.get("siglip_snapshot"), detector.get("siglip_snapshot_sha256"), "SigLIP snapshot")
    if not isinstance(detector.get("score_threshold"), (int, float)) or not 0 <= detector["score_threshold"] <= 1:
        raise ProductionRuntimeError("RT-DETR threshold is invalid")
    teacher = _mapping(doc.get("teacher"), "teacher")
    if not isinstance(teacher.get("artifact"), str) or not _is_sha(teacher.get("sha256")):
        raise ProductionRuntimeError("teacher artifact needs an absolute path and SHA-256")
    teacher_path = Path(teacher["artifact"])
    if not teacher_path.is_absolute():
        raise ProductionRuntimeError("teacher artifact path must be absolute")
    # A pipeline JSON binds directly. A committed teacher store binds its signed
    # index; ``build_teachers_from_store`` verifies every referenced chunk.
    if teacher_path.is_dir():
        _bound_file(str(teacher_path / "index.json"), teacher["sha256"], "teacher store index")
    else:
        _bound_file(str(teacher_path), teacher["sha256"], "teacher artifact")


def _load_external(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ProductionRuntimeError(f"cannot load {name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _media_catalog(path: Path):
    from .media_observer import BoundMedia
    rows = json.loads(path.read_text())
    if isinstance(rows, dict): rows = rows.get("media")
    if not isinstance(rows, list) or not rows:
        raise ProductionRuntimeError("media catalog must contain media rows")
    result = {}
    for row in rows:
        if not isinstance(row, dict): raise ProductionRuntimeError("invalid media catalog row")
        item = BoundMedia(**{key: row.get(key) for key in BoundMedia.__dataclass_fields__})
        item.validate()
        aliases = row.get("aliases", [])
        if not isinstance(aliases, list) or any(not isinstance(x, str) or not x for x in aliases):
            raise ProductionRuntimeError("media aliases are invalid")
        for key in [item.media_key, *aliases]:
            identity = (item.dataset, key)
            if identity in result: raise ProductionRuntimeError("duplicate media catalog identity")
            result[identity] = item if key == item.media_key else BoundMedia(**{**item.__dict__, "media_key": key})
    return result


def _protocols(value: object):
    from .detection_provider import DetectionProtocol
    raw = _mapping(value, "Fast protocols")
    result = {}
    for dataset, config in raw.items():
        protocol = DetectionProtocol(**_mapping(config, f"protocol {dataset}")); protocol.validate(); result[dataset] = protocol
    if set(result) != {"ucf-crime", "xd-violence"}: raise ProductionRuntimeError("Fast protocols must bind both datasets")
    return result


def _materialized_stage2_cache(media):
    """Original Stage2 cache interface over immutable, hash-bound media files."""
    from .media_observer import lease_verified_media
    class Cache:
        def request_index_for(self, annotation):
            key = annotation.get("_reactvau_relative_video")
            matches = [item for (dataset, bound_key), item in media.items() if bound_key == key]
            if len(matches) != 1 or matches[0].request_index is None:
                raise ProductionRuntimeError("materialized Stage2 media has no bound request index")
            return matches[0].request_index
        @contextmanager
        def acquire(self, relative_path, request_index):
            matches = [item for (dataset, key), item in media.items() if key == relative_path and item.request_index == request_index]
            if len(matches) != 1: raise ProductionRuntimeError("materialized Stage2 media binding is ambiguous")
            with lease_verified_media(matches[0]) as path: yield path
    return Cache()


def _stage2_dataset_class(original, cache):
    """Build a run-local fail-closed subclass without mutating ReactVAU globals."""
    from llava.train.reactvau_stage2_cache_adapter import FailClosedStage2DatasetMixin
    class Stage2Dataset(FailClosedStage2DatasetMixin, original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if self.pg_scores_dict is None:
                raise ProductionRuntimeError("Stage2 requires original PG scores")
            self.stage2_cache = cache
            self._bind_stage2_requests()
    return Stage2Dataset


def _validate_caption_subset(catalog: TrainingCatalog, path: Path) -> None:
    try: rows = json.loads(path.read_text())
    except (OSError, ValueError) as error: raise ProductionRuntimeError("caption subset is invalid JSON") from error
    if not isinstance(rows, list) or len(rows) != 2000:
        raise ProductionRuntimeError("caption subset must contain exactly 2,000 rows")
    expected = {(task.instruction["id"], task.instruction["video"]): task
                for task in catalog.tasks.values() if task.task == "caption" and task.instruction is not None}
    actual = set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), (int, str)) or not isinstance(row.get("video"), str):
            raise ProductionRuntimeError("caption subset row has no original identity")
        identity = (row["id"], row["video"])
        task = expected.get(identity)
        if task is None or row != task.instruction:
            raise ProductionRuntimeError("caption subset row differs from its original catalog instruction")
        actual.add(identity)
    if actual != set(expected) or len(actual) != len(rows):
        raise ProductionRuntimeError("caption subset differs from the fixed catalog captions")


def _validate_source_coverage(inherited: dict) -> None:
    source_manifest = Path(inherited["source_manifest"])
    document = _mapping(json.loads(source_manifest.read_text()), "ReactVAU source manifest")
    files = document.get("files")
    root = Path(inherited["external_root"])
    required = {str(path.relative_to(root)) for directory in (root / "llava", root / "eval_utils", root / "vad")
                if directory.is_dir() for path in directory.rglob("*.py")}
    if (not isinstance(files, dict) or not _REQUIRED_REACTVAU_SOURCES.issubset(files) or
            required != {name for name in files if name.startswith(("llava/", "eval_utils/", "vad/"))}):
        raise ProductionRuntimeError("ReactVAU source manifest omits runtime-critical sources")


def _bound_source_files(inherited: dict) -> dict[str, str]:
    document = _mapping(json.loads(Path(inherited["source_manifest"]).read_text()), "ReactVAU source manifest")
    files = document.get("files")
    if not isinstance(files, dict): raise ProductionRuntimeError("ReactVAU source manifest has no file map")
    return files


def _assert_inherited_modules_bound(inherited: dict, *, verify_hashes: bool = True) -> None:
    root = Path(inherited["external_root"]).resolve()
    files = _bound_source_files(inherited)
    for name, module in tuple(sys.modules.items()):
        if name not in {"detect_utils", "vad"} and not name.startswith(("llava", "eval_utils", "vad.")):
            continue
        location = getattr(module, "__file__", None)
        if location is not None:
            try: relative = str(Path(location).resolve().relative_to(root))
            except ValueError as error: raise ProductionRuntimeError(f"preloaded inherited module resolves outside bound source: {name}") from error
            if name == "detect_utils" and relative != "eval_utils/vad/detect_utils.py":
                raise ProductionRuntimeError("detect_utils does not resolve to the bound VAD helper")
            expected = files.get(relative)
            if expected is None: raise ProductionRuntimeError(f"inherited module is absent from bound source manifest: {name}")
            if verify_hashes and sha256_file(root / relative) != expected:
                raise ProductionRuntimeError(f"bound inherited module changed: {name}")
        namespace = getattr(module, "__path__", None)
        if namespace is not None:
            paths = list(namespace)
            if not paths: raise ProductionRuntimeError(f"inherited namespace has no bound path: {name}")
            for entry in paths:
                try: Path(entry).resolve().relative_to(root)
                except ValueError as error: raise ProductionRuntimeError(f"inherited namespace resolves outside bound source: {name}") from error


def _import_bound_inherited_runtime(inherited: dict) -> None:
    root = Path(inherited["external_root"]).resolve()
    vad_directory = root / "eval_utils" / "vad"
    if not vad_directory.is_dir(): raise ProductionRuntimeError("bound VAD helper directory is absent")
    _assert_inherited_modules_bound(inherited)
    if str(vad_directory) not in sys.path: sys.path.insert(0, str(vad_directory))
    for name in ("llava.train.train", "llava.train.reactvau_stage2_cache_adapter", "llava.model.llava_arch",
                 "llava.model.multimodal_encoder.siglip_encoder", "llava.model.multimodal_projector.memory_manager",
                 "llava.model.language_model.llava_qwen", "detect_utils", "vad.get_prompt",
                 "eval_utils.vad.eval_reactvau_detection"):
        importlib.import_module(name)
    _assert_inherited_modules_bound(inherited)


def _validate_detection_bindings(catalog: TrainingCatalog, fast_snapshot: Path, media_catalog: dict) -> None:
    try: document = json.loads(fast_snapshot.read_text())
    except (OSError, ValueError) as error: raise ProductionRuntimeError("Fast snapshot is invalid JSON") from error
    if document.get("schema") != "nc_rted_frozen_fast/v1" or not isinstance(document.get("media"), list):
        raise ProductionRuntimeError("Fast snapshot schema is invalid")
    fast = {}
    for row in document["media"]:
        if not isinstance(row, dict): raise ProductionRuntimeError("Fast snapshot media row is invalid")
        identity = (row.get("dataset"), row.get("media_key"))
        if identity in fast: raise ProductionRuntimeError("Fast snapshot has duplicate media identity")
        fast[identity] = row
    for task in catalog.tasks.values():
        if task.task != "detection" or task.query_index is None or task.observed_seconds is None:
            continue
        identity = (task.dataset, task.media_key)
        row, bound = fast.get(identity), media_catalog.get(identity)
        if row is None or bound is None:
            raise ProductionRuntimeError("selected detection media is absent from Fast snapshot or observer catalog")
        if (row.get("media_sha256") != bound.media_sha256 or row.get("fps") != bound.fps or
                row.get("frame_count") != bound.frame_count or row.get("height") != bound.height or row.get("width") != bound.width):
            raise ProductionRuntimeError("Fast snapshot and observer catalog bind different detection media")
        queries = row.get("queries")
        if not isinstance(queries, list) or task.query_index >= len(queries):
            raise ProductionRuntimeError("selected detection query is absent from Fast snapshot")
        query = queries[task.query_index]
        indices = query.get("frame_indices") if isinstance(query, dict) else None
        if (query.get("index") != task.query_index or not isinstance(indices, list) or not indices or
                not math.isclose(indices[-1] / float(row["fps"]), task.observed_seconds, rel_tol=0., abs_tol=1e-6)):
            raise ProductionRuntimeError("Fast query endpoint differs from selected detection task")


@contextmanager
def _bound_stage2_constructor_environment(stage2: dict, media_catalog_path: str):
    """Tell the original loader its first-three media check is cache-bound, briefly."""
    binding = stage2.get("config") if stage2.get("mode", "resolver") == "resolver" else media_catalog_path
    previous = os.environ.get("REACTVAU_STAGE2_CACHE_CONFIG")
    os.environ["REACTVAU_STAGE2_CACHE_CONFIG"] = str(binding)
    try:
        yield
    finally:
        if previous is None: os.environ.pop("REACTVAU_STAGE2_CACHE_CONFIG", None)
        else: os.environ["REACTVAU_STAGE2_CACHE_CONFIG"] = previous


def _validate_formal_admission_before_models(admission: dict, identity: dict) -> None:
    """Duplicate the worker's immutable admission checks before model allocation."""
    if admission.get("run_identity") != identity:
        raise ProductionRuntimeError("formal admission is not bound to this exact run")
    files = admission.get("source_files")
    source_digest = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if source_digest != identity["code_sha256"]:
        raise ProductionRuntimeError("formal admission source manifest differs from run identity")
    required = {str(path.resolve()) for path in Path(__file__).parent.glob("*.py")}
    if not required.issubset(files):
        raise ProductionRuntimeError("formal admission omits NC-RTED source")
    for name, expected in files.items():
        if sha256_file(name) != expected: raise ProductionRuntimeError("accepted implementation changed")


@dataclass
class ProductionRuntime:
    manifest: RuntimeManifest
    worker: Any
    admission: dict | None = None

    def run(self) -> dict:
        run = self.manifest.run
        return self.worker.run(progress_path=run["progress_path"], admission=self.admission,
                               diagnostic_updates=run.get("diagnostic_updates") if run["mode"] == "diagnostic" else None)


def assemble(manifest: RuntimeManifest, *, admission: dict | None = None) -> ProductionRuntime:
    """Load the exact inherited runtime after ``preflight`` has succeeded."""
    preflight(manifest)
    doc, run = manifest.document, manifest.run
    # These fixed data/teacher bindings are validated before ReactVAU, CUDA, or
    # model construction. Their readers only inspect committed JSON/NPZ inputs.
    from .teacher_store import build_teachers_from_store
    from .train_worker import TeacherIndex
    catalog = TrainingCatalog.load(doc["catalog"]["manifest_directory"], doc["catalog"]["training_annotations"], expected_provenance_sha256=doc["catalog"]["provenance_sha256"])
    teacher_path = Path(doc["teacher"]["artifact"])
    teachers = (TeacherIndex(build_teachers_from_store(teacher_path), catalog, identity=doc["teacher"]["sha256"])
                if teacher_path.is_dir() else TeacherIndex.load(teacher_path, catalog, expected_sha256=doc["teacher"]["sha256"]))
    identity = {"run_id": run["run_id"], "group": run["group"], "seed": str(run["seed"]), "code_sha256": doc["hashes"]["code_sha256"], "config_sha256": manifest.config_sha256, "data_sha256": catalog.identity, "teacher_sha256": doc["teacher"]["sha256"], "inherited_weights_sha256": doc["hashes"]["inherited_weights_sha256"], "runtime_sha256": doc["hashes"]["runtime_sha256"]}
    if run["mode"] == "formal":
        if admission is None: raise ProductionRuntimeError("formal assembly requires an external formal admission")
        _validate_formal_admission_before_models(admission, identity)
    elif admission is not None: raise ProductionRuntimeError("diagnostic assembly cannot receive formal admission")
    _validate_caption_subset(catalog, Path(doc["catalog"]["caption_subset"]))
    inherited, stage2 = doc["inherited"], doc["stage2_cache"]
    root = Path(inherited["external_root"])
    _assert_inherited_modules_bound(inherited)
    if str(root) not in sys.path: sys.path.insert(0, str(root))
    _import_bound_inherited_runtime(inherited)
    # Install the original fail-closed subclass before constructing the dataset.
    media = _media_catalog(Path(doc["media"]["catalog"]))
    _validate_detection_bindings(catalog, Path(doc["fast"]["snapshot"]), media)
    if stage2.get("mode", "resolver") == "materialized":
        cache = _materialized_stage2_cache(media)
    else:
        resolver = _load_external(Path(stage2["module"]), "nc_rted_stage2_resolver")
        cache = resolver.Cache(json.loads(Path(stage2["config"]).read_text()))
    from llava.train import train as train_module
    from llava import conversation as conversation_lib
    _assert_inherited_modules_bound(inherited)
    original = train_module.LazySupervisedDataset
    Stage2Dataset = _stage2_dataset_class(original, cache)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(inherited["tokenizer_directory"], local_files_only=True, model_max_length=8192)
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = "right"
    try: conversation_lib.default_conversation = conversation_lib.conv_templates["qwen_2"]
    except KeyError as error: raise ProductionRuntimeError("inherited qwen_2 conversation template is unavailable") from error
    data_args = train_module.DataArguments(data_path=doc["catalog"]["dataset_yaml"], lazy_preprocess=True,
        frames_upbound=64, frames_lowbound=4, local_num_frames=1, sample_type="dynamic_fps1", time_msg="short_online_v2")
    bridge, _ = __import__("nc_rted.train_worker", fromlist=["build_training_bridge"]).build_training_bridge(
        inherited["base_directory"], inherited["export_directory"], export_hashes=inherited["export_hashes"], seed=run["seed"], device=run["device"])
    slow = bridge.slow
    raw = slow.get_base_model() if hasattr(slow, "get_base_model") else slow
    tower = slow.get_vision_tower()
    if not getattr(tower, "is_loaded", False):
        tower.vision_tower_name = str(Path(doc["detector"]["siglip_snapshot"]).resolve())
        tower.load_model()
    import torch
    projector_dtype = next(raw.get_model().mm_projector.parameters()).dtype
    tower.to(device=run["device"], dtype=projector_dtype)
    tower.requires_grad_(False); tower.eval()
    data_args.image_processor = tower.image_processor
    data_args.is_multimodal = True
    raw.config.frame_aspect_ratio = data_args.frame_aspect_ratio
    raw.config.time_msg_type = data_args.time_msg
    raw.config.tokenizer_model_max_length = 8192
    raw.config.vision_encode_type = "image_video_memory_batch"
    if getattr(raw.config, "mm_local_num_frames", None) != 1:
        raise ProductionRuntimeError("loaded inherited projector does not retain local_num_frames=1")
    from .detector import FrozenRTDetr, InheritedSigLipAdapter
    from .observation_cache import FrozenFrameCache
    from .media_observer import CausalMediaObserver
    from .detection_media import OpenCVFrames, StreamingDetectionReader
    from .detection_provider import FrozenDetectionProvider
    from .caption_provider import Stage2CaptionProvider
    from .train_worker import CheckpointStore, TeacherIndex, TrainingWorker
    detector = FrozenRTDetr(Path(doc["detector"]["snapshot"]), device=run["device"], score_threshold=doc["detector"]["score_threshold"])
    siglip = InheritedSigLipAdapter(tower, Path(doc["detector"]["siglip_snapshot"]))
    observer = CausalMediaObserver(detector=detector, siglip=siglip, cache=FrozenFrameCache(doc["media"]["observation_cache_root"], doc["media"]["observation_cache_max_bytes"]), media_catalog=media,
        lease_resolver=lambda item: cache.acquire(item.media_key, item.request_index), decoder_factory=OpenCVFrames)
    with _bound_stage2_constructor_environment(stage2, doc["media"]["catalog"]):
        dataset = Stage2Dataset(doc["catalog"]["dataset_yaml"], tokenizer, data_args)
    captions = Stage2CaptionProvider(catalog=catalog, dataset=dataset, model=slow, vision_tower=tower, observer=observer)
    protocols = _protocols(doc["fast"]["protocols"])
    reader = StreamingDetectionReader(doc["fast"]["snapshot"], snapshot_sha256=doc["fast"]["snapshot_sha256"], fast_identity=doc["fast"]["identity"], protocols=protocols, encode=siglip)
    detections = FrozenDetectionProvider(slow, catalog, protocols, reader=reader, observation_reader=observer)
    def provider(sample_id):
        # IncrementalTrainer calls bridge.train(), which recursively toggles the
        # inherited tower. Restore its frozen inference mode before both paths.
        _assert_inherited_modules_bound(inherited, verify_hashes=False)
        tower.eval()
        return captions(sample_id) if catalog.tasks[sample_id].task == "caption" else detections(sample_id)
    worker = TrainingWorker(bridge, catalog, teachers, InheritedTaskTokenizer.from_original(tokenizer, data_args), provider,
        CheckpointStore(run["checkpoint_root"], identity), group=run["group"], seed=run["seed"])
    tower.eval()
    return ProductionRuntime(manifest, worker, admission)
