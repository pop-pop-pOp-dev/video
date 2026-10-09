import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import types

import pytest

from nc_rted.media_observer import BoundMedia
from nc_rted.production_runtime import (ProductionRuntimeError, _bound_stage2_constructor_environment,
                                        _materialized_stage2_cache, _stage2_dataset_class,
                                        _validate_caption_subset, _validate_detection_bindings,
                                        _assert_inherited_modules_bound, _configure_inherited_tokenizer,
                                        _configure_inherited_data_args, _require_loaded_final_stage2_tower, load_manifest)
from nc_rted.task_inputs import TrainingCatalog, TrainingTask


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_digest(root: Path) -> str:
    out = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file():
            out.update(str(path.relative_to(root)).encode()); out.update(b"\0")
            out.update(digest(path).encode()); out.update(b"\n")
    return out.hexdigest()


def file(root: Path, name: str, value=b"x") -> tuple[str, str]:
    path = root / name; path.write_bytes(value); return str(path), digest(path)


def directory(root: Path, name: str) -> tuple[str, str]:
    path = root / name; path.mkdir(); (path / "payload").write_text(name)
    return str(path), tree_digest(path)


def config(tmp_path: Path, *, mode="diagnostic") -> tuple[Path, str]:
    from safetensors.torch import save_file
    import torch
    annotations, annotations_sha = file(tmp_path, "train.json", b"[]")
    provenance, provenance_sha = file(tmp_path, "provenance.json", b"{}")
    subset, subset_sha = file(tmp_path, "captions.json", b"[]")
    pg_scores, pg_scores_sha = file(tmp_path, "pg.json", b"{}")
    dataset_yaml, dataset_yaml_sha = file(tmp_path, "dataset.yaml", f"datasets:\n  - json_path: {subset}\npg_scores_path: {pg_scores}\n".encode())
    stage_module, stage_module_sha = file(tmp_path, "stage.py", b"class Cache: pass\n")
    stage_config, stage_config_sha = file(tmp_path, "stage.json", b'{"status":"APPROVED_FOR_EXECUTION"}')
    fast, fast_sha = file(tmp_path, "fast.json", b"{}")
    teacher, teacher_sha = file(tmp_path, "teacher.store", b"teacher")
    media_rows = [{"dataset":"ucf-crime", "media_key":"m", "media_path":"/media/m", "media_sha256":"a" * 64,
                   "fps": 1.0, "frame_count": 1, "height": 1, "width": 1, "aliases":["caption-m"]}]
    media, media_sha = file(tmp_path, "media.json", json.dumps(media_rows).encode())
    external, _ = directory(tmp_path, "external")
    source_files = {}
    for name in ("llava/train/train.py", "llava/train/reactvau_stage2_cache_adapter.py", "llava/model/llava_arch.py",
                 "llava/model/multimodal_encoder/siglip_encoder.py", "llava/model/multimodal_projector/memory_manager.py",
                 "llava/model/language_model/llava_qwen.py", "eval_utils/vad/eval_reactvau_detection.py",
                 "eval_utils/vad/detect_utils.py", "vad/get_prompt.py"):
        parent = Path(external) / Path(name).parent; parent.mkdir(parents=True, exist_ok=True)
        _, source_files[name] = file(parent, Path(name).name)
    source_manifest, source_manifest_sha = file(tmp_path, "source-manifest.json", json.dumps({"files":source_files}).encode())
    base, base_sha = directory(tmp_path, "base")
    export, _ = directory(tmp_path, "export")
    export_hashes = {}
    for name in ("config.json", "adapter_config.json", "adapter_model.safetensors", "non_lora_trainables.bin"):
        _, export_hashes[name] = file(Path(export), name)
    tokenizer, tokenizer_sha = directory(tmp_path, "tokenizer")
    rtdetr, rtdetr_sha = directory(tmp_path, "rtdetr")
    siglip, _ = directory(tmp_path, "siglip")
    raw_config = b'{"model_type":"siglip"}'
    (Path(siglip) / "config.json").write_bytes(raw_config)
    siglip_sha = tree_digest(Path(siglip))
    final_siglip = tmp_path / "final-stage2-siglip"; final_siglip.mkdir()
    (final_siglip / "config.json").write_bytes(raw_config)
    final_tensors = {f"vision_model.test.{index}": __import__("torch").tensor([index], dtype=__import__("torch").bfloat16)
                     for index in range(421)}
    parent_export = Path(export) / "non_lora_trainables.bin"
    torch.save({"base_model.model.model.vision_tower.vision_tower." + name: value for name, value in final_tensors.items()}, parent_export)
    export_hashes["non_lora_trainables.bin"] = digest(parent_export)
    save_file(final_tensors, str(final_siglip / "model.safetensors"), metadata={"format":"pt"})
    final_files = {name: digest(final_siglip / name) for name in ("config.json", "model.safetensors")}
    final_provenance = {"schema":"nc_rted_final_stage2_vision/v1", "source_export_sha256":export_hashes["non_lora_trainables.bin"],
                        "source_key_prefix":"base_model.model.model.vision_tower.vision_tower.", "tensor_count":421,
                        "tensor_dtype":"bfloat16", "raw_config_sha256":final_files["config.json"], "files":final_files,
                        "source_key_map":{name:"base_model.model.model.vision_tower.vision_tower." + name for name in final_tensors}}
    (final_siglip / "nc_rted_provenance.json").write_text(json.dumps(final_provenance))
    final_siglip, final_siglip_sha = str(final_siglip), tree_digest(final_siglip)
    protocol = {"question_template":"Is there an anomaly?", "prompt_style":"neutral", "time_message_style":"short_online_v2",
                "memory_enhancement":True, "rt_anomaly":True, "trigger_threshold":0.5, "pool_threshold":0.5, "scoring":"yesno"}
    run_id = "diagnostic:runtime-test" if mode == "diagnostic" else "formal-runtime-test"
    run = {"run_id":run_id,"group":"A","seed":17,"device":"cpu","checkpoint_root":str(tmp_path / "checkpoints"),
           "progress_path":str(tmp_path / "progress.json"),"mode":mode}
    if mode == "diagnostic": run["diagnostic_updates"] = 1
    doc = {"schema":"nc_rted_production_runtime/v1", "run":run,
      "hashes":{"code_sha256":"1"*64,"runtime_sha256":"2"*64,"inherited_weights_sha256":"3"*64},
      "sampling":{"local_num_frames":1,"frames_upbound":64,"frames_lowbound":4,"sample_type":"dynamic_fps1","time_msg":"short_online_v2","model_max_length":8192,"vision_chunk_size":32,"projector":"original"},
      "catalog":{"manifest_directory":str(tmp_path),"training_annotations":annotations,"training_annotations_sha256":annotations_sha,"provenance":provenance,"provenance_sha256":provenance_sha,"dataset_yaml":dataset_yaml,"dataset_yaml_sha256":dataset_yaml_sha,"caption_subset":subset,"caption_subset_sha256":subset_sha,"pg_scores":pg_scores,"pg_scores_sha256":pg_scores_sha},
      "inherited":{"external_root":external,"source_manifest":source_manifest,"source_manifest_sha256":source_manifest_sha,"base_directory":base,"base_directory_sha256":base_sha,"export_directory":export,"tokenizer_directory":tokenizer,"tokenizer_sha256":tokenizer_sha,"export_hashes":export_hashes},
      "stage2_cache":{"module":stage_module,"module_sha256":stage_module_sha,"config":stage_config,"config_sha256":stage_config_sha,"accepted_status":"APPROVED_FOR_EXECUTION"},
      "fast":{"snapshot":fast,"snapshot_sha256":fast_sha,"identity":{"checkpoint":"fast-ckpt","implementation":"fast-code"},"protocols":{"ucf-crime":protocol,"xd-violence":protocol}},
      "media":{"catalog":media,"catalog_sha256":media_sha,"observation_cache_root":str(tmp_path / "cache"),"observation_cache_max_bytes":1},
      "detector":{"snapshot":rtdetr,"snapshot_sha256":rtdetr_sha,"siglip_snapshot":siglip,"siglip_snapshot_sha256":siglip_sha,"final_stage2_siglip_snapshot":final_siglip,"final_stage2_siglip_snapshot_sha256":final_siglip_sha,"score_threshold":0.3},
      "teacher":{"artifact":teacher,"sha256":teacher_sha}}
    if mode == "formal":
        admission = {"status":"PASS","formal_execution_allowed":True,"engineering_checks":{str(i):"PASS" for i in range(1,11)},"source_files":{}}
        admission_path, admission_sha = file(tmp_path, "admission.json", json.dumps(admission).encode())
    path = tmp_path / "runtime.json"; path.write_text(json.dumps(doc)); return path, digest(path)


def test_preflight_accepts_pinned_diagnostic_without_model_import(tmp_path):
    path, expected = config(tmp_path)
    assert load_manifest(path, expected_sha256=expected).run["run_id"].startswith("diagnostic:")


@pytest.mark.parametrize("section,key,value", [("sampling", "frames_upbound", 63), ("stage2_cache", "accepted_status", "REVIEW_REQUIRED_NOT_AUTHORIZED_TO_LAUNCH")])
def test_preflight_rejects_sampling_or_unaccepted_stage2(tmp_path, section, key, value):
    path, _ = config(tmp_path); doc = json.loads(path.read_text()); doc[section][key] = value; path.write_text(json.dumps(doc))
    with pytest.raises(ProductionRuntimeError): load_manifest(path, expected_sha256=digest(path))


def test_preflight_rejects_missing_detector_before_runtime_import(tmp_path):
    path, _ = config(tmp_path); doc = json.loads(path.read_text()); doc["detector"]["snapshot_sha256"] = "0" * 64; path.write_text(json.dumps(doc))
    with pytest.raises(ProductionRuntimeError, match="RT-DETR snapshot"):
        load_manifest(path, expected_sha256=digest(path))


def test_preflight_rejects_changed_derived_final_stage2_siglip_before_runtime_import(tmp_path):
    path, _ = config(tmp_path); doc = json.loads(path.read_text())
    doc["detector"]["final_stage2_siglip_snapshot_sha256"] = "0" * 64
    path.write_text(json.dumps(doc))
    with pytest.raises(ProductionRuntimeError, match="derived final-Stage2 SigLIP"):
        load_manifest(path, expected_sha256=digest(path))


def test_inherited_tokenizer_retains_pretrained_pad_when_qwen_has_no_unknown_token():
    tokenizer = type("Tokenizer", (), {"unk_token": None, "pad_token": "<pad>", "padding_side": "left"})()
    _configure_inherited_tokenizer(tokenizer)
    assert tokenizer.pad_token == "<pad>"
    assert tokenizer.padding_side == "right"
    tokenizer.unk_token = "<unk>"
    _configure_inherited_tokenizer(tokenizer)
    assert tokenizer.pad_token == "<unk>"


def test_inherited_data_args_copy_the_loaded_multimodal_delimiter_policy():
    data_args = type("DataArgs", (), {})()
    _configure_inherited_data_args(data_args, type("Config", (), {"mm_use_im_start_end": True})())
    assert data_args.mm_use_im_start_end is True
    with pytest.raises(ProductionRuntimeError, match="mm_use_im_start_end"):
        _configure_inherited_data_args(type("DataArgs", (), {})(), type("Config", (), {})())


def test_runtime_rejects_an_unloaded_tower_without_rewriting_or_reloading_it():
    tower = type("Tower", (), {"is_loaded": False, "vision_tower_name": "original", "load_model": lambda self: (_ for _ in ()).throw(AssertionError())})()
    with pytest.raises(ProductionRuntimeError, match="already contain final Stage2"):
        _require_loaded_final_stage2_tower(tower)
    assert tower.vision_tower_name == "original"


@pytest.mark.skipif(not os.environ.get("NC_RTED_REACTVAU_ROOT"), reason="requires pinned ReactVAU source")
def test_real_inherited_preprocess_receives_bound_multimodal_delimiter_policy():
    root = Path(os.environ["NC_RTED_REACTVAU_ROOT"])
    sys.path.insert(0, str(root))
    # ReactVAU's trainer imports this compatibility name from Transformers;
    # preprocess_multimodal itself is unchanged and uses Accelerate's class.
    import transformers.trainer
    from accelerate.utils import GradientAccumulationPlugin
    transformers.trainer.GradientAccumulationPlugin = GradientAccumulationPlugin
    from llava.constants import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN
    from llava.train.train import DataArguments, preprocess_multimodal
    args = DataArguments()
    args.is_multimodal = True
    _configure_inherited_data_args(args, type("Config", (), {"mm_use_im_start_end": True})())
    source = [[{"from":"human", "value":"inspect <image> now"}]]
    prepared = preprocess_multimodal(source, args)
    assert prepared[0][0]["value"].startswith(DEFAULT_IM_START_TOKEN + "<image>" + DEFAULT_IM_END_TOKEN)


def test_formal_requires_all_ten_gates_and_is_not_diagnostic(tmp_path):
    path, _ = config(tmp_path, mode="formal"); admission = tmp_path / "admission.json"
    invalid = {"status":"PASS","formal_execution_allowed":True,"engineering_checks":{str(i):"PASS" for i in range(1,10)},"source_files":{}}
    admission.write_text(json.dumps(invalid))
    with pytest.raises(ProductionRuntimeError, match="ten accepted gates"):
        from nc_rted.production_runtime import load_formal_admission
        load_formal_admission(admission, expected_sha256=digest(admission))


def test_materialized_stage2_cache_leases_the_hash_bound_media_inode(tmp_path):
    media_path = tmp_path / "clip.bin"; media_path.write_bytes(b"immutable-media")
    item = BoundMedia("ucf-crime", "relative/clip.mp4", str(media_path), digest(media_path), 1.0, 1, 1, 1, 7)
    cache = _materialized_stage2_cache({("ucf-crime", item.media_key): item})
    assert cache.request_index_for({"_reactvau_relative_video": item.media_key}) == 7
    with cache.acquire(item.media_key, 7) as leased:
        assert leased.read_bytes() == b"immutable-media"


def test_preflight_rejects_yaml_that_mentions_unbound_caption_input(tmp_path):
    path, _ = config(tmp_path)
    doc = json.loads(path.read_text())
    yaml_path = Path(doc["catalog"]["dataset_yaml"])
    yaml_path.write_text(f"datasets:\n  - json_path: {doc['catalog']['caption_subset']}\n  - json_path: /unbound.json\npg_scores_path: {doc['catalog']['pg_scores']}\n")
    doc["catalog"]["dataset_yaml_sha256"] = digest(yaml_path)
    path.write_text(json.dumps(doc))
    with pytest.raises(ProductionRuntimeError, match="fixed captions"):
        load_manifest(path, expected_sha256=digest(path))


def test_caption_subset_uses_exact_original_instruction_rows_without_dataset_field(tmp_path):
    rows, tasks = [], []
    for index in range(2000):
        instruction = {"id": index, "video": f"clips/{index}.mp4", "task": "caption", "type": "clip",
                       "conversations": [{"from":"human", "value":"<video> describe"}, {"from":"gpt", "value":f"answer {index}"}]}
        rows.append(instruction)
        tasks.append(TrainingTask(f"caption:ucf-crime:{index}", "caption", "ucf-crime", "family", instruction["video"], None, None, instruction))
    catalog = TrainingCatalog(tasks, "catalog")
    subset = tmp_path / "subset.json"; subset.write_text(json.dumps(rows))
    _validate_caption_subset(catalog, subset)
    rows[245] = {**rows[245], "conversations": [{"from":"human", "value":"<video> describe"}, {"from":"gpt", "value":"mutated"}]}
    subset.write_text(json.dumps(rows))
    with pytest.raises(ProductionRuntimeError, match="original catalog instruction"):
        _validate_caption_subset(catalog, subset)


def test_detection_binding_rejects_fast_media_hash_mismatch_before_model_load(tmp_path):
    task = TrainingTask("detection:ucf-crime:key:0", "detection", "ucf-crime", "family", "key", 2.0, 0, None, 0)
    catalog = TrainingCatalog([task], "catalog")
    fast = tmp_path / "fast.json"
    fast.write_text(json.dumps({"schema":"nc_rted_frozen_fast/v1", "media":[{"dataset":"ucf-crime", "media_key":"key",
        "media_sha256":"a" * 64, "fps":2.0, "frame_count":9, "height":8, "width":8,
        "queries":[{"index":0, "frame_indices":[0,1,2,4]}]}]}))
    media = BoundMedia("ucf-crime", "key", "/bound/key", "b" * 64, 2.0, 9, 8, 8)
    with pytest.raises(ProductionRuntimeError, match="different detection media"):
        _validate_detection_bindings(catalog, fast, {("ucf-crime", "key"):media})


@pytest.mark.parametrize("field, fast_value, media_value", [
    ("fps", 2.0, 3.0), ("frame_count", 9, 10), ("height", 8, 9), ("width", 8, 9)])
def test_detection_binding_rejects_all_metadata_mismatches(tmp_path, field, fast_value, media_value):
    task = TrainingTask("detection:ucf-crime:key:0", "detection", "ucf-crime", "family", "key", 2.0, 0, None, 0)
    catalog = TrainingCatalog([task], "catalog")
    row = {"dataset":"ucf-crime", "media_key":"key", "media_sha256":"a" * 64, "fps":2.0, "frame_count":9,
           "height":8, "width":8, "queries":[{"index":0, "frame_indices":[0,1,2,4]}]}
    row[field] = fast_value
    fast = tmp_path / "fast.json"; fast.write_text(json.dumps({"schema":"nc_rted_frozen_fast/v1", "media":[row]}))
    values = dict(fps=2.0, frame_count=9, height=8, width=8); values[field] = media_value
    media = BoundMedia("ucf-crime", "key", "/bound/key", "a" * 64, **values)
    with pytest.raises(ProductionRuntimeError, match="different detection media"):
        _validate_detection_bindings(catalog, fast, {("ucf-crime", "key"):media})


def test_detection_binding_rejects_endpoint_drift_and_accepts_exact_composition(tmp_path):
    task = TrainingTask("detection:ucf-crime:key:0", "detection", "ucf-crime", "family", "key", 2.0, 0, None, 0)
    catalog = TrainingCatalog([task], "catalog")
    row = {"dataset":"ucf-crime", "media_key":"key", "media_sha256":"a" * 64, "fps":2.0, "frame_count":9,
           "height":8, "width":8, "queries":[{"index":0, "frame_indices":[0,1,2,4]}]}
    fast = tmp_path / "fast.json"; fast.write_text(json.dumps({"schema":"nc_rted_frozen_fast/v1", "media":[row]}))
    media = BoundMedia("ucf-crime", "key", "/bound/key", "a" * 64, 2.0, 9, 8, 8)
    _validate_detection_bindings(catalog, fast, {("ucf-crime", "key"):media})
    row["queries"][0]["frame_indices"][-1] = 3; fast.write_text(json.dumps({"schema":"nc_rted_frozen_fast/v1", "media":[row]}))
    with pytest.raises(ProductionRuntimeError, match="endpoint"):
        _validate_detection_bindings(catalog, fast, {("ucf-crime", "key"):media})


def test_source_shadow_module_outside_bound_manifest_is_rejected(tmp_path, monkeypatch):
    path, _ = config(tmp_path); inherited = json.loads(path.read_text())["inherited"]
    shadow = types.ModuleType("llava.shadow"); shadow.__file__ = str(tmp_path / "other.py")
    (tmp_path / "other.py").write_text("shadow")
    monkeypatch.setitem(sys.modules, "llava.shadow", shadow)
    with pytest.raises(ProductionRuntimeError, match="outside bound source"):
        _assert_inherited_modules_bound(inherited)


@pytest.mark.parametrize("name", ["detect_utils", "vad", "vad.get_prompt"])
def test_vad_source_shadow_module_is_rejected(tmp_path, monkeypatch, name):
    path, _ = config(tmp_path); inherited = json.loads(path.read_text())["inherited"]
    shadow = types.ModuleType(name); shadow.__file__ = str(tmp_path / f"{name.replace('.', '_')}.py")
    Path(shadow.__file__).write_text("shadow")
    if name == "vad": shadow.__path__ = [str(tmp_path)]
    monkeypatch.setitem(sys.modules, name, shadow)
    with pytest.raises(ProductionRuntimeError, match="outside bound source"):
        _assert_inherited_modules_bound(inherited)


@pytest.mark.skipif(not os.environ.get("NC_RTED_REACTVAU_ROOT"), reason="requires original ReactVAU source")
def test_cold_process_imports_actual_vad_dependency_closure_from_bound_paths():
    root = Path(os.environ["NC_RTED_REACTVAU_ROOT"])
    script = r'''
import hashlib, json, os, sys, tempfile
from pathlib import Path
from nc_rted.production_runtime import _import_bound_inherited_runtime
root = Path(os.environ["NC_RTED_REACTVAU_ROOT"])
files = {}
for directory in (root / "llava", root / "eval_utils", root / "vad"):
    for path in directory.rglob("*.py"):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        files[str(path.relative_to(root))] = digest
with tempfile.TemporaryDirectory() as temporary:
    manifest = Path(temporary) / "sources.json"
    manifest.write_text(json.dumps({"files": files}))
    sys.path.insert(0, str(root))
    _import_bound_inherited_runtime({"external_root": str(root), "source_manifest": str(manifest)})
    assert Path(sys.modules["detect_utils"].__file__).resolve() == root / "eval_utils/vad/detect_utils.py"
    assert Path(sys.modules["vad.get_prompt"].__file__).resolve() == root / "vad/get_prompt.py"
'''
    environment = dict(os.environ)
    result = subprocess.run([sys.executable, "-c", script], env=environment, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


def test_preflight_rejects_changed_bound_memory_manager_source(tmp_path):
    path, _ = config(tmp_path); doc = json.loads(path.read_text()); root = Path(doc["inherited"]["external_root"])
    files = {}
    names = ["llava/train/train.py", "llava/train/reactvau_stage2_cache_adapter.py", "llava/model/llava_arch.py",
             "llava/model/multimodal_encoder/siglip_encoder.py", "llava/model/multimodal_projector/memory_manager.py",
             "llava/model/language_model/llava_qwen.py", "eval_utils/vad/eval_reactvau_detection.py",
             "eval_utils/vad/detect_utils.py", "vad/get_prompt.py"]
    for name in names:
        target = root / name; target.parent.mkdir(parents=True, exist_ok=True); target.write_text(name)
        files[name] = digest(target)
    source_manifest = tmp_path / "complete-source-manifest.json"; source_manifest.write_text(json.dumps({"files":files}))
    doc["inherited"]["source_manifest"] = str(source_manifest); doc["inherited"]["source_manifest_sha256"] = digest(source_manifest)
    path.write_text(json.dumps(doc)); load_manifest(path, expected_sha256=digest(path))
    (root / "llava/model/multimodal_projector/memory_manager.py").write_text("changed")
    with pytest.raises(ProductionRuntimeError, match="ReactVAU source manifest"):
        load_manifest(path, expected_sha256=digest(path))


@pytest.mark.skipif(not __import__("os").environ.get("NC_RTED_REACTVAU_ROOT"), reason="requires original ReactVAU source")
def test_run_local_stage2_subclass_uses_original_yaml_constructor_without_global_patch(tmp_path, monkeypatch):
    import os
    root = Path(os.environ["NC_RTED_REACTVAU_ROOT"])
    import sys
    sys.path.insert(0, str(root))
    from llava.train import train as train_module

    subset = tmp_path / "captions.json"
    subset.write_text(json.dumps([{"id":"one", "video":"clip.mp4", "task":"caption", "type":"clip",
                                   "conversations":[{"from":"human","value":"<video> describe"},{"from":"gpt","value":"answer"}]}]))
    pg = tmp_path / "pg.json"; pg.write_text(json.dumps({"clip.mp4":{"pg_scores":[0.5], "sample_interval":1}}))
    yaml_path = tmp_path / "captions.yaml"
    yaml_path.write_text(f"datasets:\n  - json_path: {subset}\n    data_root: /verified\n    media_type: video\n    video_read_type: decord\npg_scores_path: {pg}\n")
    monkeypatch.delenv("REACTVAU_STAGE2_CACHE_CONFIG", raising=False)
    class Cache:
        def request_index_for(self, annotation):
            assert annotation["_reactvau_relative_video"] == "clip.mp4"; return 3
    subclass = _stage2_dataset_class(train_module.LazySupervisedDataset, Cache())
    args = train_module.DataArguments(data_path=str(yaml_path), lazy_preprocess=True, local_num_frames=1,
                                     frames_upbound=64, frames_lowbound=4, sample_type="dynamic_fps1", time_msg="short_online_v2")
    with _bound_stage2_constructor_environment({"mode":"materialized"}, str(yaml_path)):
        dataset = subclass(str(yaml_path), tokenizer=object(), data_args=args)
        assert os.environ["REACTVAU_STAGE2_CACHE_CONFIG"] == str(yaml_path)
    assert "REACTVAU_STAGE2_CACHE_CONFIG" not in os.environ
    assert train_module.LazySupervisedDataset is not subclass
    assert dataset.list_data_dict[0]["video"] == "/verified/clip.mp4"
    assert dataset.list_data_dict[0]["_reactvau_relative_video"] == "clip.mp4"
    assert dataset.list_data_dict[0]["_reactvau_request_index"] == 3
    assert dataset.pg_scores_dict["clip.mp4"]["pg_scores"] == [0.5]
