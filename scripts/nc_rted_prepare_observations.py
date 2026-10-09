#!/usr/bin/env python3
"""Prepare real train-only teacher observations, resuming immutable windows."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
_EXECUTED_DRIVER_CODE=sys._getframe().f_code
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))

class ExtractionError(ValueError):
    pass


def sha256_file(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8<<20),b''):digest.update(chunk)
    return digest.hexdigest()


def bound_json(binding):
    raw=Path(binding['path']).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=binding['sha256']:
        raise ExtractionError('bound preparation input changed')
    return json.loads(raw)


def validate_driver_code(path,expected_sha256):
    raw=Path(path).read_bytes()
    if (hashlib.sha256(raw).hexdigest()!=expected_sha256
            or compile(raw,_EXECUTED_DRIVER_CODE.co_filename,'exec',dont_inherit=True,
                       optimize=sys.flags.optimize)!=_EXECUTED_DRIVER_CODE):
        raise ExtractionError('executing driver differs from verified source bytes')

# Hash the actual observation dependency closure, excluding unrelated evolving
# queue/runtime code. Original encoder imports are separately bound as a tree.
SOURCE_FILES=("__init__.py","observation_extraction.py","media_observer.py","detector.py","frozen_vision.py","numerics.py",
 "detection_media.py","detection_provider.py","caption_provider.py","caption_sampling.py",
 "observation_cache.py","storage_lock.py","features.py","tracking.py","observation.py","batches.py",
 "bridge.py","model.py","task_inputs.py","teacher_records.py","teacher_store.py",
 "teacher_pipeline.py","teacher.py","teacher_cache.py","alignment.py","retrieval.py","inherited_memory.py")


def source_signature(path):
    stat=path.stat()
    return (stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)


def capture_inherited_source(external, expected):
    files={str(p.relative_to(external)):p for p in external.rglob("*.py") if "__pycache__" not in p.parts}
    signatures={name:source_signature(path) for name,path in files.items()}
    hashes={name:sha256_file(path) for name,path in files.items()}
    if (not hashes or hashes != expected
            or signatures != {name:source_signature(path) for name,path in files.items()}):
        raise ExtractionError("inherited source tree differs or changed during validation")
    return signatures


def validate_inherited_source(external, expected, signatures):
    if capture_inherited_source(external,expected) != signatures:
        raise ExtractionError("inherited source changed during import or model loading")


@contextmanager
def verified_source_imports(external, expected):
    """Compile verified source bytes, bypassing timestamp-based bytecode caches."""
    captured={}
    for name,digest in expected.items():
        path=(external/name).resolve();raw=path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=digest:
            raise ExtractionError("inherited source changed before verified import")
        captured[path]=raw
    # The local checkout also contains virtualenvs and upstream assets. Only
    # the nc_rted package is its import namespace; inherited roots are dedicated
    # source trees and protect their whole namespace.
    protected=(external/'src/nc_rted') if (external/'src/nc_rted/__init__.py') in captured else external
    bound_module_names=set();bound_packages=set()
    for name in expected:
        parts=list(Path(name).with_suffix('').parts)
        if parts[0]=='src':parts=parts[1:]
        if parts[-1]=='__init__':
            parts=parts[:-1]
            if parts:bound_packages.add('.'.join(parts))
        if parts:bound_module_names.add('.'.join(parts))
        for depth in range(1,len(parts)):
            bound_packages.add('.'.join(parts[:depth]))
    def owns_name(name):
        return name in bound_module_names or name in bound_packages or any(name.startswith(package+'.') for package in bound_packages)
    for name,module in tuple(sys.modules.items()):
        origin=getattr(module,"__file__",None)
        if owns_name(name) or (origin and (Path(origin).resolve() in captured or Path(origin).resolve().is_relative_to(protected))):
            raise ExtractionError("inherited module already imported without verified source binding")

    class BoundLoader(importlib.machinery.SourceFileLoader):
        def get_code(self,fullname):
            return compile(captured[Path(self.path).resolve()],self.path,"exec",dont_inherit=True)

    class BoundFinder(importlib.abc.MetaPathFinder):
        def find_spec(self,fullname,path=None,target=None):
            spec=importlib.machinery.PathFinder.find_spec(fullname,path)
            if spec is None:
                if owns_name(fullname):raise ModuleNotFoundError('unbound protected module: '+fullname,name=fullname)
                return None
            if not spec.origin:return spec  # Namespace package, no code bytes.
            if spec.origin in {"built-in","frozen"}:
                if owns_name(fullname):raise ExtractionError('unbound executable module: '+fullname)
                return None
            origin=Path(spec.origin).resolve()
            original_origin=Path(spec.origin).absolute()
            if (origin not in captured and not origin.is_relative_to(protected)
                    and not original_origin.is_relative_to(protected) and not owns_name(fullname)):return None
            if origin not in captured or not isinstance(spec.loader,importlib.machinery.SourceFileLoader):
                raise ExtractionError("unbound inherited import: "+fullname)
            spec.loader=BoundLoader(fullname,str(origin))
            return spec

    finder=BoundFinder();sys.meta_path.insert(0,finder)
    try:yield
    finally:sys.meta_path.remove(finder)


def derived_final_stage2_config(cfg):
    """Return the fixed derived-vision inputs required by extraction schema v2."""
    value=cfg.get("derived_final_stage2_siglip")
    required={"snapshot","parent_export","parent_export_sha256","raw_config_sha256"}
    if not isinstance(value,dict) or set(value)!=required:
        raise ExtractionError("derived final-Stage2 SigLIP binding is required")
    for name in ("snapshot","parent_export"):
        if not isinstance(value[name],str) or not value[name]:
            raise ExtractionError("derived final-Stage2 SigLIP path is invalid")
    for name in ("parent_export_sha256","raw_config_sha256"):
        digest=value[name]
        if not isinstance(digest,str) or len(digest)!=64 or any(char not in "0123456789abcdef" for char in digest):
            raise ExtractionError("derived final-Stage2 SigLIP hash is invalid")
    return value


def validate_config(path, expected):
    cfg=bound_json({"path":str(path),"sha256":expected})
    if cfg.get("schema")!="nc_rted_observation_extraction/v2":raise ExtractionError("unsupported extraction configuration")
    derived_final_stage2_config(cfg)
    required={"src/nc_rted/"+n for n in SOURCE_FILES}|{"scripts/nc_rted_prepare_observations.py"}
    if set(cfg.get("code_sha256",{}))!=required:raise ExtractionError("incomplete extraction source closure")
    validate_driver_code(ROOT/'scripts/nc_rted_prepare_observations.py',cfg['code_sha256']['scripts/nc_rted_prepare_observations.py'])
    local_signatures={name:source_signature(ROOT/name) for name in required}
    for name,digest in cfg["code_sha256"].items():
        if sha256_file(ROOT/name)!=digest:raise ExtractionError("extraction source changed: "+name)
    validate_local_source(cfg['code_sha256'],local_signatures)
    external=Path(cfg["reactvau_root"]).resolve()
    signatures=capture_inherited_source(external,cfg.get("reactvau_python_sha256"))
    provenance=Path(cfg["detector_snapshot"])/"nc_rted_provenance.json"
    expected_detector=bound_json({"path":str(provenance),"sha256":cfg["detector_provenance_sha256"]})
    if type(cfg.get("frame_cache_bytes")) is not int or not 0<cfg["frame_cache_bytes"]<=1<<30:raise ExtractionError("frame cache bound must be at most 1GiB")
    if cfg.get("reserved_free_bytes")!=20<<30:raise ExtractionError("free-space floor must remain 20GiB")
    if type(cfg.get("cpu_threads")) is not int or cfg["cpu_threads"]<1:raise ExtractionError("invalid preparation CPU threads")
    if cfg.get("dtype")!="bfloat16":raise ExtractionError("preparation must preserve inherited BF16")
    device=cfg.get("device")
    if not isinstance(device,str) or not (device=="cuda" or (device.startswith("cuda:") and device[5:].isdigit())):
        raise ExtractionError("actual local CUDA observation device required")
    return cfg,external,expected_detector,signatures,local_signatures


def validate_local_source(expected,signatures):
    for name,digest in expected.items():
        path=ROOT/name
        if source_signature(path)!=signatures[name] or sha256_file(path)!=digest or source_signature(path)!=signatures[name]:
            raise ExtractionError('local source changed across verified imports: '+name)


def derived_vision_identity(binding, provenance_digest):
    return {"snapshot":str(binding.snapshot),"config_sha256":binding.config_sha256,
            "weights_sha256":binding.weights_sha256,"parent_export_sha256":binding.parent_export_sha256,
            "source_key_prefix":binding.source_key_prefix,"tensor_count":len(binding.source_key_map),
            "provenance_digest":provenance_digest(binding)}


def validate_loaded_models(cfg, expected_detector, detector_identity, siglip_identity, derived_identity,
                           numerical_policy_identity):
    """Compare constructed adapters with the snapshots captured at preflight."""
    if (detector_identity.get("provenance") != expected_detector
            or detector_identity.get("files") != expected_detector.get("files")
            or detector_identity.get("snapshot") != str(Path(cfg["detector_snapshot"]).resolve())):
        raise ExtractionError("loaded detector differs from verified preparation snapshot")
    derived=derived_final_stage2_config(cfg)
    if (siglip_identity.get("config_sha256") != derived_identity["config_sha256"]
            or siglip_identity.get("weights_sha256") != derived_identity["weights_sha256"]
            or siglip_identity.get("tower_source_sha256") != cfg["reactvau_python_sha256"]["llava/model/multimodal_encoder/siglip_encoder.py"]
            or siglip_identity.get("snapshot") != str(Path(derived["snapshot"]).resolve())
            or siglip_identity.get("numerical_policy_identity") != numerical_policy_identity):
        raise ExtractionError("loaded SigLIP differs from verified preparation snapshot")
    loaded=siglip_identity.get("derived_final_stage2_vision")
    expected={"parent_export_sha256":derived_identity["parent_export_sha256"],
              "source_key_prefix":derived_identity["source_key_prefix"],
              "tensor_count":derived_identity["tensor_count"],
              "provenance_digest":derived_identity["provenance_digest"],
              "loaded_tower_path":str(Path(derived["snapshot"]).resolve())}
    if loaded != expected:
        raise ExtractionError("loaded SigLIP derived final-Stage2 provenance differs")


def prepare_derived_final_stage2_vision(cfg):
    """Establish process numerical policy and bind derived weights before model load."""
    from nc_rted.numerics import configure_deterministic_algorithms
    from nc_rted.frozen_vision import bind_derived_final_stage2_vision, provenance_digest
    policy=configure_deterministic_algorithms()
    derived=derived_final_stage2_config(cfg)
    binding=bind_derived_final_stage2_vision(
        derived["snapshot"], expected_parent_export_sha256=derived["parent_export_sha256"],
        expected_parent_export=derived["parent_export"], expected_raw_config_sha256=derived["raw_config_sha256"])
    return policy,binding,derived_vision_identity(binding,provenance_digest)


def assert_active_numerical_policy(policy, torch):
    """Reject a process whose configured numerical policy was later changed."""
    from nc_rted.numerics import deterministic_policy
    required=deterministic_policy()
    if (policy != required or policy.identity()!=required.identity()
            or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != required.cublas_workspace_config
            or not torch.are_deterministic_algorithms_enabled()
            or torch.is_deterministic_algorithms_warn_only_enabled()):
        raise ExtractionError("deterministic numerical policy changed during observation preparation")


def construct_derived_siglip_adapter(adapter_class, tower, derived_binding, derived, policy):
    """Use the adapter's derived path; its second verification closes load-time races."""
    return adapter_class(tower,derived_binding.snapshot,
                         expected_parent_export_sha256=derived["parent_export_sha256"],
                         expected_parent_export=derived["parent_export"],
                         expected_raw_config_sha256=derived["raw_config_sha256"],
                         numerical_policy_identity=policy.identity())


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",type=Path,required=True);parser.add_argument("--config-sha256",required=True)
    parser.add_argument("--max-new-windows",type=int);parser.add_argument("--dry-run",action="store_true")
    args=parser.parse_args();cfg,external,expected_detector,source_signatures,local_signatures=validate_config(args.config,args.config_sha256)
    local_imports={name:digest for name,digest in cfg['code_sha256'].items() if name.startswith('src/')}
    with verified_source_imports(ROOT,local_imports):
        from nc_rted.observation_extraction import load_detection_inputs
        validate_local_source(cfg['code_sha256'],local_signatures)
        truths,catalog=load_detection_inputs(Path(cfg["manifest_directory"]),cfg["provenance_sha256"],cfg["media"],cfg["pts"])
        if args.dry_run:
            print(json.dumps({"status":"PREPARATION_BINDINGS_PASS_MODELS_NOT_RUN","windows":len(truths),"media":len(catalog),"formal_execution":False}));return
        from nc_rted.observation_extraction import observation_run_admission
        with observation_run_admission(Path(cfg["output"]), reserved_free_bytes=cfg["reserved_free_bytes"]) as admission:
            with verified_source_imports(external,cfg["reactvau_python_sha256"]):
                run_observation_models(args,cfg,external,expected_detector,source_signatures,local_signatures,truths,catalog,admission)


def run_observation_models(args,cfg,external,expected_detector,source_signatures,local_signatures,truths,catalog,admission):
    import torch
    policy,derived_binding,derived_identity=prepare_derived_final_stage2_vision(cfg)
    derived=derived_final_stage2_config(cfg)
    _run_observation_models_with_derived(args,cfg,external,expected_detector,source_signatures,local_signatures,
                                         truths,catalog,admission,torch,policy,derived_binding,derived_identity,derived)


def _run_observation_models_with_derived(args,cfg,external,expected_detector,source_signatures,local_signatures,
                                         truths,catalog,admission,torch,policy,derived_binding,derived_identity,derived):
    sys.path.insert(0,str(external))
    from llava.model.multimodal_encoder.siglip_encoder import SigLipVisionTower
    if Path(inspect.getfile(SigLipVisionTower)).resolve()!=external/"llava/model/multimodal_encoder/siglip_encoder.py":raise ExtractionError("inherited encoder import shadowed")
    validate_inherited_source(external,cfg["reactvau_python_sha256"],source_signatures)
    # The inherited module changes global CPU threads at import. Set an explicit
    # preparation setting after import; it enters the bound run configuration.
    torch.set_num_threads(cfg["cpu_threads"])
    device=torch.device(cfg["device"])
    if device.type!="cuda" or not torch.cuda.is_available():raise ExtractionError("actual local CUDA observation device required")
    assert_active_numerical_policy(policy,torch)
    from nc_rted.detector import FrozenRTDetr,InheritedSigLipAdapter
    from nc_rted.media_observer import CausalMediaObserver
    from nc_rted.detection_media import OpenCVFrames
    from nc_rted.observation_cache import FrozenFrameCache
    from nc_rted.observation_extraction import ObservationJournal,extract_windows
    detector=FrozenRTDetr(Path(cfg["detector_snapshot"]),device=str(device))
    tower=SigLipVisionTower(str(derived_binding.snapshot),SimpleNamespace()).to(device=device,dtype=torch.bfloat16).eval()
    siglip=construct_derived_siglip_adapter(InheritedSigLipAdapter,tower,derived_binding,derived,policy)
    validate_inherited_source(external,cfg["reactvau_python_sha256"],source_signatures)
    validate_local_source(cfg['code_sha256'],local_signatures)
    detector_identity,siglip_identity=detector.identity(),siglip.identity()
    validate_loaded_models(cfg,expected_detector,detector_identity,siglip_identity,derived_identity,policy.identity())
    assert_active_numerical_policy(policy,torch)
    binding={"config_sha256":args.config_sha256,"detector":detector_identity,"siglip":siglip_identity,
             "derived_final_stage2_vision":derived_identity,"numerical_policy_identity":policy.identity(),
             "cpu_threads":torch.get_num_threads(),"torch":str(torch.__version__)}
    ids=tuple(f"detection:{t.dataset}:{t.key}:{t.query_index}" for t in truths)
    journal=ObservationJournal(Path(cfg["output"]),binding,ids)
    cache=FrozenFrameCache(Path(cfg["frame_cache"]),cfg["frame_cache_bytes"])
    observer=CausalMediaObserver(detector=detector,siglip=siglip,cache=cache,media_catalog=catalog,decoder_factory=OpenCVFrames)
    print(json.dumps(extract_windows(truths,observer,journal,max_new_windows=args.max_new_windows,admission=admission,
                                     before_finish=lambda:assert_active_numerical_policy(policy,torch))),flush=True)

if __name__=="__main__":main()
