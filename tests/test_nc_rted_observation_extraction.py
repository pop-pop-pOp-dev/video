from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import json
import pytest
from nc_rted.observation_extraction import ExtractionError,ObservationJournal,extract_windows
from nc_rted.teacher_records import TeacherSourceTruth,CompactTeacherWindow
from nc_rted.teacher_store import load_teacher_store
from nc_rted.detector import CausalWindowObservation
from nc_rted.features import FeatureAssemblyResult,FeatureStatus


def test_bound_json_parses_the_same_bytes_it_verified(tmp_path,monkeypatch):
    import hashlib
    from nc_rted.observation_extraction import bound_json
    path=tmp_path/"input.json";original=b'{"version":"original"}'
    path.write_bytes(original);read=Path.read_bytes
    def replace_after_read(self):
        value=read(self)
        if self==path:self.write_text('{"version":"replacement"}')
        return value
    monkeypatch.setattr(Path,"read_bytes",replace_after_read)
    binding={"path":str(path),"sha256":hashlib.sha256(original).hexdigest()}
    assert bound_json(binding)=={"version":"original"}
    with pytest.raises(ExtractionError,match="changed"):bound_json(binding)


def truth(number):
    return TeacherSourceTruth("ucf-crime","source",number,"family","alias",0,"train",True,8.+number)

def window_id(t):return f"detection:{t.dataset}:{t.key}:{t.query_index}"

class Observer:
    def __init__(self):self.calls=[];self.fail_on=None
    def detection(self,dataset,key,seconds):
        self.calls.append((dataset,key,seconds))
        if seconds==self.fail_on:raise OSError("temporary decode failure")
        return CausalWindowObservation(FeatureAssemblyResult(FeatureStatus.NO_RELATION_PAIRS,()),(),())


def test_crash_resume_preserves_committed_windows_and_exact_denominator(tmp_path):
    truths=(truth(0),truth(1),truth(2));ids=tuple(map(window_id,truths));observer=Observer()
    journal=ObservationJournal(tmp_path,{"source":"fixed"},ids,reserved_free_bytes=0)
    observer.fail_on=9.
    with pytest.raises(OSError):extract_windows(truths,observer,journal)
    assert journal.read(ids[0]) is not None and journal.read(ids[1]) is None
    assert not (tmp_path/"commit.json").exists()
    observer.fail_on=None;observer.calls.clear()
    result=extract_windows(truths,observer,journal)
    assert result["computed"]==2 and result["reused"]==1
    assert [x[2] for x in observer.calls]==[9.,10.]
    records=load_teacher_store(tmp_path)
    assert {x.record["window_id"] for x in records}==set(ids)
    observer.calls.clear();again=extract_windows(truths,observer,journal)
    assert again["computed"]==0 and again["reused"]==3 and not observer.calls


def test_bound_partial_preparation_cannot_be_mistaken_for_complete_store(tmp_path):
    truths=(truth(0),truth(1));journal=ObservationJournal(tmp_path,{"source":"fixed"},tuple(map(window_id,truths)),reserved_free_bytes=0)
    result=extract_windows(truths,Observer(),journal,max_new_windows=1)
    assert result["status"]=="PARTIAL_RESUMABLE_OBSERVATIONS" and not (tmp_path/"commit.json").exists()
    with journal.writer():
        with pytest.raises(ExtractionError,match="incomplete"):journal.finish()
    with pytest.raises(ExtractionError,match="denominator"):extract_windows(truths[:1],Observer(),journal)


def test_changed_run_identity_and_corrupt_committed_window_refuse_resume(tmp_path):
    truths=(truth(0),);ids=tuple(map(window_id,truths));journal=ObservationJournal(tmp_path,{"source":"one"},ids,reserved_free_bytes=0)
    extract_windows(truths,Observer(),journal)
    with pytest.raises(ExtractionError,match="binding"):
        extract_windows(truths,Observer(),ObservationJournal(tmp_path,{"source":"two"},ids,reserved_free_bytes=0))
    chunk=next((tmp_path/"windows").glob("*/index.json"));chunk.write_text("{}")
    with pytest.raises(ValueError,match="schema|hash"):journal.read(ids[0])


def test_technical_feature_failure_is_not_persisted_as_normal_rejection(tmp_path):
    class Broken(Observer):
        def detection(self,*args):return CausalWindowObservation(FeatureAssemblyResult(FeatureStatus.INVALID_INPUT,(),"bad boxes"),(),())
    truths=(truth(0),);ids=tuple(map(window_id,truths));journal=ObservationJournal(tmp_path,{},ids,reserved_free_bytes=0)
    with pytest.raises(ExtractionError,match="technical"):extract_windows(truths,Broken(),journal)
    assert journal.read(ids[0]) is None and not (tmp_path/"commit.json").exists()


def test_concurrent_writer_refuses_without_mutating_run(tmp_path):
    ids=(window_id(truth(0)),);first=ObservationJournal(tmp_path,{"fixed":True},ids,reserved_free_bytes=0)
    second=ObservationJournal(tmp_path,{"fixed":True},ids,reserved_free_bytes=0)
    with first.writer():
        before=(tmp_path/"run.json").read_bytes()
        with pytest.raises(ExtractionError,match="already active"):
            with second.writer():pass
        assert (tmp_path/"run.json").read_bytes()==before


@pytest.mark.parametrize("artifact",["windows","index.json","commit.json","progress.json"])
def test_missing_run_binding_cannot_adopt_existing_artifacts(tmp_path,artifact):
    path=tmp_path/artifact
    if artifact=="windows":path.mkdir()
    else:path.write_text("{}")
    journal=ObservationJournal(tmp_path,{"new":"config"},(window_id(truth(0)),),reserved_free_bytes=0)
    with pytest.raises(ExtractionError,match="orphaned"):
        with journal.writer():pass
    assert path.exists() and not (tmp_path/"run.json").exists()


def test_committed_technical_rejection_blocks_resume_and_seal(tmp_path):
    from nc_rted.teacher_store import write_teacher_store
    truths=(truth(0),);identifier=window_id(truths[0])
    journal=ObservationJournal(tmp_path,{},(identifier,),reserved_free_bytes=0)
    reason="technical decode failure"
    compact=CompactTeacherWindow({"window_id":identifier,"dataset":"ucf-crime","rejection":reason},(),reason)
    with journal.writer():
        with pytest.raises(ExtractionError,match="technical"):journal.put(compact)
        write_teacher_store(journal._window_path(identifier),(compact,),reserved_free_bytes=0)
        with pytest.raises(ExtractionError,match="technical"):journal.finish()
    with pytest.raises(ExtractionError,match="technical"):extract_windows(truths,Observer(),journal)
    assert not (tmp_path/"commit.json").exists()


def test_direct_seal_cannot_rebind_completed_unsealed_windows(tmp_path):
    from nc_rted.teacher_records import feature_assembly_to_teacher_window
    t=truth(0);identifier=window_id(t)
    journal=ObservationJournal(tmp_path,{"config":"A"},(identifier,),reserved_free_bytes=0)
    value=feature_assembly_to_teacher_window(Observer().detection('','',8.).features,t,())
    with journal.writer():journal.put(value)
    other=ObservationJournal(tmp_path,{"config":"B"},(identifier,),reserved_free_bytes=0)
    for instance in (journal,other):
        with pytest.raises(ExtractionError,match="writer context"):instance.finish()
        with pytest.raises(ExtractionError,match="writer context"):instance.put(value)
    (tmp_path/"run.json").unlink()
    with pytest.raises(ExtractionError,match="writer context"):other.finish()
    with pytest.raises(ExtractionError,match="orphaned"):
        with other.writer():pass
    assert not (tmp_path/"index.json").exists() and not (tmp_path/"commit.json").exists()


def test_active_writer_rejects_removed_run_binding_before_seal(tmp_path):
    journal=ObservationJournal(tmp_path,{},(window_id(truth(0)),),reserved_free_bytes=0)
    with journal.writer():
        (tmp_path/"run.json").unlink()
        with pytest.raises(ExtractionError,match="binding changed"):journal.finish()
    assert not (tmp_path/"index.json").exists()


@pytest.mark.parametrize("changed",["detector_provenance","detector_files","siglip_config","siglip_weights","encoder_source"])
def test_snapshot_substitution_after_preflight_is_rejected(tmp_path,changed):
    import runpy
    check=runpy.run_path(str(Path(__file__).resolve().parents[1]/"scripts/nc_rted_prepare_observations.py"))["validate_loaded_models"]
    cfg={"detector_snapshot":str(tmp_path/'detector'),"derived_final_stage2_siglip":{
             "snapshot":str(tmp_path/'derived-siglip'),"parent_export":str(tmp_path/'parent.bin'),
             "parent_export_sha256":"e"*64,"raw_config_sha256":"c"*64},
         "reactvau_python_sha256":{"llava/model/multimodal_encoder/siglip_encoder.py":"1"*64}}
    expected={"revision":"fixed","files":{"model.safetensors":"a"*64}}
    detector={"snapshot":cfg['detector_snapshot'],"provenance":expected,"files":dict(expected['files'])}
    derived={"snapshot":cfg["derived_final_stage2_siglip"]["snapshot"],"config_sha256":"c"*64,
             "weights_sha256":"d"*64,"parent_export_sha256":"e"*64,
             "source_key_prefix":"base_model.model.model.vision_tower.vision_tower.","tensor_count":421,
             "provenance_digest":"f"*64}
    siglip={"snapshot":derived["snapshot"],"config_sha256":"c"*64,"weights_sha256":"d"*64,
            "tower_source_sha256":"1"*64,"numerical_policy_identity":"9"*64,
            "derived_final_stage2_vision":{"parent_export_sha256":"e"*64,
                "source_key_prefix":derived["source_key_prefix"],"tensor_count":421,
                "provenance_digest":"f"*64,"loaded_tower_path":str(Path(derived["snapshot"]).resolve())}}
    check(cfg,expected,detector,siglip,derived,"9"*64)
    if changed=="detector_provenance":detector['provenance']={**expected,"revision":"replacement"}
    elif changed=="detector_files":detector['files']={"model.safetensors":"b"*64}
    elif changed=="siglip_config":siglip['config_sha256']="e"*64
    elif changed=="siglip_weights":siglip['weights_sha256']="f"*64
    else:siglip['tower_source_sha256']="2"*64
    with pytest.raises(ValueError,match="verified preparation snapshot"):check(cfg,expected,detector,siglip,derived,"9"*64)


def test_old_raw_extraction_schema_is_rejected_before_model_preflight(tmp_path):
    import hashlib,runpy
    driver=runpy.run_path(str(Path(__file__).resolve().parents[1]/"scripts/nc_rted_prepare_observations.py"))
    config=tmp_path/'raw-v1.json';config.write_text(json.dumps({"schema":"nc_rted_observation_extraction/v1"}))
    with pytest.raises(ValueError,match="unsupported extraction configuration"):
        driver['validate_config'](config,hashlib.sha256(config.read_bytes()).hexdigest())


def test_derived_final_stage2_config_requires_exact_provenance_inputs():
    import runpy
    parse=runpy.run_path(str(Path(__file__).resolve().parents[1]/"scripts/nc_rted_prepare_observations.py"))["derived_final_stage2_config"]
    with pytest.raises(ValueError,match="required"):
        parse({"derived_final_stage2_siglip":{"snapshot":"derived"}})
    config={"derived_final_stage2_siglip":{"snapshot":"derived","parent_export":"parent.bin",
            "parent_export_sha256":"a"*64,"raw_config_sha256":"b"*64}}
    assert parse(config)==config["derived_final_stage2_siglip"]


def test_derived_policy_precedes_provenance_binding_and_failure_stops_binding(monkeypatch):
    import runpy
    import nc_rted.numerics as numerics
    import nc_rted.frozen_vision as frozen_vision
    driver=runpy.run_path(str(Path(__file__).resolve().parents[1]/"scripts/nc_rted_prepare_observations.py"))
    events=[]
    class Policy:
        def identity(self): return "policy-id"
    class Binding:
        snapshot=Path('/derived');config_sha256='a'*64;weights_sha256='b'*64
        parent_export_sha256='c'*64;source_key_prefix='prefix.';source_key_map={"weight":"prefix.weight"}
    monkeypatch.setattr(numerics,'configure_deterministic_algorithms',lambda:events.append('policy') or Policy())
    monkeypatch.setattr(frozen_vision,'bind_derived_final_stage2_vision',lambda *args,**kwargs:events.append(('binding',args,kwargs)) or Binding())
    monkeypatch.setattr(frozen_vision,'provenance_digest',lambda binding:'digest')
    cfg={"derived_final_stage2_siglip":{"snapshot":"derived","parent_export":"parent.bin",
         "parent_export_sha256":"c"*64,"raw_config_sha256":"a"*64}}
    policy,binding,identity=driver['prepare_derived_final_stage2_vision'](cfg)
    assert policy.identity()=="policy-id" and binding.snapshot==Path('/derived')
    assert events[0]=='policy' and events[1][0]=='binding'
    assert identity["provenance_digest"]=='digest'
    monkeypatch.setattr(numerics,'configure_deterministic_algorithms',lambda:(_ for _ in ()).throw(RuntimeError('policy failed')))
    events.clear()
    with pytest.raises(RuntimeError,match='policy failed'):
        driver['prepare_derived_final_stage2_vision'](cfg)
    assert events==[]


def test_derived_adapter_receives_parent_binding_and_policy_identity():
    import runpy
    driver=runpy.run_path(str(Path(__file__).resolve().parents[1]/"scripts/nc_rted_prepare_observations.py"))
    captured={}
    class Adapter:
        def __init__(self,*args,**kwargs):captured['args']=args;captured['kwargs']=kwargs
    class Policy:
        def identity(self):return 'policy-id'
    binding=type('Binding',(),{'snapshot':Path('/derived')})()
    derived={"parent_export":"/parent.bin","parent_export_sha256":"a"*64,"raw_config_sha256":"b"*64}
    adapter=driver['construct_derived_siglip_adapter'](Adapter,object(),binding,derived,Policy())
    assert isinstance(adapter,Adapter)
    assert captured['args'][1]==Path('/derived')
    assert captured['kwargs']=={"expected_parent_export_sha256":"a"*64,"expected_parent_export":"/parent.bin",
                                "expected_raw_config_sha256":"b"*64,"numerical_policy_identity":"policy-id"}


def test_active_numerical_policy_is_rechecked_before_model_work_and_sealing(monkeypatch):
    import os,runpy
    from nc_rted.numerics import deterministic_policy
    check=runpy.run_path(str(Path(__file__).resolve().parents[1]/"scripts/nc_rted_prepare_observations.py"))["assert_active_numerical_policy"]
    policy=deterministic_policy()
    class Torch:
        enabled=True;warn_only=False
        @classmethod
        def are_deterministic_algorithms_enabled(cls):return cls.enabled
        @classmethod
        def is_deterministic_algorithms_warn_only_enabled(cls):return cls.warn_only
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG",policy.cublas_workspace_config)
    check(policy,Torch)
    Torch.enabled=False
    with pytest.raises(ValueError,match="policy changed"):check(policy,Torch)
    Torch.enabled=True;monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG",":16:8")
    with pytest.raises(ValueError,match="policy changed"):check(policy,Torch)
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"]==":16:8"


def test_before_finish_callback_blocks_sealing_when_policy_changes(tmp_path):
    truths=(truth(0),);journal=ObservationJournal(tmp_path,{},(window_id(truths[0]),),reserved_free_bytes=0)
    with pytest.raises(RuntimeError,match="policy changed"):
        extract_windows(truths,Observer(),journal,before_finish=lambda:(_ for _ in ()).throw(RuntimeError("policy changed")))
    assert not (tmp_path/"commit.json").exists()


def test_inherited_source_replacement_between_preflight_and_import_fails(tmp_path):
    import runpy,hashlib
    module=runpy.run_path(str(Path(__file__).resolve().parents[1]/"scripts/nc_rted_prepare_observations.py"))
    source=tmp_path/'siglip_encoder.py';source.write_text('def forward(x): return x\n')
    expected={source.name:hashlib.sha256(source.read_bytes()).hexdigest()}
    signature=module['capture_inherited_source'](tmp_path,expected)
    source.write_text('def forward(x): return -x\n')
    with pytest.raises(ValueError,match="source tree differs"):
        module['validate_inherited_source'](tmp_path,expected,signature)

    restored=tmp_path/'restored.tmp';restored.write_text('def forward(x): return x\n');restored.replace(source)
    with pytest.raises(ValueError,match="changed during"):
        module['validate_inherited_source'](tmp_path,expected,signature)


def test_verified_import_ignores_stale_timestamp_bytecode(tmp_path,monkeypatch):
    import importlib,hashlib,os,py_compile,runpy
    module=runpy.run_path(str(Path(__file__).resolve().parents[1]/"scripts/nc_rted_prepare_observations.py"))
    name='nc_rted_stale_encoder_fixture';source=tmp_path/(name+'.py')
    source.write_text('result = "A"\n');stamp=source.stat()
    py_compile.compile(str(source),doraise=True,invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP)
    source.write_text('result = "B"\n');os.utime(source,ns=(stamp.st_atime_ns,stamp.st_mtime_ns))
    monkeypatch.syspath_prepend(str(tmp_path))
    # Establish that the fixture reproduces normal Python's stale-cache path.
    assert importlib.import_module(name).result=='A'
    sys.modules.pop(name)
    expected={source.name:hashlib.sha256(source.read_bytes()).hexdigest()}
    try:
        with module['verified_source_imports'](tmp_path,expected):
            assert importlib.import_module(name).result=='B'
    finally:sys.modules.pop(name,None)


def test_verified_import_rejects_preloaded_unbound_module(tmp_path,monkeypatch):
    import importlib,hashlib,runpy
    module=runpy.run_path(str(Path(__file__).resolve().parents[1]/"scripts/nc_rted_prepare_observations.py"))
    name='nc_rted_preloaded_encoder_fixture';source=tmp_path/(name+'.py');source.write_text('result=1\n')
    monkeypatch.syspath_prepend(str(tmp_path));importlib.import_module(name)
    try:
        with pytest.raises(ValueError,match='already imported'):
            with module['verified_source_imports'](tmp_path,{source.name:hashlib.sha256(source.read_bytes()).hexdigest()}):pass
    finally:sys.modules.pop(name,None)


def test_final_index_references_exact_payload_without_copy_and_binds_inputs(tmp_path):
    import numpy as np
    from nc_rted.features import PROCESS_FEATURE_DIM
    identifier=window_id(truth(0));binding={"config_sha256":"a"*64}
    process=np.full((4,PROCESS_FEATURE_DIM),np.nan,dtype=np.float64);process[0]=1.25
    record={"dataset":"ucf-crime","window_id":identifier,"source_family":"family","content_alias":"alias",
            "fold":0,"normal_permitted":True,"background":np.ones(1152),"background_valid":True,
            "class_composition":np.r_[1.,np.zeros(79)],"pairs":[{"pair_id":"0:1","class_pair":[0,2],
            "initial_geometry":np.zeros(5),"candidate_pair_count":1,"valid_cells":np.array([True,False,False,False]),"process_cells":process}]}
    compact=CompactTeacherWindow(record,("0:1",));journal=ObservationJournal(tmp_path,binding,(identifier,),reserved_free_bytes=0)
    with journal.writer():
        journal.put(compact)
        original=next((tmp_path/"windows").glob("*/chunks/*.npz"))
        before=original.stat().st_ino
        journal.finish()
    index=json.loads((tmp_path/"index.json").read_text())
    assert index["observation_binding"]==binding
    assert (tmp_path/index["entries"][0]["path"]).stat().st_ino==before
    loaded=load_teacher_store(tmp_path)[0]
    np.testing.assert_array_equal(loaded.record["pairs"][0]["process_cells"],process)
    assert len(list(tmp_path.rglob("*.npz")))==1


def test_local_source_change_after_capture_blocks_publication(tmp_path,monkeypatch):
    import runpy,hashlib
    module=runpy.run_path(str(Path(__file__).resolve().parents[1]/"scripts/nc_rted_prepare_observations.py"))
    validate=module['validate_local_source'];monkeypatch.setitem(validate.__globals__,'ROOT',tmp_path)
    source=tmp_path/'observation.py';source.write_text('result=1\n')
    expected={source.name:hashlib.sha256(source.read_bytes()).hexdigest()}
    signatures={source.name:module['source_signature'](source)}
    validate(expected,signatures)
    replacement=tmp_path/'replacement.tmp';replacement.write_text('result=1\n');replacement.replace(source)
    with pytest.raises(ValueError,match='local source changed'):validate(expected,signatures)


def test_verified_local_observation_import_closure_is_complete_in_fresh_process():
    import subprocess
    root=Path(__file__).resolve().parents[1]
    program='''
from pathlib import Path
import hashlib,runpy,sys
root=Path(sys.argv[1])
module=runpy.run_path(str(root/'scripts/nc_rted_prepare_observations.py'))
names=['src/nc_rted/'+name for name in module['SOURCE_FILES']]
expected={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names}
with module['verified_source_imports'](root,expected):
    from nc_rted.observation_extraction import load_detection_inputs
    from nc_rted.media_observer import CausalMediaObserver
    from nc_rted.detection_media import OpenCVFrames
    assert 'nc_rted.train_worker' not in sys.modules
'''
    result=subprocess.run([sys.executable,'-c',program,str(root)],text=True,capture_output=True)
    assert result.returncode==0,result.stdout+result.stderr


def test_running_driver_code_is_bound_even_if_path_is_replaced():
    import hashlib,runpy
    source=Path(__file__).resolve().parents[1]/'scripts/nc_rted_prepare_observations.py'
    raw=source.read_bytes();digest=hashlib.sha256(raw).hexdigest()
    original=runpy.run_path(str(source));original['validate_driver_code'](source,digest)
    changed=raw.replace(b'class ExtractionError(ValueError):',b'class ExtractionError(RuntimeError):',1)
    namespace={'__file__':str(source),'__name__':'modified_driver_fixture'}
    exec(compile(changed,str(source),'exec',dont_inherit=True),namespace)
    with pytest.raises(RuntimeError,match='executing driver differs'):
        namespace['validate_driver_code'](source,digest)


def test_verified_import_binds_symlinked_source_outside_root(tmp_path,monkeypatch):
    import importlib,hashlib,os,py_compile,runpy
    module=runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/nc_rted_prepare_observations.py'))
    root=tmp_path/'root';root.mkdir();outside=tmp_path/'outside';outside.mkdir()
    name='nc_rted_linked_encoder_fixture';source=outside/(name+'.py')
    source.write_text('result="A"\n');stamp=source.stat()
    py_compile.compile(str(source),doraise=True,invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP)
    source.write_text('result="B"\n');os.utime(source,ns=(stamp.st_atime_ns,stamp.st_mtime_ns))
    (root/source.name).symlink_to(source);monkeypatch.syspath_prepend(str(root))
    try:
        with module['verified_source_imports'](root,{source.name:hashlib.sha256(source.read_bytes()).hexdigest()}):
            assert importlib.import_module(name).result=='B'
    finally:sys.modules.pop(name,None)


def test_verified_import_rejects_preloaded_name_with_foreign_file(tmp_path,monkeypatch):
    import hashlib,runpy,types
    driver=runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/nc_rted_prepare_observations.py'))
    source=tmp_path/'foreign_encoder_fixture.py';source.write_text('result=1\n')
    forged=types.ModuleType(source.stem);forged.__file__='/different/environment/foreign_encoder_fixture.py'
    monkeypatch.setitem(sys.modules,source.stem,forged)
    with pytest.raises(ValueError,match='already imported'):
        with driver['verified_source_imports'](tmp_path,{source.name:hashlib.sha256(source.read_bytes()).hexdigest()}):pass


def test_owned_import_cannot_fall_through_to_editable_install_finder(tmp_path,monkeypatch):
    import hashlib,importlib,importlib.abc,importlib.util,runpy
    driver=runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/nc_rted_prepare_observations.py'))
    name='nc_rted_editable_fixture';source=tmp_path/(name+'.py');source.write_text('result=1\n')
    calls=[]
    class EditableFinder(importlib.abc.MetaPathFinder):
        def find_spec(self,fullname,path=None,target=None):
            if fullname==name:
                calls.append(fullname);return importlib.util.spec_from_file_location(fullname,source)
    finder=EditableFinder();sys.meta_path.append(finder)
    try:
        with driver['verified_source_imports'](tmp_path,{source.name:hashlib.sha256(source.read_bytes()).hexdigest()}):
            with pytest.raises(ModuleNotFoundError):importlib.import_module(name)
        assert calls==[]
    finally:sys.meta_path.remove(finder);sys.modules.pop(name,None)


def test_resume_metadata_reserves_allocation_before_writing(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import nc_rted.observation_extraction as module
    monkeypatch.setattr(module.shutil,'disk_usage',lambda path:SimpleNamespace(free=(20<<30)+1))
    def forbidden(*args,**kwargs):raise AssertionError('metadata tempfile before space admission')
    monkeypatch.setattr(module.tempfile,'mkstemp',forbidden)
    with pytest.raises(ExtractionError,match='disk hard limit'):
        module.atomic_json(tmp_path/'progress.json',{'counter':1})



def test_cli_acquires_run_admission_before_model_loading(tmp_path, monkeypatch):
    from contextlib import nullcontext
    import runpy
    from nc_rted.observation_extraction import observation_run_admission
    driver = runpy.run_path(str(Path(__file__).resolve().parents[1]/"scripts/nc_rted_prepare_observations.py"))
    scope = driver['main'].__globals__
    cfg = {"output": str(tmp_path), "reserved_free_bytes": 0, "code_sha256": {}, "reactvau_python_sha256": {},
           "manifest_directory": str(tmp_path), "provenance_sha256": "fixed", "media": {}, "pts": {}}
    monkeypatch.setitem(scope, 'validate_config', lambda *args: (cfg, tmp_path, {}, {}, {}))
    monkeypatch.setitem(scope, 'validate_local_source', lambda *args: None)
    monkeypatch.setitem(scope, 'verified_source_imports', lambda *args: nullcontext())
    import nc_rted.observation_extraction as extraction
    monkeypatch.setattr(extraction, 'load_detection_inputs', lambda *args: ((), {}))
    called = []
    monkeypatch.setitem(scope, 'run_observation_models', lambda *args: called.append(True))
    monkeypatch.setattr(sys, 'argv', ['prepare', '--config', str(tmp_path/'config'), '--config-sha256', 'fixed'])
    with observation_run_admission(tmp_path, reserved_free_bytes=0):
        with pytest.raises(ExtractionError, match='already active'):
            driver['main']()
    assert not called
    driver['main']()
    assert called == [True]


def test_external_admission_supports_complete_extraction_and_releases_on_failure(tmp_path):
    from nc_rted.observation_extraction import observation_run_admission
    truths = (truth(0),); ids = tuple(map(window_id, truths))
    journal = ObservationJournal(tmp_path, {}, ids, reserved_free_bytes=0)
    with observation_run_admission(tmp_path, reserved_free_bytes=0) as admission:
        result = extract_windows(truths, Observer(), journal, admission=admission)
    assert result['computed'] == 1
    with pytest.raises(ExtractionError, match='invalid observation writer admission'):
        with journal.writer(admission=admission):
            pass
