import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nc_rted.manifests import ManifestError, canonical_family, derived_media_map, official_vad_identity, previously_inspected_union, publish_manifest_set, select_caption_rows, select_detection_prefixes, source_splits

def test_xd_aliases_share_a_family_and_split():
    databases={"xd-violence":{"movie__#0":{},"movie__#1":{}},"ucf-crime":{"A":{}}}
    rows, _ = source_splits(databases,set(),[])
    xd=[row for row in rows if row['dataset']=='xd-violence']
    assert canonical_family('xd-violence','movie__#9') == 'xd-violence:movie'
    assert len({row['family'] for row in xd}) == len({row['allocation'] for row in xd}) == 1

def test_train_alias_of_test_media_is_excluded_from_incremental_sampling():
    databases={"ucf-crime":{"A":{}},"xd-violence":{}}
    identities=[{"dataset":"ucf","id":"A","split":"train","status":"matched","sha256":"same"},{"dataset":"ucf","id":"TEST","split":"test","status":"matched","sha256":"same"}]
    rows, disclosure=source_splits(databases,set(),identities)
    assert rows[0]['allocation']=='excluded_test_alias'
    assert disclosure['excluded_incremental_test_alias_families']==['ucf-crime:A']

def test_unknown_or_ambiguous_derived_media_fails_closed():
    records=[{"video":"ucf-crime/clips/train/A_E0C0.mp4"}]
    assert derived_media_map(records,{"ucf-crime":{"A":{}},"xd-violence":{}})[0]['parent_key']=='A'
    with pytest.raises(ManifestError): derived_media_map([{"video":"ucf-crime/events/train/MISSING_E0.mp4"}],{"ucf-crime":{},"xd-violence":{}})

def test_official_vad_denominators_are_exact(tmp_path):
    media=tmp_path/'media'; media.write_text('x')
    valid={"status":"matched","sha256":"hash","source":str(media)}
    rows=[{"dataset":"ucf","split":"test","id":str(i),**valid} for i in range(251)]+[{"dataset":"xd","split":"test","id":str(i),**valid} for i in range(800)]
    assert len(official_vad_identity(rows)) == 1051
    with pytest.raises(ManifestError): official_vad_identity(rows[:-1])

def test_provenance_union_requires_every_declared_family(tmp_path):
    artifact=tmp_path/'artifact.json'; artifact.write_text('{"family":"ucf-crime:A"}')
    inventory={"provenance_sources":[{"path":str(artifact),"family_count":1}],"materialized_union":{"family_count":1}}
    assert previously_inspected_union(tmp_path,inventory)=={'ucf-crime:A'}

def test_detection_labels_actual_causal_query_groups_not_future_windows():
    database={'ucf-crime':{'A':{'fps':1,'n_frames':100,'label':['Abuse'],'events':[[40,60]]}},'xd-violence':{'B':{'fps':1,'n_frames':100,'label':['Abuse'],'events':[[40,60]]}}}
    splits=[{'dataset':dataset,'key':key,'allocation':'train'} for dataset, rows in database.items() for key in rows]
    selected=select_detection_prefixes(database,splits,per_cell=1)
    assert {row['class'] for row in selected}=={'normal','anomalous'}
    normal=[row for row in selected if row['class']=='normal'][0]
    assert normal['observed_seconds'] < 40 and normal['query_index'] >= 0

def test_explicit_normal_label_never_turns_semantic_events_into_anomalies():
    database={'ucf-crime':{'A':{'fps':1,'n_frames':100,'label':[],'events':[[1,99]]}},'xd-violence':{'B':{'fps':1,'n_frames':100,'label':[],'events':[[1,99]]}}}
    splits=[{'dataset':dataset,'key':key,'allocation':'train'} for dataset, rows in database.items() for key in rows]
    with pytest.raises(ManifestError): select_detection_prefixes(database,splits,per_cell=1) # no legal anomalous cell exists

def test_positive_without_intervals_fails_and_reference_boundary_is_not_normal():
    database={'ucf-crime':{'A':{'fps':1,'n_frames':100,'label':['Abuse'],'events':[[8,20]]}},'xd-violence':{'B':{'fps':1,'n_frames':100,'label':['Abuse'],'events':[[8,20]]}}}
    splits=[{'dataset':dataset,'key':key,'allocation':'train'} for dataset, rows in database.items() for key in rows]
    selected=select_detection_prefixes(database,splits,per_cell=1)
    assert not any(row['teacher_reference_normal_eligible'] and row['observed_seconds']==8 for row in selected)
    database['ucf-crime']['A']['events']=[]
    with pytest.raises(ManifestError): select_detection_prefixes(database,splits,per_cell=1)

def test_manifest_publish_is_atomic_and_provenance_hashes_outputs(tmp_path):
    output=tmp_path/'manifests'; publish_manifest_set(output,{'a.json':{'a':1}},{'schema':'test'})
    assert (output/'provenance.json').is_file() and 'a.json' in __import__('json').loads((output/'provenance.json').read_text())['outputs']
    with pytest.raises(ManifestError): publish_manifest_set(output,{'b.json':{}},{})

def test_caption_selection_excludes_analysis_and_judgement_and_caps_family():
    split=[{'dataset':'ucf-crime','key':'A','allocation':'train'},{'dataset':'ucf-crime','key':'B','allocation':'train'}]
    rows=[]
    for key in ('A','B'):
        for number in range(4): rows.append({'id':f'{key}-{number}','video':f'ucf-crime/events/train/{key}_E{number}.mp4','type':'event','task':'description'})
        rows.extend({'id':f'{key}-{task}','video':f'ucf-crime/events/train/{key}_E9.mp4','type':'event','task':task} for task in ('analysis','judgement'))
    picked=select_caption_rows(rows,split,count=4,per_family_cap=2)
    assert {row['task'] for row in picked}=={'description'}
    assert max(__import__('collections').Counter(row['parent_key'] for row in picked).values())==2

def test_source_allocation_is_component_fold_coherent_and_reports_counts():
    database={'ucf-crime':{f'A{i}':{} for i in range(20)},'xd-violence':{f'B{i}':{} for i in range(20)}}
    rows,report=source_splits(database,{'ucf-crime:A0'},[])
    assert rows[0]['allocation']=='train' and report['source_family_allocation_counts']['ucf-crime']['train'] >= 16
def test_alias_component_weights_count_families_and_exclude_test_aliases_from_targets():
    from collections import Counter
    from nc_rted.manifests import source_splits
    db={'ucf-crime':{f'source{i}':{} for i in range(50)},'xd-violence':{}}
    identities=[]
    for i in range(10):
        identities.append(dict(dataset='ucf',id=f'source{i}',split='train',status='matched',sha256=f'pair{i//2}'))
    identities.extend([dict(dataset='ucf',id='source49',split='train',status='matched',sha256='excluded'),
                       dict(dataset='ucf',id='official_test',split='test',status='matched',sha256='excluded')])
    rows,report=source_splits(db,set(),identities)
    actual=Counter(r['allocation'] for r in {r['family']:r for r in rows}.values())
    assert sum(actual.values())==50
    assert report['source_family_allocation_counts']['ucf-crime']==dict(actual)
    assert report['eligible_source_family_counts']['ucf-crime']==49
    assert report['allocation_targets']['ucf-crime']['train']==.8*49
    for i in range(0,10,2):
        pair=[r for r in rows if r['key'] in {f'source{i}',f'source{i+1}'}]
        assert len({(r['allocation'],r['fold']) for r in pair})==1
