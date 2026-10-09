#!/usr/bin/env python3
"""Build metadata-only NC-RTED manifests. This process never opens test answers."""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nc_rted.manifests import *

def main():
 p=argparse.ArgumentParser(); p.add_argument("--root",type=Path,required=True); p.add_argument("--output",type=Path,required=True); a=p.parse_args(); root=a.root
 inventory=json.loads((root/"reports/nc_rted/previously_inspected_families_inventory.json").read_text())
 forced=previously_inspected_union(root,inventory)
 db={"ucf-crime":json.loads((root/"data/reactvau/HIVAU-70k/raw_annotations/ucf_database_train.json").read_text()),"xd-violence":json.loads((root/"data/reactvau/HIVAU-70k/raw_annotations/xd_database_train.json").read_text())}
 identity=json.loads((root/"data/reactvau/manifests/official_identity_map_20260929_v1.json").read_text())["rows"]
 legal={("ucf-crime" if row['dataset']=='ucf' else "xd-violence",row['id']) for row in identity if row.get('split')=='train' and row.get('status')=='matched' and Path(row.get('source','')).is_file()}
 db={dataset:{key:value for key,value in rows.items() if (dataset,key) in legal} for dataset,rows in db.items()}
 splits, disclosure=source_splits(db,forced,identity)
 instructions=json.loads((root/"data/reactvau/derived/instruction_available_97140_20260930/train.json").read_text())
 captions=select_caption_rows(instructions,splits,per_family_cap=4); prefixes=select_detection_prefixes(db,splits,eligible_sources=legal); media=derived_media_map(instructions,db)
 vau=vau_identity_rows(root/"data/reactvau/HIVAU-70k/instruction/merge_instruction_test_final.jsonl")
 vad=official_vad_identity(identity)
 from collections import Counter
 caption_concentration=Counter(canonical_family(row['dataset'],row['parent_key']) for row in captions)
 detection_concentration=Counter(row['family'] for row in prefixes)
 report={"source_rows":len(splits),"caption_rows":len(captions),"detection_prefixes":len(prefixes),"derived_media":len(media),"vad_rows":len(vad),"vau_rows":len(vau),"vau_unique_media":len({r['video'] for r in vau}),"caption_max_rows_per_family":max(caption_concentration.values()),"detection_max_rows_per_family":max(detection_concentration.values()),"detection_cap_interpretation":"4 prefixes per family per current-label class; at most 8 total across normal/anomalous",**disclosure}
 inputs=[root/"reports/nc_rted/previously_inspected_families_inventory.json",root/"data/reactvau/manifests/official_identity_map_20260929_v1.json",root/"data/reactvau/derived/instruction_available_97140_20260930/train.json",root/"data/reactvau/HIVAU-70k/raw_annotations/ucf_database_train.json",root/"data/reactvau/HIVAU-70k/raw_annotations/xd_database_train.json",root/"data/reactvau/HIVAU-70k/instruction/merge_instruction_test_final.jsonl"]
 provenance={"schema":"nc_rted_manifest_provenance/v1","selection_config":{"prefixes_per_family":PREFIXES_PER_FAMILY,"caption_count":2000,"detection_per_cell":1500},"inputs":{str(path):file_hash(path) for path in inputs},"test_answers_read":False,"technical_missing_train_sources":5443-len(splits)}
 publish_manifest_set(a.output,{"source_splits.json":splits,"train8000_captions.json":captions,"train8000_detection_prefixes.json":prefixes,"derived_media_map.json":media,"official_vad_identity.json":vad,"official_vau_identity.json":vau,"completeness_report.json":report},provenance)
if __name__=='__main__': main()
