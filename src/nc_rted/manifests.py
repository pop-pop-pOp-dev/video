"""Fail-closed metadata manifests for NC-RTED; never reads test answers/labels."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import os
import tempfile

DATASETS = {"ucf-crime": "ucf", "xd-violence": "xd"}
PREFIXES_PER_FAMILY = 4

class ManifestError(ValueError): pass

def canonical_family(dataset: str, key: str) -> str:
    if dataset not in DATASETS: raise ManifestError(f"unknown dataset {dataset}")
    return f"{dataset}:{key.split('__#', 1)[0] if dataset == 'xd-violence' else key}"

def stable_rank(*parts: object) -> str:
    return hashlib.sha256("\0".join(map(str, parts)).encode()).hexdigest()

def previously_inspected_union(root: Path, inventory: dict) -> set[str]:
    """Materialize every declared provenance artifact; absence is a hard error."""
    found = set()
    def collect(value, key=""):
        if isinstance(value, dict):
            for name, child in value.items(): collect(child, name)
        elif isinstance(value, list):
            for child in value: collect(child, key)
        elif isinstance(value, str) and "family" in key.lower() and value.startswith(("ucf-crime:", "xd-violence:")):
            found.add(value)
    def evidence_collect(value, local):
        if isinstance(value, dict):
            if value.get("dataset") in DATASETS and isinstance(value.get("video_key"), str): local.add(canonical_family(value["dataset"], value["video_key"]))
            for child in value.values(): evidence_collect(child, local)
        elif isinstance(value, list):
            for child in value: evidence_collect(child, local)
    for source in inventory["provenance_sources"]:
        source_name = source.get("name", "")
        path = Path(source["path"])
        if not path.is_absolute(): path = root / path
        if not path.is_file(): raise ManifestError(f"missing provenance artifact {path}")
        document=json.loads(path.read_text()); local=set()
        if source_name.startswith("evidence_scope"):
            evidence_collect(document, local)
        elif source_name == "renewal_microfit":
            local.update(document.get("selected_families", []))
        else:
            # Generic artifacts already store canonical values under family-named fields.
            def local_collect(value, key=""):
                if isinstance(value,dict):
                    for name, child in value.items(): local_collect(child,name)
                elif isinstance(value,list):
                    for child in value: local_collect(child,key)
                elif isinstance(value,str) and "family" in key.lower() and value.startswith(("ucf-crime:","xd-violence:")): local.add(value)
            local_collect(document)
        if len(local) != source["family_count"]: raise ManifestError(f"provenance extraction mismatch {source_name}: {len(local)}/{source['family_count']}")
        found.update(local)
    expected=inventory["materialized_union"]["family_count"]
    if len(found)!=expected: raise ManifestError(f"forced provenance union mismatch: {len(found)}/{expected}")
    return found

def instruction_parent(record: dict) -> tuple[str, str]:
    """Recover only a known training source key from its derived media filename."""
    video = Path(record["video"])
    dataset = {"ucf-crime": "ucf-crime", "xd-violence": "xd-violence"}.get(video.parts[0] if video.parts else "")
    if not dataset or video.suffix != ".mp4": raise ManifestError(f"unsupported instruction media: {record.get('video')}")
    key = re.sub(r"_E\d+(?:C\d+)?$", "", video.stem)
    if not key: raise ManifestError("empty derived parent")
    return dataset, key

def source_splits(databases: dict[str, dict], forced_families: set[str], identity_rows: list[dict]) -> tuple[list[dict], dict]:
    """Allocate all official train sources at family level, retaining identity warnings."""
    sources = {(dataset, key) for dataset, rows in databases.items() for key in rows}
    train_ids = {(DATASETS[dataset], key) for dataset, key in sources}
    aliases = defaultdict(list)
    for row in identity_rows:
        if row.get("sha256") and row.get("status") == "matched": aliases[row["sha256"]].append(row)
    # Same content cannot be split across an incremental allocation; include its train-side peer family.
    union = {canonical_family(dataset, key): canonical_family(dataset, key) for dataset, key in sources}
    def find(x):
        while union[x] != x: union[x] = union[union[x]]; x = union[x]
        return x
    def join(a, b):
        a, b = find(a), find(b)
        if a != b: union[b] = a
    # XD clips are one source family even when their media checksums differ.
    canonical_members=defaultdict(list)
    for dataset,key in sources: canonical_members[canonical_family(dataset,key)].append(canonical_family(dataset,key))
    for members in canonical_members.values():
        for member in members[1:]: join(members[0],member)
    cross_split_aliases = []; test_alias_train_families = set()
    for group in aliases.values():
        train_group = [row for row in group if (row.get("dataset"), row.get("id")) in train_ids]
        for row in group:
            if row.get("split") != "train" and train_group: cross_split_aliases.append({"sha256": row["sha256"], "train_rows": len(train_group), "other_split": row.get("split")})
            if row.get("split") == "test":
                for train_row in train_group:
                    dataset = next(k for k,v in DATASETS.items() if v == train_row["dataset"])
                    test_alias_train_families.add(canonical_family(dataset, train_row["id"]))
        if train_group:
            first_dataset = next(k for k,v in DATASETS.items() if v == train_group[0]["dataset"])
            first = canonical_family(first_dataset, train_group[0]["id"])
            for row in train_group[1:]:
                dataset = next(k for k,v in DATASETS.items() if v == row["dataset"])
                join(first, canonical_family(dataset, row["id"]))
    groups = defaultdict(list)
    for dataset, key in sources: groups[find(canonical_family(dataset, key))].append((dataset, key))
    components=[]
    for members in groups.values():
        family_names = {canonical_family(dataset, key) for dataset, key in members}
        alias_group = stable_rank("nc-rted-same-content-component-v1", *sorted(family_names))
        components.append({"members":members,"families":family_names,"alias_group":alias_group,"forced":bool(family_names&forced_families),"excluded":bool(family_names&test_alias_train_families)})
    # Allocate alias components jointly. Greedy deterministic minimization targets
    # 80/10/10 source-family counts independently per dataset without moving forced
    # or test-alias components.
    for component in components:
        component["family_weights"] = Counter(name.split(":",1)[0] for name in component["families"])
    # Same-content components stay indivisible, but each distinct source family
    # still counts toward the ratio. Test-alias exclusions are outside its base.
    total=Counter()
    for component in components:
        if not component["excluded"]: total.update(component["family_weights"])
    targets={dataset:{"train":.8*number,"development":.1*number,"diagnostic":.1*number} for dataset,number in total.items()}
    counts={dataset:Counter() for dataset in databases}
    for component in components:
        fixed="excluded_test_alias" if component["excluded"] else "train" if component["forced"] else None
        if fixed:
            component["allocation"]=fixed
            for dataset,weight in component["family_weights"].items(): counts[dataset][fixed]+=weight
    for component in sorted((item for item in components if "allocation" not in item),key=lambda item: stable_rank("nc-rted-source-component-v2",item["alias_group"])):
        def cost(choice):
            score=0.
            for dataset,weight in component["family_weights"].items():
                projected=counts[dataset].copy(); projected[choice]+=weight
                score+=sum((projected[name]-targets[dataset][name])**2 for name in ("train","development","diagnostic"))
            return score
        component["allocation"]=min(("train","development","diagnostic"),key=lambda choice:(cost(choice),choice))
        for dataset,weight in component["family_weights"].items(): counts[dataset][component["allocation"]]+=weight
    result = []
    for component in components:
        members,family_names,alias_group,assigned=component["members"],component["families"],component["alias_group"],component["allocation"]
        for dataset, key in sorted(members):
            result.append({"dataset": dataset, "key": key, "family": canonical_family(dataset, key), "same_content_alias_group": alias_group, "allocation": assigned, "fold": int(stable_rank("nc-rted-fold-v1", alias_group)[:8],16)%5})
    if not all(item["family"] in {x["family"] for x in result} for item in result): raise ManifestError("invalid source result")
    family_counts={dataset:dict(counts[dataset]) for dataset in counts}
    return sorted(result, key=lambda x: (x["dataset"], x["key"])), {"cross_split_same_sha256_aliases": cross_split_aliases, "excluded_incremental_test_alias_families": sorted(test_alias_train_families), "source_family_allocation_counts":family_counts, "eligible_source_family_counts":dict(total), "allocation_targets":targets, "allocation_method":"deterministic indivisible-component allocation weighted by unique source families toward 80/10/10 after test-alias exclusions; forced-train retained"}

def select_caption_rows(instructions: list[dict], split_rows: list[dict], count: int = 2000, per_family_cap: int = 4) -> list[dict]:
    allocation = {(row["dataset"], row["key"]): row["allocation"] for row in split_rows}
    strata = defaultdict(list)
    for record in instructions:
        dataset, key = instruction_parent(record)
        if record.get("task") not in {"caption", "description"}: continue
        if allocation.get((dataset, key)) != "train": continue
        strata[(dataset, record["type"], record["task"])].append(record)
    if not strata: raise ManifestError("no eligible caption rows")
    keys = sorted(strata); chosen = []; family_counts=Counter()
    # Round-robin quotas preserve every legal type/task/dataset stratum.
    base, extra = divmod(count, len(keys))
    for index, key in enumerate(keys):
        rows = sorted(strata[key], key=lambda row: stable_rank("nc-rted-caption-v1", row["id"]))
        quota = base + (index < extra)
        eligible=[]
        for row in rows:
            family=canonical_family(*instruction_parent(row))
            if family_counts[family] < per_family_cap:
                eligible.append(row); family_counts[family]+=1
                if len(eligible)==quota: break
        if len(eligible) < quota: raise ManifestError(f"caption stratum too small under family cap: {key}")
        chosen.extend(eligible)
    return [{"id": row["id"], "type": row["type"], "task": row["task"], "video": row["video"], "dataset": instruction_parent(row)[0], "parent_key": instruction_parent(row)[1]} for row in sorted(chosen, key=lambda x: x["id"])]

def derived_media_map(instructions: list[dict], databases: dict[str, dict]) -> list[dict]:
    """Exact database-key parent mapping for every available derived instruction media."""
    result = {}
    for row in instructions:
        dataset, key = instruction_parent(row)
        if key not in databases[dataset]: raise ManifestError(f"unknown derived media parent {dataset}:{key}")
        entry = {"video": row["video"], "dataset": dataset, "parent_key": key, "family": canonical_family(dataset, key)}
        previous = result.setdefault(row["video"], entry)
        if previous != entry: raise ManifestError("ambiguous derived media parent")
    return [result[key] for key in sorted(result)]

def causal_query_groups(n_frames: int, fps: float):
    """Inherited query schedule, without the historical 600-second truncation."""
    interval=max(1,int(fps/4)); count=(n_frames+interval-1)//interval
    for query_index in range((count+3)//4):
        indices=list(range(query_index*4*interval,min((query_index*4+4)*interval,n_frames),interval))
        if indices: yield query_index, indices

def select_detection_prefixes(databases: dict[str, dict], split_rows: list[dict], per_cell: int = 1500, eligible_sources: set[tuple[str,str]] | None = None) -> list[dict]:
    allocation = {(row["dataset"], row["key"]): row["allocation"] for row in split_rows}
    cells = defaultdict(list)
    for dataset, rows in databases.items():
        for key, annotation in rows.items():
            if allocation.get((dataset, key)) != "train": continue
            if eligible_sources is not None and (dataset,key) not in eligible_sources: continue
            labels=annotation.get("label")
            if not isinstance(labels,list): raise ManifestError(f"missing explicit label {dataset}:{key}")
            raw_events=annotation.get("events", [])
            if not labels:
                # Empty explicit label is normal even when descriptions annotate
                # ordinary semantic events; never infer an anomaly from events.
                events=[]
            else:
                if not raw_events: raise ManifestError(f"positive source lacks anomaly intervals {dataset}:{key}")
                events=[]
                for interval in raw_events:
                    if not isinstance(interval,list) or len(interval)!=2: raise ManifestError(f"invalid anomaly interval {dataset}:{key}")
                    start,end=map(float,interval)
                    if start < 0 or end < start: raise ManifestError(f"invalid anomaly interval {dataset}:{key}")
                    events.append((start,end))
            fps, frames = float(annotation["fps"]), int(annotation["n_frames"])
            if fps <= 0 or frames <= 0: raise ManifestError(f"bad geometry {dataset}:{key}")
            by_family_class={}
            for query_index, indices in causal_query_groups(frames,fps):
                observed_seconds=indices[-1]/fps
                label="anomalous" if any(start <= index/fps <= end for index in indices for start,end in events) else "normal"
                # This flag is distinct from current-query label and only permits a
                # fully observed prior 8-second normal reference window.
                reference_normal = observed_seconds >= 8 and not any(observed_seconds-8 <= end and observed_seconds >= start for start,end in events)
                row={"dataset":dataset,"key":key,"family":canonical_family(dataset,key),"query_index":query_index,"observed_seconds":observed_seconds,"class":label,"teacher_reference_normal_eligible":reference_normal}
                cell=(dataset,label); family=row["family"]
                bucket=by_family_class.setdefault((cell,family),[])
                bucket.append(row)
                bucket.sort(key=lambda item: stable_rank("nc-rted-prefix-v2",item["dataset"],item["key"],item["query_index"]))
                del bucket[PREFIXES_PER_FAMILY:]
            for (cell,_), rows_for_family in by_family_class.items(): cells[cell].extend(rows_for_family)
    selected = []
    for cell in [("ucf-crime","normal"),("ucf-crime","anomalous"),("xd-violence","normal"),("xd-violence","anomalous")]:
        candidates = sorted(cells[cell], key=lambda row: stable_rank("nc-rted-prefix-v2", row["dataset"],row["key"],row["query_index"]))
        per_family = Counter(); picked=[]
        for row in candidates:
            if per_family[row["family"]] >= PREFIXES_PER_FAMILY: continue
            picked.append(row); per_family[row["family"]] += 1
            if len(picked) == per_cell: break
        if len(picked) != per_cell: raise ManifestError(f"not enough legal prefixes for {cell}: {len(picked)}/{per_cell}")
        selected.extend(picked)
    return selected

def official_vad_identity(identity_rows: list[dict]) -> list[dict]:
    wanted = {("ucf","test"):251, ("xd","test"):800}; output=[]
    for dataset, split in wanted:
        rows=[r for r in identity_rows if r.get("dataset")==dataset and r.get("split")==split]
        if len(rows) != wanted[(dataset,split)]: raise ManifestError(f"official {dataset} denominator is {len(rows)}, expected {wanted[(dataset,split)]}")
        if len({row.get("id") for row in rows}) != len(rows): raise ManifestError("duplicate official VAD identity")
        if any(row.get("status") != "matched" or not row.get("sha256") or not Path(row.get("source", "")).is_file() for row in rows): raise ManifestError("unverified official VAD media identity")
        output.extend({k:r.get(k) for k in ("dataset","official_dataset","split","id","status","sha256")} for r in rows)
    return output

def vau_identity_rows(jsonl: Path) -> list[dict]:
    output=[]
    with jsonl.open() as handle:
        for line in handle:
            row=json.loads(line)
            output.append({key:row[key] for key in ("id","type","task","video","prompt")})
    if len(output)!=3339 or len({row["video"] for row in output})!=1369 or len({row["id"] for row in output}) != len(output): raise ManifestError("official VAU identity denominator mismatch")
    return output

def write_new(path: Path, value: object):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists(): raise ManifestError(f"refusing to overwrite {path}")
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")

def file_hash(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""): digest.update(block)
    return digest.hexdigest()

def publish_manifest_set(output: Path, files: dict[str, object], provenance: dict) -> None:
    """Durably publish a complete manifest directory or nothing at the target path."""
    if output.exists(): raise ManifestError("output directory already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary=Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        for name,value in files.items(): write_new(temporary/name,value)
        provenance={**provenance,"outputs":{name:file_hash(temporary/name) for name in files}}
        write_new(temporary/"provenance.json",provenance)
        directory_fd=os.open(temporary,os.O_DIRECTORY); os.fsync(directory_fd); os.close(directory_fd)
        os.replace(temporary,output)
        parent_fd=os.open(output.parent,os.O_DIRECTORY); os.fsync(parent_fd); os.close(parent_fd)
    except Exception:
        for child in temporary.glob("*"): child.unlink()
        temporary.rmdir(); raise
