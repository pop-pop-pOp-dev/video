#!/usr/bin/env python3
"""Run fixed seed17 formal training in bounded, durable execution intervals."""
from __future__ import annotations
import argparse, fcntl, hashlib, json, math, os, shutil, socket, subprocess, sys, tempfile, time
from pathlib import Path
from nc_rted_segmented_materialization import materialize_seed

GROUPS=("A","U","S","F"); PROJECT=Path("/root/autodl-tmp/lookaway-wm"); CUTOFF=1792080000; DEADLINE=1792724040; RESERVE=20<<30
QUALIFICATION_SHA="55945bd56c2885ae87b27ffb6e527e1f22de5ad94a2b7b674c69de4649a9b859"; PROFILE_SHA="4a255928065ac33c30081daa0093ae44511e26c0f5045172dfaf95b60fda545b"; GATES_SHA="a8a579ee3fa5c2d9da77c4ef94e4bb7256fe534808b7d0030b791fc1219b5a2b"
def canonical(v): return (json.dumps(v,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()
def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p): return {"path":str(Path(p).resolve()),"sha256":digest(p)}
def read_bound(b):
 p=Path(b["path"])
 if not p.is_absolute() or p.is_symlink() or digest(p)!=b["sha256"]: raise ValueError(f"bound evidence differs: {p}")
 return json.loads(p.read_text())
def publish(p,d):
 p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists(): raise ValueError(f"immutable output already exists: {p}")
 fd,t=tempfile.mkstemp(prefix=f".{p.name}.",dir=p.parent)
 try:
  with os.fdopen(fd,"wb") as s: s.write(canonical(d));s.flush();os.fsync(s.fileno())
  os.link(t,p); f=os.open(p.parent,os.O_RDONLY|os.O_DIRECTORY);os.fsync(f);os.close(f)
 finally: Path(t).unlink(missing_ok=True)
 return bind(p)
def announce(state,**fields): print(json.dumps({"state":state,"utc_epoch":time.time(),**fields},sort_keys=True),flush=True)
def segment_budget(q,a,b):
 u=float(q["measurements"]["seconds_per_bundle_update_upper_bound"]);s=float(q["measurements"]["setup_checkpoint_seconds_upper_bound"])
 if not(math.isfinite(u) and u>0 and math.isfinite(s) and s>0): raise ValueError("invalid qualified resource measurements")
 return math.ceil((b-a)*u+s)
def build_segment(m,root,start,stop,previous,spent,total,lease,expiry,started):
 q=read_bound(m["qualification"]);budget=segment_budget(q,start,stop)
 if(not 0<=start<stop<=1000 or start%50 or stop%50 or not math.isfinite(spent) or spent<0 or spent+budget>total or started+budget>min(expiry,CUTOFF,DEADLINE,q["valid_until_utc_epoch"]) or shutil.disk_usage(PROJECT).free<RESERVE): raise ValueError("next complete execution interval does not fit the authorized lease/reserve")
 d=root/"segments"/f"{start:06d}-{stop:06d}"; d.mkdir(parents=True,exist_ok=False); key=f"formal-bundle-segment-seed17-source39-{start:06d}-{stop:06d}"; ids=m["member_identities"];bundle=read_bound(m["bundle"]);common=bundle["bundle_checkpoint_root"];runs=m["member_runs"];it=m["interpreter"]
 outs=([{"path":"bundle-segment-report.json","artifact_type":"report","semantic":"formal_bundle_segment","member_identities":ids,"start_update":start,"stop_update":stop,"bundle_checkpoint_root":common}] if stop<1000 else [{"path":str((Path(runs[g]["checkpoint_root"])/"final/manifest.json").resolve()),"artifact_type":"checkpoint","semantic":"formal_training","run_identity":ids[g]} for g in GROUPS])
 if stop==1000:
  checks={g:str((Path(runs[g]["checkpoint_root"])/"final/manifest.json").resolve()) for g in GROUPS};outs.append({"path":"bundle-segment-report.json","artifact_type":"report","semantic":"formal_bundle","member_identities":ids,"final_checkpoints":checks,"bundle_checkpoint_root":common})
 cmd=[it["path"],str(Path(m["source_root"])/"scripts/nc_rted_interleaved_formal.py"),"--bundle",m["bundle"]["path"],"--bundle-sha256",m["bundle"]["sha256"],"--captured-root","{queue_capture}","--output","bundle-segment-report.json","--start-update",str(start),"--stop-update",str(stop),"--resume","auto"]
 auth={"schema":"nc_rted_resource_authorization/v1","status":"PASS","host":q["host"],"gpu_uuid":q["gpu_uuid"],"lease_id":lease,"project_volume":str(PROJECT),"max_budget_seconds":budget,"min_free_bytes":RESERVE,"deadline_utc_epoch":DEADLINE,"lease_expires_utc_epoch":expiry,"source_scope":m["scope"]};ab=publish(d/"authorization.json",auth)
 p={"physical_gpu":0,"bundle_config":m["bundle"]["path"],"bundle_config_sha256":m["bundle"]["sha256"],"member_identities":ids,"qualified_member_identities":m["qualified_member_identities"],"frozen_source_sha256":m["source_sha256"],"start_update":start,"stop_update":stop,"previous_bundle_checkpoint":previous,"source_transition":m["source_transition"],"bundle_evidence":key+":bundle","resource_authorization_evidence":key+":authorization","data_volume":str(PROJECT),"min_free_bytes":RESERVE,"run_budget_seconds":budget,"deadline_utc_epoch":DEADLINE,"execution_environment":m["environment"],"interpreter":it,"run_dir":str(root/"queue-runs"),"progress_path":runs["A"]["progress_path"],"checkpoint_roots":[runs[g]["checkpoint_root"] for g in GROUPS],"bundle_checkpoint_root":common,"expected_outputs":outs,"command":cmd,"resource_attestation":str(d/"resource-attestation.json")}
 bk=("bundle_config","bundle_config_sha256","member_identities","qualified_member_identities","frozen_source_sha256","start_update","stop_update","previous_bundle_checkpoint","source_transition");ek=("command","execution_environment","interpreter","run_dir","progress_path","checkpoint_roots","bundle_checkpoint_root","expected_outputs","start_update","stop_update","previous_bundle_checkpoint");ck=("data_volume","min_free_bytes","run_budget_seconds","deadline_utc_epoch")
 att={"schema":"nc_rted_formal_bundle_segment_resource_attestation/v1","status":"PASS","binding":{"job_key":key,**{k:p[k] for k in bk},"execution_inputs":{k:p[k] for k in ek}},"execution":{"host":q["host"],"physical_gpu":0,"gpu_uuid":q["gpu_uuid"],"lease_id":lease,"lease_expires_utc_epoch":expiry,"environment":m["environment"],"interpreter":it},"qualification":{"accepted_evidence_name":key+":qualification",**m["qualification"]},"authorization":ab,"contract":{k:p[k] for k in ck},"cumulative":{"lease_id":lease,"approved_seconds_before":spent,"approved_seconds_after":spent+budget,"lease_total_seconds":total,"accounting":"prior settled actual seconds plus current reserved upper bound"}}
 p["resource_attestation_sha256"]=publish(d/"resource-attestation.json",att)["sha256"];e={key+":bundle":(m["bundle"]["path"],m["bundle"]["sha256"]),key+":qualification":(m["qualification"]["path"],m["qualification"]["sha256"]),key+":authorization":(ab["path"],ab["sha256"])};publish(d/"payload.json",p);publish(d/"allocation.json",{"job_key":key,"scientific_target_updates":1000,"start_update":start,"stop_update":stop,"started_utc_epoch":started,"reserved_seconds":budget,"settled_actual_seconds_before":spent,"lease_total_seconds":total,"lease_expires_utc_epoch":expiry,"previous_bundle_checkpoint":previous,"source_scope":m["scope"]});return d,key,p,e
def completed_segment(queue,key,payload):
 with queue.connect() as db: row=db.execute("SELECT * FROM jobs WHERE job_key=?",(key,)).fetchone()
 if row is None or row["status"]!="SUCCEEDED": raise ValueError("queue did not publish a successful durable segment")
 rb={"path":row["output_path"],"sha256":row["output_checksum"]}; result=read_bound(rb)
 if result.get("job_key")!=key: raise ValueError("queue result names another job")
 arts=result.get("artifacts")
 if not isinstance(arts,list) or len(arts)!=len(payload["expected_outputs"]): raise ValueError("queue result has a different output denominator")
 reports=[(a,read_bound({"path":a["path"],"sha256":a["checksum"]})) for a in arts if Path(a["path"]).name=="bundle-segment-report.json"]
 if len(reports)!=1: raise ValueError("queue result does not bind exactly one segment report")
 art,report=reports[0];stop=payload["stop_update"];bp=Path(payload["bundle_checkpoint_root"])/"commits"/("final.json" if stop==1000 else f"update_{stop:06d}.json");bb=bind(bp);boundary=read_bound(bb)
 if(boundary.get("schema")!="nc_rted_bundle_checkpoint_v1" or boundary.get("completed_updates")!=stop or boundary.get("members")!=payload["member_identities"] or boundary.get("final") is not(stop==1000)): raise ValueError("completed common boundary differs")
 if report.get("members")!=payload["member_identities"] or report.get("shared_material")!="frozen_provider_only": raise ValueError("completed segment scientific identities differ")
 if stop<1000:
  if(report.get("status")!="FORMAL_BUNDLE_SEGMENT_PAUSED" or report.get("complete") is not False or report.get("start_update")!=payload["start_update"] or report.get("stop_update")!=stop or report.get("total_updates")!=1000 or {"path":report.get("bundle_checkpoint"),"sha256":report.get("bundle_checkpoint_sha256")}!=bb): raise ValueError("partial result was not a durable, nonfinal interval")
 elif report.get("status")!="FORMAL_BUNDLE_COMPLETE": raise ValueError("last interval lacks the complete formal result")
 return rb,{"path":art["path"],"sha256":art["checksum"]},bb
def run_intervals(m,root,lease,expiry):
 from nc_rted.queue import JobQueue
 begun=time.time();total=expiry-begun;publish(root/"controller-run.json",{"schema":"nc_rted_segment_controller_run/v1","pid":os.getpid(),"host":socket.gethostname(),"started_utc_epoch":begun,"lease_id":lease,"lease_expires_utc_epoch":expiry,"lease_total_seconds":total,"scientific_target_updates":1000,"seed":17,"automatic_failure_retry":False,"controller":bind(Path(__file__)),"materialization":bind(root/"materialization.json")});q=read_bound(m["qualification"]);start=0;spent=0.;previous=None;source=Path(m["source_root"])
 while start<1000:
  stop=min(1000,start+100);started=time.time();budget=segment_budget(q,start,stop)
  if started+budget>min(expiry,CUTOFF,DEADLINE,q["valid_until_utc_epoch"]) or spent+budget>total or shutil.disk_usage(PROJECT).free<RESERVE:
   publish(root/"controller-outcome.json",{"state":"PAUSED_RESOURCE_LIMIT","complete":False,"completed_updates_per_group":start,"scientific_target_updates":1000,"settled_actual_seconds":spent,"next_reserved_seconds":budget,"previous_bundle_checkpoint":previous,"utc_epoch":started});announce("PAUSED_RESOURCE_LIMIT",completed_updates_per_group=start);return
  d,key,p,e=build_segment(m,root,start,stop,previous,spent,total,lease,expiry,started);db=d/"queue.sqlite3";queue=JobQueue(db)
  for n,(path,check) in e.items(): queue.add_evidence(n,path,check,accepted=True)
  # Queue metadata reserves its protocol-required retry slots; this controller dispatches once and leaves a failure unsettled for explicit recovery.
  queue.add_job(key,"formal_bundle_segment",p,max_attempts=4);cmd=[m["interpreter"]["path"],str(source/"scripts/nc_rted_queue.py"),"--db",str(db),"worker","--owner",f"{socket.gethostname()}:segment:{start}:{os.getpid()}","--once"];publish(d/"dispatch.json",{"command":cmd,"queue":str(db),"job_key":key,"started_utc_epoch":started,"reserved_seconds":budget,"previous_bundle_checkpoint":previous});announce("SEGMENT_DISPATCHED",job_key=key,start_update=start,stop_update=stop,queue=str(db),reserved_seconds=budget)
  with(d/"queue.stdout.log").open("xb") as out: r=subprocess.run(cmd,cwd=source,env=m["environment"],stdout=out,stderr=subprocess.STDOUT)
  finished=time.time()
  try:
   if r.returncode: raise ValueError(f"queue controller exit {r.returncode}")
   rb,report,previous=completed_segment(queue,key,p)
  except Exception as err:
   publish(d/"unsettled-failure.json",{"state":"REQUIRES_RECOVERY","complete":False,"job_key":key,"started_utc_epoch":started,"observed_utc_epoch":finished,"elapsed_seconds_observed":finished-started,"reserved_seconds":budget,"prior_settled_actual_seconds":spent,"reservation_released":False,"reason":f"{type(err).__name__}: {err}","queue_returncode":r.returncode});announce("REQUIRES_RECOVERY",job_key=key,reservation_released=False);raise
  elapsed=finished-started
  if elapsed<0: raise ValueError("clock moved backwards; resource consumption cannot be settled")
  spent+=elapsed;publish(d/"settlement.json",{"schema":"nc_rted_formal_segment_settlement/v1","state":"FORMAL_BUNDLE_COMPLETE" if stop==1000 else "FORMAL_BUNDLE_SEGMENT_PAUSED","complete":stop==1000,"job_key":key,"start_update":start,"stop_update":stop,"scientific_target_updates":1000,"started_utc_epoch":started,"finished_utc_epoch":finished,"reserved_seconds":budget,"actual_elapsed_seconds":elapsed,"released_unused_seconds":max(0.,budget-elapsed),"cumulative_actual_seconds":spent,"queue_result":rb,"report":report,"common_boundary":previous});announce("SEGMENT_SETTLED",start_update=start,stop_update=stop,actual_elapsed_seconds=elapsed,cumulative_actual_seconds=spent,released_unused_seconds=max(0.,budget-elapsed))
  if elapsed>budget or finished>expiry: raise ValueError("completed segment exceeded its resource envelope; automatic continuation stopped")
  start=stop
 publish(root/"controller-outcome.json",{"state":"FORMAL_SEED17_COMPLETE","complete":True,"completed_updates_per_group":1000,"formal_models_completed":4,"formal_matrix_target":12,"settled_actual_seconds":spent,"final_bundle_checkpoint":previous,"utc_epoch":time.time()});announce("FORMAL_SEED17_COMPLETE",formal_models_completed=4,formal_matrix_target=12)
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--runtime-source-root",type=Path,required=True);p.add_argument("--output-root",type=Path,required=True);p.add_argument("--lease-id",default="pro6000-user-seven-day-rental-20261009");p.add_argument("--lease-expires-utc",type=int,default=CUTOFF);p.add_argument("--run",action="store_true");a=p.parse_args();source,root=a.runtime_source_root.resolve(),a.output_root.resolve()
 if source!=Path(__file__).resolve().parents[1] or PROJECT not in source.parents or PROJECT not in root.parents or a.lease_id!="pro6000-user-seven-day-rental-20261009" or a.lease_expires_utc!=CUTOFF:p.error("source, output or authorized lease differs from this incident-scoped controller")
 sys.path.insert(0,str(source/"src"));inv={"source_root":str(source),"output_root":str(root),"lease_id":a.lease_id,"lease_expiry":a.lease_expires_utc,"qualification":{"path":str(PROJECT/"reports/nc_rted/source39_runtime_qualification_uuid_v1.json"),"sha256":QUALIFICATION_SHA},"profile":{"path":str(PROJECT/"artifacts/nc_rted/source39_formal_qualification_profile_uuid_v1/qualification_profile.json"),"sha256":PROFILE_SHA},"gates":{"path":str(PROJECT/"reports/nc_rted/source39_recovered_gpu_gates_v1/gate_bindings.json"),"sha256":GATES_SHA}};root.parent.mkdir(parents=True,exist_ok=True)
 with(root.parent/f".{root.name}.controller.lock").open("a+") as lock:
  fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);mp=root/"materialization.json"
  if mp.exists():
   stored=json.loads(mp.read_text())
   if stored.get("invocation")!=inv:raise ValueError("immutable materialization invocation differs")
   m=stored["material"]
   for n in("bundle","qualification","profile","gates","scope"):read_bound(m[n])
  else:m=materialize_seed(source_root=source,output_root=root,qualification_binding=inv["qualification"],profile_binding=inv["profile"],gates_binding=inv["gates"],lease_id=a.lease_id,lease_expiry=a.lease_expires_utc);publish(mp,{"schema":"nc_rted_segment_materialization_receipt/v1","invocation":inv,"material":m})
  announce("CPU_MATERIALIZATION_COMPLETE",materialization=bind(mp))
  if a.run:
   if(root/"controller-run.json").exists():raise ValueError("prior controller run exists; preserve it and recover explicitly, never reset resource accounting")
   run_intervals(m,root,a.lease_id,a.lease_expires_utc)
if __name__=="__main__":main()
