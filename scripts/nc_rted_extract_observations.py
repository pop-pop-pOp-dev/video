#!/usr/bin/env python3
"""Metadata-only causal media inspection; model execution requires reviewed bindings."""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import av
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))

def media_timestamps(path: Path) -> np.ndarray:
    """Decode packet/frame timestamps without interpolation or nominal-FPS invention."""
    values=[]
    with av.open(str(path)) as container:
        stream=container.streams.video[0]
        for frame in container.decode(stream):
            if frame.pts is None or frame.time is None: raise RuntimeError("decoded frame lacks actual timestamp")
            values.append(float(frame.time))
    result=np.asarray(values,dtype=np.float64)
    if result.size and (not np.isfinite(result).all() or np.any(np.diff(result)<=0)): raise RuntimeError("nonmonotonic frame timestamps")
    return result

def digest(path: Path) -> str:
    import hashlib
    h=hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
    return h.hexdigest()

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('media',type=Path); parser.add_argument('--query-seconds',type=float,required=True); parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args(); timestamps=media_timestamps(args.media)
    from nc_rted.observation import causal_frame_indices
    indices=causal_frame_indices(timestamps,args.query_seconds)
    print(json.dumps({'media_sha256':digest(args.media),'decoded_frame_count':len(timestamps),'query_seconds':args.query_seconds,'causal_frame_indices':indices.tolist(),'causal_timestamps':timestamps[indices].tolist(),'model_execution':False,'reason':'dry-run only; frozen RT-DETR and SigLIP bindings require accepted local snapshots'}))
if __name__=='__main__': main()
