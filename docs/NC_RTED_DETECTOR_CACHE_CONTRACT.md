# NC-RTED Detector And Cache Contract

The observation boundary uses an explicit local snapshot of RT-DETR-R50 COCO through Transformers. The snapshot must contain its configuration and preprocessing files, is loaded with `local_files_only=True`, has all parameters frozen, and records SHA-256 for every snapshot file and processor configuration. There is no fallback detector, automatic download, score-derived normal label, or fabricated detection. Empty detector output is an explicit no-instance state.

Media decoding uses PyAV frame timestamps. Frames are sampled on the global causal 2 FPS clock by selecting the last decoded frame at or before each tick inside `(query-8, query]`; no interpolation, nominal-FPS timestamp, or later frame changes a prefix. Overlapping windows reuse a cache record keyed by media SHA-256, actual timestamp, detector identity, and SigLIP identity.

SigLIP records are frozen `[729,1152]` patch tensors produced by the inherited direct RGB resize-to-384 bicubic preprocessing and final `hidden_states[-1]`. Regions are pooled from those same patches using the existing verified area-overlap function. Cache values retain their original tensor dtype and are bounded by LRU eviction; it is not a whole-corpus full-patch store. Cache schema and identity mismatches fail closed.

The CLI currently provides timestamp/media dry-run inspection only. Loading an accepted RT-DETR/SigLIP snapshot and producing frozen observations remains blocked on adapter review and resource acceptance.
# RT-DETR provenance and cached observations

The RT-DETR adapter accepts only a local snapshot that includes
`nc_rted_provenance.json`.  Its provenance must name
`PekingU/rtdetr_r50vd_coco_o365`, `rtdetr_r50vd`, 80 labels, and COCO person
class zero, plus a complete SHA-256 map of every snapshot file other than the
provenance file.  The loaded Transformers config is checked again for
`model_type=rt_detr`, 80 labels, and `id2label[0] == person`.  A local generic
RT-DETR snapshot cannot inherit this binding merely by occupying the directory.
Postprocessed RT-DETR boxes are checked finite, clipped to image bounds, and
discarded only when clipping makes them degenerate; a nonfinite model box is an
explicit detector error rather than an empty observation.

The detector identity and its file hashes are calculated once during adapter
construction.  A frame cache entry holds both frozen SigLIP patches and the
normalized detector detections.  A subsequent causal window therefore does
not rerun either frozen observation stage for a cache hit.

The loaded RT-DETR configuration must also expose an R50 backbone depth vector
of `[3,4,6,3]`; provenance text alone is insufficient. `observe_causal_window`
returns the feature assembly alongside audit-only relation IDs and ordered COCO
class pairs reconstructed from the same tracking result. These fields never
enter student tensors. `build_causal_window` remains a compatibility wrapper
that returns only the feature assembly. A caller may provide `window_start_s`
for a partial caption block: frames are selected only from `(start, query]` on
the global 2 FPS clock, and the exact start is passed to feature assembly.
