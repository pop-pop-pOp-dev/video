# Frozen observation numerical contract

The added relation observer encodes every cache-missing frame with a singleton
SigLIP forward. This is part of the observation definition, applied in both cold
and warm paths. The original ReactVAU detection and caption adapters retain their
original task-specific batch sizes.

On the inherited BF16 SigLIP tower and RTX 4090, identical RGB and preprocessed
pixels yielded identical outputs across repeated, reordered and shifted batches
of 16. The same frame computed alone differed by up to 16.9140625 from its batched
result. The previous observer batched only cache misses; overlap versus fresh
computation therefore changed feature values (observed maximum difference
10.6875). Both failed probes are preserved. No tolerance was relaxed.

Singleton observation encoding removes dependence on cache occupancy. Its policy
and observer source SHA enter the frame-cache key. The frozen RT-DETR identity
also binds implementation sources, dtype and device. Normalized box metadata uses
native Python floats so strict weights-only loading needs no NumPy allowlist.

A real GPU rerun on one training video compared 16 common cached frames: every
patch tensor matched exactly, including dtype, and maximum difference was zero.
Cold 16-frame observation took about 2.62 seconds; the one-new-frame overlap took
about 1.02 seconds; peak allocated GPU memory was 1,130,277,888 bytes. These are
single-video observations, not full-system throughput estimates. The video window
had no relation pairs, so its feature equality alone does not validate a nonempty
relation branch. Slow probabilities, greedy text and longest-input backward remain
separate acceptance requirements.

The regression suite includes a batch-sensitive BF16 encoder with nonempty pairs,
checks overlap versus fresh relation features and exact cached patches, and checks
that shared frames are encoded only once. Scoped detector/cache/observer tests:
29 passed. Independent supplied-source Astra review: PASS_STATIC; no reviewer-run
tests or full-spec acceptance is implied.

Local evidence: reports/nc_rted/patch_diagnosis_v1.json,
real_observer_gpu_v1.json through real_observer_gpu_v3.json, and
review_cli/astra_detector_numerics_review_v1.json. Formal execution remains gated.

## Canonical tensor layout

The first 12-window nonempty diagnostic exposed a second cache dependence:
original SigLIP tensors had column-major patch strides `(1,729)`, while persisted
cache tensors were contiguous with strides `(1152,1)`. Patch bytes and detector
outputs agreed, but region matrix multiplication followed different reductions;
a few BF16 pooled values differed by up to 0.0078125. Cold tensors now become
contiguous before pooling, tracking appearance or global reductions, matching
the cache representation without changing their values or dtype. A regression
injects a column-major BF16 encoder output and checks the actual pooling inputs
and cold/warm relation results.

Evidence is retained in `pooling_trace_gpu_v1.json`. The initial failed fixed
12-window run remains `fixed_observer_gpu_v1.json`; the corrected run is
`fixed_observer_gpu_v2.json`. All 12 predetermined windows now have nonempty relations and exact cold/warm/
overlap patch and relation feature equality. Thirty focused tests passed; the
second exact-source Astra review returned PASS_STATIC. Full CPU regression is
recorded separately before publication. Slow output acceptance remains pending.
