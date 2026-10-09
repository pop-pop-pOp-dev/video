# NC-RTED Teacher Pipeline Contract

The teacher pipeline consumes only `nc_rted_frozen_feature_records/v1` with
`split=train`. It refuses any other split and does not inspect official-test
labels, media answers, detector outputs, or student/teacher predictions.

Every source family and content alias is bound to one of five train folds. For
target fold `k`, Q is `k`, C is `(k+1)%5`, and R is the remaining folds. Normal
permission is explicit in the input record. R supplies robust per-block scales
for every named `PROCESS_BLOCK_SLICES` block and the leave-one-source static
compatibility threshold. Unsupported R records fail the fold explicitly.

Static retrieval runs before process comparison using one aggregate static
descriptor per window: background, COCO composition, mean initial geometry over
candidate relations, candidate-relation count, and total valid-cell count.
It selects three distinct normal R windows once per Q/C context. Each query or
calibration relation then matches only inside those fixed windows by ordered
COCO class pair and initial geometry; it cannot reselect a different R window.
Process costs use R-only robust scales and the existing fixed short DTW.
Technical static failures, such as missing backgrounds, are excluded from R
before threshold fitting and reported as excluded support. A relation that
cannot match or align masks its four cells; a window is rejected only when no
supported cell remains. Window M uses its remaining valid relation/time cells,
and C uses that whole-window statistic with reported support counts.

For each Q window, C retrieval uses its primary static pair to select 64..128
whole normal C windows, >=16 families, <=4 windows/family. Each selected C
window computes its complete `M` against the same R fold population. Rejection
produces `aux_valid=false` and an explicit reason, never an empty-evidence
target. Valid rows emit F/S-identical masks and quality, F/S positions, and a
65-way joint target. U is assigned after all valid rows per dataset using the
predefined global-M ranking and stable teacher-only window hashes.

Each valid row also emits `relation_ids[E]`, one unique ID for each actual pair
in observation order (`E<=16`). ID `i` maps exactly to flattened
`mask/F_positions[4*i:4*i+4]`; the remaining fixed 64 slots have no synthetic
ID. IDs remain present even when that pair has insufficient supported cells.
They are audit-only and allow the task-input builder to reorder teacher
positions to its observed relation order; they must never enter student
features. Detection manifests should use the reconstructible identity
`detection:{dataset}:{key}:{query_index}`.

The CLI writes once to a new output directory and records SHA-256 hashes of the
frozen input and frozen configuration. It does not claim output success if no
input records or no publishable manifest exists.

The teacher-record adapter binds real assembled feature windows in memory. It
requires trusted train allocation, family, content alias, fold, normal-reference
permission, and ordered COCO class pairs. Technical statuses and missing
backgrounds yield explicit rejection metadata. It does not write full JSON
process arrays; a later writer must preserve this schema in chunked binary form.

The production teacher store uses one compressed, pickle-free NPZ payload per
accepted window and a small JSON index. Payload and index hashes bind every
source/fold/normal-permission identity and relation ordering. Rejections remain
index rows without fabricated arrays, so they remain in the fixed-window
denominator. A commit file binds the index hash and publication atomically
renames a fresh directory without overwriting an artifact. Store creation
refuses to consume the last 20 GiB of free disk; the reader verifies hashes
before passing compact records to the teacher builder.

Relation anchors may have different birth times and therefore different valid
static contexts. The adapter retains all relation pairs in observation order.
For the window static descriptor it takes the equal-weight mean of every
finite, background-valid relation anchor, after stable pair-ID ordering, and
normalizes the matching equal-weight class-composition mean. It rejects only
when no usable anchor background remains. Source truth also binds observed
seconds, and every valid relation observation must lie in the current causal
detection window. The compact batch entry sends valid records to the pipeline
and merges rejected metadata unchanged, requiring every detection window ID to
appear exactly once, including an all-rejected batch.

持久化补充：accepted零候选payload保留(0,5)/(0,4)数组形状；拒绝行仍独立。除每个chunk外，index/commit元数据按块预算，最终rename前再次检查reserve，全拒绝store同样适用。
