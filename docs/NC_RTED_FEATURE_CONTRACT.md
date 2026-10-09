# NC-RTED Feature Contract

`nc_rted.features` transforms causal tracking snapshots and caller-supplied,
frozen SigLIP patch features into relation-window inputs. It does not decode
media, call a detector, generate appearance embeddings, attach source IDs, or
read beyond the current eight-second observation window.

The verified baseline patch feature shape is `[729, 1152]`: the inherited
ReactVAU SigLIP encoder fixes `hidden_size=1152` in
`external/ReactVAU-paper/llava/model/multimodal_encoder/siglip_encoder.py`.
Assembly requires exactly this floating, finite shape for every supplied frozen
frame; its feature dimension is bound to `SIGLIP_FEATURE_DIM = 1152`. A later
backbone change must change this contract and the frozen model input
configuration together.

For every real relation/time cell the fixed student field order is:

`local_first[1152], local_second[1152], joint[1152], global[1152], signed_relative_geometry[5], signed_geometry_change_from_initial[5], relative_center_velocity_from_initial[2]`.

Its dimension is `4620`. The separate process descriptor is:

`initial_signed_geometry[5], current_signed_geometry[5], signed_geometry_change[5], relative_center_velocity[2], local_first_change[1152], local_second_change[1152], joint_change[1152]`.

Its dimension is `3473`. `STUDENT_BLOCK_SLICES` and `PROCESS_BLOCK_SLICES`
export the exact named slices for downstream alignment. Static context remains separate: reliable background
pool `[1152]` plus validity flag, normalized COCO-80 class composition, and
initial signed relative geometry `[5]`. Static fields do not become process
motion fields.

Relation candidates come from any pair co-observed during the entire prior
eight-second window, even when either endpoint is absent at the final frame.
At least one endpoint must be COCO person class `0`. Candidates order by first
co-observed frame and local indexes, then cap at 16. Within a two-second time
cell the latest actual co-observation is used, while baseline geometry, visual
change, background, and class composition always anchor to the earliest actual
shared observation in the window. Velocity is signed relative-center
displacement from that anchor divided by actual elapsed media seconds; it is
explicitly zero at the anchor. It is invariant to equal global translation of
both boxes.

Each relation exports `feature_valid[4]`, actual `observed_times_s[4]` (NaN
for missing cells), and fixed `cell_right_boundaries_s[4]`. The default
detection window is `(q-8, q]`, producing `q-6,q-4,q-2,q`. Caption callers may
pass `window_start_s` for a shorter final block, with `0 < q-start <= 8`; its
cells derive from `(start, q]` and boundaries are `min(start+2k,q)`. This avoids
overlapping a final partial caption block with its predecessor. The boundary
array is a causal grid coordinate for the bridge, never a claim that an
observation occurred at that boundary. Missing cells are represented by an
explicit false mask and `NaN` feature rows; no zero feature, interpolation, or
future-frame backfill is fabricated.

Description callers must process every causal eight-second block across the
full observed media range and only then apply the fixed 16-query aggregation.
Selecting only a final or high-score block is outside this contract.
