# NC-RTED Retrieval Contract

`nc_rted.retrieval` is a stateless pre-teacher helper. It accepts only static
context and source provenance. It has no process/motion descriptor, teacher
distance, label, test-data, detector hash, or teacher-coverage input. Full
teacher construction and alignment remain pending in their dedicated layers.

Each `StaticWindow` has a source family, a content alias, an explicit
`normal_permitted` flag, reliable frozen background, normalized class
composition, and static pair records. A pair uses an ordered COCO class pair,
initial signed geometry, candidate-pair count `[1,16]`, and valid-cell count
`[1,4]`. The orientation is never swapped: `(person, object)` differs from
`(object, person)`.

The frozen static descriptor has four equally weighted blocks:

1. Background distance is `1 - clipped_cosine(background_q, background_r)`,
   with a fixed norm epsilon `1e-8`.
2. Class composition distance is RMS divided by its vector dimension.
3. Initial geometry distance is RMS divided by its five dimensions.
4. Support distance is RMS over `[candidate_pair_count / 16, valid_cell_count / 4]`.

The returned static distance is the arithmetic mean of these four values.
Background vectors are frozen at 1152 dimensions and class compositions at
COCO-80 dimensions. Missing/unreliable query background or invalid support
causes explicit refusal; invalid candidate windows are skipped and counted in
the refusal reason. No zero-filled fallback is used. Pair matching first
requires identical ordered COCO class pairs and initial-geometry RMS `<= 0.25`,
then ranks by this static distance. Motion/process values cannot affect
selection.

All selection calls require an explicit finite nonnegative compatibility
threshold, fitted from R as below. Distances larger than it are excluded from
both references and calibration before support is checked. Reference selection
requires exactly three normal-permitted pairs from distinct source families and
content aliases, excluding the query family and alias. Ties use SHA-256 of the
fixed retrieval version, window ID, and pair ID. Calibration excludes all Q and
R families/aliases, retains at most 128 windows, requires at least 64 windows
and 16 families, and caps each family at four. Different windows from the same
family/content alias are allowed up to that cap; a content alias cannot appear
under two families. Window IDs must be unique.
These checks make Q/C/R source-family and content-alias sets disjoint.

The R compatibility threshold is the nearest-rank 95th percentile of each
normal R record's nearest compatible static descriptor from another source
family and content alias. The rank is `ceil(.95 * n) - 1` after ascending sort.
It is fit only on R records; no test fitting is available or permitted.

`assert_role_isolation` is required before full-pool construction. It verifies
that the entire Q, C, and R window pools have pairwise-disjoint source-family
and content-alias sets, rather than checking only the eventual selected rows.
R records without a compatible leave-one-source neighbor fail explicitly as
unsupported; they do not silently disappear from threshold fitting.
