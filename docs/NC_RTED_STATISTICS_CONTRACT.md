# NC-RTED detection statistics contract

This module is written before blind evaluation and accepts only caller-provided
prediction records.  It neither opens labels nor discovers inputs from a
metrics directory.

## Records and validation

One record contains a stable `sample_id`, `source_id`, binary label, and finite
score.  Every group and seed in a comparison must contain exactly the same
`sample_id -> (source_id, label)` mapping.  Each evaluated dataset must include
both classes.  Duplicate sample IDs, non-finite scores, non-binary labels, and
an empty source are errors.

## Full-sample metrics

The point estimate recomputes AUROC or AP across every scored frame with unit
weight. It is not an average of source/video metrics. AUROC is the weighted
probability that a positive score exceeds a negative score, with half credit
for ties. AP uses threshold groups, summing each recall increment times the
precision after the complete tied-score group. Unit or arbitrary positive
weights have the same definitions as sklearn's `roc_auc_score` and
`average_precision_score` with `sample_weight`.

Records are sorted once by descending score.  Bootstrap metric calls apply
source draw multiplicities to this fixed ordering, using weight
`multiplicity[source]` for every sample from that source; no per-video metric
average or source-length normalization is used. All-one multiplicities are
therefore the official unweighted metric.

## Paired source-family bootstrap

The caller supplies all A/U/S/F predictions for seeds 17, 42, and 2026.  For
each of 10,000 reproducible draws, this module samples the full set of source
families with replacement once per dataset.  That dataset's same multiplicity
vector is then used for every group and every seed.  For each requested comparison it computes the
candidate-minus-baseline difference for each seed and then takes the three-seed
mean.  The reported effect uses the original full source set; the 95% interval
is the uncentered 2.5th and 97.5th percentile interval of valid draws.

A draw that loses either class for a metric is never replaced or selected by
its result.  It is counted as degenerate.  It is absent only from the
mathematically undefined percentile distribution and counts conservatively as
an extreme draw in the centered-null p-value numerator. The result exposes
both requested and valid repeat counts. Any nonzero count sets
`inference_status=INFERENCE_UNCONFIRMED_DEGENERATE_RESAMPLES`; its interval is
descriptive and cannot support a confirmatory significance claim.

For the two-sided null test, let `d` be the original seed-mean effect and
`d*` a bootstrap effect.  The null replicate is `d* - d`; the raw p value is
`(1 + count(abs(d* - d) >= abs(d)) + degenerate_count) / (B + 1)`.
Six predeclared tests, F-A/F-U/F-S for UCF AUROC and XD AP, receive Holm's
step-down adjustment. Results also retain every seed's full-data effect plus
the mean and sample SD (`ddof=1`). Three seeds express training variability but do
not create three independent source cohorts.
