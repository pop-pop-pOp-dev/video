# NC-RTED Tracking Contract

`nc_rted.tracking` accepts detector outputs and frozen appearance vectors for
already selected causal frames. It does not decode media, run the detector, or
produce source/global identity features.

The frozen preconfiguration values are: COCO person class `0`, at most eight
detections per frame, at most sixteen unordered track pairs, `IoU >= 0.30`,
cosine similarity `>= 0.70`, and a forward association gap `<= 1.25` seconds.
They are hardcoded before model/configuration freeze and must not be changed in
response to development or official-test outcomes.

Frames must have finite, strictly increasing timestamps. A detection needs a
finite normalized positive-area `xyxy` box, class ID, confidence in `[0, 1]`,
and nonzero finite frozen appearance vector. Invalid input returns
`invalid_input` with a reason; an empty frame sequence, no valid detections,
and no current person endpoint each have separate explicit statuses.

Per frame, detections are ranked by descending confidence, class, box,
appearance vector, and original input position, then capped at eight. Existing
tracks can match only a later detection of the same class inside the fixed gap
that passes both thresholds. Candidate associations rank by descending
`IoU + cosine`; ties resolve by lower local track index then capped detection
rank. Associations are greedily committed forward only. Missing detections are
never interpolated, and later frames never revise earlier assignments.

Only tracks with a final-frame endpoint form pairs. A pair contains two distinct
tracks and at least one endpoint with COCO class `0`; its indexes are ascending,
so it is unordered. Candidates sort by endpoint recency, descending endpoint
confidence sum, then ascending indexes; the first sixteen are retained. Track
indexes are local observation indexes and are never exported as model features.
