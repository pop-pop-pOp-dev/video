from nc_rted.tracking import (
    COCO_PERSON_CLASS, MAX_DETECTIONS_PER_FRAME, MAX_TRACK_PAIRS, CausalTrack,
    Detection, FrameObservations, TrackingStatus, causal_tracks,
)


def det(box, cls=COCO_PERSON_CLASS, confidence=0.9, appearance=(1.0, 0.0)):
    return Detection(box, cls, confidence, appearance)


def test_forward_matching_is_causal_and_uses_iou_and_appearance():
    prefix = [
        FrameObservations(0.0, (det((0.1, 0.1, 0.3, 0.3)),)),
        FrameObservations(0.5, (det((0.11, 0.1, 0.31, 0.3)),)),
    ]
    result = causal_tracks(prefix)
    assert result.status is TrackingStatus.OK
    assert len(result.tracks) == 1
    assert [item.frame_index for item in result.tracks[0].observations] == [0, 1]
    extended = causal_tracks(prefix + [FrameObservations(1.0, (det((0.12, 0.1, 0.32, 0.3)),))])
    assert extended.tracks[0].observations[:2] == result.tracks[0].observations
    split = causal_tracks([FrameObservations(0.0, (det((0.1, 0.1, 0.3, 0.3)),)),
                           FrameObservations(0.5, (det((0.11, 0.1, 0.31, 0.3), appearance=(0., 1.)),))])
    assert len(split.tracks) == 2


def test_short_gap_and_invalid_observations_have_explicit_statuses():
    stale = causal_tracks([FrameObservations(0., (det((0.1, 0.1, 0.3, 0.3)),)),
                           FrameObservations(1.5, (det((0.1, 0.1, 0.3, 0.3)),))])
    assert len(stale.tracks) == 2
    assert causal_tracks([]).status is TrackingStatus.EMPTY_OBSERVATIONS
    invalid = causal_tracks([FrameObservations(0., (det((0.3, 0.2, 0.1, 0.4)),))])
    assert invalid.status is TrackingStatus.INVALID_INPUT
    assert invalid.invalid_reason


def test_frame_cap_tie_policy_and_pair_person_constraint_are_deterministic():
    many = tuple(det((i / 100, 0.0, i / 100 + .005, .1), confidence=.5) for i in range(10))
    capped = causal_tracks([FrameObservations(0., many)])
    assert len(capped.tracks) == MAX_DETECTIONS_PER_FRAME
    assert capped.tracks[0].observations[0].box_xyxy == many[0].box_xyxy
    non_person = causal_tracks([FrameObservations(0., (det((0, 0, .1, .1), cls=1),))])
    assert non_person.status is TrackingStatus.NO_PERSON_ENDPOINT
    frame = tuple(det((i / 30, 0, i / 30 + .02, .1), confidence=.9 - i / 100) for i in range(8))
    pairs = causal_tracks([FrameObservations(0., frame)])
    assert pairs.status is TrackingStatus.OK
    assert len(pairs.pairs) == MAX_TRACK_PAIRS
    assert all(pair.first_track < pair.second_track for pair in pairs.pairs)
    assert pairs.pairs == causal_tracks([FrameObservations(0., tuple(reversed(frame)))]).pairs


def test_tracks_do_not_expose_detector_or_source_identifiers():
    result = causal_tracks([FrameObservations(0., (det((0, 0, .2, .2)),))])
    observation = result.tracks[0].observations[0]
    assert isinstance(result.tracks[0], CausalTrack)
    assert not {"source_id", "track_id", "global_id", "appearance"} & set(observation.__dataclass_fields__)
