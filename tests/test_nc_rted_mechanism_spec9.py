import pytest
import torch

from nc_rted.features import FrozenFrameFeatures, SIGLIP_FEATURE_DIM, assemble_relation_features
from nc_rted.mechanism_spec9 import Spec9Error, recompute_geometry_checks, teacher_summary
from nc_rted.tracking import Detection, FrameObservations, causal_tracks


def test_teacher_summary_reports_valid_support_and_rejections():
    report = teacher_summary([
        {"aux_valid": True, "mask": [True, False, True, False], "F_positions": [.2, 0., .8, 0.],
         "relation_ids": ["a"], "F_quality": .7},
        {"aux_valid": False, "rejection": "no reference"},
    ])
    assert report["coverage"] == .5
    assert report["relation_marginal"] == {0: 1.0}
    assert report["rejection"] == {"no reference": 1}


def test_teacher_summary_rejects_invalid_teacher_shape():
    with pytest.raises(Spec9Error):
        teacher_summary([{"aux_valid": True, "mask": [True], "F_positions": [], "relation_ids": [], "F_quality": .1}])


def test_geometry_recomputation_checks_a_common_time_varying_translation():
    detection = lambda box, cls: Detection(box, cls, .9, (1., 0.))
    tracking = causal_tracks([
        FrameObservations(2.2, (detection((.1, .1, .3, .3), 0), detection((.5, .1, .7, .3), 2))),
        FrameObservations(3.2, (detection((.1, .1, .3, .3), 0), detection((.6, .1, .8, .3), 2))),
    ])
    frames = tuple(FrozenFrameFeatures(timestamp, torch.ones(729, SIGLIP_FEATURE_DIM)) for timestamp in (2.2, 3.2))
    report = recompute_geometry_checks(assembler=assemble_relation_features, tracking=tracking, frames=frames,
                                       query_s=10., translation_xy=(.01, 0.), trajectory_translation_xy=(.02, 0.))
    assert report["raw_coordinate_translation_preserves_relative_geometry"]
    assert report["common_time_varying_translation_preserves_relative_velocity"]
