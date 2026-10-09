from dataclasses import replace

import numpy as np
import pytest
import torch

from nc_rted.batches import caption_block_endpoints, pack_observation_blocks
from nc_rted.features import FeatureAssemblyResult, FeatureStatus, FrozenFrameFeatures, assemble_relation_features
from nc_rted.model import RelationTimeEvidence
from nc_rted.tracking import Detection, FrameObservations, causal_tracks


def window(query, timestamp):
    detections=(Detection((.1,.1,.3,.3),0,.9,(1.,0.)),Detection((.5,.1,.7,.3),2,.9,(1.,0.)))
    tracking=causal_tracks([FrameObservations(timestamp,detections)])
    return assemble_relation_features(tracking,(FrozenFrameFeatures(timestamp,torch.ones(729,1152)),),query)


def test_pack_retains_early_and_late_blocks_with_actual_pair_times():
    blocks=[window(8.,2.),window(16.,15.5)]
    batch=pack_observation_blocks(blocks,task='caption')
    assert batch.features.shape==(1,2,1,4,4620)
    assert batch.observed_times.shape==(1,2,1,4)
    assert batch.observed_times[batch.valid].tolist()==[2.,15.5]
    assert batch.features.dtype==torch.bfloat16 and batch.observed_times.dtype==torch.float32
    assert torch.isnan(batch.features[~batch.valid]).all()
    branch=RelationTimeEvidence(4620,24,width=24,heads=4).bfloat16().eval()
    out=branch(batch.features,batch.valid,batch.observed_times)
    assert out.valid_blocks.tolist()==[[True,True]] and torch.isfinite(out.evidence_tokens).all()


def test_empty_candidates_remain_empty_and_failures_do_not_become_normal():
    batch=pack_observation_blocks([FeatureAssemblyResult(FeatureStatus.NO_RELATION_PAIRS,())],task='detection')
    assert batch.features.shape==(1,1,0,4,4620)
    with pytest.raises(ValueError,match='technical'):
        pack_observation_blocks([FeatureAssemblyResult(FeatureStatus.TRACKING_FAILURE,())],task='caption')
    with pytest.raises(ValueError,match='one legal'):
        pack_observation_blocks([window(8.,2.),window(16.,15.)],task='detection')


def test_complete_caption_partition_includes_partial_tail():
    assert caption_block_endpoints(16.)==(8.,16.)
    assert caption_block_endpoints(17.25)==(8.,16.,17.25)
    assert caption_block_endpoints(.5)==(.5,)
    with pytest.raises(ValueError):caption_block_endpoints(float('nan'))
