import sys
from pathlib import Path
import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nc_rted.model import RelationTimeEvidence


def inputs(blocks=2):
    torch.manual_seed(9)
    features = torch.randn(1, blocks, 3, 4, 5)
    valid = torch.tensor([[[[1, 1, 1, 1], [1, 1, 0, 0], [0, 0, 0, 0]]] * blocks], dtype=torch.bool)
    times = torch.arange(blocks * 4, dtype=torch.float32).reshape(1, blocks, 4)
    return features, valid, times


def test_candidate_permutation_is_equivariant():
    model = RelationTimeEvidence(5, 7).eval(); features, valid, times = inputs()
    output = model(features, valid, times); permutation = torch.tensor([2, 0, 1])
    changed = model(features[:, :, permutation], valid[:, :, permutation], times)
    assert torch.allclose(output.quality, changed.quality)
    assert torch.allclose(output.position[:, :, permutation], changed.position, atol=1e-6)
    assert torch.allclose(output.evidence_tokens, changed.evidence_tokens, atol=1e-6)


def test_disabled_and_no_candidate_paths_are_exact_bypasses():
    model = RelationTimeEvidence(5, 7); features, valid, times = inputs(); output = model(features, valid, times)
    original = torch.randn(1, 9, 7)
    assert model.inject(original, output, enabled=False) is original
    empty = model(features, torch.zeros_like(valid), times)
    assert model.inject(original, empty) is original and torch.isfinite(empty.evidence_tokens).all()
    mixed = model(torch.cat((features, features)), torch.cat((valid, torch.zeros_like(valid))), torch.cat((times, times)))
    with pytest.raises(ValueError): model.inject(torch.randn(2, 9, 7), mixed)


def test_position_and_quality_change_actual_evidence_tokens():
    model = RelationTimeEvidence(5, 7).eval(); features, valid, times = inputs(); baseline = model(features, valid, times)
    moved = model(features * torch.tensor([1, 3, 1, 1, 1]).view(1, 1, 1, 1, 5), valid, times)
    assert not torch.allclose(baseline.position, moved.position)
    assert not torch.allclose(baseline.evidence_tokens, moved.evidence_tokens)
    with torch.no_grad(): model.quality_head.bias.fill_(-30)
    gated = model(features, valid, times)
    assert gated.evidence_tokens.abs().max() < baseline.evidence_tokens.abs().max()


def test_encoder_resampler_projection_receive_gradients_and_loss_modes_are_finite():
    model = RelationTimeEvidence(5, 7); features, valid, times = inputs(); output = model(features, valid, times)
    quality = torch.full((1, 2), .5); target = output.position.detach()
    loss = model.auxiliary_loss(output, quality, target, "F") + output.evidence_tokens.square().mean(); loss.backward()
    assert model.temporal_encoder.layers[0].self_attn.in_proj_weight.grad.abs().sum() > 0
    assert model.resampler_queries.grad.abs().sum() > 0 and model.language_projection.weight.grad.abs().sum() > 0
    assert model.auxiliary_loss(output, quality, target, "A") == 0

def test_auxiliary_valid_is_independent_and_keeps_exact_zero_target():
    model = RelationTimeEvidence(5, 7); features, valid, times = inputs(); output = model(features, valid, times, aux_valid=torch.zeros(1,2,dtype=torch.bool))
    nan_target = torch.full_like(output.position, float('nan'))
    assert model.auxiliary_loss(output, torch.zeros(1,2), nan_target, 'F') == 0
    output = model(features, valid, times, aux_valid=torch.ones(1,2,dtype=torch.bool))
    loss = model.auxiliary_loss(output, torch.zeros(1,2), torch.zeros_like(output.position), 'S')
    assert torch.isfinite(loss) and loss > 0

def test_invalid_nan_padding_and_empty_candidates_are_safe_but_valid_nan_rejects():
    model=RelationTimeEvidence(5,7); features,valid,times=inputs()
    padded=features.clone(); padded[~valid]=float('nan')
    output=model(padded,valid,times,aux_valid=torch.ones(1,2,dtype=torch.bool))
    loss=model.auxiliary_loss(output,torch.zeros(1,2),torch.zeros_like(output.position),'S') + output.evidence_tokens.square().mean()
    loss.backward(); assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    bad=features.clone(); bad[valid.nonzero()[0].unbind()]=float('nan')
    with pytest.raises(ValueError): model(bad,valid,times)
    empty=model(torch.empty(1,1,0,4,5),torch.empty(1,1,0,4,dtype=torch.bool),torch.zeros(1,1,4))
    assert empty.position.shape==(1,1,0,4) and not empty.valid_blocks.any()

def test_qnan_in_ineligible_sample_does_not_poison_another_sample():
    model=RelationTimeEvidence(5,7); features,valid,times=inputs()
    features=torch.cat((features,features)); valid=torch.cat((valid,valid)); times=torch.cat((times,times))
    output=model(features,valid,times,aux_valid=torch.tensor([[True,True],[False,False]]))
    q=torch.tensor([[.2,.5],[float('nan'),float('nan')]])
    pi=output.position.detach().clone(); pi[1]=float('nan')
    assert torch.isfinite(model.auxiliary_loss(output,q,pi,'F'))

def test_long_observed_times_have_finite_gradients():
    model=RelationTimeEvidence(5,7); features,valid,times=inputs(); times.fill_(10_000_000)
    output=model(features,valid,times); output.evidence_tokens.square().mean().backward()
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None)

def test_per_relation_timestamps_change_output_and_invalid_timestamps_are_masked():
    model=RelationTimeEvidence(5,7).eval(); features,valid,times=inputs()
    relation_times=times.unsqueeze(2).expand(-1,-1,3,-1).clone(); relation_times[:,:,1] += 4
    broadcast=model(features,valid,times); per_relation=model(features,valid,relation_times)
    assert not torch.allclose(broadcast.evidence_tokens,per_relation.evidence_tokens)
    relation_times[~valid]=float('nan'); assert torch.isfinite(model(features,valid,relation_times).evidence_tokens).all()
    relation_times=times.unsqueeze(2).expand(-1,-1,3,-1).clone(); relation_times[valid.nonzero()[0].unbind()]=-1
    with pytest.raises(ValueError): model(features,valid,relation_times)

def test_teacher_position_cannot_assign_mass_to_invalid_observation_cell():
    model=RelationTimeEvidence(5,7); features,valid,times=inputs(); output=model(features,valid,times,aux_valid=torch.ones(1,2,dtype=torch.bool))
    target=output.position.detach().clone(); target[0,0,2,0]=.1; target[0,0,0,0]-=.1
    with pytest.raises(ValueError): model.auxiliary_loss(output,torch.ones(1,2),target,'F')

def test_bf16_branch_accepts_fp32_long_timestamps_without_precasting_time():
    model=RelationTimeEvidence(5,7).to(dtype=torch.bfloat16)
    features,valid,times=inputs(); features=features.to(torch.bfloat16); times=(times+10_000_000).float()
    output=model(features,valid,times)
    loss=output.evidence_tokens.float().square().mean(); loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)

def test_bf16_quality_sigmoid_keeps_moderate_saturation_gradient_in_fp32():
    model=RelationTimeEvidence(5,7).to(dtype=torch.bfloat16)
    with torch.no_grad(): model.quality_head.bias.fill_(8)
    features,valid,times=inputs(); output=model(features.to(torch.bfloat16),valid,times.float(),aux_valid=torch.ones(1,2,dtype=torch.bool))
    loss=model.auxiliary_loss(output,torch.zeros(1,2),torch.zeros_like(output.position),'S'); loss.backward()
    assert output.quality.dtype==torch.float32 and torch.isfinite(loss) and model.quality_head.bias.grad.abs().sum() > 0


def test_caption_uses_early_and_late_blocks():
    model = RelationTimeEvidence(5, 7).eval(); features, valid, times = inputs(3)
    full = model(features, valid, times).evidence_tokens
    early_changed = features.clone(); early_changed[:, 0] += 5
    late_changed = features.clone(); late_changed[:, -1] += 5
    assert not torch.allclose(full, model(early_changed, valid, times).evidence_tokens)
    assert not torch.allclose(full, model(late_changed, valid, times).evidence_tokens)
