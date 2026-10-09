"""Use the real inherited preparation method and real tiny Qwen+PEFT modules.

This is an interface test, explicitly not full 7B/long-input capacity evidence.
"""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
from peft import LoraConfig, get_peft_model

import os
sys.path.insert(0, os.environ.get('NC_RTED_REACTVAU_ROOT', str(Path(__file__).resolve().parents[1] / 'external/ReactVAU-paper')))
from llava.model.llava_arch import LlavaMetaForCausalLM
from nc_rted.bridge import EvidenceSlowBridge, ObservationBatch, SlowInputs, TeacherBatch
from nc_rted.model import RelationTimeEvidence


class TinyInheritedQwen(Qwen2ForCausalLM):
    prepare_inputs_labels_for_LLM = LlavaMetaForCausalLM.prepare_inputs_labels_for_LLM

    def get_model(self):
        return self.model

    def get_vision_tower(self):
        return SimpleNamespace()


def build():
    torch.manual_seed(17)
    config = Qwen2Config(hidden_size=24, intermediate_size=48, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, vocab_size=32,
                         max_position_embeddings=128, attention_dropout=0.)
    config.mm_patch_merge_type = 'flat'
    config.tokenizer_model_max_length = 128
    config.frame_aspect_ratio = 'square'
    config.transformers_version = '4.47.1'
    raw = TinyInheritedQwen(config)
    slow = get_peft_model(raw, LoraConfig(r=2, lora_alpha=4, target_modules=['q_proj','v_proj']))
    evidence = RelationTimeEvidence(6, 24, width=24, heads=4)
    bridge = EvidenceSlowBridge(slow, evidence).eval()
    inputs = SlowInputs(torch.tensor([[1,-200,5,6,7]]), torch.randn(1,3,24),
                        [torch.zeros(1,3,4,4)], [(4,4)],
                        torch.tensor([[-100,-100,-100,6,7]]), torch.ones(1,5,dtype=torch.bool))
    obs = ObservationBatch(torch.randn(1,1,2,4,6), torch.ones(1,1,2,4,dtype=torch.bool),
                           torch.tensor([[[2.,4.,6.,8.]]]))
    return bridge, inputs, obs


def test_disabled_matches_inherited_preparation_exactly_and_no_candidate_is_bypass():
    bridge, inputs, obs = build()
    raw = bridge.raw_slow
    original = raw.prepare_inputs_labels_for_LLM(inputs.input_ids, None, inputs.attention_mask,
                     None, inputs.labels, inputs.images, [inputs.visual_embeddings], ['video'],
                     image_sizes=inputs.image_sizes)
    disabled = bridge.prepare(inputs, None, enabled=False)
    assert disabled.visual_embeddings is inputs.visual_embeddings
    assert disabled.evidence is None
    for key, value in zip(['input_ids','position_ids','attention_mask','past_key_values','inputs_embeds','labels'], original):
        got = disabled.arguments[key]
        assert got is None if value is None else torch.equal(got,value)
    absent = ObservationBatch(obs.features, torch.zeros_like(obs.valid), obs.observed_times)
    no_candidate = bridge.prepare(inputs, absent)
    assert no_candidate.visual_embeddings is inputs.visual_embeddings
    assert torch.equal(no_candidate.arguments['inputs_embeds'], original[4])


def test_new_tokens_reach_real_qwen_loss_and_both_trainable_components():
    bridge, inputs, obs = build()
    bridge.train()
    teacher = TeacherBatch(torch.tensor([[.8]]), torch.ones(1,1,2,4)/8,
                           torch.ones(1,1,dtype=torch.bool))
    loss = bridge(inputs, obs, task='detection', group='F', teacher=teacher)
    loss.loss.backward()
    assert loss.auxiliary_loss > 0
    for prefix in ['evidence.', 'slow.']:
        assert any(p.grad is not None and bool((p.grad != 0).any()) for n,p in bridge.named_parameters() if n.startswith(prefix))
    assert all(p.grad is None for p in bridge.parameters() if not p.requires_grad)
    bridge.eval()
    enabled = bridge.prepare(inputs, obs)
    disabled = bridge.prepare(inputs, None, enabled=False)
    assert enabled.arguments['inputs_embeds'].shape[1] == disabled.arguments['inputs_embeds'].shape[1] + 16
    assert torch.equal(enabled.visual_embeddings[:,:3], inputs.visual_embeddings)
    with torch.no_grad():
        a = bridge.slow(**enabled.arguments).logits[:,-1]
        b = bridge.slow(**disabled.arguments).logits[:,-1]
    assert not torch.equal(a,b)


def test_caption_has_all_blocks_and_zero_auxiliary_supervision():
    bridge, inputs, obs = build()
    long = ObservationBatch(obs.features.repeat(1,3,1,1,1), obs.valid.repeat(1,3,1,1),
                            torch.tensor([[[2.,4.,6.,8.],[10.,12.,14.,16.],[18.,20.,22.,24.]]]))
    result = bridge(inputs, long, task='caption', group='F')
    assert result.auxiliary_loss == 0
    assert result.evidence.valid_blocks.shape == (1,3)
    teacher = TeacherBatch(torch.ones(1,3), torch.ones(1,3,2,4)/8, torch.ones(1,3,dtype=torch.bool))
    with pytest.raises(ValueError, match='caption'):
        bridge(inputs, long, task='caption', group='F', teacher=teacher)


def test_preparation_refuses_silent_truncation_and_missing_sentinel():
    bridge, inputs, obs = build()
    bridge.raw_slow.config.tokenizer_model_max_length = 10
    with pytest.raises(ValueError, match='truncate'):
        bridge.prepare(inputs, obs)
    inputs.input_ids[0,1] = 3
    with pytest.raises(ValueError, match='sentinel'):
        bridge.prepare(inputs, obs)


def test_greedy_disabled_path_matches_qwen_parent_generation():
    from dataclasses import replace
    bridge, inputs, obs = build()
    inputs = replace(inputs, labels=None)
    original = bridge.raw_slow.prepare_inputs_labels_for_LLM(inputs.input_ids, None,
        inputs.attention_mask, None, None, inputs.images, [inputs.visual_embeddings],
        ['video'], image_sizes=inputs.image_sizes)
    decoding = dict(do_sample=False, max_new_tokens=3, use_cache=True, pad_token_id=0, eos_token_id=31)
    with torch.no_grad():
        expected = Qwen2ForCausalLM.generate(bridge.raw_slow, position_ids=original[1],
            attention_mask=original[2], inputs_embeds=original[4], **decoding)
    got = bridge.generate(inputs, None, enabled=False, generation_config=decoding)
    assert torch.equal(expected, got)


def test_bf16_cache_with_float_timestamps_backward_and_generation_mode_guard():
    from dataclasses import replace
    bridge, inputs, obs = build()
    bridge.to(torch.bfloat16)
    inputs = replace(inputs, visual_embeddings=inputs.visual_embeddings.to(torch.bfloat16),
                     images=[x.to(torch.bfloat16) for x in inputs.images])
    obs = replace(obs, features=obs.features.to(torch.bfloat16))
    assert obs.observed_times.dtype == torch.float32
    result = bridge(inputs, obs, task='detection', group='A')
    result.task_loss.backward()
    for prefix in ['slow.','evidence.']:
        assert any(p.grad is not None and bool((p.grad!=0).any()) for n,p in bridge.named_parameters() if n.startswith(prefix))
    generation_inputs=replace(inputs,labels=None)
    decoding=dict(do_sample=False,max_new_tokens=2,pad_token_id=0,eos_token_id=31)
    bridge.train()
    with pytest.raises(ValueError,match='eval'):
        bridge.generate(generation_inputs,obs,enabled=True,generation_config=decoding)
    bridge.eval()
    a=bridge.generate(generation_inputs,obs,enabled=True,generation_config=decoding)
    b=bridge.generate(generation_inputs,obs,enabled=True,generation_config=decoding)
    assert torch.equal(a,b)
    with pytest.raises(ValueError,match='device/dtype'):
        bridge.prepare(replace(inputs,visual_embeddings=inputs.visual_embeddings.float()),obs)
