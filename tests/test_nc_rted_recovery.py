import copy
import hashlib
import json
import random
import shutil

import numpy as np
import pytest
import torch
from torch import nn

from nc_rted.recovery import CheckpointStore, RecoveryError, validate_checkpoint_payload
from nc_rted.training import IncrementalTrainer, Recipe, sample_order, seed_run


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.evidence = nn.Linear(3,3)
        self.slow = nn.Module()
        self.slow.block = nn.Module()
        self.slow.block.lora_A = nn.Linear(3,1,bias=False)
        self.frozen = nn.Parameter(torch.randn(3),requires_grad=False)
        self.dropout = nn.Dropout(.2)

    def forward(self,x):
        return self.slow.block.lora_A(self.dropout(self.evidence(x)))


def make(seed=17):
    seed_run(seed)
    model = ToyModel()
    recipe = Recipe(updates=6, accumulation=2, warmup=1, save_interval=1)
    return IncrementalTrainer(model,[str(i) for i in range(12)],seed,recipe)


def objective(trainer, seen):
    def loss(sid):
        seen.append(sid)
        noise = random.random() + float(np.random.rand()) + float(torch.rand(()))
        x = torch.tensor([float(sid)/12.,noise,.25])
        return (trainer.model(x) - .4).square().mean()
    return loss


def identity(seed=17):
    value = {k:'a'*64 for k in ['code_sha256','config_sha256','data_sha256','teacher_sha256',
                               'inherited_weights_sha256','runtime_sha256']}
    value.update(run_id='F-17',group='F',seed=str(seed))
    return value


def test_interruption_replays_identical_order_rng_optimizer_and_final_tensors(tmp_path):
    uninterrupted = make(); original_frozen=uninterrupted.model.frozen.detach().clone(); seen_full=[]
    uninterrupted.run(objective(uninterrupted,seen_full))
    expected = copy.deepcopy(uninterrupted.model.state_dict())
    baseline_rng = (random.random(),float(np.random.rand()),torch.rand(3))
    first=make(); seen=[]
    store=CheckpointStore(tmp_path,identity(),min_free_bytes=0)
    first.run(objective(first,seen),save=store.save,stop_after=3)
    # Simulate publication-before-pointer crash: directory remains authoritative.
    (tmp_path/'latest.json').unlink()
    restarted=make();store.restore(restarted)
    assert restarted.completed_updates==3 and restarted.cursor==6
    restarted.run(objective(restarted,seen),save=store.save)
    assert seen==seen_full and len(set(seen))==12
    for k,v in expected.items():assert torch.equal(restarted.model.state_dict()[k],v),k
    assert torch.equal(original_frozen,restarted.model.frozen)
    assert baseline_rng[0]==random.random() and baseline_rng[1]==float(np.random.rand())
    assert torch.equal(baseline_rng[2],torch.rand(3))
    assert len(list(tmp_path.glob('update_*')))==2
    assert (tmp_path/'final/manifest.json').is_file()


def test_corruption_and_identity_change_fail_before_model_mutation(tmp_path):
    trainer=make();store=CheckpointStore(tmp_path,identity(),min_free_bytes=0)
    trainer.run(objective(trainer,[]),save=store.save,stop_after=1)
    target=store.latest()
    modified=identity();modified['data_sha256']='b'*64
    other=CheckpointStore(tmp_path,modified,min_free_bytes=0)
    with pytest.raises(RecoveryError,match='identity'):other.latest()
    with (target/'state.pt').open('ab') as stream:stream.write(b'corrupt')
    fresh=make();before=copy.deepcopy(fresh.model.state_dict())
    with pytest.raises(RecoveryError,match='corruption'):store.restore(fresh,target)
    for k,v in before.items():assert torch.equal(fresh.model.state_dict()[k],v)


def test_pending_artifacts_are_never_resumed_and_disk_guard_prevents_write(tmp_path):
    (tmp_path/'.pending_crashed').mkdir()
    (tmp_path/'.pending_crashed/state.pt').write_bytes(b'partial')
    store=CheckpointStore(tmp_path,identity(),min_free_bytes=10**30)
    assert store.latest() is None
    trainer=make();trainer.run(objective(trainer,[]),stop_after=1)
    with pytest.raises(RecoveryError,match='reserve'):store.save(trainer)
    assert not list(tmp_path.glob('update_*'))


def test_nonfinite_loss_never_advances_optimizer(tmp_path):
    trainer=make();before=copy.deepcopy(trainer.model.state_dict())
    with pytest.raises(FloatingPointError):trainer.run(lambda _:trainer.model(torch.ones(3)).sum()*float('nan'))
    assert trainer.completed_updates==0 and not trainer.optimizer.state
    for k,v in before.items():assert torch.equal(trainer.model.state_dict()[k],v)


def test_group_independent_seed_order_and_init():
    a=make();b=make();c=make(42)
    assert a.order==b.order and a.order!=c.order
    for k,v in a.model.state_dict().items():assert torch.equal(v,b.model.state_dict()[k])
    with pytest.raises(ValueError,match='unique'):sample_order(['one','one'],17)


def test_bf16_small_steps_accumulate_in_fp32_and_survive_resume(tmp_path):
    def make_bf16():
        seed_run(17)
        model=ToyModel().to(torch.bfloat16)
        with torch.no_grad():
            for p in model.parameters():p.fill_(1.)
        return IncrementalTrainer(model,[str(i) for i in range(12)],17,
            Recipe(updates=12,accumulation=1,warmup=0,save_interval=2,new_lr=.001,lora_lr=.001))
    def loss(trainer):
        return lambda _:sum(p.float().sum() for p in trainer.model.parameters() if p.requires_grad)
    full=make_bf16();full.run(loss(full))
    assert any(bool((p!=1).any()) for p in full.model.parameters() if p.requires_grad)
    first=make_bf16();store=CheckpointStore(tmp_path,identity(),min_free_bytes=0)
    first.run(loss(first),save=store.save,stop_after=4)
    restarted=make_bf16();store.restore(restarted);restarted.run(loss(restarted))
    for name,p in full.model.named_parameters():
        assert torch.equal(p,dict(restarted.model.named_parameters())[name])
    for name,p in full.master_parameters.items():
        assert p.dtype==torch.float32
        assert torch.equal(p,restarted.master_parameters[name])
    assert all(value.dtype==torch.float32 for state in full.optimizer.state.values()
               for value in state.values() if isinstance(value,torch.Tensor))


def test_default_warmup_has_one_thousand_positive_update_rates():
    seed_run(17)
    trainer=IncrementalTrainer(ToyModel(),[str(i) for i in range(8000)],17)
    rates=[]
    for _ in range(1000):
        rates.append(tuple(g['lr'] for g in trainer.optimizer.param_groups))
        trainer.optimizer.step()
        trainer.scheduler.step()
    assert rates[0]==pytest.approx((1e-4/50,5e-6/50))
    assert rates[49]==pytest.approx((1e-4,5e-6))
    assert all(a>0 and b>0 for a,b in rates)
    assert all(g['lr']==0 for g in trainer.optimizer.param_groups)


def test_optimizer_name_order_and_moment_shape_checked_before_mutation(tmp_path):
    trainer=make();store=CheckpointStore(tmp_path,identity(),min_free_bytes=0)
    trainer.run(objective(trainer,[]),save=store.save,stop_after=2)
    target=store.latest()
    # Reversing registration order leaves tensor names/values intact, but cannot
    # be allowed to silently swap Adam moments between names.
    fresh=make()
    fresh.model.evidence._parameters=dict(reversed(list(fresh.model.evidence._parameters.items())))
    fresh=IncrementalTrainer(fresh.model,[str(i) for i in range(12)],17,fresh.recipe)
    before=copy.deepcopy(fresh.model.state_dict());rng=torch.get_rng_state().clone()
    with pytest.raises(RecoveryError,match='parameter_names'):store.restore(fresh,target)
    for k,v in before.items():assert torch.equal(fresh.model.state_dict()[k],v)
    assert torch.equal(torch.get_rng_state(),rng)
    state=torch.load(target/'state.pt',weights_only=True)
    entry=next(iter(state['optimizer']['state'].values()))
    entry['exp_avg']=torch.zeros(99)
    torch.save(state,target/'state.pt')
    manifest=json.loads((target/'manifest.json').read_text())
    manifest.update(payload_bytes=(target/'state.pt').stat().st_size,
                    payload_sha256=hashlib.sha256((target/'state.pt').read_bytes()).hexdigest())
    (target/'manifest.json').write_text(json.dumps(manifest))
    fresh=make();before=copy.deepcopy(fresh.model.state_dict());rng=torch.get_rng_state().clone()
    with pytest.raises(RecoveryError,match='moment'):store.restore(fresh,target)
    for k,v in before.items():assert torch.equal(fresh.model.state_dict()[k],v)
    assert torch.equal(torch.get_rng_state(),rng)


def test_owned_interrupted_storage_is_audited_and_reclaimed(tmp_path):
    trainer=make();store=CheckpointStore(tmp_path,identity(),min_free_bytes=0)
    trainer.run(objective(trainer,[]),save=store.save,stop_after=2)
    valid=store.latest()
    pending=tmp_path/('.pending_'+'a'*32)
    retired=tmp_path/('.retired_update_000001_'+'b'*32)
    shutil.copytree(valid,pending);shutil.copytree(valid,retired)
    (pending/'state.pt').write_bytes(b'incomplete'*1024)
    foreign=tmp_path/('.pending_'+'c'*32);foreign.mkdir();(foreign/'state.pt').write_bytes(b'user artifact')
    reopened=CheckpointStore(tmp_path,identity(),min_free_bytes=0)
    assert not pending.exists() and not retired.exists() and foreign.exists()
    records=[json.loads(l) for l in (tmp_path/'reconciliation.jsonl').read_text().splitlines()]
    assert len(records)==2 and all(r['action']=='reclaim_interrupted_checkpoint' for r in records)
    fresh=make();reopened.restore(fresh);assert fresh.completed_updates==2


def test_fp32_partial_gradient_cannot_be_checkpointed(tmp_path):
    trainer=make();trainer.run(objective(trainer,[]),stop_after=1)
    store=CheckpointStore(tmp_path,identity(),min_free_bytes=0)
    master=next(iter(trainer.master_parameters.values()))
    master.grad=torch.ones_like(master)
    with pytest.raises(RecoveryError,match='partial accumulation'):store.save(trainer)


def test_payload_validator_rejects_fabricated_null_optimizer_rng_and_order(tmp_path):
    trainer=make(); store=CheckpointStore(tmp_path,identity(),min_free_bytes=0)
    trainer.run(objective(trainer,[]),save=store.save,stop_after=1)
    checkpoint=store.latest(); manifest=json.loads((checkpoint/'manifest.json').read_text())
    state=torch.load(checkpoint/'state.pt',map_location='cpu',weights_only=True)
    state['training']['order']=[None]; state['training']['order_sha256']='0'*64
    state['optimizer']['param_groups']=[{'params':[None]}]; state['rng']={'python':0,'torch':torch.tensor(0),'numpy':{},'cuda':[]}
    bad=tmp_path/'bad.pt'; torch.save(state,bad)
    with pytest.raises(RecoveryError): validate_checkpoint_payload(bad,manifest)


@pytest.mark.parametrize('corrupt', ['optimizer', 'rng', 'order'])
def test_payload_validator_rejects_each_fabricated_state_component(tmp_path, corrupt):
    trainer=make(); store=CheckpointStore(tmp_path,identity(),min_free_bytes=0)
    trainer.run(objective(trainer,[]),save=store.save,stop_after=1)
    checkpoint=store.latest(); manifest=json.loads((checkpoint/'manifest.json').read_text())
    state=torch.load(checkpoint/'state.pt',map_location='cpu',weights_only=True)
    if corrupt == 'optimizer':
        next(iter(state['optimizer']['state'].values()))['exp_avg']=None
    elif corrupt == 'rng':
        state['rng']['torch']=torch.tensor(0)
    else:
        state['training']['order']=[None]
    bad=tmp_path/f'bad-{corrupt}.pt'; torch.save(state,bad)
    with pytest.raises(RecoveryError): validate_checkpoint_payload(bad,manifest)


def test_mid_update_and_failed_callbacks_cannot_publish_advanced_rng(tmp_path):
    trainer=make();trainer.run(objective(trainer,[]),stop_after=1)
    store=CheckpointStore(tmp_path,identity(),min_free_bytes=0)
    count=0
    def loss(sid):
        nonlocal count
        count+=1
        with pytest.raises(RecoveryError,match='partial accumulation'):store.save(trainer)
        result=trainer.model(torch.ones(3)).square().sum()
        return result if count==1 else result*float('nan')
    with pytest.raises(FloatingPointError):trainer.run(loss,stop_after=2)
    assert trainer.completed_updates==1 and not trainer.at_update_boundary
    with pytest.raises(RecoveryError,match='partial accumulation'):store.save(trainer)
    with pytest.raises(RuntimeError,match='restore'):trainer.run(objective(trainer,[]),stop_after=2)
    assert not list(tmp_path.glob('update_*'))
