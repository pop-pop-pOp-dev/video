from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from nc_rted.observation_cache import FrozenFrameCache, cache_key

def test_cache_key_includes_all_frozen_identities_and_lru(tmp_path):
    first=cache_key('media',1.,{'detector':'a'},{'siglip':'a'})
    assert first != cache_key('media',1.,{'detector':'b'},{'siglip':'a'})
    cache=FrozenFrameCache(tmp_path,100000, min_free_bytes=0)
    cache.put(first,{'patches':torch.ones(2,2,dtype=torch.bfloat16)})
    value=cache.get(first); assert value['patches'].dtype==torch.bfloat16
    cache.evict(); assert cache.get(first) is not None


def test_frame_storage_is_compact_and_corruption_is_invalidated(tmp_path):
    key = cache_key('media', 1., {}, {})
    cache = FrozenFrameCache(tmp_path, 1 << 20, min_free_bytes=0)
    batch = torch.ones(16, 100, dtype=torch.bfloat16)
    cache.put(key, {'patches':batch[0]})
    loaded = cache.get(key)['patches']
    assert loaded.untyped_storage().nbytes() == loaded.numel() * loaded.element_size()
    record = torch.load(cache.path(key), weights_only=True)
    record['payload']['patches'].fill_(5)
    torch.save(record, cache.path(key))
    assert cache.get(key) is None
    assert (tmp_path / 'invalidations.jsonl').is_file()
    cache.put(key, {'patches':batch[0]})
    assert torch.equal(cache.get(key)['patches'], batch[0])


def _populate_once(root, key, count):
    cache = FrozenFrameCache(root, 1 << 20, min_free_bytes=0)
    with cache.population([key]):
        if cache.get(key) is None:
            with count.get_lock(): count.value += 1
            cache.put(key, {'patches':torch.ones(8)})


def test_concurrent_process_population_computes_same_frame_once(tmp_path):
    import multiprocessing
    ctx = multiprocessing.get_context('fork')
    count = ctx.Value('i', 0)
    key = cache_key('media', 1., {}, {})
    processes = [ctx.Process(target=_populate_once, args=(tmp_path, key, count)) for _ in range(3)]
    for process in processes: process.start()
    for process in processes:
        process.join(15)
        assert process.exitcode == 0
    assert count.value == 1
    assert torch.equal(FrozenFrameCache(tmp_path, 1 << 20, min_free_bytes=0).get(key)['patches'], torch.ones(8))


def _write_and_read(root, worker):
    cache = FrozenFrameCache(root, 12000, min_free_bytes=0)
    for index in range(20):
        key = cache_key('media', float(index % 4), {}, {})
        cache.put(key, {'patches':torch.full((512,), worker, dtype=torch.bfloat16)})
        value = cache.get(key)
        if value is not None:
            assert value['patches'].shape == (512,)
            assert torch.isfinite(value['patches']).all()


def test_concurrent_writers_and_eviction_remain_readable(tmp_path):
    import multiprocessing
    ctx = multiprocessing.get_context('fork')
    processes = [ctx.Process(target=_write_and_read, args=(tmp_path, worker)) for worker in range(3)]
    for process in processes: process.start()
    for process in processes:
        process.join(20)
        assert process.exitcode == 0
    assert not list(tmp_path.glob('*.tmp'))
    assert sum(p.stat().st_size for p in tmp_path.glob('*.pt')) <= 12000


def test_production_reserve_is_checked_before_writing(tmp_path, monkeypatch):
    import pytest
    from types import SimpleNamespace
    import nc_rted.observation_cache as module
    monkeypatch.setattr(module.shutil, 'disk_usage', lambda path: SimpleNamespace(free=0))
    with pytest.raises(OSError, match='disk hard limit'):
        FrozenFrameCache(tmp_path, 10000).put(cache_key('media', 1., {}, {}), {'patches':torch.ones(4)})
    assert not list(tmp_path.glob('*.pt'))
