"""CPU routing fixtures: live reader -> production replay/trigger/Slow/fusion."""
from datetime import datetime, timedelta, timezone
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from nc_rted.production_cold import CostMeter, LivePrefixReader, ColdError, run_cold_prefix
from nc_rted.media_observer import BoundMedia
from nc_rted.detection_provider import DetectionProtocol, DetectionMemoryReplay
from nc_rted.prediction_adapters import BlindDetectionRunner, BoundReactVAUModel


def protocol(rt=False):
    return DetectionProtocol("Is this anomalous?", "neutral", "none", False, rt, .5, .6)


def fixtures(tmp_path):
    path = tmp_path / "media.mp4"
    path.write_bytes(b"bound CPU decoder fixture; not real video")
    media = BoundMedia("ucf", "clip", str(path), hashlib.sha256(path.read_bytes()).hexdigest(), 8., 24, 8, 8)
    record = {"dataset": "ucf-crime", "key": "clip", "sample_id": "development-clip-q1",
              "query_index": 1, "observed_seconds": 14/8}
    decoders = []
    class Decoder:
        fps, frame_count, height, width = 8., 24, 8, 8
        def __init__(self, path):
            self.path, self.reads, self.closed = path, [], False
            decoders.append(self)
        def read(self, index):
            self.reads.append(index)
            return index
        def close(self):
            self.closed = True
    return media, record, Decoder, decoders


def test_live_reader_causal_padding_and_error_cleanup(tmp_path):
    media, record, decoder, decoded = fixtures(tmp_path)
    values = iter([.1, .9])
    def mutate_grid(frames):
        frames.extend([99])
        return frames
    reader = LivePrefixReader(media=media, record=record, protocol=protocol(True), decoder_factory=decoder,
                              encode=lambda frames: torch.zeros(len(frames), 729, 1152),
                              score_grid=lambda grid: [next(values)], make_grid=mutate_grid,
                              meter=CostMeter("cpu"), deadline=datetime.now(timezone.utc)+timedelta(minutes=1))
    prefix = reader("ucf", "clip", expected_media_path=media.media_path, expected_media_sha256=media.media_sha256)
    rows = list(prefix.queries)
    assert prefix.frame_count == 15
    assert rows[0].dense_patches is None
    assert rows[1].dense_patches.shape == (4, 729, 1152)
    assert rows[1].frame_indices == (8, 10, 12, 14)
    assert decoded[0].reads == list(range(0, 15, 2)) and decoded[0].closed
    with pytest.raises(ColdError, match="single-use"):
        reader("ucf", "clip", expected_media_path=media.media_path, expected_media_sha256=media.media_sha256)


def test_decoder_closes_when_fresh_fast_fails(tmp_path):
    media, record, decoder, decoded = fixtures(tmp_path)
    def fail(grid):
        raise RuntimeError("scorer failed")
    reader = LivePrefixReader(media=media, record=record, protocol=protocol(), decoder_factory=decoder,
                              encode=lambda frames: None, score_grid=fail, make_grid=list,
                              meter=CostMeter("cpu"), deadline=datetime.now(timezone.utc)+timedelta(minutes=1))
    prefix = reader("ucf", "clip", expected_media_path=media.media_path, expected_media_sha256=media.media_sha256)
    with pytest.raises(RuntimeError, match="scorer failed"):
        list(prefix.queries)
    assert decoded[0].closed and decoded[0].reads == [0, 2, 4, 6]


def test_actual_runner_replay_conditional_slow_fusion_and_reset(tmp_path, monkeypatch):
    media, record, decoder, decoded = fixtures(tmp_path)
    class Slow(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lm_head = torch.nn.Linear(2, 4, bias=False)
            self.projector = torch.nn.Linear(1152, 2, bias=False)
            self.config = SimpleNamespace(hidden_size=2, vocab_size=4)
        def get_model(self):
            return SimpleNamespace(mm_projector=SimpleNamespace(mlp=self.projector))
        def get_output_embeddings(self):
            return self.lm_head
        def forward(self, hidden, use_cache=False, return_dict=True):
            return SimpleNamespace(logits=self.lm_head(hidden))
    slow = Slow().eval()
    class Memory:
        def __init__(self, *args, **kwargs):
            self.seen = []
        def update(self, patches):
            self.seen.append(patches)
    replays = []
    def replay_factory(model, policy):
        value = DetectionMemoryReplay(model, policy, memory_factory=Memory, time_formatter=lambda *args: "")
        replays.append(value)
        return value
    monkeypatch.setattr("nc_rted.prediction_adapters.DetectionMemoryReplay", replay_factory)
    monkeypatch.setattr("nc_rted.detection_provider.detection_memory_from_stream", lambda *args, **kwargs: torch.zeros(1, 2))
    bridge = SimpleNamespace(slow=slow, raw_slow=slow, evidence=torch.nn.Identity(), prediction_evidence_enabled=False,
                             prepare=lambda inputs, observations, enabled: SimpleNamespace(arguments={"hidden": torch.ones(1, 3, 2)}))
    class Smoother:
        def __init__(self): self.resets = 0
        def reset(self): self.resets += 1
        def step(self, score): return score
    smoother = Smoother()
    runner = BlindDetectionRunner(bridge=bridge, reader="original reader", protocol=protocol(), observation_reader=None,
                                   prompt_tokenizer=SimpleNamespace(encode=lambda **kwargs: None),
                                   yes_token_ids=(0,), no_token_ids=(1,), fusion="weighted", fusion_alpha=.3, smoother=smoother)
    activations = []
    model = BoundReactVAUModel("R0", None, False, bridge, runner,
                               SimpleNamespace(media_reader=SimpleNamespace(vision_encoder=lambda frames: torch.zeros(len(frames), 729, 1152))),
                               {}, SimpleNamespace(activate_language=lambda: activations.append(True)))
    values = iter([.1, .9])
    result = run_cold_prefix(model=model, runtime=SimpleNamespace(document={"protocols": {"vad": {"ucf": vars(protocol())}}}),
                             media_row=vars(media), record=record, score_grid=lambda grid: [next(values)], make_grid=list,
                             cache_root=tmp_path / "cold", deadline=datetime.now(timezone.utc)+timedelta(minutes=1),
                             device="cpu", decoder_factory=decoder)
    assert result["slow_calls"] == 1
    assert result["component_completed_calls"]["slow_forward"] == 1
    assert result["component_completed_calls"]["fast_including_phase_transfers"] == 2
    assert "process_encoder_and_tokens" not in result["component_completed_calls"]
    queries = result["payload"]["queries"]
    assert [q["triggered"] for q in queries] == [False, True]
    assert queries[0]["final_score"] == .1
    assert queries[1]["final_score"] == pytest.approx(.3*.9 + .7*queries[1]["slow_score"])
    assert len(result["payload"]["causal_smoothed_scores"]) == 15
    assert len(replays[0].memory.seen) == 2 and smoother.resets == 1
    assert activations == [True] and decoded[0].closed
    assert runner.reader == "original reader" and runner.observation_reader is None
    assert "forward" not in vars(slow)  # nested count/timer wrappers restored


def test_sealed_endpoint_and_identity_fail_before_decode(tmp_path):
    media, record, decoder, decoded = fixtures(tmp_path)
    kwargs = dict(media=media, record={**record, "observed_seconds": 2.0}, protocol=protocol(), decoder_factory=decoder,
                  encode=lambda frames: None, score_grid=lambda grid: [.1], make_grid=list,
                  meter=CostMeter("cpu"), deadline=datetime.now(timezone.utc)+timedelta(minutes=1))
    with pytest.raises(ColdError, match="endpoint"):
        LivePrefixReader(**kwargs)
    assert not decoded
