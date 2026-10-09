from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nc_rted.batches import caption_block_endpoints
from nc_rted.bridge import ObservationBatch
from nc_rted.caption_provider import CaptionObservationAudit, CaptionProviderError, Stage2CaptionProvider
from nc_rted.features import CELL_FEATURE_DIM
from nc_rted.task_inputs import TrainingCatalog, TrainingTask


def _instruction():
    return {"id": "caption-1", "video": "train/clip.mp4", "task": "caption", "type": "clip",
            "conversations": [{"from": "human", "value": "<video> describe"}, {"from": "gpt", "value": "answer"}]}


def _catalog():
    task = TrainingTask("caption:ucf-crime:caption-1", "caption", "ucf-crime", "family", "train/clip.mp4", None, None, _instruction())
    return TrainingCatalog([task], "catalog-hash")


class _Dataset:
    def __init__(self):
        self.list_data_dict = [{**_instruction(), "_reactvau_relative_video": "train/clip.mp4"}]
        self.calls = 0

    def process_video(self, video_file, data_anno, data_args):
        return [None] * 3, "original sampler message", [0, 8, 20], 4.

    def _get_item(self, position):
        assert position == 0
        self.calls += 1
        self.process_video(self.list_data_dict[position]["video"], data_anno=self.list_data_dict[position], data_args=None)
        pixels = torch.zeros(3, 3, 384, 384)
        return {"id": "caption-1", "image": [(pixels, (384, 384), "video")], "pg_scores": [.1, .2, .3]}


class _Vision:
    def __call__(self, pixels, *, chunk_size):
        assert chunk_size == 32
        return torch.zeros(len(pixels), 729, 1152, dtype=pixels.dtype, device=pixels.device)


class _Projector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = torch.nn.Linear(1152, 1152, bias=False)
        self.requires_grad_(False)

    def forward(self, patches, *, local_num_frames, pg_scores):
        assert local_num_frames == 1 and len(pg_scores) == len(patches)
        return patches.mean(dim=1).unsqueeze(0)


class _Raw:
    class Config:
        mm_local_num_frames = 1
    config = Config()

    def __init__(self):
        self._model = type("Model", (), {"mm_projector": _Projector()})()

    def get_model(self):
        return self._model


class _Model:
    def __init__(self):
        self.raw = _Raw()

    def get_base_model(self):
        return self.raw


class _Observer:
    ready = True

    def __init__(self, blocks=2):
        self.blocks = blocks
        self.calls = []

    def observe_causal_window(self, *, sample_id, annotation, sampling):
        self.calls.append((sample_id, annotation["_reactvau_relative_video"]))
        assert sampling.frame_indices == (0, 8, 20)
        shape = (1, self.blocks, 0, 4)
        observations = ObservationBatch(torch.empty((*shape, CELL_FEATURE_DIM)), torch.zeros(shape, dtype=torch.bool), torch.empty(shape, dtype=torch.float32))
        return CaptionObservationAudit(observations, (0., 2., 5.), 9., "original sampler message", {"detector": "accepted"})


def test_provider_uses_original_item_and_preserves_original_caption_sampling_not_observer_grid():
    dataset, observer = _Dataset(), _Observer()
    provider = Stage2CaptionProvider(catalog=_catalog(), dataset=dataset, model=_Model(), vision_tower=_Vision(), observer=observer)
    material = provider("caption:ucf-crime:caption-1")
    assert dataset.calls == 1
    assert observer.calls == [("caption:ucf-crime:caption-1", "train/clip.mp4")]
    assert material.context.sampled_frame_times == (0., 2., 5.)
    assert material.context.observed_seconds == 9.
    assert material.context.time_message == "original sampler message"
    assert material.context.visual_embeddings.shape == (1, 3, 1152)
    assert material.observations.features.shape[1] == len(caption_block_endpoints(9.))
    assert not torch.is_inference(material.context.visual_embeddings)
    assert not torch.is_inference(material.observations.features)


def test_provider_refuses_unbound_detector_before_original_media_is_read():
    dataset, observer = _Dataset(), _Observer()
    observer.ready = False
    provider = Stage2CaptionProvider(catalog=_catalog(), dataset=dataset, model=_Model(), vision_tower=_Vision(), observer=observer)
    with pytest.raises(CaptionProviderError, match="RT-DETR"):
        provider("caption:ucf-crime:caption-1")
    assert dataset.calls == 0


def test_provider_rejects_missing_tail_relation_block():
    provider = Stage2CaptionProvider(catalog=_catalog(), dataset=_Dataset(), model=_Model(), vision_tower=_Vision(), observer=_Observer(blocks=1))
    with pytest.raises(CaptionProviderError, match="every 8-second block"):
        provider("caption:ucf-crime:caption-1")


def test_real_integer_instruction_ids_and_bf16_caption_boundary():
    instruction = _instruction(); instruction["id"] = 38
    task = TrainingTask("caption:ucf-crime:38", "caption", "ucf-crime", "family", instruction["video"], None, None, instruction)
    catalog = TrainingCatalog([task], "catalog-hash")
    class IntegerDataset(_Dataset):
        def _get_item(self, position):
            sample = super()._get_item(position); sample["id"] = 38
            return sample
    dataset = IntegerDataset()
    dataset.list_data_dict[0]["id"] = 38
    model = _Model(); model.raw.get_model().mm_projector.bfloat16()
    class Vision(_Vision):
        def __call__(self, pixels, *, chunk_size):
            assert pixels.dtype == torch.bfloat16
            return super().__call__(pixels, chunk_size=chunk_size)
    provider = Stage2CaptionProvider(catalog=catalog, dataset=dataset, model=model, vision_tower=Vision(), observer=_Observer())
    material = provider(task.sample_id)
    assert material.context.visual_embeddings.dtype == torch.bfloat16
    # Frozen input must still be usable for a trainable projection's backward.
    projection = torch.nn.Linear(1152, 2).bfloat16()
    projection(material.context.visual_embeddings).float().sum().backward()
    assert projection.weight.grad is not None


def test_inherited_loader_absolute_video_paths_keep_original_relative_identity():
    dataset = _Dataset()
    dataset.list_data_dict[0]["video"] = "/bound/view/train/clip.mp4"
    provider = Stage2CaptionProvider(catalog=_catalog(), dataset=dataset, model=_Model(), vision_tower=_Vision(), observer=_Observer())
    assert provider("caption:ucf-crime:caption-1").context.observed_seconds == 9.
    dataset.list_data_dict[0]["_reactvau_relative_video"] = "other.mp4"
    with pytest.raises(CaptionProviderError, match="binding drifted"):
        provider("caption:ucf-crime:caption-1")


def test_decode_annotation_and_configuration_changes_fail_before_media_read():
    from types import SimpleNamespace
    dataset = _Dataset(); dataset.data_args = SimpleNamespace(sample_type="dynamic_fps1")
    provider = Stage2CaptionProvider(catalog=_catalog(), dataset=dataset, model=_Model(), vision_tower=_Vision(), observer=_Observer())
    dataset.list_data_dict[0]["start"] = 1.
    with pytest.raises(CaptionProviderError, match="decode-controlling"):
        provider("caption:ucf-crime:caption-1")
    del dataset.list_data_dict[0]["start"]
    dataset.data_args.sample_type = "middle"
    with pytest.raises(CaptionProviderError, match="configuration changed"):
        provider("caption:ucf-crime:caption-1")
    assert dataset.calls == 0


def test_observer_cannot_see_answers_and_inference_outputs_support_backward():
    class InferenceObserver(_Observer):
        @torch.inference_mode()
        def observe_causal_window(self, *, sample_id, annotation, sampling):
            assert "conversations" not in annotation and "task" not in annotation and "type" not in annotation
            audit = super().observe_causal_window(sample_id=sample_id, annotation=annotation, sampling=sampling)
            shape = (1,2,1,4)
            observation = ObservationBatch(torch.ones(*shape,CELL_FEATURE_DIM), torch.ones(shape,dtype=torch.bool), torch.ones(shape))
            return CaptionObservationAudit(observation,audit.original_sampled_frame_times,audit.original_observed_seconds,audit.original_time_message,audit.detector_identity)
    provider = Stage2CaptionProvider(catalog=_catalog(), dataset=_Dataset(), model=_Model(), vision_tower=_Vision(), observer=InferenceObserver())
    features = provider("caption:ucf-crime:caption-1").observations.features
    assert not torch.is_inference(features)
    projection = torch.nn.Linear(CELL_FEATURE_DIM,1)
    projection(features).sum().backward()
    assert torch.isfinite(projection.weight.grad).all() and projection.weight.grad.abs().sum()>0
