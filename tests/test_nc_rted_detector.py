from pathlib import Path
import importlib.util
import json
import os
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from nc_rted.detector import (DetectorError, FrozenRTDetr, InheritedSigLipAdapter, Detection,
                              _normalized_box, build_causal_window, observe_causal_window)
from nc_rted.observation_cache import FrozenFrameCache
from PIL import Image
import torch

def test_detector_requires_explicit_complete_local_snapshot(tmp_path):
    with pytest.raises(DetectorError): FrozenRTDetr(tmp_path)
    (tmp_path/'config.json').write_text('{}')
    with pytest.raises(DetectorError): FrozenRTDetr(tmp_path)

def test_causal_window_reuses_cached_frozen_patches_and_tracks_pairs(tmp_path):
    class Detector:
        def identity(self): return {'detector':'fixed'}
        def detect(self,image): calls_detector.append(1); return (Detection((.1,.1,.3,.3),0,.9),Detection((.5,.5,.8,.8),1,.8))
    calls=[]
    calls_detector=[]
    def siglip(images): calls.append(len(images)); return torch.ones(len(images),729,1152)
    images=[Image.new('RGB',(8,8)) for _ in range(17)]; timestamps=[index*.5 for index in range(17)]
    cache=FrozenFrameCache(tmp_path,10**9,min_free_bytes=0)
    result=build_causal_window(images,timestamps,8.,Detector(),siglip,cache,'media',{'siglip':'fixed'})
    assert calls==[16] and len(calls_detector)==16 and result.status.value=='ok'
    result=build_causal_window(images,timestamps,8.,Detector(),siglip,cache,'media',{'siglip':'fixed'})
    assert calls==[16] and len(calls_detector)==16 and result.status.value=='ok'


def test_observed_window_exposes_tracking_class_pairs_and_honors_partial_start(tmp_path):
    class Detector:
        def identity(self): return {'detector':'partial'}
        def detect(self,image): return (Detection((.1,.1,.3,.3),0,.9),Detection((.5,.5,.8,.8),1,.8))
    calls=[]
    def siglip(images): calls.append(len(images)); return torch.ones(len(images),729,1152)
    images=[Image.new('RGB',(8,8)) for _ in range(21)]
    timestamps=[index*.5 for index in range(21)]
    observed=observe_causal_window(images,timestamps,9.,Detector(),siglip,FrozenFrameCache(tmp_path,10**9,min_free_bytes=0),
                                   'partial-media',{'siglip':'fixed'},window_start_s=8.)
    assert observed.features.status.value=='ok'
    assert observed.relation_class_pairs == ((0,1),)
    assert observed.relation_ids == ('0:1',)
    assert calls == [2]
    assert (observed.features.relations[0].observed_times_s[
        observed.features.relations[0].feature_valid] > 8.).all()

def test_detector_clips_out_of_bounds_boxes_and_rejects_nonfinite_output():
    assert _normalized_box(torch.tensor([-1., -2., 10., 12.]), 8, 8) == (0., 0., 1., 1.)
    assert _normalized_box(torch.tensor([-2., 1., -1., 4.]), 8, 8) is None
    with pytest.raises(DetectorError, match="nonfinite"):
        _normalized_box(torch.tensor([0., float("nan"), 1., 1.]), 8, 8)

def test_inherited_siglip_adapter_requires_exact_snapshot_and_patch_shape(tmp_path):
    class Tower:
        device=torch.device('cpu'); dtype=torch.bfloat16
        image_processor=type('Processor',(),{'preprocess':lambda self,images,return_tensors:{'pixel_values':torch.zeros(len(images),3,384,384)}})()
        def __call__(self,pixels): assert pixels.dtype==self.dtype; return torch.zeros(len(pixels),729,1152,dtype=pixels.dtype)
    with pytest.raises(DetectorError): InheritedSigLipAdapter(Tower(),tmp_path)
    (tmp_path/'config.json').write_text('{"model_type":"siglip","vision_config":{"image_size":384,"patch_size":14,"hidden_size":1152,"num_hidden_layers":27}}'); (tmp_path/'model.safetensors').write_bytes(b'x')
    with pytest.raises(DetectorError, match="original SigLipVisionTower"):
        InheritedSigLipAdapter(Tower(),tmp_path)


@pytest.mark.skipif(importlib.util.find_spec("safetensors") is None or not os.environ.get("NC_RTED_REACTVAU_ROOT"),
                    reason="requires the isolated ReactVAU environment and source root")
def test_inherited_siglip_adapter_verifies_original_tower_retained_weights(tmp_path):
    from safetensors.torch import save_file
    root = Path(os.environ["NC_RTED_REACTVAU_ROOT"])
    sys.path.insert(0, str(root))
    from llava.model.multimodal_encoder.siglip_encoder import SigLipImageProcessor, SigLipVisionTower

    snapshot = tmp_path / "siglip"; snapshot.mkdir()
    (snapshot / "config.json").write_text(json.dumps({"model_type":"siglip", "vision_config": {
        "image_size":384, "patch_size":14, "hidden_size":1152, "num_hidden_layers":27}}))
    expected = torch.tensor([1.25])
    save_file({"vision_model.embeddings.weight": expected,
               "vision_model.encoder.layers.26.deleted": torch.tensor([0.]),
               "vision_model.head.removed": torch.tensor([0.])}, str(snapshot / "model.safetensors"))

    class Inner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.vision_model = torch.nn.Module()
            self.vision_model.embeddings = torch.nn.Module()
            self.vision_model.embeddings.register_parameter("weight", torch.nn.Parameter(expected.clone()))
            self.vision_model.encoder = torch.nn.Module()
            self.vision_model.encoder.layers = torch.nn.ModuleList([torch.nn.Identity() for _ in range(26)])
            self.vision_model.head = torch.nn.Identity()
        def forward(self, pixels, output_hidden_states=True):
            return type("Result", (), {"hidden_states": (pixels.new_zeros((len(pixels),729,1152)),)})()

    tower = SigLipVisionTower.__new__(SigLipVisionTower)
    torch.nn.Module.__init__(tower)
    tower.is_loaded = True; tower.vision_tower_name = str(snapshot); tower.vision_tower = Inner()
    tower.image_processor = SigLipImageProcessor(); tower.eval(); tower.vision_tower.eval(); tower.vision_tower.requires_grad_(False)
    adapter = InheritedSigLipAdapter(tower, snapshot)
    assert adapter([Image.new("RGB", (2,2))]).shape == (1,729,1152)
    tower.image_processor.image_mean = (0., 0., 0.)
    with pytest.raises(DetectorError, match="preprocessing changed"):
        adapter.identity()
    tower.image_processor.image_mean = (.5,.5,.5)
    with torch.no_grad(): tower.vision_tower.vision_model.embeddings.weight.add_(1)
    with pytest.raises(DetectorError, match="tower changed"):
        adapter([Image.new("RGB", (2,2))])
