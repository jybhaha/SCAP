import json

import torch
from PIL import Image

from attack.attacker import MyPatchAttack as PublicAttack
from utils.patch_selector import UnresolvedPlacementError
from attack.attacker import MyPatchAttack_Eva as PublicEvaluationAttack
from models.my_patch_attack import MyPatchAttack, MyPatchAttack_Eva


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, kernel_size=1)
        self.pool = torch.nn.AdaptiveAvgPool2d(1)

    def forward(self, images):
        features = self.pool(self.conv(images)).flatten(1)
        return torch.cat((features, -features), dim=1)


class ToySegmentor:
    def segment_batch_panoptic(self, images):
        return [{"segmentation": torch.zeros(images.shape[-2:], dtype=torch.long),
                 "segments_info": [{"id": 0, "label_id": 0, "isthing": False}]}]

    def get_largest_stuff_region_batch(self, outputs):
        return ["sky"], [torch.ones_like(output["segmentation"], dtype=torch.uint8) for output in outputs]


class ToyCAM:
    def generate_cam_batch(self, images, target_classes=None):
        return torch.zeros(images.shape[0], *images.shape[-2:], device=images.device)

    def cam_to_mask_batch(self, cams, percentile=30):
        return [torch.zeros_like(cam, dtype=torch.uint8) for cam in cams]


class ToyPatchGenerator:
    def generate_patch_by_region_batch(self, region_names, color_names_list, patch_size,
                                       per_region_k, target_classes=None, **kwargs):
        patches = [[Image.new("RGB", patch_size, (120, 130, 140)) for _ in range(per_region_k)]
                   for _ in region_names]
        prompts = [["sky " + " ".join(colours)] * per_region_k for colours in color_names_list]
        return patches, prompts


def config():
    return {
        "patch_size": (2, 2), "patch_k": 1, "steps": 2, "eps": 0.1,
        "alpha": 0.05, "lambda_smooth": 0.1, "lambda_color": 0.05,
        "targeted_attack": False, "n_colors": 2, "selection_relaxation_steps": 1,
        "cam_model_name": "resnet50", "seed": 42,
    }


def components(model):
    return {
        "segmentor": ToySegmentor(), "cam_gen": ToyCAM(), "patch_gen": ToyPatchGenerator(),
    }


def test_public_entry_is_unified_and_evaluation_signature_preserved(tmp_path):
    assert PublicAttack is MyPatchAttack
    assert PublicEvaluationAttack is MyPatchAttack_Eva
    model = TinyModel().eval()
    attack = MyPatchAttack_Eva(model, device="cpu", config=config(), components=components(model))
    assert attack.model.raw_model is model
    assert attack.optimizer.model is attack.model
    images = torch.rand(1, 3, 8, 8)
    labels = torch.tensor([0])
    diagnostics_path = tmp_path / "diagnostics.json"
    adv, steps, info = attack.run(images, labels, return_info=True,
                                  diagnostics_path=diagnostics_path)
    assert adv.shape == images.shape and len(steps) == 3
    assert info["status"] == "success" and info["coords_source"] == "selector"
    assert set(info["profiling"]["stages"]) == {
        "color", "segmentation", "cam", "selection", "diffusion", "blending", "optimization"
    }
    assert info["palette_lab"] and info["prompts"]
    assert json.loads(diagnostics_path.read_text(encoding="utf-8"))["coords"] == json.loads(json.dumps(info["coords"]))


def test_external_coords_are_strictly_reused_and_ablation_avoids_heavy_components():
    model = TinyModel().eval()
    images = torch.rand(1, 3, 8, 8)
    labels = torch.tensor([0])
    external = [[(2, 2, 4, 4)]]
    attack = MyPatchAttack(model, device="cpu", config=config(), components=components(model))
    adv, info = attack.run(images, labels, coords=external, return_info=True)
    assert adv.shape == images.shape and info["coords"] == external
    assert info["selection"] is None

    ablated_config = config() | {"scene_aware_placement": False, "diffusion_init": False,
                                 "use_momentum": False, "lambda_color": 0.0}
    ablated = MyPatchAttack(model, device="cpu", config=ablated_config)
    _adv, ablated_info = ablated.run(images, labels, return_info=True)
    assert ablated.patch_gen is None and ablated.segmentor is None and ablated.cam_gen is None
    assert ablated_info["optimizer"]["algorithm"] == "I-FGSM"
    assert set(ablated_info["profiling"]["stages"]) == {
        "color", "selection", "diffusion", "blending", "optimization"
    }


def test_targeted_pipeline_requires_target():
    model = TinyModel().eval()
    targeted_config = config() | {"targeted_attack": True, "target_class": None}
    attack = MyPatchAttack(model, device="cpu", config=targeted_config, components=components(model))
    try:
        attack.run(torch.rand(1, 3, 8, 8), torch.tensor([0]))
    except ValueError as exc:
        assert "target is required" in str(exc)
    else:
        raise AssertionError("targeted attack silently accepted a missing target")


def test_configured_target_class_is_used_for_minimal_paper_invocation():
    model = TinyModel().eval()
    targeted_config = config() | {"targeted_attack": True, "target_class": 1}
    attack = MyPatchAttack(model, device="cpu", config=targeted_config, components=components(model))
    _adv, info = attack.run(torch.rand(1, 3, 8, 8), torch.tensor([0]), return_info=True)
    assert info["optimizer"]["targeted"] is True
    assert info["config"]["target_class"] == 1


def test_no_stuff_region_becomes_explicit_unresolved():
    class EmptySegmentor(ToySegmentor):
        def get_largest_stuff_region_batch(self, outputs):
            raise ValueError("no non-empty stuff region")

    model = TinyModel().eval()
    empty_components = components(model) | {"segmentor": EmptySegmentor()}
    attack = MyPatchAttack(model, device="cpu", config=config(), components=empty_components)
    try:
        attack.run(torch.rand(1, 3, 8, 8), torch.tensor([0]))
    except UnresolvedPlacementError:
        assert attack.last_diagnostics["status"] == "unresolved"
        assert attack.last_diagnostics["selection"][0]["unresolved_reason"] == "no_non_empty_stuff"
    else:
        raise AssertionError("missing stuff region was not reported as unresolved")
