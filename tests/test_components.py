from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from utils.cam_generator import CAMGenerator, resolve_target_layer
from utils.color_extractor import ColorExtractor
from utils.patch_selector import PatchSelector, PatchSelectorBatch


def test_cielab_palette_is_deterministic_and_complete():
    array = np.zeros((16, 16, 3), dtype=np.uint8)
    array[:8, :, :] = (255, 0, 0)
    array[8:, :8, :] = (0, 255, 0)
    array[8:, 8:, :] = (0, 0, 255)
    image = Image.fromarray(array)
    first = ColorExtractor(n_colors=3, random_state=42).get_dominant_palette(image)
    second = ColorExtractor(n_colors=3, random_state=42).get_dominant_palette(image)
    assert first == second
    assert set(first) == {"rgb", "lab", "names"}
    assert len(first["rgb"]) == len(first["lab"]) == len(first["names"]) == 3
    assert all(0 <= channel <= 255 for colour in first["rgb"] for channel in colour)
    assert first["lab"][0][0] >= 0


def test_progressive_relaxation_records_thresholds_and_is_deterministic():
    masks = [torch.ones(16, 16)]
    cams = [torch.ones(16, 16)]
    images = torch.rand(1, 3, 16, 16)
    selector = PatchSelectorBatch(
        masks, images, patch_size=(4, 4), cam_masks_batch=cams,
        cam_thresh_ratio=0.0, mask_thresh_ratio=0.9, relaxation_steps=2,
    )
    first = selector.get_best_patch_positions(k=2)
    diagnostics = selector.get_diagnostics()[0]
    assert diagnostics["fallback_triggered"] is True
    assert diagnostics["relaxation_steps"] == 2
    assert diagnostics["final_cam_thresh_ratio"] == pytest.approx(2 / 3)
    assert diagnostics["final_mask_thresh_ratio"] == pytest.approx(0.3)
    second = selector.get_best_patch_positions(k=2)
    assert first == second


def test_no_candidates_raises_or_single_interface_works():
    images = torch.rand(1, 3, 8, 8)
    selector = PatchSelectorBatch(
        [torch.ones(8, 8)], images, patch_size=(4, 4),
        cam_masks_batch=[torch.ones(8, 8)], cam_thresh_ratio=0,
        relaxation_steps=0, fallback_policy="raise",
    )
    with pytest.raises(ValueError, match="No patch candidate"):
        selector.get_best_patch_positions(k=1)
    single = PatchSelector(
        np.ones((8, 8), dtype=np.uint8), Image.new("RGB", (8, 8), "white"),
        patch_size=(4, 4), cam_mask=np.zeros((8, 8), dtype=np.uint8),
        cam_thresh_ratio=0.3, mask_thresh_ratio=0.2,
    )
    coords = single.get_best_patch_positions(k=1, visualize=False)
    assert coords == [(0, 0, 4, 4)]


class TinyConvClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer4 = torch.nn.ModuleList([torch.nn.Conv2d(3, 2, kernel_size=1, bias=False)])

    def forward(self, images):
        features = self.layer4[-1](images)
        pooled = features.mean(dim=(2, 3))
        return torch.stack((pooled[:, 0], pooled[:, 1]), dim=1)


def test_cam_reuses_model_normalises_once_and_reshapes():
    model = TinyConvClassifier().eval()
    generator = CAMGenerator(model_name="resnet50", model=model, device="cpu", input_size=4)
    images = torch.rand(2, 3, 4, 4)
    cams = generator.generate_cam_batch(images, target_classes=[0, 1])
    assert generator.model is model
    assert cams.shape == (2, 4, 4)
    assert torch.isfinite(cams).all()
    vit = CAMGenerator._to_spatial(torch.rand(1, 5, 8), "vit_b_16")
    swin = CAMGenerator._to_spatial(torch.rand(1, 2, 2, 8), "swin_t")
    assert vit.shape == (1, 8, 2, 2)
    assert swin.shape == (1, 8, 2, 2)
    with pytest.raises(ValueError, match="square token grid"):
        CAMGenerator._to_spatial(torch.rand(1, 6, 8), "vit_b_16")


def test_all_nine_target_layer_paths_and_unknown_error():
    layer = torch.nn.Identity()
    model = SimpleNamespace(
        layer4=[layer], features=[layer], Mixed_7c=layer,
        encoder=SimpleNamespace(ln=layer), norm=layer,
    )
    names = ["resnet50", "vgg16", "densenet121", "inception_v3", "efficientnet_b0",
             "mobilenet_v3_large", "convnext_tiny", "vit_b_16", "swin_t"]
    assert all(resolve_target_layer(model, name) is layer for name in names)
    with pytest.raises(ValueError, match="not safely implemented"):
        resolve_target_layer(model, "unknown")
