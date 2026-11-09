import importlib

import pytest
import torch

from torchattacks.attacks.nifgsm import NIFGSM
from torchattacks.attacks.sinifgsm import SINIFGSM


def _toy_inputs():
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(12, 2)).eval()
    images = torch.linspace(0, 1, 12).reshape(1, 3, 2, 2)
    labels = torch.tensor([0])
    return model, images, labels


def _assert_valid_adversarial(images, adv_images):
    assert adv_images.shape == images.shape
    assert torch.isfinite(adv_images).all()
    assert torch.all((0 <= adv_images) & (adv_images <= 1))


def test_nifgsm_forward_uses_module_torch_global():
    model, images, labels = _toy_inputs()
    attack = NIFGSM(model, eps=0.1, alpha=0.01, steps=1)
    attack.set_device("cpu")

    adv_images = attack(images, labels)

    _assert_valid_adversarial(images, adv_images)


def test_sinifgsm_forward_uses_module_torch_global():
    model, images, labels = _toy_inputs()
    attack = SINIFGSM(model, eps=0.1, alpha=0.01, steps=1, m=1)
    attack.set_device("cpu")

    step_adv_images = attack(images, labels)

    assert isinstance(step_adv_images, list) and len(step_adv_images) == 1
    _assert_valid_adversarial(images, step_adv_images[0])


@pytest.mark.parametrize("module_name", ["nifgsm_mine", "sinifgsm_mine"])
def test_mine_variants_import_required_torch_global(module_name):
    module = importlib.import_module(f"torchattacks.attacks.{module_name}")

    assert getattr(module, "torch", None) is torch
