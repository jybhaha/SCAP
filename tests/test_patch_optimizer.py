import torch
import torch.nn.functional as F
import pytest

from models.patch_optimizer import PatchOptimizer
from models.patch_optimizer_changeloss import PatchOptimizer_ChangeLoss
from utils.color_space import rgb_to_lab, rgb_to_lab_numpy


def tiny_model():
    torch.manual_seed(7)
    return torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 4 * 4, 2)).eval()


def inputs():
    torch.manual_seed(11)
    clean = 0.2 + 0.6 * torch.rand(1, 3, 4, 4)
    patched = clean.clone()
    patched[:, :, 1:3, 1:3] = 0.75
    return clean, patched


def test_joint_loss_backpropagates_and_uses_same_model():
    model = tiny_model()
    optimizer = PatchOptimizer("cpu", model=model, steps=1, lambda_color=0.5)
    _clean, patched = inputs()
    images = patched.clone().requires_grad_(True)
    mask = torch.zeros(1, 1, 4, 4)
    mask[:, :, 1:3, 1:3] = 1
    palette = torch.tensor([[[53.24, 80.09, 67.20]]])
    total, parts = optimizer.compute_losses(images, mask, torch.tensor([0]), palette_lab=palette)
    gradient, = torch.autograd.grad(total, images)
    assert optimizer.model.raw_model is model
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    assert parts["delta_e76"] >= 0 and parts["total_variation"] >= 0


def test_projection_is_patched_centred_and_outside_is_exact():
    clean, patched = inputs()
    optimizer = PatchOptimizer("cpu", model=tiny_model(), steps=5, eps=0.04, alpha=0.02, lambda_color=0)
    adv = optimizer.optimize_batch(clean, patched, [[(1, 1, 3, 3)]], torch.tensor([0]))
    outside = torch.ones(1, 3, 4, 4, dtype=torch.bool)
    outside[:, :, 1:3, 1:3] = False
    assert torch.equal(adv[outside], patched[outside])
    assert (adv - patched).abs().max() <= 0.04 + 1e-7
    assert (adv - clean).abs().max() > 0.04


def test_targeted_ce_decreases_and_untargeted_ce_increases():
    clean, patched = inputs()
    model = tiny_model()
    common = dict(device="cpu", model=model, steps=1, eps=0.2, alpha=0.05,
                  lambda_smooth=0, lambda_color=0)
    targeted = PatchOptimizer(**common).optimize_batch(
        clean, patched, [[(1, 1, 3, 3)]], torch.tensor([1]), targeted=True
    )
    untargeted = PatchOptimizer(**common).optimize_batch(
        clean, patched, [[(1, 1, 3, 3)]], torch.tensor([0]), targeted=False
    )
    with torch.no_grad():
        before_target = F.cross_entropy(model(patched), torch.tensor([1]))
        after_target = F.cross_entropy(model(targeted), torch.tensor([1]))
        before_untargeted = F.cross_entropy(model(patched), torch.tensor([0]))
        after_untargeted = F.cross_entropy(model(untargeted), torch.tensor([0]))
    assert after_target < before_target
    assert after_untargeted > before_untargeted


def test_optimize_interfaces_step_predictions_and_alias():
    clean, patched = inputs()
    model = tiny_model()
    optimizer = PatchOptimizer("cpu", model=model, steps=2, lambda_color=0)
    mask = torch.zeros(1, 1, 4, 4)
    mask[:, :, 1:3, 1:3] = 1
    direct = optimizer.optimize(clean, patched, mask, torch.tensor([0]))
    batch, steps = optimizer.optimize_batch(
        clean, patched, [[(1, 1, 3, 3)]], torch.tensor([0]), return_step_preds=True
    )
    assert torch.allclose(direct, batch)
    assert len(steps) == 2 and steps[0].shape == (1,)
    assert issubclass(PatchOptimizer_ChangeLoss, PatchOptimizer)
    with pytest.raises(ValueError, match="requires the target model"):
        PatchOptimizer("cpu", model=None)


def test_differentiable_rgb_lab_matches_opencv():
    torch.manual_seed(3)
    rgb = torch.rand(2, 5, 3, dtype=torch.float64, requires_grad=True)
    lab = rgb_to_lab(rgb)
    expected = rgb_to_lab_numpy(rgb.detach().numpy())
    assert np_allclose(lab.detach().numpy(), expected, atol=2e-4)
    lab.sum().backward()
    assert torch.isfinite(rgb.grad).all()


def np_allclose(actual, expected, atol):
    return bool(torch.as_tensor(abs(actual - expected) <= atol).all())
