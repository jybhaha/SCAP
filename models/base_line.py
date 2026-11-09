"""Pixel and patch baselines with explicit controlled-coordinate protocols."""

from __future__ import annotations

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as T

from models.imagenet_model import ensure_imagenet_normalized
from torchattacks.attacks.pgd import PGD_Eva
from torchattacks.attacks.fgsm import FGSM_Eva
from torchattacks.attacks.mifgsm import MIFGSM_Eva
from torchattacks.attacks.difgsm import DIFGSM_Eva
from torchattacks.attacks.nifgsm import NIFGSM_Eva
from torchattacks.attacks.vnifgsm import VNIFGSM_Eva
from utils.result_saver import ResultManager


class PixelAttackBaseline:
    def __init__(self, model, eps=8 / 255, default_steps=10):
        self.model = ensure_imagenet_normalized(model)
        self.eps = eps
        self.steps = default_steps
        self.result_manager = ResultManager.get_instance()

    def get_attack(self, name, **kwargs):
        name = name.upper()
        if name == "FGSM":
            return FGSM_Eva(self.model, eps=self.eps)
        if name == "MIFGSM":
            return MIFGSM_Eva(self.model, eps=self.eps, steps=self.steps, decay=1.0, **kwargs)
        if name == "DIFGSM":
            return DIFGSM_Eva(self.model, eps=self.eps, steps=self.steps, diversity_prob=0.7, **kwargs)
        if name == "NIFGSM":
            return NIFGSM_Eva(self.model, eps=self.eps, steps=self.steps, **kwargs)
        if name == "PGD":
            return PGD_Eva(self.model, eps=self.eps, alpha=2 / 255, steps=self.steps, **kwargs)
        if name == "VNIFGSM":
            return VNIFGSM_Eva(self.model, eps=self.eps, steps=self.steps, **kwargs)
        raise ValueError(f"Unsupported attack type: {name}")

    def run(self, attack_name, x, y, target=None, **kwargs):
        attack = self.get_attack(attack_name, **kwargs)
        if target is not None and hasattr(attack, "set_mode_targeted_by_label"):
            attack.set_mode_targeted_by_label(quiet=True)
        return attack(x, target if target is not None else y)


class PatchAttackBaseline:
    """Patch attacks optimised directly inside externally supplied Stage-2 boxes."""

    def __init__(self, model, input_size=(224, 224), device="cuda", num_classes=1000):
        self.model = ensure_imagenet_normalized(model).to(device).eval()
        self.input_size = input_size
        self.device = torch.device(device)
        self.num_classes = num_classes
        self.art_classifier = None
        self.methods = {
            "advpatch": self.advpatch,
            "advpatch_eva": self.advpatch_eva,
            "lavan": self.lavan,
            "lavan_eva": self.lavan_eva,
            "art_advpatch": self.art_advpatch,
            "art_advpatch_eva": self.art_advpatch_eva,
        }
        self.result_manager = ResultManager.get_instance()

    @staticmethod
    def _resolve_patch_coords(x, patch_coords=None, coords=None):
        if coords is not None and patch_coords is not None:
            raise ValueError("Pass only one of patch_coords and shared coords")
        batch_size, _, height, width = x.shape
        if coords is None:
            if patch_coords is None:
                # Legacy robustness scripts use a fixed centre location. The
                # controlled-comparison runner always supplies Stage-2 coords.
                patch_coords = {
                    "x_min": width // 2 - 32, "x_max": width // 2 + 32,
                    "y_min": height // 2 - 32, "y_max": height // 2 + 32,
                }
            resolved = [patch_coords] * batch_size
        else:
            if len(coords) != batch_size:
                raise ValueError("Shared coords must contain one list per sample")
            resolved = []
            for sample_coords in coords:
                if len(sample_coords) != 1:
                    raise ValueError("Patch baselines require exactly one shared box per sample")
                x1, y1, x2, y2 = (int(value) for value in sample_coords[0])
                resolved.append({"x_min": x1, "y_min": y1, "x_max": x2, "y_max": y2})
        for box in resolved:
            if not (0 <= box["x_min"] < box["x_max"] <= width and
                    0 <= box["y_min"] < box["y_max"] <= height):
                raise ValueError(f"Shared patch box is outside the image: {box}")
        return resolved

    @staticmethod
    def _mask_from_coords(x, resolved):
        mask = torch.zeros_like(x)
        for index, box in enumerate(resolved):
            mask[index, :, box["y_min"]:box["y_max"], box["x_min"]:box["x_max"]] = 1.0
        return mask

    @staticmethod
    def _labels(values, batch_size, device):
        labels = torch.as_tensor(values, dtype=torch.long, device=device).reshape(-1)
        if labels.numel() == 1 and batch_size > 1:
            labels = labels.expand(batch_size)
        if labels.numel() != batch_size:
            raise ValueError(f"Expected {batch_size} labels, got {labels.numel()}")
        return labels

    @staticmethod
    def _loss(logits, labels, targeted):
        ce = F.cross_entropy(logits, labels)
        return ce if targeted else -ce

    def _initial_patch(self, x, mask, patch_path=None, resolved=None):
        patch = torch.rand_like(x) * mask
        if patch_path is not None:
            if len(resolved) != 1:
                raise ValueError("patch_path initialisation is supported for one image at a time")
            box = resolved[0]
            width = box["x_max"] - box["x_min"]
            height = box["y_max"] - box["y_min"]
            image = Image.open(patch_path).convert("RGB").resize((width, height), Image.Resampling.BILINEAR)
            patch = torch.zeros_like(x)
            patch[0, :, box["y_min"]:box["y_max"], box["x_min"]:box["x_max"]] = T.ToTensor()(image).to(x.device)
        return patch.clamp(0, 1)

    @staticmethod
    def _project(patch, initial, mask, eps):
        if eps is not None:
            patch = initial + (patch - initial).clamp(-float(eps), float(eps))
        return (patch * mask).clamp(0, 1)

    def run(self, method, x, y=None, target=None, **kwargs):
        method = method.lower()
        if method not in self.methods:
            raise ValueError(f"Unsupported patch attack: {method}")
        return self.methods[method](x, y, target=target, **kwargs)

    def _lavan(self, x, y, target, patch_coords, coords, epsilon, iterations, eps,
               return_step_preds=False, **_kwargs):
        resolved = self._resolve_patch_coords(x, patch_coords=patch_coords, coords=coords)
        mask = self._mask_from_coords(x, resolved)
        patch = self._initial_patch(x, mask)
        initial = patch.clone()
        targeted = target is not None
        labels = self._labels(target if targeted else y, x.shape[0], x.device)
        step_preds = []
        for _ in range(int(iterations)):
            x_adv = (x * (1 - mask) + patch * mask).detach().requires_grad_(True)
            loss = self._loss(self.model(x_adv), labels, targeted)
            gradient = torch.autograd.grad(loss, x_adv)[0] * mask
            patch = patch - float(epsilon) * gradient
            patch = self._project(patch, initial, mask, eps)
            if return_step_preds:
                with torch.no_grad():
                    step_preds.append(self.model(x * (1 - mask) + patch * mask).argmax(dim=1).cpu())
        x_adv = (x * (1 - mask) + patch * mask).clamp(0, 1).detach()
        return (x_adv, step_preds) if return_step_preds else x_adv

    def lavan(self, x, y=None, target=None, patch_coords=None, coords=None,
              epsilon=0.01, iterations=100, eps=16 / 255, **kwargs):
        return self._lavan(x, y, target, patch_coords, coords, epsilon, iterations, eps, **kwargs)

    def lavan_eva(self, x, y=None, target=None, patch_coords=None, coords=None,
                  epsilon=0.01, iterations=100, eps=16 / 255, **kwargs):
        return self._lavan(x, y, target, patch_coords, coords, epsilon, iterations, eps,
                           return_step_preds=True, **kwargs)

    def _advpatch(self, x, y, target, patch_path, patch_coords, coords, step_size,
                  iterations, eps, return_step_preds=False, **_kwargs):
        resolved = self._resolve_patch_coords(x, patch_coords=patch_coords, coords=coords)
        mask = self._mask_from_coords(x, resolved)
        initial = self._initial_patch(x, mask, patch_path=patch_path, resolved=resolved)
        patch = initial.clone().requires_grad_(True)
        optimiser = torch.optim.Adam([patch], lr=float(step_size))
        targeted = target is not None
        labels = self._labels(target if targeted else y, x.shape[0], x.device)
        step_preds = []
        for _ in range(int(iterations)):
            x_adv = x * (1 - mask) + patch * mask
            loss = self._loss(self.model(x_adv), labels, targeted)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            with torch.no_grad():
                patch.grad *= mask
                optimiser.step()
                projected = self._project(patch, initial, mask, eps)
                patch.copy_(projected)
                if return_step_preds:
                    step_preds.append(self.model(x * (1 - mask) + patch * mask).argmax(dim=1).cpu())
        x_adv = (x * (1 - mask) + patch * mask).clamp(0, 1).detach()
        return (x_adv, step_preds) if return_step_preds else x_adv

    def advpatch(self, x, y=None, target=None, patch_path=None, patch_coords=None, coords=None,
                 step_size=0.01, iterations=100, eps=16 / 255, **kwargs):
        return self._advpatch(x, y, target, patch_path, patch_coords, coords,
                              step_size, iterations, eps, **kwargs)

    def advpatch_eva(self, x, y=None, target=None, patch_path=None, patch_coords=None, coords=None,
                     step_size=0.01, iterations=100, eps=16 / 255, **kwargs):
        return self._advpatch(x, y, target, patch_path, patch_coords, coords,
                              step_size, iterations, eps, return_step_preds=True, **kwargs)

    def _get_art_classifier(self):
        if self.art_classifier is None:
            from art.estimators.classification import PyTorchClassifier
            self.art_classifier = PyTorchClassifier(
                model=self.model,
                loss=torch.nn.CrossEntropyLoss(),
                optimizer=torch.optim.Adam(self.model.parameters(), lr=0.01),
                input_shape=(3, *self.input_shape) if hasattr(self, "input_shape") else (3, *self.input_size),
                nb_classes=self.num_classes,
                clip_values=(0.0, 1.0),
                device_type="gpu" if self.device.type == "cuda" else "cpu",
            )
        return self.art_classifier

    def _apply_art_patch(self, x_np, patch, coords):
        resolved = self._resolve_patch_coords(torch.from_numpy(x_np), coords=coords)
        patch_np = np.asarray(patch)
        if patch_np.ndim != 3:
            raise ValueError("ART patch must have three dimensions")
        if patch_np.shape[0] == 3:
            patch_np = patch_np.transpose(1, 2, 0)
        patch_image = Image.fromarray(np.rint(np.clip(patch_np, 0, 1) * 255).astype(np.uint8), mode="RGB")
        result = x_np.copy()
        for index, box in enumerate(resolved):
            width = box["x_max"] - box["x_min"]
            height = box["y_max"] - box["y_min"]
            resized = patch_image.resize((width, height), Image.Resampling.BILINEAR)
            result[index, :, box["y_min"]:box["y_max"], box["x_min"]:box["x_max"]] = (
                np.asarray(resized, dtype=np.float32).transpose(2, 0, 1) / 255.0
            )
        return result

    def _art_advpatch(self, x, y, target, patch_shape, coords, scale, max_iter, eva):
        from art.attacks.evasion import AdversarialPatch
        x_np = x.detach().cpu().numpy()
        targeted = target is not None
        labels = self._labels(target if targeted else y, x.shape[0], x.device)
        attack = AdversarialPatch(
            classifier=self._get_art_classifier(), rotation_max=22.5,
            scale_min=scale / 2, scale_max=scale, learning_rate=5.0,
            max_iter=max_iter, batch_size=x_np.shape[0], patch_shape=patch_shape,
            targeted=targeted, verbose=False,
        )
        patch, _ = attack.generate(x=x_np, y=labels.cpu().numpy())
        x_adv_np = (self._apply_art_patch(x_np, patch, coords) if coords is not None
                    else attack.apply_patch(x_np, scale=scale, patch_external=patch))
        x_adv = torch.as_tensor(x_adv_np, device=self.device).float()
        if not eva:
            return x_adv
        with torch.no_grad():
            return x_adv, [self.model(x).argmax(dim=1).cpu(), self.model(x_adv).argmax(dim=1).cpu()]

    def art_advpatch(self, x, y, target=None, patch_shape=(3, 64, 64), coords=None,
                     scale=0.3, max_iter=500, **_kwargs):
        return self._art_advpatch(x, y, target, patch_shape, coords, scale, max_iter, eva=False)

    def art_advpatch_eva(self, x, y, target=None, patch_shape=(3, 64, 64), coords=None,
                         scale=0.3, max_iter=500, **_kwargs):
        return self._art_advpatch(x, y, target, patch_shape, coords, scale, max_iter, eva=True)
