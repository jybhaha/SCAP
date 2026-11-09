"""MI-FGSM patch refinement with TV and differentiable CIELab losses."""

from __future__ import annotations

import os
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from models.imagenet_model import ensure_imagenet_normalized
from utils.color_space import rgb_to_lab


class PatchOptimizer:
    """Optimise only patch pixels, centred on the *patched* initialisation.

    Untargeted attacks minimise negative cross-entropy; targeted attacks minimise
    cross-entropy. Both use gradient descent on the combined objective. The L_inf
    projection is relative to ``patched_image_tensor`` rather than the clean image.
    """

    def __init__(
        self,
        device,
        steps=25,
        eps=8 / 255,
        alpha=5 / 255,
        decay=1.0,
        model=None,
        lambda_adv=1.0,
        lambda_smooth=0.1,
        lambda_color=0.05,
        use_momentum=True,
    ):
        if model is None:
            raise ValueError("PatchOptimizer requires the target model; no fallback model is loaded")
        self.device = torch.device(device)
        self.steps = int(steps)
        self.eps = float(eps)
        self.alpha = float(alpha)
        self.decay = float(decay)
        self.lambda_adv = float(lambda_adv)
        self.lambda_smooth = float(lambda_smooth)
        self.lambda_color = float(lambda_color)
        self.use_momentum = bool(use_momentum)
        if self.steps < 0 or self.eps < 0 or self.alpha < 0:
            raise ValueError("steps, eps and alpha must be non-negative")
        self.model = ensure_imagenet_normalized(model).to(self.device).eval()
        self.last_diagnostics = {}

    @staticmethod
    def _prepare_labels(label, batch_size, device):
        labels = torch.as_tensor(label, dtype=torch.long, device=device).reshape(-1)
        if labels.numel() == 1 and batch_size > 1:
            labels = labels.expand(batch_size)
        if labels.numel() != batch_size:
            raise ValueError(f"Expected {batch_size} labels, got {labels.numel()}")
        return labels

    @staticmethod
    def _coords_to_mask(coords_list, batch_size, height, width, device):
        if len(coords_list) != batch_size:
            raise ValueError(f"Expected coordinates for {batch_size} images, got {len(coords_list)}")
        mask = torch.zeros((batch_size, 1, height, width), dtype=torch.float32, device=device)
        for index, coords in enumerate(coords_list):
            for coord in coords:
                if len(coord) != 4:
                    raise ValueError(f"Invalid patch coordinate: {coord!r}")
                x1, y1, x2, y2 = (int(value) for value in coord)
                if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                    raise ValueError(f"Patch coordinate {(x1, y1, x2, y2)} is outside {(height, width)}")
                mask[index, 0, y1:y2, x1:x2] = 1.0
        if not bool(mask.any()):
            raise ValueError("At least one non-empty patch region is required")
        return mask

    @staticmethod
    def _total_variation_loss(images, mask):
        mask = mask.to(dtype=images.dtype)
        vertical = (images[:, :, 1:, :] - images[:, :, :-1, :]).abs()
        horizontal = (images[:, :, :, 1:] - images[:, :, :, :-1]).abs()
        vertical_mask = mask[:, :, 1:, :] * mask[:, :, :-1, :]
        horizontal_mask = mask[:, :, :, 1:] * mask[:, :, :, :-1]
        vertical_sum = (vertical * vertical_mask).sum()
        horizontal_sum = (horizontal * horizontal_mask).sum()
        count = (vertical_mask.sum() + horizontal_mask.sum()) * images.shape[1]
        return (vertical_sum + horizontal_sum) / count.clamp_min(1.0)

    @staticmethod
    def _color_consistency_loss(images, mask, palette_lab):
        """Mean DeltaE76 to the nearest actual CIELab palette colour."""
        palette = torch.as_tensor(palette_lab, dtype=images.dtype, device=images.device)
        if palette.ndim == 2:
            palette = palette.unsqueeze(0).expand(images.shape[0], -1, -1)
        if palette.ndim != 3 or palette.shape[0] != images.shape[0] or palette.shape[2] != 3:
            raise ValueError(f"palette_lab must have shape [B,K,3] or [K,3], got {tuple(palette.shape)}")
        if palette.shape[1] == 0:
            raise ValueError("palette_lab cannot be empty when lambda_color is non-zero")
        pixels = images.permute(0, 2, 3, 1)
        pixel_lab = rgb_to_lab(pixels)
        distances = torch.cdist(pixel_lab.unsqueeze(3), palette.unsqueeze(1).unsqueeze(2), p=2)
        nearest = distances.amin(dim=-1)
        weights = mask.squeeze(1).to(dtype=images.dtype)
        return (nearest * weights).sum() / weights.sum().clamp_min(1.0)

    def compute_losses(self, images, mask, label, targeted=False, palette_lab=None):
        logits = self.model(images)
        if hasattr(logits, "logits"):
            logits = logits.logits
        ce_loss = F.cross_entropy(logits, label)
        adv_loss = ce_loss if targeted else -ce_loss
        smooth_loss = self._total_variation_loss(images, mask)
        if self.lambda_color and palette_lab is None:
            raise ValueError("palette_lab is required when lambda_color is non-zero")
        color_loss = (
            self._color_consistency_loss(images, mask, palette_lab)
            if self.lambda_color and palette_lab is not None
            else images.sum() * 0.0
        )
        total = self.lambda_adv * adv_loss + self.lambda_smooth * smooth_loss + self.lambda_color * color_loss
        return total, {
            "adversarial": adv_loss.detach(),
            "cross_entropy": ce_loss.detach(),
            "total_variation": smooth_loss.detach(),
            "delta_e76": color_loss.detach(),
            "total": total.detach(),
        }

    def _optimize_with_mask(
        self,
        patched_image_tensor,
        patch_mask,
        label,
        targeted=False,
        return_step_preds=False,
        palette_lab=None,
        target_colors=None,
    ):
        if target_colors is not None:
            if palette_lab is not None:
                raise ValueError("Pass only one of palette_lab and target_colors")
            palette_lab = target_colors
        init = patched_image_tensor.detach().to(self.device)
        if init.ndim != 4 or init.shape[1] != 3:
            raise ValueError("patched_image_tensor must have shape [B,3,H,W]")
        batch_size, _, height, width = init.shape
        mask = patch_mask.detach().to(self.device)
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        if mask.ndim != 4 or mask.shape[0] not in (1, batch_size) or mask.shape[1] != 1 or mask.shape[2:] != (height, width):
            raise ValueError(f"patch_mask must have shape [B,1,H,W], got {tuple(mask.shape)}")
        mask = (mask > 0).to(dtype=init.dtype)
        if mask.shape[0] == 1:
            mask = mask.expand(batch_size, -1, -1, -1)
        labels = self._prepare_labels(label, batch_size, self.device)
        x_adv = init.clone()
        momentum = torch.zeros_like(x_adv)
        step_preds = []
        losses = []

        for _ in range(self.steps):
            x_adv = x_adv.detach().requires_grad_(True)
            total_loss, parts = self.compute_losses(x_adv, mask, labels, targeted, palette_lab)
            grad = torch.autograd.grad(total_loss, x_adv)[0]
            grad = grad * mask
            grad_norm = grad / (grad.abs().mean(dim=(1, 2, 3), keepdim=True) + 1e-12)
            momentum = self.decay * momentum + grad_norm if self.use_momentum else grad_norm
            candidate = x_adv.detach() - self.alpha * momentum.sign() * mask
            delta = (candidate - init).clamp(-self.eps, self.eps)
            projected = (init + delta).clamp(0.0, 1.0)
            x_adv = torch.where(mask.bool(), projected, init)
            losses.append({name: float(value.cpu()) for name, value in parts.items()})
            if return_step_preds:
                with torch.no_grad():
                    logits = self.model(x_adv)
                    if hasattr(logits, "logits"):
                        logits = logits.logits
                    step_preds.append(logits.argmax(dim=1).detach().cpu())

        self.last_diagnostics = {
            "algorithm": "MI-FGSM" if self.use_momentum else "I-FGSM",
            "targeted": bool(targeted),
            "steps": self.steps,
            "eps": self.eps,
            "alpha": self.alpha,
            "projection_center": "patched_initialization",
            "losses": losses,
        }
        if return_step_preds:
            return x_adv.detach(), step_preds
        return x_adv.detach()

    def optimize(self, image_tensor, patched_image_tensor, patch_mask, label, targeted=False,
                 return_step_preds=False, palette_lab=None, target_colors=None):
        """Single-mask compatibility interface; ``image_tensor`` is not a projection centre."""
        del image_tensor
        return self._optimize_with_mask(
            patched_image_tensor, patch_mask, label, targeted, return_step_preds, palette_lab, target_colors
        )

    def optimize_batch(self, image_tensor, patched_image_tensor, coords_list, label,
                       target_colors=None, targeted=False, return_step_preds=False,
                       palette_lab=None, use_momentum=None):
        """Batch coordinate interface used by existing experiments."""
        del image_tensor
        init = patched_image_tensor.detach().to(self.device)
        if init.ndim != 4:
            raise ValueError("patched_image_tensor must be a four-dimensional tensor")
        batch_size, _, height, width = init.shape
        mask = self._coords_to_mask(coords_list, batch_size, height, width, self.device)
        previous = self.use_momentum
        if use_momentum is not None:
            self.use_momentum = bool(use_momentum)
        try:
            return self._optimize_with_mask(
                init, mask, label, targeted, return_step_preds, palette_lab, target_colors
            )
        finally:
            self.use_momentum = previous

    def optimize_batch_pgd(self, image_tensor, patched_image_tensor, coords_list, label,
                           targeted=False, return_step_preds=False):
        """Legacy name; performs non-momentum iterative FGSM with the new projection."""
        return self.optimize_batch(
            image_tensor, patched_image_tensor, coords_list, label,
            targeted=targeted, return_step_preds=return_step_preds, use_momentum=False,
        )

    def visualize_adversarial_comparison(self, x: torch.Tensor, x_adv: torch.Tensor,
                                         save_dir: str = None, timestamp_str: str = "default_time",
                                         show: bool = False):
        for index in range(x.shape[0]):
            fig, axes = plt.subplots(1, 2, figsize=(6, 3))
            axes[0].imshow(TF.to_pil_image(x[index].detach().cpu()))
            axes[0].set_title("Original")
            axes[1].imshow(TF.to_pil_image(x_adv[index].detach().cpu()))
            axes[1].set_title("Adversarial")
            for axis in axes:
                axis.axis("off")
            if save_dir is not None:
                os.makedirs(save_dir, exist_ok=True)
                fig.savefig(os.path.join(save_dir, f"adv_compare_{timestamp_str}_{index}.png"),
                            dpi=300, bbox_inches="tight")
            if show:
                plt.show()
            plt.close(fig)
