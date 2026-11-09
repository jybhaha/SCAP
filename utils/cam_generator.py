"""Grad-CAM for the nine supported torchvision classifier architectures."""

from __future__ import annotations

import math
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import to_pil_image

from models.imagenet_model import IMAGENET_MEAN, IMAGENET_STD, ImageNetClassifier


def resolve_target_layer(model, model_name):
    """Resolve a spatial feature layer without constructing another model."""
    model = getattr(model, "raw_model", model)
    name = model_name.lower()
    if name == "resnet50":
        return model.layer4[-1]
    if name == "vgg16":
        return model.features[-1]
    if name == "densenet121":
        return model.features[-1]
    if name == "inception_v3":
        return model.Mixed_7c
    if name in ("efficientnet_b0", "mobilenet_v3_large", "convnext_tiny"):
        return model.features[-1]
    if name == "vit_b_16":
        return model.encoder.ln
    if name == "swin_t":
        return model.norm
    raise ValueError(
        f"CAM is not safely implemented for {model_name!r}; supported models are "
        "resnet50, vgg16, densenet121, inception_v3, efficientnet_b0, "
        "mobilenet_v3_large, convnext_tiny, vit_b_16 and swin_t"
    )


class CAMGenerator:
    def __init__(self, model_name="resnet50", device="cuda", model=None,
                 input_normalized=False, input_size=None):
        self.device = torch.device(device)
        self.model_name = model_name.lower()
        if model is None:
            from models.model_registry import build_model
            model = build_model(self.model_name, weights="DEFAULT", device=self.device)
        self.model = model.to(self.device).eval()
        self.target_layer = resolve_target_layer(self.model, self.model_name)
        # ImageNetClassifier normalises inside forward; do not normalise twice.
        self.input_normalized = bool(input_normalized or isinstance(model, ImageNetClassifier))
        self.input_size = input_size or (299 if self.model_name == "inception_v3" else 224)
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(_module, _inputs, output):
            if not isinstance(output, torch.Tensor):
                raise TypeError(f"CAM target layer must return a tensor, got {type(output).__name__}")
            self.activations = output.detach()

        def backward_hook(_module, _grad_input, grad_output):
            if not grad_output or not isinstance(grad_output[0], torch.Tensor):
                raise TypeError("CAM target layer did not produce a tensor gradient")
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def _normalize(self, tensor):
        mean = torch.tensor(IMAGENET_MEAN, dtype=tensor.dtype, device=tensor.device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, dtype=tensor.dtype, device=tensor.device).view(1, 3, 1, 1)
        return (tensor - mean) / std

    def _preprocess(self, input_data):
        if isinstance(input_data, torch.Tensor):
            tensor = input_data
            if tensor.ndim == 3:
                tensor = tensor.unsqueeze(0)
            if tensor.ndim != 4 or tensor.shape[1] != 3:
                raise ValueError("CAM tensor input must have shape [B,3,H,W]")
            tensor = tensor.to(self.device).float()
        else:
            if isinstance(input_data, np.ndarray):
                input_data = Image.fromarray(input_data.astype(np.uint8))
            if not isinstance(input_data, Image.Image):
                raise TypeError("CAM input must be a PIL image, numpy array or torch tensor")
            from torchvision.transforms.functional import to_tensor
            tensor = to_tensor(input_data.convert("RGB")).unsqueeze(0).to(self.device)
        if tensor.shape[-2:] != (self.input_size, self.input_size):
            tensor = F.interpolate(tensor, size=(self.input_size, self.input_size),
                                   mode="bilinear", align_corners=False)
        # Exactly one normalisation point for tensor, numpy and PIL inputs.
        return tensor if self.input_normalized else self._normalize(tensor)

    @staticmethod
    def _to_spatial(features, model_name):
        if features.ndim == 4:
            if model_name == "swin_t":
                return features.permute(0, 3, 1, 2)
            return features
        if features.ndim == 3 and model_name == "swin_t":
            return features.permute(2, 0, 1)
        if features.ndim == 3 and model_name != "vit_b_16":
            return features  # unbatched NCHW
        if model_name == "vit_b_16" and features.ndim in (2, 3):
            batched = features.ndim == 3
            token_tensor = features if batched else features.unsqueeze(0)
            tokens = token_tensor[:, 1:, :]
            grid = math.isqrt(tokens.shape[1])
            if grid * grid != tokens.shape[1]:
                raise ValueError(
                    f"ViT CAM requires one class token and a square token grid; got {token_tensor.shape[1]} tokens"
                )
            spatial = tokens.transpose(1, 2).reshape(token_tensor.shape[0], token_tensor.shape[2], grid, grid)
            return spatial if batched else spatial[0]
        raise ValueError(f"Cannot safely reshape {model_name} activations with shape {tuple(features.shape)}")

    @staticmethod
    def _target_tensor(target_classes, output, batch_size, device):
        if target_classes is None:
            return output.argmax(dim=1)
        if isinstance(target_classes, int):
            return torch.full((batch_size,), target_classes, dtype=torch.long, device=device)
        targets = torch.as_tensor(target_classes, dtype=torch.long, device=device).reshape(-1)
        if targets.numel() == 1 and batch_size > 1:
            targets = targets.expand(batch_size)
        if targets.numel() != batch_size:
            raise ValueError(f"Expected {batch_size} CAM target classes, got {targets.numel()}")
        return targets

    def generate_cam_batch(self, input_tensor_batch, target_classes=None):
        input_tensor = self._preprocess(input_tensor_batch).detach().requires_grad_(True)
        output = self.model(input_tensor)
        if hasattr(output, "logits"):
            output = output.logits
        if not isinstance(output, torch.Tensor) or output.ndim != 2:
            raise ValueError("Classifier must return a [B,num_classes] logits tensor for CAM")
        targets = self._target_tensor(target_classes, output, input_tensor.shape[0], self.device)
        cams = []
        for index in range(input_tensor.shape[0]):
            self.gradients = None
            torch.autograd.grad(output[index, targets[index]], input_tensor, retain_graph=True)
            if self.activations is None or self.gradients is None:
                raise RuntimeError("CAM hooks did not capture activations and gradients")
            activations = self._to_spatial(self.activations[index], self.model_name)
            gradients = self._to_spatial(self.gradients[index], self.model_name)
            if activations.shape != gradients.shape:
                raise RuntimeError("CAM activation and gradient shapes differ")
            weights = gradients.mean(dim=(1, 2), keepdim=True)
            cam = F.relu((activations * weights).sum(dim=0))
            cam = cam - cam.min()
            cam = cam / cam.max().clamp_min(1e-12)
            cams.append(cam.detach())
        self.model.zero_grad(set_to_none=True)
        return torch.stack(cams)

    def generate_cam(self, input_data, target_class=None):
        return self.generate_cam_batch(input_data, target_class)[0].cpu().numpy()

    def cam_to_mask_batch(self, cams: torch.Tensor, percentile: float = 90):
        if not 0 <= percentile <= 100:
            raise ValueError("percentile must be in [0,100]")
        if cams.ndim == 4 and cams.shape[1] == 1:
            cams = cams[:, 0]
        if cams.ndim != 3:
            raise ValueError("cams must have shape [B,H,W]")
        thresholds = torch.quantile(cams.float().flatten(1), percentile / 100.0, dim=1)
        return list((cams >= thresholds[:, None, None]).to(torch.uint8))

    def overlay_cam_on_image(self, image, cam, alpha=0.5, colormap=cv2.COLORMAP_JET):
        if isinstance(image, torch.Tensor):
            image = image.squeeze().permute(1, 2, 0).detach().cpu().numpy()
            image = np.clip(image * 255, 0, 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), colormap)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        return np.uint8(alpha * heatmap + (1 - alpha) * image)

    def save_cam_composite_4in1_batch(self, images_tensor, cams, masks_batch, save_dir,
                                      timestamp_str="default_time", show=False):
        os.makedirs(save_dir, exist_ok=True)
        for index in range(images_tensor.shape[0]):
            image = to_pil_image(images_tensor[index].detach().cpu())
            cam = cams[index].detach().cpu().numpy()
            cam_image = Image.fromarray((cam * 255).astype(np.uint8)).resize(image.size).convert("RGB")
            mask = masks_batch[index].detach().cpu().numpy().astype(np.uint8)
            mask_image = Image.fromarray(mask * 255).resize(image.size).convert("RGB")
            overlay = Image.blend(image.convert("RGB"), cam_image, 0.5)
            fig, axes = plt.subplots(2, 2, figsize=(8, 8))
            for axis, content, title in zip(axes.flat, (image, cam_image, overlay, mask_image),
                                            ("Image", "CAM", "Overlay", "Mask")):
                axis.imshow(content, cmap="gray")
                axis.set_title(title)
                axis.axis("off")
            fig.tight_layout()
            fig.savefig(os.path.join(save_dir, f"4in1_cam_mask_{timestamp_str}_{index}.png"), dpi=300)
            if show:
                plt.show()
            plt.close(fig)


def cam_to_mask(cam: np.ndarray, percentile: float = 90) -> np.ndarray:
    if not 0 <= percentile <= 100:
        raise ValueError("percentile must be in [0,100]")
    return (cam >= np.percentile(cam, percentile)).astype(np.uint8)
