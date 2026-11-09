"""Shared ImageNet preprocessing for every classifier-facing SCAP stage."""

from __future__ import annotations

import torch


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ImageNetClassifier(torch.nn.Module):
    """Wrap one user classifier without replacing its weights or modules.

    The public attack API accepts unnormalised RGB tensors in [0, 1]. Grad-CAM,
    optimisation and evaluation all call this wrapper, so preprocessing cannot
    diverge between stages. ``raw_model`` remains available for target-layer
    resolution and model-identity checks.
    """

    def __init__(self, raw_model, input_normalized=False, input_size=None):
        super().__init__()
        if not isinstance(raw_model, torch.nn.Module):
            raise TypeError("raw_model must be a torch.nn.Module")
        self.raw_model = raw_model
        self.input_normalized = bool(input_normalized)
        self.input_size = tuple(input_size) if input_size is not None else None
        self.register_buffer("imagenet_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("imagenet_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, images):
        if self.input_size is not None and images.shape[-2:] != self.input_size:
            images = torch.nn.functional.interpolate(
                images, size=self.input_size, mode="bilinear", align_corners=False
            )
        if not self.input_normalized:
            images = (images - self.imagenet_mean) / self.imagenet_std
        return self.raw_model(images)


def ensure_imagenet_normalized(model, input_size=None):
    """Idempotently apply the reference preprocessing to a classifier."""

    if isinstance(model, ImageNetClassifier):
        if input_size is not None and model.input_size is None:
            model.input_size = tuple(input_size)
        return model
    return ImageNetClassifier(model, input_size=input_size)
