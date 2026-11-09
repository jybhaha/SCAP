"""Shared RGB/CIELab conversions (sRGB, D65, 2-degree observer)."""

from __future__ import annotations

import numpy as np
import torch


_RGB_TO_XYZ = np.array(
    [[0.4124564, 0.3575761, 0.1804375],
     [0.2126729, 0.7151522, 0.0721750],
     [0.0193339, 0.1191920, 0.9503041]], dtype=np.float64,
)
_XYZ_TO_RGB = np.linalg.inv(_RGB_TO_XYZ)
_WHITE = np.array([0.95047, 1.0, 1.08883], dtype=np.float64)
_DELTA = 6.0 / 29.0


def rgb_to_lab_numpy(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB values in [0, 1] to CIELab using the shared reference formula."""
    array = np.asarray(rgb, dtype=np.float64)
    if array.shape[-1] != 3:
        raise ValueError(f"Expected a final RGB dimension of 3, got {array.shape}")
    linear = np.where(array > 0.04045, ((array + 0.055) / 1.055) ** 2.4, array / 12.92)
    xyz = (linear @ _RGB_TO_XYZ.T) / _WHITE
    f = np.where(xyz > _DELTA ** 3, np.cbrt(np.maximum(xyz, _DELTA ** 3)),
                 xyz / (3 * _DELTA ** 2) + 4.0 / 29.0)
    lab = np.stack((116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]),
                    200 * (f[..., 1] - f[..., 2])), axis=-1)
    return lab.astype(np.float32)


def lab_to_rgb_numpy(lab: np.ndarray) -> np.ndarray:
    """Convert CIELab values to clipped RGB values in [0, 1]."""
    array = np.asarray(lab, dtype=np.float64)
    if array.shape[-1] != 3:
        raise ValueError(f"Expected a final Lab dimension of 3, got {array.shape}")
    fy = (array[..., 0] + 16.0) / 116.0
    fx = fy + array[..., 1] / 500.0
    fz = fy - array[..., 2] / 200.0
    f = np.stack((fx, fy, fz), axis=-1)
    xyz = np.where(f > _DELTA, f ** 3, 3 * _DELTA ** 2 * (f - 4.0 / 29.0)) * _WHITE
    linear = xyz @ _XYZ_TO_RGB.T
    rgb = np.where(linear > 0.0031308, 1.055 * np.maximum(linear, 0.0031308) ** (1 / 2.4) - 0.055,
                   12.92 * linear)
    return rgb.clip(0.0, 1.0).astype(np.float32)


def rgb_to_lab(rgb: torch.Tensor) -> torch.Tensor:
    """Differentiably convert an RGB tensor in [0, 1] to CIELab.

    The final dimension must be RGB. All operations stay on the input device and
    preserve its floating-point dtype.
    """
    if not isinstance(rgb, torch.Tensor) or rgb.shape[-1] != 3:
        raise ValueError("rgb must be a torch.Tensor with final dimension 3")
    if not rgb.is_floating_point():
        raise TypeError("rgb must use a floating-point dtype")

    mask = rgb > 0.04045
    linear = torch.where(mask, ((rgb + 0.055) / 1.055).pow(2.4), rgb / 12.92)
    matrix = torch.as_tensor(_RGB_TO_XYZ, dtype=rgb.dtype, device=rgb.device)
    xyz = linear @ matrix.t()
    white = torch.as_tensor(_WHITE, dtype=rgb.dtype, device=rgb.device)
    xyz = xyz / white
    delta = 6.0 / 29.0
    threshold = delta ** 3
    f = torch.where(xyz > threshold, xyz.clamp_min(threshold).pow(1.0 / 3.0), xyz / (3 * delta ** 2) + 4.0 / 29.0)
    l = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return torch.stack((l, a, b), dim=-1)
