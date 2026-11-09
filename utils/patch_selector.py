"""Deterministic scene-aware patch placement with progressive relaxation."""

from __future__ import annotations

import copy
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt
from torchvision.transforms.functional import to_pil_image, to_tensor


class UnresolvedPlacementError(ValueError):
    """No admissible Stage-2 window remains after bounded relaxation."""


class PatchSelectorBatch:
    def __init__(self, binary_masks_batch, rgb_images_batch: torch.Tensor,
                 patch_size=(64, 64), cam_masks_batch=None, device=None,
                 cam_thresh_ratio=0.3, mask_thresh_ratio=0.2,
                 relaxation_steps=4, fallback_policy="best"):
        self.device = torch.device(device) if device is not None else rgb_images_batch.device
        self.binary_masks_batch = self._as_batch(binary_masks_batch).to(self.device)
        if cam_masks_batch is None:
            self.cam_masks_batch = torch.zeros_like(self.binary_masks_batch)
        else:
            self.cam_masks_batch = self._as_batch(cam_masks_batch).to(self.device)
        self.rgb_images_batch = rgb_images_batch.to(self.device)
        if self.binary_masks_batch.shape != self.cam_masks_batch.shape:
            raise ValueError("Semantic and CAM masks must have identical [B,H,W] shapes")
        self.patch_size = tuple(int(value) for value in patch_size)
        self.ph, self.pw = self.patch_size
        self.B, self.H, self.W = self.binary_masks_batch.shape
        if self.ph < 1 or self.pw < 1 or self.ph > self.H or self.pw > self.W:
            raise ValueError(f"Patch size {self.patch_size} does not fit image {(self.H, self.W)}")
        if not 0 <= cam_thresh_ratio <= 1 or not 0 <= mask_thresh_ratio <= 1:
            raise ValueError("cam_thresh_ratio and mask_thresh_ratio must be in [0,1]")
        self.initial_cam_thresh = float(cam_thresh_ratio)
        self.initial_mask_thresh = float(mask_thresh_ratio)
        self.cam_thresh_ratio = self.initial_cam_thresh
        self.mask_thresh_ratio = self.initial_mask_thresh
        self.relaxation_steps = int(relaxation_steps)
        if self.relaxation_steps < 0:
            raise ValueError("relaxation_steps must be non-negative")
        if fallback_policy not in ("best", "raise"):
            raise ValueError("fallback_policy must be 'best' or 'raise'")
        self.fallback_policy = fallback_policy
        self.diagnostics = [self._new_diagnostics() for _ in range(self.B)]
        self.dist_to_cam_batch = torch.stack([
            torch.from_numpy(distance_transform_edt(1 - self.cam_masks_batch[i].cpu().numpy().astype(np.uint8)))
            for i in range(self.B)
        ]).to(self.device)

    @staticmethod
    def _as_batch(masks):
        if isinstance(masks, list):
            masks = torch.stack([torch.as_tensor(mask) for mask in masks])
        else:
            masks = torch.as_tensor(masks)
        if masks.ndim == 2:
            masks = masks.unsqueeze(0)
        if masks.ndim != 3:
            raise ValueError("Masks must have shape [B,H,W] or be a list of [H,W] tensors")
        return (masks.float() > 0).float()

    def _new_diagnostics(self):
        return {
            "fallback_triggered": False,
            "relaxation_steps": 0,
            "initial_cam_thresh_ratio": self.initial_cam_thresh,
            "initial_mask_thresh_ratio": self.initial_mask_thresh,
            "final_cam_thresh_ratio": self.initial_cam_thresh,
            "final_mask_thresh_ratio": self.initial_mask_thresh,
            "candidate_count": 0,
            "deterministic_fallback": False,
        }

    def _score_patch(self, b, x, y):
        cam_area = self.cam_masks_batch[b, y:y + self.ph, x:x + self.pw].mean().item()
        mask_area = self.binary_masks_batch[b, y:y + self.ph, x:x + self.pw].mean().item()
        if cam_area > self.cam_thresh_ratio or mask_area < self.mask_thresh_ratio:
            return None
        distance = self.dist_to_cam_batch[b, y:y + self.ph, x:x + self.pw].mean().item()
        return distance + cam_area - mask_area

    @staticmethod
    def _is_overlap(box1, box2, iou_thresh=0.2):
        x1, y1, x2, y2 = box1
        a1, b1, a2, b2 = box2
        intersection = max(0, min(x2, a2) - max(x1, a1)) * max(0, min(y2, b2) - max(y1, b1))
        union = (x2 - x1) * (y2 - y1) + (a2 - a1) * (b2 - b1) - intersection
        return intersection / max(union, 1e-12) > iou_thresh

    def _grid(self, stride_ratio):
        if not 0 < stride_ratio <= 1:
            raise ValueError("stride_ratio must be in (0,1]")
        stride_h = max(1, int(self.ph * stride_ratio))
        stride_w = max(1, int(self.pw * stride_ratio))
        ys = list(range(0, self.H - self.ph + 1, stride_h))
        xs = list(range(0, self.W - self.pw + 1, stride_w))
        if ys[-1] != self.H - self.ph:
            ys.append(self.H - self.ph)
        if xs[-1] != self.W - self.pw:
            xs.append(self.W - self.pw)
        return ((x, y) for y in ys for x in xs)

    def _select(self, candidates, k):
        candidates = sorted(candidates, key=lambda item: (item[1], item[0][1], item[0][0]))
        selected = []
        for box, score in candidates:
            if len(selected) >= k:
                break
            if not any(self._is_overlap(box, chosen[0]) for chosen in selected):
                selected.append((box, score))
        return selected

    def _deterministic_best(self, b, k, stride_ratio):
        """Last-resort deterministic ranking; never samples random coordinates."""
        candidates = []
        for x, y in self._grid(stride_ratio):
            box = (x, y, x + self.pw, y + self.ph)
            cam_ratio = self.cam_masks_batch[b, y:y + self.ph, x:x + self.pw].mean().item()
            mask_ratio = self.binary_masks_batch[b, y:y + self.ph, x:x + self.pw].mean().item()
            violation = max(0.0, cam_ratio - self.initial_cam_thresh) + max(0.0, self.initial_mask_thresh - mask_ratio)
            candidates.append((box, violation + cam_ratio - mask_ratio))
        selected = self._select(candidates, k)
        if not selected:
            raise ValueError(f"No geometrically valid patch candidate for batch index {b}")
        return selected

    def find_top_k_candidates_single(self, b, k=3, stride_ratio=0.5):
        if k < 1:
            raise ValueError("k must be positive")
        diagnostics = self._new_diagnostics()
        for relaxation in range(self.relaxation_steps + 1):
            # Keep the final attempt bounded: cam=1/mask=0 would accept every
            # window and make an unresolved placement impossible by construction.
            fraction = relaxation / max(1, self.relaxation_steps + 1)
            self.cam_thresh_ratio = self.initial_cam_thresh + (1.0 - self.initial_cam_thresh) * fraction
            self.mask_thresh_ratio = self.initial_mask_thresh * (1.0 - fraction)
            candidates = []
            for x, y in self._grid(stride_ratio):
                score = self._score_patch(b, x, y)
                if score is not None:
                    candidates.append(((x, y, x + self.pw, y + self.ph), score))
            selected = self._select(candidates, k)
            diagnostics.update({
                "fallback_triggered": relaxation > 0,
                "relaxation_steps": relaxation,
                "final_cam_thresh_ratio": self.cam_thresh_ratio,
                "final_mask_thresh_ratio": self.mask_thresh_ratio,
                "candidate_count": len(candidates),
            })
            if selected:
                self.diagnostics[b] = diagnostics
                return selected
        if self.fallback_policy == "raise":
            self.diagnostics[b] = diagnostics
            raise UnresolvedPlacementError(
                f"No patch candidate after {self.relaxation_steps} relaxation steps for batch {b}"
            )
        selected = self._deterministic_best(b, k, stride_ratio)
        diagnostics.update({"fallback_triggered": True, "deterministic_fallback": True})
        self.diagnostics[b] = diagnostics
        return selected

    def get_best_patch_positions(self, k=3, stride_ratio=0.5):
        return [
            [box for box, _score in self.find_top_k_candidates_single(b, k=k, stride_ratio=stride_ratio)]
            for b in range(self.B)
        ]

    def get_diagnostics(self):
        return copy.deepcopy(self.diagnostics)

    def visualize_patch_selection(self, images_tensor: torch.Tensor,
                                  selected_boxes_batch: List[List[tuple]],
                                  save_dir: str = None, timestamp_str: str = "default_time",
                                  show: bool = False):
        for index in range(images_tensor.shape[0]):
            image = to_pil_image(images_tensor[index].detach().cpu()).convert("RGB")
            draw = ImageDraw.Draw(image)
            for box_index, (x1, y1, x2, y2) in enumerate(selected_boxes_batch[index]):
                draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
                draw.text((x1 + 3, y1 + 3), f"P{box_index + 1}", fill=(255, 255, 0))
            if save_dir is not None:
                os.makedirs(save_dir, exist_ok=True)
                image.save(os.path.join(save_dir, f"patch_selection_{timestamp_str}_{index}.png"))
            if show:
                plt.imshow(image)
                plt.axis("off")
                plt.show()


class PatchSelector:
    """Single-image compatibility wrapper around :class:`PatchSelectorBatch`."""

    def __init__(self, binary_mask, rgb_image, patch_size=(64, 64), cam_mask=None,
                 cam_thresh_ratio=0.1, mask_thresh_ratio=0.5,
                 relaxation_steps=4, fallback_policy="best"):
        if isinstance(rgb_image, Image.Image):
            image_tensor = to_tensor(rgb_image.convert("RGB")).unsqueeze(0)
        else:
            array = np.asarray(rgb_image)
            if array.max() > 1.0:
                array = array / 255.0
            image_tensor = torch.from_numpy(array).float().permute(2, 0, 1).unsqueeze(0)
        self._batch = PatchSelectorBatch(
            [binary_mask], image_tensor, patch_size=patch_size,
            cam_masks_batch=None if cam_mask is None else [cam_mask],
            cam_thresh_ratio=cam_thresh_ratio, mask_thresh_ratio=mask_thresh_ratio,
            relaxation_steps=relaxation_steps, fallback_policy=fallback_policy,
        )
        self.rgb_image = np.asarray(rgb_image)
        self.patch_size = patch_size

    @property
    def diagnostics(self):
        return self._batch.diagnostics[0]

    def find_top_k_candidates(self, k=3, stride_ratio=0.5):
        return self._batch.find_top_k_candidates_single(0, k=k, stride_ratio=stride_ratio)

    def get_best_patch_positions(self, k=3, stride_ratio=0.5, visualize=False):
        top_k = self.find_top_k_candidates(k=k, stride_ratio=stride_ratio)
        if visualize:
            image = Image.fromarray(self.rgb_image.astype(np.uint8)).convert("RGB")
            draw = ImageDraw.Draw(image)
            for (x1, y1, x2, y2), _score in top_k:
                draw.rectangle((x1, y1, x2, y2), outline=(255, 0, 0), width=2)
            plt.imshow(image)
            plt.axis("off")
            plt.show()
        return [box for box, _score in top_k]
