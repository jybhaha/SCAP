"""Mask2Former panoptic segmentation restricted to stuff placement regions."""

from __future__ import annotations

import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision.transforms.functional import to_pil_image
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor


DEFAULT_MASK2FORMER_CHECKPOINT = "facebook/mask2former-swin-base-ade-panoptic"
ADE20K_LABELS = [
    "wall", "building", "sky", "floor", "tree", "ceiling", "road", "bed", "windowpane",
    "grass", "cabinet", "sidewalk", "person", "earth", "door", "table", "mountain",
    "plant", "curtain", "chair", "car", "water", "painting", "sofa", "shelf", "house",
    "sea", "mirror", "rug", "field", "armchair", "seat", "fence", "desk", "rock",
    "wardrobe", "lamp", "bathtub", "railing", "cushion", "base", "box", "column",
    "signboard", "chest of drawers", "counter", "sand", "sink", "skyscraper", "fireplace",
    "refrigerator", "grandstand", "path", "stairs", "runway", "case", "pool table",
    "pillow", "screen door", "stairway", "river", "bridge", "bookcase", "blind", "coffee table",
    "toilet", "flower", "book", "hill", "bench", "countertop", "stove", "palm", "kitchen island",
    "computer", "swivel chair", "boat", "bar", "arcade machine", "hovel", "bus", "towel",
    "light", "truck", "tower", "chandelier", "awning", "streetlight", "booth", "television",
    "airplane", "dirt track", "apparel", "pole", "land", "bannister", "escalator", "ottoman",
    "bottle", "buffet", "poster", "stage", "van", "ship", "fountain", "conveyer belt",
    "canopy", "washer", "plaything", "swimming pool", "stool", "barrel", "basket", "waterfall",
    "tent", "bag", "minibike", "cradle", "oven", "ball", "food", "step", "tank", "trade name",
    "microwave", "pot", "animal", "bicycle", "lake", "dishwasher", "screen", "blanket",
    "sculpture", "hood", "sconce", "vase", "traffic light", "tray", "ashcan", "fan", "pier",
    "crt screen", "plate", "monitor", "bulletin board", "shower", "radiator", "glass", "clock",
    "flag",
]


class SemanticSegmentor:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu",
                 checkpoint=DEFAULT_MASK2FORMER_CHECKPOINT, local_files_only=False):
        self.device = torch.device(device)
        self.checkpoint = checkpoint
        self.local_files_only = bool(local_files_only)
        self.processor = Mask2FormerImageProcessor.from_pretrained(
            checkpoint, local_files_only=self.local_files_only
        )
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            checkpoint, local_files_only=self.local_files_only
        ).to(self.device).eval()

    @staticmethod
    def _to_pil(image_input):
        if isinstance(image_input, torch.Tensor):
            if image_input.ndim == 4:
                if image_input.shape[0] != 1:
                    raise ValueError("Single-image segment expects batch size one")
                image_input = image_input[0]
            if image_input.ndim != 3:
                raise ValueError("Input tensor must have shape [C,H,W] or [1,C,H,W]")
            return to_pil_image(image_input.detach().cpu())
        if isinstance(image_input, Image.Image):
            return image_input.convert("RGB")
        raise TypeError("Input must be a PIL image or torch tensor")

    def segment(self, image_input):
        image = self._to_pil(image_input)
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        result = self.processor.post_process_semantic_segmentation(
            outputs, target_sizes=[image.size[::-1]]
        )[0]
        return result.cpu().numpy()

    def segment_batch(self, images):
        return [self.segment(image) for image in images]

    def get_binary_mask(self, segmentation_map, class_name: str):
        if class_name not in ADE20K_LABELS:
            raise ValueError(f"{class_name} not in ADE20K label list")
        return (segmentation_map == ADE20K_LABELS.index(class_name)).astype(np.uint8)

    def get_largest_region_info(self, segmentation_map):
        return self.get_topk_regions_info(segmentation_map, k=1)[0]

    def get_topk_regions_info(self, segmentation_map, k=3):
        labels, counts = np.unique(segmentation_map, return_counts=True)
        results = []
        for index in np.argsort(-counts, kind="stable")[:k]:
            label_id = int(labels[index])
            if not 0 <= label_id < len(ADE20K_LABELS):
                raise ValueError(f"Unknown ADE20K label id {label_id}")
            results.append((ADE20K_LABELS[label_id],
                            (segmentation_map == label_id).astype(np.uint8), int(counts[index])))
        return results

    def segment_batch_panoptic(self, images_tensor):
        if not isinstance(images_tensor, torch.Tensor) or images_tensor.ndim != 4:
            raise ValueError("images_tensor must have shape [B,C,H,W]")
        images = [to_pil_image(image.detach().cpu()) for image in images_tensor]
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        target_sizes = [(image.height, image.width) for image in images]
        return self.processor.post_process_panoptic_segmentation(outputs=outputs, target_sizes=target_sizes)

    def _label_name(self, label_id):
        id2label = getattr(self.model.config, "id2label", {})
        return id2label.get(label_id, id2label.get(str(label_id), ADE20K_LABELS[label_id]))

    def get_largest_stuff_region_batch(self, panoptic_outputs: List[Dict]):
        """Select the largest segment explicitly marked ``isthing=False``."""
        label_names_batch, masks_batch = [], []
        for output in panoptic_outputs:
            seg_map = output["segmentation"]
            best = None
            for segment in output["segments_info"]:
                if segment.get("isthing") is not False:
                    continue
                segment_id = segment["id"]
                mask = (seg_map == segment_id).to(torch.uint8)
                area = int(mask.sum().item())
                label_id = segment.get("label_id", segment.get("category_id"))
                if label_id is None:
                    raise ValueError("Panoptic stuff segment lacks label_id/category_id")
                if best is None or area > best[0]:
                    best = (area, int(label_id), mask)
            if best is None or best[0] == 0:
                raise ValueError("Mask2Former returned no non-empty stuff region")
            label_names_batch.append(self._label_name(best[1]))
            masks_batch.append(best[2])
        return label_names_batch, masks_batch

    def visualize_panoptic_and_largest_stuff(self, images_tensor, panoptic_outputs,
                                             masks_batch, label_names_batch,
                                             save_dir=None, timestamp_str="default_time", show=False):
        for index in range(images_tensor.shape[0]):
            image = to_pil_image(images_tensor[index].detach().cpu()).convert("RGB")
            seg_map = panoptic_outputs[index]["segmentation"].detach().cpu().numpy()
            segments = panoptic_outputs[index]["segments_info"]
            colour_map = np.zeros((*seg_map.shape, 3), dtype=np.uint8)
            for segment_index, segment in enumerate(segments):
                colour = np.asarray(((37 * segment_index) % 255, (97 * segment_index) % 255,
                                     (167 * segment_index) % 255), dtype=np.uint8)
                colour_map[seg_map == segment["id"]] = colour
            panoptic_image = Image.fromarray(colour_map)
            mask = masks_batch[index].detach().cpu().numpy()
            overlay = np.asarray(image).copy()
            overlay[mask > 0] = (0.5 * overlay[mask > 0] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
            canvas = Image.new("RGB", (image.width * 3, image.height))
            canvas.paste(image, (0, 0))
            canvas.paste(panoptic_image, (image.width, 0))
            canvas.paste(Image.fromarray(overlay), (image.width * 2, 0))
            ImageDraw.Draw(canvas).text((image.width * 2 + 10, 10), label_names_batch[index], fill="white")
            if save_dir is not None:
                os.makedirs(save_dir, exist_ok=True)
                canvas.save(os.path.join(save_dir, f"segment_{timestamp_str}_{index}.png"))
            if show:
                plt.imshow(canvas)
                plt.axis("off")
                plt.show()
