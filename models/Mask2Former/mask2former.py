import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation
import os
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime
from typing import List, Dict

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm
from collections import defaultdict
from typing import Dict, List

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm
from collections import defaultdict
from torchvision.transforms.functional import to_pil_image
from datetime import datetime
from pycocotools import mask as coco_mask
import os
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
from typing import List, Dict


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
    "flag"
]

class SemanticSegmentor:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        print(f"Using device: {device}")
        self.device = device

        self.processor = Mask2FormerImageProcessor.from_pretrained(
            "models/Mask2Former/facebook/mask2former-swin-base-ade-semantic",
            local_files_only=True)
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            "models/Mask2Former/facebook/mask2former-swin-base-ade-semantic",
            local_files_only=True).to(device)
        self.model.eval()

    def segment(self, image_input):
        """
        支持 PIL.Image 或 Tensor 格式输入
        单张图像：返回 segmentation map (H, W)
        """
        if isinstance(image_input, torch.Tensor):
            if image_input.ndim == 4:
                image_input = transforms.ToPILImage()(image_input[0].cpu())
            elif image_input.ndim == 3:
                image_input = transforms.ToPILImage()(image_input.cpu())
            else:
                raise ValueError("输入 Tensor 应为 (C,H,W) 或 (B,C,H,W) 形状")

        elif not isinstance(image_input, Image.Image):
            raise TypeError("输入应为 PIL.Image 或 Tensor")

        inputs = self.processor(images=image_input, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        segmentation = self.processor.post_process_semantic_segmentation(
            outputs, target_sizes=[image_input.size[::-1]]
        )[0]
        return segmentation.cpu().numpy()

    def get_binary_mask(self, segmentation_map, class_name: str):
        if class_name not in ADE20K_LABELS:
            raise ValueError(f"{class_name} not in ADE20K label list.")
        class_idx = ADE20K_LABELS.index(class_name)
        return (segmentation_map == class_idx).astype(np.uint8) * 255

    def get_largest_region_info(self, segmentation_map):
        unique_labels, counts = np.unique(segmentation_map, return_counts=True)
        max_idx = np.argmax(counts)
        max_label_id = unique_labels[max_idx]
        max_label_name = ADE20K_LABELS[max_label_id]
        max_mask = (segmentation_map == max_label_id).astype(np.uint8) * 255
        return max_label_name, max_mask, counts[max_idx]

    def get_topk_regions_info(self, segmentation_map, k=3):
        """返回 top-k 面积最大的语义类别、掩码、面积"""
        unique_labels, counts = np.unique(segmentation_map, return_counts=True)
        topk_indices = np.argsort(-counts)[:k]
        results = []
        for idx in topk_indices:
            label_id = unique_labels[idx]
            label_name = ADE20K_LABELS[label_id]
            mask = (segmentation_map == label_id).astype(np.uint8) * 255
            area = counts[idx]
            results.append((label_name, mask, area))
        return results



    
    def segment_batch_panoptic(self, images_tensor):
        """
        批量全景分割（Panoptic Segmentation），返回每张图的 masks 和标签。

        输入:
            images_tensor: torch.Tensor，形状 [B, C, H, W]，值范围为[0,1]或[0,255]
        
        输出:
            panoptic_outputs: List[Dict]，每个元素为 dict，含：
                - 'segmentation': Tensor(H, W)，每像素为 panoptic ID（对应 segment_info 中 id）
                - 'segments_info': List[Dict]，每个 dict 包含：
                    - id: segment ID（与 segmentation 图对应）
                    - category_id: 类别 ID
                    - score: 置信度（可能存在）
                    - isthing: 是否为前景目标
        """
        if not isinstance(images_tensor, torch.Tensor):
            raise TypeError("输入必须是 torch.Tensor")
        if images_tensor.ndim != 4:
            raise ValueError("输入 tensor 应该是 [B, C, H, W] 形状")

        B = images_tensor.shape[0]

        images_tensor = images_tensor.clone()
        if images_tensor.max() <= 1.0:
            images_tensor = images_tensor * 255.0
        images_tensor = images_tensor.to(torch.uint8)

        tensor_list = [images_tensor[i].to(self.device) for i in range(B)]
        inputs = self.processor(images=tensor_list, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = [(img.shape[1], img.shape[2]) for img in tensor_list]

        panoptic_outputs = self.processor.post_process_panoptic_segmentation(
            outputs=outputs,
            target_sizes=target_sizes,
        )

        return panoptic_outputs  
    
    def get_largest_stuff_region_batch(self, panoptic_outputs: List[Dict]):
        """
        输入：
        panoptic_outputs: List[Dict]，每张图像的 panoptic 分割结果（来自 post_process_panoptic_segmentation）
    
        输出：
        label_names_batch: List[str]           ← 每张图最大的 stuff 类别名
        masks_batch:       List[Tensor(H, W)]  ← 每张图最大的 stuff 区域 mask（uint8）
        """
        label_names_batch = []
        masks_batch = []

        for output in panoptic_outputs:
            seg_map = output['segmentation']  # Tensor(H, W)
            segments = output['segments_info']

            max_area = -1
            best_label_id = None
            best_mask = None

            for seg in segments:
                

                seg_id = seg['id']
                label_id = seg['label_id']
                mask = (seg_map == seg_id).to(torch.uint8)
                area = mask.sum().item()

                if area > max_area:
                    max_area = area
                    best_label_id = label_id
                    best_mask = mask

            if best_label_id is not None:
                label_name = ADE20K_LABELS[best_label_id]
            else:
                label_name = "none"
                best_mask = torch.zeros_like(seg_map, dtype=torch.uint8)

            label_names_batch.append(label_name)
            masks_batch.append(best_mask)

        return label_names_batch, masks_batch


    def visualize_panoptic_and_largest_stuff(self,
                                         images_tensor: torch.Tensor,
                                         panoptic_outputs: List[Dict],
                                         masks_batch: List[torch.Tensor],
                                         label_names_batch: List[str],
                                         save_dir: str = None,
                                         timestamp_str: str = "default_time",
                                         show: bool = False):
        """
        Visualize panoptic segmentation + largest 'stuff' region mask + class name.

        Args:
            images_tensor: Input image batch of shape [B, C, H, W]
            panoptic_outputs: Panoptic segmentation results for each image
            masks_batch: List of masks for the largest 'stuff' region per image [H, W]
            label_names_batch: List of class names for the largest 'stuff' region per image
            save_dir: Directory to save visualizations
            timestamp_str: Timestamp string for naming, e.g., "20250625T084510"
            show: Whether to display images using plt
        """
        B = images_tensor.shape[0]

        for i in range(B):
            image = images_tensor[i]
            if image.max() <= 1.0:
                image = image * 255.0
            image = image.to(torch.uint8).cpu()
            image_pil = to_pil_image(image)

            # Panoptic segmentation coloring
            seg_map = panoptic_outputs[i]["segmentation"].cpu().numpy()
            segments_info = panoptic_outputs[i]["segments_info"]

            panoptic_color = np.zeros((seg_map.shape[0], seg_map.shape[1], 3), dtype=np.uint8)
            for seg in segments_info:
                color = np.random.randint(0, 255, size=3)
                panoptic_color[seg_map == seg['id']] = color
            panoptic_img = Image.fromarray(panoptic_color)

            # Highlight largest stuff region
            mask = masks_batch[i].cpu().numpy()
            mask_rgb = np.stack([mask * 255, np.zeros_like(mask), np.zeros_like(mask)], axis=2)
            mask_img = Image.fromarray(mask_rgb.astype(np.uint8))

            # Merge views: original + panoptic + stuff mask overlay
            w, h = image_pil.size
            canvas = Image.new("RGB", (w * 3, h))
            canvas.paste(image_pil, (0, 0))
            canvas.paste(panoptic_img, (w, 0))
            canvas.paste(Image.blend(image_pil, mask_img, alpha=0.5), (w * 2, 0))

            # Draw class name
            draw = ImageDraw.Draw(canvas)
            label_text = f" {label_names_batch[i]}"
            draw.text((w * 2 + 10, 10), label_text, fill=(255, 255, 255))

            # Save and/or display
            if save_dir is not None:
                os.makedirs(save_dir, exist_ok=True)
                filename_base = f"segment_{timestamp_str}_{i}"

                # Save high-res image
                fig, ax = plt.subplots(figsize=(12, 4))
                ax.imshow(canvas)
                ax.axis("off")
                ax.set_title(label_text, fontsize=12)

                fig.savefig(os.path.join(save_dir, f"{filename_base}.png"), dpi=300, bbox_inches='tight')
                # fig.savefig(os.path.join(save_dir, f"{filename_base}.pdf"), bbox_inches='tight')
                # fig.savefig(os.path.join(save_dir, f"{filename_base}.svg"), bbox_inches='tight')

                plt.close(fig)

            if show:
                plt.figure(figsize=(16, 6))
                plt.imshow(canvas)
                plt.axis("off")
                plt.title(label_text)
                plt.show()
