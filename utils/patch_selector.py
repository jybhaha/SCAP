import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
import cv2
import numpy as np
import torch


import os
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
from PIL import ImageDraw, Image
from typing import List
from utils.result_saver import ResultManager

def distance_transform_numpy(mask_tensor: torch.Tensor):
    """
    将二值 tensor mask 转为 numpy 后进行距离变换
    输入: mask_tensor: torch.Tensor (H, W) 0/1
    输出: torch.Tensor (H, W), float32
    """
    mask_np = mask_tensor.cpu().numpy().astype(np.uint8)
    dist_np = cv2.distanceTransform(1 - mask_np, cv2.DIST_L2, 5)
    dist_tensor = torch.from_numpy(dist_np)
    return dist_tensor


class PatchSelector:
    def __init__(self, binary_mask, rgb_image, patch_size=(64, 64), cam_mask=None):
        self.binary_mask = binary_mask.astype(np.uint8)
        self.cam_mask = cam_mask.astype(np.uint8) if cam_mask is not None else np.zeros_like(self.binary_mask)
        self.rgb_image = np.array(rgb_image)
        self.patch_size = patch_size
        self.H, self.W = binary_mask.shape

        self.dist_to_cam = distance_transform_edt(1 - self.cam_mask)
        self.rm = ResultManager.get_instance()

    def _score_patch(self, x, y):
        ph, pw = self.patch_size
        if y + ph > self.H or x + pw > self.W:
            return None

        cam_area = self.cam_mask[y:y+ph, x:x+pw]
        if cam_area.sum() > 0.1 * ph * pw:
            return None

        mask_area = self.binary_mask[y:y+ph, x:x+pw]
        if mask_area.sum() < 0.5 * ph * pw:
            return None

        dist_patch = self.dist_to_cam[y:y+ph, x:x+pw]
        dist_score = np.mean(dist_patch)
        return dist_score

    def _is_overlap(self, box1, box2, iou_thresh=0.2):
        x1, y1, x2, y2 = box1
        x1_p, y1_p, x2_p, y2_p = box2

        inter_x1 = max(x1, x1_p)
        inter_y1 = max(y1, y1_p)
        inter_x2 = min(x2, x2_p)
        inter_y2 = min(y2, y2_p)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area1 = (x2 - x1) * (y2 - y1)
        area2 = (x2_p - x1_p) * (y2_p - y1_p)
        iou = inter_area / float(area1 + area2 - inter_area + 1e-6)

        return iou > iou_thresh

    def find_top_k_candidates(self, k=3, stride_ratio=0.5):
        ph, pw = self.patch_size
        stride_h, stride_w = int(ph * stride_ratio), int(pw * stride_ratio)

        candidates = []
        for y in range(0, self.H - ph + 1, stride_h):
            for x in range(0, self.W - pw + 1, stride_w):
                score = self._score_patch(x, y)
                if score is not None:
                    box = (x, y, x + pw, y + ph)
                    candidates.append((box, score))

        if not candidates:
            raise ValueError("❌ 无法找到满足条件的patch位置")

        # 按照得分排序（越小表示越靠近CAM）
        candidates.sort(key=lambda x: x[1])

        selected = []
        for box, score in candidates:
            if len(selected) >= k:
                break
            if not any(self._is_overlap(box, sel_box) for sel_box, _ in selected):
                selected.append((box, score))

        if not selected:
            print("⚠️ 候选位置全部重叠，保留最优位置。")
            
            return [candidates[0]]

        return selected

    def visualize_top_k(self, top_k):
        debug_img = self.rgb_image.copy()
        for i, ((x1, y1, x2, y2), score) in enumerate(top_k):
            color = (0, 255 - i * 40, i * 40)
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(debug_img, f"{i+1}:{score:.1f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        plt.imshow(debug_img)
        plt.title("Top-K Patch Positions (Non-overlapping)")
        plt.axis("off")
        plt.show()

    def get_best_patch_positions(self, k=3, stride_ratio=0.5, visualize=True):
        top_k = self.find_top_k_candidates(k=k, stride_ratio=stride_ratio)
        if visualize:
            self.visualize_top_k(top_k)
        return [(x1, y1, x2, y2) for (x1, y1, x2, y2), _ in top_k]

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt

class PatchSelectorBatch:
    def __init__(self, binary_masks_batch, rgb_images_batch: torch.Tensor, patch_size=(64, 64), cam_masks_batch=None, device=None,cam_thresh_ratio = 0.3, mask_thresh_ratio=0.2):
        """
        binary_masks_batch: List[Tensor] 或 Tensor，shape=[B,H,W]，值为0/1，float或uint8
        rgb_images_batch: Tensor，shape=[B,3,H,W]
        cam_masks_batch: List[Tensor] 或 Tensor，shape=[B,H,W]，0/1
        """
        self.rm = ResultManager.get_instance()
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        #self.rm.log(f"PatchSelectorBatch initialized on device: {self.device}")
        # 转换 binary_masks_batch 为 batched tensor
        if isinstance(binary_masks_batch, list):
            #self.rm.log(f"Converting binary_masks_batch from list to tensor with {len(binary_masks_batch)} items.")
            binary_masks_batch = torch.stack([m.float().to(self.device) for m in binary_masks_batch], dim=0)
            #self.rm.log(f"Converted binary_masks_batch shape: {binary_masks_batch.shape}")
        else:
            binary_masks_batch = binary_masks_batch.float().to(self.device)
        self.binary_masks_batch = binary_masks_batch
        self.cam_thresh_ratio = cam_thresh_ratio
        self.mask_thresh_ratio = mask_thresh_ratio
       
        # 转换 cam_masks_batch 为 batched tensor（或创建为全 0）
        if cam_masks_batch is None:
            self.cam_masks_batch = torch.zeros_like(self.binary_masks_batch)
        else:
            if isinstance(cam_masks_batch, list):
                cam_masks_batch = torch.stack([m.float().to(self.device) for m in cam_masks_batch], dim=0)
            else:
                cam_masks_batch = cam_masks_batch.float().to(self.device)
        self.cam_masks_batch = cam_masks_batch.to(self.device)
        #self.rm.log(f"CAM masks batch shape: {self.cam_masks_batch.shape}")
        # RGB 图像张量
        self.rgb_images_batch = rgb_images_batch.to(self.device)

        self.patch_size = patch_size
        self.ph, self.pw = patch_size
        self.B, self.H, self.W = self.binary_masks_batch.shape
        #self.rm.log(f"Batch size: {self.B}, Image size: {self.H}x{self.W}, Patch size: {self.ph}x{self.pw}")
        # 计算批量 distance_transform_edt
        self.dist_to_cam_batch = []
        for i in range(self.B):
            cam_mask_np = (1 - self.cam_masks_batch[i].cpu().numpy()).astype(np.uint8)
            dist_map = distance_transform_edt(cam_mask_np)
            self.dist_to_cam_batch.append(torch.from_numpy(dist_map).to(self.device))
        self.dist_to_cam_batch = torch.stack(self.dist_to_cam_batch, dim=0)  # [B,H,W]
        

    def _score_patch(self, b, x, y):
        if y + self.ph > self.H or x + self.pw > self.W:
            return None
        cam_area = self.cam_masks_batch[b, y:y+self.ph, x:x+self.pw]
        mask_area = self.binary_masks_batch[b, y:y+self.ph, x:x+self.pw]
        
        #cam_thresh_ratio = 0.3     # 容忍最多 30% 落在 CAM 上
        #mask_thresh_ratio = 0.2    # 至少 20% 落在目标区域上

        if cam_area.sum() > self.cam_thresh_ratio * self.ph * self.pw:
            return None
        if mask_area.sum() < self.mask_thresh_ratio * self.ph * self.pw:
            return None
        

        
        if mask_area.sum() < 0.5 * self.ph * self.pw:
            return None

        dist_patch = self.dist_to_cam_batch[b, y:y+self.ph, x:x+self.pw]
        dist_score = dist_patch.mean().item()
        return dist_score

    def _is_overlap(self, box1, box2, iou_thresh=0.2):
        x1, y1, x2, y2 = box1
        x1_p, y1_p, x2_p, y2_p = box2

        inter_x1 = max(x1, x1_p)
        inter_y1 = max(y1, y1_p)
        inter_x2 = min(x2, x2_p)
        inter_y2 = min(y2, y2_p)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area1 = (x2 - x1) * (y2 - y1)
        area2 = (x2_p - x1_p) * (y2_p - y1_p)
        iou = inter_area / float(area1 + area2 - inter_area + 1e-6)

        return iou > iou_thresh

  
    
    def find_top_k_candidates_single(self, b, k=3, stride_ratio=0.5):
        stride_h, stride_w = int(self.ph * stride_ratio), int(self.pw * stride_ratio)
        candidates = []
        for y in range(0, self.H - self.ph + 1, stride_h):
            for x in range(0, self.W - self.pw + 1, stride_w):
                score = self._score_patch(b, x, y)
                if score is not None:
                    box = (x, y, x + self.pw, y + self.ph)
                    candidates.append((box, score))

        if not candidates:
            #self.rm.log(f"⚠️ Batch idx {b}: 无法找到满足条件的位置，随机选用 {k} 个位置")

            selected = []
            attempts = 0
            max_attempts = 100 * k  # 防止死循环
            while len(selected) < k and attempts < max_attempts:
                x = np.random.randint(0, self.W - self.pw + 1)
                y = np.random.randint(0, self.H - self.ph + 1)
                box = (x, y, x + self.pw, y + self.ph)

                if not any(self._is_overlap(box, sel_box) for sel_box, _ in selected):
                    selected.append((box, 0.0))

                attempts += 1

            if not selected:
                # 万一随机都失败，退而求其次返回一个默认框
                box = (0, 0, self.pw, self.ph)
                selected = [(box, 0.0)]
                self.rm.log(f"⚠️ Batch idx {b}: 随机位置失败，使用默认 patch {box}")

            return selected

        # 正常筛选
        candidates.sort(key=lambda x: x[1])
        selected = []
        for box, score in candidates:
            if len(selected) >= k:
                break
            if not any(self._is_overlap(box, sel_box) for sel_box, _ in selected):
                selected.append((box, score))

        if not selected:
            selected.append(candidates[0])

        return selected




    def get_best_patch_positions(self, k=3, stride_ratio=0.5):
        batch_results = []
        for b in range(self.B):
            top_k = self.find_top_k_candidates_single(b, k=k, stride_ratio=stride_ratio)
            
            batch_results.append([(x1, y1, x2, y2) for (x1, y1, x2, y2), _ in top_k])
        return batch_results


    def visualize_patch_selection(self, images_tensor: torch.Tensor,
                                selected_boxes_batch: List[List[tuple]],
                                save_dir: str = None,
                                timestamp_str: str = "default_time",
                                show: bool = False):
        """
        Visualize selected patch regions on original images (batch-wise).

        Args:
            images_tensor: Tensor of shape [B, 3, H, W]
            selected_boxes_batch: List of List of selected patch boxes per image.
                                Each box is (x1, y1, x2, y2)
            save_dir: Directory to save the visualizations
            timestamp_str: Used in output filename, e.g., "20250625T084510"
            show: Whether to display the result using matplotlib
        """
        B = images_tensor.shape[0]
        for i in range(B):
            image = images_tensor[i]
            if image.max() <= 1.0:
                image = image * 255.0
            image = image.to(torch.uint8).cpu()
            image_pil = to_pil_image(image).convert("RGB")

            draw = ImageDraw.Draw(image_pil)
            boxes = selected_boxes_batch[i]
            for box_idx, (x1, y1, x2, y2) in enumerate(boxes):
                draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
                draw.text((x1 + 3, y1 + 3), f"P{box_idx+1}", fill=(255, 255, 0))

            if save_dir is not None:
                os.makedirs(save_dir, exist_ok=True)
                filename = f"patch_selection_{timestamp_str}_{i}.png"
                save_path = os.path.join(save_dir, filename)

                fig, ax = plt.subplots(figsize=(6, 6))
                ax.imshow(image_pil)
                ax.axis("off")
                ax.set_title(f"Selected Patches - Sample {i}")
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close(fig)

            if show:
                plt.figure(figsize=(6, 6))
                plt.imshow(image_pil)
                plt.axis("off")
                plt.title(f"Selected Patches - Sample {i}")
                plt.show()
