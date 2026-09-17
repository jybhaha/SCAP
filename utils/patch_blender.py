import numpy as np
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import torch
import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional as TF
import os
from typing import List
from utils.result_saver import ResultManager

class PatchBlender:
    def __init__(self, blend_width=10):
        """
        blend_width: 边缘渐变融合宽度，单位像素
        """
        self.blend_width = blend_width
        self.rm = ResultManager.get_instance()  # 获取结果管理器实例

    def color_match(self, patch, bg_region):
        """
        对patch做颜色校正，使其均值和方差与背景区域相近
        patch, bg_region: PIL.Image RGB模式，大小相同
        返回校正后的patch PIL.Image
        """
        patch_np = np.array(patch).astype(np.float32)
        bg_np = np.array(bg_region).astype(np.float32)

        # 计算均值和标准差
        patch_mean = patch_np.mean(axis=(0, 1))
        patch_std = patch_np.std(axis=(0, 1)) + 1e-6
        bg_mean = bg_np.mean(axis=(0, 1))
        bg_std = bg_np.std(axis=(0, 1)) + 1e-6

        # 标准化后放缩到背景的均值和方差
        matched = (patch_np - patch_mean) / patch_std * bg_std + bg_mean
        matched = np.clip(matched, 0, 255).astype(np.uint8)
        return Image.fromarray(matched)

    def create_alpha_mask(self, size):
        """
        创建一个边缘透明度渐变的alpha掩码，中心不透明，边缘渐变到透明
        size: (width, height)
        返回PIL.Image，单通道灰度图
        """
        w, h = size
        mask = np.ones((h, w), dtype=np.float32)

        # 水平和垂直方向的边缘宽度限制
        bw = self.blend_width

        # 创建四个边缘的渐变
        for i in range(bw):
            alpha = i / bw
            # 左边缘
            mask[:, i] = np.minimum(mask[:, i], alpha)
            # 右边缘
            mask[:, w - 1 - i] = np.minimum(mask[:, w - 1 - i], alpha)
            # 上边缘
            mask[i, :] = np.minimum(mask[i, :], alpha)
            # 下边缘
            mask[h - 1 - i, :] = np.minimum(mask[h - 1 - i, :], alpha)

        mask = (mask * 255).astype(np.uint8)
        return Image.fromarray(mask, mode="L")

    def blend_patch(self, background, patch, position):
        """
        在background图像上，把patch融合贴到position位置
        background: PIL.Image RGB
        patch: PIL.Image RGBA或RGB
        position: (x, y) 左上角位置
        返回融合后的新图像
        """
        bg = background.convert("RGBA")
        pw, ph = patch.size
        x, y = position

        # 裁剪背景对应区域
        bg_region = bg.crop((x, y, x + pw, y + ph)).convert("RGB")

        # 颜色校正patch
        patch_rgb = patch.convert("RGB")
        patch_matched = self.color_match(patch_rgb, bg_region)

        # 创建alpha掩码（边缘透明渐变）
        alpha_mask = self.create_alpha_mask((pw, ph))

        # 如果patch本身有alpha通道，合成alpha
        if patch.mode == "RGBA":
            patch_alpha = patch.split()[-1]
            # 综合两个alpha（patch自身alpha和边缘渐变alpha）
            combined_alpha = Image.eval(patch_alpha, lambda a: a / 255.0)
            combined_alpha = Image.composite(alpha_mask, Image.new("L", (pw, ph), 255), combined_alpha)
            alpha_mask = combined_alpha.convert("L")

        # 合成RGBA图像
        patch_rgba = patch_matched.convert("RGBA")
        patch_rgba.putalpha(alpha_mask)

        # 将patch融合到背景
        bg.paste(patch_rgba, (x, y), patch_rgba)

        return bg.convert("RGB")


    

    def blend_patch_tensor_batch(self, backgrounds, patches, positions, blend_width=10):
        """
        批量融合补丁到背景图中，支持边缘渐变融合。

        参数：
            backgrounds: [B, 3, H, W] float tensor (0~1)
            patches: [B, K, 3, ph, pw] float tensor (0~1)
            positions: List[List[(x1, y1, x2, y2)]]
            blend_width: 渐变宽度（单位为像素）

        返回：
            blended_images: [B, 3, H, W]
        """
        B, _, H, W = backgrounds.shape
        _, K, _, ph, pw = patches.shape
        device = backgrounds.device

        blended_images = backgrounds.clone()

        # 构造边缘渐变 alpha mask（中心为1，边缘为0）
        xx = torch.arange(ph, device=device).view(-1, 1).expand(ph, pw)
        yy = torch.arange(pw, device=device).view(1, -1).expand(ph, pw)

        dist_top = xx
        dist_left = yy
        dist_bottom = ph - 1 - xx
        dist_right = pw - 1 - yy

        dist_edge = torch.minimum(torch.minimum(dist_top, dist_bottom),
                                torch.minimum(dist_left, dist_right)).float()

        alpha = (dist_edge / blend_width).clamp(0, 1)  # [ph, pw]
        alpha = alpha.unsqueeze(0)  # [1, ph, pw]，每通道相同
        #self.rm.log(f"Alpha mask created with blend width {blend_width} pixels.")
        for b in range(B):
            #self.rm.log(f"Blending patches for batch {b} with {K} patches.")
            # 确保 positions[b] 有足够的元素
            available_positions = len(positions[b]) if b < len(positions) else 0
            actual_patches = min(K, available_positions)
            
            for k in range(actual_patches):
                #self.rm.log(f"Blending patch {k} for batch {b} at positions {positions[b][k]}")
                x1, y1, x2, y2 = positions[b][k]
                patch = patches[b, k]  # [3, ph, pw]
                bg_region = blended_images[b, :, y1:y2, x1:x2]  # [3, ph, pw]
                #self.rm.log(f"Background region shape: {bg_region.shape}, Patch shape: {patch.shape}")
                # 匹配颜色统计量：使 patch 风格与背景一致
                patch_mean = patch.mean(dim=(1, 2), keepdim=True)
                patch_std = patch.std(dim=(1, 2), keepdim=True) + 1e-6
                bg_mean = bg_region.mean(dim=(1, 2), keepdim=True)
                bg_std = bg_region.std(dim=(1, 2), keepdim=True) + 1e-6
                patch_matched = (patch - patch_mean) / patch_std * bg_std + bg_mean
                #self.rm.log(f"Patch matched with background mean {bg_mean} and std {bg_std}")
                # 应用 alpha 融合
                blended = alpha * patch_matched + (1 - alpha) * bg_region
                #self.rm.log(f"Patch {k} blended with alpha mask at position ({x1}, {y1}) to ({x2}, {y2})")
                blended_images[b, :, y1:y2, x1:x2] = blended
                #self.rm.log(f"Patch {k} blended into background at position ({x1}, {y1}) to ({x2}, {y2})")
        
        return blended_images.clamp(0, 1)



    def visualize_patch_blending_batch(self,backgrounds: torch.Tensor,
                                    patches: torch.Tensor,
                                    positions: List[List[tuple]],
                                    blended_results: torch.Tensor,
                                    save_dir: str = None,
                                    timestamp_str: str = "default_time",
                                    show: bool = False):
        """
        可视化对比补丁融合前后图像，展示原图、硬贴图、融合图。

        参数：
            backgrounds: [B, 3, H, W] float tensor (0~1)
            patches: [B, K, 3, ph, pw] float tensor (0~1)
            positions: List[List[(x1, y1, x2, y2)]]
            blended_results: [B, 3, H, W] float tensor (0~1)
            save_dir: 可选保存路径
            timestamp_str: 命名时间戳
            show: 是否调用 plt.show()
        """
        B, K, _, ph, pw = patches.shape

        for b in range(B):
            bg = backgrounds[b].clone()
            bg_hard = bg.clone()

            # 将 patch 硬贴（不融合）
            # 确保 positions[b] 有足够的元素
            available_positions = len(positions[b]) if b < len(positions) else 0
            actual_patches = min(K, available_positions)
            
            for k in range(actual_patches):
                x1, y1, x2, y2 = positions[b][k]
                patch = patches[b, k]
                bg_hard[:, y1:y2, x1:x2] = patch

            blended = blended_results[b]

            # 转为 PIL
            bg_img = TF.to_pil_image(bg.cpu())
            hard_img = TF.to_pil_image(bg_hard.cpu())
            blended_img = TF.to_pil_image(blended.cpu())

            # 可视化
            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            axs[0].imshow(bg_img)
            axs[0].set_title("Original")
            axs[0].axis("off")

            axs[1].imshow(hard_img)
            axs[1].set_title("Hard Paste")
            axs[1].axis("off")

            axs[2].imshow(blended_img)
            axs[2].set_title("Blended")
            axs[2].axis("off")

            fig.suptitle(f"Sample {b}", fontsize=14)

            if save_dir is not None:
                os.makedirs(save_dir, exist_ok=True)
                fig.savefig(os.path.join(save_dir, f"blending_{timestamp_str}_{b}.png"), dpi=300, bbox_inches="tight")

            if show:
                plt.show()
            plt.close(fig)

