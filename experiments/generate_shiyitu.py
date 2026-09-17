"""
生成算法示意图脚本
==================

该脚本用于生成 SCAP 攻击算法的各个步骤示意图，包括：
1. 原始图片
2. 语义分割结果（区域掩码、可视化）
3. CAM热力图和掩码
4. 补丁位置选择（在原始图片上标注）
5. 生成的补丁图片
6. 使用的prompt（保存为文本文件）
7. 补丁融合后的图片
8. 优化过程中的关键步骤
9. 最终的对抗样本

使用方法：
    python experiments/generate_shiyitu.py --image-path <图片路径> --output-dir <输出目录>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms.functional import to_pil_image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.experiment_configs import get_config, get_target_class
from models.my_patch_attack import MyPatchAttack_Eva
from models.patch_optimizer_changeloss import PatchOptimizer_ChangeLoss
from torchvision import models
from torchvision.models import ResNet50_Weights
from utils.result_saver import ResultManager


class AlgorithmVisualizationGenerator:
    """算法示意图生成器"""

    def __init__(
        self,
        model,
        device: torch.device,
        config: Dict,
        output_dir: str,
        target_class: int = None,
    ):
        self.model = model
        self.device = device
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.target_class = target_class
        self.rm = ResultManager.get_instance()

        # 创建子目录
        self.subdirs = {
            "original": self.output_dir / "01_original",
            "segmentation": self.output_dir / "02_segmentation",
            "cam": self.output_dir / "03_cam",
            "patch_selection": self.output_dir / "04_patch_selection",
            "patches": self.output_dir / "05_patches",
            "prompts": self.output_dir / "06_prompts",
            "blended": self.output_dir / "07_blended",
            "optimization": self.output_dir / "08_optimization",
            "final": self.output_dir / "09_final",
        }
        for subdir in self.subdirs.values():
            subdir.mkdir(exist_ok=True)

        # 存储中间结果
        self.intermediate_results = {}

    def save_image(self, img, filename: str, subdir: str = "original", **kwargs):
        """保存图片"""
        if isinstance(img, torch.Tensor):
            img = self.tensor_to_pil(img)
        elif not isinstance(img, Image.Image):
            img = Image.fromarray(img)

        save_path = self.subdirs[subdir] / filename
        img.save(save_path, **kwargs)
        print(f"保存图片: {save_path}")

    def tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """将tensor转换为PIL图片"""
        if tensor.dim() == 4:
            tensor = tensor[0]  # 取第一个样本
        elif tensor.dim() == 2:
            tensor = tensor.unsqueeze(0).repeat(3, 1, 1)

        # 确保值在[0,1]范围内
        if tensor.max() > 1.0:
            tensor = tensor / 255.0
        tensor = torch.clamp(tensor, 0, 1)

        # 转换为PIL
        img_array = tensor.cpu().detach().numpy().transpose(1, 2, 0)
        img_array = (img_array * 255).astype("uint8")
        return Image.fromarray(img_array)

    def save_cam_heatmap(self, cam: torch.Tensor, original_img: torch.Tensor, filename: str):
        """保存CAM热力图"""
        import numpy as np

        # 将CAM归一化到[0,1]
        if cam.dim() == 2:
            cam_np = cam.cpu().numpy()
        else:
            cam_np = cam[0].cpu().numpy() if cam.dim() == 3 else cam.cpu().numpy()

        cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-8)

        # 转换为PIL图片
        orig_pil = self.tensor_to_pil(original_img)

        # 创建热力图
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(orig_pil)
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        im = axes[1].imshow(cam_np, cmap="jet", alpha=0.5)
        axes[1].imshow(orig_pil, alpha=0.5)
        axes[1].set_title("CAM Heatmap")
        axes[1].axis("off")
        plt.colorbar(im, ax=axes[1])

        save_path = self.subdirs["cam"] / filename
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"保存CAM热力图: {save_path}")

    def save_patch_selection(
        self, image: torch.Tensor, coords_list: List[List[Tuple]], filename: str
    ):
        """保存补丁位置选择可视化"""
        img_pil = self.tensor_to_pil(image)
        draw = ImageDraw.Draw(img_pil)

        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20
            )
        except:
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except:
                font = ImageFont.load_default()

        # 绘制选中的补丁区域
        for box_idx, (x1, y1, x2, y2) in enumerate(coords_list[0]):  # 只处理第一个样本
            # 绘制矩形框
            draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=4)
            # 添加标签
            label = f"P{box_idx+1}"
            draw.text((x1 + 5, y1 + 5), label, fill=(255, 255, 0), font=font)

        self.save_image(img_pil, filename, "patch_selection")
        return img_pil

    def save_prompts(self, prompts: List[List[str]], filename: str = "prompts.txt"):
        """保存prompt信息"""
        save_path = self.subdirs["prompts"] / filename

        with open(save_path, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("生成的补丁Prompt列表\n")
            f.write("=" * 60 + "\n\n")

            for batch_idx, batch_prompts in enumerate(prompts):
                f.write(f"样本 {batch_idx + 1}:\n")
                f.write("-" * 60 + "\n")
                for patch_idx, prompt in enumerate(batch_prompts):
                    f.write(f"  补丁 {patch_idx + 1}: {prompt}\n")
                f.write("\n")

        print(f"保存Prompt信息: {save_path}")

        # 同时保存为JSON格式
        json_path = self.subdirs["prompts"] / "prompts.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(prompts, f, indent=2, ensure_ascii=False)
        print(f"保存Prompt JSON: {json_path}")

    def generate_visualization(
        self, image_path: str, label: int = None, target: int = None
    ):
        """生成完整的算法示意图"""
        print(f"\n开始生成算法示意图...")
        print(f"输入图片: {image_path}")
        print(f"输出目录: {self.output_dir}")

        # 1. 加载和预处理图片
        print("\n[步骤1] 加载原始图片...")
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )
        img_pil = Image.open(image_path).convert("RGB")
        img_tensor = transform(img_pil).unsqueeze(0).to(self.device)

        # 保存原始图片
        self.save_image(img_tensor, "original_image.png", "original")

        # 获取真实标签（如果未提供）
        if label is None:
            with torch.no_grad():
                logits = self.model(img_tensor)
                label = logits.argmax(dim=1).item()
                print(f"检测到的标签: {label}")

        label_tensor = torch.tensor([label]).to(self.device)
        target_tensor = (
            torch.tensor([target if target else self.target_class]).to(self.device)
            if self.config.get("targeted_attack", False)
            else None
        )

        # 2. 创建攻击器（启用保存攻击过程）
        print("\n[步骤2] 初始化攻击器...")
        attack = MyPatchAttack_Eva(
            model=self.model,
            device=str(self.device),
            config=self.config,
            debug=True,
            save_imgs=False,
            save_attack_process=True,  # 启用保存攻击过程
        )

        # 3. 运行攻击并收集中间结果
        print("\n[步骤3] 运行攻击流程...")

        # 手动执行各个步骤以收集中间结果
        pil_imgs = attack._to_pil(img_tensor)
        color_names = attack._extract_colors(pil_imgs)
        print(f"提取的主要颜色: {color_names[0]}")

        # 语义分割
        print("\n[步骤4] 执行语义分割...")
        seg_maps, region_names, region_masks = attack._segment_semantics(img_tensor)
        print(f"检测到的语义区域: {region_names[0]}")

        # 保存语义分割结果
        seg_mask = region_masks[0]
        if hasattr(seg_mask, "cpu"):
            seg_mask_np = seg_mask.cpu().numpy()
        else:
            seg_mask_np = seg_mask

        # 创建语义区域可视化
        seg_vis = self.tensor_to_pil(img_tensor)
        seg_vis_array = torch.tensor(seg_mask_np).unsqueeze(0).repeat(3, 1, 1).float()
        seg_vis_array = seg_vis_array * 0.5 + 0.5  # 调整透明度
        seg_overlay = self.tensor_to_pil(seg_vis_array.unsqueeze(0))

        # 叠加显示
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(seg_vis)
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        axes[1].imshow(seg_vis)
        axes[1].imshow(seg_overlay, alpha=0.5)
        axes[1].set_title(f"Semantic Region: {region_names[0]}")
        axes[1].axis("off")

        save_path = self.subdirs["segmentation"] / "semantic_region.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"保存语义分割结果: {save_path}")

        # CAM生成
        print("\n[步骤5] 生成CAM热力图...")
        if self.config.get("targeted_attack", False) and target_tensor is not None:
            cam_list, cam_masks = attack._generate_cam_and_masks(
                img_tensor, label=target_tensor[0].item()
            )
        else:
            cam_list, cam_masks = attack._generate_cam_and_masks(img_tensor)

        # 保存CAM热力图
        self.save_cam_heatmap(cam_list[0], img_tensor, "cam_heatmap.png")

        # 保存CAM掩码
        cam_mask = cam_masks[0]
        if hasattr(cam_mask, "cpu"):
            cam_mask_np = cam_mask.cpu().numpy()
        else:
            cam_mask_np = cam_mask

        cam_mask_vis = torch.tensor(cam_mask_np).unsqueeze(0).repeat(3, 1, 1).float()
        cam_mask_vis = cam_mask_vis * 0.5 + 0.5
        self.save_image(cam_mask_vis.unsqueeze(0), "cam_mask.png", "cam")

        # 补丁位置选择
        print("\n[步骤6] 选择补丁位置...")
        coords_list = attack._select_patch_positions(img_tensor, region_masks, cam_masks)
        print(f"选中的补丁位置数量: {len(coords_list[0])}")
        for idx, (x1, y1, x2, y2) in enumerate(coords_list[0]):
            print(f"  补丁 {idx+1}: ({x1}, {y1}) -> ({x2}, {y2})")

        # 保存补丁位置选择可视化
        self.save_patch_selection(img_tensor, coords_list, "patch_selection.png")

        # 补丁生成
        print("\n[步骤7] 生成补丁...")
        if self.config.get("targeted_attack", False) and target_tensor is not None:
            patch_tensor, prompts = attack.patch_gen.generate_patch_by_region_batch(
                region_names=region_names,
                color_names_list=color_names,
                patch_size=self.config["patch_size"],
                per_region_k=self.config["patch_k"],
                #target_classes=target_tensor[0].item(),
                target_classes=[target_tensor[0].item()],
            )
        else:
            patch_tensor, prompts = attack.patch_gen.generate_patch_by_region_batch(
                region_names=region_names,
                color_names_list=color_names,
                patch_size=self.config["patch_size"],
                per_region_k=self.config["patch_k"],
            )

        print(f"生成的Prompt数量: {len(prompts[0])}")
        for idx, prompt in enumerate(prompts[0]):
            print(f"  Prompt {idx+1}: {prompt}")

        # 保存prompt信息
        self.save_prompts(prompts)

        # 转换补丁为tensor并保存
        to_tensor = transforms.ToTensor()
        patch_tensor_torch = torch.stack(
            [torch.stack([to_tensor(p) for p in plist]) for plist in patch_tensor]
        ).to(self.device)

        # 保存每个补丁
        for patch_idx in range(min(3, patch_tensor_torch.size(1))):
            patch_img = patch_tensor_torch[0, patch_idx]
            self.save_image(
                patch_img, f"patch_{patch_idx+1}.png", "patches"
            )

        # 补丁融合
        print("\n[步骤8] 融合补丁...")
        patched_img = attack._blend_patches(img_tensor, patch_tensor_torch, coords_list)
        self.save_image(patched_img, "blended_image.png", "blended")

        # 优化过程
        print("\n[步骤9] 执行对抗优化...")
        # 使用PatchOptimizer_ChangeLoss进行优化
        optimizer = PatchOptimizer_ChangeLoss(
            device=self.device,
            steps=self.config["steps"],
            eps=self.config["eps"],
            alpha=self.config["alpha"],
            model=self.model,
            lambda_adv=self.config.get("lambda_adv", 1.0),
            lambda_smooth=self.config.get("lambda_smooth", 0.1),
            lambda_color=self.config.get("lambda_color", 0.05),
        )

        # 保存优化过程中的关键步骤
        optimization_steps = [0, self.config["steps"] // 4, self.config["steps"] // 2, self.config["steps"] - 1]
        saved_steps = []

        # 手动执行优化步骤以保存中间结果（使用完整的损失函数）
        B, _, H, W = img_tensor.shape
        x_adv = patched_img.clone().detach().to(self.device)
        ori = img_tensor.clone().detach().to(self.device)
        label_opt = target_tensor if (self.config.get("targeted_attack", False) and target_tensor is not None) else label_tensor

        # 生成patch mask和regions
        patch_mask = torch.zeros((B, 1, H, W), dtype=torch.float32, device=self.device)
        patch_regions = []
        for i, coords in enumerate(coords_list):
            patch_regions.append([])
            for (x1, y1, x2, y2) in coords:
                patch_mask[i, 0, y1:y2, x1:x2] = 1.0
                patch_regions[i].append((x1, y1, x2, y2))

        momentum = torch.zeros_like(x_adv).to(self.device)

        # 保存初始状态
        saved_steps.append((0, x_adv.clone()))

        for step in range(1, self.config["steps"]):
            x_adv.requires_grad = True
            outputs = self.model(x_adv)

            # 计算对抗损失
            adv_loss = torch.nn.functional.cross_entropy(outputs, label_opt)
            if self.config.get("targeted_attack", False):
                adv_loss = -adv_loss

            # 提取当前patch区域用于计算辅助损失
            current_patches = []
            for i, regions in enumerate(patch_regions):
                for (x1, y1, x2, y2) in regions:
                    patch = x_adv[i, :, y1:y2, x1:x2]
                    current_patches.append(patch)

            # 计算总损失（包含平滑损失和颜色损失）
            if current_patches:
                combined_patches = torch.cat(current_patches, dim=0)
                # 确保combined_patches是4维张量[B, 3, H, W]格式
                if combined_patches.dim() == 3:
                    # 假设形状是[3*N, H, W]，需要重新调整为[N, 3, H, W]
                    num_patches = combined_patches.shape[0] // 3
                    combined_patches = combined_patches.view(num_patches, 3, combined_patches.shape[1], combined_patches.shape[2])
                smooth_loss = optimizer._total_variation_loss(combined_patches)
                color_loss = 0  # 暂时不使用颜色损失，因为需要target_colors
                total_loss = (
                    optimizer.lambda_adv * adv_loss
                    + optimizer.lambda_smooth * smooth_loss
                    + optimizer.lambda_color * color_loss
                )
            else:
                total_loss = adv_loss

            # 反向传播
            grad = torch.autograd.grad(total_loss, x_adv, retain_graph=False, create_graph=False)[0]
            grad_norm = grad / (grad.abs().mean(dim=[1, 2, 3], keepdim=True) + 1e-8)
            momentum = optimizer.decay * momentum + grad_norm

            x_adv = x_adv.detach() + optimizer.alpha * momentum.sign() * patch_mask
            x_adv = torch.max(torch.min(x_adv, ori + optimizer.eps), ori - optimizer.eps)
            x_adv = torch.clamp(x_adv, 0, 1)

            # 保存关键步骤
            if step in optimization_steps:
                saved_steps.append((step, x_adv.clone()))

        # 保存优化过程中的关键步骤
        for step, step_img in saved_steps:
            self.save_image(
                step_img, f"optimization_step_{step}.png", "optimization"
            )

        # 最终对抗样本
        print("\n[步骤10] 保存最终对抗样本...")
        self.save_image(x_adv, "final_adversarial.png", "final")

        # 创建对比图
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(self.tensor_to_pil(img_tensor))
        axes[0].set_title("Original Image", fontsize=14)
        axes[0].axis("off")

        axes[1].imshow(self.tensor_to_pil(patched_img))
        axes[1].set_title("After Patch Blending", fontsize=14)
        axes[1].axis("off")

        axes[2].imshow(self.tensor_to_pil(x_adv))
        axes[2].set_title("Final Adversarial Image", fontsize=14)
        axes[2].axis("off")

        save_path = self.output_dir / "comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"保存对比图: {save_path}")

        # 保存实验信息
        info = {
            "image_path": image_path,
            "original_label": label,
            "target_label": target_tensor[0].item() if target_tensor is not None else None,
            "detected_region": region_names[0],
            "color_names": color_names[0],
            "patch_positions": [
                {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
                for x1, y1, x2, y2 in coords_list[0]
            ],
            "prompts": prompts[0],
            "config": self.config,
        }

        info_path = self.output_dir / "experiment_info.json"
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
        print(f"保存实验信息: {info_path}")

        print("\n算法示意图生成完成！")
        print(f"所有结果保存在: {self.output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="生成SCAP算法示意图")
    parser.add_argument(
        "--image-path",
        type=str,
        required=True,
        help="输入图片路径",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./algorithm_visualization",
        help="输出目录（默认: ./algorithm_visualization）",
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default="optimized",
        help="配置名称（default/optimized）",
    )
    parser.add_argument(
        "--target-class",
        type=int,
        default=None,
        help="目标类别（用于目标攻击）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="计算设备",
    )
    parser.add_argument(
        "--lambda-adv",
        type=float,
        default=1.0,
        help="对抗损失权重",
    )
    parser.add_argument(
        "--lambda-smooth",
        type=float,
        default=0.1,
        help="平滑损失权重",
    )
    parser.add_argument(
        "--lambda-color",
        type=float,
        default=0.05,
        help="颜色一致性损失权重",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载模型
    print("加载ResNet-50模型...")
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).eval().to(device)

    # 获取配置
    config = get_config(args.config_name)
    config["lambda_adv"] = args.lambda_adv
    config["lambda_smooth"] = args.lambda_smooth
    config["lambda_color"] = args.lambda_color

    # 获取目标类别
    target_class = args.target_class if args.target_class is not None else get_target_class()

    # 创建生成器
    generator = AlgorithmVisualizationGenerator(
        model=model,
        device=device,
        config=config,
        output_dir=args.output_dir,
        target_class=target_class,
    )

    # 生成示意图
    generator.generate_visualization(
        image_path=args.image_path,
        target=target_class if config.get("targeted_attack", False) else None,
    )


if __name__ == "__main__":
    main()

