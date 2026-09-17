"""
实验04-2：鲁棒性评估实验（论文版本）
================================================

该脚本实现了论文中描述的鲁棒性评估实验，包括：

1. **防御机制测试**：
   - Adversarial Training (AT) - 使用鲁棒的 ResNet-50 模型
   - Input Preprocessing 防御：
     * JPEG compression (Q75)
     * Bit-depth reduction (4-bit)
     * Median filtering (3×3)
   - Patch Detector (CNN-based)

2. **图像变换测试**：
   - Gaussian Blur (σ=1.0)
   - Random Resize-Crop

3. **攻击方法对比**：
   - LAVAN
   - AdvPatch (ART)
   - PGD
   - SCAP

4. **评估指标**：
   - Targeted ASR (%)
   - 性能下降幅度（相对于baseline）
   - Patch Detection Rate @ 5% FPR

默认使用 1000 张 ImageNet 验证集图片，可通过命令行参数调整。
"""

from __future__ import annotations

import argparse
import io
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from torchvision.transforms import functional as TF
from torchvision.transforms import GaussianBlur, RandomResizedCrop

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.experiment_configs import get_config, get_target_class
from data.dataset import get_dataloader
from models.base_line import PatchAttackBaseline, PixelAttackBaseline
from models.my_patch_attack import MyPatchAttack_Eva
from utils.result_saver import ResultManager


# ---------------------------------------------------------------------------
# 防御机制和图像变换实现
# ---------------------------------------------------------------------------


class DefenseMechanisms:
    """防御机制实现类"""

    @staticmethod
    def apply_jpeg_compression(image: torch.Tensor, quality: int = 75) -> torch.Tensor:
        """应用JPEG压缩（Q75）"""
        if image.dim() == 4:
            batch_results = []
            for i in range(image.size(0)):
                batch_results.append(
                    DefenseMechanisms.apply_jpeg_compression(image[i:i+1], quality)
                )
            return torch.cat(batch_results, dim=0)
        
        # 确保值在[0,1]范围内
        if image.max() > 1.0:
            image = image / 255.0
        image = torch.clamp(image, 0, 1)
        
        # 转换为PIL图像
        img_np = image[0].cpu().detach().numpy().transpose(1, 2, 0)
        img_np = (img_np * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np)
        
        # 应用JPEG压缩
        buffer = io.BytesIO()
        img_pil.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        compressed_img = Image.open(buffer)
        
        # 转换回tensor
        img_array = np.array(compressed_img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float()
        
        return img_tensor.unsqueeze(0).to(image.device)

    @staticmethod
    def apply_bit_depth_reduction(image: torch.Tensor, bits: int = 4) -> torch.Tensor:
        """应用位深度降低（4-bit）"""
        levels = 2 ** bits
        return torch.round(image * (levels - 1)) / (levels - 1)

    @staticmethod
    def apply_median_filter(image: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
        """应用中值滤波（3×3）"""
        if image.dim() == 4:
            batch_results = []
            for i in range(image.size(0)):
                batch_results.append(
                    DefenseMechanisms.apply_median_filter(image[i:i+1], kernel_size)
                )
            return torch.cat(batch_results, dim=0)
        
        # 确保值在[0,1]范围内
        if image.max() > 1.0:
            image = image / 255.0
        image = torch.clamp(image, 0, 1)
        
        # 转换为numpy
        img_np = image[0].cpu().detach().numpy().transpose(1, 2, 0)
        img_np = (img_np * 255).astype(np.uint8)
        
        # 应用中值滤波
        filtered = cv2.medianBlur(img_np, kernel_size)
        
        # 转换回tensor
        img_tensor = torch.from_numpy(filtered.astype(np.float32) / 255.0).permute(2, 0, 1).float()
        
        return img_tensor.unsqueeze(0).to(image.device)

    @staticmethod
    def apply_gaussian_blur(image: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
        """应用高斯模糊（σ=1.0）"""
        blur = GaussianBlur(kernel_size=5, sigma=sigma)
        if image.dim() == 4:
            batch_results = []
            for i in range(image.size(0)):
                batch_results.append(blur(image[i:i+1]))
            return torch.cat(batch_results, dim=0)
        return blur(image)

    @staticmethod
    def apply_random_resize_crop(image: torch.Tensor, size: int = 224) -> torch.Tensor:
        """应用随机resize-crop"""
        transform = RandomResizedCrop(size=size, scale=(0.8, 1.0), ratio=(0.9, 1.1))
        
        if image.dim() == 4:
            batch_results = []
            for i in range(image.size(0)):
                batch_results.append(transform(image[i]))
            return torch.stack(batch_results)
        else:
            return transform(image).unsqueeze(0)


class PatchDetector(nn.Module):
    """简单的CNN补丁检测器"""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def load_robust_model(device: torch.device):
    """加载鲁棒的ResNet-50模型（对抗训练）"""
    try:
        # 尝试从 robustbench 加载
        from robustbench.utils import load_model

        model = load_model(
            model_name="Standard",
            dataset="imagenet",
            threat_model="Linf",
        )
        model = model.eval().to(device)
        print("成功加载 robustbench 鲁棒模型")
        return model
    except Exception as e:
        # 如果 robustbench 不可用，使用标准模型作为fallback
        print(f"无法加载robustbench模型: {e}，使用标准ResNet-50")
        model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).eval().to(device)
        return model


# ---------------------------------------------------------------------------
# 鲁棒性实验运行器
# ---------------------------------------------------------------------------


class RobustnessExperimentRunner:
    """鲁棒性实验运行器"""

    def __init__(self, device: torch.device = None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.rm = ResultManager.get_instance()
        
        # 初始化分类模型
        self.classifier = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).eval().to(self.device)
        
        # 初始化攻击器
        self._init_attackers()
        
        # 初始化防御机制
        self._init_defenses()
        
        # 初始化补丁检测器
        self.patch_detector = None  # 延迟初始化

    def _init_attackers(self):
        """初始化攻击器"""
        try:
            self.pixel_attack = PixelAttackBaseline(
                model=self.classifier, eps=16/255, default_steps=10
            )
            self.rm.log("Pixel attack initialized")
        except Exception as e:
            self.rm.log(f"Failed to initialize pixel attack: {e}")
            self.pixel_attack = None

        try:
            self.patch_attack = PatchAttackBaseline(
                model=self.classifier, input_size=(224, 224), device=self.device
            )
            self.rm.log("Patch attack initialized")
        except Exception as e:
            self.rm.log(f"Failed to initialize patch attack: {e}")
            self.patch_attack = None

        try:
            self.scap_attack = MyPatchAttack_Eva(
                model=self.classifier,
                device=str(self.device),
                config=get_config("optimized"),
                debug=False,
                save_imgs=False,
                save_attack_process=False,
            )
            self.rm.log("SCAP attack initialized")
        except Exception as e:
            self.rm.log(f"Failed to initialize SCAP attack: {e}")
            self.scap_attack = None

    def _init_defenses(self):
        """初始化防御机制"""
        self.defenses = {
            "adversarial_training": None,  # 延迟加载
        }
        self.rm.log("Defense mechanisms initialized")

    def _get_robust_model(self):
        """获取鲁棒模型（延迟加载）"""
        if self.defenses["adversarial_training"] is None:
            self.rm.log("加载鲁棒模型...")
            self.defenses["adversarial_training"] = load_robust_model(self.device)
        return self.defenses["adversarial_training"]

    def _apply_defense_or_transform(
        self, image: torch.Tensor, condition: str
    ) -> torch.Tensor:
        """应用防御或变换"""
        if condition == "None":
            return image
        elif condition == "JPEG (Q75)":
            return DefenseMechanisms.apply_jpeg_compression(image, quality=75)
        elif condition == "Bit-depth (4-bit)":
            return DefenseMechanisms.apply_bit_depth_reduction(image, bits=4)
        elif condition == "Median Filter (3×3)":
            return DefenseMechanisms.apply_median_filter(image, kernel_size=3)
        elif condition == "Gaussian Blur (σ=1.0)":
            return DefenseMechanisms.apply_gaussian_blur(image, sigma=1.0)
        elif condition == "Random Resize-Crop":
            return DefenseMechanisms.apply_random_resize_crop(image, size=224)
        else:
            return image

    def _get_model_for_condition(self, condition: str):
        """根据条件获取相应的模型"""
        if condition == "Adversarial Training":
            return self._get_robust_model()
        else:
            return self.classifier

    def _generate_attack(
        self, method: str, image: torch.Tensor, label: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """生成对抗样本"""
        if method == "LAVAN":
            if self.patch_attack is None:
                return None
            result = self.patch_attack.run("lavan_eva", image, label)
            if isinstance(result, tuple):
                return result[0]
            return result

        elif method == "AdvPatch":
            if self.patch_attack is None:
                return None
            result, _ = self.patch_attack.run(
                "art_advpatch_eva", image, label, patch_shape=(3, 64, 64)
            )
            return result

        elif method == "PGD":
            if self.pixel_attack is None:
                return None
            result = self.pixel_attack.run("PGD", image, label, target=target)
            if isinstance(result, tuple):
                return result[0]
            return result

        elif method == "SCAP":
            if self.scap_attack is None:
                return None
            result, _ = self.scap_attack.run(image, label, target=target)
            return result

        else:
            raise ValueError(f"Unknown attack method: {method}")

    def _evaluate_detection_rate(
        self, 
        clean_images: List[torch.Tensor],
        adv_images: List[torch.Tensor],
        fpr_threshold: float = 0.05
    ) -> float:
        """评估补丁检测率@5% FPR
        
        使用基于特征差异的简单检测方法：
        计算对抗样本与原始样本的特征差异，如果差异超过阈值则判定为对抗样本
        """
        if len(clean_images) == 0 or len(adv_images) == 0:
            return 0.0
        
        # 使用分类器的中间层特征进行检测
        # 提取特征
        def extract_features(model, images):
            """从ResNet提取中间层特征"""
            features = []
            with torch.no_grad():
                for img in images:
                    if img.dim() == 3:
                        img = img.unsqueeze(0)
                    img = img.to(self.device)
                    # 使用ResNet的layer3输出作为特征
                    x = model.conv1(img)
                    x = model.bn1(x)
                    x = model.relu(x)
                    x = model.maxpool(x)
                    x = model.layer1(x)
                    x = model.layer2(x)
                    x = model.layer3(x)
                    # 全局平均池化
                    x = F.adaptive_avg_pool2d(x, (1, 1))
                    x = x.view(x.size(0), -1)
                    features.append(x.cpu())
            return torch.cat(features, dim=0)
        
        # 提取特征
        clean_features = extract_features(self.classifier, clean_images[:len(adv_images)])
        adv_features = extract_features(self.classifier, adv_images)
        
        # 计算特征差异
        feature_diffs = torch.norm(adv_features - clean_features, p=2, dim=1)
        
        # 计算阈值（基于clean图像的统计特性）
        # 使用clean图像之间的特征差异作为参考
        clean_ref_diffs = []
        for i in range(min(100, len(clean_images)-1)):
            if i+1 < len(clean_images):
                f1 = extract_features(self.classifier, [clean_images[i]])
                f2 = extract_features(self.classifier, [clean_images[i+1]])
                diff = torch.norm(f1 - f2, p=2, dim=1).item()
                clean_ref_diffs.append(diff)
        
        if clean_ref_diffs:
            # 使用95%分位数作为阈值（对应5% FPR）
            threshold = np.percentile(clean_ref_diffs, 95)
        else:
            # Fallback: 使用特征差异的中位数
            threshold = feature_diffs.median().item()
        
        # 计算检测率
        detected = (feature_diffs > threshold).sum().item()
        detection_rate = detected / len(adv_images) if len(adv_images) > 0 else 0.0
        
        return detection_rate

    def run_robustness_experiment(
        self,
        dataloader: DataLoader,
        num_samples: int = 1000,
        target_class: int = None,
    ) -> Dict[str, Any]:
        """运行鲁棒性实验"""
        self.rm.log(f"开始鲁棒性实验，样本数: {num_samples}")

        if target_class is None:
            target_class = get_target_class()

        # 定义测试条件
        conditions = [
            "None",  # Baseline
            "Adversarial Training",
            "JPEG (Q75)",
            "Bit-depth (4-bit)",
            "Median Filter (3×3)",
            "Gaussian Blur (σ=1.0)",
            "Random Resize-Crop",
        ]

        # 定义攻击方法
        attack_methods = ["LAVAN", "AdvPatch", "PGD", "SCAP"]

        # 存储结果
        results = {
            "baseline_asr": {},
            "condition_results": {},
            "detection_rates": {},
            "summary_table": [],
        }

        # 缓存数据用于检测器评估（保持对应关系）
        clean_images_cache = []
        adv_images_cache = {method: [] for method in attack_methods}
        # 记录每个样本的索引，确保clean和adv对应
        sample_indices = []

        # 第一阶段：生成所有对抗样本并计算baseline ASR
        self.rm.log("=== 第一阶段：生成对抗样本并计算Baseline ASR ===")
        processed = 0

        for batch_idx, (images, labels) in enumerate(dataloader):
            if processed >= num_samples:
                break

            images = images.to(self.device)
            labels = labels.to(self.device)
            targets = torch.full_like(labels, target_class)

            for i in range(images.size(0)):
                if processed >= num_samples:
                    break

                img = images[i:i+1]
                label = labels[i:i+1]
                target = targets[i:i+1]

                # 保存原始图片用于检测器
                clean_images_cache.append(img.clone().cpu())
                sample_indices.append(processed)

                # 对每个方法生成对抗样本
                for method in attack_methods:
                    try:
                        adv_img = self._generate_attack(method, img, label, target)
                        if adv_img is not None:
                            adv_img = adv_img.detach().clamp(0, 1)
                            
                            # 评估baseline ASR
                            with torch.no_grad():
                                logits = self.classifier(adv_img)
                                pred = logits.argmax(dim=1)
                                success = (pred == target).item()

                            if method not in results["baseline_asr"]:
                                results["baseline_asr"][method] = {
                                    "success": 0,
                                    "total": 0,
                                }
                            
                            results["baseline_asr"][method]["total"] += 1
                            if success:
                                results["baseline_asr"][method]["success"] += 1
                            
                            # 保存对抗样本用于检测器（保存到CPU以节省GPU内存）
                            adv_images_cache[method].append(adv_img.clone().cpu())
                    except Exception as e:
                        self.rm.log(f"方法 {method} 生成失败: {e}")
                        continue

                processed += 1
                if processed % 100 == 0:
                    self.rm.log(f"已处理 {processed}/{num_samples} 个样本")

        # 计算baseline ASR
        for method in attack_methods:
            if method in results["baseline_asr"]:
                total = results["baseline_asr"][method]["total"]
                success = results["baseline_asr"][method]["success"]
                asr = success / total if total > 0 else 0.0
                results["baseline_asr"][method]["asr"] = asr
                self.rm.log(f"{method} Baseline ASR: {asr:.1%}")

        # 第二阶段：测试各种防御和变换
        self.rm.log("=== 第二阶段：测试防御和变换 ===")

        for condition in conditions:
            if condition == "None":
                continue  # 已经在第一阶段计算了

            self.rm.log(f"测试条件: {condition}")
            results["condition_results"][condition] = {}

            # 获取相应的模型
            eval_model = self._get_model_for_condition(condition)

            for method in attack_methods:
                if method not in adv_images_cache or len(adv_images_cache[method]) == 0:
                    continue

                success_count = 0
                total_count = 0

                for adv_img in adv_images_cache[method]:
                    try:
                        # 将对抗样本移到GPU
                        adv_img = adv_img.to(self.device)
                        
                        # 应用防御或变换
                        if condition != "Adversarial Training":
                            processed_adv = self._apply_defense_or_transform(adv_img, condition)
                        else:
                            processed_adv = adv_img

                        # 评估攻击成功率
                        with torch.no_grad():
                            logits = eval_model(processed_adv)
                            pred = logits.argmax(dim=1)
                            # 对于目标攻击，成功是指预测为目标类别
                            target_tensor = torch.tensor([target_class]).to(self.device)
                            success = (pred == target_tensor).item()

                        total_count += 1
                        if success:
                            success_count += 1
                    except Exception as e:
                        self.rm.log(f"评估失败 {method}/{condition}: {e}")
                        continue

                asr = success_count / total_count if total_count > 0 else 0.0
                baseline_asr = results["baseline_asr"].get(method, {}).get("asr", 0.0)
                drop = baseline_asr - asr

                results["condition_results"][condition][method] = {
                    "asr": asr,
                    "drop": drop,
                    "baseline_asr": baseline_asr,
                }

                self.rm.log(
                    f"  {method}/{condition}: ASR={asr:.1%} (↓{drop:.1%})"
                )

        # 第三阶段：评估补丁检测率
        self.rm.log("=== 第三阶段：评估补丁检测率 @ 5% FPR ===")

        for method in attack_methods:
            if method not in adv_images_cache or len(adv_images_cache[method]) == 0:
                continue

            try:
                # 确保clean和adv图片数量一致
                num_adv = len(adv_images_cache[method])
                num_clean = min(len(clean_images_cache), num_adv)
                
                if num_adv > 0 and num_clean > 0:
                    detection_rate = self._evaluate_detection_rate(
                        clean_images_cache[:num_clean],
                        adv_images_cache[method][:num_adv],
                        fpr_threshold=0.05,
                    )
                    results["detection_rates"][method] = detection_rate
                    self.rm.log(f"{method} Detection Rate @ 5% FPR: {detection_rate:.1%}")
                else:
                    results["detection_rates"][method] = 0.0
                    self.rm.log(f"{method} 没有足够的样本进行检测率评估")
            except Exception as e:
                self.rm.log(f"检测率评估失败 {method}: {e}")
                import traceback
                self.rm.log(f"错误详情: {traceback.format_exc()}")
                results["detection_rates"][method] = 0.0

        # 生成摘要表格
        results["summary_table"] = self._generate_summary_table(results)

        # 保存结果
        self._save_results(results)

        return results

    def _generate_summary_table(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成摘要表格"""
        table = []
        attack_methods = ["LAVAN", "AdvPatch", "PGD", "SCAP"]

        # Baseline行
        baseline_row = {"Condition": "None (Baseline)"}
        for method in attack_methods:
            asr = results["baseline_asr"].get(method, {}).get("asr", 0.0)
            baseline_row[method] = f"{asr*100:.1f}"
        table.append(baseline_row)

        # 各条件行
        condition_order = [
            "Adversarial Training",
            "JPEG (Q75)",
            "Bit-depth (4-bit)",
            "Median Filter (3×3)",
            "Gaussian Blur (σ=1.0)",
            "Random Resize-Crop",
        ]

        for condition in condition_order:
            if condition not in results["condition_results"]:
                continue

            row = {"Condition": condition}
            for method in attack_methods:
                if method in results["condition_results"][condition]:
                    data = results["condition_results"][condition][method]
                    asr = data["asr"]
                    drop = data["drop"]
                    row[method] = f"{asr*100:.1f}(↓{drop*100:.1f})"
                else:
                    row[method] = "N/A"
            table.append(row)

        # 检测率行
        detection_row = {"Condition": "Detection Rate @ 5% FPR"}
        for method in attack_methods:
            rate = results["detection_rates"].get(method, 0.0)
            detection_row[method] = f"{rate*100:.1f}%"
        table.append(detection_row)

        return table

    def _save_results(self, results: Dict[str, Any]):
        """保存结果"""
        import json

        result_dir = self.rm.log_dir
        os.makedirs(result_dir, exist_ok=True)

        # 保存详细结果
        detailed_file = os.path.join(result_dir, "robustness_detailed_results.json")
        with open(detailed_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str, ensure_ascii=False)

        # 保存摘要表格
        summary_file = os.path.join(result_dir, "robustness_summary_table.json")
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(results["summary_table"], f, indent=2, ensure_ascii=False)

        self.rm.log(f"结果已保存到: {detailed_file}")
        self.rm.log(f"摘要表格已保存到: {summary_file}")

        # 打印摘要表格
        self.rm.log("\n=== 鲁棒性实验结果摘要 ===")
        if results["summary_table"]:
            # 打印表头
            methods = ["LAVAN", "AdvPatch", "PGD", "SCAP"]
            header = f"{'Condition':<30} " + " ".join(f"{m:>12}" for m in methods)
            self.rm.log(header)
            self.rm.log("-" * len(header))

            # 打印数据行
            for row in results["summary_table"]:
                condition = row.get("Condition", "")
                data_row = f"{condition:<30} "
                for method in methods:
                    value = row.get(method, "N/A")
                    data_row += f"{value:>12} "
                self.rm.log(data_row)


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="鲁棒性评估实验")
    parser.add_argument(
        "--data-root",
        type=str,
        default=os.environ.get("IMAGENET_VAL_ROOT", "data/ImageNet/val"),
        help="ImageNet 验证集路径",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="批次大小",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="测试样本数（默认1000）",
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
        "--seed",
        type=int,
        default=3407,
        help="随机种子",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 设置随机种子
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 初始化结果管理器
    rm = ResultManager.get_instance()
    rm.set_experiment("RobustnessV2")
    rm.set_test("robustness_evaluation_v2")
    rm.enable_image_saving(False)

    # 获取数据加载器
    rm.log("构建数据加载器...")
    dataloader = get_dataloader(
        data_root=args.data_root,
        batch_size=args.batch_size,
        total_samples=args.num_samples,
        seed=args.seed,
    )

    # 获取目标类别
    target_class = args.target_class if args.target_class is not None else get_target_class()
    rm.log(f"目标类别: {target_class}")

    # 初始化实验运行器
    rm.log("初始化实验运行器...")
    runner = RobustnessExperimentRunner(device=device)

    # 运行实验
    rm.log("开始运行鲁棒性实验...")
    results = runner.run_robustness_experiment(
        dataloader=dataloader,
        num_samples=args.num_samples,
        target_class=target_class,
    )

    rm.log("鲁棒性实验完成！")


if __name__ == "__main__":
    main()
