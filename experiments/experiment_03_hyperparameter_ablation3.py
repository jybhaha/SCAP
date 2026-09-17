"""
实验03-3：损失函数权重参数消融实验
================================================

该脚本针对 PatchOptimizer_ChangeLoss 中的三个损失权重参数进行消融研究：
- lambda_adv: 对抗损失权重
- lambda_smooth: 平滑损失权重  
- lambda_color: 颜色一致性损失权重

实验设计：
1. 单参数消融：固定其他参数，测试单个参数的不同取值
2. 组合消融：测试不同参数组合的效果
3. 评估指标：ASR (Attack Success Rate)、LPIPS、SSIM

默认使用 500 张 ImageNet 验证集子集，可通过命令行参数调整。
"""

from __future__ import annotations

import argparse
import copy
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models import ResNet50_Weights

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.experiment_configs import get_config, get_target_class
from data.dataset import get_dataloader
from models.my_patch_attack import MyPatchAttack_Eva
from models.patch_optimizer_changeloss import PatchOptimizer_ChangeLoss
from utils.perceptual_evaluator import PerceptualEvaluator
from utils.result_saver import ResultManager


# ---------------------------------------------------------------------------
# 数据类与工具函数
# ---------------------------------------------------------------------------


@dataclass
class LossWeightConfig:
    """定义损失权重配置"""
    lambda_adv: float
    lambda_smooth: float
    lambda_color: float
    name: str = ""
    description: str = ""

    def __post_init__(self):
        if not self.name:
            self.name = f"λ_adv={self.lambda_adv}_smooth={self.lambda_smooth}_color={self.lambda_color}"


def set_global_seed(seed: int = 3407):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_validation_subset(
    data_root: str,
    batch_size: int,
    total_samples: int,
    seed: int,
) -> DataLoader:
    """基于 data.dataset.get_dataloader 构建验证集子集"""
    return get_dataloader(
        data_root=data_root,
        batch_size=batch_size,
        total_samples=total_samples,
        seed=seed,
    )


def cache_batches(
    dataloader: DataLoader,
    max_samples: int,
    device: torch.device,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """将数据缓存为定长批次，确保多次实验使用完全一致的样本"""
    cached: List[Tuple[torch.Tensor, torch.Tensor]] = []
    count = 0
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        cached.append((images, labels))
        count += images.size(0)
        if count >= max_samples:
            break
    return cached


# ---------------------------------------------------------------------------
# 自定义攻击类：使用 PatchOptimizer_ChangeLoss
# ---------------------------------------------------------------------------


class MyPatchAttackWithChangeLoss(MyPatchAttack_Eva):
    """使用 PatchOptimizer_ChangeLoss 的 MyPatchAttack 变体"""

    def __init__(
        self,
        model,
        device='cuda',
        config=None,
        debug=False,
        save_imgs=False,
        save_attack_process=False,
        lambda_adv=1.0,
        lambda_smooth=0.1,
        lambda_color=0.05,
    ):
        # 保存损失权重参数
        self.lambda_adv = lambda_adv
        self.lambda_smooth = lambda_smooth
        self.lambda_color = lambda_color
        
        # 调用父类初始化
        super().__init__(
            model=model,
            device=device,
            config=config,
            debug=debug,
            save_imgs=save_imgs,
            save_attack_process=save_attack_process,
        )

    def _init_modules(self):
        """重写模块初始化，使用 PatchOptimizer_ChangeLoss"""
        from models.Mask2Former.mask2former import SemanticSegmentor
        from models.StableFusion.patch_generator import PatchGenerator
        from utils.color_extractor import ColorExtractor
        from utils.patch_blender import PatchBlender

        self.segmentor = SemanticSegmentor()
        self.patch_gen = PatchGenerator(device=self.device)
        self.color_extractor = ColorExtractor()
        self.patch_blender = PatchBlender(blend_width=self.config["blend_width"])
        
        # 使用 PatchOptimizer_ChangeLoss 替代标准优化器
        self.optimizer = PatchOptimizer_ChangeLoss(
            device=self.device,
            steps=self.config["steps"],
            eps=self.config["eps"],
            alpha=self.config["alpha"],
            model=self.model,
            lambda_adv=self.lambda_adv,
            lambda_smooth=self.lambda_smooth,
            lambda_color=self.lambda_color,
        )
        
        from utils.cam_generator import CAMGenerator
        self.cam_gen = CAMGenerator(
            model_name=self.config["cam_model_name"], 
            device=str(self.device)
        )

    def _optimize(self, x, patched_x, coords, y_or_target):
        """重写优化方法，使用 PatchOptimizer_ChangeLoss 的 optimize_batch"""
        targeted = self.config["targeted_attack"]
        if self.debug:
            self._log(
                f"Optimizing adversarial images with coordinates: {coords} and target: {y_or_target}"
            )
        
        # 获取目标颜色（如果启用颜色匹配）
        target_colors = None
        if self.config.get("use_color_info", True):
            try:
                pil_imgs = self._to_pil(x)
                color_names = self._extract_colors(pil_imgs)
                # 将颜色名称转换为RGB值（简化处理）
                # 这里可以根据实际需求实现更复杂的颜色提取逻辑
                target_colors = None  # 暂时设为None，后续可以扩展
            except:
                target_colors = None
        
        # 使用 PatchOptimizer_ChangeLoss 的 optimize_batch 方法
        result = self.optimizer.optimize_batch(
            x, 
            patched_x, 
            coords, 
            y_or_target, 
            target_colors=target_colors,
            targeted=targeted, 
            return_step_preds=True
        )
        
        if isinstance(result, tuple):
            x_adv, step_preds_list = result
        else:
            x_adv = result
            step_preds_list = []
            
        if self.debug:
            self._log(f"Optimized adversarial images with targeted={targeted}.")
        if self.save_imgs:
            self._save_img(
                self.optimizer.visualize_adversarial_comparison, 
                x=x, 
                x_adv=x_adv
            )
        return x_adv, step_preds_list


# ---------------------------------------------------------------------------
# 损失权重消融实验
# ---------------------------------------------------------------------------


class LossWeightAblationExperiment:
    """损失权重参数消融实验运行器"""

    def __init__(
        self,
        cached_batches: List[Tuple[torch.Tensor, torch.Tensor]],
        classifier,
        device: torch.device,
        base_config: Dict[str, Any],
        target_class: int,
        max_samples: int,
    ):
        self.cached_batches = cached_batches
        self.classifier = classifier
        self.device = device
        self.base_config = base_config
        self.target_class = target_class
        self.max_samples = max_samples
        self.rm = ResultManager.get_instance()
        self.percep = PerceptualEvaluator(device=self.device)

    def _evaluate_config(
        self, 
        loss_config: LossWeightConfig, 
        label: str
    ) -> Dict[str, Any]:
        """评估特定损失权重配置的性能"""
        attack = MyPatchAttackWithChangeLoss(
            model=self.classifier,
            device=str(self.device),
            config=self.base_config,
            debug=False,
            save_imgs=False,
            save_attack_process=False,
            lambda_adv=loss_config.lambda_adv,
            lambda_smooth=loss_config.lambda_smooth,
            lambda_color=loss_config.lambda_color,
        )
        
        stats = {
            "total": 0,
            "success": 0,
            "lpips": [],
            "ssim": [],
        }
        
        for images, labels in self.cached_batches:
            for idx in range(images.size(0)):
                if stats["total"] >= self.max_samples:
                    break
                    
                img = images[idx : idx + 1].clone()
                label = labels[idx : idx + 1].clone()
                target_tensor = torch.full_like(label, self.target_class)
                
                try:
                    adv_img, _ = attack.run(
                        img,
                        label,
                        target=target_tensor if attack.target else None,
                        sample_idx=stats["total"],
                    )
                except Exception as exc:
                    self.rm.log(f"[{label}] 样本 {stats['total']} 攻击失败: {exc}")
                    continue

                adv_img = adv_img.detach().clamp(0, 1)
                with torch.no_grad():
                    logits_adv = self.classifier(adv_img)
                    pred_adv = logits_adv.argmax(dim=1)

                success = (
                    bool((pred_adv == target_tensor).all())
                    if attack.target
                    else bool((pred_adv != label).all())
                )
                if success:
                    stats["success"] += 1

                stats["lpips"].append(self.percep.calc_lpips(img, adv_img))
                stats["ssim"].append(self.percep.calc_ssim(img, adv_img))
                stats["total"] += 1
                
            if stats["total"] >= self.max_samples:
                break

        total = max(1, stats["total"])
        return {
            "num_samples": stats["total"],
            "asr": stats["success"] / total,
            "lpips": float(sum(stats["lpips"]) / len(stats["lpips"])) if stats["lpips"] else 0.0,
            "ssim": float(sum(stats["ssim"]) / len(stats["ssim"])) if stats["ssim"] else 0.0,
        }

    def run(self, configs: List[LossWeightConfig]) -> List[Dict[str, Any]]:
        """运行损失权重消融实验"""
        results = []
        
        for loss_config in configs:
            self.rm.log(f"=== 测试配置: {loss_config.name} ===")
            metrics = self._evaluate_config(loss_config, loss_config.name)
            
            self.rm.log(
                f"[{loss_config.name}] ASR={metrics['asr']*100:.2f}% | "
                f"LPIPS={metrics['lpips']:.3f} | SSIM={metrics['ssim']:.3f}"
            )
            
            results.append({
                "config": {
                    "lambda_adv": loss_config.lambda_adv,
                    "lambda_smooth": loss_config.lambda_smooth,
                    "lambda_color": loss_config.lambda_color,
                    "name": loss_config.name,
                    "description": loss_config.description,
                },
                "metrics": metrics,
                "table_row": {
                    "Configuration": loss_config.description or loss_config.name,
                    "λ_adv": loss_config.lambda_adv,
                    "λ_smooth": loss_config.lambda_smooth,
                    "λ_color": loss_config.lambda_color,
                    "ASR (%)": round(metrics["asr"] * 100, 1),
                    "LPIPS": round(metrics["lpips"], 3),
                    "SSIM": round(metrics["ssim"], 3),
                },
            })
        
        return results


# ---------------------------------------------------------------------------
# 实验配置构建
# ---------------------------------------------------------------------------


def build_loss_weight_configs() -> List[LossWeightConfig]:
    """构建损失权重消融实验配置"""
    configs = []
    
    # 1. 基线配置（默认值）
    configs.append(LossWeightConfig(
        lambda_adv=1.0,
        lambda_smooth=0.1,
        lambda_color=0.05,
        name="baseline",
        description="Baseline (default)"
    ))
    
    # 2. 单参数消融：lambda_adv
    for lambda_adv in [0.5, 1.0, 2.0, 5.0]:
        configs.append(LossWeightConfig(
            lambda_adv=lambda_adv,
            lambda_smooth=0.1,
            lambda_color=0.05,
            name=f"lambda_adv_{lambda_adv}",
            description=f"λ_adv={lambda_adv} (fixed others)"
        ))
    
    # 3. 单参数消融：lambda_smooth
    for lambda_smooth in [0.0, 0.05, 0.1, 0.2, 0.5]:
        configs.append(LossWeightConfig(
            lambda_adv=1.0,
            lambda_smooth=lambda_smooth,
            lambda_color=0.05,
            name=f"lambda_smooth_{lambda_smooth}",
            description=f"λ_smooth={lambda_smooth} (fixed others)"
        ))
    
    # 4. 单参数消融：lambda_color
    for lambda_color in [0.0, 0.01, 0.05, 0.1, 0.2]:
        configs.append(LossWeightConfig(
            lambda_adv=1.0,
            lambda_smooth=0.1,
            lambda_color=lambda_color,
            name=f"lambda_color_{lambda_color}",
            description=f"λ_color={lambda_color} (fixed others)"
        ))
    
    # 5. 组合消融：高对抗损失 + 不同平滑/颜色权重
    for lambda_smooth in [0.0, 0.1, 0.2]:
        for lambda_color in [0.0, 0.05, 0.1]:
            configs.append(LossWeightConfig(
                lambda_adv=2.0,
                lambda_smooth=lambda_smooth,
                lambda_color=lambda_color,
                name=f"high_adv_smooth_{lambda_smooth}_color_{lambda_color}",
                description=f"High λ_adv=2.0, λ_smooth={lambda_smooth}, λ_color={lambda_color}"
            ))
    
    # 6. 组合消融：平衡配置
    configs.append(LossWeightConfig(
        lambda_adv=1.0,
        lambda_smooth=0.2,
        lambda_color=0.1,
        name="balanced",
        description="Balanced (λ_adv=1.0, λ_smooth=0.2, λ_color=0.1)"
    ))
    
    # 7. 极端配置：仅对抗损失
    configs.append(LossWeightConfig(
        lambda_adv=1.0,
        lambda_smooth=0.0,
        lambda_color=0.0,
        name="adversarial_only",
        description="Adversarial only (no smooth/color)"
    ))
    
    return configs


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="损失函数权重参数消融实验"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=os.environ.get("IMAGENET_VAL_ROOT", "/home/jyb/0code/Data/ImageNet/val"),
        help="ImageNet 验证集路径",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="批次大小",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=500,
        help="最大样本数（默认500）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=3407,
        help="随机种子",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="计算设备",
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default="optimized",
        help="基础配置名称（default/optimized）",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_global_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    rm = ResultManager.get_instance()
    rm.set_experiment("LossWeightAblation")
    rm.set_test("loss_weight_ablation_v3")
    rm.enable_image_saving(False)
    
    rm.log("加载 ResNet-50 权重...")
    classifier = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).eval().to(device)
    
    rm.log("构建验证集子集...")
    dataloader = build_validation_subset(
        data_root=args.data_root,
        batch_size=args.batch_size,
        total_samples=args.max_samples,
        seed=args.seed,
    )
    
    rm.log("缓存数据批次...")
    cached_batches = cache_batches(dataloader, args.max_samples, device)
    rm.log(f"缓存完毕，共 {sum(b[0].size(0) for b in cached_batches)} 张图片")
    
    base_config = get_config(args.config_name)
    target_class = get_target_class()
    
    rm.log("初始化损失权重消融实验...")
    experiment = LossWeightAblationExperiment(
        cached_batches=cached_batches,
        classifier=classifier,
        device=device,
        base_config=base_config,
        target_class=target_class,
        max_samples=args.max_samples,
    )
    
    rm.log("构建损失权重配置列表...")
    loss_configs = build_loss_weight_configs()
    rm.log(f"共 {len(loss_configs)} 个配置需要测试")
    
    rm.log("开始运行损失权重消融实验...")
    results = experiment.run(loss_configs)
    
    # 生成摘要
    summary = {
        "experiment_info": {
            "total_configs": len(results),
            "max_samples": args.max_samples,
            "base_config": args.config_name,
            "target_class": target_class,
        },
        "results": results,
        "table_rows": [r["table_row"] for r in results],
    }
    
    output_path = rm.save_json(summary, filename_prefix="loss_weight_ablation_v3_results")
    rm.log(f"结果已保存：{output_path}")
    
    # 打印摘要表格
    rm.log("\n=== 损失权重消融实验结果摘要 ===")
    rm.log(f"{'Configuration':<40} {'ASR (%)':<10} {'LPIPS':<10} {'SSIM':<10}")
    rm.log("-" * 80)
    for row in summary["table_rows"]:
        rm.log(
            f"{row['Configuration']:<40} "
            f"{row['ASR (%)']:<10.1f} "
            f"{row['LPIPS']:<10.3f} "
            f"{row['SSIM']:<10.3f}"
        )


if __name__ == "__main__":
    main()

