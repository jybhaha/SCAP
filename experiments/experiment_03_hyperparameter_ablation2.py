"""
实验03-2：SCAP 关键组件与超参数敏感性实验
================================================

该脚本实现了文档描述的“消融与超参数敏感性”实验流程，涵盖：

1. **组件级累积消融**：从随机补丁放置的基线开始，依次启用语义引导、注意力规避、颜色匹配以及边界感知融合，并统计
   ResNet-50 上的 *Targeted Attack Success Rate*、LPIPS 与 SSIM。
2. **超参数敏感性分析**：基于自动化消融框架，对核心超参数 (`patch_size`、`patch_k`、`steps`、`eps`、`alpha`
   以及 `cam_percentile`) 进行系统扫参，评估指标同上。

默认使用 500 张 ImageNet 验证集子集以兼顾效率与统计可靠性，可通过命令行参数调整。
"""

from __future__ import annotations

import argparse
import copy
import os
import random
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models import ResNet50_Weights

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.experiment_configs import get_ablation_params, get_config, get_target_class
from data.dataset import get_dataloader
from models.my_patch_attack import MyPatchAttack_Eva
from utils.perceptual_evaluator import PerceptualEvaluator
from utils.result_saver import ResultManager


# ---------------------------------------------------------------------------
# 数据类与工具函数
# ---------------------------------------------------------------------------


DEFAULT_COMPONENT_FLAGS = {
    "semantic_guidance": True,
    "attention_avoidance": True,
    "color_matching": True,
    "boundary_aware_fusion": True,
}


@dataclass
class ComponentVariant:
    """定义组件级配置及其说明。"""

    name: str
    description: str
    flags: Dict[str, bool]


@dataclass
class HyperParamSpec:
    """定义单个超参数的扫参设置。"""

    name: str
    values: List[Any]
    description: str


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
    """基于 data.dataset.get_dataloader 构建验证集子集。"""
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
    """将数据缓存为定长批次，确保多次实验使用完全一致的样本。"""
    cached: List[Tuple[torch.Tensor, torch.Tensor]] = []
    collected = 0
    for images, labels in dataloader:
        if collected >= max_samples:
            break
        take = min(images.size(0), max_samples - collected)
        cached.append(
            (
                images[:take].to(device, non_blocking=True),
                labels[:take].to(device, non_blocking=True),
            )
        )
        collected += take
    return cached


# ---------------------------------------------------------------------------
# 自定义攻击器：支持按需启用/关闭核心组件
# ---------------------------------------------------------------------------


class SCAPComponentVariantAttack(MyPatchAttack_Eva):
    """在 MyPatchAttack_Eva 基础上，按需关闭各个组件。"""

    def __init__(
        self,
        model,
        device: torch.device,
        base_config: Dict[str, Any],
        component_flags: Dict[str, bool],
    ):
        self.component_flags = copy.deepcopy(DEFAULT_COMPONENT_FLAGS)
        self.component_flags.update(component_flags)
        custom_config = copy.deepcopy(base_config)
        if not self.component_flags["boundary_aware_fusion"]:
            custom_config["blend_width"] = 0
        if not self.component_flags["attention_avoidance"]:
            # 强制放宽 CAM 约束，避免被动启用
            custom_config["cam_thresh_ratio"] = 0.0
        self._rng = random.Random(base_config.get("seed", 3407))
        super().__init__(
            model=model,
            device=str(device),
            config=custom_config,
            debug=False,
            save_imgs=False,
            save_attack_process=False,
        )

    # ---- Helper overrides -------------------------------------------------
    def _segment_semantics(self, x):
        if not self.component_flags["semantic_guidance"]:
            batch, _, h, w = x.shape
            dummy_masks = [
                torch.ones(h, w, device=x.device, dtype=torch.float32) for _ in range(batch)
            ]
            dummy_names = ["uniform_region"] * batch
            return [None] * batch, dummy_names, dummy_masks
        return super()._segment_semantics(x)

    def _generate_cam_and_masks(self, x, label=None):
        if not self.component_flags["attention_avoidance"]:
            batch, _, h, w = x.shape
            zero_map = torch.zeros(h, w, device=x.device)
            cams = [zero_map for _ in range(batch)]
            masks = [zero_map for _ in range(batch)]
            return cams, masks
        return super()._generate_cam_and_masks(x, label)

    def _extract_colors(self, pil_imgs):
        if not self.component_flags["color_matching"]:
            return [[] for _ in pil_imgs]
        return super()._extract_colors(pil_imgs)

    def _select_patch_positions(self, x, region_masks, cam_masks):
        if not self.component_flags["semantic_guidance"]:
            return self._random_patch_positions(x.shape)
        return super()._select_patch_positions(x, region_masks, cam_masks)

    def _blend_patches(self, x, patch_tensor, coords):
        if not self.component_flags["boundary_aware_fusion"]:
            return self._naive_overlay(x, patch_tensor, coords)
        return super()._blend_patches(x, patch_tensor, coords)

    # ---- Internal utilities ----------------------------------------------
    def _random_patch_positions(self, img_shape: torch.Size):
        batch, _, h, w = img_shape
        ph, pw = self.config["patch_size"]
        k = self.config["patch_k"]
        coords: List[List[Tuple[int, int, int, int]]] = []
        for _ in range(batch):
            boxes = []
            for _ in range(k):
                if h - ph <= 0 or w - pw <= 0:
                    x1 = 0
                    y1 = 0
                else:
                    x1 = self._rng.randint(0, max(0, w - pw))
                    y1 = self._rng.randint(0, max(0, h - ph))
                boxes.append((x1, y1, x1 + pw, y1 + ph))
            coords.append(boxes)
        return coords

    def _naive_overlay(self, x, patch_tensor, coords):
        blended = x.clone()
        for b, boxes in enumerate(coords):
            for idx, (x1, y1, x2, y2) in enumerate(boxes):
                patch_bank_index = min(idx, patch_tensor.size(1) - 1)
                patch = patch_tensor[b, patch_bank_index]
                blended[b, :, y1:y2, x1:x2] = patch
        return blended


# ---------------------------------------------------------------------------
# 组件消融实验
# ---------------------------------------------------------------------------


class ComponentAblationExperiment:
    def __init__(
        self,
        cached_batches: List[Tuple[torch.Tensor, torch.Tensor]],
        classifier,
        lr_device: torch.device,
        base_config: Dict[str, Any],
        target_class: int,
        max_samples: int,
    ):
        self.cached_batches = cached_batches
        self.classifier = classifier
        self.device = lr_device
        self.base_config = base_config
        self.target_class = target_class
        self.max_samples = max_samples
        self.rm = ResultManager.get_instance()
        self.percep = PerceptualEvaluator(device=self.device)

    def _evaluate_attack(self, attack: MyPatchAttack_Eva, tag: str) -> Dict[str, Any]:
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
                    self.rm.log(f"[{tag}] 样本 {stats['total']} 攻击失败: {exc}")
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

    def run(self, variants: List[ComponentVariant]) -> List[Dict[str, Any]]:
        rows = []
        for variant in variants:
            self.rm.log(f"=== 组件消融：{variant.name} ===")
            attack = SCAPComponentVariantAttack(
                model=self.classifier,
                device=self.device,
                base_config=self.base_config,
                component_flags=variant.flags,
            )
            metrics = self._evaluate_attack(attack, variant.name)
            self.rm.log(
                f"[{variant.name}] ASR={metrics['asr']*100:.2f}% | "
                f"LPIPS={metrics['lpips']:.3f} | SSIM={metrics['ssim']:.3f}"
            )
            rows.append(
                {
                    "configuration": variant.name,
                    "description": variant.description,
                    "flags": variant.flags,
                    "metrics": metrics,
                    "table_row": {
                        "Configuration": variant.description,
                        "ASR (%)": round(metrics["asr"] * 100, 1),
                        "LPIPS": round(metrics["lpips"], 3),
                        "SSIM": round(metrics["ssim"], 3),
                    },
                }
            )
        return rows


# ---------------------------------------------------------------------------
# 超参数敏感性实验
# ---------------------------------------------------------------------------


class HyperparameterSensitivityExperiment:
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

    def _evaluate_config(self, config: Dict[str, Any], label: str) -> Dict[str, Any]:
        attack = MyPatchAttack_Eva(
            model=self.classifier,
            device=str(self.device),
            config=config,
            debug=False,
            save_imgs=False,
            save_attack_process=False,
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
                label_tensor = labels[idx : idx + 1].clone()
                target_tensor = torch.full_like(label_tensor, self.target_class)
                try:
                    adv_img, _ = attack.run(
                        img,
                        label_tensor,
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
                    else bool((pred_adv != label_tensor).all())
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

    def run(self, specs: List[HyperParamSpec]) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        for spec in specs:
            self.rm.log(f"=== 扫参：{spec.name} ===")
            results[spec.name] = {}
            for value in spec.values:
                cfg = copy.deepcopy(self.base_config)
                cfg[spec.name] = value
                label = f"{spec.name}={value}"
                metrics = self._evaluate_config(cfg, label)
                self.rm.log(
                    f"[{label}] ASR={metrics['asr']*100:.2f}% | "
                    f"LPIPS={metrics['lpips']:.3f} | SSIM={metrics['ssim']:.3f}"
                )
                results[spec.name][str(value)] = metrics
        return results


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def build_component_variants() -> List[ComponentVariant]:
    variants = []
    flags = copy.deepcopy(DEFAULT_COMPONENT_FLAGS)
    # 1. 随机放置基线
    flags.update(
        {
            "semantic_guidance": False,
            "attention_avoidance": False,
            "color_matching": False,
            "boundary_aware_fusion": False,
        }
    )
    variants.append(
        ComponentVariant(
            name="random_placement",
            description="Random placement",
            flags=copy.deepcopy(flags),
        )
    )
    # 2. 语义引导
    flags["semantic_guidance"] = True
    variants.append(
        ComponentVariant(
            name="semantic_guidance",
            description="+ Semantic guidance",
            flags=copy.deepcopy(flags),
        )
    )
    # 3. 注意力规避
    flags["attention_avoidance"] = True
    variants.append(
        ComponentVariant(
            name="attention_avoidance",
            description="+ Attention avoidance",
            flags=copy.deepcopy(flags),
        )
    )
    # 4. 颜色匹配
    flags["color_matching"] = True
    variants.append(
        ComponentVariant(
            name="color_matching",
            description="+ Color matching",
            flags=copy.deepcopy(flags),
        )
    )
    # 5. 边界融合（完整 SCAP）
    flags["boundary_aware_fusion"] = True
    variants.append(
        ComponentVariant(
            name="full_scap",
            description="+ Full SCAP",
            flags=copy.deepcopy(flags),
        )
    )
    return variants


def build_hyperparam_specs() -> List[HyperParamSpec]:
    params = get_ablation_params()
    return [
        HyperParamSpec(
            name="patch_size",
            values=params["patch_size"],
            description="补丁尺寸",
        ),
        HyperParamSpec(
            name="patch_k",
            values=params["patch_k"],
            description="补丁数量",
        ),
        HyperParamSpec(
            name="steps",
            values=params["steps"],
            description="优化步数",
        ),
        HyperParamSpec(
            name="eps",
            values=params["eps"],
            description="扰动上限",
        ),
        HyperParamSpec(
            name="alpha",
            values=params["alpha"],
            description="步长",
        ),
        HyperParamSpec(
            name="cam_percentile",
            values=params["cam_percentile"],
            description="CAM 阈值",
        ),
    ]


def parse_args():
    parser = argparse.ArgumentParser(description="SCAP 消融与超参数敏感性实验 V2")
    parser.add_argument(
        "--data-root",
        type=str,
        default=os.environ.get("IMAGENET_VAL_ROOT", "/home/jyb/0code/Data/ImageNet/val"),
        help="ImageNet 验证集路径",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="DataLoader batch size")
    parser.add_argument("--max-samples", type=int, default=500, help="实验采样数量")
    parser.add_argument("--seed", type=int, default=3407, help="随机种子")
    parser.add_argument("--device", type=str, default="cuda", help="运行设备，例如 cuda 或 cpu")
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="是否保存攻击中间结果（默认关闭以节省空间）",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_global_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    rm = ResultManager.get_instance()
    rm.set_experiment("SCAP_Ablation_V2")
    rm.enable_image_saving(args.save_images)

    rm.log("加载 ResNet-50 权重...")
    classifier = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).to(device).eval()

    rm.log("构建验证集子集...")
    dataloader = build_validation_subset(
        data_root=args.data_root,
        batch_size=args.batch_size,
        total_samples=args.max_samples,
        seed=args.seed,
    )
    cached_batches = cache_batches(dataloader, args.max_samples, device)
    rm.log(f"缓存完毕，共 {sum(imgs.size(0) for imgs, _ in cached_batches)} 张图片")

    base_config = copy.deepcopy(get_config("optimized"))
    base_config["targeted_attack"] = True

    target_class = get_target_class("default")

    rm.set_test("component_ablation_v2")
    component_runner = ComponentAblationExperiment(
        cached_batches=cached_batches,
        classifier=classifier,
        lr_device=device,
        base_config=base_config,
        target_class=target_class,
        max_samples=args.max_samples,
    )
    component_results = component_runner.run(build_component_variants())

    rm.set_test("hyperparameter_sensitivity_v2")
    hyper_runner = HyperparameterSensitivityExperiment(
        cached_batches=cached_batches,
        classifier=classifier,
        device=device,
        base_config=base_config,
        target_class=target_class,
        max_samples=args.max_samples,
    )
    hyper_results = hyper_runner.run(build_hyperparam_specs())

    summary = {
        "component_table": [row["table_row"] for row in component_results],
        "component_details": component_results,
        "hyperparameter_sweeps": hyper_results,
    }

    output_path = rm.save_json(summary, filename_prefix="ablation_v2_results")
    rm.log(f"结果已保存：{output_path}")


if __name__ == "__main__":
    main()

