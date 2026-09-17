#!/usr/bin/env python3
"""
实验05：模型迁移性与黑盒攻击能力评估
评估补丁的模型迁移性与黑盒攻击能力（对未见过的架构/权重/服务）

主要功能：
1. 跨模型ASR评估
2. 跨家族ASR评估（Conv→Trans）
"""

import torch
import torch.nn as nn
import torchvision.models as models
import json
import numpy as np
import time
import os
from copy import deepcopy
from utils.result_saver import ResultManager
from experiments.experiment_runner import ExperimentRunner
from data.data_loader import get_experiment_dataloader
from configs.experiment_configs import get_config
from models.my_patch_attack import MyPatchAttack


class TransferabilityEvaluator:
    """模型迁移性评估器"""
    
    def __init__(self, device='cuda'):
        self.device = torch.device(device)
        self.rm = ResultManager.get_instance()
        self._init_models()
        self._init_attackers()
        
    def _init_models(self):
        """初始化各种目标模型 - 选择代表性模型"""
        self.rm.log("Initializing target models for transferability evaluation...")
        
        # 选择代表性的模型，减少数量但保持多样性
        self.all_models = {
            # 传统CNN
            "resnet50": models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1),
            "vgg16": models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1),
            "densenet121": models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1),
            "inception_v3": models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1),
            
            # 现代CNN
            "efficientnet_b0": models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1),
            "mobilenet_v3": models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1),
            "convnext_tiny": models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1),
            
            # Transformer架构
            "vit_b_16": models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1),
            "swin_t": models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1),
        }
        
        # 将所有模型移动到设备并设置为评估模式
        for name, model in self.all_models.items():
            try:
                model.to(self.device)
                model.eval()
                self.rm.log(f"Successfully initialized {name}")
            except Exception as e:
                self.rm.log(f"Failed to initialize {name}: {e}")
                del self.all_models[name]
        
        self.rm.log(f"Successfully initialized {len(self.all_models)} representative models")
        
    def _init_attackers(self):
        """初始化攻击器"""
        self.rm.log("Initializing attackers...")
        
        # 使用ResNet50作为源模型生成补丁
        source_model = self.all_models["resnet50"]
        
        # 初始化SCAR攻击器
        try:
            self.scar_attacker = MyPatchAttack(
                model=source_model,
                device=self.device
            )
            self.rm.log("SCAR attacker initialized successfully")
        except Exception as e:
            self.rm.log(f"Failed to initialize SCAR attacker: {e}")
            self.scar_attacker = None
        
        self.rm.log("Attackers initialized successfully")
    
    def evaluate_cross_model_transferability(self, dataloader, config):
        """评估跨模型迁移性 - 详细到每个源模型对每个目标模型的攻击"""
        self.rm.log("=== Starting Cross-Model Transferability Evaluation ===")
        
        results = {
            "cross_model_detailed": {},  # 详细的跨模型攻击结果
            "model_families": {
                "traditional_cnn": {},
                "modern_cnn": {},
                "transformer": {}
            },
            "overall_statistics": {
                "total_samples": 0,
                "successful_attacks": 0,
                "overall_asr": 0.0
            }
        }
        
        total_samples = 0
        successful_attacks = 0
        
        # 定义源模型（攻击生成模型）
        source_models = {
            "resnet50": self.all_models["resnet50"],
            "vgg16": self.all_models["vgg16"],
            "vit_b_16": self.all_models["vit_b_16"],
        }
        
        # 测试每个源模型对每个目标模型的攻击
        for source_name, source_model in source_models.items():
            self.rm.log(f"Testing attacks from source model: {source_name}")
            results["cross_model_detailed"][source_name] = {}
            
            # 为当前源模型重新初始化攻击器
            try:
                source_attacker = MyPatchAttack(model=source_model, device=self.device)
            except Exception as e:
                self.rm.log(f"Failed to initialize attacker for {source_name}: {e}")
                continue
            
            # 测试所有目标模型
            for target_name, target_model in self.all_models.items():
                if source_name == target_name:
                    continue  # 跳过自己攻击自己
                
                self.rm.log(f"  Testing {source_name} -> {target_name}")
                try:
                    asr, success_count, sample_count = self._test_cross_model_attack(
                        source_attacker, target_model, target_name, dataloader, config
                    )
                    results["cross_model_detailed"][source_name][target_name] = {
                        "asr": asr,
                        "successful_attacks": success_count,
                        "total_samples": sample_count
                    }
                    total_samples += sample_count
                    successful_attacks += success_count
                    
                    # 按模型家族统计
                    family = self._get_model_family(target_name)
                    if family in results["model_families"]:
                        results["model_families"][family][target_name] = {
                            "asr": asr,
                            "successful_attacks": success_count,
                            "total_samples": sample_count
                        }
                            
                except Exception as e:
                    self.rm.log(f"Error testing {source_name} -> {target_name}: {e}")
                    continue
        
        # 计算整体统计
        results["overall_statistics"]["total_samples"] = total_samples
        results["overall_statistics"]["successful_attacks"] = successful_attacks
        results["overall_statistics"]["overall_asr"] = successful_attacks / total_samples if total_samples > 0 else 0.0
        
        return results
    
    def _get_model_family(self, model_name):
        """根据模型名称判断所属家族"""
        traditional_cnn = ["resnet50", "vgg16", "densenet121", "inception_v3"]
        modern_cnn = ["efficientnet_b0", "mobilenet_v3", "convnext_tiny"]
        transformer = ["vit_b_16", "swin_t"]
        
        if model_name in traditional_cnn:
            return "traditional_cnn"
        elif model_name in modern_cnn:
            return "modern_cnn"
        elif model_name in transformer:
            return "transformer"
        else:
            return "unknown"
    
    def _test_cross_model_attack(self, source_attacker, target_model, target_name, dataloader, config):
        """测试跨模型攻击"""
        target_model.eval()
        successful_attacks = 0
        total_samples = 0
        
        max_samples = 30  # 每个模型对测试30个样本
        
        for batch_idx, (images, labels) in enumerate(dataloader):
            if total_samples >= max_samples:
                break
                
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            for i in range(images.size(0)):
                if total_samples >= max_samples:
                    break
                    
                img = images[i:i+1]
                label = labels[i:i+1]
                
                try:
                    # 使用源攻击器生成对抗样本
                    adv_images = source_attacker.run(img, label)
                    
                    # 在目标模型上测试
                    with torch.no_grad():
                        # 原始预测
                        orig_pred = target_model(img.detach())
                        orig_pred_class = torch.argmax(orig_pred, dim=1)
                        
                        # 对抗样本预测
                        adv_pred = target_model(adv_images)
                        adv_pred_class = torch.argmax(adv_pred, dim=1)
                        
                        # 判断攻击是否成功
                        if orig_pred_class != adv_pred_class:
                            successful_attacks += 1
                        
                        total_samples += 1
                    
                except Exception as e:
                    self.rm.log(f"Attack failed for {target_name}: {e}")
                    total_samples += 1  # 仍然计数，但不算成功
                    continue
        
        asr = successful_attacks / total_samples if total_samples > 0 else 0.0
        self.rm.log(f"  {target_name} ASR: {asr:.3f} ({successful_attacks}/{total_samples})")
        
        return asr, successful_attacks, total_samples
    
    def evaluate_cross_family_transferability(self, dataloader, config):
        """评估跨家族迁移性"""
        self.rm.log("=== Starting Cross-Family Transferability Evaluation ===")
        
        results = {
            "cross_family_asr": {},
            "family_statistics": {},
            "transferability_matrix": {}
        }
        
        # 定义模型家族
        families = {
            "Traditional_CNN": ["resnet50", "vgg16", "densenet121", "inception_v3"],
            "Modern_CNN": ["efficientnet_b0", "mobilenet_v3", "convnext_tiny"],
            "Transformer": ["vit_b_16", "swin_t"]
        }
        
        # 计算每个家族的平均ASR
        for family_name, model_names in families.items():
            family_asr, family_success, family_total = self._calculate_family_asr_detailed(model_names, dataloader, config)
            results["family_statistics"][family_name] = {
                "average_asr": family_asr,
                "successful_attacks": family_success,
                "total_samples": family_total,
                "model_count": len([name for name in model_names if name in self.all_models])
            }
        
        # 构建迁移性矩阵（源家族 -> 目标家族）
        source_families = ["Traditional_CNN", "Modern_CNN", "Transformer"]
        
        for source_family in source_families:
            results["cross_family_asr"][source_family] = {}
            results["transferability_matrix"][source_family] = {}
            
            # 为源家族创建攻击器
            source_model_name = families[source_family][0]  # 取家族第一个模型作为代表
            source_model = self.all_models.get(source_model_name)
            if source_model is None:
                continue
                
            try:
                source_attacker = MyPatchAttack(model=source_model, device=self.device)
            except Exception as e:
                self.rm.log(f"Failed to create attacker for {source_family}: {e}")
                continue
            
            for target_family, target_model_names in families.items():
                target_asr, target_success, target_total = self._test_family_attack(
                    source_attacker, target_model_names, dataloader, config
                )
                
                results["cross_family_asr"][source_family][target_family] = target_asr
                results["transferability_matrix"][source_family][target_family] = {
                    "asr": target_asr,
                    "successful_attacks": target_success,
                    "total_samples": target_total
                }
        
        return results
    
    def _calculate_family_asr_detailed(self, model_names, dataloader, config):
        """计算模型家族的详细ASR"""
        total_success = 0
        total_samples = 0
        
        for model_name in model_names:
            model = self.all_models.get(model_name)
            if model is None:
                continue
            
            asr, success_count, sample_count = self._test_single_model(
                model, model_name, dataloader, config
            )
            total_success += success_count
            total_samples += sample_count
        
        family_asr = total_success / total_samples if total_samples > 0 else 0.0
        return family_asr, total_success, total_samples
    
    def _test_family_attack(self, source_attacker, target_model_names, dataloader, config):
        """测试源攻击器对目标家族的攻击"""
        total_success = 0
        total_samples = 0
        
        for model_name in target_model_names:
            target_model = self.all_models.get(model_name)
            if target_model is None:
                continue
            
            asr, success_count, sample_count = self._test_cross_model_attack(
                source_attacker, target_model, model_name, dataloader, config
            )
            total_success += success_count
            total_samples += sample_count
        
        family_asr = total_success / total_samples if total_samples > 0 else 0.0
        return family_asr, total_success, total_samples
    
    def _test_single_model(self, model, model_name, dataloader, config):
        """测试单个模型的攻击成功率"""
        model.eval()
        successful_attacks = 0
        total_samples = 0
        max_samples = 20  # 限制每个模型的测试样本数
        
        for batch_idx, (images, labels) in enumerate(dataloader):
            if total_samples >= max_samples:
                break
                
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            for i in range(images.size(0)):
                if total_samples >= max_samples:
                    break
                    
                img = images[i:i+1]
                label = labels[i:i+1]
                
                try:
                    adv_images = self.scar_attacker.run(img, label)
                    
                    with torch.no_grad():
                        orig_pred = model(img.detach())
                        orig_pred_class = torch.argmax(orig_pred, dim=1)
                        
                        adv_pred = model(adv_images)
                        adv_pred_class = torch.argmax(adv_pred, dim=1)
                        
                        if orig_pred_class != adv_pred_class:
                            successful_attacks += 1
                        
                        total_samples += 1
                    
                except Exception as e:
                    self.rm.log(f"Attack failed for {model_name}: {e}")
                    total_samples += 1
                    continue
        
        asr = successful_attacks / total_samples if total_samples > 0 else 0.0
        return asr, successful_attacks, total_samples
    
    def generate_transferability_report(self, results):
        """生成迁移性评估报告 - 保存到rm.log_dir目录"""
        # 使用rm.log_dir作为保存目录
        save_dir = os.path.join(self.rm.log_dir, "experiment_05_transferability")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # 保存详细结果
        results_file = os.path.join(save_dir, "transferability_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 生成Markdown报告
        report = self._generate_markdown_report(results)
        report_file = os.path.join(save_dir, "transferability_report.md")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        # 生成LaTeX表格
        latex_table = self._generate_latex_table(results)
        latex_file = os.path.join(save_dir, "transferability_table.tex")
        with open(latex_file, 'w', encoding='utf-8') as f:
            f.write(latex_table)
        
        self.rm.log(f"Transferability evaluation report saved to: {save_dir}")
        return save_dir
    
    def _generate_markdown_report(self, results):
        """生成Markdown格式的报告"""
        report = "# Model Transferability Evaluation Report\n\n"
        report += f"**Evaluation Time**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # 1. 跨模型迁移性结果
        if "cross_model_detailed" in results.get("cross_model_transferability", {}):
            report += "## 1. Cross-Model Transferability Results\n\n"
            
            cross_model = results["cross_model_transferability"]
            for source_name, target_results in cross_model["cross_model_detailed"].items():
                report += f"### Attacks from {source_name}\n\n"
                report += "| Target Model | ASR | Successful Attacks | Total Samples |\n"
                report += "|--------------|-----|-------------------|---------------|\n"
                
                for target_name, metrics in target_results.items():
                    report += f"| {target_name} | {metrics['asr']:.3f} | {metrics['successful_attacks']} | {metrics['total_samples']} |\n"
                report += "\n"
            
            # 添加总体统计
            if "overall_statistics" in cross_model:
                overall = cross_model["overall_statistics"]
                report += "### Overall Statistics\n\n"
                report += f"- **Total Samples Tested**: {overall['total_samples']}\n"
                report += f"- **Successful Attacks**: {overall['successful_attacks']}\n"
                report += f"- **Overall ASR**: {overall['overall_asr']:.3f}\n\n"
        
        # 2. 跨家族迁移性结果
        if "family_statistics" in results.get("cross_family_transferability", {}):
            report += "## 2. Cross-Family Transferability Results\n\n"
            
            cross_family = results["cross_family_transferability"]
            report += "### Family Statistics\n"
            report += "| Family | Average ASR | Model Count | Successful Attacks | Total Samples |\n"
            report += "|--------|-------------|-------------|-------------------|---------------|\n"
            
            for family_name, stats in cross_family["family_statistics"].items():
                report += f"| {family_name} | {stats['average_asr']:.3f} | {stats['model_count']} | {stats['successful_attacks']} | {stats['total_samples']} |\n"
            report += "\n"
            
        if "cross_family_asr" in results.get("cross_family_transferability", {}):
            cross_family = results["cross_family_transferability"]
            report += "### Cross-Family Attack Matrix\n\n"
            report += "| Source Family → Target Family | "
            target_families = list(next(iter(cross_family["cross_family_asr"].values())).keys())
            for target_family in target_families:
                report += f"{target_family} | "
            report += "\n|" + "|".join(["---"] * (len(target_families) + 1)) + "|\n"
            
            for source_family, target_results in cross_family["cross_family_asr"].items():
                report += f"| {source_family} | "
                for target_family in target_families:
                    asr = target_results.get(target_family, 0)
                    report += f"{asr:.3f} | "
                report += "\n"
        
        return report
    
    def _generate_latex_table(self, results):
        """生成LaTeX格式的表格"""
        latex = """\\begin{table}[htbp]
\\centering
\\caption{Cross-Model Transferability Attack Success Rates}
\\label{tab:transferability_asr}
\\begin{tabular}{lccc}
\\toprule
Source Model & Target Model & ASR & Samples \\\\
\\midrule
"""
        
        cross_model = results.get("cross_model_transferability", {})
        if "cross_model_detailed" in cross_model:
            for source_name, target_results in cross_model["cross_model_detailed"].items():
                for target_name, metrics in list(target_results.items())[:3]:  # 只显示前3个
                    latex += f"{source_name} & {target_name} & {metrics['asr']:.3f} & {metrics['total_samples']} \\\\\n"
        
        latex += """\\bottomrule
\\end{tabular}
\\end{table}"""
        
        return latex


def run_transferability_experiment(dataloader_mode="small"):
    """
    运行模型迁移性与黑盒攻击能力评估实验
    """
    rm = ResultManager.get_instance()
    rm.set_experiment("TransferabilityEvaluation")
    rm.set_test("cross_model_transferability")
    
    # 获取数据加载器
    dataloader = get_experiment_dataloader(dataloader_mode)
    
    # 获取配置
    config = get_config("default")
    
    # 初始化评估器
    evaluator = TransferabilityEvaluator()
    
    # 1. 跨模型迁移性评估
    rm.log("=== Starting Cross-Model Transferability Evaluation ===")
    try:
        cross_model_results = evaluator.evaluate_cross_model_transferability(dataloader, config)
        rm.log("Cross-model evaluation completed successfully")
    except Exception as e:
        rm.log(f"Cross-model evaluation failed: {e}")
        cross_model_results = {}
    
    # 2. 跨家族迁移性评估
    rm.log("=== Starting Cross-Family Transferability Evaluation ===")
    try:
        cross_family_results = evaluator.evaluate_cross_family_transferability(dataloader, config)
        rm.log("Cross-family evaluation completed successfully")
    except Exception as e:
        rm.log(f"Cross-family evaluation failed: {e}")
        cross_family_results = {}
    
    # 合并结果
    all_results = {
        "cross_model_transferability": cross_model_results,
        "cross_family_transferability": cross_family_results
    }
    
    # 生成报告
    try:
        save_dir = evaluator.generate_transferability_report(all_results)
        rm.log(f"Report generation completed successfully, saved to: {save_dir}")
    except Exception as e:
        rm.log(f"Report generation failed: {e}")
    
    # 输出结果摘要
    rm.log("=== Transferability Evaluation Results Summary ===")
    
    # 打印跨模型结果
    if cross_model_results and "cross_model_detailed" in cross_model_results:
        rm.log("Cross-Model Results:")
        for source_name, target_results in cross_model_results["cross_model_detailed"].items():
            rm.log(f"  Attacks from {source_name}:")
            for target_name, metrics in list(target_results.items())[:3]:
                rm.log(f"    -> {target_name}: ASR = {metrics['asr']:.3f}")
        
        if "overall_statistics" in cross_model_results:
            overall = cross_model_results["overall_statistics"]
            rm.log(f"  Overall Cross-Model ASR: {overall['overall_asr']:.3f}")
    
    # 打印跨家族结果
    if cross_family_results and "family_statistics" in cross_family_results:
        rm.log("Cross-Family Results:")
        for family_name, stats in cross_family_results["family_statistics"].items():
            rm.log(f"  {family_name}: Average ASR = {stats['average_asr']:.3f}")
    
    rm.log("Transferability evaluation experiment completed!")
    
    return all_results


if __name__ == '__main__':
    run_transferability_experiment(dataloader_mode="small")