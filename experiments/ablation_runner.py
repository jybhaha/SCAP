# configs/ablation_configs.py

ABLATION_CONFIGS = {
    # 1. 补丁大小消融
    "patch_size": {
        "small": {"patch_size": (16, 16)},
        "medium": {"patch_size": (24, 24)},  # 默认
        "large": {"patch_size": (32, 32)},
        "xlarge": {"patch_size": (40, 40)},
    },
    
    # 2. 补丁数量消融
    "patch_k": {
        "single": {"patch_k": 1},
        "three": {"patch_k": 3},  # 默认
        "five": {"patch_k": 5},
        "seven": {"patch_k": 7},
    },
    
    # 3. 位置选择策略消融
    "position_strategy": {
        "full": {  # 完整策略
            "use_region_mask": True,
            "use_cam_mask": True,
            "cam_thresh_ratio": 0.3,
            "mask_thresh_ratio": 0.2
        },
        "region_only": {  # 仅使用区域分割
            "use_region_mask": True,
            "use_cam_mask": False,
            "cam_thresh_ratio": 0.0,
            "mask_thresh_ratio": 0.2
        },
        "cam_only": {  # 仅使用CAM
            "use_region_mask": False,
            "use_cam_mask": True,
            "cam_thresh_ratio": 0.3,
            "mask_thresh_ratio": 0.0
        },
        "random": {  # 随机位置
            "use_region_mask": False,
            "use_cam_mask": False,
            "cam_thresh_ratio": 0.0,
            "mask_thresh_ratio": 0.0
        }
    },
    
    # 4. 补丁生成策略消融
    "generation_strategy": {
        "semantic_aware": {  # 语义感知（完整）
            "use_region_semantics": True,
            "use_color_info": True,
            "use_target_class": True
        },
        "color_only": {  # 仅颜色信息
            "use_region_semantics": False,
            "use_color_info": True,
            "use_target_class": False
        },
        "semantic_only": {  # 仅语义信息
            "use_region_semantics": True,
            "use_color_info": False,
            "use_target_class": False
        },
        "random_patches": {  # 随机补丁
            "use_region_semantics": False,
            "use_color_info": False,
            "use_target_class": False
        }
    },
    
    # 5. 优化参数消融
    "optimization": {
        "weak": {"steps": 10, "eps": 8/255, "alpha": 4/255},
        "medium": {"steps": 25, "eps": 16/255, "alpha": 8/255},  # 默认
        "strong": {"steps": 50, "eps": 24/255, "alpha": 12/255},
        "aggressive": {"steps": 100, "eps": 32/255, "alpha": 16/255},
    },
    
    # 6. CAM参数消融
    "cam_params": {
        "low_sensitivity": {"cam_percentile": 20, "cam_thresh_ratio": 0.2},
        "medium_sensitivity": {"cam_percentile": 30, "cam_thresh_ratio": 0.3},  # 默认
        "high_sensitivity": {"cam_percentile": 40, "cam_thresh_ratio": 0.4},
        "very_high_sensitivity": {"cam_percentile": 50, "cam_thresh_ratio": 0.5},
    }
}

def get_ablation_config(category, variant):
    """获取特定消融实验配置"""
    base_config = {
        "patch_size": (24, 24),
        "blend_width": 5,
        "patch_k": 3,
        "cam_model_name": "resnet50", 
        "cam_percentile": 30,
        "topk_cam_regions": 1,
        "eval_topk": 1,
        "targeted_attack": False,
        "cam_thresh_ratio": 0.3,
        "mask_thresh_ratio": 0.2,
        "steps": 25,
        "eps": 16/255,
        "alpha": 8/255,
        "use_region_mask": True,
        "use_cam_mask": True,
        "use_region_semantics": True,
        "use_color_info": True,
        "use_target_class": False
    }
    
    if category in ABLATION_CONFIGS and variant in ABLATION_CONFIGS[category]:
        ablation_config = ABLATION_CONFIGS[category][variant]
        return {**base_config, **ablation_config}
    else:
        raise ValueError(f"未知的消融配置: {category}.{variant}")


# ablation_runner.py

import torch
import os
import json
import numpy as np
from datetime import datetime
from tqdm import tqdm
from copy import deepcopy

from models.my_patch_attack import MyPatchAttack_Eva

from utils.result_saver import ResultManager


from utils.perceptual_evaluator import PerceptualEvaluator  # 使用你现有的类


class AblationExperimentRunner:
    """使用现有PerceptualEvaluator的消融实验运行器"""
    
    def __init__(self, device='cuda'):
        self.device = torch.device(device)
        self.rm = ResultManager.get_instance()
        self._init_model()
        self._init_evaluators()
        
    def _init_model(self):
        """初始化分类模型"""
        from torchvision import models
        self.classifier_model = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1
        ).eval().to(self.device)
        
    def _init_evaluators(self):
        """初始化评估器"""
        from utils.asr_evaluator import ASREvaluator
        
        self.asr_evaluator = ASREvaluator(model=self.classifier_model, device=self.device)
        self.perceptual_evaluator = PerceptualEvaluator(device=self.device)
        
    def _ensure_device_consistency(self, *tensors):
        """确保所有张量在同一个设备上"""
        processed_tensors = []
        for tensor in tensors:
            if isinstance(tensor, torch.Tensor):
                processed_tensors.append(tensor.to(self.device))
            else:
                processed_tensors.append(tensor)
        return processed_tensors if len(processed_tensors) > 1 else processed_tensors[0]
    
    def _process_attack_result(self, adv_result):
        """处理攻击结果，确保设备一致性"""
        try:
            # 如果结果是元组或列表，取第一个元素
            if isinstance(adv_result, (tuple, list)):
                adv_img = adv_result[0]
            else:
                adv_img = adv_result
            
            # 确保是张量
            if not isinstance(adv_img, torch.Tensor):
                raise ValueError(f"攻击结果不是张量: {type(adv_img)}")
            
            # 确保是4D张量 [batch, channels, height, width]
            if adv_img.dim() == 3:
                adv_img = adv_img.unsqueeze(0)
            elif adv_img.dim() != 4:
                raise ValueError(f"张量维度不正确: {adv_img.dim()}, 期望4D")
            
            # 确保在正确的设备上
            adv_img = adv_img.to(self.device)
            
            # 确保数据类型正确
            if adv_img.dtype != torch.float32:
                adv_img = adv_img.float()
            
            return adv_img
            
        except Exception as e:
            self.rm.log(f"处理攻击结果失败: {e}")
            raise

    def run_comprehensive_ablation_study(self, dataloader, num_samples=1000, save_adv_images=False):
        """
        运行全面的消融实验研究
        """
        self.rm.log("开始运行全面消融实验研究")
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "total_samples": num_samples,
            "ablation_results": {},
            "summary": {}
        }
        
        # 1. 运行基准测试（默认配置）
        self.rm.log("=== 运行基准测试 ===")
        baseline_results = self._run_baseline_experiment(dataloader, num_samples, save_adv_images)
        results["baseline"] = baseline_results
        
        # 2. 运行各个类别的消融实验
        for category in ABLATION_CONFIGS.keys():
            self.rm.log(f"=== 运行 {category} 消融实验 ===")
            category_results = self._run_category_ablation(
                dataloader, category, num_samples, save_adv_images
            )
            results["ablation_results"][category] = category_results
            
        # 3. 生成综合分析
        self.rm.log("=== 生成综合分析 ===")
        results["summary"] = self._generate_comprehensive_analysis(results)
        
        # 4. 保存结果
        self._save_ablation_results(results)
        
        self.rm.log("全面消融实验研究完成！")
        return results
    
    def _run_baseline_experiment(self, dataloader, num_samples, save_adv_images):
        """运行基准实验（默认配置）"""
        baseline_config = get_ablation_config("patch_size", "medium")
        
        attack = MyPatchAttack_Eva(
            model=self.classifier_model,
            device=self.device,
            config=baseline_config,
            debug=False,
            save_imgs=save_adv_images,
            save_attack_process=False
        )
        
        return self._evaluate_config_performance(
            attack, dataloader, num_samples, "baseline", save_adv_images
        )
    
    def _run_category_ablation(self, dataloader, category, num_samples, save_adv_images):
        """运行特定类别的消融实验"""
        category_results = {}
        
        for variant in ABLATION_CONFIGS[category].keys():
            self.rm.log(f"测试 {category}.{variant}")
            
            # 获取配置
            config = get_ablation_config(category, variant)
            
            # 创建攻击器
            attack = MyPatchAttack_Eva(
                model=self.classifier_model,
                device=self.device,
                config=config,
                debug=False,
                save_imgs=save_adv_images,
                save_attack_process=False
            )
            
            # 评估性能
            variant_results = self._evaluate_config_performance(
                attack, dataloader, num_samples, f"{category}_{variant}", save_adv_images
            )
            
            category_results[variant] = variant_results
            
        return category_results
    
    def _evaluate_config_performance(self, attack, dataloader, num_samples, config_name, save_adv_images):
        """评估特定配置的性能"""
        results = {
            "asr": 0.0,
            "confidence_before": [],
            "confidence_after": [],
            "perturbation_metrics": {},
            "convergence_steps": [],
            "successful_attacks": 0,
            "total_samples": 0,
            "time_per_sample": 0
        }
        
        total_samples = 0
        successful_attacks = 0
        confidence_before = []
        confidence_after = []
        convergence_steps = []
        perceptual_metrics_list = []
        
        start_time = datetime.now()
        
        pbar = tqdm(total=min(num_samples, len(dataloader.dataset)), desc=f"评估 {config_name}")
        
        for batch_idx, (images, labels) in enumerate(dataloader):
            if total_samples >= num_samples:
                break
                
            # 确保数据在正确的设备上
            images, labels = self._ensure_device_consistency(images, labels)
            
            for i in range(images.size(0)):
                if total_samples >= num_samples:
                    break
                    
                img = images[i:i+1]
                label = labels[i:i+1]
                
                try:
                    # 获取原始预测
                    with torch.no_grad():
                        logits_clean = self.classifier_model(img)
                        pred_clean = logits_clean.argmax(dim=1)
                        conf_clean = torch.softmax(logits_clean, dim=1).max(dim=1)[0]
                        confidence_before.append(conf_clean.item())
                    
                    # 生成对抗样本
                    adv_img, step_preds = attack.run(img, label, sample_idx=total_samples)
                    
                    # 确保对抗样本在正确的设备上
                    adv_img = self._process_attack_result(adv_img)
                    
                    # 确保step_preds中的张量也在正确设备上（如果是张量的话）
                    if step_preds and isinstance(step_preds[0], torch.Tensor):
                        step_preds = [pred.to(self.device) for pred in step_preds]
                    
                    # 计算收敛步数
                    conv_step = self._find_convergence_step(step_preds, label, attack.target)
                    convergence_steps.append(conv_step)
                    
                    # 测试攻击效果
                    with torch.no_grad():
                        logits_adv = self.classifier_model(adv_img)
                        pred_adv = logits_adv.argmax(dim=1)
                        conf_adv = torch.softmax(logits_adv, dim=1).max(dim=1)[0]
                        confidence_after.append(conf_adv.item())
                    
                    # 统计攻击成功率
                    if (attack.target and pred_adv == label) or (not attack.target and pred_adv != label):
                        successful_attacks += 1
                    
                    # 计算感知质量指标 - 使用你现有的PerceptualEvaluator
                    perceptual_metrics = self.perceptual_evaluator.evaluate(
                        img, adv_img, 
                        include_fid=False,  # FID需要大量样本，单个样本不适合
                        include_no_reference=False  # 无参考指标可选
                    )
                    perceptual_metrics_list.append(perceptual_metrics)
                    
                    # 保存对抗样本（如果启用）
                    if save_adv_images:
                        self._save_adv_sample(img, adv_img, label, pred_adv, config_name, total_samples)
                    
                    total_samples += 1
                    pbar.update(1)
                    
                    # 更新进度
                    current_asr = successful_attacks / total_samples
                    pbar.set_postfix({
                        'ASR': f'{current_asr:.2%}',
                        'L2': f'{perceptual_metrics.get("L2", 0):.4f}',
                        'PSNR': f'{perceptual_metrics.get("PSNR", 0):.2f}'
                    })
                    
                except Exception as e:
                    self.rm.log(f"样本 {total_samples} 处理失败: {e}")
                    continue
        
        pbar.close()
        
        end_time = datetime.now()
        time_per_sample = (end_time - start_time).total_seconds() / total_samples if total_samples > 0 else 0
        
        # 计算最终结果
        results["asr"] = successful_attacks / total_samples if total_samples > 0 else 0.0
        results["confidence_before"] = confidence_before
        results["confidence_after"] = confidence_after
        results["successful_attacks"] = successful_attacks
        results["total_samples"] = total_samples
        results["time_per_sample"] = time_per_sample
        results["convergence_steps"] = convergence_steps
        
        # 计算感知指标的平均值
        if perceptual_metrics_list:
            avg_perceptual_metrics = {}
            for metric_name in perceptual_metrics_list[0].keys():
                values = [metrics[metric_name] for metrics in perceptual_metrics_list]
                avg_perceptual_metrics[metric_name] = {
                    "mean": np.mean(values),
                    "std": np.std(values)
                }
            results["perceptual_metrics"] = avg_perceptual_metrics
        
        # 收敛统计
        if convergence_steps:
            results["convergence_stats"] = {
                "mean_steps": np.mean(convergence_steps),
                "std_steps": np.std(convergence_steps),
                "median_steps": np.median(convergence_steps),
                "max_steps": np.max(convergence_steps)
            }
        
        self.rm.log(f"配置 {config_name} 完成: ASR={results['asr']:.2%}, "
                   f"平均PSNR={results['perceptual_metrics']['PSNR']['mean']:.2f}dB, "
                   f"平均SSIM={results['perceptual_metrics']['SSIM']['mean']:.4f}")
        
        return results
    
    def _find_convergence_step(self, step_preds, true_label, targeted):
        """找到攻击收敛的步数"""
        if not step_preds:
            return 0
            
        # 确保标签在正确的设备上（如果是张量）
        if isinstance(true_label, torch.Tensor):
            true_label = true_label.to(self.device)
        
        for step, pred in enumerate(step_preds):
            # 确保预测在正确的设备上
            if isinstance(pred, torch.Tensor):
                pred = pred.to(self.device)
                
            if targeted:
                # 目标攻击：预测等于目标标签
                if torch.is_tensor(pred) and torch.is_tensor(true_label):
                    if (pred == true_label).all():
                        return step
                else:
                    if pred == true_label:
                        return step
            else:
                # 非目标攻击：预测不等于真实标签
                if torch.is_tensor(pred) and torch.is_tensor(true_label):
                    if (pred != true_label).all():
                        return step
                else:
                    if pred != true_label:
                        return step
        return len(step_preds) - 1  # 如果没有收敛，返回最后一步
    
    def _save_adv_sample(self, original_img, adv_img, true_label, pred_label, config_name, sample_idx):
        """保存对抗样本"""
        try:
            import torchvision.utils as vutils
            
            # 创建保存目录
            save_dir = os.path.join(self.rm.img_dir, "ablation_study", config_name)
            os.makedirs(save_dir, exist_ok=True)
            
            # 将张量移动到CPU进行保存
            original_img_cpu = original_img.cpu()
            adv_img_cpu = adv_img.cpu()
            
            # 保存原始图片
            orig_path = os.path.join(save_dir, f"sample_{sample_idx}_original.png")
            vutils.save_image(original_img_cpu, orig_path)
            
            # 保存对抗样本
            adv_path = os.path.join(save_dir, f"sample_{sample_idx}_adversarial.png")
            vutils.save_image(adv_img_cpu, adv_path)
            
            # 保存差异图
            diff = torch.abs(adv_img_cpu - original_img_cpu)
            diff_path = os.path.join(save_dir, f"sample_{sample_idx}_difference.png")
            vutils.save_image(diff, diff_path)
            
        except Exception as e:
            self.rm.log(f"保存对抗样本失败: {e}")
    
    def _generate_comprehensive_analysis(self, results):
        """生成综合分析"""
        analysis = {
            "best_configurations": {},
            "parameter_importance": {},
            "tradeoff_analysis": {},
            "recommendations": []
        }
        
        baseline_asr = results["baseline"]["asr"]
        
        # 分析每个类别的最佳配置
        for category, category_results in results["ablation_results"].items():
            best_variant = None
            best_asr = 0
            
            for variant, variant_results in category_results.items():
                if variant_results["asr"] > best_asr:
                    best_asr = variant_results["asr"]
                    best_variant = variant
            
            analysis["best_configurations"][category] = {
                "best_variant": best_variant,
                "asr": best_asr,
                "improvement_over_baseline": (best_asr - baseline_asr) / baseline_asr * 100 if baseline_asr > 0 else float('inf')
            }
        
        # 计算参数重要性（ASR范围）
        for category, category_results in results["ablation_results"].items():
            asrs = [variant_results["asr"] for variant_results in category_results.values()]
            if asrs:
                importance = max(asrs) - min(asrs)
                analysis["parameter_importance"][category] = importance
        
        # 权衡分析（ASR vs 感知质量）
        for category, category_results in results["ablation_results"].items():
            tradeoffs = {}
            for variant, variant_results in category_results.items():
                asr = variant_results["asr"]
                psnr_mean = variant_results["perceptual_metrics"]["PSNR"]["mean"]
                ssim_mean = variant_results["perceptual_metrics"]["SSIM"]["mean"]
                lpips_mean = variant_results["perceptual_metrics"]["LPIPS"]["mean"]
                
                tradeoffs[variant] = {
                    "asr": asr,
                    "psnr": psnr_mean,
                    "ssim": ssim_mean,
                    "lpips": lpips_mean,
                    "efficiency_score": asr / (lpips_mean + 1e-8)  # 使用LPIPS作为扰动度量
                }
            
            analysis["tradeoff_analysis"][category] = tradeoffs
        
        # 生成建议
        analysis["recommendations"] = self._generate_recommendations(analysis)
        
        return analysis
    
    def _generate_recommendations(self, analysis):
        """生成配置建议"""
        recommendations = []
        
        # 基于最佳配置的建议
        for category, best_config in analysis["best_configurations"].items():
            recommendations.append(
                f"在{category}类别中，建议使用{best_config['best_variant']}配置 "
                f"(ASR: {best_config['asr']:.2%}, 相对基准提升: {best_config['improvement_over_baseline']:.1f}%)"
            )
        
        # 基于参数重要性的建议
        sorted_importance = sorted(analysis["parameter_importance"].items(), 
                                 key=lambda x: x[1], reverse=True)
        
        if sorted_importance:
            most_important = sorted_importance[0][0]
            recommendations.append(
                f"最重要的参数是{most_important}，对ASR影响最大"
            )
        
        # 基于权衡分析的建议
        best_efficiency = None
        best_score = 0
        
        for category, tradeoffs in analysis["tradeoff_analysis"].items():
            for variant, metrics in tradeoffs.items():
                if metrics["efficiency_score"] > best_score:
                    best_score = metrics["efficiency_score"]
                    best_efficiency = f"{category}.{variant}"
        
        if best_efficiency:
            recommendations.append(
                f"最高效的配置是{best_efficiency} (效率分数: {best_score:.2f})"
            )
        
        return recommendations
    
    def _save_ablation_results(self, results):
        """保存消融实验结果"""
        try:
            # 创建结果目录
            result_dir = os.path.join(self.rm.base_dir, "ablation_results")
            os.makedirs(result_dir, exist_ok=True)
            
            # 保存详细结果
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_file = os.path.join(result_dir, f"ablation_study_{timestamp}.json")
            
            # 转换numpy类型为Python原生类型以便JSON序列化
            def convert_to_serializable(obj):
                if isinstance(obj, (np.integer, np.int64)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float64)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_to_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_serializable(item) for item in obj]
                else:
                    return obj
            
            serializable_results = convert_to_serializable(results)
            
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, indent=2, ensure_ascii=False)
            
            # 保存摘要报告
            summary_file = os.path.join(result_dir, f"ablation_summary_{timestamp}.txt")
            self._save_summary_report(results, summary_file)
            
            self.rm.log(f"消融实验结果已保存到: {result_file}")
            self.rm.log(f"摘要报告已保存到: {summary_file}")
            
        except Exception as e:
            self.rm.log(f"保存结果失败: {e}")
    
    def _save_summary_report(self, results, filename):
        """保存摘要报告"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("SCAR方法消融实验摘要报告\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"实验时间: {results['timestamp']}\n")
            f.write(f"总样本数: {results['total_samples']}\n\n")
            
            # 基准结果
            baseline = results["baseline"]
            f.write("基准配置结果:\n")
            f.write(f"  ASR: {baseline['asr']:.2%}\n")
            f.write(f"  平均PSNR: {baseline['perceptual_metrics']['PSNR']['mean']:.2f} dB\n")
            f.write(f"  平均SSIM: {baseline['perceptual_metrics']['SSIM']['mean']:.4f}\n")
            f.write(f"  平均LPIPS: {baseline['perceptual_metrics']['LPIPS']['mean']:.4f}\n")
            f.write(f"  平均收敛步数: {baseline.get('convergence_stats', {}).get('mean_steps', 0):.1f}\n\n")
            
            # 最佳配置
            f.write("各类别最佳配置:\n")
            for category, best in results["summary"]["best_configurations"].items():
                f.write(f"  {category}: {best['best_variant']} (ASR: {best['asr']:.2%}, "
                       f"提升: {best['improvement_over_baseline']:+.1f}%)\n")
            f.write("\n")
            
            # 参数重要性
            f.write("参数重要性排序:\n")
            sorted_importance = sorted(results["summary"]["parameter_importance"].items(), 
                                    key=lambda x: x[1], reverse=True)
            for category, importance in sorted_importance:
                f.write(f"  {category}: {importance:.4f}\n")
            f.write("\n")
            
            # 建议
            f.write("配置建议:\n")
            for i, recommendation in enumerate(results["summary"]["recommendations"], 1):
                f.write(f"  {i}. {recommendation}\n")


# 便捷函数
def run_ablation_study(dataloader, num_samples=500, save_adv_images=False, device='cuda'):
    """运行消融实验的便捷函数"""
    runner = AblationExperimentRunner(device=device)
    return runner.run_comprehensive_ablation_study(dataloader, num_samples, save_adv_images)