#!/usr/bin/env python3
"""
实验06：现实世界效果评估
基于real_world_visualizer.py设计在现实世界中的效果评估

主要功能：
1. 物理环境模拟（光照、角度、距离）
2. 感知质量评估（SSIM、PSNR、LPIPS）
3. 攻击成功率统计
4. 现实世界条件影响分析
5. 综合可视化报告
"""

import torch
import numpy as np
import cv2
import time
import os
from PIL import Image, ImageEnhance, ImageFilter
from utils.result_saver import ResultManager
from experiments.experiment_runner import ExperimentRunner
from data.data_loader import get_experiment_dataloader
from configs.experiment_configs import get_config
from models.my_patch_attack import MyPatchAttack
from utils.real_world_visualizer import RealWorldVisualizer
import matplotlib.pyplot as plt


class RealWorldEffectivenessEvaluator:
    """现实世界效果评估器"""
    
    def __init__(self, device='cuda'):
        self.device = torch.device(device)
        self.rm = ResultManager.get_instance()
        self._init_models()
        self._init_visualizer()
        
    def _init_models(self):
        """初始化模型"""
        self.rm.log("Initializing models for real-world evaluation...")
        
        # 初始化分类器
        self.classifier = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        self.classifier.to(self.device)
        self.classifier.eval()
        
        # 初始化攻击器 - 修复参数问题
        try:
            self.attacker = MyPatchAttack(
                model=self.classifier,
                device=self.device
            )
            self.rm.log("Attacker initialized successfully")
        except Exception as e:
            self.rm.log(f"Failed to initialize attacker: {e}")
            # 尝试其他初始化方式
            try:
                self.attacker = MyPatchAttack(self.classifier)
                self.rm.log("Attacker initialized with alternative method")
            except Exception as e2:
                self.rm.log(f"Alternative attacker initialization also failed: {e2}")
                self.attacker = None
        
        self.rm.log("Models initialized successfully")
    
    def _init_visualizer(self):
        """初始化可视化器"""
        self.visualizer = RealWorldVisualizer("Results/experiment_06_real_world")
    
    def simulate_physical_conditions(self, image, condition_type, intensity=0.5):
        """模拟物理环境条件"""
        try:
            # 转换为PIL图像
            if isinstance(image, torch.Tensor):
                if image.dim() == 4:
                    image = image.squeeze(0)
                image = image.permute(1, 2, 0).cpu().numpy()
                image = (image * 255).astype(np.uint8)
                image = Image.fromarray(image)
            
            if condition_type == "brightness":
                # 模拟光照变化
                enhancer = ImageEnhance.Brightness(image)
                return enhancer.enhance(1.0 + intensity * 0.5)  # 0.5-1.5倍亮度
            
            elif condition_type == "contrast":
                # 模拟对比度变化
                enhancer = ImageEnhance.Contrast(image)
                return enhancer.enhance(1.0 + intensity * 0.3)  # 0.7-1.3倍对比度
            
            elif condition_type == "blur":
                # 模拟距离/聚焦变化
                blur_radius = int(intensity * 3)  # 0-3像素模糊
                return image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            
            elif condition_type == "noise":
                # 模拟传感器噪声
                img_array = np.array(image)
                noise = np.random.normal(0, intensity * 25, img_array.shape)
                noisy_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)
                return Image.fromarray(noisy_img)
            
            elif condition_type == "rotation":
                # 模拟角度变化
                angle = intensity * 15  # 0-15度旋转
                return image.rotate(angle, fillcolor=(255, 255, 255))
            
            elif condition_type == "compression":
                # 模拟压缩失真
                from io import BytesIO
                buffer = BytesIO()
                quality = int(100 - intensity * 50)  # 50-100质量
                image.save(buffer, format='JPEG', quality=quality)
                buffer.seek(0)
                return Image.open(buffer)
            
            else:
                return image
                
        except Exception as e:
            self.rm.log(f"Failed to simulate {condition_type}: {e}")
            return image
    
    def calculate_perceptual_metrics(self, original_img, modified_img):
        """计算感知质量指标"""
        try:
            # 转换为numpy数组
            if isinstance(original_img, torch.Tensor):
                original_img = original_img.permute(1, 2, 0).cpu().numpy()
            if isinstance(modified_img, torch.Tensor):
                modified_img = modified_img.permute(1, 2, 0).cpu().numpy()
            
            # 确保值在[0,1]范围内
            original_img = np.clip(original_img, 0, 1)
            modified_img = np.clip(modified_img, 0, 1)
            
            # 计算SSIM
            ssim_score = self._calculate_ssim(original_img, modified_img)
            
            # 计算PSNR
            psnr_score = self._calculate_psnr(original_img, modified_img)
            
            # 计算LPIPS（简化版本）
            lpips_score = self._calculate_lpips(original_img, modified_img)
            
            return {
                "ssim": ssim_score,
                "psnr": psnr_score,
                "lpips": lpips_score
            }
            
        except Exception as e:
            self.rm.log(f"Failed to calculate perceptual metrics: {e}")
            return {"ssim": 0.0, "psnr": 0.0, "lpips": 0.0}
    
    def _calculate_ssim(self, img1, img2):
        """计算SSIM"""
        try:
            # 转换为灰度图
            if len(img1.shape) == 3:
                img1 = cv2.cvtColor((img1 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
                img2 = cv2.cvtColor((img2 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            else:
                img1 = (img1 * 255).astype(np.uint8)
                img2 = (img2 * 255).astype(np.uint8)
            
            # 计算SSIM
            from skimage.metrics import structural_similarity
            ssim = structural_similarity(img1, img2)
            return ssim
            
        except Exception as e:
            self.rm.log(f"SSIM calculation failed: {e}")
            return 0.0
    
    def _calculate_psnr(self, img1, img2):
        """计算PSNR"""
        try:
            # 转换为灰度图
            if len(img1.shape) == 3:
                img1 = cv2.cvtColor((img1 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
                img2 = cv2.cvtColor((img2 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            else:
                img1 = (img1 * 255).astype(np.uint8)
                img2 = (img2 * 255).astype(np.uint8)
            
            # 计算MSE
            mse = np.mean((img1 - img2) ** 2)
            if mse == 0:
                return float('inf')
            
            # 计算PSNR
            psnr = 20 * np.log10(255.0 / np.sqrt(mse))
            return psnr
            
        except Exception as e:
            self.rm.log(f"PSNR calculation failed: {e}")
            return 0.0
    
    def _calculate_lpips(self, img1, img2):
        """计算LPIPS（简化版本）"""
        try:
            # 简化的LPIPS计算，使用L2距离
            diff = np.mean((img1 - img2) ** 2)
            return np.sqrt(diff)
            
        except Exception as e:
            self.rm.log(f"LPIPS calculation failed: {e}")
            return 0.0
    
    def evaluate_real_world_effectiveness(self, dataloader, config):
        """评估现实世界效果"""
        self.rm.log("=== Starting Real-World Effectiveness Evaluation ===")
        
        # 定义物理条件类型
        condition_types = [
            "brightness", "contrast", "blur", "noise", 
            "rotation", "compression"
        ]
        
        # 定义强度级别
        intensity_levels = [0.2, 0.4, 0.6, 0.8, 1.0]
        
        results = {
            "attack_success_rates": [],
            "condition_types": [],
            "perceptual_qualities": {
                "ssim": [],
                "psnr": [],
                "lpips": []
            },
            "intensity_levels": [],
            "detailed_results": []
        }
        
        total_samples = 0
        successful_attacks = 0
        
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            for i in range(images.size(0)):
                img = images[i:i+1]
                label = labels[i:i+1]
                total_samples += 1
                
                # 确保输入张量需要梯度
                img.requires_grad_(True)
                
                # 生成对抗样本
                try:
                    adv_img = self.attacker.run(img, label)
                except Exception as e:
                    self.rm.log(f"Failed to generate adversarial samples: {e}")
                    continue
                
                # 测试不同物理条件
                for condition_type in condition_types:
                    for intensity in intensity_levels:
                        # 模拟物理条件
                        modified_adv_img = self.simulate_physical_conditions(
                            adv_img, condition_type, intensity
                        )
                        
                        # 转换为张量
                        if isinstance(modified_adv_img, Image.Image):
                            modified_adv_img = torch.from_numpy(
                                np.array(modified_adv_img)
                            ).permute(2, 0, 1).float() / 255.0
                            modified_adv_img = modified_adv_img.unsqueeze(0).to(self.device)
                        
                        # 在分类器上测试
                        with torch.no_grad():
                            orig_pred = self.classifier(img.detach())
                            orig_pred_class = torch.argmax(orig_pred, dim=1)
                            
                            adv_pred = self.classifier(modified_adv_img)
                            adv_pred_class = torch.argmax(adv_pred, dim=1)
                            
                            # 判断攻击是否成功
                            attack_success = orig_pred_class != adv_pred_class
                            if attack_success:
                                successful_attacks += 1
                            
                            # 计算感知质量指标
                            perceptual_metrics = self.calculate_perceptual_metrics(
                                img.detach(), modified_adv_img
                            )
                            
                            # 记录结果
                            results["attack_success_rates"].append(float(attack_success))
                            results["condition_types"].append(condition_type)
                            results["perceptual_qualities"]["ssim"].append(perceptual_metrics["ssim"])
                            results["perceptual_qualities"]["psnr"].append(perceptual_metrics["psnr"])
                            results["perceptual_qualities"]["lpips"].append(perceptual_metrics["lpips"])
                            results["intensity_levels"].append(intensity)
                            
                            # 详细结果
                            results["detailed_results"].append({
                                "sample_id": total_samples,
                                "condition_type": condition_type,
                                "intensity": intensity,
                                "attack_success": attack_success,
                                "perceptual_metrics": perceptual_metrics,
                                "original_class": orig_pred_class.item(),
                                "adversarial_class": adv_pred_class.item()
                            })
                
                if total_samples % 10 == 0:
                    current_asr = successful_attacks / (total_samples * len(condition_types) * len(intensity_levels))
                    self.rm.log(f"Processed {total_samples} samples, Current ASR: {current_asr:.3f}")
        
        # 计算整体统计
        overall_asr = successful_attacks / (total_samples * len(condition_types) * len(intensity_levels))
        results["overall_statistics"] = {
            "total_samples": total_samples,
            "successful_attacks": successful_attacks,
            "overall_asr": overall_asr,
            "condition_types_count": len(condition_types),
            "intensity_levels_count": len(intensity_levels)
        }
        
        return results
    
    def generate_comprehensive_analysis(self, results):
        """生成综合分析"""
        self.rm.log("=== Generating Comprehensive Analysis ===")
        
        # 使用可视化器生成各种图表
        self.visualizer.plot_attack_success_rates(results)
        self.visualizer.plot_perceptual_quality_analysis(results)
        self.visualizer.plot_condition_impact_analysis(results)
        self.visualizer.create_comprehensive_report(results)
        
        # 保存CSV结果
        self.visualizer.save_results_to_csv(results)
        
        # 生成条件影响分析
        self._analyze_condition_impact(results)
        
        # 生成强度影响分析
        self._analyze_intensity_impact(results)
        
        self.rm.log("Comprehensive analysis completed!")
    
    def _analyze_condition_impact(self, results):
        """分析不同条件的影响"""
        condition_impact = {}
        
        for condition_type in set(results["condition_types"]):
            condition_indices = [i for i, ct in enumerate(results["condition_types"]) if ct == condition_type]
            condition_asrs = [results["attack_success_rates"][i] for i in condition_indices]
            condition_ssim = [results["perceptual_qualities"]["ssim"][i] for i in condition_indices]
            
            condition_impact[condition_type] = {
                "avg_asr": np.mean(condition_asrs),
                "std_asr": np.std(condition_asrs),
                "avg_ssim": np.mean(condition_ssim),
                "std_ssim": np.std(condition_ssim),
                "sample_count": len(condition_asrs)
            }
        
        # 保存条件影响分析
        import json
        with open("Results/experiment_06_real_world/condition_impact_analysis.json", 'w') as f:
            json.dump(condition_impact, f, indent=2)
        
        self.rm.log("Condition impact analysis saved")
    
    def _analyze_intensity_impact(self, results):
        """分析不同强度的影响"""
        intensity_impact = {}
        
        for intensity in set(results["intensity_levels"]):
            intensity_indices = [i for i, il in enumerate(results["intensity_levels"]) if il == intensity]
            intensity_asrs = [results["attack_success_rates"][i] for i in intensity_indices]
            intensity_ssim = [results["perceptual_qualities"]["ssim"][i] for i in intensity_indices]
            
            intensity_impact[intensity] = {
                "avg_asr": np.mean(intensity_asrs),
                "std_asr": np.std(intensity_asrs),
                "avg_ssim": np.mean(intensity_ssim),
                "std_ssim": np.std(intensity_ssim),
                "sample_count": len(intensity_asrs)
            }
        
        # 保存强度影响分析
        import json
        with open("Results/experiment_06_real_world/intensity_impact_analysis.json", 'w') as f:
            json.dump(intensity_impact, f, indent=2)
        
        self.rm.log("Intensity impact analysis saved")


def run_real_world_effectiveness_experiment(dataloader_mode="small"):
    """
    运行现实世界效果评估实验
    Args:
        dataloader_mode: 数据加载器模式
    """
    rm = ResultManager.get_instance()
    rm.set_experiment("RealWorldEffectiveness")
    rm.set_test("physical_conditions_evaluation")
    
    # 获取数据加载器
    dataloader = get_experiment_dataloader(dataloader_mode)
    
    # 获取配置
    config = get_config("default")
    
    # 初始化评估器
    evaluator = RealWorldEffectivenessEvaluator()
    
    # 评估现实世界效果
    rm.log("=== Starting Real-World Effectiveness Evaluation ===")
    results = evaluator.evaluate_real_world_effectiveness(dataloader, config)
    
    # 生成综合分析
    evaluator.generate_comprehensive_analysis(results)
    
    # 输出结果摘要
    rm.log("=== Real-World Effectiveness Results Summary ===")
    rm.log(f"Overall ASR: {results['overall_statistics']['overall_asr']:.3f}")
    rm.log(f"Total samples: {results['overall_statistics']['total_samples']}")
    rm.log(f"Successful attacks: {results['overall_statistics']['successful_attacks']}")
    rm.log(f"Condition types tested: {results['overall_statistics']['condition_types_count']}")
    rm.log(f"Intensity levels tested: {results['overall_statistics']['intensity_levels_count']}")
    
    # 条件影响摘要
    condition_asrs = {}
    for condition_type in set(results["condition_types"]):
        condition_indices = [i for i, ct in enumerate(results["condition_types"]) if ct == condition_type]
        condition_asr = np.mean([results["attack_success_rates"][i] for i in condition_indices])
        condition_asrs[condition_type] = condition_asr
    
    rm.log("Condition-specific ASR:")
    for condition, asr in condition_asrs.items():
        rm.log(f"  {condition}: {asr:.3f}")
    
    rm.log("Real-world effectiveness experiment completed!")
    rm.log("Results and visualizations saved to: Results/experiment_06_real_world/")
    
    return results


if __name__ == '__main__':
    run_real_world_effectiveness_experiment(dataloader_mode="small") 