"""
实验：攻击成本对比实验
比较SCAP算法与其他算法的攻击成本，包括计算时间、内存使用、修改像素数等指标
"""

import torch
import time
import psutil
import os

from utils.result_saver import ResultManager
from experiments.experiment_runner import ComparisonExperimentRunner
from data.data_loader import get_experiment_dataloader


def run_attack_cost_comparison(dataloader_mode="default", num_samples=50):
    """运行攻击成本对比实验
    
    Args:
        dataloader_mode: 数据加载器模式，"default"、"test"、"small"、"large"、"large_test"
        num_samples: 要评估的样本数量
    """
    # 初始化结果管理器
    rm = ResultManager.get_instance()
    rm.set_experiment("AttackCostComparison")
    rm.set_test(f"cost_comparison_{dataloader_mode}")
    
    # 获取数据加载器
    dataloader = get_experiment_dataloader(dataloader_mode)
    
    # 运行成本对比实验
    runner = CostComparisonExperimentRunner()
    results = runner.run_cost_comparison(dataloader, num_samples=num_samples)
    
    rm.log(f"攻击成本对比实验完成！(数据模式: {dataloader_mode}, 样本数: {num_samples})")
    rm.log(f"对比结果摘要: {results['summary']}")
    
    return results


class CostComparisonExperimentRunner(ComparisonExperimentRunner):
    """攻击成本对比实验运行器"""
    
    def run_cost_comparison(self, dataloader, num_samples=50):
        """运行攻击成本对比
        
        Args:
            dataloader: 数据加载器
            num_samples: 要评估的样本数量
            
        Returns:
            包含各算法成本指标的字典
        """
        self.rm.log(f"开始运行攻击成本对比实验，评估样本数: {num_samples}")
        
        # 存储结果
        results = {
            "pixel_methods": {},
            "patch_methods": {},
            "scar_method": {},
            "summary": {}
        }
        
        # 定义要比较的方法
        pixel_methods = ["FGSM", "MIFGSM", "DIFGSM", "NIFGSM", "PGD", "VNIFGSM"]
        patch_methods = ["lavan_eva", "art_advpatch_eva"]  # 使用PatchAttackBaseline支持的方法名称
        
        # 收集每个方法的成本数据
        processed_samples = 0
        
        for batch_idx, (images, labels) in enumerate(dataloader):
            if processed_samples >= num_samples:
                break
                
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            for i in range(images.size(0)):
                if processed_samples >= num_samples:
                    break
                    
                img = images[i:i+1]
                label = labels[i:i+1]
                
                self.rm.log(f"处理样本 {processed_samples + 1}/{num_samples}")
                
                # 运行像素级方法并收集成本
                for method in pixel_methods:
                    try:
                        cost_metrics = self._measure_attack_cost(
                            method_type="pixel", 
                            method_name=method,
                            image=img,
                            label=label
                        )
                        
                        if method not in results["pixel_methods"]:
                            results["pixel_methods"][method] = []
                        results["pixel_methods"][method].append(cost_metrics)
                    except Exception as e:
                        self.rm.log(f"像素级方法 {method} 测量失败: {e}")
                
                # 运行补丁级方法并收集成本
                for method in patch_methods:
                    try:
                        cost_metrics = self._measure_attack_cost(
                            method_type="patch",
                            method_name=method,
                            image=img,
                            label=label
                        )
                        
                        if method not in results["patch_methods"]:
                            results["patch_methods"][method] = []
                        results["patch_methods"][method].append(cost_metrics)
                    except Exception as e:
                        error_msg = str(e)
                        if "Patch height and width need to be the same" in error_msg:
                            self.rm.log(f"补丁级方法 {method} 测量失败: ART库要求补丁必须是正方形，但配置中可能传入了非正方形的 patch_shape")
                            self.rm.log(f"建议: 检查配置或修改 base_line.py 中的 art_advpatch_eva 方法，确保 patch_shape 是正方形的")
                        else:
                            self.rm.log(f"补丁级方法 {method} 测量失败: {e}")
                        # 继续运行其他方法，不中断实验
                        continue
                
                # 运行SCAR方法并收集成本
                try:
                    cost_metrics = self._measure_attack_cost(
                        method_type="scar",
                        method_name="SCAR",
                        image=img,
                        label=label
                    )
                    
                    if "metrics" not in results["scar_method"]:
                        results["scar_method"]["metrics"] = []
                    results["scar_method"]["metrics"].append(cost_metrics)
                except Exception as e:
                    self.rm.log(f"SCAR方法测量失败: {e}")
                
                processed_samples += 1
        
        # 计算平均成本指标
        results["summary"] = self._calculate_average_costs(results)
        
        # 保存结果
        self._save_cost_results(results)
        
        # 生成可视化报告
        self._generate_cost_report(results)
        
        return results
    
    def _measure_attack_cost(self, method_type, method_name, image, label):
        """测量攻击方法的成本
        
        Args:
            method_type: 方法类型，"pixel"、"patch"或"scar"
            method_name: 方法名称
            image: 输入图像
            label: 真实标签
            
        Returns:
            包含成本指标的字典
        """
        # 记录开始内存使用
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # 使用perf_counter()获得更精确的时间测量（不受系统时间调整影响）
        start_time = time.perf_counter()
        
        # 运行攻击
        adv_result = None
        try:
            if method_type == "pixel" and self.pixel_attack is not None:
                adv_result = self.pixel_attack.run(method_name, image, label)
            elif method_type == "patch" and self.patch_attack is not None:
                # 正确调用PatchAttackBaseline的run方法，传入方法名称作为第一个参数
                # 对于 art_advpatch_eva，参考 experiment_runner.py 中的调用方式
                if method_name == "art_advpatch_eva":
                    # 使用与 experiment_runner.py 相同的格式: (channels, height, width)
                    # 注意: 高度和宽度都是64，确保是正方形
                    adv_result, _ = self.patch_attack.run(
                        method_name, 
                        image, 
                        label,
                        patch_shape=(3, 64, 64)  # (channels, height, width) 格式，与 experiment_runner.py 保持一致
                    )
                else:
                    adv_result = self.patch_attack.run(method_name, image, label)
                # 对于patch方法，它可能返回两个值：对抗样本和预测列表
                if isinstance(adv_result, tuple) and len(adv_result) == 2:
                    adv_result = adv_result[0]  # 只取对抗样本
            elif method_type == "scar" and self.scar_attack is not None:
                # 运行SCAR攻击，它返回 (x_adv, step_preds_list) 元组
                result = self.scar_attack.run(image, label)
                if isinstance(result, tuple) and len(result) == 2:
                    adv_result = result[0]  # 只取对抗样本
                else:
                    adv_result = result
        except Exception as e:
            self.rm.log(f"攻击方法 {method_type}/{method_name} 执行失败: {e}")
            import traceback
            self.rm.log(f"错误详情: {traceback.format_exc()}")
            adv_result = None
        
        # 记录结束时间（使用perf_counter()）
        end_time = time.perf_counter()
        
        # 计算时间差（perf_counter()应该总是单调递增，但为了安全起见使用max确保非负）
        elapsed_time = max(0.0, end_time - start_time)
        
        # 验证时间测量的合理性（记录异常情况）
        if end_time < start_time:
            self.rm.log(f"警告: 时间测量异常！start_time={start_time}, end_time={end_time}, 差值={end_time - start_time}")
            # perf_counter()理论上不应该出现这种情况，如果出现可能是系统问题
            # 使用绝对值作为fallback
            elapsed_time = abs(end_time - start_time)
        
        # 记录结束内存使用
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        
        # 计算成本指标
        cost_metrics = {
            "time_seconds": elapsed_time,
            "memory_mb": max(0.0, mem_after - mem_before),  # 确保内存差值为非负
            "success": False,
            "modified_pixels": 0,
            "modified_percentage": 0.0
        }
        
        # 记录调试信息（如果时间异常）
        if elapsed_time < 0:
            self.rm.log(f"警告: {method_type}/{method_name} 时间测量为负: {elapsed_time}")
        if elapsed_time > 300:  # 如果超过5分钟，记录警告
            self.rm.log(f"警告: {method_type}/{method_name} 执行时间过长: {elapsed_time:.2f}秒")
        
        # 如果攻击成功，计算修改的像素数
        if adv_result is not None:
            try:
                adv_image = self._process_attack_result(adv_result)
                
                # 计算修改的像素数
                diff = torch.abs(adv_image - image)
                modified_pixels = (diff > 1e-5).sum().item()
                total_pixels = image.numel() / 3  # 除以3因为有3个通道
                
                cost_metrics["modified_pixels"] = modified_pixels
                cost_metrics["modified_percentage"] = (modified_pixels / total_pixels) * 100
                
                # 检查攻击是否成功
                with torch.no_grad():
                    logits_adv = self.classifier_model(adv_image)
                    pred_adv = logits_adv.argmax(dim=1)
                    cost_metrics["success"] = (pred_adv != label).item()
                    
                    # 记录置信度变化
                    logits_clean = self.classifier_model(image)
                    conf_clean = torch.softmax(logits_clean, dim=1).max(dim=1)[0].item()
                    conf_adv = torch.softmax(logits_adv, dim=1).max(dim=1)[0].item()
                    cost_metrics["confidence_before"] = conf_clean
                    cost_metrics["confidence_after"] = conf_adv
                    
            except Exception as e:
                self.rm.log(f"计算修改像素数失败: {e}")
        
        return cost_metrics
    
    def _calculate_average_costs(self, results):
        """计算平均成本指标
        
        Args:
            results: 原始结果字典
            
        Returns:
            包含平均成本指标的摘要字典
        """
        summary = {
            "pixel_methods": {},
            "patch_methods": {},
            "scar_method": {}
        }
        
        # 计算像素级方法的平均成本
        for method, metrics_list in results["pixel_methods"].items():
            if metrics_list:
                avg_time = sum(m["time_seconds"] for m in metrics_list) / len(metrics_list)
                avg_memory = sum(m["memory_mb"] for m in metrics_list) / len(metrics_list)
                avg_modified_pixels = sum(m["modified_pixels"] for m in metrics_list) / len(metrics_list)
                avg_modified_percentage = sum(m["modified_percentage"] for m in metrics_list) / len(metrics_list)
                success_rate = sum(m["success"] for m in metrics_list) / len(metrics_list)
                
                summary["pixel_methods"][method] = {
                    "avg_time_seconds": avg_time,
                    "avg_memory_mb": avg_memory,
                    "avg_modified_pixels": avg_modified_pixels,
                    "avg_modified_percentage": avg_modified_percentage,
                    "success_rate": success_rate
                }
        
        # 计算补丁级方法的平均成本
        for method, metrics_list in results["patch_methods"].items():
            if metrics_list:
                avg_time = sum(m["time_seconds"] for m in metrics_list) / len(metrics_list)
                avg_memory = sum(m["memory_mb"] for m in metrics_list) / len(metrics_list)
                avg_modified_pixels = sum(m["modified_pixels"] for m in metrics_list) / len(metrics_list)
                avg_modified_percentage = sum(m["modified_percentage"] for m in metrics_list) / len(metrics_list)
                success_rate = sum(m["success"] for m in metrics_list) / len(metrics_list)
                
                summary["patch_methods"][method] = {
                    "avg_time_seconds": avg_time,
                    "avg_memory_mb": avg_memory,
                    "avg_modified_pixels": avg_modified_pixels,
                    "avg_modified_percentage": avg_modified_percentage,
                    "success_rate": success_rate
                }
        
        # 计算SCAR方法的平均成本
        if "metrics" in results["scar_method"] and results["scar_method"]["metrics"]:
            metrics_list = results["scar_method"]["metrics"]
            avg_time = sum(m["time_seconds"] for m in metrics_list) / len(metrics_list)
            avg_memory = sum(m["memory_mb"] for m in metrics_list) / len(metrics_list)
            avg_modified_pixels = sum(m["modified_pixels"] for m in metrics_list) / len(metrics_list)
            avg_modified_percentage = sum(m["modified_percentage"] for m in metrics_list) / len(metrics_list)
            success_rate = sum(m["success"] for m in metrics_list) / len(metrics_list)
            
            summary["scar_method"] = {
                "avg_time_seconds": avg_time,
                "avg_memory_mb": avg_memory,
                "avg_modified_pixels": avg_modified_pixels,
                "avg_modified_percentage": avg_modified_percentage,
                "success_rate": success_rate
            }
        
        return summary
    
    def _save_cost_results(self, results):
        """保存成本对比结果
        
        Args:
            results: 结果字典
        """
        try:
            import json
            import os
            
            # 创建结果目录
            result_dir = self.rm.log_dir
            os.makedirs(result_dir, exist_ok=True)
            
            # 保存详细结果
            detailed_file = os.path.join(result_dir, "attack_cost_detailed_results.json")
            with open(detailed_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, default=str, ensure_ascii=False)
            
            # 保存摘要结果
            summary_file = os.path.join(result_dir, "attack_cost_summary_results.json")
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(results["summary"], f, indent=2, default=str, ensure_ascii=False)
            
            self.rm.log(f"详细成本结果已保存到: {detailed_file}")
            self.rm.log(f"摘要成本结果已保存到: {summary_file}")
            
        except Exception as e:
            self.rm.log(f"保存成本结果失败: {e}")
    
    def _generate_cost_report(self, results):
        """生成成本对比报告
        
        Args:
            results: 结果字典
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            import os
            
            # 创建报告目录
            report_dir = os.path.join(self.rm.log_dir, "reports")
            os.makedirs(report_dir, exist_ok=True)
            
            # 提取数据进行可视化
            methods = []
            avg_times = []
            avg_modified_percentages = []
            success_rates = []
            
            # 添加像素级方法
            for method, metrics in results["summary"]["pixel_methods"].items():
                methods.append(method)
                avg_times.append(metrics["avg_time_seconds"])
                avg_modified_percentages.append(metrics["avg_modified_percentage"])
                success_rates.append(metrics["success_rate"])
            
            # 添加补丁级方法
            for method, metrics in results["summary"]["patch_methods"].items():
                methods.append(method)
                avg_times.append(metrics["avg_time_seconds"])
                avg_modified_percentages.append(metrics["avg_modified_percentage"])
                success_rates.append(metrics["success_rate"])
            
            # 添加SCAR方法
            if results["summary"]["scar_method"]:
                methods.append("SCAR")
                avg_times.append(results["summary"]["scar_method"]["avg_time_seconds"])
                avg_modified_percentages.append(results["summary"]["scar_method"]["avg_modified_percentage"])
                success_rates.append(results["summary"]["scar_method"]["success_rate"])
            
            # 创建执行时间对比图
            plt.figure(figsize=(12, 6))
            bars = plt.bar(methods, avg_times, color='skyblue')
            plt.xlabel('攻击方法')
            plt.ylabel('平均执行时间 (秒)')
            plt.title('各攻击方法执行时间对比')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # 在柱状图上添加数值标签
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.4f}',
                        ha='center', va='bottom')
            
            time_file = os.path.join(report_dir, "attack_time_comparison.png")
            plt.savefig(time_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            # 创建修改像素比例对比图
            plt.figure(figsize=(12, 6))
            bars = plt.bar(methods, avg_modified_percentages, color='lightgreen')
            plt.xlabel('攻击方法')
            plt.ylabel('平均修改像素百分比 (%)')
            plt.title('各攻击方法修改像素百分比对比')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # 在柱状图上添加数值标签
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}%',
                        ha='center', va='bottom')
            
            pixels_file = os.path.join(report_dir, "modified_pixels_comparison.png")
            plt.savefig(pixels_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            # 创建成功率对比图
            plt.figure(figsize=(12, 6))
            bars = plt.bar(methods, [rate*100 for rate in success_rates], color='salmon')
            plt.xlabel('攻击方法')
            plt.ylabel('攻击成功率 (%)')
            plt.title('各攻击方法成功率对比')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # 在柱状图上添加数值标签
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}%',
                        ha='center', va='bottom')
            
            success_file = os.path.join(report_dir, "success_rate_comparison.png")
            plt.savefig(success_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            # 创建成本效益散点图（时间 vs 成功率）
            plt.figure(figsize=(10, 8))
            plt.scatter(avg_times, [rate*100 for rate in success_rates], 
                      s=[p*10 for p in avg_modified_percentages],  # 点大小表示修改像素百分比
                      alpha=0.7, c=range(len(methods)), cmap='viridis')
            
            # 添加方法标签
            for i, method in enumerate(methods):
                plt.annotate(method, (avg_times[i], success_rates[i]*100),
                            xytext=(5, 5), textcoords='offset points')
            
            plt.xlabel('平均执行时间 (秒)')
            plt.ylabel('攻击成功率 (%)')
            plt.title('攻击方法成本效益分析（点大小表示修改像素百分比）')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            efficiency_file = os.path.join(report_dir, "cost_efficiency_analysis.png")
            plt.savefig(efficiency_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.rm.log(f"攻击成本对比报告已生成，保存在: {report_dir}")
            
        except Exception as e:
            self.rm.log(f"生成成本对比报告失败: {e}")


if __name__ == "__main__":
    # 默认运行小规模测试
    run_attack_cost_comparison(dataloader_mode="medium", num_samples=100)