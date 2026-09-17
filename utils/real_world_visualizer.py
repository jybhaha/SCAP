"""
真实世界评估结果可视化工具
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from utils.result_saver import ResultManager
import os


class RealWorldVisualizer:
    """真实世界评估结果可视化器"""
    
    def __init__(self, save_dir="Results/real_world_visualizations"):
        self.save_dir = save_dir
        self.rm = ResultManager.get_instance()
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 设置绘图样式
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_attack_success_rates(self, results, save_name="attack_success_rates"):
        """绘制攻击成功率图表"""
        success_rates = results['attack_success_rates']
        condition_types = results['condition_types']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 1. 攻击成功率分布直方图
        ax1.hist(success_rates, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(np.mean(success_rates), color='red', linestyle='--', 
                    label=f'平均值: {np.mean(success_rates):.3f}')
        ax1.axvline(np.median(success_rates), color='green', linestyle='--', 
                    label=f'中位数: {np.median(success_rates):.3f}')
        ax1.set_xlabel('攻击成功率')
        ax1.set_ylabel('频次')
        ax1.set_title('攻击成功率分布')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 不同条件下的攻击成功率
        condition_counts = {}
        condition_success_rates = {}
        
        for condition, rate in zip(condition_types, success_rates):
            if condition not in condition_counts:
                condition_counts[condition] = 0
                condition_success_rates[condition] = []
            condition_counts[condition] += 1
            condition_success_rates[condition].append(rate)
        
        # 计算每种条件的平均成功率
        avg_rates = []
        condition_names = []
        for condition, rates in condition_success_rates.items():
            avg_rates.append(np.mean(rates))
            condition_names.append(condition)
        
        bars = ax2.bar(condition_names, avg_rates, alpha=0.7, color='lightcoral')
        ax2.set_xlabel('物理条件类型')
        ax2.set_ylabel('平均攻击成功率')
        ax2.set_title('不同物理条件下的攻击成功率')
        ax2.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, rate in zip(bars, avg_rates):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{rate:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f"{save_name}.png"), dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def plot_perceptual_quality_analysis(self, results, save_name="perceptual_quality"):
        """绘制感知质量分析图表"""
        perceptual_metrics = results['perceptual_qualities']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        metrics = ['ssim', 'psnr', 'lpips']
        metric_names = ['SSIM', 'PSNR', 'LPIPS']
        colors = ['skyblue', 'lightgreen', 'lightcoral']
        
        for i, (metric, metric_name, color) in enumerate(zip(metrics, metric_names, colors)):
            if metric in perceptual_metrics and perceptual_metrics[metric]:
                values = perceptual_metrics[metric]
                
                # 直方图
                axes[i].hist(values, bins=20, alpha=0.7, color=color, edgecolor='black')
                axes[i].axvline(np.mean(values), color='red', linestyle='--', 
                               label=f'平均值: {np.mean(values):.3f}')
                axes[i].set_xlabel(metric_name)
                axes[i].set_ylabel('频次')
                axes[i].set_title(f'{metric_name} 分布')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
        
        # 感知质量指标对比
        if len(metrics) >= 2:
            metric_data = []
            metric_labels = []
            
            for metric in metrics:
                if metric in perceptual_metrics and perceptual_metrics[metric]:
                    metric_data.extend(perceptual_metrics[metric])
                    metric_labels.extend([metric.upper()] * len(perceptual_metrics[metric]))
            
            if metric_data:
                df = pd.DataFrame({
                    'Metric': metric_labels,
                    'Value': metric_data
                })
                
                sns.boxplot(data=df, x='Metric', y='Value', ax=axes[3])
                axes[3].set_title('感知质量指标对比')
                axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f"{save_name}.png"), dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def plot_condition_impact_analysis(self, results, save_name="condition_impact"):
        """绘制物理条件影响分析"""
        success_rates = results['attack_success_rates']
        condition_types = results['condition_types']
        perceptual_metrics = results['perceptual_qualities']
        
        # 创建条件影响分析数据
        condition_data = {}
        for condition, rate in zip(condition_types, success_rates):
            if condition not in condition_data:
                condition_data[condition] = {
                    'success_rates': [],
                    'ssim_scores': [],
                    'psnr_scores': [],
                    'lpips_scores': []
                }
            condition_data[condition]['success_rates'].append(rate)
        
        # 添加感知质量数据
        for i, condition in enumerate(condition_types):
            if i < len(perceptual_metrics['ssim']):
                condition_data[condition]['ssim_scores'].append(perceptual_metrics['ssim'][i])
            if i < len(perceptual_metrics['psnr']):
                condition_data[condition]['psnr_scores'].append(perceptual_metrics['psnr'][i])
            if i < len(perceptual_metrics['lpips']):
                condition_data[condition]['lpips_scores'].append(perceptual_metrics['lpips'][i])
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 攻击成功率 vs 感知质量散点图
        for condition, data in condition_data.items():
            if data['success_rates'] and data['ssim_scores']:
                axes[0].scatter(data['ssim_scores'], data['success_rates'], 
                               label=condition, alpha=0.7, s=50)
        
        axes[0].set_xlabel('SSIM')
        axes[0].set_ylabel('攻击成功率')
        axes[0].set_title('攻击成功率 vs SSIM')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. 不同条件下的平均指标对比
        conditions = list(condition_data.keys())
        avg_success_rates = [np.mean(data['success_rates']) for data in condition_data.values()]
        avg_ssim = [np.mean(data['ssim_scores']) if data['ssim_scores'] else 0 
                   for data in condition_data.values()]
        
        x = np.arange(len(conditions))
        width = 0.35
        
        bars1 = axes[1].bar(x - width/2, avg_success_rates, width, label='攻击成功率', alpha=0.7)
        axes[1].set_xlabel('物理条件')
        axes[1].set_ylabel('平均值')
        axes[1].set_title('不同条件下的平均指标')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(conditions, rotation=45)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 3. 条件类型分布饼图
        condition_counts = {}
        for condition in condition_types:
            condition_counts[condition] = condition_counts.get(condition, 0) + 1
        
        if condition_counts:
            sizes = list(condition_counts.values())
            labels = list(condition_counts.keys())
            colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
            
            axes[2].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            axes[2].set_title('物理条件分布')
        
        # 4. 攻击成功率时间序列（按实验顺序）
        axes[3].plot(range(len(success_rates)), success_rates, marker='o', alpha=0.7)
        axes[3].axhline(np.mean(success_rates), color='red', linestyle='--', 
                       label=f'平均值: {np.mean(success_rates):.3f}')
        axes[3].set_xlabel('实验序号')
        axes[3].set_ylabel('攻击成功率')
        axes[3].set_title('攻击成功率变化趋势')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f"{save_name}.png"), dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def create_comprehensive_report(self, results, save_name="comprehensive_report"):
        """创建综合报告"""
        fig = plt.figure(figsize=(20, 16))
        
        # 创建子图网格
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. 攻击成功率统计
        ax1 = fig.add_subplot(gs[0, 0])
        success_rates = results['attack_success_rates']
        ax1.hist(success_rates, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(np.mean(success_rates), color='red', linestyle='--', 
                    label=f'平均值: {np.mean(success_rates):.3f}')
        ax1.set_xlabel('攻击成功率')
        ax1.set_ylabel('频次')
        ax1.set_title('攻击成功率分布')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 感知质量指标
        ax2 = fig.add_subplot(gs[0, 1])
        perceptual_metrics = results['perceptual_qualities']
        metrics = ['ssim', 'psnr']
        metric_names = ['SSIM', 'PSNR']
        colors = ['lightgreen', 'lightcoral']
        
        for metric, metric_name, color in zip(metrics, metric_names, colors):
            if metric in perceptual_metrics and perceptual_metrics[metric]:
                values = perceptual_metrics[metric]
                ax2.hist(values, bins=15, alpha=0.7, color=color, 
                        label=f'{metric_name}: {np.mean(values):.3f}')
        
        ax2.set_xlabel('感知质量指标')
        ax2.set_ylabel('频次')
        ax2.set_title('感知质量分布')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 条件类型分布
        ax3 = fig.add_subplot(gs[0, 2])
        condition_counts = {}
        for condition in results['condition_types']:
            condition_counts[condition] = condition_counts.get(condition, 0) + 1
        
        if condition_counts:
            sizes = list(condition_counts.values())
            labels = list(condition_counts.keys())
            colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
            ax3.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax3.set_title('物理条件分布')
        
        # 4. 攻击成功率 vs 感知质量
        ax4 = fig.add_subplot(gs[1, :2])
        if perceptual_metrics['ssim'] and len(perceptual_metrics['ssim']) == len(success_rates):
            ax4.scatter(perceptual_metrics['ssim'], success_rates, alpha=0.7, s=50)
            ax4.set_xlabel('SSIM')
            ax4.set_ylabel('攻击成功率')
            ax4.set_title('攻击成功率 vs SSIM')
            ax4.grid(True, alpha=0.3)
        
        # 5. 时间序列
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.plot(range(len(success_rates)), success_rates, marker='o', alpha=0.7)
        ax5.axhline(np.mean(success_rates), color='red', linestyle='--', 
                   label=f'平均值: {np.mean(success_rates):.3f}')
        ax5.set_xlabel('实验序号')
        ax5.set_ylabel('攻击成功率')
        ax5.set_title('攻击成功率变化')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. 统计摘要
        ax6 = fig.add_subplot(gs[2, :])
        ax6.axis('off')
        
        # 计算统计信息
        stats_text = f"""
        真实世界攻击评估统计摘要
        
        攻击成功率统计:
        - 平均成功率: {np.mean(success_rates):.3f} ± {np.std(success_rates):.3f}
        - 最低成功率: {np.min(success_rates):.3f}
        - 最高成功率: {np.max(success_rates):.3f}
        - 成功率标准差: {np.std(success_rates):.3f}
        
        感知质量统计:
        """
        
        for metric_name, values in perceptual_metrics.items():
            if values:
                avg_value = np.mean(values)
                std_value = np.std(values)
                stats_text += f"- {metric_name.upper()}: {avg_value:.3f} ± {std_value:.3f}\n"
        
        stats_text += f"""
        物理条件统计:
        - 总实验次数: {len(success_rates)}
        - 条件类型数: {len(set(results['condition_types']))}
        """
        
        ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle('真实世界攻击评估综合报告', fontsize=16, fontweight='bold')
        plt.savefig(os.path.join(self.save_dir, f"{save_name}.png"), dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def save_results_to_csv(self, results, save_name="real_world_results"):
        """将结果保存为CSV文件"""
        # 创建数据框
        data = {
            'experiment_id': range(len(results['attack_success_rates'])),
            'attack_success_rate': results['attack_success_rates'],
            'condition_type': results['condition_types']
        }
        
        # 添加感知质量指标
        for metric_name, values in results['perceptual_qualities'].items():
            if values:
                data[f'{metric_name}_score'] = values[:len(results['attack_success_rates'])]
        
        df = pd.DataFrame(data)
        
        # 保存CSV文件
        csv_path = os.path.join(self.save_dir, f"{save_name}.csv")
        df.to_csv(csv_path, index=False)
        self.rm.log(f"结果已保存到: {csv_path}")
        
        return csv_path


def visualize_real_world_results(results, save_dir="Results/real_world_visualizations"):
    """可视化真实世界评估结果的便捷函数"""
    visualizer = RealWorldVisualizer(save_dir)
    
    # 生成各种可视化图表
    visualizer.plot_attack_success_rates(results)
    visualizer.plot_perceptual_quality_analysis(results)
    visualizer.plot_condition_impact_analysis(results)
    visualizer.create_comprehensive_report(results)
    
    # 保存CSV结果
    visualizer.save_results_to_csv(results)
    
    return visualizer 