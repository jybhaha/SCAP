"""
真实世界攻击评估使用示例
展示如何使用真实世界评估框架进行攻击成功率评估
"""

import torch
import numpy as np
from experiments.experiment_05_real_world import run_real_world_experiment
from configs.real_world_configs import get_real_world_config, get_physical_condition_params
from utils.real_world_visualizer import RealWorldVisualizer
from utils.result_saver import ResultManager


def example_lightweight_evaluation():
    """轻量级评估示例"""
    print("=== 轻量级真实世界评估示例 ===")
    
    # 使用轻量级配置进行快速评估
    results, report = run_real_world_experiment(
        dataloader_mode="test",
        config_name="lightweight",
        num_samples=50,  # 少量样本
        num_variations=10  # 少量变化
    )
    
    print(f"轻量级评估完成")
    print(f"平均攻击成功率: {report['avg_attack_success_rate']:.3f}")
    print(f"成功率标准差: {report['std_attack_success_rate']:.3f}")
    
    return results, report


def example_comprehensive_evaluation():
    """全面评估示例"""
    print("=== 全面真实世界评估示例 ===")
    
    # 使用全面配置进行详细评估
    results, report = run_real_world_experiment(
        dataloader_mode="test",
        config_name="comprehensive",
        num_samples=200,  # 更多样本
        num_variations=30  # 更多变化
    )
    
    print(f"全面评估完成")
    print(f"平均攻击成功率: {report['avg_attack_success_rate']:.3f}")
    print(f"成功率标准差: {report['std_attack_success_rate']:.3f}")
    
    return results, report


def example_custom_evaluation():
    """自定义评估示例"""
    print("=== 自定义真实世界评估示例 ===")
    
    # 自定义参数
    custom_config = {
        'num_samples': 100,
        'num_variations': 20,
        'physical_conditions': ['lighting', 'viewpoint', 'compression'],
        'intensity': 'heavy'  # 使用重度物理条件
    }
    
    results, report = run_real_world_experiment(
        dataloader_mode="test",
        config_name="default",
        num_samples=custom_config['num_samples'],
        num_variations=custom_config['num_variations']
    )
    
    print(f"自定义评估完成")
    print(f"平均攻击成功率: {report['avg_attack_success_rate']:.3f}")
    print(f"成功率标准差: {report['std_attack_success_rate']:.3f}")
    
    return results, report


def example_condition_analysis():
    """物理条件分析示例"""
    print("=== 物理条件影响分析示例 ===")
    
    # 分析不同物理条件对攻击成功率的影响
    conditions = ['lighting', 'viewpoint', 'noise_blur', 'compression']
    condition_results = {}
    
    for condition in conditions:
        print(f"分析条件: {condition}")
        
        # 对每种条件进行单独评估
        results, report = run_real_world_experiment(
            dataloader_mode="test",
            config_name="lightweight",
            num_samples=30,
            num_variations=5
        )
        
        condition_results[condition] = {
            'avg_success_rate': report['avg_attack_success_rate'],
            'std_success_rate': report['std_attack_success_rate'],
            'perceptual_metrics': report['perceptual_metrics']
        }
    
    # 打印条件分析结果
    print("\n物理条件影响分析结果:")
    for condition, result in condition_results.items():
        print(f"{condition}:")
        print(f"  平均成功率: {result['avg_success_rate']:.3f} ± {result['std_success_rate']:.3f}")
        for metric, value in result['perceptual_metrics'].items():
            print(f"  {metric.upper()}: {value:.3f}")
    
    return condition_results


def example_intensity_comparison():
    """强度对比示例"""
    print("=== 物理条件强度对比示例 ===")
    
    intensities = ['light', 'medium', 'heavy']
    intensity_results = {}
    
    for intensity in intensities:
        print(f"测试强度: {intensity}")
        
        # 使用不同强度进行评估
        results, report = run_real_world_experiment(
            dataloader_mode="test",
            config_name="lightweight",
            num_samples=40,
            num_variations=8
        )
        
        intensity_results[intensity] = {
            'avg_success_rate': report['avg_attack_success_rate'],
            'std_success_rate': report['std_attack_success_rate'],
            'perceptual_metrics': report['perceptual_metrics']
        }
    
    # 打印强度对比结果
    print("\n强度对比结果:")
    for intensity, result in intensity_results.items():
        print(f"{intensity}强度:")
        print(f"  平均成功率: {result['avg_success_rate']:.3f} ± {result['std_success_rate']:.3f}")
        for metric, value in result['perceptual_metrics'].items():
            print(f"  {metric.upper()}: {value:.3f}")
    
    return intensity_results


def example_visualization_demo():
    """可视化演示示例"""
    print("=== 可视化演示示例 ===")
    
    # 运行评估
    results, report = run_real_world_experiment(
        dataloader_mode="test",
        config_name="lightweight",
        num_samples=60,
        num_variations=15
    )
    
    # 创建可视化器
    visualizer = RealWorldVisualizer()
    
    # 生成各种可视化图表
    print("生成攻击成功率图表...")
    visualizer.plot_attack_success_rates(results)
    
    print("生成感知质量分析图表...")
    visualizer.plot_perceptual_quality_analysis(results)
    
    print("生成条件影响分析图表...")
    visualizer.plot_condition_impact_analysis(results)
    
    print("生成综合报告...")
    visualizer.create_comprehensive_report(results)
    
    print("保存CSV结果...")
    visualizer.save_results_to_csv(results)
    
    print("可视化演示完成")
    
    return results, report


def example_batch_evaluation():
    """批量评估示例"""
    print("=== 批量评估示例 ===")
    
    # 定义不同的评估配置
    evaluation_configs = [
        {
            'name': 'baseline',
            'config_name': 'lightweight',
            'num_samples': 50,
            'num_variations': 10
        },
        {
            'name': 'standard',
            'config_name': 'default',
            'num_samples': 100,
            'num_variations': 20
        },
        {
            'name': 'comprehensive',
            'config_name': 'comprehensive',
            'num_samples': 200,
            'num_variations': 30
        }
    ]
    
    batch_results = {}
    
    for config in evaluation_configs:
        print(f"运行配置: {config['name']}")
        
        results, report = run_real_world_experiment(
            dataloader_mode="test",
            config_name=config['config_name'],
            num_samples=config['num_samples'],
            num_variations=config['num_variations']
        )
        
        batch_results[config['name']] = {
            'results': results,
            'report': report,
            'config': config
        }
    
    # 打印批量评估结果
    print("\n批量评估结果:")
    for name, result in batch_results.items():
        report = result['report']
        print(f"{name}:")
        print(f"  平均成功率: {report['avg_attack_success_rate']:.3f} ± {report['std_attack_success_rate']:.3f}")
        print(f"  样本数量: {result['config']['num_samples']}")
        print(f"  变化次数: {result['config']['num_variations']}")
    
    return batch_results


def main():
    """主函数 - 运行所有示例"""
    print("真实世界攻击评估框架演示")
    print("=" * 50)
    
    # 设置结果管理器
    rm = ResultManager.get_instance()
    rm.set_experiment("RealWorldDemo")
    
    try:
        # 1. 轻量级评估
        print("\n1. 轻量级评估")
        lightweight_results, lightweight_report = example_lightweight_evaluation()
        
        # 2. 全面评估
        print("\n2. 全面评估")
        comprehensive_results, comprehensive_report = example_comprehensive_evaluation()
        
        # 3. 自定义评估
        print("\n3. 自定义评估")
        custom_results, custom_report = example_custom_evaluation()
        
        # 4. 条件分析
        print("\n4. 物理条件分析")
        condition_results = example_condition_analysis()
        
        # 5. 强度对比
        print("\n5. 强度对比")
        intensity_results = example_intensity_comparison()
        
        # 6. 可视化演示
        print("\n6. 可视化演示")
        viz_results, viz_report = example_visualization_demo()
        
        # 7. 批量评估
        print("\n7. 批量评估")
        batch_results = example_batch_evaluation()
        
        print("\n所有示例运行完成！")
        
        # 保存最终结果
        rm.save_metric("demo_completed", True)
        rm.log("真实世界评估演示完成")
        
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        rm.log(f"错误: {e}")


if __name__ == "__main__":
    main() 