#!/usr/bin/env python3
"""
运行baseline perceptual对比实验
基于experiment_runner.py的新功能
"""

import os
import sys
import yaml
import argparse
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    parser = argparse.ArgumentParser(description='Run Baseline Perceptual Comparison Experiment')
    parser.add_argument('--config', type=str, 
                       default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "baseline_perceptual_config.yaml"),
                       help='Configuration file path')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for evaluation')
    parser.add_argument('--max_batches', type=int, default=5, help='Maximum number of batches to process')
    parser.add_argument('--save_adv_images', action='store_true', default=True,
                       help='Save adversarial images')
    parser.add_argument('--perceptual_comparison', action='store_true', default=True,
                       help='Generate perceptual comparison')
    
    args = parser.parse_args()
    
    print("开始运行baseline perceptual对比实验...")
    print(f"配置文件: {args.config}")
    print(f"批次大小: {args.batch_size}")
    print(f"最大批次: {args.max_batches}")
    print(f"保存对抗样本: {args.save_adv_images}")
    print(f"生成perceptual比较: {args.perceptual_comparison}")
    
    # 加载配置文件
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"配置文件未找到: {args.config}")
        return False
    
    # 更新配置
    config['experiment_params']['batch_size'] = args.batch_size
    config['experiment_params']['max_batches'] = args.max_batches
    config['output']['save_adv_images'] = args.save_adv_images
    config['output']['perceptual_comparison'] = args.perceptual_comparison
    
    # 创建实验运行器
    try:
        from experiments.experiment_runner import ExperimentRunner
        runner = ExperimentRunner(config)
        
        # 运行实验
        print("开始运行实验...")
        results = runner.run_experiment()
        
        # 如果启用了perceptual比较，运行比较函数
        if args.perceptual_comparison:
            print("生成perceptual比较...")
            runner.advimg_perceptual_comparison(config)
        
        print("实验完成！")
        print(f"结果保存在: {runner.rm.base_dir}")
        
        return True
        
    except Exception as e:
        print(f"实验运行失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = main()
    if success:
        print("\n实验完成！请查看results/baseline目录下的结果。")
    else:
        print("\n实验失败，请检查错误信息。") 