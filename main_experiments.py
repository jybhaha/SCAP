"""
主程序入口
用于运行各种实验的统一入口
"""

import argparse
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from experiments.experiment_01_02 import run_baseline_comparison_experiment
from experiments.experiment_03_hyperparameter_ablation import run_hyperparameter_ablation_experiment
from experiments.experiment_05_transferability import run_transferability_experiment
from experiments.experiment_06_real_world_effectiveness import run_real_world_effectiveness_experiment
from experiments.experiment_08_real_world_api_test import run_real_world_api_test_experiment
from utils.result_saver import ResultManager


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='运行SCAR实验')
    parser.add_argument('--experiment', type=str, required=True,
                       choices=['baseline', 'ablation', 'transferability', 'real_world_effectiveness', 
                               'real_world_api_test', 'all'],
                       help='实验类型')
    parser.add_argument('--dataloader_mode', type=str, default='small',
                       choices=['small', 'medium', 'large', 'default'],
                       help='数据加载器模式')
    parser.add_argument('--comparison_mode', type=str, default='all',
                       choices=['pixel', 'patch', 'all'],
                       help='对比模式（仅用于baseline实验）')
    parser.add_argument('--config', type=str, default='default',
                       choices=['default', 'optimized'],
                       help='配置类型（仅用于ablation实验）')
    parser.add_argument('--categories', type=str, default='household',
                       help='物体类别（仅用于real_world_api_test实验）')
    
    args = parser.parse_args()
    
    # 初始化结果管理器
    rm = ResultManager.get_instance()
    rm.log(f"Starting experiment: {args.experiment}")
    rm.log(f"Dataloader mode: {args.dataloader_mode}")
    
    try:
        if args.experiment == 'baseline':
            rm.log("Running baseline comparison experiment...")
            run_baseline_comparison_experiment(
                dataloader_mode=args.dataloader_mode,
                comparison_mode=args.comparison_mode
            )
            
        elif args.experiment == 'ablation':
            rm.log("Running hyperparameter ablation experiment...")
            run_hyperparameter_ablation_experiment(
                dataloader_mode=args.dataloader_mode,
                config=args.config
            )
            
        elif args.experiment == 'transferability':
            rm.log("Running transferability evaluation experiment...")
            run_transferability_experiment(
                dataloader_mode=args.dataloader_mode
            )
            
        elif args.experiment == 'real_world_effectiveness':
            rm.log("Running real-world effectiveness experiment...")
            run_real_world_effectiveness_experiment(
                dataloader_mode=args.dataloader_mode
            )
            
        elif args.experiment == 'real_world_api_test':
            rm.log("Running real-world API test experiment...")
            run_real_world_api_test_experiment(
                dataloader_mode=args.dataloader_mode,
                categories=args.categories
            )
            
        elif args.experiment == 'all':
            rm.log("Running all experiments...")
            
            # 运行所有实验
            experiments = [
                ('baseline', lambda: run_baseline_comparison_experiment(
                    dataloader_mode=args.dataloader_mode,
                    comparison_mode=args.comparison_mode
                )),
                ('ablation', lambda: run_hyperparameter_ablation_experiment(
                    dataloader_mode=args.dataloader_mode,
                    config=args.config
                )),
                ('transferability', lambda: run_transferability_experiment(
                    dataloader_mode=args.dataloader_mode
                )),
                ('real_world_effectiveness', lambda: run_real_world_effectiveness_experiment(
                    dataloader_mode=args.dataloader_mode
                )),
                ('real_world_api_test', lambda: run_real_world_api_test_experiment(
                    dataloader_mode=args.dataloader_mode,
                    categories=args.categories
                ))
            ]
            
            for exp_name, exp_func in experiments:
                try:
                    rm.log(f"Running {exp_name} experiment...")
                    exp_func()
                    rm.log(f"{exp_name} experiment completed successfully!")
                except Exception as e:
                    rm.log(f"Error in {exp_name} experiment: {e}")
                    continue
        
        rm.log("All experiments completed!")
        
    except Exception as e:
        rm.log(f"Experiment failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main() 