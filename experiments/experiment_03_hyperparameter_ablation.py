"""
实验02：超参数消融实验
测试不同超参数对攻击效果的影响
"""

import torch
from utils.result_saver import ResultManager
from experiments.experiment_runner import ExperimentRunner
from data.data_loader import get_experiment_dataloader


def run_hyperparameter_ablation_experiment(dataloader_mode="small"):
    """运行超参数消融实验
    
    Args:
        dataloader_mode: 数据加载器模式，"default"、"test"、"small"、"large"、"large_test"
    """
    # 初始化结果管理器
    rm = ResultManager.get_instance()
    rm.set_experiment("HyperparameterAblation")
    rm.set_test("超参数实验")
    
    # 获取数据加载器（使用小规模数据以加快实验速度）
    dataloader = get_experiment_dataloader(dataloader_mode)
    
    # 运行消融实验
    runner = ExperimentRunner()
    runner.run_ablation_experiment_03(dataloader, config_name="optimized")
    
    rm.log(f"超参数消融实验完成！(数据模式: {dataloader_mode})")


if __name__ == "__main__":
    run_hyperparameter_ablation_experiment() 