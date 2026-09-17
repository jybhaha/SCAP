"""
实验01：基线对比实验
比较自定义算法与多个baseline的攻击成功率
"""

import torch
from utils.result_saver import ResultManager
from experiments.experiment_runner import ExperimentRunner
from data.data_loader import get_experiment_dataloader


def run_baseline_comparison_experiment(comparison_mode="pixel", dataloader_mode="default"):
    """运行基线对比实验
    
    Args:
        comparison_mode: 对比模式，"pixel"表示与像素级方法对比，"patch"表示与补丁级方法对比
        dataloader_mode: 数据加载器模式，"default"、"test"、"small"、"large"、"large_test"
    """
    # 初始化结果管理器
    rm = ResultManager.get_instance()
    rm.set_experiment("BaselineComparison")
    rm.set_test(f"test1_{comparison_mode}")
    
    # 获取数据加载器
    dataloader = get_experiment_dataloader(dataloader_mode)
    
    # 运行对比实验
    runner = ExperimentRunner()
    runner.run_comparison_experiment_01_02(dataloader, comparison_mode=comparison_mode)
    
    rm.log(f"基线对比实验完成！(模式: {comparison_mode}, 数据模式: {dataloader_mode})")


if __name__ == "__main__":
    run_baseline_comparison_experiment() 