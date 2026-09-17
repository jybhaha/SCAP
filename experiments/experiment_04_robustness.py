"""
实验04：鲁棒性验证
测试SCAR对抗不同防御（对抗训练、输入预处理、模型集成）和跨模型泛化能力。
"""
import torch
from utils.result_saver import ResultManager
from experiments.experiment_runner import ExperimentRunner
from data.data_loader import get_experiment_dataloader

def run_robustness_experiment(dataloader_mode="medium"):
    """
    鲁棒性验证实验主函数
    Args:
        dataloader_mode: 数据加载器模式
    """
    rm = ResultManager.get_instance()
    rm.set_experiment("Robustness")
    rm.set_test("robustness_test")
    
    # 获取数据加载器
    dataloader = get_experiment_dataloader(dataloader_mode)
    
    # 初始化实验运行器
    runner = ExperimentRunner()
    
    # 运行鲁棒性验证实验
    results = runner.run_robustness_experiment_04(dataloader, config_name="default")
    
    # 输出结果摘要
    rm.log("=== 鲁棒性验证实验结果摘要 ===")
    
    # 防御鲁棒性结果
    rm.log("防御鲁棒性测试结果:")
    for defense_name, defense_results in results["defense_robustness"].items():
        rm.log(f"  {defense_name}: ASR = {defense_results['asr']:.2%}")
    
    # 跨模型泛化结果
    rm.log("跨模型泛化测试结果:")
    for model_name, gen_results in results["cross_model_generalization"].items():
        rm.log(f"  {model_name}: ASR = {gen_results['asr']:.2%}")
    
    # 综合分析结果
    if "analysis" in results:
        analysis = results["analysis"]
        rm.log(f"整体鲁棒性评分: {analysis['overall_robustness']:.2%}")
        
        rm.log("防御有效性排名:")
        defense_ranking = sorted(
            analysis["defense_effectiveness"].items(),
            key=lambda x: x[1]["effectiveness"],
            reverse=True
        )
        for i, (defense_name, defense_info) in enumerate(defense_ranking, 1):
            rm.log(f"  {i}. {defense_name}: {defense_info['effectiveness']:.2%}")
        
        rm.log("泛化能力排名:")
        gen_ranking = sorted(
            analysis["generalization_ability"].items(),
            key=lambda x: x[1]["generalization_strength"],
            reverse=True
        )
        for i, (model_name, gen_info) in enumerate(gen_ranking, 1):
            rm.log(f"  {i}. {model_name}: {gen_info['generalization_strength']:.2%}")
    
    rm.log("鲁棒性验证实验完成，结果已保存。")
    return results

if __name__ == '__main__':
    run_robustness_experiment() 