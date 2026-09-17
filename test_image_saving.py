#!/usr/bin/env python3
"""
测试图片保存数量控制功能
"""

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from experiments.experiment_runner import ExperimentRunner
from utils.result_saver import ResultManager

def test_image_saving_control():
    """测试图片保存数量控制功能"""
    
    # 启用图片保存
    rm = ResultManager.get_instance()
    rm.enable_image_saving(True)
    
    # 创建简单的数据加载器（使用少量数据）
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    # 使用CIFAR10数据集的一小部分进行测试
    dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)
    # 只取前20个样本进行测试
    dataset = torch.utils.data.Subset(dataset, range(20))
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    
    # 创建实验运行器
    runner = ExperimentRunner(device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # 测试配置
    test_config = {
        'patch_size': (32, 32),
        'blend_width': 4,
        'patch_k': 3,
        'cam_percentile': 0.8,
        'cam_thresh_ratio': 0.5,
        'mask_thresh_ratio': 0.3,
        'steps': 10,
        'eps': 0.1,
        'alpha': 0.01,
        'targeted_attack': False
    }
    
    # 初始化攻击器
    runner._init_attackers(test_config)
    
    # 测试不同的max_save_images值
    test_cases = [
        (None, "保存所有图片"),
        (3, "最多保存3张图片"),
        (5, "最多保存5张图片"),
        (0, "不保存图片")
    ]
    
    for max_images, description in test_cases:
        print(f"\n=== 测试: {description} ===")
        rm.set_test(f"test_max_images_{max_images}")
        
        # 运行测试
        result = runner.run_baseline_method(
            method_name="TestMethod",
            attack_fn=lambda x, y: runner.my_patch_attacker.run(x, y),
            targeted=False,
            dataloader=dataloader,
            target_class=None,
            record_step_metrics=False,
            config=test_config,
            max_save_images=max_images
        )
        
        print(f"测试结果: ASR = {result['asr']:.2f}%")
        print(f"处理样本数: {result['total']}")

if __name__ == "__main__":
    test_image_saving_control() 