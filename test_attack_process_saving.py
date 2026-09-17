#!/usr/bin/env python3
"""
测试攻击过程保存功能
"""

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os

from experiments.experiment_runner import ExperimentRunner
from utils.result_saver import ResultManager
from models.my_patch_attack import MyPatchAttack

class ImageNetDataset(Dataset):
    """自定义ImageNet数据集类"""
    
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # 从文件名提取类别（这里简化处理，实际可能需要映射表）
        filename = os.path.basename(img_path)
        label = hash(filename) % 1000  # 简单的哈希映射到1000个类别
        
        # 加载图片
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def test_attack_process_saving():
    """测试攻击过程保存功能"""
    
    print("=== 测试攻击过程保存功能 ===")
    
    # 初始化结果管理器
    rm = ResultManager.get_instance()
    rm.set_experiment("TestAttackProcess")
    rm.set_test("test_saving")
    rm.enable_image_saving(True)
    
    # 创建简单的数据加载器
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    # 定义ImageNet图片路径
    imagenet_base_path = "/home/jyb/0code/Data/ImageNet/val"
    image_paths = [
        os.path.join(imagenet_base_path, "n04461696/ILSVRC2012_val_00029413.JPEG"),
    ]
    
    # 检查图片是否存在
    existing_paths = []
    for path in image_paths:
        if os.path.exists(path):
            existing_paths.append(path)
        else:
            print(f"警告: 图片不存在: {path}")
    
    if not existing_paths:
        print("错误: 没有找到任何有效的图片路径")
        return
    
    print(f"使用 {len(existing_paths)} 张图片:")
    for path in existing_paths:
        print(f"  - {path}")
    
    # 使用自定义ImageNet数据集
    dataset = ImageNetDataset(existing_paths, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # 创建实验运行器
    runner = ExperimentRunner(device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # 配置参数
    config = {
        'patch_size': (24, 24),  # 使用较小的补丁尺寸
        'blend_width': 3,
        'patch_k': 2,
        'cam_percentile': 0.8,
        'cam_thresh_ratio': 0.5,
        'mask_thresh_ratio': 0.3,
        'steps': 5,  # 减少步数以加快测试
        'eps': 0.1,
        'alpha': 0.01,
        'targeted_attack': False
    }
    
    print("创建攻击器...")
    
    # 创建攻击器（启用攻击过程保存）
    attacker = MyPatchAttack(
        model=runner.classifier_model,
        device=runner.device,
        config=config,
        debug=True,
        save_imgs=False,
        save_attack_process=True
    )
    
    print("开始测试攻击过程保存...")
    
    # 处理样本
    for sample_idx, (x, y) in enumerate(dataloader):
        print(f"处理测试样本 {sample_idx + 1}")
        
        x = x.to(runner.device)
        y = y.to(runner.device)
        
        try:
            # 运行攻击
            adv = attacker.run(x, y, sample_idx=0)
            print("攻击完成，检查保存的图片...")
            
            # 检查保存的图片
            process_dir = os.path.join(rm.img_dir, "attack_process")
            if os.path.exists(process_dir):
                print(f"攻击过程图片保存在: {process_dir}")
                
                # 列出保存的图片
                for sample_dir in os.listdir(process_dir):
                    sample_path = os.path.join(process_dir, sample_dir)
                    if os.path.isdir(sample_path):
                        print(f"\n{sample_dir} 包含的图片:")
                        for img_file in sorted(os.listdir(sample_path)):
                            print(f"  - {img_file}")
                        
                        # 检查关键图片是否存在
                        expected_files = [
                            'original_input.png',
                            'segmentation_0.png',
                            'mask_0.png',
                            'cam_0.png',
                            'cam_mask_0.png',
                            'patches_0.png',
                            'patches_1.png',
                            'blended_patched.png',
                            'adversarial_final.png'
                        ]
                        
                        missing_files = []
                        for expected_file in expected_files:
                            file_path = os.path.join(sample_path, expected_file)
                            if not os.path.exists(file_path):
                                missing_files.append(expected_file)
                        
                        if missing_files:
                            print(f"缺少的图片: {missing_files}")
                        else:
                            print("✓ 所有预期的图片都已保存")
                            
            else:
                print("❌ 攻击过程图片目录不存在")
                
        except Exception as e:
            print(f"❌ 攻击失败: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n测试完成！")

if __name__ == "__main__":
    test_attack_process_saving() 