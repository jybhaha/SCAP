#!/usr/bin/env python3
"""
攻击过程保存功能使用示例
展示如何保存攻击方案的各个步骤图片，用于绘制示意图
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

def example_save_attack_process():
    """示例：保存攻击过程图片"""
    
    print("=== 攻击过程保存功能示例 ===")
    
    # 1. 初始化结果管理器
    rm = ResultManager.get_instance()
    rm.set_experiment("AttackProcessExample")
    rm.set_test("demo")
    rm.enable_image_saving(True)
    
    # 2. 创建数据加载器
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    # 定义ImageNet图片路径
    imagenet_base_path = "data/ImageNet/val"
    image_paths = [
        os.path.join(imagenet_base_path, "n04461696/ILSVRC2012_val_00029413.JPEG"),
        os.path.join(imagenet_base_path, "n03891332/ILSVRC2012_val_00034802.JPEG"),
        os.path.join(imagenet_base_path, "n03028079/ILSVRC2012_val_00004912.JPEG"),
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
    
    # 3. 创建实验运行器
    runner = ExperimentRunner(device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # 4. 配置参数
    config = {
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
    
    # 5. 创建攻击器（启用攻击过程保存）
    attacker = MyPatchAttack(
        model=runner.classifier_model,
        device=runner.device,
        config=config,
        debug=True,
        save_imgs=False,  # 关闭原有的图片保存
        save_attack_process=True  # 启用攻击过程保存
    )
    
    print("开始处理样本...")
    
    # 6. 处理每个样本
    for sample_idx, (x, y) in enumerate(dataloader):
        print(f"\n处理样本 {sample_idx + 1}/3")
        
        x = x.to(runner.device)
        y = y.to(runner.device)
        
        # 运行攻击，自动保存过程图片
        try:
            adv = attacker.run(x, y, sample_idx=sample_idx)
            print(f"样本 {sample_idx + 1} 攻击完成")
        except Exception as e:
            print(f"样本 {sample_idx + 1} 攻击失败: {e}")
            continue
    
    print("\n攻击过程图片保存完成！")
    
    # 7. 显示保存的图片结构
    process_dir = os.path.join(rm.img_dir, "attack_process")
    if os.path.exists(process_dir):
        print(f"\n图片保存在: {process_dir}")
        print("\n保存的图片结构:")
        for sample_dir in sorted(os.listdir(process_dir)):
            sample_path = os.path.join(process_dir, sample_dir)
            if os.path.isdir(sample_path):
                print(f"\n{sample_dir}:")
                for img_file in sorted(os.listdir(sample_path)):
                    print(f"  - {img_file}")

def example_targeted_attack_process():
    """示例：保存目标攻击过程图片"""
    
    print("\n=== 目标攻击过程保存示例 ===")
    
    # 初始化
    rm = ResultManager.get_instance()
    rm.set_experiment("AttackProcessExample")
    rm.set_test("targeted_demo")
    rm.enable_image_saving(True)
    
    # 创建数据加载器
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    # 定义ImageNet图片路径
    imagenet_base_path = "data/ImageNet/val"
    image_paths = [
        os.path.join(imagenet_base_path, "n04461696/ILSVRC2012_val_00029413.JPEG"),
        os.path.join(imagenet_base_path, "n03891332/ILSVRC2012_val_00034802.JPEG"),
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
    
    # 目标攻击配置
    config = {
        'patch_size': (32, 32),
        'blend_width': 4,
        'patch_k': 3,
        'cam_percentile': 0.8,
        'cam_thresh_ratio': 0.5,
        'mask_thresh_ratio': 0.3,
        'steps': 10,
        'eps': 0.1,
        'alpha': 0.01,
        'targeted_attack': True  # 启用目标攻击
    }
    
    # 创建攻击器
    attacker = MyPatchAttack(
        model=runner.classifier_model,
        device=runner.device,
        config=config,
        debug=True,
        save_imgs=False,
        save_attack_process=True
    )
    
    # 目标类别（CIFAR10类别：0=飞机, 1=汽车, 2=鸟, 3=猫, 4=鹿, 5=狗, 6=青蛙, 7=马, 8=船, 9=卡车）
    target_class = 0  # 将图片分类为"飞机"
    
    print(f"目标攻击：将图片分类为类别 {target_class} (飞机)")
    
    # 处理样本
    for sample_idx, (x, y) in enumerate(dataloader):
        print(f"\n处理目标攻击样本 {sample_idx + 1}/2")
        
        x = x.to(runner.device)
        y = y.to(runner.device)
        
        # 运行目标攻击
        try:
            adv = attacker.run(x, y, target=torch.tensor([target_class]).to(runner.device))
            print(f"目标攻击样本 {sample_idx + 1} 完成")
        except Exception as e:
            print(f"目标攻击样本 {sample_idx + 1} 失败: {e}")
            continue
    
    print("\n目标攻击过程图片保存完成！")

def show_usage_instructions():
    """显示使用说明"""
    
    print("\n" + "="*60)
    print("攻击过程保存功能使用说明")
    print("="*60)
    
    print("\n1. 功能说明:")
    print("   - 保存攻击方案的各个步骤图片")
    print("   - 包括：原始图片、语义分割、CAM图、补丁、融合结果、对抗样本")
    print("   - 每个样本创建独立的文件夹")
    print("   - 便于绘制攻击方案示意图")
    
    print("\n2. 使用方法:")
    print("   - 设置 save_attack_process=True")
    print("   - 运行攻击，自动保存过程图片")
    print("   - 图片保存在 Results/实验名/测试名/images/attack_process/")
    
    print("\n3. 保存的图片类型:")
    print("   - original_input.png: 原始输入图片")
    print("   - segmentation_*.png: 语义分割结果")
    print("   - cam_*.png: CAM注意力图")
    print("   - patches_*.png: 生成的补丁")
    print("   - blended_patched.png: 补丁融合结果")
    print("   - adversarial_final.png: 最终对抗样本")
    
    print("\n4. 注意事项:")
    print("   - 默认只保存第一个样本的图片")
    print("   - 不影响现有程序的执行")
    print("   - 可以通过设置 save_attack_process=False 关闭")
    print("   - 建议在调试或生成示意图时使用")

if __name__ == "__main__":
    print("攻击过程保存功能演示")
    print("="*40)
    
    # 运行示例
    example_save_attack_process()
    example_targeted_attack_process()
    show_usage_instructions()
    
    print("\n演示完成！") 