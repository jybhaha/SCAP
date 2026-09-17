"""
数据加载器模块
封装数据加载和预处理逻辑
"""

import torch
from data.dataset import get_dataloader, build_category_subset_dataloader
from pathlib import Path

# 可选：常用类别的快捷清单（以 ImageNet-1K synset 为例，需与你的数据目录一致）
HOUSEHOLD_SYNSETS = [
    "n03797390","n03147509","n03942813","n02880940","n04254120","n03314780",
    "n03041632","n04398044","n04070727","n03761084","n04442312","n04517823",
    "n03636649","n03196217","n04522168",
]
TOOLS_SYNSETS = [
    "n03481172","n04154565","n04553703","n03995372","n03499033","n03000684",
]
VEHICLES_SYNSETS = [
    "n02958343","n04285008","n03770679","n03930630","n02924116","n04467665",
    "n02834778","n03790512","n04335435","n04468005","n02690373","n04266014",
    "n04273569","n04147183","n04347754",
]
SPORTS_SYNSETS = [
    "n04254680","n02802426","n04540053","n04409515","n04456115","n03445777",
    "n02883205","n02787622","n04225987","n04228054","n04336792","n03124170",
]

CATEGORY_PRESETS = {
    "household": HOUSEHOLD_SYNSETS,
    "tools": TOOLS_SYNSETS,
    "vehicles": VEHICLES_SYNSETS,
    "sports": SPORTS_SYNSETS,
}


class DataLoaderManager:
    """数据加载器管理类"""
    
    def __init__(self, data_root='data/ImageNet/train'):
        self.data_root = data_root
        self.train_root = data_root
        self.val_root = data_root.replace('/train', '/val')  # 添加验证集路径
        
    def get_dataloader(self, batch_size=4, image_size=224, total_samples=10):
        """获取数据加载器"""
        return get_dataloader(
            data_root=self.data_root,
            batch_size=batch_size,
            image_size=image_size,
            total_samples=total_samples
        )
    
    def get_test_dataloader(self, batch_size=4, image_size=224, total_samples=100):
        """获取测试数据加载器"""
        return get_dataloader(
            data_root=self.data_root,
            batch_size=batch_size,
            image_size=image_size,
            total_samples=total_samples
        )
    
    def get_small_dataloader(self, batch_size=2, image_size=224, total_samples=5):
        """获取小规模数据加载器（用于快速测试）"""
        return get_dataloader(
            data_root=self.data_root,
            batch_size=batch_size,
            image_size=image_size,
            total_samples=total_samples
        )
    
    def get_objects_dataloader(
        self,
        categories: list | str = ("household", "tools", "vehicles", "sports"),
        split: str = "train",               # "train" 或 "val"
        batch_size: int = 64,
        image_size: int = 224,
        shuffle: bool = True,
        max_per_class: int | None = None,   # 每类采样上限；评估可设置为固定值保证均衡
        seed: int = 42,
        normalize: bool = True,
        train_aug: bool = False,
        custom_class_folders: list | None = None,  # 若你不是 synset 目录，可传自定义类文件夹名列表
    ):
        """
        获取"物体类别子集" DataLoader：
        - categories 可为字符串（单个预设）或列表（多个预设名），如 "tools" 或 ["tools","vehicles"]
        - 或者使用 custom_class_folders 直接传入目录名列表（优先级高于 categories）
        """
        # 确定数据目录
        if split not in ("train", "val"):
            raise ValueError("split must be 'train' or 'val'")
        data_root = self.train_root if split == "train" else self.val_root

        # 汇总目标类别文件夹
        if custom_class_folders is not None:
            target_classes = list(custom_class_folders)
        else:
            if isinstance(categories, str):
                categories = [categories]
            target_classes = []
            for cat in categories:
                if cat not in CATEGORY_PRESETS:
                    raise ValueError(f"未知类别预设: {cat}. 可选: {list(CATEGORY_PRESETS.keys())}")
                target_classes.extend(CATEGORY_PRESETS[cat])
        
        loader, subset, class_to_idx = build_category_subset_dataloader(
            data_root=data_root,
            target_classes=target_classes,
            batch_size=batch_size,
            image_size=image_size,
            shuffle=shuffle,
            num_workers=4,
            max_per_class=max_per_class,
            seed=seed,
            normalize=normalize,
            train_aug=train_aug if split == "train" else False,
        )
        return loader, subset, class_to_idx


def get_experiment_dataloader(experiment_type="default"):
    """根据实验类型获取相应的数据加载器"""
    manager = DataLoaderManager()
    
    dataloader_configs = {
        "default": {"batch_size": 4, "total_samples": 10},
        "medium": {"batch_size": 4, "total_samples": 100},
        "small": {"batch_size": 2, "total_samples": 5},
        "large": {"batch_size": 4, "total_samples": 1000},
    }
    
    config = dataloader_configs.get(experiment_type, dataloader_configs["default"])
    return manager.get_dataloader(**config)


def get_objects_experiment_dataloader(experiment_type="default", categories="household"):
    """根据实验类型获取物体类别数据加载器"""
    manager = DataLoaderManager()
    
    dataloader_configs = {
        "default": {"batch_size": 4, "max_per_class": 10},
        "medium": {"batch_size": 4, "max_per_class": 50},
        "small": {"batch_size": 2, "max_per_class": 1},
        "large": {"batch_size": 4, "max_per_class": 200},
    }
    
    config = dataloader_configs.get(experiment_type, dataloader_configs["default"])
    
    # 获取物体类别数据加载器
    loader, subset, class_to_idx = manager.get_objects_dataloader(
        categories=categories,
        split="train",
        batch_size=config["batch_size"],
        image_size=224,
        shuffle=True,
        max_per_class=config["max_per_class"],
        seed=42,
        normalize=True,
        train_aug=False
    )
    
    return loader, subset, class_to_idx