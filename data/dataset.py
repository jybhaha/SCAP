from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
import os
import random

def get_dataloader(
    data_root='/home/jyb/0code/imagenet/train',
    batch_size=4,
    image_size=224,
    total_samples=None,  # 新增参数：随机选取图像数量
    seed=42              # 可复现性
):
    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor()
    ])

    dataset = ImageFolder(root=data_root, transform=transform)

    if total_samples is not None and total_samples < len(dataset):
        random.seed(seed)
        indices = random.sample(range(len(dataset)), total_samples)
        dataset = Subset(dataset, indices)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return dataloader


from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from pathlib import Path
import os
import random
from collections import defaultdict

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def build_category_subset_dataloader(
    data_root: str,
    target_classes: list,          # 目标类别文件夹名（如 synset ID：['n03797390', ...]）或你的自定义类文件夹名
    batch_size: int = 4,
    image_size: int = 224,
    shuffle: bool = True,
    num_workers: int = 4,
    max_per_class: int | None = None,   # 每类最多采样多少张（None 表示不限制）
    seed: int = 42,
    normalize: bool = True,
    train_aug: bool = False,            # 若是训练 loader，可开启随机增强
):
    """
    从 data_root 下按子文件夹名称过滤类别，返回该子集的 DataLoader。
    目录需形如: data_root/<class_folder>/*.jpg
    对 ImageNet：data_root 可是 train 或 val 目录，<class_folder> 通常是 synset ID。
    """
    assert os.path.isdir(data_root), f"data_root not found: {data_root}"
    random.seed(seed)

    # 预处理
    if train_aug:
        tfm = T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.08, 1.0), ratio=(3/4, 4/3)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD) if normalize else T.Lambda(lambda x: x),
        ])
    else:
        tfm = T.Compose([
            T.Resize(int(image_size * 256 / 224)),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD) if normalize else T.Lambda(lambda x: x),
        ])

    dataset_all = ImageFolder(root=data_root, transform=tfm)
    # ImageFolder.classes 为按字典序的子目录（类名）
    class_to_idx = {c: i for i, c in enumerate(dataset_all.classes)}

    # 检查目标类是否存在
    missing = [c for c in target_classes if c not in class_to_idx]
    if missing:
        raise ValueError(f"这些类别文件夹在 {data_root} 下不存在: {missing}")

    target_idx = set(class_to_idx[c] for c in target_classes)

    # 收集样本索引
    selected = []
    if max_per_class is None:
        selected = [i for i, (_, y) in enumerate(dataset_all.samples) if y in target_idx]
    else:
        per_class = defaultdict(list)
        for i, (_, y) in enumerate(dataset_all.samples):
            if y in target_idx:
                per_class[y].append(i)
        for y, idxs in per_class.items():
            random.shuffle(idxs)
            selected.extend(idxs[:max_per_class])

    subset = Subset(dataset_all, selected)

    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        drop_last=False
    )
    return loader, subset, class_to_idx