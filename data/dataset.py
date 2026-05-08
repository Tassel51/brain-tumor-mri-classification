"""
数据集加载与预处理模块
"""
import os
import random
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms

import numpy as np

from config import (
    IMG_SIZE, BATCH_SIZE, CLASS_NAMES_EN, NUM_CLASSES,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO, SEED, AUGMENTATION,
    PROCESSED_DIR
)


class BrainTumorDataset(Dataset):
    """脑肿瘤MRI数据集"""

    def __init__(self, root_dir, transform=None, class_names=None):
        """
        Args:
            root_dir: 包含类别子文件夹的根目录
            transform: 图像变换
            class_names: 类别名称列表，用于确定类别映射
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted([d for d in os.listdir(root_dir)
                               if os.path.isdir(os.path.join(root_dir, d))])

        # 如果指定了class_names，按指定顺序
        if class_names is not None:
            self.classes = [c for c in class_names if c in self.classes]

        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.samples = []

        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')):
                    path = os.path.join(cls_dir, fname)
                    self.samples.append((path, self.class_to_idx[cls_name]))

        # 验证所有图片可读
        valid_samples = []
        for path, label in self.samples:
            try:
                with Image.open(path) as img:
                    img.verify()
                valid_samples.append((path, label))
            except Exception:
                continue
        self.samples = valid_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def get_train_transforms():
    """训练数据增强流水线"""
    aug = AUGMENTATION
    transforms_list = [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation(aug["random_rotation"]),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=aug["color_jitter_brightness"],
            contrast=aug["color_jitter_contrast"],
        ),
        transforms.RandomAffine(degrees=0, translate=(aug["random_affine"], aug["random_affine"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    # Random Erasing / Cutout
    if aug["random_erasing_prob"] > 0:
        transforms_list.append(
            transforms.RandomErasing(p=aug["random_erasing_prob"], scale=(0.02, 0.1))
        )

    return transforms.Compose(transforms_list)


def get_val_transforms():
    """验证/测试数据变换"""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def split_dataset(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """按比例划分数据集"""
    total = len(dataset)
    indices = list(range(total))

    # 按标签分层采样
    labels = [dataset.samples[i][1] for i in indices]
    from sklearn.model_selection import train_test_split

    # 先分出测试集
    train_val_idx, test_idx = train_test_split(
        indices, test_size=test_ratio, random_state=seed,
        stratify=labels, shuffle=True
    )

    # 从剩余中分出验证集
    val_size_relative = val_ratio / (train_ratio + val_ratio)
    train_val_labels = [labels[i] for i in train_val_idx]
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=val_size_relative, random_state=seed,
        stratify=train_val_labels, shuffle=True
    )

    return (
        Subset(dataset, train_idx),
        Subset(dataset, val_idx),
        Subset(dataset, test_idx),
    )


def create_dataloaders(data_root, batch_size=BATCH_SIZE, num_workers=0):
    """
    从数据根目录创建DataLoader
    如果数据已按 train/val/test 分开，直接加载
    否则自动划分
    """
    train_transform = get_train_transforms()
    val_transform = get_val_transforms()

    # 检查数据是否已分好 train/val/test
    splits_found = []
    for split in ["train", "val", "test"]:
        split_path = os.path.join(data_root, split)
        if os.path.isdir(split_path) and len(os.listdir(split_path)) > 0:
            splits_found.append(split)

    if "train" in splits_found:
        # 数据已按目录划分
        train_dataset = BrainTumorDataset(
            os.path.join(data_root, "train"),
            transform=train_transform,
            class_names=CLASS_NAMES_EN
        )
        val_dataset = BrainTumorDataset(
            os.path.join(data_root, "val") if "val" in splits_found
            else os.path.join(data_root, "train"),
            transform=val_transform,
            class_names=CLASS_NAMES_EN
        )
        test_dataset = BrainTumorDataset(
            os.path.join(data_root, "test") if "test" in splits_found
            else os.path.join(data_root, "val") if "val" in splits_found
            else os.path.join(data_root, "train"),
            transform=val_transform,
            class_names=CLASS_NAMES_EN
        )
    else:
        # 统一加载并划分
        full_dataset = BrainTumorDataset(
            data_root, transform=train_transform, class_names=CLASS_NAMES_EN
        )
        train_dataset, val_dataset, test_dataset = split_dataset(
            full_dataset, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, SEED
        )
        # 测试集用验证变换
        test_dataset.dataset.transform = val_transform
        val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    print(f"数据集加载完成:")
    print(f"  训练集: {len(train_dataset)} 张")
    print(f"  验证集: {len(val_dataset)} 张")
    print(f"  测试集: {len(test_dataset)} 张")
    print(f"  类别: {CLASS_NAMES_EN}")

    return train_loader, val_loader, test_loader


def get_class_weights(train_dataset):
    """计算类别权重（用于处理类别不平衡）"""
    labels = []
    if hasattr(train_dataset, 'dataset'):
        # Subset包装
        for idx in train_dataset.indices:
            labels.append(train_dataset.dataset.samples[idx][1])
    else:
        for _, label in train_dataset:
            labels.append(label)

    class_counts = np.bincount(labels, minlength=NUM_CLASSES)
    total = len(labels)
    weights = total / (NUM_CLASSES * class_counts)
    return torch.FloatTensor(weights)
