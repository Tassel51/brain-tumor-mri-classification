"""
自定义CNN模型：从零构建的卷积神经网络
适用于脑肿瘤MRI图像分类
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """卷积块: Conv2d + BatchNorm + ReLU + Dropout"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 dropout_rate=0.25, use_bn=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
        ]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        if dropout_rate > 0:
            layers.append(nn.Dropout2d(dropout_rate))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation注意力模块"""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class InceptionBlock(nn.Module):
    """简化版Inception模块：多尺度特征提取"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid = out_channels // 4
        # 1x1分支
        self.branch1 = ConvBlock(in_channels, mid, kernel_size=1, padding=0)
        # 3x3分支
        self.branch2 = nn.Sequential(
            ConvBlock(in_channels, mid, kernel_size=1, padding=0),
            ConvBlock(mid, mid, kernel_size=3, padding=1),
        )
        # 5x5分支
        self.branch3 = nn.Sequential(
            ConvBlock(in_channels, mid, kernel_size=1, padding=0),
            ConvBlock(mid, mid, kernel_size=5, padding=2),
        )
        # 3x3池化分支
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels, mid, kernel_size=1, padding=0),
        )

    def forward(self, x):
        return torch.cat([
            self.branch1(x), self.branch2(x),
            self.branch3(x), self.branch4(x)
        ], dim=1)


class CustomCNN(nn.Module):
    """
    自定义CNN：结合多尺度卷积和注意力机制
    适用于224x224的医学图像

    结构概要:
    1. 初始卷积层 (3->64)
    2. Inception块 (64->128)
    3. SE注意力 + 深度卷积 (128->256)
    4. Inception块 (256->512)
    5. SE注意力 (512->512)
    6. 全局平均池化 + 分类头
    """

    def __init__(self, num_classes=4, dropout_rate=0.5):
        super().__init__()

        # 初始卷积
        self.initial = nn.Sequential(
            ConvBlock(3, 64, kernel_size=7, stride=2, padding=3, dropout_rate=0),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # 特征提取阶段
        self.stage1 = nn.Sequential(
            InceptionBlock(64, 128),
            ConvBlock(128, 128, kernel_size=1, padding=0, dropout_rate=0.25),
        )

        self.stage2 = nn.Sequential(
            ConvBlock(128, 256, kernel_size=3, stride=2, padding=1, dropout_rate=0.25),
            SEBlock(256),
            ConvBlock(256, 256, kernel_size=3, padding=1, dropout_rate=0.25),
            ConvBlock(256, 256, kernel_size=1, padding=0, dropout_rate=0),
        )

        self.stage3 = nn.Sequential(
            InceptionBlock(256, 512),
            ConvBlock(512, 512, kernel_size=1, padding=0, dropout_rate=0.25),
            SEBlock(512),
        )

        # 全局平均池化
        self.gap = nn.AdaptiveAvgPool2d(1)

        # 分类头
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate * 0.8),
            nn.Linear(256, num_classes),
        )

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.initial(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)

        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def get_feature_vector(self, x):
        """提取特征向量（用于可视化或分析）"""
        x = self.initial(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.gap(x)
        return x.view(x.size(0), -1)


def count_parameters(model):
    """统计模型参数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    model = CustomCNN(num_classes=4)
    total, trainable = count_parameters(model)
    print(f"CustomCNN 总参数量: {total:,}")
    print(f"CustomCNN 可训练参数量: {trainable:,}")

    dummy = torch.randn(1, 3, 224, 224)
    output = model(dummy)
    print(f"输入形状: {dummy.shape}")
    print(f"输出形状: {output.shape}")
