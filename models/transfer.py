"""
迁移学习模型：基于预训练ResNet/DenseNet的医学图像分类器
"""
import torch
import torch.nn as nn
import torchvision.models as models

from config import NUM_CLASSES, DROPOUT_RATE, FREEZE_BACKBONE


class TransferLearningModel(nn.Module):
    """
    基于预训练模型的迁移学习分类器
    支持: resnet50, resnet101, densenet121, efficientnet_b0

    使用策略:
    - 加载ImageNet预训练权重
    - 替换分类头为医学图像类别数
    - 可选择冻结骨干网络（仅训练分类头）
    """

    def __init__(self, model_name="resnet50", num_classes=NUM_CLASSES,
                 pretrained=True, dropout_rate=DROPOUT_RATE,
                 freeze_backbone=FREEZE_BACKBONE):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.backbone = self._build_backbone(model_name, pretrained)

        if freeze_backbone:
            self._freeze_backbone()

        self.num_features = self._get_num_features()
        self.classifier = self._build_classifier(dropout_rate)

    def _build_backbone(self, model_name, pretrained):
        weights = "DEFAULT" if pretrained else None

        if model_name == "resnet50":
            return models.resnet50(weights=weights)
        elif model_name == "resnet101":
            return models.resnet101(weights=weights)
        elif model_name == "resnet152":
            return models.resnet152(weights=weights)
        elif model_name == "densenet121":
            return models.densenet121(weights=weights)
        elif model_name == "densenet169":
            return models.densenet169(weights=weights)
        elif model_name == "efficientnet_b0":
            return models.efficientnet_b0(weights=weights)
        elif model_name == "efficientnet_b1":
            return models.efficientnet_b1(weights=weights)
        elif model_name == "convnext_tiny":
            return models.convnext_tiny(weights=weights)
        else:
            raise ValueError(f"不支持的模型: {model_name}")

    def _get_num_features(self):
        """获取骨干网络输出特征维度"""
        if "resnet" in self.model_name:
            return self.backbone.fc.in_features
        elif "densenet" in self.model_name:
            return self.backbone.classifier.in_features
        elif "efficientnet" in self.model_name:
            return self.backbone.classifier[1].in_features
        elif "convnext" in self.model_name:
            return self.backbone.classifier[2].in_features
        else:
            raise ValueError(f"未知模型: {self.model_name}")

    def _freeze_backbone(self):
        """冻结骨干网络参数（仅微调分类头）"""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def _build_classifier(self, dropout_rate):
        """构建分类头"""
        in_features = self._get_num_features()

        if "resnet" in self.model_name:
            return nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(in_features, self.num_classes),
            )
        elif "densenet" in self.model_name:
            return nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(in_features, self.num_classes),
            )
        elif "efficientnet" in self.model_name:
            return nn.Sequential(
                nn.Dropout(dropout_rate, inplace=True),
                nn.Linear(in_features, self.num_classes),
            )
        elif "convnext" in self.model_name:
            return nn.Sequential(
                nn.LayerNorm(in_features),
                nn.Dropout(dropout_rate),
                nn.Linear(in_features, self.num_classes),
            )
        else:
            return nn.Linear(in_features, self.num_classes)

    def forward(self, x):
        if "resnet" in self.model_name:
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)

            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x = self.backbone.layer3(x)
            x = self.backbone.layer4(x)

            x = self.backbone.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
        elif "densenet" in self.model_name:
            features = self.backbone.features(x)
            x = torch.flatten(features, 1)
            x = self.classifier(x)
        elif "efficientnet" in self.model_name:
            x = self.backbone.features(x)
            x = self.backbone.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
        elif "convnext" in self.model_name:
            x = self.backbone.features(x)
            x = self.backbone.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
        else:
            x = self.backbone(x)
            x = self.classifier(x)
        return x

    def get_feature_vector(self, x):
        """提取特征向量"""
        if "resnet" in self.model_name:
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)
            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x = self.backbone.layer3(x)
            x = self.backbone.layer4(x)
            x = self.backbone.avgpool(x)
            return torch.flatten(x, 1)
        else:
            return self.forward(x)

    def unfreeze_backbone(self):
        """解冻骨干网络（用于全模型微调）"""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("骨干网络已解冻")

    def get_trainable_params(self):
        """获取各部分的可训练参数量"""
        backbone_params = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        head_params = sum(p.numel() for p in self.classifier.parameters() if p.requires_grad)
        return backbone_params, head_params


def build_model(model_name="resnet50", num_classes=NUM_CLASSES,
                pretrained=True, freeze_backbone=False):
    """
    构建并返回模型

    返回: (model, params_str)
    """
    if model_name.startswith("custom"):
        from .cnn import CustomCNN
        model = CustomCNN(num_classes=num_classes)
    else:
        model = TransferLearningModel(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params_str = f"{model_name} | 总参数: {total_params:,} | 可训练: {trainable_params:,}"

    return model, params_str


if __name__ == "__main__":
    model, info = build_model("resnet50")
    print(info)

    dummy = torch.randn(1, 3, 224, 224)
    output = model(dummy)
    print(f"输入形状: {dummy.shape}")
    print(f"输出形状: {output.shape}")
