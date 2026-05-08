"""
模型模块：包含自定义CNN和基于迁移学习的分类模型
"""
from .cnn import CustomCNN
from .transfer import TransferLearningModel

__all__ = ["CustomCNN", "TransferLearningModel"]
