# 脑肿瘤MRI辅助诊断系统

基于 **PyTorch** 框架的计算机视觉深度学习项目，实现对脑肿瘤MRI图像的自动分类诊断。

## 项目概述

本系统对脑部MRI图像进行四分类：
| 类别 | 英文标识 | 说明 |
|------|----------|------|
| 胶质瘤 | glioma | 最常见的原发性脑肿瘤 |
| 脑膜瘤 | meningioma | 起源于脑膜的肿瘤 |
| 垂体瘤 | pituitary | 发生在垂体腺的肿瘤 |
| 正常 | notumor | 无肿瘤发现的脑部MRI |

<img width="1000" height="294" alt="image" src="https://github.com/user-attachments/assets/4e80c4a1-af19-4fd7-b226-b16a7bf53a5d" />

## 项目结构

```
脑肿瘤MRI辅助诊断系统/
├── main.py                 # 主入口（一键全流程）
├── config.py               # 配置文件（参数集中管理）
├── requirements.txt        # 依赖包列表
├── README.md               # 项目说明
├── data/
│   ├── __init__.py
│   ├── download.py         # 数据集下载（多源+合成兜底）
│   ├── dataset.py          # 数据集加载与预处理
│   └── preprocessing.py    # 图像预处理工具
├── models/
│   ├── __init__.py
│   ├── cnn.py              # 自定义CNN（多尺度+注意力机制）
│   └── transfer.py         # 迁移学习（ResNet/DenseNet等）
├── utils/
│   ├── __init__.py
│   ├── metrics.py          # 评估指标计算
│   ├── visualization.py    # 可视化（混淆矩阵/Grad-CAM/ROC等）
│   └── logger.py           # 日志记录
├── checkpoints/            # 模型检查点保存目录
└── results/                # 结果输出目录
```

## 环境要求

- Python 3.8+
- PyTorch 1.10+
- 支持CPU训练（推荐GPU加速）

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 一键运行（推荐）

自动完成：下载数据 → 训练 → 评估 全流程

```bash
python main.py --mode auto
```

### 3. 分步运行

**仅训练：**
```bash
python main.py --mode train
```

**仅评估：**
```bash
python main.py --mode evaluate
```

**单图像预测：**
```bash
python main.py --mode predict --image 图像路径.jpg
```

**批量预测：**
```bash
python main.py --mode batch_predict --image_dir ./图片目录
```

### 4. 高级选项

```bash
python main.py --mode auto --model resnet50 --epochs 30 --batch_size 16
```

## 模型架构

### 1. 自定义CNN (CustomCNN)
- 多尺度Inception模块
- SE注意力机制（Squeeze-and-Excitation）
- 深度可分离卷积
- 参数量：约 2.5M

### 2. 迁移学习模型 (TransferLearningModel)
- 支持 ResNet50/101、DenseNet121、EfficientNet 等
- 加载 ImageNet 预训练权重
- 可选冻结骨干网络进行迁移微调

## 评估指标

- Accuracy（准确率）
- Precision（精确率）
- Recall（召回率）
- F1-Score
- AUC-ROC
- Confusion Matrix（混淆矩阵）

## 可视化功能

- 训练损失/准确率曲线
- 混淆矩阵（原始/归一化）
- ROC曲线（一对多）
- 类别指标柱状图
- 样本预测结果展示
- Grad-CAM热力图
- t-SNE特征分布图

## 数据集

项目使用脑肿瘤MRI分类数据集（Brain Tumor MRI Classification），包含：
- 胶质瘤 (glioma)
- 脑膜瘤 (meningioma)
- 垂体瘤 (pituitary)
- 无肿瘤 (notumor)

**数据获取方式：**
1. Kaggle自动下载（首选，需配置Kaggle API）
2. GitHub镜像下载（备选）
3. 合成数据生成（兜底方案，确保程序可运行）

## 结果输出

训练完成后，在 `results/` 目录下会生成：
- `training_history.png` - 训练曲线
- `confusion_matrix.png` - 混淆矩阵
- `roc_curves.png` - ROC曲线
- `sample_predictions.png` - 样本预测结果
- `class_metrics.png` - 类别指标
- `feature_distribution.png` - 特征分布（t-SNE）

在 `checkpoints/` 目录下保存最佳模型参数。
