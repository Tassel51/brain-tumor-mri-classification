"""
配置文件：集中管理所有超参数和路径设置
"""
import os
import torch

# ==================== 路径配置 ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
LOG_DIR = os.path.join(BASE_DIR, "logs")

# ==================== 数据集配置 ====================
# 脑肿瘤MRI分类：glioma, meningioma, pituitary, notumor
NUM_CLASSES = 4
CLASS_NAMES = ["胶质瘤", "脑膜瘤", "垂体瘤", "正常"]
CLASS_NAMES_EN = ["glioma", "meningioma", "pituitary", "notumor"]

# 图像尺寸
IMG_SIZE = 224
IMG_CHANNELS = 3

# 数据集下载配置
DATASET_KAGGLE_ID = "sartajbhuvaji/brain-tumor-classification-mri"
DATASET_URL_FALLBACK = "https://github.com/SartajBhuvaji/Brain-Tumor-Classification-DataSet/archive/master.zip"

# ==================== 训练配置 ====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

# 训练超参数
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
MOMENTUM = 0.9

# 学习率调度
LR_SCHEDULER_STEP = 10
LR_SCHEDULER_GAMMA = 0.5

# 早停
EARLY_STOPPING_PATIENCE = 15
EARLY_STOPPING_MIN_DELTA = 1e-4

# 数据划分
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ==================== 数据增强配置 ====================
AUGMENTATION = {
    "random_rotation": 20,        # 随机旋转角度
    "random_horizontal_flip": True,
    "random_vertical_flip": False,
    "random_affine": 0.1,         # 随机仿射变换
    "color_jitter_brightness": 0.2,
    "color_jitter_contrast": 0.2,
    "random_erasing_prob": 0.25,  # Random Erasing概率
}

# ==================== 模型配置 ====================
# 可选模型: "custom_cnn", "resnet50", "resnet101"
MODEL_NAME = "resnet50"
PRETRAINED = True
FREEZE_BACKBONE = False  # 是否冻结骨干网络

# Dropout
DROPOUT_RATE = 0.5

# ==================== 日志与可视化 ====================
USE_TENSORBOARD = True
LOG_INTERVAL = 10  # 每N个batch打印一次日志
SAVE_BEST_ONLY = True
NUM_VISUALIZE = 16  # 可视化样本数量

# ==================== 创建必要目录 ====================
for dir_path in [DATA_DIR, PROCESSED_DIR, CHECKPOINT_DIR, RESULTS_DIR, LOG_DIR]:
    os.makedirs(dir_path, exist_ok=True)
