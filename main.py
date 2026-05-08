"""
脑肿瘤MRI辅助诊断系统 - 主入口
基于PyTorch的医学图像分类系统

功能：
  1. 数据集下载与准备
  2. 模型训练（支持自定义CNN和迁移学习）
  3. 模型评估（混淆矩阵、ROC曲线、Grad-CAM等）
  4. 单张图像预测
  5. 批量图像预测

用法：
  python main.py --mode train          # 完整训练+评估
  python main.py --mode evaluate       # 仅评估已有模型
  python main.py --mode predict --image path/to/image.jpg  # 单图预测
  python main.py --mode batch_predict --image_dir path/to/dir  # 批量预测
  python main.py --mode auto           # 一键全流程（推荐）
"""
import os
import sys
import argparse
import random
import numpy as np

import torch

from config import (
    DEVICE, SEED, CLASS_NAMES, CLASS_NAMES_EN, CHECKPOINT_DIR, RESULTS_DIR
)
import config


def set_seed(seed=SEED):
    """设置所有随机种子以保证可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_system_info():
    """打印系统信息"""
    print("=" * 60)
    print("        脑肿瘤MRI辅助诊断系统 (PyTorch)")
    print("=" * 60)
    print(f"  PyTorch版本: {torch.__version__}")
    print(f"  运行设备: {DEVICE}")
    if torch.cuda.is_available():
        print(f"  GPU型号: {torch.cuda.get_device_name(0)}")
        print(f"  GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"  模型: {config.MODEL_NAME}")
    print(f"  类别: {CLASS_NAMES}")
    print("=" * 60)


def download_data():
    """下载并准备数据集"""
    from data.download import prepare_dataset
    print("\n>>> 步骤 1/5: 准备数据集...")
    data_root = prepare_dataset()
    return data_root


def create_dataloaders(data_root):
    """创建数据加载器"""
    from data.dataset import create_dataloaders
    print(f"\n>>> 步骤 2/5: 加载数据...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_root, batch_size=config.BATCH_SIZE
    )
    return train_loader, val_loader, test_loader


def build_model():
    """构建模型"""
    from models.transfer import build_model
    print(f"\n>>> 步骤 3/5: 构建模型...")
    model, info = build_model(
        model_name=config.MODEL_NAME,
        pretrained=True,
        freeze_backbone=False,
    )
    print(f"  模型信息: {info}")
    return model


def train(model, train_loader, val_loader, test_loader):
    """训练模型"""
    from train import train_model
    print(f"\n>>> 步骤 4/5: 开始训练...")
    trained_model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=config.EPOCHS,
        device=DEVICE,
        experiment_name=f"{config.MODEL_NAME}_epochs{config.EPOCHS}",
    )
    return trained_model, history


def evaluate(model, test_loader):
    """评估模型"""
    from evaluate import evaluate_model
    print(f"\n>>> 步骤 5/5: 全面评估...")
    metrics = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=DEVICE,
        class_names=CLASS_NAMES,
        experiment_name=f"{config.MODEL_NAME}_epochs{config.EPOCHS}",
    )
    return metrics


def run_auto():
    """一键全流程"""
    set_seed()
    print_system_info()

    # 1. 准备数据
    data_root = download_data()

    # 2. 加载数据
    train_loader, val_loader, test_loader = create_dataloaders(data_root)

    # 3. 构建模型
    model = build_model()

    # 4. 训练
    trained_model, history = train(model, train_loader, val_loader, test_loader)

    # 5. 评估
    metrics = evaluate(trained_model, test_loader)

    print("\n" + "=" * 60)
    print("              全流程完成！")
    print("=" * 60)
    print(f"  最终测试准确率: {metrics.get('accuracy', 0):.4f}")
    print(f"  最终测试F1分数: {metrics.get('f1_macro', 0):.4f}")
    print(f"  所有结果已保存至: {RESULTS_DIR}")
    print(f"  最佳模型已保存至: {CHECKPOINT_DIR}")
    print("=" * 60)


def run_train_only():
    """仅训练"""
    set_seed()
    print_system_info()

    data_root = download_data()
    train_loader, val_loader, test_loader = create_dataloaders(data_root)
    model = build_model()
    trained_model, history = train(model, train_loader, val_loader, test_loader)
    metrics = evaluate(trained_model, test_loader)

    print(f"\n训练完成！最佳模型保存在: {CHECKPOINT_DIR}")


def run_evaluate_only():
    """仅评估"""
    from evaluate import evaluate_model, load_checkpoint
    from data.dataset import create_dataloaders
    from data.download import prepare_dataset
    from models.transfer import build_model

    set_seed()
    print_system_info()

    data_root = prepare_dataset()
    _, _, test_loader = create_dataloaders(data_root)

    model, _ = build_model(model_name=config.MODEL_NAME, pretrained=False)

    # 寻找最佳检查点
    checkpoint_files = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pth")]
    if checkpoint_files:
        latest = sorted(checkpoint_files)[-1]
        checkpoint_path = os.path.join(CHECKPOINT_DIR, latest)
        model, _ = load_checkpoint(checkpoint_path, model)
    else:
        print("未找到检查点，使用未训练的模型进行评估")

    metrics = evaluate_model(
        model, test_loader, device=DEVICE,
        class_names=CLASS_NAMES,
    )


def run_predict(image_path):
    """单张图像预测"""
    from inference import load_model_for_inference, predict_single

    set_seed()

    # 寻找最佳检查点
    checkpoint_path = None
    checkpoint_files = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pth")]
    if checkpoint_files:
        latest = sorted(checkpoint_files)[-1]
        checkpoint_path = os.path.join(CHECKPOINT_DIR, latest)

    model = load_model_for_inference(checkpoint_path, config.MODEL_NAME, DEVICE)
    result = predict_single(model, image_path, DEVICE)


def run_batch_predict(image_dir):
    """批量预测"""
    from inference import load_model_for_inference, predict_batch

    set_seed()

    checkpoint_path = None
    checkpoint_files = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pth")]
    if checkpoint_files:
        latest = sorted(checkpoint_files)[-1]
        checkpoint_path = os.path.join(CHECKPOINT_DIR, latest)

    model = load_model_for_inference(checkpoint_path, config.MODEL_NAME, DEVICE)
    results = predict_batch(model, image_dir, DEVICE)

    print(f"\n共预测 {len(results)} 张图像")
    for r in results:
        print(f"  {r['filename']:30s} -> {r['prediction']:8s} (置信度: {r['confidence']:.4f})")


def main():
    parser = argparse.ArgumentParser(
        description="脑肿瘤MRI辅助诊断系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py --mode auto                     # 一键全流程
  python main.py --mode train                     # 仅训练
  python main.py --mode evaluate                  # 仅评估
  python main.py --mode predict --image demo.jpg  # 单图预测
  python main.py --mode batch_predict --image_dir ./images  # 批量预测
        """
    )
    parser.add_argument(
        "--mode", type=str, default="auto",
        choices=["auto", "train", "evaluate", "predict", "batch_predict"],
        help="运行模式 (默认: auto)"
    )
    parser.add_argument("--image", type=str, help="预测的图像路径")
    parser.add_argument("--image_dir", type=str, help="批量预测的图像目录")
    parser.add_argument("--model", type=str, default=config.MODEL_NAME,
                        help=f"模型名称 (默认: {config.MODEL_NAME})")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS,
                        help=f"训练轮数 (默认: {config.EPOCHS})")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE,
                        help=f"批次大小 (默认: {config.BATCH_SIZE})")
    parser.add_argument("--seed", type=int, default=config.SEED,
                        help=f"随机种子 (默认: {config.SEED})")

    args = parser.parse_args()

    # 全局更新配置
    config.MODEL_NAME = args.model
    config.EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size
    config.SEED = args.seed

    # 执行模式
    if args.mode == "auto":
        run_auto()
    elif args.mode == "train":
        run_train_only()
    elif args.mode == "evaluate":
        run_evaluate_only()
    elif args.mode == "predict":
        if not args.image:
            print("错误: 预测模式需要指定 --image 参数")
            sys.exit(1)
        run_predict(args.image)
    elif args.mode == "batch_predict":
        if not args.image_dir:
            print("错误: 批量预测模式需要指定 --image_dir 参数")
            sys.exit(1)
        run_batch_predict(args.image_dir)


if __name__ == "__main__":
    main()
