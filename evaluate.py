"""
评估模块：对训练好的模型进行全面评估
包含分类报告、混淆矩阵、ROC曲线、特征可视化等
"""
import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from config import DEVICE, CHECKPOINT_DIR, RESULTS_DIR, CLASS_NAMES, CLASS_NAMES_EN
from utils.metrics import Evaluator
from utils.visualization import Visualizer


def evaluate_model(model, test_loader, device=DEVICE, class_names=None, experiment_name=None):
    """
    全面评估模型性能
    生成混淆矩阵、ROC曲线、分类报告和可视化结果
    """
    if class_names is None:
        class_names = CLASS_NAMES

    visualizer = Visualizer()

    print("\n" + "=" * 60)
    print("                  模型全面评估")
    print("=" * 60)

    # 评估
    model.eval()
    evaluator = Evaluator(num_classes=len(class_names), class_names=CLASS_NAMES_EN)

    all_images = []
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="评估中"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            evaluator.update(outputs, labels)
            _, preds = torch.max(outputs, 1)

            all_images.append(inputs.cpu())
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(preds.cpu().numpy())

    # 计算指标
    metrics = evaluator.compute_metrics()
    evaluator.print_report(metrics)

    # 1. 混淆矩阵
    cm = metrics["confusion_matrix"]
    visualizer.plot_confusion_matrix(
        cm, class_names, normalize=False,
        filename=f"confusion_matrix_{experiment_name}.png"
    )
    visualizer.plot_confusion_matrix(
        cm, class_names, normalize=True,
        filename=f"confusion_matrix_norm_{experiment_name}.png",
        title="归一化混淆矩阵"
    )

    # 2. 样本预测可视化
    all_images_tensor = torch.cat(all_images, dim=0)
    sample_indices = np.random.choice(
        len(all_images_tensor),
        min(16, len(all_images_tensor)),
        replace=False
    )
    visualizer.plot_sample_predictions(
        all_images_tensor[sample_indices],
        np.array(all_labels)[sample_indices],
        np.array(all_predictions)[sample_indices],
        filename=f"sample_predictions_{experiment_name}.png",
        class_names=class_names,
    )

    # 3. ROC曲线
    visualizer.plot_roc_curves(
        np.array(evaluator.all_labels),
        np.array(evaluator.all_probs),
        class_names=class_names,
        filename=f"roc_curves_{experiment_name}.png"
    )

    # 4. 类别指标柱状图
    visualizer.plot_class_wise_metrics(
        metrics, class_names=class_names,
        filename=f"class_metrics_{experiment_name}.png"
    )

    print(f"\n所有评估图表已保存到: {RESULTS_DIR}")
    print("=" * 60)

    return metrics


def load_checkpoint(checkpoint_path, model, device=DEVICE):
    """加载模型检查点"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"检查点不存在: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    print(f"模型已加载: {checkpoint_path}")
    print(f"  训练Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  验证准确率: {checkpoint.get('best_val_acc', 'N/A'):.4f}")

    return model, checkpoint
