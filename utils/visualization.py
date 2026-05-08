"""
可视化工具模块
包含训练曲线、混淆矩阵、Grad-CAM热力图等可视化功能
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms

from config import CLASS_NAMES, CLASS_NAMES_EN, RESULTS_DIR


# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


class Visualizer:
    """可视化工具类"""

    def __init__(self, save_dir=RESULTS_DIR):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def plot_training_history(self, history, filename="training_history.png"):
        """
        绘制训练历史曲线
        history: {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "lr": []}
        """
        epochs = range(1, len(history["train_loss"]) + 1)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # 损失曲线
        ax = axes[0]
        ax.plot(epochs, history["train_loss"], "b-", linewidth=2, label="训练损失")
        ax.plot(epochs, history["val_loss"], "r-", linewidth=2, label="验证损失")
        if "test_loss" in history:
            ax.plot(epochs, history["test_loss"], "g--", linewidth=2, label="测试损失")
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("损失 (Loss)", fontsize=12)
        ax.set_title("训练与验证损失曲线", fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # 准确率曲线
        ax = axes[1]
        ax.plot(epochs, history["train_acc"], "b-", linewidth=2, label="训练准确率")
        ax.plot(epochs, history["val_acc"], "r-", linewidth=2, label="验证准确率")
        if "test_acc" in history:
            ax.plot(epochs, history["test_acc"], "g--", linewidth=2, label="测试准确率")
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("准确率 (Accuracy)", fontsize=12)
        ax.set_title("训练与验证准确率曲线", fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

        # 学习率曲线
        ax = axes[2]
        if "lr" in history and len(history["lr"]) > 0:
            ax.plot(epochs, history["lr"], "g-", linewidth=2, label="学习率")
            ax.set_xlabel("Epoch", fontsize=12)
            ax.set_ylabel("学习率 (LR)", fontsize=12)
            ax.set_title("学习率变化曲线", fontsize=14, fontweight="bold")
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, filename)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"训练曲线已保存: {save_path}")

    def plot_confusion_matrix(self, cm, class_names=None, normalize=False,
                              filename="confusion_matrix.png", title="混淆矩阵"):
        """
        绘制混淆矩阵
        cm: numpy数组，形状 (num_classes, num_classes)
        """
        if class_names is None:
            class_names = CLASS_NAMES

        if normalize:
            cm = cm.astype("float") / (cm.sum(axis=1, keepdims=True) + 1e-10)
            fmt = ".2f"
            vmax = 1.0
        else:
            fmt = "d"
            vmax = cm.max()

        fig, ax = plt.subplots(figsize=(8, 7))

        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=vmax)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # 标注数值
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                if normalize:
                    text = f"{cm[i, j]:.2f}"
                else:
                    text = f"{cm[i, j]:d}"
                ax.text(j, i, text,
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=11)

        ax.set(
            xticks=np.arange(len(class_names)),
            yticks=np.arange(len(class_names)),
            xticklabels=class_names,
            yticklabels=class_names,
            xlabel="预测类别 (Predicted)", ylabel="真实类别 (True)",
            title=title,
        )
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(class_names)

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, filename)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"混淆矩阵已保存: {save_path}")

    def plot_class_wise_metrics(self, metrics, class_names=None, filename="class_metrics.png"):
        """绘制各类别指标柱状图"""
        if class_names is None:
            class_names = CLASS_NAMES

        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(len(class_names))
        width = 0.25

        f1_scores = [metrics.get(f"f1_{name}", 0) for name in CLASS_NAMES_EN]
        precision = [metrics.get(f"precision_{name}", 0) for name in CLASS_NAMES_EN]
        recall = [metrics.get(f"recall_{name}", 0) for name in CLASS_NAMES_EN]

        # 如果不能直接获取，从报告的macro平均填充
        if all(v == 0 for v in f1_scores):
            f1_scores = [metrics.get("f1_macro", 0)] * len(class_names)

        ax.bar(x - width, f1_scores, width, label="F1分数", color="#2E86AB")
        ax.bar(x, precision or [0]*len(class_names), width, label="精确率", color="#A23B72")
        ax.bar(x + width, recall or [0]*len(class_names), width, label="召回率", color="#F18F01")

        ax.set_xlabel("类别", fontsize=12)
        ax.set_ylabel("分数", fontsize=12)
        ax.set_title("各类别评估指标", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, fontsize=10)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim(0, 1.1)

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, filename)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"类别指标图已保存: {save_path}")

    def plot_sample_predictions(self, images, true_labels, pred_labels, probs=None,
                                class_names=None, filename="sample_predictions.png"):
        """绘制样本预测结果"""
        if class_names is None:
            class_names = CLASS_NAMES

        n = min(len(images), 16)
        cols = 4
        rows = (n + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        axes = axes.flatten() if rows > 1 else [axes] if cols == 1 else axes

        for i in range(n):
            img = images[i]
            if isinstance(img, torch.Tensor):
                img = img.cpu().numpy().transpose(1, 2, 0)
                # 反标准化
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img = img * std + mean
                img = np.clip(img, 0, 1)

            true_name = class_names[true_labels[i]]
            pred_name = class_names[pred_labels[i]]
            color = "green" if true_labels[i] == pred_labels[i] else "red"

            title = f"真实: {true_name}\n预测: {pred_name}"
            if probs is not None:
                prob_val = probs[i][pred_labels[i]] if len(probs[i]) > pred_labels[i] else 0
                title += f"\n置信度: {prob_val:.2%}"

            axes[i].imshow(img)
            axes[i].set_title(title, fontsize=9, color=color)
            axes[i].axis("off")

        # 隐藏多余的子图
        for i in range(n, len(axes)):
            axes[i].axis("off")

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, filename)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"样本预测结果已保存: {save_path}")

    def plot_grad_cam(self, model, image_tensor, target_layer, class_idx=None,
                      filename="grad_cam.png"):
        """
        Grad-CAM可视化：生成类激活热力图
        需要模型和指定层
        """
        gradients = []
        activations = []

        def forward_hook(module, input, output):
            activations.append(output)

        def backward_hook(module, grad_in, grad_out):
            gradients.append(grad_out[0])

        hook_forward = target_layer.register_forward_hook(forward_hook)
        hook_backward = target_layer.register_full_backward_hook(backward_hook)

        # 前向传播
        model.eval()
        output = model(image_tensor.unsqueeze(0))

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        # 反向传播
        model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1
        output.backward(gradient=one_hot)

        # 计算Grad-CAM
        gradients = gradients[0].cpu().data.numpy()[0]
        activations = activations[0].cpu().data.numpy()[0]

        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = np.maximum(cam, 0)  # ReLU
        cam = cv2.resize(cam, (224, 224))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-10)

        # 叠加显示
        img = image_tensor.cpu().numpy().transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img * std + mean
        img = np.clip(img, 0, 1)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(img)
        axes[0].set_title("原始图像", fontsize=12)
        axes[0].axis("off")

        axes[1].imshow(cam, cmap="jet", alpha=1.0)
        axes[1].set_title("Grad-CAM 热力图", fontsize=12)
        axes[1].axis("off")

        axes[2].imshow(img)
        axes[2].imshow(cam, cmap="jet", alpha=0.5)
        axes[2].set_title(f"叠加: {CLASS_NAMES[class_idx]}", fontsize=12)
        axes[2].axis("off")

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, filename)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        hook_forward.remove()
        hook_backward.remove()

        print(f"Grad-CAM热力图已保存: {save_path}")

    def plot_feature_distribution(self, features, labels, class_names=None,
                                  filename="feature_distribution.png"):
        """
        特征分布可视化（t-SNE降维）
        """
        try:
            from sklearn.manifold import TSNE
        except ImportError:
            print("请安装scikit-learn以使用t-SNE可视化")
            return

        if class_names is None:
            class_names = CLASS_NAMES

        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        features_2d = tsne.fit_transform(features)

        fig, ax = plt.subplots(figsize=(10, 8))
        colors = ["#2E86AB", "#A23B72", "#F18F01", "#36A85A", "#E85D75", "#5D4E8C", "#F2A65A", "#4B7F52"]

        for i in range(len(class_names)):
            mask = labels == i
            ax.scatter(
                features_2d[mask, 0], features_2d[mask, 1],
                c=colors[i % len(colors)], label=class_names[i],
                alpha=0.7, edgecolors="white", linewidth=0.5, s=40
            )

        ax.set_title("特征分布可视化 (t-SNE)", fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("t-SNE维度1")
        ax.set_ylabel("t-SNE维度2")

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, filename)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"特征分布图已保存: {save_path}")

    def plot_roc_curves(self, y_true, y_prob, class_names=None, filename="roc_curves.png"):
        """绘制ROC曲线"""
        from sklearn.metrics import roc_curve, auc

        if class_names is None:
            class_names = CLASS_NAMES

        fig, ax = plt.subplots(figsize=(9, 7))

        # 一对多ROC
        from sklearn.preprocessing import label_binarize
        y_true_bin = label_binarize(y_true, classes=range(len(class_names)))

        colors = ["#2E86AB", "#A23B72", "#F18F01", "#36A85A"]

        for i in range(len(class_names)):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=colors[i % len(colors)],
                    label=f"{class_names[i]} (AUC = {roc_auc:.3f})",
                    linewidth=2)

        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="随机猜测")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("假正类率 (FPR)", fontsize=12)
        ax.set_ylabel("真正类率 (TPR)", fontsize=12)
        ax.set_title("ROC曲线 (一对多)", fontsize=14, fontweight="bold")
        ax.legend(fontsize=10, loc="lower right")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, filename)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"ROC曲线已保存: {save_path}")


# 全局可视化器实例
visualizer = Visualizer()


def plot_training_history(history, filename="training_history.png"):
    visualizer.plot_training_history(history, filename)


def plot_confusion_matrix(cm, class_names=None, normalize=False, filename="confusion_matrix.png"):
    visualizer.plot_confusion_matrix(cm, class_names, normalize, filename)


def plot_grad_cam(model, image_tensor, target_layer, class_idx=None, filename="grad_cam.png"):
    visualizer.plot_grad_cam(model, image_tensor, target_layer, class_idx, filename)
