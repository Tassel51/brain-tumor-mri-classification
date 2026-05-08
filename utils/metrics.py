"""
评估指标计算模块
包含分类任务所需的各类评价指标
"""
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, classification_report
)


class AverageMeter:
    """跟踪平均值、当前值、总和、计数"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Evaluator:
    """模型评估器：计算分类任务各项指标"""

    def __init__(self, num_classes, class_names=None):
        self.num_classes = num_classes
        self.class_names = class_names or [str(i) for i in range(num_classes)]
        self.reset()

    def reset(self):
        self.all_preds = []
        self.all_labels = []
        self.all_probs = []
        self.loss_meter = AverageMeter()
        self.acc_meter = AverageMeter()

    def update(self, outputs, labels, loss=None):
        """更新一批预测结果"""
        _, preds = torch.max(outputs, 1)
        probs = torch.softmax(outputs, dim=1)

        self.all_preds.extend(preds.cpu().numpy())
        self.all_labels.extend(labels.cpu().numpy())
        self.all_probs.extend(probs.cpu().numpy())

        batch_acc = (preds == labels).float().mean().item()
        self.acc_meter.update(batch_acc, labels.size(0))

        if loss is not None:
            self.loss_meter.update(loss, labels.size(0))

    def compute_metrics(self):
        """计算所有评估指标"""
        y_true = np.array(self.all_labels)
        y_pred = np.array(self.all_preds)
        y_prob = np.array(self.all_probs)

        metrics = {}

        # 准确率
        metrics["accuracy"] = accuracy_score(y_true, y_pred)

        # 精确率、召回率、F1-score（宏平均和加权平均）
        metrics["precision_macro"] = precision_score(y_true, y_pred, average="macro", zero_division=0)
        metrics["recall_macro"] = recall_score(y_true, y_pred, average="macro", zero_division=0)
        metrics["f1_macro"] = f1_score(y_true, y_pred, average="macro", zero_division=0)

        metrics["precision_weighted"] = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        metrics["recall_weighted"] = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        metrics["f1_weighted"] = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        # 各类别F1-score
        per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
        for i, f1_val in enumerate(per_class_f1):
            metrics[f"f1_{self.class_names[i]}"] = f1_val

        # AUC-ROC（一对多）
        try:
            if self.num_classes == 2:
                metrics["auc"] = roc_auc_score(y_true, y_prob[:, 1])
            else:
                # 多分类AUC
                y_true_onehot = np.zeros((len(y_true), self.num_classes))
                for i, label in enumerate(y_true):
                    y_true_onehot[i, label] = 1
                metrics["auc"] = roc_auc_score(y_true_onehot, y_prob, multi_class="ovr")
        except Exception:
            metrics["auc"] = float("nan")

        # 混淆矩阵
        metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred)

        # 详细分类报告（文本格式，用于日志）
        metrics["report"] = classification_report(
            y_true, y_pred,
            target_names=self.class_names,
            zero_division=0,
            digits=4
        )

        # 平均损失
        metrics["loss"] = self.loss_meter.avg

        return metrics

    def print_report(self, metrics=None):
        """打印评估报告"""
        if metrics is None:
            metrics = self.compute_metrics()

        print("\n" + "=" * 60)
        print("                   模型评估报告")
        print("=" * 60)
        print(f"  准确率 (Accuracy):     {metrics['accuracy']:.4f}")
        print(f"  精确率 (Precision):    {metrics['precision_macro']:.4f} (macro)")
        print(f"  召回率 (Recall):       {metrics['recall_macro']:.4f} (macro)")
        print(f"  F1分数:               {metrics['f1_macro']:.4f} (macro)")
        print(f"  AUC-ROC:              {metrics.get('auc', 'N/A')}")
        print(f"  平均损失:              {metrics['loss']:.4f}")
        print("-" * 60)
        print("\n各类别指标:")
        for i, name in enumerate(self.class_names):
            print(f"  {name:12s}  F1={metrics.get(f'f1_{name}', 0):.4f}")
        print("-" * 60)
        print("\n分类报告:")
        print(metrics["report"])
        print("=" * 60)

        return metrics


class ConfusionMatrix:
    """混淆矩阵计算和统计"""

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, preds, labels):
        for p, l in zip(preds, labels):
            self.matrix[l, p] += 1

    def get_metrics(self):
        """从混淆矩阵计算各指标"""
        tp = np.diag(self.matrix)
        fp = self.matrix.sum(axis=0) - tp
        fn = self.matrix.sum(axis=1) - tp
        tn = self.matrix.sum() - (fp + fn + tp)

        # 避免除零
        sensitivity = tp / (tp + fn + 1e-10)  # 真正类率 / 召回率
        specificity = tn / (tn + fp + 1e-10)  # 真负类率
        precision = tp / (tp + fp + 1e-10)   # 精确率
        f1 = 2 * precision * sensitivity / (precision + sensitivity + 1e-10)

        return {
            "sensitivity": sensitivity,
            "specificity": specificity,
            "precision": precision,
            "f1_score": f1,
        }
