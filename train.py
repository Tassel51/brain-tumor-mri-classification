"""
训练模块：实现完整的训练循环
包含训练、验证、早停、学习率调度、模型保存等功能
"""
import os
import time
import copy
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR

from config import (
    DEVICE, EPOCHS, LEARNING_RATE, WEIGHT_DECAY, MOMENTUM,
    LR_SCHEDULER_STEP, LR_SCHEDULER_GAMMA,
    EARLY_STOPPING_PATIENCE, EARLY_STOPPING_MIN_DELTA,
    SAVE_BEST_ONLY, CHECKPOINT_DIR, MODEL_NAME, NUM_CLASSES,
)
from utils.metrics import AverageMeter, Evaluator
from utils.logger import Logger, ProgressBar
from utils.visualization import Visualizer


class EarlyStopping:
    """早停机制：当验证指标不再提升时停止训练"""

    def __init__(self, patience=10, min_delta=1e-4, mode="max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode  # "max" 或 "min"
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_state = None

        if mode == "max":
            self.best_score = float("-inf")
        else:
            self.best_score = float("inf")

    def __call__(self, current_score, model_state):
        if self.mode == "max":
            if current_score > self.best_score + self.min_delta:
                self.best_score = current_score
                self.best_state = copy.deepcopy(model_state)
                self.counter = 0
            else:
                self.counter += 1
        else:
            if current_score < self.best_score - self.min_delta:
                self.best_score = current_score
                self.best_state = copy.deepcopy(model_state)
                self.counter = 0
            else:
                self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True
            return True
        return False


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, logger):
    """
    训练一个epoch
    返回: (loss, accuracy)
    """
    model.train()
    evaluator = Evaluator(num_classes=NUM_CLASSES)

    pbar = ProgressBar(len(train_loader), f"Epoch {epoch + 1} 训练")

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        # 更新统计
        evaluator.update(outputs.detach(), labels.detach(), loss.item())

        if (batch_idx + 1) % 10 == 0 or batch_idx == len(train_loader) - 1:
            pbar.update(batch_idx, loss.item(), evaluator.acc_meter.avg)

    pbar.close()
    metrics = evaluator.compute_metrics()
    return metrics


@torch.no_grad()
def validate(model, val_loader, criterion, device, desc="验证"):
    """
    验证/测试一个epoch
    返回: (loss, accuracy, evaluator)
    """
    model.eval()
    evaluator = Evaluator(num_classes=NUM_CLASSES)

    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        evaluator.update(outputs, labels, loss.item())

    metrics = evaluator.compute_metrics()
    return metrics, evaluator


def train_model(
    model,
    train_loader,
    val_loader,
    test_loader=None,
    epochs=EPOCHS,
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    device=DEVICE,
    experiment_name=None,
    class_names=None,
):
    """
    完整训练流程
    返回: (训练好的模型, 训练历史)
    """
    logger = Logger(experiment_name=experiment_name)
    visualizer = Visualizer()

    logger.info(f"训练设备: {device}")
    logger.info(f"模型: {model.__class__.__name__}")
    logger.info(f"训练样本数: {len(train_loader.dataset)}")
    logger.info(f"验证样本数: {len(val_loader.dataset)}")
    if test_loader:
        logger.info(f"测试样本数: {len(test_loader.dataset)}")
    logger.info(f"学习率: {lr}, 权重衰减: {weight_decay}, Epochs: {epochs}")
    logger.info(f"Batch size: {train_loader.batch_size}")

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=LR_SCHEDULER_GAMMA,
        patience=max(5, EARLY_STOPPING_PATIENCE // 3),
        verbose=True, min_lr=1e-7
    )

    model = model.to(device)

    # 早停
    early_stopping = EarlyStopping(
        patience=EARLY_STOPPING_PATIENCE,
        min_delta=EARLY_STOPPING_MIN_DELTA,
        mode="max"
    )

    # 训练历史
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "test_loss": [], "test_acc": [],
        "lr": [], "best_epoch": 0,
    }

    best_val_acc = 0.0
    start_time = time.time()

    # 主训练循环
    for epoch in range(epochs):
        epoch_start = time.time()

        # 训练
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, logger
        )

        # 验证
        val_metrics, val_evaluator = validate(
            model, val_loader, criterion, device, "验证"
        )

        # 测试（如果提供）
        test_metrics = None
        if test_loader:
            test_metrics, _ = validate(
                model, test_loader, criterion, device, "测试"
            )

        # 学习率调度
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_metrics["accuracy"])

        # 记录历史
        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])
        if test_metrics:
            history["test_loss"].append(test_metrics["loss"])
            history["test_acc"].append(test_metrics["accuracy"])
        history["lr"].append(current_lr)

        # 记录日志
        elapsed = time.time() - epoch_start
        logger.log_epoch(
            epoch, epochs, train_metrics, val_metrics,
            current_lr, elapsed
        )

        # 保存最佳模型
        val_acc = val_metrics["accuracy"]
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            history["best_epoch"] = epoch + 1
            best_checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_acc": best_val_acc,
                "val_metrics": val_metrics,
                "config": {
                    "model_name": MODEL_NAME,
                    "num_classes": model.module.num_classes if hasattr(model, "module")
                    else model.classifier[-1].out_features if hasattr(model, "classifier")
                    else 4,
                    "lr": lr,
                    "epochs": epochs,
                }
            }
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"best_model_{experiment_name}.pth")
            torch.save(best_checkpoint, checkpoint_path)
            logger.info(f"最佳模型已保存: {checkpoint_path} (准确率: {best_val_acc:.4f})")

        # 早停检查
        if early_stopping(val_acc, model.state_dict()):
            logger.info(f"\n早停触发！{EARLY_STOPPING_PATIENCE}个epoch无提升。")
            break

    total_time = time.time() - start_time
    logger.info(f"\n训练完成！总耗时: {total_time:.1f}s ({total_time / 60:.1f}分钟)")
    logger.info(f"最佳验证准确率: {best_val_acc:.4f} (Epoch {history['best_epoch']})")

    # 绘制训练曲线
    visualizer.plot_training_history(history, f"training_history_{experiment_name}.png")

    # 保存历史
    logger.save_history(history)

    # 加载最佳模型参数
    model.load_state_dict(early_stopping.best_state)

    logger.close()
    return model, history
