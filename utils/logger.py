"""
日志记录模块
用于记录训练过程中的损失、指标等信息
"""
import os
import json
import time
import logging
import numpy as np
from datetime import datetime

from config import LOG_DIR


class Logger:
    """
    训练日志记录器
    支持控制台输出、文件日志和TensorBoard
    """

    def __init__(self, log_dir=LOG_DIR, experiment_name=None, use_tensorboard=True):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = experiment_name

        # 设置日志文件
        self.log_file = os.path.join(log_dir, f"{experiment_name}.log")
        self.history_file = os.path.join(log_dir, f"{experiment_name}_history.json")

        # 初始化Python logging
        self._setup_logging()

        # TensorBoard
        self.writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                tb_dir = os.path.join(log_dir, "tensorboard", experiment_name)
                self.writer = SummaryWriter(tb_dir)
                self.info(f"TensorBoard日志: {tb_dir}")
            except Exception as e:
                self.info(f"TensorBoard不可用: {e}")

        self.epoch_metrics = {}

    def _setup_logging(self):
        """配置logging"""
        self.logger = logging.getLogger(self.experiment_name)
        self.logger.setLevel(logging.INFO)

        # 文件处理器
        file_handler = logging.FileHandler(self.log_file, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)

        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter("%(message)s")
        console_handler.setFormatter(console_formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def log_epoch(self, epoch, total_epochs, train_metrics, val_metrics, lr, elapsed):
        """记录单个epoch的训练结果"""
        self.info(
            f"\nEpoch {epoch + 1}/{total_epochs} | "
            f"训练损失: {train_metrics.get('loss', 0):.4f} | "
            f"训练准确率: {train_metrics.get('accuracy', 0):.4f} | "
            f"验证损失: {val_metrics.get('loss', 0):.4f} | "
            f"验证准确率: {val_metrics.get('accuracy', 0):.4f} | "
            f"学习率: {lr:.6f} | "
            f"耗时: {elapsed:.1f}s"
        )

        # 记录到TensorBoard
        if self.writer:
            self.writer.add_scalar("Loss/train", train_metrics.get("loss", 0), epoch)
            self.writer.add_scalar("Loss/val", val_metrics.get("loss", 0), epoch)
            self.writer.add_scalar("Accuracy/train", train_metrics.get("accuracy", 0), epoch)
            self.writer.add_scalar("Accuracy/val", val_metrics.get("accuracy", 0), epoch)
            self.writer.add_scalar("LR", lr, epoch)

            skip_keys = ["loss", "accuracy", "confusion_matrix", "report"]
            for key, val in train_metrics.items():
                if key not in skip_keys and not isinstance(val, str):
                    self.writer.add_scalar(f"Train/{key}", val, epoch)
            for key, val in val_metrics.items():
                if key not in skip_keys and not isinstance(val, str):
                    self.writer.add_scalar(f"Val/{key}", val, epoch)

    def log_final_results(self, test_metrics):
        """记录最终测试结果"""
        self.info("\n" + "=" * 60)
        self.info("                 最终测试结果")
        self.info("=" * 60)
        self.info(f"  测试准确率: {test_metrics.get('accuracy', 0):.4f}")
        self.info(f"  测试F1分数: {test_metrics.get('f1_macro', 0):.4f}")
        self.info(f"  测试AUC:    {test_metrics.get('auc', 0):.4f}")
        self.info("=" * 60)

        if self.writer:
            for key, val in test_metrics.items():
                if key not in ["confusion_matrix", "report"]:
                    self.writer.add_scalar(f"Test/{key}", val, 0)

    def save_history(self, history):
        """保存训练历史到JSON"""
        # 转换NumPy类型
        cleaned = {}
        for key, values in history.items():
            if key == "confusion_matrix":
                continue
            if isinstance(values, list):
                cleaned[key] = [float(v) if isinstance(v, (np.floating, float)) else v for v in values]
            else:
                cleaned[key] = values

        with open(self.history_file, "w", encoding="utf-8") as f:
            json.dump(cleaned, f, ensure_ascii=False, indent=2)

    def close(self):
        if self.writer:
            self.writer.close()
        for handler in self.logger.handlers:
            handler.close()
            self.logger.removeHandler(handler)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class ProgressBar:
    """
    训练进度条
    显示当前batch的损失和准确率
    """

    def __init__(self, total_batches, desc="训练中"):
        self.total_batches = total_batches
        self.desc = desc
        self.start_time = time.time()

    def update(self, batch_idx, loss, acc):
        """更新进度"""
        elapsed = time.time() - self.start_time
        progress = (batch_idx + 1) / self.total_batches
        bar_len = 30
        filled = int(bar_len * progress)
        bar = "█" * filled + "░" * (bar_len - filled)

        print(
            f"\r{self.desc} |{bar}| "
            f"{batch_idx + 1}/{self.total_batches} "
            f"[损失: {loss:.4f}, 准确率: {acc:.4f}] "
            f"耗时: {elapsed:.1f}s",
            end="", flush=True
        )

    def close(self):
        print()
