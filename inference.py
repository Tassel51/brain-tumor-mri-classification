"""
推理模块：对单张或多张医学图像进行预测
支持单图推理、批量推理和结果可视化
"""
import os
import glob
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms

from config import DEVICE, CLASS_NAMES, IMG_SIZE, CHECKPOINT_DIR
from data.preprocessing import preprocess_for_inference
from models.transfer import build_model


def load_model_for_inference(checkpoint_path=None, model_name="resnet50", device=DEVICE):
    """
    加载训练好的模型用于推理
    如果未指定检查点，使用未训练的模型（仅用于演示）
    """
    model, _ = build_model(model_name=model_name, pretrained=False)
    model = model.to(device)
    model.eval()

    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"模型已加载: {checkpoint_path}")
    else:
        print("警告: 未加载预训练权重，将使用随机初始化模型进行推理")

    return model


def preprocess_single_image(image_path, img_size=IMG_SIZE):
    """
    预处理单张图像
    返回: 预处理后的图像tensor
    """
    pil_image = preprocess_for_inference(image_path, img_size)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return transform(pil_image).unsqueeze(0)


def predict_image(model, image_tensor, device=DEVICE, topk=2):
    """
    对单张图像进行预测
    返回: (预测类别索引, 置信度, 所有类别概率)
    """
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)
        probabilities = F.softmax(output, dim=1)

        top_probs, top_indices = torch.topk(probabilities, k=min(topk, probabilities.size(1)))

    return (
        top_indices[0].cpu().numpy(),
        top_probs[0].cpu().numpy(),
        probabilities[0].cpu().numpy()
    )


def predict_batch(model, image_dir, device=DEVICE):
    """
    对目录中所有图像进行批量预测
    """
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(image_dir, ext)))
        image_paths.extend(glob.glob(os.path.join(image_dir, ext.upper())))

    if not image_paths:
        print(f"在 {image_dir} 中未找到图像文件")
        return []

    results = []
    for path in sorted(image_paths)[:100]:
        try:
            tensor = preprocess_single_image(path)
            top_indices, top_probs, all_probs = predict_image(model, tensor, device)

            results.append({
                "path": path,
                "filename": os.path.basename(path),
                "prediction": CLASS_NAMES[top_indices[0]],
                "confidence": top_probs[0],
                "all_probs": {CLASS_NAMES[i]: float(all_probs[i]) for i in range(len(CLASS_NAMES))},
            })
        except Exception as e:
            print(f"处理失败: {path} - {e}")

    return results


def print_prediction_result(result):
    """打印单张图像的预测结果"""
    print("\n" + "=" * 50)
    print(f"  文件: {result['filename']}")
    print(f"  预测类别: {result['prediction']}")
    print(f"  置信度: {result['confidence']:.4f} ({result['confidence'] * 100:.2f}%)")
    print("-" * 50)
    print("  各类别概率:")
    for cls_name, prob in sorted(result["all_probs"].items(), key=lambda x: -x[1]):
        bar_len = int(prob * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        print(f"    {cls_name:8s} |{bar}| {prob:.4f}")
    print("=" * 50)


def predict_single(model, image_path, device=DEVICE):
    """简便接口：对单张图像预测并打印结果"""
    tensor = preprocess_single_image(image_path)
    top_indices, top_probs, all_probs = predict_image(model, tensor, device)

    result = {
        "filename": os.path.basename(image_path),
        "prediction": CLASS_NAMES[top_indices[0]],
        "confidence": top_probs[0],
        "all_probs": {CLASS_NAMES[i]: float(all_probs[i]) for i in range(len(CLASS_NAMES))},
    }

    print_prediction_result(result)
    return result


if __name__ == "__main__":
    # 演示用
    print("脑肿瘤MRI辅助诊断 - 推理演示")
    print("=" * 60)
    print("使用方法:")
    print("  1. predict_single(模型, '图像路径') - 单图预测")
    print("  2. predict_batch(模型, '图像目录') - 批量预测")
