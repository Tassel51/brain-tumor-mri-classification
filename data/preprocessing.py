"""
数据预处理与增强工具
"""
import cv2
import numpy as np
from PIL import Image


def apply_clahe(image):
    """应用CLAHE增强对比度（适用于医学图像）"""
    if isinstance(image, Image.Image):
        image = np.array(image)

    if len(image.shape) == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    else:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        result = clahe.apply(image)

    return Image.fromarray(result)


def preprocess_for_inference(image_path, img_size=224):
    """
    对单张推理图像进行预处理
    返回: 预处理后的PIL图像
    """
    image = Image.open(image_path).convert("RGB")
    image = apply_clahe(image)

    # 保持宽高比resize
    w, h = image.size
    if w > h:
        new_w = img_size
        new_h = int(h * img_size / w)
    else:
        new_h = img_size
        new_w = int(w * img_size / h)
    image = image.resize((new_w, new_h), Image.BILINEAR)

    # 中心裁剪
    left = (new_w - img_size) / 2
    top = (new_h - img_size) / 2
    image = image.crop((left, top, left + img_size, top + img_size))

    return image
