"""
数据集下载模块
支持从Kaggle、GitHub等多源下载，并提供合成数据生成作为兜底方案
"""
import os
import zipfile
import shutil
import requests
import numpy as np
from tqdm import tqdm
from config import DATA_DIR, DATASET_KAGGLE_ID, DATASET_URL_FALLBACK, IMG_SIZE


def download_with_kagglehub():
    """使用kagglehub下载脑肿瘤MRI数据集"""
    try:
        import kagglehub
        print("正在通过Kaggle下载脑肿瘤MRI数据集...")
        path = kagglehub.dataset_download(DATASET_KAGGLE_ID)
        print(f"下载完成: {path}")
        return path
    except Exception as e:
        print(f"kagglehub下载失败: {e}")
        return None


def download_with_requests():
    """从GitHub镜像下载数据集"""
    print("正在通过GitHub镜像下载数据集...")
    save_path = os.path.join(DATA_DIR, "dataset.zip")
    try:
        response = requests.get(DATASET_URL_FALLBACK, stream=True, timeout=30)
        total_size = int(response.headers.get("content-length", 0))
        with open(save_path, "wb") as f:
            with tqdm(total=total_size, unit="B", unit_scale=True, desc="下载中") as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
        print("下载完成")
        return save_path
    except Exception as e:
        print(f"GitHub下载失败: {e}")
        return None


def extract_and_organize(zip_path, extract_to):
    """解压并整理数据集为ImageFolder格式"""
    extract_dir = os.path.join(extract_to, "temp_extract")
    os.makedirs(extract_dir, exist_ok=True)

    print("正在解压数据集...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    # 查找数据集目录结构并整理
    organized_dir = os.path.join(extract_to, "organized")
    os.makedirs(organized_dir, exist_ok=True)

    # 查找包含类目录的文件夹（Training, Testing等）
    for root, dirs, files in os.walk(extract_dir):
        for d in dirs:
            if d.lower() in ["training", "testing", "train", "test", "val"]:
                src = os.path.join(root, d)
                dst = os.path.join(organized_dir, d.lower())
                if os.path.exists(src) and not os.path.exists(dst):
                    shutil.copytree(src, dst)
                    print(f"  找到数据目录: {d}")

        # 检查是否直接包含类别文件夹
        for class_name in ["glioma", "meningioma", "pituitary", "notumor"]:
            class_path = os.path.join(root, class_name)
            if os.path.isdir(class_path):
                # 根据父目录名确定用途
                parent = os.path.basename(root).lower()
                if parent in ["training", "train"]:
                    target = os.path.join(organized_dir, "train", class_name)
                elif parent in ["testing", "test"]:
                    target = os.path.join(organized_dir, "test", class_name)
                elif parent in ["validation", "val"]:
                    target = os.path.join(organized_dir, "val", class_name)
                else:
                    target = os.path.join(organized_dir, "train", class_name)
                os.makedirs(os.path.dirname(target), exist_ok=True)
                for fname in os.listdir(class_path):
                    shutil.copy2(os.path.join(class_path, fname), target)

    # 清理临时文件
    shutil.rmtree(extract_dir, ignore_errors=True)

    if os.path.exists(organized_dir):
        total = sum(len(files) for _, _, files in os.walk(organized_dir))
        print(f"数据集整理完成！共 {total} 张图片")
        return organized_dir
    return None


def generate_synthetic_data(output_dir, samples_per_class=50):
    """
    生成合成脑肿瘤MRI图像作为兜底方案
    使用高斯噪声和几何变换生成类MRI纹理的合成图像
    """
    print("正在生成合成医学图像数据...")
    classes = ["glioma", "meningioma", "pituitary", "notumor"]

    for split in ["train", "val", "test"]:
        if split == "train":
            n = samples_per_class
        elif split == "val":
            n = int(samples_per_class * 0.15)
        else:
            n = int(samples_per_class * 0.15)

        from PIL import Image

        for cls_name in classes:
            cls_dir = os.path.join(output_dir, split, cls_name)
            os.makedirs(cls_dir, exist_ok=True)

            for i in range(n):
                # 生成合成MRI图像
                img = np.random.randn(IMG_SIZE, IMG_SIZE) * 0.1
                # 添加椭圆状结构模拟脑部
                cy, cx = IMG_SIZE // 2, IMG_SIZE // 2
                Y, X = np.ogrid[:IMG_SIZE, :IMG_SIZE]
                mask = ((X - cx) ** 2 + (Y - cy) ** 2) < (IMG_SIZE * 0.4) ** 2
                img[mask] += 0.5

                # 添加随机纹理
                img += np.random.randn(IMG_SIZE, IMG_SIZE) * 0.05

                # 类别特定特征
                if cls_name == "glioma":
                    img[cy-20:cy+20, cx-15:cx+15] += 0.3  # 模拟肿瘤高亮区域
                elif cls_name == "meningioma":
                    img[cy-10:cy+10, cx-10:cx+10] += 0.4
                    img[cy-5:cy+5, cx-5:cx+5] += 0.2
                elif cls_name == "pituitary":
                    img[cy-8:cy+8, cx-8:cx+8] += 0.5

                # 归一化和转8位
                img = np.clip(img, 0, 1)
                img = (img * 255).astype(np.uint8)
                # 转为三通道
                img_rgb = np.stack([img, img, img], axis=-1)

                pil_img = Image.fromarray(img_rgb)
                pil_img.save(os.path.join(cls_dir, f"synthetic_{i:04d}.jpg"))

            print(f"  {split}/{cls_name}: {n} 张")

    total = sum(len(files) for _, _, files in os.walk(output_dir))
    print(f"合成数据生成完成！共 {total} 张图片")
    return output_dir


def prepare_dataset():
    """
    主入口：依次尝试多种方式获取数据集
    返回: 整理后的数据集路径，或None（使用合成数据）
    """
    organized_dir = os.path.join(DATA_DIR, "organized")

    # 如果已存在，直接返回
    if os.path.exists(organized_dir) and len(os.listdir(organized_dir)) > 0:
        print("数据集已存在，跳过下载")
        return organized_dir

    # 方法1: kagglehub
    kaggle_path = download_with_kagglehub()
    if kaggle_path:
        # kagglehub返回的是已下载路径，直接整理
        result = extract_and_organize_from_kaggle(kaggle_path, DATA_DIR)
        if result:
            return result

    # 方法2: GitHub requests下载
    zip_path = download_with_requests()
    if zip_path and os.path.exists(zip_path):
        result = extract_and_organize(zip_path, DATA_DIR)
        if result:
            os.remove(zip_path)
            return result

    # 方法3: 合成数据
    print("\n所有下载方式均失败，生成合成数据作为演示...")
    synthetic_dir = os.path.join(DATA_DIR, "synthetic")
    if os.path.exists(synthetic_dir):
        shutil.rmtree(synthetic_dir)
    result = generate_synthetic_data(synthetic_dir)
    return result


def extract_and_organize_from_kaggle(kaggle_path, output_base):
    """处理kagglehub下载的结果"""
    organized = os.path.join(output_base, "organized")
    class_mapping = {
        "glioma_tumor": "glioma",
        "meningioma_tumor": "meningioma",
        "pituitary_tumor": "pituitary",
        "no_tumor": "notumor",
        "glioma": "glioma",
        "meningioma": "meningioma",
        "pituitary": "pituitary",
        "notumor": "notumor",
    }
    splits = {"Training": "train", "Testing": "test"}

    os.makedirs(organized, exist_ok=True)

    for old_split, new_split in splits.items():
        split_path = None
        for root, dirs, files in os.walk(kaggle_path):
            if os.path.basename(root) == old_split:
                split_path = root
                break

        if split_path:
            for class_dir in os.listdir(split_path):
                class_path = os.path.join(split_path, class_dir)
                if os.path.isdir(class_path):
                    mapped = class_mapping.get(class_dir, class_dir)
                    target = os.path.join(organized, new_split, mapped)
                    os.makedirs(target, exist_ok=True)
                    for fname in os.listdir(class_path):
                        shutil.copy2(os.path.join(class_path, fname), target)

    if os.path.exists(organized) and len(os.listdir(organized)) > 0:
        total = sum(len(files) for _, _, files in os.walk(organized))
        print(f"数据集整理完成！共 {total} 张图片")
        return organized
    return None


if __name__ == "__main__":
    prepare_dataset()
