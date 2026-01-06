"""
MVTec AD 数据集下载脚本

MVTec AD 是工业异常检测的标准基准数据集，包含15个类别，共5000+张高分辨率图像。
每个类别包含正常训练图像和带有各种缺陷的测试图像。

官方网站: https://www.mvtec.com/company/research/datasets/mvtec-ad

下载方式:
1. 使用anomalib自动下载 (推荐)
2. 手动从官网下载后放到 data/ 目录
"""

import os
import sys
import tarfile
from pathlib import Path
from tqdm import tqdm

# 15个类别
CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper"
]


def download_via_huggingface(data_dir: str = "data") -> Path:
    """
    使用Hugging Face下载MVTec AD数据集
    """
    print("使用 Hugging Face 下载 MVTec AD 数据集...")
    print("提示: 请确保已设置代理:")
    print("  export http_proxy=http://127.0.0.1:7890")
    print("  export https_proxy=http://127.0.0.1:7890")
    print()

    try:
        import subprocess
        # 先安装datasets库
        subprocess.run(["uv", "pip", "install", "datasets", "fiftyone",
                       "-i", "https://pypi.tuna.tsinghua.edu.cn/simple"],
                      check=True, capture_output=True)

        import fiftyone.zoo as foz

        # 下载数据集
        dataset = foz.load_zoo_dataset(
            "mvtec-ad",
            split="train",
            dataset_dir=data_dir,
        )

        dataset_path = Path(data_dir) / "mvtec-ad"
        if dataset_path.exists():
            print(f"\n数据集下载成功! 路径: {dataset_path}")
            return dataset_path

    except Exception as e:
        print(f"Hugging Face 下载失败: {e}")
        return None


def download_via_anomalib(data_dir: str = "data") -> Path:
    """
    使用anomalib下载MVTec AD数据集

    这是推荐的方式，anomalib会处理下载和解压。
    """
    print("使用 anomalib 下载 MVTec AD 数据集...")
    print("提示: 请确保已设置代理:")
    print("  export http_proxy=http://127.0.0.1:7890")
    print("  export https_proxy=http://127.0.0.1:7890")
    print()

    try:
        from anomalib.data import MVTecAD

        # 创建数据模块，会自动下载
        datamodule = MVTecAD(
            root=data_dir,
            category="bottle",  # 下载任意一个类别会下载整个数据集
            train_batch_size=1,
            eval_batch_size=1,
        )
        datamodule.prepare_data()

        dataset_path = Path(data_dir) / "MVTec"
        if dataset_path.exists():
            print(f"\n数据集下载成功! 路径: {dataset_path}")
            return dataset_path
        else:
            # anomalib可能使用不同的路径
            for possible_path in [
                Path(data_dir) / "mvtec_anomaly_detection",
                Path(data_dir) / "MVTec",
                Path(data_dir) / "mvtec",
            ]:
                if possible_path.exists():
                    print(f"\n数据集下载成功! 路径: {possible_path}")
                    return possible_path

    except Exception as e:
        print(f"anomalib 下载失败: {e}")
        print("\n请尝试手动下载...")
        return None


def extract_tar_xz(tar_path: Path, extract_to: Path) -> None:
    """解压 tar.xz 文件"""
    print(f"正在解压 {tar_path.name}...")
    with tarfile.open(tar_path, 'r:xz') as tar:
        members = tar.getmembers()
        for member in tqdm(members, desc="解压中"):
            tar.extract(member, extract_to)
    print("解压完成!")


def manual_download_instructions():
    """显示手动下载说明"""
    print("""
================================================================================
                         MVTec AD 数据集下载指南
================================================================================

官方下载链接已失效，请使用以下备选方式:

方式1: Hugging Face (推荐)
   pip install datasets
   然后运行: python scripts/download_dataset.py --huggingface

方式2: Kaggle
   1. 访问: https://www.kaggle.com/datasets/ipythonx/mvtec-ad
   2. 下载并解压到 data/mvtec_anomaly_detection/

方式3: MVTec官网 (需要注册)
   1. 访问: https://www.mvtec.com/company/research/datasets/mvtec-ad
   2. 注册账号后下载

确保最终目录结构如下:
   data/
   └── mvtec_anomaly_detection/  (或 MVTec/)
       ├── bottle/
       ├── cable/
       ├── capsule/
       └── ... (共15个类别)

================================================================================
""")


def check_existing_dataset(data_dir: str) -> Path:
    """检查数据集是否已存在"""
    data_path = Path(data_dir)

    # 检查可能的路径
    possible_paths = [
        data_path / "mvtec_anomaly_detection",
        data_path / "MVTec",
        data_path / "mvtec",
        data_path,  # 数据直接在data_dir下
    ]

    for path in possible_paths:
        if path.exists():
            # 检查是否包含类别文件夹
            existing_cats = [d.name for d in path.iterdir() if d.is_dir() and d.name in CATEGORIES]
            if len(existing_cats) >= 10:  # 至少有10个类别
                return path

    # 检查是否有压缩包
    tar_path = Path(data_dir) / "mvtec_anomaly_detection.tar.xz"
    if tar_path.exists():
        print(f"发现压缩包: {tar_path}")
        extract_tar_xz(tar_path, Path(data_dir))
        return Path(data_dir) / "mvtec_anomaly_detection"

    return None


def get_dataset_info(dataset_path: Path) -> dict:
    """获取数据集统计信息"""
    info = {}

    for category in CATEGORIES:
        cat_path = dataset_path / category
        if not cat_path.exists():
            continue

        train_path = cat_path / "train" / "good"
        test_path = cat_path / "test"

        train_count = len(list(train_path.glob("*.png"))) if train_path.exists() else 0

        test_good = 0
        test_defect = 0
        defect_types = []

        if test_path.exists():
            for defect_dir in test_path.iterdir():
                if defect_dir.is_dir():
                    count = len(list(defect_dir.glob("*.png")))
                    if defect_dir.name == "good":
                        test_good = count
                    else:
                        test_defect += count
                        defect_types.append(defect_dir.name)

        info[category] = {
            "train": train_count,
            "test_good": test_good,
            "test_defect": test_defect,
            "defect_types": defect_types
        }

    return info


def download_mvtec_ad(data_dir: str = "data", use_huggingface: bool = False) -> Path:
    """
    下载 MVTec AD 数据集

    Args:
        data_dir: 数据保存目录
        use_huggingface: 是否使用Hugging Face下载

    Returns:
        数据集路径
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    # 1. 检查是否已存在
    existing = check_existing_dataset(data_dir)
    if existing:
        print(f"数据集已存在于: {existing}")
        return existing

    # 2. 根据选项尝试不同的下载方式
    if use_huggingface:
        result = download_via_huggingface(data_dir)
        if result:
            return result
    else:
        # 尝试使用anomalib下载
        result = download_via_anomalib(data_dir)
        if result:
            return result

    # 3. 显示手动下载说明
    manual_download_instructions()
    return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="下载 MVTec AD 数据集")
    parser.add_argument("--data-dir", type=str, default="data", help="数据保存目录")
    parser.add_argument("--info", action="store_true", help="仅显示数据集信息")
    parser.add_argument("--manual", action="store_true", help="显示手动下载说明")
    parser.add_argument("--huggingface", action="store_true", help="使用Hugging Face下载")
    args = parser.parse_args()

    if args.manual:
        manual_download_instructions()
        sys.exit(0)

    dataset_path = check_existing_dataset(args.data_dir)

    if args.info:
        if dataset_path:
            info = get_dataset_info(dataset_path)
            print("\nMVTec AD 数据集统计:")
            print("-" * 60)
            print(f"{'类别':<15} {'训练':<8} {'测试(正常)':<12} {'测试(缺陷)':<12} {'缺陷类型'}")
            print("-" * 60)

            total_train = 0
            total_test_good = 0
            total_test_defect = 0

            for cat, stat in info.items():
                total_train += stat["train"]
                total_test_good += stat["test_good"]
                total_test_defect += stat["test_defect"]
                print(f"{cat:<15} {stat['train']:<8} {stat['test_good']:<12} {stat['test_defect']:<12} {len(stat['defect_types'])}种")

            print("-" * 60)
            print(f"{'总计':<15} {total_train:<8} {total_test_good:<12} {total_test_defect:<12}")
        else:
            print("数据集未找到，请先下载")
    else:
        download_mvtec_ad(args.data_dir, use_huggingface=args.huggingface)
