"""
PatchCore 深度学习方法训练脚本

使用anomalib库实现PatchCore异常检测
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import CATEGORIES
from src.deep_learning.patchcore import PatchCoreDetector


def train_patchcore_category(
    data_root: str,
    category: str,
    output_dir: str = "results/patchcore",
    backbone: str = "wide_resnet50_2",
    image_size: int = 224,
    devices: int = 1,
    accelerator: str = "auto"
) -> Dict[str, float]:
    """
    在单个类别上训练PatchCore

    Args:
        data_root: 数据集根目录
        category: 类别名称
        output_dir: 输出目录
        backbone: 特征提取backbone
        image_size: 图像尺寸
        devices: GPU数量
        accelerator: 加速器类型

    Returns:
        测试结果字典
    """
    print(f"\n{'='*50}")
    print(f"类别: {category}")
    print(f"Backbone: {backbone}")
    print(f"{'='*50}")

    # 创建检测器
    detector = PatchCoreDetector(
        backbone=backbone,
        image_size=(image_size, image_size)
    )

    # 类别输出目录
    cat_output_dir = os.path.join(output_dir, category)
    os.makedirs(cat_output_dir, exist_ok=True)

    # 训练
    print("训练中（构建memory bank）...")
    detector.train(
        data_root=data_root,
        category=category,
        output_dir=cat_output_dir,
        devices=devices,
        accelerator=accelerator
    )

    # 测试
    print("测试中...")
    results = detector.test(
        data_root=data_root,
        category=category
    )

    # 提取关键指标
    metrics = {
        'image_auroc': results.get('image_AUROC', 0),
        'pixel_auroc': results.get('pixel_AUROC', 0),
        'image_f1': results.get('image_F1Score', 0),
    }

    print(f"\n结果:")
    print(f"  Image AUROC: {metrics['image_auroc']:.4f}")
    print(f"  Pixel AUROC: {metrics['pixel_auroc']:.4f}")
    print(f"  Image F1:    {metrics['image_f1']:.4f}")

    return metrics


def run_all_categories(
    data_root: str,
    categories: List[str] = None,
    output_dir: str = "results/patchcore",
    backbone: str = "wide_resnet50_2",
    image_size: int = 224,
    devices: int = 1,
    accelerator: str = "auto"
) -> Dict[str, Dict[str, float]]:
    """
    在所有类别上运行PatchCore

    Args:
        data_root: 数据集根目录
        categories: 类别列表（默认全部）
        output_dir: 输出目录
        backbone: 特征提取backbone
        image_size: 图像尺寸

    Returns:
        {类别: 指标} 字典
    """
    if categories is None:
        categories = CATEGORIES

    os.makedirs(output_dir, exist_ok=True)

    all_results = {}

    for category in tqdm(categories, desc="处理类别"):
        try:
            metrics = train_patchcore_category(
                data_root=data_root,
                category=category,
                output_dir=output_dir,
                backbone=backbone,
                image_size=image_size,
                devices=devices,
                accelerator=accelerator
            )
            all_results[category] = metrics
        except Exception as e:
            print(f"处理 {category} 时出错: {e}")
            import traceback
            traceback.print_exc()
            all_results[category] = {"error": str(e)}

    # 计算平均值
    valid_results = {k: v for k, v in all_results.items() if 'error' not in v}
    if valid_results:
        avg_metrics = {}
        for metric in ['image_auroc', 'pixel_auroc', 'image_f1']:
            values = [v.get(metric, 0) for v in valid_results.values()]
            avg_metrics[metric] = sum(values) / len(values) if values else 0
        all_results['average'] = avg_metrics

    # 保存结果
    result_file = os.path.join(output_dir, 'patchcore_results.json')
    with open(result_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n结果已保存至: {result_file}")

    # 打印汇总
    print("\n" + "="*70)
    print("PatchCore 结果汇总")
    print("="*70)
    print(f"{'类别':<15} {'Image AUROC':<15} {'Pixel AUROC':<15} {'F1':<10}")
    print("-"*70)

    for cat, metrics in all_results.items():
        if 'error' in metrics:
            print(f"{cat:<15} ERROR: {metrics['error'][:40]}")
        else:
            img_auroc = metrics.get('image_auroc', 0)
            pix_auroc = metrics.get('pixel_auroc', 0)
            f1 = metrics.get('image_f1', 0)
            print(f"{cat:<15} {img_auroc:.4f}          {pix_auroc:.4f}          {f1:.4f}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="PatchCore 训练")
    parser.add_argument("--data-root", type=str, default="data/mvtec_anomaly_detection",
                       help="数据集根目录")
    parser.add_argument("--category", type=str, default=None,
                       help="单个类别（默认全部）")
    parser.add_argument("--output-dir", type=str, default="results/patchcore",
                       help="输出目录")
    parser.add_argument("--backbone", type=str, default="wide_resnet50_2",
                       choices=["resnet18", "resnet50", "wide_resnet50_2"],
                       help="特征提取backbone")
    parser.add_argument("--image-size", type=int, default=224,
                       help="图像尺寸")
    parser.add_argument("--devices", type=int, default=1,
                       help="GPU数量")
    parser.add_argument("--accelerator", type=str, default="auto",
                       choices=["auto", "gpu", "cpu"],
                       help="加速器类型")

    args = parser.parse_args()

    # 检查CUDA
    if torch.cuda.is_available():
        print(f"CUDA可用: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA不可用，将使用CPU")
        args.accelerator = "cpu"

    if args.category:
        # 单个类别
        metrics = train_patchcore_category(
            data_root=args.data_root,
            category=args.category,
            output_dir=args.output_dir,
            backbone=args.backbone,
            image_size=args.image_size,
            devices=args.devices,
            accelerator=args.accelerator
        )
        print(f"\n最终结果: {metrics}")
    else:
        # 所有类别
        run_all_categories(
            data_root=args.data_root,
            output_dir=args.output_dir,
            backbone=args.backbone,
            image_size=args.image_size,
            devices=args.devices,
            accelerator=args.accelerator
        )


if __name__ == "__main__":
    main()
