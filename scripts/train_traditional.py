"""
传统方法训练脚本

使用HOG/LBP特征 + Isolation Forest/One-Class SVM进行异常检测
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import MVTecDatasetForTraditional, CATEGORIES
from src.traditional.features import extract_features_batch
from src.traditional.detector import create_detector, BaseAnomalyDetector
from src.evaluation.metrics import compute_all_metrics, InferenceTimer


def train_and_evaluate_category(
    data_root: str,
    category: str,
    feature_type: str = 'combined',
    detector_type: str = 'isolation_forest',
    image_size: int = 224,
    **detector_kwargs
) -> Dict[str, float]:
    """
    在单个类别上训练和评估

    Args:
        data_root: 数据集根目录
        category: 类别名称
        feature_type: 特征类型 ('hog', 'lbp', 'color', 'combined')
        detector_type: 检测器类型 ('isolation_forest', 'ocsvm', 'gaussian')
        image_size: 图像尺寸

    Returns:
        评估指标字典
    """
    print(f"\n{'='*50}")
    print(f"类别: {category}")
    print(f"特征: {feature_type}, 检测器: {detector_type}")
    print(f"{'='*50}")

    # 加载数据
    print("加载数据...")
    train_dataset = MVTecDatasetForTraditional(data_root, category, split="train", image_size=image_size)
    test_dataset = MVTecDatasetForTraditional(data_root, category, split="test", image_size=image_size)

    X_train, _ = train_dataset.get_images_and_labels()
    X_test, y_test = test_dataset.get_images_and_labels()

    print(f"训练集: {len(X_train)} 张图像")
    print(f"测试集: {len(X_test)} 张图像 (正常: {(y_test==0).sum()}, 异常: {(y_test==1).sum()})")

    # 提取特征
    print("提取特征...")
    timer = InferenceTimer()

    with timer:
        train_features = extract_features_batch(X_train, feature_type=feature_type)
    train_time = timer.times[-1]

    with timer:
        test_features = extract_features_batch(X_test, feature_type=feature_type)
    feature_time = timer.times[-1] / len(X_test)

    print(f"特征维度: {train_features.shape[1]}")
    print(f"特征提取时间: {feature_time*1000:.2f}ms/image")

    # 训练检测器
    print("训练检测器...")
    detector = create_detector(detector_type, **detector_kwargs)

    start_time = time.time()
    detector.fit(train_features)
    fit_time = time.time() - start_time
    print(f"训练时间: {fit_time:.2f}s")

    # 预测
    print("预测...")
    with timer:
        scores = detector.decision_function(test_features)
    inference_time = timer.times[-1] / len(X_test)
    print(f"推理时间: {inference_time*1000:.2f}ms/image")

    # 计算指标
    metrics = compute_all_metrics(y_test, scores)
    metrics['feature_time_ms'] = feature_time * 1000
    metrics['inference_time_ms'] = inference_time * 1000
    metrics['total_time_ms'] = (feature_time + inference_time) * 1000

    print(f"\n结果:")
    print(f"  Image AUROC: {metrics['image_auroc']:.4f}")
    print(f"  Image AUPR:  {metrics['image_aupr']:.4f}")
    print(f"  Image F1:    {metrics['image_f1']:.4f}")

    return metrics


def run_all_categories(
    data_root: str,
    categories: List[str] = None,
    feature_type: str = 'combined',
    detector_type: str = 'isolation_forest',
    output_dir: str = 'results/traditional',
    **detector_kwargs
) -> Dict[str, Dict[str, float]]:
    """
    在所有类别上运行

    Args:
        data_root: 数据集根目录
        categories: 类别列表（默认全部）
        feature_type: 特征类型
        detector_type: 检测器类型
        output_dir: 输出目录

    Returns:
        {类别: 指标} 字典
    """
    if categories is None:
        categories = CATEGORIES

    os.makedirs(output_dir, exist_ok=True)

    all_results = {}

    for category in tqdm(categories, desc="处理类别"):
        try:
            metrics = train_and_evaluate_category(
                data_root=data_root,
                category=category,
                feature_type=feature_type,
                detector_type=detector_type,
                **detector_kwargs
            )
            all_results[category] = metrics
        except Exception as e:
            print(f"处理 {category} 时出错: {e}")
            all_results[category] = {"error": str(e)}

    # 计算平均值
    valid_results = {k: v for k, v in all_results.items() if 'error' not in v}
    if valid_results:
        avg_metrics = {}
        for metric in ['image_auroc', 'image_aupr', 'image_f1']:
            values = [v[metric] for v in valid_results.values()]
            avg_metrics[metric] = np.mean(values)
        all_results['average'] = avg_metrics

    # 保存结果
    result_file = os.path.join(output_dir, f'{detector_type}_{feature_type}_results.json')
    with open(result_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n结果已保存至: {result_file}")

    # 打印汇总
    print("\n" + "="*60)
    print("结果汇总")
    print("="*60)
    print(f"{'类别':<15} {'AUROC':<10} {'AUPR':<10} {'F1':<10}")
    print("-"*60)

    for cat, metrics in all_results.items():
        if 'error' in metrics:
            print(f"{cat:<15} ERROR: {metrics['error'][:30]}")
        else:
            print(f"{cat:<15} {metrics['image_auroc']:.4f}     {metrics['image_aupr']:.4f}     {metrics['image_f1']:.4f}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="传统方法训练")
    parser.add_argument("--data-root", type=str, default="data/mvtec_anomaly_detection",
                       help="数据集根目录")
    parser.add_argument("--category", type=str, default=None,
                       help="单个类别（默认全部）")
    parser.add_argument("--feature-type", type=str, default="combined",
                       choices=["hog", "lbp", "color", "combined"],
                       help="特征类型")
    parser.add_argument("--detector-type", type=str, default="isolation_forest",
                       choices=["isolation_forest", "ocsvm", "gaussian"],
                       help="检测器类型")
    parser.add_argument("--output-dir", type=str, default="results/traditional",
                       help="输出目录")
    parser.add_argument("--image-size", type=int, default=224,
                       help="图像尺寸")

    args = parser.parse_args()

    if args.category:
        # 单个类别
        metrics = train_and_evaluate_category(
            data_root=args.data_root,
            category=args.category,
            feature_type=args.feature_type,
            detector_type=args.detector_type,
            image_size=args.image_size
        )
        print(f"\n最终结果: {metrics}")
    else:
        # 所有类别
        run_all_categories(
            data_root=args.data_root,
            feature_type=args.feature_type,
            detector_type=args.detector_type,
            output_dir=args.output_dir
        )


if __name__ == "__main__":
    main()
