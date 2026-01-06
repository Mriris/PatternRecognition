"""
评估指标模块

包含:
- AUROC (Area Under ROC Curve): 图像级/像素级异常检测评估
- AUPR (Area Under PR Curve): 精确率-召回率曲线下面积
- F1-Score: 在最佳阈值下的F1值
- 推理时间统计
"""

import numpy as np
from typing import Dict, Tuple, Optional
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)
import time


def compute_auroc(
    y_true: np.ndarray,
    y_score: np.ndarray
) -> float:
    """
    计算AUROC (Area Under ROC Curve)

    AUROC衡量分类器在各种阈值下区分正负样本的能力。
    值为1表示完美分类，0.5表示随机分类。

    Args:
        y_true: 真实标签 (0: 正常, 1: 异常)
        y_score: 异常分数 (越大越异常)

    Returns:
        AUROC值
    """
    if len(np.unique(y_true)) < 2:
        return 0.0
    return roc_auc_score(y_true, y_score)


def compute_aupr(
    y_true: np.ndarray,
    y_score: np.ndarray
) -> float:
    """
    计算AUPR (Area Under Precision-Recall Curve)

    AUPR在类别不平衡时比AUROC更有参考价值。
    对于异常检测任务（异常样本通常较少），AUPR是重要指标。

    Args:
        y_true: 真实标签
        y_score: 异常分数

    Returns:
        AUPR值
    """
    if len(np.unique(y_true)) < 2:
        return 0.0
    return average_precision_score(y_true, y_score)


def compute_f1_at_best_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray
) -> Tuple[float, float]:
    """
    计算最佳阈值下的F1分数

    遍历所有可能的阈值，找到使F1最大的阈值。

    Args:
        y_true: 真实标签
        y_score: 异常分数

    Returns:
        (最佳F1, 最佳阈值)
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)

    # 计算每个阈值下的F1
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)

    best_idx = np.argmax(f1_scores)
    best_f1 = f1_scores[best_idx]
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]

    return best_f1, best_threshold


def compute_pixel_auroc(
    masks_true: np.ndarray,
    anomaly_maps: np.ndarray
) -> float:
    """
    计算像素级AUROC

    用于评估异常定位的准确性。

    Args:
        masks_true: 真实mask (N, H, W)，值为0或1
        anomaly_maps: 预测的异常热力图 (N, H, W)

    Returns:
        像素级AUROC
    """
    # 展平
    y_true = masks_true.flatten()
    y_score = anomaly_maps.flatten()

    if len(np.unique(y_true)) < 2:
        return 0.0

    return roc_auc_score(y_true, y_score)


def get_precision_recall_curve(
    y_true: np.ndarray,
    y_score: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    获取精确率-召回率曲线数据

    Args:
        y_true: 真实标签
        y_score: 异常分数

    Returns:
        (precision, recall, thresholds)
    """
    return precision_recall_curve(y_true, y_score)


def get_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    获取ROC曲线数据

    Args:
        y_true: 真实标签
        y_score: 异常分数

    Returns:
        (fpr, tpr, thresholds)
    """
    return roc_curve(y_true, y_score)


def compute_confusion_matrix_at_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float
) -> np.ndarray:
    """
    计算给定阈值下的混淆矩阵

    Args:
        y_true: 真实标签
        y_score: 异常分数
        threshold: 决策阈值

    Returns:
        混淆矩阵 [[TN, FP], [FN, TP]]
    """
    y_pred = (y_score >= threshold).astype(int)
    return confusion_matrix(y_true, y_pred)


def compute_all_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    masks_true: Optional[np.ndarray] = None,
    anomaly_maps: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    计算所有评估指标

    Args:
        y_true: 图像级真实标签
        y_score: 图像级异常分数
        masks_true: 像素级真实mask（可选）
        anomaly_maps: 像素级异常热力图（可选）

    Returns:
        包含所有指标的字典
    """
    metrics = {}

    # 图像级指标
    metrics["image_auroc"] = compute_auroc(y_true, y_score)
    metrics["image_aupr"] = compute_aupr(y_true, y_score)

    best_f1, best_threshold = compute_f1_at_best_threshold(y_true, y_score)
    metrics["image_f1"] = best_f1
    metrics["best_threshold"] = best_threshold

    # 在最佳阈值下的精确率和召回率
    y_pred = (y_score >= best_threshold).astype(int)
    metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)

    # 像素级指标（如果提供）
    if masks_true is not None and anomaly_maps is not None:
        metrics["pixel_auroc"] = compute_pixel_auroc(masks_true, anomaly_maps)

    return metrics


class InferenceTimer:
    """推理时间统计器"""

    def __init__(self):
        self.times = []

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        elapsed = time.perf_counter() - self.start
        self.times.append(elapsed)

    def get_stats(self) -> Dict[str, float]:
        """获取统计信息"""
        if not self.times:
            return {"mean": 0, "std": 0, "min": 0, "max": 0}

        times = np.array(self.times)
        return {
            "mean": times.mean(),
            "std": times.std(),
            "min": times.min(),
            "max": times.max(),
            "total": times.sum(),
            "count": len(times)
        }


def format_metrics(metrics: Dict[str, float], precision: int = 4) -> str:
    """格式化输出指标"""
    lines = []
    for key, value in metrics.items():
        lines.append(f"  {key}: {value:.{precision}f}")
    return "\n".join(lines)


if __name__ == "__main__":
    # 测试代码
    np.random.seed(42)

    # 生成模拟数据
    n_normal = 100
    n_anomaly = 50

    # 正常样本分数较低，异常样本分数较高
    scores_normal = np.random.normal(0.3, 0.1, n_normal)
    scores_anomaly = np.random.normal(0.7, 0.15, n_anomaly)

    y_true = np.array([0] * n_normal + [1] * n_anomaly)
    y_score = np.concatenate([scores_normal, scores_anomaly])

    print("测试评估指标:")
    print("-" * 50)

    metrics = compute_all_metrics(y_true, y_score)
    print(format_metrics(metrics))

    print("\n精确率-召回率曲线数据点数:", len(get_precision_recall_curve(y_true, y_score)[0]))
    print("ROC曲线数据点数:", len(get_roc_curve(y_true, y_score)[0]))

    # 测试推理计时器
    timer = InferenceTimer()
    for _ in range(10):
        with timer:
            time.sleep(0.01)

    print("\n推理时间统计:")
    for k, v in timer.get_stats().items():
        print(f"  {k}: {v:.6f}s")
