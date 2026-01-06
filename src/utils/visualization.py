"""
可视化工具模块

包含:
- PR曲线绘制
- ROC曲线绘制
- 异常热力图可视化
- 结果对比图表
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 颜色方案
COLORS = {
    'traditional': '#2ecc71',  # 绿色
    'deep_learning': '#e74c3c',  # 红色
    'isolation_forest': '#3498db',  # 蓝色
    'ocsvm': '#9b59b6',  # 紫色
    'patchcore': '#e74c3c',  # 红色
}


def plot_pr_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    title: str = "Precision-Recall Curve",
    label: str = None,
    ax: Optional[plt.Axes] = None,
    color: str = None
) -> plt.Axes:
    """
    绘制精确率-召回率曲线

    Args:
        y_true: 真实标签
        y_score: 异常分数
        title: 图标题
        label: 曲线标签
        ax: matplotlib axes对象
        color: 曲线颜色

    Returns:
        matplotlib axes对象
    """
    precision, recall, _ = precision_recall_curve(y_true, y_score)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(recall, precision, label=label, color=color, linewidth=2)
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)

    if label:
        ax.legend(loc='best')

    return ax


def plot_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    title: str = "ROC Curve",
    label: str = None,
    ax: Optional[plt.Axes] = None,
    color: str = None
) -> plt.Axes:
    """
    绘制ROC曲线

    Args:
        y_true: 真实标签
        y_score: 异常分数
        title: 图标题
        label: 曲线标签
        ax: matplotlib axes对象
        color: 曲线颜色

    Returns:
        matplotlib axes对象
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(fpr, tpr, label=label, color=color, linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)

    if label:
        ax.legend(loc='lower right')

    return ax


def plot_comparison_curves(
    results: Dict[str, Tuple[np.ndarray, np.ndarray]],
    curve_type: str = 'pr',
    title: str = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    绘制多方法对比曲线

    Args:
        results: {方法名: (y_true, y_score)} 字典
        curve_type: 'pr' 或 'roc'
        title: 图标题
        save_path: 保存路径

    Returns:
        matplotlib figure对象
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = list(plt.cm.tab10.colors)

    for i, (name, (y_true, y_score)) in enumerate(results.items()):
        color = COLORS.get(name.lower().replace(' ', '_'), colors[i % len(colors)])

        if curve_type == 'pr':
            precision, recall, _ = precision_recall_curve(y_true, y_score)
            ax.plot(recall, precision, label=name, color=color, linewidth=2)
        else:
            fpr, tpr, _ = roc_curve(y_true, y_score)
            ax.plot(fpr, tpr, label=name, color=color, linewidth=2)

    if curve_type == 'pr':
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        default_title = 'Precision-Recall Curve Comparison'
    else:
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        default_title = 'ROC Curve Comparison'

    ax.set_title(title or default_title, fontsize=14)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_anomaly_heatmap(
    image: np.ndarray,
    anomaly_map: np.ndarray,
    mask: Optional[np.ndarray] = None,
    title: str = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    绘制异常热力图

    Args:
        image: 原始图像 (H, W, C)
        anomaly_map: 异常热力图 (H, W)
        mask: 真实mask (H, W)，可选
        title: 图标题
        save_path: 保存路径

    Returns:
        matplotlib figure对象
    """
    n_cols = 3 if mask is not None else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))

    # 原始图像
    axes[0].imshow(image)
    axes[0].set_title('Original Image', fontsize=12)
    axes[0].axis('off')

    # 异常热力图
    axes[1].imshow(image)
    im = axes[1].imshow(anomaly_map, cmap='jet', alpha=0.5)
    axes[1].set_title('Anomaly Heatmap', fontsize=12)
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # 真实mask
    if mask is not None:
        axes[2].imshow(mask, cmap='gray')
        axes[2].set_title('Ground Truth', fontsize=12)
        axes[2].axis('off')

    if title:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_metrics_comparison(
    metrics: Dict[str, Dict[str, float]],
    metric_names: List[str] = None,
    title: str = "Methods Comparison",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    绘制方法对比柱状图

    Args:
        metrics: {方法名: {指标名: 值}} 字典
        metric_names: 要显示的指标名列表
        title: 图标题
        save_path: 保存路径

    Returns:
        matplotlib figure对象
    """
    methods = list(metrics.keys())

    if metric_names is None:
        # 获取所有指标名
        metric_names = list(next(iter(metrics.values())).keys())

    n_metrics = len(metric_names)
    n_methods = len(methods)

    fig, ax = plt.subplots(figsize=(max(10, n_metrics * 2), 6))

    x = np.arange(n_metrics)
    width = 0.8 / n_methods

    colors = list(plt.cm.tab10.colors)

    for i, method in enumerate(methods):
        values = [metrics[method].get(m, 0) for m in metric_names]
        color = COLORS.get(method.lower().replace(' ', '_'), colors[i % len(colors)])
        bars = ax.bar(x + i * width, values, width, label=method, color=color)

        # 添加数值标签
        for bar, val in zip(bars, values):
            ax.annotate(f'{val:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x + width * (n_methods - 1) / 2)
    ax.set_xticklabels(metric_names, rotation=45, ha='right')
    ax.legend(loc='best')
    ax.set_ylim([0, 1.15])
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_category_comparison(
    results: Dict[str, Dict[str, float]],
    metric: str = 'image_auroc',
    title: str = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    绘制不同类别的指标对比

    Args:
        results: {类别名: {方法名: 指标值}} 字典
        metric: 指标名称
        title: 图标题
        save_path: 保存路径

    Returns:
        matplotlib figure对象
    """
    categories = list(results.keys())
    methods = list(next(iter(results.values())).keys())

    fig, ax = plt.subplots(figsize=(max(12, len(categories) * 0.8), 6))

    x = np.arange(len(categories))
    width = 0.8 / len(methods)

    colors = list(plt.cm.tab10.colors)

    for i, method in enumerate(methods):
        values = [results[cat].get(method, 0) for cat in categories]
        color = COLORS.get(method.lower().replace(' ', '_'), colors[i % len(colors)])
        ax.bar(x + i * width, values, width, label=method, color=color)

    ax.set_xlabel('Category', fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    ax.set_title(title or f'{metric} by Category', fontsize=14)
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.legend(loc='best')
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def create_summary_table(
    results: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None
) -> str:
    """
    创建结果汇总表格 (Markdown格式)

    Args:
        results: {方法名: {指标名: 值}} 字典
        save_path: 保存路径

    Returns:
        Markdown表格字符串
    """
    methods = list(results.keys())
    metrics = list(next(iter(results.values())).keys())

    # 表头
    header = "| Method | " + " | ".join(metrics) + " |"
    separator = "|" + "|".join([":---:"] * (len(metrics) + 1)) + "|"

    # 数据行
    rows = []
    for method in methods:
        values = [f"{results[method].get(m, 0):.4f}" for m in metrics]
        row = f"| {method} | " + " | ".join(values) + " |"
        rows.append(row)

    table = "\n".join([header, separator] + rows)

    if save_path:
        with open(save_path, 'w') as f:
            f.write(table)

    return table


if __name__ == "__main__":
    # 测试代码
    np.random.seed(42)

    # 生成模拟数据
    n = 200
    y_true = np.array([0] * 100 + [1] * 100)

    # 模拟两种方法的预测分数
    scores_method1 = np.concatenate([
        np.random.normal(0.3, 0.15, 100),
        np.random.normal(0.7, 0.15, 100)
    ])

    scores_method2 = np.concatenate([
        np.random.normal(0.35, 0.1, 100),
        np.random.normal(0.65, 0.1, 100)
    ])

    # 测试PR曲线对比
    results = {
        'Isolation Forest': (y_true, scores_method1),
        'PatchCore': (y_true, scores_method2)
    }

    fig = plot_comparison_curves(results, curve_type='pr', title='PR Curve Comparison')
    plt.savefig('test_pr_curve.png', dpi=100)
    print("PR曲线已保存: test_pr_curve.png")

    # 测试指标对比柱状图
    metrics = {
        'Isolation Forest': {'AUROC': 0.85, 'AUPR': 0.78, 'F1': 0.72},
        'One-Class SVM': {'AUROC': 0.82, 'AUPR': 0.75, 'F1': 0.68},
        'PatchCore': {'AUROC': 0.95, 'AUPR': 0.92, 'F1': 0.88}
    }

    fig = plot_metrics_comparison(metrics, title='Methods Comparison')
    plt.savefig('test_metrics.png', dpi=100)
    print("指标对比图已保存: test_metrics.png")

    # 测试汇总表格
    table = create_summary_table(metrics)
    print("\n结果汇总表格:")
    print(table)

    plt.close('all')
