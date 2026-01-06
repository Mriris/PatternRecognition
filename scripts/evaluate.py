"""
完整评估脚本

运行传统方法和深度学习方法，生成对比报告
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import CATEGORIES
from src.utils.visualization import (
    plot_metrics_comparison,
    plot_category_comparison,
    create_summary_table
)
import matplotlib.pyplot as plt


def load_results(results_dir: str) -> Dict[str, Dict]:
    """加载所有结果文件"""
    results = {}

    # 传统方法结果
    traditional_dir = os.path.join(results_dir, 'traditional')
    if os.path.exists(traditional_dir):
        for f in os.listdir(traditional_dir):
            if f.endswith('_results.json'):
                method_name = f.replace('_results.json', '')
                with open(os.path.join(traditional_dir, f)) as fp:
                    results[method_name] = json.load(fp)

    # PatchCore结果
    patchcore_file = os.path.join(results_dir, 'patchcore', 'patchcore_results.json')
    if os.path.exists(patchcore_file):
        with open(patchcore_file) as f:
            results['patchcore'] = json.load(f)

    return results


def generate_comparison_report(
    results: Dict[str, Dict],
    output_dir: str
) -> None:
    """生成对比报告"""
    os.makedirs(output_dir, exist_ok=True)

    # 提取各方法的平均指标
    avg_metrics = {}
    for method, data in results.items():
        if 'average' in data:
            avg_metrics[method] = data['average']

    if not avg_metrics:
        print("没有找到有效的结果数据")
        return

    # 1. 方法对比柱状图
    print("生成方法对比图...")
    fig = plot_metrics_comparison(
        avg_metrics,
        metric_names=['image_auroc', 'image_aupr', 'image_f1'],
        title='Methods Comparison (Average across all categories)'
    )
    plt.savefig(os.path.join(output_dir, 'methods_comparison.png'), dpi=150)
    plt.close()

    # 2. 各类别对比图
    print("生成类别对比图...")

    # 整理数据格式: {category: {method: auroc}}
    category_results = {}
    for category in CATEGORIES:
        category_results[category] = {}
        for method, data in results.items():
            if category in data and 'error' not in data[category]:
                category_results[category][method] = data[category].get('image_auroc', 0)

    # 过滤掉没有数据的类别
    category_results = {k: v for k, v in category_results.items() if v}

    if category_results:
        fig = plot_category_comparison(
            category_results,
            metric='image_auroc',
            title='Image AUROC by Category'
        )
        plt.savefig(os.path.join(output_dir, 'category_comparison.png'), dpi=150)
        plt.close()

    # 3. 生成Markdown表格
    print("生成结果表格...")
    table = create_summary_table(avg_metrics)
    with open(os.path.join(output_dir, 'results_table.md'), 'w') as f:
        f.write("# 方法对比结果\n\n")
        f.write("## 平均指标\n\n")
        f.write(table)
        f.write("\n\n## 各类别详情\n\n")

        # 详细表格
        f.write("| Category |")
        for method in results.keys():
            f.write(f" {method} |")
        f.write("\n|:---:|")
        for _ in results.keys():
            f.write(":---:|")
        f.write("\n")

        for category in CATEGORIES:
            f.write(f"| {category} |")
            for method, data in results.items():
                if category in data and 'error' not in data[category]:
                    auroc = data[category].get('image_auroc', 0)
                    f.write(f" {auroc:.4f} |")
                else:
                    f.write(" - |")
            f.write("\n")

    print(f"报告已保存至: {output_dir}")


def run_full_evaluation(
    data_root: str,
    output_dir: str,
    run_traditional: bool = True,
    run_patchcore: bool = True,
    categories: list = None
) -> None:
    """运行完整评估"""
    from scripts.train_traditional import run_all_categories as run_traditional_all
    from scripts.train_patchcore import run_all_categories as run_patchcore_all

    if categories is None:
        categories = CATEGORIES

    results = {}

    # 传统方法
    if run_traditional:
        print("\n" + "="*60)
        print("运行传统方法")
        print("="*60)

        # Isolation Forest + Combined Features
        print("\n--- Isolation Forest + Combined Features ---")
        results['isolation_forest_combined'] = run_traditional_all(
            data_root=data_root,
            categories=categories,
            feature_type='combined',
            detector_type='isolation_forest',
            output_dir=os.path.join(output_dir, 'traditional')
        )

        # One-Class SVM + Combined Features
        print("\n--- One-Class SVM + Combined Features ---")
        results['ocsvm_combined'] = run_traditional_all(
            data_root=data_root,
            categories=categories,
            feature_type='combined',
            detector_type='ocsvm',
            output_dir=os.path.join(output_dir, 'traditional')
        )

    # PatchCore
    if run_patchcore:
        print("\n" + "="*60)
        print("运行 PatchCore")
        print("="*60)

        results['patchcore'] = run_patchcore_all(
            data_root=data_root,
            categories=categories,
            output_dir=os.path.join(output_dir, 'patchcore')
        )

    # 生成报告
    print("\n" + "="*60)
    print("生成对比报告")
    print("="*60)

    generate_comparison_report(results, os.path.join(output_dir, 'report'))


def main():
    parser = argparse.ArgumentParser(description="完整评估脚本")
    parser.add_argument("--data-root", type=str, default="data/mvtec_anomaly_detection",
                       help="数据集根目录")
    parser.add_argument("--output-dir", type=str, default="results",
                       help="输出目录")
    parser.add_argument("--category", type=str, default=None,
                       help="单个类别（默认全部）")
    parser.add_argument("--skip-traditional", action="store_true",
                       help="跳过传统方法")
    parser.add_argument("--skip-patchcore", action="store_true",
                       help="跳过PatchCore")
    parser.add_argument("--report-only", action="store_true",
                       help="只生成报告（不运行训练）")

    args = parser.parse_args()

    categories = [args.category] if args.category else None

    if args.report_only:
        # 只生成报告
        results = load_results(args.output_dir)
        if results:
            generate_comparison_report(results, os.path.join(args.output_dir, 'report'))
        else:
            print("未找到结果文件")
    else:
        # 运行完整评估
        run_full_evaluation(
            data_root=args.data_root,
            output_dir=args.output_dir,
            run_traditional=not args.skip_traditional,
            run_patchcore=not args.skip_patchcore,
            categories=categories
        )


if __name__ == "__main__":
    main()
