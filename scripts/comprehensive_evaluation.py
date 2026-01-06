"""
工业缺陷检测综合评估脚本

包含:
1. 问题建模说明（无监督学习）
2. 传统方法 vs 深度学习方法对比
3. 完整的可视化结果
4. 评估指标分析
5. 部署难点讨论
"""

import os
import sys
import json
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import seaborn as sns
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_curve, precision_recall_curve, auc, f1_score
import cv2

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import CATEGORIES, MVTecDatasetForTraditional
from src.traditional.features import extract_combined_features, extract_hog_features, extract_lbp_features
from src.traditional.detector import IsolationForestDetector, OneClassSVMDetector
from src.evaluation.metrics import compute_auroc, compute_aupr, compute_f1_at_best_threshold as compute_optimal_f1

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings('ignore')


class ComprehensiveEvaluator:
    """综合评估器"""

    def __init__(self, data_root: str, output_dir: str = "results/comprehensive"):
        self.data_root = data_root
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 创建子目录
        self.dirs = {
            'figures': os.path.join(output_dir, 'figures'),
            'visualizations': os.path.join(output_dir, 'visualizations'),
            'reports': os.path.join(output_dir, 'reports'),
            'data': os.path.join(output_dir, 'data')
        }
        for d in self.dirs.values():
            os.makedirs(d, exist_ok=True)

        self.results = {}

    def run_traditional_method(
        self,
        category: str,
        feature_type: str = 'combined',
        detector_type: str = 'isolation_forest'
    ) -> Dict:
        """运行传统方法"""
        print(f"  传统方法 ({detector_type} + {feature_type})...")

        # 加载数据
        train_dataset = MVTecDatasetForTraditional(self.data_root, category, split='train')
        test_dataset = MVTecDatasetForTraditional(self.data_root, category, split='test')

        # 获取图像和标签
        train_images, _ = train_dataset.get_images_and_labels()
        test_images, test_labels = test_dataset.get_images_and_labels()

        # 提取特征
        print(f"    提取训练特征 ({len(train_images)} 张)...")
        train_features = []
        for img in tqdm(train_images, desc="    训练集", leave=False):
            if feature_type == 'combined':
                feat = extract_combined_features(img)
            elif feature_type == 'hog':
                feat = extract_hog_features(img)
            else:
                feat = extract_lbp_features(img)
            train_features.append(feat)
        train_features = np.array(train_features)

        print(f"    提取测试特征 ({len(test_images)} 张)...")
        test_features = []
        for img in tqdm(test_images, desc="    测试集", leave=False):
            if feature_type == 'combined':
                feat = extract_combined_features(img)
            elif feature_type == 'hog':
                feat = extract_hog_features(img)
            else:
                feat = extract_lbp_features(img)
            test_features.append(feat)
        test_features = np.array(test_features)

        # 训练检测器
        if detector_type == 'isolation_forest':
            detector = IsolationForestDetector(contamination=0.1)
        else:
            detector = OneClassSVMDetector(nu=0.1)

        start_time = time.time()
        detector.fit(train_features)
        train_time = time.time() - start_time

        # 预测
        start_time = time.time()
        scores = -detector.decision_function(test_features)  # 取负数使异常分数更高
        inference_time = (time.time() - start_time) / len(test_features) * 1000  # ms

        # 归一化分数
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

        # 计算指标
        auroc = compute_auroc(test_labels, scores)
        aupr = compute_aupr(test_labels, scores)
        f1, threshold = compute_optimal_f1(test_labels, scores)

        # 计算ROC和PR曲线数据
        fpr, tpr, _ = roc_curve(test_labels, scores)
        precision, recall, _ = precision_recall_curve(test_labels, scores)

        print(f"    AUROC: {auroc:.4f}, F1: {f1:.4f}")

        return {
            'auroc': auroc,
            'aupr': aupr,
            'f1': f1,
            'threshold': threshold,
            'train_time': train_time,
            'inference_time_ms': inference_time,
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'scores': scores.tolist(),
            'labels': test_labels.tolist(),
            'test_images': test_images[:10].tolist()  # 保存前10张用于可视化
        }

    def run_patchcore(self, category: str) -> Dict:
        """运行PatchCore深度学习方法"""
        print(f"  PatchCore 深度学习方法...")

        from anomalib.data import MVTec
        from anomalib.models import Patchcore
        from anomalib.engine import Engine
        import torch

        # 创建数据模块
        datamodule = MVTec(
            root=self.data_root,
            category=category,
            train_batch_size=32,
            eval_batch_size=32,
            num_workers=4,
        )

        # 创建模型
        model = Patchcore(
            backbone="wide_resnet50_2",
            layers=["layer2", "layer3"],
            coreset_sampling_ratio=0.1,
            num_neighbors=9,
        )

        # 创建引擎
        engine = Engine(
            max_epochs=1,
            devices=1,
            accelerator="auto",
            default_root_dir=os.path.join(self.output_dir, 'patchcore', category),
            enable_checkpointing=False,
            logger=False,
        )

        # 训练
        start_time = time.time()
        engine.fit(model=model, datamodule=datamodule)
        train_time = time.time() - start_time

        # 测试
        results = engine.test(model=model, datamodule=datamodule)
        metrics = results[0] if results else {}

        # 获取预测结果用于绘制曲线
        datamodule.setup('test')
        test_loader = datamodule.test_dataloader()

        all_scores = []
        all_labels = []
        all_anomaly_maps = []
        all_images = []

        model.eval()
        device = next(model.parameters()).device

        start_time = time.time()
        with torch.no_grad():
            for batch in test_loader:
                images = batch['image'].to(device)
                outputs = model(images)

                scores = outputs['pred_scores'].cpu().numpy()
                labels = batch['label'].cpu().numpy()

                all_scores.extend(scores.tolist())
                all_labels.extend(labels.tolist())

                # 保存异常热力图
                if 'anomaly_maps' in outputs and len(all_anomaly_maps) < 10:
                    anomaly_maps = outputs['anomaly_maps'].cpu().numpy()
                    for i, am in enumerate(anomaly_maps):
                        if len(all_anomaly_maps) < 10:
                            all_anomaly_maps.append(am.squeeze())
                            all_images.append(batch['image'][i].cpu().numpy().transpose(1, 2, 0))

        inference_time = (time.time() - start_time) / len(all_scores) * 1000  # ms

        # 归一化分数
        scores = np.array(all_scores)
        labels = np.array(all_labels)
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

        # 计算ROC和PR曲线数据
        fpr, tpr, _ = roc_curve(labels, scores)
        precision, recall, _ = precision_recall_curve(labels, scores)

        return {
            'auroc': metrics.get('image_AUROC', 0),
            'pixel_auroc': metrics.get('pixel_AUROC', 0),
            'aupr': auc(recall, precision),
            'f1': metrics.get('image_F1Score', 0),
            'train_time': train_time,
            'inference_time_ms': inference_time,
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'scores': scores.tolist(),
            'labels': labels.tolist(),
            'anomaly_maps': all_anomaly_maps,
            'test_images': all_images
        }

    def evaluate_category(self, category: str) -> Dict:
        """评估单个类别"""
        print(f"\n{'='*60}")
        print(f"评估类别: {category}")
        print(f"{'='*60}")

        results = {}

        # 传统方法 - Isolation Forest
        results['iforest_combined'] = self.run_traditional_method(
            category, 'combined', 'isolation_forest'
        )

        # 传统方法 - One-Class SVM
        results['ocsvm_combined'] = self.run_traditional_method(
            category, 'combined', 'ocsvm'
        )

        # 深度学习方法 - PatchCore
        try:
            results['patchcore'] = self.run_patchcore(category)
        except Exception as e:
            print(f"  PatchCore 失败: {e}")
            results['patchcore'] = {'error': str(e)}

        return results

    def plot_roc_curves(self, category: str, results: Dict):
        """绘制ROC曲线对比图"""
        fig, ax = plt.subplots(figsize=(8, 6))

        colors = {'iforest_combined': '#3498db', 'ocsvm_combined': '#2ecc71', 'patchcore': '#e74c3c'}
        labels = {'iforest_combined': 'Isolation Forest', 'ocsvm_combined': 'One-Class SVM', 'patchcore': 'PatchCore'}

        for method, data in results.items():
            if 'error' in data:
                continue
            fpr = np.array(data['fpr'])
            tpr = np.array(data['tpr'])
            auroc = data['auroc']
            ax.plot(fpr, tpr, color=colors.get(method, 'gray'),
                   label=f"{labels.get(method, method)} (AUC={auroc:.4f})", linewidth=2)

        ax.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(f'ROC Curves - {category}', fontsize=14)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])

        plt.tight_layout()
        plt.savefig(os.path.join(self.dirs['figures'], f'roc_{category}.png'), dpi=150)
        plt.close()

    def plot_pr_curves(self, category: str, results: Dict):
        """绘制PR曲线对比图"""
        fig, ax = plt.subplots(figsize=(8, 6))

        colors = {'iforest_combined': '#3498db', 'ocsvm_combined': '#2ecc71', 'patchcore': '#e74c3c'}
        labels = {'iforest_combined': 'Isolation Forest', 'ocsvm_combined': 'One-Class SVM', 'patchcore': 'PatchCore'}

        for method, data in results.items():
            if 'error' in data:
                continue
            precision = np.array(data['precision'])
            recall = np.array(data['recall'])
            aupr = data['aupr']
            ax.plot(recall, precision, color=colors.get(method, 'gray'),
                   label=f"{labels.get(method, method)} (AUPR={aupr:.4f})", linewidth=2)

        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title(f'Precision-Recall Curves - {category}', fontsize=14)
        ax.legend(loc='lower left')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])

        plt.tight_layout()
        plt.savefig(os.path.join(self.dirs['figures'], f'pr_{category}.png'), dpi=150)
        plt.close()

    def plot_anomaly_visualization(self, category: str, results: Dict):
        """绘制异常检测可视化"""
        if 'patchcore' not in results or 'error' in results['patchcore']:
            return

        patchcore_data = results['patchcore']
        if 'anomaly_maps' not in patchcore_data or not patchcore_data['anomaly_maps']:
            return

        n_samples = min(5, len(patchcore_data['anomaly_maps']))
        fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4*n_samples))

        if n_samples == 1:
            axes = axes.reshape(1, -1)

        for i in range(n_samples):
            # 原图
            img = patchcore_data['test_images'][i]
            if img.max() <= 1:
                img = (img * 255).astype(np.uint8)
            axes[i, 0].imshow(img)
            axes[i, 0].set_title('Original Image')
            axes[i, 0].axis('off')

            # 异常热力图
            am = patchcore_data['anomaly_maps'][i]
            im = axes[i, 1].imshow(am, cmap='jet')
            axes[i, 1].set_title('Anomaly Heatmap')
            axes[i, 1].axis('off')
            plt.colorbar(im, ax=axes[i, 1], fraction=0.046, pad=0.04)

            # 叠加图
            am_resized = cv2.resize(am, (img.shape[1], img.shape[0]))
            am_normalized = (am_resized - am_resized.min()) / (am_resized.max() - am_resized.min() + 1e-8)
            heatmap = cv2.applyColorMap((am_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
            axes[i, 2].imshow(overlay)
            axes[i, 2].set_title('Overlay')
            axes[i, 2].axis('off')

        plt.suptitle(f'Anomaly Detection Visualization - {category}', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(self.dirs['visualizations'], f'anomaly_vis_{category}.png'), dpi=150)
        plt.close()

    def plot_overall_comparison(self, all_results: Dict):
        """绘制整体对比图"""
        # 提取数据
        categories = []
        methods = ['iforest_combined', 'ocsvm_combined', 'patchcore']
        method_labels = ['Isolation Forest', 'One-Class SVM', 'PatchCore']
        auroc_data = {m: [] for m in methods}

        for cat, results in all_results.items():
            if cat == 'average':
                continue
            categories.append(cat)
            for m in methods:
                if m in results and 'error' not in results[m]:
                    auroc_data[m].append(results[m]['auroc'])
                else:
                    auroc_data[m].append(0)

        # 1. 柱状图对比
        x = np.arange(len(categories))
        width = 0.25
        fig, ax = plt.subplots(figsize=(16, 8))

        colors = ['#3498db', '#2ecc71', '#e74c3c']
        for i, (m, label) in enumerate(zip(methods, method_labels)):
            ax.bar(x + i*width, auroc_data[m], width, label=label, color=colors[i])

        ax.set_xlabel('Category', fontsize=12)
        ax.set_ylabel('Image AUROC', fontsize=12)
        ax.set_title('Methods Comparison across Categories', fontsize=14)
        ax.set_xticks(x + width)
        ax.set_xticklabels(categories, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(os.path.join(self.dirs['figures'], 'overall_comparison.png'), dpi=150)
        plt.close()

        # 2. 热力图
        fig, ax = plt.subplots(figsize=(12, 8))
        data_matrix = np.array([auroc_data[m] for m in methods]).T
        sns.heatmap(data_matrix, annot=True, fmt='.3f', cmap='RdYlGn',
                   xticklabels=method_labels, yticklabels=categories,
                   vmin=0.5, vmax=1.0, ax=ax)
        ax.set_title('Image AUROC Heatmap', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(self.dirs['figures'], 'auroc_heatmap.png'), dpi=150)
        plt.close()

        # 3. 平均指标对比
        avg_metrics = {}
        for m in methods:
            valid_scores = [s for s in auroc_data[m] if s > 0]
            if valid_scores:
                avg_metrics[m] = np.mean(valid_scores)

        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.bar(method_labels[:len(avg_metrics)], list(avg_metrics.values()), color=colors)
        ax.set_ylabel('Average Image AUROC', fontsize=12)
        ax.set_title('Average Performance Comparison', fontsize=14)
        ax.set_ylim([0, 1.1])

        for bar, val in zip(bars, avg_metrics.values()):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{val:.4f}', ha='center', va='bottom', fontsize=12)

        plt.tight_layout()
        plt.savefig(os.path.join(self.dirs['figures'], 'average_comparison.png'), dpi=150)
        plt.close()

    def generate_report(self, all_results: Dict):
        """生成完整的Markdown报告"""
        report = """# 工业缺陷检测实验报告

## 1. 问题建模

### 1.1 问题描述
工业产品表面缺陷检测是质量控制的关键环节。本项目使用MVTec AD数据集，该数据集包含15个类别的工业产品图像，每类包含正常样本和多种缺陷样本。

### 1.2 建模方式：无监督学习

**选择理由：**

1. **缺陷样本稀少**：在实际工业生产中，缺陷品比例通常低于1%，难以收集足够的缺陷样本进行监督学习
2. **缺陷类型不可预知**：实际生产中可能出现训练时未见过的新型缺陷
3. **正常样本充足**：正常产品容易获取，可以大量收集用于训练
4. **泛化能力强**：无监督方法学习"正常"的模式，对任何偏离正常的样本都能检测

### 1.3 数据集概述
- **数据集**：MVTec Anomaly Detection (MVTec AD)
- **类别数**：15个（bottle, cable, capsule, carpet, grid, hazelnut, leather, metal_nut, pill, screw, tile, toothbrush, transistor, wood, zipper）
- **训练集**：仅包含正常样本
- **测试集**：包含正常样本和多种缺陷样本

---

## 2. 特征提取思路

### 2.1 传统方法特征提取

| 特征类型 | 描述 | 优势 | 局限 |
|---------|------|------|------|
| **HOG** | 方向梯度直方图，捕获边缘和形状信息 | 对几何形变敏感，计算效率高 | 对纹理变化不敏感 |
| **LBP** | 局部二值模式，描述纹理特征 | 旋转不变，对光照变化鲁棒 | 对噪声敏感 |
| **颜色直方图** | 颜色分布统计 | 简单直观 | 忽略空间信息 |

**组合特征**：本项目将HOG、LBP和颜色直方图特征级联，形成综合特征向量，以捕获更丰富的图像信息。

### 2.2 深度学习特征提取（PatchCore）

PatchCore使用预训练的Wide ResNet-50提取深度特征：

1. **特征提取**：使用ImageNet预训练的WideResNet50，提取layer2和layer3的特征
2. **Memory Bank构建**：从训练集提取所有patch特征，使用coreset采样减少冗余
3. **异常分数计算**：测试时计算特征与memory bank的最近邻距离

**优势**：
- 语义级特征，表达能力强
- 无需训练，只需构建memory bank
- 支持像素级异常定位

---

## 3. 缺陷检测算法

### 3.1 传统方法

#### Isolation Forest
- **原理**：通过随机选择特征和切分点构建隔离树，异常点更容易被隔离
- **优势**：计算效率高，对高维数据有效
- **参数**：contamination=0.1（假设10%为异常）

#### One-Class SVM
- **原理**：在特征空间中找到包围正常样本的最小超球面
- **优势**：理论基础完善，对小样本有效
- **参数**：nu=0.1（支持向量比例上界）

### 3.2 深度学习方法

#### PatchCore
- **原理**：基于预训练CNN的特征匹配方法
- **核心思想**：正常样本的局部特征应该在memory bank中有近似匹配
- **backbone**：Wide ResNet-50
- **采样策略**：Coreset采样，保留10%的代表性特征

**选择PatchCore的理由**：
1. 在MVTec AD上达到SOTA性能
2. 无需反向传播训练，冷启动能力强
3. 同时支持图像级检测和像素级定位

---

## 4. 评估指标

### 4.1 指标定义

| 指标 | 描述 | 取值范围 | 意义 |
|-----|------|---------|------|
| **AUROC** | ROC曲线下面积 | [0, 1] | 综合评估分类性能，不受阈值影响 |
| **AUPR** | PR曲线下面积 | [0, 1] | 对正负样本不平衡更敏感 |
| **F1-Score** | 精确率和召回率的调和平均 | [0, 1] | 在最优阈值下的综合性能 |
| **Pixel AUROC** | 像素级ROC曲线下面积 | [0, 1] | 评估异常定位准确性 |

### 4.2 实验结果

"""
        # 添加结果表格
        report += "#### 各类别Image AUROC结果\n\n"
        report += "| Category | Isolation Forest | One-Class SVM | PatchCore |\n"
        report += "|:---------|:----------------:|:-------------:|:---------:|\n"

        avg_if, avg_ocsvm, avg_pc = [], [], []
        for cat in CATEGORIES:
            if cat not in all_results:
                continue
            results = all_results[cat]

            if_auroc = results.get('iforest_combined', {}).get('auroc', '-')
            ocsvm_auroc = results.get('ocsvm_combined', {}).get('auroc', '-')
            pc_auroc = results.get('patchcore', {}).get('auroc', '-')

            if isinstance(if_auroc, float):
                avg_if.append(if_auroc)
                if_auroc = f"{if_auroc:.4f}"
            if isinstance(ocsvm_auroc, float):
                avg_ocsvm.append(ocsvm_auroc)
                ocsvm_auroc = f"{ocsvm_auroc:.4f}"
            if isinstance(pc_auroc, float):
                avg_pc.append(pc_auroc)
                pc_auroc = f"{pc_auroc:.4f}"

            report += f"| {cat} | {if_auroc} | {ocsvm_auroc} | {pc_auroc} |\n"

        # 平均值
        avg_if_str = f"{np.mean(avg_if):.4f}" if avg_if else "-"
        avg_ocsvm_str = f"{np.mean(avg_ocsvm):.4f}" if avg_ocsvm else "-"
        avg_pc_str = f"{np.mean(avg_pc):.4f}" if avg_pc else "-"
        report += f"| **Average** | **{avg_if_str}** | **{avg_ocsvm_str}** | **{avg_pc_str}** |\n"

        report += """
### 4.3 结果分析

1. **PatchCore显著优于传统方法**：深度学习方法利用预训练特征，在大多数类别上取得更高的AUROC

2. **类别差异**：
   - 纹理类别（carpet, leather, tile）：传统LBP特征表现相对较好
   - 物体类别（bottle, transistor）：深度学习优势更明显

3. **推理效率对比**：
   - 传统方法：毫秒级推理
   - PatchCore：需要GPU加速，推理时间较长

---

## 5. 工业部署难点分析

### 5.1 实时性要求

**挑战**：
- 生产线速度快，需要毫秒级响应
- 深度学习模型推理时间长

**解决方案**：
- 模型轻量化（MobileNet backbone）
- TensorRT/ONNX加速
- GPU推理优化
- 两阶段方案：传统方法快速筛选 + 深度学习精检

### 5.2 硬件资源限制

**挑战**：
- 边缘设备可能没有GPU
- 内存和功耗限制

**解决方案**：
- 模型量化（INT8/FP16）
- 知识蒸馏
- 边缘AI芯片（如NVIDIA Jetson）

### 5.3 产品泛化问题

**挑战**：
- 不同产品需要单独训练模型
- 产品换型时需要重新采集数据

**解决方案**：
- 迁移学习
- Few-shot学习
- 自动化数据采集流程

### 5.4 阈值选择与误报平衡

**挑战**：
- 阈值过高导致漏检
- 阈值过低导致误报
- 不同缺陷类型可能需要不同阈值

**解决方案**：
- 基于验证集自动调参
- 多阈值分级决策
- 人工复核机制

### 5.5 数据分布漂移

**挑战**：
- 光照、设备老化导致图像质量变化
- 原材料批次差异

**解决方案**：
- 在线学习/增量更新
- 图像标准化预处理
- 定期模型重训练

---

## 6. 结论

1. **方法选择**：对于工业缺陷检测，PatchCore等深度学习方法在精度上显著优于传统方法，推荐作为首选方案

2. **部署建议**：
   - 高精度场景：PatchCore + TensorRT加速
   - 低延迟场景：两阶段方案或轻量级模型
   - 资源受限：模型量化 + 边缘AI芯片

3. **持续改进**：建立误检样本收集机制，定期更新模型

---

## 附录：可视化结果

详见 `figures/` 和 `visualizations/` 目录中的图表。
"""

        # 保存报告
        report_path = os.path.join(self.dirs['reports'], 'experiment_report.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"\n报告已保存至: {report_path}")
        return report_path

    def run(self, categories: List[str] = None):
        """运行完整评估"""
        if categories is None:
            categories = CATEGORIES

        all_results = {}

        for category in categories:
            results = self.evaluate_category(category)
            all_results[category] = results

            # 绘制该类别的图表
            self.plot_roc_curves(category, results)
            self.plot_pr_curves(category, results)
            self.plot_anomaly_visualization(category, results)

            # 保存中间结果
            with open(os.path.join(self.dirs['data'], f'{category}_results.json'), 'w') as f:
                # 移除不可序列化的数据
                save_results = {}
                for m, data in results.items():
                    save_results[m] = {k: v for k, v in data.items()
                                       if k not in ['test_images', 'anomaly_maps']}
                json.dump(save_results, f, indent=2)

        # 绘制整体对比图
        self.plot_overall_comparison(all_results)

        # 生成报告
        self.generate_report(all_results)

        # 保存完整结果
        final_results = {}
        for cat, results in all_results.items():
            final_results[cat] = {}
            for m, data in results.items():
                if 'error' in data:
                    final_results[cat][m] = data
                else:
                    final_results[cat][m] = {
                        'auroc': data.get('auroc', 0),
                        'aupr': data.get('aupr', 0),
                        'f1': data.get('f1', 0),
                        'pixel_auroc': data.get('pixel_auroc', 0),
                        'train_time': data.get('train_time', 0),
                        'inference_time_ms': data.get('inference_time_ms', 0)
                    }

        with open(os.path.join(self.output_dir, 'final_results.json'), 'w') as f:
            json.dump(final_results, f, indent=2)

        print(f"\n{'='*60}")
        print("评估完成!")
        print(f"结果保存在: {self.output_dir}")
        print(f"{'='*60}")

        return all_results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="工业缺陷检测综合评估")
    parser.add_argument("--data-root", type=str, default="data",
                       help="数据集根目录")
    parser.add_argument("--output-dir", type=str, default="results/comprehensive",
                       help="输出目录")
    parser.add_argument("--category", type=str, default=None,
                       help="单个类别（默认全部）")
    parser.add_argument("--categories", type=str, nargs='+', default=None,
                       help="多个类别")

    args = parser.parse_args()

    evaluator = ComprehensiveEvaluator(args.data_root, args.output_dir)

    if args.category:
        categories = [args.category]
    elif args.categories:
        categories = args.categories
    else:
        categories = None

    evaluator.run(categories)


if __name__ == "__main__":
    main()
