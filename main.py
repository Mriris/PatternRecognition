"""
工业缺陷检测完整评测脚本

在 MVTec AD 数据集上对比传统方法（隔离森林、单类SVM、高斯检测器）
和深度学习方法（PatchCore）的异常检测性能。
"""

import os
import sys
import json
import time
import warnings
import gc
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_curve, precision_recall_curve, auc, f1_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import cv2

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from src.data.dataset import CATEGORIES, MVTecDatasetForTraditional
from src.traditional.features import extract_combined_features, extract_hog_features, extract_lbp_features
from src.traditional.detector import IsolationForestDetector, OneClassSVMDetector, GaussianAnomalyDetector
from src.evaluation.metrics import compute_auroc, compute_aupr, compute_f1_at_best_threshold as compute_optimal_f1

# 中文字体设置
FONT_PATH = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
CN_FONT = fm.FontProperties(fname=FONT_PATH)
CN_FONT_S = fm.FontProperties(fname=FONT_PATH, size=10)
CN_FONT_M = fm.FontProperties(fname=FONT_PATH, size=11)
CN_FONT_L = fm.FontProperties(fname=FONT_PATH, size=12)
CN_FONT_XL = fm.FontProperties(fname=FONT_PATH, size=14)

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150

warnings.filterwarnings('ignore')

# 类别中文名
CAT_CN = {
    'bottle': '瓶子', 'cable': '电缆', 'capsule': '胶囊', 'carpet': '地毯',
    'grid': '网格', 'hazelnut': '榛子', 'leather': '皮革', 'metal_nut': '金属螺母',
    'pill': '药片', 'screw': '螺丝', 'tile': '瓷砖', 'toothbrush': '牙刷',
    'transistor': '晶体管', 'wood': '木材', 'zipper': '拉链'
}

# 方法中文名
METHOD_CN = {
    'iforest_combined': '隔离森林',
    'ocsvm_combined': '单类SVM',
    'gaussian_combined': '高斯检测器',
    'patchcore': 'PatchCore'
}


class ComprehensiveEvaluator:
    """工业缺陷检测综合评估器"""

    def __init__(self, data_root: str, output_dir: str = "results/comprehensive"):
        self.data_root = data_root
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.dirs = {
            'figures': os.path.join(output_dir, 'figures'),
            'visualizations': os.path.join(output_dir, 'visualizations'),
            'reports': os.path.join(output_dir, 'reports'),
            'data': os.path.join(output_dir, 'data'),
            'samples': os.path.join(output_dir, 'samples'),
            'features': os.path.join(output_dir, 'features')
        }
        for d in self.dirs.values():
            os.makedirs(d, exist_ok=True)

        self.results = {}
        self.method_colors = {
            'iforest_combined': '#3498db',
            'ocsvm_combined': '#2ecc71',
            'gaussian_combined': '#9b59b6',
            'patchcore': '#e74c3c'
        }

    def run_traditional_method(self, category: str, feature_type: str = 'combined',
                               detector_type: str = 'isolation_forest') -> Dict:
        """运行传统方法评估"""
        print(f"  传统方法 ({detector_type} + {feature_type})...")

        train_dataset = MVTecDatasetForTraditional(self.data_root, category, split='train')
        test_dataset = MVTecDatasetForTraditional(self.data_root, category, split='test')

        train_images, _ = train_dataset.get_images_and_labels()
        test_images, test_labels = test_dataset.get_images_and_labels()

        print(f"    提取训练集特征 ({len(train_images)} 张图像)...")
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

        print(f"    提取测试集特征 ({len(test_images)} 张图像)...")
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

        if detector_type == 'isolation_forest':
            detector = IsolationForestDetector(contamination=0.1)
        elif detector_type == 'gaussian':
            detector = GaussianAnomalyDetector()
        else:
            detector = OneClassSVMDetector(nu=0.1)

        if detector_type == 'ocsvm' and train_features.shape[1] > 200:
            print(f"    PCA降维...")
            n_components = min(100, train_features.shape[0] - 1, train_features.shape[1])
            pca = PCA(n_components=n_components)
            train_features_fit = pca.fit_transform(train_features)
            test_features_fit = pca.transform(test_features)
        else:
            train_features_fit = train_features
            test_features_fit = test_features

        start_time = time.time()
        detector.fit(train_features_fit)
        train_time = time.time() - start_time

        start_time = time.time()
        scores = detector.decision_function(test_features_fit)
        inference_time = (time.time() - start_time) / len(test_features) * 1000

        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

        auroc = compute_auroc(test_labels, scores)
        aupr = compute_aupr(test_labels, scores)
        f1, threshold = compute_optimal_f1(test_labels, scores)

        fpr, tpr, _ = roc_curve(test_labels, scores)
        precision, recall, _ = precision_recall_curve(test_labels, scores)

        predictions = (scores >= threshold).astype(int)
        cm = confusion_matrix(test_labels, predictions)

        print(f"    AUROC: {auroc:.4f}, F1: {f1:.4f}")

        # 选择展示样本
        normal_idx = np.where(test_labels == 0)[0][:4]
        anomaly_idx = np.where(test_labels == 1)[0][:4]
        sample_idx = np.concatenate([normal_idx, anomaly_idx])
        sample_images = [test_images[i] for i in sample_idx]
        sample_labels = test_labels[sample_idx]
        sample_pred = predictions[sample_idx]

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
            'predictions': predictions.tolist(),
            'confusion_matrix': cm.tolist(),
            'train_features': train_features,
            'test_features': test_features,
            'sample_images': sample_images,
            'sample_labels': sample_labels,
            'sample_pred': sample_pred
        }

    def run_patchcore(self, category: str) -> Dict:
        """运行PatchCore深度学习方法"""
        print(f"  PatchCore深度学习方法...")

        try:
            from anomalib.data import MVTec
            from anomalib.models import Patchcore
            from anomalib.engine import Engine
            import torch
        except ImportError:
            print("    anomalib未安装，跳过PatchCore")
            return {'error': 'anomalib未安装'}

        datamodule = MVTec(
            root=self.data_root,
            category=category,
            train_batch_size=32,
            eval_batch_size=32,
            num_workers=4,
        )

        model = Patchcore(
            backbone="wide_resnet50_2",
            layers=["layer2", "layer3"],
            coreset_sampling_ratio=0.1,
            num_neighbors=9,
        )

        engine = Engine(
            max_epochs=1,
            devices=1,
            accelerator="auto",
            default_root_dir=os.path.join(self.output_dir, 'patchcore', category),
            logger=False,
        )

        start_time = time.time()
        engine.fit(model=model, datamodule=datamodule)
        train_time = time.time() - start_time

        test_results = engine.test(model=model, datamodule=datamodule)

        # 解析测试结果
        if isinstance(test_results, list) and len(test_results) > 0:
            metrics = test_results[0] if isinstance(test_results[0], dict) else {}
        elif isinstance(test_results, dict):
            metrics = test_results
        else:
            metrics = {}

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
                # 处理anomalib新版本的ImageBatch格式
                if hasattr(batch, 'image'):
                    images = batch.image.to(device)
                    batch_labels = batch.gt_label.cpu().numpy()
                    batch_images_np = batch.image.cpu().numpy()
                else:
                    images = batch['image'].to(device)
                    batch_labels = batch.get('gt_label', batch.get('label')).cpu().numpy()
                    batch_images_np = batch['image'].cpu().numpy()

                outputs = model(images)

                # 处理anomalib新版本的InferenceBatch格式
                # outputs[0]: pred_score, outputs[2]: anomaly_map
                if hasattr(outputs, '__getitem__') and not isinstance(outputs, dict):
                    # InferenceBatch格式
                    scores = outputs[0].cpu().numpy()  # pred_score
                    anomaly_maps_batch = outputs[2].cpu().numpy() if len(outputs) > 2 else None  # anomaly_map
                elif isinstance(outputs, dict):
                    scores = outputs['pred_scores'].cpu().numpy()
                    anomaly_maps_batch = outputs.get('anomaly_maps', None)
                    if anomaly_maps_batch is not None:
                        anomaly_maps_batch = anomaly_maps_batch.cpu().numpy()
                elif hasattr(outputs, 'pred_score'):
                    scores = outputs.pred_score.cpu().numpy()
                    anomaly_maps_batch = outputs.anomaly_map.cpu().numpy() if hasattr(outputs, 'anomaly_map') else None
                else:
                    scores = outputs.cpu().numpy() if hasattr(outputs, 'cpu') else np.array([0])
                    anomaly_maps_batch = None

                all_scores.extend(scores.flatten().tolist())
                all_labels.extend(batch_labels.flatten().tolist())

                # 保存异常热力图
                if anomaly_maps_batch is not None and len(all_anomaly_maps) < 10:
                    for i in range(min(len(anomaly_maps_batch), 10 - len(all_anomaly_maps))):
                        am = anomaly_maps_batch[i]
                        # 确保anomaly_map是2D的
                        if am.ndim > 2:
                            am = am.squeeze()
                        if am.ndim == 2 and am.size > 0:
                            all_anomaly_maps.append(am)
                            all_images.append(batch_images_np[i].transpose(1, 2, 0))

        inference_time = (time.time() - start_time) / len(all_scores) * 1000

        scores = np.array(all_scores)
        labels = np.array(all_labels)
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

        fpr, tpr, _ = roc_curve(labels, scores)
        precision, recall, _ = precision_recall_curve(labels, scores)

        return {
            'auroc': metrics.get('image_AUROC', metrics.get('AUROC', 0)),
            'pixel_auroc': metrics.get('pixel_AUROC', 0),
            'aupr': auc(recall, precision) if len(recall) > 0 else 0,
            'f1': metrics.get('image_F1Score', metrics.get('F1Score', 0)),
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

    def evaluate_category(self, category: str, skip_deep_learning: bool = False) -> Dict:
        """评估单个类别"""
        print(f"\n{'='*60}")
        print(f"评估类别: {category} ({CAT_CN.get(category, category)})")
        print(f"{'='*60}")

        results = {}

        results['iforest_combined'] = self.run_traditional_method(
            category, 'combined', 'isolation_forest'
        )
        gc.collect()

        results['ocsvm_combined'] = self.run_traditional_method(
            category, 'combined', 'ocsvm'
        )
        gc.collect()

        results['gaussian_combined'] = self.run_traditional_method(
            category, 'combined', 'gaussian'
        )
        gc.collect()

        if not skip_deep_learning:
            try:
                results['patchcore'] = self.run_patchcore(category)
            except Exception as e:
                print(f"  PatchCore失败: {e}")
                results['patchcore'] = {'error': str(e)}
            gc.collect()

        return results

    def plot_roc_curves(self, category: str, results: Dict):
        """绘制ROC曲线对比"""
        fig, ax = plt.subplots(figsize=(8, 6))

        for method, data in results.items():
            if 'error' in data or 'fpr' not in data:
                continue
            fpr = np.array(data['fpr'])
            tpr = np.array(data['tpr'])
            auroc = data['auroc']
            color = self.method_colors.get(method, 'gray')
            label = METHOD_CN.get(method, method)
            ax.plot(fpr, tpr, color=color,
                   label=f"{label} (AUC={auroc:.4f})", linewidth=2)

        ax.plot([0, 1], [0, 1], 'k--', label='随机', linewidth=1)
        ax.set_xlabel('假阳性率 (FPR)', fontproperties=CN_FONT_M)
        ax.set_ylabel('真阳性率 (TPR)', fontproperties=CN_FONT_M)
        ax.set_title(f'ROC曲线 - {CAT_CN.get(category, category)}', fontproperties=CN_FONT_L)
        ax.legend(loc='lower right', prop=CN_FONT_S)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])

        plt.tight_layout()
        plt.savefig(os.path.join(self.dirs['figures'], f'roc_{category}.png'), dpi=150)
        plt.close()

    def plot_pr_curves(self, category: str, results: Dict):
        """绘制PR曲线对比"""
        fig, ax = plt.subplots(figsize=(8, 6))

        for method, data in results.items():
            if 'error' in data or 'precision' not in data:
                continue
            precision = np.array(data['precision'])
            recall = np.array(data['recall'])
            aupr = data['aupr']
            color = self.method_colors.get(method, 'gray')
            label = METHOD_CN.get(method, method)
            ax.plot(recall, precision, color=color,
                   label=f"{label} (AUPR={aupr:.4f})", linewidth=2)

        ax.set_xlabel('召回率', fontproperties=CN_FONT_M)
        ax.set_ylabel('精确率', fontproperties=CN_FONT_M)
        ax.set_title(f'PR曲线 - {CAT_CN.get(category, category)}', fontproperties=CN_FONT_L)
        ax.legend(loc='lower left', prop=CN_FONT_S)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])

        plt.tight_layout()
        plt.savefig(os.path.join(self.dirs['figures'], f'pr_{category}.png'), dpi=150)
        plt.close()

    def plot_score_distribution(self, category: str, results: Dict):
        """绘制异常分数分布"""
        n_methods = sum(1 for m, d in results.items() if 'error' not in d and 'scores' in d)
        if n_methods == 0:
            return

        fig, axes = plt.subplots(1, n_methods, figsize=(5*n_methods, 4))
        if n_methods == 1:
            axes = [axes]

        idx = 0
        for method, data in results.items():
            if 'error' in data or 'scores' not in data:
                continue

            scores = np.array(data['scores'])
            labels = np.array(data['labels'])
            ax = axes[idx]

            normal_scores = scores[labels == 0]
            anomaly_scores = scores[labels == 1]

            ax.hist(normal_scores, bins=30, alpha=0.6, label='正常', color='#2ecc71', density=True)
            ax.hist(anomaly_scores, bins=30, alpha=0.6, label='异常', color='#e74c3c', density=True)

            if 'threshold' in data:
                ax.axvline(x=data['threshold'], color='black', linestyle='--',
                          linewidth=2, label=f'阈值={data["threshold"]:.3f}')

            label = METHOD_CN.get(method, method)
            ax.set_title(f'{label}', fontproperties=CN_FONT_M)
            ax.set_xlabel('异常分数', fontproperties=CN_FONT_S)
            ax.set_ylabel('密度', fontproperties=CN_FONT_S)
            ax.legend(prop=CN_FONT_S)
            ax.grid(True, alpha=0.3)
            idx += 1

        plt.suptitle(f'分数分布 - {CAT_CN.get(category, category)}', fontproperties=CN_FONT_L)
        plt.tight_layout()
        plt.savefig(os.path.join(self.dirs['figures'], f'score_dist_{category}.png'), dpi=150)
        plt.close()

    def plot_confusion_matrices(self, category: str, results: Dict):
        """绘制混淆矩阵"""
        methods_with_cm = [(m, d) for m, d in results.items()
                          if 'error' not in d and 'confusion_matrix' in d]
        if not methods_with_cm:
            return

        n_methods = len(methods_with_cm)
        fig, axes = plt.subplots(1, n_methods, figsize=(5*n_methods, 4))
        if n_methods == 1:
            axes = [axes]

        for idx, (method, data) in enumerate(methods_with_cm):
            cm = np.array(data['confusion_matrix'])
            ax = axes[idx]

            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['正常', '异常'],
                       yticklabels=['正常', '异常'])

            label = METHOD_CN.get(method, method)
            ax.set_title(f'{label}\nF1={data.get("f1", 0):.4f}', fontproperties=CN_FONT_M)
            ax.set_xlabel('预测标签', fontproperties=CN_FONT_S)
            ax.set_ylabel('真实标签', fontproperties=CN_FONT_S)
            ax.set_xticklabels(['正常', '异常'], fontproperties=CN_FONT_S)
            ax.set_yticklabels(['正常', '异常'], fontproperties=CN_FONT_S)

        plt.suptitle(f'混淆矩阵 - {CAT_CN.get(category, category)}', fontproperties=CN_FONT_L)
        plt.tight_layout()
        plt.savefig(os.path.join(self.dirs['figures'], f'confusion_{category}.png'), dpi=150)
        plt.close()

    def plot_feature_visualization(self, category: str, results: Dict):
        """绘制特征空间可视化"""
        method = 'gaussian_combined'
        if method not in results or 'error' in results[method]:
            method = 'iforest_combined'
        if method not in results or 'error' in results[method]:
            return

        data = results[method]
        if 'train_features' not in data or 'test_features' not in data:
            return

        train_features = data['train_features']
        test_features = data['test_features']
        test_labels = np.array(data['labels'])

        all_features = np.vstack([train_features, test_features])
        n_train = len(train_features)

        all_labels = np.zeros(len(all_features))
        all_labels[n_train:][test_labels == 0] = 1
        all_labels[n_train:][test_labels == 1] = 2

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(all_features)

        colors = ['#3498db', '#2ecc71', '#e74c3c']
        labels_text = ['训练集', '测试-正常', '测试-异常']

        for i, (c, l) in enumerate(zip(colors, labels_text)):
            mask = all_labels == i
            axes[0].scatter(pca_result[mask, 0], pca_result[mask, 1],
                          c=c, label=l, alpha=0.6, s=30)

        axes[0].set_title(f'PCA可视化\n解释方差: {sum(pca.explained_variance_ratio_)*100:.1f}%',
                         fontproperties=CN_FONT_M)
        axes[0].set_xlabel('第一主成分', fontproperties=CN_FONT_S)
        axes[0].set_ylabel('第二主成分', fontproperties=CN_FONT_S)
        axes[0].legend(prop=CN_FONT_S)
        axes[0].grid(True, alpha=0.3)

        # t-SNE
        max_samples = 500
        if len(all_features) > max_samples:
            idx = np.random.choice(len(all_features), max_samples, replace=False)
            tsne_features = all_features[idx]
            tsne_labels = all_labels[idx]
        else:
            tsne_features = all_features
            tsne_labels = all_labels

        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(tsne_features)-1))
        tsne_result = tsne.fit_transform(tsne_features)

        for i, (c, l) in enumerate(zip(colors, labels_text)):
            mask = tsne_labels == i
            axes[1].scatter(tsne_result[mask, 0], tsne_result[mask, 1],
                          c=c, label=l, alpha=0.6, s=30)

        axes[1].set_title('t-SNE可视化', fontproperties=CN_FONT_M)
        axes[1].set_xlabel('t-SNE 1', fontproperties=CN_FONT_S)
        axes[1].set_ylabel('t-SNE 2', fontproperties=CN_FONT_S)
        axes[1].legend(prop=CN_FONT_S)
        axes[1].grid(True, alpha=0.3)

        plt.suptitle(f'特征空间可视化 - {CAT_CN.get(category, category)}', fontproperties=CN_FONT_L)
        plt.tight_layout()
        plt.savefig(os.path.join(self.dirs['features'], f'feature_vis_{category}.png'), dpi=150)
        plt.close()

    def plot_sample_gallery(self, category: str, results: Dict):
        """绘制样本展示"""
        method = 'gaussian_combined'
        if method not in results or 'error' in results[method]:
            method = 'iforest_combined'
        if method not in results or 'error' in results[method]:
            return

        data = results[method]
        if 'sample_images' not in data:
            return

        sample_images = data['sample_images']
        sample_labels = data['sample_labels']
        sample_pred = data['sample_pred']

        n_samples = min(len(sample_images), 8)
        if n_samples == 0:
            return

        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()

        for i in range(n_samples):
            img = sample_images[i]
            ax = axes[i]
            ax.imshow(img)

            true_label = '异常' if sample_labels[i] == 1 else '正常'
            pred_label = '异常' if sample_pred[i] == 1 else '正常'
            color = 'green' if sample_labels[i] == sample_pred[i] else 'red'

            ax.set_title(f'真实: {true_label}\n预测: {pred_label}',
                        fontproperties=CN_FONT_S, color=color)
            ax.axis('off')

        for i in range(n_samples, 8):
            axes[i].axis('off')

        plt.suptitle(f'检测样本展示 - {CAT_CN.get(category, category)}', fontproperties=CN_FONT_XL)
        plt.tight_layout()
        plt.savefig(os.path.join(self.dirs['samples'], f'samples_{category}.png'), dpi=150)
        plt.close()

    def plot_anomaly_visualization(self, category: str, results: Dict):
        """绘制PatchCore异常热力图"""
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
            img = patchcore_data['test_images'][i]
            # 确保img是uint8格式
            if img.max() <= 1:
                img = (img * 255).astype(np.uint8)
            elif img.dtype != np.uint8:
                img = img.astype(np.uint8)

            axes[i, 0].imshow(img)
            axes[i, 0].set_title('原图', fontproperties=CN_FONT_S)
            axes[i, 0].axis('off')

            am = patchcore_data['anomaly_maps'][i]
            im = axes[i, 1].imshow(am, cmap='jet')
            axes[i, 1].set_title('异常热力图', fontproperties=CN_FONT_S)
            axes[i, 1].axis('off')
            plt.colorbar(im, ax=axes[i, 1], fraction=0.046, pad=0.04)

            # 生成叠加图
            am_resized = cv2.resize(am, (img.shape[1], img.shape[0]))
            am_normalized = (am_resized - am_resized.min()) / (am_resized.max() - am_resized.min() + 1e-8)
            heatmap = cv2.applyColorMap((am_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            # 确保两个图像都是uint8
            overlay = cv2.addWeighted(img.astype(np.uint8), 0.6, heatmap.astype(np.uint8), 0.4, 0)
            axes[i, 2].imshow(overlay)
            axes[i, 2].set_title('叠加图', fontproperties=CN_FONT_S)
            axes[i, 2].axis('off')

        plt.suptitle(f'异常检测可视化 - {CAT_CN.get(category, category)}', fontproperties=CN_FONT_L)
        plt.tight_layout()
        plt.savefig(os.path.join(self.dirs['visualizations'], f'anomaly_vis_{category}.png'), dpi=150)
        plt.close()

    def plot_overall_comparison(self, all_results: Dict):
        """绘制总体对比图"""
        categories = []
        methods = ['iforest_combined', 'ocsvm_combined', 'gaussian_combined', 'patchcore']
        auroc_data = {m: [] for m in methods}
        f1_data = {m: [] for m in methods}

        for cat, results in all_results.items():
            if cat == 'average':
                continue
            categories.append(cat)
            for m in methods:
                if m in results and 'error' not in results[m]:
                    auroc_data[m].append(results[m].get('auroc', 0))
                    f1_data[m].append(results[m].get('f1', 0))
                else:
                    auroc_data[m].append(0)
                    f1_data[m].append(0)

        if not categories:
            return

        colors = [self.method_colors.get(m, 'gray') for m in methods]

        # 1. 柱状图对比
        x = np.arange(len(categories))
        width = 0.2
        fig, ax = plt.subplots(figsize=(max(16, len(categories)*1.5), 8))

        for i, m in enumerate(methods):
            if any(s > 0 for s in auroc_data[m]):
                values = auroc_data[m]
                ax.bar(x + i*width, values, width, label=METHOD_CN.get(m, m), color=colors[i])

        ax.set_xlabel('类别', fontproperties=CN_FONT_M)
        ax.set_ylabel('AUROC', fontproperties=CN_FONT_M)
        ax.set_title('各方法在不同类别上的性能对比', fontproperties=CN_FONT_L)
        ax.set_xticks(x + width*1.5)
        ax.set_xticklabels([CAT_CN.get(c, c) for c in categories],
                          fontproperties=CN_FONT_S, rotation=45, ha='right')
        ax.legend(prop=CN_FONT_S, loc='upper right')
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(os.path.join(self.dirs['figures'], 'overall_comparison.png'), dpi=150)
        plt.close()

        # 2. AUROC热力图
        valid_methods = [m for m in methods if any(s > 0 for s in auroc_data[m])]
        fig, ax = plt.subplots(figsize=(10, max(8, len(categories)*0.5)))
        data_matrix = np.array([auroc_data[m] for m in valid_methods]).T
        sns.heatmap(data_matrix, annot=True, fmt='.3f', cmap='RdYlGn',
                   xticklabels=[METHOD_CN.get(m, m) for m in valid_methods],
                   yticklabels=[CAT_CN.get(c, c) for c in categories],
                   vmin=0.3, vmax=1.0, ax=ax)
        ax.set_title('AUROC热力图', fontproperties=CN_FONT_L)
        ax.set_xticklabels([METHOD_CN.get(m, m) for m in valid_methods], fontproperties=CN_FONT_S)
        ax.set_yticklabels([CAT_CN.get(c, c) for c in categories], fontproperties=CN_FONT_S)
        plt.tight_layout()
        plt.savefig(os.path.join(self.dirs['figures'], 'auroc_heatmap.png'), dpi=150)
        plt.close()

        # 3. 平均性能对比
        avg_auroc = {m: np.mean([s for s in auroc_data[m] if s > 0]) if any(s > 0 for s in auroc_data[m]) else 0 for m in methods}
        avg_f1 = {m: np.mean([s for s in f1_data[m] if s > 0]) if any(s > 0 for s in f1_data[m]) else 0 for m in methods}

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        valid_methods = [m for m in methods if avg_auroc[m] > 0]
        bars = axes[0].bar([METHOD_CN.get(m, m) for m in valid_methods],
                          [avg_auroc[m] for m in valid_methods],
                          color=[self.method_colors.get(m, 'gray') for m in valid_methods])
        axes[0].set_ylabel('平均AUROC', fontproperties=CN_FONT_M)
        axes[0].set_title('平均AUROC对比', fontproperties=CN_FONT_L)
        axes[0].set_xticklabels([METHOD_CN.get(m, m) for m in valid_methods], fontproperties=CN_FONT_M)
        axes[0].set_ylim([0, 1.1])
        for bar, m in zip(bars, valid_methods):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{avg_auroc[m]:.4f}', ha='center', va='bottom', fontsize=11)
        axes[0].grid(True, alpha=0.3, axis='y')

        bars = axes[1].bar([METHOD_CN.get(m, m) for m in valid_methods],
                          [avg_f1[m] for m in valid_methods],
                          color=[self.method_colors.get(m, 'gray') for m in valid_methods])
        axes[1].set_ylabel('平均F1分数', fontproperties=CN_FONT_M)
        axes[1].set_title('平均F1对比', fontproperties=CN_FONT_L)
        axes[1].set_xticklabels([METHOD_CN.get(m, m) for m in valid_methods], fontproperties=CN_FONT_M)
        axes[1].set_ylim([0, 1.1])
        for bar, m in zip(bars, valid_methods):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{avg_f1[m]:.4f}', ha='center', va='bottom', fontsize=11)
        axes[1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(os.path.join(self.dirs['figures'], 'average_comparison.png'), dpi=150)
        plt.close()

        # 4. 雷达图
        self.plot_radar_chart(all_results, methods)

        # 5. 箱线图
        self.plot_auroc_boxplot(auroc_data, methods)

    def plot_radar_chart(self, all_results: Dict, methods: List):
        """绘制雷达图"""
        metrics_names = ['平均AUROC', '平均F1', '速度', '简洁性', '可解释性']

        auroc_scores = []
        f1_scores = []
        speed_scores = []

        for m in methods:
            aurocs = []
            f1s = []
            times = []
            for cat, results in all_results.items():
                if cat == 'average':
                    continue
                if m in results and 'error' not in results[m]:
                    aurocs.append(results[m].get('auroc', 0))
                    f1s.append(results[m].get('f1', 0))
                    times.append(results[m].get('inference_time_ms', 100))
            auroc_scores.append(np.mean(aurocs) if aurocs else 0)
            f1_scores.append(np.mean(f1s) if f1s else 0)
            avg_time = np.mean(times) if times else 100
            speed_scores.append(1 / (1 + avg_time/100))

        simplicity = {'iforest_combined': 0.9, 'ocsvm_combined': 0.7, 'gaussian_combined': 0.95, 'patchcore': 0.4}
        interpretability = {'iforest_combined': 0.6, 'ocsvm_combined': 0.5, 'gaussian_combined': 0.9, 'patchcore': 0.7}

        data = []
        for i, m in enumerate(methods):
            if auroc_scores[i] > 0:
                row = [
                    auroc_scores[i],
                    f1_scores[i],
                    speed_scores[i],
                    simplicity.get(m, 0.5),
                    interpretability.get(m, 0.5)
                ]
                data.append((m, row))

        if not data:
            return

        angles = np.linspace(0, 2*np.pi, len(metrics_names), endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

        for m, values in data:
            values = values + values[:1]
            color = self.method_colors.get(m, 'gray')
            ax.plot(angles, values, 'o-', linewidth=2, label=METHOD_CN.get(m, m), color=color)
            ax.fill(angles, values, alpha=0.15, color=color)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics_names, fontproperties=CN_FONT_M)
        ax.set_ylim(0, 1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), prop=CN_FONT_M)
        ax.set_title('方法综合特性对比', fontproperties=CN_FONT_XL, pad=20)

        plt.tight_layout()
        plt.savefig(os.path.join(self.dirs['figures'], 'radar_comparison.png'), dpi=150)
        plt.close()

    def plot_auroc_boxplot(self, auroc_data: Dict, methods: List):
        """绘制AUROC箱线图"""
        fig, ax = plt.subplots(figsize=(10, 6))

        valid_methods = [m for m in methods if any(s > 0 for s in auroc_data[m])]
        data_to_plot = [[s for s in auroc_data[m] if s > 0] for m in valid_methods]
        colors_list = [self.method_colors.get(m, 'gray') for m in valid_methods]

        bp = ax.boxplot(data_to_plot, patch_artist=True)

        for patch, color in zip(bp['boxes'], colors_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_xticklabels([METHOD_CN.get(m, m) for m in valid_methods], fontproperties=CN_FONT_M)
        ax.set_ylabel('AUROC', fontproperties=CN_FONT_M)
        ax.set_title('各方法AUROC分布', fontproperties=CN_FONT_L)
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3, axis='y')

        means = [np.mean([s for s in auroc_data[m] if s > 0]) for m in valid_methods]
        ax.scatter(range(1, len(valid_methods)+1), means, color='red', marker='D', s=50, zorder=5, label='均值')
        ax.legend(prop=CN_FONT_S)

        plt.tight_layout()
        plt.savefig(os.path.join(self.dirs['figures'], 'auroc_boxplot.png'), dpi=150)
        plt.close()

    def run(self, categories: List[str] = None, skip_deep_learning: bool = False):
        """运行完整评估"""
        if categories is None:
            categories = CATEGORIES

        all_results = {}

        for category in categories:
            results = self.evaluate_category(category, skip_deep_learning=skip_deep_learning)
            all_results[category] = results

            print(f"  生成 {category} 的可视化...")
            self.plot_roc_curves(category, results)
            self.plot_pr_curves(category, results)
            self.plot_score_distribution(category, results)
            self.plot_confusion_matrices(category, results)
            self.plot_feature_visualization(category, results)
            self.plot_sample_gallery(category, results)
            self.plot_anomaly_visualization(category, results)

            with open(os.path.join(self.dirs['data'], f'{category}_results.json'), 'w') as f:
                save_results = {}
                for m, data in results.items():
                    save_results[m] = {k: v for k, v in data.items()
                                       if k not in ['test_images', 'anomaly_maps', 'train_features',
                                                   'test_features', 'sample_images', 'sample_labels', 'sample_pred']}
                json.dump(save_results, f, indent=2)

            gc.collect()

        print("\n生成总体对比图...")
        self.plot_overall_comparison(all_results)

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

        with open(os.path.join(self.output_dir, 'final_results.json'), 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)

        print(f"\n{'='*60}")
        print("评估完成!")
        print(f"结果保存至: {self.output_dir}")
        print(f"{'='*60}")

        # 打印摘要
        print("\n结果摘要:")
        methods = ['iforest_combined', 'ocsvm_combined', 'gaussian_combined', 'patchcore']
        for m in methods:
            aurocs = []
            for cat in categories:
                if cat in all_results and m in all_results[cat] and 'error' not in all_results[cat][m]:
                    aurocs.append(all_results[cat][m].get('auroc', 0))
            if aurocs:
                print(f"  {METHOD_CN.get(m, m):12s} - 平均AUROC: {np.mean(aurocs):.4f}")

        return all_results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="工业缺陷检测完整评测")
    parser.add_argument("--data-root", type=str, default="data",
                       help="数据集根目录")
    parser.add_argument("--output-dir", type=str, default="results/comprehensive",
                       help="输出目录")
    parser.add_argument("--category", type=str, default=None,
                       help="单个类别")
    parser.add_argument("--categories", type=str, nargs='+', default=None,
                       help="多个类别")
    parser.add_argument("--skip-deep-learning", action="store_true",
                       help="跳过PatchCore深度学习方法")

    args = parser.parse_args()

    evaluator = ComprehensiveEvaluator(args.data_root, args.output_dir)

    if args.category:
        categories = [args.category]
    elif args.categories:
        categories = args.categories
    else:
        categories = None

    evaluator.run(categories, skip_deep_learning=args.skip_deep_learning)


if __name__ == "__main__":
    main()
