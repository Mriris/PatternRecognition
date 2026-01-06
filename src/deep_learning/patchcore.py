"""
基于 anomalib 的 PatchCore 深度学习异常检测

PatchCore 算法原理:
1. 使用预训练的CNN（如ResNet/WideResNet）提取图像的patch级特征
2. 从训练集（正常样本）中提取所有patch特征，构建memory bank
3. 使用coreset subsampling减少memory bank大小
4. 推理时，计算测试图像patch特征与memory bank中最近邻的距离
5. 最大距离作为图像级异常分数，距离图作为像素级异常热力图

优势:
- 无需训练（只需构建memory bank）
- 冷启动能力强（少量正常样本即可）
- 在MVTec AD上达到SOTA性能
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader

from anomalib.data import MVTec
from anomalib.models import Patchcore
from anomalib.engine import Engine


class PatchCoreDetector:
    """
    PatchCore 异常检测器封装

    提供简洁的训练和推理接口，封装anomalib的复杂配置。

    Args:
        backbone: 特征提取backbone ('resnet18', 'resnet50', 'wide_resnet50_2')
        layers: 使用的特征层
        coreset_sampling_ratio: coreset采样比例
        num_neighbors: K近邻数量
        image_size: 输入图像尺寸
    """

    def __init__(
        self,
        backbone: str = "wide_resnet50_2",
        layers: tuple = ("layer2", "layer3"),
        coreset_sampling_ratio: float = 0.1,
        num_neighbors: int = 9,
        image_size: tuple = (224, 224)
    ):
        self.backbone = backbone
        self.layers = list(layers)
        self.coreset_sampling_ratio = coreset_sampling_ratio
        self.num_neighbors = num_neighbors
        self.image_size = image_size

        self.model = None
        self.engine = None

    def _create_model(self) -> Patchcore:
        """创建PatchCore模型"""
        model = Patchcore(
            backbone=self.backbone,
            layers=self.layers,
            coreset_sampling_ratio=self.coreset_sampling_ratio,
            num_neighbors=self.num_neighbors,
        )
        return model

    def train(
        self,
        data_root: str,
        category: str,
        output_dir: str = "results",
        max_epochs: int = 1,
        devices: int = 1,
        accelerator: str = "auto"
    ) -> Dict[str, Any]:
        """
        训练模型（构建memory bank）

        PatchCore实际上不需要真正的"训练"，只需要一个epoch来构建memory bank。

        Args:
            data_root: MVTec数据集根目录
            category: 类别名称
            output_dir: 输出目录
            max_epochs: epoch数量（PatchCore只需1个）
            devices: GPU数量
            accelerator: 加速器类型

        Returns:
            训练结果字典
        """
        # 创建数据模块
        datamodule = MVTec(
            root=data_root,
            category=category,
            train_batch_size=32,
            eval_batch_size=32,
            num_workers=4,
        )

        # 创建模型
        self.model = self._create_model()

        # 创建引擎
        self.engine = Engine(
            max_epochs=max_epochs,
            devices=devices,
            accelerator=accelerator,
            default_root_dir=output_dir,
            enable_checkpointing=True,
            logger=True,
        )

        # 训练（构建memory bank）
        self.engine.fit(model=self.model, datamodule=datamodule)

        return {"status": "completed", "category": category}

    def test(
        self,
        data_root: str,
        category: str,
    ) -> Dict[str, float]:
        """
        测试模型

        Args:
            data_root: MVTec数据集根目录
            category: 类别名称

        Returns:
            测试结果字典，包含各项指标
        """
        if self.model is None or self.engine is None:
            raise RuntimeError("模型未训练，请先调用train()方法")

        # 创建数据模块
        datamodule = MVTec(
            root=data_root,
            category=category,
            eval_batch_size=32,
            num_workers=4,
        )

        # 测试
        results = self.engine.test(model=self.model, datamodule=datamodule)

        return results[0] if results else {}

    def predict(
        self,
        images: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        对图像进行预测

        Args:
            images: 输入图像tensor (N, C, H, W)

        Returns:
            image_scores: 图像级异常分数 (N,)
            anomaly_maps: 像素级异常热力图 (N, H, W)
        """
        if self.model is None:
            raise RuntimeError("模型未训练，请先调用train()方法")

        self.model.eval()
        device = next(self.model.parameters()).device

        with torch.no_grad():
            images = images.to(device)
            outputs = self.model(images)

        image_scores = outputs["pred_scores"].cpu().numpy()
        anomaly_maps = outputs["anomaly_maps"].cpu().numpy().squeeze(1)

        return image_scores, anomaly_maps

    def save(self, path: str) -> None:
        """保存模型"""
        if self.model is None:
            raise RuntimeError("模型未训练")

        torch.save({
            "model_state_dict": self.model.state_dict(),
            "config": {
                "backbone": self.backbone,
                "layers": self.layers,
                "coreset_sampling_ratio": self.coreset_sampling_ratio,
                "num_neighbors": self.num_neighbors,
                "image_size": self.image_size
            }
        }, path)

    def load(self, path: str) -> None:
        """加载模型"""
        checkpoint = torch.load(path, map_location="cpu")

        config = checkpoint["config"]
        self.backbone = config["backbone"]
        self.layers = config["layers"]
        self.coreset_sampling_ratio = config["coreset_sampling_ratio"]
        self.num_neighbors = config["num_neighbors"]
        self.image_size = config["image_size"]

        self.model = self._create_model()
        self.model.load_state_dict(checkpoint["model_state_dict"])


def train_patchcore_all_categories(
    data_root: str,
    categories: list,
    output_dir: str = "results/patchcore",
    **kwargs
) -> Dict[str, Dict[str, float]]:
    """
    在所有类别上训练并测试PatchCore

    Args:
        data_root: MVTec数据集根目录
        categories: 类别列表
        output_dir: 输出目录
        **kwargs: PatchCore参数

    Returns:
        各类别的测试结果
    """
    from tqdm import tqdm

    results = {}

    for category in tqdm(categories, desc="训练PatchCore"):
        print(f"\n{'='*50}")
        print(f"处理类别: {category}")
        print(f"{'='*50}")

        detector = PatchCoreDetector(**kwargs)

        cat_output_dir = os.path.join(output_dir, category)
        os.makedirs(cat_output_dir, exist_ok=True)

        # 训练
        detector.train(
            data_root=data_root,
            category=category,
            output_dir=cat_output_dir
        )

        # 测试
        test_results = detector.test(
            data_root=data_root,
            category=category
        )

        results[category] = test_results
        print(f"结果: {test_results}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PatchCore 训练测试")
    parser.add_argument("--data-root", type=str, default="data/mvtec_anomaly_detection")
    parser.add_argument("--category", type=str, default="bottle")
    parser.add_argument("--output-dir", type=str, default="results/patchcore")
    args = parser.parse_args()

    print(f"训练 PatchCore: {args.category}")

    detector = PatchCoreDetector()
    detector.train(
        data_root=args.data_root,
        category=args.category,
        output_dir=args.output_dir
    )

    results = detector.test(
        data_root=args.data_root,
        category=args.category
    )

    print(f"\n测试结果:")
    for key, value in results.items():
        print(f"  {key}: {value}")
