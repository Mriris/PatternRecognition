"""
传统异常检测器

包含:
- Isolation Forest: 基于随机森林的异常检测
- One-Class SVM: 单类SVM，只需正常样本训练
"""

import numpy as np
from typing import Tuple, Optional
from abc import ABC, abstractmethod
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import joblib


class BaseAnomalyDetector(ABC):
    """异常检测器基类"""

    @abstractmethod
    def fit(self, X: np.ndarray) -> 'BaseAnomalyDetector':
        """训练模型 (只使用正常样本)"""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测标签 (0: 正常, 1: 异常)"""
        pass

    @abstractmethod
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """计算异常分数 (越大越异常)"""
        pass

    def fit_predict(self, X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """训练并预测"""
        self.fit(X_train)
        labels = self.predict(X_test)
        scores = self.decision_function(X_test)
        return labels, scores

    def save(self, path: str) -> None:
        """保存模型"""
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> 'BaseAnomalyDetector':
        """加载模型"""
        return joblib.load(path)


class IsolationForestDetector(BaseAnomalyDetector):
    """
    基于Isolation Forest的异常检测器

    原理:
    Isolation Forest通过随机选择特征和切分点来"隔离"样本。
    异常样本由于与正常样本不同，通常能被更快地隔离（需要更少的切分次数）。
    算法为每个样本计算一个异常分数，分数越高表示越异常。

    优势:
    - 对高维数据效果好
    - 计算效率高
    - 不需要指定异常比例

    Args:
        n_estimators: 树的数量
        max_samples: 每棵树的样本数
        contamination: 预估的异常比例
        random_state: 随机种子
        normalize: 是否对特征进行标准化
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_samples: str = 'auto',
        contamination: float = 0.1,
        random_state: int = 42,
        normalize: bool = True
    ):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.random_state = random_state
        self.normalize = normalize

        self.scaler = StandardScaler() if normalize else None
        self.model = IsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1
        )

    def fit(self, X: np.ndarray) -> 'IsolationForestDetector':
        """
        训练模型

        Args:
            X: 正常样本特征矩阵 (N, D)
        """
        if self.scaler:
            X = self.scaler.fit_transform(X)

        self.model.fit(X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测标签

        Args:
            X: 测试样本特征矩阵 (N, D)

        Returns:
            标签数组 (0: 正常, 1: 异常)
        """
        if self.scaler:
            X = self.scaler.transform(X)

        # Isolation Forest返回 1=正常, -1=异常，需要转换
        predictions = self.model.predict(X)
        return (predictions == -1).astype(int)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        计算异常分数

        Args:
            X: 测试样本特征矩阵 (N, D)

        Returns:
            异常分数数组 (越大越异常)
        """
        if self.scaler:
            X = self.scaler.transform(X)

        # Isolation Forest的decision_function返回负值表示异常，需要取反
        scores = -self.model.decision_function(X)
        return scores


class OneClassSVMDetector(BaseAnomalyDetector):
    """
    基于One-Class SVM的异常检测器

    原理:
    One-Class SVM在特征空间中学习一个超球面来包围正常样本。
    测试时，落在超球面外的样本被判定为异常。

    优势:
    - 理论基础扎实（核方法）
    - 对于紧凑的正常样本分布效果好
    - 可以通过核函数处理非线性边界

    Args:
        kernel: 核函数类型 ('rbf', 'linear', 'poly', 'sigmoid')
        nu: 训练误差的上界，也是支持向量比例的下界
        gamma: RBF核参数
        normalize: 是否对特征进行标准化
    """

    def __init__(
        self,
        kernel: str = 'rbf',
        nu: float = 0.1,
        gamma: str = 'scale',
        normalize: bool = True
    ):
        self.kernel = kernel
        self.nu = nu
        self.gamma = gamma
        self.normalize = normalize

        self.scaler = StandardScaler() if normalize else None
        self.model = OneClassSVM(
            kernel=kernel,
            nu=nu,
            gamma=gamma
        )

    def fit(self, X: np.ndarray) -> 'OneClassSVMDetector':
        """
        训练模型

        Args:
            X: 正常样本特征矩阵 (N, D)
        """
        if self.scaler:
            X = self.scaler.fit_transform(X)

        self.model.fit(X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测标签

        Args:
            X: 测试样本特征矩阵 (N, D)

        Returns:
            标签数组 (0: 正常, 1: 异常)
        """
        if self.scaler:
            X = self.scaler.transform(X)

        # One-Class SVM返回 1=正常, -1=异常，需要转换
        predictions = self.model.predict(X)
        return (predictions == -1).astype(int)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        计算异常分数

        Args:
            X: 测试样本特征矩阵 (N, D)

        Returns:
            异常分数数组 (越大越异常)
        """
        if self.scaler:
            X = self.scaler.transform(X)

        # One-Class SVM的decision_function返回负值表示异常，需要取反
        scores = -self.model.decision_function(X)
        return scores


class GaussianAnomalyDetector(BaseAnomalyDetector):
    """
    基于高斯分布的异常检测器 (作为简单baseline)

    原理:
    假设正常样本服从多元高斯分布，计算测试样本到分布中心的马氏距离作为异常分数。

    Args:
        normalize: 是否对特征进行标准化
    """

    def __init__(self, normalize: bool = True):
        self.normalize = normalize
        self.scaler = StandardScaler() if normalize else None
        self.mean = None
        self.cov_inv = None

    def fit(self, X: np.ndarray) -> 'GaussianAnomalyDetector':
        """训练模型"""
        if self.scaler:
            X = self.scaler.fit_transform(X)

        self.mean = np.mean(X, axis=0)

        # 计算协方差矩阵的逆（添加正则化防止奇异）
        cov = np.cov(X, rowvar=False)
        cov += np.eye(cov.shape[0]) * 1e-6
        self.cov_inv = np.linalg.inv(cov)

        return self

    def predict(self, X: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """预测标签"""
        scores = self.decision_function(X)

        if threshold is None:
            # 使用3-sigma规则
            threshold = np.percentile(scores, 90)

        return (scores > threshold).astype(int)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """计算马氏距离作为异常分数"""
        if self.scaler:
            X = self.scaler.transform(X)

        diff = X - self.mean
        mahal = np.sqrt(np.sum(diff @ self.cov_inv * diff, axis=1))
        return mahal


def create_detector(
    detector_type: str = 'isolation_forest',
    **kwargs
) -> BaseAnomalyDetector:
    """
    创建异常检测器

    Args:
        detector_type: 检测器类型
            - 'isolation_forest': Isolation Forest
            - 'ocsvm': One-Class SVM
            - 'gaussian': 高斯检测器
        **kwargs: 检测器参数

    Returns:
        异常检测器实例
    """
    detectors = {
        'isolation_forest': IsolationForestDetector,
        'ocsvm': OneClassSVMDetector,
        'gaussian': GaussianAnomalyDetector
    }

    detector_cls = detectors.get(detector_type)
    if detector_cls is None:
        raise ValueError(f"未知检测器类型: {detector_type}")

    return detector_cls(**kwargs)


if __name__ == "__main__":
    # 测试代码
    np.random.seed(42)

    # 生成测试数据
    # 正常样本：均值为0
    X_train = np.random.randn(100, 50)

    # 测试样本：混合正常和异常
    X_test_normal = np.random.randn(50, 50)
    X_test_anomaly = np.random.randn(50, 50) + 3  # 异常样本偏移
    X_test = np.vstack([X_test_normal, X_test_anomaly])
    y_test = np.array([0] * 50 + [1] * 50)

    print("测试异常检测器:")
    print("-" * 50)

    for name, detector_type in [
        ("Isolation Forest", "isolation_forest"),
        ("One-Class SVM", "ocsvm"),
        ("Gaussian", "gaussian")
    ]:
        detector = create_detector(detector_type)
        detector.fit(X_train)

        predictions = detector.predict(X_test)
        scores = detector.decision_function(X_test)

        # 计算准确率
        accuracy = (predictions == y_test).mean()

        print(f"\n{name}:")
        print(f"  准确率: {accuracy:.4f}")
        print(f"  异常分数范围: [{scores.min():.4f}, {scores.max():.4f}]")
        print(f"  正常样本平均分: {scores[y_test == 0].mean():.4f}")
        print(f"  异常样本平均分: {scores[y_test == 1].mean():.4f}")
