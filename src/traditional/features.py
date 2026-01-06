"""
传统特征提取方法

包含:
- HOG (Histogram of Oriented Gradients): 捕获边缘和梯度信息
- LBP (Local Binary Patterns): 捕获纹理特征
- 颜色直方图: 捕获颜色分布
"""

import numpy as np
from typing import Tuple, Optional
from skimage.feature import hog, local_binary_pattern
from skimage.color import rgb2gray
from skimage import img_as_float
import cv2


def extract_hog_features(
    image: np.ndarray,
    orientations: int = 9,
    pixels_per_cell: Tuple[int, int] = (8, 8),
    cells_per_block: Tuple[int, int] = (2, 2),
    block_norm: str = 'L2-Hys'
) -> np.ndarray:
    """
    提取HOG (Histogram of Oriented Gradients) 特征

    HOG特征通过计算图像局部区域的梯度方向直方图来描述图像的边缘和形状信息。
    在工业缺陷检测中，HOG可以捕获缺陷区域的边缘特征。

    Args:
        image: 输入图像 (H, W, C) 或 (H, W)
        orientations: 梯度方向的bin数量
        pixels_per_cell: 每个cell的像素数
        cells_per_block: 每个block的cell数
        block_norm: 归一化方法

    Returns:
        HOG特征向量
    """
    # 转换为灰度图
    if len(image.shape) == 3:
        gray = rgb2gray(image)
    else:
        gray = image

    # 确保值在[0,1]范围
    gray = img_as_float(gray)

    # 提取HOG特征
    features = hog(
        gray,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm=block_norm,
        visualize=False,
        feature_vector=True
    )

    return features


def extract_lbp_features(
    image: np.ndarray,
    radius: int = 3,
    n_points: Optional[int] = None,
    method: str = 'uniform',
    n_bins: int = 26
) -> np.ndarray:
    """
    提取LBP (Local Binary Pattern) 特征

    LBP通过比较中心像素与周围像素的大小关系来描述纹理特征。
    在工业检测中，LBP可以捕获表面纹理的变化，对纹理类缺陷敏感。

    Args:
        image: 输入图像 (H, W, C) 或 (H, W)
        radius: LBP采样半径
        n_points: 采样点数量 (默认为 8 * radius)
        method: LBP计算方法 ('uniform', 'default', 'ror', 'nri_uniform')
        n_bins: 直方图bin数量

    Returns:
        LBP直方图特征
    """
    # 转换为灰度图
    if len(image.shape) == 3:
        gray = rgb2gray(image)
        gray = (gray * 255).astype(np.uint8)
    else:
        gray = image.astype(np.uint8)

    if n_points is None:
        n_points = 8 * radius

    # 计算LBP
    lbp = local_binary_pattern(gray, n_points, radius, method=method)

    # 计算直方图
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))

    # 归一化
    hist = hist.astype(np.float32)
    hist /= (hist.sum() + 1e-7)

    return hist


def extract_color_histogram(
    image: np.ndarray,
    bins_per_channel: int = 32,
    color_space: str = 'RGB'
) -> np.ndarray:
    """
    提取颜色直方图特征

    颜色直方图统计图像中各颜色值的分布。
    在工业检测中，颜色特征可以捕获因缺陷导致的颜色变化（如污渍、变色）。

    Args:
        image: 输入图像 (H, W, C)
        bins_per_channel: 每个通道的bin数量
        color_space: 颜色空间 ('RGB', 'HSV', 'LAB')

    Returns:
        颜色直方图特征
    """
    # 颜色空间转换
    if color_space == 'HSV':
        img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    elif color_space == 'LAB':
        img = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    else:
        img = image

    histograms = []

    for i in range(3):
        hist = cv2.calcHist(
            [img], [i], None, [bins_per_channel],
            [0, 256] if color_space != 'HSV' or i != 0 else [0, 180]
        )
        hist = hist.flatten()
        histograms.append(hist)

    # 拼接并归一化
    features = np.concatenate(histograms)
    features = features / (features.sum() + 1e-7)

    return features


def extract_combined_features(
    image: np.ndarray,
    use_hog: bool = True,
    use_lbp: bool = True,
    use_color: bool = True,
    hog_params: Optional[dict] = None,
    lbp_params: Optional[dict] = None,
    color_params: Optional[dict] = None
) -> np.ndarray:
    """
    提取组合特征

    将多种特征组合在一起，提供更全面的图像描述。

    Args:
        image: 输入图像 (H, W, C)
        use_hog: 是否使用HOG特征
        use_lbp: 是否使用LBP特征
        use_color: 是否使用颜色直方图
        hog_params: HOG参数
        lbp_params: LBP参数
        color_params: 颜色直方图参数

    Returns:
        组合特征向量
    """
    features = []

    if use_hog:
        params = hog_params or {}
        hog_feat = extract_hog_features(image, **params)
        features.append(hog_feat)

    if use_lbp:
        params = lbp_params or {}
        lbp_feat = extract_lbp_features(image, **params)
        features.append(lbp_feat)

    if use_color:
        params = color_params or {}
        color_feat = extract_color_histogram(image, **params)
        features.append(color_feat)

    return np.concatenate(features)


def extract_features_batch(
    images: np.ndarray,
    feature_type: str = 'combined',
    **kwargs
) -> np.ndarray:
    """
    批量提取特征

    Args:
        images: 图像数组 (N, H, W, C)
        feature_type: 特征类型 ('hog', 'lbp', 'color', 'combined')
        **kwargs: 特征提取参数

    Returns:
        特征矩阵 (N, D)
    """
    feature_funcs = {
        'hog': extract_hog_features,
        'lbp': extract_lbp_features,
        'color': extract_color_histogram,
        'combined': extract_combined_features
    }

    func = feature_funcs.get(feature_type)
    if func is None:
        raise ValueError(f"未知特征类型: {feature_type}")

    features = []
    for img in images:
        feat = func(img, **kwargs)
        features.append(feat)

    return np.array(features)


if __name__ == "__main__":
    # 测试代码
    import matplotlib.pyplot as plt
    from skimage.feature import hog as hog_vis

    # 生成测试图像
    test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    print("测试特征提取:")

    # HOG
    hog_feat = extract_hog_features(test_image)
    print(f"HOG特征维度: {hog_feat.shape}")

    # LBP
    lbp_feat = extract_lbp_features(test_image)
    print(f"LBP特征维度: {lbp_feat.shape}")

    # Color
    color_feat = extract_color_histogram(test_image)
    print(f"颜色直方图维度: {color_feat.shape}")

    # Combined
    combined_feat = extract_combined_features(test_image)
    print(f"组合特征维度: {combined_feat.shape}")

    # 批量提取
    batch_images = np.random.randint(0, 255, (10, 224, 224, 3), dtype=np.uint8)
    batch_feat = extract_features_batch(batch_images, feature_type='combined')
    print(f"批量特征矩阵shape: {batch_feat.shape}")
