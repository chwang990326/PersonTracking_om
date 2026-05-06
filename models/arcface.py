"""
ArcFace 人脸识别模块

ArcFace (Additive Angular Margin Loss for Deep Face Recognition) 
是一个先进的人脸识别模型，具有以下特点：
- 高准确率：在LFW等标准数据集上达到99.8%+的准确率
- 特征质量高：生成的特征向量具有良好的判别性
- 轻量级：模型大小仅为几十MB
- 快速推理：单张人脸识别时间 < 10ms

论文：https://arxiv.org/abs/1801.07698

核心算法：
- 人脸对齐：基于5个关键点的仿射变换
- 特征提取：深度卷积神经网络
- 损失函数：ArcFace损失（角度边界损失）
- 特征归一化：L2归一化
"""

import cv2
import numpy as np
from logging import getLogger
from onnxruntime import InferenceSession
from skimage.transform import SimilarityTransform
from typing import Tuple

__all__ = ["ArcFace"]

logger = getLogger(__name__)

# ArcFace的参考对齐点（标准的5个面部关键点位置）
# 这些点对应于：左眼、右眼、鼻子、左嘴角、右嘴角
reference_alignment: np.ndarray = np.array(
    [
        [38.2946, 51.6963],   # 左眼
        [73.5318, 51.5014],   # 右眼
        [56.0252, 71.7366],   # 鼻子
        [41.5493, 92.3655],   # 左嘴角
        [70.7299, 92.2041]    # 右嘴角
    ],
    dtype=np.float32
)


def estimate_norm(landmark: np.ndarray, image_size: int = 112) -> Tuple[np.ndarray, np.ndarray]:
    """
    估计人脸对齐的仿射变换矩阵。
    
    该函数基于检测到的5个面部关键点，计算将人脸对齐到标准位置的仿射变换矩阵。
    人脸对齐是人脸识别中的关键步骤，可以提高识别准确率。
    
    工作原理：
    1. 获取输入的5个关键点坐标
    2. 定义标准的5个关键点位置（参考对齐点）
    3. 计算从输入关键点到标准关键点的仿射变换
    4. 返回变换矩阵和其逆矩阵
    
    参数:
        landmark (np.ndarray): 人脸的5个关键点坐标，形状为 (5, 2)。
                             格式为 [[x1, y1], [x2, y2], ..., [x5, y5]]
                             对应于：左眼、右眼、鼻子、左嘴角、右嘴角
        image_size (int): 输出对齐人脸的大小。
                         默认值: 112 (ArcFace标准大小)
                         推荐值: 112 或 128

    返回:
        tuple: (matrix, inverse_matrix)
            matrix (np.ndarray): 2x3的仿射变换矩阵，用于对齐人脸
            inverse_matrix (np.ndarray): 2x3的逆变换矩阵
    
    异常:
        AssertionError: 如果输入的landmark形状不是 (5, 2) 或image_size不合法
    """
    # 验证输入
    assert landmark.shape == (5, 2), "Landmark array must have shape (5, 2)."
    assert image_size % 112 == 0 or image_size % 128 == 0, "Image size must be a multiple of 112 or 128."

    # 根据输出大小调整参考对齐点
    if image_size % 112 == 0:
        # 如果是112的倍数，按比例缩放
        ratio = float(image_size) / 112.0
        diff_x = 0.0
    else:
        # 如果是128的倍数，按比例缩放并调整x偏移
        ratio = float(image_size) / 128.0
        diff_x = 8.0 * ratio

    # 调整参考对齐点
    alignment = reference_alignment * ratio
    alignment[:, 0] += diff_x

    # 计算仿射变换矩阵
    # SimilarityTransform 计算从landmark到alignment的相似变换
    transform = SimilarityTransform()
    transform.estimate(landmark, alignment)

    # 提取2x3的变换矩阵
    matrix = transform.params[0:2, :]
    
    # 计算逆变换矩阵
    inverse_matrix = np.linalg.inv(transform.params)[0:2, :]

    return matrix, inverse_matrix


def face_alignment(image: np.ndarray, landmark: np.ndarray, image_size: int = 112) -> Tuple[np.ndarray, np.ndarray]:
    """
    基于面部关键点对人脸进行对齐。
    
    该函数使用仿射变换将人脸对齐到标准位置，这是人脸识别的关键预处理步骤。
    对齐后的人脸具有一致的方向和位置，有利于提高识别准确率。
    
    参数:
        image (np.ndarray): 输入图像，形状为 (height, width, 3)，BGR格式
        landmark (np.ndarray): 人脸的5个关键点坐标，形状为 (5, 2)
        image_size (int): 输出对齐人脸的大小，默认值: 112

    返回:
        tuple: (warped, M_inv)
            warped (np.ndarray): 对齐后的人脸图像，形状为 (image_size, image_size, 3)
            M_inv (np.ndarray): 逆变换矩阵
    """
    # 获取变换矩阵
    M, M_inv = estimate_norm(landmark, image_size)

    # 使用仿射变换对齐人脸
    warped = cv2.warpAffine(image, M, (image_size, image_size), borderValue=0.0)

    return warped, M_inv


class ArcFace:
    """
    ArcFace 人脸识别模型
    
    该类实现了ArcFace人脸识别模型的推理接口。ArcFace是一个先进的人脸识别模型，
    能够从人脸图像中提取高质量的特征向量，用于人脸匹配和识别。
    
    工作流程：
    1. 初始化模型，加载ONNX文件
    2. 对输入人脸进行对齐（基于关键点）
    3. 对对齐后的人脸进行预处理（缩放、归一化）
    4. 通过模型提取特征向量
    5. 返回512维的特征向量
    
    特征向量特性：
    - 维度：512维
    - 归一化：L2归一化（模长为1）
    - 相似度计算：余弦相似度
    - 准确率：在LFW数据集上达到99.8%+
    """

    def __init__(self, model_path: str) -> None:
        """
        初始化ArcFace人脸识别模型。

        参数:
            model_path (str): ArcFace模型的ONNX文件路径。
                            推荐使用: w600k_mbf.onnx (标准模型)
        
        异常:
            RuntimeError: 如果模型加载失败会抛出异常
        """
        self.model_path = model_path
        
        # 模型输入大小（ArcFace标准大小）
        self.input_size = (112, 112)
        
        # 图像归一化参数
        # 公式: normalized_image = (image - mean) / scale
        self.normalization_mean = 127.5
        self.normalization_scale = 127.5

        print(f"[ArcFace] Initializing ArcFace model from {self.model_path}")

        try:
            # 创建ONNX推理会话
            # 支持GPU加速（CUDA）和CPU推理
            self.session = InferenceSession(
                self.model_path,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )

            # 获取模型的输入配置
            input_config = self.session.get_inputs()[0]
            self.input_name = input_config.name

            # 验证模型输入大小
            input_shape = input_config.shape
            model_input_size = tuple(input_shape[2:4][::-1])
            if model_input_size != self.input_size:
                print(
                    f"[ArcFace] Warning: Model input size {model_input_size} differs from configured size {self.input_size}"
                )

            # 获取模型的输出配置
            self.output_names = [o.name for o in self.session.get_outputs()]
            self.output_shape = self.session.get_outputs()[0].shape
            
            # 特征向量的维度（通常为512）
            self.embedding_size = self.output_shape[1]

            # 验证只有一个输出节点
            assert len(self.output_names) == 1, "Expected only one output node."
            
            print(
                f"[ArcFace] Successfully initialized face encoder from {self.model_path} "
                f"(embedding size: {self.embedding_size})"
            )

        except Exception as e:
            print(f"[ArcFace] Failed to load face encoder model from '{self.model_path}': {e}")
            raise RuntimeError(f"Failed to initialize model session for '{self.model_path}'") from e

    def preprocess(self, face_image: np.ndarray) -> np.ndarray:
        """
        对人脸图像进行预处理。
        
        预处理步骤：
        1. 缩放到模型输入大小 (112, 112)
        2. 归一化像素值
        3. 转换为模型所需的格式 (NCHW)
        
        参数:
            face_image (np.ndarray): 输入人脸图像，形状为 (height, width, 3)，BGR格式
        
        返回:
            np.ndarray: 预处理后的图像blob，形状为 (1, 3, 112, 112)，可直接用于模型推理
        """
        # 缩放到模型输入大小
        resized_face = cv2.resize(face_image, self.input_size)

        if isinstance(self.normalization_scale, (list, tuple)):
            # 处理逐通道归一化
            rgb_face = cv2.cvtColor(resized_face, cv2.COLOR_BGR2RGB).astype(np.float32)

            mean_array = np.array(self.normalization_mean, dtype=np.float32)
            scale_array = np.array(self.normalization_scale, dtype=np.float32)
            normalized_face = (rgb_face - mean_array) / scale_array

            # 转换为NCHW格式 (batch, channels, height, width)
            transposed_face = np.transpose(normalized_face, (2, 0, 1))
            face_blob = np.expand_dims(transposed_face, axis=0)
        else:
            # 使用cv2.dnn进行单值归一化
            # 这是更高效的方法
            face_blob = cv2.dnn.blobFromImage(
                resized_face,
                scalefactor=1.0 / self.normalization_scale,
                size=self.input_size,
                mean=(self.normalization_mean,)*3,
                swapRB=True  # 交换R和B通道（BGR -> RGB）
            )
        return face_blob

    def get_embedding(
        self,
        image: np.ndarray,
        landmarks: np.ndarray,
        normalized: bool = False
    ) -> np.ndarray:
        """
        从人脸图像中提取特征向量。
        
        该方法是ArcFace的主要推理接口，执行以下步骤：
        1. 基于关键点对人脸进行对齐
        2. 对对齐后的人脸进行预处理
        3. 通过模型提取特征向量
        4. 可选地进行L2归一化
        
        工作流程示例：
        输入: 人脸图像 + 5个关键点
        ├─ 人脸对齐 → 对齐后的人脸 (112x112)
        ├─ 预处理 → 归一化的图像blob
        ├─ 模型推理 → 原始特征向量 (1, 512)
        └─ 展平 → 最终特征向量 (512,)
        
        参数:
            image (np.ndarray): 输入图像，形状为 (height, width, 3)，BGR格式
            landmarks (np.ndarray): 人脸的5个关键点坐标，形状为 (5, 2)。
                                   对应于：左眼、右眼、鼻子、左嘴角、右嘴角
            normalized (bool): 是否对输出特征向量进行L2归一化。
                             默认值: False
                             推荐值: False (数据库中的特征已归一化，查询时也应归一化)
        
        返回:
            np.ndarray: 人脸特征向量，形状为 (512,)
                       如果normalized=True，向量的L2范数为1
        
        异常:
            ValueError: 如果image或landmarks为None
            Exception: 如果特征提取过程出错
        """
        # 验证输入
        if image is None or landmarks is None:
            raise ValueError("Image and landmarks must not be None")

        try:
            # 步骤1: 对人脸进行对齐
            # 这是人脸识别的关键步骤，可以提高识别准确率
            aligned_face, _ = face_alignment(image, landmarks)
            
            # 步骤2: 预处理对齐后的人脸
            face_blob = self.preprocess(aligned_face)
            
            # 步骤3: 通过模型提取特征向量
            # 模型输出形状为 (1, 512)
            embedding = self.session.run(self.output_names, {self.input_name: face_blob})[0]

            # 步骤4: 可选的L2归一化
            if normalized:
                # L2归一化：使特征向量的模长为1
                # 这样可以使用余弦相似度进行匹配
                norm = np.linalg.norm(embedding, axis=1, keepdims=True)
                normalized_embedding = embedding / norm
                return normalized_embedding.flatten()

            # 展平为1维数组并返回
            return embedding.flatten()

        except Exception as e:
            print(f"[ArcFace] Error extracting face embedding: {e}")
            raise
