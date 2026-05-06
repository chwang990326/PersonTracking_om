"""
SCRFD 人脸检测模块

SCRFD (Sample and Computation Redistribution for Efficient Face Detection) 
是一个高效的人脸检测模型，具有以下特点：
- 检测速度快：50-100 FPS (GPU)
- 准确率高：在标准数据集上达到业界先进水平
- 轻量级：模型大小仅为几十MB
- 支持多尺度检测：通过FPN检测不同大小的人脸

论文：https://arxiv.org/abs/2105.04714

核心算法：
- 骨干网络：轻量级CNN
- 特征金字塔：FPN (Feature Pyramid Network)
- 检测头：多尺度检测头
- 后处理：NMS (Non-Maximum Suppression)
"""

import os
import cv2
import numpy as np
try:
    import onnxruntime
except ImportError:
    onnxruntime = None

from typing import Tuple
from models.ascend_backend import AscendInferSession, is_om_path

__all__ = ["SCRFD"]


def distance2bbox(points, distance, max_shape=None):
    """
    将距离预测解码为边界框坐标。
    
    SCRFD模型预测的是每个点到四个边界的距离，而不是直接的边界框坐标。
    该函数将这些距离转换为标准的边界框格式 [x1, y1, x2, y2]。
    
    工作原理：
    给定一个点 (x, y) 和到四个边界的距离 (left, top, right, bottom)，
    可以计算出边界框的四个角：
    - x1 = x - left
    - y1 = y - top
    - x2 = x + right
    - y2 = y + bottom

    参数:
        points (np.ndarray): 点的坐标，形状为 (n, 2)，格式为 [x, y]
        distance (np.ndarray): 到四个边界的距离，形状为 (n, 4)，格式为 [left, top, right, bottom]
        max_shape (tuple): 图像的形状 (height, width)，用于裁剪超出边界的坐标

    返回:
        np.ndarray: 解码后的边界框，形状为 (n, 4)，格式为 [x1, y1, x2, y2]
    """
    # 计算边界框的四个角
    x1 = points[:, 0] - distance[:, 0]  # 左边界
    y1 = points[:, 1] - distance[:, 1]  # 上边界
    x2 = points[:, 0] + distance[:, 2]  # 右边界
    y2 = points[:, 1] + distance[:, 3]  # 下边界
    
    # 如果提供了图像形状，则裁剪超出边界的坐标
    if max_shape is not None:
        x1 = np.clip(x1, 0, max_shape[1])  # 裁剪到 [0, width]
        y1 = np.clip(y1, 0, max_shape[0])  # 裁剪到 [0, height]
        x2 = np.clip(x2, 0, max_shape[1])
        y2 = np.clip(y2, 0, max_shape[0])
    
    # 堆叠成 (n, 4) 的数组
    return np.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(points, distance, max_shape=None):
    """
    将距离预测解码为关键点坐标。
    
    类似于 distance2bbox，该函数将预测的距离转换为关键点坐标。
    SCRFD检测的是5个关键点：左眼、右眼、鼻子、左嘴角、右嘴角。

    参数:
        points (np.ndarray): 点的坐标，形状为 (n, 2)，格式为 [x, y]
        distance (np.ndarray): 到关键点的距离，形状为 (n, 10)，格式为 [x1, y1, x2, y2, ..., x5, y5]
        max_shape (tuple): 图像的形状 (height, width)，用于裁剪超出边界的坐标

    返回:
        np.ndarray: 解码后的关键点，形状为 (n, 10)，格式为 [x1, y1, x2, y2, ..., x5, y5]
    """
    preds = []
    
    # 每次处理两个值（x和y坐标）
    for i in range(0, distance.shape[1], 2):
        # 计算关键点的x坐标
        px = points[:, i % 2] + distance[:, i]
        # 计算关键点的y坐标
        py = points[:, i % 2 + 1] + distance[:, i + 1]
        
        # 裁剪超出边界的坐标
        if max_shape is not None:
            px = np.clip(px, 0, max_shape[1])
            py = np.clip(py, 0, max_shape[0])
        
        preds.append(px)
        preds.append(py)
    
    # 堆叠成 (n, 10) 的数组
    return np.stack(preds, axis=-1)


class SCRFD:
    """
    SCRFD 人脸检测器
    
    论文: "Sample and Computation Redistribution for Efficient Face Detection"
    链接: https://arxiv.org/abs/2105.04714
    
    该类实现了SCRFD人脸检测模型的推理接口。SCRFD是一个轻量级但高效的人脸检测模型，
    能够在保证准确率的同时实现快速的推理速度。
    
    核心特性：
    - 多尺度检测：通过FPN在不同尺度检测人脸
    - 高效推理：支持GPU加速，速度快
    - 关键点检测：同时检测5个面部关键点
    - 轻量级模型：模型大小小，易于部署
    
    工作流程：
    1. 初始化模型，加载ONNX文件
    2. 对输入图像进行预处理（缩放、归一化）
    3. 通过模型进行前向推理
    4. 对输出进行解码和后处理（NMS）
    5. 返回检测到的人脸框和关键点
    """

    def __init__(
        self,
        model_path: str,
        input_size: Tuple[int] = (640, 640),
        conf_thres: float = 0.5,
        iou_thres: float = 0.4
    ) -> None:
        """
        初始化SCRFD人脸检测器。

        参数:
            model_path (str): SCRFD模型的ONNX文件路径。
                            推荐使用: det_500m.onnx (轻量级) 或 det_10g.onnx (高精度)
            input_size (tuple): 模型输入图像大小，格式为 (width, height)。
                               默认值: (640, 640)
                               推荐值: (640, 640) 或 (320, 320)
            conf_thres (float): 检测置信度阈值，范围 [0, 1]。
                               只有置信度 > 此阈值的检测结果才会被保留。
                               默认值: 0.5
                               推荐值: 0.3-0.7 (越小越容易检测到人脸)
            iou_thres (float): NMS (非极大值抑制) 的IoU阈值，范围 [0, 1]。
                              用于去除重叠的检测框。
                              默认值: 0.4
                              推荐值: 0.3-0.5
        
        异常:
            Exception: 如果模型加载失败会抛出异常
        """
        # 保存配置参数
        self.input_size = input_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        # SCRFD模型参数 ---------------
        # 特征金字塔的层数
        self.fmc = 3
        
        # 特征步长（stride）列表
        # 这些步长对应于FPN中不同层级的特征图
        # 步长越小，检测的人脸越小；步长越大，检测的人脸越大
        self._feat_stride_fpn = [8, 16, 32]
        
        # 每个位置的锚点数量
        self._num_anchors = 2
        
        # 是否检测关键点（5个面部关键点）
        self.use_kps = True

        # 图像归一化参数
        # 公式: normalized_image = (image - mean) / std
        self.mean = 127.5
        self.std = 128.0

        # 缓存中心点，用于加速推理
        # 避免重复计算相同大小的特征图的中心点
        self.center_cache = {}
        # --------------------------------

        # 初始化模型
        self._initialize_model(model_path=model_path)

    def _initialize_model(self, model_path: str):
        """
        从ONNX文件初始化模型。
        
        该方法加载SCRFD模型的ONNX文件，并创建推理会话。
        支持GPU加速（CUDA）和CPU推理。

        参数:
            model_path (str): SCRFD模型的ONNX文件路径
        
        异常:
            Exception: 如果模型加载失败会抛出异常
        """
        try:
            self.use_om = is_om_path(model_path)
            if self.use_om:
                self.session = AscendInferSession(model_path)
                self.output_names = []
                self.input_names = ["input.1"]
                print(f"[SCRFD] Ascend OM model loaded successfully from {model_path}")
                return

            if onnxruntime is None:
                raise ImportError("onnxruntime is required for non-OM SCRFD models")

            # 创建ONNX推理会话
            # providers 参数指定优先使用的执行提供者
            # "CUDAExecutionProvider": 使用NVIDIA GPU加速（如果可用）
            # "CPUExecutionProvider": 回退到CPU推理
            self.session = onnxruntime.InferenceSession(
                model_path,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )
            
            # 获取模型的输出节点名称
            # SCRFD模型通常有多个输出：置信度、边界框、关键点等
            self.output_names = [x.name for x in self.session.get_outputs()]
            
            # 获取模型的输入节点名称
            self.input_names = [x.name for x in self.session.get_inputs()]
            
            print(f"[SCRFD] Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"[SCRFD] Failed to load the model: {e}")
            raise

    def forward(self, image, threshold):
        """
        SCRFD模型的前向推理。
        
        该方法执行以下步骤：
        1. 对输入图像进行预处理（缩放、归一化）
        2. 通过ONNX模型进行推理
        3. 对模型输出进行解码
        4. 根据置信度阈值过滤检测结果
        
        参数:
            image (np.ndarray): 输入图像，形状为 (height, width, 3)，BGR格式
            threshold (float): 置信度阈值，只保留置信度 > 此值的检测结果
        
        返回:
            tuple: (scores_list, bboxes_list, kpss_list)
                scores_list: 每个尺度的置信度列表
                bboxes_list: 每个尺度的边界框列表
                kpss_list: 每个尺度的关键点列表
        """
        scores_list = []  # 存储不同尺度的置信度
        bboxes_list = []  # 存储不同尺度的边界框
        kpss_list = []    # 存储不同尺度的关键点
        
        # 获取输入图像的大小
        input_size = tuple(image.shape[0:2][::-1])

        # 预处理：使用cv2.dnn.blobFromImage进行缩放、归一化和格式转换
        # 参数说明：
        # - scalefactor: 1.0 / self.std = 1.0 / 128.0
        # - size: 目标大小
        # - mean: 归一化的均值
        # - swapRB: 是否交换R和B通道（BGR -> RGB）
        blob = cv2.dnn.blobFromImage(
            image,
            1.0 / self.std,
            input_size,
            (self.mean, self.mean, self.mean),
            swapRB=True
        )
        
        # 通过ONNX模型进行推理
        if self.use_om:
            outputs = self.session.infer(blob)
        else:
            outputs = self.session.run(self.output_names, {self.input_names[0]: blob})

        # 获取blob的高度和宽度
        input_height = blob.shape[2]
        input_width = blob.shape[3]

        # 处理不同尺度的特征图
        fmc = self.fmc  # 特征金字塔的层数
        for idx, stride in enumerate(self._feat_stride_fpn):
            # 从模型输出中提取该尺度的预测结果
            scores = np.squeeze(outputs[idx]).reshape(-1)  # 置信度预测
            bbox_preds = np.squeeze(outputs[idx + fmc]).reshape(-1, 4)  # 边界框距离预测
            bbox_preds = bbox_preds * stride  # 缩放到原始尺度
            
            if self.use_kps:
                kps_preds = np.squeeze(outputs[idx + fmc * 2]).reshape(-1, 10) * stride  # 关键点距离预测

            # 计算该尺度的特征图大小
            height = input_height // stride
            width = input_width // stride
            
            # 计算锚点中心
            key = (height, width, stride)
            if key in self.center_cache:
                # 如果已缓存，直接使用缓存的锚点中心
                anchor_centers = self.center_cache[key]
            else:
                # 否则计算锚点中心
                # np.mgrid 生成网格坐标
                anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                
                # 如果有多个锚点，复制锚点中心
                if self._num_anchors > 1:
                    anchor_centers = np.stack([anchor_centers] * self._num_anchors, axis=1).reshape((-1, 2))
                
                # 缓存锚点中心（避免重复计算）
                if len(self.center_cache) < 100:
                    self.center_cache[key] = anchor_centers

            # 根据置信度阈值过滤检测结果
            pos_inds = np.where(scores >= threshold)[0]
            
            # 解码边界框
            bboxes = distance2bbox(anchor_centers, bbox_preds)
            
            # 提取超过阈值的检测结果
            pos_scores = scores[pos_inds].reshape(-1, 1)
            pos_bboxes = bboxes[pos_inds]
            
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)
            
            # 解码关键点
            if self.use_kps:
                kpss = distance2kps(anchor_centers, kps_preds)
                kpss = kpss.reshape((kpss.shape[0], -1, 2))  # 重塑为 (n, 5, 2)
                pos_kpss = kpss[pos_inds]
                kpss_list.append(pos_kpss)
        
        return scores_list, bboxes_list, kpss_list

    def detect(self, image, max_num=0, metric="max"):
        """
        检测图像中的所有人脸。
        
        该方法执行以下步骤：
        1. 对输入图像进行预处理（缩放以适应模型输入）
        2. 调用forward方法进行推理
        3. 对检测结果进行后处理（NMS、排序等）
        4. 可选地限制返回的人脸数量
        
        参数:
            image (np.ndarray): 输入图像，形状为 (height, width, 3)，BGR格式
            max_num (int): 最多返回的人脸数量。
                          0 表示返回所有检测到的人脸
                          > 0 表示返回最多max_num个人脸
            metric (str): 选择人脸的指标，可选值：
                         "max": 按面积大小排序（返回最大的人脸）
                         其他: 按面积和中心距离的加权值排序
        
        返回:
            tuple: (det, kpss)
                det: 检测到的人脸框，形状为 (n, 5)，格式为 [x1, y1, x2, y2, confidence]
                kpss: 对应的关键点，形状为 (n, 5, 2)，每个人脸有5个关键点
        """
        # 获取模型输入大小
        width, height = self.input_size

        # 计算图像宽高比和模型宽高比
        im_ratio = float(image.shape[0]) / image.shape[1]
        model_ratio = height / width
        
        # 计算缩放后的图像大小（保持宽高比）
        if im_ratio > model_ratio:
            # 图像更高，按高度缩放
            new_height = height
            new_width = int(new_height / im_ratio)
        else:
            # 图像更宽，按宽度缩放
            new_width = width
            new_height = int(new_width * im_ratio)

        # 计算缩放因子（用于后续坐标转换）
        det_scale = float(new_height) / image.shape[0]
        
        # 缩放图像
        resized_image = cv2.resize(image, (new_width, new_height))

        # 创建模型输入大小的图像（填充黑色边框）
        det_image = np.zeros((height, width, 3), dtype=np.uint8)
        det_image[:new_height, :new_width, :] = resized_image

        # 前向推理
        scores_list, bboxes_list, kpss_list = self.forward(det_image, self.conf_thres)
        if not bboxes_list or sum(len(b) for b in bboxes_list) == 0:
            empty_det = np.empty((0, 5), dtype=np.float32)
            empty_kps = np.empty((0, 5, 2), dtype=np.float32) if self.use_kps else None
            return empty_det, empty_kps

        # 合并不同尺度的检测结果
        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]  # 按置信度从高到低排序
        
        # 将边界框坐标缩放回原始图像大小
        bboxes = np.vstack(bboxes_list) / det_scale

        # 将关键点坐标缩放回原始图像大小
        if self.use_kps:
            kpss = np.vstack(kpss_list) / det_scale

        # 合并边界框和置信度
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]
        
        # 应用NMS去除重叠的检测框
        keep = self.nms(pre_det, iou_thres=self.iou_thres)
        det = pre_det[keep, :]
        
        # 对关键点应用相同的NMS过滤
        if self.use_kps:
            kpss = kpss[order, :, :]
            kpss = kpss[keep, :, :]
        else:
            kpss = None
        
        # 如果指定了最大人脸数量，则选择最相关的人脸
        if 0 < max_num < det.shape[0]:
            # 计算每个人脸的面积
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            
            # 计算图像中心
            image_center = image.shape[0] // 2, image.shape[1] // 2
            
            # 计算每个人脸中心到图像中心的距离
            offsets = np.vstack(
                [
                    (det[:, 0] + det[:, 2]) / 2 - image_center[1],
                    (det[:, 1] + det[:, 3]) / 2 - image_center[0],
                ]
            )
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
            
            # 根据指标选择人脸
            if metric == "max":
                # 按面积排序（返回最大的人脸）
                values = area
            else:
                # 按面积和中心距离的加权值排序
                values = (area - offset_dist_squared * 2.0)
            
            # 选择前max_num个人脸
            bindex = np.argsort(values)[::-1]
            bindex = bindex[0:max_num]
            det = det[bindex, :]
            if kpss is not None:
                kpss = kpss[bindex, :]
        
        return det, kpss

    def nms(self, dets, iou_thres):
        """
        非极大值抑制 (Non-Maximum Suppression, NMS)
        
        该方法用于去除重叠的检测框。NMS的工作原理：
        1. 按置信度从高到低排序所有检测框
        2. 选择置信度最高的框，添加到保留列表
        3. 计算该框与其他框的IoU (Intersection over Union)
        4. 删除IoU > 阈值的框（认为是重复检测）
        5. 重复步骤2-4，直到没有框剩余
        
        IoU计算公式：
        IoU = 交集面积 / 并集面积
            = 交集面积 / (框1面积 + 框2面积 - 交集面积)
        
        参数:
            dets (np.ndarray): 检测结果，形状为 (n, 5)，格式为 [x1, y1, x2, y2, confidence]
            iou_thres (float): IoU阈值，范围 [0, 1]。
                             IoU > 此值的框会被删除。
                             推荐值: 0.3-0.5
        
        返回:
            list: 保留的检测框的索引列表
        """
        # 提取边界框坐标和置信度
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        # 计算每个框的面积
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        
        # 按置信度从高到低排序
        order = scores.argsort()[::-1]

        keep = []  # 保留的框的索引
        
        # 迭代处理所有框
        while order.size > 0:
            # 选择置信度最高的框
            i = order[0]
            keep.append(i)
            
            # 计算该框与其他框的交集
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            # 计算交集的宽度和高度
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            
            # 计算交集面积
            inter = w * h
            
            # 计算IoU
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            # 保留IoU <= 阈值的框（即不重叠的框）
            indices = np.where(ovr <= iou_thres)[0]
            order = order[indices + 1]

        return keep
