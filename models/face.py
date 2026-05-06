"""
人脸识别模块 - 使用SCRFD + AdaFace架构

该模块实现了一个完整的人脸检测和识别系统，使用AdaFace替代ArcFace。
主要特点：
- 使用SCRFD进行高效的人脸检测
- 使用AdaFace进行质量自适应的人脸识别 (CVPR 2022)
- 使用FAISS进行快速相似度搜索
- 完全兼容原有接口

关键参数建议：
- detection_threshold: 0.5 (范围 0.3-0.7，越小越容易检测到人脸)
- similarity_threshold: 0.35 (范围 0.3-0.5，越小越容易匹配)
"""

import numpy as np
import os
import cv2
import time
import logging
from models.scrfd import SCRFD
from models.adaface import AdaFace
from models.face_database import FaceDatabase
from models.ascend_backend import resolve_model_path


class FaceRecognizer:
    """
    SCRFD + AdaFace 人脸识别系统
    
    该类集成了SCRFD人脸检测器和AdaFace人脸识别器，提供了完整的人脸检测和识别功能。
    
    工作流程：
    1. 初始化时加载SCRFD检测模型和AdaFace识别模型
    2. 从人脸库目录加载所有已知人脸的特征向量
    3. 在推理时，先用SCRFD检测人脸，再用AdaFace提取特征
    4. 通过FAISS进行快速相似度匹配
    """
    
    def __init__(self, 
                 face_gallery_path='faceImage',
                 scrfd_model_path='weights/det_10g.onnx',
                 adaface_model_path='weights/adaface_ir50_ms1mv2.ckpt',
                 architecture='ir_50',
                 similarity_threshold=0.35,
                 detection_threshold=0.5,
                 db_path='./database/face_database'):
        """
        初始化SCRFD+AdaFace人脸识别系统。

        参数:
            face_gallery_path (str): 存放已知人脸照片的根文件夹路径。
            scrfd_model_path (str): SCRFD人脸检测模型的ONNX文件路径。
            adaface_model_path (str): AdaFace人脸识别模型的ckpt文件路径。
            architecture (str): AdaFace模型架构 ('ir_18', 'ir_34', 'ir_50', 'ir_101')
            similarity_threshold (float): 人脸识别相似度阈值，范围[0, 1]。
            detection_threshold (float): 人脸检测置信度阈值，范围[0, 1]。
            db_path (str): FAISS数据库存储路径
        """
        print("[FaceRecognizer] 初始化SCRFD+AdaFace人脸识别系统...")
        
        # 保存配置参数
        self.face_gallery_path = face_gallery_path
        self.similarity_threshold = similarity_threshold
        self.detection_threshold = detection_threshold
        self.db_path = db_path
        scrfd_model_path = resolve_model_path('weights/det_10_640.om', scrfd_model_path)
        adaface_model_path = resolve_model_path('weights/adaface_ir50_ms1mv2_b1.om', adaface_model_path)
        
        # 初始化SCRFD人脸检测器
        try:
            self.detector = SCRFD(
                model_path=scrfd_model_path,
                input_size=(640, 640),
                conf_thres=detection_threshold
            )
            print("[FaceRecognizer] SCRFD检测器初始化成功")
        except Exception as e:
            print(f"[FaceRecognizer] 错误：SCRFD检测器初始化失败: {e}")
            raise

        # 初始化AdaFace人脸识别器
        try:
            self.recognizer = AdaFace(
                model_path=adaface_model_path,
                architecture=architecture
            )
            print("[FaceRecognizer] AdaFace识别器初始化成功")
        except Exception as e:
            print(f"[FaceRecognizer] 错误：AdaFace识别器初始化失败: {e}")
            raise

        # 初始化FAISS人脸数据库
        self.face_db = FaceDatabase(db_path=db_path)
        
        # 从人脸库目录加载所有已知人脸的特征向量
        self._load_face_gallery()
        
        print("[FaceRecognizer] 人脸识别系统初始化完成")

    def _load_face_gallery(self):
        """
        从人脸库目录加载所有已知人脸的特征向量。
        
        目录结构要求：
        faceImage/
        ├── person_01/
        │   ├── face1.jpg
        │   └── ...
        ├── person_02/
        │   └── face1.jpg
        └── ...
        """
        if not os.path.isdir(self.face_gallery_path):
            print(f"[FaceRecognizer] 警告: 人脸库文件夹 '{self.face_gallery_path}' 不存在")
            return

        print(f"[FaceRecognizer] 从人脸库加载已知人脸: {self.face_gallery_path}")
        
        loaded_count = 0
        
        for person_id in os.listdir(self.face_gallery_path):
            person_dir = os.path.join(self.face_gallery_path, person_id)
            
            if not os.path.isdir(person_dir):
                continue

            for img_file in os.listdir(person_dir):
                if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue

                img_path = os.path.join(person_dir, img_file)
                try:
                    image = cv2.imread(img_path)
                    if image is None:
                        print(f"[FaceRecognizer] 警告: 无法读取图像 {img_path}")
                        continue

                    # 使用SCRFD检测人脸
                    bboxes, kpss = self.detector.detect(image, max_num=1)
                    
                    if len(kpss) == 0:
                        print(f"[FaceRecognizer] 警告: 在 {img_path} 中未检测到人脸")
                        continue

                    # 使用AdaFace提取人脸特征向量
                    embedding = self.recognizer.get_embedding(image, kpss[0])
                    
                    # 将特征向量添加到FAISS数据库
                    self.face_db.add_face(embedding, person_id)
                    loaded_count += 1
                    
                except Exception as e:
                    print(f"[FaceRecognizer] 错误: 处理 {img_path} 时出错: {e}")
                    continue

        print(f"[FaceRecognizer] 成功加载 {loaded_count} 个人脸特征")

    def detect_and_recognize(self, face_detector, frame, person_crop, crop_x1, crop_y1):
        """
        在一个人的裁剪图像中检测面部，并尝试识别它们。
        
        参数:
            face_detector: 已弃用，保留用于接口兼容
            frame: 完整的视频帧 (未使用，保留用于接口兼容)
            person_crop (np.ndarray): 裁剪出的人体图像
            crop_x1 (int): 人体裁剪区域在原图中的左上角x坐标
            crop_y1 (int): 人体裁剪区域在原图中的左上角y坐标

        返回:
            tuple: (face_boxes_global, face_ids)
        """
        if person_crop.size == 0:
            return [], []

        try:
            # 使用SCRFD检测人脸
            bboxes, kpss = self.detector.detect(person_crop, max_num=0)
            
            if len(bboxes) == 0:
                return [], []

            face_boxes_global = []
            face_ids = []

            for bbox, kps in zip(bboxes, kpss):
                try:
                    # 使用AdaFace提取人脸特征向量
                    embedding = self.recognizer.get_embedding(person_crop, kps)
                    
                    # 在FAISS数据库中搜索最匹配的已知人脸
                    person_id, similarity = self.face_db.search(embedding, self.similarity_threshold)
                    face_ids.append(person_id)
                    
                    # 将相对坐标转换为全局坐标
                    x1, y1, x2, y2, conf = bbox.astype(np.int32)
                    global_box = [
                        x1 + crop_x1,
                        y1 + crop_y1,
                        x2 + crop_x1,
                        y2 + crop_y1
                    ]
                    face_boxes_global.append(global_box)
                    
                except Exception as e:
                    print(f"[FaceRecognizer] 处理人脸时出错: {e}")
                    face_ids.append('Unknown')
                    
                    if 'bbox' in locals():
                        x1, y1, x2, y2, conf = bbox.astype(np.int32)
                        global_box = [
                            x1 + crop_x1,
                            y1 + crop_y1,
                            x2 + crop_x1,
                            y2 + crop_y1
                        ]
                        face_boxes_global.append(global_box)
                    
            return face_boxes_global, face_ids
            
        except Exception as e:
            print(f"[FaceRecognizer] 检测和识别过程中出错: {e}")
            return [], []

    def reload_gallery(self):
        """重新加载人脸库"""
        self.face_db = FaceDatabase(db_path=self.db_path)
        self._load_face_gallery()
        print("[FaceRecognizer] 人脸库已重新加载")
