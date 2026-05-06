import sys
import os
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import numpy as np
import cv2

# 动态添加 transreid_pytorch 到系统路径，确保能导入其内部模块
# # sys.path.append(os.path.join(os.path.dirname(__file__), 'transreid_pytorch'))

from models.transreid_pytorch.config import cfg
from models.transreid_pytorch.model import make_model
from models.ascend_backend import AscendInferSession, is_om_path, l2_normalize

class PersonViTFeatureExtractor:
    def __init__(self, model_path, config_file, device='cuda'):
        """
        初始化 PersonViT 特征提取器s
        :param model_path: 下载的 .pth 权重文件路径
        :param config_file: 对应的 .yml 配置文件路径 (如 vit_base.yml)
        :param device: 'cuda' 或 'cpu'
        """
        self.device = device
        self.use_om = is_om_path(model_path)
        
        # 解冻
        cfg.defrost()

        cfg.merge_from_file(config_file)
        self.input_size = tuple(cfg.INPUT.SIZE_TEST)
        self.pixel_mean = np.asarray(cfg.INPUT.PIXEL_MEAN, dtype=np.float32).reshape(3, 1, 1)
        self.pixel_std = np.asarray(cfg.INPUT.PIXEL_STD, dtype=np.float32).reshape(3, 1, 1)

        if not self.use_om:
            # [新增] 关键修复：将预训练路径强制指向传入的 model_path
            # 这解决了两个问题：
            # 1. 避免了 PRETRAIN_PATH 为空导致的 FileNotFoundError
            # 2. 避免了加载作者硬编码的 Linux 路径
            cfg.MODEL.PRETRAIN_PATH = model_path

        cfg.freeze()

        if self.use_om:
            print(f"[PersonViT] Loading Ascend OM model from {model_path}")
            self.session = AscendInferSession(model_path)
            self.model = None
            self.transforms = None
            return
        
        # 2. 构建模型
        # 注意：num_class 等参数主要影响分类头，推理时只用骨干网，
        # 但为了加载权重不报错，最好保持默认 (Market1501通常是751类)
        self.model = make_model(cfg, num_class=751, camera_num=0, view_num=0)
        
        
        # 3. 加载权重
        print(f"[PersonViT] Loading weights from {model_path}")
        self.model.load_param(model_path)
        self.model.to(device)
        self.model.eval()
        
        # 4. 定义预处理 (与训练时保持一致)
        self.transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TEST), # 读取配置中的尺寸，通常是 [256, 128]
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
        ])

    def _preprocess_om_image(self, img):
        if isinstance(img, np.ndarray):
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(img, Image.Image):
            rgb = np.asarray(img.convert("RGB"))
        else:
            raise ValueError(f"Input must be numpy array or PIL Image, got {type(img)}")

        height, width = self.input_size
        rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)
        chw = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        return (chw - self.pixel_mean) / self.pixel_std

    def _infer_om_batch(self, images):
        if not images:
            return torch.empty(0, 768)

        features = []
        for img in images:
            blob = self._preprocess_om_image(img)[None].astype(np.float32, copy=False)
            outputs = self.session.infer(blob)
            feature = np.asarray(outputs[0], dtype=np.float32).reshape(1, -1)
            features.append(feature[0])

        features = l2_normalize(np.stack(features, axis=0).astype(np.float32), axis=1)
        return torch.from_numpy(features)

    def __call__(self, input_image):
        """
        执行推理，接口模仿 torchreid
        :param input_image: 单个图片(numpy/PIL) 或 图片列表(list of numpy/PIL)
        :return: 归一化的特征向量 Tensor, shape [batch_size, feature_dim]
        """
        if self.use_om:
            images = input_image if isinstance(input_image, list) else [input_image]
            return self._infer_om_batch(images)

        # [新增] 支持批量输入 (List of images)
        if isinstance(input_image, list):
            # 递归调用处理每一张图，然后拼接
            # 注意：为了效率，最好是构建一个 batch tensor 一次性推理，而不是循环调用
            # 下面是构建 batch tensor 的高效写法：
            
            batch_tensors = []
            for img in input_image:
                # 1. 格式转换
                if isinstance(img, np.ndarray):
                    image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                elif isinstance(img, Image.Image):
                    image = img
                else:
                    raise ValueError(f"List item must be numpy array or PIL Image, got {type(img)}")
                
                # 2. 预处理 (得到 [C, H, W])
                batch_tensors.append(self.transforms(image))
            
            # 堆叠成 [Batch, C, H, W]
            if not batch_tensors:
                return torch.empty(0, 768).to(self.device) # 返回空特征
                
            img_tensor = torch.stack(batch_tensors).to(self.device)

        # [原有逻辑] 单张图片处理
        else:
            # 1. 格式转换
            if isinstance(input_image, np.ndarray):
                image = Image.fromarray(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
            elif isinstance(input_image, Image.Image):
                image = input_image
            else:
                raise ValueError(f"Input must be numpy array or PIL Image, got {type(input_image)}")

            # 2. 预处理 -> [1, C, H, W]
            img_tensor = self.transforms(image).unsqueeze(0).to(self.device)
        
        # 3. 推理 (Batch Inference)
        with torch.no_grad():
            # PersonViT 可能需要 cam_label 和 view_label
            # 构造与 batch size 匹配的 dummy labels
            batch_size = img_tensor.size(0)
            dummy_cam = torch.zeros(batch_size, dtype=torch.int64).to(self.device)
            dummy_view = torch.zeros(batch_size, dtype=torch.int64).to(self.device)
            
            # 提取特征
            features = self.model(img_tensor, cam_label=dummy_cam, view_label=dummy_view)
            
            # 4. 归一化 (ReID 中非常重要，cosine 相似度需要)
            features = F.normalize(features, p=2, dim=1)
            
        return features
