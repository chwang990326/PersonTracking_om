import os
import cv2
import numpy as np
import torch
import threading
from datetime import datetime
from collections import deque
import shutil
import time
from models.personvit_adapter import PersonViTFeatureExtractor
from models.reid_state import SharedIdentityStore
from models.transreid_pytorch.utils.reranking import re_ranking
from models.ascend_backend import resolve_model_path

auto_save = True

class PersonReidentifier:
    """
    基于预定义的身份库（Gallery）进行人物重识别。
    """

    def __init__(self, identity_folder='identity', 
                 model_path='./config/transformer_120.pth',
                 # config_file='models/transreid_pytorch/configs/msmt17/vit_base_ics_384.yml',
                 config_file='./models/transreid_pytorch/configs/market/vit_base.yml',
                 similarity_threshold=0.9, 
                 known_similarity_threshold=None,
                 unknown_similarity_threshold=None,
                 device='cuda', max_age=30, iou_threshold=0.3, feature_smooth_alpha=0.8,
                 pose_estimator=None, verify_interval=2,
                 feature_extractor=None,
                 id_generator=None,
                 shared_identity_store=None,
                 shared_unknown_store=None): 
        """
        初始化ReID系统。

        参数:
            identity_folder (str): 存放已知人物图片的根文件夹。
            model_name (str): 使用的torchreid模型名称。
            similarity_threshold (float): 兼容旧调用的统一相似度阈值别名。
            known_similarity_threshold (float): 已知人员 ReID 匹配阈值。
            unknown_similarity_threshold (float): 未知人员 ReID 匹配阈值。
            device (str): 使用的计算设备 ('cuda' 或 'cpu')。
            max_age (int): 追踪对象消失后保持的最大帧数。
            iou_threshold (float): IOU匹配的阈值。
            feature_smooth_alpha (float): 特征平滑系数 (0-1)，越大越依赖历史特征。
            pose_estimator: YOLO姿态估计器实例，用于验证图像质量
            verify_interval (int): 身份校验间隔帧数，用于纠正追踪过程中的身份错误
            id_generator (callable): 全局唯一的ID生成函数，返回下一个可用的临时ID
        """
        print("[ReID] 初始化系统...")
        model_path = resolve_model_path('weights/transformer_120.om', model_path)
        self.identity_folder = identity_folder
        if known_similarity_threshold is None:
            known_similarity_threshold = similarity_threshold
        if unknown_similarity_threshold is None:
            unknown_similarity_threshold = similarity_threshold
        self.known_similarity_threshold = known_similarity_threshold
        self.unknown_similarity_threshold = unknown_similarity_threshold
        self.similarity_threshold = self.known_similarity_threshold
        self.device = self._get_device(device)
        
        # 时序追踪参数
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.feature_smooth_alpha = feature_smooth_alpha
        self.verify_interval = verify_interval
        
        # 自动采集的质量控制参数
        self.MIN_CROP_HEIGHT = 100         # 最小高度
        self.EDGE_MARGIN = 5              # 距离边缘至少15像素
        self.MIN_TRACK_AGE = 3            # 追踪稳定多少帧后才允许采集(纠错机制)
        
        # [修改] 追踪状态管理
        # 映射表: { bytetrack_id: {'person_id': real_id, 'feature': smooth_feat, 'frames_since_verify': 0} }
        self.track_mapper = {} 
        
        self.next_track_id = 0 # 保留用于生成临时ID
        
        # --- 优化配置 ---
        self.MAX_GALLERY_IMAGES = 12       # 稍微增加上限，允许更多角度
        self.DIVERSITY_THRESHOLD = 0.85   # 已知身份目录内的多样性阈值：相似度高于此值则认为重复，不保存
        self.TEMP_ID_DEDUP_THRESHOLD = 0.95 # 临时ID跨库去重阈值：相似度高于此值则跳过落盘
        self.SAFE_UPDATE_IOU_THRESHOLD = 0.13 # 安全更新阈值：如果与他人重叠超过此值，不更新库
        # ----------------
        
        # --- 锚点防漂移配置 ---
        self.MAX_ANCHOR_IMAGES = 2             # [需求1] 每个已知ID最多保留的锚点图数量
        self.ANCHOR_DIVERSITY_THRESHOLD = 0.95 # [需求1] 锚点去重：与已有锚点相似度>此值则拒绝
        self.ANCHOR_GATE_THRESHOLD = 0.75      # [需求2] 普通特征准入：与锚点相似度需>=此值
        self.MAX_UNKNOWN_REID_IMAGES = 3
        self.anchor_gallery = {}               # 锚点特征库 { person_id: np.ndarray(N, dim) }
        self.gallery_files = {}
        self.anchor_files = {}
        self.gallery_lock = threading.RLock()
        self.shared_identity_store = shared_identity_store
        self.shared_unknown_store = shared_unknown_store
        
        # 定义临时ID范围
        self.TEMP_ID_START = 0
        self.TEMP_ID_END = 99 # 扩大一点范围
        
        # 启动时清理过期档案
        self._cleanup_temp_identities(max_days=1) 

        # [修改] ID生成策略：优先使用外部传入的生成器（全局唯一），否则使用本地逻辑
        self.id_generator = id_generator

        if self.id_generator is None:
            # 初始化自动ID生成器 (本地逻辑，仅用于单机单摄场景)
            max_existing_id = self.TEMP_ID_START - 1
            if os.path.exists(self.identity_folder):
                for d in os.listdir(self.identity_folder):
                    if d.isdigit():
                        did = int(d)
                        if self.TEMP_ID_START <= did <= self.TEMP_ID_END:
                            max_existing_id = max(max_existing_id, did)
            
            self.next_auto_id = max_existing_id + 1
            if self.next_auto_id > self.TEMP_ID_END:
                self.next_auto_id = self.TEMP_ID_START
            print(f"[ReID] (本地模式) 临时ID范围: {self.TEMP_ID_START}-{self.TEMP_ID_END}, 当前起始: {self.next_auto_id}")
        else:
            print(f"[ReID] (全局模式) 使用外部共享的ID生成器")

        # 1. 初始化深度特征提取器
        if feature_extractor is not None:
            # [新增] 使用传入的共享模型，避免重复初始化导致的 CfgNode immutable 错误
            print(f"[ReID] 使用共享的特征提取模型")
            self.feature_extractor = feature_extractor
        else:
            print(f"[ReID] 初始化 PersonViT 模型...")
            print(f"       Config: {config_file}")
            print(f"       Weights: {model_path}")
                
            self.feature_extractor = PersonViTFeatureExtractor(
                    model_path=model_path,
                    config_file=config_file,
                    device=self.device
                )
            print("[ReID] 特征提取模型加载完成。")

        # 2. 构建特征画廊
        if self.shared_identity_store is not None:
            self.gallery_lock = self.shared_identity_store.lock
            self.feature_gallery = self.shared_identity_store.feature_gallery
            self.anchor_gallery = self.shared_identity_store.anchor_gallery
            self.gallery_files = self.shared_identity_store.gallery_files
            self.anchor_files = self.shared_identity_store.anchor_files
        else:
            self.feature_gallery = self._build_feature_gallery()

        # [新增] 保存姿态估计器
        self.pose_estimator = pose_estimator
        
        # [新增] 定义上半身关键点索引 (COCO格式)
        # 0: 鼻子, 5: 左肩, 6: 右肩, 11: 左髋, 12: 右髋
        self.UPPER_BODY_KEYPOINTS = {
            'nose': 0,
            'left_shoulder': 5,
            'right_shoulder': 6,
            'left_hip': 11,
            'right_hip': 12
        }
        self.KEYPOINT_CONFIDENCE_THRESHOLD = 0.5  # 关键点置信度阈值

    def _get_device(self, requested_device):
        """检查并返回可用的设备。"""
        if requested_device == 'cuda' and not torch.cuda.is_available():
            # print("[ReID] 警告: 未检测到CUDA，自动切换到CPU模式。")
            return 'cpu'
        # print(f"[ReID] 使用设备: {requested_device.upper()}")
        return requested_device

    @staticmethod
    def _normalize_identity_value(person_id):
        if isinstance(person_id, str):
            return person_id.strip()
        return person_id

    def _is_temporary_identity(self, person_id):
        person_id = self._normalize_identity_value(person_id)
        return isinstance(person_id, int) or (
            isinstance(person_id, str) and person_id.isdigit()
        )

    def _is_known_identity(self, person_id):
        person_id = self._normalize_identity_value(person_id)
        if person_id in (None, '', -1):
            return False
        return not self._is_temporary_identity(person_id)

    def _is_auto_save_enabled(self):
        return bool(auto_save)

    def _allocate_temporary_id(self):
        """
        分配一个全局唯一的临时ID，优先使用外部生成器，如果没有则使用本地逻辑。
        """
        if self.id_generator:
            return self.id_generator()

        temp_id = self.next_auto_id
        self.next_auto_id += 1
        if hasattr(self, 'TEMP_ID_END') and self.next_auto_id > self.TEMP_ID_END:
            self.next_auto_id = self.TEMP_ID_START
        return temp_id

    def refresh_track_state(self, active_track_ids):
        """
        刷新追踪状态，移除长时间未出现的轨迹。
        """
        active_track_set = set(active_track_ids or [])
        keys_to_remove = []

        for track_id, track_info in self.track_mapper.items():
            if track_id in active_track_set:
                track_info['missed_frames'] = 0
                continue

            missed_frames = track_info.get('missed_frames', 0) + 1
            track_info['missed_frames'] = missed_frames
            if missed_frames > self.max_age:
                keys_to_remove.append(track_id)

        for track_id in keys_to_remove:
            self.track_mapper.pop(track_id, None)

    def _build_expanded_anchor_crop(self, frame, crop_coords):
        if frame is None or crop_coords is None:
            return None

        crop_x1, crop_y1, crop_x2, crop_y2 = map(int, crop_coords)
        crop_w = max(0, crop_x2 - crop_x1)
        crop_h = max(0, crop_y2 - crop_y1)
        if crop_w == 0 or crop_h == 0:
            return None

        pad_x = max(1, int(round(crop_w * 0.1)))
        pad_y = max(1, int(round(crop_h * 0.1)))
        x1 = max(0, crop_x1 - pad_x)
        y1 = max(0, crop_y1 - pad_y)
        x2 = min(frame.shape[1], crop_x2 + pad_x)
        y2 = min(frame.shape[0], crop_y2 + pad_y)
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2].copy()

    def _generate_prefixed_filename(self, target_dir, prefix):
        base_name = datetime.now().strftime(f"{prefix}_%Y%m%d_%H%M%S")
        filename = f"{base_name}.jpg"
        save_path = os.path.join(target_dir, filename)
        suffix = 1
        while os.path.exists(save_path):
            filename = f"{base_name}_{suffix}.jpg"
            save_path = os.path.join(target_dir, filename)
            suffix += 1
        return filename

    def _generate_anchor_filename(self, target_dir):
        return self._generate_prefixed_filename(target_dir, 'anchor')

    def _get_gallery_features(self, person_id):
        with self.gallery_lock:
            return self.feature_gallery.get(person_id)

    def _get_anchor_features(self, person_id):
        with self.gallery_lock:
            return self.anchor_gallery.get(person_id)

    def _has_gallery_identity(self, person_id):
        with self.gallery_lock:
            return self._is_known_identity(person_id) and person_id in self.feature_gallery

    def _gallery_size(self):
        with self.gallery_lock:
            return len(self.feature_gallery)

    def _gallery_items_snapshot(self):
        with self.gallery_lock:
            return list(self.feature_gallery.items())

    def _get_unknown_gallery_features(self, person_id):
        """
        获取未知身份库中指定ID的特征向量。
        """
        if self.shared_unknown_store is None:
            return None
        with self.shared_unknown_store.lock:
            return self.shared_unknown_store.reid_feature_gallery.get(person_id)

    def _unknown_gallery_items_snapshot(self):
        """
        获取未知身份库的快照列表，格式为 [(person_id, features), ...]。
        """
        if self.shared_unknown_store is None:
            return []
        with self.shared_unknown_store.lock:
            return list(self.shared_unknown_store.reid_feature_gallery.items())

    def _append_unknown_gallery_file_entry(self, person_id, filename, feature_vector):
        """
        将一个新的特征向量添加到未知身份库中指定ID的条目下。
        如果该ID不存在，则创建一个新的条目；
        如果存在，则将特征向量追加到现有的特征矩阵中，并记录文件名。
        """
        if self.shared_unknown_store is None:
            return
        current_feat = feature_vector.reshape(1, -1)
        with self.shared_unknown_store.lock:
            if person_id not in self.shared_unknown_store.reid_feature_gallery:
                self.shared_unknown_store.reid_feature_gallery[person_id] = current_feat
                self.shared_unknown_store.reid_gallery_files[person_id] = [filename]
            else:
                self.shared_unknown_store.reid_feature_gallery[person_id] = np.vstack((
                    self.shared_unknown_store.reid_feature_gallery[person_id], current_feat
                ))
                self.shared_unknown_store.reid_gallery_files.setdefault(person_id, []).append(filename)
            self.shared_unknown_store.touch_entity(person_id)

    def _remove_unknown_gallery_file_entry(self, person_id, filename):
        if self.shared_unknown_store is None:
            return False
        with self.shared_unknown_store.lock:
            return self._remove_feature_entry(
                self.shared_unknown_store.reid_feature_gallery,
                self.shared_unknown_store.reid_gallery_files,
                person_id,
                filename,
            )

    def _find_best_unknown_match(self, feature):
        best_match_id = None
        best_similarity = 0.0

        for person_id, gallery_features in self._unknown_gallery_items_snapshot():
            similarity = self._compute_similarity(feature, gallery_features)
            if similarity >= self.unknown_similarity_threshold and similarity > best_similarity:
                best_match_id = person_id
                best_similarity = similarity

        return best_match_id, best_similarity


    def _append_gallery_file_entry(self, person_id, filename, feature_vector, is_anchor=False):
        current_feat = feature_vector.reshape(1, -1)
        with self.gallery_lock:
            if person_id not in self.feature_gallery:
                self.feature_gallery[person_id] = current_feat
                self.gallery_files[person_id] = [filename]
            else:
                self.feature_gallery[person_id] = np.vstack((self.feature_gallery[person_id], current_feat))
                self.gallery_files.setdefault(person_id, []).append(filename)

            if is_anchor:
                if person_id not in self.anchor_gallery:
                    self.anchor_gallery[person_id] = current_feat
                    self.anchor_files[person_id] = [filename]
                else:
                    self.anchor_gallery[person_id] = np.vstack((self.anchor_gallery[person_id], current_feat))
                    self.anchor_files.setdefault(person_id, []).append(filename)

    @staticmethod
    def _drop_feature_row(features, row_index):
        if features is None or len(features) == 0:
            return None
        if len(features) == 1:
            return None
        return np.delete(features, row_index, axis=0)

    def _remove_feature_entry(self, store, file_store, person_id, filename):
        files = list(file_store.get(person_id, []))
        if filename not in files:
            return False

        row_index = files.index(filename)
        features = store.get(person_id)
        if features is None or row_index >= len(features):
            return False

        updated_features = self._drop_feature_row(features, row_index)
        updated_files = files[:row_index] + files[row_index + 1:]
        if updated_features is None or not updated_files:
            store.pop(person_id, None)
            file_store.pop(person_id, None)
        else:
            store[person_id] = updated_features
            file_store[person_id] = updated_files
        return True

    def _remove_gallery_file_entry(self, person_id, filename):
        with self.gallery_lock:
            removed = self._remove_feature_entry(self.feature_gallery, self.gallery_files, person_id, filename)
            if filename.startswith('anchor_'):
                self._remove_feature_entry(self.anchor_gallery, self.anchor_files, person_id, filename)
            return removed

    def _merge_store_entries(self, store, file_store, source_id, target_id):
        if source_id not in store:
            return

        source_features = store[source_id]
        source_files = list(file_store.get(source_id, []))
        if target_id not in store:
            store[target_id] = source_features
            file_store[target_id] = source_files
        else:
            store[target_id] = np.vstack((store[target_id], source_features))
            file_store.setdefault(target_id, []).extend(source_files)

        del store[source_id]
        file_store.pop(source_id, None)

    def _build_feature_gallery(self):
        """
        遍历identity文件夹，为每个已知人物构建特征向量库。
        这是整个系统的“记忆库”。
        """
        entries = SharedIdentityStore.scan_identity_entries(self.identity_folder)
        gallery, anchor_gallery, gallery_files, anchor_files = SharedIdentityStore.load_gallery_from_entries(
            entries,
            self.feature_extractor,
            verbose=True,
        )

        with self.gallery_lock:
            self.anchor_gallery.clear()
            self.anchor_gallery.update(anchor_gallery)
            self.gallery_files.clear()
            self.gallery_files.update(gallery_files)
            self.anchor_files.clear()
            self.anchor_files.update(anchor_files)

        if not gallery:
            print("[ReID] 警告: 特征画廊为空，将无法识别任何人。请检查'identity'文件夹结构。")
        else:
            print(f"[ReID] 特征画廊构建完成，共包含 {len(gallery)} 个已知身份。")
        return gallery

    def _compute_similarity(self, query_feature, gallery_features):
        """
        计算一个查询特征与一个身份的所有画廊特征之间的相似度。
        采用标准的余弦相似度，解决小样本下分数固定的问题。
        """
        # 1. 统一转为 Numpy
        if isinstance(query_feature, torch.Tensor):
            query_feature = query_feature.cpu().numpy()
        if isinstance(gallery_features, torch.Tensor):
            gallery_features = gallery_features.cpu().numpy()
            
        # 2. 维度处理
        if len(query_feature.shape) == 1:
            query_feature = query_feature.reshape(1, -1) # (1, dim)
        
        # 3. 特征归一化 (L2 Normalization)
        # ReID 特征通常在超球面上比较，余弦相似度等价于归一化后的欧氏距离
        q_norm = np.linalg.norm(query_feature, axis=1, keepdims=True)
        g_norm = np.linalg.norm(gallery_features, axis=1, keepdims=True)
        
        # 加上极小值防止除零
        query_feature = query_feature / (q_norm + 1e-12)
        gallery_features = gallery_features / (g_norm + 1e-12)
        
        # 4. 计算余弦相似度 (Cosine Similarity)
        # 结果范围 [-1, 1]
        sim_matrix = np.dot(query_feature, gallery_features.T)
        
        # 5. 取最大相似度作为该身份的得分
        # (即：该人与库中这个ID下的某张照片最像的程度)
        max_sim = np.max(sim_matrix)
        
        # 截断到 [0, 1] 范围
        return float(np.clip(max_sim, 0, 1))

    def _cosine_similarity(self, feat1, feat2):
        """计算两个特征向量的余弦相似度。"""
        norm1 = np.linalg.norm(feat1)
        norm2 = np.linalg.norm(feat2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return np.dot(feat1, feat2) / (norm1 * norm2)

    #  IOU 计算函数
    def _compute_iou(self, box1, box2):
        """
        计算两个矩形框的IOU (Intersection over Union)。
        box: [x1, y1, x2, y2]
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union = area1 + area2 - intersection
        if union <= 0:
            return 0.0
        return intersection / union

    # --- 新增辅助函数：检查时间戳重叠 ---
    def _is_timestamp_overlap(self, box, frame_shape):
        """
        检查检测框是否与右上角时间戳区域重叠。
        时间戳区域：右上角 25% 宽度，8% 高度。
        """
        if frame_shape is None:
            return False
            
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = box
        
        # 时间戳区域定义 (右上角)
        ts_x_min = int(w * 0.75)
        ts_y_max = int(h * 0.08)
        
        # 检查是否有重叠
        # 框的右边 > 时间戳左边 AND 框的顶边 < 时间戳底边
        if x2 > ts_x_min and y1 < ts_y_max:
            return True
        return False

    # --- 新增辅助函数：智能保存图片到Gallery ---
    # [修改] 增加 force 参数
    def _smart_save_to_gallery(self, person_id, image, feature_vector, force=False):
        """
        智能保存：
        force (bool): 是否强制保存（用于人脸识别确认后的高质量图）
        """
        if not self._is_auto_save_enabled():
            return False

        target_dir = os.path.join(self.identity_folder, str(person_id))
        
        # 判断是否为数字ID (临时ID)
        is_temp_id = str(person_id).isdigit()
        
        # --- 策略分支 ---
        
        # 1. 如果是临时ID (数字)：保持原有逻辑 (只存1张，不更新)
        if is_temp_id:
            if not force:
                if not self._check_temp_identity_dedup(feature_vector, person_id):
                    return False
                if os.path.exists(target_dir):
                    existing_files = [f for f in os.listdir(target_dir) if f.lower().endswith(('.jpg', '.png'))]
                    if len(existing_files) > 0:
                        return False
        
        # 2. 如果是已知ID (字符串)：执行换装更新策略
        else:
            if not force:
                # [需求2: 普通特征准入] 存在锚点时，必须通过锚点门控防止漂移
                if not self._check_anchor_gate(person_id, feature_vector):
                    return False
                
                # 原有的多样性检查（避免重复保存相似特征）
                if not self._check_feature_diversity(person_id, feature_vector):
                    return False
            
            # [需求3: 双轨制淘汰] 检查容量，仅淘汰普通特征
            self._manage_gallery_capacity(person_id)

        # --- 公共检查 ---
        
        # 关键点质量检查 (上半身完整性)
        if force:
            # [优化] force 模式仍做最低限度质量检查：尺寸和模糊度
            h, w = image.shape[:2]
            if h < 50 or w < 30:
                print(f"[ReID] force 保存跳过: 图像尺寸过小 ({w}x{h})")
                return False
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            if blur_score < 30:
                print(f"[ReID] force 保存跳过: 图像过于模糊 (blur_score={blur_score:.1f})")
                return False
        else:
            # 非 force 模式：要求上半身完整
            if not self._check_upper_body_quality(image, require_face=True):
                return False

        # --- 执行保存 ---
        try:
            os.makedirs(target_dir, exist_ok=True)
            
            filename = f"auto_{int(time.time() * 1000)}.jpg"
            save_path = os.path.join(target_dir, filename)
            cv2.imwrite(save_path, image)
            
            # 更新内存库
            self._append_gallery_file_entry(person_id, filename, feature_vector, is_anchor=False)
            
            current_count = len([f for f in os.listdir(target_dir) if f.lower().endswith(('.jpg', '.png'))])
            
            if not is_temp_id:
                print(f"[ReID] ID {person_id} 特征库已更新 (换装/新角度), 当前样本数: {current_count}")
            else:
                print(f"[ReID] ID {person_id} 补充新特征 (上半身完整, 当前样本数: {current_count})")
                
            return True
        except Exception as e:
            print(f"[ReID] 保存失败: {e}")
            return False

    # --- 新增辅助函数：合并身份 ---
    def _merge_identities(self, temp_id, real_id):
        """
        将临时ID的数据合并到真实ID中。
        [优化] 当人脸识别成功时，我们认为当前帧质量优于之前的临时抓拍。
        因此，策略改为：清理旧的临时ID数据，为保存当前高质量帧腾出空间。
        """
        print(f"[ReID] 正在合并身份: {temp_id} -> {real_id}")
        
        src_dir = os.path.join(self.identity_folder, str(temp_id))
        
        # 1. 直接删除旧的临时文件夹
        if os.path.exists(src_dir):
            try:
                shutil.rmtree(src_dir)
                print(f"[ReID] 已清理临时ID {temp_id} 的旧数据 (将被当前高质量人脸帧替代)")
            except Exception as e:
                print(f"[ReID] 删除临时文件夹失败: {e}")

        # 2. 合并内存中的特征库
        with self.gallery_lock:
            self._merge_store_entries(self.feature_gallery, self.gallery_files, temp_id, real_id)
            self._merge_store_entries(self.anchor_gallery, self.anchor_files, temp_id, real_id)
            
        # [修复] 移除 moved_count，改为简单的完成提示
        print(f"[ReID] 身份合并操作完成: {temp_id} -> {real_id}")

    def _check_overlap(self, current_idx, all_boxes):
        """
        检查当前框是否与其他框有显著重叠。
        如果重叠，说明该图片可能包含其他人，不适合作为Gallery样本。
        """
        current_box = all_boxes[current_idx]
        for i, other_box in enumerate(all_boxes):
            if i == current_idx:
                continue
            
            iou = self._compute_iou(current_box, other_box)
            if iou > self.SAFE_UPDATE_IOU_THRESHOLD:
                # [修改] 打印具体的IOU值方便调试
                print(f"[Debug] 检测到重叠 (IOU={iou:.2f} > {self.SAFE_UPDATE_IOU_THRESHOLD:.2f})，跳过Gallery更新。")
                return True # 存在重叠
        return False

    def get_switch_from(self, track_id):
        """返回当前请求周期内该轨迹发生切换时的旧 ID。"""
        track_info = self.track_mapper.get(track_id)
        if track_info is None:
            return None

        switch_from = track_info.get('switch_from')
        if switch_from is None:
            return None
        return str(switch_from)

    def bind_track_identity(self, track_id, new_person_id):
        """
        将一个新的身份绑定到指定的轨迹ID上，并返回旧的身份ID（如果有的话）。
        """
        if track_id not in self.track_mapper:
            return None

        new_person_id = self._normalize_identity_value(new_person_id)
        old_person_id = self.track_mapper[track_id]['person_id']
        if str(old_person_id) == str(new_person_id):
            return old_person_id

        self.track_mapper[track_id]['switch_from'] = old_person_id
        self.track_mapper[track_id]['person_id'] = new_person_id
        self.track_mapper[track_id]['frames_since_verify'] = 0
        return old_person_id

    def identify(self, frame, boxes, track_ids=None):
        """
        使用 YOLO ByteTrack 的 ID 进行身份关联。
        """
        if len(boxes) == 0:
            return [], []

        # 1. 裁剪图像
        person_crops = []
        valid_indices = [] # 记录有效的索引，防止crop失败导致对齐错误
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            crop = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
            if crop.size > 0:
                person_crops.append(crop)
                valid_indices.append(i)

        if not person_crops:
            return [-1] * len(boxes), [0.0] * len(boxes)

        # 2. 提取特征
        query_features = self.feature_extractor(person_crops).cpu().numpy()
        
        assigned_ids = [-1] * len(boxes)
        assigned_scores = [0.0] * len(boxes)
        
        # 如果没有 track_ids (比如第一帧或者追踪失败)，退化为纯特征匹配
        if track_ids is None or len(track_ids) == 0:
            # 纯特征匹配逻辑，或者直接返回
            for feat_idx, original_idx in enumerate(valid_indices):
                feature = query_features[feat_idx]
                best_match_id, best_similarity = self._find_best_match_in_gallery(feature)
                
                if best_match_id is not None and best_similarity > self.known_similarity_threshold:
                    assigned_ids[original_idx] = best_match_id
                    assigned_scores[original_idx] = best_similarity
                else:
                    assigned_ids[original_idx] = -1
                    assigned_scores[original_idx] = 0.0

        # 3. 遍历每个检测框进行身份分配
        # 注意：query_features 的索引对应 valid_indices
        for feat_idx, original_idx in enumerate(valid_indices):
            track_id = track_ids[original_idx]
            feature = query_features[feat_idx]
            
            # 归一化特征
            feature = feature / (np.linalg.norm(feature) + 1e-12)

            current_person_id = None
            match_score = 0.0

            # --- 情况 A: 这个 Track ID 已经在映射表中 ---
            if track_id in self.track_mapper:
                track_info = self.track_mapper[track_id]
                # switch_from 是单次响应有效的瞬时字段，每次处理该轨迹时先清空
                track_info['switch_from'] = None
                
                # [修复] 必须在这里累加 age，否则它永远是 1
                if 'age' not in track_info:
                    track_info['age'] = 1
                else:
                    track_info['age'] += 1

                current_person_id = track_info['person_id']
                
                # 更新平滑特征
                old_feature = track_info['feature']
                new_feature = self.feature_smooth_alpha * old_feature + (1 - self.feature_smooth_alpha) * feature
                self.track_mapper[track_id]['feature'] = new_feature / np.linalg.norm(new_feature)
                
                # [修改] 周期性校验 (纠错机制)
                self.track_mapper[track_id]['frames_since_verify'] += 1
                if self.track_mapper[track_id]['frames_since_verify'] >= self.verify_interval: 
                    best_gallery_id, best_sim = self._find_best_match_in_gallery(feature)
                    
                    should_correct = False
                    should_demote = False
                    is_current_known = self._has_gallery_identity(current_person_id)
                    current_sim = 0.0

                    if is_current_known:
                        current_features = self._get_gallery_features(current_person_id)
                        current_sim = self._compute_similarity(feature, current_features) if current_features is not None else 0.0
                    
                    if best_gallery_id is not None and best_gallery_id != current_person_id:
                        if not is_current_known:
                            # 1. 当前是临时ID -> 只要匹配度达标就修正 (从未知变已知)
                            if best_sim > self.known_similarity_threshold:
                                should_correct = True
                                print(f"[ReID] 发现已知身份: Track {track_id} ({current_person_id}) -> {best_gallery_id} (Sim: {best_sim:.2f})")
                        else:
                            # 2. 当前已经是已知ID -> 需要更强的证据才切换 (防止跳变)
                            if best_sim > current_sim + 0.15 or (current_sim < self.known_similarity_threshold - 0.1 and best_sim > self.known_similarity_threshold):
                                should_correct = True
                                print(f"[ReID] 纠正身份: Track_Id: {track_id} current_id:{current_person_id}(sim:{current_sim:.2f}) -> {best_gallery_id}({best_sim:.2f})")

                    if not is_current_known:
                        best_unknown_id, best_unknown_sim = self._find_best_unknown_match(feature)
                        if best_unknown_id is not None and best_unknown_id != current_person_id:
                            should_correct = True
                            best_gallery_id = best_unknown_id
                            print(f"[ReID] 发现未知身份: Track {track_id} ({current_person_id}) -> {best_unknown_id} (Sim: {best_unknown_sim:.2f})")

                    if (
                        is_current_known
                        and not should_correct
                        and current_sim < self.known_similarity_threshold - 0.15
                        and (best_gallery_id is None or best_sim < self.known_similarity_threshold)
                    ):
                        should_demote = True
                        print(f"[ReID] 已知身份降级: Track {track_id} {current_person_id} -> 临时ID (current_sim={current_sim:.2f})")

                    if should_correct:
                        self.track_mapper[track_id]['switch_from'] = current_person_id
                        current_person_id = best_gallery_id
                        self.track_mapper[track_id]['person_id'] = current_person_id
                        # 修正后，重置平滑特征为当前帧特征，消除历史错误积累
                        self.track_mapper[track_id]['feature'] = feature 
                    elif should_demote:
                        self.track_mapper[track_id]['switch_from'] = current_person_id
                        current_person_id = self._allocate_temporary_id()
                        self.track_mapper[track_id]['person_id'] = current_person_id
                        self.track_mapper[track_id]['feature'] = feature
                    
                    self.track_mapper[track_id]['frames_since_verify'] = 0
                
                # 计算置信度 (与Gallery的相似度)
                current_features = self._get_gallery_features(current_person_id)
                if current_features is not None:
                    match_score = self._compute_similarity(feature, current_features)
                else:
                    match_score = 0.0 # 临时ID

            # --- 情况 B: 这是一个新的 Track ID ---
            else:
                # 在 Gallery 中寻找匹配
                best_match_id, best_similarity = self._find_best_match_in_gallery(feature)
                
                if best_match_id is not None:
                    # 匹配到已知身份
                    current_person_id = best_match_id
                    match_score = best_similarity
                else:
                    best_unknown_id, best_unknown_similarity = self._find_best_unknown_match(feature)
                    if best_unknown_id is not None:
                        current_person_id = best_unknown_id
                        match_score = best_unknown_similarity
                    else:
                        # 未匹配，分配新临时ID
                        # [修改] 根据模式调用不同的 ID 获取方式
                        current_person_id = self._allocate_temporary_id()
                        match_score = 0.0
                
                # 注册到映射表
                self.track_mapper[track_id] = {
                    'person_id': current_person_id,
                    'feature': feature,
                    'frames_since_verify': 0,
                    'age': 1, # [新增] 初始化 Age
                    'switch_from': None,
                    'missed_frames': 0
                }
            
            assigned_ids[original_idx] = current_person_id
            assigned_scores[original_idx] = match_score

        self.refresh_track_state(track_ids)

        return assigned_ids, assigned_scores

    def _find_best_match_in_gallery(self, feature):
        """辅助函数：在特征库中寻找最佳匹配"""
        best_match_id = None
        best_similarity = 0.0
        
        gallery_items = self._gallery_items_snapshot()
        for person_id, gallery_features in gallery_items:
            similarity = self._compute_similarity(feature, gallery_features)
            
            # 动态阈值
            current_threshold = self.known_similarity_threshold
            if self._gallery_size() > 5: 
                current_threshold += 0.02

            if similarity >= current_threshold and similarity > best_similarity:
                best_match_id = person_id
                best_similarity = similarity
        
        return best_match_id, best_similarity

    def update_identity(self, track_id, new_person_id, person_crop=None,
                        box=None, all_boxes=None, frame=None, crop_coords=None):
        """
        当人脸识别确认身份后，修正 ReID 的身份映射。
        参数:
            track_id: 当前人的追踪ID
            new_person_id: 人脸识别出的真实身份ID
            person_crop: (可选) 人物裁剪图，用于保存到Gallery磁盘
            box: (可选) 当前人的检测框 [x1,y1,x2,y2]，用于重叠检测
            all_boxes: (可选) 当前帧所有检测框，用于重叠检测
            frame: (可选) 完整帧，用于生成外扩锚点图
            crop_coords: (可选) 当前人物crop坐标 (x1, y1, x2, y2)
        """
        new_person_id = self._normalize_identity_value(new_person_id)
        if (
            new_person_id == 'Unknown'
            or not self._is_known_identity(new_person_id)
            or track_id not in self.track_mapper
        ):
            return

        old_id = self.track_mapper[track_id]['person_id']

        # 标记是否发生了身份切换
        identity_changed = str(old_id) != str(new_person_id)
        
        # 如果身份确实发生了变化 (例如从 临时ID 变成了 'ZhangSan')
        if identity_changed:
            print(f"[ReID] 人脸校准: Track {track_id} ({old_id}) -> {new_person_id}")
            
            # 1. 身份合并 (如果是临时ID -> 真实ID)
            is_temp_id = isinstance(old_id, int) or (isinstance(old_id, str) and old_id.isdigit())
            if is_temp_id:
                self._merge_identities(old_id, new_person_id)
            
            # 2. 更新当前 Track 的映射
            self.track_mapper[track_id]['switch_from'] = old_id
            self.track_mapper[track_id]['person_id'] = new_person_id
            
            # 3. 重置校验计数器，避免刚修正完又被ReID改回去
            self.track_mapper[track_id]['frames_since_verify'] = -30

        if not self._is_auto_save_enabled():
            return

        # 4. 尝试更新特征库
        if person_crop is not None and person_crop.size > 0:
            # [新增] 重叠检测：如果当前框与其他人框重叠过大，跳过保存
            if box is not None and all_boxes is not None and len(all_boxes) > 1:
                for other_box in all_boxes:
                    if np.array_equal(box, other_box):
                        continue
                    iou = self._compute_iou(box, other_box)
                    if iou > self.SAFE_UPDATE_IOU_THRESHOLD:
                        print(f"[ReID] 人脸校准保存跳过: 检测到重叠 (IOU={iou:.2f})")
                        return

            try:
                # [需求1: 锚点特征管理] 人脸识别确认的特征作为高置信度锚点保存
                feature_vec = self.track_mapper[track_id]['feature'].copy()
                anchor_crop = self._build_expanded_anchor_crop(frame, crop_coords)
                if anchor_crop is None or anchor_crop.size == 0:
                    anchor_crop = person_crop
                self._save_anchor_to_gallery(new_person_id, anchor_crop, feature_vec)
                
            except Exception as e:
                print(f"[ReID] 人脸校准后保存锚点失败: {e}")

    def _check_basic_image_quality(self, image):
        """
        基础图像质量检查：尺寸和模糊度。
         - 尺寸过小的图像通常质量较差，不适合保存。
        """
        if image is None or image.size == 0:
            return False
        h, w = image.shape[:2]
        if h < 150 or w < 100:
            return False
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        return blur_score >= 30

    def save_unknown_reid_sample(self, track_id, person_id, image, box=None, frame_shape=None, all_boxes=None, current_idx=None):
        """
        智能保存未知身份的 ReID 样本到共享存储。
        """
        if (
            self.shared_unknown_store is None
            or not self._is_auto_save_enabled()
            or person_id in (-1, None)
            or not self._is_temporary_identity(person_id)
            or track_id not in self.track_mapper
            or image is None
            or image.size == 0
        ):
            return False

        current_age = self.track_mapper[track_id].get('age', 0)
        if current_age < self.MIN_TRACK_AGE:
            return False

        if box is not None and frame_shape is not None and not self._check_image_geometry(box, frame_shape):
            return False

        if all_boxes is not None and current_idx is not None and self._check_overlap(current_idx, all_boxes):
            return False

        if not self._check_upper_body_quality(image, require_face=False):
            return False

        if not self._check_basic_image_quality(image):
            return False

        feature_vec = self.track_mapper[track_id]['feature'].copy()
        current_features = self._get_unknown_gallery_features(person_id)
        if current_features is not None and len(current_features) > 0:
            sim = self._compute_similarity(feature_vec, current_features)
            if sim > self.TEMP_ID_DEDUP_THRESHOLD:
                return False

        target_dir = os.path.join(self.shared_unknown_store.reid_folder, str(person_id))
        os.makedirs(target_dir, exist_ok=True)
        existing_files = list(self.shared_unknown_store.reid_gallery_files.get(person_id, []))
        if len(existing_files) >= self.MAX_UNKNOWN_REID_IMAGES:
            oldest = existing_files[0]
            oldest_path = os.path.join(target_dir, oldest)
            if os.path.exists(oldest_path):
                os.remove(oldest_path)
            self._remove_unknown_gallery_file_entry(person_id, oldest)

        filename = self._generate_prefixed_filename(target_dir, 'reid')
        save_path = os.path.join(target_dir, filename)
        cv2.imwrite(save_path, image)
        self._append_unknown_gallery_file_entry(person_id, filename, feature_vec)
        return True

    def _cleanup_temp_identities(self, max_days=1):
        """
        清理过期的临时档案。
        
        参数:
            max_days (int): 临时档案保留的最大天数。超过这个时间的数字ID文件夹将被删除。
        """
        print(f"[ReID] 正在清理超过 {max_days} 天的临时档案...")
        if not os.path.exists(self.identity_folder):
            return

        current_time = time.time()
        deleted_count = 0

        for d in os.listdir(self.identity_folder):
            dir_path = os.path.join(self.identity_folder, d)
            
            # 只清理数字命名的文件夹 (假设这些是自动生成的临时ID)
            # 且必须是文件夹
            if os.path.isdir(dir_path) and d.isdigit():
                # 获取文件夹最后修改时间
                mtime = os.path.getmtime(dir_path)
                
                # 如果超过了保留期限
                if (current_time - mtime) > (max_days * 24 * 3600):
                    try:
                        shutil.rmtree(dir_path) # 递归删除文件夹
                        deleted_count += 1
                        # print(f"[ReID] 已删除过期临时档案: {d}")
                    except Exception as e:
                        print(f"[ReID] 删除失败 {d}: {e}")
        
        if deleted_count > 0:
            print(f"[ReID] 清理完成，共删除了 {deleted_count} 个过期临时档案。")
        else:
            print("[ReID] 没有发现过期的临时档案。")
    
    # [新增] 添加关键点质量检查方法
    def _check_upper_body_quality(self, image, require_face=True):
        """
        检查图像是否包含完整的上半身关键点。
        参数:
            require_face (bool): 是否强制要求检测到正面人脸
        """
        if self.pose_estimator is None:
            print("[ReID] 警告: 未配置姿态估计器，跳过关键点质量检查。")
            return True
        
        try:
            # 使用姿态估计器检测关键点
            pose_results = self.pose_estimator(image.copy(), verbose=False, conf=0.7)
            
            if not pose_results or len(pose_results[0].keypoints.data) == 0:
                print("[ReID] 关键点检测失败或无关键点。")
                return False
            
            # 获取关键点数据 [x, y, confidence]
            keypoints = pose_results[0].keypoints.data[0].cpu().numpy()
            
            # ====== 检测头部关键点 ======
            # COCO格式：0-鼻子, 1-左眼, 2-右眼, 3-左耳, 4-右耳
            has_nose = (0 < len(keypoints) and keypoints[0, 2] > self.KEYPOINT_CONFIDENCE_THRESHOLD)
            has_left_eye = (1 < len(keypoints) and keypoints[1, 2] > self.KEYPOINT_CONFIDENCE_THRESHOLD)
            has_right_eye = (2 < len(keypoints) and keypoints[2, 2] > self.KEYPOINT_CONFIDENCE_THRESHOLD)
            
            visible_head_parts = 0
            head_keypoints = [0, 1, 2, 3, 4]
            for kp_idx in head_keypoints:
                if kp_idx < len(keypoints) and keypoints[kp_idx, 2] > self.KEYPOINT_CONFIDENCE_THRESHOLD:
                    visible_head_parts += 1
            
            # ====== 检测肩部关键点 ======
            left_shoulder_visible = (
                self.UPPER_BODY_KEYPOINTS['left_shoulder'] < len(keypoints) and
                keypoints[self.UPPER_BODY_KEYPOINTS['left_shoulder'], 2] > self.KEYPOINT_CONFIDENCE_THRESHOLD
            )
            right_shoulder_visible = (
                self.UPPER_BODY_KEYPOINTS['right_shoulder'] < len(keypoints) and
                keypoints[self.UPPER_BODY_KEYPOINTS['right_shoulder'], 2] > self.KEYPOINT_CONFIDENCE_THRESHOLD
            )
            
            # ====== 检测髋部关键点 ======
            left_hip_visible = (
                self.UPPER_BODY_KEYPOINTS['left_hip'] < len(keypoints) and
                keypoints[self.UPPER_BODY_KEYPOINTS['left_hip'], 2] > self.KEYPOINT_CONFIDENCE_THRESHOLD
            )
            right_hip_visible = (
                self.UPPER_BODY_KEYPOINTS['right_hip'] < len(keypoints) and
                keypoints[self.UPPER_BODY_KEYPOINTS['right_hip'], 2] > self.KEYPOINT_CONFIDENCE_THRESHOLD
            )
            
            # ====== 判断逻辑 ======
            if require_face:
                # 必须有鼻子和双眼，且双肩可见，且双侧髋部可见
                is_frontal_face = has_nose and (has_left_eye and has_right_eye)
                return is_frontal_face and left_shoulder_visible and right_shoulder_visible and left_hip_visible and right_hip_visible

            # [普通模式] 允许背面
            is_frontal_view = (
                visible_head_parts >= 1 and 
                left_shoulder_visible and right_shoulder_visible and 
                (left_hip_visible and right_hip_visible)
            )
            
            is_back_view = (
                left_shoulder_visible and right_shoulder_visible and
                left_hip_visible and right_hip_visible
            )
            
            return is_frontal_view or is_back_view
        

        except Exception as e:
            print(f"[ReID] 姿态检测异常: {e}")
            return False

    # --- 新增辅助函数：检查图像几何约束 ---
    def _check_image_geometry(self, box, frame_shape):
        """
        检查检测框的几何属性是否满足采集要求：
        1. 不紧贴边缘
        2. 不遮挡时间戳
        3. 尺寸足够大
        """

        if frame_shape is None: 
            print(f"[Debug] frame_shape 为空")
            return False
        
        h_img, w_img = frame_shape[:2]
        x1, y1, x2, y2 = box
        w_box = x2 - x1
        h_box = y2 - y1
        
        # # 1. 检查边缘边距
        # if (x1 < self.EDGE_MARGIN or y1 < self.EDGE_MARGIN or 
        #     x2 > w_img - self.EDGE_MARGIN or y2 > h_img - self.EDGE_MARGIN):
        #     print(f"[Debug] 过于靠边")
        #     return False
            
        # 2. 检查时间戳重叠
        if self._is_timestamp_overlap(box, frame_shape):
            print(f"[Debug] 遮挡时间戳")
            return False
            
        # 3. 检查尺寸
        if h_box < self.MIN_CROP_HEIGHT:
            print(f"[Debug] 高度不足: {h_box} < {self.MIN_CROP_HEIGHT}")
            return False
            
        return True

    # --- 新增辅助函数：临时ID跨库去重 ---
    def _check_temp_identity_dedup(self, new_feature, current_person_id):
        """
        检查临时ID候选样本是否与当前特征库中的任一身份过于相似。
        返回: True (允许保存), False (与现有库高度重复，跳过保存)
        """
        gallery_items = self._gallery_items_snapshot()
        if not gallery_items:
            return True

        best_match_id = None
        best_similarity = 0.0
        for person_id, gallery_features in gallery_items:
            if gallery_features is None or len(gallery_features) == 0:
                continue

            similarity = self._compute_similarity(new_feature, gallery_features)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_id = person_id

        if best_match_id is not None and best_similarity >= self.TEMP_ID_DEDUP_THRESHOLD:
            print(
                f"[ReID] 临时ID去重拦截: ID {current_person_id} 与库中身份 {best_match_id} "
                f"相似度 {best_similarity:.3f} >= {self.TEMP_ID_DEDUP_THRESHOLD:.3f}，跳过落盘"
            )
            return False

        return True
    # --- 新增辅助函数：检查特征多样性 ---
    def _check_feature_diversity(self, person_id, new_feature):
        """
        检查新特征是否与现有特征库足够不同（用于判断是否换装/新角度）。
        返回: True (有差异，建议保存), False (太相似，跳过)
        """
        gallery_features = self._get_gallery_features(person_id)
        if gallery_features is None:
            return True # 库里没特征，肯定要存
        
        # 计算新特征与库中所有特征的相似度
        # new_feature: (dim,) -> (1, dim)
        # gallery_features: (N, dim)
        sims = self._compute_similarity(new_feature, gallery_features)
        
        # 注意：_compute_similarity 返回的是最大相似度 float
        # 如果最大相似度 > 阈值，说明库里已经有非常像的照片了（衣服没变）
        # 阈值建议：0.92 左右。太低会导致存太多重复图，太高会漏掉细微变化。
        if sims > self.DIVERSITY_THRESHOLD: 
            # print(f"[Debug] ID {person_id} 特征重复 (Max Sim: {sims:.3f})，跳过更新")
            return False
            
        print(f"[ReID] ID {person_id} 检测到新特征 (Max Sim: {sims:.3f})，可能是换装或新角度")
        return True

    # --- [需求2] 锚点门控检查 ---
    def _check_anchor_gate(self, person_id, new_feature):
        """
        [普通特征准入规则] 锚点强监督门控。
        存在锚点时：新特征与锚点的最高相似度必须 >= ANCHOR_GATE_THRESHOLD，否则拦截。
        无锚点时：降级跳过，由后续多样性检查兜底。
        """
        anchor_features = self._get_anchor_features(person_id)
        if anchor_features is None or len(anchor_features) == 0:
            return True  # 无锚点降级：跳过门控
        sim = self._compute_similarity(new_feature, anchor_features)
        
        if sim < self.ANCHOR_GATE_THRESHOLD:
            print(f"[ReID] 锚点门控拦截: ID {person_id} 新特征与锚点相似度 {sim:.3f} < {self.ANCHOR_GATE_THRESHOLD}，疑似漂移")
            return False
        return True

    # --- [需求1] 锚点特征保存 ---
    def _save_anchor_to_gallery(self, person_id, image, feature_vector):
        """
        [锚点特征管理] 将人脸识别确认的特征作为锚点保存。
        执行多样性拦截（>0.95拒绝冗余）和独立容量限制（最多2张）。
        """
        if not self._is_auto_save_enabled():
            return False

        person_id = self._normalize_identity_value(person_id)
        if not self._is_known_identity(person_id):
            print(f"[ReID] 锚点保存跳过: ID {person_id} 不是已知身份。")
            return False

        target_dir = os.path.join(self.identity_folder, str(person_id))
        
        # --- 基础质量检查（尺寸和模糊度）---
        h, w = image.shape[:2]
        if h < 50 or w < 30:
            print(f"[ReID] 锚点保存跳过: 图像尺寸过小 ({w}x{h})")
            return False
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        if blur_score < 30:
            print(f"[ReID] 锚点保存跳过: 图像过于模糊 (blur_score={blur_score:.1f})")
            return False
        
        # --- [需求1: 多样性拦截] 与已有锚点相似度 > 0.95 则属于冗余锚点，拒绝保存 ---
        current_anchors = self._get_anchor_features(person_id)
        if current_anchors is not None and len(current_anchors) > 0:
            sim = self._compute_similarity(feature_vector, current_anchors)
            if sim > self.ANCHOR_DIVERSITY_THRESHOLD:
                print(f"[ReID] 锚点去重拦截: ID {person_id} 相似度 {sim:.3f} > {self.ANCHOR_DIVERSITY_THRESHOLD}，冗余锚点")
                return False
        
        # --- [需求1: 独立容量限制] 最多 MAX_ANCHOR_IMAGES 张，超出删最早的 ---
        os.makedirs(target_dir, exist_ok=True)
        existing_anchors = sorted([f for f in os.listdir(target_dir)
                                   if f.lower().endswith(('.jpg', '.png')) and f.startswith('anchor_')])
        
        if len(existing_anchors) >= self.MAX_ANCHOR_IMAGES:
            oldest = existing_anchors[0]
            try:
                os.remove(os.path.join(target_dir, oldest))
                self._remove_gallery_file_entry(person_id, oldest)
                print(f"[ReID] 锚点容量轮替: ID {person_id} 删除最早锚点 {oldest}")
            except Exception as e:
                print(f"[ReID] 删除旧锚点失败: {e}")
        
        # --- 执行保存（使用 anchor_ 前缀与普通特征物理隔离）---
        try:
            filename = self._generate_anchor_filename(target_dir)
            save_path = os.path.join(target_dir, filename)
            cv2.imwrite(save_path, image)
            
            # 更新内存中的锚点库
            self._append_gallery_file_entry(person_id, filename, feature_vector, is_anchor=True)
            
            anchor_count = len([f for f in os.listdir(target_dir)
                                 if f.startswith('anchor_') and f.lower().endswith(('.jpg', '.png'))])
            print(f"[ReID] 锚点保存成功: ID {person_id}，当前锚点数: {anchor_count}")
            return True
        except Exception as e:
            print(f"[ReID] 锚点保存失败: {e}")
            return False

    # --- [需求3] 双轨制淘汰保护：管理画廊容量 ---
    def _manage_gallery_capacity(self, person_id):
        """
        [双轨制淘汰保护] 确保每个人的画廊不会无限膨胀。
        只淘汰普通特征(auto_前缀)，锚点特征(anchor_前缀)绝对豁免。
        """
        target_dir = os.path.join(self.identity_folder, str(person_id))
        if not os.path.exists(target_dir):
            return

        files = sorted([f for f in os.listdir(target_dir) if f.lower().endswith(('.jpg', '.png'))])
        
        if len(files) <= self.MAX_GALLERY_IMAGES:
            return

        # 找出所有自动采集的文件 (以 auto_ 开头)
        auto_files = [f for f in files if f.startswith('auto_')]
        
        # 如果有自动采集的文件，按时间戳排序删除最早的
        if auto_files:
            # auto_files 已经是按文件名排序的（因为时间戳在文件名里），直接删第一个
            file_to_remove = auto_files[0]
            try:
                os.remove(os.path.join(target_dir, file_to_remove))
                self._remove_gallery_file_entry(person_id, file_to_remove)
                print(f"[ReID] ID {person_id} 画廊已满，轮替删除旧样本: {file_to_remove}")
            except Exception as e:
                print(f"[ReID] 删除文件失败: {e}")
        else:
            # 如果全是人工放的图，通常不删，或者强制删最早的
            pass

    def _process_auto_save(self, track_id, person_id, image, feature, box, frame_shape, all_boxes, current_idx):
        """
        处理自动采集逻辑：
        结合纠错机制（Track Age）、几何约束、重叠检测和姿态质量。
        """
        if not self._is_auto_save_enabled():
            return

        # 1. 纠错机制：必须追踪稳定一定帧数后才采集
        current_age = self.track_mapper[track_id].get('age', 0)
        if current_age < self.MIN_TRACK_AGE:
            # print(f"[Debug] ID {person_id} (Track {track_id}) 追踪时间不足: {current_age}/{self.MIN_TRACK_AGE}")
            return

        # 2. 临时ID只存1张；已知ID交给 _smart_save_to_gallery 的多样性检查决定
        is_temp_id = str(person_id).isdigit()
        if is_temp_id:
            target_dir = os.path.join(self.identity_folder, str(person_id))
            if os.path.exists(target_dir):
                files = [f for f in os.listdir(target_dir) if f.lower().endswith(('.jpg', '.png'))]
                if len(files) > 0:
                    return

        # 3. 几何约束检查 (边缘、时间戳、尺寸、比例)
        if not self._check_image_geometry(box, frame_shape):
            # print(f"[Debug] ID {person_id} 几何检查未通过 (太小/靠边/比例不对)")
            return
         
        # 4. 重叠检查 (避免多人重叠)
        if self._check_overlap(current_idx, all_boxes):
            # print(f"[Debug] ID {person_id} 检测到与其他框重叠，跳过")
            return
        
        # 5. 姿态质量检查 (强制正面人脸 + 完整上半身)
        # 必须检测到完整人脸和上半身关节
        if not self._check_upper_body_quality(image, require_face=True):
            # print(f"[Debug] ID {person_id} 姿态质量不达标 (关键点缺失)")
            return
        
        # 6. 执行保存
        print(f"[ReID] >>> 自动采集成功: ID {person_id} (Track {track_id}) <<<")
        self._smart_save_to_gallery(person_id, image, feature)

