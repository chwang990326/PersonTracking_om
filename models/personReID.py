import os
import cv2
import numpy as np
import torch
import threading
import time
from models.personvit_adapter import PersonViTFeatureExtractor
from models.reid_state import SharedIdentityStore
from models.transreid_pytorch.utils.reranking import re_ranking

auto_save = True

class PersonReidentifier:
    """
    基于预定义的身份库（Gallery）进行人物重识别。
    """

    def __init__(self, identity_folder='identity',
                 model_path='weights/transformer_120_16.om',
                 config_file='./models/transreid_pytorch/configs/market/vit_base.yml',
                 similarity_threshold=0.9,
                 known_similarity_threshold=None,
                 unknown_similarity_threshold=None,
                 device='cuda', max_age=30, iou_threshold=0.3, feature_smooth_alpha=0.8,
                 pose_estimator=None, verify_interval=2,
                 feature_extractor=None,
                 id_generator=None,
                 shared_identity_store=None,
                 shared_unknown_store=None,
                 redis_memory=None):
        """
        初始化ReID系统。

        参数:
            identity_folder (str): 存放已知人物图片的根文件夹。
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
            redis_memory: RedisIdentityMemory 实例（可选，有则使用 Redis 中央记忆）
        """
        print("[ReID] 初始化系统...")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ReID model was not found: {model_path}")
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
        self.redis_memory = redis_memory

        # ID生成策略：优先使用外部传入的生成器（全局唯一），否则使用本地逻辑
        self.id_generator = id_generator

        if self.id_generator is None:
            # 本地 fallback: 使用简单的自增计数器（不再扫描文件系统）
            self._local_id_counter = 1
            print(f"[ReID] (本地模式) 使用本地自增 ID 计数器（不扫描 identity 目录）")
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
        if person_id in (None, '', -1):
            return False
        if isinstance(person_id, int):
            return True
        if isinstance(person_id, str):
            # 数字字符串（兼容旧数据）
            if person_id.isdigit():
                return True
            # unknown:UUID (如 unknown:a1b2c3d4...)
            if person_id.startswith("unknown:"):
                return True
            # 纯 UUID hex（兼容旧格式）
            if len(person_id) == 32 and all(c in '0123456789abcdef' for c in person_id):
                return True
        return False

    def _is_known_identity(self, person_id):
        person_id = self._normalize_identity_value(person_id)
        if person_id in (None, '', -1):
            return False
        return not self._is_temporary_identity(person_id)

    def _is_auto_save_enabled(self):
        return bool(auto_save)

    def _allocate_temporary_id(self):
        """
        分配一个全局唯一的临时ID（UUID），优先使用外部生成器（Redis），
        否则使用本地 uuid4。
        """
        if self.id_generator:
            return self.id_generator()

        import uuid
        return f"unknown:{uuid.uuid4().hex}"

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
        Redis 模式下不支持直接获取原始特征，返回 None。
        """
        return None

    def _unknown_gallery_items_snapshot(self):
        """
        获取未知身份库的快照列表，格式为 [(person_id, features), ...]。
        Redis 模式下不支持本地快照，返回空列表。
        """
        return []

    def _append_unknown_gallery_file_entry(self, person_id, filename, feature_vector):
        """
        将新特征写入未知身份库。
        Redis 模式下，样本已在 add_unknown_reid_sample 中写入，此处仅 touch 续期。
        """
        if self.shared_unknown_store is None:
            return
        self.shared_unknown_store.touch_entity(person_id)

    def _remove_unknown_gallery_file_entry(self, person_id, filename):
        """Redis 模式下由 release_unknown / release_if_empty 管理，本地无需操作。"""
        return False

    def _find_best_unknown_match(self, feature):
        """使用 Redis 中央记忆库检索 unknown ReID。"""
        if self.redis_memory is not None and self.redis_memory.available:
            try:
                unknown_id, similarity = self.redis_memory.search_unknown_reid(
                    feature, self.unknown_similarity_threshold
                )
                if unknown_id is not None:
                    return unknown_id, similarity
                return None, 0.0
            except Exception as e:
                print(f"[ReID] Redis 搜索 unknown_reid 失败: {e}")

        return None, 0.0

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

    def _smart_save_to_gallery(self, person_id, image, feature_vector, force=False):
        """
        智能保存已知身份 ReID 特征到 Redis 中央记忆库。
        force: 是否强制保存（用于人脸识别确认后的高质量图）
        image 参数保留用于接口兼容，实际不再写盘。
        """
        if not self._is_auto_save_enabled():
            return False

        is_temp_id = self._is_temporary_identity(person_id)

        # 临时ID：不保存已知库，只记录即可
        if is_temp_id:
            if not force:
                if not self._check_temp_identity_dedup(feature_vector, person_id):
                    return False
                if self.redis_memory is not None and self.redis_memory.available:
                    count = self.redis_memory.count_known_samples(str(person_id), "reid")
                    if count > 0:
                        return False
                elif self._get_gallery_features(person_id) is not None:
                    return False

        # 已知ID：执行换装更新策略
        else:
            if not force:
                if not self._check_anchor_gate(person_id, feature_vector):
                    return False
                if not self._check_feature_diversity(person_id, feature_vector):
                    return False

            # 容量管理委托给 Redis
            if self.redis_memory is not None and self.redis_memory.available:
                pass  # add_known_reid_sample 内部有容量管理

        # 公共质量检查
        if force:
            h, w = image.shape[:2]
            if h < 50 or w < 30:
                return False
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if cv2.Laplacian(gray, cv2.CV_64F).var() < 30:
                return False
        else:
            if not self._check_upper_body_quality(image, require_face=True):
                return False

        # 执行保存（写入 Redis + 更新本地缓存）
        try:
            if self.redis_memory is not None and self.redis_memory.available:
                self.redis_memory.add_known_reid_sample(str(person_id), feature_vector, is_anchor=False)

            # 更新本地内存库（保持本地缓存同步）
            self._append_gallery_file_entry(person_id, f"auto_{int(time.time() * 1000)}", feature_vector, is_anchor=False)

            if not is_temp_id:
                print(f"[ReID] ID {person_id} 特征已写入 Redis (换装/新角度)")
            else:
                print(f"[ReID] ID {person_id} 补充新特征已写入 Redis")
            return True
        except Exception as e:
            print(f"[ReID] 保存失败: {e}")
            return False

    def _merge_identities(self, temp_id, real_id):
        """
        将临时ID的数据合并到真实ID中。
        不再涉及文件系统操作，仅合并内存中的特征库。
        """
        print(f"[ReID] 正在合并身份: {temp_id} -> {real_id}")

        # 合并内存中的特征库
        with self.gallery_lock:
            self._merge_store_entries(self.feature_gallery, self.gallery_files, temp_id, real_id)
            self._merge_store_entries(self.anchor_gallery, self.anchor_files, temp_id, real_id)

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
        """辅助函数：在特征库中寻找最佳匹配（优先 Redis）。"""
        # 动态阈值
        current_threshold = self.known_similarity_threshold
        gallery_size = self._gallery_size()
        if gallery_size > 5:
            current_threshold += 0.02

        # 优先使用 Redis 中央记忆库
        if self.redis_memory is not None and self.redis_memory.available:
            try:
                person_id, similarity = self.redis_memory.search_known_reid(feature, current_threshold)
                if person_id is not None:
                    return person_id, similarity
                return None, 0.0
            except Exception as e:
                print(f"[ReID] Redis 搜索 known_reid 失败，回退到本地: {e}")

        # 本地线性扫描 fallback
        best_match_id = None
        best_similarity = 0.0
        gallery_items = self._gallery_items_snapshot()
        for person_id, gallery_features in gallery_items:
            similarity = self._compute_similarity(feature, gallery_features)
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
            is_temp_id = self._is_temporary_identity(old_id)
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
        保存未知身份的 ReID 特征到 Redis 中央记忆库。
        image 参数保留用于接口兼容和质量检查，实际不再写盘。
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

        # 去重：通过 Redis 已有样本检查
        if self.redis_memory is not None and self.redis_memory.available:
            try:
                top_matches = self.redis_memory.search_unknown_reid(
                    feature_vec, self.TEMP_ID_DEDUP_THRESHOLD, top_k=5
                )
                if top_matches[0] is not None and str(top_matches[0]) == str(person_id) and top_matches[1] > self.TEMP_ID_DEDUP_THRESHOLD:
                    return False
            except Exception:
                pass

        # 写入 Redis 中央记忆库（内部已包含 touch/续期）
        if self.redis_memory is not None and self.redis_memory.available:
            try:
                self.redis_memory.add_unknown_reid_sample(str(person_id), feature_vec)
            except Exception as e:
                print(f"[ReID] Redis 写入 unknown_reid 失败: {e}")
                return False

        return True

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

    def _save_anchor_to_gallery(self, person_id, image, feature_vector):
        """
        锚点特征保存到 Redis 中央记忆库。
        执行多样性拦截（>0.95拒绝冗余）和独立容量限制（最多2张）。
        image 参数保留用于接口兼容和质量检查，实际不再写盘。
        """
        if not self._is_auto_save_enabled():
            return False

        person_id = self._normalize_identity_value(person_id)
        if not self._is_known_identity(person_id):
            return False

        # 基础质量检查（尺寸和模糊度）
        h, w = image.shape[:2]
        if h < 50 or w < 30:
            return False
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if cv2.Laplacian(gray, cv2.CV_64F).var() < 30:
            return False

        # 多样性拦截：与已有锚点相似度 > 0.95 则拒绝
        current_anchors = self._get_anchor_features(person_id)
        if current_anchors is not None and len(current_anchors) > 0:
            sim = self._compute_similarity(feature_vector, current_anchors)
            if sim > self.ANCHOR_DIVERSITY_THRESHOLD:
                print(f"[ReID] 锚点去重拦截: ID {person_id} 相似度 {sim:.3f} > {self.ANCHOR_DIVERSITY_THRESHOLD}，冗余锚点")
                return False

        # 执行保存（写入 Redis + 更新本地缓存）
        try:
            if self.redis_memory is not None and self.redis_memory.available:
                self.redis_memory.add_known_reid_sample(str(person_id), feature_vector, is_anchor=True)

            filename = f"anchor_{int(time.time())}"
            self._append_gallery_file_entry(person_id, filename, feature_vector, is_anchor=True)

            print(f"[ReID] 锚点保存成功: ID {person_id}")
            return True
        except Exception as e:
            print(f"[ReID] 锚点保存失败: {e}")
            return False

    def _manage_gallery_capacity(self, person_id):
        """容量管理已委托给 Redis add_known_reid_sample 内部处理。"""
        pass

    def _process_auto_save(self, track_id, person_id, image, feature, box, frame_shape, all_boxes, current_idx):
        """
        处理自动采集逻辑（不再写盘，特征写入 Redis）。
        结合纠错机制（Track Age）、几何约束、重叠检测和姿态质量。
        """
        if not self._is_auto_save_enabled():
            return

        current_age = self.track_mapper[track_id].get('age', 0)
        if current_age < self.MIN_TRACK_AGE:
            return

        # 临时ID只存1张
        is_temp_id = self._is_temporary_identity(person_id)
        if is_temp_id:
            if self._get_gallery_features(person_id) is not None:
                return

        if not self._check_image_geometry(box, frame_shape):
            return

        if self._check_overlap(current_idx, all_boxes):
            return

        if not self._check_upper_body_quality(image, require_face=True):
            return

        print(f"[ReID] >>> 自动采集成功: ID {person_id} (Track {track_id}) <<<")
        self._smart_save_to_gallery(person_id, image, feature)

