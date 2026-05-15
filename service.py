import os
import torch
import cv2
import numpy as np
import base64
import threading
import time
from collections import defaultdict, deque

try:
    import onnxruntime as ort
except ImportError:
    ort = None

from models.loader import load_config
from models.geometry import (
    get_world_coords_from_pose, image_to_world_plane, 
    world_to_cad, create_person_info
)
from models.posture_classifier import classify_posture_with_verification
from models.personReID import PersonReidentifier
from models.face import FaceRecognizer
from models.unknown_entity_store import UnknownEntityStore
from utils.get_model import getmodel
from models.actionclassifier import preprocess_crops_for_video_cls, postprocess, crop_and_pad
# [新增] 导入特征提取器类
from models.personvit_adapter import PersonViTFeatureExtractor
from models.reid_state import SharedIdentityStore
from models.ascend_action import AscendActionModel
from models.ascend_backend import is_om_path, resolve_model_path
from models.ascend_yolo import create_yolo_model
from utils.profiler import RequestProfiler

DETECTOR_ONNX_PATH = resolve_model_path('weights/yolo26x.om', 'weights/yolo26x.onnx')
POSE_ONNX_PATH = resolve_model_path('weights/yolo26s-pose.om', 'weights/yolo26s-pose.onnx')
PHONE_DETECTOR_ONNX_PATH = resolve_model_path('weights/yolo26x.om', 'weights/yolo26x.onnx')
REID_MODEL_PATH = resolve_model_path('weights/transformer_120_16.om', './config/transformer_120.pth')
ACTION_MODEL_PATH = resolve_model_path('weights/best094nophone.om', 'weights/best094nophone.pt')
CELL_PHONE_CLASS_ID = 67

class CameraConfigError(RuntimeError):
    pass


class VisionAnalysisService:
    def __init__(self):
        print(">>> 正在初始化视觉分析服务...")
        
        # 1. 基础路径配置
        self.config_base_path = 'config'
        self.order_config_path = 'config/caculate_order.yaml'
        self.identity_folder = 'identity' # [显式定义]
        
        # 2. 加载共享模型
        self.using_om_models = any(
            is_om_path(path)
            for path in (DETECTOR_ONNX_PATH, POSE_ONNX_PATH, PHONE_DETECTOR_ONNX_PATH, REID_MODEL_PATH, ACTION_MODEL_PATH)
        )
        self.device = torch.device('cpu' if self.using_om_models else ('cuda' if torch.cuda.is_available() else 'cpu'))
        yolo_cuda_available = False
        if ort is not None:
            try:
                yolo_cuda_available = 'CUDAExecutionProvider' in ort.get_available_providers()
            except Exception:
                yolo_cuda_available = False
        self.yolo_device = 'cpu' if self.using_om_models else (0 if (self.device.type == 'cuda' or yolo_cuda_available) else 'cpu')
        print(f">>> 行为识别模型将使用设备: {self.device}")
        self.detector_path = DETECTOR_ONNX_PATH
        self.pose_path = POSE_ONNX_PATH
        
        print(">>> 加载 YOLO & Pose 模型...")
        
        self.person_detector = create_yolo_model(self.detector_path, task='detect') 
        self.phone_detector = create_yolo_model(PHONE_DETECTOR_ONNX_PATH, task='detect')
        self.pose_estimator = create_yolo_model(self.pose_path, task='pose')
        
        # [新增] 初始化共享的 ReID 特征提取模型 (只加载一次，节省显存并避免配置冲突)
        print(">>> 加载 ReID 特征提取模型 (共享)...")
        self.shared_reid_extractor = PersonViTFeatureExtractor(
            model_path=REID_MODEL_PATH,
            config_file='./models/transreid_pytorch/configs/market/vit_base.yml',
            device=str(self.device)
        )
        self.shared_identity_store = self._create_shared_identity_store()
        
        # [新增] 全局未知 ID 管理 (0-99 空闲复用)
        self.id_lock = threading.Lock()
        self.TEMP_ID_START = 0
        self.TEMP_ID_END = 99

        print(">>> 加载行为识别模型...")
        # 加载 Uniformer 模型
        self.action_model_path = ACTION_MODEL_PATH
        self.action_model = None
        self.action_model_lock = threading.Lock()
        print(">>> UniFormer 行为识别模型按需加载，默认不初始化")
        self.PHONE_CONFIDENCE_THRESHOLD = 0.8
        # 推理间隔保持为4帧（即每隔4帧，取过去16帧进行一次识别，形成滑动窗口）
        # [修改] 将原先的 2 改为 1，实现逐帧平滑判断，与 OM 逻辑对齐
        self.action_infer_interval = 1 
        self.v_max_mmps = 6000.0
        self.margin_mm = 100.0
        self.deadband_mm = 30.0
        self.ttl_keep_state = 1.0
        self.outlier_confirm = 2

        # [新增] 行为识别动态阈值配置
        self.ACTION_THRESHOLDS = {
            'climbing': 0.85,
            'falling_down': 0.0,
            'smoking': 0.85,      # 假设标签名为 smoking
            'reaching_high': 0.85, # 假设标签名为 reaching_high
            'sleeping': 0.8,
        }

        print(">>> 加载人脸识别模型 (AdaFace)...")
        self.shared_face_recognizer = FaceRecognizer(
            face_gallery_path='faceImage',
            scrfd_model_path='weights/det_10g.onnx',
            adaface_model_path='weights/adaface_ir50_ms1mv2.ckpt',
            architecture='ir_50',
            similarity_threshold=0.45,
            detection_threshold=0.65,
            db_path='./database/face_database'
        )
        self.image_face_recognizer = FaceRecognizer(
            face_gallery_path='faceImage',
            scrfd_model_path='weights/det_10g.onnx',
            adaface_model_path='weights/adaface_ir50_ms1mv2.ckpt',
            architecture='ir_50',
            similarity_threshold=0.35,
            detection_threshold=0.3,
            db_path='./database/face_database'
        )
        self.shared_unknown_store = UnknownEntityStore(
            face_folder='unknownFace',
            reid_folder='unknownIdentity',
            face_db_path='./database/unknown_face_database',
            temp_id_start=self.TEMP_ID_START,
            temp_id_end=self.TEMP_ID_END,
            ttl_seconds=300.0,
            face_similarity_threshold=self.shared_face_recognizer.similarity_threshold,
        )

        # 相机状态管理
        self.camera_states = {}
        self.camera_states_lock = threading.Lock()
        self.camera_params_cache = {}
        self.camera_params_lock = threading.Lock()
        self.FACE_CONFIRM_THRESHOLD = 3  # 确认人脸识别结果需要连续出现的次数
        self.identity_refresh_interval = 100.0
        self.identity_refresh_lock = threading.Lock()
        self.last_identity_refresh_check = 0.0
        print(">>> 服务初始化完成。")

    # [新增] 线程安全的 ID 生成器回调函数
    def _generate_next_global_id(self):
        with self.id_lock:
            return self.shared_unknown_store.allocate_id()

    def _create_shared_identity_store(self):
        return SharedIdentityStore(
            identity_folder=self.identity_folder,
            feature_extractor=self.shared_reid_extractor
        )

    def _get_action_model(self):
        if self.action_model is not None:
            return self.action_model

        with self.action_model_lock:
            if self.action_model is None:
                print(">>> 加载 UniFormer 行为识别模型...")
                if is_om_path(self.action_model_path):
                    self.action_model = AscendActionModel(self.action_model_path)
                else:
                    self.action_model = getmodel(self.action_model_path).eval().to(self.device)
        return self.action_model

    @staticmethod
    def _reset_uniformer_state(state):
        state['video_cache'] = defaultdict(lambda: deque(maxlen=16))
        state['frame_counter'] = defaultdict(int)
        state['action_cache'] = {}
        state['action_pred_cache'] = {}

    @staticmethod
    def _is_temporary_identity(person_id):
        """
        判断一个 person_id 是否属于临时未知 ID 的范围。
        """
        return isinstance(person_id, int) or (isinstance(person_id, str) and person_id.isdigit())

    def _is_strict_unknown_face_ready(self, kp_data):
        """
        严格模式下的人脸识别准备条件：关键点 1-4 的置信度都必须大于 0.5。
         - 这种方式可以更准确地判断是否正脸朝向摄像头，从而减少误判，但可能会略微降低召回率。
         - 如果关键点数据不足或格式不正确，则默认返回 False，表示不满足条件。
         - 注意：关键点索引和置信度位置需要根据实际模型输出进行调整，这里假设 kp_data[i, 2] 是第 i 个关键点的置信度。
        """
        if kp_data is None or len(kp_data) <= 4:
            return False
        return bool(
            kp_data[1, 2] > 0.5 and
            kp_data[2, 2] > 0.5 and
            kp_data[3, 2] > 0.5 and
            kp_data[4, 2] > 0.5
        )

    def _extract_face_candidates(self, person_crop, crop_x1, crop_y1):
        """
        从给定的人体裁剪图像中提取人脸候选区域，并进行识别匹配。
        """
        if person_crop is None or person_crop.size == 0:
            return []

        try:
            bboxes, kpss = self.shared_face_recognizer.detector.detect(person_crop, max_num=0)
        except Exception:
            return []

        if len(bboxes) == 0:
            return []

        candidates = []
        for bbox, kps in zip(bboxes, kpss):
            try:
                embedding = self.shared_face_recognizer.recognizer.get_embedding(person_crop, kps)
                person_id, similarity = self.shared_face_recognizer.face_db.search(
                    embedding,
                    self.shared_face_recognizer.similarity_threshold,
                )
                x1, y1, x2, y2, _ = bbox.astype(np.int32)
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(person_crop.shape[1], x2)
                y2 = min(person_crop.shape[0], y2)
                face_crop = person_crop[y1:y2, x1:x2].copy()
                candidates.append({
                    'person_id': person_id,
                    'similarity': similarity,
                    'embedding': embedding,
                    'face_crop': face_crop,
                    'global_box': [x1 + crop_x1, y1 + crop_y1, x2 + crop_x1, y2 + crop_y1],
                })
            except Exception:
                continue
        return candidates

    def _detect_cell_phone_conf(self, person_crop):
        if person_crop is None or person_crop.size == 0:
            return 0.0

        try:
            phone_results = self.phone_detector.predict(
                person_crop,
                verbose=False,
                classes=[CELL_PHONE_CLASS_ID],
                conf=0.05,
                device=self.yolo_device,
            )
        except Exception:
            return 0.0

        if not phone_results:
            return 0.0

        boxes = phone_results[0].boxes
        if boxes is None or boxes.conf is None or len(boxes.conf) == 0:
            return 0.0

        return float(boxes.conf.max().item())

    def _get_state_key(self, person_id, track_id):
        if person_id not in (-1, None):
            return f"pid:{person_id}"
        if track_id is not None:
            return f"track:{track_id}"
        return None

    def _estimate_observation_quality(self, detected_kp_count, keypoints_global, posture, keypoint_type):
        quality = 0.1
        if keypoints_global is not None:
            quality += 0.4
        quality += (max(0, min(int(detected_kp_count), 17)) / 17.0) * 0.4
        if posture == "Unknown":
            quality -= 0.1
        if keypoint_type == "box_bottom":
            quality -= 0.15
        return float(np.clip(quality, 0.05, 1.0))

    def _adaptive_filter_params(self, quality_score):
        if quality_score >= 0.7:
            return 0.55, 0.12
        if quality_score >= 0.4:
            return 0.35, 0.08
        return 0.18, 0.04

    def _cleanup_stale_track_states(self, track_filter_states, current_time_sec, ttl_keep_state):
        stale_keys = [
            state_key
            for state_key, state in track_filter_states.items()
            if current_time_sec - state.get("last_t", current_time_sec) > ttl_keep_state
        ]
        for state_key in stale_keys:
            track_filter_states.pop(state_key, None)

    def _migrate_track_state(self, track_filter_states, old_key, new_key, current_time_sec, ttl_keep_state):
        if old_key is None or new_key is None or old_key == new_key:
            return
        old_state = track_filter_states.get(old_key)
        if old_state is None:
            return
        if current_time_sec - old_state.get("last_t", current_time_sec) > ttl_keep_state:
            return
        if new_key not in track_filter_states:
            track_filter_states[new_key] = old_state
        track_filter_states.pop(old_key, None)

    def _smooth_world_xy(
        self,
        track_filter_states,
        state_key,
        current_time_sec,
        observed_xy,
        quality_score,
        v_max_mmps,
        margin_mm,
        deadband_mm,
        outlier_confirm,
    ):
        if state_key is None or observed_xy is None:
            return observed_xy

        obs = np.array(observed_xy, dtype=np.float32)
        state = track_filter_states.get(state_key)
        if state is None:
            track_filter_states[state_key] = {
                "pos": obs,
                "vel": np.zeros(2, dtype=np.float32),
                "outlier_count": 0,
                "last_t": current_time_sec,
            }
            return float(obs[0]), float(obs[1])

        prev_pos = state["pos"]
        prev_vel = state["vel"]
        prev_t = state.get("last_t", current_time_sec)
        dt = max(1e-3, current_time_sec - prev_t)

        pred_pos = prev_pos + prev_vel * dt
        residual = obs - pred_pos
        residual_norm = float(np.linalg.norm(residual))
        gate_threshold = float(v_max_mmps * dt + margin_mm)
        outlier_count = int(state.get("outlier_count", 0))

        if residual_norm > gate_threshold:
            outlier_count += 1
            if outlier_count < outlier_confirm:
                used_obs = pred_pos
            elif residual_norm > 1e-6:
                used_obs = pred_pos + residual * (gate_threshold / residual_norm)
            else:
                used_obs = pred_pos
        else:
            outlier_count = 0
            used_obs = obs

        alpha, beta = self._adaptive_filter_params(quality_score)
        innovation = used_obs - pred_pos
        filtered_pos = pred_pos + alpha * innovation
        filtered_vel = prev_vel + (beta / dt) * innovation

        if float(np.linalg.norm(filtered_pos - prev_pos)) < deadband_mm:
            filtered_pos = prev_pos.copy()
            filtered_vel = prev_vel * 0.5

        state["pos"] = filtered_pos
        state["vel"] = filtered_vel
        state["outlier_count"] = outlier_count
        state["last_t"] = current_time_sec
        return float(filtered_pos[0]), float(filtered_pos[1])

    def _refresh_identity_store_if_needed(self, force=False):
        if self.shared_identity_store is None:
            return False

        if force:
            self.shared_identity_store.reload(verbose=True)
            with self.identity_refresh_lock:
                self.last_identity_refresh_check = time.time()
            print(">>> ReID identity 库已手动热更新。")
            return True

        now = time.time()
        with self.identity_refresh_lock:
            if now - self.last_identity_refresh_check < self.identity_refresh_interval:
                return False
            self.last_identity_refresh_check = now

        changed = self.shared_identity_store.refresh_if_changed(verbose=True)
        if changed:
            print(">>> 检测到 identity 库变化，已热更新 ReID gallery。")
        return changed

    def get_camera_params(self, camera_id):
        config_file = f"camera_params_{camera_id}.yaml"
        full_path = os.path.join(self.config_base_path, config_file)

        if not os.path.exists(full_path):
            raise CameraConfigError(f"相机配置文件不存在: {full_path}")

        try:
            camera_mtime = os.path.getmtime(full_path)
            order_mtime = os.path.getmtime(self.order_config_path)
        except OSError as e:
            raise CameraConfigError(f"读取相机配置文件失败 {camera_id}: {e}") from e

        with self.camera_params_lock:
            cached = self.camera_params_cache.get(camera_id)
            if (
                cached is not None
                and cached['config_path'] == full_path
                and cached['camera_mtime'] == camera_mtime
                and cached['order_mtime'] == order_mtime
            ):
                return cached['params']

        try:
            params = load_config(full_path, self.order_config_path)
            with self.camera_params_lock:
                self.camera_params_cache[camera_id] = {
                    'params': params,
                    'config_path': full_path,
                    'camera_mtime': camera_mtime,
                    'order_mtime': order_mtime,
                }
            return params
        except Exception as e:
            raise CameraConfigError(f"无法加载相机配置 {camera_id}: {e}") from e

    def get_camera_state(self, camera_id):
        with self.camera_states_lock:
            if camera_id not in self.camera_states:
                self.camera_states[camera_id] = {
                    # [新增] 为每个相机初始化独立的检测器
                    'detector': create_yolo_model(self.detector_path, task='detect'),
                    
                    'reidentifier': PersonReidentifier(
                        identity_folder=self.identity_folder,
                        known_similarity_threshold=0.9,
                        unknown_similarity_threshold=0.95,
                        pose_estimator=self.pose_estimator,
                        feature_extractor=self.shared_reid_extractor, # [修改] 传入共享模型
                        id_generator=self._generate_next_global_id,   # [新增] 传入全局
                        shared_identity_store=self.shared_identity_store,
                        shared_unknown_store=self.shared_unknown_store
                    ),
                    # [修改] video_cache 长度改为 16，适配新的 Uniformer 输入要求
                    'video_cache': defaultdict(lambda: deque(maxlen=16)),
                    'frame_counter': defaultdict(int),
                    'action_cache': {},
                    # [新增] 记录每个人的上一帧原始行为预测结果，用于连续比对
                    'action_pred_cache': {},
                    'uniformer_enabled': False,
                    # [新增] 人脸识别投票表: {track_id: {face_id: consecutive_count}}
                    'face_vote_table': {},
                    # [新增] 记录每个 track_id 上一帧识别到的 face_id，用于判定"连续"
                    'face_last_seen': {},
                    'track_filter_states': {}
                }

            return self.camera_states[camera_id]

    def detect_person_from_image(self, image, camera_id=None, 
                                 enable_face=True,
                                 enable_behavior=False,
                                 enable_uniformer=False,
                                 enable_positioning=True,
                                 enable_tracking=True,
                                 profiler=None):
        """
        处理单张图片的核心方法，包含检测、识别、行为分析等功能。
         - 参数说明：
            - image: 输入的图像数据，格式为 NumPy 数组（BGR）。
            - camera_id: 可选的相机ID，用于加载特定的相机配置和管理状态。
            - enable_face: 是否启用人脸识别功能。
            - enable_behavior: 是否启用行为识别功能。
            - enable_positioning: 是否启用姿态估计和定位功能。
            - enable_tracking: 是否启用跟踪功能（需要配合 ReID 使用）。
        """
        if profiler is None:
            profiler = RequestProfiler()

        if not camera_id:
            camera_id = "default"

        with profiler.section("4_State_And_Config"):
            self._refresh_identity_store_if_needed()
            self.shared_unknown_store.cleanup_stale()
            camera_params = self.get_camera_params(camera_id)

            state = self.get_camera_state(camera_id)
            uniformer_enabled = bool(enable_behavior and enable_uniformer)
            if state.get('uniformer_enabled', False) != uniformer_enabled:
                self._reset_uniformer_state(state)
                state['uniformer_enabled'] = uniformer_enabled

            action_model = self._get_action_model() if uniformer_enabled else None
            detector = state['detector']
            reidentifier = state['reidentifier']
            person_video_cache = state['video_cache']
            person_frame_counter = state['frame_counter']
            person_action_cache = state['action_cache']
            action_pred_cache = state['action_pred_cache']
            face_vote_table = state['face_vote_table']
            face_last_seen = state['face_last_seen']
            track_filter_states = state['track_filter_states']
            current_time_sec = time.time()

        frame = image
        persons_result = []

        with profiler.section("5_Person_Detection"):
            if enable_tracking:
                results = detector.track(
                    frame,
                    verbose=False,
                    classes=[0],
                    conf=0.5,
                    device=self.yolo_device,
                    persist=True,
                    tracker="bytetrack.yaml",
                )
            else:
                results = detector.predict(frame, verbose=False, classes=[0], conf=0.5, device=self.yolo_device)

        boxes = results[0].boxes.xyxy.cpu().numpy()
        has_person = len(boxes) > 0

        if not has_person:
            face_vote_table.clear()
            face_last_seen.clear()
            if enable_tracking:
                reidentifier.refresh_track_state([])
            self.shared_unknown_store.cleanup_stale()
            self._cleanup_stale_track_states(track_filter_states, current_time_sec, self.ttl_keep_state)
            return False, []

        track_ids = None
        if results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.int().cpu().tolist()

        assigned_ids = [-1] * len(boxes)
        assigned_confs = [0.0] * len(boxes)
        current_track_ids = track_ids if track_ids is not None else [-(i + 1) for i in range(len(boxes))]

        active_track_ids = set(current_track_ids)
        stale_vote_ids = [tid for tid in list(face_vote_table.keys()) if tid not in active_track_ids]
        for stale_track_id in stale_vote_ids:
            face_vote_table.pop(stale_track_id, None)
            face_last_seen.pop(stale_track_id, None)
        for stale_track_id in list(face_last_seen.keys()):
            if stale_track_id not in active_track_ids:
                face_last_seen.pop(stale_track_id, None)

        if enable_tracking and has_person:
            with profiler.section("6_Person_ReID"):
                try:
                    reid_ids, reid_confs = reidentifier.identify(frame, boxes, track_ids=current_track_ids)
                    if len(reid_ids) == len(boxes):
                        assigned_ids = reid_ids
                        assigned_confs = reid_confs
                    else:
                        print(f"[Warn] ReID count mismatch ({len(reid_ids)} vs {len(boxes)}). Using raw detections.")
                except Exception as e:
                    print(f"[Error] ReID identification error: {e}")

        for current_idx, (box, person_id, conf, track_id) in enumerate(zip(boxes, assigned_ids, assigned_confs, current_track_ids)):
            if isinstance(person_id, list):
                person_id = person_id[0] if len(person_id) > 0 else -1

            switch_from = reidentifier.get_switch_from(track_id) if enable_tracking else None
            x1, y1, x2, y2 = map(int, box)

            padding = 10
            crop_x1 = max(0, x1 - padding)
            crop_y1 = max(0, y1 - padding)
            crop_x2 = min(frame.shape[1], x2 + padding)
            crop_y2 = min(frame.shape[0], y2 + padding)
            base_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2].copy()

            action_label, action_conf = None, 0.0
            id_resource = "ReID"
            if uniformer_enabled and person_id != -1:
                with profiler.section("7_Action_Uniformer"):
                    action_crop = crop_and_pad(frame, box, margin_percent=50)
                    person_video_cache[person_id].append(action_crop)
                    person_frame_counter[person_id] += 1

                    if (len(person_video_cache[person_id]) == 16 and
                        person_frame_counter[person_id] % self.action_infer_interval == 0):
                        seq = list(person_video_cache[person_id])
                        input_tensor = preprocess_crops_for_video_cls(seq).to(self.device)
                        with torch.no_grad():
                            outputs = action_model(input_tensor)
                        pred_labels, pred_confs = postprocess(outputs)
                        person_action_cache[person_id] = (pred_labels[0], pred_confs[0])

            if uniformer_enabled and person_id in person_action_cache:
                current_pred_label, current_pred_conf = person_action_cache[person_id]
                current_pred_conf = float(current_pred_conf)
                threshold = self.ACTION_THRESHOLDS.get(current_pred_label, 0.8)
                if current_pred_conf < threshold:
                    current_effective_label = 'normal'
                else:
                    current_effective_label = current_pred_label

                last_effective_label = action_pred_cache.get(person_id, 'normal')
                if current_effective_label == 'normal':
                    action_label = 'normal'
                    action_conf = current_pred_conf
                elif current_effective_label == last_effective_label:
                    action_label = current_effective_label
                    action_conf = current_pred_conf
                else:
                    action_label = 'normal'
                    action_conf = current_pred_conf
                action_pred_cache[person_id] = current_effective_label

            if enable_behavior:
                with profiler.section("8_Phone_Detection"):
                    person_box_crop = frame[
                        max(0, y1):min(frame.shape[0], y2),
                        max(0, x1):min(frame.shape[1], x2),
                    ]
                    phone_conf = self._detect_cell_phone_conf(person_box_crop)
                    if phone_conf > self.PHONE_CONFIDENCE_THRESHOLD:
                        action_label = 'looking_at_phone'
                        action_conf = phone_conf

            world_coords = None
            detected_kp_count = 0
            kp_data = None
            keypoints_global = None
            posture = "Unknown"
            keypoint_type = "box_bottom"

            if enable_face or enable_positioning:
                with profiler.section("9_Pose_Estimation"):
                    pose_results = self.pose_estimator(base_crop.copy(), verbose=False, conf=0.7, device=self.yolo_device)
                    if pose_results and len(pose_results[0].keypoints.data) > 0:
                        kp_data = pose_results[0].keypoints.data[0].cpu().numpy()
                        detected_kp_count = int(np.sum(kp_data[:, 2] > 0.5))
                        keypoints_global = kp_data.copy()
                        keypoints_global[:, 0] += crop_x1
                        keypoints_global[:, 1] += crop_y1

            if enable_face:
                profiler.start("10_Face_Recognition")
                face_candidates = self._extract_face_candidates(base_crop.copy(), crop_x1, crop_y1)
                primary_face = face_candidates[0] if face_candidates else None
                detected_face_id = None
                unknown_face_candidate = None
                if primary_face is not None:
                    if primary_face['person_id'] != 'Unknown':
                        detected_face_id = primary_face['person_id']
                    else:
                        unknown_face_candidate = primary_face

                if detected_face_id is not None:
                    if str(person_id) == str(detected_face_id):
                        if enable_tracking:
                            reidentifier.update_identity(
                                track_id,
                                detected_face_id,
                                base_crop,
                                box=box,
                                all_boxes=boxes,
                                frame=frame,
                                crop_coords=(crop_x1, crop_y1, crop_x2, crop_y2),
                            )
                            if switch_from is None:
                                switch_from = reidentifier.get_switch_from(track_id)
                        person_id = detected_face_id
                        face_vote_table.pop(track_id, None)
                        face_last_seen.pop(track_id, None)
                    else:
                        last_face = face_last_seen.get(track_id)
                        if last_face == detected_face_id:
                            face_vote_table[track_id] = face_vote_table.get(track_id, 0) + 1
                        else:
                            face_vote_table[track_id] = 1

                        face_last_seen[track_id] = detected_face_id
                        if face_vote_table[track_id] >= self.FACE_CONFIRM_THRESHOLD:
                            print(f"[Face] 投票确认通过: Track {track_id} -> {detected_face_id} (连续 {face_vote_table[track_id]} 帧)")
                            previous_person_id = person_id
                            if enable_tracking:
                                reidentifier.update_identity(
                                    track_id,
                                    detected_face_id,
                                    base_crop,
                                    box=box,
                                    all_boxes=boxes,
                                    frame=frame,
                                    crop_coords=(crop_x1, crop_y1, crop_x2, crop_y2),
                                )
                            if self._is_temporary_identity(previous_person_id):
                                self.shared_unknown_store.release_entity(previous_person_id)
                            switch_from = str(previous_person_id) if previous_person_id not in (-1, None) else switch_from
                            person_id = detected_face_id
                            face_vote_table.pop(track_id, None)
                            face_last_seen.pop(track_id, None)
                    id_resource = "face"
                else:
                    face_vote_table.pop(track_id, None)
                    face_last_seen.pop(track_id, None)

                    if enable_tracking and self._is_temporary_identity(person_id):
                        self.shared_unknown_store.touch_entity(person_id)
                        strict_unknown_face_ready = self._is_strict_unknown_face_ready(kp_data)
                        if unknown_face_candidate is not None and strict_unknown_face_ready:
                            matched_unknown_id, _ = self.shared_unknown_store.search_face_embedding(unknown_face_candidate['embedding'])
                            if matched_unknown_id is not None:
                                if str(person_id) != str(matched_unknown_id):
                                    old_person_id = reidentifier.bind_track_identity(track_id, matched_unknown_id)
                                    if switch_from is None:
                                        switch_from = reidentifier.get_switch_from(track_id)
                                    if self._is_temporary_identity(old_person_id) and str(old_person_id) != str(matched_unknown_id):
                                        self.shared_unknown_store.release_if_empty(old_person_id)
                                    person_id = matched_unknown_id
                                self.shared_unknown_store.touch_entity(person_id)
                                id_resource = "face"
                            elif person_id == -1:
                                allocated_id = self._generate_next_global_id()
                                if allocated_id != -1:
                                    old_person_id = reidentifier.bind_track_identity(track_id, allocated_id)
                                    if switch_from is None:
                                        switch_from = reidentifier.get_switch_from(track_id)
                                    if self._is_temporary_identity(old_person_id) and old_person_id not in (-1, allocated_id):
                                        self.shared_unknown_store.release_if_empty(old_person_id)
                                    person_id = allocated_id
                                    self.shared_unknown_store.touch_entity(person_id)
                                    id_resource = "face"

                            if person_id != -1:
                                face_saved = self.shared_unknown_store.add_face_sample(
                                    person_id,
                                    unknown_face_candidate['face_crop'],
                                    unknown_face_candidate['embedding'],
                                )
                                # 未知身份图必须依赖本帧未知人脸样本成功保存，避免出现只存身份图不存人脸。
                                if face_saved:
                                    reidentifier.save_unknown_reid_sample(
                                        track_id,
                                        person_id,
                                        base_crop,
                                        box=box,
                                        frame_shape=frame.shape,
                                        all_boxes=boxes,
                                        current_idx=current_idx,
                                    )
                profiler.stop("10_Face_Recognition")

            if enable_positioning:
                profiler.start("11_World_Coord_Calc")
                if keypoints_global is not None:
                    posture, _ = classify_posture_with_verification(keypoints_global, camera_params, ratio_threshold=0.7)
                    world_coords, keypoint_type, _ = get_world_coords_from_pose(posture, keypoints_global, camera_params)
                else:
                    reference_point = ((x1 + x2) / 2, y2)
                    world_coords = image_to_world_plane(reference_point, camera_params, assumed_height=0.0)
                    keypoint_type = "box_bottom"

            state_key = self._get_state_key(person_id, track_id)
            if switch_from is not None and person_id not in (-1, None):
                self._migrate_track_state(
                    track_filter_states,
                    f"pid:{switch_from}",
                    f"pid:{person_id}",
                    current_time_sec,
                    self.ttl_keep_state,
                )
                state_key = f"pid:{person_id}"
            if person_id not in (-1, None) and track_id is not None:
                self._migrate_track_state(
                    track_filter_states,
                    f"track:{track_id}",
                    f"pid:{person_id}",
                    current_time_sec,
                    self.ttl_keep_state,
                )
                state_key = f"pid:{person_id}"

            if world_coords is not None:
                quality_score = self._estimate_observation_quality(
                    detected_kp_count,
                    keypoints_global,
                    posture,
                    keypoint_type,
                )
                smoothed_xy = self._smooth_world_xy(
                    track_filter_states,
                    state_key,
                    current_time_sec,
                    (float(world_coords[0]), float(world_coords[1])),
                    quality_score,
                    self.v_max_mmps,
                    self.margin_mm,
                    self.deadband_mm,
                    self.outlier_confirm,
                )
                if smoothed_xy is not None:
                    world_coords = (smoothed_xy[0], smoothed_xy[1])

            cad_coords = None
            if world_coords is not None:
                cad_coords = world_to_cad(world_coords, camera_params)
            if enable_positioning:
                profiler.stop("11_World_Coord_Calc")

            pid_str = str(person_id) if person_id != -1 else None 
            raw_track_id = None
            if enable_tracking and track_id is not None and track_id >= 0:
                raw_track_id = str(track_id)
            if switch_from is None and enable_tracking:
                switch_from = reidentifier.get_switch_from(track_id)
            w_coords_list = [0.0, 0.0, 0.0]
            if cad_coords is not None:
                w_coords_list = [float(cad_coords[0]), float(cad_coords[1]), 0.0]

            behavior_list = []
            if action_label:
                behavior_list.append({
                    "behavior_type": action_label,
                    "confidence": float(action_conf),
                    "duration": 0.0
                })

            x1_f, y1_f, x2_f, y2_f = float(x1), float(y1), float(x2), float(y2)
            x_mid = (x1_f + x2_f) / 2.0
            y_mid = (y1_f + y2_f) / 2.0
            bbox_anchor_points = {
                "top_left": [x1_f, y1_f],
                "top_center": [x_mid, y1_f],
                "top_right": [x2_f, y1_f],
                "middle_left": [x1_f, y_mid],
                "middle_center": [x_mid, y_mid],
                "middle_right": [x2_f, y_mid],
                "bottom_left": [x1_f, y2_f],
                "bottom_center": [x_mid, y2_f],
                "bottom_right": [x2_f, y2_f],
            }
            
            persons_result.append({
                "person_id": pid_str,
                "track_id": raw_track_id,
                "id_resource": id_resource,
                "switch_from": switch_from,
                "conf": float(conf),
                "world_coordinates": w_coords_list,
                "behavior_events": behavior_list,
                "bounding_box": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "bbox_anchor_points": bbox_anchor_points,
                "keypoint_count": int(detected_kp_count)
            })

        self._cleanup_stale_track_states(track_filter_states, current_time_sec, self.ttl_keep_state)
        return len(persons_result) > 0, persons_result

    def reload_library(self):
        """重载人脸库并热更新 ReID 库，保留在线轨迹状态"""
        try:
            self.shared_face_recognizer = FaceRecognizer(
                face_gallery_path='faceImage',
                scrfd_model_path='weights/det_10g.onnx',
                adaface_model_path='weights/adaface_ir50_ms1mv2.ckpt',
                architecture='ir_50',
                similarity_threshold=0.45,
                detection_threshold=0.65,
                db_path='./database/face_database'
            )
            self.image_face_recognizer = FaceRecognizer(
                face_gallery_path='faceImage',
                scrfd_model_path='weights/det_10g.onnx',
                adaface_model_path='weights/adaface_ir50_ms1mv2.ckpt',
                architecture='ir_50',
                similarity_threshold=0.35,
                detection_threshold=0.3,
                db_path='./database/face_database'
            )
            self._refresh_identity_store_if_needed(force=True)
            self.shared_unknown_store.clear_all()
            return True
        except Exception as e:
            print(f"重载失败: {e}")
            return False

    def verify_face_from_image(self, image):
        """
        Run face verification on a single image.

        Returns:
            tuple: (person_id, face_detected)
        """
        if image is None or image.size == 0:
            return None, False

        return self.image_face_recognizer.verify_person_id(image)
