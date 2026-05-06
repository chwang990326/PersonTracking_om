import os
import json
import cv2
import numpy as np
import torch
from collections import defaultdict, deque
from tqdm import tqdm
from models.loader import load_config, get_video_streams
from models.geometry import (
    get_world_coords_from_pose, 
    draw_annotations, 
    image_to_world_plane, 
    world_to_cad,
    create_person_info
)
from models.posture_classifier import classify_posture_with_verification
from models.personReID import PersonReidentifier
from models.face import FaceRecognizer
from models.ascend_backend import is_om_path, resolve_model_path
from models.ascend_yolo import create_yolo_model
# --- 新增导入 ---
# from utils.get_model import getmodel
# from models.actionclassifier import preprocess_crops_for_video_cls, postprocess

DETECTOR_ONNX_PATH = resolve_model_path('weights/yolo26x.om', 'weights/yolo26x.onnx')
POSE_ONNX_PATH = resolve_model_path('weights/yolo26s-pose.om', 'weights/yolo26s-pose.onnx')

def main():
    # --- 1. 配置和初始化 ---
    POSTURE_METHOD = 'traditional'
    CONFIG_PATH = 'config/camera_params_207.yaml'
    ORDER_CONFIG_PATH = 'config/caculate_order.yaml'
    VIDEO_IN_PATH = 'video/0326测试_5fps.mp4'
    
    os.makedirs('results', exist_ok=True)
    VIDEO_OUT_PATH = 'results/优化人脸识别_出参测试.mp4'
    JSON_OUT_PATH = 'results/location_207.json'
    RATIO_THRESHOLD = 0.7

    device = torch.device('cpu' if any(is_om_path(path) for path in (DETECTOR_ONNX_PATH, POSE_ONNX_PATH)) else ('cuda' if torch.cuda.is_available() else 'cpu'))

    print("正在加载配置和模型...")
    camera_params = load_config(CONFIG_PATH, ORDER_CONFIG_PATH)
    
    # 加载YOLO模型
    # person_detector = YOLO('weights/yolo11s.pt')
    # pose_estimator = YOLO('weights/yolo11s-pose.pt')

    person_detector = create_yolo_model(DETECTOR_ONNX_PATH, task='detect') 
    pose_estimator = create_yolo_model(POSE_ONNX_PATH, task='pose')
    
    # 注意：不再需要YOLO人脸检测器，SCRFD已内置在FaceRecognizer中

    # --- 初始化自定义ReID系统 ---
    face_recognizer = FaceRecognizer(
            face_gallery_path='faceImage',
            scrfd_model_path='weights/det_10g.onnx',
            adaface_model_path='weights/adaface_ir50_ms1mv2.ckpt',
            architecture='ir_50',
            similarity_threshold=0.45,
            detection_threshold=0.7,
            db_path='./database/face_database'
        )

    reidentifier = PersonReidentifier(
        identity_folder='identity',
        similarity_threshold=0.9,
        pose_estimator = pose_estimator
        # ,face_recognizer=face_recognizer
    )
    


    # --- 初始化行为识别模块 ---
    # print("正在加载行为识别模型...")
    
    # action_model = getmodel('config/bestaction19089092.pt').eval().to(device)
    
    # # 行为识别缓存
    # person_video_cache = defaultdict(lambda: deque(maxlen=8))
    # person_frame_counter = defaultdict(int)
    # person_action_cache = {}
    # action_infer_interval = 4
    # ------------------------

    cap, writer = get_video_streams(VIDEO_IN_PATH, VIDEO_OUT_PATH)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    all_persons_data = []
    frame_idx = 0

    print(f"开始处理视频... 使用自定义ReID方案进行人员识别。")
    
    with tqdm(total=total_frames, desc="处理进度") as pbar:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # 1. 人体检测 (修改为 track 模式)
            # persist=True 保持跨帧追踪，tracker 指定配置文件
            person_results = person_detector.track(frame, verbose=False, classes=[0], conf=0.5, persist=True, tracker="bytetrack.yaml")
            
            frame_data = {'frame_id': frame_idx, 'persons': []}
            
            # 获取检测框和追踪ID
            if person_results[0].boxes.id is not None:
                boxes = person_results[0].boxes.xyxy.cpu().numpy()
                track_ids = person_results[0].boxes.id.int().cpu().tolist()
            else:
                boxes = []
                track_ids = []
            
            # 2. ReID 识别 (传入 track_ids)
            assigned_ids = []
            assigned_confs = [] 
            if len(boxes) > 0:
                # [修改] 传入 track_ids
                assigned_ids, assigned_confs = reidentifier.identify(frame, boxes, track_ids) 
            
            # ============================================================
            # 步骤A: 收集所有人的裁剪图像并更新缓存 (行为识别准备)
            # ============================================================
            person_metadata = []
            
            # [修改] 增加 assigned_confs 到 zip
            # [修改] 增加 track_ids 到 zip，以便后续传给 update_identity
            # 注意：如果 track_ids 为空，zip 会自动处理
            current_track_ids = track_ids if len(track_ids) > 0 else [-1] * len(boxes)
            
            for box, person_id, conf, track_id in zip(boxes, assigned_ids, assigned_confs, current_track_ids):
                x1, y1, x2, y2 = map(int, box)
                padding = 10
                crop_x1 = max(0, x1 - padding)
                crop_y1 = max(0, y1 - padding)
                crop_x2 = min(frame.shape[1], x2 + padding)
                crop_y2 = min(frame.shape[0], y2 + padding)
                person_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2].copy()
                
                # 更新视频缓存
                # if person_id != -1:
                #     person_video_cache[person_id].append(person_crop)
                #     person_frame_counter[person_id] += 1
                
                person_metadata.append({
                    'person_id': person_id,
                    'track_id': track_id, # [新增] 保存 track_id
                    'conf': conf, 
                    'box': box,
                    'crop_coords': (crop_x1, crop_y1, crop_x2, crop_y2),
                    'person_crop': person_crop
                })

            # ============================================================
            # 步骤B: 批量行为识别
            # ============================================================
            # action_results = {}
            # batch_person_ids = []
            # batch_sequences = []
            
            # for meta in person_metadata:
            #     pid = meta['person_id']
            #     # 只有当缓存满8帧且达到推理间隔时才进行推理
            #     if (pid != -1 and 
            #         len(person_video_cache[pid]) == 8 and
            #         person_frame_counter[pid] % action_infer_interval == 0):
            #         
            #         batch_person_ids.append(pid)
            #         batch_sequences.append(list(person_video_cache[pid]))
            
            # if len(batch_sequences) > 0:
            #     batch_crops = [preprocess_crops_for_video_cls(seq) for seq in batch_sequences]
            #     batch_input = torch.cat(batch_crops, dim=0).to(device)
            #     
            #     with torch.no_grad():
            #         batch_outputs = action_model(batch_input)
            #     
            #     pred_labels, pred_confs = postprocess(batch_outputs)
            #     
            #     for pid, label, conf in zip(batch_person_ids, pred_labels, pred_confs):
            #         action_results[pid] = (label, conf)
            #         person_action_cache[pid] = (label, conf) # 更新缓存

            # ============================================================
            # 步骤C: 处理每个被识别出的人物 (Face, Pose, Geometry, Draw)
            # ============================================================
            for meta in person_metadata:
                person_id = meta['person_id']
                track_id = meta['track_id'] # [新增] 获取 track_id
                conf = meta.get('conf', 0.0) # [新增] 获取置信度
                box = meta['box']
                crop_x1, crop_y1, crop_x2, crop_y2 = meta['crop_coords']
                person_crop = meta['person_crop']
                x1, y1, x2, y2 = map(int, box)

                # 获取行为识别结果 (优先取当前帧推理结果，否则取缓存)
                action_label = None
                action_conf = 0.0
                # if person_id in action_results:
                #     action_label, action_conf = action_results[person_id]
                # elif person_id in person_action_cache:
                #     action_label, action_conf = person_action_cache[person_id]
                
                behavior_events = []
                # if action_label:
                #     behavior_events.append({
                #         "behavior_type": action_label,
                #         "confidence": float(action_conf),
                #         "duration": 0.0
                #     })

                # 人脸识别
                face_boxes, face_ids = face_recognizer.detect_and_recognize(
                    None, frame, person_crop.copy(), crop_x1, crop_y1
                )

                # 身份修正逻辑
                if len(face_ids) > 0:
                    face_id = face_ids[0]
                    if face_id != 'Unknown':
                        # [修改] 传入 track_id 进行精确更新
                        if track_id != -1:
                            reidentifier.update_identity(
                                track_id,
                                face_id,
                                person_crop,
                                box=box,
                                all_boxes=boxes,
                                frame=frame,
                                crop_coords=(crop_x1, crop_y1, crop_x2, crop_y2),
                            )
                            person_id = face_id # 更新当前帧显示的ID

                # 姿态估计
                keypoints = None
                posture = "Unknown"
                world_coords, keypoint_type, assumed_z = None, None, None
                cad_coords = None
                verification_info = {}

                pose_results = pose_estimator(person_crop.copy(), verbose=False, conf=0.7)

                if pose_results and len(pose_results[0].keypoints.data) > 0:
                    keypoints_data_crop = pose_results[0].keypoints.data[0].cpu().numpy()
                    keypoints = keypoints_data_crop.copy()
                    keypoints[:, 0] += crop_x1
                    keypoints[:, 1] += crop_y1

                    posture, verification_info = classify_posture_with_verification(
                        keypoints, camera_params, RATIO_THRESHOLD
                    )
                    world_coords, keypoint_type, assumed_z = get_world_coords_from_pose(
                        posture, keypoints, camera_params
                    )
                else:
                    reference_point = ((x1 + x2) / 2, y2)
                    assumed_z = 0.0
                    world_coords = image_to_world_plane(
                        reference_point, camera_params, assumed_height=assumed_z
                    )
                    keypoint_type = "box_bottom"

                cad_coords = world_to_cad(world_coords, camera_params)
                
                # 创建信息并绘图
                person_info = create_person_info(
                    person_id=person_id,
                    posture=posture,
                    posture_method=POSTURE_METHOD,
                    keypoints=keypoints,
                    world_coords=world_coords,
                    keypoint_type=keypoint_type,
                    assumed_z=assumed_z,
                    cad_coords=cad_coords,
                    verification_info=verification_info,
                    face_boxes=face_boxes,
                    face_ids=face_ids,
                    behavior_events=behavior_events,
                    reid_confidence=conf # [新增] 传入置信度
                )
                frame_data['persons'].append(person_info)
                
                draw_annotations(
                    frame, person_id, box, cad_coords, keypoints, 
                    posture, POSTURE_METHOD, keypoint_type, assumed_z,
                    verification_info=verification_info,
                    face_boxes=face_boxes,
                    face_ids=face_ids,
                    behavior_events=behavior_events,
                    reid_confidence=conf # [新增] 传入置信度
                )

            all_persons_data.append(frame_data)
            writer.write(frame)
            frame_idx += 1
            pbar.update(1)

    # --- 3. 保存结果 ---
    print("处理完成,正在保存结果...")
    
    # 直接保存所有帧的数据
    with open(JSON_OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(all_persons_data, f, indent=4, ensure_ascii=False)

    cap.release()
    writer.release()
    
    print(f"\n标注视频已保存至: {VIDEO_OUT_PATH}")
    print(f"坐标数据已保存至: {JSON_OUT_PATH}")

if __name__ == '__main__':
    main()
