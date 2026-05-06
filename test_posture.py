import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from tqdm import tqdm

from models.loader import load_config, get_video_streams
from models.geometry import (
    get_world_coords_from_pose, 
    draw_annotations, 
    image_to_world_plane, 
    world_to_cad
)
from models.posture_classifier import classify_posture_with_verification

# --- 模型路径配置 ---
DETECTOR_ONNX_PATH = 'weights/yolo11s.onnx'
POSE_ONNX_PATH = 'weights/yolo11s-pose.onnx'

def main():
    # --- 1. 基础配置 ---
    POSTURE_METHOD = 'traditional'
    CONFIG_PATH = 'config/camera_params_207.yaml'
    ORDER_CONFIG_PATH = 'config/caculate_order.yaml'
    VIDEO_IN_PATH = 'video/NVR_ch3_main_wang_climing.mp4'  # 请替换为您需要测试的视频
    
    os.makedirs('results', exist_ok=True)
    VIDEO_OUT_PATH = 'results/test_posture_head_offset.mp4'
    RATIO_THRESHOLD = 0.7

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"正在加载配置和模型... 使用设备: {device}")
    
    # 加载相机参数
    camera_params = load_config(CONFIG_PATH, ORDER_CONFIG_PATH)
    
    # 加载 YOLO 检测与姿态模型
    person_detector = YOLO(DETECTOR_ONNX_PATH, task='detect') 
    pose_estimator = YOLO(POSE_ONNX_PATH, task='pose')

    # --- 防抖与跳变分析缓存 ---
    # CAD 坐标缓存，用于计算轨迹跳变距离 (对齐 service.py)
    prev_coords_cache = {}
    
    # [新增] 头部关键点缓存 (鼻子坐标)，用于计算像素偏移量: {track_id: (nose_x, nose_y)}
    prev_head_cache = {}

    # --- 打开视频流 ---
    cap, writer = get_video_streams(VIDEO_IN_PATH, VIDEO_OUT_PATH)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 0

    print("开始处理视频... (专注姿态估计与头部跳变分析)")
    
    with tqdm(total=total_frames, desc="处理进度") as pbar:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # 1. 人体检测与追踪 (使用 ByteTrack)
            results = person_detector.track(
                frame, verbose=False, classes=[0], conf=0.5, persist=True, tracker="bytetrack.yaml"
            )
            
            boxes = results[0].boxes.xyxy.cpu().numpy()
            
            # 解析 track_ids
            track_ids = []
            if results[0].boxes.id is not None:
                track_ids = results[0].boxes.id.int().cpu().tolist()
            else:
                # 兜底：如果没有 track_id，分配负数临时ID
                track_ids = [-(i+1) for i in range(len(boxes))]

            # 清理离开画面的追踪ID对应的缓存，防止内存泄漏
            active_track_ids = set(track_ids)
            stale_ids = [tid for tid in prev_coords_cache.keys() if tid not in active_track_ids]
            for stale_id in stale_ids:
                del prev_coords_cache[stale_id]
                # [新增] 同步清理过期目标的头部缓存
                if stale_id in prev_head_cache:
                    del prev_head_cache[stale_id]

            # 2. 遍历每个人物进行姿态和坐标分析
            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = map(int, box)
                
                # 裁剪图像 (带10像素Padding，与 service.py 严格对齐)
                padding = 10
                crop_x1 = max(0, x1 - padding)
                crop_y1 = max(0, y1 - padding)
                crop_x2 = min(frame.shape[1], x2 + padding)
                crop_y2 = min(frame.shape[0], y2 + padding)
                person_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                
                # --- 姿态估计 ---
                keypoints = None
                current_nose_kpt = None # [新增] 用于保存当前帧的有效鼻子坐标
                posture = "Unknown"
                world_coords, keypoint_type, assumed_z = None, None, None
                cad_coords = None
                verification_info = {}

                pose_results = pose_estimator(person_crop.copy(), verbose=False, conf=0.7)

                if pose_results and len(pose_results[0].keypoints.data) > 0:
                    kps_data_crop = pose_results[0].keypoints.data[0].cpu().numpy()
                    
                    # [核心修改 1] 提取当前帧有效的头部关键点像素坐标 (像素坐标无需 Z 轴信息)
                    # COCO index 0 为鼻子
                    if kps_data_crop.shape[0] > 0 and kps_data_crop[0, 2] > 0.5:
                        # 坐标还原到原图尺寸
                        current_nose_kpt = (int(kps_data_crop[0, 0] + crop_x1), int(kps_data_crop[0, 1] + crop_y1))
                    
                    keypoints = kps_data_crop.copy()
                    # 坐标还原到原图 (用于 3D 投影逻辑)
                    keypoints[:, 0] += crop_x1
                    keypoints[:, 1] += crop_y1

                    # 姿态判定
                    posture, verification_info = classify_posture_with_verification(
                        keypoints, camera_params, RATIO_THRESHOLD
                    )
                    # 坐标转换
                    world_coords, keypoint_type, assumed_z = get_world_coords_from_pose(
                        posture, keypoints, camera_params
                    )
                else:
                    # 兜底逻辑：找不到关键点时使用底部中心
                    reference_point = ((x1 + x2) / 2, y2)
                    assumed_z = 0.0
                    world_coords = image_to_world_plane(
                        reference_point, camera_params, assumed_height=assumed_z
                    )
                    keypoint_type = "box_bottom"

                # 转换为 CAD 坐标系
                if world_coords is not None:
                    cad_coords = world_to_cad(world_coords, camera_params)

                # --- 核心分析 1：计算 CAD 坐标轨迹跳变距离 (保持原逻辑) ---
                cad_jump_distance = 0.0
                if cad_coords is not None and track_id in prev_coords_cache:
                    prev_cad = prev_coords_cache[track_id]
                    # 计算欧氏距离 (CAD坐标系下单位通常是 mm 或 m)
                    cad_jump_distance = np.sqrt((cad_coords[0] - prev_cad[0])**2 + (cad_coords[1] - prev_cad[1])**2)
                
                # --- [核心修改 2]：计算头部关键点像素偏移量 ---
                head_pixel_offset = 0.0
                # 如果当前目标头部检测成功，并且上一帧有记录
                if current_nose_kpt is not None and track_id in prev_head_cache:
                    prev_head = prev_head_cache[track_id]
                    # 计算 2D 像素平面的欧氏距离
                    head_pixel_offset = np.sqrt((current_nose_kpt[0] - prev_head[0])**2 + (current_nose_kpt[1] - prev_head[1])**2)
                
                # --- [核心修改 3] 更新上一帧的坐标记忆缓存 ---
                # 更新 CAD 坐标
                if cad_coords is not None:
                    prev_coords_cache[track_id] = cad_coords
                
                # 更新头部鼻子坐标缓存
                if current_nose_kpt is not None:
                    prev_head_cache[track_id] = current_nose_kpt

                # --- 渲染绘制 (增加像素偏移量显示) ---
                # 1. 调用通用的绘画函数 (空数据传入 behavior/face/ReID)
                draw_annotations(
                    frame, 
                    person_id=str(track_id), # 测试脚本直接显示 track_id
                    box=box, 
                    cad_coords=cad_coords, 
                    keypoints=keypoints, 
                    posture=posture, 
                    posture_method=POSTURE_METHOD, 
                    keypoint_type=keypoint_type, 
                    assumed_z=assumed_z,
                    verification_info=verification_info,
                    face_boxes=[], face_ids=[], behavior_events=[], reid_confidence=1.0
                )
                
                # 动态计算绘制 Y 坐标，防止文本重叠和超出顶部
                current_y = int(max(20, y1 - 10))
                
                # [核心修改 4] 绘制 "像素偏移量" 标签在上方 (使用绿色突出显示)
                if head_pixel_offset > 0:
                    text_color = (0, 255, 0) # 绿色
                    head_text = f"H.Offset: {head_pixel_offset:.1f}px"
                    cv2.putText(
                        frame, head_text, 
                        (int(x1), current_y), # 画在原框顶部
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2, cv2.LINE_AA
                    )
                    current_y -= 22 # 文本上移

                # 2. 在其上方绘制 "轨迹跳变距离" (原来黄/红色的 Jump)
                if cad_jump_distance > 0:
                    text_color = (0, 0, 255) if cad_jump_distance > 500 else (0, 255, 255) # 距离>500标红，否则标黄
                    jump_text = f"Jump: {cad_jump_distance:.1f}"
                    cv2.putText(
                        frame, jump_text, 
                        (int(x1), current_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2, cv2.LINE_AA
                    )

            # 写入结果帧
            writer.write(frame)
            frame_idx += 1
            pbar.update(1)

    print("\n处理完成, 正在释放资源...")
    cap.release()
    writer.release()
    print(f"✅ 姿态与坐标分析视频已保存至: {VIDEO_OUT_PATH}")

if __name__ == '__main__':
    main()