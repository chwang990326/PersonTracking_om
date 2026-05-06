import numpy as np
import math
from models.geometry import image_to_world_and_back_to_pixel

# --- 1. 辅助函数 ---

def calculate_angle(p1, p2, p3):
    """计算由p1, p2, p3三点构成的角度,p2为顶点。返回角度和三个点的最低置信度."""
    if p1 is None or p2 is None or p3 is None:
        return None, 0.0
    
    vec1 = p1['coords'] - p2['coords']
    vec2 = p3['coords'] - p2['coords']
    
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    
    if norm_vec1 == 0 or norm_vec2 == 0:
        return None, 0.0
        
    cosine_angle = dot_product / (norm_vec1 * norm_vec2)
    angle_rad = math.acos(max(-1.0, min(1.0, cosine_angle)))
    angle_deg = math.degrees(angle_rad)
    
    confidence = min(p1['conf'], p2['conf'], p3['conf'])
    return angle_deg, confidence

def get_keypoint(keypoints, index):
    """获取单个关键点,如果置信度低则返回None。"""
    if keypoints[index, 2] > 0.3:
        return {'coords': keypoints[index, :2], 'conf': keypoints[index, 2]}
    return None

# --- 2. 核心分类逻辑 ---

def classify_posture_with_verification(keypoints, camera_params, ratio_threshold=0.7):
    """
    姿态分类 + 站立验证的集成函数
    
    返回:
        posture: 最终姿态
        verification_info: 验证信息字典(如果进行了验证)
    """
    # 先进行基础姿态分类
    initial_posture = classify_posture(keypoints)
    
    # 如果初步判定为站立,进行二次验证
    if initial_posture == 'Standing':
        verified_posture, verification_info = verify_standing_posture(
            keypoints, camera_params, ratio_threshold
        )
        return verified_posture, verification_info
    
    # 其他姿态不需要验证
    return initial_posture, {}


def classify_posture(keypoints):
    """
    使用分层决策和特征置信度对姿态进行更精确的分类。
    新增 "Bending" 状态,并优化遮挡处理。
    """
    # --- 准备阶段: 获取所有需要的关键点 ---
    kps = {i: get_keypoint(keypoints, i) for i in range(17)}
    
    # --- 决策层 1: 躺姿判断 (最优先) ---
    visible_kps_coords = np.array([kp['coords'] for kp in kps.values() if kp is not None])
    if len(visible_kps_coords) < 4:
        return "Unknown"

    min_x, min_y = np.min(visible_kps_coords, axis=0)
    max_x, max_y = np.max(visible_kps_coords, axis=0)
    bbox_height = max_y - min_y
    bbox_width = max_x - min_x

    if bbox_height < 1e-5: 
        return "Lying Down"
    aspect_ratio = bbox_width / bbox_height
    if aspect_ratio > 1.5:
        return "Lying Down"

    # --- 决策层 2: 基于腿部角度的核心判断 ---
    left_knee_angle, lk_conf = calculate_angle(kps[11], kps[13], kps[15])
    right_knee_angle, rk_conf = calculate_angle(kps[12], kps[14], kps[16])
    
    left_hip_angle, lh_conf = calculate_angle(kps[5], kps[11], kps[13])
    right_hip_angle, rh_conf = calculate_angle(kps[6], kps[12], kps[14])

    # 计算平均膝关节角度和其置信度
    if left_knee_angle is not None and right_knee_angle is not None:
        avg_knee_angle = (left_knee_angle + right_knee_angle) / 2
        knee_conf = (lk_conf + rk_conf) / 2
    elif left_knee_angle is not None:
        avg_knee_angle = left_knee_angle
        knee_conf = lk_conf
    elif right_knee_angle is not None:
        avg_knee_angle = right_knee_angle
        knee_conf = rk_conf
    else:
        avg_knee_angle = None
        knee_conf = 0.0

    # 计算平均髋关节角度和其置信度
    if left_hip_angle is not None and right_hip_angle is not None:
        avg_hip_angle = (left_hip_angle + right_hip_angle) / 2
        hip_conf = (lh_conf + rh_conf) / 2
    elif left_hip_angle is not None:
        avg_hip_angle = left_hip_angle
        hip_conf = lh_conf
    elif right_hip_angle is not None:
        avg_hip_angle = right_hip_angle
        hip_conf = rh_conf
    else:
        avg_hip_angle = None
        hip_conf = 0.0

    # 如果腿部特征可信,则进行精确判断
    if knee_conf > 0.3:
        if avg_knee_angle > 130:
            return "Standing"
        
        if avg_knee_angle < 100:
            is_hip_acute = hip_conf > 0.3 and avg_hip_angle < 75
            is_knee_acute = knee_conf > 0.3 and avg_knee_angle < 75

            hip_y = (kps[11]['coords'][1] + kps[12]['coords'][1]) / 2 if kps[11] and kps[12] else None
            ankle_y = (kps[15]['coords'][1] + kps[16]['coords'][1]) / 2 if kps[15] and kps[16] else None
            is_hip_close_to_ankle = False
            if hip_y and ankle_y and bbox_height > 0:
                if hip_y > ankle_y - (bbox_height * 0.1):
                    is_hip_close_to_ankle = True

            if is_hip_acute and is_knee_acute:
                return "Squatting"
            
            if is_hip_close_to_ankle and is_knee_acute:
                return "Squatting"

            return "Sitting"

    # --- 决策层 3: 遮挡回退逻辑 (腿部不可见) ---
    shoulder_center = (kps[5]['coords'] + kps[6]['coords']) / 2 if kps[5] and kps[6] else None
    hip_center = (kps[11]['coords'] + kps[12]['coords']) / 2 if kps[11] and kps[12] else None

    if shoulder_center is not None and hip_center is not None:
        torso_vec = shoulder_center - hip_center
        torso_angle_rad = np.arctan2(abs(torso_vec[0]), abs(torso_vec[1]))
        torso_angle_deg = np.degrees(torso_angle_rad)
        
        if torso_angle_deg > 60:
            return "Bending" # 弯腰特征很明显，保留
        # 【修改】：删除了 < 30 返回 Sitting 的逻辑，让直立的半身人进入下方的 BBox 比例判断

    # --- 决策层 4: 最终回退逻辑 (根据整体框的细长程度判断) ---
    if bbox_height / bbox_width > 1.4:
        return "Standing" # 竖长的框判为站立，并会在后续触发 3D 真实高度二次验证
    else:
        return "Sitting"  # 方正/扁平的框判为坐着


def verify_standing_posture(keypoints, camera_params, ratio_threshold=0.7):
    """
    验证站立姿态
    
    返回:
        verified_posture: 验证后的姿态 (不再根据比例降级)
        verification_info: 验证信息字典
    """
    # 获取头部关键点 (鼻子,索引0)
    head_kp = keypoints[0]
    if head_kp[2] <= 0.5:
        return 'Standing', {}  # 修复：明确返回两个值
    head_pixel = head_kp[:2]
    
    # 获取脚部关键点
    left_ankle = keypoints[15]
    right_ankle = keypoints[16]
    
    ankle_kp = None
    if left_ankle[2] > 0.5 and right_ankle[2] > 0.5:
        ankle_kp = (left_ankle[:2] + right_ankle[:2]) / 2
    elif left_ankle[2] > 0.5:
        ankle_kp = left_ankle[:2]
    elif right_ankle[2] > 0.5:
        ankle_kp = right_ankle[:2]
    else:
        return 'Standing', {}  # 修复：明确返回两个值
    
    # 计算预测头部像素坐标
    predicted_head_pixel = image_to_world_and_back_to_pixel(
        ankle_kp, camera_params, assumed_height=-1700
    )
    
    if predicted_head_pixel is None:
        return 'Standing', {}  # 修复：明确返回两个值
    
    # 计算距离和比例
    head_foot_distance = np.linalg.norm(head_pixel - ankle_kp)
    pred_foot_distance = np.linalg.norm(predicted_head_pixel - ankle_kp)
    
    if pred_foot_distance == 0:
        return 'Standing', {}  # 修复：明确返回两个值
    
    ratio = head_foot_distance / pred_foot_distance
    
    # 构建验证信息 (保留了计算过程供你参考和输出)
    verification_info = {
        'original_posture': 'Standing',
        'ratio': float(ratio),
        'ratio_threshold': ratio_threshold,
        'head_foot_distance': float(head_foot_distance),
        'pred_foot_distance': float(pred_foot_distance),
        'head_pixel_x': float(head_pixel[0]),
        'head_pixel_y': float(head_pixel[1]),
        'predicted_head_pixel_x': float(predicted_head_pixel[0]),
        'predicted_head_pixel_y': float(predicted_head_pixel[1])
    }
    
    # === 删除降级判定，直接保持 Standing ===
    verified_posture = 'Standing'
    
    verification_info['verified_posture'] = verified_posture
    
    return verified_posture, verification_info