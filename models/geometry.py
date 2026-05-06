import numpy as np
import cv2

# 定义COCO关键点连接关系
# 每个元组代表一条需要连接的线，元组内的数字是关键点的索引
SKELETON_CONNECTIONS = [
    # 头部
    (0, 1), (0, 2), (1, 3), (2, 4),
    # 躯干
    (5, 6), (5, 11), (6, 12), (11, 12),
    # 左臂
    (5, 7), (7, 9),
    # 右臂
    (6, 8), (8, 10),
    # 左腿
    (11, 13), (13, 15),
    # 右腿
    (12, 14), (14, 16)
]

# 定义一个颜色列表用于绘制不同骨架
SKELETON_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), 
    (255, 255, 0), (255, 0, 255), (0, 255, 255)
]

def get_color_index(person_id):
    """辅助函数：将任意类型的ID转换为用于选择颜色的整数索引"""
    if isinstance(person_id, int):
        return person_id
    try:
        # 尝试转换数字字符串 "1" -> 1
        return int(person_id)
    except (ValueError, TypeError):
        # 对于 "Unknown", "LiSi" 等字符串，使用哈希值
        return abs(hash(str(person_id)))

def draw_skeleton_and_keypoints(frame, keypoints, person_id):
    """在图像上绘制单个人的骨架和关键点。"""
    # 根据person_id选择一个固定的颜色，以区分不同的人
    # 修改：使用辅助函数处理非整数ID
    color_idx = get_color_index(person_id)
    color = SKELETON_COLORS[color_idx % len(SKELETON_COLORS)]
    
    # 绘制关键点连接线（骨架）
    for connection in SKELETON_CONNECTIONS:
        kp_idx1, kp_idx2 = connection
        # 确保两个关键点都可见（置信度 > 0.5）
        if keypoints[kp_idx1, 2] > 0.5 and keypoints[kp_idx2, 2] > 0.5:
            pt1 = tuple(map(int, keypoints[kp_idx1, :2]))
            pt2 = tuple(map(int, keypoints[kp_idx2, :2]))
            cv2.line(frame, pt1, pt2, color, 2)

    # 绘制关键点圆圈
    for i in range(keypoints.shape[0]):
        if keypoints[i, 2] > 0.5: # 只绘制可见的关键点
            center = tuple(map(int, keypoints[i, :2]))
            cv2.circle(frame, center, 4, color, -1) # -1表示实心圆

def get_world_coords_from_pose(posture, keypoints, camera_params):
    """
    根据姿态和可用的关键点，动态选择最佳参考点和Z高度来计算世界坐标。
    返回: (world_coords, keypoint_type, assumed_z) 或 (None, None, None)
    """
    mapping_rules = camera_params.get('posture_height_mapping', {})
    
    # 获取当前姿态对应的规则列表，如果姿态未知，则使用 "Unknown" 的规则
    rules = mapping_rules.get(posture, mapping_rules.get("Unknown", []))

    for rule in rules:
        rule_type = rule['type']
        assumed_z = rule['z']
        indices = rule['indices']
        
        # 提取该规则对应的所有可见关键点
        visible_points = []
        for idx in indices:
            if keypoints[idx, 2] > 0.5: # 使用较高的置信度阈值
                visible_points.append(keypoints[idx, :2])
        
        # 如果找到了至少一个可见的关键点
        if visible_points:
            # 计算这些点的平均像素坐标作为参考点
            reference_point = np.mean(visible_points, axis=0)
            
            # (修改) 使用这个参考点和对应的z高度(assumed_height)来计算世界坐标
            world_coords = image_to_world_plane(reference_point, camera_params, assumed_height=assumed_z)
            
            if world_coords is not None:
                # 成功计算，返回所有信息
                return world_coords, rule_type, assumed_z
    
    # 如果遍历完所有规则都无法计算，则返回None
    return None, None, None


def image_to_world_plane(image_point, camera_params, assumed_height=0.0):
    """
    将图像上的一个点投影到世界坐标系中 Z=assumed_height 的平面上。
    """
    cam_matrix = camera_params['camera_matrix']
    dist_coeffs = camera_params['dist_coeffs']
    rvec = camera_params['rvec']
    tvec = camera_params['tvec']
    board_to_room_R = camera_params['board_to_room_R']
    board_to_room_t = camera_params['board_to_room_t']
    #   t:
    #   - 1246072.0
    #   - 578116.0
    #   - 0.0


    # 1. 畸变校正图像点 (使用标准cv2.undistortPoints)
    u, v = image_point
    pixel_coords = np.array([[u, v]], dtype=np.float32)

    undistorted_coords = cv2.undistortPoints(
        pixel_coords, cam_matrix, dist_coeffs
    )[0][0]

    x_norm, y_norm = undistorted_coords

    # 2. 构建从世界到相机的旋转矩阵
    R, _ = cv2.Rodrigues(rvec)

    # 3. 计算相机光心在世界坐标系中的位置
    camera_center_board = -R.T @ tvec

    # 4. 计算射线方向（在世界（标定板）坐标系中）
    # 将归一化的图像点(x_norm, y_norm, 1)从相机坐标系旋转到世界坐标系
    ray_direction_board = R.T @ np.array([[x_norm], [y_norm], [1.0]])
    ray_direction_board = ray_direction_board / np.linalg.norm(ray_direction_board)

    # 5. 计算射线与 Z=assumed_height 平面的交点
    # 射线方程: P = camera_center_board + t * ray_direction_board
    # 平面方程: P_z = assumed_height
    # t = (assumed_height - camera_center_z) / ray_direction_z
    if abs(ray_direction_board[2, 0]) < 1e-6: # 射线平行于Z平面
        return None
        
    t = (assumed_height - camera_center_board[2, 0]) / ray_direction_board[2, 0]
    
    if t < 0: # 交点在相机后面，无效
        return None

    intersection_board = (camera_center_board + t * ray_direction_board)

    # print("intersection_board shape:", intersection_board.flatten().shape)
    # print("intersection_board:", intersection_board.flatten())

    return (intersection_board[0, 0], intersection_board[1, 0])

def world_to_cad(world_coords, camera_params):
    """将世界坐标(标定板坐标系)转换为CAD坐标(房间坐标系)。"""
    if world_coords is None:
        return None
    
    # 获取标定板到房间的转换参数
    board_to_room_R = camera_params['board_to_room_R']
    board_to_room_t = camera_params['board_to_room_t']
    
    # 构建标定板坐标系下的3D点 (假设z=0,在标定板平面上)
    world_point = np.array([world_coords[0], world_coords[1], 0.0])

    # print("board_to_room_R @ world_point shape:", (board_to_room_R @ world_point).shape)
    # print("board_to_room_R @ world_point:", board_to_room_R @ world_point)
    # print("board_to_room_t shape:", board_to_room_t.shape)
    # print("board_to_room_t:", board_to_room_t[0],board_to_room_t[1])
    
    # 应用旋转和平移,转换到CAD(房间)坐标系
    # 运算逻辑: (3x3矩阵) @ (3x1向量) + (3x1向量) -> 结果为 (3x1向量)
    if board_to_room_t.shape == (2,):
        board_to_room_t = np.array([board_to_room_t[0], board_to_room_t[1], 0.0])
    
    cad_point = board_to_room_R @ world_point + board_to_room_t

    
    # 返回CAD坐标的x和y分量
    return (cad_point[0], cad_point[1])

def draw_annotations(
    frame, person_id, box, cad_coords, keypoints, 
    posture, posture_method, keypoint_type, assumed_z,
    verification_info=None,
    face_boxes=None,
    face_ids=None,
    behavior_events=None,
    reid_confidence=0.0): # [新增] 参数
    """在图像上绘制单个人的骨架、关键点和标注信息。"""
    
    # 绘制骨架和关键点
    if keypoints is not None:
        draw_skeleton_and_keypoints(frame, keypoints, person_id)

    # 绘制锚框
    x1, y1, x2, y2 = map(int, box)
    color_idx = get_color_index(person_id)
    color = SKELETON_COLORS[color_idx % len(SKELETON_COLORS)]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    # 准备文本标签
    label_id = f"ID: {person_id}"
    if reid_confidence > 0:
        label_id += f" ({reid_confidence:.2f})" # [修改] 显示置信度
        
    label_coords = "CAD: N/A"
    if cad_coords is not None:
        label_coords = f"CAD:({cad_coords[0]:.0f}, {cad_coords[1]:.0f})"
    label_posture = f"Pose: {posture}"
    
    # (新增) 准备行为识别标签
    label_behavior = ""
    if behavior_events and len(behavior_events) > 0:
        # 显示置信度最高的行为
        b = behavior_events[0]
        label_behavior = f"Act: {b['behavior_type']} ({b['confidence']:.2f})"

    # ...existing code... (验证信息文本生成保持不变)
    verification_text = ""
    if verification_info and verification_info.get('original_posture') == 'standing':
        verified_posture = verification_info.get('verified_posture', 'standing')
        pixel_distance = verification_info.get('pixel_distance', 0)
        verification_text = f"Ver: {verified_posture} ({pixel_distance:.0f}px)"
        # ... (绘制头部点的代码保持不变)

    # 计算背景框大小
    (w_id, h_id), _ = cv2.getTextSize(label_id, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    (w_coords, h_coords), _ = cv2.getTextSize(label_coords, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    (w_posture, h_posture), _ = cv2.getTextSize(label_posture, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    (w_behavior, h_behavior), _ = cv2.getTextSize(label_behavior, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    
    # 动态计算总高度
    total_h = h_id + h_coords + h_posture + 25
    if label_behavior: total_h += h_behavior + 5
    if verification_text: total_h += 20

    max_w = max(w_id, w_coords, w_posture, w_behavior) + 10
    
    # 绘制背景
    cv2.rectangle(frame, (x1, y1 - total_h), (x1 + max_w, y1), color, -1)
    
    # 确定字体颜色
    b, g, r = color
    luminance = (r * 299 + g * 587 + b * 114) / 1000
    text_color = (0, 0, 0) if luminance > 128 else (255, 255, 255)

    # 逐行绘制文本
    y_pos = y1 - total_h + 15
    
    if verification_text:
        cv2.putText(frame, verification_text, (x1 + 5, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
        y_pos += 20
        
    if label_behavior:
        cv2.putText(frame, label_behavior, (x1 + 5, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
        y_pos += h_behavior + 5

    cv2.putText(frame, label_posture, (x1 + 5, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
    y_pos += h_posture + 5
    cv2.putText(frame, label_id, (x1 + 5, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
    y_pos += h_id + 5
    cv2.putText(frame, label_coords, (x1 + 5, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

    # ...existing code... (绘制面部框保持不变)
    if face_boxes and face_ids:
        for face_box, face_id in zip(face_boxes, face_ids):
            fx1, fy1, fx2, fy2 = map(int, face_box)
            cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (0, 255, 0), 1)
            cv2.putText(frame, str(face_id), (fx1, fy1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

def create_person_info(person_id, posture, posture_method, keypoints, 
                       world_coords, keypoint_type, assumed_z, 
                       cad_coords, verification_info,
                       face_boxes=None,
                       face_ids=None,
                       behavior_events=None,
                       reid_confidence=0.0): # [新增] 参数
    """创建人员信息字典"""
    person_info = {
        'person_id': person_id,
        'reid_confidence': float(reid_confidence), # [新增] 保存置信度
        'posture': posture,
        'posture_method': posture_method,
        'has_keypoints': keypoints is not None,
        'keypoints': keypoints.tolist() if keypoints is not None else None
    }
    
    # --- 新增：行为事件 ---
    person_info['behavior_events'] = behavior_events if behavior_events else []
    # --------------------

    if face_boxes:
        serializable_face_boxes = []
        for box in face_boxes:
            serializable_face_boxes.append([float(coord) for coord in box])
        person_info['face_boxes'] = serializable_face_boxes
    
    if face_ids:
        person_info['face_ids'] = face_ids

    if verification_info:
        person_info['posture_verification'] = verification_info
    
    if world_coords is not None:
        person_info['world_x_mm'] = float(world_coords[0])
        person_info['world_y_mm'] = float(world_coords[1])
        person_info['keypoint_type'] = keypoint_type
        person_info['assumed_z_mm'] = assumed_z
    
    if cad_coords is not None:
        person_info['cad_x'] = float(cad_coords[0])
        person_info['cad_y'] = float(cad_coords[1])
    
    return person_info

def world_to_pixel(world_coords, camera_params, z=1700):
    """
    将世界坐标反投影到图像像素坐标
    """
    # 提取相机参数
    cam_matrix = camera_params['camera_matrix']
    dist_coeffs = camera_params['dist_coeffs']
    rvec = camera_params['rvec']
    tvec = camera_params['tvec']
    board_to_room_R = camera_params['board_to_room_R']
    board_to_room_t = camera_params['board_to_room_t']
    
    # 将房间坐标系转换回标定板坐标系
    room_point = np.array([world_coords[0], world_coords[1], z])
    board_point = board_to_room_R.T @ (room_point - board_to_room_t)
    
    # 将3D点投影到图像平面
    image_points, _ = cv2.projectPoints(
        board_point.reshape(1, 1, 3),
        rvec,
        tvec,
        cam_matrix,
        dist_coeffs
    )
    
    return image_points[0][0]
	
def image_to_world_and_back_to_pixel(image_point, camera_params, assumed_height=-1700):
    """
    将图像像素坐标转换为世界坐标，然后再转换回像素坐标
    
    参数:
        image_point: 图像像素坐标 (u, v)
        camera_params: 相机参数字典
        assumed_height: 假设的高度值，默认-1700mm
    
    返回:
        tuple: 恢复的像素坐标 (x, y)，如果出错则返回 None
    """
    try:
        # 1. 提取相机参数
        cam_matrix = camera_params['camera_matrix']
        dist_coeffs = camera_params['dist_coeffs']
        rvec = camera_params['rvec']
        tvec = camera_params['tvec']
        board_to_room_R = camera_params['board_to_room_R']
        board_to_room_t = camera_params['board_to_room_t']
        
        # 2. 畸变校正图像点
        u, v = image_point
        pixel_coords = np.array([[u, v]], dtype=np.float32)
        
        undistorted_coords = cv2.undistortPoints(
            pixel_coords, cam_matrix, dist_coeffs
        )[0][0]
        
        x_norm, y_norm = undistorted_coords
        
        # 3. 构建从世界到相机的旋转矩阵
        R, _ = cv2.Rodrigues(rvec)
        
        # 4. 计算相机光心在世界坐标系中的位置
        camera_center_board = -R.T @ tvec
        
        # 5. 计算射线方向（在世界坐标系中）
        ray_direction_board = R.T @ np.array([[x_norm], [y_norm], [1.0]])
        ray_direction_board = ray_direction_board / np.linalg.norm(ray_direction_board)
        
        # 6. 计算射线与 Z=0 平面的交点
        if abs(ray_direction_board[2, 0]) < 1e-6:
            return None
            
        t = (0 - camera_center_board[2, 0]) / ray_direction_board[2, 0]
        
        if t < 0:
            return None
            
        intersection_board = (camera_center_board + t * ray_direction_board)
        
        # 7. 将交点的Z坐标设为assumed_height
        intersection_board[2, 0] = assumed_height
        
        # 8. 将世界坐标转换回像素坐标
        # 将标定板坐标系下的点转换到房间坐标系
        room_point = board_to_room_R @ np.array([intersection_board[0, 0], intersection_board[1, 0], intersection_board[2, 0]]) + board_to_room_t
        
        # 将房间坐标系转换回标定板坐标系
        board_point = np.linalg.inv(board_to_room_R) @ (room_point - board_to_room_t)
        
        # 将3D点投影到图像平面
        image_points, _ = cv2.projectPoints(
            board_point.reshape(1, 1, 3),
            rvec,
            tvec,
            cam_matrix,
            dist_coeffs
        )
        
        recovered_pixel = image_points[0][0]
        
        # 9. 只返回像素坐标
        return (recovered_pixel[0], recovered_pixel[1])
        
    except Exception as e:
        return None

