import yaml
import numpy as np
import cv2
"""
加载视频和相机标定参数的工具模块
"""

def load_config(camera_config_path, order_config_path):
    """加载并解析相机标定和计算规则YAML文件。"""
    # (修改) 明确指定使用 utf-8 编码打开文件
    with open(camera_config_path, 'r', encoding='utf-8') as f:
        camera_config = yaml.safe_load(f)
    
    with open(order_config_path, 'r', encoding='utf-8') as f:
        order_config = yaml.safe_load(f)
    
    # --- 1. 适配不同格式的 dist_coeffs ---
    dist_coeffs_raw = camera_config['dist_coeffs']
    # 格式A (204): [[k1, k2...]] 列表的列表 -> 取第一个
    # 格式B (102): [k1, k2...] 扁平列表 -> 直接用
    if len(dist_coeffs_raw) > 0 and isinstance(dist_coeffs_raw[0], (list, tuple)):
        dist_coeffs = np.array(dist_coeffs_raw[0])
    else:
        dist_coeffs = np.array(dist_coeffs_raw)

    # --- 2. 适配不同格式的 extrinsics ---
    extrinsics_raw = camera_config['extrinsics']
    if isinstance(extrinsics_raw, list):
        # 格式A (204): 列表形式，取第一帧
        ext_data = extrinsics_raw[0]
    else:
        # 格式B (102): 字典形式
        ext_data = extrinsics_raw
    
    # 兼容 rvec / rotation_vector
    rvec = ext_data.get('rvec')
    if rvec is None:
        rvec = ext_data.get('rotation_vector')
        
    # 兼容 tvec / translation_vector
    tvec = ext_data.get('tvec')
    if tvec is None:
        tvec = ext_data.get('translation_vector')

    # --- 3. 适配 board_to_room 的平移向量键名 (t vs T) ---
    b2r_config = camera_config['board_to_room']
    
    # [修复] 使用 .get() 避免 KeyError，同时支持 't' 和 'T'
    # 之前的代码直接用 ['t'] 访问，遇到只有 'T' 的配置(102)会直接报错崩溃
    b2r_t = b2r_config.get('t')
    if b2r_t is None:
        b2r_t = b2r_config.get('T')

    params = {
        'camera_matrix': np.array(camera_config['camera_matrix']),
        'dist_coeffs': dist_coeffs, # 使用适配后的畸变系数
        'image_width': camera_config['image_width'],
        'image_height': camera_config['image_height'],
        # 使用适配后的外参
        'rvec': np.array(rvec),
        'tvec': np.array(tvec),
        'board_to_room_R': np.array(b2r_config['R']),
        'board_to_room_t': np.array(b2r_t).flatten(), # 使用适配后的 T/t
        # (新增) 加载姿态高度映射规则
        'posture_height_mapping': order_config.get('posture_height_mapping', {}),
        # (新增) 加载CAD坐标转换参数
        'cad_transform': order_config.get('cad_transform', {'x_offset': 0, 'y_offset': 0})
    }
    return params

def get_video_streams(input_path, output_path):
    """
        获取视频读取和写入对象。
        返回读取对象和写入对象，供后续视频处理使用
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"无法打开视频文件: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    return cap, writer