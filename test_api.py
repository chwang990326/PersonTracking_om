"""
人员检测接口测试
"""

import os
import requests
import base64
import json
from datetime import datetime


# ==================== 配置：修改这里的文件夹路径 ====================
FOLDER_PATH = "./monitorSnap/"  # 修改为你的图片文件夹路径
API_URL = "http://localhost:8000/api/v1/person/detect"

def parse_filename_time(filename):
    """
    解析文件名中的时间，用于排序
    文件名格式如: 1997863184173895681_20260330094835752.jpg
    提取 endTime: 20260330094835752 (YYYYMMDDHHMMSSXXX)
    """
    base_name = os.path.basename(filename)
    name_without_ext = os.path.splitext(base_name)[0]
    parts = name_without_ext.split('_')
    
    if len(parts) >= 2:
        end_time_str = parts[-1]
        if len(end_time_str) >= 14:
            try:
                # 前14位是 YYYYMMDDHHMMSS，后面的是毫秒
                dt_str = end_time_str[:14]
                ms_str = end_time_str[14:]
                dt = datetime.strptime(dt_str, "%Y%m%d%H%M%S")
                ms = int(ms_str) if ms_str else 0
                # 转为微秒
                return dt.replace(microsecond=ms * 1000)
            except ValueError:
                pass
    return datetime.min

if not os.path.exists(FOLDER_PATH):
    print(f"文件夹不存在: {FOLDER_PATH}")
    exit(1)

# 获取文件夹下所有图片
image_files = [os.path.join(FOLDER_PATH, f) for f in os.listdir(FOLDER_PATH) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# 按时间戳顺序对文件进行排序
image_files.sort(key=parse_filename_time)

for img_path in image_files:
    dt = parse_filename_time(img_path)
    timestamp_iso = dt.isoformat() if dt != datetime.min else datetime.now().isoformat()
    
    print(f"\n========================================")
    print(f"正在处理图片: {img_path}")
    print(f"时间戳: {timestamp_iso}")
    
    # 读取图片并转Base64
    with open(img_path, 'rb') as f:
        image_base64 = base64.b64encode(f.read()).decode('utf-8')
    
    # 构造请求
    request_data = {
        "image": image_base64,
        "camera_id": "207",
        "timestamp": timestamp_iso,
        "enable_face_recognition": True,
        "enable_behavior_detection": True,
        "enable_uniformer_inference": True,
    }

    # 调用接口
    try:
        response = requests.post(API_URL, json=request_data)
        result = response.json()
        
        extracted_info = []
        def extract_persons(data):
            if isinstance(data, dict):
                # 如果当前字典包含 person_id，则提取我们关注的字段
                if "person_id" in data:
                    extracted_info.append({
                        "person_id": data.get("person_id"),
                        "track_id": data.get("track_id"),
                        "id_resource": data.get("id_resource"),
                        "switch_from": data.get("switch_from"),
                        "conf": data.get("conf")
                    })
                # 继续递归查找
                for v in data.values():
                    extract_persons(v)
            elif isinstance(data, list):
                for item in data:
                    extract_persons(item)
                    
        extract_persons(result)
        
        if extracted_info:
            print(json.dumps(extracted_info, indent=4, ensure_ascii=False))
        else:
            print("未提取到相关人员字段，完整返回：")
            print(json.dumps(result, indent=4, ensure_ascii=False))
            
    except Exception as e:
        print(f"请求失败: {e}")
