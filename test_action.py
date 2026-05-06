import cv2
import requests
import base64
import json
import time
from datetime import datetime

# ==================== 配置区 ====================
VIDEO_PATH = "video/normal_2_4_8_ch1_1_20260114172216385_20260114172343013.mp4"
API_URL = "http://localhost:8000/api/v1/person/detect"
CAMERA_ID = "2001274229076791297"
# ===============================================

def send_frame_to_api(frame):
    """将 OpenCV 帧编码并发送到 API，返回 JSON 结果"""
    # 编码图像
    success, buffer = cv2.imencode('.jpg', frame)
    if not success:
        return {"error": "图像编码失败"}
    
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    
    # 构造请求体
    request_data = {
        "image": image_base64,
        "camera_id": CAMERA_ID,
        "timestamp": datetime.now().isoformat(),
        "enable_face_recognition": True,
        "enable_behavior_detection": True,
        "enable_uniformer_inference": True,
    }
    
    try:
        # 发送请求
        response = requests.post(API_URL, json=request_data, timeout=15)
        
        # 解析 JSON
        try:
            return response.json()
        except json.JSONDecodeError:
            return {"error": "Invalid JSON response", "raw_text": response.text}
            
    except Exception as e:
        return {"error": f"Request failed: {str(e)}"}

def process_video():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"无法打开视频: {VIDEO_PATH}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f">>> 开始处理视频: {VIDEO_PATH}")
    print(f">>> 总帧数: {total_frames}")
    print(">>> 模式: 每 5 帧调用一次接口\n")

    frame_count = 0
    api_call_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # === 修改逻辑核心 ===
        # 如果当前帧号不能被 5 整除，则跳过本次循环，不调用接口
        if frame_count % 5 != 0:
            continue
        # ====================

        # 只有能被 5 整除的帧（5, 10, 15...）才会执行到这里
        api_call_count += 1
        
        # 1. 调用 API
        result = send_frame_to_api(frame)
        
        # 2. 打印结果
        print(f"--- Frame {frame_count} (Sent) ---")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print("-" * 30)

    end_time = time.time()
    cap.release()
    
    print(f"\n>>> 处理完成。")
    print(f">>> 视频总帧数: {frame_count}")
    print(f">>> 实际接口调用次数: {api_call_count}")
    print(f">>> 总耗时: {end_time - start_time:.2f} 秒")

if __name__ == "__main__":
    process_video()
