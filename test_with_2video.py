"""
视频流人员检测接口测试 - 多路并发版
"""

import requests
import base64
import json
import cv2
import copy
import threading
import time
from datetime import datetime

# ==================== 配置 ====================
# 模拟两路视频输入，可以是同一个文件
VIDEO_PATH_1 = "./video/两名未知人员204_5fps.mp4"
VIDEO_PATH_2 = "./video/两名未知人员206_5fps.mp4" 

API_URL = "http://localhost:8000/api/v1/person/detect"

def remove_base64_data(data, max_len=100):
    """
    递归遍历字典或列表，将过长的字符串（通常是Base64）替换为占位符，
    以便在控制台打印时保持整洁。
    """
    if isinstance(data, dict):
        new_data = {}
        for k, v in data.items():
            if isinstance(v, str) and len(v) > max_len:
                new_data[k] = f"<Base64 Data (len={len(v)}) hidden>"
            else:
                new_data[k] = remove_base64_data(v, max_len)
        return new_data
    elif isinstance(data, list):
        return [remove_base64_data(item, max_len) for item in data]
    else:
        return data

def process_video_stream(video_path, camera_id, thread_name):
    """
    处理单个视频流的线程函数
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[{thread_name}] 无法打开视频文件: {video_path}")
        return

    frame_count = 0
    print(f"[{thread_name}] 开始处理视频: {video_path} (Camera: {camera_id})")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"[{thread_name}] 视频处理结束")
                break

            frame_count += 1
            
            # 1. 编码
            _, buffer = cv2.imencode('.jpg', frame)
            image_base64 = base64.b64encode(buffer).decode('utf-8')

            # 2. 构造请求
            request_data = {
                "image": image_base64,
                "camera_id": camera_id,
                "timestamp": datetime.now().isoformat(),
                "enable_face_recognition": True,
                "enable_behavior_detection": True,
                "enable_uniformer_inference": True,
                "enable_tracking": True
            }

            # 3. 调用接口
            try:
                start_time = datetime.now()
                response = requests.post(API_URL, json=request_data)
                cost_time = (datetime.now() - start_time).total_seconds() * 1000

                if response.status_code == 200:
                    result = response.json()
                    
                    # 取出返回的 data 中的 persons 列表
                    persons = result.get('data', {}).get('persons', [])
                    
                    # 按照要求格式化输出每个人员的信息
                    if not persons:
                         print(f"[{thread_name}] Camera:{camera_id} | Frame {frame_count} | 识别结果: 无人员")
                    else:
                        person_details = []
                        for p in persons:
                            p_id = p.get('person_id')
                            source = p.get('id_resource', '未知')
                            switch = p.get('switch_from')
                            
                            detail_str = f"ID:{p_id} (来源:{source})"
                            if switch is not None:
                                detail_str += f" [从 {switch} 切换]"
                            person_details.append(detail_str)
                            
                        details_output = " | ".join(person_details)
                        print(f"[{thread_name}] Camera:{camera_id} | Frame {frame_count} | 识别详情: {details_output}")
                    
                    # 如果需要详细调试，可以取消下面的注释
                    # clean_result = remove_base64_data(result)
                    # print(f"[{thread_name}] Result: {json.dumps(clean_result, ensure_ascii=False)}")
                else:
                    print(f"[{thread_name}] Frame {frame_count} 请求失败: {response.status_code}")
                    # print(response.text)

            except Exception as e:
                print(f"[{thread_name}] Frame {frame_count} 请求异常: {e}")
                break
            
            # 模拟不同步的请求频率
            # time.sleep(0.01)

    finally:
        cap.release()

def main():
    print(">>> 启动多路视频并发测试 <<<")
    print("-" * 50)

    # 创建两个线程，模拟两个不同的摄像头
    # Camera 204
    t1 = threading.Thread(
        target=process_video_stream, 
        args=(VIDEO_PATH_1, "204", "Stream-1")
    )
    
    # Camera 206
    t2 = threading.Thread(
        target=process_video_stream, 
        args=(VIDEO_PATH_2, "206", "Stream-2")
    )

    # 启动线程
    t1.start()
    t2.start()

    # 等待结束
    t1.join()
    t2.join()
    
    print("-" * 50)
    print(">>> 测试完成")


if __name__ == "__main__":
    main()
