"""
视频流人员检测接口测试 - 单路测试版
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
# 模拟1路视频输入，可以是同一个文件
VIDEO_PATH_1 = "video/0327测试_5fps.mp4"
# VIDEO_PATH_2 = "video/reidtest3_3fps.mp4" 

API_URL = "http://localhost:8000/api/v1/person/detect"
ENCODE_FORMAT = ".png"

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
    print(f"[{thread_name}] 开始处理视频: {video_path} (Camera: {camera_id}, Encode: {ENCODE_FORMAT})")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"[{thread_name}] 视频处理结束")
                break

            frame_count += 1
            
            # 1. 编码
            success, buffer = cv2.imencode(ENCODE_FORMAT, frame)
            if not success:
                print(f"[{thread_name}] Frame {frame_count} 编码失败: {ENCODE_FORMAT}")
                break
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
                    
                    # 获取 data 对象
                    data = result.get('data', {})
                    
                    # 获取 exist_person 字段
                    exist_person = data.get('exist_person', False)
                    
                    # 解析每个人员信息的 id_resource 和 switch_from
                    persons = data.get('persons', [])
                    person_details = []
                    for p in persons:
                        p_id = p.get('person_id')
                        resource = p.get('id_resource', '未知')
                        switch = p.get('switch_from')
                        
                        # 格式化输出字符
                        detail_str = f"ID:{p_id} (来源:{resource})"

                        detail_str += f" [从 {switch} 切换]"

                        detail_str += f" [原始Track_ID: {p.get('track_id')}]"
                        
                        person_details.append(detail_str)
                    
                    details_output = " | ".join(person_details) if person_details else "无人员"
                    
                    # 在日志中打印详细的 ID 来源和切换状态（已删除耗时输出并带有 switch 逻辑）
                    print(f"[{thread_name}] Frame {frame_count} | 是否有人: {exist_person} | 识别详情: {details_output}")
                    
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
    print(">>> 启动单路视频测试 <<<")
    print(f">>> 当前图像编码格式: {ENCODE_FORMAT}")
    print("-" * 50)

    # 直接调用处理函数，不使用多线程
    process_video_stream(VIDEO_PATH_1, "207", "Single-Stream")

    # --- 原多路并发代码已注释 ---
    # # 创建两个线程，模拟两个不同的摄像头
    # # Camera 207
    # t1 = threading.Thread(
    #     target=process_video_stream, 
    #     args=(VIDEO_PATH_1, "207", "Stream-1")
    # )
    
    # # Camera 206 (假设有这个配置，或者系统会自动处理新ID)
    # t2 = threading.Thread(
    #     target=process_video_stream, 
    #     args=(VIDEO_PATH_2, "206", "Stream-2")
    # )

    # # 启动线程
    # t1.start()
    # t2.start()

    # # 等待结束
    # t1.join()
    # t2.join()
    
    print("-" * 50)
    print(">>> 测试完成")


if __name__ == "__main__":
    main()

