import cv2
import numpy as np
import os
from typing import List, Tuple, Optional, Dict, Any
import json
from ultralytics import YOLO
from pathlib import Path
import yaml
from tqdm import tqdm # 引入tqdm以显示进度条

class PersonTrackingSystem:
    def __init__(self, calibration_file: str):
        self.camera_matrix = None
        self.dist_coeffs = None
        self.rvec = None
        self.tvec = None
        # self.load_calibration_parameters(calibration_file) # 对于此测试，标定参数非必需，可以注释掉
        self.model = self.init_yolov8_model()
        
    def load_calibration_parameters(self, calibration_file: str):
        """兼容两种格式的标定文件加载"""
        try:
            if calibration_file.endswith('.yaml') or calibration_file.endswith('.yml'):
                with open(calibration_file, 'r', encoding='utf-8') as f:
                    calib_data = yaml.safe_load(f)
                
                self.camera_matrix = np.array(calib_data['camera_matrix'])
                self.dist_coeffs = np.array(calib_data['dist_coeffs'][0])
                
                if 'extrinsics' in calib_data and len(calib_data['extrinsics']) > 0:
                    self.rvec = np.array(calib_data['extrinsics'][0]['rvec']).reshape(3, 1)
                    self.tvec = np.array(calib_data['extrinsics'][0]['tvec']).reshape(3, 1)
                else:
                    raise ValueError("标定文件中缺少外参数据")
            else:  
                fs = cv2.FileStorage(calibration_file, cv2.FILE_STORAGE_READ)
                if not fs.isOpened():
                    raise IOError(f"无法打开标定文件: {calibration_file}")
                self.camera_matrix = fs.getNode("camera_matrix").mat()
                self.dist_coeffs = fs.getNode("dist_coeffs").mat()
                self.rvec = fs.getNode("rvecs").getNode("rvec_0").mat()
                self.tvec = fs.getNode("tvecs").getNode("tvec_0").mat()
                fs.release()
            print("标定参数加载成功")
        except Exception as e:
            print(f"加载标定参数失败: {e}")
            raise

    def init_yolov8_model(self):
        """初始化 YOLOv8 模型"""
        try:
            model_path = Path("config/yolov8n.pt")
            if not model_path.exists():
                raise FileNotFoundError(f"模型文件未找到: {model_path}")
            model = YOLO(model_path)
            print("YOLOv8 模型加载成功")
            return model
        except Exception as e:
            print(f"模型加载失败: {e}")
            raise
            
    def detect_person_pixels(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        使用 YOLOv8 检测人物并返回边界框 (x1, y1, x2, y2) 格式。
        """
        results = self.model(frame, verbose=False, classes=[0], conf=0.5)
        boxes = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                boxes.append((x1, y1, x2, y2))
        return boxes

    def process_video(self, video_path: str, output_video_path: str, output_json_path: str):
        """
        处理视频，检测人体，并保存带标注的视频和JSON结果。
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"错误: 无法打开视频文件 {video_path}")
            return

        # 获取视频属性以创建写入器
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        all_frames_data = []
        frame_idx = 0
        
        print(f"开始处理视频: {video_path}")
        with tqdm(total=total_frames, desc="测试处理进度") as pbar:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break

                # 检测人体
                person_boxes = self.detect_person_pixels(frame)
                
                frame_data = {'frame_id': frame_idx, 'persons': []}
                
                # 绘制锚框并记录数据
                for i, box in enumerate(person_boxes):
                    x1, y1, x2, y2 = box
                    # 绘制矩形框
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # 绘制标签
                    label = f"Person {i+1}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # 记录JSON数据
                    frame_data['persons'].append({
                        'person_id': i + 1,
                        'box': [x1, y1, x2, y2]
                    })

                all_frames_data.append(frame_data)
                writer.write(frame)
                frame_idx += 1
                pbar.update(1)

        # 释放资源
        cap.release()
        writer.release()

        # 保存JSON文件
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(all_frames_data, f, indent=4, ensure_ascii=False)
        
        print(f"处理完成。")
        print(f"标注视频已保存至: {output_video_path}")
        print(f"检测结果已保存至: {output_json_path}")

if __name__ == "__main__":
    # --- 配置 ---
    VIDEO_IN_PATH = 'video/output_video.mp4'
    # 对于这个简单的测试，我们不需要标定文件，但构造函数需要它
    # CALIBRATION_FILE = 'config/camera_params.yaml' 
    OUTPUT_DIR = 'test'

    # --- 执行 ---
    # 创建输出文件夹，如果不存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    VIDEO_OUT_PATH = os.path.join(OUTPUT_DIR, 'test.mp4')
    JSON_OUT_PATH = os.path.join(OUTPUT_DIR, 'test.json')

    # 初始化系统 (传入一个虚拟的文件路径，因为我们注释掉了加载)
    system = PersonTrackingSystem(calibration_file="dummy_path.yaml")
    
    # 处理视频
    system.process_video(VIDEO_IN_PATH, VIDEO_OUT_PATH, JSON_OUT_PATH)