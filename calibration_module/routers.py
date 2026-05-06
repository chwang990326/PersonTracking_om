import base64
import cv2
import numpy as np
import time
import uvicorn
import yaml
import os
from fastapi import FastAPI, HTTPException
from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import List, Optional

# 导入上面写好的服务
from .calibration_service import CalibrationService

# app = FastAPI(title="相机标定接口服务")
# service = CalibrationService()

# [修改 3] 实例化 Router
router = APIRouter()
service = CalibrationService()

# --- 1. 定义数据模型 (参照文档 3.3) ---

class CalibrationRequest(BaseModel):
    camera_id: str = Field(..., description="相机唯一标识")
    # 文档中的 'origin_images' 对应这里的 '标定板原点图片' (用于外参)
    origin_images: List[str] = Field(..., description="标定板原点图片(Base64)")
    # 文档中的 'arbitrary_images' 对应这里的 '任意角度标定图片' (用于内参)
    arbitrary_images: List[str] = Field(..., description="任意角度标定图片(Base64)")
    camera_orientation: str = Field(..., description="相机朝向, 如 X+Y+")
    calibration_world_coordinates: List[List[float]] = Field(..., description="标定板4角世界坐标")

class ExtrinsicsData(BaseModel):
    rotation_vector: List[List[float]]
    translation_vector: List[List[float]]

class BoardToRoomData(BaseModel):
    R: List[List[float]]
    T: List[float]

class CalibrationData(BaseModel):
    camera_id: str
    camera_matrix: List[List[float]]
    dist_coeffs: List[float] # 文档示例中是一维数组 [-0.123, ...]
    image_width: int
    image_height: int
    extrinsics: ExtrinsicsData
    board_to_room: BoardToRoomData
    calibration_time: float

class BaseResponse(BaseModel):
    code: int
    message: str
    data: Optional[CalibrationData] = None
    timestamp: int

# --- 2. 辅助函数 ---

def base64_to_cv2(base64_str_list):
    """批量转换 Base64 为 OpenCV 图像"""
    images = []
    for b64 in base64_str_list:
        try:
            if "," in b64:
                b64 = b64.split(",")[1]
            img_data = base64.b64decode(b64)
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is not None:
                images.append(img)
        except Exception:
            continue # 忽略错误的图片
    return images

def get_current_timestamp():
    return int(time.time() * 1000)

def save_calibration_to_yaml(data: CalibrationData):
    """将标定结果保存为YAML文件"""
    filename = f"camera_params_{data.camera_id}.yaml"
    try:
        # 兼容 Pydantic v1 (.dict()) 和 v2 (.model_dump())
        if hasattr(data, 'model_dump'):
            data_dict = data.model_dump()
        else:
            data_dict = data.dict()
            
        # 基于模块路径回溯到项目根目录，避免依赖运行时工作目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        config_dir = os.path.join(project_root, "config")
        os.makedirs(config_dir, exist_ok=True)
        file_path = os.path.join(config_dir, filename)

        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(data_dict, f, sort_keys=False, allow_unicode=True)
        print(f"标定参数已保存到文件: {file_path}")
    except Exception as e:
        print(f"保存YAML文件失败: {e}")

# --- 3. 接口实现 ---

@router.post("/api/v1/camera/calibrate", response_model=BaseResponse)
async def calibrate_camera(request: CalibrationRequest):
    try:
        # 1. 解析图片
        # 'arbitrary_images' -> 用于内参 (需要多张)
        intrinsic_imgs = base64_to_cv2(request.arbitrary_images)
        # 'origin_images' -> 用于外参 (需要定点)
        anchor_imgs = base64_to_cv2(request.origin_images)
        
        if len(intrinsic_imgs) < 3:
            return BaseResponse(
                code=1201, 
                message="标定失败，任意角度标定图片(arbitrary_images)数量不足或无法解析，至少需要3张", 
                timestamp=get_current_timestamp()
            )
            
        if len(anchor_imgs) == 0:
            return BaseResponse(
                code=1201, 
                message="标定失败，原点图片(origin_images)无效", 
                timestamp=get_current_timestamp()
            )

        # 2. 调用标定服务
        # 注意：这里我们使用 request.camera_orientation 作为 direction
        result = service.calibrate(
            intrinsic_images=intrinsic_imgs,
            anchor_images=anchor_imgs,
            direction=request.camera_orientation.upper(),
            calibration_world_coordinates=request.calibration_world_coordinates # 传入世界坐标
        )
        
        # 3. 格式化结果 (Numpy -> List)
        # 内参矩阵 3x3
        camera_matrix_list = result["camera_matrix"].tolist()
        
        # 畸变系数
        dist_coeffs_list = result["dist_coeffs"].flatten().tolist()
        
        # 外参 (rvec, tvec)
        rvec_list = result["rvec"].tolist() 
        tvec_list = result["tvec"].tolist()
        
        # Board to Room (使用服务返回的动态计算值)
        board_to_room_R = service.R_room_from_board.tolist()
        board_to_room_T = result["room_origin"].tolist()
        
        # 4. 构造响应数据
        resp_data = CalibrationData(
            camera_id=request.camera_id,
            camera_matrix=camera_matrix_list,
            dist_coeffs=dist_coeffs_list,
            image_width=result["image_width"],
            image_height=result["image_height"],
            extrinsics=ExtrinsicsData(
                rotation_vector=rvec_list,
                translation_vector=tvec_list
            ),
            board_to_room=BoardToRoomData(
                R=board_to_room_R,
                T=board_to_room_T
            ),
            calibration_time=round(result["cost_time"], 2)
        )
        
        # 保存结果到本地 YAML 文件
        save_calibration_to_yaml(resp_data)
        
        return BaseResponse(
            code=0,
            message="标定成功",
            data=resp_data,
            timestamp=get_current_timestamp()
        )
        
    except Exception as e:
        print(f"Error: {e}")
        # 捕获标定过程中的算法错误 (如 findChessboardCorners 失败)
        return BaseResponse(
            code=1201, # 对应文档：业务逻辑错误/标定失败
            message=f"标定失败: {str(e)}",
            timestamp=get_current_timestamp()
        )

# if __name__ == "__main__":
    # uvicorn.run(app, host="0.0.0.0", port=8080)
