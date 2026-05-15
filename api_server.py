import time
import base64
import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field
from typing import Dict, List, Optional

from calibration_module.routers import router as calibration_router

from service import CameraConfigError, VisionAnalysisService
from utils.profiler import ProfilingStats, RequestProfiler

# --- 1. 定义 Request/Response 模型 (根据文档) ---

class PersonDetectRequest(BaseModel):
    """3.2 接口入参定义"""
    image: str = Field(..., description="Base64 编码的图像数据")
    camera_id: str = Field(..., description="相机 ID")
    associated_camera_ids: Optional[List[str]] = []
    timestamp: str = Field(..., description="ISO8601 时间戳")
    enable_face_recognition: bool = False
    enable_behavior_detection: bool = False
    enable_uniformer_inference: bool = False
    enable_spatial_positioning: bool = True
    enable_target_tracking: bool = True

class BehaviorEvent(BaseModel):
    behavior_type: str
    confidence: float
    duration: float

class PersonInfo(BaseModel):
    person_id: Optional[str]
    track_id: Optional[str] = Field(None, description="YOLO/ByteTrack 当前帧返回的原始轨迹 ID")
    id_resource: str = Field(..., description="身份来源：face 表示本帧由有效人脸结果确认，ReID 表示来自 ReID 路径")
    switch_from: Optional[str] = Field(None, description="本次请求中发生 ID 切换时的旧 ID")
    conf: float = Field(0.0, description="ReID 识别置信度 (0.0-1.0)")    # 置信度含义
                                                                        # 0.0：临时 ID（未匹配到已知身份）
                                                                        # 0.5-0.9：与已知身份的相似度（阈值为 0.9）
                                                                        # > 0.9：高置信度匹配
    world_coordinates: List[float] # [X, Y, Z]
    behavior_events: List[BehaviorEvent]
    bounding_box: List[float] # [x, y, w, h]
    bbox_anchor_points: Dict[str, List[float]]
    keypoint_count: int = Field(0, description="识别到的有效关节点数量")

class DetectResponseData(BaseModel):
    exist_person: bool
    persons: List[PersonInfo]
    

class BaseResponse(BaseModel):
    code: int
    message: str
    data: Optional[DetectResponseData] = None
    timestamp: int


class ThirdFaceVerifyRequest(BaseModel):
    picBase64: str = Field(..., description="face image base64")


class ThirdFaceVerifyData(BaseModel):
    personId: str = Field("", description="matched person ID")


class ThirdFaceVerifyResponse(BaseModel):
    code: str
    msg: str
    data: ThirdFaceVerifyData

# --- 2. 初始化 App 和 服务 ---
app = FastAPI(title="视频人员识别接口服务", version="1.0")

# 将标定模块的路由注册到主应用中
app.include_router(calibration_router, tags=["相机标定服务"])

# 全局服务实例 (启动时加载模型)
service = VisionAnalysisService()
api_profiler_stats = ProfilingStats("api")

def get_current_timestamp():
    return int(time.time() * 1000)

# --- 3. 辅助函数：Base64 转 OpenCV ---
def base64_to_cv2(base64_str):
    try:
        # 去掉可能存在的头部 "data:image/jpeg;base64,"
        if "," in base64_str:
            base64_str = base64_str.split(",")[1]
        
        img_data = base64.b64decode(base64_str)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        return None

# --- 4. 接口实现 ---
app.include_router(calibration_router, tags=["相机标定服务"])


@app.post("/api/v1/person/detect", response_model=BaseResponse)
async def person_detect(request: PersonDetectRequest):
    profiler = RequestProfiler()
    profiler.start("0_Request_Total")
    """
    3.2 人员检测接口
    """
    # 1. Base64 解码
    with profiler.section("1_Base64_JPEG_Decode"):
        img = base64_to_cv2(request.image)
    if img is None:
        response = BaseResponse(
            code=1001,
            message="图片数据格式错误",
            timestamp=get_current_timestamp()
            )
        profiler.stop("0_Request_Total")
        api_profiler_stats.record(profiler)
        return response

    try:
        # 2. 调用业务层逻辑
        # [修改] 正确解包返回的元组 (注意 service.py 返回顺序是 exist_person, persons)
        with profiler.section("2_Service_Detect"):
            service_profiler = RequestProfiler()
            exist_person, persons = service.detect_person_from_image(
                img,
                camera_id=request.camera_id,
                enable_face=request.enable_face_recognition,
                enable_behavior=request.enable_behavior_detection,
                enable_uniformer=request.enable_uniformer_inference,
                enable_positioning=request.enable_spatial_positioning,
                enable_tracking=request.enable_target_tracking,
                profiler=service_profiler,
            )
        profiler.merge(service_profiler)
        
        # 3. 构造成功响应
        with profiler.section("3_Response_Build"):
            response = BaseResponse(
                code=0,
            message="操作成功",
            # [修改] 将两个参数分别传入
            data=DetectResponseData(exist_person=exist_person, persons=persons),
            timestamp=get_current_timestamp()
        )
        profiler.stop("0_Request_Total")
        api_profiler_stats.record(profiler)
        return response

    except CameraConfigError as e:
        response = BaseResponse(
            code=1202,
            message=str(e),
            timestamp=get_current_timestamp()
        )
        profiler.stop("0_Request_Total")
        api_profiler_stats.record(profiler)
        return response
    except Exception as e:
        # 捕获未知异常
        print(f"Internal Error: {e}")
        response = BaseResponse(
            code=1400,
            message=f"系统内部错误: {str(e)}",
            timestamp=get_current_timestamp()
        )
        profiler.stop("0_Request_Total")
        api_profiler_stats.record(profiler)
        return response

@app.get("/api/v1/face/refresh", response_model=BaseResponse)
async def face_refresh():
    """
    3.4 加载人脸库接口
    """
    success = service.reload_library()
    
    if success:
        return BaseResponse(
            code=0,
            message="加载成功",
            timestamp=get_current_timestamp()
        )
    else:
        return BaseResponse(
            code=1201, # 业务逻辑错误
            message="加载失败",
            timestamp=get_current_timestamp()
        )

@app.post("/third/face/verify", response_model=ThirdFaceVerifyResponse)
async def third_face_verify(request: ThirdFaceVerifyRequest):
    """
    Third-party face verification API.
    """
    img = base64_to_cv2(request.picBase64)
    empty_data = ThirdFaceVerifyData(personId="")

    if img is None:
        return ThirdFaceVerifyResponse(
            code="1001",
            msg="invalid image data",
            data=empty_data,
        )

    try:
        person_id, face_detected = service.verify_face_from_image(img)
        if not face_detected:
            return ThirdFaceVerifyResponse(
                code="1201",
                msg="no face detected",
                data=empty_data,
            )

        if not person_id:
            return ThirdFaceVerifyResponse(
                code="1202",
                msg="person not found",
                data=empty_data,
            )

        return ThirdFaceVerifyResponse(
            code="0",
            msg="success",
            data=ThirdFaceVerifyData(personId=person_id),
        )
    except Exception as e:
        print(f"Internal Error: {e}")
        return ThirdFaceVerifyResponse(
            code="1400",
            msg=f"internal error: {str(e)}",
            data=empty_data,
        )


if __name__ == "__main__":
    # 启动服务
    uvicorn.run(app, host="0.0.0.0", port=8130)

