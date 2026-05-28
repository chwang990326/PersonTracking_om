import time
import base64
import asyncio
import threading
import cv2
import numpy as np
import uvicorn
from dataclasses import dataclass
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
    data: Optional[object] = None
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
@dataclass
class CameraFrameJob:
    camera_id: str
    request: PersonDetectRequest
    image: np.ndarray
    profiler: RequestProfiler
    loop: asyncio.AbstractEventLoop
    future: asyncio.Future


class CameraQueueState:
    def __init__(self, camera_id: str):
        self.camera_id = camera_id
        self.lock = threading.Lock()
        self.processing = False
        self.pending_job = None


class CameraOrderedProcessor:
    """
    Serialize stateful tracking/ReID calls by camera_id.

    Each camera has at most one frame being processed and one latest waiting
    frame. When overloaded, the older waiting frame is dropped so processing
    stays close to real time.
    """

    def __init__(self, vision_service: VisionAnalysisService):
        self.service = vision_service
        self.states = {}
        self.states_lock = threading.Lock()

    def _get_state(self, camera_id: str) -> CameraQueueState:
        with self.states_lock:
            state = self.states.get(camera_id)
            if state is None:
                state = CameraQueueState(camera_id)
                self.states[camera_id] = state
            return state

    def _complete_job(self, job: CameraFrameJob, response: BaseResponse):
        def set_result():
            if not job.future.done():
                job.future.set_result(response)

        job.loop.call_soon_threadsafe(set_result)

    def _dropped_response(self) -> BaseResponse:
        return BaseResponse(
            code=1203,
            message="旧帧已丢弃，已保留最新等待帧",
            timestamp=get_current_timestamp(),
        )

    def _process_job(self, job: CameraFrameJob) -> BaseResponse:
        request = job.request
        try:
            with job.profiler.section("2_Service_Detect"):
                service_profiler = RequestProfiler()
                exist_person, persons = self.service.detect_person_from_image(
                    job.image,
                    camera_id=request.camera_id,
                    enable_face=request.enable_face_recognition,
                    enable_behavior=request.enable_behavior_detection,
                    enable_uniformer=request.enable_uniformer_inference,
                    enable_positioning=request.enable_spatial_positioning,
                    enable_tracking=request.enable_target_tracking,
                    profiler=service_profiler,
                )
            job.profiler.merge(service_profiler)

            with job.profiler.section("3_Response_Build"):
                return BaseResponse(
                    code=0,
                    message="操作成功",
                    data=DetectResponseData(exist_person=exist_person, persons=persons),
                    timestamp=get_current_timestamp(),
                )

        except CameraConfigError as e:
            return BaseResponse(
                code=1202,
                message=str(e),
                timestamp=get_current_timestamp(),
            )
        except Exception as e:
            print(f"Internal Error: {e}")
            return BaseResponse(
                code=1400,
                message=f"系统内部错误: {str(e)}",
                timestamp=get_current_timestamp(),
            )

    def _run_camera_loop(self, state: CameraQueueState, first_job: CameraFrameJob):
        job = first_job
        while job is not None:
            response = self._process_job(job)
            self._complete_job(job, response)

            with state.lock:
                job = state.pending_job
                state.pending_job = None
                if job is None:
                    state.processing = False
                    return

    async def submit(self, request: PersonDetectRequest, image: np.ndarray, profiler: RequestProfiler) -> BaseResponse:
        camera_id = request.camera_id or "default"
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        job = CameraFrameJob(
            camera_id=camera_id,
            request=request,
            image=image,
            profiler=profiler,
            loop=loop,
            future=future,
        )

        state = self._get_state(camera_id)
        dropped_job = None
        start_worker = False

        with state.lock:
            if state.processing:
                dropped_job = state.pending_job
                state.pending_job = job
            else:
                state.processing = True
                start_worker = True

        if dropped_job is not None:
            print(
                f"[CameraQueue] drop stale waiting frame "
                f"camera_id={dropped_job.camera_id} timestamp={dropped_job.request.timestamp}"
            )
            self._complete_job(dropped_job, self._dropped_response())

        if start_worker:
            thread = threading.Thread(
                target=self._run_camera_loop,
                args=(state, job),
                name=f"camera-processor-{camera_id}",
                daemon=True,
            )
            thread.start()

        return await future


camera_processor = CameraOrderedProcessor(service)


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
        response = await camera_processor.submit(request, img, profiler)
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
    result = service.reload_library()
    success = result.get("success", False) if isinstance(result, dict) else bool(result)
    
    if success:
        return BaseResponse(
            code=0,
            message="加载成功",
            data=result if isinstance(result, dict) else None,
            timestamp=get_current_timestamp()
        )
    else:
        return BaseResponse(
            code=1201, # 业务逻辑错误
            message=result.get("error", "加载失败") if isinstance(result, dict) else "加载失败",
            data=result if isinstance(result, dict) else None,
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
    uvicorn.run(app, host="0.0.0.0", port=8136)

