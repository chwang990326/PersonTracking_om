# FastAPI 视频人员识别服务

## 项目概述

本项目现已封装为基于 FastAPI 的视频人员识别服务，主入口为 `api_server.py`。服务接收单帧图像（Base64 编码）和相机标识，输出结构化的人员检测、身份识别和空间定位结果。

当前服务链路由 `VisionAnalysisService.detect_person_from_image()` 串联，核心能力包括：

- 人员检测与目标追踪
- 行人重识别（ReID）
- 人脸识别纠偏
- 行为识别
- 姿态估计与空间定位
- 相机标定与参数加载

`main.py` 仍保留在仓库中，但仅用于本地视频测试、可视化调试和算法复现，不作为服务入口。

## 服务入口

- 主入口：`api_server.py`
- 主要接口：
  - `POST /api/v1/person/detect`
  - `GET /api/v1/face/refresh`

服务端典型执行顺序如下：

1. 接收 HTTP 请求并解码 Base64 图像
2. 根据 `camera_id` 加载相机参数
3. 执行人员检测与追踪
4. 使用 ReID 分配当前身份
5. 在启用时使用人脸识别进行确认或纠偏
6. 执行姿态估计与空间定位
7. 返回结构化 JSON 结果

其中身份确认机制以 ReID 为主线，人脸识别在识别到有效人脸且满足当前规则时用于身份确认或修正。

当前 `identity/` 自动写盘规则如下：

- 纯数字 ID 视为未知临时身份，不自动裁剪，也不自动写入 `identity/`
- 只有非数字的已知身份，且本帧识别到有效人脸时，才会自动保存 `anchor_` 特征图
- 纯 ReID 命中已知身份但当前帧没有有效人脸时，不会新增任何 `auto_` 样本

更完整的算法说明见：

- `docs/人员检测流程.md`

## 目录结构

```text
PersonTracking/
├── api_server.py
├── service.py
├── main.py
├── requirements.txt
├── README.md
├── config/
├── models/
├── calibration_module/
├── weights/
├── identity/
├── faceImage/
├── docs/
├── video/
└── results/
```

目录说明：

- `api_server.py`：FastAPI 服务入口
- `service.py`：服务主流程编排与算法调用
- `main.py`：本地视频测试与可视化脚本
- `config/`：模型配置、相机参数配置
- `models/`：检测、ReID、人脸识别、姿态估计等算法模块
- `calibration_module/`：标定相关接口与逻辑
- `weights/`：模型权重文件
- `identity/`：ReID 身份库
- `faceImage/`：人脸库
- `docs/`：详细算法与流程文档

## 环境准备

建议在独立虚拟环境中运行项目。

### 安装依赖

```bash
conda create -n person_tracking python=3.9
conda activate person_tracking
pip install -r requirements.txt
```

项目依赖包括但不限于：

- `fastapi`
- `uvicorn`
- `torch`
- `torchvision`
- `ultralytics`
- `opencv-python`
- `numpy`
- `pyyaml`

### 模型与配置

项目默认按当前仓库结构读取模型与配置文件，例如：

- `config/yolov8n.onnx`
- `config/yolov8n-pose.onnx`
- `config/transformer_120.pth`
- `weights/det_10g.onnx`
- `weights/adaface_ir50_ms1mv2.ckpt`

请确保上述模型文件及相关配置已经就位。

## 启动服务

在项目根目录下可直接运行：

```bash
python api_server.py
```

也可以使用：

```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000
```

默认服务监听：

- `http://0.0.0.0:8000`

## 测试脚本

`main.py` 用于本地视频文件测试、可视化输出和算法调试。它会读取代码中配置的视频路径，执行检测、ReID、人脸识别和姿态定位，并将结果写入视频与 JSON 文件。

它适合用于：

- 单视频离线调试
- 算法效果可视化
- 与服务端结果做对照验证

它不用于对外提供 HTTP 服务。

## 接口说明

### 1. 人员检测接口

- URL：`POST /api/v1/person/detect`
- Content-Type：`application/json`

#### 请求字段

- `image`：Base64 编码图像
- `camera_id`：相机 ID，用于加载对应 `camera_params_{camera_id}.yaml`
- `associated_camera_ids`：关联相机 ID 列表，当前为预留字段
- `timestamp`：请求时间戳字符串
- `enable_face_recognition`：是否启用人脸识别，当前代码默认 `false`
- `enable_behavior_detection`：是否启用行为识别，默认 `false`
- `enable_spatial_positioning`：是否启用空间定位，默认 `true`
- `enable_target_tracking`：是否启用目标追踪/ReID，默认 `true`

#### 最小 `curl` 调用示例

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/person/detect" \
  -H "Content-Type: application/json" \
  -d '{
    "image": "BASE64_IMAGE_DATA",
    "camera_id": "207",
    "associated_camera_ids": [],
    "timestamp": "2026-03-27T10:00:00+08:00",
    "enable_face_recognition": true,
    "enable_behavior_detection": false,
    "enable_spatial_positioning": true,
    "enable_target_tracking": true
  }'
```

### 2. 人脸库刷新接口

- URL：`GET /api/v1/face/refresh`
- 作用：重新加载人脸库并清空所有相机的轨迹上下文

## 响应字段说明

### 顶层响应

- `code`：状态码，`0` 表示成功
- `message`：状态说明
- `data`：业务数据，失败时可能为空
- `timestamp`：毫秒级时间戳

### `data` 结构

- `exist_person`：当前帧是否检测到人
- `persons`：人员结果列表

### `persons[*]` 字段

- `person_id`
  - 当前返回的最终身份
  - 可能为已知身份字符串、临时数字 ID 字符串，或 `null`

- `track_id`
  - 当前检测框对应的原始轨迹 ID
  - 启用 tracking 且存在有效轨迹时返回字符串，否则为 `null`

- `id_resource`
  - 当前身份来源
  - `face`：本帧存在有效人脸识别结果，且最终 `person_id` 与该人脸身份一致
  - `ReID`：当前身份来自 ReID 路径或未被本帧有效人脸结果确认

- `switch_from`
  - 若本次请求中发生身份切换，记录切换前的旧 ID
  - 未发生切换时为 `null`

- `conf`
  - ReID 置信度
  - 表示当前身份与 ReID 身份库的匹配程度
  - 注意：该字段不是人脸相似度

- `world_coordinates`
  - 三维坐标数组 `[X, Y, Z]`
  - 当前实现中通常返回 `[x, y, 0.0]`

- `behavior_events`
  - 行为识别结果列表
  - 未启用行为识别时通常为空数组

- `bounding_box`
  - 检测框 `[x, y, width, height]`

- `bbox_anchor_points`
  - 检测框 9 个锚点的像素坐标，坐标类型为 `float`
  - 锚点含义依次对应：上左、上中、上右、中左、中中、中右、下左、下中、下右
  - 返回字段结构如下：

```json
"bbox_anchor_points": {
  "top_left": [x1, y1],
  "top_center": [x_mid, y1],
  "top_right": [x2, y1],
  "middle_left": [x1, y_mid],
  "middle_center": [x_mid, y_mid],
  "middle_right": [x2, y_mid],
  "bottom_left": [x1, y2],
  "bottom_center": [x_mid, y2],
  "bottom_right": [x2, y2]
}
```

- `keypoint_count`
  - 有效关键点数量

## 部署/使用注意事项

1. 必须存在对应的 `config/camera_params_{camera_id}.yaml`，否则检测接口会返回配置错误。
2. `faceImage/` 是人脸库，`identity/` 是 ReID 身份库。
3. `identity/` 中纯数字命名目录会被视为临时身份缓存，不参与已知身份库加载。
