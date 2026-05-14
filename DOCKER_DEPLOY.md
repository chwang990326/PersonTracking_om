# PersonTracking_om Docker 部署说明

本文档基于当前仓库代码和已有的 Ascend/NPU 部署笔记整理，目标是把 `PersonTracking_om` 部署到 Docker 容器中，并为后续多 worker 性能测试留出可执行方案。

## 1. 当前项目的部署要点

- 服务入口：`api_server.py`
- 当前默认服务端口：`8130`
- 当前 OM 推理入口：
  - `weights/yolo26x.om`
  - `weights/yolo26s-pose.om`
  - `weights/transformer_120_16.om`
  - `weights/best094nophone.om`
  - `weights/det_10_640.om`
  - `weights/adaface_ir50_ms1mv2_b1.om`
- Ascend 推理封装：`models/ascend_backend.py`
- 当前代码会优先加载 `.om` 模型；若 `.om` 文件存在，`resolve_model_path()` 会优先走 OM 路径。

## 2. 部署前准备

### 2.1 宿主机要求

- 已安装 Ascend 驱动
- 已安装 CANN / Ascend Toolkit
- 宿主机可以正常执行：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
npu-smi info
```

- 宿主机存在以下设备节点：

```bash
/dev/davinci0
/dev/davinci_manager
/dev/devmm_svm
/dev/hisi_hdc
```

如果后续需要绑定其他 NPU 卡，例如第 1 张卡、第 2 张卡，需要补充对应的 `/dev/davinciX`。

### 2.2 项目目录建议

假设项目目录为：

```bash
/home/wangchenhao/PersonTracking_om
```

建议至少保证以下目录完整：

```bash
weights/
config/
identity/
faceImage/
database/
video/
```

## 3. OM 模型准备

当前代码依赖的 OM 文件名需要与仓库中的路径保持一致。下面的 `atc` 命令建议输出成当前代码实际使用的文件名。

### 3.1 UniFormer 行为识别

```bash
atc \
  --model=./best094nophone.onnx \
  --framework=5 \
  --output=./best094nophone \
  --soc_version=Ascend310P3 \
  --input_shape="input:1,3,16,224,224" \
  --log=debug
```

生成后放到：

```bash
weights/best094nophone.om
```

### 3.2 ReID

```bash
atc \
  --model=./transformer_120.onnx \
  --framework=5 \
  --output=./transformer_120_16 \
  --soc_version=Ascend310P3 \
  --input_shape="images:1,3,256,128" \
  --input_format=NCHW \
  --log=info
```

生成后放到：

```bash
weights/transformer_120_16.om
```

### 3.3 人脸检测 SCRFD

```bash
atc \
  --model=./det_10g.onnx \
  --framework=5 \
  --output=./det_10_640 \
  --soc_version=Ascend310P3 \
  --input_shape="input.1:1,3,640,640" \
  --input_format=NCHW \
  --log=info
```

生成后放到：

```bash
weights/det_10_640.om
```

### 3.4 AdaFace

```bash
atc \
  --model=./adaface_ir50_ms1mv2.onnx \
  --framework=5 \
  --output=./adaface_ir50_ms1mv2_b1 \
  --soc_version=Ascend310P3 \
  --input_shape="input:1,3,112,112" \
  --input_format=NCHW \
  --log=info
```

生成后放到：

```bash
weights/adaface_ir50_ms1mv2_b1.om
```

### 3.5 YOLO 检测

```bash
atc \
  --model=./yolo26x.onnx \
  --framework=5 \
  --output=./yolo26x \
  --soc_version=Ascend310P3 \
  --input_shape="images:1,3,640,640" \
  --log=debug
```

生成后放到：

```bash
weights/yolo26x.om
```

### 3.6 YOLO Pose

```bash
atc \
  --model=./yolo26s-pose.onnx \
  --framework=5 \
  --output=./yolo26s-pose \
  --soc_version=Ascend310P3 \
  --input_shape="images:1,3,640,640" \
  --log=debug
```

生成后放到：

```bash
weights/yolo26s-pose.om
```

## 4. Dockerfile 建议

推荐基于 Ascend 官方 CANN 基础镜像构建。下面是适配当前项目的参考 Dockerfile。

```dockerfile
FROM swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:8.3.rc1-310p-openeuler24.03-py3.11

USER root
WORKDIR /app

# 需要代理时再打开，默认不要把账号密码硬编码进镜像
# ENV HTTP_PROXY=http://<proxy-host>:<port>
# ENV HTTPS_PROXY=http://<proxy-host>:<port>

RUN yum install -y zlib-devel git mesa-libGL && yum clean all

COPY requirements.txt /app/
RUN pip3 install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --break-system-packages

RUN pip3 install --no-cache-dir wheel setuptools --break-system-packages && \
    pip3 install --no-cache-dir git+https://gitee.com/ascend/tools.git#subdirectory=ais-bench_workload/tool/ais_bench/backend --break-system-packages && \
    pip3 install --no-cache-dir git+https://gitee.com/ascend/tools.git#subdirectory=ais-bench_workload/tool/ais_bench --break-system-packages

COPY . /app

RUN echo "/usr/local/Ascend/ascend-toolkit/latest/runtime/lib64" >> /etc/ld.so.conf.d/ascend.conf && \
    echo "/usr/local/Ascend/ascend-toolkit/latest/compiler/lib64" >> /etc/ld.so.conf.d/ascend.conf && \
    echo "/usr/local/Ascend/driver/lib64" >> /etc/ld.so.conf.d/ascend.conf && \
    echo "/usr/local/Ascend/driver/lib64/common" >> /etc/ld.so.conf.d/ascend.conf && \
    echo "/usr/local/Ascend/driver/lib64/driver" >> /etc/ld.so.conf.d/ascend.conf && \
    ldconfig

ENV LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/ascend-toolkit/latest/runtime/lib64:${LD_LIBRARY_PATH:-}
ENV PYTHONPATH=/usr/local/Ascend/ascend-toolkit/latest/pyACL/python/site-packages/acl:${PYTHONPATH:-}
ENV MODEL_BACKEND=om
ENV ASCEND_DEVICE_ID=0

CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8130", "--workers", "1"]
```

## 5. 构建镜像

在项目根目录执行：

```bash
sudo docker build --network host -t person_tracking_om:v1 .
```

如果需要导出：

```bash
sudo docker save -o person_tracking_om_v1.tar person_tracking_om:v1
```

## 6. 单容器单 worker 运行

先从单 worker 跑通，再做并发性能测试。

```bash
sudo docker run -itd \
  --name person_tracking_om_1w \
  --privileged \
  --device /dev/davinci0 \
  --device /dev/davinci_manager \
  --device /dev/devmm_svm \
  --device /dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /var/log/npu:/var/log/npu \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/Ascend/add-ons:/usr/local/Ascend/add-ons \
  -v /usr/slog:/usr/slog \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /home/wangchenhao/PersonTracking_om:/app \
  -p 8130:8130 \
  -e MODEL_BACKEND=om \
  -e ASCEND_DEVICE_ID=0 \
  person_tracking_om:v1
```

说明：

- `-v /home/wangchenhao/PersonTracking_om:/app` 用于直接挂载代码和模型，便于调试
- 如果要做纯镜像发布，可以不挂载源码，而是在 `docker build` 时把项目打进镜像
- 当前端口以 `8130` 为准，因为仓库里的 `api_server.py` 当前默认就是这个端口

查看日志：

```bash
sudo docker logs -f person_tracking_om_1w
```

进入容器：

```bash
sudo docker exec -it person_tracking_om_1w bash
```

## 7. 多 worker 性能测试建议

### 7.1 直接开多个 Uvicorn worker

可以把 Dockerfile 或运行命令中的 `--workers 1` 改成更大的值，例如：

```bash
uvicorn api_server:app --host 0.0.0.0 --port 8130 --workers 2
```

但要注意：

- 每个 worker 都会独立加载一套模型
- 每个 worker 都会独立初始化 Ascend 推理会话
- NPU 显存 / 内存占用会按 worker 数量近似叠加
- `identity/`、`database/`、`unknownFace/`、`unknownIdentity/` 这类目录会被多个 worker 共享，压测前要明确是否允许共享读写

因此更稳妥的顺序是：

1. 先测试 `workers=1`
2. 再测试 `workers=2`
3. 观察启动时间、NPU 内存占用、吞吐和错误率
4. 再决定是否继续增加 worker

### 7.2 更推荐的压测方式

如果后续目标是测吞吐，而不是单进程极限，建议优先考虑：

- 单容器单 worker，多容器并发
- 或单 worker 绑定单卡，多实例分卡部署

这样更容易定位瓶颈，也更符合 Ascend 设备的资源隔离方式。

例如不同端口启动多个实例：

```bash
sudo docker run -itd \
  --name person_tracking_om_a \
  --device /dev/davinci0 \
  --device /dev/davinci_manager \
  --device /dev/devmm_svm \
  --device /dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /home/wangchenhao/PersonTracking_om:/app \
  -p 8131:8130 \
  -e MODEL_BACKEND=om \
  -e ASCEND_DEVICE_ID=0 \
  person_tracking_om:v1
```

```bash
sudo docker run -itd \
  --name person_tracking_om_b \
  --device /dev/davinci0 \
  --device /dev/davinci_manager \
  --device /dev/devmm_svm \
  --device /dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /home/wangchenhao/PersonTracking_om:/app \
  -p 8132:8130 \
  -e MODEL_BACKEND=om \
  -e ASCEND_DEVICE_ID=0 \
  person_tracking_om:v1
```

如果服务器上有多张卡，推荐把不同容器绑到不同的 `ASCEND_DEVICE_ID`。

## 8. 服务可用性检查

### 8.1 健康检查思路

当前项目没有单独的 `/health` 接口，最简单的检查方式是直接请求业务接口，或查看日志是否成功加载 OM 模型。

日志中至少应看到：

- `acl init success`
- `open device 0 success`
- `load model ...om success`

### 8.2 本地接口测试

项目里已有测试脚本：

- `test_api.py`：视频调用 `POST /api/v1/person/detect`
- `test_face_verify.py`：测试 `/third/face/verify`

例如：

```bash
python test_api.py --max-frames 20
```

## 9. 常见问题

### 9.1 看到日志里写 `cpu`，是否说明没有走 NPU

不是。

当前代码在 `service.py` 中把 PyTorch 侧设备显示为 `cpu`，但 `.om` 模型实际通过 `ais_bench` 的 `InferSession` 在 Ascend NPU 上运行。是否走到 NPU，要看：

- `acl init success`
- `open device X success`
- `load model ...om success`

### 9.2 端口占用

如果出现：

```text
address already in use
```

说明宿主机端口已被占用。处理方式：

```bash
ss -lntp | grep 8130
```

或者修改映射端口，例如：

```bash
-p 9130:8130
```

### 9.3 `502 Bad Gateway`

如果测试脚本请求本地接口时返回 `502`，优先检查：

- 是否误走了系统代理 `HTTP_PROXY` / `HTTPS_PROXY`
- 是否请求到了外层网关而不是本机 `127.0.0.1`
- 容器内服务是否真的监听在 `8130`

## 10. 建议的落地顺序

1. 先确认所有 `.om` 文件名与代码一致
2. 用 `workers=1` 跑通容器
3. 用 `test_api.py` 验证单路视频
4. 记录单 worker 的时延、吞吐、NPU 占用
5. 再增加 worker 或增加容器实例做压测

## 11. 可选后续工作

后续如果要把部署流程固定下来，建议继续补齐：

- 仓库内正式 `Dockerfile`
- `.dockerignore`
- `docker-compose.yml` 或等价启动脚本
- 独立 `/health` 接口
- 压测记录模板
- 单 worker / 多 worker 的对比数据文档
