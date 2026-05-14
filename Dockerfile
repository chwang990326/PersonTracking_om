FROM swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:8.3.rc1-310p-openeuler24.03-py3.11

USER root
WORKDIR /app

# 需要代理时再打开，默认不要把账号密码硬编码进镜像
ENV HTTP_PROXY="http://wangchenhao:wangchenhao@192.168.100.222:7890"
ENV HTTPS_PROXY="http://wangchenhao:wangchenhao@192.168.100.222:7890"

RUN yum install -y zlib-devel git mesa-libGL && yum clean all

COPY requirements.txt /app/
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . /app

RUN pip3 install --no-cache-dir wheel setuptools --break-system-packages && \
    pip3 install --no-cache-dir /app/third_party/aclruntime-0.0.2-cp311-cp311-linux_aarch64.whl --break-system-packages && \
    pip3 install --no-cache-dir /app/third_party/ais_bench-0.0.2-py3-none-any.whl --break-system-packages

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