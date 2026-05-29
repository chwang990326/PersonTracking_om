#!/usr/bin/env bash
set -euo pipefail

# =========================
# PersonTracking 集群启动脚本
# =========================
#
# 这个脚本负责一次性启动：
#   1. Redis 路由缓存
#   2. 算法 Docker 容器
#   3. 每个 Docker 内的多个 uvicorn 单 worker 服务
#   4. Gateway 网关服务
#
# 常用改法：
#   - 改 NPU_DEVICES 控制启动哪些 NPU，例如 "0,1" 或 "0,1,2,3"
#   - 改 WORKERS_PER_DOCKER 控制每张卡的 Docker 内启动几个 worker
#   - 改 START_PORT 控制算法 worker 端口起点
#
# 端口分配规则：
#   port = START_PORT + device_index + worker_index * device_count
#
# 例子：
#   NPU_DEVICES="0,1" WORKERS_PER_DOCKER=3 START_PORT=8131
#
#   npu0_worker0 -> 8131
#   npu1_worker0 -> 8132
#   npu0_worker1 -> 8133
#   npu1_worker1 -> 8134
#   npu0_worker2 -> 8135
#   npu1_worker2 -> 8136

# 项目目录。服务器上代码挂载/运行的根目录。
PROJECT_DIR="${PROJECT_DIR:-/home/wangchenhao/PersonTracking_om}"

# 算法镜像名。
IMAGE_NAME="${IMAGE_NAME:-person_tracking_full:v2}"

# 要启用的 NPU 卡号，逗号分隔。
NPU_DEVICES="${NPU_DEVICES:-0,1}"

# 每张 NPU 对应的 Docker 内启动几个单 worker uvicorn 进程。
WORKERS_PER_DOCKER="${WORKERS_PER_DOCKER:-2}"

# 算法 worker 对外暴露端口起点。
START_PORT="${START_PORT:-8131}"

# Gateway 对外服务端口。
GATEWAY_PORT="${GATEWAY_PORT:-8130}"

# 算法 Docker 容器名前缀，最终容器名类似 person_tracking_om_d0。
CONTAINER_PREFIX="${CONTAINER_PREFIX:-person_tracking_om_d}"

# Redis 容器配置。Redis 用于保存 camera_id -> pipeline 的路由关系。
REDIS_CONTAINER="${REDIS_CONTAINER:-gateway-redis}"
REDIS_IMAGE="${REDIS_IMAGE:-redis:8.0-alpine}"
REDIS_HOST_PORT="${REDIS_HOST_PORT:-6379}"
REDIS_PASSWORD="${REDIS_PASSWORD:-redisForPersonTracking}"
REDIS_DB="${REDIS_DB:-1}"
REDIS_KEY_PREFIX="${REDIS_KEY_PREFIX:-gateway}"
HOST_GATEWAY_NAME="${HOST_GATEWAY_NAME:-host.docker.internal}"

# Gateway 配置文件由本脚本自动生成，Gateway 启动时读取它。
GATEWAY_CONFIG="${GATEWAY_CONFIG:-${PROJECT_DIR}/config/gateway_pipelines.json}"

# Gateway 日志和 pid 文件。
GATEWAY_PID_FILE="${GATEWAY_PID_FILE:-${PROJECT_DIR}/gateway.pid}"

# 等待后端/Gateway 启动成功的最长秒数。
WAIT_SECONDS="${WAIT_SECONDS:-180}"

# 挂载到算法 Docker /app 的宿主机目录。默认就是项目目录。
APP_MOUNT="${APP_MOUNT:-${PROJECT_DIR}}"

# 挂载到算法 Docker /app/faceImage 的宿主机人脸库目录。
# 会放在 -v ${APP_MOUNT}:/app 后面，确保覆盖 /app/faceImage。
FACE_IMAGE_MOUNT="${FACE_IMAGE_MOUNT:-/home/wangchenhao/PersonTracking_om/faceImage}"

# 性能耗时统计默认关闭。
# 启动时设置 ENABLE_PROFILING=1 才会传入算法 Docker。
# PROFILE_LOG_EVERY 表示每多少个请求打印一次聚合耗时日志。
ENABLE_PROFILING="${ENABLE_PROFILING:-0}"
PROFILE_LOG_EVERY="${PROFILE_LOG_EVERY:-100}"

# 启动 Gateway 使用的 Python 命令。如果在 conda 环境里运行脚本，默认 python 即可。
# 统一使用 sudo docker。
DOCKER=(sudo docker)

log() {
  printf '[start] %s\n' "$*"
}

trim() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "$value"
}

container_exists() {
  local name="$1"
  "${DOCKER[@]}" ps -a --format '{{.Names}}' | grep -Fxq "$name"
}

container_running() {
  local name="$1"
  "${DOCKER[@]}" ps --format '{{.Names}}' | grep -Fxq "$name"
}

find_gateway_pids() {
  ps -eo pid=,args= \
    | awk -v port="$GATEWAY_PORT" '
        /uvicorn/ && /gateway_server:app/ && $0 ~ "--port " port {
          print $1
        }
      ' \
    | grep -v "^$$$" || true
}

kill_pids() {
  local label="$1"
  shift
  local pids=("$@")
  if [[ "${#pids[@]}" -eq 0 ]]; then
    return
  fi

  log "stopping ${label}: ${pids[*]}"
  kill "${pids[@]}" >/dev/null 2>&1 || true
  sleep 2

  local alive=()
  local pid
  for pid in "${pids[@]}"; do
    if kill -0 "$pid" >/dev/null 2>&1; then
      alive+=("$pid")
    fi
  done

  if [[ "${#alive[@]}" -gt 0 ]]; then
    log "force stopping ${label}: ${alive[*]}"
    kill -9 "${alive[@]}" >/dev/null 2>&1 || true
  fi
}

wait_url() {
  local url="$1"
  local name="$2"
  local i
  for ((i = 1; i <= WAIT_SECONDS; i++)); do
    if curl --noproxy '*' -fsS "$url" >/dev/null 2>&1; then
      log "$name is ready: $url"
      return 0
    fi
    sleep 1
  done
  log "ERROR: timed out waiting for $name: $url"
  return 1
}

# 停掉旧 Gateway，避免端口 8130 被占用。
stop_existing_gateway() {
  if [[ -f "$GATEWAY_PID_FILE" ]]; then
    local pid
    pid="$(cat "$GATEWAY_PID_FILE" 2>/dev/null || true)"
    if [[ -n "$pid" ]] && kill -0 "$pid" >/dev/null 2>&1; then
      kill_pids "existing gateway pid file process" "$pid"
    fi
    rm -f "$GATEWAY_PID_FILE"
  fi

  mapfile -t gateway_pids < <(find_gateway_pids)
  kill_pids "existing gateway processes" "${gateway_pids[@]}"
}

# 启动 Redis。如果容器已经存在，就复用；如果不存在，就用 redis:8.0-alpine 创建。
start_redis() {
  if container_exists "$REDIS_CONTAINER"; then
    if container_running "$REDIS_CONTAINER"; then
      log "redis container already running: $REDIS_CONTAINER"
    else
      log "starting existing redis container: $REDIS_CONTAINER"
      "${DOCKER[@]}" start "$REDIS_CONTAINER" >/dev/null
    fi
  else
    log "creating redis container: $REDIS_CONTAINER"
    "${DOCKER[@]}" run -d \
      --name "$REDIS_CONTAINER" \
      --restart=always \
      -p "${REDIS_HOST_PORT}:6379" \
      "$REDIS_IMAGE" \
      redis-server --requirepass "$REDIS_PASSWORD" >/dev/null
  fi

  local i
  for ((i = 1; i <= 30; i++)); do
    if "${DOCKER[@]}" exec "$REDIS_CONTAINER" redis-cli -a "$REDIS_PASSWORD" ping >/dev/null 2>&1; then
      log "redis is ready: ${REDIS_CONTAINER} localhost:${REDIS_HOST_PORT}"
      return 0
    fi
    sleep 1
  done
  log "ERROR: redis is not ready"
  return 1
}

# 解析 NPU_DEVICES，例如 "0,1,2" -> DEVICES=(0 1 2)。
parse_devices() {
  IFS=',' read -r -a RAW_DEVICES <<< "$NPU_DEVICES"
  DEVICES=()
  local item
  for item in "${RAW_DEVICES[@]}"; do
    item="$(trim "$item")"
    if [[ -n "$item" ]]; then
      DEVICES+=("$item")
    fi
  done
  if [[ "${#DEVICES[@]}" -eq 0 ]]; then
    log "ERROR: NPU_DEVICES is empty"
    exit 1
  fi
}

# 根据卡下标和 worker 下标计算端口。
# 注意这里使用 device_index，不是实际 device id，所以即使 NPU_DEVICES="2,5" 也能连续分配端口。
port_for_worker() {
  local device_index="$1"
  local worker_index="$2"
  local device_count="$3"
  printf '%s' "$((START_PORT + device_index + worker_index * device_count))"
}

# 生成 Gateway 配置文件。
# Gateway 会读取这个文件，知道当前有哪些后端 worker 端口可以转发。
generate_gateway_config() {
  local config_dir
  config_dir="$(dirname "$GATEWAY_CONFIG")"
  mkdir -p "$config_dir"

  log "writing gateway config: $GATEWAY_CONFIG"
  {
    printf '{\n'
    printf '  "pipelines": [\n'
    local first=1
    local device_index worker_index device port pipe_id
    for ((worker_index = 0; worker_index < WORKERS_PER_DOCKER; worker_index++)); do
      for device_index in "${!DEVICES[@]}"; do
        device="${DEVICES[$device_index]}"
        port="$(port_for_worker "$device_index" "$worker_index" "${#DEVICES[@]}")"
        pipe_id="npu${device}_worker${worker_index}"
        if [[ "$first" -eq 0 ]]; then
          printf ',\n'
        fi
        first=0
        printf '    {\n'
        printf '      "id": "%s",\n' "$pipe_id"
        printf '      "url": "http://%s:%s"\n' "$HOST_GATEWAY_NAME" "$port"
        printf '    }'
      done
    done
    printf '\n'
    printf '  ]\n'
    printf '}\n'
  } > "$GATEWAY_CONFIG"
}

# 删除旧算法容器，避免容器名/端口冲突。
stop_existing_algorithm_containers() {
  local device name
  for device in "${DEVICES[@]}"; do
    name="${CONTAINER_PREFIX}${device}"
    if container_exists "$name"; then
      log "removing existing container: $name"
      "${DOCKER[@]}" rm -f "$name" >/dev/null
    fi
  done
}

# 启动某一张 NPU 对应的算法 Docker。
# 一个 Docker 内会启动 WORKERS_PER_DOCKER 个 uvicorn 进程，每个进程 workers=1 且独立端口。
start_algorithm_container() {
  local device="$1"
  local device_index="$2"
  local name="${CONTAINER_PREFIX}${device}"
  local docker_args=()
  local inner_cmd=""
  local worker_index port

  docker_args+=(
    run -d
    --name "$name"
    --restart=always
    --add-host "${HOST_GATEWAY_NAME}:host-gateway"
    --privileged
    --device "/dev/davinci${device}"
    --device /dev/davinci_manager
    --device /dev/devmm_svm
    --device /dev/hisi_hdc
    -v /usr/local/dcmi:/usr/local/dcmi
    -v /var/log/npu:/var/log/npu
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver
    -v /usr/local/Ascend/add-ons:/usr/local/Ascend/add-ons
    -v /usr/slog:/usr/slog
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi
    -v "${APP_MOUNT}:/app"
    -v "${FACE_IMAGE_MOUNT}:/app/faceImage"
    -w /app
    -e "ASCEND_DEVICE_ID=${device}"
    -e "REDIS_HOST=${HOST_GATEWAY_NAME}"
    -e "REDIS_PORT=${REDIS_HOST_PORT}"
    -e "REDIS_DB=${REDIS_DB}"
    -e "REDIS_PASSWORD=${REDIS_PASSWORD}"
    -e "REDIS_KEY_PREFIX=${REDIS_KEY_PREFIX}"
  )

  if [[ "$ENABLE_PROFILING" == "1" ]]; then
    docker_args+=(
      -e ENABLE_PROFILING=1
      -e "PROFILE_LOG_EVERY=${PROFILE_LOG_EVERY}"
    )
  fi

  for ((worker_index = 0; worker_index < WORKERS_PER_DOCKER; worker_index++)); do
    port="$(port_for_worker "$device_index" "$worker_index" "${#DEVICES[@]}")"
    docker_args+=(-p "${port}:${port}")
    inner_cmd+="uvicorn api_server:app --host 0.0.0.0 --port ${port} --workers 1 & "
  done

  if [[ "$device_index" -eq 0 ]]; then
    docker_args+=(
      -p "${GATEWAY_PORT}:${GATEWAY_PORT}"
      -e "GATEWAY_CONFIG=/app/config/gateway_pipelines.json"
      -e "GATEWAY_PORT=${GATEWAY_PORT}"
    )
    inner_cmd+="uvicorn gateway_server:app --host 0.0.0.0 --port ${GATEWAY_PORT} --workers 1 & "
  fi

  inner_cmd+="wait"

  docker_args+=("$IMAGE_NAME" bash -lc "$inner_cmd")

  log "starting algorithm container: $name device=$device workers=$WORKERS_PER_DOCKER"
  "${DOCKER[@]}" "${docker_args[@]}" >/dev/null
}

# 等待所有算法 worker 的 /docs 可访问。
wait_algorithm_ports() {
  local device_index worker_index port
  for device_index in "${!DEVICES[@]}"; do
    for ((worker_index = 0; worker_index < WORKERS_PER_DOCKER; worker_index++)); do
      port="$(port_for_worker "$device_index" "$worker_index" "${#DEVICES[@]}")"
      wait_url "http://127.0.0.1:${port}/docs" "backend:${port}"
    done
  done
}

# 启动 Gateway，并让它读取本脚本生成的 GATEWAY_CONFIG。
main() {
  if [[ "$WORKERS_PER_DOCKER" -le 0 ]]; then
    log "ERROR: WORKERS_PER_DOCKER must be positive"
    exit 1
  fi

  parse_devices
  mkdir -p "$PROJECT_DIR/config"

  log "project_dir=$PROJECT_DIR"
  log "image=$IMAGE_NAME"
  log "npu_devices=${DEVICES[*]}"
  log "workers_per_docker=$WORKERS_PER_DOCKER"
  log "start_port=$START_PORT"
  log "face_image_mount=$FACE_IMAGE_MOUNT"
  log "enable_profiling=$ENABLE_PROFILING"
  if [[ "$ENABLE_PROFILING" == "1" ]]; then
    log "profile_log_every=$PROFILE_LOG_EVERY"
  fi

  start_redis
  generate_gateway_config
  stop_existing_gateway
  stop_existing_algorithm_containers

  local device_index
  for device_index in "${!DEVICES[@]}"; do
    start_algorithm_container "${DEVICES[$device_index]}" "$device_index"
  done

  wait_algorithm_ports
  wait_url "http://127.0.0.1:${GATEWAY_PORT}/health" "gateway"

  log "all services started"
  log "gateway: http://127.0.0.1:${GATEWAY_PORT}/health"
  log "pipelines: http://127.0.0.1:${GATEWAY_PORT}/gateway/pipelines"
}

main "$@"
