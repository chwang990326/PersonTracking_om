#!/usr/bin/env bash
set -euo pipefail

# =========================
# PersonTracking 集群关闭脚本
# =========================
#
# 默认关闭：
#   1. Gateway 进程
#   2. 算法 Docker 容器
#
# 默认不关闭 Redis：
#   Redis 保存 camera_id -> pipeline 的路由关系。
#   现在 Gateway 已经给路由加了 TTL，默认 5 分钟无请求会自动释放。
#   因此 Redis 可以长期运行。
#
# 如果你确实要一起停 Redis：
#   STOP_REDIS=1 ./stop_all_services.sh

# 项目目录，用于找到 gateway.pid。
PROJECT_DIR="${PROJECT_DIR:-/home/wangchenhao/PersonTracking_om}"

# 要关闭哪些 NPU 对应的算法容器，必须和启动时的 NPU_DEVICES 一致。
NPU_DEVICES="${NPU_DEVICES:-0,1}"

# Gateway 端口，用于兜底查杀 Gateway 进程。
GATEWAY_PORT="${GATEWAY_PORT:-8130}"

# 算法 Docker 容器名前缀，最终容器名类似 person_tracking_om_d0。
CONTAINER_PREFIX="${CONTAINER_PREFIX:-person_tracking_om_d}"

# Redis 容器名。
REDIS_CONTAINER="${REDIS_CONTAINER:-gateway-redis}"

# 默认不停止 Redis。设置 STOP_REDIS=1 才会停止。
STOP_REDIS="${STOP_REDIS:-0}"

# Gateway 启动脚本写入的 pid 文件。
GATEWAY_PID_FILE="${GATEWAY_PID_FILE:-${PROJECT_DIR}/gateway.pid}"

# 统一使用 sudo docker。
DOCKER=(sudo docker)

log() {
  printf '[stop] %s\n' "$*"
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
}

# 先根据 gateway.pid 停 Gateway，再按端口做一次兜底清理。
stop_gateway() {
  if [[ -f "$GATEWAY_PID_FILE" ]]; then
    local pid
    pid="$(cat "$GATEWAY_PID_FILE" 2>/dev/null || true)"
    if [[ -n "$pid" ]] && kill -0 "$pid" >/dev/null 2>&1; then
      kill_pids "gateway pid file process" "$pid"
    fi
    rm -f "$GATEWAY_PID_FILE"
  fi

  log "stopping gateway processes on port ${GATEWAY_PORT}"
  mapfile -t gateway_pids < <(find_gateway_pids)
  kill_pids "gateway processes" "${gateway_pids[@]}"
}

# 删除算法 Docker 容器。
stop_algorithm_containers() {
  local device name
  for device in "${DEVICES[@]}"; do
    name="${CONTAINER_PREFIX}${device}"
    if container_exists "$name"; then
      log "removing algorithm container: $name"
      "${DOCKER[@]}" rm -f "$name" >/dev/null
    else
      log "algorithm container not found: $name"
    fi
  done
}

# 默认保留 Redis；只有 STOP_REDIS=1 时才删除 Redis 容器。
stop_redis_if_requested() {
  if [[ "$STOP_REDIS" != "1" ]]; then
    log "redis left running; set STOP_REDIS=1 to remove ${REDIS_CONTAINER}"
    return
  fi

  if container_exists "$REDIS_CONTAINER"; then
    log "removing redis container: $REDIS_CONTAINER"
    "${DOCKER[@]}" rm -f "$REDIS_CONTAINER" >/dev/null || true
  else
    log "redis container not found: $REDIS_CONTAINER"
  fi
}

main() {
  parse_devices
  stop_gateway
  stop_algorithm_containers
  stop_redis_if_requested
  log "all requested services stopped"
}

main "$@"
