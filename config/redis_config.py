"""
Redis 中央身份记忆库配置。

所有配置项均支持环境变量覆盖，并可通过 dict 在代码中覆盖。
优先级：代码传入 dict > 环境变量 > 默认值。
"""

import os


def _env_int(name: str, default: int) -> int:
    val = os.getenv(name)
    return int(val) if val else default


def _env_float(name: str, default: float) -> float:
    val = os.getenv(name)
    return float(val) if val else default


# Redis 连接
REDIS_HOST = os.getenv("REDIS_HOST", "127.0.0.1")
REDIS_PORT = _env_int("REDIS_PORT", 6379)
REDIS_DB = _env_int("REDIS_DB", 0)
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "redisForPersonTracking")
REDIS_KEY_PREFIX = os.getenv("REDIS_KEY_PREFIX", "person_tracking")

# TTL 设置
UNKNOWN_TTL_SECONDS = _env_int("UNKNOWN_TTL_SECONDS", 300)
REDIS_CLAIM_LOCK_TTL_MS = _env_int("REDIS_CLAIM_LOCK_TTL_MS", 3000)

# 向量维度（从模型输出推断）
KNOWN_FACE_DIM = 512
KNOWN_REID_DIM = 768
UNKNOWN_FACE_DIM = 512
UNKNOWN_REID_DIM = 768

# 容量上限（对齐当前代码中的常量）
MAX_KNOWN_REID_SAMPLES = 12
MAX_KNOWN_ANCHOR_SAMPLES = 2
MAX_UNKNOWN_FACE_SAMPLES = 3
MAX_UNKNOWN_REID_SAMPLES = 3

# 搜索设置
VECTOR_SEARCH_K = 100

# 去重阈值（cosine similarity）
DEDUP_THRESHOLD = 0.95

# 锁重试设置
CLAIM_LOCK_RETRY_SLEEP_MS = 50
CLAIM_LOCK_MAX_RETRIES = 60

# 连接池设置
REDIS_POOL_MAX_CONNECTIONS = 10
REDIS_SOCKET_CONNECT_TIMEOUT = 2
REDIS_SOCKET_TIMEOUT = 2
