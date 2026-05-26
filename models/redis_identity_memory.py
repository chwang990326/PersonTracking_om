"""
Redis 中央身份记忆库。

维护四类向量空间：
  - idx:known_face    (512-dim, 人脸)
  - idx:known_reid    (768-dim, ReID)
  - idx:unknown_face  (512-dim, 人脸)
  - idx:unknown_reid  (768-dim, ReID)

每个向量空间对应一个 RediSearch HASH 索引，通过 FT.SEARCH + KNN 进行向量检索。
Unknown 实体通过 EXPIRE + last_seen 维护生命周期。
创建使用 SETNX claim lock + 二次检索保证并发安全。
"""

import hashlib
import json
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import redis


class RedisUnavailableError(Exception):
    """Redis 不可用时抛出。"""


class RedisIdentityMemory:
    """Redis 中央身份记忆库。"""

    # Lua 脚本：安全释放锁（验证 token 后再 DEL）
    UNLOCK_SCRIPT = """
    if redis.call('GET', KEYS[1]) == ARGV[1] then
        return redis.call('DEL', KEYS[1])
    else
        return 0
    end
    """

    # Lua 脚本：touch unknown（批量 EXPIRE + 更新 last_seen）
    TOUCH_UNKNOWN_SCRIPT = """
    local entity_key = KEYS[1]
    local sample_set_key = KEYS[2]
    local ttl = tonumber(ARGV[1])
    local now = tonumber(ARGV[2])
    redis.call('EXPIRE', entity_key, ttl)
    redis.call('HSET', entity_key, 'last_seen', now)
    local samples = redis.call('SMEMBERS', sample_set_key)
    for _, sample_key in ipairs(samples) do
        redis.call('EXPIRE', sample_key, ttl)
    end
    redis.call('EXPIRE', sample_set_key, ttl)
    return #samples
    """

    def __init__(self, config: dict = None):
        """
        config dict 键（可选，覆盖环境变量/默认值）:
          REDIS_HOST, REDIS_PORT, REDIS_DB, REDIS_PASSWORD,
          REDIS_KEY_PREFIX, UNKNOWN_TTL_SECONDS, REDIS_CLAIM_LOCK_TTL_MS,
          KNOWN_FACE_DIM, KNOWN_REID_DIM, UNKNOWN_FACE_DIM, UNKNOWN_REID_DIM,
          VECTOR_SEARCH_K, MAX_KNOWN_REID_SAMPLES, MAX_KNOWN_ANCHOR_SAMPLES,
          MAX_UNKNOWN_FACE_SAMPLES, MAX_UNKNOWN_REID_SAMPLES,
          DEDUP_THRESHOLD, CLAIM_LOCK_RETRY_SLEEP_MS, CLAIM_LOCK_MAX_RETRIES,
        """
        from config.redis_config import (
            REDIS_HOST,
            REDIS_PORT,
            REDIS_DB,
            REDIS_PASSWORD,
            REDIS_KEY_PREFIX,
            UNKNOWN_TTL_SECONDS,
            REDIS_CLAIM_LOCK_TTL_MS,
            KNOWN_FACE_DIM,
            KNOWN_REID_DIM,
            UNKNOWN_FACE_DIM,
            UNKNOWN_REID_DIM,
            VECTOR_SEARCH_K,
            MAX_KNOWN_REID_SAMPLES,
            MAX_KNOWN_ANCHOR_SAMPLES,
            MAX_UNKNOWN_FACE_SAMPLES,
            MAX_UNKNOWN_REID_SAMPLES,
            DEDUP_THRESHOLD,
            CLAIM_LOCK_RETRY_SLEEP_MS,
            CLAIM_LOCK_MAX_RETRIES,
            REDIS_POOL_MAX_CONNECTIONS,
            REDIS_SOCKET_CONNECT_TIMEOUT,
            REDIS_SOCKET_TIMEOUT,
        )

        self._config = config or {}
        self.host = self._config.get("REDIS_HOST", REDIS_HOST)
        self.port = self._config.get("REDIS_PORT", REDIS_PORT)
        self.db = self._config.get("REDIS_DB", REDIS_DB)
        self.password = self._config.get("REDIS_PASSWORD", REDIS_PASSWORD)
        self.key_prefix = self._config.get("REDIS_KEY_PREFIX", REDIS_KEY_PREFIX)
        self.unknown_ttl = self._config.get("UNKNOWN_TTL_SECONDS", UNKNOWN_TTL_SECONDS)
        self.claim_lock_ttl_ms = self._config.get("REDIS_CLAIM_LOCK_TTL_MS", REDIS_CLAIM_LOCK_TTL_MS)

        self.known_face_dim = self._config.get("KNOWN_FACE_DIM", KNOWN_FACE_DIM)
        self.known_reid_dim = self._config.get("KNOWN_REID_DIM", KNOWN_REID_DIM)
        self.unknown_face_dim = self._config.get("UNKNOWN_FACE_DIM", UNKNOWN_FACE_DIM)
        self.unknown_reid_dim = self._config.get("UNKNOWN_REID_DIM", UNKNOWN_REID_DIM)

        self.vector_search_k = self._config.get("VECTOR_SEARCH_K", VECTOR_SEARCH_K)
        self.max_known_reid_samples = self._config.get("MAX_KNOWN_REID_SAMPLES", MAX_KNOWN_REID_SAMPLES)
        self.max_known_anchor_samples = self._config.get("MAX_KNOWN_ANCHOR_SAMPLES", MAX_KNOWN_ANCHOR_SAMPLES)
        self.max_unknown_face_samples = self._config.get("MAX_UNKNOWN_FACE_SAMPLES", MAX_UNKNOWN_FACE_SAMPLES)
        self.max_unknown_reid_samples = self._config.get("MAX_UNKNOWN_REID_SAMPLES", MAX_UNKNOWN_REID_SAMPLES)
        self.dedup_threshold = self._config.get("DEDUP_THRESHOLD", DEDUP_THRESHOLD)
        self.claim_lock_retry_sleep_ms = self._config.get("CLAIM_LOCK_RETRY_SLEEP_MS", CLAIM_LOCK_RETRY_SLEEP_MS)
        self.claim_lock_max_retries = self._config.get("CLAIM_LOCK_MAX_RETRIES", CLAIM_LOCK_MAX_RETRIES)

        self.pool_max_connections = self._config.get("REDIS_POOL_MAX_CONNECTIONS", REDIS_POOL_MAX_CONNECTIONS)
        self.socket_connect_timeout = self._config.get("REDIS_SOCKET_CONNECT_TIMEOUT", REDIS_SOCKET_CONNECT_TIMEOUT)
        self.socket_timeout = self._config.get("REDIS_SOCKET_TIMEOUT", REDIS_SOCKET_TIMEOUT)

        # Key 前缀
        self._prefix_known_face = f"{self.key_prefix}:known_face"
        self._prefix_known_reid = f"{self.key_prefix}:known_reid"
        self._prefix_unknown_face = f"{self.key_prefix}:unknown_face"
        self._prefix_unknown_reid = f"{self.key_prefix}:unknown_reid"
        self._prefix_unknown_entity = f"{self.key_prefix}:unknown_entity"
        self._prefix_unknown_samples = f"{self.key_prefix}:unknown_samples"
        self._prefix_unknown_counter = f"{self.key_prefix}:unknown_counter"
        self._prefix_claim_lock = f"{self.key_prefix}:claim_lock"
        self._prefix_import_sig = f"{self.key_prefix}:import_sig"

        # 索引名
        self._idx_known_face = f"idx:{self.key_prefix}:known_face"
        self._idx_known_reid = f"idx:{self.key_prefix}:known_reid"
        self._idx_unknown_face = f"idx:{self.key_prefix}:unknown_face"
        self._idx_unknown_reid = f"idx:{self.key_prefix}:unknown_reid"

        self._pool: Optional[redis.ConnectionPool] = None
        self._available = False
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # 连接管理
    # ------------------------------------------------------------------

    @property
    def available(self) -> bool:
        return self._available

    def _client(self) -> redis.Redis:
        if self._pool is None:
            raise RedisUnavailableError("Redis 连接池未初始化，请先调用 connect()")
        return redis.Redis(connection_pool=self._pool)

    def connect(self) -> bool:
        """创建连接池并通过 PING 验证。返回 True 表示成功。"""
        try:
            self._pool = redis.ConnectionPool(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password or None,
                max_connections=self.pool_max_connections,
                socket_connect_timeout=self.socket_connect_timeout,
                socket_timeout=self.socket_timeout,
                retry_on_timeout=True,
                decode_responses=False,
            )
            client = self._client()
            client.ping()
            self._available = True
            print(f"[RedisIdentityMemory] 连接成功 {self.host}:{self.port} (db={self.db})")

            # 预加载 Lua 脚本
            self._unlock_sha = client.script_load(self.UNLOCK_SCRIPT)
            self._touch_unknown_sha = client.script_load(self.TOUCH_UNKNOWN_SCRIPT)

            return True
        except Exception as e:
            print(f"[RedisIdentityMemory] 连接失败: {e}")
            self._available = False
            self._pool = None
            return False

    def disconnect(self):
        if self._pool:
            self._pool.disconnect()
            self._pool = None
        self._available = False

    def ping(self) -> bool:
        """健康检查。"""
        if not self._available:
            return False
        try:
            return self._client().ping()
        except Exception:
            self._available = False
            return False

    def _ensure_available(self):
        if not self._available:
            raise RedisUnavailableError("Redis 不可用，操作被拒绝")

    # ------------------------------------------------------------------
    # 索引管理
    # ------------------------------------------------------------------

    def ensure_indexes(self) -> bool:
        """幂等创建四个向量索引。返回 True 表示全部就绪。"""
        self._ensure_available()
        # 检测向量检索能力
        self._check_redis_search_capability()

        indexes = [
            (self._idx_known_face, self._prefix_known_face, self.known_face_dim),
            (self._idx_known_reid, self._prefix_known_reid, self.known_reid_dim),
            (self._idx_unknown_face, self._prefix_unknown_face, self.unknown_face_dim),
            (self._idx_unknown_reid, self._prefix_unknown_reid, self.unknown_reid_dim),
        ]

        client = self._client()
        for idx_name, prefix, dim in indexes:
            self._create_index_if_missing(client, idx_name, prefix, dim)

        print(f"[RedisIdentityMemory] 四个向量索引已就绪")
        return True

    def _check_redis_search_capability(self):
        """检测 Redis 是否支持 RediSearch 向量检索。"""
        client = self._client()
        try:
            client.execute_command("FT._LIST")
        except Exception as e:
            raise RedisUnavailableError(
                f"Redis 不支持向量检索 (FT.* 命令不可用): {e}"
            )

    def _create_index_if_missing(self, client: redis.Redis, idx_name: str, prefix: str, dim: int):
        try:
            client.execute_command("FT.INFO", idx_name)
            print(f"[RedisIdentityMemory] 索引 {idx_name} 已存在，跳过创建")
            return
        except Exception:
            pass

        try:
            client.execute_command(
                "FT.CREATE",
                idx_name,
                "ON", "HASH",
                "PREFIX", "1", f"{prefix}:",
                "SCHEMA",
                "embedding", "VECTOR", "FLAT", "6",
                "TYPE", "FLOAT32",
                "DIM", str(dim),
                "DISTANCE_METRIC", "COSINE",
            )
            print(f"[RedisIdentityMemory] 索引 {idx_name} 创建成功 (dim={dim})")
        except Exception as e:
            raise RedisUnavailableError(f"创建索引 {idx_name} 失败: {e}")

    def drop_indexes(self):
        """删除所有四个索引。"""
        self._ensure_available()
        client = self._client()
        for idx_name in [self._idx_known_face, self._idx_known_reid,
                         self._idx_unknown_face, self._idx_unknown_reid]:
            try:
                client.execute_command("FT.DROPINDEX", idx_name, "DD")
            except Exception:
                pass
        print("[RedisIdentityMemory] 所有索引已删除")

    # ------------------------------------------------------------------
    # 向量序列化
    # ------------------------------------------------------------------

    @staticmethod
    def _embedding_to_bytes(embedding: np.ndarray) -> bytes:
        return embedding.astype(np.float32).tobytes()

    @staticmethod
    def _bytes_to_embedding(data: bytes, dim: int) -> np.ndarray:
        arr = np.frombuffer(data, dtype=np.float32).copy()
        if len(arr) != dim:
            raise ValueError(f"向量维度不匹配: 期望 {dim}, 实际 {len(arr)}")
        return arr

    @staticmethod
    def _cosine_similarity(feat1: np.ndarray, feat2: np.ndarray) -> float:
        norm1 = np.linalg.norm(feat1)
        norm2 = np.linalg.norm(feat2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(feat1, feat2) / (norm1 * norm2))

    # ------------------------------------------------------------------
    # 通用 KNN 搜索
    # ------------------------------------------------------------------

    def _search_internal(
        self,
        index_name: str,
        embedding: np.ndarray,
        threshold: float,
        top_k: int,
        id_field: str,
    ) -> Tuple[Optional[Any], float]:
        """
        通用向量检索。
        threshold: cosine similarity 阈值（内部转为 cosine distance）
        id_field: 结果中用于分组的字段名（person_id 或 unknown_id）
        返回 (best_id, best_similarity)，未匹配返回 (None, 0.0)
        """
        self._ensure_available()
        embedding = np.asarray(embedding, dtype=np.float32).ravel()
        vec_blob = self._embedding_to_bytes(embedding)
        max_distance = 1.0 - threshold

        client = self._client()
        try:
            query_str = f"(*)=>[KNN {top_k} @embedding $vec AS dist]"
            results = client.execute_command(
                "FT.SEARCH",
                index_name,
                query_str,
                "SORTBY", "dist", "ASC",
                "RETURN", "2", id_field, "dist",
                "LIMIT", "0", str(top_k),
                "PARAMS", "2", "vec", vec_blob,
                "DIALECT", "2",
            )
        except Exception as e:
            raise RedisUnavailableError(f"FT.SEARCH 失败 (index={index_name}): {e}")

        return self._parse_knn_results(results, id_field, max_distance)

    def _parse_knn_results(
        self,
        results: list,
        id_field: str,
        max_distance: float,
    ) -> Tuple[Optional[Any], float]:
        """
        解析 FT.SEARCH KNN 返回结果。
        结果格式: [total_count, key1, [field1, val1, field2, val2], key2, [...]]
        每个命中项是一个 list: [field1, val1_bytes, field2, val2_bytes]
        按 id_field 分组，每组取最高 similarity（= 1.0 - min_distance）
        """
        if not results or results[0] == 0:
            return None, 0.0

        best_per_id: Dict[str, float] = {}

        for i in range(1, len(results), 2):
            if i + 1 >= len(results):
                break
            fields = results[i + 1]
            if not isinstance(fields, list) or len(fields) < 4:
                continue

            try:
                f0_name = fields[0].decode() if isinstance(fields[0], bytes) else fields[0]
                f0_val = fields[1]
                f1_name = fields[2].decode() if isinstance(fields[2], bytes) else fields[2]
                f1_val = fields[3]

                if f0_name == id_field:
                    id_val = f0_val.decode() if isinstance(f0_val, bytes) else f0_val
                    dist_val = f1_val
                elif f1_name == id_field:
                    id_val = f1_val.decode() if isinstance(f1_val, bytes) else f1_val
                    dist_val = f0_val
                else:
                    continue

                distance = float(dist_val)
                similarity = 1.0 - distance

                if distance <= max_distance:
                    if id_val not in best_per_id or similarity > best_per_id[id_val]:
                        best_per_id[id_val] = similarity
            except (ValueError, TypeError, IndexError):
                continue

        if not best_per_id:
            return None, 0.0

        best_id = max(best_per_id, key=best_per_id.get)
        return best_id, best_per_id[best_id]

    # ------------------------------------------------------------------
    # Known 特征导入（幂等）
    # ------------------------------------------------------------------

    def _compute_import_signature(self, entries: list) -> str:
        """计算 entries 的 SHA256 签名用于幂等检测。"""
        hasher = hashlib.sha256()
        for entry in sorted(entries, key=lambda e: (str(e.get("person_id", "")), str(e.get("filename", "")))):
            hasher.update(str(entry.get("person_id", "")).encode())
            hasher.update(str(entry.get("filename", "")).encode())
            hasher.update(b"|")
        return hasher.hexdigest()

    def _get_import_sig_key(self, import_type: str) -> str:
        return f"{self._prefix_import_sig}:{import_type}"

    def import_known_faces(self, entries: list) -> int:
        """
        批量导入 known_face 特征（幂等）。
        entries: [{"person_id": str, "embedding": np.ndarray, "filename": str}, ...]
        返回新导入数量。
        """
        return self._import_known(entries, "face", self._prefix_known_face, self.known_face_dim)

    def import_known_reid(self, entries: list) -> int:
        """
        批量导入 known_reid 特征（幂等）。
        entries: [{"person_id": str, "embedding": np.ndarray, "filename": str, "is_anchor": bool}, ...]
        返回新导入数量。
        """
        return self._import_known(entries, "reid", self._prefix_known_reid, self.known_reid_dim)

    def _import_known(self, entries: list, import_type: str, key_prefix: str, dim: int) -> int:
        """通用 known 特征导入。"""
        self._ensure_available()
        if not entries:
            return 0

        sig = self._compute_import_signature(entries)
        sig_key = self._get_import_sig_key(import_type)
        client = self._client()

        existing_sig = client.get(sig_key)
        if existing_sig and existing_sig.decode() == sig:
            print(f"[RedisIdentityMemory] import_known_{import_type}: 签名未变，跳过导入 ({len(entries)} 条)")
            return 0

        imported = 0
        batch_size = 100
        pipe = None

        for idx, entry in enumerate(entries):
            if idx % batch_size == 0:
                if pipe:
                    pipe.execute()
                pipe = client.pipeline(transaction=False)

            person_id = str(entry["person_id"])
            filename = str(entry.get("filename", f"sample_{idx}"))
            embedding = np.asarray(entry["embedding"], dtype=np.float32).ravel()

            if len(embedding) != dim:
                print(f"[RedisIdentityMemory] 跳过 {person_id}/{filename}: 维度不匹配 ({len(embedding)} != {dim})")
                continue

            sample_key = f"{key_prefix}:{person_id}:{idx}"
            mapping = {
                "person_id": person_id,
                "embedding": self._embedding_to_bytes(embedding),
                "created_at": str(time.time()),
            }

            if import_type == "reid":
                mapping["is_anchor"] = "1" if entry.get("is_anchor") else "0"

            pipe.hset(sample_key, mapping=mapping)
            imported += 1

        if pipe:
            pipe.execute()

        client.set(sig_key, sig)
        print(f"[RedisIdentityMemory] import_known_{import_type}: 导入 {imported} 条")
        return imported

    # ------------------------------------------------------------------
    # Known 检索
    # ------------------------------------------------------------------

    def search_known_face(
        self, embedding: np.ndarray, threshold: float, top_k: int = None
    ) -> Tuple[Optional[str], float]:
        """在 known_face 索引中搜索。返回 (person_id, similarity)。"""
        k = top_k or self.vector_search_k
        person_id, similarity = self._search_internal(
            self._idx_known_face, embedding, threshold, k, "person_id"
        )
        if person_id is not None:
            person_id = person_id.decode() if isinstance(person_id, bytes) else str(person_id)
        return person_id, similarity

    def search_known_reid(
        self, embedding: np.ndarray, threshold: float, top_k: int = None
    ) -> Tuple[Optional[str], float]:
        """在 known_reid 索引中搜索。返回 (person_id, similarity)。"""
        k = top_k or self.vector_search_k
        person_id, similarity = self._search_internal(
            self._idx_known_reid, embedding, threshold, k, "person_id"
        )
        if person_id is not None:
            person_id = person_id.decode() if isinstance(person_id, bytes) else str(person_id)
        return person_id, similarity

    # ------------------------------------------------------------------
    # Unknown 检索
    # ------------------------------------------------------------------

    def search_unknown_face(
        self, embedding: np.ndarray, threshold: float, top_k: int = None
    ) -> Tuple[Optional[int], float]:
        k = top_k or self.vector_search_k
        unknown_id, similarity = self._search_internal(
            self._idx_unknown_face, embedding, threshold, k, "unknown_id"
        )
        return int(unknown_id) if unknown_id is not None else None, similarity

    def search_unknown_reid(
        self, embedding: np.ndarray, threshold: float, top_k: int = None
    ) -> Tuple[Optional[int], float]:
        k = top_k or self.vector_search_k
        unknown_id, similarity = self._search_internal(
            self._idx_unknown_reid, embedding, threshold, k, "unknown_id"
        )
        return int(unknown_id) if unknown_id is not None else None, similarity

    # ------------------------------------------------------------------
    # Unknown find-or-create（并发安全）
    # ------------------------------------------------------------------

    def find_or_create_unknown(
        self, embedding: np.ndarray, threshold: float, modality: str
    ) -> int:
        """
        中央 find-or-create。并发安全。
        Args:
            embedding: 特征向量
            threshold: cosine similarity 阈值
            modality: 'face' 或 'reid'
        Returns: unknown_id
        """
        self._ensure_available()
        embedding = np.asarray(embedding, dtype=np.float32).ravel()
        threshold = float(threshold)

        search_fn = self.search_unknown_face if modality == "face" else self.search_unknown_reid

        lock_name = f"{self._prefix_claim_lock}:unknown"
        lock_token = str(uuid.uuid4())

        for attempt in range(self.claim_lock_max_retries):
            # Step 1: 检索
            existing_id, similarity = search_fn(embedding, threshold, self.vector_search_k)
            if existing_id is not None:
                return existing_id

            # Step 2: 抢锁
            token = self._acquire_claim_lock(lock_name, self.claim_lock_ttl_ms)
            if token is None:
                time.sleep(self.claim_lock_retry_sleep_ms / 1000.0)
                continue

            try:
                # Step 3: 二次检索
                existing_id, similarity = search_fn(embedding, threshold, self.vector_search_k)
                if existing_id is not None:
                    return existing_id

                # Step 4: 分配新 ID
                client = self._client()
                new_id = client.incr(self._prefix_unknown_counter)

                # Step 5: 创建实体主状态
                entity_key = f"{self._prefix_unknown_entity}:{new_id}"
                now = time.time()
                client.hset(entity_key, mapping={
                    "unknown_id": str(new_id),
                    "face_count": "0",
                    "reid_count": "0",
                    "last_seen": str(now),
                    "created_at": str(now),
                    "status": "active",
                })
                client.expire(entity_key, self.unknown_ttl)

                # Step 6: 创建样本集合
                sample_set_key = f"{self._prefix_unknown_samples}:{new_id}"
                client.expire(sample_set_key, self.unknown_ttl)

                return new_id
            finally:
                self._release_claim_lock(lock_name, token)

        raise RedisUnavailableError(
            f"find_or_create_unknown 重试 {self.claim_lock_max_retries} 次后仍无法获取锁"
        )

    def allocate_unknown_id(self) -> int:
        """简单 INCR 分配 unknown ID（调用方已确认需要新 ID）。"""
        self._ensure_available()
        client = self._client()
        new_id = client.incr(self._prefix_unknown_counter)
        entity_key = f"{self._prefix_unknown_entity}:{new_id}"
        now = time.time()
        client.hset(entity_key, mapping={
            "unknown_id": str(new_id),
            "face_count": "0",
            "reid_count": "0",
            "last_seen": str(now),
            "created_at": str(now),
            "status": "active",
        })
        client.expire(entity_key, self.unknown_ttl)
        sample_set_key = f"{self._prefix_unknown_samples}:{new_id}"
        client.expire(sample_set_key, self.unknown_ttl)
        return new_id

    def _acquire_claim_lock(self, lock_name: str, ttl_ms: int) -> Optional[str]:
        """SET NX PX 抢锁。返回 token 或 None。"""
        client = self._client()
        token = str(uuid.uuid4())
        result = client.set(lock_name, token, nx=True, px=ttl_ms)
        return token if result else None

    def _release_claim_lock(self, lock_name: str, token: str):
        """Lua 脚本安全释放锁。"""
        try:
            client = self._client()
            client.evalsha(self._unlock_sha, 1, lock_name, token)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Unknown 样本管理
    # ------------------------------------------------------------------

    def add_unknown_face_sample(self, unknown_id: int, embedding: np.ndarray) -> bool:
        return self._add_unknown_sample(
            unknown_id, embedding, self._prefix_unknown_face,
            self.unknown_face_dim, self.max_unknown_face_samples, "face_count"
        )

    def add_unknown_reid_sample(self, unknown_id: int, embedding: np.ndarray) -> bool:
        return self._add_unknown_sample(
            unknown_id, embedding, self._prefix_unknown_reid,
            self.unknown_reid_dim, self.max_unknown_reid_samples, "reid_count"
        )

    def _add_unknown_sample(
        self,
        unknown_id: int,
        embedding: np.ndarray,
        key_prefix: str,
        dim: int,
        max_samples: int,
        count_field: str,
    ) -> bool:
        self._ensure_available()
        embedding = np.asarray(embedding, dtype=np.float32).ravel()
        if len(embedding) != dim:
            return False

        unknown_id = int(unknown_id)
        client = self._client()
        entity_key = f"{self._prefix_unknown_entity}:{unknown_id}"
        sample_set_key = f"{self._prefix_unknown_samples}:{unknown_id}"

        # 检查实体是否存在
        if not client.exists(entity_key):
            # 实体不存在时自动创建
            now = time.time()
            client.hset(entity_key, mapping={
                "unknown_id": str(unknown_id),
                "face_count": "0",
                "reid_count": "0",
                "last_seen": str(now),
                "created_at": str(now),
                "status": "active",
            })

        # 去重：检查已有样本
        existing_sample_keys = client.smembers(sample_set_key)
        if existing_sample_keys:
            pipe = client.pipeline(transaction=False)
            for sk in existing_sample_keys:
                pipe.hget(sk, "embedding")
            existing_embeddings_bytes = pipe.execute()

            for eb in existing_embeddings_bytes:
                if eb is None:
                    continue
                try:
                    existing_emb = np.frombuffer(eb, dtype=np.float32).copy()
                    sim = self._cosine_similarity(embedding, existing_emb)
                    if sim > self.dedup_threshold:
                        return False
                except Exception:
                    continue

        # 容量管理
        all_samples = sorted(
            [sk.decode() if isinstance(sk, bytes) else sk for sk in (existing_sample_keys or [])],
            key=lambda k: int(k.rsplit(":", 1)[-1]) if k.rsplit(":", 1)[-1].isdigit() else 0,
        )
        if len(all_samples) >= max_samples and all_samples:
            oldest_key = all_samples[0]
            client.delete(oldest_key)
            client.srem(sample_set_key, oldest_key)

        # 生成新 sample idx
        existing_indices = []
        for sk in all_samples:
            try:
                existing_indices.append(int(sk.rsplit(":", 1)[-1]))
            except (ValueError, IndexError):
                pass
        next_idx = max(existing_indices) + 1 if existing_indices else 0

        # 写入样本
        sample_key = f"{key_prefix}:{unknown_id}:{next_idx}"
        now = time.time()
        client.hset(sample_key, mapping={
            "unknown_id": str(unknown_id),
            "embedding": self._embedding_to_bytes(embedding),
            "created_at": str(now),
        })
        client.expire(sample_key, self.unknown_ttl)

        # 更新样本集
        client.sadd(sample_set_key, sample_key)
        client.expire(sample_set_key, self.unknown_ttl)

        # 更新计数器
        current_count = len(client.smembers(sample_set_key))
        client.hset(entity_key, count_field, str(current_count))
        client.hset(entity_key, "last_seen", str(now))
        client.expire(entity_key, self.unknown_ttl)

        return True

    # ------------------------------------------------------------------
    # Unknown 生命周期管理
    # ------------------------------------------------------------------

    def touch_unknown(self, unknown_id: int):
        """更新 unknown entity 及所有样本的 TTL 和 last_seen。"""
        if unknown_id in (-1, None):
            return
        self._ensure_available()
        unknown_id = int(unknown_id)
        client = self._client()
        entity_key = f"{self._prefix_unknown_entity}:{unknown_id}"
        sample_set_key = f"{self._prefix_unknown_samples}:{unknown_id}"
        now = time.time()

        try:
            client.evalsha(
                self._touch_unknown_sha,
                2,
                entity_key,
                sample_set_key,
                str(self.unknown_ttl),
                str(now),
            )
        except Exception:
            # fallback: 逐个 EXPIRE
            client.expire(entity_key, self.unknown_ttl)
            client.hset(entity_key, "last_seen", str(now))
            client.expire(sample_set_key, self.unknown_ttl)

    def release_unknown(self, unknown_id: int):
        """删除 unknown entity 及所有关联数据。"""
        if unknown_id in (-1, None):
            return
        self._ensure_available()
        unknown_id = int(unknown_id)
        client = self._client()
        entity_key = f"{self._prefix_unknown_entity}:{unknown_id}"
        sample_set_key = f"{self._prefix_unknown_samples}:{unknown_id}"

        sample_keys = client.smembers(sample_set_key)
        if sample_keys:
            client.delete(*sample_keys)
        client.delete(sample_set_key)
        client.delete(entity_key)

    def release_if_empty(self, unknown_id: int):
        """如果 face_count 和 reid_count 均为 0，则释放。"""
        if unknown_id in (-1, None):
            return
        self._ensure_available()
        unknown_id = int(unknown_id)
        client = self._client()
        entity_key = f"{self._prefix_unknown_entity}:{unknown_id}"

        if not client.exists(entity_key):
            return

        face_count = int(client.hget(entity_key, "face_count") or 0)
        reid_count = int(client.hget(entity_key, "reid_count") or 0)
        if face_count == 0 and reid_count == 0:
            self.release_unknown(unknown_id)

    def cleanup_stale_unknowns(self) -> int:
        """扫描并清理过期/孤儿 unknown 数据。返回清理数。

        扫描样本集 keys，检查对应实体是否存在，不存在则清理孤儿样本。
        这样即使实体 key 已过期被删除，也能正确清理。
        """
        self._ensure_available()
        client = self._client()
        cleaned = 0

        sample_set_pattern = f"{self._prefix_unknown_samples}:*"
        cursor = 0
        while True:
            cursor, keys = client.scan(cursor, match=sample_set_pattern, count=100)
            for key in keys:
                if isinstance(key, bytes):
                    key = key.decode()
                try:
                    unknown_id = key.rsplit(":", 1)[-1]
                except (ValueError, IndexError):
                    continue
                entity_key = f"{self._prefix_unknown_entity}:{unknown_id}"
                if not client.exists(entity_key):
                    # 实体已过期/删除，清理孤儿样本
                    sample_keys = client.smembers(key)
                    if sample_keys:
                        client.delete(*sample_keys)
                    client.delete(key)
                    cleaned += 1
            if cursor == 0:
                break

        return cleaned

    # ------------------------------------------------------------------
    # Known 样本管理
    # ------------------------------------------------------------------

    def add_known_reid_sample(
        self, person_id: str, embedding: np.ndarray, is_anchor: bool = False
    ) -> bool:
        """添加 known ReID 样本。带容量管理。"""
        self._ensure_available()
        embedding = np.asarray(embedding, dtype=np.float32).ravel()
        if len(embedding) != self.known_reid_dim:
            return False

        person_id = str(person_id)
        client = self._client()

        # 计数已有样本
        existing = self._list_known_samples_by_person(person_id, "reid", client)
        non_anchor_keys = [k for k, a in existing if not a]
        anchor_keys = [k for k, a in existing if a]

        # 去重
        for sk, _ in existing:
            try:
                eb = client.hget(sk, "embedding")
                if eb:
                    existing_emb = np.frombuffer(eb, dtype=np.float32).copy()
                    sim = self._cosine_similarity(embedding, existing_emb)
                    if sim > self.dedup_threshold:
                        return False
            except Exception:
                continue

        # 容量管理
        if is_anchor and len(anchor_keys) >= self.max_known_anchor_samples:
            oldest = anchor_keys[0]
            client.delete(oldest)
        elif not is_anchor and len(non_anchor_keys) >= self.max_known_reid_samples:
            oldest = non_anchor_keys[0]
            client.delete(oldest)

        # 写入
        next_idx = len(existing)
        sample_key = f"{self._prefix_known_reid}:{person_id}:{next_idx}"
        client.hset(sample_key, mapping={
            "person_id": person_id,
            "is_anchor": "1" if is_anchor else "0",
            "embedding": self._embedding_to_bytes(embedding),
            "created_at": str(time.time()),
        })
        return True

    def add_known_face_sample(self, person_id: str, embedding: np.ndarray) -> bool:
        """添加 known face 样本。"""
        self._ensure_available()
        embedding = np.asarray(embedding, dtype=np.float32).ravel()
        if len(embedding) != self.known_face_dim:
            return False

        person_id = str(person_id)
        client = self._client()
        existing = self._list_known_samples_by_person(person_id, "face", client)

        # 去重
        for sk in existing:
            try:
                eb = client.hget(sk, "embedding")
                if eb:
                    existing_emb = np.frombuffer(eb, dtype=np.float32).copy()
                    sim = self._cosine_similarity(embedding, existing_emb)
                    if sim > self.dedup_threshold:
                        return False
            except Exception:
                continue

        next_idx = len(existing)
        sample_key = f"{self._prefix_known_face}:{person_id}:{next_idx}"
        client.hset(sample_key, mapping={
            "person_id": person_id,
            "embedding": self._embedding_to_bytes(embedding),
            "created_at": str(time.time()),
        })
        return True

    def _list_known_samples_by_person(
        self, person_id: str, modality: str, client: redis.Redis = None
    ) -> list:
        """返回指定已知身份的样本 key 列表 + is_anchor 标志。"""
        if client is None:
            client = self._client()
        prefix = self._prefix_known_reid if modality == "reid" else self._prefix_known_face
        pattern = f"{prefix}:{person_id}:*"
        results = []
        cursor = 0
        while True:
            cursor, keys = client.scan(cursor, match=pattern, count=100)
            for key in keys:
                key_str = key.decode() if isinstance(key, bytes) else key
                if modality == "reid":
                    is_anchor = client.hget(key_str, "is_anchor")
                    is_anchor = is_anchor == b"1" if is_anchor else False
                else:
                    is_anchor = False
                results.append((key_str, is_anchor))
            if cursor == 0:
                break
        return sorted(results, key=lambda x: x[0])

    def count_known_samples(
        self, person_id: str, modality: str, is_anchor: bool = None
    ) -> int:
        """统计已知身份的样本数。"""
        self._ensure_available()
        client = self._client()
        existing = self._list_known_samples_by_person(person_id, modality, client)
        if is_anchor is None:
            return len(existing)
        return sum(1 for _, a in existing if a == is_anchor)

    # ------------------------------------------------------------------
    # 管理方法
    # ------------------------------------------------------------------

    def clear_all(self):
        """清除所有本项目相关的 Redis key 并重建索引。"""
        self._ensure_available()
        client = self._client()
        pattern = f"{self.key_prefix}:*"
        cursor = 0
        deleted = 0
        while True:
            cursor, keys = client.scan(cursor, match=pattern, count=100)
            if keys:
                client.delete(*keys)
                deleted += len(keys)
            if cursor == 0:
                break

        self.drop_indexes()
        print(f"[RedisIdentityMemory] 已清除 {deleted} 个 key")
        self.ensure_indexes()

    def flush_unknowns(self):
        """清除所有 unknown 相关 key。"""
        self._ensure_available()
        client = self._client()
        for prefix in [self._prefix_unknown_face, self._prefix_unknown_reid,
                        self._prefix_unknown_entity, self._prefix_unknown_samples]:
            pattern = f"{prefix}:*"
            cursor = 0
            while True:
                cursor, keys = client.scan(cursor, match=pattern, count=100)
                if keys:
                    client.delete(*keys)
                if cursor == 0:
                    break
        client.delete(self._prefix_unknown_counter)
        print("[RedisIdentityMemory] unknown 数据已清除")

    def get_stats(self) -> dict:
        """获取统计信息。"""
        self._ensure_available()
        client = self._client()
        stats = {"available": self._available}

        for idx_name in [self._idx_known_face, self._idx_known_reid,
                         self._idx_unknown_face, self._idx_unknown_reid]:
            try:
                info = client.execute_command("FT.INFO", idx_name)
                info_dict = {}
                for i in range(0, len(info), 2):
                    if i + 1 < len(info):
                        k = info[i].decode() if isinstance(info[i], bytes) else str(info[i])
                        v = info[i + 1]
                        info_dict[k] = int(v) if isinstance(v, (int, float)) else v
                stats[idx_name] = info_dict.get("num_docs", 0)
            except Exception:
                stats[idx_name] = -1

        stats["unknown_counter"] = int(client.get(self._prefix_unknown_counter) or 0)

        entity_pattern = f"{self._prefix_unknown_entity}:*"
        cursor = 0
        entity_count = 0
        while True:
            cursor, keys = client.scan(cursor, match=entity_pattern, count=100)
            entity_count += len(keys)
            if cursor == 0:
                break
        stats["unknown_entity_count"] = entity_count

        return stats
