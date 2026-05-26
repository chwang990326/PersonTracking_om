"""
未知身份存储 —— Redis 中央身份记忆库的薄包装层。

所有运行时状态委托给 RedisIdentityMemory，本地仅保留线程锁。
不再使用本地目录、本地 FAISS 索引、本地 ID 分配。
"""

import threading
from typing import Optional, Tuple

import numpy as np


class UnknownEntityStore:
    """未知身份存储（委托 RedisIdentityMemory）。"""

    def __init__(
        self,
        redis_memory=None,
        ttl_seconds: float = 300.0,
        face_similarity_threshold: float = 0.5,
        **kwargs,
    ):
        self._redis = redis_memory
        self.ttl_seconds = ttl_seconds
        self.face_similarity_threshold = face_similarity_threshold
        self.lock = threading.RLock()

    # ------------------------------------------------------------------
    # ID 分配
    # ------------------------------------------------------------------

    def allocate_id(self) -> str:
        """分配全局唯一 unknown ID（委托 Redis UUID）。"""
        if self._redis is None or not self._redis.available:
            return -1
        try:
            return self._redis.allocate_unknown_id()
        except Exception as e:
            print(f"[UnknownEntityStore] allocate_id 失败: {e}")
            return -1

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def touch_entity(self, unknown_id):
        """更新 unknown 的 TTL 和 last_seen。"""
        if unknown_id in (-1, None) or self._redis is None or not self._redis.available:
            return
        try:
            self._redis.touch_unknown(str(unknown_id))
        except Exception as e:
            print(f"[UnknownEntityStore] touch_entity 失败: {e}")

    def release_if_empty(self, unknown_id):
        """如果没有任何样本，则释放该 unknown。"""
        if unknown_id in (-1, None) or self._redis is None or not self._redis.available:
            return
        try:
            self._redis.release_if_empty(str(unknown_id))
        except Exception as e:
            print(f"[UnknownEntityStore] release_if_empty 失败: {e}")

    def release_entity(self, unknown_id):
        """直接释放 unknown。"""
        if unknown_id in (-1, None) or self._redis is None or not self._redis.available:
            return
        try:
            self._redis.release_unknown(str(unknown_id))
        except Exception as e:
            print(f"[UnknownEntityStore] release_entity 失败: {e}")

    def cleanup_stale(self):
        """清理过期 unknown（委托 Redis EXPIRE + 孤儿清理）。"""
        if self._redis is None or not self._redis.available:
            return
        try:
            self._redis.cleanup_stale_unknowns()
        except Exception as e:
            print(f"[UnknownEntityStore] cleanup_stale 失败: {e}")

    # ------------------------------------------------------------------
    # 人脸检索 & 样本
    # ------------------------------------------------------------------

    def search_face_embedding(self, embedding: np.ndarray) -> Tuple[Optional[str], float]:
        """在未知人脸库中搜索。返回 (unknown_id, similarity)。"""
        if self._redis is None or not self._redis.available:
            return None, 0.0
        try:
            return self._redis.search_unknown_face(embedding, self.face_similarity_threshold)
        except Exception as e:
            print(f"[UnknownEntityStore] search_face_embedding 失败: {e}")
            return None, 0.0

    def add_face_sample(self, unknown_id, face_image, embedding: np.ndarray) -> bool:
        """
        添加人脸样本到 unknown。
        face_image 参数保留用于接口兼容，实际不再写盘，仅存储特征向量。
        """
        if unknown_id in (-1, None) or self._redis is None or not self._redis.available:
            return False
        if embedding is None:
            return False
        try:
            return self._redis.add_unknown_face_sample(str(unknown_id), embedding)
        except Exception as e:
            print(f"[UnknownEntityStore] add_face_sample 失败: {e}")
            return False

    # ------------------------------------------------------------------
    # 全局清理
    # ------------------------------------------------------------------

    def clear_all(self):
        """清空所有 unknown 数据。"""
        if self._redis is None or not self._redis.available:
            return
        try:
            self._redis.flush_unknowns()
        except Exception as e:
            print(f"[UnknownEntityStore] clear_all 失败: {e}")
