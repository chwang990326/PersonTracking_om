"""Redis-backed identity memory for global storage, versions and locks.

The real-time recognition path should use LocalIdentityCache. Methods named
``search_*`` remain as Redis-side diagnostic fallbacks and intentionally use
plain data reads instead of becoming the service's default query path.
"""

from __future__ import annotations

import hashlib
import threading
import time
import uuid
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import redis
except ImportError:  # pragma: no cover - depends on deployment environment
    redis = None


class RedisUnavailableError(Exception):
    """Raised when Redis is unavailable for a required operation."""


class RedisIdentityMemory:
    UNLOCK_SCRIPT = """
    if redis.call('GET', KEYS[1]) == ARGV[1] then
        return redis.call('DEL', KEYS[1])
    else
        return 0
    end
    """

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

    VERSION_KINDS = ("known_face", "known_reid", "unknown_face", "unknown_reid")

    def __init__(self, config: dict = None):
        from config.redis_config import (
            CLAIM_LOCK_MAX_RETRIES,
            CLAIM_LOCK_RETRY_SLEEP_MS,
            DEDUP_THRESHOLD,
            KNOWN_FACE_DIM,
            KNOWN_REID_DIM,
            MAX_KNOWN_ANCHOR_SAMPLES,
            MAX_KNOWN_REID_SAMPLES,
            MAX_UNKNOWN_FACE_SAMPLES,
            MAX_UNKNOWN_REID_SAMPLES,
            REDIS_CLAIM_LOCK_TTL_MS,
            REDIS_DB,
            REDIS_HOST,
            REDIS_KEY_PREFIX,
            REDIS_PASSWORD,
            REDIS_POOL_MAX_CONNECTIONS,
            REDIS_PORT,
            REDIS_SOCKET_CONNECT_TIMEOUT,
            REDIS_SOCKET_TIMEOUT,
            UNKNOWN_FACE_DIM,
            UNKNOWN_REID_DIM,
            UNKNOWN_TTL_SECONDS,
            VECTOR_SEARCH_K,
        )

        self._config = config or {}
        self.host = self._config.get("REDIS_HOST", REDIS_HOST)
        self.port = self._config.get("REDIS_PORT", REDIS_PORT)
        self.db = self._config.get("REDIS_DB", REDIS_DB)
        self.password = self._config.get("REDIS_PASSWORD", REDIS_PASSWORD)
        self.key_prefix = self._config.get("REDIS_KEY_PREFIX", REDIS_KEY_PREFIX)

        self.unknown_ttl = int(self._config.get("UNKNOWN_TTL_SECONDS", UNKNOWN_TTL_SECONDS))
        self.known_online_ttl = int(self._config.get("KNOWN_ONLINE_REID_TTL_SECONDS", 86400))
        self.claim_lock_ttl_ms = int(self._config.get("REDIS_CLAIM_LOCK_TTL_MS", REDIS_CLAIM_LOCK_TTL_MS))

        self.known_face_dim = int(self._config.get("KNOWN_FACE_DIM", KNOWN_FACE_DIM))
        self.known_reid_dim = int(self._config.get("KNOWN_REID_DIM", KNOWN_REID_DIM))
        self.unknown_face_dim = int(self._config.get("UNKNOWN_FACE_DIM", UNKNOWN_FACE_DIM))
        self.unknown_reid_dim = int(self._config.get("UNKNOWN_REID_DIM", UNKNOWN_REID_DIM))

        self.vector_search_k = int(self._config.get("VECTOR_SEARCH_K", VECTOR_SEARCH_K))
        self.max_known_reid_samples = int(self._config.get("MAX_KNOWN_REID_SAMPLES", MAX_KNOWN_REID_SAMPLES))
        self.max_known_anchor_samples = int(self._config.get("MAX_KNOWN_ANCHOR_SAMPLES", MAX_KNOWN_ANCHOR_SAMPLES))
        self.max_unknown_face_samples = int(self._config.get("MAX_UNKNOWN_FACE_SAMPLES", MAX_UNKNOWN_FACE_SAMPLES))
        self.max_unknown_reid_samples = int(self._config.get("MAX_UNKNOWN_REID_SAMPLES", MAX_UNKNOWN_REID_SAMPLES))
        self.dedup_threshold = float(self._config.get("DEDUP_THRESHOLD", DEDUP_THRESHOLD))
        self.claim_lock_retry_sleep_ms = int(self._config.get("CLAIM_LOCK_RETRY_SLEEP_MS", CLAIM_LOCK_RETRY_SLEEP_MS))
        self.claim_lock_max_retries = int(self._config.get("CLAIM_LOCK_MAX_RETRIES", CLAIM_LOCK_MAX_RETRIES))

        self.pool_max_connections = int(self._config.get("REDIS_POOL_MAX_CONNECTIONS", REDIS_POOL_MAX_CONNECTIONS))
        self.socket_connect_timeout = float(self._config.get("REDIS_SOCKET_CONNECT_TIMEOUT", REDIS_SOCKET_CONNECT_TIMEOUT))
        self.socket_timeout = float(self._config.get("REDIS_SOCKET_TIMEOUT", REDIS_SOCKET_TIMEOUT))

        self._prefix_known_face = f"{self.key_prefix}:known_face"
        self._prefix_known_reid = f"{self.key_prefix}:known_reid"
        self._prefix_known_reid_base = f"{self.key_prefix}:known_reid_base"
        self._prefix_known_reid_online = f"{self.key_prefix}:known_reid_online"
        self._prefix_unknown_face = f"{self.key_prefix}:unknown_face"
        self._prefix_unknown_reid = f"{self.key_prefix}:unknown_reid"
        self._prefix_unknown_entity = f"{self.key_prefix}:unknown_entity"
        self._prefix_unknown_samples = f"{self.key_prefix}:unknown_samples"
        self._prefix_unknown_counter = f"{self.key_prefix}:unknown_counter"
        self._prefix_import_sig = f"{self.key_prefix}:import_sig"
        self._prefix_lock = f"{self.key_prefix}:lock"

        self._set_known_face = f"{self.key_prefix}:keys:known_face"
        self._set_known_reid_base = f"{self.key_prefix}:keys:known_reid_base"
        self._set_known_reid_online = f"{self.key_prefix}:keys:known_reid_online"
        self._set_unknown_face = f"{self.key_prefix}:keys:unknown_face"
        self._set_unknown_reid = f"{self.key_prefix}:keys:unknown_reid"

        self._idx_known_face = f"idx:{self.key_prefix}:known_face"
        self._idx_known_reid = f"idx:{self.key_prefix}:known_reid"
        self._idx_unknown_face = f"idx:{self.key_prefix}:unknown_face"
        self._idx_unknown_reid = f"idx:{self.key_prefix}:unknown_reid"

        self._pool: Optional[redis.ConnectionPool] = None
        self._available = False
        self._lock = threading.RLock()
        self._unlock_sha = None
        self._touch_unknown_sha = None

    @property
    def available(self) -> bool:
        return self._available

    def _client(self) -> redis.Redis:
        if redis is None:
            raise RedisUnavailableError("Python package 'redis' is not installed")
        if self._pool is None:
            raise RedisUnavailableError("Redis connection pool is not initialized")
        return redis.Redis(connection_pool=self._pool)

    def connect(self) -> bool:
        try:
            if redis is None:
                raise RedisUnavailableError("Python package 'redis' is not installed")
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
            self._unlock_sha = client.script_load(self.UNLOCK_SCRIPT)
            self._touch_unknown_sha = client.script_load(self.TOUCH_UNKNOWN_SCRIPT)
            self._available = True
            print(f"[RedisIdentityMemory] connected {self.host}:{self.port} db={self.db}")
            return True
        except Exception as e:
            print(f"[RedisIdentityMemory] connect failed: {e}")
            self._available = False
            self._pool = None
            return False

    def disconnect(self):
        if self._pool is not None:
            self._pool.disconnect()
        self._pool = None
        self._available = False

    def ping(self) -> bool:
        if not self._available:
            return False
        try:
            return bool(self._client().ping())
        except Exception:
            self._available = False
            return False

    def _ensure_available(self):
        if not self._available:
            raise RedisUnavailableError("Redis is unavailable")

    @staticmethod
    def _decode(value):
        if isinstance(value, bytes):
            return value.decode()
        return value

    @staticmethod
    def _embedding_to_bytes(embedding: np.ndarray) -> bytes:
        return np.asarray(embedding, dtype=np.float32).reshape(-1).tobytes()

    @staticmethod
    def _bytes_to_embedding(data: bytes, dim: int) -> np.ndarray:
        arr = np.frombuffer(data, dtype=np.float32).copy()
        if len(arr) != dim:
            raise ValueError(f"embedding dim mismatch: expected {dim}, got {len(arr)}")
        return arr

    @staticmethod
    def _cosine_similarity(feat1: np.ndarray, feat2: np.ndarray) -> float:
        feat1 = np.asarray(feat1, dtype=np.float32).reshape(-1)
        feat2 = np.asarray(feat2, dtype=np.float32).reshape(-1)
        denom = float(np.linalg.norm(feat1) * np.linalg.norm(feat2))
        if denom <= 1e-12:
            return 0.0
        return float(np.dot(feat1, feat2) / denom)

    def _signature(self, entries: Iterable[dict]) -> str:
        hasher = hashlib.sha256()
        for entry in sorted(entries or [], key=lambda e: (str(e.get("person_id", "")), str(e.get("filename", "")))):
            hasher.update(str(entry.get("person_id", "")).encode())
            hasher.update(str(entry.get("unknown_id", "")).encode())
            hasher.update(str(entry.get("filename", "")).encode())
            embedding = entry.get("embedding")
            if embedding is not None:
                hasher.update(self._embedding_to_bytes(embedding))
        return hasher.hexdigest()

    def _version_key(self, kind: str) -> str:
        return f"{self.key_prefix}:version:{kind}"

    def get_versions(self) -> Dict[str, int]:
        self._ensure_available()
        client = self._client()
        values = client.mget([self._version_key(kind) for kind in self.VERSION_KINDS])
        versions = {}
        for kind, value in zip(self.VERSION_KINDS, values):
            try:
                versions[kind] = int(value or 0)
            except (TypeError, ValueError):
                versions[kind] = 0
        return versions

    def bump_version(self, kind: str) -> int:
        if kind not in self.VERSION_KINDS:
            raise ValueError(f"unknown version kind: {kind}")
        self._ensure_available()
        return int(self._client().incr(self._version_key(kind)))

    def _delete_members(self, client: redis.Redis, set_key: str):
        keys = list(client.smembers(set_key) or [])
        if keys:
            client.delete(*keys)
        client.delete(set_key)

    def _replace_entries(
        self,
        entries: Iterable[dict],
        set_key: str,
        key_prefix: str,
        id_field: str,
        dim: int,
        version_kind: str,
        signature_name: Optional[str] = None,
        signature: Optional[str] = None,
        ttl_seconds: Optional[int] = None,
    ) -> int:
        self._ensure_available()
        entries = list(entries or [])
        signature = signature if signature is not None else self._signature(entries)
        client = self._client()

        if signature_name:
            sig_key = f"{self._prefix_import_sig}:{signature_name}"
            existing_sig = client.get(sig_key)
            if existing_sig is not None and self._decode(existing_sig) == signature:
                return 0
        else:
            sig_key = None

        self._delete_members(client, set_key)

        now = time.time()
        pipe = client.pipeline(transaction=False)
        written = 0
        for idx, entry in enumerate(entries):
            embedding = np.asarray(entry.get("embedding"), dtype=np.float32).reshape(-1)
            owner_id = entry.get(id_field) or entry.get("owner_id")
            if owner_id in (None, "") or len(embedding) != dim:
                continue

            sample_key = f"{key_prefix}:{idx}"
            mapping = {
                "sample_key": sample_key,
                id_field: str(owner_id),
                "embedding": self._embedding_to_bytes(embedding),
                "created_at": str(entry.get("created_at", now)),
                "filename": str(entry.get("filename", "")),
                "is_anchor": "1" if entry.get("is_anchor") else "0",
            }
            pipe.hset(sample_key, mapping=mapping)
            if ttl_seconds is not None:
                pipe.expire(sample_key, int(ttl_seconds))
            pipe.sadd(set_key, sample_key)
            written += 1
        if sig_key:
            pipe.set(sig_key, signature)
        pipe.execute()

        self.bump_version(version_kind)
        return written

    def replace_known_faces(self, entries: Iterable[dict], signature: Optional[str] = None) -> int:
        return self._replace_entries(
            entries,
            self._set_known_face,
            self._prefix_known_face,
            "person_id",
            self.known_face_dim,
            "known_face",
            signature_name="known_face",
            signature=signature,
        )

    def replace_known_reid_base(self, entries: Iterable[dict], signature: Optional[str] = None) -> int:
        return self._replace_entries(
            entries,
            self._set_known_reid_base,
            self._prefix_known_reid_base,
            "person_id",
            self.known_reid_dim,
            "known_reid",
            signature_name="known_reid_base",
            signature=signature,
        )

    def import_known_faces(self, entries: list) -> int:
        return self.replace_known_faces(entries)

    def import_known_reid(self, entries: list) -> int:
        return self.replace_known_reid_base(entries)

    def _list_from_set(self, set_key: str, dim: int, id_field: str, include_ttl: bool = False) -> List[dict]:
        self._ensure_available()
        client = self._client()
        sample_keys = list(client.smembers(set_key) or [])
        if not sample_keys:
            return []

        now = time.time()
        entries = []
        pipe = client.pipeline(transaction=False)
        for key in sample_keys:
            pipe.hgetall(key)
            if include_ttl:
                pipe.ttl(key)
        raw = pipe.execute()

        step = 2 if include_ttl else 1
        stale_keys = []
        for idx, key in enumerate(sample_keys):
            data = raw[idx * step]
            ttl = raw[idx * step + 1] if include_ttl else None
            if not data or (include_ttl and ttl == -2):
                stale_keys.append(key)
                continue
            try:
                embedding = self._bytes_to_embedding(data[b"embedding"], dim)
            except Exception:
                continue

            owner_id = data.get(id_field.encode())
            if owner_id is None:
                continue
            owner_id = self._decode(owner_id)
            sample_key = self._decode(data.get(b"sample_key") or key)
            entry = {
                "sample_key": str(sample_key),
                id_field: str(owner_id),
                "owner_id": str(owner_id),
                "embedding": embedding,
                "created_at": float(self._decode(data.get(b"created_at") or 0) or 0),
                "filename": self._decode(data.get(b"filename") or ""),
                "is_anchor": self._decode(data.get(b"is_anchor") or "0") in ("1", "true", "True"),
            }
            if include_ttl:
                if ttl is not None and ttl >= 0:
                    entry["expires_at"] = now + float(ttl)
                else:
                    entry["expires_at"] = None
            entries.append(entry)

        if stale_keys:
            client.srem(set_key, *stale_keys)
        return entries

    def list_known_faces(self) -> List[dict]:
        return self._list_from_set(self._set_known_face, self.known_face_dim, "person_id")

    def list_known_reid_base(self) -> List[dict]:
        return self._list_from_set(self._set_known_reid_base, self.known_reid_dim, "person_id")

    def list_known_reid_online(self) -> List[dict]:
        return self._list_from_set(self._set_known_reid_online, self.known_reid_dim, "person_id", include_ttl=True)

    def list_known_reid_all(self) -> List[dict]:
        return self.list_known_reid_base() + self.list_known_reid_online()

    def list_unknown_faces(self) -> List[dict]:
        return self._list_from_set(self._set_unknown_face, self.unknown_face_dim, "unknown_id", include_ttl=True)

    def list_unknown_reid(self) -> List[dict]:
        return self._list_from_set(self._set_unknown_reid, self.unknown_reid_dim, "unknown_id", include_ttl=True)

    def _add_known_sample(
        self,
        person_id: str,
        embedding: np.ndarray,
        set_key: str,
        key_prefix: str,
        dim: int,
        version_kind: str,
        is_anchor: bool = False,
        ttl_seconds: Optional[int] = None,
    ):
        self._ensure_available()
        embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if len(embedding) != dim:
            return False

        person_id = str(person_id)
        client = self._client()
        sample_key = f"{key_prefix}:{person_id}:{time.time_ns()}"
        now = time.time()
        mapping = {
            "sample_key": sample_key,
            "person_id": person_id,
            "embedding": self._embedding_to_bytes(embedding),
            "created_at": str(now),
            "filename": "",
            "is_anchor": "1" if is_anchor else "0",
        }
        pipe = client.pipeline(transaction=False)
        pipe.hset(sample_key, mapping=mapping)
        if ttl_seconds is not None:
            pipe.expire(sample_key, int(ttl_seconds))
        pipe.sadd(set_key, sample_key)
        pipe.execute()
        self.bump_version(version_kind)
        entry = {
            "sample_key": sample_key,
            "person_id": person_id,
            "owner_id": person_id,
            "embedding": embedding.copy(),
            "created_at": now,
            "is_anchor": bool(is_anchor),
        }
        if ttl_seconds is not None:
            entry["expires_at"] = now + int(ttl_seconds)
        return entry

    def add_known_reid_online_sample(self, person_id, embedding, is_anchor: bool = False, ttl_seconds: int = 86400):
        return self._add_known_sample(
            person_id,
            embedding,
            self._set_known_reid_online,
            self._prefix_known_reid_online,
            self.known_reid_dim,
            "known_reid",
            is_anchor=is_anchor,
            ttl_seconds=int(ttl_seconds or self.known_online_ttl),
        )

    def add_known_reid_sample(self, person_id: str, embedding: np.ndarray, is_anchor: bool = False):
        return self.add_known_reid_online_sample(person_id, embedding, is_anchor=is_anchor, ttl_seconds=self.known_online_ttl)

    def add_known_face_sample(self, person_id: str, embedding: np.ndarray):
        return self._add_known_sample(
            person_id,
            embedding,
            self._set_known_face,
            self._prefix_known_face,
            self.known_face_dim,
            "known_face",
        )

    def allocate_unknown_id(self) -> str:
        self._ensure_available()
        client = self._client()
        seq = int(client.incr(self._prefix_unknown_counter))
        unknown_id = str(1000000 + seq)
        self._ensure_unknown_entity(client, unknown_id)
        return unknown_id

    def _ensure_unknown_entity(self, client: redis.Redis, unknown_id: str):
        entity_key = f"{self._prefix_unknown_entity}:{unknown_id}"
        now = time.time()
        if not client.exists(entity_key):
            client.hset(entity_key, mapping={
                "unknown_id": str(unknown_id),
                "face_count": "0",
                "reid_count": "0",
                "last_seen": str(now),
                "created_at": str(now),
                "status": "active",
            })
        client.expire(entity_key, self.unknown_ttl)
        client.expire(f"{self._prefix_unknown_samples}:{unknown_id}", self.unknown_ttl)

    def add_unknown_face_sample(self, unknown_id, embedding: np.ndarray):
        return self._add_unknown_sample(
            unknown_id,
            embedding,
            self._set_unknown_face,
            self._prefix_unknown_face,
            self.unknown_face_dim,
            self.max_unknown_face_samples,
            "face_count",
            "unknown_face",
        )

    def add_unknown_reid_sample(self, unknown_id, embedding: np.ndarray):
        return self._add_unknown_sample(
            unknown_id,
            embedding,
            self._set_unknown_reid,
            self._prefix_unknown_reid,
            self.unknown_reid_dim,
            self.max_unknown_reid_samples,
            "reid_count",
            "unknown_reid",
        )

    def _add_unknown_sample(
        self,
        unknown_id,
        embedding: np.ndarray,
        global_set_key: str,
        key_prefix: str,
        dim: int,
        max_samples: int,
        count_field: str,
        version_kind: str,
    ):
        self._ensure_available()
        embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if len(embedding) != dim:
            return False

        unknown_id = str(unknown_id)
        client = self._client()
        self._ensure_unknown_entity(client, unknown_id)
        entity_key = f"{self._prefix_unknown_entity}:{unknown_id}"
        sample_set_key = f"{self._prefix_unknown_samples}:{unknown_id}"

        existing_all = [
            self._decode(key)
            for key in (client.smembers(sample_set_key) or [])
        ]
        existing = sorted(key for key in existing_all if str(key).startswith(f"{key_prefix}:{unknown_id}:"))

        if existing:
            pipe = client.pipeline(transaction=False)
            for key in existing:
                pipe.hget(key, "embedding")
            for data in pipe.execute():
                if data is None:
                    continue
                try:
                    if self._cosine_similarity(embedding, np.frombuffer(data, dtype=np.float32).copy()) > self.dedup_threshold:
                        return False
                except Exception:
                    continue

        pipe = client.pipeline(transaction=False)
        if len(existing) >= max_samples and existing:
            oldest = existing[0]
            pipe.delete(oldest)
            pipe.srem(sample_set_key, oldest)
            pipe.srem(global_set_key, oldest)

        sample_key = f"{key_prefix}:{unknown_id}:{time.time_ns()}"
        now = time.time()
        expires_at = now + self.unknown_ttl
        pipe.hset(sample_key, mapping={
            "sample_key": sample_key,
            "unknown_id": unknown_id,
            "embedding": self._embedding_to_bytes(embedding),
            "created_at": str(now),
            "filename": "",
            "is_anchor": "0",
        })
        pipe.expire(sample_key, self.unknown_ttl)
        pipe.sadd(sample_set_key, sample_key)
        pipe.expire(sample_set_key, self.unknown_ttl)
        pipe.sadd(global_set_key, sample_key)
        pipe.hset(entity_key, "last_seen", str(now))
        pipe.expire(entity_key, self.unknown_ttl)
        pipe.execute()

        current_count = len([
            key for key in client.smembers(sample_set_key) or []
            if self._decode(key).startswith(f"{key_prefix}:{unknown_id}:")
        ])
        client.hset(entity_key, count_field, str(current_count))
        self.bump_version(version_kind)

        return {
            "sample_key": sample_key,
            "unknown_id": unknown_id,
            "owner_id": unknown_id,
            "embedding": embedding.copy(),
            "created_at": now,
            "expires_at": expires_at,
            "is_anchor": False,
        }

    def touch_unknown(self, unknown_id):
        if unknown_id in (-1, None):
            return
        self._ensure_available()
        unknown_id = str(unknown_id)
        client = self._client()
        entity_key = f"{self._prefix_unknown_entity}:{unknown_id}"
        sample_set_key = f"{self._prefix_unknown_samples}:{unknown_id}"
        now = time.time()
        try:
            if self._touch_unknown_sha:
                client.evalsha(
                    self._touch_unknown_sha,
                    2,
                    entity_key,
                    sample_set_key,
                    str(self.unknown_ttl),
                    str(now),
                )
            else:
                raise RuntimeError("touch script not loaded")
        except Exception:
            client.expire(entity_key, self.unknown_ttl)
            client.hset(entity_key, "last_seen", str(now))
            samples = client.smembers(sample_set_key) or []
            if samples:
                pipe = client.pipeline(transaction=False)
                for sample_key in samples:
                    pipe.expire(sample_key, self.unknown_ttl)
                pipe.expire(sample_set_key, self.unknown_ttl)
                pipe.execute()

    def release_unknown(self, unknown_id):
        if unknown_id in (-1, None):
            return
        self._ensure_available()
        unknown_id = str(unknown_id)
        client = self._client()
        entity_key = f"{self._prefix_unknown_entity}:{unknown_id}"
        sample_set_key = f"{self._prefix_unknown_samples}:{unknown_id}"
        sample_keys = [self._decode(key) for key in (client.smembers(sample_set_key) or [])]
        face_deleted = any(key.startswith(f"{self._prefix_unknown_face}:{unknown_id}:") for key in sample_keys)
        reid_deleted = any(key.startswith(f"{self._prefix_unknown_reid}:{unknown_id}:") for key in sample_keys)

        pipe = client.pipeline(transaction=False)
        if sample_keys:
            pipe.delete(*sample_keys)
            pipe.srem(self._set_unknown_face, *sample_keys)
            pipe.srem(self._set_unknown_reid, *sample_keys)
        pipe.delete(sample_set_key)
        pipe.delete(entity_key)
        pipe.execute()

        if face_deleted:
            self.bump_version("unknown_face")
        if reid_deleted:
            self.bump_version("unknown_reid")

    def release_if_empty(self, unknown_id):
        if unknown_id in (-1, None):
            return
        self._ensure_available()
        client = self._client()
        entity_key = f"{self._prefix_unknown_entity}:{unknown_id}"
        if not client.exists(entity_key):
            return
        face_count = int(client.hget(entity_key, "face_count") or 0)
        reid_count = int(client.hget(entity_key, "reid_count") or 0)
        if face_count == 0 and reid_count == 0:
            self.release_unknown(unknown_id)

    def acquire_lock(self, name: str, ttl_ms: int) -> Optional[str]:
        self._ensure_available()
        token = uuid.uuid4().hex
        lock_key = f"{self._prefix_lock}:{name}"
        return token if self._client().set(lock_key, token, nx=True, px=int(ttl_ms)) else None

    def release_lock(self, name: str, token: str):
        if token is None:
            return
        try:
            client = self._client()
            lock_key = f"{self._prefix_lock}:{name}"
            if self._unlock_sha:
                client.evalsha(self._unlock_sha, 1, lock_key, token)
            else:
                client.eval(self.UNLOCK_SCRIPT, 1, lock_key, token)
        except Exception:
            pass

    def _acquire_claim_lock(self, lock_name: str, ttl_ms: int) -> Optional[str]:
        return self.acquire_lock(lock_name, ttl_ms)

    def _release_claim_lock(self, lock_name: str, token: str):
        self.release_lock(lock_name, token)

    def _search_entries(self, entries: Iterable[dict], embedding: np.ndarray, threshold: float):
        query = np.asarray(embedding, dtype=np.float32).reshape(-1)
        best_id = None
        best_similarity = 0.0
        now = time.time()
        for entry in entries:
            expires_at = entry.get("expires_at")
            if expires_at is not None and expires_at <= now:
                continue
            owner_id = entry.get("person_id") or entry.get("unknown_id") or entry.get("owner_id")
            if owner_id is None:
                continue
            similarity = self._cosine_similarity(query, entry["embedding"])
            if similarity >= threshold and similarity > best_similarity:
                best_id = str(owner_id)
                best_similarity = similarity
        return best_id, best_similarity

    def search_known_face(self, embedding: np.ndarray, threshold: float, top_k: int = None):
        return self._search_entries(self.list_known_faces(), embedding, threshold)

    def search_known_reid(self, embedding: np.ndarray, threshold: float, top_k: int = None):
        return self._search_entries(self.list_known_reid_all(), embedding, threshold)

    def search_unknown_face(self, embedding: np.ndarray, threshold: float, top_k: int = None):
        return self._search_entries(self.list_unknown_faces(), embedding, threshold)

    def search_unknown_reid(self, embedding: np.ndarray, threshold: float, top_k: int = None):
        return self._search_entries(self.list_unknown_reid(), embedding, threshold)

    def find_or_create_unknown(self, embedding: np.ndarray, threshold: float, modality: str) -> str:
        self._ensure_available()
        search_fn = self.search_unknown_face if modality == "face" else self.search_unknown_reid
        add_fn = self.add_unknown_face_sample if modality == "face" else self.add_unknown_reid_sample
        lock_name = f"claim_unknown:{modality}"

        existing_id, _ = search_fn(embedding, threshold)
        if existing_id is not None:
            return existing_id

        token = self.acquire_lock(lock_name, self.claim_lock_ttl_ms)
        if token is None:
            raise RedisUnavailableError("failed to acquire unknown claim lock")
        try:
            existing_id, _ = search_fn(embedding, threshold)
            if existing_id is not None:
                return existing_id
            unknown_id = self.allocate_unknown_id()
            add_fn(unknown_id, embedding)
            return unknown_id
        finally:
            self.release_lock(lock_name, token)

    def _list_known_samples_by_person(self, person_id: str, modality: str, client: redis.Redis = None) -> list:
        person_id = str(person_id)
        if modality == "face":
            return [entry["sample_key"] for entry in self.list_known_faces() if entry.get("person_id") == person_id]
        return [
            (entry["sample_key"], bool(entry.get("is_anchor")))
            for entry in self.list_known_reid_all()
            if entry.get("person_id") == person_id
        ]

    def count_known_samples(self, person_id: str, modality: str, is_anchor: bool = None) -> int:
        entries = self._list_known_samples_by_person(person_id, modality)
        if modality == "face" or is_anchor is None:
            return len(entries)
        return sum(1 for _, anchor in entries if anchor == is_anchor)

    def cleanup_stale_unknowns(self) -> int:
        self._ensure_available()
        client = self._client()
        cleaned = 0
        cursor = 0
        pattern = f"{self._prefix_unknown_samples}:*"
        while True:
            cursor, keys = client.scan(cursor, match=pattern, count=100)
            for key in keys:
                key_str = self._decode(key)
                unknown_id = key_str.rsplit(":", 1)[-1]
                entity_key = f"{self._prefix_unknown_entity}:{unknown_id}"
                if client.exists(entity_key):
                    continue
                sample_keys = client.smembers(key) or []
                if sample_keys:
                    client.delete(*sample_keys)
                    client.srem(self._set_unknown_face, *sample_keys)
                    client.srem(self._set_unknown_reid, *sample_keys)
                client.delete(key)
                cleaned += 1
            if cursor == 0:
                break
        if cleaned:
            self.bump_version("unknown_face")
            self.bump_version("unknown_reid")
        return cleaned

    def ensure_indexes(self) -> bool:
        """Best-effort optional Redis vector index setup for diagnostics."""
        self._ensure_available()
        client = self._client()
        try:
            client.execute_command("FT._LIST")
        except Exception as e:
            print(f"[RedisIdentityMemory] optional vector index support unavailable: {e}")
            return True

        specs = [
            (self._idx_known_face, self._prefix_known_face, self.known_face_dim),
            (self._idx_known_reid, self._prefix_known_reid_base, self.known_reid_dim),
            (self._idx_unknown_face, self._prefix_unknown_face, self.unknown_face_dim),
            (self._idx_unknown_reid, self._prefix_unknown_reid, self.unknown_reid_dim),
        ]
        for idx_name, prefix, dim in specs:
            try:
                client.execute_command("FT.INFO", idx_name)
                continue
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
            except Exception as e:
                print(f"[RedisIdentityMemory] optional index create failed ({idx_name}): {e}")
        return True

    def drop_indexes(self):
        if not self._available:
            return
        client = self._client()
        for idx_name in (self._idx_known_face, self._idx_known_reid, self._idx_unknown_face, self._idx_unknown_reid):
            try:
                client.execute_command("FT.DROPINDEX", idx_name, "DD")
            except Exception:
                pass

    def clear_all(self):
        self._ensure_available()
        client = self._client()
        cursor = 0
        while True:
            cursor, keys = client.scan(cursor, match=f"{self.key_prefix}:*", count=200)
            if keys:
                client.delete(*keys)
            if cursor == 0:
                break

    def flush_unknowns(self):
        self._ensure_available()
        client = self._client()
        deleted_face = client.scard(self._set_unknown_face) > 0
        deleted_reid = client.scard(self._set_unknown_reid) > 0
        for prefix in (
            self._prefix_unknown_face,
            self._prefix_unknown_reid,
            self._prefix_unknown_entity,
            self._prefix_unknown_samples,
        ):
            cursor = 0
            while True:
                cursor, keys = client.scan(cursor, match=f"{prefix}:*", count=200)
                if keys:
                    client.delete(*keys)
                if cursor == 0:
                    break
        client.delete(self._set_unknown_face, self._set_unknown_reid, self._prefix_unknown_counter)
        if deleted_face:
            self.bump_version("unknown_face")
        if deleted_reid:
            self.bump_version("unknown_reid")

    def get_stats(self) -> dict:
        self._ensure_available()
        client = self._client()
        stats = {
            "available": self._available,
            "known_face_count": int(client.scard(self._set_known_face)),
            "known_reid_base_count": int(client.scard(self._set_known_reid_base)),
            "known_reid_online_count": int(client.scard(self._set_known_reid_online)),
            "unknown_face_count": int(client.scard(self._set_unknown_face)),
            "unknown_reid_count": int(client.scard(self._set_unknown_reid)),
            "versions": self.get_versions(),
        }
        cursor = 0
        entity_count = 0
        while True:
            cursor, keys = client.scan(cursor, match=f"{self._prefix_unknown_entity}:*", count=100)
            entity_count += len(keys)
            if cursor == 0:
                break
        stats["unknown_entity_count"] = entity_count
        return stats
