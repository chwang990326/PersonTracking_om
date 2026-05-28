"""pytest fixtures for Redis identity memory tests.

Uses real Redis with an isolated key prefix (test_pt) so tests do not
interfere with production data (person_tracking prefix).
"""

import numpy as np
import pytest

from models.redis_identity_memory import RedisIdentityMemory

TEST_CONFIG = {
    "REDIS_HOST": "127.0.0.1",
    "REDIS_PORT": 6379,
    "REDIS_DB": 0,
    "REDIS_PASSWORD": "redisForPersonTracking",
    "REDIS_KEY_PREFIX": "test_pt",
    "UNKNOWN_TTL_SECONDS": 5,
    "REDIS_CLAIM_LOCK_TTL_MS": 2000,
    "KNOWN_FACE_DIM": 512,
    "KNOWN_REID_DIM": 768,
    "UNKNOWN_FACE_DIM": 512,
    "UNKNOWN_REID_DIM": 768,
    "VECTOR_SEARCH_K": 10,
    "MAX_KNOWN_REID_SAMPLES": 12,
    "MAX_KNOWN_ANCHOR_SAMPLES": 2,
    "MAX_UNKNOWN_FACE_SAMPLES": 3,
    "MAX_UNKNOWN_REID_SAMPLES": 3,
    "DEDUP_THRESHOLD": 0.95,
    "CLAIM_LOCK_RETRY_SLEEP_MS": 10,
    "CLAIM_LOCK_MAX_RETRIES": 30,
    "REDIS_POOL_MAX_CONNECTIONS": 5,
    "REDIS_SOCKET_CONNECT_TIMEOUT": 2,
    "REDIS_SOCKET_TIMEOUT": 2,
}


def _flush_test_prefix(mem: RedisIdentityMemory):
    """Delete all keys under the test prefix."""
    try:
        client = mem._client()
        cursor = 0
        while True:
            cursor, keys = client.scan(cursor, match="test_pt:*", count=100)
            if keys:
                client.delete(*keys)
            if cursor == 0:
                break
    except Exception:
        pass


@pytest.fixture
def face_embedding():
    """Generate a random normalized 512-dim face embedding."""
    vec = np.random.randn(512).astype(np.float32)
    return vec / np.linalg.norm(vec)


@pytest.fixture
def reid_embedding():
    """Generate a random normalized 768-dim ReID embedding."""
    vec = np.random.randn(768).astype(np.float32)
    return vec / np.linalg.norm(vec)


@pytest.fixture
def known_face_entries():
    """Sample known face import entries."""
    entries = []
    for pid in ["Alice", "Bob", "Charlie"]:
        for i in range(2):
            vec = np.random.randn(512).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            entries.append({
                "person_id": pid,
                "embedding": vec,
                "filename": f"{pid}_face_{i}.jpg",
            })
    return entries


@pytest.fixture
def known_reid_entries():
    """Sample known ReID import entries."""
    entries = []
    for pid in ["Alice", "Bob", "Charlie"]:
        for i in range(3):
            vec = np.random.randn(768).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            entries.append({
                "person_id": pid,
                "embedding": vec,
                "filename": f"{pid}_reid_{i}.jpg",
                "is_anchor": (i == 0),
            })
    return entries


@pytest.fixture
def redis_memory():
    """Create a RedisIdentityMemory connected to real Redis with test prefix."""
    mem = RedisIdentityMemory(config=TEST_CONFIG)
    if not mem.connect():
        pytest.skip("Redis connection failed - is Redis running and is redis-py installed?")
    mem.ensure_indexes()
    _flush_test_prefix(mem)

    yield mem

    _flush_test_prefix(mem)
    mem.drop_indexes()
    mem.disconnect()
