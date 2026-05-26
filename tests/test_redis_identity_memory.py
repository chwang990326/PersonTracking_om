"""Unit tests for RedisIdentityMemory module.

Requires a real Redis 8 instance at 127.0.0.1:6379 with RediSearch.
Tests use the isolated prefix `test_pt` – no production data is touched.
"""

import threading
import time

import numpy as np
import pytest

from models.redis_identity_memory import RedisIdentityMemory, RedisUnavailableError


def _norm(vec):
    return vec / (np.linalg.norm(vec) + 1e-12)


class TestRedisConnection:
    """Connection management tests."""

    def test_init_with_config(self):
        mem = RedisIdentityMemory(config={
            "REDIS_HOST": "10.0.0.1",
            "REDIS_PORT": 9999,
            "REDIS_KEY_PREFIX": "custom",
        })
        assert mem.host == "10.0.0.1"
        assert mem.port == 9999
        assert mem.key_prefix == "custom"

    def test_default_not_available(self):
        mem = RedisIdentityMemory()
        assert not mem.available

    def test_ping_returns_false_when_not_connected(self):
        mem = RedisIdentityMemory()
        assert not mem.ping()

    def test_available_property_after_connect(self, redis_memory):
        assert redis_memory.available

    def test_ping_after_connect(self, redis_memory):
        assert redis_memory.ping()

    def test_ensure_available_raises_when_not_connected(self):
        mem = RedisIdentityMemory()
        with pytest.raises(RedisUnavailableError):
            mem._ensure_available()


class TestIndexManagement:
    """Index creation and management tests."""

    def test_ensure_indexes_idempotent(self, redis_memory):
        assert redis_memory.ensure_indexes()
        assert redis_memory.ensure_indexes()

    def test_drop_indexes(self, redis_memory):
        redis_memory.drop_indexes()
        assert redis_memory.ensure_indexes()


class TestKnownImport:
    """Known identity import tests."""

    def test_import_known_faces_empty(self, redis_memory):
        count = redis_memory.import_known_faces([])
        assert count == 0

    def test_import_known_reid_empty(self, redis_memory):
        count = redis_memory.import_known_reid([])
        assert count == 0

    def test_import_known_faces_idempotent(self, redis_memory, known_face_entries):
        count1 = redis_memory.import_known_faces(known_face_entries)
        assert count1 == len(known_face_entries)
        count2 = redis_memory.import_known_faces(known_face_entries)
        assert count2 == 0

    def test_import_known_reid(self, redis_memory, known_reid_entries):
        count = redis_memory.import_known_reid(known_reid_entries)
        assert count == len(known_reid_entries)


class TestEmbeddingSerialization:
    """Vector serialization roundtrip tests."""

    def test_roundtrip_512(self):
        original = _norm(np.random.randn(512).astype(np.float32))
        data = RedisIdentityMemory._embedding_to_bytes(original)
        restored = RedisIdentityMemory._bytes_to_embedding(data, 512)
        assert np.allclose(original, restored, atol=1e-6)

    def test_roundtrip_768(self):
        original = _norm(np.random.randn(768).astype(np.float32))
        data = RedisIdentityMemory._embedding_to_bytes(original)
        restored = RedisIdentityMemory._bytes_to_embedding(data, 768)
        assert np.allclose(original, restored, atol=1e-6)

    def test_cosine_similarity_identical(self):
        vec = _norm(np.random.randn(512).astype(np.float32))
        sim = RedisIdentityMemory._cosine_similarity(vec, vec)
        assert abs(sim - 1.0) < 1e-5

    def test_cosine_similarity_orthogonal(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        sim = RedisIdentityMemory._cosine_similarity(a, b)
        assert abs(sim - 0.0) < 1e-5


class TestUnknownIDAllocation:
    """Unknown ID allocation tests."""

    def test_allocate_id_returns_increasing(self, redis_memory):
        id1 = redis_memory.allocate_unknown_id()
        id2 = redis_memory.allocate_unknown_id()
        assert id2 > id1

    def test_allocate_id_creates_entity(self, redis_memory):
        new_id = redis_memory.allocate_unknown_id()
        client = redis_memory._client()
        entity_key = f"test_pt:unknown_entity:{new_id}"
        assert client.exists(entity_key)

    def test_allocate_id_sample_set_after_add(self, redis_memory, face_embedding):
        new_id = redis_memory.allocate_unknown_id()
        redis_memory.add_unknown_face_sample(new_id, face_embedding)
        client = redis_memory._client()
        sample_set_key = f"test_pt:unknown_samples:{new_id}"
        assert client.exists(sample_set_key)


class TestUnknownSampleManagement:
    """Unknown sample add/touch/release tests."""

    def test_add_face_sample(self, redis_memory, face_embedding):
        uid = redis_memory.allocate_unknown_id()
        result = redis_memory.add_unknown_face_sample(uid, face_embedding)
        assert result

    def test_add_reid_sample(self, redis_memory, reid_embedding):
        uid = redis_memory.allocate_unknown_id()
        result = redis_memory.add_unknown_reid_sample(uid, reid_embedding)
        assert result

    def test_dedup_same_embedding(self, redis_memory, face_embedding):
        uid = redis_memory.allocate_unknown_id()
        assert redis_memory.add_unknown_face_sample(uid, face_embedding)
        assert not redis_memory.add_unknown_face_sample(uid, face_embedding)

    def test_capacity_eviction(self, redis_memory, face_embedding):
        uid = redis_memory.allocate_unknown_id()
        for i in range(4):
            vec = _norm(face_embedding + np.random.randn(512).astype(np.float32) * 0.1)
            result = redis_memory.add_unknown_face_sample(uid, vec)
            assert result

        client = redis_memory._client()
        sample_set_key = f"test_pt:unknown_samples:{uid}"
        sample_keys = client.smembers(sample_set_key)
        assert len(sample_keys) <= 3

    def test_touch_unknown(self, redis_memory):
        uid = redis_memory.allocate_unknown_id()
        client = redis_memory._client()
        entity_key = f"test_pt:unknown_entity:{uid}"
        initial_ttl = client.ttl(entity_key)
        time.sleep(1)
        redis_memory.touch_unknown(uid)
        new_ttl = client.ttl(entity_key)
        assert new_ttl >= 3

    def test_touch_unknown_none_safe(self, redis_memory):
        redis_memory.touch_unknown(None)
        redis_memory.touch_unknown(-1)

    def test_release_unknown(self, redis_memory, face_embedding):
        uid = redis_memory.allocate_unknown_id()
        redis_memory.add_unknown_face_sample(uid, face_embedding)
        redis_memory.release_unknown(uid)

        client = redis_memory._client()
        entity_key = f"test_pt:unknown_entity:{uid}"
        assert not client.exists(entity_key)

    def test_release_if_empty(self, redis_memory):
        uid = redis_memory.allocate_unknown_id()
        redis_memory.release_if_empty(uid)
        client = redis_memory._client()
        entity_key = f"test_pt:unknown_entity:{uid}"
        assert not client.exists(entity_key)

    def test_release_if_empty_with_samples(self, redis_memory, face_embedding):
        uid = redis_memory.allocate_unknown_id()
        redis_memory.add_unknown_face_sample(uid, face_embedding)
        redis_memory.release_if_empty(uid)
        client = redis_memory._client()
        entity_key = f"test_pt:unknown_entity:{uid}"
        assert client.exists(entity_key)

    def test_cleanup_stale_unknowns(self, redis_memory, face_embedding):
        uid = redis_memory.allocate_unknown_id()
        redis_memory.add_unknown_face_sample(uid, face_embedding)
        client = redis_memory._client()
        entity_key = f"test_pt:unknown_entity:{uid}"
        # Delete the entity to simulate expiration — orphaning the sample keys
        client.delete(entity_key)
        cleaned = redis_memory.cleanup_stale_unknowns()
        assert cleaned >= 1


class TestKnownSampleManagement:
    """Known identity sample management tests."""

    def test_add_known_reid_sample(self, redis_memory, reid_embedding):
        result = redis_memory.add_known_reid_sample("Alice", reid_embedding, is_anchor=False)
        assert result

    def test_add_known_anchor_sample(self, redis_memory, reid_embedding):
        result = redis_memory.add_known_reid_sample("Alice", reid_embedding, is_anchor=True)
        assert result

    def test_add_known_face_sample(self, redis_memory, face_embedding):
        result = redis_memory.add_known_face_sample("Alice", face_embedding)
        assert result

    def test_count_known_samples(self, redis_memory, reid_embedding):
        redis_memory.add_known_reid_sample("Bob", reid_embedding, is_anchor=True)
        redis_memory.add_known_reid_sample("Bob", reid_embedding + 0.1, is_anchor=False)

        total = redis_memory.count_known_samples("Bob", "reid")
        anchors = redis_memory.count_known_samples("Bob", "reid", is_anchor=True)
        non_anchors = redis_memory.count_known_samples("Bob", "reid", is_anchor=False)

        assert total == 2
        assert anchors == 1
        assert non_anchors == 1


class TestKnownSearch:
    """Known identity vector search tests."""

    def test_search_known_face_hit(self, redis_memory, known_face_entries):
        redis_memory.import_known_faces(known_face_entries)
        # Search with one of the imported embeddings
        query = known_face_entries[0]["embedding"]
        person_id, sim = redis_memory.search_known_face(query, threshold=0.5, top_k=3)
        assert person_id == known_face_entries[0]["person_id"]
        assert sim > 0.9

    def test_search_known_face_miss(self, redis_memory, known_face_entries):
        redis_memory.import_known_faces(known_face_entries)
        # Use a very different embedding
        query = _norm(np.random.randn(512).astype(np.float32))
        person_id, sim = redis_memory.search_known_face(query, threshold=0.99, top_k=3)
        assert person_id is None

    def test_search_known_reid_hit(self, redis_memory, known_reid_entries):
        redis_memory.import_known_reid(known_reid_entries)
        query = known_reid_entries[0]["embedding"]
        person_id, sim = redis_memory.search_known_reid(query, threshold=0.5, top_k=3)
        assert person_id == known_reid_entries[0]["person_id"]
        assert sim > 0.9

    def test_search_known_reid_miss(self, redis_memory, known_reid_entries):
        redis_memory.import_known_reid(known_reid_entries)
        query = _norm(np.random.randn(768).astype(np.float32))
        person_id, sim = redis_memory.search_known_reid(query, threshold=0.99, top_k=3)
        assert person_id is None


class TestFindOrCreateUnknown:
    """Unknown find-or-create tests."""

    def test_find_existing_by_search(self, redis_memory, reid_embedding):
        uid = redis_memory.allocate_unknown_id()
        redis_memory.add_unknown_reid_sample(uid, reid_embedding)
        result = redis_memory.find_or_create_unknown(reid_embedding, 0.5, "reid")
        assert result == uid

    def test_find_or_create_new_when_no_match(self, redis_memory, reid_embedding):
        result = redis_memory.find_or_create_unknown(reid_embedding, 0.99, "reid")
        assert isinstance(result, int)
        assert result > 0

    def test_allocate_id_basic(self, redis_memory):
        uid = redis_memory.allocate_unknown_id()
        assert isinstance(uid, int)
        assert uid > 0

    def test_find_or_create_face(self, redis_memory, face_embedding):
        uid = redis_memory.allocate_unknown_id()
        redis_memory.add_unknown_face_sample(uid, face_embedding)
        result = redis_memory.find_or_create_unknown(face_embedding, 0.5, "face")
        assert result == uid


class TestConcurrentFindOrCreate:
    """Concurrent find-or-create tests."""

    def test_concurrent_same_embedding_single_id(self, redis_memory, reid_embedding):
        """Multiple threads searching the same embedding should get the same unknown_id.

        First create an unknown with a sample, then run concurrent find_or_create
        to verify they all find the existing ID.
        """
        # Pre-register an unknown with a sample
        uid = redis_memory.allocate_unknown_id()
        redis_memory.add_unknown_reid_sample(uid, reid_embedding)

        results = []
        errors = []
        lock = threading.Lock()

        def worker():
            try:
                found_id = redis_memory.find_or_create_unknown(reid_embedding, 0.5, "reid")
                with lock:
                    results.append(found_id)
            except Exception as e:
                with lock:
                    errors.append(str(e))

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Errors: {errors}"
        unique_ids = set(results)
        assert len(unique_ids) == 1, f"Expected single ID, got {unique_ids}"
        assert list(unique_ids)[0] == uid


class TestUnknownSearch:
    """Unknown vector search tests."""

    def test_search_unknown_reid(self, redis_memory, reid_embedding):
        uid = redis_memory.allocate_unknown_id()
        redis_memory.add_unknown_reid_sample(uid, reid_embedding)
        result_id, sim = redis_memory.search_unknown_reid(reid_embedding, 0.5, top_k=3)
        assert result_id == uid
        assert sim > 0.9

    def test_search_unknown_face(self, redis_memory, face_embedding):
        uid = redis_memory.allocate_unknown_id()
        redis_memory.add_unknown_face_sample(uid, face_embedding)
        result_id, sim = redis_memory.search_unknown_face(face_embedding, 0.5, top_k=3)
        assert result_id == uid
        assert sim > 0.9


class TestStats:
    """Stats and admin tests."""

    def test_get_stats(self, redis_memory):
        stats = redis_memory.get_stats()
        assert stats["available"] is True
        assert "unknown_counter" in stats
        assert "unknown_entity_count" in stats

    def test_flush_unknowns(self, redis_memory):
        uid = redis_memory.allocate_unknown_id()
        redis_memory.flush_unknowns()
        client = redis_memory._client()
        entity_key = f"test_pt:unknown_entity:{uid}"
        assert not client.exists(entity_key)
