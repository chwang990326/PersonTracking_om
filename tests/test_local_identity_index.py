import time

import numpy as np

from models.local_identity_index import LocalIdentityCache, LocalVectorIndex


def _norm(vec):
    return vec / (np.linalg.norm(vec) + 1e-12)


def test_local_vector_index_rebuild_search_add_and_features():
    index = LocalVectorIndex(4, "test")
    alice = _norm(np.array([1, 0, 0, 0], dtype=np.float32))
    bob = _norm(np.array([0, 1, 0, 0], dtype=np.float32))

    count = index.rebuild([
        {"sample_key": "a1", "person_id": "alice", "embedding": alice},
        {"sample_key": "b1", "person_id": "bob", "embedding": bob},
    ])
    assert count == 2

    owner_id, similarity, entry = index.search(alice, threshold=0.9)
    assert owner_id == "alice"
    assert similarity > 0.99
    assert entry["sample_key"] == "a1"

    index.add({"sample_key": "a2", "person_id": "alice", "embedding": alice})
    features = index.get_features_by_id("alice")
    assert features.shape == (2, 4)
    assert index.count_by_id("alice") == 2


def test_local_vector_index_filters_expired_unknown_entries():
    index = LocalVectorIndex(4, "unknown")
    active = _norm(np.array([1, 0, 0, 0], dtype=np.float32))
    expired = _norm(np.array([0, 1, 0, 0], dtype=np.float32))
    now = time.time()

    count = index.rebuild([
        {"sample_key": "old", "unknown_id": "1000001", "embedding": expired, "expires_at": now - 1},
        {"sample_key": "new", "unknown_id": "1000002", "embedding": active, "expires_at": now + 60},
    ])
    assert count == 1

    owner_id, similarity, _ = index.search(expired, threshold=0.1)
    assert owner_id is None
    assert similarity == 0.0

    owner_id, similarity, _ = index.search(active, threshold=0.9)
    assert owner_id == "1000002"
    assert similarity > 0.99


class FakeRedisMemory:
    available = True
    unknown_ttl = 300

    def __init__(self):
        self.versions = {"known_face": 1, "known_reid": 1, "unknown_face": 1, "unknown_reid": 1}
        self.known_faces = []
        self.known_reid = []
        self.unknown_faces = []
        self.unknown_reid = []

    def get_versions(self):
        return dict(self.versions)

    def list_known_faces(self):
        return list(self.known_faces)

    def list_known_reid_all(self):
        return list(self.known_reid)

    def list_unknown_faces(self):
        return list(self.unknown_faces)

    def list_unknown_reid(self):
        return list(self.unknown_reid)


def test_local_identity_cache_syncs_versions_and_searches():
    redis_memory = FakeRedisMemory()
    face = _norm(np.random.randn(512).astype(np.float32))
    reid = _norm(np.random.randn(768).astype(np.float32))
    redis_memory.known_faces = [{"sample_key": "kf1", "person_id": "alice", "embedding": face}]
    redis_memory.unknown_reid = [{
        "sample_key": "ur1",
        "unknown_id": "1000001",
        "embedding": reid,
        "expires_at": time.time() + 60,
    }]

    cache = LocalIdentityCache(redis_memory, sync_interval=0)
    assert cache.maybe_sync(force=True)

    person_id, face_sim = cache.search_known_face(face, threshold=0.9)
    unknown_id, reid_sim = cache.search_unknown_reid(reid, threshold=0.9)

    assert person_id == "alice"
    assert face_sim > 0.99
    assert unknown_id == "1000001"
    assert reid_sim > 0.99
