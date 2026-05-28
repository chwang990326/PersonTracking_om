"""Worker-local identity vector indexes backed by FAISS or NumPy."""

import logging
import threading
import time
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import faiss
except Exception:  # pragma: no cover - exercised only when faiss is absent
    faiss = None


LOGGER = logging.getLogger(__name__)


def _entry_owner_id(entry: dict) -> Optional[str]:
    owner_id = (
        entry.get("owner_id")
        or entry.get("person_id")
        or entry.get("unknown_id")
    )
    if owner_id in (None, ""):
        return None
    return str(owner_id)


class LocalVectorIndex:
    """Small copy-on-write vector index for a single embedding space."""

    def __init__(self, dim: int, name: str = ""):
        self.dim = int(dim)
        self.name = name or f"dim{dim}"
        self._lock = threading.RLock()
        self._entries: List[dict] = []
        self._matrix = np.empty((0, self.dim), dtype=np.float32)
        self._index = self._new_index()
        self._has_expiring_entries = False

    def _new_index(self):
        if faiss is None:
            return None
        return faiss.IndexFlatIP(self.dim)

    @staticmethod
    def _normalize(embedding) -> Optional[np.ndarray]:
        if embedding is None:
            return None
        arr = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(arr))
        if norm <= 1e-12:
            return None
        return arr / norm

    @staticmethod
    def _is_expired(entry: dict, now: Optional[float] = None) -> bool:
        expires_at = entry.get("expires_at")
        if expires_at in (None, ""):
            return False
        try:
            expires_at = float(expires_at)
        except (TypeError, ValueError):
            return False
        if now is None:
            now = time.time()
        return expires_at <= now

    def _prepare(self, entries: Iterable[dict]) -> Tuple[List[dict], np.ndarray, object, bool]:
        now = time.time()
        clean_entries: List[dict] = []
        vectors: List[np.ndarray] = []
        has_expiring = False

        for idx, entry in enumerate(entries or []):
            if not isinstance(entry, dict) or self._is_expired(entry, now):
                continue
            embedding = self._normalize(entry.get("embedding"))
            if embedding is None or embedding.shape[0] != self.dim:
                continue
            owner_id = _entry_owner_id(entry)
            if owner_id is None:
                continue

            sample_key = entry.get("sample_key") or f"{self.name}:local:{idx}"
            normalized_entry = dict(entry)
            normalized_entry["sample_key"] = str(sample_key)
            normalized_entry["owner_id"] = owner_id
            normalized_entry["embedding"] = np.asarray(entry.get("embedding"), dtype=np.float32).reshape(-1).copy()
            if entry.get("expires_at") not in (None, ""):
                has_expiring = True

            clean_entries.append(normalized_entry)
            vectors.append(embedding.astype(np.float32, copy=False))

        if vectors:
            matrix = np.vstack(vectors).astype(np.float32, copy=False)
        else:
            matrix = np.empty((0, self.dim), dtype=np.float32)

        index = self._new_index()
        if index is not None and len(matrix) > 0:
            index.add(matrix)

        return clean_entries, matrix, index, has_expiring

    def rebuild(self, entries: Iterable[dict]) -> int:
        """Rebuild the full index and publish it atomically."""
        clean_entries, matrix, index, has_expiring = self._prepare(entries)
        with self._lock:
            self._entries = clean_entries
            self._matrix = matrix
            self._index = index
            self._has_expiring_entries = has_expiring
        return len(clean_entries)

    def add(self, entry: dict) -> bool:
        """Add one entry by rebuilding from the current snapshot plus the new row."""
        if not isinstance(entry, dict) or self._is_expired(entry):
            return False
        with self._lock:
            entries = list(self._entries)
        entries.append(entry)
        self.rebuild(entries)
        return True

    def remove(self, sample_key: str) -> bool:
        sample_key = str(sample_key)
        with self._lock:
            entries = [e for e in self._entries if str(e.get("sample_key")) != sample_key]
            changed = len(entries) != len(self._entries)
        if changed:
            self.rebuild(entries)
        return changed

    def prune_expired(self) -> int:
        with self._lock:
            entries = list(self._entries)
        return self.rebuild(entries)

    def touch_owner(self, owner_id, expires_at: float) -> int:
        owner_id = str(owner_id)
        changed = False
        with self._lock:
            entries = []
            for entry in self._entries:
                updated = dict(entry)
                if updated.get("owner_id") == owner_id and updated.get("expires_at") is not None:
                    updated["expires_at"] = float(expires_at)
                    changed = True
                entries.append(updated)
        if changed:
            return self.rebuild(entries)
        return self.count()

    def search(self, embedding, threshold: float, top_k: int = 1):
        query = self._normalize(embedding)
        if query is None or query.shape[0] != self.dim:
            return None, 0.0, None

        with self._lock:
            entries = list(self._entries)
            matrix = self._matrix
            index = self._index
            has_expiring = self._has_expiring_entries

        if len(entries) == 0 or matrix.shape[0] == 0:
            return None, 0.0, None

        now = time.time()
        search_k = int(max(1, top_k))
        if has_expiring:
            search_k = len(entries)
        else:
            search_k = min(search_k, len(entries))

        if index is not None:
            similarities, indices = index.search(query.reshape(1, -1).astype(np.float32), search_k)
            candidates = zip(indices[0], similarities[0])
        else:
            sims = matrix @ query
            order = np.argsort(-sims)[:search_k]
            candidates = ((int(i), float(sims[i])) for i in order)

        best_id = None
        best_similarity = 0.0
        best_entry = None
        for idx, similarity in candidates:
            idx = int(idx)
            if idx < 0 or idx >= len(entries):
                continue
            entry = entries[idx]
            if self._is_expired(entry, now):
                continue
            similarity = float(similarity)
            if similarity >= threshold and similarity > best_similarity:
                best_id = entry["owner_id"]
                best_similarity = similarity
                best_entry = entry

        return best_id, best_similarity, best_entry

    def get_features_by_id(self, owner_id) -> Optional[np.ndarray]:
        owner_id = str(owner_id)
        now = time.time()
        with self._lock:
            vectors = [
                np.asarray(entry["embedding"], dtype=np.float32).reshape(-1)
                for entry in self._entries
                if entry.get("owner_id") == owner_id and not self._is_expired(entry, now)
            ]
        if not vectors:
            return None
        return np.vstack(vectors).astype(np.float32, copy=False)

    def count_by_id(self, owner_id) -> int:
        owner_id = str(owner_id)
        now = time.time()
        with self._lock:
            return sum(
                1
                for entry in self._entries
                if entry.get("owner_id") == owner_id and not self._is_expired(entry, now)
            )

    def count(self) -> int:
        now = time.time()
        with self._lock:
            return sum(1 for entry in self._entries if not self._is_expired(entry, now))


class LocalIdentityCache:
    """Version-synced worker-local identity indexes."""

    KINDS = ("known_face", "known_reid", "unknown_face", "unknown_reid")

    def __init__(self, redis_memory, sync_interval: float = 0.75):
        self.redis_memory = redis_memory
        self.sync_interval = float(sync_interval)
        self.known_face = LocalVectorIndex(512, "known_face")
        self.known_reid = LocalVectorIndex(768, "known_reid")
        self.unknown_face = LocalVectorIndex(512, "unknown_face")
        self.unknown_reid = LocalVectorIndex(768, "unknown_reid")
        self._lock = threading.RLock()
        self._versions: Dict[str, int] = {kind: -1 for kind in self.KINDS}
        self._last_sync_check = 0.0

    def _redis_available(self) -> bool:
        return bool(self.redis_memory is not None and getattr(self.redis_memory, "available", False))

    def _index_for_kind(self, kind: str) -> LocalVectorIndex:
        return {
            "known_face": self.known_face,
            "known_reid": self.known_reid,
            "unknown_face": self.unknown_face,
            "unknown_reid": self.unknown_reid,
        }[kind]

    def maybe_sync(self, force: bool = False) -> bool:
        if not self._redis_available():
            return False

        now = time.time()
        with self._lock:
            if not force and now - self._last_sync_check < self.sync_interval:
                self.unknown_face.prune_expired()
                self.unknown_reid.prune_expired()
                return False
            self._last_sync_check = now

        try:
            versions = self.redis_memory.get_versions()
        except Exception as e:
            LOGGER.warning("identity version sync failed: %s", e)
            return False

        changed = []
        with self._lock:
            for kind in self.KINDS:
                version = int(versions.get(kind, 0))
                if force or version != self._versions.get(kind, -1):
                    changed.append(kind)

        if not changed:
            self.unknown_face.prune_expired()
            self.unknown_reid.prune_expired()
            return False

        return self.force_sync(kind=changed, versions=versions)

    def force_sync(self, kind=None, versions: Optional[Dict[str, int]] = None) -> bool:
        if not self._redis_available():
            return False

        if kind is None:
            kinds = list(self.KINDS)
        elif isinstance(kind, (list, tuple, set)):
            kinds = list(kind)
        else:
            kinds = [kind]

        try:
            versions = versions or self.redis_memory.get_versions()
            for one_kind in kinds:
                if one_kind == "known_face":
                    entries = self.redis_memory.list_known_faces()
                elif one_kind == "known_reid":
                    entries = self.redis_memory.list_known_reid_all()
                elif one_kind == "unknown_face":
                    entries = self.redis_memory.list_unknown_faces()
                elif one_kind == "unknown_reid":
                    entries = self.redis_memory.list_unknown_reid()
                else:
                    continue
                self._index_for_kind(one_kind).rebuild(entries)
                with self._lock:
                    self._versions[one_kind] = int(versions.get(one_kind, 0))
            return True
        except Exception as e:
            LOGGER.warning("identity force sync failed: %s", e)
            return False

    def get_versions(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._versions)

    def loaded_counts(self) -> Dict[str, int]:
        return {
            "known_face": self.known_face.count(),
            "known_reid": self.known_reid.count(),
            "unknown_face": self.unknown_face.count(),
            "unknown_reid": self.unknown_reid.count(),
        }

    def search_known_face(self, embedding, threshold: float, top_k: int = 1):
        owner_id, similarity, _ = self.known_face.search(embedding, threshold, top_k)
        return owner_id, similarity

    def search_known_reid(self, embedding, threshold: float, top_k: int = 1):
        owner_id, similarity, _ = self.known_reid.search(embedding, threshold, top_k)
        return owner_id, similarity

    def search_unknown_face(self, embedding, threshold: float, top_k: int = 1):
        owner_id, similarity, _ = self.unknown_face.search(embedding, threshold, top_k)
        return owner_id, similarity

    def search_unknown_reid(self, embedding, threshold: float, top_k: int = 1):
        owner_id, similarity, _ = self.unknown_reid.search(embedding, threshold, top_k)
        return owner_id, similarity

    def get_known_reid_features(self, person_id):
        return self.known_reid.get_features_by_id(person_id)

    def get_unknown_reid_features(self, unknown_id):
        return self.unknown_reid.get_features_by_id(unknown_id)

    def add_local_unknown_face(self, unknown_id, embedding, sample_key=None, expires_at=None):
        entry = {
            "sample_key": sample_key or f"local:unknown_face:{unknown_id}:{time.time_ns()}",
            "unknown_id": str(unknown_id),
            "embedding": embedding,
            "created_at": time.time(),
            "expires_at": expires_at or (time.time() + getattr(self.redis_memory, "unknown_ttl", 300)),
        }
        try:
            return self.unknown_face.add(entry)
        except Exception as e:
            LOGGER.warning("failed to add local unknown face sample: %s", e)
            return False

    def add_local_unknown_reid(self, unknown_id, embedding, sample_key=None, expires_at=None):
        entry = {
            "sample_key": sample_key or f"local:unknown_reid:{unknown_id}:{time.time_ns()}",
            "unknown_id": str(unknown_id),
            "embedding": embedding,
            "created_at": time.time(),
            "expires_at": expires_at or (time.time() + getattr(self.redis_memory, "unknown_ttl", 300)),
        }
        try:
            return self.unknown_reid.add(entry)
        except Exception as e:
            LOGGER.warning("failed to add local unknown reid sample: %s", e)
            return False

    def add_local_known_reid(self, person_id, embedding, sample_key=None, is_anchor=False, expires_at=None):
        entry = {
            "sample_key": sample_key or f"local:known_reid:{person_id}:{time.time_ns()}",
            "person_id": str(person_id),
            "embedding": embedding,
            "created_at": time.time(),
            "is_anchor": bool(is_anchor),
        }
        if expires_at is not None:
            entry["expires_at"] = expires_at
        try:
            return self.known_reid.add(entry)
        except Exception as e:
            LOGGER.warning("failed to add local known reid sample: %s", e)
            return False

    def touch_unknown(self, unknown_id, ttl_seconds=None):
        ttl_seconds = ttl_seconds or getattr(self.redis_memory, "unknown_ttl", 300)
        expires_at = time.time() + float(ttl_seconds)
        self.unknown_face.touch_owner(unknown_id, expires_at)
        self.unknown_reid.touch_owner(unknown_id, expires_at)
