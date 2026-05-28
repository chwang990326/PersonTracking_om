"""Service-level shared store for temporary unknown identities."""

import os
import shutil
import threading
import time
from datetime import datetime

import cv2
import numpy as np

from models.face_database import FaceDatabase


class UnknownEntityStore:
    def __init__(
        self,
        face_folder='unknownFace',
        reid_folder='unknownIdentity',
        face_db_path='./database/unknown_face_database',
        temp_id_start=0,
        temp_id_end=99,
        ttl_seconds=300.0,
        face_similarity_threshold=0.5,
        redis_memory=None,
        identity_cache=None,
    ):
        self.face_folder = face_folder
        self.reid_folder = reid_folder
        self.face_db_path = face_db_path
        self.temp_id_start = temp_id_start
        self.temp_id_end = temp_id_end
        self.ttl_seconds = float(ttl_seconds)
        self.face_similarity_threshold = face_similarity_threshold
        self.redis_memory = redis_memory
        self.identity_cache = identity_cache
        self.max_face_images = 3
        self.lock = threading.RLock()
        self.entities = {}
        self.face_embeddings = {}
        self.face_files = {}
        self.reid_feature_gallery = {}
        self.reid_gallery_files = {}
        self.face_db = FaceDatabase(db_path=self.face_db_path)
        self._next_local_id = self.temp_id_start
        self._last_touch_time = {}

        os.makedirs(self.face_folder, exist_ok=True)
        os.makedirs(self.reid_folder, exist_ok=True)

    def _redis_available(self):
        return bool(self.redis_memory is not None and getattr(self.redis_memory, "available", False))

    @staticmethod
    def _normalize_id(unknown_id):
        if unknown_id in (-1, None):
            return None
        return str(unknown_id)

    def _remember_entity_locked(self, unknown_id):
        now = time.time()
        entity = self.entities.setdefault(
            str(unknown_id),
            {
                'unknown_id': str(unknown_id),
                'last_seen_time': now,
                'face_count': 0,
                'reid_count': 0,
            },
        )
        entity['last_seen_time'] = now
        entity['face_count'] = len(self.face_files.get(str(unknown_id), []))
        entity['reid_count'] = len(self.reid_gallery_files.get(str(unknown_id), []))
        return entity

    def clear_all(self):
        """Management-only clear. Service reload does not call this by default."""
        with self.lock:
            for folder in (self.face_folder, self.reid_folder):
                if os.path.isdir(folder):
                    shutil.rmtree(folder, ignore_errors=True)
                os.makedirs(folder, exist_ok=True)

            if os.path.isdir(self.face_db_path):
                shutil.rmtree(self.face_db_path, ignore_errors=True)

            self.entities.clear()
            self.face_embeddings.clear()
            self.face_files.clear()
            self.reid_feature_gallery.clear()
            self.reid_gallery_files.clear()
            self.face_db = FaceDatabase(db_path=self.face_db_path)

        if self._redis_available():
            self.redis_memory.flush_unknowns()

    def allocate_id(self):
        """Allocate a globally unique numeric unknown ID when Redis is available."""
        if self._redis_available():
            unknown_id = self.redis_memory.allocate_unknown_id()
            with self.lock:
                self._remember_entity_locked(unknown_id)
            return unknown_id

        with self.lock:
            for _ in range(self.temp_id_end - self.temp_id_start + 1):
                unknown_id = str(self._next_local_id)
                self._next_local_id += 1
                if self._next_local_id > self.temp_id_end:
                    self._next_local_id = self.temp_id_start
                if unknown_id not in self.entities:
                    self._remember_entity_locked(unknown_id)
                    return unknown_id
        return -1

    def touch_entity(self, unknown_id):
        unknown_id = self._normalize_id(unknown_id)
        if unknown_id is None:
            return

        now = time.time()
        if self.identity_cache is not None:
            self.identity_cache.touch_unknown(unknown_id, self.ttl_seconds)
        with self.lock:
            self._remember_entity_locked(unknown_id)
            last_touch = self._last_touch_time.get(unknown_id, 0.0)
            if now - last_touch < 1.0:
                return
            self._last_touch_time[unknown_id] = now

        if self._redis_available():
            try:
                self.redis_memory.touch_unknown(unknown_id)
            except Exception as e:
                print(f"[UnknownEntityStore] touch_unknown failed: {e}")

    def release_if_empty(self, unknown_id):
        unknown_id = self._normalize_id(unknown_id)
        if unknown_id is None:
            return

        if self._redis_available():
            try:
                self.redis_memory.release_if_empty(unknown_id)
            except Exception as e:
                print(f"[UnknownEntityStore] release_if_empty failed: {e}")

        with self.lock:
            has_face = len(self.face_files.get(unknown_id, [])) > 0
            has_reid = len(self.reid_gallery_files.get(unknown_id, [])) > 0
            if not has_face and not has_reid:
                self._remove_entity_locked(unknown_id)

    def release_entity(self, unknown_id):
        unknown_id = self._normalize_id(unknown_id)
        if unknown_id is None:
            return

        if self._redis_available():
            try:
                self.redis_memory.release_unknown(unknown_id)
            except Exception as e:
                print(f"[UnknownEntityStore] release_unknown failed: {e}")

        with self.lock:
            self._remove_entity_locked(unknown_id)

    def cleanup_stale(self):
        """Local-only cleanup. Redis cleanup is a manual/background task."""
        now = time.time()
        with self.lock:
            stale_ids = [
                unknown_id
                for unknown_id, entity in self.entities.items()
                if now - entity.get('last_seen_time', 0.0) > self.ttl_seconds
            ]
            for unknown_id in stale_ids:
                self._remove_entity_locked(unknown_id)

    def _remove_entity_locked(self, unknown_id):
        unknown_id = str(unknown_id)
        self.entities.pop(unknown_id, None)
        self.face_embeddings.pop(unknown_id, None)
        self.face_files.pop(unknown_id, None)
        self.reid_feature_gallery.pop(unknown_id, None)
        self.reid_gallery_files.pop(unknown_id, None)
        self._last_touch_time.pop(unknown_id, None)

        for folder in (self.face_folder, self.reid_folder):
            target_dir = os.path.join(folder, unknown_id)
            if os.path.isdir(target_dir):
                shutil.rmtree(target_dir, ignore_errors=True)

        self._rebuild_face_db_locked()

    def search_face_embedding(self, embedding):
        if self.identity_cache is not None:
            unknown_id, similarity = self.identity_cache.search_unknown_face(
                embedding,
                self.face_similarity_threshold,
            )
            if unknown_id is not None:
                return str(unknown_id), similarity

        with self.lock:
            if self.face_db is None:
                return None, 0.0
            person_id, similarity = self.face_db.search(embedding, self.face_similarity_threshold)
        if person_id == 'Unknown':
            return None, similarity
        return str(person_id), similarity

    def claim_by_reid(self, feature, threshold):
        return self._claim_unknown(feature, threshold, modality="reid")

    def claim_by_face(self, embedding, threshold):
        return self._claim_unknown(embedding, threshold, modality="face")

    def _claim_unknown(self, embedding, threshold, modality):
        cache_kind = f"unknown_{modality}"
        search_fn = (
            self.identity_cache.search_unknown_face
            if modality == "face"
            else self.identity_cache.search_unknown_reid
        ) if self.identity_cache is not None else None
        add_fn = (
            self.redis_memory.add_unknown_face_sample
            if modality == "face"
            else self.redis_memory.add_unknown_reid_sample
        ) if self._redis_available() else None

        if search_fn is not None:
            unknown_id, similarity = search_fn(embedding, threshold)
            if unknown_id is not None:
                self.touch_entity(unknown_id)
                return str(unknown_id), similarity

        if not self._redis_available() or self.identity_cache is None:
            unknown_id = self.allocate_id()
            if unknown_id == -1:
                return None, 0.0
            self._add_local_claim_sample(unknown_id, embedding, modality)
            return str(unknown_id), 0.0

        lock_name = f"claim_unknown_{modality}"
        token = None
        retry_sleep = min(0.02, max(0.01, self.redis_memory.claim_lock_retry_sleep_ms / 1000.0))
        max_retries = min(3, max(1, self.redis_memory.claim_lock_max_retries))

        for _ in range(max_retries):
            token = self.redis_memory.acquire_lock(lock_name, self.redis_memory.claim_lock_ttl_ms)
            if token is not None:
                break
            self.identity_cache.force_sync(cache_kind)
            unknown_id, similarity = search_fn(embedding, threshold)
            if unknown_id is not None:
                self.touch_entity(unknown_id)
                return str(unknown_id), similarity
            time.sleep(retry_sleep)

        if token is None:
            return None, 0.0

        try:
            self.identity_cache.force_sync(cache_kind)
            unknown_id, similarity = search_fn(embedding, threshold)
            if unknown_id is not None:
                self.touch_entity(unknown_id)
                return str(unknown_id), similarity

            unknown_id = self.allocate_id()
            if unknown_id == -1:
                return None, 0.0
            entry = add_fn(unknown_id, embedding)
            if entry:
                if modality == "face":
                    self.identity_cache.add_local_unknown_face(
                        unknown_id,
                        embedding,
                        sample_key=entry.get("sample_key"),
                        expires_at=entry.get("expires_at"),
                    )
                else:
                    self.identity_cache.add_local_unknown_reid(
                        unknown_id,
                        embedding,
                        sample_key=entry.get("sample_key"),
                        expires_at=entry.get("expires_at"),
                    )
            self._add_local_claim_sample(unknown_id, embedding, modality)
            return str(unknown_id), 0.0
        finally:
            self.redis_memory.release_lock(lock_name, token)

    def _add_local_claim_sample(self, unknown_id, embedding, modality):
        unknown_id = str(unknown_id)
        with self.lock:
            self._remember_entity_locked(unknown_id)
            if modality == "reid":
                feat = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
                current = self.reid_feature_gallery.get(unknown_id)
                if current is None:
                    self.reid_feature_gallery[unknown_id] = feat
                else:
                    self.reid_feature_gallery[unknown_id] = np.vstack((current, feat))

    def add_face_sample(self, unknown_id, face_image, embedding):
        if unknown_id in (-1, None) or face_image is None or face_image.size == 0:
            return False

        unknown_id = str(unknown_id)
        redis_entry = None
        redis_success = False
        if self._redis_available():
            try:
                redis_entry = self.redis_memory.add_unknown_face_sample(unknown_id, embedding)
                redis_success = bool(redis_entry)
            except Exception as e:
                print(f"[UnknownEntityStore] add_unknown_face_sample failed: {e}")
                return False

        with self.lock:
            self._remember_entity_locked(unknown_id)
            embeddings = list(self.face_embeddings.get(unknown_id, []))
            files = list(self.face_files.get(unknown_id, []))
            for existing_embedding in embeddings:
                sim = float(
                    np.dot(existing_embedding, embedding)
                    / ((np.linalg.norm(existing_embedding) * np.linalg.norm(embedding)) + 1e-12)
                )
                if sim > 0.95 and files:
                    return True

            target_dir = os.path.join(self.face_folder, unknown_id)
            os.makedirs(target_dir, exist_ok=True)

            if len(files) >= self.max_face_images:
                oldest = files.pop(0)
                oldest_path = os.path.join(target_dir, oldest)
                if os.path.exists(oldest_path):
                    os.remove(oldest_path)
                if embeddings:
                    embeddings.pop(0)

            filename = self._generate_filename(target_dir, 'face')
            save_path = os.path.join(target_dir, filename)
            cv2.imwrite(save_path, face_image)

            embeddings.append(np.asarray(embedding, dtype=np.float32).copy())
            files.append(filename)
            self.face_embeddings[unknown_id] = embeddings
            self.face_files[unknown_id] = files
            self.entities[unknown_id]['face_count'] = len(files)
            self._rebuild_face_db_locked()

        if self.identity_cache is not None and (redis_success or redis_entry):
            try:
                self.identity_cache.add_local_unknown_face(
                    unknown_id,
                    embedding,
                    sample_key=redis_entry.get("sample_key") if isinstance(redis_entry, dict) else None,
                    expires_at=redis_entry.get("expires_at") if isinstance(redis_entry, dict) else None,
                )
            except Exception as e:
                print(f"[UnknownEntityStore] local unknown face add failed: {e}")

        return True

    def _rebuild_face_db_locked(self):
        self.face_db = FaceDatabase(db_path=self.face_db_path)
        for unknown_id, embeddings in self.face_embeddings.items():
            for embedding in embeddings:
                self.face_db.add_face(embedding, str(unknown_id))

    @staticmethod
    def _generate_filename(target_dir, prefix):
        base_name = datetime.now().strftime(f"{prefix}_%Y%m%d_%H%M%S")
        filename = f"{base_name}.jpg"
        candidate = os.path.join(target_dir, filename)
        suffix = 1
        while os.path.exists(candidate):
            filename = f"{base_name}_{suffix}.jpg"
            candidate = os.path.join(target_dir, filename)
            suffix += 1
        return filename
