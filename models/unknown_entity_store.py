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
    ):
        """
        初始化未知身份存储器，设置文件夹路径、ID范围、TTL和相似度阈值。
            - face_folder: 存储未知身份面部图片的文件夹路径。
            - reid_folder: 存储未知身份ReID特征的文件夹路径。
            - face_db_path: 面部数据库的存储路径。
            - temp_id_start: 分配未知身份ID的起始值。
            - temp_id_end: 分配未知身份ID的结束值。
            - ttl_seconds: 未知身份的TTL（生存时间）阈值，单位为秒。
            - face_similarity_threshold: 面部相似度的阈值，用于判断是否将新的面部样本添加到现有未知身份中。
        """
        self.face_folder = face_folder
        self.reid_folder = reid_folder
        self.face_db_path = face_db_path
        self.temp_id_start = temp_id_start
        self.temp_id_end = temp_id_end
        self.ttl_seconds = ttl_seconds
        self.face_similarity_threshold = face_similarity_threshold
        self.max_face_images = 3
        self.lock = threading.RLock()
        self.entities = {}
        self.face_embeddings = {}
        self.face_files = {}
        self.reid_feature_gallery = {}
        self.reid_gallery_files = {}
        self.face_db = None
        self.clear_all()

    def clear_all(self):
        """
        清理所有未知身份数据，包括文件系统中的图片和数据库中的记录。
        这个方法会删除所有未知身份的相关数据，并重置内部状态。
        """
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

    def allocate_id(self):
        """
        分配一个新的未知身份ID，范围在 [temp_id_start, temp_id_end] 之间。
        """
        with self.lock:
            for unknown_id in range(self.temp_id_start, self.temp_id_end + 1):
                if unknown_id in self.entities:
                    continue
                self.entities[unknown_id] = {
                    'unknown_id': unknown_id,
                    'last_seen_time': time.time(),
                    'face_count': 0,
                    'reid_count': 0,
                }
                return unknown_id
        return -1

    def touch_entity(self, unknown_id):
        """
        更新未知身份的最后访问时间和相关统计信息。
        """
        if unknown_id in (-1, None):
            return
        unknown_id = int(unknown_id)
        with self.lock:
            entity = self.entities.setdefault(
                unknown_id,
                {
                    'unknown_id': unknown_id,
                    'last_seen_time': time.time(),
                    'face_count': 0,
                    'reid_count': 0,
                },
            )
            entity['last_seen_time'] = time.time()
            entity['face_count'] = len(self.face_files.get(unknown_id, []))
            entity['reid_count'] = len(self.reid_gallery_files.get(unknown_id, []))

    def release_if_empty(self, unknown_id):
        """
        如果指定的未知身份ID没有任何关联的面部图片或ReID特征，则释放该ID。
        """
        if unknown_id in (-1, None):
            return
        unknown_id = int(unknown_id)
        with self.lock:
            has_face = len(self.face_files.get(unknown_id, [])) > 0
            has_reid = len(self.reid_gallery_files.get(unknown_id, [])) > 0
            if not has_face and not has_reid:
                self._remove_entity_locked(unknown_id)

    def release_entity(self, unknown_id):
        """
        直接释放指定的未知身份ID，无论其是否有关联的数据。
        """
        if unknown_id in (-1, None):
            return
        with self.lock:
            self._remove_entity_locked(int(unknown_id))

    def cleanup_stale(self):
        """
        清理所有过期的未知身份ID，这些ID的最后访问时间超过了TTL阈值。
        """
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
        """
        内部方法：在持有锁的情况下删除指定的未知身份ID及其相关数据。
        """
        self.entities.pop(unknown_id, None)
        self.face_embeddings.pop(unknown_id, None)
        self.face_files.pop(unknown_id, None)
        self.reid_feature_gallery.pop(unknown_id, None)
        self.reid_gallery_files.pop(unknown_id, None)

        for folder in (self.face_folder, self.reid_folder):
            target_dir = os.path.join(folder, str(unknown_id))
            if os.path.isdir(target_dir):
                shutil.rmtree(target_dir, ignore_errors=True)

        self._rebuild_face_db_locked()

    def search_face_embedding(self, embedding):
        """
        在未知身份库中搜索与给定面部特征向量相似的条目。
        返回匹配的未知身份ID和相似度，如果没有匹配则返回 (None, 0.0)。
        """
        with self.lock:
            if self.face_db is None:
                return None, 0.0
            person_id, similarity = self.face_db.search(embedding, self.face_similarity_threshold)
        if person_id == 'Unknown':
            return None, similarity
        try:
            return int(person_id), similarity
        except (TypeError, ValueError):
            return None, similarity

    def add_face_sample(self, unknown_id, face_image, embedding):
        """
        将一个新的面部样本添加到未知身份库中指定ID的条目下。
         - 如果该ID不存在，则创建一个新的条目；
        """
        if unknown_id in (-1, None) or face_image is None or face_image.size == 0:
            return False

        unknown_id = int(unknown_id)
        with self.lock:
            self.touch_entity(unknown_id)
            embeddings = list(self.face_embeddings.get(unknown_id, []))
            for existing_embedding in embeddings:
                sim = float(
                    np.dot(existing_embedding, embedding)
                    / ((np.linalg.norm(existing_embedding) * np.linalg.norm(embedding)) + 1e-12)
                )
                if sim > 0.95:
                    return False

            target_dir = os.path.join(self.face_folder, str(unknown_id))
            os.makedirs(target_dir, exist_ok=True)

            files = list(self.face_files.get(unknown_id, []))
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

            embeddings.append(embedding.copy())
            files.append(filename)
            self.face_embeddings[unknown_id] = embeddings
            self.face_files[unknown_id] = files
            self.entities[unknown_id]['face_count'] = len(files)
            self._rebuild_face_db_locked()
            return True

    def _rebuild_face_db_locked(self):
        """
        内部方法：在持有锁的情况下重建面部数据库索引。
         - 这个方法会清空当前的面部数据库，并将未知身份库中的所有面部样本重新添加到数据库中，以确保搜索功能能够正确地访问最新的数据。
        """
        self.face_db = FaceDatabase(db_path=self.face_db_path)
        for unknown_id, embeddings in self.face_embeddings.items():
            for embedding in embeddings:
                self.face_db.add_face(embedding, str(unknown_id))

    @staticmethod
    def _generate_filename(target_dir, prefix):
        """
        生成一个唯一的文件名，格式为 "{prefix}_YYYYMMDD_HHMMSS.jpg"。
         - 如果在同一秒内生成多个文件，则会在文件名后添加一个递增的后缀，以避免文件名冲突。
        """
        base_name = datetime.now().strftime(f"{prefix}_%Y%m%d_%H%M%S")
        filename = f"{base_name}.jpg"
        candidate = os.path.join(target_dir, filename)
        suffix = 1
        while os.path.exists(candidate):
            filename = f"{base_name}_{suffix}.jpg"
            candidate = os.path.join(target_dir, filename)
            suffix += 1
        return filename

