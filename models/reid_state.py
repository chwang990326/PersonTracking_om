import os
import threading
from collections import defaultdict

import cv2


class SharedIdentityStore:
    """Service-level shared gallery state for all camera-local reidentifiers."""

    def __init__(self, identity_folder, feature_extractor):
        self.identity_folder = identity_folder
        self.feature_extractor = feature_extractor
        self.lock = threading.RLock()
        self.feature_gallery = {}
        self.anchor_gallery = {}
        self.gallery_files = {}
        self.anchor_files = {}
        self._last_signature = tuple()
        self.reload(verbose=True)

    @staticmethod
    def _normalize_person_id(person_id_str):
        try:
            return int(person_id_str)
        except ValueError:
            return person_id_str

    @staticmethod
    def _is_temporary_identity(person_id):
        return isinstance(person_id, int) or (
            isinstance(person_id, str) and person_id.isdigit()
        )

    @classmethod
    def scan_identity_entries(cls, identity_folder):
        if not os.path.isdir(identity_folder):
            os.makedirs(identity_folder, exist_ok=True)

        entries = []
        for person_id_str in sorted(os.listdir(identity_folder)):
            person_folder = os.path.join(identity_folder, person_id_str)
            if not os.path.isdir(person_folder):
                continue

            person_id = cls._normalize_person_id(person_id_str)
            if cls._is_temporary_identity(person_id):
                continue

            filenames = sorted(
                f for f in os.listdir(person_folder)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            )
            for filename in filenames:
                file_path = os.path.join(person_folder, filename)
                try:
                    stat = os.stat(file_path)
                except OSError:
                    continue
                entries.append(
                    {
                        "person_id": person_id,
                        "person_id_str": person_id_str,
                        "filename": filename,
                        "path": file_path,
                        "size": stat.st_size,
                        "mtime_ns": stat.st_mtime_ns,
                    }
                )
        return entries

    @staticmethod
    def signature_from_entries(entries):
        return tuple(
            (entry["person_id_str"], entry["filename"], entry["size"], entry["mtime_ns"])
            for entry in entries
        )

    @classmethod
    def load_gallery_from_entries(cls, entries, feature_extractor, verbose=True):
        grouped_entries = defaultdict(list)
        for entry in entries:
            grouped_entries[entry["person_id"]].append(entry)

        gallery = {}
        anchor_gallery = {}
        gallery_files = {}
        anchor_files = {}

        for person_id, person_entries in grouped_entries.items():
            valid_filenames = []
            valid_images = []
            for entry in person_entries:
                image = cv2.imread(entry["path"])
                if image is None:
                    continue
                valid_filenames.append(entry["filename"])
                valid_images.append(image)

            if not valid_images:
                continue

            features = feature_extractor(list(valid_images))
            all_features = features.cpu().numpy()
            gallery[person_id] = all_features
            gallery_files[person_id] = list(valid_filenames)

            anchor_indices = []
            anchor_names = []
            for index, filename in enumerate(valid_filenames):
                if filename.startswith("anchor_"):
                    anchor_indices.append(index)
                    anchor_names.append(filename)
            if anchor_indices:
                anchor_gallery[person_id] = all_features[anchor_indices]
                anchor_files[person_id] = anchor_names

            if verbose:
                anchor_count = len(anchor_indices)
                log_msg = f"[ReID] ID {person_id}: 加载 {len(valid_images)} 张样本"
                if anchor_count > 0:
                    log_msg += f" (含 {anchor_count} 张锚点)"
                print(log_msg + "。")

        return gallery, anchor_gallery, gallery_files, anchor_files

    def _apply_gallery_state(self, gallery, anchor_gallery, gallery_files, anchor_files, signature):
        with self.lock:
            self.feature_gallery.clear()
            self.feature_gallery.update(gallery)
            self.anchor_gallery.clear()
            self.anchor_gallery.update(anchor_gallery)
            self.gallery_files.clear()
            self.gallery_files.update(gallery_files)
            self.anchor_files.clear()
            self.anchor_files.update(anchor_files)
            self._last_signature = signature

    def reload(self, verbose=True):
        entries = self.scan_identity_entries(self.identity_folder)
        signature = self.signature_from_entries(entries)
        gallery, anchor_gallery, gallery_files, anchor_files = self.load_gallery_from_entries(
            entries,
            self.feature_extractor,
            verbose=verbose,
        )
        self._apply_gallery_state(gallery, anchor_gallery, gallery_files, anchor_files, signature)

        if not gallery:
            print("[ReID] 警告: 特征画廊为空，将无法识别任何人。请检查'identity'文件夹结构。")
        elif verbose:
            print(f"[ReID] 特征画廊构建完成，共包含 {len(gallery)} 个已知身份。")
        return True

    def refresh_if_changed(self, verbose=True):
        entries = self.scan_identity_entries(self.identity_folder)
        signature = self.signature_from_entries(entries)
        with self.lock:
            if signature == self._last_signature:
                return False

        gallery, anchor_gallery, gallery_files, anchor_files = self.load_gallery_from_entries(
            entries,
            self.feature_extractor,
            verbose=verbose,
        )
        self._apply_gallery_state(gallery, anchor_gallery, gallery_files, anchor_files, signature)

        if not gallery:
            print("[ReID] 警告: 特征画廊为空，将无法识别任何人。请检查'identity'文件夹结构。")
        elif verbose:
            print(f"[ReID] identity 库变更，已热更新特征画廊，共包含 {len(gallery)} 个已知身份。")
        return True
