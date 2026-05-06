from collections import deque

import numpy as np


class TimeLockManager:
    def __init__(
        self,
        enabled=True,
        window_frames=15,
        normalize_identity=None,
        is_temporary_identity=None,
    ):
        self.enabled = bool(enabled)
        self.window_frames = max(1, int(window_frames))
        self._normalize_identity = normalize_identity or self._default_normalize_identity
        self._is_temporary_identity = is_temporary_identity or self._default_is_temporary_identity

    @staticmethod
    def _default_normalize_identity(person_id):
        if isinstance(person_id, str):
            person_id = person_id.strip()
            if person_id.isdigit():
                return int(person_id)
            return person_id
        return person_id

    @staticmethod
    def _default_is_temporary_identity(person_id):
        return isinstance(person_id, int) or (isinstance(person_id, str) and person_id.isdigit())

    def sync_config(self, enabled, window_frames):
        self.enabled = bool(enabled)
        self.window_frames = max(1, int(window_frames))

    def create_state(self):
        return {
            'frame_seq': 0,
            'person_count_history': deque(maxlen=self.window_frames),
            'track_birth_state': {},
            'reid_time_lock': {
                'active': False,
                'locked_k': 0,
                'member_ids': [],
                'member_features': {},
                'locked_at_frame': None,
                'last_high_conf_new_track_frame': None,
            },
        }

    def reset_if_disabled(self, state):
        if self.enabled:
            return
        lock_state = state['reid_time_lock']
        lock_state['last_high_conf_new_track_frame'] = None
        if lock_state.get('active'):
            self._clear_lock_state(lock_state)

    def begin_frame(self, state, reidentifier):
        self._ensure_history(state)
        if not self.enabled:
            return None

        state['frame_seq'] += 1
        if self.is_active(state):
            self.refresh_member_features(reidentifier, state)
        return state['frame_seq']

    def is_active(self, state):
        lock_state = state['reid_time_lock']
        return bool(self.enabled and lock_state.get('active') and lock_state.get('member_ids'))

    def contains_identity(self, state, person_id):
        if not self.is_active(state):
            return False

        person_key = self._person_key(person_id)
        if person_key is None:
            return False

        return person_key in {
            self._person_key(member_id)
            for member_id in state['reid_time_lock'].get('member_ids', [])
        }

    def allows_known_face_identity(self, state, person_id):
        person_key = self._person_key(person_id)
        if person_key is None:
            return False
        if not self.is_active(state):
            return True
        return self.contains_identity(state, person_id) or self._is_known_identity(person_id)

    def admit_known_face_identity(self, state, known_person_id, previous_person_id=None):
        if not self.is_active(state):
            return False

        normalized_known_id = self._normalize_identity(known_person_id)
        if not self._is_known_identity(normalized_known_id):
            return False

        lock_state = state['reid_time_lock']
        member_ids = list(lock_state.get('member_ids', []))
        previous_member_key = self._person_key(previous_person_id)
        known_member_key = self._person_key(normalized_known_id)
        replaced_previous_member = False

        if previous_member_key is not None and self._is_temporary_identity(previous_person_id):
            for idx, member_id in enumerate(member_ids):
                if self._person_key(member_id) == previous_member_key:
                    if known_member_key in {self._person_key(item) for item in member_ids}:
                        member_ids.pop(idx)
                    else:
                        member_ids[idx] = normalized_known_id
                    replaced_previous_member = True
                    break
        if not replaced_previous_member and known_member_key not in {self._person_key(item) for item in member_ids}:
            member_ids.append(normalized_known_id)

        normalized_member_ids = []
        member_keys = set()
        for member_id in member_ids:
            member_key = self._person_key(member_id)
            if member_key is None or member_key in member_keys:
                continue
            member_keys.add(member_key)
            normalized_member_ids.append(self._normalize_identity(member_id))

        previous_keys = {
            self._person_key(member_id)
            for member_id in lock_state.get('member_ids', [])
        }
        if previous_keys == member_keys:
            return False

        lock_state['member_ids'] = normalized_member_ids
        lock_state['locked_k'] = len(normalized_member_ids)
        lock_state['member_features'] = {}
        return True

    def get_locked_temp_ids(self, state):
        if not self.is_active(state):
            return set()

        return {
            int(member_id)
            for member_id in state['reid_time_lock'].get('member_ids', [])
            if self._is_temporary_identity(member_id)
        }

    def get_protected_person_ids(self, state):
        if not self.is_active(state):
            return None
        return list(state['reid_time_lock'].get('member_ids', []))

    def build_context(self, state):
        if not self.is_active(state):
            return None

        member_features = {}
        for person_key, features in state['reid_time_lock'].get('member_features', {}).items():
            if features is None:
                continue
            member_features[str(person_key)] = features.copy()

        return {
            'active': True,
            'member_ids': list(state['reid_time_lock'].get('member_ids', [])),
            'member_features': member_features,
        }

    def refresh_member_features(self, reidentifier, state):
        lock_state = state['reid_time_lock']
        if not self.is_active(state):
            lock_state['member_features'] = {}
            return

        existing_features = lock_state.get('member_features', {})
        current_features = {}
        member_keys = {
            self._person_key(member_id)
            for member_id in lock_state.get('member_ids', [])
        }

        for track_info in reidentifier.track_mapper.values():
            person_id = track_info.get('person_id')
            person_key = self._person_key(person_id)
            if person_key is None or person_key not in member_keys:
                continue

            feature = track_info.get('feature')
            if feature is None:
                continue

            feature_row = np.asarray(feature, dtype=np.float32).reshape(1, -1)
            if person_key not in current_features:
                current_features[person_key] = feature_row
            else:
                current_features[person_key] = np.vstack((current_features[person_key], feature_row))

        merged_features = {}
        for member_id in lock_state.get('member_ids', []):
            person_key = self._person_key(member_id)
            if person_key is None:
                continue
            if person_key in current_features:
                merged_features[person_key] = current_features[person_key]
            elif person_key in existing_features and existing_features[person_key] is not None:
                merged_features[person_key] = existing_features[person_key].copy()

        lock_state['member_features'] = merged_features

    def record_empty_frame(self, state, frame_seq, reidentifier):
        if not self.enabled or frame_seq is None:
            return
        state['person_count_history'].append(0)
        self._update_track_birth_state(state['track_birth_state'], [], [], frame_seq, reidentifier)

    def record_tracks(self, state, current_track_ids, detection_confs, frame_seq, reidentifier):
        if not self.enabled or frame_seq is None:
            return

        state['person_count_history'].append(len(current_track_ids))
        high_conf_new_track = self._update_track_birth_state(
            state['track_birth_state'],
            current_track_ids,
            detection_confs,
            frame_seq,
            reidentifier,
        )
        if high_conf_new_track:
            state['reid_time_lock']['last_high_conf_new_track_frame'] = frame_seq
            self._clear_lock_state(state['reid_time_lock'])

        if self.is_active(state):
            self.refresh_member_features(reidentifier, state)

    def update_post_frame(self, state, frame_seq, confirmed_person_ids, reidentifier):
        if not self.enabled or frame_seq is None:
            return

        self._update_lock_state(
            state['reid_time_lock'],
            state['person_count_history'],
            frame_seq,
            confirmed_person_ids,
        )
        if self.is_active(state):
            self.refresh_member_features(reidentifier, state)

    def _ensure_history(self, state):
        history = state['person_count_history']
        if history.maxlen != self.window_frames:
            state['person_count_history'] = deque(history, maxlen=self.window_frames)

    def _person_key(self, person_id):
        normalized = self._normalize_identity(person_id)
        if normalized in (-1, None, ''):
            return None
        return str(normalized)

    def _is_known_identity(self, person_id):
        normalized = self._normalize_identity(person_id)
        person_key = self._person_key(normalized)
        if person_key is None:
            return False
        if normalized == 'Unknown':
            return False
        return not self._is_temporary_identity(normalized)

    def _update_track_birth_state(self, track_birth_state, current_track_ids, detection_confs, frame_seq, reidentifier):
        active_track_set = set(current_track_ids or [])
        high_conf_new_track = False

        for track_id, det_conf in zip(current_track_ids or [], detection_confs or []):
            state = track_birth_state.get(track_id)
            if state is None:
                state = {
                    'first_frame': frame_seq,
                    'last_frame': frame_seq,
                    'consecutive_frames': 1,
                    'emitted_high_conf': False,
                }
            else:
                if state.get('last_frame') == frame_seq - 1:
                    state['consecutive_frames'] = int(state.get('consecutive_frames', 0)) + 1
                else:
                    state['consecutive_frames'] = 1
                state['last_frame'] = frame_seq

            state['last_conf'] = float(det_conf)
            if (
                not state.get('emitted_high_conf', False)
                and state['consecutive_frames'] >= reidentifier.MIN_TRACK_AGE
                and float(det_conf) >= 0.5
            ):
                state['emitted_high_conf'] = True
                high_conf_new_track = True

            track_birth_state[track_id] = state

        stale_track_ids = [
            track_id
            for track_id, state in list(track_birth_state.items())
            if track_id not in active_track_set
            and frame_seq - int(state.get('last_frame', frame_seq)) > max(reidentifier.max_age, self.window_frames)
        ]
        for track_id in stale_track_ids:
            track_birth_state.pop(track_id, None)

        return high_conf_new_track

    def _get_stable_person_count(self, person_count_history):
        if len(person_count_history) < self.window_frames:
            return None

        recent_counts = list(person_count_history)[-self.window_frames:]
        stable_count = recent_counts[0]
        if all(count == stable_count for count in recent_counts):
            return stable_count
        return None

    def _has_recent_high_conf_new_track(self, lock_state, frame_seq):
        last_frame = lock_state.get('last_high_conf_new_track_frame')
        if last_frame is None:
            return False
        return (frame_seq - int(last_frame)) < self.window_frames

    @staticmethod
    def _clear_lock_state(lock_state):
        lock_state['active'] = False
        lock_state['locked_k'] = 0
        lock_state['member_ids'] = []
        lock_state['member_features'] = {}
        lock_state['locked_at_frame'] = None

    def _activate_lock_state(self, lock_state, member_ids, frame_seq):
        normalized_member_ids = []
        member_keys = set()
        for member_id in member_ids:
            normalized_member_id = self._normalize_identity(member_id)
            member_key = self._person_key(normalized_member_id)
            if member_key is None or member_key in member_keys:
                continue
            member_keys.add(member_key)
            normalized_member_ids.append(normalized_member_id)

        previous_keys = {
            self._person_key(member_id)
            for member_id in lock_state.get('member_ids', [])
        }
        if previous_keys != member_keys:
            lock_state['member_features'] = {}

        if not lock_state.get('active') or previous_keys != member_keys:
            lock_state['locked_at_frame'] = frame_seq

        lock_state['active'] = True
        lock_state['locked_k'] = len(normalized_member_ids)
        lock_state['member_ids'] = normalized_member_ids

    def _update_lock_state(self, lock_state, person_count_history, frame_seq, confirmed_person_ids):
        stable_count = self._get_stable_person_count(person_count_history)
        recent_high_conf_track = self._has_recent_high_conf_new_track(lock_state, frame_seq)
        confirmed_member_ids = []
        confirmed_keys = set()
        for person_id in confirmed_person_ids:
            member_key = self._person_key(person_id)
            if member_key is None or member_key in confirmed_keys:
                continue
            confirmed_keys.add(member_key)
            confirmed_member_ids.append(self._normalize_identity(person_id))

        if recent_high_conf_track:
            self._clear_lock_state(lock_state)
            return

        if stable_count == 0:
            self._clear_lock_state(lock_state)
            return

        if bool(self.enabled and lock_state.get('active') and lock_state.get('member_ids')):
            locked_k = int(lock_state.get('locked_k') or len(lock_state.get('member_ids', [])))
            if stable_count is None or stable_count == locked_k:
                return
            if stable_count > 0 and len(confirmed_member_ids) == stable_count:
                self._activate_lock_state(lock_state, confirmed_member_ids, frame_seq)
            else:
                self._clear_lock_state(lock_state)
            return

        if stable_count is None or stable_count <= 0:
            return
        if len(confirmed_member_ids) == stable_count:
            self._activate_lock_state(lock_state, confirmed_member_ids, frame_seq)
