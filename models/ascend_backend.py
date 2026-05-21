import os
import threading
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np


ASCEND_DEVICE_ENV = "ASCEND_DEVICE_ID"


def requested_backend() -> str:
    """Return the active model backend."""
    return "om"


def is_om_path(path) -> bool:
    return str(path).lower().endswith(".om")


def resolve_model_path(om_path: str, legacy_path: str = None) -> str:
    """Require OM models. The legacy_path argument is kept for old call sites."""
    if Path(om_path).exists():
        return om_path
    raise FileNotFoundError(f"OM model was not found: {om_path}")


def get_ascend_device_id() -> int:
    value = os.getenv(ASCEND_DEVICE_ENV, "0").strip()
    try:
        return int(value)
    except ValueError:
        return 0


def _as_numpy_array(output):
    if isinstance(output, np.ndarray):
        return output
    if hasattr(output, "to_host"):
        output = output.to_host()
    if hasattr(output, "numpy"):
        return output.numpy()
    return np.asarray(output)


class AscendInferSession:
    """Small wrapper around ais-bench InferSession for fixed-shape OM inference."""

    def __init__(self, model_path: str, device_id: int = None):
        self.model_path = model_path
        self.device_id = get_ascend_device_id() if device_id is None else int(device_id)

        try:
            from ais_bench.infer.interface import InferSession
        except ImportError as exc:
            raise RuntimeError(
                "OM inference requires ais_bench on the Huawei NPU server. "
                "Install/enable the CANN ais-bench Python package before running OM models."
            ) from exc

        self.session = InferSession(self.device_id, model_path)
        self._lock = threading.Lock()

    def infer(self, inputs) -> List[np.ndarray]:
        if isinstance(inputs, np.ndarray):
            feed: Sequence[np.ndarray] = [inputs]
        elif isinstance(inputs, (list, tuple)):
            feed = inputs
        else:
            feed = [np.asarray(inputs)]

        feed = [
            np.ascontiguousarray(inp.astype(np.float32, copy=False))
            if isinstance(inp, np.ndarray) and inp.dtype != np.float32
            else np.ascontiguousarray(inp)
            for inp in feed
        ]
        with self._lock:
            outputs = self.session.infer(feed)
        if not isinstance(outputs, (list, tuple)):
            outputs = [outputs]
        return [_as_numpy_array(output) for output in outputs]


def to_numpy(input_data) -> np.ndarray:
    if isinstance(input_data, np.ndarray):
        return input_data
    if hasattr(input_data, "detach"):
        input_data = input_data.detach()
    if hasattr(input_data, "cpu"):
        input_data = input_data.cpu()
    if hasattr(input_data, "numpy"):
        return input_data.numpy()
    return np.asarray(input_data)


def l2_normalize(features: np.ndarray, axis: int = 1, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(features, axis=axis, keepdims=True)
    return features / np.maximum(norm, eps)
