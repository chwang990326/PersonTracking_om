import os
import threading
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np


MODEL_BACKEND_ENV = "MODEL_BACKEND"
ASCEND_DEVICE_ENV = "ASCEND_DEVICE_ID"


def requested_backend() -> str:
    """Return auto, om, or legacy."""
    return os.getenv(MODEL_BACKEND_ENV, "auto").strip().lower() or "auto"


def is_om_path(path) -> bool:
    return str(path).lower().endswith(".om")


def resolve_model_path(om_path: str, legacy_path: str = None) -> str:
    """Prefer OM models when present, while keeping old paths usable during migration."""
    backend = requested_backend()
    om_exists = Path(om_path).exists()

    if backend in {"om", "ascend", "npu"}:
        if not om_exists:
            raise FileNotFoundError(
                f"MODEL_BACKEND={backend} requires OM model, but it was not found: {om_path}"
            )
        return om_path

    if backend in {"legacy", "onnx", "torch", "pytorch", "cuda", "cpu"}:
        if legacy_path is None:
            raise FileNotFoundError(f"No legacy model path configured for {om_path}")
        return legacy_path

    if om_exists:
        return om_path
    if legacy_path is not None:
        return legacy_path
    return om_path


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
