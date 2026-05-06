import numpy as np

from models.ascend_backend import AscendInferSession, to_numpy


class AscendActionModel:
    """Torch-like callable wrapper for the fixed-shape UniFormer OM model."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.session = AscendInferSession(model_path)

    def eval(self):
        return self

    def to(self, device):
        return self

    def __call__(self, input_tensor):
        input_np = to_numpy(input_tensor).astype(np.float32, copy=False)
        outputs = self.session.infer(input_np)
        logits = outputs[0].astype(np.float32, copy=False)
        try:
            import torch

            return torch.from_numpy(logits)
        except ImportError:
            return logits

