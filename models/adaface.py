"""
AdaFace Model Wrapper for Face Recognition

This class wraps the AdaFace model to provide the same interface as ArcFace,
using the original project's face_alignment function for preprocessing.
"""
import cv2
import numpy as np
import torch
from logging import getLogger

from models.net import build_model
from utils.helpers import face_alignment
from models.ascend_backend import AscendInferSession, is_om_path

__all__ = ["AdaFace"]

logger = getLogger(__name__)


class AdaFace:
    """
    AdaFace Model for Face Recognition

    This class implements a face encoder using the AdaFace architecture,
    providing the same interface as ArcFace for drop-in replacement.
    """

    def __init__(self, model_path: str, architecture: str = 'ir_50') -> None:
        """
        Initializes the AdaFace face encoder model.

        Args:
            model_path (str): Path to AdaFace checkpoint file (.ckpt).
            architecture (str): Model architecture. Options: 'ir_18', 'ir_34', 'ir_50', 'ir_101'.
                               Defaults to 'ir_50'.

        Raises:
            RuntimeError: If model initialization fails.
        """
        self.model_path = model_path
        self.architecture = architecture
        self.input_size = (112, 112)
        self.embedding_size = 512
        self.use_om = is_om_path(model_path)

        # Determine device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        logger.info(f"Initializing AdaFace model from {self.model_path}")
        logger.info(f"Using device: {self.device}")

        try:
            if self.use_om:
                self.session = AscendInferSession(model_path)
                self.model = None
                logger.info(f"Successfully initialized AdaFace OM encoder from {self.model_path}")
                return

            # Build model architecture
            self.model = build_model(architecture)

            # Load pretrained weights
            # Note: weights_only=False is needed for PyTorch Lightning checkpoints
            statedict = torch.load(model_path, map_location=self.device, weights_only=False)

            # Handle different checkpoint formats
            if 'state_dict' in statedict:
                # PyTorch Lightning format
                model_statedict = {
                    key[6:]: val for key, val in statedict['state_dict'].items()
                    if key.startswith('model.')
                }
            else:
                # Direct state dict
                model_statedict = statedict

            self.model.load_state_dict(model_statedict)
            self.model.to(self.device)
            self.model.eval()

            logger.info(
                f"Successfully initialized AdaFace encoder from {self.model_path} "
                f"(architecture: {architecture}, embedding size: {self.embedding_size})"
            )

        except Exception as e:
            logger.error(f"Failed to load AdaFace model from '{self.model_path}'", exc_info=True)
            raise RuntimeError(f"Failed to initialize AdaFace model for '{self.model_path}'") from e

    def preprocess(self, face_image: np.ndarray) -> torch.Tensor:
        """
        Preprocess the face image for AdaFace model.

        AdaFace expects BGR input normalized with mean=0.5, std=0.5

        Args:
            face_image (np.ndarray): Input face image in BGR format, 112x112.

        Returns:
            torch.Tensor: Preprocessed image tensor ready for inference.
        """
        # Ensure correct size
        if face_image.shape[:2] != self.input_size:
            face_image = cv2.resize(face_image, self.input_size)

        # AdaFace preprocessing: BGR, normalize to [-1, 1]
        # Formula: (img / 255 - 0.5) / 0.5 = img / 127.5 - 1
        img = face_image.astype(np.float32)
        img = (img / 255.0 - 0.5) / 0.5  # Normalize to [-1, 1]

        # Convert HWC to CHW format
        img = img.transpose(2, 0, 1)

        # Add batch dimension and convert to tensor
        tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)

        return tensor

    def get_embedding(
        self,
        image: np.ndarray,
        landmarks: np.ndarray,
        normalized: bool = True
    ) -> np.ndarray:
        """
        Extract face embedding from an image using facial landmarks for alignment.

        This method maintains the same interface as ArcFace for drop-in replacement.

        Args:
            image (np.ndarray): Input image in BGR format.
            landmarks (np.ndarray): 5-point facial landmarks for alignment.
            normalized (bool): Whether to return L2-normalized embedding. Defaults to True.
                              Note: AdaFace model already outputs normalized embeddings.

        Returns:
            np.ndarray: Face embedding vector (512-dimensional).

        Raises:
            ValueError: If inputs are invalid.
        """
        if image is None or landmarks is None:
            raise ValueError("Image and landmarks must not be None")

        try:
            # Use original project's face alignment function
            aligned_face, _ = face_alignment(image, landmarks)

            # Preprocess for AdaFace
            face_tensor = self.preprocess(aligned_face)

            if self.use_om:
                input_np = face_tensor.numpy().astype(np.float32, copy=False)
                outputs = self.session.infer(input_np)
                embedding = np.asarray(outputs[0], dtype=np.float32).flatten()
            else:
                face_tensor = face_tensor.to(self.device)

                # Get embedding
                with torch.no_grad():
                    embedding, norm = self.model(face_tensor)

                # Convert to numpy
                embedding = embedding.cpu().numpy().flatten()

            # AdaFace already outputs normalized embeddings
            # But if explicitly requested, ensure normalization
            if normalized:
                norm_val = np.linalg.norm(embedding)
                if norm_val > 0:
                    embedding = embedding / norm_val

            return embedding

        except Exception as e:
            logger.error(f"Error extracting face embedding: {e}")
            raise

    def get_embedding_batch(
        self,
        images: list,
        landmarks_list: list,
        normalized: bool = True
    ) -> list:
        """
        Extract face embeddings from multiple images in batch.

        Args:
            images (list): List of input images in BGR format.
            landmarks_list (list): List of 5-point facial landmarks.
            normalized (bool): Whether to return L2-normalized embeddings.

        Returns:
            list: List of face embedding vectors.
        """
        if len(images) != len(landmarks_list):
            raise ValueError("Number of images must match number of landmarks")

        embeddings = []
        for image, landmarks in zip(images, landmarks_list):
            embedding = self.get_embedding(image, landmarks, normalized)
            embeddings.append(embedding)

        return embeddings
