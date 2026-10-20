"""Image encoder used by the label-free head. Weights stay frozen."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from PIL import Image

from killing.inference.embeddinggemma import DIMENSIONS, MODEL_ID, REVISION
from killing.ml.head import LinearHead


class ImageEncoder(Protocol):
    model_id: str
    revision: str
    dimensions: int

    def encode(self, images: list[Image.Image]) -> np.ndarray: ...


def load_encoder(device: str = "cuda") -> ImageEncoder:
    try:
        from killing.inference.embeddinggemma import EmbeddingGemmaEncoder

        return EmbeddingGemmaEncoder(device=device)
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "EmbeddingGemma requires the encoder extra. "
            "Install with: uv sync --extra encoder"
        ) from exc


def require_matching_encoder(head: LinearHead, encoder: ImageEncoder) -> None:
    if (
        encoder.model_id != head.encoder_model_id
        or encoder.revision != head.encoder_revision
    ):
        raise ValueError(
            "Encoder revision does not match the trained head: "
            f"{encoder.model_id}@{encoder.revision} vs "
            f"{head.encoder_model_id}@{head.encoder_revision}"
        )


def encoder_identity() -> tuple[str, str, int]:
    return MODEL_ID, REVISION, DIMENSIONS
