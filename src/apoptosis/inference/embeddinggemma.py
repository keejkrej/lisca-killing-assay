"""Frozen EmbeddingGemma 2 encoder for label-free viability."""

from __future__ import annotations

import numpy as np
from PIL import Image

MODEL_ID = "google/embeddinggemma-2"
REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"
DIMENSIONS = 768


class EmbeddingGemmaEncoder:
    """Assay-specific weights. The host supplies batching and does not train this."""

    model_id = MODEL_ID
    revision = REVISION
    dimensions = DIMENSIONS

    def __init__(self, device: str = "cuda") -> None:
        import torch
        from sentence_transformers import (  # ty: ignore[unresolved-import]
            SentenceTransformer,
        )

        self.model = SentenceTransformer(
            MODEL_ID,
            revision=REVISION,
            device=device,
            model_kwargs={"torch_dtype": torch.float32},
        )
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.identity = f"{MODEL_ID}@{REVISION}:float32:rgb-v1"
        self.device = device

    def encode(self, images: list[Image.Image]) -> np.ndarray:
        import torch

        with torch.inference_mode():
            return self.model.encode(
                [{"image": image} for image in images],
                batch_size=len(images),
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
