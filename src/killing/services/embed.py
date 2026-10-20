from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from killing.ml.encoder import ImageEncoder
from killing.ml.frame_dataset import ManifestSample, load_brightfield_image


def embed_images(
    encoder: ImageEncoder,
    images: list[Image.Image],
    *,
    batch_size: int = 8,
) -> np.ndarray:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not images:
        raise ValueError("Need images to embed")
    batches = [
        np.asarray(encoder.encode(images[start : start + batch_size]), dtype=np.float32)
        for start in range(0, len(images), batch_size)
    ]
    return np.concatenate(batches, axis=0)


def embed_samples(
    encoder: ImageEncoder,
    data_dir: Path,
    samples: list[ManifestSample],
    *,
    batch_size: int = 8,
) -> np.ndarray:
    images = [load_brightfield_image(data_dir, sample) for sample in samples]
    return embed_images(encoder, images, batch_size=batch_size)


def save_embeddings(
    path: Path,
    samples: list[ManifestSample],
    vectors: np.ndarray,
) -> None:
    if len(samples) != len(vectors):
        raise ValueError("Sample and embedding counts differ")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        ids=np.array([sample.sample_id for sample in samples], dtype=object),
        vectors=np.asarray(vectors, dtype=np.float32),
    )


def load_aligned_embeddings(
    path: Path,
    samples: list[ManifestSample],
) -> np.ndarray:
    payload = np.load(path, allow_pickle=True)
    ids = [str(item) for item in payload["ids"].tolist()]
    vectors = np.asarray(payload["vectors"], dtype=np.float32)
    if len(ids) != len(vectors):
        raise ValueError("Embedding cache ids and vectors differ in length")
    index = {item: i for i, item in enumerate(ids)}
    missing = [sample.sample_id for sample in samples if sample.sample_id not in index]
    if missing:
        raise ValueError(f"Embedding cache is missing {missing[0]}")
    return np.stack([vectors[index[sample.sample_id]] for sample in samples])
