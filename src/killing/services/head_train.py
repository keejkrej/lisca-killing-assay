from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from killing.ml.encoder import ImageEncoder, load_encoder
from killing.ml.frame_dataset import ManifestSample, load_manifest_samples
from killing.ml.head import LinearHead, fit_linear_head, save_head
from killing.services.embed import (
    embed_samples,
    load_aligned_embeddings,
    save_embeddings,
)


def default_head_dir(project_root: Path) -> Path:
    return project_root / "runs" / "viability"


def default_head_path(project_root: Path) -> Path:
    return default_head_dir(project_root) / "head.json"


def default_embeddings_path(project_root: Path) -> Path:
    return default_head_dir(project_root) / "embeddings.npz"


def _resolve_embeddings(
    samples: list[ManifestSample],
    data_dir: Path,
    *,
    encoder: ImageEncoder | None,
    embeddings_path: Path | None,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, ImageEncoder | None]:
    if embeddings_path is not None and embeddings_path.is_file():
        return load_aligned_embeddings(embeddings_path, samples), encoder
    if encoder is None:
        encoder = load_encoder(device=device)
    return embed_samples(encoder, data_dir, samples, batch_size=batch_size), encoder


def train_linear_head(
    manifest_path: Path,
    output_dir: Path,
    *,
    encoder: ImageEncoder | None = None,
    embeddings_path: Path | None = None,
    svm_c: float = 1.0,
    batch_size: int = 8,
    device: str = "cuda",
) -> LinearHead:
    payload = json.loads(manifest_path.read_text())
    data_dir = Path(payload["data_dir"])
    samples = load_manifest_samples(manifest_path)
    if not samples:
        raise ValueError(f"No samples in {manifest_path}")
    vectors, encoder = _resolve_embeddings(
        samples,
        data_dir,
        encoder=encoder,
        embeddings_path=embeddings_path,
        batch_size=batch_size,
        device=device,
    )
    train = [i for i, sample in enumerate(samples) if sample.split == "train"]
    if not train:
        raise ValueError("Manifest has no train split")
    if encoder is None:
        from killing.ml.encoder import encoder_identity

        model_id, revision, _dimensions = encoder_identity()
    else:
        model_id, revision = encoder.model_id, encoder.revision
    head = fit_linear_head(
        vectors[train],
        [samples[i].label for i in train],
        encoder_model_id=model_id,
        encoder_revision=revision,
        svm_c=svm_c,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    save_head(head, output_dir / "head.json")
    save_embeddings(output_dir / "embeddings.npz", samples, vectors)
    return head
