from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from killing.ml.encoder import ImageEncoder, load_encoder, require_matching_encoder
from killing.ml.frame_dataset import ManifestSample, load_manifest_samples
from killing.ml.head import load_head, predict_classes
from killing.services.embed import embed_samples, load_aligned_embeddings


@dataclass(frozen=True)
class SplitMetrics:
    split: str
    samples: int
    accuracy: float
    f1: float
    precision: float
    recall: float
    true_viable: int
    true_dead: int
    pred_viable: int
    pred_dead: int
    confusion: list[list[int]]


def _binary_metrics(
    preds: list[int],
    labels: list[int],
) -> tuple[float, float, float, float, list[list[int]]]:
    pairs = list(zip(preds, labels, strict=True))
    tp = sum(1 for pred, label in pairs if pred == 1 and label == 1)
    tn = sum(1 for pred, label in pairs if pred == 0 and label == 0)
    fp = sum(1 for pred, label in pairs if pred == 1 and label == 0)
    fn = sum(1 for pred, label in pairs if pred == 0 and label == 1)
    total = len(labels)
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    denom = precision + recall
    f1 = (2 * precision * recall / denom) if denom else 0.0
    return accuracy, f1, precision, recall, [[tn, fp], [fn, tp]]


def _as_binary(name: str) -> int:
    if name == "viable":
        return 1
    if name == "dead":
        return 0
    msg = f"Unsupported class {name!r}"
    raise ValueError(msg)


def _embeddings_for_samples(
    samples: list[ManifestSample],
    data_dir: Path,
    encoder: ImageEncoder | None,
    embeddings_path: Path | None,
    batch_size: int,
    device: str,
) -> np.ndarray:
    if embeddings_path is not None:
        return load_aligned_embeddings(embeddings_path, samples)
    if encoder is None:
        encoder = load_encoder(device=device)
    return embed_samples(encoder, data_dir, samples, batch_size=batch_size)


def evaluate_head(
    head_path: Path,
    manifest_path: Path,
    *,
    encoder: ImageEncoder | None = None,
    embeddings_path: Path | None = None,
    batch_size: int = 8,
    device: str = "cuda",
) -> list[SplitMetrics]:
    head = load_head(head_path)
    if encoder is not None:
        require_matching_encoder(head, encoder)
    data_dir = Path(json.loads(manifest_path.read_text())["data_dir"])
    results: list[SplitMetrics] = []
    for split in ("train", "val"):
        samples = load_manifest_samples(manifest_path, split)
        vectors = _embeddings_for_samples(
            samples,
            data_dir,
            encoder,
            embeddings_path,
            batch_size,
            device,
        )
        preds = [_as_binary(name) for name in predict_classes(head, vectors)]
        labels = [int(sample.label) for sample in samples]
        accuracy, f1, precision, recall, confusion = _binary_metrics(preds, labels)
        true_dead = sum(1 for label in labels if label == 0)
        true_viable = len(labels) - true_dead
        pred_dead = sum(1 for pred in preds if pred == 0)
        results.append(
            SplitMetrics(
                split=split,
                samples=len(labels),
                accuracy=accuracy,
                f1=f1,
                precision=precision,
                recall=recall,
                true_viable=true_viable,
                true_dead=true_dead,
                pred_viable=len(preds) - pred_dead,
                pred_dead=pred_dead,
                confusion=confusion,
            )
        )
    return results
