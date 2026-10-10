"""Causal temporal features from frozen embeddings, without encoder adaptation."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from killing.embedding.reference_classification import unit_vectors


def temporal_embeddings(
    rows: list[dict[str, Any]],
    embeddings: np.ndarray,
    *,
    mode: str,
    window: int = 3,
) -> np.ndarray:
    """Keep row order; use only the same ROI's current and earlier observations.

    trailing_mean averages the last `window` sampled observations, including the
    current one. baseline_delta concatenates current embedding and its difference
    from the earliest available embedding of that ROI. Neither reads labels.
    """
    vectors = unit_vectors(embeddings)
    if len(rows) != len(vectors):
        raise ValueError("Manifest and embedding row counts differ")
    if mode not in {"frame", "trailing_mean", "baseline_delta"} or window < 1:
        raise ValueError("Unknown temporal mode or nonpositive window")
    groups: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        groups[str(row["group"])].append(i)
    dimension = vectors.shape[1] * (2 if mode == "baseline_delta" else 1)
    result = np.empty((len(rows), dimension), dtype=np.float64)
    for indices in groups.values():
        if len({rows[i]["split"] for i in indices}) != 1:
            raise ValueError("An ROI group occurs in both reference and evaluation")
        if (
            len({(rows[i].get("channel", 0), rows[i].get("z", 0)) for i in indices})
            != 1
        ):
            raise ValueError("Temporal context cannot mix channels or Z planes")
        order = sorted(indices, key=lambda i: rows[i]["frame"])
        frames = [rows[i]["frame"] for i in order]
        if len(frames) != len(set(frames)):
            raise ValueError("Duplicate frame within an ROI group")
        for position, i in enumerate(order):
            if mode == "frame":
                result[i] = vectors[i]
            elif mode == "trailing_mean":
                history = order[max(0, position - window + 1) : position + 1]
                result[i] = vectors[history].mean(axis=0)
            else:
                result[i] = np.concatenate((vectors[i], vectors[i] - vectors[order[0]]))
    return unit_vectors(result)
