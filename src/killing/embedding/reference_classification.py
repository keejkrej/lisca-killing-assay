"""Small classifiers on frozen image embeddings; no encoder training."""

from __future__ import annotations

from typing import Any

import numpy as np


def unit_vectors(values: np.ndarray) -> np.ndarray:
    vectors = np.asarray(values, dtype=np.float64)
    if vectors.ndim != 2 or not all(vectors.shape):
        raise ValueError("Embeddings must be a nonempty rows-by-features matrix")
    if not np.isfinite(vectors).all():
        raise ValueError("Embeddings must be finite")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if (norms == 0).any():
        raise ValueError("Zero embeddings cannot be compared by cosine similarity")
    return vectors / norms


def compare_classifiers(
    reference: np.ndarray,
    labels: list[str],
    query: np.ndarray,
    *,
    neighbors: int = 3,
    svm_c: float = 1.0,
) -> dict[str, Any]:
    """Return predictions and supporting neighbors; scores are not probabilities."""
    from sklearn.svm import LinearSVC

    reference = unit_vectors(reference)
    query = unit_vectors(query)
    if reference.shape[1] != query.shape[1]:
        raise ValueError("Reference and query embedding dimensions differ")
    if len(labels) != len(reference) or any(not label.strip() for label in labels):
        raise ValueError("Every reference needs a nonempty label")
    if len(set(labels)) < 2:
        raise ValueError("At least two reference classes are required")
    if neighbors < 1 or svm_c <= 0 or not np.isfinite(svm_c):
        raise ValueError("neighbors and svm_c must be positive")

    classes = np.unique(labels)
    label_array = np.asarray(labels)
    k = min(neighbors, len(reference))
    knn_predictions, supports, nearest = [], [], []
    # Bound the similarity matrix independently of the query collection's size.
    for start in range(0, len(query), 128):
        similarities = query[start : start + 128] @ reference.T
        for row in similarities:
            indices = np.argsort(-row, kind="stable")[:k]
            # Inverse cosine distance; exact matches dominate without division by zero.
            weights = 1.0 / np.maximum(1.0 - row[indices], 1e-6)
            votes = np.array(
                [weights[label_array[indices] == label].sum() for label in classes]
            )
            winner = int(votes.argmax())
            knn_predictions.append(str(classes[winner]))
            supports.append(float(votes[winner] / votes.sum()))
            nearest.append(
                [
                    {"reference_index": int(i), "cosine_similarity": float(row[i])}
                    for i in indices
                ]
            )

    svm = LinearSVC(C=svm_c, class_weight="balanced", random_state=0, max_iter=10000)
    svm.fit(reference, labels)
    return {
        "knn": {
            "predictions": knn_predictions,
            "vote_support": supports,
            "neighbors": nearest,
        },
        "svm": {
            "predictions": svm.predict(query).tolist(),
            "decision_scores": svm.decision_function(query).tolist(),
            "classes": svm.classes_.tolist(),
        },
    }


def evaluate_manifest(
    rows: list[dict[str, Any]],
    embeddings: np.ndarray,
    *,
    neighbors: int = 3,
    svm_c: float = 1.0,
) -> dict[str, Any]:
    """Evaluate explicit reference/evaluation groups; unlabelled rows are predicted."""
    from sklearn.metrics import accuracy_score, confusion_matrix, recall_score

    if len(rows) != len(embeddings):
        raise ValueError("Manifest and embedding row counts differ")
    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Image IDs must be unique")
    if any(row["split"] not in {"reference", "evaluation"} for row in rows):
        raise ValueError("Every row must have a reference or evaluation split")
    reference_groups = {row["group"] for row in rows if row["split"] == "reference"}
    evaluation_groups = {row["group"] for row in rows if row["split"] == "evaluation"}
    if reference_groups & evaluation_groups:
        raise ValueError("An ROI group occurs in both reference and evaluation splits")

    def usable(row: dict[str, Any]) -> bool:
        return bool(row.get("label")) and bool(row["label"] != "uncertain")

    ref = [
        i for i, row in enumerate(rows) if row["split"] == "reference" and usable(row)
    ]
    query = [i for i, row in enumerate(rows) if row["split"] == "evaluation"]
    if not ref or not query:
        raise ValueError("Need labeled references and separate evaluation images")
    result = compare_classifiers(
        embeddings[ref],
        [rows[i]["label"] for i in ref],
        embeddings[query],
        neighbors=neighbors,
        svm_c=svm_c,
    )
    scored = [j for j, i in enumerate(query) if usable(rows[i])]
    truth = [rows[query[j]]["label"] for j in scored]
    classes = sorted(set(truth) | {rows[i]["label"] for i in ref})
    metrics = {}
    for method, output in result.items():
        predictions = [output["predictions"][j] for j in scored]
        metrics[method] = (
            None
            if not truth
            else {
                "accuracy": float(accuracy_score(truth, predictions)),
                "macro_recall": float(
                    recall_score(
                        truth,
                        predictions,
                        labels=sorted(set(truth)),
                        average="macro",
                        zero_division=0,
                    )
                ),
                "classes": classes,
                "confusion_matrix": confusion_matrix(
                    truth, predictions, labels=classes
                ).tolist(),
            }
        )
    for neighbors_row in result["knn"]["neighbors"]:
        for match in neighbors_row:
            match["id"] = rows[ref[match.pop("reference_index")]]["id"]
    return {
        "reference_ids": [rows[i]["id"] for i in ref],
        "query_ids": [rows[i]["id"] for i in query],
        "scored_ids": [rows[query[j]]["id"] for j in scored],
        "annotations": [
            {
                key: row.get(key)
                for key in ("id", "group", "split", "label", "label_source", "notes")
            }
            for row in rows
        ],
        "evaluation_counts": {
            "total_images": len(query),
            "scored_images": len(scored),
            "roi_groups": len(evaluation_groups),
            "class_support": {label: truth.count(label) for label in classes},
        },
        "label_sources": sorted(
            {row.get("label_source") or "unspecified" for row in rows if usable(row)}
        ),
        "metrics_note": (
            "Agreement with supplied labels, not validated biological accuracy. "
            "Scores are not probabilities."
        ),
        "metrics": metrics,
        "results": result,
    }
