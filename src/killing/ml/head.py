"""Linear SVM head on frozen embeddings. The encoder is never trained."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from killing.embedding.reference_classification import unit_vectors

CLASS_DEAD = "dead"
CLASS_VIABLE = "viable"
INT_TO_CLASS = {0: CLASS_DEAD, 1: CLASS_VIABLE}


@dataclass(frozen=True)
class LinearHead:
    kind: str
    svm_c: float
    classes: tuple[str, ...]
    coef: np.ndarray
    intercept: np.ndarray
    encoder_model_id: str
    encoder_revision: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "svm_c": self.svm_c,
            "classes": list(self.classes),
            "coef": self.coef.tolist(),
            "intercept": self.intercept.tolist(),
            "encoder": {
                "model_id": self.encoder_model_id,
                "revision": self.encoder_revision,
            },
        }


def class_name(label: int | str) -> str:
    if isinstance(label, str):
        name = label.strip()
        if name in {CLASS_DEAD, CLASS_VIABLE}:
            return name
        msg = f"Unsupported class name {label!r}"
        raise ValueError(msg)
    if label in INT_TO_CLASS:
        return INT_TO_CLASS[label]
    msg = f"Unsupported integer label {label!r}"
    raise ValueError(msg)


def fit_linear_head(
    embeddings: np.ndarray,
    labels: list[int | str],
    *,
    encoder_model_id: str,
    encoder_revision: str,
    svm_c: float = 1.0,
) -> LinearHead:
    from sklearn.svm import LinearSVC

    names = [class_name(label) for label in labels]
    if len(set(names)) < 2:
        raise ValueError("Training the head needs both viable and dead examples")
    if svm_c <= 0 or not np.isfinite(svm_c):
        raise ValueError("svm_c must be positive")
    vectors = unit_vectors(embeddings)
    svm = LinearSVC(C=svm_c, class_weight="balanced", random_state=0, max_iter=10000)
    svm.fit(vectors, names)
    return LinearHead(
        kind="linear-svc",
        svm_c=float(svm_c),
        classes=tuple(str(item) for item in svm.classes_.tolist()),
        coef=np.asarray(svm.coef_, dtype=np.float64),
        intercept=np.asarray(svm.intercept_, dtype=np.float64),
        encoder_model_id=encoder_model_id,
        encoder_revision=encoder_revision,
    )


def decision_scores(head: LinearHead, embeddings: np.ndarray) -> np.ndarray:
    vectors = unit_vectors(embeddings)
    coef = np.asarray(head.coef, dtype=np.float64)
    intercept = np.asarray(head.intercept, dtype=np.float64).reshape(-1)
    if coef.ndim != 2 or coef.shape[0] != 1 or intercept.shape != (1,):
        raise ValueError("Binary linear head must have shape (1, dim) coefficients")
    if coef.shape[1] != vectors.shape[1]:
        raise ValueError("Head coefficient dimension does not match embeddings")
    return vectors @ coef.reshape(-1) + float(intercept[0])


def predict_classes(head: LinearHead, embeddings: np.ndarray) -> list[str]:
    scores = decision_scores(head, embeddings)
    positive, negative = head.classes[1], head.classes[0]
    return [positive if score > 0 else negative for score in scores]


def dead_probability(head: LinearHead, embeddings: np.ndarray) -> np.ndarray:
    """Logistic of the dead side of the decision score. Not a calibrated probability."""
    scores = decision_scores(head, embeddings)
    if CLASS_DEAD not in head.classes:
        raise ValueError("Head must include a dead class")
    prob_positive = 1.0 / (1.0 + np.exp(-np.clip(scores, -50, 50)))
    if head.classes[1] == CLASS_DEAD:
        return prob_positive.astype(np.float32)
    return (1.0 - prob_positive).astype(np.float32)


def save_head(head: LinearHead, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(head.to_payload(), indent=2) + "\n")


def load_head(path: Path) -> LinearHead:
    payload = json.loads(path.read_text())
    encoder = payload.get("encoder") or {}
    classes = payload.get("classes")
    if payload.get("kind") != "linear-svc":
        raise ValueError("Unsupported head kind")
    if not isinstance(classes, list) or len(classes) != 2:
        raise ValueError("Head must declare two classes")
    return LinearHead(
        kind="linear-svc",
        svm_c=float(payload["svm_c"]),
        classes=(str(classes[0]), str(classes[1])),
        coef=np.asarray(payload["coef"], dtype=np.float64),
        intercept=np.asarray(payload["intercept"], dtype=np.float64),
        encoder_model_id=str(encoder["model_id"]),
        encoder_revision=str(encoder["revision"]),
    )
