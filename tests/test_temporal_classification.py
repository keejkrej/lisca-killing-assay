from typing import Any

import numpy as np
import pytest

from apoptosis.embedding.temporal_classification import temporal_embeddings


def _rows() -> list[dict[str, Any]]:
    return [
        {"group": "a", "split": "reference", "frame": 20},
        {"group": "b", "split": "evaluation", "frame": 0},
        {"group": "a", "split": "reference", "frame": 0},
        {"group": "a", "split": "reference", "frame": 10},
    ]


@pytest.mark.parametrize("mode", ["trailing_mean", "baseline_delta"])
def test_temporal_features_are_causal_and_do_not_mix_rois(mode: str) -> None:
    rows = _rows()
    vectors = np.array([[0, 1.0], [-1, 0], [1, 0], [1, 1]])
    before = temporal_embeddings(rows, vectors, mode=mode)
    changed = vectors.copy()
    changed[0] = [-1, 1]  # Changing the future must not affect earlier frames.
    changed[1] = [0, -1]  # Or a different ROI's features.
    after = temporal_embeddings(rows, changed, mode=mode)
    np.testing.assert_allclose(before[[2, 3]], after[[2, 3]])
    assert not np.allclose(before[0], after[0])
    np.testing.assert_allclose(np.linalg.norm(before, axis=1), 1)


def test_trailing_window_uses_frame_order_but_preserves_output_order() -> None:
    vectors = np.array([[0, 1.0], [-1, 0], [1, 0], [1, 1]])
    result = temporal_embeddings(_rows(), vectors, mode="trailing_mean", window=2)
    expected = np.array([1 / np.sqrt(2), 1 + 1 / np.sqrt(2)])
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(result[0], expected)
    np.testing.assert_allclose(result[2], [1, 0])


def test_baseline_delta_starts_with_zero_change_and_ignores_labels() -> None:
    rows = _rows()
    rows[2]["label"] = "dead"  # Baseline is not assumed to be viable.
    vectors = np.array([[0, 1.0], [-1, 0], [1, 0], [1, 1]])
    result = temporal_embeddings(rows, vectors, mode="baseline_delta")
    np.testing.assert_allclose(result[2], [1, 0, 0, 0])
    np.testing.assert_allclose(result[0], np.array([0, 1, -1, 1]) / np.sqrt(3))


@pytest.mark.parametrize("mutation", ["split", "frame", "channel"])
def test_invalid_temporal_groups_are_rejected(mutation: str) -> None:
    rows = _rows()
    rows[0][mutation] = {"split": "evaluation", "frame": 0, "channel": 1}[mutation]
    with pytest.raises(ValueError):
        temporal_embeddings(rows, np.ones((4, 2)), mode="trailing_mean")
