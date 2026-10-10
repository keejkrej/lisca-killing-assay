from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import tifffile

from killing.embedding.experiment import (
    display_plane,
    embed_sample,
    import_frame_manifest,
    review_sequences,
    sample_rois,
)
from killing.embedding.reference_classification import (
    compare_classifiers,
    evaluate_manifest,
    unit_vectors,
)


def test_knn_and_svm_classify_separate_queries_and_return_evidence() -> None:
    pytest.importorskip("sklearn")
    result = compare_classifiers(
        np.array([[1, 0], [1, 0.1], [-1, 0], [-1, -0.1]]),
        ["present", "present", "empty", "empty"],
        np.array([[4, 0.1], [-8, 0.1]]),
        neighbors=3,
    )
    assert result["knn"]["predictions"] == ["present", "empty"]
    assert result["svm"]["predictions"] == ["present", "empty"]
    assert result["knn"]["neighbors"][0][0]["reference_index"] == 0
    assert 0 <= result["knn"]["vote_support"][0] <= 1


@pytest.mark.parametrize(
    "vectors", [np.zeros((2, 3)), np.array([[np.nan]]), np.array([1, 2])]
)
def test_invalid_vectors_are_rejected(vectors: np.ndarray) -> None:
    with pytest.raises(ValueError):
        unit_vectors(vectors)


def _rows() -> list[dict[str, Any]]:
    return [
        {
            "id": "a",
            "group": "roi1",
            "split": "reference",
            "label": "present",
            "label_source": "visual-provisional",
        },
        {
            "id": "b",
            "group": "roi2",
            "split": "reference",
            "label": "empty",
            "label_source": "visual-provisional",
        },
        {
            "id": "c",
            "group": "roi3",
            "split": "evaluation",
            "label": "present",
            "label_source": "visual-provisional",
        },
        {"id": "d", "group": "roi4", "split": "evaluation", "label": "uncertain"},
    ]


def test_evaluation_excludes_uncertain_labels_but_predicts_them() -> None:
    pytest.importorskip("sklearn")
    result = evaluate_manifest(_rows(), np.array([[1, 0], [-1, 0], [1, 0.1], [0, 1]]))
    assert result["scored_ids"] == ["c"]
    assert result["query_ids"] == ["c", "d"]
    assert result["metrics"]["svm"]["accuracy"] == 1
    assert result["results"]["knn"]["neighbors"][0][0]["id"] == "a"
    assert result["label_sources"] == ["visual-provisional"]


def test_frames_of_one_roi_cannot_leak_between_splits() -> None:
    pytest.importorskip("sklearn")
    rows = _rows()
    rows[2]["group"] = "roi1"
    with pytest.raises(ValueError, match="both reference and evaluation"):
        evaluate_manifest(rows, np.ones((4, 3)))


def test_unlabeled_evaluation_has_no_accuracy() -> None:
    pytest.importorskip("sklearn")
    rows = _rows()
    rows[2]["label"] = None
    result = evaluate_manifest(rows, np.array([[1, 0], [-1, 0], [1, 0.1], [0, 1]]))
    assert result["metrics"] == {"knn": None, "svm": None}


def test_sampling_uses_channel_metadata_and_holds_out_whole_rois(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    position = workspace / "roi" / "Pos0"
    position.mkdir(parents=True)
    rois = []
    ramp = np.arange(16, dtype=np.uint16).reshape(4, 4)
    for roi in range(4):
        file_name = f"Roi{roi}.tif"
        with tifffile.TiffWriter(position / file_name) as writer:
            for _ in range(3):
                writer.write(ramp)
                writer.write(np.full((4, 4), 500, dtype=np.uint16))
        rois.append(
            {
                "fileName": file_name,
                "roi": roi,
                "bbox": {"roi": roi, "x": 0, "y": 0, "w": 4, "h": 4},
            }
        )
    (position / "index.json").write_text(
        json.dumps(
            {
                "axisOrder": "TCZYX",
                "position": 0,
                "timeCount": 3,
                "channelCount": 2,
                "zCount": 1,
                "timeIndices": [0, 5, 10],
                "rois": rois,
            }
        )
    )
    output = tmp_path / "sample"
    manifest = sample_rois(workspace, output, roi_count=4, frames_per_roi=2)
    assert len(manifest["rows"]) == 8
    assert all(row["frame"] == row["stack_frame"] * 5 for row in manifest["rows"])
    ref = {r["group"] for r in manifest["rows"] if r["split"] == "reference"}
    held_out = {r["group"] for r in manifest["rows"] if r["split"] == "evaluation"}
    assert ref and held_out and not ref & held_out
    from PIL import Image

    with Image.open(output / manifest["rows"][0]["image"]) as image:
        assert np.asarray(image).max() == 255  # Selected BF, not constant reporter.
    assert (output / "contact-01.jpg").is_file()
    with pytest.raises(ValueError, match="already exists"):
        sample_rois(workspace, output)
    again = sample_rois(workspace, tmp_path / "again", roi_count=4, frames_per_roi=2)
    assert manifest == again
    sequence = review_sequences(output, frames=2)
    assert len(sequence["groups"]) == 4
    assert (output / "sequence-01.jpg").is_file()
    source = tmp_path / "labeled.json"
    samples = [
        {
            "position": "Pos0",
            "roi_id": roi,
            "time_index": frame,
            "label": int(frame == 0),
            "split": "train" if roi < 2 else "val",
        }
        for roi in range(4)
        for frame in range(3)
    ]
    source.write_text(json.dumps({"data_dir": str(workspace), "samples": samples}))
    imported = import_frame_manifest(
        source,
        tmp_path / "imported",
        frame_stride=2,
        label_map={"0": "dead", "1": "viable"},
    )
    assert len(imported["rows"]) == 8
    assert imported["rows"][0]["label"] == "viable"
    assert imported["rows"][1]["label"] == "dead"
    assert imported["rows"][1]["frame"] == 10
    assert imported["rows"][4]["split"] == "evaluation"
    samples.append({**samples[0], "split": "val"})
    source.write_text(json.dumps({"data_dir": str(workspace), "samples": samples}))
    with pytest.raises(ValueError, match="both splits"):
        import_frame_manifest(source, tmp_path / "leaking")


def test_constant_image_normalization_is_finite() -> None:
    assert not np.asarray(display_plane(np.ones((4, 4)))).any()


def test_fixed_contrast_preserves_darkening() -> None:
    bright = np.arange(16, dtype=float).reshape(4, 4)
    dark = bright / 4
    assert np.asarray(display_plane(dark, (0, 15))).max() < 70
    assert np.asarray(display_plane(bright, (0, 15))).max() == 255


def test_label_changes_reuse_cache_and_encoder_is_frozen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys
    from contextlib import nullcontext
    from types import SimpleNamespace

    from PIL import Image

    Image.new("RGB", (8, 8), "white").save(tmp_path / "a.png")
    manifest = {
        "preprocessing": "test",
        "rows": [{"id": "a", "image": "a.png", "label": "x"}],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    calls = []

    class Parameter:
        def requires_grad_(self, value: bool) -> None:
            assert value is False

    class Encoder:
        def __init__(self, *args, **kwargs):
            calls.append("load")

        def eval(self):
            calls.append("eval")

        def parameters(self):
            return [Parameter()]

        def encode(self, batch, **kwargs):
            assert set(batch[0]) == {"image"}
            calls.append("encode")
            return np.array([[3.0, 4.0]])

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(float32="float32", inference_mode=nullcontext),
    )
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=Encoder),
    )
    monkeypatch.setitem(
        sys.modules, "huggingface_hub", SimpleNamespace(HfApi=lambda: None)
    )
    result = embed_sample(tmp_path, revision="a" * 40, device="cpu")
    assert not result["cache_reused"]
    assert calls == ["load", "eval", "encode"]
    np.testing.assert_allclose(np.load(tmp_path / "embeddings.npy"), [[0.6, 0.8]])
    manifest["rows"][0]["label"] = "corrected"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    assert embed_sample(tmp_path)["cache_reused"]
    assert calls == ["load", "eval", "encode"]
    Image.new("RGB", (8, 8), "black").save(tmp_path / "a.png")
    assert not embed_sample(tmp_path, device="cpu")["cache_reused"]
    assert calls.count("encode") == 2
