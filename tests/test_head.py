from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import tifffile
from PIL import Image

from killing.core.labels import LabelStore
from killing.core.roi import RoiRef
from killing.inference.embeddinggemma import MODEL_ID, REVISION
from killing.ml.frame_dataset import load_manifest_samples
from killing.ml.head import (
    CLASS_DEAD,
    CLASS_VIABLE,
    dead_probability,
    fit_linear_head,
    load_head,
    predict_classes,
    save_head,
)
from killing.ml.preprocess import frame_to_image
from killing.services.dataset_build import build_dataset_manifest
from killing.services.evaluate import evaluate_head
from killing.services.head_train import train_linear_head
from killing.services.inference import infer_all_cells


class FakeEncoder:
    model_id = MODEL_ID
    revision = REVISION
    dimensions = 2

    def encode(self, images: list[Image.Image]) -> np.ndarray:
        rows = []
        for image in images:
            gray = np.asarray(image.convert("L"), dtype=np.float32)
            alive = 1.0 if float(gray.mean()) > 1.0 else 0.0
            rows.append([alive, 1.0 - alive])
        return np.asarray(rows, dtype=np.float32)


def test_frame_to_image_is_rgb_without_resizing() -> None:
    frame = np.arange(12, dtype=np.uint16).reshape(3, 4)
    image = frame_to_image(frame)
    assert image.mode == "RGB"
    assert image.size == (4, 3)


def test_linear_head_roundtrip_and_dead_scores(tmp_path: Path) -> None:
    embeddings = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
    head = fit_linear_head(
        embeddings,
        [1, 1, 0, 0],
        encoder_model_id=MODEL_ID,
        encoder_revision=REVISION,
    )
    assert set(head.classes) == {CLASS_DEAD, CLASS_VIABLE}
    assert predict_classes(head, embeddings) == [
        CLASS_VIABLE,
        CLASS_VIABLE,
        CLASS_DEAD,
        CLASS_DEAD,
    ]
    probs = dead_probability(head, embeddings)
    assert probs[0] < 0.5 < probs[2]
    path = tmp_path / "head.json"
    save_head(head, path)
    loaded = load_head(path)
    assert predict_classes(loaded, embeddings) == predict_classes(head, embeddings)


def test_fit_requires_both_classes() -> None:
    with pytest.raises(ValueError, match="viable and dead"):
        fit_linear_head(
            np.array([[1.0, 0.0], [0.9, 0.1]]),
            [1, 1],
            encoder_model_id=MODEL_ID,
            encoder_revision=REVISION,
        )


def _blob(value: int, size: int = 32) -> np.ndarray:
    yy, xx = np.ogrid[:size, :size]
    mask = (yy - size / 2) ** 2 + (xx - size / 2) ** 2 <= (size / 4) ** 2
    frame = np.full((size, size), 10, dtype=np.uint16)
    frame[mask] = value
    return frame


def test_train_eval_predict_with_frozen_encoder_head(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    for position, death_frame in (("Pos0", 4), ("Pos1", 5)):
        pos_dir = data_dir / "roi" / position
        pos_dir.mkdir(parents=True)
        pages = []
        for frame in range(8):
            bf = (
                _blob(220)
                if frame < death_frame
                else np.full((32, 32), 10, dtype=np.uint16)
            )
            toto = _blob(40 if frame < death_frame else 500)
            pages.extend([bf, toto])
        tifffile.imwrite(
            pos_dir / "Roi0.tif", np.stack(pages), photometric="minisblack"
        )
    labels = tmp_path / "labels.json"
    store = LabelStore(labels)
    store.set_death_frame(data_dir, RoiRef("Pos0", 0), 4)
    store.set_death_frame(data_dir, RoiRef("Pos1", 0), 5)

    dataset_dir = tmp_path / "dataset"
    manifest = build_dataset_manifest(
        data_dir=data_dir,
        labels_path=labels,
        output_dir=dataset_dir,
        val_fraction=0.5,
        seed=0,
    )
    assert manifest.train_cells == 1
    assert manifest.val_cells == 1
    train_labels = {
        sample.label
        for sample in load_manifest_samples(dataset_dir / "manifest.json", "train")
    }
    assert train_labels == {0, 1}

    output_dir = tmp_path / "run"
    encoder = FakeEncoder()
    head = train_linear_head(
        dataset_dir / "manifest.json",
        output_dir,
        encoder=encoder,
        device="cpu",
    )
    assert (output_dir / "head.json").is_file()
    assert (output_dir / "embeddings.npz").is_file()
    assert head.encoder_model_id == MODEL_ID

    metrics = evaluate_head(
        output_dir / "head.json",
        dataset_dir / "manifest.json",
        encoder=encoder,
        embeddings_path=output_dir / "embeddings.npz",
        device="cpu",
    )
    assert {item.split for item in metrics} == {"train", "val"}
    assert all(item.samples == 8 for item in metrics)
    assert all(item.accuracy == 1.0 for item in metrics)

    results = infer_all_cells(
        data_dir,
        output_dir / "head.json",
        encoder=encoder,
        device="cpu",
    )
    assert len(results) == 2
    by_pos = {cell.position: cell for cell in results}
    assert by_pos["Pos0"].death_time_viability == 4
    pos1 = by_pos["Pos1"].death_probability
    assert all(value < 0.5 for value in pos1[:5])
    assert all(value > 0.5 for value in pos1[5:])
    assert json.loads((output_dir / "head.json").read_text())["kind"] == "linear-svc"
