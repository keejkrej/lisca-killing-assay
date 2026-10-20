from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

from killing.core.roi import CHANNEL_BRIGHTFIELD, RoiRef, page_index, roi_path
from killing.ml.preprocess import frame_to_image, normalize_frame

__all__ = [
    "ManifestSample",
    "load_brightfield_image",
    "load_manifest_samples",
    "normalize_frame",
    "sample_id",
]


@dataclass(frozen=True)
class ManifestSample:
    position: str
    roi_id: int
    time_index: int
    label: int
    split: str

    @property
    def sample_id(self) -> str:
        return sample_id(self.position, self.roi_id, self.time_index)


def sample_id(position: str, roi_id: int, time_index: int) -> str:
    return f"{position}/Roi{roi_id}/t{time_index}"


def load_manifest_samples(
    manifest_path: Path, split: str | None = None
) -> list[ManifestSample]:
    payload = json.loads(manifest_path.read_text())
    samples = [ManifestSample(**item) for item in payload["samples"]]
    if split is None:
        return samples
    return [sample for sample in samples if sample.split == split]


def load_brightfield_image(data_dir: Path, sample: ManifestSample) -> Image.Image:
    roi = RoiRef(position=sample.position, roi_id=sample.roi_id)
    stack = tifffile.imread(roi_path(data_dir, roi))
    frame = np.asarray(stack[page_index(sample.time_index, CHANNEL_BRIGHTFIELD)])
    return frame_to_image(frame)
