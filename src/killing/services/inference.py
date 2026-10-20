from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import tifffile

from killing.core.roi import discover_rois, roi_frame_count, roi_path
from killing.core.toto import (
    death_time_from_probability,
    death_time_from_toto,
    position_median_traces,
    toto_alive_signal,
    toto_trace,
)
from killing.ml.encoder import ImageEncoder, load_encoder, require_matching_encoder
from killing.ml.head import LinearHead, dead_probability, load_head
from killing.ml.preprocess import stack_brightfield_image
from killing.services.embed import embed_images
from killing.services.head_train import default_head_path

POSITION_META = {
    "Pos0": {"label": "100 nM STS", "color": "#1f77b4"},
    "Pos28": {"label": "500 nM STS", "color": "#d62728"},
    "Pos70": {"label": "Control", "color": "#2ca02c"},
}


@dataclass(frozen=True)
class CellInference:
    position: str
    roi_id: int
    timepoints: int  # Frame count; key kept so existing inference.json loads
    death_time_toto: int
    death_time_viability: int
    toto_raw: list[float]
    toto_alive: list[float]
    death_probability: list[float]


def _predict_death_probability(
    encoder: ImageEncoder,
    head: LinearHead,
    stack: np.ndarray,
    frame_count: int,
    batch_size: int,
) -> np.ndarray:
    images = [stack_brightfield_image(stack, frame) for frame in range(frame_count)]
    vectors = embed_images(encoder, images, batch_size=batch_size)
    return dead_probability(head, vectors)


def infer_all_cells(
    data_dir: Path,
    head_path: Path,
    *,
    encoder: ImageEncoder | None = None,
    batch_size: int = 8,
    device: str = "cuda",
) -> list[CellInference]:
    head = load_head(head_path)
    if encoder is None:
        encoder = load_encoder(device=device)
    require_matching_encoder(head, encoder)

    all_rois = discover_rois(data_dir)
    medians = position_median_traces(data_dir, all_rois)

    results: list[CellInference] = []
    for roi in all_rois:
        stack = tifffile.imread(roi_path(data_dir, roi))
        frame_count = roi_frame_count(data_dir, roi)
        trace = toto_trace(data_dir, roi)
        # Use position-median residual for jump detection: suppresses common artifacts
        # (e.g. global illumination/focus shifts) while highlighting cell-specific
        # intensity increases (dye uptake on death). This applies to all positions,
        # including controls (Pos70 cells can still die).
        pos_median = medians.get(roi.position)
        contrast = (trace - pos_median) if pos_median is not None else trace
        death_prob = _predict_death_probability(
            encoder=encoder,
            head=head,
            stack=stack,
            frame_count=frame_count,
            batch_size=batch_size,
        )
        results.append(
            CellInference(
                position=roi.position,
                roi_id=roi.roi_id,
                timepoints=frame_count,
                death_time_toto=death_time_from_toto(
                    contrast, frame_count, raw_trace=trace
                ),
                death_time_viability=death_time_from_probability(
                    death_prob, frame_count
                ),
                toto_raw=trace.tolist(),
                toto_alive=toto_alive_signal(trace).tolist(),
                death_probability=death_prob.tolist(),
            )
        )
    return results


def save_inference(results: list[CellInference], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(item) for item in results]
    output_path.write_text(json.dumps(payload, indent=2) + "\n")


def load_inference(path: Path) -> list[CellInference]:
    payload = json.loads(path.read_text())
    return [CellInference(**item) for item in payload]


def resolve_head_path(project_root: Path, head: Path | None = None) -> Path:
    path = head or default_head_path(project_root)
    if not path.is_file():
        msg = f"No trained head at {path}"
        raise FileNotFoundError(msg)
    return path
