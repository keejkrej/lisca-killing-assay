"""Read ``roi/Pos{n}/index.json`` without the product workspace package."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RoiEntry:
    roi: int
    file_name: str
    shape: tuple[int, int, int, int, int]


@dataclass(frozen=True)
class PositionIndex:
    position: int
    axis_order: str
    time_count: int
    channel_count: int
    z_count: int
    rois: list[RoiEntry]
    time_indices: list[int]


def load_position_index(workspace: Path, position: int) -> PositionIndex:
    """Stack shape is time, channel, z, then bbox height and width."""
    path = workspace / "roi" / f"Pos{position}" / "index.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    axis_order = str(raw.get("axisOrder", "TCZYX")).upper()
    time_count = int(raw.get("timeCount", 1))
    channel_count = int(raw.get("channelCount", 1))
    z_count = int(raw.get("zCount", 1))
    rois: list[RoiEntry] = []
    for entry in raw.get("rois", []):
        bbox_raw = entry.get("bbox")
        if not isinstance(bbox_raw, dict):
            raise ValueError(f"{path} ROI entry is missing bbox")
        rois.append(
            RoiEntry(
                roi=int(entry["roi"]),
                file_name=str(entry["fileName"]),
                shape=(
                    time_count,
                    channel_count,
                    z_count,
                    int(bbox_raw["h"]),
                    int(bbox_raw["w"]),
                ),
            )
        )
    raw_indices = raw.get("timeIndices")
    if raw_indices is None:
        time_indices = list(range(time_count))
    else:
        time_indices = [int(value) for value in raw_indices]
        if len(time_indices) != time_count:
            raise ValueError(
                f"{path}: timeIndices length {len(time_indices)} "
                f"does not match timeCount {time_count}"
            )
    return PositionIndex(
        position=int(raw["position"]),
        axis_order=axis_order,
        time_count=time_count,
        channel_count=channel_count,
        z_count=z_count,
        rois=rois,
        time_indices=time_indices,
    )
