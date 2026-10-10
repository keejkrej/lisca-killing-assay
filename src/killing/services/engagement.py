"""Write fluorescent-engagement tables for one Workspace.

Studio still draws the figures. This service writes the same
``analysis/Pos{n}/engagement.csv`` columns as the Rust crate.
"""

from __future__ import annotations

import csv
from pathlib import Path

from killing.core.engagement import (
    FluorescenceHistogram,
    count_engagements,
    mask_with_scale,
    tumor_mask,
)
from killing.core.mapping import SampleChannels, load_samples
from killing.services.roi_stack import (
    frame,
    load_stack,
    read_position_index,
    validate_channel,
)


def run_engagement(workspace: Path) -> None:
    samples, interval_minutes = load_samples(workspace)
    positions = []
    seen: set[int] = set()
    for sample in samples:
        for position in sample.positions:
            if position not in seen:
                seen.add(position)
                positions.append(position)
    for position in positions:
        owners = [sample for sample in samples if position in sample.positions]
        if len(owners) != 1:
            raise ValueError(f"Pos{position} must belong to one sample")
        run_position(workspace, owners[0], position, interval_minutes)


def run_position(
    workspace: Path,
    sample: SampleChannels,
    position: int,
    interval_minutes: float,
) -> Path:
    pos_dir = workspace / "roi" / f"Pos{position}"
    if not pos_dir.is_dir():
        raise FileNotFoundError(
            f"No ROI directory found for position {position}: {pos_dir}"
        )
    index = read_position_index(pos_dir / "index.json")
    if not sample.signal:
        raise ValueError(f"sample {sample.name} has no signal channel for engagement")
    signal = sample.signal[0]
    validate_channel(index, sample.segmentation)
    validate_channel(index, signal)
    stacks = [load_stack(pos_dir / roi["fileName"], index) for roi in index["rois"]]
    histogram = FluorescenceHistogram()
    for stack in stacks:
        for stack_t in range(index["timeCount"]):
            histogram.sample(frame(stack, index, stack_t, signal))
    scale = histogram.scale()
    rows: list[dict[str, object]] = []
    for roi, stack in zip(index["rois"], stacks, strict=True):
        for stack_t in range(index["timeCount"]):
            source_t = index["timeIndices"][stack_t]
            tumor = tumor_mask(frame(stack, index, stack_t, sample.segmentation))
            engagers = mask_with_scale(frame(stack, index, stack_t, signal), scale)
            height, width = frame(stack, index, stack_t, signal).shape
            counts = count_engagements(tumor, engagers, width, height)
            rows.append(
                {
                    "roi": roi["roi"],
                    "t": source_t,
                    "minutes": f"{source_t * interval_minutes:.4f}",
                    "tumor_cells": counts.tumor_cells,
                    "t_cells": counts.t_cells,
                    "engagements": counts.engagements,
                }
            )
    if not rows:
        raise ValueError(f"position {position} has no ROI frames to score")
    rows.sort(key=lambda row: (int(row["roi"]), int(row["t"])))
    output = workspace / "analysis" / f"Pos{position}" / "engagement.csv"
    _write_csv(
        output,
        ["roi", "t", "minutes", "tumor_cells", "t_cells", "engagements"],
        rows,
    )
    return output


def _write_csv(path: Path, headers: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
