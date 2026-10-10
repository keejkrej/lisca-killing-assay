"""Write death-reporter fluorescence tables for one Workspace.

Columns match the Rust crate: ``roi,t,area,background,sum,corrected`` under
``analysis/Pos{n}/ch{m}.csv``. The segmentation channel is not read.
"""

from __future__ import annotations

import csv
from pathlib import Path

from apoptosis.core.fluorescence import measure_frame
from apoptosis.core.mapping import SampleChannels, load_samples
from apoptosis.services.roi_stack import (
    frame,
    load_stack,
    read_position_index,
    validate_channel,
)


def run_fluorescence(workspace: Path) -> None:
    samples, _interval = load_samples(workspace)
    positions: list[int] = []
    seen: set[int] = set()
    for sample in samples:
        for position in sample.positions:
            if position not in seen:
                seen.add(position)
                positions.append(position)
    if not positions:
        raise ValueError("sample mapping defines no positions")
    for position in positions:
        owners = [sample for sample in samples if position in sample.positions]
        run_position(workspace, owners, position)


def run_position(workspace: Path, samples: list[SampleChannels], position: int) -> None:
    if not samples:
        raise ValueError(f"no sample in assay.json lists Pos{position}")
    pos_dir = workspace / "roi" / f"Pos{position}"
    index = read_position_index(pos_dir / "index.json")
    wrote = False
    seen: set[int] = set()
    for sample in samples:
        for signal in sample.signal:
            if signal in seen:
                continue
            seen.add(signal)
            validate_channel(index, signal)
            _write_channel(workspace, pos_dir, index, position, signal)
            wrote = True
    if not wrote:
        raise ValueError(f"Pos{position} has no signal channel")


def _write_channel(
    workspace: Path,
    pos_dir: Path,
    index: dict,
    position: int,
    signal: int,
) -> None:
    rows: list[dict[str, object]] = []
    for roi in index["rois"]:
        stack = load_stack(pos_dir / roi["fileName"], index)
        for stack_t in range(int(index["timeCount"])):
            stats = measure_frame(frame(stack, index, stack_t, signal))
            rows.append(
                {
                    "roi": roi["roi"],
                    "t": index["timeIndices"][stack_t],
                    "area": stats["area"],
                    "background": stats["background"],
                    "sum": stats["sum"],
                    "corrected": stats["corrected"],
                }
            )
    if not rows:
        raise ValueError(f"No fluorescence rows for Pos{index['position']}")
    rows.sort(key=lambda row: (int(row["roi"]), int(row["t"])))
    path = workspace / "analysis" / f"Pos{position}" / f"ch{signal}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["roi", "t", "area", "background", "sum", "corrected"],
        )
        writer.writeheader()
        writer.writerows(rows)
