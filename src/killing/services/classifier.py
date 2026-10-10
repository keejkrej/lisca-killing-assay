"""Write cleaned labels, death times, and the kill curve.

Reads ``results/predictions.csv`` (``t,crop,label,pos,sample``; extra columns
are ignored). Writes ``predictions_cleaned.csv``, ``death_times.csv``, and
``kill_curve.csv`` beside it. Sample order follows ``assay.json``.
"""

from __future__ import annotations

import csv
from pathlib import Path

from killing.core.classifier import clean_predictions, death_times, kill_curve
from killing.core.mapping import load_samples


def run_clean(workspace: Path) -> None:
    predictions = workspace / "results" / "predictions.csv"
    rows = _read_predictions(predictions)
    cleaned = clean_predictions(rows)
    times = death_times(cleaned)
    samples, _interval = load_samples(workspace)
    results = workspace / "results"
    _write(
        results / "predictions_cleaned.csv",
        ["t", "crop", "label", "pos", "sample"],
        [
            [
                str(int(row["t"])),
                str(int(row["crop"])),
                "true" if row["label"] else "false",
                str(int(row["pos"])),
                str(row["sample"]),
            ]
            for row in cleaned
        ],
    )
    _write(
        results / "death_times.csv",
        ["crop", "death_time", "pos", "sample"],
        [
            [str(crop), str(times[key]), str(pos), sample]
            for key in sorted(times)
            for pos, sample, crop in [key]
        ],
    )
    curve_rows: list[list[str]] = []
    for sample in samples:
        for t, n_alive in kill_curve(times, sample.name):
            curve_rows.append([str(t), str(n_alive), sample.name])
    _write(results / "kill_curve.csv", ["t", "n_alive", "sample"], curve_rows)


def _read_predictions(path: Path) -> list[dict[str, object]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        for required in ("t", "crop", "label", "sample"):
            if required not in fields:
                raise ValueError(f"missing {required} column in {path.name}")
        rows: list[dict[str, object]] = []
        for record in reader:
            pos_raw = record.get("pos") or "0"
            rows.append(
                {
                    "t": int(float(record["t"])),
                    "crop": int(float(record["crop"])),
                    "label": record["label"].strip().lower() in {"true", "1", "yes"},
                    "pos": int(float(pos_raw)) if pos_raw.strip() else 0,
                    "sample": record["sample"],
                }
            )
    return rows


def _write(path: Path, headers: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(headers)
        writer.writerows(rows)
