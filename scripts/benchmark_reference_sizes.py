"""Measure few-example classification on a fixed validation split, without tuning."""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from killing.embedding.experiment import compare_sample, write_json
from killing.embedding.reference_classification import evaluate_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    # Check cache alignment and image hashes before reusing vectors.
    full = compare_sample(args.directory)
    manifest = json.loads(
        (args.directory / "manifest.json").read_text(encoding="utf-8")
    )
    rows = manifest["rows"]
    vectors = np.load(args.directory / "embeddings.npy", allow_pickle=False)
    classes = sorted(
        {
            r["label"]
            for r in rows
            if r["split"] == "reference"
            and r.get("label")
            and r["label"] != "uncertain"
        }
    )
    pools = {
        label: [
            i
            for i, r in enumerate(rows)
            if r["split"] == "reference" and r.get("label") == label
        ]
        for label in classes
    }
    results = []
    for count in (8, 16, 32, 64, 128, 256):
        per_class = count // len(classes)
        if per_class < 1 or any(len(pool) < per_class for pool in pools.values()):
            continue
        trials: list[dict[str, Any]] = []
        for seed in range(42, 47):
            rng = np.random.default_rng(seed)
            selected = {
                int(i)
                for pool in pools.values()
                for i in rng.permutation(pool)[:per_class]
            }
            subset = [
                {**r, "label": None}
                if r["split"] == "reference" and i not in selected
                else r
                for i, r in enumerate(rows)
            ]
            report = evaluate_manifest(subset, vectors)
            trials.append(
                {
                    "seed": seed,
                    "reference_ids": report["reference_ids"],
                    "metrics": report["metrics"],
                }
            )
        result = {
            "reference_images": per_class * len(classes),
            "trials": trials,
            "mean_metrics": {
                method: {
                    metric: float(
                        np.mean([trial["metrics"][method][metric] for trial in trials])
                    )
                    for metric in ("accuracy", "macro_recall")
                }
                for method in ("knn", "svm")
            },
        }
        results.append(result)
        print(
            json.dumps(
                {key: result[key] for key in ("reference_images", "mean_metrics")}
            )
        )
    write_json(
        args.directory / "reference-size-comparison.json",
        {
            "note": (
                "Class-balanced random reference subsets, five fixed seeds; "
                "same validation ROIs throughout. Not independent test sets "
                "or a hyperparameter search."
            ),
            "results": results,
            "all_references": {
                "count": len(full["reference_ids"]),
                "metrics": full["metrics"],
            },
        },
    )


if __name__ == "__main__":
    main()
