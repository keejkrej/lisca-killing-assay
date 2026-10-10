"""Audit frozen-embedding errors and compare three fixed, causal feature choices."""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from killing.embedding.experiment import compare_sample, write_json
from killing.embedding.reference_classification import evaluate_manifest
from killing.embedding.temporal_classification import temporal_embeddings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--positive-label", default="dead")
    parser.add_argument("--negative-label", default="viable")
    args = parser.parse_args()
    directory = args.directory
    baseline = compare_sample(directory)  # Verify image hashes and vector ordering.
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    rows = manifest["rows"]
    vectors = np.load(directory / "embeddings.npy", allow_pickle=False)
    by_id = {row["id"]: row for row in rows}
    output = directory / "context-comparison"
    output.mkdir(exist_ok=True)
    reports = {"frame": baseline}
    for mode in ("trailing_mean", "baseline_delta"):
        features = temporal_embeddings(rows, vectors, mode=mode, window=3)
        reports[mode] = evaluate_manifest(rows, features)
    summary: dict[str, Any] = {
        "note": (
            "Exploratory comparison on the same validation ROIs, not an untouched "
            "test. Three fixed feature choices; k=3 and C=1 throughout. "
            "Temporal context uses only current and earlier observations of the "
            "same ROI, without reading labels or adapting the encoder."
        ),
        "context": {
            "window_observations": 3,
            "frame_stride": manifest.get("frame_stride"),
        },
        "embedding": baseline["embedding"],
        "evaluation_context": baseline["evaluation_context"],
        "methods": {},
    }
    for mode, report in reports.items():
        write_json(output / f"{mode}.json", report)
        methods = {}
        for method, result in report["results"].items():
            per_roi: dict[str, dict[str, Any]] = {}
            errors = []
            for image_id, prediction in zip(
                report["query_ids"], result["predictions"], strict=True
            ):
                row = by_id[image_id]
                group = per_roi.setdefault(
                    row["group"],
                    {
                        "images": 0,
                        "errors": 0,
                        "false_positive": 0,
                        "false_negative": 0,
                    },
                )
                if not row.get("label") or row["label"] == "uncertain":
                    continue
                group["images"] += 1
                if prediction != row["label"]:
                    group["errors"] += 1
                    group["false_positive"] += int(prediction == args.positive_label)
                    group["false_negative"] += int(row["label"] == args.positive_label)
                    errors.append(
                        {
                            "id": image_id,
                            "group": row["group"],
                            "frame": row["frame"],
                            "label": row["label"],
                            "prediction": prediction,
                        }
                    )
            methods[method] = {
                "metrics": report["metrics"][method],
                "per_roi": per_roi,
                "errors": errors,
            }
        summary["methods"][mode] = methods
        print(json.dumps({"mode": mode, "metrics": report["metrics"]}), flush=True)
    write_json(output / "summary.json", summary)

    # A representative error per ROI and classifier, rather than many near-duplicates.
    chosen = {}
    for method in ("knn", "svm"):
        seen = set()
        for error in summary["methods"]["frame"][method]["errors"]:
            key = (error["group"], error["label"])
            if key not in seen:
                chosen[error["id"]] = error
                seen.add(key)
    selected = list(chosen.values())
    predictions = {
        method: dict(
            zip(
                baseline["query_ids"],
                baseline["results"][method]["predictions"],
                strict=True,
            )
        )
        for method in ("knn", "svm")
    }
    for start in range(0, len(selected), 12):
        subset = selected[start : start + 12]
        sheet = Image.new("RGB", (4 * 250, ((len(subset) + 3) // 4) * 300), "#202020")
        draw = ImageDraw.Draw(sheet)
        for j, error in enumerate(subset):
            x, y = (j % 4) * 250, (j // 4) * 300
            with Image.open(directory / by_id[error["id"]]["image"]) as image:
                sheet.paste(ImageOps.contain(image.convert("RGB"), (240, 240)), (x, y))
            lines = [
                error["id"],
                f"Manual label: {error['label']}",
                f"kNN: {predictions['knn'][error['id']]} / "
                f"SVM: {predictions['svm'][error['id']]}",
            ]
            for line, text in enumerate(lines):
                draw.text((x + 2, y + 242 + line * 17), text, fill="white")
        sheet.save(output / f"errors-{start // 12 + 1:02d}.jpg")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    groups: dict[str, list[str]] = defaultdict(list)
    for image_id in baseline["query_ids"]:
        groups[by_id[image_id]["group"]].append(image_id)
    fig, axes = plt.subplots(
        len(groups), 1, figsize=(13, 1.65 * len(groups)), squeeze=False
    )
    colors = {args.negative_label: "#24856e", args.positive_label: "#c74953"}
    for ax, (group, image_ids) in zip(axes[:, 0], groups.items(), strict=True):
        image_ids.sort(key=lambda image_id: by_id[image_id]["frame"])
        frames = [by_id[image_id]["frame"] for image_id in image_ids]
        tracks = {
            "Manual label": [by_id[image_id].get("label") for image_id in image_ids]
        }
        for mode, report in reports.items():
            for method in ("knn", "svm"):
                mapped = dict(
                    zip(
                        report["query_ids"],
                        report["results"][method]["predictions"],
                        strict=True,
                    )
                )
                tracks[f"{mode} / {method}"] = [
                    mapped[image_id] for image_id in image_ids
                ]
        for i, labels in enumerate(tracks.values()):
            ax.scatter(
                frames,
                [i] * len(frames),
                c=[colors.get(label, "#aaaaaa") for label in labels],
                marker="s",
                s=35,
            )
        ax.set_yticks(range(len(tracks)), tracks.keys(), fontsize=7)
        ax.set_title(group, loc="left", fontsize=10)
        ax.set_ylim(len(tracks) - 0.5, -0.5)
        ax.tick_params(axis="x", labelsize=8)
        ax.spines[["top", "right", "left"]].set_visible(False)
    axes[-1, 0].set_xlabel("Acquisition frame (every tenth frame embedded)")
    fig.legend(
        handles=[Patch(color=color, label=label) for label, color in colors.items()],
        loc="upper right",
    )
    fig.suptitle(
        "Figure 6 validation ROIs: manual labels and frozen-embedding predictions",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output / "validation-trajectories.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
