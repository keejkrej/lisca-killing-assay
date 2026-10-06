"""Plot a held-out ROI's cached predictions and an offline one-transition fit."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageOps

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--group", required=True)
    args = parser.parse_args()
    root = args.directory
    manifest = json.loads((root / "manifest.json").read_text())
    rows = sorted(
        [r for r in manifest["rows"] if r["group"] == args.group],
        key=lambda r: r["frame"],
    )
    if not rows or any(r["split"] != "evaluation" for r in rows):
        raise ValueError("Choose a nonempty held-out evaluation ROI")
    frames = np.array([r["frame"] for r in rows])
    scores = {}
    for mode in ("frame", "trailing_mean"):
        report = json.loads((root / "context-comparison" / f"{mode}.json").read_text())
        result = report["results"]["knn"]
        if set(report["metrics"]["knn"]["classes"]) != {"dead", "viable"}:
            raise ValueError("Expected binary dead/viable predictions")
        mapped = dict(
            zip(
                report["query_ids"],
                [
                    support if label == "viable" else 1 - support
                    for label, support in zip(
                        result["predictions"], result["vote_support"], strict=True
                    )
                ],
                strict=True,
            )
        )
        scores[mode] = np.array([mapped[r["id"]] for r in rows])

    # Offline least squares over all possible viable -> dead change points.
    # Endpoints explicitly allow always dead and always viable. Labels are unused.
    candidates = np.arange(len(rows) + 1)
    states = (np.arange(len(rows))[None, :] < candidates[:, None]).astype(float)
    losses = ((states - scores["trailing_mean"]) ** 2).sum(axis=1)
    best = np.flatnonzero(np.isclose(losses, losses.min(), rtol=0, atol=1e-10))
    index = int(best[0])  # Report ties, rather than claiming a unique event.
    event = int(frames[index]) if index < len(rows) else None
    output = root / "single-roi" / args.group.replace("/", "-")
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "group": args.group,
        "frames": frames.tolist(),
        "viable_vote_support": {k: v.tolist() for k, v in scores.items()},
        "fitted_viability": states[index].tolist(),
        "first_dead_sample_frame": event,
        "equally_optimal_change_indices": best.tolist(),
        "sum_squared_error": float(losses[index]),
        "best_constant_sum_squared_error": float(min(losses[0], losses[-1])),
        "note": "Offline one-way step fitted to trailing-mean kNN vote support; "
        "not a calibrated probability or online event detector. Allows constant "
        "states. Manual labels used only for display. Frames sampled every tenth "
        "frame; no exact event time inferred between samples.",
    }
    (output / "trajectory.json").write_text(json.dumps(summary, indent=2) + "\n")

    fig = plt.figure(figsize=(12, 6.5), layout="constrained")
    grid = fig.add_gridspec(2, 6, height_ratios=[3, 1.4])
    ax = fig.add_subplot(grid[0, :])
    ax.plot(frames, scores["frame"], ".:", color="0.65", label="Single-frame kNN")
    ax.plot(
        frames,
        scores["trailing_mean"],
        "o-",
        color="#3585b5",
        markersize=4,
        label="3-observation kNN viable vote support",
    )
    ax.step(
        frames,
        states[index],
        where="post",
        color="#bd4c2f",
        linewidth=2.5,
        label="Offline one-transition fit",
    )
    truth = [1 if r["label"] == "viable" else 0 for r in rows]
    ax.step(
        frames,
        truth,
        where="post",
        color="#24856e",
        linestyle="--",
        linewidth=1.8,
        label="Manual labels (sampled)",
    )
    ax.set(
        xlabel="Acquisition frame",
        ylabel="Viability / viable vote support",
        ylim=(-0.08, 1.12),
        title=f"{args.group}: frozen-embedding viability",
    )
    ax.legend(loc="lower left", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    for col, j in enumerate(np.linspace(0, len(rows) - 1, 6, dtype=int)):
        tile = fig.add_subplot(grid[1, col])
        with Image.open(root / rows[j]["image"]) as im:
            tile.imshow(im)
        tile.set_title(f"Frame {frames[j]}", fontsize=9)
        tile.axis("off")
    fig.suptitle(
        "Every tenth frame; vote support is not a probability. "
        "Step fit uses the whole sequence.",
        fontsize=10,
    )
    fig.savefig(output / "viability.png", dpi=170)
    plt.close(fig)

    animation = []
    for j, row in enumerate(rows):
        canvas = Image.new("RGB", (480, 550), "#202020")
        with Image.open(root / row["image"]) as im:
            tile = ImageOps.contain(im.convert("RGB"), (480, 480))
            canvas.paste(tile, ((480 - tile.width) // 2, 0))
        draw = ImageDraw.Draw(canvas)
        draw.text((12, 490), f"{args.group} | Frame {frames[j]}", fill="white")
        draw.text(
            (12, 510),
            f"Viable vote: {scores['trailing_mean'][j]:.2f} | "
            f"Offline step: {int(states[index, j])}",
            fill="white",
        )
        draw.text(
            (12, 530),
            "Every tenth frame; playback is not acquisition time",
            fill="white",
        )
        animation.append(canvas)
    animation[0].save(
        output / "sampled-movie.gif",
        save_all=True,
        append_images=animation[1:],
        duration=350,
        loop=0,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
