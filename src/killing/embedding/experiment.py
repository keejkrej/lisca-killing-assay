"""Reproducible ROI sampling and frozen EmbeddingGemma inference experiments."""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
from PIL import Image, ImageDraw, ImageOps

from killing.embedding.reference_classification import evaluate_manifest, unit_vectors
from killing.embedding.roi_index import load_position_index


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def display_plane(
    plane: np.ndarray, limits: tuple[float, float] | None = None
) -> Image.Image:
    values = np.asarray(plane, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Expected a finite grayscale image")
    low, high = np.percentile(values, [1, 99]) if limits is None else limits
    if high <= low:
        pixels = np.zeros(values.shape, dtype=np.uint8)
    else:
        pixels = (np.clip((values - low) / (high - low), 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(pixels).convert("RGB")


def import_frame_manifest(
    manifest_path: Path,
    output: Path,
    *,
    frame_stride: int = 10,
    workspace: Path | None = None,
    label_map: dict[str, str] | None = None,
    label_source: str = "existing-frame-manifest",
) -> dict[str, Any]:
    """Import labeled frames, retaining an existing train/val split by whole ROI.

    Input: data_dir and samples with position, roi_id, time_index, label, split.
    time_index indexes the stored stack; acquisition frames come from index.json.
    """
    if frame_stride < 1 or output.exists():
        raise ValueError("Use a positive stride and a new output directory")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    workspace = workspace or Path(payload["data_dir"])
    if label_map is not None and (
        not isinstance(label_map, dict)
        or any(not v.strip() for v in label_map.values())
    ):
        raise ValueError("label_map must map original labels to nonempty class names")
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for sample in payload["samples"]:
        if sample["split"] not in {"train", "val"}:
            raise ValueError("Expected train/val splits in source manifest")
        if label_map is not None and str(sample["label"]) not in label_map:
            raise ValueError(f"No label mapping for {sample['label']!r}")
        groups[(str(sample["position"]), int(sample["roi_id"]))].append(sample)
    if not groups:
        raise ValueError("No labeled samples")
    for samples in groups.values():
        if len({sample["split"] for sample in samples}) != 1:
            raise ValueError("Source manifest puts an ROI in both splits")
    output.mkdir(parents=True)
    (output / "images").mkdir()
    rows: list[dict[str, Any]] = []
    for (position, roi_id), samples in groups.items():
        index = load_position_index(workspace, int(position.removeprefix("Pos")))
        if index.axis_order != "TCZYX":
            raise ValueError("Only TCZYX ROI stacks are supported")
        roi = next((roi for roi in index.rois if roi.roi == roi_id), None)
        if roi is None:
            raise ValueError(f"Unknown ROI {position}/Roi{roi_id}")
        path = workspace / "roi" / position / roi.file_name
        with tifffile.TiffFile(path) as tif:
            if len(tif.pages) != index.time_count * index.channel_count * index.z_count:
                raise ValueError(f"Page count mismatch: {path}")
            for sample in samples:
                frame = int(sample["time_index"])
                if frame < 0 or frame >= index.time_count:
                    raise ValueError(f"Frame out of range: {path}, {frame}")
                if frame % frame_stride:
                    continue
                source_frame = index.time_indices[frame]
                image_id = f"{position}-Roi{roi_id}-t{source_frame}-c0-z0"
                image_path = f"images/{image_id}.png"
                plane = tif.pages[frame * index.channel_count * index.z_count].asarray()
                display_plane(plane).save(output / image_path)
                key = str(sample["label"])
                label = key if label_map is None else label_map[key]
                rows.append(
                    {
                        "id": image_id,
                        "image": image_path,
                        "group": f"{position}/Roi{roi_id}",
                        "source": str(path.resolve()),
                        "frame": source_frame,
                        "stack_frame": frame,
                        "channel": 0,
                        "z": 0,
                        "split": "reference"
                        if sample["split"] == "train"
                        else "evaluation",
                        "label": label,
                        "label_source": label_source,
                        "notes": "Imported label; no model-generated labels.",
                    }
                )
    ids = [row["id"] for row in rows]
    if not rows or len(ids) != len(set(ids)):
        raise ValueError("The selected manifest must contain unique images")
    result = {
        "workspace": str(workspace.resolve()),
        "preprocessing": "per-frame percentile 1/99 uint8 RGB",
        "source_manifest": str(manifest_path.resolve()),
        "source_manifest_sha256": hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest(),
        "frame_stride": frame_stride,
        "evaluation_context": {
            "status": "existing-label-validation",
            "biologically_validated": False,
            "note": (
                "Original train/validation ROI split; agreement with supplied labels."
            ),
        },
        "rows": rows,
    }
    write_json(output / "manifest.json", result)
    return result


def sample_rois(
    workspace: Path,
    output: Path,
    *,
    roi_count: int = 24,
    frames_per_roi: int = 3,
    channel: int = 0,
    z: int = 0,
    seed: int = 42,
    contrast: str = "baseline",
) -> dict[str, Any]:
    """Use LiSCA index metadata to avoid confusing channels with time frames."""
    if roi_count < 2 or frames_per_roi < 1 or channel < 0 or z < 0:
        raise ValueError("Need at least two ROIs, positive frame count, and valid C/Z")
    if output.exists():
        raise ValueError("Sample output already exists; choose a new directory")
    if contrast not in {"baseline", "frame"}:
        raise ValueError("contrast must be baseline or frame")
    candidates = []
    for index_path in sorted((workspace / "roi").glob("Pos*/index.json")):
        index = load_position_index(workspace, int(index_path.parent.name[3:]))
        if index.axis_order != "TCZYX":
            raise ValueError(f"Unsupported axis order in {index_path}")
        for roi in index.rois:
            path = index_path.parent / roi.file_name
            if not path.resolve().is_relative_to(index_path.parent.resolve()):
                raise ValueError(f"ROI file escapes its position: {path}")
            if path.is_file():
                candidates.append((path, roi.shape, index.time_indices))
    if len(candidates) < 2:
        raise ValueError("Need at least two indexed ROI TIFFs in workspace/roi/Pos*/")
    rng = random.Random(seed)
    selected = rng.sample(candidates, min(roi_count, len(candidates)))
    reference_count = min(len(selected) - 1, max(1, int(len(selected) * 0.7)))
    output.mkdir(parents=True)
    (output / "images").mkdir()
    rows: list[dict[str, Any]] = []
    for number, (path, shape, time_indices) in enumerate(selected):
        time_count, channel_count, z_count, height, width = shape
        if channel >= channel_count or z >= z_count:
            raise ValueError(f"Channel or Z outside shape {shape}: {path}")
        frames = sorted(rng.sample(range(time_count), min(frames_per_roi, time_count)))
        with tifffile.TiffFile(path) as tif:
            if len(tif.pages) != time_count * channel_count * z_count:
                raise ValueError(f"TIFF page count disagrees with index: {path}")
            limits = None
            if contrast == "baseline":
                baseline = tif.pages[channel * z_count + z].asarray()
                low, high = np.percentile(baseline, [1, 99])
                limits = (float(low), float(high))
            for frame in frames:
                page = (frame * channel_count + channel) * z_count + z
                plane = tif.pages[page].asarray()
                if plane.shape != (height, width):
                    raise ValueError(f"TIFF plane shape disagrees with index: {path}")
                source_frame = time_indices[frame]
                image_id = (
                    f"{path.parent.name}-{path.stem}-t{source_frame}-c{channel}-z{z}"
                )
                image_path = f"images/{image_id}.png"
                display_plane(plane, limits).save(output / image_path)
                rows.append(
                    {
                        "id": image_id,
                        "image": image_path,
                        "group": f"{path.parent.name}/{path.stem}",
                        "source": str(path.resolve()),
                        "frame": source_frame,
                        "stack_frame": frame,
                        "channel": channel,
                        "z": z,
                        "split": "reference"
                        if number < reference_count
                        else "evaluation",
                        "label": None,
                        "label_source": None,
                        "notes": "",
                        "contrast_limits": limits,
                    }
                )
    manifest = {
        "workspace": str(workspace.resolve()),
        "seed": seed,
        "preprocessing": (
            f"{contrast} percentile 1/99 clip to uint8, grayscale replicated to RGB; "
            "baseline uses the first stored frame of each ROI"
        ),
        "rows": rows,
    }
    write_json(output / "manifest.json", manifest)
    # Paginate so both humans and multimodal reviewers can inspect actual pixels.
    for start in range(0, len(rows), 24):
        subset = rows[start : start + 24]
        sheet = Image.new("RGB", (4 * 260, ((len(subset) + 3) // 4) * 294), "#202020")
        draw = ImageDraw.Draw(sheet)
        for j, row in enumerate(subset):
            x, y = (j % 4) * 260, (j // 4) * 294
            with Image.open(output / row["image"]) as image:
                sheet.paste(image.resize((256, 256)), (x, y))
            draw.text((x + 2, y + 258), row["id"], fill="white")
            draw.text((x + 2, y + 274), row["split"], fill="white")
        sheet.save(output / f"contact-{start // 24 + 1:02d}.jpg")
    return manifest


def review_sequences(directory: Path, *, frames: int = 8) -> dict[str, Any]:
    """Review temporal context at a fixed per-ROI scale, without changing labels."""
    if frames < 2 or frames > 16:
        raise ValueError("Use between 2 and 16 overview frames")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in manifest["rows"]:
        groups.setdefault(str(row["group"]), []).append(row)
    strips = []
    metadata = []
    for group, rows in groups.items():
        row = rows[0]
        source = Path(row["source"])
        position = int(source.parent.name[3:])
        index = load_position_index(Path(manifest["workspace"]), position)
        sampled = {int(r["stack_frame"]) for r in rows}
        selected = sorted(
            sampled
            | set(np.linspace(0, index.time_count - 1, frames, dtype=int).tolist())
        )
        strip = Image.new("RGB", (len(selected) * 180, 222), "#202020")
        draw = ImageDraw.Draw(strip)
        draw.text(
            (4, 2),
            f"{group} | {row['split']} | fixed baseline contrast | * sampled",
            fill="white",
        )
        with tifffile.TiffFile(source) as tif:
            offset = int(row["channel"]) * index.z_count + int(row["z"])
            low, high = np.percentile(tif.pages[offset].asarray(), [1, 99])
            limits = (float(low), float(high))
            for j, frame in enumerate(selected):
                page = frame * index.channel_count * index.z_count + offset
                plane = tif.pages[page].asarray()
                image = ImageOps.contain(display_plane(plane, limits), (176, 176))
                x = j * 180
                strip.paste(image, (x, 22))
                marker = " *" if frame in sampled else ""
                draw.text(
                    (x + 2, 201),
                    f"Frame {index.time_indices[frame]}{marker}",
                    fill="white",
                )
        strips.append(strip)
        metadata.append(
            {"group": group, "contrast_limits": limits, "stack_frames": selected}
        )
    for start in range(0, len(strips), 4):
        subset = strips[start : start + 4]
        sheet = Image.new(
            "RGB", (max(s.width for s in subset), len(subset) * 222), "#202020"
        )
        for j, strip in enumerate(subset):
            sheet.paste(strip, (0, j * 222))
        sheet.save(directory / f"sequence-{start // 4 + 1:02d}.jpg")
    result = {"groups": metadata}
    write_json(directory / "sequences.json", result)
    return result


def embed_sample(
    directory: Path,
    *,
    model_id: str = "google/embeddinggemma-2",
    revision: str | None = None,
    device: str = "cuda",
    batch_size: int = 8,
) -> dict[str, Any]:
    """Labels are excluded from the cache identity; the encoder is always frozen."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if not manifest["rows"]:
        raise ValueError("The sample contains no images")
    images = [directory / row["image"] for row in manifest["rows"]]
    from huggingface_hub import HfApi  # ty: ignore[unresolved-import]

    # Resolve main once to an immutable commit; subsequent offline reuse uses it.
    meta_path = directory / "embeddings.json"
    previous = json.loads(meta_path.read_text()) if meta_path.exists() else None
    if revision is None and previous and previous["model_id"] == model_id:
        revision = str(previous["revision"])
    if revision is None or not re.fullmatch(r"[0-9a-f]{40}", revision):
        resolved = HfApi().model_info(model_id, revision=revision).sha
        if not isinstance(resolved, str):
            raise ValueError("The model revision could not be resolved")
        revision = resolved
    identity = {
        "model_id": model_id,
        "revision": revision,
        "ids": [row["id"] for row in manifest["rows"]],
        "image_sha256": [
            hashlib.sha256(path.read_bytes()).hexdigest() for path in images
        ],
        "preprocessing": manifest["preprocessing"],
        "normalize": True,
        "dtype": "float32",
        "image_prompt": None,
    }
    cache = directory / "embeddings.npy"
    if (
        previous
        and cache.exists()
        and all(previous.get(k) == v for k, v in identity.items())
    ):
        return {**previous, "cache_reused": True}

    import torch
    from sentence_transformers import (  # ty: ignore[unresolved-import]
        SentenceTransformer,
    )

    model = SentenceTransformer(
        model_id,
        revision=revision,
        device=device,
        model_kwargs={"torch_dtype": torch.float32},
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    batches = []
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            batch = []
            for path in images[start : start + batch_size]:
                with Image.open(path) as image:
                    batch.append({"image": image.convert("RGB")})
            encoded = model.encode(
                batch,
                batch_size=batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            batches.append(np.asarray(encoded, dtype=np.float32))
            print(
                f"Embedded {min(start + batch_size, len(images))}/{len(images)}",
                flush=True,
            )
    vectors = unit_vectors(np.concatenate(batches)).astype(np.float32)
    np.save(cache, vectors, allow_pickle=False)
    metadata = {**identity, "shape": list(vectors.shape)}
    write_json(meta_path, metadata)
    return {**metadata, "cache_reused": False}


def compare_sample(
    directory: Path, *, neighbors: int = 3, svm_c: float = 1.0
) -> dict[str, Any]:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    metadata = json.loads((directory / "embeddings.json").read_text(encoding="utf-8"))
    if metadata["ids"] != [row["id"] for row in manifest["rows"]]:
        raise ValueError("Manifest order differs from the embedding cache")
    hashes = [
        hashlib.sha256((directory / row["image"]).read_bytes()).hexdigest()
        for row in manifest["rows"]
    ]
    if hashes != metadata["image_sha256"]:
        raise ValueError("Images changed since embedding; run embed again")
    vectors = np.load(directory / "embeddings.npy", allow_pickle=False)
    result = evaluate_manifest(
        manifest["rows"], vectors, neighbors=neighbors, svm_c=svm_c
    )
    result["embedding"] = metadata
    result["parameters"] = {"neighbors": neighbors, "svm_c": svm_c}
    result["evaluation_context"] = manifest.get(
        "evaluation_context", {"status": "exploratory", "biologically_validated": False}
    )
    write_json(directory / "comparison.json", result)
    return result
