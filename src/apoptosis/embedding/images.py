"""Bounded image decoding and explicit ROI TIFF axes."""

import io
import math
from typing import Any

import numpy as np
import tifffile
from PIL import Image, UnidentifiedImageError

MAX_PIXELS = 1_048_576


def decode_image(raw: bytes) -> Image.Image:
    try:
        with Image.open(io.BytesIO(raw)) as image:
            if image.width * image.height > MAX_PIXELS:
                raise ValueError("Reference image exceeds one megapixel")
            if image.format not in {"PNG", "JPEG"}:
                raise ValueError("References must be PNG or JPEG")
            return image.convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise ValueError("Cannot decode reference image") from error


def movie_info(
    raw: bytes,
    channel: int,
    z: int,
    stride: int,
    index: dict[str, Any] | None = None,
    filename: str | None = None,
) -> dict[str, Any]:
    with tifffile.TiffFile(io.BytesIO(raw)) as tif:
        series = tif.series[0]
        axes, shape = series.axes, series.shape
        if index is not None:
            if index.get("axisOrder") != "TCZYX":
                raise ValueError("ROI index must declare TCZYX axis order")
            entries = index.get("rois", [])
            if not isinstance(entries, list) or any(
                not isinstance(r, dict) for r in entries
            ):
                raise ValueError("ROI index must contain a list of ROI objects")
            matches = [r for r in entries if r.get("fileName") == filename]
            if len(matches) != 1:
                raise ValueError("Movie filename must match one ROI in index.json")
            shape = tuple(matches[0].get("shape", []))
            if len(shape) != 5 or any(type(n) is not int or n <= 0 for n in shape):
                raise ValueError("Invalid TCZYX shape in index.json")
            if tuple(shape[-2:]) != tuple(series.shape[-2:]):
                raise ValueError("ROI index dimensions do not match TIFF")
            axes = "TCZYX"
        elif axes in {"QYX", "IYX"}:
            raise ValueError(
                "TIFF has no time/channel axes; include its ROI index.json"
            )
        if not axes.endswith("YX") or any(a not in "TCZYX" for a in axes):
            raise ValueError(
                f"Unsupported TIFF axes {axes}; use TYX or TCZYX ROI stacks"
            )
        dims = dict(zip(axes, shape, strict=True))
        if dims["Y"] * dims["X"] > MAX_PIXELS:
            raise ValueError("ROI exceeds one megapixel")
        if dims.get("T", 1) > 4096:
            raise ValueError("Movie exceeds 4096 frames")
        if channel >= dims.get("C", 1) or z >= dims.get("Z", 1):
            raise ValueError("Channel or Z is outside this TIFF")
        if len(series.pages) != math.prod(shape[:-2]):
            raise ValueError("TIFF must store one 2D plane per page")
        frame_ids = (
            index.get("timeIndices", list(range(dims.get("T", 1))))
            if index is not None
            else list(range(dims.get("T", 1)))
        )
        if (
            len(frame_ids) != dims.get("T", 1)
            or any(type(n) is not int or n < 0 for n in frame_ids)
            or any(a >= b for a, b in zip(frame_ids, frame_ids[1:], strict=False))
        ):
            raise ValueError("Invalid acquisition Frame IDs in index.json")
        return {
            "axes": axes,
            "shape": shape,
            "frames": list(range(0, dims.get("T", 1), stride)),
            "frame_ids": frame_ids[::stride],
            "frame_source": "roi-index" if index is not None else "tiff-index",
        }


def movie_images(
    raw: bytes,
    info: dict[str, Any],
    frames: list[int],
    channel: int,
    z: int,
    contrast: str,
) -> list[Image.Image]:
    axes, shape = info["axes"], info["shape"]
    with tifffile.TiffFile(io.BytesIO(raw)) as tif:

        def pixels(frame: int) -> np.ndarray:
            coord = {"T": frame, "C": channel, "Z": z}
            index = (
                int(
                    np.ravel_multi_index(tuple(coord[a] for a in axes[:-2]), shape[:-2])
                )
                if axes[:-2]
                else 0
            )
            page = tif.series[0].pages[index]
            if page is None:
                raise ValueError("Missing TIFF page")
            array = page.asarray().astype(np.float32)
            if not np.isfinite(array).all():
                raise ValueError("TIFF contains non-finite pixels")
            return array

        fixed = np.percentile(pixels(0), [1, 99]) if contrast == "baseline" else None
        result = []
        for frame in frames:
            array = pixels(frame)
            low, high = fixed if fixed is not None else np.percentile(array, [1, 99])
            scaled = np.clip((array - low) / max(float(high - low), 1e-6), 0, 1)
            result.append(
                Image.fromarray((scaled * 255).astype(np.uint8)).convert("RGB")
            )
        return result


def classify(vectors: np.ndarray, collection: dict[str, Any]) -> dict[str, Any]:
    references = np.array(collection["vectors"], dtype=np.float32)
    references /= np.linalg.norm(references, axis=1, keepdims=True)
    labels = np.array(collection["labels"])
    similarities = vectors @ references.T
    indices = np.argsort(-similarities, axis=1, kind="stable")[:, :3]
    weights = 1 / np.maximum(
        1 - np.take_along_axis(similarities, indices, axis=1), 1e-6
    )
    viable = (weights * (labels[indices] == "viable")).sum(axis=1) / weights.sum(axis=1)
    return {
        "viable_support": viable.tolist(),
        "predictions": np.where(viable > 0.5, "viable", "dead").tolist(),
    }


def step_fit(support: list[float], frames: list[int]) -> dict[str, Any]:
    # Linear-time least-squares comparison, including the two constant states.
    values = np.asarray(support)
    losses = (
        np.r_[0, np.cumsum((1 - values) ** 2)]
        + np.r_[np.cumsum((values**2)[::-1])[::-1], 0]
    )
    best = np.flatnonzero(np.isclose(losses, losses.min(), rtol=0, atol=1e-10))
    index = int(best[0])
    return {
        "viability": (np.arange(len(values)) < index).astype(int).tolist(),
        "first_dead_frame": frames[index] if index < len(frames) else None,
        "state": "always_dead"
        if index == 0
        else "always_viable"
        if index == len(frames)
        else "transition",
        "tied_indices": best.tolist(),
        "retrospective": True,
    }
