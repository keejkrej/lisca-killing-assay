"""Viable/dead reference classification on ROI movies."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import math
from collections.abc import Awaitable, Callable
from contextlib import AbstractAsyncContextManager
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import tifffile
from fastapi import File, Form, HTTPException, Request, UploadFile
from PIL import Image
from pydantic import BaseModel, Field

MAX_PIXELS = 1_048_576


class Example(BaseModel):
    image_base64: str = Field(max_length=2_800_000)
    label: str = Field(pattern="^(viable|dead)$")
    group: str = Field(min_length=1, max_length=160)


class References(BaseModel):
    name: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    contrast: str = Field(default="frame", pattern="^(frame|baseline)$")
    examples: list[Example] = Field(min_length=2, max_length=128)


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
                not isinstance(entry, dict) for entry in entries
            ):
                raise ValueError("ROI index must contain a list of ROI objects")
            matches = [entry for entry in entries if entry.get("fileName") == filename]
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
        if not axes.endswith("YX") or any(axis not in "TCZYX" for axis in axes):
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
    """Cosine kNN over viable/dead reference vectors. This is not the SVM fit."""
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


def import_references(store: Any, directory: Path, encoder: Any, name: str) -> None:
    manifest = json.loads((directory / "manifest.json").read_text())
    metadata = json.loads((directory / "embeddings.json").read_text())
    rows = manifest["rows"]
    if (
        metadata["model_id"] != encoder.model_id
        or metadata["revision"] != encoder.revision
    ):
        raise ValueError("Reference encoder revision does not match server")
    if metadata["ids"] != [row["id"] for row in rows]:
        raise ValueError("Reference embedding order does not match manifest")
    vectors = np.load(directory / "embeddings.npy", allow_pickle=False)
    if (
        vectors.shape != (len(rows), encoder.dimensions)
        or not np.isfinite(vectors).all()
    ):
        raise ValueError("Invalid reference embeddings")
    if (np.linalg.norm(vectors, axis=1) == 0).any():
        raise ValueError("Zero reference embeddings are invalid")
    reference_groups = {row["group"] for row in rows if row["split"] == "reference"}
    evaluation_groups = {row["group"] for row in rows if row["split"] == "evaluation"}
    if reference_groups & evaluation_groups:
        raise ValueError("Reference and evaluation ROIs overlap")
    indices = [
        i
        for i, row in enumerate(rows)
        if row["split"] == "reference" and row.get("label") in {"viable", "dead"}
    ]
    labels = [rows[i]["label"] for i in indices]
    if set(labels) != {"viable", "dead"}:
        raise ValueError("References must contain viable and dead examples")
    value = {
        "name": name,
        "contrast": "frame",
        "labels": labels,
        "groups": [rows[i]["group"] for i in indices],
        "vectors": vectors[indices].tolist(),
        "source": str(directory),
    }
    try:
        existing = store.collection(name)
    except ValueError:
        store.add_collection(value)
    else:
        if existing != value:
            raise ValueError("Reference set already exists with different contents")


def prepare(store: Any, encoder: Any, reference_dir: Path | None) -> None:
    if reference_dir is not None:
        import_references(store, reference_dir, encoder, "fig6-apoptosis")


def mount(
    app: Any,
    *,
    decode_image: Callable[[bytes], Image.Image],
    max_body: int,
    admission: Callable[[Request], AbstractAsyncContextManager[None]],
    embed_images: Callable[[Any, list[Image.Image]], Awaitable[np.ndarray]],
) -> None:
    @app.get("/v1/references")
    async def references(request: Request) -> list[dict[str, Any]]:
        return [
            {
                "name": item["name"],
                "count": len(item["labels"]),
                "contrast": item["contrast"],
            }
            for item in request.app.state.engine.store.collections()
        ]

    @app.post("/v1/references", status_code=201)
    async def add_references(body: References, request: Request) -> dict[str, Any]:
        if {example.label for example in body.examples} != {"viable", "dead"}:
            raise ValueError("Include at least one viable and one dead reference")
        async with admission(request):
            engine = request.app.state.engine
            images = await asyncio.to_thread(
                lambda: [
                    decode_image(base64.b64decode(example.image_base64, validate=True))
                    for example in body.examples
                ]
            )
            vectors = await embed_images(engine, images)
            engine.store.add_collection(
                {
                    "name": body.name,
                    "contrast": body.contrast,
                    "labels": [example.label for example in body.examples],
                    "groups": [example.group for example in body.examples],
                    "vectors": vectors.tolist(),
                    "source": "user-examples",
                }
            )
            return {"name": body.name, "count": len(images), "contrast": body.contrast}

    @app.post("/v1/viability")
    async def viability(
        request: Request,
        movie: Annotated[UploadFile, File()],
        reference_set: Annotated[str, Form()],
        group: Annotated[str, Form(min_length=1, max_length=160)],
        channel: Annotated[int, Form(ge=0)] = 0,
        z: Annotated[int, Form(ge=0)] = 0,
        stride: Annotated[int, Form(ge=1, le=100)] = 1,
        history: Annotated[int, Form(ge=1, le=10)] = 1,
        roi_index: Annotated[UploadFile | None, File()] = None,
    ) -> dict[str, Any]:
        async with admission(request):
            engine = request.app.state.engine
            collection = engine.store.collection(reference_set)
            if group in collection["groups"]:
                raise ValueError(
                    "This ROI is in the reference set; use a different ROI"
                )
            raw = await movie.read(max_body + 1)
            if len(raw) > max_body:
                raise HTTPException(413, "Movie exceeds 64 MiB")
            index = None
            if roi_index:
                index_bytes = await roi_index.read(1_048_577)
                if len(index_bytes) > 1_048_576:
                    raise ValueError("ROI index exceeds 1 MiB")
                index = json.loads(index_bytes)
                if not isinstance(index, dict):
                    raise ValueError("ROI index must be a JSON object")
            info = await asyncio.to_thread(
                movie_info, raw, channel, z, stride, index, movie.filename
            )
            vectors = []
            for start in range(0, len(info["frames"]), 32):
                if await request.is_disconnected():
                    raise HTTPException(499, "Client disconnected")
                images = await asyncio.to_thread(
                    movie_images,
                    raw,
                    info,
                    info["frames"][start : start + 32],
                    channel,
                    z,
                    collection["contrast"],
                )
                vectors.extend(await embed_images(engine, images))
            array = np.stack(vectors)
            result = await asyncio.to_thread(classify, array, collection)
            support = result["viable_support"]
            smoothed = [
                float(np.mean(support[max(0, i - history + 1) : i + 1]))
                for i in range(len(support))
            ]
            return {
                "group": group,
                "reference_set": reference_set,
                "model_id": engine.encoder.model_id,
                "revision": engine.encoder.revision,
                "frames": info["frame_ids"],
                "frame_source": info["frame_source"],
                "channel": channel,
                "z": z,
                "stride": stride,
                "contrast": collection["contrast"],
                "history": history,
                **result,
                "smoothed_support": smoothed,
                "step": step_fit(smoothed, info["frame_ids"]),
                "note": (
                    "Vote support is not a calibrated probability. Step is a "
                    "retrospective one-transition fit. Frame IDs use the ROI "
                    "index when supplied, otherwise stored TIFF indices."
                ),
            }
