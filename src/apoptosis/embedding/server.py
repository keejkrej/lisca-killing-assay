"""Private-tailnet inference service. Run with python -m apoptosis.embedding.server."""

from __future__ import annotations

import argparse
import asyncio
import base64
import hmac
import json
import os
import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .engine import MODEL, REVISION, BatchEngine, BusyError, FrozenEncoder, Store
from .images import classify, decode_image, movie_images, movie_info, step_fit

MAX_BODY = 64 * 1024 * 1024


class Example(BaseModel):
    image_base64: str = Field(max_length=2_800_000)
    label: str = Field(pattern="^(viable|dead)$")
    group: str = Field(min_length=1, max_length=160)


class References(BaseModel):
    name: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    contrast: str = Field(default="frame", pattern="^(frame|baseline)$")
    examples: list[Example] = Field(min_length=2, max_length=128)


class EmbeddingRequest(BaseModel):
    images: list[str] = Field(min_length=1, max_length=32)


class BodyTooLarge(HTTPException):
    def __init__(self):
        super().__init__(413, "Upload exceeds 64 MiB")


class RequestGuard:
    def __init__(self, app, token: str):
        self.app, self.token = app, token

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] == "OPTIONS":
            return await self.app(scope, receive, send)
        headers = dict(scope["headers"])
        expected = f"Bearer {self.token}".encode()
        if not hmac.compare_digest(headers.get(b"authorization", b""), expected):
            return await JSONResponse({"detail": "Invalid server token"}, 401)(
                scope, receive, send
            )
        size = 0

        async def bounded_receive():
            nonlocal size
            message = await receive()
            size += len(message.get("body", b""))
            if size > MAX_BODY:
                raise BodyTooLarge()
            return message

        try:
            length = int(headers.get(b"content-length", b"0"))
            if length > MAX_BODY:
                raise BodyTooLarge()
            await self.app(scope, bounded_receive, send)
        except BodyTooLarge:
            await JSONResponse({"detail": "Upload exceeds 64 MiB"}, 413)(
                scope, receive, send
            )


def import_references(store: Store, directory: Path, name: str) -> None:
    manifest = json.loads((directory / "manifest.json").read_text())
    metadata = json.loads((directory / "embeddings.json").read_text())
    rows = manifest["rows"]
    if metadata["model_id"] != MODEL or metadata["revision"] != REVISION:
        raise ValueError("Reference encoder revision does not match server")
    if metadata["ids"] != [r["id"] for r in rows]:
        raise ValueError("Reference embedding order does not match manifest")
    vectors = np.load(directory / "embeddings.npy", allow_pickle=False)
    if vectors.shape != (len(rows), 768) or not np.isfinite(vectors).all():
        raise ValueError("Invalid reference embeddings")
    if (np.linalg.norm(vectors, axis=1) == 0).any():
        raise ValueError("Zero reference embeddings are invalid")
    reference_groups = {r["group"] for r in rows if r["split"] == "reference"}
    evaluation_groups = {r["group"] for r in rows if r["split"] == "evaluation"}
    if reference_groups & evaluation_groups:
        raise ValueError("Reference and evaluation ROIs overlap")
    indices = [
        i
        for i, r in enumerate(rows)
        if r["split"] == "reference" and r.get("label") in {"viable", "dead"}
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


def create_app(
    *,
    token: str,
    data_dir: Path,
    origins: list[str],
    encoder=None,
    batch_size: int = 8,
    wait_ms: float = 10,
    reference_dir: Path | None = None,
    device: str = "cuda",
) -> FastAPI:
    if len(token) < 24 or token.startswith("replace-"):
        raise ValueError("LISCA_INFERENCE_TOKEN must have at least 24 characters")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        actual_encoder = encoder or await asyncio.to_thread(FrozenEncoder, device)
        store = Store(data_dir / "inference.sqlite3", actual_encoder.identity)
        if reference_dir:
            import_references(store, reference_dir, "fig6-apoptosis")
        engine = BatchEngine(
            actual_encoder, store, batch_size=batch_size, wait_ms=wait_ms
        )
        app.state.engine = engine
        app.state.active = 0
        engine.start()
        try:
            yield
        finally:
            await engine.close()

    app = FastAPI(title="LiSCA inference", version="1.0", lifespan=lifespan)
    app.add_middleware(RequestGuard, token=token)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type"],
    )

    @app.exception_handler(BusyError)
    async def busy(_request, error):
        return JSONResponse({"detail": str(error)}, 503, headers={"Retry-After": "2"})

    @app.exception_handler(ValueError)
    async def invalid(_request, error):
        return JSONResponse({"detail": str(error)}, 422)

    @app.exception_handler(sqlite3.IntegrityError)
    async def conflict(_request, _error):
        return JSONResponse({"detail": "Reference set name already exists"}, 409)

    @asynccontextmanager
    async def admission(request: Request):
        if request.app.state.active >= 4:
            raise BusyError("Four requests already active; retry shortly")
        request.app.state.active += 1
        try:
            yield
        finally:
            request.app.state.active -= 1

    async def embed_images(engine, images):
        vectors = []
        for start in range(0, len(images), 32):
            vectors.extend(
                await asyncio.gather(
                    *[engine.embed(image) for image in images[start : start + 32]]
                )
            )
        return np.stack(vectors)

    @app.get("/v1/health")
    async def health(request: Request):
        engine = request.app.state.engine
        return {
            "api_version": 1,
            "model_id": MODEL,
            "revision": REVISION,
            "dimensions": 768,
            "device": engine.encoder.device,
            "batch_size": engine.batch_size,
            "queue_depth": engine.queue.qsize(),
            "stats": engine.stats,
        }

    @app.get("/v1/references")
    async def references(request: Request):
        return [
            {"name": c["name"], "count": len(c["labels"]), "contrast": c["contrast"]}
            for c in request.app.state.engine.store.collections()
        ]

    @app.post("/v1/references", status_code=201)
    async def add_references(body: References, request: Request):
        if {e.label for e in body.examples} != {"viable", "dead"}:
            raise ValueError("Include at least one viable and one dead reference")
        async with admission(request):
            engine = request.app.state.engine
            images = await asyncio.to_thread(
                lambda: [
                    decode_image(base64.b64decode(e.image_base64, validate=True))
                    for e in body.examples
                ]
            )
            vectors = await embed_images(engine, images)
            engine.store.add_collection(
                {
                    "name": body.name,
                    "contrast": body.contrast,
                    "labels": [e.label for e in body.examples],
                    "groups": [e.group for e in body.examples],
                    "vectors": vectors.tolist(),
                    "source": "user-examples",
                }
            )
            return {"name": body.name, "count": len(images), "contrast": body.contrast}

    @app.post("/v1/embeddings")
    async def embeddings(body: EmbeddingRequest, request: Request):
        async with admission(request):
            images = await asyncio.to_thread(
                lambda: [
                    decode_image(base64.b64decode(s, validate=True))
                    for s in body.images
                ]
            )
            vectors = await embed_images(request.app.state.engine, images)
            return {
                "model_id": MODEL,
                "revision": REVISION,
                "vectors": vectors.tolist(),
            }

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
    ):
        async with admission(request):
            engine = request.app.state.engine
            collection = engine.store.collection(reference_set)
            if group in collection["groups"]:
                raise ValueError(
                    "This ROI is in the reference set; use a different ROI"
                )
            raw = await movie.read(MAX_BODY + 1)
            if len(raw) > MAX_BODY:
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
            # Classify raw frames, then optionally average their support causally.
            # Reference embeddings remain single-frame exemplars.
            result = await asyncio.to_thread(classify, array, collection)
            support = result["viable_support"]
            smoothed = [
                float(np.mean(support[max(0, i - history + 1) : i + 1]))
                for i in range(len(support))
            ]
            return {
                "group": group,
                "reference_set": reference_set,
                "model_id": MODEL,
                "revision": REVISION,
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

    return app


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8910)
    parser.add_argument(
        "--data-dir", type=Path, default=Path.home() / "data/lisca-inference"
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=os.environ.get("LISCA_INFERENCE_REFERENCES"),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--batch-wait-ms", type=float, default=10)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    app = create_app(
        token=os.environ.get("LISCA_INFERENCE_TOKEN", ""),
        data_dir=args.data_dir,
        reference_dir=args.reference_dir,
        origins=os.environ.get(
            "LISCA_INFERENCE_ORIGINS",
            "http://localhost:18767,http://127.0.0.1:18767,tauri://localhost,http://tauri.localhost,https://tauri.localhost",
        ).split(","),
        batch_size=args.batch_size,
        wait_ms=args.batch_wait_ms,
        device=args.device,
    )
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=1,
        limit_concurrency=16,
        timeout_keep_alive=30,
        access_log=False,
    )


if __name__ == "__main__":
    main()
