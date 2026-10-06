import asyncio
import base64
import io
import time

import numpy as np
import pytest
import tifffile
from PIL import Image

from apoptosis.embedding.engine import BatchEngine, BusyError, Store
from apoptosis.embedding.images import movie_images, movie_info, step_fit

TOKEN = "test-only-token-with-at-least-24-characters"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}


class FakeEncoder:
    identity = "test-encoder-v1"
    device = "test"

    def __init__(self):
        self.calls = []

    def encode(self, images):
        self.calls.append(len(images))
        time.sleep(0.005)
        return np.array([[np.asarray(im).mean() / 255, 1] for im in images])


def png(value):
    stream = io.BytesIO()
    Image.new("RGB", (8, 8), (value, value, value)).save(stream, format="PNG")
    return base64.b64encode(stream.getvalue()).decode()


def test_batching_dedup_cache_and_restart(tmp_path):
    async def run():
        encoder = FakeEncoder()
        path = tmp_path / "cache.db"
        engine = BatchEngine(encoder, Store(path, encoder.identity), wait_ms=20)
        engine.start()
        images = [Image.new("RGB", (8, 8), (v, v, v)) for v in [0, 128, 128]]
        result = await asyncio.gather(*[engine.embed(im) for im in images])
        np.testing.assert_array_equal(result[1], result[2])
        assert encoder.calls == [2]
        assert engine.stats["deduplicated"] == 1
        await engine.close()
        engine = BatchEngine(encoder, Store(path, encoder.identity))
        engine.start()
        await engine.embed(images[0])
        assert encoder.calls == [2]
        await engine.close()

    asyncio.run(run())


def test_queue_backpressure_and_shutdown_resolve_waiters(tmp_path):
    async def run():
        engine = BatchEngine(
            FakeEncoder(),
            Store(tmp_path / "cache.db", "test"),
            batch_size=1,
            queue_size=1,
        )
        pending = asyncio.create_task(engine.embed(Image.new("RGB", (2, 2))))
        await asyncio.sleep(0)
        with pytest.raises(BusyError):
            await engine.embed(Image.new("RGB", (2, 2), "white"))
        await engine.close()
        with pytest.raises(BusyError):
            await pending

    asyncio.run(run())


def test_auth_cors_reference_creation_and_movie(tmp_path):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    pytest.importorskip("multipart")
    from fastapi.testclient import TestClient

    from apoptosis.embedding.server import create_app

    app = create_app(
        token=TOKEN,
        data_dir=tmp_path,
        origins=["http://localhost:18767"],
        encoder=FakeEncoder(),
    )
    with TestClient(app) as client:
        assert client.get("/v1/health").status_code == 401
        assert client.get("/v1/health", headers=HEADERS).json()["api_version"] == 1
        response = client.options(
            "/v1/health",
            headers={
                "Origin": "http://localhost:18767",
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "Authorization",
            },
        )
        assert (
            response.headers["access-control-allow-origin"] == "http://localhost:18767"
        )
        examples = {
            "name": "example",
            "examples": [
                {"label": "dead", "group": "ref-a", "image_base64": png(0)},
                {"label": "viable", "group": "ref-b", "image_base64": png(255)},
            ],
        }
        assert (
            client.post("/v1/references", json=examples, headers=HEADERS).status_code
            == 201
        )
        assert (
            client.post("/v1/references", json=examples, headers=HEADERS).status_code
            == 409
        )
        stream = io.BytesIO()
        tifffile.imwrite(
            stream,
            np.arange(4 * 8 * 8, dtype=np.uint16).reshape(4, 8, 8),
            metadata={"axes": "TYX"},
            photometric="minisblack",
        )
        args = {"reference_set": "example", "group": "query", "stride": "2"}
        result = client.post(
            "/v1/viability",
            data=args,
            files={"movie": ("roi.tif", stream.getvalue())},
            headers=HEADERS,
        )
        assert result.status_code == 200, result.text
        assert result.json()["frames"] == [0, 2]
        assert len(result.json()["viable_support"]) == 2
        args["group"] = "ref-a"
        assert (
            client.post(
                "/v1/viability",
                data=args,
                files={"movie": ("roi.tif", stream.getvalue())},
                headers=HEADERS,
            ).status_code
            == 422
        )
        assert (
            client.post(
                "/v1/embeddings", json={"images": ["invalid"]}, headers=HEADERS
            ).status_code
            == 422
        )
        assert (
            client.post(
                "/v1/embeddings",
                content=b"",
                headers={**HEADERS, "Content-Length": str(65 * 1024 * 1024)},
            ).status_code
            == 413
        )


def test_tiff_channel_and_baseline_contrast():
    pixels = np.arange(64, dtype=np.uint16).reshape(8, 8)
    movie = np.stack(
        [np.stack([pixels, pixels * 2]), np.stack([pixels // 2, pixels * 3])]
    )
    stream = io.BytesIO()
    tifffile.imwrite(stream, movie, metadata={"axes": "TCYX"}, photometric="minisblack")
    raw = stream.getvalue()
    info = movie_info(raw, 1, 0, 1)
    frames = movie_images(raw, info, [0, 1], 0, 0, "baseline")
    assert np.asarray(frames[1]).mean() < np.asarray(frames[0]).mean()
    with pytest.raises(ValueError):
        movie_info(raw, 2, 0, 1)


def test_lisca_index_prevents_channel_time_confusion():
    stream = io.BytesIO()
    pixels = np.arange(64, dtype=np.uint16).reshape(8, 8)
    planes = np.stack([pixels, pixels * 3, pixels // 2, pixels * 4])
    tifffile.imwrite(stream, planes, metadata=None, photometric="minisblack")
    raw = stream.getvalue()
    with pytest.raises(ValueError, match="include its ROI index"):
        movie_info(raw, 0, 0, 1)
    index = {
        "axisOrder": "TCZYX",
        "timeIndices": [10, 25],
        "rois": [{"fileName": "Roi4.tif", "shape": [2, 2, 1, 8, 8]}],
    }
    info = movie_info(raw, 0, 0, 1, index, "Roi4.tif")
    assert info["frames"] == [0, 1]
    assert info["frame_ids"] == [10, 25]
    images = movie_images(raw, info, [0, 1], 0, 0, "baseline")
    assert np.asarray(images[1]).mean() < np.asarray(images[0]).mean()
    with pytest.raises(ValueError, match="match one ROI"):
        movie_info(raw, 0, 0, 1, index, "wrong.tif")


@pytest.mark.parametrize(
    "support,state,event",
    [
        ([1, 1, 1], "always_viable", None),
        ([0, 0, 0], "always_dead", 0),
        ([1, 1, 0], "transition", 20),
    ],
)
def test_step_does_not_require_death(support, state, event):
    result = step_fit(support, [0, 10, 20])
    assert result["state"] == state
    assert result["first_dead_frame"] == event
