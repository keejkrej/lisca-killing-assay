import io

import numpy as np
import pytest
import tifffile

from killing.inference.viability import movie_images, movie_info, step_fit


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
