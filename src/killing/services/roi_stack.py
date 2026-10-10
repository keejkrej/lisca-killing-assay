"""Read a Position ROI stack written by LiSCA crop."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tifffile


def read_position_index(path: Path) -> dict:
    payload = json.loads(path.read_text())
    if str(payload.get("axisOrder", "")).upper() != "TCZYX":
        raise ValueError(f"{path} axisOrder must be TCZYX")
    time_count = int(payload["timeCount"])
    indices = payload.get("timeIndices")
    if indices is None:
        indices = list(range(time_count))
    elif len(indices) != time_count:
        raise ValueError(f"{path} timeIndices length does not match timeCount")
    payload["timeIndices"] = [int(value) for value in indices]
    return payload


def validate_channel(index: dict, channel: int) -> None:
    count = int(index["channelCount"])
    if channel < 0 or channel >= count:
        raise ValueError(f"channel must be between 0 and {count - 1}, got {channel}")


def load_stack(path: Path, index: dict) -> np.ndarray:
    pages = tifffile.imread(path)
    array = np.asarray(pages)
    if array.ndim == 2:
        array = array[None, ...]
    time_count = int(index["timeCount"])
    channel_count = int(index["channelCount"])
    z_count = int(index["zCount"])
    expected = time_count * channel_count * z_count
    if array.shape[0] != expected:
        raise ValueError(
            f"{path} page count mismatch: expected {expected}, got {array.shape[0]}"
        )
    return array


def frame(stack: np.ndarray, index: dict, time: int, channel: int) -> np.ndarray:
    z_count = int(index["zCount"])
    channel_count = int(index["channelCount"])
    page = time * channel_count * z_count + channel * z_count
    return np.asarray(stack[page], dtype=np.float64)
