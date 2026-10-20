from __future__ import annotations

import numpy as np
from PIL import Image

from killing.core.roi import CHANNEL_BRIGHTFIELD, page_index


def normalize_frame(frame: np.ndarray) -> np.ndarray:
    low, high = np.percentile(frame, (1, 99))
    if high <= low:
        high = low + 1
    scaled = np.clip((frame.astype(np.float32) - low) / (high - low), 0, 1)
    return scaled.astype(np.float32)


def frame_to_image(frame: np.ndarray) -> Image.Image:
    """Percentile-scaled grayscale replicated to RGB. The encoder resizes."""
    scaled = normalize_frame(frame.copy())
    return Image.fromarray((scaled * 255).astype(np.uint8)).convert("RGB")


def stack_brightfield_image(stack: np.ndarray, frame: int) -> Image.Image:
    return frame_to_image(stack[page_index(frame, CHANNEL_BRIGHTFIELD)].copy())
