"""Death-reporter fluorescence on an existing ROI crop.

Area is the pixel count. Background is the 10th percentile. Corrected
fluorescence is the sum minus area times background. The segmentation
channel is not read.
"""

from __future__ import annotations

import numpy as np

from apoptosis.core.engagement import full_frame_roi_stats

__all__ = ["full_frame_roi_stats", "measure_frame"]


def measure_frame(frame: np.ndarray) -> dict[str, float | int]:
    area, intensity, background, corrected = full_frame_roi_stats(frame)
    return {
        "area": area,
        "background": background,
        "sum": intensity,
        "corrected": corrected,
    }
