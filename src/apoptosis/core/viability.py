from __future__ import annotations

from apoptosis.core.roi import is_healthy_label


def frame_is_viable(death_frame: int, frame: int, frame_count: int) -> bool:
    if is_healthy_label(death_frame, frame_count):
        return True
    return frame < death_frame


def frame_label(death_frame: int, frame: int, frame_count: int) -> int:
    return int(frame_is_viable(death_frame, frame, frame_count))
