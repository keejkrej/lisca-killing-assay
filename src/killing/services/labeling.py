from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from killing.core.labels import CellLabel, LabelStore
from killing.core.roi import (
    CHANNEL_BRIGHTFIELD,
    CHANNEL_TOTO,
    RoiRef,
    discover_rois,
    frame_to_png,
    is_healthy_label,
    roi_frame_count,
)


@dataclass(frozen=True)
class RoiSummary:
    position: str
    roi_id: int
    key: str
    frame_count: int
    labeled: bool
    death_frame: int | None
    is_healthy: bool | None


@dataclass
class LabelingSession:
    data_dir: Path
    labels_path: Path
    _rois: list[RoiRef] | None = None
    _frame_counts: dict[str, int] | None = None

    def __post_init__(self) -> None:
        if not self.data_dir.exists():
            msg = f"Data directory does not exist: {self.data_dir}"
            raise FileNotFoundError(msg)

    @property
    def store(self) -> LabelStore:
        return LabelStore(self.labels_path)

    def rois(self) -> list[RoiRef]:
        if self._rois is None:
            self._rois = discover_rois(self.data_dir)
        return self._rois

    def frame_count_for(self, roi: RoiRef) -> int:
        if self._frame_counts is None:
            self._frame_counts = {}
        if roi.key not in self._frame_counts:
            self._frame_counts[roi.key] = roi_frame_count(self.data_dir, roi)
        return self._frame_counts[roi.key]

    def list_rois(self) -> list[RoiSummary]:
        summaries: list[RoiSummary] = []
        for roi in self.rois():
            label = self.store.get(roi)
            frame_count = self.frame_count_for(roi)
            summaries.append(
                RoiSummary(
                    position=roi.position,
                    roi_id=roi.roi_id,
                    key=roi.key,
                    frame_count=frame_count,
                    labeled=label is not None,
                    death_frame=None if label is None else label.death_frame,
                    is_healthy=None
                    if label is None
                    else is_healthy_label(label.death_frame, frame_count),
                )
            )
        return summaries

    def get_roi(self, position: str, roi_id: int) -> RoiRef:
        roi = RoiRef(position=position, roi_id=roi_id)
        if not (self.data_dir / "roi" / position / roi.filename).exists():
            msg = f"ROI not found: {roi.key}"
            raise FileNotFoundError(msg)
        return roi

    def render_frame(
        self,
        roi: RoiRef,
        frame: int,
        channel: int = CHANNEL_BRIGHTFIELD,
    ) -> bytes:
        return frame_to_png(self.data_dir, roi, frame, channel)

    def save_label(self, roi: RoiRef, death_frame: int) -> CellLabel:
        return self.store.set_death_frame(self.data_dir, roi, death_frame)

    def roi_detail(self, roi: RoiRef) -> dict[str, object]:
        label = self.store.get(roi)
        frame_count = self.frame_count_for(roi)
        return {
            "position": roi.position,
            "roi_id": roi.roi_id,
            "key": roi.key,
            "frame_count": frame_count,
            "channels": {
                "brightfield": CHANNEL_BRIGHTFIELD,
                "toto": CHANNEL_TOTO,
            },
            "label": None
            if label is None
            else {
                "death_frame": label.death_frame,
                "is_healthy": is_healthy_label(label.death_frame, frame_count),
                "labeled_at": label.labeled_at,
            },
        }
