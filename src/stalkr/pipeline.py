from __future__ import annotations

from typing import Any, Protocol

from .interfaces import DetectionAdapter, Tracker
from .types import FrameDetections, Track


class Detector(Protocol):
    """Runs inference and returns model-specific outputs."""

    def predict(self, frame: Any) -> Any: ...


class TrackingPipeline:
    """Thin composition layer for detector + adapter + tracker."""

    def __init__(
        self,
        detector: Detector,
        adapter: DetectionAdapter,
        tracker: Tracker,
    ) -> None:
        self.detector = detector
        self.adapter = adapter
        self.tracker = tracker

    def process(
        self,
        frame: Any,
        *,
        frame_index: int | None = None,
        timestamp: float | None = None,
    ) -> list[Track]:
        raw_output = self.detector.predict(frame)
        detections = self.adapter.parse(
            raw_output,
            frame_index=frame_index,
            timestamp=timestamp,
        )
        return self.tracker.update(detections)

    def update(self, detections: FrameDetections) -> list[Track]:
        return self.tracker.update(detections)
