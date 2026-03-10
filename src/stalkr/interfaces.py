from __future__ import annotations

from typing import Any, Protocol

from .types import FrameDetections, Track


class DetectionAdapter(Protocol):
    """Converts model-specific outputs into canonical detections."""

    def parse(
        self,
        raw_output: Any,
        *,
        frame_index: int | None = None,
        timestamp: float | None = None,
    ) -> FrameDetections: ...


class Tracker(Protocol):
    """Minimal online tracker contract for video processing loops."""

    def update(self, detections: FrameDetections) -> list[Track]: ...

    def reset(self) -> None: ...

