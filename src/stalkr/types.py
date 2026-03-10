from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from typing import Any, Literal

if TYPE_CHECKING:
    import numpy as np

TrackState = Literal["tracked", "lost", "removed"]
Box = tuple[float, float, float, float]
Point = tuple[float, float]


@dataclass(slots=True)
class Detection:
    box: Box
    score: float
    class_id: int | None = None
    embedding: tuple[float, ...] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FrameDetections:
    items: list[Detection]
    frame_index: int | None = None
    timestamp: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    _boxes_cache: "np.ndarray | None" = field(default=None, init=False, repr=False)

    def boxes_array(self) -> "np.ndarray":
        if self._boxes_cache is None:
            try:
                import numpy as np
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "boxes_array() requires numpy to be installed."
                ) from exc
            self._boxes_cache = np.asarray([item.box for item in self.items], dtype=np.float32)
            if self._boxes_cache.size == 0:
                self._boxes_cache = self._boxes_cache.reshape(0, 4)
        return self._boxes_cache


@dataclass(slots=True)
class Track:
    track_id: int
    box: Box
    score: float
    class_id: int | None
    state: TrackState
    hits: int
    age: int
    misses: int
    path: list[Point]
    velocity: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    mean: list[float] = field(default_factory=list)
    covariance: list[list[float]] = field(default_factory=list)
    embedding: tuple[float, ...] | None = None
    frame_index: int | None = None
    timestamp: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def center(self) -> Point:
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
