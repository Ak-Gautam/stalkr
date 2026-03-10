from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

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
