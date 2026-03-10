from .interfaces import DetectionAdapter, Tracker
from .pipeline import TrackingPipeline
from .tracker import LightweightTracker, TrackerConfig
from .types import Detection, FrameDetections, Track, TrackState

StalkrTracker = LightweightTracker

__all__ = [
    "Detection",
    "DetectionAdapter",
    "FrameDetections",
    "LightweightTracker",
    "StalkrTracker",
    "TrackingPipeline",
    "Track",
    "Tracker",
    "TrackerConfig",
    "TrackState",
]
