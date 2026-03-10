# stalkr

`stalkr` is a lightweight tracking-by-detection package intended to sit behind
custom detector stacks such as RF-DETR-Mac. The baseline implementation in this
repo stays dependency-light and cross-platform, while exposing enough extension
points for MLX, appearance embeddings, and detector-specific optimizations.

The broader goal across `stalkr` and
[`rf-detr-mac`](https://github.com/Ak-Gautam/rf-detr-mac) is a simple pipeline
for video understanding on desktop and edge systems: detect objects, track them
across frames, and then build downstream operations such as mask extraction,
counting, region-based events, and movement analysis on top.

## Design goals

- Lightweight core with no mandatory heavy ML runtime
- Pluggable detector adapters and tracker policies
- Good real-time defaults for video overlays and path drawing
- Clear upgrade path for Apple Silicon optimizations

## Current tracker

The initial tracker is an online, two-stage association tracker with:

- A constant-velocity Kalman filter for motion prediction
- Hungarian assignment for global data association
- ByteTrack-style high-confidence then low-confidence matching
- Configurable score thresholds
- IoU-first matching with optional appearance fusion
- Track lifecycle states: tracked, lost, removed
- Hit and miss counters on every track
- Per-track point history for drawing trajectories on video frames

This is intentionally small enough to evolve toward ByteTrack or OC-SORT style
behavior without inheriting the weight of a large benchmarking repository.

## Intended usage

`stalkr` is the tracker layer. It should be easy to pair with OpenCV for:

- drawing bounding boxes on video frames
- drawing track paths and motion trails
- counting objects across lines or regions
- exporting track histories for analytics or post-processing

The detector stays outside this repo. Your model adapter should convert detector
outputs into `FrameDetections`, then `stalkr` handles identity association and
track state management.

## Quick start

```python
from stalkr import Detection, FrameDetections, LightweightTracker

tracker = LightweightTracker()

detections = FrameDetections(
    frame_index=0,
    items=[
        Detection(box=(10, 20, 110, 180), score=0.92, class_id=0),
        Detection(box=(220, 50, 290, 160), score=0.81, class_id=0),
    ],
)

tracks = tracker.update(detections)

for track in tracks:
    print(track.track_id, track.box, track.path)
```

## RF-DETR-Mac integration shape

Your detector integration layer should convert model outputs into
`FrameDetections` and call `tracker.update(...)` per frame. The returned
`Track` objects include box coordinates, score, lifecycle state, and path
history that can be rendered on top of the video.

## Roadmap

- detector adapters for RF-DETR-Mac and other model outputs
- OpenCV helpers for drawing boxes, labels, and path lines
- counting primitives for lines, zones, and directional flow
- optional mask-aware tracking and mask propagation hooks
- optional MLX-backed appearance features for difficult re-identification
- optional accelerated assignment backends without changing the public API

## Next steps

- Add a `numpy` or `lap` accelerated assignment backend
- Reuse RF-DETR decoder/query features as appearance descriptors
- Add an MLX-native embedding head for difficult re-identification cases
- Add camera motion compensation for moving-camera footage

## Credits

Built by [Gautam](https://github.com/Ak-Gautam).
Follow on [GitHub](https://github.com/Ak-Gautam) and
[X / Twitter](https://x.com/Gautam_A_k).

## License

This project is licensed under the Apache-2.0 license. See
[`LICENSE`](./LICENSE) for the full text.

Inspired by [roboflow/trackers](https://github.com/roboflow/trackers).
