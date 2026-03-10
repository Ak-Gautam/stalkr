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

## Common workflows

### Track objects frame by frame

```python
from stalkr import Detection, FrameDetections, LightweightTracker

tracker = LightweightTracker()

for frame_index, model_output in enumerate(video_outputs):
    detections = FrameDetections(
        frame_index=frame_index,
        items=[
            Detection(box=box, score=score, class_id=class_id)
            for box, score, class_id in model_output
        ],
    )
    tracks = tracker.update(detections)

    for track in tracks:
        print(track.track_id, track.box, track.state)
```

### Draw boxes and track trails with OpenCV

```python
import cv2

for track in tracks:
    x1, y1, x2, y2 = map(int, track.box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
    cv2.putText(
        frame,
        f"id={track.track_id}",
        (x1, max(0, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 220, 0),
        1,
        cv2.LINE_AA,
    )

    for start, end in zip(track.path, track.path[1:]):
        cv2.line(
            frame,
            tuple(map(int, start)),
            tuple(map(int, end)),
            (255, 180, 0),
            2,
        )
```

### Count objects crossing a line

```python
line_x = 320
counted_ids = set()
count = 0

for track in tracks:
    if track.track_id in counted_ids or len(track.path) < 2:
        continue

    previous_x = track.path[-2][0]
    current_x = track.path[-1][0]

    crossed_line = previous_x < line_x <= current_x
    if crossed_line:
        counted_ids.add(track.track_id)
        count += 1
```

## Integration

Your detector integration layer should convert model outputs into
`FrameDetections` and call `tracker.update(...)` per frame. The returned
`Track` objects include box coordinates, score, lifecycle state, and path
history that can be rendered on top of the video.

This project is designed to pair cleanly with
[`rf-detr-mac`](https://github.com/Ak-Gautam/rf-detr-mac), but the tracking
API is intentionally model-agnostic so it can also sit behind other detector
stacks.

## Acknowledgements

Created and maintained by [Gautam](https://github.com/Ak-Gautam).
Additional updates and project notes are shared on
[X / Twitter](https://x.com/Gautam_A_k).
The tracker design is informed in part by
[roboflow/trackers](https://github.com/roboflow/trackers).

## License

Released under the Apache-2.0 license. See [`LICENSE`](./LICENSE).
