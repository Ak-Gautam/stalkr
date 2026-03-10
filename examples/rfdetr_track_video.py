from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RFDETR_REPO = REPO_ROOT.parent / "rf-detr-mac"

if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(RFDETR_REPO / "src") not in sys.path:
    sys.path.insert(0, str(RFDETR_REPO / "src"))

import cv2
import numpy as np
from rfdetr import RFDETRLarge

from stalkr import RFDETRDetectionAdapter, TrackerConfig, TrackingPipeline
from stalkr.tracker import LightweightTracker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RF-DETR Large + stalkr on a video and save an annotated output."
    )
    parser.add_argument("--input", type=Path, required=True, help="Path to the input video.")
    parser.add_argument("--output", type=Path, required=True, help="Path to the output annotated video.")
    parser.add_argument(
        "--weights",
        type=Path,
        default=RFDETR_REPO / "rf-detr-large-2026.pth",
        help="Path to the RF-DETR Large checkpoint.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.35,
        help="Detection confidence threshold.",
    )
    parser.add_argument(
        "--device",
        choices=("mps", "cpu"),
        default="mps",
        help="Inference device.",
    )
    parser.add_argument(
        "--max-age",
        type=int,
        default=30,
        help="How many missed frames a track is kept alive.",
    )
    parser.add_argument(
        "--history-size",
        type=int,
        default=32,
        help="How many center points to keep per track trail.",
    )
    return parser.parse_args()


def color_for_track(track_id: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(track_id)
    color = rng.integers(64, 256, size=3)
    return int(color[0]), int(color[1]), int(color[2])


def draw_track(frame: np.ndarray, track, class_name: str | None) -> np.ndarray:
    x1, y1, x2, y2 = (int(value) for value in track.box)
    color = color_for_track(track.track_id)
    label = f"id={track.track_id}"
    if class_name:
        label = f"{class_name} {label}"
    label = f"{label} {track.score:.2f}"

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        frame,
        label,
        (x1, max(18, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )

    if len(track.path) > 1:
        for start, end in zip(track.path, track.path[1:]):
            cv2.line(
                frame,
                (int(start[0]), int(start[1])),
                (int(end[0]), int(end[1])),
                color,
                2,
                cv2.LINE_AA,
            )
    return frame


def main() -> int:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input video not found: {args.input}")
    if not RFDETR_REPO.exists():
        raise FileNotFoundError(f"rf-detr-mac repo not found beside stalkr: {RFDETR_REPO}")
    if not args.weights.exists():
        raise FileNotFoundError(f"RF-DETR Large weights not found: {args.weights}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    model = RFDETRLarge(device=args.device, pretrain_weights=str(args.weights))
    pipeline = TrackingPipeline(
        detector=model,
        adapter=RFDETRDetectionAdapter(),
        tracker=LightweightTracker(
            TrackerConfig(
                max_age=args.max_age,
                history_size=args.history_size,
                min_hits=1,
            )
        ),
    )
    class_names = model.class_names

    capture = cv2.VideoCapture(str(args.input))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open input video: {args.input}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open output video for writing: {args.output}")

    frame_index = 0
    try:
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            tracks = pipeline.process(
                frame_rgb,
                frame_index=frame_index,
                timestamp=frame_index / fps,
            )

            annotated = frame_bgr.copy()
            for track in tracks:
                class_name = class_names.get(track.class_id) if track.class_id is not None else None
                annotated = draw_track(annotated, track, class_name)

            writer.write(annotated)
            frame_index += 1
    finally:
        capture.release()
        writer.release()

    print(f"Saved annotated video to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())