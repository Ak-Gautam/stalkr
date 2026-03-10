from __future__ import annotations

import argparse
import shutil
import subprocess
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
    parser.add_argument(
        "--no-transcode",
        action="store_true",
        help="Skip ffmpeg H.264 transcoding and keep the intermediate OpenCV MP4 output.",
    )
    return parser.parse_args()


def intermediate_output_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}.intermediate{output_path.suffix}")


def transcode_to_platform_mp4(source_path: Path, output_path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "ffmpeg is required for the default H.264 export path. "
            "Install ffmpeg or rerun with --no-transcode."
        )

    commands = [
        [
            ffmpeg,
            "-y",
            "-i",
            str(source_path),
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_path),
        ],
        [
            ffmpeg,
            "-y",
            "-i",
            str(source_path),
            "-an",
            "-c:v",
            "h264_videotoolbox",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_path),
        ],
    ]

    errors: list[str] = []
    for command in commands:
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode == 0:
            return
        errors.append(result.stderr.strip() or "ffmpeg failed without stderr output")

    raise RuntimeError("\n\n".join(errors))


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
    raw_output_path = intermediate_output_path(args.output) if not args.no_transcode else args.output
    if raw_output_path.exists():
        raw_output_path.unlink()
    if not args.no_transcode and args.output.exists():
        args.output.unlink()

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
        str(raw_output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open output video for writing: {raw_output_path}")

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

    if not args.no_transcode:
        transcode_to_platform_mp4(raw_output_path, args.output)
        raw_output_path.unlink(missing_ok=True)

    print(f"Saved annotated video to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())