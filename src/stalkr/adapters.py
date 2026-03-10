from __future__ import annotations

from typing import Any

from .types import Detection, FrameDetections


class RFDETRDetectionAdapter:
    """Convert RF-DETR supervision detections into stalkr frame detections."""

    def parse(
        self,
        raw_output: Any,
        *,
        frame_index: int | None = None,
        timestamp: float | None = None,
    ) -> FrameDetections:
        try:
            boxes = raw_output.xyxy
            scores = raw_output.confidence
            class_ids = raw_output.class_id
        except AttributeError as exc:
            raise TypeError(
                "RFDETRDetectionAdapter expects a supervision Detections-like object "
                "with xyxy, confidence, and class_id attributes."
            ) from exc

        if class_ids is None:
            class_ids = [None] * len(boxes)

        items = [
            Detection(
                box=tuple(float(value) for value in box),
                score=float(score),
                class_id=None if class_id is None else int(class_id),
            )
            for box, score, class_id in zip(boxes, scores, class_ids)
        ]

        return FrameDetections(
            items=items,
            frame_index=frame_index,
            timestamp=timestamp,
        )