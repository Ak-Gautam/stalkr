from __future__ import annotations

from math import sqrt

from .types import Box


def clamp_box(box: Box) -> Box:
    x1, y1, x2, y2 = box
    left = min(x1, x2)
    top = min(y1, y2)
    right = max(x1, x2)
    bottom = max(y1, y2)
    return (left, top, right, bottom)


def box_center(box: Box) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def iou(box_a: Box, box_b: Box) -> float:
    ax1, ay1, ax2, ay2 = clamp_box(box_a)
    bx1, by1, bx2, by2 = clamp_box(box_b)

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union


def cosine_similarity(
    embedding_a: tuple[float, ...] | None,
    embedding_b: tuple[float, ...] | None,
) -> float | None:
    if embedding_a is None or embedding_b is None:
        return None
    if len(embedding_a) != len(embedding_b) or not embedding_a:
        return None

    dot = sum(a * b for a, b in zip(embedding_a, embedding_b, strict=False))
    mag_a = sqrt(sum(a * a for a in embedding_a))
    mag_b = sqrt(sum(b * b for b in embedding_b))
    if mag_a == 0.0 or mag_b == 0.0:
        return None
    return dot / (mag_a * mag_b)
