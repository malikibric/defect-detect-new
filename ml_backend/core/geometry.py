"""Geometry helpers shared across ML backend services."""

from __future__ import annotations

from typing import List


BoundingBoxLike = List[float]


def calculate_iou(box1: BoundingBoxLike, box2: BoundingBoxLike) -> float:
    """Calculate Intersection over Union (IoU) between two boxes."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    union_area = (w1 * h1) + (w2 * h2) - intersection_area
    return intersection_area / union_area if union_area > 0 else 0.0


def calculate_bbox_similarity(bbox1: BoundingBoxLike, bbox2: BoundingBoxLike) -> float:
    """Combine IoU and relative area similarity into one score."""
    iou = calculate_iou(bbox1, bbox2)

    area1 = bbox1[2] * bbox1[3]
    area2 = bbox2[2] * bbox2[3]
    max_area = max(area1, area2)
    size_ratio = min(area1, area2) / max_area if max_area > 0 else 0.0

    return 0.7 * iou + 0.3 * size_ratio
