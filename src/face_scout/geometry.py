"""Pure geometry helpers: bounding boxes, IoU, and lip-region extraction."""

from __future__ import annotations

import numpy as np

from .types import BBox

# MediaPipe FaceMesh lip landmark indices (outer then inner contour). These feed
# Phase 2 (lip reading); kept here so the mesh->lips mapping lives in one place.
LIPS_OUTER = (
    61,
    146,
    91,
    181,
    84,
    17,
    314,
    405,
    321,
    375,
    291,
    185,
    40,
    39,
    37,
    0,
    267,
    269,
    270,
    409,
)
LIPS_INNER = (
    78,
    95,
    88,
    178,
    87,
    14,
    317,
    402,
    318,
    324,
    308,
    191,
    80,
    81,
    82,
    13,
    312,
    311,
    310,
    415,
)
LIPS_IDX = LIPS_OUTER + LIPS_INNER


def bbox_from_landmarks(landmarks: np.ndarray) -> BBox:
    """Tight axis-aligned box around a set of (x, y[, z]) landmarks."""
    xs = landmarks[:, 0]
    ys = landmarks[:, 1]
    return (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))


def iou(a: BBox, b: BBox) -> float:
    """Intersection-over-union of two boxes; 0.0 if they do not overlap."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def centroid(b: BBox) -> tuple[float, float]:
    """Center point of a box."""
    x1, y1, x2, y2 = b
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def lip_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """Subset of a full (>=468-point) face mesh covering the lips."""
    return landmarks[list(LIPS_IDX)]
