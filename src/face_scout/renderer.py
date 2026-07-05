"""Draws the tracking/identity overlay onto a BGR frame."""

from __future__ import annotations

import cv2
import numpy as np

from .types import Track

_GREEN = (0, 255, 0)
_RED = (0, 0, 255)
_WHITE = (255, 255, 255)
_YELLOW = (0, 255, 255)
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_tracks(
    frame: np.ndarray,
    tracks: list[Track],
    min_hits: int = 3,
    draw_mesh: bool = True,
) -> np.ndarray:
    """Overlay boxes, mesh points, track ids, and identity labels in place."""
    for t in tracks:
        if t.hits < min_hits:
            continue
        x1, y1, x2, y2 = (int(v) for v in t.bbox)
        known = t.identity != "Unknown"
        color = _GREEN if known else _RED
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"#{t.track_id} {t.identity}"
        if known:
            label += f" {t.identity_score:.2f}"
        cv2.putText(frame, label, (x1, max(12, y1 - 8)), _FONT, 0.6, color, 2)

        if draw_mesh and t.landmarks is not None and len(t.landmarks) > 2:
            for px, py in t.landmarks[:, :2].astype(int):
                cv2.circle(frame, (int(px), int(py)), 1, _WHITE, -1)
    return frame


def draw_fps(frame: np.ndarray, fps: float) -> np.ndarray:
    cv2.putText(frame, f"{fps:.1f} FPS", (10, 25), _FONT, 0.7, _YELLOW, 2)
    return frame
