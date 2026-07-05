"""Core data structures and the interfaces the hardware adapters implement.

Keeping these here (numpy-only) lets the whole pure-logic core be imported and
tested without OpenCV, MediaPipe, or InsightFace installed. Adapters implement the
Protocols; tests inject fakes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

# Axis-aligned bounding box in pixel coordinates: (x1, y1, x2, y2).
BBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class FaceObservation:
    """One face detected in one frame."""

    bbox: BBox
    landmarks: np.ndarray  # (N, 2) or (N, 3) pixel coordinates


@dataclass
class Track:
    """A face followed across frames under a stable id."""

    track_id: int
    bbox: BBox
    landmarks: np.ndarray
    age: int = 0  # frames since last matched (0 == matched this frame)
    hits: int = 0  # total frames this track has been matched
    identity: str = "Unknown"
    identity_score: float = 0.0


class FrameSource(Protocol):
    """A source of BGR frames (webcam, video file, or a test fake)."""

    def read(self) -> np.ndarray | None:
        """Return the next frame, or None when the source is exhausted."""
        ...

    def release(self) -> None: ...


class Landmarker(Protocol):
    """Detects faces and their dense mesh landmarks in a frame."""

    def detect(self, frame: np.ndarray) -> list[FaceObservation]: ...


class Embedder(Protocol):
    """Produces an identity embedding per face detected in a frame."""

    def embed(self, frame: np.ndarray) -> list[tuple[BBox, np.ndarray]]:
        """Return (bbox, L2-normalized embedding) for each detected face."""
        ...
