"""Tunable parameters for the pipeline, in one place."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

MODELS_DIR = Path("models")
GALLERY_DIR = Path("gallery")


@dataclass(frozen=True)
class Config:
    camera_index: int = 0
    max_faces: int = 4

    # Tracking
    iou_threshold: float = 0.3
    max_age: int = 15
    min_hits: int = 3  # frames before a track is drawn / recognized

    # Identity
    sim_threshold: float = 0.35  # ArcFace cosine; below this -> "Unknown"
    recognize_every: int = 10  # run identity only every Nth frame (throttle)

    # Paths
    landmarker_model: Path = MODELS_DIR / "face_landmarker.task"
    gallery_path: Path = GALLERY_DIR / "gallery.npz"

    # Enrollment
    enroll_shots: int = 5

    # InsightFace
    insight_model: str = "buffalo_l"
    det_size: tuple[int, int] = field(default=(640, 640))
