"""MediaPipe FaceLandmarker adapter (implements the Landmarker protocol).

Uses the modern Tasks API in VIDEO mode, which tracks landmarks frame-to-frame.
The 468-point mesh (incl. dense lips) is the substrate for Phase 2 (lip reading).
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from .geometry import bbox_from_landmarks
from .types import FaceObservation

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)


def ensure_model(path: Path) -> Path:
    """Download the FaceLandmarker task bundle to ``path`` if it is missing."""
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, path)  # noqa: S310 (fixed https URL)
    return path


class MediaPipeLandmarker:
    """Per-frame face mesh detection returning pixel-space FaceObservations."""

    # VIDEO mode needs strictly increasing timestamps; we synthesize ~30 fps.
    _FRAME_MS = 33

    def __init__(self, model_path: str | Path, num_faces: int = 4) -> None:
        ensure_model(Path(model_path))
        options = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=num_faces,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        self._t = 0

    def detect(self, frame: np.ndarray) -> list[FaceObservation]:
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(mp_image, self._t)
        self._t += self._FRAME_MS

        observations: list[FaceObservation] = []
        for face in result.face_landmarks:
            # MediaPipe returns normalized coords; z shares x's scale by convention.
            pts = np.array([[lm.x * w, lm.y * h, lm.z * w] for lm in face], dtype=np.float32)
            observations.append(FaceObservation(bbox=bbox_from_landmarks(pts), landmarks=pts))
        return observations

    def close(self) -> None:
        self._landmarker.close()
