"""OpenCV webcam frame source (implements the FrameSource protocol)."""

from __future__ import annotations

import cv2
import numpy as np


class WebcamCapture:
    """Reads BGR frames from a camera device by index."""

    def __init__(self, camera_index: int = 0) -> None:
        self._cap = cv2.VideoCapture(camera_index)
        if not self._cap.isOpened():
            raise RuntimeError(f"cannot open camera index {camera_index}")

    def read(self) -> np.ndarray | None:
        ok, frame = self._cap.read()
        return frame if ok else None

    def release(self) -> None:
        self._cap.release()
