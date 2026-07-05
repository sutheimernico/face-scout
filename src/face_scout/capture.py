"""OpenCV frame sources (implement the FrameSource protocol)."""

from __future__ import annotations

from pathlib import Path

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


class VideoFileSource:
    """Reads BGR frames from a video file — lets the pipeline run without a camera."""

    def __init__(self, path: str | Path) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"video file not found: {path}")
        self._cap = cv2.VideoCapture(str(path))
        if not self._cap.isOpened():
            raise RuntimeError(f"cannot open video file: {path}")

    def read(self) -> np.ndarray | None:
        ok, frame = self._cap.read()
        return frame if ok else None  # None at end of file

    def release(self) -> None:
        self._cap.release()
