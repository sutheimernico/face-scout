"""Live loop: thin I/O around the Pipeline (capture, render, display, FPS)."""

from __future__ import annotations

import time

import cv2

from . import renderer
from .capture import WebcamCapture
from .config import Config
from .pipeline import build_pipeline

_WINDOW = "face-scout"


def run(config: Config | None = None) -> None:
    config = config or Config()

    source = WebcamCapture(config.camera_index)
    pipeline = build_pipeline(config)
    if not pipeline.has_recognition:
        print("Gallery is empty — running tracking only. Enroll a face to get identity.")

    fps = 0.0
    last = time.monotonic()
    try:
        while True:
            frame = source.read()
            if frame is None:
                break

            tracks = pipeline.process(frame)
            renderer.draw_tracks(frame, tracks, min_hits=config.min_hits)

            now = time.monotonic()
            dt = now - last
            last = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)  # exponential smoothing
            renderer.draw_fps(frame, fps)

            cv2.imshow(_WINDOW, frame)
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):  # q or Esc
                break
    finally:
        source.release()
        pipeline.close()
        cv2.destroyAllWindows()
