"""Main live loop: capture -> landmarks -> track -> (throttled) identity -> render."""

from __future__ import annotations

import time

import cv2

from . import renderer
from .capture import WebcamCapture
from .config import Config
from .embedder import InsightFaceEmbedder
from .gallery import Gallery
from .landmarker import MediaPipeLandmarker
from .recognizer import assign_identities
from .tracker import Tracker

_WINDOW = "face-scout"


def run(config: Config | None = None) -> None:
    config = config or Config()

    source = WebcamCapture(config.camera_index)
    landmarker = MediaPipeLandmarker(config.landmarker_model, num_faces=config.max_faces)
    tracker = Tracker(iou_threshold=config.iou_threshold, max_age=config.max_age)

    gallery = Gallery.load(config.gallery_path) if config.gallery_path.exists() else Gallery()
    embedder = InsightFaceEmbedder(config.insight_model, config.det_size) if len(gallery) else None
    if embedder is None:
        print("Gallery is empty — running tracking only. Enroll a face to get identity.")

    frame_idx = 0
    fps = 0.0
    last = time.monotonic()
    try:
        while True:
            frame = source.read()
            if frame is None:
                break

            observations = landmarker.detect(frame)
            tracks = tracker.update(observations)

            if embedder is not None and frame_idx % config.recognize_every == 0:
                embeddings = embedder.embed(frame)
                assign_identities(
                    tracks, embeddings, gallery, config.sim_threshold, config.iou_threshold
                )

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
            frame_idx += 1
    finally:
        source.release()
        landmarker.close()
        cv2.destroyAllWindows()
