"""Per-frame processing pipeline: landmarks -> tracks -> throttled identity.

This is the wiring that used to live inside the live loop, pulled out so it can be
driven headlessly and unit-tested with fake adapters. It performs no I/O: the
caller supplies frames and handles capture, rendering, and display.
"""

from __future__ import annotations

import numpy as np

from .config import Config
from .gallery import Gallery
from .recognizer import assign_identities
from .tracker import Tracker
from .types import Embedder, Landmarker, Track


class Pipeline:
    def __init__(
        self,
        landmarker: Landmarker,
        tracker: Tracker,
        gallery: Gallery,
        config: Config,
        embedder: Embedder | None = None,
    ) -> None:
        self._landmarker = landmarker
        self._tracker = tracker
        self._gallery = gallery
        self._config = config
        self._embedder = embedder
        self._frame_idx = 0

    @property
    def frame_index(self) -> int:
        return self._frame_idx

    @property
    def has_recognition(self) -> bool:
        return self._embedder is not None

    def process(self, frame: np.ndarray) -> list[Track]:
        """Advance one frame and return the current tracks."""
        observations = self._landmarker.detect(frame)
        tracks = self._tracker.update(observations)
        # Throttle identity: it does not change frame-to-frame, so skip most frames.
        if self._embedder is not None and self._frame_idx % self._config.recognize_every == 0:
            embeddings = self._embedder.embed(frame)
            assign_identities(
                tracks,
                embeddings,
                self._gallery,
                self._config.sim_threshold,
                self._config.iou_threshold,
            )
        self._frame_idx += 1
        return tracks

    def close(self) -> None:
        close = getattr(self._landmarker, "close", None)
        if callable(close):
            close()


def build_pipeline(config: Config) -> Pipeline:
    """Assemble a pipeline with the real hardware adapters (needs a camera/GPU)."""
    # Imported lazily so this module stays importable (and testable) without the
    # heavy CV libraries installed.
    from .embedder import InsightFaceEmbedder
    from .landmarker import MediaPipeLandmarker

    landmarker = MediaPipeLandmarker(config.landmarker_model, num_faces=config.max_faces)
    tracker = Tracker(iou_threshold=config.iou_threshold, max_age=config.max_age)
    gallery = Gallery.load(config.gallery_path) if config.gallery_path.exists() else Gallery()
    embedder = InsightFaceEmbedder(config.insight_model, config.det_size) if len(gallery) else None
    return Pipeline(landmarker, tracker, gallery, config, embedder)
