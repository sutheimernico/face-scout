"""InsightFace identity embedder (implements the Embedder protocol).

Runs InsightFace's own detector + ArcFace on a frame and returns one
L2-normalized embedding per face. CPU by default; the caller throttles how often
this runs, since identity does not change frame-to-frame.
"""

from __future__ import annotations

import numpy as np
from insightface.app import FaceAnalysis

from .types import BBox


class InsightFaceEmbedder:
    def __init__(
        self, model_name: str = "buffalo_l", det_size: tuple[int, int] = (640, 640)
    ) -> None:
        self._app = FaceAnalysis(name=model_name, providers=["CPUExecutionProvider"])
        self._app.prepare(ctx_id=0, det_size=det_size)

    def embed(self, frame: np.ndarray) -> list[tuple[BBox, np.ndarray]]:
        out: list[tuple[BBox, np.ndarray]] = []
        for face in self._app.get(frame):  # frame is BGR, as InsightFace expects
            x1, y1, x2, y2 = face.bbox
            emb = np.asarray(face.normed_embedding, dtype=np.float64)  # already unit-length
            out.append(((float(x1), float(y1), float(x2), float(y2)), emb))
        return out
