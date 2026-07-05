"""Per-frame lip-landmark normalization from the full face mesh.

Uses speech-invariant references so the normalization does not cancel the very
motion we want to read: translation by the nose tip, scale by the inter-ocular
distance. The mouth cannot be the reference because it moves while speaking.
"""

from __future__ import annotations

import numpy as np

from ..geometry import LIPS_IDX

# Stable MediaPipe FaceMesh reference indices.
NOSE_TIP = 1  # translation origin
EYE_OUTER_LEFT = 33  # inter-ocular scale endpoints
EYE_OUTER_RIGHT = 263


def normalize_lips(mesh: np.ndarray) -> np.ndarray:
    """Return translation- and scale-normalized lip landmarks, shape (len(LIPS_IDX), 2).

    ``mesh`` is a full MediaPipe face mesh (>=468 points, (N, 2) or (N, 3)); only
    x/y are used. Raises ``ValueError`` on a degenerate mesh (zero eye distance).
    """
    pts = mesh[:, :2].astype(np.float64)
    origin = pts[NOSE_TIP]
    scale = float(np.linalg.norm(pts[EYE_OUTER_LEFT] - pts[EYE_OUTER_RIGHT]))
    if scale <= 0:
        raise ValueError("degenerate mesh: zero inter-ocular distance")
    lips = pts[list(LIPS_IDX)]
    return (lips - origin) / scale
