"""Associates identity embeddings to live tracks.

Pure logic (numpy only): the embeddings come from an ``Embedder`` adapter, the
matching comes from the ``Gallery``. Identity is written back onto the tracks and
is *sticky* — a track keeps its label between throttled recognition passes.
"""

from __future__ import annotations

import numpy as np

from .gallery import Gallery
from .geometry import iou
from .types import BBox, Track


def assign_identities(
    tracks: list[Track],
    embeddings: list[tuple[BBox, np.ndarray]],
    gallery: Gallery,
    sim_threshold: float,
    iou_threshold: float = 0.3,
) -> None:
    """Attach each embedding's gallery match to its best-overlapping track.

    Each embedding is bound to the track with the highest IoU above
    ``iou_threshold``; that track's identity/score are updated in place. Tracks
    with no overlapping embedding this pass keep their previous identity.
    """
    for bbox, emb in embeddings:
        best_ti, best_iou = -1, iou_threshold
        for ti, track in enumerate(tracks):
            score = iou(track.bbox, bbox)
            if score >= best_iou:
                best_ti, best_iou = ti, score
        if best_ti < 0:
            continue
        match = gallery.match(emb, sim_threshold)
        tracks[best_ti].identity = match.name
        tracks[best_ti].identity_score = match.score
