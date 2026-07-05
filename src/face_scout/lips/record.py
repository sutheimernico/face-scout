"""Capture a labeled lip utterance from a frame source.

The testable core (``meshes_to_sequence``, ``record_utterance``, ``save_utterance``)
works on any finite ``FrameSource`` + ``Landmarker`` — a video clip or a test fake.
The live interactive webcam recorder (start/stop overlay) is the camera-bound
Needs-Nico piece and stays out of here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..geometry import LIPS_IDX
from ..types import FrameSource, Landmarker
from .dataset import Sample, save_sample
from .normalize import normalize_lips


def meshes_to_sequence(meshes: list[np.ndarray]) -> np.ndarray:
    """Normalize a list of full-mesh frames into a (T, P, 2) lip sequence."""
    if not meshes:
        return np.zeros((0, len(LIPS_IDX), 2))
    return np.stack([normalize_lips(m) for m in meshes])


def record_utterance(
    source: FrameSource, landmarker: Landmarker, min_frames: int = 4
) -> np.ndarray:
    """Read a finite source to its end and return the normalized lip sequence.

    Frames with no detected face are skipped; the first face is used (single
    speaker). Raises ``ValueError`` if fewer than ``min_frames`` faces were seen.
    """
    meshes: list[np.ndarray] = []
    while True:
        frame = source.read()
        if frame is None:
            break
        faces = landmarker.detect(frame)
        if faces:
            meshes.append(faces[0].landmarks)

    sequence = meshes_to_sequence(meshes)
    if sequence.shape[0] < min_frames:
        raise ValueError(f"utterance too short: {sequence.shape[0]} frames (< {min_frames})")
    return sequence


def save_utterance(
    directory: str | Path, label: str, session_id: str, sequence: np.ndarray, index: int
) -> Path:
    """Persist one captured utterance as a dataset sample."""
    name = f"{label}_{session_id}_{index:03d}"
    return save_sample(directory, Sample(label, session_id, sequence), name)
