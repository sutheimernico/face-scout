"""Turn a variable-length lip-landmark sequence into a fixed-size feature vector.

An utterance is a sequence of normalized lip frames, shape (T, P, 2). We trim
low-motion (silent) ends, resample the active span to a fixed length by linear
interpolation, and flatten. The temporal-NN upgrade (deferred, see ADR 0001) will
pad+mask instead of resampling.
"""

from __future__ import annotations

import numpy as np


def frame_velocities(sequence: np.ndarray) -> np.ndarray:
    """Per-frame mean landmark speed; frame 0 is 0. Shape (T,)."""
    if sequence.shape[0] == 0:
        return np.zeros((0,))
    diffs = np.diff(sequence, axis=0)  # (T-1, P, 2)
    mag = np.linalg.norm(diffs, axis=2).mean(axis=1)  # (T-1,)
    return np.concatenate([[0.0], mag])


def trim_silence(sequence: np.ndarray, threshold: float) -> np.ndarray:
    """Drop leading/trailing frames below the velocity ``threshold``.

    Returns the active span, or an empty (0, P, 2) slice if nothing exceeds it.
    """
    if sequence.shape[0] == 0:
        return sequence
    active = np.where(frame_velocities(sequence) >= threshold)[0]
    if active.size == 0:
        return sequence[0:0]
    return sequence[active[0] : active[-1] + 1]


def resample(sequence: np.ndarray, length: int) -> np.ndarray:
    """Linearly resample a (T, P, 2) sequence to (length, P, 2) over time."""
    t = sequence.shape[0]
    if t == 0:
        raise ValueError("cannot resample an empty sequence")
    if length <= 0:
        raise ValueError("length must be positive")
    seq = sequence.astype(np.float64)
    if t == length:
        return seq
    src = np.linspace(0.0, t - 1, num=t)
    dst = np.linspace(0.0, t - 1, num=length)
    flat = seq.reshape(t, -1)  # (T, P*2)
    out = np.empty((length, flat.shape[1]))
    for c in range(flat.shape[1]):
        out[:, c] = np.interp(dst, src, flat[:, c])
    return out.reshape(length, sequence.shape[1], sequence.shape[2])


def to_feature_vector(
    sequence: np.ndarray, length: int, velocity_threshold: float = 0.0
) -> np.ndarray:
    """Trim (if a threshold is given), resample to ``length``, flatten to 1-D."""
    trimmed = trim_silence(sequence, velocity_threshold) if velocity_threshold > 0 else sequence
    if trimmed.shape[0] == 0:
        raise ValueError("sequence is empty after silence trimming")
    return resample(trimmed, length).reshape(-1)
