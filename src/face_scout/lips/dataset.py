"""Labeled lip-utterance samples: on-disk store and session-aware splitting.

Each sample is one recorded utterance (a normalized lip-landmark sequence) tagged
with a label and a ``session_id``. Splits hold out whole sessions so no take or
frame leaks across train/val (a frame-level split inflates accuracy — see ADR 0001).

Recorded lip geometry is biometric-adjacent and stays local: the dataset dir is
git-ignored, never committed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Sample:
    label: str
    session_id: str
    sequence: np.ndarray  # (T, P, 2) normalized lip frames


def save_sample(directory: str | Path, sample: Sample, name: str) -> Path:
    """Write one sample to ``directory/name.npz`` and return the path."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.npz"
    np.savez(
        path,
        sequence=sample.sequence.astype(np.float32),
        label=sample.label,
        session_id=sample.session_id,
    )
    return path


def load_dataset(directory: str | Path) -> list[Sample]:
    """Load all samples from a directory, ordered by filename."""
    directory = Path(directory)
    samples: list[Sample] = []
    for path in sorted(directory.glob("*.npz")):
        data = np.load(path, allow_pickle=False)
        samples.append(
            Sample(
                label=str(data["label"]),
                session_id=str(data["session_id"]),
                sequence=data["sequence"],
            )
        )
    return samples


def labels(samples: list[Sample]) -> list[str]:
    return sorted({s.label for s in samples})


def sessions(samples: list[Sample]) -> list[str]:
    return sorted({s.session_id for s in samples})


def session_split(
    samples: list[Sample], val_sessions: list[str]
) -> tuple[list[Sample], list[Sample]]:
    """Split by session: samples whose ``session_id`` is in ``val_sessions`` go to val."""
    val_set = set(val_sessions)
    train = [s for s in samples if s.session_id not in val_set]
    val = [s for s in samples if s.session_id in val_set]
    return train, val
