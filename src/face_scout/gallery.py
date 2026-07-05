"""Enrollment gallery: stores one mean embedding per person and matches by cosine.

Embeddings are biometric data — the on-disk gallery lives under ``gallery/`` which
is git-ignored and never committed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


def l2_normalize(v: np.ndarray) -> np.ndarray:
    """Return the unit-length version of a vector (unchanged if it is zero)."""
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


@dataclass(frozen=True)
class Match:
    name: str
    score: float


class Gallery:
    """A name -> L2-normalized mean-embedding store with cosine matching."""

    def __init__(self) -> None:
        self._embeddings: dict[str, np.ndarray] = {}

    def enroll(self, name: str, embeddings: list[np.ndarray]) -> None:
        """Register a person from one or more embeddings (mean, then normalize)."""
        if not embeddings:
            raise ValueError("need at least one embedding to enroll")
        stack = np.stack([l2_normalize(e.astype(np.float64)) for e in embeddings])
        self._embeddings[name] = l2_normalize(stack.mean(axis=0))

    def names(self) -> list[str]:
        return list(self._embeddings)

    def __len__(self) -> int:
        return len(self._embeddings)

    def match(self, embedding: np.ndarray, threshold: float) -> Match:
        """Best cosine match; ``Unknown`` if the top score is below ``threshold``."""
        if not self._embeddings:
            return Match("Unknown", 0.0)
        q = l2_normalize(embedding.astype(np.float64))
        best_name, best_score = "Unknown", -1.0
        for name, ref in self._embeddings.items():
            score = float(np.dot(q, ref))  # cosine, both operands unit-length
            if score > best_score:
                best_name, best_score = name, score
        return Match(best_name if best_score >= threshold else "Unknown", best_score)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        names = list(self._embeddings)
        matrix = np.stack([self._embeddings[n] for n in names]) if names else np.zeros((0, 0))
        np.savez(path, names=np.array(names, dtype=object), matrix=matrix)

    @classmethod
    def load(cls, path: str | Path) -> Gallery:
        g = cls()
        data = np.load(Path(path), allow_pickle=True)  # local, trusted gallery file
        names = list(data["names"])
        matrix = data["matrix"]
        for i, name in enumerate(names):
            g._embeddings[str(name)] = matrix[i]
        return g
