"""v1 lip-reading classifier: fixed-length landmark features -> label (sklearn).

A lightweight wrapper (ADR 0001): each utterance becomes a fixed-length feature
vector via ``sequence.to_feature_vector``, and a RandomForest classifies it.
RandomForest is the default because it gives native ``predict_proba``, needs no
feature scaling or internal CV, and behaves well on tiny tabular datasets. The
temporal-NN upgrade (PyTorch) is deferred.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .dataset import Sample
from .sequence import to_feature_vector


def build_xy(
    samples: list[Sample], length: int, velocity_threshold: float = 0.0
) -> tuple[np.ndarray, list[str]]:
    """Feature matrix (N, length*P*2) and label list from samples."""
    features = [to_feature_vector(s.sequence, length, velocity_threshold) for s in samples]
    labels = [s.label for s in samples]
    return np.array(features), labels


class LipClassifier:
    def __init__(self, length: int = 32, velocity_threshold: float = 0.0, model=None) -> None:
        self.length = length
        self.velocity_threshold = velocity_threshold
        self._model = model
        self.classes_: list[str] = []

    @classmethod
    def default(cls, length: int = 32, velocity_threshold: float = 0.0) -> LipClassifier:
        from sklearn.ensemble import RandomForestClassifier

        return cls(
            length, velocity_threshold, RandomForestClassifier(n_estimators=200, random_state=0)
        )

    def fit(self, samples: list[Sample]) -> LipClassifier:
        x, y = build_xy(samples, self.length, self.velocity_threshold)
        if len(set(y)) < 2:
            raise ValueError("need at least two distinct labels to train")
        self._model.fit(x, y)
        self.classes_ = [str(c) for c in self._model.classes_]
        return self

    def _features(self, sequence: np.ndarray) -> np.ndarray:
        return to_feature_vector(sequence, self.length, self.velocity_threshold).reshape(1, -1)

    def predict(self, sequence: np.ndarray) -> str:
        return str(self._model.predict(self._features(sequence))[0])

    def predict_proba(self, sequence: np.ndarray) -> dict[str, float]:
        proba = self._model.predict_proba(self._features(sequence))[0]
        return dict(zip(self.classes_, proba.tolist(), strict=True))

    def save(self, path: str | Path) -> None:
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "length": self.length,
                "velocity_threshold": self.velocity_threshold,
                "model": self._model,
                "classes": self.classes_,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> LipClassifier:
        import joblib

        data = joblib.load(Path(path))  # local, trusted model file
        obj = cls(data["length"], data["velocity_threshold"], data["model"])
        obj.classes_ = data["classes"]
        return obj
