"""Train and evaluate the lip-reading classifier from an on-disk dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .dataset import Sample, labels, load_dataset, session_split
from .model import LipClassifier


@dataclass(frozen=True)
class EvalReport:
    accuracy: float
    labels: list[str]  # row/column order of the confusion matrix
    confusion: np.ndarray  # (L, L) int, rows = true, cols = predicted
    n: int


def train_model(
    dataset_dir: str | Path,
    out_path: str | Path,
    length: int = 32,
    velocity_threshold: float = 0.0,
) -> LipClassifier:
    """Train on every sample in the directory and save the model."""
    samples = load_dataset(dataset_dir)
    if not samples:
        raise ValueError(f"no samples found in {dataset_dir}")
    clf = LipClassifier.default(length, velocity_threshold).fit(samples)
    clf.save(out_path)
    return clf


def confusion_matrix(true: list[str], pred: list[str], order: list[str]) -> np.ndarray:
    index = {label: i for i, label in enumerate(order)}
    matrix = np.zeros((len(order), len(order)), dtype=int)
    for t, p in zip(true, pred, strict=True):
        matrix[index[t], index[p]] += 1
    return matrix


def evaluate(model: LipClassifier, samples: list[Sample]) -> EvalReport:
    if not samples:
        raise ValueError("no samples to evaluate")
    order = sorted(set(labels(samples)) | set(model.classes_))
    true = [s.label for s in samples]
    pred = [model.predict(s.sequence) for s in samples]
    accuracy = float(np.mean([t == p for t, p in zip(true, pred, strict=True)]))
    return EvalReport(accuracy, order, confusion_matrix(true, pred, order), len(samples))


def evaluate_split(
    dataset_dir: str | Path,
    val_sessions: list[str],
    length: int = 32,
    velocity_threshold: float = 0.0,
) -> EvalReport:
    """Train on the train sessions, evaluate on the held-out val sessions."""
    train, val = session_split(load_dataset(dataset_dir), val_sessions)
    if not train or not val:
        raise ValueError("need non-empty train and val splits (check the val sessions)")
    clf = LipClassifier.default(length, velocity_threshold).fit(train)
    return evaluate(clf, val)
