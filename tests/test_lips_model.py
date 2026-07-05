import numpy as np
import pytest

from face_scout.lips.dataset import Sample
from face_scout.lips.model import LipClassifier, build_xy

P, T = 3, 10


def _utterance(kind, seed):
    rng = np.random.default_rng(seed)
    seq = np.zeros((T, P, 2))
    ramp = np.linspace(0, 1, T)
    if kind == "x":
        seq[:, :, 0] = ramp[:, None]
    elif kind == "y":
        seq[:, :, 1] = ramp[:, None]
    # "const" stays at zero
    return seq + rng.normal(0, 0.01, seq.shape)


def _samples(label, kind, n):
    return [
        Sample(label, f"s{i % 3}", _utterance(kind, seed=hash((label, i)) % (2**32)))
        for i in range(n)
    ]


def _training_set():
    return _samples("a", "x", 6) + _samples("b", "y", 6) + _samples("c", "const", 6)


def test_build_xy_shapes():
    x, y = build_xy(_training_set(), length=8)
    assert x.shape == (18, 8 * P * 2)
    assert len(y) == 18


def test_classifier_learns_separable_classes():
    clf = LipClassifier.default(length=8).fit(_training_set())
    assert clf.predict(_utterance("x", seed=999)) == "a"
    assert clf.predict(_utterance("y", seed=998)) == "b"
    assert clf.predict(_utterance("const", seed=997)) == "c"


def test_predict_proba_is_a_distribution():
    clf = LipClassifier.default(length=8).fit(_training_set())
    proba = clf.predict_proba(_utterance("x", seed=1))
    assert set(proba) == {"a", "b", "c"}
    assert abs(sum(proba.values()) - 1.0) < 1e-6


def test_fit_requires_two_labels():
    with pytest.raises(ValueError, match="two distinct labels"):
        LipClassifier.default(length=8).fit(_samples("a", "x", 4))


def test_save_load_roundtrip(tmp_path):
    clf = LipClassifier.default(length=8).fit(_training_set())
    path = tmp_path / "model.joblib"
    clf.save(path)
    loaded = LipClassifier.load(path)
    assert loaded.classes_ == clf.classes_
    assert loaded.predict(_utterance("y", seed=42)) == "b"
