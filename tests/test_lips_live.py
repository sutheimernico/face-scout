import numpy as np

from face_scout.lips.dataset import Sample
from face_scout.lips.live import predict_meshes
from face_scout.lips.model import LipClassifier
from face_scout.lips.normalize import EYE_OUTER_LEFT, EYE_OUTER_RIGHT

P_MESH, T = 468, 10


def _mesh(seed, kind):
    rng = np.random.default_rng(seed)
    mesh = rng.uniform(0, 100, size=(P_MESH, 3))
    mesh[EYE_OUTER_LEFT] = [10.0, 50.0, 0.0]
    mesh[EYE_OUTER_RIGHT] = [90.0, 50.0, 0.0]
    # Bias the lip region so "x" and "y" utterances are separable after normalization.
    if kind == "x":
        mesh[61] = [80.0, 50.0, 0.0]
    else:
        mesh[61] = [50.0, 80.0, 0.0]
    return mesh


def _utterance(kind, seed):
    return [_mesh(seed + t, kind) for t in range(T)]


def _trained_model():
    samples = []
    for i in range(6):
        from face_scout.lips.record import meshes_to_sequence

        samples.append(Sample("a", f"s{i % 3}", meshes_to_sequence(_utterance("x", 100 + i * 10))))
        samples.append(Sample("b", f"s{i % 3}", meshes_to_sequence(_utterance("y", 500 + i * 10))))
    return LipClassifier.default(length=8).fit(samples)


class FakeModel:
    def __init__(self, proba):
        self._proba = proba

    def predict_proba(self, sequence):
        return self._proba


def test_predict_empty_buffer_is_unknown():
    assert predict_meshes(FakeModel({"a": 1.0}), []) == ("?", 0.0)


def test_predict_below_confidence_is_unknown():
    label, conf = predict_meshes(
        FakeModel({"a": 0.7, "b": 0.3}), [_mesh(1, "x")], min_confidence=0.8
    )
    assert label == "?"
    assert conf == 0.7


def test_predict_above_confidence_returns_label():
    label, conf = predict_meshes(
        FakeModel({"a": 0.7, "b": 0.3}), [_mesh(1, "x")], min_confidence=0.5
    )
    assert label == "a"
    assert conf == 0.7


def test_predict_with_real_model():
    model = _trained_model()
    label, _ = predict_meshes(model, _utterance("x", seed=9000))
    assert label == "a"
