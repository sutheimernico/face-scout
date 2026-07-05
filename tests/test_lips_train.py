import numpy as np
from typer.testing import CliRunner

from face_scout.cli import app
from face_scout.lips.dataset import Sample, save_sample
from face_scout.lips.train import confusion_matrix, evaluate_split, train_model

P, T = 3, 10
_KIND = {"a": "x", "b": "y", "c": "const"}


def _utterance(kind, seed):
    rng = np.random.default_rng(seed)
    seq = np.zeros((T, P, 2))
    ramp = np.linspace(0, 1, T)
    if kind == "x":
        seq[:, :, 0] = ramp[:, None]
    elif kind == "y":
        seq[:, :, 1] = ramp[:, None]
    return seq + rng.normal(0, 0.01, seq.shape)


def _write_dataset(directory):
    i = 0
    for session in ("s0", "s1", "s2"):
        for label, kind in _KIND.items():
            for _ in range(2):
                seq = _utterance(kind, seed=hash((session, label, i)) % (2**32))
                save_sample(directory, Sample(label, session, seq), f"{label}_{session}_{i}")
                i += 1


def test_confusion_matrix_counts():
    m = confusion_matrix(["a", "a", "b"], ["a", "b", "b"], ["a", "b"])
    assert m.tolist() == [[1, 1], [0, 1]]


def test_train_model_saves_and_learns(tmp_path):
    data = tmp_path / "ds"
    _write_dataset(data)
    model_path = tmp_path / "lips.joblib"
    clf = train_model(data, model_path, length=8)
    assert model_path.exists()
    assert clf.classes_ == ["a", "b", "c"]


def test_evaluate_split_on_held_out_session(tmp_path):
    data = tmp_path / "ds"
    _write_dataset(data)
    report = evaluate_split(data, val_sessions=["s2"], length=8)
    assert report.n == 6  # 3 labels x 2 samples in s2
    assert report.accuracy >= 0.8  # synthetic classes are cleanly separable
    assert report.confusion.shape == (3, 3)


def test_cli_lips_train_smoke(tmp_path):
    data = tmp_path / "ds"
    _write_dataset(data)
    model_path = tmp_path / "lips.joblib"
    result = CliRunner().invoke(
        app, ["lips", "train", "--data", str(data), "--out", str(model_path), "--length", "8"]
    )
    assert result.exit_code == 0, result.output
    assert model_path.exists()
