import numpy as np

from face_scout.lips.dataset import (
    Sample,
    labels,
    load_dataset,
    save_sample,
    session_split,
    sessions,
)


def _sample(label, session, t=4, p=3):
    rng = np.random.default_rng(abs(hash((label, session))) % (2**32))
    return Sample(label=label, session_id=session, sequence=rng.uniform(size=(t, p, 2)))


def test_save_load_roundtrip(tmp_path):
    s = _sample("hello", "s1", t=5, p=3)
    save_sample(tmp_path, s, "hello_001")
    loaded = load_dataset(tmp_path)
    assert len(loaded) == 1
    assert loaded[0].label == "hello"
    assert loaded[0].session_id == "s1"
    assert loaded[0].sequence.shape == (5, 3, 2)
    assert np.allclose(loaded[0].sequence, s.sequence, atol=1e-6)


def test_load_dataset_orders_by_filename(tmp_path):
    save_sample(tmp_path, _sample("b", "s1"), "002")
    save_sample(tmp_path, _sample("a", "s1"), "001")
    assert [s.label for s in load_dataset(tmp_path)] == ["a", "b"]


def test_labels_and_sessions_unique_sorted():
    samples = [_sample("yes", "s1"), _sample("no", "s2"), _sample("yes", "s2")]
    assert labels(samples) == ["no", "yes"]
    assert sessions(samples) == ["s1", "s2"]


def test_session_split_holds_out_whole_sessions():
    samples = [
        _sample("yes", "A"),
        _sample("no", "A"),
        _sample("yes", "B"),
        _sample("no", "C"),
    ]
    train, val = session_split(samples, val_sessions=["B"])
    assert {s.session_id for s in val} == {"B"}
    assert {s.session_id for s in train} == {"A", "C"}
    # No session appears on both sides -> no leakage.
    assert not ({s.session_id for s in train} & {s.session_id for s in val})
