import numpy as np

from face_scout.gallery import Gallery, l2_normalize


def test_l2_normalize_unit_length():
    assert abs(np.linalg.norm(l2_normalize(np.array([3.0, 4.0]))) - 1.0) < 1e-9


def test_l2_normalize_zero_vector_unchanged():
    v = np.zeros(3)
    assert np.allclose(l2_normalize(v), v)


def test_enroll_and_match_picks_closest():
    g = Gallery()
    g.enroll("nico", [np.array([1.0, 0.0, 0.0])])
    g.enroll("alex", [np.array([0.0, 1.0, 0.0])])
    m = g.match(np.array([0.9, 0.1, 0.0]), threshold=0.3)
    assert m.name == "nico"
    assert m.score > 0.3


def test_match_below_threshold_is_unknown():
    g = Gallery()
    g.enroll("nico", [np.array([1.0, 0.0, 0.0])])
    m = g.match(np.array([0.0, 0.0, 1.0]), threshold=0.3)  # orthogonal -> cos 0
    assert m.name == "Unknown"


def test_match_empty_gallery_is_unknown():
    g = Gallery()
    assert g.match(np.array([1.0, 0.0]), threshold=0.3).name == "Unknown"


def test_mean_embedding_is_normalized():
    g = Gallery()
    g.enroll("nico", [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])])
    assert abs(np.linalg.norm(g._embeddings["nico"]) - 1.0) < 1e-9


def test_enroll_requires_embeddings():
    g = Gallery()
    try:
        g.enroll("nobody", [])
    except ValueError:
        return
    raise AssertionError("expected ValueError for empty enrollment")


def test_save_load_roundtrip(tmp_path):
    g = Gallery()
    g.enroll("nico", [np.array([1.0, 0.0, 0.0])])
    g.enroll("alex maier", [np.array([0.0, 1.0, 0.0])])  # name with a space
    p = tmp_path / "gallery.npz"
    g.save(p)
    g2 = Gallery.load(p)
    assert sorted(g2.names()) == ["alex maier", "nico"]
    assert g2.match(np.array([1.0, 0.0, 0.0]), threshold=0.3).name == "nico"


def test_save_load_empty_gallery(tmp_path):
    g = Gallery()
    p = tmp_path / "empty.npz"
    g.save(p)
    assert Gallery.load(p).names() == []
