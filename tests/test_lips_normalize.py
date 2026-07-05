import numpy as np
import pytest

from face_scout.geometry import LIPS_IDX
from face_scout.lips.normalize import EYE_OUTER_LEFT, EYE_OUTER_RIGHT, normalize_lips


def _mesh(seed=0):
    rng = np.random.default_rng(seed)
    mesh = rng.uniform(0, 100, size=(468, 3))
    # Guarantee a non-degenerate inter-ocular distance.
    mesh[EYE_OUTER_LEFT] = [10.0, 50.0, 0.0]
    mesh[EYE_OUTER_RIGHT] = [90.0, 50.0, 0.0]
    return mesh


def test_output_shape():
    assert normalize_lips(_mesh()).shape == (len(LIPS_IDX), 2)


def test_translation_invariant():
    mesh = _mesh()
    shifted = mesh + np.array([37.0, -12.0, 5.0])
    assert np.allclose(normalize_lips(mesh), normalize_lips(shifted))


def test_scale_invariant():
    mesh = _mesh()
    scaled = mesh * 2.0  # scaling about the world origin
    assert np.allclose(normalize_lips(mesh), normalize_lips(scaled))


def test_degenerate_mesh_raises():
    mesh = _mesh()
    mesh[EYE_OUTER_LEFT] = mesh[EYE_OUTER_RIGHT]  # zero inter-ocular distance
    with pytest.raises(ValueError, match="inter-ocular"):
        normalize_lips(mesh)
