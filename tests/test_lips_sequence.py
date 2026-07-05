import numpy as np
import pytest

from face_scout.lips.sequence import (
    frame_velocities,
    resample,
    to_feature_vector,
    trim_silence,
)


def seq_x(xs):
    """A (T, 1, 2) sequence moving along x only."""
    a = np.zeros((len(xs), 1, 2))
    a[:, 0, 0] = xs
    return a


def test_frame_velocities_static_is_zero():
    assert np.allclose(frame_velocities(seq_x([5, 5, 5])), [0, 0, 0])


def test_frame_velocities_constant_motion():
    assert np.allclose(frame_velocities(seq_x([0, 0, 1, 2, 3, 3, 3])), [0, 0, 1, 1, 1, 0, 0])


def test_trim_silence_keeps_active_span():
    trimmed = trim_silence(seq_x([0, 0, 0, 1, 2, 3, 3, 3, 3]), threshold=0.5)
    assert np.allclose(trimmed[:, 0, 0], [1, 2, 3])


def test_trim_silence_all_static_returns_empty():
    assert trim_silence(seq_x([2, 2, 2]), threshold=0.5).shape[0] == 0


def test_resample_same_length_is_identity():
    s = seq_x([0, 1, 2])
    assert np.allclose(resample(s, 3), s)


def test_resample_interpolates():
    assert np.allclose(resample(seq_x([0, 4]), 5)[:, 0, 0], [0, 1, 2, 3, 4])


def test_resample_single_frame_repeats():
    assert np.allclose(resample(seq_x([7]), 3)[:, 0, 0], [7, 7, 7])


def test_resample_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        resample(np.zeros((0, 1, 2)), 4)


def test_to_feature_vector_size():
    feat = to_feature_vector(seq_x([0, 1, 2]), length=4)
    assert feat.shape == (4 * 1 * 2,)


def test_to_feature_vector_empty_after_trim_raises():
    with pytest.raises(ValueError, match="empty after"):
        to_feature_vector(seq_x([2, 2, 2]), length=4, velocity_threshold=1.0)
