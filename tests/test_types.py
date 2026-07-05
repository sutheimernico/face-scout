import dataclasses

import numpy as np
import pytest

from face_scout.types import FaceObservation, Track


def test_track_defaults():
    t = Track(track_id=0, bbox=(0.0, 0.0, 1.0, 1.0), landmarks=np.zeros((2, 2)))
    assert t.age == 0
    assert t.hits == 0
    assert t.identity == "Unknown"
    assert t.identity_score == 0.0


def test_track_is_mutable():
    t = Track(track_id=0, bbox=(0.0, 0.0, 1.0, 1.0), landmarks=np.zeros((2, 2)))
    t.identity = "nico"
    t.hits += 1
    assert t.identity == "nico"
    assert t.hits == 1


def test_face_observation_is_frozen():
    obs = FaceObservation(bbox=(0.0, 0.0, 1.0, 1.0), landmarks=np.zeros((2, 2)))
    with pytest.raises(dataclasses.FrozenInstanceError):
        obs.bbox = (1.0, 1.0, 2.0, 2.0)  # type: ignore[misc]
