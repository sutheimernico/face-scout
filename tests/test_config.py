import dataclasses
from pathlib import Path

import pytest

from face_scout.config import Config


def test_defaults_are_sane():
    c = Config()
    assert c.recognize_every >= 1  # guards against modulo-by-zero in the pipeline
    assert -1.0 <= c.sim_threshold <= 1.0  # cosine range
    assert 0.0 <= c.iou_threshold <= 1.0
    assert c.max_age >= 0
    assert c.enroll_shots >= 1
    assert c.max_faces >= 1


def test_paths_are_path_objects():
    c = Config()
    assert isinstance(c.landmarker_model, Path)
    assert isinstance(c.gallery_path, Path)
    assert c.gallery_path.suffix == ".npz"


def test_config_is_frozen():
    c = Config()
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.sim_threshold = 0.9  # type: ignore[misc]


def test_overrides_apply():
    c = Config(camera_index=2, recognize_every=5, enroll_shots=3)
    assert (c.camera_index, c.recognize_every, c.enroll_shots) == (2, 5, 3)
