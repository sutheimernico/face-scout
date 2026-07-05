import numpy as np
import pytest

from face_scout.geometry import LIPS_IDX
from face_scout.lips.dataset import load_dataset
from face_scout.lips.normalize import EYE_OUTER_LEFT, EYE_OUTER_RIGHT
from face_scout.lips.record import meshes_to_sequence, record_utterance, save_utterance
from face_scout.types import FaceObservation


def _mesh(seed):
    rng = np.random.default_rng(seed)
    mesh = rng.uniform(0, 100, size=(468, 3))
    mesh[EYE_OUTER_LEFT] = [10.0, 50.0, 0.0]
    mesh[EYE_OUTER_RIGHT] = [90.0, 50.0, 0.0]
    return mesh


class FakeSource:
    def __init__(self, frames):
        self._frames = list(frames)
        self._i = 0

    def read(self):
        if self._i >= len(self._frames):
            return None
        frame = self._frames[self._i]
        self._i += 1
        return frame

    def release(self):
        pass


class FakeLandmarker:
    """Emits one face per frame, or none when the frame is None-flagged."""

    def __init__(self, meshes_per_frame):
        self._meshes = meshes_per_frame

    def detect(self, frame):
        mesh = self._meshes[frame]  # frame is an index into the script
        if mesh is None:
            return []
        return [FaceObservation(bbox=(0, 0, 1, 1), landmarks=mesh)]


def test_meshes_to_sequence_shape():
    seq = meshes_to_sequence([_mesh(1), _mesh(2), _mesh(3)])
    assert seq.shape == (3, len(LIPS_IDX), 2)


def test_meshes_to_sequence_empty():
    assert meshes_to_sequence([]).shape == (0, len(LIPS_IDX), 2)


def test_record_utterance_collects_frames():
    meshes = {0: _mesh(0), 1: _mesh(1), 2: _mesh(2), 3: _mesh(3)}
    seq = record_utterance(FakeSource([0, 1, 2, 3]), FakeLandmarker(meshes), min_frames=4)
    assert seq.shape == (4, len(LIPS_IDX), 2)


def test_record_utterance_skips_faceless_frames():
    meshes = {0: _mesh(0), 1: None, 2: _mesh(2), 3: None, 4: _mesh(4), 5: _mesh(5)}
    seq = record_utterance(FakeSource([0, 1, 2, 3, 4, 5]), FakeLandmarker(meshes), min_frames=3)
    assert seq.shape[0] == 4  # 4 frames had a face


def test_record_utterance_too_short_raises():
    meshes = {0: _mesh(0), 1: _mesh(1)}
    with pytest.raises(ValueError, match="too short"):
        record_utterance(FakeSource([0, 1]), FakeLandmarker(meshes), min_frames=4)


def test_save_utterance_roundtrips(tmp_path):
    seq = meshes_to_sequence([_mesh(i) for i in range(5)])
    save_utterance(tmp_path, "hello", "s1", seq, index=0)
    loaded = load_dataset(tmp_path)
    assert len(loaded) == 1
    assert loaded[0].label == "hello"
    assert loaded[0].session_id == "s1"
    assert loaded[0].sequence.shape == (5, len(LIPS_IDX), 2)
