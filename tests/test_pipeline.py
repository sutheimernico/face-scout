import numpy as np

from face_scout.config import Config
from face_scout.gallery import Gallery
from face_scout.pipeline import Pipeline
from face_scout.tracker import Tracker
from face_scout.types import FaceObservation


class FakeLandmarker:
    """Replays a scripted list of observations, one entry per frame."""

    def __init__(self, per_frame):
        self._per_frame = per_frame
        self._i = 0

    def detect(self, frame):
        obs = self._per_frame[min(self._i, len(self._per_frame) - 1)]
        self._i += 1
        return obs


class CountingEmbedder:
    """Returns a fixed result and counts how often embed() is called."""

    def __init__(self, result):
        self.calls = 0
        self._result = result

    def embed(self, frame):
        self.calls += 1
        return self._result


def _face(bbox):
    x1, y1, x2, y2 = bbox
    return FaceObservation(bbox=bbox, landmarks=np.array([[x1, y1], [x2, y2]], dtype=float))


def _gallery_with_nico():
    g = Gallery()
    g.enroll("nico", [np.array([1.0, 0.0, 0.0])])
    return g


def _pipeline(landmarker, gallery, config, embedder=None):
    tracker = Tracker(iou_threshold=config.iou_threshold, max_age=config.max_age)
    return Pipeline(landmarker, tracker, gallery, config, embedder)


def test_identity_flows_to_track_and_is_sticky_and_throttled():
    cfg = Config(recognize_every=2, min_hits=1)
    bbox = (0.0, 0.0, 10.0, 10.0)
    lm = FakeLandmarker([[_face(bbox)]] * 4)
    emb = CountingEmbedder([(bbox, np.array([1.0, 0.0, 0.0]))])
    pipe = _pipeline(lm, _gallery_with_nico(), cfg, emb)
    frame = np.zeros((20, 20, 3), dtype=np.uint8)

    t0 = pipe.process(frame)  # frame 0 -> recognizes
    assert t0[0].identity == "nico"
    tid = t0[0].track_id
    assert emb.calls == 1

    t1 = pipe.process(frame)  # frame 1 -> throttled, identity stays (sticky)
    assert t1[0].identity == "nico"
    assert t1[0].track_id == tid
    assert emb.calls == 1

    pipe.process(frame)  # frame 2 -> recognizes again
    assert emb.calls == 2


def test_track_id_is_stable_across_moving_frames():
    cfg = Config(min_hits=1)
    lm = FakeLandmarker([[_face((0, 0, 10, 10))], [_face((1, 1, 11, 11))], [_face((2, 2, 12, 12))]])
    pipe = _pipeline(lm, Gallery(), cfg, embedder=None)
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    ids = {pipe.process(frame)[0].track_id for _ in range(3)}
    assert ids == {0}  # one persistent track


def test_no_embedder_tracks_without_identity():
    cfg = Config(min_hits=1)
    lm = FakeLandmarker([[_face((0, 0, 10, 10))]])
    pipe = _pipeline(lm, Gallery(), cfg, embedder=None)
    tracks = pipe.process(np.zeros((20, 20, 3), dtype=np.uint8))
    assert tracks[0].identity == "Unknown"
    assert pipe.has_recognition is False


def test_frame_index_advances():
    cfg = Config(min_hits=1)
    lm = FakeLandmarker([[]])
    pipe = _pipeline(lm, Gallery(), cfg, embedder=None)
    assert pipe.frame_index == 0
    pipe.process(np.zeros((4, 4, 3), dtype=np.uint8))
    assert pipe.frame_index == 1
