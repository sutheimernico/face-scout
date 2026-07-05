import numpy as np

from face_scout.tracker import Tracker
from face_scout.types import FaceObservation


def obs(x1, y1, x2, y2):
    lm = np.array([[x1, y1], [x2, y2]], dtype=float)
    return FaceObservation(bbox=(float(x1), float(y1), float(x2), float(y2)), landmarks=lm)


def test_new_detection_spawns_track():
    t = Tracker()
    tracks = t.update([obs(0, 0, 10, 10)])
    assert len(tracks) == 1
    assert tracks[0].track_id == 0
    assert tracks[0].hits == 1
    assert tracks[0].age == 0


def test_stable_id_across_frames():
    t = Tracker()
    t.update([obs(0, 0, 10, 10)])
    tracks = t.update([obs(1, 1, 11, 11)])  # overlaps -> same track
    assert len(tracks) == 1
    assert tracks[0].track_id == 0
    assert tracks[0].hits == 2


def test_two_faces_get_distinct_ids():
    t = Tracker()
    tracks = t.update([obs(0, 0, 10, 10), obs(100, 100, 110, 110)])
    assert sorted(tk.track_id for tk in tracks) == [0, 1]


def test_track_ages_then_is_dropped():
    t = Tracker(max_age=2)
    t.update([obs(0, 0, 10, 10)])
    t.update([])
    assert t.tracks[0].age == 1
    t.update([])
    assert t.tracks[0].age == 2
    assert t.update([]) == []  # exceeds max_age -> dropped


def test_reappearing_face_gets_new_id_after_drop():
    t = Tracker(max_age=0)
    t.update([obs(0, 0, 10, 10)])  # id 0
    t.update([])  # ages to 1 > max_age -> dropped
    tracks = t.update([obs(0, 0, 10, 10)])
    assert tracks[0].track_id == 1
