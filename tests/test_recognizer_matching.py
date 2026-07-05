import numpy as np

from face_scout.gallery import Gallery
from face_scout.recognizer import assign_identities
from face_scout.types import Track


def make_track(track_id, bbox):
    return Track(track_id=track_id, bbox=bbox, landmarks=np.zeros((2, 2)))


def gallery_with_nico():
    g = Gallery()
    g.enroll("nico", [np.array([1.0, 0.0, 0.0])])
    return g


def test_assigns_identity_to_overlapping_track():
    g = gallery_with_nico()
    tracks = [make_track(0, (0, 0, 10, 10)), make_track(1, (100, 100, 110, 110))]
    embeddings = [((0.0, 0.0, 10.0, 10.0), np.array([1.0, 0.0, 0.0]))]
    assign_identities(tracks, embeddings, g, sim_threshold=0.3)
    assert tracks[0].identity == "nico"
    assert tracks[0].identity_score > 0.3
    assert tracks[1].identity == "Unknown"  # untouched default


def test_no_overlap_keeps_previous_identity_sticky():
    g = gallery_with_nico()
    tracks = [make_track(0, (0, 0, 10, 10))]
    tracks[0].identity = "previous"
    embeddings = [((200.0, 200.0, 210.0, 210.0), np.array([1.0, 0.0, 0.0]))]
    assign_identities(tracks, embeddings, g, sim_threshold=0.3)
    assert tracks[0].identity == "previous"


def test_below_threshold_marks_matched_track_unknown():
    g = gallery_with_nico()
    tracks = [make_track(0, (0, 0, 10, 10))]
    embeddings = [((0.0, 0.0, 10.0, 10.0), np.array([0.0, 0.0, 1.0]))]  # orthogonal
    assign_identities(tracks, embeddings, g, sim_threshold=0.3)
    assert tracks[0].identity == "Unknown"
