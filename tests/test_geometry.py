import numpy as np

from face_scout.geometry import LIPS_IDX, bbox_from_landmarks, centroid, iou, lip_landmarks


def test_bbox_from_landmarks():
    lm = np.array([[10, 20], [30, 5], [15, 40]], dtype=float)
    assert bbox_from_landmarks(lm) == (10.0, 5.0, 30.0, 40.0)


def test_iou_identical_is_one():
    b = (0.0, 0.0, 10.0, 10.0)
    assert iou(b, b) == 1.0


def test_iou_disjoint_is_zero():
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_iou_partial_overlap():
    # two 10x10 boxes sharing a 10x5 strip: inter=50, union=150
    assert abs(iou((0, 0, 10, 10), (0, 5, 10, 15)) - 1 / 3) < 1e-9


def test_centroid():
    assert centroid((0.0, 0.0, 10.0, 20.0)) == (5.0, 10.0)


def test_lip_landmarks_shape():
    mesh = np.arange(468 * 3, dtype=float).reshape(468, 3)
    assert lip_landmarks(mesh).shape == (len(LIPS_IDX), 3)


def test_lip_indices_are_valid_mesh_range():
    assert all(0 <= i < 468 for i in LIPS_IDX)
