import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")  # adapter test: needs OpenCV, but no camera/GL

from face_scout.capture import VideoFileSource  # noqa: E402  (after importorskip)


def _write_clip(path, frames, size):
    w, h = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (w, h))
    if not writer.isOpened():
        pytest.skip("no MJPG encoder available in this OpenCV build")
    for i in range(frames):
        writer.write(np.full((h, w, 3), i * 10, dtype=np.uint8))
    writer.release()


def test_video_file_source_reads_all_frames(tmp_path):
    path = tmp_path / "clip.avi"
    w, h, n = 32, 24, 5
    _write_clip(path, n, (w, h))

    src = VideoFileSource(path)
    count = 0
    while True:
        frame = src.read()
        if frame is None:
            break
        assert frame.shape == (h, w, 3)
        count += 1
    src.release()
    assert count == n


def test_video_file_source_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        VideoFileSource(tmp_path / "does_not_exist.avi")
