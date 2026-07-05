"""Enrollment mode: capture N clean shots of one person and store a mean embedding."""

from __future__ import annotations

import cv2

from .capture import WebcamCapture
from .config import Config
from .embedder import InsightFaceEmbedder
from .gallery import Gallery

_WINDOW = "face-scout enroll"


def enroll(name: str, config: Config | None = None) -> None:
    config = config or Config()

    source = WebcamCapture(config.camera_index)
    embedder = InsightFaceEmbedder(config.insight_model, config.det_size)
    gallery = Gallery.load(config.gallery_path) if config.gallery_path.exists() else Gallery()

    shots: list = []
    print(
        f"Enrolling '{name}'. SPACE captures a shot (exactly one face in view), "
        f"q/Esc aborts. Need {config.enroll_shots}."
    )
    try:
        while len(shots) < config.enroll_shots:
            frame = source.read()
            if frame is None:
                break

            faces = embedder.embed(frame)
            disp = frame.copy()
            status = f"{len(shots)}/{config.enroll_shots} captured"
            if len(faces) == 1:
                x1, y1, x2, y2 = (int(v) for v in faces[0][0])
                cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
            elif len(faces) == 0:
                status += " | no face"
            else:
                status += " | multiple faces - show only one"
            cv2.putText(disp, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow(_WINDOW, disp)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                print("aborted — nothing enrolled")
                return
            if key == ord(" ") and len(faces) == 1:
                shots.append(faces[0][1])
                print(f"captured {len(shots)}/{config.enroll_shots}")
    finally:
        source.release()
        cv2.destroyAllWindows()

    if not shots:
        print("no shots captured — nothing enrolled")
        return

    gallery.enroll(name, shots)
    gallery.save(config.gallery_path)
    print(f"enrolled '{name}' from {len(shots)} shots -> {config.gallery_path}")
