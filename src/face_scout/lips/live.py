"""Live lip-reading prediction.

``predict_meshes`` is the testable core: normalize a buffer of full-mesh frames
and classify it, returning "?" below a confidence floor. ``run_live`` is the
push-to-talk webcam loop around it (camera + display: Needs Nico, not unit-tested).
"""

from __future__ import annotations

import numpy as np

from ..config import Config
from .model import LipClassifier
from .record import meshes_to_sequence


def predict_meshes(
    model: LipClassifier, meshes: list[np.ndarray], min_confidence: float = 0.0
) -> tuple[str, float]:
    """Predict a label + confidence from a mesh buffer; '?' if below min_confidence."""
    sequence = meshes_to_sequence(meshes)
    if sequence.shape[0] == 0:
        return "?", 0.0
    proba = model.predict_proba(sequence)
    label, confidence = max(proba.items(), key=lambda kv: kv[1])
    return (label, confidence) if confidence >= min_confidence else ("?", confidence)


def run_live(model_path: str, config: Config | None = None, min_confidence: float = 0.5) -> None:
    """Push-to-talk webcam lip reading: hold SPACE while speaking, release to predict.

    Camera + display path — verified by hand on a machine with a webcam.
    """
    import cv2

    from ..capture import WebcamCapture
    from ..landmarker import MediaPipeLandmarker

    config = config or Config()
    model = LipClassifier.load(model_path)
    source = WebcamCapture(config.camera_index)
    landmarker = MediaPipeLandmarker(config.landmarker_model, num_faces=1)

    buffer: list[np.ndarray] = []
    recording = False
    result = ""
    try:
        while True:
            frame = source.read()
            if frame is None:
                break
            faces = landmarker.detect(frame)
            if recording and faces:
                buffer.append(faces[0].landmarks)

            banner = "REC" if recording else (result or "hold SPACE to speak")
            cv2.putText(frame, banner, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow("face-scout lips", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):  # toggle push-to-talk
                recording = not recording
                if not recording and buffer:
                    label, conf = predict_meshes(model, buffer, min_confidence)
                    result = f"{label} ({conf:.2f})"
                    buffer = []
                elif recording:
                    result = ""
    finally:
        source.release()
        landmarker.close()
        cv2.destroyAllWindows()
