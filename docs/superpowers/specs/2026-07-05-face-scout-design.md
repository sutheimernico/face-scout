# face-scout — Design (Phase 1: Live Face Tracking + Identity)

**Date:** 2026-07-05
**Status:** Approved (verbal go)
**Author:** Nico Sutheimer

## Goal

A webcam application that, in real time:

1. **Detects** faces in the camera stream.
2. **Tracks** each face with a stable ID across frames ("immer tracks").
3. Produces dense **face-mesh landmarks** (incl. lips) as the foundation for later
   lip reading.
4. **Recognizes identity** — matches a tracked face against a gallery of enrolled
   people ("that is Nico") and shows a name + confidence.

Delivered as a **portfolio piece**: clean modules, unit tests for the
hardware-free logic, README + architecture doc + a demo GIF/video (a live webcam
app cannot be deployed statically to GitHub Pages).

Phases 2 (lip reading) and 3 (song/audio recognition) are out of scope here and
get their own specs. Phase 1 is deliberately built so its lip landmarks feed
Phase 2.

## Tech stack

Two responsibilities that no single library does well together, so we split them:

- **Geometry / tracking / lips → MediaPipe FaceLandmarker** (Tasks API).
  Real-time, 468 landmarks incl. dense lips, VIDEO mode with frame-to-frame
  landmark tracking. Direct substrate for Phase 2.
- **Identity → InsightFace** (`buffalo_l`, ArcFace embeddings via `onnxruntime`,
  CPU). pip wheels instead of dlib compilation, SOTA recognition.

Rejected alternatives:
- `face_recognition` (dlib) all-in-one: dlib build friction on WSL, slower on CPU,
  only 68 landmarks, and MediaPipe would be added for Phase 2 anyway.
- MediaPipe only, identity later: user wants identity now.

## Architecture

Modules under `src/face_scout/`, each with one clear purpose and a narrow
interface. Interfaces (`typing.Protocol`) let tests inject fakes so the whole
pure-logic core runs green without the heavy CV libraries installed.

| Module | Responsibility | Heavy deps? |
|---|---|---|
| `types` | Dataclasses (`FaceObservation`, `Track`) + `Landmarker`/`Embedder`/`FrameSource` protocols | no (numpy) |
| `geometry` | bbox from landmarks, IoU, centroid, lip-region helpers | no (numpy) |
| `capture` | `WebcamCapture` (OpenCV `VideoCapture`) behind `FrameSource` | opencv |
| `landmarker` | MediaPipe FaceLandmarker adapter → `FaceObservation`s per frame | mediapipe |
| `tracker` | IoU/centroid association → stable track IDs + track lifetime state | no (numpy) |
| `gallery` | enrollment store: add/mean-embed/save/load + cosine `match` | no (numpy) |
| `recognizer` | InsightFace embedder; throttled; associates identity to a track via `gallery` | insightface, onnxruntime |
| `renderer` | overlay: box, mesh, track ID, name, confidence, FPS | opencv |
| `config` | thresholds, throttle interval, model paths | no |
| `app` | main loop wiring capture→landmarker→tracker→recognizer→renderer | (composes) |
| `enroll` | enrollment mode: capture N shots of one person, mean-embed, store | (composes) |
| `cli` | argparse entrypoint: `run` \| `enroll` | no |

### Data flow (run)

```
frame → landmarker → tracker (stable IDs)
                        │
                        ├── every N frames: recognizer → embedding → gallery.match → identity
                        │
                        └→ renderer (box + mesh + track ID + name + confidence + FPS) → display
```

Identity is computed on a throttled cadence (default every 10 frames) because a
person's identity does not change frame-to-frame; geometry/tracking runs on every
frame. The recognizer runs InsightFace's own detector on the throttled frame and
associates its result to a live track by highest bbox IoU.

## Key decisions / parameters

- **Track association:** greedy IoU matching per frame; a track survives
  `max_age` frames without a match before it is dropped; new detections above a
  min IoU-miss spawn new tracks.
- **Identity matching:** cosine similarity between the ArcFace embedding and each
  enrolled gallery mean-embedding; best match wins if `>= sim_threshold`
  (default 0.35 for ArcFace), else `"Unknown"`. Identity is *sticky* per track —
  once assigned it persists until re-computed, to avoid flicker.
- **Enrollment:** capture `n_shots` (default 5) frames of one clear frontal face,
  embed each, store the L2-normalized mean under a name.

## Error handling

- No camera / cannot open device → clear error message, non-zero exit.
- No face in frame → empty track list, app keeps running.
- Recognizer confidence `< sim_threshold` → label `"Unknown"`.
- Enrollment with no/multiple faces in frame → reject that shot, prompt again.
- Missing MediaPipe model file → clear message with the download path.

## Testing

Hardware-free, no camera, no heavy libs required:

- `tracker`: synthetic bbox sequences → correct stable IDs, aging, re-spawn.
- `gallery`: add/save/load round-trip; cosine match picks the right identity;
  below-threshold → `Unknown`; mean-embedding is L2-normalized.
- `geometry`: IoU, bbox-from-landmarks, lip-region extraction on known inputs.
- `recognizer` association: IoU association of embeddings to tracks with fakes.

The live loop (`app`, `capture`, `landmarker`, `renderer`) stays thin and is
verified manually on a machine with a webcam, plus a recorded demo clip.

## Privacy

Face embeddings are biometric data. The gallery store stays local and is listed
in `.gitignore`. No embeddings or captured frames are committed.

## Out of scope (future specs)

- **Phase 2 — lip reading:** sequence model (e.g. CNN+GRU) over the lip-landmark
  / mouth-ROI sequences already produced by `tracker`.
- **Phase 3 — song/audio recognition:** separate subsystem (audio fingerprinting
  à la Shazam, or audio embeddings).
