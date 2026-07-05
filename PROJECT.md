# face-scout — Project Charter

## What
A real-time webcam application that detects, **tracks**, and **identifies** faces,
and produces dense face-mesh landmarks (including lips) as the foundation for a
later lip-reading system.

## Why
Portfolio piece exploring hardware (webcam) + computer vision. Deliberately staged
so Phase 1 (this repo) yields the lip landmarks that Phase 2 (lip reading) needs.

## Scope

### Phase 1 — this repo
- Live face detection + stable multi-face **tracking** across frames.
- Dense **face-mesh landmarks** (MediaPipe FaceLandmarker, 468 points incl. lips).
- **Identity recognition** against an enrolled gallery (InsightFace / ArcFace).
- Enrollment mode to register known people.
- Live overlay: box, mesh, track ID, name, confidence, FPS.

### Future — separate specs
- **Phase 2 — lip reading:** sequence model over lip-landmark / mouth-ROI streams.
- **Phase 3 — song/audio recognition:** separate audio subsystem.

## Constraints
- **Local only.** Face embeddings are biometric data — the gallery stays on disk,
  in `.gitignore`, never committed.
- **CPU-first.** Runs without a GPU; identity is throttled to keep FPS usable.
- **Testable core.** All non-hardware logic (tracking, matching, geometry, gallery)
  behind interfaces with unit tests; the live loop stays thin.

## Stack
Python 3.11+, MediaPipe (geometry/landmarks/tracking), InsightFace + onnxruntime
(identity), OpenCV (capture + rendering), Typer (CLI). Managed with `uv`.

## Status
Phase 1 in progress (started 2026-07-05). Design: `docs/superpowers/specs/2026-07-05-face-scout-design.md`.
