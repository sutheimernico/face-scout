# face-scout — Implementation Plan (Phase 1)

Spec: `docs/superpowers/specs/2026-07-05-face-scout-design.md`

Build order goes pure-logic-first so the core is fully unit-tested before the
hardware adapters (which cannot be tested without a camera) are wired in.

## Milestone 1 — Testable core (no camera, no heavy CV libs)
- [x] `types.py` — `FaceObservation`, `Track` dataclasses; `FrameSource`,
      `Landmarker`, `Embedder` protocols.
- [x] `geometry.py` — `bbox_from_landmarks`, `iou`, `centroid`, `lip_landmarks`.
- [x] `tracker.py` — greedy IoU association, stable IDs, aging, re-spawn.
- [x] `gallery.py` — add / mean-embed / save / load / cosine `match`.
- [x] `config.py` — thresholds, throttle, model paths.
- [x] Tests: `test_geometry`, `test_tracker`, `test_gallery`.

## Milestone 2 — Recognition association
- [x] `recognizer.py` — associate embeddings to live tracks by IoU; sticky
      identity; throttling handled by the caller.
- [x] Tests: `test_recognizer_matching` (fakes, no InsightFace).

## Milestone 3 — Hardware adapters (real libs)
- [x] `capture.py` — OpenCV `WebcamCapture` behind `FrameSource`.
- [x] `landmarker.py` — MediaPipe FaceLandmarker adapter (+ model auto-download).
- [x] `embedder.py` — InsightFace embedder implementation.
- [x] `renderer.py` — overlay drawing.

## Milestone 4 — App + CLI
- [x] `app.py` — main loop wiring the pipeline; keyboard quit.
- [x] `enroll.py` — enrollment mode (N shots → mean embedding → gallery).
- [x] `cli.py` — Typer app: `run` | `enroll`.

## Milestone 5 — Portfolio polish
- [x] README with architecture diagram + usage.
- [x] `docs/architecture.md`.
- [ ] Manual verification on a machine with a webcam; record demo GIF/video
      into `assets/` and embed in README. **(Nico — needs a camera + display.)**

## Milestone 6 — Testability & camera-free verification (Phase 1 hardening)
- [x] Extract a headless `Pipeline` (inject `Landmarker`/`Embedder`) that owns the
      per-frame wiring; make `app.run` thin I/O around it.
- [x] End-to-end pipeline test with fake landmarker + embedder: identity flows to
      the correct track across frames, throttle + stickiness honored.
- [x] `VideoFileSource` (`FrameSource` over a video file) + `run --video PATH` for
      camera-free runs; unit test on a generated clip.
- [x] Smoke tests for `types` and `config` defaults.

## Milestone 7 — Phase 2 kickoff: lip reading (DESIGN)
- [x] Phase-boundary self-challenge: SOTA scan of lip-reading approaches; write
      `docs/adr/0001-lip-reading-approach.md`.
- [x] Phase 2 design spec in `docs/superpowers/specs/`.
- [x] Expand this plan with Phase 2 build tasks (data → features → model → eval).

## Milestone 8 — Phase 2 data & features (pure numpy, testable now)
Spec: `docs/superpowers/specs/2026-07-05-lip-reading-phase2-design.md`
- [x] `lips/normalize.py` — normalize lip landmarks from the full mesh (inter-ocular
      scale, nose-tip translation); return normalized lip subset. + tests.
- [x] `lips/sequence.py` — velocity-based silence trim, fixed-length resample,
      flatten to a feature vector. + tests.
- [x] `lips/dataset.py` — labeled sample store (save/load `.npz`, label + session_id)
      and session-aware train/val split (no frame leakage). + tests.

## Milestone 9 — Phase 2 model (adds scikit-learn)
- [x] `lips/model.py` — sklearn classifier wrapper (fit/predict/predict_proba,
      save/load); pin scikit-learn with justification. + tests on synthetic data.
- [ ] `lips train` / `lips eval` CLI — train from a dataset dir, report accuracy +
      confusion matrix. + test on a synthetic dataset.

## Milestone 10 — Phase 2 recording & live (Needs Nico to run; logic tested with fakes)
- [ ] `lips/record.py` — utterance capture on the Phase 1 pipeline (start/stop,
      collect per-frame lip landmarks, store labeled). Logic tested with injected frames.
- [ ] `lips run` — live prediction overlay (rolling buffer → normalize+resample →
      classify). Manual verify (**Needs Nico**: webcam).

## Verification
- [x] `uv run pytest` green (24 tests, the whole pure core).
- [x] `uv run ruff check` clean.
- [x] Adapter modules import; MediaPipe + InsightFace API surface verified.
- [ ] Live loop + real CV inference — verified by hand on Nico's machine (blocked
      in the build sandbox: headless, no camera, missing system OpenGL).

## Outcome (2026-07-05)

Phase 1 implemented as designed. Pure-logic core (tracker, gallery, geometry,
recognizer association) is fully unit-tested — **24 tests green**, ruff clean.
The hardware adapters (OpenCV capture, MediaPipe FaceLandmarker, InsightFace
embedder, OpenCV renderer) are written and **import cleanly in the full
environment**, with the MediaPipe/InsightFace API surface smoke-checked.

**Verification boundary (honest):** real CV inference and the live webcam loop
could **not** be run in the build sandbox — it is headless (no camera) and lacks
system OpenGL (`libGLESv2.so.2`), which MediaPipe's vision tasks require at
runtime. The MediaPipe model bundle downloads and the landmarker constructs fine;
inference itself needs a display/GPU. This is an environment limitation, not a
code fault.

**Open (for Nico, on a machine with a webcam):**
1. `uv run face-scout enroll <name>`, then `uv run face-scout run` — confirm live
   tracking, mesh, and identity end-to-end.
2. Tune `sim_threshold` / `recognize_every` to the real CPU FPS.
3. Record the demo GIF/video into `assets/` and embed it in the README.

**Deviations from the spec:** none of substance. `embedder.py` was split out from
`recognizer.py` so the pure association logic stays importable without InsightFace.
