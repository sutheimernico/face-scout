# Architecture

face-scout is split into a **pure-logic core** (numpy only, fully unit-tested) and
a thin layer of **hardware/library adapters** behind protocols. This keeps the
parts that actually contain logic — tracking, matching, geometry — testable
without a camera or the heavy CV libraries, and isolates the parts that can only
be verified on real hardware.

## Data flow

```
                 ┌────────────┐   frame    ┌──────────────┐  observations
   webcam ──────▶│  capture   │───────────▶│  landmarker  │───────────────┐
                 │ (OpenCV)   │            │ (MediaPipe)  │                │
                 └────────────┘            └──────────────┘                ▼
                                                                     ┌───────────┐
                                                                     │  tracker  │
                                                                     │ (IoU/age) │
                                                                     └─────┬─────┘
                                                              stable tracks│
                        every Nth frame                                    ▼
   ┌────────────┐  embeddings   ┌──────────────┐   match    ┌───────────────────┐
   │  embedder  │──────────────▶│  recognizer  │───────────▶│  tracks + identity │
   │(InsightFace)│              │ (assoc+stick)│  (gallery) └─────────┬─────────┘
   └────────────┘               └──────────────┘                      ▼
                                                                 ┌───────────┐
                                                                 │ renderer  │──▶ window
                                                                 │ (OpenCV)  │
                                                                 └───────────┘
```

## Boundaries

The interfaces live in `types.py` as `typing.Protocol`s:

- **`FrameSource`** — `read() -> frame | None`. Implemented by `WebcamCapture`;
  a test or a video-file source can implement the same three methods.
- **`Landmarker`** — `detect(frame) -> list[FaceObservation]`. Implemented by
  `MediaPipeLandmarker`.
- **`Embedder`** — `embed(frame) -> list[(bbox, embedding)]`. Implemented by
  `InsightFaceEmbedder`.

Everything downstream of these interfaces (`tracker`, `gallery`, `recognizer`)
consumes plain dataclasses and numpy arrays, so it neither knows nor cares which
library produced them.

## Why two vision libraries

No single library does dense geometry **and** identity well:

- **MediaPipe FaceLandmarker** gives 468 landmarks per face incl. a dense lip
  contour, runs real-time on CPU, and in VIDEO mode tracks landmarks
  frame-to-frame. Those lip landmarks are exactly what **Phase 2 (lip reading)**
  needs, so building Phase 1 on it is the deliberate through-line.
- **InsightFace** (`buffalo_l`, ArcFace) gives SOTA face embeddings via pip
  wheels and `onnxruntime` — no dlib compilation, unlike `face_recognition`.

They are associated per frame by bounding-box IoU: the embedder runs on a
throttled cadence, and each embedding is attached to the live track it most
overlaps.

## Key design decisions

- **Throttled identity.** A person's identity does not change frame-to-frame, so
  the embedder runs every `recognize_every` frames (default 10). Geometry and
  tracking run every frame. This is the main lever for keeping FPS usable on CPU.
- **Sticky identity.** A track keeps its last label between recognition passes, so
  the on-screen name does not flicker to `Unknown` on the frames where identity
  is not recomputed.
- **Greedy IoU tracking.** Simple, fast, and adequate for the near-frontal,
  low-count webcam case. Tracks survive `max_age` missed frames so a stable id and
  its identity persist through brief occlusions.
- **Mean-embedding gallery.** Enrollment averages several L2-normalized
  embeddings, which is more robust than a single shot; matching is plain cosine
  similarity with a threshold.

## Testing strategy

- **Unit-tested (no camera, no heavy libs):** `geometry`, `tracker`, `gallery`,
  and the `recognizer` association logic — the whole pure core.
- **Import/API-verified:** the adapter modules import cleanly and the MediaPipe /
  InsightFace API surface they call is checked in the build.
- **Manual only:** the live loop (`app`), `capture`, and `renderer`, plus real CV
  inference, need a camera and a display/GPU with OpenGL — verified by hand on the
  target machine and captured as the demo clip.
