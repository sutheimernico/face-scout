# face-scout Phase 2 — Lip Reading (Design)

**Date:** 2026-07-05
**Status:** Approved (autopilot; ADR 0001)
**Depends on:** Phase 1 (lip landmarks from `geometry.lip_landmarks`)
**Approach:** ADR `docs/adr/0001-lip-reading-approach.md`

## Goal

Closed-vocabulary silent-speech recognition from the webcam for a **single
speaker**: recognize a small, fixed set of words / letters / commands from lip
motion, reusing the Phase 1 face-mesh lip landmarks. Local, free, CPU-only.

**In scope (v1):** record labeled utterances → normalize lip-landmark sequences →
resample to fixed length → train a lightweight classifier → predict live.
**Out of scope:** open-set sentence transcription, multi-speaker, pixel mouth-ROI
CNNs (see ADR). No paid resources.

## Architecture

New subpackage `src/face_scout/lips/`, pure-logic-first so everything except the
camera loop is unit-tested without hardware.

| Module | Responsibility | Deps |
|---|---|---|
| `lips/normalize.py` | per-frame lip normalization from the **full** mesh | numpy |
| `lips/sequence.py` | silence-trim + fixed-length resample + flatten to a feature vector | numpy |
| `lips/dataset.py` | labeled sample store (save/load) + **session-aware** split | numpy |
| `lips/model.py` | sklearn classifier wrapper (fit/predict/proba, save/load) | scikit-learn |
| `lips/record.py` | recording mode on the Phase 1 pipeline (utterance capture) | (composes) |
| `lips` CLI | `lips record` \| `lips train` \| `lips eval` \| `lips run` | typer |

## Key design decisions

- **Scale normalization must not use the mouth.** Mouth width/height change while
  speaking — that motion *is* the signal. Normalize scale by a speech-invariant
  reference: **inter-ocular distance** (outer eye corners, mesh indices 33 & 263).
  Normalize translation by a stable point (nose tip, mesh index 1). Therefore
  `normalize` takes the **full 468-point mesh** (available on every Phase 1
  `Track.landmarks`) and returns the normalized lip subset — the lip landmarks
  alone are insufficient for scale.
- **Rotation:** not corrected. Keep the head roughly frontal (ADR); full
  rotation-invariance is over-engineering for a single-user frontal setup.
- **Variable length:** v1 resamples each utterance to a fixed number of frames
  (linear interpolation over time) and flattens → one feature vector per
  utterance. The temporal NN upgrade (deferred) will use padding+masking instead.
- **Silence trimming:** drop leading/trailing frames whose lip-landmark velocity
  is below a threshold, so the classifier sees the utterance, not the pauses.
- **Session-aware split:** samples carry a `session_id`; train/val split holds out
  **whole sessions**, never individual frames or takes, to catch condition
  overfitting (a frame-level split leaks and inflates accuracy).

## Data

- **Sample:** `{ label: str, session_id: str, sequence: float32[T, P, 2] }` where
  `P = len(LIPS_IDX)` normalized lip points. Stored one `.npz` per sample under a
  dataset directory; the dir name groups a vocabulary/user.
- **Biometric-adjacent:** recorded lip geometry stays local — the dataset dir is
  git-ignored, never committed (same rule as the identity gallery).
- **Volume:** ~30-50 reps/class minimum, 50-100+ for confusable visemes, across
  sessions/lighting (ADR).

## Data flow

- **Train:** dataset dir → per sample normalize + trim + resample + flatten →
  session-aware split → `model.fit` → model file + label set.
- **Predict (live):** Phase 1 gives per-frame full mesh → rolling buffer of
  normalized lip frames → on trigger (key / voice-activity proxy) trim + resample +
  flatten → `model.predict_proba` → overlay top label + confidence.

## Error handling

- No face / mesh in a frame → that frame is skipped in the buffer.
- Utterance too short after trimming → rejected with a message (record) or
  low-confidence `?` (live).
- Empty dataset / single class → training refuses with a clear error.
- Unknown-below-confidence prediction → shown as `?`.

## Testing

- `normalize`, `sequence`, `dataset` — pure numpy, fully unit-tested (synthetic
  meshes and sequences; round-trip save/load; split leakage guard).
- `model` — tested on synthetic separable data (fit → high train accuracy;
  save/load round-trip).
- `train`/`eval` CLI — tested end-to-end on a synthetic dataset.
- `record` / `lips run` live loop — logic tested with injected frames; the camera
  path is **Needs Nico** (webcam), never faked as real accuracy numbers.

## Deferred (future ADR/spec)

- PyTorch temporal NN (GRU / 1D-CNN) over variable-length sequences once data grows.
- Pixel mouth-ROI CNN if the landmark approach plateaus on confusable classes.
