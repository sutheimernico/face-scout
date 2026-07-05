# face-scout

Real-time webcam **face tracking** with dense **mesh landmarks** and **identity
recognition** — and the deliberate foundation for a later lip-reading system.

Point it at your webcam: it detects every face, follows each one under a stable
id across frames, draws the 468-point face mesh (including lips), and — for people
you have enrolled — labels who they are with a confidence score.

> Phase 1 of a staged project. The lip landmarks produced here are the input for
> **Phase 2 (lip reading)**; **Phase 3 (song/audio recognition)** is a separate
> future subsystem. See [PROJECT.md](PROJECT.md).

<!-- Demo: record on a machine with a webcam and drop the clip here. -->
<!-- ![demo](assets/demo.gif) -->

## What it does

- **Detect + track** multiple faces live, each with a stable id that survives
  brief occlusions.
- **Dense mesh** — MediaPipe's 468 landmarks per face, incl. dense lip contour.
- **Identity** — matches each face against an enrolled gallery (InsightFace /
  ArcFace embeddings), shown as `name + cosine score`, else `Unknown`.
- **Enrollment** — a guided mode to register a person from several shots.

## How it works

Two responsibilities, two libraries, cleanly separated:

| Concern | Library | Why |
|---|---|---|
| Geometry / tracking / lips | **MediaPipe FaceLandmarker** (Tasks API, VIDEO mode) | 468 landmarks incl. dense lips, real-time, frame-to-frame tracking — the Phase 2 substrate |
| Identity | **InsightFace** (`buffalo_l`, ArcFace, `onnxruntime` CPU) | SOTA embeddings via pip wheels (no dlib build) |

```
frame ─▶ landmarker ─▶ tracker (stable ids)
                          │
                          ├─ every Nth frame: embedder ─▶ gallery.match ─▶ identity
                          │
                          └─▶ renderer (box + mesh + id + name + confidence + FPS) ─▶ window
```

Identity runs on a throttle (default every 10 frames) because a person's identity
does not change frame-to-frame; geometry/tracking runs on every frame. Identity is
associated to the correct track by bounding-box IoU and is *sticky* between passes
to avoid label flicker.

## Install

Needs Python 3.11+ and [uv](https://docs.astral.sh/uv/). A webcam is required to
run the live app (the pure logic is unit-tested without one).

```bash
uv sync
```

The MediaPipe model bundle is downloaded automatically on first run; InsightFace
downloads its model pack (`buffalo_l`) on first use.

## Usage

Enroll yourself first (SPACE captures a shot with exactly one face in view):

```bash
uv run face-scout enroll nico          # capture 5 shots -> gallery
```

Then run the live tracker:

```bash
uv run face-scout run                   # q or Esc to quit
uv run face-scout run --camera 1        # pick a different device
```

With an empty gallery it still tracks and draws the mesh — it just labels every
face `Unknown` until you enroll someone.

Read live from a recorded clip instead of the webcam:

```bash
uv run face-scout run --video clip.mp4
```

## Lip reading (Phase 2, v1)

Closed-vocabulary, single-speaker lip reading built on the Phase 1 lip landmarks
(approach: [ADR 0001](docs/adr/0001-lip-reading-approach.md); design:
[spec](docs/superpowers/specs/2026-07-05-lip-reading-phase2-design.md)). Record a
few clips per word, train a classifier, and predict live:

```bash
# 1. Record labeled utterances (one short clip per utterance). Use a fresh
#    --session per recording day so evaluation can hold whole sessions out.
uv run face-scout lips record --label yes --session day1 --video yes_01.mp4

# 2. Train, and evaluate honestly on a held-out session.
uv run face-scout lips train --data lips_data --out models/lips.joblib
uv run face-scout lips eval  --data lips_data --val-session day2

# 3. Live push-to-talk: hold SPACE while speaking, release to predict.
uv run face-scout lips run --model models/lips.joblib
```

Realistic accuracy for a 10-30 word single-user vocabulary is ~80-95% (ADR 0001);
the error floor is visually confusable visemes (p/b/m), and the head must stay
roughly frontal. The v1 classifier is a landmark-feature RandomForest; a PyTorch
temporal model is the deferred upgrade once more data is recorded.

## Architecture

Modules under `src/face_scout/`, each with one purpose and a narrow interface, so
the pure-logic core is fully unit-tested without any camera or heavy CV library.
See [docs/architecture.md](docs/architecture.md) and the design spec in
[`docs/superpowers/specs/`](docs/superpowers/specs/).

| Module | Responsibility |
|---|---|
| `types` | dataclasses + `FrameSource`/`Landmarker`/`Embedder` protocols |
| `geometry` | bbox, IoU, centroid, lip-region extraction |
| `tracker` | greedy IoU association → stable ids + aging |
| `gallery` | enrollment store + cosine matching |
| `recognizer` | associate embeddings to tracks; sticky identity |
| `capture` · `landmarker` · `embedder` · `renderer` | hardware/library adapters |
| `pipeline` | headless per-frame wiring (landmarks → track → throttled identity) |
| `app` · `enroll` · `cli` | live loop, enrollment, Typer CLI |
| `lips/` | Phase 2 — normalize · sequence · dataset · model · record · live |

## Tests

```bash
uv run pytest        # pure-logic core + Phase 2 (normalize, sequence, dataset, model)
uv run ruff check .
```

The camera/display paths (`app`, `capture`, `landmarker`, `renderer`, `lips run`)
are thin and verified by hand on a machine with a webcam.

## Privacy

Face embeddings and recorded lip data are biometric. The gallery (`gallery/`) and
lip dataset (`lips_data/`) stay local and git-ignored — **no embeddings, lip
recordings, or captured frames are ever committed.**

## Roadmap

- **Phase 1 (this repo)** — tracking + mesh + identity. ✅
- **Phase 2 — lip reading** — closed-vocabulary landmark classifier
  (record → train → eval → run). ✅ *code complete; pending live webcam verification.*
- **Phase 3 — song/audio recognition** — separate audio subsystem. *(future)*

## License

MIT — see [LICENSE](LICENSE).
