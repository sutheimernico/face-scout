# face-scout — LOOP (per-iteration prompt for the autonomous build agent)

You are a fresh headless agent. You do ONE high-value thing, verify it, commit it, and exit.
Progress lives on disk (this file, `PROJECT.md`/`PLAN.md`, git history, `AUTOPILOT_LOG.md`) — never
in context.

## Per-iteration protocol

1. Read the workspace-level `AUTOPILOT.md` (global rules), then this `LOOP.md`, then `PLAN.md` + `PROJECT.md`.
2. Confirm you are on branch `autopilot/work` (the runner guarantees this; if not, stop).
3. Pick the SINGLE highest-value open `- [ ]` task (top-to-bottom, earlier phases first). If a phase
   boundary is reached, run the once-per-phase self-challenge/SOTA step first (write an ADR if it
   changes the plan).
4. Do that one task. Small, reviewable diff. Read existing code before writing; match conventions.
   New logic ships with a test. Hardware/library access stays behind the `types` protocols and is
   faked in tests.
5. Run the gate: `uv run pytest -q` (green) AND `uv run ruff check .` (clean). If red, fix or revert.
6. On green: commit (Conventional Commits, English, imperative), check off the task in `PLAN.md`,
   append a one-line note to `AUTOPILOT_LOG.md`. Then exit.
7. If a task needs a paid resource or a Nico-only input: move it to "Needs Nico", pick another, or
   exit. Never sign up for anything paid. Never fake data or metrics.

## Project-specific hard constraints (never override)

- **Local & free.** No paid APIs, no cloud. Models are the free MediaPipe FaceLandmarker task and
  the free InsightFace `buffalo_l` pack, both downloaded on demand.
- **Biometric data stays local.** Face embeddings / captured frames are NEVER committed — `gallery/`,
  `models/`, `*.npz`, `assets/captures/` are git-ignored. This is a core design constraint.
- **Testable without hardware.** The camera + GL are not available in the build sandbox. Keep all
  logic behind the `FrameSource`/`Landmarker`/`Embedder` protocols and unit-test it with fakes.
  Anything needing a real camera/display/GPU (live loop, real CV inference, demo capture) is a
  **Needs Nico** item — never fake it.
- **Phase discipline.** Phase 1 = tracking+mesh+identity. Phase 2 = lip reading (builds on the lip
  landmarks). Phase 3 = song/audio recognition (separate subsystem). Each phase gets its own spec;
  do not fold future phases into the current one.
- **Simplest solution that meets the task (YAGNI).** Pin new deps with a one-line justification.

## Gate (objective done-check)

`uv run pytest -q` green + `uv run ruff check .` clean. Commit only a green gate.

## Where things are

- Design spec: `docs/superpowers/specs/2026-07-05-face-scout-design.md` · ADRs: `docs/adr/`
- Code: `src/face_scout/` (one responsibility per file) · Tests: `tests/` · CLI: `face-scout run|enroll`
- Pure core (tested here): `types` `geometry` `tracker` `gallery` `recognizer`
- Adapters (Needs-Nico to run): `capture` `landmarker` `embedder` `renderer` · Loop: `app`
- Models/gallery: `models/`, `gallery/` (both git-ignored)
