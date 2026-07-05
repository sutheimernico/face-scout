# face-scout — Autopilot Log

One line per iteration: date · task · result.

- 2026-07-05 · scaffold + Phase 1 (tracking+mesh+identity, 24 tests) · done (interactive, on feat/phase1-face-tracking)
- 2026-07-05 · wire face-scout into autopilot loop (LOOP.md, log, register, PLAN Phase 1.5/2) · done
- 2026-07-05 · extract headless Pipeline + fake end-to-end tests (identity/throttle/stickiness), thin app.run · done (28 tests green)
- 2026-07-05 · VideoFileSource + `run --video PATH` for camera-free runs, clip round-trip test · done (30 tests green)
- 2026-07-05 · config + types smoke tests (defaults, frozen contract, Track mutability) · done (37 tests green) — Milestone 6 complete
- 2026-07-05 · Phase 2 SOTA scan + ADR 0001 (landmark-based closed-vocab lip reading; sklearn v1, torch upgrade deferred) · done
- 2026-07-05 · Phase 2 design spec + PLAN Milestones 8-10 (data/features → model → record/live) · done — Milestone 7 complete
- 2026-07-05 · lips/normalize.py (nose-tip translation + inter-ocular scale) + invariance tests · done (41 tests green)
- 2026-07-05 · lips/sequence.py (velocity trim, linear resample, feature vector) + tests · done (51 tests green)
- 2026-07-05 · lips/dataset.py (labeled .npz store + session-aware split) + tests · done (55 tests green) — Milestone 8 complete
- 2026-07-05 · lips/model.py (RandomForest wrapper: fit/predict/proba/save/load) + synthetic tests, pin scikit-learn · done (60 tests green)
- 2026-07-05 · lips train/eval CLI + lips/train.py (session-split eval, confusion matrix) + tests; ruff bugbear config for typer · done (64 tests green) — Milestone 9 complete
- 2026-07-05 · lips/record.py (finite-source utterance capture) + `lips record --video` + tests; lips_data/ gitignored · done (70 tests green)
