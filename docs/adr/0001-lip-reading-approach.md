# ADR 0001 — Lip-reading approach (Phase 2)

- **Status:** Accepted
- **Date:** 2026-07-05
- **Context phase:** Phase 1 (tracking + mesh + identity) complete; this ADR opens Phase 2.

## Context

Phase 2 adds silent-speech recognition ("lip reading"). Phase 1 already produces,
per face per frame in real time, MediaPipe's 468-point face mesh including ~40
dense lip landmarks (`geometry.lip_landmarks`). The system must stay **local,
free, and CPU-first**, and its training data will be **small and self-recorded by
one user** via webcam.

## Decision drivers

- Local & free; no paid cloud, no multi-GPU training (AUTOPILOT constraint).
- Reuse the Phase 1 lip landmarks rather than build a second vision pipeline.
- Tiny, single-speaker, same-camera dataset — data efficiency dominates.
- Honest scope: a working closed-vocabulary classifier beats a broken attempt at
  open-set sentence transcription.

## What the SOTA does (and why it does not transfer)

The strong lineage — LipNet (STCNN+BiGRU+CTC), 3D-CNN+ResNet+BiLSTM word-level on
LRW (~92-93%), Conformer VSR, AV-HuBERT, Auto-AVSR (0.9% WER on LRS3) — all assume
**hundreds of hours of curated video and hundreds of GPU/TPU-hours**. Open, in-the-
wild sentence benchmarks (LRS2/LRS3) remain unsolved (~14-30% WER in 2025-26).
None of this transfers to a CPU + a few dozen self-recorded clips; it is useful
only as architectural inspiration (temporal-conv → RNN/Conformer → CTC).

## Options considered

1. **Landmark sequence → lightweight classifier.** Normalize the MediaPipe lip
   landmarks per frame, resample each utterance to a fixed length, flatten, and
   classify with sklearn (SVM / RandomForest / DTW-kNN). Cheap, data-efficient,
   debuggable; loses texture cues (tongue, teeth).
2. **Landmark sequence → small temporal NN.** A 1-2 layer GRU or 1D-CNN (PyTorch,
   CPU) over the variable-length landmark sequence. Better temporal modeling; needs
   more data (~50+/class) and adds a heavy dependency.
3. **Pixel mouth-ROI CNN (LipNet-style).** Higher accuracy ceiling for larger
   vocabularies, but data- and compute-hungry, less robust to webcam/lighting
   variance, and duplicates MediaPipe's detection work.

Evidence: landmark-only is discouraging for **open-set, multi-speaker** VSR (a 2025
MediaPipe+ST-GCN study found landmarks insufficient alone for LRS2/LRS3). But for a
**closed, single-speaker, same-camera** vocabulary the calculus flips — Google's
Isolated Sign Language Recognition (landmark-only, ~250 closed classes) reaches
>85-90%. Realistic expectation for a 10-30 word/letter single-user classifier:
**~80-95%**, with the ceiling set by visually confusable visemes (p/b/m), not by
the landmark modality itself.

## Decision

**Landmark-based, closed-vocabulary, single-speaker lip reading.**

- **v1 baseline:** per-frame normalized lip landmarks (translation via a stable
  reference point, scale via a stable inter-landmark distance) → fixed-length
  resampled, flattened sequence → a lightweight sklearn classifier. Trains in
  seconds on tens of samples per class.
- **Upgrade path (only once data grows):** a small PyTorch GRU / 1D-CNN over the
  variable-length, padded+masked landmark sequence.
- **Pixel mouth-ROI CNN:** explicitly deferred — revisit only if the landmark-only
  approach plateaus on confusable classes.

Data collection (built on Phase 1): ~30-50 reps/class minimum (50-100+ for
confusable classes), recorded across sessions/lighting. Normalize translation +
scale per frame; keep the head roughly frontal rather than engineering full
rotation-invariance. Trim silence via a landmark-velocity threshold. **Split by
session/take, never by frame**, and hold out whole sessions for validation to
catch condition-overfitting.

## Consequences

- **Positive:** reuses Phase 1 output directly; trains locally on CPU in seconds;
  small honest scope with a credible 80-95% target; no heavy dependency for v1.
- **Negative:** capped accuracy vs pixel methods; confusable visemes will be the
  error floor; MediaPipe degrades on profile/occlusion, so the head must stay
  roughly frontal; open-vocabulary transcription is out of reach and is not a goal.
- **Dependency:** v1 adds `scikit-learn` only. PyTorch is deferred to the upgrade
  step so the v1 data/feature pipeline stays light and fully unit-testable.

## References

- LipNet — https://arxiv.org/abs/1611.01599
- Conformers Are All You Need for VSR — https://arxiv.org/html/2302.10915v2
- Auto-AVSR — https://arxiv.org/abs/2303.14307 · repo https://github.com/mpc001/auto_avsr
- AV-HuBERT — https://arxiv.org/abs/2201.01763
- Landmark-Guided Cross-Speaker Lip Reading — https://arxiv.org/html/2403.16071v1
- Point-Visual Fusion for Phoneme-Level VSR (MediaPipe+ST-GCN) — https://arxiv.org/html/2507.18863v1
- LipLearner: Customizable Silent Speech on Mobile — https://arxiv.org/abs/2302.05907
- Google Isolated Sign Language Recognition (landmark-only, closed-vocab) — https://github.com/JosephZahar/Google-Isolated-Sign-Language-Recognition-Kaggle
- Survey of Advancement in Lip Reading Models (2025) — https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/ipr2.70095
