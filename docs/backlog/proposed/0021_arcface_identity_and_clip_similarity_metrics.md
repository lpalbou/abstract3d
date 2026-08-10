# 0021 — ArcFace-grade identity metric + CLIP photo↔render similarity

## Status
Proposed (2026-07-21)

## Context
v2 ships a cheap identity extra (`sface_identity`: OpenCV SFace, 37 MB) and
LPIPS face crops. Calibration (2026-07-21, 9 bundles) shows they are
trustworthy at the poles (e20 lips-on-nose → 0.30, below OpenCV's 0.363
same-identity threshold; e15 best texture → 0.82) but noisy midfield (e19
"recovered" texture → 0.42). The face-reconstruction literature standard is
an ArcFace-class encoder (MICA uses pretrained ArcFace as its identity
backbone [arXiv:2204.06607]); glasses + render domain shift are exactly the
robustness gap between SFace-tier and ArcFace-tier embeddings.

## Proposal
1. **ArcFace identity**: InsightFace `buffalo_l` recognition ONNX (~166 MB,
   runs on onnxruntime already present via rembg) — cosine photo vs
   textured front render, replacing/backing `sface_identity`. Optionally
   average over ±15° near-front renders to damp view luck.
2. **CLIP similarity**: image-embedding cosine photo↔render (identity +
   semantics at coarse grain; the field's CLIP-score for image prompts
   [Hunyuan3D 2.0, arXiv:2501.12202]). OpenCLIP ViT-B/32 ~350 MB — accept
   only if a smaller distilled variant proves sufficient; otherwise defer.

## Why not now
Model sizes sit at/over the accepted download tier and add insightface /
open_clip dependencies; SFace+LPIPS already cover the pole cases the
operator cares about.

## Acceptance
- On the calibration set: identity metric must rank e20 and e21 (texture
  catastrophic) below e10/e15/e11 (texture passable) with a margin larger
  than SFace's, and behave monotonically under a synthetic corruption
  ladder (progressive UV scramble of e15's texture).
- Local-only, cached models, loud degradation notes, optional extras.
