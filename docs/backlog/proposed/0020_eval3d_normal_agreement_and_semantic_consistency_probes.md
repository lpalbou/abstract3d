# 0020 — Eval3D-class probes: geometry↔texture normal agreement + DINO semantic view-consistency

## Status
Proposed (2026-07-21)

## Context
The v2 evaluation panel (`scripts/model_quality_sweep_v2.py`,
`docs/research/evaluation_methods_2026.md`) measures mesh defects on normal
renders and texture placement against the photo. Two Eval3D-class probes
[arXiv:2504.18509, CVPR 2025] were researched, judged valuable, and deferred
as too heavy for the current no-new-heavy-deps constraint:

1. **Geometry↔texture normal agreement**: render RGB (textured) views, run a
   monocular normal estimator (Omnidata / DSINE / StableNormal class,
   ~300 MB+ checkpoints), and compare predicted normals against the mesh's
   own rendered normals (pixel-wise angular error). Large disagreement =
   texture paints structure the geometry does not carry (ghost glasses,
   lips-on-nose) or geometry carries structure the texture contradicts.
   This is the one researched metric that scores the e20/e21 texture
   failure class GEOMETRICALLY rather than photometrically.
2. **DINO semantic view-consistency**: project surface points into per-view
   DINO feature maps and score per-point feature std across views (Janus /
   semantic drift detector). Needs DINOv2-small (~90 MB) + a projection
   pass; our GeoRenderer position buffer already provides the 3D↔pixel
   correspondence, so the projection half is nearly free.

## Why not now
Both need new model downloads beyond the accepted small-ONNX tier and (for
Omnidata-class) new dependency stacks. The current panel already ranks the
calibration set faithfully without them.

## Acceptance
- Runs locally on MPS/CPU, models cached under `~/.cache/abstract3d_eval/`.
- On the 9-bundle calibration set: normal-agreement metric must rank
  e20/e21 (texture catastrophic) worse than e10/e15 (texture passable)
  WITHOUT using the photo; semantic-consistency metric must not contradict
  the operator mesh verdicts.
- Wired as optional extras in `model_quality_sweep_v2.py` behind try/except
  with loud notes, never rank drivers until validated on a second subject.
