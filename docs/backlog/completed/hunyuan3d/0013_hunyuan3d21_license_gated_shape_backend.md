# Completed: Hunyuan3D-2.1 license-gated shape backend with shared projection texturing

## Metadata

- Created: 2026-07-04
- Status: Completed
- Completed: 2026-07-04

## ADR status

- Governing ADRs: [ADR 0005](../../../adr/0005_hunyuan3d21_license_gated_official_shape_backend.md)
- ADR impact: Added ADR 0005

## Context

The catalog lacked the strongest locally runnable open-weight geometry model. Hunyuan3D-2.1
was debated earlier because its Community License is territory-restricted (excludes EU/UK/
South Korea), which disqualifies it from the permissive validated tier but not from an
explicitly gated experimental tier. Its shape stage is fully self-contained: the conditioner
weights ship inside the official checkpoint and surface extraction is plain marching cubes,
so no gated companion model blocks local use (unlike TRELLIS.2).

## What was delivered

- `abstract3d:hunyuan3d21-local` backend wrapping the official pinned source snapshot
  (`Tencent-Hunyuan/Hunyuan3D-2.1` @ `82920d64`) and the official fp16 DiT checkpoint
  (`tencent/Hunyuan3D-2.1/hunyuan3d-dit-v2-1`, 7.4 GB), with an explicit license gate on every
  weight-touching path and license labels in provider listings and bundle metadata.
- An adaptive coarse-to-fine volume decoder replacing the upstream hierarchical decoder. The
  upstream decoder starts at 63^3 (loses thin structures entirely: the starship proof produced
  zero surface cells and crashed) and relies on multi-axis advanced-indexing scatter that
  misbehaves on Apple MPS. The replacement starts dense at 128^3, refines sign-change bands
  with exact-doubling level schedules, keeps index bookkeeping on the host, and falls back to
  dense decoding for objects the coarse grid misses.
- `abstract3d.texturing`: the TripoSR bake generalized into a backend-agnostic projection
  texture pipeline (source-pose estimation, 2D photo registration, photometric refinement,
  contour alpha erosion, exposure harmonization, seam-feathered best-view-biased blending,
  geometry-verified mirror completion, 3D inverse-distance fill for unseen texels, edge
  bleed). TripoSR now routes its bake through the same module and contributes its triplane
  field as the unseen-texel prior; Hunyuan uses the 3D fill.
- Full `Scene3DManager` protocol support, `abstract3d[hunyuan3d]` extra, model catalog entry,
  docs (README, models, architecture, ADR 0005), and weight-free unit tests.

## Validation

- `pytest -q tests` (117 tests) green.
- Local Apple-`mps` proof bundles for the four checked objects (owl, chair, starship, face)
  under `artifacts/validation/final-proof/`, generated at 30 steps / octree 384 / fp16.
- On the checked objects, Hunyuan geometry is materially stronger than the validated TripoSR
  baseline (watertight single-body meshes, thin structures preserved: chair legs, starship
  antennae, owl feather relief).

## Boundaries and follow-ups

- Geometry stage only; the official PaintPBR texture stack stays out of scope (CUDA-only
  rasterization kernels). Textured output uses the shared projection bake.
- Single-view projection texturing remains approximate on hidden surfaces and on subjects
  whose proportions the model reinterprets (faces); reference views improve it.
- Remains experimental until the repository's promotion gates and license policy say
  otherwise.
