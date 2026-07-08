# Planned: Textured GLB packaging and UV-safe cleanup

## Metadata

- Created: 2026-06-23
- Status: Planned
- Completed: N/A

## ADR status

- Governing ADRs: [ADR 0001](../../../adr/0001_scene3d_local_first_glb_contract.md), [ADR 0002](../../../adr/0002_validated_backend_uses_pinned_triposr_and_composed_t23d.md)
- ADR impact: None

## Context

The repository's public contract is `glb`-first, but the official TripoSR texture bake path is
atlas-oriented and produces UV data plus a texture image. Geometry cleanup also matters because
topology-changing edits after UV generation can invalidate the baked result.

## Current code reality

- `_write_bundle(...)` in `src/abstract3d/backends/triposr_runtime.py` currently assumes one
  primary mesh payload, one `scene.obj`, preview renders, and `metadata.json`.
- The current TripoSR cleanup pass uses PyMeshLab operations that can change topology:
  - small-component pruning
  - marching-cube cleanup
  - smoothing
  - hole repair
  - vertex merging
- The current bundle contract has no slots for:
  - texture atlas images
  - UV previews
  - material manifests
  - raw baked intermediates

## Problem

If texture baking is added without explicit packaging and cleanup ordering, `abstract3d` risks
either breaking the current `glb` contract or silently producing textured assets whose UVs no
longer match the exported geometry.

## What we want to do

Define and implement a textured bundle/export path that preserves the current `glb`-first contract
while enforcing UV-safe cleanup order.

## Why

The geometry-quality work already modifies topology. That is acceptable before UV unwrap and bake,
but not after unless the system rebakes or labels the result as degraded.

## Requirements

- keep `glb` as the primary artifact for the public contract
- preserve access to raw baked atlas outputs for debugging and proof
- avoid topology-changing cleanup after UV generation unless the runtime rebakes
- make the cleanup order explicit in metadata

## Suggested implementation

- split TripoSR cleanup into two phases:
  - `pre_bake_geometry_cleanup`
  - `post_bake_safe_finalize`
- allow topology-changing operations only before UV unwrap and bake
- restrict post-bake operations to UV-safe work such as:
  - normal recomputation
  - metadata updates
  - packaging
- if later simplification or topology edits are requested after bake:
  - rebake automatically, or
  - reject the request with an explicit warning
- extend the bundle layout to include:
  - primary textured `scene.glb`
  - optional raw `scene.obj`
  - `texture.png`
  - `uv_preview.png`
  - `material.json` or equivalent structured metadata if useful
  - `metadata.json`
- keep source input and contact sheets in the same bundle so texture proof stays self-contained

## Scope

- textured export packaging
- cleanup-order enforcement
- UV/material debug artifacts
- metadata for textured bundles

## Non-goals

- implementing a generic mesh-reduction system after bake in this item
- replacing the current geometry cleanup heuristics wholesale in this item
- broadening the public contract to full multi-map PBR in this item

## Dependencies and related tasks

- `src/abstract3d/backends/triposr_runtime.py`
- `src/abstract3d/rendering.py`
- `src/abstract3d/types.py`
- [0010_triposr_official_texture_bake_integration.md](0010_triposr_official_texture_bake_integration.md)
- [0012_textured_validation_suite_and_promotion_gate.md](0012_textured_validation_suite_and_promotion_gate.md)

## Expected outcomes

- textured bundles remain consistent with the geometry they present
- the repository preserves its `glb`-first public surface while still exposing useful bake
  diagnostics
- future agents can see exactly which cleanup stages are safe before and after UV generation

## Validation

- verify textured exports reload successfully in a GLB-capable viewer
- verify UV/atlas artifacts are emitted for textured runs
- verify metadata records pre-bake and post-bake cleanup stages separately
- run `pytest -q tests`

## Progress checklist

- [ ] split cleanup ordering into pre-bake and post-bake stages
- [ ] define textured bundle contents
- [ ] add UV/atlas diagnostic artifacts
- [ ] reject or rebake topology-changing post-bake requests

## Guidance for the implementing agent

Treat UV generation as a boundary. Once UVs and textures exist, topology-changing cleanup is no
longer "free."
