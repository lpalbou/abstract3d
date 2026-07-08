# Planned: Textured validation suite and promotion gate

## Metadata

- Created: 2026-06-23
- Status: Planned
- Completed: N/A

## ADR status

- Governing ADRs: [ADR 0001](../../../adr/0001_scene3d_local_first_glb_contract.md), [ADR 0002](../../../adr/0002_validated_backend_uses_pinned_triposr_and_composed_t23d.md)
- ADR impact: None

## Context

High-quality textured output cannot be claimed credibly without proof assets that show both
geometry and texture quality under fixed viewpoints and recorded runtime costs.

## Current code reality

- `scripts/validate_local.py` already supports case-oriented validation, subprocess isolation, and
  docs-ready publication for current geometry-focused proof lanes.
- Existing contact sheets focus on source image plus rendered mesh views, vertex/face counts, and
  runtime timings.
- There is no canonical textured benchmark pack, no atlas preview convention, and no textured
  promotion gate today.

## Problem

Without a textured validation contract, texture work can regress silently or be claimed as
production-ready based on a few cherry-picked renders.

## What we want to do

Create a textured proof methodology that is simple enough to run repeatedly and strict enough to
gate promotion.

## Why

Texture quality is especially easy to overclaim. The proof surface needs to show the input image,
texture atlas, consistent object views, and measured runtime/memory data for both `i23d` and
composed `t23d`.

## Requirements

- preserve the same camera viewpoints across comparison runs
- validate both `i23d` and composed `t23d`
- show source-image provenance for `t23d`
- record texture resolution, timings, file sizes, and memory data
- preserve failure cases rather than dropping them from reports

## Suggested implementation

- extend the validator to support textured runs with explicit presets:
  - `smoke`: `1024`
  - `quality`: `2048`
  - `hero`: `4096`
- define a canonical textured benchmark pack with at least:
  - one smooth/ceramic object
  - one furniture object
  - one hard-surface product
  - one organic figurine
- extend contact sheets to include:
  - source image
  - for `t23d`, generated source image
  - identical geometry/textured view angles
  - texture atlas panel
  - key parameters and runtime stats
- add hard checks:
  - UVs present
  - texture image present
  - atlas dimensions match request
  - textured GLB reload succeeds
  - no missing required bundle artifacts
- define promotion criteria:
  - the backend cannot be called "validated textured" until the canonical benchmark pack passes the
    textured proof lane

## Scope

- textured validation harness work
- contact-sheet and summary conventions
- benchmark-case curation
- promotion gate definition

## Non-goals

- benchmarking unofficial backends in this item
- defining a universal scene3d perceptual metric in this item
- hiding failed textured cases from the published proof surface

## Dependencies and related tasks

- `scripts/validate_local.py`
- `src/abstract3d/rendering.py`
- `docs/benchmarks.md`
- `docs/assets/validation/`
- [0010_triposr_official_texture_bake_integration.md](0010_triposr_official_texture_bake_integration.md)
- [0011_textured_glb_packaging_and_uv_safe_cleanup.md](0011_textured_glb_packaging_and_uv_safe_cleanup.md)

## Expected outcomes

- textured runs have a repeatable proof surface instead of ad hoc screenshots
- quality, latency, and memory tradeoffs are visible for `1024`, `2048`, and `4096`
- promotion decisions become evidence-based instead of narrative-based

## Validation

- run at least one textured `i23d` case and one textured `t23d` case through the new harness
- verify failure rows still emit a bundle and summary entry
- verify summaries capture texture resolution, timings, and asset sizes
- run `pytest -q tests`

## Progress checklist

- [ ] define the canonical textured benchmark pack
- [ ] add textured presets and artifact checks to the validator
- [ ] extend the contact-sheet layout for atlas-aware proof
- [ ] publish at least one docs-ready textured comparison package
- [ ] define the promotion rule for a validated textured backend

## Guidance for the implementing agent

Do not publish textured proof with inconsistent camera poses or without the atlas panel. Reviewers
need to see both the object and the texture evidence in one package.
