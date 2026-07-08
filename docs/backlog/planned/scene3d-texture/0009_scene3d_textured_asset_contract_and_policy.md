# Planned: Scene3D textured asset contract and policy

## Metadata

- Created: 2026-06-23
- Status: Planned
- Completed: N/A

## ADR status

- Governing ADRs: [ADR 0001](../../../adr/0001_scene3d_local_first_glb_contract.md), [ADR 0002](../../../adr/0002_validated_backend_uses_pinned_triposr_and_composed_t23d.md)
- ADR impact: May revise existing ADR

## Context

`abstract3d` currently validates a local TripoSR path and exports `scene3d` bundles centered on a
primary `glb` artifact, preview renders, and metadata.

Official upstream TripoSR already supports texture baking through `--bake-texture` and
`--texture-resolution`, but the current `abstract3d` runtime still exports vertex-colored meshes
only. The repository also carries an explicit non-goal against silently widening Step1X from
geometry-only into the full textured stack.

## Current code reality

- `src/abstract3d/backends/triposr_runtime.py` currently extracts meshes with vertex colors and
  writes `glb`, `obj`, previews, and `metadata.json`.
- `src/abstract3d/cli.py` exposes geometry cleanup controls but no texture-mode or
  texture-resolution controls.
- `docs/architecture.md` defines a `glb`-first artifact contract and explicitly keeps Step1X
  texture out of scope.
- `docs/models.md` keeps Step1X texture outside the supported backend boundary.
- The TripoSR upstream repo documents a baked-texture mode and a `2048` default texture atlas
  resolution, implemented with `xatlas` and `moderngl`.

## Problem

The repository has no first-class contract for textured `scene3d` outputs, so future texture work
would otherwise be forced into ad hoc flags, ambiguous bundle contents, and backend-specific
behavior.

## What we want to do

Define a durable, minimal textured-output contract before integrating any backend-specific bake
path.

## Why

If textured export is not made explicit now, later runtime work will either break the current
GLB-first contract or implicitly tie the public surface to one backend's internal format.

## Requirements

- keep textured output local-first
- preserve the current `glb`-centric artifact contract
- support both `image_to_scene3d` and composed `text_to_scene3d`
- keep a fast/raw mode for vertex-color exports
- keep vertex-colored output as the stable default until textured proof closes
- keep Step1X texture outside the approved path unless policy changes explicitly
- keep the first material model simple enough to validate and ship

## Suggested implementation

- add an explicit texture surface to the public contract:
  - `texture_mode=vertex_color|baked_basecolor`
  - `texture_resolution` with `2048` as the quality default and `4096` as an explicit hero mode
  - `texture_device` or equivalent policy surface for local bake execution
- define textured metadata keys:
  - `texture_mode`
  - `texture_resolution`
  - `texture_artifacts`
  - `texture_timings_s`
  - `texture_warnings`
  - `uv_present`
  - `material_count`
- keep the v1 textured material model intentionally narrow:
  - one UV set
  - one base-color atlas
  - no full PBR stack yet
- preserve provenance for `t23d`:
  - prompt
  - generated source image
  - textured result
- keep the staged internal boundary explicit:
  - `geometry -> optional texture -> bundle`
- require explicit notes when the runtime falls back from baked texture to vertex colors

## Scope

- public scene3d texture contract
- CLI/API parameter surface
- artifact and metadata schema
- durable scope boundaries for textured backends

## Non-goals

- implementing a backend-agnostic custom baker in this item
- authorizing Step1X texture shipping in this item
- adding roughness, metallic, normal, or multi-material authoring in this item

## Dependencies and related tasks

- `src/abstract3d/backends/triposr_runtime.py`
- `src/abstract3d/cli.py`
- `src/abstract3d/types.py`
- `docs/architecture.md`
- `docs/models.md`
- [0010_triposr_official_texture_bake_integration.md](0010_triposr_official_texture_bake_integration.md)
- [0002_step1x_texture_stage_research.md](../../proposed/step1x/0002_step1x_texture_stage_research.md)

## Expected outcomes

- textured outputs become an explicit part of the `scene3d` surface rather than a hidden backend
  behavior
- future backend work can target a stable contract
- Step1X texture remains clearly outside the approved scope until promoted separately

## Validation

- update the public operation schemas and verify they can represent textured and raw modes
- verify bundle metadata can distinguish `vertex_color` and `baked_basecolor`
- run `pytest -q tests`

## Progress checklist

- [ ] define texture modes and defaults
- [ ] define textured metadata keys and bundle expectations
- [ ] define fallback labeling for degraded runs
- [ ] document the explicit Step1X texture boundary in the contract

## Guidance for the implementing agent

Keep the v1 contract small. The first goal is a reliable textured asset surface, not a universal
material authoring system.
