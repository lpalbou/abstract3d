# Planned: TripoSR official texture bake integration

## Metadata

- Created: 2026-06-23
- Status: Planned
- Completed: N/A

## ADR status

- Governing ADRs: [ADR 0001](../../../adr/0001_scene3d_local_first_glb_contract.md), [ADR 0002](../../../adr/0002_validated_backend_uses_pinned_triposr_and_composed_t23d.md)
- ADR impact: None

## Context

The cleanest approved path to textured `scene3d` output is the official upstream TripoSR bake
pipeline rather than an immediate jump to a generic baker or to the full Step1X texture stack.

## Current code reality

- Update 2026-07-04: this item's core scope has shipped. The TripoSR default is now
  `texture_mode=baked_basecolor` at `texture_resolution=2048`, routed through the shared
  projection bake in `src/abstract3d/texturing.py` with the TripoSR triplane field as the
  unseen-texel prior. The notes below describe the pre-delivery state and are preserved for
  history.
- `src/abstract3d/backends/triposr_runtime.py` previously called `extract_mesh(..., True, ...)`
  and exported vertex-colored meshes only.
- The pinned upstream TripoSR repo ships:
  - `run.py` with `--bake-texture`
  - `--texture-resolution` defaulting to `2048`
  - `tsr/bake_texture.py` using `xatlas` and a standalone `moderngl` context to rasterize a UV
    atlas and query colors from the TripoSR field
- The current runtime dependency gate in `triposr_runtime.py` does not include the full upstream
  bake dependency set.

## Problem

The validated backend already has an official texture path upstream, but `abstract3d` does not yet
load, control, or expose it.

## What we want to do

Integrate the official TripoSR texture bake path into `abstract3d` as the first approved textured
backend implementation.

## Why

Using the upstream-supported bake path is the lowest-risk way to improve texture quality while
staying inside the current validated backend family.

## Requirements

- preserve local execution
- keep TripoSR as the first validated textured backend
- support both `i23d` and composed `t23d`
- expose explicit texture-resolution control
- preserve safe fallback behavior when bake dependencies or runtime context are unavailable

## Suggested implementation

- extend the TripoSR runtime dependency gate to cover the upstream bake path:
  - `xatlas`
  - `moderngl`
- add runtime controls:
  - `texture_mode`
  - `texture_resolution`
  - optional bake-device/runtime policy knobs when needed
- keep the baked path opt-in until the textured proof item closes successfully
- change the TripoSR sequence from:
  - `extract mesh -> cleanup -> export vertex-color mesh`
  to:
  - `extract mesh -> geometry cleanup -> UV unwrap and bake -> export textured asset bundle`
- preserve a raw/debug mode that still exports vertex colors without baking
- capture bake timings and warnings separately from geometry timings
- keep `t23d` on the same path by feeding the generated source image into the same textured TripoSR
  pipeline

## Scope

- TripoSR runtime integration
- runtime dependency expansion
- new texture controls and metadata
- `t23d` and `i23d` textured routing through TripoSR

## Non-goals

- replacing the official TripoSR bake math in this item
- integrating Step1X texture in this item
- solving full PBR authoring in this item

## Dependencies and related tasks

- `src/abstract3d/backends/triposr_runtime.py`
- `src/abstract3d/cli.py`
- `src/abstract3d/integrations/abstractcore_plugin.py`
- `pyproject.toml`
- [0009_scene3d_textured_asset_contract_and_policy.md](0009_scene3d_textured_asset_contract_and_policy.md)
- [0011_textured_glb_packaging_and_uv_safe_cleanup.md](0011_textured_glb_packaging_and_uv_safe_cleanup.md)

## Expected outcomes

- TripoSR can emit a baked-texture path through the public `abstract3d` surface
- the same texture path works for `i23d` and composed `t23d`
- degraded runs can fail explicitly or fall back explicitly instead of silently exporting raw meshes
- the current stable default can remain vertex-colored output until textured proof justifies
  promotion

## Validation

- generate one textured `i23d` case and one textured `t23d` case locally
- verify texture-resolution settings are reflected in emitted metadata and artifacts
- run `pytest -q tests`

## Progress checklist

- [ ] add bake-path dependency checks
- [ ] add texture controls to the TripoSR runtime and CLI surface
- [ ] integrate official bake flow after geometry cleanup
- [ ] preserve explicit raw-mode fallback behavior
- [ ] record bake timings and warnings in metadata

## Guidance for the implementing agent

Stay close to the upstream TripoSR bake path first. Do not redesign the math unless the official
path proves incompatible with the target runtime.
