# Scene3D texture backlog track

## Status

Planned

## Purpose

This track captures the approved engineering path for adding high-quality textured `scene3d`
outputs to `abstract3d` without silently widening the repository into the full upstream Step1X
texture stack.

The approved strategy is intentionally narrow:

- make textured export a first-class `scene3d` contract
- implement the first validated textured path with the official TripoSR bake pipeline
- route both `i23d` and composed `t23d` through that same path
- keep vertex-colored output as the stable default until the textured promotion gate closes
- keep Step1X texture as a separate proposed follow-up until its runtime and policy boundaries are
  approved explicitly

## Items

- `0009_scene3d_textured_asset_contract_and_policy.md`: define texture modes, metadata, artifact
  packaging, CLI/API knobs, and durable boundaries for textured outputs.
- `0010_triposr_official_texture_bake_integration.md`: integrate the official TripoSR atlas-bake
  path and expose texture-resolution controls.
- `0011_textured_glb_packaging_and_uv_safe_cleanup.md`: preserve the GLB-first artifact contract
  while enforcing cleanup order that does not invalidate baked UVs and textures.
- `0012_textured_validation_suite_and_promotion_gate.md`: publish the proof methodology, benchmark
  pack, and promotion gate for textured `i23d` and composed `t23d`.

## Reading Order

1. Read the current validated-backend policy in
   `../../../adr/0002_validated_backend_uses_pinned_triposr_and_composed_t23d.md`.
2. Read the public runtime shape in `../../../architecture.md`.
3. Execute this track in numeric order from `0009` through `0012`.
4. Read `../../proposed/step1x/0002_step1x_texture_stage_research.md` only if work tries to widen
   the approved scope beyond the TripoSR-first track.

## Relevant References

- `src/abstract3d/backends/triposr_runtime.py`
- `src/abstract3d/cli.py`
- `src/abstract3d/rendering.py`
- `src/abstract3d/types.py`
- `scripts/validate_local.py`
- `docs/architecture.md`
- `docs/models.md`
- `docs/benchmarks.md`
- `docs/backlog/proposed/step1x/0002_step1x_texture_stage_research.md`

## Non-Goals

- This track does not authorize shipping the official Step1X texture stack.
- This track does not require a backend-agnostic custom baker before the first textured release.
- This track does not promote full PBR material generation beyond a base-color textured asset.
- This track does not change the current `t23d` composition model away from `text -> image ->
  scene3d`.
