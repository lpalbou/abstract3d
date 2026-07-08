# Step1X backlog track

## Status

Planned

## Purpose

This track preserves active Step1X implementation work that is approved but not yet complete.

## Items

- `0003_step1x_quality_tuning_on_apple_silicon.md`: active Apple-local Step1X quality recovery, proof discipline, and promotion-gate work.

## Reading Order

1. Read the completed implementation record in `../../completed/step1x/0001_step1x_geometry_backend.md`.
2. Execute `0003_step1x_quality_tuning_on_apple_silicon.md`.
3. Review remaining proposals in `../../proposed/step1x/`.

## Relevant References

- `src/abstract3d/backends/`
- `src/abstract3d/scene3d_manager.py`
- `src/abstract3d/integrations/abstractcore_plugin.py`
- `docs/architecture.md`
- `docs/models.md`
- `docs/adr/0001_scene3d_local_first_glb_contract.md`
- `docs/adr/0002_validated_backend_uses_pinned_triposr_and_composed_t23d.md`
- `docs/adr/0003_trellis2_uses_official_upstream_assets_only.md`
- `docs/adr/0004_step1x_geometry_only_official_backend_with_local_compatibility_patches.md`

## Non-Goals

- This track does not authorize shipping the full Step1X texture stack without separate evidence and policy review.
