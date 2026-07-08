# Step1X MLX backlog track

## Status

Planned

## Purpose

This track captures the approved engineering path for running the official Step1X geometry stack on
Apple Silicon through MLX rather than continuing to rely exclusively on the current PyTorch `mps`
lane.

## Items

- `0004_step1x_mlx_reference_harness_and_failure_baseline.md`: establish the reproducible
  reference harness, failure baseline, and parity targets before porting math.
- `0005_step1x_mlx_geometry_component_loader_and_weight_mapping.md`: build the component loader
  and weight-mapping layer for the official Step1X geometry checkpoints.
- `0006_step1x_mlx_denoiser_attention_port.md`: port the Step1X geometry denoiser and its
  attention math onto MLX.
- `0007_step1x_mlx_conditioning_autoencoder_and_mesh_decode.md`: port image conditioning,
  label-conditioning, autoencoder decode, and safe mesh extraction.
- `0008_step1x_mlx_runtime_integration_validation_and_promotion_gate.md`: integrate the MLX path
  into `abstract3d`, add controls, and enforce proof-based promotion gates.

## Reading Order

1. Read the current completed implementation record in
   `../../completed/step1x/0001_step1x_geometry_backend.md`.
2. Read the active Apple-local PyTorch recovery item in
   `../step1x/0003_step1x_quality_tuning_on_apple_silicon.md`.
3. Execute this track in numeric order from `0004` through `0008`.

## Relevant References

- `src/abstract3d/backends/step1x_runtime.py`
- `scripts/step1x_extract_vae.py`
- `artifacts/validation/rocket-i23d/step1x/`
- `artifacts/validation/rocket-i23d/step1x-base-stable/`
- `docs/assets/validation/local-step1x-dynamic/`
- `../abstractvision`
- `/Users/albou/projects/gh/sbx/mlx-gen`
- `docs/adr/0004_step1x_geometry_only_official_backend_with_local_compatibility_patches.md`

## Non-Goals

- This track does not authorize switching to unofficial Step1X checkpoints.
- This track does not authorize shipping the Step1X texture stack.
- This track does not deprecate TripoSR as the validated default unless a later proof item closes
  successfully.
