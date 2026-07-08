# Backlog Overview

> Note: `artifacts/validation/` paths cited as evidence in backlog entries may refer to
> the local validation archive. Only the experiments that reflect the current state of
> the code are versioned; superseded experiment bundles are kept locally.

## Project Summary

`abstract3d` provides the `scene3d` capability for AbstractFramework with a local-first backend contract, a primary `glb` artifact, and proof-oriented validation. TripoSR remains the validated default path. High-quality textured output is now an approved planned workstream centered on the official TripoSR bake path, while the Step1X geometry backend remains experimental and Apple-local quality recovery remains active planned work.

## Current Counts

- Planned: 10
- Proposed: 1
- Completed: 3
- Deprecated: 0
- Recurrent: 2

## Priority Now

Active planned work:

1. [0009_scene3d_textured_asset_contract_and_policy.md](planned/scene3d-texture/0009_scene3d_textured_asset_contract_and_policy.md)
2. [0010_triposr_official_texture_bake_integration.md](planned/scene3d-texture/0010_triposr_official_texture_bake_integration.md)
3. [0011_textured_glb_packaging_and_uv_safe_cleanup.md](planned/scene3d-texture/0011_textured_glb_packaging_and_uv_safe_cleanup.md)
4. [0012_textured_validation_suite_and_promotion_gate.md](planned/scene3d-texture/0012_textured_validation_suite_and_promotion_gate.md)
5. [0004_step1x_mlx_reference_harness_and_failure_baseline.md](planned/step1x-mlx/0004_step1x_mlx_reference_harness_and_failure_baseline.md)
6. [0005_step1x_mlx_geometry_component_loader_and_weight_mapping.md](planned/step1x-mlx/0005_step1x_mlx_geometry_component_loader_and_weight_mapping.md)
7. [0006_step1x_mlx_denoiser_attention_port.md](planned/step1x-mlx/0006_step1x_mlx_denoiser_attention_port.md)
8. [0007_step1x_mlx_conditioning_autoencoder_and_mesh_decode.md](planned/step1x-mlx/0007_step1x_mlx_conditioning_autoencoder_and_mesh_decode.md)
9. [0008_step1x_mlx_runtime_integration_validation_and_promotion_gate.md](planned/step1x-mlx/0008_step1x_mlx_runtime_integration_validation_and_promotion_gate.md)
10. [0003_step1x_quality_tuning_on_apple_silicon.md](planned/step1x/0003_step1x_quality_tuning_on_apple_silicon.md)

Recommended next proposed follow-up after that:

1. [0002_step1x_texture_stage_research.md](proposed/step1x/0002_step1x_texture_stage_research.md)

## Topic Tracks

### Completed Tracks

- [step1x](completed/step1x/README.md): delivered Step1X implementation work.
  - [0001_step1x_geometry_backend.md](completed/step1x/0001_step1x_geometry_backend.md)
- hunyuan3d: delivered the license-gated Hunyuan3D-2.1 shape backend, the shared projection texturing module, and the multi-view geometry path.
  - [0013_hunyuan3d21_license_gated_shape_backend.md](completed/hunyuan3d/0013_hunyuan3d21_license_gated_shape_backend.md)
  - [0014_multiview_geometry_and_reference_view_qa.md](completed/hunyuan3d/0014_multiview_geometry_and_reference_view_qa.md)

### Proposed Tracks

- [step1x](proposed/step1x/README.md): Step1X follow-ups that are worth preserving but are not yet approved implementation work.
  - [0002_step1x_texture_stage_research.md](proposed/step1x/0002_step1x_texture_stage_research.md)

### Planned Tracks

- [scene3d-texture](planned/scene3d-texture/README.md): approved high-quality texture support for
  the validated scene3d path, starting with official TripoSR baking and preserving Step1X texture
  as a separate proposed track.
  - [0009_scene3d_textured_asset_contract_and_policy.md](planned/scene3d-texture/0009_scene3d_textured_asset_contract_and_policy.md)
  - [0010_triposr_official_texture_bake_integration.md](planned/scene3d-texture/0010_triposr_official_texture_bake_integration.md)
  - [0011_textured_glb_packaging_and_uv_safe_cleanup.md](planned/scene3d-texture/0011_textured_glb_packaging_and_uv_safe_cleanup.md)
  - [0012_textured_validation_suite_and_promotion_gate.md](planned/scene3d-texture/0012_textured_validation_suite_and_promotion_gate.md)
- [step1x](planned/step1x/README.md): active Step1X Apple-local quality recovery work.
  - [0003_step1x_quality_tuning_on_apple_silicon.md](planned/step1x/0003_step1x_quality_tuning_on_apple_silicon.md)
- [step1x-mlx](planned/step1x-mlx/README.md): approved MLX port track for the official Step1X
  geometry stack on Apple Silicon.
  - [0004_step1x_mlx_reference_harness_and_failure_baseline.md](planned/step1x-mlx/0004_step1x_mlx_reference_harness_and_failure_baseline.md)
  - [0005_step1x_mlx_geometry_component_loader_and_weight_mapping.md](planned/step1x-mlx/0005_step1x_mlx_geometry_component_loader_and_weight_mapping.md)
  - [0006_step1x_mlx_denoiser_attention_port.md](planned/step1x-mlx/0006_step1x_mlx_denoiser_attention_port.md)
  - [0007_step1x_mlx_conditioning_autoencoder_and_mesh_decode.md](planned/step1x-mlx/0007_step1x_mlx_conditioning_autoencoder_and_mesh_decode.md)
  - [0008_step1x_mlx_runtime_integration_validation_and_promotion_gate.md](planned/step1x-mlx/0008_step1x_mlx_runtime_integration_validation_and_promotion_gate.md)

## Planned Ledger

| ID | Title | Path | Status | Comment |
|---|---|---|---|---|
| 0009 | Scene3D textured asset contract and policy | `docs/backlog/planned/scene3d-texture/0009_scene3d_textured_asset_contract_and_policy.md` | Planned | Makes textured output a first-class `scene3d` contract with explicit texture modes, metadata, and scope boundaries while keeping Step1X texture outside the approved path. |
| 0010 | TripoSR official texture bake integration | `docs/backlog/planned/scene3d-texture/0010_triposr_official_texture_bake_integration.md` | Planned | Integrates the upstream TripoSR bake path so both `i23d` and composed `t23d` can emit high-quality textured assets through the validated backend family. |
| 0011 | Textured GLB packaging and UV-safe cleanup | `docs/backlog/planned/scene3d-texture/0011_textured_glb_packaging_and_uv_safe_cleanup.md` | Planned | Preserves the current GLB-first artifact contract while enforcing cleanup ordering that does not invalidate UVs or baked textures. |
| 0012 | Textured validation suite and promotion gate | `docs/backlog/planned/scene3d-texture/0012_textured_validation_suite_and_promotion_gate.md` | Planned | Defines the canonical textured proof methodology, benchmark pack, and evidence-based promotion gate for textured scene3d outputs. |
| 0003 | Step1X quality tuning on Apple Silicon | `docs/backlog/planned/step1x/0003_step1x_quality_tuning_on_apple_silicon.md` | Planned | Export-axis and topology recovery improved, and the validator is now case-isolated and memory-guarded, but promotion is still blocked by remaining chair/owl quality gaps and by an espresso `i23d` case that still exceeds safe Apple-local memory limits. |
| 0004 | Step1X MLX reference harness and failure baseline | `docs/backlog/planned/step1x-mlx/0004_step1x_mlx_reference_harness_and_failure_baseline.md` | Planned | Start here if the repo commits to the MLX lane; this item defines the parity harness and preserved failure baseline needed before low-level math work. |
| 0005 | Step1X MLX geometry component loader and weight mapping | `docs/backlog/planned/step1x-mlx/0005_step1x_mlx_geometry_component_loader_and_weight_mapping.md` | Planned | Builds the MLX-side loader and mapping layer for the official base and label geometry checkpoints. |
| 0006 | Step1X MLX denoiser and attention port | `docs/backlog/planned/step1x-mlx/0006_step1x_mlx_denoiser_attention_port.md` | Planned | Ports the geometry denoiser off the fragile PyTorch `mps` SDPA path and onto MLX-owned execution. |
| 0007 | Step1X MLX conditioning, autoencoder, and mesh decode | `docs/backlog/planned/step1x-mlx/0007_step1x_mlx_conditioning_autoencoder_and_mesh_decode.md` | Planned | Extends the MLX lane beyond denoising so final meshes can be traced through conditioning and decode diagnostics. |
| 0008 | Step1X MLX runtime integration, validation, and promotion gate | `docs/backlog/planned/step1x-mlx/0008_step1x_mlx_runtime_integration_validation_and_promotion_gate.md` | Planned | Wires the MLX lane into `abstract3d` and preserves proof discipline before any promotion decision. |

## Proposed Ledger

| ID | Title | Path | Status | Promotion criteria |
|---|---|---|---|---|
| 0002 | Step1X texture stage research | `docs/backlog/proposed/step1x/0002_step1x_texture_stage_research.md` | Proposed | Promote only if the dependency stack meets the repo license/runtime policy or the policy changes explicitly. |

## Completed Ledger

| ID | Title | Original path | Final path | Completed | Outcome | Key validation |
|---|---|---|---|---|---|---|
| 0001 | Official Step1X-3D geometry backend | `docs/backlog/planned/step1x/0001_step1x_geometry_backend.md` | `docs/backlog/completed/step1x/0001_step1x_geometry_backend.md` | `2026-06-22` | Completed | `pytest -q tests`; targeted `abstractcore` scene3d tests; historical proof assets under `docs/assets/validation/local-step1x/`; refreshed Apple-local reference assets under `docs/assets/validation/local-step1x-label-refresh/` |
| 0013 | Hunyuan3D-2.1 license-gated shape backend with shared projection texturing | created directly as completed work | `docs/backlog/completed/hunyuan3d/0013_hunyuan3d21_license_gated_shape_backend.md` | `2026-07-04` | Completed | `pytest -q tests` (117 tests); Apple-`mps` four-object proof bundles under `artifacts/validation/final-proof/` |
| 0014 | Multi-view geometry conditioning (Hunyuan3D-2mv) and reference-view QA | created directly as completed work | `docs/backlog/completed/hunyuan3d/0014_multiview_geometry_and_reference_view_qa.md` | `2026-07-04` | Completed | `pytest -q tests` (121 tests); Apple-`mps` multi-view face proof under `artifacts/validation/iter2-multiview/face-2mv-final/` |

## Deprecated Ledger

No deprecated backlog items are recorded yet.

## Recurrent Tasks

- [backlog-and-adr-hygiene.md](recurrent/backlog-and-adr-hygiene.md)
- [post-completion-follow-up-triage.md](recurrent/post-completion-follow-up-triage.md)

## Operating Process

### Add New Work

1. Inspect code and docs before writing backlog text.
2. Assign the next unused four-digit global ID.
3. Add or update the item in the correct lifecycle directory.
4. Update this overview in the same change.

### Complete Work

1. Finish implementation, tests, docs, and validation.
2. Update ADR state before closure when the work changes durable policy.
3. Append a completion report to the item.
4. Move it to `completed/` and update counts and ledgers here.

### Deprecate Work

1. Append a deprecation report.
2. Move the item to `deprecated/`.
3. Update counts, ledgers, and links here.

## Planning Notes

- TripoSR remains the validated default backend.
- High-quality texturing shipped through the shared `abstract3d.texturing` projection bake:
  `baked_basecolor@2048` is the live TripoSR default, and the same bake also serves the
  license-gated Hunyuan3D-2.1 backend. The remaining scene3d-texture backlog items should be
  re-triaged against that shipped reality (most of their scope is delivered; what remains is
  the formal promotion-gate bookkeeping in item 0012).
- Step1X is implemented, local, and benchmarked, but its current Apple-local quality and speed profile does not justify promotion beyond experimental status.
- Apple-local Step1X recovery is now active planned work because the original proof was rejected and the refreshed `local-step1x-label-refresh` evidence is still short of a publishable promotion proof.
- Focused Apple-local chair diagnostics now show recovered topology and export-axis behavior, but not enough shape quality to justify promotion.
- The current Step1X direction remains narrower than the full upstream repo: geometry only, official assets only, and no silent adoption of the texture stack.
- Step1X texture remains a separate proposed item because the official texture stack is a different
  runtime and policy class from the approved TripoSR-first texturing track.
- A new MLX track is now planned because the strongest local failures remain upstream of mesh export,
  inside the PyTorch `mps` denoiser/attention path.
- The MLX track is intended to coexist with the current PyTorch Step1X lane until there is proof
  that the MLX route is materially better on quality, completion rate, and memory safety.
