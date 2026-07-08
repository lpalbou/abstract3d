# Planned: Step1X MLX geometry component loader and weight mapping

## Metadata

- Created: 2026-06-23
- Status: Planned
- Completed: N/A

## ADR status

- Governing ADRs: [ADR 0004](../../../adr/0004_step1x_geometry_only_official_backend_with_local_compatibility_patches.md)
- ADR impact: None

## Context

The current local Step1X path loads the official geometry checkpoints through PyTorch and the
official upstream code. An MLX runtime will need its own loader surface and internal component
structure.

## Current code reality

- The current `abstract3d` Step1X integration relies on the official source snapshot plus local
  compatibility patches under `src/abstract3d/backends/step1x_runtime.py`.
- The upstream Step1X geometry stack is composed of at least:
  - the `FluxDenoiser` geometry transformer
  - the `Dinov2CLIPEncoder` visual condition model
  - the `LabelEncoder` for the label checkpoint
  - the `MichelangeloAutoencoder` shape model / decoder
- `/Users/albou/projects/gh/sbx/mlx-gen` already ships reusable MLX patterns for:
  - loading HF or prepared weights into MLX trees
  - declarative key mapping from HF names into MLX structures
  - retaining local metadata about quantization and saved layouts
- No Step1X-specific MLX component definitions or weight maps exist yet.

## Problem

We cannot port inference cleanly until the official checkpoint family can be loaded into stable MLX
component trees with explicit, testable mappings.

## What we want to do

Build the MLX-side loader and component definitions for the official Step1X geometry checkpoints.

## Why

Bad weight mapping causes silent correctness failures that look like "poor model quality." The
loader layer needs to be explicit and testable before denoising logic is trusted.

## Requirements

- support the official base geometry checkpoint
- support the official label geometry checkpoint
- keep checkpoint provenance pinned to official upstream repo ids
- support a local snapshot cache without remote-code execution
- support partial loading by component for debugging:
  - visual encoder only
  - label encoder only
  - denoiser only
  - shape autoencoder only
- provide loader diagnostics for missing or unmapped weights

## Suggested implementation

- create a new MLX-oriented backend module tree under `src/abstract3d/backends/step1x_mlx/`
- define MLX component schemas modeled after the structure used in `mlx-gen`
- add explicit HF-to-MLX key mappings per component, with transform hooks where tensor layout differs
- add a loader report that records:
  - total keys loaded
  - keys mapped
  - keys intentionally skipped
  - keys left unresolved

## Scope

- MLX component definitions
- weight loading and mapping
- provenance checks for official checkpoints
- component-level loader tests

## Non-goals

- full denoising execution in this item
- mesh-quality tuning in this item
- texture-stage loading in this item

## Dependencies and related tasks

- [0004_step1x_mlx_reference_harness_and_failure_baseline.md](0004_step1x_mlx_reference_harness_and_failure_baseline.md)
- `/Users/albou/projects/gh/sbx/mlx-gen/src/mflux/models/common/weights/loading/weight_loader.py`
- `/Users/albou/projects/gh/sbx/mlx-gen/src/mflux/models/common/weights/mapping/weight_mapper.py`
- `src/abstract3d/backends/step1x_runtime.py`

## Expected outcomes

- official Step1X geometry checkpoints load into explicit MLX component trees
- mapping coverage is measurable and regression-testable
- future denoiser/conditioner work can assume stable component boundaries

## Validation

- add unit tests for key-mapping coverage and expected transforms
- verify both base and label checkpoints can be resolved into MLX structures from a local cache
- verify unresolved-key reporting is readable and stable
- run `pytest -q tests`

## Progress checklist

- [ ] define MLX component boundaries for Step1X geometry
- [ ] implement official base checkpoint loader
- [ ] implement official label checkpoint loader
- [ ] add loader diagnostics and mapping-coverage tests
- [ ] document required local snapshot layout

## Guidance for the implementing agent

Do not bury loader transforms inside ad hoc inference code. Keep the mapping layer separable so it
can be audited independently when geometry quality looks wrong.
