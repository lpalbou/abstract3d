# Planned: Step1X MLX runtime integration, validation, and promotion gate

## Metadata

- Created: 2026-06-23
- Status: Planned
- Completed: N/A

## ADR status

- Governing ADRs: [ADR 0004](../../../adr/0004_step1x_geometry_only_official_backend_with_local_compatibility_patches.md)
- ADR impact: None

## Context

Even if the MLX port works in isolation, it is not finished until it is wired into `abstract3d`
with residency controls, artifact outputs, docs, and a proof gate that can justify real use.

## Current code reality

- `abstract3d` currently exposes Step1X through the experimental PyTorch backend path only.
- The package already knows how to:
  - package artifact bundles
  - emit contact sheets
  - route through `Scene3DManager`
  - expose backend residency state to `abstractcore`
- The repo already has a proof discipline for publishing or withholding validation assets.
- No `abstract3d:step1x-mlx` backend id or MLX-specific runtime integration exists yet.

## Problem

Without explicit integration and proof gates, a future MLX prototype could either remain unusable
or be promoted too early.

## What we want to do

Integrate the MLX Step1X path into `abstract3d` as an explicit backend with validation and
promotion rules.

## Why

The port only matters if users can actually run it locally, inspect its outputs, and understand its
status relative to TripoSR.

## Requirements

- add a distinct backend id for the MLX Step1X lane
- keep the existing PyTorch Step1X lane available for comparison until the MLX lane is proven
- expose MLX-specific controls for:
  - memory limits
  - cache release
  - quantization or precision policy
  - component offload policy
- preserve the current artifact bundle contract
- publish docs-ready proof assets only after visual review
- define promotion thresholds explicitly:
  - quality
  - completion rate
  - memory safety
  - latency

## Suggested implementation

- implement `abstract3d:step1x-mlx` as a separate backend or sub-backend under the Step1X family
- add backend-selection support in CLI and manager surfaces
- extend the validation harness to compare:
  - TripoSR
  - Step1X PyTorch `mps`
  - Step1X MLX
- require a benchmark/contact-sheet package before any status change beyond experimental

## Scope

- backend integration
- CLI and manager routing
- residency and memory controls
- proof harness integration
- docs-facing proof packaging

## Non-goals

- removing TripoSR as the validated default in this item
- hiding failed MLX cases
- texture-stage promotion

## Dependencies and related tasks

- [0004_step1x_mlx_reference_harness_and_failure_baseline.md](0004_step1x_mlx_reference_harness_and_failure_baseline.md)
- [0005_step1x_mlx_geometry_component_loader_and_weight_mapping.md](0005_step1x_mlx_geometry_component_loader_and_weight_mapping.md)
- [0006_step1x_mlx_denoiser_attention_port.md](0006_step1x_mlx_denoiser_attention_port.md)
- [0007_step1x_mlx_conditioning_autoencoder_and_mesh_decode.md](0007_step1x_mlx_conditioning_autoencoder_and_mesh_decode.md)
- `src/abstract3d/scene3d_manager.py`
- `src/abstract3d/cli.py`
- `scripts/validate_local.py`

## Expected outcomes

- the MLX Step1X lane is runnable through the public `abstract3d` surface
- proof assets can compare MLX against TripoSR and the existing PyTorch lane
- the repo has an explicit promotion gate instead of hand-wavy status changes

## Validation

- add targeted unit tests for backend selection and metadata surfacing
- extend the validator to emit a docs-ready comparison package
- rerun the canonical benchmark set and preserve all rows, including failures
- run `pytest -q tests`
- run targeted `abstractcore` scene3d routing tests

## Progress checklist

- [ ] add an explicit MLX Step1X backend surface
- [ ] expose MLX runtime controls through manager and CLI
- [ ] extend the validator to compare MLX against existing lanes
- [ ] publish a proof package only if the MLX lane clears the promotion thresholds
- [ ] decide whether MLX stays experimental or earns a stronger status

## Guidance for the implementing agent

Promotion is an evidence problem, not a code-completion problem. If the MLX lane still produces
bad geometry, keep it experimental and publish that fact clearly.
