# Planned: Step1X MLX denoiser and attention port

## Metadata

- Created: 2026-06-23
- Status: Planned
- Completed: N/A

## ADR status

- Governing ADRs: [ADR 0004](../../../adr/0004_step1x_geometry_only_official_backend_with_local_compatibility_patches.md)
- ADR impact: None

## Context

The strongest evidence so far points to Step1X Apple-local instability in the denoiser attention
path, not primarily in mesh export.

## Current code reality

- The current vendored Step1X geometry attention path calls
  `scaled_dot_product_attention(query, key, value, ...)` directly in PyTorch.
- The local machine is running `torch 2.8.0`, and there are current PyTorch issues involving MPS
  SDPA memory behavior and non-contiguous-tensor correctness.
- `step1x_runtime.py` already makes several Apple-local concessions:
  - lower guidance defaults
  - smaller face budgets
  - CPU-side surface extraction
  - explicit MPS memory caps
  - runtime offload before export
- `/Users/albou/projects/gh/sbx/mlx-gen` contains a working MLX attention utility layer, including
  QKV reshaping, optional float32 norm steps for stability, and MLX SDPA calls.
- No Step1X geometry denoiser blocks exist in MLX today.

## Problem

The PyTorch `mps` lane is too fragile in the exact area Step1X depends on most heavily: repeated
attention-based denoising over geometry latents.

## What we want to do

Port the Step1X geometry denoiser and its attention math to MLX so Apple-local inference no longer
depends on the fragile PyTorch `mps` SDPA path.

## Why

If the denoiser stays on PyTorch `mps`, memory and correctness risks remain upstream of any mesh
cleanup. That means geometry quality will keep collapsing before the export stage can help.

## Requirements

- reproduce the official denoiser architecture closely enough to compare against the reference
  harness
- preserve rotary / positional behavior
- preserve conditioning injection semantics
- support explicit manual-attention fallback in MLX if the fused MLX path diverges
- record per-step timings and peak-memory snapshots
- keep one deterministic seed path for parity debugging

## Suggested implementation

- port one attention block first and compare against PyTorch reference tensors
- then port the full denoiser stack block-by-block
- add an execution flag that can switch between:
  - MLX fused attention
  - explicit matmul-softmax-matmul attention
- treat contiguity, dtype promotion, and norm precision as first-class correctness concerns

## Scope

- denoiser module port
- attention implementation
- scheduler-step execution around the denoiser
- parity diagnostics for intermediate hidden states

## Non-goals

- texture generation
- UI or docs polish
- changing default public routing in this item

## Dependencies and related tasks

- [0004_step1x_mlx_reference_harness_and_failure_baseline.md](0004_step1x_mlx_reference_harness_and_failure_baseline.md)
- [0005_step1x_mlx_geometry_component_loader_and_weight_mapping.md](0005_step1x_mlx_geometry_component_loader_and_weight_mapping.md)
- `/Users/albou/projects/gh/sbx/mlx-gen/src/mflux/models/flux/model/flux_transformer/common/attention_utils.py`
- `src/abstract3d/backends/step1x_runtime.py`

## Expected outcomes

- the Step1X denoiser can execute locally through MLX on Apple Silicon
- denoiser parity can be measured against reference checkpoints instead of judged only by final mesh
- the known PyTorch `mps` SDPA failure path is no longer on the critical path for the MLX lane

## Validation

- compare single-block and multi-block outputs against the reference harness within explicit error
  tolerances
- compare selected denoising-timestep latent statistics across PyTorch and MLX runs
- verify the MLX lane can finish at least the rocket case without PyTorch `mps`
- run `pytest -q tests`

## Progress checklist

- [ ] port one denoiser block and prove parity
- [ ] implement MLX attention execution with a manual fallback mode
- [ ] port the full denoiser stack
- [ ] add per-step memory and timing instrumentation
- [ ] finish one end-to-end denoising trajectory through MLX

## Guidance for the implementing agent

Treat this as a numerical-parity task first, not a UX task. If final meshes are bad, stop and
compare intermediate latent statistics before tuning downstream extraction.
