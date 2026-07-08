# Planned: Step1X MLX reference harness and failure baseline

## Metadata

- Created: 2026-06-23
- Status: Planned
- Completed: N/A

## ADR status

- Governing ADRs: [ADR 0004](../../../adr/0004_step1x_geometry_only_official_backend_with_local_compatibility_patches.md)
- ADR impact: None

## Context

The current Step1X implementation is real, local, and official-only, but its Apple-local PyTorch
`mps` path still fails at the quality and stability bar required for promotion.

## Current code reality

- `src/abstract3d/backends/step1x_runtime.py` currently runs the official Step1X geometry pipeline
  through PyTorch, with Apple-specific caps, CPU-side surface extraction, and export cleanup.
- The stronger Apple-local runs fail before mesh export with `MPS backend out of memory`, preserved
  in `artifacts/validation/rocket-i23d/step1x/result.json` and
  `artifacts/validation/rocket-i23d/step1x-base-stable/result.json`.
- The vendored official Step1X attention path still calls
  `scaled_dot_product_attention(...)` directly from
  `~/.cache/abstract3d/vendor/step1x/.../step1x3d_geometry/models/attention_processor.py`.
- The upstream official repo documents CUDA-first inference and heavier reference settings in
  `app.py` and `inference.py`, while our local runtime is intentionally much more conservative on
  Apple `mps`.
- No MLX runtime for Step1X exists in `abstract3d` today.
- `/Users/albou/projects/gh/sbx/mlx-gen` already contains reusable MLX patterns for attention math,
  weight loading, and HF-to-MLX weight mapping.

## Problem

We do not yet have a deterministic reference harness that can tell a future MLX port whether it is
numerically wrong, merely slower, or actually closer to the intended upstream geometry behavior.

## What we want to do

Create the reference harness and failure baseline first, so the MLX port is validated against real
golden artifacts and not against memory of what the model "should" look like.

## Why

Without a reference harness, low-level ports drift silently. A denoiser port can appear to "work"
while numerically diverging enough to damage geometry quality.

## Requirements

- capture exact official Step1X source revision, subfolder, and runtime parameters used for each
  reference case
- preserve a small canonical benchmark set:
  - the rocket `i23d` case
  - at least one symmetric furniture case
  - at least one sharp asymmetric product case
  - at least one composed `t23d` case
- record both successful and failed PyTorch paths
- define parity checkpoints beyond final mesh:
  - conditioning embeddings
  - denoiser latent snapshots at selected timesteps
  - decoded field or grid statistics before marching cubes
- keep the harness local-only and official-only

## Suggested implementation

- add a dedicated `scripts/step1x_mlx_reference_harness.py`
- emit structured bundles with:
  - source image
  - normalized preprocessing output
  - runtime request
  - selected intermediate tensors or aggregate statistics
  - final mesh stats
  - contact sheet
- support `capture-mode=stats-only` first so the harness does not explode storage usage
- add a comparison mode that can diff PyTorch `mps`, PyTorch `cpu`, and future `mlx` rows

## Scope

- reference harness creation
- benchmark-case curation
- failure-baseline capture
- reproducibility metadata

## Non-goals

- porting model math in this item
- changing default backend routing in this item
- publishing success claims in this item

## Dependencies and related tasks

- `src/abstract3d/backends/step1x_runtime.py`
- `scripts/step1x_extract_vae.py`
- `docs/assets/validation/local-step1x-dynamic/`
- `artifacts/validation/rocket-i23d/`
- `../abstractvision`
- `/Users/albou/projects/gh/sbx/mlx-gen`
- [0005_step1x_mlx_geometry_component_loader_and_weight_mapping.md](0005_step1x_mlx_geometry_component_loader_and_weight_mapping.md)

## Expected outcomes

- a deterministic benchmark pack exists for Step1X geometry work on Apple Silicon
- future MLX work can compare intermediate states instead of guessing
- the repo has one clear place to look for "known-bad" and "known-good-enough" Step1X rows

## Validation

- run the harness on at least two existing failing Apple-local cases and one successful reduced
  case
- verify the harness can resume without rerunning completed rows
- verify emitted metadata includes source snapshot, model subfolder, timings, and selected tensor
  stats
- run `pytest -q tests`

## Progress checklist

- [ ] define the canonical Step1X benchmark pack
- [ ] implement the harness entrypoint and artifact schema
- [ ] capture current PyTorch `mps` baseline rows
- [ ] capture at least one CPU reference row where feasible
- [ ] document how later MLX rows should be compared

## Guidance for the implementing agent

Do not start the MLX math port before the harness exists. If storage pressure becomes a problem,
prefer stable aggregate tensor statistics over full raw tensor dumps.
