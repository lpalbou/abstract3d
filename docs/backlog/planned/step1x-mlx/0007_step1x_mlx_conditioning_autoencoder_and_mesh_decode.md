# Planned: Step1X MLX conditioning, autoencoder, and mesh decode

## Metadata

- Created: 2026-06-23
- Status: Planned
- Completed: N/A

## ADR status

- Governing ADRs: [ADR 0004](../../../adr/0004_step1x_geometry_only_official_backend_with_local_compatibility_patches.md)
- ADR impact: None

## Context

A denoiser port alone is not sufficient. Step1X geometry quality also depends on faithful image
conditioning, label injection, and shape decode into a field that can be meshed safely.

## Current code reality

- The current PyTorch integration still delegates image conditioning and label-conditioning to the
  official upstream runtime.
- The local runtime already contains custom label heuristics for the label checkpoint, and those
  heuristics are part of the current Apple-local operating policy.
- `scripts/step1x_extract_vae.py` exists as a subprocess-based helper for CPU-side mesh extraction
  after denoising on Apple `mps`.
- The current local path uses CPU-side extraction specifically because the PyTorch accelerator path
  is not reliable enough.
- No MLX implementation exists today for:
  - the visual conditioning path
  - the label-conditioning path
  - the shape autoencoder decode
  - the field-to-mesh handoff

## Problem

Even with a working MLX denoiser, the Step1X lane will not be trustworthy until conditioning and
decode are ported or cleanly bridged with explicit validation.

## What we want to do

Port or explicitly bridge the conditioning and decode stages needed for a production-worthy MLX
geometry path.

## Why

The user-visible output is the mesh. If the denoiser is correct but the conditioning embeddings or
decoded field are wrong, the port will still look like a bad model.

## Requirements

- preserve preprocessing semantics used by the official geometry pipeline
- support both the base checkpoint and the label checkpoint
- preserve local label-policy overrides where they are still justified
- keep mesh extraction safe on Apple Silicon:
  - CPU extraction is acceptable initially
  - a later accelerator extraction path must be proven, not assumed
- expose decoded-field statistics for debugging

## Suggested implementation

- port the conditioning path first, including image resize / normalize policy and label encoding
- port the shape autoencoder decode path next
- keep marching cubes / trimesh export on CPU for the first usable MLX milestone
- only revisit accelerator-side extraction after the geometry path itself is sound

## Scope

- conditioning encoders
- label-conditioning bridge
- shape autoencoder decode
- field-to-mesh handoff
- mesh-decode diagnostics

## Non-goals

- texturing
- experimental unofficial conditioners
- export-format redesign

## Dependencies and related tasks

- [0004_step1x_mlx_reference_harness_and_failure_baseline.md](0004_step1x_mlx_reference_harness_and_failure_baseline.md)
- [0005_step1x_mlx_geometry_component_loader_and_weight_mapping.md](0005_step1x_mlx_geometry_component_loader_and_weight_mapping.md)
- [0006_step1x_mlx_denoiser_attention_port.md](0006_step1x_mlx_denoiser_attention_port.md)
- `scripts/step1x_extract_vae.py`
- `src/abstract3d/backends/step1x_runtime.py`

## Expected outcomes

- the full geometry path can execute with MLX-owned inference for the critical model stages
- final meshes can be traced back through conditioning and decoded-field diagnostics
- future geometry failures can be localized to conditioning, denoising, or decode instead of being
  treated as one opaque "bad output"

## Validation

- compare conditioning outputs and decoded-field statistics against the reference harness
- run at least one successful end-to-end `i23d` case through the MLX lane
- preserve mesh artifacts and high-resolution contact sheets for the same benchmark set
- run `pytest -q tests`

## Progress checklist

- [ ] port or bridge visual conditioning
- [ ] port or bridge label conditioning
- [ ] port the shape autoencoder decode path
- [ ] keep a safe CPU mesh-extraction fallback for the first usable milestone
- [ ] validate at least one end-to-end MLX geometry case

## Guidance for the implementing agent

Do not insist on accelerator-side mesh extraction for the first success milestone. A correct MLX
denoiser plus safe CPU extraction is more valuable than a fully accelerated but numerically suspect
pipeline.
