# Proposed: Step1X texture stage research

## Metadata

- Created: 2026-06-22
- Status: Proposed
- Completed: N/A

## ADR status

- Governing ADRs: [ADR 0001](../../../adr/0001_scene3d_local_first_glb_contract.md)
- ADR impact: May revise existing ADR

## Context

Official upstream Step1X-3D publishes a texture adapter checkpoint and texture code, but the texture pipeline currently loads `stabilityai/stable-diffusion-xl-base-1.0`, `madebyollin/sdxl-vae-fp16-fix`, CUDA-only rasterization tooling, and other NVIDIA-centric dependencies.

## Current code reality

- `abstract3d` now ships an experimental local Step1X geometry backend.
- The repository currently treats permissive licensing and local-first runtime as meaningful promotion criteria for validated backends.
- The upstream texture path is not a standalone fully permissive official stack today because it depends on SDXL base weights under CreativeML Open RAIL++-M and CUDA-oriented rendering components.

## Problem or opportunity

The texture stage is strategically interesting because Step1X quality claims include textured outputs, but shipping it now would blur both the license boundary and the runtime/platform boundary.

## Proposed direction

Keep the texture path in the backlog as a research follow-up instead of silently excluding it from memory or shipping it prematurely.

## Why it might matter

If a future official texture stack becomes policy-compatible, or if the project explicitly accepts the external license/runtime tradeoff, a textured Step1X path could meaningfully improve output quality.

## Promotion criteria

- Explicit policy approval for the non-permissive texture dependency stack, or replacement with a permissive official alternative
- Clear backend boundary between geometry-only and textured variants
- Real local validation on the target runtime class the repo chooses to support

## Validation ideas

- Audit all official texture-time dependency licenses
- Validate whether a local texture run can complete reproducibly on supported hardware
- Measure storage, memory, and latency of the full textured path

## Non-goals

- This proposal does not authorize shipping the full texture path now.

## Guidance for future agents

Re-check official upstream dependencies before promotion. Do not assume the Apache-2.0 label on the main Step1X repo makes the entire runtime stack permissive.
