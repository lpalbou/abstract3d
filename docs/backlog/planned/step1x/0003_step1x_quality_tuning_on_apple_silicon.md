# Planned: Step1X quality tuning on Apple Silicon

## Metadata

- Created: 2026-06-22
- Status: Planned
- Completed: N/A

## ADR status

- Governing ADRs: [ADR 0002](../../../adr/0002_validated_backend_uses_pinned_triposr_and_composed_t23d.md), [ADR 0004](../../../adr/0004_step1x_geometry_only_official_backend_with_local_compatibility_patches.md)
- ADR impact: None

## Context

The repository now ships an experimental local Step1X geometry backend and has checked Apple-local proof assets for both TripoSR and Step1X.

## Current code reality

- `docs/assets/validation/local-step1x/` contains the historical rejected Step1X baseline.
- `docs/assets/validation/local-step1x-label/` contains the older pre-refresh Apple-local Step1X reference lane.
- `docs/assets/validation/local-step1x-label-refresh/` contains the current refreshed Apple-local Step1X reference lane.
- `docs/assets/validation/comparison-step1x-vs-triposr/` contains the checked comparison package.
- `artifacts/validation/step1x-diagnostics/8step-teapot/` preserves a higher-step teapot diagnostic run for tuning follow-up.
- On the refreshed checked Apple-local profile, Step1X is still much slower than TripoSR and still weak on chair and owl. The teapot is materially better than the original baseline, but the espresso `i23d` case still exceeded safe Apple-local memory limits.
- The supported Step1X runtime already uses the official geometry checkpoint family, official source snapshot, a local compatibility patch layer, `float32` on `mps`, deterministic CPU post-extraction cleanup on the label-geometry path, vertex welding, export-axis canonicalization, and a `200000` face budget on Apple `mps`.
- The Step1X runtime now feeds raw source images plus preprocessing policy into the official pipeline so the upstream image-preprocess path runs exactly once.
- Thin-structure prompts such as chairs now auto-raise the Apple-local Step1X octree resolution from `128` to `192`.
- The Apple-local Step1X runtime now caps per-process MPS memory conservatively and releases the model runtime before CPU-side mesh export/render by default.
- The Apple-local Step1X validator now isolates each case in its own subprocess and defaults to a `64 GiB` RSS guard.
- Focused chair diagnostics on the older exported artifact showed that vertex welding collapsed the chair mesh topology from `7604` bodies to `1` watertight body, but the recovered geometry silhouette still remained below a production bar.
- An official-like higher-quality chair stress profile (`50` steps, guidance `7.5`, octree `384`) did not complete on this Apple `mps` lane.
- A stronger local chair profile (`12` steps, guidance `4.5`, octree `256`) also proved too slow to use as a safe default on the checked Apple-local machine.

## Problem

The local Step1X backend now has better Apple-local defaults and recovery diagnostics, but it still does not clear the promotion bar for production-quality Step1X geometry on the checked benchmark set.

## What we want to do

Finish the Apple-local Step1X quality work as an explicit planned track instead of leaving it as an optional proposal.

## Why

The user explicitly rejected the original Step1X proof as non-working, and the repository now contains both recovery fixes and new evidence that the promotion gate is still unresolved.

## Requirements

- keep the runtime local-only and official-only
- preserve TripoSR as the validated default unless new proof justifies a broader change
- improve Step1X `i23d` and `t23d` quality on Apple `mps` without weakening provenance or proof discipline
- do not auto-publish failed or unreviewed validation runs into the docs proof surface

## Suggested implementation

- keep the current MPS-tuned Step1X defaults:
  - automatic background removal for opaque inputs
  - lower default guidance on Apple `mps`
  - deterministic CPU post-extraction mesh cleanup on the Apple `mps` label-geometry path
  - vertex welding and canonical export-axis recovery before bundle export
  - lower default face budget on Apple `mps`
  - a higher default octree for thin-structure prompts
  - MLX cache release before Step1X geometry after composed `t23d` image generation
- prefer focused case-isolated recovery diagnostics before republishing the whole Step1X proof lane
- continue using targeted recovery diagnostics under `artifacts/validation/step1x-recovery/`
- only promote new Step1X proof assets after visual review of the same benchmark class

## Scope

- Apple-local Step1X quality work
- proof methodology and publish gating for Step1X validation
- recovery diagnostics and benchmark narration

## Non-goals

- widening Step1X into the texture stack
- weakening the TripoSR validated-default rule without new proof
- silently routing failed Step1X requests through another backend

## Dependencies and related tasks

- `docs/assets/validation/local-step1x/`
- `docs/assets/validation/local-step1x-label/`
- `docs/assets/validation/local-step1x-label-refresh/`
- `docs/assets/validation/step1x-recovery/`
- `artifacts/validation/step1x-recovery/`
- `src/abstract3d/backends/step1x_runtime.py`
- `scripts/validate_local.py`
- [ADR 0004](../../../adr/0004_step1x_geometry_only_official_backend_with_local_compatibility_patches.md)

## Expected outcomes

- Step1X Apple-local behavior is better than the original blob-like baseline on repeatable recovery cases
- the repository clearly separates historical bad proof, recovery diagnostics, and any future publishable proof
- future agents can see exactly why Step1X is still experimental and what evidence is missing

## Validation

- rerun focused Step1X recovery cases and preserve bundles under `artifacts/validation/step1x-recovery/`
- compare against the original bad baseline using docs-ready comparison sheets
- run `pytest -q tests`
- run targeted `abstractcore` scene3d routing/control-plane tests

## Progress checklist

- [x] fix Step1X Apple-local runtime defaults and cache handoff
- [x] preserve recovery artifacts and before/after comparison sheets
- [x] stop `scripts/validate_local.py` from auto-publishing proof assets by default
- [x] publish the current `local-step1x-label` Apple-local reference lane separately from the historical baseline
- [x] republish a refreshed Apple-local Step1X reference lane with case-isolated validation and failure preservation
- [x] recover Step1X export topology with vertex welding and canonical export axes on Apple `mps`
- [x] add MLX cache release after generated validation inputs in `scripts/validate_local.py`
- [ ] recover a Step1X `t23d` proof case that is visually good enough to publish
- [ ] decide whether Step1X Apple-local stays experimental, becomes blocked for some modes, or earns a stronger proof status

## Guidance for the implementing agent

Use the original bad Step1X proof as the historical baseline, not as a publishable success case. Do not claim Step1X promotion until the new proof is visually reviewed and materially better on comparable inputs.
