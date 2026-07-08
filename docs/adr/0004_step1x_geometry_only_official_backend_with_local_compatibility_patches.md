# ADR 0004: Step1X Uses An Official Geometry-Only Backend With Local Compatibility Patches

Status: Accepted.

## Context

Abstract3D needs a local Step1X integration that stays compatible with the repository's local-first, explicit-provenance, and reproducible-proof rules.

Official upstream Step1X publishes multiple subpaths. The geometry checkpoint is the only path that fits the current repository boundary:

- it supports local `image_to_scene3d`
- it does not provide native text-only 3D in the supported checkpoint
- the texture stage widens the dependency and runtime surface beyond the scope accepted here

The pinned official Step1X source snapshot also requires compatibility work on the checked Apple-local Python and `transformers` stack. On Apple `mps`, the repository now also relies on a tuned local runtime profile:

- `float32` on `mps`
- automatic background removal for opaque inputs
- deterministic CPU post-extraction cleanup on the label-geometry path, including floater removal, component pruning, vertex welding, and face reduction
- a lower default guidance scale than upstream
- a lower default face budget than upstream on `mps`
- a higher default octree resolution for thin-structure prompts on Apple `mps`
- stable-pose export-axis canonicalization for generated Step1X meshes
- a conservative per-process MPS memory cap on Apple Silicon
- non-resident Step1X execution by default on Apple `mps`, with runtime release before mesh export/render
- MLX cache release between composed `t23d` image generation and Step1X geometry inference
- isolated per-case Step1X validation on Apple `mps`, with a default `64 GiB` RSS guard in the local proof harness

## Decision

Abstract3D supports Step1X only as an experimental local geometry backend:

- backend id: `abstract3d:step1x-local`
- provider alias: `step1x`
- model family: the official `stepfun-ai/Step1X-3D` repository
- supported checkpoint family: `Step1X-3D-Geometry-1300m` plus `Step1X-3D-Geometry-Label-1300m`
- current Apple-local reference lane: `Step1X-3D-Geometry-Label-1300m`
- `t23d`: composed `text -> image -> geometry`

Abstract3D also owns a small local compatibility patch layer on top of the pinned official Step1X source snapshot. That patch layer is part of the supported runtime contract for this repo.

Apple-local Step1X proof publication is opt-in. Recovery or validation runs may generate local artifacts by default, but docs-ready proof assets should only be published after explicit review.

Texture output is not part of the supported Step1X backend in this repository.

## Consequences

### Positive

- The repository gets a real local Step1X backend without weakening the validated TripoSR default path.
- Provenance stays explicit because the supported model repo, source snapshot, and checkpoint family are pinned.
- The Step1X runtime can be tested, benchmarked, and documented through the same bundle and validation surfaces as the rest of `abstract3d`.

### Negative

- The supported Step1X path is narrower than the full upstream project.
- The runtime depends on a maintained compatibility patch layer, so upgrades require deliberate revalidation.
- The Step1X backend remains experimental until its quality and performance justify promotion beyond that status.
- The Apple-local Step1X profile is still not equivalent to the upstream CUDA-centric defaults, so proof and benchmark narration must stay explicit about the local tuning layer.

### Neutral

- Future Step1X variants can still be added later, but they should use explicit new backend ids rather than silently changing the meaning of the geometry-only path.

## Enforcement

- Reject non-official Step1X model ids in the backend selector.
- Reject non-geometry Step1X checkpoint subfolders in the supported backend.
- Keep docs, benchmarks, and error messages explicit that the supported Step1X backend is geometry-only and experimental.
- Treat the local compatibility patch layer as part of the supported runtime surface, not as an incidental hidden fix.
- Keep the Apple-local tuned-default behavior explicit in runtime metadata and operator-facing docs.
- Keep the default Step1X export frame explicit: canonical upright exports are the supported local contract unless the operator disables that behavior deliberately.
- Keep `local-step1x/` documented as the historical rejected baseline, `local-step1x-label/` as the older pre-refresh reference lane, and `local-step1x-label-refresh/` as the current refreshed Apple-local reference lane.
- Do not auto-publish Step1X proof assets from the validation harness without an explicit publish destination.
- Do not present Step1X as the validated default backend without updating docs, tests, and proof assets together.

## Validation

- unit tests for official-model selection, checkpoint selection, and source patching
- unit tests for Apple-local Step1X default profile behavior
- `abstractcore` routing tests for request-scoped `provider="step1x"` selection
- checked refreshed Apple-local reference assets under `docs/assets/validation/local-step1x-label-refresh/`
- preserved historical baseline assets under `docs/assets/validation/local-step1x/`
- checked recovery diagnostics under `docs/assets/validation/step1x-recovery/`
- checked comparison assets under `docs/assets/validation/comparison-step1x-vs-triposr/`

## Backlog Links

- [0001_step1x_geometry_backend.md](../backlog/completed/step1x/0001_step1x_geometry_backend.md)
- [0003_step1x_quality_tuning_on_apple_silicon.md](../backlog/planned/step1x/0003_step1x_quality_tuning_on_apple_silicon.md)
- [0002_step1x_texture_stage_research.md](../backlog/proposed/step1x/0002_step1x_texture_stage_research.md)

## Related

- [ADR 0001](0001_scene3d_local_first_glb_contract.md)
- [ADR 0002](0002_validated_backend_uses_pinned_triposr_and_composed_t23d.md)
- [Model strategy](../models.md)
- [Benchmarks](../benchmarks.md)
