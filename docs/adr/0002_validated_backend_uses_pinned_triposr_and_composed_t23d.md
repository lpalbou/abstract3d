# ADR 0002: The Validated Backend Uses Pinned TripoSR And Composed `t23d`

Status: Accepted.

## Context

The permissive open-source 3D model landscape is strong but uneven. The most capable candidates are often CUDA-centric, large, or not yet validated on the Apple-local path used for this repository. The package still needs one backend that can be installed, tested, benchmarked, and documented end to end now.

## Decision

Abstract3D validates and ships:

- `stabilityai/TripoSR` as the default reconstruction backend
- a pinned upstream TripoSR source snapshot
- an internal checkpoint-key compatibility remap for `transformers` v5
- an MPS-friendly marching-cubes fallback path
- composed `text_to_scene3d` through `abstractvision` image generation followed by TripoSR reconstruction

Heavier permissive models such as TRELLIS, TRELLIS.2, InstantMesh, and LGM remain cataloged as research candidates until they clear the same reproducible validation bar.

Step1X-3D is governed separately by [ADR 0004](0004_step1x_geometry_only_official_backend_with_local_compatibility_patches.md) as an experimental geometry-only backend. That work does not change this ADR's core rule: TripoSR remains the validated default path in this repository.

## Consequences

### Positive

- The validated path fits the current storage and local-runtime constraints.
- `i23d` and composed `t23d` both work through one backend contract.
- The repo can ship real benchmark artifacts instead of only speculative model recommendations.

### Negative

- `t23d` is not yet a native text-only 3D generator.
- The default quality ceiling is bounded by single-image reconstruction and the quality of the composed source image.

### Neutral

- Research candidates remain important and should keep being tracked in the public model catalog.

## Enforcement

- Keep the pinned TripoSR source snapshot explicit.
- Do not require `trust_remote_code` at runtime for the validated backend path.
- Keep benchmark docs honest about `t23d` being a composed path.
- Do not promote a research candidate to “validated” without updating docs, benchmarks, and tests together.

## Validation

- unit coverage for checkpoint-key remapping, bundle output, and validation-suite generation
- AbstractCore host tests for direct capability calls, multimodal output routing, and residency control-plane routing
- checked-in benchmark assets under `docs/assets/validation/local-triposr/`

## Backlog Links

- No dedicated backlog item is created in this repo yet; this ADR governs the validated backend baseline.

## Related

- [ADR 0001](0001_scene3d_local_first_glb_contract.md)
- [ADR 0004](0004_step1x_geometry_only_official_backend_with_local_compatibility_patches.md)
- [Model strategy](../models.md)
- [Benchmarks](../benchmarks.md)
