# ADR 0001: Scene3D Is Local-First And GLB-Centric

Status: Accepted.

## Context

Abstract3D exists to extend AbstractCore with 3D generation. Without a clear capability contract, the package would drift into a mix of experimental wrappers, remote-only services, and incompatible artifact shapes that would be hard to route, store, and validate across the AbstractFramework stack.

## Decision

Abstract3D defines a first-class `scene3d` capability with:

- canonical tasks `text_to_scene3d` and `image_to_scene3d`
- `glb` as the primary artifact contract
- local-first execution as the default operating model
- explicit plugin discovery through `abstractcore.capabilities_plugins`
- no silent remote fallback in the validated path

`obj` and zipped bundles are supported as secondary output forms, but the design center is still a `glb`-first contract.

## Consequences

### Positive

- AbstractCore can route 3D outputs the same way it routes image, video, voice, and music outputs.
- Artifact stores and host runtimes get one stable primary format.
- Validation and benchmarking are simpler because the default path is local and explicit.

### Negative

- Some high-quality remote or CUDA-heavy research stacks are intentionally not first-class defaults.
- The validated path is narrower than the total open-source 3D ecosystem.

### Neutral

- Additional backends can be added later if they honor the same `scene3d` contract.

## Enforcement

- Keep `scene3d` as a first-class capability in AbstractCore integration surfaces.
- Treat silent remote fallback as a contract violation.
- Keep `glb` as the default output in public examples and validation scripts.
- Require docs and benchmarks to name the validated backend path explicitly.

## Validation

- package-local tests for `abstract3d`
- AbstractCore capability, multimodal-output, and residency tests
- reproducible local validation run with contact sheet and per-case stats

## Backlog Links

- No dedicated backlog item is created in this repo yet; this ADR governs the initial implementation baseline.

## Related

- [ADR 0002](0002_validated_backend_uses_pinned_triposr_and_composed_t23d.md)
- [ADR 0004](0004_step1x_geometry_only_official_backend_with_local_compatibility_patches.md)
- [Architecture](../architecture.md)
- [AbstractCore integration](../integration-abstractcore.md)
