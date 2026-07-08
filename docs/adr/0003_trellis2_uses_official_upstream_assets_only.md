# ADR 0003: TRELLIS.2 Uses Official Upstream Assets Only

Status: Accepted.

## Context

TRELLIS.2 is strong enough to merit direct integration work, but the public ecosystem around it already includes contributor quantizations, repacks, and alternate distribution shapes. Abstract3D needs one durable rule for what counts as a supported TRELLIS.2 input to the runtime, tests, and docs.

## Decision

The TRELLIS.2 backend in Abstract3D accepts only official upstream assets:

- source code from the pinned `microsoft/TRELLIS.2` repository snapshot
- model checkpoints from the official `microsoft/TRELLIS.2-4B` repository
- officially required companion checkpoints from their official repositories, including the sparse-structure decoder from `microsoft/TRELLIS-image-large` and the DINOv3 companion model `facebook/dinov3-vitl16-pretrain-lvd1689m`

Contributor quantizations, repacks, mirrors, and alternate encoder repositories are not part of the supported runtime contract.

If an official reduced-precision TRELLIS.2 release appears later, it can be added as a new supported official variant with code, tests, and docs updated together.

## Consequences

### Positive

- The runtime, tests, and docs all point at the same source of truth.
- Benchmark and validation results remain attributable to the official upstream release.
- Operators get clearer failure behavior when a gated companion model is missing.

### Negative

- Some lower-footprint community variants are intentionally unavailable through the supported backend path.
- The official DINOv3 companion model can block local validation until access is granted.

### Neutral

- Local authorized snapshot paths are still supported when they contain the official companion files.

## Enforcement

- Reject non-official TRELLIS.2 model ids in the backend selector.
- Reject non-official DINOv3 repository ids in the companion-model selector.
- Keep install docs, model docs, and error messages explicit about the official-only rule.
- Do not publish TRELLIS.2 benchmark claims without naming the official asset set used.

## Validation

- unit tests for official-model selection and non-official rejection
- unit tests for actionable DINOv3 gating errors and local snapshot overrides
- AbstractCore routing tests that prove request-scoped `scene3d` backend selection is real

## Related

- [ADR 0001](0001_scene3d_local_first_glb_contract.md)
- [ADR 0002](0002_validated_backend_uses_pinned_triposr_and_composed_t23d.md)
- [Model strategy](../models.md)
