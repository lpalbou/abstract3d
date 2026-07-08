# ADR 0005: Hunyuan3D-2.1 Is A License-Gated Official Shape Backend

Status: Accepted.

## Context

Hunyuan3D-2.1 delivers the strongest single-image geometry among the locally runnable
open-weight models in this repository's catalog, and its shape stage is fully self-contained
(the DINOv2-L conditioner ships inside the official checkpoint, surface extraction is plain
marching cubes). But its weights are distributed under the Tencent Hunyuan 3D 2.1 Community
License, which is territory-restricted: it excludes the European Union, the United Kingdom,
and South Korea outright, and adds separate terms for large-scale commercial use. This
repository's validated path is intentionally permissive-only (ADR 0002), so the model cannot
join the validated tier, yet excluding it entirely would hide the strongest available local
geometry from operators who can lawfully use it.

## Decision

Abstract3D ships Hunyuan3D-2.1 as an experimental, explicitly license-gated backend:

- backend id `abstract3d:hunyuan3d21-local` with provider aliases `hunyuan3d21`, `hunyuan3d`, `hunyuan`
- official assets only: pinned `Tencent-Hunyuan/Hunyuan3D-2.1` source snapshot and the official
  `tencent/Hunyuan3D-2.1` checkpoint repository; contributor quantizations and mirrors are rejected
- the backend refuses to download or run official weights until the operator opts in through
  `scene3d_hunyuan_license_accepted=true` or `ABSTRACT3D_HUNYUAN_ACCEPT_LICENSE=1`, and the
  refusal message names the territorial restriction
- geometry stage only: the official PaintPBR texture stack (CUDA-only rasterization kernels,
  additional model weights) stays out of scope; textured output goes through the shared
  projection bake in `abstract3d.texturing`
- local compatibility patches are allowed at the runtime-wrapper level (adaptive volume
  decoder, device fallbacks) but never modify the pinned upstream snapshot silently

## Consequences

### Positive

- Operators outside the excluded territories get the strongest local geometry with a
  reproducible official-assets-only contract.
- The license boundary is enforced in code, recorded in metadata, and stated in docs, so
  bundles are attributable and the restriction cannot be missed.
- The permissive validated tier (TripoSR) remains uncontaminated.

### Negative

- Operators in the EU, UK, and South Korea cannot lawfully use the backend, and the gate makes
  that friction explicit rather than hiding it.
- The official texture stage is not available, so PBR parity with upstream marketing material
  is out of scope.

### Neutral

- The backend stays experimental regardless of quality until the repository's proof gates and
  license policy say otherwise.

## Enforcement

- Reject non-official model ids in the backend selector.
- Keep the license gate on every path that downloads or runs weights.
- Record `license` and `license_note` in provider listings and bundle metadata.
- Do not publish Hunyuan3D benchmark claims without naming the official asset set used.

## Validation

- unit tests for the license gate, official-model enforcement, dtype policy, postprocess
  behavior, axis canonicalization, and the adaptive volume decoder level schedule
- local proof bundles for the four checked objects under `artifacts/validation/`

## Related

- [ADR 0002](0002_validated_backend_uses_pinned_triposr_and_composed_t23d.md)
- [ADR 0003](0003_trellis2_uses_official_upstream_assets_only.md)
- [ADR 0004](0004_step1x_geometry_only_official_backend_with_local_compatibility_patches.md)
- [Model strategy](../models.md)
