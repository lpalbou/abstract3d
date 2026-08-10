# Proposed: MV-Adapter ig2mv as clay-conditioned view generator (Apache-2.0 lane)

## Metadata

- Created: 2026-07-21
- Status: Proposed
- Completed: N/A

## ADR status

- Governing ADRs: [ADR 0001](../../adr/0001_scene3d_local_first_glb_contract.md) (local-first)
- ADR impact: none expected (new optional view source; existing gates keep authority)

## Context

Our synthesized side/back views are the measured weak input of the bust pipeline
(docs/research/viewgen_audit.md): freestanding i2i draws carry 40-degree pose errors and
identity flips; the identity-preserving route currently in the recipe is the REMOTE
default editor (money/privacy tension the repo's gates exist to prevent). MV-Adapter's
ig2mv mode generates 6 views of an object CONDITIONED on its geometry (position/normal
maps) plus a reference image — the exact contract of our "clay-guided view" role, as a
trained joint-attention model instead of per-view i2i.

## Verified facts (2026-07-21, docs/research/generative_process_2026.md)

- License: Apache-2.0 for BOTH the repo and the HF weights (`huanngzh/mv-adapter`,
  card metadata `license: apache-2.0`) — closes the "verify weights terms" caveat in
  multiview_consistency_2026.md §3e. The permissive option in this space.
- Variants: `mvadapter_ig2mv_sdxl.safetensors` (3.6 GB adapter; SDXL base; ~16 GB
  CUDA-reported) and SD2.1 variants (<10 GB). 768 px output on SDXL.
- The adapter rides stock diffusers attention (no custom kernels advertised in the
  UNet path) — MPS-plausible, UNVERIFIED. Their TEXTURE bake path requires CV-CUDA —
  CUDA-only, unusable for us; we keep OUR projection bake as the aggregator (which is
  the pipeline's strength: registration machinery, authority classes, gates).
- Geometry conditioning consumes position/normal renders we can produce ourselves with
  the existing moderngl/trimesh stack (same maps our clay renders already produce).

## Proposed direction

1. MPS smoke spike (pattern: `scripts/experimental/hy3dpaint_mps_spike.py`): load
   SDXL + ig2mv adapter fp16 on MPS, generate 6 views for one non-person fixture mesh
   at 768; measure wall/RSS. If SDXL is impractical on MPS, fall back to the SD2.1
   variant (512 px) and judge quality honestly.
2. Feed our own clay position/normal renders (elevation 0, azimuths matching the 2mv
   rig) instead of their render stack; photo enters as the reference image.
3. Wire as an optional view source for (a) texture references and (b) 2mv geometry
   conditioning — both behind the existing acceptance gates (row law, head-band pose
   ruler, <2% feature offsets, protect-observed-texels). The gates stay the authority;
   this only changes the generator.

## Why it might matter

- Replaces the remote-editor dependency for identity-bearing view synthesis with a
  permissive local model — restores the "no remote APIs for the operator's face" line
  without giving up the registered-view recipe.
- Joint multi-view attention removes the independent-draw variance class our audits
  keep measuring (double-mouth mechanism).

## Risks (stated plainly)

- Face identity through reference-attention is unproven for this model class
  (Objaverse-trained); the viewgen audit showed identity is the hardest axis locally.
- SDXL on MPS is slow; 6-view joint generation may be minutes-per-set. Measure first.
- MPS correctness of the custom attention processors is unverified (torch 2.10 SDPA
  caveats in backlog 0020 apply here too).

## Promotion criteria

- MPS run within the 20 GB profile producing 6 views that pass the existing gates on
  a fixture subject, then on the bust subject with operator review of the face crop.
- Measured better-or-equal identity + pose vs the current best local recipe (L9-class)
  on the same clay, same acceptance instruments.

## Non-goals

- Not adopting their CV-CUDA texture bake; not replacing the projection bake.
