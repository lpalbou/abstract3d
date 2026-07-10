# Model Strategy

## Validated Default Backend

The validated default backend is:

| Model | Tasks | License | Local status | Notes |
|---|---|---|---|---|
| `stabilityai/TripoSR` | `image_to_scene3d`, composed `text_to_scene3d` | MIT | Validated on Apple Silicon | Default backend in this repo. Uses `abstractvision` only for the image-composition step of `t23d`, and now ships a baked base-color texture path by default. |

Measured local footprint for the validated path used in this repo:

- TripoSR cache: `1.6G`
- pinned TripoSR source snapshot: `63M`

Validated TripoSR output profile:

- default `mc_resolution=256`
- deterministic CPU cleanup before export
- default `texture_mode=baked_basecolor`
- default `texture_resolution=2048`
- explicit `texture_resolution=4096` hero mode when operators accept higher time and larger bundle size

## Experimental Step1X Geometry Backend

The repository also ships an experimental local Step1X backend:

| Model path | Tasks | License | Local status | Notes |
|---|---|---|---|---|
| `stepfun-ai/Step1X-3D/{Step1X-3D-Geometry-1300m,Step1X-3D-Geometry-Label-1300m}` | `image_to_scene3d`, composed `text_to_scene3d` | Apache-2.0 | Implemented and benchmarked on Apple Silicon | Geometry-only backend using official Step1X geometry weights with composed `t23d` via `abstractvision`. The current Apple-local reference lane uses label geometry by default with automatic base-checkpoint fallback for sharp asymmetric `i23d` cases on Apple `mps`. |

Official local assets for the supported path:

- Step1X geometry subset: `6.8G`
- pinned Step1X source snapshot: `147M`

Current boundary:

- the supported backend id is `abstract3d:step1x-local`
- the supported provider alias is `step1x`
- the backend accepts only the official Step1X model repo and the official geometry checkpoint family
- the texture stage is out of scope here
- the Apple `mps` path uses `float32`
- the current Apple `mps` path uses isolated CPU-side mesh helpers after denoising, deterministic post-extraction cleanup, vertex welding, export-axis canonicalization, and automatic base-checkpoint fallback for sharp asymmetric `i23d` cases
- thin-structure prompts such as chairs auto-raise Step1X octree resolution to `192`

Current benchmark takeaway:

- the backend works locally and produces reproducible bundles
- the current Apple-local reference lane lives under `docs/assets/validation/local-step1x-dynamic/`
- that lane is materially better than the historical `local-step1x/` baseline and now completes all four checked proof cases safely under guarded execution
- on the checked Apple-local proof cases, it is still slower than TripoSR and still weaker on chair and owl, so it remains experimental

## Experimental License-Gated Hunyuan3D-2.1 Shape Backend

The repository ships an experimental Hunyuan3D-2.1 shape backend behind an explicit license gate:

| Model path | Tasks | License | Local status | Notes |
|---|---|---|---|---|
| `tencent/Hunyuan3D-2.1/hunyuan3d-dit-v2-1` | `image_to_scene3d`, composed `text_to_scene3d` | tencent-hunyuan-community | Implemented and benchmarked on Apple Silicon | Official 3.3B flow-matching DiT plus shape VAE. The strongest geometry in the local catalog on the checked proof objects. Textured output uses the shared projection bake; the official CUDA-only PaintPBR stage is out of scope. |
| `tencent/Hunyuan3D-2mv/hunyuan3d-dit-v2-mv` | `image_to_scene3d`, composed `text_to_scene3d` | tencent-hunyuan-community | Implemented and benchmarked on Apple Silicon | Official multi-view shape stage (2.0 family, 1.1B DiT). Conditions geometry on up to four views (front/left/back/right). Reference views whose angles snap to those slots (within 25°) drive shape reconstruction as well as texture. `-fast` and `-turbo` subfolders are accepted via `model_subfolder`. |

License boundary:

- The Tencent Hunyuan 3D 2.1 Community License is territory-restricted: it excludes the European Union, the United Kingdom, and South Korea, and adds terms for large-scale commercial use. The `Hunyuan3D-2mv` repository ships under the same community license family, so the same gate applies.
- Because `abstract3d` cannot know where it runs, the backend refuses to download or run official weights until the operator opts in explicitly with `scene3d_hunyuan_license_accepted=true` or `ABSTRACT3D_HUNYUAN_ACCEPT_LICENSE=1`.
- The backend id is `abstract3d:hunyuan3d21-local`; provider aliases are `hunyuan3d21`, `hunyuan3d`, and `hunyuan`.
- Only the official `tencent/Hunyuan3D-2.1` and `tencent/Hunyuan3D-2mv` repositories are accepted; quantized or mirrored repacks are rejected.

Official local assets for the supported path:

- Hunyuan3D-2.1 DiT subset (`hunyuan3d-dit-v2-1`, fp16): `7.4G`
- Hunyuan3D-2mv DiT subset (`hunyuan3d-dit-v2-mv`, fp16): `4.9G`
- pinned Hunyuan3D-2.1 source snapshot: `~330M` (also serves the 2mv checkpoint through a config namespace remap, verified key-exact at load)

Multi-view geometry conditioning (`Hunyuan3D-2mv`):

- Pass `model="tencent/Hunyuan3D-2mv"` together with the standard `texture_reference_images` / `texture_reference_angles` inputs.
- References whose angles snap to the trained view slots (`front`=0°, `left`=+90°, `back`=180°, `right`=-90°, tolerance 25°) are added to the geometry conditioning dictionary; off-slot references (a 45° three-quarter shot) still contribute to the texture bake.
- Generation metadata records `multiview_conditioning` and the `geometry_views` used, so runs are auditable.
- On the checked face proof, front + left/right profiles replaced the hallucinated back of the head with a plausible multi-view-constrained shape, and the ADR 0008 texture cycle (photometric source pose, overlap registration, witness gating) took the adversarial QA harness from 66 failed checks to 9-10 (proof bundle: `artifacts/validation/iter3-multiview-fixed/face-2mv/`). Observed coverage reads ~0.40 under the gated accounting; earlier 0.74-0.91 figures counted mirror-fill as observation and are retracted.
- The bake runs in the canonical orthographic frame for this backend (ADR 0007): photos are recentered exactly as the model's own preprocessor does, so source registration is deterministic. Reference photos may be cropped differently; they are aligned by crop-aware edge-chamfer registration.

Apple-local operating profile:

- `float16` on `mps` (the DiT and VAE are fp16-stable there); `float32` on CPU
- default `num_inference_steps=50` (the checked local proof lane used `30`, the community quality/speed sweet spot), `guidance_scale=5.0`, `octree_resolution=384`
- an adaptive coarse-to-fine volume decoder replaces the upstream hierarchical decoder, which starts too coarse for thin structures and misbehaves on `mps`
- the runtime is released before texture bake and preview rendering unless `scene3d_hunyuan_keep_resident` is set

Operator overrides (config key / environment variable):

- `scene3d_hunyuan_license_accepted` / `ABSTRACT3D_HUNYUAN_ACCEPT_LICENSE`: explicit license opt-in
- `scene3d_hunyuan_source_dir` / `ABSTRACT3D_HUNYUAN_SOURCE_DIR`: local pinned source snapshot override
- `scene3d_hunyuan_volume_decoder` / `ABSTRACT3D_HUNYUAN_VOLUME_DECODER`: `adaptive` (default), `vanilla`, or `hierarchical_upstream`
- `scene3d_hunyuan_texture_mode` / `ABSTRACT3D_HUNYUAN_TEXTURE_MODE`: `baked_basecolor` (default) or `none` for geometry-only exports
- `scene3d_hunyuan_num_inference_steps`, `scene3d_hunyuan_guidance_scale`, `scene3d_hunyuan_octree_resolution`, `scene3d_hunyuan_num_chunks`, `scene3d_hunyuan_max_facenum`: generation defaults

## Experimental Official TRELLIS.2 Path

The repository still includes an experimental local TRELLIS.2 backend, but it is not part of the permissive validated path:

| Model path | Tasks | License | Local status | Notes |
|---|---|---|---|---|
| `microsoft/TRELLIS.2-4B` | `image_to_scene3d`, composed `text_to_scene3d` | MIT plus gated companion dependency | Implemented, not validated end to end here | Official-only backend surface. End-to-end local proof remains blocked by the required gated DINOv3 companion model. |

The supported TRELLIS.2 runtime accepts only:

- the official `microsoft/TRELLIS.2-4B` repository
- the official sparse-structure decoder from `microsoft/TRELLIS-image-large`
- the official DINOv3 companion model from `facebook/dinov3-vitl16-pretrain-lvd1689m`

Because that official DINOv3 dependency is gated and carries its own license terms, TRELLIS.2 is not promoted as the validated local default. The backend itself is accepted and maintained; running it is an explicit operator decision.

DINOv3 license summary (the "DINOv3 License", Meta; review the full text before accepting):

- non-exclusive, worldwide, royalty-free license to use, modify, and create derivative works; commercial use is permitted
- redistribution of the model or derivatives must carry the same Agreement, and distributors must display "Built with DINOv3" prominently
- acceptable-use restrictions apply, including prohibitions on military/warfare, nuclear, espionage, and other trade-control-restricted end uses
- access is gated: Meta reviews each request (typically within a few days) and availability excludes comprehensively sanctioned jurisdictions
- full text: `https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m/blob/main/LICENSE.md` (also at `https://ai.meta.com/resources/models-and-libraries/dinov3-license/`)

Unblocking TRELLIS.2 locally requires operator acknowledgment and action:

1. Sign in to Hugging Face, open `facebook/dinov3-vitl16-pretrain-lvd1689m`, review the DINOv3 License, and submit the access request (Meta's DINOv3 download page is the alternative official channel and emails a direct link).
2. Once approved, authenticate the machine (`hf auth login`) and let Abstract3D download the companion model on the next run, or point `scene3d_trellis2_dino_model` / `ABSTRACT3D_TRELLIS2_DINO_MODEL` at the authorized local snapshot directory containing `config.json`, `model.safetensors`, and `preprocessor_config.json`.
3. If you distribute assets or tooling that embed DINOv3 or its derivatives, carry the license and the "Built with DINOv3" notice as the Agreement requires. Abstract3D surfaces this note in its own docs because the runtime cannot verify downstream distribution.

## Companion Image Models (reference generation and captioning)

Generated reference completion (`texture_reference_generation`) and composed
`t23d` route image synthesis through `abstractvision`; these are the models
validated with that path on Apple Silicon:

| Model | Role | Approx footprint | Validated status |
|---|---|---:|---|
| `AbstractFramework/flux.2-klein-4b-8bit` | i2i reference synthesis (default class) | `7G` | Validated on the four-subject zero-hint suite; ~45 s/view (M-series, 768 px). Sufficient for objects; floor-accepts wet-look hair on human subjects. |
| `AbstractFramework/flux.2-klein-9b-8bit` | i2i reference synthesis (recommended for people) | `17G` | Strict-passes the portrait family where 4B floor-accepts; ~110 s/view. HF-gated: accept the FLUX.2-klein license and authenticate. |
| `AbstractFramework/qwen-image-edit-2511-8bit` | i2i editor (CFG-capable, honors negative prompts) | `28G` | Downloads and registers; did not reach a first denoise step on the 48 GB validation host — parked until a larger-memory host validates it. |
| `Salesforce/blip-image-captioning-base` | automatic subject captioning (`abstract3d.captioning`) | `1G` | Validated: captions feed the material-free noun extraction; never contributes material words (structural stoplist). |

Notes:

- Negative prompts are inert on guidance-distilled FLUX.2-klein routes; the
  positive instruction and the acceptance gates carry material fidelity there.
- Set the provider/model with `scene3d_image_provider` / `scene3d_image_model`
  or `ABSTRACT3D_IMAGE_PROVIDER` / `ABSTRACT3D_IMAGE_MODEL`.

## Research Catalog

These candidates matter strategically, but they are not promoted to validated backends here yet.

| Model | Tasks | License | Approx footprint | Current status |
|---|---|---|---:|---|
| `microsoft/TRELLIS-image-large` | `image_to_scene3d` | MIT | `3.3G` | Research only |
| `microsoft/TRELLIS-text-large` | `text_to_scene3d` | MIT | `2.3G` | Research only |
| `TencentARC/InstantMesh` | `image_to_scene3d` | Apache-2.0 | `7.3G` | Research only |
| `ashawkey/LGM` | `image_to_scene3d`, `text_to_scene3d` | MIT | `5.0G` | Research only |
| `openai/shap-e` | `image_to_scene3d`, `text_to_scene3d` | MIT | n/a | Planned investigation |

## Selection Policy

To replace or augment the validated default backend, a candidate should satisfy all of the following:

- permissive license for the actually required local stack
- reproducible local setup
- explicit artifact contract for `glb` or equivalent scene mesh outputs
- no silent remote fallback
- benchmark proof committed through the same documentation surface used here

See:

- [ADR 0002](adr/0002_validated_backend_uses_pinned_triposr_and_composed_t23d.md)
- [ADR 0004](adr/0004_step1x_geometry_only_official_backend_with_local_compatibility_patches.md)
- [Benchmarks](benchmarks.md)
