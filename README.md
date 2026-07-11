# Abstract3D

Local-first 3D generation for the AbstractFramework ecosystem.

`abstract3d` extends `abstractcore` with a first-class `scene3d` capability for:

- `image_to_scene3d` (`i23d`)
- `text_to_scene3d` (`t23d`)

The validated default path in this repository remains:

- `stabilityai/TripoSR` for single-image 3D reconstruction
- `abstractvision` plus a provider-neutral composed image stage for `t23d`
- `glb` as the primary output contract, with optional `obj` and zipped bundles
- backend-owned TripoSR defaults of `mc_resolution=256`, `cleanup=presentation`, and `texture_mode=baked_basecolor` at `texture_resolution=2048`

The repository also ships an experimental local Step1X backend:

- backend id: `abstract3d:step1x-local`
- provider alias: `step1x`
- scope: official geometry checkpoint only
- `t23d`: composed `text -> image -> geometry`
- current status: the checked Apple-local reference lane now uses dynamic official-checkpoint routing on `mps` with label geometry by default and base-geometry fallback for sharp asymmetric `i23d` cases such as the espresso proof case; the shipped Apple-local runtime also uses isolated CPU mesh helpers, deterministic cleanup, vertex welding, and canonical export axes. Step1X still remains experimental because the current four-case proof leaves the chair below bar and the owl weak.

## What You Get

- A Python manager API: `Scene3DManager`
- A validated local TripoSR backend: `abstract3d:triposr`
- An experimental local Step1X geometry backend: `abstract3d:step1x-local`
- An experimental, license-gated Hunyuan3D-2.1 shape backend: `abstract3d:hunyuan3d21-local`
- An experimental local TRELLIS.2 backend: `abstract3d:trellis2-local`
- A shared projection texture bake (`abstract3d.texturing`) with canonical-frame orthographic or perspective projection, strict first-surface visibility, crop-aware photo registration, multi-view blending with per-texel conflict resolution, mirror completion, and crease-aware harmonic fill
- AbstractCore capability plugin registration through `abstractcore.capabilities_plugins`
- Bundle outputs with `scene.glb`, `scene.obj`, `input.png`, `preview.png`, `contact_sheet.png`, and `metadata.json`
- Textured TripoSR bundles that also include `texture.png`, `uv_preview.png`, and OBJ sidecars when the baked path is enabled
- A reproducible validation harness: [`scripts/validate_local.py`](scripts/validate_local.py)
- A focused texture-proof pack builder: [`scripts/triposr_texture_proof.py`](scripts/triposr_texture_proof.py)
- A public model catalog for validated, experimental, blocked, and research-stage model families

## Current State (v0.2.0)

Development status: **Alpha**. Object-centric generation only: one centered subject per
image; multi-object scenes, cluttered backgrounds, and strong occlusion are out of scope.

### Operating systems

| Platform | Status |
| --- | --- |
| macOS, Apple Silicon (`mps`) | **Validated.** The entire proof and certification record was produced on this profile (Apple M5 Max, 128 GB, Python 3.12). |
| Linux / Windows, NVIDIA or AMD | Implemented (`abstract3d[gpu]` extra, `--device cuda`), **not validated** — no checked proof run exists on these hosts. |
| CPU-only (any OS) | Works for the library surface and tests; full generation is impractically slow and unchecked. |

### Backends

| Backend | Generation | Status |
| --- | --- | --- |
| `abstract3d:triposr` (`stabilityai/TripoSR`) | `i23d`, composed `t23d` | **Validated default.** Fast (20-60 s/object on `mps`), permissive license. |
| `abstract3d:hunyuan3d21-local` (`tencent/Hunyuan3D-2.1`) | `i23d`, composed `t23d` | Experimental, license-gated. Strongest checked local geometry (7-13 min/object at 30 steps). License excludes EU, UK, South Korea; requires `ABSTRACT3D_HUNYUAN_ACCEPT_LICENSE=1`. |
| `tencent/Hunyuan3D-2mv` (same backend) | multi-view `i23d` | Experimental, same license gate. Front/left/back/right reference views condition the geometry itself. |
| `abstract3d:step1x-local` (`stepfun-ai/Step1X-3D`) | `i23d` geometry only | Experimental. All four checked cases complete under guarded Apple-local execution, but chair and owl remain below bar. |
| `abstract3d:trellis2-local` (Microsoft TRELLIS.2) | `i23d` | Accepted but **blocked at runtime** on gated DINOv3 credentials: operators must request `facebook/dinov3-vitl16-pretrain-lvd1689m` access themselves. |

### Texture pipeline

All textured output flows through the shared projection bake (`abstract3d.texturing`),
which closed a six-cycle adversarial zero-defect program on 2026-07-07
(23 defects fixed, 10 proven capture limits, 0 open — see
[`artifacts/validation/texture-cycle-proofs/CERTIFICATION.md`](artifacts/validation/texture-cycle-proofs/CERTIFICATION.md)):

- canonical-frame orthographic (or perspective) projection with strict first-surface visibility
- crop-aware photo registration and photometric pose estimation
- multi-view blending with per-texel conflict resolution and gradient-domain compositing
- confidence-gated mirror completion and crease-aware mesh-graph harmonic fill
- generated reference completion (`texture_reference_generation`, default `auto`):
  when only one photo is provided, render the mesh from the unseen angles, synthesize
  matching photos through the configured `abstractvision` i2i provider, and feed them
  into the bake as completion-only witnesses — they may only paint surface no photo
  observed; texels the photo covers credibly are inviolable (measured on the certified
  owl: observed coverage 0.30 -> 0.81). Fully autonomous: the source photo is captioned
  automatically (BLIP) and all text — captions and user prompts alike — is reduced to a
  material-free noun before prompting (the source photo is the only material authority).
  Every candidate is shape-locked (silhouette IoU vs the mesh's own clay render) and
  must strictly pass three calibrated material-fidelity oracles (band-pass relief,
  part-palette identity, baked speculars) with a prompt-escalating retry ladder;
  rejected angles fall back to witnessed-texture fill, which cannot flip materials.
  The finished bake must then pass a whole-bake A/B acceptance gate — bake with and
  without the generated views, ship the generated bake only if it does not regress
  photo fidelity, brightness, or seam metrics (upholstered near-planar subjects like
  the validation chair auto-reject here; the baseline ships with the verdict recorded).
  `auto` fires only with an explicitly configured image provider (FLUX.2-klein-4b for
  objects; 9b recommended where 4b floor-fails). Person subjects are refused in BOTH
  modes — no gate can defend facial identity, so synthesizing people requires the
  separate explicit acknowledgment (`--texture-reference-allow-person`), which puts a
  `person_warning` on the record. Generated views are plausible synthesis, not ground
  truth, and every bundle records full provenance (prompts, seeds, per-attempt gate
  metrics, image hashes, acceptance verdict)

Texture color is exact where a photo observed the surface and reconstructed elsewhere;
regions no photo can see remain approximations unless you add more views (real or generated).

Benchmark proof, contact sheets, and comparison assets: [`docs/benchmarks.md`](docs/benchmarks.md)

Generation and evaluation methodology: [`docs/methodology.md`](docs/methodology.md)

## Installation

Lightweight base package:

```bash
pip install abstract3d
```

That base install includes the lightweight `abstractvision` package contract, so composed `t23d`
can use remote OpenAI or OpenAI-compatible image generation without hardcoding any local image
runtime. Local 3D runtimes still stay behind explicit extras.

Validated TripoSR runtime:

```bash
pip install "abstract3d[triposr]"
```

Experimental Step1X geometry runtime:

```bash
pip install "abstract3d[step1x]"
```

Experimental license-gated Hunyuan3D-2.1 shape runtime:

```bash
pip install "abstract3d[hunyuan3d]"
```

Compatibility alias for callers that still request the historical composed `t23d` extra:

```bash
pip install "abstract3d[t23d]"
```

Apple-local profile for `abstract3d`, `abstractvision`, the validated TripoSR path, and the experimental Step1X path:

```bash
pip install "abstract3d[apple]"
```

GPU-local profile for Linux/Windows NVIDIA or AMD hosts:

```bash
pip install "abstract3d[gpu]"
```

If you want host-side plugin discovery from AbstractCore:

```bash
pip install "abstractcore[scene3d]"
```

Notes:

- `abstract3d[step1x]` installs the official Step1X geometry runtime surface only. The supported checkpoint family is `stepfun-ai/Step1X-3D/{Step1X-3D-Geometry-1300m,Step1X-3D-Geometry-Label-1300m}`.
- The Step1X texture stage is intentionally out of scope here.
- The Step1X Apple-local path uses `float32` on `mps` for stability.
- The current Apple-local Step1X proof lane uses `Step1X-3D-Geometry-Label-1300m` by default and automatically falls back to `Step1X-3D-Geometry-1300m` for sharp asymmetric `i23d` cases on Apple `mps`.
- The shipped Apple-local Step1X runtime also welds vertices after extraction, canonicalizes export axes, and raises the default octree to `192` for thin-structure prompts such as chairs.
- On Apple `mps`, the Step1X runtime now also caps per-process MPS memory to a conservative local default and releases the model runtime before mesh export/render unless you explicitly keep it resident.
- The Step1X validation harness now runs each Apple-local case in its own subprocess and defaults to a `64 GiB` RSS guard so one pathological case does not take down the whole suite.
- Backend-only extras such as `abstract3d[triposr]`, `abstract3d[step1x]`, and `abstract3d[trellis2]` are truthful `i23d` runtime installs. `pip install abstract3d` already includes the lightweight `abstractvision` composition contract; use `abstract3d[apple]` or `abstract3d[gpu]` when you want local composed `t23d`.
- `t23d` composes through `abstractvision`. By default, `abstract3d` now passes no hardcoded provider and uses `scene3d_image_provider` / `scene3d_image_model`, `ABSTRACT3D_IMAGE_PROVIDER` / `ABSTRACT3D_IMAGE_MODEL`, or the configured `abstractvision` default.
- The checked Apple-local proof suite still used an explicit `mlx-gen` image model (`AbstractFramework/flux.2-klein-4b-8bit`), but that is a validation-lane choice, not the runtime default.

## Quick Start

Inspect the validated catalog:

```bash
abstract3d catalog --validated-only --json
```

### Image → 3D model (recommended path)

One photo in, a fully textured GLB out. The strongest local pipeline is the
license-gated Hunyuan3D-2.1 backend with generated reference completion: the
pipeline reconstructs the mesh, synthesizes the unseen angles from the mesh's
own geometry guides through your local image model, and bakes everything into
one texture — protected by the whole-bake acceptance gate (if the generated
views would make the result worse than the photo-only bake, the photo-only
bake ships and the verdict is recorded in `metadata.json`):

```bash
export ABSTRACT3D_HUNYUAN_ACCEPT_LICENSE=1        # review the Tencent license first
export ABSTRACT3D_IMAGE_PROVIDER=mlx-gen          # local image model for the unseen angles
export ABSTRACT3D_IMAGE_MODEL=AbstractFramework/flux.2-klein-4b-8bit

abstract3d i23d ./owl-photo.png \
  --output-dir ./out/owl \
  --backend hunyuan3d21 \
  --device mps
```

`texture_reference_generation` defaults to `auto`: it fires only when a local
image provider is configured, and it refuses person subjects outright (no
gate can defend facial identity). To synthesize views of a person anyway,
add the explicit acknowledgment `--texture-reference-allow-person`.

### Text → 3D model

The same pipeline, with the source photo composed from your prompt first:

```bash
export ABSTRACT3D_HUNYUAN_ACCEPT_LICENSE=1
export ABSTRACT3D_IMAGE_PROVIDER=mlx-gen
export ABSTRACT3D_IMAGE_MODEL=AbstractFramework/flux.2-klein-4b-8bit

abstract3d t23d "a red sports car, studio photo" \
  --output-dir ./out/sports-car \
  --backend hunyuan3d21 \
  --device mps
```

Every bundle records full provenance: the composed/source image, generated
reference views (`generated_*.png` with their geometry guides), per-attempt
gate metrics, and the acceptance verdict.

### Lighter-weight variants

Generate a mesh from an image with the lightweight validated backend:

```bash
abstract3d i23d ./object.png --output-dir ./out/object --device mps --format glb
```

For TripoSR, that command now defaults to `mc_resolution=256`, `cleanup=presentation`, and a baked `2048` base-color texture. Use `--texture-mode vertex_color` when you want the faster lightweight export, and `--cleanup none` only when you want the raw marching-cubes mesh for comparison.

For front-view portraits and other roughly left-right-symmetric subjects, add mirror-based texture completion:

```bash
abstract3d i23d ./portrait.png \
  --output-dir ./out/portrait-triposr \
  --backend triposr \
  --device mps \
  --mc-resolution 256 \
  --texture-mode baked_basecolor \
  --texture-completion mirror_symmetry
```

When you have a credible auxiliary view, you can also feed it into the texture bake:

```bash
abstract3d i23d ./portrait.png \
  --output-dir ./out/portrait-triposr-multiview \
  --backend triposr \
  --device mps \
  --mc-resolution 256 \
  --texture-mode baked_basecolor \
  --texture-completion mirror_symmetry \
  --texture-reference-image ./portrait-side-left.png \
  --texture-reference-angle side_left
```

Generate a mesh from an image with explicit TripoSR settings:

```bash
abstract3d i23d ./object.png \
  --output-dir ./out/object-triposr \
  --backend triposr \
  --model stabilityai/TripoSR \
  --device mps \
  --mc-resolution 256 \
  --cleanup presentation \
  --texture-mode baked_basecolor \
  --texture-resolution 2048 \
  --format glb
```

That explicit form is useful when you want a reproducible validated command in scripts or docs.

Generate the lightweight vertex-color variant explicitly:

```bash
abstract3d i23d ./object.png \
  --output-dir ./out/object-triposr-fast \
  --backend triposr \
  --device mps \
  --mc-resolution 256 \
  --cleanup presentation \
  --texture-mode vertex_color \
  --format glb
```

Generate a mesh with the license-gated Hunyuan3D-2.1 backend (strongest local geometry; review
the Tencent Hunyuan Community License first — it excludes the EU, UK, and South Korea):

```bash
ABSTRACT3D_HUNYUAN_ACCEPT_LICENSE=1 abstract3d i23d ./object.png \
  --output-dir ./out/object-hunyuan \
  --backend hunyuan3d21 \
  --device mps \
  --num-inference-steps 30
```

When you have photos from several angles, use the multi-view checkpoint
(`tencent/Hunyuan3D-2mv`, same license gate) so the extra views constrain the geometry itself,
not just the texture. References whose angles snap to the trained front/left/back/right slots
condition the shape; all references feed the texture bake:

```bash
ABSTRACT3D_HUNYUAN_ACCEPT_LICENSE=1 abstract3d i23d ./portrait.png \
  --output-dir ./out/portrait-multiview \
  --backend hunyuan3d21 \
  --model tencent/Hunyuan3D-2mv \
  --device mps \
  --num-inference-steps 30 \
  --texture-completion mirror_symmetry \
  --texture-reference-image ./portrait-left-profile.png \
  --texture-reference-angle side_left \
  --texture-reference-image ./portrait-right-profile.png \
  --texture-reference-angle side_right
```

Generate a mesh from text through AbstractVision composition:

```bash
abstract3d t23d "a ceramic teapot with a curved spout and matte glaze" \
  --output-dir ./out/teapot \
  --backend triposr \
  --device mps
```

Reproduce the checked Apple-local proof lane explicitly:

```bash
abstract3d t23d "a ceramic teapot with a curved spout and matte glaze" \
  --output-dir ./out/teapot-proof \
  --backend triposr \
  --device mps \
  --image-provider mlx-gen \
  --image-model AbstractFramework/flux.2-klein-4b-8bit
```

Use `--texture-resolution 4096` only for slower hero-asset exports. On the checked local proof objects it increased texture time and file size materially, but did not improve quality enough to justify changing the default.

Run the checked local validated proof suite:

```bash
python scripts/validate_local.py --backend triposr --device mps --mc-resolution 256
```

The checked textured TripoSR proof lane is `docs/assets/validation/triposr-texture-proof/`. The older full-suite geometry baseline remains under `docs/assets/validation/local-triposr/`. Experimental Step1X reference lanes remain documented in [`docs/benchmarks.md`](docs/benchmarks.md).

Publish docs-ready proof assets only when you explicitly want to replace or add checked assets:

```bash
python scripts/validate_local.py \
  --backend triposr \
  --device mps \
  --mc-resolution 256 \
  --publish-doc-assets ./docs/assets/validation/my-triposr-run
```

## Python Usage

```python
from abstract3d import Scene3DManager

scene3d = Scene3DManager(backend_id="triposr")

image_result = scene3d.i23d(
    "./object.png",
    output_dir="./out/object-triposr",
    format="glb",
    device="mps",
)

text_result = scene3d.t23d(
    "a mid-century lounge chair with walnut legs and woven fabric",
    output_dir="./out/chair-triposr",
    device="mps",
)
```

## Documentation

- Published doc site: [lpalbou.github.io/abstract3d](https://lpalbou.github.io/abstract3d/)
- Docs index: [`docs/README.md`](docs/README.md)
- Getting started: [`docs/getting-started.md`](docs/getting-started.md)
- Architecture: [`docs/architecture.md`](docs/architecture.md)
- API and CLI: [`docs/api.md`](docs/api.md)
- Methodology: [`docs/methodology.md`](docs/methodology.md)
- Model strategy: [`docs/models.md`](docs/models.md)
- Benchmarks and proof assets: [`docs/benchmarks.md`](docs/benchmarks.md)
- Focused texture proof: [`docs/assets/validation/triposr-texture-proof/contact_sheet.png`](docs/assets/validation/triposr-texture-proof/contact_sheet.png)
- Focused portrait texture completion proof: [`docs/assets/validation/triposr-portrait-texture-proof/contact_sheet.png`](docs/assets/validation/triposr-portrait-texture-proof/contact_sheet.png)
- AbstractCore integration: [`docs/integration-abstractcore.md`](docs/integration-abstractcore.md)
- FAQ: [`docs/faq.md`](docs/faq.md)
- Troubleshooting: [`docs/troubleshooting.md`](docs/troubleshooting.md)
- ADRs: [`docs/adr/README.md`](docs/adr/README.md)

## Limits

- The validated backend is not a native text-only 3D model. `t23d` is composed as:
  1. text to image in `abstractvision`
  2. image to 3D in the selected local backend
- The validated TripoSR path now emits UV-baked base-color textures by default, but the visible gain is bounded by the underlying reconstructed color field. A larger atlas does not guarantee dramatically sharper materials.
- On single-view textured exports, the baked TripoSR path now prefers observed-view reprojection on front-facing texels and falls back to the reconstructed color field elsewhere. This improves texture placement on view-aligned details, but it does not solve weak geometry reconstruction.
- On single-view portraits and other symmetric front-view subjects, TripoSR can also apply `mirror_symmetry` texture completion to fill some uncovered front-side texels from the visible half. This improves hidden-side texture modestly, but it does not replace portrait-specific geometry reconstruction.
- The Step1X backend is experimental, geometry-only, and slower than the validated TripoSR path on the checked Apple-local benchmark cases.
- The current checked Step1X Apple-local reference lane is [`docs/assets/validation/local-step1x-dynamic/contact_sheet.png`](docs/assets/validation/local-step1x-dynamic/contact_sheet.png). In the current four-case suite, all four cases complete under guarded Apple-local execution, the teapot is recognizable, the chair is materially better than the blob baseline but still below bar, the espresso `i23d` case is materially better on the automatic base-checkpoint fallback, and the owl remains weak.
- The espresso checkpoint comparison is published under [`docs/assets/validation/step1x-espresso-checkpoint-comparison/contact_sheet.png`](docs/assets/validation/step1x-espresso-checkpoint-comparison/contact_sheet.png).
- A focused rocket `i23d` comparison across the two shipped local backends is published under [`docs/assets/validation/rocket-i23d-comparison/comparison_contact_sheet.png`](docs/assets/validation/rocket-i23d-comparison/comparison_contact_sheet.png).
- A focused TripoSR raw-vs-clean rocket proof is published under [`docs/assets/validation/triposr-rocket-cleanup/comparison_contact_sheet.png`](docs/assets/validation/triposr-rocket-cleanup/comparison_contact_sheet.png).
- A focused TripoSR texture proof comparing regular vertex-color output against baked texture output is published under [`docs/assets/validation/triposr-texture-proof/contact_sheet.png`](docs/assets/validation/triposr-texture-proof/contact_sheet.png).
- An official-like higher-quality chair stress run (`50` steps, guidance `7.5`, octree `384`) did not complete on this Apple `mps` lane, so upstream-like high-quality settings are not yet operationally safe here.
- Multi-object scenes, cluttered backgrounds, and strong occlusion are out of scope for the validated path.
- TRELLIS.2 is an accepted experimental backend, but it stays outside the permissive validated path because its required companion model (`facebook/dinov3-vitl16-pretrain-lvd1689m`) is gated behind Meta's DINOv3 License: operators must request and accept access themselves (commercial use permitted; "Built with DINOv3" attribution required on distribution; military/trade-control uses prohibited). See [`docs/models.md`](docs/models.md) for the license summary and the unblock steps.
- The Hunyuan3D-2.1 backend is experimental and license-gated: the Tencent Hunyuan Community License excludes the EU, UK, and South Korea, so the backend requires an explicit operator acknowledgment (`ABSTRACT3D_HUNYUAN_ACCEPT_LICENSE=1`) before it downloads or runs official weights. Its geometry is the strongest in the local catalog on the checked proof objects; its textures come from the shared projection bake, which is exact where the photo sees the surface and approximate elsewhere.
