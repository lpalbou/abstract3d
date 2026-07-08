# Troubleshooting

## `t23d` fails with an AbstractVision dependency error

Cause:

- `abstractvision` is not installed
- or no remote/local image provider is configured

Fix:

```bash
pip install abstract3d
```

Then either:

- rely on the configured `abstractvision` default provider/model
- set `scene3d_image_provider` / `scene3d_image_model`
- set `ABSTRACT3D_IMAGE_PROVIDER` / `ABSTRACT3D_IMAGE_MODEL`
- pass `--image-provider` / `--image-model` explicitly for one command

If you want local image composition in the same environment, install a platform profile:

```bash
pip install "abstract3d[apple]"
# or
pip install "abstract3d[gpu]"
```

## Step1X outputs a dense blob or inflated mesh

Cause:

- the Step1X backend in this repo is still experimental
- it is geometry-only
- the historical 4-step Apple-local proof was bad enough to be preserved as a failure baseline
- even with the current MPS-tuned recovery defaults, some `t23d` cases still underperform on object-centric studio inputs

Fix:

- start with the current Step1X Apple-local defaults; do not force the old 4-step profile

```bash
abstract3d i23d ./object.png --output-dir ./out/object-step1x --backend step1x --device mps --mc-resolution 128
```

- the Apple-local Step1X runtime already welds vertices and writes canonical upright exports, so compare the resulting `scene.glb` before assuming the issue is only a viewer-axis problem
- thin-structure prompts such as chairs automatically raise Step1X octree resolution to `192`; do not override that downward unless you are measuring a regression intentionally
- on Apple `mps`, Step1X now drops the runtime before mesh export/render and applies a conservative memory cap by default; if you override that behavior, expect much higher unified-memory pressure
- the Step1X Apple-local validator also defaults to a `64 GiB` RSS guard per case; if a case still dies, inspect its `result.json` instead of rerunning blindly
- compare the same input against the validated TripoSR path

```bash
abstract3d i23d ./object.png --output-dir ./out/object-triposr --backend triposr --device mps --mc-resolution 256
```

- keep one centered subject with a clear silhouette
- avoid clutter, heavy cropping, and strong self-occlusion
- if a refreshed Step1X suite still leaves chair-like thin structures or carved figurines unrecognizable, treat that as a model limitation on this Apple-local lane, not just an axis/export bug
- for Step1X `t23d`, inspect the saved `input.png`; if the composed image is not a strong product-style object shot, rerun or use a manually curated image through `i23d`
- treat Step1X as an experimental option, not the validated default

See [Benchmarks](benchmarks.md) for the checked comparison contact sheet.

## Step1X source bootstrap fails

Cause:

- the pinned official Step1X source snapshot could not be cloned
- `ABSTRACT3D_STEP1X_SOURCE_DIR` points at the wrong directory

Fix:

- ensure the directory contains `step1x3d_geometry`
- ensure it also contains `.abstract3d-source.json`
- or unset `ABSTRACT3D_STEP1X_SOURCE_DIR` and let Abstract3D bootstrap the pinned official snapshot

When a valid external Step1X source snapshot is provided, Abstract3D uses a managed patched copy for runtime use rather than rewriting the original source tree in place.

## Hunyuan3D-2.1 refuses to run with a license acknowledgment error

The Hunyuan3D-2.1 backend is intentionally gated because the Tencent Hunyuan 3D 2.1 Community
License is territory-restricted (it excludes the European Union, the United Kingdom, and South
Korea) and adds terms for large-scale commercial use.

1. Review the license at `https://huggingface.co/tencent/Hunyuan3D-2.1/blob/main/LICENSE`.
2. If its terms apply to you, opt in explicitly:

```bash
export ABSTRACT3D_HUNYUAN_ACCEPT_LICENSE=1
# or scene3d_hunyuan_license_accepted=true in the owner config
```

The gate is enforced on every path that downloads or runs official weights, so there is no
setting that skips the review step silently.

## TRELLIS.2 fails while loading the official DINOv3 companion model

Cause:

- the official companion model `facebook/dinov3-vitl16-pretrain-lvd1689m` is gated behind the
  DINOv3 License (Meta): commercial use is permitted, but access requires an approved request,
  redistribution must carry the license plus a "Built with DINOv3" notice, and
  military/trade-control end uses are prohibited
- your environment does not have approved access yet, or is not authenticated
- or `ABSTRACT3D_TRELLIS2_DINO_MODEL` points at an incomplete local snapshot

Fix:

1. Sign in to Hugging Face, open the model page, review and accept the DINOv3 License
   (`https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m`). Meta reviews requests,
   typically within a few days; approval arrives by email.
2. Authenticate this machine (`hf auth login`) and retry; Abstract3D caches the companion
   model locally.
3. Alternatively point `ABSTRACT3D_TRELLIS2_DINO_MODEL` at a local authorized snapshot
   directory that contains:
   - `config.json`
   - `model.safetensors`
   - `preprocessor_config.json`
4. Do not substitute contributor mirrors or alternate encoder repositories; the backend
   rejects them intentionally (ADR 0003). Accepting the DINOv3 License is your acknowledgment
   as the operator; Abstract3D cannot accept it on your behalf.

## The mesh is incomplete or blobby

Cause:

- the source image is not object-centric
- the silhouette is weak
- the object is cropped

Fix:

- use one centered subject
- keep the object fully visible
- prefer neutral backgrounds
- disable extra props and clutter

For TripoSR specifically:

- start with the validated default profile before pushing the marching-cubes resolution higher

```bash
abstract3d i23d ./object.png --output-dir ./out/object-triposr --backend triposr --device mps --mc-resolution 256
```

- compare the cleaned export against the raw marching-cubes export if the result still looks faceted in your viewer

```bash
abstract3d i23d ./object.png --output-dir ./out/object-triposr-raw --backend triposr --device mps --mc-resolution 256 --cleanup none
```

- if the raw mesh is materially worse than the cleaned one, the issue is mainly mesh cleanup and shading rather than model collapse
- if both raw and cleaned meshes are still lumpy, the limitation is mostly the reconstructed field, not just the postprocess

## TripoSR textured output is softer than expected

Cause:

- the baked atlas only records the color detail already present in the reconstructed TripoSR field
- raising atlas resolution cannot restore detail that the source field never recovered

Fix:

- start with the validated default textured path first

```bash
abstract3d i23d ./object.png --output-dir ./out/object-triposr --backend triposr --device mps --mc-resolution 256
```

- compare against the lighter vertex-color export when you want to isolate whether the issue is the baked atlas or the underlying field

```bash
abstract3d i23d ./object.png --output-dir ./out/object-triposr-fast --backend triposr --device mps --mc-resolution 256 --texture-mode vertex_color
```

- use `--texture-resolution 4096` only for deliberate hero-asset tests; on the checked local proof objects it increased runtime and file size materially without enough quality gain to change the validated default
- on single-view portraits or other strongly view-dependent inputs, the baked path now reprojects the observed input view onto front-facing texels first; if the face or silhouette is still wrong after that, the remaining issue is geometry reconstruction rather than atlas resolution
- inspect the focused texture proof under [Benchmarks](benchmarks.md) before assuming the issue is just viewer-side filtering

## Hidden-side portrait texture is weak

Cause:

- TripoSR reconstructs geometry from one image, so the far side of a portrait still depends heavily on inferred shape
- the default baked texture path only reprojects what the source view can actually see

Fix:

- start with mirror-based completion on the validated TripoSR lane

```bash
abstract3d i23d ./portrait.png \
  --output-dir ./out/portrait \
  --backend triposr \
  --device mps \
  --mc-resolution 256 \
  --texture-mode baked_basecolor \
  --texture-completion mirror_symmetry
```

- if you already have one good auxiliary view, add it with `--texture-reference-image` and `--texture-reference-angle`
- treat extra reference views as a texture aid, not a replacement for multi-view geometry reconstruction
- inspect the focused portrait proof under [Benchmarks](benchmarks.md); the hidden-side improvement is real but still bounded by TripoSR geometry

## First run is slower than later runs

Cause:

- model downloads and model loading are one-time costs

Fix:

- warm the caches before benchmarking
- compare steady-state case times separately from first-load time

See [Benchmarks](benchmarks.md) for the profile used in the checked proof runs.

## Background removal downloads extra weights

Cause:

- preprocessing can trigger `rembg` downloads when the input image has no alpha channel and automatic removal is enabled

Fix:

- allow the one-time download if you need automatic foreground extraction
- only force `remove_background=False` when your input already has a clean alpha mask and you are deliberately overriding the default

## MPS behaves poorly on your case

Fix:

- for Step1X `t23d`, the runtime now releases MLX cache before geometry inference; if you still hit a bad case, save the composed image and retry it through `i23d`
- retry with `device="cpu"`
- reduce `mc_resolution`
- keep the source image to one centered object
- for Step1X, remember that the backend already falls back to `float32` on `mps`
