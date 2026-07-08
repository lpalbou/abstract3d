# Getting Started

## Choose Your Profile

Backend cheat sheet:

- `triposr`: validated default, fastest, permissive MIT stack.
- `hunyuan3d21`: strongest geometry, experimental, requires an explicit license acknowledgment
  (`ABSTRACT3D_HUNYUAN_ACCEPT_LICENSE=1`; the Tencent license excludes the EU, UK, and South Korea).
  With multiple photos, add `--model tencent/Hunyuan3D-2mv` so front/left/back/right reference
  views condition the shape itself (same license gate).
- `step1x`: experimental geometry-only backend.
- `trellis2`: accepted experimental backend; requires you to request and accept Meta's gated
  DINOv3 License first (commercial use permitted, attribution and acceptable-use terms apply —
  see the license summary in `docs/models.md`).

Lightweight base package:

```bash
pip install abstract3d
```

That installs the shared `scene3d` surface plus the lightweight `abstractvision` package contract.
It is the right profile when you want remote OpenAI or OpenAI-compatible image composition for
`t23d`, or when a host environment owns the actual scene3d backend selection.

Validated `i23d`:

```bash
pip install "abstract3d[triposr]"
```

Experimental Step1X geometry runtime:

```bash
pip install "abstract3d[step1x]"
```

Compatibility alias for callers that still request the historical composed `t23d` extra:

```bash
pip install "abstract3d[t23d]"
```

Backend-only extras stay honest `i23d` runtime installs. `pip install abstract3d` already includes
the provider-neutral `abstractvision` composition contract; add `apple` or `gpu` when you actually
want local composed text-to-3D.

Apple-local profile with `abstractvision`, the validated TripoSR path, and the experimental Step1X path:

```bash
pip install "abstract3d[apple]"
```

GPU-local profile with `abstractvision`, the validated TripoSR path, and the experimental Step1X path:

```bash
pip install "abstract3d[gpu]"
```

AbstractCore host integration:

```bash
pip install "abstractcore[scene3d]"
```

## First Successful Runs

Generate a mesh with the validated backend:

```bash
abstract3d i23d ./object.png --output-dir ./out/object --device mps --format glb
```

For TripoSR, that command now defaults to `mc_resolution=256`, `cleanup=presentation`, and `texture_mode=baked_basecolor` at `texture_resolution=2048`.

For front-view portraits and other symmetric subjects, start with:

```bash
abstract3d i23d ./portrait.png \
  --output-dir ./out/portrait \
  --device mps \
  --texture-completion mirror_symmetry
```

Generate the faster lightweight variant explicitly:

```bash
abstract3d i23d ./object.png \
  --output-dir ./out/object-fast \
  --device mps \
  --format glb \
  --texture-mode vertex_color
```

Generate a mesh with the experimental Step1X geometry backend:

```bash
abstract3d i23d ./object.png \
  --output-dir ./out/object-step1x \
  --backend step1x \
  --model stepfun-ai/Step1X-3D \
  --device mps \
  --mc-resolution 128 \
  --format glb
```

On Apple `mps`, Step1X now defaults to an 8-step lower-guidance recovery profile. Thin-structure prompts such as chairs also auto-raise the Step1X octree resolution to `192`, and exported Step1X meshes use a canonical upright frame by default.

Generate a mesh from text through AbstractVision composition:

```bash
abstract3d t23d "a ceramic teapot with a curved spout and matte glaze" \
  --output-dir ./out/teapot \
  --device mps
```

That path uses the configured `abstractvision` provider/model unless you override it. `abstract3d` resolves composed-image settings in this order:

- explicit `--image-provider` / `--image-model`
- `scene3d_image_provider` / `scene3d_image_model`
- `ABSTRACT3D_IMAGE_PROVIDER` / `ABSTRACT3D_IMAGE_MODEL`
- the configured `abstractvision` default

Reproduce the checked Apple-local proof lane explicitly:

```bash
abstract3d t23d "a ceramic teapot with a curved spout and matte glaze" \
  --output-dir ./out/teapot-proof \
  --device mps \
  --image-provider mlx-gen \
  --image-model AbstractFramework/flux.2-klein-4b-8bit
```

Each successful run writes a bundle directory with:

- `scene.glb`
- `scene.obj`
- `input.png`
- `preview.png`
- `contact_sheet.png`
- `metadata.json`

Textured TripoSR runs also write:

- `texture.png`
- `uv_preview.png`
- `scene.mtl` plus texture sidecars for OBJ-compatible toolchains

## Validation Suites

Run the checked TripoSR proof suite:

```bash
python scripts/validate_local.py --backend triposr --device mps --mc-resolution 256
```

Run the checked Step1X proof suite:

```bash
python scripts/validate_local.py --backend step1x --device mps --mc-resolution 128
```

Those commands generate:

- per-case bundles under `artifacts/validation/local-<backend>/`
- docs-ready proof assets only when `--publish-doc-assets <dir>` is passed explicitly

For Step1X, the raw default validation output without an explicit `--output-dir` or `--model-subfolder` lands under `artifacts/validation/local-step1x-label/`. The checked docs-ready Apple-local reference lane is published separately under `docs/assets/validation/local-step1x-dynamic/`.

## First-Run Downloads

Measured local footprint on the checked Apple-local machine:

- `stabilityai/TripoSR`: `1.6G`
- pinned TripoSR source snapshot: `63M`
- `stepfun-ai/Step1X-3D` geometry subset: `6.8G`
- pinned Step1X source snapshot: `147M`
- optional Apple-local proof image model cache (`AbstractFramework/flux.2-klein-4b-8bit` via `mlx-gen`): `8.0G`

Step1X notes:

- the supported official geometry checkpoint family is `Step1X-3D-Geometry-1300m` plus `Step1X-3D-Geometry-Label-1300m`
- the current Apple-local reference lane defaults to the label-geometry checkpoint
- the runtime stays local and uses a pinned official source snapshot
- the Apple `mps` path uses `float32`
- the Apple `mps` label-geometry path applies deterministic CPU mesh cleanup after extraction, vertex welding, and export-axis canonicalization
- thin-structure prompts such as chairs auto-raise Step1X octree resolution to `192`
- the Apple `mps` path defaults to non-resident Step1X execution and a conservative per-process MPS memory cap so mesh export/render does not keep the full model loaded unnecessarily
- the Step1X Apple-local validator now isolates each case in its own subprocess and defaults to a `64 GiB` RSS guard
- the texture stage is not installed or used by the supported backend

Optional background removal can also download a `rembg` model on first use. Step1X now enables opaque-image removal by default on Apple-local recovery runs; only force `remove_background=False` when you already have a clean alpha mask and deliberately want to override that behavior.

TripoSR notes:

- the validated backend now applies a deterministic CPU cleanup pass after marching-cubes extraction
- the validated backend now also bakes a UV base-color texture by default at `2048`
- the validated backend can optionally use `mirror_symmetry` texture completion for single front-view symmetric subjects
- `4096` is available as an explicit hero-mode override, but the checked proof objects did not improve enough to justify promoting it to the default
- the default cleanup profile removes small disconnected components, runs marching-cubes-specific cleanup, applies light Taubin smoothing, repairs non-manifold cases when possible, and repairs normals
- the focused raw-vs-clean rocket proof for this cleanup pass is published under [docs/assets/validation/triposr-rocket-cleanup/comparison_contact_sheet.png](assets/validation/triposr-rocket-cleanup/comparison_contact_sheet.png)
- the focused texture proof comparing regular vertex-color and baked texture output is published under [docs/assets/validation/triposr-texture-proof/contact_sheet.png](assets/validation/triposr-texture-proof/contact_sheet.png)
- the focused portrait texture-completion proof is published under [docs/assets/validation/triposr-portrait-texture-proof/contact_sheet.png](assets/validation/triposr-portrait-texture-proof/contact_sheet.png)

## What To Expect

- TripoSR remains the validated default path in this repo.
- Step1X is available locally and benchmarked, but it remains experimental.
- The checked Apple-local Step1X reference suite under `docs/assets/validation/local-step1x-dynamic/` is materially better than the old blob-like baseline, but chair and owl remain below a production bar.
- A higher-quality chair stress profile (`50` steps, guidance `7.5`, octree `384`) did not complete on this Apple `mps` lane.
- On the checked Apple-local benchmark cases, Step1X produced much denser meshes and took longer than TripoSR, while TripoSR produced more recognizable object shapes.

See [Methodology](methodology.md) for the command patterns and review criteria, and [Benchmarks](benchmarks.md) for the checked contact sheets and comparison assets.

## Next Reading

- [Architecture](architecture.md)
- [API and CLI](api.md)
- [Model strategy](models.md)
- [Benchmarks](benchmarks.md)
- [Troubleshooting](troubleshooting.md)
