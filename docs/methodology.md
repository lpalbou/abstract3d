# Methodology

This page explains how `abstract3d` generates, measures, and reviews local 3D outputs in this repository.

## Generation Methodologies

`abstract3d` uses three workflows:

- validated `i23d`: a single centered object image goes directly into the local TripoSR backend
- composed `t23d`: text is rendered to an image through `abstractvision`, then reconstructed by the selected `i23d` backend
- experimental Step1X `i23d`: the official Step1X geometry checkpoints run locally, with geometry output only and no texture stage

The validated default remains TripoSR. Step1X is available for local geometry experiments and focused comparisons, but it is not promoted to the validated default path.

## Apple-Local Operating Profile

The checked Apple Silicon profile follows these rules:

- run one heavy inference job at a time
- keep TripoSR as the baseline for recognizable object reconstruction
- use TripoSR `mc_resolution=256` as the quality-oriented default when callers do not override it
- apply deterministic TripoSR CPU postprocessing after marching-cubes extraction
- use TripoSR `baked_basecolor@2048` as the validated texture default
- reserve `texture_resolution=4096` for explicit slower hero-asset exports only
- keep Step1X on `mps` `float32`
- apply a conservative per-process Step1X MPS cap
- release Step1X runtime memory before export and preview rendering
- use CPU-side Step1X surface extraction and cleanup after denoising
- use deterministic cleanup, vertex welding, and stable-pose export canonicalization on Step1X

The current TripoSR cleanup pass is intentionally bounded. It is a geometry cleanup stage, not a second reconstruction model. The shipped profile applies:

- small disconnected-component pruning
- MeshLab marching-cubes-specific cleanup
- light Taubin smoothing
- non-manifold repair
- small-hole repair when possible
- final normal repair

When the baked texture path is enabled, the shipped TripoSR methodology is:

- extract cleaned geometry first
- unwrap the mesh into a UV atlas
- rasterize per-texel world positions from the atlas
- project the observed input view back onto front-facing texels when that view supports them
- optionally project extra curated reference views onto additional texels
- optionally apply `mirror_symmetry` completion for uncovered front-side texels on front-view symmetric subjects
- fall back to the TripoSR color field for the remaining texels
- export a textured GLB plus atlas previews and OBJ sidecars

For proof suites, `scripts/validate_local.py` can also isolate Step1X cases in subprocesses and apply an RSS guard so one pathological run does not take down the full suite.

## Hunyuan3D-2.1 Operating Profile

The license-gated Hunyuan3D-2.1 backend follows these checked Apple-local rules:

- `float16` on `mps`, official defaults `guidance_scale=5.0` and `octree_resolution=384`
- the checked proof lane uses `num_inference_steps=30` (community quality/speed sweet spot; official default is `50`)
- adaptive coarse-to-fine volume decoding with host-side bookkeeping (the upstream hierarchical decoder loses thin structures and misbehaves on `mps`)
- quadric decimation to a `120000`-face budget before texture bake: beyond that, marching-cubes micro-detail fragments the UV atlas into thousands of tiny charts and produces salt-and-pepper texel noise
- textures come from the shared projection bake with source-pose estimation, since Hunyuan reconstructs in a canonical object frame and the photo's viewpoint must be recovered before projection

## Multi-View Capture Guidance

When several photos of the subject are available, prefer the multi-view path (`--model tencent/Hunyuan3D-2mv` on the Hunyuan backend):

- shoot as close to the trained slots as possible: front (0°), left (+90°), right (-90°), back (180°); the snap tolerance is 25°
- CHIRALITY MATTERS: `side_left` (+90) expects the camera on the subject's LEFT — for a face, the nose points image-LEFT in that photo. A swapped pair paints each side with the other side's content, and the disagreement gate cannot reliably detect the swap on near-symmetric subjects. Verify labels against the photo content, not filenames.
- declared angles should be measured, not assumed: a "three-quarter" photo is often +15..+30, and declaring +45 measurably hurts the bake
- side photos must show the same subject state (hair, clothing, expression, lighting); synthesized or inconsistent side views are attenuated by the texture QA gate but still cost geometry quality when used for conditioning
- off-slot angles are not used for geometry but still improve the texture bake at their true angle
- the checked face proof (front + both profiles): observed texture coverage 0.19 -> 0.91 pre-gating (final bundles report lower observed ratios because low-confidence mirror fill is now excluded), hallucinated bald back of the head replaced by a hair-consistent shape (post-ADR-0007 bundle: `artifacts/validation/iter3-multiview-fixed/face-2mv/`)
- reference photos should share one segmentation: when synthesizing an opposite-side view by mirroring, mirror the SEGMENTED image (alpha included) so both sides carry identical mattes, and record it as fabricated (mirrored illumination, lost asymmetries)

## Review Methodology

Every successful run writes a bundle directory with:

- `scene.glb`
- `scene.obj`
- `input.png`
- `preview.png`
- `contact_sheet.png`
- `metadata.json`

Textured TripoSR runs also include:

- `texture.png`
- `uv_preview.png`
- OBJ sidecars such as `scene.mtl`

The review order used in this repo is:

1. inspect the contact sheet first
2. check whether the object stays recognizable across the rendered views
3. inspect the texture crop or atlas when the baked path is enabled
4. check orientation, support geometry, and whether thin parts collapse
5. inspect `metadata.json` for timings, mesh density, topology, texture mode, preview renderer, and memory

The most important quality checks are:

- silhouette fidelity
- stable upright axes
- topology health and disconnected-body count
- mesh density relative to the object class
- visual recognizability from multiple angles

### Proof-artifact publication checklist (texture bundles)

Certified proof bundles (e.g. the face multi-view asset) are only
overwritten from a verified staging copy. The checklist, in order:

1. Bake with the CANONICAL RECIPE for the asset. For the face proof:
   source view = `remove_background_robust(input.png)` as `rgba` plus the
   RAW `input.png` as `identity_image`, left/right `_clean` profile
   references at +/-90, `texture_completion="auto"`,
   `projection_model="orthographic"`, resolution 2048. Bakes without
   `identity_image` register the repair correspondence in a measurably
   different basin and must never ship.
2. Verify determinism when the pipeline is expected deterministic:
   re-bake and compare texture md5 (one hash per recipe per tree).
3. Stage the full bundle (texture.png, scene.glb, obj sidecars with
   NEUTRAL material factors — `Kd 1 / Ks 0`, viewers multiply `map_Kd`
   by `Kd` — preview, contact sheet, uv preview, metadata with the
   publication block).
4. Run ALL harnesses on the exact staged bytes before overwriting:
   the raw 28-view battery (raw MAE within budget, every detector
   green), the compensated identity report, and `scripts/texture_qa.py`
   (13/13). Record the results in the metadata publication block.
5. Re-run the same harnesses on the artifact directory after the copy
   (certify the exact published bytes, not the staging copy).
6. Frozen assets (ship/owl proof bundles) stay byte-identical: after any
   texturing change, re-bake both canaries with their certified recipes
   (the ship uses `source_pose_override=(30, 15)`) and compare md5
   against the on-disk textures.

## Metrics

The checked benchmark and focused-comparison reports use these metrics:

- preprocessing time
- inference time
- mesh extraction and cleanup time
- texture bake time
- total generation time
- final vertex and face counts
- output bundle size
- process RSS
- final MPS allocated memory

Contact sheets are the primary qualitative proof. `metadata.json`, `summary.json`, and `summary.md` provide the quantitative side.

## Command Patterns

Generate a 3D object from one image with the validated backend:

```bash
abstract3d i23d ./object.png \
  --output-dir ./out/object \
  --backend triposr \
  --device mps \
  --format glb \
  --remove-background
```

That validated command now resolves to the baked TripoSR texture path by default.

For front-view portraits or other symmetric subjects, use mirror completion explicitly:

```bash
abstract3d i23d ./portrait.png \
  --output-dir ./out/portrait \
  --backend triposr \
  --device mps \
  --mc-resolution 256 \
  --texture-mode baked_basecolor \
  --texture-completion mirror_symmetry
```

When you have one credible extra view, add it as a texture reference instead of replacing the source image:

```bash
abstract3d i23d ./portrait.png \
  --output-dir ./out/portrait-multiview \
  --backend triposr \
  --device mps \
  --mc-resolution 256 \
  --texture-mode baked_basecolor \
  --texture-completion mirror_symmetry \
  --texture-reference-image ./portrait-side-left.png \
  --texture-reference-angle side_left
```

Generate the lighter vertex-color variant explicitly:

```bash
abstract3d i23d ./object.png \
  --output-dir ./out/object-fast \
  --backend triposr \
  --device mps \
  --mc-resolution 256 \
  --cleanup presentation \
  --texture-mode vertex_color \
  --remove-background
```

Inspect the raw TripoSR marching-cubes mesh without the cleanup pass:

```bash
abstract3d i23d ./object.png \
  --output-dir ./out/object-raw \
  --backend triposr \
  --device mps \
  --mc-resolution 256 \
  --cleanup none \
  --remove-background
```

Generate a 3D object from one image with the experimental Step1X backend:

```bash
abstract3d i23d ./object.png \
  --output-dir ./out/object-step1x \
  --backend step1x \
  --model stepfun-ai/Step1X-3D \
  --device mps \
  --mc-resolution 128 \
  --format glb \
  --remove-background
```

Generate a 3D object from text through composed `t23d`:

```bash
abstract3d t23d "a silver toy rocket with a pointed nose cone and three fins" \
  --output-dir ./out/rocket \
  --backend triposr \
  --device mps
```

For the checked Apple-local proof lane only, the same command can be pinned to the explicit local image generator used on that machine:

```bash
abstract3d t23d "a silver toy rocket with a pointed nose cone and three fins" \
  --output-dir ./out/rocket-proof \
  --backend triposr \
  --device mps \
  --image-provider mlx-gen \
  --image-model AbstractFramework/flux.2-klein-4b-8bit
```

Run the checked proof suites:

```bash
python scripts/validate_local.py --backend triposr --device mps --mc-resolution 256
python scripts/validate_local.py --backend step1x --device mps --mc-resolution 128
```

Build the focused TripoSR texture proof package:

```bash
python scripts/triposr_texture_proof.py \
  --output-dir ./artifacts/validation/triposr-texture-proof \
  --publish-dir ./docs/assets/validation/triposr-texture-proof
```

## Focused Comparison Example

This repository also keeps focused single-image comparisons when a specific object class matters. The checked rocket `i23d` comparison is published under [Benchmarks](benchmarks.md) and uses the same artifact contract as the full proof suites:

- per-backend `scene.glb`
- per-backend `contact_sheet.png`
- a combined comparison sheet
- machine-readable summary metadata

The portrait hidden-side texture study follows the same pattern and is also published under [Benchmarks](benchmarks.md).

## Related Docs

- [Getting started](getting-started.md)
- [API and CLI](api.md)
- [Model strategy](models.md)
- [Benchmarks and validation](benchmarks.md)
- [Troubleshooting](troubleshooting.md)
