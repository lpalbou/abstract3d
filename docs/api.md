# API And CLI

## Python Surface

The public entry point is `Scene3DManager`.

```python
from abstract3d import Scene3DManager

scene3d = Scene3DManager(backend_id="triposr")

result = scene3d.i23d(
    "./object.png",
    output_dir="./out/object-triposr",
    format="glb",
    device="mps",
)
```

Useful methods:

- `available_providers(task=None)`
- `list_models(task=None, provider=None)`
- `list_operations(task=None)`
- `load_resident_model(request)`
- `list_loaded_models(filters=None)`
- `unload_resident_model(request)`
- `t23d(prompt, **kwargs)`
- `i23d(image, **kwargs)`
- `generate(prompt="", task=None, **kwargs)`
- `validate_suite(...)`

Built-in backend ids are:

- `abstract3d:triposr` or `triposr`
- `abstract3d:step1x-local` or `step1x`
- `abstract3d:hunyuan3d21-local` or `hunyuan3d21` / `hunyuan3d` / `hunyuan` (license-gated)
- `abstract3d:trellis2-local` or `trellis2`

## Mesh Operations (`abstract3d.mesh_ops`)

Deterministic operations over any supported mesh file (GLB/GLTF/OBJ/STL/PLY/
OFF/3MF/DAE in; GLB/OBJ/STL/PLY/OFF out). No model runtime; requires the
lightweight `abstract3d[mesh]` extra.

```python
from abstract3d.mesh_ops import (
    analyze_mesh, transform_mesh, compose_scene,
    convert_mesh, repair_mesh, render_preview,
)

report = analyze_mesh("./scene.glb")          # MeshReport dataclass
print(report.summary())

transform_mesh("./scene.glb", "./big.glb", scale=2.0, rotate_deg=[0, 90, 0])
compose_scene(
    [
        {"path": "./teapot.glb", "translate": [-0.5, 0, 0]},
        {"path": "./cup.glb", "translate": [0.5, 0, 0], "scale": 1.2},
    ],
    "./scene.glb",
)
convert_mesh("./scene.glb", "./scene.stl")
repair_mesh("./scan.glb", "./scan_clean.glb")
render_preview("./scene.glb", "./scene_preview.png")
```

Conventions: files are processed in their native file frame; transform
composition order is scale -> mirror -> rotate X, Y, Z -> translate about the
file-frame origin (`center=True` re-centers on the bounding-box center
first). Volume is only reported for watertight geometry.

## AI Tools (`abstract3d.tools`)

Eight LLM-callable tools returning JSON strings with explicit `success`
markers: `generate_3d_object`, `analyze_3d_object`, `transform_3d_object`,
`compose_3d_scene`, `convert_3d_object`, `repair_3d_object`,
`render_3d_preview`, `list_3d_backends`.

```python
from abstract3d.tools import abstract3d_tools, register_tools

# Directly with AbstractCore generation:
resp = llm.generate("Analyze ./scene.glb", tools=abstract3d_tools())

# Or into a ToolRegistry you own:
from abstractcore.tools.registry import ToolRegistry
registry = ToolRegistry()
register_tools(registry)
```

`SCENE3D_TOOL_CLASSIFICATION` maps each tool to
`{mutating, remote_write_capable, downloads_model_weights}` for approval
layers. See [`integration-abstractcore.md`](integration-abstractcore.md) for
the ruled explicit-import contract and the server endpoint.

## Result Shape

Successful generation returns a dictionary containing:

- `data`: raw bytes when no artifact store is present
- `content_type`
- `mime_type`
- `format`
- `backend_id`
- `model_id`
- `metadata`

Important metadata keys:

- `task`
- `device`
- `appearance_mode`
- `cleanup_mode`
- `vertex_count`
- `face_count`
- `timings_s`
- `memory`
- `bundle_dir`
- `contact_sheet_path`
- `metadata_path`
- `surface_cleanup`
- `postprocess_cleanup`
- `postprocess_warnings`
- `topology`
- `topology_before_cleanup`
- `texture_mode`
- `texture_resolution`
- `texture_completion`
- `texture_artifacts`
- `texture_warnings`
- `uv_present`
- `material_count`
- `preview_renderer`

For textured TripoSR bundles, `texture_artifacts` also records:

- `projection_mode`
- `observed_coverage_ratio`
- `observed_view_stats`
- `reference_view_count`
- `reference_view_paths`
- `texture_completion`
- `symmetry_completion`

Step1X-specific metadata also records:

- `geometry_only`
- `native_text_to_scene3d`
- `composed_text_to_scene3d`
- `geometry_subfolder`
- `label_condition`
- `num_inference_steps`
- `guidance_scale`
- `max_facenum`
- `background_removal_policy`
- `surface_cleanup`
- `postprocess_cleanup`
- `postprocess_warnings`
- `topology`
- `octree_resolution_policy`
- `export_axis_canonicalization`
- `preview_axis_canonicalization`
- `runtime_memory`
- `patchset_version`

## CLI

### `catalog`

```bash
abstract3d catalog --validated-only --json
```

Lists the validated backend plus experimental and research-stage candidates.

### `i23d`

```bash
abstract3d i23d ./object.png --output-dir ./out/object-step1x --backend step1x --device mps
```

Main options:

- `--backend triposr|step1x|hunyuan3d21|trellis2`
- `--format glb|obj|zip`
- `--output-dir`
- `--device`
- `--mc-resolution`
- `--cleanup presentation|none`
- `--texture-mode vertex_color|baked_basecolor`
- `--texture-resolution`
- `--texture-completion none|mirror_symmetry|auto` (`auto` applies mirror completion only when the mesh itself is measurably left-right symmetric; Hunyuan3D defaults to `auto`)
- `--texture-reference-image`
- `--texture-reference-angle`
- `--texture-reference-synthesized true|false|auto` (pairs positionally with `--texture-reference-image`, like `--texture-reference-angle`: `true` marks that reference as synthesized — it completes unobserved surface but may never overwrite photo-observed texels; `false` pins it as a real photo with full paint authority; `auto`/omitted defers to filename inference — references named like the pipeline's own generated outputs (`geometry_view_synthesized_*`, `texture_reference_generated_*`) are treated as synthesized automatically, with a metadata note. Per-reference authority is recorded under `texture_artifacts.reference_authority`.)
- `--texture-reference-remove-background`
- `--num-inference-steps`
- `--guidance-scale`
- `--shape-candidates` (hunyuan3d21: best-of-N shape selection; see below)
- `--quality standard|high|best` (hunyuan3d21: preset for `--shape-candidates` — 1/2/3; the explicit flag overrides the preset)
- `--geometry-conditioning single|multiview|auto|loop` (hunyuan3d21: shape-stage conditioning for single-photo flows; `loop` is the calibrated two-pass bust recipe — see below)
- `--texture-reference-allow-person` (person-specific attestation for any mode that synthesizes views of a person; no gate defends facial identity)
- `--chunk-size`
- `--model`
- `--remove-background`

Hunyuan3D-2.1 notes: the backend requires `ABSTRACT3D_HUNYUAN_ACCEPT_LICENSE=1` (or `scene3d_hunyuan_license_accepted=true`), `--mc-resolution` maps to its octree resolution, and `--texture-mode` accepts `baked_basecolor` (default, shared projection bake) or `none` for geometry-only exports.

Best-of-N shape selection (hunyuan3d21): with `--shape-candidates N` (option `shape_candidates`, config key `scene3d_hunyuan_shape_candidates`, default 1) the shape stage runs N times sequentially with spaced seeds (candidate *i* draws at `seed + 1000*i`), each candidate is postprocessed and ranked, the best ships, and the texture stage keeps the original base seed so reference generation is unchanged. Ranking is measured against the input photo: normalized silhouette IoU over a coarse pose sweep plus concave-detail (convex-hull-minus-mask) IoU at the matching pose, combined with watertightness/single-body; dihedral-RMS smoothness is recorded per candidate as a diagnostic but carries no score weight (a weighted smoothness term was measured to reward melted candidates) — weights are calibrated on the persisted corpus (see `CHANGELOG.md`). Every candidate's seed, metrics, and postprocess record land in metadata under `shape_candidates` (with `selected` flags and top-level `shape_seed`); a discarded draw is never silent. Cost: each extra candidate adds about one shape-stage time (~21–28 min measured on Apple `mps` at octree 512); ranking adds seconds. With `N=1` the pipeline is exactly the historical single-draw path (no ranking renders, unchanged metadata).

Multi-view geometry (same backend, same license gate): pass `--model tencent/Hunyuan3D-2mv` and repeat `--texture-reference-image` / `--texture-reference-angle`. References whose angles snap to the trained `front`/`left`/`back`/`right` slots (within 25°) condition the shape reconstruction as well as the texture bake; the result metadata records `multiview_conditioning` and `geometry_views`.

Loop geometry conditioning (hunyuan3d21; the calibrated one-photo bust recipe): `--geometry-conditioning loop` runs the full two-pass reconstruction loop in one call —

```bash
ABSTRACT3D_HUNYUAN_ACCEPT_LICENSE=1 ABSTRACT3D_IMAGE_PROVIDER=mlx-gen \
ABSTRACT3D_IMAGE_MODEL=AbstractFramework/flux.2-klein-9b-8bit \
ABSTRACT3D_IMAGE_LORA_ADAPTERS="$HOME/Library/Caches/mflux/loras/Flux2-Klein-9B-consistency-V2.safetensors@1.0" \
abstract3d i23d ./photo.png --output-dir ./out/bust \
  --backend hunyuan3d21 --device mps \
  --geometry-conditioning loop --texture-reference-allow-person \
  --num-inference-steps 50 --mc-resolution 512
```

The two LoRA/model lines are the measured winning view-generation recipe
(viewgen bench 2026-07-21 §6: klein-9b + the consistency LoRA at scale 1.0
halves pose error and eliminates the identity/expression fail classes);
`ABSTRACT3D_IMAGE_LORA_ADAPTERS` (config `scene3d_image_lora_adapters`)
accepts `;`-separated `path[@scale]` entries or a JSON list and forwards
verbatim to the local image route on every synthesis call. Requested
adapters are recorded per run under `reference_generation.lora_adapters`
(plus `lora_applied_file_count` per attempt when the backend reports it).

1. **Pass 1** reconstructs a scaffold mesh from the (robustly matted) front photo alone on the Hunyuan3D-2mv checkpoint.
2. The missing canonical views (`back`, `side_left`, `side_right`) are synthesized against that mesh's own clay renders through the **identity i2i route** (photo as the primary edit image, clay as a separate `reference_images` entry — the composite two-panel canvas measurably flips person identity on local editors) and gated: the standard silhouette/material/specular acceptance, row consistency against the clay guide, and the **pose-honesty ruler** (head-band-IoU sweep over candidate azimuths) at TWO points — inside the draw ladder, a candidate measured decisively >15° off its declared azimuth is rejected and **redrawn** (viewgen bench 2026-07-21 §6.3; recorded per attempt, `pose_unmeasured` when no clay ruler exists), and after acceptance a view decisively >20° off may not condition (it re-declares to its measured azimuth for the texture bake — the striated-mouth incident traced to ~50° three-quarter views sold as 90° profiles).
3. Every conditioning view (front included) is cut to one anatomical **window** (`head_top → shoulder + 0.22·span`), so the 2mv bbox recentring puts the same anatomy at the same normalized rows; **pass 2** reconstructs from the windowed set (capped at 3 tags).
4. The texture bake consumes the **full-span** views only, each angle exactly once, at its **measured** azimuth, replayed through the full texture acceptance machinery with synthesized-reference protection (windowed variants never bake — they starve texels below the window).
5. Self-verification renders oblique raking-light closeups plus a front clay render and runs a duplication-autocorrelation flag; a firing flag ships as `quality_verdict=degraded` (the CLI exits 3) with the review-flag caveat in the reason. Renders land under `loop_verification/` in the bundle; windowed conditioning pixels persist as `geometry_view_windowed_*.png`.

Loop requirements (all refuse loudly, never degrade silently): an explicitly configured **local** image provider (`ABSTRACT3D_IMAGE_PROVIDER` / `scene3d_image_provider` — the loop never routes the photo through a remote default), the multi-view checkpoint (no explicit flagship `--model`), no explicit `--texture-reference-image` (the loop owns view synthesis), and `--texture-reference-allow-person` for person subjects. Expect roughly 2× generation time (two DiT passes; `timings_s.pass1_inference` and `timings_s.inference` record both). The knobs above (`50` steps / `512` octree) reproduce the validated bust recipe; without them the conservative 2mv family defaults (30/384) apply.

Runtime expectations at `512` octree on Apple Silicon (`mps`): a full loop run takes on the order of 2¼ hours of compute (measured 2026-07-22 on one M-class machine: pass-1 shape ~34 min, view synthesis ~28 min, pass-2 shape ~41 min, mesh postprocess ~3 min, texture ~30 min; see [Benchmarks](benchmarks.md) for the stage table). Two operational notes for long runs:

- The volume decode logs its level schedule and ~5% chunk progress at INFO (`volume decode [coarse 128^3]: chunk N/M`), so a working decode is distinguishable from a hang in the run log. The diffusion phases print no progress by design.
- `timings_s` values are awake-process time (`time.perf_counter`), which on macOS excludes system sleep. An unattended machine that idle-sleeps mid-run pauses the generation and inflates wall-clock time without inflating `timings_s`. Wrap long unattended runs in `caffeinate -dims <command>` to keep the machine awake.

### `t23d`

```bash
abstract3d t23d "a carved wooden owl figurine" \
  --output-dir ./out/owl \
  --backend step1x \
  --device mps
```

Additional options:

- `--backend triposr|step1x|hunyuan3d21|trellis2`
- `--cleanup presentation|none`
- `--texture-mode vertex_color|baked_basecolor`
- `--texture-resolution`
- `--texture-completion none|mirror_symmetry|auto` (`auto` applies mirror completion only when the mesh itself is measurably left-right symmetric; Hunyuan3D defaults to `auto`)
- `--image-provider`
- `--image-model`
- `--image-width`
- `--image-height`
- `--image-seed`
- `--guidance-scale`

If `--image-provider` / `--image-model` are omitted, composed `t23d` resolves them from `scene3d_image_provider` / `scene3d_image_model`, `ABSTRACT3D_IMAGE_PROVIDER` / `ABSTRACT3D_IMAGE_MODEL`, or the configured `abstractvision` default.

### `validate`

```bash
abstract3d validate --output-dir ./out/validation --backend step1x --device mps
```

Runs a compact proof suite and emits a summary contact sheet plus `stats.json`.

The validator also accepts `--texture-mode`, `--texture-resolution`, and `--texture-completion` so TripoSR texture comparisons can be reproduced without custom scripts.

## Reproducible Validation Script

The CLI `validate` command is a light wrapper.

For the checked benchmark workflow, use:

```bash
python scripts/validate_local.py --backend triposr --device mps --mc-resolution 256
python scripts/validate_local.py --backend step1x --device mps --mc-resolution 128
```

Without an explicit `--output-dir`, the Step1X script defaults to `artifacts/validation/local-step1x-label/` when you stay on the label geometry checkpoint family. The checked docs-ready Apple-local reference lane is published separately under `docs/assets/validation/local-step1x-dynamic/`.

That script also writes:

- `summary.json`
- `summary.md`
- per-case `result.json` files for isolated Step1X Apple-local runs

Docs-ready assets are only published when you pass `--publish-doc-assets <dir>`.

For the recommended command patterns and evaluation criteria, see [Methodology](methodology.md).

## Host Integration

When you call AbstractCore through `llm.scene3d.*` or `generate(..., output={"modality": "scene3d"})`, use the `provider` field to select the backend explicitly. Example values are `triposr`, `step1x`, and `trellis2`.

For `abstractcore` examples and residency routing, see [AbstractCore integration](integration-abstractcore.md).
