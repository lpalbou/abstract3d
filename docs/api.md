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
- `--texture-reference-remove-background`
- `--num-inference-steps`
- `--guidance-scale`
- `--shape-candidates` (hunyuan3d21: best-of-N shape selection; see below)
- `--quality standard|high|best` (hunyuan3d21: preset for `--shape-candidates` — 1/2/3; the explicit flag overrides the preset)
- `--chunk-size`
- `--model`
- `--remove-background`

Hunyuan3D-2.1 notes: the backend requires `ABSTRACT3D_HUNYUAN_ACCEPT_LICENSE=1` (or `scene3d_hunyuan_license_accepted=true`), `--mc-resolution` maps to its octree resolution, and `--texture-mode` accepts `baked_basecolor` (default, shared projection bake) or `none` for geometry-only exports.

Best-of-N shape selection (hunyuan3d21): with `--shape-candidates N` (option `shape_candidates`, config key `scene3d_hunyuan_shape_candidates`, default 1) the shape stage runs N times sequentially with spaced seeds (candidate *i* draws at `seed + 1000*i`), each candidate is postprocessed and ranked, the best ships, and the texture stage keeps the original base seed so reference generation is unchanged. Ranking is measured against the input photo: normalized silhouette IoU over a coarse pose sweep plus concave-detail (convex-hull-minus-mask) IoU at the matching pose, combined with watertightness/single-body; dihedral-RMS smoothness is recorded per candidate as a diagnostic but carries no score weight (a weighted smoothness term was measured to reward melted candidates) — weights are calibrated on the persisted corpus (see `CHANGELOG.md`). Every candidate's seed, metrics, and postprocess record land in metadata under `shape_candidates` (with `selected` flags and top-level `shape_seed`); a discarded draw is never silent. Cost: each extra candidate adds about one shape-stage time (~21–28 min measured on Apple `mps` at octree 512); ranking adds seconds. With `N=1` the pipeline is exactly the historical single-draw path (no ranking renders, unchanged metadata).

Multi-view geometry (same backend, same license gate): pass `--model tencent/Hunyuan3D-2mv` and repeat `--texture-reference-image` / `--texture-reference-angle`. References whose angles snap to the trained `front`/`left`/`back`/`right` slots (within 25°) condition the shape reconstruction as well as the texture bake; the result metadata records `multiview_conditioning` and `geometry_views`.

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
