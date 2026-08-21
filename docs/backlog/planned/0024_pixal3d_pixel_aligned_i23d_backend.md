# Planned: Pixal3D pixel-aligned i23d backend with native PBR texture stage

## Metadata

- Created: 2026-08-21
- Status: Planned
- Completed: N/A

## ADR status

- Governing ADRs: [ADR 0001](../../adr/0001_scene3d_local_first_glb_contract.md), [ADR 0003](../../adr/0003_trellis2_uses_official_upstream_assets_only.md), [ADR 0009](../../adr/0009_export_material_truth_and_fill_detail_synthesis.md)
- ADR impact: Needs new ADR. ADR 0003 binds the *TRELLIS.2* backend to official Microsoft
  assets. Pixal3D is a different first-party upstream (TencentARC) that happens to fork the
  TRELLIS.2 code, so it must not be admitted by widening the TRELLIS.2 model selector. It needs
  its own official-only ADR mirroring 0003. A second ADR question is raised by the native PBR
  stage: ADR 0009 governs exports whose materials are derived from observed source pixels, and
  Pixal3D emits *model-generated* metallic/roughness maps, which is a different truth class.

## Context

`abstract3d` currently has four local backends (`triposr`, `step1x`, `hunyuan3d21`, `trellis2`).
TripoSR is the validated default; the others are experimental. Every textured export in the
repository today comes from the shared projection bake in `abstract3d.texturing` — the repo has
no model that generates texture natively, and no model that emits PBR channels at all.

Pixal3D (SIGGRAPH 2026, Tsinghua BNRist + Tencent ARC Lab, MIT license,
[GitHub](https://github.com/TencentARC/Pixal3D),
[weights](https://huggingface.co/TencentARC/Pixal3D), [arXiv:2605.10922](https://arxiv.org/abs/2605.10922))
generates geometry *and* PBR textures from a single image. Instead of injecting image features
loosely through cross-attention, it back-projects pixel features into the 3D volume, establishing
direct pixel-to-3D correspondences. Its `main` branch is built on the TRELLIS.2 backbone — the
exact backbone this repository already vendors, pins, and patches for Apple Silicon.

There is no reference to Pixal3D anywhere in this repository (`grep -ri pixal .` returns nothing).
It is not in `model_catalog.py`, not in `backends/`, not in docs, not in the backlog.

## Current code reality

Inspected on 2026-08-21 against `src/abstract3d/backends/trellis2_runtime.py` (1551 lines), the
vendored snapshot at `~/.cache/abstract3d/vendor/trellis2/75fbf018.../trellis2`, and a fresh clone
of `TencentARC/Pixal3D@main`.

**What already exists and is directly reusable:**

- `trellis2_runtime.py` already pins and vendors upstream TRELLIS.2 at commit
  `75fbf0183001ed9876c8dbb35de6b68552ee08bd` (`_TRELLIS2_REPO_URL`, `_TRELLIS2_COMMIT`), with a
  source manifest, cache layout, and `_sys_path` import isolation.
- The DINOv3 companion gate is already built: `_DINO_MODEL_ID =
  "facebook/dinov3-vitl16-pretrain-lvd1689m"`, `_configured_dino_source`, `_download_dino_model`,
  `_validate_local_dino_snapshot`, and a license-explanatory `DependencyUnavailableError`.
  Pixal3D's `pipeline.json` declares the *same* DINOv3 conditioner and the *same* `briaai/RMBG-2.0`
  rembg model.
- Device selection (`_select_device`), MPS cache release, and MPS memory telemetry already exist.
- Apple-safe shims already exist in `_install_trellis2_shims` / `_patch_sparse_attention`:
  `easydict`, `trellis2.modules.sparse.conv_none`, a local `o_voxel.convert.flexible_dual_grid_to_mesh`,
  and `SPARSE_CONV_BACKEND=none` + `ATTN=sdpa` forcing.
- The checkpoint naming and `pipeline.json` schema are identical in shape. Pixal3D's
  `pipeline.json` even declares `"name": "Trellis2ImageTo3DPipeline"`.

**Measured delta between `pixal3d/` and the vendored `trellis2/` package** (`diff -rq`): four new
files and a small set of modified ones. Only these matter for inference:

- new `pixal3d/modules/sparse/attention/proj_attention.py` (99 lines) and
  `pixal3d/modules/attention/proj_attention.py` (101 lines) — pure PyTorch, no custom kernels.
- new `pixal3d/pipelines/pixal3d_image_to_3d.py` (783 lines) — the cascade driver.
- new `pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py` (1530 lines) — hosts
  `DinoV3ProjFeatureExtractor`, used at inference time.
- modified: `models/structured_latent_flow.py` (68 lines), `modules/sparse/attention/full_attn.py`
  (37), `pipelines/trellis2_image_to_3d.py` (25), `modules/sparse/attention/windowed_attn.py` (15),
  `modules/sparse/config.py` (4). The rest of the diff is training/dataset code.

**Pixal3D adds a native `sdpa` sparse-attention backend upstream** (`full_attn.py` gains an
`elif config.ATTN == 'sdpa'` branch; `config.py` accepts `'sdpa'` and `'flash_attn_4'`). Our
`_sparse_sdpa` monkey-patch exists precisely because upstream TRELLIS.2 lacked this. For a Pixal3D
backend the patch should be dropped in favor of the upstream path — fewer shims, not more.

**What is genuinely new and unproven here:**

- **`flex_gemm` on the texture path.** `pixal3d/modules/sparse/config.py` defaults `CONV =
  'flex_gemm'` (we already override to `none`), but `pixal3d/pipelines/trellis2_texturing.py`
  imports `flex_gemm` at module top level and calls `flex_gemm.ops.grid_sample.grid_sample_3d`,
  as do `representations/mesh/base.py`, `renderers/mesh_renderer.py`, and
  `renderers/pbr_mesh_renderer.py`. FlexGEMM is CUDA-only. Our current TRELLIS.2 path never
  exercises a texture decoder, so this surface has never been shimmed. A CPU/MPS `grid_sample_3d`
  equivalent is the single largest unknown.
- **MoGe-2 as a new companion model.** `inference.py` loads `Ruicheng/moge-2-vitl`
  (`git+https://github.com/microsoft/MoGe.git`, MIT) to estimate camera FOV from a wild image
  before back-projection. This is a new gate/provenance/footprint surface not covered by ADR 0003's
  companion list.
- **Footprint.** 24 GB of weights vs. 16.2 GB for `microsoft/TRELLIS.2-4B`: `ss_flow_img_dit_1_3B_64`
  (5.36 GB), `slat_flow_img2shape_dit_1_3B_512` (5.55 GB), `slat_flow_img2shape_dit_1_3B_1024`
  (5.55 GB), `slat_flow_imgshape2tex_dit_1_3B_1024` (5.55 GB), plus `shape_dec_next_dc` and
  `tex_dec_next_dc` (948 MB each) and `ss_dec_conv3d_16l8` (148 MB). Default cascade is `1536`;
  `--low_vram` drops to `1024` with on-demand module loading.
- **`cuda` is hardcoded ~155 times** in the `pixal3d` package plus throughout `inference.py`
  (`device="cuda"` defaults, `PYTORCH_CUDA_ALLOC_CONF`, `FLEX_GEMM_AUTOTUNE_CACHE_PATH`).
- **`natten==0.21.0`** is an install step in the README (CUDA-arch build required) but is *not
  imported anywhere* in the cloned tree — it appears to be inherited from the TRELLIS.2 base
  environment or a demo-only pin. This must be verified by preflight, not assumed either way.
- **DINOv3 mirror conflict.** Upstream `inference.py` hardcodes the contributor mirror
  `camenduru/dinov3-vitl16-pretrain-lvd1689m`, which our companion-model selector explicitly
  rejects. `pipeline.json` names the official `facebook/...` repo. We must use the official repo
  and confirm the weights are equivalent.
- The `paper` branch is built on Direct3D-S2, not TRELLIS.2. Only `main` is in scope.
- The `trellis2` backend this work builds on is itself still `experimental` in `model_catalog.py`
  and has no checked Apple-local proof bundle.

## Problem

The repository has no native texture-generating backend and no PBR-emitting backend. The highest-
leverage candidate for both is unintegrated, and it is the *cheapest* candidate to integrate
because it forks a backbone we already vendor, pin, and patch — yet nobody has established whether
its texture decode path can run without CUDA.

## What we want to do

Add Pixal3D as a distinct, official-only, MIT-licensed image-to-3D backend (`backend_kind =
"pixal3d"`), reusing the existing TRELLIS.2 vendoring, DINOv3 gate, and Apple shim infrastructure,
and gate the work behind a cheap feasibility spike that is allowed to fail closed.

## Why

- **License posture is the best in the catalog.** MIT weights and MIT code, with no
  territory-restriction and no acknowledgment gate — unlike `tencent/Hunyuan3D-2.1`, which is
  currently our highest-quality geometry backend but excludes EU/UK/South Korea operators. The only
  gate is DINOv3, which the repo already handles.
- **It is the first native PBR path.** Every textured export today is a projection bake of
  observed source pixels, with unseen regions filled by synthesis (ADR 0009). A generative PBR
  stage is a categorically different capability, and it directly informs the open
  `scene3d-texture` track.
- **Integration cost is unusually low for the capability gained.** The inference-relevant delta
  is ~1000 new lines of pure-PyTorch plus ~150 modified lines against a snapshot we already
  vendor. Pixal3D also *removes* one of our shims by shipping upstream SDPA sparse attention.
- **It targets a failure class we have measured.** Pixel-aligned back-projection is aimed at
  input-fidelity loss, which is the same class of defect the multi-view and reference-view work
  (items 0014, 0016, 0018) keeps circling.

## Requirements

- Keep the runtime local-only and official-only: source from a pinned `TencentARC/Pixal3D` commit,
  weights from `TencentARC/Pixal3D`, DINOv3 from `facebook/dinov3-vitl16-pretrain-lvd1689m`,
  MoGe-2 from `Ruicheng/moge-2-vitl`. Reject mirrors, quantizations, and repacks — including the
  `camenduru/...` DINOv3 mirror hardcoded upstream.
- Register Pixal3D as its own `backend_kind`, not as a new model id accepted by the TRELLIS.2
  selector. ADR 0003's official-only rule for TRELLIS.2 stays intact and unwidened.
- Do not displace TripoSR as the validated default. Pixal3D enters as `experimental`.
- Preserve the `glb`-first artifact contract (ADR 0001) and export material truth (ADR 0009).
  A PBR export must carry real metallic/roughness channels, not trimesh `SimpleMaterial` defaults,
  and must pass `scripts/check_export_materials.py`.
- Generated PBR channels must be recorded in generation metadata as *model-synthesized*, distinct
  from observed-pixel bakes, so proof surfaces cannot conflate them.
- Fail with an actionable error, never a silent fallback to another backend, when a companion
  model, dependency, or accelerator is unavailable.

## Suggested implementation

Phase the work so the expensive part is never started on an unproven path.

**Phase 1 — feasibility spike (do this first; it may close the item).**

- Vendor `TencentARC/Pixal3D@main` at a pinned commit alongside the existing TRELLIS.2 snapshot,
  reusing `_source_manifest` conventions.
- Write a preflight that answers, on the target Apple-`mps` host, with evidence:
  1. Is `natten` actually imported on the inference path? (Static evidence says no; confirm.)
  2. Can the shape cascade run with `ATTN_BACKEND=sdpa` + `SPARSE_CONV_BACKEND=none` using
     Pixal3D's own upstream SDPA branch, with our `_sparse_sdpa` patch *disabled*?
  3. Does the texture stage's `flex_gemm.ops.grid_sample.grid_sample_3d` have a viable
     `torch.nn.functional.grid_sample`-based replacement, and is it numerically close enough?
  4. What is peak resident memory at `--resolution 1024` with on-demand loading?
- If (3) has no answer, the geometry-only subset is still worth landing — say so explicitly rather
  than abandoning the item.

**Phase 2 — backend.**

- Add `src/abstract3d/backends/pixal3d_runtime.py` modeled on `trellis2_runtime.py`. Reuse the
  DINOv3 gate helpers rather than copying them; factor the shared pieces out if the duplication
  crosses roughly 200 lines.
- Add a MoGe-2 companion loader with the same provenance/gating discipline, and release its memory
  before the cascade runs (upstream already does this: `moge_model.cpu()` after camera estimation).
- Replace hardcoded `cuda` with the existing `_select_device` result; force `float32` on `mps`
  where the Step1X and TRELLIS.2 lanes already found bf16 unreliable.
- Expose `resolution` (1024/1536) and low-VRAM/on-demand loading as backend config, defaulting
  conservatively on Apple.
- Add the catalog entry to `model_catalog.py`: `model_id="TencentARC/Pixal3D"`,
  `provider_id="pixal3d"`, `backend_kind="pixal3d"`, `tasks=("image_to_scene3d",)`,
  `license="MIT"`, `status="experimental"`, `footprint_gb=24.0`. Composed `t23d` is out of scope
  until `i23d` is proven.

**Phase 3 — PBR export and proof.**

- Route the generated metallic/roughness/basecolor into the existing PBR export path and the
  material gates.
- Run the same benchmark cases used by the Hunyuan3D and Step1X lanes so the comparison is
  apples-to-apples, and preserve bundles under `artifacts/validation/pixal3d/`.

If Phase 2 and Phase 3 both survive Phase 1, split them into a `planned/pixal3d/` topic track
rather than growing this file.

## Scope

- Pixal3D `main`-branch (TRELLIS.2-backbone) image-to-3D as a new experimental backend.
- The MoGe-2 camera-estimation companion, gated and provenance-tracked.
- A non-CUDA `grid_sample_3d` path if and only if the spike shows one is viable.
- A new ADR fixing the Pixal3D official-asset rule.
- Catalog, docs (`docs/models.md`, `docs/troubleshooting.md`), and `ACKNOWLEDGEMENTS.md` entries.

## Non-goals

- Widening ADR 0003 or the TRELLIS.2 model selector to accept Pixal3D checkpoints.
- The `paper` branch / Direct3D-S2 implementation.
- Training, fine-tuning, or the `data_toolkit`.
- Composed `t23d` through Pixal3D, until `i23d` is proven locally.
- Promoting Pixal3D over TripoSR as the validated default.
- Building CUDA kernels, or requiring an NVIDIA host for the supported path.
- Replacing the `abstract3d.texturing` projection bake. The generative PBR stage is an additional
  texture mode, not a substitute for observed-pixel texturing.

## Dependencies and related tasks

- `src/abstract3d/backends/trellis2_runtime.py` — read `_install_trellis2_shims`,
  `_patch_sparse_attention`, `_variant_for_model`, `_download_dino_model` before starting.
- `src/abstract3d/model_catalog.py`
- `src/abstract3d/texturing.py`, `src/abstract3d/material_gates.py`,
  `scripts/check_export_materials.py`
- [ADR 0003](../../adr/0003_trellis2_uses_official_upstream_assets_only.md) — the template for the
  new Pixal3D asset ADR.
- [ADR 0009](../../adr/0009_export_material_truth_and_fill_detail_synthesis.md) — export material
  truth applies to generated PBR too.
- [0012 Textured validation suite and promotion gate](scene3d-texture/0012_textured_validation_suite_and_promotion_gate.md)
  — the promotion methodology this backend should be judged by.
- [0009 Scene3D textured asset contract and policy](scene3d-texture/0009_scene3d_textured_asset_contract_and_policy.md)
  — a native PBR mode is a new texture mode under that contract.
- Upstream: <https://github.com/TencentARC/Pixal3D>, <https://huggingface.co/TencentARC/Pixal3D>,
  <https://github.com/microsoft/MoGe>, <https://github.com/JeffreyXiang/FlexGEMM>

## Expected outcomes

- A documented, evidence-backed answer to "can Pixal3D run locally without CUDA?", whichever way
  it lands.
- If yes: a `pixal3d` backend that produces a `glb` with model-generated PBR channels from a single
  image on the checked Apple-local host, registered as `experimental`, with an actionable failure
  message for every gate it can hit.
- If no: this item is deprecated with the spike evidence preserved, and the blocking surface
  (`flex_gemm.grid_sample_3d`, memory ceiling, or otherwise) named precisely enough that a future
  CUDA host or upstream change can reopen it.
- The TRELLIS.2 backend and ADR 0003 are unchanged by this work.

## Validation

- `pytest -q tests`
- Unit tests for official-asset selection and non-official rejection (including rejection of the
  `camenduru/...` DINOv3 mirror), mirroring the existing TRELLIS.2 selector tests.
- Unit tests for actionable MoGe-2 and DINOv3 gating errors.
- Numerical parity check for any `grid_sample_3d` replacement against a CUDA reference or a
  synthetic ground-truth grid.
- `scripts/check_export_materials.py` and `scripts/texture_qa.py` material gates on every exported
  Pixal3D asset.
- Case-isolated, memory-guarded local runs in the style of `scripts/validate_local.py`, with
  bundles preserved under `artifacts/validation/pixal3d/` — failures preserved, not discarded.
- Side-by-side contact sheets against TripoSR and Hunyuan3D-2.1 on the shared benchmark cases.
- No proof asset is published to `docs/assets/validation/` without visual review.

## Progress checklist

- [ ] Phase 1: pin and vendor `TencentARC/Pixal3D@main`; record the commit
- [ ] Phase 1: preflight — confirm whether `natten` is on the inference path
- [ ] Phase 1: run the shape cascade with upstream `ATTN_BACKEND=sdpa`, our patch disabled
- [ ] Phase 1: determine whether `flex_gemm.grid_sample_3d` has a viable non-CUDA replacement
- [ ] Phase 1: measure peak memory at resolution 1024 with on-demand loading
- [ ] Phase 1: write up the go/no-go with evidence; deprecate this item here if no-go
- [ ] Phase 2: new ADR for the Pixal3D official-asset rule
- [ ] Phase 2: `pixal3d_runtime.py` with device selection, DINOv3 gate reuse, MoGe-2 companion
- [ ] Phase 2: `model_catalog.py` entry + selector tests
- [ ] Phase 3: PBR channels through the export path and material gates
- [ ] Phase 3: benchmark bundles under `artifacts/validation/pixal3d/`
- [ ] Docs: `docs/models.md`, `docs/troubleshooting.md`, `ACKNOWLEDGEMENTS.md`

## Guidance for the implementing agent

Re-read `trellis2_runtime.py` before writing a line — most of what this backend needs already
exists there, and the correct instinct is to reuse or extract it, not to fork a second 1500-line
runtime. Verify the upstream delta yourself against the pinned vendored TRELLIS.2 snapshot; the
measurements in `Current code reality` were taken on 2026-08-21 against `main` and will drift.

Do not widen the TRELLIS.2 selector to admit Pixal3D checkpoints, however convenient it looks —
`pipeline.json` claiming `"name": "Trellis2ImageTo3DPipeline"` is a naming artifact of the fork,
not evidence that this is a TRELLIS.2 asset.

Phase 1 is allowed to kill this item. A well-evidenced no-go with the blocking surface named is a
successful outcome, not a failure; record it and move the item to `deprecated/`.
