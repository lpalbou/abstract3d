# Improving the single-photo -> 3D bust process: geometry + texture (adversarial research distillation, 2026-07-21)

Scope: evidence-ranked adoption plan for replacing/augmenting the current pipeline
(Hunyuan3D-2mv conditioned on photo + i2i-synthesized views -> our projection bake,
`texturing.py`) under our constraints: local-first, Apple Silicon MPS, <=20 GB unified
memory for the product profile (this dev box has 128 GB — spikes can measure without OOM
risk), no remote APIs for the operator's face. Every claim below was checked against the
actual repo/model card; "CUDA-only" is said plainly where true. This document recommends;
it changes no pipeline code. Companion docs: `multiview_consistency_2026.md` (consistency
mechanisms), `viewgen_audit.md` (why our synthesized views are the weak input).

## Axis 1 — Native texture pipelines (the operator's bet, verified)

**Bet under review: "hunyuan3d-paint on MPS replacing our projection bake." Verdict:
FEASIBLE — with two corrections.** (1) The OFFICIAL repo is CUDA-only as shipped, but the
blockers are build-system, not algorithmic, and community ports already clear them.
(2) hy3dpaint does not replace projection baking with something else — its own final stage
IS weighted back-projection + UV inpaint (`bake_from_multiview`, `bake_exp=4` facing
weights — the same architecture as our bake). What it replaces is our WEAK INPUT: instead
of freestanding i2i views that must be registered post-hoc, it renders normal+position
maps from the mesh's own cameras and generates the views CONDITIONED on those maps with
cross-view attention — the views are pixel-registered to the mesh by construction, so the
misregistration class that paints lips on noses cannot occur in that lane
[Hunyuan3D-2.1 repo, `hy3dpaint/textureGenPipeline.py`, verified in our vendored snapshot
`~/.cache/abstract3d/vendor/hunyuan3d21/82920d64...`].

Facts verified against the pinned source + official README [Tencent-Hunyuan/Hunyuan3D-2.1]:

- **Components**: hunyuan3d-paintpbr-v2-1 (2B multiview UNet, SD2.1-class, custom diffusers
  pipeline `hunyuanpaintpbr`, ~3.7 GB) + facebook/dinov2-giant (~4.5 GB, reference-image
  encoder) + RealESRGAN_x4plus (~65 MB, optional SR) + two compiled extensions.
- **The two extensions**: `custom_rasterizer` — upstream `setup.py` is an unconditional
  `CUDAExtension` (fails on macOS with `CUDA_HOME not set`), BUT `rasterizer.cpp` carries
  full CPU implementations (`rasterize_image_cpu`, `barycentricFromImgcoordCPU`) and
  device-dispatches at runtime; only the build script and a CUDA-including header block
  macOS. `DifferentiableRenderer/mesh_inpaint_processor.cpp` is pure C++ (pybind11); the
  reported macOS failure is a clang `-bundle`/`-dynamiclib` flag conflict
  [Hunyuan3D-2 issue #320]. Both are packaging fixes, not ports.
- **Working macOS ports (all verified to exist; none verified by us end-to-end yet)**:
  - `agenticvibes/ComfyUI-Hunyuan3d-Paint` (pin `3d8e931`): standalone-importable
    `hy3dpaint/` fork of 2.1 paint with a CPU/CUDA-branching `setup.py` (CppExtension +
    `-DCUDA_AVAILABLE` guard), MPS backend with chunked attention, CPU rasterizer fallback,
    and an optional MLX UNet backend (~3-5x faster; converted weights
    `AgenticVibes/hunyuan3d-2.1-mlx`, ~4 GB). Port code MIT; Tencent code keeps its license.
  - `ZimengXiong/Hunyuan3D-MLX`: full shape+paint MLX/Swift ports with measured numbers —
    paint PBR (512 render, 15 steps, 4096 atlas, +SR): 344 s, **~39 GB peak**; shape small:
    20.9 s, ~5.6 GB. The 39 GB is over our 20 GB budget AS CONFIGURED (6 views, 4096 atlas,
    SR on); the knobs (`max_num_view` 6->4, `texture_size` 4096->2048, SR off) are the
    obvious levers, unmeasured — the integration must measure.
  - `Rc121122/ComfyUI-Hunyuan3DWrapper`: Hunyuan3D-**2.0** paint full workflow on MPS,
    reported working on M1 Pro 32 GB and Mac Mini M4 24 GB [Hunyuan3D-2 issue #151].
- **Official VRAM claim**: 10 GB shape / 21 GB texture / 29 GB total (CUDA, defaults:
  6 views, 2048 render, 4096 atlas, SR) [official README]. The texture figure exceeds our
  20 GB budget at defaults; see knobs above.
- **License**: the paint weights live in the SAME `tencent/Hunyuan3D-2.1` repo under the
  SAME Tencent Hunyuan 3D 2.1 Community License our backend already gates
  (`scene3d_hunyuan_license_accepted`). No new gate needed. Standing facts of that license,
  stated plainly: territory excludes EU/UK/South Korea; >100M MAU needs a separate license.
- **Identity risk (honest)**: the reference photo enters as a 512x512 "style" image via
  reference-attention + DINO features. The model is trained on Objaverse-class assets;
  face identity preservation is UNPROVEN. Our measured lesson (viewgen audit L1-L11): local
  editors flip person identity readily. Mitigation that composes with what we have: keep
  the photo-anchored front projection + protect-observed-texels (backlog 0017 machinery)
  and give hy3dpaint output completion-only authority — it paints what the photo never saw.

Catalog of the other named candidates (what actually runs locally):

| Pipeline | What it is | Local/MPS verdict | License |
|---|---|---|---|
| TEXTure / Text2Tex | iterative depth-aware SD inpainting on UV | **CUDA-only in practice**: hard deps kaolin (CUDA; CPU install "only a fraction of operations") resp. pytorch3d+xformers; torch<=2.5 pins [TEXTurePaper; daveredrum/Text2Tex; kaolin docs] | MIT / non-commercial (Text2Tex) |
| SyncMVD | synchronized per-step latent fusion in UV space | Linux + NVIDIA + PyTorch3D stated requirement; SD1.5+ControlNet base. Architecture ideas (winner-take-most consistency weighting) already absorbed into our bake plan [LIU-Yuxin/SyncMVD] | code MIT |
| MV-Adapter ig2mv | plug-in adapter making SDXL/SD2.1 emit 6 geometry-conditioned views | adapter+diffusers = MPS-plausible (no custom kernels in the UNet path); **their texture BAKE requires CV-CUDA — CUDA-only**. Use their views with OUR bake instead. SDXL ~16 GB, SD2.1 <10 GB (CUDA-reported) [huanngzh/MV-Adapter README] | **Apache-2.0 confirmed for repo AND HF weights** (closes the "verify weights" question in multiview_consistency_2026.md) |
| TRELLIS.2 texture stage | shape-conditioned PBR texture gen (`example_texturing.py`) | **CUDA-only**: o-voxel/flex/nvdiffrast/nvdiffrec/flash-attn compiled deps; Linux + 24 GB NVIDIA stated. Our `trellis2_runtime.py` ships geometry-only with sparse shims (`shape_only: True`) for exactly this reason | MIT (model+code) |
| Hunyuan3D-2.0 paint | RGB (non-PBR) 1.3B paint + delight model | proven on MPS via Rc121122 wrapper (24 GB Macs); superseded by 2.1 PBR quality-wise | Tencent community |
| Meshy / Rodin / Hunyuan 3D Engine 3.x | hosted | **excluded** (remote; operator-face policy) | — |

## Axis 2 — Geometry alternatives for human busts from one photo

- **FLAME/DECA/EMOCA (parametric heads + displacement)**: face-region only. DECA's own
  limitations section: occluders (glasses, hair, facial hair) are "explained by shape
  deformations" — glasses are not modeled, hair is not reconstructed, and the template ends
  at the neck (no shoulders — not a bust) [DECA SIGGRAPH'21 §limitations; EMOCA CVPR'22].
  Useful as a face-geometry PRIOR or landmark oracle, not as our bust generator. Our
  subject wears glasses — disqualifying as primary.
- **PSHuman**: cross-scale multiview diffusion + SMPL-X carving; full-body focus, requires
  >40 GB VRAM at its only released resolution (768), repo dormant since 2024-12, promised
  512 model never shipped [pengHTYX/PSHuman FAQ]. Out of budget. MIT.
- **SIFU / ECON**: clothed full-body reconstruction; require SMPL(-X) fits + CUDA rendering
  stacks (pytorch3d/kaolin class); designed for full-body photos, not head-and-shoulders
  crops. No MPS story found. Not bust-first.
- **Head-specialized generative (PanoHead/ID-Sculpt/PercHead/MVCHead class)**: strongest
  identity machinery on paper (ArcFace losses, ID-aware guidance) [ID-Sculpt AAAI'25;
  PercHead arXiv:2511.02777; MVCHead CVPR'26] but all are EG3D/3DGS-lane with CUDA custom
  rasterizers, mostly bald-head templates, and glasses remain a documented weak class.
  Watch, don't adopt.
- **TripoSG**: 1.5B rectified-flow shape DiT, MIT, "CUDA-enabled GPU >=8GB" stated; no
  official MPS support; geometry only (no texture stage) [VAST-AI-Research/TripoSG]. A
  candidate for the same MPS-compat treatment as our step1x backend, but it is single-view
  only — it would not consume our registered multiview evidence, and our 2mv lane's
  measured value is exactly that conditioning.
- **Step1X-3D / TRELLIS.2 (our own backends)**: both are live geometry backends
  (`step1x_runtime.py` with compat patchset v4 + MPS memory caps; `trellis2_runtime.py`
  shape-only with sparse shims). Both are generalist object models; neither has
  human-specific training claims. TRELLIS.2's 1536^3 resolution is attractive but its
  4B DiT + 24 GB CUDA floor puts full-resolution runs out of the 20 GB profile.
- **Hunyuan3D-2.5 / 3.x**: NOT open-weight. 2.1 is the last open release; 2.5 (LATTICE,
  10B) / 3.0 / 3.1 are hosted-only via Tencent cloud [official GitHub news; meshy.ai
  comparison page]. Third-party leaderboards listing 2.5 as "Open" are wrong.
- **MV-diffusion-first (Era3D/Wonder3D/Zero123++ -> reconstruction) vs direct-DiT**:
  unchanged from `multiview_consistency_2026.md` §3e: Era3D is AGPL + CUDA-pinned deps;
  Wonder3D license ambiguity; Zero123++ weights CC-BY-NC. None are person-specialized;
  identity through a 6-view SD2-class bottleneck at 256-512 px is worse than our
  photo-anchored route. The field's own human results route through person-trained models
  (PSHuman) or identity losses (ID-Sculpt) — generic MV models are the wrong identity tool.

## Axis 3 — The flagship-2.1-on-MPS shredding bug class

Our failure: all 3 flagship runs shredded (e9: euler -17225, non-watertight, 512/50, mps
fp16, seed 2025) while 2mv at the same dtype/device is healthy. Field evidence gathered:

1. **The model is NOT inherently MPS-broken**: `VladimirTalyzin/hunyuan3d-2.1-mac` runs the
   SAME flagship DiT on MPS fp16 (M4 Pro, octree 256, 50 steps, 2-5 min/mesh) with only
   device-string/ckpt-loader/autocast-no-op patches — none of which change numerics we
   don't already handle. So our shredding is environment- or config-specific.
2. **Torch 2.10.0 (our venv) is inside a documented MPS SDPA silent-garbage window**:
   non-contiguous q/k/v dispatched to the post-2.8 "fast" SDPA path produce garbage
   (SDXL-on-MPS black/noise reports on torch 2.10/2.11; works on 2.9)
   [pytorch#163597]; a separate 2^32-element score-matrix overflow corrupts outputs
   silently [pytorch#179352, fix #179592]; a 2-pass kernel OOB was fixed only in 2.11
   [pytorch#174861]. Both Hunyuan DiTs call SDPA on `rearrange`/`transpose`-produced
   (non-contiguous) tensors; the flagship differs from 2mv in head_dim (128 vs 64),
   MoE layers, and `sdp_kernel(enable_math=False)` wrappers — enough surface for a
   version-specific kernel-path divergence.
3. **The flagship DiT is attention-precision-sensitive by demonstration**: ComfyUI had to
   force `low_precision_attention=False` because SageAttention's quantized kernels produce
   all-NaN DiT output — "3D occupancy prediction requires higher precision at voxel
   boundaries; image diffusion tolerates this, 3D does not" [Comfy-Org/ComfyUI#12772].
   The upstream attention code itself carries the comment "TODO: eps should be 1/65530 if
   using fp16" on qk_norm — an admitted fp16 fragility.
4. Scheduler is exonerated: 2.1's `schedulers.py` already casts timesteps/sigmas to fp32
   with an explicit "mps does not support float64" guard (verified in vendored source).

Diagnosis ladder (cheap, decisive, in order — filed as backlog 0020): (a) monkeypatch
`F.scaled_dot_product_attention` to `.contiguous()` inputs + math backend, rerun e9 seed
2025 at 384/30; (b) same run under torch 2.9.x in a scratch venv; (c) dtype=bfloat16 (fp32
range, same memory); (d) fp32 qk_norm/softmax only. Any single green run un-blocks the
strongest local single-view geometry (3.3B vs 1.1B) AND a better clay for the whole loop.

## Axis 4 — Conditioning multiview DiTs when only one real view exists

- **Official 2mv envelope**: dict-conditioning with tags front (required) / left / back
  (/right), 512px, centered, uniform background; official snippets use {front,left,back};
  our own measurement stands: 4 views shred (822/559 bodies), 1-3 healthy, 2-3 > 1
  [tencent/Hunyuan3D-2mv card; KnowledgeBase 4-view cliff]. No official statement on
  pseudo-view tolerance exists — the model card assumes real captures. Nothing found that
  supersedes our window law + identity-route recipe; the field's equivalents are
  registration-first too.
- **Pseudo-view practice in the field**: PSHuman conditions its multiview diffusion on
  SMPL-X renders (a parametric clay — the same role our mesh clay plays); IM-3D/Ouroboros3D
  close the render->regenerate loop (our loop-2 lane); ACT-R generates ordered sequences
  and re-rolls low-consistency seeds (our redraw-until-accepted gate is the same shape)
  [arXiv:2402.08682; 2406.03184; 2505.08239]. Independent freestanding draws remain the
  worst configuration — everything we measured in the viewgen audit agrees.
- **Elevation discipline**: MVGBench: no MV method is robust off its training elevation —
  keep synthesized views at elevation 0 exactly (our row law already requires this)
  [arXiv:2507.00006]. hy3dpaint's own view selection confirms the convention: its candidate
  cameras are elevation-0 azimuths {0,90,180,270} + top/bottom, weighted front-heavy
  (1 / 0.1 / 0.5), ±20° ring at weight 0.01 (verified in `textureGenPipeline.py`).
- **View dropout**: MVDiffusion++ trains WITH view dropout (tolerates missing views);
  Hunyuan3D-2mv documents optionality only through its tag dict — our measured 3-view cap
  is the operative rule, not a published one.

## ADOPTION RANKING (our constraints; effort = engineer-days to first honest verdict)

| # | Adoption | Effort | Expected gain | Risk | License |
|---|---|---|---|---|---|
| 1 | **hy3dpaint-2.1 as synthesized-view/texture source on MPS** (agenticvibes fork lane), output gated through our existing bake authority (photo keeps front, paint completes) | 2-4 d (spike: hours) | kills the misregistration defect class for non-photo texels; PBR albedo (delight built in); replaces the entire fragile i2i view lane for texture | memory at defaults (21-39 GB reported) must be knob-reduced under 20 GB; face identity via reference-attention unproven; diffusers 0.38 API drift vs their custom pipeline | Tencent 2.1 community (already gated); port code MIT |
| 2 | **Fix flagship 2.1 MPS shredding** (backlog 0020 ladder) | 1-2 d | 3.3B single-view geometry + better clay everywhere; possibly also explains other MPS fp16 fragilities | may be a torch-version bug we can only pin, not fix (then: version gate + contiguous/math-SDPA patch) | already gated |
| 3 | **MV-Adapter ig2mv (SDXL) clay-conditioned views** into OUR bake + 2mv conditioning (backlog 0021) | 3-5 d | Apache-2.0 end-to-end view synthesis, geometry-conditioned (no more freestanding draws); drop-in for the texture-ref role | MPS unverified; SDXL slow on MPS; face identity unproven; their bake path (CV-CUDA) unusable — must keep ours | Apache-2.0 (weights confirmed) |
| 4 | Keep our projection bake as aggregator + add SyncMVD-class per-texel cross-view consistency weighting | 2-3 d | directly targets the residual glasses-ghost class | no new views — input quality still bounds it | ours |
| 5 | TRELLIS.2 texture stage port to MPS | weeks | PBR at high res, MIT | o-voxel/nvdiffrast/flash-attn CUDA kernel ports — out of scope | MIT |

Explicitly rejected for now: TEXTure/Text2Tex (kaolin/pytorch3d CUDA), PSHuman (>40 GB),
FLAME-family as primary (no glasses/hair/shoulders), Hunyuan3D-2.5+ (not released),
hosted texture services (operator-face policy).

**On the alternative "our projection bake is the right architecture, fix registration
instead"**: half-true, and the evidence says both. The bake architecture IS what the field
uses (hy3dpaint's final stage is the same weighted back-projection), and e20 shows the
registered lane can reach 0.75 coverage. But the i2i VIEW SOURCE is measured as the
bottleneck (40° pose errors, identity flips, 9-21% feature-row shifts, remote-route
dependency for identity) — rank 1 replaces exactly that source with mesh-conditioned
generation while KEEPING our bake as the authority layer. That is the synthesis, not a
choice between them.

## Rank-1 integration sketch (hy3dpaint on MPS)

Where it lands in our tree (new code isolated, current lane untouched as fallback):

- `src/abstract3d/backends/hy3dpaint_runtime.py` (NEW): pinned-source bootstrap of the
  agenticvibes fork (`3d8e931`) following the `_clone_repo`/`_patch_*_source` pattern from
  `step1x_runtime.py`; builds `custom_rasterizer` (CppExtension) + `mesh_inpaint_processor`
  at bootstrap with compiled-artifact caching; wraps `Hunyuan3DPaintPipeline` with
  device="mps", `use_remesh=False` (we feed our decimated 120k mesh; their remesh targets
  40k — measure both), SR off initially, `max_num_view=4`, `render_size=1024`,
  `texture_size=2048` (memory posture; raise after measurement).
- `src/abstract3d/backends/hunyuan3d_runtime.py`: optional `texture_backend="hy3dpaint"`
  branch after shape; existing projection bake remains the default and the fallback
  (`#FALLBACK` warning when paint fails/OOMs).
- `src/abstract3d/texturing.py`: accept paint-produced per-view images as texture
  references with `synthesized=True` authority (protect-observed-texels keeps the photo's
  texels) — OR accept its baked atlas as the completion layer under the same protection.
  Two integration modes; A/B them with `scripts/bust_assessment.py` + oblique closeups.
- License: reuse the existing `scene3d_hunyuan_license_accepted` gate (same license text).
- Downloads: `hunyuan3d-paintpbr-v2-1/*` (~3.7 GB) + `facebook/dinov2-giant` (~4.5 GB)
  (+ optional MLX weights ~4 GB later). Disk ~8.2 GB; budget approved paths only.
- Memory budget: UNet 2B fp16 ~4 GB + DINO-giant ~2.4 GB + VAE/text encoder ~1 GB +
  4-view latents/renders at 1024 ~2-3 GB + 2048 atlas tensors <1 GB => ~10-12 GB estimated
  peak WITHOUT SR; the 21-39 GB reported figures are 6-view/4096/SR configs. Must measure
  (spike reports load-time RSS; first full run reports generation peak).
- Fallback story: any refusal (build failure, OOM, identity gate failure on the face crop)
  falls back to the current e20 recipe; paint output that fails our acceptance gates is
  discarded per the floor-only surrender rule.

**Spike RUN on this machine (2026-07-21, `scripts/experimental/hy3dpaint_mps_spike.py`,
report at `~/.cache/abstract3d/experimental/hy3dpaint-spike/spike_report.json`) — all 10
stages green; generation deliberately NOT run (GPU contention discipline):**

- `custom_rasterizer` COMPILED on macOS as a plain CppExtension (fork `3d8e931` setup.py)
  and the CPU kernel rasterized a real triangle correctly (findices 64x64, barycentric
  64x64x3, 1250 covered texels) — the "CUDA-only" blocker is build-script-only, confirmed.
- `mesh_inpaint_processor` COMPILED on macOS (new build dep: `pip install pybind11` —
  now in the venv; the first attempt refused on exactly that).
- Weights downloaded: paint 3.7 GB (1090 s) + dinov2-giant (680 s).
- The FULL paint pipeline LOADED on MPS fp16 in 9 s: UNet 1.96B params on mps:0 fp16,
  DINO-giant loaded, view_size 512; DINO dry-forward on MPS returned (1, 257, 1536).
- Refusals found and shimmed (must be fixed properly at integration): (a) diffusers 0.38
  refuses the model-dir custom UNet code without `trust_remote_code=True` — the spike
  shims it; integration must VENDOR the UNet class (abstractmusic precedent: no
  trust_remote_code at runtime); (b) the paint VAE ships only pickle `.bin` (no
  safetensors) — diffusers warns "Defaulting to unsafe serialization"; convert once at
  bootstrap; (c) the fork copies its patched unet modules INTO the HF snapshot (cache
  mutation) — integration must load from a copied model dir.
- Still unproven, said plainly: generation-time peak memory and face-identity quality —
  the two go/no-go questions the first full run must answer.

## Sources

- Tencent-Hunyuan/Hunyuan3D-2.1 (README, hy3dpaint sources incl. `textureGenPipeline.py`,
  `custom_rasterizer/setup.py`, `rasterizer.cpp/.h`, `schedulers.py`, `moe_layers.py`) —
  verified in the pinned vendored snapshot. VRAM 10/21/29 GB: official README.
- Hunyuan3D-2 issues #151 (MacOS support thread: MPS shape, metal rasterizer port,
  Rc121122 24 GB workflow), #320 (M2 build errors), #265 (timestep warning);
  Hunyuan3D-2.1 issue #32 (macOS ports incl. NetLops). Comfy-Org/ComfyUI PR #12772
  (SageAttention NaN in 2.1 DiT).
- agenticvibes/ComfyUI-Hunyuan3d-Paint (`3d8e931`, README + setup.py verified);
  ZimengXiong/Hunyuan3D-MLX (measured wall/memory table);
  VladimirTalyzin/hunyuan3d-2.1-mac (install.sh/fix.sh/gradio_app.py read: patches are
  device-strings, autocast no-op, ckpt loader; flagship works on MPS fp16).
- pytorch/pytorch #163597 (MPS SDPA non-contiguous fast-path garbage, 2.8+),
  #179352 + PR #179592 (2^32 score-matrix overflow), #174861 (2-pass OOB, fixed 2.11),
  #176767 (value-dim shape bug).
- LIU-Yuxin/SyncMVD (Linux+NVIDIA+PyTorch3D); TEXTurePaper (kaolin); daveredrum/Text2Tex
  (pytorch3d+xformers, torch 1.12); kaolin install docs (CUDA required for full function).
- huanngzh/MV-Adapter (Apache-2.0 repo + HF weights; ig2mv >16 GB SDXL / <10 GB SD2.1;
  texture path requires CV-CUDA); microsoft/TRELLIS.2 (MIT; Linux + 24 GB NVIDIA;
  o-voxel/flash-attn/nvdiffrast/nvdiffrec compile deps).
- pengHTYX/PSHuman (MIT; >40 GB VRAM FAQ; dormant); DECA (SIGGRAPH'21 limitations: facial
  hair/occluders as shape deformations; no hair/glasses model); EMOCA CVPR'22;
  ID-Sculpt AAAI'25; PercHead arXiv:2511.02777; MVCHead CVPR'26 (identity machinery,
  CUDA lanes); VAST-AI-Research/TripoSG (MIT, CUDA >=8 GB stated).
- tencent/Hunyuan3D-2mv model card (dict conditioning, official 3-view snippet);
  meshy.ai Hunyuan comparison + official GitHub news (2.5/3.x hosted-only; 2.1 last open
  release; EU/UK/KR territory exclusion). MVGBench arXiv:2507.00006 (elevation
  robustness); MVDiffusion++ arXiv:2402.12712 (view dropout); IM-3D arXiv:2402.08682;
  Ouroboros3D arXiv:2406.03184; ACT-R arXiv:2505.08239.
