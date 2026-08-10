# Evaluating Single-Image 3D Reconstructions: Mesh and Texture as Independent Axes (research distillation, 2026-07)

Scope: state of the art in EVALUATING single-image 3D reconstructions — mesh quality and texture
quality separately, each at multiple viewing angles (operator ruling 2026-07-21) — and what
transfers to our constraints: human busts from one photo (Hunyuan3D-2mv on MPS + projection bake),
local-only, torch/torchvision/onnxruntime/OpenCV available, no remote calls. Companion documents:
`multiview_consistency_2026.md` (generation-side consistency), `texture_forensics.md`.
This document justified `scripts/model_quality_sweep_v2.py`; calibration evidence is at the bottom.

## 1. Taxonomy: what can be measured, given what reference

**(A) Ground-truth-mesh metrics (inapplicable here).** Benchmarks with 3D GT report Chamfer
distance, F-score@τ, volume/surface IoU, normal consistency after rigid alignment: GSO/OmniObject3D
evaluations sample ~16k surface points, CD + F-score@0.02 [AR-1-to-3, arXiv:2503.12929]; Hunyuan3D
2.0 itself reports V-IoU/S-IoU vs GT occupancy [arXiv:2501.12202 §5.1]. Face-specific benchmarks
refine this: NoW computes scan-to-mesh distance after landmark-seeded rigid alignment
[now.is.tue.mpg.de]; REALY does region-wise (nose/mouth/forehead/cheek) alignment + NMSE, showing
global alignment can hide region defects [realy3dface.com; github.com/czh-98/REALY]. We have no
scan, so this family only applies if we ever acquire one; the region-wise idea (score the mouth
band separately, not the whole head) transfers directly and is used in v2.

**(B) Reference-image metrics (our front view).** The input photo is ground truth at its own pose.
Image-to-3D papers render the asset at the input pose and compare: PSNR/SSIM/LPIPS + CLIP
similarity [Cycle3D, AAAI 2025; AR-1-to-3 §4.1]. Hunyuan3D 2.0 evaluates textured assets with
FID_CLIP, CMMD, CLIP-score, LPIPS between renders and input prompts/images [arXiv:2501.12202 §5.3].
For faces the field adds IDENTITY: MICA uses a pretrained ArcFace as the identity encoder and the
metrical-face literature evaluates identity preservation via face-recognition embedding cosine
between input and render [arXiv:2204.06607]. Landmark reprojection error and silhouette IoU are the
classic sparse/dense alignment measures [DenseRaC, arXiv:1910.00116; Profile3DMM, arXiv:2605.01746
uses silhouette IoU + boundary Chamfer at test time].

**(C) No-reference / self-consistency metrics (our oblique views).** Away from the photo pose
nothing external exists, so the field measures internal coherence. Eval3D [CVPR 2025,
arXiv:2504.18509] probes a generated asset with foundation models and scores DISAGREEMENT:
geometric consistency = mesh-rendered normals vs normals predicted from the RGB render (Omnidata);
semantic consistency = per-3D-point std of DINO features across views. MVGBench [ICCV 2025,
arXiv:2507.00006] fits two 3DGS from disjoint view subsets and reports CD/depth (geometry) +
cPSNR/cSSIM/cLPIPS (texture) between them — "3D self-consistency" without GT. GPT-4V-as-evaluator
[CVPR 2024, Wu et al.] feeds 4–9-view RGB grids AND surface-normal-render grids, finding normal
renders necessary for geometry judgments. User studies remain the gold standard everywhere
[Text2Tex ICCV 2023; TEXTure arXiv:2302.01721; Hunyuan3D 2.0's 50-volunteer study] — our operator
verdicts are exactly this, used as calibration labels.

**Protocol norms.** View counts for image metrics: 20 fixed viewpoints at 512² [Text2Tex §4.1],
20 views at 224² [AR-1-to-3], up to 180 views vs GT [Cycle3D]. These sweeps exist to average out
view luck for FID-class statistics; for DEFECT DETECTION on one subject, a small set of adversarial
poses (raking obliques) beats many random ones — our wave-3 lesson (front/45° hid what 55° reveals)
is the same lesson as REALY's region-wise argument, applied to poses. Minimal honest panel for one
asset: input-pose reference comparison + ≥2 off-reference poses + geometry inspected independently
of texture (normals), which is what v2 implements.

## 2. Rendering FOR evaluation (why v1's mesh metrics failed)

v1 scored clay renders lit by a headlight (diffuse = n·v): positive on every visible surface, so
creases read as gentle valleys and parallel mouth ridges nearly vanish — the flat-lighting failure
the operator saw past because Blender's default viewport shading is directional. The graphics
practice for surface inspection is the opposite: MatCaps ("Check" variants) exist because they give
"higher surface curvature readability" for "small surface details and imperfections"
[developer.blender.org/docs/features/interface/matcaps]; sculptors orbit a metallic MatCap to catch
dents and flats [novedge.com ZBrush tips]. Raking light is the photographic equivalent. But every
lit render still convolves geometry with a lighting choice.

**Normal-map rendering removes the choice**: emit view-space normals as RGB. It is the standard
geometry-evaluation surface in the current literature — Eval3D renders normal maps as one of its
two probe surfaces [arXiv:2504.18509 §3]; GPT-4V-as-evaluator adds world-normal renders because RGB
alone fails geometry comparisons [CVPR 2024]; Wonder3D/Era3D generate normal maps precisely because
they carry geometry independent of shading [multiview_consistency_2026.md §1]. On our stack this is
~40 lines of moderngl (fragment shader `n*0.5+0.5`, second float attachment carrying world
positions so image regions can be anchored in 3D), with a numpy vertex-splat fallback at
vertex-density-matched resolution when no GL context exists — both implemented in
`model_quality_sweep_v2.GeoRenderer`. matplotlib (`Poly3DCollection` with unshaded facecolors =
face normals) is a third option, not needed so far.

## 3. What transfers, and what v2 does with it

Constraint recap: local, MPS-capable but CPU-fine at this scale, no new heavy deps. Available and
verified on this machine: OpenCV 4.10 with `FaceDetectorYN` (YuNet, 232 KB ONNX, 75,856 params,
5 landmarks, millisecond CPU [Wu et al., Mach. Intell. Res. 2023; opencv_zoo]) and
`FaceRecognizerSF` (SFace, 37 MB ONNX, 0.994 LFW-class accuracy [opencv_zoo]); `lpips` pip package
(pure-Python over torch; squeeze backbone ~5 MB) [github.com/richzhang/PerceptualSimilarity];
torchmetrics ships DISTS/LPIPS alternatives. InsightFace/ArcFace, DINO, CLIP, Omnidata are heavier
(hundreds of MB + new deps) → backlog 0021/0022.

**Mesh axis (both metrics on normal renders, mouth-region-anchored).** YuNet detects the face on
the FRONT normal render (works: conf 0.66–0.88 across all 9 calibration bundles); its landmarks are
lifted to 3D through the position buffer, defining one anatomical mouth band that every view then
measures (REALY's region-wise principle; the band is defined in nose-to-mouth spans, not pixels).
  - `mesh_front_defect` / `mesh_oblique_defect` = coherent-ridge energy: vertical gradient of the
    vertical view-space normal component, smoothed ALONG the ridge direction so horizontally
    coherent striations/duplicated creases survive while isotropic micro-texture (stubble,
    marching-cubes noise) cancels; patch resampled to fixed rows for scale invariance. This is a
    curvature statistic computed in image space — the render IS the curvature transport — and it is
    the direct quantitative analog of the operator's raking-light inspection.
  - Duplication surcharge: in a narrow band around the mouth line, one mouth = one dominant crease;
    the prominence of the SECOND-deepest crease (profile binned by world-z, view-independent)
    surcharges `mesh_front_defect` when it rivals the first (e2's double mouth: 0.476 vs the 0.35
    free threshold; all operator-good bundles ≤0.34).

**Texture axis (kept from v1 — they already reproduced operator verdicts).** `tex_front_dE` (CIE76
ΔE vs photo over the shared face band) is a reference-image metric of family (B); Hunyuan3D 2.0's
LPIPS-vs-input plays the same role [arXiv:2501.12202]. `tex_oblique_ghost` (extra dark bands +
saturation speckles) is a hand-rolled no-reference detector for our two known texture failure
classes. MVGBench-class cross-view texture self-consistency (cLPIPS) does NOT transfer as-is:
a baked texture is 3D-consistent by construction; the inconsistency lives in the multi-view
GENERATION stage, which this repo already gates (row law, `view_consistency.py`).

**Optional extras (never rank drivers; loud notes when missing).**
  - `sface_identity`: SFace cosine photo ↔ textured front render, MICA-style identity preservation
    [arXiv:2204.06607]. Calibration: flags e20 hardest (0.30 — below OpenCV's 0.363 same-identity
    threshold; its lips are painted on the nose) and scores e15's best-texture 0.82. But e19
    (texture "recovered") scores 0.42 — glasses + render domain shift make single readings noisy.
  - `lpips_face`: LPIPS-squeeze on landmark-anchored face crops, photo ↔ textured front. Ranks e20
    worst (0.576) / e15 best (0.326), agreeing with the operator's texture poles.

**Measured and REJECTED on calibration data (kept as diagnostics at most).**
  - Landmark-row deltas photo↔render (mission hypothesis): silhouette-normalized nose/mouth rows
    differ by ≤0.05 noise across good AND bad bundles — every bundle carves features at roughly
    photo heights because upstream row gates already enforce it; box-relative variants measure
    detection-box placement, not geometry. Reported as `landmark_row_delta` (red flag >~0.2), not
    ranked. Silhouette IoU after registration: 0.92–0.95 with no verdict correlation (bust
    silhouettes are all near-identical; the defects live inside the silhouette; also e20 scored
    LOWEST — it has the fullest hair). YuNet confidence on the normal render: correlates (good
    0.72–0.88, bad 0.66–0.76) but opaque and narrow — diagnostic only. Flat-plate normal-mode
    concentration (slab detector): no discrimination at any region tried (the e10/e15 "slab" is a
    smooth fused hair mass, not a plane). Raw autocorrelation of edge/ny profiles (v1's approach,
    any window): natural face periodicity (nose/lips/chin) autocorrelates at 0.4–0.8 for good and
    bad alike — the v1 `mesh_oblique` failure reproduced, root-caused, replaced.

## 4. Calibration result (2026-07-21, 9 bundles, operator verdicts as labels)

`mesh = mesh_front_defect + mesh_oblique_defect` (lower better), run via
`scripts/model_quality_sweep_v2.py --render-dir out/assessment/v2_normals`:

| rank | bundle | mesh_front | mesh_obl | verdict (operator) |
|---|---|---|---|---|
| 1 | e21_single_refs | 35.79 | 30.20 | mesh good |
| 2 | e20_fixed_views | 35.55 | 30.86 | mesh good (champion) |
| 3 | e18_windowed | 40.35 | 34.59 | mesh good |
| 4 | e19_split_consumers | 42.88 | 34.80 | mesh good |
| 5 | e11_2mv_reg_hq | 48.53 | 38.88 | (unpinned; slab noted) |
| 6 | e17_clean4 | 48.35 | 41.30 | striated at raking light |
| 7 | e10_2mv_registered_refs | 51.28 | 40.92 | mesh wrong (soft face/slab) |
| 8 | e15_cleanfront | 52.59 | 44.81 | mesh wrong (striated mouth) |
| 9 | e2_2mv_explicit_refs | 59.62 | 39.23 | worst (double mouth) |

All four operator-good bundles rank above all operator-bad ones ON EACH AXIS separately
(front: good ≤42.9 < bad ≥48.4; oblique: good ≤34.8 < bad ≥38.9); e2 is worst overall (the
duplication surcharge +7.6 is what pushes it past e15). Texture columns unchanged from v1
(`tex_front_dE`: e11/e10/e15 best, e18 starved-worst; ghost: e2 worst 8.0). Non-face smoke test
(x-wing GLB): loud fallback notes, no crash, extras skipped on a sub-threshold face false-positive.

Honest limits: n=9, one subject, one generator — the thresholds (mouth-band extents, ridge window,
0.35 duplication allowance) are calibrated instruments, not universal constants; re-verify on the
next subject. e2's oblique score (39.2) sits at the bad-group floor because its defect is coarse
duplication more than fine striation — the surcharge covers it. sface/lpips extras are single-view
and noisy midfield; trust them at the poles only.

## 5. Too heavy for now → filed as backlog

- 0020: Eval3D-class geometry↔texture agreement (RGB-predicted normals vs mesh normals; needs a
  normal estimator like Omnidata/DSINE ~300 MB+) and DINO per-point semantic view-consistency.
- 0021: ArcFace-class identity metric (InsightFace ONNX ~166 MB) + CLIP photo↔render similarity as
  a texture-semantics metric; SFace/LPIPS already give the cheap tier.
- 0022: promote v2 mesh metrics into the `--geometry-conditioning loop` self-verification
  (`quality_verdict`) so reconstruction runs self-gate on the calibrated instrument.

## Sources

Primary: REALY (realy3dface.com; github.com/czh-98/REALY) · NoW (now.is.tue.mpg.de) · MICA
(arXiv:2204.06607) · Eval3D (arXiv:2504.18509, CVPR 2025) · MVGBench (arXiv:2507.00006, ICCV 2025)
· GPT-4V evaluator (Wu et al., CVPR 2024) · Hunyuan3D 2.0 (arXiv:2501.12202) · Cycle3D (AAAI 2025,
doi:10.1609/aaai.v39i7.32787) · AR-1-to-3 (arXiv:2503.12929) · Text2Tex (ICCV 2023,
arXiv:2303.11396) · TEXTure (arXiv:2302.01721) · MV-consistent texturing survey point
(arXiv:2403.15559) · LPIPS (github.com/richzhang/PerceptualSimilarity; torchmetrics DISTS) · YuNet
(Wu et al., Machine Intelligence Research 2023; opencv_zoo) · SFace (opencv_zoo) · MatCap practice
(developer.blender.org; novedge.com). Verified locally 2026-07-21: cv2 4.10 FaceDetectorYN /
FaceRecognizerSF, torch 2.10 (MPS true), onnxruntime 1.24, lpips 0.1.x (squeeze), moderngl render
path on macOS CGL.
