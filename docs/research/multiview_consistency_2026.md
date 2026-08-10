# Multi-View Consistency for One-Photo + Synthesized-Views Reconstruction (research distillation, 2026-07)

Scope: state-of-the-art mechanisms for keeping model-synthesized additional views mutually consistent
so that mesh conditioning (Hunyuan3D-2mv) and the projection texture bake stop carving/painting
duplicated features (live defect: facial features synthesized at different heights across views →
double mouth in geometry and texture). Sources: the Feed-Forward 3D survey (CGF 2026) [S1], the
3DV 2026 accepted-papers list [S2], and the primary papers cited below. This document recommends;
it changes no code.

## 1. Taxonomy of consistency mechanisms

**(A) Joint generation with cross-view attention.** Instead of sampling each view independently
(each sample is a *different* draw from the conditional distribution — the root cause of our defect),
the joint family denoises all views in one process so information flows between them: Zero123++ tiles
six views into a single 3×2 frame so one latent models the joint distribution [Zero123++,
arXiv:2310.15110]; MVDream [arXiv:2308.16512, ICLR 2024] and Wonder3D [arXiv:2310.15008, CVPR 2024]
add multi-view self-attention (Wonder3D additionally generates normal maps jointly with RGB);
SyncDreamer [arXiv:2309.03453, ICLR 2024] synchronizes views through a 3D noise volume; MVDiffusion
uses correspondence-aware attention [arXiv:2307.01097, NeurIPS 2023]; EscherNet conditions any number
of target views on any number of reference views via relative camera positional encoding (CaPE)
[arXiv:2402.03908, CVPR 2024]. Dense cross-view attention is the cost driver; CTR3D (3DV 2026 oral)
reduces cross-view tokens to scale it [S2]. MVGBench's systematic evaluation finds video-prior models
(e.g. SV3D [arXiv:2403.12008]) currently balance 3D consistency and image quality best [MVGBench,
arXiv:2507.00006, ICCV 2025].

**(B) Epipolar / row-structured attention.** A geometry-aware restriction of (A): attention is only
computed along epipolar correspondences instead of over all pixel pairs. Epipolar attention appears
in generalizable NeRF/3DGS (GNT, PixelSplat's epipolar transformer [arXiv:2312.12337, CVPR 2024])
and in camera-controlled video diffusion (CamCo) [S1]. Era3D is the direct precedent for our setting:
in a canonical rig of orthographic cameras at elevation 0°, epipolar lines degenerate to image rows,
so a cheap row-wise attention layer enforces the epipolar prior with ~12× less compute at 512²
[Era3D, arXiv:2405.11616, NeurIPS 2024]. Section 2 verifies the geometry.

**(C) Post-hoc alignment / warping.** Generate first, then fix: ConsistNet runs per-view diffusion
in parallel and aligns the results [S1]; AlignCVC re-frames the generation→reconstruction pipeline as
distribution alignment — soft alignment (score distillation) on the multi-view generator and hard
(adversarial) alignment on the reconstructor, plug-and-play over existing model pairs, and it also
collapses inference to as few as 4 steps [AlignCVC, arXiv:2506.23150, AAAI 2026]. Our existing
`register_matte_to_clay` (scale/shift search maximizing silhouette IoU) and the flow-based
`reference_flow` repair are pipeline-local members of this family; their known limit is that
similarity/flow warps correct placement, not content identity disagreements.

**(D) Iterative refinement (reconstruct → re-render → re-generate).** Close the loop between the 3D
state and the 2D generator so the 3D consensus disciplines the next 2D sample: IM-3D renders the
current 3D Gaussian model, adds noise, re-invokes the (video) generator, re-reconstructs — the loop
is closed "two or three times per generated asset" and one iteration already resolves the
superimposed-copies artifact (their Fig. 4 — the same failure class as our double mouth) [IM-3D,
arXiv:2402.08682, ICML 2024]. Ouroboros3D moves the feedback *inside* the denoising loop: at each
step the reconstruction's rendered color/coordinate maps condition the next denoising step
[arXiv:2406.03184, CVPR 2025]. Instruct-NeRF2NeRF's Iterative Dataset Update alternates editing
single dataset views with 3D optimization until the 3D representation absorbs a consistent consensus
[arXiv:2303.12789, ICCV 2023]. Our optional clay-conditioned regeneration is one un-closed iteration
of this loop.

**(E) Reconstruction-side robustness.** Accept that inputs disagree and make the 3D stage tolerant:
robust image-space losses over 3DGS (IM-3D) [arXiv:2402.08682]; view dropout during training
(MVDiffusion++ [arXiv:2402.12712, ECCV 2024]) [S1]; monocular-depth priors stabilizing multi-view
matching (DepthSplat) [S1]; and per-view attenuation at texture time (Section 3c). Our acceptance
gates (silhouette IoU ≥ 0.75, palette identity) are this family's front line — but they judge each
view against the *clay*, never against *each other*, which is exactly the blind spot the defect
exploits: two views can each match the silhouette while placing the mouth at different heights.

## 2. The degenerate-epipolar insight — verified

**Claim under review**: for same-elevation turntable views under near-orthographic projection,
cross-view feature correspondence reduces to same-image-row.

**Verdict: confirmed, with a sharper condition than "same elevation" — the elevation must be 0°.**
Era3D states and proves it as Proposition 1: "If two orthogonal [orthographic] cameras look at the
origin with their y coordinate aligned with gravity direction and their elevations of 0°, then for a
pixel with coordinate (x,y)=(u,v) on one camera, its corresponding epipolar line on other views is
y=v" [Era3D §3.3, arXiv:2405.11616]. The mechanism: with orthographic projection the image row of a
3D point is `v = u_cam · X` (dot product with the camera's up vector); at elevation 0° with no roll,
*every* camera on the turntable shares the identical world-up vector, so `v` equals the point's world
height for all azimuths, and viewing rays (horizontal) never change a point's height coordinate.

**Where it holds and breaks, precisely:**
- **Holds exactly**: orthographic projection + all optical axes horizontal (elevation 0°) + zero
  roll + shared vertical axis. Azimuth is free — any pair of turntable views qualifies.
- **Perspective breaks it gradually**: epipolar lines fan through the epipole instead of staying
  horizontal; the row approximation is exact only on the principal row and degrades with field of
  view and with vertical distance from it. Era3D's whole camera-canonicalization module exists
  because real photos are perspective: it *predicts* focal length/elevation of the input and emits
  orthographic elevation-0 outputs rather than assuming the input obeys the rig
  [Era3D §3.1–3.2]. Under weak perspective (object depth ≪ camera distance, long focal), the error
  is small; camera-model mismatch is real enough that Zero123++ v1.2 pins its output rig to a fixed
  30° FOV and Era3D's authors identify perspective-assumption failure as a first-class distortion
  source [Zero123++ repo, SUDO-AI-3D/zero123plus; Era3D §1].
- **Non-zero elevation breaks it even when shared**: at elevation e ≠ 0 each camera's up vector
  tilts differently in world space. With up_cam = (−sin e·cos a, −sin e·sin a, cos e), a point at
  horizontal radius ρ and phase φ lands on row v = −ρ·sin e·cos(φ−a) + h·cos e, so between azimuths
  the row shifts by up to 2ρ·sin e (derivable in two lines; consistent with Era3D restricting its
  Proposition 1 to e = 0). At e = 15° that is ≈ 0.52·ρ — for a facial feature a few centimeters off
  the turntable axis, several percent of head height: visibly "a mouth at different heights".
- **Camera roll breaks it directly**: rows rotate away from the world-horizontal; a roll of θ
  displaces a feature at horizontal offset x by ~x·sin(θ) rows.

**Implication for abstract3d**: our clay renders are generated at a fixed declared elevation with an
orthographic option already in the bake (`projection_model="orthographic"`, methodology.md). If the
reference-generation rig keeps elevation at 0° (or the harness corrects for the predictable non-zero
offset by projecting through the mesh), then *"same feature ⇒ same row" is a hard constraint we get
for free* — usable both as a generation-time prior (Section 3b) and as a nearly free consistency
metric (Section 4). Caveat: the i2i model's outputs may embed implicit perspective and small roll
that the current scale/shift registration does not remove; the row constraint is therefore a gate,
not an assumption.

## 3. Concrete recommendations for this pipeline

**(a) i2i denoise strength / guide-conditioning weight.** Reported working points for
depth/structure-conditioned generation: practitioner consensus puts img2img denoising strength at
0.35–0.55 for strong structure retention (0.6+ trades structure for makeover) and ControlNet-style
conditioning scale at 0.5–0.8 for depth-class guides, with depth guidance "falling apart" below ~0.5
and over-constraining ("stiff", prompt-ignoring) near 1.0; a two-pass schedule (structure-lock at
0.75–0.85, then detail pass at 0.35–0.5 or early-stop conditioning around 60% of steps) is the
standard escape from the fidelity/quality bind [practitioner guides: sider.ai ControlNet guide,
theneuralbase.com conditioning-scale course, modl.run ControlNet guide — *community values, not
peer-reviewed*]. Official anchor points: Zero123++'s shipped ControlNet example (normal generation
conditioned on a depth render) runs conditioning_scale=1.0 at CFG 4 — a full-strength geometry lock
is *wanted* when the guide is trusted geometry [SUDO-AI-3D/zero123plus `examples/normal_gen.py`];
Era3D samples at CFG 3.0, 40 DDIM steps [Era3D §4]; Hunyuan3D-2.1's official default is
guidance 5.0 [docs/methodology.md]. For our klein (distilled FLUX-family) backend, which exposes
steps/seed but no strength dial, the measured in-repo equivalents already agree with the literature's
direction: 4 distilled steps are too few for micro-texture, the 8/12 alternating ladder is measured
best (reference_generation.py), and the composite photo+clay panel plays the role of a
conditioning-scale ≈ high setting. **Expected effect**: keeping an effective strength ≤ ~0.55 (or the
panel+IoU-gate equivalent) preserves the clay's feature *placement*, which is the consistency-bearing
signal; **cost**: lower strength also copies clay blandness — the escalation ladder already
compensates. If abstractvision later exposes true depth/normal ControlNet for FLUX, start at
conditioning scale 0.6–0.8 with CFG ~4–5 and only then relax.

**(b) View generation order matters — go sequential-autoregressive.** Evidence that *unordered
independent* sampling is the worst configuration: 3DiM introduced autoregressive generation with
stochastic conditioning on previously generated views precisely because independent draws break
consistency [arXiv:2210.04628, ICLR 2023]; ViewFusion extends this auto-regressive scheme with
interpolated denoising [CVPR 2024]; ACT-R (3DV 2026) generates an *ordered sequence* along an
adaptive orbit through a video model, explicitly "instead of producing an unordered set of views
independently or simultaneously", and filters low-consistency sequences by re-rolling seeds
[arXiv:2505.08239]. MVGBench adds a robustness argument for orderly rigs: no evaluated method is
robust to off-training elevations, so keep every synthesized view at the model's training-canonical
elevation [arXiv:2507.00006]. **Recommendation**: (1) fix the ring order front → ±45° → ±90° → 180°;
(2) after a view is *accepted*, add it (nearest accepted neighbor) to the next view's composite panel
alongside the source photo and clay — each new view is then conditioned on committed evidence, not
just on the mesh; (3) keep the real photo in every panel as the drift anchor — sequential pipelines
accumulate error (the survey notes this failure mode for sequential reconstruction, e.g. Spann3R
[S1]), and the photo is the only non-synthetic anchor we own. **Expected effect**: removes the
independent-draw variance that produces feature-height disagreement between adjacent views; **cost**:
serializes generation (no parallel angle synthesis; wall-clock ≈ unchanged on MPS since we already
run one heavy job at a time), slightly larger conditioning canvas.

**(c) Per-view texture paint weighting by consistency score.** The field's texture fusers weight
per-texel contributions by view reliability: cosine(view direction, surface normal) raised to an
exponent α, with α *scheduled* low→high so early fusion builds consensus and late fusion preserves
detail [SyncMVD, "Text-Guided Texturing by Synchronized Multi-View Diffusion", SIGGRAPH Asia 2024,
arXiv:2311.12891]; TexFusion aggregates per-view denoising into a shared latent texture each step and screens views by
image-space UV derivatives (quality ∝ sampling density) [ICCV 2023]; MVPaint fuses decoded images
into UV space with the same cosine weighting [CVPR 2025]; the classic precedent is view-dependent
texture mapping's smooth view blending [Debevec et al., EGRW 1998]. Our bake already applies a
facing law (`strength**2` on visible vertices, facing_threshold=0.2). **Recommendation**: multiply
that geometric weight by a per-view *consistency* score s_v ∈ [0,1] computed against the other
accepted views (Section 4 metrics: row-displacement + mesh-mediated reprojection agreement), and
sharpen the blend (winner-take-most, higher effective α) in regions where views disagree — averaging
two mouths paints two mouths; picking the dominant-consistency view paints one. **Expected effect**:
directly suppresses the painted double-feature even when a bad view slips past the gates; **cost**:
one extra weight map per view; no new dependencies.

**(d) Iteration count for reconstruct→re-render→re-generate loops.** IM-3D closes the loop "two or
three times per generated asset" and shows the first refinement iteration resolves most
superimposed-geometry artifacts, with reconstruction taking "a few seconds" per subsequent iteration
in their 3DGS setting [arXiv:2402.08682, §4.2 + Fig. 4]. AlignCVC's stated motivation is that
feedback approaches "struggle with noisy and unstable reconstruction outputs that limit effective
CVC improvement" — the diminishing-returns mechanism: once the reconstruction's own noise dominates
the residual inconsistency, further loops stop paying [arXiv:2506.23150]. Instruct-NeRF2NeRF style dataset-update runs
many micro-iterations, but that regime assumes a differentiable scene being optimized, not a
feed-forward shape DiT [arXiv:2303.12789]. **Recommendation**: budget **2 full loops** (initial
reconstruction + one regenerate-with-new-clay + reconstruct), and trigger the second loop only when
the Section 4 disagreement metrics fail — evidence says iteration 1→2 pays, 3+ is mostly cost. On
our MPS profile each extra loop ≈ one Hunyuan3D-2mv pass (~minutes) + one gated i2i ladder per angle
(bounded by the 15-attempt cap), so unconditional looping is not free the way IM-3D's 3DGS refit is.

**(e) Locally-runnable JOINT multi-view models (Apple Silicon, ≤ 20 GB) — catalog, not endorsement.**
None of these are MPS-verified by us; "plausible" below means standard diffusers UNet pipelines with
no hard CUDA-kernel dependency in the inference path. abstract3d is MIT-licensed — AGPL/NC weights
are a distribution decision for the maintainer, flagged per row.

| Model | Checkpoint | Base / VRAM (reported, CUDA) | License | MPS outlook |
|---|---|---|---|---|
| Zero123++ v1.2 (+ depth/normal ControlNet) | `sudo-ai/zero123plus-v1.2`, `sudo-ai/controlnet-zp12-normal-gen-v1` | SD2-class; 6 views tiled in one 640×960 frame, fixed elevations 20°/−10°; modest (<8 GB fp16) | code Apache-2.0; **weights CC-BY-NC-4.0** (no commercial pipeline) | plausible (plain diffusers custom pipeline); NC gate is the blocker [SUDO-AI-3D/zero123plus] |
| MV-Adapter (i2mv / **ig2mv geometry-guided**) | `huanngzh/mv-adapter` (`mvadapter_i2mv_sdxl`, `mvadapter_ig2mv_sdxl`, SD2.1 variants) | SDXL ~13–16 GB, SD2.1 <10 GB (reported, CUDA) | **Apache-2.0** (GitHub repo incl. ComfyUI ext; HF card shows no separate license — verify weights terms before shipping) | best candidate: adapter over stock diffusers, no custom kernels advertised; ig2mv consumes geometry — a drop-in for our clay-guided role [huanngzh/MV-Adapter] |
| Wonder3D | `flamehaze1115/wonder3d-v1.0` | SD2-class, 6 views ×(RGB+normal), elevation-0 in the *input camera frame* (matches our rig); ~<10 GB | repo LICENSE now MIT (2025) but HF pipeline/weights still tagged **AGPL-3.0** — verify before shipping | plausible (diffusers custom pipeline; xformers optional) [xxlong0/Wonder3D] |
| Era3D | `pengHTYX/MacLab-Era3D-512-6view[-ortho]` | SD2.1-unclip, 512², 6 views; row-wise attention | **AGPL-3.0** (code and, per README, products embedding the model) | doubtful: repo environment pins CUDA-oriented deps (xformers; tiny-cuda-nn for its NeuS stage); the diffusion stage itself is diffusers-based but unverified off-CUDA. Geometrically the ideal rig match [pengHTYX/Era3D] |
| EscherNet | `kxic/eschernet-4dof` / `-6dof` | SD-class; N reference → M target views (fits autoregressive accumulation) | license not confirmed in this pass — check before any use | plausible architecture; unverified [kxhit/eschernet] |
| SV3D / video-prior models | Stability weights | video diffusion, 21-frame orbits; heavier | Stability community license (restrictions) | doubtful under 20 GB on MPS at useful resolution; noted because MVGBench ranks video-prior models best on the consistency/quality trade-off [arXiv:2507.00006] |

**Recommendation**: if a joint model is ever adopted, spike MV-Adapter ig2mv (Apache, geometry-
conditioned — closest contract match to "clay in, photo-like views out") on an MPS smoke test before
any integration work; treat Zero123++ as the reference *baseline* for joint-generation quality that
our gated-i2i lane must beat, license permitting. Do not integrate anything without a demonstrated
MPS run within the memory profile.

## 4. Metrics: measuring multi-view / 3D self-consistency

**Field practice.** MVGBench splits the N generated views into two (near-)disjoint subsets, fits an
independent 3DGS to each, and reports the discrepancy between the two reconstructions: Chamfer
distance on covariance-resampled surface points and rendered depth error (geometry), plus
cPSNR/cSSIM/cLPIPS between same-camera renderings (texture) — explicitly designed so no ground-truth
3D is needed [arXiv:2507.00006 §3]. MEt3R measures generated-image consistency via DUSt3R-based
warping (heavier dependency; noted, not recommended here) [cited in MVGBench related work].
Classical stand-ins: reprojection error and pose-conditioned rendering metrics [S1 §evaluation].
"Appreciate the View" (3DV 2026) argues task-aware NVS evaluation over raw PSNR [S2].

**Recommended for our harness (no new heavy deps — OpenCV, scikit-image, scipy, moderngl, trimesh
are already extras):**

1. **Row-displacement of matched features (the degenerate-epipolar metric).** For every pair of
   accepted same-elevation views, match sparse features (OpenCV ORB/SIFT, already available) inside
   the registered mattes and report the median |Δrow| normalized by silhouette height; at elevation
   0° any nonzero value is inconsistency by Proposition 1 [Era3D]. A featureless-subject fallback
   with numpy only: per-row gradient-energy profiles of the matte interior, aligned by correlation —
   the displacement of profile peaks *is* the feature-height variance ("mouth at different heights"
   is precisely a peak-row disagreement). Cost: milliseconds; catches the live defect class directly.
2. **Mesh-mediated reprojection error.** Project accepted view i onto the current mesh (the bake's
   projector already computes per-texel world positions), render that texture into view j's camera
   (moderngl), and compare against accepted view j in the mutually visible region (masked Lab ΔE or
   SSIM via scikit-image). This is classical reprojection error with the mesh as the correspondence
   oracle [S1]; it also works at non-zero elevations where the row metric degrades. Doubles as a
   photometric pose check: minimizing it over small pose perturbations before the bake is the
   render-and-compare pose refinement of iNeRF [arXiv:2012.05877, IROS 2021] — consistent with the
   repo's existing rule that declared angles be measured, not assumed [docs/methodology.md].
3. **MVGBench-lite disjoint-bake self-consistency.** Split accepted views into two subsets, run the
   (cheap, model-free) projection bake twice onto the same mesh, and compare the two atlases on
   co-covered texels (Lab ΔE percentile). Full MVGBench semantics would re-run Hunyuan3D-2mv per
   subset — reserve that expensive geometry variant for release proofs, run the bake-only variant in
   CI. Directly implements the "compare reconstructions from disjoint generated views" idea at our
   budget [arXiv:2507.00006].

Thresholds should be calibrated on the existing critic-labeled acceptance sets (the repo already
maintains measured gate calibrations); the literature does not supply transferable absolute numbers
for these object-scale metrics.

## Sources

- [S1] Zhang et al., "Advances in Feed-Forward 3D Reconstruction and View Synthesis: A Survey", CGF 2026. — [S2] "Awesome 3DV 2026 Papers" (accepted-paper list, 3DV 2026, Vancouver).
- Era3D: Li et al., NeurIPS 2024, arXiv:2405.11616. — MVGBench: Xie et al., ICCV 2025, arXiv:2507.00006. — AlignCVC: Liang et al., AAAI 2026, arXiv:2506.23150.
- IM-3D: Melas-Kyriazi et al., ICML 2024, arXiv:2402.08682. — Ouroboros3D: Wen et al., CVPR 2025, arXiv:2406.03184. — Instruct-NeRF2NeRF: Haque et al., ICCV 2023, arXiv:2303.12789.
- ACT-R: Wang, Zhao, Zhang, 3DV 2026, arXiv:2505.08239. — 3DiM: Watson et al., ICLR 2023, arXiv:2210.04628. — ViewFusion: Yang et al., CVPR 2024. — EscherNet: Kong et al., CVPR 2024, arXiv:2402.03908.
- Zero123++: Shi et al., arXiv:2310.15110. — Wonder3D: Long et al., CVPR 2024, arXiv:2310.15008. — SyncDreamer: Liu et al., ICLR 2024, arXiv:2309.03453. — MVDream: Shi et al., ICLR 2024, arXiv:2308.16512.
- MVDiffusion: Tang et al., NeurIPS 2023, arXiv:2307.01097. — MVDiffusion++: Tang et al., ECCV 2024, arXiv:2402.12712. — MV-Adapter: Huang et al., arXiv:2412.03632. — SV3D: Voleti et al., ECCV 2024, arXiv:2403.12008.
- PixelSplat: Charatan et al., CVPR 2024, arXiv:2312.12337. — iNeRF: Yen-Chen et al., IROS 2021, arXiv:2012.05877. — TexFusion: Cao et al., ICCV 2023. — SyncMVD: Liu, Xie, Liu, Wong, SIGGRAPH Asia 2024, arXiv:2311.12891.
- MVPaint: Cheng et al., CVPR 2025. — VDTM: Debevec et al., EGRW 1998. — Practitioner guides (non-peer-reviewed, labeled in §3a): sider.ai, theneuralbase.com, modl.run ControlNet guides.
