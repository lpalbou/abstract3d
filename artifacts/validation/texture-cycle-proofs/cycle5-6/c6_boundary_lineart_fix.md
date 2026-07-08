# CYCLE-6 SOLVER — FACE-22 (region-boundary line-art on neck/chest)

**Scope:** Critic 1's single cycle-6 order — provenance of the thin
line-art contours on the neck/chest (the az0 glyph cluster "ΔΔ|" and the
az-22.5 closed chest contour), fix in the owning stage's vocabulary,
re-publish under the full battery. Repo `/Users/albou/abstract3d`, work
under `/tmp/c6/`. Date: 2026-07-07.

## VERDICT: three owners found by instrumented ablation; fixed in-mechanism; published

| gate @2048 (canonical recipe) | c5 published | c6 published |
|---|---|---|
| comp identity[front] SSIM/MAE (gate 0.70/15.0) | 0.7022 / 13.86 | **0.7037 / 14.91 PASS** |
| comp full battery | PASS (0 failed) | **PASS (0 failed)** |
| comp sides L / R | 0.6774/12.52, 0.6942/14.64 | 0.6870/12.24, 0.7059/14.39 |
| raw battery | FAIL 1 (front SSIM 0.676 raw-diagnostic; MAE 21.45) | FAIL 1 (same class: 0.678; **MAE 21.67 green**; ALL detectors green) |
| texture_qa | PASS 13/13 | **PASS 13/13** |
| determinism | 3 bakes, 1 hash | **3 bakes, 1 hash** (`928705f3edfc9036348c12bf34435d9d`) |
| ship canary md5 | b8e2b0d4... | **b8e2b0d4... == on-disk** (bit-identical) |
| owl canary md5 | ff746509... | **ff746509... == on-disk** (bit-identical) |
| test suite | 230 passed, 1 xfailed | **235 passed, 2 xfailed** (+4 new tests mine) |
| FACE-22 glyph (az0 5x, critic framing) | legible "ΔΔ\|" cluster | **gone** (smooth-skin hp p2 −6.7 → −1.6 /255) |
| FACE-22 closed contour (az-22.5 4x/5x) | crisp closed loop | **gone** (p2 −4.6 → −2.7 /255; soft tone valley remains, no line-art) |

Published to `artifacts/validation/iter3-multiview-fixed/face-2mv/` via
the FACE-21 checklist (staged → all three harnesses on the staged bytes →
commit → re-verified in place). Publication block in the metadata carries
the verification + md5s. Before/after at the critic's exact crop specs:
`/tmp/c6/before_after_critic_frames.png`.

## 1. PROVENANCE (order 1) — instrumented, ablated, texel-attributed

**Method.** (a) A stage-capture bake (`capstages6.py`) snapshotting the
texture after every stage (blend → solve → mirror/rescue → harmonic →
smooth → film → trace-commit → pale-chips → bottom-cap → detail → floor
→ fringe); the instrumented bake reproduces the published texture md5
`4d21c7ea` exactly, so the captures are the published pipeline's own
states. (b) The defect texels recovered from the critic's crops: his two
probe boxes (az0 450-650×700-900; az-22.5 400-600×720-920 at 1000 px)
mapped through the baked mesh to atlas texels; thin dark line-art =
negative high-pass (masked Gaussian, σ4) below −0.006 on smooth skin
(base > 0.6), components ≥ 8 texels → **4281 stroke texels in 66
components**, incl. the glyph cluster and the closed contour. (c) Per
component: the stage ladder of its high-pass contrast (birth stage) +
overlap with each mechanism's internal masks, captured in-bake
(`capinternal*.py`: film authority/gap/clamp/refill + S field + geodesic
distances; trace-commit committed mask; shadow-apron baseline/weight).
(d) Confirming per-mechanism ablation bakes (`ablate_film.py`,
`ablate_trace.py`) + difference maps (`ablation_sheet.png`).

**Verdict — three owners, two of them the critic's named suspects:**

| stroke class | texels | owner (mechanism, in its own terms) | evidence |
|---|---|---|---|
| GLYPH cluster + patch borders + segments | 2698 (63%) | `film_band_gradient.repaint_film_band` operating OUTSIDE its field's support: neck/chest at d_mass 9-24 pooled-transition-lengths (strokes p5 8.8T, p50 13.4T vs the honest apron's p50 2.0T), S~0.66. The GLYPH = small envelope-CLAMP components (hp −0.058 born at the film stage, comp 16: 91% in-clamp); segments = authority-stamp borders and displaced-refill component borders. | stage walk `stagewalk2048_*`; class maps `classmap_az0/azm22.png`; field measurements `field_analysis.py`; ablation: glyph gone with repaint off |
| az-22.5 CLOSED CONTOUR | 564 rim + committed surround | `commit_trace_deposits` blob RIMS: the commit retones blob interiors from ring anchors; the border mixtures sit below `deviation_min` by construction (mixture deviation = coverage × deposit deviation) and keep the old darker tone — a closed outline (rim lum 0.639 vs retoned interior 0.718). C4-3's named suspicion, in the trace commit. `commit_pale_chips` measured ZERO overlap (cleared). | signed decomposition `attrib_signed.py` (ring +0.034 at the trace stage while the stroke stays); `trace_internal.npz` overlap 0.42 ring share |
| apron-interior stripes | 487 (11%) | `reconcile_shadow_aprons` interior weight dips (bw p5 0.167 inside the component core) — inside the ACCEPTED c5 shadow gradient, cosmetic at the critic's framings; left untreated this cycle (see §4 honest limits) | `apron_check.py`, `apron_overlay_az0.png` |
| pre-existing blend/solve content steps | ~530 + shared | born at post_blend (photo content displaced at grazing incidence + soft tone patches) — witnessed content, not a mechanism's border; the commit's own conservative gates (residue rule) refuse to sweep the neighborhood, correctly | `class_dedupe.py`, `debug_residue` gate logs |
| mirror-copy borders | 153-46% of m22 residual | mirror completion pastes the lit twin verbatim (+16/255 vs destination ring); the border prints. Root cause: the gradient-domain solve runs BEFORE mirror completion, so the legacy seam-leveling's completion-tone reconciliation has no equivalent — a missing handoff, not a bug in mirror itself | `wash_prov.py`, stage ladder at pre_fill |

## 2. THE FIXES (order 2) — each in the owning mechanism's vocabulary

**A. Film repaint: FIELD SUPPORT BOUND + STAMP BORDER FEATHER**
(`film_band_gradient.py`, constants `FIELD_SUPPORT_TRANSITIONS = 6.0`,
`STAMP_BORDER_FEATHER_TEXELS = 6.0`). The S field is a RATIO of geodesic
distances (d_base/(d_base+d_mass)) — scale-free, so it takes
mid-transition values arbitrarily far from the mass, where the pooled
falloff profile it interpolates was never measured. The mechanism's own
scale bounds its domain: the operating shell requires d_mass ≤ 6
transition lengths, feathered over the last transition (6 separates the
measured stroke sites, p5 8.8T, from the honest apron, zone p50 2.0T,
with margin both sides). Authority stamps blend composite → photo over 6
texels at treated-region borders — EXCEPT against the dark mass, where
the stamp continues the mass's own content (feathering there would
re-open the putty gap at the wisp roots) — which also removes the
support-cut chroma seams (comp chroma_seam 0.49-0.69 → 0.13-0.23 at
az+22.5/+70, the fix1→fix4 iteration). NOT a paint-over: beyond its
support the mechanism now does nothing, and the surface keeps the
composite's witnessed state.

**B. Trace commit: RIM FEATHER** (`texturing.commit_trace_deposits`,
parameter `rim_feather_texels = 3`). After each commit (blob + swept
islands), texels within 3 texels carrying the SAME evidence class
(direct, winner ≤ trace_w50, bright ball context, outside the film
commit) blend toward the ring-anchor inverse-square tone with
distance-decayed alpha, ONE-SIDED (only darker-than-target texels move —
feature edges and legitimate dark surround are structurally untouchable;
confident-witness rim texels are excluded by the weight gate). One
measured subtlety: the interpolation anchors must EXCLUDE the feather
band — a rim mixture bright enough to be a ring anchor otherwise
dominates its own target (distance ~0) and pins its darkness in place
(measured 483 → 2678 feathered texels after exclusion; the residual
outline visibly dissolves at the m22 site).

**C. Completion tone match** (`texturing.tone_match_completion_components`,
new; called in the mirror-completion block of `bake_projection_texture`,
multi-view bakes only — same scoping precedent as strand_comb /
tone_bottom_cap, so single-photo canaries are structurally untouched).
The missing gradient-domain handoff for completion tone: PURE-BRIGHT
mirror components against BRIGHT destination rings take one
component-level log-median gain (clamped ±0.25, detail verbatim: 4172
texels on the face bake). Mixed-material copies and dark-ring components
stay verbatim — measured in the ladder (fix5/fix6): rescaling them
re-classifies their own dark micro-content and mints dark_debris
0.0031-0.0036 vs the 0.003 gate, in BOTH gain directions.

## 3. MEASUREMENTS (the full ladder is in /tmp/c6/, every arm a 2048 canonical bake)

| arm | comp front | comp battery | strokes (2 probe boxes) | glyph probe | m22 probe |
|---|---|---|---|---|---|
| c5 published | 0.7022 / 13.86 | PASS | 4281 tx | 36 | 2126 |
| fix1 (A hard cut) | 0.700 / 14.9 | FAIL 4 (3 chroma seams at the cut + gate edge) | 3891 | 31 | 1857 |
| fix3 (A2 field-ops only + C hole-fill) | — | — | 4180 | 9 | 2100 (apron regression, C reverted) |
| fix4 (A + stamp feather + B v1) | 0.7020 / 14.93 | PASS 0 | 3854 | 21 | 1837 |
| fix5 (…+ D unrestricted) | 0.704 / 14.88 | FAIL 1 (dark_debris az-35 0.0031) | 3866 | 21 | 1835 |
| fix6/7 (…+ D bright-ring; floor no-op) | 0.7091 / 14.7 | FAIL 2 (dark_debris az-22.5 ×2) | 3843 | 21 | 1835 |
| fix8/9 (…+ D pure-bright) | 0.7036 / 14.89 | PASS 0 | 3866 | 21 | 1835 |
| **repo final (B with anchor exclusion)** | **0.7037 / 14.91** | **PASS 0** | **3474** | **21** | **1538** |

- **The two probe sites at the critic's own framings** (his crop specs
  re-run: `before_after_critic_frames.png`): the az0 glyph is GONE at 5x
  and 6x (smooth-skin hp p2 in the glyph box −6.7 → −1.6 /255); the
  az-22.5 closed contour is GONE at 4x and 5x (p2 −4.6 → −2.7 /255) —
  what remains there is a soft wide tone valley (the uncommitted
  witnessed deposit + blend-content boundary), not line-art.
- **48-view paired sweep** (`sweep48_contours.py`, candidate vs c5
  published, same detector): closed 163.7k stroke-texels, new 94.4k —
  the per-texel counts move with ANY tone change; every one of the four
  largest "new" components was visually inspected
  (`sweep48_new_worst.png`) and each site is CLEANER in the c6 render
  (the detector re-fires on the dark side of now-softer boundaries at
  shifted positions). No new line-art was found at any of the 48 views.
- **Ablation difference maps**: `ablation_sheet.png` (published vs
  film-off vs trace-off at both probes) — the glyph disappears with the
  film repaint off; the m22 crisp outline disappears with the trace
  commit off (leaving the soft uncommitted deposit) — closing the
  attribution loop in both directions.

## 4. HONEST LIMITS + refusals kept

- **Comp MAE 14.91 vs published 13.86** (gate 15.0, margin 0.09): the
  removed film clamp had been darkening the neck/chest toward its
  (unsupported) envelope — tone the gate partially credited. The SSIM
  side IMPROVED (+0.0015, margin +0.0037 over the floor). The gate holds
  on the exact published bytes, bit-deterministically. If the tree must
  buy MAE back, the honest lever is the shadow-apron mechanism's
  geography (its component floors), not the film field.
- **A soft tone valley remains at the m22 site** (p2 −2.7/255 on smooth
  skin): it is the witnessed blend content (front-photo displaced
  content + region tone patches) that the trace commit's own
  whole-neighborhood residue rule REFUSES to sweep (gate log
  `bake_dbgres.log`: refused by residue islands that are large or inside
  the shadow-apron's feature-contrast halo). I did not override the
  refusal ledger: it exists to prevent the partial-cleanup unmasking
  class, and the residue is below line-art contrast at 4x.
- **Apron-interior stripes** (11% of the original stroke set, hp p50
  −0.024 inside the c5-accepted shadow gradient): my one attempt
  (fill_holes on merged fragments, fix3) made the apron stronger and
  its edge crisper — REVERTED. The honest in-vocabulary fix is
  smoothing the kept-mask's internal weight dips without growing the
  domain; left for the apron mechanism's owner with the evidence filed
  (`apron_check.py`, `apron_overlay_az0.png`) — at the critic's crop
  framings these read as part of the accepted shadow's texture.
- **The fringe stage re-judges the rendered texture** (C4-3's coupling
  note): my upstream changes shift its stamp decisions; all detectors
  are green on the final tree, but the coupling remains a shared-tree
  fact.
- Compute: the three mechanisms add < 10 s combined at 2048; canary
  bakes pay nothing (structural no-ops / multi-view gates, verified
  bit-identical).

## 5. PATCHES + TESTS (all in repo; suite 235 passed, 2 xfailed)

- `src/abstract3d/film_band_gradient.py`: `FIELD_SUPPORT_TRANSITIONS`,
  `STAMP_BORDER_FEATHER_TEXELS` + the support field, shell bound,
  support feather, stamp border feather; stats key
  `field_support_transitions`.
- `src/abstract3d/texturing.py`: `commit_trace_deposits` rim feather
  (`rim_feather_texels`, stats `rim_feathered`, docstring block);
  `tone_match_completion_components` (new) + multi-view-gated call in
  the mirror-completion block (stats `tone_matched_texels` in
  `symmetry_completion`).
- `tests/test_film_band_gradient.py`: +2
  (`test_repaint_field_support_bound_refuses_far_treatment`,
  `test_stamp_border_feather_ramps_into_untreated_surface`).
- `tests/test_texturing.py`: +2
  (`test_commit_trace_deposits_rim_feather_closes_border_mixtures` —
  incl. the one-sided/witness-gate protections;
  `test_tone_match_completion_components_scopes_and_matches` — matched /
  mixed-refused / dark-ring-refused).
- `CHANGELOG.md`: one FACE-22 entry with the measured rationale.
- `docs/KnowledgeBase.md`: +3 insights (ratio-field support bounds;
  commit rims and the anchor self-exclusion; completion tone handoff
  under gradient compositing), nothing removed.

## 6. ARTIFACTS INDEX (/tmp/c6/)

- **Published bundle**: `artifacts/validation/iter3-multiview-fixed/face-2mv`
  (texture md5 `928705f3edfc9036348c12bf34435d9d`; publication block with
  the full verification). Staging + harness runs: `publish_staging/`,
  `qc_staged*`, `qa_staged*`, `tq_staged*`, re-verified in place:
  `qc_published*`, `qa_published*`, `tq_published*`.
- **Provenance**: `capstages6.py` + `stages6_2048.npz` (per-stage
  captures; instrumented bake md5 == published `4d21c7ea`),
  `capinternal*.py` + `film_internal.npz` / `film_fields.npz` /
  `film_inputs.npz` / `shadow_internal.npz` / `trace_internal.npz`,
  `locate_defect*.py`, `attrib_2048.py` (birth-stage table),
  `attrib_signed.py`, `attrib_film.py` (sub-mechanism table),
  `class_map.py` → `classmap_az0/azm22.png`, `field_analysis.py`
  (d_mass distributions), `domain_viz.py` → `film_domain.png`,
  `border_geometry.py`, `apron_check.py`, `wash_prov.py`,
  `class_dedupe.py`, gate logs `bake_dbg*.log` (commit refusal ladder).
- **Ablations**: `ablate_film.py`, `ablate_trace.py`,
  `bundle_ablfilm2048`, `bundle_abltrace2048`, `ablation_sheet.png`.
- **Fix ladder**: `proto_fix*.py` (v1-v8), `bundle_fix*`, `qc_fix*`,
  sheets `fix1_sheet.png` … `fix8_sheet.png`, `stroke_delta.py`,
  `fix8_m22_classes.png`, `debug_commit*.py`, `debug_residue.py`.
- **Final evidence**: `bundle_repo1/2/3` (determinism triplet, one md5),
  `before_after_critic_frames.png` (the critic's own crop specs),
  `compare_strokes.py` (probe counters), `sweep48_contours.py` +
  `sweep48_new_worst.png` (48-view paired sweep), `qa_repo1*`,
  `qc_repo1*`, `tq_repo1*`; canaries `/tmp/c5/bundle_c6ship`,
  `/tmp/c5/bundle_c6owl` (md5s == on-disk).

## 7. REPRO

```bash
source .venv/bin/activate
python /tmp/c6/bake.py check --res 2048          # canonical recipe -> md5 928705f3...
python /tmp/c2d/qa_shadecomp.py artifacts/validation/iter3-multiview-fixed/face-2mv --out /tmp/out_comp --shading-comp
python /tmp/verdict1/qa.py artifacts/validation/iter3-multiview-fixed/face-2mv --out /tmp/out_raw
python scripts/texture_qa.py artifacts/validation/iter3-multiview-fixed/face-2mv
python /tmp/c5/bake_asset.py ship shipcheck --res 2048   # b8e2b0d4...
python /tmp/c5/bake_asset.py owl owlcheck --res 2048     # ff746509...
python -m pytest tests/ -q                       # 235 passed, 2 xfailed
```
