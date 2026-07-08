# SOLVER D (cycle 2) — front-identity gap decomposition + ORDER 8 (elf-ear texture half)

Repo: `/Users/albou/abstract3d`, tree as of 2026-07-05 ~23:00 (texturing.py
untracked/shared; no texturing edits shipped by me — see D1c). All scripts,
bundles, crops, JSON under `/tmp/c2d/`. Harness = `/tmp/verdict1/qa.py`
exact protocol (render at declared pose az+20 el+8 through
`abstract3d.rendering.render_mesh_views` @896, alpha-bbox + NCC-refined
similarity registration, 5+11 px erosions, gray SSIM win11 + RGB MAE over
the joint mask).

Baselines I baked and measured (standard repro: front az0 source + ±90
clean profiles, `texture_completion="auto"`, `projection_model=
"orthographic"`, **pose pinned +20/+8** via `source_pose_override` — the
free estimator now rejects (NCC 0.0052 < floor) and pins az0, which is the
pose lane's problem, not mine):

| bundle | verdict1 | identity[front] | texture_qa |
|---|---|---|---|
| shipped `face-2mv` (T0 asset) | FAIL 8 | **0.632 / 22.07** | PASS 0/13 |
| `bundle_pin20` (fresh 2048, pinned) | FAIL 8 (`qa_pin20.log`) | 0.636 / 22.95 | PASS 0/13 (`tqa_pin20`) |
| `bundle_pin20_1024` (fresh 1024, pinned) | FAIL 11 (`qa_pin20_1024.log`) | 0.639 / 21.3 | PASS 0/13 |

The 8 fails on pin20 = the pre-existing family (az−90 eyes ×2 [FACE-15,
ORDER 5 lane], −22.5/−35/−45 el10 eye undercounts + −22.5 debris
[FACE-03 family], identity[front] ×2). Nothing new introduced by anything
I ran; I ship no pipeline change (measured reasons in D1c).

---

## D1. The identity gap, decomposed exactly

### D1.1 Per-region table (gate protocol @896; regions from the harness's
own color classifier: face hull core / hairline mixture band (hull-boundary
∩ hair-dilation, half-width 4.5% bbox) / hair mass / background-adjacent
rim / rest)

Shipped face-2mv, RAW gate (total **0.632 / 22.07**, bias = render−photo
luminance):

| region | share | SSIM | MAE | lum bias | region→perfect ⇒ total |
|---|---|---|---|---|---|
| face_core | 62.8% | 0.692 | 22.7 | **−12.3** | 0.826 / 7.8 |
| hairline band | 18.1% | 0.516 | 26.5 | +15.5 | **0.720 / 17.3** |
| hair | 11.1% | 0.522 | 12.9 | +0.1 | 0.685 / 20.6 |
| rim | 4.8% | 0.414 | 24.1 | +9.7 | 0.660 / 20.9 |
| other | 3.3% | 0.819 | 13.9 | −9.9 | 0.638 / 21.6 |

Fresh `bundle_pin20` reads the same shape (face_core 0.723/23.0 bias −14.6;
hairline 0.475/30.8; totals 0.636/22.95). At a 2048-px gate render the same
shipped bundle scores **0.719/22.15** (SSIM gate would pass at that render
size — the 0.63 SSIM number lives at 896 where win11 spans coarser
structure; measurement note for the verdict owner, `d1_2048.log`).

Attribution of the three lanes named in my brief:

- **(a) hairline band** — 18% share at SSIM 0.52 / MAE 26.5, bias +15.5
  (beige film brighter than the photo's hair). Band→perfect alone lifts the
  gate to 0.720/17.3 = PASS raw, margin 0.02. This is solver A's ORDER 4
  lane (their `/tmp/c2a/NOTES.md` M1-M3 mechanisms match my numbers: the
  band is front-painted mixture + zone-core fill). Not duplicated by me.
- **(b) renderer shading** — quantified below: **5.2 MAE units + 0.023
  SSIM** of the total gap at the gate; face-core bias −12.3 → +0.7 after
  compensation. Bigger than the earlier "~3 MAE" estimate.
- **(c) genuine albedo error in face_core** — the worst-window census
  (`d1/pin20_core/worst_windows.png`, provenance in §D1.4): the ear
  cluster (3 of the top 5 windows), under-eye flake fields, temple film
  (A's), nose-wing/mouth-corner ghosts. Mechanisms measured in §D1.4;
  none of my candidate fixes was a net win (§D1.5).

### D1.2 (b) exactly: the renderer dims the gate systematically

The repo renderer shades textured meshes with `shade = 0.88 + 0.12*diffuse`
(world-fixed light `(0.45,−0.35,0.82)`, `rendering.py`). The gate compares
`albedo*shade` vs the photo. Measured at the gate's exact conditions
(`d1_shading_floor.py`):

- shade over the compared mask: p5 0.878 / p50 0.906 / p95 0.984;
- **a PERFECT texture scores SSIM 0.977 / MAE 11.45** (face hull floor
  15.2 MAE, hair 2.9) — half the 22.0 MAE budget is consumed by the
  renderer's own shading before any texture defect is counted;
- the term is multiplicative and texture-independent: it shifts every
  candidate equally; it is measurement bias, not albedo signal. A texture
  hack (brightening albedo to beat the raw gate) would ship a wrong
  texture to every other viewer — rejected by construction.

**Proposed calibrated correction IN THE HARNESS** (deliverables:
`/tmp/c2d/qa_shading_patch.diff`, runnable `/tmp/c2d/qa_shadecomp.py`,
opt-in `--shading-comp`):
render the same mesh with a pure-white texture at the same pose = the
exact per-pixel shade field through the identical shader + resampling
chain; multiply the REGISTERED photo by it before scoring. Photo-side
multiplication (not render division) avoids amplifying quantization noise;
the NCC alignment search is untouched (mean/variance-normalized, measured
insensitive), so registration is bit-identical to the raw gate;
photo-self calibration untouched. MAE budgets re-tightened so strictness
is preserved: front 22.0→15.0, sides 30.0→24.0 (raw budget minus the
measured floor, keeping the same relative margin vs the input ceiling).

Measured populations (896, gate protocol):

| bundle | RAW ssim/mae | COMP ssim/mae |
|---|---|---|
| perfect texture (floor) | 0.977 / 11.45 | ~1.0 / ~0 |
| front-only pinned bake (input ceiling, §D1.6) | 0.719 / 16.55 | 0.744 / 8.34 |
| shipped face-2mv | 0.632 / 22.07 | 0.656 / 16.83 |
| fresh pin20 | 0.636 / 22.95 | 0.663 / 16.45 (full run `qa_pin20_comp.log`: side_left 0.669/14.9, side_right 0.707/15.1, all localMin improve) |
| pin20 + band-perfect counterfactual | 0.730 / 17.43 | 0.754 / 11.25 |

Ordering between bundles is preserved (deltas <0.005 SSIM); the
systematic −11..−15 luminance bias on the face hull cancels to ~+1.
Under COMP the combined path (my (b) + A's band) passes with real margin
(0.754/11.25 vs 0.70/15.0); under the RAW gate the same path passes at
0.730/17.4 with a 0.02 SSIM margin that any regression eats. Insight
recorded in `docs/KnowledgeBase.md` ("Identity gates that compare shaded
renders to photos carry a perfect-texture floor").

### D1.3 Provenance decomposition (what CONTENT drags the gate)

Gate pixels classified by their texel's blend-time winner
(`d1_provenance_decompose.py`, pin20):

| class | share | SSIM | MAE | bias | class→perfect ⇒ total |
|---|---|---|---|---|---|
| front-won | 40.2% | 0.733 | 17.8 | −10.9 | 0.743 / 15.8 |
| side_left-won | 23.2% | 0.533 | 32.3 | **−15.5** | **0.744 / 15.5** |
| side_right-won | 10.1% | 0.625 | 24.2 | +9.7 | 0.674 / 20.5 |
| fill | 25.6% | 0.591 | 21.3 | −6.4 | 0.740 / 17.5 |
| mirror | 0.8% | 0.291 | 50.2 | +15.3 | 0.641 / 22.5 |

Two readings: (1) front-won content is fine (0.733 incl. shading dim);
the drag is *side_left-won + fill* pixels — the temple band, curtain
edges, ear, jaw — content the front photo witnesses differently at the
gate pose. (2) side_left's −15.5 bias exceeds the shading −10.9: the
residual ≈ −4.6 is the left profile's own baked lighting on its exclusive
territory, which `delight_projections`' overlap-proximity fade
*deliberately* preserves (protecting identity[side_left] at ±90, which
passes 0.639/21.5 with 8.5 MAE margin). Rebalancing that fade could buy
the front gate ~1 MAE at the side gates' expense — delight-owner's lever
(ORDER 2 lane), documented here with the numbers, not touched by me.

### D1.4 Worst-window census (face_core damage, pin20, gate protocol)

Top non-overlapping 49-px windows + per-window texel provenance
(`d1_window_attrib.py`; crops `d1/pin20_core/worst_windows.png`):

| # | (x,y)@896 | SSIM | MAE | attribution (measured) |
|---|---|---|---|---|
| 0 | (589,490) ear | 0.05 | 59 | 71% side_left w~0.77 skin (photo-faithful ear content, parallax-displaced vs front photo) + 25% fill; 9% mirror |
| 1 | (540,594) below-ear "tape" | 0.11 | 43 | 100% side_left w~0.51 skin — side's jaw skin lands where the front photo shows the hair-curtain edge (geometry/parallax) |
| 2 | (458,405) under-eye R | 0.20 | 38 | 84% front w~0.49, sampled skin 0.62/hair 0.26 — front's own lash/skin mixture (FACE-03) |
| 3 | (503,335) temple film | 0.28 | 28 | 47% fill + 31% front + 21% side_left — A's band (FACE-01) |
| 4 | (622,440) sideburn/curtain | 0.28 | 37 | 83% side_left w~0.36 skin vs front's hair curtain (FACE-10 family) |
| 5 | (318,407) temple L | 0.30 | 36 | 80% front at w p50 **0.042** mixture stamps — the ε-weight class (A's M1) |
| 7 | (589,703) jaw curtain | 0.35 | 62 | 100% side_left w~0.53 sampled HAIR 0.72 — curtain displaced vs front photo (parallax) |
| 8 | (490,621) mouth-corner | 0.47 | 28 | 100% side_left w p50 0.127 (p10 0.005) skin — the ORDER-6 ε-weight ghost class |
| 9 | (555,653) chin/jaw | 0.49 | 57 | 100% side_left w~0.91 mixed — high-confidence side content vs front photo |

The ear cluster (#0/#1/#4 + ring) carries the largest coherent share:
ear+41px ring = 4.45% of compared pixels at SSIM 0.31 / MAE 47;
ear-ring→perfect ⇒ total 0.666/20.85 (`d1_ear_counterfactual.py`).

### D1.5 (c) fixes attempted — measured, none shipped (all replayed
against identical captured projections; `replay.py` reproduces the full
blend→floor chain bit-close, |Δ| mean 0.5/255)

| variant | mechanism | identity Δ | side effects | verdict |
|---|---|---|---|---|
| `demote1` | ORDER-6 as specified: reference-won texels w<0.2 zeroed where source-confident 3D ball dominates (4.1k texels) | +0.0005 / +0.03 | none visible | no-op on tonight's stack — the mouth-corner ghost family now sits at w 0.13-0.9 (reference_flow-era), not 0.006-0.2 |
| `srcanchor1` | cap ref weight ≤0.6·src where src≥0.3 (5.1k texels) | +0.001 / −0.04 | none | no-op |
| `matdem` | demote refs at YCrCb skin↔hair class conflict with the source's 0.2-facing sample (5.0k) | +0.002 / −0.07 | az0 dark_debris 0.0023→0.0035 (NEW fail), pale up | trade — rejected |
| `matconf1` | matdem + front grazing-band witness on uncovered texels | +0.006 / −0.46 | dark_debris fails at 5 views (0.0043-0.0061) | trade — rejected |
| `grzfill` | front 0.2-facing samples as fill-only witness (10.9k texels) | +0.002 / −0.08 | dark_debris fails at 6 views | trade — rejected |
| `src02` | drop the multiview source facing cliff 0.4→0.2 outright | **+0.011 / −0.67** | chroma_seam FAIL −22.5 ×2, el10 eye losses persist, pale up | the identity gate loves front content; every other view pays — rejected (this cliff is load-bearing, as its in-code comment claims) |
| `flake` | post-blend low-confidence-vs-consensus flake drop (9.1k) | **−0.017** / −0.30 | eye loss at −22.5 el0 | destroys valid low-weight content — same failure class as solver4's blanket demotion; rejected |

Conclusion: the ε-weight debris lever (ORDER 6) is already spent on this
tree (reference_flow + conflict resolution moved the ghost families into
w 0.1-0.9 or into fill), and every content-addition/demotion variant
trades identity points for debris/seam regressions elsewhere. Zero-defect
means no trades: **no texturing.py change shipped from this lane.** The
remaining face_core damage is: A's band (#3/#5), parallax between
profile-painted surfaces and the front photo (#0/#1/#4/#7/#9 — geometry-
bound, see ceiling), front's own mixture stamps at the lash line (#2,
same fused-band class as A's M1), and shading (b).

### D1.6 Ceiling experiment (claimed, therefore run)

`bundle_frontonly` = pinned bake from the front photo alone,
`texture_completion="none"`, `fill_detail_gain=0` — the texture holds the
photo's own pixels wherever the photo sees the surface; nothing can beat
it there from these inputs. Gate protocol results (`d1_ceiling.py`):

| measurement | RAW | shading-COMP |
|---|---|---|
| front-witnessed pixels only (72% of mask), front-only bundle | **0.759 / 16.05** | **0.786 / 6.70** |
| same pixels, real pipeline (pin20) | 0.711 / 19.75 | 0.741 / 12.02 |
| full mask, front-only bundle | 0.719 / 16.55 | 0.744 / 8.34 |
| full mask, pin20 | 0.636 / 22.95 | 0.663 / 16.45 |

Honest ceiling statement: (1) on the pixels the front photo witnesses,
the pipeline sits 0.045-0.048 SSIM below its own input ceiling — that
residual is projection/gradient-composite/registration wear, worth ~+0.03
total if fully recovered; (2) the other 28% of the mask can never reach
the witnessed region's quality (the front photo does not see it — profile
parallax and fill live there); the front-only probe fills it with smooth
front-anchored diffusion and still only reaches 0.719 total RAW. So
**0.70 RAW SSIM at 896 is reachable from these inputs only if the
hairline band is fixed (A's lane; counterfactual 0.730) and nothing else
regresses — a ≤0.03 margin; with the shading compensation accepted the
same state passes at 0.754/11.3 with real margin.** MAE ≤22 RAW is
reachable the same way (17.4); under COMP the budget is 15.0 and the
band-fixed state reads 11.3.

---

## D2. ORDER 8 — elf-ear texture half

### D2.1 The specified defect does not reproduce at the apex on this tree

Claim under test (RULING #8 / solve4 G2 hand-off): "the ear apex painted
with skin where all three photos show hair."

Witness-assignment audit at the apex, instrumented pinned bake
(`d2_apex_provenance.py`, apex = top-30%-z of solve4's geometric ear
component, d<0.02, ~1.5k texels/side @2048):

| side | final texture | front | mirror | profile (only real witness) | fill share |
|---|---|---|---|---|---|
| left | **skin 0.02 / hair 0.97** (lum p50 22) | cover 2%, sampled 100% hair | 0% | side_left cover 37%, sampled **hair 0.97 / skin 0.03** | 51% (hair-toned) |
| right | **skin 0.05 / hair 0.93** (lum p50 20) | cover 0% | 0% | side_right cover 43%, sampled **hair 0.96 / skin 0.04** | 47% (hair-toned) |

(The 2-5% skin texels are the ear-body boundary crossing the band's lower
edge; the profile photos themselves sample 3-4% skin at those texels —
paint == witness. Estimator-free az0 bake reads 0.00 skin at the apex on
both sides, same conclusion.)

Wider bands and balls (`d2_apex_final_colors.py`, `d2_apex_ball.py`):
apex balls r=0.06/0.10 around the apex vertex = 100% hair on shipped,
v14, AND fresh pinned bakes; skin+pale in the r=0.22 ball is 39-48%
profile-painted texels whose *photo content is skin* (86%) + fill
continuation — no view paints skin out of hair samples anywhere near the
ear (`profile_win_sampled_hair = 0`, `front_win_skin = 0` over the whole
d<0.05 ear region; `d2_skin_witness.py`).

Boundary check against the photos (`d2_photo_boundary.py`, canonical
recenter + the projector's own ortho row→z map): the profiles' own
skin-top sits at z=+0.358 (left) / +0.274 (right); the PAINTED skin-top
on the mesh ear stops at z≈+0.21 on both sides — the paint runs
**6.2% / 2.9% of the frame LOWER (more conservative) than the photos'
own skin/hair boundary**. Per-z-slice, final skin fraction tracks the
profile photo's sampled skin fraction almost exactly (table in
`d2_skin_witness.py` output; no slice paints skin above where the photo
shows skin).

Reconciliation with solve4's G2 numbers: their "33-38% skin-like" was
measured over the WHOLE ear component (their `g2_tip_provenance.py`
queries all `g2_ear_*_verts` at d<0.03), which includes the ear BODY that
all three photos legitimately show as skin (my equivalent: 43-46%
component skin on all bundles), and their bake was the pose-drifted
16:54-tip era. At the apex proper their own numbers agree with mine
(hair-dominant).

### D2.2 Acceptance battery (the ORDER's verification, run on the
current stack at both resolutions)

Ear crops at az {±45, ±90, ±135} × el {0,10}, apex-disc skin/hair
fractions measured in render space (`d2_ear_crops.py`; sheets
`d2_accept_left.png`, `d2_accept_right.png`; metrics
`d2_accept_metrics.json`):

- 2048: apex-disc skin ≤ 0.059 (median 0.004), hair ≥ 0.92 at all 16
  side-visible view×side combinations;
- 1024: apex-disc skin ≤ 0.087 (median 0.024), hair ≥ 0.89 (the residual
  "skin" pixels are the ear-body edge caught by the disc, visible in the
  sheets — the apex line itself is hair everywhere);
- **hair-toned apex: VERIFIED at all required views, both resolutions.**

No-regression evidence (no pipeline change was needed, so this is the
baseline state, recorded): verdict1 @2048 FAIL 8 (pre-existing family,
`qa_pin20.log`), @1024 FAIL 11 (`qa_pin20_1024.log`); texture_qa PASS
0/13 at both (`tqa_pin20*`). md5s of my bundles in the artifact index.

### D2.3 What actually remains of the "elf read" (and whose it is)

1. **Geometry**: the pointed apex silhouette — solve4's conservative-clamp
   PROVEN-LIMIT stands (apex prominence ≥0.05 vs 0.02 cap); capture-side
   remedy documented by them. The skin patch that ends in a point at
   ±22.5-45° is the *ear body* (photo-faithful content) wrapped on
   pointed geometry; paint cannot round it.
2. **FACE-07 debris on/near the ear**: pale shards = ~3% of the ear band,
   **90-96% FILL texels** inside the fully-contested (layered-zone) ear
   band (`d2_shards.py`) — mixture anchors propagated by fill, the same
   anchor class as A's M3 (their P2 hair-toned band fill is the general
   mechanism; the ear band should be inside its domain — coordination
   note left for A, not duplicated here). The below-lobe "tape" strip is
   window #1: high-weight side-photo skin vs front's curtain edge —
   parallax, geometry-bound (D1.4).
3. At az0 the "pointed pale patch against hair" pixels around the ears
   are ≥95% beige mottle ON THE HAIR CURTAIN around the ear (skin-class
   render pixels whose texels are profile-won hair samples + fill), i.e.
   FACE-01/10 band-family, not ear-apex paint (`d2_az0_patch.py`,
   winner maps `d2_az0patch_*.png`).

---

## Deliverables index (/tmp/c2d/)

- **Patches**: `qa_shading_patch.diff` (+ runnable `qa_shadecomp.py`,
  generator `qa_shading_patch.py`) — proposed identity-gate illumination
  compensation, opt-in flag, recalibrated MAE budgets, full population
  table above. Repo: `docs/KnowledgeBase.md` +1 insight (shading floor).
  No texturing.py changes (D1.5: nothing measured better without trades).
- **Bundles**: `bundle_pin20` (2048 pinned baseline + provenance),
  `bundle_pin20_1024`, `bundle_frontonly` (ceiling), `bundle_base`
  (estimator-free az0 — documents the pose gate pinning to the wrong
  default for this subject), experiment bundles `bundle_{demote1,
  srcanchor1,matdem,matconf1,grzfill,src02,flake,replay0}`.
- **Analysis**: `d1_decompose.py` (+ `d1/` region tables @896+@2048, raw
  + comp), `d1_shading_floor.py`, `d1_provenance_decompose.py`,
  `d1_window_attrib.py`, `d1_ear_counterfactual.py`, `d1_ceiling.py`,
  `d1_facecore.py`; D2: `d2_apex_provenance.py`, `d2_apex_final_colors.py`,
  `d2_apex_ball.py`, `d2_skin_witness.py`, `d2_photo_boundary.py`,
  `d2_ear_crops.py`, `d2_az0_patch.py`, `d2_shards.py`,
  `d2_component_coverage.py`; infra: `bake.py` (instrumented),
  `replay.py` (offline blend→floor replay, validated |Δ| 0.5/255),
  `quick_identity.py`, `crop_ab.py`.
- **Key crops**: `d1/pin20_core/worst_windows.png`,
  `d2_accept_{left,right}.png`, `d2_ab_apex_sheet.png` (shipped vs fresh),
  `d2_az0_4x_sheet.png`, `d2_ear90_4x.png`, `d2_photo_boundary_*.png`,
  `ab_ear_variants.png`, `ab_matconf1.png`.

## Repro

```bash
source .venv/bin/activate
python /tmp/c2d/bake.py pin20 --res 2048 --instrument --pin 20,8
python /tmp/verdict1/qa.py /tmp/c2d/bundle_pin20 --out /tmp/c2d/qa_pin20
python scripts/texture_qa.py /tmp/c2d/bundle_pin20            # PASS required
python /tmp/c2d/qa_shadecomp.py /tmp/c2d/bundle_pin20 --shading-comp   # (b)
python /tmp/c2d/d1_decompose.py /tmp/c2d/bundle_pin20 --tag t           # D1 table
python /tmp/c2d/d2_ear_crops.py cand=/tmp/c2d/bundle_pin20 --tag t      # D2 battery
```

## Final statement on the identity ceiling

From these three inputs, at the gate's 896-px protocol: the front-
witnessed ceiling is SSIM 0.759 / MAE 16.1 RAW (0.786 / 6.7 compensated);
the full-mask input ceiling probe reads 0.719 / 16.6 RAW. The shippable
path to the gates is therefore: **hairline-band fix (solver A) ⇒ 0.730 /
17.4 RAW (passes, 0.03 margin); + the shading compensation in the harness
⇒ 0.754 / 11.3 vs 0.70 / 15.0 (passes with margin).** My lane's measured
contributions: the compensation (+0.024 SSIM / −5.2 MAE of measurement
bias removed, fully justified, patch delivered), the decomposition that
scopes A's band as the deciding fix, and the negative results that close
off the ε-weight demotion lane (already spent) and content-addition lanes
(debris trades) for this cycle. The ear-family identity damage (#0/#1/#4:
ear ring = 4.45% of pixels at SSIM 0.31) is profile-vs-front parallax on
pointed geometry — texture-side ceiling proven by the witness audit
(every ear texel already carries its best witness's true content); the
remedy is geometric (regeneration with ear references), consistent with
solve4's PROVEN-LIMIT.

ORDER 8 texture half: the apex is hair-painted on the current stack at
declared pose (both resolutions, all required views, verified with
provenance + photo-boundary + crops above); the residual pointed read is
the geometry half plus FACE-07 fill-anchor debris whose general mechanism
belongs to the band-fill fix (A's P2) — recommendation and evidence
handed over, no duplicate mechanism landed.
