# SOLVER 3 — CYCLE 3, ORDER 4: SHIP-03 "nose melt" — FIXED (root cause: source-frame registration, not stretch demotion)

Repo: `/Users/albou/abstract3d`. Verdict-ready bundles: `/tmp/c3_3/final2/ship_{1024,2048}`
(ship), `/tmp/c3_3/final3/{owl,face}_{1024,2048}` (canaries, final tree).
Patches: `src/abstract3d/texturing.py` (+`projected_frame_center_px`,
`recenter_to_canonical_frame(center_px=...)`, override-pose frame
registration, fill-detail amplitude floor, `source_registration` stats),
`src/abstract3d/backends/hunyuan3d_runtime.py` (metadata copy),
`scripts/texture_qa.py` (frame-faithful visibility reconstruction).
Tests: 5 new in `tests/test_texturing.py`; full suite 197 passed + 1 xfail.
CHANGELOG.md + docs/KnowledgeBase.md updated (2 new insights, 1 scope rule).

## TL;DR

The melt was never a demotion-curve failure. The canonical recenter
centers the PHOTO's alpha-bbox at the frame center; the orthographic
projector centers the WORLD ORIGIN. At the ship's overridden source pose
(az+30/el+15) the mesh's camera-plane bbox center projects **(+54, −28) px**
off the frame center at 1024 — so every photo sample landed ~54 px off the
surface that imaged it. On the broadside hull the offset mostly slides
content along the hull (looks plausible); at the prow — surface turning
away, strong content gradients at the silhouette — it drags dark
under-hull/background-adjacent pixels onto the nose and smears rim content
across the concavity: the "melt". Registering the photo to the projector's
frame (deterministic, mesh+pose only, no content search) fixes it:
source-pose fidelity **MAE 45.5 → 18.1, SSIM 0.092 → 0.600**, and the
head-on nose goes from molten streaks to a readable intake/grill with
panel structure. The prescribed stretch-demotion lever was measured and is
reported below as an honest negative result. texture_qa: **ship PASS
13/13 at 1024 AND 2048**; owl PASS 13/13 both res; face PASS 13/13 both
res with verdict1 failure set unchanged (2 pre-existing identity gates).

Money shot: `/tmp/c3_3/FINAL_ship_before_after.png` (shipped ledger bundle
vs fixed, az0 / el−20 / az15, 2x+4x).

## 1. Provenance (which stage produces the melt, with numbers)

Instrumented bake at 1024 (`provenance_bake.py` → `instr_1024/`), az0-el0
render unprojected to texels via a texel-ID pass (`provenance2.py`):

- The az0 4x nose crop is **mostly unwitnessed surface**: of its on-surface
  pixels, ~10.5% map to photo-observed texels, ~1% to mirror copies, ~44%
  to harmonic fill, the rest to chart-gap texels colored by edge bleed
  (the tip's UV charts are heavily fragmented).
- The DARK melt pixels: 13,957 fill vs 2,335 observed vs 300 mirror. The
  dark observed texels carry near-zero confidence — **weight p50 0.005**
  (facing p50 0.35, stretch p50 1.83, 40% > 2) — but in a single-view bake
  ANY weight > 0 is coverage truth: they anchor the fill with full
  authority (`fill = 44%` of the melt) and the fill-floor's dark-evidence
  exemption then correctly tone-tracks them (fill lum 66 vs anchor lum 72,
  within the 1.35 headroom), preserving the streaks.
- The stretch machinery (my cycle-1 Jacobian confidence) had already
  demoted these claims — geometry factor p50 0.20–0.25 at the nose — but
  weight demotion is inert for single-view fill anchoring: nothing
  downstream distinguishes weight 0.005 from 0.99 once coverage is binary.
  That is the precise sense in which the order's "under-demotion" suspicion
  was true-but-not-actionable: steeper curves change nothing until they
  vacate coverage, and vacating measured worse (§3).
- The geometry itself is crisp (untextured renders,
  `geom_renders/geom_lit_az+0_el+0.png`): rectangular prow frames, inset
  panels, grill bumps. Pure texture defect.
- **The registration smoking gun** (`cutoff_analysis.py`, silhouette
  check): projecting the mesh at the bake's camera vs the recentered
  photo's alpha — IoU 0.675 as-is; best IoU 0.805 at shift (−63, +6) px,
  matching the predicted bbox-center offset (−54 in x; the y landscape is
  flat). Per-asset offsets: ship az+30/el+15 = (+54.2, −27.9) px; face
  az+20/el+8 = (+16.2, +8.5); owl az0 = (−1.2, +0.9).

## 2. The ceiling experiment (perfect checker at the same witness geometry)

`ceiling_checker.py` / `ceiling_regfix.py`: triplanar 0.025·diag checker
painted into the atlas (GT), rendered from az+30/el+15 with the projector's
own ortho math (zero registration/reconstruction error by construction),
baked through the unmodified pipeline, compared texel-exact per stretch band
(`checker_survival*.json`):

| stretch band | binary agreement, OLD registration | agreement, FIXED |
|---|---|---|
| 1.0–1.25 | 0.44 | 0.63 |
| 1.25–1.5 | 0.43 | **0.72** |
| 1.5–2.5 | 0.44–0.47 | 0.61–0.64 |
| 2.5–3 | 0.44 | 0.57 |
| 3–4 | 0.43 | **0.71** |
| 4–6 | 0.44 | 0.73 |

(0.5 = chance.) Reading: with the old registration even PERFECT content at
NOMINAL sampling decorrelates to chance — the offset, not the witness
geometry, destroyed content. With registration fixed, witnessed content
survives structured at every stretch the demotion curve currently passes
(the ≥3 bands that survive the p=2 curve + speckle filter still carry 0.71
agreement). So: **no capture-side ceiling for the witnessed sector** — the
prow-on second photo the order contemplated is NOT required for the melt.
What remains capture-limited is the UNWITNESSED part of the nose (the
~44% fill): statistical material only, same accepted content limit as
SHIP-01/04/07 (remedy there stays a prow-on or ±45 photo).

## 3. The prescribed lever, measured honestly (negative result)

Hard vacate tail on the stretch demotion curve (factor = 0 beyond a
cutoff), single-view ship, before the registration fix (`bake_vacate.py`):

| cutoff | coverage | vacated | src-pose MAE/SSIM | melt at az0 | texture_qa 1024 |
|---|---|---|---|---|---|
| none (base) | 0.179 | — | 45.5 / 0.092 | molten | PASS 13/13 |
| >2.0 | 0.085 | 52% | 44.7 / 0.093 | reduced streaks, still molten | PASS 13/13 |
| >3.0 | 0.163 | 11% | 45.4 / 0.092 | ~unchanged | PASS 13/13 |
| >4.5 | 0.178 | 1.5% | 45.5 / 0.092 | unchanged | PASS 13/13 |

On top of the registration fix: cutoff 2.0 costs MAE 18.1 → 24.3 and SSIM
0.600 → 0.436 (surrenders half the witnessed hull to fill) for no visible
nose gain; cutoff 3.0 changes nothing visible. **Not landed.** The existing
p=2 curve plus the registration fix is the measured optimum; the checker
table (§2) confirms the curve's survivors carry real content once
registration is right.

## 4. What landed in the repo

1. **Projector-frame registration for overridden poses**
   (`projected_frame_center_px`, `recenter_to_canonical_frame(center_px)`).
   Deterministic mesh+pose function; no content search. SCOPE RULE
   (measured, documented in KnowledgeBase): applies to OVERRIDDEN source
   poses only — an ESTIMATED pose (gradient_ncc) was searched against the
   legacy-centered photo, so pose and frame are co-adapted; forcing the
   projector frame under the estimated pose broke the face (verdict1
   failures 2 → 10, front SSIM 0.630 → 0.598, doubled features at az0).
   Registering the estimator itself to the projector frame is future work
   for the face lane. At the canonical front both conventions agree to
   ~1 px by construction (owl).
2. **Metadata + harness faithfulness**: bundles record
   `source_registration` (method, dx/dy px); `texture_qa.py` reconstructs
   per-view visibility from the same frame (absent key = legacy — old
   bundles unaffected). Without this the harness's region attribution
   shifts by exactly the offset (measured: observed-cellular 0.051 vs
   0.046 on identical textures).
3. **Fill-detail amplitude floor** (`synthesize_fill_detail`): transferred
   amplitude floored at the observed population's p25 RAW-residual
   amplitude. Grazing-smeared donors carry artificially quiet statistics;
   fill anchored by them shipped as literal flat plateaus with straight
   chart-edge borders — exposed at 2048 as texel.facet_cellular 0.092 vs
   0.091 (an 11k-texel flat cell). With the floor: 0.012 vs 0.092, fill
   energy 0.615 → 0.620, sigma guard + granite test untouched. The RAW
   quantile (not the smoothed amplitude field) is load-bearing: the
   smoothed field spreads sparse line energy over flat texels and its
   quantile would inject line-level noise (caught by the existing granite
   regression test during development).

Tests added (all deterministic, no GL): `projected_frame_center_px`
exactness vs the projector's own math + canonical-front no-op;
`center_px` placement + legacy default; end-to-end miniature SHIP-03
(off-origin block, synthetic photo in projector frame, two-tone recovery
through the full bake); amplitude-floor plateau break + no-granite bound.

## 5. Acceptance evidence

- **Head-on crops**: `/tmp/c3_3/FINAL_ship_before_after.png` — at az0
  2x/4x the fixed nose shows the intake grill's internal structure, the
  bright rim frame, and panel edges above (readable structure at the
  witnessed sector); el−20 and az15 rows show the under-nose streaking
  replaced by hull-consistent material. Residual: the unwitnessed fill
  sector reads as granular material, not panels — accepted SHIP-01/04
  content limit, remedy = additional photo.
- **texture_qa (final repo code)**:
  - ship_1024: **PASS 13/13** (energy 0.600, seams p95 23.8, smears 0)
  - ship_2048: **PASS 13/13** (energy 0.620, facets 0.012 vs 0.092,
    smears 0)
  - owl_1024 / owl_2048: **PASS 13/13** both (code path identical to
    tree baseline for estimated/declined poses; rebaked + verified)
  - face_1024 / face_2048: **PASS 13/13** both
- **verdict1 face battery** (2048): failure set identical to the
  tree-baseline control baked the same hour (`final2/face_legacy_2048`):
  2 failures — identity[front] SSIM 0.630 < 0.70 and mean|RGB| 22.2 >
  22.0 — the pre-existing FACE-14 gate owned by orders 1/5. All detector
  classes green (eyes 1/1 at ±90, dark ≤ 0.0007, crown ≤ 0.0007, no
  3-blob, seams ≤ 0.24). Bit-identity note: face/owl take the legacy
  code path by construction after the scope rule; the md5s of the final3
  face texture vs the legacy control confirm the path equality.
- **Repro**: `python /tmp/c3_3/bake_base.py OUT ship 1024|2048` (uses
  `remove_background_robust` + `bake_projection_texture(...,
  texture_completion="auto", projection_model="orthographic",
  source_pose_override=(30,15))`), then
  `python scripts/texture_qa.py OUT/ship_RES`.

## 6. Honest limits

- The nose's unwitnessed sector remains statistical material (no panel
  content from one photo) — accepted P3-class limit; a prow-on or ±45 el
  photo remains the content remedy. The MELT (structured-content
  destruction at the witnessed sector) is fixed.
- The scope rule leaves ESTIMATED-pose bakes (face) on the legacy frame:
  their ~16 px offset is real but co-compensated by the pose estimator;
  fixing it properly requires estimating against projector-frame renders
  — flagged for the face lane, not smuggled in here.
- texture_qa's visibility reconstruction still over-attributes ~14% of
  observed texels at rims (qa 0.211 vs bake 0.187 coverage) — inherent
  to reconstructing without the bake's speckle/feather demotions; my
  metadata change removed the 54 px component, the residual is
  rim-local. Emitting bake masks with a containment check is a possible
  cycle-4 harness upgrade (governance question, not smuggled in).
- The identical-hash claim for the ship at 1024 across the scope
  refactor was verified by rebake (md5-equal textures final vs final2);
  the 2048 pair differs from the pre-amp-floor bundle exactly at the
  floored fill texels, as intended.
