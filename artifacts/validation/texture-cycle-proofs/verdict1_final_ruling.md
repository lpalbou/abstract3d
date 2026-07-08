# VERDICT AGENT 1 — FINAL RULING: candidate_v13 vs candidate_v13raw

Date: 2026-07-05. Harness: `/tmp/verdict1/qa.py` (pose-aware identity gates,
sweep gate, 49-px local window, 28 views at 896 px, photo-calibrated
detectors). All four bundles rerun on this exact harness today; numbers below
are locked and reproducible (`results.json` under `/tmp/verdict1/qa_out/`).

| bundle | failed checks | front SSIM @ declared pose | front localMin | coverage |
|---|---|---|---|---|
| shipped iter3 (user-rejected) | **66** | 0.516 | 0.002 | 0.569 |
| v9 | 64 | 0.529 | — | 0.571 |
| v13 (flap-cleaned mesh) | **10** | 0.611 | 0.009 FAIL | 0.411 |
| v13raw (original mesh) | **9** | 0.628 | **0.054 PASS** | 0.395 |

## RULING: ship candidate = **v13raw**. Verdict: **FAIL against the PASS bar, by a small and mostly-explained margin.**

The 66→9 collapse in failed checks is real, not detector drift: the same
harness, same thresholds, same photo calibration. The front hemisphere is
transformed — one face, two eyes, one nose, one mouth at every azimuth in
{0, ±22.5, ±35, ±45} at el {0, 8, 10}. I counted doubled features by eye at
full resolution on all of those views: **zero** in both candidates (the v9
duplicate iris at −22.5/−45 is gone; the v13 harness "3 eyes at +22.5/el10"
is a temple-debris blob miscounted as an eye — see below). dark_debris,
lip_in_hair, skin_in_hair, pale_film, chroma_seam: **zero failures in both
candidates across all 28 views** (max dark_debris 0.0022 vs gate 0.003;
worst seam 0.31 vs gate 0.45).

### Why v13raw over v13 (evidence)

The two differ only in mesh: v13 got fixer3's below-hairline flap cleanup.
That cleanup **hurt the face**:

1. **Black dash fragments under both eyes at the front pose.** v13's worst
   49-px face window is 0.009 at (302,412) — hard black strokes on the lower
   lid/cheek (`qa_out/v13/evidence/identity_front_worst_window.png`,
   `final_inspect/v13_eyes_zoom.png`). v13raw has beige flakes in the same
   place instead — visibly milder — and passes the local gate (0.054).
   The dashes are newly exposed un-textured flap-cut edges: a v13-specific
   regression not present in v9/v13raw.
2. **Eye failures 5 vs 2.** v13's extra failures are profile/oblique eye
   undercounts (−90 el0/el10, −70 el10, +90 el10) plus the debris blob at
   +22.5/el10; v13raw fails only −90 el0/el10, and by eye the −90 eye is
   *present* (iris sliver + lash line + specular, slightly smudged tear
   duct: `final_inspect/v13raw_m90_eye_zoom.png`) — a detector sensitivity
   limit at grazing angle, not a destroyed feature. Severity: minor.
3. **Front identity:** v13raw 0.628 global / 0.054 local vs v13 0.611 / 0.009.
   Face-hull-only SSIM: v13raw 0.672 vs v13 0.628.

v13 wins only on crown_flakes (1 small failure vs 4). v13raw's four are
beige hair-band patches on the right-rear quadrant (−45/−70/−135:
0.0012–0.0052 of foreground, `qa_out/v13raw/evidence/az-070.0_el00_annotated.png`)
— they sit in hair away from the face and read as lighter-brown discoloration
at 1:1. Facial black dashes are worse than hair discoloration. **v13raw wins.**

## PASS-bar table for the winner (v13raw)

| # | Bar item (REPORT_V9 §T4) | Result | Verdict |
|---|---|---|---|
| 1a | Front SSIM ≥ 0.70 @ declared pose | 0.628 | **FAIL** (margin 0.072) |
| 1b | Front mean\|RGB\| ≤ 22 | 23.5 | **FAIL** (margin 1.5) |
| 1c | Worst 49-px face window ≥ 0.05, all declared views | front 0.054 PASS; side_left 0.178 PASS; side_right **−0.137 FAIL** | **FAIL** (ear etch) |
| 1d | Profile SSIM ≥ 0.55, mean\|RGB\| ≤ 30 | side_left 0.673/18.2; side_right 0.649/25.4 | PASS |
| 2 | Pose sweep peaks within ±5° of declared azimuth | peak at +0° offset | **PASS** (declared +17.5° confirmed) |
| 3 | Detector gates, all 28 views | eye_count −90 ×2 (undercount); crown_flakes ×4; all other classes clean | **FAIL** (6 checks) |
| 4 | Evidence standard (≥768 px, 1:1 crops, calibration) | this report + `final_inspect/` + `qa_out/v13raw/evidence/` | PASS |

Two measurement caveats I owe the owner on items 1a/1b, from decomposition
(not excuses — quantified): (i) the render is globally darker than the photo
by RGB (−13, −10, −8) because the repo renderer multiplies albedo by
0.88+0.12·diffuse; removing that single global offset leaves MAE 20.3 —
i.e. ~3 of the 1.5-over-gate MAE units are renderer shading, not texture.
(ii) global SSIM mixes hair with face: face-hull SSIM is 0.672, hair 0.470 —
photo hair strands can never be reproduced strand-for-strand by a baked
texture at this mesh resolution. The 0.70 bar stays (it was calibrated
including these effects, and the shipped left profile hit 0.780), but the
honest reading is: the face region is close to the bar; the deficit
concentrates in the hairline band and hair.

## Remaining defects, severity, fixability (v13raw)

| Defect | Views | Severity vs PASS bar | New regression? | Fixability |
|---|---|---|---|---|
| Painted hairline band: beige wash between forehead and hair mass, flake-edged; crown patches −45/−70/−135 | all front-hemisphere + right-rear | **Major** (obvious at 1:1: reads as painted/receding band, not hair; subtle at arm's length) | Same root cause as v9's film band, now surrendered-to-fill instead of debris — improved, different failure texture | Texture-logic: yes — the zone gate that surrendered the band could fill with hair-toned gradient sampled from adjacent strands instead of skin-beige; the band boundary needs feathering into the dark hair |
| Ear etch on right profile: pale flap-flakes + tone patches on ear geometry; black flecks at left earlobe (+45/+90) | ±90, +45 | **Major at close-up**, minor at distance; blocks bar item 1c | Same as v9 blocker #4, unchanged | Partly: exclude profile-photo hair texels from baking onto ear geometry (depth/normal disagreement is large there). Ear shape itself is mesh, see below |
| Flat back-of-head: strand texture lost, uniform chocolate wash (`final_inspect/back_triptych.png`) | 135–180° | **Major for 360°/turntable use**, cosmetic for front-facing | **NEW vs v9** — direct cost of coverage 0.57→0.40 (the harness has no back-texture gate, so this never shows in the fail count) | Texture-logic: yes — the surrendered rear zone needs strand-synthesis or mirrored-hair fill, not flat harmonic fill |
| Pointed "elf" ears silhouette | 0–45° both sides | **Major** (non-human silhouette at three-quarter views) | No — identical in shipped mesh (`final_inspect/ear_shipped_vs_v13raw.png`); Hunyuan geometry | **Inherent to mesh** — no texture fix; needs geometry regeneration or ear-region remeshing |
| Under-eye beige flakes, nose-flank pale seam column, chin streak | 0, ±22.5 | Minor (1:1 only) | Improved vs v9 (was black debris + ghosts) | Texture-logic: yes — same film-edge exclusion as hairline |
| Profile eye slightly smudged (tear-duct flakes), detector undercounts at −90 | ±90 | Minor | Improved vs v9 | Partly texture (sharpen eye-region source priority); partly detector sensitivity at grazing angles |
| Front global tone −10 RGB | front | Minor (uniform, invisible without reference) | Unchanged all versions | Renderer shading model, or bake-time albedo compensation |

Blockers ranked in REPORT_V9 → status: #1 hairline film band **largely
fixed** (45 checks → 4, zero dark_debris); #2 eye ghosts **fixed** (0
overcounts by machine and by eye); #3 rear-left flake spill **moved, not
gone** (now right-rear crown patches); #4 ear etch **unchanged**; #5
left-profile hair structure improved in v13 (0.714) but not v13raw (0.673,
above the 0.55 bar either way).

## Final assessment for the project owner

v13raw is the first bundle from this pipeline whose front hemisphere I can
call presentable at full resolution: the identity-destroying defects that
sank the shipped bundle — doubled faces, ear-on-cheek, black hairline
debris, milky films — are gone, verified by eye at 896 px across fourteen
azimuths and by a 66→9 collapse on the same calibrated harness that failed
the shipped bundle. It is still not a PASS: the hairline reads as a painted
beige band at 1:1, the right-profile ear carries baked-on flakes, the back
of the head lost its strand texture to the coverage surrender (a new,
harness-invisible regression), and the ears are pointed — a mesh defect no
texture pass will fix. Supportable claims: "single coherent face at all
front-hemisphere angles, source pose verified at +17.5°, front-face SSIM
0.63 vs photo (face region 0.67), zero debris/ghost detections at
photo-calibrated thresholds; suitable for front-facing and three-quarter
presentation at arm's length." Not supportable: "matches the photo"
(0.628 < 0.70 bar), "clean at 1:1 zoom" (hairline band, ear flakes),
"360°-ready" (flat back, elf ears), or any coverage number quoted as a
quality proxy — 0.40 here produced a far better result than 0.57 did in v9.
Recommended order if iteration continues: hair-toned fill for the
surrendered band and rear scalp (kills the band + back regressions), ear
texel exclusion (clears bar 1c), then re-measure 1a/1b — the face region is
already within ~0.03 SSIM of the bar once the band stops dragging it.

---

# v14 ADDENDUM (2026-07-05, same harness)

v14 = v13raw + contrast-conditioned witness gate (surrender only where the
photo's local luminance spread ≥ 0.055; layered hair-over-hair keeps
painting). Raw mesh. Coverage 0.395 → 0.462. My run: **6 failed checks**
(eye_count 4, identity[front] 2) vs v13raw's 9.

## T1 findings

**(a) Back of head — NOT fixed, only different at the margins.** Central-back
strand-texture proxies are statistically identical to v13raw (local luminance
std 1.34 vs 1.33; gradient energy 4.84 vs 4.79) against the shipped rear's
2.09 / 7.86 — the center is still a smooth chocolate wash
(`final_inspect/back_center_zoom.png`, `back_triptych_v14.png`). What the
witness gate actually delivered in the rear: the beige crown patches at
−45/−70/−135 are gone (crown_flakes 4 → 0, max reading 0.0007 vs gate
0.0008) and the rear quarters keep their profile-painted outer sheet. The
"rear repaints from the profiles" claim is supported only for the quarter
zones and the discoloration patches, not for the central back. The
flat-wash regression stands, downgraded from "new regression" to "known
limitation: profiles cannot witness the central rear."

**(b) Hairline band — mottle returned, band is busier than v13raw.** At az 0
el 8 (`final_inspect/v14_hairline_zoom.png` vs `v13raw_hairline_zoom.png`):
both temples now show pale flakes interleaved with dark curls where v13raw
had a smooth beige wash; a dark hair curl is painted inside the left temple
band — conspicuous at 1:1, and the eye detector counts it as a third eye at
az 0 (both elevations; `qa_out/v14/evidence/az+000.0_el00_annotated.png`).
This is the direct cost of "keep painting where the photo shows contrast":
the paint that returns at the temple boundary is the same film-band mix the
zone gate had suppressed. New front-view detector failure class vs v13raw
(v13raw az 0: eyes=2 clean, both elevations).

**(c) Under-eye — confirmed clean of v13's black dashes.** Beige tear-duct
flakes and a small gray dash at the nose bridge, same as v13raw
(`final_inspect/v14_eyes_zoom.png`); front local window 0.056 ≥ 0.05 PASS.
The v13 black-stroke regression was the flap-cut mesh, and it is absent on
the raw mesh as expected.

**(d) Front SSIM 0.616 vs 0.628 — the drop is real and sits at the hairline,
not in hair variance.** Region decomposition: face-hull 0.672 → 0.659
(−0.013), hair 0.470 → 0.459 (−0.011), and the top quarter of the face hull
(forehead/hairline band) 0.569 → 0.547 (−0.022) — the regression
concentrates exactly where the mottle returned. MAE unchanged at 23.5.
Does it matter: at arm's length no (both fail the 0.70 bar; the face is
equally readable); at 1:1 yes, the temples are visibly busier. It is a
small, real, wrong-direction change on the payload view.

## T2 — Final ranking: **v14 is the ship candidate**, replacing v13raw.

The trade: v14 gives up 0.013 face-region SSIM and two az-0 eye-count hits
(one painted curl blob) in exchange for clearing two items my ruling listed
as majors — the **right-profile ear etch** (side_right local window −0.137
FAIL → +0.118 PASS, global 0.649 → 0.679, best profile numbers of any
bundle; REPORT_V9 ranked blocker #4, open since v9, now closed) and **all
crown/flake detections** (crown_flakes 4 → 0; every debris/film/seam class
now zero across all 28 views). Trading two majors for one minor-to-major
plus a 0.013 face deficit is favorable. 66 (shipped) → 9 (v13raw) → 6 (v14)
on the identical harness.

### Updated PASS-bar table (v14)

| # | Bar item | v13raw | v14 | Verdict |
|---|---|---|---|---|
| 1a | Front SSIM ≥ 0.70 | 0.628 | 0.616 | **FAIL** by 0.084 — texture logic (temple mottle: top-band 0.547) + inherent share (hair 0.459; renderer shading ≈3 MAE units) |
| 1b | Front mean\|RGB\| ≤ 22 | 23.5 | 23.5 | **FAIL** by 1.5 — ≈3 units renderer shading (0.88+0.12·diffuse dimming), remainder texture tone; bake-time albedo compensation or harness offset-removal would close it |
| 1c | Worst 49-px window ≥ 0.05, all views | side_right −0.137 FAIL | front 0.056, side_left 0.147, side_right 0.118 | **PASS** — first bundle to clear 1c |
| 1d | Profiles SSIM ≥ 0.55, MAE ≤ 30 | pass | 0.680/17.9, 0.679/24.7 | **PASS** |
| 2 | Pose sweep within ±5° | pass | peak +0° | **PASS** |
| 3 | Detector gates, 28 views | 6 fails | eye_count ×4 (az0 curl blob ×2 = texture logic; −90 sliver undercount ×2 = detector sensitivity at grazing angle, eye visually present); all debris/film/flake/seam classes **zero** | **FAIL** (4 checks) |
| 4 | Evidence standard | pass | this addendum + `final_inspect/` | **PASS** |

Unmet items and fixability: **1a** texture logic first (clean the temple-band
paint quality — the witness gate decides *where* to paint but not *how
well*; the returning paint is mottled film-mix), inherent floor thereafter
(hair strands + shading; face region 0.659 needs ≈0.04 more once the band
stops dragging). **1b** split renderer-shading (≈3 units, fixable at bake or
measurement) / photo-tone (small). **3** az-0 curl blob is texture logic
(same band cleanup); the −90 undercounts are minor and part detector
limitation. Nothing unmet is geometry-bound except the standing elf-ear
silhouette and central-rear witness gap, which no texture pass reaches.

## Final owner-facing assessment (supersedes the one above)

v14 is the bundle to ship, and the first whose profiles pass every identity
gate including the local-damage windows — the ear etch that survived four
iterations is gone, and every debris, film, flake, and seam detector is at
zero across all 28 views. The front hemisphere remains presentable at arm's
length with a verified +17.5° source pose, but it moved slightly backwards
at the hairline: the contrast-conditioned gate re-admitted mottled paint at
the temples (face-region SSIM 0.659 vs v13raw's 0.672, and a painted curl
blob the eye detector flags at az 0), so "clean at 1:1" is still not a
supportable claim. The advertised back-of-head fix did not materialize:
central-rear strand texture is measurably identical to v13raw's flat wash
(~35% below the shipped bundle's), so supportable phrasing is "rear
discoloration patches removed," not "rear texture restored." Elf-ear
geometry stands. Supportable claims for v14: single coherent face at all
front-hemisphere angles; all three declared views pass structural identity
locally (worst windows +0.056/+0.147/+0.118); zero calibrated-detector
debris; front SSIM 0.616 (face region 0.659) vs the 0.70 bar. If one more
iteration is budgeted, spend it on the temple-band paint quality and the
az-0 curl — that is the entire remaining gap between v14 and a defensible
front-view PASS.

## Evidence index

- `final_inspect/v13raw_az*_el08.png`, `v13_az*_el08.png`, `v14_az*_el08.png` — front hemisphere at source elevation
- `final_inspect/v14_{hairline,eyes}_zoom.png` — v14 3× band crops (compare `v13raw_*`)
- `final_inspect/back_triptych_v14.png`, `back_center_zoom.png` — v13raw/v14/shipped rear comparison + central-back 2× crop
- `final_inspect/{v14,v13raw}_rear_quarters.png` — ±135° pairs
- `qa_out/v14/` — annotated failing views, identity pairs, `results.json`
- `final_inspect/{v13,v13raw}_{hairline,eyes,nosemouth}_zoom.png` — 3× band crops
- `final_inspect/{v13,v13raw}_m90_eye_zoom.png` — 4× profile-eye crops
- `final_inspect/back_triptych.png` — v9/v13/v13raw back-of-head comparison
- `final_inspect/ear_shipped_vs_v13raw.png` — elf-ear silhouette is pre-existing mesh geometry
- `qa_out/{v13,v13raw}/evidence/` — every annotated failing view + identity worst windows
- `qa_out/{iter3_final,v9,v13,v13raw}/results.json` — full machine-readable runs
