# CRITIC 1 — CYCLE-2 RULING
2026-07-06 05:40 CEST. Prior state: `/tmp/critic1/RULING.md` (cycle 1: 31
OPEN / 0 FIXED, FAIL). This ruling covers the freshly rebaked on-disk
bundles and the two harness-governance questions. Everything below is
re-verified by my own runs at owner conditions (16 az x el {-20,0,+15} at
1000 px + 2x/4x region crops + declared-pose renders + both harnesses +
full test suite). Evidence: `/tmp/critic1/c2/`, sheets under
`/tmp/critic1/evidence/sheets/c2_*.png`.

## What I verified (my runs, not solver numbers)

| bundle (on disk, md5 of texture.png) | verdict1 qa | texture_qa |
|---|---|---|
| face-2mv (05bcae03…, baked 05:05) | **FAIL 2** — identity[front] SSIM 0.630 < 0.70, MAE 22.2 > 22.0; every detector class green (eyes 1/1 at −90 both els; dark ≤ 0.0030; crown ≤ 0.0007; pale ≤ 0.0008; seams ≤ 0.26; no 3-eye blobs) | **PASS 13/13** |
| hunyuan-starship (ef660b57…, 01:08) | n/a (face harness) | **PASS 13/13** (smears 0, facets 0, energy 0.631, brightness 0.845 [matte_robust]) |
| hunyuan-owl (fb13af0b…, 01:18) | n/a | **PASS 13/13** (smears 0, energy 0.683, brightness 0.891 [matte_robust]) |

Test suite: **192/192 passed** (my run). Pose provenance in metadata:
face az+20/el+8 (gradient_ncc 0.0152, accepted), ship az+30/el+15
(override, honest), owl az0 (estimator declined at 0.0254 — gate working).
Shading-compensated identity gate re-run by me on the disk bundle:
0.654 / 16.9 (matches solver D's population table within 0.002).

## T1 — Ledger walk (all 31 entries)

### face-2mv — 6 FIXED, 1 PROVEN-LIMIT, 12 OPEN

| ID | status | my evidence / basis |
|---|---|---|
| FACE-01 film band | **OPEN (IMPROVED, blocking->major)** | Torn beige film, black holes, dark curls: gone. What remains is a smooth TAUPE BAND from parting to ear, plainly visible at 2x at the declared pose — not hair, not skin (`c2_band_judgment.png` photo-vs-render pairs; `c2_face_pairs1/3.png`). Wisp-remnant *strand content* is the accepted G3-class limit; the band's TONE (taupe vs the photo's near-black hair mass) is still fixable and stays OPEN. |
| FACE-02 black parting holes / az−90 crack | **FIXED** | az0 el0/el+15 parting, az−90 streak, az−67.5: no black holes or cracks in any crop (`c2_face_pairs1.png` parting pair, `c2_face_pairs2.png` hairline-streak pair); dark/crown detectors green at all 28 views. |
| FACE-03 under-eye flakes | **OPEN (IMPROVED)** | Heavy 3D chip clusters reduced to a few small chips + thin gray dashes, still visible at 4x at az0/±22.5/srcpose (`c2_face_pairs1/2.png`). |
| FACE-04 mouth-corner smear / below-lip / chin | **OPEN (IMPROVED)** | Corner smear smaller; a red-dark dash at the lower-lip edge + beige chips at the chin remain at 4x (`c2_face_pairs1.png`). |
| FACE-05 nose seam column / columella blob | **OPEN (IMPROVED)** | Blob shrunk; pale sliver + faint seam column remain at 4x. |
| FACE-06 front tone split + eye seam | **OPEN (IMPROVED)** | Split much subtler at 1x; folded into the identity gate, which still fails. |
| FACE-07 ear texture debris | **OPEN (IMPROVED, minor)** | Dark strokes gone both ears; small pale chips remain at 4x (`c2_face_pairs2.png` ear pairs). Solver D traced residue to fill anchors in the contested ear band — same family as the band fill. |
| FACE-08 elf ears | **PROVEN-LIMIT (closed)** | Geometry: solve4's conservative-clamp bound stands (apex prominence ≥0.05 vs 0.02 cap). Texture half verified DONE by solver D (apex is hair-painted; paint runs more conservative than the photos' own skin boundary — their D2 audit; my crops agree). Remedy documented: geometry regeneration with ear references. |
| FACE-09 rear leopard mottle | **OPEN (unchanged)** + PL(strands) | 4x blotches essentially unchanged (`c2_face_pairs3.png`, `c2_glow_exact.png` back pairs). Strand *content* stays PROVEN-LIMIT (cycle-1 ceiling); blotch-free directional material (owl-grade grain) is demonstrated achievable and NOT yet delivered here. |
| FACE-10 curtain tan stripe / skin halo | **OPEN (unchanged, minor)** | Tan inner-edge striping still present at ±135 (`c2_face_pairs3.png`). |
| FACE-11 chest straps / white blob / lace | **OPEN (unchanged, minor)** | Same smears at az0 chest (`c2_face_pairs3.png`). |
| FACE-12 bust cut disc | **OPEN (unchanged, minor)** | Tan disc + dark rim streaks at el−20 unchanged. |
| FACE-13 crown flaps + mottle | **OPEN (unchanged, minor)** | Ragged crown silhouette (mesh) + mottle at el+15 unchanged; no ceiling experiment filed. |
| FACE-14 identity[front] | **OPEN (IMPROVED)** | RAW 0.630/22.2 vs 0.70/22.0 (was 0.613/22.9). COMP protocol (my run): 0.654/16.9 vs 0.70/15.0 — fails BOTH protocols. Solver D's counterfactual: band-fixed ⇒ 0.730 RAW. The path runs through FACE-01. |
| FACE-15 az−90 profile eye | **FIXED** | eyes=1/1 at −90 el0/el10 (my harness run); structured eye in my 4x crops at el0 AND el15; no transplant artifacts at −67.5/−45/−22.5/az0; the v14-era −45 el10 trade did not reappear (`c2_face_pairs2.png`, `c2_glow_exact.png`). |
| FACE-16 third-eye curl blob | **FIXED** | No 3-count at any of 28 views (my run); no curl-in-film at ±22.5 (crops). |
| FACE-17 nose/lip ghost regression | **FIXED** (as regression class) | No doubled lip pair, no nose ghost column in any az0/±22.5 crop; residual pale sliver tracked under FACE-05. |
| FACE-18 pose-drift regression | **FIXED** | Score-floor gate verified across all three assets (face accepted at 0.0152 → +20/+8, sweep peak +0; owl declined at 0.0254 → pinned frontal; ship override honest). |
| FACE-19 delight side_right tone regression | **FIXED** | side_right MAE 20.8 / SSIM 0.680 / localMin +0.214 — best profile numbers measured on this asset; no relighting artifacts in my profile crops. |

### hunyuan-starship — 3 FIXED, 3 PROVEN-LIMIT, 2 OPEN

| ID | status | my evidence / basis |
|---|---|---|
| SHIP-01 fill side clay-cloud | **PROVEN-LIMIT (closed, content)** | The blocking "clay blank" read is gone: fill now carries granular material texture at 2x/4x with zero facets/smears/seam failures (`c2_ship_pairs.png`, `c2_ship_detail.png`). Panel CONTENT on unwitnessed surfaces stays impossible from one photo (accepted cycle-1 P3 + solver C's σ-guard/structure-transfer boundary). Remedy: side/rear reference photos or a generative texture prior. Honest note: the fill reads as marble-like statistical material, clearly distinguishable from the photo side's panels on a turntable. |
| SHIP-02 dark fill fragments 4x | **FIXED** | close.dark_smears_4x = 0 (my run); no fragments in my region crops. |
| SHIP-03 bow/nose melt | **OPEN (unchanged)** | Dark stretched streaks at the nose concavity + frayed wing leading edges at az0 el{−20,0} unchanged (`c2_ship_pairs.png` nose pair, `c2_ship_detail.png`). Nobody claimed it this cycle; no ceiling experiment filed — stays OPEN (suspected grazing-projection class, S2/S3). |
| SHIP-04 underside/engines featureless | **PROVEN-LIMIT (closed, content)** | Featureless wash replaced by oriented material texture at el−20/az180 (`c2_ship_pairs.png` underside/engines pairs); mechanical CONTENT = same accepted limit as SHIP-01. |
| SHIP-05 glow blob rear-top | **OPEN (IMPROVED, minor)** | Softened and textured-over but a soft bright zone remains at az−135 el+15 (`c2_glow_exact.png`). |
| SHIP-06 global tone | **FIXED** | 0.845 against the SUBJECT reference (T2.1 accepted below); margin 0.125; decomposed: −6.0% viewer shading, remainder within photo-natural range. |
| SHIP-07 photo->fill seams mid-hull | **PROVEN-LIMIT (closed, content)** | Tone steps within calibrated allowance (seam gate green, my crops show no hard steps); the remaining visible boundary is the observed/synthesized CONTENT frontier — panel lines cannot be continued without inventing content (same accepted limit; structure transfer documented out-of-scope after measured-worse prototypes). |
| SHIP-08 fill-energy collapse regression | **FIXED** | fill_energy 0.631 / facet_cellular 0.046 / facet fields 0 (my run); 4x crops show calibrated texture, no granite noise (σ guard held). |

### hunyuan-owl — 3 FIXED, 1 PROVEN-LIMIT, 0 OPEN

| ID | status | my evidence / basis |
|---|---|---|
| OWL-01 brightness 0.567 | **FIXED (reclassified: predominantly measurement artifact)** | The old gate measured the light-gray BACKDROP (100% of frame classified foreground). Against the subject (same matte the bake consumes): 0.891, margin 0.17 (my run). True residual −3.9% albedo (single-view delighting unidentifiable — documented limit), −7.3% harness viewer shading. My side-by-side crops read tone-faithful. |
| OWL-02 dark smears 4x | **FIXED** | 0 (my run); concavity crops clean. |
| OWL-03 rear flat wash | **PROVEN-LIMIT (closed, content)** | Back now carries dense directional carved-grain material consistent with the front at 2x/4x (`c2_owl_pairs.png`); the specific carved-feather CONTENT stays with the accepted single-photo limit. This is the material bar FACE-09 should meet. |
| OWL-04 pose lottery regression | **FIXED** | Estimator declined (0.0254 < floor) → pinned frontal az0; metadata honest; renders aligned. |

### New-defect hunt (cycle-2 mechanisms) — no new entries

- **Film-band commitment edges**: no black rims, no new flake fringes; the
  band boundary itself stays under FACE-01.
- **Transplanted eye disc off-profile**: checked az−67.5/−45/−22.5/az0 and
  el15 — no disc edges, no doubling, eye counts correct at all 28 views.
- **Energy-calibrated fill noise at 4x**: ship marble and owl grain are
  uniform, facet-free, smear-free; no injected-granite signature (σ guard
  verified by gate + eye). Style note for the owner below.

## T2 — Harness governance rulings

### T2.1 — texture_qa owl-brightness matting fix: **ACCEPTED**

The gate's stated purpose is albedo fidelity of the textured SUBJECT. The
old "non-white" heuristic classified 100% of the owl frame as foreground
(gray backdrop, median 205) and scored the render against the backdrop —
an objective measurement error, not a strictness feature. The fix measures
the subject through the SAME matte the bake consumed (consistency of
reference), records the method in the gate line and results.json
(`[matte_robust]`), and falls back explicitly (`heuristic_nonwhite`) when
the matte is degenerate. Decisive honesty evidence: the change moves
allowances in BOTH directions (owl seam allowance loosened 52.2→61.2, ship
TIGHTENED 71.8→60.0; coverage reconciliation got stricter, qa 0.261→0.211
vs bake 0.177) — this is a truth fix, not flattery. Photos still pass
calibration. Consequence: OWL-01 is reclassified (see table). Standing
condition: the method string must remain in every gate line; if the matte
model is unavailable, the fallback's degraded status must stay visible.
Noted risk (accepted): harness and bake now share the segmentation failure
mode; for brightness/seam references that is correct by design (the gate
measures fidelity to what the bake textured), and the front-visibility
reconstruction change makes region attribution MORE faithful, not less.

### T2.2 — verdict1 shading compensation (opt-in): **VALID, ADJUNCT-ONLY this cycle — raw gate stays authoritative; both must be reported side by side**

Verified by me: the mechanism (white-texture render = exact per-pixel
shade field through the identical shader/resampling chain; photo-side
multiplication; registration untouched) is sound, and the measured
perfect-texture floor (SSIM 0.977 / MAE 11.45 — half the 22.0 MAE budget
consumed by the harness's own viewer before any texture defect counts) is
a genuine measurement artifact: the render is double-shaded (photo shading
baked into albedo × renderer shade) while the photo is single-shaded.
Compensating and RE-TIGHTENING (22→15 front, 30→24 sides) is the right
construction, and my own run reproduces solver D's numbers (0.654/16.9 vs
their 0.656/16.8 on a bundle two bakes apart).

Why not adopt as the gate now: (1) the SSIM budget was NOT re-tightened —
compensation lifts SSIM by ~+0.024 at the relevant range while the bar
stays 0.70, which is a ~0.02 effective relaxation on the failing metric of
the failing asset; adopting it mid-cycle would move the goalposts exactly
where the number is short. (2) Gate changes while a gate is red require a
cycle boundary. Terms for next-cycle adoption: SSIM bar rises 0.70→0.72
(the measured floor delta, keeping strictness parity), MAE budgets 15/24
as proposed, compensation-mode recorded in results.json, and the raw
numbers still reported as diagnostics. Under BOTH protocols today the
face front-identity fails (raw 0.630/22.2 vs 0.70/22.0; comp 0.654/16.9
vs 0.70/15.0) — so this ruling flatters nothing.

## Counts and verdict

| status | count | entries |
|---|---|---|
| FIXED | **12** | FACE-02,15,16,17,18,19; SHIP-02,06,08; OWL-01,02,04 |
| PROVEN-LIMIT (closed, remedy documented) | **5** | FACE-08; SHIP-01,04,07; OWL-03 |
| OPEN | **14** | FACE-01,03,04,05,06,07,09,10,11,12,13,14; SHIP-03,05 |

**VERDICT: FAIL.** 14 OPEN entries remain; zero-open is the bar. This is
the honest arithmetic: 31 → 14, with every cycle-1 blocker either fixed
(black holes, fill clay-cloud read, canary regressions) or reduced to a
residual — but "reduced" is not zero, and two of the fourteen (FACE-01
band tone, FACE-14 identity) are major and owner-visible at 2x.

## Owner-facing paragraph (what you will see in MeshVault)

Opening the three files today: the **starship** and **owl** are, for the
first time, clean at close zoom — no dark fragments, no honeycomb facets,
no flat clay anywhere; the owl's back carries convincing carved-grain
texture and its tone now matches the statue rather than the studio
backdrop; the ship's unphotographed side carries plausible worn-metal
material instead of smooth cloud. What neither asset can have from a
single photo is invented CONTENT — the ship's far side shows statistical
material, not panel lines, and orbiting from the photographed side to the
far side you will see that character change; the same applies to the
owl's rear feather carving and the ship's nose concavity, which still
shows stretched dark streaks at head-on views (open defect). The **face**
is one coherent, recognizable woman at every angle with correct eyes at
both profiles; the black parting holes and torn film edges are gone. At
close zoom you will still find: a smooth putty-gray band along the
temple hairlines where the photo shows wispy near-black hair (the top
remaining defect — fixable, the fill tone must commit to the hair mass's
darkness), a few small beige chips under the eyes and at the mouth
corner/chin, a soft blotchy character in the rear hair at 4x, the pointed
ear silhouette (mesh geometry — needs regeneration with ear references;
texture verified correct), and the unchanged chest-strap smears and bust
under-disc. The front view matches the photo at SSIM 0.63 against a 0.70
bar; roughly a third of the numeric gap is the hairline band, a third is
the harness's own viewer shading (now measured exactly: a perfect texture
scores 0.977/11.45), and the rest is small parallax residue — with the
band fixed, the raw bar is projected reachable (0.730 counterfactual).

## Orders for cycle 3, ranked

1. **FACE-01 band tone commitment** (the deciding fix): the committed
   band must inherit the LOCAL hair-mass tone gradient (near-black at the
   mass boundary, blending to skin at the face edge), not the current
   uniform taupe. Acceptance: srcpose/az0/±22.5 2x crops read as
   hair-to-skin gradient; skin_in_hair at +22.5 ≤ 0.002; identity[front]
   RAW ≥ 0.70 (D's counterfactual says the budget is there).
2. **FACE-03/04/05 residual chips and dashes**: target zero visible at 4x
   at az0/±22.5/srcpose. Use B's disc localization + A's commit semantics;
   heed D's negative results (blind demotion trades debris for identity).
3. **FACE-09 rear material**: apply the owl-grade directional-grain
   statistics to the hair fill (elongated, low-contrast, combed along the
   G3 multigrid field); acceptance: az180/±135 4x with no leopard blotches.
4. **SHIP-03 nose melt**: unclaimed for two cycles. Either fix (grazing
   stretch demotion + retone at the concavity) or file the ceiling
   experiment; until then it is OPEN and major at head-on views.
5. **FACE-14**: re-measure after order 1; adopt the compensated protocol
   under T2.2's terms (SSIM 0.72 / MAE 15-24) at the cycle boundary if
   desired — both protocols reported either way.
6. **Minors sweep**: FACE-07 residue (fold into order 1's band-fill
   domain per D's shard provenance), FACE-10 curtain stripe, FACE-11
   straps, FACE-12 bust disc, FACE-13 crown, SHIP-05 glow.
7. Keep the canary discipline: owl+ship rebake and full battery after
   every face-lane mechanism lands (this cycle they stayed bit-identical
   under face-lane changes — verified; keep it that way).
