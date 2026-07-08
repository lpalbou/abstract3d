# CRITIC 1 — CYCLE 3 RULING
2026-07-06 ~21:45 CEST. All numbers below are MY OWN runs on the ON-DISK bundles
(face texture md5 733aa932 20:31, ship b8e2b0d4 07:36, owl ff746509 20:57).
Evidence: `/tmp/critic1/c3/` (renders, harness logs), sheets
`/tmp/critic1/evidence/sheets/c3_*.png`. Prior state: RULING_CYCLE2.md.

## VERDICT: FAIL — 10 OPEN entries (1 major regression introduced this cycle)

The cycle delivered three genuine kills: the FACE-01 film band is gone as a
class, SHIP-03's nose melt is fixed at the root with a proper ceiling
experiment, and FACE-09's leopard is combed to the material bar. The identity
gate's MAE half is green for the first time. But the band repaint's billboard
mechanism stamped hard BLACK STROKE artifacts across five-plus views of the
face (new entry FACE-20, major, visible at 1x), and the zero-open standard
does not trade a beige defect for a black one.

## MY HARNESS STATE (on-disk bundles, my runs)

| bundle | verdict1 | texture_qa |
|---|---|---|
| face-2mv | FAIL 1 — identity[front] SSIM 0.651 < 0.70 ONLY; MAE 21.5 ≤ 22.0 GREEN (first time); sides 0.660/19.2 lm 0.115, 0.670/21.0 lm 0.180; all 28-view detectors green | PASS 13/13 (energy 1.263, brightness 0.948) |
| face-2mv COMPENSATED (my run of /tmp/c2d/qa_shadecomp.py) | front 0.677 / 14.34 — MAE passes the re-tightened 15.0; SSIM below both 0.70 and 0.72 | — |
| hunyuan-starship | n/a (non-face) | PASS 13/13 (energy 0.620, facets 0.012, smears 0, brightness 0.829) |
| hunyuan-owl | n/a | PASS 13/13 (energy 0.708, smears 0, brightness 0.891); pose gate DECLINED the az−5/el−8 drift candidate (metadata: az0, estimated=false, candidate score 0.0528 recorded) — the parent's relative-margin fix verified working on disk |

## T2 — IDENTITY CEILING ADJUDICATION

**The claim**: raw front SSIM 0.70 is unrealizable from these inputs; the
green frontier is 0.65–0.69; V20 reaches 0.719 only by collapsing everything
else.

**My verification**: I ran the full verdict1 battery on the solver's saved
`bundle_V20_2048` myself: front **0.719/19.8** — and **30 failed checks**:
side_left collapses to 0.546, side_right to 0.548, `lip_in_hair` at az0,
`skin_in_hair` 0.0153–0.0239 across six-plus views, DOUBLED EYES at az−90 and
az+135, crown flakes. The verifier's independent audit matches. The
counterfactual bounds (D's 0.7545 metric-space, C1's 0.7692 gate-mapped) edit
the comparison, not a realizable texture. The frontier was mapped across ~30
A/B arms of the strongest mechanism family fielded to date.

**Ruling (a)**: the ceiling proof MEETS MY BAR — as a JOINT claim. Raw 0.70
is reachable in isolation (V20 proves that), but 0.70 with the side identities
and all detectors green is demonstrated unrealizable from these three photos
with every mechanism class fielded across three cycles, reproduced by two
independent parties and by me. The apron is co-witnessed by three photos whose
contents collide under parallax; the identity budget there is zero-sum. This
is the same evidence quality I accepted for the ear-parallax and ship-content
limits.

**Ruling (b)**: per my cycle-2 terms (this is the cycle boundary) and critic
2's advisory:

1. **The compensated protocol is ADOPTED as the authoritative identity
   measure.** The shading bias is a proven measurement artifact
   (perfect-texture floor SSIM 0.977 / MAE 11.45 under the render protocol);
   the raw 0.70 constant sits ABOVE the demonstrated joint input ceiling
   (raw green frontier 0.65–0.69 ≈ comp 0.68–0.72) — an absolute bar above
   the input ceiling is a broken gate that invites V20-style collateral
   gaming. Raw numbers continue to be REPORTED for continuity.
2. **Budgets anchored to the input ceiling, not absolute constants**: comp
   MAE ≤ 15.0 (enforced, passes at 14.34 — my run); comp SSIM floor set at
   **0.70** = the demonstrated joint ceiling (~0.706–0.716 comp across the
   frontier) minus a small margin. NOT 0.72: that number is above what these
   inputs can realize with every other gate green.
3. **FACE-14 stays OPEN — not PROVEN-LIMIT.** The on-disk bundle measures
   comp 0.677 vs the 0.70 anchored bar. The gap decomposition is measured,
   and it is NOT all input-limit: +0.031 SSIM sits in the protected feature
   regions (the V20−V21 gap — the tear-duct/lash-line/lip-edge residue that
   order 2 proved un-committable under witness demotion but which a
   feature-aware repair lane could bank), and the new FACE-20 strokes are
   photo-absent structure that costs the same gate. Demonstrated, reachable
   headroom exists; the entry is not at its ceiling. It moves to
   PROVEN-LIMIT only when the feature-repair lane's budget is banked or
   proven unbankable by the same standard of evidence.

**Harness governance (also adjudicated)**: solver 3's `texture_qa`
visibility-reconstruction fix (reads `source_registration`, reconstructs
per-view visibility in the frame the bake actually used; absent key = legacy)
is ACCEPTED — it makes region attribution true to the bake (measured 0.051 vs
0.046 attribution drift on identical textures), relaxes nothing. Their
proposed cycle-4 upgrade (bakes emit masks, harness does one-sided
containment) is endorsed, not required.

## T1 — LEDGER WALK (C2 open entries → C3 state; my crops)

Sheets: `c3_face1/2/3.png`, `c3_newdef.png`, `c3_final_checks.png`,
`c3_shipowl.png` (all C2-vs-C3 pairs at identical crop centers).

| entry | C3 state | my evidence |
|---|---|---|
| FACE-01 film band (blocking→major in C2, taupe band residual) | **FIXED** | The taupe putty sheet is REPLACED by near-black hair blending to skin carrying the photos' own strand layout at srcpose/az0/±22.5/±45 (`c3_face1.png` all pairs vs PHOTO tiles). Residual: a narrow pale wisp ribbon at the left temple edge that reads as wisps over hair — the acceptance's read. skin_in_hair at +22.5 is 0.0044 vs my 0.002 ask: unmet, but the islands are the below-ear tape/ear complex (FACE-07 territory), the photos' own calibration range (0.0004–0.0053) brackets it, and darkening was measured to trade into dark_debris. I close the band entry and carry the ear complex under FACE-07. The mechanism's COST is FACE-20 (below). |
| FACE-03 under-eye chips | **OPEN (IMPROVED, minor)** | Open-skin chip fields visibly cleaned at az−22.5/az0 4x (`c3_face2.png` row 3). Tear-duct whites + lash-line dashes remain at 4x. The PROVEN-TRADE evidence (committing them costs eye_count; ring votes 0.30–0.81 vs 0.96 bar) is accepted as a limit OF THE CURRENT MECHANISM, not of the inputs — the documented remedy is a feature-aware repair lane (pipeline work), so the entry stays open, not proven-limit. |
| FACE-04 mouth/chin | **OPEN (IMPROVED, minor)** | Chin flakes and mouth-area pale chips cleaned; corner smear smaller (`c3_face2.png` rows 1–2, srcpose mouth pair). Remaining: lip-edge dark-red dash (same trade class as FACE-03) + a broader-but-smoother tan wash on the neck below the chin (`c3_final_checks.png` tile pair 1–2). |
| FACE-05 pale seam column | **OPEN (unchanged, minor)** | Pale column inner-eye→cheek→philtrum still present at 4x (`c3_face2.png` nose pair). S1 compositing lane; honestly declared untouched by all three solvers. |
| FACE-06 front tone split | **MERGED → FACE-14** | The split is what the identity gate measures; one entry, one number. |
| FACE-07 ear debris | **OPEN (unchanged, minor)** | Pale chips at both ears persist; left ear-band pale fraction ticked UP 0.025→0.046 (clamp feathering, solver-disclosed). The NEW black arcs at the ears are FACE-20, not this entry. |
| FACE-09 rear leopard | **FIXED (material read)** + PL(strand content, C2 grant) | Rosette leopard GONE at az180/±135 4x — combed low-contrast directional grain (`c3_face3.png` rows 1–2); blotch az180 5.23→2.35 (verifier), the hard upper-back rectangle relaxed to a soft rectangular shadow — still faintly visible at 4x with straight edges: logged as the entry's residual, cosmetic. Strand-level CONTENT remains the C2 proven-limit (capture remedy: rear photo). |
| FACE-10 curtain stripes / skin halo | **FIXED** | Behind-ear pale stripes cleaned by the commit class (`c3_face3.png` +112.5 pair). The ±135 inner-edge striping that remains is OBSERVED photo content (witness w90 0.19–0.37, depth-tested provenance) — rendering photo truth is not a pipeline defect; demoting confident witnesses is what I forbid elsewhere. |
| FACE-11 chest straps | **OPEN (unchanged, minor)** | Same smears (`c3_face3.png` chest pair). The consensus veto is CORRECT by its own semantics (side_left reads the surround 47–52% bright — a real garment boundary); the fix needs garment-boundary modeling. Fixable, therefore open. |
| FACE-12 bust disc | **OPEN (IMPROVED, minor)** | Dark radial rim slivers committed/cleaned (`c3_final_checks.png` disc pair); the tan disc WASH on the synthetic cut face remains — needs geometry-aware bottom-cap toning (S4 lane). |
| FACE-13 crown | **OPEN (unchanged, minor)** | Mesh flaps are S4 with still no ceiling experiment filed; mottle partially combed where donors qualify. |
| FACE-14 identity[front] | **OPEN (IMPROVED)** | RAW 0.651/21.5 — the MAE gate is GREEN for the first time (was 22.9 → 22.2 → 21.5); SSIM 0.651 is the ONLY harness failure across all three assets. COMP (authoritative from now): 0.677/14.34 vs anchored 0.70/15.0 — MAE green, SSIM 0.023 short with the headroom localized in the feature lane (+0.031 measured) and FACE-20 removal. See T2. |
| **FACE-20 (NEW) billboard dark strokes** | **OPEN (major, REGRESSION)** | Hard black stroke/arc artifacts stamped by the band repaint's source-billboarding + domain borders: a jagged black crack from the parting down the LEFT TEMPLE past the brow at az0 (visible at 1x, `c3_newdef.png` 6x tile vs C2's clean same-region); a feathered dark streak along the temple silhouette at az−22.5; a black ragged line at the az−90 hairline frontier; and NEW black arcs TRACING THE EAR HELIX at az+90/+112.5 (`c3_newdef.png` rows 1–2: C2 ears have no arc). All detector-green (they read as hair-connected). The solvers disclosed the az0 stroke and az−22.5 streak; the ear arcs were not disclosed. An owner sees the az0 crack in the first five seconds of a turntable. |
| SHIP-03 nose melt | **FIXED** | On-disk bytes = the fix (md5 b8e2b0d4 verified). az0: the molten streak concavity now reads as a structured dark intake with internal grill detail and bright rim frame; el−20 under-nose streaks replaced by hull-consistent material; srcpose crisp (`c3_shipowl.png` rows 1–2, `c3_final_checks.png` ship pairs). Root cause (54 px override-pose frame registration) is proven by the checker ceiling experiment — content decorrelated to CHANCE at all stretch bands under the old registration, survives 0.61–0.73 fixed — this is exactly the experiment class my protocol demands. Registration scope rule (overridden poses only; estimator co-adapted to legacy frame) verified honest: face pose estimated, ship override, owl canonical. Rim/registration side-effect hunt at 16 az × 3 el: clean. Residual wing-edge fray is thin-geometry (S4), noted, not melt. |
| SHIP-05 glow blob | **OPEN (unchanged, minor)** | Soft bright zone at az−135 el+15 unchanged (`c3_final_checks.png` glow pair); no new evidence filed this cycle. |
| OWL (canary) | **HELD** | Fresh on-disk rebake PASS 13/13 (my run); the mid-cycle estimator drift (az−5/el−8 at 0.0528 displacing declared frontal) that cost a 4x dark-pocket gate is closed by the parent's relative-margin fix — my metadata read confirms the decline (az0, estimated=false, candidate score recorded); foot-pocket 4x crop clean vs the drift-era harsh striped patch (`c3_shipowl.png` row 4). |

Previously closed entries re-verified standing: FACE-02, FACE-15 (eyes 1/1 at
±90 my run, structured eye in my −90 crop), FACE-16, FACE-17 (no ghosts in my
az0/±22.5 crops), FACE-18 (pose +20/+8 at 0.0152, sweep peak +0), FACE-19
(side_right 0.670/21.0 lm +0.180), SHIP-02/06/08, OWL-01/02/04; PROVEN-LIMIT
grants standing: FACE-08, SHIP-01/04/07, OWL-03.

## COUNTS (all entries ever ledgered: 34)

- **FIXED: 16** (FACE-01, -02, -09, -10, -15, -16, -17, -18, -19; SHIP-02,
  -03, -06, -08; OWL-01, -02, -04)
- **PROVEN-LIMIT (closed with documented capture remedies): 5** (FACE-08;
  SHIP-01, -04, -07; OWL-03)
- **MERGED: 1** (FACE-06 → FACE-14)
- **OPEN: 10** — FACE-20 (major, new), FACE-14 (major, gate), FACE-03, -04,
  -05, -07, -11, -12, -13, SHIP-05 (minor), + FACE-09's cosmetic rectangle
  residual carried inside its entry.

**FAIL** under the zero-open standard.

## CYCLE-4 ORDERS (ranked)

1. **FACE-20 — remove the billboard stroke/arc artifacts.** Band lane. The
   curtain-edge and ear-helix strokes are stamped content the stamping pose
   does not witness, plus domain-border feathering that prints crease lines.
   Candidate levers, in the mechanism's own vocabulary: an off-pose displacement
   veto for high-contrast elongated stamps (the strokes are the parallax dual
   of the third-eye class the feature moat already vetoes — extend the veto to
   silhouette-adjacent dark strokes), and widening the border feather at the
   curtain/ear complex. Acceptance: no new dark stroke at any of the 48
   battery views at 2x vs the C2 baseline; detectors stay green; identity does
   not regress below comp 0.677.
2. **FACE-14 → comp ≥ 0.70/15.0 — bank the feature-lane budget.** The +0.031
   measured headroom lives in the tear-duct/lash-line/lip-edge residue
   (FACE-03/04's remainder). Mechanism per solver 2's own analysis: extend
   rescue-disc transplant semantics to partial feature fringes (feature-aware
   repair), NOT witness demotion. Closing FACE-20 also feeds this gate
   (photo-absent black structure costs SSIM).
3. **FACE-05 — the pale seam column** (S1 compositing): last untouched
   front-face tone artifact; gradient-domain lane owns it.
4. **Minors sweep**: FACE-07 ear chips (with the FACE-20 fix opening the ear
   complex for treatment), FACE-09 rectangle tone-equalization inside the
   comb regime, FACE-11 garment-boundary modeling, FACE-12 disc toning (S4),
   FACE-13 crown flap clamp-or-ceiling-experiment, SHIP-05 glow
   source-trace.
5. **Governance**: comp budgets stay anchored — if a cycle demonstrates a
   different joint ceiling (either direction) with V20-quality evidence, the
   0.70 comp floor is re-anchored, not negotiated.

## OWNER-FACING PARAGRAPH

Open the three files in MeshVault tonight and this is what you get. The
STARSHIP is done: the nose that used to melt into gray streaks head-on is a
structured intake with a readable grill, the hull carries crisp panel
detail on the photo side and plausible granular material on the far side, and
every remaining softness is the documented single-photo content limit — a
prow-on or ±45° second photo removes it. The OWL is stable and clean at every
angle and both zoom levels. The FACE at arm's length is the best it has been:
the beige film that sat over the temples for three cycles is gone — the
hairline now grades from dark hair into skin with real strand texture — the
under-eye and chin chips are swept, and the rear hair reads as combed material
instead of leopard spots. But at close zoom the fix that killed the film band
left its own fingerprints: hard black strokes — one crossing the left temple
head-on, arcs tracing both ears in profile — that read as painted cracks, and
faint pale residue persists at the tear ducts, lash lines, and one lip edge.
Those strokes are mechanical artifacts of the new repaint and are the next
cycle's first target; they are fixable, so I am not passing this. What is
genuinely at the input's ceiling: front-identity texture match beyond ~0.70
compensated (three photos fight over the same hairline apron under parallax —
remedy: one photo at ±30–45° or a cleaner hairline in the front shot), strand-
level rear hair content (remedy: one rear photo), and the ship/owl far-side
content limits already granted. Everything else on the open list is pipeline
work, not physics.

— Critic 1
