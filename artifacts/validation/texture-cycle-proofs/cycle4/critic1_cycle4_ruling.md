# CRITIC 1 — CYCLE 4 RULING
2026-07-07 ~04:15 CEST. All numbers are MY OWN runs. Evidence:
`/tmp/critic1/c4/` (renders, harness logs, my bake bundles under
`/tmp/c4_2/bundle_critic_b{1,2,3}` + `/tmp/critic1/c4/bundle_noident`),
sheets `/tmp/critic1/evidence/sheets/c4_*.png`. Prior: RULING_CYCLE3.md.

## VERDICT: FAIL — 6 OPEN (one gate-major, no blocking visual defect left)

The four FACE-20 stroke sites are dead at every view I rendered, the
FACE-05 column and the bust-disc wash are gone, and the feature fringes are
visibly repaired. The identity gate is not yet met (comp 0.688 vs my 0.70
floor), and the cycle shipped its best texture into the artifact directory
with the WRONG BAKE RECIPE — the on-disk bundle underperforms the tip it
was baked from. That is a publication defect, and zero-open means zero.

## T1a — THE VARIANCE INVESTIGATION (ordered first; finding: THERE IS NO VARIANCE)

The integrated-tip discrepancy (on-disk raw 0.646/22.0 + debris marginal vs
the solvers' paired 0.688/14.4) is fully explained, with hashes:

1. **The face bake is bit-deterministic on this tip.** I baked the standard
   recipe three times back-to-back (`/tmp/c4_2/bake.py`, which passes the
   source view's `identity_image` = the bundle's un-matted input.png).
   All three textures carry ONE md5 (`c39d65bf…`) and identical numbers:
   raw 0.662/21.7 (MAE green), comp **0.688/14.4**, verdict1 FAIL 1
   (front SSIM only), ALL detectors green including az−35 el10
   dark_debris. Zero spread across three draws. The C3-era
   "metric-deterministic, not bit-deterministic" note does NOT describe
   the current tip; whatever nondeterminism existed then is gone.
2. **The on-disk artifact is not a bad draw — it is a different recipe.**
   I re-baked with the SAME standard recipe minus `identity_image`
   (`/tmp/critic1/c4/bake_noident.py`): the result is BIT-IDENTICAL to
   the on-disk bundle (md5 `5881359e…` both) and reproduces its exact
   failure set: raw 0.646 / MAE 22.04 (fails the 22.0 budget by the
   margin), comp 0.673/14.9, dark_debris 0.0033 (7 islands, gate 0.003)
   at az−35 el10. Solver 2's report §9 warned precisely about this: the
   fringe stage must register against the identity contract's own photo;
   with only the matted rgba it lands in a measurably different
   registration basin. The 03:32 artifact rebake omitted it — the fringe
   stamps land differently (15 → 11 stamps), the raw MAE crosses its
   budget edge, and the debris marginal appears.
3. **Conclusion: the cycle-4 gains SURVIVE integration** — the correct-
   recipe tip measures exactly the solvers' paired end-state (comp
   0.688/14.4, +0.015 over C3's 0.673) with every detector green — and
   the on-disk artifact must be RE-PUBLISHED from the correct recipe.
   No cross-lane cancellation exists: my b1/b2/b3 carry all four
   mechanisms simultaneously (stroke veto + fringe repair + specular
   reconcile + pale chips/bottom cap) at the paired numbers.

**Governance consequence (my population-statistics call, T2's mandate):
moot under proven determinism — there is nothing to take a median over.
The gate applies to THE SHIPPED ARTIFACT baked under the CANONICAL RECIPE,
which is hereby fixed as: front photo with `identity_image` = un-matted
input.png, `remove_background_robust` matte for the rgba, ±90 `_clean`
profiles as references, `texture_completion="auto"`, orthographic, 2048.
Artifact publication must verify both harnesses on the exact bytes before
overwriting the bundle (the md5 of a correct publish today: `c39d65bf…`).
If future work reintroduces nondeterminism, the gate applies to the
shipped bake and publication requires a same-recipe re-verification run.**

## T1b — LEDGER WALK (my crops: `c4_strokes/features/minors/col_band.png`, C2|C3|C4 columns)

Ship/owl first: on-disk md5s equal my C3-certified hashes (`b8e2b0d4`,
`ff746509` — untouched as designed); texture_qa re-run by me: **PASS 13/13
both**. All ship/owl closures stand unchanged.

| entry | C4 state | my evidence |
|---|---|---|
| **FACE-20 billboard strokes (the cycle's #1)** | **FIXED** | All four ruling sites dead on the ON-DISK battery: az0 6x — the jagged temple crack is gone (soft warm shadow band in its place); az−22.5 4x — no silhouette streak; az−90 3x — no ragged hairline line; az+90/+112.5 3x — no ear-helix arcs (`c4_strokes.png`, C3 column vs C4 column). The refill's soft gray-taupe residue reads as temple/ear shading at 2–3x, mottled film at 4x — same class and extent as the C3-accepted wisp residue, no new entry. Solver 1's provenance (veto consulted only inside the feature moat; strokes at moat 0.0 with veto consensus 0.7–1.0) is the honest root cause, and their 48-view sweep instrument now exists so this class cannot ship blind again. |
| FACE-05 pale seam column | **FIXED** | The provenance overturn is accepted — it was measured, not asserted: the column is the source photo's baked nose-ridge specular (photo lum 218.6 in-column vs 187.7 control, desaturated), present in the PRE-solve blend, zero membrane rails inside, winner share 1.00 front — S1 compositing exonerated by instrumented capture. The cross-view diffuse-consensus reconciliation removes it: my 6x az0 pair shows the pale band collapsed to near-skin uniformity (`c4_col_band.png` row 1); the legitimate ridge highlight is NOT flattened (checked against the photo at 6x — the ridge line survives); no new dark unmasking (the dark-content standoff was measured in after their first version tripped debris at 5 views). Philtrum/below-lip chips remain under FACE-04. |
| FACE-12 bust disc | **FIXED** | `tone_bottom_cap`: the tan marble wash is gone at el−20 2x; the disc reads as skin continuation at the front arc, hair at the rear (`c4_minors.png` row 1). Rim slivers were already committed in C3. The remaining flatness is the synthetic cut face itself (S4 geometry, inherent). |
| FACE-07 ear debris | **PROVEN-LIMIT (closed)** | `commit_pale_chips` (the dark-context dual, with the measured 1.2e-3 area cap) visibly reduces the ear-band chips at ±90/±112.5 (`c4_minors.png` rows 3–4). The measured remainder: a CONFIDENT witnessed subpopulation (w90 0.63–0.73 — real skin between strands, photo truth, untouchable under the witness contract I enforce everywhere) plus the ear-cluster parallax class granted since C2. Remedy: capture with ears exposed/hair tucked. |
| FACE-11 chest straps | **PROVEN-LIMIT (closed)** | The impossibility evidence is now complete: the strap is witnessed ONLY at the shoulder crests; its continuation domain carries ZERO witnesses in ANY view (front-projection coverage map black on the chest slab) onto a torso the mesh truncates. Synthesizing it = inventing photo-absent structure — the exact class FACE-20 just demonstrated as a regression source. Same family as SHIP-01/04/07. Remedy: a photo framing the chest. |
| FACE-13 crown | **OPEN (minor, geometry-only now)** | The mottle lever is closed by the ceiling experiment I ordered: clamping ALL pale crown texels barely changes the el60/80 read and dims the REAL parting (52% of the mottle is confidently witnessed scalp/parting content). PROVEN-TRADE accepted for the texture lever. The S4 mesh-flap silhouette remains open — no conservative-clamp bound (the FACE-08 precedent) has ever been filed for it. Remedy: mesh repair or crown photo. |
| SHIP-05 glow blob | **PROVEN-LIMIT (closed)** | Source-trace on an instrumented pin that reproduces the on-disk ship md5 EXACTLY: the zone is 97% fill, tone WITHIN its donors' consensus (0.95 ratio), bright from the harmonic stage, detail energy normal. The "glow" is macro-structure absence on a smoothly lit fill span — the single-photo far-side content family (SHIP-01/04/07). A tone-ceiling would fight legitimate interpolation. The ship texture was correctly left byte-identical. Remedy: port-side/rear photo. |
| FACE-03 under-eye/tear-duct | **OPEN (IMPROVED, cosmetic)** | Fringe repair at 2048 visibly reduces the tear-duct whites and lash-line dashes (`c4_features.png` rows 1–2, 5); the transplanted eye's interior is correctly untouched (FACE-15 grant). Remaining at 4x: faint fringe at the transplant ring + the 2048 mouth-complex formation shortfall (their documented follow-up). Not proven-limit — the 1024 arm proves the mechanism clears it when the complex forms fully. |
| FACE-04 mouth/chin/neck | **OPEN (IMPROVED, minor)** | Below-lip and chin chips further cleaned; the lip-edge dark-red dash is PARTIALLY reduced (visible at 4x, `c4_features.png` row 3); the neck/jaw tan wash persists (the 0.0102 gate-loss cluster, co-witnessed apron class). |
| FACE-09 rectangle residual | **OPEN (cosmetic, re-attributed to the band lane)** | Unchanged at az180 4x (`c4_minors.png` row 2). Solver 3's stage capture is accepted: the rectangle is created between surface-smooth and post-commit at 2048 — the film-band REPAINT's rear extent (81% fill darkened −11/255 with a straight boundary), absent at 1024 where the repaint no-ops. The comb-lane equalizers were correctly NOT shipped (both prototyped, both inherit the step). The band lane owns its repaint boundary. |
| FACE-14 identity[front] | **OPEN — see T2** | |
| **FACE-21 (NEW) artifact publication recipe** | **OPEN (minor, publication not pipeline)** | The on-disk bundle (md5 `5881359e`) was baked without `identity_image`: it fails raw MAE by the margin (22.04 > 22.0), carries the az−35 el10 debris marginal (0.0033, 7 islands — small dark specks at brow/lip/neck in my annotated evidence), and gives away comp 0.015 vs the same tip correctly baked (my b1/b2/b3: `c39d65bf`, all green but the SSIM gate). Fix: re-publish from the canonical recipe with pre-overwrite harness verification. |

Standing unchanged: FACE-01/02/10/15..19 FIXED (spot-checked: eyes 1/1 at
±90, no ghosts, pose +20/+8 at 0.0152 in every bake log); FACE-08 PL;
FACE-06 merged; SHIP-01/02/03/04/06/07/08, OWL-01..04 as ruled in C3.

New-defect hunt (stroke-veto refill edges, fringe-stamp borders,
specular-flattening of the real highlight, full 48-view grids): nothing
new found. The refill shadow and the temple film residue are the C3-
accepted classes at equal-or-lower visibility; no stamp borders at 4x;
the nose ridge highlight survives (photo-checked at 6x).

## T2 — FACE-14 RULING (against MY anchored comp gate 0.70 / 15.0)

- Measurement basis (justified above): the shipped artifact under the
  canonical recipe. Determinism verified (three bakes, one hash), so
  best-of vs median is moot; there is exactly one number per recipe.
- **Correct-recipe tip: comp 0.688 / 14.4 — MAE GREEN with margin, SSIM
  0.012 short of the floor. FACE-14 stays OPEN.**
- The remaining 0.012 is decomposed and REACHABLE (solver 2's gate-loss
  geography, which I accept as measured): neck/jaw wash 0.0102 (FACE-04's
  apron class) + hair-curtain right 0.0086 (band lane) + temple refill
  residue 0.0060 (band lane) + 2048 mouth-complex formation 0.0026 (fringe
  lane's documented follow-up) = ~0.027 of addressable budget, of which
  the gate needs 0.012. The granted-limit clusters (ear 0.0183, transplant
  disc interior 0.0047) are excluded from this arithmetic and NOT counted
  against any lane.
- **No re-anchoring.** My governance clause requires V20-quality evidence
  that the joint ceiling moved; none was filed (solver 2 explicitly
  declines to claim a ceiling, and their 1024 full-battery PASS at
  0.708/14.7 demonstrates the mechanism family's ceiling sits ABOVE the
  gate). The 0.70/15.0 comp floor stands.
- The raw battery remains diagnostic: raw MAE must stay ≤ 22.0 on the
  published artifact (the correct recipe gives 21.7; the wrong one gave
  22.04 — one more reason FACE-21 blocks).

## COUNTS (35 entries ever)

- **FIXED: 19** (FACE-01, -02, -05, -09 material, -10, -12, -15, -16,
  -17, -18, -19, -20; SHIP-02, -03, -06, -08; OWL-01, -02, -04)
- **PROVEN-LIMIT (closed, capture remedies documented): 8** (FACE-07,
  -08, -11; SHIP-01, -04, -05, -07; OWL-03)
- **MERGED: 1** (FACE-06 → FACE-14)
- **OPEN: 6** — FACE-14 (major, gate: comp 0.688 vs 0.70), FACE-21
  (minor, publication), FACE-04 (minor), FACE-03 (cosmetic), FACE-09
  rectangle (cosmetic), FACE-13 flaps (minor, S4).

**FAIL** under the zero-open standard — but for the first time the open
list contains no blocking or major VISUAL defect: one gate at 0.012, one
wrong-recipe publish, and three small residuals with named owners.

## CYCLE-5 ORDERS (ranked)

1. **FACE-21: re-publish the face artifact** from the canonical recipe
   (expected texture md5 `c39d65bf…` on this tree) with pre-overwrite
   verification: verdict1 raw (MAE ≤ 22.0, detectors green incl. az−35
   el10), comp report, texture_qa 13/13. Publication checklist goes into
   the repo docs; no bake without `identity_image` ships again.
2. **FACE-14 (+0.012 comp SSIM), two lanes in parallel:**
   a. Band lane: the temple refill residue (0.0060) and hair-curtain
   right (0.0086) — the refill tone can move toward the witnessed curtain
   content; this also owns the FACE-09 rectangle boundary (0.0 cosmetic
   but same mechanism, close both).
   b. Fringe lane: 2048 mouth-complex formation in world space
   (voxel-graph clustering per their own follow-up note; 0.0026 + the
   FACE-04 lip-edge dash closes with it).
   c. Compositing lane (if a+b land short): the neck/jaw apron wash
   (0.0102) — co-witnessed class; any general treatment must keep the
   witness contract.
3. **FACE-13**: file the conservative-clamp geometry bound for the crown
   flaps (FACE-08 precedent) or accept the capture remedy and close.
4. Hold everything else frozen: ship/owl bundles stay byte-identical
   (md5s in this ruling); any texturing.py change re-runs the canary
   md5 pairs.

## OWNER-FACING PARAGRAPH

Open the three files tonight and the starship and owl are exactly the
certified assets from the last ruling — untouched to the byte, both
passing every gate. The face took four real steps this cycle: the black
strokes that cycle 3 left on the temple and around the ears are gone at
every angle I rendered; the pale column beside the nose — which turned
out to be the photo's own baked-in shine, not a compositing error — has
been reconciled away without flattening the real highlight; the eye
corners and lash lines are visibly cleaner; and the underside of the bust
now reads as skin and hair instead of tan marble. What remains open is
thin: the front-identity score sits at 0.688 against my 0.70 floor with
the remaining 0.012 mapped to three named regions (neck wash, hair
curtain, one mouth residue), a faint rectangular tone patch on the upper
back at close zoom, and small eye-corner residue. One administrative
defect blocks everything: the file currently in the artifact directory
was baked with a subtly wrong recipe and measures below what this
pipeline actually produces — it must be re-published from the canonical
bake (the correct bytes are proven bit-reproducible; I baked them three
times and got the same hash). Nothing on the open list is proven
impossible; the capture-side limits (ears, chest strap, crown, ship far
side) are all documented with the exact photo that would remove each.

— Critic 1
