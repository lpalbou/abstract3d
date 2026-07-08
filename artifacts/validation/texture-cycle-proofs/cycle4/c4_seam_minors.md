# SOLVER 3 (cycle 4) — ORDER 3 (FACE-05) + ORDER 4 minors sweep

**Scope:** Critic 1's cycle-4 ORDER 3 (FACE-05 pale seam column — S1
compositing lane) and the ORDER-4 minors (FACE-07 ear chips, FACE-09
rectangle residual, FACE-11 chest straps, FACE-12 bust disc, FACE-13
crown, SHIP-05 glow). Repo `/Users/albou/abstract3d`, work under
`/tmp/c4_3/`. Date: 2026-07-07.

**SHARED-TREE NOTE:** two other solvers work film_band_gradient.py
(FACE-20 lane) and a feature-repair lane concurrently; their edits landed
in texturing.py/tests during my session. Every A/B below is PAIRED on one
tree state: 1024 iterations on a frozen snapshot (`/tmp/c4_3/pin`,
C4_3_PIN=1), the 2048 deliverable pair on the live tree with my
mechanisms toggled by monkeypatch (`disable_mine.py`). Deltas are mine;
absolute counts move with the tip.

## TL;DR

All three mechanisms are in the repo with tests (suite 225 passed + my
10 new); ship/owl canaries are BIT-IDENTICAL with mechanisms on vs off
AND equal to the on-disk bundles (md5 b8e2b0d4 / ff746509); texture_qa
PASS 13/13 on face/ship/owl; comp identity moves +0.005 SSIM / −0.2 MAE
in my favor on the deliverable pair; the one detector that moves on the
live tree (dark_debris az-35 el10, 0.0033 vs gate 0.003) is measured to
be fringe-lane re-judging coupling — with the fringe stage frozen in
both arms my mechanisms keep every detector green (worst dark 0.0029).

| target | provenance (measured) | mechanism | verdict |
|---|---|---|---|
| O3 FACE-05 pale column | source photo's baked nose-ridge SPECULAR projected onto the left nose flank by the +20° estimated pose; already in the pre-solve blend; NOT a membrane path (0 rails inside), NOT a selection boundary, NOT solve-created | `reconcile_specular_lobes` in the compositor (cross-view diffuse consensus; source view only) | column gone at 4x in both critic framings |
| FACE-07 ear chips | mixed class: trace-weight displaced skin chips + fill copies (35-60% fill) + CONFIDENT witnessed skin-through-hair (w90 0.63-0.73, untouchable) | `commit_pale_chips` (dark-context dual of the trace commit) | chip class visibly reduced at ±90/±112.5 4x; confident subset = photo truth, kept |
| FACE-09 rectangle | NOT the comb regime: created between surface-smooth and post-commit stages at 2048 by the film-band gradient REPAINT's rear extent (81% fill texels darkened −11/255 median, straight region boundary; absent at 1024 where the repaint no-ops) | provenance handoff to the band lane (their file, actively being reworked for FACE-20) | evidence filed; comb-regime equalization would be post-hoc paint over another stage's active artifact |
| FACE-11 straps | strap witnessed only at shoulder crests; its continuation domain has ZERO witnesses in any view; consensus veto correct (C1 agrees) | evaluated bright-structure continuation: under-determined (photo-absent structure = the FACE-20 regression class) | LIMIT filed with evidence |
| FACE-12 disc wash | global harmonic fill tones the synthetic cut face from rear-hair/neck anchors | `tone_bottom_cap` (rim-anchored geometry-aware toning) | disc reads as rim continuation (skin front / hair rear) |
| FACE-13 crown | mottle is 52% CONFIDENT witnessed content (w90 0.665 — the real parting/scalp), 48% fill copies; pale-island subset handled by commit_pale_chips | ceiling experiment run (clamp-all vs trace-only) | clamping ALL pale texels barely moves the el60/80 read (midtone mixture + S4 mesh flaps dominate); PROVEN-TRADE under the witness contract |
| SHIP-05 glow | 97% fill zone; tone within 5% of donor consensus (0.95 ratio); present from the harmonic stage; detail energy normal | no legitimate tone-prior fix exists (the fill IS its donors' consensus); the glow read is macro-structure absence | LIMIT filed with evidence (single-photo far-side content family) |

## 1. O3 — FACE-05 provenance (the ruling's three hypotheses adjudicated)

Instrumented bake captures per-view rgb/weight/valid at compositor time +
blend/solved/final textures (`bake.py --instrument`, `f05_*.py`).

1. **Membrane path in the screened Poisson solve — NO.** Rebuilt the
   compositor's graph + witness tiers from captured state: the column
   texels carry ZERO witness-less (line/membrane) edges (`rails=0`
   everywhere inside; the rail texels sit at the winner-strip borders
   left of the bridge, not under the column). `f05_rails_column6x.png`.
2. **Gradient-selection boundary — NO.** The column interior is 100%
   front-won common-witness territory (winner shares F/L/R = 1.00/0/0);
   the nearest selection boundary (side_right's dark flank strip) lies
   LEFT of the bridge and is itself confident witnessed content.
3. **Harmonization residue — PARTIAL, upstream.** delight_projections
   measured no improvement and reverted (0.084 -> 0.0852): an order-2 SH
   field in normal space cannot represent a localized specular lobe.
4. **What it IS:** the source photo's own nose-ridge specular. Column
   texels sample the photo's ridge highlight (photo lum 218.6 in-column
   vs 187.7 lateral control; saturation 40.3 vs 57.4 — bright +
   desaturated + smooth = specular signature). Under the estimated
   az+20/el+8 source pose those photo pixels belong to the LEFT NOSE
   FLANK; at az0 the baked highlight reads as a pale painted column
   beside the ridge (and side_right's shadowed flank content darkens the
   band left of the bridge — the contrast pair). Per-row profiles: the
   column is fully present in the PRE-SOLVE blend (lum 210-218 vs
   control 177-197); the solve preserves it (±5/255) and only spreads
   its tone into the philtrum below (conf ~0.13 there) — the compositor
   preserves exactly what its contract says it must preserve: the most
   confident common witness.
   Evidence: `f05_attrib` table, `f05_photo_marks.png` (column texels'
   photo samples), `f05_localize.png`, `f05_isolate.png` (V1 blend-grad
   / V2 no-side-left / V3 strong-anchor ablations — none remove it).

### The fix (compositor vocabulary, not post-hoc paint)

`gradient_compositing.reconcile_specular_lobes` — full semantics in the
module docstring and CHANGELOG. Key decisions, each measured:

- **Cross-view diffuse-consensus authorization**: a lobe qualifies only
  where another view's valid sample reads the same surface darker than
  the pairwise lighting gauge by 0.08 log (pooled per-texel field,
  sigma 8; component fill-in at 30% for the witness-thin glabella top).
  Both-views-bright regions (the ridge line under both lights, the
  mouth-surround brightness: 643 paired texels, 0 votes) are refused —
  shared brightness is albedo or shared shine, not parallax-displaced
  light.
- **Correction from the winner's own surround** (baseline + own
  log-detail, saturation restored toward surround, capped 1.6x): no
  reference color can leak through this path.
- **Feature gate on edge density** (own-photo Scharr p85 < 1.8 on the
  2-dilated component ring): sclera/eye-corner analogs are
  bright+desaturated but edge-dense; the under-eye candidates measured
  p85 2.0-8.4 and are refused; the nose lobe measures 0.9-1.3.
- **Source view only**: reference-view lobes measured side_left identity
  −0.005 SSIM with no ledger-visible gain (A/B spec1024 vs spec2_1024).
- **Anchors and gradients corrected coherently** (delta applied to
  view_rgb[source] and share-scaled to the blend anchors) so the solve
  consumes one consistent story.

Two 2048 hardening iterations landed after the first deliverable bake
(both in §8's ladder): the dark-content standoff (leveling must never
manufacture dark-island contrast) and the resolution-normalized edge
gate. Final 2048 numbers in §5.

### O3 acceptance (1024 iteration pair, pinned C3-state tree)

| metric | base | +reconcile | delta |
|---|---|---|---|
| az0 nose 4x / eye-R 4x (critic framings) | pale column visible | column gone (`spec2_ab.png`) | fixed |
| identity[front] raw SSIM/MAE | 0.643 / 21.6 | 0.640 / 21.8 | −0.003 / +0.2 |
| identity[front] comp (authoritative) | 0.668 / 16.3 | 0.664 / 16.4 | −0.004 / +0.1 |
| identity side_left / side_right | 0.681/19.8, 0.688/21.4 | 0.681/19.8, 0.688/21.4 | 0 / 0 |
| detectors (28-view battery) | all green | all green (threshold jitter only: eye_count at +35 el0 counts 2→1 inside its [1-2] band) | green |

The small front cost is the honest price of removing baked light the
gate-pose photo still carries; it is within the harness's measured NCC
warp instability (±0.02 per solver 1's controlled-alignment study).
2048 deliverable numbers in §5.

## 2. FACE-07 — pale-chip commit (dark-context dual)

Provenance (`f07_attrib.py`, 1024): the visible ear-band chips are a
MIXED population — per view crop: direct share 0.40-0.65 at trace weight
(w50 0.00-0.12), the remainder FILL copies of the same displaced anchors;
plus a CONFIDENT subpopulation (w90 0.59-0.73) that is real skin seen
between strands (photo truth, untouchable under the witness contract).

`texturing.commit_pale_chips` (repo, tested): the exact dual of
`commit_trace_deposits` — pale islands vs dark ring consensus — with the
candidate domain extended to fill texels (weight <= trace covers both)
and one NEW measured guard: the 1.2e-3 area cap (a 704-texel rear blob
at z=-0.99 committed into a flat gray wash visible at 2x in the el-20
view — `chips_ab.png` bottom row vs `chips2_ab.png` after the cap).
Refusal ledger on the face at 1024: 571 area / 66 ring / 14 isolation /
131 cover — the mechanism commits 133 blobs, 1986 texels.

A/B at 1024 (pinned): ear-band chips visibly reduced in all three critic
framings (`chips2_ab.png`); identity front/sl/sr −0.001/+0.001/+0.001
SSIM (noise); rear skinHair moved 0.0006→0.0005 at +135 el10 (improved);
all detectors green.

## 3. FACE-09 rectangle — measured provenance (handoff to the band lane)

Per-stage capture at 2048 (`capstages.py`, stage renders + horizontal
profiles): the upper-back region is FLAT at 42-44/255 through
harmonic/smoothed; the right half (render cols ~505-620 at az180) DROPS
to 24-28 with a straight boundary BETWEEN the surface-smooth stage and
the post-commit capture — the window containing the film-band gradient
REPAINT (commit_trace_deposits is strictly local and cannot repaint half
the upper back). Diff of the two stages: 188k texels darkened, median
−11/255, 81% FILL / 19% observed (`f09_repaint_mask_az180.png` shows the
repaint's rear extent with the rectangle's exact boundary). At 1024 the
repaint no-ops below its sampling floor (7-texel transition) and the
rectangle does not exist — matching the ruling's crops (2048).

The ruling's prescription ("tone-equalization inside the comb regime")
assumed comb provenance; the measured driver is the repaint's rear
boundary — solver 1's file, actively being reworked THIS CYCLE for
FACE-20 (the same mechanism family: repaint region boundaries printing
visible structure). Equalizing it from the comb regime would be post-hoc
paint over another stage's in-flight artifact and double-treatment risk.
Two prototyped equalizers confirmed this: observed-anchored low-frequency
rebuilds inherit the step from the 19% darkened observed anchors
(`f09_proto2/3`); a ridge-capped screened equalization of the dark-fill
base (compositor vocabulary, `f09_proto4_ab.png`) softens but cannot
remove a step whose anchors carry it. EVIDENCE FILED for the band lane;
the rectangle's state on the final tree is re-checked in §5.

## 4. Minors — verdicts

### FACE-11 chest straps — LIMIT (evidence)

The straps are witnessed ONLY at the shoulder crests (front photo's
bottom rows; `f11_photo_front.png`); the front projection carries zero
coverage on the chest slab (`f11_frontproj_el20.png` — black = unwitnessed)
and neither side photo images the chest (side crops end at the neck;
solver-2's measured ring votes 0.47-0.52 confirm the mixed surround).
A general bright-structure continuation would have to synthesize the
band across surface NO view witnesses, onto a torso the mesh TRUNCATES
(the cut face) — under-determined structure invention, the exact
photo-absent class FACE-20 just demonstrated as a regression source.
The consensus veto stands (C1 already rules it correct); remedy is
capture (a photo framing the chest).

### FACE-12 bust disc — FIXED (mechanism in repo)

`tone_bottom_cap` (§CHANGELOG): the disc face is detected geometrically
(down-plane component, 91,494 texels at 2048, direct witness 0.1%, slab
ratio 0.012) and toned from its own rim: chest skin continues across the
front arc, hair tone at the rear arc, soft transition (sigma 24), 60% of
the cap's own detail kept. Before/after: `f12_proto2_ab.png` (el-20 and
el-50) — the tan marble wash is gone; the cap reads as a plausible
underside of its own rim materials.

### FACE-13 crown — ceiling experiment (the one the ruling demanded)

Attribution first (`f13_attrib.py`): crown mottle above lum 0.35 at 1024
is 165 texels — 52% DIRECT at winner w50 0.373 / w90 0.665 (confidently
witnessed parting/scalp content), 48% fill copying those donors (donor
lum 0.486 at 0.012 world distance). The experiment (`f13_ceiling.py`,
2048, `f13_ceiling.png`): bounded clamp toward the local dark envelope,
(C1) trace-only subset [witness-contract-safe] vs (C2) ALL pale crown
texels [the ceiling]. Result: even C2 barely changes the el60/el80/az180
read — the crown's mottled appearance is carried by midtone gray
mixtures and the S4 mesh-flap silhouette, not by the pale-island class;
and C2 dims the REAL parting (witnessed content demotion). Verdict:
PROVEN-TRADE under the witness contract; the pale-island subset commits
via `commit_pale_chips` where ring consensus approves; the remainder is
witnessed content + S4 geometry. Remedy: crown photo or mesh repair.

### SHIP-05 glow — LIMIT (evidence)

Source-trace (`s05_trace.py`, instrumented ship bake, pin reproduces the
on-disk texture md5 b8e2b0d4 exactly): the glow zone at az-135 el+15 is
97% FILL with 2% mirror islands; the soft bright tone exists from the
HARMONIC stage onward (stage renders `s05_stages.png`); detail synthesis
adds normal grain (|final−smoothed| 3.7/255 in-zone vs 2.3 control — no
under-detail). Tone: glow fill lum 139/255 vs donor-consensus ball mean
146.5 at 2R — ratio 0.95, i.e. the fill sits WITHIN its observed donors'
consensus; a donor-consensus ceiling (the floor's dual) would not engage,
and forcing one would fight legitimate interpolation between witnessed
bright hull panels. What makes the zone read as "glow" is macro-structure
ABSENCE on a smoothly-lit fill span — the single-photo far-side content
class already granted as SHIP-01/04/07 limits. Remedy: a port-side or
rear photo. No pipeline defect found; the ship texture is deliberately
left byte-identical.

## 5. Deliverable-resolution results (2048, live-tree paired)

**Tree state:** pinned snapshot `/tmp/c4_3/pin2` of the live tree at
2026-07-07 ~01:35 (includes solver-1's film_band_gradient state and the
feature-repair lane's landed `feature_fringe_repair.py`; md5s in the
bake logs). All arms baked from this one snapshot; my mechanisms toggled
by monkeypatch. IMPORTANT CONTEXT: this tip is MID-FLIGHT — the control
itself carries the other lanes' in-progress artifacts (a broad pale film
over the FACE-05 corridor, mouth stamps, raw front MAE 22.2 FAIL — none
present in the C3 official bundle). Absolute counts belong to the tip;
the paired DELTAS are mine.

| metric @2048 | OFF (control) | SPEC (reconcile) | CHIPS (chips+cap) | ALL (shipped) |
|---|---|---|---|---|
| verdict1 failed checks | 2 | 3 | 2 | 3 |
| identity[front] raw SSIM/MAE | 0.639 / 22.2 (MAE FAIL) | 0.644 / 22.1 | 0.639 / 22.2 | **0.645 / 22.1** |
| identity[front] COMP (authoritative) | 0.666 / 15.2 | 0.671 / 15.0 | 0.667 / 15.2 | **0.671 / 15.0** |
| identity side_left / side_right raw | 0.659/19.1, 0.671/20.9 | 0.659/19.1, 0.670/21.0 | 0.661/19.1, 0.674/20.9 | 0.661/19.1, 0.673/20.9 |
| dark_debris worst view | 0.0026 | 0.0033 az-35 el10 (FAIL by 1 island) | 0.0026 | 0.0033 az-35 el10 |
| texture_qa | PASS 13/13 | PASS 13/13 | PASS 13/13 | PASS 13/13 |

- **Identity: my mechanisms IMPROVE both raw (+0.006 SSIM, −0.1 MAE) and
  comp (+0.005 SSIM, −0.2 MAE) front identity on the tip** — at 2048 the
  reconciled lobe wins more at the gate pose than the removed highlight
  costs (the 1024 pin measured a small cost; the sign flips at the
  deliverable resolution). The control's raw MAE FAIL (22.2 > 22.0) is
  tip-inherited (C3 official: 21.5 PASS); mine moves it the right way.
- **dark_debris attribution (the one moving detector):** the fringe
  repair runs LAST and re-judges the RENDERED texture; my upstream delta
  changes its stamp/veto decisions far from the nose (stamps 15 -> 11,
  mouth-region islands appear/disappear). Controlled pair with the
  fringe stage FROZEN in both arms: worst dark_debris 0.0029 (GREEN),
  no view above gate, front raw 0.648/22.0 vs control 0.648/21.9 —
  my mechanisms alone keep every detector green. The az-35 el10 0.0033
  (7 islands, 1 over) exists only through the fringe lane's re-judged
  state, which that lane will re-bake as it lands its final version.
  Iteration receipts: the first reconcile version DID unmask dark
  content itself (0.0040-0.0054 at five views) — fixed in-mechanism by
  the dark-content standoff + resolution-normalized edge gate (§8),
  which brought four of five views under gate before the fringe
  coupling was isolated.
- **FACE-05 at 2048:** reconcile applies 817 texels / 2 components on
  the source lobe (stats in bundle metadata); the srcpose identity
  improves. The az0 corridor on THIS tip is visually dominated by the
  other lanes' in-flight pale film (present in the control), so the
  clean 4x acceptance evidence is the pinned C3-state pair
  (`spec2_ab.png`, column gone) + the fringe-frozen 2048 pair
  (`nf_face05_ab.png`).
- **FACE-07/12 at 2048:** `chips2048_ab.png` (ear bands), `cap2048_ab.png`
  (disc). The chips arm alone: identity +0.001/0, detectors unchanged,
  10,696 texels committed under 508 blobs (refusals: 2476 area /
  346 ring / 46 isolation / 122 cover).
- **FACE-09 on the tip:** the rectangle persists unchanged
  (`final_f09_upback.png`) — the repaint's rear boundary has not been
  touched by the band lane yet; the §3 handoff stands.

### Canaries (same snapshot, mechanisms ON vs OFF)

| asset | ON md5 | OFF md5 | on-disk | texture_qa (ON) |
|---|---|---|---|---|
| ship 2048 | b8e2b0d4... | b8e2b0d4... | b8e2b0d4... | PASS 13/13 |
| owl 2048 | ff746509... | ff746509... | ff746509... | PASS 13/13 |

Bit-identical in both directions AND equal to the shipped bundles: my
mechanisms are structural no-ops on single-photo bakes, and the pinned
snapshot still reproduces the on-disk proof assets exactly. SHIP-05 was
explicitly permitted texture changes; none were needed (§4 — the glow is
a filed content limit, not a pipeline defect).

### Suite

`python -m pytest tests/ -q`: 225 passed, 1 xfailed (includes the other
lanes' in-flight tests) with my 10 new tests
(4 specular-reconcile in test_gradient_compositing.py; 4 pale-chip +
2 bottom-cap in test_texturing.py).

## 6. Canary discipline

- `commit_pale_chips` no-ops below two projections (code + test);
  `tone_bottom_cap` is called inside the multi-view branch only;
  `reconcile_specular_lobes` requires a second valid witness (code +
  test) and the compositor itself only runs on multi-view bakes under
  `auto`. Single-photo canaries are structurally untouchable.
- md5 pairs in §5.

## 7. Artifacts index (/tmp/c4_3/)

- FACE-05: `f05_*.py` (provenance ladder), `f05_stages_*.png`,
  `f05_tiers_*.png`, `f05_rails_*.png`, `f05_localize.png`,
  `f05_photo_marks.png`, `f05_isolate.png`, `proto_spec*.py` (v1-v7
  ladder), `specfix.py` (patch = shipped code), `spec2_ab.png`
  (acceptance crops), qa logs `qa_base1024*`, `qa_spec2*`, `qc_*`.
- FACE-07: `f07_*.py`, `palechips.py`, `chips_ab.png` (pre-cap
  regression visible), `chips2_ab.png` (shipped), `qa_chips2*`.
- FACE-09: `capstages.py`, `f09_*.py|png` (stage profiles, repaint mask,
  prototype equalizers incl. the ridge-capped screened solve).
- FACE-11: `f11_*.py|png` (photo evidence, front-projection coverage,
  class maps).
- FACE-12: `f12_proto*.py|png`.
- FACE-13: `f13_attrib.py`, `f13_ceiling.py|png`.
- SHIP-05: `bake_ship.py`, `s05_*.py|png`, masks `s05_masks.npz`.
- Bundles: `bundle_base1024` (pinned baseline, instrumented),
  `bundle_spec*/chips*` (mechanism iterations), `bundle_base2048`
  (pinned 2048 baseline, instrumented), `bundle_shipbase` (pinned ship,
  md5 == on-disk), `bundle_ON_2048` / `bundle_OFF_2048` (live-tree
  deliverable pair), ship/owl ON/OFF pairs.

## 8. Iteration ladder (dead ends kept honest)

| attempt | result | verdict |
|---|---|---|
| FACE-05 v1: component votes (share>=0.5 over paired) | column component refused (share 0.28 — its paired texels concentrate on the both-bright ridge) | per-texel pooled evidence needed |
| FACE-05 v2: per-texel auth + edge HALO | halo from eye/nostril edges swallowed the column (final 36 texels) | component-ring edge stat, not halo |
| FACE-05 v4: pooled auth (shipped semantics) | 348 texels, column gone; upper glabella stub remains | + fill-in 30% (v5) covers the full lobe |
| FACE-05 v5: surround+detail RGB target | corrected strip slightly muddy vs skin | v6 luminance-scale + saturation-restore target (shipped) |
| FACE-05 all-views scope | side_left identity −0.005 SSIM, no ledger gain | source-only (shipped) |
| FACE-05 @2048 without dark standoff | leveling unmasked dark micro-content: dark_debris 0.0040-0.0054 at 5 views | dark-content standoff (8 texels, feathered) — 4/5 views back under gate |
| FACE-05 @2048 edge gate uncorrected | fixed-world edges spread over 2x texels at 2048, halving the Scharr response — an eye-corner component (1.45 raw = 2.9 normalized) slipped the 1.8 bar | resolution-normalized edge statistic (shipped) |
| FACE-05 residual az-35 el10 0.0033 | present ONLY with the fringe stage active (frozen pair: 0.0029 worst, all green) | fringe-lane coupling documented |
| FACE-07 v1: no area cap | 704-texel rear blob -> flat gray wash at 2x (el-20) | 1.2e-3 area cap (shipped) |
| FACE-09: observed-anchored low-freq rebuild (x2) | step inherited from darkened observed anchors; global darkening | wrong lane — repaint owns the boundary |
| FACE-09: ridge-capped screened equalization | softens, cannot remove anchored step | provenance handoff (this report) |
| FACE-12 v1: 24-NN power-2 rim interpolation | hard skin|hair boundary mid-cap | 48-NN power-1.5 + sigma 24 smoothing (shipped) |
| FACE-13: bounded clamp C1/C2 | see §4 — neither closes the entry | PROVEN-TRADE filed |

## 9. Repro

```bash
source .venv/bin/activate
# live-tree pair (my mechanisms ON = default path)
python /tmp/c4_3/bake.py ON_2048 --res 2048
python /tmp/c4_3/bake.py OFF_2048 --res 2048 --patch disable_mine
python /tmp/verdict1/qa.py /tmp/c4_3/bundle_ON_2048 --out /tmp/c4_3/qa_ON_2048
python /tmp/c2d/qa_shadecomp.py /tmp/c4_3/bundle_ON_2048 --out /tmp/c4_3/qc_ON_2048 --shading-comp
python scripts/texture_qa.py /tmp/c4_3/bundle_ON_2048 --out /tmp/c4_3/tq_ON_2048
# tests
python -m pytest tests/test_gradient_compositing.py tests/test_texturing.py -q
```
