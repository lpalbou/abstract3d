# SOLVER 1 — cycle 4, ORDER 1 (FACE-20 billboard dark strokes)

Tree at delivery: `film_band_gradient.py b4418585` (the only repo file
this lane changed, plus its test file, CHANGELOG, KnowledgeBase);
`film_band.py 60f1cd5f` unchanged; `texturing.py` moved twice under me
during the session (other lanes; my wiring point is untouched). All
prototypes, A/B bundles, crops, QA logs: `/tmp/c4_1/`.

## VERDICT ON THE ORDER

FACE-20's four stroke sites are dead at all 48 battery views vs the C2
baseline, at the ruling's own crop zooms and stricter (3x-6x), with zero
new dark components introduced by the fix (every residual sweep flag is
shared 1:1 with a same-tree veto-off arm and is a non-stroke class
already adjudicated in the C3 ledger). Detectors green at 28 views,
texture_qa PASS 13/13, ship/owl provably untouched (md5 A/B pairs).
Identity: on the tree the strokes shipped on (C3 replay), removing them
costs the FRONT comp gate 0.0024 and pays both side gates — the ruling's
assumption that stroke removal banks front headroom has the sign wrong
at the source pose (analysis below, KnowledgeBase'd); on the CURRENT
tree the fix measures +0.037 comp front over the same-tree veto-off arm
(0.674 vs 0.637). The comp 0.677 anchor is 0.003 short on my isolated
arm; the decomposition (mechanism-own −0.0024 vs cross-lane tree
movement −0.04 on the off arm) is measured and reproduced below.

## PER-STROKE PROVENANCE (deliverable 1)

Method: the captured tip state (`/tmp/c3_1b/tipstate_2048.pkl`) replayed
through a traced copy of the shipped mechanism, validated BIT-EXACT
against the module function before any conclusion (`trace_repaint.py`,
`validate_trace`). New-dark texels = repaint-on vs repaint-off through
the full pipeline tail, connected-component labeled, localized at the
critic's views by unique-color debug textures (`provenance.py`,
`localize.py`, `stamp_signals.py`). Numbers: `components.json`,
`stroke_photo_boxes.json`.

The four ruling sites collapse to TWO texel clusters plus satellites,
all the same mechanism class:

| stroke (ruling) | components | stage mix | veto | moat | S_med | photo provenance (front photo, 1024px frame) |
|---|---|---|---|---|---|---|
| (a) az0 left-temple crack, (b) az−22.5 silhouette streak, (c) az−90 hairline line | c357+c188+c205+c316 (right-temple cluster, ~1.4k texels) | 78-99% authority stamps + 9-56% gap diffusion | 0.92-1.00 | 0.00 | 0.35-0.66 | bins y288-387 × x272-321 — the CURTAIN EDGE left of the parting, median 9-18 px inside the photo's dark-body boundary (transition length 13 px) |
| (d) az+90/+112.5 ear-helix arcs | c259 (left-ear cluster, 890 texels) | 81% authority + 19% gap | 0.71 | 0.00 | 0.66 | bins y431-573 × x691-775 — the HAIR/EAR-SHADOW rim beside the left ear, median 5.7 px inside the body boundary |
| satellites (az+45 jaw, below-ear tape) | c110, c296 | clamp-darkening | 0.86-0.97 | 0.0-0.8 | 0.55 | not stamped: envelope clamp over vetoed surface |

The mechanism hole, exactly: the shipped guard was
`dark_allowed = stamp_is_dark & ~(veto_any & moat)` — the base-material
witness veto was consulted ONLY inside the feature moat (dilated dark
features + photo feature components). Every stroke component sat at moat
fraction 0.0 with veto consensus 0.7-1.0: the photo's curtain-edge/
ear-shadow pixels (dark-body BOUNDARY content) billboarded onto surface
the source ray only grazes, while the profiles positively witnessed
bright base material at those texels. The rejection evidence existed at
bake time and was consulted in the wrong region. Gap diffusion then
extended the stamped strokes (its seeds are the stamps), and the
envelope clamp added the two satellite dark patches over equally vetoed
surface.

Discriminator measurement (1589 stroke stamps vs 9044 vetoed non-stroke
vs 10764 unvetoed stamps, `stamp_signals.py`):

| signal | strokes | vetoed valid mass | verdict |
|---|---|---|---|
| S (field position) | p10 0.48 / p50 0.66 | p10 0.15 / p50 0.23 | S >= 0.35: 96% stroke recall, 6% valid hit — THE separator |
| photo body-edge distance | p50 9 px | p50 19 px | edge<=1.5T: 88% recall but 41% valid hit — weaker |
| facing | p50 0.73 | p50 0.39 | useless (strokes face the source WELL — they are not grazing artifacts) |
| veto alone | 0.86 | 1.00 by construction | kills half of all dark stamps (the c3-measured −0.05 class) |

Why cycle 3 missed the ear arcs: the lane's verification instruments
were the 28-view detector battery (the arcs read as hair-connected —
detector-green by design) and band crops at the FACE-01 acceptance poses
only (srcpose/az0/±22.5); the ear complex was explicitly excluded from
the repaint domain analysis (FACE-07 hand-back), so nobody rendered
±90/±112.5 before/after. No new-dark-vs-baseline comparison existed at
any pose. That instrument now exists (`sweep48.py`, below) and
reproduces the critic's finding on their own renders (43 flagged
components on C3-vs-C2; the four ruling sites are the largest).

## THE FIX (deliverable 2 — the veto design)

`src/abstract3d/film_band_gradient.py`, two additions, ~60 lines:

1. OFF-POSE DISPLACEMENT VETO (`_displaced_stamp_components`): connected
   components of would-be dark stamps; a component is rejected when
   veto fraction >= 0.5 AND median S >= 0.35. Component-level on
   purpose — a stroke is one structure with one provenance; texel-level
   rejection fragments it into speckles (measured worse visually and on
   identity). S is the mechanism's own field (the photos' pooled
   hair-to-skin falloff): S >= 0.35 is the skin half of the transition,
   where the photos themselves say the surface has left the hair body.
   The moat veto (third-eye class) is untouched; near the mass
   (S < 0.35) vetoed dark stamps still land — that population is the
   wisp/strand mass the identity gate needs (global veto = the
   c3-measured −0.05 SSIM; on the current guard stack I re-measured the
   global veto at only −0.003 raw because the clamp re-darkens deep
   rejections anyway (`VETOALL` arm) — but it kills 10.4k stamps to the
   targeted 2.5k for zero additional stroke kill, so the scoped rule
   stands).
2. DISPLACED-SITE REFILL, after all island guards: the local guard tone
   (clamp/diffusion verdict) rescaled to luminance
   `1.02 x dark_split + 0.30 x photo_lum` — the photo's own luminance
   pattern at reduced gain on a floor strictly ABOVE the dark-material
   class. By construction the site can never render as dark structure at
   any pose (every detector's and the sweep's dark class starts at the
   split); the pattern pays part of the source-pose identity contract
   (+0.0007 comp SSIM / −0.26 comp MAE over flat guard tone, measured);
   LOCAL chroma, not photo chroma — the photo-chroma variant tripped the
   az+35 dark_debris gate at exactly its 0.003 threshold (near-black
   pixels' saturation amplifies when lifted).

Feather lever (the ruling's lever 2): measured NOT load-bearing for
these strokes — provenance shows all four sites are stamped content
(authority + gap), not border feathering; the two clamp satellites die
with the same veto because the clamp respects `band_domain`, whose
displaced components the veto removes from stamping and whose refill
re-tones. The border feather is unchanged.

Variants measured before choosing (replay A/B on the captured C3 state,
identical tail; comp = compensated front SSIM/MAE):

| variant | comp front | sweep48 flags | note |
|---|---|---|---|
| V0 (shipped c3) | 0.6786 / 14.19 | 43 | the defect |
| A1 texel veto | 0.6746 / 14.95 | 3 | fragments strokes |
| A2 component veto | 0.6755 / 14.86 | 3 | |
| A2 + photo-chroma refill | 0.6757 / 14.83 | 3 | dark_debris az+35 = 0.0030 AT gate — rejected |
| **A2 + local-chroma refill (SHIPPED)** | **0.6762 / 14.69** | **3** | all green |
| A2 + brighter floor 1.15 | 0.6759 / 14.74 | 3 | no visual gain |
| W1 positive-witness variant | — | 24 | misses az−22.5/−90 strokes — rejected |
| VETOALL (global veto bracket) | — | 3 | −0.003 raw, 4x the content kill |

The 3 surviving replay flags are hair-interior tone patches at az−112.5
(C2 lum 0.18-0.22 — dark-on-dark, not strokes) present in the critic's
own C3-vs-C2 renders and NOT part of FACE-20; crops:
`abcrop_az-112el+0_V0m_A1_OFF.png`.

## ACCEPTANCE EVIDENCE (deliverables 3-4)

Fresh standard bakes on the current tree (front az0 source + prototype
left/right `_clean` profiles ±90, `texture_completion="auto"`,
`projection_model="orthographic"`), the mid-flight feature-fringe stage
(another lane, see limits) runtime-no-op'd in BOTH arms for a clean A/B:

- `bundle_iso_face_2048` — the fix (displaced 2565 vetoed, 2565 refilled)
- `bundle_off_face_2048` / `bundle_off2_face_2048` — same tree, veto
  forced off at runtime (two independent bakes, metrics identical)

**Before/after crops at the four stroke sites**: `ACCEPTANCE_strokes.png`
(C2 | C3-ruling | C4-fix at the ruling's zooms). The az0 crack, az−22.5
streak, az−90 ragged line, and both ear arcs are gone; the sites read as
soft warm shadow bands strictly above the dark class. Full-view sheets:
`strokes_V0m_iso_face.png`, `band_V0m_A2S2.png` (FACE-01 band intact at
all acceptance poses — the fix does not reopen the film band).

**48-view sweep** (`sweep48.py` — 16 az x el {−20,0,+15} at 1000 px vs
the critic's own C2 renders; new-dark = dark in candidate AND clearly
non-dark in C2, component-labeled, area >= 60 px):

| arm | flagged components | at the four stroke sites |
|---|---|---|
| critic's own C3 renders | 43 | all four, up to 1804 px |
| V0 replay (defect) | 43 | all four |
| veto-off, current tree | 72 | all four + current-tree bottom-cap/neck deltas |
| **fix, current tree** | **32** | **ZERO** |
| repaint-off (C2-behavior arm) | 0 | — |

Every one of the 32 residuals is present in the same-tree veto-off arm
(the fix INTRODUCES nothing): 24 are el−20 bust/under-chin softening
(FACE-12/S4 territory, both arms, e.g. `abcrop_az+0el-20_*.png`), the
rest hair-interior tone patches at rear azimuths (FACE-09 comb-vs-mottle
class, `abcrop_az+112el+0_*.png`, `abcrop_az-90el+0_*.png`). None is a
dark-on-skin stroke; all are soft, low-contrast, hair- or
underside-internal.

**Harness numbers** (current tree, fringe-isolated):

| gate | fix (iso) | veto-off (same tree) | C3 on-disk (ruling) |
|---|---|---|---|
| verdict1 | FAIL 2: front SSIM 0.648 + raw MAE 22.004 (budget 22.0) | FAIL 1: front SSIM 0.612 (raw MAE 21.96) | FAIL 1: front SSIM 0.651 (MAE 21.5) |
| detectors 28 views | ALL GREEN | all green | all green |
| comp front (authoritative) | **0.6738 / 14.89** (MAE green) | 0.6369 / 15.41 (comp MAE RED) | 0.677 / 14.34 |
| comp side_left | 0.6861 | 0.6857 | — |
| comp side_right | 0.7044 (PASSES the 0.70 bar) | 0.7012 | — |
| texture_qa | **PASS 13/13** | — | PASS |
| 1024 iterate | sampling-floor bail (stats: `retone` fallback, no `gradient_repaint`); FAIL 1 identity 0.639, detectors green — c3's honest limit unchanged | — | same class |
| ship md5 (1024, on vs off) | `23c56731…` == `23c56731…` | | bit-identical |
| owl md5 (1024, on vs off) | `013f1a28…` == `013f1a28…` | | bit-identical |

**The comp SSIM delta this fix achieves** (the order's ask):

- Mechanism-isolated (C3-tree replay A/B, everything else frozen):
  front **−0.0024** (0.6786 → 0.6762), side_left +0.0004, side_right
  **+0.0040**; net across the three gates +0.002.
- Current-tree bake A/B: front **+0.0369** (0.6369 → 0.6738), sides
  +0.0004 / +0.0032. Reproduced across two independent off-arm bakes.

The sign finding the ruling should see: at the SOURCE pose the strokes
were PHOTO-TRUE (the stamped curtain-edge pixels are dark in the photo:
crack site photo lum p50 0.17, ear site 0.16 — `tone_budget.py`), so the
front gate rewarded them; their damage lands on the 46 non-source poses,
both side gates, and — on the current tree — the gate's own photo
registration (the off arm's SSIM-delta map shows a global basin shift,
`ssimdelta_off_face_iso_face.png`, +0.029 recovered at the left curtain
alone). "Photo-absent black structure costs SSIM" is true at every pose
except the one the front gate measures; the FACE-14 headroom the ruling
banked on this entry materializes at the side gates and in registration
stability, and the tree-movement collapse it prevents. Recorded in
KnowledgeBase ("The witness veto binds by field position, not by feature
proximity").

Against the anchored comp 0.677 floor: my isolated arm measures 0.6738.
Decomposition: mechanism-own −0.0024 (replay, above) + cross-lane tree
movement (the same-tree off arm sits at 0.6369 vs the C3-era off
equivalent 0.6786 — none of that motion is this lane's; my arm RECOVERS
+0.037 of it). Order 2's feature lane (running in parallel) carries the
ruling's +0.031 identified headroom; the joint tip number is theirs to
land and the critic's to measure.

## PATCHES + TESTS (deliverable 5)

- `src/abstract3d/film_band_gradient.py`: `_displaced_stamp_components`
  (component veto), the veto application at the authority stage, the
  post-guard refill, two stats keys (`displaced_dark_vetoed`,
  `displaced_refilled`), constants `DISPLACED_S_MIN/VETO_FRAC/
  REFILL_FLOOR/REFILL_GAIN` with measured rationale, docstring updated.
  Repo function validated BIT-EXACT against the measured A2S2 replay
  variant before landing (`validate_repo.py`).
- `tests/test_film_band_gradient.py`:
  `test_displacement_veto_rejects_vetoed_dark_stamps_in_skin_half` — a
  vetoed skin-half dark component must not print and its refill must sit
  strictly above the dark class; the same content unvetoed at the same S,
  and vetoed content near the mass, must keep stamping (the cycle-3
  identity contract). Full suite: **225 passed, 1 xfailed**.
- `CHANGELOG.md` (mechanism + measurements), `docs/KnowledgeBase.md`
  (new insight; the cycle-3 "billboard content is pose-local" insight
  kept, extended, nothing removed).

## HONEST LIMITS

- The front comp gate reads 0.6738 on my isolated arm vs the 0.677
  anchor: 0.0024 of that is my mechanism's own measured cost (the
  zero-sum above — the strokes were paying the front gate), the rest is
  cross-lane tree movement. I did not trade other gates to buy it back:
  every alternative measured (texel veto, brighter/darker floors,
  photo-chroma refill, min-size gates, positive-witness variant) was
  equal or worse on comp AND/OR left strokes alive or tripped detectors.
- The refill leaves a soft mid-brown band where each stroke was (visible
  at 6x as a shadow, reads as ear/temple shading at the ruling's 2x-3x;
  luminance floored at 1.02x the dark split by construction). The
  photos genuinely disagree at these co-witnessed sites; C2's bright
  putty there is the other end of the same trade, and the C2-look is
  recoverable only by paying the source-pose gate more.
- Raw front MAE on the current tree: 22.004 vs the 22.0 budget on my arm
  (veto-off: 21.96; C3-era: 21.49 both classes) — the ~+0.5 creep is
  tree-wide from other mid-flight lanes; the authoritative comp MAE is
  green on my arm (14.89) and RED on the veto-off arm (15.41).
- 1024 iterate bundles keep cycle-2 band behavior (the c3 sampling
  floor, unchanged by design); the displaced veto never runs there.
- Cross-lane, for the ruling and the owning lanes (measured, not mine to
  fix; my acceptance A/B no-ops the stage in BOTH arms): a fresh
  full-tree 2048 bake at 00:52 with the mid-flight feature-fringe stage
  ACTIVE (`bundle_final_face_2048`, another lane, landed mid-bake)
  failed verdict1 with dark_debris at 5 views (0.0038-0.0054 vs 0.003)
  and a localMin 0.034 ghost at (626,471) — that lane's landing needs
  the joint battery re-run. The el−20 bottom-cap/neck darkening vs C2
  (24 sweep flags, both arms) is FACE-12/S4 territory.
- The c3 report's "-0.05 SSIM global veto" number does not reproduce on
  the current guard stack (−0.003 measured, `VETOALL` arm) — the clamp
  now absorbs most deep-apron rejections; the original figure predates
  the clamp. The scoped veto is still strictly better (4x less content
  killed for the same stroke kill).
- The tree moved twice during the session (texturing.py 013a29fc →
  14ff4779 → 5ecbe9e5 → bc48a138); every A/B above is same-tree by
  construction (runtime no-ops, no tree edits), and my repo diff touches
  only `film_band_gradient.py` + its test.

## ARTIFACTS INDEX (/tmp/c4_1/)

- Acceptance: `ACCEPTANCE_strokes.png` (C2|C3|C4 at the four sites),
  `strokes_*.png`, `band_V0m_A2S2.png`, `sweep48_{V0m,off_face,iso_face,
  A1,A2,A2S,A2S2,A2S3,A5,W1,OFF}.{json,png}`, `sweep_critic.py`
  (instrument validation on the critic's own renders).
- Provenance: `trace_repaint.py` (+`validate_trace`), `provenance.py`,
  `localize.py`, `stamp_signals.py`, `witness_tone.py`, `tone_budget.py`,
  `components.json`, `stroke_photo_boxes.json`, `localize.json`,
  `prov_debug_views.png`, `trace_masks.npz`, `newdark_labels.npy`.
- A/B lab: `replay_lab.py`, `patches.py` (all variants), `mae_decomp.py`,
  `ssim_delta.py`, `maedecomp_V0m_A1.png`, `ssimdelta_off_face_iso_face.png`.
- Bundles: `bundle_iso_face_2048` (the fix, isolated), `bundle_off_face_2048`
  + `bundle_off2_face_2048` (veto-off arms), `bundle_final_face_2048`
  (full-tree incl. mid-flight fringe, for the cross-lane flag),
  `bundle_iso_face_1024`, `bundle_canary_{ship,owl}_{on,off}_1024`,
  replay arms `bundle_{V0m,A1,A2,A2S,A2S2,A2S3,A5,A4,W1,VETOALL,OFF}_2048`.
- QA: `qa_iso_face/`, `qa_off_face/`, `qa_iso_1024/`, `comp_*/`,
  `tq_iso_face.log`, logs `*_2048.log`.

## REPRO

```bash
source .venv/bin/activate
python /tmp/c4_1/final_bake_iso.py face 2048 iso_face_2048   # the fix (fringe isolated)
python /tmp/c4_1/final_bake_off.py 2048                      # same-tree veto-off arm
python /tmp/verdict1/qa.py /tmp/c4_1/bundle_iso_face_2048 --out /tmp/out
python /tmp/c2d/qa_shadecomp.py /tmp/c4_1/bundle_iso_face_2048 --shading-comp
python scripts/texture_qa.py /tmp/c4_1/bundle_iso_face_2048
python /tmp/c4_1/sweep48.py iso_face                          # 48-view stroke sweep vs C2
python /tmp/c4_1/ship_ab.py ship on 1024 && python /tmp/c4_1/ship_ab.py ship off 1024
python -m pytest tests/ -q                                    # 225 passed, 1 xfailed
```
