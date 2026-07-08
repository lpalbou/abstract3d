# ZERO-DEFECT TEXTURE CYCLE — FINAL CERTIFICATION
Critic 1 (project-owner proxy) — 2026-07-07 ~13:55 CEST.
Authority chain: `/tmp/critic1/LEDGER.md`, `RULING.md` (c1),
`RULING_CYCLE2/3/4.md`, `RULING_FINAL.md` (c5), this document (c6,
definitive). Verification evidence: `/tmp/critic1/cert/` (my bakes,
harness logs, renders), sheets `/tmp/critic1/evidence/sheets/cert_*.png`.

## VERDICT: **PASS** — zero OPEN entries under the zero-defect standard

Every entry ever ledgered across six cycles is FIXED (23), PROVEN-LIMIT
with a documented capture remedy (10), or merged (1). The last open entry
— FACE-22, the neck/chest line-art — is closed on evidence I produced
myself. Nothing visible at my 4x standard remains that this pipeline
could have prevented from these inputs.

## WHAT IS CERTIFIED (all numbers from MY OWN runs on the exact bytes)

| asset | texture md5 | harness state (mine) | determinism |
|---|---|---|---|
| `artifacts/validation/iter3-multiview-fixed/face-2mv` | `928705f3edfc9036348c12bf34435d9d` | COMPENSATED battery (authoritative identity gate, anchored 0.70/15.0): **PASS 0 failed checks — front 0.704/14.9, side_right 0.706, side_left 0.687**. RAW battery: all 28-view detectors green (worst dark_debris 0.0027 vs 0.003, worst skin_in_hair 0.0036 vs 0.010), raw MAE **21.7 ≤ 22.0 green**; raw front SSIM 0.678 documented as diagnostic (contains the measured renderer-shading artifact; compensated protocol authoritative since my cycle-3 boundary ruling, budgets anchored to the demonstrated input ceiling). `texture_qa` **PASS 13/13**. | **My independent canonical-recipe bake reproduces the published texture BIT-EXACTLY** (md5 equal). With the solver's determinism triplet: four independent bakes, one hash. |
| `artifacts/validation/final-proof/hunyuan-starship` | `b8e2b0d47ec4336a17067b59e1718455` | `texture_qa` PASS 13/13 (my cycle-4/5 re-runs); nose-melt fix certified cycle 3 with a perfect-checker ceiling experiment | Byte-identical since 2026-07-06 07:36 through three subsequent cycles of face work (canary contract held at every ruling); critic 2's fresh recipe rebake reproduces the hash bit-exactly |
| `artifacts/validation/final-proof/hunyuan-owl` | `ff746509ccb9429a6161cd40657df080` | `texture_qa` PASS 13/13 (my re-runs); pose gate honestly declines the drift candidate (relative-margin fix verified cycle 3) | Byte-identical since 2026-07-06 20:57; critic 2's fresh rebake reproduces bit-exactly |

Test suite on the certified tree: **235 passed, 2 xfailed** (my run).
The face bundle's metadata carries the publication block (recipe, md5,
pre-overwrite verification) per the FACE-21 checklist, now in
`docs/methodology.md`.

## CYCLE-6 VERIFICATION DETAIL (T1)

- **FACE-22 closed at my own crop framings.** At the exact framings that
  opened the entry (az0 glyph zone 5x/6x, az−22.5 contour 4x/5x, plus
  az±45/±67.5/±90 neck at 3x): the "ΔΔ|" glyph cluster and the closed
  chest contour are GONE (`cert_face22.png`, `cert_neckside.png` — C5 vs
  C6 columns). What remains is a soft wide tone valley (~2–3/255, the
  disclosed witnessed-content residual) — the same amplitude class as
  the FACE-09 residual I ruled below the visible-defect bar in cycle 5,
  and it reads as neck shading at native contrast. A 2–98% contrast
  stretch (`cert_neck_stretch.png`) can still trace faint fragments —
  the stretch is my provenance instrument, not an acceptance condition;
  at owner conditions (4x, native contrast) there is no line-art read.
- **The three-owner provenance is accepted as measured**: film repaint
  operating outside its field's support (the ratio field S is scale-free
  and was extrapolating 9–24 transition-lengths from the mass where its
  profile was never measured — the support bound is the mechanism's own
  scale honestly applied), trace-commit blob rims (border mixtures below
  `deviation_min` by construction), and the missing completion tone
  handoff under gradient compositing. The instrumented capture
  reproduced the c5 published md5 exactly, and both ablations close the
  attribution loop in both directions.
- **Fix surfaces carry no new printing** (`cert_fixsurfaces.png`): the
  temple band and az+22.5 wisps are unchanged from their accepted
  states; no chroma seams at the support cut (the stamp feather measured
  chroma_seam 0.49–0.69 → 0.13–0.23 at az+22.5/+70 during their
  ladder, and my comp battery shows seams ≤ 0.31 everywhere); no rim
  artifacts at the chip/mouth sites; the az−90/az+90 curtain and ear
  read as their C5-accepted states with slightly better completion tone;
  rear and disc unchanged. My 48-view grids show no regression at any
  azimuth or elevation.
- **Disclosed residuals ruled**: the "soft witnessed valley" at the
  az−22.5 site is witnessed blend content whose sweep the trace commit's
  whole-neighborhood residue rule correctly refuses (the rule exists to
  prevent the measured partial-cleanup unmasking class; overriding
  refusal ledgers to chase sub-visibility tone is how regressions were
  minted in cycles 1 and 3) — below the bar, documented. The
  "apron-interior stripes" sit inside the cycle-5-accepted shadow
  gradient and read as its texture at my framings — cosmetic,
  sub-visibility, evidence filed with the apron mechanism's owner for
  any future tone work.

## THE KNIFE-EDGE RULING (T2)

Critic 2's two flags, my measurements on the certified bytes, and my
call:

1. **Comp SSIM margin +0.0037** (0.7037 vs the 0.70 floor) and **comp
   MAE margin +0.09** (14.91 vs 15.0 — now the thinnest margin):
   **shippable as-is.** The identity budgets were anchored to the
   demonstrated input ceiling by my cycle-3 governance — the pipeline
   SHOULD ship near them; a wide margin would mean either the ceiling
   evidence was wrong or the gate was soft. The margins are
   bit-deterministic properties of frozen bytes, reproduced by me
   end-to-end (bake → hash → battery). Thin margins are a maintenance
   constraint, not an artifact defect — they are wired into the
   contract below.
2. **Raw dark detector**: my run on the certified bytes measures the
   worst view at **0.0027 vs the 0.003 gate (90%)** — improved from
   cycle 5's 0.0029, deterministic, green. No action needed on these
   bytes.
3. **Critic 2's cumulative-veto hardening (the advancing-baseline
   re-arm): ADOPTED AS MANDATORY — as the first item of any future
   pipeline change, not as a blocker for this certification.**
   Rationale, consistent with my precedents: my gates measure the truth
   of shipped bytes; the certified artifact's detectors are measured
   green and frozen bytes cannot creep. The creep vector is real for
   FUTURE bakes (measured: ~7 stamps produced +0.00096 at one view,
   triple the single-stamp budget, inside the letter of every per-stamp
   check), and the absolute detectors that bound it sit at 90–97%
   utilization — the next mechanism that lands without the cumulative
   bound will spend margin that no longer exists. Requiring process
   hardening forward (like the FACE-21 checklist) while certifying
   measured bytes is exactly how this cycle has governed since cycle 4.

## COMPLETE PROVEN-LIMIT REGISTER (10 entries — every one carries its capture remedy)

| entry | limit (proven by) | capture remedy |
|---|---|---|
| FACE-03 eye-corner micro-residue | repair measured to cost eye_count/debris at every attempt across 3 cycles; absolute detectors bind | frontal capture with cleaner eye-corner resolution |
| FACE-07 ear complex | witnessed skin-between-strands (w90 0.63–0.73) + ear-cluster parallax (C2 grant) | capture with ears exposed / hair tucked |
| FACE-08 elf-ear silhouette | geometry: conservative-clamp bound (C2) | better geometry source / multi-view reconstruction |
| FACE-11 chest straps | zero witnesses on the continuation domain in ANY view; mesh truncates the torso | a photo framing the chest |
| FACE-13 crown flaps + mottle | texture ceiling experiment (clamp-all barely moves the read, dims the real parting); flaps are mesh topology | crown photo (mottle) / mesh repair (flaps) |
| SHIP-01 far-side fill material | single-photo content limit (C2 ceiling grant) | prow-on or ±45° second photo |
| SHIP-04 underside/engines | same content family | underside/rear photo |
| SHIP-05 glow zone | 97% fill at donor-consensus tone (0.95); glow = macro-structure absence | port-side / rear photo |
| SHIP-07 photo→fill seams | content family, tone within calibrated allowance | any second viewpoint |
| OWL-03 rear carving detail | single-photo content limit | rear photo |
| (registry note) | curtain-right strand mismatch + temple tone — filed cycle 5 with ceiling experiments (full/tone-only bounds); parallax/witness family, subsumed under the FACE-07/08 grants | curtain pinned back at capture, or curtain-sheet geometry |

## MAINTENANCE CONTRACT (binding for any future change)

Any change to `src/abstract3d/`, the bake recipes, or the harnesses
requires, BEFORE re-publishing any bundle:

1. **First item of the next pipeline change**: implement critic 2's
   cumulative-baseline veto in the fringe stage (refuse if post >
   original-baseline + 0.0003 AND post > original battery worst) — the
   filed recommendation is adopted, it closes the n×0.0003 re-arm creep
   without touching the exemption.
2. Canonical-recipe determinism check: ≥ 2 independent bakes, one hash,
   published via the staging checklist (`docs/methodology.md`) with the
   publication block updated.
3. The full gate set on the exact staged bytes: compensated battery
   (comp front ≥ 0.70 / ≤ 15.0, zero failed checks), raw battery (all
   detectors green, raw MAE ≤ 22.0), `texture_qa` 13/13 — on all three
   bundles.
4. Canary md5 A/B: ship `b8e2b0d4…` and owl `ff746509…` must reproduce
   bit-exactly with the change on and off, or the asset must be
   re-certified through a full owner-conditions battery.
5. Knife-edge watch: comp SSIM +0.0037, comp MAE +0.09, raw dark
   0.0027/0.003 — any change consuming more than half of any of these
   margins requires a fresh critic battery, not just the harnesses.
6. The identity budgets stay anchored to the input ceiling: re-anchoring
   in either direction requires V20-quality evidence (the cycle-3
   standard: reproduced artifact + frontier mapping + independent
   verification).

## FINAL LEDGER (34 entries ever — closing state)

- **FIXED: 23** — FACE-01, -02, -04, -05, -09, -10, -12, -14, -15, -16,
  -17, -18, -19, -20, -21, -22; SHIP-02, -03, -06, -08; OWL-01, -02, -04.
- **PROVEN-LIMIT (closed, remedies above): 10** — FACE-03, -07, -08,
  -11, -13; SHIP-01, -04, -05, -07; OWL-03.
- **MERGED: 1** — FACE-06 → FACE-14.
- **OPEN: 0.**

Provenance corrections carried on the record: the FACE-05 column was the
source photo's baked specular (S1 exonerated, cycle 4); the neck wash was
the reference's lit tone over the source's genuine cast shadow (my
cycle-4 apron attribution had the sign reversed — corrected cycle 5);
FACE-22 decomposed to three owners (film field support, trace-commit
rims, completion tone handoff — cycle 6), of which two were my named
suspects and the third was found by the stage walk.

## CERTIFICATION STATEMENT FOR THE PROJECT OWNER

The three assets in this repository are certified against the zero-defect
standard as of 2026-07-07:

**What you get.** The starship's head-on intake is structured and
readable, its photo side crisp, its far side plausible granular hull;
the owl is clean at every angle and both zoom levels; the face reads as
the subject from every one of the 48 battery views — the film band,
black strokes, chips, pale column, doubled features, leopard rear, tan
disc, and the neck line-art are gone, the identity gate is met
(0.704 compensated against the 0.70 floor anchored to what these three
photos can physically realize), and every number above reproduces
bit-exactly from the recorded recipes: bake it again and you get these
exact bytes.

**What zero-defect means here — no more and no less.** Within THESE
inputs (one frontal photo + two profile references for the face; one
photo each for ship and owl): nothing visible at 1000 px and 2x/4x crops
across 16 azimuths and 3 elevations that the pipeline could have
prevented, every gate met on deterministic bytes, and every remaining
imperfection PROVEN — by ceiling experiment, witness-coverage audit, or
measured trade — to be a property of the inputs, each with the exact
photograph that would remove it listed in the register above. Ten such
limits stand. They are not promises to fix later; they are the honest
boundary of what these photographs contain.

**Standing obligations.** The certification binds future work through
the maintenance contract: determinism re-proof, full gate set on staged
bytes, canary hashes, and the cumulative-veto hardening as the first
item of the next pipeline change. Any bake that ships without that
protocol is outside this certification.

— Critic 1, owner's proxy
