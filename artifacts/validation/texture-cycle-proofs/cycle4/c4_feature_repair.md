# SOLVER 2 (cycle 4) — Order 2: feature-fringe repair lane banks the identity budget

**Scope:** Critic 1's cycle-4 ORDER 2 — bank the measured +0.031
compensated-SSIM feature-lane headroom (tear-duct whites, lash-line
dashes, lip-edge dark-red dash: FACE-03/04's protected-feature residue)
to close FACE-14's compensated identity gate (comp >= 0.70 SSIM /
<= 15.0 MAE). Repo `/Users/albou/abstract3d`, work under `/tmp/c4_2/`.
Date: 2026-07-06/07. Harnesses: `/tmp/c2d/qa_shadecomp.py` (comp + raw)
and `scripts/texture_qa.py`.

**SHARED-TREE NOTE:** other cycle-4 lanes landed mid-session
(`commit_pale_chips`, `gradient_compositing.reconcile_specular_lobes`,
film-band work). Every headline number below is therefore PAIRED on the
final tip: control = identical bake with my stage monkeypatched to a
no-op (`disable_fringe.py`), candidate = default path. My prototype
baseline (pinned pre-landing tree) is reported where used.

## TL;DR

| gate @1024 (paired, same tip; shipped config = `bundle_fringe1024i`) | control (stage off) | candidate (stage on) | delta |
|---|---|---|---|
| comp identity[front] SSIM/MAE | 0.663 / 16.4 | **0.708 / 14.7** | **+0.045 / -1.7** |
| comp verdict (full 28-view battery) | FAIL 2 (front SSIM+MAE) | **PASS (0 failed checks)** | first full green |
| raw identity[front] SSIM/MAE | 0.639 / 21.8 | 0.680 / **21.4** | +0.041 / -0.4 (raw MAE <= 22 held) |
| comp sides L / R | 0.707/13.0, 0.718/14.9 | 0.706/13.0, 0.708/15.1 | -0.001, -0.010 (green, budgets 0.55/24.0) |
| detector deltas vs control, 28 views | — | ONE in-bounds eye-count flip (+22.5 el0: 1->2, bounds [1,2]) | no new failures |
| texture_qa | — | **PASS (0 failed gates)** | held |
| ship canary (single photo) | md5 799a291ab187c39c9343202368ee094e | **identical** | structural no-op |
| owl canary (single photo) | md5 013f1a28fdf291af127740184d0360c8 | **identical** | structural no-op |

At the deliverable resolution (2048, §6): comp 0.674/14.89 -> **0.688 /
14.4** (+0.014 SSIM; the MAE half of the gate is GREEN with margin), raw
MAE 22.0 -> 21.7, ONE detector delta across 28 views (debris -35el0
0.0013->0.0018, gate 0.0030), texture_qa PASS. The comp SSIM floor is
not yet met at 2048; §6 decomposes exactly where the remaining 0.012
sits (predominantly other lanes' clusters + the protected disc
interior).

The critic's ordered bar was comp >= 0.70/15.0 with the +0.031
feature-lane headroom banked: measured banked at 1024 = **+0.045
comp SSIM** (the order's +0.031 was the C3 decomposition at 2048; my
paired 1024 delta exceeds it because the lane also carries part of the
brow-film residue the stamps repair).

Repro (standard face bake; the estimator accepts az+20/el+8 at
gradient_ncc 0.0152, no pin needed):

```bash
source .venv/bin/activate
python /tmp/c4_2/bake.py NAME --res 1024            # candidate (default path)
python /tmp/c4_2/bake.py CTRL --res 1024 --patch disable_fringe  # paired control
python /tmp/c2d/qa_shadecomp.py /tmp/c4_2/bundle_NAME --out ... --shading-comp
python scripts/texture_qa.py /tmp/c4_2/bundle_NAME
```

The bake passes `identity_image` (the bundle's `input.png`, un-matted) on
the source view — see §4.1 for why the mechanism requires the identity
contract's own photo.

## 1. Provenance (the target deposits, measured on the instrumented baseline)

Instrumented 1024 bake on the pinned tree (`bundle_base1024/state.npz`),
per-blob analysis (`prov_targets.py`, `px2texel.py` mapping the C3 crops'
defects to texel blobs):

- **Left tear-duct white chip** (the subject-left eye, C3 `c3_face2.png`
  row 3): 63-texel bright blob at (+0.66,+0.11,+0.16), w50 0.25 (trace),
  winner = front photo, photo evidence AT the blob reads lum 0.85 == the
  blob (the chip IS the photo's own displaced caruncle/specular content).
  Its bilateral twin (y=-0.11) reads weak (w50 0.006) — INSIDE the rescue
  disc: the twin's content is the disc's transplant.
- **The right (transplanted) eye's tear-duct chips**: in-rescue blobs at
  y=-0.08..-0.09 — exactly mirror(+0.11) + the disc's placement_shift
  (+0.0246): the rescue disc COPIED the healthy side's chip into the twin.
  Both eyes' chips are one root defect plus its transplant copy.
- **Lash-line dashes** (below the rescue eye / nose-bridge side): dark
  blobs at w50 0.02-0.05, winner side_right at trace (e.g. 101 texels at
  (+0.71,-0.04,+0.11)); twins clean and confident (lum 0.84, w 0.74).
- **Lip-edge dark-red dash**: 47-texel dark blob at (+0.73,-0.04,-0.23),
  w50 0.05, winner front — displaced lip content (the cycle-1 measured
  per-feature mismatch: the mouth wants a different few-px correction
  than the eyes; no global transform satisfies both). Its mirror lands on
  the lip border itself (twin dev 0.171) — the bilateral-twin lever is
  measurably unavailable at the mouth.
- **Gate-loss decomposition** (`gate_decompose.py`, comp identity at the
  declared pose): the eye/mouth loss clusters sum to ~0.027-0.031
  SSIM-if-perfect — reproducing Critic 1's +0.031 feature-lane number on
  my baseline; the ear (0.012), strap (0.011), hair-curtain (0.010) and
  neck-wash (0.008) clusters are other lanes' territory and were not
  touched.

Key provenance conclusion: the deposits are the SOURCE PHOTO'S OWN
content, displaced under the projector correspondence — so the photo
UNDER THE IDENTITY CORRESPONDENCE (the gate's own registration) shows
clean material exactly where the bake shows each chip. That
correspondence is simultaneously the repair evidence and the banked
metric.

## 2. Mechanism (repo: `src/abstract3d/feature_fringe_repair.py`, wired last in `bake_projection_texture`)

Full design rationale in the module docstring; the semantics in one
paragraph: detect FEATURE COMPLEXES (confident strong-contrast dark cores
clustered at feature scale — no feature classes, no hand masks — plus the
bake's own rescue discs); rebuild the identity gate's correspondence
in-bake (render the current texture at the declared source pose, register
the identity photo to it with the gate's own construction: bbox map +
BT.601/area-average NCC refinement; z-buffer first-surface visibility;
bilinear-sample per texel); then repair with RESCUE-TRANSPLANT SEMANTICS
(tone match + feather + whole patch), the photo as the twin:

1. **Whole-complex corrective stamps**, domain ladder per complex:
   FULL (re-register everything the source itself witnessed) ->
   TRACE-ONLY (keep every confident texel) -> skip. Never-demote across
   witnesses: content a NON-source view confidently won is never
   overwritten in any mode. Bright-material context only (the hair mass
   and frontier belong to the band lane); dark photo content outside the
   feature's own radius is trimmed (photo hair over bake skin at the jaw
   = the co-witnessed parallax conflict, band-lane territory).
2. **Structure-preservation vetoes** on every stamp candidate, relative
   to the pre-repair state: texel-space first (under the renderer's own
   shade model, 6 poses), then render-space with the pipeline renderer at
   15 views (el0: 0/±22.5/±35/-45/-70/±90; el10: 0/±22.5/-45/-70/-90).
   No new and no lost compact dark blob at anatomical-feature size
   (>= 0.0009x the foreground bbox — the eye detectors' own floor);
   sub-feature micro-island fraction budget +0.0003 per view.
3. **Deposit-scale photo patches** for gate-contradicted trace-witness
   blobs outside stamped complexes (bright-source-only).
4. **Rescue-disc lane**: disc interiors are never photo-stamped (the
   disc fired because the photo evidence there is bad); fringe deposits
   in the protected ring re-copy through the disc's own anchored
   correspondence (mirror + placement_shift) from confident witnesses on
   CURRENT colors; the disc is refreshed LAST (whole-disc re-transplant)
   so healthy-side repairs propagate into the twin — this is what removes
   the transplant-copied tear-duct chip.
5. **Placement**: the stage runs after the fill floor, immediately before
   export, multi-view only; repaired texels are marked completion.

## 3. What is visibly fixed at 4x-6x

At 1024 (`ab_targets_h.png`, control | candidate):

- **Left tear-duct**: the beige/white chip at the inner corner replaced
  by the photo's lid/skin content (az0 + srcpose rows).
- **Lash lines**: the pale/dark dash fields below both eyes smoothed to
  lid skin; no hard fragments left.
- **Lip edge**: the below-lip dark-red dash and the pale chips are gone;
  the mouth reads as ONE soft mouth (the full-mode mouth stamp
  re-registers the photo's lips; the old doubled lip edge is repaired).
- **Rescue eye fringe**: the chip copies below the transplanted eye
  cleaned by ring recopies + disc refresh; the transplanted eye itself is
  intentionally untouched (its photo evidence is the smear the disc
  replaced; C3's FACE-15 grant stands).

At 2048 (`ab_targets_2048e.png`): tear-duct and lash-line rows improved
(chips softened/replaced), below-lip pale chips cleaned, the mouth at
srcpose reads cleaner; the az0 lip-edge dash is only PARTIALLY reduced —
the 2048 mouth-complex formation issue of §6. The 1024 pair proves the
mechanism removes it when the complex forms at full mouth extent.

## 4. Measured failure ladder (what it took to get here)

| attempt | result | fix |
|---|---|---|
| offline prototype on final colors (gate evidence + deposit patches + twin transplants) | +0.009 comp; localMin -0.007 (in-disc patchwork anti-correlated the worst window) | disc interior belongs to the disc: recopy only the fringe ring, refresh the disc whole |
| whole-complex stamps, atlas-connected domain | +0.025 comp but debris 0.0017->0.0033 at az0 (UV-chart cut left the old lip line as an isolated dash) | world-ball patch, no atlas trimming; smooth hole fill |
| full-domain mouth stamp | third eye at az0 (photo's soft slit re-shaded into a compact 45px blob; old content was 101px elongated) | structure vetoes + domain ladder (never-demote trace fallback) |
| brow stamps into hair-fringe fill | eye flip at el10, debris at ±35 | bright-material-context scope (the chip machinery's own rule) |
| in-repo v1: stage before detail/floor | +0.002 only; detail/floor repainted fill around repairs into new islands at 6 views | run LAST, on the shipped colors |
| in-repo v2: matted rgba as registration target | registration in the wrong basin (dx 0.065 vs 0.025); banked nothing | register against the identity contract's own photo (`identity_image`) |
| in-repo v3: mean-gray + bilinear NCC; symmetric-border closing | wrong optimum (NCC 0.503 vs 0.905 at the gate's metric); bbox 45px short | BT.601 gray, area-average downsample, flood-from-corner hole rule (bit-faithful gate construction) |
| in-repo v4: sub-feature "new blob" vetoes | a 22px speck vetoed the whole-mouth repair; micro-fraction baseline too noisy to bind | anatomical feature floor (0.0009x bbox) for blob vetoes; sub-feature growth budget +0.0003 |
| in-repo v5 (shipped) | comp 0.708/14.7 PASS full battery | — |

Insights recorded in `docs/KnowledgeBase.md` (3 new entries: gate-correspondence
bit-fidelity; re-registration vs detector-scale structure + veto ladder;
render-evidence stages run on shipped colors).

## 5. Canary discipline

- Single-view bakes are STRUCTURAL no-ops (`len(projections) < 2` exits
  before any work; enforced by `test_repair_feature_fringes_noop_contracts`).
- Measured (same tree, stage on vs off, 1024): ship texture md5
  799a291ab187c39c9343202368ee094e == identical; owl texture md5
  013f1a28fdf291af127740184d0360c8 == identical.
- Tests: `tests/test_texturing.py` +4 (registration recovers a known
  similarity; compact-blob classifier semantics — compact in, elongated
  slit out, speck counts as micro; texel structure veto fires on
  created/lost-confident blobs, allows re-registration shifts and
  trace-defect cleanup; no-op contracts). Full texturing suite 73/73 on
  the tip.

## 6. Deliverable resolution (2048, paired on the final tip)

Shipped-config pair: `bundle_fringe2048e` (candidate, final module
state) vs `bundle_ctrl2048d` (stage-off control, same tip;
film_band_gradient.py md5 b44185852c... stable across all 2048 pairs —
solver 1's landed FACE-20 state is INSIDE both numbers):

| gate @2048 | control | candidate | delta |
|---|---|---|---|
| comp identity[front] | 0.674 / 14.89 | **0.688 / 14.4** | **+0.014 / -0.5** |
| raw identity[front] | 0.648 / 22.0 | 0.662 / **21.7** | +0.014 / -0.3 (raw MAE budget held) |
| comp sides L / R | 0.686/12.2, 0.704/14.3 | 0.685/12.3, 0.698/14.5 | green (budgets held) |
| battery vs control | FAIL 1 (front SSIM) | FAIL 1 (front SSIM 0.688 < 0.70) | ONE delta across 28 views: debris -35el0 0.0013->0.0018 (gate 0.0030); no eye flips |
| texture_qa | — | **PASS (0 failed gates)** | held |

The comp MAE half of the gate is GREEN at 2048 with margin (14.4 vs
15.0). The comp SSIM half lands at 0.688 — +0.014 banked, 0.012 short of
the 0.70 floor. WHERE THE REMAINDER SITS (gate loss decomposition on the
candidate, `gate_loss_f2048d.json` — cluster geography identical for
d/e):

| cluster (gain-if-perfect) | location | lane |
|---|---|---|
| 0.0183 | ear complex | FACE-07 / ear-parallax (C2 proven-limit class) |
| 0.0102 | neck/jaw wash | FACE-04 remainder, co-witnessed apron class |
| 0.0086 | hair curtain right | band lane |
| 0.0060 | left temple/brow strokes | FACE-20 (solver 1's active lane) |
| 0.0047 | transplanted right eye interior | rescue-disc content (FACE-15 grant; this lane must not restamp it — §9) |
| 0.0026 + scattered <=0.0019 | mouth-left residue + small | the 2048 residue of my lane (see below) |

Honest reading: at 2048 the mechanism banks +0.014 with every detector
green, while the SAME mechanism at 1024 passes the whole battery
outright at 0.708/14.7 (+0.045 paired). The difference is measured, not
mysterious: (a) at 2048 two complexes are veto-refused on micro-island
growth that the 1024 baseline noise absorbed (complex 0 full-mode
0.0039->0.0051 at -22.5, complex 1 both modes 0.0024->0.0034 at -35 —
the vetoes are doing exactly their job against the knife-edge debris
gates); (b) the 2048 core clustering forms a smaller mouth complex
(r 0.045 vs 0.11) that leaves the dash zone half-covered — I
resolution-scaled the clustering morphology (see module) which
reproduces world-space semantics, but the 2048 dark-core field itself
clusters differently and the full 1024-equivalent mouth ball does not
form. The unbanked remainder above my lane's residue decomposes into
other lanes' clusters (ear/neck/curtain/temple) plus the rescue-disc
interior this lane is forbidden to touch. Closing the last 0.012 at
2048 needs the FACE-20 stroke cluster (0.0060 sits exactly there) and
band-lane curtain work — as the critic's own order anticipated
("Closing FACE-20 also feeds this gate") — plus, if demanded, a
mouth-complex formation fix in this lane (documented as the first
follow-up).

## 7. Artifacts index (/tmp/c4_2/)

- **Bundles (shipped config)**: `bundle_fringe1024i` (1024 candidate,
  PASS) / `bundle_ctrl1024` (paired control); `bundle_fringe2048e`
  (2048 candidate) / `bundle_ctrl2048d` (2048 control);
  `bundle_ship{A,B}`, `bundle_owl{A,B}` (canary pairs, md5-identical).
  Iteration bundles (`fringe1024[c-h]`, `fringe2048[b-d]`) retained for
  the failure ladder's evidence.
- **QA outputs**: `qa_*_comp`, `qa_*_raw` (+ `.log`s), texture_qa under
  /tmp/texture_qa/.
- **Crop sheets**: `ab_targets_final1024.png` + `ab_targets_h.png` (the
  1024 acceptance A/B at 6x), `ab_targets_2048e.png` (deliverable
  resolution), `targets_*.png` (4-view grids),
  `gate_loss_*.png`/`gate_locs_*.png` (loss decomposition sheets),
  `window_g_*.png` (gate-frame photo|render pairs).
- **Analysis**: `prov_targets.py`, `px2texel.py`, `gate_decompose.py`,
  `offline_repair.py` (the prototype lab with all measured arms),
  `corr_*.py` / `reg_*.py` (registration forensics),
  `crops_targets.py`, `ab_targets.py`.
- **Infrastructure**: `bake.py` (face bake, passes identity_image),
  `bake_asset.py` (canaries), `disable_fringe.py` (paired-A/B toggle),
  `pin/` (pre-session tree snapshot, md5s in `pin_md5.txt`).

## 8. Patches + tests + docs

- `src/abstract3d/feature_fringe_repair.py` (NEW, ~900 lines with
  rationale): the full mechanism of §2.
- `src/abstract3d/texturing.py`: stage wiring (after the fill floor,
  before export) + `stats["feature_fringe_repair"]`.
- `tests/test_texturing.py`: +4 tests (§5).
- `CHANGELOG.md` (1 entry), `docs/KnowledgeBase.md` (+3 insights).
- Not shipped, documented: bilateral twin-patch transplants for
  non-rescue deposits (superseded by the photo lane — the photo is the
  richer twin for source-witnessed defects); world-delta displaced
  stamps (measured weaker than gate re-projection: +0.0083 vs +0.0124 at
  the eye complex); keep-dark-cores middle domain mode (never won the
  ladder).

## 9. Honest limits

- **The 2048 SSIM floor is not closed by this lane alone**: 0.688 vs
  0.70, with the remainder decomposed in §6 (ear/neck/curtain/temple
  clusters = other lanes; disc interior = protected; mouth-complex
  formation = this lane's documented follow-up). The 1024 full-battery
  PASS (0.708/14.7) demonstrates the mechanism's ceiling is above the
  gate when the whole ladder ships; I am NOT claiming FACE-14 closed at
  the deliverable resolution, and I am NOT filing a ceiling experiment
  either — the headroom is demonstrated reachable, exactly as the
  critic's T2 ruling framed it.
- The transplanted right eye's INTERIOR stays the mirror transplant:
  its photo evidence is the very smear the disc replaced (grazing +
  misregistration), so this lane must not restamp it. The eye's residual
  gate loss (~0.005 SSIM window) is the disc's placement/content bound —
  C3's FACE-15 grant territory, not this order's.
- The comp gate margin at 1024 is +0.008 SSIM / 0.3 MAE; knife-edge
  detectors (the +35 debris 0.0021, the in-bounds eye flips) sit inside
  their budgets but this asset keeps living near thresholds (the harness
  property documented since cycle 2).
- The mechanism assumes the identity contract's photo is available to
  the bake (`identity_image`); with only the matted rgba it still runs
  but registers in a measurably different basin (documented in the
  wiring comment) — callers of the proof lanes should pass the bundle's
  input.png through. The shared final artifact rebake must do the same.
- Compute: the stage adds ~1-3 min to a multi-view face bake (gate
  render + 15-view veto battery at 896 per stamp candidate); single-view
  bakes pay nothing.
- 2048 core clustering differs from 1024's despite world-space scaling
  of every floor and morphology parameter (measured: the mouth complex
  forms at r 0.045 vs 0.11); the first follow-up for whoever continues
  this lane is complex formation directly in world space (voxel-graph
  clustering like the rescue detector) instead of atlas morphology.
