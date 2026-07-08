# SOLVER C (cycle 2) — ship/owl single-view: smears provenance, honest brightness reference, fill-character restoration

Targets: Critic 1 ORDERS 3 + 7 for the single-view assets (SHIP-04/SHIP-08,
OWL-01/02/03). Repo: `/Users/albou/abstract3d` (shared tree, single commit +
uncommitted cycle work; my edits listed in §5). All bakes in this report are
MY OWN fresh bakes on the current tip unless labeled "shipped". Repro:
`python /tmp/c2c/bake_one.py {ship|owl|face} {1024|2048} OUT [--pin AZ EL]`
(wraps `/tmp/solve3/bake_lib.py`: `bake_projection_texture(...,
texture_completion="auto", projection_model="orthographic")`, matted front
photo, export contract both harnesses accept).

## Headline

- **Both single-view assets pass ALL texture_qa gates at BOTH resolutions**
  (fresh bakes, current tip, my calibration patch in):
  ship 1024/2048 PASS 13/13, owl 1024/2048 PASS 13/13. Dark smears 0 and
  facet fields 0 at 4x everywhere; fill energy 0.58/0.63 (ship) and
  0.58/0.68 (owl); brightness 0.84/0.85 and 0.88/0.89.
- Face guard: shipped `face-2mv` texture_qa PASS 13/13 (both res measured
  at 2048-native texture; fresh 1024+2048 rebakes also PASS 13/13) and
  verdict1 = **8 ≤ 8** on the shipped bundle AND on my fresh 2048 rebake.
- C1's "fresh rebake still shows fragments" is reconciled with evidence:
  the artifact bundles currently in `artifacts/validation/final-proof/`
  are OLD-pipeline bakes (metadata: `projection_only_plus_inpaint`,
  `inpaint_diffusion`, no fill_floor/detail/compositing stats — those
  stages never ran on them). The floor does not "exempt" their fragments;
  it never executed. Every current-pipeline bake I ran (8 configurations)
  has zero fragments.
- The owl brightness gate was measuring the BACKDROP, not the subject:
  harness fixed to matte the photo with the same `remove_background_robust`
  the bake uses. Owl 0.567 → 0.891 on the same texture; the residual −11%
  decomposes into the QA viewer's own diffuse shading term (−7.3%) and
  −3.9% true albedo deficit — inside the gate, nothing left to fix in the
  bake.

---

## C1 — SHIP `close.dark_smears_4x`: fragment-by-fragment provenance

### What is actually failing, and where it came from

`scripts/texture_qa.py` on the CURRENT shipped bundles (post my harness fix;
old harness gives the same verdicts at slightly different probe placement):

| bundle (shipped) | dark_smears_4x | fragments |
|---|---|---|
| hunyuan-starship | **2 FAIL** | concavity_04 (az−43 el−13), fill_03 (az−89 el+60) |
| hunyuan-owl | **4 FAIL** | concavity_02 (az−108 el31), 04 (az23 el−4), 05 (az38 el10), 06 (az−12 el20) |

Texel-level provenance (`/tmp/c2c/trace_fragments.py` — renders a UV-index
texture at each probe, unprojects every fragment bbox to texture texels,
classifies against both the old and the matte-fixed region reconstruction;
JSON: `/tmp/c2c/trace_{ship,owl}_shipped.json`):

| fragment | texels | mean/min luminance | region split (obs/sym/fill) |
|---|---|---|---|
| ship concavity_04 | 139 | 48.9 / 0 | 8 / 0 / **131** |
| ship fill_03 | 52 | 41.1 / 3 | 11 / 0 / **41** |
| owl concavity_02 | 25 | 82.2 / 27 | 8 / 0 / 17 |
| owl concavity_04 | 37 | 93.5 / 9 | **31** / 0 / 6 |
| owl concavity_05 | 35 | 89.5 / 9 | **29** / 0 / 6 |
| owl concavity_06 | 40 | 58.3 / 27 | 16 / 0 / 24 |

Ship fragments are the classic transported-darkness class (fill texels at
luminance ~41–49 against a brighter context) — exactly what
`enforce_fill_luminance_floor` lifts. Owl fragments are majority-observed
dark content whose screen components spill ≥40% into adjacent un-lifted
fill. The region split is identical under the old and fixed harness
reconstruction — **no fragment survives because of a floor exemption.**

### Why the floor "didn't fire": it never ran

The shipped bundles' metadata proves their textures predate the current
pipeline: `projection_mode: "projection_only_plus_inpaint"` and
`unseen_fill_mode: "inpaint_diffusion"` are strings that no longer exist in
`bake_projection_texture` (current modes: `..._plus_fill3d` /
`..._plus_symmetry...`; fills: `mesh_harmonic` / `nearest_observed_3d` /
`backend_color_field`); there are no `fill_floor` / `fill_detail` /
`delight` / `compositing` stats keys; `texture_completion: none` despite
"auto" being requested (current auto applies mirror at symmetry score 0.98);
texture bake time 34–66 s at 2048 vs ~200 s through the current stack. The
files were (re)written into `artifacts/` tonight at 22:58/23:04 with
Jul-4-era content (geometry md5s match the critic's T0 table exactly).
Whoever refreshed the artifacts copied old-pipeline outputs; the ruling's
core complaint (owner-visible artifacts never rebaked through the fixes)
still stood at the time of writing.

### The floor DOES fire identically post-integration — measured

Every fresh current-tip bake shows `fill_floor.applied: true`, 70k–203k
lifted texels, and **dark_smears_4x = 0**:

| fresh bake | pose | smears 4x | fill_floor lifted |
|---|---|---|---|
| ship 1024 (pinned +30/+15) | override | 0 | 70,676 |
| ship 2048 (pinned) | override | 0 | 203,057 |
| ship 2048 (estimator free) | gradient_ncc → +30/+15, score 0.038 | 0 | (same texture) |
| ship 2048 (compositing=**gradient_domain** forced) | pinned | 0 | 201k |
| owl 1024 / 2048 (estimator free → declared az0) | gradient_ncc declined | 0 / 0 | 7k / 25k |

The specific worry in my orders — "their floor may not fire identically
post-integration with gradient compositing" — is disproven twice over:
single-view `auto` resolves to LEGACY (the compositor is not even in the
single-view path; `compositing.applied: false` in every bundle above), and
FORCING `gradient_domain` on the ship changes the composite by <1/255
(solver 1's claim, reconfirmed) with all 13 gates still green
(`/tmp/c2c/gradck/`). Determinism cross-check: pinned and estimator-free
ship bakes produce **bit-identical textures** (md5 ef660b57…).

Ship-ready refreshed bundles (fresh 2048 textures + preserved backend
generation metadata + regenerated preview/contact sheet, texture_qa PASS
13/13 re-verified on the exact directories): `/tmp/c2c/ship_ready/
{hunyuan-starship,hunyuan-owl}` — shipping is
`cp -a /tmp/c2c/ship_ready/hunyuan-starship/. artifacts/validation/final-proof/hunyuan-starship/`
(same for owl). I did not overwrite `artifacts/` myself: three other
solvers are landing concurrent fixes (pose gate, face lanes) and the
cycle-final rebake should carry all of them at once — the critic rebakes at
claim time either way.

---

## C2 — OWL brightness 0.567: the gate measured the backdrop, not the subject

### Root cause (measured)

The owl `input.png` is RGB (768², unmatted, light-gray studio backdrop).
The harness's photo-side foreground for RGB inputs was "any channel >18
from pure white", which classified **100.0% of the frame** as foreground
(backdrop median luminance 205 ≠ white). The brightness reference became
the backdrop's 203 instead of the subject's 129, so a render whose subject
tone was in range read 0.567. The subject itself (the same
`remove_background_robust` matte the bake consumes: 30.2% of frame) has
median luminance 129. Ship photo: 7.2% of frame was backdrop-counted
(median 201), biasing 0.752 → true 0.845. Face photo: near-zero bias
(0.960 unchanged) — the heuristic was accidentally right there, which is
why this never fired on the face lane. Numbers: `/tmp/c2c/probe_owl_photo.py`.

### Harness fix (`scripts/texture_qa.py`, general logic, documented)

New `photo_foreground()` used by ALL photo-side references: RGBA photos
keep their alpha; RGB photos are matted with the SAME
`remove_background_robust` the bake pipeline applies before projecting;
degenerate mattes (<0.5% or >98% of frame) and unavailable matte models
fall back to the old heuristic **explicitly** (`heuristic_nonwhite`
recorded in results.json and printed in the gate line). Applied to:

- `viewer.brightness_ratio` (the C2 gate),
- `photo_seam_allowance` (same-material deltaE bands no longer sample
  backdrop pixels; owl allowance 52.2 → 61.2, ship 71.8 → 60.0 — both
  directions, both honest),
- `--calibrate-photos`,
- **front-view visibility reconstruction** (`bundle_view_specs`): texels
  whose rays land on background pixels are no longer counted as observed.
  Ship coverage reconciliation qa 0.261 vs bake 0.177 → qa 0.211 — the
  region split every close-zoom/texel gate keys on is now faithful to what
  the bake painted. (This is why my fill-energy "before" numbers below are
  slightly stricter than the old harness's.)

The photos still pass calibration (`--calibrate-photos`: ok, facet_field
false, finite allowance). Face/ship/owl T0-class bundles re-audited under
the fixed harness — no gate flips anywhere except the intended ones.

### Residual deficit, attributed (`/tmp/c2c/probe_brightness_attrib.py`)

On my fresh owl 2048 bake: photo subject median 129; flat (unshaded)
render 124 → **0.961 true albedo ratio**; shaded viewer-truth render 115 →
0.891 reported. So of the remaining −10.9%: **−7.3% is the QA viewer's own
diffuse shading term** (0.88 + 0.12·N·L — the harness's viewer model, not
the texture) and **−3.9% is real albedo deficit** (photo shading baked into
projected texels + fill tone tracking those texels). 0.891 clears the 0.72
floor with 24% margin; the remaining −3.9% albedo gap is a single-photo
shading-removal question (delighting needs a second view to be
identifiable — solver 3's documented limit), so I did not chase it further.
Ship: flat 0.899, shaded 0.845 (shading −6.0%) — same structure, SHIP-06's
margin is now 0.125 instead of 0.03.

---

## C3 — Fill character: energy ≥ 0.5 at BOTH resolutions with smears/facets still zero

### Where the energy went (stage-by-stage instrumentation, current tip)

`/tmp/c2c/instrument_fill.py` captures the texture after every fill stage
of the real bake and measures the QA statistic (Scharr energy, eroded
regions, unobserved = fill + mirror):

| stage (ship 1024) | fill/observed energy ratio |
|---|---|
| harmonic fill + smooth (base) | 0.054 |
| + synthesize_fill_detail (gain 0.7, old defaults) | 0.431 |
| + enforce_fill_luminance_floor | **0.394 → gate FAIL** |

The floor costs only ~0.04 (solver 3's design holds); the problem is the
detail synthesis undershooting the gate line before the floor ever runs.
Decomposition (`/tmp/c2c/probe_detail_stats.py`, ship 1024): the applied
detail reaches only σ ratio 0.587 × frequency ratio 0.693 ≈ 0.41 of the
observed gradient energy. Three multiplicative causes: donor amplitude
transfer 0.84× (color-similarity weighting selects donors darker/quieter
than the observed median), carrier frequency 0.69× (3-texel finest octave
vs photo residual peaking at 1–3 texels), base luminance 0.79×
(multiplicative log-detail on a darker fill base yields proportionally
less LINEAR gradient — the gate measures linear luminance). Open-loop gain
raises cannot fix this: at gain 1.0 the ±0.25 log clip saturates (17% of
texels already at the clip at 0.7) and bought only +0.017.

### Fix (repo patch, general logic): closed-loop energy calibration

`synthesize_fill_detail` now (a) moves the finest carrier octave to ~2
texels (`wavelength_texels` 3→2, `octaves` 2→3; band 2–8 texels,
resolution-invariant by construction since octaves are pitch-scaled), and
(b) closes the loop on the gate's own statistic: it provisionally applies
the detail (clip + seam ramp included), measures the realized fill Scharr
energy on the atlas grid, and solves one global scale (secant, 2–3
evaluations, ~0.3 s at 2048) so the fill lands at `gain` × observed
energy. Three bounds, each load-bearing:

- **never below 1**: an already-rich fill (face rear-hair streaks at ratio
  1.06) is never dampened — monotone non-destructive;
- **never above `energy_calibration_max` = 3**;
- **σ guard**: the applied log-σ may not exceed the observed population's
  band-matched residual σ (lowpass at half the carrier's coarsest
  wavelength). Gradient parity may not be bought with granite: on
  edge-dominated content (flat hull plates + sparse panel lines) matching
  edge energy through noise amplitude would need σ far above the photo's
  stochastic band; the guard caps exactly there and the shortfall stays
  visible in `fill_detail.energy_calibration` (bake metadata) instead of
  being hidden.

Order of operations kept detail→floor: the mission asked me to check the
floor's position relative to the solve — single-view has no solve (legacy
compositing), and the lab A/B (`/tmp/c2c/offline_lab.py`) measured
floor-then-detail leaving 2× more under-floor darkness (texel dark
fraction 0.0071 vs 0.0034; 19 vs 10 dark components) because detail dips
below the floor line post-hoc, exactly the class the smear detector
flags. Detail-then-floor preserves the floor's post-guarantee; the floor's
energy cost at the calibrated level measured 0.05–0.06 (ship) / ~0.001
(owl) — priced into the margin.

### Results (fresh bakes, current tip, fixed harness; "before" = same tip without the calibration patch)

| bundle | fill energy before → after (gate ≥0.5) | smears 4x | facet fields 4x | facet_cellular | seams p95/med (allow) | brightness |
|---|---|---|---|---|---|---|
| ship 1024 | 0.390 → **0.578** | 0 | 0 | 0.037 (≤0.080) | 25.5/7.5 (60.0/23.9) | 0.837 |
| ship 2048 | 0.501 → **0.631** | 0 | 0 | 0.046 (≤0.094) | 23.7/5.6 | 0.845 |
| owl 1024 | 0.433 → **0.582** | 0 | 0 | 0.040 (≤0.080) | 38.7/11.3 (61.2/23.1) | 0.884 |
| owl 2048 | 0.602 → **0.683** | 0 | 0 | 0.007 (≤0.080) | 45.2/8.9 | 0.891 |

Calibration telemetry (bake stats): ship 1024 scale 3.0 (σ guard head-room
0.21 vs 0.26 band), ship 2048 scale 1.39, owl 1024 scale 3.0, owl 2048
scale 1.12 — the per-asset/per-resolution variation is exactly why a fixed
gain could never satisfy both resolutions simultaneously.

The cycle-1 facet_cellular/facet_fields failure class (verify3: ship 0.445,
owl 3 fields) is absent at both resolutions: those readings were the
harmonic base showing through under-amplified detail (my lab measures the
naked base at cellular 0.45/flat 0.70 — CLAHE reads its vertex plateaus as
cells; with calibrated detail the plateaus are buried: 0.007–0.046).

### Rear material read (SHIP-04 / OWL-03, accepted bar)

Before/after sheets at the ledger regions (`/tmp/c2c/crops/`):
`ship_4x_before_after.png`, `ship_2x_before_after.png` (underside az180/az0
el−20, engines az180, starboard az−45), `owl_4x_before_after.png`,
`owl_2x_before_after.png` (back az180/az±135). The owl back reads as dense
directional carved-grain consistent with the observed front (no leopard
blotches, no facet fields, observed|fill seam 45.2 vs allowance 61.2); the
ship underside/engines carry oriented panel-flow texture instead of the
soft wash. Content (specific greebles/panel layouts) remains impossible
from one photo — that half of SHIP-04/OWL-03 stays with the critic's
accepted P3 for content truth; the achievable "clean plausible material"
bar is what these bundles now deliver, gate-verified.

### Face guard (shared machinery)

- texture_qa: shipped `face-2mv` PASS 13/13 under the fixed harness; my
  fresh 1024 and 2048 rebakes PASS 13/13 (fill energy 1.159/1.040 — the
  calibration's lower bound 1.0 left the hair fill untouched as designed;
  the octave change nudged 1.06→1.16, inside the gate both sides).
- verdict1 (`/tmp/verdict1/qa.py`): shipped face-2mv = **8 fails** (≤ 8
  required); my fresh 2048 rebake = **8 fails** (same failure set:
  identity front SSIM/MAE, 3× dark_debris marginals, −90 eye_count ×2,
  side_right worst-window). Fresh 1024 = 9 (was 11–13 on this tip earlier
  in the cycle per solver reports; 1024 is not in the guard but reported
  for honesty).
- Full test suite: **187 passed, 0 failed** (`tests/`), including my 2 new
  tests and the 51-test texturing module.

---

## 5. Patches + tests (my edits on the shared tree)

| file | change |
|---|---|
| `src/abstract3d/texturing.py` | `synthesize_fill_detail`: finest octave ~2 texels (`wavelength_texels` 2.0, `octaves` 3), closed-loop energy calibration + σ guard + `stats_out` telemetry (docstring documents the measured decomposition and bounds); bake wiring passes `stats_out=fill_detail_stats` |
| `scripts/texture_qa.py` | `photo_foreground()` (matte-first photo reference, explicit fallback) used by brightness gate, seam allowance, photo calibration, and front-view visibility alpha; gate line + results.json report the method |
| `tests/test_texturing.py` | `test_synthesize_fill_detail_energy_calibration_reaches_gate` (dark-base fill must reach ≥0.5× observed energy, and ≤1.1× — no noise injection), `test_synthesize_fill_detail_calibration_never_injects_granite` (edge-dominated observed content: σ guard caps, honest shortfall reported); existing 45 tests untouched and passing |
| `CHANGELOG.md` | two entries (fill-character restoration; harness photo reference) with the measured numbers |
| `docs/KnowledgeBase.md` | two critical insights: closed-loop calibration of synthesized statistics against the gate's own metric; harness photo references must pass through the same matte as the bake |

Deliverables under `/tmp/c2c/`: `base/` + `after/` bundles (1024+2048,
before/after QA logs), `face_base/`, `face_after/`, `gradck/` (forced
gradient-domain ship), `unpinned/` (estimator-free ship), `ship_ready/`
(artifact-refresh candidates, QA re-verified), `crops/` (before/after 2x/4x
sheets), `trace_*.json` (fragment provenance), `instr_*.json` +
`detailstats_*.json` + `lab*.json` (stage instrumentation), probe scripts.

## 6. Honest limits

- **Content, not just character:** the calibrated fill matches the
  observed material's gradient statistics and orientation flow; it cannot
  invent the ship's actual panel layout or the owl's actual carving from
  one photo (critic's accepted P3). SHIP-01's "crisp panels vs cloud" gap
  narrows but the starboard side remains statistically-synthesized
  material, not model-kit panels; a rear/side reference photo or a
  generative texture prior is the capture-side remedy.
- **σ guard is a real ceiling on edge-dominated fills:** ship 1024 wanted
  scale ~3.5 for full 0.7× parity and got 3.0 within the σ cap (landed
  0.578, comfortably over the 0.5 gate). An asset whose observed side is
  ONLY hard edges on flat plates would cap earlier; the shortfall is
  reported in bake stats, and the honest fix there is structure transfer
  (out of scope, prototyped worse per the docstring's history).
- **Owl residual −3.9% albedo deficit** (flat-render 0.961) is baked photo
  shading; single-view delighting is unidentifiable (needs a second
  witness). Documented, inside the gate.
- **The QA viewer's −6…−7% diffuse shading term** is part of the harness's
  viewer model, not removable by any texture; if the gate floor ever
  tightens past ~0.93 it will bind before the texture does.
- **Shipped artifacts:** I prepared but did not install the artifact
  refresh (shared-tree discipline; three concurrent solver lanes). Until
  `ship_ready/` (or a later full-stack rebake) is copied in, the
  owner-visible bundles keep their old-pipeline fragments — that is an
  integration step, one `cp -a` away, not an open pipeline defect.
- **Shared-tree caveat:** all "before" numbers were measured on tonight's
  tip (texturing.py mtime 22:48 state); co-solver edits landing after my
  final verification (00:30–01:40) could shift absolute numbers. My last
  full-suite run (187 passed) and the after-bundle QA logs pin the state I
  verified.
