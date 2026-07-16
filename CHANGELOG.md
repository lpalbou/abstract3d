# Changelog

## Unreleased

### Added (coverage-driven adaptive reference-angle planning)

The generated-reference angles are no longer unconditionally the static
`DEFAULT_ANGLES` (back / side_left / side_right / top): those slots
encode the ASSUMPTION of a canonical front source photo, and the x-wing
incident measured what happens when the assumption is false — from its
(0,+33) top-elevation source the whole static set predicts 0.257 of
quality-weighted witnessed surface while the planner's set predicts
0.453 (+76%), because the underside (first pick, gain 0.223 — 3x the
static set's best slot) was never a static slot and the static top
duplicates surface the photo and the planner's elevated picks already
witness (residual gain 0.013, below the stop floor).

- **Planner** (`reference_generation.plan_reference_angles`): greedy
  selection over a fixed 16-candidate view sphere (equatorial ring at
  45° steps — one facing-0.2 view reaches ±78.5° of azimuth, so the
  ring over-covers; ±55° rings at 90° steps — 55° is the measured
  production top angle and sees vertical normals at facing 0.82; no
  poles — the ±55° rings already witness polar caps at 4x the paint
  floor, and poles are the least generator-feasible class), maximizing
  marginal newly-witnessed surface under the bake's own paint-weight
  law (`((facing-0.2)/0.8)^2` — a binary-cutoff objective measurably
  inflates coverage with grazing content the bake paints at near-zero
  weight). Surface the source photo witnesses is LOCKED (the bake's
  `protect_observed_texels` absolute mode forbids generated content
  there, so no candidate may claim it). Stops at `budget` (4) or when
  the marginal gain drops below `min_gain` = 0.025, measured on the
  recorded twelve-bundle fleet: every genuinely-new-region slot
  predicts 0.029+ (every budget-4 pick 0.032+), the redundant class
  0.022 and below. Deterministic, CPU-only, no rendering (~0.1 s on
  120k-face production meshes). Witness math is the existing
  `project_source_witness` projection, factored into a shared
  vertex-visibility helper (`_view_projection`) and verified
  bit-identical on fleet meshes.
- **One pose estimate per run** (hunyuan runtime + `rebake_bundle`):
  the refs-off A/B baseline is now baked BEFORE reference generation
  (the generation flow always baked it — after the candidate — so this
  reorders, costing zero extra bakes), and its pose-guard verdict is
  the single source-pose statement threaded to angle planning, to
  generation conditioning (`source_pose` — historically hardcoded
  (0,0) in the pipeline while the bake estimated, so the anchor-class
  witnessed-consistency gate judged the WRONG region on pose-estimated
  subjects), and through the A/B verdict. Threading the pose through
  `source_pose_override` was measured and REJECTED: an override flips
  every view's registration to the projector frame — semantics the
  estimated-pose lane must not inherit (measured on the face proof:
  verdict1 failures 2 -> 10 when estimated poses are re-centered) and
  which would break the certified owl byte-identity even at (0,0).
- **Option surface**: `texture_reference_angle_planning` /
  `--texture-reference-angle-planning` / `reference_angle_planning`
  (rebake) ∈ {auto, adaptive, static}; explicit
  `texture_reference_generation_angles` continues to override
  everything (unchanged contract). The plan — planned angles, per-angle
  predicted gains, coverage curve, static counterfactual, and the
  planning mode that actually chose the angles — is persisted in
  `texture_artifacts.reference_generation.angle_plan` in every
  generation-active bundle regardless of mode (persist-for-diagnosis).
- **Default: `auto`** — adaptive exactly when the pose lane COMMITS a
  non-canonical source pose (estimated, or an explicit non-zero rebake
  override), static otherwise. Justification from the fleet planning
  table (2026-07-15, /tmp/cov1): on every declared-pose subject (owl,
  chair, face) the adaptive advantage is small (+6-7% relative
  predicted coverage) and inside the planner's own noise, so the
  certified declared-pose flows keep their static sets and stay
  byte-identical; on every estimated-pose subject the static set
  leaves large measured coverage on the table (+15% portrait, +40-57%
  cars, +76% x-wing incident, +110% starship) because the static slots
  double-cover the photo hemisphere and miss its antipode. The pose
  guard's double-keyed commit semantics bound the wrong-plan risk: a
  pose only moves on decisive shape evidence, and planning under a
  deliberately mis-estimated pose (±15° on every fleet subject) costs
  at most 3.3% of the true-pose plan's predicted coverage while still
  beating the static set everywhere (sensitivity study in
  /tmp/cov1/report.md).

Validated (MPS, the production mlx-gen + flux.2-klein-4b-8bit
configuration; full logs and proof renders in /tmp/cov1):

- Full suite **398 passed, 3 skipped, 3 xfailed** (entry 382 + 16 new
  tests); parity canaries **3/3** (the refs-off paths this program must
  not touch are pinned byte-level and pass unchanged).
- **Owl e2e (auto -> static lane)**: texture.png md5 a275ea10… and
  geometry md5 c5d2409b… — **byte-identical to the approved standard**
  (6th reproduction); ACCEPT healthy; coverage 0.83; the angle plan is
  recorded in metadata (adaptive counterfactual +6%, not taken).
- **X-wing incident photo i23d (auto -> adaptive lane)**: pose
  silhouette_rescue (0,+33); planned underside_right (gain 0.223) /
  underside_left / top_right / side_left — the underside first, the
  incident's redundant top dropped (residual 0.013 < 0.025 floor); 4/4
  planned angles accepted (2 needed one texture-family retry);
  whole-bake ACCEPT healthy; **coverage 0.5525 vs 0.4132** for the
  static-angle run of the same photo on the same tree (/tmp/xfix3,
  +34% relative); texture_qa battery quiet (exit 0); underside render
  carries plating structure where the static run had fill.
- **Starship rebake (auto -> adaptive)**: plan underside_right /
  top_right / underside_left / side_right, 4/4 accepted first try
  (IoU 0.92-0.96); ACCEPT healthy; coverage **0.7033**; battery quiet
  (exit 0); the certified proof's flat-fill underside now carries
  witnessed plating structure.
- **Sports-car v7 rebake (anchor class, auto -> adaptive)**: 3/4
  accepted — the planned underside_rear is strict-rejected 6/6
  (palette_flip family; floor-only candidates are recorded, never
  baked), the correct refusal for a class the generator cannot render
  faithfully; coverage **0.3528 vs 0.3353** for the static-mode rebake
  on the same tree (both ACCEPT healthy, battery quiet on both). The
  historical v7 product bundle shipped refs-off at coverage 0.112.
- Pose threading is a deliberate behavior change for the anchor-class
  gates: the witnessed-consistency gate now judges the region the
  photo ACTUALLY witnesses (measured on the pinned v7 parity views:
  the historically-accepted side_left FAILS the witnessed veto at the
  true pose with tile median 16.96 — under the old (0,0) assumption
  the gate judged it in the wrong lane and could not see the
  contradiction). Ladders re-roll more on cars (v7 static-mode top:
  6/6 witness_veto) and acceptance sets shift within the strict-line
  contract; the pre-existing `close.dark_smears_4x` refs-on-car class
  (KnowledgeBase) measures 3 (historical bundle) / 36 (static rebake)
  / 55 (adaptive rebake) — present in BOTH modes, invisible at product
  angles, battery quiet; the count tracks how much rim-adjacent
  surface the references witness (fill has nothing to detect).

### Validated (x-wing incident program — elevated-capture pose + catastrophic-baseline regime; adversarial validation on the merged tree; end-to-end proof)

Validator pass (2026-07-15, /tmp/xfix3) over the two concurrent
landings below: the pose-lane extended elevation search
(`texturing.py`) and the catastrophic-baseline acceptance regime
(`bake_acceptance.py` + call sites). No neutralizations were needed —
no fleet case regressed on the merged tree.

- **Suite**: 375 passed at entry -> **382 passed, 3 skipped, 3
  xfailed** on the merged tree (+2 pose tests, +5 regime tests; zero
  failures). Parity canaries **3/3** at entry, at the pose landing,
  and on the final merged tree (P1 pose (17.5, 8.0) exact /
  identical_frac 1.000; P2 c84f2e49… twice; P3 e32ba995… twice).
- **Fleet pose matrix reproduced independently** (own harness, entry
  worktree vs merged tree, field-by-field including the full guard
  trail): **11/11 recorded mover/stayer verdicts bit-identical**
  (movers fresh_car/car_a/car_b/v7/v4/v2/starship/portrait2mv at
  their recorded poses; stayers owl/chair/face at declared), the only
  additive change being the `extended_search` trail key
  (consulted=False on all 11, in-band best 0.80-0.97). The x-wing
  moves declared (0,0) -> **silhouette_rescue (0, +33)** with the
  extension consulted (core best 0.7343 < 0.75 action floor,
  extended best 0.8506).
- **Acceptance pins reproduced independently**: fix1 12/12 at the
  pinned margins (mistoned brighten 2.8155/2.7995, misreg dark
  0.1004, darktoned 0.3324); hue1 18/18 (rotations 1.978/1.312/6.508
  REFUSE, hue15 0.623 ACCEPT, car_bo3 vetoed 0.008 / raw 1.870);
  afix2 battery (accepts <= 0.0036 blotch, v7cand_stamp votes at
  0.0245); gfix2 2048 rebuilds read healthy-regime ACCEPT with
  collapse=False. Catastrophic-regime fixtures rerun on the landed
  gate: x-wing pre-fix pair ACCEPTs with every voting axis green
  (brighten 0.814 recorded as warning); chair guards clean ACCEPT /
  +25 L REFUSE (hue-abs 0.282) / misreg REFUSE (mirror 1.495) /
  rotated REFUSE (hue-abs 5.177). Entry-tree control reproduced BOTH
  pre-fix defects: the clean rescue REFUSED (brighten 1.025 +
  confounded hue 1.038) and the misreg candidate ACCEPTED (a real
  pre-fix acceptance hole the mirror axis closes).
- **End-to-end proof (MPS, the incident configuration)**:
  - **The user's exact command** (t23d x-wing): exit 0 **healthy**,
    zero warnings, 33 min. Fresh t2i draw (top-elevation class);
    pose silhouette_rescue (0, +15) matching its photo, observed
    coverage **0.6255** (incident: 0.0095), 4/4 references accepted,
    whole-bake **ACCEPT in the healthy regime**, battery quiet.
    Hostile turnaround at el 0/10/50 + top + underside: the texture
    wraps the ship everywhere (hull plating, cockpit), no
    fill-dominated product surface, no historical artifact class.
  - **Elevated-capture i23d control on the exact incident photo**:
    pose **(0, +33)** — the landscape-predicted basin — coverage
    0.4132 (source view 0.2064, efficiency 0.462), healthy, 4/4 refs,
    ACCEPT healthy regime, battery quiet, texture wraps at all
    elevations. The pose fix is validated against the exact failing
    input independent of t2i draw variance.
  - **Owl e2e regression (references on)**: geometry md5 c5d2409b…
    == the approved standard (5th byte-identical reproduction) and
    texture.png md5 a275ea10… **byte-identical to the approved
    proof bake**; ACCEPT 17.02 -> 17.24 / 19.02; battery quiet.
- **Honest residuals** (tracked, no regression): (a) the fresh t23d
  x-wing fails the standalone `texel.facet_cellular` QA gate (fill
  cellular fraction 0.111 vs its 0.080 line — straight-edged
  fill-detail cells on the never-witnessed underside), a pre-existing
  fill-synthesis class, not a program regression: the approved
  legacy chair proof measures 0.542 on the same gate (6.8x its line)
  and the i23d control PASSES (0.033); battery quiet on all three.
  (b) The incident bundle itself passes standalone texture_qa exit 0
  (uniform fill carries nothing for artifact detectors to fire on;
  only the registration-floor warning names it) — the coverage
  floors in bundle metadata (quality_verdict) remain the load-bearing
  detection for the fill class, as designed.

### Fixed (texture acceptance — catastrophic-baseline regime: a collapsed baseline is not an A/B reference; the x-wing incident, gate side)

Live incident (x-wing bundle, 2026-07-15; fix program /tmp/xfix2;
concurrent with the pose-lane fix below): the source pose failed
(coverage 0.0095), the no-references baseline failed the single-view
sanity floors, and `evaluate_generated_bake` still used that ~99%-fill
bake as the A/B reference — refusing a healthy candidate (4 accepted
references, IoU 0.81-0.94, artifact battery quiet, fidelity within
slack) on ONE axis: tone brightening 0.814 vs budget 0.7 at az90_el50.
Brightening a fill baseline is what correct references DO; the
directional budgets were calibrated on baselines with witnessed
coverage 0.112-0.83 (fix1's fleet). The user received the fill
texture.

- **Regime boundary (measured, not a special case)**: the gate now
  computes both bakes' single-view sanity verdicts from their stats
  (the signal the runtime already records — threaded to the call
  sites, never recomputed) and derives `baseline_regime` from the
  corpus-calibrated registration-collapse line (`artifact_gates`
  floors: source coverage < 0.10 AND capture efficiency < 0.25).
  Healthy/pinned baselines measure >= 0.1088 / 0.2944 (the one
  sub-0.10 live bundle, car_final at 0.0572, is rescued by its 0.3273
  efficiency); the measured catastrophes sit at 0.0096/0.090 (x-wing),
  0.0498/0.166 (v4 ghost), 0.013/0.030 (the broken-chair guards).
  Deliberately NOT the sanity-floors verdict itself: the pinned v7
  baseline fails the 0.12 total floor at 0.112 while fix1 PROVED the
  A/B axes calibrated at that coverage — user-visible degradation and
  A/B-semantic collapse are different, separately measured lines.
- **Catastrophic regime** (baseline collapsed): fidelity, brightness,
  darken and the battery keep their A/B votes (the photo is external
  truth and fill is dark-biased and blotch-free by construction —
  added == absolute, measured 0.0000 baseline blotch on both broken
  baselines). Brighten-A/B records loudly but cannot vote (measured
  0.81-1.03 on CORRECT rescues vs 10.85 on a +25 L mis-tone — same
  sign, no boundary; no absolute replacement exists: honestly-bright
  unseen surface measures 6.26 of above-photo-band mass where the
  mis-tone measures 0.28 — inverted). Hue switches to the ABSOLUTE
  source-band form (`_band_distance_damage`, floor 10 deg, budget
  0.15: catastrophic-lane accepts <= 0.035, the +25 L mis-tone's
  gamut-bend 0.284 = 1.9x over, a bake-dominating 30-deg rotation
  5.18 = 35x; colorless photos keep the legacy A/B charge
  fail-closed). A NEW mirror-consistency axis (`_mirror_pair_damage`,
  az90 vs mirrored az-90 at both elevations, floor 15 L, budget 1.0,
  gated on the bake's own geometry-symmetry score >= 0.95; fleet
  scores 0.966-0.985) catches displaced content: the
  8%-content-shifted back measures 1.496-1.499 in BOTH regimes vs
  fleet accepts <= 0.516 (the starship's honest texture asymmetry).
  The candidate must also measurably FIX the collapse (total coverage
  >= the 0.12 floor and > baseline) or both bakes are broken and the
  baseline ships degraded exactly as before.
- **Metadata honesty**: `metrics["baseline_regime"]` records the
  regime, the collapse values, and both sanity verdicts; every tone /
  hue / mirror metric carries its vote flag; both call sites
  (`rebake_bundle`, `hunyuan3d_runtime._run_generation`) reuse the
  gate's recorded sanity verdict for whichever side ships, and a
  catastrophic-accept ships the candidate with its own floors verdict
  (source rows inherited from the broken registration until the pose
  fix heals them) plus explicit postprocess warnings.
- **X-wing outcome (CPU rebuilds of the shipped bundle + its 4
  persisted references at 2048)**: pre-pose-fix tree — collapse fires
  (0.0096/0.090), the candidate fixes coverage 0.0095 -> 0.3336, every
  voting axis green (fidelity 16.63 -> 17.51 vs max 18.63; brightness
  35.29 -> 42.04; darken 0.0/0.03; hue-abs 0.000/0.15; mirror
  0.000/1.0 at geometry score 0.9853; added blotch 0.0/0.009) and the
  bake ACCEPTS with brighten 0.814 recorded as a loud warning — the
  incident verdict flips for the measured reason. Post-pose-fix tree —
  the pose lane finds (0, +33), the baseline passes the collapse line
  (coverage 0.212/efficiency 0.476) and the gate hands back to the
  normal healthy A/B path seamlessly (verified on the merged tree:
  regime healthy, candidate ACCEPT).
- **Chair-incident guard** (wrong refs must never ship because a bad
  baseline flattered them — here inverted: a bad baseline must not
  damn correct ones either): fix1's chair fixtures rebuilt over a
  DELIBERATELY collapsed baseline (center-occluded source alpha,
  coverage 0.0179). Correct back ref ACCEPTS (previously refused by
  the same poisoned axes: brighten 1.025 + fill-confounded hue 1.038);
  +25 L mis-tone REFUSES via absolute hue 0.282; 8% content shift
  REFUSES via mirror consistency 1.495; 30-deg hue rotation REFUSES
  via absolute hue 5.18. Measured limits recorded (KnowledgeBase): the
  bright-side pure-L mis-tone and, under degraded photo evidence, the
  -25 L dark mis-tone present no absolute evidence separable from
  honest subjects (the portrait's legitimate back ref sits -25.9 L
  from its photo) — the healthy lane keeps refusing those classes
  decisively (pinned +/-25 L verdicts unchanged, margins 4x/11x).
- **Pins**: all 12 fix1 fixture verdicts, hue1's 3 chroma-rotation
  REFUSEs + live pairs (car_bo3 vetoed 0.008 / raw 1.870 reproduced to
  the third digit), the hue15 probe, gfix2's gate rows
  (fresh_car/v7/owl full @2048) and afix2's battery calibration
  (v7cand_stamp 0.0245 refuses; all 12 accept pairs quiet) hold on the
  landed gate — every pinned baseline stays in the healthy regime by
  measurement, so the healthy path is bit-unchanged. Suite green with
  5 new regime tests; parity canaries 3/3 with the historical md5
  pins.

### Fixed (source pose — elevated captures outside the rescue lane's search band; the x-wing incident)

Live user incident (`out/x-wing`): the t2i source photo is a
top-three-quarter capture (true camera ~el +33), a routine real-world
class (top-down product photos, aerial vehicles, figurines shot from
above). The NCC spike was rightly vetoed by the silhouette guard
(`ncc_vetoed_by_silhouette`, score_at_declared -0.064), but the rescue
lane only searched el {-15..+15}, so the bake shipped declared (0,0):
observed coverage 0.0095 / capture efficiency 0.090 — the texture was
fill. Measured landscape (extended grid, el +/-60): the true basin is
(az 0, el +30) registered-silhouette IoU 0.857 vs 0.734 best-in-band
(below the 0.75 action floor) and 0.617 declared.

Fix (`estimate_pose_with_silhouette_guard`): EXTENDED ELEVATION SEARCH,
coarse-to-fine, consulted ONLY when the calibrated band cannot itself
justify action (in-band best registered IoU < `rescue_min_best_riou`
0.75). Below that floor the guard probes elevation tiers +/-25/40/55
(full azimuth window at step 10), refines one grid-resolution
neighborhood (az +/-5, el +/-8) around the probe argmax, and merges the
evidence into the UNCHANGED veto/override/rescue logic; the trail
records `extended_search`. Above the floor the photo is explained
within the band and the extension never runs — measured on the recorded
fleet (in-band best 0.80-0.97 everywhere), so every calibrated verdict
keeps a bit-identical decision trail and pays zero extra renders. The
double-key thresholds transfer to the high-elevation basin without
retuning, with measured margins: riou gap 0.234 (2.3x the 0.10 key),
declared aspect err 1.093 (7.3x the 0.15 key; the frontal render's
bbox aspect is ~3x the elevated photo's — wings edge-on vs plan view),
basin best 0.851 vs
floor 0.75, and the basin pose's own aspect err 0.014-0.10 stays safely
below the commit-side override key. Budget: worst case ~74 extra
renders+registrations, ~4-6 s CPU on a 120k-face mesh, on top of the
~105-pose calibrated band (measured guard wall time 10.2 -> 14.4 s on
the x-wing; fleet cases unchanged at 10-11 s).

Results: x-wing pose (0,0) -> silhouette_rescue **(0, +33)** (riou
0.851); source-only CPU rebake coverage **0.0095 -> 0.2135 (22x)**,
capture efficiency 0.090 -> 0.475 (floors pass). Fleet regression
(12 cases): fresh car (40,15), gfix3 draw A ncc (25.9,15), draw B
rescue (25,8), v7 ncc (17.5,8), v4 rescue (+20,0), v2 rescue (+30,8),
starship ncc (30,15), portrait ncc (20,8), owl/chair/face declared —
all reproduce exactly (trail-identical); parity canaries 3/3. Synthetic
controls: fleet meshes rendered at known elevated poses recover within
one grid step (car (10,40) exact; x-wing (5,45) -> (5,48)); a
round/pose-tolerant subject (owl at el 40) keeps declared honestly —
its in-band registration is 0.868, above the floor, and pose barely
moves its silhouette (the projection cost of staying is small by the
same measure). Characterized limit (recorded, not silently widened):
a capture whose opposite-sign elevation MIRROR registers in-band just
above the floor (synthetic car at el -35: its el +15 mirror registers
0.763) ships the in-band rescue exactly as before — catching it would
require far-band candidates to join calibrated plateaus at
jitter-scale margins, which the recorded fleet forbids. The NCC lane's
elevation candidates are measured NOT necessary to extend (its validity
assumption is already broken on this class — score_at_declared -0.064 —
and the rescue lands within one grid step of the coverage optimum
without it); left untouched. New tests:
`test_pose_guard_recovers_elevated_capture_via_extended_search`,
`test_pose_guard_extension_stays_out_of_calibrated_band_decisions`.

### Validated (five-agent artifact program — integration matrix on the merged tree; end-to-end proof pack)

Integrator-validator pass (2026-07-14, /tmp/afix5) over the four
concurrent landings: A1 roof-block C0 tone fields (`texturing.py`), A2
artifact-detector battery (`artifact_gates.py` + gate/QA wiring), A3
multi-view geometry conditioning (`hunyuan3d_runtime.py`), A4 reference
resolution/sampling lanes (`reference_generation.py`,
`triposr_runtime.py`). No neutralizations were needed: no fleet case
regressed on the merged tree.

- **Suite**: 335 passed at entry -> **375 passed, 3 skipped, 3 xfailed**
  on the final merged tree (+40 tests from the four programs; zero
  failures at every intermediate landing). Parity canaries **3/3** with
  the historical pins (P1 identical_frac 1.000, pose (17.5, 8.0); P2
  c84f2e49…; P3 face e32ba995… twice) — the single-photo lane is
  bit-untouched by the whole program.
- **Pinned fixtures** (gate-only, persisted GLBs): all 16 rows hold
  under the battery-active gate — 12 fix1 verdicts, 3 chroma rotation
  REFUSEs (1.312/1.978/6.508 vs budget 1.0), hue15 probe 0.623 ACCEPT.
- **Pairs rebuilt from scratch on the merged tree** (2048, production
  kwargs): car_bo3 ACCEPT (fidelity 32.09 -> 32.06/34.09; hue raw 1.852
  vetoed to 0.011 — hue1's headline reproduces), car_final ACCEPT
  (32.31 -> 33.35/34.31), owl ACCEPT (17.02 -> 17.24/19.02), v7 ACCEPT
  (34.06 -> 32.66 — candidate more photo-faithful than baseline).
  Battery quiet on all four (added blotch <= 0.0024 vs budget 0.009);
  the v7_candidate stamp pair refuses on BOTH fidelity (41.73) and the
  battery axis (0.0245) on the merged tree. A1's roof-block kill
  verified on the rebuilt live pair: el-55 canopy clean, no lattice.
- **End-to-end proof matrix (MPS, sequential)**:
  - **Car t23d `--quality best`** (best-of-3; A3's measured
    recommendation keeps geometry conditioning OUT of presets): exit 0
    healthy, seed 3025 selected (score 1.0761 vs 1.0713/1.063), no
    slab, 1 body; 4/4 references accepted; whole-bake **ACCEPT**
    (fidelity 33.75 -> 33.72/35.75, battery blotch 0.00095/0.009, zero
    warnings). Hostile turnaround + top + crops: NO mini-photo stamp,
    NO roof blocks, NO white rim blotches, NO mottle patchwork, NO
    visible seams, NO slab, wheels attached, cabin intact.
  - **Owl i23d** (references on): healthy, geometry md5 c5d2409b… ==
    the approved standard (fourth byte-identical shape reproduction);
    4/4 refs accepted, whole-bake ACCEPT 17.02 -> 17.24/19.02 (A1's
    CPU prediction to the digit), battery quiet, texture_qa PASS.
  - **Starship rebake** (persisted refs, merged tree): ACCEPT 19.77 ->
    19.51/21.77, tone 0/0.03, battery 0.0, seam informational
    0.0089 -> 0.0088.
  - **Fresh chair t23d** (default path): healthy, watertight single
    body, no slab; per-view gates honestly rejected 3/4 reference
    draws, accepted top; whole-bake ACCEPT with fidelity IMPROVING
    26.54 -> 24.87/28.54; battery quiet. Residuals recorded (below).
- **A4 validation spec executed** (the generation runs it specced):
  per-angle `render_size="auto"` PASSES its decision rule on the car
  (4/4 accepts within attempts, sharpness 1.51x back / 1.99x top at a
  common scale, rebake ACCEPT with fidelity 31.03 better than the
  768-ref candidate's 32.06, cost 2.46x <= 3x) but FAILS on the owl:
  0/12 accepts at 1024-1216 — silhouette healthy (0.90-0.97), the
  fixed-pixel relief/flat texture bands refuse every draw. **Default
  stays 768** per the rule; the recorded fix path is gate-band
  rescaling by frame ratio (KnowledgeBase insight), not size retreat.
- **Honest residuals** (tracked, none in the 8 historical artifact
  classes): (a) refs-on car bakes carry 10-13 sub-visible dark
  micro-fragments in never-witnessed underbody concavities — measured
  pre-existing (gfix3 car_b: 10 on the pre-program tree), invisible in
  all product views, next step in KnowledgeBase; (b) the fresh chair
  draw's bake reads darker than its hi-key photo (viewer brightness
  0.26 vs the 0.72 QA floor; coverage 0.047 on the thin-spindle
  subject) — a pre-existing chair-class weakness, not a regression:
  the APPROVED legacy chair proof measures 0.388 on the same gate
  (with 9 legacy-era gate failures inc. baseColorFactor 0.4); the A/B
  acceptance gate is structurally blind to a defect both arms share,
  and the standalone QA catches it loudly. Candidate ship-time lever:
  a viewer-brightness floor in the single-view sanity battery, run on
  BOTH branches; (c) away-from-photo surfaces remain softer than
  photo-witnessed ones (A4's density table names the mechanism;
  quality-weighted sovereignty specced as the successor program).

### Added (artifact-detector battery — measured detectors for every shipped artifact class; one voting axis, the rest recorded loudly)

Adversarial fix program 2026-07-14 (/tmp/afix2): the user's demand is
"no more artifacts like all those we have seen so far", so every
artifact class this project has shipped got a detector, calibrated on
the preserved corpus (bad exemplars: the sportscar_v7_candidate
mini-photo stamp, the car_bo3 roof patch-blocks, the diagsub white rim
blotches, the v5/v6/maxvis fill mottle and clear-coat smears, the v4
coverage-0.05 ghost; good corpus with ZERO false fires required: owl
standards, face proof, starship, chair, portrait, sportscar_v7, gfix3
car_a/car_b, car_bo3 minus its known-bad roof). New module
`src/abstract3d/artifact_gates.py`; wiring in
`bake_acceptance.evaluate_generated_bake` (A/B battery on the existing
turnaround renders — a candidate must not ADD artifact mass, same
directional doctrine as the tone axes) and `scripts/texture_qa.py`
(standalone battery per bundle, warn-only).

- **Voting axis — added foreign pale blotch** (classes: image-in-image
  stamp payload + white/foreign rim blotches): compact bright
  desaturated components vs the local surface context (context sigma
  6% of the subject diagonal, dL > 18, dC > 18, area >= 0.15%,
  elongation <= 6 — the measured strip/blob boundary: legit tail-light
  bands and sills measure 8-18, every bad-class component 1.0-5.6).
  Worst-view ADDED max-component budget 0.009: labeled-accept pairs
  measure <= 0.0036 (9 of 12 fix1 pairs exactly 0.0000; pinned-accept
  car_bo3 0.0026), the rebuilt stamp incident measures 0.0244 — budget
  2.5x above the accepts, 2.7x under the incident. The stamp pair is
  ALSO refused independently by photo fidelity (41.73 vs 34.06+2.0 at
  the true pose), so the battery is defense-in-depth there; it is the
  only axis that catches the same payload landing on never-witnessed
  surface (synthetic back-side stamp fixture in the tests).
- **Recorded + loud warnings (measured to be non-votable)**:
  - added pale WASH (baked specular / clear-coat smear class), warn
    at +0.010: fix1 accepts measure <= 0.0018, but the pinned-ACCEPT
    car_bo3 pair adds 0.0133 (its roof blocks read as wash — the
    warning names the exact live defect the fringe-repair line is
    fixing), so a vote would flip a pinned verdict.
  - added RECT-HAZE CELLS (patch-block grids), warn at +2: integer
    count with a measured +/-1 noise floor on the accept corpus.
  - added dark patchwork: recorded only — legitimate references add
    up to +0.063 of mid-band structure (real glass frames over blank
    fill), far above any damage exemplar.
- **Why template matching does NOT carry class 1a** (honest dead end,
  measured): five variants (raw/gradient full-frame NCC, 3x3 part-tile
  consistency voting, own-scale-excluded search, texture- and
  render-space) all fail — the legitimate bake IS a photo copy at its
  own scale, texture space carries the atlas packing pitch (~53 px
  periodicity in EVERY texture) and coherent reference content, and
  the incident stamp is warped (its registration was degenerate by
  construction), which caps rigid-template NCC below the legitimate
  matches. The stamp's invariant payload is its baked-in background
  ring — which the blotch axis measures directly.
- **texture_qa standalone battery** (any bundle, single-photo bakes
  included): 8-view turnaround battery + two stats-based checks —
  fill-cap mottle risk (fill-detail scale pegged at its 3.0 cap on a
  >= 50%-fill subject: the measured 1024-mottle CAUSE; maxvis fires at
  3.0/0.839, v7@2048 measures 1.109) and registration floors (source
  coverage < 0.10 AND capture efficiency < 0.25: the v4 ghost fires at
  0.0498/0.166; weakest good bundle v7 at 0.1088/0.294). All
  standalone render detectors WARN only: their absolute good-corpus
  margins (1.14-1.8x) sit below the project's zero-false-fire voting
  bar — the A/B form in the gate is where they can vote. Measured
  standalone: owl PASS with battery quiet; v7_candidate warns blotch
  0.0340 + background-contamination 0.0306 + registration floors;
  maxvis warns wash 0.0703 + fill-cap; exit codes of the existing
  good-corpus gates unchanged.
- **Known misses, stated**: absolute wash cannot catch v6 (0.0379)
  or the bo3 roof (0.0376) under the good-corpus max (0.0450 — v7's
  own baked photo speculars are the same physics); diagsub_top_rear /
  side_right (0.0120/0.0086) sit under the blotch warn line; the
  dark-patchwork render statistic does not separate maxvis (0.0952)
  from the labeled-good v7 (0.0925) at all — that class is warned on
  its recorded CAUSE instead.
- **Pinned verdicts proven unchanged** with the battery voting: all 12
  fix1 fixture pairs, the fix3 chroma-rotation pairs, hue1's live
  car_bo3 (ACCEPT) / car_final pairs — reruns of the fix programs'
  own harnesses; margins in /tmp/afix2/report.md.
- **Tests**: `tests/test_artifact_gates.py` (9 synthetic fixtures: per
  class fire/quiet, strip-vs-blob elongation, A/B direction — added
  fires, inherited/removed never; stats-check incident values) plus 2
  end-to-end gate tests (a back-side stamp is refused by the battery
  ALONE; inherited blotches ship). `evaluate_generated_bake` reports
  gain a `warnings` list (recorded verbatim into bundle metadata by
  the existing call sites).

### Added (multi-view geometry conditioning for single-photo flows — `geometry_conditioning`)

Fix program 2026-07-14 (/tmp/afix3). Both mesh audits (2026-07) convicted
CONDITIONING STARVATION as the primary cause of hallucinated geometry on
self-occluding subjects (melted cockpit interiors, detached wheels, deck
blobs on cars) and ranked "feed the shape stage more views" as the first
fix: the `Hunyuan3D-2mv` dict path was already integrated and proven (face
proof), but it only fired when the CALLER passed reference views — which
single-photo `t23d`/`i23d` never do. The new `geometry_conditioning`
option (`single` default / `multiview` / `auto`; config
`scene3d_hunyuan_geometry_conditioning`, env
`ABSTRACT3D_HUNYUAN_GEOMETRY_CONDITIONING`, CLI `--geometry-conditioning`)
closes the gap at the source:

- **Pre-shape view synthesis (meshless)**: before any mesh exists, the
  missing canonical views (`back`, `side_left`, `side_right` per the 2mv
  tag map; caller-provided references keep their tags first) are
  synthesized from the matted source photo with the LOCAL i2i generator
  (rotate-style prompt, material-free subject noun from the caption/hint,
  shared negative prompt; seeds `base + 50000 + 1000*i`, 8/12-step
  ladder, 2 attempts per view).
- **Gated before trusted** (a wrong conditioning view is worse than
  single-view — the checkpoint TRUSTS its tags): matte sanity, subject
  identity via `part_material_fidelity`'s floor line (chroma-collapse
  guard included), and two orthographic silhouette identities — the BACK
  silhouette of any object is its mirrored FRONT silhouette (floor 0.52),
  and LEFT/RIGHT silhouettes are exact mirrors of each other, so a
  surviving side pair that disagrees (mirror-IoU < 0.68) is dropped WHOLE
  (blame between the two is unattributable). Floors calibrated on 24 real
  draws across 4 spaced seeds on a frontal (owl) and a three-quarter
  (car) source: healthy backs 0.954-0.976 (frontal) / 0.580-0.678 (3/4 —
  the source's own off-axis angle is the relation's noise floor);
  wrong-subject swaps 0.398-0.437; healthy pairs 0.729-0.953;
  front-echo-as-side lies on the elongated class 0.615-0.658.
- **Loud fallback**: person subjects are refused under the same doctrine
  as texture reference generation (caption + `is_person_subject`,
  fail-closed when the captioner is unavailable,
  `texture_reference_allow_person` is the person-specific attestation for
  BOTH lanes — it attests the same act); zero surviving views falls back
  to the exact single-view flagship path with a warning plus a
  machine-readable `geometry_conditioning` metadata block (requested vs
  applied, per-view gate metrics, fallback reason). `auto` additionally
  requires an explicitly configured image provider (never silently
  remote) and yields to an explicitly requested single-view model;
  explicit `multiview` + an explicit `Hunyuan3D-2.1` model is an
  `InvalidRequestError` (contradictory explicit requests are not silently
  resolved).
- **Model selection**: with conditioning views in hand the shape stage
  loads `tencent/Hunyuan3D-2mv` (auto-selected only when no explicit
  model was given); the DiT load is DEFERRED until after synthesis so the
  i2i pool and the DiT never co-reside on the unified-memory pool.
- **Texture lane**: accepted synthesized views are OFFERED to the texture
  bake as the FIRST ladder attempt of their angle through the full,
  unmodified reference-acceptance machinery (matting, clay-silhouette
  registration + IoU, texture/material/specular gates, whole-bake A/B) —
  nothing is bypassed; a view the mesh diverged from fails the clay gate
  and the ladder regenerates normally (`replayed_labels` recorded).
- **Persist-for-diagnosis**: accepted views land in the bundle as
  `geometry_view_synthesized_*.png`, rejected candidates downscaled under
  `rejected_geometry_views/` (budget-capped), all gate metrics in
  metadata.
- **Host hardening**: 2/2 synthesis probes died SIGSEGV (exit 139) in
  skimage `lab2rgb` (float64 GEMM -> Accelerate `cblas_dgemm`) with the
  MLX pool resident, even under `VECLIB_MAXIMUM_THREADS=1` (KB: "Host:
  Accelerate BLAS segfaults"); `_harden_skimage_color_convert()` swaps
  skimage's 3x3 color matmul for an einsum (verified identical to 1e-12
  before the swap; darwin-only; applied only on the multiview path).
  After the swap: 24/24 probe generations completed.
- **2mv family regime + hard view cap (both measured)**: the 2mv
  checkpoint now runs its OWN validated regime (30 steps / octree 384 —
  the model-card snippet and the checked face proof) instead of
  inheriting the flagship's 512/50, on every 2mv route (synthesized or
  caller references; explicit options and configured owner defaults still
  win). And conditioning is HARD-CAPPED at 3 simultaneous views with
  priority front > back > side_left > side_right: on identical
  conditioning images, seed, and settings, the 4-view dict shredded the
  field into film-shell debris TWICE independently (559 raw bodies /
  euler +693 at 384/30; 822 raw bodies / +887 at 512/50) while every 1-3
  view subset produced a healthy single-body car — and 2-3 views beat
  single view (raw euler: front-only -174, front+back -141,
  front+left+back -117, front+sides -65). Every proven upstream usage
  stays at <= 3 views; the cap drops the lowest-priority tag loudly
  (warning + `dropped_views` in metadata) and a dropped view still
  reaches the texture lane through the replay.
- **Quality presets**: deliberately NOT routed through multiview (see the
  A/B below — it is a measured trade, not an upgrade); presets keep their
  shape-candidate counts and `geometry_conditioning` stays an explicit
  opt-in.
- **Default path untouched**: `geometry_conditioning` unset/`single` is
  the historical byte-identical flow (no person gate, no synthesis, no
  new metadata keys — pinned by test), and the caller-supplied-reference
  2mv path is unchanged apart from an equivalent nearest-tag refactor
  (`_mv_snap_tag`, pinned by test) and the two measured protections
  above.

A/B (2026-07-14, MPS, sequential, geometry-only, same base seed per pair,
each arm at its shipping regime — flagship 512/50 single-view vs 2mv
384/30 with the capped 3 views; H-program ranking metrics vs the source
matte; per-draw table + clay sheets in the A3 program report): multiview
cut the car's spurious-handle load 32-61% (euler -130 vs -192 at seed
2025, -74 vs -132 at 3025; raw -117 vs -179, -48 vs -124), smoothed
panels (dihedral RMS -2.2/-2.4 deg), conditioned the rear instead of
inventing it, and kept the owl standard's perfect topology (euler +2 both
arms) at photo-IoU parity (max delta 0.013) and wall-time parity. It
consistently LOST concave sharpness (photo concavity IoU -0.05 on cars,
-0.10 on the owl) and fine carved detail, putting the calibrated
composite score 0.018-0.040 behind single-view on every pair; a control
at octree 512 with the capped 3 views recovered only 0.013 of the car's
0.051 concavity gap while adding a detached body — the loss is mostly
the 1.1B 2.0-family checkpoint, not the family grid. Verdict: a real
topology/panel win and a real detail cost, so it ships as an explicit
opt-in for topology-critical subjects, not as a preset default.

### Added (reference resolution budget: measured density tables, per-angle synthesis sizing, footprint-aware projection sampling)

Resolution/sampling lane of the roof-patchwork program (A4, 2026-07-14,
/tmp/afix4): measured the texels-per-reference-pixel budget end-to-end
on the shipped car_bo3 bundle and the owl control, then fixed the two
in-scope constraints the measurements convicted. Density method: project
the atlas texel grid into each registered reference frame (the
projector's own sample maps), take the per-texel Jacobian's singular
values (photo px per texel step), and histogram density = 1/sigma over
each view's OWNED roof texels. Findings of record (2048 atlas,
production kwargs):

- The car roof splits ~47% / 41% between the FRONT PHOTO (grazing,
  worst-direction density p50 2.24 / p90 3.16 texels per photo px — a
  2-3x anisotropic smear) and the TOP reference (p50 1.43 / p90 2.06);
  the healthy owl comparator's top reference owns 62% of its crown at
  p50 1.22. Ownership of half the roof by the most-starved witness is
  the sovereignty doctrine (`protect_observed_texels` absolute), not a
  sampling bug — recorded for the doctrine's owner.
- EVERY 768-frame generated reference under-delivers the canonical
  conditioning frame (1024 x 0.85 border = 870 px subject side) by
  ~1.36x: the adaptive clay framing letterboxes the subject to ~0.83 of
  any frame at any angle, so synthesis carries ~637 true px into an
  870-px slot before the atlas ever samples it.
- The distinctive BLOCK mosaic on the dark canopy was NOT sampling:
  the projector's bilinear output is smooth; the blocks entered at
  `equalize_projection_tone` stage 2 (voxel-lattice field — root-caused
  and fixed by the parallel afix1 program; independently confirmed here
  by stage bisection, and the pre/post A/B crop ships in
  /tmp/afix4/tone_blocks_pre_vs_post_afix1.png).

Changes:

- `reference_generation.generate_reference_views` accepts
  `render_size="auto"`: per-angle frame sizing that closes the measured
  letterbox deficit — each angle's clay silhouette extent (the exact
  framing the generation reproduces) sets the frame so the subject's
  true pixels meet the canonical 870-px demand
  (`reference_render_size`: ceil to 64, never below the 768 base,
  capped at 1280). Int values keep the historical single-size behavior
  byte-for-byte; the chosen size and fill fraction are recorded per
  angle. Default remains 768 pending A5's MPS validation of gate
  pass-rates at 1024+ synthesis (spec in /tmp/afix4/validation_spec.md).
- `_tripo_project_observed_texture` gains `sample_filter="footprint"`
  (`_tripo_footprint_filtered_colors`): anisotropic area-average for
  MINIFIED texels — mip level by sigma_min (the well-sampled direction
  stays sharp), Gaussian probes along the major footprint axis,
  alpha-premultiplied so matte background never bleeds into rims;
  magnified texels and every alpha/visibility decision keep the exact
  bilinear values. OFF by default (default path structurally unchanged;
  parity canaries 3/3 with pinned md5s, face P3 bit-identical).
  Measured basis: 1024 fleet bakes sample references at sigma_max ~1.9
  on roof regions (past texel Nyquist — a checkerboard reference at
  grazing aliases into stable false blocks, now pinned by a synthetic
  regression test), while 2048 bakes sit at sigma ~0.8-1.05 where the
  filter is inert by design. Refs-on A/B under production kwargs
  (persisted car_bo3 refs, 1024): texture delta mean 1.6/255, p99 14 —
  the aliasing band; renders neutral-or-cleaner (crops in /tmp/afix4/).
  Call-site wiring (generated views only) is specced for the validator,
  not landed — texturing.py belongs to a parallel lane this cycle.
- Tests: checkerboard-at-grazing de-aliasing + resolvable-detail
  preservation + magnification bit-identity for the footprint filter;
  per-angle size computation (letterbox deficit closure, quantization,
  cap, degenerate masks) and end-to-end "auto" plumbing with a mock
  generator; unknown render_size modes are rejected.

### Fixed (roof-block "image inside an image" — the tone-consensus field printed its voxel lattice)

Fix program 2026-07-14 (/tmp/afix1), second stamp-class incident,
user-reported on the shipped best-of-3 car (/tmp/hue1/car_bo3_refs):
rectangular blocks of foreign-looking image content across the
roof/glass at elevated views. Provenance forensics (per-stage write
masks over an instrumented rebake that reproduces the shipped texture
md5-identically): the blocks enter BETWEEN projection and blending —
`equalize_projection_tone` stage 2 (the local consensus field) wrote
293k texels of the top reference's projection with a multiplicative
exposure field SATURATED at its +-0.5 log cap, and the field was
piecewise-CONSTANT over the voxel lattice of `_voxel_neighborhood_mean`
(cell 0.03 x diagonal ~ 0.069 world units): a flat car roof intersects
that lattice in axis-aligned world-space rectangles, so adjacent cells
stepped by up to e^1.0 in luminance and rendered as exposure BLOCKS
(measured: within-cell field std 0.47x total — most of the "field" was
the lattice; the registered top photo's canopy is smooth). The blocks
survive the gradient-domain composite (it equalizes across ownership
boundaries, not inside one view's witnessed interior) and ship. The
first stamp incident (miniature photo on the hood) was a COPY lane
with a degenerate correspondence; this one is an EXPOSURE lane whose
correction field itself carried fabricated spatial structure.

- **Fix (`texturing._voxel_field_mean_c0` + the three field builders)**:
  statistics that become multiplicative correction fields must be C0.
  New trilinearly-interpolated variant of the voxel box mean (same
  binning, same 3x3x3 zero-padded box sums, same NaN contract;
  evaluation interpolates the cell-centered means with unoccupied
  corners at zero weight — box-mean-exact at cell centers, continuous
  everywhere by construction). Used by: `equalize_projection_tone`
  stage 2 (deviation numerator, evidence density, fade density — plus a
  new SUPPORT-EDGE fade `clip(density / fade_density_full)` closing the
  residual hard step where the ratio field meets its evidence-support
  boundary), stage 1's overlap-proximity fade, and
  `delight_projections`' fade. The box-mean variant remains for every
  gate/decision consumer (fill floor, film band, rescue detection —
  calibrated statistics pinned by the single-photo canaries).
- **Also removed**: the `fade[fit_overlap] = 1.0` hard overrides in
  delight/tone stage 1 — forcing full field strength on scattered
  overlap texels steps against their faded neighbors (same artifact
  class at texel scale); dense overlap bands reach fade 1.0 through
  their own density (full fade at 12% ball occupancy), sparse slivers
  now honestly keep partial correction.
- **Provenance (stats name every writer)**: delight/tone accept
  branches now record `written_texels` (the write-mask footprint) in
  their per-view stats rows; a refused row records nothing and the
  projection is bit-identical (pinned by test).
- **Measured on the live pair**: candidate rebake under the shipped
  recipe — roof blocks GONE (before/after el-55 renders + close crops
  in /tmp/afix1/report.md); tone stage 2 still applies (it is the
  measured-good lane that reconciles synthesized regional shading; its
  field is now smooth), witness-ranked accept gates unchanged.
  Single-photo canaries structurally untouched (delight/tone are
  multi-view-only): parity canaries 3/3 with pinned md5s; face
  P3 md5 bit-identical. First-incident regression (sportscar_v7
  candidate rebake): no hood mini-photo, fringe-repair fail-closed veto
  unchanged.
- **Residual, out of this fix's scope (recorded honestly)**: the
  shipped car_bo3 scene.glb was baked by a harness that OMITTED the
  production kwargs (perspective projection at the declared az0 pose
  instead of orthographic + estimated az35/el15), which displaced
  front-photo content onto the roof glass under `protect_observed_
  texels`' absolute sovereignty (the photo's 0.007-weight displaced
  claims beat the top reference's 0.59 head-on claims by doctrine).
  Under production kwargs that class does not exist; ship-path harnesses
  must use the bundle rebake kwargs. Same-class watch items for their
  owners: `film_band.retone_film_band` gathers COLOR targets with the
  box statistic (commit-band-scoped, dominance-scaled — no block
  sighting on the fleet; flagged), `enforce_fill_luminance_floor` lifts
  toward box-mean floors (fill-only, minority-gated, canary-pinned).
- **Tests**: +3 (`test_voxel_field_mean_c0_is_continuous_and_center_
  exact` — continuity across cell boundaries, center-exactness vs the
  box mean, NaN contract; `test_equalize_projection_tone_field_prints_
  no_lattice_blocks` — the displaced-content block class: applied field
  max adjacent-texel log step 0.069 vs 0.372 pre-fix on the same
  fixture, provenance rows mandatory;
  `test_equalize_projection_tone_refusal_is_bit_identical_no_op` —
  fail-closed honesty: refused rows claim no write and the projection
  is bit-identical).

### Fixed (whole-bake hue axis false refusal on never-witnessed surface: two-population source-evidence veto)

Fix program 2026-07-13 (/tmp/hue1), first live sighting of the gfix3
watch item: the composition hue axis REFUSED a correct 4-reference car
candidate (best-of-3 run, /tmp/hfix2/car_bo3) at 1.869 deg-mass vs the
1.0 budget (worst view az180_el10, p95 rotation 19.1 deg) and shipped
the fill-mottled baseline — while all four references were verified
hue-faithful to the input photo. Mechanism (measured on the rebuilt
pair, which reproduces the live 1.869 to the third digit): at az180 the
source photo witnesses nothing, so the axis compared candidate
REFERENCE content hue against baseline FILL-mottle hue and charged the
improvement as damage — the same baseline-side confound class that
retired the seam axis's vote.

- **Fix (`bake_acceptance.py`, two-population hue judgment)**: the
  drift charge `max(angle - 10, 0)` now stands only where the
  candidate's own smoothed hue ALSO sits off the source photo's hue
  evidence (`_source_hue_band`: circular [q2, q98] of the photo's
  saturation-gated a/b fields, smoothed at the axis sigma rescaled to
  the photo's foreground extent) by more than `hue_evidence_margin_deg`
  (3). Pixels ON the subject's own hue evidence are the added-content
  population — never damage, whatever the baseline carries there;
  off-evidence pixels keep the full baseline-drift charge. A colorless
  photo yields no band and the legacy single-population charge applies
  unchanged (fail-closed; measured: the starship photo has no
  saturated mass and its verdict is bit-unchanged).
- **Why not co-observed masking** (the obvious fix): a hue-ROTATED
  reference also lands on baseline-fill surface, so restricting the
  comparison to co-observed pixels would let the pinned rotation
  fixtures ship — reopening the exact hole the axis exists to close.
  The evidence veto cannot: rotated content sits off the photo band by
  construction (geometric texture-space observedness masks were also
  measured NOT exposed by bake stats, and a gate-side projector
  replica would drift from bake truth on every per-texel gate).
- **Calibration** (all pairs rebuilt CPU-only at production kwargs;
  margins vs the unchanged 1.0 budget): live car_bo3 pair 1.870 raw ->
  **0.008 ACCEPT** (125x under); fix3 fresh-draw car_final hue axis
  0.973 raw -> 0.000 (its refusal now carried by the honest
  fidelity/brightness/tone axes alone, unchanged); pinned rotations
  REFUSE with 20-deg back **1.312** (1.3x over), 30-deg back 1.978,
  30-deg sides 6.508; new 15-deg probe 0.623 ACCEPT — 4x above the
  worst labeled accept (car 0.157), so the refusal boundary lands
  between 1.5x and 2x the sanctioned 10-deg drift. Margin sweep: at
  margin 2 the live pair keeps 0.265 of band-edge noise; at margin 4
  the 20-deg refusal thins to 1.11x; 3 is the measured saddle. All 12
  fix1 fixture verdicts unchanged (accepts' vetoed masses only drop:
  worst 0.460 -> 0.157; the four tone refusals refuse identically).
- **Observability**: `composition_hue_damage` now records `worst_raw`
  (pre-veto mass), `worst_raw_view`, `evidence_margin_deg`, and the
  `source_band` actually used — a large raw/vetoed gap is the
  fill-confound signature, visible in every verdict.
- **Tests**: +2 (`test_fill_hue_confound_matching_source_hue_is_accepted`
  — the false-refusal population as a synthetic fixture, raw charge
  over budget, vetoed verdict ACCEPT;
  `test_colorless_photo_keeps_legacy_hue_charge` — no evidence, no
  veto). `test_constant_l_hue_rotation_is_rejected` additionally pins
  that the veto leaves the rotation charge standing (the trap case).
  Suite 335 passed / 3 skipped / 3 xfailed; parity canaries 3/3.
- Note for reproduction: the 30-deg-back pair measures 2.114 on a quiet
  host vs the 2.310 recorded under concurrent load (render-environment
  sensitivity of the offscreen GL pass, both far over budget); the
  20-deg and 30-deg-sides fixtures reproduce their pinned digits
  exactly.

### Validated (adversarial pass over the best-of-N shape-candidate landing; one ranking-weight defect found and fixed)

Independent adversary-validator program (2026-07-13, /tmp/hfix2): the
ranking problem was calibrated BEFORE reading the implementation (spec
with measured inequalities in /tmp/hfix2/ranking_spec.md; trap fixtures
built from the persisted fleet — laplacian melt-severity ladder of the
real car_b draw at 8/40/150/400 iterations, synthetic watertight slab,
the real pre-cutter slab car and its production-cutter repair), then the
landed ranker was attacked, then the feature was run end-to-end on MPS.

- **Defect found and fixed (ranking weights)**: the original
  `0.10 * smoothness` term was a monotone reward for the melt direction.
  Measured on the real-geometry melt ladder: melting strictly improves
  dihedral RMS (15.09° -> 11.93/10.34/9.89° at 8/40/150 iterations) while
  quality strictly degrades, and the anti-melt concavity axis does NOT
  collapse on realistic melts (true 0.2355 vs 0.2696/0.2398/0.2693 —
  only the convex-hull extreme collapses to 0.0272), so at +0.10 the
  40-iteration melt outscored the true draw by +0.0101 and the builder's
  own 60-iteration calibration melt by +0.0181. Fix:
  `_SHAPE_RANK_WEIGHT_SMOOTHNESS` 0.10 -> 0.0 (dihedral RMS stays
  recorded per candidate as a diagnostic). Measured cost on legitimate
  orderings: none (car_b > car_a +0.102 -> +0.096, now decided by
  watertightness; v7 > v4, slab, hull, wrong-subject margins unchanged
  or wider — the hull is still rejected at -0.069). New regression test
  `test_shape_ranking_never_rewards_smoothing` (a laplacian melt of a
  lumpy sphere — smoother by >0.5 — must not outrank the true surface,
  and the score must be invariant to the smoothness value). Residual
  documented honestly: an 8-iteration melt still ties +0.0116 ABOVE the
  true draw via concavity-axis noise (good-draw concavity spread 0.24-
  0.40 is 5x that delta); no measured 2D signal separates that class
  (IoU at 256/512 px, contour distance, complexity matching, interior
  gradient NCC vs its own ±0.1 pose jitter all fail), so the achievable
  guarantee is non-preference, which the zero weight provides.
- **Ranker verdict vs the adversarial spec** (fixed tree, binding run):
  slab dominance PASS (+0.35 synthetic / +0.13 real pre-vs-post-cut);
  melt non-preference PASS at 40/150/400 iterations (+0.0004/+0.0207/
  +0.1159 to the true draw); genus-neutrality PASS (v7 euler -210
  outranks v4 -106; the approved face proof at -418 is not punished);
  watertightness bounded PASS (a watertight slab cannot outrank a
  non-watertight honest draw); determinism PASS (bit-identical scores
  on re-evaluation).
- **Best-of-3 car end-to-end** (t23d, MPS, `--shape-candidates 3`, exit
  0, healthy, 99.6 min): 3 candidates recorded with full metrics, seeds
  2025/3025/4025; selected = argmax (1.2203 vs 1.2078/1.2146); score
  arithmetic re-verified from recorded metrics; the ranking function
  re-run on the shipped geometry reproduces the selected row bit-exactly
  twice. The three candidates were REDRAWN standalone from their
  recorded seeds: the selected draw's vertices/faces are float-exact
  equal to the shipped geometry (the candidate loop is RNG-hygienic; an
  explicit per-candidate generator isolates draws), the two siblings'
  recorded metrics reproduce exactly, and an independent finer-sweep
  instrument agrees with the selection (0.9338 vs 0.9321/0.9195 —
  the selected draw is also the best car IoU measured across the whole
  persisted fleet, above v7's 0.9243). Cost claim verified: inference
  4597.2 s = exactly the sum of the three draws (22.6-29.7 min each),
  ONE texture stage (21.7 min), ranking 2.0 s total. Texture lane note
  (outside this feature): the whole-bake gate refused the generated
  references on hue (1.869 vs budget 1.0 at az180_el10, coverage 0.20)
  and shipped the baseline loudly — the gfix3 hue-only watch item's
  first live sighting; refused refs persisted and visually plausible.
- **Default-path neutrality PROVEN at the byte level** (owl i23d, no
  flag, exit 0, healthy, 64.8 min): geometry.glb, texture.png, AND
  scene.glb md5-identical to the approved standard (third independent
  byte-identical shape reproduction of the owl across tree states);
  metadata carries no candidate keys and the historical timings keys
  exactly; 4/4 references accepted, whole-bake ACCEPT, fidelity
  17.02 -> 17.27 — the gfix3-recorded numbers to the digit.
- **Host note**: one attempt of the car run died SIGSEGV (exit 139) in
  Accelerate `cblas_dgemm` DURING the texture stage, after all three
  candidate draws completed — the documented pre-existing host class
  (KnowledgeBase), unrelated to this feature; the retry completed.
- **Suite**: 333 passed, 3 skipped, 3 xfailed at exit (entry: 332/3/3);
  parity canaries 3/3 PASS on the fixed tree (P1 pose (17.5, 8.0) exact,
  identical_frac 1.000; P2/P3 md5-stable). No pins re-pinned.

### Added (hunyuan3d21 — best-of-N shape candidate selection with a `--quality` option)

Measured motivation (2026-07-13, /tmp/gfix3 car_a vs car_b — same code,
same settings, different DiT draws): draw A euler -240 / not watertight /
dihedral RMS 17.9° baked to 28.6 ΔE baseline photo-fidelity; draw B euler
-146 / watertight / RMS 15.1° baked to 22.2 ΔE — visibly better everywhere
downstream. Draw luck of the shape DiT is the dominant remaining quality
factor; ranking N draws costs seconds against ~21-28 min per draw.

- **Option surface**: backend option `shape_candidates` (int >= 1, default
  1), config key `scene3d_hunyuan_shape_candidates`, CLI `--shape-candidates`
  plus a `--quality standard|high|best` preset (1/2/3 candidates; the
  explicit flag overrides the preset). Wired through the `list_operations`
  parameter schema under the strict unknown-option contract; other
  backends reject the option loudly. Each extra candidate adds about one
  shape-stage time (~21-28 min measured on MPS at octree 512); the source
  image is drawn ONCE and the texture stage runs ONCE at the ORIGINAL base
  seed (reference generation unchanged). Candidate i draws at
  `seed + 1000*i`, sequentially (MPS unified-memory discipline: one raw
  draw plus the best-so-far survivor in memory; transient buffers freed
  between draws).
- **Ranking metric** (score = `1.0*silhouette_iou + 0.35*concavity_iou +
  0.20*topology`; dihedral-RMS smoothness is recorded per candidate but
  carries no score weight — see the adversarial-validation entry below for
  the measured reason the original 0.10 weight was removed): normalized
  silhouette IoU between the
  candidate's clay silhouette (256 px renders, constant-background
  threshold — no segmentation model on synthetic clay) and the source
  matte, max over a coarse pose grid (azimuth -40°..+40° step 10°,
  elevation {0°, 10°, 20°} — brackets every measured fleet pose; a FINER
  sweep measurably hurts: at 2.5° refinement the lumpy car_a refines above
  car_b, 0.9238 vs 0.9168); concave-detail IoU (convex-hull-minus-mask on
  both sides, at the SAME argmax pose) — the anti-melt axis; topology
  (0.5*watertight + 0.5*single-body). Smoothness (`1 - dihedral_rms/45°`,
  clamped) is computed and recorded per candidate as a diagnostic only.
  Photo terms drop out symmetrically when the matte is unusable
  (recorded as a warning). Ties keep the earlier candidate (closest to the
  base seed).
- **Calibration table** (shipped implementation on the persisted corpus;
  every candidate scored against its own photo matte except the probes,
  which attack car_b's matte):

  | mesh | watertight | body | RMS° | smooth | sil IoU | conc IoU | score |
  |---|---|---|---|---|---|---|---|
  | car_b (good draw) | yes | 1 | 15.09 | 0.665 | 0.9078 | 0.2355 | **1.2567** |
  | car_a (lumpy draw) | no | 1 | 17.90 | 0.602 | 0.8987 | 0.2730 | 1.1545 |
  | sportscar_v7 (good) | yes | 1 | 17.60 | 0.609 | 0.9294 | 0.3995 | **1.3301** |
  | car_final (slab class) | no | 1 | 15.10 | 0.664 | 0.8009 | 0.0573 | 0.9874 |
  | owl proof | yes | 1 | 7.88 | 0.825 | 0.9605 | 0.7548 | 1.5072 |
  | chair proof | yes | 1 | 8.60 | 0.809 | 0.7267 | 0.4326 | 1.1590 |
  | starship proof | yes | 1 | 12.45 | 0.723 | 0.7141 | 0.1339 | 1.0333 |
  | face proof | yes | 1 | 27.57 | 0.387 | 0.8628 | 0.5934 | 1.3092 |
  | ADVERSARY melt60(car_b) | yes | 1 | 10.07 | 0.776 | 0.9029 | 0.2693 | 1.2748 |
  | ADVERSARY hull(car_b) | yes | 1 | 4.57 | 0.898 | 0.9115 | 0.0272 | 1.2109 |
  | wrong subject owl→car matte | yes | 1 | 7.88 | 0.825 | 0.4114 | 0.0511 | 0.7118 |
  | wrong subject chair→car matte | yes | 1 | 8.60 | 0.809 | 0.5315 | 0.0115 | 0.8164 |

  (Score column = the original calibration run, which still carried a
  0.10 smoothness term; shipped scores are the same minus 0.10*smooth —
  orderings quoted below are re-verified at the shipped weights.)
  Orderings all correct: car_b > car_a (+0.096, decided by watertightness
  where photo terms measure within noise), v7 > slab (+0.343),
  car_b > convex hull (+0.069 — the hull WINS silhouette IoU 0.9115 vs
  0.9078 and smoothness 0.898 vs 0.665; only the concavity collapse
  0.2355 -> 0.0272 rejects it), every honest draw > wrong-subject probes
  (>= 0.34 margin). Weight inequalities derived from these measurements:
  smoothness carries zero weight so smoothing buys nothing by
  construction; topology (0.20) decides same-subject ties
  without ever outranking a photo-term class gap (slab: 0.24).
- **Measured honesty bound** (documented in `docs/KnowledgeBase.md`): a
  soft Laplacian melt of the SAME draw (melt60 row) is NOT separable from
  the true draw by single-matte evidence — it keeps the witnessed
  silhouette (0.9029) and its concavity survives (wheels shrink, gaps
  widen: 0.2693). Real bad DiT draws are wrong-silhouette blobs (the
  unit-test adversary), which lose 0.1-0.5 on the photo terms. With the
  smoothness weight at zero, smoothing buys nothing at all — melting is
  never a winning strategy, it is merely not always a losing one (a soft
  melt can still TIE within concavity noise; see the validation entry).
- **Metadata evidence contract**: N > 1 records a `shape_candidates` array
  (per candidate: seed, all ranking metrics, `photo_iou_pose`, raw/post
  topology, postprocess record, per-candidate ground-slab report,
  inference and ranking seconds, `selected` flag; failed draws recorded as
  `status: no_surface` — an unranked discard would repeat the
  rejected-references evidence-destruction mistake), plus top-level
  `shape_seed` and `timings_s.shape_selection`. `seed` stays the BASE
  seed. With N=1 (the fleet default) the path is exactly the historical
  one: no matte extraction, no ranking render, no new metadata keys
  (guarded by tests).
- **Tests**: 14 new backend tests (ranking trap: the blob wins every
  internal metric and must lose on photo agreement; non-watertight
  penalty; seed spacing base/+1000/+2000; candidate metadata; no-surface
  rows; all-fail raise; N=1 untouched with ranking helpers monkeypatched
  to raise; option validation 0/-2/"abc" fail before any draw; config-key
  default) + 4 CLI tests (forwarding, preset mapping, explicit-overrides-
  preset, absence when unset). Full suite green: 332 passed, 3 skipped,
  3 xfailed.

### Validated (integrator pass over the combined ground-slab + photo-sovereignty tree; two fresh car draws ship with references)

Adversarial integrator-validator pass over tonight's two landings
(ground-slab cutter in `hunyuan3d_runtime`; photo-sovereignty
composition fixes in `texturing.py`) on top of yesterday's validated
program. Full report with artifacts: /tmp/gfix3/report.md. No
neutralizations: no fleet case regressed.

- **Test suite**: 314 passed + 3 skipped (opt-in canaries) + 3 xfailed
  at entry and at exit. Parity canaries 3/3 (P1 refs-off 2048 rebake
  bit-identical to the pin, P2 md5 c84f2e49… twice, P3 face md5
  e32ba995… twice) — no re-pin needed: the single-photo lane is
  untouched by the combined tree.
- **Pinned pairs reproduce G2's verdicts exactly** (production
  semantics, 2048, CPU): fresh-draw car ACCEPT (fidelity
  32.31 -> 33.02 vs max 34.31, tone 0/0.0025, hue 0.9726, pose
  silhouette_rescue (40,15)); v7 ACCEPT (34.06 -> 32.64, tone 0/0,
  hue 0.233, ledger p50/p95 0.179/0.389).
- **Car t23d, TWO fresh draws, both ship WITH references** (the
  measured first for fresh draws; user's command + seed-4242 variant,
  MPS, VECLIB_MAXIMUM_THREADS=1): draw A — healthy, no slab drawn
  (`ground_slab: null`), 4/4 refs accepted, whole-bake **ACCEPT**
  (fidelity 28.62 -> 29.30 vs max 30.62, tone 0/0, hue 0.352,
  coverage 0.389); draw B — healthy, no slab, 4/4 refs accepted on
  attempt 1, **ACCEPT** (22.24 -> 22.51 vs max 24.24, hue 0.526,
  coverage 0.430, pose rescue (25,8) score 0.904). Hostile render
  review vs the input photos: photo-true at the source pose, coherent
  reference-painted far sides, no mini-photo stamp, no slab in
  renders; residual cosmetic defect class: low-res patchwork on the
  roof/glass-top region at elevated views (both draws).
- **G2's hue watch item did NOT trigger** (fresh draws measure 0.35
  and 0.53 vs budget 1.0); per the scoped mandate the axis is left
  unchanged and the margins are recorded. The
  baseline-fill-hue-confound class remains a watch item (the pinned
  fresh-car pair sits at 0.97).
- **Owl i23d end-to-end**: healthy, 4/4 refs accepted, whole-bake
  ACCEPT (fidelity 17.02 -> 17.27, tone 0/0, hue 0.034, coverage
  0.829), renders indistinguishable from the approved standard — G2's
  rebake prediction reproduced end-to-end.
- **CPU fleet rebakes** (persisted refs): starship ACCEPT
  (19.77 -> 19.63), chair ACCEPT (24.65 -> 24.62), portrait ACCEPT
  (12.60 -> 12.86 — MORE photo-faithful than the pre-G2 tree's 14.12;
  render A/B shows reduced pale chips).
- **G1 cutter matrix reproduced on the final tree**: car_final slab
  removed (plate 0.9713 / lamina 0.6198 / overhang 1.302 / cut
  38.27%), five controls bit-identical.
- **Break attempts**: (a) slab cutter vs pedestal class — a real
  pedestal-subject t23d draw (chess knight on a display base) ships
  untouched with 11x margin on the plate condition (Hunyuan draws
  bases as solid rounded discs; plate 0.026 vs 0.30); synthetic
  probes measure the ONE collision class — a BY-DESIGN thin (<5% H
  top skin), wide (overhang > 1.05), razor-flat display disc is
  geometrically indistinguishable from the defect signature and WOULD
  be amputated (KB entry; metadata `ground_slab` records any cut).
  Threshold edges behave exactly as documented (lamina 4.6%/5.4% H,
  overhang 1.02/1.08). (b) absolute sovereignty vs the historical
  ramp on the owl pins — NO surrendered-surface class: absolute is
  more photo-faithful (17.93 vs 18.23 forced-ramp, pre-G2 pin 18.73),
  ledger p95 0.4142 vs pin 0.4289, informational seam improves,
  coverage unchanged; render deltas are detail-scale (p99 <= 20/255)
  at the relief edges where ownership switched. (c) determinism —
  two separate-process back-angle generations on the fresh draw-A
  bundle: payload md5 a5443e62… twice, every gate metric identical,
  processed PNG md5 80696660… twice.

### Fixed (texture bake — photo sovereignty through the full composition chain; the fresh-draw refusal)

The last blocker for references shipping on hard subjects: on a
LOW-COVERAGE subject at an ESTIMATED pose, adding hue- and
material-correct generated references regressed photo fidelity AT THE
TRUE SOURCE POSE versus the no-references baseline (fresh-draw car:
32.31 -> 36.51 dE at (40, 15), +2.0 allowed — the whole-bake gate
rightly refused and the mottled baseline shipped; the pinned v7
candidate sat at the slack line draw-to-draw). Photo sovereignty
doctrine says the photo's own surface must be essentially untouched by
adding references, so the regression was a defect of the composition
math, not of the references. Stage-decomposition harness (selective
neutralization, per-stage fidelity at the true pose on witnessed
texels, atlas-space ownership channels): /tmp/gfix2/report.md.

Measured attribution of the +3.18 dE regression (fresh car, 1024
diagnostics; channels on photo-witnessed atlas texels,
|candidate - baseline| LAB dE):

- SUB-FLOOR REPLACEMENT: `protect_observed_texels`' linear ramp below
  the 0.02 floor handed the photo's weakest witnessed band — 25% of
  its witnessed atlas on this subject (grazing facing, concavity
  demotions) — to references at up to 30x the photo's weight: 39.1 dE
  mean over 20.3k texels, the largest single channel (neutralizing it
  alone recovered 2.17 of the 3.18).
- POISSON TONE DIFFUSION: texels witnessed ONLY by the photo (every
  generated weight zero) still moved 14.1 dE mean — the screened
  Poisson compositor's proportional screening leaves weakly-witnessed
  photo texels on a 20-60 texel equalization decay length, and the
  tone step at every photo|reference ownership boundary redistributed
  INTO the photo (skipping the solve collapsed the channel to 1.7 dE;
  the solve itself is load-bearing elsewhere and stays).
- TONE LANES PULLING THE WRONG WAY: the symmetric witness gate of
  `equalize_projection_tone` VETOED a field that improved photo
  agreement 0.208 -> 0.198 because generated-mutual agreement paid for
  it (tone_off recovered 0.33); `delight_projections`' aggregate gate
  let a side reference relight toward two other generated views'
  invented lighting with zero real overlap in its gate mass
  (delight_off recovered 0.41).
- Registration warps (width-profile + dense flow) measured -2.89 when
  disabled, but the probe showed them load-bearing for alignment
  (back ref IoU 0.754 -> 0.945): their harm routed entirely through
  the sovereignty channels above, so they stay unchanged.
- Conflict resolution's photo-claim kills measured net-GOOD
  (grazing-band fidelity 34.6 vs 41.4 with kills disabled): a head-on
  reference outranking the photo's stretched rim smear under strong
  disagreement is correct and is kept.

Landed (all in `texturing.py`):

- `protect_observed_texels(mode="absolute")` (bake call site):
  generated weight is zeroed wherever ANY real view holds POSITIVE
  weight — full single-view sovereignty, the doctrine's own wording
  ("a real photo's stretched rim content outranks plausible
  synthesis"); the historical ramp remains the default for direct
  callers. Handoff smoothing falls to the blend feather and the
  gradient-domain composite (measured: v7 ledger p95 0.398 -> 0.389,
  no regression).
- PHOTO-ANCHOR PIN at the compositing call site (generated-references
  bakes only): real-witnessed texels enter the screened-Poisson solve
  at full anchor confidence with the source boost over the photo's
  whole witnessed set, bounding synthetic tone influence to the blend
  feather's own handoff scale (~9 texels at 1024). Bakes with REAL
  reference photos keep proportional screening (cross-view tone
  equalization is the solve's purpose there).
- WITNESS-RANKED gates in `equalize_projection_tone` and (generated
  views only) `delight_projections`: a correction that measurably
  worsens real-photo agreement never ships; one that measurably
  improves it ships even when generated-mutual agreement pays for it;
  real-absent falls back to the previous rule. Photos are evidence and
  define tone; mutual consistency among synthesized views is
  subordinate.
- PHOTO AUTHORITY ON THE SUB-FLOOR BAND in stage 2 of
  `equalize_projection_tone`: wherever a real view holds any positive
  weight, the consensus is the real-witness reading with full
  authority (no self-inclusion, no other generated view), and the
  real-witnessed band enters the field's evidence even below the pair
  fit floor — grazing photo samples are smeared in detail but valid in
  REGIONAL tone, which is all the stage consumes.

Validation (production gate `evaluate_generated_bake`, 2048 CPU
rebakes from persisted accepted references, /tmp/gfix2):

- Fresh-draw car (/tmp/fix3/car_final, pose silhouette_rescue
  (40, 15)): fidelity 32.31 -> 36.51 BEFORE (refused, +4.20 over
  budget; hue axis co-fired 1.46) => 32.31 -> 33.02 AFTER — inside the
  +2.0 slack, verdict ACCEPT (tone 0.000/0.003, hue 0.97, brightness
  improved). The references now ship on the hard subject.
- Pinned v7 pair: stays ACCEPT; fidelity 34.06 -> 32.64 (was 33.86
  pre-fix — the candidate is now more photo-faithful than either),
  tone 0/0, hue 0.23, handoff ledger p50/p95 0.179/0.389 vs pinned
  0.182/0.398 (F2's gains preserved).
- Owl control: stays ACCEPT, equal-or-better on every axis — fidelity
  17.02 -> 17.27 (pre-fix candidate measured 18.34), tone 0/0, hue
  0.03, informational seam 0.0251 -> 0.0216, renders indistinguishable
  from the shipped standard.
- Single-photo lanes structurally untouched (every change is gated on
  generated views being present): parity canaries P1/P2/P3 green —
  refs-off 2048 rebake bit-identical to the pin, two 1024 rebakes
  md5-identical (c84f2e49…), face 2048 md5 e32ba995… unchanged across
  two rebakes.
- New regression tests (tests/test_texturing.py): absolute-mode
  protection semantics; witness-ranked gate ships photo-conforming
  corrections and still fails closed against real-class worsening;
  sub-floor witnessed band reconciled toward the photo; END-TO-END
  synthetic sphere bake where a deliberately tone-offset generated
  reference must contribute (essentially) zero delta on
  photo-witnessed texels (p99 <= 2/255). Full suite green.

### Fixed (Hunyuan3D shape — ground-slab removal for the documented vehicle ground-plane defect)

A fresh car t23d draw (/tmp/fix3/car_final, seed 2025) shipped with a
large flat GROUND SLAB fused under the mesh: a near-horizontal plate
extending beyond the body footprint, the upstream Hunyuan3D-2.1 weakness
documented in issue #48 ("Prevent ground floor from shape generation";
the 2026-07-11 mesh audit /tmp/mesh3/report.md cites it). Incidence is
draw-dependent (the earlier sportscar_v7 from the SAME subject hint and
seed has no slab); the slab also corrupts texture registration
(silhouettes stop matching the photo).

Measured slab signature (car_final vs 5 controls: v7 car, owl on its
legitimate carved base, chair legs, starship fins, face bust — full
matrix in /tmp/gfix1/report.md). Three independent conditions, ANDed:

- **plate**: bottom-anchored near-planar down-facing skin covering
  >= 30% of the whole-mesh footprint hull (slab 0.97; owl base 0.49;
  every other control <= 0.04);
- **lamina**: an exposed up-facing top skin within 5% of mesh height
  above the plate, laterally inside the plate hull — the plate is a
  thin SHEET, not the underside of a solid base (slab 0.62 at 2.0%-H
  thickness; owl 0.00 — its carved base top sits at 10-11% H; face
  0.06; others 0.00);
- **overhang**: plate hull extends beyond the convex footprint of
  everything above it (slab 1.30; strongest control 0.65 — legitimate
  bases/feet always sit INSIDE the subject's footprint).

Landed (`hunyuan3d_runtime._hunyuan_cut_ground_slab`, called from
`_hunyuan_postprocess_mesh` after floater removal and BEFORE decimation
so the freed face budget returns to the subject): planar face cut just
above the slab's top skin (clearance max(0.5% H, 25% of slab
thickness)), orphan sweep with the existing 0.5%-of-total floater rule,
open rim tolerated (the bake projects onto visible surface, decimation
preserves boundaries, `topology.is_watertight` is recorded not gated).
Fail-closed budget: if the cut would remove > 50% of the surface the
cutter refuses, warns, and the run ships `quality_verdict=degraded`
(the subject must remain the majority of its own mesh; the real slab
cut removes 38.3%). Measurements are recorded in the postprocess
`applied` list (`ground_slab_removed:120000->111327@plate=0.9713,...`)
and in metadata `ground_slab`; a cleanly cut slab demotes nothing.

Validation matrix (production cutter on persisted bundles): car_final
slab removed (before/after renders in /tmp/gfix1/), post-cut 1 body,
euler -131 vs -132; sportscar_v7, owl, chair, starship, face all
bit-identical (detector returns None). Unit fixtures encode the
signature (slab-on-wheeled-box cut; thick-base spared; leg contacts
spared; over-budget plate refused). Suite: 310 passed.

Prompt-side mitigation probe (mlx-gen flux.2-klein-4b-8bit, 8 images,
/tmp/gfix1/probe_results.json): a trailing "subject isolated, no floor,
no ground shadow" suffix did NOT reduce floor shadows (shadow depth
175.9 -> 186.6 at seed 2025; the model keeps drawing contact shadows on
grounded subjects), and a mid-prompt "floating in mid-air, no floor and
no ground shadow" clause DID lift the car but replaced the contact
shadow with a large detached shadow patch below the subject (shadow
area 0.075 -> 0.136) and changed the wheel/body composition — worse
conditioning, not better. NO prompt change landed; the mesh-side cutter
is the fix. (Pending validator confirmation at the shape stage, where
image-shadow -> slab causality can be tested with shape inference.)

Adversarial integrator pass over the whole uncommitted program (per-view
two-key gates + retry ladder, rebake/pipeline parity + identity-repair
background fix, whole-bake tone-damage gate, consensus tone
reconciliation). Full report with artifacts: /tmp/fix3/report.md.

- **Test suite**: 300 passed + 3 xfailed at entry; 304 passed +
  3 skipped (opt-in parity canaries) + 3 xfailed after this pass's
  additions.
- **Pinned car pair** (sportscar_v7 + twokey views, production
  semantics, 2048): gate ACCEPT reproduced — tone dark/bright
  0.000/0.000, fidelity 34.06 -> 33.86 at (17.5, 8.0)
  origin=baseline_stats, handoff ledger 10196 texels p50 0.182 /
  p95 0.398 (F1 margins and F2 ledger numbers within noise).
- **Fleet rebakes from persisted references** (CPU): starship ACCEPT
  (fidelity 19.77 -> 19.58), chair ACCEPT (24.65 -> 24.62, seam info
  improved), portrait ACCEPT (12.60 -> 14.12, inside +2.0 slack).
  Texel drift vs stored outputs is attributed: F2's intended
  relighting (starship/portrait) and the pose-guard correcting the
  chair's old az -27.5 estimator spike to the honest (0,0) — fresh
  chair measures MORE photo-faithful than the stored bake (24.62 vs
  24.80) with the stored bake's black armrest holes gone.
- **Owl i23d end-to-end** (fresh generation): healthy, 4/4 references
  accepted, whole-bake ACCEPT (fidelity 17.02 -> 18.34 within slack,
  tone 0/0, hue 0.068/1.0), coverage 0.83, renders indistinguishable
  from the approved owl_redo_refs standard. No regression.
- **Car t23d end-to-end** (user's verbatim command): exit 0 with
  `VECLIB_MAXIMUM_THREADS=1` (two prior attempts died in the HOST's
  threaded Accelerate GEMM — pre-existing .ips signatures; KB entry),
  closed-roof input, 4/4 references accepted on attempt 1 (seed 2025).
  The run exposed the pose-guard dead zone fixed above (shipped
  baseline at (0,0), coverage 0.057, "healthy"); post-fix rebake:
  pose silhouette_rescue (40, 15), coverage 0.170, sanity floors
  pass, product visibly photo-true from the source pose. The
  whole-bake gate refused THIS draw's candidate on an independent
  fidelity regression (37.32 vs 32.31+2.0) — references still ship
  only when they earn it; the pinned v7 candidate keeps ACCEPTing.
- **Retry-ladder determinism**: two separate-process back-angle
  generations at base seed 2025 — raw payload md5-identical, all gate
  metrics identical, same accepted seed. (Both under
  VECLIB_MAXIMUM_THREADS=1; the threaded-GEMM host bug is the prime
  suspect for the previously recorded same-seed drift.)
- **Break attempts**: F2's consensus pass held all worst cases
  (sub-floor overlap -> structural no-op; content confound -> full
  revert; source-starved component -> relative-only reconciliation;
  evidence-faded deep-exclusive bit-identity) — now pinned as tests.
  F1's tone axis had one measured hole (chroma-only damage) — closed
  (see the hue-damage entry above).
- **Parity canaries landed** (the parity audit's P1/P2 spec):
  `artifacts/validation/parity/sportscar_v7/` +
  `scripts/parity_canary.py` + opt-in `tests/test_parity_canary.py`
  (ABSTRACT3D_PARITY_CANARY=1; CPU-only). First run: 3/3 pass — P1
  refs-off 2048 rebake bit-identical to the pin (1.000 / max delta 0),
  P2 two 1024 rebakes md5-identical AND equal to the July parity
  audit's recorded md5 (c84f2e49…: the single-photo lane is
  bit-stable across the whole fix program), P3 face 2048 md5-stable
  (e32ba995…, the fix-2 canary value).

### Fixed (source pose — the guard's reject-but-don't-rescue dead zone; sanity floors on the refused-candidate branch)

End-to-end integrator run (fresh t23d car draw, seed 2025): the pose
guard's shape-decisive override VETOED the NCC commit on decisive
double-keyed evidence (commit riou 0.773 vs basin best 0.896 @ (40,15),
commit aspect err 0.159), then the rescue lane's own second key — the
DECLARED pose's aspect error, 0.113 < 0.15 — refused to move, so the
bake ran at declared (0,0): coverage 0.0574, all non-front surface
mottled fill. Measured A/B at the basin pose (40,15): coverage 0.1877
(3.3x), photo fidelity 32.71 -> 21.59 dE, single-view sanity floors
pass (at (0,0) two of three floors fail). Fix: an override veto IS
decisive shape evidence, so it now counts as the rescue's second key
(`estimate_pose_with_silhouette_guard`; the double-key doctrine is
preserved — the override itself required the commit-side aspect key).
Fleet control (7 bundles): only the fresh car moves ((0,0) ->
silhouette_rescue (40,15)); v7 car/starship/portrait keep their NCC
commits, owl/face keep declared, and the chair's override veto still
correctly refuses to move (basin gap 0.064 < 0.10 — its (0,0) is
honest). Companion fix: when the whole-bake gate refuses a candidate,
the shipped BASELINE now runs the same `evaluate_single_view_bake`
sanity floors as the no-references branch (both `hunyuan3d_runtime`
and `rebake_bundle`) — the measured incident shipped a coverage-0.058
baseline with verdict "healthy" while the floors existed one branch
below (`quality_verdict=degraded` + loud warning now).

### Added (texture acceptance — composition hue-damage axis closes the measured chroma hole)

Adversarial integrator probe (2026-07, /tmp/fix3): the whole-bake gate's
tone axis is L-based, so CHROMA-only damage — a car back reference
hue-rotated 30 deg at constant L (red -> orange) — passed every voting
axis (fidelity IMPROVED 34.12 -> 33.75 at the estimated pose, tone
damage 0.000/0.000, brightness clean) and the visibly orange-backed bake
would have shipped. In production the per-view two-key gates refuse that
reference one layer earlier (consensus 31.1 > 16, cloud evidence 28.5 >
11), but `rebake_bundle` accepts CALLER-provided references ungated —
the whole-bake gate is the only defense on that path. Landed
(`bake_acceptance._chroma_field_damage`): coherent low-band hue-rotation
axis, same construction as the tone axis (masked smoothing sigma 24 @
512) on the LAB ab fields, angle between candidate/baseline ab vectors
integrated beyond a 10-deg floor (the pipeline's own sanctioned ab
drift: `match_tone_lab` clamps ab at 10, ~11 deg at anchor chroma) on
mutually saturated surface (smoothed chroma >= 15; the saturated-area
weight keeps gray subjects structurally quiet — starship measures 0.000
with sat_frac 0.00), budget 1.0. Raw ab-displacement was measured
NON-separating first (legit content replacement displaces ab 4.1-5.4 vs
damage 5.2 — inverted); the ANGLE separates. Calibration on the 12
fix-program pairs + 3 synthesized chroma pairs: every labeled accept
<= 0.460 (car 0.202, owl 0.109, portrait 0.133, chairs 0.31-0.46 —
2.2x under budget), hue-rotated back @30deg 2.310, @20deg 1.528 (1.5x
over budget), both rotated sides @30deg 6.609; all 11 prior verdicts
unchanged. Known limit (tracked, mirror of the tone axis's dark-back
limit): a subject whose true unseen side is a different hue family
(two-tone vehicles) would ship the baseline — safe direction, missed
rescue; no fixture of that class exists yet. New tests:
`test_constant_l_hue_rotation_is_rejected`,
`test_scattered_hue_detail_is_not_rejected`.

### Fixed (texture bake — view-boundary tone consistency for generated references)

The car candidate (4 accepted references, visually correct content)
shipped an objectively poor VIEW-BOUNDARY composite: handoff ledger
boundary step p50 0.214 / p95 0.446, with every dominant boundary pair
93-99% LUMINANCE — tone levels, not content. Root cause, measured: the
delight lane's gauge never reaches the photo on low-coverage subjects
(at its 0.05 pair floor the source participates in ZERO pairs >= 400
texels — front|top_rear 0, front|side_right 55), and its joint SH solve
sacrifices small rim-dominated overlaps to the dominant pair
(back|top_rear, 7.7k texels at log-ratio 1.1 — one generated view
reading 3x brighter than another on shared surface), so the sides'
corrections WORSENED their own overlaps (0.250 -> 0.279, 0.231 -> 0.319)
and the fail-closed revert correctly refused them: the composite shipped
half-relit. Landed (`texturing.equalize_projection_tone`, called from
`bake_projection_texture` after delight; stats key `tone_consensus`):

- **Stage 1 — consensus tone levels**: one log-luminance gain per
  generated view, solved jointly over pairwise overlap medians (panorama
  gain compensation) at the photo-evidence floor 0.02
  (`protect_observed_texels`' own floor; reconnects the gauge chain —
  front|top_rear carries 1.8k texels there). Real photo views are
  gauge-FIXED (photos define the level; synthesis conforms); components
  with no real member keep their weighted-mean level (the common mode is
  unobservable from ratios). Chained-gauge cap halving as in delight.
- **Stage 2 — local consensus field**: the residual disagreement is
  spatially varying (log-ratio IQRs 0.68-1.55 — regional shading each
  generator invented), which is what the SH fit had to revert on the
  sides. Per generated view, the deviation from the witness consensus —
  weighted with the DOWNSTREAM composition semantics, i.e. the photo's
  reading wherever it holds a protected claim — is voxel-ball smoothed
  at 3% of the subject diagonal (radius sweep in the docstring), capped
  at the chained amplitude budget, faded by evidence density,
  luminance-only.
- **Witness-ranked fail-closed gate** for both stages: a correction
  ships only if it measurably improves agreement with one witness class
  (real photos / other generated views) WITHOUT worsening the other — a
  single mass-weighted aggregate structurally silences the photo class
  (its rim overlap carries ~1% of a big view's pair mass; measured:
  top_rear's field improved the photo pair and reverted on the
  aggregate).
- **Handoff ledger pair attribution** (`blend_projections`): the
  `handoff_seams` ledger now names the owner pairs with per-pair
  p50/p95, luminance share, and co-witnessed fraction (pixel-inert;
  this is the instrument that located the defect).

Measured on the car candidate (2048, before -> after): handoff p50
0.214 -> 0.190, p95 0.446 -> 0.395; per-pair p95: back|top_rear
0.508 -> 0.326, side_right|top_rear 0.405 -> 0.338, side_left|top_rear
0.445 -> 0.399, back|side_right 0.379 -> 0.195. Photo fidelity at the
diagnostic pose improved 35.29 -> 35.02 (baseline 32.83). Render-space
long-edge ratio at the worst view (az0_el50) 0.047 -> 0.044 (baseline
0.026): the residual is measured to be mostly NOT view-boundary tone —
long-edge attribution puts 1786/2934 px on observed|fill class borders
and 55 px on ownership boundaries, and the shipped composite's ownership
steps are already small (p50 0.011 / p95 0.084: the screened-Poisson
composite bridges them); what remains is reference content contours the
featureless baseline fill lacks (the mis-ranking the acceptance-gate
entry below closes) plus the source's protected stretched rim band
(photo sovereignty; not repaintable by tone lanes). Controls: owl ledger
p50 0.177 -> 0.163 / p95 0.459 -> 0.429 with fidelity 18.80 -> 18.73 and
worst-view seam 0.027 -> 0.024 (all views far under its recorded 0.045
acceptance ceiling); face single-photo bake md5-identical across two
rebakes before and after (e32ba995d43da1d68fbc06fb3f0a44a8) — the pass
is structurally inert without generated views. Full suite green
(300 passed). New tests:
`test_equalize_projection_tone_levels_chained_views`,
`test_equalize_projection_tone_pins_real_views`,
`test_equalize_projection_tone_local_field_regional_deviation`,
`test_equalize_projection_tone_fails_closed_on_content_confound`,
`test_blend_projections_handoff_ledger_attributes_pairs`.

### Fixed (texture acceptance — whole-bake gate: composition tone damage replaces the long-edge seam vote)

Closes the "seam budget's structural bias" item the parity audit left
open. An 11-fixture calibration program (production-semantics rebakes of
car@2048/@1024, owl, portrait, starship, chair — plus synthesized
true-regression chairs: +/-25 L mis-toned back under both compositors,
8% content-shifted back, and a +12 L control inside the sanctioned
range) measured the old long-edge seam axis mis-ranking BOTH sides on
the current stack:

- It refused the labeled-GOOD car candidate (worst-view long-edge ratio
  0.0592 -> 0.0886 against the +0.02 budget) whose fidelity and
  brightness both improved at the true pose (34.06 -> 33.82 dE) — the
  counted edges are the roof glass frame and real panel contours that
  references legitimately add to previously blank fill (overlay proof in
  the fix report).
- It passed ALL FOUR synthesized true regressions (mis-toned +25 L:
  seam delta +0.013 gradient / +0.016 legacy; content-shifted: -0.005;
  -25 L: -0.006): gradient-domain compositing smooths real tone damage
  below any step threshold, so the sharp-step class the axis was
  calibrated on (v2 chair, 0.102 -> 0.138) no longer exists in the form
  it detects. The texture-space `handoff_seams` ledger cannot replace it
  either: it is blind to reference-to-fill frontiers (the chair
  candidate records boundary_texels == 0 while carrying an obvious back
  handoff — its front/back observed sets never touch in UV).

Landed in `bake_acceptance.evaluate_generated_bake`:

- **Composition tone damage axis (votes)**: per turnaround view, smooth
  each side's L over its own foreground (normalized convolution, sigma
  24 @ 512 — texture/specular detail that references legitimately change
  decorrelates below this scale; measured sweep sigma 8/16/24 gives
  1.5x/2.4x/4.7x accept-vs-refuse separation), take the signed delta on
  the co-foreground interior, and integrate the excess beyond a 25 L
  floor per direction. The floor is the pipeline's own sanctioned
  tone-adjustment budget (generation tone-match clamps at 15 L,
  in-bake harmonization/leveling within ~10 L) — correct content cannot
  legitimately move further. Directional budgets because baseline fill
  is dark-biased by construction (references legitimately BRIGHTEN it:
  measured up to 0.228 on labeled accepts; legitimate darkening beyond
  the floor is identically 0.000 on all 7 accepts): darken budget 0.03
  (3.3x under the weakest true regression, 0.100), brighten budget 0.7
  (3.1x over the worst accept, 4.0x under the weakest mis-tone, 2.80).
- **Long-edge seam ratio and handoff ledger stay in `metrics`,
  vote-less** — fleet drift stays observable, and the measured
  mis-ranking is documented in the module docstring.
- **Fidelity pose from the bake's own stats**: the gate now takes
  `baseline_stats`/`candidate_stats` and resolves the fidelity pose
  itself (explicit `source_pose` still wins; provenance recorded in
  metrics). Both call sites (`hunyuan3d_runtime`, `rebake_bundle`) pass
  stats instead of hand-extracting the pose. The measured (0,0) trap —
  a diagnostic bake without `projection_model="orthographic"` never
  estimates a pose and silently gates at (0,0), charging ~9 dE of pose
  error to both sides — is now visible in the verdict (`origin:
  "default"`).

Calibration (fixture -> verdict -> margin, /tmp/fix1/report.md for the
full program): car@2048 ACCEPT (dark 0.000/0.03, bright 0.006/0.7),
car@1024 ACCEPT (0.000, 0.005), owl ACCEPT (0.000, 0.000), portrait
ACCEPT (0.000, 0.224), starship ACCEPT (0.000, 0.000), chair-clean
ACCEPT (0.000, 0.0003), chair +12 L ACCEPT (0.000, 0.228); chair +25 L
REFUSE at 4.0x budget (both compositors), chair -25 L REFUSE at 11x,
chair content-shift REFUSE at 3.3x. The previously-refused clean car
candidate now ships; every synthesized regression the old gate passed
now refuses. Note for the record: a pure-translation "mis-registration"
(shifting the reference frame 8%) is a measured NULL — the canonical
alpha-bbox recenter undoes it exactly (candidate texture md5-identical
to the clean rebuild); the real class is content displaced INSIDE the
silhouette, which is what the shipped fixture synthesizes.

### Fixed (identity repair — miniature-photo stamp on gray-background subjects)

User-reported: a small duplication of the texture inside the texture on
the car's hood. Root cause: `_register_photo_to_render` (feature-fringe
repair) classified photo foreground by distance from WHITE (calibrated
on the white-background face proof); the car photo's neutral-GRAY studio
backdrop made the whole frame "foreground", the bbox correspondence
degenerated, the NCC search pegged at its scale boundary (recorded:
registration [1.125, 0, 0.1025] — the exact signature the code's own
comment warns about), and the repair stamped a miniature of the entire
photo — background included — onto the hood. Fixes: (a) background
estimated from the frame's border median (reduces to the white rule on
white backgrounds, correct on gray); (b) fail-closed veto — the repair
refuses to run when the correspondence NCC < 0.55 or the scale sits at
the search bound (the certified face alignment scores ~0.9; below the
floor the "evidence" is not the surface it claims). Measured: the
mini-photo is gone from the rebuilt candidate; the stage remains active
and beneficial on the face proof. The owl/face never hit this because
their backgrounds are white — the failure class was gray-backdrop
product photos, which is most studio photography.

### Fixed (texture acceptance — parity + calibration audits)

Two adversarial audits closed the loop on why car textures oscillated
between runs. Parity audit: NEITHER prior car bake had shipped a single
generated texel — the whole-bake gate refused both candidates and both
paths shipped single-photo baselines (md5-proven), differing only by
resolution (2048 vs 1024: the 1024 fill-detail calibration hits its
hard cap and mottles ~88%-fill subjects 5.8x darker). Calibration
audit: both per-view strict thresholds sat AT the correct-class median
(per-seed strict pass 3-8%), so any acceptance was a coin flip carried
by generator seed stochasticity (worst_part std 5.29 dE) plus k-means
gate noise (3.15 dE; a recorded 24.12 REJECT of a clean red back
re-scored at 15.2 mean under resampling). Landed, each with its
measurement:

- **Rebake/pipeline parity** (`rebake_bundle`): threads the bundle's
  recorded seed and caption (was: hardcoded 11 and None — guaranteed
  different candidates per path); persists refused candidates'
  generated views (was: refusal destroyed the evidence); both paths now
  gate photo-fidelity at the baseline's estimated pose, not (0,0)
  (pose error charged ~9 dE to both sides and inflated the candidate
  regression +0.88 -> +4.03 on the az-17.5 car; owl-class subjects at
  declared (0,0) unaffected).
- **5-restart consensus** (`part_material_fidelity`): median over 5
  fixed k-means restarts of the k-ensemble (gate RNG 3.15 -> 1.13 dE at
  ~2s CPU against a ~200s generation).
- **Two-key anchor-class strict line** (`generate_reference_views` +
  new `gate_witnessed_consistency` + `cloud_evidence_delta`): measured
  on 71 labeled candidates, NO global palette threshold separates
  correct from wrong on vivid gloss (correct 3.9-24.4, wrong 11.1-40 —
  a wrong red-painted glass canopy scored 11.97, clean red backs
  23.99/24.12). The residual wrongness is POSITIONAL: the gate projects
  the source photo onto the mesh (witnessed vertices, no fill), renders
  expected colors from the candidate's angle, and vetoes on
  chroma_flip/bright_flip/tile-median (12/14 wrong tops, 0/10 false
  fires). Witnessed angles accept at consensus <= 22 + veto;
  witness-starved angles (a back witnesses ~600 px from a front photo)
  at consensus <= 16 + cloud-evidence <= 11. flat_delta leaves anchor
  strict entirely (zero discrimination; it alone blocked 3
  material-passing correct backs). Fleet controls unchanged (13/13).
- **Best-of-6 spaced-seed ladder** for anchor angles (seeds
  base+1000*attempt, steps alternating 8/12, early stop): measured 97%
  angle acceptance / 94% rerun agreement vs coin-flip; floor-band still
  never ships (the glass-painted-red failure is seed-STABLE — 3/8
  seeds — so seed consensus would ratify it; the positional veto is the
  guard, not agreement).

End-to-end validation (car rebake, 5 angles): per-view acceptance is
now deterministic and correct — back accepted on attempt 6 (13.35 +
cloud 8.45; its 6-seed trail shows the ladder working), both sides and
top_rear on attempt 1, and the top correctly refused on ALL 6 attempts
by the witness veto (canopy painted body-red every seed: the exact
class that must never ship). The whole-bake gate still refuses the
composite (fidelity +3.2 dE, seams 0.026 -> 0.055): the accepted
references paint white rim blotches onto the hood near the source's
coverage edge. That projection-stage defect (and the seam budget's
structural bias against low-coverage subjects) is the remaining open
item on this line.

### Fixed (mesh quality — three-audit program on the sports-car geometry)

Three adversarial audits (comparative mesh forensics; shape-path code
audit with controlled decode/generation A/Bs; external research +
symmetry/repair prototyping) investigated why the owl mesh is excellent
while car meshes came out lumpy with wrong topology. The decisive
measurement ACQUITS the pipeline's surfacing: decoding identical car
latents with our `_AdaptiveVolumeDecoder` and upstream's dense
`VanillaVolumeDecoder` yields ZERO sign disagreements across 57M grid
vertices and bit-identical meshes (ours ~10x faster) — every observed
defect already exists in the DiT/VAE field. The defects toggle with the
INPUT IMAGE: the t2i stage drew open-cockpit convertibles (12-21%
deep-shadow pixels vs owl 2.8%), and the shape model invents what it
cannot see — melted cabin (50x the interior angle-defect energy of a
closed coupe at identical seed/settings), all four wheels floating
1.4-1.9 grid cells off the arches (the "5 bodies"), ~200 micro-tunnels
on unseen surfaces. Much of the headline genus is REAL: a spoked wheel
alone is genus ~10, so genus is not gated. Changes landed:

- **t23d closed-body input policy** (`_default_text_to_image_prompt`):
  for subjects whose t2i priors default to open thin-shell forms
  (vehicle/boat noun list), the source-image prompt now biases to
  "closed body, no open top" UNLESS the user's text asks for an open
  form (convertible/roadster/cockpit/...). Measured on the car A/B:
  cockpit junk eliminated, mirror deviation -17%, dihedral RMS -11%.
- **Default octree resolution 384 -> 512**: measured strictly better at
  equal adaptive-decode time (car dihedral roughness 8.0% -> 6.7%,
  topology converged; 256 is catastrophic: euler -208). Guidance stays
  5.0 (7.5 fused wheels into the body), steps stay 50.
- **Floater filter matched to upstream** (0.5% of total faces, was 2%
  of largest component — measured 3.2x more aggressive; a detached side
  mirror sits exactly in the amputation range).
- **Mesh-aware quality verdict**: disconnected bodies after floater
  removal now ship `quality_verdict: degraded` with the reason (the v5
  car shipped "healthy" with four floating wheels — the verdict never
  looked at the mesh). Raw pre-cleanup topology is recorded as
  `topology_raw` so surfacing regressions stay visible.
- **Adaptive-decoder band hardening**: the refinement band now scales
  with the field's measured near-surface logit magnitude
  (1.5x the sign-change-shell median) instead of a fixed 0.95 — the
  shipped checkpoint cleared the fixed band by 0.003 logit units, and a
  saturating field would silently weld thin gaps shut through the
  interpolated fill (proven synthetically: 21k sign flips).
- **Docstring correction**: upstream `HierarchicalVolumeDecoding` fails
  from an int-truncation bug (cell size cast to 0), not from
  thin-geometry/MPS scatter as previously claimed here.

End-to-end validation. Owl regression at the new defaults: unchanged
excellence (genus 0, 1 body, healthy verdict, clean bake). Car t23d
rerun: the first wording of the closed-body policy (trailing "closed
body, no open top") measurably FAILED — the t2i model still drew a
convertible; the landed subject-position clause ("a hardtop with a
fully closed solid roof...", placed directly after the user text) drew
closed hardtops on every tested seed — prompt placement matters as
much as content (same finding as the texture color anchor). With the
closed-roof input the shipped car is 1 connected body (wheels attached;
v5: 4 floating), the cabin-junk failure surface is gone, and — the
compounding win — 3 of 4 texture references now pass STRICT material
gates (9.56-11.21 vs 0 accepted at v5's strict line), because the
references no longer show a dark open cockpit. Remaining top-view gap:
its candidates still reject at 17.84 (fill covers the roof), tracked
with the anchor-class recalibration.

Tracked, not landed: multi-view geometry conditioning for hard classes
(the 2mv checkpoint path is already wired for i23d reference views;
synthesizing views BEFORE the shape stage in t23d is the follow-up both
audits rank as the real fix for open/complex forms), an opt-in
voxel-SDF repair stage (measured: genus 67 -> 9, 5 bodies -> 1 at 0.57%
chamfer), and the symmetry detector as a diagnostic gate (bilateral
subjects measure support >= 0.946, organic ones <= 0.804; post-hoc
vertex symmetrization REGRESSED dihedral roughness on every subject and
is rejected).

### Fixed (generated references — gloss-class audit follow-through)

The sports-car incident audit (seed-exact reconstruction of all 12
rejected attempts + an 11-candidate prompt/model ladder, every artifact
persisted) proved the rejections were 100% generation infidelity — the
i2i editor echoed the gray clay guide, collapsing foreground chroma to
2-28% of the source — and that the generator IS capable once the prompt
carries the measured color. Changes, each with the measurement that
forced it:

- **Chroma-collapse guard now uses chroma DISPERSION** (std of
  foreground ab magnitudes, `part_material_fidelity`): the
  chroma-of-mean form left a fully-gray PORTRAIT strict-passing the
  entire gate battery (skin's mean chroma sits below the arming floor)
  and cancels complementary hues. Measured separation: collapsed
  candidates <= 0.18 of the source's dispersion (gray portrait 0.06),
  every accepted fleet view >= 0.53; guard arms at source std >= 5,
  fails below ratio 0.35. A collapse verdict now records
  `worst_part_delta_e` 40.0 instead of None (None scored as ZERO palette
  penalty in candidate ranking, letting a collapsed candidate outrank
  floor-band ones in the record).
- **Part gate scores the minimum over k in {3,4,5}**: single-k k-means
  adds 14-24 points of pure correspondence noise near part boundaries —
  a PERFECT copy plus one realistic windshield reflection scored 25.3
  (the reflection steals a centroid); the ensemble scores it 1.56. True
  palette flips stay far under every k (hue+30: 30.95); all 9
  previously-accepted fleet views stay accepted (max 9.79).
- **Smooth-finish prompt variant when the color anchor fires**
  (`_view_prompt`): the relief enumeration ("carving depth, grooves,
  grain, cracks, fibers") plus "Do not smooth, polish, glaze" plus "no
  gloss" rendered craquelure on a crack-free car in 8 of 8 candidates —
  the audit refuted the old "self-normalizing relief vocabulary" claim
  for this class. With the anchor + smooth wording the same model, seed
  and steps produced clean red paint at material strict-pass 10.87
  (validated against the audit's persisted candidate K). The texture
  escalation clause is likewise skipped for anchor subjects (it
  re-injects the craquelure bias) and no longer fires on material-only
  failures (the mechanism change for those is the steps/seed re-roll).
- **flat_delta strict relaxes to 0.18 for anchor-marked subjects**
  (gate call in `generate_reference_views`): strict 0.12 punishes the
  smoothness that is CORRECT for vivid gloss (clean-paint candidate
  0.138 rejected while 7 of 8 craquelure candidates passed — the
  source's band energy is specular structure and panel lines, not
  micro-relief). Floor 0.20 and relief_ratio strict unchanged.
- **Triage metadata**: every failed attempt now records a
  `failure_family` string (silhouette / chroma_collapse / palette_flip /
  texture / speculars — the gray-car diagnosis needed a full
  regeneration to learn what one string could have said), material rows
  persist the per-k ensemble and dispersion numbers, and the
  rejected-image budget rises 12 -> 15 (5 angles x 3 attempts; 12
  silently dropped a fifth angle's rejects).

- **Pose guard: shape-decisive override of an accepted NCC commit**
  (`estimate_pose_with_silhouette_guard`): the v4 car exposed the gap —
  a photometric spike can land BETWEEN the declared pose and the truth
  (commit az 12.5: registered silhouette IoU 0.787, ABOVE declared
  0.746, so the worse-than-declared veto passed it, while the true
  basin at az -25/+35 measured 0.908 and coverage collapsed to 5.2%).
  An accepted commit is now additionally tested against the rescue
  lane's double-keyed decisive evidence, with the aspect key measured
  at the COMMIT pose (a correct commit renders the photo's aspect, so
  its own small aspect error blocks the override). Revalidated: car v4
  az +20, car v2 az +30, chair az 0 (override corrects a spurious NCC
  commit at az -27.5 back to declared), owl/face unchanged, starship's
  correct NCC commit az +30 untouched.

End-to-end validation (sports car, t23d, Klein-4B): v2-v4 shipped
degraded (exit 3, coverage 0.05, zero accepted references, gray or
misprojected candidates). With the full fix set the same command ships
HEALTHY (exit 0): pose recovered by silhouette rescue (az -25, riou
0.900 vs 0.736 declared), both side views accepted with clean red
paint (ensemble 6.9/9.4), coverage 0.194. Back and top remain
fill-completed: their best candidates score ensemble 12.42-13.03
against strict 12 — visually correct red but inside the [12, 14.2]
ambiguity band between the worst accepted fleet view (9.79) and the
smallest measured true failure (14.2). Strict is NOT loosened to admit
them (the chair incident is the precedent against shipping the
ambiguity band); recalibrating the anchor-class strict line on the
now-persisted corpus is the tracked follow-up.

Known limitations carried forward from the audit, tracked not fixed:
`image_strength` is parsed and silently discarded by the FLUX.2 edit
routes (abstractvision backend — dead lever, removed from any planned
escalation ladder); G2 baked-specular threshold cannot fire on
dark-palette subjects (recalibration held pending a corpus margin
check); despecular treats 0% of large clear-coat lobes (inert-safe;
real clear-coat handling belongs to cross-view reconciliation).

### Fixed (generated references — four-audit root-cause program on the owl)

Four adversarial audits (fusion math, geometry-anchored alignment, tone
junction physics, mesh forensics) converged on the user-visible owl
defects (displaced tail, distorted wing, left flank band). Mesh audit
verdict: the mesh is clean and symmetric — every defect was pipeline-
caused. Root causes fixed, in causal order:

- **Clay guides were pitch black off-front** (`rendering.py`): the
  offscreen renderer's key light was FIXED IN WORLD SPACE, so back/side
  clay renders sat at the Lambert clamp floor (measured: back interior a
  constant 24/255 over a 16/255 canvas — the i2i generator was
  conditioned on a near-invisible blob and INVENTED the interior features
  it was never shown: painted tail 0.36x the real wedge width, wing-V
  rotated and displaced 57 px). Untextured renders now default to a
  HEADLIGHT (light = view direction: n.v > 0 on every visible surface, no
  clamp, no terminator) so guides show the true relief from every angle;
  regenerated references paint the tail on the mesh's actual wedge.
  `lighting="fixed"` remains for the photometric pose estimator, whose
  margins were calibrated against the legacy shading (the headlight
  gradient field flipped a validated declared-pose case).
- **Registration matched far-side views against surface they cannot see**
  (`register_reference_by_source_overlap`): the overlap set had no
  visibility test — for an orthographic back camera every front texel is
  "in frame" THROUGH the body, so the fit aligned the back photo against
  front-photo content and drove every reference to the search bound
  (+-0.08 = 61 px: the mechanical part of "the tail moved"). Overlap now
  requires the surface to face the reference camera, and a silhouette
  guard rejects any warp that costs more than 1% of the view's on-mesh
  sample coverage (texture-locking fits measured 46-61 px shifts on
  views whose silhouette IoU was already 0.93+).
- **Two-band fusion edge math** (`blend_projections`): the low band is
  now a WEIGHT-carrying normalized convolution (binary-coverage masks
  admitted weight-crushed rim texels at full strength; the one-sided edge
  mean then turned rim shading into a large negative detail band —
  measured as a flank strip darker than BOTH witnesses, Y 0.31 vs
  0.52/0.57), and detail ownership argmaxes MASK-NORMALIZED smoothed
  weights with a 30%-of-local-best eligibility floor (plain smoothing
  halves a view's weight within sigma of its own coverage edge, letting
  grazing views own detail strips they saw 3x worse).
- **Photo sovereignty extended to the photo's own coverage edge**
  (`protect_observed_texels` floor 0.25 -> 0.02): correctly-registered
  references repainted obliquely-photographed front surfaces (ear tops,
  brow ridges at facing 0.2-0.5) with drifted tone — photo fidelity
  deltaE 19.2 -> 25.3; the old registration bug had masked the leak by
  pushing reference rims off the mesh. Stretched oblique photo content is
  the subject's own appearance; synthesis owns only what the photo never
  saw. Whole-bake gate on the final owl: PASS (fidelity 18.86 < 19.46).
- **Delight now applies to chain-constrained views** (consensus
  application): the SH delight fit always constrained far-side views
  through reference-reference overlap pairs, but the application and its
  fade were keyed to SOURCE overlap — structurally zero for a back view
  (facing cones end at ~66 and start at ~101 degrees), and the fade
  zeroed the sides' corrections exactly at the side|back junction where
  the seams form. Application, clipping, fade, and the revert gate are
  now keyed to the union of each view's FITTED overlaps; views whose
  gauge reaches the source only through a chain get half the amplitude
  budget.

### Changed (bake fusion — two-band detail ownership, research-grounded)

A three-track literature review (photogrammetry texturing practice,
generative texture pipelines, view-fusion theory) unanimously identified
the softmax-weighted view average as structurally wrong for detail:
averaging views with residual registration error is convolution with the
error kernel (two views offset by d texels null every feature finer than
~2d), and the softmax bias is a NO-OP exactly at weight ties — the whole
equal-facing ridge between adjacent view cones degenerates to a plain
average. Production tools (Metashape "mosaic", Baumberg two-band,
Hunyuan3D-Paint single-ownership) never average high frequencies.

- `blend_projections` now uses TWO-BAND fusion for multi-view bakes
  (`detail_fusion="two_band"`): the low band (tone, lighting) keeps the
  softmax average with its wide smooth transitions; the 2-8 px detail
  band is winner-take-all from the single best view per texel — argmax
  over weight maps smoothed at 8 texels (smooth the WEIGHTS, not the
  labels, or ownership dithers into slivers on the tie ridge), with a
  ~1.5-texel feather at handoffs (the band is only zero-mean BELOW the
  split scale; a zero-width switch still steps by local content).
  Single-view bakes are bit-identical; `detail_fusion="average"`
  restores the previous behavior.
- New HANDOFF-SEAM LEDGER in the blend stats (`handoff_seams`): tone
  disagreement (owners' low-band delta) measured exactly at
  detail-ownership boundaries, in texture space where the handoffs are
  known. Groundwork for the acceptance gate: a render-space seam metric
  cannot separate a genuine handoff seam from a crisp carved contour
  (measured: their side-tone distributions overlap completely).
- Measured and REVERTED, documented for re-landing: bicubic registration
  warps and cubic projection sampling recover 5-7% of the relief band
  each (owl back: 12.69 -> 13.54 texel-space band RMS), but their edge
  overshoot raises the acceptance gate's long-strong-edge statistic by
  the labeled chair-regression magnitude — real seams and restored
  carved contours become indistinguishable in the one metric that
  auto-protects unattended users. They return when the gate consumes the
  handoff ledger instead.
- Iteration protocol upgrade: every candidate bake in this program was
  screenshotted through MeshVault's headless viewer-truth endpoint
  (`GET /api/screenshot`) before judgment, and the full iteration ladder
  (10 bakes, 20 labeled contact sheets) ships in the review folder.
  Six-view angle densification (back_left/back_right at 135°) was
  generated, gate-checked, and parked: without cross-view consensus
  alignment the extra overlap washes tone (fidelity regression caught by
  the whole-bake gate), which is the Zhou-Koltun-style alignment stage
  on the roadmap.

### Changed (generated references — adversarial round 2: completion-only protection, strict-only baking, person bypass)

An independent adversarial review of the four-subject v2 bakes identified
three systemic failures the material gates could not see; each is now
closed at the layer that owns it:

- **Generated views complete, never revise** (`protect_observed_texels`,
  `texturing.py`): after the tone/consistency stages (which need overlap
  texels for their statistics), synthesized weight is zeroed wherever the
  strongest REAL view holds a credible claim (weight >= 0.25, the
  conflict-resolution priority floor), with a linear ramp below the floor
  so the real-rim -> generated-content handoff stays smooth. Root cause:
  weight subordination (x0.6) loses per-texel contests but the feathered
  blend still AVERAGES synthesis into photo-covered texels — measured as
  rust mottling on the chair's front fabric and skin blotches on the
  portrait's front face. A real photo is evidence; a generated view is
  plausible synthesis; synthesis must contribute nothing where evidence
  exists. A/B on the v2 reference sets: chair front-view contamination
  removed (10.4% of front pixels reverted to photo truth), zero regression
  on generated-exclusive surface.
- **Floor-accepts are reported, never baked** (`generate_reference_views`):
  selection now requires a STRICT pass on all three material oracles. The
  v2 chair measured why: a floor-accepted top view leaked stained fabric
  straight into the bake. A wrong texture on an unseen angle is a worse
  product defect than the featureless fill it displaces — fill is dull,
  wrong material is broken. Floor-only ladders surface in the report as
  `rejection_reason` with full per-attempt metrics.
- **Person subjects are refused unless explicitly acknowledged**
  (`person_policy`, default `"skip"`): the review measured generated side
  views drifting to a DIFFERENT person's face — different age, nose, skin
  — while every material gate strict-passed, because no gate in the stack
  measures facial identity. Both `auto` AND `on` refuse people: "on" is a
  texture-quality opt-in, not identity-synthesis consent. Synthesis of a
  person requires the person-specific acknowledgment
  (`allow_person_subjects` on `rebake_bundle`,
  `texture_reference_allow_person` on the backend,
  `--texture-reference-allow-person` on the CLI), which puts a
  `person_warning` on the record. The check FAILS CLOSED: the photo is
  captioned even when a non-person hint exists (a hint that doesn't name
  a person is not evidence of absence), and an unavailable captioner
  refuses instead of proceeding — an unavailable check is not a
  permission grant. Detection tokenizes alphabetic runs ("woman's"
  matches "woman") over a wide person-word list (incl. baby/human/bride);
  the robust upgrade path (face detector, identity-embedding floor) is
  tracked in the KnowledgeBase.
- **Whole-bake A/B acceptance gate** (`bake_acceptance.py`): per-view
  strict gates are structurally blind to COMPOSITION-level failure — on
  the chair, every shipped view strict-passed and the finished bake still
  regressed below the no-references baseline (a deltaE ~42 tone step
  where the generated top hands off to the protected front). When
  generated views enter a bake, the pipeline now also bakes the
  no-references baseline and ships the generated bake only if it does not
  regress three render-space axes: photo fidelity at the source pose,
  front brightness, and long coherent seam edges (extent-filtered so
  texture detail — grain, panel lines, plumage — never counts as a seam;
  the budget is ABSOLUTE, not a baseline multiple, because a multiplicative
  allowance lets a bad baseline launder a worse candidate). Calibrated on
  the labeled four-subject set: the chair auto-rejects (seam 0.138 vs
  ceiling 0.122) and ships its baseline with the verdict recorded in
  metadata; owl, spaceship, and portrait pass. Close-zoom triage of the
  owl's 13 dark-smear fragments localized all of them to the unwitnessed
  underside band (crevice shading speckle, worst per-view delta L -3.4 vs
  baseline); the close-range harness numbers stay on the record and the
  detector was left exactly as certified rather than recalibrated to
  flatter the feature.

### Changed (generated references v3 — zero-hint operation, material-identity gates)

- Reference generation is now fully autonomous: no subject hint is required
  anywhere in the API surface. When no user prompt exists, the source photo
  is captioned automatically (BLIP, `abstract3d.captioning`); when one does,
  it is used — and EITHER text is reduced to a material-free noun phrase
  (`extract_subject_noun`, a stoplist over material/finish/color vocabulary)
  before it may enter the generation prompt. Root cause, proven twice: any
  material claim in prompt text overrides the source photo's pixels (a
  hand-written "ceramic with glaze" hint regenerated a carved-wood owl as
  glazed pottery), while a correct claim adds nothing the photo doesn't
  carry. The prompt template has no free-text slot at all now; the source
  photo is the only material authority.
- New composite instruction (adversary-designed, subject-agnostic): leads
  with material-NEUTRAL relief vocabulary ("surface relief, carving depth,
  grooves, grain, cracks, fibers, micro-texture" — self-normalizing: for a
  smooth subject, copying its relief exactly yields smooth), names the
  output "a real photograph" (naming it a render biases CG-smooth output),
  maps materials PART BY PART (a wood-frame/fabric-seat chair must not
  spread one part's material onto the other), and forbids re-interpretation
  without naming any material class. A person clause (triggered by
  person-category caption words) anchors human subjects to "living person,
  real skin, real hair strands" — without it, i2i editors systematically
  render the clay panel as a sculpture.
- Acceptance is now a three-oracle gate stack run on the FINAL processed
  pixels (despecular and tone-match happen before gating, so the gate
  judges what the bake consumes), each catching a failure family the others
  are blind to, all calibrated on the critic-labeled v1+v2 result set:
  `texture_fidelity` (band-pass relief ratio + flat-fraction growth;
  catches wood→glaze smoothing), `part_material_fidelity` (k-means part
  palette, chroma-first distance with an L tolerance band for unseen-side
  shading; catches upholstery→camouflage flips the texture gate passes),
  and `gate_baked_speculars` (glossy highlight fields). Retry ladder:
  IoU failures re-roll the seed (stochastic), texture/material failures
  escalate the prompt (systematic bias); every IoU-passing candidate is
  scored, and only a STRICT pass may ship (see the adversarial round-2
  entry below — floor-only candidates are reported, never baked).
- Conditioning canvas fixes: both panels letterboxed (no anisotropic
  stretch of the material the model must copy), clay foreground composited
  onto the same dark background as the source panel (background mismatch
  read as "different photo sessions"), and the echo-crop heuristic now
  catches ANY wider-than-tall canvas echo (the old >=1.6 aspect test missed
  4:3 echoes and burned whole retry ladders).
- Despecular is relief-aware: pixels inside high band-pass-energy
  neighborhoods are exempt (carved-ridge micro-highlights satisfy the
  specular predicate; blending them toward the body estimate erased exactly
  the relief the transfer must preserve), and when the source photo itself
  flags a similar fraction under the same predicate, the correction blend
  is scaled down (measured 2% false-positive floor on matte carved wood).
- `auto` mode gate relaxed accordingly: it still requires an explicitly
  configured local image provider (never a silent remote route), but no
  longer requires a subject hint. Zero-hint four-subject validation
  (owl / chair / spaceship / portrait, FLUX.2-klein): 14/16 angles
  accepted with materials preserved; the two rejections are honest (a
  chair profile whose IoU never clears the gate, and a chair side whose
  camouflage-mottle material flip the part gate caught on every ladder
  attempt — that sector falls back to witnessed-texture fill, which
  cannot flip materials). Klein-9B resolves the portrait family (strict
  passes where 4B floor-accepts wet-look hair); documented as the
  recommended model for human subjects.
- KNOWN LIMIT (documented in KnowledgeBase): semantic re-rendering that
  preserves palette AND relief energy (v1's "sculpted goo" hair) is
  invisible to every foreground statistic tested; the countermeasure is
  generator quality (Klein-9B), not gating.

### Changed (generated references v2 — composite conditioning for source coherence)

- A coherence audit showed clay-only conditioning produced shape-correct
  but materially unfaithful views (the owl came back pale cream: LAB
  distance 28.4 from the source, hue correlation 0.44) — the i2i model
  never saw the source photo. The conditioning image is now a COMPOSITE:
  source photo (left panel) + clay render (right panel) with a
  texture-transfer instruction. Same model, coherence doubled: LAB
  distance 7.4, hue correlation 0.82; texture QA PASS on all three proof
  bundles including the previously-open fill-energy gate
  (`artifacts/validation/generated-reference-completion/`, v2).
- New `register_matte_to_clay` similarity registration (downsampled IoU
  search, winning transform applied at full resolution) absorbs the
  editor's small reframing before the acceptance gate, keeping the shape
  lock (raw composite IoU 0.74-0.83 -> registered 0.89-0.97).
- `conditioning` strategy parameter ("composite" default, "clay" and
  "rotate" available); provenance now records the strategy and per-attempt
  registration. Model notes: FLUX.2-klein-9B is HF-gated (stored token
  expired — refresh to enable); Qwen-Image-Edit-2511 8-bit downloads and
  registers but did not produce a first denoise step within 8 minutes on
  this host and is parked with the "rotate" strategy ready for it.

### Added (generated reference views — single-photo coverage completion)

- `abstract3d.reference_generation`: when a caller provides only ONE photo,
  the pipeline can synthesize the unseen angles and feed them into the
  certified bake as ordinary references: clay-render the reconstructed mesh
  from each target angle (silhouette lock; moderngl renderer REQUIRED — the
  matplotlib fallback's decimated silhouette would blind the gate), condition
  an `abstractvision` i2i generation on that render, gate acceptance on
  silhouette IoU >= 0.75 against the clay silhouette, suppress baked specular
  highlights (pale desaturated blobs vs the local diffuse body estimate),
  and cap-limited LAB tone matching toward the source photo (mean shift
  clamped per channel so a legitimately different unseen side is never
  whitewashed into the front photo's statistics; pre-match distance and the
  applied shift are recorded). Measured on the certified owl: observed
  coverage 0.30 -> 0.83 with four generated views (back/left/right/top),
  every acceptance first-attempt (IoU 0.92-0.98).
- Generated views are SUBORDINATED witnesses in the bake: their projection
  weights are attenuated (0.6) so they lose every per-texel contest against
  real photo content; the source view keeps its single-view facing semantics
  and scarcity-rescue stays off unless a REAL reference exists (generated
  views must not flip the certified single-photo regime). `observed_view_stats`
  rows and bundle metadata mark generated views explicitly.
- Hunyuan backend option `texture_reference_generation` (auto/on/off,
  default auto) with `texture_reference_generation_angles` (labels or
  `label:azimuth,elevation` entries — the validated starship underside is
  `bottom:0,-75`); CLI flags for both; `rebake_bundle` gains
  `generate_references` + `generation_angles` + `subject_hint`.
- Adversarially hardened before landing (1 controller agent, 15 findings):
  "auto" fires ONLY with an explicitly configured image provider (never the
  remote fallback route) AND a non-empty subject hint (the i2i model
  conditions on an untextured clay render; without subject knowledge it
  invents materials for exactly the default one-photo user) — otherwise it
  skips with an actionable warning; "on" expresses explicit intent. Full
  provenance is recorded per bundle: resolved provider/model, prompts,
  negative prompt, seeds, per-attempt IoU, accepted-image hashes, clay
  renderer, tone shifts; generated photos and their clay conditions are
  persisted as `texture_reference_generated_*.png`. The un-matted source
  photo now rides as `identity_image` on the backend's source view (the
  fringe-repair correspondence needs it). Honest scope, documented: a
  generated view is plausible synthesis, not ground truth — content on
  fully unobserved regions (a person's back of head) is invented, and the
  three side views are generated independently (no cross-view content
  consistency beyond tone).

### Added (executable golden-bake regression harness + public bundle API)

- `scripts/golden_bake.py` turns the certification's determinism claim into an
  executable gate: it rebakes the three certified proof assets through their
  canonical recipes and fails unless every baked `texture.png` reproduces the
  published hash bit-exactly. `--profile` adds per-stage wall time and RSS/MPS
  memory attribution (one process per asset so peaks stay attributable).
- `abstract3d.bundle` — the previously script-only rebake path is now a
  supported API: `load_bundle` / `prepare_observed_views` / `rebake_bundle`
  load a bundle's canonical `geometry.glb`, rebuild the observed-view list
  (source matting, reference angles, the identity-image contract), rebake,
  and write a versioned bundle revision (`schema_version`, texture md5,
  trimmed bake stats). Documented caveat: TripoSR bundles rebake without the
  resident triplane color prior; the certified Hunyuan bundles rebake with
  full fidelity.
- `abstract3d.profiling` — read-only stage/memory profiler (background RSS
  sampler + externally-wrapped stage functions), used by the harness;
  profiled runs stay bit-identical because nothing touches array state.

### Changed (strict generation-option contract)

- Backends now REJECT unknown generation options with the new typed
  `InvalidRequestError` instead of silently ignoring them (the CLI itself
  was sending diffusion knobs to the feed-forward TripoSR path with no
  effect and no warning). Each backend consumes its supported options and
  the leftovers fail loudly — before the expensive inference stage on the
  diffusion backends, so a typo costs milliseconds, not minutes. Envelope
  keys (`artifact_store`, `run_id`, `tags`, `metadata`) are exempt.
- The CLI forwards only explicitly-set flags (None-valued options are no
  longer sprayed at every backend) and exposes the mesh-density controls
  that were previously Python/config-only: `--octree-resolution` and
  `--max-facenum` (hunyuan3d21/step1x).
- Composed `t23d` image options are consumed through one helper
  (`pop_composition_kwargs`), so `image_provider/model/width/height/seed`
  are recognized composition keys on every backend and unknown `image_*`
  spellings fail like any other typo.

### Added (visual quality review protocol)

- `artifacts/validation/quality-review/`: reproducible per-backend quality
  scoring — two representative bundles per backend/task inspected through
  the headless MeshVault MCP server (structural `describe_scene` + three
  canonical renders each), with every render, the rubric, and per-group
  evidence versioned (`scores.json`). The benchmark table now carries
  mesh/texture quality scores and the model license per backend.

### Added (generation statistics + headless MeshVault verification)

- `scripts/generation_stats.py` aggregates wall time, stage times, and mesh
  density (vertices/faces) from every bundle `metadata.json`; the summary
  table and the time/density control matrix (what governs mesh size per
  backend, defaults, and Python-vs-CLI exposure) are published in
  `docs/benchmarks.md`. Known exposure gap recorded: `octree_resolution` /
  `max_facenum` are honored as Python kwargs and config keys but have no
  CLI flags yet.
- The rebaked assets were verified through the MeshVault MCP server driven
  headless over stdio JSON-RPC (`load_model` + `screenshot`), in addition
  to the interactive app check; proof render in
  `artifacts/validation/bake-performance-program/`.

### Changed (bake performance program — outputs bit-identical)

All optimizations below reproduce the certified texture hashes bit-exactly
(verified per-change on captured stage inputs AND end-to-end by the golden
harness; before/after evidence in
`artifacts/validation/bake-performance-program/`). Measured on the golden
recipes at res 2048 (Apple M5 Max): owl 258 s -> 88 s (2.9x), face
220 s -> 167 s (1.3x), ship 59 s -> 55 s (1.1x). Memory peaks -0.15 GB
(ship/owl); the peak-structure analysis is recorded in the profiles.

- `mirror_fill_from_observed`: the exact-NN mirror-twin lookup now runs
  parallel (`workers=-1`) and pruned at the acceptance threshold
  (`distance_upper_bound`) — most mirror twins land nowhere near an observed
  texel (1.6% acceptance measured on the owl), and unbounded exact-NN
  backtracking dominated the stage (167 s -> 1.1 s, 148x, bitwise-identical:
  pruned misses return inf and are dropped by the same `valid` mask).
- `synthesize_fill_detail`: donor k-NN queries go through `_balanced_query`,
  which randomizes query order before scipy's per-thread chunking (atlas-
  ordered queries give whole chunks of far-from-tree texels to one straggler
  thread) and undoes the permutation on return — exact same per-point
  results, 3.8x on the owl donor query. Full-atlas statistics intermediates
  (~0.5 GB) are released before the long query phase; the two observed
  quantiles they feed are computed ahead, unchanged.
- `commit_pale_chips`: per-blob work (masks, isolation dilation, gathers)
  now runs inside each blob's bounding window via `find_objects` (margin
  covers the dilation) with the loop-invariant plain-domain colors hoisted —
  474 committed / 2264 candidate blobs previously paid full-atlas ops each
  (42.8 s -> 0.8 s, 53x, bitwise-identical).
- `commit_trace_deposits`: eval units are stored as (window, local-mask)
  pairs; world-space ring/residue tests evaluate over precomputed flat
  domains (row-major extraction preserves reduction order bit-exactly);
  full-atlas masks are materialized only for units that actually commit
  (17.7 s -> 6.9 s on the face proof, bitwise-identical).
- Index maps for the flat domains use int32 (identical indexing behavior,
  half the footprint).

## 0.2.0 (2026-07-08)

First public release of the standalone repository (`github.com/lpalbou/abstract3d`).
Validated operating profile: Apple Silicon (`mps`), Python 3.12. See `README.md`
for the current backend/OS support matrix.

### Release engineering

- CI/CD on GitHub Actions (`.github/workflows/release.yml`): test matrix
  (ubuntu + macOS, CPU torch, headless GL), sdist/wheel build with twine check,
  GitHub release on `v*` tags, optional PyPI publication (skips without the
  `PYPI_API_TOKEN` secret), and a MkDocs Material doc site deployed to GitHub
  Pages (`https://lpalbou.github.io/abstract3d/`).
- Versioning policy for validation artifacts: only current-state experiments
  are versioned (certified bundles, certification record, generated-reference
  proofs); superseded experiment archives stay local (`.gitignore` allowlist).
- Cross-host portability fixes surfaced by CI: the Hunyuan license gate now
  fires before the optional-dependency check; Step1X seeding falls back to a
  CPU torch generator on builds without the resident backend; the
  gradient-domain determinism test encodes the portable contract (<= 1 LSB at
  quantization boundaries) while bit-identity remains the measured guarantee
  on the validated Apple-local profile.

### Cycle 8 — viewer-orientation export (EXP-01)

- Textured exports now present the glTF viewer frame: the pipeline's canonical object
  frame is Z-up / front +X while glTF mandates Y-up / front +Z, so every
  standards-compliant viewer displayed exports lying sideways. `_mesh_export_bytes` and
  the OBJ exporter bake the exact axis permutation (x, y, z) -> (y, z, x) into exported
  vertices (float-exact; texture bytes verified byte-identical) and stamp a persisted
  `abstract3d_export_frame` marker (glTF extras, survives round-trips). The repo
  renderer and the texture-QA harness detect the marker and rotate marked meshes back
  into canonical-frame math, so all gates measure identically (verified: face raw
  identity improved to SSIM 0.676 under the marker-compensated render; texture_qa
  13/13 on all three assets). Internal working files (`geometry.glb`, consumed by
  rebakes) keep the canonical frame via `viewer_frame=False`. All three certified
  bundles re-exported upright; MeshVault verification:
  `artifacts/validation/texture-cycle-proofs/upright_verification.png`.

### Cycle 7 — reference leverage

#### Hardened (MANDATORY first item of this pipeline change, per the certification contract)

- `feature_fringe_repair._render_structure_veto` — CUMULATIVE-BASELINE VETO
  (critic 2's cycle-5 recommendation, adopted as mandatory by the cycle-6
  certification): the advancing per-stamp baseline re-armed the +0.0003
  micro-island budget with every acceptance (measured: ~7 stamps produced
  +0.00096 at one view, triple the single-stamp budget, inside the letter of
  every per-stamp check). The veto now also refuses any candidate whose
  post-stamp micro fraction exceeds BOTH the view's ORIGINAL pre-repair
  fraction + 0.0003 AND the original battery-wide worst; the photo-truth
  exemption bound is pinned to the ORIGINAL battery worst for the same
  reason. Per-stamp advancing semantics unchanged. MD5-neutral on the
  certified face: the canonical-recipe 2048 bake reproduces
  `928705f3edfc9036348c12bf34435d9d` bit-exactly (predicted by critic 2's own
  measurement — the accepted creep stayed under the original battery worst).
  Test: `test_render_veto_cumulative_baseline_closes_rearm_creep`.

#### Added (reference-leverage ledger — permanent instrumentation)

- The project owner's standing critique ("the pipeline under-leverages the
  reference photos") is now measurable per bake: `bake_projection_texture`
  stats carry `leverage` — per view potential/painted/won texels with
  per-gate surrender attribution (facing gate / layered-zone gate /
  downstream kills / union drops), plus union ratios (photo-visible,
  direct-painted, leverage, surrendered-visible, unobservable) and the
  mirror-over-photo-visible watch (G4). The projector emits per-view
  diagnostic maps (`potential`, `zone`, `facing`, `geometry_factor`,
  `scarce_weight`, exact per-texel stretch). `scripts/texture_qa.py` prints
  the ledger as a non-gating reporting block and stores it in results.json.
  Instrumentation is md5-neutral (pinned-vs-current 1024 pair bit-identical).
- Measured on the certified face at 1024 (the honest inventory the critique
  asked for): photo-visible union 50.7% of surface, direct-painted 45.0%,
  leverage 88.8%, photo-visible-but-surrendered 5.8% (12,449 texels),
  unobservable 49.3%. The "sees 57% / paints 21%" reading compared the
  geometric-visibility union against the CONFIDENT-weight set (winner weight
  >= 0.35 = 25.0%); the painted-at-any-weight set was already 45%.

#### Added (G1: witness-scarcity admission — `admit_scarce_witnesses`)

- On texels NO view claims at its strict facing threshold (the certified
  bake surrendered 4.8% of the surface that at least one photo sees:
  jaw/cheek silhouette bands, under-chin, hairline/crown transitions), the
  bake now admits below-threshold witness claims bounded by the EXACT
  per-texel sampling stretch (<= 4.0, the texel->photo Jacobian — facing is
  a tilt proxy and cannot see collapsed mappings), above a grazing floor
  (facing > 0.05), still respecting first-surface visibility, photo alpha,
  the layered-zone surrender, and the stretch/concavity demotion.
  "Stretched content beats no content" — the single-view doctrine —
  generalized to per-texel witness scarcity; where ANY strict witness
  exists, the calibrated strict gates keep the texel and every scarce claim
  stays discarded.
- Admission guards (each measured load-bearing at 1024; unguarded admission
  lifted dark_debris 0.0022 -> 0.0038 vs the 0.003 gate): (1) contradiction
  of a color-consistent confident consensus (the mirror-copy guard's rule);
  (2) like-material support in BOTH directions (dark-on-bright is the
  debris/flake class, bright-on-dark is the FACE-07 pale-chip class —
  measured crown-flake failures 0 -> 0.0022 without it); (3) dark-mass
  adjacency (dark commitment requires the dark BODY; a nearby dark FEATURE
  licenses nothing); (4) a FEATURE MOAT (no admission within 0.044 x scale
  of a strong dark feature core — parallax-displaced feature adjacency was
  the measured debris source, and features are strictly witnessed by
  construction so the moat costs no leverage).
- Placement (two measured non-local failures): admission happens AFTER the
  global compositing solve as a strictly local paint (an early admission
  re-shaded photo-true content 20+ px away through the Poisson anchor set
  and flipped three knife-edge debris detectors; the fringe stage's
  pre-repair baseline inherited the drift and its exemption bound loosened)
  and BEFORE mirror completion (real observation beats symmetry guess —
  mirror no longer guesses surface a real witness paints). Rescued texels
  inherit the delight/harmonization tone corrections (the application masks
  now include scarce candidates; the fits never see them).
- `consolidate_unwitnessed_debris`: render-informed lift (the fringe lane's
  displaced-refill discipline) of isolated bright-ringed sub-feature dark
  islands whose first-surface texels are predominantly UNWITNESSED — fill
  pockets re-partitioned by the admission that the fill floor's
  anchor-tracking exemption correctly keeps but the absolute debris
  detectors count. Runs only when scarcity admission ran (bakes without it
  stay bit-identical); the island construction anchors its dark split to
  the light material's own median (the binding detectors' construction)
  and feature-class blobs are protected by the render battery's own
  footprint.
- `scarcity_rescue="auto"` enables the mechanism for multi-view bakes only;
  single-photo proof assets are pinned regression canaries: fresh 2048
  bakes with the change ON reproduce ship `b8e2b0d4...` and owl
  `ff746509...` bit-exactly. A measurement-only single-view ablation
  (ship, rescue forced on) admits 578 texels (+0.1 point leverage,
  texture_qa 13/13) — not worth re-certifying a frozen canary, so the
  auto scope stands.
- Face results (all gates green at both resolutions): at 1024,
  direct-painted 45.0% -> 45.4%, comp identity 0.705/14.7, detectors
  green, texture_qa 13/13, bit-deterministic. At 2048 (canonical recipe,
  determinism pair `2baf7408...`): compensated battery PASS 0 failures
  (front 0.70182/14.876), raw detectors green with the worst dark
  IMPROVED (0.0027 -> 0.00262) and comp MAE margin IMPROVED
  (0.09 -> 0.124), texture_qa 13/13, direct-painted 43.4% (+8,224
  photo-witnessed texels at the jaw/cheek/under-chin bands).
- PUBLICATION: the 2048 candidate is STAGED, NOT published. The comp SSIM
  knife-edge consumed 50.8% of its certified margin (0.7037 -> 0.70182,
  half-margin line 0.00185, consumed 0.00188) — per the certification's
  maintenance contract §5, more than half of a knife-edge margin requires
  a fresh critic battery, not just the harnesses. The certified bundle
  (`928705f3`) stays on disk; staged bytes + full harness evidence:
  `/tmp/c7/staging_face2048` and `/tmp/c7/REPORT.md`.

### Certified (zero-defect adversarial program, cycles 1-6)

- The three proof assets (multi-view face, single-view starship, single-view owl) are
  CERTIFIED at the zero-open-defect standard by the program's independent verdict agent:
  23 defect-ledger entries FIXED, 10 closed as PROVEN-LIMIT with the exact capture remedy
  documented per entry, 0 OPEN. Certification document (ledger state, maintenance
  contract, knife-edge watch thresholds, and the honest definition of zero-defect within
  the given inputs): `artifacts/validation/texture-cycle-proofs/CERTIFICATION.md`. Final
  face state: compensated identity 0.704/14.9 (anchored gate 0.70/15.0) with the full
  28-view compensated battery at zero failures, raw detectors green, texture_qa 13/13 on
  all three assets, and four independent canonical-recipe bakes sharing one texture hash
  (bit-deterministic pipeline). All six cycle rulings, both critics' mathematical
  reviews, and every solver report are preserved under
  `artifacts/validation/texture-cycle-proofs/`.
- Late-cycle mechanisms (each adversarially verified before certification):
  gradient-domain view compositing (screened Poisson over the texel surface graph),
  validated dense reference flow, film-band gradient repaint with off-pose displacement
  veto, feature-fringe repair driven by the identity gate's own correspondence,
  shadow-apron reconciliation (source cast-shadow truth vs reference lit tone),
  world-space voxel-graph feature-complex clustering, trace-deposit commit with rim
  feathering, field-support bounds on geodesic tone extrapolation, and completion tone
  matching — plus the publication checklist born from the FACE-21 incident (no bake
  ships without `identity_image` and pre-overwrite harness verification).
- Adopted forward-process governance from the certification: Critic 2's cumulative-veto
  hardening for the fringe lane's growth budget is the MANDATORY first item of any
  future pipeline change, and any texturing change re-proves determinism, re-runs the
  full gate set on staged bytes, and re-verifies the frozen canary hashes before
  publication.

### Fixed (FACE-22: region-boundary line-art on smooth skin — cycle 6)

Thin line-art contours on the neck/chest (a glyph-like cluster at az0 and
a large closed contour at az-22.5), pipeline-attributed (both the front
photo and the reference profiles are clean under contrast stretch at
those regions). Provenance established by a fully instrumented bake
(per-stage texture captures + per-mechanism ablation bakes + internal
mask captures at 2048; all difference maps in the cycle-6 evidence):
three mechanisms printed the marks, each fixed in its own vocabulary.

- `film_band_gradient.repaint_film_band` — FIELD SUPPORT BOUND
  (`FIELD_SUPPORT_TRANSITIONS`): the geodesic S field is a distance
  RATIO and takes mid-transition values arbitrarily far from the hair
  mass, so the repaint treated neck/chest skin at 9-24 pooled-profile
  transition lengths from the mass (S~0.66) where the measured falloff
  profile has no support: its envelope clamp printed the az0 glyph
  cluster (small clamp components, hp -0.058), its stamp borders and
  displaced-refill component borders printed contour segments and the
  az-22.5 closed contour. Treatment is now confined to within 6
  transition lengths of the mass (film strokes measured d_mass p5 8.8T
  vs the honest apron's p50 2.0T), feathered over the last transition;
  and authority stamps blend composite -> photo over
  `STAMP_BORDER_FEATHER_TEXELS` at treated-region borders (mass borders
  exempt — the stamp continues the mass content there), which also
  removes the support-cut chroma seams (measured 0.49-0.69 -> 0.13-0.23
  at az+22.5/+70).
- `texturing.commit_trace_deposits` — RIM FEATHER (`rim_feather_texels`):
  the deposit's antialiased border mixtures sit below `deviation_min` by
  construction (mixture deviation = coverage x deposit deviation), so
  the commit retoned the interior and left a 1-3 texel dark outline —
  the az-22.5 closed contour's crisp component. Rim texels carrying the
  same evidence class (direct, trace-weight, bright ball context,
  outside the film commit) now blend toward the ring-anchor tone,
  distance-decayed and ONE-SIDED (only darker-than-target texels move);
  the interpolation anchors exclude the feather band itself (a rim
  mixture bright enough to be an anchor otherwise pins its own darkness
  in place — measured: 483 -> 2678 feathered texels after exclusion).
- `texturing.tone_match_completion_components` (new, called from the
  mirror-completion block, multi-view bakes only): mirror completion
  copies the twin verbatim; on a lighting-asymmetric subject each copy
  lands at a tone offset from its destination (measured +16/255 on the
  chest) and its border prints as a contour. The legacy seam leveling
  reconciled mirror regions, but the gradient-domain solve runs BEFORE
  mirror completion — this is the missing handoff. Pure-bright copies
  against bright destination rings take a component-level log-median
  gain (clamped, detail verbatim); mixed-material copies and dark-ring
  components stay verbatim (measured: rescaling them re-classifies
  their own dark micro-content and mints dark_debris islands,
  0.0031-0.0036 vs the 0.003 gate).

Measured at 2048 (canonical recipe): the glyph cluster and the closed
contour are gone at the critic's crop framings; stroke-texel detector
inside the two probe boxes 4281 -> 3474 (glyph probe 36 -> 21, az-22.5
probe 2126 -> 1538; the remainder is soft pre-existing blend-content
boundaries, not line-art). Compensated battery PASS (identity[front]
0.7037/14.91 vs gate 0.70/15.0), raw battery: detectors all green, raw
front MAE 21.67 green (SSIM raw-diagnostic per the cycle-4 ruling),
texture_qa PASS 13/13. Single-photo canaries (ship/owl) bit-identical
(the new tone match is multi-view-gated; the film repaint and the
commit are structurally multi-view-only).

### Added (source-shadow apron reconciliation — the FACE-04/FACE-14 neck wash)

`gradient_compositing.reconcile_shadow_aprons` (+ `apply_shadow_apron_scale`),
wired into `composite_gradient_domain` beside the specular reconcile: the
DUAL of the FACE-05 mechanism. Where a REFERENCE view wins co-witnessed
surface with its own lit reading while the SOURCE photo (the identity
contract holder) validly samples the same surface substantially darker —
its cast shadow (chin/jaw onto the neck: measured -0.35 log vs a -0.08
pairwise gauge, source projection weight ~0 at the down-sloping neck) —
the composite carries the source's shading baseline there. The identity
gate at the source pose compares that surface against the source photo,
and the renderer's flat-biased headlight (~0.9 at the neck) cannot absorb
a real cast shadow, so only the albedo can carry it. Guards, each with a
measured counterexample: source-valid-only (no witness demotion where the
source has no evidence — the photo-curtain parallax band beside the wash
stays untreated), pairwise lighting gauge + margin (exposure differences
are not shadows), source-photo edge-density refusal (the curtain edge is
edge-dense; a shadow is smooth — refused components measured p85 2.8-5.8
vs the shadow's 0.5-1.3), world-ball fragment merge before the size floor
(the atlas cuts one apron into sub-floor UV fragments), one-sided
darkening only with per-consumer detail preserved verbatim (the correction
reduces to a smooth luminance scale; no reference chroma or detail is
imported, no brightening path exists). Measured at 2048 (canonical recipe,
paired A/B): compensated identity[front] +0.005 SSIM / -0.2 MAE, sides
within budgets, all detectors green; single-view bakes are structural
no-ops (no reference can win a texel) — ship/owl canaries bit-identical.

### Changed (feature-fringe repair: world-space complex formation + photo-truth exemption)

Three cycle-5 changes to `feature_fringe_repair`, each closing a measured
2048-resolution shortfall of the cycle-4 mechanism:

- Complex formation now clusters core texels in WORLD SPACE
  (`_cluster_core_texels_world`: voxel-graph connected components at a
  link cell of 0.006x the mesh diagonal, the rescue detector's
  construction) instead of atlas morphology. Atlas dilation counts
  TEXELS, so the same world gap spans twice the texels at 2048 and UV
  chart cuts fragment one physical feature into sub-floor pieces before
  any world merge can see them (measured: the mouth complex formed at
  r 0.045 vs the 1024 run's 0.11 and the lip-edge dash stayed
  half-covered; the chin complex never formed at all). World clustering
  is resolution-independent and chart-blind by construction.
- The render-space structure veto's micro-island growth budget carries a
  PHOTO-TRUTH EXEMPTION: a new sub-feature island whose pixels render
  from stamped texels the registered photo CONFIRMS is the photo's own
  anatomy (lip-corner line, lash fragments), not invented structure —
  measured: the chin/mouth-surround stamp that banks +0.006 compensated
  SSIM was refused for printing exactly what the photo prescribes. The
  exemption is BOUNDED by the battery's own pre-repair worst case (no
  view may become the new worst micro-island offender; measured
  unbounded, the eye complex's full re-registration pushed two views
  past the absolute debris detectors at 0.0030/0.0032 vs the 0.003
  gate) and feature-size new blobs stay banned unconditionally. The
  veto baseline now advances with each ACCEPTED stamp so a stamp's own
  exempted content is not counted as growth against later candidates.
- A final render-informed speck consolidation
  (`_consolidate_render_specks`) lifts repaired texels that render as
  NEW isolated sub-feature dark islands at any battery pose (relative
  to the pre-repair baseline) to just above the dark class under that
  view's own shading — the FACE-20 displaced-refill floor discipline at
  micro scale — with texels rendering inside any pre-existing
  feature-class blob's own pixels protected (measured: an unprotected
  lift brightened the az+90 profile eye's under-lash mass and
  eye_count dropped 1 -> 0). Whole-complex stamps also apply an
  in-stamp speck guard lifting stamp-made bright-ringed micro specks.

Measured end-to-end at 2048 (canonical recipe, all cycle-5 changes,
paired against the cycle-4 published baseline): compensated
identity[front] 0.688/14.42 -> 0.702/13.86 (comp gate 0.70/15.0 PASSES),
raw MAE 21.69 -> 21.45 (budget 22.0), full 28-view detector battery
green, texture_qa PASS 13/13, bit-deterministic across three bakes; at
1024 the full compensated battery stays PASS (0.705/14.8). Ship/owl
canaries bit-identical to the certified on-disk hashes.

### Added (feature-fringe repair — the protected-feature deposit class)

`feature_fringe_repair.repair_feature_fringes` (new module, wired into
`bake_projection_texture` after the fill floor, multi-view only): the
FACE-03/04 residue that `commit_trace_deposits` measurably cannot treat —
displaced-content chips and dashes INSIDE protected feature complexes
(tear-duct whites, lash-line dashes, the lip-edge dark-red dash), whose
surround consensus is feature-mixed by construction (cycle-3: ring votes
0.30-0.81 vs the 0.96 bar; committing them cost eye_count) — is repaired
with the photo's own content under the identity correspondence. The stage
rebuilds the identity gate's own registration in-bake (render at the
declared source pose, alpha-bbox map + NCC-refined similarity against the
caller-provided `identity_image`), z-buffers first-surface visibility,
and applies rescue-disc transplant semantics (tone match + feather +
whole patch) at two scales: whole-complex corrective stamps (mode ladder
full -> trace-only under a never-demote rule: non-source confident
content is never overwritten; source-confident content may be
re-registered to the gate correspondence) and deposit-scale patches.
Rescue-disc interiors are never photo-stamped (the disc fired because the
photo evidence there is bad); their fringe deposits re-copy through the
disc's own anchored correspondence and the disc is refreshed last so
healthy-side repairs propagate into the twin (the transplanted eye's
tear-duct chip was measured to be a COPY of the healthy side's chip).
Every stamp passes structure-preservation vetoes: a texel-space check
under the renderer's own shading model, then a render-space check with
the pipeline renderer at 15 views (no new/lost anatomical-feature-size
compact dark blob vs the pre-repair render; sub-feature micro-island
fraction budget +0.0003). Measured at 1024 (face proof, paired A/B):
compensated identity[front] 0.668/16.26 -> 0.708/14.7 (raw 0.643/21.6 ->
0.680/21.4), full 28-view battery PASS with zero detector regressions;
tear-duct/lash-line/lip-edge crops visibly repaired at 4x. Ship/owl
canaries bit-identical (single-view structural no-op, enforced by test).

### Added (off-pose displacement veto — the FACE-20 billboard strokes)

`film_band_gradient`: the hairline gradient repaint's source-authority
stage stamped hard BLACK stroke/arc artifacts across five-plus views (a
jagged crack down the left temple at az0, a feathered streak along the
temple silhouette at az-22.5, a ragged line at the az-90 hairline, black
arcs tracing the ear helix at az+90/+112.5). Provenance (replay-traced
per stroke): every stroke component was 80-99% source authority stamps
plus their gap-diffusion extension, carrying the front photo's own
CURTAIN-EDGE / EAR-SHADOW pixels (photo-space bins within ~1 transition
length of the dark-body boundary) billboarded onto surface the source ray
only grazes — and every component was ALREADY VETOED by another view's
base-material witness (veto consensus 0.7-1.0) while sitting fully
outside the feature moat, the only place the shipped mechanism consulted
the veto. The fix extends the veto by field position instead of by moat:
a connected would-be dark-stamp component whose texels are mostly vetoed
(>= 0.5) AND whose median S sits in the skin half of the transition
(>= 0.35, where the photos' own pooled falloff says the surface has left
the hair body) is rejected as parallax-displaced content
(`_displaced_stamp_components`). Near the mass (S below the gate) equally
vetoed dark stamps remain — they are the wisp/strand content whose global
veto was measured at -0.05 SSIM in cycle 3. Rejected sites refill AFTER
all guards: the local guard tone rescaled to the photo's luminance
pattern at 0.30 gain on a floor 1.02x the dark-material split — strictly
above the dark class, so the site cannot render as a dark stroke at any
pose by construction, while still paying part of the source-pose identity
contract. Measured (2048 face, same-tree A/B, fringe stage isolated): all
40 stroke-class components dead across the 48-view battery vs the C2
baseline with zero new flags introduced by the veto (residual flags are
shared 1:1 with the veto-off arm); front identity comp 0.674 vs 0.637
(veto-off, same tree) / raw 0.648 vs 0.612; side identities unchanged or
better; all 28-view detectors green; texture_qa PASS 13/13. Single-view
bakes structurally unreachable (ship/owl md5 pairs verified identical
with the veto forced on/off).

### Added (specular-lobe reconciliation in the gradient compositor — the pale seam column)

The pale desaturated column running inner-eye -> nose flank -> philtrum at
4x (three cycles open) was provenance-traced to the SOURCE PHOTO'S OWN
BAKED SPECULAR: under the estimated +20 deg head turn the nose-ridge
highlight projects onto the left nose flank (photo lum 218 vs 188 lateral
surround, saturation 40 vs 57 — the bright+desaturated signature), and the
screened-Poisson composite faithfully preserves it (the column exists in
the pre-solve blend; the solve moves it by <= 5/255; membrane/rail count
inside it is ZERO — neither a membrane path nor a selection boundary).
`gradient_compositing.reconcile_specular_lobes` (new, applied inside
`composite_gradient_domain` before gradient selection) reconciles the
source view's smooth bright+desaturated lobes against the cross-view
diffuse consensus: another view's valid sample reading the same surface
darker beyond the pairwise lighting gauge authorizes the lobe as
view-dependent light; the correction rebuilds those texels from the
source's OWN surround tone plus its own log-detail (no reference color is
ever imported), with saturation restored toward the surround. Feature
protection: edge-dense components (sclera/teeth class) refused by an
own-photo Scharr gate NORMALIZED to the reference resolution (a
fixed-world edge halves its per-texel response at 2048, and the
uncorrected 1024-calibrated bar stopped refusing eye-adjacent
components); dark-material context excluded; a DARK-CONTENT STANDOFF
feathers the correction to zero near texels substantially darker than
the surround (leveling the bright base right against dark micro-content
unmasked it into the debris counter — measured 0.0040-0.0054 at five
views without the standoff, all green with it under a frozen
downstream); reference-view lobes deliberately out of scope (measured
-0.005 side identity for no ledger gain). Single-view bakes no-op
structurally (no second witness). Measured: the az0 4x column is gone at
1024 (identity cost -0.003 raw SSIM); at 2048 the mechanism IMPROVES
identity[front] raw +0.006 SSIM / -0.1 MAE and comp +0.005 / -0.2 —
sides unchanged, texture_qa PASS, ship/owl bit-identical.

### Added (pale-chip commit — the dark-context dual of the trace-deposit commit)

`texturing.commit_pale_chips`: isolated PALE islands in DARK material
context (the FACE-07 ear-band class) — skin/mixture content displaced into
hair at trace witness weight, plus completion texels that copied those
anchors (measured population at both ears: 35-60% fill) — are vacated and
retoned from their validated dark ring anchors when every qualifying
witness reads the blob's plain 3D ring uniformly dark (>= 96% dark votes,
cover/single-cover gates as the bright-context commit). Guards: confident
witnesses never touched (trace w50 <= 0.30); chips 2-connected to a big
bright component are frontier slivers of real material and refused; area
cap 1.2e-3 of direct texels (without it a 700-texel rear blob committed
into a visibly flat gray wash — measured); film-commit and rescue
territories excluded; >= 2 projections required (single-view ring
consensus is vacuous), so single-photo canaries are structurally
untouchable. Measured at 1024: ear-band chips visibly reduced at
az+-90/112.5 4x, detectors within noise, identity unchanged.

### Added (synthetic cut-face toning — bust disc tone from its own rim)

`texturing.tone_bottom_cap`: the truncated bust's planar cut face is
synthetic geometry no photo witnessed, yet the global harmonic fill toned
it with a tan/taupe marble fed by rear-hair and neck anchors (FACE-12's
disc wash). The cut face is detected geometrically (down-facing planar
component >= 0.5% of surface, direct witness < 1%, thin slab) and toned by
inverse-distance interpolation of its OWN RIM's observed content (chest
skin at the front rim, hair curtain at the rear), smoothed at 24 texels,
keeping 60% of the cap's log-detail. Multi-view bakes only this cycle
(single-photo proof assets are pinned regression canaries — same scoping
precedent as the strand comb).

### Added (film-band gradient repaint — hairline apron tone from the photos' own falloff)

The committed film band (cycle-2 mechanism) still rendered as a smooth
putty-taupe stripe at 2x from the declared pose: the mesh fuses the wispy
hairline into a smooth APRON tens of texels wide, every photo compresses
its narrow (4-10 px) wisp-transition ribbon across that whole apron (the
front view's bins sit at median 1 px from its dark body across the apron),
and the commit's retone covered only ~8% of the visible band with an
attenuated pull. `film_band_gradient.repaint_film_band` (new) rebuilds the
apron:

- geodesic profile field on the texel surface graph: two multi-source
  Dijkstra fields (photo-confirmed dark mass; photo-space skin ring) and
  the photos' own skin-side falloff profiles pooled into S(u) give a tone
  target that is near-black at the hair-mass boundary and blends into the
  local skin tone at the face edge — the photo's own gradient;
- source authority: apron texels the source view images first-surface at
  solid alpha take the source photo's color verbatim (real strand layout;
  statistical tone alone measured 0.60-0.62 identity vs 0.65-0.69 with
  content) under measured guards: base-material witness veto inside the
  feature moat (the parallax-doubled-brow / third-eye class), standoff
  from reference-confident and reference-dominant territory (side
  identity contracts; side worst-window 0.116 -> 0.031 without it, crown
  flakes at az-70/-90), outermost-sheet depth corridor (inner curtain
  sheets sprayed skin shreds at 1024), feathered domain borders (hard
  edges printed dark crease lines at az-35);
- gap diffusion + envelope clamp: unreachable apron texels take
  graph-diffused stamp colors under a field-consistency gate; remaining
  over-envelope texels clamp one-sidedly (darkening only — brows, lashes
  and all legitimately dark content untouchable by construction);
- island guards on the final state: small treated dark components with no
  pre-existing dark-observed anchor revert; bright shell components
  disconnected from both the skin ring and protected blobs pull to the
  envelope;
- repainted texels are exempt from the fill-luminance floor (they carry
  the photos' falloff, not fill statistics; the floor re-lifted darkened
  curtain texels into pale shreds, measured at 1024);
- sampling floor: the mechanism requires the hairline transition to span
  >= 7 texels (measured working point 9.6 at 2048, failing point 4.8 at
  1024 on the face proof); below it the cycle-2 retone remains.

Face proof asset (2048, full verdict1 battery): failures 2 -> 1
(identity[front] MAE gate now passes at 21.5/22.0; SSIM 0.630 -> 0.651
against the 0.70 bar), all 28-view detectors green, side identities keep
their margins. Single-view bakes are untouched by construction
(starship texture bit-identical, mechanism-on vs off). Tests:
`tests/test_film_band_gradient.py`.

### Added (trace-deposit commit — multi-witness consensus retone for chip/dash debris)

The residual chip/dash class at close zoom (FACE-03/04/05 family: beige
flakes and gray dashes under the eyes, mouth-corner smears, chin flakes,
strap slivers): small deposits of DISPLACED view content that win texels
at TRACE witness weight on surface every confident witness reads as
uniform bright skin. Measured populations on the face proof at 1024:
chip blobs carry winner weight w50 0.02-0.29 while legitimate features
(lash lines, nostrils, lip borders) sit at w50 0.44-0.93 — weight
separates the classes where color-deviation thresholds measurably cannot
(cycle-2 negative results: flake deviation p50 0.12-0.26 vs legit
front-eye trace texels at 0.399).

`commit_trace_deposits` (new in texturing.py) retones such deposits from
their own validated surround, blob-by-blob, under film-band-style commit
semantics — every gate carries a measured counterexample from this cycle:

- blob-level trace gates (w50/w90): content ANY view confidently
  witnesses is never demoted;
- multi-witness bright consensus on the deposit's plain 3D ring (a
  world-space ball — atlas dilation crosses UV charts and picked up hair
  texels that veto valid commits); single-witness consensus requires
  dominant ring coverage, zero-witness consensus refuses (vacuous);
- BRIGHT deposits near a confident strong-contrast core (per-texel
  confident witness at high |contrast| — lash lines, sclera, lip
  borders) are refused: ambiguous with the feature's own fringe;
  committing them measurably washed the eye corner (eye_count 2->1 at
  az0 el10, 1->0 at ±90). Ball-mean witness cannot serve as the core
  signal: the ball mean around a trace chip is lifted by its confident
  surround (chin dash: ball weight 0.42 vs own w50 0.047);
- isolation: a dark deposit connected to a larger dark component (hair
  frontier whose dark side is unwitnessed fill, lip line) is never
  committed — committing frontier slivers painted pale streaks into the
  hair mass and dropped profile eye_count at ±90 el10;
- WHOLE-NEIGHBORHOOD rule: a blob commits only if every sub-threshold
  residue island inside its ring (mid-gray dashes, chip shadow edges at
  lum 0.45-0.60 on 0.73-median skin) is itself sweepable under the same
  consensus; partial cleanup UNMASKS the residue as new isolated dark
  islands on the cleaned surround (measured: dark_debris 0.0022 ->
  0.0037 at az0 without the rule, identical to control 0.0022 with it);
- retone from the validated BRIGHT ring anchors only (inverse-square 3D
  interpolation): membrane refill drags adjacent feature darkness across
  the hole (measured dark_debris 0.0024 -> 0.0044 at az-22.5), and the
  ring's own deviation filter admits feature-dark texels near
  boundaries;
- placement: runs late (after mirror completion, rescue, film retone;
  before detail synthesis) as a strictly local recolor — committing at
  the outlier stage cascaded through the Poisson anchors, rescue-disc
  localization and fill calibration (whole-face render diff mean 4.1/255,
  14% of pixels > 8/255) and flipped knife-edge detectors far from any
  chip; rescue-disc footprints are protected from both detection and
  retone (an unprotected retone erased the rescued -90 profile eye).

Multi-view bakes only (>= 2 projections): with one witness the ring
consensus collapses to the winner's own photo, exactly the
`commit_film_band` vacuity argument. Face proof A/B at 1024 (same pinned
tree, chips on vs off): dark_debris IDENTICAL to control at every gated
view, eye counts identical, identity[front] SSIM 0.648 vs 0.649 with MAE
21.0 vs 21.1, 50 blobs + 21 residue islands retoned; visible chip subset
(cheek/chin flakes, mouth-corner pale chips, bust-rim slivers, curtain
stripes) cleaned at 4x. Single-photo bakes are untouched by construction
(measured bit-identical ship/owl textures with the stage enabled vs
disabled).

### Added (strand-comb fill regime — combed low-contrast statistics for fiber material)

The rear hair fill read as leopard mottle (FACE-09): the blotch lives in
BOTH the coarse value-noise octaves of the fill detail pass (rosette
scale) and the harmonic membrane's tone wash (measured at 1000 px
renders: blotch statistic 4.6 for the raw membrane, 6.4 after the
default detail pass — the noise ADDS rosettes on hair-class fill).
`synthesize_fill_detail` gains an opt-in strand regime
(`strand_comb=True`; per-texel: donor anisotropy >= 0.40 AND base darker
than 0.55x the observed bright-half median):

- orientation from a MULTIGRID-propagated global field
  (`_multigrid_orientation_field`: anisotropy-weighted structure-tensor
  anchors pooled into coarse surface voxels, tensor diffusion over the
  voxel k-NN graph with seeded cells re-anchored) — donor-local
  orientation is noise deep inside the fill domain (solver-4 G3's
  measurement, |cos| p50 0.999 after propagation);
- carrier keeps only the finest octave, combed with extended LIC (48
  steps): the coarse octaves ARE the rosettes, and fine carriers buy
  more gradient energy per contrast unit, so the closed-loop energy
  calibration lands at visibly LOWER contrast for the same fill-energy
  gate;
- the BASE fill tone is advected along the same field (sparse
  index-doubling kernel, strides 1..2^8 LIC steps) so membrane tone
  blotches elongate into strand-parallel streams, and transferred
  amplitude is scaled 0.6x (elongated LOW-contrast statistics, the bar
  the owl's rear grain set).

Measured on the face proof at 1024: rear blotch 6.4 -> 5.0 (az180) and
7.9 -> 7.4 (az-135) against the 4.6 membrane floor, with fill/observed
Scharr energy 0.93 (gate >= 0.5). Enabled for multi-view bakes; single
photo proof bakes keep the default path (empty strand regime is
bit-identical by construction and by test), preserving the ship/owl
regression canaries.

### Fixed (SHIP-03 nose melt — projector-frame photo registration for ortho bakes)

At head-on views (az 0..30, el -20..+15) the starship's prow rendered as
"melted" smeared streaks. Root cause (measured, not the suspected
grazing-stretch demotion): the canonical recenter centers the PHOTO's
alpha-bbox at the frame center, while the orthographic projector centers
the WORLD ORIGIN — and away from the canonical front pose those two
centers diverge by the mesh bbox's projected offset (starship at
az+30/el+15: +54/-28 px at 1024; face at az+20/el+8: +16/+8 px; owl at
az0: ~1 px). Every photo sample therefore landed tens of pixels off the
surface that imaged it; at the prow (surface turning away, high content
gradient at the silhouette) the offset dragged dark under-hull and
background-adjacent content onto the nose and stretched rim content
across the concavity. A perfect-content synthetic-checker probe at the
same witness geometry measured the ceiling: with the offset, projected
content decorrelates from ground truth even at nominal sampling stretch
(binary checker agreement ~0.5 = chance); with the registration fixed,
agreement 0.72/0.71 at stretch 1.25-1.5/3-4 — the witness geometry
itself was never the limit at moderate stretch.

- `projected_frame_center_px` (new): the pixel where the mesh's
  camera-plane bbox center lands under the projector's own convention —
  deterministic, no content-based search.
- `recenter_to_canonical_frame` gains `center_px`; the ortho bake path
  registers views to the projector frame for OVERRIDDEN source poses
  (external capture facts the model never consumed; references keep
  their content-based residual registration on top). ESTIMATED poses
  (gradient_ncc) keep the legacy frame: the estimator searched az/el for
  the best gradient alignment of the legacy-centered photo, so pose and
  frame are co-adapted — re-centering one side alone was measured worse
  on the face proof (verdict1 failures 2 -> 10, front SSIM
  0.630 -> 0.598); registering the estimator itself to the projector
  frame is future work that belongs to the face lane. At the canonical
  front the two conventions agree to ~1 px by construction.
- Bake stats and bundle metadata record `source_registration`
  (`mesh_bbox_center`, dx/dy px); `scripts/texture_qa.py` reconstructs
  per-view visibility from the same frame so region attribution stays
  faithful (absent key = legacy behavior, old bundles unaffected).
- `synthesize_fill_detail`: transferred amplitude is FLOORED at the
  observed population's p25 raw-residual amplitude (per channel).
  Grazing-smeared donors carry artificially quiet statistics; fill
  anchored by them shipped as literal flat plateaus with straight
  chart-edge boundaries (an 11k-texel flat cell tripped
  texel.facet_cellular 0.092 vs 0.091 at 2048 after the registration
  fix exposed it). With the floor: facet_cellular 0.012, fill energy
  0.615 -> 0.620, sigma guard and granite test untouched.

Starship A/B (same tree, only the registration): source-pose render vs
photo MAE 45.5 -> 18.1, SSIM 0.092 -> 0.600; az0 4x nose crops go from
molten streaks to readable intake/grill structure; `texture_qa` PASS
13/13 at 1024 AND 2048 (dark smears 0 at both). The prescribed
alternative lever — steepening the Jacobian stretch demotion into a
coverage vacate — was prototyped and measured NOT better: cutoff 2.0
cleared residual streak anchors but surrendered 52% of witnessed
coverage (src-pose MAE 24.3, SSIM 0.436) and at cutoff >= 3 the melt
stayed; the negative result and numbers are in the cycle report. Owl
(estimated/declined pose -> legacy frame): bit-identical code path,
PASS 13/13 at both resolutions. Face (estimated pose -> legacy frame):
bit-identical code path, texture_qa PASS 13/13 at both resolutions,
verdict1 failure set unchanged vs the tree baseline (2 identity
failures, both pre-existing).

### Added (film-band commitment — multi-view material consensus for fused film bands)

The temple/hairline "film band" defect class (beige painted sheet, black
parting flecks, skin-flake mottle interleaved with dark curls): generated
meshes fuse wispy hair films INTO the head as one surface, so the
layered-density zone gate cannot see them (no second sheet => layered
density 0.02-0.05 << the 0.10 gate) while the photo pixels there are
bright skin+hair mixtures whose stamps win texels and read as painted
sheet; surrendered/unobserved remainders inherit the harmonic membrane's
mixed skin+hair tone (pale curtain).

`film_band.py` (new) adds a multi-view MATERIAL COMMITMENT on top of the
zone gate, all scale-free and subject-agnostic:

- per view (computed in the projector alongside the zone gate): the
  strong zone grows into connected weak evidence — any layered density at
  contrast, near the photo's dark-material main body, with substantial
  dark coverage of the window's foreground (the foreground normalization
  keeps silhouette rims meaningful); small components are dropped
  (membrane handles local ambiguity). Each view also carries a base
  WITNESS VETO map (imaged bins with no zone flag and < 0.25 dark
  coverage witness base material along their ray).
- commitment (`commit_film_band`) requires: some view's large-component
  extension flags the texel first-surface, NO view vetoes it, EVERY view
  imaging it first-surface flags it (flag consensus — a fused wisp
  floater aligns with the dark body from one pose only; committing it
  detaches under parallax into a floating dark blob, the "third-eye"
  class), and at least two imaging witnesses (single-witness consensus is
  vacuous; measured painting dark spots at ear-rim/crown silhouette skin).
- commit-coupled surrender: at committed texels whose local observed
  context is dark-dominated (voxel-ball dark/bright claim ratio at two
  scales), BRIGHT mixture claims of every view are vacated; dark claims
  are film-consistent content and stay (vacating them paled the
  rear-quarter temple ribbons over the crown-flake gate, az-135
  0.0006 -> 0.0027). Where we cannot commit, baseline claims stay —
  surrender-without-commitment leaves the membrane anchored by whatever
  survives nearby (measured: lash-dark anchors bled through a vacated
  eyelid rim as a floating dash).
- film retone (`retone_film_band`, after `texel_surface_smooth`, before
  detail synthesis): committed fill takes its tone from dark-material
  OBSERVED anchors only (octant-binned voxel-ball means at growing
  scales), scaled by photo wispiness and the same dark-dominance factor;
  mirror destinations inside the commit are removed; zero-weight rim
  coverage inside the film zone is demoted to fill
  (`demote_unwitnessed_rim`).

Face proof A/B (same tip, verdict1 harness): @1024 failed checks 4 -> 1
(the pre-existing front-identity SSIM; az-135 crown flakes x2 and az-45
dark debris all cleared), @2048 2 -> 2 (both pre-existing front-identity;
mean|RGB| 22.3 -> 22.2), az+22.5 el0 eye_count 1 -> 2 (correct), no
3-blob eye failures at any azimuth; hairline crops at az 0/±22.5 show
hair-toned fill where the beige sheet/pale curtain was (temple beige
remnants that survive are kept mixture claims under the witness veto —
committing them was measured strictly worse). `scripts/texture_qa.py`
PASS at both resolutions. Single-view bakes (starship/owl) are
bit-identical with the mechanism present vs disabled (md5-verified);
`commit_film_band` no-ops below two views by construction.

### Added (mirror twin rescue — general weak-twin feature transplant in the bake)

Mirror completion only writes UNOBSERVED texels, but on near-symmetric
subjects a feature region can be observed yet badly witnessed: every
covering view sees it at grazing incidence or through a misregistered
duplicate reference, so the texels carry a smear that no per-texel gate
downstream can repair (all covering witnesses agree on the wrong
content). Measured on the face proof at az -90: eye-disc ball witness
weight 0.16 vs the healthy twin's 0.55, harness `eye_count` 0, and the
resulting broken eye dragged the side_right identity registration 1.3%
off — the -0.132 worst-window "ghost" at the ear was a registration
artifact of the broken eye, not ear-texel damage (forcing the corrected
registration onto the unfixed render scores that window at +0.47).

`detect_mirror_rescue_discs` (new) finds such regions generally, with no
feature-class knowledge: strong-side discs that are confidently
witnessed, locally contrastful, and carry a coherent dark core, whose
mirror twin is observed but >= 2x weaker witnessed AND feature-empty
(pointwise blob response <= 0.5x the core's). Detected discs drive the
existing `mirror_rescue_disc` transplant (tone-matched, feathered) inside
`bake_projection_texture`, after mirror completion, under the same
geometry-symmetry gate (score >= 0.55); transplanted texels count as
completion, not photo truth. Gates that keep legitimately asymmetric
content untouched, each with a regression test: content well-witnessed
on both sides never triggers (twin-weight ratio); unobserved twins belong
to mirror completion (twin-coverage); a twin with its own comparable
structure is left alone (feature-emptiness, sampled pointwise because
ball averages dilute edge responses); discs straddling the symmetry
plane are refused entirely (a half-transplant guarantees a mid-feature
seam — measured painting a black dash on the face lane's front lips).
Two placement/tone refinements, both measured load-bearing: the
transplant is anchored along the mirror axis on the twin's own
evidence-weighted feature-dark centroid (capped at 0.4x the feature
radius — the pure geometric mirror position pulled the source-pose
identity registration 1.3% and its SSIM 0.632 -> 0.601; in-plane anchor
components are noise and re-rolled a bistable registration, so only the
axis component is used), and the tone-matching ring averages only
source-mask texels (in-bake the annulus contains not-yet-filled texels
whose zeros biased the offset ~0.02 dark, pushing transplanted skin
flecks across the dark-debris gate). Face proof A/B at 2048 (same tip,
verdict1 harness): failed checks 8 -> 2 (both remaining are the
pre-existing front-identity SSIM/MAE, 0.632/22.1 -> 0.629/22.3); az -90
eye_count 0/0 -> 1/1 at both elevations; identity[side_right] worst
window -0.132 -> +0.219 (gate 0.05) with SSIM 0.657 -> 0.682;
dark_debris at az -22.5/-35 all under the 0.003 gate (was
0.0030-0.0035); scripts/texture_qa.py stays PASS 13/13. Single-photo
bakes (starship/owl) fire zero discs (geometry scores 0.98 but the twin
side is unobserved) and their textures are bit-identical with the
detector disabled (md5-verified on the starship).

### Documented (identity-gate shading floor — measurement bias, not albedo signal)

Photo-vs-render identity metrics carry a perfect-texture penalty from the
preview renderer's own shading (`shade = 0.88 + 0.12*diffuse`): measured
SSIM 0.977 / mean|RGB| 11.45 for a PERFECT texture at the face lane's
declared pose. The term is texture-independent, so the correction belongs
in the measurement (photo multiplied by the white-texture shade field,
MAE budget re-tightened by the removed floor), never in the texture.
Full calibration data and the proposed harness patch:
`/tmp/c2d/REPORT.md` + `docs/KnowledgeBase.md` ("Identity gates that
compare shaded renders to photos carry a perfect-texture floor").
No pipeline code changed by this analysis lane.

### Fixed (fill-character restoration — closed-loop energy calibration in `synthesize_fill_detail`)

The fill-detail synthesis transferred observed log-residual SIGMA to the
fill, but the quality bar (`texture_qa` `texel.fill_gradient_energy_ratio`,
gate >= 0.5) judges LINEAR-luminance gradient energy — an open-loop proxy
that systematically undershoots. Measured decomposition on the starship
proof at 1024 (fill/observed energy 0.43 at gain 0.7, gate FAIL): donor
amplitude transfer 0.84x (color-similarity weighting favors donors
darker/quieter than the observed median), carrier frequency 0.69x (the
3-texel finest noise octave carries less per-sigma gradient than photo
micro-texture at 1-3 texels), base luminance 0.79x (multiplicative
log-detail on a darker fill base yields proportionally less linear
gradient). Two changes, both resolution-invariant:

- Finest carrier octave moved to ~2 texels (`wavelength_texels` 3 -> 2,
  `octaves` 2 -> 3, band now 2..8 texels): restores per-sigma spectral
  energy at every resolution.
- CLOSED-LOOP CALIBRATION: the pass provisionally applies the detail
  (clip + seam ramp included), measures the realized fill gradient energy
  with the same Scharr operator the QA uses, and solves (secant, 2-3
  evaluations) one global scale that lands the fill at `gain` x the
  observed energy. Bounds: never below 1 (already-rich fills — face hair
  streaks — are never dampened), never above `energy_calibration_max`
  (3.0), and never past a sigma guard that caps the fill's log-sigma at
  the observed population's band-matched residual sigma — gradient parity
  may not be bought with granite on edge-dominated subjects; any shortfall
  is reported in the bake stats (`fill_detail.energy_calibration`), not
  hidden.

Measured (fresh single-view bakes, current tree, `texture_qa`):
starship fill energy 0.39 -> 0.58 (1024) / 0.50 -> 0.63 (2048), owl
0.43 -> 0.58 (1024) / 0.60 -> 0.69 (2048), dark smears 0 and facet fields
0 at 4x throughout, seams within allowance; face (multi-view) stays PASS
with fill energy 1.06 -> 1.16 (its calibration correctly resolves to
scale 1.0 — the sigma guard binds). Tests:
`test_synthesize_fill_detail_energy_calibration_reaches_gate`,
`test_synthesize_fill_detail_calibration_never_injects_granite`.

### Fixed (texture QA photo reference — matte the photo like the bake does)

`scripts/texture_qa.py` derived every photo-side reference (viewer-truth
brightness, seam allowance, photo calibration, and the front view's
visibility alpha) from a "non-white" heuristic on RGB inputs. On unmatted
photos with non-white backdrops the heuristic measures the BACKGROUND:
on the owl proof photo (light-gray studio backdrop, ~205 median
luminance) it classified 100% of the frame as foreground, inflating the
brightness reference to 203 vs the subject's true 129 and failing
`viewer.brightness_ratio` at 0.567 on bakes whose subject tone was in
range — the gate measured backdrop bias, not albedo fidelity. The harness
now mattes RGB photos with the same `remove_background_robust` the bake
pipeline itself applies before projecting (RGBA photos keep their alpha;
if the matte model is unavailable or degenerate the old heuristic remains
as an explicit `heuristic_nonwhite` fallback recorded in results.json).
Same-bundle deltas (current-tip bakes): owl brightness 0.567 -> 0.891
(2048) / 0.562 -> 0.884 (1024), ship 0.752 -> 0.845, face 0.960
(unchanged; its photo is near-fully non-white so the heuristic was
accidentally right). The front-view visibility reconstruction also stops
counting background-ray texels as observed (ship coverage reconciliation
qa 0.261 vs bake 0.177 -> qa 0.211).

### Added (dense residual reference registration — strictly-local validated lattice flow)

Global similarity registration (width-profile matching + overlap similarity
search) cannot satisfy per-feature displacements on generated geometry: the
nose, mouth and eyes each want a DIFFERENT small 2D correction (measured on
the face lane: nose −10 px, mouth (−4,+4), eyes (+4,0) at 512), so
reference photos paint ghost lip/lash fragments next to the source's
features. New module `abstract3d/reference_flow.py`, wired into
`bake_projection_texture` directly after `register_reference_by_source_overlap`
(orthographic multi-view references only; single-view bakes verified
bit-identical):

- Energy: Charbonnier photometric residual of the gain-corrected reference
  against the SOURCE'S PAINTED TRUTH splatted into the reference's image
  plane through the shared surface (first-surface visibility, source
  confidence x reference-facing evidence weighting), regularized by a
  bending (thin-plate) energy on a coarse-to-fine control lattice
  (64/32/16 px), Gauss-Newton with a 2%-of-frame displacement cap. The
  photo is warped exactly once (flow upsampled to the native canvas).
- Safety architecture, each clause anchored to a measured failure: per-cell
  validation (>= 20% weighted-L1 improvement AND absolute post-warp error
  within 1.25x the median of improving cells), a one-ring evidence leash
  (adjacent cells keep flow only with substantive non-worsening own
  evidence), reference-facing evidence gating, and strictly-zero
  displacement everywhere else. Global extension of band-fit corrections
  was measured harmful twice (a residual affine collapsed side identity
  0.706 -> 0.587; even a pure translation moved the hair mass and tripped
  skin_in_hair at az +-135) — hence STRICTLY LOCAL.
- Validation: injected known warps (shift / rotation / barrel / local bump
  / combined) recovered to <= 0.7 px median inside the evidence band at the
  512 solve scale; acceptance additionally gated on a >= 2% overlap-error
  improvement with >= 3 validated cells, else the input photo is returned
  untouched.

Measured (face 3-view lane, same-tip A/B off -> on): 1024 harness failures
12 -> 10 (front identity SSIM/MAE/worst-window and side_left worst-window
failures cleared); 2048 failures 8 -> 8 with front SSIM +0.006 / MAE -0.8
and two dark_debris lines swapped at the 0.003 gate; ghost-lip fragments at
the mouth visibly reduced at both resolutions (crops in the cycle report).
Starship single-view lane: bit-identical texture with the stage on/off.
Tests: `tests/test_reference_flow.py` (injected-warp recovery, strict
locality, unreachable-content rejection, no-overlap identity).

### Added (photometric delighting of reference views — SH-in-normal-space shading removal)

Photos carry their own lighting, and two registered photos of the same
subject disagree on every shared surface point as a smooth function of the
surface NORMAL (each light shades each orientation differently). That
disagreement survived exposure harmonization (a scalar gain cannot express
a normal-dependent field), leaked into view-handoff tone steps, and gets
doubled by any viewer relight. New `texturing.delight_projections`, run on
the atlas projections before harmonization/gating:

- Model: Lambertian formation I_v = A * S_v(n); on OVERLAP texels the
  log-luminance ratio log Y_u - log Y_v = B(n) . (c_u - c_v) cancels the
  albedo EXACTLY. B is the order-2 real SH basis in the normal
  (Ramamoorthi & Hanrahan: >99% of distant-light irradiance energy), so
  genuine albedo detail — high-frequency in normal space — is outside the
  model span by construction.
- Estimation: joint weighted ridge LS over all overlapping view pairs
  with gauge c_source = 0 (references are relit to the SOURCE's light; the
  common lighting component is unobservable from ratios, and the source
  photo is the identity anchor everywhere else in the pipeline). Huber
  IRLS with MAD-adaptive threshold rejects content outliers
  (misregistration hair-over-skin) without rejecting legitimately strong
  shading ratios; the fitted field is clipped to the overlap's own
  [p1, p99] +- 0.1 (exclusive-region normals the fit never saw cannot
  receive extrapolated inventions) and capped at |log| <= 1.
- Application: luminance-only (chroma untouched), multiplied into the
  reference's covered texels; the existing per-channel exposure gain then
  handles only residual white balance (measured on the face lane: gains
  drop to ~1.02 after delighting).
- Overlap-proximity fade: the correction applies fully near the overlap
  surface (where seams form) and fades to zero deep inside the
  reference's EXCLUSIVE territory, where that photo is the only witness
  and per-view identity outranks consistency with a light no camera sees
  from there. An adversarial bisect measured the unfaded version
  relighting a profile's whole exclusive side (identity MAE vs its own
  photo 26.4 -> 39.5) and disabled the stage; the fade keeps the handoff
  fix with exclusive-side drift measured at 0.002 mean|RGB| (stats row
  `exclusive_mean_abs_delta`), and the stage is re-enabled.
- Revert-on-confound: kept per reference only when that reference's
  overlap mean|RGB| disagreement against the source DROPS by > 0.002 —
  the same statistic family as the exposure gate it generalizes (the DC
  gain is this model's order-0 term).

Measured (face 3-view lane, 1024): side_right overlap disagreement
0.085 -> 0.063 (-26%) with the correction kept; side_left reverts (its
overlap disagreement is content mismatch, exactly the confound the gate
exists for); synthetic two-light sphere proof x3.2 disagreement drop
capped (x70 uncapped), recovered albedos agree on overlap. Tests:
`test_delight_projections_recovers_agreeing_albedo_on_two_light_sphere`,
`test_delight_projections_fade_protects_exclusive_territory`,
`test_delight_projections_keeps_chroma_and_reverts_on_confound`.

### Added (geometric witness confidence — sampling-stretch and concavity terms in the projector)

Projection weight was `alpha * facing^2 * witness_factor`; facing measures
LOCAL TILT only, and the eye-socket class of defect rides through it: a
socket wall can face the camera acceptably while the composed texel->photo
mapping collapses, so one photo pixel smears down the whole wall.
`_tripo_projection_geometry_confidence` adds two exact terms:

- STRETCH: the texel->photo Jacobian J = [ds/dcol, ds/drow] by finite
  differences of the projector's own sample maps (exact for both camera
  models; chart-boundary pairs masked). sigma_min = smallest singular
  value = worst-direction sampling pitch; stretch = nominal / sigma_min
  with nominal = median sigma_min over well-facing texels (facing > 0.7).
  The nominal makes the statistic invariant to photo/atlas resolution AND
  to legitimate chart anisotropy (normalizing by sigma_max instead was
  built first and measurably mis-scored healthy texels on anisotropic
  charts — cylinder test). Weight *= 1/(1 + max(stretch-1, 0))^p with
  p = 2, measured by sweep on the face proof (adversarial harness, all
  else fixed): p=0 13 failures, p=1 13 (sub-threshold dark-debris
  improvements only), p=2 8 — three dark-debris views cleared, the az -70
  eye recovered, front identity MAE fail cleared — with texture_qa fully
  green and single-view assets within noise of p=1.
- CONCAVITY: mean curvature from the normal-field divergence over the
  surface (div n ~ (dn . dp)/|dp|^2 along both atlas axes), normalized to
  concavity = -0.5 div n * (0.02 * diagonal). Texels BOTH concave
  (> 0.35) and grazing (facing < 0.5) multiply by 0.25: concave interiors
  catch stretched/misplaced content exactly where the witness is weakest,
  while a well-facing concave eye keeps its claim (legitimate socket
  content and shading survive).

Per-projection stats key `geometry_confidence`. Synthetic proofs: rim
collapse demoted on an anisotropic-chart cylinder (front factor 1.0, rim
< 0.35); sharp-trench demotion strictly inside the concave interior with
a convex-ridge control undemoted. Tests:
`test_projection_geometry_confidence_stretch_demotes_collapsed_mapping`,
`test_projection_geometry_confidence_demotes_concave_grazing_only`.

### Added (synthesized-texel luminance floor — zero dark fill fragments at close zoom)

Provenance audit of the close-zoom "spurious dark fragment" failures on
the single-view proofs (starship 4, owl 6 at 4x): dark observed anchors
(intake interiors, panel shadows, occasional background-adjacent grazing
samples) seed the harmonic fill, whose maximum-principle solve freely
TRANSPORTS that darkness across hidden surface (measured: fill blobs at
luminance 14-26 whose nearest observed anchors sit at 61-114). Observed
texels carry photo evidence; fill texels carry none, so a context floor
is a legitimate prior. New `texturing.enforce_fill_luminance_floor`, run
as the bake's last color pass over FILL texels only (observed AND
mirror-completed texels bit-identical by construction — mirror copies
carry their twin's evidence, and an adversarial pupil-analog test showed
a local floor cannot tell a mirrored pupil from a defect):

- context floor: plain ball mean m1 at two world scales (R, 2R;
  R = 0.035 x diagonal) with a dark-minority gate per scale (smoothstep
  to zero as the ball's dark fraction passes 0.30 -> 0.45): a defect
  pocket is by definition a local anomaly, so regional darkness (hair
  mass, shaded hull side, hairline shadow bands) stands the floor down —
  an ungated floor measurably turned the face's hairline band into a
  pale film (verdict-harness pale_film 0.0055-0.0062 > 0.005 gate);
- donor-consensus floor ("donor validation"): the same ball statistics
  over direct-observed donors only, catching fill far darker than every
  donor around it;
- sheet-awareness: every ball statistic bins by dominant normal-axis
  direction (6 bins) and each texel reads every bin EXCEPT the opposite
  one — a Euclidean ball on thin-crust meshes otherwise judges a shaded
  underside against the sunlit topside millimeters away through the
  shell (measured: sheet-blind floor dropped the starship
  fill-gradient-energy gate 0.57 -> 0.48; a Hamming<=1 octant pooling
  variant let the face's rear hair read forward skin as context — the
  critic-measured "skin patches in rear hair" failure);
- dark-evidence exemption: a connected dark component (3D voxel
  connectivity at max(2 texel pitches, 0.003 x diagonal) — the WORLD
  floor prevents resolution-dependent fragmentation measured as an
  owl-only-at-2048 seam spike) containing >= 8 observed texels whose
  tone its fill TRACKS (fill mean <= 1.35 x evidence mean) is a
  witnessed feature continued into hidden surface (the owl's wing
  markings) and keeps its own tone; components failing the tracking
  test (starship engine-halo smears at 2-3.7x their cavity tone) get the
  full floor. Without this, lifting a legitimate marking manufactured a
  fresh observed|fill tone seam (owl p95 29 -> 52..60, gate 52.2);
- application: saturating per-pixel depth compression in log-luminance
  (remaining depth = residual_depth * (1 - exp(-d/residual_depth))):
  monotone (no posterization at the floor line), slope 1 at zero depth
  (no visible boundary), bounded remaining depth 0.10. A base/residual
  split was built first and measurably leaked dark bands wider than the
  base radius but narrower than the context ball (owl crease bands);
  compactness restrictions and boundary feathers were tried and
  reverted — they left pocket edges under the floor, re-detected as
  fresh smaller fragments. Pixels deeper than 1.2 below the floor blend
  toward the context consensus color (bright-half mean scaled to
  target: near-black 8-bit pixels carry no usable chroma to multiply);
- floor_ratio 0.65 vs the detector's 0.45: deliberate headroom because
  the render-window reference mixes brighter cross-sheet content than
  the sheet-aware ball (measured up to ~1.2x on the owl's wing creases).

Applied to the SHIPPED ledger bundles as a texture post-process (masks
reconstructed the same way the QA harness does): starship 4 -> 0 and owl
6 -> 0 dark fragments at 4x, with seam, facet-cellular, and
fill-gradient-energy gates all remaining green (ship fill energy 0.57 ->
0.55, fill Scharr edge energy +0.4% — hull panel lines survive). Fresh
bakes at 1024 and 2048 on all three proof assets: zero dark fragments
across every bundle. Stats key `fill_floor`. Tests:
`test_enforce_fill_luminance_floor_lifts_pockets_keeps_lines_and_dark_regions`,
`test_enforce_fill_luminance_floor_spares_mirror_features_and_opposite_sheets`,
`test_enforce_fill_luminance_floor_donor_anchor_catches_transported_darkness`.

### Added (mirror-consistency disc rescue for weakly-witnessed features)

`texturing.mirror_rescue_disc(colors_rgb, positions_texture=..., center=...,
radius=...)`: replaces a world-space feature disc's texels with their mirror
twins' content, tone-matched to the destination's surrounding annulus and
feathered at the edge. Complements `mirror_fill_from_observed`, which only
writes UNOBSERVED texels: a feature region can be observed yet badly
witnessed (all views at grazing incidence, or a mirrored duplicate reference
landing misregistered), leaving displaced content that no downstream gate
can repair because every covering view agrees on the wrong pixels. Measured
root cause on the face lane's right eye at az -90 (the "eye_count 0 at -90"
QA failure): the eye disc's best witnesses average blend weight 0.14 vs the
mirror twin's 0.50, and the painted iris band sits ~0.04 mesh units below
its mirror-correct position with a second stray band at the brow — the
detector sees two thin high-aspect fragments instead of one eye. Applying
the rescue to the frozen v14 ship candidate (disc centered on the twin of
the PASSING left eye, found by the QA detector itself) cleared all four
profile eye_count failures (az +/-90 x el 0/10) and improved the right
profile's worst 49-px identity window from 0.118 to 0.194; harness failures
6 -> 5 (the remaining trade: the az -45 el 10 eye blob fragments under the
transplanted specular highlights and undercounts, while visually reading as
a more structured eye). Geometry ceiling DISPROVEN for this defect: the
mesh's squinted lid aperture (0.20-0.29 vs the photo's 0.43) still renders
a machine-detectable and human-readable eye when correct content lands on
it — the defect was texture placement, not geometry. The function performs
only the geometry-driven transplant; callers decide WHERE (QA localization,
witness-quality maps). Test:
`test_mirror_rescue_disc_transplants_twin_feature_tone_matched` (synthetic
folded plane: twin feature transplants, tone offset cancelled, locality and
untouched-twin guarantees).

### Added (gradient-domain view compositing — one screened-Poisson composite replaces the seam-patch stack)

Multi-view composition previously fought tone seams with a stack of local
patches (softmax color blend, per-region seam-leveling offsets), and the
residuals were still visible: a mouth-crossing chroma seam at az 0, cheek
tone patchwork at close zoom, and dark-debris marginals at the
22.5–45-degree views. Root cause: every patch operated in the COLOR domain,
where exposure disagreement between witnesses is indistinguishable from
content. New module `gradient_compositing` composites the views in the
GRADIENT domain instead and solves one screened Poisson system over the
observed texel surface graph, running between outlier filtering and
completion so mirror/fill propagate equalized colors (the same relative
order the legacy path gives its leveling offsets). Energy: per-edge match
to a composited target gradient plus confidence-weighted soft anchors to
the blended colors; SPD normal equations; multigrid-preconditioned CG,
float32, deterministic.

- Graph: UV-grid 4-neighborhood within charts (3D jump guard) plus KD
  chart-stitch edges, with a normal-agreement gate so thin-shell sheets
  never stitch; the atlas solves as ONE closed surface (chart cuts proven
  invisible on a two-chart synthetic sphere, and broken when stitching is
  disabled).
- Target gradients: most confident common witness per edge (photo edges
  survive verbatim — no cross-view averaging); one-sided witnesses at
  winner-take-all handoffs; zero-gradient membrane only where no view
  sampled both endpoints.
- Witness-less (line) edges carry two measured safeguards: their weight
  scales with reference/resolution (boundary edges appear once per
  crossing row, so their energy per world length otherwise doubles per
  resolution octave — the mid-face chroma seam eliminated at 1024
  reappeared at 2048 through exactly this), and a material gate (the
  screened-Poisson analog of seam leveling's `boundary_cap`) releases
  edges whose color step exceeds ~0.18 so hair|skin and ear-fold borders
  are never tinted toward each other.
- Screening: lambda proportional to blend confidence with a 0.1 floor
  (rim texels otherwise drift freely and wash fragmented coverage),
  source-view claims boosted 4x above weight 0.4 (the photo's identity
  contract), completion left to inherit via fill, all rescaled by
  resolution^-2 so the equalization decay length is fixed in world units.
- Solver: geometric-aggregation multigrid V(1,1)-cycle as CG
  preconditioner (voxel-clustered levels, Galerkin operators, damped
  Jacobi, coarsest-level splu); float32 iteration with float64 scalar
  accumulation; converges in 21-141 iterations at ~1e-5 relative residual
  where plain Jacobi-CG needed ~1000; full compositor 1.8 s (face 1024) to
  9.2 s (face 2048) on this host.

Synthetic ground truth (two-chart sphere, two views, one corrupted):
additive exposure offset recovered exactly up to one global constant
(RMS < 2e-3), handoff discontinuity killed >20x, checkerboard edges keep
>= 95% contrast across the handoff, gain+vignette handled with a smooth
sub-visibility error field. `bake_projection_texture` gains
`compositing="auto"|"legacy"|"gradient_domain"` (auto = gradient_domain
for multi-view bakes, legacy for single-view where the solve measurably
only jitters threshold-marginal detectors; legacy remains selectable).
A/B on the proof assets (both QA harnesses, both resolutions, single
frozen tree, alignment-controlled identity): face adversarial failures
11 -> 10 (1024) and 8 -> 8 with strictly better defect magnitudes (2048:
top dark-debris 0.0045 -> 0.0033), dark-debris failing views 5 -> 1 at
1024; identity SSIM at controlled alignment equal or better on front and
side_left under either warp (front 0.619 -> 0.637, side_left
0.704 -> 0.707 under the baseline warp; the raw harness numbers are
warp-landing-confounded, verified at pixel level); texture_qa face and
ship PASS at parity or better medians.
Tests: `tests/test_gradient_compositing.py` (graph stitching/guards,
gradient selection rules incl. line classification, screened-Poisson
recovery/anchoring, end-to-end bake, determinism).

### Fixed (texture-cycle integration — pose stability, single-view outliers, acceptance harness)

- Hardened `estimate_pose_photometric` against two measured failure modes. First, the
  scorer compared photo and render gradients in mismatched frames: compact heads
  tolerated it, but an elongated subject's projected aspect swings with elevation and the
  correlation degraded into noise — the starship's true pose (az +30, el +15) was
  unrecoverable, and el +15 was not even in the elevation grid. Renders are now aligned
  into the photo's frame with crop-immune anchors (subject top row, silhouette centroid,
  mean width over common rows — full-bbox recentering was tested and rejected because it
  breaks on cropped photos), and the elevation grid spans +/-15 with local refinement on
  both axes. Ship observed coverage: 0.062 -> 0.20. Second, on bilaterally symmetric
  meshes the mirror pose (+az vs -az) is a structural near-tie for gradient-magnitude
  content, and 0.1% vertex jitter could flip the azimuth SIGN (an adversarial QA rebake
  drew a sign-flipped pose and produced a 65-failure bake). A chirality tie-break now
  correlates the horizontal ANTI-SYMMETRIC luminance components of photo and render —
  sign-opposite between mirror poses — and decides the sign with a margin symmetric
  content cannot dilute. Face pose is stable at +12.5..+20 across jitter trials.
- The surface outlier filter now also runs for single-view bakes: the foreign-view
  condition is vacuous with one witness, but the same-view color-extreme condition
  catches dark background-adjacent rim misprojections (measured on the starship: the
  surviving dark fragments at 4x zoom are exactly this class; 854 texels dropped).
- Regenerated the face-2mv, hunyuan-starship, and hunyuan-owl proof bundles end-to-end on
  the integrated pipeline. The face bundle passes all `scripts/texture_qa.py` gates
  (materials, viewer-truth brightness, fill character, facets, seams, close-zoom probes);
  the starship passes 12/13 (residual: 4 small dark fill fragments at 4x zoom across all
  probe crops, documented in ADR 0009); the owl's remaining brightness-gate failure is a
  harness calibration artifact (its input photo is unmatted white background, inflating
  the photo-side reference).
- Fixed `python -m abstract3d.cli` silently exiting 0 without running anything (missing
  `__main__` guard); the `abstract3d` console script was unaffected.

### Fixed (fifth adversarial cycle — observed-region close-range defects)

Owner-visible complaints at close zoom on the multi-view face: a vertical
tone seam down the nose/philtrum where front-photo content meets profile
content, and dark copy fragments on cheeks near the hairline. Provenance was
instrumented per texel (per-view weight/winner maps rendered from the bake's
own captures) before fixing; both fixes are general-purpose (no face logic):

- Added `level_composed_seams` (wired into multi-view bakes between mirror
  completion and fill): Ivanov/Lempitsky-style seam leveling on the mesh
  graph. Per-texel region = winning view or mirror fill; one additive
  low-frequency offset field per region cancels tone steps at region
  boundaries while high-frequency content is preserved. Two safeguards are
  load-bearing and covered by tests: boundary edges whose color step exceeds
  `boundary_cap` are genuine material borders (hair|skin) and are excluded
  (an uncapped solve tinted the ear region and dropped the right-profile
  identity's worst face window from +0.10 to -0.13 SSIM), and vertices whose
  winning witness is confident are pinned toward zero correction, so
  leveling only recolors the weak/contested bands between confident zones
  (each photo stays ground truth on surface it saw well). Face-2mv at 1024:
  adversarial harness failures 16 -> 14 (mid-face chroma-seam failures
  3 -> 1), identity SSIM front 0.614 -> 0.619, side_left 0.612 -> 0.703,
  side_right 0.688 -> 0.690; single-view bakes are structurally unaffected
  (verified bit-identical on the starship lane).
- Added a consensus guard to `mirror_fill_from_observed`: geometry is never
  perfectly symmetric (0.966 on the face), so twin lookups near material
  boundaries could copy hairline hair onto cheek skin (measured: half the
  dark defect pixels on the left cheek at close zoom were such copies). A
  copy is rejected only when the destination's observed 3D neighborhood is
  color-consistent AND the copy contradicts it; feature-rich destinations
  accept copies unchanged, rejected texels fall to the harmonic fill.
  Verified inert where twins are legitimate (starship mirror completion:
  guard on/off textures identical).

### Fixed (fourth adversarial cycle — hidden-surface fill quality)

Owner-visible complaint: texture on surface NOT visible in the input photos
read as flat "painted" mush (rear head, ship hull), with faceted flat-color
polygon blocks at close zoom and patchwork mottle on large single-view fills.
Root causes measured and fixed (metrics on the face-2mv and hunyuan-starship
proof assets at 1024, defect views + 4x zoom crops):

- Fixed the facet blocks: `mesh_graph_harmonic_fill` assigned every unseen
  texel its nearest VERTEX's solved color, so each vertex's Voronoi cell
  rendered as one flat polygon (~59k vertices serving ~4.2M texels at 2048).
  Texels now blend the 3 nearest vertices with inverse-distance weights
  (fill-region flat-plateau fraction 0.45 -> 0.21 on the face).
- Added `texel_surface_smooth`: a Jacobi relaxation of fill texels over the
  k-nearest-neighbor graph of texel 3D positions (observed texels fixed as
  Dirichlet anchors, normal-agreement-weighted edges). Runs after BOTH fill
  paths; it removes the residual vertex-cell seams of the harmonic fill and
  the donor-set patchwork of the KD fallback (fill Laplacian energy down
  30x / 8x respectively). Zoom crops show smooth material instead of blocks.
- Added `synthesize_fill_detail`: propagated fill is the correct average
  color but has ZERO micro-texture, which is exactly the "painted wash". The
  pass transfers observed texture STATISTICS — robust (L1) local residual
  amplitude and structure-tensor streak orientation, per material via
  normal + base-color matched surface-nearest donors — carried by
  deterministic multi-octave 3D value noise (seamless across UV charts),
  smeared along the transferred orientation (LIC) in proportion to donor
  anisotropy, applied multiplicatively in log domain, amplitude-capped at
  the observed p90 and feathered at the observed seam. Fill/observed
  local-variance ratio: 0.53 -> 0.83 (face rear hair, reads as combed
  streaks), 0.15 -> 0.59 (ship hull grain). Copying observed residual
  STRUCTURE (shift-map quilting) was prototyped and measured worse (chaotic
  panel fragments); statistics transfer makes hidden surface read as the
  same material, not the same content — a documented, honest limit.
  `bake_projection_texture(fill_detail_gain=...)` scales it; 0 disables.
- Added `texture_completion="auto"` (bake + both backends + CLI): apply
  mirror completion iff the mesh's own left-right symmetry score passes the
  existing >= 0.55 gate. The Hunyuan backend now defaults to "auto" — a
  single photo of the starship observes 6-9% of its texels while the
  geometry is 0.98 symmetric; the mirrored twin of the observed sliver is
  real panel content where any propagated fill is wash. Explicit "none" /
  "mirror_symmetry" behave exactly as before; `stats["texture_completion"]`
  reports the resolved mode and `texture_completion_requested` the request.
- Adversarial face-gate harness (28 views x 8 detectors + identity):
  failures did not increase (7 -> 6 at 1024, 6 -> 5 at 2048 on identical
  captures; the remaining failures are projection/pose classes untouched
  by the fill stage). All unit tests pass; fill passes are deterministic
  (fixed seed, hash-based noise).

### Fixed (third adversarial cycle — textured-export material factors)

- Fixed textured GLB/OBJ exports darkening ~60% and rendering as metal in
  spec-compliant viewers: the bake assembler built its `TextureVisuals` with a
  default trimesh `SimpleMaterial`, whose 0.4 gray diffuse became
  `baseColorFactor [0.4, 0.4, 0.4, 1.0]` on GLB export while `metallicFactor`
  was omitted entirely (the glTF default is 1.0, fully metallic). Baked-texture
  meshes now carry an explicit `PBRMaterial` (white base color, metallic 0.0,
  roughness 1.0) so the baked albedo is the authored surface color in any
  viewer. Affects TripoSR and Hunyuan3D textured exports; geometry-only
  exports are unchanged.
- Fixed the OBJ sidecar MTL carrying `Ka/Kd/Ks 0.4`: the OBJ exporter now maps
  the mesh's PBR factors onto explicit Phong constants (`Ka/Kd 1.0`, `Ks 0.0`
  for non-metallic baked albedo — a non-zero Ks would add a synthetic sheen on
  top of photo-derived colors in Phong viewers).
- Preview renders now multiply the sampled texture by the material's base
  color factor (both the ModernGL and matplotlib paths), matching what
  spec-compliant viewers show; previously previews sampled the raw texture and
  masked the darkening defect entirely.
- Added `scripts/check_export_materials.py`, a reusable GLB/MTL material
  factor auditor (`--strict` for CI-style gating), and
  `tests/test_export_materials.py` regression tests that export through the
  real helpers and assert the GLB JSON factors and MTL lines.
- Repaired the shipped `final-proof/hunyuan-starship` and
  `iter3-multiview-fixed/face-2mv` bundles in place (scene.glb materials JSON
  and scene.mtl only; geometry and texture bytes verified bit-identical).

### Added (second adversarial cycle — six-agent audit of the multi-view face bake)

- Added photometric source-pose estimation for canonical-frame bakes
  (`estimate_pose_photometric`): the ortho path no longer assumes the conditioning photo
  was taken from the canonical front. Multi-view reconstruction canonicalizes the OBJECT
  (symmetry plane onto the world axes), not the camera, so a photo of a subject whose head
  is turned sits 15-25 degrees away from the canonical front; five independent
  measurements put the checked face photo at azimuth +15..20 / elevation +8, and
  projecting it at 0 was the dominant cause of the doubled-face artifact. The estimator
  correlates signed gradient VECTOR fields (magnitude is bilaterally symmetric on faces
  and cannot tell a pose from its mirror) between the photo and untextured renders, with
  interior-distance weighting so pose-insensitive silhouette edges do not swamp the
  signal; the declared pose wins unless beaten by a real margin, and a genuinely frontal
  input returns "not estimated". See ADR 0008.
- Added overlap-photometric reference registration
  (`register_reference_by_source_overlap`): after the source view projects, each reference
  photo is aligned by minimizing source-weighted RGB disagreement over mutually observed
  texels. Silhouette registration aligns outlines — on heads, the hair contour — and left
  interior features displaced by ~6% of the frame (the profile's eye painted on the
  temple); registering interior content to the source's painted truth removed the doubled
  eye and cut the adversarial QA harness failures from 62 to 14 in one change.
- Added crop-immune width-profile registration (`register_view_by_width_profile`) as the
  coarse aligner for reference photos: row-wise silhouette widths below the subject's top
  form a scale-sensitive signature that survives cropping differences (area-IoU rewards
  degenerate blow-ups and edge-chamfer locks onto long edges on cropped photos).
- Added a layered-density witness gate to the projector (`layered_zone_gate`): photo
  regions where more than 10% of projected samples land a thin gap (3 epsilon to
  0.03 x diagonal) behind the first surface are imaging stacked film shells (hair wisps
  over a scalp); sub-pixel aim — not content — decides which sheet each pixel stamps, and
  the pixels are themselves material mixtures, so the view surrenders the whole region to
  better views or fill. Pixel-level gating was measured and rejected (survivors between
  layered pixels still anchor flakes). With mirror-source gating this cut hairline
  flake/debris failures by 77-80% (dark-debris view failures 22 -> 0).
- Mirror completion sources are now excluded from any view's contested layered band, and
  the confidence floor returns to 0.35 in the bake call: provenance tracing showed >90% of
  hairline flake islands were mirror/harmonic COPIES of a few low-confidence mixture
  anchors, not direct projections. Disabling mirror completion entirely measured worse
  (the far cheek degrades to harmonic mush) — gating, not removal.
- The source view stops painting beyond ~66 degrees off-axis in ortho multi-view bakes
  (per-role facing threshold 0.4): beyond that the source's samples are stretched rim
  content, and reference photos are the better witness. Single-view and perspective bakes
  keep the wide threshold (stretched content beats no content when nothing else covers).

### Fixed (second adversarial cycle)

- Fixed four numerically verified math defects found by a line-by-line audit (81 checks
  against analytic ground truth): the outlier filter's 2-hop consensus let every texel
  vote for itself (the diagonal of A@A equals vertex degree), so foreign-view misprojection
  islands self-certified and were never dropped — the consensus now excludes self-votes and
  binarizes path counts; the splat silhouette's dilation biased every edge-based
  registration toward +4% scale (a pixel-perfect canonical photo registered at 1.04) — an
  erosion pass restores the true rim; the strict z-buffer's scalar epsilon made smooth
  tilted surfaces occlude THEMSELVES (up to 40% of visible texels at 55-75 degree tilt
  demoted to milky fill) — the epsilon is now slope-aware (standard shadow-mapping
  practice), keeping the base tolerance for front-on sheets; horizontal registration
  shifts were converted back through photo WIDTH while fitted in a height-normalized
  frame, corrupting non-square photos on the perspective path.
- Removed the source view's residual silhouette registration in ortho mode: the canonical
  recenter IS the registration, and the residual scale/shift search chased reconstruction
  error in the geometry, displacing the photo's features (measured: it doubled the
  duplicate-feature count at three-quarter views).
- Retuned the conflict-resolution source-priority floor 0.45 -> 0.25 after ablation: 0.45
  handed contested cheek texels to stretched reference content; 0.25 still lets a head-on
  reference overrule truly grazing source rim samples.
- Guarded the mesh-graph harmonic fill against singular solves (fully unobserved
  disconnected components now fall back to the KD fill instead of painting black).
- The texture pipeline's evidence standard is now an adversarial QA harness (20 views x 5
  defect detectors + pose-aware identity gates, calibrated so the reference photos
  themselves pass): the rejected face bundle scored 66 failed checks; after this cycle the
  same bundle recipe scores 9-10, with the doubled-feature and ghost classes at zero.
  Remaining known limits are documented in ADR 0008 (eye-region geometry slivers, the
  front-vs-profile photo tone band, and hairline wispiness that a baked opaque texture
  cannot represent).

### Added

- Added `abstract3d.segmentation`: robust subject matting that prefers the
  `isnet-general-use` checkpoint and cleans the matte (dominant components kept, pinholes
  closed) before any geometric use. The default u2net checkpoint amputated 40% of the
  subject on the checked profile photo (dark hair against a light background), silently
  corrupting the alpha-driven framing of every downstream stage.
- Added canonical-frame orthographic projection to the shared texture bake
  (`projection_model="orthographic"`, `canonical_border_ratio`): the bake replicates the
  shape model's own image preprocessing (bounding-box recenter at the training border
  ratio) and projects with the orthographic half-extent that reproduces that exact frame,
  making source-photo registration deterministic. The Hunyuan backend uses this mode; the
  perspective model remains for TripoSR. See ADR 0007.

- Added an experimental, license-gated Hunyuan3D-2.1 shape backend
  (`abstract3d:hunyuan3d21-local`, aliases `hunyuan3d21`, `hunyuan3d`, `hunyuan`) wrapping the
  official `tencent/Hunyuan3D-2.1` flow-matching DiT and shape VAE. The backend refuses to
  download or run weights until the operator acknowledges the territory-restricted Tencent
  Hunyuan Community License (`scene3d_hunyuan_license_accepted` or
  `ABSTRACT3D_HUNYUAN_ACCEPT_LICENSE=1`). Includes an adaptive coarse-to-fine volume decoder
  (host-side bookkeeping, exact-doubling level schedule) that replaces the upstream
  hierarchical decoder, which lost thin structures and misbehaved on Apple MPS. Ships with a
  new `abstract3d[hunyuan3d]` extra.
- Added `abstract3d.texturing`, a backend-agnostic projection texture bake shared by TripoSR
  and Hunyuan3D-2.1: silhouette-based source-pose estimation with an angular prior and IoU
  acceptance gate, 2D photo-to-silhouette registration plus interior-edge photometric
  refinement, contour alpha erosion, exposure harmonization for reference views, seam-feathered
  best-view-biased blending, geometry-verified mirror completion (3D observed-twin lookup),
  3D inverse-distance fill for unseen texels, and mesh-scale-aware camera distance
  estimation. TripoSR keeps its triplane color field as the unseen-texel prior.
- Added a CPU UV-atlas rasterizer fallback for the TripoSR texture bake so `baked_basecolor`
  works on hosts without OpenGL 3.3 / geometry-shader support (headless Linux, some Windows
  drivers). The ModernGL path remains the default when available.
- Textured preview renders now use a softer headlight shading model so contact sheets review
  the baked albedo rather than a single hard key light.
- Documented the TRELLIS.2 acceptance posture: the backend is maintained, and the gated
  DINOv3 companion is documented with a license summary (Meta DINOv3 License: commercial use
  permitted, "Built with DINOv3" attribution on distribution, acceptable-use restrictions,
  reviewed access), explicit unblock steps in `docs/models.md` and `docs/troubleshooting.md`,
  an upgraded runtime error message with the same guidance, and an acknowledgement entry in
  `ACKNOWLEDGEMENTS.md`.
- Added multi-view geometry conditioning through the official `tencent/Hunyuan3D-2mv`
  checkpoint (same license gate as 2.1, loaded from the pinned 2.1 source via a key-exact
  config namespace remap). Reference views whose angles snap to the trained front/left/back/
  right slots condition the shape itself; generation metadata records the views used. On the
  checked face proof, front + both profiles raised observed texture coverage from 0.19 to
  0.74 and replaced the hallucinated back of the head.
- Upgraded the multi-view texture bake through two adversarial review/empirical-attack
  cycles: per-reference pose solving in a window around the declared angle (real photo
  angles are routinely 10-20 degrees off their label; the refined pose is accepted only
  when it beats the declared pose's silhouette IoU by a clear margin), overlap-based
  per-channel color harmonization with a revert-on-confound rule (gains that fail to
  reconcile the overlap indicate content mismatch, not exposure), a reprojection-error QA
  gate evaluated against the union of previously accepted views (catches
  reference-vs-reference conflicts), per-texel best-witness conflict resolution (localized
  disagreement zeroes the weaker witness only on disputed texels instead of punishing the
  whole view), a mesh-scale-relative depth-occlusion tolerance (a fixed normalized
  tolerance let hair sheets bake through onto the face on large meshes), and a mesh-graph
  harmonic fill (Dirichlet Laplace solve over mesh edges, normal-weighted KD fallback) so
  hidden regions diffuse smooth colors along the surface instead of borrowing across
  space. The world-frame azimuth-sector masks in the projector now apply only when no
  rendered depth map is available: the depth test strictly dominates them, and empirically
  the sector mask discarded about half of the depth-validated texels on profile views
  (face proof coverage with front + both profiles: 0.50 -> 0.74).

### Fixed

- Integrated four adversarial audit findings (each proven on ground-truth harnesses before
  fixing): the pose-search grid never scored the DECLARED CENTER pose
  (`arange(-window, window, step)` excludes it unless window is a step multiple — rebuilt
  from symmetric integer offsets); `estimate_camera_distance` normalized column extents by
  width while the projector's NDC unit is height-based (up to +49% distance error on
  landscape frames) and used a one-pass linear correction biased 10-16% for deep subjects
  (now a 3-step fixed-point iteration); silhouette registration squashed photo masks
  anisotropically into the square comparison frame (now an aspect-preserving letterbox);
  and exposure harmonization gains are now gated on per-texel log-ratio spread (a true
  exposure shift is one tight multiplicative relation; content-mismatched overlap produced
  0.5-clamped gains that tinted whole views).
- Removed `refine_registration_photometric` from the default bake path: an adversarial
  ground-truth test recovered 0 of 15 injected known shifts, and the NCC objective proposed
  nearly the same warp regardless of the true offset (a constant attractor). The function
  remains available for callers.
- Mirror-symmetry completion now only copies from CONFIDENT observed texels
  (blend weight >= 0.35): 89% of unrestricted mirror sources on the checked face proof were
  grazing rim samples, and copying them fabricated a bright skin patch on the hidden crown.
- Fixed the ghosted "second face" on textured previews: the preview renderer's diffuse
  lighting re-drew the mesh's own geometric features (eye sockets, brow ridges, lip
  creases) over the photo albedo wherever geometry and texture disagree by a few percent.
  Textured meshes now render with a flat-biased headlight (12% diffuse cue); an
  independent CPU rasterization of the same textured mesh was used to prove the texture
  itself was correct. Also guarded a GLSL uniform that the shader change made eliminable.
- Fixed duplicate feature stamping onto hidden crust sheets: projector visibility is now a
  strict per-photo-pixel first-surface z-buffer built from the projected texels themselves
  (every surface texel occludes regardless of facing or photo alpha, 3x3 conservative
  widening, epsilon 0.25% of the surface diagonal). Replaces the GL depth-map tolerance
  test — any tolerance loose enough to survive depth-map interpolation stamped both sheets
  of the 0.005-0.02-unit hair films that generated meshes grow — and removes the
  world-frame azimuth sector masks entirely.
- Fixed reference-photo registration for differently-cropped photos: silhouette matching
  now scores symmetric edge-chamfer distance instead of region IoU (region IoU rewards
  degenerate blow-ups where the mismatch leaves the frame), and rows/columns touching the
  photo frame are treated as crop lines rather than shape boundaries.
- Added a mesh-surface outlier filter: a two-hop mesh-graph consensus iteratively erodes
  observed texels whose winning view AND color are foreign to their neighborhood (rim
  misprojections such as forehead pixels on hair-shell tips), demoting them to unobserved
  so the fill replaces them.
- Conflict resolution now gives the SOURCE photo priority on disputed texels wherever it
  faces the surface well (weight above 0.45): the user's actual photo outranks synthesized
  or auxiliary references on well-seen surface, while grazing rim content still defers to
  the best-facing witness.
- The mesh-graph harmonic fill now uses crease-aware edge conductance (normal-agreement
  squared), so hidden-region color diffuses along smooth sheets instead of leaking across
  shell fusion seams (skin bleeding up onto hair films).
- Fixed the pure-Python `torchmcubes` fallback so extracted meshes use the native torchmcubes
  `(x, y, z)` vertex convention and outward face winding. Without native torchmcubes installed
  (the default, since the package ships no wheels), every TripoSR mesh came out with swapped
  X/Z axes and inward-facing normals, which also silently collapsed observed-view texture
  coverage (facing weights saw inward normals). Environments with a locally compiled
  torchmcubes were unaffected, which is why earlier proof assets looked correct.
- Fixed a stale trimesh normal cache after `repair.fix_normals` in the TripoSR and Step1X
  cleanup passes: `Trimesh.invert` intentionally preserves cached normals across its cache
  clear, so vertex normals cached before the repair stayed inward even after the winding was
  corrected. The cleanup now drops the cache and recomputes normals from the repaired faces.
- Fixed TRELLIS.2 device selection to fall back to an available accelerator instead of
  returning `mps`/`cuda` unconditionally when explicitly requested on hosts without them.
- Made the validation harness process-guard portable: `start_new_session`/`os.killpg` are now
  used only on POSIX, with a psutil-based process-tree fallback for Windows.
- Changed the validation harness default device from `mps` to `auto`.

- Fixed composed `t23d` to default to automatic background segmentation instead of forcing
  `remove_background=False`. The official TripoSR pipeline always segments and recenters the
  subject before inference; feeding it opaque studio-background images degenerated thin
  subjects (e.g. chairs) into billboard-like sheets. The composed chair proof went from a
  collapsed 168-face sheet to a recognizable 77k-face chair with this fix alone.
- Fixed the observed-view projector to clip bilinear gather indices; texels projecting outside
  the image frame crashed the bake (newly reachable with estimated camera distances).
- Made the depth-occlusion renderer fall back to facing-only visibility when no standalone GL
  context exists instead of failing the whole texture bake.
- Fixed the validation harness memory guard to apply per heavy backend regardless of the
  requested device string; the previous `device == "mps"` check silently disabled the Step1X
  64 GiB guard once the default device became `auto`. Hunyuan3D runs now get the same default
  guard.
- Texture completion and inpaint gating now use the true projection coverage instead of the
  feathered blend weights, so seam-adjacent observed texels are no longer overwritten by
  mirror or inpaint fill.
- Unseen-texel fill now happens in 3D surface space (inverse-distance weighting over the
  nearest observed texels via a KD-tree) instead of UV space. UV-space fills bleed colors
  across unrelated xatlas charts and produced patchwork/speckle noise on hidden regions; the
  3D formulation lets thin parts (a chair back) borrow correctly from their opposite face.
  The upstream Hunyuan texture pipeline reaches the same conclusion with its mesh-graph
  inpainter.
- Projection weights are despeckled (small isolated coverage islands removed) and the
  position/normal atlases are no longer border-dilated, which used to let contaminated chart
  -gap texels pass the facing test and bake shadow pixels as speckle.
- Hunyuan3D meshes are decimated to a 120k-face budget before texture bake: above that,
  marching-cubes micro-detail fragments the UV atlas into thousands of confetti charts
  (3315 at 200k faces vs 87 at 120k on the owl proof) that show up as salt-and-pepper noise.
- Fixed the matplotlib preview fallback to honor the requested image size (it rendered at
  1.6x due to a figsize/dpi mismatch) and to survive GLB textures carried as raw encoded
  bytes, both of which broke previews on GL-less hosts.
- Fixed the pure-Python marching-cubes fallback to return an empty mesh for fields with no
  iso-crossing, matching native torchmcubes instead of raising.

## 0.1.0

- Added the first production-oriented `scene3d` plugin surface for AbstractCore.
- Added a validated local backend based on `stabilityai/TripoSR`.
- Added composed `text_to_scene3d` through `abstractvision` image generation plus TripoSR reconstruction.
- Added CLI commands for `catalog`, `i23d`, `t23d`, and `validate`.
- Added bundle outputs with previews, contact sheets, and per-case metadata.
- Added a reproducible local validation harness in [`scripts/validate_local.py`](scripts/validate_local.py).
- Added docs, ADRs, and benchmark assets for the validated Apple-local path.
