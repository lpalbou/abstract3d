# Evaluation strategy v2 — general comparisons, worst-angle acceptance (2026-07-21)

Adversarial rebuild of the model evaluation after the operator's ruling on the
previous attempts: the evaluation kept declaring bad models good, and the
first corrections were one-example rules ("i don't want specific rule to be
created out of ONE FUCKING EXAMPLE"). This document defines the general
evaluation, shows the calibration evidence over the 8-model validation set,
names what was REJECTED and why, and names the residual blind spots honestly.

Implementation: `src/abstract3d/model_evaluation.py` (library, thresholds in
`EvaluationConfig`), `scripts/evaluate_model.py` (CLI + verdict JSON),
`tests/test_model_evaluation.py` (synthetic unit cases).

## 1. Why the previous evaluations failed (read before changing anything)

`scripts/model_quality_sweep.py` (the failed 4-metric panel):

- **Two angles per axis.** Front + one RIGHT oblique. e10's left-side smear
  and every left-only defect were invisible by construction. No worst-angle
  semantics existed to catch them even if rendered.
- **`mesh_front` (clay-vs-photo edge-row NCC)** compared shape-shaded clay
  against albedo-dominated photo rows; the correlation was carried by the one
  dominant edge (glasses band) and noise elsewhere — non-discriminative
  (measured again this round: 0.60–0.70 across good AND bad meshes).
- **`mesh_oblique` (autocorrelation duplication + bumpiness)** ran on
  flat-lit clay that hides the ridges the operator sees under raking light,
  and its bumpiness term measured hair/stubble as much as defects.
- **`tex_front_dE` (CIE76 over the face band)** punishes global exposure
  differences more than placement errors: a correctly-placed darker texture
  scores worse than lips painted on the nose at matched tone. Also mean-
  aggregated: localized ghosts dilute into a large agreeing shirt.
- **`tex_oblique_ghost` (extra dark glasses-like bands)** is exactly the
  banned class: a ghost-glasses detector generalized from one failure.

`scripts/model_quality_sweep_v2.py` repeated the sin with better instruments:
mouth-region ridge energy with five mouth-specific constants and a
double-mouth "surcharge" — a mouth detector, calibrated on the failure set.

`scripts/bust_assessment.py` contributed two survivors: the subject-mask /
bbox conventions, and the honest calibration style (thresholds documented
with the evidence on both sides). Its duplication autocorrelation, re-tested
on headlight clay shading this round, does NOT separate (see §5).

## 2. The general structure

Two axes (mesh, texture), each scored by a few general comparisons, each
reduced to its WORST angle, and an axis passes only if every measure passes.
Never a mean: nine good angles must not outvote one bad one, because the
operator inspects the orbit.

The single unifying principle: **the photo is the only ground truth.** Every
measure is either (a) a direct comparison against the photo where the photo
can see, or (b) an anatomical invariance the photo pins from the front
(bilateral symmetry, height-strata of materials, closed-mouth surface
character, profile articulation of a human face).

### Texture axis

**T1 source-agreement (structure + chroma).** The photo is projected onto
the mesh through the front camera: world positions from the evaluation
renderer are mapped into photo pixels via a subject-box affine whose row
component is refined by feature-row registration (scale/shift search
maximizing edge-row-profile NCC — the registration the recenter-only bake
doctrine lacks, so a bake that painted displaced rows *disagrees* with the
projection instead of being reproduced by it). Visibility = world normal
facing the front camera (cos > 0.25) + depth-map occlusion test. At every
angle with front-visible overlap (0, ±30, ±60):

- *structure*: block-matched local NCC of luminance-equalized gradient
  fields, patchwise (48 px, ±8 px search); a patch disagrees when its best
  local NCC < 0.25 or when exactly one side carries structure. Score =
  disagreeing-patch fraction. Catches fabricated content (ghost frames,
  pasted face patches), displaced content (lips-on-nose class), and missing
  content, at whatever angle exposes them — one mechanism, no defect names.
- *chroma*: delta-E on the a,b channels only (p95), angles 0/±30. Exposure
  cannot mask wrong-colored content; L is excluded entirely.

**T2 palette-consistency.** At every angle including the back: after ONE
global exposure normalization (L gain/offset estimated at the front pose,
where correspondence is guaranteed — exposure is global, fabrications are
local, so the map cannot hide a defect), every subject pixel must be
colorable from the photo's own palette at the corresponding HEIGHT BAND
(6 bands, ±1 band of registration slack). Height-banding rides the same
orthographic-turntable invariance as the feature rows: an upright subject
shows the same material strata at the same heights from every azimuth.
Per-band K-means palettes; per-band outlier thresholds self-calibrated from
the photo's own nearest-palette distance distribution (p99.5, floor 10 ΔE).
Catches out-of-palette fabrication anywhere on the orbit (white/silver
smears, saturated speckle, wrong-material patches) — including angles where
no photo comparison exists.

### Mesh axis

**M1 mirror-symmetry.** IoU of the +az silhouette against the horizontally
mirrored −az silhouette, NO alignment step: under this renderer a
bilaterally symmetric mesh produces exactly mirror-identical frames, so any
deviation — including a framing-scale difference — is asymmetry signal.
±90° pairs are mathematically degenerate (opposite orthographic views share
one occluding contour; verified exact on the data) and are never scored.
Pairs: 30/60/135. Catches one-sided bulges, slabs, lateral deformations.

**M2 profile-articulation.** A human profile articulates several
protrusions (brow, nose, lips, chin — any face, no template); a "soft face"
reconstructs only a nose bump. Measure: summed prominence of SECONDARY
leading-edge extrema in the face band at ±90 (one view — the pair is
mirror-degenerate), in head-depth units, on both silhouette edges, keeping
the more articulated one. The largest extremum is excluded (every face has
a nose); what remains measures articulation without naming features.

**M3 cavity-mass.** Fraction of face-band pixels in a LOCAL shading valley
(darker than their mask-normalized Gaussian neighborhood by > 0.12) under
headlight clay shading, angles 0/±30. Local contrast — not absolute
darkness — so uniformly dark down-facing surfaces (under-jaw) contribute
nothing, while carved cavities and duplicated creases (open-mouth slot,
doubled lips, striation valleys) are exactly local deficits. The photo pins
the expected state: a closed-mouth portrait legitimizes only thin crease
lines. Scope note: this assumes the source photo shows no large open
cavities (the pipeline's portrait contract); a gaping-mouth source would
need the threshold re-pinned from its own photo.

**M4 silhouette floor.** IoU-maximizing scale/shift registration of the
front silhouette against the photo matte (the codebase's own matte-to-clay
registration doctrine as a measure). Measured NON-SEPARATING within the
validation family (every plausible bust lands 0.88–0.91, and the good e20
does not top the list) — so it is NOT a discriminating gate. Kept only as
an out-of-family floor (0.80): e6's collapsed flagship shape scores 0.60
and dies here. Reported as a diagnostic otherwise.

### Acceptance rule (the one-shot pipeline contract)

```json
{
  "accept": bool,                    // mesh_ok AND texture_ok
  "mesh_ok": bool, "texture_ok": bool,
  "failing_measures": ["texture.palette_outliers@-90.0", ...],
  "worst_angle": {"texture": {"source_structure": {"angle": "30.0", "value": 0.23}, ...}}
}
```

`scripts/evaluate_model.py --glb ... --photo ...` prints the full per-angle
matrix and exits 0 only on acceptance. Runtime: ~4–15 s per model on this
machine (renders the 10-view orbit fresh from the GLB at 768 px; no reuse
needed — well under the 3-minute budget).

## 3. Validation matrix (worst-angle scores over the operator-verdict set)

Numbers from `scripts/experimental/eval_calibration_run.py` (this tree,
2026-07-21). Direction: `structure/chroma/palette/cavity` lower-better
(max-aggregated), `symmetry/articulation/silhouette` higher-better
(min-aggregated). Thresholds in brackets.

| model | tex structure [≤0.32] | tex chroma [≤11.5] | tex palette [≤0.015] | mesh symmetry [≥0.93] | mesh articulation [≥0.055] | mesh cavity [≤0.072] | silhouette [≥0.80] |
|---|---|---|---|---|---|---|---|
| e10_2mv_registered_refs | 0.067 | 10.38 | **0.031** | **0.826** | **0.043** | 0.059 | 0.896 |
| e11_2mv_reg_hq | **0.394** | **12.49** | 0.003 | **0.840** | **0.031** | 0.062 | 0.890 |
| e15_cleanfront | 0.050 | 9.98 | 0.007 | **0.693** | **0.050** | 0.060 | 0.911 |
| e17_clean4 | 0.125 | **13.24** | 0.006 | **0.916** | 0.065 | 0.055 | 0.897 |
| e18_windowed | **0.407** | **15.52** | 0.006 | **0.863** | 0.093 | **0.084** | 0.894 |
| e20_fixed_views | **0.391** | **15.40** | 0.001 | 0.945 | 0.064 | 0.062 | 0.883 |
| e20_rebake_fixed | 0.234 | 9.72 | 0.002 | 0.945 | 0.064 | 0.062 | 0.883 |
| e21_single_refs | **0.727** | **24.86** | 0.001 | **0.876** | **0.028** | 0.047 | 0.888 |
| e2 (legacy double mouth) | 0.177 | **11.93** | 0.001 | **0.829** | **0.033** | 0.060 | 0.901 |
| e6 (legacy flagship slab) | 0.071 | **17.06** | **0.047** | **0.602** | 0.099 | **0.180** | **0.597** |

Bold = fails that threshold. Resulting verdicts vs the operator's:

| model | operator verdict | evaluation verdict | reproduced? |
|---|---|---|---|
| e18 | BAD mesh (open mouth), BAD texture | mesh FAIL (symmetry 0.863, cavity 0.084), texture FAIL (structure 0.407, chroma 15.5) | yes |
| e17 | BAD mesh (bulbous nose, doubled lips), BAD texture | mesh FAIL (symmetry 0.916), texture FAIL (chroma 13.2) | yes |
| e10 | BAD mesh, texture "less worse" front / side disqualifying | mesh FAIL (symmetry 0.826, articulation 0.043), texture FAIL (palette 0.031 at az −90 — the smear side, exactly) | yes |
| e15 | mesh wrong, texture "less worse" | mesh FAIL (symmetry 0.693, articulation 0.050), texture PASS | yes — see note (a) |
| e20_fixed_views | mesh GOOD, texture wrong | mesh PASS, texture FAIL (structure 0.391, chroma 15.4) | yes |
| e21 | BAD texture (ghost glasses) | texture FAIL (structure 0.727 — the highest of all: the doubled frames disagree with the projection at every angle), mesh FAIL | yes (mesh un-verdicted; see note (b)) |
| e20_rebake_fixed | best overall, NOT perfect | ONLY model accepted; scores visibly imperfect (structure 0.234 of 0.32 budget, chroma 9.7 of 11.5 — the residual jaw streak and lens speculars are inside the margins, not at zero) | yes |
| e11 (un-verdicted) | (slab, paint spill per REPORT) | mesh FAIL, texture FAIL | consistent |
| legacy double mouth (e2) | BAD mesh | mesh FAIL (symmetry 0.829, articulation 0.033) | yes — see note (c) |
| legacy slab-in-hair (e6, also collapsed shape) | BAD mesh | mesh FAIL (symmetry 0.602, cavity 0.180, silhouette 0.597) | yes |

Notes, honestly:

(a) **e15 texture passes.** The operator called e15's texture "less worse"
    — not good, not the champion. The evaluation accepts its texture axis
    (front structure/chroma are genuinely close to the photo) while failing
    the model on mesh. The COMBINED verdict (reject) matches the operator.
    If the operator's "less worse" was meant as a texture rejection, this is
    a mismatch on the sub-axis: e15's texture sits inside the same score
    region as e20_rebake's accepted texture and no general measure separates
    them (they differ mainly in mesh-driven silhouette noise). Named as a
    limit rather than tuned away with a rule.

(b) **e21 mesh fails (symmetry 0.876, articulation 0.028)** with no operator
    mesh verdict on record. Its clay shows a soft, low-articulation face —
    the same class the operator rejected on e10 ("soft face") — so the axis
    verdict is consistent with the standard, but it is unvalidated: recorded
    as a prediction, not a reproduced verdict.

(c) **The e2 double mouth is caught by symmetry + articulation, NOT by the
    cavity measure** (its striated mouth carves shallow valleys: cavity
    0.060, below threshold). The measured articulation of its profile is a
    near-featureless soft curve (0.033) and its mirror pairs disagree
    (0.829) — it fails the mesh axis three times over. The
    duplication-autocorrelation instrument that bust_assessment calibrated
    for this defect on matplotlib renders does NOT separate on headlight
    clay shading (§5) and was not shipped.

Threshold placement (gap evidence, worst-angle values):

- `tex_structure ≤ 0.32`: accepted worst 0.234 (e20_rebake) vs failing best
  0.391 (e20_fixed). Cut at the midpoint region, nearer the accept side.
- `tex_chroma ≤ 11.5`: accepted worst 10.38 (e10 — which fails elsewhere)
  vs failing best 11.93 (e2). e20_rebake at 9.72.
- `tex_palette ≤ 0.015`: clean models ≤ 0.007 vs e10 at 0.031 (and e6
  0.047). More than 4× separation.
- `mesh_symmetry ≥ 0.93`: good meshes 0.945 vs failing best 0.916 (e17).
- `mesh_articulation ≥ 0.055`: good 0.064 vs failing best 0.050 (e15).
  The thinnest gap in the panel — flagged as fragile; a future subject with
  a genuinely flat profile would need re-calibration against ITS photo.
- `mesh_cavity ≤ 0.072`: good worst 0.062 vs e18 0.084 (open mouth) and e6
  0.180. e17's doubled lips (0.055) are NOT caught here (they are caught by
  symmetry); the cavity measure earns its place on e18 + e6 + the
  open-cavity class generally.
- `silhouette ≥ 0.80`: family 0.88–0.91 vs e6 at 0.60. A floor, not a gate.

## 4. How each observed defect class maps to a general comparison

| defect class (all observed instances) | general comparison that catches it | why it generalizes |
|---|---|---|
| fabricated expression (e18 open mouth) | M3 cavity-mass (+ M1 symmetry it also broke) | any carved cavity absent from a closed-mouth photo adds local valley mass; nothing mouth-specific — a gouged cheek or carved eye socket lands in the same statistic |
| doubled features (e17 lips; legacy e2 double mouth) | M1 symmetry + M2 articulation (excess/deficient profile structure) | duplication deforms the profile and breaks left-right agreement; no feature template |
| deformed anatomy (e17 bulbous nose) | M1 symmetry | a deformed blob is never bilaterally clean at silhouette level |
| soft/vague face (e10, e21) | M2 articulation | missing protrusions = missing secondary extrema, whoever they belong to |
| slab in hair (e10/e11/e15 class, e6 worst case) | M1 symmetry (one-sided slabs), M4 floor (gross) | a plate is either asymmetric or out-of-silhouette |
| misplaced texture (e20 lips-on-nose, displaced features) | T1 structure+chroma vs the ROW-REGISTERED projection | the projection carries the photo's features at the photo's rows; displaced paint disagrees locally |
| ghost content (e21 ghost glasses, e18 ghosts/patches) | T1 structure (render structure where projection has none) | fabrication = structure without photo support, whatever it depicts |
| out-of-palette smears (e10 white/silver side) | T2 banded palette at ALL angles | the photo defines what colors this subject's surface can be, per height stratum |
| face patches on hair (e18) | T1 structure + T2 banded palette (skin is out-of-palette at the crown band) | band-locality makes "legal color, wrong stratum" catchable without naming hair |
| wrong texture at side angles generally | worst-angle aggregation | a defect at any angle IS the score |

## 5. Rejected measures (with the numbers that killed them)

- **Clay-vs-photo edge-row NCC (v1 `mesh_front`)** — measured 0.60–0.70
  across good and bad meshes (dominant glasses edge carries it). Rejected:
  non-separating.
- **Clay-vs-photo 2D structure agreement (M2 candidate of round 2)** —
  disagreeing-patch fraction 0.33–0.50 with the GOOD mesh at 0.50 (shape
  shading vs albedo is a modality gap block matching cannot bridge).
  Rejected: inverted/non-separating.
- **Profile leading-edge curvature-row NCC vs photo rows** — e18 0.27 vs
  e20 0.43 but e17 0.35 ≈ e2 0.41: no clean cut, and the photo's edge rows
  (albedo) are not the profile's curvature rows (shape). Rejected;
  articulation (M2) replaced it with a photo-free anatomical prior.
- **Duplication autocorrelation on headlight clay (bust_assessment port)**
  — peak ratios 0.18–0.40 with NO separation (e2 double mouth 0.21 < e18
  0.40 < clean e20 0.24 interleaved). The 2026-07-20 calibration (0.071 vs
  0.034) came from a different renderer/lighting; on this evaluation's
  shading the instrument does not discriminate. Rejected here; the defect
  class is covered by M1+M2 (verified on e2).
- **Shading-noise (high-frequency residual)** — 0.016–0.023 across ALL
  models, good and bad. Faces fail by structure, not roughness, in this
  family. Rejected: non-separating.
- **Whole-subject palette with L-discounted distance (round-1 T2)** —
  missed e10's ACHROMATIC white smear entirely (0.02 worst angle,
  indistinguishable from clean). Root cause: discounting L makes white ≈
  gray shirt highlight. Replaced by exposure-normalized full-LAB distance +
  height banding (0.031 vs ≤0.007 clean — separation appeared only after
  both fixes).
- **Mean-aggregated anything** — e10's side smear averages into a clean
  front; e21's ghosts average into a large agreeing shirt. Worst-angle or
  nothing.
- **CIE76 full delta-E (v1 `tex_front_dE`)** — punishes exposure; ranked
  the rebake below darker-but-wrong textures in v1's own history. Replaced
  by the L-equalized structure + a,b-only chroma split.
- **SFace identity similarity** (v2 extra) — measured blind to the
  ghost-glasses class in v2's own documentation; identity embeddings
  compress exactly the cues this evaluation needs. Not shipped; noted as
  available (cv2 + cached ONNX) for future identity-drift questions, which
  are a different question than defect detection.
- **YuNet face-detection confidence on clay as a "face prior"** — tested
  this round: e10's soft face scores 0.82 (highest!) while the good e20
  scores 0.75; confidence tracks smoothness, not correctness. Rejected:
  non-separating and semantically wrong for this job.

## 6. Audit of the concurrent per-defect detector suite (Adversary G)

`scripts/defect_detectors.py` + `scripts/full_turntable.py` were written in
parallel. The turntable renderer is good and complementary (both sides +
back + labeled contact sheets — the evidence pass; this evaluation renders
its own buffers and does not depend on it). The detector suite is honest
work with real calibration tables, but it is FIVE NAMED DEFECT DETECTORS
(`open_mouth`, `face_inflation`, `profile_ripples`, `side_smear`,
`skin_on_crown`, `ghost_glasses`) — each an instance of the class the
operator forbade: `open_mouth` encodes a lip window in nose-anchored
coordinates; `ghost_glasses` encodes an eyewear strip and lens material
model; `skin_on_crown` encodes a skin-color model over a crown zone. Each
would have to be re-invented for the NEXT defect (a ghost EAR, paint on the
NECK, an open EYE). Where their mechanisms are general they are already
subsumed: `side_smear` ≈ T2 palette (theirs is a bright-desaturated special
case; T2 catches any out-of-palette color), `profile_ripples` ≈ M2/M1,
`open_mouth` ⊂ M3 cavity-mass, `face_inflation` is the one instrument with
no general counterpart here (nose-to-neck protrusion ratio — kept unshipped;
M1 catches e17 regardless). Verdict: keep `full_turntable.py` as the render
pass; the detectors stand as calibration DOCUMENTATION of the defect
classes, not as the acceptance gate.

## 7. Residual blind spots (named, not hidden)

1. **In-palette, in-band, back-hemisphere fabrication.** A hair-colored
   wrong pattern on the back of the head passes T2 (legal color, legal
   band) and is invisible to T1 (no photo there). No general photo-grounded
   measure exists for content the photo never saw; the only honest
   instruments are cross-view self-consistency priors that would need their
   own validation set.
2. **Bilaterally-symmetric geometry fabrication.** A perfectly symmetric
   wrong feature (e.g., symmetric horns) passes M1; M3/M2 catch it only if
   it carves cavities or deforms the profile band. The silhouette floor
   (0.80) is the only backstop for gross cases.
3. **Texture at extreme grazing angles.** T1 stops at ±60 (projection
   validity); ±90 is covered by T2 only — a ghost at ±90 that stays
   in-palette and in-band escapes. e10's smear was out-of-palette, so it
   was caught; a skin-toned smear on the jaw at 90° would not be.
4. **The articulation gap is thin** (0.050 fail vs 0.064 pass). A subject
   with a genuinely soft profile (round face, full beard covering the chin
   line) could false-fail; the threshold must be re-derived per subject
   class if the pipeline moves beyond this portrait family.
5. **Expression fabrication milder than a cavity.** A slight smile (lip
   curve change, no carved slot) changes neither cavities nor symmetry
   measurably at 768 px. Catching it generally needs landmark-level
   expression comparison (a face-landmark model on clay renders is
   unreliable — YuNet confidences 0.18–0.29 on several clay fronts here).
6. **Single-subject calibration.** All thresholds are calibrated on one
   subject's validation set (the assignment's set). The MEASURES are
   general; the CUTS are this-subject-validated. The config dataclass
   exists precisely so the next subject's ladder re-pins them, and the
   palette/row machinery self-calibrates from whatever photo is passed.
7. **Renderer coupling.** Mirror symmetry's "exact for symmetric meshes"
   property and the cavity statistics are properties of THIS renderer's
   camera/shading laws (replicated from `abstract3d.rendering`). A renderer
   change requires re-verifying both (unit tests pin the invariants
   synthetically).

## 8. Reproduction

```bash
# full validation matrix (10 models, ~2 min)
.venv/bin/python scripts/experimental/eval_calibration_run.py

# one model, full matrix + verdict JSON + projection debug images
.venv/bin/python scripts/evaluate_model.py \
  --photo /tmp/laurent_front_clean4.png \
  --glb out/laurent-bust-redo/e20_rebake_fixed/scene.glb \
  --json-out /tmp/verdict.json --debug-dir /tmp/eval_debug

# unit tests (synthetic, no GL)
.venv/bin/python -m pytest tests/test_model_evaluation.py -q
```
