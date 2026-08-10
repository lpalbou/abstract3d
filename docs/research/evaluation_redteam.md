# Evaluation red-team rebuild — 2026-07-21

Adversarial audit of the model evaluation after the operator found severe
defects the automated panel (`scripts/model_quality_sweep.py`) never flagged:
an open-mouth mesh (e18), a completely wrong left texture side (e10), a
bulbous doubled-lip profile (e17), and ghost glasses (e21 — the only one the
old panel caught). This document maps every operator-flagged defect to the
detector that now catches it, with two-way validation evidence, the re-scored
ranking, and the blind spots that remain.

## Why the old evaluation failed (verified, not assumed)

1. **Single-side sampling.** The old panel rendered azimuths 0, +30, +55
   only — the subject's LEFT hemisphere (negative azimuths in the new
   convention) was never rendered. e10's left side is a white/silver wreck
   at az −60/−90; measured bright-blob fraction 0.081/0.085 vs 0.000 on its
   own right side. No metric can catch pixels that are never rendered.
2. **Shading-based mesh metrics.** `mesh_front`/`mesh_oblique` measured
   shaded clay pixels (lighting × geometry). Re-measured on today's renders,
   the duplication autocorrelation ranks e10 (single mouth, operator-proven)
   *above* e2 (the historical double mouth): 0.093 vs 0.089 front — the
   metric is noise at these operating points. The new geometry detectors
   read orthographic **depth maps** (pure geometry, no light).
3. **Zero semantic checks.** Nothing asked "is the mouth open?", "is the
   profile anatomical?", "is that skin on the hair?". The new panel encodes
   each operator-visible defect class as its own detector.
4. **Averaging.** A conflated or averaged score buries one ruined angle
   under six clean ones. The new panel takes the **minimum over angles** for
   both axes; one bad angle fails the model, as it does for the user.

Environment note: no facial-landmark stack is importable in the venv
(mediapipe / dlib / face_recognition / insightface / face_alignment all
missing; cv2 is present). The mouth/profile detectors therefore use
nose-anchored depth-map geometry instead of landmarks — no new dependencies
were added.

## New tooling

| script | role |
| --- | --- |
| `scripts/full_turntable.py` | Evidence pass: clay + textured renders at az {0, ±30, ±60, ±90, ±135, 180} × elev {0, 15}, 1536 px, face close-ups for the 7 front-hemisphere azimuths, labeled contact sheet + manifest per model, under `out/laurent-bust-redo/review/turntable/<model>/`. Azimuth sign verified empirically (synthetic red marker at glTF +X): **positive az = subject's anatomical LEFT**. Camera framing is automatic (bbox fit). |
| `scripts/defect_detectors.py` | The detector suite (below). Geometry detectors read orthographic depth maps rendered from the GLB; texture detectors read the turntable renders anchored by the mesh's own nose row. |
| `scripts/evaluation_panel.py` | Worst-angle scoring: per-detector margin `1 − value/(2·threshold)` (1 = perfect, 0.5 = at threshold, 0 = 2× threshold); mesh score = min over {az 0, +90, −90}; texture score = min over {az 0, ±30, ±60, ±90}. PASS iff score > 0.5. Emits JSON + markdown ranking. |
| `scripts/validate_detectors.py` | Two-way validation harness. Re-runs every detector on its named known-positive and known-negative models; non-zero exit on any violation. Run after any threshold or code change. |

## Defect → detector map (with two-way validation evidence)

All values below are measured, not asserted; re-run `validate_detectors.py`
to reproduce. "Fires/passes" = operator's verdict reproduced.

| operator-flagged defect | detector (metric) | threshold | fails (measured) | passes (measured) |
| --- | --- | --- | --- | --- |
| **Double mouth** (e2, wave-2 baseline) | `double_mouth` — count of distinct seams (valleys, prominence ≥ 0.003 radius units) in mid-sagittal depth profile, window nose+[0.14, 0.36]·Wf | ≥ 2 seams | e2 = **2** seams (0.0031, 0.0050) | all 8 current candidates ≤ 1 (e20 = 1, e15/e17 = 0) |
| **Open mouth** (e18; photo mouth closed) | `open_mouth` — deepest valley prominence in lip window nose+[0.08, 0.45]·Wf of the mid-sagittal depth profile | ≥ 0.0105 | e18 = **0.0125** | e20 = 0.0068; worst closed model e11 = 0.0086; e10 0.0046, e21 0.0043, e17 0.0029, e15 0.0025 |
| **Bulbous nose + doubled-lip profile** (e17) | `face_inflation` — (nose protrusion − neck trough) / face width, unit-consistent via the render's own orthographic scale | ≥ 0.650 | e17 = **0.687** | next worst e2 = 0.609; e18 0.597, e10 0.595, e20 0.551 |
| same, second signal | `profile_ripples` — convex ridges (prominence ≥ 0.0045·Wf) below the nose on the ±90° silhouettes, worst side | ≥ 2 | e17 = **2**, e18 = 2 (parted lips read as an extra ridge — honest fire) | e2/e10/e11/e15/e20/e20_rebake/e21 = **0** |
| **Striated mouth** (e15, e17 — the wave-3 raking-light rejects) | `mouth_striation` — fine parallel ridges (prominence ≥ 0.0008 after 0.06·Wf detrend) in nose+[0.03, 0.45]·Wf, 1536 px depth | ≥ 3 ridges | e15 = **3**, e17 = **3** (also e10 = 3, e11 = 3, e2 = 3 — consistent with the operator's "mesh wrong" for the e10 class) | e20 = **2**, e20_rebake = 2, e18 = 2, e21 = 2 |
| **Left side smear** (e10: white/silver chaos over neck/jaw/cheek at az −60/−90) | `side_smear` — connected bright low-saturation blob fraction (max ch > 150, sat < 0.25, blobs ≥ 0.2% of face zone) over the face band, at ALL 7 azimuths | ≥ 0.020 | e10 az−60 = **0.081**, az−90 = **0.085** (also fires: e17 −90 0.057, e15 −90 0.032, e18 −90 0.029) | e20_rebake right side ≤ **0.0092**, e20 all angles ≤ 0.0038, e11 ≤ 0.0062, e21 ≤ 0.0144; photo front baseline 0.013 |
| **Face patches on hair** (phantom second face on the left hair mass; part of "texture awful") | `skin_on_crown` — skin-colored blob fraction of the crown zone (above nose_row − 0.12·H; nose row taken from the DEPTH map so a displaced texture cannot move the zone) | ≥ 0.12 | e10 az−30 = **0.213**, e11 = 0.177, e21 = 0.163, e15 = 0.125 | e17 = 0.081, e20 = 0.075, e18 = 0.075, e20_rebake = 0.074 |
| **Ghost glasses on cheeks** (e21; e20's right-side ghost) | `ghost_glasses` — port of the validated `tex_oblique_ghost` band count (extra wide dark row-bands beyond the widest + saturation speckle) at its validated operating point: textured render az +30, elev 12, 1536 px | ≥ 2 | e21 = **3.00**, e17 = 6.00, e20_fixed = 2.00, e18 = 2.00 | e20_rebake = **0.00**, e10 = 0.00, e15 = 0.00, e11 = 1.00 |

Wf = face width at the nose row; H = subject height. Every face window is
anchored on the nose tip measured on the front depth map (global mid-sagittal
depth maximum — verified unambiguous on all 9 calibration models, glasses
included) so windows track the subject, not the framing.

## Re-scored ranking (all 8 candidates, worst-angle gating)

Scores: margin at the worst angle; PASS iff > 0.5 (all detectors under
threshold at every angle). Full per-angle numbers:
`out/laurent-bust-redo/review/turntable/panel_scores.json`.

| rank | model | mesh | mesh verdict | mesh worst angle | texture | texture verdict | texture worst angle | fired detectors |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | e20_rebake_fixed | 0.576 | PASS | az+0 | 0.690 | PASS | az−30 | none |
| 2 | e20_fixed_views | 0.576 | PASS | az+0 | 0.500 | FAIL | az+30 | ghost_glasses |
| 3 | e18_windowed | 0.405 | FAIL | az+0 | 0.268 | FAIL | az−90 | open_mouth; profile_ripples; side_smear az−60/−90; ghost_glasses |
| 4 | e11_2mv_reg_hq | 0.500 | FAIL | az+0 | 0.261 | FAIL | az−30 | mouth_striation; skin_on_crown az+0/−30 |
| 5 | e21_single_refs | 0.573 | PASS | az+0 | 0.250 | FAIL | az+30 | skin_on_crown az+0/−30; ghost_glasses |
| 6 | e15_cleanfront | 0.500 | FAIL | az+0 | 0.190 | FAIL | az−90 | mouth_striation; side_smear az−30/−60/−90; skin_on_crown az−30 |
| 7 | e10_2mv_registered_refs | 0.500 | FAIL | az+0 | 0.000 | FAIL | az−30 | mouth_striation; side_smear az−30/−60/−90; skin_on_crown az+0/−30 |
| 8 | e17_clean4 | 0.472 | FAIL | az+0 | 0.000 | FAIL | az+30 | mouth_striation; face_inflation; profile_ripples; side_smear az−30/−60/−90; ghost_glasses |

### Cross-check against the operator's verdicts

- **e10 / e15 — "texture less worse (front), mesh wrong":** reproduced.
  Both FAIL mesh via `mouth_striation` (the soft/striated face class); both
  FAIL texture on the LEFT hemisphere the old panel never rendered (e10's
  worst texture angle az−30/−60/−90 at margin 0.000; front is its best
  texture angle — "less worse" — but the crown check still fires there).
- **e20_fixed_views — "mesh good, texture wrong":** reproduced exactly.
  Best mesh of the set (0.576 PASS); texture FAILS via the ghost band on its
  right oblique (score 2.00 = the operator's right-side glasses ghost).
- **e18 / e17 / e21 — "bad":** all FAIL. e18: open mouth (0.0125) + parted
  lip ridges + left smear + ghost band. e17: bulbous profile (0.687) +
  ripples + striation + worst texture. e21: mesh passes (its listed defect
  was texture), ghost glasses 3.00 + phantom-face-on-hair fire.
- **e20_rebake_fixed — "claims fixed texture": VERIFIED, with one caveat.**
  The right-oblique ghost band drops 2.00 → 0.00 (the fix's target), all
  smear/crown values sit far under threshold (max side smear 0.0092, crown
  0.074), and it is the only model passing both axes. Caveat (visual audit,
  `tex_e00_az-030.png`): small residual ghost fragments near the left ear
  and a dark wedge on the left jaw are still visible to the eye but below
  every threshold that can be calibrated against the current negatives; the
  ported ghost metric cannot gate the LEFT oblique at all (see blind spot 2).

## Residual blind spots (honest list)

1. **Single-subject calibration.** Every threshold is calibrated on THIS
   bust family (dark hair, black shirt, dark glasses, closed mouth, beard).
   The palette-based texture detectors (`side_smear` bright/low-sat,
   `skin_on_crown` warm-hue) encode this subject's appearance; a blond or
   white-shirt subject needs re-calibration. The geometry detectors
   (nose-anchored, Wf-normalized) should transfer better but are also
   validated only here.
2. **Left-oblique ghost gating.** The ported ghost band metric returns
   3.0–13.0 at az −30 for EVERY model including the cleanest — this
   subject's left beard/jaw shadow reads as extra dark bands. The left value
   is recorded as advisory only (`advisory_left` in the JSON); a left-side
   lens echo below the crown line and dimmer than the smear threshold would
   escape the current gates. (e20_rebake's residual left fragments are
   exactly this class — visible, sub-threshold, un-gated.)
3. **e18's skin-patches-on-hair.** They sit at temple level, BELOW the crown
   line, mixed with legitimate temple skin — `skin_on_crown` cannot separate
   them (e18 crown = 0.075, under the 0.12 gate). e18 is caught by four
   other detectors, but a hypothetical model whose ONLY defect is temple-
   level skin-on-hair would pass the texture gates.
4. **"Soft face" / identity likeness.** No detector measures whether the
   carved face is SHARP and looks like the person (the e10 "soft face"
   complaint fires here only via its striation side effect). A
   photo-referenced depth comparison (photo-space landmarks → expected
   relief) is the missing instrument; without landmarks in this venv it
   remains open.
5. **Head slab.** The wave-2 "head slab" (flat panel in the hair mass,
   visible in e15's az−60 clay) measures at 0.054 planar-patch fraction vs
   0.020–0.034 for the others — direction is right but the margin is too
   thin to validate both ways against the operator record, so it is NOT
   gated. Turntable contact sheets remain the human check for it.
6. **Back hemisphere texture.** Detectors score az 0/±30/±60/±90; ±135/180
   are rendered in the turntable but only eyeballed. The old
   `bust_assessment.py` back-luminance check still covers the
   unpainted-dark-back class if run separately.
7. **Threshold margins.** The open-mouth gap between the positive (0.0125)
   and the worst negative (0.0086) is ~1.45×; mouth_striation separates 3 vs
   2 ridges with prominences near 0.003. These hold on all 9 calibration
   models but are not wide margins; new models scoring within ±20% of a
   threshold should get a human look at the contact sheet before any
   ship/kill decision.
8. **Depth maps miss texture-only mouth state.** `open_mouth` reads
   geometry; a CLOSED mesh with an open mouth *painted* on (texture-only
   fabricated expression) would pass it. The texture side of that class has
   no validated detector yet (no positive case in the current set).

## Evidence locations

- Turntable renders + contact sheets + manifests:
  `out/laurent-bust-redo/review/turntable/<model>/` (8 models).
- Panel scores (per-angle, per-detector raw values + thresholds):
  `out/laurent-bust-redo/review/turntable/panel_scores.json`, ranking table
  in `panel_ranking.md` beside it.
- Validation transcript: run
  `.venv python scripts/validate_detectors.py` from the repo root
  (17 checks, exit 0 = all verdicts reproduced).
