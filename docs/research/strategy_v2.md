# Strategy v2 — photo → 3D bust, ratified end-to-end (adversarial review S1, 2026-07-21)

Status: RATIFIED REPLACEMENT for the implicit strategy that produced e2–e21.
This document is the operator-ordered answer to "no rules built from single
examples": every stage states what it GUARANTEES to the next, how that
guarantee is VERIFIED by general-purpose logic, and what happens on failure
(reject / redraw / degrade loudly — never silently continue). It changes no
code; the implementation plan at the end is the work queue.

Evidence base (all measured, all in-repo): `docs/research/texture_forensics.md`
(TF), `docs/research/viewgen_audit.md` (VA), `docs/research/
multiview_consistency_2026.md` (MC), `docs/research/generative_process_2026.md`
(GP), `docs/KnowledgeBase.md` (KB), `out/laurent-bust-redo/REPORT.md` (R),
`docs/research/independent_verification.md` (IV — the independent full-orbit
verification of all 8 candidate models, 95-row defect table; folded 2026-07-21,
see §14). External sources are cited by the names MC/GP already verified.

---

## 1. The attack — why the old strategy failed structurally

The operator's rejected artifacts (striated mouths e15/e17, fabricated
expression e18-cond, bulbous profile e17, lips-on-nose / lens-ghost /
jaw-blotch e20, smears e10/e21) are not seven bugs. They are seven surfaces
of five structural defects in the strategy itself. The independent
verification (IV, 95-row defect table over all 8 models) subsequently
confirmed every operator claim, graded ALL EIGHT bundles blocker on at
least one axis, and clarified one fact: e10's smear sits on the subject's
RIGHT (image-left facing him), az240–330 (IV operator cross-check +
orientation convention).

- **D1 — Generation was trusted on properties never verified.** The gates
  judged silhouette, palette, speculars — never "is this the same subject in
  the same state at the declared pose". Measured consequences: every local
  composite draw painted a DIFFERENT person and passed (VA §3, L1–L10); a
  side view with parted lips + teeth conditioned the DiT while the photo's
  mouth is closed (e18-cond, `review/forensics/d1_e18cond_sideleft_mouth.png`
  vs `d1_photo_mouth.png`); a windowed side_right twin with an added earbud
  and different glasses reached the bake (TF §0, NCC 0.204).
- **D2 — Instruments measured the wrong thing.** Full-bust silhouette IoU is
  pose-blind on busts (a ~50° view scores 0.76+ against a 90° clay, VA §1;
  the bake's own refiner would have KEPT the wrong 90°, TF §H2); 256 px
  thumbnails shipped striations (KB "thumbnails hide blocking defects");
  the speculars gate miscalibrated on a dark subject and BLOCKED honest
  regeneration (R, backlog 0019) — which invited the bypass in D4.
- **D3 — Doctrine substituted for measurement.** "The canonical recenter IS
  the registration" is exact for single-view meshes and false by +25–42 px
  (8–13 % of head height) on 2mv-conditioned meshes; the front photo itself
  painted lips on the nose and glasses on the forehead, winning 75–100 % of
  those texels (TF §1, §H4). Declared poses (±90°) were trusted while true
  poses were 65°/−67.5° (TF §H2) or 50° (VA §2 set A).
- **D4 — Gate failure was handled by bypass, not degradation.** e17 consumed
  side_right and back views that had FAILED their gates (VA §0); the
  loop-regen script routed the operator's face to a remote provider with
  placebo seeds via `owner=None` (VA §0). Nothing structural prevented
  either.
- **D5 — Provenance was optional.** Past bundles did not archive their
  inputs (out/bust/README.md "KNOWN GAP"); defect forensics had to
  reconstruct which pixels conditioned what, and one wave's diagnosis cost
  an isolated re-bake to even become admissible (TF §0).

The per-defect patches shipped so far (row law, pose ruler, window law,
split-consumer flag, duplicate-angle guard, source row-law rebake) are all
KEPT — each is measured general logic, not a one-example rule. What v2 adds
is the missing spine: one general verification mechanism ahead of every
consumer, degradation ladders instead of bypasses, and provenance as a
structural acceptance requirement.

---

## 2. Ratified architecture

```
photo ──► [A INTAKE] ──► [B SCAFFOLD MESH]        (2mv, front only, pass 1)
              │                 │ clay renders
              │                 ▼
              │          [C VIEW SYNTHESIS] ◄─── generator routes: mesh-conditioned /
              │                 │                LoRA-local-i2i / plain-local-i2i (§5)
              │                 ▼
              │          [D SAME-SUBJECT VERIFICATION] ── reject → redraw (budget) → drop angle
              │                 │ accepted views (windowed + full-span variants, roles)
              │                 ▼
              │          [E GEOMETRY PASS 2]      (2mv, windowed set, ≤3 tags)
              │                 │ mesh
              │                 ▼
              │          [F TEXTURE]              (row-registered source + full-span refs
              │                 │                  @measured az + completion + consistency weights)
              │                 ▼
              └────────► [G FULL-ORBIT EVALUATION] ── fail → one contingency loop (C..F against
                                │                      the pass-2 clay), then degrade loudly
                                ▼
                         [H BUNDLE]  1_source/ 2_views/ 3_model/ 4_renders/ + PROVENANCE.md
```

Loop budget: R1 (scaffold clay → views → pass 2) is mandatory; R2 is a
contingency triggered ONLY by a G failure, run once, against the pass-2
mesh's clay. Evidence: IM-3D closes the loop 2–3 times and iteration 1→2
resolves the superimposed-features class; 3+ is mostly cost (MC §3d); e20 —
the only operator-accepted geometry — came from exactly scaffold + one
conditioned pass (R wave 3).

Scope rule (class-based, not subject-based): person-class subjects
(`is_person_subject`) take this loop pipeline with a mandatory LOCAL image
provider; object-class subjects keep the meshless gated lane that the fleet
validated (KB "Meshless conditioning views obey two exact silhouette
identities"). The person/remote refusal is structural, not etiquette
(VA §0: `owner=None` measured leaking the photo + money to gpt-image-1).

---

## 3. Per-stage contracts

Format per stage: GUARANTEE (to the next stage) / VERIFY (general
mechanism) / ON FAILURE.

### A. Intake

- GUARANTEE: one matted source at known resolution, archived byte-exact
  with its matte and intake report in `1_source/`; subject class decided
  (person → loop + local provider); alpha window anchors computable.
- VERIFY: matte sanity (subject neither empty nor whole-frame —
  `_photo_matte_mask` bounds); anchor extraction succeeds (`window_anchors`
  raises on no subject); resolution ≥ the gate-calibration frame (KB
  "Reference-acceptance gates are fixed-pixel" — frame-size changes silently
  re-tune every calibrated floor, so intake normalizes to the calibrated
  frame and records the scale).
- ON FAILURE: refuse the run with the named reason. There is no degraded
  intake — every later guarantee keys on this stage.

### B. Scaffold mesh (pass 1)

- GUARANTEE: a watertight single-body front-conditioned mesh whose clay
  renders serve as the geometry hypothesis for view synthesis; family
  regime knobs recorded (the validated bust recipe is 512/50; family
  defaults 384/30 travel with the checkpoint — KB 4-view-cliff corollary b).
- VERIFY: existing health floors — bodies == 1 on single-subject
  generations, watertight, no shred signature (disconnection, not genus —
  KB: an accepted face measures genus 210; gate on DISCONNECTION).
- ON FAILURE: re-roll seed once (stochastic), then refuse. A broken scaffold
  poisons every downstream stage; there is nothing to degrade to yet.

### C. View synthesis

- GUARANTEE: for each requested angle, candidate views drawn by a LOCAL,
  seed-recording generator route, raw bytes persisted BEFORE any gating
  (VA §4 honest residual d: an unpersisted good draw is unrecoverable —
  billing/interruption is real), tone-matched and despecularized INSIDE the
  attempt loop so gates judge the exact pixels the consumers will see
  (KB gate-ordering law).
- VERIFY: stage D (below) — synthesis itself asserts nothing about quality.
- ON FAILURE (route level): a generator route that cannot run locally or
  cannot record real seeds is not a legal route for person subjects — hard
  refusal, cites VA §0 (placebo seeds, remote leak).

### D. Same-subject verification (the ONE mechanism — §4)

- GUARANTEE: every view handed to any consumer shows the SAME SUBJECT in
  the SAME STATE at its MEASURED pose, with feature rows lawful vs the
  geometry hypothesis, exactly one view per declared angle per consumer,
  windowed/full-span variants role-tagged.
- VERIFY: the channel battery of §4, all floors calibrated on the labeled
  archive (§4.4).
- ON FAILURE: redraw within budget (re-roll seed for stochastic failures:
  framing/IoU; escalate prompt/route for systematic ones: material/identity
  — the measured retry semantics, KB "three oracles"); after budget
  exhaustion DROP THE ANGLE and degrade the run shape (2mv is healthy at
  1–3 views, KB 4-view cliff; texture falls back to mirror/witnessed fill
  for that sector — the refusal branch ships a different product and that
  product has its own acceptance, KB law). One sanctioned correction per
  view (the row-remap family); corrected pixels re-earn EVERY channel
  (existing re-gate doctrine in `_synthesize_geometry_views`).

### E. Geometry conditioning (pass 2)

- GUARANTEE: a mesh conditioned only by verified, windowed, pose-honest
  views, ≤ 3 tags, tag priority front > back > side.
- VERIFY: window law applied from ONE shared anchor set across all views
  (fixes the e12 pair-drop class — backlog 0018 folds into the contract);
  tag snapping refuses views whose MEASURED azimuth is decisively off the
  tag (POSE_MAX_DELTA_DEG = 20°, decisive gap 0.10, head-band ruler);
  4-view cap enforced (measured shred: 822/559 bodies at 4 views — KB).
- ON FAILURE: fewer views (healthy envelope), ultimately the front-only
  single-view run — the known-good floor. Never condition on an unverified
  view; a wrong conditioning view is worse than single-view because the
  checkpoint TRUSTS its tags (code comment, measured double-mouth).

### F. Texture

- GUARANTEE: an atlas where photo-observed texels are painted only by
  photo content registered to the MESH'S OWN anatomy, references paint at
  measured azimuths exactly once per angle, synthesized content completes
  but never overwrites (protect_observed_texels absolute mode), and every
  authority decision is recorded per reference. THREE FURTHER GUARANTEES
  (IV fold — the three verified blockers live in this stage):
  (f1) **boundary continuity** — no view's projection ends in an
  unfeathered step on a continuous material surface: paint fades by
  incidence into completion/neighbor content at visibility boundaries, and
  the seam-band step stays under the calibrated allowance (the rebake's
  jagged right-jaw crack, IV e20_rebake #1, is the class; TF §4 had
  already recorded it as an "inherited class" residual and IV grades it
  BLOCKER — it survives the shipped gradient-domain compositing, so tone
  compositing alone is measured-insufficient);
  (f2) **palette-lawful completion** — fill/inpaint/completion texels draw
  their statistics only from like-material observed donors; content whose
  chroma lies outside every source-photo material part's band never ships
  in a completion region (IV blocker 2: cream neck/nape washes, cyan patch
  behind the ear, gray crown smoke, chrome window smears — the fill lane
  has no palette constraint today; the existing fill-luminance floor is
  L-only and the KB's own finding is that L-based axes are chroma-blind);
  (f3) **uniform-material commitment** — mesh regions that every
  witnessing view images as one uniform dark/glossy material (multi-view
  consensus, no positive base-material witness, ≥2 witnesses, dark-
  dominated context — the KB film-band commitment conditions verbatim)
  are committed to that material's consensus statistics instead of
  per-texel cross-material blending (IV blocker 3: all 8 models
  contaminate the sunglasses lenses — skin/amber in lens, ghost eyes
  through translucency, doubled frames).
- VERIFY: source-to-mesh row registration measured EVERY bake (front clay
  render vs source photo through the row law + feature anchors; correction
  applied only when decisive, monotonicity asserted, post-warp anchor
  landing verified — TF §4 step 3 generalized; on single-view meshes the
  corrector must measure as a no-op, which is the regression guard);
  duplicate-angle guard (raises, in code); split-consumer flag (in code);
  reference pose refine re-enabled ONLY with a head-band objective (the
  full-silhouette objective maximizes at the WRONG pose — TF §H2:
  0.957@90° vs true 65°); (f1) seam-band step metric over view-boundary
  bands (the `texture_qa` instrument, promoted from harness to gate,
  recalibrated on IV's labeled rows); (f2) per-material-region palette
  agreement of completion texels vs the source parts (the existing
  chroma-first k-means LAB part machinery pointed at fill regions);
  (f3) purity check inside committed regions (chroma/texture dispersion
  within the material class band).
- ON FAILURE: a reference failing acceptance is dropped from the bake with
  a recorded reason and the coverage cost is reported (TF §4 measured the
  honest trade: correct poses cost −1.25 coverage points); completion
  failure (hy3dpaint OOM/build/identity) falls back to
  mirror-symmetry/witnessed fill with a `#FALLBACK` note — never silent;
  fill texels that cannot be sourced within palette stay at the witnessed-
  fill floor rather than importing out-of-palette content; a region
  failing material-consensus conditions is NOT committed (the KB
  "third-eye class" guard: single-witness or contested regions blend
  normally — commitment without consensus painted defects).

### G. Full-orbit evaluation (§8)

- GUARANTEE: an accept/degraded/failed verdict computed from renders the
  defect classes cannot hide in, with the evidence attached. The render
  protocol adopts IV's instrument as the baseline (it found every defect
  the operator saw plus 80+ more): 12 azimuths at el 0 + 6 at el +20,
  clay AND textured, ≥ 1024 px, face crops at {0, ±30, ±60, ±90}, plus
  the 2048 px raking-light obliques — crown/nape blockers only show at
  el +20 or az180 (IV e20 #6–8, rebake #4).
- VERIFY: §8's acceptance rule (worst-angle discipline; same-subject
  battery reused against the RENDERS at every evidence pose; named
  render-side detector family).
- ON FAILURE: one contingency loop (R2), then degraded verdict (exit 3)
  with named reasons. Never a silent accept.

### H. Bundle / provenance

- GUARANTEE: the run is self-contained and auditable without a rerun:
  `1_source/` exact input bytes + matte + intake report; `2_views/` every
  candidate (accepted AND rejected) with role-bearing filenames
  (`geometry_*`/`texture_*`), raw draw bytes, seeds, route + LoRA identity,
  per-view verification reports; `3_model/` mesh + texture + metadata +
  bake stats; `4_renders/` the full evaluation orbit; `PROVENANCE.md`
  generated by the pipeline, not by hand.
- VERIFY: a bundle-completeness audit is PART of the G acceptance rule — a
  run that cannot prove its inputs cannot self-declare acceptance. This
  makes the provenance gap structural to close (out/bust/README.md), not
  disciplinary.
- ON FAILURE: verdict capped at degraded; the missing artifact is named.

Process-integrity rule spanning all stages: gate bypass does not exist in
one-shot mode. An experimental bypass requires an explicit flag, lands in
PROVENANCE.md, and caps the verdict at degraded. Evidence for why: e17
shipped from gate-FAILED views (VA §0); the 0019 miscalibration turned a
blocked gate into an invitation to bypass. When a gate blocks everything,
the run DEGRADES (fewer views); recalibration happens against the labeled
archive, never inline.

---

## 4. (design question 1) The general view-verification mechanism

### 4.1 Principle — one mechanism, several channels

**Same-subject agreement**: a generated view is accepted only if it is
explainable as the SAME SUBJECT in the SAME STATE under the DECLARED
(measured) rigid transformation of the source evidence. Verification =
predict, from the source photo + the current geometry hypothesis + already
accepted views, what the declared view must show wherever prediction is
possible; measure disagreement in a common frame; reject on decisive
disagreement. No channel knows what a mouth, a nose, or an accessory is —
every region is judged by the same rule, which is exactly what the
operator's no-one-example-rules ruling requires.

The channels, each with its domain (where it can predict) and its measured
precedent:

| # | channel | predicts | catches (measured precedent) |
|---|---|---|---|
| C1 | pose | the transformation itself: head-band-IoU azimuth sweep must agree with the declared angle | 40–50° under-rotation sold as profiles (VA set A); 20–25° errors the bake felt (TF §H2). Full-bust IoU is banned as a pose instrument (pose-blind, VA §1) |
| C2 | silhouette | mirror identities: back = mirrored front, left = mirrored right; clay-IoU where a clay exists | wrong-subject swaps (0.398–0.437 vs healthy 0.729+, KB); implausible backs; floors scaled by the source's own off-axis error |
| C3 | rows | same-elevation row law (Era3D Prop. 1): feature rows lawful vs the reference frame | the double-mouth class: 7–15 % row displacement scored 0.00 at lawful rows (R); e17's 9.5–21.4 %-low faces (VA set B) |
| C4 | contour | leading-edge profile of the view vs the clay's, feature-anchored, ≤ 2 % of head height per anchor | e17's bulbous nose / +6.6 % jaw elongation — a contour-curve disagreement, not a "nose detector" (VA §2/§4) |
| C5 | covisible content | mesh-mediated reprojection: project the source photo onto the geometry hypothesis, render at the view's MEASURED pose, compare in the co-visible region (masked Lab ΔE / SSIM) | e18's parted lips: the photo-implied mouth region content (closed) vs the drawn content (open, teeth) disagrees locally; also would have caught lips-on-nose at bake time (MC §4 metric 2, iNeRF-class) |
| C6 | semantic patches | DINOv2 patch correspondence source↔view; matched patches must agree within the view-change norm | expression changes at grazing covisibility, the earbud/different-glasses twin (TF §0), accessory hallucinations (VA back-view strap) |
| C7 | identity | embedding similarity source↔view where a face is visible (ArcFace-class via onnxruntime, or DINO-global fallback) | the systematic person flips every composite arm produced while passing all existing gates (VA L1–L10 labeled WRONG vs L9/L11 RIGHT) |
| C8 | pair/chain | mutual agreement of accepted views: left/right mirror pair; adjacency chaining (each new view checked against the nearest ACCEPTED view) for regions the source never saw | side-pair lies (both dropped, blame unattributable — in code); back-view content the front photo cannot constrain (MC §3b sequential-autoregressive evidence: independent draws are the worst configuration) |

### 4.2 Decision rule

- Core channels C1–C3 must be MEASURABLE for every view; an unmeasurable
  core channel is a rejection (fail closed — the existing doctrine: an
  unverifiable candidate must not condition the checkpoint).
- Content channels C4–C7 apply where their domain is non-empty (C7 abstains
  on a back view; C5 abstains where covisibility is below a floor).
  Abstention is RECORDED in the per-view report, never treated as a pass,
  and a view must be covered by at least one content channel or it is
  rejected (a view nobody can check is a view nobody may trust).
- Any decisive channel failure rejects the candidate. "Decisive" is a
  calibrated margin per channel, floors-not-walls (KB acceptance-gate law).
- Exactly one correction attempt per view, drawn from the sanctioned remap
  family (row translation / single-segment linear y-remap / shared-window
  harmonization); corrected pixels re-earn ALL channels.
- Consumers never see unverified pixels: D sits between synthesis and BOTH
  consumers (conditioning and bake), and the bake-side acceptance replay
  keeps its single-view sanity floors (KB 0017 scoping).

### 4.3 Why this catches the operator's three named defects as one mechanism

- **e18 parted lips**: C5 — the front photo observes a closed mouth; its
  reprojection through the scaffold to the side view's measured pose (~65°,
  where the mouth region is still covisible) predicts closed-mouth content;
  the drawn parted lips disagree locally. C6 independently: the matched
  mouth-region patches disagree beyond the view-change norm. No mouth
  detector — every covisible/matched region is judged identically.
- **e17 deformed profile**: C3 catches the 9.5–21.4 % row shifts; C4
  catches the residual +6.6 % jaw/chin elongation and any bulbous-nose
  leading-edge deviation from the clay contour; C1 catches the pose lie
  that let it condition at the wrong tag.
- **Identity flip**: C7 (embedding margin, calibrated on the ready-made
  labeled set L1–L11) plus C6 in aggregate (a different person disagrees
  broadly, not locally).

### 4.4 Calibration governance (general, not per-defect)

Every floor is calibrated on the labeled archive and re-verified after ANY
preprocessing change (KB: mask-normalized filtering, matte erosion, and
resize policy each flipped verdicts). The archive already exists:

- labeled FAIL: e18-cond parted-lips views, set A 50°-poses, set B low-face
  views, L-ladder WRONG-person arms, e15/e17 renders, the earbud twin — and
  since the IV fold, the FULL 95-row IV defect table across all 8 models
  (per-row severity + evidencing render path: the richest labeled corpus
  the project has; render-side detectors calibrate against it);
- labeled PASS: `viewgen_fixed/*` (accepted at ≤ 1.6 % features),
  L9/L11 arms, the object fleet's accepted views (the zero-false-fire bar
  for C5/C6 on non-person subjects). CORRECTED BY IV: `e20_rebake_fixed`
  closeups were listed here as labeled PASS on the strength of TF §4 —
  IV refutes that in the strong form (jagged jaw seam, amber/teal lens
  garbage, cream neck/nape washes; texture rank 3, BLOCKER). Its
  CENTER-FACE crops remain valid positive exemplars for the row-
  registration axis specifically (IV: "best single face crop across all
  8"), but the bundle as a whole moves to labeled FAIL. There is currently
  NO person-lane bundle labeled ACCEPT — positive calibration comes from
  the object fleet, synthetic probes, and center-face crops, and the first
  person-lane ACCEPT must be operator-validated, never self-declared.

Dependencies honesty: no face library is installed (checked: insightface /
mediapipe / dlib absent; onnxruntime 1.24 + transformers 5.9 + cv2 4.10
present). C7 therefore specifies an ArcFace-class ONNX model on bare
onnxruntime, with DINOv2 (transformers) as the fallback backbone — and the
subject wears sunglasses, which degrades face embeddings. EXPERIMENT
NAMED (W4): run both backbones over L1–L11; C7 is adopted only if the
labeled margin survives the sunglasses; otherwise C6-aggregate carries
identity and says so in the report.

### 4.5 Residual blind spots, stated

- Depth along the viewing axis that no view constrains (a nose's protrusion
  from a front photo) is not fully verifiable; the mechanism bounds it by
  cross-view consensus (C8) and by the loop (R2 re-verifies against the
  improved clay). It cannot certify it.
- Semantic re-rendering that preserves palette AND relief energy (the
  "sculpted goo" class) has no known gate in any statistic family tested
  (KB deliberate blind-spot note); the countermeasure is generator quality,
  which is what §5's route ladder buys.

---

## 5. (design question 2) View-synthesis architecture ruling

Measured standings:

- **Freestanding i2i draws** (current meshless rotate lane): worst measured
  configuration for persons — 40–50° pose errors that no gate saw (VA set
  A), identity flips on every composite arm (VA §3), row violations
  (R root cause). REJECTED as primary for persons; retained ONLY for the
  object-class meshless lane where the silhouette identities are the
  calibrated oracle (KB).
- **Clay-guided identity-route i2i** (current loop-2): the only local recipe
  that produced an operator-accepted result (e20 via `viewgen_fixed`), but
  its identity-preserving arm was measured on the REMOTE editor (L11) —
  banned in the one-shot person flow — and the local klein identity arm
  (L9: photo-primary + clay-reference) holds identity at the cost of face
  placement following the photo (+17 % mouth), repaired by the row-alignment
  + ≤ 2 % acceptance recipe at 4–6 draws/angle (VA §4).
- **LoRA-assisted local i2i** (NEW): unmeasured here — S2's bench owns the
  numbers. Placement is settled now: LoRA adapters are GENERATOR-ROUTE
  PARAMETERS of stage C on the local edit route, threaded through
  abstractvision's existing plumbing (verified present end-to-end:
  request-level `lora_adapters` → mflux `lora_paths/scales/target_roles`,
  adapter cache resolution includes the mflux cache). The two LoRA families
  target exactly the two measured local weaknesses: multi-angle LoRAs →
  the under-rotation class; consistency LoRA → the identity-flip class.
  What ANY LoRA route must GUARANTEE (identical to any generator): local
  routing, real recorded seeds, and passing the SAME §4 battery — LoRA
  changes the draw distribution, never the acceptance. S2's bench must
  report, per arm (LoRA × base × conditioning layout): pose-delta
  distribution (C1 ruler), identity margin (C7/C6), row verdicts (C3),
  draws-to-accept, wall-clock. The arm with the best accepted-views-per-
  minute at zero identity failures becomes the primary local generator.
- **Mesh-conditioned generation** (hy3dpaint lane, GP rank 1): views
  pixel-registered to the mesh by construction — the misregistration class
  cannot occur in that lane (GP axis 1, verified in the vendored source:
  its final stage is the same weighted back-projection architecture as our
  bake). Its identity preservation through DINO reference-attention is
  UNPROVEN (GP honest-risk note) and its memory at defaults exceeds the
  20 GB product profile (21–39 GB reported; knobs unmeasured). It requires
  a mesh, so it can never be the pre-mesh generator; inside the loop it is
  the strongest candidate for BOTH the texture-completion views and (via
  windowed copies) pass-2 conditioning views.

RULING:

1. **Primary architecture**: the two-pass clay-guided loop (scaffold →
   verified views → pass 2), ratified by e20 and by the field's precedents
   (IM-3D/Ouroboros3D/PSHuman all condition on a geometry hypothesis —
   GP axis 4).
2. **Primary view generator inside the loop**: hy3dpaint-conditioned views
   once W6's spike proves MPS + identity + memory (registration by
   construction beats any post-hoc gate); until then, the best LoRA-local
   arm from S2's bench; until THAT lands, the current L9-layout local
   recipe with the §4 battery and redraw budget. Each is acceptable only
   behind the same verification — the fallback ladder changes generators,
   never guarantees.
3. **Freestanding draws**: never for persons; objects keep them with the
   calibrated identity oracles.
4. **Remote editors**: never in the one-shot person flow (structural
   refusal stays); permissible only in operator-attended experiments with
   explicit attestation, recorded.

---

## 6. (design question 3) Geometry conditioning ruling

- **View count**: ≤ 3 tags, priority front > back > side. Evidence: the
  4-view cliff is a hard measured envelope (822/559 raw bodies at 4 views;
  every 1–3 subset healthy; 2–3 > 1 — KB). RATIFIED.
- **Window law**: HOLDS and is RATIFIED — cut every conditioning view
  (front included) to one anatomical window (head_top → shoulder +
  0.22·span) so bbox recentring puts the same anatomy at the same
  normalized rows. Evidence: anchor spread 15–33 % → 3.4 % (R wave 3 audit
  a); e18 = first striation-free result at raking light. REFINEMENT
  (folds backlog 0018): the window is computed ONCE from shared anchors
  and applied identically to all views of the set — per-view independent
  crops measurably re-break the side-pair mirror gate (e12 pair-drop).
- **Pose honesty**: RATIFIED as a two-sided contract. Conditioning: a view
  whose head-band-ruler azimuth is decisively > 20° off its declared tag
  may not condition (2mv tags are trained positions; the striation incident
  is the precedent). Texture: measured numeric azimuths ALWAYS (TF §4
  used @65/−67.5/180; the parser exists). The ruler is head-band-only, by
  law (full-bust IoU measured pose-blind in both directions — VA §1,
  TF §H2). Back views: the ruler sweep is flat (±0.04 IoU) — pose honesty
  for backs rests on C2's mirror identity instead; the abstention is
  recorded.
- **Split-consumer law**: RATIFIED as shipped (consumer ∈
  both|geometry|texture; windowed variants ride `geometry`, full-span ride
  `texture`; duplicate-angle guard raises on two synthesized refs at one
  pose — TF §5). Evidence: windowed twins in the bake caused 31–64 px
  vertical content disagreement over 220k–485k co-painted texels (TF §H1);
  windowed-only texture starves below the cut (e18 sides at 6 % coverage —
  R). One content class, two consumers, two variants, roles recorded.

---

## 7. (design question 4) Texture ruling

- **Projection bake stays the authority layer.** The bake architecture is
  what the field itself ships (hy3dpaint's final stage is the same weighted
  back-projection — GP axis 1); e20's registered lane reached 0.75 coverage;
  the corrected bake (fixB) removed the four marquee defect classes at the
  four inspected angles (TF §4: nose-lips smear GONE, lens ghost GONE,
  beard confined) — and IV's full orbit then REFUTED "fixed" in the strong
  form: center registration improved (IV: best single face crop of all 8)
  while blocker-grade collateral appeared elsewhere (jagged right-jaw seam,
  amber/teal lens garbage, cream neck/nape washes — IV e20_rebake #1–4).
  Both facts stand: registration was the right axis AND four-angle closeup
  verification is not verification — the full-orbit protocol (stage G) is
  the only inspection standard this strategy recognizes. Photo protection
  (protect_observed_texels absolute mode) is RATIFIED — the photo is the
  only non-synthetic evidence we own (KB 0017).
- **Source row registration is now a measured contract, not a doctrine**
  (the H4 fix productized): every bake measures source-vs-mesh feature-row
  agreement (front clay render vs source through the row law + feature
  anchors) and applies the feature-anchored piecewise correction only when
  decisive, with monotonicity asserted and post-warp anchor landing
  verified (TF §4 step 3: glasses-bottom landed 397 vs target 392). On
  single-view meshes the corrector must measure as a no-op — that is the
  regression guard for the lane where recenter-only is measured-correct.
- **hy3dpaint integration mode**: RATIFIED as **per-view images fed as
  synthesized references into OUR bake** (GP integration sketch, first
  mode), NOT as an opaque baked-atlas merge. Arguments: (a) per-view refs
  flow through the SAME §4 acceptance and the same authority machinery —
  one code path, per-view provenance, the duplicate-angle and
  split-consumer guards apply; (b) hy3dpaint's cameras are known by
  construction, so measured-azimuth honesty is free; (c) an atlas is a
  single take-it-or-leave-it layer whose failure semantics are coarse.
  The atlas-as-completion mode remains the named A/B (W6 acceptance) —
  if per-view extraction proves awkward in their pipeline, the A/B decides
  with orbit evidence, not taste. Failure semantics: any hy3dpaint failure
  → `#FALLBACK` to mirror/witnessed completion; paint views failing §4 are
  discarded per the floor-only surrender rule (KB).
- **Per-texel cross-view consistency weighting** (SyncMVD-class
  winner-take-most sharpening where views disagree) is ADOPTED as the
  residual-ghost killer (GP rank 4; MC §3c: averaging two mouths paints two
  mouths). Targets DISAGREEMENT defects: mottling, doubled soft edges,
  ghost content where two views both paint (fixB's cheek mottling — TF §4).
  SCOPE HONESTY (IV fold): it does NOT close IV blocker 1's boundary half —
  a projection-boundary seam is a step where one view's coverage ENDS, not
  a region where views disagree; the rebake's jagged jaw seam survived the
  shipped gradient-domain compositing (active on every multi-view bake,
  `texturing.py` compositing="auto"), so tone-domain leveling is also
  measured-insufficient for that class. Boundary continuity is its own
  contract (stage F f1) and its own work item (W11).

---

## 8. (design question 5) Evaluation gate

- **Rendering** (IV's instrument adopted as the baseline — it found every
  operator defect plus 80+ more): 12 azimuths at el 0 + 6 azimuths at
  el +20, clay AND textured, ≥ 1024 px; face crops at {0, ±30, ±60, ±90}
  clay + textured; PLUS 2048 px oblique raking-light closeups at the four
  oblique angles. Evidence: striations invisible at front/45 and at 256 px
  shipped twice (R wave 3 trigger; KB thumbnail law); raking 55° light
  reveals that class; crown/nape blockers only show at el +20 or az180
  (IV e20 #6–8, rebake #4); and TF §4's four-angle closeups declared a
  "fix" that IV's 36-render orbit refuted — partial orbits are how this
  project has been fooled three times now.
- **Worst-angle discipline**: the score of a bundle is its WORST angle,
  never its mean (IV's own grading rule: "one BLOCKER angle makes the
  model a BLOCKER") — every marquee defect was angle-local.
- **Source-vs-render agreement**: the SAME §4 battery, reused against
  `1_source/` on BOTH render sets: textured renders (texture identity)
  AND clay renders (shape identity — C4 contour anchors + concave-detail
  axis catch melt/deformation that silhouette and watertightness cannot:
  e11's mannequin face and e15's button nose are watertight single bodies,
  IV e11 #1 / e15 #4; KB "Silhouette IoU cannot rank same-silhouette
  melts"). Calibrated allowance: render-vs-photo carries a perfect-texture
  floor (shading/material gap — KB identity-gate law), so floors come from
  the labeled archive — which since the IV fold contains NO person-lane
  ACCEPT bundle (§4.4): floors separate the 95 labeled defect rows from
  the object fleet's clean views and the rebake's center-face crops.
- **Named render-side detector families** (the "calibrated detector set"
  is now enumerated; each is a §3 stage-F contract's verifier pointed at
  renders):
  d1 = §4 battery on clay + textured renders at every pose where channel
       domains are non-empty (abstentions recorded);
  d2 = palette-lawfulness of completion/fill regions (f2): per-material-
       region chroma-first distance vs the source part palette — catches
       cream/cyan/chrome washes AND in-palette-globally but out-of-REGION
       content (skin patches inside hair regions: IV e21 #6, e10 #5);
  d3 = seam-band step metric over view-boundary bands (f1; the
       `texture_qa` seam instrument promoted from harness to gate);
  d4 = material-purity check inside committed uniform-material regions
       (f3): dispersion/foreign-chroma within the committed band;
  d5 = duplication/ghost-structure flag (autocorrelation family — fires
       on repeated facial structure: ghost faces, doubled features; stays
       review-class: a firing d5 caps at degraded, never auto-accepts).
- **Self-consistency**: MVGBench-lite disjoint-bake comparison (split
  accepted refs into two subsets, bake twice, Lab ΔE percentile on
  co-covered texels — MC §4 metric 3, model-free and CI-cheap), RATIFIED.

**The single acceptance rule** (self-declared honestly; REVISED by the IV
fold — the pre-fold rule listed e20_rebake_fixed as an ACCEPT exemplar and
carried no detector that fires on its jaw seam or neck wash, i.e. it would
have accepted a bundle the verifier graded BLOCKER on three axes):

```
ACCEPTED  iff  (a) geometry health: watertight ∧ bodies==1 ∧ no shred signature
           ∧ (b) process integrity: every stage gate passed or degraded ON RECORD,
                 zero bypasses, bundle completeness audit green (H)
           ∧ (c) agreement: §4 battery vs 1_source ≥ calibrated floors on BOTH
                 clay and textured renders (shape identity + texture identity)
           ∧ (d) worst-angle: detectors d1–d4 silent on EVERY orbit/crop/closeup
                 render, d5 silent (a firing d5 caps at degraded)
DEGRADED  iff any of (a–d) fails but a product shipped  → exit 3, named reasons,
          evidence renders attached
FAILED    otherwise (nothing shippable)                  → named refusal
```

Calibration criterion (W9, binding): replayed over the eight archived
bundles, the rule must grade ALL EIGHT below ACCEPTED, and every IV
BLOCKER row must trip at least one named detector (the 95-row table is the
confusion-matrix corpus). A rule that passes any IV blocker row is wrong
by construction.

Honesty clause: ACCEPTED is a claim about the measured defect classes, and
"success" still belongs to the operator — every verdict ships with the
orbit evidence precisely so external validation is cheap (the repo's own
rule: success is validated by others, never self-declared; IV is the
living precedent — it refuted a self-declared fix within hours).

---

## 9. (design question 6) The one-shot flow

`photo in → accepted bundle out`, zero human intervention:

1. **A intake** → `1_source/` archived; person class ⇒ loop + local
   provider required (hard refusals if unconfigured — current loop
   contract, RATIFIED).
2. **B pass 1** front-only scaffold (512/50 validated bust knobs), health
   floors, seed recorded. Budget: 1 re-roll.
3. **C+D view round R1** against scaffold clay: for each of
   {back, side_left, side_right}: draw → despecular/tone → §4 battery →
   accept or redraw. Budget: 6 draws/angle (VA measured 4–6 needed);
   re-roll vs escalate per failure nature; every raw draw persisted to
   `2_views/raw/`. Angle exhausted ⇒ drop angle, record, continue.
4. **E pass 2** on the shared-window set (≤ 3 tags, pose-honest), health
   floors. Both-sides-dropped or all-angles-exhausted ⇒ single-view run
   (the known-good floor) with the degradation recorded.
5. **F texture**: source row-registration (measured, maybe no-op) →
   full-span refs at measured azimuths, one per angle, synthesized
   protection → completion (hy3dpaint when integrated, else mirror/auto)
   → consistency-weighted blend.
6. **G evaluation**: full orbit + closeups + battery + flags → verdict.
   On failure: **R2 contingency loop once** — regenerate views against the
   pass-2 clay (better hypothesis), recondition, rebake, re-evaluate. Then
   final verdict; degraded is a legal, loud outcome.
7. **H bundle**: layout emitted natively during the run (each stage writes
   its own artifacts at creation time — that is what makes provenance
   structural); `PROVENANCE.md` + `REPORT.md` auto-generated; verdict +
   exit code (0 accepted / 3 degraded / ≠0 failed).

Budgets (MPS discipline: one heavy job at a time, sequential): 2 DiT
passes ≈ 2 × ~25 min at 512/50 (+1 pass if R2 fires); i2i ≤ 6 draws ×
3 angles × ~2–5 min (klein 8/12-step measured; ≤ 24 draws hard cap incl.
R2); bake ~15 min class; renders + battery minutes. Envelope ≈ 2.5–4 h
worst case; every stage records `timings_s`.

---

## 10. REJECTED alternatives (each with the reason it lost)

| alternative | rejected because (evidence) |
|---|---|
| 4-view 2mv conditioning | shreds: 822/559 raw bodies vs healthy at 1–3 views (KB, /tmp/afix3) |
| freestanding meshless views for persons | 40–50° pose errors passed all gates; identity flips; row violations (VA set A, §3) |
| composite two-panel conditioning on local editors | flips person identity on EVERY arm — steps/guidance/seed/prompt immaterial (VA L1–L10) |
| remote default editor in the one-shot person flow | privacy + money leak via `owner=None`; placebo seeds; no reproducibility; billing hard limit hit mid-audit (VA §0, §4) |
| `strength` as a fidelity knob | dead on the flux2 edit route — byte-identical draws L1≡L2; mflux raises in edit mode (VA §0) |
| CFG guidance > 1 on klein edit | anatomy worse (+13→24 % mouth), identity worse, 5–10× cost (VA L5/L6) |
| more steps as a fidelity ladder | non-monotonic (L3 best, L4 worse at 3.5× cost); per-draw variance dominates (VA §3) |
| full-bust silhouette IoU as pose/registration instrument | pose-blind on busts in both directions (VA §1; TF §H2: refiner would keep the wrong 90°) |
| whole-frame affine row alignment | drags the torso, −0.11 IoU; piecewise face-zone alignment is the measured fix (VA §4) |
| recenter-only source registration on multiview meshes | +25–42 px misregistration painted the marquee defects; front view won 75–100 % of those texels (TF §1, §H4) |
| two synthesized refs at one declared angle | 31–64 px mutual content offset over 220k–485k texels; never legitimate (TF §H1; guard now raises) |
| bypassing failed gates to ship a run | e17 consumed gate-failed views → operator rejection; bypass is now structural-refusal + verdict cap (VA §0) |
| post-hoc vertex symmetrization | regresses dihedral +6…+51 % for −1…−11 % mirror gain (KB) |
| smoothness-weighted shape ranking | rewards melted candidates (KB / api.md shape-candidates note) |
| 256 px / front-only verification | shipped striated mouths twice; raking-light obliques are the standard (R wave 3) |
| partial-orbit acceptance of a texture fix (four-angle closeups) | TF §4's fixB verified clean at l30/l55/r30/r55 and IV's 36-render orbit then found blocker-grade collateral at az90/az180/az270/az330 (jaw seam, lens garbage, neck/nape washes — IV rebake #1–4); acceptance renders the FULL protocol or it is not acceptance |
| gradient-domain compositing as sufficient seam treatment | the rebake's jagged jaw seam shipped THROUGH active gradient-domain compositing (IV rebake #1; compositing="auto"→gradient_domain on multi-view bakes) — tone leveling cannot fix structural content misalignment at a projection boundary; boundary continuity is its own contract (f1/W11) |
| luminance-only fill flooring as the completion guard | the cream/cyan/chrome washes are CHROMA defects; KB's own law says L-based axes are chroma-blind, and `enforce_fill_luminance_floor` is L-only — palette lawfulness needs the chroma-first per-region mechanism (f2/W12) |
| per-texel blending on uniform glossy dark materials (lens treated like skin) | contaminated on all 8 of 8 models (IV blocker 3) — the general fix is consensus-gated material commitment (KB film-band law), not more blending (f3/W13) |
| Era3D / Wonder3D / Zero123++ adoption now | AGPL/NC licenses, CUDA-pinned deps, person identity through 256–512 px SD2-class bottleneck (MC §3e; GP axis 2) |
| TEXTure / Text2Tex / SyncMVD-as-code / TRELLIS.2 texture | CUDA-only dependency stacks (kaolin/pytorch3d/CV-CUDA/nvdiffrast) (GP axis 1) |
| FLAME/DECA-family as primary geometry | no glasses, no hair, no shoulders — the subject wears glasses (GP axis 2) |
| baked-atlas-as-completion as the DEFAULT hy3dpaint mode | opaque single-layer authority, coarse failure semantics, skips per-view §4 gates; stays as a named A/B only (GP sketch; §7) |
| naive standalone re-bake outside the runtime registration | measured garbage, 0.36 coverage (R split-consumer note) |
| bespoke per-defect detectors (mouth-open, nose-shape…) | operator-forbidden one-example rules; subsumed by §4's channel battery |

---

## 11. Adoption sequencing

**This week (no GPU contention, composes from existing pieces):**

1. W11a + W12a seam/palette FORENSICS on the shipped rebake (CPU-only,
   TF-instrument reruns) — the two unattributed blocker mechanisms; every
   texture fix downstream keys on their answers.
2. Bundle-native emission + completeness audit (W1) — structural, closes D5.
3. Verification battery v1 wiring (W3) from existing instruments (pose
   ruler, row law, mirror identities, contour gate) + the C5 covisible
   channel (the bake's projector + moderngl renderer already exist).
4. Identity-backbone spike (W4) on the L1–L11 labeled set — decides C7.
5. Source row-registration productization (W2) — CPU-verifiable, accepted
   on the FULL IV orbit (never on the fixing angles).
6. Fold S2's LoRA bench when it lands (W5) — pick the local generator.
7. hy3dpaint MPS spike (W6a — script exists, hours, load-time only).

**Next (GPU-bench gated or dependent on the forensics):**

8. W11b/c boundary continuity fix + seam gate; W12b/c palette-lawful
   completion + d2 detector; W13 uniform-material commitment (lens class).
9. hy3dpaint per-view integration behind §4 (W6b).
10. Consistency-weighted bake (W7 — disagreement defects only).
11. One-shot orchestrator + budgets + degradation ladder (W8) and the
    calibrated evaluation gate (W9 — requires d2–d4 to exist).
12. Flagship-2.1 MPS shred ladder (W10, backlog 0020) — better clay for
    every loop; parallel track.

---

## 12. Implementation plan (ordered; acceptance measurable by full-orbit evaluation)

| # | work item | acceptance criteria | size |
|---|---|---|---|
| W1 | Bundle-native emission (`1_source/ 2_views/ 3_model/ 4_renders/` + PROVENANCE.md written by the stages themselves) + completeness audit wired into the verdict | a fresh loop run yields a bundle where every accepted view maps to raw bytes + seed + route + per-view report; audit script green; audit failure caps verdict at degraded (test) | 1 d |
| W2 | Source→mesh row registration in the bake path (measure always, correct when decisive, no-op on single-view meshes, monotonicity + anchor-landing verification) | center-face registration ≥ fixB's (IV: best face crop of all 8 — that axis is proven and kept) with the four TF marquee defects absent at l30/r30 2048 px closeups; flagship-lane golden bakes byte-stable (no-op proof); AND no new d2/d3 flags anywhere on the FULL IV orbit protocol vs e20_fixed_views — the rebake's lesson is that this axis must be accepted on the whole orbit, never on the fixing angles (IV rebake #1–4) | 1–2 d |
| W3 | Same-subject battery v1: C1–C4 unified behind one per-view report + C5 covisible reprojection channel; single decision rule + abstention recording; wired ahead of BOTH consumers | labeled-archive separation: e18-cond parted-lips views REJECTED (C5/C6 fire), set A 50° views REJECTED (C1), set B low faces REJECTED/corrected-once (C3/C4), `viewgen_fixed/*` ACCEPTED; zero false rejects on the object fleet's accepted views | 2–3 d |
| W4 | Identity-backbone spike: ArcFace-class ONNX (onnxruntime) vs DINOv2 (transformers) on L1–L11 + e20 view sets (subject wears sunglasses — the margin must survive) | measured margin table; adopt C7 iff WRONG/RIGHT arms separate with zero false accepts; else C6-aggregate documented as the identity carrier | 0.5–1 d |
| W5 | LoRA route fold-in: thread `lora_adapters` through `reference_generation` requests (abstractvision plumbing verified present: request `lora_adapters` → mflux `lora_paths/scales/target_roles`); adopt S2's winning arm as primary local generator; record LoRA id+scale+hash in `2_views/` provenance. CACHE-ROOT GOTCHA (verified in source): mflux computes its cache via `platformdirs.user_cache_dir("mflux")` = `~/Library/Caches/mflux` on macOS, while abstractvision's adapter-cache fallback hardcodes `~/.cache/mflux` — the operator's cached LoRAs are invisible to abstractvision's resolution on this host unless `MFLUX_CACHE_DIR` is exported or absolute file paths are passed; W5 must fix the default to platformdirs or pass paths explicitly | on S2's bench protocol: accepted-views-per-minute ≥ current local recipe at zero identity failures, pose deltas within the C1 gate, real seeds (byte-reproducible draws); no remote calls (test); a LoRA resolved from the real macOS cache without env overrides (test) | 0.5–1 d after S2 |
| W6 | hy3dpaint lane: (a) MPS spike (build extensions, load to mps fp16, RSS report — script exists); (b) per-view integration: its views enter as synthesized refs through §4 + split-consumer; A/B vs atlas-completion mode; memory knobs (4 views / 1024 render / 2048 atlas / SR off) measured | (a) pipeline loads under 20 GB RSS at reduced knobs; (b) orbit A/B vs mirror completion: rear-flank + glasses-ghost residuals reduced at 2048 px closeups, front photo texels byte-protected (test), identity spot-check on the face crop passes §4 | spike: hours; integration: 2–4 d |
| W7 | Consistency-weighted bake: per-view agreement score (C3 row displacement + C5 mesh-mediated reprojection) multiplying the facing law, winner-take-most sharpening where views DISAGREE (scope: disagreement defects only — boundary seams are W11's, fill palette is W12's) | doubled/ghost content from co-painting views (IV e20 #2 doubled arm band, e11 #8 doubled glasses, cheek mottling) measurably reduced on the full IV orbit; fleet golden bakes non-regressed (existing A/B harness) | 2–3 d |
| W8 | One-shot orchestrator: budgets (6 draws/angle, ≤ 24 total, R2 once), degradation ladder (drop angle → fewer tags → single-view floor), verdict + exit codes, wall-clock envelope recorded | single command on the laurent photo → accepted-or-degraded bundle with zero prompts; kill tests: exhausted angle degrades on record; forced gate-block degrades instead of bypassing (test); envelope ≤ 4 h logged | 2–3 d |
| W9 | Evaluation gate: the IV render protocol (12×2 orbit + face crops + oblique closeups, clay + tex) + §8 rule with detectors d1–d5; floors calibrated on the labeled archive | replayed over ALL EIGHT archived bundles: none grades ACCEPTED, and every IV BLOCKER row trips ≥1 named detector (95-row confusion table in the report, per-row detector attribution); zero d1–d4 fires on the object fleet's accepted bundles (false-fire floor) | 1–2 d |
| W10 | Flagship-2.1 MPS shred ladder (backlog 0020: contiguous+math SDPA, torch 2.9 scratch venv, bf16, fp32 qk_norm) | one green flagship run at 384/30 (better clay for every loop) or a pinned version gate with the failing kernel path named | 1–2 d |
| W11 | Boundary continuity (IV blocker 1, boundary half): (a) FORENSIC first — TF §2-style per-texel attribution of the rebake's right-jaw seam (weight-step at a visibility boundary vs content disagreement between front and side at the boundary vs row-warp-induced shift of the boundary; the mechanism is NOT yet attributed and the fix depends on it); (b) fix per mechanism (incidence-feathered handoff into completion/neighbor content at every view's coverage edge; warp-aware boundary recompute if (c) is the cause); (c) promote the `texture_qa` seam-band step metric from harness to bake gate (d3), allowance recalibrated on IV's labeled seam rows | (a) attribution report with winner-share numbers for the seam band; (b) e20-geometry re-bake shows no seam-band step above allowance at ANY of the 36 IV poses while center-face registration stays ≥ fixB; (c) d3 fires on shipped e20_rebake_fixed #1 and e21 #5, silent on object-fleet goldens | 2–3 d |
| W12 | Palette-lawful completion (IV blocker 2): (a) FORENSIC first — attribute the cream/cyan/chrome content's source per defect (reference-matte background contamination vs fill/inpaint propagating contaminated anchors vs source-photo background at grazing incidence; KB "completion stages propagate bad anchors" predicts the anchor route but IV rows span multiple lanes); (b) donor-constrained fill: completion texels draw statistics only from like-material observed donors (the KB fill-law family: "fill must reproduce material statistics", "synthetic cut faces take their rim's tone"), out-of-palette anchors excluded at the source; (c) render-side d2 detector: per-material-region chroma-first distance of completion regions vs the source part palette (reuses the `part_material_fidelity` machinery — chroma-first with L tolerance, exists in code) | (a) per-defect source attribution for IV rebake #3/#4, e20_fixed #3/#7, e10 #1, e18 #6/#10; (b) corrected e20-geometry bake: zero d2 fires over the full orbit, neck/nape/behind-ear regions read skin/hair/shirt statistics; (c) d2 fires on all six listed shipped rows, zero fires on object-fleet goldens (multi-material subjects are the false-fire trap) | 2–3 d |
| W13 | Uniform-material commitment (IV blocker 3, the lens class made general): segment uniform dark/glossy material regions from the SOURCE PHOTO's part palette (k-means LAB parts — exists) mapped onto the mesh via the registered front projection; commit a region to its consensus material statistics ONLY under the KB commitment conditions (all witnessing views flag-consensus, no positive base-material witness, ≥2 witnesses, dark-dominated context — the film-band law verbatim, so the "third-eye" false-commit class stays guarded); witnessed speculars inside committed regions reconcile by diffuse consensus (existing KB specular-lobe law); d4 purity check inside committed regions | on e20-geometry bakes: lenses render as uniform opaque dark material at every IV face-crop pose (no skin/amber/ghost-eye/landscape content — the classes of IV rebake #2/#9, e20 #1's translucent band, e21 #2, e17 #7/#8, e11 #8, e15 #1); no commitment fires on any non-consensus region (single-witness/contested test — the KB third-eye guard); generalization: an object-fleet subject with a uniform glossy dark part commits correctly, subjects without one are provable no-ops | 2–3 d |

Dependency notes: W3 precedes W5/W6b (routes are judged by the battery);
W1 precedes W8/W9 (verdicts require the audit); W4 feeds W3's C7 slot;
W11/W12's forensic halves (a) are CPU-only and independent — they can run
immediately; their fix halves and W13 precede W9's final calibration (the
detectors d2–d4 must exist before the gate that names them); W10 is
parallel and independent. The e20 mesh is the standing test geometry for
W2/W7/W11/W12/W13 (IV mesh rank 1, no mesh blocker).

---

## 13. Risks not closed (named experiments, no guesses)

1. **Identity verification under sunglasses at profile poses** — face
   embeddings may not separate; W4's labeled-set spike decides, and the
   honest fallback (C6 aggregate) is weaker against subtle drift. Until
   then identity at grazing poses rests on C5+C6+C8 consensus.
2. **hy3dpaint identity through DINO reference-attention** is unproven for
   real faces (GP risk note), and its memory envelope at reduced knobs is
   unmeasured — W6a/b measure both; the photo-protection authority layer
   bounds the blast radius to never-photo-observed texels either way.
3. **LoRA bench outcome unknown** (S2's lane): if no local arm passes §4 at
   acceptable draw budgets, the one-shot person flow keeps the current L9
   recipe (slow, 4–6 draws/angle) and its acceptance rate is the pipeline's
   throughput bound.
4. **Monocular depth ambiguity** (e.g. true nose protrusion) is
   unverifiable from one photo where no accepted view constrains it —
   bounded by consensus and the loop, not eliminated.
5. **The palette+relief-preserving semantic re-render class** ("sculpted
   goo") has no known gate (KB); generator quality is the only measured
   countermeasure.
6. **Gate-calibration drift**: every §4 floor is only as good as the
   labeled archive; the archive must grow with every operator-labeled
   accept/reject (the 0019 speculars miscalibration is the precedent for
   what happens when it does not).
7. **In-palette displaced content at weak-covisibility angles** (IV fold):
   a defect like e17 #6's black ghost drip on the beard is palette-lawful
   (beard-dark), boundary-free, and sits where the source photo's
   covisibility is thin — d2/d3 stay silent, d1's C5 has thin domain, and
   d5 only fires on repeated structure. This is the acceptance rule's
   weakest axis; mitigation is bake-side prevention (W7 winner-take-most +
   W12 donor constraints) rather than render-side detection. Named
   experiment: after W12, replay d1–d5 over the e17 rows and measure which
   of its six texture blockers still evade all detectors — that residue
   defines the next detector's requirement, from evidence.
8. **No positive person-lane exemplar exists** (IV verdict: all 8 fail at
   blocker grade): every calibration positive is an object-fleet bundle, a
   synthetic probe, or a center-face crop. The first person-lane ACCEPT
   the revised rule produces must go to the operator for validation before
   it is trusted as a calibration anchor — self-declared success is
   structurally banned (and IV just demonstrated why, refuting fixB within
   hours of its "fixed" claim).

---

## 14. Independent verification fold (2026-07-21)

`docs/research/independent_verification.md` (IV) landed while this
document was being written: an independent full-orbit inspection of all 8
candidate models (36 renders + 14 face crops each, 95-row defect table,
worst-view grading), verdict "no bundle is presentable — all eight fail at
blocker grade", with the e20_rebake_fixed "fixed texture" claim REFUTED in
the strong form. This section records what the fold changed and what was
already covered, blocker by blocker.

### Blocker 1 — registration/projection boundary damage

- ALREADY COVERED (registration half): stage F's measured source-row
  registration contract + W2. IV independently confirms the axis was right
  ("the rebake's direction is real — its center face is the best single
  face crop across all 8 models").
- NOT COVERED (boundary half) — ADDED: the rebake's jagged right-jaw seam
  (IV rebake #1) shipped through BOTH existing mechanisms this document had
  relied on: gradient-domain compositing was active (multi-view default)
  and W7's consistency weighting is scoped to view DISAGREEMENT, not to
  the step where one view's coverage ends. New contract f1 (boundary
  continuity + seam-band gate d3, promoted from the `texture_qa` harness)
  and W11, whose first half is forensic because the seam's mechanism
  (weight step vs boundary content disagreement vs row-warp-induced) is
  not yet attributed — TF §4 recorded the streak only as an "inherited
  class" residual; IV showed the rebake made it stronger.

### Blocker 2 — occlusion fill pulls out-of-palette background

- NOT COVERED — ADDED. Honest accounting of why the old text missed it:
  the fill guards in code are luminance-only (`enforce_fill_luminance_
  floor`), and KB's own law records that L-based axes are chroma-blind;
  the §4 battery gates VIEWS, and no channel inspects completion texels.
  New contract f2 (palette-lawful completion: donor-constrained fill from
  like-material observed statistics — the KB fill-law family — plus the
  per-material-region chroma-first detector d2 built on the existing
  `part_material_fidelity` machinery) and W12, forensic-first (the
  cream/cyan/chrome sources span candidate routes: matte contamination,
  contaminated fill anchors, grazing background — attribution before fix).
  d2's per-REGION form also covers the in-palette-globally/out-of-region
  family (skin patches embedded in hair: IV e21 #6, e10 #5).

### Blocker 3 — lens material purity

- PARTIALLY ADJACENT, NOT COVERED — ADDED. W7 was tagged with "lens-edge
  reflection content" but winner-take-most still paints SOME view's lens
  content, which on 8 of 8 models is contaminated. The general mechanism
  (not a sunglasses detector) already exists as KB law from the hair-film
  class: consensus-gated uniform-material commitment (all witnesses agree,
  no positive base-material witness, ≥2 witnesses, dark context) — the
  lens is an instance of the uniform dark/glossy material class, same as
  film bands and glazed parts. New contract f3 + d4 purity check + W13,
  with the KB "third-eye" false-commit guard and an object-fleet
  generalization criterion so the mechanism is provably not bespoke.

### Acceptance rule — changed, and the old one was provably wrong

The pre-fold rule cited `e20_rebake_fixed` as W9's ACCEPT exemplar and
its worst-angle clause named no detector that fires on a boundary seam or
a palette wash: it would have ACCEPTED a bundle IV grades BLOCKER on three
axes. Changes: (c) now runs the battery on clay AND textured renders
(melted-but-watertight meshes — IV e11 #1, e15 #4 — pass rule (a) and are
only caught by shape-identity channels); (d) enumerates detector families
d1–d5 instead of "calibrated detector set"; W9's binding criterion is now
the 95-row confusion table (all eight bundles must grade below ACCEPTED;
every BLOCKER row must trip a named detector); `e20_rebake_fixed` moved
from labeled PASS to labeled FAIL (center-face crops remain positive
exemplars for the registration axis only); the render protocol adopts
IV's instrument (el +20 ring and az180 are where crown/nape blockers
live). Known residual weakness stated in Risks §7: in-palette,
boundary-free displaced content at weak-covisibility angles (e17 #6
class) can evade d1–d5; prevention is bake-side, and the post-W12 replay
over e17's rows is the named experiment that measures the residue.

### Also folded

- e10 orientation fact corrected (§1): the smear is on the subject's
  RIGHT, az240–330 (IV operator cross-check).
- §4.4's labeled archive now includes the full IV table (the richest
  labeled corpus the project has); the "no person-lane ACCEPT exists"
  consequence is recorded there and in Risks §8.
- Rejected-alternatives gains four rows (partial-orbit acceptance,
  gradient-domain-as-seam-fix, L-only fill flooring, per-texel blending on
  uniform glossy materials).
- e20's mesh is confirmed the standing test geometry (IV mesh rank 1, no
  mesh blocker); its crown dents at el +20 and the shoulder sash stay
  secondary (IV's own call), queued behind the texture blockers.
