# Viewgen audit — are the synthesized angle views anatomically consistent? (2026-07-21)

Adversarial audit of the GENERATION half of the bust-striation incident: the
operator rejected e11/e15/e17 reconstructions (striated multi-ridge mouths,
one bulbous nose profile). A parallel audit covers the registration math;
this one measures whether the synthesized side/back views themselves are
anatomically consistent with the front photo and with the clay guides that
were supposed to govern them, and what generation recipe fixes them.

Everything below is measured from the shipped artifacts. Tools:
`scripts/experimental/viewgen_audit_{measure,analyze,ladder,ladder2,regen,reprocess,remeasure}.py`,
raw numbers under `/tmp/viewgen_audit/` (annotated per-image overlays in
`/tmp/viewgen_audit/overlays/` — every landmark row was verified against the
pixels before the tables below were trusted).

## 0. Provenance corrections (what actually made these pixels)

The audit brief's framing ("clay-guided i2i, composite conditioning, steps
8/12") describes only ONE of the two sets. Verified from bundle metadata,
code paths, and a live reproduction:

- **Set A** (`hunyuan_multiview/texture_reference_generated_{side_left,side_right,back}.png`,
  consumed by e11/e15 as texture references and, as the same raw payloads,
  by the 2mv geometry conditioning): the pixels came from the MESHLESS
  "rotate" lane (`_synthesize_geometry_views`, prompt "Rotate the camera to
  show this exact man wearing blind face…", seeds 52025/53025/54025, steps 8,
  mlx-gen `flux.2-klein-9b-8bit`) and were only REPLAYED through the texture
  acceptance machinery (`replayed_labels: [back, side_left, side_right]`;
  per-angle "generation" times 3.4–3.9 s = no i2i ran). **The clay guides
  never conditioned these pixels** — they only gated them post-hoc
  (registration + full-bust IoU 0.75). Only the `top` view (316 s, 2
  attempts) was really clay-conditioned.
- **Set B** (`loop_views_e11/loop_view_*.png`, consumed by e17 and e13): the
  true clay-guided composite lane (clay from the e11 mesh) — but through the
  **default provider route**, not the local klein. `regen_views_from_mesh.py`
  passes `owner=None`, so `resolve_image_generation_request` returns `{}`
  (the recorded `image_request` in `regen_report.json` is empty) and the
  abstractvision capability resolves its configured default:
  **openai / gpt-image-1** (remote). Evidence: the binding resolution
  reproduced live (`_resolve_backend_binding(provider=None, model=None)` →
  `openai / gpt-image-1`); generation time 31.3 s vs 132 s for a local
  klein 8-step at the same canvas; ladder arm L11 (same empty-request call)
  returned a 1024×1024 asset in 37 s with gpt-image-1-class identity
  preservation, where every local-klein arm flips the person (§3).
  Consequences: the recorded seeds/steps in `regen_report.json` were
  **placebo** (the OpenAI edit path sends neither), and the loop script
  silently sent a real person's photo (and money) to a remote provider —
  `auto_generation_ready` exists to refuse exactly this, and `owner=None`
  scripting bypasses it.
  - Only `side_left` passed the gates in the recorded run (accepted at IoU
    0.7773 after `scale 0.88` registration). `side_right` (3× silhouette
    fail ≤ 0.734) and `back` (2× speculars fail — the 0019 miscalibration —
    +1 silhouette fail) were REJECTED; the files on disk with mtimes 6–8
    minutes after the report are a later gate-bypassed rerun; e17 consumed
    them anyway.
- **`strength` is a dead knob on the production route.** Verified at three
  layers: `resolve_image_generation_request` never emits it; the
  abstractvision mlx backend's flux2 edit branch never forwards
  `image_strength` into the mflux call (read: `_edit_image_impl`, flux2
  branch builds `kwargs` without it); and mflux's `Flux2KleinEdit.generate_image`
  RAISES if it ever arrives ("image_strength is only supported for latent
  image-to-image mode, not edit-reference mode"). Empirical proof: ladder
  arms L1 (steps 8) and L2 (steps 8 + `strength=0.5`), same seed, produced
  **byte-identical payloads** (md5 `08f195a690fb29470ad718acae440f9b` both).
  FLUX.2-klein edit-reference mode has NO denoise-strength axis at all —
  generation always runs full denoise from seed noise with the reference as
  attention tokens. The "denoise strength too high/too low" hypothesis is
  structurally unanswerable on this route; the real fidelity axes are the
  conditioning layout, steps, guidance, prompt, and acceptance/redraw.

## 1. Anatomy measurement

Subject-agnostic landmark extractor (no face detector), per image:
top-of-head, shoulder flare, glasses band (dark-band-with-skin-below on
photos/generations; leading-edge protrusion shelf on clays), and the profile
leading-edge extrema nose_tip → nose_base → mouth (lip bump) → chin. Front
photo: glasses band, width-minimum chin (row 580; visually the beard bottom
~578), horizontal-edge nose_base/mouth (low confidence, excluded from
acceptance math). Two normalizations:

- **registered rows / clay head height** — both images share the clay's
  frame (the shipped views ARE registered onto their clays), so
  `(view_row − clay_row) / clay_head_height` is the axis that catches
  whole-face shifts; this is what the texture projector and the 2mv
  row law actually feel. Head height = top-of-head → chin.
- **fraction of own head height** vs the front photo — crop-invariant
  internal proportions (the front photo is cropped mid-torso, so
  subject-height fractions are not comparable across these images).

Effective head azimuth is estimated by sweeping the set's own mesh clay
through candidate azimuths, registering the view onto each
(`register_matte_to_clay`), and scoring **head-band** IoU (rows above the
clay's shoulder flare). Full-bust IoU is pose-blind on busts — measured: a
~50° three-quarter view scores 0.76–0.77 full-bust IoU against a 90° profile
clay, comfortably above the production 0.75 gate, because the torso
dominates the silhouette.

Caveats owned: the glasses feature compares a photometric extractor against
a geometric one (±3–5% cross-extractor bias — reported, never gated);
`chin` on profiles is the beard-front bump (beard-inclusive, exactly like
the clay's carved beard); the pose ruler is meaningful on profiles (sweep
contrast ~0.15 IoU) and nearly flat on backs (±0.04 over ±50°, so back
estimates are noise).

## 2. The tables

### Set A (rotate-lane views replayed as texture refs; e11/e15)

Offsets = registered rows / clay head height (positive = view feature BELOW
the clay's). Clay = the set's own `*_clay.png` (hunyuan_multiview mesh).

| view | glasses | nose_tip | nose_base | mouth | chin | est. azimuth (nominal) | head-band IoU est/nominal |
|---|---|---|---|---|---|---|---|
| side_left | −12.5% | −1.5% | −4.1% | −0.9% | −6.4% | **+50.0° (+90°)** | 0.892 / 0.771 |
| side_right | −12.8% | −2.3% | −5.5% | −4.1% | −8.7% | **−52.5° (−90°)** | 0.916 / 0.769 |
| back | n/a | n/a | n/a | n/a | n/a | +185° (+180°) | 0.809 / 0.805 |

vs the front photo (fraction of own head height; internal proportions):
side_left nose_base +0.5%, mouth +4.2%; side_right nose_base +0.7%, mouth
+2.8%; glasses +8.0/+8.1% (cross-extractor bias applies).

**Verdict on set A: the anatomy is roughly right but the POSE is wrong by
~40°.** Both "profiles" are three-quarter views (50°/−52.5°). The rotate
prompt asks for "left side profile"; the model under-rotates and no gate
measures pose (full-bust IoU passes at 0.76+). Feature-row offsets vs clay
(−1 to −9%) are secondary to this: when a 50° view is baked/conditioned AS a
90° view, mouth and nose content is projected onto profile texels that
belong to cheek/jaw geometry — mouth ghosting and the bulbous-nose class on
any arm consuming these views (e11, e15 texture; the 2mv conditioning of the
whole redo family consumed the same raw payloads as
`geometry_view_synthesized_*`).

### Set B (clay-guided composite via the default remote editor; e17, e13)

Clay = fresh renders of the e11 mesh (same code path as the loop run;
`/tmp/viewgen_audit/e11_clays/`).

| view | glasses | nose_tip | nose_base | mouth | chin | est. azimuth (nominal) | head-band IoU est/nominal |
|---|---|---|---|---|---|---|---|
| side_left | +11.5% | +10.9% | +4.5% | +6.4% | +16.0% | +80° (+90°) | 0.814 / 0.780 |
| side_right | +21.8% | +24.0% | +18.6% | +22.1% | +20.8% | −97.5° (−90°) | 0.777 / 0.745 |
| back | n/a | n/a | n/a | n/a | n/a | +180° (+180°) | 0.899 / 0.899 |

vs the front photo (fraction of own head height): side_left nose_base
−8.1%, mouth −7.6% (compressed mid-face); side_right −0.4%, +1.5%
(internal proportions right).

Decomposition into rigid shift + residual distortion (mean over
nose_tip/nose_base/mouth/chin, then residuals):

| view | mean shift | residuals after removing the shift |
|---|---|---|
| A/side_left | −3.2% | ntip +1.7, nbase −0.9, mouth +2.3, chin −3.2 |
| A/side_right | −5.1% | ntip +2.8, nbase −0.4, mouth +1.1, chin −3.6 |
| B/side_left | **+9.5%** | ntip +1.4, nbase −5.0, mouth −3.0, **chin +6.6** |
| B/side_right | **+21.4%** | ntip +2.6, nbase −2.8, mouth +0.7, chin −0.6 |

**Verdict on set B: pose is held (composite conditioning works for pose)
but the face sits 9.5–21.4% of head height LOW in the clay frame.**
side_right is almost a PURE shift (residuals ≤ 2.8%): the editor drew
correct internal proportions in a different framing, and the whole-bust
scale+shift registration (which optimizes torso-dominated IoU) left the
face 65 px low — every mouth texel it contributes lands on chin/neck
geometry, and as loop conditioning it hands the 2mv DiT a second mouth row.
side_left carries a real distortion on top (chin +6.6% residual after the
shift — a longer jaw/beard than the mesh). The e17 mouth striation is these
two views, quantified. Corroboration from the shipped row gate
(`view_consistency_report`, each view vs the front photo): all six A/B
views verdict `correctable` with row-profile score 0.00 at lawful rows and
best remap shifts of −98…−164 rows on a 1000-row frame.

## 3. Recipe ladder (side_left, e11 clay guide, seed 11 unless noted)

Local route = mlx-gen `AbstractFramework/flux.2-klein-9b-8bit` (the set A
explicit route), production composite canvas (photo | darkened clay,
1536×768), production negative prompt (dropped by klein — no CFG at
guidance 1.0). Offsets = registered rows / clay head height after
production registration. chromaR = generated foreground mean chroma /
photo's (photo = 5.8; 1.0 = tone-faithful; the clay-gray failure mode would
read ≪ 1 — it never occurred; every arm overshoots chroma instead).

| arm | knobs | iou | est. az | glasses | ntip | nbase | mouth | chin | chromaR | sec | person |
|---|---|---|---|---|---|---|---|---|---|---|---|
| L1 | steps 8 | 0.901 | 95.0° | −1.0% | −5.8% | −4.5% | −2.6% | −8.6% | 1.95 | 147 | **WRONG** (blond, aged, thin beard) |
| L2 | steps 8 + `strength=0.5` | — | — | — | — | — | — | — | — | 149 | **byte-identical to L1** (knob dead) |
| L3 | steps 12 | 0.904 | 95.0° | +3.9% | −0.6% | +0.6% | +2.9% | −3.9% | 1.96 | 286 | WRONG |
| L4 | steps 16 | 0.905 | 95.0° | +0.3% | −4.8% | −2.9% | −1.9% | −7.7% | 1.99 | 1017 | WRONG |
| L5 | steps 12, guidance 2.5 | 0.904 | 87.5° | +11.2% | +11.9% | +13.5% | +23.7% | +19.9% | 2.05 | 1537 | WRONG (worse) |
| L6 | steps 12, guidance 4.0 | 0.888 | 95.0° | +10.3% | +2.6% | +4.5% | +13.8% | +9.9% | 2.18 | 1789 | WRONG |
| L7 | rotate conditioning, steps 8 | 0.779 | **77.5°** | +10.6% | +8.6% | +12.2% | +19.2% | +16.4% | 1.41 | 753 | close (dark curls) |
| L8 | steps 12 + identity/proportion clause | 0.905 | 95.0° | −1.0% | −5.1% | −3.9% | −1.9% | −9.0% | 1.91 | 1570 | WRONG |
| L9 | separate refs: photo primary + clay as 2nd reference, steps 12 | 0.841 | 82.5° | +8.6% | +3.2% | +5.8% | +17.0% | +13.8% | **1.26** | 1064 | **RIGHT** |
| L10 | composite, seed 1011, steps 12 | 0.906 | 92.5° | +3.2% | −1.3% | −0.3% | +10.9% | +6.4% | 2.20 | 862 | WRONG (flip is systematic, not seed luck) |
| L11 | default route (empty request = the set B call), "steps 8" | 0.767 | 82.5° | +17.6% | +14.4% | +20.5% | +32.0% | +31.1% | 1.67 | **37** | **RIGHT** (gpt-image-1) |

Ladder findings:

1. **`strength` verified dead** (L1≡L2). No denoise-strength axis exists on
   the flux2 edit-reference route; do not add a passthrough knob for it.
2. **Steps are non-monotonic for anatomy** (L3 best, L4 worse than L3 and
   3.5× the cost under MPS memory pressure) — consistent with the recorded
   8→12 both-directions finding; per-draw variance dominates steps.
3. **CFG guidance hurts** (L5/L6): anatomy worse (+13→24% mouth), identity
   worse, 5–10× slower (two transformer passes/step + MPS pressure). Keep
   klein at its distilled guidance 1.0.
4. **The composite canvas itself destroys identity on the local editor**:
   every composite arm (any steps/guidance/seed/prompt) painted a DIFFERENT
   person — the photo occupies ~a third of one half-panel and klein-9b
   regenerates the whole canvas at full denoise from its prior. The person
   clause (present in all these prompts) and the L8 proportion clause do
   not rescue it. This is exactly the identity failure `is_person_subject`
   / `person_policy="skip"` exists for; the bust work overrode it with the
   acknowledgment and got the warned-about outcome.
5. **Separate references restore identity locally** (L9: photo as primary
   edit image, clay as a second `reference_images` entry — both reach the
   model full-resolution): right person, best tone (chromaR 1.26), pose
   82.5°. But the face placement then follows the photo, not the clay
   (+17% mouth) — identity adherence and clay-geometry adherence TRADE OFF
   across every route tested.
6. **The remote default editor** (L11 = set B's actual route) preserves
   identity and costs 37 s, but places the face up to 32% low pre-alignment
   and reframes freely (raw sizes 1024×1024/1536×1024 vary per draw, no
   seed control server-side).
7. Photo-realism: no arm went clay-gray (min chromaR 1.26); the failure
   direction is the opposite (over-chromatic studio look, chromaR ~2 on
   composite arms) — the production capped LAB tone match handles this and
   was kept in the final recipe.

## 4. The chosen recipe and the regenerated set

No raw generation on ANY route passes a <2%-per-feature line against the
clay guide. The winning recipe composes the identity-preserving route with
a feature-anchored alignment and an explicit acceptance gate:

1. **Generate** via the identity-preserving route (the configured default
   editor — the same one set B actually used), production composite canvas
   + person clause + proportion clause, N draws per angle (draw variance is
   the acceptance bottleneck: 4–6 draws needed per angle in practice).
2. **Register** with the production `register_matte_to_clay`; when the
   draw framing is tight, a finer local search (scales 0.80–0.95, shift
   step 0.01) recovers draws the coarse 0.02-step search leaves ~0.007 IoU
   under the gate.
3. **Row-align faces onto the clay's rows**: least-squares
   `view_row = a·clay_row + b` over detected nose_tip/nose_base/mouth/chin
   (slope clamped 0.78–1.35), applied PIECEWISE — affine in the face zone
   (above chin + 10% head), identity below the clay shoulder, cosine blend
   through the neck, monotonicity asserted (fold ⇒ refuse). Iterated up to
   3× (contour re-detection moves a few px per pass). Whole-frame affine
   was measured first and rejected: it dragged the torso off and cost
   −0.11 IoU. Back views: two-point (crown, shoulder) fit.
4. **Accept** only when every measurable facial feature (nose_tip,
   nose_base, mouth, chin; glasses reported but not gated —
   cross-extractor bias) sits within **2% of clay head height** AND
   post-alignment silhouette IoU ≥ 0.75 (the production line). Back views:
   crown row < 2% + IoU ≥ 0.75. Otherwise redraw.
5. Production specular suppression + capped LAB tone match, THEN measure
   the exact shipped pixels.

Shipped: `out/laurent-bust-redo/viewgen_fixed/{side_left,side_right,back}.png`
+ `viewgen_fixed_report.json` (all offsets, poses, alignment params, raw
md5s). All three accepted by the gate:

| view | worst feature offset | measured features (frac of clay head height) | IoU after align | est. azimuth |
|---|---|---|---|---|
| side_left | **1.6%** | ntip −1.6, nbase +1.3, mouth +1.6, chin −1.6 | 0.806 | 77.5° (nominal 90; iou@est 0.938 vs 0.862 @90) |
| side_right | **0.96%** | ntip +0.6, nbase −1.0, mouth +0.6, chin 0.0 | 0.754 | −102.5° (nominal −90; 0.805 vs 0.757 @−90) |
| back | **0.0% (crown)** | no facial features from behind | 0.827 | sweep flat (±0.04 IoU over ±50°) — estimate not meaningful |

Honest residuals, on the record: (a) the pose ruler still reads the shipped
profiles 12° inside/outside true profile — the remote editor under-rotates
like every other arm; at these magnitudes the head-band IoU gap vs nominal
is 0.05–0.08, far smaller than set A's 0.12–0.15 at 40°, and the feature
rows (what the striation mechanism feels) are gated; (b) the shipped back
carries a small hallucinated strap/temple strip left of the neck (from the
photo's headset cable; ~sub-2% of foreground, lands on hair/neck texels);
(c) the remote route has no seed control — the report records raw md5s, but
exact reproduction of a draw is impossible (redraw-until-accepted is the
contract instead); (d) mid-run the provider hit its billing hard limit —
side_right shipped from a persisted earlier raw draw reprocessed offline
(every draw is persisted precisely because draws are unrecoverable).

## 5. Prompt findings

- The composite template (`_view_prompt`) pins material/color/pose/framing
  and (for persons) same-person/age/skin/hair — but contains NO proportion
  language. The rotate template says "same proportions and shape" but the
  meshless lane's real failure is POSE (40° under-rotation), which its
  wording ("seen from its left side profile… fully visible and centered")
  does not fix.
- Measured: adding an explicit proportion clause ("eyebrows, nose base,
  mouth line and chin at exactly the same heights… do not enlarge or shrink
  any facial feature", L8) changed nothing measurable on the local
  composite (offsets within noise of L1/L3, person still flipped). Prompt
  wording cannot rescue a conditioning layout the editor ignores.
- The identity-preserving arms (L9 separate refs, default route) DO benefit
  from identity/proportion wording in the sense that it costs nothing and
  the shipped set B side_left (right person) carried the person clause; but
  the decisive variables are the conditioning LAYOUT (separate refs vs
  composite) and the ROUTE, not the words.
- Shipped knob: `generate_reference_views(..., prompt_suffix=...)` — the
  smallest change that lets operators run prompt experiments (this audit
  needed a code edit to test L8) without forking the material-free
  template. Recorded in the report (`report["prompt_suffix"]`, and the
  per-angle `prompt` field carries the final text). Test:
  `test_generate_reference_views_prompt_suffix_slot`.

## 6. Blame verdict: generation vs registration

Both, in different proportions per set — with numbers:

- **Set A (e11/e15): generation-dominant.** The side views are ~50°
  three-quarter views sold as 90° profiles (40° pose error, the rotate
  lane's under-rotation). No registration can fix a wrong pose; the
  registration lane's contribution is that its acceptance instrument
  (full-bust IoU 0.75 + whole-silhouette scale/shift) is structurally
  UNABLE to see the pose error (0.76–0.77 at 40° off). Feature-row offsets
  after registration are moderate (−1…−9%).
- **Set B (e17): shared, tilted toward registration for side_right and
  toward generation for side_left.** side_right: internal anatomy correct
  (residuals ≤ 2.8%) but the whole face 21.4% low after whole-bust
  registration — a head-weighted or feature-anchored registration absorbs
  almost all of it (the audit's alignment took the same draw class to
  ≤ 1%). side_left: 9.5% shift (registration-absorbable) PLUS a real
  +6.6% chin/jaw elongation (generation's). Additionally, side_right and
  back shipped to e17 despite FAILED gates (bypass) — a process failure
  ahead of both lanes.
- **The double-feature/striation mechanism** (multi-ridge mouths): views
  whose mouth/nose rows disagree with the mesh by 6–22% of head height
  were consumed both as texture references (painting mouth content onto
  chin/neck texels) and as 2mv conditioning (the DiT trusts its tags and
  carves BOTH row hypotheses — the documented double-mouth mechanism).
  Generation put the features on the wrong rows or the wrong pose;
  registration as shipped had no head-local term to catch or correct it.
  The fix needs both halves: pose/feature-row-aware acceptance at
  generation time (this audit's gate: <2% per feature vs the clay), and
  head-weighted/feature-anchored registration (the parallel audit's lane).

## 7. Reproduce

```bash
cd <workspace>/abstract3d
PY=<workspace>/.venv/bin/python

# clay guides for set B (e11 mesh, byte-faithful to the loop run)
$PY scripts/experimental/viewgen_audit_render_clays.py

# tables (sets A+B): landmarks, overlays, pose sweeps, offsets
$PY scripts/experimental/viewgen_audit_analyze.py     # -> /tmp/viewgen_audit/analysis.json

# recipe ladder (8 arms, ~1.5 h on shared MPS, sequential)
$PY scripts/experimental/viewgen_audit_ladder.py      # L1-L8
$PY scripts/experimental/viewgen_audit_ladder2.py     # L9-L11
$PY scripts/experimental/viewgen_audit_remeasure.py   # re-measure w/o i2i spend

# best-recipe regeneration (remote default route; redraws until accepted)
$PY scripts/experimental/viewgen_audit_regen.py                 # all angles
$PY scripts/experimental/viewgen_audit_regen.py side_right      # one angle, merges report
$PY scripts/experimental/viewgen_audit_reprocess.py             # offline, from saved raws

# the shipped production knob's test
$PY -m pytest tests/test_reference_generation.py -q -k prompt_suffix
```

MPS discipline: all i2i calls ran sequentially; competing generations were
checked (`pgrep -f i23d`) before each stage. Observed MPS-pressure
runtimes: klein-9b 8-step ≈ 2.2 min, 12-step ≈ 5 min, 16-step ≈ 17 min,
12-step + CFG ≈ 26–30 min at 1536×768.
