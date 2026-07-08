# VERDICT 1 — Multi-View Face Proof Audit

- Date: 2026-07-05
- Auditor: hostile results audit (verdict agent 1). No repository files were modified.
- Scope: `artifacts/validation/iter3-multiview-fixed/face-2mv/` (shipped proof), `artifacts/validation/iter2-multiview/face-2mv-final/` (prior iteration), `artifacts/validation/final-proof/hunyuan-face/` (single-view baseline), and the claims about them in `CHANGELOG.md`, `docs/benchmarks.md`, `docs/models.md`, `docs/adr/0007`.

## Verdict: FAIL

The shipped bundle `iter3-multiview-fixed/face-2mv` fails 39 independent QA checks
across 15 of 20 rendered views plus 2 of 3 photo-identity comparisons. The user's
complaint is confirmed and measurable: doubled facial features (up to 5 eye-like
blobs where 2 are possible), a fully formed second mouth painted inside the hair,
black debris across the hairline at every face-bearing azimuth including the source
view, and skin flakes shredding the hairline. The bundle is a real improvement over
iter2 (which fails 47 checks and all 20 views), but the documentation describes it
as clean and coherent, which the artifact does not support. Several coverage numbers
in the docs (0.74 / 0.89 / 0.91) do not match the shipped bundle's own metadata
(0.5687).

## How to reproduce

```bash
cd /Users/albou/abstract3d && source .venv/bin/activate
python /tmp/verdict1/qa.py artifacts/validation/iter3-multiview-fixed/face-2mv
# exit code 0 = PASS, 1 = FAIL; full scores in <out>/results.json,
# full-resolution crops of every failure in <out>/evidence/
```

Any bundle directory containing `scene.glb` (+ optional `input.png`,
`metadata.json`) works. Reference photos default to the bundle `input.png` (front)
and the PREFERRED profile photos documented in
`artifacts/validation/face-multiview-prototype/README.md` when the bundle metadata
declares those views; override with `--ref side_left=PATH` etc. Sanity-check the
detector calibration anytime with `--calibrate-photos` (runs the identical battery
on the reference photos; all three pass).

Runtime ~40 s per bundle. Renders use the repository's own renderer
(`abstract3d.rendering.render_mesh_views`, ModernGL path, flat-biased textured
shading) at 896 px — the same code path as the shipped contact sheets, so the
defects cannot be blamed on a foreign renderer. `--size` below 768 is rejected.

## T2 — Scores side by side

Verdict and per-gate worst-view measurements (photo-calibrated gate in
parentheses; render grid = azimuths 0/±35/±70/±90/±135/180 × elevations 0/10):

| Measurement (worst view) | gate | iter3 (shipped) | iter2 | single-view |
|---|---|---|---|---|
| verdict | all pass | **FAIL (39 checks, 15/20 views)** | FAIL (47, 20/20) | FAIL (16, 11/20) |
| identity front: SSIM / mean\|RGB\| | ≥0.70 / ≤22 | **0.358 / 54.6** | 0.318 / 63.6 | 0.317 / 66.9 |
| identity side_left: SSIM / mean\|RGB\| | ≥0.55 / ≤30 | **0.780 / 19.4 — passes** | 0.352 / 64.9 | n/a |
| identity side_right: SSIM / mean\|RGB\| | ≥0.55 / ≤30 | **0.448 / 47.5** | 0.335 / 73.1 | n/a |
| eye-like blobs (anatomical cap 2 front / 1 profile) | within bounds | **5 at az0 el10, 2 at az+90** | 4 at az0 | 2 at az+70; 0 where 1 required (right side destroyed) |
| dark debris on skin, foreground fraction | ≤0.003 | **0.0139 at az0** | 0.0098 | 0.0025 (passes) |
| hairline skin flakes (crown band) | ≤0.0008 | **0.0051 at az+35** | 0.0212 | 0.0008 (passes) |
| skin islands in hair | ≤0.010 | **0.0189 at az−35** | 0.0095 (passes) | 0.0252 |
| pale film in face hull (largest component) | ≤0.005 | 0.0008 (passes) | **0.0109** | 0.0095 |
| chroma seam column fraction | ≤0.45 | 0.169 (passes) | **0.744** | 0.239 (passes) |
| lip-colored blobs ringed by hair | 0 | **2 (a second mouth at az−35)** | 0 after gate hardening | 0 |
| failure-free azimuths | all | az−135, az+180 only | none | az0/±90/±135/180 partially |

Failure-kind counts: iter3 = 6 eye_count, 12 dark_debris, 13 crown_flakes,
2 lip_in_hair, 2 skin_in_hair, 4 identity; iter2 = 5 eye_count, 12 dark_debris,
17 crown_flakes, 4 pale_film, 3 chroma_seam, 6 identity; single = 7 eye_count,
3 skin_in_hair, 4 pale_film, 2 identity.

Read the check counts as gate violations, not a quality ranking between failure
modes: the single-view bundle fires fewer checks than iter3 not because it is
closer to shippable but because its dominant defect is different — the entire
right half of the face is smeared away (7 views where an anatomically required
eye is undetectable), while iter3's defects are additive contamination (debris,
flakes, duplicated features) on top of otherwise present features.

### Defect inventory of the shipped bundle (full-resolution evidence)

All paths under `/tmp/verdict1/`. Every crop is saved at native render scale or
larger (nearest-neighbor upscale to ≥384 px), so nothing here depends on shrinking.

1. **Doubled face.** At az0 the render contains two overlapping face instances
   offset roughly 100 px right / 40 px down at 896 px: two well-formed eye blobs at
   (303,308) and (385,364) plus feature-scale dark blobs where no feature can be,
   4–5 total against an anatomical cap of 2
   (`report_evidence/az+000_el00_annotated.png`, red boxes;
   `qa_out/iter3/evidence/az+000_el00_eye_303_308.png` is a fully formed duplicate
   eye). At az+90 a second eye-scale blob sits in the temple debris field
   (cap 1 at profile).
2. **Second mouth inside the hair.** At az−35 a complete red-lipped mouth is
   painted at the jaw/hair boundary, ringed by hair
   (`report_evidence/az-035_el00_lip_in_hair_279_561.png`). Fires at both
   elevations.
3. **Black debris across the hairline and face.** 32 dark micro-islands on skin at
   az0 (1.39% of foreground vs photo baseline ≤0.19%); present at every
   face-bearing azimuth, worst at the source view — the view the docs call clean.
4. **Hairline shredded into skin flakes.** 8–16 isolated skin islands in the crown
   band per view (up to 0.51% of foreground vs photo baseline ≤0.035%), visible as
   the pink-on-black confetti in `report_evidence/az+090_el00_annotated.png`.
5. **Identity loss vs the photos it was baked from.** Front: SSIM 0.358 / mean|RGB|
   54.6 against `input.png` (gates 0.70 / 22; the photo-vs-itself control scores
   1.000 / 0.0) — see `report_evidence/identity_front_pair.png`. Right profile:
   0.448 / 47.5 with a doubled-nose ghost
   (`report_evidence/identity_side_right_pair.png`). The pale ghost nose/mouth and
   milky patches the user saw are part of this identity failure (they sit exactly
   where the front bake and the mirrored-right bake disagree).

## Claims not supported by the artifacts

Quotes are exact; measurements are from `qa_out/*/results.json` and the bundles'
own `metadata.json`.

1. `docs/benchmarks.md` ("Multi-View Face Proof"): *"observed texture coverage:
   `0.19` (single view) -> `0.91` (multi-view geometry + bake)"* — *contradicted*:
   the shipped bundle's `metadata.json` records `observed_coverage_ratio: 0.5687`.
   No artifact in the repo records 0.91 for this bundle. (`docs/methodology.md`
   line 72 calls 0.91 "pre-gating" and admits final bundles report lower ratios;
   benchmarks.md and models.md quote 0.91 with no such caveat.)
2. `docs/benchmarks.md`: *"the turntable is coherent at all azimuths: correct face
   at the source view, clean profiles matching the reference photos,
   hair-consistent back"* — *demolished in two of three parts*: the source view is
   the single dirtiest view (eye count 4–5 vs cap 2, dark debris 1.39% vs 0.3%
   gate, identity SSIM 0.358 vs 0.70 gate); the right profile does not match its
   reference (SSIM 0.448, doubled nose). "Clean profiles" is half-true: the LEFT
   profile does match (0.780 / 19.4, passes) — see calibration section.
   "Hair-consistent back" is supported (az180 passes every detector at both
   elevations).
3. `docs/benchmarks.md`: *"known residual limit: … appear as small flakes at the
   hairline from three-quarter angles"* — *materially understated*: flakes and
   debris fail the gates at ten face-bearing views including the frontal source
   view, not only three-quarter angles, and the doubled features / second mouth /
   identity loss are not mentioned as limits at all.
4. `docs/models.md` line 81: *"front + left/right profiles raised observed texture
   coverage from 0.19 (single view) to 0.91"* — *contradicted*: shipped metadata
   says 0.5687 (the honest claim is 0.19 → 0.57 observed, a real 3× improvement).
   The second half — *"replaced the hallucinated back of the head with a plausible
   multi-view-constrained shape"* — is supported.
5. `CHANGELOG.md` 0.2.0: *"On the checked face proof, front + both profiles raised
   observed texture coverage from 0.19 to 0.74 and replaced the hallucinated back
   of the head."* — *contradicted*: 0.74 matches the superseded iter2 bundle
   (0.7606), not the shipped iter3 proof (0.5687). Same number reappears as
   *"(face proof coverage with front + both profiles: 0.50 -> 0.74)"*.
6. `CHANGELOG.md` 0.2.0 (Fixed): *"Fixed the ghosted 'second face' on textured
   previews: … an independent CPU rasterization of the same textured mesh was used
   to prove the texture itself was correct."* — *not supported for the shipped
   bundle*: the doubled features are in the baked albedo (the flat-biased preview
   shader is exactly what qa.py renders with, and eye-count still measures 4–5 at
   az0; the duplicate-eye crop is baked texture, not shading).
7. `CHANGELOG.md` 0.2.0 (Fixed): *"Fixed duplicate feature stamping onto hidden
   crust sheets: projector visibility is now a strict per-photo-pixel first-surface
   z-buffer …"* — the mechanism may exist, but *the defect class it claims to fix
   is present in the artifact shipped as proof* (duplicate eyes at az0/az+90,
   duplicate mouth at az−35).
8. `docs/adr/0007` (Consequences): *"The checked face bake is clean at the source
   view and coherent from all azimuths; observed coverage on the corrected face
   mesh reaches ~0.89 at 2048"* — *both contradicted*: source view scores worst of
   all 20 views; metadata says 0.5687.

## T3 — Evidence methodology audit

The developer's acceptance evidence was `/tmp/face_v8_turntable.png` (3360×420,
eight 420 px panels at 45° steps, elevation 0 only) and
`artifacts/validation/iter3-multiview-fixed/face_before_after_fixed.png` (the same
eight panels over an iter2 row; pixel-identical bottom row, mean|diff| = 0.00).
Plus coverage ratios from metadata. Three quantified failures of that methodology:

1. **Pixel budget.** The median hairline-debris island measures 64 px² (8 px
   square-equivalent side) at 896 px render scale. In a 420 px panel that is 3.8 px;
   when the 3360 px sheet is viewed fit-to-width on a 1512 px laptop screen each
   panel displays at 189 px and the median island is **1.7 px — one antialiased
   pixel**. The ghost mouth (~90×35 px at 896) displays at 19×7 px. Nothing that
   fails this audit is legible at the scale the evidence was reviewed at.
   `report_evidence/hairline_debris_az0.png` shows the same crop at their 420 px
   (nearest-upscaled) next to 896 px.
2. **The defects were in their own evidence, unexamined.** At 100% zoom the 420 px
   panels DO show the doubled nose, ghost mouth, and milky patch — see
   `report_evidence/doubled_face_az-45.png` (their az315 panel, 2.1× zoom, left)
   against the 896 px render (right). The failure was not only resolution; it was
   that no one inspected the panels at even 1:1, and no check existed that would
   force it.
3. **Coverage is not a quality metric.** `observed_coverage_ratio` counts texels
   any accepted view claims — including every misprojected flake and duplicated
   feature audited above. iter2 scores HIGHER coverage (0.7606) than the shipped
   iter3 (0.5687) while failing all 20 views and all three identity checks.
   Reporting coverage as the headline improvement metric selected for the wrong
   thing. (Automated detectors, unlike human review, survive downscaling: the
   qa.py battery fires 17 checks at 420 px vs 18 at 896 px on iter3 el0 — the
   methodology gap was the absence of any detector, not just small pixels.)

### Minimum evidence standard for any future "it works" claim

1. **Resolution:** every reviewed view rendered at ≥768 px (this harness uses
   896); no more than 4 panels per screen-width row in any composite sheet.
   Contact-sheet thumbnails are for navigation, never for acceptance.
2. **Azimuth grid:** at minimum {0, ±35, ±70, ±90, ±135, 180} × elevations
   {0, 10}. The worst defects in all three audited bundles live at ±35/±70 —
   between the declared views, where no 45°-step turntable panel and no declared
   view looks.
3. **Mandatory 1:1 crops** (≥384 px on the short side) of: hairline band, both
   eye regions, nose/mouth, each ear, and each declared-view/photo overlap zone.
4. **Metrics gated, not eyeballed:** the qa.py battery (identity SSIM + mean|RGB|
   per declared photo, eye-count bounds, lip-in-hair, dark-debris, crown-flakes,
   skin-in-hair, pale-film, chroma-seam) with the thresholds below, PASS required
   on every view. Coverage ratios may be reported only alongside these gates,
   never instead of them.
5. **Negative-control calibration shipped with the claim:** the reference photos
   must pass the same battery (`--calibrate-photos`), proving the detectors do not
   pass defects and do not fail good images.

## Acceptance criteria for calling this fixed

A candidate bundle may be called fixed only when
`python /tmp/verdict1/qa.py BUNDLE` exits 0, i.e. simultaneously, on all 20 views:

| Gate | Threshold | Photo baseline | Shipped bundle today |
|---|---|---|---|
| identity front SSIM / mean\|RGB\| | ≥ 0.70 / ≤ 22 | 1.000 / 0.0 (self) | 0.358 / 54.6 |
| identity profile SSIM / mean\|RGB\| | ≥ 0.55 / ≤ 30 | 0.780 / 19.4 achieved by iter3 left | right: 0.448 / 47.5 |
| eye-like blobs | 2 at ≤20°, 1–2 to 75°, 1 at 75–95°, ≤1 to 150°, 0 beyond | exactly 2 / 1 / 1 | 4–5 at az0 |
| lip-colored blobs ringed by hair | 0 | 0 on all photos | 2 |
| dark debris on skin | ≤ 0.003 (face-bearing) / 0.0035 (rear) | ≤ 0.0019 | 0.0139 |
| crown skin flakes | ≤ 0.0008 | ≤ 0.00035 | 0.0051 |
| skin islands in hair | ≤ 0.010 | ≤ 0.0053 | 0.0189 |
| pale film (largest, of face hull) | ≤ 0.005 | ≤ 0.0011 | passes (0.0008) |
| chroma seam column fraction | ≤ 0.45 | ≤ 0.32 | passes (0.169) |

Each gate sits between the photo-population maximum and the defective-view
minimum with ≥1.3× margin to the photos, so a bake as clean as the actual
photographs passes comfortably; iter3's left profile already passes identity and
pale-film/seam gates, proving passable scores are reachable by this pipeline.
Required visuals per the evidence standard above must accompany the PASS.

## Calibration — what is genuinely good (credit where due)

- **Back of the head (az180) is clean** at both elevations: hair texture is
  coherent, no flakes, no skin bleed. The "hair-consistent back" and "replaced the
  hallucinated back" claims are supported. az−135 also passes everything.
- **The left profile matches its reference photo well**: SSIM 0.780 / mean|RGB|
  19.4 (passes gates a real photo pair would pass). Geometry conditioning on that
  side did its job; that view fails only on hairline debris/flakes.
- **iter3 is a real improvement over iter2** on every axis measured: 39 vs 47
  failed checks, 15/20 vs 20/20 failed views, identity 0.358/0.780/0.448 vs
  0.318/0.352/0.335, pale-film and seam gates now pass. The "before/after
  improvement" story is true; the error was declaring the after-state good.
- The 0.19 → 0.57 observed-coverage gain is real and meaningful — it is the
  overstatement to 0.74/0.89/0.91 that is unsupported.

## Harness scope and honesty notes

- Detectors are screen-space and color-class based (skin/hair/lip via
  YCrCb + Lab), calibrated on this subject (fair-skinned, dark-haired). For a
  different subject, re-verify with `--calibrate-photos` before trusting gates.
- Identity registration is alpha-bbox anisotropic scaling — deliberately crude but
  sufficient: it scores the photo-vs-itself control at 1.000/0.0 and the genuinely
  good left profile at 0.780/19.4, while every defective declared view scores
  ≤0.45/≥47. It measures identity destruction, not sub-pixel alignment.
- The eye detector counts "eye-scale dark blobs enclosed by skin". At az0 two of
  the 4–5 counted blobs are the true doubled-eye pair and the rest are eye-scale
  debris in skin-flooded hair regions; both are hard defects, and the anatomical
  cap is violated either way. Brows are excluded (aspect gate), verified on all
  three photos (exact counts 2/1/1, zero false positives).
- One earlier false positive (a genuine ear at az−135 flagged as lip-in-hair) was
  found during self-review and eliminated by requiring true lip pigment
  (a* ≥ 149; photo lips measure ~151, ears ≤146). Detectors were re-validated
  against the photos after every such change; the run above is the final
  configuration.

## File map

- `qa.py` — the harness (one command per bundle, exit code = verdict)
- `qa_out/{iter3,iter2,single}/results.json` — full scores
- `qa_out/*/views/` — all 20 renders at 896 px per bundle
- `qa_out/*/evidence/` — 237 full-resolution failure crops + annotated views (iter3)
- `report_evidence/` — the crops cited in this report
- `scale_study/` — 420 px vs 896 px comparison data and pairs (T3)
- `calib/photo_calibration.json` — proof the reference photos pass every gate
