# VERDICT 1 — Candidate v9 Re-Ruling (pose-aware audit)

- Date: 2026-07-05
- Candidate: `/tmp/candidate_v9` (estimated source pose az +15 / el +8, gradient_ncc; thin flap shells below the hairline removed)
- Baseline: `artifacts/validation/iter3-multiview-fixed/face-2mv` (shipped proof)
- Harness: `/tmp/verdict1/qa.py`, updated this session (see "Harness self-audit"); both bundles re-measured with the identical updated battery. Photo negative-controls still pass every gate (`--calibrate-photos`: 3× ok).

## Verdict: v9 is still a FAIL — but it is a real, measurable improvement, and my previous "41 vs 39, no improvement" reading was wrong

With the pose-aware identity gate and the extended azimuth grid
({0, ±22.5, ±35, ±45, ±70, ±90, ±135, 180} × {0, 10} = 28 views), the score is
**shipped: FAIL, 66 failed checks — v9: FAIL, 64 failed checks**. The raw counts
are near parity, but the composition is not: v9 eliminated the three worst defect
classes outright (whole-face ghost doubling, the second mouth in the hair, the
ear/jaw skin cascades) and its remaining failures concentrate in two classes
(hairline debris band, eye-region ghosts). Check counts are gate violations, not
a quality score; the class table below is the ruling that matters.

## T1 — Harness self-audit: the pose blind spot was real, and it was double

Two independent measurement errors in my original identity gate suppressed v9's
scores relative to the shipped bundle:

1. **Wrong comparison pose.** The gate compared the az-0 render against
   `input.png`. The photo is ground truth exactly at its own viewpoint and
   nowhere else: with the subject's head yaw at ~+15–17°, an az-0 render shows
   the head rotated away from the photo's viewpoint, so bbox-aligned SSIM drops
   from parallax even if every texel is perfect. The shipped bundle *declared*
   az 0 (hard-assumed), so it was measured at its own declared pose; v9 declares
   +15/+8 and was measured ~15° off its declared pose. That asymmetry is what
   produced "no improvement".
2. **Bbox-only registration.** The old alpha-bbox mapping absorbed no residual
   framing offset (elevation shift, bust cropping). A masked-NCC similarity
   refinement (scale ±12%, shift ±10%, luminance-only, grid+local search) was
   added before comparison. Controls: photo-vs-itself still scores SSIM 1.000 /
   MAE 0.0 with an identity residual (1.0, 0, 0); the shipped left profile is
   unchanged (0.780, residual (1.0, 0, 0)); it cannot repair content damage,
   only framing.

Magnitude of the two effects on v9's front identity (SSIM / mean|RGB|):
old gate (az 0, bbox-only) **0.321 / 69.2** → refined registration at az 0
**0.421 / 46.3** → refined at the declared pose az +15 / el +8 **0.529 / 29.2**.
About +0.10 SSIM was parallax and +0.10 was framing. I retract the earlier
"41 vs 39 ⇒ no improvement" comparison: it measured my gate's pose assumption,
not the candidate's texture.

**Which comparison is the correct acceptance gate.** The gate now renders at the
pose the bake *declares* for each photo (`metadata source_pose` /
`observed_view_stats`, both metadata layouts supported; absent metadata falls
back to az 0 — full backward compatibility, no flags needed). Justification:
the photo is only ground truth at its own viewpoint, so texture fidelity must be
measured there. One caveat, handled explicitly: rendering at the bake's own
declared pose is partially self-fulfilling — a bake that projected the photo at
a *wrong* pose reprojects its own error and can still match the photo. So the
declared pose itself is validated by a non-optional **pose sweep**: identity is
re-measured at ±5/±10° around the declared azimuth, and if it peaks ≥10° away
(by >0.02 SSIM) the run FAILS with "declared source pose is wrong". Doubling
detectors and profile identity remain the second, independent pose check (a
wrong source pose stamps features twice against the profile/mirror views).
The az-0 comparison is kept as a recorded, non-gating diagnostic.

**The sweep validates the developer's pose fix:** v9's front identity peaks at
exactly +0° offset from the declared +15° (0.529 at +0 vs 0.518/0.453 at ±5 and
0.460/0.429 at ±10). The +15° estimate is corroborated by my independent
measurement; the shipped bundle's hard-assumed az 0 is refuted (its own front
view scores 0.516 *with* refinement and its face carries the doubling that a
~15° pose error produces).

Also added while auditing my own gates (both recalibrated against the photos,
which still pass): a worst-49-px-window SSIM gate inside the face hull
(`identity_local_min = 0.05`) because global SSIM averages away localized
ghosts once registration is refined (photo-self: 1.0; clean profile window:
+0.12; ghost/etched zones: −0.02…−0.18); and an ear exemption in
`skin_in_hair` for rear views (|az| > 110°), because ears behind the head are
legitimately compact skin islands ringed by hair (solidity ≥ 0.55, 0.2–1.5% of
foreground, mid-height band) — without it the rear view of *any* correct head
fails; shredded cascades remain flagged (fragmented, off-band, or oversized).

## T2 — Class-by-class ruling, shipped → v9

Identity at each bundle's own declared poses (refined registration, gates:
front ≥0.70 / ≤22, profile ≥0.55 / ≤30, local window ≥0.05):

| Reference | shipped SSIM/MAE (localMin) | v9 SSIM/MAE (localMin) | ruling |
|---|---|---|---|
| front @ declared pose | 0.516 / 30.9 (+0.002) | **0.529 / 29.2 (−0.018)** | marginal improvement; both FAIL all three gates |
| side_left @ +90 | 0.780 / 19.4 (+0.054) | 0.673 / 17.5 (+0.035) | global drop is hair-region structure after flap removal (hair SSIM 0.801→0.591); face-hull SSIM 0.732→0.710, unchanged within noise. Passes global gates, now marginally fails the local window at the forehead flake zone |
| side_right @ −90 | 0.629 / 19.0 (−0.157) | 0.633 / 24.9 (−0.175) | unchanged; both FAIL the local window at the etched ear zone |

Detector classes over all 28 views (mean / worst view):

| Class (gate) | shipped | v9 | ruling |
|---|---|---|---|
| skin_in_hair (≤0.010) | mean 0.0054, max **0.0199** | mean 0.0003, max 0.0008 | **FIXED** — the ear/jaw skin cascades are gone; every view passes with 12× margin |
| lip_in_hair (0) | 4 instances (second mouth) | **0** | **FIXED** |
| pale_film (≤0.005) | max 0.0008 | max 0.0008 | both pass; the milky second-face wash visible in shipped renders is gone by eye (see mouth pair below) |
| dark_debris (≤0.003) | mean 0.0043, max 0.0139 (az0) | mean 0.0039, max 0.0072 (az+22.5) | improved at the worst view (halved), class unchanged: fails 21 of 28 views, 12–24 islands per view |
| crown_flakes (≤0.0008) | mean 0.0025, max 0.0051 | mean 0.0028, max 0.0057 | **unchanged**, and newly present at rear-left views that were clean (−70 el0: 0.0003→0.0022, −90 el0: 0.0003→0.0012) |
| eye_count (anatomical bounds) | 11 views violate; total overcount 13 | 14 views violate; total overcount 20 | changed character: see by-eye count below |
| chroma_seam (≤0.45) | max 0.255, passes | max 0.210, passes | both pass |

### By-eye duplicate-feature count at ±22.5° and ±45° (full resolution, features only — eye/nose/mouth structures, debris excluded)

| View | shipped (by eye) | v9 (by eye) |
|---|---|---|
| +22.5 | ghost nose outline + ghost mouth bar + milky second-face wash on the left cheek: **2 duplicated features + wash** | **1 partial** duplicate lash-arc/crease right of the left eye; no iris, no nose/mouth ghost |
| −22.5 | fully formed ghost eye (iris+catchlight) + doubled nose (two nostril sets) + pale mouth bar + second mouth fragment in the hair: **4** | **1 full** duplicate iris-bearing eye between nose and right eye, plus a dark liner smudge at the mouth corner |
| +45 | ghost jaw/chin outline + ghost nose tip: **2 partial** | **0** (brow-band debris only) |
| −45 | ghost eye (iris) + doubled nostril + mouth trace: **3** | **1 full** duplicate iris (same structure as −22.5) |

Evidence pairs (nearest-neighbor zoom, no downscaling):
`report_evidence_v9/pair_m225_eyeband.png`, `pair_m225_mouth.png`,
`pair_p225_eyeband.png`, `pair_m45_eyeband.png`, `pair_m70_crown.png`.
The mouth pair is decisive for the pose fix: shipped shows two nostril sets and
two mouths at −22.5; v9 shows one nose (tip flakes) and one clean mouth.

**Ruling on the doubling:** the front-hemisphere *whole-face* doubling (offset
second face: nose+mouth+eye quartets and the milky wash) is genuinely gone.
What persists is *eye-level* ghosting: one fully formed duplicate iris on the
right-hemisphere views (−22.5/−45) and a partial lash-arc on the left
(+22.5/+35). The detector's higher v9 overcount (20 vs 13) reflects eye-scale
*debris blobs* in the brow band plus these ghosts — at the shipped bundle's
+22.5 the detector under-counted because its ghosts merge into the milky wash,
which is exactly why the by-eye count above, not the blob count, is the
doubling ruling.

## T3 — Residual defects: classification

1. **Brow/hairline debris band** (dark flakes on skin + pale skin flakes at the
   hairline; worst crop `report_evidence_v9/v9_identity_front_worst_window.png`,
   the identity gate's worst window lands in this band at (624,355) with local
   SSIM −0.018): **(a) improved at the worst view** (az0 dark debris halved,
   0.0139→0.0063) and **(b) same root cause** — the film-shell band above the
   brows survived the conservative flap cleanup; hair pixels bake onto brow
   skin and skin pixels onto hair films along the entire band. Fails
   dark_debris on 21/28 views and crown_flakes on 24/28. This is now the
   dominant blocker.
2. **Eye-region ghosts** (duplicate iris at −22.5/−45; partial lash-arc at
   +22.5/+35; dark lash/liner dashes under the eyes,
   `report_evidence_v9/pair_m225_eyeband.png`): **(b) same root cause as the
   old doubling — multi-view conflict at the eye zone — at much narrower
   extent.** The surviving full iris sits on the hemisphere textured by the
   *fabricated mirrored* right profile, consistent with the front-vs-mirror
   conflict surviving the pose fix at the highest-contrast feature.
3. **Rear-left crown flakes** (−70/−90 views, `pair_m70_crown.png`):
   **(c) NEW regression, mild.** Shipped measured 0.0003 (clean) at −70 el0;
   v9 measures 0.0022 with visible black flecks. Rotating the source
   projection +15° moved its hairline contact zone into views that the az-0
   projection left clean. Same film-band root cause, newly exposed location.
4. **Etched ear on the right profile** (local window −0.175 at (354,395),
   `v9_identity_side_right_worst_window.png`): **(b) unchanged** (shipped
   −0.157) — profile-photo hair strands baked across the ear geometry.
5. **Left-profile hair structure** (global SSIM 0.780→0.673 while MAE improved
   19.4→17.5): decomposition shows the drop is entirely in the hair region
   (0.801→0.591); face-hull SSIM is unchanged (0.732→0.710). **(c) regression
   in hair-surface structure caused by the flap removal** — cosmetically minor
   (the v9 profile's face is visibly better-defined), passes the global gate,
   but it shows the cleanup roughened the hair sheet.

## T4 — Updated PASS bar

`python /tmp/verdict1/qa.py BUNDLE_DIR` must exit 0. The battery is now, on all
28 views (14 azimuths × 2 elevations, 896 px, repo renderer):

1. **Pose-aware identity (gate, replaces the az-0 gate):** at each declared
   reference pose from metadata (front: `source_pose`; profiles:
   `observed_view_stats`; fallback az 0 when absent): front SSIM ≥ 0.70 and
   mean|RGB| ≤ 22; profiles SSIM ≥ 0.55 and mean|RGB| ≤ 30; worst 49-px face
   window SSIM ≥ 0.05 on every declared view.
2. **Pose sweep (gate):** front identity must peak within ±5° of the declared
   azimuth; peaking ≥10° away fails the run ("declared source pose is wrong").
   The az-0 comparison is recorded as a diagnostic, never gated.
3. **Detector gates unchanged** (photo-calibrated): eye_count within anatomical
   bounds per azimuth; lip_in_hair = 0; dark_debris ≤ 0.003 face / 0.0035 rear;
   crown_flakes ≤ 0.0008; skin_in_hair ≤ 0.010 (ears exempted only behind the
   head, |az| > 110°, compact solidity ≥ 0.55 mid-height islands);
   pale_film ≤ 0.005; chroma_seam ≤ 0.45.
4. **Evidence standard unchanged** from REPORT.md §T3 (≥768 px renders, 1:1
   crops of hairline/eyes/nose-mouth/ears, calibration shipped with the claim).

Current distance to the bar, ranked blockers for the next iteration:

1. Hairline film band (dark_debris + crown_flakes, 45 of 64 failed checks):
   geometry-side removal of the brow-band films, or projection-side exclusion
   of texels whose ray crosses a film edge. Fixing this alone likely also
   clears the front identity local window and most of the SSIM deficit.
2. Eye-zone ghosts (14 eye_count checks + front/side_right local windows):
   the mirrored right profile must lose to the front photo at the eye region
   (source-priority already exists for well-facing texels; the duplicate iris
   at −22.5 shows it is not winning there).
3. Rear-left flake spill (new): re-run the film-band fix check at −70/−90.
4. Ear etching on profiles: exclude profile-photo hair strands from baking
   onto ear geometry (depth/normal disagreement at the ear is large).

Numbers context for the bar: v9 front = 0.529 vs bar 0.70; the photo-self
control = 1.000; the best view this pipeline has produced (shipped left
profile) = 0.780. The bar is reachable; v9 is not there.

## File map (this session)

- `qa.py` — updated harness (pose-aware identity + sweep gate + local window +
  extended grid + rear-ear exemption; backward compatible, auto-detects
  metadata poses; ~8 s per bundle)
- `qa_out/v9/`, `qa_out/iter3_v2/` — full results + evidence crops for both
  bundles under the identical updated battery (`results.json`, `views/`,
  `evidence/`)
- `report_evidence_v9/` — the comparison pairs and worst-window crops cited here
- `calib_v2/photo_calibration.json` — reference photos pass the updated battery
- Prior audit (shipped bundle, original battery): `REPORT.md`
