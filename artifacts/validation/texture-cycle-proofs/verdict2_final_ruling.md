# VERDICT AGENT 2 — FINAL PERCEPTUAL RULING
## Candidates: /tmp/candidate_v13 (flap-cleaned mesh) vs /tmp/candidate_v13raw (original mesh)

Date: 2026-07-05. Prior review: `/tmp/verdict2/REPORT.md` (12-defect catalog, REJECT).
All new evidence under `/tmp/verdict2/final/` (48 views per candidate at 1000 px, el {-10, 0, +15},
plus el +8 identity renders, 48 zoomed crops, YuNet metrics, overlays, extracted atlases).

---

## VERDICT

**The catastrophe is fixed; the blemishes are not.** Both candidates now show ONE coherent,
recognizable face at every azimuth — the doubled-face chimera, the ghost mouths in the hair,
the ear-on-cheek, and the milky film over the mouth are gone. What remains is a persistent
class of *localized* texture debris (skin-flake fields at the temples/hairline, black hair
flecks on forehead and ear, a smudged nose bridge, and a doubled/seamed mouth at the exact
front view) plus a residual ~5–10° registration offset and ~+9% vertical stretch of the
painted face relative to the photo.

- **Better candidate: `v13raw`.** It wins the views that matter most (frontal mouth integrity,
  identity ratios at the metadata pose and at az 0) and loses only marginally on hairline debris.
- **Demanding client: NOT shippable as-is** — acceptance items 1, 3, 4 still fail at full
  resolution (details below).
- **Typical image-to-3D user: acceptable with honest caveats** — the asset now survives a
  360° turntable without any "what is THAT" moment; remaining flaws read as blemishes,
  not as a broken face.
- **Showcase ruling:** I would sign off on `v13raw` as a *documented work-in-progress
  showcase* ("known limitations: hairline debris, front-view mouth seam"), NOT as a
  polished flagship result. See closing paragraph.

---

## T1 — Defect catalog walk (original #1–#12)

Protocol: 16 azimuths × el {-10, 0, +15} at 1000 px per candidate
(`final/renders_v13*/`), overview grids (`final/grids/`), 24 zoomed 2× crops per candidate
(`final/crops/`). Status is for BOTH candidates unless split.

| # | Original defect (blocking in old bundle) | Status v13 | Status v13raw | Evidence |
|---|---|---|---|---|
| 1 | Doubled/displaced facial features (two face copies ~20–25° apart; 3 eyes, 2 noses, 2 mouths at az 0) | **IMPROVED — one residual instance.** Single eye pair and nose everywhere; but at az 0 the mouth is clearly doubled: a complete faded second lip pair sits below-left of the true lips. Nose bridge carries a ghost band of displaced texture. | **IMPROVED, closer to gone.** Same single-copy face; mouth shows a dark seam cutting under the lower lip and a partial ghost at the left corner — visible but materially weaker than v13's full second mouth. | `crops/v13_az0_nose_mouth.png`, `crops/v13raw_az0_nose_mouth.png`, `crops/*_az0_face.png` |
| 2 | Ghost face fragments painted into hair (detector fired at az ±112.5 on hair content) | **GONE.** No painted eyes/lips anywhere in the hair mass at any of the 48 views. Detector hits at ±112.5 now anchor on the REAL profile/ear visible at grazing angle, plus weak (0.52–0.53) false positives on hairline flake mottle — no facial features present in pixels. | **GONE** (same caveat). | `identity/ghostcheck_*.png`, `crops/*_azp112_hair.png`, `crops/*_azm112_hair.png` |
| 3 | Skin-colored flake fields over scalp hair (ghost eyebrow, shredded crown) | **IMPROVED, still clearly present.** The crown-wide shredding and ghost eyebrow are gone. Remaining: pale flake fields at both temples and along the parting at az −67.5…+67.5 (el 0 and +15), and a beige debris band above the brow at ±90. | Same; marginally larger temple mottle at az 0, heavier band at −67.5. | `crops/v13_az0_hairline.png`, `crops/v13raw_az0_hairline.png`, `crops/v13_azm45_hairline.png`, `crops/*_azp90_temple.png` |
| 4 | Black flake debris on forehead/hairline skin | **IMPROVED, still present.** Black flecks along the upper forehead boundary at az 0/±22.5/±45, a dark diagonal streak at −67.5, black specks on both ears. | Same; the −67.5 streak is thicker, ear-lobe blob larger. | `crops/*_az0_hairline.png`, `renders_*/scene_az-0067.5_el+00.png`, `crops/*_azp90_ear.png` |
| 5 | Milky translucent patches over facial features ("bar mouth", film under eyes) | **LARGELY GONE.** No film over the mouth or eyes. Remaining beige smudges are localized: tear-line fragments under the subject-right eye (with small black dashes), a pale band down the nose bridge, a patch on the columella/nose tip, a dab on the chin. | Same pattern; nose-tip/columella patch slightly larger, tear-line similar. | `crops/*_az0_eyes.png`, `crops/*_az0_nose_mouth.png`, `crops/*_azm22_face.png` |
| 6 | Ear-like patch painted on the cheek/sideburn | **GONE.** Ears sit on ears at every view; cheeks are clean skin. The ears themselves are mottled (pale flakes + black specks) — folded into #3/#4 above. | **GONE** (same ear mottle). | `crops/*_azp90_ear.png`, `crops/*_azm90_ear.png`, `crops/*_azp45_face.png` |
| 7 | Eye smeared toward temple at ±90 | **GONE at +90** (clean, natural profile eye). At −90 a small beige patch sits at the outer eye corner — residual, minor. | Same. | `crops/*_azp90_profile.png`, `crops/*_azm90_profile.png` |
| 8 | Pink lip-colored debris in hair | **GONE.** No pink blobs in any hair region at 48 views. | **GONE.** | `crops/*_az0_left_hair_curtain.png`, hair crops |
| 9 | Desaturated smear band on back of head | **GONE.** Back hemisphere is a uniform dark mass. Note: a wide, featureless harmonic-fill zone (no strand detail) covers much of the back — even, but low-detail. | **GONE** (same low-detail note). | `crops/*_az180_back.png`, `renders_v13raw/scene_az+0180.0_el+00.png` |
| 10 | Ragged skin fringe on hair silhouette | **MOSTLY GONE.** Silhouettes are hair-colored; a thin skin-tone halo persists along the hair/face boundary behind the ear at ±45…±112.5. | Same. | `crops/*_azp112_hair.png`, `crops/*_azp45_face.png` |
| 11 | Strap/chest fragments smeared | **IMPROVED.** Straps render as coherent white bands; minor smears at the bust cut edge and a light-brown throat patch remain. | Same. | `crops/*_az0_neck_chest.png` |
| 12 | Faceted polygons on chin/jaw | **UNCHANGED** (geometry stage; mild faceting on chin silhouette at 1000 px; minor). | Same. | `crops/*_az0_face.png` bottom edge |
| NEW | — | **v13-specific NEW:** the az-0 doubled mouth is *worse in v13 than v13raw* (a full faded second lip pair vs a seam). No other new defect classes found in either candidate at 48 views each. | — | `crops/v13_az0_nose_mouth.png` |

Bottom line of the walk: 5 of the 12 catalog entries are GONE, 5 are IMPROVED-but-present,
1 mostly gone, 1 unchanged-minor. Every remaining visible defect belongs to two families:
(a) hair/skin boundary debris (#3/#4/#10 remnants), (b) front-view feature seams
(#1/#5 remnants: doubled mouth, nose-bridge smudge, tear-line fragments).

---

## T2 — Identity check (YuNet 5-point, photo vs render at the metadata pose)

The metadata source pose is az +15°/el +8° (v13) and az +17.5°/el +8° (v13raw) in the
pipeline convention, which maps to **renderer az −15/−17.5, el +8** (sign convention verified
empirically: ratio error is minimized on the negative-azimuth side, exactly as in the first
review where pipeline +20 ≈ renderer −22.5).

| ratio | photo | v13 @ az−15/el8 | Δ | v13raw @ az−17.5/el8 | Δ | old bundle az 0 (for scale) |
|---|---|---|---|---|---|---|
| eye-to-mouth / interocular | 1.080 | 1.178 | **+9.1%** | 1.178 | **+9.1%** | 1.210 (+12.0%) |
| mouth width / interocular | 0.807 | 0.741 | −8.1% | 0.791 | **−2.0%** | 0.787 (−2.4%) |
| nose drop / interocular | 0.644 | 0.670 | +4.0% | 0.633 | **−1.8%** | 0.693 (+7.6% vs photo) |
| interocular / box height | 0.318 | 0.335 | +5.4% | 0.327 | +3.0% | 0.338 (+6.5%) |
| eye-line roll (deg) | 0.58 | −5.16 | Δ5.7° | −3.05 | **Δ3.6°** | +10.76 (Δ10.2°) |

At the canonical front view (az 0, el 0): v13 roll −1.95°, e2m +13.9%; v13raw roll −0.41°,
e2m +9.9%. The 10.8° roll pathology of the old bundle is gone; both candidates now read as
*level* frontal heads.

Azimuth sweep (el 8, 5° steps, `identity/azimuth_sweep_final.json`): the ratio-error minimum
sits at **az −5° for both candidates** (v13 err 0.180, v13raw err 0.137) with a flat plateau
from −20° to 0°. Two readings: (a) the painted face is registered within ~5–10° of the
geometric front — versus 22.5° off in the old bundle; (b) the eye-to-mouth ratio never drops
below +5.5% at any azimuth, so the ~+9% vertical stretch of the painted lower face is a real
residual (projection/mesh proportion mismatch), not a pose artifact.

Overlay blends at the metadata pose (`identity/overlay_v13_metapose.png`,
`identity/blend_zoom_v13.png`): eyes and brows of the warped photo and the render coincide;
the render's mouth sits slightly low-left; jaw outline near-coincident.

**Same-person judgment, stated plainly:** Yes — for the first time in this project, the
rendered head reads as *the woman in the photo*: same hair mass and parting, same face shape,
same eye makeup and lip tone, same overall gestalt. It reads as her *after mild vertical
stretching of the lower face* (~9%) — like a subtly elongated caricature of the photo — and
at az 0 the v13 mouth doubling breaks the spell on close inspection. v13raw at three-quarter
views is genuinely convincing; a stranger matching the render to the photo would pair them
without hesitation. That is a categorical change from the old bundle, whose front view was
not a face at all.

---

## T3 — Acceptance items (from my T5 list, in order of importance)

| # | Criterion (abbreviated) | v13 | v13raw | Evidence / magnitude |
|---|---|---|---|---|
| 1 | az 0 (and el −10/+15): exactly ONE feature set aligned to geometry; ratios within 3% of photo; roll < 2°; no milky patches on mouth/eyes; no ghost features in frame | **FAIL** | **FAIL (closer)** | Roll passes (−1.95°/−0.41°). One eye pair and nose ✓. Mouth: v13 doubled (full second lip pair), v13raw seamed — both violate "exactly one set". Ratios: e2m +13.9% (v13) / +9.9% (v13raw) ≫ 3%. Tear-line + nose-bridge smudges persist. `crops/*_az0_face.png` |
| 2 | ±22.5/±45: single nose/mouth/eye pair; no second-copy fragments on cheeks; no ear-on-cheek | **PASS** | **PASS** | Single features at all four views, both candidates; cheeks clean; ears on ears. Flake debris at these views is scored under item 3. `crops/*_azm22_face.png`, `*_azp45_face.png` |
| 3 | Hairline band az −67.5…+67.5, el 0/+15: forehead/temple skin free of black flakes; scalp hair free of skin flakes/ghost brows | **FAIL** | **FAIL** | Temple flake fields and parting mottle at az 0/±22.5/±45/±67.5; black flecks on forehead boundary; dark streak at −67.5 (worse in v13raw). Far smaller than the old crown-wide shredding, but plainly visible at 1000 px. `crops/*_az0_hairline.png`, `*_azm45_hairline.png` |
| 4 | ±90: eye not smeared into temple; ear reads as ear, clean sideburn, no black streaks | **FAIL (near-pass)** | **FAIL (near-pass)** | Eyes clean (+90) / tiny beige corner patch (−90); ears anatomically correct BUT carry black debris blobs at the lobe and pale flake mottle; brow-line debris band above the profile brow. `crops/*_azp90_ear.png`, `*_azm90_profile.png` |
| 5 | ±112.5…±157.5: no face detectable in hair; no skin patches inside hair other than ear | **PASS on intent, FAIL by letter** | **PASS on intent, FAIL by letter** | No facial features painted in hair (pixel inspection). Detector still fires 0.52–0.88: the strong hits are the REAL profile/ear legitimately visible at ±112.5; weak hits (0.52–0.53) anchor on hairline flake mottle and strand texture (v13 +157.5). By my written threshold (<0.3) this fails; by its intent (no ghost faces) it passes. `identity/ghostcheck_*.png` |
| 6 | az 180: no desaturated band >15% head width; fringe skin-free except ears | **PASS** | **PASS** | Uniform back mass; no smear band; fringe hair-colored. (Note: broad featureless harmonic zone = low detail, not a violation.) `crops/*_az180_back.png` |
| 7 | texture.png: no facial content in hair charts; harmonic fraction over face-visible charts small | **FAIL** | **FAIL** | Atlas still shatters into thousands of confetti charts with face fragments scattered among hair charts (`final/texture_v13_1k.png`); `observed_coverage_ratio` dropped to 0.41 (v13) / 0.40 (v13raw) vs 0.57 before — i.e. ~60% of texels are now fill/symmetry content. Renders prove the extra copies no longer surface visibly, but the atlas remains fragile to any UV/mesh change. |

**Score: v13 1 PASS + 2 intent-PASS of 7; v13raw the same, with smaller failure magnitudes on
items 1 and 4's identity-adjacent parts.** Items 1–5 do not all pass, so per my own protocol
neither candidate is acceptable *to a demanding client*:

- **Item 1 failure size:** v13raw — mouth seam ~15 px at 1000 px render, e2m ratio +9.9%
  (limit 3%); v13 — full duplicate mouth ~40 px offset, e2m +13.9%. **Blocking for a demanding
  client** (the mouth is where everyone looks), **not blocking for a typical library user**
  at v13raw's level (reads as a rendering seam, not a deformity; invisible past ~0.5 m
  viewing distance on a turntable).
- **Item 3 failure size:** flake fields covering roughly 2–5% of the visible head area at
  front/three-quarter views. **Blocking for a demanding client; borderline for a typical
  user** (comparable to matting errors common in one-click photogrammetry outputs).
- **Item 4 failure size:** debris blobs of a few hundred px² at the ear/lobe. **Not blocking
  for a typical user; a demanding client would flag it.**

**Better candidate: `v13raw`.** Rationale: it wins item 1 outright (seam vs full duplicate
mouth; +9.9% vs +13.9% e2m; roll −0.41° vs −1.95°), wins the metadata-pose identity table on
3 of 5 rows, and its only relative losses (slightly heavier hairline streak at −67.5, larger
ear-lobe blob) are item-3/4 magnitudes, not category changes. The flap-cleaning in v13 bought
no visible silhouette benefit at any of the 48 reviewed views — and its bake landed a worse
mouth registration.

---

## T4 — Closing paragraph for the project owner (plain language)

Six agent-cycles ago I called the face bundle terrible, and it was: two faces fighting on one
head, a mouth in the hair, an ear on a cheek. That is over. The pose-estimation and
registration fixes did exactly what they were supposed to do: the photo's yaw is now measured
instead of assumed (metadata says +15–17.5°, my independent sweep confirms the painted face
sits within ~5–10° of the geometry, versus 22.5° off before), and every view of both
candidates shows one face, one mouth, one nose, in the right places, recognizably the woman
in the photograph. On a slow turntable, v13raw is the first artifact of this project I would
show to another person without apologizing first. What still stands between "fixed" and
"good" is a single family of remaining sins: hair/skin boundary debris — pale flakes over the
temples and parting, black flecks on the forehead and ears — plus one front-view seam through
the mouth and a smudged nose bridge, and a subtle ~9% vertical stretch of the painted face
that a side-by-side with the photo reveals. None of these is a chimera; all of them are
visible at full resolution to anyone who looks for ten seconds. My ruling: ship `v13raw` as
the multi-view face showcase **only** with an honest limitations section (front-view mouth
seam, hairline debris, softened detail in fill regions) and ideally a "known issues" crop
strip so users' expectations are set by you and not by their disappointment; do not label it
"fixed" or "high-fidelity". If you want the unqualified showcase, the remaining work is
narrower than what you just did: kill the mouth-region view conflict at the source pose,
and stop skin/hair texels from crossing the boundary — the same two crops (az 0 mouth,
az 0 hairline) will tell you the day it's done.

---

### Evidence index

- Renders: `final/renders_v13/`, `final/renders_v13raw/` (55 images each, 1000 px)
- Grids: `final/grids/grid_{v13,v13raw}_el{+00,+15}.png`
- Crops (2× nearest): `final/crops/{v13,v13raw}_*.png` (24 each)
- Identity: `final/identity/identity_final.json`, `landmarks_*.png`,
  `azimuth_sweep_final.json`, `overlay_*_metapose.png`, `blend_zoom_*.png`,
  `ghostcheck_*.png`
- Atlases: `final/texture_{v13,v13raw}.png` (+ 1k previews)
- Scripts (reproducible): `final/render_candidates.py`, `final/make_crops.py`,
  `final/identity_final.py`, `final/azimuth_sweep_final.py`, `final/overlay_final.py`
