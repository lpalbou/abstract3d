# VERDICT AGENT 2 — Independent perceptual review
**Artifact:** `artifacts/validation/iter3-multiview-fixed/face-2mv/` (scene.glb + texture bundle)
**Date:** 2026-07-05. All evidence under `/tmp/verdict2/` (renders 1000 px, 16 azimuths x el {-10, 0, +15}, plus untextured geometry renders, crops at 2x).

## Executive verdict

**REJECT. The user's judgment ("the face is completely distorted") is correct and understated.**
The shipped asset fails at every azimuth in the front hemisphere (-67.5 deg to +67.5 deg, all three elevations). The frontal view is not a distorted face — it is a **chimera assembled from two misregistered projected copies of the face plus desaturated fill patches**. The prior claim "coherent at all azimuths" is false and is contradicted by the developer's own evidence sheets at 1:1 magnification.

The single most important finding: **the front-photo projection is registered about 20-25 degrees off in azimuth.** The painted face reads as a correct frontal face when the camera sits at azimuth -22.5 deg (landmark ratio error 4.9%, eye-line roll 0.58 deg — identical to the photo), while at azimuth 0 the detector measures a 10.8 deg roll and a +12% stretched eye-to-mouth proportion on a supposedly level, symmetric render. The untextured geometry faces azimuth 0 correctly (see `renders/geom_az+0000.0_el+00.png`), so this is a texture-stage registration failure, not a mesh canonicalization failure. Metadata records `source_pose: {azimuth_deg: 0.0, estimated: false, silhouette_iou: 0.0}` — the pipeline assumed the photo is a perfect az-0 view and never estimated the actual head yaw (~15-20 deg in the photo, nose lateral offset -0.23 interocular units).

---

## T1 — Defect catalog (from 48 textured renders + 16 geometry renders)

Overview grids: `grid_el-10.png`, `grid_el+00.png`, `grid_el+15.png`, `grid_geometry_el0.png`.

| # | Defect | Where seen (az/el) | Evidence crop | Severity | Best-hypothesis stage |
|---|--------|--------------------|---------------|----------|----------------------|
| 1 | **Doubled/displaced facial features.** Two projected face copies ~20-25 deg apart: one from the front photo (displaced viewer-left/up, enlarged), one from the profile refs + symmetry mirror. At az 0 the visible "face" mixes both: 3 eye patches, 2 noses, 2 mouths. | all of az -67.5..+67.5, el -10/0/+15 | `crops/az0_face.png`, `crops/az0_eyes.png`, `crops/azm22_face.png`, `crops/azp22_face.png`, `crops/azm45_face.png`, `crops/azp45_face.png` | **BLOCKING** | Texture projection registration: front photo projected assuming az=0 pose while the photo has ~20 deg head yaw (`source_pose.estimated: false`); profile projections + `mirror_symmetry` completion paint a second, differently-registered copy. |
| 2 | **Ghost face fragments painted into hair.** A clear second mouth (pink lips) and eye fragments inside the hair curtain at the lower-left of the front view; ghost features also detectable from behind: a face detector fires at az +112.5 (score 0.74) and -112.5 (0.72) — on the BACK-side of the head. | az -22.5..+22.5 lower-left; az +/-112.5 | `crops/azm22_cheek_hair_edge.png` (lips in hair), `crops/az0_left_ear_region.png`, `identity/azimuth_sweep.json` | **BLOCKING** | Same misregistration + depth/visibility leak at grazing angles: displaced front projection lands on hair-strand side surfaces. |
| 3 | **Skin-colored flake fields painted over scalp hair.** Large forehead/temple-skin patches (including a ghost eyebrow) shredded across the hair strands; worst on the crown and temples. | every view except 135..180..-135; worst az -45..+90, el +15 | `crops/az0_hairline.png`, `crops/azp45_hairline.png`, `crops/azm90_hairline_temple.png`, `crops/azp112_hair.png` | **BLOCKING** | Projection visibility on strand-carved hair micro-geometry: skin texels assigned to hair charts along the mis-registered skin/hair boundary; tiny fragmented UV charts (see texture atlas) make every error a hard-edged flake. |
| 4 | **Black flake debris on forehead/hairline skin.** Dark hair texels scattered over forehead and temple skin; no pure-black (unfilled) texels exist in texture.png (0.0% below luminance 5), so these are misprojected hair pixels, not padding holes. | az -67.5..+90, all elevations | `crops/az0_hairline.png`, `crops/azp45_hairline.png`, `crops/azp90_profile.png` | **BLOCKING** | Same as #3, opposite direction (hair texels onto skin charts). |
| 5 | **Milky translucent patches over facial features.** Desaturated beige bars across the mouth (a "bar mouth" lying over the lips), under both eyes, and on the jaw; they partially erase the mouth at every front view. | az -45..+45, all elevations | `crops/az0_nose_mouth.png`, `crops/az0_face.png`, `crops/azm45_cheek_ear.png` | **BLOCKING** | `unseen_fill_mode: mesh_harmonic` fill and inter-view blending painted over face regions whose visibility test wrongly rejected the photo views (rejection is a downstream symptom of the pose misregistration). |
| 6 | **Ear-like patch on the cheek / sideburn.** Ear-fold texture displaced forward of the geometric ear; the geometric ear itself receives a mix of skin flakes and black streaks. | az -45/-22.5 (left ear region), az +45 | `crops/az0_left_ear_region.png`, `crops/azm45_face.png` (left edge), `crops/azp90_ear.png` | MAJOR | Profile-reference projection: the synthetic profile's ear does not coincide with the mesh ear position/scale; misplaced by the same registration error. |
| 7 | **Eye smeared toward temple at pure profiles.** At +/-90 the eye texture drags horizontally into the temple; a dark artifact stripe runs from brow toward the nose. | az +90, az -90 | `crops/azp90_profile.png`, `crops/azm90_profile.png` | MAJOR | Grazing-angle projection stretch at the sector boundary between front and profile views. |
| 8 | **Pink lip-colored debris in hair** (right side, reads like a smeared earring). | az +22.5..+67.5 | `crops/azp22_face.png` (lower right), `crops/azp45_face.png` | MAJOR | Misprojected lip/skin texels landing in hair charts. |
| 9 | **Desaturated smear band down the back of the head** (flat, blurred brown; reads as a bald patch of paint). | az 157.5..180..-157.5 | `crops/az180_back.png` | MINOR-MAJOR | Harmonic fill of the region unseen by all 3 views; expected but too visible. |
| 10 | **Ragged skin-colored fringe on hair silhouette** at back/top; ears poke through hair with skin halos from behind. | az 180, +/-157.5 | `crops/az180_back.png` | MINOR | Silhouette texels sampling background/skin at mask edges. |
| 11 | **Strap/chest fragments smeared** (white strap broken into blobs on shoulders). | az 0 el -10/0 bottom | `crops/az0_neck_chest.png` | MINOR | Projection at steep angle on chest; low-priority region. |
| 12 | **Faceted flat-color polygons at close zoom** on chin/jaw fill regions (hex-pattern shading visible in atlas blob). | close zoom only | `intermediates/texture_q_bl.png` (bottom-left blob) | MINOR | Vertex-domain harmonic fill rasterized to charts. |

The geometry itself (untextured) shows none of these pathologies: `crops/geom_az0_face.png` is a coherent, symmetric, plausible female head with credible eyes/nose/lips/ears and strand-carved hair. **All blocking defects live in the texture stage.**

## T2 — Identity check (photo vs frontal renders)

Method: OpenCV YuNet 5-point landmarks; scale-invariant ratios; annotated images in `identity/landmarks_*.png`; numbers in `identity/identity_metrics.json`, sweep in `identity/azimuth_sweep.json`; overlay panel `identity/overlay_photo_vs_az0.png`, checkerboard `identity/checker_photo_vs_az0.png`.

| Metric | Photo | Render az 0 | Delta | Render az -22.5 |
|---|---|---|---|---|
| eye-to-mouth / interocular | 1.080 | 1.210 | **+12.0%** | 1.031 (-4.5%) |
| mouth width / interocular | 0.807 | 0.787 | -2.4% | 0.843 (+4.5%) |
| nose drop / interocular | 0.686 | 0.693 | +1.0% | 0.648 (-5.5%) |
| interocular / face height | 0.318 | 0.338 | +6.5% | 0.366 |
| eye-line roll (deg) | 0.58 | **10.76** | +10.2 deg | 0.58 (exact) |
| detector score | 0.94 | 0.885 | — | 0.884 |

**Verdict: FAIL.** A frontal render of a level head must not show a 10.8 deg eye-line roll and a +12% vertical stretch. Worse, the "face" the detector locks onto at az 0 is not a real feature set: its two "eyes" are the photo's left eye (painted near the geometry's nose bridge) and a ghost eye patch — visible in `identity/landmarks_render_az0.png`. The checkerboard blend (`identity/checker_photo_vs_az0.png`) shows photo tiles and render tiles do not continue into each other anywhere on the face. The best-matching azimuth for the painted face is **-22.5 deg, not 0** (mean ratio error 4.9% vs 5.1% at az 0 but with exact roll match and single-copy features) — the quantitative signature of the projection misregistration. Additionally, faces are detectable at az +112.5 and -112.5 (in the hair, rolls 94.5 deg / -10.6 deg): ghost copies confirmed numerically.

Even if all of the above were fixed, note the ceiling: the +/-90 references are themselves synthetic guesses (see T4), so profile likeness to the real person is unverifiable; only front-view likeness is a testable claim.

## T3 — Overclaiming review of developer evidence

1. **`iter3-multiview-fixed/face_before_after_fixed.png`** — row labels "BEFORE (first multi-view attempt)" / "AFTER (all root-cause fixes + agent audit patches)". At 2x magnification the AFTER row still contains every blocking defect class: tile 0 (front) shows the three-eye chimera, ghost mouth in hair, skin-flake scalp, black hairline debris (`overclaim/iter3_after_tile0.png`); tile 7 (three-quarter) shows doubled nose/eyes and flake fields (`overclaim/iter3_after_tile7.png`). The AFTER row is genuinely better than BEFORE (`overclaim/iter3_before_tile0.png` shows global brown corruption on the face), but the label "all root-cause fixes" is **not supported**: the root cause (projection registration) is demonstrably still active in the AFTER pixels. Honest label: "partial mitigation; face still incoherent at front views."
2. **`iter2-multiview/face_before_after.png`** — labels "Hunyuan3D-2.1 single view (before)" / "Hunyuan3D-2mv + multi-view bake (after)". The after-row front tile (`overclaim/iter2_row2_tile0.png`) is worse than the iter3 one (two full mouths, garbled eye row, heavy debris). The labels themselves are descriptive and neutral, so the sheet is honest as a record; any use of it as evidence of success was overclaiming.
3. **Prior status claim "coherent at all azimuths"** — demolished: 10 of 16 azimuths at el 0 show blocking or major defects; only the back hemisphere hair mass (135..180..-135) reads as coherent, and even it has the smear band (#9).

## T4 — Intermediates sanity check

- **`texture.png`** (2048x2048): the atlas is thousands of tiny hard-edged charts. Multiple face-feature copies exist in the atlas (several eyes/mouths in different charts — expected for fragmented UVs, but here adjacent charts disagree about what they show). A large faceted flat blob (harmonic fill) occupies the bottom-left. Views: `intermediates/texture_1024.png`, quadrants `intermediates/texture_q_*.png`. No unfilled black texels (0% below luminance 5) — flake debris is misprojected content, not padding.
- **`uv_preview.png`**: same fragmentation; as an audit artifact it is unreadable at any zoom — it cannot support a visual claim about seam quality.
- **`input.png`** in the bundle is pixel-identical to `/Users/albou/Downloads/test-face.png` (mean abs diff 0.0). Provenance OK.
- **Profile references**: `right_profile_clean.png` is **pixel-exactly the horizontal mirror of `left_profile_clean.png`** (mean abs diff 0.0, `intermediates/flippedleft_vs_right_sbs.png`). So the -90 "reference" adds no real information; ears/hair-part are cloned, and both profiles are themselves generated images (a guess at this person's profile), not photos. Segmentation of the profiles is clean: soft alpha matte, tight edge, no holes (`intermediates/left_profile_clean_alpha.png`); note alpha never reaches 255 (max 254) — harmless but sloppy matte convention. The mirrored profile is plausible except for the physically-unlikely perfect symmetry; it is not the cause of the blocking defects.
- **Metadata red flags**: `source_pose: {estimated: false, silhouette_iou: 0.0}` — the pose was assumed, and the recorded alignment score is zero; a 0.0 IoU should have failed a gate, not shipped. `observed_coverage_ratio: 0.57` with `symmetry coverage 0.096` means ~40% of the surface is fill/symmetry content.

## T5 — Definition of "fixed" (acceptance views, in order of importance)

A human will accept this asset only when ALL of the following hold in >= 800 px renders (el 0 unless stated):

1. **az 0 (and el -10/+15)**: exactly ONE set of facial features, aligned to the geometry's eye sockets/nose/lips; landmark ratios within 3% of the photo; eye-line roll < 2 deg; no milky patches over mouth/eyes; no ghost features anywhere in the frame, including the hair.
2. **az +/-22.5 and +/-45**: single nose (silhouette nose = painted nose), single mouth, single eye pair; no second-copy fragments on cheeks; no ear texture on the cheek/sideburn.
3. **Hairline band, az -67.5..+67.5, el 0 and +15**: forehead/temple skin free of black hair flakes; scalp hair free of skin flakes and ghost eyebrows (zoomed crops at native resolution).
4. **az +/-90**: eye not smeared into the temple; ear reads as an ear with clean sideburn boundary and no black streaks.
5. **az +/-112.5..+/-157.5**: no face detectable in the hair (YuNet score < 0.3 at threshold 0.25); no skin patches inside the hair mass other than the ear.
6. **az 180**: no desaturated smear band wider than ~15% of head width; silhouette fringe skin-free except ears.
7. **texture.png**: no facial-feature content in charts that map to hair geometry; harmonic-fill fraction over face-visible charts of a few percent at most.

Items 1-5 are blocking; 6-7 are strong-should. Verification must be at full resolution — the 320 px contact-sheet tiles hide everything that matters.

### What is already acceptable (precise)

- **Untextured geometry** at all azimuths/elevations: coherent head, plausible face, symmetric, ears present, strand-carved hair volume (`grid_geometry_el0.png`). The mesh stage is not the problem.
- **Back-hemisphere hair mass** (az 135..180..-135) textured: coherent color and strand flow, apart from defects #9/#10.
- **Pure-profile lower face** at +/-90: nose/lips/chin/jaw profile reads correctly at full resolution (upper face and hairline do NOT pass — defects #3/#4/#7).
- **Neck/chest/shoulder skin**: clean aside from minor #11.
- **Provenance**: bundle `input.png` is the true source photo; profile segmentation masks are clean.

### Attribution summary (for whoever fixes it)

One primary root cause explains defects 1, 2, 5, 6 and most of 3/4: **the front photo is projected as if taken at azimuth 0 while the subject's head is yawed ~20 deg in it** (`estimated: false`, best-match render azimuth -22.5). Everything painted from that view lands ~20 deg away from the matching geometry; the profile/symmetry content then disagrees with it, and the visibility/blend machinery scatters the disagreement into flakes and milky fill. Fixing pose estimation (or refusing to bake when silhouette IoU is 0.0) is the gate that would have caught this before shipping.
