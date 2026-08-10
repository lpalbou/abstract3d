# Independent adversarial verification — laurent-bust-redo candidates

Date: 2026-07-21. Verifier: Adversary I (independent; project metric scripts NOT used for judgment).
Instrument: full orbit renders read visually, both clay and textured, at 1024–1600 px.

## Method

- Renderer: `scripts/verifier_orbit_render.py` (written for this audit). It calls only
  `abstract3d.rendering.render_mesh_views` (moderngl headlight renderer — pyrender is not
  installed in the venv) and makes no quality judgment. Clay = geometry-only copy of the
  mesh (headlight shading, even illumination, creases read as valleys); tex = the GLB's
  baked texture near-flat-lit so albedo dominates.
- Coverage per model: azimuths every 30° at el 0 (12 views) + az {0,60,120,180,240,300}
  at el +20, clay + tex (36 renders), plus face crops at az {0,±30,±60,±90} clay + tex
  (14 crops), plus labeled orbit/face contact sheets and one annotated defect sheet.
- Evidence root: `out/laurent-bust-redo/review/verifier/<model>/`. All paths in the table
  below are relative to that root. Per-model annotated sheets: `<model>/sheet_annotated.png`.
- Judgments were made by reading the rendered images (sheets first, then full-resolution
  drilling on ~20 suspect views). One small measurement script was used only to test mesh
  identity between the two e20 variants (vertex/face counts + geometry hash): they are
  **byte-identical meshes**; only the texture differs.

### Orientation convention (matters for "left/right" claims)

Cameras orbit counterclockwise: **az90 shows the subject's left side, az270 the subject's
right**. In a front view (az0), image-left is the **subject's right**. The ground-truth
photo shows the over-ear cable on image-right (his left as a mirrored selfie shows it);
the models place the cable/earpiece consistently with the photo's image layout. Operator
reports of "left-side" defects given below are mapped to explicit azimuths, which is
unambiguous.

Severity scale: **BLOCKER** = obviously wrong to any human at arm's length;
**MAJOR** = wrong on inspection; **MINOR** = cosmetic. Model grades use the worst view,
never the average — one BLOCKER angle makes the model a BLOCKER.

## Defect table

One row per defect. Angles are azimuth/elevation of the evidencing render (el 0 unless
stated). Paths relative to `out/laurent-bust-redo/review/verifier/`.

### e10_2mv_registered_refs

| # | Defect | Angle | Severity | Evidence |
|---|--------|-------|----------|----------|
| 1 | White/chrome smear projected over right jaw + neck (window/shirt content on skin) | az240–az330 tex | BLOCKER | `e10_2mv_registered_refs/az270_el0_tex.png`, `az300_el0_tex.png` |
| 2 | Ghost second-face fragment (eye/brow/cheek) at right sideburn inside the smear | az300 tex | BLOCKER | `e10_2mv_registered_refs/az300_el0_tex.png` |
| 3 | Pale mottled smear at right temple (image-left in front view) | az0 tex | MAJOR | `e10_2mv_registered_refs/face_az0_tex.png` |
| 4 | Red parted-lip texture (photo: closed mouth) | az0/az90 tex | MAJOR | `e10_2mv_registered_refs/face_az0_tex.png` |
| 5 | Tan "cardboard" rectangular patch in hair at back/crown | az180 tex | MAJOR | `e10_2mv_registered_refs/az180_el0_tex.png` |
| 6 | Gray wash/streaks in right-side hair | az300 tex | MINOR | `e10_2mv_registered_refs/az300_el0_tex.png` |
| 7 | Skin-color bleed on shirt shoulder/sleeve | az240 tex | MINOR | `e10_2mv_registered_refs/az240_el0_tex.png` |
| 8 | Nose flattened + widened, flared nostrils, upturned deformed tip | az0/az90 clay | MAJOR | `e10_2mv_registered_refs/face_az0_clay.png`, `face_az90_clay.png` |
| 9 | Lips sculpted parted/everted ("statue lips"; photo closed) | az0/az90 clay | MAJOR | `e10_2mv_registered_refs/face_az0_clay.png` |
| 10 | Earpiece rendered as a crude brick at right temple | az0 clay | MAJOR | `e10_2mv_registered_refs/face_az0_clay.png` |
| 11 | Cable broken/kinked with floating segment below ear | az90 clay | MINOR | `e10_2mv_registered_refs/face_az90_clay.png` |
| 12 | Lumpy cheek surface; stepped hair shelf at crown; jagged bust-cut shoulder edge | various clay | MINOR | `e10_2mv_registered_refs/az90_el0_clay.png` |

### e11_2mv_reg_hq

| # | Defect | Angle | Severity | Evidence |
|---|--------|-------|----------|----------|
| 1 | Face reads melted/mannequin: nose eroded to a hooked stub with notch, near-flat profile — lips protrude further than the nose | az90 clay | BLOCKER | `e11_2mv_reg_hq/face_az90_clay.png`, `face_az0_clay.png` |
| 2 | Wide slit "grimace" mouth carved into the face | az0 clay | MAJOR | `e11_2mv_reg_hq/face_az0_clay.png` |
| 3 | Glasses: winged plate with V-notch; nose nub pokes through the lens plane | az0 clay | MAJOR | `e11_2mv_reg_hq/face_az0_clay.png` |
| 4 | Left ear melted flat into the head | az90 clay | MAJOR | `e11_2mv_reg_hq/az90_el0_clay.png` |
| 5 | Shredded hair: thin flakes/spikes at crown and top | orbit clay | MAJOR | `e11_2mv_reg_hq/sheet_clay_orbit.png` |
| 6 | Wedge hair fin at back-left crown | az180 clay | MINOR | `e11_2mv_reg_hq/az180_el0_clay.png` |
| 7 | Floating wire stub below jaw (detached geometry) | az90 clay | MINOR | `e11_2mv_reg_hq/face_az90_clay.png` |
| 8 | Doubled glasses in texture: second translucent pair layered under/around the dark pair, skin visible through lens bottoms | az0 tex | MAJOR | `e11_2mv_reg_hq/face_az0_tex.png` |
| 9 | Pale skin smear on right cheek + large black blotch at jaw hinge + over-bright pale ear | az330 tex | MAJOR | `e11_2mv_reg_hq/face_az330_tex.png` |
| 10 | Dark chips/shadow notches along jawline | az330 tex | MINOR | `e11_2mv_reg_hq/face_az330_tex.png` |
| 11 | Pale streak smears in beard | az0 tex | MINOR | `e11_2mv_reg_hq/face_az0_tex.png` |

### e15_cleanfront

| # | Defect | Angle | Severity | Evidence |
|---|--------|-------|----------|----------|
| 1 | Front texture combo: flesh-colored brick floating in hair (earpiece) + white streak projected across left lens edge, lens bottom translucent | az0 tex | BLOCKER | `e15_cleanfront/face_az0_tex.png` |
| 2 | Floating antenna rod/paddle above right temple, attached only at glasses hinge — visible in silhouette | az0/az300 clay | MAJOR | `e15_cleanfront/face_az0_clay.png`, `az0_el0_clay.png` |
| 3 | Square slab patch embedded in right forehead/hairline | az0 clay | MAJOR | `e15_cleanfront/face_az0_clay.png` |
| 4 | Button/clown nose (tiny round tip vs photo's broader nose) | az90 clay | MAJOR | `e15_cleanfront/face_az90_clay.png` |
| 5 | Pursed protruding duck lips with wide mouth groove | az90 clay | MAJOR | `e15_cleanfront/face_az90_clay.png` |
| 6 | Acne-like bump noise (raised welts) across cheeks and jaw | az0/az30 clay | MAJOR | `e15_cleanfront/face_az30_clay.png` |
| 7 | Orange/brown smudge in right-side hair + gray-yellow blob on right shoulder | az300 tex | MAJOR | `e15_cleanfront/az300_el0_tex.png` |
| 8 | Beige/gray "mold" smudges in hair; bright pale ear with skin smear in front | az330 tex | MAJOR | `e15_cleanfront/face_az330_tex.png` |
| 9 | Big protruding ears with slab behind | az90 clay | MINOR | `e15_cleanfront/az90_el0_clay.png` |
| 10 | Cable with floating dangling loop over chest | az30 clay | MINOR | `e15_cleanfront/az30_el0_clay.png` |
| 11 | Toga-like drape folds on right shoulder | az240–300 clay | MINOR | `e15_cleanfront/az240_el0_clay.png` |
| 12 | Mustache gap (pale patch under nose); tan shoulder smudge; pale neck streak under right jaw | az0/az60 tex | MINOR | `e15_cleanfront/face_az0_tex.png`, `az60_el0_tex.png` |

### e17_clean4

| # | Defect | Angle | Severity | Evidence |
|---|--------|-------|----------|----------|
| 1 | Glasses are a giant fused visor slab swallowing the upper face, merged into the nose bridge, extending past the cheeks with beak overhang | az0/az270 clay | BLOCKER | `e17_clean4/face_az0_clay.png`, `face_az270_clay.png` |
| 2 | Hair shredded into tentacle spikes — silhouette wrong at arm's length | az90/az270 clay | BLOCKER | `e17_clean4/az90_el0_clay.png` |
| 3 | Bulbous nose blob fused under the visor (profile) | az270 clay | BLOCKER | `e17_clean4/face_az270_clay.png` |
| 4 | Doubled everted duck lips, S-profile, parted gap | az270 clay | BLOCKER | `e17_clean4/face_az270_clay.png` |
| 5 | Bright white glow smear on right cheek in front of ear ("ghost stripe" with dark blotch) | az270/az300 tex | BLOCKER | `e17_clean4/face_az270_tex.png` |
| 6 | Black ghost silhouette drip painted over right beard/cheek | az270 tex | BLOCKER | `e17_clean4/face_az270_tex.png` |
| 7 | Visor texture: dark band on top, translucent skin/beard below, gray flared wings, nostril dots painted on the visor edge | az0 tex | BLOCKER | `e17_clean4/face_az0_tex.png` |
| 8 | Lens misprojection: white/skin blobs + dark bars inside lens | az270 tex | MAJOR | `e17_clean4/face_az270_tex.png` |
| 9 | Vertical striping on neck skin (left side) | az90–120 tex | MAJOR | `e17_clean4/az90_el0_tex.png` |
| 10 | Rolled scarf/sash ridge across right shoulder + chest | az240 clay | MAJOR | `e17_clean4/az240_el0_clay.png` |
| 11 | Tan smudges + bright tears on shirt (back/shoulders) | az180/az240 tex | MINOR | `e17_clean4/az240_el0_tex.png` |
| 12 | Thin neck relative to head; asymmetric hunched shoulders | az90 clay | MINOR | `e17_clean4/az90_el0_clay.png` |

### e18_windowed

| # | Defect | Angle | Severity | Evidence |
|---|--------|-------|----------|----------|
| 1 | Mouth OPEN: parted lips with dark gap, in clay AND texture (photo: closed) | az0/az90 | BLOCKER | `e18_windowed/face_az0_clay.png`, `face_az90_clay.png`, `face_az0_tex.png` |
| 2 | Wrong glasses: thin swooping aviator visor with center gap; the nose pierces through the bridge notch | az0 clay | BLOCKER | `e18_windowed/face_az0_clay.png` |
| 3 | Ghost eyes + brow band painted on the forehead above the visor | az0 tex | BLOCKER | `e18_windowed/face_az0_tex.png` |
| 4 | Glasses texture painted smaller/lower than mesh — dark butterfly blob mid-visor spilling onto cheeks | az0 tex | BLOCKER | `e18_windowed/face_az0_tex.png` |
| 5 | Whole-head double exposure: hair strands bleed across the face, ghost second face on the right side | az90/az270 tex | BLOCKER | `e18_windowed/az90_el0_tex.png`, `az270_el0_tex.png` |
| 6 | White-gray smoke/curtain smear over neck, shoulders, shirt | az240 tex | BLOCKER | `e18_windowed/az240_el0_tex.png` |
| 7 | Knob/bun blob at top-center hairline | az0 clay | MAJOR | `e18_windowed/face_az0_clay.png` |
| 8 | Nose overlong and droopy (witch-like) in profile | az90 clay | MAJOR | `e18_windowed/face_az90_clay.png` |
| 9 | Gray marbled smear on back of shirt; bald-ish gray crown patch | az180 tex | MAJOR | `e18_windowed/az180_el0_tex.png` |
| 10 | Chrome smear on right neck | az240 tex | MAJOR | `e18_windowed/az240_el0_tex.png` |
| 11 | Diagonal strap/fold ridge across chest with ragged sleeve edges | az240 clay | MAJOR | `e18_windowed/az240_el0_clay.png` |
| 12 | Window-light streaks in hair on both sides | az90/az270 tex | MAJOR | `e18_windowed/az90_el0_tex.png` |

### e20_fixed_views

Mesh rows also apply to `e20_rebake_fixed` (byte-identical geometry).

| # | Defect | Angle | Severity | Evidence |
|---|--------|-------|----------|----------|
| 1 | Global upward texture displacement (~10–15% of face height): painted glasses band occupies the top half of the lens area with translucent skin/beard below; beard rises to the glasses line; nose tip carries mustache texture; red lips at the philtrum; eyebrows visible above the frame | all face views tex | BLOCKER | `e20_fixed_views/face_az0_tex.png`, `face_az90_tex.png` |
| 2 | Doubled glasses-arm band: arm texture on the mesh arm AND again on the hair above it | az90 tex | MAJOR | `e20_fixed_views/face_az90_tex.png` |
| 3 | Cyan/turquoise out-of-palette patch behind the ear at the hairline | az270 tex (also visible az180) | MAJOR | `e20_fixed_views/az270_el0_tex.png` |
| 4 | Dark drip seam on right cheek (precursor of the rebake's crack) | az330 tex | MAJOR | `e20_fixed_views/face_az330_tex.png` |
| 5 | Pale ear patches + white streaks in front of ears; displaced sideburn on cheek | az90/az270 tex | MAJOR | `e20_fixed_views/face_az90_tex.png` |
| 6 | Pale streak/bald patch at front crown hairline | az0 el20 tex | MINOR | `e20_fixed_views/az0_el20_tex.png` |
| 7 | Warm amber glow at nape/collar | az180 tex | MINOR | `e20_fixed_views/az180_el0_tex.png` |
| 8 | MESH: crown dents + spikes, crater-like lumps | az0 el20 clay | MINOR–MAJOR | `e20_fixed_views/az0_el20_clay.png` |
| 9 | MESH: draped sash ridge across right shoulder/chest | az240–300 clay | MINOR | `e20_fixed_views/az240_el0_clay.png` |
| 10 | MESH: left ear protrudes with slab behind; irregular hairline ridge | az120 clay | MINOR | `e20_fixed_views/az120_el0_clay.png` |
| 11 | MESH: slightly weak chin, abrupt neck-jaw transition | az90 clay | MINOR | `e20_fixed_views/face_az90_clay.png` |

Mesh positives (for the record): correct flat-top slab glasses with temple arms, straight
proportioned nose, closed well-formed mouth, formed ears, clean back — the best mesh of
the eight by a clear margin.

### e20_rebake_fixed

Texture only (mesh identical to e20_fixed_views). Center-face registration IS improved
over e20_fixed_views (mustache/nose/lips land in place) — but new blocker-class damage
was introduced elsewhere. The "fixed" claim is **refuted**.

| # | Defect | Angle | Severity | Evidence |
|---|--------|-------|----------|----------|
| 1 | Jagged dark seam/crack down the right cheek and jaw with pale infill (stronger and more jagged than e20_fixed's) | az0/az330 tex | BLOCKER | `e20_rebake_fixed/face_az330_tex.png`, `face_az0_tex.png` |
| 2 | Amber/orange/teal misprojection filling the right lens (reads like a landscape photo in the lens) | az270/az330 tex | BLOCKER | `e20_rebake_fixed/az270_el0_tex.png`, `face_az330_tex.png` |
| 3 | Cream/beige vertical wash down the left neck from below the ear to the collar (out of palette — wall/pillow color on skin) | az90 tex | BLOCKER | `e20_rebake_fixed/az90_el0_tex.png` |
| 4 | Large cream nape patch from hairline to collar, spilling left | az180 tex | BLOCKER | `e20_rebake_fixed/az180_el0_tex.png` |
| 5 | Dark crescent "bald hole" behind the left ear | az90 tex | MAJOR | `e20_rebake_fixed/az90_el0_tex.png` |
| 6 | Red blotch on right cheek near the ear (+ teal specks) | az330 tex | MAJOR | `e20_rebake_fixed/face_az330_tex.png` |
| 7 | Gray smoke wash + pale streaks on the crown hair | az330/az180 tex | MAJOR | `e20_rebake_fixed/face_az330_tex.png` |
| 8 | Patchy beard: pale center gap at chin, washed-out right side | az0 tex | MAJOR | `e20_rebake_fixed/face_az0_tex.png` |
| 9 | Hair/skin streaks painted on the left lens; flesh streak along glasses-arm top edge | az0/az60 tex | MAJOR | `e20_rebake_fixed/face_az0_tex.png`, `az60_el0_tex.png` |
| 10 | Ghost eyebrow along the lens upper edge | az0 tex | MINOR | `e20_rebake_fixed/face_az0_tex.png` |
| 11 | Skin blotches at right sleeve/armpit; black jagged blob on right arm; skin patch at bottom-left sleeve | az180/az240 tex | MINOR | `e20_rebake_fixed/az180_el0_tex.png` |
| 12 | Beard pigment bleed onto nose tip | az90 tex | MINOR | `e20_rebake_fixed/az90_el0_tex.png` |

### e21_single_refs

| # | Defect | Angle | Severity | Evidence |
|---|--------|-------|----------|----------|
| 1 | Ghost eye + eyebrow painted in the hair above the glasses arm | az300 tex | BLOCKER | `e21_single_refs/face_az300_tex.png` |
| 2 | Translucent lenses with duplicated dark eye-patch fragments showing through | az300/az0 tex | BLOCKER | `e21_single_refs/face_az300_tex.png`, `face_az0_tex.png` |
| 3 | Skewed dark trapezoid lens patches misaligned with the mesh lens | az0 tex | MAJOR | `e21_single_refs/face_az0_tex.png` |
| 4 | Black drip tendril from the left jaw down the neck; smeared beard edges | az0/az90 tex | MAJOR | `e21_single_refs/face_az0_tex.png` |
| 5 | Bright pale patch on right cheek + jagged black seam | az330 tex | MAJOR | `e21_single_refs/face_az330_tex.png` |
| 6 | Phantom skin patches at both temples embedded in the hair | az0 tex | MAJOR | `e21_single_refs/face_az0_tex.png` |
| 7 | MESH: glasses shape wrong — thin curved wraparound with scalloped lower edge (photo: large flat-top slab) | az0 clay | MAJOR | `e21_single_refs/face_az0_clay.png` |
| 8 | MESH: forehead shelf — hairline juts like a cap brim with deep temple crease | az90 clay | MAJOR | `e21_single_refs/face_az90_clay.png` |
| 9 | MESH: left ear crater (pit in the concha) | az90 clay | MAJOR | `e21_single_refs/face_az90_clay.png` |
| 10 | MESH: deep nasolabial scowl folds + downturned mouth (photo expression neutral) | az0 clay | MINOR–MAJOR | `e21_single_refs/face_az0_clay.png` |
| 11 | MESH: antenna strands sprouting from the crown | az0 clay | MINOR | `e21_single_refs/az0_el0_clay.png` |
| 12 | Doubled top bar on frame; tan bleed on right shoulder/sleeve; pale hairline streaks | az300/az240 tex | MINOR | `e21_single_refs/face_az300_tex.png`, `az240_el0_tex.png` |

## Operator cross-check

| Operator claim | Verdict | Evidence |
|---|---|---|
| e18: open mouth + awful texture | **CONFIRMED** (both, blocker-grade) | `e18_windowed/face_az0_clay.png`, `face_az0_tex.png`, `az90_el0_tex.png` |
| e10: left-side white smear + mediocre front | **CONFIRMED**, side clarified: the smear sits on the subject's RIGHT (= image-left when facing him), az240–330; front is mediocre (temple smear, flat nose, statue lips) | `e10_2mv_registered_refs/az270_el0_tex.png`, `face_az0_tex.png` |
| e17: bulbous nose / doubled lips at profile + bad texture | **CONFIRMED** (all three, blocker-grade) | `e17_clean4/face_az270_clay.png`, `face_az270_tex.png` |
| e21: ghost glasses | **CONFIRMED** — ghost eye/brow above the arm, translucent lenses with eye fragments, skewed lens patches | `e21_single_refs/face_az300_tex.png`, `face_az0_tex.png` |
| e20: good mesh / displaced texture | **CONFIRMED** — best mesh of 8; texture globally shifted up ~10–15% of face height | `e20_fixed_views/face_az0_tex.png`, `az0_el20_clay.png` |
| e20_rebake_fixed: texture fixed | **REFUTED** — center registration improved, but new blockers: stronger right-jaw seam, amber/teal lens garbage, cream neck wash + cream nape patch. Checked both sides, right jaw seam, temples, and back of head as instructed | `e20_rebake_fixed/face_az330_tex.png`, `az270_el0_tex.png`, `az90_el0_tex.png`, `az180_el0_tex.png` |

## MESH ranking (worst-view basis; clay renders)

| Rank | Model | Worst-view grade | Basis |
|---|---|---|---|
| 1= | e20_fixed_views | MINOR–MAJOR | Correct flat-top glasses, straight nose, closed mouth, formed ears. Worst view: crown dents/spikes at el20. No blocker. |
| 1= | e20_rebake_fixed | MINOR–MAJOR | Byte-identical mesh to e20_fixed_views. |
| 3 | e21_single_refs | MAJOR | Coherent head, closed mouth, decent nose — but wrong glasses model (scalloped wraparound), forehead shelf, ear crater. |
| 4 | e10_2mv_registered_refs | MAJOR | Proportions OK, but flattened wide nose, statue parted lips, earpiece brick — identity features deformed. |
| 5 | e15_cleanfront | MAJOR (borderline BLOCKER) | Button nose, duck lips, acne bump noise, floating antenna rod in silhouette. |
| 6 | e18_windowed | BLOCKER | OPEN mouth; wrong swooping glasses with nose piercing the bridge; knob at hairline. |
| 7 | e11_2mv_reg_hq | BLOCKER | Face melted to mannequin: nose a stub, near-flat profile, slit mouth, flat ear, shredded hair. |
| 8 | e17_clean4 | BLOCKER | Visor fused over half the face, shredded tentacle hair, doubled duck lips. Worst mesh by far. |

## TEXTURE ranking (worst-view basis; tex renders)

| Rank | Model | Worst-view grade | Basis |
|---|---|---|---|
| 1 | e15_cleanfront | BLOCKER (borderline) | Least-bad of 8, but the front combo (flesh brick in hair + white lens streak) is arm's-length visible; beige "mold" hair smudges on the sides. |
| 2 | e11_2mv_reg_hq | MAJOR | Cleanest by defect class: doubled translucent glasses front-center; pale cheek smear + black jaw blotch az330. No out-of-palette chrome/ghost faces. |
| 3 | e20_rebake_fixed | BLOCKER | Best center-face registration, but jagged jaw seam, amber/teal lens garbage, cream neck+nape wash across many angles. |
| 4 | e10_2mv_registered_refs | BLOCKER | Chrome/white smear over right jaw+neck with ghost face fragment; front merely mediocre. |
| 5 | e21_single_refs | BLOCKER | Ghost eye/brow in hair, translucent lens fragments, black drip tendril, cheek seam. |
| 6 | e20_fixed_views | BLOCKER | Systematic displacement wrecks every face view including the hero front (beard on nose, lips at philtrum). |
| 7 | e17_clean4 | BLOCKER | White glow smear + black ghost drip + visor band with painted nostril dots. |
| 8 | e18_windowed | BLOCKER | Whole-head double exposure, ghost eyes on forehead, smoke shirt — texture chaos everywhere. |

## Verdict

**No current bundle is presentable as "good." All eight fail on at least one axis at
blocker grade.**

- The only good mesh (e20 pair) is bound to two broken textures: e20_fixed_views is
  globally displaced; e20_rebake_fixed traded displacement for seams, lens garbage, and
  out-of-palette neck/nape washes.
- The least-bad textures (e15, e11) sit on deformed meshes: e15's button nose / duck lips
  / antenna rod, e11's melted face. Neither mesh should carry the final texture.
- The rebake's direction (fixing registration) is real — its center face is the best
  single face crop across all 8 models — but the collateral damage it introduces is
  arm's-length visible from multiple angles, so it cannot ship either.

### Top 3 blockers between the best current model (e20 mesh + its texture pipeline) and acceptable

1. **Texture-to-mesh registration/projection.** e20_fixed's bake lands the whole face
   ~10–15% too high (beard on nose, lips at philtrum, glasses band in the top half of the
   lens); the rebake corrects the center but produces a jagged seam down the right
   cheek/jaw and misprojected content at visibility boundaries. Until a bake lands
   features on the mesh features without seams at projection boundaries, every candidate
   fails at arm's length.
2. **Occlusion fill pulls out-of-palette background color.** Regions unseen or grazed by
   the source views get filled with room/backdrop colors instead of hair/skin/shirt:
   cream/beige wash on left neck + nape (rebake), cyan patch behind the ear (fixed),
   gray smoke on the crown, chrome/white window smears (e10/e18). Inpainting must be
   constrained to the subject's palette per region (hair/skin/cloth).
3. **Glasses lens material purity.** On every one of the 8 models the lenses are
   contaminated — skin/hair painted into the lens, translucency exposing ghost eyes,
   amber/teal misprojections, doubled frames. The flat dark sunglasses are the face's
   dominant identity feature; they need a dedicated lens mask/material (single uniform
   dark slab, opaque) rather than being baked like skin.

Secondary (mesh side, not in the top 3): e20's crown dents/spikes at el20 and the
shoulder sash ridge — worth fixing once texture blockers fall.

## Evidence index

- Renders + labeled sheets: `out/laurent-bust-redo/review/verifier/<model>/`
  (36 orbit renders + 14 face crops + 4 labeled sheets per model).
- Annotated defect sheet per model: `out/laurent-bust-redo/review/verifier/<model>/sheet_annotated.png`.
- Render script: `scripts/verifier_orbit_render.py`. Annotation script (also the
  machine-readable defect log): `scripts/verifier_annotate.py`.
