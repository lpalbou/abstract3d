# CORRELATION AGENT A — MeshVault visual forensics of the certified assets
Date: 2026-07-07. Repo: `/Users/albou/abstract3d`. Viewer: MeshVault MCP
(stdio, headless Chromium; tools per `/Users/albou/MeshVault/docs/mcp.md`).
All screenshots (186 PNGs at 1024 px, of which 54 are historical-state
shots): `artifacts/validation/texture-cycle-proofs/meshvault/agentA/`
(referenced below by file name only). Raw tool outputs: `/tmp/mva/t1_*.json`,
`/tmp/mva/t2_compare.json`, `/tmp/mva/t4_measure.json`; driver scripts
`/tmp/mva/*.py` (client.py is the persistent MCP client; every number in
this report is reproducible by re-running them).

## EXECUTIVE SUMMARY

- The three certified assets, inspected in the owner's own external viewer,
  read as certified: every ledgered defect class is either absent at the
  certification's viewing standard or present exactly as its disclosed
  proven-limit residual, at its registered site (T1, T4-5). I found no
  unregistered defect site at any zoom.
- MeshVault's registration instrument proves the six cycles were
  texture-only: every surviving historical face GLB from v8 onward samples
  to chamfer 0.0 against the certified geometry; the one geometric outlier
  (goodalpha) is the documented iter2->iter3 mesh swap, quantified here at
  0.7% normalized chamfer with hair-crust-local deviations (T2).
- The correlation table (T3) maps 20 defect classes: what the viewer shows
  now, what it shows on the surviving historical states, the causing code
  with the specific math error, the fixing mechanism, and the guarding
  tests — all verified line-by-line in source, with load-bearing quotes.
- Fresh findings the harnesses never gated (T4): one non-manifold edge in
  the face hair crust (recorded as `is_watertight: false` in metadata but
  never localized or gated); exports are Z-up against glTF's +Y-up
  convention (deliberate internal canonicalization leaking through the
  export boundary); no real-world scale (2 m bust, 22 cm interpupillary
  distance, measured); the certified ship bundle's metadata lost its
  generation block (only `texture_artifacts` remains). Each filed with
  severity and code owner; none blocks the certification's own terms.

Method note (independence): I worked viewer-first. Every observation was made
on MeshVault renders/instruments of the exact certified bytes, THEN correlated
against the rulings/ADRs, and the named code was read to verify the mechanism
line-by-line. Where my instrument disagrees with or extends the harness
record, that is filed in T4 rather than smoothed over.

Viewer conventions established during T1 (needed to reproduce my shots):
- The assets are Z-up with the subject facing +X. MeshVault's orbit presets
  are Y-up world conventions, so "front" (az0) looks at the CROWN of the
  face asset. All `*_up_*` and `*_close_*` shots were taken after
  `rotate {axis:x, degrees:-90}`; project azimuth `a` maps to MeshVault
  orbit azimuth `a+90`, elevations 1:1. The `*_walk_*` series (pre-rotation
  turntable) is kept as evidence of the export frame itself.
- `find_best_view` (detail-ranked, run on the UNROTATED exports; MeshVault
  spherical az/el around viewer Y): face az120/el0 -> camera unit
  (0.87, 0, -0.5) in file coords = project az~0 / el~-30 — the FACE side,
  from slightly below (the auto-upright corrected the roll). Ship az30/el20
  -> (0.47, 0.34, 0.81) = project az~+36 / el~+54 — above the port bow, the
  photo-witnessed quadrant. Owl az90/el0 -> file +X = the project front
  exactly.

---

## T1 — Deep inspection of the three certified assets

### Numbers from MeshVault's own instruments

| asset | triangles | dims (m) | volume | open edges | non-manifold | degenerate | sliver% | dihedral mean/p95 | edge med/p95 |
|---|---|---|---|---|---|---|---|---|---|
| face-2mv scene.glb | 119,999 | 1.678 x 1.707 x 1.992 | 1.767 | 0 | **1** (at [0.273,-0.476,0.404]) | 0 | 0 | 22.44 / 69.62 deg | 0.0152 / 0.0370 |
| starship scene.glb | 120,000 | 1.995 x 1.048 x 0.404 | 0.152 | 0 | 0 | 0 | 0 | 7.69 / 26.55 deg | 0.0075 / 0.0168 |
| owl scene.glb | 120,000 | 1.337 x 1.169 x 1.994 | 1.241 | 0 | 0 | 0 | 0 | 4.77 / 17.22 deg | 0.0111 / 0.0235 |

- `describe_scene` verdicts: ship and owl "No geometry issues detected";
  face "No blocking geometry issues (1 informational: non_manifold_edges)".
  All three are CLOSED (watertight — volume computed, zero open edges),
  which I did not expect for generated meshes; the bust cut and ship hull
  are capped.
- geometry.glb twins: identical triangle counts and stats; the face
  geometry.glb carries the same single non-manifold edge — so the flaw is
  in the shape backend's marching-cubes output, not introduced by texturing
  or export (scene vs geometry BIN chunks differ only by material/texture).
- Material truth (describe_scene materials block): `MeshStandardMaterial,
  color #ffffff, metalness 0, roughness 1`, one 2048x2048 sRGB baseColor
  texture, doubleSided false — the exact ADR-0009 identity contract,
  confirmed by an independent viewer's parser.
- Scale sanity: all three land at 1.99-2.00 m max dimension ("glTF units
  are meters" hint). See T4-2.

### Screenshot index (all 1024 px; every file carries an `_img0.png` suffix from the driver, omitted below)

- Walkarounds (12 + 4 extra angles each, export frame):
  `face_walk_*`, `ship_walk_*`, `owl_walk_*`, `*_walkx_*`.
- Upright project-frame batteries: `face_up_az{0,±22.5,±45,±90,135,180}_el0`,
  `face_up_az0_el{-20,+15}`, `face_up_azp20_elp8` (declared source pose);
  `ship_up_az{0,30/15,-30,±90,180,0/-25,150/15}`; `owl_up_az{0,±45,±90,180,
  0/-25,0/30}`.
- Historical defect-site close-ups (focus tool, radius 0.15-0.55):
  face: `face_close_{temple_L,temple_R,hairline_front,eyes,
  eye_R_transplant,mouth_chin,nose,neck_glyphzone,neck_m22,ear_L,ear_R,
  rear_hair,crown,nonmanifold_pt}`; ship: `ship_close_{nose_front,nose_low,
  underside,fill_side}`; owl: `owl_close_{base_feet,face,rear}`.
- Cross-sections (set_clip): `face_section_temple_top{,60}` (horizontal cut
  at temple height through the film-shell band), `face_section_sagittal_L`
  (+ `_clay`), `ship_section_length{,_clay}`, `ship_section_cross{,_clay}`.
- Geometry-only twins: `{face,ship,owl}_geom_clay_*`, `*_geom_normals_*`.

### What the certified assets look like NOW (my observations)

FACE (certified texture md5 `928705f3…`):
- az0/±22.5/srcpose: single coherent face; NO beige film sheet at the
  temples/hairline (thin skin-tone wisp ribbons over hair-toned fill
  remain, reading as loose strands); NO black parting hole; NO pale nose
  column; ONE mouth; two structured eyes at ±90 (`face_up_azm90_elp0`,
  `face_up_azp90_elp0`); no third-eye blob at +22.5.
- Neck/chest (FACE-22 sites): `face_close_neck_glyphzone`, `face_close_
  neck_m22` — no "ΔΔ|" glyphs, no closed contour; a soft wide tone valley
  and a gray-taupe under-jaw shading band remain, reading as shading (the
  disclosed witnessed-content residual).
- Rear: `face_close_rear_hair` — vertical combed strand texture; a faint
  blotch cluster at the occiput is still traceable at this ~6x framing
  (disclosed FACE-09 residual class; no leopard read at 1x-2x).
- At MeshVault zooms beyond the certification's 4x standard, the
  proven-limit residuals become visible again (filed with severities in
  T4): eye-corner micro-chips (`face_close_eyes`), a pink below-lip
  micro-chip pixel patch (`face_close_mouth_chin`), the ear-helix
  black/white "zipper" residue (`face_close_ear_L`), film-shell flakes
  catching light at the temples.
- Cross-section `face_section_temple_top`: the horizontal cut exposes the
  hair mass as a HOLLOW crust of paper-thin shells (the film-shell band
  itself, 0.01-0.09 diag standoff documented in ADR 0008 cause 3) — the
  physical structure the layered-zone gate, film-band commit, and
  film-band-gradient repaint all exist to manage.

SHIP (texture md5 `b8e2b0d4…`, byte-identical since 2026-07-06 07:36):
- `ship_close_nose_front`: the head-on intake reads structured — dark
  intake cavity, internal grill masses, bright rim frame, panel edges
  above. No molten streaks (SHIP-03 fixed state confirmed in an
  independent viewer).
- `ship_up_azp30_elp15` (photo pose): crisp panel lines, greebles;
  `ship_close_fill_side` / far side: plausible granular hull material, no
  panel content (SHIP-01 proven-limit as documented), no dark pits at the
  wing root, no facet cells.
- `ship_close_underside` + `ship_up_azp0_elm25`: oriented flow-grain
  material (the LIC streaking of `synthesize_fill_detail` is directly
  recognizable), no mechanical detail (SHIP-04 limit), no dark smears.
- Cross-sections `ship_section_{length,cross}_clay`: the hull is a thin
  closed shell with internal cavities; wing plates are single thin sheets.
  Nothing pathological (no interpenetrating duplicate sheets like the face
  hair crust) — consistent with zero non-manifold edges.
- The `ship_walk_*` series documents the export frame: prow +X, up +Z, so
  the MeshVault "front" preset (`ship_walk_0_0`) is a top-down plan view
  and `ship_walk_90_0` is the prow head-on. Not a defect; a convention
  note for anyone driving spec-convention viewers (see T4-2).

OWL (texture md5 `ff746509…`):
- Clean at every battery angle and close-up; `owl_close_base_feet`: claws
  crisp, base underside a soft speckled tan wash (fill territory) with a
  visible carving->fill texture-character handoff at the feet line —
  the OWL-03/SHIP-07 content-limit family, at cosmetic strength.
- `owl_close_rear`: dense directional carved-grain fill; no leopard
  blotches, no facet fields (the OWL-03 accepted bar: material yes,
  carving content no).
- Best-view instrument agrees the owl's most detailed side is the project
  front (score 0.7473, the highest of the three assets).

---

## T2 — Historical comparison (the defect archaeology)

Surviving states compared (all loaded and screenshotted at the same sites
as the certified state — `hist_<tag>_up_{az0,azp22,azm22}` and
`hist_<tag>_close_{temple_L,hairline,eyes,mouth,neck}`):

| tag | file | era |
|---|---|---|
| goodalpha | /tmp/face_goodalpha.glb | pre-cycle "good alpha" bake, iter2 mesh (88,948 atlas verts) |
| v8 | /tmp/face_v8.glb | the "TERRIBLE" v8 (first iter3-mesh bake, 2026-07-05 04:39) |
| posefix2 | /tmp/posefix2.glb | pose-fix era rebake (06:31) |
| v14 | /tmp/candidate_v14/scene.glb | cycle-1 candidate lane tip |
| c3 | /tmp/c3_3/final2/face_2048/scene.glb | cycle-3 official state |
| c4pub | /tmp/c5/bundle_pub/scene.glb | cycle-4 published baseline (texture md5 c39d65bf) |
| certified | artifacts/validation/iter3-multiview-fixed/face-2mv | cycle-6 final (928705f3) |

### compare_models results (geometric registration, MeshVault)

Full JSON: `/tmp/mva/t2_compare.json`. Reference = certified face scene.glb,
4096 samples/model, `align:false` (in-place comparison — pose changes would
surface):

| candidate | classification | chamferMean | chamferP95 | Hausdorff | structural |
|---|---|---|---|---|---|
| face-2mv geometry.glb | **identical** | 0.0 | 0.0 | 0.0 | tris equal; verts 99,532 vs 59,396 (scene = atlas-expanded vmapping; geometry = raw mesh) |
| c4pub (/tmp/c5/bundle_pub) | **identical** | 0.0 | 0.0 | 0.0 | verts equal 99,532 |
| c3 (/tmp/c3_3/final2/face_2048) | **identical** | 0.0 | 0.0 | 0.0 | verts equal |
| v14 (/tmp/candidate_v14) | **identical** | 0.0 | 0.0 | 0.0 | verts equal |
| v8 (/tmp/face_v8.glb) | **identical** | 0.0 | 0.0 | 0.0 | verts equal |
| posefix2 (/tmp/posefix2.glb) | **identical** | 0.0 | 0.0 | 0.0 | verts equal |
| goodalpha (/tmp/face_goodalpha.glb) | **near_identical** | 0.0483 (norm excess 0.0071) | 0.1247 | 0.3375 | verts 88,948 vs 99,532 — the iter2 mesh |

Ship and owl: scene.glb vs geometry.glb both classify **identical** at
chamfer 0.0 (ship verts 75,248 vs 60,002; owl 68,270 vs 60,002 — same
atlas-expansion delta).

Readings:
1. **The six adversarial cycles are geometrically verifiable as
   texture-only.** Every surviving face state from v8 (2026-07-05 04:39)
   through the certified bytes samples to chamfer 0.0 against the certified
   geometry — no vertex ever moved. This independently confirms, from the
   geometry side, what the canary md5 discipline claimed from the byte
   side, and it cleanly segments every T3 row into texture-lane causes.
2. **The goodalpha outlier is the mesh swap, quantified.** goodalpha (the
   pre-v8 "good" state) is a DIFFERENT mesh: iter2 (88,948 atlas verts) vs
   iter3 (99,532), chamfer excess 0.7% of the diagonal, local deviations up
   to 0.34 units (Hausdorff) in the hair-shell region. The cycle-1 ablation
   study (`/tmp/fixer2/REPORT.md`) concluded the notorious "v7 -> v8
   TERRIBLE jump" was dominated by this regenerated-mesh swap, not by the
   audit code landed that night — MeshVault's registration instrument
   reproduces that finding independently: same triangle count (119,999),
   different crust topology, sub-percent global shape change concentrated
   where every subsequent texture defect lived (the hair shells).
3. `align:false` returned no pose deltas — all historical bundles share the
   canonical frame; no historical state was a pose-transformed export.
4. Instrument-limit note (honest negative): re-running goodalpha with
   `align:true` (2048 samples) did NOT improve on the in-place comparison —
   registration returned `converged: false`, a spurious 5.65 deg rotation,
   and a WORSE chamfer (0.0687 vs 0.0483), downgrading the label to
   "same_shape_modified". This is exactly the documented compare_models
   limit (near-identical shapes give ambiguous transforms; trust
   classification + asymmetry, not the transform). The in-place row is the
   truthful one: same canonical frame, sub-percent shape delta concentrated
   in the hair crust.

### The money sequences (same site, same framing, across history)

All states were shot at the SAME five sites with the SAME focus/orbit calls
(`hist_<tag>_close_{temple_L,hairline,eyes,mouth,neck}` + 4 upright views),
so the sequences are directly comparable. My readings per state:

**Temple band (FACE-01 site), `*_close_temple_L`:**
- goodalpha/v8: hard black crust strips over the temple with dense pale
  skin-fleck salt (mixture stamps + propagated copies) and beige film
  fringes reaching over the brow; v8 worst.
- v14: a large opaque skin-tone FILM SHEET hangs over the brow/temple edge
  (the classic "shower-cap edge"); flake fingers at the boundary.
- c3 (`hist_c3_close_temple_L`): the beige SHEET is gone, but a bright
  white-pale film-flake cluster still hangs at the temple silhouette edge
  (the kept bright-wisp remnants of the c2a honest-limits note), with
  red-brown rim segments behind it — the "edges now harder, with dark rim
  segments" state of the cycle-2/3 record.
- c4pub: film gone; soft warm shadow at the brow (the FACE-20 refill band).
- certified: same site reads as loose wisp ribbons over hair-toned fill;
  brow stroke residue only as a soft dashed shadow at ~8x
  (`face_close_temple_L`).

**Front az0 (`hist_*_up_az0` vs `face_up_azp0_elp0`):**
- v8 (`hist_v8_up_az0`): the doubled-feature catastrophe verbatim — a
  displaced photo eye mid-cheek, second eye fragments at the temple, brow
  smeared across the forehead, mouth slab displaced below-right, gray milky
  patches (the self-rejected/fill zones), skin flecks through the hair.
- goodalpha (`hist_goodalpha_up_az0`): same classes milder + a BLACK DRIP
  streak under the viewer-left eye (the FACE-02 black-debris family) and
  black strap fragments on the chest (FACE-11 family).
- posefix2 (`hist_posefix2_up_az0`): the doubling is GONE — one aligned
  feature set (the pose-era fixes landed); what remains is the ledgered
  cycle-1 world: temple film sheets with flake fringes, under-eye flake
  fields, mouth-corner dark dash, black strap fragments on the chest.
- v14 (`hist_v14_up_az0`, `hist_v14_close_temple_L`): coherent single face;
  a large opaque skin-tone film SHEET hangs over the brow/temple edge with
  flake fingers (the v14 "shower-cap" state the cycle-1 ruling described);
  under-eye flakes persist.
- c3 (`hist_c3_up_az0`): film band committed to hair tone — the beige sheet
  is gone at az0; pale film residue at both temple boundaries; under-eye
  chips and the eye-seam tone split still read; mouth-corner dark dash
  present (the FACE-20/chips era).
- c3 -> c4pub -> certified: monotone cleanup at every site; certified is
  the only state with no beige film, no strokes, no chips at 1x-4x reads.

**Neck (FACE-22 site), `*_close_neck`:**
- c4pub (`hist_c4pub_close_neck`): the thin line-art is DIRECTLY VISIBLE in
  my shot — small glyph-like dark strokes on smooth neck skin (the "ΔΔ|"
  class) plus white speck dots at the jaw/hair curtain boundary.
- certified (`face_close_neck_m22`): no line-art at the same framing; a
  soft tone valley and the honest under-jaw shadow band remain.

**Rear (FACE-09), `hist_v8_up_az180` vs `face_up_azp180_elp0` /
`face_close_rear_hair`:**
- v8: near-black rear mass, hard vertical crust strips, no material read.
- certified: brown combed-strand material with directional striation; faint
  occiput blotch cluster at ~6x (the disclosed residual).

Ship: no pre-fix ship GLB survives in /tmp (the c3 bundle's texture is
already md5-identical to the certified bytes: `b8e2b0d4…`), so the SHIP-03
before/after rests on the repo's own archived sheets
(`cycle2-3/ship_nose_before_after.png`, `/tmp/c3_3/FINAL_ship_before_
after.png`) plus my NOW close-ups (`ship_close_nose_front/low`) — filed as
a record gap, not re-verified independently (T4 note).

---

## T3 — THE CORRELATION TABLE

Column key — NOW: what MeshVault shows on the certified bytes (shot refs
from T1). THEN: what it shows on the surviving historical GLBs (shot refs
from T2, `hist_<tag>_*`). CAUSE: the specific code/math defect that created
the class (verified in-source; quotes below the table). FIX: the shipped
mechanism (file:function). TEST: the guarding test(s) in `tests/`.

| # | defect class (ledger) | MeshVault NOW | MeshVault THEN | CAUSING code (the specific error) | FIXING code | guarding test |
|---|---|---|---|---|---|---|
| 1 | Beige film band over temples/hairline + black parting debris + third-eye curl (FACE-01/02/16, blocking) | Band gone; thin wisp ribbons over hair-toned fill at temples; no parting hole; no 3rd-eye blob. `face_up_azp0_elp0`, `face_close_temple_L/R`, `face_close_hairline_front` | v8: pale mixture-fleck salt through black crust + hair painted across the forehead + black jagged debris strokes (`hist_v8_close_temple_L`, `hist_v8_close_hairline`); v14: opaque skin-tone film SHEET with flake fingers over the temple (`hist_v14_close_temple_L`); the exact T0 ledger asset (md5 44587ff) does not survive — the black parting HOLE of FACE-02 rests on the ledger's own crops | The mesh FUSES the wispy hairline into the head, so the layered-zone gate cannot see a second sheet: measured layered density 0.017-0.054 vs the gate's `layered_zone_density: float = 0.10` (`backends/triposr_runtime.py:1663`) while photo contrast is high — bright skin+hair mixture pixels were stamped as direct paint (55-74% of the sheet), then propagated by fill/mirror | `film_band.py`: per-view film maps in the projector; `commit_film_band` (flag consensus among first-surface views + no base-witness veto + >=2 witnesses + dark dominance) vacates BRIGHT mixture claims; `retone_film_band` tones committed fill from dark observed anchors; mirror banned inside the commit; `demote_unwitnessed_rim` | `test_film_band.py::test_commit_film_band_requires_flag_consensus_and_no_veto`, `::test_retone_film_band_pulls_committed_fill_toward_dark_anchors`, `::test_commit_film_band_noop_for_single_view`; boundary suite `test_film_band_boundaries.py` |
| 2 | Billboard black strokes: az0 temple crack, az-22.5 silhouette streak, az-90 hairline line, ear-helix arcs (FACE-20) | Stroke sites read as soft warm shadow bands; no dark-on-skin strokes at any battery angle. `face_close_temple_L` (brow), `face_up_azm90_elp0` | c3-era only (between c3 and c4 states; the c3 GLB predates the class's worst form). The certified vs `hist_c3_*` temple/hairline comparison shows the C4 refill band vs C3's harder edges | The film-band repaint's guard consulted the base-material witness veto ONLY inside the feature moat: `dark_allowed = stamp_is_dark & ~(veto_any & moat)` — photo dark-body BOUNDARY pixels (curtain edge, ear shadow) billboarded onto grazed surface with veto consensus 0.7-1.0 but moat fraction 0.0 (c4 provenance: components at 78-99% authority stamps) | `film_band_gradient.py:197 _displaced_stamp_components` — component-level veto: reject when veto fraction >= `DISPLACED_VETO_FRAC = 0.5` AND median S >= `DISPLACED_S_MIN = 0.35` (the skin half of the photos' own pooled hair->skin falloff); displaced-site refill floored `DISPLACED_REFILL_FLOOR = 1.02` x dark split (a stroke is unprintable by construction) + `DISPLACED_REFILL_GAIN = 0.30` photo luminance | `test_film_band_gradient.py::test_displacement_veto_rejects_vetoed_dark_stamps_in_skin_half` |
| 3 | Chips & dashes: under-eye flakes, tear-duct white chips, lash dashes, lip-edge dark-red dash, mouth-corner smear (FACE-03/04) | Eyes/mouth read clean at 1024 full-frame; at ~8-10x zoom micro-residues remain (T4-5). `face_close_eyes`, `face_close_mouth_chin`, `face_close_eye_R_transplant` | goodalpha/v8: flake clusters under both eyes, mouth-corner smear; `hist_v8_close_mouth` additionally shows the full doubled-mouth state (row 7's class at the same site). `hist_goodalpha_close_eyes`, `hist_v8_close_eyes` | Displaced view content at TRACE witness weight: chip blobs w50 0.02-0.29 vs legit features w50 0.44-0.93 (c4 provenance); the tear-duct chip was the photo's OWN displaced caruncle content (photo lum 0.85 == blob), and the rescue disc COPIED the healthy side's chip into the twin (placement_shift +0.0246); per-feature few-px mismatch means no global transform fixes eyes and mouth simultaneously | `texturing.py:5866 commit_trace_deposits` (blob-level: candidate w50 <= 0.30 deviating >= 0.045 from voxel-ball context, ring consensus >= 0.96 bright, confident content NEVER demoted, rim feather row 10) + `feature_fringe_repair.py` (rebuilds the identity gate's own correspondence in-bake — BT.601/area-average NCC, z-buffer visibility — and repairs with rescue-transplant semantics; disc interiors never photo-stamped, disc refreshed LAST so healthy-side repairs propagate into the twin) | `test_texturing.py::test_commit_trace_deposits_retones_consensus_contradicted_chip`, `::test_commit_trace_deposits_never_touches_confident_content`, `::test_commit_trace_deposits_bright_near_feature_protected`, `::test_fringe_registration_recovers_similarity`, `::test_repair_feature_fringes_noop_contracts` |
| 4 | Pale seam column nose->philtrum->chin (FACE-05) | Gone: nose flank continuous skin; nose tip clean. `face_close_nose`, `face_up_azp0_elp0` | v8/goodalpha: pale desaturated column beside the nose ridge; v8 also nose-tip beige blob. `hist_v8_close_eyes` (bridge), `hist_goodalpha_up_az0` | NOT a compositor bug: the source photo's own baked nose-ridge SPECULAR, projected under the estimated az+20 pose onto the left nose flank (column texels sample photo lum 218.6 vs 187.7 lateral control, saturation 40.3 vs 57.4 — bright+desaturated+smooth = specular signature). Present in the pre-solve blend; the screened-Poisson solve correctly preserves the most confident witness. (Cycle-4 exoneration: 0 membrane rails in the column; the FACE-05 "column" of the T0 ledger was additionally the source photo's own baked specular — S1 exonerated) | `gradient_compositing.py:391 reconcile_specular_lobes` — cross-view diffuse-consensus authorization (another view's valid sample reads the same surface darker than the pairwise lighting gauge by >= `gauge_margin 0.08` log); correction from the winner's OWN surround (baseline + own log-detail, saturation restored, `saturation_boost_max 1.6`); edge-density refusal `edge_p85_max 1.8` (sclera analogs are edge-dense); source view only; dark-content standoff 8 texels | `test_gradient_compositing.py::test_reconcile_specular_lobes_flattens_authorized_lobe`, `::test_reconcile_specular_lobes_keeps_shared_bright_content`, `::test_reconcile_specular_lobes_refuses_edge_dense_features`, `::test_reconcile_specular_lobes_noop_single_view` |
| 5 | Front identity: tone split + eye seam + gate FAIL 0.613 (FACE-06/14) and neck/jaw wash | Face reads as the subject at `face_up_azp20_elp8` (declared pose); under-jaw carries a soft gray-taupe shadow band (honest print of the photo's cast shadow). Certified comp identity 0.704/14.9 vs gate 0.70/15.0 | v8: vertical tone boundary left-of-nose through the subject-right eye. `hist_v8_up_az0` | Composite of rows 1,3,4 PLUS two measurement-layer truths: (a) the harness renderer's own shading floor — a PERFECT texture scores SSIM 0.977/MAE 11.45 at the raw gate (`rendering.py` shades textured previews `shade = 0.88 + 0.12*diffuse`); (b) the neck wash provenance had the SIGN REVERSED in cycle 4: the wash was side_right's LIT tone (w50 0.17-0.42, lum 0.775) over the front's genuine cast shadow (valid at lum 0.54, log delta -0.35 vs -0.08 gauge) — the albedo must carry the source's shadow | (a) compensated identity protocol (photo x white-render shade field; budgets re-anchored 22.0->15.0) adopted as authoritative in cycle 3; (b) `gradient_compositing.py:638 reconcile_shadow_aprons` — where a reference wins co-witnessed bright surface and the SOURCE validly reads it darker beyond gauge+margin with smooth deviation (edge-density refusal), the composite adopts the source's shading baseline, one-sided darkening, luminance-only, pre-solve; world-ball fragment merge before size floors | `test_gradient_compositing.py::test_reconcile_shadow_aprons_carries_source_shadow`, `::test_reconcile_shadow_aprons_requires_source_evidence`, `::test_reconcile_shadow_aprons_refuses_edge_dense_content`, `::test_reconcile_shadow_aprons_noop_single_view`; boundary suite `test_shadow_apron_boundaries.py` |
| 6 | Profile eye erased/smudged at az-90; side_right worst-window -0.132 (FACE-15, + FACE-18 pose half) | Structured eye at both profiles. `face_up_azm90_elp0`, `face_up_azp90_elp0` | v8/goodalpha/v14: -90 eye an unstructured sliver/smear. `hist_v8_close_eyes` at profile framings, `hist_v14_up_az0` | Two independent causes: (i) the -90 profile photo's eye content was smudged mixture (weak witness), and the broken eye — a strong NCC feature — dragged the harness's profile registration ~1.3% off, manufacturing the ear worst-window (-0.132 -> +0.473 at fixed alignment, ZERO ear texels changed: a pure registration artifact); (ii) cycle-1 estimator drift (az+20 -> +12.5 at NCC 0.0052) erased profile eyes on fresh bakes (FACE-18) | `texturing.py:5447 detect_mirror_rescue_discs` + `:5716 mirror_rescue_disc` — 8-gate transplant (symmetry >= 0.55; strong side W >= 0.35, F >= 0.05; twin observed Ct >= 0.25; twin weakness Wt <= 0.5W; coherent dark core DoG <= -0.12; feature-empty twin POINTWISE; plane-crossing refusal; dedupe/max 4). Axis-anchored placement (cap 0.4 r_feat — the pure geometric position made the repaired eye a second NCC attractor, bistable identity flip), content-aware tone ring, whole-disc only (every partial-keep variant minted eye_count=3). Pose half: row 17's acceptance gate | `test_texturing.py::test_detect_mirror_rescue_discs_fires_on_weak_feature_empty_twin`, `::test_detect_mirror_rescue_discs_no_fire_on_asymmetric_content`, `::test_detect_mirror_rescue_discs_no_fire_when_twin_unobserved`, `::test_detect_mirror_rescue_discs_no_fire_when_twin_has_own_feature`, `::test_mirror_rescue_disc_transplants_twin_feature_tone_matched` |
| 7 | Doubled features / "three-quarter face" distortion + milky patches (the original multi-view failure; FACE-17 ghost-lip class) | Single feature set from every angle (T1 battery) | goodalpha and earlier (the 2026-07-04 face-multiview-proof is documented tainted evidence); v8's split tone. In `hist_goodalpha_up_azp22/azm22` the cheek ghosting is directly visible | FIVE stacked causes, each verified: (1) preview renderer re-lit textured meshes (strong diffuse) — drew the mesh's own eye sockets OVER the photo albedo = ghosted second face; (2) perspective pinhole (fovy 40) for effectively-orthographic photos; (3) GL depth-map tolerance stamped BOTH sheets of thin hair crusts; (4) canonical-frame path assumed the photo is at az0 — OBJECT canonicalization confused with CAMERA pose (photo actually az+15..20); (5) silhouette registration aligns the HAIR OUTLINE, leaving interior features displaced (58 px nose error at 1024); plus the math-audit cluster: outlier filter self-vote (`reach = adjacency + adjacency @ adjacency` — diagonal = vertex degree, islands vote for themselves), splat dilation +1.3 px biasing every registration to +4% scale, scalar z-buffer epsilon self-rejecting 39.6% of 55-75 deg tilted texels (milky streaks), `shift_x * width` back-conversion on height-normalized fits | (1) `rendering.py` flat-biased preview `shade = 0.88 + 0.12 * diffuse` (line ~246); (2) `projection_model="orthographic"` + `canonical_border_ratio` (ADR 0007); (3) strict per-texel z-buffer from projected surface texels + 3x3 conservative widening + slope-aware epsilon `epsilon = 0.0025*diagonal_zb + 2.5*pixel_world*slope` (`triposr_runtime.py:1829`); (4) `texturing.py:511 estimate_pose_photometric` (signed gradient VECTOR field + interior-distance weighting); (5) `register_reference_by_source_overlap` (texturing.py:828) — register interior content to the source's painted truth; math cluster: `reach.setdiag(0.0); reach.eliminate_zeros(); reach.data[:] = 1.0` (texturing.py:4079), splat erosion `binary_erosion(mask, iterations=1)` (texturing.py:84), `shift_x * height` (texturing.py:1375,1502) | `test_texturing.py::test_outlier_filter_drops_island_without_self_votes`, `::test_filter_projection_outliers_drops_foreign_islands`, `::test_projector_strict_zbuffer_rejects_hidden_sheet`, `::test_pose_estimator_recovers_injected_yaw_and_rejects_frontal`, `::test_register_reference_by_source_overlap_recovers_injected_shift`, `::test_recenter_to_canonical_frame_matches_hunyuan_convention`, `::test_projector_layered_zone_gate_surrenders_film_band` |
| 8 | Rear hair leopard mottle (FACE-09) + the 2048 "rectangle" sub-defect | Combed vertical strand striation; faint occiput blotch cluster at ~6x (disclosed residual); no rectangle step. `face_close_rear_hair`, `face_up_azp180_elp0` | v8: near-black rear mass, hard crust strips, no material read (`hist_v8_up_az180`); goodalpha adds a large irregular dark-blotch region mid-back + a white speckle patch on the right shoulder (`hist_goodalpha_up_az180`) | (a) No view witnesses the central rear (profiles see it edge-on) — capture-set limit; the MOTTLE was mixture anchors + fill copying them (>90% of flake islands were propagated copies, not direct projections — ADR 0008); (b) the 2048 rectangle: `repaint_film_band`'s REAR EXTENT darkened 188k texels (median -11/255, 81% fill) with a straight region boundary between surface-smooth and post-commit stages — another mechanism's boundary printing, absent at 1024 where the repaint no-ops below its sampling floor | (a) layered-zone gate + content-contrast amendment (`layered_zone_min_contrast 0.055`, `triposr_runtime.py:1664`) refuses mixture stamps; `strand_comb` + `synthesize_fill_detail` render hair MATERIAL; capture remedy on the register (rear photo). (b) dissolved by the C4 film-band-gradient rework (S-field clamp share fades 0.59->0.00 over ~80 px; c5 measured ~3/255 residual step); the S-field cliff at `CLAMP_S_MAX = 0.90` remains the structural owner if it re-sharpens | `test_texturing.py::test_strand_comb_reduces_fill_blotch_and_noops_when_off`, `::test_strand_comb_bit_identical_when_regime_empty`, `::test_multigrid_orientation_field_propagates_coherently`; `test_film_band_gradient.py::test_repaint_noops_without_second_view_or_mass` |
| 9 | Neck/chest line-art: "ΔΔ\|" glyph cluster + closed chest contour + apron stripes (FACE-22, the last OPEN entry) | No glyphs, no closed contour; soft wide tone valley remains (reads as neck shading). `face_close_neck_glyphzone`, `face_close_neck_m22` | c4pub (`hist_c4pub_close_neck`): the pre-c6 state at the same framing; the c5-era strokes were measured 4281 texels in 66 components | THREE owners (instrumented stage-capture + ablation, both directions): (1) 63% — `repaint_film_band` operating OUTSIDE its field's support: S = d_base/(d_base+d_mass) is a scale-free RATIO taking mid-transition values arbitrarily far from the mass (strokes at d_mass 9-24 transition-lengths vs honest apron p50 2.0T); the GLYPH was small envelope-CLAMP components (hp -0.058, 91% in-clamp); (2) the closed contour — `commit_trace_deposits` blob RIMS: border mixtures sit below `deviation_min` BY CONSTRUCTION (mixture deviation = coverage x deposit deviation) and keep the old darker tone = a closed outline (rim lum 0.639 vs retoned 0.718); (3) mirror completion pastes the lit twin verbatim (+16/255 vs destination ring) — the gradient-domain solve runs BEFORE mirror completion, so legacy seam-leveling's completion-tone reconciliation had NO equivalent (a missing handoff) | (1) `film_band_gradient.py` `FIELD_SUPPORT_TRANSITIONS = 6.0` (treatment confined to <= 6 transition-lengths of the mass, feathered over the last) + `STAMP_BORDER_FEATHER_TEXELS = 6.0` (composite->photo blend at treated borders, EXCEPT against the dark mass); (2) `texturing.py commit_trace_deposits(rim_feather_texels=3)` — one-sided darker-only rim blend toward ring-anchor tone, anchors EXCLUDING the feather band (a rim mixture bright enough to anchor otherwise pins its own darkness); (3) `texturing.py:1690 tone_match_completion_components` — pure-bright mirror components against bright rings take ONE component-level log-median gain (clamped +-0.25); mixed/dark-ring components stay verbatim (rescaling them mints dark_debris in BOTH gain directions) | `test_film_band_gradient.py::test_repaint_field_support_bound_refuses_far_treatment`, `::test_stamp_border_feather_ramps_into_untreated_surface`; `test_texturing.py::test_commit_trace_deposits_rim_feather_closes_border_mixtures`, `::test_tone_match_completion_components_scopes_and_matches` |
| 10 | Bust cut disc: tan wash + radial rim streaks from below (FACE-12) | Disc reads as a plausible underside of its rim materials (skin front arc / hair rear). `face_up_azp0_elm20`, `face_section_sagittal_L_clay` | v8-era el-20 views: tan marble wash + dark radial rim streaks | The synthetic cut face (91,494 texels at 2048, direct witness 0.1%) was toned by the GLOBAL harmonic fill from rear-hair/neck anchors — correct-on-average, wrong-in-kind for a synthetic surface | `texturing.py:6558 tone_bottom_cap` — geometric detection (down-plane component, slab ratio) + rim-anchored toning (48-NN power-1.5, sigma 24 transition), 60% of the cap's own detail kept; multi-view branch only | `test_texturing.py::test_tone_bottom_cap_tones_cut_face_from_rim`, `::test_tone_bottom_cap_noop_without_planar_cap` |
| 11 | Ear complex: pale shards + dark strokes; elf-ear silhouette; crown flaps + mottle (FACE-07/08/13 — PROVEN-LIMIT register) | Ear texture largely clean; a black/white serrated "zipper" residue along the helix at ~10x (`face_close_ear_L`); pointed apex silhouette persists; crown flaps break the silhouette at el+40 (`face_close_crown`) | Same sites on v8/goodalpha carry heavy shard debris (hist close-ups) | (a) shards: mixture anchors in the fully-contested ear band, 90-96% FILL copies; (b) witnessed truth: skin genuinely seen between strands (w90 0.63-0.73) — untouchable under the witness contract; (c) elf apex: GEOMETRY (Hunyuan decimation), texture-half exonerated by the apex witness audit (paint runs 2.9-6.2% MORE conservative than the photos' own skin/hair boundary); (d) crown: 52% confidently witnessed real parting + S4 mesh flaps | `commit_pale_chips` (texturing.py:6324, the dark-context dual with 1.2e-3 area cap) reduces the shard class; ear clamp (conservative, silhouette-safe) bounds the apex; crown ceiling experiment (clamp-all barely moves the read, dims the real parting) closed FACE-13 as PROVEN-TRADE. Register remedies: ears exposed at capture / better geometry source / crown photo | `test_texturing.py::test_commit_pale_chips_retones_dark_consensus_island`, `::test_commit_pale_chips_never_touches_confident_pale`, `::test_commit_pale_chips_refuses_bright_frontier_sliver`, `::test_commit_pale_chips_noop_single_view` |
| 12 | Ship nose melt: dark streaked concavity + frayed wing edges (SHIP-03) | Head-on intake structured and readable: dark cavity, internal grill, bright rim frame. `ship_close_nose_front`, `ship_close_nose_low`, `ship_up_azp0_elp0` | (No pre-fix ship GLB survives in /tmp; the c3 bundle's texture is already md5-identical to certified — `b8e2b0d4` both; the before/after record is `/tmp/c3_3/FINAL_ship_before_after.png` and `cycle2-3/ship_nose_before_after.png`) | A pure FRAME-REGISTRATION error: `recenter_to_canonical_frame` centers the photo's ALPHA-BBOX at the frame center; the orthographic projector centers the WORLD ORIGIN. At the overridden pose az+30/el+15 the mesh's camera-plane bbox center projects (+54.2, -27.9) px off frame center at 1024 — every photo sample landed ~54 px off the surface that imaged it; at the prow (surface turning away, strong silhouette gradients) that dragged dark under-hull/background-adjacent pixels onto the nose. Proven by ceiling experiment: a PERFECT triplanar checker decorrelated to chance (0.43-0.47 binary agreement) under the old registration and survives structured (0.57-0.73) with the fix — the demotion-curve hypothesis was false (weight demotion is inert for single-view fill anchoring: ANY weight > 0 is coverage truth) | `texturing.py:178 projected_frame_center_px` (mesh's camera-plane bbox center under the projector's own camera convention; docstring carries the whole provenance) + `recenter_to_canonical_frame(center_px=...)` (texturing.py:123); SCOPE RULE: overridden poses only — an ESTIMATED pose was searched against the legacy-centered photo (co-adapted), forcing the projector frame under it broke the face (front SSIM 0.630->0.598) | `test_texturing.py::test_projected_frame_center_px_matches_projector_convention`, `::test_recenter_to_canonical_frame_center_px_places_bbox_center`, `::test_bake_projection_frame_registration_recovers_offset_content` (end-to-end miniature SHIP-03) |
| 13 | Dark fill fragments at 4x in concavities (SHIP-02, OWL-02) | Zero dark pits/smears at any close-up: `ship_close_fill_side`, `ship_close_underside`, `owl_close_base_feet` | The T0 artifact bundles carried them because their textures PREDATED the current pipeline (metadata: `projection_mode: "projection_only_plus_inpaint"`, no fill_floor/detail stats — the fixing stages never ran on those bytes); trace: ship concavity_04 = 131/139 fragment texels were FILL at luminance ~41-49 | Transported darkness: fill texels anchored by near-zero-confidence dark observed texels (weight p50 0.005 at the ship nose) — in a single-view bake ANY weight > 0 anchors the fill with full authority | `texturing.py:3162 enforce_fill_luminance_floor` — lifts feature-dark pockets in SYNTHESIZED texels to a context floor, with a dark-EVIDENCE exemption (`evidence_headroom = 1.35`: fill tone-tracking genuine dark anchors is kept — the same exemption that correctly preserved the pre-registration-fix nose streaks, row 12) | `test_texturing.py::test_enforce_fill_luminance_floor_lifts_pockets_keeps_lines_and_dark_regions`, `::test_enforce_fill_luminance_floor_donor_anchor_catches_transported_darkness`, `::test_enforce_fill_luminance_floor_spares_mirror_features_and_opposite_sheets` |
| 14 | Fill character: facet/Voronoi cells, flat plateaus, energy collapse (SHIP-01 visual half, SHIP-08, OWL-03, verify3 facet_cellular 0.445) | Far-side hull = plausible granular material with oriented flow-grain (LIC streaking visible, `ship_close_underside`); owl rear = directional carved grain (`owl_close_rear`); no flat cells | (T0 owl/ship bundles; no /tmp survivor) — the class is documented in `tex2_fill_quality.md` | (a) vertex-domain harmonic fill assigned each texel its NEAREST vertex color — every ~70-texel Voronoi cell rendered as one flat facet; (b) detail synthesis undershot the gate multiplicatively (donor amplitude 0.84x, carrier frequency 0.69x, base luminance 0.79x ~= 0.41 of observed energy) and a fixed gain could never satisfy both resolutions; (c) grazing-smeared donors carry artificially quiet statistics -> fill plateaus with straight chart-edge borders | (a) 3-nearest-vertex IDW + `texel_surface_smooth` Jacobi pass; (b) `texturing.py:2611 synthesize_fill_detail` closed-loop energy calibration — provisionally applies detail, measures realized fill Scharr energy, solves one global scale (secant) to land at gain x observed energy, bounded [1, 3] with a sigma guard (log-sigma may not exceed the observed band-matched residual — "gradient parity may not be bought with granite"); (c) amplitude floor at the observed population's p25 RAW-residual amplitude | `test_texturing.py::test_synthesize_fill_detail_energy_calibration_reaches_gate`, `::test_synthesize_fill_detail_calibration_never_injects_granite`, `::test_synthesize_fill_detail_amplitude_floor_breaks_quiet_donor_plateaus`, `::test_texel_surface_smooth_removes_plateau_steps_and_keeps_anchors`; harness `test_texture_qa.py::test_facet_detector_fires_on_voronoi_fill` |
| 15 | Global tone darkening: ship 25% dark, owl "44% dark" (SHIP-06, OWL-01) | Both assets read at photo-consistent tone under MeshVault's PBR+IBL (walkarounds) | (harness history; T0 numbers 0.752 / 0.567) | TWO stacked causes: (a) the REAL one — export material defaults (row 16); (b) a HARNESS artifact — the brightness gate's photo reference classified the unmatted light-gray BACKDROP as foreground ("any channel > 18 from white" matched 100.0% of the owl frame, reference lum 203 vs subject 129) | (b) `scripts/texture_qa.py photo_foreground()` — RGBA keeps alpha; RGB photos matted with the SAME `remove_background_robust` the bake uses; degenerate mattes fall back EXPLICITLY (`heuristic_nonwhite` recorded). Owl 0.567 -> 0.891 on the same texture; residual decomposed: -7.3% QA viewer's own diffuse term + -3.9% true albedo deficit (single-view delighting unidentifiable — documented limit) | `test_texture_qa.py` (photo-reference behavior), `::test_material_gates_fail_dark_metal_and_pass_fixed` |
| 16 | Export material truth: assets rendered dark + metallic in ANY spec viewer (the defect MeshVault itself surfaced, ADR 0009) | `describe_scene` materials: `color #ffffff, metalness 0, roughness 1` — identity contract confirmed by a second independent parser; assets render at authored albedo under IBL (all T1 shots) | The pre-fix GLBs carried `baseColorFactor [0.4,0.4,0.4,1.0]` + ABSENT metallicFactor (glTF default 1.0 = fully metallic): up to ~10x too dark under IBL (metal has no diffuse; the 0.4-scaled albedo became specular F0); MTL sidecars `Ka/Kd/Ks 0.4` | `_tripo_build_textured_mesh` used trimesh `SimpleMaterial(image=...)` whose defaults are 0.4 gray, and `SimpleMaterial.to_pbr()` OMITS metallicFactor (trimesh 4.12.2 visual/material.py:200); the repo's own preview renderer sampled the RAW texture ignoring all factors — the only renderer anyone checked was the one that could not see the defect | `triposr_runtime.py:2152` explicit `PBRMaterial(baseColorFactor=(255,255,255,255), metallicFactor=0.0, roughnessFactor=1.0)`; `:2166 _tripo_obj_material_from_pbr` (Ka/Kd = base color, Ks = base x metallic = 0 — literal "Ks 1.0" measured +221% washout and was refused); `rendering.py _material_base_color_factor` — the preview now MULTIPLIES by the factor so it can never look better than a real viewer; standing checker `scripts/check_export_materials.py --strict` | `test_export_materials.py` (all 6: `::test_textured_glb_carries_identity_material_factors`, `::test_textured_obj_mtl_carries_identity_phong_factors`, `::test_material_base_color_factor_matches_spec_viewer_semantics`, `::test_preview_sampling_applies_base_color_factor`, `::test_rendered_preview_darkens_with_defective_base_color_factor`, `::test_untextured_exports_stay_unchanged`) |
| 17 | Pose lottery / drift: owl estimator turned a frontal statue to az+32.5 (OWL-04); face drift +20->+12.5 erased profile eyes (FACE-18); first estimator rotated a KNOWN-FRONTAL photo 25 deg (math-audit Finding 1) | Owl front is the owl's real front (best_view az90-MeshVault = project az0); face srcpose battery green | verify3 owl: wood grain smeared off the right flank + white horn blob (record); `hist_v8`/`posefix2` show the drift-era face states | The gradient scorer compared photo and render in MISMATCHED frames (compact heads tolerated it; the elongated starship's projected aspect swings with elevation); elevation grid lacked +-15; on bilaterally symmetric meshes the mirror pose is a near-tie (0.1% vertex jitter flipped the argmax); and the original acceptance margin (0.002) was 5x smaller than the measured false-positive margin (0.0105) | `texturing.py:511 estimate_pose_photometric`: crop-immune frame anchors (subject top, silhouette centroid, mean width over common rows) before correlation; elevation grid +-15 with local refinement; CHIRALITY TIE-BREAK (`:753` — horizontal anti-symmetric luminance correlation, sign-opposite between mirror poses); ACCEPTANCE GATE (`:801-811`): `best_score > min_peak_score (0.008)` — bad commits measured 0.0043-0.0052, genuine matches 0.012-0.038 — AND margin over declared `max(0.002, 0.25 * declared_score)`; anti-correlated argmax rejected outright | `test_texturing.py::test_pose_estimator_recovers_injected_yaw_and_rejects_frontal`, `::test_estimate_view_pose_centers_search_on_declared_angle` |
| 18 | side_right identity MAE 24->32-40 under delighting (FACE-19) | side battery green (comp sides 0.687/0.706) | (cycle-1 regression state; no GLB survivor — harness record) | `delight_projections`' order-2 SH correction was applied across each reference's ENTIRE coverage: helping the overlap handoff but relighting the reference's EXCLUSIVE territory away from its own photo (identity MAE 26.4 -> 39.5), where that photo is the only witness | `texturing.py:3664 delight_projections` — overlap-proximity fade (full correction near the overlap surface, zero deep inside exclusive territory; `fade_radius_frac 0.06`, voxel-ball density) + revert-on-confound: kept per reference only if it reduces overlap disagreement by > `improvement_margin 0.002` (`:3880 if after < before - improvement_margin`) | `test_texturing.py::test_delight_projections_fade_protects_exclusive_territory`, `::test_delight_projections_keeps_chroma_and_reverts_on_confound`, `::test_delight_projections_recovers_agreeing_albedo_on_two_light_sphere` |
| 19 | Matte amputation: rembg u2net dropped 40% of the subject (dark hair vs light backdrop), corrupting every alpha-driven stage | (upstream of all certified bakes) | The 2026-07-04 face-multiview-proof artifacts are documented TAINTED EVIDENCE (chirality-swapped refs + defective mattes) | rembg's default `u2net` checkpoint is a general salient-object model that amputates low-contrast regions | `segmentation.py` — `_PREFERRED_MODELS = ("isnet-general-use",)` with explicit fallback + matte cleaning (largest components, closed pinholes) before any geometric use of alpha | `test_texturing.py::test_clean_alpha_mask_removes_floaters_and_holes` |
| 20 | Photo->fill seams interrupting panel lines (SHIP-07) + owl feet/base handoff | Visible as a texture-character handoff at cosmetic strength (`owl_close_base_feet` feet line; ship spine crossover in `ship_up_azp150_elp15`) | T0: hard patch boundaries cutting panel lines mid-hull | Single-photo content limit: where photo projection ends, statistics-only fill begins; tone was additionally off before seam allowance was measured against MATTED photos (row 15) | `level_composed_seams` (texturing.py:2171 — Ivanov/Lempitsky-style low-frequency offset field, material-boundary cap, confidence pinning) + calibrated seam allowance in the harness; content half stays PROVEN-LIMIT (remedy: any second viewpoint) | `test_texturing.py::test_level_composed_seams_cancels_tone_step_and_keeps_detail`, `::test_level_composed_seams_skips_material_edges`, `::test_level_composed_seams_pins_confident_witnesses` |

### Load-bearing lines verified in source (quotes)

Row 12 (SHIP-03), `texturing.py:186-198` (docstring of `projected_frame_center_px`):
"The orthographic sample map centers the WORLD ORIGIN at the frame center;
the canonical recenter centers the photo's ALPHA BBOX there. Those two
conventions agree only when the mesh's bbox center projects onto the camera
axis — true at the canonical front …, false in general at other poses
(measured: starship +54/-28 px at az+30/el+15, face +16/+8 px at az+20/el+8,
owl ~1 px at az0)."
CORROBORATION IN THE SHIPPED BYTES: the certified ship bundle's own
metadata records `source_registration: {method: "mesh_bbox_center",
frame_center_dx_px: 54.24, frame_center_dy_px: -27.94, frame_size: 1024}` —
the exact correction, frozen in the artifact; and `fill_floor: {applied:
true, lifted_texels: 113070, evidence_exempt_components: 31}` — row 13's
mechanism, live in the certified bake.

Row 16 (material truth), `triposr_runtime.py:2146-2157`:
"trimesh's SimpleMaterial defaults to a 0.4 gray diffuse and its GLB
conversion omits metallicFactor (glTF then defaults to 1.0, fully metallic):
spec-compliant viewers rendered exports ~60% darker and mirror-dark under
image-based lighting." -> `PBRMaterial(baseColorFactor=(255,255,255,255),
metallicFactor=0.0, roughnessFactor=1.0)`.

Row 7 (ghost face), `rendering.py:236-246`: "strong ridge shading re-draws
the mesh's own geometric features (eye sockets, brows, lips) over the photo
albedo and reads as a ghosted second face … A 12% diffuse cue keeps just
enough depth" -> `shade = 0.88 + 0.12 * diffuse`.

Row 7 (outlier self-vote), `texturing.py:4075-4081`: "a planted foreign
island voted 28-vs-14 for itself and was never dropped … Binarize because
off-diagonal 2-hop PATH COUNTS (up to 2 through triangles) similarly
overweighted island mutual support" -> `reach.setdiag(0.0);
reach.eliminate_zeros(); reach.data[:] = 1.0`.

Row 7 (z-buffer self-rejection), `triposr_runtime.py:1824-1830`:
`slope = sqrt(1 - facing^2)/max(facing, 0.05); epsilon = 0.0025*diagonal_zb
+ 2.5*pixel_world*slope` — the scalar-epsilon self-rejection band (39.6% of
visible texels zeroed at 55-75 deg tilt, demoted to milky fill) is closed by
the tilt-proportional bias; front-on sheets keep the base epsilon exactly
(slope = 0), preserving the two-sheet ghosting protection.

Row 1 (film band blindness), `triposr_runtime.py:1659-1664`: the layered
zone gate thresholds — `layered_zone_density 0.10`, `layered_zone_min_
contrast 0.055` — the fused hairline measured density 0.017-0.054, BELOW the
gate: structurally invisible to the second-sheet detector, which is why the
class needed its own mechanism (`film_band.py`) rather than a tuning.

Row 9 (FACE-22 support bound), `film_band_gradient.py:132-146`: "the S
field is a RATIO of geodesic distances, so it takes mid-transition values
arbitrarily far from the hair mass (measured … neck/chest texels at 9-24
pooled-profile transition lengths carried S~0.66 and were treated as
'hairline apron') … treatment is confined to within FIELD_SUPPORT_
TRANSITIONS of the mass … 6 separates the measured stroke sites (d_mass
p5 = 8.8 T) from the honest apron (film commitment zone p50 = 2.0 T) with
margin on both sides."

Row 2 (stroke veto), `film_band_gradient.py:113-123`: "0.35 is the measured
boundary separating the stroke class (S_med 0.35-0.66) from the valid vetoed
wisp mass (S p50 0.23)" -> `DISPLACED_S_MIN = 0.35`, `DISPLACED_VETO_FRAC =
0.5`, component-level in `_displaced_stamp_components` (:197).

Row 17 (pose gate), `texturing.py:784-811`: "adversarial verification
measured bad commits at scores 0.0043-0.0052 (a frontal statue moved to
az +32.5) while genuine matches score 0.012-0.038" -> `min_peak_score 0.008`
floor + `required_margin = max(min_margin, 0.25 * declared_score)`.

Row 5 (fringe cumulative-veto context, certified maintenance item 1),
`feature_fringe_repair.py:901-956`: the in-loop veto baseline ADVANCES with
each accepted stamp and the photo-truth exemption is bounded by
`battery_worst_micro` — the exact mechanism critic 2's mandatory
cumulative-baseline hardening (measured ~7 stamps = +0.00096 at one view,
triple the single-stamp budget) extends. Confirmed present as the FIRST-ITEM
obligation of the maintenance contract, not yet implemented in the fringe
stage.

---

## T4 — Fresh-eyes findings (MeshVault instruments vs the harness record)

Honest classification rule: a finding is NEW only if no harness gates it AND
the repo record does not already carry it as a proven limit / disclosed
residual. Known-but-resurfaced items are listed for completeness with their
register entry.

### T4-1. PARTIALLY NEW (minor) — the face's one non-manifold edge, localized and reconciled

`get_mesh_stats`/`describe_scene` flag exactly one non-manifold edge on
face-2mv, at [0.273, -0.476, 0.404] (project frame: inside the hair-shell
crust above/behind the subject-right ear; close-up
`face_close_nonmanifold_pt`). Present in geometry.glb AND scene.glb — born
in the shape backend's output, untouched by texturing/export. Ship and owl:
zero.
What the record already knew: the face bundle's metadata topology block
records `is_watertight: false, euler_number: -605` (trimesh's strict
definition fails on exactly this edge class; the owl records watertight
true / euler 2). So non-watertightness was RECORDED but never localized,
never gated, and never reconciled with the visual record. MeshVault's
instruments refine it to: CLOSED surface (0 open edges, volume computable)
with exactly ONE >2-face edge, i.e. "solid except one topological pinch in
the hair crust", plus euler -605 = the hair shells carry hundreds of
handles/tunnels — a quantified topological signature of the crust-film
geometry that drove the film-band defect family.
Severity: cosmetic today; real for downstream mesh processing (booleans,
simplification, re-unwrap, print pre-flight).
Code owner: `backends/hunyuan3d_runtime.py` (mesh extraction/decimation).
Cheap guard: a connectivity line (open/non-manifold/degenerate counts) in
the bundle audit; today no harness reads the topology block it already has.

### T4-2. NEW-with-nuance (minor, interop) — exports are Z-up; glTF's convention is +Y-up, and the canonicalization went TOWARD Z-up on purpose

The certified GLBs are in the project frame: subject facing +X, up +Z (face
bounds: Y extent 1.707 < Z extent 1.992). The glTF 2.0 spec defines +Y as
up; a spec-following viewer's "front" preset therefore looks at the CROWN
(my pre-rotation walkaround `face_walk_0_0` shows exactly that; the ship's
"front" preset is a top-down plan view). MeshVault copes (auto-upright,
find_best_view), but conforming viewers/DCCs that trust orientation lay the
assets down.
The nuance: this is DELIBERATE — the bundle metadata records
`export_axis_canonicalization: ["yup_front_z_to_zup_front_x"]`, i.e. the
pipeline actively converts the backend's Y-up output INTO Z-up for its own
canonical frame (which the projector, harness renderer — up = [0,0,1] in
`_tripo_camera_position`, `triposr_runtime.py:1296-1301` — and all
registration math depend on). The choice is internally consistent and
documented in metadata; what is missing is the INVERSE conversion at the
export boundary, so the internal convention leaks into the interchange
files. The repo's own previews can never see it, the same masking CLASS as
the ADR-0009 material lesson (much lower severity).
Severity: minor (one click in any viewer; systematic in pipelines).
Code owner: export path in `backends/triposr_runtime.py` — a Y-up
conversion (rotation node or baked transform) at GLB assembly would satisfy
the spec without touching the bake frame.

### T4-3. Mostly-known (informational, positive) — all three assets are closed solids; the ship's topology record is missing

`get_mesh_stats` computes real volumes (face 1.767, ship 0.152, owl 1.241;
null for open surfaces in this tool) with ZERO open edges — the bust cut
and hull are capped. The metadata already records topology for face and owl
(T4-1), so "closed" is known-but-unGATED rather than new. What IS new: the
STARSHIP bundle's metadata contains ONLY the `texture_artifacts` block (a
rich bake record: source_registration, fill_floor, pose override — used as
corroboration in T3) — but the GENERATION-time block (topology, backend id,
license, axis canonicalization) that face and owl carry was lost in some
texture republish along the cycles. Severity: informational; the certified
ship bundle has no on-disk record of its geometry provenance beyond
geometry.glb itself. Owner: the publish/staging flow (`docs/methodology.md`
checklist) — worth adding "metadata carries the generation block" to the
publication checklist.

### T4-4. NEW (cosmetic, scale sanity) — assets carry no real-world scale

All three export at ~2.0 m max dimension ("1.99 m — glTF units are meters"
in describe_scene). Cross-check with the measure tool
(`/tmp/mva/t4_measure.json`): interpupillary distance on the face measures
0.220 units = 22.0 cm — 3.5x life size (human IPD ~6.3 cm); bust height
1.992 m; ship length 1.995 m. The +-1 normalized box is the shape backend's training
convention, never rescaled at export, and no harness carries a scale-sanity
line. Any metric consumer (AR "view in your room", e-commerce embeds, game
engines with physics) receives a 2 m owl.
Severity: cosmetic for turntable viewing (viewers auto-frame); real for
metric consumers. Code owner: export/bundle assembly (a `unit_scale` or a
measured-height metadata field, or an optional real-height parameter at
export). Honest note: no ledger entry ever mentioned units — this axis was
simply never on the table.

### T4-5. KNOWN (proven-limit register), resurfaced beyond the 4x standard

MeshVault's focus tool frames defect sites at an effective ~8-12x, beyond
the certification's declared standard (1000 px + 2x/4x crops). At that
magnification the proven-limit register re-materializes exactly where it
says it should — I verified each against the register rather than filing
them as new:

| observation (my shot) | register entry | reads at cert standard? |
|---|---|---|
| brown checkered micro-chips at the viewer-left eye inner corner, pale under-eye streak (`face_close_eyes`) | FACE-03 eye-corner micro-residue (repair measured to cost eye_count/debris at every attempt across 3 cycles) | no — clean at 4x |
| pink below-lip micro-chip with hard texel edges + wine-red lip-line remnant (`face_close_mouth_chin`, top of `face_close_neck_m22`) | FACE-04 residue family closed by the c5 mouth stamp; residual disclosed | no — soft at 4x |
| black/white serrated "zipper" along the ear helix (`face_close_ear_L`) | FACE-07 ear complex (witnessed skin-between-strands w90 0.63-0.73 + parallax; C2 grant) | no — reads as hair detail |
| occiput blotch cluster inside combed rear hair (`face_close_rear_hair`) | FACE-09 residual (~2-3/255 class, ruled below the visible-defect bar cycle 5) | no — smooth at native contrast |
| skin-pale patches painted on hair-shell interiors, visible only through the crust gaps (`face_close_nonmanifold_pt`) | film-shell mixture-anchor family (ADR 0007 residual limitation) | no — hidden by outer shells |
| tiny dark-dotted "checker" flecks inside the pale wisp ribbons at the hairline (`face_close_hairline_front`, viewer-left temple) | FACE-01 honest-limit ("bright wisp remnants … kept claims under the witness veto / dominance gate", c2a report) | no — wisp ribbons read as strands at 4x |
| crown flaps + mottle from top-down angles (`face_close_crown`, `face_walk_0_0`) | FACE-13 PROVEN-LIMIT (52% confidently witnessed real parting; flaps are mesh topology) | partially — the register's own el60/80 finding |

The certification's boundary statement ("nothing visible at 1000 px and
2x/4x that the pipeline could have prevented") SURVIVES this instrument at
its own terms; what MeshVault adds is that the proven-limit register is
geographically accurate — every re-surfaced residual sits exactly at a
registered site, and I found NO unregistered defect site at any zoom.

### T4-6. Instrument corroborations (no action, filed as evidence)

- **Cross-sections**: `face_section_temple_top` exposes the hair mass as a
  hollow crust of paper-thin shells — the physical film-shell band of ADR
  0008 (0.01-0.09 diag standoff): direct visual confirmation of the
  geometry that made hairline pixels ambiguous witnesses. The ship sections
  (`ship_section_{length,cross}_clay`) show a clean thin closed shell with
  no duplicate interpenetrating sheets — consistent with the ship never
  having needed the film-band machinery (its single-view bake also
  structurally bypasses it).
- **Anomaly**: no cross-section anomalies found on ship/owl (no internal
  floating geometry, no inverted normals in the normals render mode
  `*_geom_normals_*`).
- **Dihedral roughness** (relative indicator): face 22.4 deg mean / 69.6
  p95 vs ship 7.7 and owl 4.8 — the face's number is dominated by the hair
  crust, quantifying WHY hair-adjacent texels dominate six cycles of
  texture work.
- **find_best_view** (lighting-independent detail ranking): face = the face
  side from ~30 deg below (project az~0/el~-30), ship = above the port bow
  (project az~+36/el~+54, the photo-witnessed quadrant), owl = the project
  front exactly. An independent instrument ranking the PHOTO-WITNESSED
  surfaces as the detail-densest quadrants is itself a corroboration of the
  fill-vs-witnessed asymmetry every SHIP-01-family entry describes.
- **compare_models** geometric identity: scene.glb == geometry.glb at
  chamfer 0.0 (face; ship/owl in T2 results) — six cycles of texture work
  verifiably never moved a vertex, which is exactly what the canary
  md5-discipline claimed from the bytes side.

### T4-7. Process observation for the record

The maintenance contract's first item (critic 2's cumulative-baseline veto
in the fringe stage) is not yet in `feature_fringe_repair.py` — the in-loop
veto baseline still advances per accepted stamp with the exemption bounded
by `battery_worst_micro` only (`feature_fringe_repair.py:901-956`). That
matches the certification's wording (mandatory AS THE FIRST ITEM of any
future pipeline change, not a blocker on frozen bytes) — filed here so the
next change cannot claim ignorance.
