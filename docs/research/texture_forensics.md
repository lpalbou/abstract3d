# Texture-bake forensics — why e20's texture is misplaced, measured (2026-07-21)

Adversarial audit of the TEXTURE half of the e20 champion
(`out/laurent-bust-redo/e20_fixed_views/`): the operator confirmed the mesh
and rejected the texture — lips/mouth colors on the nose tip, a ghost face
fragment "inside the right glasses lens", beard patches on cheeks, a black
blotch on the jaw. Every claim below is a computed number from an exact
isolated reproduction of the shipped bake; nothing is inferred from renders
alone.

Tools (all under `scripts/experimental/`):
`texture_forensics_bake.py` (instrumented isolation re-run of the
production bake on the shipped `geometry.glb`; capture is harness-side
monkey-patching that never modifies data), `texture_forensics_analyze.py`
(per-texel winner/weight/region attribution from the capture),
`texture_forensics_ruler2.py` (the viewgen-audit head-band IoU azimuth
ruler pointed at the e20 mesh), `texture_forensics_source_rowlaw.py`
(the corrected-bake source preparation). Raw outputs: `/tmp/texfor/`.

## 0. Exact reproduction (the evidence is admissible)

The shipped bake call was reconstructed and re-run in isolation on
`e20_fixed_views/geometry.glb` (bake-frame mesh; GLB round-trip verified
vertex-identical):

- source = `windowed_set_v2/win_laurent_front_clean4.png` (the WINDOWED
  front; `input.png` is byte-identical to the full front's RGB because the
  window edits alpha only — the alpha-cut identifies which one was passed),
- references, in CLI order, all `--texture-reference-synthesized`:
  windowed `windowed_set_v2/win_{back,side_left,side_right}.png` at
  180/90/−90, then full-span `viewgen_fixed/{back,side_left,side_right}.png`
  at the same three angles,
- `bake_projection_texture(..., texture_resolution=2048,
  texture_completion="auto", projection_model="orthographic",
  canonical_border_ratio=0.15)`.

Reproduction quality: all 7 per-view `coverage_ratio` /
`capture_efficiency` rows match the bundle metadata to the 4th decimal
(e.g. side_right twins 0.1543/0.1165), `source_pose` score/veto match
(0.0153 / −0.0055 / `ncc_vetoed_by_silhouette`), texture MAE vs shipped
`texture.png` = **0.001/255** with **100.0%** of surface texels within
2/255. The capture's per-view weights therefore ARE the shipped bake's.

Twin content identity (head-region NCC between same-angle pairs):
side_left pair **1.000** (true alpha-window copy), back pair **0.676**
(different draw), side_right pair **0.204** (different draws: earbud +
different glasses in the windowed one). e20 did not just pass windowed
copies — two of the three pairs are two independent syntheses of the same
declared angle.

## 1. Anatomy ground truth (the number everything else hangs on)

Canonical-frame feature rows (1024 px frame, orthographic az 0), photo
measured by dark-band/nostril scan + landmark zoom, mesh by dense
leading-edge profile of 800k surface samples:

| feature | photo row | e20 mesh row | offset (px) | offset (% head height*) |
|---|---|---|---|---|
| crown | 159 | 146 | −13 | −4% |
| glasses band center | 324 | 363 | **+39** | +12% |
| nostril line | 408 | ~450 | **+42** | +13% |
| mouth center | 444 | 480 | **+36** | +11% |
| chin bottom | 490 | 515 | **+25** | +8% |

\* head height photo crown→chin = 331 px.

The mesh's face sits ~25–42 px LOWER in the canonical frame than the
photo's (and is internally ~10% taller): the 2mv DiT (windowed-set
conditioning) reconstructs its own canonical object; the "canonical
recenter IS the registration" doctrine (texturing.py, orthographic source
path) is exact for the single-view flagship lane but does NOT hold for
this multi-view-conditioned mesh. The bake registers the source by
recenter ONLY (no scale/shift/warp; `reg2d` deliberately skipped for the
source, pose refine disabled, dense flow references-only), so nothing in
the pipeline could absorb these offsets.

## 2. Per-defect attribution (winner shares from the shipped bake's weights)

Regions are 3D boxes on the mesh's OWN anatomy (bake frame,
`regions_geo.json`); "winner" = argmax of final per-view blend weights;
"samples rows" = which photo rows the winning view painted there (for the
front view at az 0, texel canvas row ≡ photo row).

| defect (operator's words) | mesh region | winner view(s) | what actually painted it | mechanism |
|---|---|---|---|---|
| "lips/mouth colors ON THE NOSE TIP" | nose tip (z 0.15–0.21) | **front 100%** (2,359/2,359 texels) | photo rows 424–444 = **mustache + upper lip** band (shipped rgb (32,33,36)) | **H4** source row misregistration (+40 px) |
| "ghost face INSIDE the right lens" | forehead between hair-overhang underside and visor shelf (z 0.43–0.575) | **front 100%** (11,998 texels) | photo rows 267–318 = **forehead skin (58% skin-classified) + glasses TOP band** — a bright skin band framed by two dark bands reads as a face inside a lens | **H4** (+39 px at the glasses) |
| (same defect, lens surface itself) | visor shelf right half (z 0.28–0.41, y<0) | side_right 59% + front 39% | front: photo rows 335–380 = glasses bottom + **nose-bridge skin**; side_right: its own lens rows 306–359 (dark + reflections) | H4 + minor H3 |
| "beard patches on cheeks" | left/right cheek (z 0–0.28) | **side views 67%** (15.8k/23.6k and 15.6k/23.3k texels; front only 33%) | side-view columns 379/646, rows 375–500 = **sideburn/beard mass**, placed 17–24° too far frontward | **H2** pose error |
| "black blotch on the jaw" | jaw (z −0.12–0.015) | **front 75%** (17.9k texels) | photo rows 511–559 = **chin-bottom + neck shadow / collar** one notch up | **H4** (+25 px at the chin) |
| doubled soft glasses/hair edges, marbled patches (r30 view) | everywhere twins co-paint | first twin of each pair | second twin subordinated to 0.3–0.6% winner share but its offset content still feeds feathered blending + gradient-domain compositing | **H1** duplicate-angle twins |

Cross-check on content landing (independent of the winner maps): the
side_left views' LIP-box pixels landed at world azimuth **+31.6°**
(full-span twin) / **+24.6°** (windowed twin) at chin height — on the
cheek, not the mouth (the front view's mouth content sits at az +8.2°).
The side views' NOSE-box pixels landed on the philtrum/mustache zone
(z 0.21 vs the nose's 0.13–0.27). Sideways displacement ≈ the measured
pose error; vertical displacement ≈ their own residual row error.

## 3. Hypothesis verdicts

### H1 duplicate-angle poisoning — CONFIRMED as a mechanism, minor as a painter

Each twin registers independently (its own alpha bbox → its own canonical
scale; then width-profile + 2D + overlap + flow stages run per view):

- registration disagreement (windowed vs full-span twin): width-profile
  scale 1.07 vs 1.09 (side_left), 0.91 vs 0.77 (side_right), 1.01 vs 1.03
  (back), plus differing reg2d scale/shift picks;
- vertical CONTENT offset between registered twins (masked NCC argmax over
  the head band): **back 57 px, side_left 31 px, side_right 9 px** @1024
  ≈ 0.13 / 0.060 / 0.017 world units ≈ **16% / 7% / 2% of head height**;
- co-painted texels: back **484,801**, side_left **305,054**, side_right
  **220,158**; color MAE on co-painted texels (after delight/tone
  equalization) 12.7 / 11.3 / 9.8 (0–255).

The consistency gate + first-into-union order subordinate the SECOND twin
to 0.3–0.6% global winner share (15.6k/7.4k/10.2k texels), so twins do not
own the marquee defects — but their offset copies still enter feathering
and the gradient-domain composite (their painted sets stay 225k–506k
texels), producing the doubled soft edges and marbled ownership visible at
r30, and they double the bake's reference cost. The arrangement is never
legitimate: two independent syntheses of one angle are mutually
inconsistent by construction (REPORT.md wave-2 law).

### H2 pose error paints sideways — CONFIRMED, and bigger than the audit's 12°

Head-band IoU ruler (the viewgen-audit instrument: production
`register_matte_to_clay` on the full bust, IoU scored above the shoulder
flare — full-bust IoU is pose-blind on busts) against the E20 MESH:

| view | declared | measured | IoU@measured | IoU@declared |
|---|---|---|---|---|
| full side_left | +90° | **+65.0°** | 0.923 | 0.840 |
| full side_right | −90° | **−67.5°** | 0.862 | 0.796 |
| win side_left | +90° | +67.5° | 0.945 | 0.860 |
| win side_right | −90° | −70.0° | 0.905 | 0.846 |
| back | 180° | (sweep flat — noise, audit caveat) | | |
| front | 0° | −2.5° (flat) | 0.785 | 0.785 |

The bake feels **20–25°** of azimuth error (the audit's 12.5° was
measured against the GENERATION clay; the e20 mesh's head differs again).
The pose-refinement machinery (`estimate_view_pose`, window ±25°, step 5°,
accept on silhouette IoU > declared + 0.02) could numerically have reached
65°, but it is **hard-disabled on the orthographic path**
(`texturing.py` sets `refine_reference_poses = False`; every e20
registration row reads `pose: {refined: false}`) — and even enabled it
would have kept 90°: its full-silhouette objective MAXIMIZES near 90°
(0.957@90 vs 0.920@50 on side_left) because the torso dominates. Only a
head-band objective sees the error. Effect measured in §2: side content
rotated 17–24° frontward onto the cheeks (67% cheek ownership) = the
beard patches.

### H3 lens transparency — REFUTED as the ghost's cause

The ghost is not a view painting THROUGH the lens: side views win ≤2% of
every frontal defect region except the visor shelf itself (59% of the
right shelf half, painting their own lens pixels — dark + reflections).
The bright "face inside the lens" is the FRONT view painting the photo's
forehead-skin + glasses-top rows onto the mesh FOREHEAD (100% front-won,
58% skin) between two dark bands (hair-overhang underside above, visor
shelf below) — an H4 consequence, not transparency.

### H4 source-frame misregistration — the DOMINANT mechanism (discovered)

§1's table is the defect map: every front-facing defect is the front
photo's content displaced +25…+42 px (8–13% of head height) downward
relative to the mesh's anatomy — mustache/lips onto the nose, glasses onto
the forehead, nose-bridge skin onto the visor, neck shadow onto the jaw.
The front view wins 75–100% of all frontal texels (real photo, protection
absolute), so it, not the references, painted the marquee defects. Root
cause: mesh-vs-photo canonical-frame proportion mismatch (2mv windowed
conditioning) meeting a registration doctrine that is recenter-only for
the source. e18 partially masked this by ACCEPTING a gradient-NCC source
pose (az −1.2, el −19 — the tilt compensates part of the row offset);
e19/e20's estimate was `ncc_vetoed_by_silhouette` → declared (0,0).

## 4. Corrected bake (shipped: `out/laurent-bust-redo/e20_rebake_fixed/`)

Recipe (all input-side; the final bake runs the UNPATCHED production
`bake_projection_texture` via
`texture_forensics_bake.py --no-capture`):

1. **Full-span references only** — the windowed twins never reach the bake
   (kills H1).
2. **Ruler-measured azimuths**: side_left `@65`, side_right `@-67.5`,
   back `@180` (kills H2; numeric angles ride the existing
   `--texture-reference-angle` parser).
3. **Source row law** (`texture_forensics_source_rowlaw.py`, kills H4):
   feature-anchored piecewise-linear row displacement of the source photo
   in RAW space — anchors from §1 (+39/+42/+36/+25 canonical px at
   glasses/nostril/mouth/chin, ÷ the recenter scale 870/916), alpha-bbox
   endpoints FIXED (crown 232, window cut 975) so the production recenter
   mapping stays bit-identical, monotonicity asserted. Post-warp
   verification: glasses-bottom edge lands at row 397 vs mesh target 392.
   This is the viewgen-audit row law (docs/research/viewgen_audit.md §4
   step 3) applied to the source at bake-input time.

Ablation (both baked, closeups rendered at 2048 via
`scripts/oblique_closeups.py`):

- **fixA** (steps 1+2 only — the brief's example recipe): twin marbling and
  doubled bands gone, beard mostly off the cheeks; **nose-lips smear and
  forehead ghost REMAIN** (they are front-view defects; /tmp/texfor/fixA*).
- **fixB = fixA + step 3** (the shipped rebake): inspected at l30/l55/
  r30/r55 —
  - nose-lips smear: **GONE** (skin nose with nostrils; single mustache +
    lip line at the mouth),
  - lens ghost / double band: **GONE** (one dark visor on the visor
    geometry; forehead under hair),
  - beard: confined to jaw/chin; cheeks read as skin,
  - honest residuals: a dark crack-shaped streak at the right cheek/jaw
    boundary (inherited class, present worse in shipped e20), skin-toned
    reflection content in the right lens's lower edge (from the side_right
    reference's own lens reflections), mild cheek mottling, and the
    forehead reads fully hair-covered (the row law stretches the hair band
    ~24% vertically).

Final-bundle bake stats: coverage 0.7372 (front 0.2512 / back 0.2726 /
side_left@65 0.1227 / side_right@-67.5 0.1102); source pose declared
(0,0), NCC estimate vetoed by silhouette exactly as in e20. The final
uninstrumented bake's texture is byte-identical (MAE 0.0) to the
instrumented fixB run — which also proves the capture wrappers never
touched data. Verification renders:
`out/laurent-bust-redo/e20_rebake_fixed/closeups/e20fix_*.png` — inspected:
at l30 (the operator's failing view) the nose is skin with nostrils and
the mustache/lips sit on the mouth; at r30 the right lens carries no face
fragment (only the reference's own specular reflections at the lower rim);
no doubled bands at any of the four angles.

Note the coverage is LOWER than e20's 0.7497 (−1.25 points, and each side
view's own coverage drops from ~0.165 to ~0.11-0.12): the ruler poses
point both side cameras more frontward, so less rear-flank surface is
directly painted and more falls to mirror completion/fill — placement
correctness bought at a small direct-coverage cost. If more direct side
coverage is wanted, the right instrument is regenerating true-profile
views (the viewgen lane), not re-declaring wrong poses.

## 5. Production diff (this audit's lane: texturing.py + bake-call assembly)

Shipped in this change, tests green (`tests/test_texturing.py` +
`tests/test_hunyuan3d_backend_unit.py` + `tests/test_triposr_backend_unit.py`
= 205 passed on the settled tree, including the two new tests; one full-suite
flake during the run was a concurrent-edit import race from the parallel
pipeline-integration work — the "loop" enum test — green on rerun):

1. **Duplicate-angle guard** (`texturing.bake_projection_texture`): two
   SYNTHESIZED references at the same declared pose now raise a loud
   `ValueError` naming both labels and the two sanctioned escapes
   (conditioning-only consumer, or per-view measured azimuths). Real
   photos at one angle remain legitimate witnesses and are not gated.
   Test: `test_bake_refuses_duplicate_angle_synthesized_references`.
2. **Split-consumer flag** (the conditioning-only mechanism the windowed
   twins needed): reference views accept `consumer` ∈
   `both|geometry|texture` (mapping key or positional
   `texture_reference_consumers`, scalar broadcast, loud on unknown
   tokens). `hunyuan3d_runtime`: geometry-tag snapping skips
   `texture`-only views; `observed_views` assembly skips `geometry`-only
   views (they never reach the bake). `triposr_runtime` prepare drops
   `geometry`-only views (no reference-driven geometry stage exists
   there). Test:
   `test_texture_reference_consumer_normalization_and_prepare_filter`.

Deliberately NOT changed (other lanes / needs its own A/B): CLI flag
plumbing for `--texture-reference-consumer` (integration owner);
re-enabling reference pose refine on the orthographic path (its
full-silhouette objective is the wrong instrument on busts — a head-band
objective would be the real fix); productizing the source row law (the
recenter-only source doctrine is measured-correct on the single-view
flagship lane; the fix must key on multiview-conditioned geometry, which
is pipeline-integration territory). `view_consistency.py` and
`reference_generation.py` untouched.

## 6. Reproduction commands

```bash
# exact e20 re-run with instrumentation (15 min CPU)
.venv/bin/python scripts/experimental/texture_forensics_bake.py \
  --mesh out/laurent-bust-redo/e20_fixed_views/geometry.glb \
  --source out/laurent-bust-redo/windowed_set_v2/win_laurent_front_clean4.png \
  --ref back=out/laurent-bust-redo/windowed_set_v2/win_back.png \
  --ref side_left=out/laurent-bust-redo/windowed_set_v2/win_side_left.png \
  --ref side_right=out/laurent-bust-redo/windowed_set_v2/win_side_right.png \
  --ref back=out/laurent-bust-redo/viewgen_fixed/back.png \
  --ref side_left=out/laurent-bust-redo/viewgen_fixed/side_left.png \
  --ref side_right=out/laurent-bust-redo/viewgen_fixed/side_right.png \
  --out /tmp/texfor/repro_winfirst
# NOTE: with the new duplicate-angle guard this exact command now REFUSES
# (that is the point); re-running the forensics requires distinct angles
# or a pre-guard checkout.

# attribution
.venv/bin/python scripts/experimental/texture_forensics_analyze.py \
  --capture /tmp/texfor/repro_winfirst \
  --bundle out/laurent-bust-redo/e20_fixed_views \
  --regions /tmp/texfor/regions.json --out /tmp/texfor/analysis

# pose ruler + source row law + corrected bake
.venv/bin/python scripts/experimental/texture_forensics_ruler2.py
.venv/bin/python scripts/experimental/texture_forensics_source_rowlaw.py
.venv/bin/python scripts/experimental/texture_forensics_bake.py \
  --mesh out/laurent-bust-redo/e20_fixed_views/geometry.glb \
  --source /tmp/texfor/source_rowlaw.png \
  --ref back=out/laurent-bust-redo/viewgen_fixed/back.png@180 \
  --ref side_left=out/laurent-bust-redo/viewgen_fixed/side_left.png@65 \
  --ref side_right=out/laurent-bust-redo/viewgen_fixed/side_right.png@-67.5 \
  --out out/laurent-bust-redo/e20_rebake_fixed --no-capture
```

Instrument caveats owned: region skin-fractions are 3D-box statistics
(boxes include flanks/undersides a viewer never sees — grade placement
with the sample-row bands and the renders, not box color means); the probe
splats are nearest-texel debug renders (box-picking only); the head-band
ruler's back-view estimates are noise (flat sweep, audit caveat).
