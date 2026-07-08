# TEXTURE AGENT 2 — Hidden-surface fill quality (facets, flat mush, patchwork)

Mission: the owner's complaint that texture on surface NOT visible in the
input image(s) is low quality — flat "painted" mush on the rear head /
hairline, faceted flat-color polygon blocks at close zoom, patchwork mottle
on the single-view starship.

Everything below was prototyped under `/tmp/tex2/` against captured
pre-fill bake states (monkeypatch capture of `bake_projection_texture`
internals; `tex2lib.py`, `fill_variants.py`, `run_*.py`), then implemented
in the repo and re-verified end-to-end with fresh bakes through the real
pipeline. All logic is general-purpose: thresholds derive from data
statistics (mesh scale, texel pitch, observed-population statistics), no
subject-specific branches.

## Verdicts per hypothesis

### H1 — texel-domain smoothing after the vertex harmonic solve: CONFIRMED, implemented

Root cause verified: `mesh_graph_harmonic_fill` solved colors on VERTICES
and assigned each unseen texel its nearest vertex's color. Each vertex's
Voronoi cell (~70 texels at 2048 on the proof meshes) rendered as one flat
polygon — the "faceted blocks at close zoom" (contrast-boosted atlas crop:
`exp/face_atlas_fill_crop_boost.png`).

Two-stage fix, both measured:

| face fill region (1024 atlas) | flat-plateau frac | step frac | Laplacian energy |
|---|---|---|---|
| baseline (nearest-vertex) | 0.451 | 0.245 | 0.0241 |
| + IDW-3 vertex interpolation | 0.210 | 0.240 | 0.0145 |
| + texel-graph smoothing (12 Jacobi iters, k=8, observed anchored) | 0.183 | 0.158 | 0.0008 |

Ship equivalents: Laplacian 0.0035 → 0.0001. 4x-zoom renders show smooth
material instead of polygon mosaic (`exp/h1/render_*`). Barycentric
rasterization of vertex colors was also built and measured — equivalent to
IDW-3 within noise, so the simpler IDW (no atlas plumbing, no GL) shipped.

### H2 — detail synthesis in fill regions: CONFIRMED for statistics transfer, REJECTED for structure transfer

The flat wash is real: fill/observed local-variance ratio was 0.53 (face)
and 0.15 (ship) before; the fill had the correct average color and zero
micro-texture.

- Structure transfer (the literal H2 as written — copy observed high-pass
  residual patches): built twice (k-NN residual blending, then coherent
  shift-map quilting with material gating and variance-preserving
  blending). Both REJECTED on visual evidence: k-NN blending produces
  level-set "topographic" banding (`exp/h2/render_ship_h2a`), coherent
  copying produces chaotic misplaced panel fragments and bark-like shreds
  (`exp/h2/render_ship_h2c`, `render_face_h2b`). Metrics hit target
  (ratio ~1.0) while looking worse than the wash — the metric alone is
  gameable, crops decided.
- Statistics transfer (adopted): robust L1 local residual amplitude
  (log-domain, per channel; L1 not RMS so sparse panel-line edges do not
  register as noise amplitude — RMS transfer rendered as granite) +
  structure-tensor streak orientation, both transferred per material
  (normal agreement + base-color match) from surface-nearest observed
  donors; carried by deterministic multi-octave 3D value noise (seamless
  across UV charts by construction), LIC-smeared along the transferred
  orientation in proportion to donor anisotropy; applied multiplicatively,
  zero-mean, p90-capped, seam-feathered.

Measured (1024, full stack, `exp/final1024/`):

| fill/observed ratio | local std | gradient energy |
|---|---|---|
| face before → after | 0.53 → 0.83 | 0.60 → 1.04 |
| ship (completion none) before → after | 0.15 → 0.59 | 0.03 → 0.49 |

Rear head now reads as dark combed-streak hair mass instead of the
chocolate wash; ship hull reads as toned metal grain instead of billiard
plastic (`final2048/pair_face_az+180_el+0_zoom4x.png`,
`pair_ship_az+210_el-20.png`). The gradient-energy target ≥ 0.5 is met on
the face (1.04) and just under on the ship at the default gain (0.49 at
0.7; 0.51 at gain 0.8 — visuals preferred 0.7; gain is a caller parameter).

### H3 — mirror completion for single-view symmetric objects: CONFIRMED, implemented as "auto"

Fresh rebake of the ship with current code: observed coverage 0.062 (the
recorded bundle's 0.2357 came from the old perspective+wrong-pose bake).
`mesh_mirror_symmetry_score(ship, axis=1) = 0.9826` (face: 0.966, gate:
0.55). With mirror completion the ship's unobserved starboard side and
stern pick up REAL mirrored panel content where the fill was wash:
coverage 0.062 → 0.093 observed+mirror at 1024 (0.065 → 0.097 at 2048),
and the az+300 view shows genuine window/panel rows
(`exp/h3/render_h3_h1ab/`). Implemented as `texture_completion="auto"`
(bake + both backends + CLI); the Hunyuan backend now DEFAULTS to auto.
Explicit "none"/"mirror_symmetry" behave exactly as before; the same
0.55 score gate plus the existing confident-source and contested-exclusion
gates protect asymmetric subjects.

### H4 — anti-patchwork for the KD fallback fill: CONFIRMED, implemented

The KD inverse-distance fill's per-texel donor sets change abruptly
between neighbors — that discontinuity IS the patchwork mottle. The same
texel-graph smoothing now runs after the KD path too (fill Laplacian
energy: ship 0.0091 → 0.0011, face 0.0183 → 0.0024; before/after zooms
`exp/h4/render_ship_kd*`). Donor-constraint variants (normal gating alone)
were measured insufficient — the donor-set discontinuity, not donor
choice, dominates; smoothing addresses it directly.

## Repo changes (scoped to fill/completion + wiring)

`src/abstract3d/texturing.py`
- `mesh_graph_harmonic_fill`: nearest-vertex texel assignment → IDW over
  3 nearest vertices.
- New `texel_surface_smooth(...)`: Jacobi relaxation of fill texels over
  the KD graph of texel 3D positions; observed texels are Dirichlet
  anchors; normal-agreement-weighted edges.
- New `synthesize_fill_detail(...)`: the statistics-transfer pass
  (deterministic, seed parameter, `gain=0` disables).
- `bake_projection_texture`: new `fill_detail_gain=0.7` parameter; both
  passes run for propagated fills (`mesh_harmonic` / `nearest_observed_3d`),
  NOT for `backend_color_field` (TripoSR triplane prior — unmeasured, has
  its own character); `texture_completion="auto"` resolved via symmetry
  score; stats now report `texture_completion` (resolved),
  `texture_completion_requested`, and `fill_detail`.

`src/abstract3d/backends/hunyuan3d_runtime.py`: default completion
"none" → "auto"; schema enum + "auto".
`src/abstract3d/backends/triposr_runtime.py`: completion-mode parser and
schema accept "auto" (default unchanged "none").
`src/abstract3d/cli.py`: `--texture-completion` accepts `auto`.
`tests/test_texturing.py`: 6 new tests — smoothing relaxes a facet step
and keeps anchors byte-identical; detail synthesis adds variance to a flat
fill, preserves mean color, leaves observed texels untouched, is
deterministic, and `gain=0` is identity; harmonic fill interpolates at
texel resolution (unique-value count); end-to-end auto completion resolves
to mirror on a symmetric mesh. Full suite: 155 passed (re-run on the
current shared tree after other agents' edits landed).
`CHANGELOG.md`, `docs/KnowledgeBase.md` (3 new insights), `docs/api.md`.

## Regression guard (verdict1 face gates)

`python /tmp/verdict1/qa.py` on same-capture rebuilt bundles:

| bundle | failures |
|---|---|
| 1024 baseline fill | 7 |
| 1024 new stack | 6 |
| 2048 baseline fill (`final2048/face_before`) | 6 |
| 2048 new stack (`final2048/face_after`) | 5 |

Failures did not increase (mission gate). Remaining failures are
projection/pose classes untouched by fill (front identity SSIM/MAE at the
declared pose, missing right-profile eye, one 0.0032-vs-0.003 dark_debris
at az −45 whose evidence crops show pre-existing hairline shadows and lip
geometry, present in both before and after).

## Evidence index

- `final2048/pair_*` — labeled before/after sheets, all defect views + 4x zooms.
- `exp/h1/`, `exp/h2/`, `exp/h3/`, `exp/h4/`, `exp/final1024/` — per-hypothesis textures, renders, metrics JSON.
- `state/` — captured bake states (NPZ) for reproduction.
- `qa_out/` — full qa.py outputs (results.json + evidence crops).

## Honest limits and findings outside my lane

1. Detail transfer cannot invent content. The synthesized fill reads as
   the correct MATERIAL (hair streaks, hull grain), not the correct
   CONTENT (an actual hairline whorl, a specific panel layout). That
   requires generative inpainting, out of projection-bake scope —
   documented in KnowledgeBase.
2. Ship source-pose estimation looks wrong (other agents' lane, evidence
   left in place): `estimate_pose_photometric` picked az 42.5 / el −8 for
   the starship photo; the photo clearly looks DOWN on the deck (old
   recorded pose el +15 matches; renders `exp/ship_pose_*.png`). Its
   elevation grid is (−8, 0, +8) — +15 is not even a candidate. Coverage
   at the estimated pose is 0.062 vs 0.198 at the recorded pose, so a
   pose fix would roughly triple observed ship content and further shrink
   the fill problem. My fill improvements are pose-independent.
3. Ship gradient-energy ratio at default gain is 0.49 (target 0.5); the
   face is 1.04. Pushing gain to 0.8 crosses the line (0.51) but looks
   busier; I kept 0.7 and exposed the parameter.
4. The remaining face QA failures (identity front, missing right-profile
   eye) are projection/registration defects that fill cannot repair.
