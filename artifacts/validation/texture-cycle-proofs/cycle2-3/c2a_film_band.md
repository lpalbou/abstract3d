# SOLVER A — cycle 2, ORDER 4 (film-band defect class: FACE-01 / FACE-02 / FACE-16)

Tree at delivery: `texturing.py fd6ca719 / triposr_runtime.py 776516c9 /
film_band.py 43a6f305` (the shared tree moved 4 times during the session;
every A/B below is same-tree unless noted).

## WHAT SHIPPED (repo)

New module `src/abstract3d/film_band.py` + integration:

- `triposr_runtime._tripo_project_observed_texture`: computes per-view
  film maps alongside the existing layered-zone gate (purely additive —
  no weight change in the projector).
- `texturing.bake_projection_texture`: `commit_film_band` before the
  blend (vacates bright mixture claims under the commit mask, updates
  `contested`), `demote_unwitnessed_rim` after the blend, mirror
  destinations removed inside the commit, `retone_film_band` after
  `texel_surface_smooth` / before `synthesize_fill_detail`, stats row
  `film_band`.
- Tests: `tests/test_film_band.py` (5) — dark-body split, zone extension
  + veto maps, consensus/veto/dominance commit semantics, single-view
  no-op, retone toward dark anchors with bright-context damping. Full
  suite 192/192 green.

## THE MECHANISM (what the data forced)

The band is a FUSED film: the mesh merges the wispy hairline into the
head, so there is no second sheet for the zone gate (layered density
0.017-0.054 << 0.10) while local contrast is high (std ~0.10). The beige
sheet at az0 is 55-74% direct front paint of bright skin+hair mixture
pixels; the pale curtain is harmonic membrane fill mixing skin+hair
anchors; the dark curls are side-view contested self-claims; the parting
flecks are shadow-stamped/edge texels. Full measurement chain in
`NOTES.md`; every discriminator that failed is listed there with numbers
(geometric relief, normal dispersion, gap spectra — all dead ends; the
photo's dark-body statistics are the only signal that separates).

Final architecture (`film_band.py`, all scale-free):

1. ZONE EXTENSION per view: hysteresis growth of the strong zone into
   `std>0.055 & density>0.02 & alpha>0.1 & dark-coverage-of-foreground
   >0.25 & near the dark-material main body`; large components only.
2. COMMITMENT: flag consensus among first-surface-imaging views + no
   base-witness veto (imaged un-flagged bin with df<0.25) + >=2 imaging
   witnesses + dark-dominance of the local observed claims.
3. COMMIT-COUPLED SURRENDER: vacate BRIGHT claims (all views) at
   committed texels; dark claims are film-consistent and stay.
4. FILM RETONE: committed fill toned from dark OBSERVED anchors
   (octant voxel-ball, growing scales), scaled by photo wispiness and
   dark-dominance; mirror banned inside the commit; zero-weight rim
   coverage demoted.

## MEASURED RESULTS (same-tree A/B, `repo_off` = mechanism disabled)

verdict1 (`/tmp/verdict1/qa.py`):

| bundle | fails | detail |
|---|---|---|
| baseline 1024 | 4 | crown az-135 el0 0.0009, el10 0.0011; dark_debris az-45 el10 0.0032; identity[front] SSIM 0.641 |
| candidate 1024 | 1 | identity[front] SSIM 0.643 (pre-existing, slightly better) |
| baseline 2048 | 2 | identity[front] SSIM 0.629, mean\|RGB\| 22.3 (pre-existing) |
| candidate 2048 | 2 | identity[front] SSIM 0.630, mean\|RGB\| 22.2 (same set, MAE better) |

- eye_count az+22.5 el0 @2048: baseline 1 (under-detects) -> candidate 2
  (correct); NO eyes=3 (third-eye blob) at any azimuth in either arm.
- `scripts/texture_qa.py`: PASS at 1024 and 2048 (both arms).
- ship/owl single-view: textures BIT-IDENTICAL with the mechanism
  present vs disabled (md5, `ship_owl_check.py`); `commit_film_band`
  no-ops below two views by construction.
- Acceptance crops: `ACCEPTANCE_2048.png`, `ACCEPTANCE_2048_el8.png`
  (baseline vs candidate, az 0/±22.5, 4x hairline band).

## PER-DIRECTION VERDICTS

- A1 HAIR-TONED SURRENDER — SHIPPED, the core win. Two changes vs the
  brief's sketch, both forced by measurement: (i) the tone source is the
  dark-material observed anchors in texel space (octant voxel-balls),
  not the zone gate's boundary sides — boundary anchors at the hairline
  are themselves mixtures; (ii) commitment (not just fill biasing) —
  the beige sheet is mostly DIRECT PAINT, so the fill prior alone leaves
  it untouched; bright claims must be vacated at committed texels.
- A2 PARTING HOLES — RESOLVED by the same mechanism at the acceptance
  poses: the az0 parting flecks were front shadow-stamps + membrane
  pockets; commit+retone replaces them with hair-consistent tone (see
  ACCEPTANCE az0 (0.50,0.28) tiles). Solver 3's dark-evidence floor was
  verified live (`floor_stats`) and untouched — the flecks were
  evidence-backed or observed, so the floor correctly never fired on
  them; the fix had to come at claim level, not fill level.
- A3 STRAND-BRIDGE SYNTHESIS — PARTIAL, deliberately indirect: with the
  band committed to hair tone, `synthesize_fill_detail`'s
  color-similarity donor weighting picks hair donors and carries hair
  micro-statistics across the band (visible in the el8 sheet). An
  explicit structure-tensor/LIC bridge was NOT implemented: cycle 1
  proved LIC noise renders "hair-like material", not strands, and the
  committed band now reads as combed hair continuation at 4x without it.
- A4 THIRD-EYE CURL (FACE-16) — PROVENANCE + GENERAL PREVENTION. The
  class is a fused wisp FLOATER committed dark from its single
  consistent pose: photo-consistent from that view, parallax-detached
  everywhere else. I reproduced the class live twice while building P1
  (eyelid dash at az0 from side-view zone flags; ear-rim spots at 2048
  from single-witness commits) and killed it generally with the flag
  consensus + >=2-witness + first-surface restrictions. On the current
  tip the 3-blob does not fire in either arm at 2048; the candidate
  cannot re-introduce it (commitment requires cross-view agreement the
  floater geometry cannot satisfy).

## HISTORY NOT REPEATED (verified against the v13/v14 trade)

- Flat-color fill reads "painted": the retone is spatially varying
  (anchor interpolation) + wispiness/dominance-scaled + detail synthesis
  on top.
- Stamping mixtures reads "flaky": mixture stamps are VACATED only where
  film tone takes over; elsewhere they stay (no new flake contrast).
- Pixel gating loses to region gating: extension works on region
  components; commitment is a texel-level consensus over REGION flags.
- Blanket demotion fails (solve4 G2): the vacate is bright-only,
  commit-masked, dominance-gated — 3.1k claims at 2048, not 21k texels.

## DEAD ENDS KILLED BY DATA (so cycle 3 does not repeat them)

Full detail in NOTES.md; headlines:
- Geometric discriminators for the fused band: relief, normal
  dispersion, gap spectra — no separation (the whole upper head ripples).
- Cross-view contested kills and weak-claim floors: strip valid eye
  paint through ray-swath zones (2D zone bins flag 3D occluded texels —
  first-surface restriction is mandatory on all zone-derived texel maps).
- Facing-based occlusion tests on this mesh: winding is inconsistent
  (median signed first-surface facing 0.01) — unusable; use depth
  corridors or photo-space tests instead.
- Depth-corridor witness veto: no usable corridor exists (floaters hang
  up to 0.44 diag in front of what they obscure; rear-through-head rays
  start at ~0.5 diag) — the veto must be photo-space (df at the bin).
- Ray-march "floater guard" (base-witnessed content behind the texel):
  cancels the legitimate temple curtain too (skin is always behind hair
  along the front axis) — replaced by flag consensus.

## HONEST LIMITS

- Bright wisp remnants at the az±22.5 temples remain (kept claims under
  the witness veto / dominance gate): committing them was measured
  strictly worse (crown-flake regressions at az-135, 0.0027 vs gate
  0.0008). They now read as thin wisp ribbons over hair-toned fill, not
  as a beige sheet; eliminating them entirely needs content synthesis
  (strand strokes), not statistics — same ceiling as solve4 G3.
- identity[front] SSIM 0.63-0.64 fails its 0.7 gate in BOTH arms at both
  resolutions (pre-existing; my changes move it +0.002/-0.1 MAE in the
  right direction). Not this order's defect class.
- The mechanism engages only for multi-view bakes (consensus and
  witness vetoes are multi-view concepts); single-view film bands keep
  baseline behavior — intentional, keeps ship/owl bit-identical.
- At 1024 the commit is proportionally smaller than at 2048 (2960 vs
  17630 texels; first-surface classification is coarser) — the visible
  effect at 1024 is correspondingly milder. Both resolutions improve on
  their own baselines.

## ARTIFACTS

- Final bundles: `/tmp/c2a/repo_dom4_{1024,2048}` (candidate),
  `/tmp/c2a/repo_off_{1024,2048}` (same-tree baseline).
- QA outputs: `/tmp/c2a/qa_repo_dom4_*`, `/tmp/c2a/qa_repo_off_*`,
  `/tmp/texture_qa/repo_dom4_*`.
- Acceptance sheets: `/tmp/c2a/ACCEPTANCE_2048.png`,
  `/tmp/c2a/ACCEPTANCE_2048_el8.png`; per-step A/Bs: `veto_ab*.png`,
  `consensus_ab.png`, `commit_ab.png`, `crown135_*.png`,
  `ear_witness_overlay.png`, `dash_*.png`.
- Prototype/measurement code: `/tmp/c2a/*.py` (instrumented bake `lib.py`
  + `proto_bake.py`; probes: provenance, zone, holes, ribbons, floaters).
- Repo: `src/abstract3d/film_band.py`,
  `src/abstract3d/backends/triposr_runtime.py` (film maps),
  `src/abstract3d/texturing.py` (integration),
  `tests/test_film_band.py`, `CHANGELOG.md`, `docs/KnowledgeBase.md`
  (+2 insights).
