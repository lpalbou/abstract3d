# CYCLE-5 SOLVER — ORDER 2: the final +0.012 compensated SSIM (FACE-14)

**Scope:** Critic 1's cycle-5 ORDER 2 — close the FACE-14 compensated
identity gate (comp >= 0.70 SSIM / <= 15.0 MAE) from the freshly published
baseline (comp 0.688/14.42), attacking the ruling's own decomposition in
order: L1 neck/jaw wash (0.0102), L2 curtain/temple (0.0086 + 0.0060 +
FACE-09), L3 mouth-complex formation at 2048 (0.0026). Repo
`/Users/albou/abstract3d`, work under `/tmp/c5/`. Date: 2026-07-07.
Harnesses: `/tmp/verdict1/qa.py` (raw), `/tmp/c2d/qa_shadecomp.py`
(comp, authoritative gate), `scripts/texture_qa.py`.

## VERDICT: THE GATE CLEARS — comp identity[front] 0.7022 / 13.86, published

| gate @2048 (canonical recipe) | published c4 baseline | c5 final (published) |
|---|---|---|
| comp identity[front] SSIM/MAE (gate 0.70/15.0) | 0.6877 / 14.42 FAIL | **0.7022 / 13.86 PASS** |
| comp full 28-view battery | FAIL 1 | **PASS (0 failed checks)** |
| comp sides L / R | 0.6852/12.27, 0.6976/14.48 | 0.6774/12.52, 0.6942/14.64 (budgets 0.55/24 green) |
| raw identity[front] | 0.662 / 21.69 | 0.6765 / **21.45** (MAE budget 22.0 green; SSIM raw-diagnostic) |
| raw battery detectors, 28 views | all green | **all green** (worst dark 0.0029 az-22.5; skinHair 0.0044 vs 0.010; eyes in bounds everywhere) |
| texture_qa | PASS 13/13 | **PASS 13/13** |
| determinism | 3 bakes, 1 hash | **3 bakes, 1 hash** (`4d21c7ea04066b026d9b63592be907e2`) |
| ship canary md5 | b8e2b0d4... | **b8e2b0d4... == on-disk** (certified recipe, `source_pose_override=(30,15)`) |
| owl canary md5 | ff746509... | **ff746509... == on-disk** |
| test suite | 225+1 | **230 passed, 1 xfailed** (+5 new tests) |

Published to `artifacts/validation/iter3-multiview-fixed/face-2mv/` via the
FACE-21 checklist (staged, all three harnesses verified on the exact staged
bytes, then re-verified on the artifact directory after the copy; checklist
now in `docs/methodology.md`). Publication block in the bundle metadata
records the verification.

**Banked ledger (each arm a full canonical-recipe bake on one tree):**

| arm | comp front | delta | mechanism |
|---|---|---|---|
| published c4 baseline (= my re-bake, md5 `c39d65bf`) | 0.6877 / 14.42 | — | |
| + L1 shadow-apron reconcile | 0.6924 / 14.18 | +0.0047 | `reconcile_shadow_aprons` |
| + L3a world-space complex clustering | 0.6961 / 14.04 | +0.0037 | `_cluster_core_texels_world` |
| + L3b photo-truth exemption (bounded) + speck consolidation | **0.7022 / 13.86** | +0.0061 | veto exemption + `_consolidate_render_specks` |
| control: shadow OFF on the final tree | 0.6974 / 14.10 | −0.0048 vs final | isolates L1 on the tip (consistent with the c4-tree pair) |

Total banked: **+0.0145** (needed +0.0123). MAE improved 14.42 -> 13.86.

## L1 — NECK/JAW APRON WASH (mechanism: `gradient_compositing.reconcile_shadow_aprons`)

**Provenance (instrumented canonical bake, `/tmp/c5/prov_*.py`) — the
ruling's hypothesis had the sign REVERSED.** The candidate mechanism was
"the photo's under-chin shadow baked into albedo". Measured: the OPPOSITE.
The wash texels (gate frame 313-391 x 595-740) are won 92% by side_right
(w50 0.17-0.42, its sample lum 0.775 — its own lit reading), while the
FRONT photo validly samples the same surface at lum 0.54 — its under-chin
cast shadow — with projection weight ~0 (facing p50 0.45 at the
down-sloping neck vs the ortho-source facing threshold 0.4; feathered to
nothing). Front-vs-side_right log delta at the wash: **-0.35 vs a -0.08
global gauge**; the source's reading is a very smooth field (p5-p95
0.52-0.56 — a cast shadow, not strands). The gate wants comp lum ~130/255
= exactly the source's reading; the renderer's flat-biased headlight
(~0.9 at the neck, measured with a white-texture render) cannot absorb a
real cast shadow, so only the albedo can carry it.

**The mechanism** (the dual of `reconcile_specular_lobes`, same file, same
guard vocabulary): where a reference wins co-witnessed bright surface and
the source validly reads it darker beyond the pairwise gauge + margin,
with the deviation smooth in the source's own photo (edge-density refusal:
the curtain edge measures p85 2.8-5.8 and is refused; the shadow apron
0.5-1.3), the composite adopts the source's shading baseline
(gauge-corrected into the composite's exposure frame). One-sided darkening
only; every consumer (each reference's rgb + the blend anchors) keeps its
own detail verbatim (the correction reduces to a smooth per-consumer
luminance scale — no chroma or content import path exists); world-ball
fragment merge before the size floor (the atlas cuts one apron into
sub-floor UV fragments — measured 0.054-0.125 world gaps at merge radius
0.061). Runs pre-solve so gradients and anchors carry one story.
Source-valid-only is the witness contract: the photo-curtain parallax band
inside the same gate cluster (photo hair over side-witnessed skin, 0.0049
loss share) carries NO source evidence and is NOT treated (see limits).

**Measured failure ladder:** naive absolute-target on my reconstructed
texture −0.0103 (artifact of the reconstruction path, not the mechanism —
KnowledgeBase'd); offset-field extrapolation beyond components −0.0014
(darkens photo-bright surround); no-gauge target +0.0027 with double the
side cost (the gauge matters); pooled-zone variant +0.0008 (too diffuse);
absolute-target confined to kept components (shipped) **+0.0034** at
texture level, **+0.0047** through the pipeline (gradients + anchors carry
it). Sides: side_left 0.6852 -> 0.6795 (budget 0.55 — the left-jaw apron
component is honest: same class, its own photo evidence), side_right
0.6976 -> 0.6949.

**Crops:** `FINAL_L1_neck_jaw.png` (photo | published | c5): the flat lit
tan on the neck becomes the photo's shadow gradient; the under-jaw apron
carries the source's tone. A/B at az0/±35: `jawband_az*.png` — reads as
under-chin shading; all detectors green.

## L2 — CURTAIN RIGHT + TEMPLE + FACE-09 (measured; mostly geometry/witness-bound)

**Curtain (0.0086):** full-resolution anatomy (`loss_anatomy.py`) splits
the cluster into: (a) 0.0082 render-dark-where-photo-skin — the mesh
curtain's inner edge oscillates against the photo's silhouette; those
texels are side_left's CONFIDENT witness (w50 0.92, facing 0.97 — the
curtain surface really is there and really is hair; the front's bright
0.688 samples there are grazing misprojections, facing 0.37, weight 0);
(b) 0.0157 both-bright strand-texture mismatch on skin side_left
confidently claims (w 0.77 — real pale skin seen between curtain and
cheek at az+90). Treating either demotes a confident witness onto
photo-absent structure at its own gate — the FACE-20 regression class.
CEILING experiments (unshippable gate-frame paint, `ceiling_l2.py`): full
content +0.0155, tone-only +0.0047 — the loss is STRUCTURE (strands), not
tone, and the structure belongs to a sheet the mesh hangs differently.
FILED as the ear-parallax family (capture remedy: the same subject with
the curtain pinned back, or curtain-sheet geometry). The film-band lane's
pale stripe piece (0.14 of the zone): my fringe-lane stamps reached parts
of the inner edge (visible in `FINAL_L2a_curtain_right.png`); the cluster
measures 0.0086 -> 0.0081.

**Temple (0.0060):** the zone is 38% film-band domain / 34% clamp / 16%
FACE-20 refill; side_right's weight there is ~0 (side_confident 0.01) so
no witness bars tone movement — but the GATE wants lum 38/255 (the
photo's curtain crosses the temple at the gate registration) and the
refill floor (1.02x the dark split ~= 0.44) is the C4 stroke-unprintability
invariant: tone cannot approach the photo without reintroducing the
FACE-20 stroke class. Measured levers: film-zone-confined clamp
strengthening +0.0004 (best of 4 variants; global variant −0.0044 —
darkens photo-bright surround); tone-only ceiling +0.0020, full-content
ceiling +0.0056. The reachable piece is STRAND STRUCTURE the source photo
carries only on the curtain sheet (billboard class, vetoed by design).
FILED: the C4-accepted residue is within ~0.002 of its honest tone
optimum; the rest is the same parallax family as the curtain.

**FACE-09 rectangle:** on this tip the az180 upper-back step measures
~3/255 across the old boundary with the clamp share fading 0.59 -> 0.00
over ~80 px (`f09_check.py`; contrast-stretched render
`FINAL_f09_az180_stretch.png`, published vs c5 indistinguishable) — the
C4 film-band rework already dissolved the crisp 2048 boundary solver 3
measured (−11/255, 81% fill). No further mechanism shipped; evidence
filed. The S-field cliff at `CLAMP_S_MAX` remains the structural owner if
it ever re-sharpens.

## L3 — MOUTH-COMPLEX FORMATION AT 2048 (the fringe lane's own follow-up)

Three repo changes to `feature_fringe_repair.py`, each closing a measured
2048 shortfall:

1. **World-space voxel-graph clustering** (`_cluster_core_texels_world`,
   link cell 0.006x mesh diagonal — the rescue detector's construction).
   Atlas morphology's linking reach is a texel count: at 2048 the same
   world gap spans twice the texels and UV cuts fragment features below
   the size floor BEFORE the world merge sees them. Measured complexes
   old -> new at 2048: the mouth ball 127 tx r 0.045 -> 154 tx r 0.059,
   the CHIN complex (0.687, 0.0, -0.238), r 0.192 — never formed at all
   under atlas morphology, now forms and stamps 28,298 texels; both ears
   now form (matching the 1024 semantics). +0.0037 comp.
2. **Photo-truth exemption, bounded** in the render veto's micro-island
   growth budget: the chin/mouth stamps were refused because their only
   "growth" was the photo's OWN lip-corner line (330 px, measured
   pixel-provenance: renders from stamped texels the registered photo
   confirms). New islands rendering >= 60% from photo-confirmed stamped
   texels are exempt from the RELATIVE budget — but the view must stay at
   or below the battery's own pre-repair WORST micro fraction (the
   unbounded version shipped the eye complex's full re-registration and
   two battery views crossed the ABSOLUTE debris detectors at
   0.0030/0.0032 vs 0.003 — absolute detectors bind regardless of
   provenance; the bound refuses that stamp back to trace mode, which
   banks clean). Feature-size new blobs stay banned unconditionally. The
   veto baseline now advances with each ACCEPTED stamp (one accepted
   stamp's exempted content otherwise counts as growth against every
   later candidate — measured as a blanket veto of all following
   complexes).
3. **Final render-informed speck consolidation**
   (`_consolidate_render_specks`) + an in-stamp speck guard: repaired
   texels rendering as NEW isolated sub-feature dark islands at any
   battery pose are lifted just above the dark class under that view's
   own shading (the FACE-20 displaced-refill floor discipline at micro
   scale). Texels inside any pre-existing feature-class blob's own pixel
   footprint are protected — measured: an unprotected lift brightened
   the az+90 profile eye's under-lash mass (eye_count 1 -> 0); a
   bounding-box footprint shadowed liftable debris (two views stayed
   over the gate); the shipped pixel-exact footprint threads both.

Net stamps at 2048 (final): chin FULL 28,298 tx, mouth FULL 6,085 tx,
brow/temple FULL 17,988 tx, left-ear FULL 884 tx, forehead FULL 2,998 tx,
nose-bridge FULL 4,784 tx, eye TRACE 9,308 tx (full mode correctly
refused by the bounded exemption). +0.0061 comp over the clustering arm.
The FACE-04 lip-edge dark-red dash closes with the mouth stamp
(`FINAL_mouth_az0.png`, 4x: dash and below-lip chips gone; the mouth
reads as one soft mouth). At 1024 the full compensated battery stays
PASS (0.705/14.8; the eye complex's stamp is now veto-refused there too —
one honest regression vs c4's 0.708, inside the battery's margins).

## HONEST LIMITS

- **The comp SSIM margin is +0.0022** — real but thin. It is
  bit-deterministic (three bakes, one hash) and survives the exact
  published bytes (re-verified in place), but any future tree movement
  can spend it. The MAE margin (+1.14) and raw-MAE margin (+0.55) are
  healthier.
- **The named neck CLUSTER did not close** (0.0102 -> 0.0106 in the
  ledger): its subject-right wash fragments sit below honest component
  floors and its curtain-band half is the witness-bound parallax class.
  L1's +0.005 came from the same physical class on the ADJACENT left-jaw
  apron (gate x 457-538) plus global MAE — the ruling's decomposition
  arithmetic (L1+L2 ~ 0.019) landed as +0.0145 with different geography.
  The gate cleared anyway; the per-cluster map is in
  `bundle_l3h2048/gate_loss_l3h.json`.
- **Curtain/temple are filed as parallax/witness-bound** with ceiling
  numbers (curtain full 0.0155 / tone 0.0047; temple full 0.0056 / tone
  0.0020) — closing them at texture level requires either demoting
  0.77-0.92-confident witnesses or printing curtain-sheet content onto
  grazed surface (the FACE-20 class). Capture remedy documented.
- **The raw front SSIM stays below verdict1's 0.70** (0.6765; that gate
  compares the raw render including baked shading against the photo —
  the anchored gate is the compensated one per the C4 T2 ruling; raw MAE
  is green with margin).
- The az0 jaw-line now carries a soft gray-taupe shadow band (the honest
  print of the photo's under-chin shadow + the chin stamp's tone field).
  It reads as shading at 2-3x; at 6x its chroma is slightly grayer than
  the photo's warm shadow (luminance-only rescale by design — chroma
  import paths are how reference tints leak). Cosmetic; detectors green.
- The speck consolidation lifted only 17 texels on the final bake (the
  bounded exemption already keeps stamps conservative); it exists as the
  safety net for the accepted stamps' fragments and fires more when
  stamps carry more fine content.
- Compute: the fringe stage adds ~1-2 min at 2048 (the exemption's lazy
  truth renders + the consolidation battery); the shadow reconcile is
  <5 s. Single-view bakes pay nothing for either (structural no-ops,
  enforced by tests + measured canaries).
- I did not re-run solver 1's 48-view stroke sweep; the 28-view raw
  battery, the 15-view veto battery, and the consolidation views cover
  the stroke-class angles, and my only darkening mechanism floors at
  0.45x on skin whose measured corrections sit at 0.79-0.91x (target
  luminance bounded >= 0.35 by `min_source_luminance`).

## PATCHES + TESTS (all in repo, suite 230 passed + 1 xfailed)

- `src/abstract3d/gradient_compositing.py`: `reconcile_shadow_aprons` +
  `apply_shadow_apron_scale` (new, ~230 lines with measured rationale),
  wired into `composite_gradient_domain` (`shadow_reconcile=True`
  default, multi-view only by construction), stats row
  `shadow_reconcile`.
- `src/abstract3d/feature_fringe_repair.py`: `_cluster_core_texels_world`
  (world voxel-graph clustering), `_first_surface_projection`,
  `_feature_blob_footprint`, `_micro_island_components`, the bounded
  photo-truth exemption + advancing baseline in
  `_render_structure_veto`, the in-stamp speck guard in `_gate_stamp`,
  `_consolidate_render_specks` (+ stats key `speck_lifted_texels`).
- `src/abstract3d/texturing.py`: one stats key (`shadow_reconcile` in the
  solver stats copy). `film_band_gradient.py` untouched (md5 `b4418585`
  unchanged).
- `tests/test_gradient_compositing.py`: +4 (shadow apron carries the
  source's reading and is one-sided; source-invalid aprons refuse — the
  witness contract; edge-dense content refuses; single-view no-op).
- `tests/test_texturing.py`: +1 (world clustering links across an atlas
  cut, keeps world-distant content separate).
- `CHANGELOG.md` (2 entries), `docs/KnowledgeBase.md` (+5 insights,
  nothing removed), `docs/methodology.md` (the publication checklist —
  the FACE-21 order's outstanding documentation item).

## ARTIFACTS INDEX (/tmp/c5/)

- **Published bundle**: `artifacts/validation/iter3-multiview-fixed/face-2mv`
  (texture md5 `4d21c7ea04066b026d9b63592be907e2`, publication block in
  metadata). Staging + verification: `publish_staging/`, `publish.py`,
  `publish_verification.json`, `qa_published_final/`, `qc_published_final/`,
  `tq_published_final.log`.
- **Bundles**: `bundle_pub` (published c4 baseline copy), `bundle_base2048`
  (instrumented baseline, `state.npz`), `bundle_shadow2048` (L1),
  `bundle_l3a2048` (L1+clustering), `bundle_l3h2048` (final; `det1/det2`
  determinism twins), `bundle_ctrl2048` (shadow-off control on final tree),
  `bundle_face1024` (1024 sanity), `bundle_ship2048b`/`bundle_owl2048`
  (canaries), iteration arms `bundle_l3b/c/d/e/f/g`.
- **QA runs**: `qa_*`/`qc_*`/`tq_*` per arm (logs + results.json).
- **Crops**: `FINAL_L1_neck_jaw.png`, `FINAL_L2a_curtain_right.png`,
  `FINAL_L2b_temple.png`, `FINAL_L3_mouth_chin.png`, `FINAL_mouth_az0.png`,
  `FINAL_eyes_az0.png`, `FINAL_f09_az180_stretch.png`, `jawband_az*.png`,
  `debris_ab_*.png`, `eye90_diff.png`, ledger sheets `gate_loss_*.png` /
  `gate_locs_*.png`, anatomy sheets `anatomy_*.png`, `clcrops_*.png`.
- **Provenance/analysis**: `prov_neck.py`, `prov_wash.py`, `prov_blueband.py`,
  `gauge_check.py`, `wedge_check.py`, `loss_anatomy.py`, `comp_locate.py`,
  `frag_diag.py`, `frag_atlas*.py`, `dip_visibility.py`, `prov_band2.py`,
  `film_masks.py`/`film_zone_report.py` (film-band internal fields),
  `f09_check.py`, `veto_autopsy.py`, `island_geom.py`, `speck_debug.py`,
  `eye_kill_check.py`, `complex_ab.py` (clustering A/B),
  `ceiling_l2.py`/`ceiling_all.py` (ceiling experiments),
  prototypes `proto_shadow*.py` (v1-v8 ladder), `proto_temple.py`.
- **Infra**: `bake.py` (canonical recipe), `bake_asset.py` (canaries with
  the ship's certified `source_pose_override`), `disable_shadow.py`
  (paired-control patch), `capture_film.py`, `fringe_replay.py`.

## REPRO

```bash
source .venv/bin/activate
python /tmp/c5/bake.py final2048 --res 2048                  # canonical recipe (md5 4d21c7ea...)
python /tmp/c5/bake.py ctrl2048 --res 2048 --patch disable_shadow   # L1 control
python /tmp/verdict1/qa.py artifacts/validation/iter3-multiview-fixed/face-2mv --out /tmp/out_raw
python /tmp/c2d/qa_shadecomp.py artifacts/validation/iter3-multiview-fixed/face-2mv --out /tmp/out_comp --shading-comp
python scripts/texture_qa.py artifacts/validation/iter3-multiview-fixed/face-2mv
python /tmp/c5/bake_asset.py ship shipcheck --res 2048       # canary (b8e2b0d4...)
python /tmp/c5/bake_asset.py owl owlcheck --res 2048         # canary (ff746509...)
python -m pytest tests/ -q                                   # 230 passed, 1 xfailed
```
