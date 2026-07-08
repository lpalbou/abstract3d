# TEXTURE AGENT 3 — Observed-region close-range defects (multi-view face)

**Asset:** `artifacts/validation/iter3-multiview-fixed/face-2mv/` (mesh `geometry.glb`, views: bundle `input.png` az 0 source, `left_profile_clean.png` az +90, `right_profile_clean.png` az -90; ortho, mirror completion).
**Date:** 2026-07-05. Evidence under `/tmp/tex3/` (key crops in `report_evidence/`). All A/B baked at 1024 for iteration and 2048 for finals; renders at size 1000, crops at 2-3x (matches the owner's ~2000 px viewing distance).

**IMPORTANT CONTEXT:** the pipeline shifted under this work twice (renderer `baseColorFactor` honoring at 12:34; `texturing.py` +25 KB — pose refinement, fill detail synthesis, reference registration — at 13:38, which moved the estimated source pose az 17.5→20.0). Every A/B below was re-based onto the code tip current at measurement time; the deliverable comparisons (last section) are same-tip, same-inputs, mechanisms on/off. Bakes are bit-deterministic (verified: two identical bakes diff 0.0), so on/off deltas are real.

---

## 1. Provenance instrumentation (method)

`instrumented_bake.py` wraps the projector, blend, mirror fill, harmonic fill and outlier filter, dumping per-view raw/final weight+RGB atlases, blend coverage, mirror/harmonic masks (`instr/capture.npz`). Provenance is rendered two ways:

- **Winner/stage maps baked as textures** onto the same mesh and rendered from the same cameras as RGB (`prov_*.glb`, `instr/prov/*.png`: red=front, green=side_left, blue=side_right, magenta=mirror, gray=harmonic).
- **Exact screen→texel tracing** (`defect_trace.py`): a replica of the repo renderer's orthographic camera + a z-buffered texel splat maps every screen pixel to its atlas texel; defect rectangles then report composition (`instr/defect_trace.json`), dark-pixel-only composition (`dark_pixel_trace.py`), winner-weight histograms (`weight_hist.py`), photo-gradient x stretch statistics (`stretch_stats.py`), and per-view photo-pixel oversubscription (`density_stats.py`).

![provenance az0](report_evidence/provenance_az0_face.png)
*(az 0 face: RGB | winner map | stage map. Note how much of the mid-face is black/gray = no confident direct witness.)*

## 2. Per-defect provenance and verdicts

Numbers from the pre-13:38 tip (pose az 17.5 el 8); the structural conclusions were re-confirmed on the current tip.

### D1 — eye-socket smears + spurious dark blobs (QA az0 eye_count 3>2)
**Provenance:** dark pixels are 87–91% DIRECT projection, split ~50/40 between the front photo and the right profile. Front facing there is 0.5–0.57 (weight q50 0.03–0.12 vs 0.68–0.74 in clean areas); photo-gradient x stretch is 10–40x the clean-region level (nose-bridge blob: q90 0.17–0.23 vs 0.006). The blobs are the photo's own lash/iris/nostril pixels landing displaced (geometry-vs-photo feature mismatch, below) and stretched. Mip-map bleed ruled out (mip vs no-mip renders identical). Mirror copies contribute 4–11%.
**Verdict:** partially fixed. The az-0 eye_count failures present at session start (3>2 at el0+el10) are gone on the current tip after the morning's pose fix landed (az 20 vs 17.5 moves the photo's turf boundary off the eye); my guard removes the mirror-copy component. The direct displaced-content component REMAINS (see dark_debris at az -22.5..-45) — root cause is per-feature registration, see Limits.

### D2 — mid-face vertical tone seam (highest-value)
**Provenance:** the winner switches front→right-profile exactly at the nose line (winner share front 0.82 right of the seam vs 0.60/0.31 split left of it); the pale zone left of the seam has winner weights q50 0.13 vs 0.73 on the photo side, with 8% harmonic gray and mirror interleaved. The tone difference is content-level: the photo's own left-half shading (lum 0.808→0.754 across the face) vs the profile's synthetic even skin (0.779) — consistent with the mission brief's gain-residual finding.
**Fix (shipped):** `level_composed_seams` — per-region low-frequency offset fields on the mesh graph (Ivanov/Lempitsky seam leveling; regions = winning view / mirror fill). Two load-bearing safeguards, both discovered by measurement: (1) **boundary cap** — cross-region edges with step > 0.18 mean|RGB| are real material borders (hair|skin) and are excluded; an uncapped solve tinted the ear region and dropped the right-profile identity's worst face window from +0.10 to −0.13 SSIM; (2) **confidence pinning** — vertices whose winning witness is confident (>0.45) are pinned to zero correction, so the photo stays ground truth and leveling only bridges the weak bands between confident zones. A luma-preserving variant (chroma-only correction) was tried and looked worse (left face stayed gray; luma IS most of the seam) — study sheet `report_evidence/leveling_luma_study.png`.
**Verdict:** measurably improved, not eliminated. 1024 same-tip A/B: harness failures 16→14 with the mid-face chroma-seam failures 3→1; my scanline seam metric (fraction of face rows whose max luminance step exceeds the photo's own within-skin p99) 0.56→0.45 at az0 (0.375→0.33 at az−20); identity front 0.614→0.619, side_left 0.612→0.703, side_right 0.688→0.690. At 2048 the effect is subtler (see Limits). Remaining step is dominated by the photo's own baked-in shading gradient — content, not composition.

### D3 — ghost lip below-left of the mouth
**Provenance:** dark ghost pixels: 60% direct (front photo's lip content displaced down-left; winner front 0.59), 28% MIRROR COPIES of that displaced content, 13% harmonic. The right profile paints only bright skin there (lum 0.78). Displacement cause: per-feature geometry-photo mismatch — block-matching the photo against the untextured render at the bake pose shows the mouth wants (−4,+4) px @512, the nose (−10,+2), the eyes (+4,0): NO global 2D transform registers all features (measured, `local_registration_check.py`).
**Verdict:** the mirror-copy component is fixed by the consensus guard; the direct component shrank with the pose change (compare `report_evidence/ab2048_mouth.png` — the ghost line under the lip is fainter but present). Full removal needs local flow registration (prototyped, honest A/B failed — see Limits).

### D4 — dark stripe left of the nose
**Provenance:** 84–92% direct, winner = right profile (0.84): the profile's dark nose-contour shading wins a band on the nose flank adjacent to front-photo content — the same winner-switch family as D2, plus the photo's own nose shadow.
**Verdict:** improved by leveling (the band's tone is bridged; nose crop `report_evidence/ab2048_face.png`); the geometric shading content itself remains by design (leveling must not delete shading).

### D5 — tone patchwork on cheeks (observed-region winner switches; fill facets are agent 2's)
**Provenance:** left cheek is 78% right-profile turf at weak weights with front at 0.15; the DARK patches there were 50% direct profile content and **50% mirror copies** of hairline-adjacent content landing on skin (dark_pixel_trace: `D5_left_cheek_patch` mirror 0.496).
**Fix (shipped):** consensus guard in `mirror_fill_from_observed` — a copy is rejected only when the destination's observed 3D neighborhood is color-consistent (spread ≤ 0.09) AND the copy contradicts it (deviation > 0.22). Feature-rich destinations (eyes/lips: high spread) accept everything, so legitimate completion is untouched; rejected texels fall to harmonic fill. Verified inert on the starship (guard on/off textures identical) and on legitimate copies (face: 20–35 of ~11k copies rejected).
**Verdict:** mirror-copy component fixed at the cause; winner-switch tone component improved by leveling. Texel-scale patch INTERLEAVING (many small regions) remains at 2048 — see Limits.

## 3. Repo patches (shipped) + tests

`src/abstract3d/texturing.py`:
1. **`level_composed_seams(...)`** (new, ~180 lines): mesh-graph seam leveling as above; wired into `bake_projection_texture` for multi-view bakes only (single-view bakes structurally skip it — starship lane verified). Stats key `seam_leveling`.
2. **`mirror_fill_from_observed(..., consensus_guard=True, consensus_radius_ratio=0.03, consensus_max_spread=0.09, consensus_contrast=0.22)`**: guarded mirror copies as above.

`tests/test_texturing.py` (+4 tests, all passing; full suite 159 passed):
- `test_level_composed_seams_cancels_tone_step_and_keeps_detail` — synthetic two-region sphere with a 0.10 tone step + high-frequency stripes: step collapses >3x, stripes survive.
- `test_level_composed_seams_skips_material_edges` — hair|skin-magnitude boundary produces no leveling at all.
- `test_level_composed_seams_pins_confident_witnesses` — the confident region moves >2x less than the weak one.
- `test_mirror_fill_consensus_guard_rejects_material_crossing_copies` — mirrored-sheets fixture where twins cross a hair band: guard keeps skin copies, rejects hair-onto-skin; without the guard the dark copies land (scenario validity check).

Docs: CHANGELOG (fifth cycle section), KnowledgeBase (two new insights + failed-approach evidence).

## 4. Final same-tip A/B (mechanisms off → on, same code, same inputs)

| metric | 1024 off | 1024 on | 2048 off | 2048 on |
|---|---|---|---|---|
| qa.py failures | 16 | **14** | 9 | 9 |
| mid-face chroma_seam failures | 3 | **1** | 0 | 0 |
| identity front SSIM / MAE | 0.614 / 22.6 | **0.619 / 22.6** | 0.606 / 22.7 | **0.613 / 22.9** |
| identity side_left SSIM | 0.612 | **0.703** | 0.677 | **0.685** |
| identity side_right SSIM | 0.688 | **0.690** | 0.680 | 0.680 |
| seam metric az0 (rows > photo p99) | 0.559 | **0.541** | 0.555 | 0.567 |
| seam metric az+20 el8 | 0.525 | **0.480** | 0.511 | **0.474** |
| starship single-view | — | — | bit-identical | bit-identical |

Failure SETS at 2048 are identical off/on (dark_debris right-side views, az±90-family eye_count, front identity thresholds — all pre-existing classes owned by the projection/registration stage). No new failure class is introduced by the patches at either resolution; identity does not regress (side_right local window 0.136→0.124, above the 0.05 gate).

Crops: `report_evidence/ab1024_face.png`, `ab2048_face.png`, `ab2048_eyes.png`, `ab2048_mouth.png`.

## 5. Negative results (measured, reverted — kept as evidence)

- **Smear gate** (photo-gradient x stretch attenuation in the projector): 10–40x statistical separation on defect regions, but the REAL left eye sits at the same obliquity as the smears under a turned-head pose — the gate destroyed it (`report_evidence/smear_gate_failure.png`). Local signal alone cannot distinguish "feature painted where it belongs" from "feature smeared next to itself".
- **Quality-based mirror override** (strong twin beats weak direct, graph-smoothed): replaced whole weak zones with mirror content; boundaries speckled and the left eye was overwritten (`report_evidence/mirror_override_failure.png`).
- **Conflict threshold 0.25→0.18 + sub-0.05-weight demotion**: zeroed legitimate front content; pale mirror bands cut across the face (`report_evidence/conflict_minev_failure.png`).
- **Winner-agreement-conditioned blend weights** (sigma 0.08–0.10): no measurable change — the defect texels are mostly single-witness, nothing contradicts them.
- **Local flow registration of views to the mesh** (`local_flow.py`, block-matching on interior gradient fields, smooth confidence-weighted field, magnitude-capped 2%): fixes the D3 displacement in principle (mouth cell wants (−4,+4) px and the ghost visibly fades) but the full-bake A/B failed: warping the SYNTHETIC profiles on their weak gradient correlation damaged side identity (0.70→0.61), and source-only flow still increased dark_debris at az+45 and tripped a front local-window failure. Prototype + measurements retained for a future cycle; per-cell acceptance gating is the missing piece.

## 6. Remaining limits (honest)

1. **The photo's own baked-in shading** (lum 0.808→0.754 left-to-right across her face) is the largest surviving component of D2 at close range. Leveling bridges region tones but must not delete within-region shading; removing it is a delighting problem, out of scope for a projection bake.
2. **Displaced high-contrast fragments (D1/D3 direct component, dark_debris fails at az −22.5..−45)**: root cause is per-feature geometry-photo mismatch (nose −10 px, mouth (−4,+4), eyes (+4,0) @512 — non-rigid). Needs validated local flow (prototype above) or better geometry; every shortcut tried (smear gate, thresholds) destroyed legitimate content.
3. **At 2048 leveling weakens**: regions fragment at texel scale while the field lives on ~59k vertices; per-vertex dominant-region means mix content. A texel-graph or multi-resolution solve is the natural extension.
4. **az ±90 eye_count 0 failures** predate this work (profile eye smeared by the profile photo's own stretch at the sector boundary) and are a projector/registration class, not composition.
5. The harness's front-identity thresholds (SSIM ≥ 0.7 / MAE ≤ 22) remain unmet on ANY tip measured today (best 0.640); they gate the whole front-hemisphere stack (pose, projection, blending), not just composition.

## 7. Reproduction

```bash
source .venv/bin/activate
# instrumented capture + provenance renders
python /tmp/tex3/instrumented_bake.py /tmp/tex3/instr
python /tmp/tex3/provenance_render.py /tmp/tex3/instr /tmp/tex3/instr/prov
python /tmp/tex3/defect_trace.py /tmp/tex3/instr /tmp/tex3/rects_az0.json
# same-tip A/B (mechanisms off via monkeypatch / on via repo)
python /tmp/tex3/rebake_variants.py BASE2048 disable_new --resolution 2048
python /tmp/tex3/rebake_variants.py REPO2048 none --resolution 2048
python /tmp/verdict1/qa.py /tmp/tex3/bake_REPO2048 --out /tmp/tex3/qa_REPO2048
# crops + metrics
python /tmp/tex3/render_crops.py /tmp/tex3/bake_REPO2048/scene.glb /tmp/tex3/b_REPO2048
python /tmp/tex3/metrics.py /tmp/tex3/b_REPO2048/views <bundle>/input.png out.json
# starship lane
python /tmp/tex3/ship_check.py
```
