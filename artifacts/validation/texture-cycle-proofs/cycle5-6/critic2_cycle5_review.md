# CRITIC 2 — cycle-5 mathematical review (M1/M2/M3, single solver)

Date: 2026-07-07 08:34–09:2x. Repo `/Users/albou/abstract3d`. Prior:
REVIEW.md (c1), REVIEW_CYCLE2.md (c2). Tools:
`/tmp/critic2/tools/attack_c5_m1.py`, `attack_c5_m2.py`,
`attack_c5_m3_bound.py`; evidence `/tmp/critic2/c5/`. Tree STABLE
throughout (no mtime changes during review).

## Headline reproductions (all match the report)

| claim | my measurement | verdict |
|---|---|---|
| comp identity[front] 0.7022/13.86, full comp battery 0 fails | 0.702/13.9, **PASS 0 failed checks** (my own run of /tmp/c2d/qa_shadecomp.py on the published bundle) | **VERIFIED** |
| raw battery: MAE green, SSIM raw-diagnostic fail only | FAIL 1 = front raw SSIM 0.676 < 0.70; raw MAE 21.4 <= 22.0 green | **VERIFIED** (comp gate is the anchored one per the c4 ruling) |
| texture_qa PASS 13/13 all three on-disk bundles | face PASS, ship PASS, owl PASS (my runs) | **VERIFIED** |
| determinism md5 4d21c7ea | **four** independent bakes, one hash: published bundle + solver det1 + det2 + MY fresh canonical bake (962 s) all `4d21c7ea04066b026d9b63592be907e2` | **VERIFIED** |
| ship canary b8e2b0d4 | my fresh `bake_asset.py ship` (408 s): `b8e2b0d47ec4336a17067b59e1718455` == on-disk | **VERIFIED** |
| owl canary ff746509 | my fresh `bake_asset.py owl`: `ff746509ccb9429a6161cd40657df080` == on-disk | **VERIFIED** |
| suite 230 + 1 xfail | 230+1 at review start; **231 passed + 2 xfailed** after my additions | **VERIFIED** |
| shadow-off control 0.6974/14.10 (M1 load-bearing) | my comp run on their bundle_ctrl2048: **0.697/14.1** — fails the 0.70 gate without M1 | **VERIFIED** |

## M1 `reconcile_shadow_aprons`

| attack | result |
|---|---|
| (b) one-sidedness — can it brighten? | **NO — VERIFIED at both levels.** Trigger: source-brighter fields produce no candidates (deviation must be < −margin; my T1 no-fire). Application: `scale = clip(exp(...), floor, 1.0)` — algebraically capped at 1; measured max delta −0.000 on a fired case |
| (c) gauge stability — fitted to this face? | **MEASURED PER BAKE, not fitted — VERIFIED.** The −0.08 was this face's measured value. My T4: a pure +0.20-log exposure shift between views does NOT fire (gauge absorbs it); T3b: gauge tracks the injected shift to −0.195 and still isolates the local deviation. The pairwise-median construction is subject-independent; only `gauge_margin=0.10` and the guard thresholds are global constants |
| (a) dark albedo / occluder counterexamples | **BOUNDARY CONFIRMED, two constructions.** (1) Smooth dark occluder in the source photo (out-of-focus foreground blob): fires, darkens co-witnessed surface by up to −0.23 luminance — indistinguishable from a cast shadow by any signal the mechanism has. This is the source-authority doctrine's accepted trade (same class as printing the photo's real cast shadow). (2) Dark-albedo region with a SHARP edge: refused only when darker than `min_source_luminance` 0.35; at 0.42 it FIRES — the "edge-density refusal" measures edge density over the component + 2 px (a superset including the smooth interior), so it refuses edge-dense REGIONS (strand fields, the curtain), NOT clean regions with sharp BOUNDARIES. A misregistered clothing/collar boundary (smooth interior landing on co-witnessed skin via a few-px registration shift) is therefore reachable and would darken a band; bounded by component floors + vote pooling + the [0.45, 1] scale clip, and empirically absent on the proof asset (all detectors green, side budgets held) |
| (d) single-view structural no-op | **VERIFIED**: `len(view_rgb) < 2 -> None` (unit-checked); domain requires a reference to win. Canary md5s reproduce bit-exact (above) |

Regression tests added — `tests/test_shadow_apron_boundaries.py`:
`test_pure_exposure_shift_does_not_fire` (hard guard on the gauge) +
`test_smooth_foreground_occluder_is_not_printed` (xfail, documents the
doctrine boundary).

**Verdict: KEEP.** +0.0047 comp is real (control reproduces), the guards
hold where claimed, and the two boundaries I constructed are doctrine
limits (no single-source-photo signal can separate them), now documented
and pinned.

## M2 `_cluster_core_texels_world`

| attack | result |
|---|---|
| resolution independence | **VERIFIED**: identical cluster counts and centroids (to 1e-4 world) for the same world content sampled at 256/512/1024 atlas density |
| chart-blindness (UV-seam complex) | **VERIFIED**: one physical blob split across two atlas islands (cols 226..286 with a cut at 256) forms exactly ONE cluster |
| degenerate merging (assigned) | **THRESHOLD QUANTIFIED**: single-linkage voxel clustering merges two distinct features at edge gaps <= ~1.1 link cells and separates at >= ~1.5 cells (voxel-phase dependent between 1 and 2 cells). Link cell = 0.006 x diagonal: on the face (interocular ~0.38, link 0.021) anatomy is safely separated; on assets whose distinct features sit closer than ~1-2% of the mesh diagonal they WILL form one complex and be stamped together. Downstream this is gated by the render veto (a merged stamp that creates/loses feature blobs is refused), so the failure mode degrades to a refused stamp, not a shipped defect |

**Verdict: KEEP.** The construction is the right one (world units, chart
agnostic); the merge threshold is inherent to single-linkage and
documented.

## M3 bounded photo-truth exemption + speck consolidation

| attack | result |
|---|---|
| the bound prevents new-worst-offender views | **VERIFIED END-TO-END on the shipped bytes, with a semantics finding.** My independent battery-wide micro-fraction profile (their own `_compact_dark_blobs_px` construction, 15 veto views, 896 px) on c4-published vs c5-published: battery worst 0.003659 -> 0.003672 (**+1.3e-5, same view az−22.5 el0 — a tie within noise; no new worst offender**). BUT az−35 el0 grew +0.00096 (0.0021 -> 0.0031): legal under the ADVANCING-baseline semantics, which is the finding: |
| | **SEMANTICS FINDING (SUSPICIOUS, recommendation filed):** the docstring's invariant is enforced PER STAMP against a baseline that advances with each acceptance. Cumulative growth across n accepted stamps is bounded by n x 0.0003 (non-exempt creep; the +0.0003 budget re-arms per acceptance) plus exempt growth up to the battery-worst. Measured: ~7 stamps produced +0.00096 at one view — inside the letter of every per-stamp check, triple the single-stamp budget. All absolute detectors stayed green (raw battery worst dark 0.0029 vs 0.003 gate — 97% of the gate, knife-edge). RECOMMENDATION: add a cumulative veto vs the ORIGINAL pre-repair baseline (keep `pre_renders`' micro map; refuse if post > original + 0.0003 AND post > original battery worst). Cheap, closes the creep without touching the exemption |
| photo-confirmed test under wrong-basin registration | **INHERITED RISK, not independently falsified.** `gate_ok` is built by registering the source photo to the CURRENT render with the identity gate's own construction; under a wrong-basin registration the exemption would confirm displaced content self-consistently. Unreachable on the current recipe (pose pinned az+20/el+8 at NCC 0.0152, bit-deterministic — four bakes, one hash), governed by the FACE-21 checklist. Flagged for Critic 1's ledger as a coupling: the exemption's validity is exactly as strong as the registration it inherits |
| pixel-exactness of the feature footprint | **QUALIFIED WORDING**: the footprint is the feature blobs' own pixels dilated by 2 plus a splat z-buffer visibility — pixel-exact modulo a 2-px collar and splat approximation. The distinction that matters (pixel footprint vs bounding box) is real and their measured eye/brow outcome depends on it; the collar is fine print |
| speck consolidation | 17 texels lifted on the final bake (their number; consistent with the conservative stamps). No independent attack mounted beyond the footprint check above — the mechanism's blast radius on the shipped bake is 17 texels |

**Verdict: KEEP, with the cumulative-bound recommendation.**

## Integration

- **Determinism:** four independent full bakes (their det twins, the
  published bundle, my fresh recipe run) -> one md5. The strongest
  determinism evidence any cycle has produced.
- **Frozen-canary contract:** both canaries REBAKED FRESH by me from the
  certified recipes reproduce the certified hashes bit-exactly
  (ship 408 s, owl ~19 min). The contract is real, not aspirational.
- **Cross-mechanism (M1 tone under M3 stamps):** order in the pipeline is
  M1 (pre-solve, compositor) -> fringe stage (post-solve). Their banked
  arms are internally consistent (L1-only +0.0047; final-tree shadow-off
  control −0.0048 — I reproduced the control's numbers exactly), so M1's
  tone shift did not destabilize M3's acceptances on this asset. The
  photo-truth confirmation threshold (0.6 share) sits far from the
  measured shares; no interaction pathology found.
- **The margin:** +0.0022 comp SSIM is the thinnest gate margin this
  pipeline has shipped. It is bit-reproducible today; any tree movement
  respends it. The MAE margins (+1.14 comp, +0.55 raw) are healthier.
  My az−35 micro finding (0.0031 vs 0.003-family gates) and the raw
  dark worst (0.0029) are the two knife-edges to watch in cycle 6.

## Recommendation matrix

| mechanism | recommendation |
|---|---|
| M1 shadow-apron reconcile | **KEEP** (guards verified; 2 boundary tests added; note the edge guard is region-density, not boundary, semantics) |
| M2 world voxel clustering | **KEEP** (all three claims verified; merge threshold documented) |
| M3 exemption + consolidation | **KEEP** + implement the cumulative-baseline veto (closes the n x 0.0003 creep the advancing baseline re-arms) |
| Published face artifact | sound to keep: every headline number reproduces on the exact shipped bytes, and the recipe re-derives them bit-exactly |
