# SOLVER B (cycle 2) — Orders 5+6: mirror twin rescue integrated; side_right window + dark_debris closed

**Scope:** Critic 1 orders 5 (integrate `mirror_rescue_disc` into the current-stack
bake, FACE-15) and 6 (FACE-03/04/05 trace-ghost family: B2 side_right worst window,
B3 dark_debris marginals). Repo: `/Users/albou/abstract3d`, work under `/tmp/c2b/`.
Date: 2026-07-06 (post-midnight session). All numbers measured with
`/tmp/verdict1/qa.py` (896) and `scripts/texture_qa.py` on same-tip 2048 bakes;
bakes are deterministic (pose gate pins az+20 el+8 at NCC 0.0152 every run).

## TL;DR

| target | before (same-tip control) | after (integrated) | status |
|---|---|---|---|
| B1 eye_count az−90 el0/el10 | 0 / 0 | **1 / 1** | FIXED |
| B2 identity[side_right] worst window | **−0.132** at (425,429) | **+0.219** (gate 0.05); SSIM 0.657→0.682, MAE 20.7 | FIXED |
| B3 dark_debris az−22.5/−35 (4 lines) | 0.0030/0.0027/0.0033/0.0035 | 0.0025/0.0021/0.0027/0.0029 (gate 0.003) | FIXED |
| verdict1 failed checks @2048 | 8 | **2** (both pre-existing front identity, FACE-06/14) | net −6 |
| scripts/texture_qa.py face | PASS 13/13 | **PASS 13/13** | held |
| ship/owl no-fire | — | 0 discs; ship texture **md5-identical** with detector off | proven |

The two remaining fails are `identity[front]` SSIM 0.629 < 0.7 and MAE 22.3 > 22.0 —
the pre-existing FACE-06/14 aggregate (control: 0.632/22.1; my footprint −0.003/+0.2,
front worst-window 0.054 vs control 0.062, both PASS the 0.05 gate). Front lane is
owned by orders 2/4 (solvers c2a/c2d active there this cycle).

Repro of the final state:
```bash
source .venv/bin/activate
python /tmp/c2b/bake_instr.py FINAL --resolution 2048    # standard face bake, instrumented
python /tmp/verdict1/qa.py /tmp/c2b/bake_FINAL --out /tmp/c2b/qa_FINAL
python scripts/texture_qa.py /tmp/c2b/bake_FINAL
# reference outputs: /tmp/c2b/bake_integj2048 + qa_integj2048 (2 fails) + tq_face (PASS)
```
The bake call is the standard recipe (front az0 source + left +90 + right −90
`*_clean.png` references, `texture_completion="auto"`,
`projection_model="orthographic"`); the artifact bundle in `artifacts/` was left to
the shared final-rebake step — my reference bundles live in /tmp/c2b.

---

## 1. Provenance (ordered "provenance first")

### 1.1 The side_right −0.132 window is a REGISTRATION ARTIFACT of the broken eye

The worst 49-px window at (425,429) sits on the right EAR. Texel forensics
(deterministic 1024 re-bake reproduces the artifact bundle's failure exactly;
instrumented capture of every stage):

- The window's 500 texels are 99% observed, won by side_right at weights spread
  0.006–0.99 (not predominantly trace); 111/129 damage-classified texels sit inside
  the FRONT's contested (layered-zone) band. Stage colors show NO stage introduces
  the damage: blend / Poisson / final are the same content (`prov_b2.py`,
  `prov_b2b.py`).
- The decisive experiment (`window pair` + forced-registration cross-check): the
  qa harness registers the photo with a global NCC refinement. On the control, that
  refinement lands at residual (1.025, 0.120, **0.060**); with the eye repaired it
  lands at (1.025, 0.120, **0.047**). Forcing the repaired-bake's registration onto
  the UNCHANGED control render scores the same ear window at **+0.473** — no ear
  texel changed. The broken −90 eye (a strong NCC feature) dragged the profile
  registration ~1.3% off, and at the wrong alignment the high-contrast ear/hair
  boundaries anti-correlate. FACE-15's eye defect and B2's window are ONE defect.
- Corollary recorded in KnowledgeBase: worst-window "ghosts" can be pure
  registration artifacts of a defect elsewhere; check with a fixed-alignment
  re-score before attributing texel damage.

### 1.2 B3 dark_debris blobs are trace-weight deposits (order-6 family confirmed)

Blob-level texel provenance at az−22.5/−35 (`prov_b3.py`, 1024): the under-eye
islands are observed texels won by FRONT at winner-weight p50 0.02–0.06 and
side_right at p50 0.007–0.023 — exactly solver 2's 0.006–0.2 trace family under
confident skin surroundings. They sit inside the rescue disc's footprint on the
right side; the transplant replaces them wholesale (measured: all four gated lines
under 0.003 after).

### 1.3 Gated demotion (order 6 as literally specified): measured, and it loses

Two implementations of "demote trace-weight claims contradicted by the confident
local consensus", measured end-to-end on captured bakes (`proto_demote2.py`,
`exp_offline.py --demote`):

| variant | side_right window | eyes −90 | dark_debris −22.5 el0 → | verdict |
|---|---|---|---|---|
| per-texel demote + mirror/nearest refill (715 tx) | −0.135 (unchanged) | 0/0 | 0.0030 → 0.0046 | REGRESSION |
| component-gated demote + consensus refill (264 tx) | −0.137 (unchanged) | 0/0 | 0.0030 → 0.0047, eyes −22.5 el10 1→0 | REGRESSION |

Root cause of the failure mode: the flakes' consensus deviation (p50 0.12–0.26) sits
BELOW any threshold that spares legitimate feature texels (front-eye trace texels
measure dev p50 0.399 — they'd demote first), and every refill (mirror twin, nearest
confident) reads as new pale patches that the debris/eye detectors flag. This
reproduces solver 4's blanket-demotion failure with the gates the critic asked for;
the numbers are the evidence that the DEMOTION lever cannot close this family on
this stack, while the transplant lever closes it outright (§2). Not shipped.

---

## 2. The general mechanism (order 5): `detect_mirror_rescue_discs` + integration

### 2.1 Design (all quantities bake-internal, no feature classes, no hand coordinates)

Per-texel maps over DIRECT observed texels (voxel-ball stats, `_voxel_ball_stats`,
O(N), deterministic; ball = 2% of mesh diagonal; luminance surface-smoothed at
sigma = atlas/512 so speckle averages out and the scale is resolution-invariant):
witness quality W (ball-mean RAW winner weight — feathered blend weights measurably
starve the detector), feature energy F (ball std), blob response DoG (smoothed lum −
ball mean), twin witness Wt / twin density Ct (same stats queried at mirrored
positions).

A transplant disc fires only through ALL of:

1. **geometry symmetry** ≥ 0.55 (same gate as mirror completion);
2. **strong side**: W ≥ 0.35 AND F ≥ 0.05, off the symmetry plane;
3. **twin observed**: Ct ≥ 0.25 — unobserved twins belong to mirror COMPLETION
   (this is what silences single-photo bakes);
4. **twin weakness**: Wt ≤ 0.5·W — content well-witnessed on both sides is
   legitimate asymmetry and is NEVER touched;
5. **coherent dark core**: DoG ≤ −0.12 over ≥ 3e-4 of the direct count (bright
   speculars excluded — transplanting them fragments features, the v14-era −45
   trade);
6. **feature-empty twin**: pointwise |DoG| at the mirrored core ≤ 0.5× the core's
   (pointwise, because ball averages dilute any blob wider than the ball to ~0 and
   blinded this gate in the first implementation);
7. **plane-crossing refusal**: |center_axis| > disc radius (a half-transplant on a
   plane-straddling feature painted a black dash on the front lips — measured);
8. dedupe + max 4 discs; radius = clip(1.15·comp_p95, 2.5·r_feat, 0.0575·scale).

On the face exactly ONE disc fires (the eye: core 597 texels @2048, own witness
0.53 vs twin 0.14, own blob response 0.26 vs twin 0.05), stable across 1024/2048.

### 2.2 Placement + tone (both measured load-bearing; §3 has the failure ladder)

- **Axis-anchored placement**: destination texel x copies from mirror(x − s) where
  s is the MIRROR-AXIS component of (twin's evidence-weighted feature-dark centroid
  − geometric mirror center), capped at 0.4·r_feat. The pure geometric position made
  the repaired eye a second NCC attractor and pulled the SOURCE-pose identity
  registration into a bistable flip (two optima 1e-4 apart in NCC, 0.03 apart in
  SSIM: 0.632→0.601). The axis component agreed across every estimator tried
  (0.024–0.028); the in-plane components flipped signs between estimators and
  re-rolled the flip — noise, excluded.
- **Content-aware tone ring**: the destination annulus mean is computed over
  source-mask texels only; in-bake the ring contains not-yet-filled texels whose
  zeros biased the tone offset ~0.02 dark and pushed transplanted skin flecks over
  the dark-debris gate (−35 el10 0.0031 with the biased ring, 0.0029 with the fix).
- Transplants are WHOLE-DISC. Every partial-keep variant (source-witnessed
  exclusion, confidence ramps 0.25–0.5, dark-aware keeps, lateral-normal cuts)
  left kept fragments + transplanted band as SEPARATE blobs → eye_count=3 at
  az0/−22.5 (three new fails). Whole-disc + axis anchor keeps front localMin at
  0.054–0.062 (PASS) with no doubling.
- Wiring: after mirror completion in `bake_projection_texture`, on post-Poisson
  colors; transplanted texels are marked observed-as-completion (excluded from
  `direct_observed_mask`, exempt from fill/detail/floor exactly like mirror texels);
  stats in `stats["mirror_rescue"]`.

### 2.3 No-fire proofs

- **Ship** (single photo, symmetry 0.983): 0 discs; texture **bit-identical**
  (md5 bb603def…) to a same-tip rebake with the detector monkeypatched out.
- **Owl** (canary, symmetry 0.979): 0 discs (`/tmp/c2b/owl_rebake/metadata.json`).
  Their texture_qa fails (ship facet_cellular 0.386, owl fill energy 0.19) are the
  fill lane's open SHIP-08/OWL items — identical with my code disabled.
- **Synthetic**, in `tests/test_texturing.py` (4 new tests + 1 existing, suite
  51/51): fires on a weak feature-empty twin and end-to-end restores the feature;
  no-fire on (a) deliberately asymmetric content confidently witnessed on both
  sides, (b) unobserved twin (ship analog), (c) a weak twin carrying its OWN
  comparable structure at the mirrored location.

Accepted limit (documented, inherent to any mirror prior): a twin that is BOTH
weakly witnessed AND feature-empty is indistinguishable from a smeared defect; if
the true content there were a unique feature the photos barely saw, the transplant
overwrites it — the same wager mirror completion already makes for unobserved
texels, extended by the six gates above.

## 3. Measured iteration ladder (what failed on the way, so nobody retries blind)

| attempt | result | verdict |
|---|---|---|
| offline transplant, qa-localized disc (solver-4 recipe on current stack) | closes B1/B2/B3 at 2048 | mechanism confirmed; needs general detection |
| detection on FEATHERED blend weights | 0 discs (ball means starved) | raw winner weights are the witness signal |
| twin emptiness via ball average | fired on feature-bearing twin | pointwise sampling |
| 2nd disc on the mouth (plane-straddling) | black dash + flake fringe on front lips | plane-crossing refusal (gate 7) |
| full transplant, geometric mirror placement | front identity 0.632→0.601, localMin 0.045 FAIL; content at FIXED alignment identical (0.628/0.062) — pure NCC bistability | placement anchor needed |
| source-witnessed exclusion / confidence ramps / dark-aware / lateral cuts | eye_count=3 at az0 ×2, −22.5 (doubled features) | whole-disc only |
| full 3D evidence anchor (uncapped / 0.7·r_feat cap) | −90 eyes 0/0 (moved off the profile band) / front flip re-rolled | cap at 0.4·r_feat, axis-only |
| in-bake tone ring incl. unfilled texels | −35 el10 debris 0.0030–0.0032 marginal FAIL | content-aware ring |
| final (axis anchor 0.024, r=0.174, content ring) | **2 fails, all targets closed, 1024+2048 consistent** | shipped |

Honest robustness note: the disc radius sits in a measured window (~0.172–0.180 at
this mesh scale): larger re-triggers the front NCC pull, smaller re-exposes
under-eye flakes. The default (0.0575·scale cap) lands mid-window and behaves
identically at 1024 and 2048, but the window exists because two independent
verdict1 gates hover at their thresholds on this asset; the gates' knife-edge
nature (0.0001-level debris margins) is a harness property solver 2 documented
before me. If the front lane's film-band fix changes the NCC landscape, re-measure.

## 4. Before/after (deliverable-resolution 2048, same tip)

verdict1: control `qa_ctrl2048.log` (8 fails) → final `qa_integj2048.log` (2 fails).
1024 cross-check `bake_integj1024`: same disc (center/radius/shift within 2%), all
owned targets pass (side_right localMin 0.167, eyes 1/1). Crops (nearest-neighbor):

- `/tmp/c2b/report_eye_m90.png` — the −90 eye at el0/el10, before (undetectable
  sliver) vs after (structured eye; the harness's own detector localizes it).
- `/tmp/c2b/report_ear_window.png` — photo | before | after at (425,429): the ear
  content is visually unchanged; the score −0.132→+0.219 is the corrected
  registration (per §1.1 this window was collateral of the eye).
- `/tmp/c2b/report_debris_views.png` — az−22.5/−35 el10 under-eye region.
- `/tmp/c2b/eye_m90_photo_base_rescue_p90mirr.png` — photo truth vs before vs
  after vs the mirrored healthy +90 eye (the transplant's source).

## 5. Patches + tests + docs

- `src/abstract3d/texturing.py`: `_voxel_ball_stats` (new), 
  `detect_mirror_rescue_discs` (new, ~230 lines incl. rationale), 
  `mirror_rescue_disc` gains `source_shift` + content-aware tone ring, 
  rescue block in `bake_projection_texture` (after mirror completion) + 
  `stats["mirror_rescue"]`.
- `tests/test_texturing.py`: `_folded_plane_scene` helper + 4 detector tests
  (fire, asymmetric-confident no-fire, unobserved-twin no-fire, feature-bearing
  twin no-fire). Full texturing suite 51/51 on the end-of-session tip.
- `CHANGELOG.md` (mechanism + A/B numbers), `docs/KnowledgeBase.md` (2 insights:
  weak-twin transplant gate stack + placement/tone/registration findings).
- Not shipped, documented: gated trace demotion (2 variants, both regress — §1.3);
  partial-keep transplant variants (§3).

## 6. Hand-offs / notes for the shared tree

- identity[front] SSIM/MAE (FACE-06/14) remain the front lane's; my change moves
  them −0.003/+0.2 within the failing regime and front localMin passes. If the
  front solver's registration-sensitive work lands, §1.1's fixed-alignment
  re-scoring technique separates their content effect from NCC flips.
- The az−45 el10 "fragment trade" from the v14-era integration does NOT survive
  into this one: eyes=1 and debris 0.0020 at that view (order 5's condition met).
- `eyes=1` at az+22.5/+35 el0 in my runs matches the control — bounds [1,2] pass;
  the +22.5 el10 3-blob (FACE-16) does not appear in either.
- The artifact bundle `artifacts/validation/iter3-multiview-fixed/face-2mv` was
  rebaked mid-cycle by another solver (22:56); I did NOT overwrite it. Whoever runs
  the final artifact rebake gets my fix for free (it is in the default bake path);
  `/tmp/c2b/bake_integj2048` is a ready reference bundle.
