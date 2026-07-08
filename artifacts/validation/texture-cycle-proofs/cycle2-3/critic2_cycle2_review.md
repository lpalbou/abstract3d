# CRITIC 2 — cycle-2 mathematical review (mechanisms A/B/C/D)

Date: 2026-07-06 05:07–06:0x. Repo `/Users/albou/abstract3d`. Prior review:
/tmp/critic2/REVIEW.md (cycle 1). Tools for every reproduction below:
`/tmp/critic2/tools/attack_a1_filmband.py`, `attack_a2_bistable.py`,
`attack_a3_calibration.py`, `attack_a4_shading_floor.py`,
`instr_c2_bake.py`; artifacts under `/tmp/critic2/c2/`. A proof here is a
command the owner can rerun.

Tree state at review: texturing.py mtime 04:43, film_band.py 04:43 — the
tree was STABLE throughout this review (unlike cycle 1). Test suite at
start: **192 passed**. After my additions: **193 passed, 1 xfailed**.

**Cycle-1 defect closed:** my cycle-1 CONFIRMED-DEFECT (mirror-source-gate
500-texel cliff) has been fixed in the tree (graceful top-up to the anchor
minimum by descending weight); my two xfail regression tests now PASS as
hard guards. Verified this review.

---

## SOLVER A — film-band commitment (`film_band.py`)

| claim | verdict | my evidence |
|---|---|---|
| verdict1 4 -> 1 @1024 (candidate vs same-tree off) | **VERIFIED** | their logs reproduce (repo_off 4 / repo_dom4 1) AND my own fresh instrumented 1024 bake through my own driver scores **FAIL 1** (identity[front] SSIM 0.643 — their exact number) |
| 2048: 2 -> 2 with MAE better | **VERIFIED** | their qa_repo_dom4_2048 log FAIL 2; on-disk face (pre-mechanism bake) FAIL 2 at 0.630/22.2 (my run) — the identity-only failure set matches |
| texture_qa PASS both arms | **VERIFIED** | my independent run on the on-disk face: PASS 13/13 |
| ship/owl BIT-IDENTICAL (mechanism no-ops single-view) | **VERIFIED (code + their md5)** | `commit_film_band` returns None for < 2 views (read); film maps are computed but never consumed single-view |
| flag consensus + veto + >=2 witnesses prevent floater commits (FACE-16) | **QUALIFIED — witness independence overstated** | see A1 findings below |
| "vacate blocks legitimate bright content" (veto/dominance protection) | **QUALIFIED — protection is scale-bounded** | see A1 findings below |

### A1 attack results (assigned)

**(i) Mirrored-photo witness counting — CONFIRMED, quantified.** The
`n_img >= 2` cross-confirmation counts the fabricated right profile (a
mirror copy of the left) as an independent witness. Unit construction:
two identical views satisfy the consensus and commit fires. Real-asset
census (my instrumented bake, capture inside `commit_film_band`):
**868 of 2,960 committed texels (29%) are witnessed by the profile pair
alone** (front+side_left 510, front+side_right 440, all three 1,142).
For that 29%, "cross-view agreement the floater geometry cannot satisfy"
is one photo voting twice: a mirror-symmetric fused floater WOULD pass
consensus there. Severity on this asset: low (the pair-only zone is the
rear hairline band, content is hair either way; harness shows no
regression) — but the report's independence argument does not hold in
that zone, and subjects with asymmetric rear content would rely on a
consensus that cannot see the asymmetry.

**(ii) Bright-legitimate-content erasure — failure mode CONFIRMED by
construction, with measured boundaries.** Three regimes measured on
their shipped functions (tools/attack_a1_filmband.py):

| construction (1024 plane, 2-view) | outcome |
|---|---|
| thin glossy sheen (6 tx) inside committed dark band, dark band 39% of claimed | **ERASED**: all sheen claims vacated (both views), retone 0.62 -> 0.21. The veto cannot engage (every imaging view flags the band — no non-flagging witness exists) and dominance cannot block (sheen << dominance ball ~1-2% of diagonal) |
| wide sheen (240 tx, ball-scale) | KEPT — dark-dominance blocks the vacate (their designed guard works at/above ball scale) |
| dark band >= ~50% of claimed area | mechanism silently NO-OPS (fails safe): "median of the bright half" collapses onto the dark mode, nothing classifies dark, no vacate/retone. The "scale-free two-mode split" is a quantile heuristic, not a mode split — dark-majority subjects (owl-like, rear-view-dominant) never engage the mechanism |

Regression tests added: `tests/test_film_band_boundaries.py` —
`test_film_band_dark_majority_fails_safe` (hard guard on the safe
degenerate regime) + `test_film_band_keeps_thin_bright_sheen` (xfail,
documents the sheen boundary; flips to a guard if the protection is ever
extended below ball scale).

**Verdict: KEEP.** Measured wins are real and reproduce exactly; the two
boundaries above are documented capability limits (specular hair sheen
narrower than ~1% of mesh diagonal inside a committed band will be
erased on future glossy-haired subjects), not regressions on the proof
assets.

---

## SOLVER B — mirror twin rescue (`detect_mirror_rescue_discs`)

| claim | verdict | my evidence |
|---|---|---|
| face 8 -> 2 @2048 | **VERIFIED** | their qa_ctrl2048 (8) / qa_integj2048 (2) logs; my instrumented 1024 bake fires the same single disc (right eye, 1,950 texels) and lands FAIL 1 |
| B2 side_right window flip (−0.132 -> +0.219) is a REGISTRATION artifact, no texture change at the ear | **VERIFIED NUMERICALLY (assigned)** | my full reproduction (attack_a2_bistable.py) with the harness's own registration: ear-window render delta **0.00/255** (their bundles, az−90 @896); self-registration residuals (1.025, 0.120, **0.0600**) control vs (1.025, 0.120, **0.0475**) final — matches their claimed 0.060→0.047 and the "~1.3%" drag (dy 0.0125); window score self-aligned control **−0.124** vs final **+0.541**; forcing FINAL's alignment onto the UNCHANGED control render: **+0.541** — the alignment alone produces the entire flip. (Their +0.219/+0.473 differ from my +0.541 only through their run's own worst-window localization; sign, mechanism, and residuals reproduce.) |
| dark_debris B3 closed by the transplant | **VERIFIED (their logs; consistent failure sets in my runs)** | the four gated lines absent from qa_integj2048 and from my 1024 candidate |
| transplant determinism | **VERIFIED** | my determinism pair (two identical 1024 bakes WITH the rescue firing): texture sha256 **identical** |
| both-twins-weak behavior | **VERIFIED SAFE (code)** | detection requires a strong side (W >= 0.35, F >= 0.05); two weak sides cannot trigger; W-vs-0.5·W excludes both-fire on the same region |
| ship md5-identical / owl 0 discs | **VERIFIED (logic + their md5)** | twin-coverage gate Ct >= 0.25 requires DIRECT twin observation; single-photo bakes cover twins by mirror COMPLETION (not direct), so Ct ~ 0 |

**HARNESS FINDING (documented for Critic 1):** verdict1's worst-window
statistic swings by ~0.67 under a 1.25%-of-bbox alignment shift on
UNCHANGED pixels. Any cross-bundle window comparison needs their
fixed-alignment re-scoring technique (now in KnowledgeBase); window
scores from different registrations are not comparable evidence.

**Minor (theoretical) sequencing hazard, flagged not filed:** in the bake
loop, `rescue_source_mask` is precomputed while `blend["rgb"]` mutates
per disc — with multiple discs a later transplant could read an earlier
transplant's destination as "healthy source". Unreachable on the proof
assets (exactly 1 disc fires; max 4 with overlap dedupe); recommend
masking sources against prior destinations if multi-disc assets appear.

**Verdict: KEEP.**

---

## SOLVER C — closed-loop fill-energy calibration + harness matting

| claim | verdict | my evidence |
|---|---|---|
| ship/owl PASS 13/13 at BOTH resolutions | **VERIFIED** | on-disk 2048 bundles (byte-identical to their ship_ready — hashes match) PASS 13/13 under my own harness runs; their after/ship_1024 re-run by me independently: PASS (fill energy 0.578, their exact number); owl_1024 log consistent |
| the on-disk artifacts were old-pipeline bakes | **CONFIRMED + RESOLVED** | current on-disk metadata now carries current-pipeline modes and fill_floor stats; textures = their ship_ready (installed after their report was written) |
| calibration does not Goodhart (assigned A3) | **VERIFIED with one documented one-sidedness** | smooth-plastic synthetic: scale 1.0, realized ratio 1.001 — zero noise injection on genuinely smooth observed content, upper gate safe; edge-plates: no granite (σ 0.115 vs observed band 0.096 at scale 1, ratio 2.16 < 2.5); resolution invariance holds (ratios 1.001/1.000 and 2.16/2.10 at 512/1024). ONE-SIDEDNESS: the σ guard only has authority when calibration RAISES the scale — at the lower bound 1.0 an over-energetic unit fill is accepted as-is (by design, "never dampen"); the upper texture_qa gate (<= 2.5) is the only guard there. Documented, not a defect |
| σ guard construction | **VERIFIED (math read)** | band-matched residual σ via masked Gaussian lowpass at half the coarsest carrier wavelength; L1/0.798 Gaussian σ estimate is correct; guard rescales down only |
| fill visual character at 4x | **VERIFIED (visual)** | my collage of the on-disk ship's own QA evidence crops: oriented mottled panel-flow material, no granite, no facet plateaus (vs cycle-1's flat wash / facet fields) |
| owl brightness root cause (backdrop measured, not subject) + matte-first fix | **VERIFIED (math read + gate outcomes)** | photo_foreground: RGBA alpha unchanged; RGB matted with the bake's own matte; degenerate-matte fallback EXPLICIT and recorded in results.json. All three assets pass calibration under the fixed harness (my runs) |

**Minor flag:** `photo_foreground`'s cache key falls back to `id(photo)`
when no cache_key/filename exists (one caller, `photo_seam_allowance`,
passes none) — id() reuse after GC could theoretically cross-contaminate
masks within a process. One-line fix (pass the path); no observed impact.

**Verdict: KEEP (both the pipeline calibration and the harness matting
change; the harness change is Critic 1's to adjudicate — my analysis
found the math sound and the fallback honest).**

---

## SOLVER D — identity decomposition + opt-in shading compensation

| claim | verdict | my evidence |
|---|---|---|
| perfect-texture floor SSIM 0.977 / MAE 11.45 @896 gate protocol | **VERIFIED (independent re-derivation, assigned A4)** | my own construction (white-texture shade field through the identical renderer; albedo reconstructed by division): SSIM **0.977** (exact), MAE **10.75** measured + **11.38** analytic `mean(A·(1−s))` vs their 11.45 (my division route carries quantization; the bracket 10.8–11.4 confirms ~11 of the 22 MAE budget is renderer shading). Shade field percentiles p5/p50/p95 = 0.878/0.906/0.988 vs their 0.878/0.906/0.984 |
| compensation is opt-in, registration untouched | **VERIFIED (patch read)** | `--shading-comp` argparse flag; photo-side multiplication after NCC alignment; raw gate default unchanged |
| re-tightened budgets "preserve strictness" | **QUALIFIED** | front 22→15: in floor-relative terms comp is LOOSER (raw allowed 10.55 MAE of texture error above the floor; comp allows 15 above ~0); in ceiling-relative terms roughly preserved (raw margin over their measured input ceiling 5.45, comp 6.66). The 15.0/24.0 constants embed THIS asset's populations. ADVISORY to Critic 1: accept the compensation math (exact by construction — verified), but set budgets anchored to the compensated input ceiling (ceiling + fixed margin) rather than absolute constants, or the gate's meaning drifts on the next asset |
| ORDER-8 elf-ear texture half does not reproduce (apex is hair-painted) | **PLAUSIBLE-VERIFIED (review)** | multi-probe evidence (witness audit, photo-boundary z-map, ball stats, 16-view acceptance battery with metrics I spot-checked: apex skin 0.002–0.008 in d2_accept_metrics.json, matching their ≤0.059 claim); their reconciliation of solve4's 33-38% (whole ear component incl. legit ear-body skin) vs apex-proper is arithmetically coherent. Not independently re-instrumented (their probes are consistent with each other and with the harness state; cost/benefit did not justify a re-instrumentation) |
| no pipeline change shipped | **VERIFIED** | tree diff shows their lane touched only docs; the D1.5 negative-results table (7 variants, all trades) is consistent with cycle-1's spent-lever findings |

**Verdict: KEEP the analysis; ADOPT the compensation patch subject to
Critic 1's budget policy (my recommendation above).**

---

## INTEGRATION (A5)

- **Test suite:** 192 passed at review start; **193 passed + 1 xfailed**
  after my boundary tests. My cycle-1 mirror-gate xfail tests now pass
  (cliff fixed in-tree) and act as hard guards.
- **On-disk artifact bundles (as they sit):** ship PASS 13/13, owl PASS
  13/13, face PASS 13/13 (texture_qa, my runs); face verdict1 **FAIL 2**
  (identity[front] SSIM 0.630 / MAE 22.2 — the pre-existing FACE-06/14
  family; the on-disk face bundle predates this cycle's mechanisms:
  film_band/mirror_rescue stats absent from its metadata). The A+B
  candidates measure FAIL 1 @1024 / FAIL 2 @2048 on fresh bakes — the
  final artifact rebake should carry all four lanes.
- **Determinism:** two identical fresh 1024 face bakes (film commit
  active, rescue disc firing): texture sha256 **byte-identical**.
- **Cross-mechanism interaction (film band x rescue):** measured inside
  one instrumented bake — the rescue disc (right eye, 1,950 texels) and
  the film COMMIT mask overlap in **0 texels** (731 texels of the wider
  pre-consensus band overlap the disc). Ordering is film-vacate (before
  blend) -> rescue (after completion): the rescue detector reads
  post-film weights, fired correctly, net result 1 fail. No pathological
  interaction found on this asset; the mechanisms partition the
  eye/hairline territory rather than fight over it.
- **Pose-lane observation (not mine, flagged):** c2d reports the free
  estimator REJECTING at 2048 (NCC 0.0052 < floor, pins az0) while my
  1024 bakes estimate az+20/el+8 at NCC 0.0152 every run — the free
  estimator's accept/reject flips with resolution on the same asset.
  The pose lane should own a resolution-consistency check before the
  final rebake relies on free estimation at 2048.

## Recommendation matrix

| mechanism | recommendation |
|---|---|
| A film-band commitment | **KEEP** (boundaries documented + regression-tested: sub-ball-scale bright sheen erasure; dark-majority no-op; 29% correlated-witness share) |
| B mirror twin rescue | **KEEP** (B2 explanation verified numerically; harness window-fragility documented for Critic 1; multi-disc source sequencing flagged for future) |
| C energy calibration + harness matting | **KEEP** (Goodhart attacks failed to break it; σ-guard one-sidedness documented; cache-key nit) |
| D shading compensation patch | **ADOPT via Critic 1** with ceiling-anchored budgets instead of absolute 15.0/24.0 |
| Final artifact rebake | REQUIRED to carry A+B onto the face bundle (on-disk face predates both); ship/owl already current |
