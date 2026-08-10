# e22 one-shot loop refusal — forensics (2026-07-21)

The first live one-shot `geometry_conditioning='loop'` run (e22,
`/tmp/laurent_front_clean4.png`, seed 2025 default, explicit
`--num-inference-steps 50 --mc-resolution 512`, klein + consistency LoRA +
identity conditioning) refused after ~80 minutes:

    geometry_conditioning='loop' produced no eligible conditioning views
    (no views were synthesized (see reference_generation report))

Nothing was persisted (the refusal path wrote no bundle — defect 1, fixed
this wave), so the diagnosis ran as a standalone probe
(`scripts/experimental/probe_e22_ladder.py`) that regenerates the pass-1
scaffold with e22's exact settings and drives the production ladder
(`generate_reference_views`, identity conditioning, e22's exact seeds
72025+) against it. Everything below is measured; probe artifacts in
`/tmp/e22_probe*` (session-local).

## 1. Which gate killed the views

**Silhouette IoU — the FIRST gate.** All side_left draws (seeds
72025/72026/72027, steps 8/12/12 — e22's exact ladder) died at:

| seed | registered IoU | threshold | registration |
|---|---|---|---|
| 72025 | 0.609 | 0.75 | scale 0.88, fy −0.04 |
| 72026 | 0.614 | 0.75 | scale 0.88, fy −0.04 |
| 72027 | 0.603 | 0.75 | scale 0.88 fx 0.02, fy −0.04 |

No draw ever reached the texture/material/speculars/pose gates
(`failure_family: "silhouette"` on every attempt row). The registration
search pinning at its 0.88 scale boundary is the tell: the draws were
trying to shrink onto a smaller, wrong silhouette.

## 2. Why: the pass-1 scaffold is garbage (root cause)

The e22 command's explicit 512/50 knobs reached the PASS-1 front-only
scaffold draw. Regenerated with e22's exact settings, the raw pass-1 mesh
decomposed into **295 components** (postprocess kept 6 bodies); its own
front clay registers onto the source photo it was reconstructed from at
IoU **0.415**. The 2mv-family regime (384/30) is WORSE front-only: 20
bodies, photo IoU **0.226** (floating debris islands). Reference points,
same instrument (register_matte_to_clay + silhouette_iou, production
defaults):

| mesh | photo-vs-front-clay IoU | bodies |
|---|---|---|
| e22 pass-1 scaffold (2mv front-only, 512/50, seed 2025) | 0.415 | 6 (raw 295 comps) |
| 2mv front-only, 384/30, seed 2025 | 0.226 | 20 |
| e20 champion (2mv + 3 conditioning views, 512/50) | 0.883 | 1 |
| synthetic control (disc photo vs sphere clay) | 0.925 | 1 |

Known-good draws confirm the silhouette gate is calibrated right: viewgen
bench arm E views measure 0.82–0.90 against the e20 clay but 0.50–0.67
against the shredded scaffold's clay. The gate honestly rejected views of
a man judged against debris-blob clays — 9 draws (~55 min of i2i), all
doomed before the first one started.

**The deeper finding: front-only 2mv was never validated.** Every
validated bust mesh in the record (e10–e21) conditioned the 2mv
checkpoint on ≥ 3 views (e21 "single_refs" included — its metadata shows
front+back+left tags). The loop automation introduced "scaffold from the
front photo alone (2mv, single front tag)" as a new, unmeasured step; the
2mv checkpoint is trained to fuse multiple tagged views, and a single-tag
dict appears to be out of distribution regardless of regime.

## 3. The compounding killer class: speculars on dark subjects (0019)

Even with a healthy scaffold, the BACK view class was measured
mass-rejected by `gate_baked_speculars` on this subject (dark hair/
beard/shirt): every plausible labeled back view fails the lightness-only
predicate — e11 back 0.0242, arm E back 0.0065, arm B back 0.0120 vs the
0.005 blob budget (5/5 historical rejections, backlog 0019). The
hot-pixel chroma distributions split the classes cleanly: lit-skin hot
pixels are chromatic (p10 17.9–26.5), true gloss cores near-white
(≤ ~5). Fix shipped: achromatic hot key (chroma < 14) + source
self-calibration; see `docs/backlog/completed/hunyuan3d/0019_*.md`
(including the honest limit found while validating: the e18 back's
fabricated head-mount was only ever caught by the same accident that
rejected every plausible back — content fabrication on pose-honest backs
is the strategy's C6/C8 channel class, still in the work queue; the e18
SIDE views reject on their real defect, decisive pose lies of 42.5/55°).

## 4. Pose acceptance gate: stricter than ratified

`POSE_ACCEPT_MAX_DELTA_DEG` was 15° while the RATIFIED rule
(strategy_v2 §stage contracts, stage E) is: refuse only when delta > 20°
AND decisive (iou_gap ≥ 0.10). Not e22's killer (no draw reached the pose
gate), but the in-ladder ruler measures against the pass-1 SCAFFOLD's
clays — a noisier instrument than the e20-class mesh the bench calibrated
on — so a stricter-than-ratified line compounds with instrument noise.
Aligned to the ratified two-key rule (one rule, both judges).

## 5. Fixes shipped this wave

1. **Refusal forensics persist** (`_persist_loop_refusal_artifacts` +
   wrapper): any loop failure after pass 1 writes `refusal_report.json`,
   `pass1_mesh.glb` + `pass1_clay_*.png`, `rejected_geometry_views/`
   (+ `raw/` full-res payloads) into the bundle dir, then re-raises.
2. **Progressive attempt log**: `<bundle>/loop_refgen_attempts.jsonl` —
   one JSON line per draw attempt as it completes
   (`generate_reference_views(on_attempt=)`), silhouette failures now
   keep pixel evidence too.
3. **Scaffold health fail-fast**: photo-vs-front-clay registered IoU
   floor 0.60 right after pass 1 (bands above); an e22-class scaffold now
   refuses in seconds with the numbers instead of after ~55 min of
   doomed draws.
4. **Pass-1 checkpoint fix**: see the addendum below (measured after the
   first draft of this note).
5. **Speculars 0019** + **pose 20°/0.10** recalibrations (sections 3–4).

## 6. Addendum: pass-1 checkpoint decision (measured)

Front-only scaffold candidates, same seed/subject, photo-vs-front-clay
registered IoU (health floor 0.60):

| candidate | IoU | bodies | verdict |
|---|---|---|---|
| 2mv front-tag, 512/50 | 0.415 | 6 (raw 295 comps) | shredded (e22's actual pass 1) |
| 2mv front-tag, 384/30 | 0.226 | 20 | worse — regime is not the axis |
| flagship 2.1 single-view, 512/50 | **0.776** | **1** (raw 3476 comps cleaned) | usable scaffold |

The axis is the CHECKPOINT, not the regime: the 2mv model fuses multiple
tagged views and a one-entry dict is outside its validated envelope in
both directions. DECISION SHIPPED: `_run_loop_pass1` loads the flagship
(`tencent/Hunyuan3D-2.1`) with the plain source image at the flagship's
own 512/50 regime, regardless of the run's model/knob choices (those
govern pass 2 — the 2mv windowed-set pass the e18-e20 recipe calibrated).
`pass1.model_id` is recorded in metadata; a knob divergence is warned.
Cost: one extra checkpoint load per loop run (~35 s; pass 1 already
cleared the runtime before the i2i pool loads, so peak memory is
unchanged). Probes: `scripts/experimental/probe_e22_ladder.py`
(`--pass1-steps/--pass1-octree/--pass1-only`) and
`scripts/experimental/probe_e22_pass1_flagship.py`.
