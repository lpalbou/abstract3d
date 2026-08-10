# 0022 — Promote v2 normal-render mesh metrics into the loop `quality_verdict`

## Status
Proposed (2026-07-21)

## Context
`--geometry-conditioning loop` (hunyuan3d21) self-verifies with oblique
raking closeups + the duplication-autocorrelation flag ported from
`scripts/bust_assessment.py`. That flag is the v1-era instrument: it also
fires on legitimate paired structure (lips/glasses) and runs on lit clay
renders. The v2 sweep replaced it with landmark-anchored coherent-ridge
energy + a second-crease duplication surcharge on NORMAL renders, and the
v2 instrument reproduced the operator's mesh verdicts on all 9 calibration
bundles (`docs/research/evaluation_methods_2026.md` §4), including the
striated-mouth class (e15/e17) that the v1 flag missed at headlight
lighting.

## Proposal
1. Extract `GeoRenderer` + `ridge_energy` + `dup_second_crease` +
   landmark anchoring from `scripts/model_quality_sweep_v2.py` into
   `src/abstract3d/mesh_quality.py` (script keeps thin CLI).
2. Feed `mesh_front_defect` / `mesh_oblique_defect` into the loop's
   `quality_verdict` with calibrated thresholds (good/bad boundary sits at
   ~45 front / ~37 oblique on the calibration subject; encode as
   configurable floors, re-verify on the next subject before trusting
   absolute values — ranking is what was validated, not the raw scale).
3. Record the scores + the normal-strip render path in bundle metadata so
   every future claim ships with the evaluation surface (operator
   directive: inputs ship with every claim).

## Dependencies
YuNet ONNX (232 KB, cached under `~/.cache/abstract3d_eval/`) with the
z-quantile fallback band when no face is detected — the loop must not hard-
fail on a detection miss; it should degrade to the fallback band with a
loud provenance note.

## Acceptance
- Loop runs on the e18-class recipe reproduce the v2 sweep scores within
  render-size noise.
- A deliberately striated candidate (e15 geometry) is refused by the
  verdict; the e20 champion passes.
- Unit tests cover: detection miss → fallback band; GL context failure →
  splat fallback; non-face subject → no crash, verdict marked
  low-confidence.
