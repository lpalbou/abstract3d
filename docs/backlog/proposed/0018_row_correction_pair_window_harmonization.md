# 0018 — Row-corrected side views must share one crop window (pair-drop defect)

## Status
Proposed (2026-07-20)

## Evidence (live run)
`out/laurent-bust-redo/e12_full_pipeline_gated`: the row-consistency gate
(2026-07-20 wave) row-corrected all three freestanding synthesized views
(verdict "correctable", shift_frac 0.14–0.15). Both side views then PASSED
the material re-gate individually (`row_corrected_material.passed=true`)
but were dropped WITHOUT a recorded failure family, leaving conditioning
at front+back only. The resulting mesh split into 4 bodies and every
texture refgen angle failed registration (silhouette IoU 0.22–0.61),
ending in `quality_verdict: degraded`.

## Root cause
`align_view_rows` corrects each side view INDEPENDENTLY against the source
photo profile. Each correction picks its own crop window, so the two
corrected sides no longer share framing — and the side-PAIR mirror
consistency gate (left/right orthographic silhouettes must be exact
mirrors; both drop when they disagree, blame unattributable) then rejects
the pair that was individually consistent. The pair gate is right; the
per-view correction violating pair framing is the defect. A second defect:
the pair-drop path records no `failure` on the attempt rows (silent in
provenance; only visible by absence).

## Fix direction
1. After row correction of both sides, harmonize the pair to ONE window:
   intersect the two corrected windows (or re-run `align_view_rows` on the
   second side with the first side's window imposed) BEFORE the pair
   mirror check.
2. The pair-drop path must stamp `failure_family="side_pair_mirror"` on
   both attempt rows (persist-for-diagnosis doctrine).
3. Regression test: scripted candidates where both sides are correctable
   with different shifts must either both survive with a common window or
   both carry the recorded pair failure.

## Interaction note
The two-pass loop (condition on clay-registered views regenerated from a
first-pass mesh — validated e10/e11/e13) does not hit this path: loop
views are registered by `register_matte_to_clay` against one mesh and
share framing by construction. The row gate remains the safety for the
meshless first pass.
