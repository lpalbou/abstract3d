# Completed: G2 baked-speculars gate miscalibrated for dark-dominant subjects

## Metadata

- Created: 2026-07-20 (e11 back-view regeneration, `out/laurent-bust-redo/loop_views_e11`)
- Status: Completed
- Completed: 2026-07-21 (e22 refusal forensics wave)

## Original evidence (live run)

Regenerating a BACK reference view for a dark-haired, dark-shirted human
bust (`out/laurent-bust-redo/loop_views_e11`): 5/5 attempts across two
seed ladders failed `failure_family="speculars"` with silhouette IoU
0.71–0.82 (otherwise acceptable). With the gate bypassed (experiment
override), the same generation was accepted at IoU 0.778 and the resulting
view is visually clean — matte hair, matte shirt, no baked highlights
(`loop_view_back.png`).

## Root cause

`material_gates.gate_baked_speculars` flagged pixels with
`L > median(L) + 60` — lightness only. For a dark-dominant subject
(median L ≈ 15–25), legitimately lit skin (L ≈ 75–85) sits 60+ above the
median and forms blobs far larger than `max_blob_fraction=0.005`. The
predicate conflated dark-palette subjects with glossy highlights; by
construction the gate could NEVER fire on mid-tone subjects (median 55 →
threshold 115 > the L range), so it was structurally a dark-subject-only
rejector. Confirmed as a killer class in the e22 one-shot loop refusal
(2026-07-21).

## Fix shipped (2026-07-21)

The missing physical key: a baked specular is bright AND DESATURATED —
the same predicate `suppress_specular_highlights` already uses. Corpus
measurement (this subject's real labeled views): plausible dark bust
views' hot pixels are chromatic lit skin (chroma p10 17.9–26.5, p50
20.9–33.5); true gloss cores are near-white (chroma ≤ ~5). Two changes in
`gate_baked_speculars`:

1. **Achromatic key** (always on): hot pixels additionally require
   `chroma < 14` (splits the measured bands with margin on both sides).
   Effect: e11/arm-E/arm-B dark backs+profiles drop from worst blobs
   0.006–0.024 (all failing) to 0.00002–0.002 (passing) while synthetic
   2%-of-foreground gloss fields — pure white AND warm white — stay
   caught at 0.0197.
2. **Source self-calibration** (the item's proposed direction, when the
   caller has the source): the source photo's own worst blob under the
   SAME predicate is the subject's legitimate near-white floor (white
   lettering, bright trim); the pass line becomes
   `max(0.005, 2 × source_floor)`. A generation is only rejected for
   gloss the source itself does not justify. `generate_reference_views`
   and `evaluate_material_fidelity` both pass the source now.

The failure row records `median_lightness`, `specular_chroma_max`, and
`effective_max_blob_fraction` (plus `source_floor_blob_fraction` when
calibrated) so future miscalibrations are visible in bundle provenance
without a rerun — the item's observability ask.

## Validation

- Plausible dark views (e11 loop views ×3, viewgen-bench arm E backs +
  profiles, arm B back): all pass (previously the backs all failed).
- Synthetic gloss fields (2% of foreground; pure white and warm white)
  on a real dark back view: rejected.
- e18-era garbage views (`out/laurent-bust-redo/hunyuan_multiview/*`):
  measured against the e20 mesh with the ratified two-key pose rule, the
  side views reject on their REAL defect — pose lies (side_left measured
  +47.5 for declared +90, delta 42.5°, gap 0.186 decisive; side_right
  −35.0 for −90, delta 55°, gap 0.164) — which the pose acceptance gate
  now catches in-ladder. HONEST LIMIT: the e18 BACK (fabricated
  head-mount hardware; pose honest at 5°, row verdict "correctable") is
  no longer rejected by any current gate — the old speculars rejection of
  it (0.00532 vs 0.005, on CHROMATIC skin-toned hot pixels) was
  accidental: the same numbers rejected every plausible back too. A gate
  that fails good and bad views alike on one number is not
  discriminating. Content-fabrication on pose-honest backs is the C6/C8
  channel class in `docs/research/strategy_v2.md` §4 (work queue), not a
  speculars question; the current recipe (identity conditioning +
  consistency LoRA) was bench-measured fabrication-free on backs (arm E).
- Tests: `tests/test_material_gates.py::test_speculars_*` (4 tests pin
  the dark-subject pass, the gloss catches, the source calibration and
  the recorded numbers).
