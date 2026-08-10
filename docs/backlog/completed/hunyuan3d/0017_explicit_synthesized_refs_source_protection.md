# Completed: Source-photo texel protection for explicit synthesized references

## Metadata

- Created: 2026-07-20 (webcam-bust 10-experiment matrix, operator session)
- Status: Completed
- Completed: 2026-07-20

## Context

The 2026-07-20 experiment matrix proved the operator's recipe: i2i-synthesize
the unseen angles from the source photo, then feed them back as EXPLICIT
`--texture-reference-image/--texture-reference-angle` witnesses. Observed
coverage jumped 0.27 -> 0.73 and the back/side views became readable
(experiments e1/e2/e7/e8 vs the auto lane's completion-only contribution of
zero reference views).

The cost surfaced at high resolution: explicit references carry FULL paint
authority everywhere, including texels the source photo already observed.
Synthesized side views bled paint onto the front at grazing angles (chest
smudges, softer face) — e1/e2/e8 fronts are visibly worse than the photo-only
front, while their backs are far better. The auto reference-generation lane
already has the exact needed control (`protect_observed_texels` absolute
mode: generated content may never overwrite credibly photo-observed texels)
but the explicit lane does not apply it.

## Proposal

- Add a per-reference `synthesized` marker (CLI: e.g.
  `--texture-reference-synthesized`, or infer when the reference file carries
  the pipeline's own generation provenance) that applies the same
  protect-observed-texels lock the auto lane uses.
- Result combines the measured best of both: photo-authoritative front
  (e1-quality) + synthesized-witnessed back/sides (e2-quality).
- Default stays unchanged for real photos (full authority is correct for a
  genuine second photo).

## Evidence

- out/laurent-bust-redo/ experiment matrix + final_comparison_hires.png
- e2 metadata: coverage 0.7303, healthy, 1 body; front bleed visible at
  1024px renders (/tmp/e2_front.png vs /tmp/e1_front.png during the session)

## Completion report (2026-07-20)

Implemented exactly as proposed, reusing the auto lane's mechanism (no fork):
a synthesized-flagged explicit reference gets the per-view `generated` flag
the auto reference-generation lane sets, so it rides the identical in-bake
doctrine — weight subordination (x0.6), generated-class delight/tone gates,
and `protect_observed_texels(mode="absolute")` zeroing its weight wherever
any real view holds positive weight.

- Marker: `--texture-reference-synthesized true|false|auto` (option
  `texture_reference_synthesized`) pairs positionally with
  `--texture-reference-image`, mirroring `--texture-reference-angle`; loud
  errors on positional mismatch and unrecognized tokens.
- Inference: reference paths matching the pipeline's own generated naming
  (`geometry_view_synthesized_*`, `texture_reference_generated_*`) are
  treated as synthesized automatically; provenance recorded as
  `synthesized_source: filename_inference` plus a metadata note. Explicit
  flags (`explicit_flag`) always win, including `false` suppressing
  inference. Real photos keep full authority unchanged.
- Metadata: `texture_artifacts.reference_authority` (per shipped view:
  role, authority `full`/`protected_completion_only`, provenance, zeroed
  texel count) + `texture_artifacts.generated_protection` in both TripoSR
  and Hunyuan3D bundles.
- Scoping: caller-flagged synthesized references do NOT arm the auto lane's
  whole-bake A/B acceptance (operator's explicit witnesses; no doubled bake
  time) and are not re-persisted under `texture_reference_generated_*`.

Validation: `pytest -q tests/test_texturing.py tests/test_reference_flow.py
tests/test_cli.py` (124 passed, incl. new bake-level protection tests,
filename-inference/normalizer tests, CLI pairing tests) and
`pytest -q tests/test_triposr_backend_unit.py tests/test_hunyuan3d_backend_unit.py
tests/test_reference_generation.py` (128 passed, incl. the new
full-pipeline Hunyuan test: one bake, no A/B, authority metadata, no
generated-naming persistence for caller files).
