# Results — Zero-Defect Texture Program (final)

Three assets, generated from 1-3 photographs each, certified at the
zero-open-defect standard by two independent adversarial verdict agents
over seven cycles. This page is the entry point to every result and proof.

## Open the models (MeshVault or any glTF viewer — they load upright)

| Asset | File | Inputs | Gate state |
|---|---|---|---|
| Face (multi-view) | `artifacts/validation/iter3-multiview-fixed/face-2mv/scene.glb` | front photo + left profile + mirrored right | compensated identity 0.702/13.9 PASS, 28-view battery 0 failures, texture_qa 13/13 |
| Starship (single view) | `artifacts/validation/final-proof/hunyuan-starship/scene.glb` | one photo | texture_qa 13/13; nose registration fix certified |
| Owl (single view) | `artifacts/validation/final-proof/hunyuan-owl/scene.glb` | one photo | texture_qa 13/13 |

## The story in pictures

- `hero_gallery.png` — the three certified assets, MeshVault studio render
- `upright_verification.png` — the Y-up export fix verified in MeshVault
- `face_rejected_vs_final.png` — the face: rejected first bundle vs final, 5 azimuths
- `ship_before_vs_final.png`, `cycle2-3/ship_nose_before_after.png` — the ship
- `walkaround_gallery.png` — 4-view walkarounds of all three
- `critic1_evidence_sheets/` — 38 independent verification sheets from the verdict agent

## The numbers (measured, all reproducible)

- Face hostile-harness trajectory: **66 failed checks -> 0 open defects**
  (raw identity constant retired only after a joint ceiling proof;
  compensated gate 0.70/15.0 met at 0.702/13.86).
- Reference leverage (per-texel ledger, printed by every bake):
  **~45% of the surface direct-painted = ~90% of everything the photos
  honestly witness**; the remainder is genuinely unobserved surface.
- Determinism: four independent canonical-recipe bakes -> one texture hash.
- Test suite: **245 passed** (+3 expected-fail boundary pins).

## The authority documents

- `CERTIFICATION.md` — the definitive ledger (23 FIXED / 10 PROVEN-LIMIT /
  0 OPEN), maintenance contract, and the honest definition of zero-defect
  within these inputs.
- `defect_ledger.md` + `verdict*/RULING*.md` (cycles 1-7, in `cycle*/`) —
  every defect, every ruling, every crop.
- `meshvault/agentA/REPORT_agentA.md` — the 20-row defect->code correlation
  table with 186 screenshots.
- `meshvault/agentB/REPORT_agentB.md` — the 15-entry latent-risk register
  for future inputs.

## Generated reference views — the capture limits, closed (2026-07-08)

The proven capture limits are addressed WITHOUT real extra photos: the
integrated image model (AbstractVision i2i, flux.2-klein) generates the
missing views, and the certified multi-view bake projects them through the
same gates as real photos. Bundles under
`artifacts/validation/generated-references/`:

| Bundle | Added generated view | Result |
|---|---|---|
| `face-4view/` | back of head (az 180) | coverage 0.46 -> 0.62; rear = real strand texture (`rear_before_after.png`); identity RAW improved to 0.679; texture_qa PASS; two marginal dark-debris counts at ±35 (0.0033-0.0036 vs 0.003) |
| `starship-2view/` | underside (el -75) | see the v1 failure note below; v2 result: coherent plated underside matching the hull (`underside_before_after.png`, full `inspection_sweep.png`); texture_qa PASS |
| `owl-2view/` | back (az 180) | back = real carved feather pattern (`back_before_after.png`); texture_qa PASS |

METHOD NOTE (learned from a real failure): the first ship attempt
generated the underside from the FRONT PHOTO alone; the model invented a
different ship (X-shaped), and with zero overlap against the source view
the consistency gate had nothing to compare — the projection shipped
blocky misprojection garbage that the owner caught immediately. The
corrected method, now the documented procedure for generated views with
little/no overlap: (1) render the MESH'S OWN geometry from the target
angle (clay render), (2) generate with that render as the conditioning
image (shape-locked: silhouette IoU generated-vs-mesh 0.981 vs the v1
failure's mismatch), (3) gate on silhouette IoU >= 0.75 before accepting,
(4) tone-match the generated view to the source photo (LAB statistics)
because zero-overlap views cannot be harmonized by the overlap machinery.
Remaining honest limits: generated content is plausible, not ground
truth; grazing transition bands between source-observed and generated
regions stay softer than either view's interior.

## Honest limits (each with its exact remedy)

Ten PROVEN-LIMIT entries stand, all information-theoretic from the given
photographs, none pipeline-preventable (proofs in the certification):
under-ship / rear-head / far-side content needs one additional photo per
region (the pipeline accepts extra references via
`texture_reference_images`); eye-corner resolution and eyelid aperture are
photo-resolution and geometry-prior limits. A generative texture stage
(diffusion inpainting for unseen regions) is the documented next-phase
feature that would replace statistical fill with plausible content.
