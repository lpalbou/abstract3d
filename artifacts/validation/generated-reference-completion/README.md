# Generated Reference Completion — Product-Path Proof (2026-07-09)

Single-photo texture coverage completed by synthesized views, produced
through the productized path (`rebake_bundle(generate_references=...)` /
Hunyuan `texture_reference_generation`), not hand-driven scripts. One
adversarial controller agent audited the design mid-build (15 findings,
3 blockers); every blocker and major is addressed in the shipped code or
documented below.

## Bundles (all at texture resolution 1024)

| bundle | angles generated | acceptance IoU | observed coverage | texture QA |
| --- | --- | --- | --- | --- |
| `owl/` | back, side_left, side_right, top | 0.92-0.98, first attempt | 0.30 -> **0.84** | 10/11 (see below) |
| `starship/` | bottom (0,-75), back | 0.765, 0.766 | 0.18 -> 0.29 | material + close gates pass |
| `face/` | back, side_left, side_right, top | 0.78-0.98 | 0.47 -> 0.67 | identity drill reviewed |

Review sheets: `owl_review_sheet.png` (source, generated views, bake vs
certified single-view), `owl_v3_vs_v5.png` (hardening iterations),
`ship_face_before_after.png` (certified vs generated-completion renders).

## The one open QA gate, honestly

`texel.fill_gradient_energy_ratio` reads 0.469 on the owl (gate: >= 0.5).
The gate compares synthesized-region detail against observed-region detail.
With generated references, the "synthesized" remainder is the ~16% of the
surface no view reaches (deep concavities, the base) while the observed
region now includes the busiest carved surfaces — the RATIO drops because
the denominator got richer, not because fill got worse (absolute fill
energy is comparable to the certified bake's). The earlier dark-smear
failures (82-140 fragments) were a real defect of the first product-path
bundles and are at ZERO after the hardening passes (subordinated weights +
single-view source semantics + despecular). The fill-energy gate reads
0.469 vs the 0.5 floor; recalibrating the gate for generated-completion
bundles requires its own A/B and is left open rather than tuned to pass.

## Honest limits (documented, by design)

- A generated view is plausible synthesis, not ground truth. Content on
  fully unobserved regions is invented from the mesh shape, the subject
  prompt, and the source photo's tone. The face's generated back shows
  plausible hair with tone variation the certified mirror-fill did not
  have; whether that is an improvement is subject-dependent, which is why
  the source photo always wins contested texels and the feature never
  fires without an explicit provider + subject hint ("auto") or explicit
  intent ("on").
- The three side views are generated independently: no cross-view content
  consistency beyond tone matching.
- The ship's generations sit near the IoU gate (0.765 vs 0.75): complex
  silhouettes are harder for i2i; rejected angles simply fall back to the
  certified fill.
- Tone matching is cap-limited (L +/-15); the ship's dark hull hit the cap
  (recorded `clipped: true` in metadata) so its generated underside reads
  lighter than the certified wash — auditable in each bundle's
  `generated_references.angles[].tone_match`.

## Reproduction

```bash
ABSTRACT3D_IMAGE_PROVIDER=mlx-gen \
ABSTRACT3D_IMAGE_MODEL=AbstractFramework/flux.2-klein-4b-8bit \
python - <<'EOF'
from abstract3d import bundle
bundle.rebake_bundle(
    "artifacts/validation/final-proof/hunyuan-owl",
    output_dir="out/owl-generated",
    generate_references="on",
    subject_hint="a carved ceramic owl figurine with warm cream and brown glaze",
    texture_resolution=1024,
)
EOF
```

Every bundle's `metadata.json` carries full provenance: resolved
provider/model, prompts, negative prompt, seeds, per-attempt IoU,
accepted-image hashes, clay renderer, and per-channel tone shifts.
