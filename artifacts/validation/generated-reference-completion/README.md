# Generated Reference Completion — Product-Path Proof (2026-07-09, v2)

Single-photo texture coverage completed by synthesized views through the
productized path. **v2 replaces the first proof set after a coherence
audit**: the initial clay-conditioned generations were shape-correct but
only partially faithful to the source's materials (the owl came back pale
cream instead of its warm carved brown). The root cause: the i2i model
never saw the source photo — only an untextured clay render plus a text
prompt.

## The fix: composite conditioning (source + clay in one canvas)

The conditioning image now pairs the SOURCE PHOTO (left panel) with the
clay render of the target angle (right panel), and the instruction is a
texture transfer: "repaint the right image with the exact materials of the
left object." Measured on the owl back view with the SAME model
(FLUX.2-klein-4B 8-bit, local mlx-gen):

| metric | clay-conditioned (v1) | composite (v2) |
| --- | --- | --- |
| foreground LAB distance to source | 28.4 | **7.4** |
| hue-histogram correlation | 0.44 | **0.82** |
| silhouette IoU (after registration) | 0.92 | 0.92 |

A new similarity registration (`register_matte_to_clay`) absorbs the
model's small reframing before the IoU gate, so composite conditioning
keeps the shape lock. See `strategy_compare.png` for the side-by-side and
`owl_coherence_final.png` for certified vs v1 vs v2 full turnarounds.

## Bundles (texture resolution 1024, all texture-QA PASS)

| bundle | angles | IoU | coverage | texture QA |
| --- | --- | --- | --- | --- |
| `owl/` | back, side_left, side_right, top | 0.89-0.92 | 0.30 -> **0.85** | **PASS** (`owl_texture_qa.json`) |
| `starship/` | bottom (0,-75), back | 0.90, 0.95 | 0.18 -> 0.45 | **PASS** |
| `face/` | back, side_left, side_right, top | 0.80-0.97 | 0.47 -> 0.66 | **PASS** |

The fill-energy gate that was open in v1 passes in v2 (the coherent
generated content carries comparable gradient energy to the observed
region).

## Model findings (candidates the user asked to try)

- **FLUX.2-klein-4B 8-bit + composite conditioning** — the shipped
  default: coherence doubled with no new model, ~140 s/angle on this host.
- **FLUX.2-klein-9B 8-bit** — repo is auto-gated on Hugging Face and the
  locally stored HF token has expired: `huggingface-cli login` with a fresh
  token, then `abstractvision download AbstractFramework/flux.2-klein-9b-8bit`.
  Untested until then.
- **Qwen-Image-Edit-2511 8-bit** (30 GB) — downloaded and registered, but
  generation on this host did not produce a first denoise step within 8
  minutes (20B model; two orphaned attempts also destabilized concurrent
  MPS work — one segfaulted a running bake). Parked: usable via
  `ABSTRACT3D_IMAGE_MODEL=AbstractFramework/qwen-image-edit-2511-8bit`
  when its runtime path matures; camera-rotation conditioning ("rotate"
  strategy, implemented) is the natural fit for it because instruction
  following, not silhouette lock, is its strength.

## Honest limits

- A generated view is plausible synthesis, not ground truth; the source
  photo always wins contested texels, and the feature only fires with an
  explicitly configured provider + subject hint ("auto") or explicit "on".
- Side views are generated independently (tone-matched, not
  content-synchronized).
- The face's composite back is closer to the source hair than v1 but
  reads slightly glossier than the certified two-real-profile bake — real
  reference photos remain strictly better than synthesis when available.

Provenance per bundle in `metadata.json`: conditioning strategy, resolved
provider/model, prompts, negative prompt, seeds, per-attempt IoU +
registration, accepted-image hashes, clay renderer, tone shifts. Generated
photos and clay conditions ship alongside (`generated_*.png`).
