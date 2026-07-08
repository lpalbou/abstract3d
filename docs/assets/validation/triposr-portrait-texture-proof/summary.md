# TripoSR Portrait Texture Completion Proof

This focused proof compares three textured TripoSR exports for the same front-view portrait input:

- baseline baked texture only
- baked texture plus `mirror_symmetry`
- baked texture plus `mirror_symmetry` and one synthetic `side_left` texture reference

Published proof sheet:

- [contact_sheet.png](contact_sheet.png)

Measured metadata:

| Variant | Projection mode | Observed coverage | Symmetry coverage | Output bytes |
|---|---|---:|---:|---:|
| baseline | `hybrid_observed_view_plus_triplane` | `0.2894` | `0.0` | `6,245,992` |
| mirror symmetry | `hybrid_observed_plus_symmetry_plus_triplane` | `0.2894` | `0.12` | `6,253,140` |
| mirror symmetry + side ref | `hybrid_multiview_plus_symmetry_plus_triplane` | `0.3644` | `0.12` | `6,389,920` |

Visual read:

- `mirror_symmetry` is the safest improvement for a single centered portrait because it fills some hidden front-side texels from the visible half without repainting the whole atlas.
- adding one auxiliary side reference increases texture coverage more than symmetry alone, but on this case the visible gain is still modest because the underlying TripoSR geometry remains the dominant limit.
- none of these variants turn TripoSR into a portrait-specialized head generator; the hidden-side texture can improve, but the far-side facial structure is still bounded by the reconstructed mesh.

Recommended usage:

```bash
abstract3d i23d ./portrait.png \
  --output-dir ./out/portrait \
  --backend triposr \
  --device mps \
  --mc-resolution 256 \
  --texture-mode baked_basecolor \
  --texture-completion mirror_symmetry
```

Add one curated auxiliary texture reference only when you already have a credible additional view:

```bash
abstract3d i23d ./portrait.png \
  --output-dir ./out/portrait-multiview \
  --backend triposr \
  --device mps \
  --mc-resolution 256 \
  --texture-mode baked_basecolor \
  --texture-completion mirror_symmetry \
  --texture-reference-image ./portrait-side-left.png \
  --texture-reference-angle side_left
```
