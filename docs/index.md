# Abstract3D

Local-first 3D generation for the AbstractFramework ecosystem.

`abstract3d` extends `abstractcore` with a first-class `scene3d` capability:

- `image_to_scene3d` (`i23d`) — one centered subject photo in, textured `glb` out
- `text_to_scene3d` (`t23d`) — composed text-to-image-to-3D through `abstractvision`

## Current State (v0.2.0)

Development status: **Alpha**, object-centric generation only.

- **Validated platform**: macOS on Apple Silicon (`mps`). The full proof and
  certification record was produced on this profile. Linux/Windows GPU installs are
  implemented (`abstract3d[gpu]`) but not yet validated by a checked proof run.
- **Validated default backend**: `stabilityai/TripoSR` (fast, permissive license).
- **Strongest checked geometry**: `tencent/Hunyuan3D-2.1` and multi-view
  `Hunyuan3D-2mv` — experimental, license-gated (the Tencent Hunyuan Community License
  excludes the EU, UK, and South Korea; explicit operator acknowledgment required).
- **Experimental**: `stepfun-ai/Step1X-3D` geometry; TRELLIS.2 is accepted but blocked
  at runtime on gated DINOv3 credentials.
- **Texture pipeline**: the shared projection bake closed a six-cycle adversarial
  zero-defect program (23 defects fixed, 10 proven capture limits, 0 open), covering
  canonical-frame orthographic projection, strict first-surface visibility, crop-aware
  registration, multi-view conflict resolution, gradient-domain compositing, mirror
  completion, harmonic fill, and optional generated reference views.

## Where To Start

- [Getting started](getting-started.md) — install profiles and first successful runs
- [Models](models.md) — backend catalog, licenses, and validation status
- [Benchmarks](benchmarks.md) — checked proof lanes and certification results
- [API and CLI](api.md) — `Scene3DManager` and the `abstract3d` command
- [Architecture](architecture.md) and [Methodology](methodology.md)

## Project

- [Repository](https://github.com/lpalbou/abstract3d)
- [Changelog](https://github.com/lpalbou/abstract3d/blob/main/CHANGELOG.md)
- [License (MIT)](https://github.com/lpalbou/abstract3d/blob/main/LICENSE)
