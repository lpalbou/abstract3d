# TripoSR texture upgrade proof

- View azimuth: `35.0`
- View elevation: `20.0`

| Case | Source | Regular mode | Improved mode | Preview renderer | Regular total (s) | Improved total (s) | Regular bytes | Improved bytes |
|---|---|---|---|---|---:|---:|---:|---:|
| i23d owl | `artifacts/validation/triposr-texture-comparison/owl/vertex_color/input.png` | `vertex_color` | `baked_basecolor@2048` | `moderngl` | 14.0933 | 31.5108 | 2,658,932 | 6,214,716 |
| i23d rocket | `artifacts/validation/triposr-texture-comparison/rocket/vertex_color/input.png` | `vertex_color` | `baked_basecolor@2048` | `moderngl` | 11.8820 | 15.7260 | 469,104 | 2,205,456 |
| t23d-derived chair | `artifacts/validation/triposr-texture-comparison/chair_t23d_source/vertex_color/input.png` | `vertex_color` | `baked_basecolor@2048` | `moderngl` | 11.9460 | 26.4171 | 2,274,792 | 5,127,716 |
