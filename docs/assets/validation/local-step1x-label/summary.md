# Abstract3D Local Validation

- Backend: `step1x`
- Platform: `macOS-26.3-arm64-arm-64bit`
- Python: `3.12.13`
- Device: `mps`
- Marching cubes resolution: `128`
- Image model for composed/object inputs: `AbstractFramework/flux.2-klein-4b-8bit`

## Aggregate

- Cases: `4` (`t23d=2`, `i23d=2`)
- Average total time: `82.2275` s
- Average inference time: `53.3108` s
- Average mesh extraction time: `25.2481` s
- Average preprocessing time: `1.2063` s
- Average text-to-image composition time: `4.9245` s
- Average RSS: `9.2823` GiB
- Average MPS allocated: `7.465` GiB
- Average vertices: `134551.25`
- Average faces: `197836.0`

## Generated Inputs

- `01_input.png`: `768x768`, `437739` bytes
- `02_input.png`: `768x768`, `755486` bytes

## Per Case

| Case | Mode | Total s | Image s | Prep s | Infer s | Mesh s | Vertices | Faces | RSS GiB | MPS GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 01_t23d | t23d | 46.7076 | 3.3999 | 1.0736 | 25.6641 | 16.57 | 122938 | 200000 | 9.0727 | 7.465 |
| 02_t23d | t23d | 69.9051 | 6.4492 | 0.5109 | 42.6911 | 20.2539 | 128969 | 191344 | 12.2729 | 7.465 |
| 03_i23d | i23d | 87.6378 | None | 1.382 | 57.4348 | 28.821 | 156416 | 200000 | 7.76 | 7.465 |
| 04_i23d | i23d | 124.6593 | None | 1.8586 | 87.4531 | 35.3476 | 129882 | 200000 | 8.0238 | 7.465 |
