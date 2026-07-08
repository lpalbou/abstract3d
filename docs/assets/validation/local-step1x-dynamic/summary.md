# Abstract3D Local Validation

- Backend: `step1x`
- Platform: `macOS-26.3-arm64-arm-64bit`
- Python: `3.12.13`
- Device: `mps`
- Marching cubes resolution: `128`
- Image model for composed/object inputs: `AbstractFramework/flux.2-klein-4b-8bit`

## Aggregate

- Cases: `4` (`t23d=2`, `i23d=2`)
- Successful cases: `4`
- Failed cases: `0`
- Average total time: `101.8601` s
- Average inference time: `55.1116` s
- Average mesh extraction time: `38.4166` s
- Average preprocessing time: `6.2733` s
- Average text-to-image composition time: `4.1172` s
- Average RSS: `3.2993` GiB
- Average MPS allocated: `0.0008` GiB
- Average vertices: `57146.0`
- Average faces: `104932.5`

## Generated Inputs

- `01_input.png`: `768x768`, `437739` bytes
- `02_input.png`: `768x768`, `755486` bytes

## Per Case

| Case | Mode | Status | Total s | Image s | Prep s | Infer s | Mesh s | Vertices | Faces | RSS GiB | MPS GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 01_t23d | t23d | succeeded | 83.298 | 4.0509 | 5.8002 | 34.2857 | 39.1612 | 47536 | 89010 | 4.9253 | 0.001 |
| 02_t23d | t23d | succeeded | 111.2721 | 4.1835 | 5.9208 | 59.2726 | 41.8952 | 74398 | 132484 | 4.9484 | 0.0006 |
| 03_i23d | i23d | succeeded | 121.4867 | None | 6.7083 | 74.6448 | 40.1336 | 63516 | 114178 | 1.6578 | 0.0005 |
| 04_i23d | i23d | succeeded | 91.3835 | None | 6.6639 | 52.2431 | 32.4765 | 43134 | 84058 | 1.6658 | 0.001 |
