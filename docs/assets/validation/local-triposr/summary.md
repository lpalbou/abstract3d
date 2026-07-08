# Abstract3D Local Validation

- Platform: `macOS-26.3-arm64-arm-64bit`
- Python: `3.12.13`
- Device: `mps`
- Marching cubes resolution: `128`
- Image model for composed/object inputs: `AbstractFramework/flux.2-klein-4b-8bit`

## Aggregate

- Cases: `4` (`t23d=2`, `i23d=2`)
- Average total time: `6.3624` s
- Average inference time: `0.4029` s
- Average mesh extraction time: `2.2017` s
- Average preprocessing time: `0.0047` s
- Average text-to-image composition time: `7.5063` s
- Average RSS: `17.9945` GiB
- Average MPS allocated: `1.5649` GiB
- Average vertices: `10742.0`
- Average faces: `21400.0`

## Generated Inputs

- `01_input.png`: `768x768`, `437739` bytes
- `02_input.png`: `768x768`, `755486` bytes

## Per Case

| Case | Mode | Total s | Image s | Prep s | Infer s | Mesh s | Vertices | Faces | RSS GiB | MPS GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 01_t23d | t23d | 10.4663 | 7.2816 | 0.0028 | 0.5091 | 2.6728 | 11555 | 23012 | 15.4359 | 1.5649 |
| 02_t23d | t23d | 11.0053 | 7.731 | 0.0022 | 0.5099 | 2.7622 | 8003 | 15920 | 18.8197 | 1.5649 |
| 03_i23d | i23d | 1.9036 | None | 0.0061 | 0.2924 | 1.6051 | 13448 | 26800 | 18.8612 | 1.5649 |
| 04_i23d | i23d | 2.0745 | None | 0.0076 | 0.3002 | 1.7667 | 9962 | 19868 | 18.8613 | 1.5649 |
