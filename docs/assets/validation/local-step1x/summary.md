# Abstract3D Local Validation

- Backend: `step1x`
- Platform: `macOS-26.3-arm64-arm-64bit`
- Python: `3.12.13`
- Device: `mps`
- Marching cubes resolution: `128`
- Image model for composed/object inputs: `AbstractFramework/flux.2-klein-4b-8bit`

## Aggregate

- Cases: `4` (`t23d=2`, `i23d=2`)
- Average total time: `47.7379` s
- Average inference time: `42.5325` s
- Average mesh extraction time: `None` s
- Average preprocessing time: `None` s
- Average text-to-image composition time: `10.4107` s
- Average RSS: `2.3423` GiB
- Average MPS allocated: `6.7512` GiB
- Average vertices: `228019.5`
- Average faces: `342951.0`

## Generated Inputs

- `01_input.png`: `768x768`, `437739` bytes
- `02_input.png`: `768x768`, `755486` bytes

## Per Case

| Case | Mode | Total s | Image s | Prep s | Infer s | Mesh s | Vertices | Faces | RSS GiB | MPS GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 01_t23d | t23d | 35.4239 | 3.2218 | None | 32.2021 | None | 238699 | 341326 | 2.0448 | 6.7512 |
| 02_t23d | t23d | 74.637 | 17.5996 | None | 57.0374 | None | 233527 | 345340 | 2.2975 | 6.7512 |
| 03_i23d | i23d | 41.7664 | None | None | 41.7664 | None | 229545 | 352288 | 2.5088 | 6.7512 |
| 04_i23d | i23d | 39.1242 | None | None | 39.1242 | None | 210307 | 332850 | 2.5181 | 6.7512 |
