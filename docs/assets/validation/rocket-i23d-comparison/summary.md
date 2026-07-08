# Rocket `i23d` Comparison

- Input image: `/Users/albou/Desktop/rocket.png`
- Comparison sheet: `comparison_contact_sheet.png`

## TripoSR

- Model: `stabilityai/TripoSR`
- Bundle: `/Users/albou/tmp/abstractframework/abstract3d/artifacts/validation/rocket-i23d/triposr/01_i23d`
- GLB: `/Users/albou/tmp/abstractframework/abstract3d/artifacts/validation/rocket-i23d/triposr/01_i23d/scene.glb`
- Contact sheet: `triposr_contact_sheet.png`
- Vertices: `3293`
- Faces: `6568`
- Total time: `2.8472` s
- Inference time: `0.3576` s
- Mesh time: `1.2093` s
- RSS after generation: `3409313792` bytes
- MPS allocated after generation: `1680249088` bytes

## Step1X

- Model: `stepfun-ai/Step1X-3D`
- Geometry checkpoint: `Step1X-3D-Geometry-Label-1300m`
- Bundle: `/Users/albou/tmp/abstractframework/abstract3d/artifacts/validation/rocket-i23d/step1x-stable/01_i23d`
- GLB: `/Users/albou/tmp/abstractframework/abstract3d/artifacts/validation/rocket-i23d/step1x-stable/01_i23d/scene.glb`
- Contact sheet: `step1x_contact_sheet.png`
- Vertices: `106117`
- Faces: `127734`
- Total time: `101.4467` s
- Inference time: `50.2726` s
- Mesh time: `43.5793` s
- RSS after generation: `1812037632` bytes
- MPS allocated after generation: `1048576` bytes
- Label override used for this stable Apple-local run: `{'geometry_type': 'sharp', 'symmetry': 'x'}`

## Stronger Step1X Probes

- `label_14step_4p5`: failed with Apple `mps` out-of-memory, result log at `/Users/albou/tmp/abstractframework/abstract3d/artifacts/validation/rocket-i23d/step1x/result.json`
- `base_8step_3p0`: failed with Apple `mps` out-of-memory, result log at `/Users/albou/tmp/abstractframework/abstract3d/artifacts/validation/rocket-i23d/step1x-base-stable/result.json`
