# FAQ

## Is `t23d` a native text-to-3D model today?

No. The supported paths in this repo are composed:

1. text to image in `abstractvision`
2. image to 3D in the selected local backend

That is true for both the validated TripoSR path and the experimental Step1X geometry path.

## Why is TripoSR still the validated default if Step1X is implemented?

Because the checked Apple-local proof cases favored TripoSR.

Step1X now exists as a real local backend, but on the checked benchmark profile it was slower, used more MPS memory, and produced less recognizable shapes than TripoSR on the same four validation cases.

## Does Step1X provide native text-to-3D in this repo?

No. The supported Step1X backend uses only the official geometry checkpoint, and that path is image-conditioned. `t23d` remains composed through `abstractvision`.

## Does this repo support textured Step1X output?

No. The supported Step1X backend is geometry-only.

The full upstream texture stage is intentionally out of scope here because it widens the dependency and runtime surface beyond the backend boundary accepted in this repository.

## What does Abstract3D generate well?

Single centered objects with clean silhouettes.

Examples that work best:

- product-style photos
- figurines
- furniture
- appliances

## What does it not do well yet?

- cluttered scenes
- full rooms
- many interacting objects
- strong self-occlusion or missing silhouettes

## What formats are supported?

- `glb`
- `obj`
- `zip` bundles

`glb` is the primary contract.

## Can I use it through AbstractCore?

Yes. Install `abstract3d` into the same environment as `abstractcore`, then call `llm.scene3d.*` directly or use `generate(..., output={"modality": "scene3d"})`.

See [AbstractCore integration](integration-abstractcore.md).

## Does it require CUDA?

Not for the validated TripoSR path or the experimental Step1X geometry path in this repo.

TripoSR is validated on Apple Silicon with MPS. The Step1X geometry backend also runs locally on Apple Silicon with MPS, using `float32` for stability, but it remains experimental.

## Does official TRELLIS.2 ship an `8-bit` or `fp8` checkpoint?

Not in the official model tree referenced by this package as of `2026-06-22`. The published TRELLIS.2 files are `bf16` and `fp16`.
