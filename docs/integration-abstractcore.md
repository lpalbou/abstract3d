# AbstractCore Integration

## Install

Install `abstract3d` into the same environment as `abstractcore`.

Base install:

```bash
pip install abstract3d
pip install "abstractcore[scene3d]"
```

That profile includes the lightweight `abstractvision` package contract for provider-neutral composed
`t23d`. Add local runtime extras only when the same environment should run the image and 3D models
in-process.

Validated TripoSR path:

```bash
pip install "abstract3d[triposr]"
pip install "abstractcore[scene3d]"
```

Experimental Step1X geometry path:

```bash
pip install "abstract3d[step1x]"
pip install "abstractcore[scene3d]"
```

If you also want the composed Apple-local `t23d` path:

```bash
pip install "abstract3d[apple]"
```

For Linux/Windows GPU hosts, use:

```bash
pip install "abstract3d[gpu]"
```

Compatibility alias for callers that still request the historical composed `t23d` extra:

```bash
pip install "abstract3d[t23d]"
```

## Capability Discovery

Abstract3D registers itself through the `abstractcore.capabilities_plugins` entry-point group.

The exposed backend ids are:

- `abstract3d:triposr`
- `abstract3d:step1x-local`
- `abstract3d:trellis2-local`

The public provider aliases accepted by `abstractcore` routing are:

- `triposr`
- `step1x`
- `trellis2`

The license-gated Hunyuan3D-2.1 backend is intentionally not registered with the
AbstractCore plugin yet: host-side routing would make it too easy to trigger a
license-restricted download from a generic capability call. It remains available through the
direct `Scene3DManager` API and the CLI, both of which enforce the explicit license
acknowledgment.

## Direct Capability Calls

```python
from abstractcore.providers.base import BaseProvider
from abstractcore.core.types import GenerateResponse

class DemoProvider(BaseProvider):
    def __init__(self):
        super().__init__(model="demo")
        self.provider = "demo"
    def _generate_internal(self, prompt, messages=None, system_prompt=None, tools=None, media=None, stream=False, **kwargs):
        return GenerateResponse(content=prompt, model=self.model)
    def get_capabilities(self):
        return []
    def unload_model(self, model_name: str) -> None:
        return None
    def list_available_models(self, **kwargs):
        return [self.model]

llm = DemoProvider()

mesh_default = llm.scene3d.i23d("./object.png", format="glb")
mesh_step1x = llm.scene3d.i23d(
    "./object.png",
    provider="step1x",
    model="stepfun-ai/Step1X-3D",
    format="glb",
)
```

## Unified `generate(...)` Output Routing

Text to 3D:

```python
resp = llm.generate(
    text="A glossy red cube.",
    output={"modality": "scene3d", "provider": "triposr", "format": "glb"},
)
```

Image to 3D with Step1X:

```python
resp = llm.generate(
    text="Turn this object into a mesh.",
    media={"type": "image", "path": "./object.png", "role": "source"},
    output={
        "modality": "scene3d",
        "provider": "step1x",
        "model": "stepfun-ai/Step1X-3D",
        "task": "image_to_scene3d",
        "format": "glb",
    },
)
```

## Residency Routing

The `abstractcore` control-plane task aliases accept:

- `scene3d`
- `scene3d_generation`
- `text_to_scene3d`
- `t23d`
- `image_to_scene3d`
- `i23d`

That means server-side residency routes such as `/acore/models/load`, `/acore/models/loaded`, and `/acore/models/unload` can operate on `scene3d` backends the same way they already operate on voice, audio, vision, and music backends.

## Current Boundary

- TripoSR remains the validated default path.
- Step1X is available as a local experimental geometry-only backend.
- TRELLIS.2 remains implemented but is not part of the permissive validated path.
- Composed `t23d` stays provider-neutral: `abstract3d` can forward explicit `image_provider` / `image_model`, or read `scene3d_image_provider` / `scene3d_image_model` and `ABSTRACT3D_IMAGE_PROVIDER` / `ABSTRACT3D_IMAGE_MODEL`, but otherwise falls through to the configured `abstractvision` default.

For request-scoped backend selection, `scene3d` uses the `provider` field with values such as `triposr`, `step1x`, `trellis2`, `abstract3d:triposr`, or `abstract3d:step1x-local`.
