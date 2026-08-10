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
- `abstract3d:hunyuan3d21-local`
- `abstract3d:trellis2-local`

The public provider aliases accepted by `abstractcore` routing are:

- `triposr`
- `step1x`
- `hunyuan3d21` (also `hunyuan3d`)
- `trellis2`

The license-gated Hunyuan3D-2.1 backend registers with the plugin but keeps its
explicit license gate: the backend constructor is inert, and any download or
generation refuses loudly until the operator opts in with
`ABSTRACT3D_HUNYUAN_ACCEPT_LICENSE=1` or `scene3d_hunyuan_license_accepted=true`
(the Tencent Hunyuan Community License excludes the EU, UK, and South Korea).
The plugin's `config_hint` carries the acknowledgment sentence so the registry
error is self-explanatory.

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

## Server Endpoint (OpenAI-Compatible Surface)

The AbstractCore server exposes 3D generation as an extension endpoint
(OpenAI has no 3D API; the shape follows `/v1/audio/music`):

```bash
# Text -> 3D
curl -X POST http://127.0.0.1:8000/v1/scene3d/generations \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a ceramic teapot", "provider": "triposr", "format": "glb"}' \
  --output teapot.glb

# Image -> 3D (base64 source image)
curl -X POST http://127.0.0.1:8000/v1/scene3d/generations \
  -H "Content-Type: application/json" \
  -d "{\"image_b64\": \"$(base64 -i object.png)\", \"provider\": \"triposr\"}" \
  --output object.glb
```

- Returns raw model bytes (`model/gltf-binary`, `model/obj`, or
  `application/zip`) with provenance headers `X-AbstractCore-Backend-Id`,
  `X-AbstractCore-Model`, and `X-AbstractCore-Task`.
- `POST /{provider}/v1/scene3d/generations` is the provider-scoped alias.
- Unknown body fields return 422 naming the supported option list;
  `output_dir` and other host-control options are not forwardable over HTTP.
- Without a registered scene3d plugin the endpoint returns 501 with an
  install hint; a license-gated refusal returns 403.
- Discovery rides the generic capability routes:
  `GET /v1/capabilities/scene3d/providers` and `/v1/capabilities/scene3d/models`.

## AI Tools

`abstract3d.tools` ships eight tools an AI/agent can call (JSON-string
outputs with explicit `success` markers): `generate_3d_object`,
`analyze_3d_object`, `transform_3d_object`, `compose_3d_scene`,
`convert_3d_object`, `repair_3d_object`, `render_3d_preview`, and
`list_3d_backends`.

Tools follow AbstractCore's explicit-import contract (there is deliberately
no entry-point auto-registration for tools — they are a security surface):

```python
from abstract3d.tools import abstract3d_tools, abstract3d_tool_definitions, register_tools

# Pass directly to generate():
resp = llm.generate("Analyze ./scene.glb", tools=abstract3d_tools())

# Or register into a registry you own:
from abstractcore.tools.registry import ToolRegistry
registry = ToolRegistry()
register_tools(registry)
```

`SCENE3D_TOOL_CLASSIFICATION` carries the per-tool classification
(`mutating`, `remote_write_capable`, `downloads_model_weights`) in
AbstractCore's inventory vocabulary so approval layers can gate consistently.
Mesh manipulation/analysis needs the lightweight `abstract3d[mesh]` extra;
generation needs a backend extra (for example `abstract3d[triposr]`).

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
- Hunyuan3D-2.1 is registered but license-gated (explicit acknowledgment required; see above).
- TRELLIS.2 remains implemented but is not part of the permissive validated path.
- Composed `t23d` stays provider-neutral: `abstract3d` can forward explicit `image_provider` / `image_model`, or read `scene3d_image_provider` / `scene3d_image_model` and `ABSTRACT3D_IMAGE_PROVIDER` / `ABSTRACT3D_IMAGE_MODEL`, but otherwise falls through to the configured `abstractvision` default.

For request-scoped backend selection, `scene3d` uses the `provider` field with values such as `triposr`, `step1x`, `hunyuan3d21`, `trellis2`, `abstract3d:triposr`, or `abstract3d:hunyuan3d21-local`.
