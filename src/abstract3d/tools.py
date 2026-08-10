"""AI-facing tools: generate, analyze, transform, compose, convert, repair, preview.

These functions wrap Abstract3D's generation and mesh-operation surfaces in a
shape designed for LLM tool calling:

- every tool returns a JSON string with an explicit ``success`` field, so
  AbstractCore's tool registry can classify failures without exceptions
  leaking model-hostile tracebacks;
- parameters are JSON-simple (strings, numbers, lists, objects);
- module import stays light — model runtimes and mesh dependencies load at
  call time with actionable install hints on failure.

When AbstractCore is installed, each tool carries a full ``ToolDefinition``
(via ``@abstractcore.tools.tool``) and :func:`register_tools` can install the
set into a ``ToolRegistry`` in one call. Without AbstractCore the same
callables still work as plain functions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .errors import Abstract3DError

_TOOL_TAGS = ["3d", "scene3d", "abstract3d"]


def _tool_decorator(**meta: Any) -> Callable[[Callable[..., str]], Callable[..., str]]:
    """Return abstractcore's @tool decorator when available, else a no-op."""
    try:
        from abstractcore.tools import tool as abstractcore_tool  # noqa: PLC0415
    except Exception:
        def _noop(f: Callable[..., str]) -> Callable[..., str]:
            return f

        return _noop
    return abstractcore_tool(**meta)


def _ok(**payload: Any) -> str:
    out: Dict[str, Any] = {"success": True}
    out.update(payload)
    return json.dumps(out, sort_keys=True, default=str)


def _fail(error: Exception | str) -> str:
    return json.dumps({"success": False, "error": str(error)}, default=str)


def _json_safe_metadata(metadata: Any, *, keys: tuple[str, ...]) -> Dict[str, Any]:
    if not isinstance(metadata, dict):
        return {}
    out: Dict[str, Any] = {}
    for key in keys:
        value = metadata.get(key)
        # Absent facts stay absent — a literal null (e.g. "seed": null on a
        # feed-forward backend) is noise in an LLM context, not information.
        if isinstance(value, (str, int, float, bool)):
            out[key] = value
    return out


_FALSE_STRINGS = {"false", "0", "no", "off", ""}
_TRUE_STRINGS = {"true", "1", "yes", "on"}


def _coerce_bool(value: Any, *, field: str) -> bool:
    """Coerce tool-call booleans that arrive as strings.

    Several tool-call wire formats preserve raw strings, and Python's
    truthiness would silently invert intent (`bool("false") is True` —
    the framework's 2026-02-20 coercion-risk class). Unrecognized strings
    fail loudly instead of guessing.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in _FALSE_STRINGS:
            return False
        if text in _TRUE_STRINGS:
            return True
    raise ValueError(f"{field} expects a boolean (true/false), got {value!r}")


@_tool_decorator(
    name="generate_3d_object",
    description=(
        "Generate a 3D object (GLB by default) from a text prompt or a source image, "
        "writing a bundle (mesh + previews + metadata) under output_dir."
    ),
    tags=_TOOL_TAGS,
    when_to_use=(
        "When the user asks to create a 3D object from a description or photo. image_path = "
        "image-to-3D, else composed text-to-3D. Use image_seed for reproducibility; `seed` only "
        "works on sampling backends (hunyuan3d21/step1x), never triposr."
    ),
    examples=[
        {
            "description": "Text to 3D",
            "arguments": {"prompt": "a ceramic teapot with a curved spout", "output_dir": "./out/teapot"},
        },
        {
            "description": "Image to 3D",
            "arguments": {"image_path": "./owl.png", "output_dir": "./out/owl"},
        },
    ],
)
def generate_3d_object(
    output_dir: str,
    prompt: Optional[str] = None,
    image_path: Optional[str] = None,
    backend: Optional[str] = None,
    model: Optional[str] = None,
    format: str = "glb",
    device: Optional[str] = None,
    seed: Optional[int] = None,
    image_seed: Optional[int] = None,
) -> str:
    """Generate a 3D object from text (composed t23d) or an image (i23d)."""
    try:
        from .scene3d_manager import Scene3DManager  # noqa: PLC0415

        if not str(output_dir or "").strip():
            return _fail("output_dir is required")
        if not (prompt or image_path):
            return _fail("Provide a prompt (text-to-3D) or an image_path (image-to-3D).")

        manager = Scene3DManager(backend_id=backend)
        fmt = str(format or "glb").strip().lower() or "glb"
        kwargs: Dict[str, Any] = {"output_dir": str(output_dir), "format": fmt}
        if model:
            kwargs["model"] = str(model)
        if device:
            kwargs["device"] = str(device)
        if seed is not None:
            # Sampler seed: only backends that sample accept it; feed-forward
            # backends (triposr) refuse it loudly in milliseconds (preflight).
            kwargs["seed"] = int(seed)
        if image_seed is not None and not image_path:
            kwargs["image_seed"] = int(image_seed)

        if image_path:
            source = Path(str(image_path)).expanduser()
            if not source.is_file():
                return _fail(f"image_path not found: {source}")
            result = manager.i23d(str(source), prompt=(str(prompt) if prompt else None), **kwargs)
        else:
            result = manager.t23d(str(prompt), **kwargs)

        metadata = result.get("metadata") if isinstance(result, dict) else {}
        bundle_dir = None
        if isinstance(metadata, dict):
            raw_bundle = metadata.get("bundle_dir") or metadata.get("output_dir")
            if raw_bundle:
                bundle_dir = Path(str(raw_bundle)).expanduser()

        # Primary-path contract (no filesystem guessing): the returned bytes
        # ARE the primary. Bundles write the primary as `scene.<fmt>`; zip
        # results exist only as bytes, so the tool persists them itself.
        primary_path: Optional[str] = None
        if fmt == "zip":
            zip_path = Path(str(output_dir)).expanduser() / "bundle.zip"
            data = result.get("data") if isinstance(result, dict) else None
            if isinstance(data, (bytes, bytearray)):
                zip_path.parent.mkdir(parents=True, exist_ok=True)
                zip_path.write_bytes(bytes(data))
                primary_path = str(zip_path)
        elif bundle_dir is not None and (bundle_dir / f"scene.{fmt}").is_file():
            primary_path = str(bundle_dir / f"scene.{fmt}")
        if primary_path is None:
            data = result.get("data") if isinstance(result, dict) else None
            if isinstance(data, (bytes, bytearray)):
                fallback = Path(str(output_dir)).expanduser() / f"generated.{fmt}"
                fallback.parent.mkdir(parents=True, exist_ok=True)
                fallback.write_bytes(bytes(data))
                primary_path = str(fallback)

        return _ok(
            task="image_to_scene3d" if image_path else "text_to_scene3d",
            output_dir=str(Path(str(output_dir)).expanduser()),
            primary_path=primary_path,
            bundle_dir=str(bundle_dir) if bundle_dir is not None else None,
            format=str(result.get("format") if isinstance(result, dict) else fmt),
            backend_id=str(result.get("backend_id")) if isinstance(result, dict) and result.get("backend_id") else backend,
            metadata=_json_safe_metadata(metadata, keys=("model_id", "device", "seed", "output_bytes")),
        )
    except Abstract3DError as e:
        return _fail(e)
    except Exception as e:  # keep tool output model-safe
        return _fail(f"{type(e).__name__}: {e}")


@_tool_decorator(
    name="analyze_3d_object",
    description=(
        "Analyze a 3D mesh file (GLB/OBJ/STL/PLY) and report vertex/face counts, bounds, "
        "watertightness, volume, area, components, and UV/texture presence."
    ),
    tags=_TOOL_TAGS,
    when_to_use="When you need facts about an existing 3D file before or after modifying it.",
    examples=[{"description": "Inspect a GLB", "arguments": {"path": "./out/teapot/scene.glb"}}],
)
def analyze_3d_object(path: str) -> str:
    """Report the structure of a mesh file as JSON."""
    try:
        from .mesh_ops import analyze_mesh  # noqa: PLC0415

        report = analyze_mesh(path)
        return _ok(report=report.to_dict(), summary=report.summary())
    except Abstract3DError as e:
        return _fail(e)
    except Exception as e:
        return _fail(f"{type(e).__name__}: {e}")


@_tool_decorator(
    name="transform_3d_object",
    description=(
        "Transform a 3D mesh file: scale, rotate (degrees around X/Y/Z), translate, "
        "mirror across an axis, or re-center, then write the result."
    ),
    tags=_TOOL_TAGS,
    when_to_use=(
        "When the user asks to resize, rotate, move, mirror, or center a 3D object. Omitting "
        "output_path overwrites writable inputs (glb/obj/stl/ply/off); gltf/3mf/dae need an explicit one."
    ),
    examples=[
        {
            "description": "Double the size and rotate 90 degrees around Y",
            "arguments": {"input_path": "./scene.glb", "output_path": "./scene_big.glb", "scale": 2.0, "rotate_deg": [0, 90, 0]},
        }
    ],
)
def transform_3d_object(
    input_path: str,
    output_path: Optional[str] = None,
    scale: Optional[float] = None,
    scale_xyz: Optional[List[float]] = None,
    rotate_deg: Optional[List[float]] = None,
    translate: Optional[List[float]] = None,
    mirror_axis: Optional[str] = None,
    center: bool = False,
) -> str:
    """Apply scale/rotate/translate/mirror/center to a mesh file."""
    try:
        from .mesh_ops import transform_mesh  # noqa: PLC0415

        if scale is not None and scale_xyz is not None:
            return _fail("Provide either scale (uniform) or scale_xyz (per-axis), not both.")
        result = transform_mesh(
            input_path,
            output_path,
            scale=scale_xyz if scale_xyz is not None else scale,
            rotate_deg=rotate_deg,
            translate=translate,
            mirror_axis=mirror_axis,
            center=_coerce_bool(center, field="center"),
        )
        return _ok(**result)
    except Abstract3DError as e:
        return _fail(e)
    except Exception as e:
        return _fail(f"{type(e).__name__}: {e}")


@_tool_decorator(
    name="compose_3d_scene",
    description=(
        "Compose several 3D mesh files into one scene file, with optional per-part "
        "scale/rotate/translate placement transforms."
    ),
    tags=_TOOL_TAGS,
    when_to_use=(
        "When the user asks to arrange multiple 3D objects together into one scene. Each part is "
        '{"path": "...", "name"?, "scale"?: number|[x,y,z], "rotate_deg"?: [x,y,z], '
        '"translate"?: [x,y,z], "mirror_axis"?: "x"|"y"|"z"}.'
    ),
    examples=[
        {
            "description": "Two objects side by side",
            "arguments": {
                "parts": [
                    {"path": "./teapot.glb", "translate": [-0.5, 0, 0]},
                    {"path": "./cup.glb", "translate": [0.5, 0, 0]},
                ],
                "output_path": "./scene.glb",
            },
        }
    ],
)
def compose_3d_scene(parts: List[Dict[str, Any]], output_path: str) -> str:
    """Merge mesh files into one exported scene with per-part transforms."""
    try:
        from .mesh_ops import compose_scene  # noqa: PLC0415

        if isinstance(parts, str):
            parts = json.loads(parts)
        result = compose_scene(parts, output_path)
        return _ok(**result)
    except Abstract3DError as e:
        return _fail(e)
    except Exception as e:
        return _fail(f"{type(e).__name__}: {e}")


@_tool_decorator(
    name="convert_3d_object",
    description=(
        "Convert a 3D mesh file to another format chosen by the output suffix "
        "(glb/obj/stl/ply/off); geometry is preserved."
    ),
    tags=_TOOL_TAGS,
    when_to_use=(
        "When the user needs the same 3D object in a different file format. "
        "Material fidelity depends on the target format (STL drops materials)."
    ),
    examples=[{"description": "GLB to STL for printing", "arguments": {"input_path": "./scene.glb", "output_path": "./scene.stl"}}],
)
def convert_3d_object(input_path: str, output_path: str) -> str:
    """Convert a mesh file to the format implied by output_path's suffix."""
    try:
        from .mesh_ops import convert_mesh  # noqa: PLC0415

        return _ok(**convert_mesh(input_path, output_path))
    except Abstract3DError as e:
        return _fail(e)
    except Exception as e:
        return _fail(f"{type(e).__name__}: {e}")


@_tool_decorator(
    name="repair_3d_object",
    description=(
        "Conservative mesh cleanup: weld duplicate vertices, drop degenerate faces, fix "
        "winding/normals, optionally fill small holes."
    ),
    tags=_TOOL_TAGS,
    when_to_use=(
        "When a mesh has holes, inverted normals, or duplicate geometry to clean before use. "
        "Reports before/after counts so you can see what changed."
    ),
    examples=[{"description": "Clean a scanned mesh", "arguments": {"input_path": "./scan.glb", "output_path": "./scan_clean.glb"}}],
)
def repair_3d_object(input_path: str, output_path: Optional[str] = None, fill_holes: bool = True) -> str:
    """Run basic cleanup on a mesh file and report what changed."""
    try:
        from .mesh_ops import repair_mesh  # noqa: PLC0415

        return _ok(**repair_mesh(input_path, output_path, fill_holes=_coerce_bool(fill_holes, field="fill_holes")))
    except Abstract3DError as e:
        return _fail(e)
    except Exception as e:
        return _fail(f"{type(e).__name__}: {e}")


@_tool_decorator(
    name="render_3d_preview",
    description=(
        "Render turntable preview images of a 3D mesh file into one PNG contact strip "
        "for visual inspection."
    ),
    tags=_TOOL_TAGS,
    when_to_use="When you need to see what a 3D file looks like (before/after checks, visual QA).",
    examples=[{"description": "Preview a GLB", "arguments": {"input_path": "./scene.glb"}}],
)
def render_3d_preview(
    input_path: str,
    output_path: Optional[str] = None,
    size: int = 420,
) -> str:
    """Render preview views of a mesh into a PNG contact strip."""
    try:
        from .mesh_ops import render_preview  # noqa: PLC0415

        return _ok(**render_preview(input_path, output_path, size=int(size)))
    except Abstract3DError as e:
        return _fail(e)
    except Exception as e:
        return _fail(f"{type(e).__name__}: {e}")


@_tool_decorator(
    name="list_3d_backends",
    description=(
        "List Abstract3D generation backends and model families with status "
        "(validated/experimental), license, and platform notes."
    ),
    tags=_TOOL_TAGS,
    when_to_use="When choosing a 3D generation backend or checking what is installed/validated.",
    examples=[{"description": "Show validated backends only", "arguments": {"validated_only": True}}],
)
def list_3d_backends(validated_only: bool = False) -> str:
    """Return the Abstract3D model catalog as JSON rows."""
    try:
        from .model_catalog import catalog_rows  # noqa: PLC0415

        rows = catalog_rows(validated_only=_coerce_bool(validated_only, field="validated_only"))
        return _ok(count=len(rows), backends=rows)
    except Abstract3DError as e:
        return _fail(e)
    except Exception as e:
        return _fail(f"{type(e).__name__}: {e}")


TOOL_FUNCTIONS: tuple[Callable[..., str], ...] = (
    generate_3d_object,
    analyze_3d_object,
    transform_3d_object,
    compose_3d_scene,
    convert_3d_object,
    repair_3d_object,
    render_3d_preview,
    list_3d_backends,
)

# Package-owned classification in AbstractCore's inventory vocabulary
# (`mutating` = local state change, `remote_write_capable` = remote effects)
# plus one domain tag per the ruled general shape for domain classification
# tags (semantics decision:domain-tool-classification-tags, commons 2026-07-19;
# shape aligned with abstractcamera so runtime/gateway consume both uniformly):
#
# `downloads_model_weights`: true when invoking the tool may fetch model
# weights over the network on first use (the bandwidth/disk/supply-chain
# consent fact) — approval layers may want to gate it in unattended contexts.
SCENE3D_TOOL_CLASSIFICATION: Dict[str, Dict[str, bool]] = {
    "generate_3d_object": {"mutating": True, "remote_write_capable": False, "downloads_model_weights": True},
    "analyze_3d_object": {"mutating": False, "remote_write_capable": False, "downloads_model_weights": False},
    "transform_3d_object": {"mutating": True, "remote_write_capable": False, "downloads_model_weights": False},
    "compose_3d_scene": {"mutating": True, "remote_write_capable": False, "downloads_model_weights": False},
    "convert_3d_object": {"mutating": True, "remote_write_capable": False, "downloads_model_weights": False},
    "repair_3d_object": {"mutating": True, "remote_write_capable": False, "downloads_model_weights": False},
    "render_3d_preview": {"mutating": True, "remote_write_capable": False, "downloads_model_weights": False},
    "list_3d_backends": {"mutating": False, "remote_write_capable": False, "downloads_model_weights": False},
}


def abstract3d_tools() -> List[Callable[..., str]]:
    """Return the Abstract3D tool callables (usable with or without AbstractCore)."""
    return list(TOOL_FUNCTIONS)


def abstract3d_tool_definitions() -> List[Any]:
    """Return AbstractCore ToolDefinitions for all Abstract3D tools.

    This is the ruled explicit-import contract (core, commons 2026-07-19):
    callers register these consciously — there is deliberately no entry-point
    auto-registration for tools. Requires AbstractCore (the definitions are
    attached by its @tool decorator at import time); raises loudly otherwise.
    """
    definitions: List[Any] = []
    for fn in TOOL_FUNCTIONS:
        tool_def = getattr(fn, "_tool_definition", None)
        if tool_def is None:
            raise Abstract3DError(
                "Tool definitions require AbstractCore (the @tool decorator was not "
                "available at import time). Install it with: pip install abstractcore, "
                "then re-import abstract3d.tools."
            )
        definitions.append(tool_def)
    return definitions


def abstract3d_tool_specs() -> List[Dict[str, Any]]:
    """Return JSON-safe tool specs ({name, description, parameters}) for
    native tool declaration, annotated with the package classification."""
    specs: List[Dict[str, Any]] = []
    for tool_def in abstract3d_tool_definitions():
        spec = {
            "name": str(tool_def.name),
            "description": str(tool_def.description or ""),
            "parameters": dict(tool_def.parameters or {}),
        }
        spec.update(SCENE3D_TOOL_CLASSIFICATION.get(str(tool_def.name), {}))
        specs.append(spec)
    return specs


# Back-compat friendly alias retained for callers that predate the ruled
# `abstract3d_*` accessor naming.
get_tool_functions = abstract3d_tools


def register_tools(registry: Any) -> List[Any]:
    """Register all Abstract3D tools into an explicit AbstractCore ToolRegistry.

    Core's *global* tool registration is deprecated — pass the registry you
    own (or pass ``abstract3d_tools()`` directly to ``generate(tools=...)``).
    """
    if registry is None or not callable(getattr(registry, "register", None)):
        raise Abstract3DError(
            "register_tools needs an explicit ToolRegistry with a .register() method "
            "(AbstractCore's global registration is deprecated). "
            "Alternatively pass abstract3d_tools() to generate(tools=...)."
        )
    return [registry.register(fn) for fn in TOOL_FUNCTIONS]


__all__ = [
    "SCENE3D_TOOL_CLASSIFICATION",
    "TOOL_FUNCTIONS",
    "abstract3d_tool_definitions",
    "abstract3d_tool_specs",
    "abstract3d_tools",
    "analyze_3d_object",
    "compose_3d_scene",
    "convert_3d_object",
    "generate_3d_object",
    "get_tool_functions",
    "list_3d_backends",
    "register_tools",
    "render_3d_preview",
    "repair_3d_object",
    "transform_3d_object",
]
