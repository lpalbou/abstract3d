"""Turntable preview rendering for mesh files.

Split from `mesh_ops` (which owns the deterministic load/analyze/transform/
compose/convert/repair operations): preview rendering is a presentation
concern with its own dependency surface (Pillow + moderngl-or-matplotlib
through the shared `abstract3d.rendering` renderer).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

from .errors import DependencyUnavailableError, InvalidRequestError
from .mesh_ops import _MESH_EXTRA_HINT, _resolve_input_path, as_single_mesh, load_mesh


def render_preview(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    *,
    size: int = 420,
    azimuths: Sequence[float] = (35.0, 125.0, 215.0, 305.0),
    elevation: float = 20.0,
) -> Dict[str, Any]:
    """Render turntable preview views of a mesh file into one PNG contact strip.

    Uses the shared Abstract3D renderer (moderngl when available, matplotlib
    fallback). Requires Pillow plus one of moderngl/matplotlib — all included
    in the `abstract3d[mesh]` extra.
    """
    try:
        from PIL import Image  # noqa: PLC0415
    except Exception as e:  # pragma: no cover
        raise DependencyUnavailableError(_MESH_EXTRA_HINT) from e

    from .rendering import get_last_render_backend, render_mesh_views  # noqa: PLC0415

    p = _resolve_input_path(input_path)
    out = Path(str(output_path)).expanduser() if output_path else p.with_suffix(".preview.png")
    if out.suffix.lower() != ".png":
        raise InvalidRequestError("render_preview writes a .png file.")
    out.parent.mkdir(parents=True, exist_ok=True)

    mesh = as_single_mesh(load_mesh(p))
    if p.suffix.lower() in {".glb", ".gltf"}:
        # glTF is Y-up / front +Z BY SPEC, but the shared renderer's camera
        # math is canonical-frame (Z-up / front +X). Stamp the export-frame
        # marker so render_mesh_views applies its one un-rotation. Metadata
        # markers alone are not enough: multi-geometry scenes lose per-part
        # metadata through Scene.to_mesh(), which rendered composed scenes
        # lying on their sides.
        try:
            mesh.metadata["abstract3d_export_frame"] = "gltf_yup_front_pz"
        except Exception:
            pass
    views = render_mesh_views(mesh, size=int(size), azimuths=tuple(azimuths), elevation=float(elevation))
    if not views:
        raise DependencyUnavailableError(
            "No preview renderer produced output. Install moderngl or matplotlib "
            '(both included in: pip install "abstract3d[mesh]").'
        )

    strip = Image.new("RGB", (size * len(views), size), "#ebe7df")
    for i, view in enumerate(views):
        tile = view.convert("RGB").copy()
        tile.thumbnail((size, size))
        strip.paste(tile, (i * size + (size - tile.width) // 2, (size - tile.height) // 2))
    strip.save(out)

    return {
        "input_path": str(p),
        "output_path": str(out),
        "view_count": len(views),
        "azimuths_deg": [float(a) for a in azimuths],
        "elevation_deg": float(elevation),
        "renderer": get_last_render_backend(),
        "output_bytes": int(out.stat().st_size),
    }


__all__ = ["render_preview"]
