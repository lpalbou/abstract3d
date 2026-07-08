"""Public bundle API: load generated bundles and rebake their textures.

Every generated bundle deliberately persists `geometry.glb` in the CANONICAL
frame (Z-up, front +X) precisely so the texture can be redone later with new
or additional reference views — the entire zero-defect certification program
and the generated-reference-views proofs were produced through this path.
Until now that path only existed as ad-hoc scripts; this module makes it a
supported API (and the golden-bake regression harness runs through it, so
the canonical recipe below is executable and hash-verified).

Canonical rebake recipe (matches the certified proof assets):

- mesh: `geometry.glb` loaded with `process=False` (topology must not change)
- source view: `input.png` through robust background removal, azimuth 0
- reference views: RGBA photos with angles relative to the source viewpoint
- bake: `bake_projection_texture(..., projection_model="orthographic",
  texture_completion="auto")` at the bundle's texture resolution
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

BUNDLE_SCHEMA_VERSION = 1


@dataclass
class LoadedBundle:
    """A generated bundle on disk, resolved to its rebake inputs."""

    bundle_dir: Path
    metadata: Dict[str, Any]

    @property
    def geometry_path(self) -> Path:
        return self.bundle_dir / "geometry.glb"

    @property
    def input_path(self) -> Path:
        return self.bundle_dir / "input.png"

    @property
    def texture_path(self) -> Path:
        return self.bundle_dir / "texture.png"

    def geometry_mesh(self) -> Any:
        """The canonical-frame geometry mesh (safe for rebakes)."""

        import trimesh

        if not self.geometry_path.exists():
            raise FileNotFoundError(
                f"{self.geometry_path} is missing: this bundle cannot be rebaked. "
                "Bundles produced before the geometry.glb contract must be regenerated."
            )
        return trimesh.load(str(self.geometry_path), force="mesh", process=False)


def load_bundle(bundle_dir: Union[str, Path]) -> LoadedBundle:
    bundle_dir = Path(bundle_dir)
    metadata_path = bundle_dir / "metadata.json"
    metadata: Dict[str, Any] = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
    return LoadedBundle(bundle_dir=bundle_dir, metadata=metadata)


def _resolve_angle(angle: Union[str, float, Tuple[float, float]]) -> Tuple[float, float, str]:
    """Resolve an angle spec (label, degrees, or (az, el)) to (az, el, label)."""

    from .backends.triposr_runtime import _tripo_parse_texture_reference_angle

    return _tripo_parse_texture_reference_angle(angle)


def prepare_observed_views(
    source_image: Any,
    references: Sequence[Mapping[str, Any]] = (),
    *,
    remove_source_background: bool = True,
) -> List[Dict[str, Any]]:
    """Build the `observed_views` list for `bake_projection_texture`.

    `source_image` is a path or PIL image for the azimuth-0 source photo.
    Each reference mapping carries `image` (path or PIL) plus `angle`
    (a label like "side_left", degrees, or an (azimuth, elevation) pair) and
    optional `remove_background` / `label` keys. Reference angles are
    interpreted relative to the source viewpoint, matching the bake contract.
    """

    from PIL import Image

    from .segmentation import remove_background_robust

    def to_image(value: Any) -> Any:
        if isinstance(value, (str, Path)):
            return Image.open(value)
        return value

    original = to_image(source_image)
    if remove_source_background:
        source = remove_background_robust(original)
    else:
        source = original.convert("RGBA")
    views: List[Dict[str, Any]] = [
        # `identity_image` carries the un-matted photo: multi-view bakes build
        # the identity correspondence against it (matted silhouettes snap the
        # registration into a different basin — see the fringe-repair stage).
        {"rgba": source, "azimuth_deg": 0.0, "elevation_deg": 0.0,
         "label": "front", "role": "source", "identity_image": original}
    ]
    for index, reference in enumerate(references, start=1):
        image = to_image(reference["image"])
        if reference.get("remove_background"):
            rgba = remove_background_robust(image)
        else:
            rgba = image.convert("RGBA")
        azimuth, elevation, label = _resolve_angle(reference.get("angle", 0.0))
        views.append(
            {
                "rgba": rgba,
                "azimuth_deg": float(azimuth),
                "elevation_deg": float(elevation),
                "label": str(reference.get("label") or label or f"reference_{index:02d}"),
                "role": "reference",
            }
        )
    return views


def rebake_bundle(
    bundle_dir: Union[str, Path],
    *,
    output_dir: Optional[Union[str, Path]] = None,
    references: Sequence[Mapping[str, Any]] = (),
    texture_resolution: Optional[int] = None,
    texture_completion: str = "auto",
    projection_model: str = "orthographic",
    source_pose_override: Optional[Tuple[float, float]] = None,
    scarcity_rescue: str = "auto",
    compositing: str = "auto",
    write_outputs: bool = True,
) -> Tuple[Any, Dict[str, Any]]:
    """Rebake a bundle's texture from its persisted canonical geometry.

    Returns `(textured_mesh, stats)`. With `write_outputs`, writes a bundle
    revision into `output_dir` (default: `<bundle>/rebake/`): viewer-frame
    `scene.glb`, `texture.png`, `uv_preview.png`, and `metadata.json` with
    the bake stats, input hashes, and schema version.

    Caveat documented from the review: TripoSR bundles rebake without the
    triplane color prior (`base_color_fn` requires the resident model), so
    unseen texels fall back to projection fill; Hunyuan bundles rebake with
    full fidelity. The certified proof assets are Hunyuan bundles.
    """

    from . import texturing
    from .backends.triposr_runtime import _mesh_export_bytes

    loaded = load_bundle(bundle_dir)
    mesh = loaded.geometry_mesh()
    resolution = int(
        texture_resolution
        or loaded.metadata.get("texture_resolution")
        or 2048
    )
    views = prepare_observed_views(loaded.input_path, references)

    started = time.perf_counter()
    textured, stats = texturing.bake_projection_texture(
        mesh,
        observed_views=views,
        texture_resolution=resolution,
        texture_completion=texture_completion,
        projection_model=projection_model,
        source_pose_override=source_pose_override,
        scarcity_rescue=scarcity_rescue,
        compositing=compositing,
    )
    bake_seconds = round(time.perf_counter() - started, 3)

    if write_outputs:
        out = Path(output_dir) if output_dir else (loaded.bundle_dir / "rebake")
        out.mkdir(parents=True, exist_ok=True)
        (out / "scene.glb").write_bytes(_mesh_export_bytes(textured, file_type="glb"))
        stats["texture_image"].save(out / "texture.png")
        uv_preview = stats.get("uv_preview")
        if uv_preview is not None:
            uv_preview.save(out / "uv_preview.png")
        texture_md5 = hashlib.md5((out / "texture.png").read_bytes()).hexdigest()
        clean_stats = {
            k: v
            for k, v in stats.items()
            if k not in ("texture_image", "uv_preview", "vmapping", "indices", "uvs")
        }
        metadata = {
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "kind": "rebake",
            "source_bundle": str(loaded.bundle_dir),
            "texture_resolution": resolution,
            "texture_completion": texture_completion,
            "projection_model": projection_model,
            "source_pose_override": source_pose_override,
            "scarcity_rescue": scarcity_rescue,
            "compositing": compositing,
            "reference_count": len(views) - 1,
            "bake_seconds": bake_seconds,
            "texture_png_md5": texture_md5,
            "stats": _json_safe(clean_stats),
        }
        (out / "metadata.json").write_text(json.dumps(metadata, indent=1, default=str))
    return textured, stats


def _json_safe(value: Any) -> Any:
    """Best-effort conversion of bake stats to JSON-serializable values."""

    import numpy as np

    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.size > 64:
            return f"<ndarray shape={value.shape} dtype={value.dtype}>"
        return value.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
