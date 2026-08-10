"""Deterministic mesh operations: load, analyze, transform, compose, convert, repair.

This module is the manipulation/analysis half of Abstract3D. Generation
(`Scene3DManager`, the backends) produces assets; this module operates on any
mesh file on disk — including assets produced elsewhere — with no model
runtime and no network access.

Conventions
-----------
- Files are read and written in their native file frame. glTF/GLB uses Y-up
  with +Z toward the viewer; OBJ has no mandated frame. No hidden
  re-orientation is applied by transforms: `rotate_deg=(x, y, z)` means
  exactly those axes in the file's own frame.
- Transform composition order is scale -> rotate (X, then Y, then Z) ->
  translate, i.e. the standard TRS order applied about the file-frame origin
  (use ``center=True`` to re-center on the bounding-box centroid first).
- Dependencies (numpy, trimesh, Pillow) are imported lazily so the base
  install stays light; missing dependencies raise a loud, actionable error.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from .errors import DependencyUnavailableError, InvalidRequestError

_MESH_EXTRA_HINT = (
    'Mesh operations need numpy + trimesh (and Pillow for previews). '
    'Install them with: pip install "abstract3d[mesh]"'
)

# Formats trimesh can both load and export reliably for our use cases.
SUPPORTED_LOAD_SUFFIXES = {".glb", ".gltf", ".obj", ".stl", ".ply", ".off", ".3mf", ".dae"}
SUPPORTED_EXPORT_SUFFIXES = {".glb", ".obj", ".stl", ".ply", ".off"}


def _import_trimesh():
    try:
        import trimesh  # noqa: PLC0415
    except Exception as e:  # pragma: no cover - exercised via error-path tests
        raise DependencyUnavailableError(_MESH_EXTRA_HINT) from e
    return trimesh


def _import_numpy():
    try:
        import numpy  # noqa: PLC0415
    except Exception as e:  # pragma: no cover
        raise DependencyUnavailableError(_MESH_EXTRA_HINT) from e
    return numpy


def _resolve_input_path(path: Union[str, Path]) -> Path:
    p = Path(str(path)).expanduser()
    if not p.exists() or not p.is_file():
        raise InvalidRequestError(f"Mesh file not found: {p}")
    if p.suffix.lower() not in SUPPORTED_LOAD_SUFFIXES:
        raise InvalidRequestError(
            f"Unsupported mesh format {p.suffix!r} for {p.name}. "
            f"Supported inputs: {', '.join(sorted(SUPPORTED_LOAD_SUFFIXES))}."
        )
    return p


def _resolve_output_path(path: Union[str, Path], *, default_reference: Optional[Path] = None) -> Path:
    p = Path(str(path)).expanduser()
    suffix = p.suffix.lower()
    if not suffix and default_reference is not None:
        p = p.with_suffix(default_reference.suffix)
        suffix = p.suffix.lower()
    if suffix not in SUPPORTED_EXPORT_SUFFIXES:
        raise InvalidRequestError(
            f"Unsupported export format {suffix!r} for {p.name}. "
            f"Supported outputs: {', '.join(sorted(SUPPORTED_EXPORT_SUFFIXES))}."
        )
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _load_file(p: Path, *, process: bool) -> Any:
    """trimesh.load with corrupt-file errors wrapped into actionable text."""
    trimesh = _import_trimesh()
    try:
        return trimesh.load(str(p), process=process)
    except InvalidRequestError:
        raise
    except Exception as e:
        raise InvalidRequestError(
            f"{p} could not be parsed as {p.suffix.lower().lstrip('.') or 'a mesh file'} "
            f"(corrupt or unsupported content): {type(e).__name__}: {e}"
        ) from e


def load_mesh(path: Union[str, Path]) -> Any:
    """Load a mesh file into a `trimesh.Trimesh` or `trimesh.Scene`.

    Loads the STORED geometry verbatim (``process=False``): trimesh's default
    load-time processing re-welds vertices on every load, and repeated
    load/save cycles compound — floats drift under transforms, previously
    distinct vertices fall inside the weld tolerance, and thin triangles
    collapse to zero area (independently measured by MeshVault on a composed
    scene: 2 -> 6 degenerate triangles through one compose pass).
    Operations must not silently re-topologize their inputs; welding is
    :func:`repair_mesh`'s explicit job.

    Format note: glTF/GLB ALWAYS loads as a Scene (even single-geometry
    files); most single-mesh formats (STL/PLY/OFF) load as a Trimesh.
    Callers that need one concrete mesh can use :func:`as_single_mesh`.
    """
    p = _resolve_input_path(path)
    return _load_file(p, process=False)


def as_single_mesh(loaded: Any) -> Any:
    """Concatenate a Scene into one Trimesh; pass a Trimesh through unchanged."""
    trimesh = _import_trimesh()
    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise InvalidRequestError("Scene contains no geometry.")
        return loaded.to_mesh()
    if isinstance(loaded, trimesh.Trimesh):
        return loaded
    raise InvalidRequestError(f"Unsupported loaded mesh type: {type(loaded).__name__}")


def _require_triangle_geometry(target: Any, *, source_name: str, operation: str) -> None:
    """Refuse point clouds / face-less content on write operations.

    Analysis refuses these loudly; write operations must match — a transform
    of a points-only GLB used to write an empty mesh and report success.
    """
    trimesh = _import_trimesh()
    if isinstance(target, trimesh.Trimesh):
        meshes = [target]
    elif isinstance(target, trimesh.Scene):
        meshes = [g for g in target.geometry.values() if isinstance(g, trimesh.Trimesh)]
    else:
        meshes = []
    if not any(len(getattr(m, "faces", ())) > 0 for m in meshes):
        raise InvalidRequestError(
            f"{source_name} contains no triangle geometry to {operation} "
            "(point clouds and empty scenes are not supported)."
        )


@dataclass(frozen=True)
class MeshReport:
    """JSON-safe structural analysis of one mesh file."""

    path: str
    format: str
    file_bytes: int
    geometry_count: int
    vertex_count: int
    face_count: int
    bounds_min: Tuple[float, float, float]
    bounds_max: Tuple[float, float, float]
    extents: Tuple[float, float, float]
    centroid: Tuple[float, float, float]
    surface_area: float
    is_watertight: bool
    volume: Optional[float]
    euler_number: Optional[int]
    # None = could not be computed (missing graph engine / failure) — a
    # wrong number must never masquerade as a fact in an LLM context.
    connected_components: Optional[int]
    duplicate_faces: Optional[int]
    degenerate_faces: Optional[int]
    has_uv: bool
    has_texture: bool
    has_vertex_colors: bool
    material_names: List[str] = field(default_factory=list)
    geometry_names: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        volume_s = f"{self.volume:.6g}" if self.volume is not None else "n/a (not watertight)"
        components_s = (
            str(self.connected_components)
            if self.connected_components is not None
            else "unknown (no graph engine — pip install scipy)"
        )
        return (
            f"{Path(self.path).name}: {self.vertex_count} vertices, {self.face_count} faces, "
            f"{self.geometry_count} geometrie(s), {components_s} component(s); "
            f"extents {tuple(round(v, 4) for v in self.extents)}; "
            f"watertight={self.is_watertight}; surface_area={self.surface_area:.6g}; "
            f"volume={volume_s}; uv={self.has_uv}; texture={self.has_texture}"
        )


def _vector3(value: Any) -> Tuple[float, float, float]:
    # Strings iterate character-by-character ("123" would silently parse as
    # [1, 2, 3]); reject them so the loud error names the real problem.
    if isinstance(value, (str, bytes)):
        raise InvalidRequestError(f"Expected a 3-vector like [x, y, z], got the string {value!r}")
    try:
        seq = [float(v) for v in value]
    except (TypeError, ValueError) as e:
        raise InvalidRequestError(f"Expected a 3-vector like [x, y, z], got {value!r}") from e
    if len(seq) != 3:
        raise InvalidRequestError(f"Expected a 3-vector like [x, y, z], got {value!r}")
    return (seq[0], seq[1], seq[2])


def _finite_or_none(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def analyze_mesh(path: Union[str, Path]) -> MeshReport:
    """Compute a structural report for one mesh file.

    Works on any supported mesh file. Volume is only reported for watertight
    geometry (an open mesh has no well-defined enclosed volume).
    """
    trimesh = _import_trimesh()
    np = _import_numpy()
    p = _resolve_input_path(path)
    # Topology metrics (watertight, volume, components, euler) are only
    # meaningful on a welded view: formats like glTF legitimately duplicate
    # vertices for per-vertex normals/UVs, and an unwelded box would read as
    # 12 disconnected islands. Analysis therefore uses trimesh's processed
    # load; write operations (transform/compose/convert) deliberately do not.
    loaded = _load_file(p, process=True)

    geometry_names: List[str] = []
    material_names: List[str] = []
    if isinstance(loaded, trimesh.Scene):
        geometry_names = [str(name) for name in loaded.geometry.keys()]
        meshes = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise InvalidRequestError(f"{p.name} contains no triangle geometry.")
        merged = loaded.to_mesh()
    else:
        merged = as_single_mesh(loaded)
        meshes = [merged]
        geometry_names = [p.stem]

    has_uv = False
    has_texture = False
    has_vertex_colors = False
    for mesh in meshes:
        visual = getattr(mesh, "visual", None)
        if visual is None:
            continue
        uv = getattr(visual, "uv", None)
        if uv is not None and len(uv) > 0:
            has_uv = True
        material = getattr(visual, "material", None)
        if material is not None:
            name = str(getattr(material, "name", "") or "").strip()
            if name:
                material_names.append(name)
            image = getattr(material, "image", None) or getattr(material, "baseColorTexture", None)
            if image is not None:
                has_texture = True
        kind = str(getattr(visual, "kind", "") or "")
        if kind == "vertex":
            has_vertex_colors = True

    face_count = int(len(merged.faces))
    vertex_count = int(len(merged.vertices))
    if face_count == 0 or vertex_count == 0:
        raise InvalidRequestError(f"{p.name} contains no triangle geometry.")

    bounds = merged.bounds
    is_watertight = bool(merged.is_watertight)
    volume = _finite_or_none(merged.volume) if is_watertight else None
    try:
        euler = int(merged.euler_number)
    except Exception:
        euler = None
    # A metric that cannot be computed reports None, never a guessed number:
    # trimesh needs a graph engine (scipy/networkx) for split(), and reporting
    # "1 component" without one would put a false fact into an LLM context.
    components: Optional[int]
    try:
        components = int(len(merged.split(only_watertight=False)))
    except Exception:
        components = None

    duplicate_faces: Optional[int]
    try:
        unique = np.unique(np.sort(merged.faces, axis=1), axis=0)
        duplicate_faces = int(face_count - len(unique))
    except Exception:
        duplicate_faces = None
    degenerate_faces: Optional[int]
    try:
        degenerate_faces = int(face_count - int((merged.area_faces > 1e-12).sum()))
    except Exception:
        degenerate_faces = None

    return MeshReport(
        path=str(p),
        format=p.suffix.lower().lstrip("."),
        file_bytes=int(p.stat().st_size),
        geometry_count=int(len(meshes)),
        vertex_count=vertex_count,
        face_count=face_count,
        bounds_min=_vector3(bounds[0]),
        bounds_max=_vector3(bounds[1]),
        extents=_vector3(merged.extents),
        centroid=_vector3(merged.bounding_box.centroid),
        surface_area=float(merged.area),
        is_watertight=is_watertight,
        volume=volume,
        euler_number=euler,
        connected_components=components,
        duplicate_faces=duplicate_faces,
        degenerate_faces=degenerate_faces,
        has_uv=has_uv,
        has_texture=has_texture,
        has_vertex_colors=has_vertex_colors,
        material_names=sorted(set(material_names)),
        geometry_names=geometry_names,
    )


def _build_trs_matrix(
    *,
    scale: Optional[Union[float, Sequence[float]]] = None,
    rotate_deg: Optional[Sequence[float]] = None,
    translate: Optional[Sequence[float]] = None,
    mirror_axis: Optional[str] = None,
) -> Any:
    np = _import_numpy()
    matrix = np.eye(4)

    if scale is not None:
        if isinstance(scale, (int, float)):
            factors = (float(scale),) * 3
        else:
            factors = _vector3(scale)
        if any(f == 0 for f in factors):
            raise InvalidRequestError("scale factors must be non-zero")
        s = np.eye(4)
        s[0, 0], s[1, 1], s[2, 2] = factors
        matrix = s @ matrix

    if mirror_axis is not None:
        axis = str(mirror_axis).strip().lower()
        index = {"x": 0, "y": 1, "z": 2}.get(axis)
        if index is None:
            raise InvalidRequestError(f"mirror_axis must be one of x, y, z (got {mirror_axis!r})")
        m = np.eye(4)
        m[index, index] = -1.0
        matrix = m @ matrix

    if rotate_deg is not None:
        rx, ry, rz = _vector3(rotate_deg)
        for angle_deg, axis_vec in ((rx, (1.0, 0.0, 0.0)), (ry, (0.0, 1.0, 0.0)), (rz, (0.0, 0.0, 1.0))):
            if not angle_deg:
                continue
            angle = math.radians(float(angle_deg))
            x, y, z = axis_vec
            c, s_ = math.cos(angle), math.sin(angle)
            t = 1.0 - c
            r = np.array(
                [
                    [t * x * x + c, t * x * y - s_ * z, t * x * z + s_ * y, 0.0],
                    [t * x * y + s_ * z, t * y * y + c, t * y * z - s_ * x, 0.0],
                    [t * x * z - s_ * y, t * y * z + s_ * x, t * z * z + c, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
            matrix = r @ matrix

    if translate is not None:
        tx, ty, tz = _vector3(translate)
        t = np.eye(4)
        t[0, 3], t[1, 3], t[2, 3] = tx, ty, tz
        matrix = t @ matrix

    return matrix


def transform_mesh(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    *,
    scale: Optional[Union[float, Sequence[float]]] = None,
    rotate_deg: Optional[Sequence[float]] = None,
    translate: Optional[Sequence[float]] = None,
    mirror_axis: Optional[str] = None,
    center: bool = False,
) -> Dict[str, Any]:
    """Apply a TRS transform to a mesh file and write the result.

    Composition order: (optional re-centering) -> scale -> mirror ->
    rotate X, then Y, then Z -> translate, all about the file-frame origin.
    Mirroring uses a negative-determinant transform; trimesh re-winds faces
    so surface orientation stays correct.

    Returns a JSON-safe dict with the output path and before/after bounds.
    """
    trimesh = _import_trimesh()
    np = _import_numpy()
    p = _resolve_input_path(input_path)
    out = _resolve_output_path(output_path or p, default_reference=p)

    if not any(v is not None for v in (scale, rotate_deg, translate, mirror_axis)) and not center:
        raise InvalidRequestError(
            "transform_mesh needs at least one of: scale, rotate_deg, translate, mirror_axis, center=True."
        )

    # Multi-part files transform as scenes so node structure and per-part
    # materials survive; flattening here would silently merge parts.
    # process=False: transform what is stored, never re-topologize (see
    # load_mesh).
    loaded = _load_file(p, process=False)
    target = loaded if isinstance(loaded, (trimesh.Trimesh, trimesh.Scene)) else as_single_mesh(loaded)
    is_scene = isinstance(target, trimesh.Scene)
    _require_triangle_geometry(target, source_name=p.name, operation="transform")
    if is_scene and out.suffix.lower() not in {".glb", ".obj"}:
        # Single-mesh formats cannot carry scene structure; flatten explicitly.
        target = target.to_mesh()
        is_scene = False

    bounds_before = [_vector3(target.bounds[0]), _vector3(target.bounds[1])]

    if center:
        centroid = (np.asarray(target.bounds[0], dtype=float) + np.asarray(target.bounds[1], dtype=float)) / 2.0
        shift = np.eye(4)
        shift[:3, 3] = -centroid
        target.apply_transform(shift)

    matrix = _build_trs_matrix(
        scale=scale, rotate_deg=rotate_deg, translate=translate, mirror_axis=mirror_axis
    )
    target.apply_transform(matrix)

    target.export(str(out))
    bounds_after = [_vector3(target.bounds[0]), _vector3(target.bounds[1])]
    merged = target.to_mesh() if is_scene else target
    return {
        "input_path": str(p),
        "output_path": str(out),
        "format": out.suffix.lower().lstrip("."),
        "scene": bool(is_scene),
        "vertex_count": int(len(merged.vertices)),
        "face_count": int(len(merged.faces)),
        "bounds_before": bounds_before,
        "bounds_after": bounds_after,
        "applied": {
            "center": bool(center),
            "scale": scale if scale is None or isinstance(scale, (int, float)) else list(_vector3(scale)),
            "mirror_axis": mirror_axis,
            "rotate_deg": None if rotate_deg is None else list(_vector3(rotate_deg)),
            "translate": None if translate is None else list(_vector3(translate)),
        },
    }


def compose_scene(
    parts: Sequence[Mapping[str, Any]],
    output_path: Union[str, Path],
) -> Dict[str, Any]:
    """Compose several mesh files into one scene and export it.

    Each part is a mapping:
    ``{"path": str, "name": str?, "scale": float|[x,y,z]?,``
    ``"rotate_deg": [x,y,z]?, "translate": [x,y,z]?, "mirror_axis": "x"|"y"|"z"?}``.
    Per-part transforms follow the :func:`transform_mesh` composition order
    and are baked into the part's node transform (geometry is not mutated).
    """
    trimesh = _import_trimesh()
    if not parts:
        raise InvalidRequestError("compose_scene needs at least one part.")
    out = _resolve_output_path(output_path)
    if out.suffix.lower() not in {".glb", ".obj"}:
        raise InvalidRequestError("Scene composition exports .glb (recommended) or .obj.")

    np = _import_numpy()
    scene = trimesh.Scene()
    # Collision-proof naming: track the FINAL assigned names, not just base
    # counts — parts named ["box", "box_2", "box"] must never collide (a
    # node-name collision makes trimesh silently orphan a part's geometry).
    assigned_names: set[str] = set()

    def _unique_name(base: str) -> str:
        if base not in assigned_names:
            assigned_names.add(base)
            return base
        suffix = 2
        while f"{base}_{suffix}" in assigned_names:
            suffix += 1
        name = f"{base}_{suffix}"
        assigned_names.add(name)
        return name

    manifest: List[Dict[str, Any]] = []
    for index, part in enumerate(parts):
        if not isinstance(part, Mapping):
            raise InvalidRequestError(f"Part {index} must be an object with at least a 'path'.")
        raw_path = part.get("path")
        if not raw_path:
            raise InvalidRequestError(f"Part {index} is missing 'path'.")
        p = _resolve_input_path(raw_path)
        unknown = sorted(set(part.keys()) - {"path", "name", "scale", "rotate_deg", "translate", "mirror_axis"})
        if unknown:
            raise InvalidRequestError(f"Part {index} has unsupported keys: {', '.join(unknown)}.")

        base = str(part.get("name") or p.stem).strip() or p.stem
        node_name = _unique_name(base)

        part_matrix = _build_trs_matrix(
            scale=part.get("scale"),
            rotate_deg=part.get("rotate_deg"),
            translate=part.get("translate"),
            mirror_axis=part.get("mirror_axis"),
        )

        # Byte-faithful composition: keep each part's stored vertex data
        # verbatim and compose the placement into NODE transforms. Flattening
        # a part with to_mesh() bakes its node transforms into vertices and
        # the GLB re-quantization to float32 collapses thin triangles
        # (independently measured by MeshVault: 2 -> 6 degenerate triangles
        # through one baked compose pass).
        loaded = _load_file(p, process=False)
        _require_triangle_geometry(loaded, source_name=p.name, operation="compose")
        if isinstance(loaded, trimesh.Trimesh):
            scene.add_geometry(loaded, node_name=node_name, geom_name=node_name, transform=part_matrix)
        elif isinstance(loaded, trimesh.Scene):
            for sub_index, sub_node in enumerate(loaded.graph.nodes_geometry):
                node_matrix, geom_name = loaded.graph.get(sub_node)
                geometry = loaded.geometry.get(geom_name)
                if geometry is None:
                    continue
                sub_name = (
                    node_name
                    if len(loaded.graph.nodes_geometry) == 1
                    else _unique_name(f"{node_name}_{sub_index}")
                )
                scene.add_geometry(
                    geometry,
                    node_name=sub_name,
                    geom_name=sub_name,
                    transform=np.asarray(part_matrix) @ np.asarray(node_matrix, dtype=float),
                )
        else:
            raise InvalidRequestError(f"Part {index} ({p.name}) is not a mesh or scene.")
        manifest.append({"name": node_name, "path": str(p)})

    scene.export(str(out))
    merged = scene.to_mesh()
    return {
        "output_path": str(out),
        "format": out.suffix.lower().lstrip("."),
        "part_count": len(manifest),
        "parts": manifest,
        "vertex_count": int(len(merged.vertices)),
        "face_count": int(len(merged.faces)),
        "bounds": [_vector3(merged.bounds[0]), _vector3(merged.bounds[1])],
    }


def convert_mesh(input_path: Union[str, Path], output_path: Union[str, Path]) -> Dict[str, Any]:
    """Convert a mesh file to another format (format inferred from suffix).

    Honest limits: format capabilities differ — e.g. STL carries no
    materials/UVs, and OBJ exports approximate PBR materials. Geometry is
    preserved; rich material fidelity is only guaranteed GLB->GLB.
    """
    trimesh = _import_trimesh()
    p = _resolve_input_path(input_path)
    out = _resolve_output_path(output_path)
    if out.suffix.lower() == p.suffix.lower() and Path(out) == p:
        raise InvalidRequestError("convert_mesh output must differ from the input path.")

    # process=False: convert what is stored, never re-topologize (see load_mesh).
    loaded = _load_file(p, process=False)
    target = loaded if isinstance(loaded, trimesh.Scene) else as_single_mesh(loaded)
    _require_triangle_geometry(target, source_name=p.name, operation="convert")
    if out.suffix.lower() in {".stl", ".off", ".ply"} and isinstance(target, trimesh.Scene):
        target = target.to_mesh()
    target.export(str(out))

    merged = target.to_mesh() if isinstance(target, trimesh.Scene) else target
    return {
        "input_path": str(p),
        "output_path": str(out),
        "input_format": p.suffix.lower().lstrip("."),
        "output_format": out.suffix.lower().lstrip("."),
        "vertex_count": int(len(merged.vertices)),
        "face_count": int(len(merged.faces)),
        "output_bytes": int(out.stat().st_size),
    }


def _repair_one_mesh(trimesh: Any, mesh: Any, *, fill_holes: bool) -> None:
    mesh.merge_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    try:
        trimesh.repair.fix_winding(mesh)
        trimesh.repair.fix_normals(mesh)
    except Exception:
        pass
    if fill_holes:
        try:
            trimesh.repair.fill_holes(mesh)
        except Exception:
            pass


def repair_mesh(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    *,
    fill_holes: bool = True,
) -> Dict[str, Any]:
    """Basic mesh cleanup: weld vertices, drop degenerate/duplicate faces,
    fix winding/normals, optionally fill small holes.

    Multi-part scenes are repaired PART BY PART — node structure, part names,
    and per-part materials are preserved (flattening a scene is not "repair").
    Returns before/after structural counts so callers can see what changed.
    This is conservative cleanup, not remeshing — badly broken geometry stays
    broken (and the report says so via `is_watertight`).
    """
    trimesh = _import_trimesh()
    p = _resolve_input_path(input_path)
    out = _resolve_output_path(output_path or p, default_reference=p)

    # Load unprocessed so `before` reports the file's true counts — trimesh's
    # default load processing would silently weld vertices before we measure.
    loaded = _load_file(p, process=False)
    _require_triangle_geometry(loaded, source_name=p.name, operation="repair")

    if isinstance(loaded, trimesh.Scene):
        meshes = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        target: Any = loaded
        if out.suffix.lower() not in {".glb", ".obj"}:
            # Single-mesh formats cannot carry scene structure; flatten
            # explicitly (stated in the report via `scene: false`).
            target = as_single_mesh(loaded)
            meshes = [target]
    else:
        target = loaded
        meshes = [target]

    def _counts() -> Dict[str, Any]:
        return {
            "vertex_count": int(sum(len(m.vertices) for m in meshes)),
            "face_count": int(sum(len(m.faces) for m in meshes)),
            "is_watertight": bool(all(m.is_watertight for m in meshes)),
        }

    before = _counts()
    for mesh in meshes:
        _repair_one_mesh(trimesh, mesh, fill_holes=fill_holes)
    after = _counts()

    target.export(str(out))
    return {
        "input_path": str(p),
        "output_path": str(out),
        "scene": bool(isinstance(target, trimesh.Scene)),
        "part_count": len(meshes),
        "before": before,
        "after": after,
        "filled_holes": bool(fill_holes),
    }


def report_to_json(report: MeshReport) -> str:
    """Serialize a MeshReport to a compact JSON string."""
    return json.dumps(report.to_dict(), sort_keys=True)


def render_preview(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Render turntable preview views (implementation: `mesh_preview`)."""
    from .mesh_preview import render_preview as _render_preview  # noqa: PLC0415

    return _render_preview(*args, **kwargs)


__all__ = [
    "MeshReport",
    "SUPPORTED_EXPORT_SUFFIXES",
    "SUPPORTED_LOAD_SUFFIXES",
    "analyze_mesh",
    "as_single_mesh",
    "compose_scene",
    "convert_mesh",
    "load_mesh",
    "render_preview",
    "repair_mesh",
    "report_to_json",
    "transform_mesh",
]
