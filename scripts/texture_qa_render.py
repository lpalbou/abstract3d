"""Rendering and geometry support for the close-range texture QA harness.

This module owns everything that touches the GPU or the mesh:

* raw GLB parsing (material truth must come from the shipped bytes, not from
  a loader that normalizes or repairs materials),
* UV-space rasterization of per-texel world positions and normals (the texel
  maps every texture-space detector needs),
* reconstruction of per-view observed-texel masks by replicating the bake
  projector's facing + splatted z-buffer visibility semantics
  (`_tripo_project_observed_texture`), so the harness can derive the
  observed/fill split from bundle metadata instead of hardcoding regions,
* mesh concavity probes (highest-curvature concave clusters: eye sockets,
  intakes) via a 1-ring shape-operator estimate,
* a viewer-truth renderer that applies the exported baseColorFactor to the
  texture the way any spec-compliant PBR viewer (MeshVault) does, with
  orthographic zoom crops for close-range inspection and a nearest-sampled
  region-mask pass so screen pixels can be attributed to observed / symmetry
  / synthesized-fill texels.

Coordinate conventions follow the repository renderer: z-up, azimuth 0 looks
from +x, texture row r maps to v = 1 - r/(H-1) (PIL top row = v 1).
"""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

REPO_SRC = str(Path(__file__).resolve().parents[1] / "src")


def _ensure_repo_src() -> None:
    import sys

    if REPO_SRC not in sys.path:
        sys.path.insert(0, REPO_SRC)


# ---------------------------------------------------------------------------
# raw GLB parsing (material truth)
# ---------------------------------------------------------------------------

def parse_glb(path: Path) -> Tuple[dict, bytes]:
    """Return (gltf_json, binary_chunk) from the raw GLB bytes."""
    data = Path(path).read_bytes()
    if data[:4] != b"glTF":
        raise ValueError(f"{path} is not a GLB container")
    offset = 12
    gltf_json: Optional[dict] = None
    binary = b""
    while offset + 8 <= len(data):
        chunk_len, chunk_type = struct.unpack_from("<II", data, offset)
        chunk = data[offset + 8 : offset + 8 + chunk_len]
        if chunk_type == 0x4E4F534A:  # JSON
            gltf_json = json.loads(chunk)
        elif chunk_type == 0x004E4942:  # BIN
            binary = chunk
        offset += 8 + chunk_len + ((4 - chunk_len % 4) % 4 if chunk_len % 4 else 0)
    if gltf_json is None:
        raise ValueError(f"{path}: GLB carries no JSON chunk")
    return gltf_json, binary


def glb_image_bytes(gltf: dict, binary: bytes, image_index: int) -> Optional[bytes]:
    images = gltf.get("images") or []
    if image_index >= len(images):
        return None
    image = images[image_index]
    view_index = image.get("bufferView")
    if view_index is None:
        return None
    view = (gltf.get("bufferViews") or [])[view_index]
    start = int(view.get("byteOffset", 0))
    return binary[start : start + int(view["byteLength"])]


# ---------------------------------------------------------------------------
# texel maps: per-texel world position / normal in texture pixel grid
# ---------------------------------------------------------------------------

@dataclass
class TexelMaps:
    positions: np.ndarray  # (H, W, 3) float32 world positions
    normals: np.ndarray    # (H, W, 3) float32 unit normals
    surface: np.ndarray    # (H, W) bool: texel covered by a UV chart
    resolution: int

    @property
    def diagonal(self) -> float:
        pts = self.positions[self.surface]
        if len(pts) == 0:
            return 1.0
        return float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))


def build_texel_maps(mesh, resolution: int) -> TexelMaps:
    """Rasterize world positions and normals into the mesh's own UV layout.

    Rendering happens in UV space (position = UV mapped to NDC) with the
    world attributes as varyings; the vertical flip on readback matches the
    repository texture convention (row 0 = v 1), so these maps align
    pixel-for-pixel with the shipped texture.png / GLB texture.
    """
    import moderngl

    uv = np.asarray(mesh.visual.uv, dtype=np.float32)
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    normals = np.asarray(mesh.vertex_normals, dtype=np.float32)

    tri_uv = uv[faces].reshape(-1, 2)
    tri_pos = vertices[faces].reshape(-1, 3)
    tri_nrm = normals[faces].reshape(-1, 3)

    ctx = moderngl.create_context(standalone=True)
    try:
        program = ctx.program(
            vertex_shader="""
                #version 330
                in vec2 in_uv;
                in vec3 in_val;
                out vec3 v_val;
                void main() {
                    gl_Position = vec4(in_uv * 2.0 - 1.0, 0.0, 1.0);
                    v_val = in_val;
                }
            """,
            fragment_shader="""
                #version 330
                in vec3 v_val;
                out vec4 f_out;
                void main() { f_out = vec4(v_val, 1.0); }
            """,
        )
        target = ctx.texture((resolution, resolution), 4, dtype="f4")
        fbo = ctx.framebuffer(color_attachments=[target])
        fbo.use()
        outputs = []
        for values in (tri_pos, tri_nrm):
            packed = np.concatenate([tri_uv, values], axis=1).astype(np.float32)
            vbo = ctx.buffer(packed.tobytes())
            vao = ctx.vertex_array(program, [(vbo, "2f 3f", "in_uv", "in_val")])
            ctx.disable(moderngl.DEPTH_TEST)
            fbo.clear(0.0, 0.0, 0.0, 0.0)
            vao.render()
            raw = np.frombuffer(fbo.read(components=4, dtype="f4"), dtype=np.float32)
            grid = raw.reshape(resolution, resolution, 4)[::-1].copy()
            outputs.append(grid)
            vao.release()
            vbo.release()
        fbo.release()
        target.release()
        program.release()
    finally:
        ctx.release()

    positions, normals_map = outputs
    surface = positions[:, :, 3] > 0.0
    nrm = normals_map[:, :, :3]
    length = np.linalg.norm(nrm, axis=2, keepdims=True)
    nrm = np.divide(nrm, np.maximum(length, 1e-8))
    return TexelMaps(positions[:, :, :3].copy(), nrm, surface, resolution)


# ---------------------------------------------------------------------------
# per-view visibility (observed-mask reconstruction)
# ---------------------------------------------------------------------------

def camera_position(azimuth_deg: float, elevation_deg: float, distance: float) -> np.ndarray:
    az, el = math.radians(azimuth_deg), math.radians(elevation_deg)
    return np.array(
        [math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)],
        dtype=np.float32,
    ) * float(distance)


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    forward = target - eye
    forward = forward / max(float(np.linalg.norm(forward)), 1e-8)
    up = up / max(float(np.linalg.norm(up)), 1e-8)
    side = np.cross(forward, up)
    side = side / max(float(np.linalg.norm(side)), 1e-8)
    up = np.cross(side, forward)
    view = np.eye(4, dtype=np.float32)
    view[0, :3] = side
    view[1, :3] = up
    view[2, :3] = -forward
    view[:3, 3] = -view[:3, :3] @ eye
    return view


@dataclass
class ViewSpec:
    label: str
    azimuth_deg: float
    elevation_deg: float
    camera_distance: float
    projection_model: str  # "orthographic" | "perspective"
    ortho_half_extent: Optional[float] = None
    fovy_deg: float = 40.0
    alpha: Optional[np.ndarray] = None  # optional (H,W) photo alpha in [0,1]


def project_visibility(maps: TexelMaps, view: ViewSpec, *,
                       frame: int = 1024, facing_threshold: float = 0.2) -> np.ndarray:
    """Which surface texels does this camera actually see?

    Replicates the bake projector's semantics: facing test, frustum test,
    first-surface z-buffer built by splatting the projected texels themselves
    (3x3 min-filtered), and the slope-aware epsilon. When the view carries a
    photo alpha, texels projecting onto transparent photo pixels are excluded
    (they were never painted by that view).
    """
    from scipy.ndimage import minimum_filter

    height = width = int(frame)
    pos = maps.positions
    mask = maps.surface
    eye = camera_position(view.azimuth_deg, view.elevation_deg, view.camera_distance)
    to_cam = eye[None, None, :] - pos
    to_cam_n = np.linalg.norm(to_cam, axis=2, keepdims=True)
    facing = np.sum(maps.normals * np.divide(to_cam, np.maximum(to_cam_n, 1e-8)), axis=2)

    view_mat = look_at(eye, np.zeros(3, dtype=np.float32), np.array([0, 0, 1], np.float32))
    cam = pos @ view_mat[:3, :3].T + view_mat[:3, 3]
    x_cam, y_cam, z_cam = cam[:, :, 0], cam[:, :, 1], cam[:, :, 2]

    if view.projection_model == "orthographic":
        half = float(view.ortho_half_extent or 1.0)
        scale = 0.5 * height / max(half, 1e-6)
        sx = scale * x_cam + width / 2.0 - 0.5
        sy = -scale * y_cam + height / 2.0 - 0.5
        pixel_world = np.full_like(x_cam, 1.0 / scale)
    else:
        focal = 0.5 * height / math.tan(0.5 * math.radians(view.fovy_deg))
        depth = np.maximum(-z_cam, 1e-6)
        sx = focal * x_cam / depth + width / 2.0 - 0.5
        sy = -focal * y_cam / depth + height / 2.0 - 0.5
        pixel_world = depth / focal

    valid = (
        mask & (z_cam < -1e-4)
        & (sx >= 0.0) & (sx <= width - 1.0)
        & (sy >= 0.0) & (sy <= height - 1.0)
        & (facing > facing_threshold)
    )

    occluder = mask & (z_cam < -1e-4) & (sx >= -0.5) & (sx <= width - 0.5) \
        & (sy >= -0.5) & (sy <= height - 0.5)
    visible = valid.copy()
    if occluder.any():
        depth_world = -z_cam
        bx = np.clip(np.round(sx).astype(np.int32), 0, width - 1)
        by = np.clip(np.round(sy).astype(np.int32), 0, height - 1)
        nearest = np.full((height, width), np.inf, dtype=np.float32)
        np.minimum.at(nearest, (by[occluder], bx[occluder]), depth_world[occluder])
        nearest = minimum_filter(nearest, size=3, mode="nearest")
        slope = np.sqrt(np.clip(1.0 - facing**2, 0.0, 1.0)) / np.maximum(facing, 0.05)
        epsilon = 0.0025 * maps.diagonal + 2.5 * pixel_world * slope
        visible &= depth_world <= nearest[by, bx] + epsilon

    if view.alpha is not None and visible.any():
        alpha = view.alpha
        ah, aw = alpha.shape
        ax = np.clip(np.round(sx / max(width - 1, 1) * (aw - 1)).astype(np.int32), 0, aw - 1)
        ay = np.clip(np.round(sy / max(height - 1, 1) * (ah - 1)).astype(np.int32), 0, ah - 1)
        visible &= alpha[ay, ax] > 0.5
    return visible


# ---------------------------------------------------------------------------
# concavity probes (auto-derived defect-prone locations)
# ---------------------------------------------------------------------------

@dataclass
class Probe:
    kind: str
    position: np.ndarray  # world coordinates
    normal: np.ndarray    # outward viewing direction
    score: float
    label: str = ""

    def view_angles(self) -> Tuple[float, float]:
        n = self.normal / max(float(np.linalg.norm(self.normal)), 1e-8)
        azimuth = math.degrees(math.atan2(float(n[1]), float(n[0])))
        elevation = math.degrees(math.asin(float(np.clip(n[2], -1.0, 1.0))))
        return azimuth, float(np.clip(elevation, -60.0, 60.0))


def concavity_probes(mesh, *, max_probes: int = 5,
                     vertex_darkness: Optional[np.ndarray] = None) -> List[Probe]:
    """Concave high-curvature clusters: the eye-socket / intake class of
    location where fill smears and dark fragments concentrate.

    Uses the 1-ring shape operator sign: for edge (i, j),
    (n_j - n_i) . (p_j - p_i) < 0 means the surface bends toward the
    normals (concave). Per-vertex concavity is the negated mean over
    incident edges, graph-smoothed twice; probes are greedy non-max
    suppressed cluster peaks.

    With `vertex_darkness` (0..1 per vertex, e.g. sampled from the baked
    texture) a SECOND selection pass ranks by concavity x darkness, so
    concave regions carrying dark texture content (eye sockets with smears,
    shadowed intakes) are probed even when blander concavities (nape folds)
    dominate the pure-curvature ranking.
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    normals = np.asarray(mesh.vertex_normals, dtype=np.float64)
    edges = np.asarray(mesh.edges_unique, dtype=np.int64)
    dv = vertices[edges[:, 1]] - vertices[edges[:, 0]]
    dn = normals[edges[:, 1]] - normals[edges[:, 0]]
    k_edge = np.einsum("ij,ij->i", dn, dv) / np.maximum(
        np.einsum("ij,ij->i", dv, dv), 1e-12
    )
    concavity = np.zeros(len(vertices))
    counts = np.zeros(len(vertices))
    np.add.at(concavity, edges[:, 0], -k_edge)
    np.add.at(concavity, edges[:, 1], -k_edge)
    np.add.at(counts, edges[:, 0], 1.0)
    np.add.at(counts, edges[:, 1], 1.0)
    concavity /= np.maximum(counts, 1.0)
    for _ in range(2):  # graph smoothing: isolated noisy vertices are not probes
        smoothed = np.zeros_like(concavity)
        np.add.at(smoothed, edges[:, 0], concavity[edges[:, 1]])
        np.add.at(smoothed, edges[:, 1], concavity[edges[:, 0]])
        concavity = 0.5 * concavity + 0.5 * smoothed / np.maximum(counts, 1.0)

    diagonal = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    min_separation = 0.10 * diagonal
    probes: List[Probe] = []

    def select(score: np.ndarray, budget: int) -> None:
        order = np.argsort(-score)
        floor = float(np.percentile(score, 99.0))
        taken = 0
        for idx in order:
            if score[idx] <= max(floor, 0.0) or taken >= budget:
                break
            p = vertices[idx]
            if any(np.linalg.norm(p - pr.position) < min_separation for pr in probes):
                continue
            near = np.linalg.norm(vertices - p, axis=1) < 0.04 * diagonal
            normal = normals[near].mean(axis=0)
            normal /= max(float(np.linalg.norm(normal)), 1e-8)
            probes.append(Probe("concavity", p.astype(np.float32),
                                normal.astype(np.float32), float(score[idx])))
            taken += 1

    select(concavity, max_probes)
    if vertex_darkness is not None and len(vertex_darkness) == len(vertices):
        positive = np.clip(concavity, 0.0, None)
        select(positive * np.clip(vertex_darkness, 0.0, 1.0), max(2, max_probes // 2))
    for i, probe in enumerate(probes, 1):
        probe.label = f"concavity_{i:02d}"
    return probes


# ---------------------------------------------------------------------------
# viewer-truth renderer
# ---------------------------------------------------------------------------

class ViewerTruthRenderer:
    """Renders the shipped asset the way a spec-compliant PBR viewer shades
    its base color: texture * baseColorFactor. The repository preview
    renderer ignores the factor, which is exactly the flattery this class
    removes. Shading keeps the repository's flat-dominant textured model
    (0.88 + 0.12 diffuse) so texture content, not lighting, is measured.

    Also renders arbitrary substitute textures (region masks) with nearest
    sampling so screen pixels can be attributed to texel regions.
    """

    def __init__(self, mesh, texture: Image.Image,
                 base_color_factor: Sequence[float] = (1.0, 1.0, 1.0, 1.0)):
        import moderngl

        self._moderngl = moderngl
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.int32)
        self.center = 0.5 * (vertices.min(axis=0) + vertices.max(axis=0))
        centered = vertices - self.center
        self.radius = float(np.max(np.linalg.norm(centered, axis=1))) or 1.0
        centered = centered / self.radius
        normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
        uv = np.asarray(mesh.visual.uv, dtype=np.float32)

        tri = np.concatenate(
            [centered[faces].reshape(-1, 3), normals[faces].reshape(-1, 3),
             uv[faces].reshape(-1, 2)], axis=1
        ).astype(np.float32)
        self._centered = centered

        self.ctx = moderngl.create_context(standalone=True)
        self.program = self.ctx.program(
            vertex_shader="""
                #version 330
                in vec3 in_pos; in vec3 in_nrm; in vec2 in_uv;
                out vec3 v_nrm; out vec2 v_uv;
                uniform mat4 u_mvp;
                void main() {
                    gl_Position = u_mvp * vec4(in_pos, 1.0);
                    v_nrm = in_nrm; v_uv = in_uv;
                }
            """,
            fragment_shader="""
                #version 330
                in vec3 v_nrm; in vec2 v_uv;
                out vec4 f_color;
                uniform sampler2D u_tex;
                uniform vec3 u_factor;
                uniform vec3 u_light;
                uniform int u_flat;
                void main() {
                    vec3 base = texture(u_tex, vec2(v_uv.x, 1.0 - v_uv.y)).rgb;
                    base *= u_factor;
                    if (u_flat == 1) { f_color = vec4(base, 1.0); return; }
                    float diffuse = max(dot(normalize(v_nrm), normalize(u_light)), 0.0);
                    f_color = vec4(base * (0.88 + 0.12 * diffuse), 1.0);
                }
            """,
        )
        self.vbo = self.ctx.buffer(tri.tobytes())
        self.vao = self.ctx.vertex_array(
            self.program, [(self.vbo, "3f 3f 2f", "in_pos", "in_nrm", "in_uv")]
        )
        self.program["u_light"].value = (0.45, -0.35, 0.82)
        self.factor = tuple(float(c) for c in base_color_factor[:3])
        self._textures: Dict[str, object] = {}
        self.set_texture("base", texture, nearest=False)

    def set_texture(self, name: str, image: Image.Image, *, nearest: bool) -> None:
        moderngl = self._moderngl
        rgb = image.convert("RGB")
        tex = self.ctx.texture(rgb.size, 3, rgb.tobytes())
        if nearest:
            tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        else:
            tex.build_mipmaps()
            tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        self._textures[name] = tex

    def _view_projection(self, azimuth: float, elevation: float,
                         target_world: Optional[np.ndarray], zoom: float) -> np.ndarray:
        target = np.zeros(3, dtype=np.float32)
        if target_world is not None:
            target = ((np.asarray(target_world, dtype=np.float32) - self.center)
                      / self.radius)
        eye = target + camera_position(azimuth, elevation, 3.2)
        view = look_at(eye, target, np.array([0, 0, 1], np.float32))
        cam = self._centered @ view[:3, :3].T + view[:3, 3]
        half = float(np.max(np.abs(cam[:, :2]))) * 1.18 / max(zoom, 1e-6)
        projection = np.eye(4, dtype=np.float32)
        projection[0, 0] = 1.0 / max(half, 1e-6)
        projection[1, 1] = 1.0 / max(half, 1e-6)
        projection[2, 2] = -2.0 / 15.9
        projection[2, 3] = -16.1 / 15.9
        return projection @ view

    def render(self, azimuth: float, elevation: float, *, size: int = 896,
               texture: str = "base", apply_factor: bool = True,
               target_world: Optional[np.ndarray] = None, zoom: float = 1.0,
               flat: bool = False,
               background: Tuple[float, float, float] = (0.95, 0.95, 0.93)) -> np.ndarray:
        moderngl = self._moderngl
        render_size = int(size) * 2
        fbo = self.ctx.simple_framebuffer((render_size, render_size), components=4)
        fbo.use()
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.CULL_FACE)
        fbo.clear(*background, 1.0)
        self._textures[texture].use(0)
        self.program["u_factor"].value = self.factor if apply_factor else (1.0, 1.0, 1.0)
        self.program["u_flat"].value = 1 if flat else 0
        mvp = self._view_projection(azimuth, elevation, target_world, zoom)
        self.program["u_mvp"].write(mvp.astype(np.float32).T.tobytes())
        self.vao.render()
        image = Image.frombytes("RGBA", fbo.size, fbo.read(components=4),
                                "raw", "RGBA", 0, -1).convert("RGB")
        fbo.release()
        resample = Image.Resampling.NEAREST if flat else Image.Resampling.LANCZOS
        if render_size != size:
            image = image.resize((size, size), resample)
        return np.asarray(image)

    def release(self) -> None:
        for tex in self._textures.values():
            tex.release()
        self.vao.release()
        self.vbo.release()
        self.program.release()
        self.ctx.release()


# ---------------------------------------------------------------------------
# bundle loading helpers
# ---------------------------------------------------------------------------

@dataclass
class Bundle:
    directory: Path
    mesh: object
    metadata: dict
    texture: Image.Image           # texture actually shipped inside the GLB
    gltf: dict
    binary: bytes = field(repr=False, default=b"")

    @property
    def texture_array(self) -> np.ndarray:
        return np.asarray(self.texture.convert("RGB"))


def load_bundle(directory: Path) -> Bundle:
    _ensure_repo_src()
    import trimesh

    directory = Path(directory)
    gltf, binary = parse_glb(directory / "scene.glb")
    mesh = trimesh.load(directory / "scene.glb", force="mesh")
    # Exports present the glTF viewer frame (Y-up / front +Z) and carry a
    # persisted marker; this harness's probe/camera math is canonical-frame
    # (Z-up / front +X), so marked bundles are rotated back on load.
    if mesh.metadata.get("abstract3d_export_frame") == "gltf_yup_front_pz":
        mesh = mesh.copy()
        mesh.apply_transform(
            np.array(
                [
                    [0.0, 0.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
        )
    metadata = {}
    meta_path = directory / "metadata.json"
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text())

    texture: Optional[Image.Image] = None
    materials = gltf.get("materials") or []
    if materials:
        pbr = materials[0].get("pbrMetallicRoughness") or {}
        tex_info = pbr.get("baseColorTexture")
        if tex_info is not None:
            textures = gltf.get("textures") or []
            source = textures[int(tex_info.get("index", 0))].get("source")
            payload = glb_image_bytes(gltf, binary, int(source))
            if payload:
                import io

                texture = Image.open(io.BytesIO(payload)).convert("RGB")
    if texture is None:  # fall back to the sidecar so detectors can still run
        sidecar = directory / "texture.png"
        if sidecar.exists():
            texture = Image.open(sidecar).convert("RGB")
        else:
            raise FileNotFoundError(f"{directory}: no texture in GLB nor texture.png")
    return Bundle(directory, mesh, metadata, texture, gltf, binary)
