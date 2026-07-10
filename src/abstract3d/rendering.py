"""Preview rendering and contact-sheet helpers for generated meshes."""

from __future__ import annotations

import io
import math
from typing import Iterable, List, Sequence

import numpy as np
from PIL import Image, ImageDraw

_LAST_RENDERER_BACKEND: str | None = None


def _material_base_color_factor(mesh) -> np.ndarray:
    """Return the material's base color factor as linear RGB in 0-1.

    Spec viewers multiply the base color texture by ``baseColorFactor``
    (glTF) or ``Kd`` (OBJ). Previews must apply the same factor, otherwise a
    defective material (e.g. trimesh's 0.4 gray SimpleMaterial default)
    renders correctly in-repo while every external viewer shows a darkened
    asset. PBR materials expose ``baseColorFactor``; simple materials expose
    the equivalent ``diffuse`` color, which trimesh copies into
    ``baseColorFactor`` on GLB export.
    """
    material = getattr(getattr(mesh, "visual", None), "material", None)
    factor = getattr(material, "baseColorFactor", None) if material is not None else None
    if factor is None:
        factor = getattr(material, "diffuse", None) if material is not None else None
    if factor is None:
        return np.ones(3, dtype=np.float32)
    raw = np.asarray(factor).reshape(-1)
    if raw.size < 3:
        return np.ones(3, dtype=np.float32)
    rgb = raw[:3].astype(np.float32)
    # Integer factors follow trimesh's 0-255 convention, float factors the
    # glTF 0-1 convention.
    if raw.dtype.kind in "ui" or float(rgb.max(initial=0.0)) > 1.0:
        rgb = rgb / 255.0
    return np.clip(rgb, 0.0, 1.0)


def _sample_texture_vertex_colors(mesh) -> np.ndarray | None:
    visual = getattr(mesh, "visual", None)
    if visual is None:
        return None
    uv = getattr(visual, "uv", None)
    if uv is None:
        return None
    material = getattr(visual, "material", None)
    image = getattr(material, "image", None) if material is not None else None
    if image is None:
        image = getattr(material, "baseColorTexture", None) if material is not None else None
    if image is None:
        image = getattr(visual, "image", None)
    if image is None:
        return None
    uv = np.asarray(uv, dtype=np.float32)
    if len(uv) != len(mesh.vertices):
        return None
    if isinstance(image, (bytes, bytearray)):
        # GLB materials can carry the texture as raw encoded bytes.
        try:
            import io

            image = Image.open(io.BytesIO(bytes(image)))
        except Exception:
            return None
    if isinstance(image, Image.Image):
        image_np = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    else:
        try:
            image_np = np.asarray(image, dtype=np.float32)
        except Exception:
            return None
        if image_np.ndim != 3 or image_np.shape[2] < 3:
            return None
        if image_np.max() > 1.0:
            image_np = image_np / 255.0
        image_np = image_np[:, :, :3]
    height, width = image_np.shape[:2]
    xs = np.clip(np.rint(uv[:, 0] * (width - 1)), 0, width - 1).astype(np.int32)
    ys = np.clip(np.rint((1.0 - uv[:, 1]) * (height - 1)), 0, height - 1).astype(np.int32)
    # Match spec-viewer shading: texel color is modulated by the material's
    # base color factor.
    return image_np[ys, xs] * _material_base_color_factor(mesh)[None, :]


def _look_at_matrix(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
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


def _orthographic_projection(half_extent: float, *, near: float = 0.1, far: float = 16.0) -> np.ndarray:
    extent = max(float(half_extent), 1e-3)
    projection = np.eye(4, dtype=np.float32)
    projection[0, 0] = 1.0 / extent
    projection[1, 1] = 1.0 / extent
    projection[2, 2] = -2.0 / max(far - near, 1e-6)
    projection[2, 3] = -(far + near) / max(far - near, 1e-6)
    return projection


def _mesh_vertex_colors(mesh, vertex_count: int) -> np.ndarray:
    visual = getattr(mesh, "visual", None)
    vertex_colors = getattr(visual, "vertex_colors", None) if visual is not None else None
    if not isinstance(vertex_colors, np.ndarray) or len(vertex_colors) != vertex_count:
        return np.full((vertex_count, 3), 0.72, dtype=np.float32)
    colors = vertex_colors[:, :3].astype(np.float32)
    if colors.max(initial=0.0) > 1.0:
        colors = colors / 255.0
    return np.clip(colors, 0.0, 1.0)


def _mesh_texture_image(mesh) -> Image.Image | None:
    visual = getattr(mesh, "visual", None)
    if visual is None:
        return None
    material = getattr(visual, "material", None)
    image = getattr(material, "image", None) if material is not None else None
    if image is None:
        image = getattr(material, "baseColorTexture", None) if material is not None else None
    if image is None:
        image = getattr(visual, "image", None)
    if image is None:
        return None
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] < 3:
        return None
    if array.dtype != np.uint8:
        array = np.clip(array, 0.0, 1.0)
        array = (array * 255.0).astype(np.uint8)
    return Image.fromarray(array[:, :, :3], mode="RGB")


def _create_standalone_context(moderngl_module, *, attempts: int = 3):
    """Create a standalone GL context, retrying on transient failures.

    On macOS, CGL context creation can fail or segfault-adjacent error when
    another process is churning GL contexts at the same moment (observed
    with a concurrently running GL test suite / viewer backend). A short
    backoff-and-retry absorbs the transient contention; a persistent
    failure still raises the last error.
    """

    import time as _time

    last_error: Exception | None = None
    for attempt in range(int(attempts)):
        try:
            return moderngl_module.create_context(standalone=True)
        except Exception as exc:  # glcontext raises plain Exception subclasses
            last_error = exc
            _time.sleep(0.2 * (attempt + 1))
    raise last_error  # type: ignore[misc]


def _render_mesh_views_moderngl(
    mesh,
    *,
    size: int,
    azimuths: Sequence[float],
    elevation: float,
) -> List[Image.Image]:
    import moderngl

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    if len(vertices) == 0 or len(faces) == 0:
        return []

    center = 0.5 * (vertices.min(axis=0) + vertices.max(axis=0))
    centered = vertices - center
    radius = float(np.max(np.linalg.norm(centered, axis=1))) or 1.0
    centered = centered / radius

    normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
    if normals.shape != centered.shape:
        normals = np.zeros_like(centered)
        triangle_normals = np.cross(
            centered[faces[:, 1]] - centered[faces[:, 0]],
            centered[faces[:, 2]] - centered[faces[:, 0]],
        )
        lengths = np.linalg.norm(triangle_normals, axis=1, keepdims=True)
        triangle_normals = np.divide(triangle_normals, np.maximum(lengths, 1e-8))
        np.add.at(normals, faces[:, 0], triangle_normals)
        np.add.at(normals, faces[:, 1], triangle_normals)
        np.add.at(normals, faces[:, 2], triangle_normals)
        normal_lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = np.divide(normals, np.maximum(normal_lengths, 1e-8))

    vertex_colors = _mesh_vertex_colors(mesh, len(centered))
    visual = getattr(mesh, "visual", None)
    uv = getattr(visual, "uv", None) if visual is not None else None
    has_texture = isinstance(uv, np.ndarray) and len(uv) == len(centered) and _mesh_texture_image(mesh) is not None

    tri_positions = centered[faces].reshape(-1, 3).astype(np.float32)
    tri_normals = normals[faces].reshape(-1, 3).astype(np.float32)
    tri_colors = vertex_colors[faces].reshape(-1, 3).astype(np.float32)
    tri_uv = np.zeros((len(tri_positions), 2), dtype=np.float32)
    if has_texture:
        tri_uv = np.asarray(uv, dtype=np.float32)[faces].reshape(-1, 2).astype(np.float32)

    packed = np.concatenate([tri_positions, tri_normals, tri_colors, tri_uv], axis=1).astype(np.float32)
    render_size = max(int(size), 64) * 2
    ctx = _create_standalone_context(moderngl)
    program = ctx.program(
        vertex_shader="""
            #version 330
            in vec3 in_pos;
            in vec3 in_nrm;
            in vec3 in_color;
            in vec2 in_uv;
            out vec3 v_nrm;
            out vec3 v_color;
            out vec2 v_uv;
            uniform mat4 u_mvp;
            void main() {
                gl_Position = u_mvp * vec4(in_pos, 1.0);
                v_nrm = in_nrm;
                v_color = in_color;
                v_uv = in_uv;
            }
        """,
        fragment_shader="""
            #version 330
            in vec3 v_nrm;
            in vec3 v_color;
            in vec2 v_uv;
            out vec4 f_color;
            uniform sampler2D u_tex;
            uniform int u_use_texture;
            uniform vec3 u_base_color_factor;
            uniform vec3 u_light;
            uniform vec3 u_view_dir;
            void main() {
                vec3 base = v_color;
                if (u_use_texture == 1) {
                    // Spec viewers multiply the base color texture by the
                    // material's baseColorFactor; previews must match so a
                    // bad factor is visible here before anyone ships it.
                    base = texture(u_tex, vec2(v_uv.x, 1.0 - v_uv.y)).rgb * u_base_color_factor;
                }
                vec3 normal = normalize(v_nrm);
                float diffuse = max(dot(normal, normalize(u_light)), 0.0);
                // Textured previews review the baked albedo, so shading must
                // stay close to flat: strong ridge shading re-draws the
                // mesh's own geometric features (eye sockets, brows, lips)
                // over the photo albedo and reads as a ghosted second face
                // whenever geometry and texture disagree by even a few
                // percent. A 12% diffuse cue keeps just enough depth to see
                // silhouettes. Untextured geometry keeps the strong single
                // key light so shape readability stays high.
                float shade;
                if (u_use_texture == 1) {
                    shade = 0.88 + 0.12 * diffuse;
                } else {
                    shade = 0.24 + 0.76 * diffuse;
                }
                f_color = vec4(base * shade, 1.0);
            }
        """,
    )
    vertex_buffer = ctx.buffer(packed.tobytes())
    vao = ctx.vertex_array(
        program,
        [(vertex_buffer, "3f 3f 3f 2f", "in_pos", "in_nrm", "in_color", "in_uv")],
    )
    framebuffer = ctx.simple_framebuffer((render_size, render_size), components=4)
    texture = None
    dummy_texture = ctx.texture((1, 1), 3, bytes([255, 255, 255]))
    dummy_texture.use(0)
    if has_texture:
        texture_image = _mesh_texture_image(mesh)
        if texture_image is not None:
            texture = ctx.texture(texture_image.size, 3, texture_image.tobytes())
            texture.build_mipmaps()
            texture.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
            texture.use(0)
    program["u_use_texture"].value = 1 if texture is not None else 0
    base_color_factor = _material_base_color_factor(mesh)
    if "u_base_color_factor" in program:
        program["u_base_color_factor"].value = (
            float(base_color_factor[0]),
            float(base_color_factor[1]),
            float(base_color_factor[2]),
        )
    program["u_light"].value = (0.45, -0.35, 0.82)

    images: List[Image.Image] = []
    target = np.zeros(3, dtype=np.float32)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    camera_distance = 3.2
    for azimuth in azimuths:
        azimuth_rad = math.radians(float(azimuth))
        elevation_rad = math.radians(float(elevation))
        eye = np.array(
            [
                math.cos(elevation_rad) * math.cos(azimuth_rad),
                math.cos(elevation_rad) * math.sin(azimuth_rad),
                math.sin(elevation_rad),
            ],
            dtype=np.float32,
        ) * camera_distance
        view = _look_at_matrix(eye, target, up)
        camera_space = centered @ view[:3, :3].T + view[:3, 3]
        half_extent = float(np.max(np.abs(camera_space[:, :2]))) * 1.18
        projection = _orthographic_projection(half_extent)
        mvp = projection @ view
        eye_norm = eye / max(float(np.linalg.norm(eye)), 1e-8)
        # The GLSL compiler eliminates uniforms that end up unused (e.g.
        # u_view_dir when the textured branch needs no camera fill term);
        # writing an eliminated uniform raises, so guard by membership.
        if "u_view_dir" in program:
            program["u_view_dir"].value = (float(eye_norm[0]), float(eye_norm[1]), float(eye_norm[2]))

        framebuffer.use()
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.disable(moderngl.CULL_FACE)
        ctx.clear(0.95, 0.95, 0.93, 1.0)
        program["u_mvp"].write(mvp.astype(np.float32).T.tobytes())
        vao.render()

        rendered = Image.frombytes(
            "RGBA",
            framebuffer.size,
            framebuffer.read(components=4),
            "raw",
            "RGBA",
            0,
            -1,
        ).convert("RGB")
        if render_size != size:
            rendered = rendered.resize((size, size), Image.Resampling.LANCZOS)
        images.append(rendered)

    vao.release()
    vertex_buffer.release()
    program.release()
    framebuffer.release()
    if texture is not None:
        texture.release()
    dummy_texture.release()
    ctx.release()
    return images


def _render_mesh_views_matplotlib(mesh, *, size: int = 420, azimuths: Sequence[float] = (35.0, 125.0, 215.0, 305.0), elevation: float = 20.0) -> List[Image.Image]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    centered = vertices - vertices.mean(axis=0, keepdims=True)
    scale = float(np.max(np.linalg.norm(centered, axis=1))) or 1.0
    centered = centered / scale
    use_sampled_faces = len(faces) > 300000

    facecolors = np.full((len(faces), 3), [0.56, 0.62, 0.72], dtype=np.float32)
    visual = getattr(mesh, "visual", None)
    vertex_colors = getattr(visual, "vertex_colors", None)
    if isinstance(vertex_colors, np.ndarray) and len(vertex_colors) == len(vertices):
        colors = vertex_colors[:, :3].astype(np.float32)
        if colors.max() > 1.0:
            colors = colors / 255.0
        facecolors = colors[faces].mean(axis=1)
    else:
        sampled_texture_colors = _sample_texture_vertex_colors(mesh)
        if isinstance(sampled_texture_colors, np.ndarray) and len(sampled_texture_colors) == len(vertices):
            facecolors = sampled_texture_colors[faces].mean(axis=1)

    if use_sampled_faces:
        step = max(1, len(faces) // 120000)
        preview_faces = faces[::step]
        preview_facecolors = facecolors[::step]
    else:
        preview_faces = faces
        preview_facecolors = facecolors

    preview_vertices = centered[preview_faces]
    normals = np.cross(
        preview_vertices[:, 1] - preview_vertices[:, 0],
        preview_vertices[:, 2] - preview_vertices[:, 0],
    )
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, np.maximum(lengths, 1e-8))
    light = np.array([0.4, -0.3, 0.85], dtype=np.float32)
    light = light / np.linalg.norm(light)
    intensity = 0.2 + 0.8 * np.maximum(normals @ light, 0.0)
    shaded_facecolors = np.clip(preview_facecolors * intensity[:, None], 0.0, 1.0)

    images: List[Image.Image] = []
    for azimuth in azimuths:
        fig = plt.figure(figsize=(size / 100.0, size / 100.0), dpi=160)
        ax = fig.add_subplot(111, projection="3d")
        surface = Poly3DCollection(
            preview_vertices,
            facecolors=shaded_facecolors,
            edgecolors="none",
            linewidths=0.0,
        )
        ax.add_collection3d(surface)
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-1.0, 1.0)
        ax.set_zlim(-1.0, 1.0)
        ax.view_init(elev=elevation, azim=float(azimuth))
        ax.set_axis_off()
        ax.set_box_aspect((1.0, 1.0, 1.0))
        if hasattr(ax, "set_proj_type"):
            ax.set_proj_type("ortho")
        fig.tight_layout(pad=0)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", transparent=False, facecolor="#f3f2ee")
        plt.close(fig)
        buf.seek(0)
        rendered = Image.open(buf).convert("RGB")
        # figsize * dpi does not equal the requested pixel size (dpi 160 vs
        # the /100 figsize divisor), so normalize explicitly to honor the
        # size contract that the ModernGL path already meets.
        if rendered.size != (int(size), int(size)):
            rendered = rendered.resize((int(size), int(size)), Image.LANCZOS)
        images.append(rendered)
    return images


def render_mesh_views(mesh, *, size: int = 420, azimuths: Sequence[float] = (35.0, 125.0, 215.0, 305.0), elevation: float = 20.0) -> List[Image.Image]:
    global _LAST_RENDERER_BACKEND
    _LAST_RENDERER_BACKEND = None
    # Exports present the glTF viewer frame (Y-up / front +Z) and carry a
    # persisted marker; this renderer's camera math is canonical-frame
    # (Z-up / front +X), so marked meshes are rotated back before rendering.
    # Without this, every render of a re-loaded exported scene.glb would be
    # lying sideways (and every harness comparing against photos would
    # silently measure a rotated subject).
    try:
        marker = getattr(mesh, "metadata", {}).get("abstract3d_export_frame")
    except Exception:
        marker = None
    if marker == "gltf_yup_front_pz":
        import numpy as np

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
    try:
        images = _render_mesh_views_moderngl(mesh, size=size, azimuths=azimuths, elevation=elevation)
        if images:
            _LAST_RENDERER_BACKEND = "moderngl"
            return images
    except Exception:
        pass
    images = _render_mesh_views_matplotlib(mesh, size=size, azimuths=azimuths, elevation=elevation)
    _LAST_RENDERER_BACKEND = "matplotlib"
    return images


def get_last_render_backend() -> str | None:
    return _LAST_RENDERER_BACKEND


def build_case_contact_sheet(
    *,
    title: str,
    source_image: Image.Image,
    views: Sequence[Image.Image],
    stats_lines: Iterable[str],
    panel_size: int = 320,
) -> Image.Image:
    background = Image.new("RGB", (panel_size * 3, panel_size * 2 + 110), "#f3f2ee")
    source = source_image.convert("RGB").copy()
    source.thumbnail((panel_size, panel_size))
    top_left = Image.new("RGB", (panel_size, panel_size), "#ebe7df")
    top_left.paste(
        source,
        ((panel_size - source.width) // 2, (panel_size - source.height) // 2),
    )
    background.paste(top_left, (0, 0))
    for index, image in enumerate(list(views)[:5], start=1):
        tile = image.convert("RGB").copy()
        tile.thumbnail((panel_size, panel_size))
        canvas = Image.new("RGB", (panel_size, panel_size), "#ebe7df")
        canvas.paste(tile, ((panel_size - tile.width) // 2, (panel_size - tile.height) // 2))
        col = index % 3
        row = 0 if index < 3 else 1
        background.paste(canvas, (col * panel_size, row * panel_size))
    draw = ImageDraw.Draw(background)
    draw.rectangle((0, panel_size * 2, panel_size * 3, panel_size * 2 + 110), fill="#1f2329")
    draw.text((18, panel_size * 2 + 14), str(title), fill="#f7f6f2")
    y = panel_size * 2 + 44
    for line in stats_lines:
        draw.text((18, y), str(line), fill="#d7dde5")
        y += 18
    return background


def stack_contact_sheets(items: Sequence[Image.Image], *, columns: int = 1, gutter: int = 18, background: str = "#ddd7ca") -> Image.Image:
    if not items:
        raise ValueError("At least one contact sheet is required.")
    cols = max(1, int(columns))
    rows = (len(items) + cols - 1) // cols
    max_width = max(item.width for item in items)
    max_height = max(item.height for item in items)
    sheet = Image.new(
        "RGB",
        (cols * max_width + (cols + 1) * gutter, rows * max_height + (rows + 1) * gutter),
        background,
    )
    for index, item in enumerate(items):
        row = index // cols
        col = index % cols
        x = gutter + col * (max_width + gutter)
        y = gutter + row * (max_height + gutter)
        sheet.paste(item, (x, y))
    return sheet
