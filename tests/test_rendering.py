from __future__ import annotations

import io

import numpy as np
import pytest
import trimesh
from PIL import Image
from trimesh.visual import TextureVisuals
from trimesh.visual.material import SimpleMaterial

from abstract3d.rendering import get_last_render_backend, render_mesh_views


def _triangle_mesh() -> trimesh.Trimesh:
    vertices = np.array(
        [
            [-0.7, -0.6, 0.0],
            [0.7, -0.6, 0.0],
            [0.0, 0.8, 0.0],
        ],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def test_render_mesh_views_handles_vertex_colors() -> None:
    mesh = _triangle_mesh()
    mesh.visual.vertex_colors = np.array(
        [
            [240, 60, 60, 255],
            [60, 240, 60, 255],
            [60, 60, 240, 255],
        ],
        dtype=np.uint8,
    )
    try:
        images = render_mesh_views(mesh, size=180, azimuths=(35.0,), elevation=20.0)
    except Exception as exc:  # pragma: no cover - environment specific
        pytest.skip(f"Preview renderer unavailable in this environment: {exc}")
    assert len(images) == 1
    assert images[0].size == (180, 180)
    assert get_last_render_backend() in {"moderngl", "matplotlib"}


def test_render_mesh_views_handles_uv_texture() -> None:
    mesh = _triangle_mesh()
    texture = Image.new("RGB", (16, 16), (220, 40, 40))
    payload = io.BytesIO()
    texture.save(payload, format="PNG")
    payload.seek(0)
    mesh.visual = TextureVisuals(
        uv=np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]], dtype=np.float32),
        image=texture,
        material=SimpleMaterial(image=payload.getvalue()),
    )
    try:
        images = render_mesh_views(mesh, size=180, azimuths=(35.0,), elevation=20.0)
    except Exception as exc:  # pragma: no cover - environment specific
        pytest.skip(f"Preview renderer unavailable in this environment: {exc}")
    assert len(images) == 1
    assert images[0].size == (180, 180)
    assert get_last_render_backend() in {"moderngl", "matplotlib"}
