"""Regression tests for textured-export material factors.

Baked-texture exports must render the texture as authored in spec-compliant
viewers: every factor multiplied on top of the base color texture has to be
identity. These tests exercise the repo's real export helpers (no GPU) and
assert the GLB JSON factors, the MTL lines, and that previews honor the
material factor instead of masking defects.
"""

from __future__ import annotations

import importlib.util
import io
import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh
from PIL import Image
from trimesh.visual.material import PBRMaterial, SimpleMaterial
from trimesh.visual.texture import TextureVisuals

from abstract3d.backends.triposr_runtime import (
    _mesh_export_bytes,
    _tripo_build_textured_mesh,
    _tripo_export_obj_with_textures,
)
from abstract3d.rendering import (
    _material_base_color_factor,
    _sample_texture_vertex_colors,
    render_mesh_views,
)

_CHECKER_PATH = Path(__file__).resolve().parents[1] / "scripts" / "check_export_materials.py"
_SPEC = importlib.util.spec_from_file_location("check_export_materials", _CHECKER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
checker = importlib.util.module_from_spec(_SPEC)
# Dataclasses resolve postponed annotations through sys.modules, so the
# module must be registered before execution.
sys.modules[_SPEC.name] = checker
_SPEC.loader.exec_module(checker)

_TEXTURE_COLOR = (200, 50, 25)


def _textured_quad_mesh(texture_color=_TEXTURE_COLOR) -> trimesh.Trimesh:
    """Build a tiny textured mesh through the repo's real bake assembler."""
    base = trimesh.Trimesh(
        vertices=np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32),
        faces=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
        process=False,
    )
    return _tripo_build_textured_mesh(
        base,
        bake_output={
            "vmapping": np.arange(4, dtype=np.int64),
            "indices": base.faces,
            "uvs": np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32),
        },
        texture_image=Image.new("RGB", (8, 8), texture_color),
    )


def test_textured_glb_carries_identity_material_factors(tmp_path) -> None:
    glb_bytes = _mesh_export_bytes(_textured_quad_mesh(), file_type="glb")
    gltf = checker.parse_glb_json(glb_bytes)
    materials = gltf.get("materials", [])
    assert len(materials) == 1
    pbr = materials[0]["pbrMetallicRoughness"]
    assert "baseColorTexture" in pbr
    assert pbr.get("baseColorFactor", [1.0, 1.0, 1.0, 1.0]) == [1.0, 1.0, 1.0, 1.0]
    # metallicFactor must be explicit: the glTF default is 1.0 (fully metal).
    assert pbr["metallicFactor"] == 0.0
    assert 0.85 <= pbr.get("roughnessFactor", 1.0) <= 1.0
    assert materials[0].get("emissiveFactor", [0.0, 0.0, 0.0]) == [0.0, 0.0, 0.0]

    glb_path = tmp_path / "scene.glb"
    glb_path.write_bytes(glb_bytes)
    reports = checker.audit_glb_materials(glb_path)
    assert reports and all(not report.violations for report in reports)


def test_textured_obj_mtl_carries_identity_phong_factors(tmp_path) -> None:
    mesh = _textured_quad_mesh()
    obj_bytes, sidecars = _tripo_export_obj_with_textures(mesh)

    assert b"usemtl material_0" in obj_bytes
    mtl_text = sidecars["scene.mtl"].decode("utf-8")
    assert "Ka 1.00000000 1.00000000 1.00000000" in mtl_text
    assert "Kd 1.00000000 1.00000000 1.00000000" in mtl_text
    # Baked albedo has no authored specular; a non-zero Ks would add a
    # synthetic sheen on top of photo-derived colors in Phong viewers.
    assert "Ks 0.00000000 0.00000000 0.00000000" in mtl_text
    assert "map_Kd material_0.png" in mtl_text

    # The texture sidecar must be the baked atlas, bit-for-bit in pixels.
    sidecar_image = Image.open(io.BytesIO(sidecars["material_0.png"])).convert("RGB")
    expected = np.full((8, 8, 3), _TEXTURE_COLOR, dtype=np.uint8)
    assert np.array_equal(np.asarray(sidecar_image), expected)

    # Export must not permanently mutate the mesh's PBR material.
    assert isinstance(mesh.visual.material, PBRMaterial)
    assert mesh.visual.material.metallicFactor == 0.0

    mtl_path = tmp_path / "scene.mtl"
    mtl_path.write_bytes(sidecars["scene.mtl"])
    reports = checker.audit_mtl_materials(mtl_path)
    assert reports and all(not report.violations for report in reports)


def test_untextured_exports_stay_unchanged() -> None:
    plain = trimesh.Trimesh(
        vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
        faces=np.array([[0, 1, 2]], dtype=np.int64),
        process=False,
    )
    glb_bytes = _mesh_export_bytes(plain, file_type="glb")
    gltf = checker.parse_glb_json(glb_bytes)
    for material in gltf.get("materials", []):
        assert "baseColorTexture" not in material.get("pbrMetallicRoughness", {})
    obj_bytes = _mesh_export_bytes(plain, file_type="obj")
    assert b"usemtl" not in obj_bytes


def test_material_base_color_factor_matches_spec_viewer_semantics() -> None:
    textured = _textured_quad_mesh()
    assert np.allclose(_material_base_color_factor(textured), [1.0, 1.0, 1.0])

    # trimesh's SimpleMaterial default diffuse is 0.4 gray: previews must
    # show that darkening exactly as a spec viewer would.
    legacy = _textured_quad_mesh()
    legacy.visual.material = SimpleMaterial(image=Image.new("RGB", (8, 8), _TEXTURE_COLOR))
    assert np.allclose(_material_base_color_factor(legacy), [0.4, 0.4, 0.4], atol=1e-6)

    # Materials without factors keep the glTF default (identity).
    bare = _textured_quad_mesh()
    bare.visual.material = PBRMaterial(baseColorTexture=Image.new("RGB", (8, 8), _TEXTURE_COLOR))
    assert np.allclose(_material_base_color_factor(bare), [1.0, 1.0, 1.0])


def test_preview_sampling_applies_base_color_factor() -> None:
    fixed = _textured_quad_mesh()
    sampled = _sample_texture_vertex_colors(fixed)
    assert sampled is not None
    assert np.allclose(sampled * 255.0, np.tile(_TEXTURE_COLOR, (4, 1)), atol=1.0)

    defective = _textured_quad_mesh()
    defective.visual = TextureVisuals(
        uv=np.asarray(defective.visual.uv, dtype=np.float32),
        material=SimpleMaterial(image=Image.new("RGB", (8, 8), _TEXTURE_COLOR)),
    )
    darkened = _sample_texture_vertex_colors(defective)
    assert darkened is not None
    assert np.allclose(darkened, sampled * 0.4, atol=2.0 / 255.0)


def test_rendered_preview_darkens_with_defective_base_color_factor() -> None:
    fixed = _textured_quad_mesh(texture_color=(255, 255, 255))
    defective = _textured_quad_mesh(texture_color=(255, 255, 255))
    defective.visual.material.baseColorFactor = (102, 102, 102, 255)

    try:
        bright = render_mesh_views(fixed, size=96, azimuths=(35.0,), elevation=20.0)[0]
        dark = render_mesh_views(defective, size=96, azimuths=(35.0,), elevation=20.0)[0]
    except Exception as exc:  # pragma: no cover - environment specific
        pytest.skip(f"Preview renderer unavailable in this environment: {exc}")

    bright_np = np.asarray(bright, dtype=np.float32)
    dark_np = np.asarray(dark, dtype=np.float32)
    changed = np.abs(bright_np - dark_np).max(axis=2) > 8.0
    assert changed.any(), "defective baseColorFactor must alter the preview"
    ratio = float(dark_np[changed].mean() / max(bright_np[changed].mean(), 1e-6))
    assert 0.3 <= ratio <= 0.5
