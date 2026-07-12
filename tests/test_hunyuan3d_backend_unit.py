from __future__ import annotations

import numpy as np
import pytest

# Import torch before pymeshlab-backed helpers run: both bundle an OpenMP
# runtime on macOS and initializing pymeshlab's first crashes torch later.
import torch  # noqa: F401
import trimesh

from abstract3d.errors import CapabilityNotSupportedError
from abstract3d.backends import hunyuan3d_runtime as runtime


def test_backend_registry_exposes_hunyuan_aliases() -> None:
    from abstract3d.backends import BACKEND_FACTORIES

    for alias in ("abstract3d:hunyuan3d21-local", "hunyuan3d21", "hunyuan3d", "hunyuan"):
        assert alias in BACKEND_FACTORIES


def test_license_gate_blocks_runtime_without_acknowledgment(monkeypatch) -> None:
    monkeypatch.delenv("ABSTRACT3D_HUNYUAN_ACCEPT_LICENSE", raising=False)
    backend = runtime.Hunyuan3DShapeBackend(owner=None)
    with pytest.raises(CapabilityNotSupportedError, match="license"):
        backend._load_runtime(model_id=None, device="cpu", dtype=None)


def test_license_gate_accepts_env_acknowledgment(monkeypatch) -> None:
    monkeypatch.setenv("ABSTRACT3D_HUNYUAN_ACCEPT_LICENSE", "1")
    assert runtime._license_accepted(None) is True
    monkeypatch.setenv("ABSTRACT3D_HUNYUAN_ACCEPT_LICENSE", "0")
    assert runtime._license_accepted(None) is False


def test_model_selection_accepts_official_families_only() -> None:
    assert runtime._resolve_model_selection(None) == ("tencent/Hunyuan3D-2.1", "hunyuan3d-dit-v2-1")
    assert runtime._resolve_model_selection("tencent/Hunyuan3D-2.1") == (
        "tencent/Hunyuan3D-2.1",
        "hunyuan3d-dit-v2-1",
    )
    assert runtime._resolve_model_selection("tencent/Hunyuan3D-2.1/hunyuan3d-dit-v2-1") == (
        "tencent/Hunyuan3D-2.1",
        "hunyuan3d-dit-v2-1",
    )
    assert runtime._resolve_model_selection("tencent/Hunyuan3D-2mv") == (
        "tencent/Hunyuan3D-2mv",
        "hunyuan3d-dit-v2-mv",
    )
    assert runtime._resolve_model_selection("tencent/Hunyuan3D-2mv", "hunyuan3d-dit-v2-mv-fast") == (
        "tencent/Hunyuan3D-2mv",
        "hunyuan3d-dit-v2-mv-fast",
    )
    assert runtime._resolve_model_selection("tencent/Hunyuan3D-2mv/hunyuan3d-dit-v2-mv-turbo") == (
        "tencent/Hunyuan3D-2mv",
        "hunyuan3d-dit-v2-mv-turbo",
    )
    with pytest.raises(CapabilityNotSupportedError):
        runtime._resolve_model_selection("someone/Hunyuan3D-2.1-8bit")
    with pytest.raises(CapabilityNotSupportedError):
        runtime._resolve_model_selection("tencent/Hunyuan3D-2mv", "hunyuan3d-dit-v2-mv-unknown")


def test_remap_mv_config_rewrites_namespace_only_when_needed(tmp_path) -> None:
    mv_config = tmp_path / "config.yaml"
    mv_config.write_text(
        "model:\n  target: hy3dgen.shapegen.models.Hunyuan3DDiT\n"
        "image_processor:\n  target: hy3dgen.shapegen.preprocessors.MVImageProcessorV2\n",
        encoding="utf-8",
    )
    remapped = runtime._remap_mv_config(mv_config, tmp_path / "cache")
    text = remapped.read_text(encoding="utf-8")
    assert "hy3dgen.shapegen." not in text
    assert "hy3dshape.models.Hunyuan3DDiT" in text
    assert "hy3dshape.preprocessors.MVImageProcessorV2" in text

    native_config = tmp_path / "native.yaml"
    native_config.write_text("model:\n  target: hy3dshape.models.X\n", encoding="utf-8")
    assert runtime._remap_mv_config(native_config, tmp_path / "cache") == native_config


def test_select_dtype_policy() -> None:
    assert runtime._select_dtype("cpu") == "float32"
    assert runtime._select_dtype("mps") == "float16"
    assert runtime._select_dtype("cuda") == "float16"
    assert runtime._select_dtype("mps", "float32") == "float32"
    assert runtime._select_dtype("cpu", "fp16") == "float16"


def test_postprocess_keeps_significant_components_and_outward_normals() -> None:
    big = trimesh.creation.icosphere(subdivisions=5, radius=1.0)
    # 80 faces vs 20480: 0.39% of total faces, below the upstream-matched
    # 0.5%-of-total floater rule (the old 2%-of-largest rule was measured
    # 3.2x more aggressive and would amputate genuine detached parts the
    # size of a side mirror).
    floater = trimesh.creation.icosphere(subdivisions=1, radius=0.02)
    floater.apply_translation([3.0, 0.0, 0.0])
    combined = trimesh.util.concatenate([big, floater])

    cleaned, applied, warnings = runtime._hunyuan_postprocess_mesh(combined, max_facenum=0)

    assert cleaned.body_count == 1
    assert any(step.startswith("keep_significant_components") for step in applied)
    vertices = np.asarray(cleaned.vertices)
    normals = np.asarray(cleaned.vertex_normals)
    outward = ((vertices - vertices.mean(axis=0)) * normals).sum(axis=1)
    assert float((outward > 0).mean()) > 0.95


def test_postprocess_keeps_detached_parts_above_upstream_floater_rule() -> None:
    # A genuine detached part (1.5% of total faces — a side-mirror-sized
    # component) must SURVIVE: the upstream rule keeps anything >= 0.5%
    # of total faces.
    big = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
    part = trimesh.creation.icosphere(subdivisions=1, radius=0.05)
    part.apply_translation([2.0, 0.0, 0.0])
    combined = trimesh.util.concatenate([big, part])

    cleaned, _applied, _warnings = runtime._hunyuan_postprocess_mesh(combined, max_facenum=0)

    assert cleaned.body_count == 2


def test_postprocess_decimates_to_face_budget() -> None:
    mesh = trimesh.creation.icosphere(subdivisions=5)
    assert len(mesh.faces) > 5000

    cleaned, applied, _warnings = runtime._hunyuan_postprocess_mesh(mesh, max_facenum=5000)

    assert len(cleaned.faces) <= 5000
    assert any(step.startswith("quadric_decimation") for step in applied)


def test_canonicalize_axes_maps_yup_frontz_to_zup_frontx() -> None:
    # A marker mesh: tall along +Y (up in glTF), nose toward +Z (front).
    mesh = trimesh.creation.box(extents=(0.2, 1.0, 0.5))
    rotated, applied = runtime._hunyuan_canonicalize_axes(mesh)

    assert applied == ["yup_front_z_to_zup_front_x"]
    extents = rotated.extents
    # up (was Y, 1.0) must now be Z; front (was Z, 0.5) must now be X.
    assert abs(extents[2] - 1.0) < 1e-6
    assert abs(extents[0] - 0.5) < 1e-6
    assert abs(extents[1] - 0.2) < 1e-6


def test_adaptive_volume_decoder_recovers_sphere_from_synthetic_field() -> None:
    import torch

    class _SphereDecoder:
        """Analytic SDF stand-in for the VAE geo decoder."""

        def __call__(self, *, queries, latents):
            del latents
            radius = 0.6
            distance = torch.linalg.norm(queries, dim=-1)
            return (radius - distance).unsqueeze(-1)

    decoder = runtime._AdaptiveVolumeDecoder(coarse_resolution=32)
    latents = torch.zeros((1, 4, 8), dtype=torch.float32)
    grid = decoder(latents, _SphereDecoder(), bounds=1.01, num_chunks=200000, octree_resolution=64)

    assert tuple(grid.shape) == (1, 65, 65, 65)
    from skimage import measure

    verts, faces, _, _ = measure.marching_cubes(grid[0].numpy(), 0.0)
    # Rescale from grid indices to world units and check the radius.
    verts_world = verts / 64.0 * 2.02 - 1.01
    radii = np.linalg.norm(verts_world, axis=1)
    assert abs(float(radii.mean()) - 0.6) < 0.03


def test_adaptive_volume_decoder_handles_non_power_of_two_final_resolution() -> None:
    import torch

    class _SphereDecoder:
        def __call__(self, *, queries, latents):
            del latents
            radius = 0.6
            distance = torch.linalg.norm(queries, dim=-1)
            return (radius - distance).unsqueeze(-1)

    # 96 = 24 * 2 * 2: the schedule must be exact doublings ([24, 48, 96]),
    # exercising the same shape class as the production default 384
    # (96 -> 192 -> 384). A misaligned final level (e.g. 256 -> 384) was the
    # bug that silently capped effective resolution.
    decoder = runtime._AdaptiveVolumeDecoder(coarse_resolution=24)
    latents = torch.zeros((1, 4, 8), dtype=torch.float32)
    grid = decoder(latents, _SphereDecoder(), bounds=1.01, num_chunks=500000, octree_resolution=96)

    assert tuple(grid.shape) == (1, 97, 97, 97)

    # Direct refinement proof: near-surface cells must carry the EXACT
    # analytic values (the decoder queried them at the fine level), not a
    # trilinear upsample of a coarser level. The sphere SDF has curvature,
    # so an upsample from 48^3 deviates by ~h^2/(2R) ~= 1.5e-3 near the
    # surface; genuinely refined cells deviate by 0.
    axes = np.linspace(-1.01, 1.01, 97, dtype=np.float64)
    gx, gy, gz = np.meshgrid(axes, axes, axes, indexing="ij")
    analytic = 0.6 - np.sqrt(gx**2 + gy**2 + gz**2)
    values = grid[0].numpy().astype(np.float64)
    near_surface = np.abs(analytic) < (2.02 / 96.0)
    assert near_surface.any()
    max_near_surface_error = float(np.abs(values - analytic)[near_surface].max())
    assert max_near_surface_error < 1e-5

    from skimage import measure

    verts, faces, _, _ = measure.marching_cubes(grid[0].numpy(), 0.0)
    verts_world = verts / 96.0 * 2.02 - 1.01
    radii = np.linalg.norm(verts_world, axis=1)
    assert abs(float(radii.mean()) - 0.6) < 0.015
    assert float(radii.std()) < 0.01


def test_adaptive_volume_decoder_falls_back_to_dense_for_tiny_objects() -> None:
    import torch

    class _TinySphereDecoder:
        def __call__(self, *, queries, latents):
            del latents
            # So small the coarse grid misses the interior entirely at 8^3.
            radius = 0.04
            distance = torch.linalg.norm(queries - 0.07, dim=-1)
            return (radius - distance).unsqueeze(-1)

    decoder = runtime._AdaptiveVolumeDecoder(coarse_resolution=8)
    latents = torch.zeros((1, 4, 8), dtype=torch.float32)
    grid = decoder(latents, _TinySphereDecoder(), bounds=1.01, num_chunks=500000, octree_resolution=64)

    assert tuple(grid.shape) == (1, 65, 65, 65)
    assert bool((grid > 0).any())


def test_available_providers_flags_license(monkeypatch) -> None:
    monkeypatch.delenv("ABSTRACT3D_HUNYUAN_ACCEPT_LICENSE", raising=False)
    backend = runtime.Hunyuan3DShapeBackend(owner=None)
    providers = backend.available_providers()
    assert providers[0]["provider_id"] == "hunyuan3d21"
    assert providers[0]["license"] == "tencent-hunyuan-community"
    assert "European Union" in providers[0]["license_note"]
    assert providers[0]["status"] == "license_acknowledgment_required"

    monkeypatch.setenv("ABSTRACT3D_HUNYUAN_ACCEPT_LICENSE", "1")
    accepted = backend.available_providers()
    assert accepted[0]["status"] in {"available", "install_required"}
