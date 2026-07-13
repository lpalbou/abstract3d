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


def _grid_box(extents, translation, cells: int = 12) -> trimesh.Trimesh:
    """Axis-aligned box with subdivided faces (production meshes never have
    footprint-sized single triangles; the cutter's area statistics must be
    exercised on realistically tessellated skins)."""
    box = trimesh.creation.box(extents=extents)
    for _ in range(10):
        if len(box.faces) >= cells * cells:
            break
        box = box.subdivide()
    box.apply_translation(translation)
    return box


def _slab_on_box_fixture() -> trimesh.Trimesh:
    """Y-up (Hunyuan native frame): a car-proportioned body on four wheel
    pads standing on a thin, overhanging ground plate — the measured slab
    signature (plate covering the footprint, exposed thin lamina 2% of mesh
    height above the bottom, hull overhanging the subject ~1.5x; the real
    incident measured plate 0.97 / lamina 0.61 / overhang 1.30 with the cut
    removing 38% of the surface)."""
    body = _grid_box((1.0, 0.5, 0.6), (0.0, 0.35, 0.0))
    wheels = [
        _grid_box((0.16, 0.1, 0.12), (x, 0.07, z), cells=8)
        for x in (-0.35, 0.35)
        for z in (-0.22, 0.22)
    ]
    slab = _grid_box((1.25, 0.012, 0.75), (0.0, 0.014, 0.0))
    return trimesh.util.concatenate([body, *wheels, slab])


def test_ground_slab_cutter_removes_overhanging_thin_plate() -> None:
    mesh = _slab_on_box_fixture()
    cut, report = runtime._hunyuan_cut_ground_slab(mesh, up_axis=(0.0, 1.0, 0.0))

    assert report is not None
    assert report["action"] == "removed"
    assert report["plate_footprint_frac"] >= runtime._SLAB_PLATE_FOOTPRINT_MIN
    assert report["lamina_ratio"] >= runtime._SLAB_LAMINA_MIN
    assert report["overhang_ratio"] >= runtime._SLAB_OVERHANG_MIN
    # The plate (extending to +-0.625 laterally) must be gone; the subject
    # (to +-0.5) must survive with its lateral extent intact.
    vertices = np.asarray(cut.vertices)
    assert vertices[:, 0].max() < 0.55
    assert vertices[:, 0].min() > -0.55
    assert abs(vertices[:, 0].max() - 0.5) < 0.05
    # Only the sliver below the cut plane is lost from the subject.
    assert vertices[:, 1].max() > 0.5


def test_ground_slab_cutter_spares_legitimate_thick_base() -> None:
    # Owl-proof proxy: a THICK carved base under the subject (base top at
    # ~10% of mesh height — no exposed thin lamina) that does not overhang
    # the subject's footprint. Measured control: owl base plate covers 49%
    # of the footprint yet lamina and overhang both refuse.
    subject = _grid_box((0.6, 1.6, 0.6), (0.0, 1.0, 0.0))
    base = _grid_box((0.9, 0.2, 0.9), (0.0, 0.1, 0.0))
    mesh = trimesh.util.concatenate([subject, base])

    cut, report = runtime._hunyuan_cut_ground_slab(mesh, up_axis=(0.0, 1.0, 0.0))

    assert report is None
    assert len(cut.faces) == len(mesh.faces)


def test_ground_slab_cutter_spares_thin_leg_contacts() -> None:
    # Chair proxy: a seat on four thin legs — bottom contact area is tiny
    # relative to the footprint hull (measured chair: 0.2%), so the plate
    # condition never arms.
    seat = _grid_box((1.0, 0.1, 1.0), (0.0, 1.0, 0.0))
    legs = [
        _grid_box((0.08, 1.0, 0.08), (x, 0.5, z), cells=4)
        for x in (-0.45, 0.45)
        for z in (-0.45, 0.45)
    ]
    mesh = trimesh.util.concatenate([seat, *legs])

    cut, report = runtime._hunyuan_cut_ground_slab(mesh, up_axis=(0.0, 1.0, 0.0))

    assert report is None
    assert len(cut.faces) == len(mesh.faces)


def test_ground_slab_cutter_refuses_when_cut_exceeds_budget() -> None:
    # Fail-closed: a huge plate with a tiny pole on it fires the detector,
    # but cutting would remove ~97% of the surface — the "subject" must
    # remain the majority of its own mesh, so the cutter refuses and
    # reports instead of amputating.
    pole = _grid_box((0.1, 0.5, 0.1), (0.0, 0.26, 0.0), cells=4)
    plate = _grid_box((2.0, 0.012, 2.0), (0.0, 0.006, 0.0), cells=24)
    mesh = trimesh.util.concatenate([pole, plate])

    cut, report = runtime._hunyuan_cut_ground_slab(mesh, up_axis=(0.0, 1.0, 0.0))

    assert report is not None
    assert report["action"] == "refused"
    assert report["cut_area_frac"] > runtime._SLAB_MAX_CUT_AREA_FRAC
    assert len(cut.faces) == len(mesh.faces)


def test_postprocess_records_ground_slab_removal_and_keeps_report() -> None:
    mesh = _slab_on_box_fixture()

    cleaned, applied, _warnings = runtime._hunyuan_postprocess_mesh(mesh, max_facenum=0)

    assert any(step.startswith("ground_slab_removed:") for step in applied)
    report = cleaned.metadata.get("abstract3d_ground_slab")
    assert report is not None and report["action"] == "removed"
    assert np.asarray(cleaned.vertices)[:, 0].max() < 0.55


def test_postprocess_flags_refused_ground_slab_as_warning() -> None:
    # cells=10 keeps the pole above the 0.5%-of-total floater floor (the
    # earlier cleanup step must not silently delete the subject before the
    # slab check reads the mesh).
    pole = _grid_box((0.1, 0.5, 0.1), (0.0, 0.26, 0.0), cells=10)
    plate = _grid_box((2.0, 0.012, 2.0), (0.0, 0.006, 0.0), cells=24)
    mesh = trimesh.util.concatenate([pole, plate])

    cleaned, applied, warnings = runtime._hunyuan_postprocess_mesh(mesh, max_facenum=0)

    assert not any(step.startswith("ground_slab_removed") for step in applied)
    assert any("ground slab detected but not cut" in warning for warning in warnings)
    report = cleaned.metadata.get("abstract3d_ground_slab")
    assert report is not None and report["action"] == "refused"


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
