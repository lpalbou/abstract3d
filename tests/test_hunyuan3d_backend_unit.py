from __future__ import annotations

import json

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


class _WrinkledSphereDecoder:
    """Deterministic, irregular analytic field for bit-identity tests.

    The wrinkle keeps the refinement mask non-trivial (neither empty nor
    full) and the 0.98 radius pushes surface cells against the grid walls
    so the fine-index clamp (np.minimum against the last vertex) fires.
    """

    def __call__(self, *, queries, latents):
        import torch

        del latents
        radius = torch.linalg.norm(queries, dim=-1)
        wrinkle = 0.02 * (
            torch.sin(9.0 * queries[..., 0])
            * torch.sin(11.0 * queries[..., 1])
            * torch.sin(13.0 * queries[..., 2])
        )
        logit = 165.0 * (0.98 + wrinkle - radius)
        return torch.clamp(logit, -10.0, 10.0).unsqueeze(-1)


def _legacy_adaptive_decode(decoder, latents, geo_decoder, *, octree_resolution, num_chunks):
    """The pre-2026-07-22 refinement assembly, replicated verbatim.

    Builds the (scale+1)^3 * N fine-index row matrix, clamps it with one
    np.minimum pass, scatters it in one fancy assignment, re-derives the
    refinement set with np.argwhere, gathers through strided index columns,
    and copies the zoom output through a fresh astype. The production
    decoder replaced these with per-offset scatters, an np.nonzero tuple,
    and astype(copy=False); this replica is the equality baseline proving
    the replacement changed no byte of the assembled field.
    """
    import torch
    from scipy import ndimage

    bounds = 1.01
    bbox_min = np.asarray([-bounds] * 3, dtype=np.float64)
    bbox_max = np.asarray([bounds] * 3, dtype=np.float64)
    final_resolution = int(octree_resolution)
    resolutions = [final_resolution]
    while resolutions[0] > decoder.coarse_resolution and resolutions[0] % 2 == 0:
        resolutions.insert(0, resolutions[0] // 2)

    def grid_axes(resolution):
        xs = np.linspace(bbox_min[0], bbox_max[0], resolution + 1, dtype=np.float64)
        ys = np.linspace(bbox_min[1], bbox_max[1], resolution + 1, dtype=np.float64)
        zs = np.linspace(bbox_min[2], bbox_max[2], resolution + 1, dtype=np.float64)
        return xs, ys, zs

    # _query_points is shared deliberately: the patch only added progress
    # logging there (accumulation is unchanged list + one torch.cat), so the
    # legacy-vs-new delta under test is exactly the loop assembly.
    def query(points):
        return decoder._query_points(
            points,
            latents=latents,
            geo_decoder=geo_decoder,
            num_chunks=num_chunks,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

    coarse_resolution = resolutions[0]
    xs, ys, zs = grid_axes(coarse_resolution)
    grid_x, grid_y, grid_z = np.meshgrid(xs, ys, zs, indexing="ij")
    coarse_points = np.stack([grid_x, grid_y, grid_z], axis=-1).reshape(-1, 3).astype(np.float32)
    grid = query(coarse_points).numpy().reshape(
        coarse_resolution + 1, coarse_resolution + 1, coarse_resolution + 1
    )
    assert len(resolutions) > 1 and (grid > 0.0).any()

    current_resolution = coarse_resolution
    for next_resolution in resolutions[1:]:
        inside = grid > 0.0
        surface = np.zeros_like(inside)
        for axis in range(3):
            changed = np.diff(inside, axis=axis)
            pad_lo = [(0, 0)] * 3
            pad_hi = [(0, 0)] * 3
            pad_lo[axis] = (0, 1)
            pad_hi[axis] = (1, 0)
            surface |= np.pad(changed, pad_lo, mode="constant")
            surface |= np.pad(changed, pad_hi, mode="constant")
        band = decoder.band
        if surface.any():
            shell_scale = float(np.median(np.abs(grid)[surface]))
            band = max(decoder.band, 1.5 * shell_scale)
        surface |= np.abs(grid) < band
        surface = ndimage.binary_dilation(surface, iterations=2)

        scale = next_resolution // current_resolution
        fine_shape = (next_resolution + 1,) * 3
        fine_mask = np.zeros(fine_shape, dtype=bool)
        coarse_idx = np.argwhere(surface)
        base = coarse_idx * scale
        block = np.stack(
            np.meshgrid(np.arange(scale + 1), np.arange(scale + 1), np.arange(scale + 1), indexing="ij"),
            axis=-1,
        ).reshape(-1, 3)
        fine_idx = (base[:, None, :] + block[None, :, :]).reshape(-1, 3)
        np.minimum(fine_idx, next_resolution, out=fine_idx)
        fine_mask[fine_idx[:, 0], fine_idx[:, 1], fine_idx[:, 2]] = True

        xs, ys, zs = grid_axes(next_resolution)
        refine_idx = np.argwhere(fine_mask)
        refine_points = np.stack(
            [xs[refine_idx[:, 0]], ys[refine_idx[:, 1]], zs[refine_idx[:, 2]]], axis=-1
        ).astype(np.float32)
        refine_logits = query(refine_points).numpy()

        zoom_factor = tuple(fs / cs for fs, cs in zip(fine_shape, grid.shape))
        next_grid = ndimage.zoom(grid, zoom_factor, order=1, mode="nearest").astype(np.float32)
        next_grid[refine_idx[:, 0], refine_idx[:, 1], refine_idx[:, 2]] = refine_logits
        grid = next_grid
        current_resolution = next_resolution
    return grid


def test_adaptive_volume_decoder_bit_identical_to_legacy_assembly() -> None:
    """The 2026-07-22 assembly rewrite must not change one byte of the field.

    Same synthetic decoder, same schedule (8 -> 16 -> 32): the production
    decoder's grid must equal the frozen legacy assembly exactly
    (np.array_equal on float32 — bit identity, not tolerance), while the
    refinement must remain non-trivial (the field differs from a pure
    coarse upsample, so the equality is not vacuous).
    """
    import torch
    from scipy import ndimage

    decoder = runtime._AdaptiveVolumeDecoder(coarse_resolution=8)
    latents = torch.zeros((1, 4, 8), dtype=torch.float32)
    geo_decoder = _WrinkledSphereDecoder()

    produced = decoder(
        latents, geo_decoder, bounds=1.01, num_chunks=97, octree_resolution=32
    )[0].numpy()
    legacy = _legacy_adaptive_decode(
        decoder, latents, geo_decoder, octree_resolution=32, num_chunks=97
    )

    assert produced.dtype == legacy.dtype == np.float32
    assert produced.shape == legacy.shape == (33, 33, 33)
    assert np.array_equal(produced, legacy)

    # Non-vacuousness: refinement genuinely overwrote interpolated cells.
    coarse_axes = np.linspace(-1.01, 1.01, 9, dtype=np.float64)
    gx, gy, gz = np.meshgrid(coarse_axes, coarse_axes, coarse_axes, indexing="ij")
    coarse_points = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3).astype(np.float32)
    coarse_grid = (
        decoder._query_points(
            coarse_points,
            latents=latents,
            geo_decoder=geo_decoder,
            num_chunks=97,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        .numpy()
        .reshape(9, 9, 9)
    )
    upsample_only = ndimage.zoom(
        ndimage.zoom(coarse_grid, (17 / 9,) * 3, order=1, mode="nearest"),
        (33 / 17,) * 3,
        order=1,
        mode="nearest",
    )
    assert not np.array_equal(produced, upsample_only)


def test_adaptive_volume_decoder_logs_chunk_progress(caplog) -> None:
    """Per-chunk INFO progress at a ~5% cadence (operability contract).

    e22v2 ran the 512-octree decode with zero output for its whole
    duration; the silence was misread as a hang. Every query pass must
    announce its size and emit chunk lines — bounded, so a 4,000-chunk
    dense pass logs ~20 lines, not 4,000.
    """
    import logging as _logging

    import torch

    caplog.set_level(_logging.INFO, logger="abstract3d.backends.hunyuan3d_runtime")
    decoder = runtime._AdaptiveVolumeDecoder(coarse_resolution=8)
    latents = torch.zeros((1, 4, 8), dtype=torch.float32)
    decoder(
        latents, _WrinkledSphereDecoder(), bounds=1.01, num_chunks=97, octree_resolution=16
    )

    messages = [record.getMessage() for record in caplog.records]
    level_lines = [m for m in messages if m.startswith("adaptive volume decode: level 8 -> 16")]
    assert len(level_lines) == 1
    assert "refining" in level_lines[0]

    coarse_chunks = [m for m in messages if m.startswith("volume decode [coarse 8^3]: chunk ")]
    refine_chunks = [m for m in messages if m.startswith("volume decode [refine 8 -> 16]: chunk ")]
    # 9^3=729 points at 97/chunk = 8 chunks; every-5% floors to every chunk.
    assert len(coarse_chunks) == 8
    assert coarse_chunks[-1].endswith("chunk 8/8 (100%)")
    # The refine pass must both announce totals and reach 100%.
    assert any("querying" in m and "[refine 8 -> 16]" in m for m in messages)
    assert refine_chunks and refine_chunks[-1].endswith("(100%)")
    # Cadence bound: never more than ~21 progress lines per query pass.
    assert len(refine_chunks) <= 21


def test_adaptive_volume_decoder_logs_are_bounded_at_scale(caplog) -> None:
    """At a large chunk count the every-5% cadence caps the line count."""
    import logging as _logging

    import torch

    caplog.set_level(_logging.INFO, logger="abstract3d.backends.hunyuan3d_runtime")
    decoder = runtime._AdaptiveVolumeDecoder(coarse_resolution=8)
    points = np.zeros((4200, 3), dtype=np.float32)

    class _ZeroDecoder:
        def __call__(self, *, queries, latents):
            del latents
            return torch.zeros(queries.shape[:-1] + (1,), dtype=torch.float32)

    decoder._query_points(
        points,
        latents=torch.zeros((1, 4, 8), dtype=torch.float32),
        geo_decoder=_ZeroDecoder(),
        num_chunks=1,
        device=torch.device("cpu"),
        dtype=torch.float32,
        label="volume decode [bounded]",
    )
    chunk_lines = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("volume decode [bounded]: chunk ")
    ]
    # 4200 chunks at every-5% cadence: 20 cadence lines (+1 if the final
    # chunk is off-cadence; here 4200 % 210 == 0, so exactly 20).
    assert len(chunk_lines) == 20
    assert chunk_lines[-1].endswith("chunk 4200/4200 (100%)")


def test_ensure_decode_logging_respects_existing_config() -> None:
    """The visibility helper must never fight an operator's logging setup:
    it attaches exactly one handler to the ``abstract3d`` namespace, only
    when the root and package loggers are both handler-free, and is
    idempotent across decodes."""
    import logging as _logging

    package_logger = _logging.getLogger("abstract3d")
    root_logger = _logging.getLogger()
    saved_package = list(package_logger.handlers)
    saved_root = list(root_logger.handlers)
    saved_level = package_logger.level
    try:
        # Case 1: operator configured the root logger -> do nothing.
        package_logger.handlers = []
        root_logger.handlers = [_logging.NullHandler()]
        runtime._ensure_decode_logging()
        assert package_logger.handlers == []

        # Case 2: nothing configured -> exactly one handler, idempotent.
        root_logger.handlers = []
        runtime._ensure_decode_logging()
        runtime._ensure_decode_logging()
        assert len(package_logger.handlers) == 1
        assert package_logger.getEffectiveLevel() <= _logging.INFO
    finally:
        package_logger.handlers = saved_package
        root_logger.handlers = saved_root
        package_logger.setLevel(saved_level)


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


# -- best-of-N shape candidate ranking ----------------------------------------


def _car_like_mesh() -> trimesh.Trimesh:
    """Canonical-frame (Z-up, front +X) box body on four wheels.

    The silhouette carries the concave detail the ranking must defend:
    a ground gap between the wheels and wheel bumps below the body.
    """
    body = trimesh.creation.box(extents=(1.6, 0.7, 0.5))
    body.apply_translation((0.0, 0.0, 0.15))
    parts = [body]
    for x in (-0.55, 0.55):
        for y in (-0.3, 0.3):
            wheel = trimesh.creation.cylinder(radius=0.16, height=0.12, sections=24)
            wheel.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2.0, (1.0, 0.0, 0.0)))
            wheel.apply_translation((x, y, -0.25))
            parts.append(wheel)
    return trimesh.util.concatenate(parts)


def _melted_blob_for(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """The trap adversary: a perfectly smooth, watertight, single-body
    ellipsoid spanning the same bounding box — every INTERNAL metric
    (smoothness, topology) is better than the true mesh's; only photo
    agreement can reject it."""
    blob = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    extents = mesh.extents
    blob.apply_scale((extents[0] / 2.0, extents[1] / 2.0, extents[2] / 2.0))
    blob.apply_translation(mesh.bounds.mean(axis=0))
    return blob


def _matte_for(mesh: trimesh.Trimesh, *, azimuth: float = 10.0, elevation: float = 10.0):
    """Photo-matte stand-in: the mesh's own silhouette at one witnessed
    pose (the ranking never sees which pose was used)."""
    from abstract3d.rendering import render_mesh_views

    views = render_mesh_views(mesh, size=256, azimuths=(azimuth,), elevation=elevation)
    return runtime._clay_silhouette_mask(views[0])


def test_dihedral_rms_separates_smooth_from_edgy() -> None:
    smooth = trimesh.creation.icosphere(subdivisions=3)
    edgy = _car_like_mesh()
    assert runtime._dihedral_rms_deg(smooth) < runtime._dihedral_rms_deg(edgy)


def test_mask_convex_hull_recovers_notch() -> None:
    mask = np.zeros((64, 64), dtype=bool)
    mask[16:48, 16:48] = True
    mask[30:48, 28:36] = False  # notch open to the bottom edge of the square
    hull = runtime._mask_convex_hull(mask)
    negative = hull & ~mask
    assert negative.sum() >= 0.8 * (18 * 8)
    # A convex mask has an (essentially) empty negative region.
    convex = np.zeros((64, 64), dtype=bool)
    convex[16:48, 16:48] = True
    residue = runtime._mask_convex_hull(convex) & ~convex
    assert residue.sum() <= 8


def test_photo_matte_mask_rejects_unsegmented_frames() -> None:
    from PIL import Image

    opaque = Image.new("RGBA", (64, 64), (200, 180, 160, 255))
    assert runtime._photo_matte_mask(opaque) is None
    empty = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    assert runtime._photo_matte_mask(empty) is None
    subject = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    for x in range(20, 44):
        for y in range(20, 44):
            subject.putpixel((x, y), (200, 180, 160, 255))
    mask = runtime._photo_matte_mask(subject)
    assert mask is not None and bool(mask[32, 32]) and not bool(mask[2, 2])


def test_shape_ranking_photo_agreement_beats_smoothness_trap() -> None:
    """THE design-requirement trap: a melted blob is smoother AND
    topologically cleaner than the true mesh; only photo agreement
    (silhouette + concavity) may decide against it."""
    clean = _car_like_mesh()
    blob = _melted_blob_for(clean)
    matte = _matte_for(clean)

    clean_metrics = runtime.evaluate_shape_candidate(clean, matte_mask=matte)
    blob_metrics = runtime.evaluate_shape_candidate(blob, matte_mask=matte)

    # The trap is armed: the blob wins every internal metric.
    assert blob_metrics["smoothness"] > clean_metrics["smoothness"]
    assert blob_metrics["topology_score"] >= clean_metrics["topology_score"]
    # Photo agreement catches it anyway.
    assert clean_metrics["photo_iou"] > blob_metrics["photo_iou"]
    assert clean_metrics["photo_concavity_iou"] > blob_metrics["photo_concavity_iou"]
    assert runtime.score_shape_candidate(clean_metrics) > runtime.score_shape_candidate(blob_metrics)


def test_shape_ranking_never_rewards_smoothing() -> None:
    """Adversarial-validation trap (2026-07-13, /tmp/hfix2): a laplacian
    melt of the TRUE mesh — not a wrong-silhouette ellipsoid — strictly
    improves dihedral RMS while destroying real detail. Measured on the
    persisted car_b draw, a 0.10 smoothness weight ranked the melt ABOVE
    the true draw (+0.0101 at 40 iterations); the weight must stay 0 so
    smoothing is never a winning direction. The fixture is a lumpy sphere
    (marching-cubes-noise stand-in) whose melt keeps the silhouette but
    flattens concave detail; the score must prefer the true surface and
    a smoothness-weighted score measurably prefers the melt's smoothness
    term (guarding against the weight quietly coming back)."""
    rng = np.random.default_rng(7)
    bumpy = trimesh.creation.icosphere(subdivisions=5, radius=1.0)
    bumpy.vertices += bumpy.vertex_normals * rng.normal(0.0, 0.01, size=len(bumpy.vertices))[:, None]
    melted = bumpy.copy()
    trimesh.smoothing.filter_laplacian(melted, lamb=0.5, iterations=25)

    matte = _matte_for(bumpy)
    bumpy_metrics = runtime.evaluate_shape_candidate(bumpy, matte_mask=matte)
    melted_metrics = runtime.evaluate_shape_candidate(melted, matte_mask=matte)

    # The trap is armed: the melt is much smoother and topologically equal.
    assert melted_metrics["smoothness"] > bumpy_metrics["smoothness"] + 0.5
    assert melted_metrics["topology_score"] == bumpy_metrics["topology_score"]
    # The score must not reward it.
    assert runtime.score_shape_candidate(bumpy_metrics) > runtime.score_shape_candidate(melted_metrics)
    # And the smoothness term carries no weight at all: scores are
    # invariant to the smoothness value (rewarding it inverts the real
    # melt ladder measured on the persisted car_b draw).
    assert runtime._SHAPE_RANK_WEIGHT_SMOOTHNESS == 0.0
    perturbed = dict(bumpy_metrics)
    perturbed["smoothness"] = 0.0
    assert runtime.score_shape_candidate(perturbed) == runtime.score_shape_candidate(bumpy_metrics)


def test_shape_ranking_penalizes_non_watertight_variant() -> None:
    clean = _car_like_mesh()
    holey = clean.copy()
    centroids = holey.triangles_center
    holey.update_faces(~((centroids[:, 2] > 0.35) & (np.abs(centroids[:, 0]) < 0.3)))
    assert not holey.is_watertight

    matte = _matte_for(clean)
    clean_metrics = runtime.evaluate_shape_candidate(clean, matte_mask=matte)
    holey_metrics = runtime.evaluate_shape_candidate(holey, matte_mask=matte)

    assert clean_metrics["topology_score"] > holey_metrics["topology_score"]
    assert runtime.score_shape_candidate(clean_metrics) > runtime.score_shape_candidate(holey_metrics)


def test_score_shape_candidate_geometry_only_when_matte_missing() -> None:
    mesh = trimesh.creation.icosphere(subdivisions=2)
    metrics = runtime.evaluate_shape_candidate(mesh, matte_mask=None)
    assert metrics["photo_iou"] is None
    assert metrics["photo_concavity_iou"] is None
    score = runtime.score_shape_candidate(metrics)
    expected = (
        runtime._SHAPE_RANK_WEIGHT_TOPOLOGY * metrics["topology_score"]
        + runtime._SHAPE_RANK_WEIGHT_SMOOTHNESS * metrics["smoothness"]
    )
    assert abs(score - expected) < 1e-9
    # Photo terms strictly increase the score when present.
    with_photo = dict(metrics)
    with_photo["photo_iou"] = 0.9
    with_photo["photo_concavity_iou"] = 0.3
    assert runtime.score_shape_candidate(with_photo) > score


def test_list_operations_schema_exposes_shape_candidates() -> None:
    backend = runtime.Hunyuan3DShapeBackend(owner=None)
    operations = backend.list_operations()
    schema = operations[-1]["parameter_schema"]["properties"]
    assert "shape_candidates" in schema
    assert schema["shape_candidates"]["type"] == "integer"
    assert schema["shape_candidates"]["minimum"] == 1


# -- best-of-N generation loop (faked diffusion runtime) -----------------------


class _FakeShapePipeline:
    """Stands in for Hunyuan3DDiTFlowMatchingPipeline: records the seed of
    every generator it receives and the conditioning image (single image or
    tagged mv dict), and returns the prepared mesh for that call (None
    entries model a draw that produced no surface)."""

    def __init__(self, meshes, capture) -> None:
        from types import SimpleNamespace

        self.vae = SimpleNamespace(volume_decoder=None)
        self._meshes = list(meshes)
        self._capture = capture

    def __call__(self, *, image, num_inference_steps, guidance_scale, octree_resolution,
                 num_chunks, mc_algo, generator, output_type, enable_pbar):
        call_index = len(self._capture["seeds"])
        self._capture["seeds"].append(int(generator.initial_seed()))
        self._capture.setdefault("images", []).append(image)
        self._capture.setdefault("settings", []).append(
            {"num_inference_steps": num_inference_steps,
             "octree_resolution": octree_resolution}
        )
        mesh = self._meshes[min(call_index, len(self._meshes) - 1)]
        return [mesh.copy() if mesh is not None else None]


def _install_fake_runtime(monkeypatch, backend, tmp_path, meshes, capture):
    """Route _run_generation through a fake pipeline and a fake pinned
    source tree (only the volume_decoders module is imported from it).
    The fake load honors the requested model id, so the mv checkpoint
    selection of the multiview path is observable through the stats."""
    import sys as _sys

    monkeypatch.setenv("ABSTRACT3D_HUNYUAN_ACCEPT_LICENSE", "1")
    source_root = tmp_path / "vendor"
    package_dir = source_root / "hy3dshape" / "hy3dshape" / "models" / "autoencoders"
    package_dir.mkdir(parents=True)
    (source_root / "hy3dshape" / "hy3dshape" / "__init__.py").write_text("", encoding="utf-8")
    (source_root / "hy3dshape" / "hy3dshape" / "models" / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "volume_decoders.py").write_text(
        "class VanillaVolumeDecoder:\n    pass\n\n\nclass HierarchicalVolumeDecoding:\n    pass\n",
        encoding="utf-8",
    )
    for name in [n for n in list(_sys.modules) if n == "hy3dshape" or n.startswith("hy3dshape.")]:
        monkeypatch.delitem(_sys.modules, name, raising=False)

    pipeline = _FakeShapePipeline(meshes, capture)

    def _fake_load_runtime(*, model_id=None, device=None, dtype=None, model_subfolder=None):
        resolved_repo, resolved_subfolder = runtime._resolve_model_selection(
            model_id, model_subfolder
        )
        backend._pipeline = pipeline
        backend._resident_device = "cpu"
        backend._resident_dtype = "float32"
        backend._last_runtime_stats = {
            "load_s": 0.01,
            "model_id": resolved_repo,
            "subfolder": resolved_subfolder,
            "multiview_capable": resolved_repo == runtime._MV_MODEL_ID,
            "source_dir": str(source_root),
            "device": "cpu",
            "dtype": "float32",
        }
        return pipeline

    monkeypatch.setattr(backend, "_load_runtime", _fake_load_runtime)
    return pipeline


def _alpha_disc_image(size: int = 96):
    """Subject-on-transparent RGBA input (skips rembg in preprocessing)."""
    from PIL import Image

    array = np.zeros((size, size, 4), dtype=np.uint8)
    yy, xx = np.mgrid[0:size, 0:size]
    disc = (yy - size / 2.0) ** 2 + (xx - size / 2.0) ** 2 <= (size * 0.35) ** 2
    array[disc] = (180, 160, 140, 255)
    return Image.fromarray(array, "RGBA")


def test_generation_seed_spacing_and_candidate_metadata(monkeypatch, tmp_path) -> None:
    backend = runtime.Hunyuan3DShapeBackend(owner=None)
    capture: dict = {"seeds": []}
    # The disc matte matches the sphere, not the elongated boxes: the
    # middle candidate must win on photo agreement.
    box_a = trimesh.creation.box(extents=(1.8, 0.4, 0.4))
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=0.7)
    box_b = trimesh.creation.box(extents=(0.4, 1.8, 0.4))
    _install_fake_runtime(monkeypatch, backend, tmp_path, [box_a, sphere, box_b], capture)

    result = backend.i23d(
        _alpha_disc_image(), device="cpu", texture_mode="none", seed=7, shape_candidates=3,
    )

    # Spec'd seed spacing: candidate i draws at base + 1000*i.
    assert capture["seeds"] == [7, 1007, 2007]
    metadata = result["metadata"]
    assert metadata["seed"] == 7  # base seed preserved for texture stages
    assert metadata["shape_seed"] == 1007  # the sphere's draw
    rows = metadata["shape_candidates"]
    assert [row["seed"] for row in rows] == [7, 1007, 2007]
    assert [row["selected"] for row in rows] == [False, True, False]
    for row in rows:
        assert row["status"] == "ranked"
        assert row["score"] is not None
        assert row["photo_iou"] is not None
        assert row["photo_concavity_iou"] is not None
        assert "watertight" in row and "body_count" in row
        assert "dihedral_rms_deg" in row and "topology_raw" in row
    assert rows[1]["photo_iou"] > rows[0]["photo_iou"]
    assert rows[1]["photo_iou"] > rows[2]["photo_iou"]
    assert metadata["timings_s"]["shape_selection"] >= 0.0
    # The exported mesh IS the selected candidate (sphere-sized vertex set,
    # not a box's 8 corners).
    assert metadata["vertex_count"] > 100


def test_generation_records_no_surface_candidates(monkeypatch, tmp_path) -> None:
    backend = runtime.Hunyuan3DShapeBackend(owner=None)
    capture: dict = {"seeds": []}
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=0.7)
    _install_fake_runtime(monkeypatch, backend, tmp_path, [None, sphere], capture)

    result = backend.i23d(
        _alpha_disc_image(), device="cpu", texture_mode="none", seed=11, shape_candidates=2,
    )

    rows = result["metadata"]["shape_candidates"]
    assert rows[0]["status"] == "no_surface"
    assert rows[0]["selected"] is False and rows[0]["score"] is None
    assert rows[1]["selected"] is True
    assert result["metadata"]["shape_seed"] == 1011
    assert any("produced no surface" in w for w in result["metadata"]["postprocess_warnings"])


def test_generation_raises_when_all_candidates_fail(monkeypatch, tmp_path) -> None:
    from abstract3d.errors import Abstract3DError

    backend = runtime.Hunyuan3DShapeBackend(owner=None)
    capture: dict = {"seeds": []}
    _install_fake_runtime(monkeypatch, backend, tmp_path, [None, None], capture)

    with pytest.raises(Abstract3DError, match="no surface in any of 2"):
        backend.i23d(
            _alpha_disc_image(), device="cpu", texture_mode="none", seed=3, shape_candidates=2,
        )


def test_single_candidate_path_is_untouched(monkeypatch, tmp_path) -> None:
    """The fleet default (N=1) must be EXACTLY the historical path: one
    draw at the base seed, no matte extraction, no ranking render, no new
    metadata keys."""
    backend = runtime.Hunyuan3DShapeBackend(owner=None)
    capture: dict = {"seeds": []}
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=0.7)
    _install_fake_runtime(monkeypatch, backend, tmp_path, [sphere], capture)

    def _forbidden(*args, **kwargs):
        raise AssertionError("ranking must not run on the N=1 path")

    monkeypatch.setattr(runtime, "evaluate_shape_candidate", _forbidden)
    monkeypatch.setattr(runtime, "score_shape_candidate", _forbidden)
    monkeypatch.setattr(runtime, "_photo_matte_mask", _forbidden)

    result = backend.i23d(_alpha_disc_image(), device="cpu", texture_mode="none", seed=42)

    assert capture["seeds"] == [42]
    metadata = result["metadata"]
    assert metadata["seed"] == 42
    assert "shape_candidates" not in metadata
    assert "shape_seed" not in metadata
    assert "shape_selection" not in metadata["timings_s"]


def test_explicit_single_candidate_matches_default_path(monkeypatch, tmp_path) -> None:
    backend = runtime.Hunyuan3DShapeBackend(owner=None)
    capture: dict = {"seeds": []}
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=0.7)
    _install_fake_runtime(monkeypatch, backend, tmp_path, [sphere], capture)

    result = backend.i23d(
        _alpha_disc_image(), device="cpu", texture_mode="none", seed=42, shape_candidates=1,
    )

    assert capture["seeds"] == [42]
    assert "shape_candidates" not in result["metadata"]


def test_shape_candidates_option_validation(monkeypatch, tmp_path) -> None:
    from abstract3d.errors import InvalidRequestError

    backend = runtime.Hunyuan3DShapeBackend(owner=None)
    capture: dict = {"seeds": []}
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=0.7)
    _install_fake_runtime(monkeypatch, backend, tmp_path, [sphere], capture)

    for bad in (0, -2, "abc"):
        with pytest.raises(InvalidRequestError, match="shape_candidates"):
            backend.i23d(
                _alpha_disc_image(), device="cpu", texture_mode="none", shape_candidates=bad,
            )
    # Validation failed loudly BEFORE any diffusion draw.
    assert capture["seeds"] == []


def test_texture_reference_angle_planning_option_validation(monkeypatch, tmp_path) -> None:
    from abstract3d.errors import InvalidRequestError

    backend = runtime.Hunyuan3DShapeBackend(owner=None)
    capture: dict = {"seeds": []}
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=0.7)
    _install_fake_runtime(monkeypatch, backend, tmp_path, [sphere], capture)

    with pytest.raises(InvalidRequestError, match="texture_reference_angle_planning"):
        backend.i23d(
            _alpha_disc_image(), device="cpu", texture_mode="none",
            texture_reference_angle_planning="clever",
        )
    # A typo fails in milliseconds, before any diffusion draw.
    assert capture["seeds"] == []

    # every valid mode is consumed silently (geometry-only run: the
    # planning lane itself is exercised by the bundle/rebake tests)
    for mode in ("auto", "adaptive", "static"):
        result = backend.i23d(
            _alpha_disc_image(), device="cpu", texture_mode="none",
            texture_reference_angle_planning=mode,
        )
        assert result["metadata"]["appearance_mode"] == "geometry_only"


def test_list_operations_schema_exposes_angle_planning() -> None:
    backend = runtime.Hunyuan3DShapeBackend(owner=None)
    operations = backend.list_operations(task="image_to_scene3d")
    schema = operations[0]["parameter_schema"]["properties"]
    assert schema["texture_reference_angle_planning"]["enum"] == [
        "auto", "adaptive", "static"]


def test_texture_stage_adaptive_planning_threads_pose_and_plan(
        monkeypatch, tmp_path) -> None:
    """Full texture-stage integration on the fake runtime: the baseline
    bake runs FIRST, its pose estimate reaches generation's source_pose,
    the adaptive plan chooses the generated angles, and the plan (with
    predicted gains) lands in texture_artifacts.reference_generation."""

    import io

    from PIL import Image

    # Host-only workaround (documented in the validation harnesses): after
    # real torch tensor work, skimage's colorconv matmul through Accelerate
    # segfaults on this macOS host. The einsum route is numerically
    # identical (asserted) and only this test runs torch work + lab2rgb in
    # one process.
    import skimage.color.colorconv as _colorconv

    def _convert_einsum(matrix, arr):
        arr = _colorconv._prepare_colorarray(arr)
        return np.einsum("...i,ji->...j", arr, matrix.astype(arr.dtype))

    probe = np.random.default_rng(1).random((5, 7, 3))
    matrix = np.asarray([[3.24, -1.54, -0.50], [-0.97, 1.88, 0.04],
                         [0.06, -0.20, 1.06]])
    assert float(np.abs(_convert_einsum(matrix, probe)
                        - (probe @ matrix.T)).max()) < 1e-12
    monkeypatch.setattr(_colorconv, "_convert", _convert_einsum)

    backend = runtime.Hunyuan3DShapeBackend(owner=None)
    capture: dict = {"seeds": []}
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=0.7)
    _install_fake_runtime(monkeypatch, backend, tmp_path, [sphere], capture)

    monkeypatch.setattr(runtime, "has_image_composer", lambda owner: True)
    monkeypatch.setattr(
        "abstract3d.image_composition.resolve_image_generation_request",
        lambda owner: {})
    monkeypatch.setattr(
        "abstract3d.captioning.caption_image", lambda image, **kw: "test object")

    def fake_matte(img):
        arr = np.asarray(img.convert("RGBA")).copy()
        dark = arr[:, :, :3].astype(int).sum(axis=2) < 90
        arr[:, :, 3] = np.where(dark, 0, 255).astype(np.uint8)
        return Image.fromarray(arr, "RGBA")

    monkeypatch.setattr(
        "abstract3d.segmentation.remove_background_robust", fake_matte)

    def tinting_echo(prompt, image, **kwargs):
        # Right panel of the composite canvas = the clay at the requested
        # angle; tint its foreground with the source color so both the
        # silhouette gate and the material gates pass for ANY planned
        # angle without knowing the plan in advance.
        canvas = Image.open(io.BytesIO(image)).convert("RGB")
        right = canvas.crop((canvas.width // 2, 0, canvas.width, canvas.height))
        arr = np.asarray(right).copy()
        arr[arr.sum(axis=2) > 90] = (180, 160, 140)
        buffer = io.BytesIO()
        Image.fromarray(arr).save(buffer, format="PNG")
        return buffer.getvalue()

    monkeypatch.setattr(
        "abstract3d.reference_generation.default_i2i_generator",
        lambda owner: tinting_echo)

    seen_pose: dict = {}
    from abstract3d import reference_generation as refgen

    real_generate = refgen.generate_reference_views

    def spy_generate(mesh_arg, source, **kwargs):
        seen_pose["source_pose"] = kwargs.get("source_pose")
        return real_generate(mesh_arg, source, **kwargs)

    monkeypatch.setattr(
        "abstract3d.reference_generation.generate_reference_views",
        spy_generate)

    result = backend.i23d(
        _alpha_disc_image(),
        device="cpu",
        texture_mode="baked_basecolor",
        texture_resolution=64,
        texture_reference_generation="on",
        texture_reference_angle_planning="adaptive",
    )
    metadata = result["metadata"]
    report = metadata["texture_artifacts"]["reference_generation"]
    plan = report["angle_plan"]
    assert plan["mode"] == "adaptive"
    assert plan["angles_source"] == "adaptive"
    assert plan["selected"], "the sphere plan must select angles"
    assert all(row["predicted_gain"] >= plan["min_gain"]
               for row in plan["selected"])
    # generation ran exactly the planned angles, conditioned on the
    # baseline bake's pose statement
    generated_labels = [row["label"] for row in report["angles"]]
    assert generated_labels == [row["label"] for row in plan["selected"]]
    assert seen_pose["source_pose"] == tuple(plan["source_pose"])
    source_pose = metadata["texture_artifacts"]["source_pose"]
    assert seen_pose["source_pose"] == (
        float(source_pose["azimuth_deg"]), float(source_pose["elevation_deg"]))
    # the A/B machinery ran against the pre-baked baseline
    assert "bake_acceptance" in report
    # accepted views reached the bake
    view_labels = [row.get("label") for row in
                   metadata["texture_artifacts"]["observed_view_stats"]]
    for label in generated_labels:
        if any(entry["label"] == label and entry["accepted"]
               for entry in report["angles"]):
            assert label in view_labels


def test_shape_candidates_config_key_default(monkeypatch, tmp_path) -> None:
    from types import SimpleNamespace

    owner = SimpleNamespace(config={"scene3d_hunyuan_shape_candidates": "2"})
    backend = runtime.Hunyuan3DShapeBackend(owner=owner)
    capture: dict = {"seeds": []}
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=0.7)
    box = trimesh.creation.box(extents=(1.8, 0.4, 0.4))
    _install_fake_runtime(monkeypatch, backend, tmp_path, [box, sphere], capture)

    result = backend.i23d(_alpha_disc_image(), device="cpu", texture_mode="none", seed=5)

    assert capture["seeds"] == [5, 1005]
    assert len(result["metadata"]["shape_candidates"]) == 2


# -- multi-view geometry conditioning ------------------------------------------


def test_mv_snap_tag_snaps_within_tolerance_only() -> None:
    assert runtime._mv_snap_tag(0.0) == ("front", 0.0)
    assert runtime._mv_snap_tag(92.0) == ("left", 2.0)
    assert runtime._mv_snap_tag(180.0) == ("back", 0.0)
    assert runtime._mv_snap_tag(-171.0) == ("back", 9.0)
    assert runtime._mv_snap_tag(-88.0) == ("right", 2.0)
    # Exactly on the tolerance boundary still snaps (<=), beyond does not.
    assert runtime._mv_snap_tag(-115.0) == ("right", 25.0)
    assert runtime._mv_snap_tag(45.0) is None
    assert runtime._mv_snap_tag(135.0) is None


def _composer_owner():
    """Owner with a deterministic local vision handle: has_image_composer
    and auto_generation_ready are both True regardless of what optional
    packages the test venv carries."""
    from types import SimpleNamespace

    return SimpleNamespace(
        config={},
        vision=SimpleNamespace(
            t2i=lambda *args, **kwargs: b"",
            i2i=lambda *args, **kwargs: b"",
        ),
    )


def _two_tone_disc_image(size: int = 96):
    """Chromatic subject on transparent background: a red disc with a WHITE
    quadrant. The dispersion guard of the material gate measures the std of
    the foreground chroma MAGNITUDES, so arming it needs parts of unequal
    chroma (red ~58 + white ~0), not merely different hues (red vs green
    both measure ~58-60 and leave the dispersion below the arming floor)."""
    from PIL import Image

    array = np.zeros((size, size, 4), dtype=np.uint8)
    yy, xx = np.mgrid[0:size, 0:size]
    disc = (yy - size / 2.0) ** 2 + (xx - size / 2.0) ** 2 <= (size * 0.35) ** 2
    array[disc] = (200, 30, 30, 255)
    quadrant = disc & (yy < size / 2.0) & (xx < size / 2.0)
    array[quadrant] = (240, 240, 240, 255)
    return Image.fromarray(array, "RGBA")


def _image_bytes(image) -> bytes:
    import io

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _stub_view_consistency(monkeypatch, *, verdicts=("consistent",), align=None):
    """Script the sibling view-consistency stage through the runtime's
    lazy resolver (`_view_consistency_api`) — the module ships in
    parallel and may not exist in this checkout, which is exactly why
    the runtime exposes the resolver seam. `verdicts` are consumed one
    per report call in order; the last one repeats. Returns a call log
    of (args...) tuples so tests can pin the frozen argument order."""
    queue = list(verdicts)
    log = {"report": [], "align": []}

    def _report(reference, candidate):
        verdict = queue.pop(0) if len(queue) > 1 else queue[0]
        report = {
            "score": {"consistent": 0.93, "correctable": 0.61}.get(verdict, 0.12),
            "best_shift_rows": 0 if verdict == "consistent" else 9,
            "shift_frac": 0.0 if verdict == "consistent" else 0.0937,
            "verdict": verdict,
        }
        log["report"].append((reference, candidate, report))
        return report

    def _align(image, reference, **kwargs):
        corrected, report = (align or (lambda i, r: (i, {"applied_shift_rows": 9})))(
            image, reference
        )
        log["align"].append((image, reference, corrected))
        return corrected, report

    monkeypatch.setattr(runtime, "_view_consistency_api", lambda: (_report, _align))
    return log


def test_synthesize_geometry_views_gates_and_side_pair(monkeypatch) -> None:
    """End-to-end gate battery on real images: a mirror-consistent back
    view passes, and a side pair whose silhouettes disagree is dropped
    WHOLE (blame between the two sides is unattributable)."""
    from PIL import Image

    from abstract3d import segmentation

    monkeypatch.setattr(
        segmentation, "remove_background_robust", lambda image: image.convert("RGBA")
    )
    _stub_view_consistency(monkeypatch)
    source = _two_tone_disc_image()

    # Same two-tone material as the source (so the material gate passes and
    # the PAIR gate is what fires), but a silhouette that cannot be the
    # mirror of the left view's disc.
    bar = np.zeros((96, 96, 4), dtype=np.uint8)
    bar[18:78, 38:58] = (200, 30, 30, 255)
    bar[18:38, 38:58] = (240, 240, 240, 255)
    bar_image = Image.fromarray(bar, "RGBA")

    def _generator(prompt, image, **kwargs):
        if "behind" in prompt:
            return _image_bytes(source)  # mirror-consistent back
        if "left side" in prompt:
            return _image_bytes(source)  # disc
        return _image_bytes(bar_image)  # right side: silhouette lie

    accepted, records, rejected = runtime._synthesize_geometry_views(
        None,
        source,
        subject_noun="disc toy",
        base_seed=7,
        labels=runtime._GEOMETRY_VIEW_ANGLES,
        attempts=1,
        image_generator=_generator,
    )

    assert [view["label"] for view in accepted] == ["back"]
    by_label = {record["label"]: record for record in records}
    assert by_label["back"]["accepted"] is True
    assert by_label["back"]["attempts"][0]["back_mirror_iou"] > 0.9
    assert by_label["side_left"]["accepted"] is False
    assert by_label["side_right"]["accepted"] is False
    assert "side-pair mirror disagreement" in by_label["side_left"]["failure"]
    # The dropped pair is persisted for diagnosis.
    assert {row["label"] for row in rejected} == {"side_left", "side_right"}
    # Accepted views carry replay provenance for the texture lane.
    assert accepted[0]["raw_bytes"] and accepted[0]["raw_payload_md5"]


def test_synthesize_geometry_views_rejects_chroma_collapse(monkeypatch) -> None:
    """A near-monochrome generation of a chromatic subject means the
    generator lost the subject; its geometry cannot be trusted either."""
    from PIL import Image

    from abstract3d import segmentation

    monkeypatch.setattr(
        segmentation, "remove_background_robust", lambda image: image.convert("RGBA")
    )
    _stub_view_consistency(monkeypatch)
    source = _two_tone_disc_image()
    gray = np.zeros((96, 96, 4), dtype=np.uint8)
    yy, xx = np.mgrid[0:96, 0:96]
    disc = (yy - 48.0) ** 2 + (xx - 48.0) ** 2 <= (96 * 0.35) ** 2
    gray[disc] = (128, 128, 128, 255)

    accepted, records, rejected = runtime._synthesize_geometry_views(
        None,
        source,
        subject_noun="disc toy",
        base_seed=7,
        labels=(("back", 180.0),),
        attempts=1,
        image_generator=lambda prompt, image, **kwargs: _image_bytes(
            Image.fromarray(gray, "RGBA")
        ),
    )

    assert accepted == []
    assert records[0]["accepted"] is False
    assert "subject identity" in records[0]["attempts"][0]["failure"]
    assert "chroma collapse" in records[0]["attempts"][0]["failure"]
    assert len(rejected) == 1


# -- row-consistency acceptance stage (the double-mouth defect) ----------------


def test_row_consistency_consistent_view_passes_untouched(monkeypatch) -> None:
    """Verdict 'consistent' accepts exactly as before the stage existed:
    no alignment runs, and the report lands in the provenance record."""
    from abstract3d import segmentation

    monkeypatch.setattr(
        segmentation, "remove_background_robust", lambda image: image.convert("RGBA")
    )
    source = _two_tone_disc_image()
    log = _stub_view_consistency(monkeypatch, verdicts=("consistent",))

    accepted, records, rejected = runtime._synthesize_geometry_views(
        None,
        source,
        subject_noun="disc toy",
        base_seed=7,
        labels=(("back", 180.0),),
        attempts=1,
        image_generator=lambda prompt, image, **kwargs: _image_bytes(source),
    )

    assert [view["label"] for view in accepted] == ["back"]
    # Frozen sibling API argument order: report(reference, candidate),
    # with the SOURCE PHOTO in the reference slot (no clay exists yet).
    assert log["report"][0][0] is source
    assert log["align"] == []  # untouched: no correction ran
    attempt = records[0]["attempts"][0]
    assert attempt["row_consistency"]["verdict"] == "consistent"
    assert "row_corrected" not in attempt
    assert accepted[0]["row_consistency"]["verdict"] == "consistent"
    assert accepted[0]["row_corrected"] is False
    assert records[0]["row_consistency"]["verdict"] == "consistent"
    assert rejected == []


def test_row_consistency_correctable_view_aligned_and_regated(monkeypatch) -> None:
    """Verdict 'correctable' routes the candidate through align_view_rows
    and the corrected pixels re-earn acceptance through the existing
    gates (matte, subject identity, back-mirror) — the conditioner must
    see the CORRECTED image, with the correction on the record."""
    from PIL import Image

    from abstract3d import segmentation

    monkeypatch.setattr(
        segmentation, "remove_background_robust", lambda image: image.convert("RGBA")
    )
    source = _two_tone_disc_image()

    def _shift_rows(image, reference):
        # A genuine row correction: same subject 9 rows lower (still fully
        # in frame), so matte/material/mirror re-gates legitimately pass.
        array = np.roll(np.asarray(image.convert("RGBA")), 9, axis=0)
        return Image.fromarray(array, "RGBA"), {"applied_shift_rows": 9}

    log = _stub_view_consistency(
        monkeypatch, verdicts=("correctable",), align=_shift_rows
    )

    accepted, records, rejected = runtime._synthesize_geometry_views(
        None,
        source,
        subject_noun="disc toy",
        base_seed=7,
        labels=(("back", 180.0),),
        attempts=1,
        image_generator=lambda prompt, image, **kwargs: _image_bytes(source),
    )

    assert [view["label"] for view in accepted] == ["back"]
    # align_view_rows(candidate, reference): the aligner receives the exact
    # image the report judged, with the source photo as the reference.
    assert log["align"][0][0] is log["report"][0][1]
    assert log["align"][0][1] is source
    # The conditioner sees the CORRECTED pixels, not the original.
    assert accepted[0]["rgba"] is log["align"][0][2]
    attempt = records[0]["attempts"][0]
    assert attempt["row_consistency"]["verdict"] == "correctable"
    assert attempt["row_corrected"] is True
    assert attempt["row_alignment"] == {"applied_shift_rows": 9}
    # Re-gate evidence recorded from the corrected image.
    assert "row_corrected_material" in attempt
    assert attempt["row_corrected_back_mirror_iou"] > 0.9
    assert accepted[0]["row_corrected"] is True
    assert records[0]["row_corrected"] is True
    assert rejected == []


def test_row_consistency_correction_must_repass_existing_gates(monkeypatch) -> None:
    """A correction is a mutation: corrected pixels that fail the existing
    gate battery are rejected, not accepted on the original's passes."""
    from PIL import Image

    from abstract3d import segmentation

    monkeypatch.setattr(
        segmentation, "remove_background_robust", lambda image: image.convert("RGBA")
    )
    source = _two_tone_disc_image()
    gray = np.zeros((96, 96, 4), dtype=np.uint8)
    yy, xx = np.mgrid[0:96, 0:96]
    disc = (yy - 48.0) ** 2 + (xx - 48.0) ** 2 <= (96 * 0.35) ** 2
    gray[disc] = (128, 128, 128, 255)
    gray_image = Image.fromarray(gray, "RGBA")

    log = _stub_view_consistency(
        monkeypatch,
        verdicts=("correctable",),
        align=lambda image, reference: (gray_image, {"applied_shift_rows": 9}),
    )

    accepted, records, rejected = runtime._synthesize_geometry_views(
        None,
        source,
        subject_noun="disc toy",
        base_seed=7,
        labels=(("back", 180.0),),
        attempts=1,
        image_generator=lambda prompt, image, **kwargs: _image_bytes(source),
    )

    assert accepted == []
    assert records[0]["accepted"] is False
    attempt = records[0]["attempts"][0]
    assert attempt["row_corrected"] is True
    assert "row consistency" in attempt["failure"]
    assert "subject-identity re-gate" in attempt["failure"]
    assert "chroma collapse" in attempt["failure"]
    assert len(log["align"]) == 1
    # The rejected (corrected) candidate is persisted for diagnosis.
    assert len(rejected) == 1 and rejected[0]["label"] == "back"


def test_row_consistency_inconsistent_view_rejected_with_reason(monkeypatch) -> None:
    """Verdict 'inconsistent' rejects with the reason recorded; the
    candidate travels back for rejected_geometry_views/ persistence."""
    from abstract3d import segmentation

    monkeypatch.setattr(
        segmentation, "remove_background_robust", lambda image: image.convert("RGBA")
    )
    source = _two_tone_disc_image()
    log = _stub_view_consistency(monkeypatch, verdicts=("inconsistent",))

    accepted, records, rejected = runtime._synthesize_geometry_views(
        None,
        source,
        subject_noun="disc toy",
        base_seed=7,
        labels=(("back", 180.0),),
        attempts=1,
        image_generator=lambda prompt, image, **kwargs: _image_bytes(source),
    )

    assert accepted == []
    assert records[0]["accepted"] is False
    attempt = records[0]["attempts"][0]
    assert attempt["row_consistency"]["verdict"] == "inconsistent"
    assert "row consistency" in attempt["failure"]
    assert "double-mouth" in attempt["failure"]
    assert log["align"] == []  # inconsistent is never "repaired"
    assert len(rejected) == 1 and rejected[0]["label"] == "back"


def test_row_consistency_unknown_verdict_fails_closed(monkeypatch) -> None:
    """A verdict outside the frozen contract must reject, never accept
    (an unverifiable candidate must not condition the checkpoint)."""
    from abstract3d import segmentation

    monkeypatch.setattr(
        segmentation, "remove_background_robust", lambda image: image.convert("RGBA")
    )
    source = _two_tone_disc_image()
    _stub_view_consistency(monkeypatch, verdicts=("almost-fine",))

    accepted, records, _rejected = runtime._synthesize_geometry_views(
        None,
        source,
        subject_noun="disc toy",
        base_seed=7,
        labels=(("back", 180.0),),
        attempts=1,
        image_generator=lambda prompt, image, **kwargs: _image_bytes(source),
    )

    assert accepted == []
    failure = records[0]["attempts"][0]["failure"]
    assert "unrecognized verdict" in failure and "fail closed" in failure


def test_generation_metadata_carries_row_consistency_fields(
    monkeypatch, tmp_path
) -> None:
    """The geometry_views metadata rows thread row_consistency (and
    row_corrected only when a correction actually ran) from the accepted
    synthesized views; the source-photo front row is never gated."""
    backend = runtime.Hunyuan3DShapeBackend(owner=_composer_owner())
    capture: dict = {"seeds": []}
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=0.7)
    _install_fake_runtime(monkeypatch, backend, tmp_path, [sphere], capture)

    from abstract3d import captioning

    monkeypatch.setattr(
        captioning, "caption_image", lambda image, **kwargs: "a toy disc"
    )

    corrected_report = {
        "score": 0.61,
        "best_shift_rows": 9,
        "shift_frac": 0.0937,
        "verdict": "correctable",
    }
    consistent_report = {
        "score": 0.93,
        "best_shift_rows": 0,
        "shift_frac": 0.0,
        "verdict": "consistent",
    }

    def _fake_synthesis(owner, source_rgba, *, subject_noun, base_seed, labels, **kwargs):
        views = [
            {
                "label": "back",
                "azimuth_deg": 180.0,
                "elevation_deg": 0.0,
                "rgba": _alpha_disc_image(),
                "raw_bytes": b"raw-back",
                "raw_payload_md5": "0" * 32,
                "seed": base_seed,
                "row_consistency": corrected_report,
                "row_corrected": True,
            },
            {
                "label": "side_left",
                "azimuth_deg": 90.0,
                "elevation_deg": 0.0,
                "rgba": _alpha_disc_image(),
                "raw_bytes": b"raw-left",
                "raw_payload_md5": "1" * 32,
                "seed": base_seed + 1,
                "row_consistency": consistent_report,
                "row_corrected": False,
            },
        ]
        records = [{"label": view["label"], "accepted": True} for view in views]
        return views, records, []

    monkeypatch.setattr(runtime, "_synthesize_geometry_views", _fake_synthesis)

    result = backend.i23d(
        _alpha_disc_image(),
        device="cpu",
        texture_mode="none",
        seed=11,
        geometry_conditioning="multiview",
    )

    metadata = result["metadata"]
    assert metadata["geometry_conditioning"]["applied"] == "multiview"
    tags = {row["tag"]: row for row in metadata["geometry_views"]}
    assert tags["back"]["row_consistency"] == corrected_report
    assert tags["back"]["row_corrected"] is True
    assert tags["left"]["row_consistency"] == consistent_report
    assert "row_corrected" not in tags["left"]  # only real corrections land
    assert "row_consistency" not in tags["front"]  # the user's own photo


def test_geometry_person_gate_fails_closed(monkeypatch) -> None:
    from abstract3d import captioning

    # Captioner unavailable + no hint evidence: refuse (an unavailable
    # check is not a permission grant).
    monkeypatch.setattr(captioning, "caption_image", lambda image, **kwargs: None)
    proceed, record = runtime._geometry_person_gate(
        _alpha_disc_image(), subject_hint=None, allow_person=False
    )
    assert proceed is False
    assert "captioner unavailable" in record["refusal"]

    # Person named by the caption: refuse without the acknowledgment...
    monkeypatch.setattr(
        captioning, "caption_image", lambda image, **kwargs: "a portrait of a man"
    )
    proceed, record = runtime._geometry_person_gate(
        _alpha_disc_image(), subject_hint=None, allow_person=False
    )
    assert proceed is False and record["person_detected"] is True

    # ...and proceed with it, warning on the record.
    proceed, record = runtime._geometry_person_gate(
        _alpha_disc_image(), subject_hint=None, allow_person=True
    )
    assert proceed is True and "person_warning" in record

    # A person-naming hint refuses without ever needing the captioner.
    proceed, record = runtime._geometry_person_gate(
        _alpha_disc_image(), subject_hint="a woman sitting", allow_person=False
    )
    assert proceed is False and record["person_detected"] is True


def test_generation_multiview_conditions_pipeline_with_tagged_views(
    monkeypatch, tmp_path
) -> None:
    backend = runtime.Hunyuan3DShapeBackend(owner=_composer_owner())
    capture: dict = {"seeds": []}
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=0.7)
    _install_fake_runtime(monkeypatch, backend, tmp_path, [sphere], capture)

    from abstract3d import captioning

    monkeypatch.setattr(
        captioning, "caption_image", lambda image, **kwargs: "a toy disc"
    )

    def _fake_synthesis(owner, source_rgba, *, subject_noun, base_seed, labels, **kwargs):
        views = [
            {
                "label": label,
                "azimuth_deg": azimuth,
                "elevation_deg": 0.0,
                "rgba": _alpha_disc_image(),
                "raw_bytes": b"raw-view-bytes",
                "raw_payload_md5": "0" * 32,
                "seed": base_seed + 50_000,
            }
            for label, azimuth in labels
        ]
        records = [{"label": label, "accepted": True} for label, _ in labels]
        return views, records, []

    monkeypatch.setattr(runtime, "_synthesize_geometry_views", _fake_synthesis)

    result = backend.i23d(
        _alpha_disc_image(),
        device="cpu",
        texture_mode="none",
        seed=11,
        geometry_conditioning="multiview",
        output_dir=str(tmp_path / "bundle"),
    )

    conditioning = capture["images"][0]
    assert isinstance(conditioning, dict)
    # The measured 4-view cliff caps conditioning at 3 tags: front (the
    # photo), back, one side; the second side is dropped (priority order)
    # but stays available to the texture lane.
    assert set(conditioning) == {"front", "back", "left"}
    metadata = result["metadata"]
    assert metadata["model_id"] == "tencent/Hunyuan3D-2mv/hunyuan3d-dit-v2-mv"
    assert metadata["multiview_conditioning"] is True
    record = metadata["geometry_conditioning"]
    assert record["requested"] == "multiview"
    assert record["applied"] == "multiview"
    assert record["fallback_reason"] is None
    tags = {row["tag"]: row for row in metadata["geometry_views"]}
    assert set(tags) == {"front", "back", "left"}
    assert tags["back"]["synthesized"] is True
    assert [row["tag"] for row in record["dropped_views"]] == ["right"]
    assert any("capped" in warning for warning in metadata["postprocess_warnings"])
    assert metadata["timings_s"]["geometry_view_synthesis"] is not None
    # Synthesized conditioning views are persisted for diagnosis.
    assert (tmp_path / "bundle" / "geometry_view_synthesized_back.png").exists()
    assert record["synthesized_view_paths"]


def test_generation_multiview_falls_back_loudly_when_gates_reject_all(
    monkeypatch, tmp_path
) -> None:
    backend = runtime.Hunyuan3DShapeBackend(owner=_composer_owner())
    capture: dict = {"seeds": []}
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=0.7)
    _install_fake_runtime(monkeypatch, backend, tmp_path, [sphere], capture)

    from abstract3d import captioning

    monkeypatch.setattr(
        captioning, "caption_image", lambda image, **kwargs: "a toy disc"
    )
    monkeypatch.setattr(
        runtime,
        "_synthesize_geometry_views",
        lambda *args, **kwargs: (
            [],
            [{"label": "back", "accepted": False, "failure": "gate"}],
            [],
        ),
    )

    result = backend.i23d(
        _alpha_disc_image(),
        device="cpu",
        texture_mode="none",
        seed=11,
        geometry_conditioning="multiview",
    )

    # Single-view fallback runs the exact known-good path: single image
    # conditioning on the 2.1 flagship.
    from PIL import Image

    assert isinstance(capture["images"][0], Image.Image)
    metadata = result["metadata"]
    assert metadata["model_id"] == "tencent/Hunyuan3D-2.1/hunyuan3d-dit-v2-1"
    assert metadata["multiview_conditioning"] is False
    record = metadata["geometry_conditioning"]
    assert record["applied"] == "single_view"
    assert "failed the acceptance gates" in record["fallback_reason"]
    assert any(
        "geometry_conditioning fell back to single-view" in warning
        for warning in metadata["postprocess_warnings"]
    )


def test_generation_multiview_refuses_person_subjects(monkeypatch, tmp_path) -> None:
    backend = runtime.Hunyuan3DShapeBackend(owner=_composer_owner())
    capture: dict = {"seeds": []}
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=0.7)
    _install_fake_runtime(monkeypatch, backend, tmp_path, [sphere], capture)

    from abstract3d import captioning

    monkeypatch.setattr(
        captioning, "caption_image", lambda image, **kwargs: "a portrait of a man"
    )

    def _forbidden_synthesis(*args, **kwargs):
        raise AssertionError("synthesis must not run for a person subject")

    monkeypatch.setattr(runtime, "_synthesize_geometry_views", _forbidden_synthesis)

    result = backend.i23d(
        _alpha_disc_image(),
        device="cpu",
        texture_mode="none",
        geometry_conditioning="multiview",
    )

    metadata = result["metadata"]
    record = metadata["geometry_conditioning"]
    assert record["applied"] == "single_view"
    assert record["person_check"]["person_detected"] is True
    assert "person" in record["fallback_reason"]
    assert metadata["model_id"] == "tencent/Hunyuan3D-2.1/hunyuan3d-dit-v2-1"


def test_generation_multiview_person_acknowledgment_proceeds(monkeypatch, tmp_path) -> None:
    backend = runtime.Hunyuan3DShapeBackend(owner=_composer_owner())
    capture: dict = {"seeds": []}
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=0.7)
    _install_fake_runtime(monkeypatch, backend, tmp_path, [sphere], capture)

    from abstract3d import captioning

    monkeypatch.setattr(
        captioning, "caption_image", lambda image, **kwargs: "a portrait of a man"
    )
    monkeypatch.setattr(
        runtime,
        "_synthesize_geometry_views",
        lambda owner, source_rgba, *, subject_noun, base_seed, labels, **kwargs: (
            [
                {
                    "label": "back",
                    "azimuth_deg": 180.0,
                    "elevation_deg": 0.0,
                    "rgba": _alpha_disc_image(),
                    "raw_bytes": b"raw",
                    "raw_payload_md5": "0" * 32,
                    "seed": base_seed,
                }
            ],
            [{"label": "back", "accepted": True}],
            [],
        ),
    )

    result = backend.i23d(
        _alpha_disc_image(),
        device="cpu",
        texture_mode="none",
        geometry_conditioning="multiview",
        texture_reference_allow_person=True,
    )

    metadata = result["metadata"]
    record = metadata["geometry_conditioning"]
    assert record["applied"] == "multiview"
    assert "person_warning" in record["person_check"]
    assert metadata["model_id"].startswith("tencent/Hunyuan3D-2mv")


def test_multiview_path_runs_mv_family_regime(monkeypatch, tmp_path) -> None:
    """The 2mv checkpoint is a 2.0-family model with its own validated
    regime (official snippet + checked face proof: 30 steps / octree 384).
    Running it at the flagship's 512/50 was measured catastrophic
    (2026-07-14 A/B: 822 raw bodies, euler +887 on the car), so the
    multiview path must swap the family defaults in — while explicit
    caller options still win."""
    from abstract3d import captioning

    def _fake_synthesis(owner, source_rgba, *, subject_noun, base_seed, labels, **kwargs):
        views = [
            {
                "label": "back",
                "azimuth_deg": 180.0,
                "elevation_deg": 0.0,
                "rgba": _alpha_disc_image(),
                "raw_bytes": b"raw",
                "raw_payload_md5": "0" * 32,
                "seed": base_seed,
            }
        ]
        return views, [{"label": "back", "accepted": True}], []

    monkeypatch.setattr(
        captioning, "caption_image", lambda image, **kwargs: "a toy disc"
    )
    monkeypatch.setattr(runtime, "_synthesize_geometry_views", _fake_synthesis)

    backend = runtime.Hunyuan3DShapeBackend(owner=_composer_owner())
    capture: dict = {"seeds": []}
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=0.7)
    _install_fake_runtime(monkeypatch, backend, tmp_path, [sphere], capture)
    backend.i23d(
        _alpha_disc_image(), device="cpu", texture_mode="none",
        geometry_conditioning="multiview",
    )
    assert capture["settings"][0] == {
        "num_inference_steps": runtime._MV_DEFAULT_NUM_INFERENCE_STEPS,
        "octree_resolution": runtime._MV_DEFAULT_OCTREE_RESOLUTION,
    }

    # Explicit options outrank the family default.
    backend = runtime.Hunyuan3DShapeBackend(owner=_composer_owner())
    capture = {"seeds": []}
    _install_fake_runtime(monkeypatch, backend, tmp_path / "b", [sphere], capture)
    backend.i23d(
        _alpha_disc_image(), device="cpu", texture_mode="none",
        geometry_conditioning="multiview", num_inference_steps=50,
        octree_resolution=512,
    )
    assert capture["settings"][0] == {
        "num_inference_steps": 50,
        "octree_resolution": 512,
    }

    # The default single-view path keeps the flagship regime.
    backend = runtime.Hunyuan3DShapeBackend(owner=None)
    capture = {"seeds": []}
    _install_fake_runtime(monkeypatch, backend, tmp_path / "c", [sphere], capture)
    backend.i23d(_alpha_disc_image(), device="cpu", texture_mode="none")
    assert capture["settings"][0] == {
        "num_inference_steps": runtime._DEFAULT_NUM_INFERENCE_STEPS,
        "octree_resolution": runtime._DEFAULT_OCTREE_RESOLUTION,
    }


def test_caller_synthesized_references_protected_without_ab_gate(
    monkeypatch, tmp_path
) -> None:
    """Explicit references flagged synthesized (backlog 0017) ride the auto
    lane's in-bake protection (the per-view `generated` flag) WITHOUT arming
    the auto lane's A/B acceptance machinery: exactly one bake runs, the
    per-reference authority lands in texture_artifacts, filename inference
    marks the pipeline's own generated files, and caller files are never
    re-persisted under the generated naming."""
    import skimage.color.colorconv as _colorconv

    # Same host-only Accelerate workaround as the texture-stage test above
    # (torch tensor work + lab conversions in one process).
    def _convert_einsum(matrix, arr):
        arr = _colorconv._prepare_colorarray(arr)
        return np.einsum("...i,ji->...j", arr, matrix.astype(arr.dtype))

    monkeypatch.setattr(_colorconv, "_convert", _convert_einsum)

    backend = runtime.Hunyuan3DShapeBackend(owner=None)
    capture: dict = {"seeds": []}
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=0.7)
    _install_fake_runtime(monkeypatch, backend, tmp_path, [sphere], capture)

    # One reference synthesized BY FILENAME (a pipeline-generated file fed
    # back), one real photo (PIL payload: no filename, never inferred).
    synthesized_path = tmp_path / "texture_reference_generated_back.png"
    _alpha_disc_image().save(synthesized_path)

    from abstract3d import texturing

    bake_view_labels: list[list] = []
    real_bake = texturing.bake_projection_texture

    def spy_bake(mesh, **kwargs):
        bake_view_labels.append(
            [(view.get("label"), bool(view.get("generated")))
             for view in kwargs["observed_views"]])
        return real_bake(mesh, **kwargs)

    monkeypatch.setattr(texturing, "bake_projection_texture", spy_bake)

    bundle_dir = tmp_path / "bundle"
    result = backend.i23d(
        _alpha_disc_image(),
        device="cpu",
        texture_mode="baked_basecolor",
        texture_resolution=64,
        output_dir=str(bundle_dir),
        texture_reference_images=[str(synthesized_path), _alpha_disc_image()],
        texture_reference_angles=["back", "side_left"],
    )

    # Exactly ONE bake: caller-provided synthesized witnesses are the
    # operator's explicit request — no baseline A/B second-guesses them.
    assert bake_view_labels == [
        [("front", False), ("back", True), ("side_left", False)]
    ]

    metadata = result["metadata"]
    artifacts = metadata["texture_artifacts"]
    authority = {row["label"]: row for row in artifacts["reference_authority"]}
    assert authority["front"]["authority"] == "full"
    assert authority["front"]["role"] == "source"
    assert authority["back"]["authority"] == "protected_completion_only"
    assert authority["back"]["synthesized"] is True
    assert authority["back"]["synthesized_source"] == "filename_inference"
    assert authority["side_left"]["authority"] == "full"
    assert artifacts["generated_protection"]["applied"] is True
    assert "back" in artifacts["generated_protection"]["zeroed_by_view"]
    assert authority["back"]["protection"]["zeroed_texels"] == (
        artifacts["generated_protection"]["zeroed_by_view"]["back"])
    # The auto lane's report never appears (nothing was generated).
    assert "reference_generation" not in artifacts
    # Inference is provenance-noted for the report.
    assert any("treated as synthesized" in note for note in metadata["notes"])
    # Caller files are not re-persisted under the pipeline's generated naming.
    assert "generated_reference_paths" not in artifacts
    assert not list(bundle_dir.glob("texture_reference_generated_*.png"))


def test_caller_references_also_respect_view_cap(monkeypatch, tmp_path) -> None:
    """Four caller-tagged references hit the same measured 4-view cliff:
    the cap drops the lowest-priority tag with a loud warning (the
    pre-existing uncapped behavior shipped the shredding regime)."""
    backend = runtime.Hunyuan3DShapeBackend(owner=None)
    capture: dict = {"seeds": []}
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=0.7)
    _install_fake_runtime(monkeypatch, backend, tmp_path, [sphere], capture)

    result = backend.i23d(
        _alpha_disc_image(),
        device="cpu",
        texture_mode="none",
        model="tencent/Hunyuan3D-2mv",
        texture_reference_images=[_alpha_disc_image() for _ in range(3)],
        texture_reference_angles=["side_left", "side_right", "back"],
    )

    conditioning = capture["images"][0]
    assert isinstance(conditioning, dict)
    assert set(conditioning) == {"front", "back", "left"}
    metadata = result["metadata"]
    assert metadata["multiview_conditioning"] is True
    tags = {row["tag"] for row in metadata["geometry_views"]}
    assert tags == {"front", "back", "left"}
    assert any("capped" in warning for warning in metadata["postprocess_warnings"])


def test_explicit_mv_model_with_references_runs_mv_family_regime(
    monkeypatch, tmp_path
) -> None:
    """The pre-existing caller-reference 2mv route gets the same family
    regime (it previously inherited the flagship 512/50)."""
    backend = runtime.Hunyuan3DShapeBackend(owner=None)
    capture: dict = {"seeds": []}
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=0.7)
    _install_fake_runtime(monkeypatch, backend, tmp_path, [sphere], capture)

    result = backend.i23d(
        _alpha_disc_image(),
        device="cpu",
        texture_mode="none",
        model="tencent/Hunyuan3D-2mv",
        texture_reference_images=[_alpha_disc_image()],
        texture_reference_angles=["side_left"],
    )

    assert capture["settings"][0] == {
        "num_inference_steps": runtime._MV_DEFAULT_NUM_INFERENCE_STEPS,
        "octree_resolution": runtime._MV_DEFAULT_OCTREE_RESOLUTION,
    }
    assert result["metadata"]["multiview_conditioning"] is True


def test_geometry_conditioning_option_validation(monkeypatch, tmp_path) -> None:
    from abstract3d.errors import InvalidRequestError

    backend = runtime.Hunyuan3DShapeBackend(owner=None)
    capture: dict = {"seeds": []}
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=0.7)
    _install_fake_runtime(monkeypatch, backend, tmp_path, [sphere], capture)

    with pytest.raises(InvalidRequestError, match="geometry_conditioning"):
        backend.i23d(
            _alpha_disc_image(), device="cpu", texture_mode="none",
            geometry_conditioning="frobnicate",
        )
    # Validation failed loudly BEFORE any diffusion draw.
    assert capture["seeds"] == []


def test_geometry_conditioning_multiview_rejects_explicit_flagship_model(
    monkeypatch, tmp_path
) -> None:
    from abstract3d.errors import InvalidRequestError

    backend = runtime.Hunyuan3DShapeBackend(owner=_composer_owner())
    capture: dict = {"seeds": []}
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=0.7)
    _install_fake_runtime(monkeypatch, backend, tmp_path, [sphere], capture)

    with pytest.raises(InvalidRequestError, match="multi-view"):
        backend.i23d(
            _alpha_disc_image(), device="cpu", texture_mode="none",
            model="tencent/Hunyuan3D-2.1", geometry_conditioning="multiview",
        )
    assert capture["seeds"] == []


def test_geometry_conditioning_auto_falls_back_without_provider(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.delenv("ABSTRACT3D_IMAGE_PROVIDER", raising=False)
    backend = runtime.Hunyuan3DShapeBackend(owner=None)
    capture: dict = {"seeds": []}
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=0.7)
    _install_fake_runtime(monkeypatch, backend, tmp_path, [sphere], capture)

    def _forbidden_synthesis(*args, **kwargs):
        raise AssertionError("auto must not synthesize without an explicit provider")

    monkeypatch.setattr(runtime, "_synthesize_geometry_views", _forbidden_synthesis)

    result = backend.i23d(
        _alpha_disc_image(), device="cpu", texture_mode="none",
        geometry_conditioning="auto",
    )

    metadata = result["metadata"]
    record = metadata["geometry_conditioning"]
    assert record["requested"] == "auto"
    assert record["applied"] == "single_view"
    assert record["fallback_reason"]
    assert metadata["model_id"] == "tencent/Hunyuan3D-2.1/hunyuan3d-dit-v2-1"


def test_single_mode_never_touches_synthesis(monkeypatch, tmp_path) -> None:
    """The default path (geometry_conditioning unset) must be EXACTLY the
    historical single-view flow: no person gate, no synthesis, no new
    metadata keys."""
    backend = runtime.Hunyuan3DShapeBackend(owner=None)
    capture: dict = {"seeds": []}
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=0.7)
    _install_fake_runtime(monkeypatch, backend, tmp_path, [sphere], capture)

    def _forbidden(*args, **kwargs):
        raise AssertionError("geometry conditioning must not run on the default path")

    monkeypatch.setattr(runtime, "_synthesize_geometry_views", _forbidden)
    monkeypatch.setattr(runtime, "_geometry_person_gate", _forbidden)

    result = backend.i23d(_alpha_disc_image(), device="cpu", texture_mode="none", seed=42)

    metadata = result["metadata"]
    assert "geometry_conditioning" not in metadata
    assert "geometry_view_synthesis" not in metadata["timings_s"]
    assert metadata["multiview_conditioning"] is False


def test_list_operations_schema_exposes_geometry_conditioning() -> None:
    backend = runtime.Hunyuan3DShapeBackend(owner=None)
    operations = backend.list_operations()
    schema = operations[-1]["parameter_schema"]["properties"]
    assert schema["geometry_conditioning"]["enum"] == [
        "single", "multiview", "auto", "loop",
    ]


def test_generate_references_with_replay_serves_synthesized_bytes_first(
    monkeypatch,
) -> None:
    """The replay generator hands the pre-shape view to the texture lane's
    FIRST ladder attempt and delegates every later attempt to the real
    generator; reports from per-angle calls merge into one record."""
    from abstract3d import reference_generation as refgen

    served: dict = {}

    def _fake_generate(mesh, source, *, owner=None, angles=(), image_generator=None, **kwargs):
        label = str(angles[0][0])
        if image_generator is None:
            served[label] = ["no-generator"]
            return (
                [{"label": label}],
                {"angles": [{"label": label}], "accepted": 1, "rejected": 0},
            )
        first = image_generator("prompt", b"conditioning", seed=1)
        second = image_generator("prompt", b"conditioning", seed=2)
        served[label] = [first, second]
        return (
            [{"label": label}],
            {"angles": [{"label": label}], "accepted": 1, "rejected": 0},
        )

    monkeypatch.setattr(refgen, "generate_reference_views", _fake_generate)
    monkeypatch.setattr(
        refgen,
        "default_i2i_generator",
        lambda owner: (lambda prompt, image, **kwargs: b"fresh-generation"),
    )

    views, report = runtime._generate_references_with_replay(
        object(),
        object(),
        owner=None,
        angles=(("back", 180.0, 0.0), ("top", 0.0, 55.0)),
        replay_sources={"back": b"replayed-bytes"},
    )

    assert served["back"] == [b"replayed-bytes", b"fresh-generation"]
    assert served["top"] == ["no-generator"]
    assert [view["label"] for view in views] == ["back", "top"]
    assert report["accepted"] == 2
    assert report["replayed_labels"] == ["back"]
    assert [row["label"] for row in report["angles"]] == ["back", "top"]


# -- loop conditioning (the calibrated two-pass bust recipe) -------------------


def _loop_owner():
    """Composer owner with an explicitly PINNED local image provider (the
    loop's provider requirement) — loop must never resolve owner=None."""
    from types import SimpleNamespace

    return SimpleNamespace(
        config={"scene3d_image_provider": "mlx-gen"},
        vision=SimpleNamespace(
            t2i=lambda *args, **kwargs: b"",
            i2i=lambda *args, **kwargs: b"",
        ),
    )


def _fake_loop_plan(*, side_left_pose_delta: float = 7.5):
    """Scripted synthesize_loop_views: three accepted views; side_left's
    measured azimuth is `82.5` (within the pose gate by default). Windowed
    images are DISTINCT objects from full-span ones so consumer-separation
    asserts on identity. Returns (fake_fn, state)."""
    state: dict = {}

    def fake(mesh0, source_rgba, *, owner, angles, seed, subject_hint,
             person_attested, image_request, view_consistency, **kwargs):
        state["owner"] = owner
        state["seed"] = seed
        state["image_request"] = dict(image_request)
        state["person_attested"] = person_attested
        views = []
        windowed = {}
        for label, azimuth, _elev in angles:
            full = _alpha_disc_image()
            win = _alpha_disc_image(48)
            windowed[label] = win
            measured = azimuth
            eligible = True
            refusal = None
            if label == "side_left":
                measured = azimuth - side_left_pose_delta
                if side_left_pose_delta > 20.0:
                    eligible = False
                    refusal = "pose honesty: measured azimuth off the declared angle"
            views.append({
                "label": label,
                "azimuth_deg": float(azimuth),
                "elevation_deg": 0.0,
                "rgba": full,
                "raw_bytes": f"raw-{label}".encode(),
                "raw_payload_md5": "f" * 32,
                "seed": int(seed),
                "clay_render": None,
                "conditioning_eligible": eligible,
                "bake_eligible": True,
                "measured_azimuth_deg": float(measured),
                "pose": {"measurable": True, "measured_deg": float(measured),
                         "delta_deg": abs(float(azimuth) - float(measured))},
                "row_consistency": {"verdict": "consistent", "score": 0.6},
                **({"windowed_rgba": win} if eligible else {}),
                **({"conditioning_refusal": refusal} if refusal else {}),
            })
        state["views"] = views
        state["front_windowed"] = _alpha_disc_image(48)
        state["windowed"] = windowed
        return {
            "views": views,
            "refgen_report": {"accepted": len(views), "rejected": 0, "angles": []},
            "front_windowed": state["front_windowed"],
            "window": {"k": 0.22, "per_view": {}, "kept_ratio_spread": 0.003},
            "seconds": 1.2,
        }

    return fake, state


def _stub_loop_self_verification(monkeypatch, *, suspect: bool = False):
    record = {
        "duplication": {
            "duplication_suspect": suspect,
            "peak_ratio": 0.071 if suspect else 0.02,
            "peak_lag_px": 46 if suspect else None,
            "threshold": 0.05,
        },
        "oblique": {},
        "textured_rendered": False,
    }
    monkeypatch.setattr(
        "abstract3d.loop_conditioning.self_verification",
        lambda mesh, *, textured, **kw: (record, {}),
    )
    return record


def test_loop_mode_two_pass_sequencing_and_windowed_conditioning(
    monkeypatch, tmp_path
) -> None:
    """PASS 1 conditions on the front alone (tagged one-entry dict), PASS 2
    conditions on the WINDOWED front + windowed survivors (capped at 3
    tags), and the loop record carries pass-1 stats + both timings."""
    backend = runtime.Hunyuan3DShapeBackend(owner=_loop_owner())
    capture: dict = {"seeds": []}
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=0.7)
    _install_fake_runtime(monkeypatch, backend, tmp_path, [sphere, sphere], capture)

    from abstract3d import captioning

    monkeypatch.setattr(captioning, "caption_image", lambda image, **kw: "a toy disc")
    fake_plan, state = _fake_loop_plan()
    monkeypatch.setattr("abstract3d.loop_conditioning.synthesize_loop_views", fake_plan)
    _stub_loop_self_verification(monkeypatch)

    result = backend.i23d(
        _alpha_disc_image(),
        device="cpu",
        texture_mode="none",
        seed=11,
        geometry_conditioning="loop",
        output_dir=str(tmp_path / "bundle"),
    )

    # Two DiT invocations: pass 1 (front photo alone, PLAIN image on the
    # flagship single-view checkpoint — a single-tag 2mv dict was measured
    # to shred, e22 forensics) then pass 2 (windowed set on 2mv).
    assert len(capture["images"]) == 2
    pass1 = capture["images"][0]
    assert not isinstance(pass1, dict)
    pass2 = capture["images"][1]
    assert isinstance(pass2, dict)
    # Cap at 3 tags: front + back + left (priority order drops "right").
    assert set(pass2) == {"front", "back", "left"}
    # THE SPLIT-CONSUMER INVARIANT, conditioning half: pass 2 sees the
    # WINDOWED image objects, never the full-span ones.
    assert pass2["front"] is state["front_windowed"]
    assert pass2["back"] is state["windowed"]["back"]
    assert pass2["left"] is state["windowed"]["side_left"]
    full_span = {id(view["rgba"]) for view in state["views"]}
    assert all(id(image) not in full_span for image in pass2.values())
    # Both draws at the base seed (different conditioning = different draw).
    assert capture["seeds"] == [11, 11]

    metadata = result["metadata"]
    record = metadata["geometry_conditioning"]
    assert record["requested"] == "loop"
    assert record["applied"] == "loop"
    assert record["fallback_reason"] is None
    assert record["pass1"]["seed"] == 11
    assert record["pass1"]["inference_s"] >= 0.0
    assert record["window"]["k"] == 0.22
    assert len(record["loop_views"]) == 3
    assert metadata["timings_s"]["pass1_inference"] is not None
    assert metadata["timings_s"]["geometry_view_synthesis"] == 1.2
    assert metadata["model_id"].startswith("tencent/Hunyuan3D-2mv")
    assert metadata["multiview_conditioning"] is True
    assert metadata["loop_self_verification"]["duplication"]["duplication_suspect"] is False
    tags = {row["tag"]: row for row in metadata["geometry_views"]}
    assert tags["front"]["windowed"] is True
    assert tags["back"]["windowed"] is True
    assert tags["left"]["measured_azimuth_deg"] == 82.5
    # Provider pinning: the plan received the owner-resolved request and
    # the loop seed offset.
    assert state["image_request"].get("provider") == "mlx-gen"
    assert state["owner"] is backend._owner
    assert state["seed"] == 11 + runtime._LOOP_VIEW_SEED_OFFSET
    # Windowed conditioning pixels persist for diagnosis.
    assert (tmp_path / "bundle" / "geometry_view_windowed_front.png").exists()
    assert (tmp_path / "bundle" / "geometry_view_windowed_back.png").exists()
    # Full-span views persist under the synthesized naming (bake witnesses).
    assert (tmp_path / "bundle" / "geometry_view_synthesized_back.png").exists()


def test_loop_mode_person_gate_refuses_loudly_and_attestation_proceeds(
    monkeypatch, tmp_path
) -> None:
    from abstract3d.errors import InvalidRequestError
    from abstract3d import captioning

    backend = runtime.Hunyuan3DShapeBackend(owner=_loop_owner())
    capture: dict = {"seeds": []}
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=0.7)
    _install_fake_runtime(monkeypatch, backend, tmp_path, [sphere, sphere], capture)
    monkeypatch.setattr(
        captioning, "caption_image", lambda image, **kw: "a portrait of a man")

    with pytest.raises(InvalidRequestError, match="person"):
        backend.i23d(
            _alpha_disc_image(), device="cpu", texture_mode="none",
            geometry_conditioning="loop",
        )
    # Refused BEFORE any DiT draw (loop must not burn a pass-1 on it).
    assert capture["seeds"] == []

    fake_plan, state = _fake_loop_plan()
    monkeypatch.setattr("abstract3d.loop_conditioning.synthesize_loop_views", fake_plan)
    _stub_loop_self_verification(monkeypatch)
    result = backend.i23d(
        _alpha_disc_image(), device="cpu", texture_mode="none",
        geometry_conditioning="loop", texture_reference_allow_person=True,
    )
    assert result["metadata"]["geometry_conditioning"]["applied"] == "loop"
    assert state["person_attested"] is True
    assert "person_warning" in result["metadata"]["geometry_conditioning"]["person_check"]


def test_loop_mode_requires_local_provider_and_mv_model(monkeypatch, tmp_path) -> None:
    from abstract3d.errors import InvalidRequestError

    monkeypatch.delenv("ABSTRACT3D_IMAGE_PROVIDER", raising=False)
    # No provider pin anywhere: loop refuses (never the remote default).
    from types import SimpleNamespace

    composer_only = SimpleNamespace(
        config={}, vision=SimpleNamespace(t2i=lambda *a, **k: b""))
    backend = runtime.Hunyuan3DShapeBackend(owner=composer_only)
    capture: dict = {"seeds": []}
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=0.7)
    _install_fake_runtime(monkeypatch, backend, tmp_path, [sphere], capture)
    with pytest.raises(InvalidRequestError, match="local image provider"):
        backend.i23d(
            _alpha_disc_image(), device="cpu", texture_mode="none",
            geometry_conditioning="loop",
        )
    assert capture["seeds"] == []

    # An explicit flagship model contradicts the loop (2mv both passes).
    backend = runtime.Hunyuan3DShapeBackend(owner=_loop_owner())
    _install_fake_runtime(monkeypatch, backend, tmp_path / "b", [sphere], capture)
    with pytest.raises(InvalidRequestError, match="multi-view"):
        backend.i23d(
            _alpha_disc_image(), device="cpu", texture_mode="none",
            model="tencent/Hunyuan3D-2.1", geometry_conditioning="loop",
        )
    assert capture["seeds"] == []


def test_loop_mode_rejects_explicit_reference_views(monkeypatch, tmp_path) -> None:
    from abstract3d.errors import InvalidRequestError

    backend = runtime.Hunyuan3DShapeBackend(owner=_loop_owner())
    capture: dict = {"seeds": []}
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=0.7)
    _install_fake_runtime(monkeypatch, backend, tmp_path, [sphere], capture)

    with pytest.raises(InvalidRequestError, match="loop"):
        backend.i23d(
            _alpha_disc_image(), device="cpu", texture_mode="none",
            geometry_conditioning="loop",
            texture_reference_images=[_alpha_disc_image()],
            texture_reference_angles=["side_left"],
        )
    assert capture["seeds"] == []


def test_loop_mode_refuses_unwindowable_subject_before_pass1(
    monkeypatch, tmp_path
) -> None:
    """The window law anchors on a head-to-shoulder span; a subject it
    cannot fit refuses in milliseconds — never after a pass-1 DiT draw."""
    from abstract3d.errors import InvalidRequestError
    from abstract3d import captioning
    from PIL import Image

    backend = runtime.Hunyuan3DShapeBackend(owner=_loop_owner())
    capture: dict = {"seeds": []}
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=0.7)
    _install_fake_runtime(monkeypatch, backend, tmp_path, [sphere], capture)
    monkeypatch.setattr(captioning, "caption_image", lambda image, **kw: "a toy bar")

    # A uniform-width bar: the shoulder anchor lands one row under the
    # head top (degenerate span).
    bar = np.zeros((96, 96, 4), dtype=np.uint8)
    bar[40:60, 8:88] = (180, 160, 140, 255)

    with pytest.raises(InvalidRequestError, match="windowable"):
        backend.i23d(
            Image.fromarray(bar, "RGBA"), device="cpu", texture_mode="none",
            geometry_conditioning="loop",
        )
    assert capture["seeds"] == []


def test_loop_mode_pose_refused_view_conditions_nothing_but_rides_to_bake(
    monkeypatch, tmp_path
) -> None:
    """A view measured >20 deg off its declared angle may not condition
    (trained tag positions) but re-declares to the MEASURED azimuth for
    the texture lane."""
    backend = runtime.Hunyuan3DShapeBackend(owner=_loop_owner())
    capture: dict = {"seeds": []}
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=0.7)
    _install_fake_runtime(monkeypatch, backend, tmp_path, [sphere, sphere], capture)

    from abstract3d import captioning

    monkeypatch.setattr(captioning, "caption_image", lambda image, **kw: "a toy disc")
    fake_plan, state = _fake_loop_plan(side_left_pose_delta=40.0)
    monkeypatch.setattr("abstract3d.loop_conditioning.synthesize_loop_views", fake_plan)
    _stub_loop_self_verification(monkeypatch)

    result = backend.i23d(
        _alpha_disc_image(),
        device="cpu",
        texture_mode="none",
        seed=11,
        geometry_conditioning="loop",
    )

    pass2 = capture["images"][1]
    # side_left refused from conditioning: back fills its tag; side_right
    # takes "right" (cap keeps front/back/right when left never claimed).
    assert "left" not in pass2
    assert set(pass2) == {"front", "back", "right"}
    rows = {row["label"]: row
            for row in result["metadata"]["geometry_conditioning"]["loop_views"]}
    assert rows["side_left"]["conditioning_eligible"] is False
    assert rows["side_left"]["bake_eligible"] is True
    assert rows["side_left"]["measured_azimuth_deg"] == 50.0
    assert "pose honesty" in rows["side_left"]["conditioning_refusal"]


def test_loop_mode_zero_eligible_views_raises_loudly(monkeypatch, tmp_path) -> None:
    from abstract3d.errors import Abstract3DError
    from abstract3d import captioning

    backend = runtime.Hunyuan3DShapeBackend(owner=_loop_owner())
    capture: dict = {"seeds": []}
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=0.7)
    _install_fake_runtime(monkeypatch, backend, tmp_path, [sphere, sphere], capture)
    monkeypatch.setattr(captioning, "caption_image", lambda image, **kw: "a toy disc")

    def empty_plan(mesh0, source_rgba, **kwargs):
        return {
            "views": [],
            "refgen_report": {"accepted": 0, "rejected": 3, "angles": []},
            "front_windowed": _alpha_disc_image(48),
            "window": {"k": 0.22, "per_view": {}, "kept_ratio_spread": None},
            "seconds": 0.4,
        }

    monkeypatch.setattr("abstract3d.loop_conditioning.synthesize_loop_views", empty_plan)

    with pytest.raises(Abstract3DError, match="no eligible"):
        backend.i23d(
            _alpha_disc_image(), device="cpu", texture_mode="none",
            geometry_conditioning="loop",
        )
    # Pass 1 ran (its draw is on the record); pass 2 never started.
    assert len(capture["seeds"]) == 1


def test_loop_mode_refusal_persists_forensics_bundle(monkeypatch, tmp_path) -> None:
    """A loop refusal must leave the evidence on disk AND still raise
    (e22, 2026-07-21: an ~80-minute run refused with every per-attempt
    reason only in memory — no bundle, no rejected pixels, no pass-1
    mesh). The refusal bundle carries the per-attempt refgen report, the
    plan rows, the pass-1 mesh + its clay renders, and the rejected view
    images including the raw payloads."""
    from abstract3d.errors import Abstract3DError
    from abstract3d import captioning

    backend = runtime.Hunyuan3DShapeBackend(owner=_loop_owner())
    capture: dict = {"seeds": []}
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=0.7)
    _install_fake_runtime(monkeypatch, backend, tmp_path, [sphere, sphere], capture)
    monkeypatch.setattr(captioning, "caption_image", lambda image, **kw: "a toy disc")

    raw_payload = b"\x89PNG-fake-raw-draw"

    def rejecting_plan(mesh0, source_rgba, **kwargs):
        # Every draw rejected by the ladder: the refgen report carries the
        # per-attempt gate verdicts and the rejected pixels (downscaled +
        # raw), exactly what generate_reference_views produces when an
        # angle exhausts its budget.
        return {
            "views": [],
            "refgen_report": {
                "accepted": 0,
                "rejected": 3,
                "angles": [
                    {
                        "label": "back",
                        "azimuth_deg": 180.0,
                        "accepted": False,
                        "attempts": [
                            {
                                "seed": 72025,
                                "silhouette_iou": 0.81,
                                "failure_family": "speculars",
                                "speculars": {"passed": False,
                                              "worst_blob_fraction": 0.041},
                            }
                        ],
                        "rejection_reason": "floor-only candidates",
                    }
                ],
                "rejected_images": [
                    {
                        "label": "back",
                        "attempt": 0,
                        "image": _alpha_disc_image(64),
                        "raw_bytes": raw_payload,
                        "seed": 72025,
                        "failure_family": "speculars",
                    }
                ],
            },
            "front_windowed": _alpha_disc_image(48),
            "window": {"k": 0.22, "per_view": {}, "kept_ratio_spread": None},
            "seconds": 0.4,
        }

    monkeypatch.setattr(
        "abstract3d.loop_conditioning.synthesize_loop_views", rejecting_plan)

    bundle_dir = tmp_path / "refused_bundle"
    with pytest.raises(Abstract3DError, match="no eligible"):
        backend.i23d(
            _alpha_disc_image(), device="cpu", texture_mode="none",
            geometry_conditioning="loop", output_dir=str(bundle_dir),
        )

    # The refusal report: status, error, and the full per-attempt record.
    report_path = bundle_dir / "refusal_report.json"
    assert report_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "refused"
    assert "no eligible conditioning views" in report["error"]
    record = report["geometry_conditioning"]
    angle_row = record["reference_generation"]["angles"][0]
    assert angle_row["attempts"][0]["failure_family"] == "speculars"
    assert angle_row["attempts"][0]["speculars"]["worst_blob_fraction"] == 0.041
    assert record["reference_generation"]["rejected"] == 3
    assert record["pass1"]["seed"] == runtime._DEFAULT_SEED
    # No pixel/byte payloads may leak into the JSON.
    assert "rejected_images" not in record.get("reference_generation", {})
    assert "raw_bytes" not in json.dumps(report)

    # Pass-1 mesh + the clay renders the gates judged against.
    assert (bundle_dir / "pass1_mesh.glb").stat().st_size > 0
    for label in ("front", "side_left", "side_right", "back"):
        assert (bundle_dir / f"pass1_clay_{label}.png").exists()

    # Rejected pixels: downscaled triage copy + full raw payload.
    assert (bundle_dir / "rejected_geometry_views" / "back_a0.webp").exists()
    raw_path = (
        bundle_dir / "rejected_geometry_views" / "raw" / "back_a0_seed72025.png")
    assert raw_path.read_bytes() == raw_payload


def test_loop_mode_pass1_pinned_to_flagship_checkpoint_and_regime(
    monkeypatch, tmp_path
) -> None:
    """The pass-1 scaffold draw runs the FLAGSHIP single-view checkpoint at
    its own validated regime (512/50) regardless of the run's knobs: a
    single-front-tag 2mv draw shreds at every regime (e22 forensics:
    512/50 → 295-component debris, photo IoU 0.415; 384/30 → 20 bodies,
    IoU 0.226; flagship → 1 body, IoU 0.776). Pass 2 keeps the run's
    knobs; the divergence is warned; the pin is recorded."""
    from abstract3d import captioning

    backend = runtime.Hunyuan3DShapeBackend(owner=_loop_owner())
    capture: dict = {"seeds": []}
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=0.7)
    _install_fake_runtime(monkeypatch, backend, tmp_path, [sphere, sphere], capture)
    monkeypatch.setattr(captioning, "caption_image", lambda image, **kw: "a toy disc")
    fake_plan, _state = _fake_loop_plan()
    monkeypatch.setattr("abstract3d.loop_conditioning.synthesize_loop_views", fake_plan)
    _stub_loop_self_verification(monkeypatch)

    # No explicit knobs: pass 2 resolves the 2mv family defaults (384/30)
    # while pass 1 pins the flagship regime (512/50) — divergence warned.
    result = backend.i23d(
        _alpha_disc_image(), device="cpu", texture_mode="none",
        geometry_conditioning="loop",
    )
    assert capture["settings"][0] == {
        "num_inference_steps": runtime._DEFAULT_NUM_INFERENCE_STEPS,
        "octree_resolution": runtime._DEFAULT_OCTREE_RESOLUTION,
    }
    assert capture["settings"][1] == {
        "num_inference_steps": runtime._MV_DEFAULT_NUM_INFERENCE_STEPS,
        "octree_resolution": runtime._MV_DEFAULT_OCTREE_RESOLUTION,
    }
    metadata = result["metadata"]
    record = metadata["geometry_conditioning"]
    assert record["pass1"]["model_id"] == runtime._OFFICIAL_MODEL_ID
    assert record["pass1"]["num_inference_steps"] == (
        runtime._DEFAULT_NUM_INFERENCE_STEPS)
    assert any("pass 1 ran the flagship single-view regime" in warning
               for warning in metadata["postprocess_warnings"])
    # The shipped mesh's model stays the 2mv checkpoint (pass 2).
    assert metadata["model_id"].startswith("tencent/Hunyuan3D-2mv")
    # Scaffold health is measured and recorded on the healthy path too.
    health = record["scaffold_health"]
    assert health["measured"] is True
    assert health["healthy"] is True
    assert health["photo_vs_front_clay_iou"] >= 0.60

    # e22's exact knobs (512/50): pass 1 and pass 2 coincide — no warning.
    capture["seeds"].clear()
    capture["images"] = []
    capture["settings"] = []
    result = backend.i23d(
        _alpha_disc_image(), device="cpu", texture_mode="none",
        geometry_conditioning="loop",
        num_inference_steps=50, octree_resolution=512,
    )
    assert capture["settings"] == [
        {"num_inference_steps": 50, "octree_resolution": 512},
        {"num_inference_steps": 50, "octree_resolution": 512},
    ]
    assert not any("pass 1 ran the flagship" in warning
                   for warning in result["metadata"]["postprocess_warnings"])


def test_loop_mode_unhealthy_scaffold_refuses_fast_with_forensics(
    monkeypatch, tmp_path
) -> None:
    """A pass-1 scaffold that cannot register onto its own source photo
    (the e22 shredding class) refuses BEFORE any i2i draw — with the IoU
    in the error and the forensics bundle on disk."""
    from abstract3d.errors import Abstract3DError
    from abstract3d import captioning

    backend = runtime.Hunyuan3DShapeBackend(owner=_loop_owner())
    capture: dict = {"seeds": []}
    # A thin sliver: its front clay cannot register onto the disc photo.
    sliver = trimesh.creation.box(extents=(0.03, 1.9, 0.03))
    _install_fake_runtime(monkeypatch, backend, tmp_path, [sliver, sliver], capture)
    monkeypatch.setattr(captioning, "caption_image", lambda image, **kw: "a toy disc")

    def must_not_synthesize(*args, **kwargs):
        raise AssertionError("no i2i draw may run on an unhealthy scaffold")

    monkeypatch.setattr(
        "abstract3d.loop_conditioning.synthesize_loop_views", must_not_synthesize)

    bundle_dir = tmp_path / "unhealthy_bundle"
    with pytest.raises(Abstract3DError, match="scaffold mesh does not match"):
        backend.i23d(
            _alpha_disc_image(), device="cpu", texture_mode="none",
            geometry_conditioning="loop", output_dir=str(bundle_dir),
        )

    # Only the pass-1 draw ran; the refusal persisted its evidence.
    assert len(capture["seeds"]) == 1
    report = json.loads(
        (bundle_dir / "refusal_report.json").read_text(encoding="utf-8"))
    assert "scaffold mesh does not match" in report["error"]
    health = report["geometry_conditioning"]["scaffold_health"]
    assert health["healthy"] is False
    assert health["photo_vs_front_clay_iou"] < 0.60
    assert (bundle_dir / "pass1_mesh.glb").stat().st_size > 0


def test_loop_mode_crash_after_pass1_persists_partial_forensics(
    monkeypatch, tmp_path
) -> None:
    """Any later loop failure (not just the no-eligible-views refusal)
    persists what exists so far: a synthesis crash after pass 1 leaves the
    pass-1 mesh + record on disk and re-raises the original error."""
    from abstract3d import captioning

    backend = runtime.Hunyuan3DShapeBackend(owner=_loop_owner())
    capture: dict = {"seeds": []}
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=0.7)
    _install_fake_runtime(monkeypatch, backend, tmp_path, [sphere, sphere], capture)
    monkeypatch.setattr(captioning, "caption_image", lambda image, **kw: "a toy disc")

    def crashing_plan(mesh0, source_rgba, **kwargs):
        raise RuntimeError("synthesis stack fell over mid-ladder")

    monkeypatch.setattr(
        "abstract3d.loop_conditioning.synthesize_loop_views", crashing_plan)

    bundle_dir = tmp_path / "crashed_bundle"
    with pytest.raises(RuntimeError, match="mid-ladder"):
        backend.i23d(
            _alpha_disc_image(), device="cpu", texture_mode="none",
            geometry_conditioning="loop", output_dir=str(bundle_dir),
        )

    report = json.loads(
        (bundle_dir / "refusal_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "refused"
    assert "mid-ladder" in report["error"]
    assert report["geometry_conditioning"]["pass1"]["seed"] == runtime._DEFAULT_SEED
    assert (bundle_dir / "pass1_mesh.glb").stat().st_size > 0


def test_loop_mode_self_verification_degrades_on_duplication_flag(
    monkeypatch, tmp_path
) -> None:
    from abstract3d import captioning

    backend = runtime.Hunyuan3DShapeBackend(owner=_loop_owner())
    capture: dict = {"seeds": []}
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=0.7)
    _install_fake_runtime(monkeypatch, backend, tmp_path, [sphere, sphere], capture)
    monkeypatch.setattr(captioning, "caption_image", lambda image, **kw: "a toy disc")
    fake_plan, _state = _fake_loop_plan()
    monkeypatch.setattr("abstract3d.loop_conditioning.synthesize_loop_views", fake_plan)
    _stub_loop_self_verification(monkeypatch, suspect=True)

    result = backend.i23d(
        _alpha_disc_image(), device="cpu", texture_mode="none",
        geometry_conditioning="loop",
    )

    metadata = result["metadata"]
    assert metadata["quality_verdict"]["verdict"] == "degraded"
    assert any("duplication autocorrelation" in reason
               for reason in metadata["quality_verdict"]["reasons"])
    assert metadata["loop_self_verification"]["duplication"]["duplication_suspect"] is True


def test_loop_mode_bake_consumes_fullspan_at_measured_azimuths(
    monkeypatch, tmp_path
) -> None:
    """The texture lane in loop mode: full-span raws replay FIRST at their
    MEASURED azimuths through the identity route; windowed images never
    reach the bake's observed views."""
    import skimage.color.colorconv as _colorconv

    def _convert_einsum(matrix, arr):
        arr = _colorconv._prepare_colorarray(arr)
        return np.einsum("...i,ji->...j", arr, matrix.astype(arr.dtype))

    monkeypatch.setattr(_colorconv, "_convert", _convert_einsum)

    backend = runtime.Hunyuan3DShapeBackend(owner=_loop_owner())
    capture: dict = {"seeds": []}
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=0.7)
    _install_fake_runtime(monkeypatch, backend, tmp_path, [sphere, sphere], capture)

    from abstract3d import captioning

    monkeypatch.setattr(captioning, "caption_image", lambda image, **kw: "a toy disc")
    fake_plan, state = _fake_loop_plan()
    monkeypatch.setattr("abstract3d.loop_conditioning.synthesize_loop_views", fake_plan)
    _stub_loop_self_verification(monkeypatch)

    from abstract3d import reference_generation as refgen

    refgen_calls: list = []

    def _fake_generate(mesh, source, *, owner=None, angles=(), image_generator=None, **kwargs):
        first_payloads = {}
        if image_generator is not None:
            for angle in angles:
                first_payloads[str(angle[0])] = image_generator(
                    "prompt", b"conditioning", seed=1)
        refgen_calls.append({
            "angles": tuple(angles),
            "conditioning": kwargs.get("conditioning"),
            "image_request": dict(kwargs.get("image_request") or {}),
            "first_payloads": first_payloads,
        })
        views = [
            {
                "label": str(label),
                "azimuth_deg": float(azimuth),
                "elevation_deg": float(elevation),
                "rgba": _alpha_disc_image(),
                "role": "reference",
                "generated": True,
            }
            for label, azimuth, elevation in angles
        ]
        return views, {
            "angles": [{"label": str(a[0]), "accepted": True} for a in angles],
            "accepted": len(views), "rejected": 0,
        }

    monkeypatch.setattr(refgen, "generate_reference_views", _fake_generate)
    monkeypatch.setattr(
        refgen, "default_i2i_generator",
        lambda owner: (lambda prompt, image, **kwargs: b"fresh"))

    from abstract3d import texturing

    bake_views: list = []
    real_bake = texturing.bake_projection_texture

    def spy_bake(mesh, **kwargs):
        bake_views.append(list(kwargs["observed_views"]))
        return real_bake(mesh, **kwargs)

    monkeypatch.setattr(texturing, "bake_projection_texture", spy_bake)

    result = backend.i23d(
        _alpha_disc_image(),
        device="cpu",
        texture_mode="baked_basecolor",
        texture_resolution=64,
        texture_reference_angle_planning="static",
        seed=11,
        geometry_conditioning="loop",
    )

    # Every refgen call in loop mode runs the identity route with the
    # pinned provider request.
    assert refgen_calls
    for call in refgen_calls:
        assert call["conditioning"] == "identity"
        assert call["image_request"].get("provider") == "mlx-gen"
    # Loop labels bake at their MEASURED azimuths (side_left 82.5), each
    # exactly once; the static extra (top) keeps its slot.
    angle_by_label: dict = {}
    for call in refgen_calls:
        for label, azimuth, elevation in call["angles"]:
            assert label not in angle_by_label  # exactly once across calls
            angle_by_label[label] = (azimuth, elevation)
    assert angle_by_label["side_left"] == (82.5, 0.0)
    assert angle_by_label["back"] == (180.0, 0.0)
    assert angle_by_label["side_right"] == (-90.0, 0.0)
    assert angle_by_label["top"] == (0.0, 55.0)
    # The loop raws replay as each loop angle's FIRST ladder attempt.
    served = {}
    for call in refgen_calls:
        served.update(call["first_payloads"])
    assert served["back"] == b"raw-back"
    assert served["side_left"] == b"raw-side_left"
    assert served["side_right"] == b"raw-side_right"
    # "top" has no replay: it rides the pending batch with the DEFAULT
    # generator (no injected replay generator), i.e. a fresh identity draw.
    assert "top" not in served
    # THE SPLIT-CONSUMER INVARIANT, bake half: no windowed object is ever
    # an observed view.
    windowed_ids = {id(image) for image in state["windowed"].values()}
    windowed_ids.add(id(state["front_windowed"]))
    for views in bake_views:
        for view in views:
            assert id(view.get("rgba")) not in windowed_ids
    assert result["metadata"]["texture_artifacts"]["reference_generation"]["accepted"] >= 3
