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


def test_synthesize_geometry_views_gates_and_side_pair(monkeypatch) -> None:
    """End-to-end gate battery on real images: a mirror-consistent back
    view passes, and a side pair whose silhouettes disagree is dropped
    WHOLE (blame between the two sides is unattributable)."""
    from PIL import Image

    from abstract3d import segmentation

    monkeypatch.setattr(
        segmentation, "remove_background_robust", lambda image: image.convert("RGBA")
    )
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
    assert schema["geometry_conditioning"]["enum"] == ["single", "multiview", "auto"]


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
