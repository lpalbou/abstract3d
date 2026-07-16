from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from abstract3d import texturing


def test_harmonize_view_exposure_matches_mean_and_std() -> None:
    rng = np.random.default_rng(3)
    primary = np.zeros((64, 64, 4), dtype=np.float32)
    primary[:, :, 0] = 0.6 + 0.05 * rng.standard_normal((64, 64))
    primary[:, :, 1] = 0.4 + 0.05 * rng.standard_normal((64, 64))
    primary[:, :, 2] = 0.3 + 0.05 * rng.standard_normal((64, 64))
    primary[:, :, 3] = 1.0

    reference = primary.copy()
    reference[:, :, :3] = np.clip(reference[:, :, :3] * 0.5 + 0.1, 0.0, 1.0)

    corrected = texturing.harmonize_view_exposure(reference, primary_rgba=primary)

    for channel in range(3):
        assert abs(float(corrected[:, :, channel].mean()) - float(primary[:, :, channel].mean())) < 0.03


def test_harmonize_view_exposure_keeps_low_coverage_reference_unchanged() -> None:
    primary = np.ones((32, 32, 4), dtype=np.float32) * 0.5
    reference = np.zeros((32, 32, 4), dtype=np.float32)
    reference[:2, :2, 3] = 1.0  # under 64 foreground pixels

    out = texturing.harmonize_view_exposure(reference, primary_rgba=primary)

    assert np.allclose(out, reference)


def test_feather_projection_weight_ramps_boundary() -> None:
    weight = np.zeros((32, 32), dtype=np.float32)
    weight[8:24, 8:24] = 1.0

    feathered = texturing.feather_projection_weight(weight, feather_texels=4.0)

    # interior stays full strength, boundary is attenuated
    assert feathered[16, 16] == 1.0
    assert 0.0 < feathered[8, 16] < 1.0
    assert feathered[0, 0] == 0.0


def test_blend_projections_prefers_best_view_over_average() -> None:
    shape = (16, 16)
    red = np.zeros((*shape, 4), dtype=np.float32)
    red[:, :, 0] = 1.0
    red[:, :, 3] = 1.0
    blue = np.zeros((*shape, 4), dtype=np.float32)
    blue[:, :, 2] = 1.0
    blue[:, :, 3] = 1.0
    strong = {"rgba": red, "weight": np.full(shape, 0.9, dtype=np.float32), "coverage_ratio": 1.0, "label": "a"}
    weak = {"rgba": blue, "weight": np.full(shape, 0.3, dtype=np.float32), "coverage_ratio": 1.0, "label": "b"}

    blended = texturing.blend_projections(
        [strong, weak], atlas_shape=shape, sharpness=5.0, feather_texels=0.0
    )

    # with sharp best-view bias, red should strongly dominate
    assert float(blended["rgb"][8, 8, 0]) > 0.85
    assert float(blended["rgb"][8, 8, 2]) < 0.15
    assert len(blended["view_stats"]) == 2


def test_blend_projections_zero_sharpness_is_weighted_average() -> None:
    shape = (8, 8)
    red = np.zeros((*shape, 4), dtype=np.float32)
    red[:, :, 0] = 1.0
    red[:, :, 3] = 1.0
    blue = np.zeros((*shape, 4), dtype=np.float32)
    blue[:, :, 2] = 1.0
    blue[:, :, 3] = 1.0
    a = {"rgba": red, "weight": np.full(shape, 0.5, dtype=np.float32), "coverage_ratio": 1.0}
    b = {"rgba": blue, "weight": np.full(shape, 0.5, dtype=np.float32), "coverage_ratio": 1.0}

    blended = texturing.blend_projections([a, b], atlas_shape=shape, sharpness=0.0, feather_texels=0.0)

    assert abs(float(blended["rgb"][4, 4, 0]) - 0.5) < 1e-5
    assert abs(float(blended["rgb"][4, 4, 2]) - 0.5) < 1e-5


def test_inpaint_unseen_texels_fills_hidden_surface() -> None:
    shape = (32, 32)
    colors = np.zeros((*shape, 4), dtype=np.float32)
    surface = np.zeros(shape, dtype=bool)
    surface[4:28, 4:28] = True
    observed = np.zeros(shape, dtype=bool)
    observed[4:28, 4:16] = True
    colors[observed] = [0.8, 0.2, 0.1, 1.0]

    filled = texturing.inpaint_unseen_texels(colors, surface_mask=surface, observed_mask=observed)

    unseen = surface & ~observed
    assert (filled[unseen, 3] == 1.0).all()
    # filled colors should look like the observed reds, not black
    assert float(filled[unseen, 0].mean()) > 0.5


def test_inpaint_unseen_texels_neutral_when_nothing_observed() -> None:
    shape = (16, 16)
    colors = np.zeros((*shape, 4), dtype=np.float32)
    surface = np.ones(shape, dtype=bool)
    observed = np.zeros(shape, dtype=bool)

    filled = texturing.inpaint_unseen_texels(colors, surface_mask=surface, observed_mask=observed)

    assert np.allclose(filled[:, :, :3], 0.5)
    assert (filled[:, :, 3] == 1.0).all()


def _flat_strip_state(shape=(24, 48), observed_cols=8):
    """Planar surface strip in the atlas: positions vary linearly with UV,
    normals constant, a left band observed. Shared by the fill-stage tests."""
    height, width = shape
    positions = np.zeros((height, width, 4), dtype=np.float32)
    xs = np.linspace(0.0, 1.0, width, dtype=np.float32)
    ys = np.linspace(0.0, 0.5, height, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    positions[:, :, 0] = grid_x
    positions[:, :, 1] = grid_y
    positions[:, :, 3] = 1.0
    normals = np.zeros((height, width, 4), dtype=np.float32)
    normals[:, :, 2] = 1.0
    normals[:, :, 3] = 1.0
    observed = np.zeros((height, width), dtype=bool)
    observed[:, :observed_cols] = True
    return positions, normals, observed


def test_texel_surface_smooth_removes_plateau_steps_and_keeps_anchors() -> None:
    positions, normals, observed = _flat_strip_state()
    height, width = observed.shape
    colors = np.zeros((height, width, 4), dtype=np.float32)
    colors[observed] = [0.8, 0.4, 0.2, 1.0]
    # Facet-like fill: two flat-color blocks with a hard step between them
    # (the defect the pass exists to remove).
    fill = ~observed
    colors[:, 8:28, :3] = 0.3
    colors[:, 28:, :3] = 0.6
    colors[fill, 3] = 1.0

    smoothed = texturing.texel_surface_smooth(
        colors,
        positions_texture=positions,
        normals_texture=normals,
        observed_mask=observed,
    )

    # Observed anchors are untouched.
    assert np.allclose(smoothed[observed], colors[observed])
    # The hard step across columns 27|28 is relaxed: the maximum horizontal
    # neighbor delta inside the fill drops well below the original 0.3.
    fill_interior = fill.copy()
    fill_interior[:, :9] = False
    deltas = np.abs(np.diff(smoothed[:, :, 0], axis=1))[:, 9:-1]
    assert float(deltas.max()) < 0.1


def test_synthesize_fill_detail_matches_observed_amplitude() -> None:
    rng = np.random.default_rng(5)
    positions, normals, observed = _flat_strip_state(shape=(32, 64), observed_cols=24)
    height, width = observed.shape
    colors = np.full((height, width, 4), 0.5, dtype=np.float32)
    # Observed band carries real micro-texture; fill is a flat wash.
    colors[observed, :3] = 0.5 + 0.12 * rng.standard_normal((int(observed.sum()), 1)).astype(np.float32)
    colors[:, :, :3] = np.clip(colors[:, :, :3], 0.0, 1.0)
    colors[:, :, 3] = 1.0
    fill = ~observed

    out = texturing.synthesize_fill_detail(
        colors,
        positions_texture=positions,
        normals_texture=normals,
        observed_mask=observed,
        gain=1.0,
    )

    # Observed texels byte-identical; fill gains variance it had none of.
    assert np.allclose(out[observed], colors[observed])
    deep_fill = fill.copy()
    deep_fill[:, :32] = False  # skip the seam feather band
    assert float(out[deep_fill][:, 0].std()) > 0.02
    assert float(colors[deep_fill][:, 0].std()) < 1e-6
    # Zero-mean application: the fill's average color survives.
    assert abs(float(out[deep_fill][:, 0].mean()) - 0.5) < 0.05
    # Deterministic for fixed inputs.
    out2 = texturing.synthesize_fill_detail(
        colors,
        positions_texture=positions,
        normals_texture=normals,
        observed_mask=observed,
        gain=1.0,
    )
    assert np.array_equal(out, out2)


def test_synthesize_fill_detail_zero_gain_is_identity() -> None:
    positions, normals, observed = _flat_strip_state()
    colors = np.full((*observed.shape, 4), 0.4, dtype=np.float32)

    out = texturing.synthesize_fill_detail(
        colors,
        positions_texture=positions,
        normals_texture=normals,
        observed_mask=observed,
        gain=0.0,
    )

    assert np.array_equal(out, colors)


def _scharr_energy(rgb: np.ndarray, region: np.ndarray) -> float:
    """Mean Scharr gradient magnitude of luminance over the eroded region
    (the statistic behind texture_qa's fill_gradient_energy_ratio gate)."""
    from scipy.ndimage import binary_erosion, convolve1d

    lum = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    gx = convolve1d(convolve1d(lum, [1.0, 0.0, -1.0], axis=1), [3.0, 10.0, 3.0], axis=0)
    gy = convolve1d(convolve1d(lum, [1.0, 0.0, -1.0], axis=0), [3.0, 10.0, 3.0], axis=1)
    inner = binary_erosion(region, iterations=2)
    return float(np.hypot(gx, gy)[inner].mean())


def test_synthesize_fill_detail_energy_calibration_reaches_gate() -> None:
    """Stochastic-textured observed region next to a DARKER flat fill: the
    open-loop sigma transfer undershoots the QA gate's linear gradient
    statistic on darker bases (multiplicative log-detail yields
    proportionally less linear gradient — the measured 0.79x luminance
    factor on the starship), so the closed loop must amplify (scale > 1)
    until the fill's gradient energy reaches the gate line
    (>= 0.5x observed at gain 0.7)."""
    rng = np.random.default_rng(11)
    positions, normals, observed = _flat_strip_state(shape=(128, 256), observed_cols=96)
    colors = np.full((*observed.shape, 4), 0.45, dtype=np.float32)
    noise = 0.10 * rng.standard_normal((*observed.shape, 1)).astype(np.float32)
    colors[:, :, :3] = np.clip(0.45 + np.where(observed[:, :, None], noise, 0.0), 0.0, 1.0)
    # Fill base darker than observed at the ratio measured on the starship
    # proof (0.79x): the open loop undershoots there, the closed loop must
    # recover it, and the sigma guard still has headroom (a far darker
    # base would honestly cross the granite line and stay capped — that
    # regime is covered by the granite test below).
    colors[~observed, :3] = 0.35
    colors[:, :, 3] = 1.0

    stats: dict = {}
    out = texturing.synthesize_fill_detail(
        colors,
        positions_texture=positions,
        normals_texture=normals,
        observed_mask=observed,
        gain=0.7,
        stats_out=stats,
    )

    fill = ~observed
    fill_deep = fill.copy()
    fill_deep[:, :104] = False  # skip the seam feather band
    e_obs = _scharr_energy(colors[:, :, :3], observed)
    e_fill = _scharr_energy(out[:, :, :3], fill_deep)
    calibration = stats.get("energy_calibration") or {}
    assert calibration.get("scale", 0.0) > 1.0, calibration
    assert e_fill >= 0.5 * e_obs, (e_fill, e_obs, calibration)
    # and not noise-injected past the observed level
    assert e_fill <= 1.1 * e_obs, (e_fill, e_obs, calibration)


def test_synthesize_fill_detail_calibration_never_injects_granite() -> None:
    """Edge-dominated observed content (flat plates + sparse strong panel
    lines) has high TOTAL gradient energy but a near-silent stochastic
    band. The calibration may amplify only within its caps and must NOT
    chase the edge energy with noise amplitude: the fill's realized
    log-sigma stays far below the line-dominated residual sigma, the
    scale respects `energy_calibration_max`, and the honest shortfall is
    visible in the reported energies."""
    positions, normals, observed = _flat_strip_state(shape=(128, 256), observed_cols=96)
    colors = np.full((*observed.shape, 4), 0.5, dtype=np.float32)
    # strong dark panel lines every 16 columns, otherwise perfectly flat
    for col in range(0, 96, 16):
        colors[:, col:col + 2, :3] = 0.1
    colors[:, :, 3] = 1.0

    stats: dict = {}
    out = texturing.synthesize_fill_detail(
        colors,
        positions_texture=positions,
        normals_texture=normals,
        observed_mask=observed,
        gain=0.7,
        stats_out=stats,
    )

    calibration = stats.get("energy_calibration") or {}
    assert 1.0 <= calibration.get("scale", 0.0) <= 3.0, calibration
    # sigma accounting is reported and the applied sigma respects the guard
    sigma_obs = calibration.get("sigma_observed_band")
    sigma_applied = calibration.get("sigma_at_scale")
    assert sigma_obs is not None and sigma_applied is not None
    assert sigma_applied <= sigma_obs * 1.05 + 1e-6, calibration
    # no granite: the flat-plate fill never reaches edge-level contrast —
    # its realized log-sigma stays below half the line-residual sigma
    fill_deep = ~observed
    fill_deep[:, :104] = False
    log_fill = np.log(np.clip(out[fill_deep][:, 0], 0.0, 1.0) + 0.02)
    assert float(log_fill.std()) <= 0.5 * sigma_obs, (float(log_fill.std()), calibration)


def test_mesh_graph_harmonic_fill_interpolates_between_vertices() -> None:
    import trimesh

    # Sphere with the +X hemisphere observed in red-to-dark gradient; the
    # fill on the -X side must vary smoothly at TEXEL resolution rather
    # than in per-vertex flat blocks.
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=0.5)
    size = 96
    positions = np.zeros((size, size, 4), dtype=np.float32)
    # Wrap the sphere surface into the atlas via spherical coordinates.
    us = np.linspace(-np.pi, np.pi, size, endpoint=False, dtype=np.float32)
    vs = np.linspace(-0.4 * np.pi, 0.4 * np.pi, size, dtype=np.float32)
    grid_u, grid_v = np.meshgrid(us, vs)
    positions[:, :, 0] = 0.5 * np.cos(grid_v) * np.cos(grid_u)
    positions[:, :, 1] = 0.5 * np.cos(grid_v) * np.sin(grid_u)
    positions[:, :, 2] = 0.5 * np.sin(grid_v)
    positions[:, :, 3] = 1.0
    observed = positions[:, :, 0] > 0.05
    colors = np.zeros((size, size, 4), dtype=np.float32)
    colors[observed, 0] = np.clip(positions[observed, 1] + 0.5, 0.0, 1.0)
    colors[observed, 3] = 1.0

    filled = texturing.mesh_graph_harmonic_fill(
        mesh,
        positions_texture=positions,
        observed_mask=observed,
        colors_rgba=colors,
    )

    assert filled is not None
    unseen = ~observed
    assert (filled[unseen, 3] == 1.0).all()
    # Texel-resolution smoothness: with ~40 texels between vertices on this
    # atlas, nearest-vertex assignment repeats each vertex color across its
    # whole cell (few unique values, large plateaus); IDW interpolation
    # yields far more distinct values than the mesh has vertices nearby.
    fill_values = filled[unseen, 0]
    assert len(np.unique(np.round(fill_values, 4))) > 200


def test_bake_projection_texture_auto_completion_on_symmetric_mesh() -> None:
    import trimesh

    mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.5)
    observed = Image.new("RGBA", (96, 96), (180, 90, 40, 255))

    # An OFF-CENTER view: the observed hemisphere is asymmetric across the
    # left-right plane, so confident observed texels legitimately have
    # hidden mirror twins to fill. (A dead-frontal view of a sphere is
    # already y-symmetric — the old version of this test only saw fill
    # because the source gate's junk-texel fallback cliff let grazing rim
    # samples act as twins, which is exactly the defect the gate exists to
    # prevent.)
    _, stats = texturing.bake_projection_texture(
        mesh,
        observed_views=[{"rgba": observed, "azimuth_deg": 40.0, "elevation_deg": 0.0, "label": "front"}],
        texture_resolution=96,
        texture_completion="auto",
    )

    # A sphere is perfectly mirror-symmetric: auto must resolve to mirror
    # completion and record both the request and the resolution.
    assert stats["texture_completion_requested"] == "auto"
    assert stats["texture_completion"] == "mirror_symmetry"
    assert stats["symmetry_completion"]["mode"] == "mirror_symmetry"


def test_projector_survives_out_of_frustum_samples() -> None:
    # With a very close camera, part of the surface projects outside the
    # image plane; the gather must clip instead of wrapping or crashing.
    from abstract3d.backends import triposr_runtime as rt

    positions = np.zeros((16, 16, 4), dtype=np.float32)
    xs = np.linspace(-2.0, 2.0, 16, dtype=np.float32)
    ys = np.linspace(-2.0, 2.0, 16, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    positions[:, :, 0] = 0.0
    positions[:, :, 1] = grid_x
    positions[:, :, 2] = grid_y
    positions[:, :, 3] = 1.0
    normals = np.zeros_like(positions)
    normals[:, :, 0] = 1.0
    normals[:, :, 3] = 1.0
    observed = Image.new("RGBA", (32, 32), (10, 200, 30, 255))

    projection = rt._tripo_project_observed_texture(
        observed,
        mesh=None,
        positions_texture=positions,
        normals_texture=normals,
        camera_distance=1.1,
    )

    assert np.isfinite(projection["weight"]).all()
    assert projection["rgba"].shape == positions.shape


def test_blend_projections_reports_raw_coverage_over_feathered_weight() -> None:
    shape = (24, 24)
    rgba = np.zeros((*shape, 4), dtype=np.float32)
    rgba[:, :, 0] = 1.0
    rgba[:, :, 3] = 1.0
    weight = np.zeros(shape, dtype=np.float32)
    weight[4:20, 4:20] = 0.8

    blended = texturing.blend_projections(
        [{"rgba": rgba, "weight": weight, "coverage_ratio": 1.0}],
        atlas_shape=shape,
        sharpness=3.0,
        feather_texels=6.0,
    )

    # Feathering shrinks the blend weights but coverage must keep the full
    # observed footprint so fill stages never overwrite observed texels.
    assert bool(blended["coverage"][4, 4])
    assert float(blended["weight"][4, 4]) < 0.8
    # Rim texels still receive a color even where feathered weight is ~0.
    assert float(blended["rgb"][4, 4, 0]) > 0.5


def test_harmonize_and_gate_projection_fixes_exposure_and_gates_disagreement() -> None:
    shape = (32, 32)
    source_rgba = np.zeros((*shape, 4), dtype=np.float32)
    source_rgba[:, :, 0] = 0.6
    source_rgba[:, :, 1] = 0.4
    source_rgba[:, :, 2] = 0.3
    source_rgba[:, :, 3] = 1.0
    source = {"rgba": source_rgba, "weight": np.full(shape, 0.8, dtype=np.float32)}

    # Underexposed but consistent reference: harmonization recovers it and
    # the gate leaves its weight alone.
    dim = {
        "rgba": source_rgba.copy() * np.array([0.5, 0.5, 0.5, 1.0], dtype=np.float32),
        "weight": np.full(shape, 0.6, dtype=np.float32),
    }
    stats = texturing.harmonize_and_gate_projection(dim, source_projection=source)
    assert stats["harmonized"] is True
    assert stats["weight_scale"] == 1.0
    assert abs(float(dim["rgba"][16, 16, 0]) - 0.6) < 0.02

    # Structurally different reference (wrong pose / wrong subject): gains
    # cannot reconcile it, so the gate attenuates or rejects its weight.
    rng = np.random.default_rng(5)
    noisy = {
        "rgba": np.concatenate(
            [rng.random((*shape, 3), dtype=np.float32), np.ones((*shape, 1), dtype=np.float32)],
            axis=2,
        ),
        "weight": np.full(shape, 0.6, dtype=np.float32),
    }
    stats2 = texturing.harmonize_and_gate_projection(noisy, source_projection=source)
    assert stats2["disagreement"] > 0.16
    assert stats2["weight_scale"] < 1.0
    assert float(noisy["weight"].max()) < 0.6


def test_harmonize_and_gate_projection_skips_low_overlap() -> None:
    shape = (16, 16)
    source = {
        "rgba": np.ones((*shape, 4), dtype=np.float32) * 0.5,
        "weight": np.zeros(shape, dtype=np.float32),
    }
    reference = {
        "rgba": np.ones((*shape, 4), dtype=np.float32),
        "weight": np.full(shape, 0.5, dtype=np.float32),
    }
    stats = texturing.harmonize_and_gate_projection(reference, source_projection=source)
    assert stats["overlap_texels"] == 0
    assert stats["harmonized"] is False
    assert float(reference["weight"].max()) == 0.5


def test_resolve_projection_conflicts_keeps_best_witness_per_texel() -> None:
    shape = (24, 24)
    agree = np.full((*shape, 4), 0.5, dtype=np.float32)
    strong = {"rgba": agree.copy(), "weight": np.full(shape, 0.9, dtype=np.float32), "label": "strong"}
    weak = {"rgba": agree.copy(), "weight": np.full(shape, 0.4, dtype=np.float32), "label": "weak"}
    # Disputed band: the weak view paints a very different color there.
    weak["rgba"][8:12, :, :3] = 0.95

    stats = texturing.resolve_projection_conflicts([strong, weak])
    assert stats["conflict_texels"] == 4 * shape[1]
    # The weaker witness loses only the disputed band.
    assert float(weak["weight"][10, 10]) == 0.0
    assert float(weak["weight"][0, 0]) == pytest.approx(0.4)
    # The stronger witness keeps everything.
    assert float(strong["weight"][10, 10]) == pytest.approx(0.9)


def test_resolve_projection_conflicts_ignores_agreeing_views() -> None:
    shape = (16, 16)
    base = np.full((*shape, 4), 0.5, dtype=np.float32)
    a = {"rgba": base.copy(), "weight": np.full(shape, 0.8, dtype=np.float32), "label": "a"}
    b = {"rgba": base.copy() + 0.03, "weight": np.full(shape, 0.6, dtype=np.float32), "label": "b"}
    stats = texturing.resolve_projection_conflicts([a, b])
    assert stats["conflict_texels"] == 0
    assert float(b["weight"].min()) == pytest.approx(0.6)


def test_estimate_view_pose_centers_search_on_declared_angle() -> None:
    import trimesh

    # An asymmetric wedge: pose is recoverable from the silhouette.
    mesh = trimesh.creation.box(extents=(1.2, 0.5, 0.3))
    mesh.apply_translation([0.2, 0.0, 0.0])
    rgba = np.zeros((96, 96, 4), dtype=np.float32)

    # Degenerate mask: must fall back to the CENTER pose, not azimuth 0.
    pose = texturing.estimate_view_pose(
        mesh,
        observed_rgba=rgba,
        center_azimuth_deg=90.0,
        center_elevation_deg=5.0,
        azimuth_window_deg=20.0,
    )
    assert pose["azimuth_deg"] == 90.0
    assert pose["elevation_deg"] == 5.0


def test_recenter_to_canonical_frame_matches_hunyuan_convention() -> None:
    # A subject occupying a corner must end centered with the larger side
    # at (1 - border_ratio) of the canvas.
    image = np.zeros((200, 200, 4), dtype=np.uint8)
    image[120:180, 20:50, :3] = 200
    image[120:180, 20:50, 3] = 255
    out = texturing.recenter_to_canonical_frame(
        Image.fromarray(image, mode="RGBA"), size=100, border_ratio=0.2
    )
    arr = np.asarray(out)
    assert arr.shape == (100, 100, 4)
    ys, xs = np.where(arr[:, :, 3] > 12)
    height = ys.max() - ys.min() + 1
    center_y = (ys.max() + ys.min()) / 2.0
    center_x = (xs.max() + xs.min()) / 2.0
    assert abs(height - 80) <= 2          # larger side fills 1-0.2 of 100
    assert abs(center_y - 49.5) <= 1.5    # centered
    assert abs(center_x - 49.5) <= 1.5


def test_canonical_ortho_half_extent_reproduces_framing() -> None:
    import trimesh

    box = trimesh.creation.box(extents=(0.4, 1.0, 1.6))
    half = texturing.canonical_ortho_half_extent(
        box, azimuth_deg=0.0, elevation_deg=0.0, border_ratio=0.15
    )
    # From the front the camera-plane extents are y=1.0 (width) and z=1.6
    # (height); the larger one over 2*(1-0.15).
    assert abs(half - 1.6 / (2 * 0.85)) < 1e-3


def test_orthographic_projection_round_trip_on_sphere() -> None:
    import math
    import trimesh

    from abstract3d.backends.triposr_runtime import (
        _tripo_camera_position,
        _tripo_look_at_matrix,
        _tripo_make_texture_atlas,
        _tripo_project_observed_texture,
        _tripo_rasterize_normal_atlas,
        _tripo_rasterize_position_atlas,
    )

    radius = 0.9
    mesh = trimesh.creation.icosphere(subdivisions=4, radius=radius)

    def field(points):
        return np.clip(points * 0.5 / radius + 0.5, 0.0, 1.0)

    # Analytic orthographic photo via ray-sphere intersection with the
    # projector's own camera basis.
    size = 384
    half_extent = 1.1
    eye = _tripo_camera_position(azimuth_deg=0.0, elevation_deg=0.0, camera_distance=3.0)
    view = _tripo_look_at_matrix(eye, np.zeros(3, dtype=np.float32), np.array([0, 0, 1], dtype=np.float32))
    rot = view[:3, :3].astype(np.float64)
    cam_right, cam_up, cam_back = rot[0], rot[1], rot[2]
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float64)
    scale = 0.5 * size / half_extent
    ox = (xs - size / 2 + 0.5) / scale
    oy = -(ys - size / 2 + 0.5) / scale
    origins = eye.astype(np.float64) + ox[..., None] * cam_right + oy[..., None] * cam_up
    direction = -cam_back
    b = origins @ direction
    c = (origins * origins).sum(axis=2) - radius * radius
    disc = b * b - c
    hit = disc > 0
    t = -b - np.sqrt(np.where(hit, disc, 0.0))
    points = origins + direction[None, None, :] * t[..., None]
    photo = np.zeros((size, size, 4), dtype=np.float32)
    photo[hit, :3] = field(points[hit]).astype(np.float32)
    photo[hit, 3] = 1.0
    photo_image = Image.fromarray((photo * 255).astype(np.uint8), mode="RGBA")

    atlas = _tripo_make_texture_atlas(mesh, texture_resolution=256, texture_padding=2)
    kwargs = dict(
        atlas_vmapping=atlas["vmapping"],
        atlas_indices=atlas["indices"],
        atlas_uvs=atlas["uvs"],
        texture_resolution=256,
        texture_padding=2,
    )
    positions = _tripo_rasterize_position_atlas(mesh, **kwargs)
    normals = _tripo_rasterize_normal_atlas(mesh, **kwargs)
    projection = _tripo_project_observed_texture(
        photo_image,
        mesh=mesh,
        positions_texture=positions,
        normals_texture=normals,
        azimuth_deg=0.0,
        elevation_deg=0.0,
        camera_distance=3.0,
        projection_model="orthographic",
        ortho_half_extent=half_extent,
    )
    covered = projection["weight"] > 0.05
    assert covered.mean() > 0.05
    true_colors = field(np.asarray(positions)[:, :, :3])[covered]
    sampled = np.asarray(projection["rgba"])[:, :, :3][covered]
    assert float(np.abs(sampled - true_colors).mean()) < 0.02


def test_projector_strict_zbuffer_rejects_hidden_sheet() -> None:
    import trimesh

    from abstract3d.backends.triposr_runtime import (
        _tripo_make_texture_atlas,
        _tripo_project_observed_texture,
        _tripo_rasterize_normal_atlas,
        _tripo_rasterize_position_atlas,
    )

    # Two parallel unit quads facing +X, one at x=0.5 (front) and one at
    # x=0.3 (hidden 0.2 behind). Both face the camera; only the front one
    # may be painted.
    def quad(x):
        vertices = np.array(
            [[x, -0.5, -0.5], [x, 0.5, -0.5], [x, 0.5, 0.5], [x, -0.5, 0.5]], dtype=np.float64
        )
        faces = np.array([[0, 1, 2], [0, 2, 3]])
        return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    mesh = trimesh.util.concatenate([quad(0.5), quad(0.3)])
    atlas = _tripo_make_texture_atlas(mesh, texture_resolution=128, texture_padding=2)
    kwargs = dict(
        atlas_vmapping=atlas["vmapping"],
        atlas_indices=atlas["indices"],
        atlas_uvs=atlas["uvs"],
        texture_resolution=128,
        texture_padding=2,
    )
    positions = _tripo_rasterize_position_atlas(mesh, **kwargs)
    normals = _tripo_rasterize_normal_atlas(mesh, **kwargs)
    photo = np.zeros((128, 128, 4), dtype=np.float32)
    photo[:, :, 0] = 1.0
    photo[:, :, 3] = 1.0
    projection = _tripo_project_observed_texture(
        Image.fromarray((photo * 255).astype(np.uint8), mode="RGBA"),
        mesh=mesh,
        positions_texture=positions,
        normals_texture=normals,
        azimuth_deg=0.0,
        elevation_deg=0.0,
        camera_distance=3.0,
        projection_model="orthographic",
        ortho_half_extent=1.0,
    )
    pos = np.asarray(positions)
    weight = np.asarray(projection["weight"])
    front_texels = (np.abs(pos[:, :, 0] - 0.5) < 1e-3) & (pos[:, :, 3] > 0)
    hidden_texels = (np.abs(pos[:, :, 0] - 0.3) < 1e-3) & (pos[:, :, 3] > 0)
    assert weight[front_texels].max() > 0.5
    assert weight[hidden_texels].max() == 0.0


def test_projector_scarce_weight_is_disjoint_bounded_rescue_band() -> None:
    """The projector's witness-scarcity candidates (G1): claims strictly
    between the grazing floor (0.05) and the role facing threshold, never
    overlapping strict claims, bounded by the exact per-texel stretch."""
    import trimesh

    from abstract3d.backends.triposr_runtime import (
        _tripo_make_texture_atlas,
        _tripo_project_observed_texture,
        _tripo_rasterize_normal_atlas,
        _tripo_rasterize_position_atlas,
    )

    mesh = trimesh.creation.icosphere(subdivisions=4, radius=0.9)
    atlas = _tripo_make_texture_atlas(mesh, texture_resolution=256, texture_padding=2)
    kwargs = dict(
        atlas_vmapping=atlas["vmapping"],
        atlas_indices=atlas["indices"],
        atlas_uvs=atlas["uvs"],
        texture_resolution=256,
        texture_padding=2,
    )
    positions = _tripo_rasterize_position_atlas(mesh, **kwargs)
    normals = _tripo_rasterize_normal_atlas(mesh, **kwargs)
    photo = np.full((384, 384, 4), 255, dtype=np.uint8)
    projection = _tripo_project_observed_texture(
        Image.fromarray(photo, mode="RGBA"),
        mesh=mesh,
        positions_texture=positions,
        normals_texture=normals,
        azimuth_deg=0.0,
        elevation_deg=0.0,
        camera_distance=3.0,
        projection_model="orthographic",
        ortho_half_extent=1.1,
        facing_threshold=0.4,
    )
    weight = np.asarray(projection["weight"])
    scarce = np.asarray(projection["scarce_weight"])
    facing = np.asarray(projection["facing"], dtype=np.float32)

    assert scarce.max() > 0.0, "sphere rim must produce scarce candidates"
    # disjoint: no texel carries both a strict and a scarce claim
    assert not ((weight > 0) & (scarce > 0)).any()
    # the band: scarce claims live strictly between floor and threshold
    band = scarce > 0
    assert float(facing[band].min()) > 0.05
    assert float(facing[band].max()) <= 0.4 + 1e-3
    # on a smooth sphere the extreme rim (facing ~ floor) is heavily
    # stretched; the stretch bound must refuse some of the geometric band
    geometric_band = (np.asarray(positions)[:, :, 3] > 0) & \
        (facing > 0.05) & ~(facing > 0.4) & (weight == 0.0)
    assert int(band.sum()) < int(geometric_band.sum()), \
        "stretch bound must refuse part of the grazing band"


def test_admit_scarce_witnesses_only_writes_orphans() -> None:
    """Scarce claims are admitted ONLY on texels no view strictly paints;
    strictly-claimed texels keep their calibrated gates untouched."""
    from abstract3d.texturing import admit_scarce_witnesses

    shape = (32, 32)
    surface = np.ones(shape, dtype=bool)

    def projection(weight, scarce):
        rgba = np.zeros((*shape, 4), dtype=np.float32)
        rgba[:, :, :3] = 0.5
        rgba[:, :, 3] = np.where(weight > 0, 1.0, 0.0)
        return {"weight": weight.astype(np.float32),
                "scarce_weight": scarce.astype(np.float32),
                "rgba": rgba, "label": "v"}

    strict_a = np.zeros(shape, np.float32)
    strict_a[:, :16] = 0.8                      # A strictly claims the left half
    scarce_a = np.zeros(shape, np.float32)
    scarce_a[:, 16:20] = 0.02                   # A's rescue band
    scarce_b = np.zeros(shape, np.float32)
    scarce_b[:, 8:24] = 0.03                    # B's rescue overlaps A's strict turf

    a = projection(strict_a, scarce_a)
    b = projection(np.zeros(shape, np.float32), scarce_b)
    stats = admit_scarce_witnesses([a, b], surface_mask=surface)

    assert stats["applied"]
    # strict turf untouched: B's scarce claim on columns 8..16 refused
    assert (np.asarray(b["weight"])[:, 8:16] == 0.0).all()
    # orphan band admitted for both claimants (blend arbitrates downstream)
    assert (np.asarray(a["weight"])[:, 16:20] > 0.0).all()
    assert (np.asarray(b["weight"])[:, 16:24] > 0.0).all()
    # admitted texels become valid content (rgba alpha raised)
    assert (np.asarray(b["rgba"])[:, 16:24, 3] == 1.0).all()
    # A's strict weights bit-identical on its own turf
    assert (np.asarray(a["weight"])[:, :16] == 0.8).all()
    union = np.zeros(shape, dtype=bool)
    union[:, 16:24] = True
    assert stats["admitted_texels"] == int(union.sum())

    # single projection without scarce map: graceful no-op
    bare = {"weight": strict_a.copy(), "rgba": a["rgba"].copy()}
    stats2 = admit_scarce_witnesses([bare], surface_mask=surface)
    assert not stats2["applied"]


def test_admit_scarce_witnesses_consensus_guard_refuses_boundary_debris() -> None:
    """The consensus guard (measured on the face proof: unguarded
    admission lifted dark_debris 0.0022 -> 0.0038): a DARK scarce claim
    on bright-dominated confident context is boundary-displaced mixture
    content and is refused; bright claims (the owner's left-jaw example
    class) and dark claims inside dark regions are admitted."""
    from abstract3d.texturing import admit_scarce_witnesses

    size = 64
    surface = np.ones((size, size), dtype=bool)
    positions = np.zeros((size, size, 4), dtype=np.float32)
    xs, ys = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    positions[:, :, 0] = xs
    positions[:, :, 2] = ys
    positions[:, :, 3] = 1.0

    # strict confident witness everywhere except a 8-column orphan band;
    # left half bright skin, right half dark hair
    weight = np.full((size, size), 0.8, np.float32)
    weight[:, 28:36] = 0.0
    rgba = np.zeros((size, size, 4), dtype=np.float32)
    rgba[:, :32, :3] = 0.75          # bright material
    rgba[:, 32:, :3] = 0.12          # dark material
    rgba[:, :, 3] = np.where(weight > 0, 1.0, 0.0)

    scarce = np.zeros((size, size), np.float32)
    scarce[:, 28:36] = 0.02
    # candidate content: dark claims on the bright side of the band,
    # bright claims next to them, dark claims on the dark side
    rgba[:, 28:30, :3] = 0.10        # dark on bright context -> refuse
    rgba[:, 30:32, :3] = 0.7         # bright on bright context -> keep
    rgba[:, 32:36, :3] = 0.10        # dark on dark context -> keep

    projection = {"weight": weight, "scarce_weight": scarce,
                  "rgba": rgba, "label": "v"}
    # radius/neighbor floor scaled to the 64x64 test grid (defaults are
    # calibrated for real 1024+ atlases where the ball holds thousands)
    stats = admit_scarce_witnesses(
        [projection], surface_mask=surface, positions_texture=positions,
        consensus_radius_ratio=0.06, consensus_min_neighbors=8)
    out_weight = np.asarray(projection["weight"])
    assert (out_weight[:, 28:30] == 0.0).all(), "dark-on-bright must refuse"
    assert (out_weight[:, 30:32] > 0.0).all(), "bright-on-bright must admit"
    assert (out_weight[:, 33:36] > 0.0).all(), "dark-on-dark must admit"
    assert stats["consensus_refused"] >= int(size * 2)


def test_leverage_ledger_attributes_surrender_per_gate() -> None:
    """The leverage ledger: potential/painted/won accounting with
    mutually-exclusive surrender attribution (facing gate, zone gate,
    downstream kill, union drop) and rescued-below-threshold counting."""
    from abstract3d.texturing import assemble_leverage_ledger

    shape = (32, 32)
    surface = np.zeros(shape, dtype=bool)
    surface[4:28, 4:28] = True
    facing = np.full(shape, 0.8, np.float32)
    facing[:, 20:] = 0.15                       # grazing band
    zone = np.zeros(shape, dtype=bool)
    zone[4:10, 4:12] = True                     # layered-zone surrender
    weight = np.where(surface & (facing > 0.4) & ~zone, 0.7, 0.0).astype(np.float32)
    weight[6:8, 22] = 0.02                      # a rescued below-threshold claim
    potential = surface.copy()
    direct = weight > 0

    ledger = assemble_leverage_ledger(
        [{"weight": weight, "potential": potential, "zone": zone,
          "facing": facing.astype(np.float16), "facing_threshold": 0.4,
          "label": "front"}],
        surface_mask=surface, direct_union_mask=direct)

    assert ledger["available"]
    row = ledger["views"][0]
    assert row["rescued_texels"] == 2
    below = surface & ~(facing > 0.4)
    assert row["surrendered_facing_gate"] == int(below.sum()) - 2
    assert row["surrendered_zone_gate"] == int((surface & (facing > 0.4) & zone).sum())
    assert row["surrendered_downstream"] == 0
    assert row["painted_texels"] == int(direct.sum())
    total = (row["painted_texels"] + row["surrendered_facing_gate"]
             + row["surrendered_zone_gate"])
    assert total == int(surface.sum()), "attribution must partition potential"
    # graceful degradation without projector diagnostics
    assert assemble_leverage_ledger(
        [{"weight": weight}], surface_mask=surface,
        direct_union_mask=direct) == {"available": False}


def test_resolve_projection_conflicts_prioritizes_well_facing_source() -> None:
    shape = (16, 16)
    source = {
        "rgba": np.full((*shape, 4), 0.2, dtype=np.float32),
        "weight": np.full(shape, 0.6, dtype=np.float32),   # solid facing
        "label": "source",
    }
    reference = {
        "rgba": np.full((*shape, 4), 0.9, dtype=np.float32),
        "weight": np.full(shape, 0.8, dtype=np.float32),   # even better facing
        "label": "reference",
    }
    stats = texturing.resolve_projection_conflicts([source, reference])
    # Where the source sees the surface well it is ground truth and wins
    # regardless of the reference's higher weight.
    assert stats["conflict_texels"] > 0
    assert float(source["weight"].max()) == pytest.approx(0.6)
    assert float(reference["weight"].max()) == 0.0


def test_resolve_projection_conflicts_defers_grazing_source() -> None:
    shape = (16, 16)
    source = {
        "rgba": np.full((*shape, 4), 0.2, dtype=np.float32),
        "weight": np.full(shape, 0.2, dtype=np.float32),   # grazing rim content
        "label": "source",
    }
    reference = {
        "rgba": np.full((*shape, 4), 0.9, dtype=np.float32),
        "weight": np.full(shape, 0.8, dtype=np.float32),   # head-on witness
        "label": "reference",
    }
    stats = texturing.resolve_projection_conflicts([source, reference])
    # At grazing angles the source's stretched rim samples defer to the
    # head-on reference.
    assert stats["conflict_texels"] > 0
    assert float(source["weight"].max()) == 0.0
    assert float(reference["weight"].max()) == pytest.approx(0.8)


def test_filter_projection_outliers_drops_foreign_islands() -> None:
    import trimesh

    from abstract3d.backends.triposr_runtime import (
        _tripo_make_texture_atlas,
        _tripo_rasterize_position_atlas,
    )

    mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    atlas = _tripo_make_texture_atlas(mesh, texture_resolution=128, texture_padding=2)
    positions = _tripo_rasterize_position_atlas(
        mesh,
        atlas_vmapping=atlas["vmapping"],
        atlas_indices=atlas["indices"],
        atlas_uvs=atlas["uvs"],
        texture_resolution=128,
        texture_padding=2,
    )
    pos = np.asarray(positions)
    surface = pos[:, :, 3] > 0
    shape = surface.shape

    # View 0 dominates everywhere with dark color; view 1 wins a tiny
    # island and paints it bright — a foreign island that must be dropped.
    w0 = np.where(surface, 0.8, 0.0).astype(np.float32)
    w1 = np.zeros(shape, dtype=np.float32)
    ys, xs = np.where(surface)
    island = (ys[:60], xs[:60])
    w1[island] = 0.95
    rgb = np.full((*shape, 3), 0.15, dtype=np.float32)
    rgb[island] = 0.95
    projections = [
        {"rgba": np.zeros((*shape, 4), np.float32), "weight": w0, "label": "a"},
        {"rgba": np.zeros((*shape, 4), np.float32), "weight": w1, "label": "b"},
    ]
    drop = texturing.filter_projection_outliers(
        mesh,
        positions_texture=positions,
        projections=projections,
        blended_rgb=rgb,
        observed_mask=surface,
    )
    assert drop[island].mean() > 0.5
    keep = surface.copy()
    keep[island] = False
    assert drop[keep].mean() < 0.02


def test_pose_estimator_recovers_injected_yaw_and_rejects_frontal() -> None:
    """estimate_pose_photometric must recover a known camera yaw and must NOT
    move a genuinely frontal input (the constant-attractor failure mode that
    sank the first NCC refiner)."""
    import trimesh

    from abstract3d.rendering import render_mesh_views

    # An asymmetric-featured subject: a box with a protruding nose-like knob
    # so gradients carry pose information.
    body = trimesh.creation.box(extents=(1.0, 0.8, 1.2))
    knob = trimesh.creation.icosphere(subdivisions=2, radius=0.25)
    knob.apply_translation([0.55, 0.15, 0.2])
    mesh = trimesh.util.concatenate([body, knob])

    def render_rgba(azimuth: float) -> Image.Image:
        rendered = render_mesh_views(mesh, azimuths=(azimuth,), elevation=0.0, size=384)[0]
        array = np.asarray(rendered.convert("RGBA"), dtype=np.float32) / 255.0
        background = array[2, 2, :3]
        mask = np.abs(array[:, :, :3] - background).sum(axis=2) > 0.08
        rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
        rgba[:, :, :3] = (array[:, :, :3] * 255).astype(np.uint8)
        rgba[:, :, 3] = np.where(mask, 255, 0)
        return Image.fromarray(rgba)

    frontal = texturing.estimate_pose_photometric(
        mesh, render_rgba(0.0), azimuth_window_deg=20.0, elevation_candidates=(0.0,)
    )
    assert frontal["estimated"] is False

    turned = texturing.estimate_pose_photometric(
        mesh, render_rgba(15.0), azimuth_window_deg=20.0, elevation_candidates=(0.0,)
    )
    assert turned["estimated"] is True
    assert abs(float(turned["azimuth_deg"]) - 15.0) <= 5.0


def _winged_pose_mesh():
    """A winged subject whose silhouette is strongly elevation-dependent
    (the x-wing class): thin wide wings edge-on at el 0, plan-view from
    above. A tail fin breaks the top/bottom elevation mirror and an
    off-axis knob breaks bilateral symmetry (chirality carrier)."""
    import trimesh

    body = trimesh.creation.box(extents=(1.4, 0.34, 0.34))
    wings = trimesh.creation.box(extents=(0.55, 1.7, 0.06))
    fin = trimesh.creation.box(extents=(0.28, 0.06, 0.5))
    fin.apply_translation([-0.55, 0.0, 0.35])
    knob = trimesh.creation.icosphere(subdivisions=2, radius=0.16)
    knob.apply_translation([0.62, 0.22, 0.08])
    return trimesh.util.concatenate([body, wings, fin, knob])


def _render_pose_photo(mesh, azimuth: float, elevation: float) -> Image.Image:
    from abstract3d.rendering import render_mesh_views

    rendered = render_mesh_views(
        mesh, azimuths=(azimuth,), elevation=elevation, size=384)[0]
    array = np.asarray(rendered.convert("RGBA"), dtype=np.float32) / 255.0
    background = array[2, 2, :3]
    mask = np.abs(array[:, :, :3] - background).sum(axis=2) > 0.08
    rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    rgba[:, :, :3] = (array[:, :, :3] * 255).astype(np.uint8)
    rgba[:, :, 3] = np.where(mask, 255, 0)
    return Image.fromarray(rgba)


def test_pose_guard_recovers_elevated_capture_via_extended_search() -> None:
    """The x-wing incident class: a capture elevated far outside the
    calibrated band (el +/-15) must be recovered by the extended-elevation
    rescue within one grid step (az 5 / el 8), and the extension must
    report itself in the trail (entered because the calibrated band's best
    registered IoU sits below the action floor)."""
    mesh = _winged_pose_mesh()
    photo = _render_pose_photo(mesh, 10.0, 40.0)

    result = texturing.estimate_pose_with_silhouette_guard(
        mesh, photo, azimuth_window_deg=15.0)

    extension = (result.get("guard_trail") or {}).get("extended_search") or {}
    assert extension.get("consulted") is True, extension
    assert result["estimated"] is True, result
    assert abs(float(result["azimuth_deg"]) - 10.0) <= 5.0, result
    assert abs(float(result["elevation_deg"]) - 40.0) <= 8.0, result


def test_pose_guard_extension_stays_out_of_calibrated_band_decisions() -> None:
    """Fleet-matrix invariant: when the calibrated band explains the photo
    (best in-band registered IoU above the action floor — true for every
    recorded fleet case, measured 0.80-0.97), the extended search must not
    even be consulted, so in-band movers and stayers keep their calibrated
    behavior bit-identically."""
    mesh = _winged_pose_mesh()

    frontal = texturing.estimate_pose_with_silhouette_guard(
        mesh, _render_pose_photo(mesh, 0.0, 0.0), azimuth_window_deg=15.0)
    frontal_extension = (frontal.get("guard_trail") or {}).get("extended_search") or {}
    assert frontal_extension.get("consulted") is False, frontal_extension
    assert float(frontal["azimuth_deg"]) == 0.0 and float(frontal["elevation_deg"]) == 0.0

    in_band = texturing.estimate_pose_with_silhouette_guard(
        mesh, _render_pose_photo(mesh, 10.0, 8.0), azimuth_window_deg=15.0)
    in_band_extension = (in_band.get("guard_trail") or {}).get("extended_search") or {}
    assert in_band_extension.get("consulted") is False, in_band_extension
    assert abs(float(in_band["azimuth_deg"]) - 10.0) <= 5.0, in_band
    assert abs(float(in_band["elevation_deg"]) - 8.0) <= 8.0, in_band


def test_projector_layered_zone_gate_surrenders_film_band() -> None:
    """A thin plate hovering over a larger plate is a film shell: the covered
    band must be surrendered (weights zeroed) and marked contested, while the
    unobstructed region keeps painting."""
    import trimesh

    from abstract3d.backends.triposr_runtime import (
        _tripo_make_texture_atlas,
        _tripo_project_observed_texture,
        _tripo_rasterize_normal_atlas,
        _tripo_rasterize_position_atlas,
    )

    base = trimesh.creation.box(extents=(2.0, 2.0, 0.05))
    film = trimesh.creation.box(extents=(2.0, 0.6, 0.02))
    film.apply_translation([0.0, -0.5, 0.06])  # hover 0.035 over the base top
    mesh = trimesh.util.concatenate([base, film])

    resolution = 128
    atlas = _tripo_make_texture_atlas(mesh, texture_resolution=resolution, texture_padding=2)
    raster_kwargs = dict(
        atlas_vmapping=atlas["vmapping"],
        atlas_indices=atlas["indices"],
        atlas_uvs=atlas["uvs"],
        texture_resolution=resolution,
        texture_padding=2,
    )
    positions = _tripo_rasterize_position_atlas(mesh, **raster_kwargs)
    normals = _tripo_rasterize_normal_atlas(mesh, **raster_kwargs)

    # Two-material content (stripes): the zone gate surrenders layered
    # regions only when the mixture has real material contrast — a uniform
    # photo (hair-over-hair) must keep painting.
    photo_array = np.zeros((256, 256, 4), dtype=np.uint8)
    photo_array[:, :, 3] = 255
    stripes = (np.arange(256) // 8) % 2 == 0
    photo_array[:, stripes, :3] = (230, 210, 190)
    photo_array[:, ~stripes, :3] = (40, 30, 25)
    photo = Image.fromarray(photo_array, "RGBA")
    # Camera above (+z): use elevation 90 via azimuth 0/elevation near 90.
    result = _tripo_project_observed_texture(
        photo,
        positions_texture=positions,
        normals_texture=normals,
        azimuth_deg=0.0,
        elevation_deg=89.0,
        camera_distance=3.0,
        projection_model="orthographic",
        ortho_half_extent=1.3,
    )
    contested = np.asarray(result["contested"])
    assert contested.any(), "film band must be marked contested"

    pos = np.asarray(positions)
    surface = pos[:, :, 3] > 0
    weight = np.asarray(result["weight"])
    # Texels on the base plate top UNDER the film (y in the film's span,
    # z near the base top) must carry zero weight.
    base_top = surface & (np.abs(pos[:, :, 2] - 0.025) < 0.01)
    under_film = base_top & (pos[:, :, 1] > -0.75) & (pos[:, :, 1] < -0.25)
    clear_area = base_top & (pos[:, :, 1] > 0.25)
    if under_film.any() and clear_area.any():
        assert float(weight[under_film].mean()) <= 0.05
        assert float(weight[clear_area].mean()) > 0.1


def test_outlier_filter_drops_island_without_self_votes() -> None:
    """The 2-hop consensus must not let an island certify itself: a foreign
    island wider than one ring still gets dropped (the self-vote bug made the
    filter a no-op for exactly this case)."""
    import trimesh

    from abstract3d.backends.triposr_runtime import (
        _tripo_make_texture_atlas,
        _tripo_rasterize_position_atlas,
    )

    mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    atlas = _tripo_make_texture_atlas(mesh, texture_resolution=128, texture_padding=2)
    positions = _tripo_rasterize_position_atlas(
        mesh,
        atlas_vmapping=atlas["vmapping"],
        atlas_indices=atlas["indices"],
        atlas_uvs=atlas["uvs"],
        texture_resolution=128,
        texture_padding=2,
    )
    pos = np.asarray(positions)
    surface = pos[:, :, 3] > 0
    shape = surface.shape

    # A COMPACT island in 3D: all texels whose surface points sit inside a
    # small cap around a chosen center vertex, so interior members' one-rings
    # are island members (the self-support configuration that survived the
    # buggy filter).
    center = np.array([0.0, 0.0, 1.0])
    points = pos[:, :, :3]
    cap = surface & (np.linalg.norm(points - center, axis=2) < 0.35)
    assert cap.sum() > 100

    w0 = np.where(surface, 0.8, 0.0).astype(np.float32)
    w0[cap] = 0.0
    w1 = np.zeros(shape, dtype=np.float32)
    w1[cap] = 0.95
    rgb = np.full((*shape, 3), 0.15, dtype=np.float32)
    rgb[cap] = 0.95
    projections = [
        {"rgba": np.zeros((*shape, 4), np.float32), "weight": w0, "label": "a"},
        {"rgba": np.zeros((*shape, 4), np.float32), "weight": w1, "label": "b"},
    ]
    drop = texturing.filter_projection_outliers(
        mesh,
        positions_texture=positions,
        projections=projections,
        blended_rgb=rgb,
        observed_mask=surface,
    )
    assert drop[cap].mean() > 0.5, "compact foreign cap must erode"


def test_register_reference_by_source_overlap_recovers_injected_shift() -> None:
    """The overlap-photometric registration must recover a known injected
    shift of the reference photo (the acceptance bar the removed NCC refiner
    failed)."""
    import trimesh

    from abstract3d.backends.triposr_runtime import (
        _tripo_make_texture_atlas,
        _tripo_project_observed_texture,
        _tripo_rasterize_normal_atlas,
        _tripo_rasterize_position_atlas,
    )

    mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    resolution = 128
    atlas = _tripo_make_texture_atlas(mesh, texture_resolution=resolution, texture_padding=2)
    raster_kwargs = dict(
        atlas_vmapping=atlas["vmapping"],
        atlas_indices=atlas["indices"],
        atlas_uvs=atlas["uvs"],
        texture_resolution=resolution,
        texture_padding=2,
    )
    positions = _tripo_rasterize_position_atlas(mesh, **raster_kwargs)
    normals = _tripo_rasterize_normal_atlas(mesh, **raster_kwargs)

    # Textured photo with strong structure.
    rng = np.random.default_rng(11)
    tile = (rng.random((16, 16, 3)) * 255).astype(np.uint8)
    pattern = np.kron(tile, np.ones((16, 16, 1), dtype=np.uint8))
    photo_array = np.dstack([pattern, np.full(pattern.shape[:2], 255, dtype=np.uint8)])
    photo = Image.fromarray(photo_array, "RGBA")

    source = _tripo_project_observed_texture(
        photo,
        positions_texture=positions,
        normals_texture=normals,
        azimuth_deg=0.0,
        elevation_deg=0.0,
        camera_distance=3.0,
        projection_model="orthographic",
        ortho_half_extent=1.2,
    )

    # Same photo shifted by a known offset plays the misregistered reference
    # at the same pose (full overlap with the source).
    dx_true = 0.03  # fraction of width
    width, height = photo.size
    shifted = photo.transform(
        (width, height),
        Image.AFFINE,
        (1.0, 0.0, dx_true * width, 0.0, 1.0, 0.0),
        resample=Image.BILINEAR,
        fillcolor=(0, 0, 0, 0),
    )

    _, stats = texturing.register_reference_by_source_overlap(
        shifted,
        positions_texture=positions,
        source_projection=source,
        azimuth_deg=0.0,
        elevation_deg=0.0,
        camera_distance=3.0,
        ortho_half_extent=1.2,
    )
    assert stats["applied"] is True
    # The PIL AFFINE above moves the content LEFT by dx_true; the optimizer's
    # convention moves content RIGHT by its dx, so recovery means
    # dx ~= +dx_true.
    assert abs(float(stats["shift_x"]) - dx_true) <= 0.015
    assert float(stats["err_after"]) < float(stats["err_before"]) - 0.005


def test_clean_alpha_mask_removes_floaters_and_holes() -> None:
    from abstract3d.segmentation import clean_alpha_mask

    image = np.zeros((100, 100, 4), dtype=np.uint8)
    image[20:80, 20:80, 3] = 255      # subject
    image[45:50, 45:50, 3] = 0        # pinhole
    image[5:8, 5:8, 3] = 255          # floater
    cleaned = np.asarray(clean_alpha_mask(Image.fromarray(image, mode="RGBA")))
    assert cleaned[46, 46, 3] > 0     # hole closed
    assert cleaned[6, 6, 3] == 0      # floater removed
    assert cleaned[50, 50, 3] == 255  # subject intact


def test_bake_projection_texture_end_to_end_on_simple_mesh() -> None:
    import trimesh

    mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.5)
    observed = Image.new("RGBA", (128, 128), (200, 60, 40, 255))

    textured, stats = texturing.bake_projection_texture(
        mesh,
        observed_views=[{"rgba": observed, "azimuth_deg": 0.0, "elevation_deg": 0.0, "label": "front"}],
        texture_resolution=128,
    )

    # The strict first-surface z-buffer trims about one pixel of rim (its
    # 3x3 conservative widening), so the front hemisphere of a sphere at
    # matched photo/atlas resolution covers just under 10% of all texels.
    assert stats["observed_coverage_ratio"] > 0.08
    assert stats["unseen_fill_mode"] in {"mesh_harmonic", "nearest_observed_3d", "backend_color_field"}
    assert stats["texture_image"].size == (128, 128)
    assert textured.visual.uv is not None
    # the projected side of the sphere must pick up the observed red
    texture = np.asarray(stats["texture_image"], dtype=np.float32)
    assert float(texture[:, :, 0].mean()) > float(texture[:, :, 2].mean())


def test_level_composed_seams_cancels_tone_step_and_keeps_detail() -> None:
    import trimesh

    # Sphere atlas split into two regions along the x=0 plane with a
    # constant tone offset between them (a synthetic view-handoff seam) and
    # a high-frequency stripe pattern riding on both sides. Leveling must
    # cancel the step at the boundary while the stripes survive.
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.5)
    size = 128
    us = np.linspace(-np.pi, np.pi, size, endpoint=False, dtype=np.float32)
    vs = np.linspace(-0.4 * np.pi, 0.4 * np.pi, size, dtype=np.float32)
    grid_u, grid_v = np.meshgrid(us, vs)
    positions = np.zeros((size, size, 4), dtype=np.float32)
    positions[:, :, 0] = 0.5 * np.cos(grid_v) * np.cos(grid_u)
    positions[:, :, 1] = 0.5 * np.cos(grid_v) * np.sin(grid_u)
    positions[:, :, 2] = 0.5 * np.sin(grid_v)
    positions[:, :, 3] = 1.0

    region = np.where(positions[:, :, 0] > 0.0, 0, 1).astype(np.int32)
    stripes = 0.02 * np.sign(np.sin(24.0 * grid_u)).astype(np.float32)
    colors = np.zeros((size, size, 3), dtype=np.float32)
    colors[:, :, :] = (0.60 + stripes)[:, :, None]
    colors[region == 1] += 0.10  # the seam: region 1 is uniformly brighter

    offsets = texturing.level_composed_seams(
        mesh,
        positions_texture=positions,
        colors_rgb=colors,
        region_map=region,
    )

    assert offsets is not None
    corrected = colors + offsets
    # Region-mean agreement (the stripes are zero-mean, so region means
    # isolate the tone step): the 0.10 step must collapse by >3x.
    step_before = abs(float(colors[region == 0, 0].mean()) - float(colors[region == 1, 0].mean()))
    step_after = abs(
        float(corrected[region == 0, 0].mean()) - float(corrected[region == 1, 0].mean())
    )
    assert step_after < 0.35 * step_before
    # High-frequency stripes survive: within-region stripe amplitude stays.
    stripe_amp_after = float(np.abs(np.diff(corrected[64, :, 0])).max())
    assert stripe_amp_after > 0.02


def test_level_composed_seams_skips_material_edges() -> None:
    import trimesh

    # Two regions whose colors differ by far more than any exposure seam
    # (hair against skin). The boundary cap must exclude those edges: with
    # no seam-like boundary at all the solver returns None (nothing to do).
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.5)
    size = 96
    us = np.linspace(-np.pi, np.pi, size, endpoint=False, dtype=np.float32)
    vs = np.linspace(-0.4 * np.pi, 0.4 * np.pi, size, dtype=np.float32)
    grid_u, grid_v = np.meshgrid(us, vs)
    positions = np.zeros((size, size, 4), dtype=np.float32)
    positions[:, :, 0] = 0.5 * np.cos(grid_v) * np.cos(grid_u)
    positions[:, :, 1] = 0.5 * np.cos(grid_v) * np.sin(grid_u)
    positions[:, :, 2] = 0.5 * np.sin(grid_v)
    positions[:, :, 3] = 1.0

    region = np.where(positions[:, :, 0] > 0.0, 0, 1).astype(np.int32)
    colors = np.full((size, size, 3), 0.75, dtype=np.float32)  # skin
    colors[region == 1] = 0.15  # hair: |delta| = 0.60 >> boundary_cap

    offsets = texturing.level_composed_seams(
        mesh,
        positions_texture=positions,
        colors_rgb=colors,
        region_map=region,
    )

    assert offsets is None


def test_level_composed_seams_pins_confident_witnesses() -> None:
    import trimesh

    # Region 0 is a confident witness (weight 0.9), region 1 is weak. The
    # confident side must keep its tone (photo = ground truth); the weak
    # side must move toward it.
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.5)
    size = 96
    us = np.linspace(-np.pi, np.pi, size, endpoint=False, dtype=np.float32)
    vs = np.linspace(-0.4 * np.pi, 0.4 * np.pi, size, dtype=np.float32)
    grid_u, grid_v = np.meshgrid(us, vs)
    positions = np.zeros((size, size, 4), dtype=np.float32)
    positions[:, :, 0] = 0.5 * np.cos(grid_v) * np.cos(grid_u)
    positions[:, :, 1] = 0.5 * np.cos(grid_v) * np.sin(grid_u)
    positions[:, :, 2] = 0.5 * np.sin(grid_v)
    positions[:, :, 3] = 1.0

    region = np.where(positions[:, :, 0] > 0.0, 0, 1).astype(np.int32)
    colors = np.full((size, size, 3), 0.70, dtype=np.float32)
    colors[region == 1] = 0.60
    confidence = np.where(region == 0, 0.9, 0.1).astype(np.float32)

    offsets = texturing.level_composed_seams(
        mesh,
        positions_texture=positions,
        colors_rgb=colors,
        region_map=region,
        confidence_map=confidence,
    )

    assert offsets is not None
    mean_confident = float(np.abs(offsets[region == 0]).mean())
    mean_weak = float(np.abs(offsets[region == 1]).mean())
    assert mean_weak > 2.0 * mean_confident


def test_mirror_fill_consensus_guard_rejects_material_crossing_copies() -> None:
    # Two parallel sheets mirrored across y=0: left (y=-0.25) fully observed,
    # right (y=+0.25) observed except a hole. The hole's twins land on the
    # left sheet, whose content is skin except a dark hair band across from
    # PART of the hole. The destination neighborhood (observed right-sheet
    # skin around the hole) is color-consistent, so the guard must reject
    # the dark copies while keeping the skin-colored ones.
    half = 32
    size = 64
    xs = np.linspace(0.0, 1.0, half, dtype=np.float32)
    zs = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    positions = np.zeros((size, size, 4), dtype=np.float32)
    for row in range(size):
        positions[row, :half, 0] = xs
        positions[row, half:, 0] = xs
        positions[row, :half, 1] = -0.25
        positions[row, half:, 1] = 0.25
        positions[row, :, 2] = zs[row]
    positions[:, :, 3] = 1.0

    observed = np.ones((size, size), dtype=bool)
    hole_rows = slice(20, 44)
    hole_cols = slice(int(half + 8), int(half + 14))  # right-sheet x ~ 0.26..0.42
    observed[hole_rows, hole_cols] = False

    colors = np.full((size, size, 3), 0.75, dtype=np.float32)
    # dark hair band on the LEFT sheet across from the hole's upper half only
    colors[20:32, 8:14] = 0.10

    kwargs = dict(
        positions_texture=positions,
        observed_mask=observed,
        colors_rgb=colors,
        axis=1,
        max_distance_ratio=0.05,
        consensus_radius_ratio=0.08,
    )
    fill_rgb, fill_mask = texturing.mirror_fill_from_observed(consensus_guard=True, **kwargs)
    filled = fill_mask[hole_rows, hole_cols]
    values = fill_rgb[hole_rows, hole_cols][filled]
    assert filled.any()
    # No dark hair copies may land inside the skin-consistent hole.
    assert float(values.min()) > 0.5

    # Without the guard the dark band IS copied (the scenario really does
    # exercise the guard).
    fill_rgb2, fill_mask2 = texturing.mirror_fill_from_observed(consensus_guard=False, **kwargs)
    values2 = fill_rgb2[hole_rows, hole_cols][fill_mask2[hole_rows, hole_cols]]
    assert float(values2.min()) < 0.5


# ---------------------------------------------------------------------------
# photometric delighting (SH-in-normal-space shading removal)
# ---------------------------------------------------------------------------

def _sphere_two_light_case(seed: int = 7, resolution: int = 160):
    """Two 'projections' of one albedo sphere under different lights.

    Spherical-parameterization atlas (theta x phi); albedo carries a
    low-frequency tint plus high-frequency dark freckles; each view has its
    own Lambertian-plus-ambient shading and a facing-derived weight map with
    a wide overlap lens between the two camera directions.
    """
    rng = np.random.default_rng(seed)
    thetas = np.linspace(0.05, np.pi - 0.05, resolution)
    phis = np.linspace(-np.pi, np.pi, resolution, endpoint=False)
    theta, phi = np.meshgrid(thetas, phis, indexing="ij")
    normals = np.stack(
        [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)],
        axis=2,
    )

    base = 0.55 + 0.1 * np.sin(3.0 * phi) * np.sin(2.0 * theta)
    freckles = np.zeros_like(base)
    yy, xx = np.mgrid[0:resolution, 0:resolution]
    for _ in range(120):
        cx, cy = rng.uniform(0, resolution, 2)
        radius = rng.uniform(1.5, 4.0)
        freckles -= 0.35 * np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * radius * radius)))
    albedo_lum = np.clip(base + freckles, 0.05, 1.0)
    albedo = np.stack([albedo_lum, albedo_lum * 0.8, albedo_lum * 0.7], axis=2)

    def view(light_dir, camera_dir, label):
        light = np.asarray(light_dir, dtype=np.float64)
        light /= np.linalg.norm(light)
        shading = 0.25 + 0.75 * np.clip(normals @ light, 0.0, None)
        rgba = np.zeros((resolution, resolution, 4), dtype=np.float32)
        rgba[:, :, :3] = np.clip(albedo * shading[:, :, None], 0.0, 1.0)
        camera = np.asarray(camera_dir, dtype=np.float64)
        camera /= np.linalg.norm(camera)
        weight = np.clip((normals @ camera - 0.2) / 0.8, 0.0, 1.0) ** 2
        rgba[:, :, 3] = (weight > 0).astype(np.float32)
        return {"rgba": rgba, "weight": weight.astype(np.float32), "label": label}

    views = [
        view([0.8, 0.5, 0.6], [1.0, 0.0, 0.0], "source"),
        view([-0.3, -0.9, 0.2], [0.2, 0.97, 0.05], "reference"),
    ]
    normals_texture = np.concatenate(
        [normals, np.ones((*normals.shape[:2], 1))], axis=2
    ).astype(np.float32)
    return views, normals_texture, albedo


def test_delight_projections_recovers_agreeing_albedo_on_two_light_sphere() -> None:
    projections, normals_texture, _ = _sphere_two_light_case()
    w0 = projections[0]["weight"]
    w1 = projections[1]["weight"]
    overlap = (w0 > 0.05) & (w1 > 0.05)
    rgb0 = np.asarray(projections[0]["rgba"])[:, :, :3]
    rgb1_before = np.asarray(projections[1]["rgba"])[:, :, :3].copy()
    before = float(np.abs(rgb1_before[overlap] - rgb0[overlap]).mean())

    stats = texturing.delight_projections(projections, normals_texture=normals_texture)

    rgb1_after = np.asarray(projections[1]["rgba"])[:, :, :3]
    after = float(np.abs(rgb1_after[overlap] - rgb0[overlap]).mean())
    assert stats["applied"], "delighting must engage on a pure two-light case"
    # The albedo cancels in the overlap ratio, so the recovered albedos must
    # agree far better than the lit inputs did (the residual is the order-2
    # SH truncation of log(ambient + clamped Lambert) plus the amplitude cap).
    assert before / max(after, 1e-9) > 3.0
    # source is the gauge: its colors must be bit-identical
    assert np.array_equal(rgb0, np.asarray(projections[0]["rgba"])[:, :, :3])


def test_delight_projections_fade_protects_exclusive_territory() -> None:
    """With positions, the correction applies near the overlap and fades to
    zero deep inside the reference's exclusive region: overlap disagreement
    still drops, exclusive texels far from the overlap stay bit-identical
    (per-view identity contract at that photo's own pose)."""
    projections, normals_texture, _ = _sphere_two_light_case()
    # unit sphere: positions = normals
    positions_texture = normals_texture.copy()
    w0 = projections[0]["weight"]
    w1 = projections[1]["weight"]
    overlap = (w0 > 0.05) & (w1 > 0.05)
    rgb1_before = np.asarray(projections[1]["rgba"])[:, :, :3].copy()
    rgb0 = np.asarray(projections[0]["rgba"])[:, :, :3]
    before = float(np.abs(rgb1_before[overlap] - rgb0[overlap]).mean())

    stats = texturing.delight_projections(
        projections, normals_texture=normals_texture,
        positions_texture=positions_texture)

    rgb1_after = np.asarray(projections[1]["rgba"])[:, :, :3]
    after = float(np.abs(rgb1_after[overlap] - rgb0[overlap]).mean())
    assert stats["applied"] and after < before

    # deep-exclusive texels (far from any overlap point) are untouched
    from scipy.spatial import cKDTree

    normals = normals_texture[:, :, :3]
    tree = cKDTree(normals[overlap])
    exclusive = (w1 > 0.05) & ~overlap
    distances, _ = tree.query(normals[exclusive], k=1, workers=-1)
    deep = np.zeros_like(exclusive)
    deep[exclusive] = distances > 0.5  # far in world units (unit sphere)
    if deep.any():
        assert float(np.abs(rgb1_after[deep] - rgb1_before[deep]).max()) < 1e-5


def test_delight_projections_keeps_chroma_and_reverts_on_confound() -> None:
    projections, normals_texture, _ = _sphere_two_light_case()
    # chroma ratio invariance on the corrected view
    rgb_before = np.asarray(projections[1]["rgba"])[:, :, :3].copy()
    texturing.delight_projections(projections, normals_texture=normals_texture)
    rgb_after = np.asarray(projections[1]["rgba"])[:, :, :3]
    sample = (rgb_before[:, :, 0] > 0.05) & (rgb_after[:, :, 0] > 1e-4)
    ratio_rg_before = rgb_before[:, :, 0][sample] / np.maximum(rgb_before[:, :, 1][sample], 1e-6)
    ratio_rg_after = rgb_after[:, :, 0][sample] / np.maximum(rgb_after[:, :, 1][sample], 1e-6)
    unclipped = (rgb_after.max(axis=2)[sample] < 0.999)
    assert np.allclose(ratio_rg_before[unclipped], ratio_rg_after[unclipped], atol=5e-3), (
        "the correction must be luminance-only (chroma untouched)"
    )

    # confound: two views whose overlap difference is CONTENT (random),
    # not normal-dependent shading -> no correction may be kept
    rng = np.random.default_rng(3)
    confound = [
        {
            "rgba": np.clip(p["rgba"] + rng.uniform(-0.3, 0.3, p["rgba"].shape).astype(np.float32) * (i > 0), 0, 1),
            "weight": p["weight"],
            "label": p["label"],
        }
        for i, p in enumerate(projections)
    ]
    reference_before = np.asarray(confound[1]["rgba"], dtype=np.float32).copy()
    stats = texturing.delight_projections(confound, normals_texture=normals_texture)
    applied_rows = [row for row in stats["views"] if row.get("applied")]
    if applied_rows:  # if it engaged, it must have measurably improved
        for row in applied_rows:
            assert row["disagreement_after"] < row["disagreement_before"] - 0.002
    else:
        assert np.array_equal(reference_before, np.asarray(confound[1]["rgba"]))


# ---------------------------------------------------------------------------
# consensus tone equalization (level gains + local consensus field)
# ---------------------------------------------------------------------------

def _tone_case_views(shape=(96, 96), seed: int = 7):
    """Planar three-view chain: real source, two generated references.

    Column bands with wide overlaps: source [0, 40), gen mid [30, 70),
    gen far [60, end). Base albedo is shared; per-view level/field errors
    are injected by the individual tests.
    """
    rng = np.random.default_rng(seed)
    base = 0.45 + 0.05 * rng.standard_normal((*shape, 3)).astype(np.float32)
    positions = np.zeros((*shape, 4), dtype=np.float32)
    xs, ys = np.meshgrid(
        np.linspace(-1.0, 1.0, shape[1]), np.linspace(-1.0, 1.0, shape[0]))
    positions[:, :, 0] = xs
    positions[:, :, 1] = ys
    positions[:, :, 3] = 1.0

    def view(rgb, cols, label, generated):
        rgba = np.concatenate(
            [np.clip(rgb, 0.0, 1.0),
             np.ones((*shape, 1), dtype=np.float32)], axis=2)
        weight = np.zeros(shape, dtype=np.float32)
        weight[:, cols[0]:cols[1]] = 0.5
        return {"rgba": rgba, "weight": weight, "label": label,
                "generated": generated}

    return base, positions, view


def test_equalize_projection_tone_levels_chained_views() -> None:
    """Scalar stage: level offsets are recovered through the pair chain
    (far view has NO source overlap), the chained view's cap halves, and
    corrected views agree with the shared albedo."""
    base, positions, view = _tone_case_views()
    src = view(base, (0, 40), "front", generated=False)
    mid = view(base * 1.6, (30, 70), "mid", generated=True)
    far = view(base * 0.7, (60, 96), "far", generated=True)
    projections = [src, mid, far]

    stats = texturing.equalize_projection_tone(
        projections, positions_texture=positions, source_index=0)

    rows = {row["label"]: row for row in stats["views"]
            if "gain_log" in row}
    assert rows["mid"]["applied"] and rows["far"]["applied"]
    # true gains: log(1.6) = 0.470, log(0.7) = -0.357 (chain-consistent, so
    # the joint solve recovers them; tolerance covers clipping at 0/1)
    assert abs(rows["mid"]["gain_log"] - np.log(1.6)) < 0.05
    assert abs(rows["far"]["gain_log"] + np.log(1 / 0.7)) < 0.05
    assert not rows["mid"]["chained_gauge"]
    assert rows["far"]["chained_gauge"]
    mid_rgb = np.asarray(projections[1]["rgba"])[:, :, :3]
    far_rgb = np.asarray(projections[2]["rgba"])[:, :, :3]
    assert abs(float(mid_rgb[:, 30:70].mean()) - float(base[:, 30:70].mean())) < 0.01
    # full conformance on the overlap evidence; the exclusive far edge
    # keeps its own level (the correction fades with its evidence)
    assert abs(float(far_rgb[:, 60:68].mean()) - float(base[:, 60:68].mean())) < 0.01
    far_original = np.clip(base * 0.7, 0.0, 1.0)
    assert abs(float(far_rgb[:, 90:].mean()) - float(far_original[:, 90:].mean())) < 0.01


def test_equalize_projection_tone_pins_real_views() -> None:
    """Real photo views are gauge-fixed: a bright REAL reference is never
    corrected (photos define the level); the generated view conforms."""
    base, positions, view = _tone_case_views()
    src = view(base, (0, 40), "front", generated=False)
    real_ref = view(base * 1.5, (30, 70), "real_side", generated=False)
    gen = view(base * 0.7, (60, 96), "gen_back", generated=True)
    projections = [src, real_ref, gen]
    real_before = np.array(projections[1]["rgba"], copy=True)

    stats = texturing.equalize_projection_tone(
        projections, positions_texture=positions, source_index=0)

    assert np.array_equal(real_before, np.asarray(projections[1]["rgba"]))
    labels_with_gain = {row["label"] for row in stats["views"]
                        if "gain_log" in row}
    assert labels_with_gain == {"gen_back"}
    # single-view input is a structural no-op
    assert texturing.equalize_projection_tone(
        [dict(src)], source_index=0) == {
        "applied": False, "views": [], "pairs": []}


def test_equalize_projection_tone_local_field_regional_deviation() -> None:
    """Stage 2: a zero-median REGIONAL deviation (bright top half, dark
    bottom half — invisible to any scalar gain) is reconciled toward the
    protected photo consensus; deep-exclusive texels stay bit-identical
    (fade) and chroma ratios are untouched (luminance-only)."""
    base, positions, view = _tone_case_views(shape=(128, 128))
    src = view(base, (0, 60), "front", generated=False)
    warped = base.copy()
    warped[:64, 30:] *= 1.45
    warped[64:, 30:] /= 1.45
    gen = view(warped, (30, 128), "gen_side", generated=True)
    projections = [src, gen]
    gen_before = np.asarray(projections[1]["rgba"])[:, :, :3].copy()

    stats = texturing.equalize_projection_tone(
        projections, positions_texture=positions, source_index=0)

    field_rows = [row for row in stats["views"]
                  if row.get("stage") == "local_field"]
    assert len(field_rows) == 1 and field_rows[0]["applied"]
    # scalar stage cannot see the deviation (median ~ 0); the field must
    # carry the correction
    scalar_rows = [row for row in stats["views"] if "gain_log" in row]
    assert all(abs(row["gain_log"]) < 0.1 for row in scalar_rows)
    gen_after = np.asarray(projections[1]["rgba"])[:, :, :3]
    overlap = slice(35, 55)
    before_err = float(np.abs(gen_before[:, overlap] - base[:, overlap]).mean())
    after_err = float(np.abs(gen_after[:, overlap] - base[:, overlap]).mean())
    # substantially reconciled toward the photo (the protected consensus is
    # the photo's own reading, so conformance is near-full up to the
    # smoothing band at the half boundary and the density fade)
    assert after_err < 0.6 * before_err
    # deep-exclusive territory (far from any overlap in world units) is
    # bit-identical: the correction dies with its evidence
    deep = slice(100, 128)
    assert np.array_equal(gen_before[:, deep], gen_after[:, deep])
    # luminance-only: r/g ratios preserved where unclipped
    sample = (gen_before[:, overlap, 1] > 0.05) & (gen_after[:, overlap].max(axis=2) < 0.999)
    ratio_before = gen_before[:, overlap, 0][sample] / np.maximum(gen_before[:, overlap, 1][sample], 1e-6)
    ratio_after = gen_after[:, overlap, 0][sample] / np.maximum(gen_after[:, overlap, 1][sample], 1e-6)
    assert np.allclose(ratio_before, ratio_after, atol=5e-3)


def test_equalize_projection_tone_fails_closed_on_content_confound() -> None:
    """Overlap disagreement that is CONTENT (random, not tone) must not
    ship a correction: under the witness-RANKED gate, any applied row must
    have left the real-photo class no worse AND measurably improved one
    class; otherwise the view is bit-identical."""
    base, positions, view = _tone_case_views()
    rng = np.random.default_rng(3)
    src = view(base, (0, 40), "front", generated=False)
    noisy = np.clip(
        base + rng.uniform(-0.3, 0.3, base.shape).astype(np.float32), 0, 1)
    gen = view(noisy, (30, 70), "gen", generated=True)
    projections = [src, gen]
    gen_before = np.array(projections[1]["rgba"], copy=True)

    stats = texturing.equalize_projection_tone(
        projections, positions_texture=positions, source_index=0)

    applied_rows = [row for row in stats["views"] if row.get("applied")]
    if applied_rows:
        for row in applied_rows:
            before = row["disagreement_before"]
            after = row["disagreement_after"]
            assert any(after[k] < before[k] - 0.002 for k in before)
            if "real" in before:
                assert after["real"] <= before["real"] + 0.002
    else:
        assert np.array_equal(gen_before, np.asarray(projections[1]["rgba"]))


def test_equalize_projection_tone_tiny_overlap_is_structural_noop() -> None:
    """Adversarial worst case (integrator program): ONE generated view
    whose only source overlap is below the 400-texel pair floor must be a
    structural no-op — no fit, no correction, every pixel bit-identical.
    (Fitting a whole-view gain on statistical dust was the failure mode
    the pair floor exists to refuse.)"""
    base, positions, view = _tone_case_views()
    src = view(base, (0, 40), "front", generated=False)
    # 2 overlap columns x 96 rows = 192 texels < 400
    gen = view(base * 2.0, (38, 96), "gen", generated=True)
    projections = [src, gen]
    before = [np.array(p["rgba"], copy=True) for p in projections]

    stats = texturing.equalize_projection_tone(
        projections, positions_texture=positions, source_index=0)

    assert stats == {"applied": False, "views": [], "pairs": []}
    for prior, projection in zip(before, projections):
        assert np.array_equal(prior, np.asarray(projection["rgba"]))


def test_equalize_projection_tone_source_starved_keeps_common_mode() -> None:
    """Adversarial worst case: an all-generated component (no view pairs
    with the source) has an unobservable common mode — the pass may only
    reconcile the views RELATIVELY (mean-zero gauge in log space), never
    relight the group toward some invented level, and the source stays
    bit-identical."""
    base, positions, view = _tone_case_views()
    src = view(base, (0, 30), "front", generated=False)
    gen_a = view(base * 1.8, (40, 70), "gen_a", generated=True)
    gen_b = view(base * 0.9, (60, 96), "gen_b", generated=True)
    projections = [src, gen_a, gen_b]
    src_before = np.array(src["rgba"], copy=True)
    overlap = slice(60, 70)

    def log_mean(projection):
        rgb = np.asarray(projection["rgba"], np.float64)[:, overlap, :3]
        return float(np.log(np.clip(rgb, 1e-4, None)).mean())

    common_before = 0.5 * (log_mean(gen_a) + log_mean(gen_b))
    gap_before = abs(log_mean(gen_a) - log_mean(gen_b))

    stats = texturing.equalize_projection_tone(
        projections, positions_texture=positions, source_index=0)

    assert np.array_equal(src_before, np.asarray(src["rgba"]))
    # the source participates in no pair: the fit graph is generated-only
    assert all("front" not in row["views"] for row in stats["pairs"])
    common_after = 0.5 * (log_mean(gen_a) + log_mean(gen_b))
    gap_after = abs(log_mean(gen_a) - log_mean(gen_b))
    # relative disagreement shrinks; the joint (common-mode) level does not
    # move beyond numerical residue of the mean-zero gauge
    assert gap_after < 0.5 * gap_before
    assert abs(common_after - common_before) < 0.02


def test_voxel_field_mean_c0_is_continuous_and_center_exact() -> None:
    """The C0 voxel statistic: equals the box mean at cell centers, is
    continuous across cell boundaries (no lattice steps), and keeps the
    NaN contract where no occupied cell is in reach."""
    cell = 0.25
    # dense line of points along x at y=z=0.375 (mid-cell), values = x
    xs = np.linspace(0.001, 0.999, 4001)
    points = np.stack([xs, np.full_like(xs, 0.375),
                       np.full_like(xs, 0.375)], axis=1)
    values = xs.copy()

    out = texturing._voxel_field_mean_c0(points, values, cell)

    # continuity: adjacent query points 0.00025 apart may not step — the
    # box-mean variant steps by ~cell (0.25) at every cell boundary
    steps = np.abs(np.diff(out))
    assert float(steps.max()) < 0.01, (
        "C0 statistic must not step at voxel-lattice boundaries "
        f"(max step {steps.max():.4f})")
    box = texturing._voxel_neighborhood_mean(points, values, cell)
    box_steps = np.abs(np.diff(box))
    assert float(box_steps.max()) > 0.1, (
        "fixture must actually straddle lattice boundaries")
    # center-exactness: at a cell's center the interpolation reproduces
    # the 3x3x3 box mean (grid origin = the point cloud's min corner;
    # tolerance covers the 2.5e-4 sample-grid offset from the true center)
    center_x = float(xs.min()) + 1.5 * cell
    center_index = int(np.argmin(np.abs(xs - center_x)))
    assert abs(out[center_index] - box[center_index]) < 1e-3
    # NaN contract under `select`: with no contributor anywhere near, the
    # query reads NaN exactly like the box variant
    select = xs > 0.9
    sparse = texturing._voxel_field_mean_c0(points, values, cell,
                                            select=select)
    assert np.isnan(sparse[0]) and np.isfinite(sparse[-1])


def test_equalize_projection_tone_field_prints_no_lattice_blocks() -> None:
    """THE ROOF-BLOCK CLASS (car_bo3 live incident, second stamp-class
    defect): a generated view fed DISPLACED content builds a violent
    consensus-deviation field; the smoothed field used to be
    piecewise-constant over the voxel lattice and printed rectangular
    exposure blocks ("image inside an image") on flat surfaces. The
    applied multiplicative correction must now be C0: no adjacent-texel
    log step above the content-driven gradient scale (measured on this
    fixture: 0.372 with the box statistic, 0.069 with the C0 one), and
    every applied row must name its write footprint (provenance).
    """
    shape = (128, 128)
    rng = np.random.default_rng(7)
    base = 0.45 + 0.05 * rng.standard_normal((*shape, 3)).astype(np.float32)
    yy, xx = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]),
                         indexing="ij")
    base[((yy - 40) ** 2 + (xx - 34) ** 2) < 14 ** 2] *= 1.9
    base = np.clip(base, 0.0, 1.0)
    positions = np.zeros((*shape, 4), dtype=np.float32)
    xs, ys = np.meshgrid(np.linspace(-1, 1, shape[1]),
                         np.linspace(-1, 1, shape[0]))
    positions[:, :, 0] = xs
    positions[:, :, 1] = ys
    positions[:, :, 3] = 1.0

    def view(rgb, cols, label, generated):
        rgba = np.concatenate(
            [np.clip(rgb, 0, 1), np.ones((*shape, 1), np.float32)], axis=2)
        weight = np.zeros(shape, np.float32)
        weight[:, cols[0]:cols[1]] = 0.5
        return {"rgba": rgba, "weight": weight, "label": label,
                "generated": generated}

    src = view(base, (0, 80), "front", generated=False)
    displaced = np.roll(np.roll(base, 22, axis=0), 18, axis=1)
    gen = view(displaced, (0, 128), "gen_top", generated=True)
    projections = [src, gen]
    gen_before = np.asarray(projections[1]["rgba"], np.float32)[:, :, :3].copy()

    stats = texturing.equalize_projection_tone(
        projections, positions_texture=positions, source_index=0)

    gen_after = np.asarray(projections[1]["rgba"], np.float32)[:, :, :3]
    changed = np.abs(gen_after - gen_before).max(axis=2) > 1e-6
    applied_rows = [row for row in stats["views"] if row.get("applied")]
    if changed.any():
        # honest provenance: every writer names its footprint
        assert applied_rows, "texels changed but no stats row claims a write"
        assert all(row.get("written_texels", 0) > 0 for row in applied_rows)
        # C0 contract: the applied multiplicative field may not step
        # between adjacent texels of the same flat surface beyond the
        # content-driven gradient scale (the block class printed 0.372)
        log_ratio = (np.log(np.clip(gen_after.mean(axis=2), 1e-3, None))
                     - np.log(np.clip(gen_before.mean(axis=2), 1e-3, None)))
        worst = 0.0
        for axis in (0, 1):
            a = np.take(log_ratio, range(0, shape[axis] - 1), axis=axis)
            b = np.take(log_ratio, range(1, shape[axis]), axis=axis)
            ca = np.take(changed, range(0, shape[axis] - 1), axis=axis)
            cb = np.take(changed, range(1, shape[axis]), axis=axis)
            both = ca & cb
            if both.any():
                worst = max(worst, float(np.abs(a - b)[both].max()))
        assert worst < 0.15, (
            f"lattice-block class: adjacent-texel field step {worst:.3f}")
    else:
        # refusal must be a true no-op: bit-identical, no write claimed
        assert not applied_rows
        assert np.array_equal(gen_before, gen_after)


def test_equalize_projection_tone_refusal_is_bit_identical_no_op() -> None:
    """Fail-closed honesty: when every gate refuses (content confound —
    the displaced-content case whose correction cannot measurably help
    any witness class), the projection must be BIT-IDENTICAL and no row
    may claim a write footprint. (The first stamp incident's lesson
    generalized: a lane that cannot verify its evidence must decline to
    write at all, and the stats must say so.)"""
    shape = (96, 96)
    rng = np.random.default_rng(3)
    base = 0.45 + 0.05 * rng.standard_normal((*shape, 3)).astype(np.float32)
    positions = np.zeros((*shape, 4), dtype=np.float32)
    xs, ys = np.meshgrid(np.linspace(-1, 1, shape[1]),
                         np.linspace(-1, 1, shape[0]))
    positions[:, :, 0] = xs
    positions[:, :, 1] = ys
    positions[:, :, 3] = 1.0

    def view(rgb, cols, label, generated):
        rgba = np.concatenate(
            [np.clip(rgb, 0, 1), np.ones((*shape, 1), np.float32)], axis=2)
        weight = np.zeros(shape, np.float32)
        weight[:, cols[0]:cols[1]] = 0.5
        return {"rgba": rgba, "weight": weight, "label": label,
                "generated": generated}

    src = view(base, (0, 40), "front", generated=False)
    noisy = np.clip(
        base + rng.uniform(-0.3, 0.3, base.shape).astype(np.float32), 0, 1)
    gen = view(noisy, (30, 70), "gen", generated=True)
    projections = [src, gen]
    gen_before = np.array(projections[1]["rgba"], copy=True)

    stats = texturing.equalize_projection_tone(
        projections, positions_texture=positions, source_index=0)

    refused_rows = [row for row in stats["views"] if not row.get("applied")]
    for row in refused_rows:
        assert "written_texels" not in row
    if not any(row.get("applied") for row in stats["views"]):
        assert np.array_equal(gen_before, np.asarray(projections[1]["rgba"]))


def test_blend_projections_handoff_ledger_attributes_pairs() -> None:
    """The handoff ledger names the owner pairs, separates luminance from
    chroma steps, and reports co-witnessing."""
    shape = (64, 64)
    lum_step = np.full((*shape, 4), 0.5, dtype=np.float32)
    lum_step[:, :, 3] = 1.0
    brighter = lum_step.copy()
    brighter[:, :, :3] = 0.62  # equal-channel (pure luminance) step
    chroma = lum_step.copy()
    chroma[:, :, 0] = 0.62  # red up, blue down: mostly chroma
    chroma[:, :, 2] = 0.38

    def weights(cols):
        w = np.zeros(shape, dtype=np.float32)
        w[:, cols[0]:cols[1]] = 0.9
        return w

    a = {"rgba": lum_step, "weight": weights((0, 36)), "label": "left"}
    b = {"rgba": brighter, "weight": weights((28, 64)), "label": "right"}
    blended = texturing.blend_projections(
        [a, b], atlas_shape=shape, feather_texels=2.0)
    pairs = blended["handoff_seams"]["pairs"]
    assert len(pairs) == 1
    assert sorted(pairs[0]["views"]) == ["left", "right"]
    assert pairs[0]["lum_share_p50"] > 0.9
    assert pairs[0]["co_witnessed_frac"] > 0.9

    c = {"rgba": chroma, "weight": weights((28, 64)), "label": "right"}
    blended = texturing.blend_projections(
        [a, c], atlas_shape=shape, feather_texels=2.0)
    assert blended["handoff_seams"]["pairs"][0]["lum_share_p50"] < 0.5


# ---------------------------------------------------------------------------
# photo sovereignty for generated references (fresh-draw car decomposition,
# /tmp/gfix2: the composition math itself regressed source-pose fidelity)
# ---------------------------------------------------------------------------

def test_protect_observed_texels_absolute_mode_zeroes_under_any_evidence() -> None:
    """Absolute mode: generated weight is zero wherever ANY real view holds
    positive weight — including the sub-floor grazing band the ramp used to
    hand to synthesis at up to 30x the photo's weight (measured 39 dE mean
    contamination over that band on the fresh-draw car)."""
    real_weight = np.array([[0.8, 0.25], [0.005, 0.0]], np.float32)
    generated_weight = np.full((2, 2), 0.6, np.float32)
    real = {"label": "source", "weight": real_weight.copy()}
    generated = {"label": "back", "generated": True,
                 "weight": generated_weight.copy()}

    stats = texturing.protect_observed_texels(
        [real, generated], mode="absolute")

    out = np.asarray(generated["weight"], np.float32)
    assert out[0, 0] == 0.0                # strong real evidence
    assert out[0, 1] == 0.0                # credible real evidence
    assert out[1, 0] == 0.0                # SUB-FLOOR real evidence: still zero
    assert out[1, 1] == pytest.approx(0.6)  # no real evidence: intact
    assert np.array_equal(np.asarray(real["weight"]), real_weight)
    assert stats["mode"] == "absolute"
    assert stats["protected_texels"] == 3
    assert stats["zeroed_by_view"]["back"] == 3
    # fail-closed structure unchanged: no generated views -> no-op
    assert texturing.protect_observed_texels(
        [real], mode="absolute")["applied"] is False
    # the historical ramp stays the default for direct callers
    ramped = {"label": "back", "generated": True,
              "weight": generated_weight.copy()}
    texturing.protect_observed_texels(
        [{"label": "source", "weight": real_weight.copy()}, ramped])
    assert 0.0 < np.asarray(ramped["weight"])[1, 0] < 0.6


def test_equalize_projection_tone_ranked_gate_prefers_photo_agreement() -> None:
    """Witness-RANKED gate: a correction that improves agreement with the
    REAL photo ships even when generated-mutual agreement pays for it.

    Fixture: two generated views mistoned x1.4 but mutually CONSISTENT.
    Correcting the first toward the photo transiently breaks the
    generated-mutual agreement (the second is still mistoned when the
    first is judged) — the retired symmetric gate vetoed exactly this
    (measured on the fresh-draw car: the side view's photo-conforming
    field reverted because generated-mutual moved 0.188 -> 0.204, and
    source-pose fidelity paid the tone error)."""
    base, positions, view = _tone_case_views()
    src = view(base, (0, 40), "front", generated=False)
    gen_a = view(base * 1.4, (20, 70), "gen_a", generated=True)
    gen_b = view(base * 1.4, (60, 96), "gen_b", generated=True)
    projections = [src, gen_a, gen_b]

    stats = texturing.equalize_projection_tone(
        projections, positions_texture=positions, source_index=0)

    rows = {row["label"]: row for row in stats["views"] if "gain_log" in row}
    assert rows["gen_a"]["applied"], (
        "photo-conforming correction must not be held hostage by "
        "generated-mutual consistency")
    assert rows["gen_a"]["disagreement_after"]["real"] < \
        rows["gen_a"]["disagreement_before"]["real"] - 0.002
    # the second view then conforms through the (now corrected) chain
    assert rows["gen_b"]["applied"]
    a_rgb = np.asarray(projections[1]["rgba"])[:, :, :3]
    b_rgb = np.asarray(projections[2]["rgba"])[:, :, :3]
    assert abs(float(a_rgb[:, 30:60].mean()) - float(base[:, 30:60].mean())) < 0.02
    # near its evidence (the gen_a overlap); deep-exclusive territory
    # legitimately keeps its own level (the fade contract)
    assert abs(float(b_rgb[:, 62:74].mean()) - float(base[:, 62:74].mean())) < 0.03
    # ranked does NOT mean unranked in reverse: a correction that would
    # WORSEN the real class must still revert (fail closed) — pin by
    # asserting every applied row left the real class no worse.
    for row in stats["views"]:
        if row.get("applied") and "real" in (
                row.get("disagreement_before") or {}):
            assert row["disagreement_after"]["real"] <= \
                row["disagreement_before"]["real"] + 0.002


def test_equalize_projection_tone_photo_authority_on_subfloor_witness_band() -> None:
    """Stage 2's consensus is the photo's reading on the WHOLE
    real-witnessed band, including texels whose photo weight sits under
    the pair fit floor: grazing photo samples are smeared in detail but
    valid in regional tone (before the fix, the reference's own
    self-weight dominated the consensus there and the field pulled it
    AWAY from the photo on exactly the surface it was about to own)."""
    base, positions, view = _tone_case_views(shape=(128, 128))
    src = view(base, (0, 50), "front", generated=False)
    # photo weight: credible on [0, 25), grazing sub-floor on [25, 50)
    src["weight"][:, 25:50] = 0.01
    # regional (zero-median) mistoning so the scalar stage cannot fix it
    warped = base.copy()
    warped[:64, 20:] *= 1.4
    warped[64:, 20:] /= 1.4
    gen = view(warped, (20, 128), "gen", generated=True)
    projections = [src, gen]
    gen_before = np.asarray(projections[1]["rgba"])[:, :, :3].copy()

    stats = texturing.equalize_projection_tone(
        projections, positions_texture=positions, source_index=0)

    field_rows = [row for row in stats["views"]
                  if row.get("stage") == "local_field"]
    assert len(field_rows) == 1 and field_rows[0]["applied"]
    gen_after = np.asarray(projections[1]["rgba"])[:, :, :3]
    subfloor_band = slice(30, 46)
    before_err = float(
        np.abs(gen_before[:, subfloor_band] - base[:, subfloor_band]).mean())
    after_err = float(
        np.abs(gen_after[:, subfloor_band] - base[:, subfloor_band]).mean())
    assert after_err < 0.6 * before_err, (
        "the sub-floor witnessed band must be reconciled toward the photo")


def test_bake_generated_tone_offset_reference_zero_delta_on_witnessed() -> None:
    """END-TO-END sovereignty: adding a deliberately tone-offset GENERATED
    reference must contribute (essentially) ZERO delta on photo-witnessed
    texels — the surface the source pose renders. Pins the three coupled
    mechanisms measured on the fresh-draw car (/tmp/gfix2): absolute
    protection (no sub-floor blend replacement), the screened-Poisson
    photo-anchor pin (no tone diffusion across the ownership boundary:
    the pure-photo channel measured 14.1 dE mean from the solve alone),
    and the photo-anchored tone consensus."""
    import trimesh

    mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.5)
    size = 256

    def analytic_photo(azimuth, tone_scale):
        """Exact hole-free orthographic photo of the sphere: for a camera
        at elevation 0, the projector's cam-y axis IS world z, so shading
        by world height is a closed-form function of the pixel (a
        splatted photo leaves aliasing pinholes that bake as isolated
        interior fill specks and confound the sovereignty assertion).
        The shading range is deliberately SHALLOW (0.54-0.90): a strongly
        dark bottom reads as a dark-material mass to the film-band
        machinery, whose gradient repaint then legitimately writes
        photo-derived content — a different mechanism than the one under
        test."""
        half_extent = texturing.canonical_ortho_half_extent(
            mesh, azimuth_deg=azimuth, elevation_deg=0.0, border_ratio=0.15)
        scale = 0.5 * size / half_extent
        ys, xs = np.mgrid[0:size, 0:size].astype(np.float32)
        x_cam = (xs - size / 2.0 + 0.5) / scale
        y_cam = (size / 2.0 - ys - 0.5) / scale
        inside = x_cam**2 + y_cam**2 <= (0.5 * 0.995) ** 2
        shade = np.clip(0.72 + 0.18 * y_cam / 0.5, 0.0, 1.0)
        photo = np.zeros((size, size, 4), dtype=np.uint8)
        for channel, level in enumerate((200.0, 70.0, 60.0)):
            photo[:, :, channel] = np.where(
                inside,
                np.clip(level * shade * tone_scale, 0, 255), 0
            ).astype(np.uint8)
        photo[:, :, 3] = np.where(inside, 255, 0).astype(np.uint8)
        return Image.fromarray(photo, mode="RGBA")

    source_view = {
        "rgba": analytic_photo(0.0, 1.0), "azimuth_deg": 0.0,
        "elevation_deg": 0.0, "label": "front", "role": "source"}
    # the reference reads the shared surface 35% brighter: a deliberate
    # tone offset that MUST stay on the reference's side of the boundary
    reference_view = {
        "rgba": analytic_photo(100.0, 1.35), "azimuth_deg": 100.0,
        "elevation_deg": 0.0, "label": "gen_side", "role": "reference",
        "generated": True}

    bake_kwargs = dict(
        texture_resolution=size,
        texture_completion="none",
        projection_model="orthographic",
        source_pose_override=(0.0, 0.0),
        fill_detail_gain=0.0,
    )
    # Record each bake's own fill set (the floor stage receives it as
    # `synthesized_mask`): chart-edge texels the photo never painted are
    # FILL in both bakes, and fill tone legitimately differs when the
    # observed set changes — the sovereignty contract covers only texels
    # the photo actually painted.
    fill_masks = []
    real_floor = texturing.enforce_fill_luminance_floor

    def recording_floor(colors, **kw):
        fill_masks.append(np.array(kw["synthesized_mask"], copy=True))
        return real_floor(colors, **kw)

    texturing.enforce_fill_luminance_floor = recording_floor
    try:
        _, baseline_stats = texturing.bake_projection_texture(
            mesh, observed_views=[dict(source_view)], **bake_kwargs)
        _, candidate_stats = texturing.bake_projection_texture(
            mesh, observed_views=[dict(source_view), dict(reference_view)],
            **bake_kwargs)
    finally:
        texturing.enforce_fill_luminance_floor = real_floor

    baseline = np.asarray(
        baseline_stats["texture_image"].convert("RGB"), np.float32)[::-1]
    candidate = np.asarray(
        candidate_stats["texture_image"].convert("RGB"), np.float32)[::-1]

    # witnessed interior: the sphere is convex, so facing IS visibility;
    # stand clear of the coverage-edge feather band (facing 0.35 ~ 10+
    # texels inside the 0.2 cutoff at this resolution).
    from abstract3d.backends.triposr_runtime import (
        _tripo_camera_position, _tripo_make_texture_atlas,
        _tripo_rasterize_normal_atlas, _tripo_rasterize_position_atlas,
        _tripo_texture_padding)

    atlas = _tripo_make_texture_atlas(
        mesh, texture_resolution=size,
        texture_padding=_tripo_texture_padding(size))
    raster_kwargs = dict(
        atlas_vmapping=atlas["vmapping"], atlas_indices=atlas["indices"],
        atlas_uvs=atlas["uvs"], texture_resolution=size, texture_padding=0)
    positions = np.asarray(_tripo_rasterize_position_atlas(
        mesh, **raster_kwargs))
    normals = np.asarray(_tripo_rasterize_normal_atlas(
        mesh, **raster_kwargs))[:, :, :3]
    surface = positions[:, :, 3] > 0
    unit = normals / np.maximum(
        np.linalg.norm(normals, axis=2, keepdims=True), 1e-8)
    eye = _tripo_camera_position(
        azimuth_deg=0.0, elevation_deg=0.0, camera_distance=3.0)
    facing = unit @ (eye / np.linalg.norm(eye)).astype(np.float32)
    witnessed_interior = (
        surface & (facing > 0.35) & ~fill_masks[0] & ~fill_masks[1])
    assert int(witnessed_interior.sum()) > 2000

    delta = np.abs(candidate - baseline)[witnessed_interior]
    assert float(np.percentile(delta, 99)) <= 2.0, (
        "adding a tone-offset generated reference changed photo-witnessed "
        f"interior texels (p99 {np.percentile(delta, 99):.2f}/255)")
    assert float(delta.mean()) <= 0.5


# ---------------------------------------------------------------------------
# synthesized-texel luminance floor (dark fill fragment sweep)
# ---------------------------------------------------------------------------

def test_enforce_fill_luminance_floor_lifts_pockets_keeps_lines_and_dark_regions() -> None:
    size = 220
    positions = np.zeros((size, size, 4), dtype=np.float32)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    positions[:, :, 0] = xx / 50.0
    positions[:, :, 1] = yy / 50.0
    positions[:, :, 3] = 1.0
    surface = np.ones((size, size), dtype=bool)
    observed = np.zeros((size, size), dtype=bool)
    observed[:, :70] = True

    rgba = np.zeros((size, size, 4), dtype=np.float32)
    rgba[:, :, :3] = 0.6
    rgba[:, :, 3] = 1.0
    rgba[:, 180:, :3] = 0.06          # uniformly dark BAND (legit dark region)
    rgba[80:100, 120:140, :3] = 0.04  # feature-dark pocket in bright fill
    rgba[40:41, 100:160, :3] = 0.08   # thin dark line (panel joint analog)

    out, stats = texturing.enforce_fill_luminance_floor(
        rgba,
        positions_texture=positions,
        surface_mask=surface,
        synthesized_mask=surface & ~observed,
    )
    assert stats["applied"]
    # pocket lifted toward context
    assert float(out[85:95, 125:135, :3].mean()) > 0.15
    # thin line survives: compressed toward the floor but still clearly the
    # darkest local structure (the compression guarantee, not verbatim)
    line_mean = float(out[40, 110:150, :3].mean())
    assert line_mean < 0.45
    assert line_mean < float(out[50, 110:150, :3].mean()) - 0.1
    # interior of the uniformly dark band keeps its darkness (context is dark)
    assert float(out[60:160, 205:, :3].mean()) < 0.12
    # observed texels are never touched
    assert np.array_equal(out[observed], rgba[observed])


def test_enforce_fill_luminance_floor_spares_mirror_features_and_opposite_sheets() -> None:
    """Two critic-measured regression scenarios, now structural guarantees:

    1. pupil-analog: a dark feature written by MIRROR completion carries
       evidence (its observed twin) and is outside `synthesized_mask` at
       the call site — it must ship bit-identical no matter how bright its
       surroundings are.
    2. rear-hair sheet: fill on a dark BACK sheet millimeters behind a
       bright front sheet (opposite normals) must be judged against its
       own sheet only — no lift toward the front sheet's brightness.
    """
    size = 200
    positions = np.zeros((size, size, 4), dtype=np.float32)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    # left half: front sheet (z=0, normal +x); right half: back sheet
    # (z=-0.05), SAME world xy footprint, normal -x
    half = size // 2
    positions[:, :, 0] = np.where(xx < half, 0.0, -0.05)
    positions[:, :, 1] = np.where(xx < half, xx, xx - half) / 50.0
    positions[:, :, 2] = yy / 50.0
    positions[:, :, 3] = 1.0
    normals = np.zeros((size, size, 4), dtype=np.float32)
    normals[:, :, 0] = np.where(xx < half, 1.0, -1.0)
    normals[:, :, 3] = 1.0
    surface = np.ones((size, size), dtype=bool)

    rgba = np.zeros((size, size, 4), dtype=np.float32)
    rgba[:, :half, :3] = 0.75            # front sheet: bright skin
    rgba[:, half:, :3] = 0.10            # back sheet: dark hair
    rgba[:, :, 3] = 1.0

    observed = np.zeros((size, size), dtype=bool)
    observed[:, :half] = True            # front sheet observed
    observed[::5, half:] = True          # back sheet: sparse dark donors
    mirror = np.zeros((size, size), dtype=bool)
    mirror[60:80, 20:40] = True          # mirrored pupil on the front sheet
    rgba[60:80, 20:40, :3] = 0.05        # near-black feature in bright skin

    synthesized = surface & ~observed & ~mirror
    out, stats = texturing.enforce_fill_luminance_floor(
        rgba,
        positions_texture=positions,
        surface_mask=surface,
        synthesized_mask=synthesized,
        donor_mask=observed & ~mirror,
        normals_texture=normals,
    )
    # 1. the mirrored pupil ships bit-identical
    assert np.array_equal(out[60:80, 20:40], rgba[60:80, 20:40])
    # 2. dark back-sheet fill stays dark (its own donors are dark; the
    #    bright front sheet is the OPPOSITE direction bin and excluded)
    back_fill = np.zeros((size, size), dtype=bool)
    back_fill[:, half:] = True
    back_fill &= synthesized
    assert float(out[back_fill][:, :3].mean()) < 0.13


def test_enforce_fill_luminance_floor_donor_anchor_catches_transported_darkness() -> None:
    """Transported-darkness class (measured on the starship underside):
    a synthesized pocket in a MID-DARK synthesized surrounding, whose
    OBSERVED anchors nearby are clearly brighter. The context floor alone
    can excuse it (the surrounding fill is dark too); the donor-consensus
    floor must lift it toward what the anchors justify. Control: the same
    pocket next to equally-dark anchors is a legitimate continuation and
    must keep its darkness."""
    size = 220
    positions = np.zeros((size, size, 4), dtype=np.float32)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    positions[:, :, 0] = xx / 50.0
    positions[:, :, 1] = yy / 50.0
    positions[:, :, 3] = 1.0
    surface = np.ones((size, size), dtype=bool)
    observed = np.zeros((size, size), dtype=bool)
    observed[::7, ::7] = True  # sparse anchor lattice around the pocket
    # realistic geometry: the pocket sits INSIDE fill, donors surround it
    # at a distance (no donor within the boundary feather of the pocket)
    yy2, xx2 = np.mgrid[0:size, 0:size]
    observed &= ((yy2 - 100) ** 2 + (xx2 - 100) ** 2) > 24 ** 2

    rgba = np.zeros((size, size, 4), dtype=np.float32)
    rgba[:, :, :3] = 0.17          # mid-dark synthesized wash
    rgba[:, :, 3] = 1.0
    rgba[observed] = 0.55          # anchors are much brighter
    rgba[observed, 3] = 1.0
    rgba[90:110, 90:110, :3] = 0.05  # near-black transported pocket

    out, stats = texturing.enforce_fill_luminance_floor(
        rgba,
        positions_texture=positions,
        surface_mask=surface,
        synthesized_mask=surface & ~observed,
    )
    assert stats["applied"]
    assert float(out[95:105, 95:105, :3].mean()) > 0.15, (
        "pocket contradicting bright donors must be lifted"
    )

    # control: same geometry, anchors as dark as the pocket -> no real lift
    rgba2 = rgba.copy()
    rgba2[observed] = 0.06
    rgba2[:, :, :3][~observed] = 0.055
    out2, _ = texturing.enforce_fill_luminance_floor(
        rgba2,
        positions_texture=positions,
        surface_mask=surface,
        synthesized_mask=surface & ~observed,
    )
    assert float(out2[95:105, 95:105, :3].mean()) < 0.12, (
        "continuation of uniformly dark anchored content must survive"
    )


# ---------------------------------------------------------------------------
# geometric witness confidence (stretch + concavity)
# ---------------------------------------------------------------------------

def test_projection_geometry_confidence_stretch_demotes_collapsed_mapping() -> None:
    from abstract3d.backends.triposr_runtime import _tripo_projection_geometry_confidence

    # Cylinder atlas parameterized by (theta, height), orthographic camera
    # along +z: texel->photo pitch collapses toward the silhouette rims.
    # The parameterization is deliberately ANISOTROPIC (theta pitch != height
    # pitch): healthy front-on texels must still measure stretch ~1.
    resolution = 160
    theta = np.linspace(0.06, np.pi - 0.06, resolution)
    height = np.linspace(-1.0, 1.0, resolution)
    tt, hh = np.meshgrid(theta, height)
    positions = np.stack([np.cos(tt), hh, np.sin(tt)], axis=2).astype(np.float32)
    normals = np.stack([np.cos(tt), np.zeros_like(tt), np.sin(tt)], axis=2).astype(np.float32)
    surface = np.ones((resolution, resolution), dtype=bool)
    sample_x = 80.0 * positions[:, :, 0]
    sample_y = -80.0 * positions[:, :, 1]
    facing = normals[:, :, 2]

    factor, stats = _tripo_projection_geometry_confidence(
        positions_xyz=positions, normals_xyz=normals, surface_mask=surface,
        sample_x=sample_x, sample_y=sample_y, facing=facing,
        stretch_power=1.0, concavity_ball_frac=0.02, concavity_threshold=0.35,
        concavity_facing=0.5, concavity_demote=0.25,
    )
    mid = resolution // 2
    rim = (tt < 0.25) | (tt > np.pi - 0.25)
    assert float(factor[mid, mid]) > 0.9, "front-on texels keep full confidence"
    assert float(factor[rim].mean()) < 0.35, "collapsed rim mappings are demoted"
    assert stats["stretch_p99"] is not None and stats["stretch_p99"] > 3.0


def test_projection_geometry_confidence_demotes_concave_grazing_only() -> None:
    from abstract3d.backends.triposr_runtime import _tripo_projection_geometry_confidence

    resolution = 200
    u = np.linspace(-1, 1, resolution)
    gx, gy = np.meshgrid(u, u)
    depth = 0.35 * np.exp(-gx ** 2 / 0.004)  # sharp trench, camera along +z
    positions = np.stack([gx, gy, -depth], axis=2).astype(np.float32)
    slope = np.gradient(-depth, u[1] - u[0], axis=1)
    normals = np.stack([-slope, np.zeros_like(slope), np.ones_like(slope)], axis=2)
    normals = (normals / np.linalg.norm(normals, axis=2, keepdims=True)).astype(np.float32)
    surface = np.ones((resolution, resolution), dtype=bool)

    kwargs = dict(
        surface_mask=surface,
        stretch_power=0.0,  # isolate the concavity term
        concavity_ball_frac=0.02, concavity_threshold=0.35,
        concavity_facing=0.5, concavity_demote=0.25,
    )
    factor, stats = _tripo_projection_geometry_confidence(
        positions_xyz=positions, normals_xyz=normals,
        sample_x=80.0 * positions[:, :, 0], sample_y=-80.0 * positions[:, :, 1],
        facing=normals[:, :, 2], **kwargs,
    )
    demoted = factor < 0.9
    assert stats["concave_demoted"] > 0, "trench interior must be demoted"
    assert float(np.abs(gx[demoted]).max()) < 0.06, "demotion stays inside the trench"

    # convex control: the same ridge flipped upward must not be demoted
    # (tolerance: a handful of smoothing-boundary texels at the sharp apex)
    positions_up = positions.copy()
    positions_up[:, :, 2] = depth
    normals_up = normals.copy()
    normals_up[:, :, 0] *= -1.0
    factor_up, stats_up = _tripo_projection_geometry_confidence(
        positions_xyz=positions_up, normals_xyz=normals_up,
        sample_x=80.0 * positions_up[:, :, 0], sample_y=-80.0 * positions_up[:, :, 1],
        facing=normals_up[:, :, 2], **kwargs,
    )
    assert stats_up["concave_demoted"] <= 10, "convex ridge must stay undemoted"


def _folded_plane_scene(size: int = 128):
    """Two mirror-symmetric sheets (y = +0.5 / -0.5) sharing x/z, as one
    atlas: left half = +y side, right half = -y side."""
    positions = np.zeros((size, size, 4), dtype=np.float32)
    xs, ys = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    positions[:, : size // 2, 0] = xs[:, : size // 2]
    positions[:, : size // 2, 1] = 0.5
    positions[:, : size // 2, 2] = ys[:, : size // 2]
    positions[:, size // 2 :, 0] = xs[:, : size // 2]
    positions[:, size // 2 :, 1] = -0.5
    positions[:, size // 2 :, 2] = ys[:, : size // 2]
    positions[:, :, 3] = 1.0
    on_plus = np.zeros((size, size), dtype=bool)
    on_plus[:, : size // 2] = True
    feature_r = np.sqrt(positions[:, :, 0] ** 2 + positions[:, :, 2] ** 2)
    return positions, on_plus, feature_r


def test_detect_mirror_rescue_discs_fires_on_weak_feature_empty_twin() -> None:
    """A confidently witnessed dark feature whose mirror twin is observed
    weakly AND feature-empty must yield exactly one transplant disc at the
    mirrored location; the returned disc must drive `mirror_rescue_disc`
    to restore the twin feature end to end."""
    from abstract3d.texturing import detect_mirror_rescue_discs, mirror_rescue_disc

    positions, on_plus, feature_r = _folded_plane_scene()
    colors = np.full((*on_plus.shape, 3), 0.8, dtype=np.float32)
    blob_plus = on_plus & (feature_r < 0.14)
    colors[blob_plus] = 0.05  # coherent dark feature (iris analog)

    observed = np.ones(on_plus.shape, dtype=bool)
    weight = np.where(on_plus, 0.6, 0.15).astype(np.float32)  # weak twin

    discs = detect_mirror_rescue_discs(
        positions_texture=positions,
        colors_rgb=colors,
        observed_mask=observed,
        observed_weight=weight,
    )
    assert len(discs) == 1, f"expected exactly one disc, got {discs}"
    disc = discs[0]
    assert disc["center"][1] < 0, "transplant disc must be on the weak (-y) side"
    assert abs(disc["center"][0]) < 0.1 and abs(disc["center"][2]) < 0.1, (
        "disc must center on the mirrored feature")

    rescued, stats = mirror_rescue_disc(
        colors,
        positions_texture=positions,
        center=disc["center"],
        radius=disc["radius"],
        source_mask=observed & (weight >= 0.35),
        feather_texels=1.0,
    )
    assert stats["rescued_texels"] > 0
    blob_minus = ~on_plus & (feature_r < 0.10)
    assert float(rescued[blob_minus].mean()) < 0.35, (
        "detected disc must transplant the feature onto the weak twin")


def test_detect_mirror_rescue_discs_no_fire_on_asymmetric_content() -> None:
    """Deliberately asymmetric content, confidently witnessed on BOTH
    sides, must never trigger: differing features are legitimate when both
    sides carry real evidence (twin-weakness gate)."""
    from abstract3d.texturing import detect_mirror_rescue_discs

    positions, on_plus, feature_r = _folded_plane_scene()
    colors = np.full((*on_plus.shape, 3), 0.8, dtype=np.float32)
    colors[on_plus & (feature_r < 0.14)] = 0.05  # dark blob on +y
    off_center = np.sqrt(positions[:, :, 0] ** 2 + (positions[:, :, 2] - 0.4) ** 2)
    colors[~on_plus & (off_center < 0.14)] = 0.05  # DIFFERENT blob on -y

    observed = np.ones(on_plus.shape, dtype=bool)
    weight = np.full(on_plus.shape, 0.6, dtype=np.float32)  # both confident

    discs = detect_mirror_rescue_discs(
        positions_texture=positions,
        colors_rgb=colors,
        observed_mask=observed,
        observed_weight=weight,
    )
    assert discs == [], f"confident asymmetric content must not fire: {discs}"


def test_detect_mirror_rescue_discs_no_fire_when_twin_unobserved() -> None:
    """Single-photo bakes (ship/owl lane): the far side is UNOBSERVED, and
    completing it belongs to mirror completion — the rescue must stay
    silent (twin-coverage gate)."""
    from abstract3d.texturing import detect_mirror_rescue_discs

    positions, on_plus, feature_r = _folded_plane_scene()
    colors = np.full((*on_plus.shape, 3), 0.8, dtype=np.float32)
    colors[on_plus & (feature_r < 0.14)] = 0.05

    observed = on_plus.copy()  # -y side never witnessed
    weight = np.where(on_plus, 0.6, 0.0).astype(np.float32)

    discs = detect_mirror_rescue_discs(
        positions_texture=positions,
        colors_rgb=colors,
        observed_mask=observed,
        observed_weight=weight,
    )
    assert discs == [], f"unobserved twin must not fire: {discs}"


def test_detect_mirror_rescue_discs_no_fire_when_twin_has_own_feature() -> None:
    """A weakly witnessed twin that already carries structure of comparable
    prominence at the mirrored location is NOT a smear — the
    feature-emptiness gate must refuse the transplant. (A twin whose
    structure is several times weaker than the healthy side's is
    indistinguishable from a smeared defect and remains transplant-
    eligible by design; the gate's contract is comparable prominence.)"""
    from abstract3d.texturing import detect_mirror_rescue_discs

    positions, on_plus, feature_r = _folded_plane_scene()
    colors = np.full((*on_plus.shape, 3), 0.8, dtype=np.float32)
    colors[on_plus & (feature_r < 0.14)] = 0.05
    # twin carries its own feature of comparable contrast, differently
    # shaped (half-disc): real content, weakly witnessed
    colors[~on_plus & (feature_r < 0.14) & (positions[:, :, 2] > 0.0)] = 0.05

    observed = np.ones(on_plus.shape, dtype=bool)
    weight = np.where(on_plus, 0.6, 0.15).astype(np.float32)

    discs = detect_mirror_rescue_discs(
        positions_texture=positions,
        colors_rgb=colors,
        observed_mask=observed,
        observed_weight=weight,
    )
    assert discs == [], f"feature-bearing twin must not fire: {discs}"


def test_mirror_rescue_disc_transplants_twin_feature_tone_matched() -> None:
    """A dark feature on the +y half must transplant onto a weakly-witnessed
    -y disc, tone-matched to the destination's surround, feathered, and
    strictly local (texels outside the disc stay bit-identical)."""
    from abstract3d.texturing import mirror_rescue_disc

    size = 128
    positions = np.zeros((size, size, 4), dtype=np.float32)
    xs, ys = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    # two symmetric sheets: left half of the atlas is the +y side, right
    # half the -y side, sharing x/z coordinates (a folded plane)
    positions[:, : size // 2, 0] = xs[:, : size // 2]
    positions[:, : size // 2, 1] = 0.5
    positions[:, : size // 2, 2] = ys[:, : size // 2]
    positions[:, size // 2 :, 0] = xs[:, : size // 2]
    positions[:, size // 2 :, 1] = -0.5
    positions[:, size // 2 :, 2] = ys[:, : size // 2]
    positions[:, :, 3] = 1.0

    colors = np.full((size, size, 3), 0.8, dtype=np.float32)
    # +y side carries a dark iris-like blob at (x,z)=(0,0) and a brighter
    # global tone (composition shading difference the rescue must cancel)
    on_plus = np.zeros((size, size), dtype=bool)
    on_plus[:, : size // 2] = True
    r_plus = np.sqrt(positions[:, :, 0] ** 2 + positions[:, :, 2] ** 2)
    blob_plus = on_plus & (r_plus < 0.12)
    colors[on_plus] += 0.1
    colors[blob_plus] = 0.05
    # -y side has NO feature (weak-witness smear analog: flat skin)
    baseline = colors.copy()

    rescued, stats = mirror_rescue_disc(
        colors,
        positions_texture=positions,
        center=(0.0, -0.5, 0.0),
        radius=0.3,
        feather_texels=1.0,
    )
    assert stats["rescued_texels"] > 0

    # the dark blob must now exist on the -y side at the mirrored location
    on_minus = ~on_plus
    blob_minus = on_minus & (r_plus < 0.10)
    assert float(rescued[blob_minus].mean()) < 0.35, "twin feature must transplant"

    # tone matching: the copied surround must match the destination's tone
    # (0.8), not carry the +y side's brighter 0.9
    surround_minus = on_minus & (r_plus > 0.16) & (r_plus < 0.26)
    assert abs(float(rescued[surround_minus].mean()) - 0.8) < 0.03, (
        "transplant must be tone-matched to the destination surround"
    )

    # locality: texels far outside the disc (and the whole +y side) unchanged
    far = r_plus > 0.45
    assert np.allclose(rescued[far], baseline[far], atol=1e-5)
    assert np.allclose(rescued[on_plus & (r_plus > 0.16)],
                       baseline[on_plus & (r_plus > 0.16)], atol=1e-5)


def _bbox_offset_test_mesh():
    """Elongated block whose camera-plane bbox center is displaced from the
    world origin at an oblique pose (the SHIP-03 geometry class)."""
    import trimesh

    mesh = trimesh.creation.box(extents=(1.6, 0.6, 0.3))
    mesh.apply_translation((0.25, 0.1, 0.0))  # bbox center off-origin
    return mesh


def test_projected_frame_center_px_matches_projector_convention() -> None:
    """The predicted center must equal where the projector's own sample map
    sends the mesh bbox center (exactness, both axes), and must reduce to
    the frame center when the bbox center IS the origin."""
    import trimesh

    from abstract3d.backends.triposr_runtime import (
        _tripo_camera_position,
        _tripo_look_at_matrix,
    )

    mesh = _bbox_offset_test_mesh()
    size, border = 1024, 0.15
    azimuth, elevation = 30.0, 15.0
    cx, cy = texturing.projected_frame_center_px(
        mesh, azimuth_deg=azimuth, elevation_deg=elevation,
        size=size, border_ratio=border)

    # independent computation through the projector's sample-map math
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    eye = _tripo_camera_position(
        azimuth_deg=azimuth, elevation_deg=elevation, camera_distance=3.0)
    view = _tripo_look_at_matrix(
        eye, np.zeros(3, np.float32), np.array([0, 0, 1], np.float32))
    cam = vertices @ view[:3, :3].T + view[:3, 3]
    bbox_center = 0.5 * (cam.min(axis=0) + cam.max(axis=0))
    half_extent = texturing.canonical_ortho_half_extent(
        mesh, azimuth_deg=azimuth, elevation_deg=elevation, border_ratio=border)
    ortho_scale = 0.5 * size / half_extent
    expected_x = ortho_scale * bbox_center[0] + size / 2.0
    expected_y = -ortho_scale * bbox_center[1] + size / 2.0
    assert abs(cx - expected_x) < 1e-3
    assert abs(cy - expected_y) < 1e-3
    # the offset is real at this pose (the defect class exists)
    assert abs(cx - size / 2.0) > 5.0

    # centered geometry at its canonical front: no offset (old behavior)
    centered = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    fx, fy = texturing.projected_frame_center_px(
        centered, azimuth_deg=0.0, elevation_deg=0.0, size=size,
        border_ratio=border)
    assert abs(fx - size / 2.0) < 1e-3
    assert abs(fy - size / 2.0) < 1e-3


def test_recenter_to_canonical_frame_center_px_places_bbox_center() -> None:
    """center_px must place the photo's alpha-bbox center at the requested
    pixel; default must stay the exact legacy centering."""
    photo = np.zeros((200, 300, 4), dtype=np.uint8)
    photo[60:140, 90:210] = (200, 120, 80, 255)  # bbox 80x120
    image = Image.fromarray(photo, mode="RGBA")

    legacy = texturing.recenter_to_canonical_frame(image, size=256, border_ratio=0.15)
    legacy_alpha = np.asarray(legacy)[:, :, 3] > 12
    rows = np.nonzero(legacy_alpha.any(axis=1))[0]
    cols = np.nonzero(legacy_alpha.any(axis=0))[0]
    legacy_center = (0.5 * (cols[0] + cols[-1]), 0.5 * (rows[0] + rows[-1]))
    assert abs(legacy_center[0] - 127.5) <= 1.0
    assert abs(legacy_center[1] - 127.5) <= 1.0

    shifted = texturing.recenter_to_canonical_frame(
        image, size=256, border_ratio=0.15, center_px=(140.0, 118.0))
    shifted_alpha = np.asarray(shifted)[:, :, 3] > 12
    rows = np.nonzero(shifted_alpha.any(axis=1))[0]
    cols = np.nonzero(shifted_alpha.any(axis=0))[0]
    shifted_center = (0.5 * (cols[0] + cols[-1]), 0.5 * (rows[0] + rows[-1]))
    # integer paste + bbox parity allow up to ~2 px of quantization
    assert abs(shifted_center[0] - 140.0) <= 2.0
    assert abs(shifted_center[1] - 118.0) <= 2.0
    # scale rule unchanged: larger bbox side fills (1-border)*size
    assert abs((cols[-1] - cols[0] + 1) - round(0.85 * 256)) <= 2


def test_bake_projection_frame_registration_recovers_offset_content() -> None:
    """End-to-end: bake an off-origin block from a synthetic photo rendered
    in the PROJECTOR's own frame. With mesh_bbox_center registration the
    observed texels must reproduce the photo's two-tone split; the legacy
    bbox-center recenter must show a measurably larger error (this is the
    SHIP-03 defect in miniature)."""
    import trimesh

    from abstract3d.backends.triposr_runtime import (
        _tripo_camera_position,
        _tripo_look_at_matrix,
        _tripo_make_texture_atlas,
        _tripo_rasterize_position_atlas,
    )

    from abstract3d.backends.triposr_runtime import _tripo_rasterize_normal_atlas

    mesh = _bbox_offset_test_mesh()
    azimuth, elevation = 30.0, 15.0
    size = 512

    # ground truth: two-tone split along world x (dark bow, bright stern)
    resolution = 256
    atlas = _tripo_make_texture_atlas(
        mesh, texture_resolution=resolution, texture_padding=0)
    raster_kwargs = dict(
        atlas_vmapping=atlas["vmapping"], atlas_indices=atlas["indices"],
        atlas_uvs=atlas["uvs"], texture_resolution=resolution,
        texture_padding=0)
    positions = np.asarray(
        _tripo_rasterize_position_atlas(mesh, **raster_kwargs), dtype=np.float32)
    normals = np.asarray(
        _tripo_rasterize_normal_atlas(mesh, **raster_kwargs), dtype=np.float32)
    surface = positions[:, :, 3] > 0
    dark_side = positions[:, :, 0] > 0.25  # split at the block's center

    # synthetic photo rendered with the projector's exact ortho math
    half_extent = texturing.canonical_ortho_half_extent(
        mesh, azimuth_deg=azimuth, elevation_deg=elevation, border_ratio=0.15)
    eye = _tripo_camera_position(
        azimuth_deg=azimuth, elevation_deg=elevation, camera_distance=3.0)
    view = _tripo_look_at_matrix(
        eye, np.zeros(3, np.float32), np.array([0, 0, 1], np.float32))
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    cam = vertices @ view[:3, :3].T + view[:3, 3]
    scale = 0.5 * size / half_extent
    px = scale * cam[:, 0] + size / 2.0
    py = -scale * cam[:, 1] + size / 2.0
    photo = np.zeros((size, size, 4), dtype=np.uint8)
    # splat surface points; the box is convex, so back-face culling IS the
    # exact visibility test (no z-fighting pixels left to a far face)
    samples, face_idx = trimesh.sample.sample_surface(mesh, 120000, seed=7)
    face_normals = np.asarray(mesh.face_normals, np.float32)[np.asarray(face_idx)]
    view_dir = eye / np.linalg.norm(eye)
    front = face_normals @ view_dir.astype(np.float32) > 0.05
    samples = np.asarray(samples, np.float32)[front]
    cams = samples @ view[:3, :3].T + view[:3, 3]
    sx = np.clip(scale * cams[:, 0] + size / 2.0, 0, size - 1).astype(int)
    sy = np.clip(-scale * cams[:, 1] + size / 2.0, 0, size - 1).astype(int)
    tone = np.where(samples[:, 0] > 0.25, 40, 220).astype(np.uint8)
    photo[sy, sx, 0] = tone
    photo[sy, sx, 1] = tone
    photo[sy, sx, 2] = tone
    photo[sy, sx, 3] = 255
    from scipy.ndimage import grey_closing

    for channel in range(4):
        photo[:, :, channel] = grey_closing(photo[:, :, channel], size=5)
    photo_image = Image.fromarray(photo, mode="RGBA")

    textured, stats = texturing.bake_projection_texture(
        mesh,
        observed_views=[{
            "rgba": photo_image, "azimuth_deg": 0.0, "elevation_deg": 0.0,
            "label": "front", "role": "source"}],
        texture_resolution=resolution,
        texture_completion="none",
        projection_model="orthographic",
        source_pose_override=(azimuth, elevation),
        fill_detail_gain=0.0,
    )
    assert stats["source_registration"]["method"] == "mesh_bbox_center"
    assert abs(stats["source_registration"]["frame_center_dx_px"]) > 4.0

    baked = np.asarray(stats["texture_image"].convert("RGB"), dtype=np.float32)
    baked = baked[::-1]  # texture image is v-flipped relative to atlas rows
    lum = baked.mean(axis=2)
    # score only texels the source view could actually witness (camera-facing
    # under the projector's own facing threshold); hidden texels are fill.
    eye = _tripo_camera_position(
        azimuth_deg=azimuth, elevation_deg=elevation, camera_distance=3.0)
    to_camera = eye[None, None, :] - positions[:, :, :3]
    to_camera /= np.maximum(np.linalg.norm(to_camera, axis=2, keepdims=True), 1e-8)
    normal_vectors = normals[:, :, :3]
    normal_vectors = normal_vectors / np.maximum(
        np.linalg.norm(normal_vectors, axis=2, keepdims=True), 1e-8)
    facing = (normal_vectors * to_camera).sum(axis=2)
    witnessed = surface & (facing > 0.35)
    bright_err = abs(float(lum[witnessed & ~dark_side].mean()) - 220.0)
    dark_err = abs(float(lum[witnessed & dark_side].mean()) - 40.0)
    assert bright_err < 45.0, f"bright half off by {bright_err}"
    assert dark_err < 45.0, f"dark half off by {dark_err}"


def test_synthesize_fill_detail_amplitude_floor_breaks_quiet_donor_plateaus() -> None:
    """Fill anchored by artificially quiet (grazing-smeared) donors must not
    ship as a literal flat plateau: the observed population's low-quantile
    raw-residual amplitude floors the transfer (SHIP-03 follow-up: an 11k
    texel flat cell tripped texel.facet_cellular at 2048)."""
    rng = np.random.default_rng(11)
    positions, normals, observed = _flat_strip_state(shape=(48, 96), observed_cols=32)
    height, width = observed.shape
    colors = np.full((height, width, 4), 0.5, dtype=np.float32)
    # observed: top three quarters textured, bottom quarter nearly flat
    # (the grazing-smeared donor analog)
    textured_rows = slice(0, 36)
    flat_rows = slice(36, 48)
    obs_idx = np.zeros((height, width), dtype=bool)
    obs_idx[:, :32] = True
    noise = 0.10 * rng.standard_normal((height, width, 1)).astype(np.float32)
    colors[:, :, :3] = np.clip(0.5 + np.where(obs_idx[:, :, None], noise, 0.0), 0.0, 1.0)
    colors[flat_rows, :32, :3] = 0.5  # quiet donors
    colors[:, :, 3] = 1.0

    out = texturing.synthesize_fill_detail(
        colors,
        positions_texture=positions,
        normals_texture=normals,
        observed_mask=obs_idx,
        gain=1.0,
    )

    # fill rows adjacent to the QUIET donors, deep past the seam feather
    quiet_fill = np.zeros((height, width), dtype=bool)
    quiet_fill[flat_rows, 48:] = True
    sigma_quiet = float(out[quiet_fill][:, 0].std())
    # fill next to textured donors, for scale
    textured_fill = np.zeros((height, width), dtype=bool)
    textured_fill[textured_rows, 48:] = True
    sigma_textured = float(out[textured_fill][:, 0].std())
    assert sigma_quiet > 0.01, (
        f"quiet-donor fill stayed flat (sigma {sigma_quiet:.4f}); the "
        "amplitude floor must impose the observed population's minimum "
        "stochastic level")
    # ... while remaining at or below the textured-donor level (no granite)
    assert sigma_quiet <= sigma_textured * 1.05


# ---------------------------------------------------------------------------
# commit_trace_deposits (cycle-3: FACE-03/04/05 chip/dash class)
# ---------------------------------------------------------------------------

def _deposit_scene(size: int = 160):
    """One flat sheet with two projections: view A (confident, bright skin
    everywhere it images) and view B (trace weight everywhere)."""
    positions = np.zeros((size, size, 4), dtype=np.float32)
    xs, ys = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    positions[:, :, 0] = xs
    positions[:, :, 2] = ys
    positions[:, :, 3] = 1.0
    skin = np.full((size, size, 3), 0.8, dtype=np.float32)
    return positions, skin


def _projection(weight, rgb):
    rgba = np.concatenate([rgb, (weight > 0)[:, :, None].astype(np.float32)], axis=2)
    return {"weight": weight.astype(np.float32), "rgba": rgba.astype(np.float32)}


def test_commit_trace_deposits_retones_consensus_contradicted_chip() -> None:
    """A small dark blob won at TRACE weight, on a surround every witnessing
    view reads as bright skin, must be retoned to the ring tone."""
    from abstract3d.texturing import commit_trace_deposits

    positions, skin = _deposit_scene()
    size = positions.shape[0]
    blob = np.zeros((size, size), dtype=bool)
    blob[78:82, 78:82] = True

    colors = np.zeros((size, size, 4), dtype=np.float32)
    colors[:, :, :3] = skin
    colors[blob, :3] = 0.3  # the deposit (displaced dark content)
    colors[:, :, 3] = 1.0

    weight_a = np.full((size, size), 0.5, dtype=np.float32)
    weight_a[blob] = 0.0            # the confident view never imaged the blob
    rgb_a = skin.copy()
    weight_b = np.full((size, size), 0.05, dtype=np.float32)  # trace winner
    rgb_b = skin.copy()
    rgb_b[blob] = 0.3

    stats: dict = {}
    out = commit_trace_deposits(
        colors,
        positions_texture=positions,
        observed_mask=np.ones((size, size), dtype=bool),
        projections=[_projection(weight_a, rgb_a), _projection(weight_b, rgb_b)],
        stats_out=stats,
    )
    assert stats["applied"], "trace deposit on bright consensus must commit"
    assert float(out[blob, :3].mean()) > 0.7, (
        f"deposit must be retoned to ring tone, got {out[blob, :3].mean():.3f}")


def test_commit_trace_deposits_never_touches_confident_content() -> None:
    """The same dark blob CONFIDENTLY witnessed (a mole, a nostril) must
    never be demoted — content any view confidently witnesses stays."""
    from abstract3d.texturing import commit_trace_deposits

    positions, skin = _deposit_scene()
    size = positions.shape[0]
    blob = np.zeros((size, size), dtype=bool)
    blob[78:82, 78:82] = True

    colors = np.zeros((size, size, 4), dtype=np.float32)
    colors[:, :, :3] = skin
    colors[blob, :3] = 0.3
    colors[:, :, 3] = 1.0

    weight_a = np.full((size, size), 0.5, dtype=np.float32)  # sees the mole
    rgb_a = skin.copy()
    rgb_a[blob] = 0.3
    weight_b = np.full((size, size), 0.05, dtype=np.float32)
    rgb_b = skin.copy()

    out = commit_trace_deposits(
        colors,
        positions_texture=positions,
        observed_mask=np.ones((size, size), dtype=bool),
        projections=[_projection(weight_a, rgb_a), _projection(weight_b, rgb_b)],
    )
    assert float(out[blob, :3].mean()) < 0.35, "confident content was demoted"


def test_commit_trace_deposits_noop_single_view() -> None:
    """With one witness the ring consensus is the winner's own photo —
    vacuous by construction; single-view bakes must be bit-identical."""
    from abstract3d.texturing import commit_trace_deposits

    positions, skin = _deposit_scene()
    size = positions.shape[0]
    colors = np.zeros((size, size, 4), dtype=np.float32)
    colors[:, :, :3] = skin
    colors[70:74, 70:74, :3] = 0.3
    colors[:, :, 3] = 1.0
    weight = np.full((size, size), 0.05, dtype=np.float32)

    out = commit_trace_deposits(
        colors,
        positions_texture=positions,
        observed_mask=np.ones((size, size), dtype=bool),
        projections=[_projection(weight, skin.copy())],
    )
    assert np.array_equal(out, colors), "single-view input must pass through"


def test_commit_trace_deposits_bright_near_feature_protected() -> None:
    """A BRIGHT trace deposit beside a confident strong-contrast feature
    core (lash line analog) is ambiguous with the feature's own fringe and
    must be refused; the same deposit far from any core commits."""
    from abstract3d.texturing import commit_trace_deposits

    positions, skin = _deposit_scene()
    size = positions.shape[0]
    feature = np.zeros((size, size), dtype=bool)
    feature[60:63, 40:120] = True      # confident dark line
    near = np.zeros((size, size), dtype=bool)
    near[65:69, 76:80] = True          # bright fleck beside the line
    far = np.zeros((size, size), dtype=bool)
    far[120:124, 76:80] = True         # same fleck, far from the line

    colors = np.zeros((size, size, 4), dtype=np.float32)
    colors[:, :, :3] = skin
    colors[feature, :3] = 0.1
    colors[near, :3] = 0.97
    colors[far, :3] = 0.97
    colors[:, :, 3] = 1.0

    weight_a = np.full((size, size), 0.5, dtype=np.float32)
    weight_a[near | far] = 0.0
    rgb_a = skin.copy()
    rgb_a[feature] = 0.1
    weight_b = np.full((size, size), 0.05, dtype=np.float32)
    rgb_b = skin.copy()
    rgb_b[feature] = 0.1
    rgb_b[near | far] = 0.97

    out = commit_trace_deposits(
        colors,
        positions_texture=positions,
        observed_mask=np.ones((size, size), dtype=bool),
        projections=[_projection(weight_a, rgb_a), _projection(weight_b, rgb_b)],
    )
    assert float(out[near, :3].mean()) > 0.9, (
        "bright deposit inside the feature halo must be left alone")
    assert float(out[far, :3].mean()) < 0.9, (
        "the same bright deposit far from any feature core must commit")


def test_commit_trace_deposits_rim_feather_closes_border_mixtures() -> None:
    """FACE-22 (cycle 6): the deposit's antialiased border mixtures sit
    below the deviation bar, so the commit retones the interior and the
    rim keeps the old darker tone — a closed line-art outline. The rim
    feather must pull darker-than-ring rim mixtures toward the ring tone;
    the one-sidedness must leave brighter-than-ring rim texels alone."""
    from abstract3d.texturing import commit_trace_deposits

    positions, skin = _deposit_scene()
    size = positions.shape[0]
    blob = np.zeros((size, size), dtype=bool)
    blob[78:81, 72:88] = True          # dash-shaped deposit
    dark_rim = np.zeros((size, size), dtype=bool)
    dark_rim[77, 72:88] = True         # faint dark mixture band (trace)
    confident_rim = np.zeros((size, size), dtype=bool)
    confident_rim[81, 72:88] = True    # same mixture, CONFIDENT witness

    # 0.70 mixtures: measured sub-deviation against the blob-contaminated
    # voxel ball (dev p50 0.014, max 0.043 < 0.045 bar) and above the
    # residue class (0.70 > 0.82 x bright_median = 0.656).
    colors = np.zeros((size, size, 4), dtype=np.float32)
    colors[:, :, :3] = skin
    colors[blob, :3] = 0.3
    colors[dark_rim, :3] = 0.70
    colors[confident_rim, :3] = 0.70
    colors[:, :, 3] = 1.0

    weight_a = np.full((size, size), 0.5, dtype=np.float32)
    weight_a[blob | dark_rim] = 0.0    # confident view owns confident_rim
    rgb_a = skin.copy()
    rgb_a[confident_rim] = 0.70
    weight_b = np.full((size, size), 0.05, dtype=np.float32)
    rgb_b = skin.copy()
    rgb_b[blob] = 0.3
    rgb_b[dark_rim] = 0.70

    stats: dict = {}
    out = commit_trace_deposits(
        colors,
        positions_texture=positions,
        observed_mask=np.ones((size, size), dtype=bool),
        projections=[_projection(weight_a, rgb_a), _projection(weight_b, rgb_b)],
        stats_out=stats,
    )
    assert stats["applied"]
    assert stats.get("rim_feathered", 0) > 0
    assert float(out[blob, :3].mean()) > 0.7, "interior must retone"
    assert float(out[dark_rim, :3].mean()) > 0.77, (
        f"trace rim mixtures must lift toward ring tone, got "
        f"{out[dark_rim, :3].mean():.3f}")
    assert float(np.abs(out[confident_rim, :3] - 0.70).max()) < 1e-5, (
        "witness gate: confidently-witnessed rim texels must stay verbatim")


def test_tone_match_completion_components_scopes_and_matches() -> None:
    """FACE-22 (cycle 6): mirror-completion copies tone-match to their
    destination ring ONLY when the copy is pure-bright and the ring is
    bright skin; mixed-material copies and dark-ring components stay
    verbatim (rescaling them re-classifies their own dark micro-content
    and mints dark_debris islands — measured in the cycle-6 ladder)."""
    from abstract3d.texturing import tone_match_completion_components

    size = 128
    surface = np.ones((size, size), dtype=bool)
    observed = np.ones((size, size), dtype=bool)
    colors = np.full((size, size, 3), 0.72, dtype=np.float32)  # observed skin

    add = np.zeros((size, size), dtype=bool)
    add[20:32, 20:32] = True           # A: pure-bright copy, +offset
    add[60:72, 60:72] = True           # B: mixed copy (dark micro-content)
    add[100:112, 20:32] = True         # C: bright copy on a DARK ring
    observed &= ~add
    colors[96:116, 16:36] = 0.2        # dark destination region around C
    colors[100:112, 20:32] = 0.72      # (values under add are ignored)

    fill = np.zeros((size, size, 3), dtype=np.float32)
    fill[20:32, 20:32] = 0.88
    fill[60:72, 60:72] = 0.88
    fill[64:66, 64:66] = 0.15          # the mixed copy's dark content
    fill[100:112, 20:32] = 0.88

    out, matched = tone_match_completion_components(
        fill,
        add_mask=add,
        colors_rgb=colors,
        observed_mask=observed,
        surface_mask=surface,
    )
    assert matched == 144, f"only the pure-bright component matches, {matched}"
    # A moved toward the ring tone (0.72), detail-preserving scale
    assert abs(float(out[26, 26].mean()) - 0.72) < 0.04
    # B (mixed) and C (dark ring) stay verbatim
    assert float(np.abs(out[60:72, 60:72] - fill[60:72, 60:72]).max()) < 1e-6
    assert float(np.abs(out[100:112, 20:32] - 0.88).max()) < 1e-6


def test_commit_trace_deposits_rejects_dark_frontier_sliver() -> None:
    """A dark trace sliver CONNECTED to a large dark mass (hair frontier
    whose dark side is fill, so the direct-only ring sees just the bright
    side) must be refused by the isolation gate: committing it paints pale
    streaks into the mass (measured at +90 el10 on the face proof)."""
    from abstract3d.texturing import commit_trace_deposits

    positions, skin = _deposit_scene()
    size = positions.shape[0]
    mass = np.zeros((size, size), dtype=bool)
    mass[40:120, 0:40] = True          # large dark mass, UNOBSERVED (fill)
    sliver = np.zeros((size, size), dtype=bool)
    sliver[78:81, 40:46] = True        # direct trace sliver at its frontier

    colors = np.zeros((size, size, 4), dtype=np.float32)
    colors[:, :, :3] = skin
    colors[mass, :3] = 0.25
    colors[sliver, :3] = 0.25
    colors[:, :, 3] = 1.0

    observed = np.ones((size, size), dtype=bool)
    observed[mass] = False             # the mass is fill, not witnessed

    weight_a = np.full((size, size), 0.5, dtype=np.float32)
    weight_a[sliver | mass] = 0.0
    rgb_a = skin.copy()
    weight_b = np.full((size, size), 0.05, dtype=np.float32)
    weight_b[mass] = 0.0
    rgb_b = skin.copy()
    rgb_b[sliver] = 0.25

    out = commit_trace_deposits(
        colors,
        positions_texture=positions,
        observed_mask=observed,
        projections=[_projection(weight_a, rgb_a), _projection(weight_b, rgb_b)],
    )
    assert float(out[sliver, :3].mean()) < 0.35, (
        "frontier sliver of a connected dark mass must not be retoned")


# ---------------------------------------------------------------------------
# strand comb (cycle-3: FACE-09 rear-hair combed low-contrast fill)
# ---------------------------------------------------------------------------

def _hair_fill_scene(size: int = 192):
    """Flat sheet: left third OBSERVED — top half bright skin analog (the
    subject's bright material, anchoring the dark-ratio split), bottom half
    dark hair with horizontal streaks (strongly anisotropic donors) — and
    the right two thirds dark fill carrying membrane-style tone blotches."""
    positions = np.zeros((size, size, 4), dtype=np.float32)
    xs, ys = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    positions[:, :, 0] = xs
    positions[:, :, 2] = ys
    positions[:, :, 3] = 1.0
    normals = np.zeros((size, size, 4), dtype=np.float32)
    normals[:, :, 1] = 1.0
    observed = np.zeros((size, size), dtype=bool)
    observed[:, : size // 3] = True

    colors = np.zeros((size, size, 4), dtype=np.float32)
    rows = np.arange(size)[:, None].astype(np.float32)
    streaks = 0.16 + 0.10 * (0.5 + 0.5 * np.sin(rows * 2.2))
    colors[:, :, :3] = 0.2
    for c in range(3):
        colors[:, : size // 3, c] = streaks[:, :1]
    colors[: size // 4, : size // 3, :3] = 0.8  # bright skin analog
    # fill base carries membrane-style tone blotches (the leopard source)
    blotch = 0.05 * np.sin(xs * 9.0) * np.sin(ys * 7.0)
    for c in range(3):
        colors[:, size // 3 :, c] = np.clip(0.2 + blotch[:, size // 3 :], 0, 1)
    colors[observed, 3] = 1.0
    return positions, normals, observed, colors


def test_multigrid_orientation_field_propagates_coherently() -> None:
    """Anchors with one orientation on the observed side must propagate a
    coherent (|cos| ~ 1) direction to queries deep inside the fill domain."""
    from abstract3d.texturing import _multigrid_orientation_field

    rng = np.random.default_rng(5)
    anchors = np.zeros((400, 3), dtype=np.float64)
    anchors[:, 0] = rng.uniform(-1.0, -0.4, 400)   # left band
    anchors[:, 2] = rng.uniform(-1.0, 1.0, 400)
    direction = np.tile([0.0, 0.0, 1.0], (400, 1))  # combed along +z
    # queries tile the CONNECTED fill domain from the anchor frontier to
    # the deep right side (the propagation medium is the surface itself;
    # a detached query island has no path to the anchors by design)
    queries = np.zeros((1200, 3), dtype=np.float64)
    queries[:, 0] = rng.uniform(-0.4, 1.0, 1200)
    queries[:, 2] = rng.uniform(-1.0, 1.0, 1200)
    normals = np.tile([0.0, 1.0, 0.0], (1200, 1))

    directions, ok = _multigrid_orientation_field(
        queries, anchors, direction, np.ones(400), normals, scale=2.8)
    deep = queries[:, 0] > 0.4
    assert (ok & deep).sum() > 0.9 * deep.sum()
    alignment = np.abs(directions[ok & deep] @ np.array([0.0, 0.0, 1.0]))
    assert float(np.median(alignment)) > 0.98, (
        f"field must stay combed deep into the domain, |cos| p50 "
        f"{np.median(alignment):.3f}")


def test_strand_comb_reduces_fill_blotch_and_noops_when_off() -> None:
    """strand_comb=True on dark anisotropic-donor fill must reduce the
    coarse tone blotch; strand_comb=False must remain the default output."""
    from scipy.ndimage import gaussian_filter

    from abstract3d.texturing import synthesize_fill_detail

    positions, normals, observed, colors = _hair_fill_scene()
    fill = ~observed

    off = synthesize_fill_detail(
        colors.copy(), positions_texture=positions, normals_texture=normals,
        observed_mask=observed, gain=0.7, strand_comb=False)
    on = synthesize_fill_detail(
        colors.copy(), positions_texture=positions, normals_texture=normals,
        observed_mask=observed, gain=0.7, strand_comb=True)

    def coarse_blotch(rgba):
        lum = rgba[:, :, :3].mean(axis=2)
        coarse = gaussian_filter(lum, 4.0)
        deep = fill.copy()
        deep[:, : positions.shape[1] // 3 + 12] = False  # past seam feather
        return float(coarse[deep].std())

    assert coarse_blotch(on) < coarse_blotch(off) * 0.9, (
        f"combing must reduce coarse tone blotch: on {coarse_blotch(on):.4f} "
        f"vs off {coarse_blotch(off):.4f}")


def test_strand_comb_bit_identical_when_regime_empty() -> None:
    """On BRIGHT low-anisotropy material (single-photo canary analog) the
    strand regime is empty and strand_comb=True must produce EXACTLY the
    default output (the ship/owl md5 guarantee)."""
    from abstract3d.texturing import synthesize_fill_detail

    positions, normals, observed, colors = _hair_fill_scene()
    # brighten everything far above the dark-ratio split: regime empty
    colors[:, :, :3] = np.clip(colors[:, :, :3] + 0.6, 0, 1)

    off = synthesize_fill_detail(
        colors.copy(), positions_texture=positions, normals_texture=normals,
        observed_mask=observed, gain=0.7, strand_comb=False)
    on = synthesize_fill_detail(
        colors.copy(), positions_texture=positions, normals_texture=normals,
        observed_mask=observed, gain=0.7, strand_comb=True)
    assert np.array_equal(off, on), (
        "empty strand regime must leave the default path bit-identical")


# ---------------------------------------------------------------------------
# feature_fringe_repair (cycle-4: FACE-03/04 protected-feature fringe class)
# ---------------------------------------------------------------------------

def test_fringe_registration_recovers_similarity() -> None:
    """The gate-correspondence registration (bbox map + NCC refinement)
    must recover a known scale/shift between photo and render."""
    from abstract3d.feature_fringe_repair import (
        _register_photo_to_render,
        _render_foreground,
    )

    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(7)
    size = 320
    render = np.full((size, size, 3), 242, dtype=np.uint8)  # renderer bg
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    disc = (yy - 160) ** 2 + (xx - 160) ** 2 < 90 ** 2
    texture = gaussian_filter(
        rng.uniform(90, 200, (size, size, 3)), (5, 5, 0)).astype(np.uint8)
    render[disc] = texture[disc]
    dark = (yy - 150) ** 2 + (xx - 150) ** 2 < 12 ** 2
    render[dark] = 40

    # photo: same content on a bigger canvas with alpha, offset + scaled
    photo = np.zeros((400, 400, 4), dtype=np.uint8)
    from scipy.ndimage import zoom as nd_zoom

    scaled = nd_zoom(render, (1.05, 1.05, 1.0), order=1)
    mask = nd_zoom(disc.astype(np.float32), (1.05, 1.05), order=0) > 0.5
    photo[20:20 + scaled.shape[0], 30:30 + scaled.shape[1], :3] = scaled[:380, :370]
    photo[20:20 + scaled.shape[0], 30:30 + scaled.shape[1], 3] = (
        mask[:380, :370] * 255)

    fg = _render_foreground(render)
    warped, both, residual = _register_photo_to_render(photo, render, fg)
    assert warped is not None and both.sum() > 4000
    error = np.abs(warped[both].astype(np.float32)
                   - render[both].astype(np.float32)).mean()
    assert error < 12.0, f"registration failed to align content ({error:.1f})"


def test_fringe_compact_blob_classifier_semantics() -> None:
    """The veto's blob class: compact bright-ringed dark blob IN, elongated
    slit and hair-mass OUT."""
    from abstract3d.feature_fringe_repair import _compact_dark_blobs_px

    size = 400
    rgb = np.full((size, size, 3), 200, dtype=np.uint8)
    fg = np.zeros((size, size), bool)
    fg[40:360, 40:360] = True
    rgb[~fg] = 242
    rgb[100:124, 100:124] = 30          # compact blob (24px, ringed by skin)
    rgb[250:258, 60:340] = 30           # elongated slit (aspect 35)
    rgb[300:306, 200:206] = 30          # sub-feature speck (micro-island)
    blobs, micro = _compact_dark_blobs_px(rgb, fg, (4.0, 40.0))
    assert len(blobs) == 1, f"expected exactly the compact blob, got {len(blobs)}"
    assert abs(blobs[0]["center"][0] - 112) < 4 and abs(blobs[0]["center"][1] - 112) < 4
    # micro-island fraction counts the SPECK, not the feature-scale blob
    # or the slit (36 px on a 320x320 bbox)
    assert 0.0 < micro < 0.001


def test_fringe_world_clustering_links_across_atlas_cut() -> None:
    """One physical feature atlased as TWO distant UV charts must form ONE
    complex: world voxel-graph clustering is chart-blind, where atlas
    morphology fragments it below the size floor (the measured 2048
    mouth-complex formation failure)."""
    from abstract3d.feature_fringe_repair import _cluster_core_texels_world

    size = 256
    points = np.zeros((size, size, 3), np.float32)
    core = np.zeros((size, size), bool)
    # chart A: left half of a feature at world x in [0, 0.05]
    xs, ys = np.meshgrid(np.linspace(0.0, 0.05, 24), np.linspace(0.0, 0.05, 24))
    points[10:34, 10:34, 0] = xs
    points[10:34, 10:34, 1] = ys
    core[10:34, 10:34] = True
    # chart B: right half at world x in [0.055, 0.105] — adjacent in world,
    # 150 texels away in the atlas
    points[10:34, 180:204, 0] = xs + 0.055
    points[10:34, 180:204, 1] = ys
    core[10:34, 180:204] = True
    # unrelated distant feature
    points[200:212, 200:212, 0] = 5.0
    core[200:212, 200:212] = True

    labels = _cluster_core_texels_world(points, core, link_world=0.02)
    a = labels[20, 20]
    b = labels[20, 190]
    c = labels[205, 205]
    assert a >= 0 and a == b, "world-adjacent chart halves must cluster together"
    assert c >= 0 and c != a, "world-distant content must stay separate"


def test_fringe_texel_structure_veto_fires_on_new_blob_only() -> None:
    """Creating a compact dark blob on bright material vetoes; moving an
    existing blob a few texels (re-registration) does not."""
    from abstract3d.feature_fringe_repair import _texel_structure_veto

    size = 200
    positions = np.zeros((size, size, 4), dtype=np.float32)
    xs, ys = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    positions[:, :, 0] = xs
    positions[:, :, 2] = ys
    positions[:, :, 3] = 1.0
    normals = np.zeros((size, size, 3), dtype=np.float32)
    normals[:, :, 0] = 1.0  # facing +x, frontal for az0
    surface = positions[:, :, 3] > 0
    scale = float(np.linalg.norm([2.0, 0.0, 2.0]))
    r_ctx = 0.02 * scale

    before = np.full((size, size, 4), 0.8, dtype=np.float32)
    before[90:104, 90:104, :3] = 0.2       # existing feature blob
    confident = np.zeros((size, size), bool)
    confident[88:106, 88:106] = True       # the feature is witnessed

    moved = before.copy()
    moved[90:104, 90:104, :3] = 0.8
    moved[92:106, 93:107, :3] = 0.2        # same blob, shifted 2-3 texels
    patch = np.zeros((size, size), bool)
    patch[80:120, 80:120] = True
    assert _texel_structure_veto(
        positions[:, :, :3], normals, surface, before, moved, patch,
        bright_median=0.8, r_ctx=r_ctx, scale=scale,
        confident_mask=confident) is None

    created = before.copy()
    created[150:164, 150:164, :3] = 0.2    # NEW compact blob
    patch2 = np.zeros((size, size), bool)
    patch2[140:180, 140:180] = True
    veto = _texel_structure_veto(
        positions[:, :, :3], normals, surface, before, created, patch2,
        bright_median=0.8, r_ctx=r_ctx, scale=scale,
        confident_mask=confident)
    assert veto is not None and "new_dark_blob" in veto

    smeared = before.copy()
    smeared[90:104, 90:104, :3] = 0.8      # CONFIDENT feature destroyed
    veto = _texel_structure_veto(
        positions[:, :, :3], normals, surface, before, smeared, patch,
        bright_median=0.8, r_ctx=r_ctx, scale=scale,
        confident_mask=confident)
    assert veto is not None and "lost_dark_blob" in veto

    # the SAME loss with the blob at TRACE witness is a repaired defect,
    # not a smeared feature: no veto
    assert _texel_structure_veto(
        positions[:, :, :3], normals, surface, before, smeared, patch,
        bright_median=0.8, r_ctx=r_ctx, scale=scale,
        confident_mask=np.zeros((size, size), bool)) is None


def test_render_veto_cumulative_baseline_closes_rearm_creep(monkeypatch) -> None:
    """The advancing per-stamp baseline re-arms the +0.0003 micro budget
    with every acceptance (measured: ~7 stamps -> +0.00096 at one view,
    triple the single-stamp budget, inside the letter of every per-stamp
    check). The cumulative-baseline veto (critic 2's cycle-5 hardening,
    adopted as mandatory by the cycle-6 certification) refuses a candidate
    whose post-stamp micro fraction exceeds BOTH the view's ORIGINAL
    pre-repair fraction + 0.0003 AND the original battery worst — while
    growth up to the original battery worst stays admissible (the
    photo-truth exemption's own bound is untouched)."""
    import abstract3d.feature_fringe_repair as ffr

    size = 896

    def speck_render(count: int) -> np.ndarray:
        rgb = np.full((size, size, 3), 242, dtype=np.uint8)
        rgb[48:848, 48:848] = 200                     # bright subject
        rgb[700:760, 700:760] = 30                    # constant dark mass
        for i in range(count):                        # 100-px specks
            row = 100 + 40 * i
            rgb[row:row + 10, 100:110] = 30
        return rgb

    class _MeshStub:
        vertices = np.array(
            [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0],
             [1.0, -1.0, 0.0], [-1.0, 1.0, 0.0]], dtype=np.float32)

    mesh = _MeshStub()
    r_feat_world = 0.05

    def entry_for(rgb: np.ndarray, az: float, el: float) -> dict:
        fg = ffr._render_foreground(rgb)
        camera = ffr._renderer_camera(mesh, az, el, size)
        r_px = r_feat_world * camera["px_per_world"]
        blobs, micro = ffr._compact_dark_blobs_px(rgb, fg, (0.25 * r_px, 2.6 * r_px))
        return {"blobs": blobs, "micro": micro, "rgb": rgb, "fg": fg}

    # one battery view at az0: original 2 specks; the advancing baseline
    # carries one prior acceptance (3 specks, +1.6e-4 — inside the
    # per-stamp budget); the candidate adds one more (4 specks): its
    # per-stamp delta is again +1.6e-4 < 3e-4, but cumulatively
    # +3.1e-4 > 3e-4 over the original AND above the original battery
    # worst (the az0 view IS the battery worst) -> must refuse.
    original = [((0.0, (0.0,)), [entry_for(speck_render(2), 0.0, 0.0)])]
    advancing = [((0.0, (0.0,)), [entry_for(speck_render(3), 0.0, 0.0)])]
    after_rgb = speck_render(4)

    per_stamp_delta = after_rgb is not None  # readability anchor
    assert per_stamp_delta
    monkeypatch.setattr(
        ffr, "_render_with_colors",
        lambda mesh_, atlas_, colors_, azimuths, el, size_: [after_rgb
                                                             for _ in azimuths])

    veto, _ = ffr._render_structure_veto(
        mesh, None, advancing, colors_after=None, r_feat_world=r_feat_world,
        size=size, original_renders=original)
    assert veto is not None and "cumulative" in veto, (
        f"re-arm creep past the original battery worst must refuse, got {veto}")

    # sanity of the construction: each per-stamp delta really is inside
    # the single budget (the OLD semantics would have accepted this)
    m2 = original[0][1][0]["micro"]
    m3 = advancing[0][1][0]["micro"]
    m4 = entry_for(after_rgb, 0.0, 0.0)["micro"]
    assert m3 - m2 <= 0.0003 and m4 - m3 <= 0.0003 and m4 - m2 > 0.0003

    # same candidate under a battery whose ORIGINAL worst view is far
    # above the grown view: growth stays below the pre-repair worst
    # offender, the exemption bound semantics -> admissible.
    worst_rgb = speck_render(8)
    original_wide = [
        ((0.0, (0.0,)), [entry_for(speck_render(2), 0.0, 0.0)]),
        ((10.0, (45.0,)), [entry_for(worst_rgb, 45.0, 10.0)]),
    ]
    advancing_wide = [
        ((0.0, (0.0,)), [entry_for(speck_render(3), 0.0, 0.0)]),
        ((10.0, (45.0,)), [entry_for(worst_rgb, 45.0, 10.0)]),
    ]

    def render_by_view(mesh_, atlas_, colors_, azimuths, el, size_):
        return [after_rgb if abs(float(az)) < 1.0 else worst_rgb
                for az in azimuths]

    monkeypatch.setattr(ffr, "_render_with_colors", render_by_view)
    veto, after_entries = ffr._render_structure_veto(
        mesh, None, advancing_wide, colors_after=None,
        r_feat_world=r_feat_world, size=size, original_renders=original_wide)
    assert veto is None and after_entries is not None, (
        f"growth below the original battery worst must stay admissible: {veto}")


def test_repair_feature_fringes_noop_contracts() -> None:
    """Single-view input and missing source image are STRUCTURAL no-ops
    (the identity correspondence needs the multi-view contract); the
    single-photo proof assets must stay bit-identical."""
    from abstract3d.feature_fringe_repair import repair_feature_fringes

    size = 96
    positions = np.zeros((size, size, 4), dtype=np.float32)
    xs, ys = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    positions[:, :, 0] = xs
    positions[:, :, 2] = ys
    positions[:, :, 3] = 1.0
    normals = np.zeros((size, size, 3), dtype=np.float32)
    normals[:, :, 0] = 1.0
    colors = np.full((size, size, 4), 0.7, dtype=np.float32)
    weight = np.full((size, size), 0.5, dtype=np.float32)
    projection = {"weight": weight,
                  "rgba": np.concatenate(
                      [colors[:, :, :3], np.ones((size, size, 1), np.float32)],
                      axis=2)}

    stats: dict = {}
    out, mask = repair_feature_fringes(
        None, atlas=None, colors_rgba=colors,
        positions_texture=positions, normals_texture=normals,
        projections=[projection], source_image=None,
        source_azimuth_deg=0.0, source_elevation_deg=0.0,
        stats_out=stats)
    assert mask is None and out is colors and not stats["applied"]

    out, mask = repair_feature_fringes(
        None, atlas=None, colors_rgba=colors,
        positions_texture=positions, normals_texture=normals,
        projections=[projection, projection], source_image=None,
        source_azimuth_deg=0.0, source_elevation_deg=0.0)
    assert mask is None and out is colors


# ---------------------------------------------------------------------------
# commit_pale_chips (cycle-4: FACE-07 ear-band pale chip class)
# ---------------------------------------------------------------------------

def _pale_chip_scene(size: int = 160):
    """Flat sheet split into a bright skin mass (left) and dark hair
    (right), with the chip placed deep inside the dark half."""
    positions = np.zeros((size, size, 4), dtype=np.float32)
    xs, ys = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    positions[:, :, 0] = xs
    positions[:, :, 2] = ys
    positions[:, :, 3] = 1.0
    base = np.full((size, size, 3), 0.15, dtype=np.float32)   # dark hair
    base[:, : size // 2] = 0.8                                # bright skin mass
    return positions, base


def test_commit_pale_chips_retones_dark_consensus_island() -> None:
    """A pale island at trace weight deep in hair, whose ring every witness
    reads uniformly dark, must be retoned to the dark ring tone."""
    from abstract3d.texturing import commit_pale_chips

    positions, base = _pale_chip_scene()
    size = positions.shape[0]
    blob = np.zeros((size, size), dtype=bool)
    blob[78:82, 118:122] = True    # deep inside the dark half

    colors = np.zeros((size, size, 4), dtype=np.float32)
    colors[:, :, :3] = base
    colors[blob, :3] = 0.7         # displaced pale content
    colors[:, :, 3] = 1.0

    weight_a = np.full((size, size), 0.5, dtype=np.float32)
    weight_a[blob] = 0.0
    rgb_a = base.copy()
    weight_b = np.full((size, size), 0.05, dtype=np.float32)  # trace winner
    rgb_b = base.copy()
    rgb_b[blob] = 0.7

    stats: dict = {}
    out = commit_pale_chips(
        colors,
        positions_texture=positions,
        observed_mask=np.ones((size, size), dtype=bool),
        projections=[_projection(weight_a, rgb_a), _projection(weight_b, rgb_b)],
        stats_out=stats,
    )
    assert stats["applied"], "pale chip on dark consensus must commit"
    assert float(out[blob, :3].mean()) < 0.3, (
        f"chip must be retoned to dark ring tone, got {out[blob, :3].mean():.3f}")


def test_commit_pale_chips_never_touches_confident_pale() -> None:
    """The same pale island CONFIDENTLY witnessed (skin seen between
    strands) is photo truth and must never be demoted."""
    from abstract3d.texturing import commit_pale_chips

    positions, base = _pale_chip_scene()
    size = positions.shape[0]
    blob = np.zeros((size, size), dtype=bool)
    blob[78:82, 118:122] = True

    colors = np.zeros((size, size, 4), dtype=np.float32)
    colors[:, :, :3] = base
    colors[blob, :3] = 0.7
    colors[:, :, 3] = 1.0

    weight_a = np.full((size, size), 0.5, dtype=np.float32)   # confident winner
    rgb_a = base.copy()
    rgb_a[blob] = 0.7
    weight_b = np.full((size, size), 0.05, dtype=np.float32)
    rgb_b = base.copy()

    out = commit_pale_chips(
        colors,
        positions_texture=positions,
        observed_mask=np.ones((size, size), dtype=bool),
        projections=[_projection(weight_a, rgb_a), _projection(weight_b, rgb_b)],
    )
    assert np.allclose(out[blob, :3], 0.7, atol=1e-5), "confident pale content demoted"


def test_commit_pale_chips_refuses_bright_frontier_sliver() -> None:
    """A pale sliver CONNECTED to the big bright mass is the frontier of
    real material, not an island — must be refused."""
    from abstract3d.texturing import commit_pale_chips

    positions, base = _pale_chip_scene()
    size = positions.shape[0]
    sliver = np.zeros((size, size), dtype=bool)
    edge_col = size // 2
    sliver[78:82, edge_col:edge_col + 4] = True   # touching the skin mass

    colors = np.zeros((size, size, 4), dtype=np.float32)
    colors[:, :, :3] = base
    colors[sliver, :3] = 0.7
    colors[:, :, 3] = 1.0

    weight_a = np.full((size, size), 0.5, dtype=np.float32)
    weight_a[sliver] = 0.0
    rgb_a = base.copy()
    weight_b = np.full((size, size), 0.05, dtype=np.float32)
    rgb_b = base.copy()
    rgb_b[sliver] = 0.7

    out = commit_pale_chips(
        colors,
        positions_texture=positions,
        observed_mask=np.ones((size, size), dtype=bool),
        projections=[_projection(weight_a, rgb_a), _projection(weight_b, rgb_b)],
    )
    assert np.allclose(out[sliver, :3], 0.7, atol=1e-5), "frontier sliver demoted"


def test_commit_pale_chips_noop_single_view() -> None:
    """Single witness = vacuous ring consensus; must pass through."""
    from abstract3d.texturing import commit_pale_chips

    positions, base = _pale_chip_scene()
    size = positions.shape[0]
    colors = np.zeros((size, size, 4), dtype=np.float32)
    colors[:, :, :3] = base
    colors[78:82, 118:122, :3] = 0.7
    colors[:, :, 3] = 1.0
    weight = np.full((size, size), 0.05, dtype=np.float32)

    out = commit_pale_chips(
        colors,
        positions_texture=positions,
        observed_mask=np.ones((size, size), dtype=bool),
        projections=[_projection(weight, base.copy())],
    )
    assert np.array_equal(out, colors), "single-view input must pass through"


# ---------------------------------------------------------------------------
# tone_bottom_cap (cycle-4: FACE-12 synthetic cut-face toning)
# ---------------------------------------------------------------------------

def _cap_scene(size: int = 160):
    """A bottom cap plane (normals -z, unobserved, wrong gray tone) whose
    atlas neighborhood is an observed warm rim."""
    positions = np.zeros((size, size, 4), dtype=np.float32)
    xs, ys = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    positions[:, :, 0] = xs
    positions[:, :, 1] = ys
    positions[:, :, 2] = -0.98
    positions[:, :, 3] = 1.0
    cap = np.zeros((size, size), dtype=bool)
    cap[40:120, 40:120] = True
    positions[cap, 2] = -1.0

    normals = np.zeros((size, size, 4), dtype=np.float32)
    normals[:, :, 2] = 1.0
    normals[cap, 2] = -1.0
    normals[:, :, 3] = 1.0

    colors = np.zeros((size, size, 4), dtype=np.float32)
    colors[:, :, :3] = np.array([0.75, 0.6, 0.5], np.float32)  # warm rim
    colors[cap, :3] = 0.45                                      # gray wash
    colors[:, :, 3] = 1.0

    observed = ~cap
    return positions, normals, colors, observed, cap


def test_tone_bottom_cap_tones_cut_face_from_rim() -> None:
    from abstract3d.texturing import tone_bottom_cap

    positions, normals, colors, observed, cap = _cap_scene()
    stats: dict = {}
    out = tone_bottom_cap(
        colors,
        positions_texture=positions,
        normals_texture=normals,
        observed_mask=observed,
        direct_observed_mask=observed,
        stats_out=stats,
    )
    assert stats["applied"], "planar unobserved cap must be detected"
    center = out[75:85, 75:85, :3].reshape(-1, 3).mean(axis=0)
    assert abs(float(center[0]) - 0.75) < 0.08, f"cap not toned from rim: {center}"
    assert float(center[0]) > float(center[2]), "rim warmth (R>B) not inherited"


def test_tone_bottom_cap_noop_without_planar_cap() -> None:
    from abstract3d.texturing import tone_bottom_cap

    positions, normals, colors, observed, cap = _cap_scene()
    normals[:, :, 2] = 1.0  # no downward-facing plane anywhere
    out = tone_bottom_cap(
        colors,
        positions_texture=positions,
        normals_texture=normals,
        observed_mask=observed,
        direct_observed_mask=observed,
    )
    assert np.array_equal(out, colors), "no cap => bit-identical pass-through"
