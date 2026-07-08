from __future__ import annotations

import numpy as np
from PIL import Image

from abstract3d import texturing
from abstract3d.gradient_compositing import (
    build_texel_surface_graph,
    composite_gradient_domain,
    select_composite_gradients,
    solve_screened_poisson,
)


def _two_chart_sphere(res: int = 128, gap: int = 4):
    """Unit sphere atlased as two lat-long charts (north band on the atlas
    left, south band on the right) with a texel gap between the charts, so
    cross-chart continuity can only come from 3D stitch edges."""
    half = res // 2
    positions = np.zeros((res, res, 4), dtype=np.float32)
    for chart, (col0, col1) in enumerate(((0, half - gap), (half + gap, res))):
        width = col1 - col0
        phi = (np.arange(width) + 0.5) / width * 2.0 * np.pi
        if chart == 0:
            theta = (np.arange(res) + 0.5) / res * (np.pi / 2.0)
        else:
            theta = np.pi / 2.0 + (np.arange(res) + 0.5) / res * (np.pi / 2.0)
        ph, th = np.meshgrid(phi, theta)
        positions[:, col0:col1, 0] = np.sin(th) * np.cos(ph)
        positions[:, col0:col1, 1] = np.sin(th) * np.sin(ph)
        positions[:, col0:col1, 2] = np.cos(th)
        positions[:, col0:col1, 3] = 1.0
    normals = positions.copy()  # unit sphere: normal == position
    return positions, normals


def _two_views(positions: np.ndarray, albedo: np.ndarray, *, offset: float,
               hard_handoff: bool = True):
    """Two views along azimuth +/-40 deg; view B carries a constant
    additive exposure offset (the classic tone-seam corruption). With
    `hard_handoff` the losing view's weight is zeroed per texel — exactly
    what winner-take-all conflict resolution produces."""
    surface = positions[:, :, 3] > 0.0
    xyz = positions[:, :, :3]
    a, b = np.deg2rad(40.0), np.deg2rad(140.0)
    axes = (
        np.array([np.cos(a), np.sin(a), 0.0], dtype=np.float32),
        np.array([np.cos(b), np.sin(b), 0.0], dtype=np.float32),
    )
    views = []
    for index, axis in enumerate(axes):
        facing = (xyz * axis[None, None, :]).sum(axis=2)
        # As in the real projector, VALID SAMPLING (photo alpha) extends
        # past the weight cutoff: rim texels carry colors but no weight.
        valid = surface & (facing > 0.10)
        weight = np.where(surface & (facing > 0.18), facing, 0.0).astype(np.float32)
        rgb = albedo + (offset if index == 1 else 0.0)
        rgb = np.where(valid[:, :, None], np.clip(rgb, 0.0, 1.0), 0.0)
        rgba = np.concatenate(
            [rgb, valid[:, :, None].astype(np.float32)], axis=2
        ).astype(np.float32)
        views.append({"rgba": rgba, "weight": weight})
    if hard_handoff:
        keep0 = views[0]["weight"] >= views[1]["weight"]
        views[0]["weight"] = np.where(keep0, views[0]["weight"], 0.0)
        views[1]["weight"] = np.where(~keep0, views[1]["weight"], 0.0)
    return views


def _handoff_jump(rgb: np.ndarray, views, surface: np.ndarray,
                  gt: np.ndarray) -> float:
    """Mean |ERROR jump| across adjacent texel pairs whose winner differs.
    Measuring the error (rgb - gt) keeps legitimate content edges (checker
    borders crossing the handoff) out of the seam statistic."""
    weights = np.stack([v["weight"] for v in views])
    winner = weights.argmax(axis=0)
    ok = surface & (weights.max(axis=0) > 0)
    err = rgb - gt
    jumps = []
    pair_h = ok[:, 1:] & ok[:, :-1] & (winner[:, 1:] != winner[:, :-1])
    if pair_h.any():
        jumps.append(np.abs(err[:, 1:][pair_h] - err[:, :-1][pair_h]).mean(axis=1))
    pair_v = ok[1:, :] & ok[:-1, :] & (winner[1:, :] != winner[:-1, :])
    if pair_v.any():
        jumps.append(np.abs(err[1:, :][pair_v] - err[:-1, :][pair_v]).mean(axis=1))
    return float(np.concatenate(jumps).mean()) if jumps else 0.0


def test_texel_surface_graph_stitches_charts_and_guards_uv_jumps() -> None:
    positions, normals = _two_chart_sphere(res=96)
    graph = build_texel_surface_graph(positions, normals_texture=normals)
    assert graph is not None
    edges = graph["edges"]
    kind = graph["edge_kind"]
    cols = graph["nodes_rc"][:, 1]
    half = positions.shape[1] // 2
    # Stitch edges must reconnect the two charts across the packed gap.
    cross_chart = (cols[edges[:, 0]] < half) != (cols[edges[:, 1]] < half)
    assert cross_chart.any()
    assert (kind[cross_chart] == 1).all()  # only stitches cross charts
    # Grid edges never jump in 3D: adjacent-in-UV pairs from different
    # charts are 3D-distant and must have been rejected.
    xyz = positions[:, :, :3]
    rc = graph["nodes_rc"]
    grid = edges[kind == 0]
    lengths = np.linalg.norm(
        xyz[rc[grid[:, 0], 0], rc[grid[:, 0], 1]]
        - xyz[rc[grid[:, 1], 0], rc[grid[:, 1], 1]],
        axis=1,
    )
    assert float(lengths.max()) <= 4.0 * graph["pitch"] + 1e-6


def test_texel_surface_graph_does_not_stitch_opposed_sheets() -> None:
    # Two parallel flat sheets a hair apart with OPPOSITE normals (a thin
    # shell): the normal-agreement gate must refuse to stitch them even
    # though they pass the distance test.
    size = 32
    positions = np.zeros((size, size, 4), dtype=np.float32)
    normals = np.zeros((size, size, 4), dtype=np.float32)
    xs = (np.arange(size, dtype=np.float32) + 0.5) / size
    half = size // 2
    for row in range(size):
        # left chart: sheet at z=0, normal +z; right chart: z=pitch, -z.
        positions[row, :half, 0] = xs[:half] * 0.5
        positions[row, :half, 1] = row / size * 0.5
        positions[row, :half, 3] = 1.0
        normals[row, :half, 2] = 1.0
        positions[row, half:, 0] = xs[:half] * 0.5
        positions[row, half:, 1] = row / size * 0.5
        positions[row, half:, 2] = 0.5 / size  # one pitch above
        positions[row, half:, 3] = 1.0
        normals[row, half:, 2] = -1.0
    graph = build_texel_surface_graph(positions, normals_texture=normals)
    assert graph is not None
    cols = graph["nodes_rc"][:, 1]
    edges = graph["edges"]
    cross = (cols[edges[:, 0]] < half) != (cols[edges[:, 1]] < half)
    assert not cross.any()


def test_gradient_selection_max_confidence_and_one_sided_handoff() -> None:
    # A 1x4 strip of surface texels, two views:
    #   view 0 confident on cols 0-1, validly sampled everywhere;
    #   view 1 confident on cols 2-3, validly sampled cols 2-3 only.
    # Edge (1,2) is a winner-take-all handoff with no common witness: the
    # one-sided rule must take view 0's gradient there (it validly sampled
    # both endpoints), preserving the real content edge.
    positions = np.zeros((1, 4, 4), dtype=np.float32)
    positions[0, :, 0] = np.arange(4) * 0.01
    positions[0, :, 3] = 1.0
    graph = build_texel_surface_graph(positions, stitch_chart_borders=False)
    assert graph is not None and len(graph["edges"]) == 3

    rgb0 = np.zeros((1, 4, 4), dtype=np.float32)
    rgb0[0, :, 0] = [0.2, 0.3, 0.8, 0.9]  # view 0 sees a strong edge at 1|2
    rgb0[0, :, 3] = 1.0
    rgb1 = np.zeros((1, 4, 4), dtype=np.float32)
    rgb1[0, :, 0] = [0.0, 0.0, 0.75, 0.85]
    rgb1[0, 2:, 3] = 1.0
    w0 = np.array([[0.9, 0.9, 0.0, 0.0]], dtype=np.float32)
    w1 = np.array([[0.0, 0.0, 0.8, 0.8]], dtype=np.float32)
    class_map = np.zeros((1, 4), dtype=np.int32)

    gradients, line_mask = select_composite_gradients(
        graph,
        view_rgb=[rgb0, rgb1],
        view_weight=[w0, w1],
        class_map=class_map,
        filled_rgb=np.zeros((1, 4, 3), dtype=np.float32),
        view_valid=[rgb0[:, :, 3] > 0, rgb1[:, :, 3] > 0],
        rule="max_confidence",
    )
    edges = graph["edges"]
    cols = graph["nodes_rc"][:, 1]

    def edge_index(a: int, b: int) -> int:
        for k, (i, j) in enumerate(edges):
            if {cols[i], cols[j]} == {a, b}:
                return k
        raise AssertionError("edge not found")

    # Common-witness edges take their witness's gradient (sign follows the
    # stored edge orientation, compare absolute values).
    assert abs(abs(gradients[edge_index(0, 1)][0]) - 0.1) < 1e-6  # view 0
    assert abs(abs(gradients[edge_index(2, 3)][0]) - 0.1) < 1e-6  # view 1
    # Handoff edge: one-sided witness from view 0 (0.8 - 0.3 = 0.5), NOT a
    # zero membrane target and NOT view 1 (which never sampled col 1).
    assert abs(abs(gradients[edge_index(1, 2)][0]) - 0.5) < 1e-6
    # The handoff edge is a line constraint; interior edges are not.
    assert bool(line_mask[edge_index(1, 2)]) is True
    assert bool(line_mask[edge_index(0, 1)]) is False
    assert bool(line_mask[edge_index(2, 3)]) is False


def test_screened_poisson_removes_exposure_step_and_keeps_edges() -> None:
    positions, normals = _two_chart_sphere(res=128)
    surface = positions[:, :, 3] > 0.0
    xyz = positions[:, :, :3]
    # Albedo: smooth field + a hard checker edge pattern (content edges).
    checker = ((np.floor(xyz[:, :, 0] * 4.0) + np.floor(xyz[:, :, 2] * 4.0)) % 2.0)
    albedo = np.stack(
        [0.35 + 0.3 * checker, 0.45 + 0.1 * checker, 0.55 - 0.2 * checker], axis=2
    ).astype(np.float32)
    views = _two_views(positions, albedo, offset=0.09)
    observed = np.stack([v["weight"] for v in views]).max(axis=0) > 0

    # The color-domain blend keeps the exposure step at the handoff.
    blend = np.zeros_like(albedo)
    for v in views:
        take = v["weight"] > 0
        blend[take] = v["rgba"][:, :, :3][take]

    positions_masked = positions.copy()
    positions_masked[:, :, 3] = np.where(observed, positions[:, :, 3], 0.0)
    class_map = np.where(observed, 0, -1).astype(np.int32)
    filled = blend.copy()
    solved = composite_gradient_domain(
        positions_texture=positions_masked,
        normals_texture=normals,
        view_rgb=[v["rgba"] for v in views],
        view_weight=[v["weight"] for v in views],
        class_map=class_map,
        filled_rgb=filled,
        anchor_confidence=np.stack([v["weight"] for v in views]).max(axis=0),
        view_valid=[v["rgba"][:, :, 3] > 0 for v in views],
        anchor_lambda_scale=1e-9,  # pure gradient integration
        resolution_reference=128,
        cg_tol=1e-7,
        cg_max_iterations=3000,
    )
    assert solved is not None
    out, stats = solved
    assert stats["final_relative_residual"] < 1e-5

    # The additive offset vanishes: albedo recovered up to ONE global
    # constant across BOTH views and BOTH charts.
    error = (out - albedo)[observed]
    error -= error.mean(axis=0, keepdims=True)
    assert float(np.sqrt((error**2).mean())) < 2e-3
    # The handoff discontinuity collapses by >20x vs the color blend...
    assert _handoff_jump(out, views, surface, albedo) < 0.05 * _handoff_jump(
        blend, views, surface, albedo
    )
    # ...while checker content edges keep their contrast (no blur): compare
    # mean |horizontal step| on strong-edge pairs inside the observed set.
    gt_step = np.abs(albedo[:, 1:] - albedo[:, :-1]).mean(axis=2)
    pair = observed[:, 1:] & observed[:, :-1] & (gt_step > 0.15)
    out_step = np.abs(out[:, 1:] - out[:, :-1]).mean(axis=2)
    assert float((out_step[pair] / gt_step[pair]).mean()) > 0.95


def test_solve_screened_poisson_anchors_pin_confident_colors() -> None:
    # One flat chart, uniform target gradients of zero, anchors at two
    # different colors with strong lambda: the solution must stay at the
    # anchors where lambda is large and bridge smoothly in between.
    size = 48
    positions = np.zeros((1, size, 4), dtype=np.float32)
    positions[0, :, 0] = np.arange(size) * 0.01
    positions[0, :, 3] = 1.0
    graph = build_texel_surface_graph(positions, stitch_chart_borders=False)
    assert graph is not None
    anchors = np.zeros((size, 3), dtype=np.float32)
    anchors[size // 2 :] = 1.0
    lam = np.zeros(size, dtype=np.float64)
    lam[:4] = 10.0
    lam[-4:] = 10.0
    solved = solve_screened_poisson(
        graph,
        gradients=np.zeros((len(graph["edges"]), 3), dtype=np.float32),
        anchors_rgb=anchors,
        anchor_lambda=lam,
        initial_rgb=anchors,
        cg_tol=1e-9,
        cg_max_iterations=2000,
    )
    assert solved is not None
    x, _ = solved
    assert float(np.abs(x[:4] - 0.0).max()) < 0.02
    assert float(np.abs(x[-4:] - 1.0).max()) < 0.02
    interior = np.diff(x[:, 0])
    assert float(np.abs(interior).max()) < 0.08  # smooth ramp, no jump


def test_bake_projection_texture_gradient_domain_end_to_end() -> None:
    import trimesh

    mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.5)
    observed = Image.new("RGBA", (96, 96), (200, 60, 40, 255))
    views = [
        {"rgba": observed, "azimuth_deg": 0.0, "elevation_deg": 0.0, "label": "front"}
    ]

    textured, stats = texturing.bake_projection_texture(
        mesh,
        observed_views=views,
        texture_resolution=96,
        compositing="gradient_domain",
    )
    assert stats["compositing"]["mode"] == "gradient_domain"
    assert stats["compositing"]["applied"] is True
    solver = stats["compositing"]["solver"]
    assert solver["final_relative_residual"] < 1e-4
    assert solver["nodes"] > 0 and solver["edges"] > 0
    # The observed hemisphere still carries the photo color.
    texture = np.asarray(stats["texture_image"], dtype=np.float32)
    assert float(texture[:, :, 0].mean()) > float(texture[:, :, 2].mean())

    # Legacy path stays selectable and does not run the solve.
    _, legacy_stats = texturing.bake_projection_texture(
        mesh,
        observed_views=views,
        texture_resolution=96,
        compositing="legacy",
    )
    assert legacy_stats["compositing"]["mode"] == "legacy"
    assert legacy_stats["compositing"]["applied"] is False


def test_bake_projection_texture_gradient_domain_is_deterministic() -> None:
    import trimesh

    mesh = trimesh.creation.icosphere(subdivisions=2, radius=0.5)
    observed = Image.new("RGBA", (64, 64), (90, 160, 220, 255))
    views = [
        {"rgba": observed, "azimuth_deg": 0.0, "elevation_deg": 0.0, "label": "front"}
    ]
    first, first_stats = texturing.bake_projection_texture(
        mesh, observed_views=views, texture_resolution=64,
        compositing="gradient_domain",
    )
    second, second_stats = texturing.bake_projection_texture(
        mesh, observed_views=views, texture_resolution=64,
        compositing="gradient_domain",
    )
    a = np.asarray(first_stats["texture_image"], dtype=np.int16)
    b = np.asarray(second_stats["texture_image"], dtype=np.int16)
    # Bit-identity holds on the validated Apple-local profile (certified via
    # texture hashes). On other numpy builds, allocator-dependent SIMD
    # reduction grouping can wobble intermediate floats by 1 ULP, which after
    # uint8 quantization is at most 1 LSB at a rounding boundary. The
    # portable guarantee is therefore <= 1 LSB, with the overwhelming
    # majority of texels bit-identical.
    diff = np.abs(a - b)
    assert int(diff.max()) <= 1
    assert float((diff == 0).mean()) >= 0.999


# ---------------------------------------------------------------------------
# reconcile_specular_lobes (cycle-4: FACE-05 pale seam column class)
# ---------------------------------------------------------------------------

def _lobe_scene(size: int = 200):
    """Two views of one sheet: the source carries a smooth bright
    DESATURATED lobe (baked specular); the reference reads the same
    surface at the diffuse level."""
    skin = np.zeros((size, size, 3), np.float32)
    skin[:] = (0.72, 0.56, 0.50)
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32)
    lobe_field = np.exp(-(((ys - 100) / 14.0) ** 2 + ((xs - 100) / 9.0) ** 2))

    # specular blends toward the light color: luminance up, saturation down
    mix = 0.45 * lobe_field[:, :, None]
    source = np.clip(skin * (1.0 - mix) + 0.95 * mix, 0, 1)

    reference = skin.copy()

    w_src = np.full((size, size), 0.6, np.float32)
    w_ref = np.full((size, size), 0.30, np.float32)
    valid = np.ones((size, size), bool)
    observed = np.ones((size, size), bool)
    return source, reference, w_src, w_ref, valid, observed, lobe_field


def test_reconcile_specular_lobes_flattens_authorized_lobe() -> None:
    from abstract3d.gradient_compositing import reconcile_specular_lobes

    source, reference, w_src, w_ref, valid, observed, lobe = _lobe_scene()
    out = reconcile_specular_lobes(
        view_rgb=[source, reference],
        view_weight=[w_src, w_ref],
        view_valid=[valid, valid],
        observed_mask=observed,
        source_view_index=0,
        reference_resolution=source.shape[0],
    )
    assert out is not None, "authorized smooth lobe must be reconciled"
    delta, stats = out
    core = lobe > 0.6
    corrected = np.clip(source + delta, 0, 1)
    lum = corrected.mean(axis=2)
    base_lum = np.full_like(lum, np.mean((0.72, 0.56, 0.50)))
    assert float(np.abs(lum[core] - base_lum[core]).mean()) < 0.06, (
        "lobe core luminance must return to the surround level")
    rim = lobe < 0.05
    assert float(np.abs(delta[rim]).max()) < 1e-3, "surround must stay untouched"


def test_reconcile_specular_lobes_keeps_shared_bright_content() -> None:
    """When BOTH views read the region bright, consensus refuses: shared
    brightness is albedo (or shared shine) and must not be flattened."""
    from abstract3d.gradient_compositing import reconcile_specular_lobes

    source, reference, w_src, w_ref, valid, observed, lobe = _lobe_scene()
    mix = 0.45 * lobe[:, :, None]
    reference_bright = np.clip(reference * (1.0 - mix) + 0.95 * mix, 0, 1)
    out = reconcile_specular_lobes(
        view_rgb=[source, reference_bright],
        view_weight=[w_src, w_ref],
        view_valid=[valid, valid],
        observed_mask=observed,
        source_view_index=0,
        reference_resolution=source.shape[0],
    )
    assert out is None, "shared bright content must not be reconciled"


def test_reconcile_specular_lobes_refuses_edge_dense_features() -> None:
    """Bright+desaturated but EDGE-DENSE content (sclera/teeth analog)
    must be refused by the feature gate."""
    from abstract3d.gradient_compositing import reconcile_specular_lobes

    source, reference, w_src, w_ref, valid, observed, lobe = _lobe_scene()
    checker = ((np.indices(source.shape[:2]).sum(axis=0) // 2) % 2).astype(np.float32)
    core = lobe > 0.3
    source[core] = np.clip(
        source[core] + 0.30 * (checker[core] - 0.5)[:, None], 0, 1)
    out = reconcile_specular_lobes(
        view_rgb=[source, reference],
        view_weight=[w_src, w_ref],
        view_valid=[valid, valid],
        observed_mask=observed,
        source_view_index=0,
        reference_resolution=source.shape[0],
    )
    assert out is None, "edge-dense bright content must be refused"


def test_reconcile_specular_lobes_noop_single_view() -> None:
    from abstract3d.gradient_compositing import reconcile_specular_lobes

    source, _, w_src, _, valid, observed, _ = _lobe_scene()
    out = reconcile_specular_lobes(
        view_rgb=[source],
        view_weight=[w_src],
        view_valid=[valid],
        observed_mask=observed,
        source_view_index=0,
        reference_resolution=source.shape[0],
    )
    assert out is None, "single view has no consensus and must no-op"


# ---------------------------------------------------------------------------
# reconcile_shadow_aprons (cycle-5: FACE-04/FACE-14 neck/jaw wash class)
# ---------------------------------------------------------------------------

def _shadow_scene(size: int = 200, *, source_sees: bool = True,
                  edge_dense: bool = False):
    """A reference-won apron the source photo reads darker: the reference
    carries flat lit skin; the source validly samples the same surface with
    a smooth cast-shadow field (its weight grazes to ~0 there, so the
    reference wins every apron texel)."""
    skin = np.zeros((size, size, 3), np.float32)
    skin[:] = (0.72, 0.56, 0.50)
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32)
    # smoothness and depth model the measured neck class: deviation ~-0.35
    # log (~30% luminance) varying over tens of texels, not a hard edge
    shadow_field = np.exp(-(((ys - 130) / 32.0) ** 2 + ((xs - 100) / 26.0) ** 2))

    source = np.clip(skin * (1.0 - 0.32 * shadow_field[:, :, None]), 0, 1)
    if edge_dense:
        checker = ((np.indices(source.shape[:2]).sum(axis=0) // 2) % 2
                   ).astype(np.float32)
        core = shadow_field > 0.3
        source[core] = np.clip(
            source[core] + 0.30 * (checker[core] - 0.5)[:, None], 0, 1)
    reference = skin.copy()

    # the reference wins the apron (the source's weight collapses there);
    # elsewhere the source is confident, giving the gauge its co-witnessed
    # bright population
    w_src = np.where(shadow_field > 0.05, 0.0, 0.6).astype(np.float32)
    w_ref = np.full((size, size), 0.30, np.float32)
    valid_src = np.ones((size, size), bool) if source_sees else shadow_field < 0.05
    valid = np.ones((size, size), bool)
    observed = np.ones((size, size), bool)
    positions = np.zeros((size, size, 4), np.float32)
    positions[:, :, 0] = xs / size
    positions[:, :, 1] = ys / size
    positions[:, :, 3] = 1.0
    return (source, reference, w_src, w_ref, valid_src, valid, observed,
            positions, shadow_field)


def test_reconcile_shadow_aprons_carries_source_shadow() -> None:
    from abstract3d.gradient_compositing import (
        apply_shadow_apron_scale,
        reconcile_shadow_aprons,
    )

    (source, reference, w_src, w_ref, valid_src, valid, observed,
     positions, shadow) = _shadow_scene()
    out = reconcile_shadow_aprons(
        view_rgb=[source, reference],
        view_weight=[w_src, w_ref],
        view_valid=[valid_src, valid],
        observed_mask=observed,
        positions=positions,
        source_view_index=0,
        reference_resolution=source.shape[0],
    )
    assert out is not None, "source-witnessed smooth shadow apron must qualify"
    baseline_log, blend_weight, stats = out
    corrected = apply_shadow_apron_scale(
        reference, valid,
        target_baseline_log=baseline_log, blend_weight=blend_weight,
        reference_resolution=source.shape[0],
    )
    core = shadow > 0.6
    src_lum = source.mean(axis=2)
    cor_lum = corrected.mean(axis=2)
    assert float(np.abs(cor_lum[core] - src_lum[core]).mean()) < 0.06, (
        "apron core must move to the source's shadow reading")
    rim = shadow < 0.02
    ref_lum = reference.mean(axis=2)
    assert float(np.abs(cor_lum[rim] - ref_lum[rim]).max()) < 1e-3, (
        "surround must stay untouched")
    assert bool((cor_lum <= ref_lum + 1e-6).all()), "correction is one-sided"


def test_reconcile_shadow_aprons_requires_source_evidence() -> None:
    """Where the source never validly samples the apron (parallax bands
    behind its own occluders), the reference's confident claim must stay:
    no witness demotion without source evidence."""
    from abstract3d.gradient_compositing import reconcile_shadow_aprons

    (source, reference, w_src, w_ref, valid_src, valid, observed,
     positions, shadow) = _shadow_scene(source_sees=False)
    out = reconcile_shadow_aprons(
        view_rgb=[source, reference],
        view_weight=[w_src, w_ref],
        view_valid=[valid_src, valid],
        observed_mask=observed,
        positions=positions,
        source_view_index=0,
        reference_resolution=source.shape[0],
    )
    assert out is None, "source-invalid apron must not be treated"


def test_reconcile_shadow_aprons_refuses_edge_dense_content() -> None:
    """A dark-reading source region that is EDGE-DENSE (hair strands, a
    necklace crossing the surface) is content disagreement, not a cast
    shadow, and must be refused."""
    from abstract3d.gradient_compositing import reconcile_shadow_aprons

    (source, reference, w_src, w_ref, valid_src, valid, observed,
     positions, shadow) = _shadow_scene(edge_dense=True)
    out = reconcile_shadow_aprons(
        view_rgb=[source, reference],
        view_weight=[w_src, w_ref],
        view_valid=[valid_src, valid],
        observed_mask=observed,
        positions=positions,
        source_view_index=0,
        reference_resolution=source.shape[0],
    )
    assert out is None, "edge-dense disagreement must be refused"


def test_reconcile_shadow_aprons_noop_single_view() -> None:
    from abstract3d.gradient_compositing import reconcile_shadow_aprons

    (source, _, w_src, _, valid_src, _, observed, positions, _) = _shadow_scene()
    out = reconcile_shadow_aprons(
        view_rgb=[source],
        view_weight=[w_src],
        view_valid=[valid_src],
        observed_mask=observed,
        positions=positions,
        source_view_index=0,
        reference_resolution=source.shape[0],
    )
    assert out is None, "single view: no reference can win, structural no-op"
