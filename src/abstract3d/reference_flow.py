"""Dense residual registration of reference views to the source's painted
truth on their shared-surface overlap.

Global similarity registration (width-profile matching, overlap similarity
search) leaves NON-RIGID per-feature residuals on generated geometry: the
nose, mouth and eyes each want a different small 2D correction (measured on
the face lane: nose -10 px, mouth (-4,+4), eyes (+4,0) at 512 — no global
transform satisfies all three), so reference photos paint ghost lip and
lash fragments next to the source's features. This module solves a SMALL,
SMOOTH, STRICTLY LOCAL displacement field over each reference's image
plane and warps the photo ONCE before projection:

  minimize  sum_x m(x) * rho( R(x + D(x)) - g * T(x) )
          + lambda * sum_nodes |L_n d|^2          (bending regularizer)

  R   reference photo (canonical frame, gain-corrected)
  T   target: source-witnessed texel colors splatted into the reference's
      image plane through the shared surface (first-surface visibility)
  m   evidence: source confidence x reference facing x splat density
  D   bilinear control lattice (coarse-to-fine 64/32/16 px spacing),
      Gauss-Newton with Charbonnier data term, magnitude cap 2% of frame
  L   normalized lattice Laplacian; bending (thin-plate) energy is zero on
      affine fields so residual global components are never shrunk

Safety architecture (each clause established by a measured failure):

  per-cell validation   a 48 px cell keeps its flow only when the warp
                        improves its weighted L1 by >= 20% AND its post-
                        warp error is within 1.25x the median of improving
                        cells (unreachable content — fabricated mirror
                        photos, stochastic hair — must not move).
  evidence leash        unvalidated cells adjacent to validated ones keep
                        flow only with substantive own evidence that does
                        not worsen; everything else decays to ZERO. Global
                        extension of band-fit corrections was measured
                        harmful twice (affine: side identity 0.706->0.587;
                        translation: hair moved, skin_in_hair at az+-135).
  reference facing      evidence is trusted only where the reference sees
                        the surface head-on (grazing texels compress in
                        its frame and amplify flow into smears).
  acceptance            the warp applies only when the overlap error
                        improves >= 2% with >= 3 validated cells; the photo
                        is resampled exactly once (flow upsampled to the
                        native canvas, single bilinear warp).

Validated on injected known warps (shift / rotation / barrel / local bump /
combined) at recovery <= 0.7 px median inside the evidence band, and gated
on full-bake improvement of the adversarial QA harnesses at 1024 and 2048.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np

__all__ = ["estimate_reference_flow", "solve_lattice_flow", "masked_error"]


# --------------------------------------------------------------------------
# geometry: texel -> reference-view image coordinates + visibility
# --------------------------------------------------------------------------

def project_texels_to_view(
    positions_texture: Any,
    *,
    azimuth_deg: float,
    elevation_deg: float,
    camera_distance: float,
    ortho_half_extent: float,
    canvas: int,
) -> Dict[str, Any]:
    """Per-texel image coords + depth in a view frame (projector formulas)."""
    from .backends.triposr_runtime import _tripo_camera_position, _tripo_look_at_matrix

    positions = np.asarray(positions_texture, dtype=np.float32)
    surface = positions[:, :, 3] > 0.0
    eye = _tripo_camera_position(
        azimuth_deg=float(azimuth_deg),
        elevation_deg=float(elevation_deg),
        camera_distance=float(camera_distance),
    )
    view = _tripo_look_at_matrix(
        eye, np.zeros(3, dtype=np.float32), np.array([0.0, 0.0, 1.0], dtype=np.float32)
    )
    homogeneous = np.concatenate(
        [positions[:, :, :3], np.ones((*positions.shape[:2], 1), dtype=np.float32)], axis=2
    )
    camera_space = homogeneous @ view.T
    ortho_scale = 0.5 * float(canvas) / max(float(ortho_half_extent), 1e-6)
    sample_x = ortho_scale * camera_space[:, :, 0] + float(canvas) / 2.0 - 0.5
    sample_y = -ortho_scale * camera_space[:, :, 1] + float(canvas) / 2.0 - 0.5
    depth = -camera_space[:, :, 2]
    in_frame = (
        surface
        & (sample_x >= 0.0)
        & (sample_x <= float(canvas - 1))
        & (sample_y >= 0.0)
        & (sample_y <= float(canvas - 1))
    )
    return {
        "sample_x": sample_x,
        "sample_y": sample_y,
        "depth": depth,
        "in_frame": in_frame,
        "surface": surface,
    }


def first_surface_visible(view_proj: Dict[str, Any], canvas: int) -> Any:
    """Projector-style first-surface test: bin z-buffer + 3x3 min filter."""
    sample_x, sample_y = view_proj["sample_x"], view_proj["sample_y"]
    depth, candidates = view_proj["depth"], view_proj["in_frame"]
    if not candidates.any():
        return np.zeros_like(candidates)
    bins_x = np.clip(np.round(sample_x).astype(np.int32), 0, canvas - 1)
    bins_y = np.clip(np.round(sample_y).astype(np.int32), 0, canvas - 1)
    nearest = np.full((canvas, canvas), np.inf, dtype=np.float32)
    np.minimum.at(nearest, (bins_y[candidates], bins_x[candidates]), depth[candidates])
    try:
        from scipy.ndimage import minimum_filter

        nearest = minimum_filter(nearest, size=3, mode="nearest")
    except Exception:
        pass
    spread = float(depth[candidates].max() - depth[candidates].min())
    epsilon = 0.01 * max(spread, 1e-6)
    visible = np.zeros_like(candidates)
    visible[candidates] = depth[candidates] <= nearest[bins_y[candidates], bins_x[candidates]] + epsilon
    return visible


def reference_facing(
    positions_texture: Any,
    normals_texture: Any,
    *,
    azimuth_deg: float,
    elevation_deg: float,
    camera_distance: float,
) -> Any:
    """Per-texel cosine of surface tilt toward the reference camera.

    A 2D correspondence is trustworthy only where the reference sees the
    surface reasonably head-on: grazing texels compress in its image plane,
    so flow there amplifies into visible smears (measured on the profile's
    own eye).
    """
    from .backends.triposr_runtime import _tripo_camera_position

    positions = np.asarray(positions_texture, dtype=np.float32)
    normals = np.asarray(normals_texture, dtype=np.float32)[:, :, :3]
    norm = np.linalg.norm(normals, axis=2, keepdims=True)
    normals_unit = np.divide(normals, np.maximum(norm, 1e-8))
    eye = _tripo_camera_position(
        azimuth_deg=float(azimuth_deg),
        elevation_deg=float(elevation_deg),
        camera_distance=float(camera_distance),
    )
    to_camera = eye[None, None, :] - positions[:, :, :3]
    distance = np.linalg.norm(to_camera, axis=2, keepdims=True)
    view_dir = np.divide(to_camera, np.maximum(distance, 1e-8))
    return np.sum(normals_unit * view_dir, axis=2)


# --------------------------------------------------------------------------
# target construction: splat source truth into the reference frame
# --------------------------------------------------------------------------

def splat_source_to_view(
    view_proj: Dict[str, Any],
    *,
    source_rgb: Any,
    source_weight: Any,
    canvas: int,
    grid: Optional[int] = None,
    min_source_weight: float = 0.25,
    fill_sigma: float = 1.0,
) -> Tuple[Any, Any]:
    """Bilinear splat of source-witnessed texel colors into the view frame.

    `grid` (defaults to canvas) sets the output resolution: splatting at
    the solve scale keeps texel density above one sample per pixel (a
    full-resolution splat dithers into holes). A weight-aware Gaussian
    (premultiplied colors / weight) closes residual sub-pixel holes without
    dragging the band edge. Texels hidden in this view are excluded up
    front (first-surface visibility).
    """
    from scipy.ndimage import gaussian_filter

    grid = int(grid or canvas)
    scale = grid / float(canvas)
    visible = first_surface_visible(view_proj, canvas)
    take = visible & (np.asarray(source_weight, dtype=np.float32) > float(min_source_weight))
    if not take.any():
        return np.zeros((grid, grid, 3), np.float32), np.zeros((grid, grid), np.float32)
    xs = view_proj["sample_x"][take] * scale
    ys = view_proj["sample_y"][take] * scale
    colors = np.asarray(source_rgb, dtype=np.float32)[take]
    weights = np.asarray(source_weight, dtype=np.float32)[take]

    accumulator = np.zeros((grid, grid, 3), np.float64)
    weight_sum = np.zeros((grid, grid), np.float64)
    x0 = np.floor(xs).astype(np.int32)
    y0 = np.floor(ys).astype(np.int32)
    fx = xs - x0
    fy = ys - y0
    for dx, dy, w in (
        (0, 0, (1 - fx) * (1 - fy)),
        (1, 0, fx * (1 - fy)),
        (0, 1, (1 - fx) * fy),
        (1, 1, fx * fy),
    ):
        cx = np.clip(x0 + dx, 0, grid - 1)
        cy = np.clip(y0 + dy, 0, grid - 1)
        contribution = w * weights
        np.add.at(weight_sum, (cy, cx), contribution)
        np.add.at(accumulator, (cy, cx), colors * contribution[:, None])
    accumulator = gaussian_filter(accumulator, (fill_sigma, fill_sigma, 0))
    weight_sum = gaussian_filter(weight_sum, fill_sigma)
    filled = weight_sum > 1e-6
    target = np.zeros((grid, grid, 3), np.float32)
    target[filled] = (accumulator[filled] / weight_sum[filled][:, None]).astype(np.float32)
    return target, weight_sum.astype(np.float32)


# --------------------------------------------------------------------------
# lattice flow solver
# --------------------------------------------------------------------------

def _bilinear_sample(image: Any, xs: Any, ys: Any) -> Any:
    h, w = image.shape[:2]
    xs = np.clip(xs, 0.0, w - 1.001)
    ys = np.clip(ys, 0.0, h - 1.001)
    x0 = xs.astype(np.int32)
    y0 = ys.astype(np.int32)
    fx = (xs - x0)[..., None] if image.ndim == 3 else xs - x0
    fy = (ys - y0)[..., None] if image.ndim == 3 else ys - y0
    c00 = image[y0, x0]
    c10 = image[y0, x0 + 1]
    c01 = image[y0 + 1, x0]
    c11 = image[y0 + 1, x0 + 1]
    return c00 * (1 - fx) * (1 - fy) + c10 * fx * (1 - fy) + c01 * (1 - fx) * fy + c11 * fx * fy


def _lattice_basis(shape_hw: Tuple[int, int], nodes_y: int, nodes_x: int):
    """Pixel -> (node indices, bilinear weights) for a regular lattice."""
    h, w = shape_hw
    gy = np.linspace(0.0, nodes_y - 1.0, h, dtype=np.float32)
    gx = np.linspace(0.0, nodes_x - 1.0, w, dtype=np.float32)
    gyy, gxx = np.meshgrid(gy, gx, indexing="ij")
    j0 = np.minimum(gyy.astype(np.int32), nodes_y - 2)
    i0 = np.minimum(gxx.astype(np.int32), nodes_x - 2)
    fy = gyy - j0
    fx = gxx - i0
    weights = np.stack([(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy], axis=0)
    node_ids = np.stack(
        [j0 * nodes_x + i0, j0 * nodes_x + i0 + 1,
         (j0 + 1) * nodes_x + i0, (j0 + 1) * nodes_x + i0 + 1],
        axis=0,
    )
    return node_ids, weights


def _lattice_laplacian(nodes_y: int, nodes_x: int):
    from scipy import sparse

    n = nodes_y * nodes_x
    rows, cols, vals = [], [], []

    def add(a: int, b: int) -> None:
        rows.extend([a, a, b, b])
        cols.extend([a, b, b, a])
        vals.extend([1.0, -1.0, 1.0, -1.0])

    for j in range(nodes_y):
        for i in range(nodes_x):
            k = j * nodes_x + i
            if i + 1 < nodes_x:
                add(k, k + 1)
            if j + 1 < nodes_y:
                add(k, k + nodes_x)
    return sparse.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()


def solve_lattice_flow(
    reference: Any,
    target: Any,
    weight: Any,
    *,
    node_spacings: Tuple[int, ...] = (64, 32, 16),
    smoothness: float = 0.35,
    charbonnier_delta: float = 0.02,
    cap_px: float = 12.0,
    iterations: int = 6,
    min_node_support: float = 4.0,
    anchor: float = 1e-4,
) -> Dict[str, Any]:
    """Coarse-to-fine Gauss-Newton lattice flow (see module docstring).

    reference/target: HxWxC float32 in [0,1] (same frame); weight: HxW.
    Returns dict with flow_x/flow_y (backward: warped(x) = ref(x + D(x)))
    and per-level stats. Node displacements are magnitude-capped; nodes
    without data support take the harmonic extension of supported ones
    (the caller's gate decides where any flow is finally applied).
    """
    from scipy import sparse
    from scipy.ndimage import gaussian_filter
    from scipy.sparse.linalg import spsolve

    h, w = reference.shape[:2]
    flow_y = np.zeros((h, w), np.float32)
    flow_x = np.zeros((h, w), np.float32)
    weight = np.asarray(weight, dtype=np.float32)
    mask = weight > 0
    if not mask.any():
        return {"flow_x": flow_x, "flow_y": flow_y, "levels": []}
    mean_weight = float(weight[mask].mean())

    yy, xx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")

    stats_levels = []
    for spacing in node_spacings:
        # Pyramid smoothing widens the linearization basin at coarse levels:
        # a node-spacing-proportional blur lets the coarse lattice lock onto
        # displacements beyond the raw gradients' ~2 px validity.
        sigma = max(float(spacing) / 8.0, 1.0)
        reference_level = gaussian_filter(reference, (sigma, sigma, 0))
        target_level = gaussian_filter(target, (sigma, sigma, 0))

        nodes_y = max(int(math.ceil(h / spacing)) + 1, 4)
        nodes_x = max(int(math.ceil(w / spacing)) + 1, 4)
        n = nodes_y * nodes_x
        node_ids, basis = _lattice_basis((h, w), nodes_y, nodes_x)
        laplacian = _lattice_laplacian(nodes_y, nodes_x)
        # Bending (thin-plate) regularizer: zero energy on affine fields, so
        # residual affine components are never shrunk; only curvature is
        # penalized. (A first-difference membrane penalty was measured to
        # bias rotation recovery by ~2 px.)
        degree = np.asarray(laplacian.diagonal()).ravel()
        normalized = sparse.diags(1.0 / np.maximum(degree, 1.0)) @ laplacian
        bending = (normalized.T @ normalized).tocsr()

        # current flow projected onto the lattice
        d_x = np.zeros(n, np.float64)
        d_y = np.zeros(n, np.float64)
        denominator = np.zeros(n, np.float64)
        for k in range(4):
            ids = node_ids[k][mask]
            np.add.at(denominator, ids, basis[k][mask])
            np.add.at(d_x, ids, basis[k][mask] * flow_x[mask])
            np.add.at(d_y, ids, basis[k][mask] * flow_y[mask])
        has = denominator > 1e-9
        d_x[has] /= denominator[has]
        d_y[has] /= denominator[has]

        support = np.zeros(n, np.float64)
        for k in range(4):
            np.add.at(support, node_ids[k][mask], (basis[k] * weight)[mask])

        lam = float(smoothness) * mean_weight * (spacing ** 2)
        for _ in range(int(iterations)):
            fx_full = np.zeros((h, w), np.float32)
            fy_full = np.zeros((h, w), np.float32)
            for k in range(4):
                fx_full += basis[k] * d_x[node_ids[k]].astype(np.float32)
                fy_full += basis[k] * d_y[node_ids[k]].astype(np.float32)
            warped = _bilinear_sample(reference_level, xx + fx_full, yy + fy_full)
            residual = warped - target_level
            gy, gx = np.gradient(warped, axis=(0, 1))
            rho = np.sqrt((residual ** 2).sum(axis=2) + charbonnier_delta ** 2)
            data_w = weight / np.maximum(rho, 1e-6)

            gxx_term = (gx * gx).sum(axis=2) * data_w
            gxy_term = (gx * gy).sum(axis=2) * data_w
            gyy_term = (gy * gy).sum(axis=2) * data_w
            bx_term = -(gx * residual).sum(axis=2) * data_w
            by_term = -(gy * residual).sum(axis=2) * data_w

            blocks: Dict[str, Tuple[list, list, list]] = {}
            rhs_x = np.zeros(n)
            rhs_y = np.zeros(n)
            for a in range(4):
                ids_a = node_ids[a][mask]
                wa = basis[a][mask]
                np.add.at(rhs_x, ids_a, wa * bx_term[mask])
                np.add.at(rhs_y, ids_a, wa * by_term[mask])
                for b in range(4):
                    ids_b = node_ids[b][mask]
                    wab = wa * basis[b][mask]
                    for name, term in (("xx", gxx_term), ("xy", gxy_term), ("yy", gyy_term)):
                        acc = blocks.setdefault(name, ([], [], []))
                        acc[0].append(ids_a)
                        acc[1].append(ids_b)
                        acc[2].append(wab * term[mask])

            def assemble(name: str):
                r, c, v = blocks[name]
                return sparse.coo_matrix(
                    (np.concatenate(v), (np.concatenate(r), np.concatenate(c))), shape=(n, n)
                ).tocsr()

            a_xx = assemble("xx") + lam * bending + anchor * sparse.identity(n)
            a_yy = assemble("yy") + lam * bending + anchor * sparse.identity(n)
            a_xy = assemble("xy")
            system = sparse.bmat([[a_xx, a_xy], [a_xy.T, a_yy]]).tocsr()
            rhs = np.concatenate([rhs_x - lam * (bending @ d_x), rhs_y - lam * (bending @ d_y)])
            try:
                delta = spsolve(system, rhs)
            except Exception:
                break
            if not np.all(np.isfinite(delta)):
                break
            d_x += delta[:n]
            d_y += delta[n:]
            magnitude = np.hypot(d_x, d_y)
            over = magnitude > cap_px
            if over.any():
                rescale = cap_px / magnitude[over]
                d_x[over] *= rescale
                d_y[over] *= rescale

        supported = support >= float(min_node_support)
        if supported.any() and (~supported).any():
            interior = ~supported
            l_ii = laplacian[interior][:, interior]
            l_ib = laplacian[interior][:, supported]
            regularized = l_ii + 1e-9 * sparse.identity(int(interior.sum()))
            try:
                d_x[interior] = spsolve(regularized, -l_ib @ d_x[supported])
                d_y[interior] = spsolve(regularized, -l_ib @ d_y[supported])
            except Exception:
                pass

        flow_x = np.zeros((h, w), np.float32)
        flow_y = np.zeros((h, w), np.float32)
        for k in range(4):
            flow_x += basis[k] * d_x[node_ids[k]].astype(np.float32)
            flow_y += basis[k] * d_y[node_ids[k]].astype(np.float32)
        stats_levels.append(
            {
                "spacing": spacing,
                "supported_nodes": int(supported.sum()),
                "total_nodes": int(n),
                "flow_p50_px": round(float(np.median(np.hypot(flow_x, flow_y)[mask])), 3),
            }
        )

    return {"flow_x": flow_x, "flow_y": flow_y, "levels": stats_levels}


def masked_error(
    reference: Any,
    target: Any,
    weight: Any,
    flow_x: Optional[Any] = None,
    flow_y: Optional[Any] = None,
) -> float:
    """Weighted mean |RGB| disagreement on the evidence band (optionally
    after warping the reference by the given backward flow)."""
    h, w = reference.shape[:2]
    if flow_x is not None:
        yy, xx = np.meshgrid(
            np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij"
        )
        warped = _bilinear_sample(reference, xx + flow_x, yy + flow_y)
    else:
        warped = reference
    mask = weight > 0
    if not mask.any():
        return float("nan")
    difference = np.abs(warped - target).mean(axis=2)
    return float((difference[mask] * weight[mask]).sum() / weight[mask].sum())


# --------------------------------------------------------------------------
# per-cell validation gate
# --------------------------------------------------------------------------

def _validation_gate(
    reference: Any,
    target: Any,
    weight: Any,
    flow_x: Any,
    flow_y: Any,
    *,
    cell: int = 48,
    blur_sigma: float = 12.0,
    min_cell_weight: float = 12.0,
    min_improvement: float = 0.20,
    absolute_factor: float = 1.25,
) -> Tuple[Any, Dict[str, Any]]:
    """Smooth [0,1] gate keeping flow only in cells where it is validated.

    Cell criteria: enough evidence mass; weighted L1 improves by
    >= min_improvement; post-warp error within absolute_factor of the
    median of improving cells (an unreachable target means the flow is
    chasing content that cannot match — moving pixels there only drags
    boundaries around). Unvalidated neighbors of validated cells keep flow
    only with substantive own evidence that does not worsen (thin slivers
    next to hair boundaries measured their improvement on a handful of
    pixels while dragging unwitnessed strands). Everything else — hair,
    far side, no-data territory — stays at identically ZERO displacement.
    A "flat cell" leash exemption and an NCC >= 0.35 cell criterion were
    both tried and measured harmful (strand drag across the ear; kept-set
    thinning that read as local distortions).
    """
    from scipy.ndimage import binary_dilation, gaussian_filter

    h, w = reference.shape[:2]
    yy, xx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
    warped = _bilinear_sample(reference, xx + flow_x, yy + flow_y)
    err_before_map = np.abs(reference - target).mean(axis=2) * weight
    err_after_map = np.abs(warped - target).mean(axis=2) * weight

    cells_y = max(int(math.ceil(h / cell)), 1)
    cells_x = max(int(math.ceil(w / cell)), 1)
    keep = np.zeros((cells_y, cells_x), np.float32)
    improving_after = []
    cache = []
    for j in range(cells_y):
        for i in range(cells_x):
            ys = slice(j * cell, min((j + 1) * cell, h))
            xs = slice(i * cell, min((i + 1) * cell, w))
            mass = float(weight[ys, xs].sum())
            if mass < min_cell_weight:
                cache.append((j, i, mass, None, None))
                continue
            before = float(err_before_map[ys, xs].sum()) / mass
            after = float(err_after_map[ys, xs].sum()) / mass
            cache.append((j, i, mass, before, after))
            if after < before * (1.0 - min_improvement):
                improving_after.append(after)
    median_after = float(np.median(improving_after)) if improving_after else np.inf
    kept = 0
    evaluated = 0
    for j, i, mass, before, after in cache:
        if before is None:
            continue
        evaluated += 1
        if after < before * (1.0 - min_improvement) and after <= absolute_factor * median_after:
            keep[j, i] = 1.0
            kept += 1

    validated = keep > 0.5
    leash_ring = binary_dilation(validated, iterations=1) & ~validated
    keep_leashed = validated.astype(np.float32)
    for j, i, mass, before, after in cache:
        if not leash_ring[j, i] or mass < 0.5 * min_cell_weight:
            continue
        ys = slice(j * cell, min((j + 1) * cell, h))
        xs = slice(i * cell, min((i + 1) * cell, w))
        cell_before = float(err_before_map[ys, xs].sum()) / mass
        cell_after = float(err_after_map[ys, xs].sum()) / mass
        if cell_after <= cell_before * 1.02:
            keep_leashed[j, i] = 1.0

    gate = np.repeat(np.repeat(keep_leashed, cell, axis=0), cell, axis=1)[:h, :w]
    if gate.shape != (h, w):  # ragged last cells
        padded = np.zeros((h, w), np.float32)
        padded[: gate.shape[0], : gate.shape[1]] = gate
        gate = padded
    gate = gaussian_filter(gate, blur_sigma)
    gate = np.clip(gate * 1.3 - 0.05, 0.0, 1.0)
    return gate.astype(np.float32), {"cells_kept": kept, "cells_evaluated": evaluated}


# --------------------------------------------------------------------------
# orchestration on a reference view
# --------------------------------------------------------------------------

def estimate_reference_flow(
    reference_rgba: Any,
    *,
    positions_texture: Any,
    source_projection: Dict[str, Any],
    azimuth_deg: float,
    elevation_deg: float,
    camera_distance: float,
    ortho_half_extent: float,
    normals_texture: Optional[Any] = None,
    solve_scale: int = 512,
    smoothness: float = 0.35,
    cap_frac: float = 0.02,
    min_rel_improvement: float = 0.02,
    min_cells_kept: int = 3,
    min_source_weight: float = 0.25,
    min_reference_facing: float = 0.30,
    full_reference_facing: float = 0.65,
) -> Tuple[Any, Dict[str, Any]]:
    """Estimate + apply dense residual flow for one reference view.

    reference_rgba: PIL RGBA in the canonical square canvas. Returns the
    (possibly once-warped) PIL image and a stats dict; on any rejection the
    input image is returned untouched.
    """
    from PIL import Image
    from scipy.ndimage import gaussian_filter, zoom

    image = reference_rgba.convert("RGBA") if hasattr(reference_rgba, "convert") else reference_rgba
    canvas = image.size[0]
    array = np.asarray(image, dtype=np.float32) / 255.0

    stats: Dict[str, Any] = {"applied": False, "reason": None}

    view_proj = project_texels_to_view(
        positions_texture,
        azimuth_deg=azimuth_deg,
        elevation_deg=elevation_deg,
        camera_distance=camera_distance,
        ortho_half_extent=ortho_half_extent,
        canvas=canvas,
    )
    evidence_weight = np.asarray(source_projection["weight"], dtype=np.float32)
    if normals_texture is not None:
        facing = reference_facing(
            positions_texture,
            normals_texture,
            azimuth_deg=azimuth_deg,
            elevation_deg=elevation_deg,
            camera_distance=camera_distance,
        )
        facing_gate = np.clip(
            (facing - float(min_reference_facing))
            / max(float(full_reference_facing) - float(min_reference_facing), 1e-6),
            0.0,
            1.0,
        )
        evidence_weight = evidence_weight * facing_gate
    target_small, weight_small = splat_source_to_view(
        view_proj,
        source_rgb=np.asarray(source_projection["rgba"], dtype=np.float32)[:, :, :3],
        source_weight=evidence_weight,
        canvas=canvas,
        grid=int(solve_scale),
        min_source_weight=min_source_weight,
    )
    if weight_small.max() <= 0:
        stats["reason"] = "no_overlap"
        return image, stats

    factor = canvas // int(solve_scale)
    if factor > 1:
        small = zoom(array, (1.0 / factor, 1.0 / factor, 1.0), order=1)
    else:
        factor = 1
        small = array

    ref_alpha = small[:, :, 3]
    ref_rgb = small[:, :, :3]
    weight_small = weight_small * (ref_alpha > 0.5)

    if weight_small.max() > 0:
        weight_small = np.clip(
            weight_small / np.percentile(weight_small[weight_small > 0], 60), 0.0, 1.0
        )
    weight_small = gaussian_filter(weight_small, 1.0)
    band = weight_small > 0.15
    if int(band.sum()) < 400:
        stats["reason"] = "band_too_small"
        return image, stats
    weight_small = np.where(band, weight_small, 0.0)

    # Per-channel gain reference->target (robust median): the flow must
    # register geometry, not chase the exposure difference the pipeline's
    # harmonization stage owns.
    gains = []
    for channel in range(3):
        ref_vals = ref_rgb[:, :, channel][band]
        target_vals = target_small[:, :, channel][band]
        usable = ref_vals > 0.05
        if not usable.any():
            gains.append(1.0)
            continue
        ratio = np.median(target_vals[usable] / np.maximum(ref_vals[usable], 0.05))
        gains.append(float(np.clip(ratio, 0.6, 1.6)))
    ref_gained = np.clip(ref_rgb * np.asarray(gains, dtype=np.float32)[None, None, :], 0.0, 1.0)

    cap_px = float(cap_frac) * small.shape[0]
    solved = solve_lattice_flow(
        ref_gained, target_small, weight_small, smoothness=smoothness, cap_px=cap_px
    )
    gate, gate_stats = _validation_gate(
        ref_gained, target_small, weight_small, solved["flow_x"], solved["flow_y"]
    )
    flow_x = solved["flow_x"] * gate
    flow_y = solved["flow_y"] * gate

    err_before = masked_error(ref_gained, target_small, weight_small)
    err_after = masked_error(ref_gained, target_small, weight_small, flow_x, flow_y)
    flow_magnitude = np.hypot(flow_x, flow_y)
    stats.update(
        {
            "gains": [round(g, 4) for g in gains],
            "err_before": round(err_before, 5),
            "err_after": round(err_after, 5) if np.isfinite(err_after) else None,
            "flow_p50_px": round(float(np.median(flow_magnitude[band])), 2),
            "flow_max_px": round(float(flow_magnitude.max()), 2),
            "band_px": int(band.sum()),
            "cells_kept": gate_stats["cells_kept"],
            "cells_evaluated": gate_stats["cells_evaluated"],
        }
    )
    if (
        not np.isfinite(err_after)
        or err_after > err_before * (1.0 - float(min_rel_improvement))
        or gate_stats["cells_kept"] < int(min_cells_kept)
    ):
        stats["reason"] = "no_improvement"
        return image, stats

    if factor > 1:
        flow_x_full = zoom(flow_x, (factor, factor), order=1) * factor
        flow_y_full = zoom(flow_y, (factor, factor), order=1) * factor
    else:
        flow_x_full, flow_y_full = flow_x, flow_y
    yy, xx = np.meshgrid(
        np.arange(canvas, dtype=np.float32), np.arange(canvas, dtype=np.float32), indexing="ij"
    )
    warped = _bilinear_sample(array, xx + flow_x_full, yy + flow_y_full)
    warped_image = Image.fromarray(np.clip(warped * 255.0, 0, 255).astype(np.uint8), mode="RGBA")
    stats["applied"] = True
    stats["reason"] = "accepted"
    return warped_image, stats
