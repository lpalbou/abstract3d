"""Gradient-domain (screened Poisson) view compositing on the texel surface graph.

Replaces the additive patch stack (softmax color blend -> per-region seam
leveling offsets) with ONE principled composite. Per RGB channel, over the
surface texel graph G = (V, E):

    E(x) = sum_{(i,j) in E} ((x_i - x_j) - g_ij)^2  +  sum_i lambda_i (x_i - c_i)^2

* V: atlas texels with rasterized surface positions.
* E: UV-grid 4-neighbor edges within charts (guarded by a 3D distance test
  so packed-chart neighbors are never connected) plus KD chart-stitch edges
  that reconnect chart borders to their true 3D continuation on other
  charts, so the solve sees ONE closed surface instead of per-chart islands.
* g_ij: the target gradient, composited per edge from the views. The most
  confident COMMON witness (both endpoints seen by that view) supplies its
  own gradient verbatim, so every real edge (lips, lashes, panel lines)
  survives; winner-take-all handoffs with no common witness fall back to a
  one-sided witness where a view validly sampled both endpoints, and to a
  zero-gradient membrane bridge otherwise. Mirror-completed and
  synthesized-fill texels contribute the gradients of their completed
  colors, so completion content rides into the same solve unchanged.
* lambda_i: screening (soft Dirichlet) weight proportional to the blend
  confidence: where a photo saw the surface well the solve may only move
  its color by a smooth low-amplitude field; weakly witnessed and completed
  texels are dictated by gradients.

Why this eliminates tone seams mathematically: an exposure difference
between two views is (locally) a constant error field, and constants vanish
under the gradient operator — no target gradient anywhere encodes the step,
so the minimizer spreads the disagreement as a smooth ramp with decay
length 1/sqrt(lambda) texels while every witnessed edge is preserved. This
is the screened-Poisson generalization of texture-atlas seam leveling
(Perez et al. gradient-domain compositing + Ivanov/Lempitsky seam
leveling), with no region map, no offset cap, and no boundary-step
heuristics.

Solver: the normal equations (L + Lambda) x = b are symmetric positive
definite. Plain Jacobi-CG needs ~1000 iterations for the smooth
equalization modes, so CG is preconditioned with one V(1,1)-cycle of a
geometric-aggregation multigrid built by voxel-clustering the texels' 3D
positions (Galerkin coarse operators, damped-Jacobi smoothing, direct
coarsest solve); the whole spectrum then converges in a few tens of
iterations. All 3 channels share the matrix and are solved as one (N, 3)
block. Everything is deterministic for fixed inputs.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


def build_texel_surface_graph(
    positions_texture: Any,
    *,
    normals_texture: Optional[Any] = None,
    stitch_chart_borders: bool = True,
    max_grid_jump_pitches: float = 4.0,
    stitch_radius_pitches: float = 2.0,
    stitch_neighbors: int = 4,
    min_stitch_normal_dot: float = 0.5,
) -> Optional[Dict[str, Any]]:
    """Build the surface texel graph: UV-grid edges + chart-stitch edges.

    Grid edges connect UV-adjacent surface texels whose 3D separation stays
    below `max_grid_jump_pitches` times the texel pitch (UV neighbors that
    straddle a packed chart boundary are 3D-distant and must not couple).
    Border texels (surface texels missing at least one 4-neighbor) are
    reconnected in 3D to their nearest surface texels across the chart cut.
    Stitch candidates must lie within `stitch_radius_pitches` * pitch and,
    when normals are available, agree in orientation
    (`min_stitch_normal_dot`), which prevents stitching two distinct
    thin-shell sheets that pass close in space (hair films over foreheads).

    Returns a dict with `index_map` (H, W texel -> node id, -1 off
    surface), `nodes_rc` (N, 2 row/col), `edges` (M, 2 deduplicated node
    pairs), `edge_kind` (M, 0 = grid / 1 = stitch), `pitch`, `node_count`;
    or None when scipy is unavailable and stitching was requested.
    """
    import numpy as np

    positions = np.asarray(positions_texture, dtype=np.float32)
    surface = positions[:, :, 3] > 0.0
    height, width = surface.shape
    xyz = positions[:, :, :3]

    index_map = np.full((height, width), -1, dtype=np.int32)
    rows, cols = np.nonzero(surface)
    node_count = len(rows)
    if node_count == 0:
        return None
    index_map[rows, cols] = np.arange(node_count, dtype=np.int32)
    nodes_rc = np.stack([rows, cols], axis=1).astype(np.int32)

    # --- UV-grid edges (right and down neighbors) -------------------------
    pair_h = surface[:, :-1] & surface[:, 1:]
    dist_h = np.linalg.norm(xyz[:, 1:] - xyz[:, :-1], axis=2)
    pitch = float(np.median(dist_h[pair_h])) + 1e-12 if pair_h.any() else 1e-3
    jump = float(max_grid_jump_pitches) * pitch

    edges_list: List[Any] = []
    ok_h = pair_h & (dist_h <= jump)
    if ok_h.any():
        edges_list.append(
            np.stack([index_map[:, :-1][ok_h], index_map[:, 1:][ok_h]], axis=1)
        )
    pair_v = surface[:-1, :] & surface[1:, :]
    dist_v = np.linalg.norm(xyz[1:, :] - xyz[:-1, :], axis=2)
    ok_v = pair_v & (dist_v <= jump)
    if ok_v.any():
        edges_list.append(
            np.stack([index_map[:-1, :][ok_v], index_map[1:, :][ok_v]], axis=1)
        )
    grid_edges = (
        np.concatenate(edges_list, axis=0)
        if edges_list
        else np.zeros((0, 2), dtype=np.int32)
    )

    # --- chart-stitch edges ------------------------------------------------
    stitch_edges = np.zeros((0, 2), dtype=np.int32)
    if stitch_chart_borders:
        try:
            from scipy.spatial import cKDTree
        except Exception:
            return None
        # A border texel has at least one missing or jump-rejected UV
        # neighbor: exactly where the atlas cut a chart.
        has_left = np.zeros_like(surface)
        has_right = np.zeros_like(surface)
        has_up = np.zeros_like(surface)
        has_down = np.zeros_like(surface)
        has_right[:, :-1] = ok_h
        has_left[:, 1:] = ok_h
        has_down[:-1, :] = ok_v
        has_up[1:, :] = ok_v
        border = surface & ~(has_left & has_right & has_up & has_down)
        border_idx = index_map[border]
        if len(border_idx) > 0:
            all_points = xyz[rows, cols]
            tree = cKDTree(all_points)
            k = int(stitch_neighbors) + 1
            distances, neighbors = tree.query(
                all_points[border_idx], k=k, workers=-1,
                distance_upper_bound=float(stitch_radius_pitches) * pitch,
            )
            distances = np.atleast_2d(distances)
            neighbors = np.atleast_2d(neighbors)
            src = np.repeat(border_idx, k)
            dst = neighbors.reshape(-1)
            dist = distances.reshape(-1)
            valid = np.isfinite(dist) & (dst < node_count)
            src, dst = src[valid], dst[valid].astype(np.int32)
            not_self = src != dst
            src, dst = src[not_self], dst[not_self]
            # Pairs already UV-adjacent (|dr| + |dc| <= 1) belong to the
            # grid pass; keeping them here would double their edge weight.
            d_rc = np.abs(
                nodes_rc[src].astype(np.int64) - nodes_rc[dst].astype(np.int64)
            )
            far_in_uv = (d_rc[:, 0] + d_rc[:, 1]) > 1
            src, dst = src[far_in_uv], dst[far_in_uv]
            if normals_texture is not None and len(src) > 0:
                normals = np.asarray(normals_texture, dtype=np.float32)[:, :, :3]
                norm = np.linalg.norm(normals, axis=2, keepdims=True)
                normals = np.divide(normals, np.maximum(norm, 1e-8))
                flat_normals = normals[rows, cols]
                agree = (flat_normals[src] * flat_normals[dst]).sum(axis=1)
                keep = agree >= float(min_stitch_normal_dot)
                src, dst = src[keep], dst[keep]
            if len(src) > 0:
                stitch_edges = np.stack([src, dst], axis=1).astype(np.int32)

    edges = np.concatenate([grid_edges, stitch_edges], axis=0)
    if len(edges) == 0:
        return None
    kind = np.concatenate([
        np.zeros(len(grid_edges), dtype=np.uint8),
        np.ones(len(stitch_edges), dtype=np.uint8),
    ])
    # Symmetric dedup (stitches are discovered from both endpoints).
    lo = np.minimum(edges[:, 0], edges[:, 1])
    hi = np.maximum(edges[:, 0], edges[:, 1])
    key = lo.astype(np.int64) * np.int64(node_count) + hi.astype(np.int64)
    order = np.argsort(key, kind="stable")
    key_sorted = key[order]
    first = np.ones(len(key_sorted), dtype=bool)
    first[1:] = key_sorted[1:] != key_sorted[:-1]
    sel = order[first]
    return {
        "index_map": index_map,
        "nodes_rc": nodes_rc,
        "edges": np.stack([lo[sel], hi[sel]], axis=1),
        "edge_kind": kind[sel],
        "pitch": pitch,
        "node_count": node_count,
    }


def select_composite_gradients(
    graph: Mapping[str, Any],
    *,
    view_rgb: Sequence[Any],
    view_weight: Sequence[Any],
    class_map: Any,
    filled_rgb: Any,
    view_valid: Optional[Sequence[Any]] = None,
    min_witness_weight: float = 0.05,
    rule: str = "max_confidence",
    softmax_sharpness: float = 8.0,
) -> Any:
    """Composite the per-edge target gradients g_ij from the views.

    Per edge (i, j), every view v with weight >= `min_witness_weight` at
    BOTH endpoints is a witness of quality q_v = min(w_v[i], w_v[j]) and
    proposes g = rgb_v[i] - rgb_v[j]. `rule` picks the winner:

    * "max_confidence": the best witness's gradient verbatim. No cross-view
      averaging can ghost misregistered detail, so edges stay photo-crisp.
    * "weighted": softmax(q)-weighted witness average (measured on the face
      lane: nearly identical seam metrics, slightly softer detail; kept for
      ablations).

    Observed edges with NO common witness are exactly the winner-take-all
    handoffs (conflict resolution zeroes the losing view where views
    disagreed, and facing gates end coverage). There a ONE-SIDED witness is
    used when available: a view confident on one endpoint whose projection
    validly SAMPLED the other (`view_valid`, the sampled photo alpha —
    weight may be zero there). Real content edges that cross a handoff
    (lip borders, panel lines) survive this way; the sampled-but-zeroed
    colors only ever contribute a color DIFFERENCE, never an absolute
    color. When not even a confident one-sided witness exists, any view
    with POSITIVE weight on one endpoint and valid samples on both is
    accepted as a last-tier witness (rim texels carry small weights but
    their sampled differences are still real photo content, and a wrong
    zero target acts as an error dipole that weak screening spreads over
    1/sqrt(lambda) texels). Only with no witness at all does the target
    fall to 0 and the membrane term bridges the handoff smoothly — the
    seam-killing behavior.

    Class handling (class_map: 0 observed, 1 mirror-completed, 2 fill,
    < 0 off-surface): every edge with at least one completion endpoint —
    completion-internal AND observed|completion borders — takes the
    gradient of `filled_rgb` (the completed texture). The completion
    stages already build content that is C0 against its observed boundary
    (harmonic interpolation, surface smoothing, zero-mean detail), so the
    completed texture is the correct witness of its own transitions.
    A zero-gradient membrane at those borders was measured to FLATTEN
    fragmented regions instead: gated coverage splinters ears/hairlines
    into thousands of tiny completion islands (median 4 texels on the
    face lane), each border edge pulled both sides toward equality, and
    the ear folds washed out (side-profile identity SSIM 0.703 -> 0.606).
    Re-deriving border gradients from a one-sided photo witness was
    rejected on causality grounds: many completion texels exist BECAUSE
    the photo content there was judged untrustworthy (outlier drop,
    layered-zone mixture), so the photo must not re-imprint them.

    Returns (gradients, line_mask): the (M, 3) float32 target gradients and
    a (M,) bool mask of LINE-SUPPORTED constraint edges, both aligned with
    graph["edges"]. Area edges sample a smooth target field (their count
    per world area scales with resolution², like the anchors); line edges
    carry boundary data (winner-take-all handoffs, class borders) whose
    count scales with resolution ONLY — the caller must down-weight them
    by (reference/resolution) or their influence relative to everything
    else doubles per resolution octave (measured: the mid-face chroma seam
    that the solve eliminates at 1024 reappeared at 2048 through exactly
    this imbalance).
    """
    import numpy as np

    edges = graph["edges"]
    nodes_rc = graph["nodes_rc"]
    node_rows = nodes_rc[:, 0]
    node_cols = nodes_rc[:, 1]
    rows_a = node_rows[edges[:, 0]]
    cols_a = node_cols[edges[:, 0]]
    rows_b = node_rows[edges[:, 1]]
    cols_b = node_cols[edges[:, 1]]

    classes = np.asarray(class_map)
    class_a = classes[rows_a, cols_a]
    class_b = classes[rows_b, cols_b]

    gradients = np.zeros((len(edges), 3), dtype=np.float32)
    line_mask = np.zeros(len(edges), dtype=bool)

    both_observed = (class_a == 0) & (class_b == 0)
    if both_observed.any() and len(view_rgb) > 0:
        sub = np.nonzero(both_observed)[0]
        ra, ca, rb, cb = rows_a[sub], cols_a[sub], rows_b[sub], cols_b[sub]
        quality = np.zeros((len(view_rgb), len(sub)), dtype=np.float32)
        for v, weight in enumerate(view_weight):
            w = np.asarray(weight, dtype=np.float32)
            qa = w[ra, ca]
            qb = w[rb, cb]
            q = np.minimum(qa, qb)
            q[(qa < min_witness_weight) | (qb < min_witness_weight)] = 0.0
            quality[v] = q
        best = quality.max(axis=0)
        has_witness = best > 0.0
        if rule == "weighted":
            weights = np.exp(
                np.clip(float(softmax_sharpness) * (quality - best[None, :]), -30.0, 0.0)
            ) * (quality > 0.0)
            total = weights.sum(axis=0)
            accum = np.zeros((len(sub), 3), dtype=np.float32)
            for v, rgb in enumerate(view_rgb):
                colors = np.asarray(rgb, dtype=np.float32)[:, :, :3]
                accum += weights[v][:, None] * (colors[ra, ca] - colors[rb, cb])
            with np.errstate(invalid="ignore"):
                accum = np.where(
                    (total > 0)[:, None], accum / np.maximum(total, 1e-9)[:, None], 0.0
                )
            accum[~has_witness] = 0.0
            gradients[sub] = accum
        else:
            winner = quality.argmax(axis=0)
            g_sel = np.zeros((len(sub), 3), dtype=np.float32)
            for v, rgb in enumerate(view_rgb):
                take = has_witness & (winner == v)
                if not take.any():
                    continue
                colors = np.asarray(rgb, dtype=np.float32)[:, :, :3]
                g_sel[take] = colors[ra[take], ca[take]] - colors[rb[take], cb[take]]
            gradients[sub] = g_sel

        # Everything without a common witness is handoff-line data.
        line_mask[sub[~has_witness]] = True
        if (~has_witness).any():
            orphan = np.nonzero(~has_witness)[0]
            o_ra, o_ca = ra[orphan], ca[orphan]
            o_rb, o_cb = rb[orphan], cb[orphan]
            one_quality = np.zeros((len(view_rgb), len(orphan)), dtype=np.float32)
            for v, weight in enumerate(view_weight):
                w = np.asarray(weight, dtype=np.float32)
                if view_valid is not None:
                    valid = np.asarray(view_valid[v], dtype=bool)
                else:
                    valid = w > 0.0
                qa = w[o_ra, o_ca]
                qb = w[o_rb, o_cb]
                va = valid[o_ra, o_ca]
                vb = valid[o_rb, o_cb]
                both_valid = va & vb
                strength = np.maximum(qa, qb)
                # Tier 1: confident on one endpoint (>= min_witness_weight);
                # tier 2: any positive weight (rim texels) — ranked strictly
                # below every tier-1 witness by the 1e-3 scale.
                confident = strength >= min_witness_weight
                positive = strength > 0.0
                one_quality[v] = np.where(
                    both_valid & confident, strength,
                    np.where(both_valid & positive, 1e-3 * strength, 0.0),
                )
            one_winner = one_quality.argmax(axis=0)
            one_best = one_quality.max(axis=0)
            usable_any = one_best > 0.0
            g_one = np.zeros((len(orphan), 3), dtype=np.float32)
            for v, rgb in enumerate(view_rgb):
                take = usable_any & (one_winner == v)
                if not take.any():
                    continue
                colors = np.asarray(rgb, dtype=np.float32)[:, :, :3]
                g_one[take] = (
                    colors[o_ra[take], o_ca[take]] - colors[o_rb[take], o_cb[take]]
                )
            gradients[sub[orphan]] = g_one

    filled = np.asarray(filled_rgb, dtype=np.float32)[:, :, :3]
    touches_completion = (class_a > 0) | (class_b > 0)
    if touches_completion.any():
        sub = np.nonzero(touches_completion)[0]
        gradients[sub] = (
            filled[rows_a[sub], cols_a[sub]] - filled[rows_b[sub], cols_b[sub]]
        )
    # Class borders (observed|mirror, observed|fill, mirror|fill) are lines.
    line_mask[class_a != class_b] = True
    return gradients, line_mask


def _masked_atlas_gaussian(field: Any, mask: Any, sigma: float) -> Any:
    """Gaussian smoothing of `field` restricted to `mask` (density-normalized)."""
    import numpy as np
    from scipy.ndimage import gaussian_filter

    weights = mask.astype(np.float32)
    density = gaussian_filter(weights, sigma)
    smoothed = gaussian_filter(field * weights, sigma)
    return np.where(density > 1e-6, smoothed / np.maximum(density, 1e-6), 0.0)


def reconcile_specular_lobes(
    *,
    view_rgb: Sequence[Any],
    view_weight: Sequence[Any],
    view_valid: Sequence[Any],
    observed_mask: Any,
    source_view_index: int = 0,
    min_winner_weight: float = 0.20,
    sigma_base: float = 12.0,
    sigma_lobe: float = 4.0,
    sigma_pool: float = 8.0,
    tau_log: float = 0.06,
    tau_sat: float = 0.015,
    gauge_margin: float = 0.08,
    pair_density_min: float = 0.04,
    vote_ratio_min: float = 0.45,
    fill_in_share: float = 0.30,
    edge_p85_max: float = 1.8,
    min_component_texels: int = 30,
    min_other_luminance: float = 0.25,
    min_base_luminance: float = 0.30,
    saturation_boost_max: float = 1.6,
    dark_standoff_texels: float = 8.0,
    dark_content_log: float = -0.12,
    reference_resolution: int = 1024,
) -> Optional[Tuple[Any, Dict[str, Any]]]:
    """Cross-view diffuse-consensus reconciliation of the source view's
    baked specular lobes (FACE-05 "pale seam column" class).

    THE DEFECT: a photo's own view-dependent shading — most visibly the
    nose-ridge specular highlight — is real witnessed content at the
    photo's pose, but it is LIGHT, not albedo. Projected under an
    estimated head turn it lands on surface that other poses render
    elsewhere (a +20 deg source pose paints the ridge highlight onto the
    nose FLANK), where it reads as a pale desaturated column at az0. The
    screened-Poisson composite faithfully preserves it: the source is the
    most confident common witness, so its gradients AND its anchors carry
    the lobe (measured: the column is fully present in the pre-solve
    blend; solve delta within +-5/255; rails/membrane count inside the
    column is ZERO — this is NOT a membrane or selection-boundary
    artifact).

    THE PRINCIPLE: albedo is Lambertian-consistent across views; a smooth
    positive luminance deviation of the winner that (a) carries the
    specular signature (brighter AND less saturated than its own local
    surround at shading scale, on a bright base material) and (b) is
    read DARKER by another view's valid sample of the same surface
    beyond that pair's global lighting gauge, is view-dependent shading
    the composite may not bake. Where both views agree the surface is
    bright (the ridge line itself under both lights), consensus keeps
    it — only the parallax-displaced lobe body is reconciled.

    THE CORRECTION rebuilds lobe texels from the winner's OWN local
    surround: luminance target = surround baseline + the texel's own
    log-detail (micro-texture preserved verbatim), saturation restored
    toward the surround's level (specular adds white light; capped by
    `saturation_boost_max`). No other view's color is ever imported, so
    misregistered reference content cannot leak through this path.

    Scope guards (each with a measured counterexample):
    - SOURCE VIEW ONLY: reference-view lobes measured -0.005 side-profile
      identity SSIM with no ledger-visible gain (their lobes render at
      poses the battery scores against the same photo that carries them).
    - Cross-view authorization is a pooled per-texel field (votes among
      valid darker-reading samples, Gaussian-pooled at `sigma_pool`),
      with `fill_in_share` component fill-in: one physical lobe is one
      phenomenon, but its top (glabella) may be witnessed by no second
      view — a component >= 30% authorized qualifies entirely. Zero
      authorization (single-view bakes structurally) => no-op.
    - Feature refusal: components whose 2-dilated ring carries strong
      own-photo edges (Scharr p85 >= `edge_p85_max`) are refused —
      sclera/teeth/eye corners are bright+desaturated but edge-dense,
      while a shading lobe is smooth by construction.
    - `min_base_luminance` keeps dark-material context out (hair sheen
      belongs to the film/band machinery).
    - DARK-CONTENT STANDOFF (`dark_standoff_texels`): leveling the bright
      base around pre-existing dark micro-content UNMASKS it — dark
      chips that hid inside the bright lobe read as fresh dark islands
      against the corrected surround (the debris counter re-fires; the
      trace-commit lane measured the same economics for partial
      cleanups). The correction weight feathers to zero near texels
      substantially darker than the surround baseline, so the leveling
      never manufactures dark-island contrast (measured at 2048:
      dark_debris 0.0040/0.0036 at az0/-22.5 without the standoff vs
      0.0014/0.0021 baseline; the column fix itself does not need the
      dark-adjacent zone — its body is smooth skin).

    Returns (delta_rgb, stats) where delta applies to the source view's
    rgb (and, share-scaled, to the blend anchors), or None when nothing
    qualifies.
    """
    import numpy as np
    from scipy.ndimage import binary_dilation, convolve1d
    from scipy.ndimage import label as cc_label

    weights = [np.asarray(w, dtype=np.float32) for w in view_weight]
    valids = [np.asarray(v, dtype=bool) for v in view_valid]
    rgbs = [np.asarray(r, dtype=np.float32)[:, :, :3] for r in view_rgb]
    if len(rgbs) < 2:
        return None
    observed = np.asarray(observed_mask, dtype=bool)
    resolution = observed.shape[0]
    s = resolution / float(reference_resolution)

    weight_stack = np.stack(weights)
    winner = weight_stack.argmax(axis=0)
    winner_weight = weight_stack.max(axis=0)

    eps = 0.02

    def lum(a: Any) -> Any:
        return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]

    def sat(a: Any) -> Any:
        return a.max(axis=2) - a.min(axis=2)

    win_idx = int(source_view_index)
    if not (0 <= win_idx < len(rgbs)):
        return None
    domain = (
        observed & (winner == win_idx)
        & (winner_weight >= float(min_winner_weight)) & valids[win_idx]
    )
    if int(domain.sum()) < 500:
        return None
    rgb_v = rgbs[win_idx]
    lum_v = lum(rgb_v)
    sat_v = sat(rgb_v)
    log_lum = np.log(np.clip(lum_v, 0.0, 1.0) + eps)

    # Robust surround baseline: exclude bright candidates AND strong dark
    # deviants (brows/nostril shadows) so the baseline tracks the mid-tone
    # skin, not the features.
    base_log = _masked_atlas_gaussian(log_lum, domain, sigma_base * s)
    bright0 = domain & (log_lum - base_log > tau_log)
    dark0 = domain & (log_lum - base_log < -0.10)
    keep_mask = domain & ~bright0 & ~dark0
    base_log2 = _masked_atlas_gaussian(log_lum, keep_mask, sigma_base * s)
    base_sat = _masked_atlas_gaussian(sat_v, keep_mask, sigma_base * s)
    delta_log = log_lum - base_log2
    candidates = (
        domain & (delta_log > tau_log) & (base_sat - sat_v > tau_sat)
        & (np.exp(base_log2) - eps > float(min_base_luminance))
    )
    if not candidates.any():
        return None

    gx = convolve1d(convolve1d(lum_v, [1.0, 0.0, -1.0], axis=1), [3.0, 10.0, 3.0], axis=0)
    gy = convolve1d(convolve1d(lum_v, [1.0, 0.0, -1.0], axis=0), [3.0, 10.0, 3.0], axis=1)
    # Per-texel gradient response scales inversely with resolution (a
    # fixed-world-size edge spreads over s times more texels), so the
    # edge statistic is normalized back to the reference resolution:
    # without this the 1024-calibrated feature gate stops refusing
    # eye-adjacent components at 2048 (measured: an eye-corner component
    # at edge_p85 1.45*s slipped past the 1.8 bar and its leveling
    # unmasked lash-adjacent dark content into the debris counter).
    edge = np.where(domain, np.hypot(gx, gy) * s, 0.0)

    votes = np.zeros(candidates.shape, dtype=bool)
    paired = np.zeros(candidates.shape, dtype=bool)
    for v in range(len(rgbs)):
        if v == win_idx:
            continue
        other_lum = lum(rgbs[v])
        other_log = np.log(np.clip(other_lum, 0.0, 1.0) + eps)
        both_confident = observed & (weights[win_idx] >= 0.25) & (weights[v] >= 0.25)
        gauge = (
            float(np.median((log_lum - other_log)[both_confident]))
            if int(both_confident.sum()) >= 200 else 0.0
        )
        pair = domain & valids[v] & (other_lum > float(min_other_luminance))
        votes |= pair & ((log_lum - other_log) - gauge > float(gauge_margin))
        paired |= pair

    vote_density = _masked_atlas_gaussian(votes.astype(np.float32), domain, sigma_pool * s)
    pair_density = _masked_atlas_gaussian(paired.astype(np.float32), domain, sigma_pool * s)
    authorized = (pair_density > float(pair_density_min)) & (
        vote_density / np.maximum(pair_density, 1e-6) > float(vote_ratio_min)
    )

    min_component = max(4, int(round(float(min_component_texels) * s * s)))
    final_candidates = candidates & authorized
    labels_all, _ = cc_label(binary_dilation(candidates, iterations=2))
    labels_all = np.where(candidates, labels_all, 0)
    for i in np.unique(labels_all[labels_all > 0]):
        component = labels_all == i
        n_component = int(component.sum())
        if n_component >= min_component and int(
            (final_candidates & component).sum()
        ) >= float(fill_in_share) * n_component:
            final_candidates |= component

    labels, _ = cc_label(binary_dilation(final_candidates, iterations=2))
    labels = np.where(final_candidates, labels, 0)
    kept = np.zeros(candidates.shape, dtype=bool)
    components: List[Dict[str, Any]] = []
    for i in np.unique(labels[labels > 0]):
        component = labels == i
        size = int(component.sum())
        if size < min_component:
            continue
        ring = binary_dilation(component, iterations=2) & domain
        edge_p85 = float(np.percentile(edge[ring], 85))
        if edge_p85 < float(edge_p85_max):
            kept |= component
            components.append({"texels": size, "edge_p85": round(edge_p85, 2)})
    if not kept.any():
        return None

    lobe = _masked_atlas_gaussian(np.where(kept, delta_log, 0.0), domain, sigma_lobe * s)
    lobe = np.where(domain, lobe, 0.0)
    smooth_log = _masked_atlas_gaussian(log_lum, domain, sigma_lobe * s)
    detail_log = log_lum - smooth_log
    target_lum = np.exp(base_log2 + detail_log) - eps
    lum_scale = np.clip(target_lum / np.maximum(lum_v, 1e-4), 0.25, 1.0)
    rescaled = rgb_v * lum_scale[..., None]
    rescaled_mean = rescaled.mean(axis=2, keepdims=True)
    rescaled_sat = rescaled.max(axis=2) - rescaled.min(axis=2)
    sat_scale = np.clip(
        base_sat / np.maximum(rescaled_sat, 1e-3), 1.0, float(saturation_boost_max)
    )
    target = np.clip(
        rescaled_mean + (rescaled - rescaled_mean) * sat_scale[..., None], 0.0, 1.0
    )
    blend_weight = np.where(domain, np.clip(lobe / float(tau_log), 0.0, 1.0), 0.0)
    # dark-content standoff: never level the bright base right against
    # pre-existing dark micro-content (see docstring)
    dark_content = domain & (delta_log < float(dark_content_log))
    if dark_content.any():
        from scipy.ndimage import distance_transform_edt

        dark_distance = distance_transform_edt(~dark_content)
        standoff = np.clip(
            dark_distance / max(float(dark_standoff_texels) * s, 1e-6), 0.0, 1.0
        ).astype(np.float32)
        blend_weight = blend_weight * standoff
    delta = (target - rgb_v) * blend_weight[..., None]
    stats = {
        "applied": True,
        "view": win_idx,
        "texels": int(kept.sum()),
        "components": components,
        "max_abs_delta": round(float(np.abs(delta).max()), 4),
    }
    return delta, stats


def reconcile_shadow_aprons(
    *,
    view_rgb: Sequence[Any],
    view_weight: Sequence[Any],
    view_valid: Sequence[Any],
    observed_mask: Any,
    positions: Optional[Any] = None,
    source_view_index: int = 0,
    min_winner_weight: float = 0.10,
    sigma_base: float = 4.0,
    sigma_pool: float = 8.0,
    gauge_margin: float = 0.10,
    pair_density_min: float = 0.04,
    vote_ratio_min: float = 0.45,
    edge_p85_max: float = 1.8,
    min_component_texels: int = 30,
    min_source_luminance: float = 0.35,
    min_winner_luminance: float = 0.35,
    scale_floor: float = 0.45,
    feather_texels: float = 3.0,
    merge_radius_ratio: float = 0.02,
    reference_resolution: int = 1024,
) -> Optional[Tuple[Any, Any, Dict[str, Any]]]:
    """Source-shading reconciliation of reference-won co-witnessed aprons —
    the DUAL of `reconcile_specular_lobes` (FACE-04/FACE-14 "neck/jaw tan
    wash" class).

    THE DEFECT: the mirror image of the specular-lobe class. There the
    SOURCE view's baked highlight (light, not albedo) won co-witnessed
    surface and other views read it darker; here a REFERENCE view wins
    co-witnessed surface with its own LIT reading while the source photo
    — the identity contract holder — validly samples the same surface
    substantially darker (its cast shadow: chin/jaw onto the neck).
    Measured on the face proof: the source's under-chin reading sits
    -0.35 log below side_right's at the neck apron against a -0.08
    global gauge, because the source's projection weight at the
    down-sloping neck is ~0 (grazing facing) and the lit reference wins
    every texel. The identity gate at the source pose compares that
    surface against the source photo, so the composite ships a flat lit
    "tan wash" where the photo shows a smooth shadow gradient.

    THE PRINCIPLE: where the source view VALIDLY samples bright surface
    that a reference wins, and the source's reading is systematically
    darker than the winner's beyond the pairwise lighting gauge, with the
    deviation SMOOTH at shading scale in the source's own photo (a cast
    shadow is smooth; strands, necklaces and feature borders are
    edge-dense), the composite carries the source's shading baseline
    there. The identity contract holds at the source pose — the same
    doctrine as the film-band repaint's source authority — and the
    correction never brightens, never imports the source's chroma or
    detail (each consumer keeps its own micro-texture verbatim: the
    correction is a SMOOTH per-consumer luminance rescale), and never
    touches texels the source cannot see (no witness demotion without
    source evidence: photo-curtain parallax bands stay untreated).

    Scope guards (each with a measured counterexample):
    - SOURCE-VALID ONLY: reference-exclusive territory is that view's
      identity contract (its own gate compares it); treating it would
      demote confident witnesses on no evidence.
    - Pairwise gauge: a global exposure difference between photos is not
      a shadow; the deviation must exceed the co-witnessed median by
      `gauge_margin` (measured neck deviation -0.28..-0.35 vs gauge
      -0.08: cleanly separated).
    - Edge-density refusal (`edge_p85_max`, resolution-normalized): the
      photo's hair-curtain edge crossing the neck is edge-dense and must
      not be flattened; the shadow apron measures p85 0.5-1.3.
    - WORLD-BALL component merge (`merge_radius_ratio` of the observed
      diagonal): the atlas cuts one physical apron into UV fragments
      below any honest size floor; fragments of one world region are one
      phenomenon and must be judged together (the fringe lane's complex
      rule, measured: the upper-neck fragments only pass as a cluster).
    - One-sided: luminance scale clipped to [scale_floor, 1] — this
      mechanism only darkens toward the source's shadow; brightening is
      the specular mechanism's job.

    Returns (target_baseline_log, blend_weight, stats): an absolute
    smoothed log-luminance field (the source's shading baseline brought
    into the composite's exposure frame), a feathered [0, 1] application
    mask, and stats; or None when nothing qualifies (single-view bakes
    structurally: no reference can win a texel).
    """
    import numpy as np
    from scipy.ndimage import binary_dilation, convolve1d, distance_transform_edt
    from scipy.ndimage import label as cc_label

    if len(view_rgb) < 2:
        return None
    weights = [np.asarray(w, dtype=np.float32) for w in view_weight]
    valids = [np.asarray(v, dtype=bool) for v in view_valid]
    rgbs = [np.asarray(r, dtype=np.float32)[:, :, :3] for r in view_rgb]
    observed = np.asarray(observed_mask, dtype=bool)
    resolution = observed.shape[0]
    s = resolution / float(reference_resolution)
    eps = 0.02
    src = int(source_view_index)
    if not (0 <= src < len(rgbs)):
        return None

    weight_stack = np.stack(weights)
    winner = weight_stack.argmax(axis=0)
    winner_weight = weight_stack.max(axis=0)

    lums = [r.mean(axis=2) for r in rgbs]
    logs = [np.log(np.clip(l, 0.0, 1.0) + eps) for l in lums]
    winner_lum = np.take_along_axis(np.stack(lums), winner[None], axis=0)[0]

    domain = (
        observed & (winner != src) & valids[src]
        & (lums[src] > float(min_source_luminance))
        & (winner_lum > float(min_winner_luminance))
        & (winner_weight >= float(min_winner_weight))
    )
    if int(domain.sum()) < 500:
        return None

    # pairwise lighting gauges source vs each reference: confident
    # co-witnessed texels first, bright co-valid fallback (the confident
    # population can be empty exactly where the source's weight collapses)
    gauges: Dict[int, float] = {}
    for v in range(len(rgbs)):
        if v == src:
            continue
        confident = (
            valids[src] & valids[v]
            & (weights[src] >= 0.25) & (weights[v] >= 0.25)
        )
        if int(confident.sum()) < 200:
            confident = (
                valids[src] & valids[v]
                & (lums[src] > 0.25) & (lums[v] > 0.25)
            )
        gauges[v] = (
            float(np.median((logs[src] - logs[v])[confident]))
            if int(confident.sum()) >= 200 else 0.0
        )

    deviation = np.zeros(domain.shape, dtype=np.float32)
    gauge_map = np.zeros(domain.shape, dtype=np.float32)
    for v, gauge in gauges.items():
        sel = domain & (winner == v) & valids[v]
        deviation[sel] = (logs[src] - logs[v])[sel] - gauge
        gauge_map[winner == v] = gauge
    candidates = domain & (deviation < -float(gauge_margin))
    if not candidates.any():
        return None

    # source-photo edge density, resolution-normalized (see the specular
    # mechanism for the s-normalization rationale)
    gx = convolve1d(
        convolve1d(lums[src], [1.0, 0.0, -1.0], axis=1), [3.0, 10.0, 3.0], axis=0
    )
    gy = convolve1d(
        convolve1d(lums[src], [1.0, 0.0, -1.0], axis=0), [3.0, 10.0, 3.0], axis=1
    )
    edge = np.where(valids[src], np.hypot(gx, gy) * s, 0.0)

    # pooled authorization (one shadow is one phenomenon, not per-texel noise)
    pool = _masked_atlas_gaussian(
        candidates.astype(np.float32), domain, sigma_pool * s
    )
    density = _masked_atlas_gaussian(
        domain.astype(np.float32), domain, sigma_pool * s
    )
    authorized = (density > float(pair_density_min)) & (
        pool / np.maximum(density, 1e-6) > float(vote_ratio_min)
    )
    final_candidates = candidates & authorized
    if not final_candidates.any():
        return None

    labels, _ = cc_label(binary_dilation(final_candidates, iterations=2))
    labels = np.where(final_candidates, labels, 0)

    # world-ball merge of atlas fragments before the size/edge judgment
    fragments: List[Dict[str, Any]] = []
    points = (
        np.asarray(positions, dtype=np.float32)[:, :, :3]
        if positions is not None else None
    )
    merge_radius = 0.0
    if points is not None and observed.any():
        span = points[observed]
        merge_radius = float(merge_radius_ratio) * float(
            np.linalg.norm(span.max(axis=0) - span.min(axis=0))
        )
    for i in np.unique(labels[labels > 0]):
        component = labels == i
        size = int(component.sum())
        if size < 8:
            continue
        row = {"mask": component, "size": size}
        if points is not None:
            p = points[component]
            centroid = p.mean(axis=0)
            row["centroid"] = centroid
            row["radius"] = float(
                np.percentile(np.linalg.norm(p - centroid[None, :], axis=1), 95)
            )
        fragments.append(row)
    if points is not None and merge_radius > 0.0:
        merged = True
        while merged:
            merged = False
            for i in range(len(fragments)):
                for j in range(i + 1, len(fragments)):
                    gap = float(np.linalg.norm(
                        fragments[i]["centroid"] - fragments[j]["centroid"]
                    ))
                    if gap < 0.8 * (
                        fragments[i]["radius"] + fragments[j]["radius"]
                    ) + merge_radius:
                        mask = fragments[i]["mask"] | fragments[j]["mask"]
                        p = points[mask]
                        centroid = p.mean(axis=0)
                        fragments[i] = {
                            "mask": mask, "size": int(mask.sum()),
                            "centroid": centroid,
                            "radius": float(np.percentile(
                                np.linalg.norm(p - centroid[None, :], axis=1), 95
                            )),
                        }
                        fragments.pop(j)
                        merged = True
                        break
                if merged:
                    break

    min_component = max(4, int(round(float(min_component_texels) * s * s)))
    kept = np.zeros(domain.shape, dtype=bool)
    components: List[Dict[str, Any]] = []
    for fragment in fragments:
        if fragment["size"] < min_component:
            continue
        ring = binary_dilation(fragment["mask"], iterations=2) & domain
        edge_p85 = float(np.percentile(edge[ring], 85)) if ring.any() else 99.0
        if edge_p85 >= float(edge_p85_max):
            continue
        kept |= fragment["mask"]
        components.append(
            {"texels": fragment["size"], "edge_p85": round(edge_p85, 2)}
        )
    if not kept.any():
        return None

    # the source's shading baseline over its own bright valid samples,
    # brought into the composite's exposure frame by the pairwise gauge
    source_bright = valids[src] & (lums[src] > float(min_source_luminance))
    source_baseline = _masked_atlas_gaussian(
        np.where(source_bright, logs[src], 0.0), source_bright, sigma_base * s
    )
    target_baseline_log = source_baseline - gauge_map

    inside = distance_transform_edt(kept)
    blend_weight = np.clip(
        inside / max(float(feather_texels) * s, 1e-6), 0.0, 1.0
    ).astype(np.float32)

    stats = {
        "applied": True,
        "texels": int(kept.sum()),
        "components": components,
        "gauges": {int(v): round(g, 4) for v, g in gauges.items()},
        "scale_floor": float(scale_floor),
    }
    return target_baseline_log, blend_weight, stats


def apply_shadow_apron_scale(
    rgb: Any,
    valid: Any,
    *,
    target_baseline_log: Any,
    blend_weight: Any,
    sigma_base: float = 4.0,
    scale_floor: float = 0.45,
    reference_resolution: int = 1024,
) -> Any:
    """Rescale one consumer's luminance toward the reconciled baseline.

    The consumer keeps its own detail verbatim: with target luminance
    `exp(baseline + own_detail)` and own detail `own_log - own_smooth`,
    the applied scale reduces to `exp(baseline - own_smooth)` — a smooth
    field. Clipped one-sided (darkening only) and feathered by
    `blend_weight`. Returns the corrected rgb (leading 3 channels).
    """
    import numpy as np

    eps = 0.02
    rgb = np.asarray(rgb, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    resolution = rgb.shape[0]
    s = resolution / float(reference_resolution)
    lum = rgb[:, :, :3].mean(axis=2)
    log_lum = np.log(np.clip(lum, 0.0, 1.0) + eps)
    smooth = _masked_atlas_gaussian(log_lum, valid, sigma_base * s)
    scale = np.clip(
        np.exp(target_baseline_log - smooth), float(scale_floor), 1.0
    )
    scale = 1.0 + (scale - 1.0) * np.asarray(blend_weight, dtype=np.float32)
    out = rgb.copy()
    out[:, :, :3] = np.clip(rgb[:, :, :3] * scale[..., None], 0.0, 1.0)
    return out


def _aggregation_hierarchy(
    system: Any,
    points: Any,
    pitch: float,
    *,
    first_cell_pitches: float = 3.0,
    coarsen_factor: float = 3.0,
    min_coarse_nodes: int = 400,
    max_levels: int = 8,
) -> List[Dict[str, Any]]:
    """Geometric-aggregation multigrid hierarchy (piecewise-constant P).

    Nodes are clustered by a voxel grid over their 3D positions with the
    cell size growing by `coarsen_factor` per level; each coarse operator
    is the Galerkin triple product P^T A P. Geometric (not matrix-graph)
    aggregation keeps setup O(N) and makes aggregates compact ON THE
    SURFACE — the shape of the smooth error modes of a surface Laplacian.
    The coarsest level is factorized once (splu).
    """
    import numpy as np
    from scipy import sparse

    levels: List[Dict[str, Any]] = [{"system": system.tocsr()}]
    current_points = np.asarray(points, dtype=np.float64)
    cell = float(first_cell_pitches) * float(pitch)
    for _ in range(int(max_levels)):
        n = levels[-1]["system"].shape[0]
        if n <= int(min_coarse_nodes):
            break
        keys = np.floor(current_points / cell).astype(np.int64)
        # Pack the voxel coordinates into one int64 for a fast 1D unique
        # (meshes are recentered near the origin; 2^20 cells per axis is
        # far beyond any realistic atlas extent).
        H = np.int64(1) << 20
        packed = ((keys[:, 0] + H) << 42) + ((keys[:, 1] + H) << 21) + (keys[:, 2] + H)
        _, assign = np.unique(packed, return_inverse=True)
        m = int(assign.max()) + 1
        if m >= n or m < 1:
            cell *= float(coarsen_factor)
            continue
        prolong = sparse.coo_matrix(
            (np.ones(n, dtype=system.dtype), (np.arange(n), assign)), shape=(n, m)
        ).tocsr()
        coarse = (prolong.T @ levels[-1]["system"] @ prolong).tocsr()
        counts = np.asarray(prolong.sum(axis=0)).ravel()
        centroids = np.zeros((m, 3), dtype=np.float64)
        np.add.at(centroids, assign, current_points)
        centroids /= np.maximum(counts, 1.0)[:, None]
        levels[-1]["prolong"] = prolong
        # Precompute the restriction in CSR: scipy would otherwise convert
        # prolong.T from CSC on EVERY V-cycle application.
        levels[-1]["restrict"] = prolong.T.tocsr()
        levels.append({"system": coarse})
        current_points = centroids
        cell *= float(coarsen_factor)
    try:
        from scipy.sparse.linalg import splu

        levels[-1]["solve"] = splu(levels[-1]["system"].tocsc()).solve
    except Exception:
        diag = levels[-1]["system"].diagonal()
        inverse = 1.0 / np.maximum(diag, 1e-12)
        levels[-1]["solve"] = lambda rhs: inverse[:, None] * rhs
    for level in levels:
        diag = level["system"].diagonal()
        level["inv_diag"] = 1.0 / np.maximum(diag, 1e-12)
    return levels


def _v_cycle(
    levels: List[Dict[str, Any]],
    rhs: Any,
    depth: int = 0,
    *,
    omega: float = 0.7,
    smooth_steps: int = 1,
) -> Any:
    """One symmetric V(1,1)-cycle with damped-Jacobi smoothing from x = 0.

    The symmetric construction (pre-smooth, coarse correction, post-smooth
    with the same smoother) keeps the cycle a symmetric positive operator,
    so it is a valid CG preconditioner.
    """
    level = levels[depth]
    system = level["system"]
    if "prolong" not in level:
        return level["solve"](rhs)
    inv_diag = level["inv_diag"]
    x = omega * (inv_diag[:, None] * rhs)
    for _ in range(int(smooth_steps) - 1):
        x += omega * (inv_diag[:, None] * (rhs - system @ x))
    residual = rhs - system @ x
    prolong = level["prolong"]
    x += prolong @ _v_cycle(
        levels, level["restrict"] @ residual, depth + 1,
        omega=omega, smooth_steps=smooth_steps,
    )
    for _ in range(int(smooth_steps)):
        x += omega * (inv_diag[:, None] * (rhs - system @ x))
    return x


def solve_screened_poisson(
    graph: Mapping[str, Any],
    *,
    gradients: Any,
    anchors_rgb: Any,
    anchor_lambda: Any,
    edge_weights: Optional[Any] = None,
    node_points: Optional[Any] = None,
    initial_rgb: Optional[Any] = None,
    lambda_floor: float = 1e-7,
    cg_tol: float = 5e-6,
    cg_max_iterations: int = 400,
) -> Optional[Tuple[Any, Dict[str, Any]]]:
    """Solve (L_w + Lambda) x = Lambda c + div(w g) on the texel graph.

    L_w is the graph Laplacian of `graph["edges"]` with per-edge weights
    `edge_weights` (default 1), Lambda the diagonal of per-node screening
    weights (floored at `lambda_floor` so fully unwitnessed components
    stay nonsingular), c the anchor colors and
    (div w g)_i = sum_j w_ij g_ij the weighted target-gradient divergence.
    The system is SPD (weights must be positive), solved by CG
    preconditioned with a multigrid V-cycle when `node_points` are
    provided (see `_aggregation_hierarchy`) and plain Jacobi otherwise.
    Returns (x, stats) with x float32 (N, 3), or None when scipy is
    unavailable.
    """
    import numpy as np

    try:
        from scipy import sparse
    except Exception:
        return None

    edges = np.asarray(graph["edges"], dtype=np.int64)
    n = int(graph["node_count"])
    lam64 = np.maximum(np.asarray(anchor_lambda, dtype=np.float64), 0.0) + float(
        lambda_floor
    )
    # float32 iteration when screening survives single precision: sparse
    # matvec is memory-bound, so f32 doubles throughput, and production
    # lambdas (>= ~1e-4) dominate f32 rounding of the diagonal. With
    # near-zero screening (pure gradient integration) the tiny floor would
    # literally round away against the vertex degree (4 + 1e-7 == 4 in
    # f32), leaving a singular system — those solves run in float64.
    # All CG SCALARS (dot products, norms) accumulate in float64 in both
    # modes, which keeps the recurrences stable at these tolerances.
    dtype = np.float32 if float(lam64.max(initial=0.0)) >= 1e-5 else np.float64
    if dtype == np.float32:
        # The floor must survive f32 Galerkin aggregation: coarse diagonal
        # entries sum hundreds of fine entries, and a 1e-7-per-node floor
        # vanishes against them, leaving unwitnessed components exactly
        # singular (splu then emits inf; measured on the fragmented
        # observed-only starship graph). 1e-5 is still 30-300x below any
        # production anchor.
        lam64 = np.maximum(lam64, 1e-5)
    g = np.asarray(gradients, dtype=dtype)
    c = np.asarray(anchors_rgb, dtype=dtype)
    lam = lam64.astype(dtype)

    weights = (
        np.ones(len(edges), dtype=dtype)
        if edge_weights is None
        else np.asarray(edge_weights, dtype=dtype)
    )
    i_idx, j_idx = edges[:, 0], edges[:, 1]
    degree = np.zeros(n, dtype=dtype)
    np.add.at(degree, i_idx, weights)
    np.add.at(degree, j_idx, weights)
    adjacency = sparse.coo_matrix((weights, (i_idx, j_idx)), shape=(n, n))
    adjacency = (adjacency + adjacency.T).tocsr()
    system = (sparse.diags(degree) - adjacency + sparse.diags(lam)).tocsr()
    system.data = system.data.astype(dtype)

    weighted_g = weights[:, None] * g
    b = lam[:, None] * c
    np.add.at(b, i_idx, weighted_g)
    np.add.at(b, j_idx, -weighted_g)

    x = (
        np.asarray(initial_rgb, dtype=dtype).copy()
        if initial_rgb is not None
        else c.copy()
    )
    stats: Dict[str, Any] = {"nodes": n, "edges": int(len(edges))}

    def norm(v: Any) -> float:
        return float(np.sqrt((v.astype(np.float64) ** 2).sum()))

    b_norm = norm(b) + 1e-30
    stats["initial_relative_residual"] = norm(b - system @ x) / b_norm

    levels: Optional[List[Dict[str, Any]]] = None
    if node_points is not None:
        try:
            levels = _aggregation_hierarchy(
                system, node_points, float(graph.get("pitch", 1e-3))
            )
            stats["mg_level_sizes"] = [int(l["system"].shape[0]) for l in levels]
        except Exception:
            levels = None

    inv_diag = (1.0 / np.maximum(system.diagonal(), 1e-12)).astype(dtype)

    def precondition(residual: Any) -> Any:
        if levels is not None and len(levels) > 1:
            return _v_cycle(levels, residual)
        return inv_diag[:, None] * residual

    def dot(a: Any, b_: Any) -> float:
        return float((a.astype(np.float64) * b_.astype(np.float64)).sum())

    r = b - system @ x
    z = precondition(r).astype(dtype)
    p = z.copy()
    rz = dot(r, z)
    iterations = 0
    for iterations in range(1, int(cg_max_iterations) + 1):
        ap = system @ p
        alpha = dtype(rz / max(dot(p, ap), 1e-30))
        x += alpha * p
        r -= alpha * ap
        if norm(r) / b_norm <= float(cg_tol):
            break
        z = precondition(r).astype(dtype)
        rz_new = dot(r, z)
        p = z + dtype(rz_new / max(rz, 1e-30)) * p
        rz = rz_new
    stats["cg_iterations"] = iterations
    stats["final_relative_residual"] = norm(b - system @ x) / b_norm
    if not np.isfinite(x).all():
        return None
    return x.astype(np.float32), stats


def composite_gradient_domain(
    *,
    positions_texture: Any,
    normals_texture: Optional[Any],
    view_rgb: Sequence[Any],
    view_weight: Sequence[Any],
    class_map: Any,
    filled_rgb: Any,
    anchor_confidence: Any,
    view_valid: Optional[Sequence[Any]] = None,
    gradient_rule: str = "max_confidence",
    anchor_lambda_scale: float = 3e-3,
    anchor_confidence_floor: float = 0.1,
    completion_anchor_scale: float = 0.15,
    source_view_index: Optional[int] = 0,
    source_anchor_boost: float = 4.0,
    source_confidence_floor: float = 0.4,
    material_step_cap: float = 0.18,
    resolution_reference: int = 1024,
    specular_reconcile: bool = True,
    shadow_reconcile: bool = True,
    cg_tol: float = 5e-6,
    cg_max_iterations: int = 400,
) -> Optional[Tuple[Any, Dict[str, Any]]]:
    """End-to-end gradient-domain composite; returns (rgb, stats) or None.

    `filled_rgb` is the fully completed texture (observed blend + mirror +
    fill with synthesized detail): it provides the initial guess, the
    anchor colors, and the completion-region gradients, so the solve is a
    REFINEMENT of the existing composite that redistributes tone and
    reconciles handoffs without inventing content. `anchor_confidence` is
    the per-texel blend weight in [0, 1]; `class_map` is 0 observed /
    1 mirror / 2 fill / negative off-surface.

    Screening strength: lambda_i = anchor_lambda_scale * confidence_i for
    observed texels, rescaled by (resolution_reference / resolution)^2 so
    the tone-equalization decay length 1/sqrt(lambda) stays FIXED IN WORLD
    UNITS across texture resolutions. Completion texels get
    `completion_anchor_scale` of the observed strength: enough to pin the
    completed content's level, weak enough that observed tone wins near
    class borders. Texels the SOURCE view claims confidently
    (weight >= `source_confidence_floor` in view `source_view_index`) get
    `source_anchor_boost` times the anchor: the source photo is the
    subject's actual appearance and the identity contract holds at its
    pose, so equalization against synthesized/auxiliary references may
    only recolor it with proportionally smaller amplitude — the same
    priority the per-texel conflict resolution already grants the source.
    Failure of any stage returns None so the caller keeps the input
    composite unchanged.

    With `specular_reconcile` (default, multi-view only by construction:
    it requires a second valid witness), the source view's baked specular
    lobes are reconciled against the cross-view diffuse consensus BEFORE
    gradient selection, so both the gradient witnesses and the anchors
    carry the reconciled skin (see `reconcile_specular_lobes`). The
    correction lands in `view_rgb[source]` and, scaled by the source's
    blend share, in the anchors.

    With `shadow_reconcile` (default, multi-view only by construction:
    it requires a reference to win a texel), the dual correction runs on
    reference-won co-witnessed aprons the source photo reads darker (its
    cast shadows: see `reconcile_shadow_aprons`). The corrected
    luminance lands in every valid REFERENCE view's rgb and in the
    anchors, so gradient witnesses and anchors again carry one story;
    the source view is never touched (it already carries its shadow).
    """
    import numpy as np

    positions = np.asarray(positions_texture, dtype=np.float32)
    surface = positions[:, :, 3] > 0.0
    if not surface.any():
        return None
    height = surface.shape[0]

    specular_stats: Optional[Dict[str, Any]] = None
    if (
        specular_reconcile
        and source_view_index is not None
        and view_valid is not None
        and len(view_rgb) >= 2
    ):
        reconciled = reconcile_specular_lobes(
            view_rgb=view_rgb,
            view_weight=view_weight,
            view_valid=view_valid,
            observed_mask=surface,
            source_view_index=int(source_view_index),
            reference_resolution=int(resolution_reference),
        )
        if reconciled is not None:
            delta, specular_stats = reconciled
            src = int(source_view_index)
            view_rgb = list(view_rgb)
            source_rgba = np.asarray(view_rgb[src], dtype=np.float32).copy()
            source_rgba[:, :, :3] = np.clip(source_rgba[:, :, :3] + delta, 0.0, 1.0)
            view_rgb[src] = source_rgba
            weight_stack = np.stack(
                [np.asarray(w, dtype=np.float32) for w in view_weight]
            )
            share = np.clip(
                weight_stack[src] / np.maximum(weight_stack.max(axis=0), 1e-6),
                0.0,
                1.0,
            )
            filled_rgb = np.asarray(filled_rgb, dtype=np.float32).copy()
            filled_rgb[:, :, :3] = np.clip(
                filled_rgb[:, :, :3] + delta * share[..., None], 0.0, 1.0
            )

    shadow_stats: Optional[Dict[str, Any]] = None
    if (
        shadow_reconcile
        and source_view_index is not None
        and view_valid is not None
        and len(view_rgb) >= 2
    ):
        shadow = reconcile_shadow_aprons(
            view_rgb=view_rgb,
            view_weight=view_weight,
            view_valid=view_valid,
            observed_mask=surface,
            positions=positions,
            source_view_index=int(source_view_index),
            reference_resolution=int(resolution_reference),
        )
        if shadow is not None:
            baseline_log, blend_weight, shadow_stats = shadow
            src = int(source_view_index)
            view_rgb = list(view_rgb)
            for v in range(len(view_rgb)):
                if v == src:
                    continue
                corrected = apply_shadow_apron_scale(
                    view_rgb[v],
                    np.asarray(view_valid[v], dtype=bool),
                    target_baseline_log=baseline_log,
                    blend_weight=blend_weight,
                    reference_resolution=int(resolution_reference),
                )
                rgba_v = np.asarray(view_rgb[v], dtype=np.float32).copy()
                rgba_v[:, :, :3] = corrected[:, :, :3]
                view_rgb[v] = rgba_v
            filled_corrected = apply_shadow_apron_scale(
                filled_rgb,
                surface,
                target_baseline_log=baseline_log,
                blend_weight=blend_weight,
                reference_resolution=int(resolution_reference),
            )
            filled_rgb = np.asarray(filled_rgb, dtype=np.float32).copy()
            filled_rgb[:, :, :3] = filled_corrected[:, :, :3]

    graph = build_texel_surface_graph(
        positions_texture, normals_texture=normals_texture
    )
    if graph is None:
        return None
    nodes_rc = graph["nodes_rc"]
    node_rows, node_cols = nodes_rc[:, 0], nodes_rc[:, 1]

    gradients, line_mask = select_composite_gradients(
        graph,
        view_rgb=view_rgb,
        view_weight=view_weight,
        class_map=class_map,
        filled_rgb=filled_rgb,
        view_valid=view_valid,
        rule=gradient_rule,
    )
    # Two corrections for the witness-less (line) edges, both measured
    # load-bearing:
    #
    # 1. Resolution consistency: handoff and class-border edges appear once
    #    per boundary-crossing ROW, so their count grows linearly with
    #    resolution while anchors and interior edges grow quadratically.
    #    Down-weighting them by the pitch ratio keeps the boundary energy
    #    per world length fixed across resolutions.
    # 2. Material gate (the screened-Poisson analog of seam leveling's
    #    `boundary_cap`): a SMALL step across a witness-less edge is a tone
    #    seam and must be equalized; a LARGE step is genuine material
    #    content (hair against skin at the hairline, ear folds against
    #    cheek), and demanding agreement there tints both materials toward
    #    each other — measured as a washed ear and temple (side-profile
    #    identity SSIM 0.704 -> 0.602). The gate keeps full weight for
    #    steps below `material_step_cap` and decays smoothly (Gaussian in
    #    step/cap) beyond it, so material borders keep their content while
    #    exposure steps (~0.03-0.10) still vanish.
    filled = np.asarray(filled_rgb, dtype=np.float32)[:, :, :3]
    line_weight = min(1.0, float(resolution_reference) / float(height))
    edge_weights = np.ones(len(gradients), dtype=np.float32)
    if line_mask.any():
        nodes_rc = graph["nodes_rc"]
        edges = graph["edges"]
        sub = np.nonzero(line_mask)[0]
        rc_a = nodes_rc[edges[sub, 0]]
        rc_b = nodes_rc[edges[sub, 1]]
        step = np.abs(
            filled[rc_a[:, 0], rc_a[:, 1]] - filled[rc_b[:, 0], rc_b[:, 1]]
        ).mean(axis=1)
        over = np.maximum(step / max(float(material_step_cap), 1e-6) - 1.0, 0.0)
        material_gate = np.exp(-(over**2))
        edge_weights[sub] = line_weight * material_gate.astype(np.float32)

    confidence = np.asarray(anchor_confidence, dtype=np.float32)
    classes = np.asarray(class_map)
    node_class = classes[node_rows, node_cols]
    node_confidence = confidence[node_rows, node_cols]

    scale = float(anchor_lambda_scale) * (
        float(resolution_reference) / float(height)
    ) ** 2
    # Confidence floor: rim/feather texels carry near-zero blend weight, so
    # pure proportional screening leaves them free to drift by large
    # amounts over the equalization decay length — at fragmented coverage
    # (ears, hairlines) whole slivers then wash toward their neighborhood
    # average (measured: ear folds flattened, side-profile identity SSIM
    # 0.704 -> 0.602). The floor bounds every observed texel's drift while
    # exposure-scale steps (well under the floor's implied stiffness)
    # still equalize.
    lam = scale * np.clip(
        np.maximum(node_confidence, float(anchor_confidence_floor)), 0.0, 1.0
    )
    if (
        source_view_index is not None
        and 0 <= int(source_view_index) < len(view_weight)
        and float(source_anchor_boost) > 1.0
    ):
        source_weight = np.asarray(
            view_weight[int(source_view_index)], dtype=np.float32
        )[node_rows, node_cols]
        confident_source = (node_class == 0) & (
            source_weight >= float(source_confidence_floor)
        )
        lam[confident_source] *= float(source_anchor_boost)
    lam[node_class > 0] = scale * float(completion_anchor_scale)

    anchors = filled[node_rows, node_cols]
    solved = solve_screened_poisson(
        graph,
        gradients=gradients,
        anchors_rgb=anchors,
        anchor_lambda=lam,
        edge_weights=edge_weights,
        node_points=positions[:, :, :3][node_rows, node_cols],
        initial_rgb=anchors,
        cg_tol=cg_tol,
        cg_max_iterations=cg_max_iterations,
    )
    if solved is None:
        return None
    x, stats = solved
    out = filled.copy()
    out[node_rows, node_cols] = np.clip(x, 0.0, 1.0)
    stats["pitch"] = graph["pitch"]
    stats["edge_kind_counts"] = {
        "grid": int((graph["edge_kind"] == 0).sum()),
        "stitch": int((graph["edge_kind"] == 1).sum()),
        "line_constraint": int(line_mask.sum()),
    }
    stats["line_edge_weight"] = line_weight
    stats["gradient_rule"] = gradient_rule
    stats["anchor_lambda_scale"] = scale
    stats["specular_reconcile"] = specular_stats or {"applied": False}
    stats["shadow_reconcile"] = shadow_stats or {"applied": False}
    return out, stats
