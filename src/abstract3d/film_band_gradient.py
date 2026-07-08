"""Hairline-band gradient repaint for multi-view bakes (film band, part 2).

THE DEFECT (measured on the face proof asset, cycles 2-3): generated
meshes fuse the wispy hairline into the head as a smooth APRON tens of
texels wide between the photo-confirmed dark hair mass and the true skin
edge. Every photo's projection compresses its own narrow (4-10 px)
wisp-transition ribbon across that whole apron — the front view's bins sit
at median 1 px from its dark body across the apron — and the membrane fill
mixes both materials. Rendered, the apron reads as a smooth putty-taupe
stripe where the photo shows near-black hair blending into skin. The film
COMMITMENT (film_band.py) correctly vacates bright mixture claims, but its
committed texels cover only ~8% of the visible band and its retone pulls
toward distant dark anchors with heavy attenuation: the band stays putty.

THE MECHANISM (scale-free; every design choice measured, see the values in
the docstrings below):

1. GEODESIC PROFILE FIELD. On the texel surface graph (geodesic — the
   band wraps the temples; Euclidean balls mix sheets), two multi-source
   Dijkstra fields anchor the band: d_mass to the photo-confirmed dark
   mass (large components of dark observed texels whose own bin lies
   inside some view's dark body with a dark sample) and d_base to the
   photo-space skin ring (texels imaged first-surface with a bright
   sample at least one transition-length outside that view's dark body).
   The photos' own skin-side falloff profiles, normalized per photo by
   its transition length, pool into S(u); with u = d_base/(d_base+d_mass)
   the tone target is

       T(x) = hair_local(x) + (base_local(x) - hair_local(x)) * S(u(x))

   with hair_local/base_local the geodesic-nearest anchor colors
   (graph-blurred). T is near-black at the hair-mass boundary and blends
   into the local skin tone at the face edge, following the photo's own
   gradient — the acceptance shape.

2. SOURCE AUTHORITY. The identity contract holds at the source pose, so
   apron texels the source view images first-surface (solid alpha,
   non-grazing) take the source photo's own sampled color verbatim: the
   real strand layout. Statistical tone alone measured identity[front]
   0.60-0.62 (WORSE than the 0.63 baseline — SSIM wants content, not
   wash); with authority the band carries the photo and identity reaches
   0.68. Guards, each with its measured failure:
   - dark stamps inside the feature moat (dilated dark features + the
     photo's own feature components) must carry no base-material witness
     veto from ANY view: a parallax-displaced brow/lid copy is vetoed at
     96% while valid apron stamps are not (az0 third-eye class); applied
     globally the veto kills half the valid stamps (-0.05 SSIM);
   - OFF-POSE DISPLACEMENT VETO (cycle 4, FACE-20): a connected dark-stamp
     component whose texels mostly carry a base-material witness veto AND
     whose median field position sits in the skin half of the transition
     (S >= DISPLACED_S_MIN) is parallax-displaced content — the photo's
     own curtain-edge/ear-shadow pixels billboarded onto surface the
     source ray only grazes. Stamped verbatim they print hard black
     stroke/arc artifacts across the battery (temple crack at az0,
     silhouette streak at az-22.5, hairline line at az-90, ear-helix arcs
     at az+90/+112.5 — all measured veto-consensus 0.7-1.0 at S_med
     0.35-0.66, the parallax dual of the third-eye class). Near the mass
     (S below the gate) equally vetoed dark stamps are the wisp/strand
     content the identity gate needs: the same veto applied there kills
     half the valid stamps (measured, cycle 3). Rejected sites are
     refilled AFTER all guards: the local guard tone (clamp/diffusion
     verdict) rescaled to the photo's luminance pattern at reduced gain,
     floored strictly ABOVE the dark-material class — the site keeps
     paying part of the source-pose contract but can no longer render as
     a dark stroke at any pose (measured comp identity 0.676 vs 0.675
     flat-tone, strokes gone at all 48 battery views);
   - stamps stand off texels a reference view claims above 0.5 weight:
     repainting a profile's confident territory trades identity[front]
     for identity[side] (side worst-window 0.116 -> 0.031 measured);
   - protected regions are never touched: dark features + dilation
     (brows, lashes, eyeliner) and large compact bright-confident blobs
     (ear bodies, garment straps).

3. GAP DIFFUSION + ENVELOPE CLAMP. Apron texels the source cannot reach
   (self-occluded film slivers) take graph-diffused stamp/mass colors
   under a field-consistency gate (diffused tone must respect the local
   envelope T*(1+h); ungated diffusion sprays skin into hair territory:
   skin_in_hair 0.003 -> 0.011 measured). Remaining over-envelope texels
   are luminance-clamped to the envelope (one-sided darkening; deep
   clamps blend toward T). One-sidedness makes brows, lashes, eyes and
   all legitimately dark content untouchable by construction.

4. ISLAND GUARDS on the final state. Small treated dark components with
   no pre-existing dark-observed anchor revert to their input colors
   (parallax floaters — the feature-scale island classes the eye/debris
   detectors flag). Bright shell components graph-disconnected from both
   the skin ring and the protected blobs pull to the envelope (isolated
   skin islands inside hair).

Single-view bakes and bakes without a usable dark mass / skin ring /
falloff profile return None: the caller keeps cycle-2 behavior unchanged
(the ship/owl canaries stay bit-identical by construction).
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .film_band import BRIGHT_LUMINANCE_RATIO, DARK_LUMINANCE_RATIO

# Envelope headroom over the field tone before the clamp engages.
ENVELOPE_HEADROOM = 0.10
# Source stamps require this much facing (grazing stamps smear content).
STAMP_MIN_FACING = 0.25
# Reference-view claims above this weight are that view's identity
# territory; stamps stand off (measured side worst-window collapse).
SIDE_STANDOFF_WEIGHT = 0.5
# The clamp operates inside the transition proper (S below this); the
# outer shell keeps its content (the profile is saturated there).
CLAMP_S_MAX = 0.90
# Shell extent: field defined where S is below this.
SHELL_S_MAX = 0.995
# Off-pose displacement veto (FACE-20): a would-be dark-stamp component
# is rejected when the fraction of its texels carrying a base-material
# witness veto reaches DISPLACED_VETO_FRAC and its median S sits at or
# above DISPLACED_S_MIN — the skin half of the transition, where the
# field (the photos' own pooled falloff) says the surface has left the
# hair body. Both inputs are existing mechanism vocabulary; 0.35 is the
# measured boundary separating the stroke class (S_med 0.35-0.66) from
# the valid vetoed wisp mass (S p50 0.23) on the proof asset, with the
# stroke recall/valid-kill trade flat between 0.35 and 0.5.
DISPLACED_S_MIN = 0.35
DISPLACED_VETO_FRAC = 0.5
# Rejected sites refill at the local guard tone rescaled to the photo's
# luminance pattern: floor strictly above the dark class (times the
# dark split ratio) plus a reduced-gain copy of the photo level. The
# floor is what makes a stroke unprintable; the gain pays the srcpose
# identity contract (measured: gain 0.30 recovers +0.0007 comp SSIM and
# -0.26 comp MAE over flat clamp tone with zero sweep flags).
DISPLACED_REFILL_FLOOR = 1.02
DISPLACED_REFILL_GAIN = 0.30
# FIELD SUPPORT BOUND (cycle 6, FACE-22): the S field is a RATIO of
# geodesic distances, so it takes mid-transition values arbitrarily far
# from the hair mass (measured on the face proof: neck/chest texels at
# 9-24 pooled-profile transition lengths carried S~0.66 and were treated
# as "hairline apron"). The pooled falloff profile is only MEASURED
# within a few transition lengths of the dark body — beyond that the
# envelope/tone target is extrapolation, and its component borders print
# as thin line-art contours on smooth skin (the FACE-22 glyph cluster
# and tone-region border segments; stage-ablation difference maps in the
# cycle-6 provenance run). The mechanism's own scale bounds its domain:
# treatment is confined to within FIELD_SUPPORT_TRANSITIONS of the mass
# and feathers out over the last transition length. 6 separates the
# measured stroke sites (d_mass p5 = 8.8 T) from the honest apron
# (film commitment zone p50 = 2.0 T) with margin on both sides.
FIELD_SUPPORT_TRANSITIONS = 6.0
# STAMP BORDER FEATHER (cycle 6, FACE-22): authority stamps are verbatim
# photo copies; where the stamped region ends on NON-HAIR surface the
# stamp/composite tone step printed as thin contour segments and, at the
# new support cut, as vertical chroma seams (measured: comp battery
# chroma_seam 0.49-0.69 at az+22.5/+70 with a hard cut vs 0.13-0.23
# feathered). Stamp colors blend composite -> photo over this many
# texels (world-scaled) from the treated-region border; borders shared
# with the dark mass stay verbatim — the stamp continues the mass's own
# dark content there and feathering would re-introduce putty at the
# wisp roots.
STAMP_BORDER_FEATHER_TEXELS = 6.0


def _view_direction(azimuth_deg: float, elevation_deg: float):
    import math

    import numpy as np

    azimuth = math.radians(float(azimuth_deg))
    elevation = math.radians(float(elevation_deg))
    return np.array([
        math.cos(elevation) * math.cos(azimuth),
        math.cos(elevation) * math.sin(azimuth),
        math.sin(elevation),
    ], dtype=np.float32)


def _masked_graph_blur(edges, node_count, values, weight, pinned, iterations):
    """Jacobi blur over graph edges. `pinned` nodes keep their values;
    nodes stay unset (weight 0) until the diffusion front reaches them."""
    import numpy as np
    from scipy.sparse import coo_matrix

    ones = np.ones(len(edges))
    adjacency = coo_matrix(
        (np.concatenate([ones, ones]),
         (np.concatenate([edges[:, 0], edges[:, 1]]),
          np.concatenate([edges[:, 1], edges[:, 0]]))),
        shape=(node_count, node_count)).tocsr()
    pin_values = values[pinned].copy()
    for _ in range(int(iterations)):
        numerator = adjacency @ (values * weight[:, None])
        denominator = adjacency @ weight
        reached = denominator > 0
        values[reached] = numerator[reached] / denominator[reached, None]
        weight = np.maximum(weight, reached.astype(np.float64))
        values[pinned] = pin_values
    return values, weight, adjacency


def _displaced_stamp_components(would_dark, veto_any, S_map):
    """Off-pose displacement veto over would-be dark-stamp components.

    Component-level on purpose: a stroke is a coherent structure — its
    texels share provenance (one curtain edge, one ear shadow), so the
    decision must be shared too. Texel-level rejection fragments strokes
    into speckles and kills isolated valid strand texels that happen to
    sit at high S (measured: component decision reads visually cleaner
    AND costs less identity than the texel version).
    """
    import numpy as np
    from scipy.ndimage import label

    labels, count = label(would_dark, structure=np.ones((3, 3), bool))
    if not count:
        return np.zeros_like(would_dark)
    sel = labels > 0
    lab = labels[sel]
    sizes = np.bincount(lab, minlength=count + 1)
    veto_frac = (np.bincount(lab, weights=veto_any[sel].astype(float),
                             minlength=count + 1)
                 / np.maximum(sizes, 1))
    # per-component S median via sort-groupby (S NaN outside the field
    # never counts toward the median)
    S_sel = np.where(np.isfinite(S_map[sel]), S_map[sel], -1.0)
    order = np.argsort(lab, kind="stable")
    lab_sorted = lab[order]
    S_sorted = S_sel[order]
    starts = np.searchsorted(lab_sorted, np.arange(1, count + 2))
    reject = np.zeros(count + 1, dtype=bool)
    for component in range(1, count + 1):
        if veto_frac[component] < DISPLACED_VETO_FRAC:
            continue
        s_values = S_sorted[starts[component - 1]:starts[component]]
        s_values = s_values[s_values >= 0]
        if s_values.size and np.median(s_values) >= DISPLACED_S_MIN:
            reject[component] = True
    return reject[labels]


def repaint_film_band(
    projections: Sequence[Mapping[str, Any]],
    colors_rgba: Any,
    *,
    positions_texture: Any,
    normals_texture: Optional[Any],
    observed_mask: Any,
    texture_resolution: int,
) -> Optional[Tuple[Any, Dict[str, Any]]]:
    """Repaint the hairline apron; returns (rgba, stats) or None (no-op).

    `projections` are the blend-time view records (post-vacate weights,
    sampled rgba, film maps with photo products); the FIRST projection is
    the source view whose pose carries the identity contract.
    """
    import numpy as np
    from scipy.ndimage import binary_dilation, binary_opening, label
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components, dijkstra

    from .gradient_compositing import build_texel_surface_graph

    if len(projections) < 2:
        return None
    views = [p for p in projections if isinstance(p.get("film_band"), dict)]
    if not views:
        return None
    source_projection = projections[0]
    source_maps = source_projection.get("film_band")
    if not isinstance(source_maps, dict):
        return None

    rgba = np.asarray(colors_rgba, dtype=np.float32).copy()
    colors = rgba[:, :, :3]
    original_colors = colors.copy()
    positions = np.asarray(positions_texture, dtype=np.float32)
    surface = positions[:, :, 3] > 0.0
    if not surface.any():
        return None
    observed = np.asarray(observed_mask, dtype=bool) & surface

    resolution = max(int(texture_resolution), 64)
    # Structural radii were calibrated at 2048; scaling by resolution keeps
    # them fixed in WORLD units across texture resolutions.
    radius_scale = max(0.25, resolution / 2048.0)

    def texels(count: float) -> int:
        return max(1, int(round(count * radius_scale)))

    luminance = colors.mean(axis=2)
    observed_luminance = luminance[observed]
    if observed_luminance.size < 64:
        return None
    bright_median = float(np.median(
        observed_luminance[observed_luminance >= np.median(observed_luminance)]))
    dark_observed = observed & (
        luminance < DARK_LUMINANCE_RATIO * bright_median)

    # ---- per-view photo votes + pooled profile ---------------------------
    profiles_norm: List[Tuple[Any, Any]] = []
    hair_vote = np.zeros(surface.shape, dtype=bool)
    base_vote = np.zeros(surface.shape, dtype=bool)
    weight_stack = [np.asarray(p["weight"], dtype=np.float32)
                    for p in projections]
    for projection in projections:
        maps = projection.get("film_band")
        if not isinstance(maps, dict):
            continue
        photo = maps.get("photo_products")
        if not isinstance(photo, dict):
            continue
        img_first = np.asarray(maps["img_first_texel"], dtype=bool)
        sample_lum = np.asarray(
            projection["rgba"], dtype=np.float32)[:, :, :3].mean(axis=2)
        view_bright = float(photo["bright_median"])
        hair_vote |= (img_first & photo["in_body_texel"]
                      & (sample_lum < DARK_LUMINANCE_RATIO * view_bright))
        base_vote |= (img_first & photo["beyond_transition_texel"]
                      & (sample_lum >= BRIGHT_LUMINANCE_RATIO * view_bright))
        if photo.get("profile") is not None:
            distances_px, profile = photo["profile"]
            transition_px = max(float(photo["transition_px"]), 1e-6)
            profiles_norm.append((distances_px / transition_px, profile))
    if not profiles_norm:
        return None

    # ---- surface graph + geodesic fields ----------------------------------
    graph = build_texel_surface_graph(
        positions, normals_texture=normals_texture)
    if graph is None:
        return None
    node_rows = graph["nodes_rc"][:, 0]
    node_cols = graph["nodes_rc"][:, 1]
    edges = graph["edges"]
    node_count = graph["node_count"]

    node_dark = (dark_observed & hair_vote)[node_rows, node_cols]
    dark_idx = np.nonzero(node_dark)[0]
    if len(dark_idx) < 64:
        return None
    remap = np.full(node_count, -1)
    remap[dark_idx] = np.arange(len(dark_idx))
    dark_edges = edges[node_dark[edges[:, 0]] & node_dark[edges[:, 1]]]
    dark_adjacency = coo_matrix(
        (np.ones(len(dark_edges)),
         (remap[dark_edges[:, 0]], remap[dark_edges[:, 1]])),
        shape=(len(dark_idx), len(dark_idx)))
    component_count, component = connected_components(
        dark_adjacency, directed=False)
    keep = (np.bincount(component)
            >= max(64, 0.02 * max(int(node_dark.sum()), 1)))
    mass_nodes = dark_idx[keep[component]]
    base_nodes = np.nonzero(base_vote[node_rows, node_cols])[0]
    if len(mass_nodes) < 64 or len(base_nodes) < 64:
        return None

    xyz = positions[:, :, :3][node_rows, node_cols].astype(np.float64)
    edge_lengths = np.maximum(
        np.linalg.norm(xyz[edges[:, 0]] - xyz[edges[:, 1]], axis=1), 1e-9)
    weighted_adjacency = coo_matrix(
        (np.concatenate([edge_lengths, edge_lengths]),
         (np.concatenate([edges[:, 0], edges[:, 1]]),
          np.concatenate([edges[:, 1], edges[:, 0]]))),
        shape=(node_count, node_count)).tocsr()
    d_mass, _, source_mass = dijkstra(
        weighted_adjacency, directed=False, indices=mass_nodes,
        min_only=True, return_predecessors=True)
    d_base, _, source_base = dijkstra(
        weighted_adjacency, directed=False, indices=base_nodes,
        min_only=True, return_predecessors=True)

    # ---- anchor color fields + profile blend -------------------------------
    node_color = colors[node_rows, node_cols].astype(np.float64)

    def anchor_field(source, anchors):
        valid = source >= 0
        values = np.zeros((node_count, 3))
        values[valid] = node_color[source[valid]]
        pinned = np.zeros(node_count, dtype=bool)
        pinned[anchors] = True
        values, _, _ = _masked_graph_blur(
            edges, node_count, values, valid.astype(np.float64), pinned, 12)
        return values

    hair_local = anchor_field(source_mass, mass_nodes)
    base_local = anchor_field(source_base, base_nodes)

    grid = np.linspace(0.0, 1.0, 32)
    pooled = np.median(np.stack([
        np.interp(grid, d, p, left=p[0], right=0.0) for d, p in profiles_norm
    ]), axis=0)
    pooled = np.minimum.accumulate(np.clip(pooled, 0.0, 1.0))

    # SAMPLING ADEQUACY: the repaint's structural elements (standoff
    # feather, feature moat, island guards) need the hairline transition
    # to span enough texels to be expressible; below that the treatment
    # of the multi-sheet curtain becomes incoherent — partial darkening
    # isolates pre-existing bright slivers into skin/lip islands
    # (measured at 1024: skin_in_hair 0.011 at the -22.5..-45 curtain
    # while 2048 stays green). The transition length in texels is the
    # mechanism's own scale: bail out to the cycle-2 retone below it.
    transitions_world = []
    for projection in projections:
        maps = projection.get("film_band")
        if not isinstance(maps, dict):
            continue
        photo = maps.get("photo_products")
        if not isinstance(photo, dict) or photo.get("profile") is None:
            continue
        bins_x_v = np.asarray(photo.get("bins_x"))
        if bins_x_v is None or bins_x_v.shape != surface.shape:
            continue
        # photo px -> world scale from the projection itself (least
        # squares gradient of the bin coordinate over 3D position).
        valid = np.asarray(maps["img_first_texel"], dtype=bool)
        if valid.sum() < 1000:
            continue
        rows_v, cols_v = np.nonzero(valid)
        step = max(1, len(rows_v) // 8000)
        rows_v, cols_v = rows_v[::step], cols_v[::step]
        A = np.concatenate([positions[rows_v, cols_v, :3],
                            np.ones((len(rows_v), 1), dtype=np.float32)], axis=1)
        gx, *_ = np.linalg.lstsq(
            A, bins_x_v[rows_v, cols_v].astype(np.float64), rcond=None)
        px_per_world = float(np.linalg.norm(gx[:3]))
        if px_per_world > 1e-3:
            transitions_world.append(
                float(photo["transition_px"]) / px_per_world)
    if not transitions_world:
        return None
    transition_texels = float(np.median(transitions_world)) / max(
        float(graph["pitch"]), 1e-9)
    # The floor sits between the measured working point (9.6 texels at
    # 2048 on the face proof) and the measured failing point (4.8 at
    # 1024): the transition must span at least the mechanism's own
    # structural granularity (standoff feather + guard scale, ~4 texels)
    # with margin, or its edits are sub-feature noise.
    if transition_texels < 7.0:
        return None

    # Field support (FACE-22): the treatment domain and its feather are
    # bounded by absolute geodesic distance to the dark mass in units of
    # the pooled transition length — the region where the profile the
    # field interpolates was actually measured. See
    # FIELD_SUPPORT_TRANSITIONS for the calibration.
    transition_world = float(np.median(transitions_world))
    support_world = FIELD_SUPPORT_TRANSITIONS * transition_world
    d_mass_map = np.full(surface.shape, np.inf, dtype=np.float32)
    d_mass_map[node_rows, node_cols] = d_mass.astype(np.float32)
    support = np.clip(
        (support_world - d_mass_map) / max(transition_world, 1e-9), 0.0, 1.0
    ).astype(np.float32)

    both_finite = np.isfinite(d_mass) & np.isfinite(d_base)
    u = np.zeros(node_count)
    with np.errstate(invalid="ignore", divide="ignore"):
        u[both_finite] = d_base[both_finite] / np.maximum(
            (d_mass + d_base)[both_finite], 1e-12)
    S = np.interp(u, grid, pooled)
    S = np.where(np.isfinite(d_base) & ~np.isfinite(d_mass), 1.0, S)
    S = np.where(np.isfinite(d_mass) & ~np.isfinite(d_base), 0.0, S)
    T = hair_local + (base_local - hair_local) * S[:, None]
    hair_only = ~np.isfinite(d_base)
    T[hair_only] = hair_local[hair_only]

    T_map = np.full((*surface.shape, 3), np.nan, dtype=np.float32)
    T_map[node_rows, node_cols] = T.astype(np.float32)
    S_map = np.full(surface.shape, np.nan, dtype=np.float32)
    S_map[node_rows, node_cols] = S.astype(np.float32)
    T_lum = T_map.mean(axis=2)

    mass_map = np.zeros(surface.shape, dtype=bool)
    mass_map[node_rows[mass_nodes], node_cols[mass_nodes]] = True
    ring_map = np.zeros(surface.shape, dtype=bool)
    ring_map[node_rows[base_nodes], node_cols[base_nodes]] = True
    finite_map = np.zeros(surface.shape, dtype=bool)
    finite_map[node_rows, node_cols] = both_finite
    shell = (finite_map & (S_map < SHELL_S_MAX) & ~mass_map & surface
             & (support > 0.0))

    # ---- protection ----------------------------------------------------------
    weights = np.stack(weight_stack, axis=0)
    winner_weight = weights.max(axis=0)
    protect_dark = binary_dilation(
        dark_observed & ~mass_map, iterations=texels(10))
    bright_confident = observed & (
        luminance >= BRIGHT_LUMINANCE_RATIO * bright_median
    ) & (winner_weight > 0.45)
    open_radius = max(2, texels(6))
    blob = binary_opening(
        bright_confident, structure=np.ones((open_radius, open_radius), bool))
    blob_labels, blob_count = label(blob, structure=np.ones((3, 3), bool))
    protect_blob = np.zeros(surface.shape, dtype=bool)
    if blob_count:
        sizes = np.bincount(blob_labels.ravel())
        big = np.zeros(blob_count + 1, dtype=bool)
        big[1:] = sizes[1:] >= (2 * open_radius) ** 2
        protect_blob = bright_confident & binary_dilation(
            big[blob_labels], iterations=2)
    band_domain = shell & ~protect_dark & ~protect_blob

    # ---- source authority -------------------------------------------------------
    source_rgba = np.asarray(source_projection["rgba"], dtype=np.float32)
    source_rgb = source_rgba[:, :, :3]
    source_first = np.asarray(source_maps["img_first_texel"], dtype=bool)
    source_photo = source_maps.get("photo_products") or {}
    # The projected alpha is zeroed wherever the claim weight is zero —
    # which includes every zone-VACATED apron texel, exactly the ones the
    # repaint exists for. The photo's own alpha at the bin (solid_texel)
    # is the honest sampling witness.
    if "solid_texel" in source_photo:
        source_solid = np.asarray(source_photo["solid_texel"], dtype=bool)
    else:
        source_solid = source_rgba[:, :, 3] > 0.0
    source_feature = np.asarray(
        source_photo.get("feature_texel", np.zeros(surface.shape, dtype=bool)),
        dtype=bool)
    source_lum = source_rgb.mean(axis=2)

    if normals_texture is not None:
        normals = np.asarray(normals_texture, dtype=np.float32)[:, :, :3]
        norms = np.linalg.norm(normals, axis=2, keepdims=True)
        normals = np.divide(normals, np.maximum(norms, 1e-8))
        facing = normals @ _view_direction(
            source_projection.get("azimuth_deg", 0.0),
            source_projection.get("elevation_deg", 0.0))
        # Reference-dominant territory: a reference view that images the
        # texel first-surface at solid alpha AND faces it better than the
        # source holds the identity contract there (its ±90 gate compares
        # that very surface). Treating it trades identity[front] for
        # identity[side] and seeds island contrast at the rear-quarter
        # views (measured on-tip: side_right worst-window -0.010, crown
        # flakes at az-70/-90 — all in reference-dominant territory).
        reference_dominant = np.zeros(surface.shape, dtype=bool)
        for projection in projections[1:]:
            maps = projection.get("film_band")
            if not isinstance(maps, dict):
                continue
            photo = maps.get("photo_products") or {}
            ref_first = np.asarray(maps["img_first_texel"], dtype=bool)
            if "solid_texel" in photo:
                ref_solid = np.asarray(photo["solid_texel"], dtype=bool)
            else:
                ref_solid = np.asarray(
                    projection["rgba"], dtype=np.float32)[:, :, 3] > 0.0
            ref_facing = normals @ _view_direction(
                projection.get("azimuth_deg", 0.0),
                projection.get("elevation_deg", 0.0))
            reference_dominant |= (ref_first & ref_solid
                                   & (ref_facing > np.maximum(facing, 0.0)))
    else:
        facing = np.ones(surface.shape, dtype=np.float32)
        reference_dominant = np.zeros(surface.shape, dtype=bool)
    band_domain &= ~reference_dominant
    # Feather at the standoff border: a hard treated/untreated boundary
    # prints as a near-black crease line along the curtain silhouette at
    # the between-view azimuths (measured at az-35: two new dark-debris
    # islands tracing exactly the domain edge). Stamps stay a full feather
    # length inside; the clamp fades out toward the border.
    from scipy.ndimage import distance_transform_edt

    feather = np.clip(
        distance_transform_edt(~reference_dominant) / float(texels(4)),
        0.0, 1.0).astype(np.float32)
    # The support bound feathers exactly like the standoff: stamps stop
    # half a transition inside the bound (feather >= 0.5 gate below) and
    # the clamp fades to nothing at it, so the domain edge cannot print.
    feather = np.minimum(feather, support)

    # OUTERMOST-SHEET corridor along the source axis: only the sheet the
    # source photo actually images may billboard its content. The
    # first-surface flag admits inner curtain sheets as texel density
    # drops (fewer texels per bin; measured at 1024: pink/skin shreds
    # sprayed through the curtain interior, skin_in_hair 0.0114). The
    # depth test is resolution-independent: per source bin, keep texels
    # within a fixed WORLD tolerance of the outermost surface depth.
    outermost = np.ones(surface.shape, dtype=bool)
    if "bins_y" in source_photo:
        bins_y = np.asarray(source_photo["bins_y"])
        bins_x = np.asarray(source_photo["bins_x"])
        photo_h, photo_w = source_photo["photo_shape"]
        source_axis = _view_direction(
            source_projection.get("azimuth_deg", 0.0),
            source_projection.get("elevation_deg", 0.0))
        depth = -(positions[:, :, :3] @ source_axis)
        flat_bins = bins_y.astype(np.int64) * int(photo_w) + bins_x.astype(np.int64)
        outer_depth = np.full(int(photo_h) * int(photo_w), np.inf)
        np.minimum.at(outer_depth, flat_bins[surface], depth[surface])
        pitch_world = float(graph["pitch"]) * resolution / 2048.0
        outermost = depth <= (outer_depth[flat_bins] + 4.0 * pitch_world)

    source_reach = (source_first & source_solid & outermost
                    & (facing >= STAMP_MIN_FACING) & (feather >= 0.5))

    stamp_is_dark = source_lum < DARK_LUMINANCE_RATIO * bright_median
    veto_any = np.zeros(surface.shape, dtype=bool)
    for projection in views:
        veto_any |= np.asarray(
            projection["film_band"]["veto_texel"], dtype=bool)
    moat = binary_dilation(
        dark_observed & ~mass_map, iterations=texels(20)) | source_feature
    dark_allowed = stamp_is_dark & ~(veto_any & moat)

    reference_weight = (weights[1:].max(axis=0) if len(weights) > 1
                        else np.zeros_like(winner_weight))
    side_confident = reference_weight > SIDE_STANDOFF_WEIGHT

    # OFF-POSE DISPLACEMENT VETO (FACE-20): the moat veto above only
    # binds near features; outside it, dark stamps whose surface another
    # view positively witnesses as base material landed anyway and
    # printed stroke/arc artifacts (temple crack, silhouette streak,
    # hairline line, ear-helix arcs — all measured veto-consensus
    # 0.7-1.0, moat-fraction 0.0). The field position separates them
    # from the valid vetoed wisp mass near the body (S p50 0.23): a
    # component of dark stamps in the SKIN HALF of the photos' own
    # falloff that other views veto is displaced content, not strands.
    would_dark = (band_domain & source_reach & ~side_confident
                  & dark_allowed)
    displaced = _displaced_stamp_components(would_dark, veto_any, S_map)
    dark_allowed &= ~displaced

    # Bright stamps carry skin/wisp content and belong to the TRANSITION
    # (S above ~0.25); deep in the hair field a "bright" source sample is
    # a first-surface misclassification painting skin into the curtain
    # interior (measured at 1024, where the coarser first-surface test
    # sprays pink shreds through the curtain: skin_in_hair 0.0114 and a
    # lip_in_hair blob). Dark stamps are hair-consistent at any depth.
    bright_in_transition = ~stamp_is_dark & np.isfinite(S_map) & (S_map >= 0.25)
    authority = (band_domain & source_reach & ~side_confident
                 & (dark_allowed | bright_in_transition))
    # Stamp border feather (FACE-22): verbatim photo colors ramp in from
    # the pre-repaint composite over STAMP_BORDER_FEATHER_TEXELS at every
    # treated-region border EXCEPT against the dark mass (the stamp
    # continues the mass's own content there; a feather would re-open the
    # putty gap at the wisp roots). Interior stamps stay verbatim.
    stamp_alpha = np.clip(
        distance_transform_edt(authority | mass_map)
        / float(texels(STAMP_BORDER_FEATHER_TEXELS)),
        0.0, 1.0).astype(np.float32)
    colors[authority] = (
        source_rgb[authority] * stamp_alpha[authority, None]
        + original_colors[authority] * (1.0 - stamp_alpha[authority, None]))
    rgba[:, :, 3][authority] = 1.0
    current_lum = colors.mean(axis=2)

    # ---- gap diffusion -----------------------------------------------------------
    envelope = np.where(np.isfinite(T_lum),
                        T_lum * (1.0 + ENVELOPE_HEADROOM), np.nan)
    over_envelope = current_lum > np.maximum(np.nan_to_num(envelope), 1e-6)
    clamp_candidates = (band_domain & ~authority & np.isfinite(T_lum)
                        & np.isfinite(S_map) & (S_map < CLAMP_S_MAX)
                        & over_envelope)

    node_authority = authority[node_rows, node_cols]
    gap_radius = texels(6)
    seed = node_authority.copy()
    seed[mass_nodes] = True
    seed_values = np.zeros((node_count, 3))
    seed_values[node_authority] = source_rgb[node_rows, node_cols][node_authority]
    seed_values[mass_nodes] = node_color[mass_nodes]
    diffused, diffusion_weight, plain_adjacency = _masked_graph_blur(
        edges, node_count, seed_values, seed.astype(np.float64), seed,
        2 * gap_radius)
    reach = node_authority.copy()
    for _ in range(gap_radius):
        reach = reach | (plain_adjacency @ reach.astype(np.float64) > 0)
    # Gaps are outermost-sheet only, like the stamps whose content they
    # extend: diffusing stamp colors onto hidden inner sheets sprays
    # bright content through curtain gaps at other poses (measured 1024).
    node_gap = (clamp_candidates[node_rows, node_cols]
                & outermost[node_rows, node_cols]
                & reach & ~node_authority)
    node_envelope = envelope[node_rows, node_cols]
    diffused_lum = diffused.mean(axis=1)
    consistent = (np.isfinite(node_envelope)
                  & (diffusion_weight > 0)
                  & (diffused_lum <= node_envelope))
    gap_take = node_gap & consistent
    gap_map = np.zeros(surface.shape, dtype=bool)
    gap_map[node_rows[gap_take], node_cols[gap_take]] = True
    colors[node_rows[gap_take], node_cols[gap_take]] = (
        diffused[gap_take].astype(np.float32))
    rgba[:, :, 3][gap_map] = 1.0
    current_lum = colors.mean(axis=2)

    # ---- envelope clamp (one-sided darkening, feathered at the standoff) ----
    clamp = (clamp_candidates & ~gap_map
             & (current_lum > np.maximum(np.nan_to_num(envelope), 1e-6)))
    blend_toward = envelope[clamp] * feather[clamp] + current_lum[clamp] * (
        1.0 - feather[clamp])
    scale = (blend_toward / np.maximum(current_lum[clamp], 1e-6))[:, None]
    clamped = colors[clamp] * scale
    depth = (np.clip((current_lum[clamp] - blend_toward) / 0.30, 0.0, 1.0)
             * feather[clamp])[:, None]
    colors[clamp] = (clamped * (1.0 - 0.5 * depth)
                     + np.nan_to_num(T_map[clamp]) * (0.5 * depth))
    rgba[:, :, 3][clamp] = 1.0

    applied = authority | gap_map | clamp

    # ---- island guards ---------------------------------------------------------------
    current_lum = colors.mean(axis=2)
    node_treated = applied[node_rows, node_cols]
    node_mass_mask = np.zeros(node_count, dtype=bool)
    node_mass_mask[mass_nodes] = True
    node_dark_observed = dark_observed[node_rows, node_cols]
    dark_all = ((current_lum < DARK_LUMINANCE_RATIO * bright_median)[
        node_rows, node_cols] | node_mass_mask | node_dark_observed)
    idx = np.nonzero(dark_all)[0]
    remap = np.full(node_count, -1)
    remap[idx] = np.arange(len(idx))
    guard_edges = edges[dark_all[edges[:, 0]] & dark_all[edges[:, 1]]]
    guard_adjacency = coo_matrix(
        (np.ones(len(guard_edges)),
         (remap[guard_edges[:, 0]], remap[guard_edges[:, 1]])),
        shape=(len(idx), len(idx)))
    guard_count, guard_component = connected_components(
        guard_adjacency, directed=False)
    anchored = np.zeros(guard_count, dtype=bool)
    anchored[np.unique(
        guard_component[(node_mass_mask | node_dark_observed)[idx]])] = True
    component_sizes = np.bincount(guard_component)
    small = component_sizes < max(256, 6e-4 * node_count)
    orphan_dark = np.zeros(node_count, dtype=bool)
    orphan_dark[idx] = (~anchored[guard_component]) & small[guard_component]
    revert = orphan_dark & node_treated
    if revert.any():
        rows = node_rows[revert]
        cols = node_cols[revert]
        colors[rows, cols] = original_colors[rows, cols]
        applied[rows, cols] = False

    current_lum = colors.mean(axis=2)
    node_bright = (current_lum >= DARK_LUMINANCE_RATIO * bright_median)[
        node_rows, node_cols]
    node_shell = shell[node_rows, node_cols]
    node_ring = np.zeros(node_count, dtype=bool)
    node_ring[base_nodes] = True
    node_protected = (protect_blob | protect_dark)[node_rows, node_cols]
    bright_all = node_bright & (node_shell | node_ring)
    idx = np.nonzero(bright_all)[0]
    remap = np.full(node_count, -1)
    remap[idx] = np.arange(len(idx))
    bright_edges = edges[bright_all[edges[:, 0]] & bright_all[edges[:, 1]]]
    bright_adjacency = coo_matrix(
        (np.ones(len(bright_edges)),
         (remap[bright_edges[:, 0]], remap[bright_edges[:, 1]])),
        shape=(len(idx), len(idx)))
    bright_count, bright_component = connected_components(
        bright_adjacency, directed=False)
    anchored_bright = np.zeros(bright_count, dtype=bool)
    anchored_bright[np.unique(
        bright_component[(node_ring | node_protected)[idx]])] = True
    orphan_bright = np.zeros(node_count, dtype=bool)
    orphan_bright[idx] = ~anchored_bright[bright_component]
    orphan_bright &= node_shell & ~node_protected
    node_T_lum = T_lum[node_rows, node_cols]
    kill = orphan_bright & np.isfinite(node_T_lum)
    if kill.any():
        rows = node_rows[kill]
        cols = node_cols[kill]
        target = (T_lum * (1.0 + ENVELOPE_HEADROOM))[rows, cols]
        level = colors[rows, cols].mean(axis=1)
        need = level > np.maximum(target, 1e-6)
        rows, cols = rows[need], cols[need]
        scale = (target[need] / np.maximum(level[need], 1e-6))[:, None]
        darkened = colors[rows, cols] * scale
        depth = np.clip((level[need] - target[need]) / 0.30, 0.0, 1.0)[:, None]
        colors[rows, cols] = (darkened * (1.0 - 0.5 * depth)
                              + np.nan_to_num(T_map[rows, cols]) * (0.5 * depth))
        applied[rows, cols] = True

    # DISPLACED-SITE REFILL, after every guard so the guards' verdict is
    # what gets rescaled: the site takes the photo's luminance PATTERN at
    # reduced gain on a floor strictly above the dark-material class,
    # carried by the LOCAL (clamped/diffused) chroma. The floor makes a
    # dark stroke unprintable at any pose by construction; the pattern
    # pays what the floor admits of the source-pose identity contract;
    # local chroma avoids amplifying the near-black photo pixels'
    # saturation into chroma islands (measured: photo-chroma refill
    # tripped the az+35 dark_debris gate at exactly 0.003).
    refill = displaced & would_dark
    if refill.any():
        floor_lum = (DISPLACED_REFILL_FLOOR
                     * DARK_LUMINANCE_RATIO * bright_median)
        target_lum = floor_lum + DISPLACED_REFILL_GAIN * source_lum[refill]
        level = np.maximum(colors.mean(axis=2)[refill], 1e-6)
        colors[refill] = np.clip(
            colors[refill] * (target_lum / level)[:, None], 0.0, 1.0)
        rgba[:, :, 3][refill] = 1.0
        applied[refill] = True

    stats = {
        "applied": bool(applied.any()),
        "field_support_transitions": FIELD_SUPPORT_TRANSITIONS,
        "authority_texels": int(authority.sum()),
        "gap_texels": int(gap_map.sum()),
        "clamp_texels": int(clamp.sum()),
        "guard_dark_reverted": int(revert.sum()),
        "guard_bright_killed": int(kill.sum()),
        "mass_nodes": int(len(mass_nodes)),
        "base_ring_nodes": int(len(base_nodes)),
        "displaced_dark_vetoed": int((would_dark & displaced).sum()),
        "displaced_refilled": int(refill.sum()),
    }
    return rgba, stats, applied
