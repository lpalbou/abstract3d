"""Film-band commitment for multi-view projection bakes.

THE DEFECT CLASS (face lane, cycle-2 order 4): at the hairline/temples the
mesh fuses a wispy hair film onto the head as ONE surface. The layered-zone
gate cannot see it (no second sheet => layered density ~0.02-0.05 << 0.10)
while the photo pixels there are bright skin+hair MIXTURES, so mixture
stamps win texels and render as a beige "painted sheet"; the surrendered
remainder inherits the harmonic membrane's mixed skin+hair tone (pale
curtain), and dark side-view curls interleave with both (flake mottle).

THE MECHANISM, measured piece by piece on the face proof asset:

1. ZONE EXTENSION (hysteresis): grow each view's strong layered zone into
   connected weak evidence — any layered density at high contrast near the
   DARK-MATERIAL MAIN BODY with substantial local dark coverage. The dark
   coverage gate (df > 0.25) keeps sparse wisp tips over plain skin (brow
   fringe, lash shelves) un-surrendered: vacating those lets other views'
   grazing claims land as dark dashes.

2. FILM COMMITMENT with multi-view consensus. A texel is committed to the
   film material only when
   - some view's extended zone flags it first-surface in a LARGE component
     (small zone islands are local ambiguities the membrane handles), and
   - NO view vetoes it: a view that images the texel inside its silhouette
     without zone-flagging it and with low dark coverage at its bin
     (df < 0.25) positively witnesses base material along that ray, and
   - EVERY view that images it first-surface flags it (flag consensus).
     A fused wisp floater aligns with the dark body only from one pose;
     other views image it over base content, and a dark commit detaches
     under parallax into a floating blob over skin (the FACE-16 class).

3. COMMIT-COUPLED SURRENDER: claims inside a view's ADDED (extension-only)
   zone are vacated exactly where the commit mask holds. Surrendering
   without committing leaves the membrane anchored by whatever survives
   nearby — measured: lash-dark anchors bled through a vacated eyelid rim
   as a floating dash. Where we are not confident enough to commit, the
   baseline mixture claim is strictly safer.

4. FILM RETONE: committed fill texels take their tone from DARK-MATERIAL
   OBSERVED anchors only (octant-binned voxel-ball interpolation at
   growing scales), replacing the mixed membrane tone. The blend is scaled
   by a photo-space wispiness weight (local dark-material coverage): dense
   film commits fully, sparse wisp mixtures keep the membrane.

Everything here is scale-free (thresholds relative to foreground medians,
window sizes relative to photo size, distances relative to the mesh
diagonal) and carries no subject-specific semantics.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

# Dark-material split: below 0.55 x the median of the bright half of the
# foreground luminance (two-mode split without color semantics).
DARK_LUMINANCE_RATIO = 0.55
# Bright/base split, relative to the same bright-half median.
BRIGHT_LUMINANCE_RATIO = 0.75
# Base-material witness: a non-flagged bin with less than this local dark
# coverage vetoes film commitment along its ray.
VETO_DARK_COVERAGE = 0.25
# Weak-evidence gate for the hysteresis growth.
WEAK_DENSITY = 0.02
WEAK_DARK_COVERAGE = 0.25
# Wispiness ramp: local dark coverage 0.15 -> 0.50 maps to commit 0 -> 1.
WISP_RAMP_LO = 0.15
WISP_RAMP_SPAN = 0.35
# Skin-side falloff profile: height at which the hairline transition is
# considered complete; defines each photo's transition length.
PROFILE_END = 0.10


def dark_body_mask(image_rgba01: Any, min_area_frac: float = 0.02) -> Any:
    """Large connected dark-material components of the photo foreground.

    "Dark material" is luminance below `DARK_LUMINANCE_RATIO` x the median
    of the bright half of the foreground — a scale-free two-mode split.
    Small dark components (eyes, nostrils, brows) are features of the base
    material, not the film body, and are excluded by the area floor.
    """
    import numpy as np
    from scipy.ndimage import label

    alpha = image_rgba01[:, :, 3] > 0.5
    if not alpha.any():
        return np.zeros(alpha.shape, dtype=bool)
    luminance = image_rgba01[:, :, :3].mean(axis=2)
    foreground = luminance[alpha]
    bright_median = float(np.median(foreground[foreground >= np.median(foreground)]))
    dark = alpha & (luminance < DARK_LUMINANCE_RATIO * bright_median)
    labels, count = label(dark, structure=np.ones((3, 3), dtype=bool))
    if count == 0:
        return np.zeros(alpha.shape, dtype=bool)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    min_area = float(min_area_frac) * float(alpha.sum())
    keep = np.zeros(count + 1, dtype=bool)
    keep[1:] = sizes[1:] >= min_area
    return keep[labels]


def skin_side_profile(image_rgba01: Any, max_px: int = 80):
    """Skin-anchored hairline falloff of one photo.

    Median luminance vs distance INTO the hair from the photo's
    confident-skin region, restricted to the dark-body transition corridor
    (face-interior features never enter the statistic), normalized to 1 at
    the skin plateau and 0 at the body tone, isotonic-decreasing.

    The SKIN side anchors the measurement deliberately: on fused-film
    meshes the observed dark mass under-covers the true hair volume (its
    boundary trails into the wisp apron), while the confident-skin region
    is compact and well-localized — the same asymmetry the photos show.
    Returns (distances_px, profile) or None (no dark body / skin region).
    """
    import numpy as np
    from scipy.ndimage import distance_transform_edt

    image = np.asarray(image_rgba01, dtype=np.float32)
    alpha = image[:, :, 3] > 0.5
    if not alpha.any():
        return None
    luminance = image[:, :, :3].mean(axis=2)
    body = dark_body_mask(image)
    if not body.any():
        return None
    body_lum = float(np.median(luminance[body]))
    foreground = luminance[alpha]
    plateau = float(np.median(foreground[foreground >= np.median(foreground)]))
    skin = alpha & (luminance >= BRIGHT_LUMINANCE_RATIO * plateau)
    if not skin.any():
        return None
    dist_skin = distance_transform_edt(~skin)
    dist_body = distance_transform_edt(~body)
    corridor = alpha & ~skin & (dist_body < max_px)
    distances: list = []
    tones: list = []
    for d0 in range(0, max_px, 2):
        band = corridor & (dist_skin >= d0) & (dist_skin < d0 + 2)
        if band.sum() > 30:
            distances.append(d0 + 1.0)
            tones.append(float(np.median(luminance[band])))
    if len(distances) < 4:
        return None
    profile = np.clip((np.asarray(tones) - body_lum)
                      / max(plateau - body_lum, 1e-6), 0.0, 1.0)
    return np.asarray(distances), np.minimum.accumulate(profile)


def photo_feature_components(image_rgba01: Any, transition_px: float) -> Any:
    """Soft-dark FEATURE components of a photo (brows, lids, lips).

    A soft-dark connected component that lies mostly OUTSIDE the hairline
    transition corridor is a facial feature: billboarding its content onto
    fused film geometry re-paints it parallax-displaced (the third-eye /
    doubled-brow class). Feature-sized only — large soft-dark regions
    (shaded hair, garments) are not features. Returns a photo-space mask.
    """
    import numpy as np
    from scipy.ndimage import binary_dilation, distance_transform_edt, label

    image = np.asarray(image_rgba01, dtype=np.float32)
    alpha = image[:, :, 3] > 0.5
    if not alpha.any():
        return np.zeros(alpha.shape, dtype=bool)
    luminance = image[:, :, :3].mean(axis=2)
    body = dark_body_mask(image)
    foreground = luminance[alpha]
    plateau = float(np.median(foreground[foreground >= np.median(foreground)]))
    soft_dark = alpha & (luminance < 0.72 * plateau) & ~body
    labels, count = label(soft_dark, structure=np.ones((3, 3), bool))
    if not count:
        return np.zeros(alpha.shape, dtype=bool)
    corridor = distance_transform_edt(~body) <= float(transition_px)
    in_corridor = np.bincount(labels.ravel(),
                              weights=corridor.ravel().astype(np.float64),
                              minlength=count + 1)
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    with np.errstate(invalid="ignore"):
        corridor_fraction = in_corridor / np.maximum(sizes, 1)
    photo_area = float(alpha.sum())
    is_feature = np.zeros(count + 1, dtype=bool)
    is_feature[1:] = ((corridor_fraction[1:] < 0.2)
                      & (sizes[1:] >= 64)
                      & (sizes[1:] <= 0.02 * photo_area))
    return binary_dilation(is_feature[labels], iterations=2)


def _grow_from_seeds(weak: Any, seeds: Any) -> Any:
    """Keep weak-mask components that contain at least one seed pixel."""
    import numpy as np
    from scipy.ndimage import label

    combined = weak | seeds
    labels, count = label(combined, structure=np.ones((3, 3), dtype=bool))
    if count == 0:
        return seeds.copy()
    seeded = np.unique(labels[seeds])
    seeded = seeded[seeded > 0]
    keep = np.zeros(count + 1, dtype=bool)
    keep[seeded] = True
    return keep[labels] & combined


def compute_view_film_maps(
    *,
    image_rgba01: Any,
    zone_map: Any,
    density: Any,
    local_std: Any,
    window: int,
    min_contrast: float,
    bins_y: Any,
    bins_x: Any,
    infront: Any,
    first_surface: Any,
) -> Dict[str, Any]:
    """Per-view film-band maps, computed in the projector's photo space.

    Returns texel-space masks aligned with the atlas plus the wispiness
    weight. All inputs are the projector's own zone-gate internals; this
    adds only the dark-body statistics and the hysteresis growth.
    """
    import numpy as np
    from scipy.ndimage import binary_dilation, distance_transform_edt, uniform_filter

    body = dark_body_mask(image_rgba01)
    texel_shape = infront.shape
    empty = np.zeros(texel_shape, dtype=bool)
    if not body.any():
        return {
            "zone_texel": empty,
            "added_texel": empty.copy(),
            "commit_texel": empty.copy(),
            "veto_texel": empty.copy(),
            "img_first_texel": (infront & first_surface
                                & (image_rgba01[:, :, 3] > 0.5)[bins_y, bins_x]),
            "contested_texel": empty.copy(),
            "body_weight_texel": np.zeros(texel_shape, dtype=np.float32),
        }

    photo_alpha = image_rgba01[:, :, 3]
    # Dark-material coverage OF THE FOREGROUND: normalizing by the
    # window's foreground coverage keeps the statistic meaningful at
    # silhouette rims where wisps hang over background (background
    # witnesses nothing about material; raw coverage read those bins as
    # "base" and left beige ribbons at the rear-quarter temple, measured).
    foreground_coverage = uniform_filter(
        (photo_alpha > 0.5).astype(np.float32), size=window, mode="constant")
    dark_coverage = uniform_filter(
        body.astype(np.float32), size=window, mode="constant"
    ) / np.maximum(foreground_coverage, 0.05)
    body_distance = distance_transform_edt(~body)
    support = body_distance <= float(window)

    weak = (
        (local_std > float(min_contrast))
        & (density > WEAK_DENSITY)
        & (photo_alpha > 0.1)
        & (dark_coverage > WEAK_DARK_COVERAGE)
        & support
    )
    extended = _grow_from_seeds(weak, zone_map)
    added = extended & ~zone_map

    # Commit membership: large extended components only. Small islands
    # (lash shelves, specular pockets) are local ambiguities the membrane
    # handles well; committing them painted black eye-region flecks.
    from scipy.ndimage import label

    labels, count = label(extended, structure=np.ones((3, 3), dtype=bool))
    commit_zone = np.zeros_like(extended)
    if count:
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        keep = np.zeros(count + 1, dtype=bool)
        keep[1:] = sizes[1:] >= (2 * window) ** 2
        commit_zone = keep[labels]

    # Texel-space maps. A photo-space zone describes the surface the photo
    # IMAGES; texels deeper along the ray (a brow behind a wispy fringe)
    # must not inherit its flags (measured: ray-swath flags re-toned the
    # brow into a dark blob).
    zone_texel = infront & extended[bins_y, bins_x] & first_surface
    added_texel = infront & added[bins_y, bins_x] & first_surface
    commit_texel = infront & commit_zone[bins_y, bins_x] & first_surface
    wide = binary_dilation(extended, iterations=max(1, window // 2))
    contested_texel = infront & wide[bins_y, bins_x] & first_surface

    # Base-material witness (loose: any depth along the ray). Restricting
    # the veto to a first-surface corridor cannot work here — the fused
    # geometry hangs floaters up to ~0.4 diagonal in front of the surface
    # they obscure — and a false through-head veto only preserves baseline
    # claims, which costs nothing new. Bins outside the silhouette carry
    # zero dark coverage and veto too: a texel that projects outside a
    # view's foreground is not film-confirmable by that view.
    alpha_confident = photo_alpha > 0.5
    veto_texel = (
        infront
        & ~extended[bins_y, bins_x]
        & (dark_coverage[bins_y, bins_x] < VETO_DARK_COVERAGE)
    )

    img_first_texel = infront & first_surface & alpha_confident[bins_y, bins_x]

    wisp = np.clip((dark_coverage - WISP_RAMP_LO) / WISP_RAMP_SPAN, 0.0, 1.0)
    body_weight_texel = np.where(
        infront, wisp[bins_y, bins_x], 0.0).astype(np.float32)

    # Photo-space products for the gradient repaint (film_band_gradient):
    # the photo's own hairline falloff + territory maps in texel space.
    luminance = image_rgba01[:, :, :3].mean(axis=2)
    foreground_lum = luminance[image_rgba01[:, :, 3] > 0.5]
    view_bright_median = float(np.median(
        foreground_lum[foreground_lum >= np.median(foreground_lum)])) if (
        foreground_lum.size) else 1.0
    profile = skin_side_profile(image_rgba01)
    transition_px = float(window)
    if profile is not None:
        below = np.nonzero(profile[1] <= PROFILE_END)[0]
        transition_px = float(
            profile[0][below[0]] if len(below) else profile[0][-1])
    feature = photo_feature_components(image_rgba01, transition_px)
    photo_products = {
        "bright_median": view_bright_median,
        "profile": profile,
        "transition_px": transition_px,
        "in_body_texel": infront & body[bins_y, bins_x],
        "beyond_transition_texel": infront & (
            body_distance >= transition_px)[bins_y, bins_x],
        "feature_texel": infront & feature[bins_y, bins_x],
        # The photo's own alpha at the texel's bin: the projected rgba
        # alpha is zeroed wherever the weight is zero (zone-vacated
        # claims), which is exactly the apron the repaint must reach.
        "solid_texel": infront & alpha_confident[bins_y, bins_x],
        # Texel -> photo-bin mapping for the repaint's outermost-sheet
        # (depth corridor) test. The first-surface flag alone gets coarser
        # as texel density drops (fewer texels per bin) and admits inner
        # curtain sheets at low resolutions — measured at 1024.
        "bins_y": bins_y,
        "bins_x": bins_x,
        "photo_shape": image_rgba01.shape[:2],
    }

    return {
        "zone_texel": zone_texel,
        "added_texel": added_texel,
        "commit_texel": commit_texel,
        "veto_texel": veto_texel,
        "img_first_texel": img_first_texel,
        "contested_texel": contested_texel,
        "body_weight_texel": body_weight_texel,
        "photo_products": photo_products,
    }


def _winner_dark_dominance(
    projections: Sequence[Mapping[str, Any]],
    *,
    surface_mask: Any,
    positions_texture: Any,
) -> Any:
    """Dark-dominance of the WOULD-BE observed context, in texel space.

    For each surface texel, the winning claim's luminance classifies it as
    dark or bright content; voxel-ball ratios of dark vs bright claimed
    texels (at two scales, taking the stricter) measure whether the local
    observed context is dark-dominated. Film commitment is scaled by this:
    committing hard against surviving bright context creates flake-island
    contrast (measured at az-135), while deep inside the band the context
    is dark-only and the commit stays full.
    """
    import numpy as np

    from .texturing import _voxel_neighborhood_mean

    shape = surface_mask.shape
    weight_stack = np.stack(
        [np.asarray(p["weight"], dtype=np.float32) for p in projections],
        axis=0,
    )
    winner = weight_stack.argmax(axis=0)
    winner_weight = weight_stack.max(axis=0)
    luminance = np.zeros(shape, dtype=np.float32)
    for index, projection in enumerate(projections):
        rgba = np.asarray(projection["rgba"], dtype=np.float32)
        sel = winner == index
        luminance[sel] = rgba[:, :, :3].mean(axis=2)[sel]
    claimed = surface_mask & (winner_weight > 1e-6)
    if not claimed.any():
        return np.ones(int(surface_mask.sum()), dtype=np.float64)
    claimed_luminance = luminance[claimed]
    bright_median = float(np.median(
        claimed_luminance[claimed_luminance >= np.median(claimed_luminance)]))
    dark = claimed & (luminance < DARK_LUMINANCE_RATIO * bright_median)
    bright = claimed & (luminance >= 0.75 * bright_median)

    positions = np.asarray(positions_texture, dtype=np.float32)
    points = positions[:, :, :3][surface_mask].astype(np.float64)
    covered = positions[:, :, :3][surface_mask]
    diagonal = float(np.linalg.norm(covered.max(axis=0) - covered.min(axis=0)))
    dominance = np.ones(len(points), dtype=np.float64)
    for context_frac in (0.01, 0.02):
        dark_context = _voxel_neighborhood_mean(
            points, dark[surface_mask].astype(np.float64),
            context_frac * diagonal, select=claimed[surface_mask])
        bright_context = _voxel_neighborhood_mean(
            points, bright[surface_mask].astype(np.float64),
            context_frac * diagonal, select=claimed[surface_mask])
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = dark_context / np.maximum(
                dark_context + bright_context, 1e-6)
        ratio = np.where(np.isfinite(ratio), ratio, 1.0)
        dominance = np.minimum(
            dominance, np.clip((ratio - 0.5) / 0.4, 0.0, 1.0))
    return dominance


def commit_film_band(
    projections: Sequence[Mapping[str, Any]],
    *,
    surface_mask: Any,
    positions_texture: Any,
) -> Optional[Dict[str, Any]]:
    """Combine per-view film maps into the commit mask and vacate claims.

    Mutates each projection's `weight` (vacating bright mixture claims
    under the commit mask) and `contested` (extension-zone dilation joins
    the mirror-source exclusion). Returns the commit state consumed by
    the downstream stages, or None when no view carries film maps.
    """
    import numpy as np

    views: List[Mapping[str, Any]] = [
        p for p in projections if isinstance(p.get("film_band"), dict)
    ]
    if not views or len(projections) < 2:
        return None

    shape = surface_mask.shape
    band = np.zeros(shape, dtype=bool)
    veto = np.zeros(shape, dtype=bool)
    body_weight = np.zeros(shape, dtype=np.float32)
    n_img = np.zeros(shape, dtype=np.int8)
    n_flag = np.zeros(shape, dtype=np.int8)
    for projection in views:
        maps = projection["film_band"]
        band |= maps["commit_texel"]
        veto |= maps["veto_texel"]
        body_weight = np.maximum(
            body_weight,
            np.where(maps["commit_texel"], maps["body_weight_texel"], 0.0),
        )
        n_img += maps["img_first_texel"].astype(np.int8)
        n_flag += (maps["img_first_texel"]
                   & maps["commit_texel"]).astype(np.int8)

    # Cross-confirmation: at least two views must image the texel first-
    # surface and all of them must flag it. With a single imaging witness
    # the consensus is vacuous, and single-witness commits were measured
    # to paint dark spots at silhouette-adjacent skin (ear rim, crown
    # edge) where only one profile reaches — exactly the parallax-floater
    # geometry the consensus exists to exclude.
    commit = band & ~veto & (n_flag >= n_img) & (n_img >= 2)

    # Dark-dominance of the local observed context scales both halves of
    # the commitment (claim vacate here, fill retone downstream): a
    # commit adjacent to surviving bright content flips coherent pale
    # ribbons into flake-island contrast instead of removing them.
    dominance_surface = _winner_dark_dominance(
        projections, surface_mask=surface_mask,
        positions_texture=positions_texture)
    dominance = np.zeros(shape, dtype=np.float32)
    dominance[surface_mask] = dominance_surface

    vacated = 0
    for projection in views:
        maps = projection["film_band"]
        # Vacate BRIGHT mixture claims of EVERY view at committed texels
        # whose context is dark-dominated: the commit is a texel-level
        # consensus decision, so any bright claim there is base-mixture
        # content on film material regardless of which view flagged the
        # texel. Dark stamps are film-consistent content ("hair where
        # hair belongs") and stay — removing them paled the rear-quarter
        # temple ribbons over the crown-flake gate (measured az-135:
        # 0.0006 -> 0.0027 with the unconditional vacate).
        rgba = np.asarray(projection["rgba"], dtype=np.float32)
        luminance = rgba[:, :, :3].mean(axis=2)
        valid = rgba[:, :, 3] > 0.0
        if valid.any():
            valid_luminance = luminance[valid]
            bright_median = float(np.median(
                valid_luminance[valid_luminance >= np.median(valid_luminance)]))
        else:
            bright_median = 1.0
        bright_like = luminance > DARK_LUMINANCE_RATIO * bright_median
        vacate = commit & bright_like & (dominance > 0.5)
        if vacate.any():
            weight = np.asarray(projection["weight"], dtype=np.float32)
            weight[vacate] = 0.0
            projection["weight"] = weight
            vacated += int(vacate.sum())
        contested = np.asarray(projection.get("contested", False))
        if contested.shape == shape:
            projection["contested"] = contested | maps["contested_texel"]
        else:
            projection["contested"] = maps["contested_texel"].copy()

    return {
        "commit_mask": commit,
        "band_mask": band,
        "body_weight": body_weight,
        "stats": {
            "applied": bool(commit.any()),
            "band_texels": int(band.sum()),
            "commit_texels": int(commit.sum()),
            "vacated_claims": vacated,
        },
    }


def demote_unwitnessed_rim(
    blend: Mapping[str, Any],
    *,
    zone_union: Any,
) -> int:
    """Drop rim coverage inside the film zone that carries no weight.

    When the surrender vacates the winning claim, a residual low-weight
    speck from another view can survive as raw coverage yet lose its
    weight to feathering; the blend's rim path then verbatim-copies the
    nearest observed UV neighbor (measured: a dark lash copied onto the
    vacated eyelid as a floating dash). A vacated texel with no
    weight-bearing witness is FILL, not observed. Mutates blend coverage.
    """
    import numpy as np

    coverage = np.asarray(blend["coverage"], dtype=bool)
    weight = np.asarray(blend["weight"], dtype=np.float32)
    rim_bad = coverage & (weight <= 1e-6) & zone_union
    if rim_bad.any():
        blend["coverage"] = coverage & ~rim_bad
    return int(rim_bad.sum())


def retone_film_band(
    colors_rgba: Any,
    *,
    positions_texture: Any,
    observed_mask: Any,
    commit_mask: Any,
    body_weight: Any,
    normals_texture: Optional[Any] = None,
    feather_texels: float = 2.0,
) -> Tuple[Any, Dict[str, Any]]:
    """Re-tone committed fill texels from dark-material observed anchors.

    The harmonic membrane mixes both sides of the material boundary; the
    committed band is the dark body's continuation, so its tone comes from
    dark OBSERVED anchors only — octant-binned voxel-ball means at growing
    scales (the film's own tone varies along the band; a global mean would
    flatten it). The commit is feathered at the observed boundary and
    scaled by the wispiness weight.
    """
    import numpy as np

    from .texturing import _voxel_neighborhood_mean

    stats: Dict[str, Any] = {"applied": False, "retoned_texels": 0}
    rgba = np.asarray(colors_rgba, dtype=np.float32)
    positions = np.asarray(positions_texture, dtype=np.float32)
    surface = positions[:, :, 3] > 0.0
    observed = np.asarray(observed_mask, dtype=bool) & surface
    band_fill = np.asarray(commit_mask, dtype=bool) & surface & ~observed
    if not band_fill.any() or not observed.any():
        return rgba, stats

    luminance = rgba[:, :, :3].mean(axis=2)
    observed_luminance = luminance[observed]
    bright_median = float(np.median(
        observed_luminance[observed_luminance >= np.median(observed_luminance)]))
    dark_anchor = observed & (
        luminance < DARK_LUMINANCE_RATIO * bright_median)
    if int(dark_anchor.sum()) < 64:
        return rgba, stats

    positions_xyz = positions[:, :, :3]
    covered = positions_xyz[surface]
    diagonal = float(np.linalg.norm(covered.max(axis=0) - covered.min(axis=0)))
    octants = None
    if normals_texture is not None:
        normals = np.asarray(normals_texture, dtype=np.float32)[:, :, :3]
        norms = np.linalg.norm(normals, axis=2, keepdims=True)
        normals = np.divide(normals, np.maximum(norms, 1e-8))
        surface_normals = normals[surface]
        axis = np.abs(surface_normals).argmax(axis=1)
        sign = np.take_along_axis(
            surface_normals, axis[:, None], axis=1)[:, 0] < 0
        octants = axis.astype(np.int64) * 2 + sign.astype(np.int64)

    points = positions_xyz[surface].astype(np.float64)
    anchor_select = dark_anchor[surface]
    prior = np.full((int(surface.sum()), 3), np.nan)
    for cell_frac in (0.01, 0.02, 0.04):
        need = ~np.isfinite(prior[:, 0])
        if not need.any():
            break
        gathered = np.zeros((len(points), 3))
        for channel in range(3):
            gathered[:, channel] = _voxel_neighborhood_mean(
                points,
                rgba[:, :, channel][surface].astype(np.float64),
                cell_frac * diagonal,
                select=anchor_select,
                octants=octants,
            )
        prior[need] = gathered[need]

    band_in_surface = band_fill[surface]
    have = band_in_surface & np.isfinite(prior[:, 0])
    if not have.any():
        return rgba, stats

    # DARK-DOMINANCE scaling: a committed texel whose local OBSERVED
    # context still contains surviving bright content (kept wisp ribbons
    # at silhouette rims) must not be pulled fully dark — the hard
    # contrast step against its bright neighbors reads as flake islands
    # (measured at az-135: crown-flake fraction 0.0006 -> 0.0027 with the
    # unscaled retone). Deep inside the band the observed context is
    # dark-only and the commit stays full.
    bright_anchor = observed & (
        luminance >= 0.75 * bright_median)
    dominance = np.ones(len(points), dtype=np.float64)
    for context_frac in (0.01, 0.02):
        dark_context = _voxel_neighborhood_mean(
            points, dark_anchor[surface].astype(np.float64),
            context_frac * diagonal, select=observed[surface])
        bright_context = _voxel_neighborhood_mean(
            points, bright_anchor[surface].astype(np.float64),
            context_frac * diagonal, select=observed[surface])
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = dark_context / np.maximum(
                dark_context + bright_context, 1e-6)
        ratio = np.where(np.isfinite(ratio), ratio, 1.0)
        dominance = np.minimum(
            dominance, np.clip((ratio - 0.5) / 0.4, 0.0, 1.0))

    from scipy.ndimage import distance_transform_edt

    distance = distance_transform_edt(~observed)
    ramp = np.clip(distance[surface] / float(feather_texels), 0.0, 1.0)
    weight = np.asarray(body_weight, dtype=np.float32)[surface]
    rows, cols = np.nonzero(surface)
    blend_factor = (ramp[have] * weight[have] * dominance[have])[:, None]
    rgba[rows[have], cols[have], :3] = (
        prior[have] * blend_factor
        + rgba[rows[have], cols[have], :3].astype(np.float64)
        * (1.0 - blend_factor)
    ).astype(np.float32)
    stats = {
        "applied": True,
        "retoned_texels": int(have.sum()),
        "dark_anchors": int(dark_anchor.sum()),
        "mean_commit_weight": round(float(weight[have].mean()), 3),
    }
    return rgba, stats
