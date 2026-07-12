"""Feature-aware fringe repair inside protected feature regions.

THE DEFECT CLASS (cycle-3 ledger FACE-03/04 residue): small deposits of
displaced source-photo content INSIDE feature complexes — tear-duct white
chips, lash-line dark dashes, lip-edge dark-red dashes. Their texels sit
in feature context by every measurement (cycle-3: surround-ring votes
0.30-0.81 vs the 0.96 consensus bar; bright deposits at core-distance p50
<= 0.056 vs the 1.4x halo), so `commit_trace_deposits` correctly refuses
them: committing via surround consensus was measured to wash eye corners
and drop eye counts. The repair lane that CAN treat them must be
feature-aware — replace the deposit with the feature's own evidence — not
witness demotion (both demotion variants measured as regressions,
cycle-2).

THE EVIDENCE SOURCE: the identity correspondence. The source photo is
ground truth at its declared pose; the identity gate scores the render at
that pose against the photo under a global alignment (alpha-bbox map +
NCC-refined similarity). Under THAT correspondence the photo shows clean
lid skin where the bake shows a tear-duct chip and a soft lip edge where
the bake shows a hard dash — because the deposits are the same photo's
content displaced by the per-feature geometry-photo mismatch (measured
cycle-1: eyes/mouth/nose each want a different few-px correction; no
global transform satisfies all three). This module rebuilds the gate's
correspondence inside the bake (same construction, same renderer),
samples the photo through it per texel, and repairs feature fringes with
RESCUE-DISC TRANSPLANT SEMANTICS — tone-matched, feathered, whole-patch —
with the photo as the twin.

MECHANISM (all gates measured on the face proof at 1024, /tmp/c4_2):

1. FEATURE COMPLEXES: confident strong-contrast dark cores (the
   commit_trace_deposits core signal) clustered at feature scale, plus
   the bake's own mirror-rescue discs (feature complexes by construction).
   No feature classes, no hand masks.
2. COMPLEX STAMP: within a complex's world ball, re-sample the source
   photo under the gate correspondence and stamp it as ONE coherent
   patch. Atlas-connectivity trimming is forbidden (a UV-chart cut left
   the old lip line isolated: debris 0.0017->0.0033 at az0); occlusion
   holes blend smoothly from visible neighbors (nearest-copy duplicated a
   lip edge across the hidden band and the az0 eye detector counted it as
   a THIRD eye). Never-demote applies across witnesses: texels a
   NON-SOURCE view confidently won are never overwritten (that would be
   demotion); texels the source itself confidently won may be
   re-registered (same witness, gate correspondence) in the FULL mode,
   with a TRACE-ONLY fallback mode keeping every confident texel.
3. STRUCTURE VETOES, ladder full -> trace -> skip: a stamp must neither
   create nor destroy a feature-scale compact bright-ringed dark blob.
   Texel-space check first (shading-modelled: shade = 0.88+0.12*diffuse,
   the renderer's own fragment shading), then a render-space check with
   the pipeline's own renderer at the near-frontal and profile views
   (relative to the pre-repair render: no new eye-scale blob, no isolated
   dark-debris growth). Measured: the full-mode mouth stamp reshaped the
   photo's soft lip slit into a compact 45px dark blob that the az0 eye
   detector counted as a third eye — the render veto catches it and the
   trace fallback ships clean (+0.0070 SSIM at the mouth).
4. DEPOSIT PATCHES: gate-contradicted trace-witness deposits outside the
   stamped complexes get deposit-scale patches of the same photo
   evidence (whole-patch, tone-matched on the deposit ring, feathered).
5. RESCUE-DISC LANE: the disc interior is NEVER photo-stamped (the disc
   fired because the photo evidence there is bad). Deposits in the
   protected fringe ring re-copy through the disc's own anchored
   correspondence (mirror + placement_shift) from confident witnesses;
   the disc itself is refreshed LAST (whole-disc re-transplant on the
   current colors) so root repairs on the healthy side propagate into
   the twin — the fringe chips at the transplanted eye were measured to
   be COPIES of the healthy side's chips (the disc replicated the
   tear-duct chip at the mirrored position + placement shift).

MEASURED END-TO-END (face proof, 1024, this mechanism on the C4 baseline
tree): compensated identity[front] 0.6678 -> 0.6936 SSIM / 16.26 -> 15.32
MAE (raw 0.6431 -> 0.6681 / 21.62 -> 21.05); full 28-view battery: zero
new detector failures; tear-duct chips and mouth-area chips clean at 4x.
Single-view bakes are structural no-ops (the correspondence needs a
multi-view identity contract; single-photo proof assets stay
bit-identical).
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

__all__ = ["repair_feature_fringes"]


# --------------------------------------------------------------------------
# renderer camera + gate correspondence
# --------------------------------------------------------------------------

def _renderer_camera(mesh: Any, azimuth_deg: float, elevation_deg: float,
                     size: int) -> Dict[str, Any]:
    """Reproduce `rendering.render_mesh_views`'s camera analytically:
    orthographic, eye on the unit sphere at distance 3.2 (normalized
    space), up +z, half-extent 1.18x the max |xy| of the normalized
    vertices in camera space. Returns projection callables and the
    px-per-world scale used by the vetoes."""
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    center = 0.5 * (vertices.min(axis=0) + vertices.max(axis=0))
    radius = float(np.max(np.linalg.norm(vertices - center, axis=1))) or 1.0

    azr, elr = math.radians(float(azimuth_deg)), math.radians(float(elevation_deg))
    eye = np.array([math.cos(elr) * math.cos(azr),
                    math.cos(elr) * math.sin(azr),
                    math.sin(elr)], dtype=np.float32)
    forward = -eye / np.linalg.norm(eye)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)
    centered = (vertices - center) / radius
    cam_xy = np.stack([centered @ right, centered @ true_up], axis=1)
    half_extent = float(np.max(np.abs(cam_xy))) * 1.18

    def project(points: Any) -> Tuple[Any, Any, Any]:
        p = (np.asarray(points, dtype=np.float32).reshape(-1, 3) - center) / radius
        x = (p @ right) / half_extent
        y = (p @ true_up) / half_extent
        px = (x * 0.5 + 0.5) * size
        py = (1.0 - (y * 0.5 + 0.5)) * size
        depth = -(p @ eye)
        return px, py, depth

    return {
        "project": project,
        "eye": eye,
        "px_per_world": size / (2.0 * half_extent * radius),
    }


def _render_with_colors(mesh: Any, atlas: Mapping[str, Any], colors_rgba: Any,
                        azimuths: Sequence[float], elevation: float,
                        size: int) -> List[Any]:
    """Render the mesh carrying `colors_rgba` exactly as the final export
    does (texture image + edge bleed + baked-mesh construction), through
    the repository renderer."""
    from .backends.triposr_runtime import (
        _tripo_build_textured_mesh,
        _tripo_edge_bleed_texture,
        _tripo_texture_image,
    )
    from .rendering import render_mesh_views

    texture_image = _tripo_edge_bleed_texture(_tripo_texture_image(
        np.asarray(colors_rgba, dtype=np.float32)))
    textured = _tripo_build_textured_mesh(
        mesh,
        bake_output={"vmapping": atlas["vmapping"], "indices": atlas["indices"],
                     "uvs": atlas["uvs"]},
        texture_image=texture_image,
    )
    images = render_mesh_views(textured, size=size, azimuths=list(azimuths),
                               elevation=float(elevation))
    return [np.asarray(img) for img in images]


def _render_foreground(rgb: Any) -> Any:
    """Foreground of a repo render: pixels away from the uniform clear
    color, largest component, interior holes filled (scipy form of the
    harness construction)."""
    from scipy.ndimage import binary_fill_holes
    from scipy.ndimage import label as cc_label

    rgb = np.asarray(rgb)
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]],
                            axis=0).reshape(-1, 3)
    background = np.median(border, axis=0)
    distance = np.abs(rgb.astype(np.int16) - background.astype(np.int16)).max(axis=2)
    fg = distance > 14
    labels, count = cc_label(fg)
    if count > 1:
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        fg = labels == int(sizes.argmax())
    return binary_fill_holes(fg)


def _bilinear(image: Any, xs: Any, ys: Any) -> Any:
    h, w = image.shape[:2]
    xs = np.clip(xs, 0.0, w - 1.001)
    ys = np.clip(ys, 0.0, h - 1.001)
    x0 = xs.astype(np.int32)
    y0 = ys.astype(np.int32)
    fx = xs - x0
    fy = ys - y0
    if image.ndim == 3:
        fx = fx[..., None]
        fy = fy[..., None]
    return (image[y0, x0] * (1 - fx) * (1 - fy)
            + image[y0, x0 + 1] * fx * (1 - fy)
            + image[y0 + 1, x0] * (1 - fx) * fy
            + image[y0 + 1, x0 + 1] * fx * fy)


def _register_photo_to_render(photo_rgba: Any, render_rgb: Any,
                              render_fg: Any) -> Tuple[Any, Any, Tuple]:
    """The identity gate's own correspondence: anisotropic alpha-bbox map
    of the photo onto the render's foreground bbox, then a grid-searched
    2D similarity residual maximizing masked luminance NCC. The repair
    must target exactly the correspondence the gate scores — a private
    registration would bank nothing."""
    from scipy.ndimage import zoom as nd_zoom

    from scipy.ndimage import binary_dilation, binary_erosion
    from scipy.ndimage import label as cc_label

    photo = np.asarray(photo_rgba, dtype=np.float32)
    photo_rgb = photo[:, :, :3]
    if photo.shape[2] == 4 and float(photo[:, :, 3].min()) <= 8.0:
        # a real matte: alpha carries the silhouette
        photo_fg = photo[:, :, 3] > 8
    else:
        # RGB photo (or opaque alpha): background distance against the
        # photo's own BORDER median, not literal white. The white-distance
        # rule (|rgb-255| > 18) silently classified a neutral-gray studio
        # background as foreground (measured on the car photo: the whole
        # frame became "foreground", the bbox map degenerated, the NCC
        # residual pegged at the search boundary, and the repair stamped
        # a miniature of the photo — background included — onto the
        # hood). Studio photos put background at every border; the
        # border-median estimate reduces to the white rule on white-
        # background photos and generalizes to gray ones.
        border = np.concatenate([
            photo_rgb[0, :], photo_rgb[-1, :],
            photo_rgb[:, 0], photo_rgb[:, -1],
        ], axis=0)
        background = np.median(border, axis=0)
        photo_fg = np.abs(photo_rgb - background[None, None, :]).max(axis=2) > 18
    # the identity gate's foreground rule: largest component, closed,
    # then hole-filling BY FLOOD FROM THE (0,0) CORNER — background
    # pockets connected to other borders but not to that corner count as
    # holes and are filled (measured: 52k px of bottom-frame pockets on
    # the face photo; a border-connected fill instead shrinks the bbox
    # and lands the correspondence in a different basin)
    labels, count = cc_label(photo_fg)
    if count > 1:
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        photo_fg = labels == int(sizes.argmax())
    # closing with NEUTRAL borders (dilate border 0, erode border 1):
    # a symmetric border value lets the erosion eat border-touching
    # foreground, which measurably shrank the bbox by 45 px on a photo
    # whose subject touches the frame and re-based the whole bbox map
    structure = np.ones((7, 7), bool)
    photo_fg = binary_erosion(
        binary_dilation(photo_fg, structure=structure, border_value=0),
        structure=structure, border_value=1)
    inverse_labels, _ = cc_label(~photo_fg)
    outside = inverse_labels[0, 0]
    photo_fg = photo_fg | (~photo_fg & (inverse_labels != outside))
    pys, pxs = np.nonzero(photo_fg)
    rys, rxs = np.nonzero(render_fg)
    if len(pys) < 64 or len(rys) < 64:
        return None, None, (1.0, 0.0, 0.0)
    px0, px1, py0, py1 = pxs.min(), pxs.max(), pys.min(), pys.max()
    rx0, rx1, ry0, ry1 = rxs.min(), rxs.max(), rys.min(), rys.max()
    rw, rh = int(rx1 - rx0 + 1), int(ry1 - ry0 + 1)
    crop = photo_rgb[py0:py1 + 1, px0:px1 + 1]
    crop_fg = photo_fg[py0:py1 + 1, px0:px1 + 1]
    warped_crop = nd_zoom(crop, (rh / crop.shape[0], rw / crop.shape[1], 1.0),
                          order=1)
    warped_fg = nd_zoom(crop_fg.astype(np.float32),
                        (rh / crop.shape[0], rw / crop.shape[1]), order=0) > 0.5
    h, w = render_rgb.shape[:2]
    canvas = np.zeros((h, w, 3), np.float32)
    canvas_fg = np.zeros((h, w), bool)
    canvas[ry0:ry0 + rh, rx0:rx0 + rw] = warped_crop[:rh, :rw]
    canvas_fg[ry0:ry0 + rh, rx0:rx0 + rw] = warped_fg[:rh, :rw]

    # NCC-refined similarity residual at a small solve scale. The
    # luminance weights and the area-average downsampling MUST match the
    # identity gate's construction (BT.601 gray + INTER_AREA): a
    # mean-channel bilinear variant measurably selects a different
    # optimum (NCC 0.503 vs 0.905 under the gate's own metric) and the
    # stamped content then banks nothing.
    small = 224

    def to_gray(rgb: Any) -> Any:
        arr = np.asarray(rgb, np.float32)
        return (0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1]
                + 0.114 * arr[:, :, 2])

    def area_down(field: Any) -> Any:
        if h % small == 0 and w % small == 0:
            fy, fx = h // small, w // small
            return field.reshape(small, fy, small, fx).mean(axis=(1, 3))
        return nd_zoom(field, (small / h, small / w), order=1)

    g_r = area_down(to_gray(render_rgb))
    g_p = area_down(to_gray(canvas))
    m_r = nd_zoom(render_fg.astype(np.float32), (small / h, small / w),
                  order=0) > 0.5
    m_p = nd_zoom(canvas_fg.astype(np.float32), (small / h, small / w),
                  order=0) > 0.5
    yy, xx = np.meshgrid(np.arange(small, dtype=np.float32),
                         np.arange(small, dtype=np.float32), indexing="ij")
    c = (small - 1) / 2.0

    def ncc(scale: float, dx: float, dy: float) -> float:
        sx = (xx - c) / scale + c - dx * small / scale
        sy = (yy - c) / scale + c - dy * small / scale
        wp = _bilinear(g_p, sx, sy)
        wm = _bilinear(m_p.astype(np.float32), sx, sy) > 0.5
        both = wm & m_r
        if both.sum() < 400:
            return -1.0
        a = wp[both] - wp[both].mean()
        b = g_r[both] - g_r[both].mean()
        denominator = float(np.sqrt((a * a).sum() * (b * b).sum()))
        return float((a * b).sum() / denominator) if denominator > 1e-6 else -1.0

    best = (1.0, 0.0, 0.0)
    best_score = ncc(*best)
    for scale in (0.90, 0.95, 1.0, 1.05, 1.10):
        for dx in np.linspace(-0.10, 0.10, 9):
            for dy in np.linspace(-0.12, 0.12, 9):
                score = ncc(scale, float(dx), float(dy))
                if score > best_score:
                    best_score, best = score, (scale, float(dx), float(dy))
    cs, cdx, cdy = best
    for scale in (cs - 0.025, cs, cs + 0.025):
        for dx in np.linspace(cdx - 0.02, cdx + 0.02, 5):
            for dy in np.linspace(cdy - 0.025, cdy + 0.025, 5):
                score = ncc(scale, float(dx), float(dy))
                if score > best_score:
                    best_score, best = score, (scale, float(dx), float(dy))

    scale, dx, dy = best
    yy_f, xx_f = np.meshgrid(np.arange(h, dtype=np.float32),
                             np.arange(w, dtype=np.float32), indexing="ij")
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    sx = (xx_f - cx) / scale + cx - dx * w / scale
    sy = (yy_f - cy) / scale + cy - dy * h / scale
    warped = _bilinear(canvas, sx, sy)
    warped_mask = _bilinear(canvas_fg.astype(np.float32), sx, sy) > 0.5
    return warped, warped_mask & render_fg, (*best, best_score)


def _build_gate_photo_evidence(
    mesh: Any,
    atlas: Any,
    colors_rgba: Any,
    positions_texture: Any,
    source_image: Any,
    azimuth_deg: float,
    elevation_deg: float,
    size: int = 896,
) -> Optional[Dict[str, Any]]:
    """Per-texel photo evidence under the identity correspondence: render
    the current texture at the declared source pose, register the source
    photo to it (the gate's construction), z-buffer first-surface
    visibility, bilinear-sample the registered photo at each visible
    texel's screen position."""
    from scipy.ndimage import minimum_filter

    render_rgb = _render_with_colors(mesh, atlas, colors_rgba, [azimuth_deg],
                                     elevation_deg, size)[0]
    render_fg = _render_foreground(render_rgb)
    photo_array = np.asarray(source_image) if not hasattr(source_image, "mode") \
        else np.asarray(source_image if source_image.mode in ("RGB", "RGBA")
                        else source_image.convert("RGB"))
    warped, warped_mask, residual = _register_photo_to_render(
        photo_array, render_rgb, render_fg)
    if warped is None:
        return None
    # FAIL-CLOSED on a degenerate correspondence: repairing with
    # misregistered photo evidence stamps displaced content (measured on
    # the car: NCC pegged at the scale search boundary 1.125 and the
    # repair pasted a miniature of the whole photo onto the hood). The
    # gate's own alignment on the certified face proof scores NCC ~0.9;
    # below 0.55 the "evidence" is not the surface it claims to be, and
    # NO repair is strictly better than a confident wrong one.
    ncc_score = float(residual[3]) if len(residual) > 3 else 1.0
    scale_pegged = abs(float(residual[0]) - 1.0) >= 0.124
    if ncc_score < 0.55 or scale_pegged:
        return None

    positions = np.asarray(positions_texture, dtype=np.float32)
    surface = positions[:, :, 3] > 0.0
    points = positions[:, :, :3]
    camera = _renderer_camera(mesh, azimuth_deg, elevation_deg, size)
    px, py, depth = camera["project"](points.reshape(-1, 3))
    px = px.reshape(surface.shape)
    py = py.reshape(surface.shape)
    depth = depth.reshape(surface.shape)

    ix = np.clip(np.round(px).astype(np.int32), 0, size - 1)
    iy = np.clip(np.round(py).astype(np.int32), 0, size - 1)
    zbuffer = np.full((size, size), np.inf, np.float32)
    np.minimum.at(zbuffer, (iy[surface], ix[surface]), depth[surface])
    zbuffer = minimum_filter(zbuffer, size=3)
    spread = float(depth[surface].max() - depth[surface].min()) if surface.any() else 1.0
    visible = surface & (depth <= zbuffer[iy, ix] + 0.01 * max(spread, 1e-6))

    sampled = _bilinear(warped.astype(np.float32) / 255.0
                        if warped.max() > 1.5 else warped.astype(np.float32),
                        px, py)
    ok = visible & warped_mask[iy, ix]
    colors = np.asarray(colors_rgba, dtype=np.float32)[:, :, :3]
    residual_map = np.where(ok, np.abs(colors - sampled).mean(axis=2), 0.0)
    return {
        "gate_rgb": sampled.astype(np.float32),
        "gate_ok": ok,
        "residual": residual_map.astype(np.float32),
        "registration": residual,
    }


# --------------------------------------------------------------------------
# feature complexes + fringe deposits
# --------------------------------------------------------------------------

def _cluster_core_texels_world(points: Any, core: Any, link_world: float) -> Any:
    """WORLD-SPACE voxel-graph clustering of core texels (the rescue
    detector's construction): occupied voxels at cell = `link_world` are
    the graph nodes, 26-neighborhood adjacency the edges, and each texel
    inherits its voxel's connected component.

    Replaces atlas morphology (dilate + label), whose linking reach is a
    TEXEL count: one physical feature atlased as several UV charts (or
    sampled at 2048 where the same world gap spans twice the texels)
    fragments into sub-floor pieces before any world merge can see them —
    measured on the face proof: the 2048 mouth complex formed at r 0.045
    vs the 1024 run's 0.11, leaving the lip-edge dash half-covered, while
    resolution-scaled dilation iterations still could not cross the UV
    cut behind the dash. Voxel linking is resolution-independent and
    chart-blind by construction.

    Returns an int32 atlas map of cluster ids (-1 off-core).
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    labels_map = np.full(core.shape, -1, np.int32)
    rows, cols = np.nonzero(core)
    if len(rows) == 0:
        return labels_map
    p = points[rows, cols].astype(np.float64)
    cell = max(float(link_world), 1e-9)
    keys = np.floor(p / cell).astype(np.int64)
    # pack the voxel coordinate into one int64 (meshes are recentered
    # near the origin; 2^20 cells per axis far exceeds any atlas extent)
    H = np.int64(1) << 20
    packed = ((keys[:, 0] + H) << 42) + ((keys[:, 1] + H) << 21) + (keys[:, 2] + H)
    unique_keys, voxel_of_texel = np.unique(packed, return_inverse=True)
    n_voxels = len(unique_keys)
    # adjacency between occupied voxels across the 26-neighborhood: probe
    # each occupied voxel's 13 forward offsets against the occupied set
    key_order = np.argsort(unique_keys)
    sorted_keys = unique_keys[key_order]
    offsets = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if (dx, dy, dz) > (0, 0, 0):
                    offsets.append((dx << 42) + (dy << 21) + dz)
    edges_a: List[Any] = []
    edges_b: List[Any] = []
    for off in offsets:
        probe = unique_keys + off
        pos = np.searchsorted(sorted_keys, probe)
        pos_ok = pos < n_voxels
        hit = np.zeros(n_voxels, bool)
        hit[pos_ok] = sorted_keys[pos[pos_ok]] == probe[pos_ok]
        if hit.any():
            edges_a.append(np.nonzero(hit)[0])
            edges_b.append(key_order[pos[hit]])
    if edges_a:
        ea = np.concatenate(edges_a)
        eb = np.concatenate(edges_b)
        graph = coo_matrix((np.ones(len(ea)), (ea, eb)),
                           shape=(n_voxels, n_voxels))
    else:
        graph = coo_matrix((n_voxels, n_voxels))
    _, voxel_cluster = connected_components(graph, directed=False)
    labels_map[rows, cols] = voxel_cluster[voxel_of_texel].astype(np.int32)
    return labels_map


def _feature_complexes(points: Any, surface: Any, direct: Any, winner_weight: Any,
                       blob_response: Any, rescue_discs: Sequence[Dict[str, Any]],
                       r_ctx: float, scale: float,
                       max_radius_ratio: float = 0.08,
                       link_ratio: float = 0.006) -> List[Dict[str, Any]]:
    """Confident strong-contrast dark cores clustered at feature scale +
    the bake's rescue discs. Dark-evidence-weighted centroids (the anchor
    discipline every placement decision in this lane uses). All texel
    floors are resolution-scaled: fixed counts silently shrink the
    world-space scope at higher atlas resolutions.

    Clustering runs in WORLD SPACE (`_cluster_core_texels_world`, linking
    cell `link_ratio` * mesh diagonal ~ the 1024 atlas morphology's
    measured world reach), so complex formation is identical across
    texture resolutions and UV layouts."""
    area_scale = (surface.shape[0] / 1024.0) ** 2
    core = direct & (winner_weight >= 0.35) & (blob_response <= -0.12)
    cluster_map = _cluster_core_texels_world(points, core,
                                             float(link_ratio) * scale)
    complexes: List[Dict[str, Any]] = []
    for index in np.unique(cluster_map[cluster_map >= 0]):
        mask = cluster_map == index
        size = int(mask.sum())
        if size < max(int(24 * area_scale), 8):
            continue
        p = points[mask]
        weight = winner_weight[mask] * np.clip(-blob_response[mask], 0.0, None)
        centroid = (p * weight[:, None]).sum(axis=0) / max(float(weight.sum()), 1e-9)
        radius = float(np.percentile(np.linalg.norm(p - centroid[None, :], axis=1), 95))
        complexes.append({"size": size, "centroid": centroid, "radius": radius,
                          "kind": "core", "mask": mask})
    merged = True
    while merged:
        merged = False
        for i in range(len(complexes)):
            for j in range(i + 1, len(complexes)):
                distance = float(np.linalg.norm(
                    complexes[i]["centroid"] - complexes[j]["centroid"]))
                if distance < 0.8 * (complexes[i]["radius"] + complexes[j]["radius"]) + r_ctx:
                    mask = complexes[i]["mask"] | complexes[j]["mask"]
                    p = points[mask]
                    weight = winner_weight[mask] * np.clip(-blob_response[mask], 0.0, None)
                    centroid = (p * weight[:, None]).sum(axis=0) / max(
                        float(weight.sum()), 1e-9)
                    radius = float(np.percentile(
                        np.linalg.norm(p - centroid[None, :], axis=1), 95))
                    complexes[i] = {"size": int(mask.sum()), "centroid": centroid,
                                    "radius": radius, "kind": "core", "mask": mask}
                    complexes.pop(j)
                    merged = True
                    break
            if merged:
                break
    # feature-scale cap: larger clusters are material masses (hair
    # frontier), which belong to the band lane
    complexes = [c for c in complexes if c["radius"] <= max_radius_ratio * scale]
    for c in complexes:
        if abs(float(c["centroid"][1])) < c["radius"]:
            # plane-straddling features (mouth) are bilateral: symmetrize
            # the ball so neither corner sits at the boundary (a cut
            # feature corner measurably reads as a doubled feature)
            center_sym = c["centroid"].copy()
            center_sym[1] = 0.0
            p = points[c["mask"]]
            c["radius"] = max(c["radius"], float(np.percentile(
                np.linalg.norm(p - center_sym[None, :], axis=1), 95)))
            c["centroid"] = center_sym
        del c["mask"]
    for disc in rescue_discs:
        complexes.append({
            "size": int(disc.get("core_texels", 0)),
            "centroid": np.asarray(disc["center"], np.float32),
            "radius": float(disc["radius"]),
            "kind": "rescue_disc",
            "disc": disc,
        })
    return complexes


def _fringe_deposits(points: Any, surface: Any, direct: Any, winner_weight: Any,
                     gate_ok: Any, residual: Any, hair_component: Any,
                     rescue_footprint: Any, complexes: Sequence[Dict[str, Any]],
                     r_ctx: float, bright_context_lum: Any,
                     bright_median: float, residual_min: float = 0.085
                     ) -> List[Dict[str, Any]]:
    """Gate-contradicted deposits at trace witness inside complex halos:
    connected blobs whose content the registered photo contradicts.
    Confident-witness blobs outside rescue footprints are NEVER deposits
    (never-demote); hair-frontier slivers and dark-material context
    belong to the band lane (photo-hair-vs-bake-skin conflicts at the
    jaw measurably stamped dark hair smudges onto chin skin — debris
    0.0024->0.0056 at az-22.5)."""
    from scipy.ndimage import label as cc_label

    halo = np.zeros(surface.shape, bool)
    complex_index = np.full(surface.shape, -1, np.int32)
    for k in sorted(range(len(complexes)), key=lambda i: complexes[i]["radius"]):
        c = complexes[k]
        distance = np.linalg.norm(points - c["centroid"][None, None, :], axis=2)
        inside = surface & (distance < c["radius"] + 2.2 * r_ctx)
        halo |= inside
        complex_index[inside & (complex_index < 0)] = k

    contradiction = gate_ok & (residual >= float(residual_min))
    candidates = direct & (winner_weight <= 0.30) & contradiction
    candidates |= rescue_footprint & surface & contradiction
    candidates &= halo
    candidates &= bright_context_lum > 0.55 * float(bright_median)

    area_scale = (surface.shape[0] / 1024.0) ** 2
    blobs, count = cc_label(candidates, structure=np.ones((3, 3), bool))
    deposits: List[Dict[str, Any]] = []
    for index in range(1, count + 1):
        mask = blobs == index
        area = int(mask.sum())
        if area < max(int(4 * area_scale), 4):
            continue
        weights = winner_weight[mask]
        w50 = float(np.percentile(weights, 50))
        w90 = float(np.percentile(weights, 90))
        in_rescue = float((mask & rescue_footprint).sum()) / area
        if in_rescue < 0.5 and (w50 > 0.30 or w90 > 0.40):
            continue
        if float((mask & hair_component).sum()) / area > 0.35:
            continue
        k = int(np.bincount(complex_index[mask][complex_index[mask] >= 0]).argmax()) \
            if (complex_index[mask] >= 0).any() else -1
        if k < 0:
            continue
        deposits.append({
            "mask": mask, "area": area, "complex": k,
            "center": points[mask].mean(axis=0),
            "w50": w50, "w90": w90, "in_rescue": in_rescue,
            "residual": float(residual[mask].mean()),
        })
    deposits.sort(key=lambda d: -d["area"])
    return deposits


# --------------------------------------------------------------------------
# structure vetoes
# --------------------------------------------------------------------------

def _compact_dark_blobs_px(rgb: Any, fg: Any, radius_band_px: Tuple[float, float],
                           size_ref: int = 896) -> Tuple[List[Dict[str, Any]], float]:
    """Compact bright-ringed dark blobs + isolated dark micro-island
    fraction in a render — the doubling/debris signal the vetoes compare
    pre/post. Same semantic construction as the texel check: dark = below
    0.55x the bright-half median, largest dark component (the hair mass)
    excluded. The micro-island fraction counts SUB-FEATURE-size islands
    only (>= the resolution-scaled speck floor, < the anatomical feature
    floor of 0.0009x the foreground bbox): the first version summed every
    ring-bright dark component including features, and the noisier
    baseline masked a real 0.0024->0.0056 debris regression."""
    from scipy.ndimage import binary_dilation
    from scipy.ndimage import label as cc_label

    lum = np.asarray(rgb, np.float32).mean(axis=2)
    fg_lum = lum[fg]
    if fg_lum.size < 256:
        return [], 0.0
    size = rgb.shape[0]
    fg_rows, fg_cols = np.nonzero(fg)
    bbox_area = float((fg_rows.max() - fg_rows.min() + 1)
                      * (fg_cols.max() - fg_cols.min() + 1))
    feature_floor = 0.0009 * bbox_area
    speck_floor = max(int(24 * (size / float(size_ref)) ** 2), 8)
    bright_median = float(np.median(fg_lum[fg_lum >= np.median(fg_lum)]))
    dark = fg & (lum < 0.55 * bright_median)
    labels, count = cc_label(dark, structure=np.ones((3, 3), bool))
    sizes = np.bincount(labels.ravel())
    if len(sizes):
        sizes[0] = 0
    hair_id = int(sizes.argmax()) if sizes.size > 1 else -1
    blobs: List[Dict[str, Any]] = []
    micro_px = 0
    for index in range(1, count + 1):
        if index == hair_id:
            continue
        mask = labels == index
        area = int(sizes[index])
        if area < min(speck_floor, 12):
            continue
        rows, cols = np.nonzero(mask)
        height = rows.max() - rows.min() + 1
        width = cols.max() - cols.min() + 1
        radius = 0.5 * float(max(height, width))
        ring = binary_dilation(mask, iterations=3) & ~mask & fg
        bright_share = float((lum[ring] > 0.55 * bright_median).mean()) \
            if ring.any() else 0.0
        if bright_share >= 0.45 and speck_floor <= area < feature_floor:
            micro_px += area
        aspect = max(height, width) / max(min(height, width), 1)
        solidity = area / max(int(height) * int(width), 1)
        if area < 12 or not (radius_band_px[0] <= radius <= radius_band_px[1]):
            continue
        if aspect > 3.4 or solidity < 0.30 or bright_share < 0.5:
            continue
        blobs.append({"center": np.array([cols.mean(), rows.mean()]),
                      "radius": radius, "area": area, "mask": mask})
    return blobs, micro_px / max(int(fg.sum()), 1)


def _blob_matched(blob: Dict[str, Any], pool: Sequence[Dict[str, Any]]) -> bool:
    for other in pool:
        distance = float(np.linalg.norm(blob["center"] - other["center"]))
        if distance <= 1.5 * max(blob["radius"], other["radius"]) and \
                0.25 <= other["area"] / max(blob["area"], 1) <= 4.0:
            return True
    return False


def _texel_structure_veto(points: Any, normals: Any, surface: Any,
                          colors_before: Any, colors_after: Any, patch: Any,
                          bright_median: float, r_ctx: float, scale: float,
                          confident_mask: Optional[Any] = None
                          ) -> Optional[str]:
    """Fast texel-space pre-filter of the structure rule, under the
    renderer's own shading model at the near-frontal poses (shading dims
    tilted surface, which is what fragments an elongated dark feature
    into the compact class a shading-blind check misses).

    The LOSS rule protects only blobs carrying CONFIDENT content (>= 30%
    confident texels): a vanishing trace-content blob is a repaired
    defect — protecting it would veto the repair's whole purpose
    (measured: the mouth trace stamp cleaned a 30px chip fragment at the
    -90 profile and an unconditional loss rule refused the stamp)."""
    from scipy.ndimage import binary_dilation
    from scipy.ndimage import label as cc_label

    region = binary_dilation(patch, iterations=4) & surface
    lum_before = np.asarray(colors_before, np.float32)[:, :, :3].mean(axis=2)
    lum_after = np.asarray(colors_after, np.float32)[:, :, :3].mean(axis=2)
    n = np.asarray(normals, np.float32)[:, :, :3]
    n = n / np.maximum(np.linalg.norm(n, axis=2, keepdims=True), 1e-8)

    # area floors are RESOLUTION-SCALED: a fixed texel count silently
    # quarters the world-space floor at 2048 (measured: an 86-texel
    # "lost blob" veto at 2048 = a 21-texel speck in 1024 terms, and the
    # over-firing ladder dropped the banked delta from +0.045 to +0.010)
    resolution_scale = (surface.shape[0] / 1024.0) ** 2
    area_floor = max(int(48 * resolution_scale), 12)

    def texel_blobs(lum_field: Any, visible: Any) -> List[Dict[str, Any]]:
        dark = visible & (lum_field < 0.55 * bright_median)
        labels, count = cc_label(dark, structure=np.ones((3, 3), bool))
        out = []
        for index in range(1, count + 1):
            mask = labels == index
            area = int(mask.sum())
            if area < area_floor:
                continue
            p = points[mask]
            center = p.mean(axis=0)
            radius = float(np.percentile(
                np.linalg.norm(p - center[None, :], axis=1), 90))
            if not (0.25 * r_ctx <= radius <= 2.2 * r_ctx):
                continue
            rows, cols = np.nonzero(mask)
            height = rows.max() - rows.min() + 1
            width = cols.max() - cols.min() + 1
            if max(height, width) / max(min(height, width), 1) > 3.4:
                continue
            if area / max(int(height) * int(width), 1) < 0.30:
                continue
            ring = binary_dilation(mask, iterations=3) & ~mask & surface
            if not ring.any() or float(
                    (lum_field[ring] > 0.55 * bright_median).mean()) < 0.5:
                continue
            confident_share = 0.0
            if confident_mask is not None:
                confident_share = float(confident_mask[mask].mean())
            out.append({"center": center, "radius": radius, "area": area,
                        "confident": confident_share})
        return out

    for az, el in ((0.0, 0.0), (22.5, 0.0), (-22.5, 0.0),
                   (0.0, 10.0), (22.5, 10.0), (-22.5, 10.0)):
        azr, elr = math.radians(az), math.radians(el)
        eye = np.array([math.cos(elr) * math.cos(azr),
                        math.cos(elr) * math.sin(azr),
                        math.sin(elr)], np.float32)
        shade = 0.88 + 0.12 * np.clip(n @ eye, 0.0, 1.0)
        visible = region & ((n @ eye) > 0.05)
        pre = texel_blobs(lum_before * shade, visible)
        post = texel_blobs(lum_after * shade, visible)
        for blob in post:
            if not _blob_matched(blob, pre):
                return (f"new_dark_blob a{blob['area']} az{az:+.1f}el{el:.0f}")
        for blob in pre:
            if blob["confident"] >= 0.30 and not _blob_matched(blob, post):
                return f"lost_dark_blob a{blob['area']} az{az:+.1f}el{el:.0f}"
    return None


def _micro_island_components(rgb: Any, fg: Any) -> List[Dict[str, Any]]:
    """The micro-island population of `_compact_dark_blobs_px`'s fraction
    budget, as individual components (for the photo-truth exemption)."""
    from scipy.ndimage import binary_dilation
    from scipy.ndimage import label as cc_label

    lum = np.asarray(rgb, np.float32).mean(axis=2)
    fg_lum = lum[fg]
    if fg_lum.size < 256:
        return []
    size = rgb.shape[0]
    rows, cols = np.nonzero(fg)
    bbox_area = float((rows.max() - rows.min() + 1)
                      * (cols.max() - cols.min() + 1))
    feature_floor = 0.0009 * bbox_area
    speck_floor = max(int(24 * (size / 896.0) ** 2), 8)
    bright_median = float(np.median(fg_lum[fg_lum >= np.median(fg_lum)]))
    dark = fg & (lum < 0.55 * bright_median)
    labels, count = cc_label(dark, structure=np.ones((3, 3), bool))
    sizes = np.bincount(labels.ravel())
    if len(sizes):
        sizes[0] = 0
    hair_id = int(sizes.argmax()) if sizes.size > 1 else -1
    out: List[Dict[str, Any]] = []
    for index in range(1, count + 1):
        if index == hair_id:
            continue
        area = int(sizes[index])
        if not (speck_floor <= area < feature_floor):
            continue
        mask = labels == index
        ring = binary_dilation(mask, iterations=3) & ~mask & fg
        if not ring.any() or float((lum[ring] > 0.55 * bright_median).mean()) < 0.45:
            continue
        ys, xs = np.nonzero(mask)
        out.append({"center": np.array([xs.mean(), ys.mean()]),
                    "radius": 0.5 * float(max(ys.max() - ys.min() + 1,
                                              xs.max() - xs.min() + 1)),
                    "area": area, "mask": mask})
    return out


def _first_surface_projection(mesh: Any, surface_points: Any, az: float,
                              el: float, size: int) -> Tuple[Any, Any, Any]:
    """(px, py, visible) of surface texels under the renderer camera with
    a splat z-buffer (the px2texel construction used across this lane)."""
    from scipy.ndimage import minimum_filter

    camera = _renderer_camera(mesh, az, el, size)
    px, py, depth = camera["project"](surface_points)
    ix = np.clip(np.round(px).astype(np.int32), 0, size - 1)
    iy = np.clip(np.round(py).astype(np.int32), 0, size - 1)
    zbuf = np.full((size, size), np.inf, np.float32)
    np.minimum.at(zbuf, (iy, ix), depth)
    zbuf = minimum_filter(zbuf, size=3)
    depth_spread = float(depth.max() - depth.min()) or 1.0
    visible = depth <= zbuf[iy, ix] + 0.01 * depth_spread
    return px, py, visible


def _feature_blob_footprint(mesh: Any, pre_renders: Any, surface_points: Any,
                            size: int = 896) -> Any:
    """Per-surface-texel mask of feature anatomy: texels rendering inside
    a FEATURE-CLASS compact dark blob's own pixels
    (`_compact_dark_blobs_px`, the doubling detectors' geometry) at any
    battery pose of the pre-repair baseline. Atlas geometry cannot make
    this distinction (measured: the profile eye's under-lash mass,
    r 0.65 * r_ctx, vs a brow tail that reads as pure debris,
    r 0.74 * r_ctx — inseparable by world radius), but the render
    battery already knows where the features are. Pixel-exact on
    purpose: a bounding-box footprint shadows debris NEAR features (the
    brow tail beside the eye stayed unliftable and two battery views
    stayed over the debris gate)."""
    from scipy.ndimage import binary_dilation

    protected = np.zeros(len(surface_points), bool)
    for (el, azimuths), base in pre_renders:
        for az, base_entry in zip(azimuths, base):
            blobs = [b for b in (base_entry.get("blobs") or [])
                     if b.get("mask") is not None]
            if not blobs:
                continue
            px, py, visible = _first_surface_projection(
                mesh, surface_points, az, el, size)
            ix = np.clip(np.round(px).astype(np.int32), 0, size - 1)
            iy = np.clip(np.round(py).astype(np.int32), 0, size - 1)
            union = np.zeros((size, size), bool)
            for blob in blobs:
                union |= blob["mask"]
            union = binary_dilation(union, iterations=2)
            protected |= visible & union[iy, ix]
    return protected


def _render_structure_veto(mesh: Any, atlas: Any, base_renders: Any,
                           colors_after: Any, r_feat_world: float,
                           size: int = 896,
                           photo_truth_mask: Optional[Any] = None,
                           original_renders: Optional[Any] = None
                           ) -> Tuple[Optional[str], Optional[List[Any]]]:
    """Authoritative render-space structure rule, relative to the
    pre-repair render at the same views, mirroring the doubling/debris
    detector split: no NEW and no LOST compact dark blob at
    ANATOMICAL-FEATURE size (>= 0.0009x the foreground bbox, the eye
    detectors' own floor — a 22 px sub-feature speck must not veto a
    whole-feature repair; it is debris-class and belongs to the fraction
    budget below); and the isolated sub-feature dark micro-island
    fraction may not grow past its baseline + 0.0003 (the knife-edge
    debris margins measured throughout this lane).

    PHOTO-TRUTH EXEMPTION (`photo_truth_mask`, an atlas mask of stamped
    texels whose content the registered photo CONFIRMS at low residual):
    the growth budget exists to catch structure the repair INVENTS —
    displaced copies, misregistered fragments, unmasked debris. A new
    micro island whose pixels render from photo-confirmed stamped texels
    is the photo's own anatomy (measured at 2048: the mouth stamp
    replaced the below-lip smudge with the photo's lip-corner line, and
    the line — real content, 330 px, just under the feature floor — was
    the only 'growth'; the whole-mouth repair was refused for printing
    exactly what the photo prescribes). Feature-size NEW blobs stay
    unconditionally banned (doubling class), as do photo-contradicted
    micro islands. The exemption is evaluated lazily (one extra render
    at the failing view only).

    The exemption is BOUNDED by the battery's own pre-repair worst case:
    photo-true growth is admissible only while the view's absolute micro
    fraction stays at or below the highest pre-repair fraction across
    the battery — the repair never makes any view the new worst
    offender. (Measured: the unbounded exemption shipped the eye
    complex's full re-registration, whose photo-true lash fragments
    pushed two battery views past the absolute debris detectors —
    0.0030/0.0032 vs the 0.003 gate — while those same fragments double
    as the +-90 profile eye's anatomy and cannot be consolidated away;
    the bound refuses that stamp back to trace mode, which banks clean.)

    Returns (veto_reason, after_entries): on pass (None reason) the
    caller MUST replace its baseline with `after_entries` — the accepted
    stamp's own content (e.g. an exempted photo-true line) is otherwise
    counted as growth against every LATER candidate (measured: one
    accepted eye stamp turned the shared stale baseline into a blanket
    0.0039->0.0051 veto for every following complex including the
    mouth).

    CUMULATIVE-BASELINE VETO (`original_renders`, the pre-repair state
    before ANY acceptance — critic 2's cycle-5 hardening, adopted as
    mandatory by the cycle-6 certification): the advancing baseline above
    judges each candidate on its own delta, but it RE-ARMS the +0.0003
    micro budget with every acceptance — n accepted stamps may drift a
    view by n x 0.0003 while every per-stamp check passes (measured:
    ~7 stamps produced +0.00096 at one view, triple the single-stamp
    budget, with the absolute debris detectors at 90-97% utilization).
    The cumulative rule closes the re-arm without touching the
    exemption: a candidate is refused when the view's post-stamp micro
    fraction exceeds BOTH its ORIGINAL pre-repair fraction + 0.0003 AND
    the original battery-wide worst fraction. Growth up to the original
    battery worst stays admissible (the photo-truth exemption's own
    bound — the repair never makes any view the new worst offender of
    the PRE-REPAIR battery); unbounded per-acceptance re-arming is not.
    The exemption bound itself is pinned to the ORIGINAL battery worst
    for the same reason (an advancing bound re-arms the same way)."""
    truth_renders: Dict[Tuple[float, float], Any] = {}
    truth_colors: Optional[Any] = None
    if photo_truth_mask is not None and photo_truth_mask.any():
        truth = photo_truth_mask.astype(np.float32)
        truth_colors = np.concatenate(
            [np.repeat(truth[:, :, None], 3, axis=2),
             np.ones((*truth.shape, 1), np.float32)], axis=2)
    if original_renders is None:
        original_renders = base_renders
    original_micro: Dict[Tuple[float, float], float] = {}
    for (el, azimuths), base in original_renders:
        for az, entry in zip(azimuths, base):
            original_micro[(float(az), float(el))] = float(entry["micro"])
    battery_worst_micro = max(original_micro.values(), default=0.0)
    after_entries: List[Tuple[Any, List[Dict[str, Any]]]] = []
    for (el, azimuths), base in base_renders:
        after = _render_with_colors(mesh, atlas, colors_after, azimuths, el, size)
        entries: List[Dict[str, Any]] = []
        for az, base_entry, after_rgb in zip(azimuths, base, after):
            fg = _render_foreground(after_rgb)
            rows, cols = np.nonzero(fg)
            bbox_area = float((rows.max() - rows.min() + 1)
                              * (cols.max() - cols.min() + 1)) if len(rows) else 1.0
            feature_floor = 0.0009 * bbox_area
            camera = _renderer_camera(mesh, az, el, size)
            r_px = r_feat_world * camera["px_per_world"]
            band = (0.25 * r_px, 2.6 * r_px)
            post_blobs, post_micro = _compact_dark_blobs_px(after_rgb, fg, band)
            pre_blobs, pre_micro = base_entry["blobs"], base_entry["micro"]
            for blob in post_blobs:
                if blob["area"] >= feature_floor and not _blob_matched(
                        blob, pre_blobs):
                    return (f"az{az:+.1f}el{el:.0f} new blob a{blob['area']}",
                            None)
            for blob in pre_blobs:
                if blob["area"] >= feature_floor and not _blob_matched(
                        blob, post_blobs):
                    return (f"az{az:+.1f}el{el:.0f} lost blob a{blob['area']}",
                            None)
            if post_micro > pre_micro + 0.0003:
                exempt_px = 0
                if truth_colors is not None:
                    key = (az, el)
                    if key not in truth_renders:
                        truth_renders[key] = _render_with_colors(
                            mesh, atlas, truth_colors, [az], el, size)[0]
                    confirmed = (np.asarray(truth_renders[key], np.float32)
                                 .mean(axis=2) > 128.0)
                    pre_islands = _micro_island_components(
                        base_entry.get("rgb"), base_entry.get("fg")
                    ) if base_entry.get("rgb") is not None else pre_blobs
                    for island in _micro_island_components(after_rgb, fg):
                        if _blob_matched(island, pre_islands):
                            continue
                        share = float(confirmed[island["mask"]].mean())
                        if share >= 0.6:
                            exempt_px += island["area"]
                adjusted = post_micro - exempt_px / max(int(fg.sum()), 1)
                if adjusted > pre_micro + 0.0003:
                    return (f"az{az:+.1f}el{el:.0f} micro-island growth "
                            f"{pre_micro:.4f}->{post_micro:.4f}", None)
                if exempt_px and post_micro > battery_worst_micro:
                    return (f"az{az:+.1f}el{el:.0f} micro-island growth "
                            f"{pre_micro:.4f}->{post_micro:.4f} "
                            f"(exemption bound {battery_worst_micro:.4f})",
                            None)
            # cumulative-baseline veto (see docstring): judged against the
            # ORIGINAL pre-repair state, so per-acceptance re-arming of the
            # +0.0003 budget cannot accumulate past the battery's own
            # pre-repair worst case.
            origin_micro = original_micro.get((float(az), float(el)), pre_micro)
            if (post_micro > origin_micro + 0.0003
                    and post_micro > battery_worst_micro):
                return (f"az{az:+.1f}el{el:.0f} cumulative micro-island "
                        f"growth {origin_micro:.4f}->{post_micro:.4f} "
                        f"(original battery worst "
                        f"{battery_worst_micro:.4f})", None)
            entries.append({"blobs": post_blobs, "micro": post_micro,
                            "rgb": after_rgb, "fg": fg})
        after_entries.append(((el, azimuths), entries))
    return None, after_entries


def _consolidate_render_specks(mesh: Any, atlas: Any, colors: Any,
                               positions: Any, normals: Any, surface: Any,
                               repaired_total: Any, pre_renders: Any,
                               bright_median: float, r_feat_world: float,
                               size: int = 896) -> Tuple[Any, int]:
    """Final render-informed speck consolidation over the repair's own
    texels (the FACE-20 displaced-refill discipline at micro scale).

    The photo-truth exemption ships stamps whose only 'growth' is the
    photo's own anatomy; most of it is feature-adjacent structure (lip
    lines, lash mass edges) — but its detached pieces (a brow tail, an
    eye-corner tip) read at oblique battery poses as isolated dark
    islands on skin, which the ABSOLUTE debris detectors count
    regardless of provenance (measured: two battery views at
    0.0030/0.0032 vs the 0.003 gate, every flagged island a stamped
    photo-true fragment). This pass renders the final state at the veto
    battery views, finds NEW sub-feature bright-ringed dark islands
    relative to the PRE-REPAIR baseline, and lifts the repaired texels
    rendering into them so they leave the dark class AT THAT VIEW'S OWN
    SHADING (the renderer's fragment model, 0.88 + 0.12 * facing: an
    albedo above the split still renders dark when tilted — the first
    version tested albedo only and lifted nothing). The luminance
    pattern survives at reduced contrast (the displaced-refill floor
    rule), so the content keeps paying the gate what the floor admits
    while no isolated dark speck of the repair's own making survives at
    any battery pose. Feature-scale blobs (the doubling detectors' own
    floor) are untouched; pre-existing islands are untouched."""
    from scipy.ndimage import binary_dilation, minimum_filter
    from scipy.ndimage import label as cc_label

    lifted_total = 0
    colors = np.asarray(colors, np.float32)
    points = positions[:, :, :3]
    normal_vectors = np.asarray(normals, np.float32)[:, :, :3]
    normal_vectors = normal_vectors / np.maximum(
        np.linalg.norm(normal_vectors, axis=2, keepdims=True), 1e-8)
    surface_rows, surface_cols = np.nonzero(surface)
    surface_points = points[surface_rows, surface_cols]
    repaired_flat = repaired_total[surface_rows, surface_cols]
    if not repaired_flat.any():
        return colors, 0

    # FEATURE-BLOB PROTECTION: a lifted texel may belong, at ANOTHER
    # battery pose, to a legitimate feature blob (measured: the az-22.5
    # lift brightened texels forming the az+90 profile eye's upper lash
    # mass — eye_count 1 -> 0). Texels rendering inside a feature-class
    # blob's own pixels at any battery pose of the pre-repair baseline
    # are feature anatomy, never lifted (`_feature_blob_footprint`).
    protected_flat = _feature_blob_footprint(mesh, pre_renders,
                                             surface_points, size)
    for (el, azimuths), base in pre_renders:
        after = _render_with_colors(mesh, atlas, colors, azimuths, el, size)
        for az, base_entry, after_rgb in zip(azimuths, base, after):
            fg = _render_foreground(after_rgb)
            rows, cols = np.nonzero(fg)
            if not len(rows):
                continue
            bbox_area = float((rows.max() - rows.min() + 1)
                              * (cols.max() - cols.min() + 1))
            feature_floor = 0.0009 * bbox_area
            render_lum = np.asarray(after_rgb, np.float32).mean(axis=2)
            fg_lum = render_lum[fg]
            render_bright = float(np.median(
                fg_lum[fg_lum >= np.median(fg_lum)]))
            render_split = 0.55 * render_bright / 255.0
            pre_islands = _micro_island_components(
                base_entry["rgb"], base_entry["fg"])
            new_islands = [
                island for island in _micro_island_components(after_rgb, fg)
                if island["area"] < feature_floor
                and not _blob_matched(island, pre_islands)
            ]
            if not new_islands:
                continue
            # FIRST-SURFACE mapping of the islands back to texels: only
            # texels that actually RENDER as the island's pixels may be
            # lifted. A 2D box test leaks through depth (measured: the
            # box around an eye-corner island selected eye-interior
            # texels on the surface behind it and the profile eye blob
            # vanished — eye_count 0 at az+70/+90).
            px, py, visible = _first_surface_projection(
                mesh, surface_points, az, el, size)
            ix = np.clip(np.round(px).astype(np.int32), 0, size - 1)
            iy = np.clip(np.round(py).astype(np.int32), 0, size - 1)
            island_union = np.zeros((size, size), bool)
            for island in new_islands:
                island_union |= binary_dilation(island["mask"], iterations=1)
            in_island = (island_union[iy, ix] & visible & repaired_flat
                         & ~protected_flat)
            if not in_island.any():
                continue
            sel_rows = surface_rows[in_island]
            sel_cols = surface_cols[in_island]
            level = colors[sel_rows, sel_cols, :3].mean(axis=1)
            eye_axis = _renderer_camera(mesh, az, el, size)["eye"]
            shade = 0.88 + 0.12 * np.clip(
                normal_vectors[sel_rows, sel_cols] @ eye_axis, 0.0, 1.0)
            # target: leave the dark class under THIS view's shading
            target = 1.02 * render_split / np.maximum(shade, 0.5)
            need = level < target
            if not need.any():
                continue
            sel_rows = sel_rows[need]
            sel_cols = sel_cols[need]
            lift = target[need] / np.maximum(level[need], 1e-6)
            colors[sel_rows, sel_cols, :3] = np.clip(
                colors[sel_rows, sel_cols, :3] * lift[:, None], 0.0, 1.0)
            lifted_total += int(need.sum())
    return colors, lifted_total


# --------------------------------------------------------------------------
# repairs (rescue transplant semantics: tone match + feather + whole patch)
# --------------------------------------------------------------------------

def _feathered_write(colors: Any, surface: Any, patch: Any, target_rgb: Any,
                     feather: float) -> Any:
    from scipy.ndimage import binary_erosion, gaussian_filter

    out = np.asarray(colors, np.float32).copy()
    target = out.copy()
    rows, cols = np.nonzero(patch)
    target[rows, cols, :3] = target_rgb
    weight = binary_erosion(patch, np.ones((3, 3), bool), iterations=1
                            ).astype(np.float32)
    # feather width is atlas-resolution-scaled (a fixed texel sigma halves
    # the world-space blend band at 2048)
    weight = gaussian_filter(weight, float(feather) * surface.shape[0] / 1024.0)
    weight = np.clip(weight * 1.2, 0.0, 1.0)
    weight[~patch] = 0.0
    weight[~surface] = 0.0
    out[:, :, :3] = (out[:, :, :3] * (1 - weight[..., None])
                     + target[:, :, :3] * weight[..., None])
    return out


def _gate_stamp(colors: Any, evidence: Dict[str, Any], comp: Dict[str, Any],
                *, points: Any, surface: Any, direct: Any, winner_weight: Any,
                winner_index: Any, source_index: int, rescue_disc_mask: Any,
                bright_context_lum: Any, bright_median: float, r_ctx: float,
                scale: float, keep_confident: bool, feather: float = 2.5
                ) -> Tuple[Any, Any, str]:
    """Whole-complex corrective stamp of the photo's own content under the
    identity correspondence (see module docstring, step 2)."""
    from scipy.spatial import cKDTree

    gate_ok = evidence["gate_ok"]
    gate_rgb = evidence["gate_rgb"]
    residual = evidence["residual"]

    distance = np.linalg.norm(points - comp["centroid"][None, None, :], axis=2)
    patch = surface & (distance < comp["radius"] + 2.2 * r_ctx)
    # never-demote across witnesses: non-source confident content stays
    side_confident = direct & (winner_weight >= 0.35) & (winner_index != source_index)
    patch &= ~side_confident
    if keep_confident:
        patch &= ~(direct & (winner_weight >= 0.35))
    patch &= ~rescue_disc_mask
    patch &= bright_context_lum > 0.55 * bright_median
    # CONTENT-CONFLICT TRIM: dark photo content belongs to the stamp only
    # within the feature's own extent (iris, lash mass, lip slit — the
    # complex's detection evidence). In the OUTER halo band, dark photo
    # content over the bake's bright context is the co-witnessed hairline
    # conflict (photo hair claiming skin under parallax) — band-lane
    # territory; stamping it measurably sprayed dark debris islands
    # across six views (0.0024->0.0056 at az-22.5).
    photo_dark = gate_ok & (gate_rgb.mean(axis=2) < 0.55 * bright_median)
    patch &= ~(photo_dark & (distance > comp["radius"]))
    valid = patch & gate_ok
    if int(valid.sum()) < 64:
        return colors, None, "patch_unsampled"

    source = gate_rgb.copy()
    hole = patch & ~gate_ok
    hole_far = np.zeros(surface.shape, bool)
    if hole.any():
        ok_rows, ok_cols = np.nonzero(valid)
        tree = cKDTree(points[ok_rows, ok_cols])
        hole_rows, hole_cols = np.nonzero(hole)
        k = min(12, len(ok_rows))
        hole_distance, hole_index = tree.query(points[hole_rows, hole_cols],
                                               k=k, workers=-1)
        hole_distance = np.atleast_2d(np.asarray(hole_distance))
        hole_index = np.atleast_2d(np.asarray(hole_index))
        near = hole_distance[:, 0] < 0.012 * scale
        inverse = 1.0 / (hole_distance[near] ** 2 + 1e-10)
        neighbor_colors = gate_rgb[ok_rows[hole_index[near]],
                                   ok_cols[hole_index[near]]]
        source[hole_rows[near], hole_cols[near]] = (
            (inverse[..., None] * neighbor_colors).sum(axis=1)
            / inverse.sum(axis=1, keepdims=True))
        hole_far[hole_rows[~near], hole_cols[~near]] = True
    fill_fraction = 1.0 - float(hole_far.sum()) / max(int(patch.sum()), 1)
    if fill_fraction < 0.6:
        return colors, None, f"patch_coverage {fill_fraction:.2f}"

    # low-frequency stage-tone field taught by CLEAN texels only
    from scipy.ndimage import gaussian_filter

    clean = gate_ok & direct & (residual < 0.08)
    weight_field = np.zeros(surface.shape, np.float32)
    weight_field[clean] = 1.0
    delta_field = np.zeros((*surface.shape, 3), np.float32)
    colors_np = np.asarray(colors, np.float32)
    delta_field[clean] = colors_np[clean, :3] - gate_rgb[clean]
    sigma = max(3.0, surface.shape[0] / 128.0)
    denominator = gaussian_filter(weight_field, sigma)
    tone = np.zeros((*surface.shape, 3), np.float32)
    for channel in range(3):
        numerator = gaussian_filter(delta_field[:, :, channel] * weight_field, sigma)
        tone[:, :, channel] = np.where(
            denominator > 0.03, numerator / np.maximum(denominator, 1e-6), 0.0)

    stamp_core = patch & ~hole_far
    rows, cols = np.nonzero(stamp_core)
    target = np.clip(source[rows, cols] + tone[rows, cols], 0.0, 1.0)
    target_map = np.asarray(colors, np.float32)[:, :, :3].copy()
    target_map[rows, cols] = target
    # IN-STAMP SPECK GUARD: photo content re-sampled at 2048 keeps fine
    # dark micro-detail (lip-line tips, nostril-shadow specks) that the
    # 1024 sampling blurred away; rendered obliquely those specks read as
    # isolated dark debris and the render veto refuses the whole stamp on
    # micro-island growth (measured: the chin/mouth-surround complex that
    # banked at 1024 was refused at 2048 on +0.0010 growth at az-35).
    # The regression source is removed INSIDE the stamp instead of
    # relaxing the veto: bright-ringed, sub-feature-size, isolated dark
    # specks in the stamped content are lifted to just above the dark
    # split (the displaced-refill floor discipline); feature-scale blobs
    # and specks connected to dark masses are feature content and stay.
    from scipy.ndimage import binary_dilation as _dilate
    from scipy.ndimage import label as _cc_label

    area_scale = (surface.shape[0] / 1024.0) ** 2
    speck_max = max(int(round(40 * area_scale)), 8)
    stamp_lum = target_map.mean(axis=2)
    dark_split = 0.55 * float(bright_median)
    stamp_dark = stamp_core & (stamp_lum < dark_split)
    dark_any = surface & (stamp_lum < dark_split)
    speck_labels, speck_count = _cc_label(dark_any,
                                          structure=np.ones((3, 3), bool))
    if speck_count:
        sizes = np.bincount(speck_labels.ravel())
        for index in np.unique(speck_labels[stamp_dark]):
            if index == 0 or sizes[index] > speck_max:
                continue
            blob = speck_labels == index
            if int((blob & stamp_core).sum()) != int(blob.sum()):
                continue  # touches pre-existing content: not stamp-made
            ring = _dilate(blob, iterations=2) & surface & ~blob
            if not ring.any():
                continue
            if float((stamp_lum[ring] >= dark_split).mean()) < 0.6:
                continue  # ring not bright: part of a dark mass/feature
            b_rows, b_cols = np.nonzero(blob)
            level = np.maximum(target_map[b_rows, b_cols].mean(axis=1), 1e-6)
            lift = (1.02 * dark_split) / level
            target_map[b_rows, b_cols] = np.clip(
                target_map[b_rows, b_cols] * lift[:, None], 0.0, 1.0)
    target = target_map[rows, cols]
    out = _feathered_write(colors, surface, stamp_core, target, feather)
    return out, stamp_core, "ok"


def _photo_patch(colors: Any, evidence: Dict[str, Any], deposit: Dict[str, Any],
                 *, points: Any, surface: Any, direct: Any,
                 bright_median: float, residual_clean: float = 0.045,
                 margin: int = 2, feather: float = 1.5) -> Tuple[Any, Any, str]:
    """Deposit-scale whole-patch stamp of the photo's local content. The
    SOURCE must be bright-material content: a photo-hair-vs-bake-skin
    conflict at the jaw silhouette is a band-lane tone problem, and
    stamping the photo's dark hair onto skin measurably sprayed debris
    islands across six views."""
    from scipy.ndimage import binary_dilation

    gate_ok = evidence["gate_ok"]
    gate_rgb = evidence["gate_rgb"]
    residual = evidence["residual"]
    patch = binary_dilation(deposit["mask"], iterations=margin) & surface
    cover = float((gate_ok & patch).sum()) / max(int(patch.sum()), 1)
    if cover < 0.75:
        return colors, None, f"gate_cover {cover:.2f}"
    source_lum = float(np.median(gate_rgb[patch & gate_ok].mean(axis=1)))
    if source_lum < 0.55 * float(bright_median):
        return colors, None, f"dark_source {source_lum:.2f}"
    ring = binary_dilation(patch, iterations=margin + 4) & ~patch & direct \
        & (residual < residual_clean) & gate_ok
    tone = np.zeros(3, np.float32)
    colors_np = np.asarray(colors, np.float32)
    if int(ring.sum()) >= 12:
        tone = colors_np[ring, :3].mean(axis=0) - gate_rgb[ring].mean(axis=0)
    rows, cols = np.nonzero(patch)
    target = np.clip(gate_rgb[rows, cols] + tone[None, :], 0.0, 1.0)
    out = _feathered_write(colors, surface, patch, target, feather)
    return out, patch, "ok"


def _rescue_recopy(colors: Any, deposit: Dict[str, Any], disc: Dict[str, Any],
                   *, points: Any, surface: Any, confident_mask: Any,
                   deviation: Any, scale: float, margin: int = 2,
                   feather: float = 1.5) -> Tuple[Any, Any, str]:
    """Re-copy a rescue-fringe patch through the disc's own anchored
    correspondence (mirror + placement_shift) from confident witnesses on
    the CURRENT colors, so upstream root repairs propagate."""
    from scipy.ndimage import binary_dilation
    from scipy.spatial import cKDTree

    conf_rows, conf_cols = np.nonzero(confident_mask)
    if len(conf_rows) < 64:
        return colors, None, "no_confident_sources"
    tree = cKDTree(points[conf_rows, conf_cols])
    shift = np.asarray(disc.get("placement_shift", [0.0, 0.0, 0.0]), np.float32)
    patch = binary_dilation(deposit["mask"], iterations=margin) & surface
    rows, cols = np.nonzero(patch)
    query = points[rows, cols] - shift[None, :]
    query[:, 1] *= -1.0
    distance, index = tree.query(query, k=1, workers=-1)
    if float(np.median(distance)) > 0.01 * scale:
        return colors, None, "twin_unsampled"
    colors_np = np.asarray(colors, np.float32)
    source = colors_np[conf_rows[index], conf_cols[index], :3]
    ring = binary_dilation(patch, iterations=margin + 4) & ~patch & surface \
        & (deviation < 0.045) & ~deposit["mask"]
    tone = np.zeros(3, np.float32)
    if int(ring.sum()) >= 12:
        ring_query = points[ring] - shift[None, :]
        ring_query[:, 1] *= -1.0
        ring_distance, ring_index = tree.query(ring_query, k=1, workers=-1)
        keep = np.asarray(ring_distance) < 0.01 * scale
        if keep.sum() >= 8:
            tone = (colors_np[ring][:, :3][keep].mean(axis=0)
                    - colors_np[conf_rows[ring_index[keep]],
                                conf_cols[ring_index[keep]], :3].mean(axis=0))
    target = np.clip(source + tone[None, :], 0.0, 1.0)
    out = _feathered_write(colors, surface, patch, target, feather)
    return out, patch, "ok"


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

def repair_feature_fringes(
    mesh: Any,
    *,
    atlas: Any,
    colors_rgba: Any,
    positions_texture: Any,
    normals_texture: Any,
    projections: Sequence[Any],
    source_image: Any,
    source_azimuth_deg: float,
    source_elevation_deg: float,
    source_index: int = 0,
    rescue_discs: Sequence[Dict[str, Any]] = (),
    observed_mask: Optional[Any] = None,
    stats_out: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, Optional[Any]]:
    """Run the feature-fringe repair lane (see module docstring).

    Returns (colors, repaired_mask); repaired_mask is None when the stage
    is a structural no-op (single view, no evidence, no complexes). The
    caller marks repaired texels as completion (excluded from fill
    detail/floor statistics) exactly like rescue-disc texels.
    """
    stats: Dict[str, Any] = {"applied": False, "stamps": [], "patches": 0,
                             "recopies": 0, "disc_refreshed": False}
    if stats_out is not None:
        stats_out.update(stats)
    if len(projections) < 2 or source_image is None:
        return colors_rgba, None

    positions = np.asarray(positions_texture, dtype=np.float32)
    surface = positions[:, :, 3] > 0.0
    points = positions[:, :, :3]
    colors = np.asarray(colors_rgba, dtype=np.float32).copy()

    weight_stack = np.stack(
        [np.asarray(p["weight"], dtype=np.float32) for p in projections], axis=0)
    winner_weight = weight_stack.max(axis=0)
    winner_index = weight_stack.argmax(axis=0)
    direct = surface & (winner_weight > 1e-6)
    if observed_mask is not None:
        direct &= np.asarray(observed_mask, dtype=bool)
    if int(direct.sum()) < 1024:
        return colors_rgba, None

    direct_points = points[direct]
    scale = float(np.linalg.norm(direct_points.max(axis=0)
                                 - direct_points.min(axis=0)))
    if scale <= 0.0:
        return colors_rgba, None
    r_ctx = 0.02 * scale

    luminance = colors[:, :, :3].mean(axis=2)
    direct_lum = luminance[direct]
    bright_median = float(np.median(direct_lum[direct_lum >= np.median(direct_lum)]))
    if bright_median <= 0.0:
        return colors_rgba, None

    from scipy.ndimage import gaussian_filter
    from scipy.ndimage import label as cc_label

    from .texturing import _voxel_ball_stats, mirror_rescue_disc

    _, ball_mean, _ = _voxel_ball_stats(direct_points, direct_lum, r_ctx,
                                        direct_points)
    blob_response = np.zeros(surface.shape, np.float32)
    blob_response[direct] = direct_lum - ball_mean
    ball_rgb = np.zeros((*surface.shape, 3), np.float32)
    for channel in range(3):
        _, channel_mean, _ = _voxel_ball_stats(
            direct_points, colors[:, :, channel][direct], r_ctx, direct_points)
        ball_rgb[direct, channel] = channel_mean
    deviation = np.abs(colors[:, :, :3] - ball_rgb).mean(axis=2)
    deviation[~direct] = 0.0

    sigma_bright = max(2.0, surface.shape[0] / 256.0)
    surface_weight = gaussian_filter(surface.astype(np.float32), sigma_bright)
    surface_lum = gaussian_filter(np.where(surface, luminance, 0.0), sigma_bright)
    bright_context_lum = np.where(surface_weight > 0.05,
                                  surface_lum / np.maximum(surface_weight, 1e-6),
                                  0.0)

    rescue_footprint = np.zeros(surface.shape, bool)
    rescue_disc_mask = np.zeros(surface.shape, bool)
    for disc in rescue_discs:
        center = np.asarray(disc["center"], np.float32)
        distance = np.linalg.norm(points - center[None, None, :], axis=2)
        same_side = ((points[:, :, 1] * center[1]) >= 0.0
                     if abs(float(center[1])) > 1e-6
                     else np.ones(surface.shape, bool))
        rescue_footprint |= surface & same_side & (distance < 1.3 * float(disc["radius"]))
        rescue_disc_mask |= surface & same_side & (distance < float(disc["radius"]))

    complexes = _feature_complexes(points, surface, direct, winner_weight,
                                   blob_response, rescue_discs, r_ctx, scale)
    if not complexes:
        return colors_rgba, None

    evidence = _build_gate_photo_evidence(
        mesh, atlas, colors, positions_texture, source_image,
        source_azimuth_deg, source_elevation_deg)
    if evidence is None:
        return colors_rgba, None
    stats["registration"] = [round(float(v), 4) for v in evidence["registration"]]

    dark_mask = surface & (luminance < 0.55 * bright_median)
    dark_labels, _ = cc_label(dark_mask, structure=np.ones((3, 3), bool))
    dark_sizes = np.bincount(dark_labels.ravel())
    if len(dark_sizes):
        dark_sizes[0] = 0
    hair_component = (dark_labels == int(dark_sizes.argmax())) \
        if dark_sizes.size > 1 else np.zeros(surface.shape, bool)

    deposits = _fringe_deposits(points, surface, direct, winner_weight,
                                evidence["gate_ok"], evidence["residual"],
                                hair_component, rescue_footprint, complexes,
                                r_ctx, bright_context_lum, bright_median)
    if not deposits and not rescue_discs:
        return colors_rgba, None

    # pre-repair render baseline for the render veto (one pass, reused by
    # every candidate), at the identity harness's own render size (896:
    # the 640 variant's aliasing noise measurably inflated micro-island
    # baselines on 2048 textures and over-vetoed safe stamps)
    # the veto battery spans the eye-gated near-frontal fan AND the
    # negative-azimuth sweep where the first integration's debris
    # regressions actually appeared (az-45/-70/-90 el10 were outside the
    # original set and shipped 0.0021->0.0044 growths)
    veto_views = [(0.0, (0.0, 22.5, -22.5, 35.0, -35.0, -45.0, -70.0,
                         -90.0, 90.0)),
                  (10.0, (0.0, 22.5, -22.5, -45.0, -70.0, -90.0))]
    veto_size = 896
    base_renders: List[Tuple[Any, List[Dict[str, Any]]]] = []
    for el, azimuths in veto_views:
        rendered = _render_with_colors(mesh, atlas, colors, azimuths, el,
                                       veto_size)
        entries = []
        for az, rgb in zip(azimuths, rendered):
            fg = _render_foreground(rgb)
            camera = _renderer_camera(mesh, az, el, veto_size)
            r_px = r_ctx * camera["px_per_world"]
            blobs, micro = _compact_dark_blobs_px(rgb, fg, (0.25 * r_px, 2.6 * r_px))
            entries.append({"blobs": blobs, "micro": micro, "rgb": rgb, "fg": fg})
        base_renders.append(((el, azimuths), entries))

    normals = np.asarray(normals_texture, dtype=np.float32)
    repaired_total = np.zeros(surface.shape, bool)
    # the ORIGINAL pre-repair baseline: the final speck consolidation and
    # the render veto's CUMULATIVE bound judge against the state before
    # ANY repair, while the in-loop per-stamp baseline advances with each
    # accepted stamp (each candidate judged on its own delta; cumulative
    # drift capped at the original battery worst — the critic-2 hardening)
    pre_renders = base_renders

    # 1. complex stamps, mode ladder full -> trace -> skip
    for k, comp in enumerate(complexes):
        if comp["kind"] != "core":
            continue
        for mode, keep_confident in (("full", False), ("trace", True)):
            candidate, stamp_mask, why = _gate_stamp(
                colors, evidence, comp,
                points=points, surface=surface, direct=direct,
                winner_weight=winner_weight, winner_index=winner_index,
                source_index=source_index, rescue_disc_mask=rescue_disc_mask,
                bright_context_lum=bright_context_lum,
                bright_median=bright_median, r_ctx=r_ctx, scale=scale,
                keep_confident=keep_confident)
            if why != "ok":
                continue
            veto = _texel_structure_veto(points, normals, surface, colors,
                                         candidate, stamp_mask, bright_median,
                                         r_ctx, scale,
                                         confident_mask=direct
                                         & (winner_weight >= 0.35))
            if veto is not None:
                stats["stamps"].append({"complex": k, "mode": mode,
                                        "veto": f"texel:{veto}"})
                continue
            veto, accepted_renders = _render_structure_veto(
                mesh, atlas, base_renders, candidate, r_ctx,
                photo_truth_mask=stamp_mask & evidence["gate_ok"],
                original_renders=pre_renders)
            if veto is not None:
                stats["stamps"].append({"complex": k, "mode": mode,
                                        "veto": f"render:{veto}"})
                continue
            colors = candidate
            if accepted_renders is not None:
                # each candidate is judged on ITS OWN structural delta:
                # an accepted stamp's (exempted, photo-true) content must
                # not count as growth against every later candidate
                base_renders = accepted_renders
            repaired_total |= stamp_mask
            stats["stamps"].append({
                "complex": k, "mode": mode,
                "texels": int(stamp_mask.sum()),
                "centroid": [round(float(v), 3) for v in comp["centroid"]],
                "radius": round(float(comp["radius"]), 3)})
            break

    # 2. deposit patches outside stamped regions; rescue-fringe recopies
    confident_mask = direct & (winner_weight >= 0.35)
    for deposit in deposits:
        if repaired_total[deposit["mask"]].mean() > 0.5:
            continue
        in_disc = float((deposit["mask"] & rescue_disc_mask).sum()) / deposit["area"]
        if in_disc >= 0.3:
            continue  # the disc interior belongs to the disc refresh below
        if deposit["in_rescue"] >= 0.5:
            if rescue_discs:
                candidate, patch, why = _rescue_recopy(
                    colors, deposit, rescue_discs[0], points=points,
                    surface=surface, confident_mask=confident_mask,
                    deviation=deviation, scale=scale)
                if why == "ok":
                    colors = candidate
                    repaired_total |= patch
                    stats["recopies"] += 1
            continue
        candidate, patch, why = _photo_patch(colors, evidence, deposit,
                                             points=points, surface=surface,
                                             direct=direct,
                                             bright_median=bright_median)
        if why == "ok":
            colors = candidate
            repaired_total |= patch
            stats["patches"] += 1

    # 3. disc refresh LAST: whole-disc re-transplant on current colors so
    # root repairs on the healthy side propagate into the twin
    if rescue_discs and repaired_total.any():
        for disc in rescue_discs:
            refreshed, refresh_stats = mirror_rescue_disc(
                colors[:, :, :3],
                positions_texture=positions_texture,
                center=disc["center"],
                radius=float(disc["radius"]),
                axis=1,
                source_mask=confident_mask,
                source_shift=disc.get("placement_shift"),
                feather_texels=3.0,
            )
            if refresh_stats.get("rescued_texels"):
                colors[:, :, :3] = refreshed
                stats["disc_refreshed"] = True

    if not repaired_total.any():
        return colors_rgba, None

    # 4. final render-informed speck consolidation over repaired texels
    colors, lifted = _consolidate_render_specks(
        mesh, atlas, colors, positions, normals, surface, repaired_total,
        pre_renders, bright_median, r_ctx)
    stats["speck_lifted_texels"] = int(lifted)

    stats["applied"] = True
    stats["repaired_texels"] = int(repaired_total.sum())
    if stats_out is not None:
        stats_out.update(stats)
    return colors, repaired_total
