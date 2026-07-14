"""Backend-agnostic UV texture baking from observed and reference views.

This module generalizes the validated TripoSR bake pipeline so geometry-only
backends (Hunyuan3D-2.1, Step1X) can emit textured assets from the same
projection math:

1. unwrap the mesh into a UV atlas (xatlas)
2. rasterize per-texel world positions and normals (GPU with CPU fallback)
3. project each observed/reference view onto texels that face its camera,
   with depth-occlusion tests
4. blend views with facing-weighted, best-view-biased weights and feathered
   seams, after harmonizing reference exposure against the primary view
5. fill texels no view could see: an optional backend color field
   (e.g. the TripoSR triplane) wins when present, otherwise diffusion-based
   atlas inpainting from the projected texels
6. bleed island borders so mipmaps and bilinear sampling stay clean

The per-step math lives in `backends/triposr_runtime.py` (it is shared, not
TripoSR-specific); this module owns the generic orchestration and the blend
improvements.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple


def _splat_silhouette(
    vertices: Any,
    *,
    azimuth_deg: float,
    elevation_deg: float,
    camera_distance: float,
    fovy_deg: float = 40.0,
    size: int = 96,
    projection_model: str = "perspective",
    ortho_half_extent: Optional[float] = None,
) -> Any:
    """Approximate the mesh silhouette from a camera pose by point splatting.

    Projects (subsampled) vertices through the same pinhole model as the
    texture projector, splats them into a small binary mask, and closes gaps
    morphologically. This avoids any GL dependency and is fast enough to
    evaluate dozens of candidate poses, which is all silhouette-based pose
    scoring needs.
    """
    import math

    import numpy as np

    from .backends.triposr_runtime import _tripo_camera_position, _tripo_look_at_matrix

    eye = _tripo_camera_position(
        azimuth_deg=float(azimuth_deg),
        elevation_deg=float(elevation_deg),
        camera_distance=float(camera_distance),
    )
    view = _tripo_look_at_matrix(eye, np.zeros(3, dtype=np.float32), np.array([0.0, 0.0, 1.0], dtype=np.float32))
    homogeneous = np.concatenate([vertices, np.ones((len(vertices), 1), dtype=np.float32)], axis=1)
    camera_space = homogeneous @ view.T
    if str(projection_model) == "orthographic":
        half_extent = float(ortho_half_extent or 1.0)
        scale = 0.5 * size / max(half_extent, 1e-6)
        px = scale * camera_space[:, 0] + size / 2.0
        py = -scale * camera_space[:, 1] + size / 2.0
    else:
        depth = np.maximum(-camera_space[:, 2], 1e-6)
        focal = 0.5 * size / math.tan(0.5 * math.radians(float(fovy_deg)))
        px = focal * camera_space[:, 0] / depth + size / 2.0
        py = -focal * camera_space[:, 1] / depth + size / 2.0
    mask = np.zeros((size, size), dtype=bool)
    valid = (px >= 0) & (px < size) & (py >= 0) & (py < size)
    mask[py[valid].astype(np.int32), px[valid].astype(np.int32)] = True
    try:
        from scipy.ndimage import binary_closing, binary_dilation, binary_erosion

        mask = binary_dilation(mask, iterations=1)
        mask = binary_closing(mask, structure=np.ones((3, 3), dtype=bool), iterations=2)
        # The dilation grows the silhouette rim by ~1.3 px, which biases any
        # edge-based registration toward ENLARGING the photo by 2r/D
        # (measured: a pixel-perfect canonical photo registered at scale
        # 1.04). Erode once to restore the true rim; interior gaps stay
        # closed because closing already ran.
        mask = binary_erosion(mask, iterations=1, border_value=False)
    except Exception:
        pass
    return mask


def _normalized_mask_iou(mask_a: Any, mask_b: Any, *, size: int = 64) -> float:
    """Scale/translation-invariant IoU: crop each mask to its bounding box,
    pad to square (preserving aspect ratio, which is a real pose cue), resize
    to a common grid, and intersect. This decouples pose scoring from camera
    distance, which is fitted separately once the pose is chosen."""
    import numpy as np

    def _normalize(mask: Any) -> Optional[Any]:
        mask = np.asarray(mask, dtype=bool)
        rows = np.nonzero(mask.any(axis=1))[0]
        cols = np.nonzero(mask.any(axis=0))[0]
        if len(rows) == 0 or len(cols) == 0:
            return None
        cropped = mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]
        side = max(cropped.shape)
        padded = np.zeros((side, side), dtype=bool)
        top = (side - cropped.shape[0]) // 2
        left = (side - cropped.shape[1]) // 2
        padded[top : top + cropped.shape[0], left : left + cropped.shape[1]] = cropped
        from PIL import Image

        image = Image.fromarray((padded * 255).astype(np.uint8))
        return np.asarray(image.resize((size, size), Image.BILINEAR)) > 127

    normalized_a = _normalize(mask_a)
    normalized_b = _normalize(mask_b)
    if normalized_a is None or normalized_b is None:
        return 0.0
    intersection = float(np.logical_and(normalized_a, normalized_b).sum())
    union = float(np.logical_or(normalized_a, normalized_b).sum())
    return intersection / union if union > 0 else 0.0


def recenter_to_canonical_frame(
    observed_rgba: Any,
    *,
    size: int = 1024,
    border_ratio: float = 0.15,
    center_px: Optional[Tuple[float, float]] = None,
) -> Any:
    """Recenter a photo to the canonical training frame of shape models.

    Hunyuan-family models preprocess every conditioning image the same way
    (`ImageProcessorV2.recenter`): crop to the alpha bounding box, scale so
    the larger side fills `1 - border_ratio` of a square canvas, center.
    The reconstructed mesh corresponds to THAT frame under an orthographic
    camera — so replicating the recenter makes photo-to-mesh registration
    deterministic instead of a silhouette-matching estimation problem.

    `center_px` places the subject's bbox center at an explicit pixel
    (default: the frame center). The orthographic projector centers the
    WORLD ORIGIN in its sample map, and away from the canonical front the
    mesh's camera-plane bbox center is displaced from the origin — callers
    pass `projected_frame_center_px` for that pose so the photo lands on
    the surface that imaged it (see SHIP-03 provenance: at az+30/el+15 the
    displacement is 54 px at 1024, which smeared the prow with off-surface
    content).
    """
    import numpy as np
    from PIL import Image

    image = observed_rgba.convert("RGBA") if hasattr(observed_rgba, "convert") else observed_rgba
    array = np.asarray(image, dtype=np.uint8)
    alpha = array[:, :, 3]
    rows = np.nonzero(alpha.max(axis=1) > 12)[0]
    cols = np.nonzero(alpha.max(axis=0) > 12)[0]
    if len(rows) == 0 or len(cols) == 0:
        return image.resize((size, size), Image.LANCZOS)
    top, bottom = int(rows[0]), int(rows[-1]) + 1
    left, right = int(cols[0]), int(cols[-1]) + 1
    cropped = image.crop((left, top, right, bottom))
    width, height = cropped.size
    desired = int(round(size * (1.0 - float(border_ratio))))
    scale = desired / max(width, height)
    new_w, new_h = max(1, int(round(width * scale))), max(1, int(round(height * scale)))
    resized = cropped.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    if center_px is None:
        center_x, center_y = size / 2.0, size / 2.0
    else:
        center_x, center_y = float(center_px[0]), float(center_px[1])
    canvas.paste(
        resized,
        (int(round(center_x - new_w / 2.0)), int(round(center_y - new_h / 2.0))),
    )
    return canvas


def projected_frame_center_px(
    mesh: Any,
    *,
    azimuth_deg: float,
    elevation_deg: float,
    size: int = 1024,
    border_ratio: float = 0.15,
) -> Tuple[float, float]:
    """Pixel where the mesh's camera-plane bbox center lands in the
    canonical frame at a pose, under the projector's own camera convention.

    The orthographic sample map centers the WORLD ORIGIN at the frame
    center; the canonical recenter centers the photo's ALPHA BBOX there.
    Those two conventions agree only when the mesh's bbox center projects
    onto the camera axis — true at the canonical front (the model itself
    recentered the conditioning image), false in general at other poses
    (measured: starship +54/-28 px at az+30/el+15, face +16/+8 px at
    az+20/el+8, owl ~1 px at az0). Registering each photo to THIS center
    removes the discrepancy deterministically, with no content-based
    search, and degenerates to the plain recenter exactly where the old
    assumption held.
    """
    import numpy as np

    from .backends.triposr_runtime import _tripo_camera_position, _tripo_look_at_matrix

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    eye = _tripo_camera_position(
        azimuth_deg=float(azimuth_deg),
        elevation_deg=float(elevation_deg),
        camera_distance=3.0,
    )
    view = _tripo_look_at_matrix(
        eye, np.zeros(3, dtype=np.float32), np.array([0.0, 0.0, 1.0], dtype=np.float32)
    )
    camera_space = vertices @ view[:3, :3].T + view[:3, 3]
    center_x = 0.5 * (float(camera_space[:, 0].min()) + float(camera_space[:, 0].max()))
    center_y = 0.5 * (float(camera_space[:, 1].min()) + float(camera_space[:, 1].max()))
    extent_x = float(camera_space[:, 0].max() - camera_space[:, 0].min())
    extent_y = float(camera_space[:, 1].max() - camera_space[:, 1].min())
    half_extent = max(extent_x, extent_y) / (2.0 * max(1.0 - float(border_ratio), 1e-6))
    ortho_scale = 0.5 * float(size) / max(half_extent, 1e-6)
    return (
        float(size) / 2.0 + ortho_scale * center_x,
        float(size) / 2.0 - ortho_scale * center_y,
    )


def canonical_ortho_half_extent(
    mesh: Any,
    *,
    azimuth_deg: float,
    elevation_deg: float,
    border_ratio: float = 0.15,
) -> float:
    """Orthographic half-extent that reproduces the canonical framing.

    Under the canonical recenter (subject's larger camera-plane extent fills
    `1 - border_ratio` of the frame), the matching orthographic camera must
    map the mesh's larger camera-plane extent to the same fraction:

        2 * half_extent * (1 - border_ratio) = max(extent_x, extent_y)
    """
    import numpy as np

    from .backends.triposr_runtime import _tripo_camera_position, _tripo_look_at_matrix

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    eye = _tripo_camera_position(
        azimuth_deg=float(azimuth_deg), elevation_deg=float(elevation_deg), camera_distance=3.0
    )
    view = _tripo_look_at_matrix(eye, np.zeros(3, dtype=np.float32), np.array([0.0, 0.0, 1.0], dtype=np.float32))
    camera_space = vertices @ view[:3, :3].T + view[:3, 3]
    extent_x = float(camera_space[:, 0].max() - camera_space[:, 0].min())
    extent_y = float(camera_space[:, 1].max() - camera_space[:, 1].min())
    return max(extent_x, extent_y) / (2.0 * max(1.0 - float(border_ratio), 1e-6))


def estimate_view_pose(
    mesh: Any,
    *,
    observed_rgba: Any,
    azimuth_step_deg: float = 15.0,
    azimuth_window_deg: float = 75.0,
    elevation_candidates_deg: Sequence[float] = (-15.0, 0.0, 15.0, 30.0),
    prior_strength: float = 0.12,
    default_distance: float = 1.9,
    center_azimuth_deg: float = 0.0,
    center_elevation_deg: float = 0.0,
    min_iou: float = 0.45,
) -> Dict[str, float]:
    """Estimate which camera pose an observed photo was taken from.

    Scores candidate poses by silhouette similarity between the photo's
    alpha mask and a splatted mesh silhouette, picks the best, then fits the
    camera distance at that pose. The search runs in a window around a prior
    pose (`center_azimuth_deg`/`center_elevation_deg`): the canonical front
    for source photos on canonicalizing backends (Hunyuan3D), or a reference
    view's declared angle when refining nominal labels like `side_left`.

    Silhouette IoU alone is dangerously ambiguous for near-rotationally-
    symmetric subjects (a human head scores ~0.95 at many azimuths), so the
    search is regularized two ways: candidates stay inside the window, and
    the score carries a prior penalty proportional to angular distance from
    the center pose:

        score = IoU - prior_strength * (|az - center_az|/window
                                        + 0.5 * |el - center_el|/45)

    This keeps genuinely off-label poses recoverable while preventing
    high-IoU-but-wrong poses from winning on subjects whose silhouette
    barely changes with azimuth. Falls back to the center pose when the best
    IoU stays under `min_iou`.
    """
    import numpy as np

    image = np.asarray(observed_rgba, dtype=np.float32)
    center_azimuth = float(center_azimuth_deg)
    center_elevation = float(center_elevation_deg)
    result = {
        "azimuth_deg": center_azimuth,
        "elevation_deg": center_elevation,
        "camera_distance": float(default_distance),
        "iou": 0.0,
    }
    if image.ndim != 3 or image.shape[2] < 4:
        return result
    alpha = image[:, :, 3]
    observed_mask = alpha > (0.5 * alpha.max() if alpha.max() > 1.0 else 0.5)
    if observed_mask.mean() < 0.005:
        return result

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    if len(vertices) == 0:
        return result
    if len(vertices) > 15000:
        step = len(vertices) // 15000 + 1
        vertices = vertices[::step]

    # Keep the whole mesh inside the splat frustum regardless of mesh scale
    # (Hunyuan meshes fill [-1, 1]; TripoSR spans about half of that). The
    # bounding-sphere containment condition for a camera with half-angle
    # fovy/2 is distance >= radius / sin(fovy/2); the 1.05 adds margin.
    import math

    bounding_radius = float(np.linalg.norm(vertices, axis=1).max()) or 1.0
    splat_distance = max(
        float(default_distance) * 1.6,
        bounding_radius / math.sin(math.radians(20.0)) * 1.05,
    )

    window = abs(float(azimuth_window_deg))

    def scored(azimuth: float, elevation: float) -> tuple[float, float]:
        silhouette = _splat_silhouette(
            vertices,
            azimuth_deg=float(azimuth),
            elevation_deg=float(elevation),
            camera_distance=splat_distance,
        )
        iou = _normalized_mask_iou(silhouette, observed_mask)
        penalty = float(prior_strength) * (
            abs(azimuth - center_azimuth) / max(window, 1e-6)
            + 0.5 * abs(elevation - center_elevation) / 45.0
        )
        return iou - penalty, iou

    candidates: List[Dict[str, float]] = []
    center_iou = 0.0
    # Build the grid from symmetric integer step offsets so the CENTER pose
    # is always scored. `arange(-window, window, step)` excludes the center
    # whenever window is not a step multiple (window 20, step 15 yields
    # {-20, -5, +10}), which silently made "declared pose" comparisons
    # against a candidate that was never evaluated.
    steps = int(window // max(float(azimuth_step_deg), 1e-6))
    azimuths = center_azimuth + np.arange(-steps, steps + 1, dtype=np.float64) * float(azimuth_step_deg)
    for elevation in elevation_candidates_deg:
        for azimuth in azimuths:
            score, iou = scored(float(azimuth), float(elevation))
            candidates.append(
                {"azimuth_deg": float(azimuth), "elevation_deg": float(elevation), "score": score, "iou": iou}
            )
            if abs(azimuth - center_azimuth) < 1e-6 and abs(float(elevation) - center_elevation) < 1e-6:
                center_iou = iou
    candidates.sort(key=lambda row: row["score"], reverse=True)
    best = candidates[0] if candidates else {
        "azimuth_deg": center_azimuth,
        "elevation_deg": center_elevation,
        "score": -1.0,
        "iou": 0.0,
    }
    best_score, best_iou = best["score"], best["iou"]
    best_azimuth, best_elevation = best["azimuth_deg"], best["elevation_deg"]

    # Local refinement at half the grid step around the winner.
    half_step = float(azimuth_step_deg) / 2.0
    for delta in (-half_step, half_step):
        score, iou = scored(best_azimuth + delta, best_elevation)
        if score > best_score:
            best_score, best_iou = score, iou
            best_azimuth = best_azimuth + delta

    # An unconvincing best match means the silhouette carries no usable pose
    # signal (degenerate masks, extreme mismatch); trusting it would place
    # the projection arbitrarily. Fall back to the center pose.
    if best_iou < float(min_iou):
        best_azimuth, best_elevation = center_azimuth, center_elevation

    distance = estimate_camera_distance(
        mesh,
        observed_rgba=observed_rgba,
        azimuth_deg=best_azimuth,
        elevation_deg=best_elevation,
        default_distance=default_distance,
    )
    return {
        "azimuth_deg": round(best_azimuth, 2),
        "elevation_deg": round(best_elevation, 2),
        "camera_distance": round(float(distance), 4),
        "iou": round(float(best_iou), 4),
        "center_iou": round(float(center_iou), 4),
        "candidates": [
            {
                "azimuth_deg": round(row["azimuth_deg"], 2),
                "elevation_deg": round(row["elevation_deg"], 2),
                "iou": round(row["iou"], 4),
                "score": round(row["score"], 4),
            }
            for row in candidates[:8]
        ],
    }


def estimate_camera_distance(
    mesh: Any,
    *,
    observed_rgba: Any,
    azimuth_deg: float,
    elevation_deg: float,
    default_distance: float = 1.9,
    fovy_deg: float = 40.0,
) -> float:
    """Estimate the projection camera distance that matches the observed view.

    Different backends emit meshes at different world scales (TripoSR spans
    roughly one unit, Hunyuan3D fills [-1, 1]). Projecting the observed photo
    back onto the surface therefore needs a per-mesh camera distance, or the
    subject overflows/underfills the virtual frustum and texel lookups land
    on the wrong pixels. For a pinhole camera the projected object height
    scales inversely with distance, so one projection pass at the default
    distance gives the correction factor directly:

        d* = d0 * (projected_extent / observed_foreground_extent)

    Falls back to `default_distance` when the observed alpha gives no usable
    foreground bounds.
    """
    import math

    import numpy as np

    from .backends.triposr_runtime import _tripo_camera_position, _tripo_look_at_matrix

    image = np.asarray(observed_rgba, dtype=np.float32)
    if image.ndim != 3 or image.shape[2] < 4:
        return float(default_distance)
    height, width = image.shape[:2]
    alpha = image[:, :, 3]
    foreground = alpha > (0.5 * alpha.max() if alpha.max() > 1.0 else 0.5)
    coverage = float(foreground.mean())
    if coverage < 0.01 or coverage > 0.99:
        return float(default_distance)
    rows = np.nonzero(foreground.any(axis=1))[0]
    cols = np.nonzero(foreground.any(axis=0))[0]
    # Both axes normalize by HEIGHT: the projector's NDC unit is height
    # based (focal = 0.5*H/tan), so a width-normalized column extent makes
    # wide subjects in landscape frames look smaller by W/H and biases the
    # distance up to tens of percent.
    observed_extent = max(
        (rows[-1] - rows[0] + 1) / float(height),
        (cols[-1] - cols[0] + 1) / float(height),
    )

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    if len(vertices) == 0:
        return float(default_distance)
    if len(vertices) > 20000:
        step = len(vertices) // 20000 + 1
        vertices = vertices[::step]
    homogeneous = np.concatenate([vertices, np.ones((len(vertices), 1), dtype=np.float32)], axis=1)
    focal = 0.5 / math.tan(0.5 * math.radians(float(fovy_deg)))

    def projected_extent_at(distance: float) -> float:
        eye = _tripo_camera_position(
            azimuth_deg=float(azimuth_deg),
            elevation_deg=float(elevation_deg),
            camera_distance=float(distance),
        )
        view = _tripo_look_at_matrix(
            eye, np.zeros(3, dtype=np.float32), np.array([0.0, 0.0, 1.0], dtype=np.float32)
        )
        camera_space = homogeneous @ view.T
        depth = np.maximum(-camera_space[:, 2], 1e-6)
        ndc_x = focal * camera_space[:, 0] / depth
        ndc_y = focal * camera_space[:, 1] / depth
        return max(float(ndc_x.max() - ndc_x.min()), float(ndc_y.max() - ndc_y.min()))

    if observed_extent <= 1e-6:
        return float(default_distance)
    # Fixed-point iteration: the linear correction d* = d0 * proj/obs is
    # exact only for a subject plane through the origin; real subjects
    # extend toward the camera, so one pass is biased by 10-16% (worse for
    # deep subjects). Two or three re-projections at the updated distance
    # converge well under 1%.
    estimated = float(default_distance)
    for _ in range(3):
        extent = projected_extent_at(estimated)
        if extent <= 1e-6:
            return float(default_distance)
        updated = float(
            np.clip(
                estimated * extent / observed_extent,
                0.5 * float(default_distance),
                4.0 * float(default_distance),
            )
        )
        if abs(updated - estimated) < 0.005 * max(estimated, 1e-6):
            estimated = updated
            break
        estimated = updated
    return estimated


def estimate_pose_photometric(
    mesh: Any,
    observed_rgba: Any,
    *,
    azimuth_window_deg: float = 40.0,
    azimuth_step_deg: float = 5.0,
    elevation_candidates: Sequence[float] = (-15.0, -8.0, 0.0, 8.0, 15.0),
    size: int = 256,
    border_ratio: float = 0.15,
    min_margin: float = 0.002,
    min_peak_score: float = 0.008,
) -> Dict[str, Any]:
    """Estimate the camera pose of a photo by signed-gradient correlation.

    Silhouettes cannot recover the pose of near-rotationally-symmetric
    subjects (a head's outline barely changes over +/-25 degrees of yaw),
    yet generative backends canonicalize the OBJECT (symmetry plane to the
    world axes), not the input camera: a photo of a subject whose head is
    turned lands the camera 15-25 degrees away from the canonical front.
    Interior features disambiguate what silhouettes cannot: the mesh's
    creases (eye sockets, nose, lips) produce shading gradients at the same
    anatomical locations as the photo's albedo/shading gradients.

    Two properties of the scorer are load-bearing, both established
    empirically against ground truth (photos with independently measured
    head yaw):

    - The correlation must be over the signed gradient VECTOR field
      (gx, gy), not gradient magnitude. Faces are bilaterally symmetric in
      gradient magnitude, so a magnitude scorer peaks equally at the true
      pose and its mirror; signed vectors break the tie (the mirror pose
      anti-correlates horizontally) and yield a single peak.
    - Silhouette-edge gradients must be de-emphasized (interior distance
      weighting), otherwise the outline — which is pose-insensitive on
      heads — swamps the interior feature signal.

    If no candidate beats the declared pose by ``min_margin``, the declared
    pose wins and ``estimated`` stays False: weak evidence must not move
    the projection.
    """
    import numpy as np
    from PIL import Image
    from scipy.ndimage import distance_transform_edt, gaussian_filter

    from .rendering import render_mesh_views

    result: Dict[str, Any] = {
        "azimuth_deg": 0.0,
        "elevation_deg": 0.0,
        "estimated": False,
        "score": None,
        "score_at_declared": None,
    }

    image = observed_rgba.convert("RGBA") if hasattr(observed_rgba, "convert") else observed_rgba
    canonical = recenter_to_canonical_frame(image, border_ratio=border_ratio)
    photo = np.asarray(canonical.resize((size, size), Image.BILINEAR), dtype=np.float32) / 255.0
    photo_alpha = photo[:, :, 3] > 0.5
    if photo_alpha.mean() < 0.02:
        return result

    def interior_weight(mask: Any) -> Any:
        distances = distance_transform_edt(mask)
        peak = float(distances.max())
        if peak <= 0:
            return distances
        return np.sqrt(distances / peak)

    def gradient_field(gray: Any, mask: Any) -> Tuple[Any, Any]:
        smooth = gaussian_filter(gray, 1.5)
        gy, gx = np.gradient(smooth)
        # RMS-normalize inside the mask so photo and render contribute at
        # comparable scales regardless of contrast.
        rms = float(np.sqrt(np.mean((gx**2 + gy**2)[mask]))) if mask.any() else 0.0
        rms = max(rms, 1e-8)
        weights = interior_weight(mask)
        return gx / rms * weights, gy / rms * weights

    photo_gx, photo_gy = gradient_field(photo[:, :, :3].mean(axis=2), photo_alpha)

    photo_rows = np.nonzero(photo_alpha.any(axis=1))[0]
    photo_top = int(photo_rows[0]) if len(photo_rows) else 0
    photo_bottom = int(photo_rows[-1]) if len(photo_rows) else size - 1
    photo_widths = photo_alpha.sum(axis=1).astype(np.float64)
    photo_cols = np.arange(size, dtype=np.float64)
    photo_centroid_x = float(
        (photo_alpha * photo_cols[None, :]).sum() / max(photo_alpha.sum(), 1)
    )
    # Rows at the frame border are crop lines, not subject shape.
    photo_usable = photo_bottom - photo_top - (2 if photo_bottom >= size - 2 else 0)

    def rendered_frame(azimuth: float, elevation: float) -> Optional[Tuple[Any, Any]]:
        """Render the mesh at a pose and align it into the photo's frame.

        Alignment uses CROP-IMMUNE anchors: subject TOP, horizontal
        centroid, and mean silhouette width over the rows both observe.
        Full-bbox recentering breaks on cropped photos (a chest-cropped
        portrait's bbox scale disagrees with the full-bust render), while
        no alignment at all breaks on elongated subjects whose projected
        aspect swings with elevation (the starship). Width-over-common-rows
        is the same signature register_view_by_width_profile validated.
        Returns (gray, mask) in the photo frame, or None.
        """
        try:
            # The scorer's margins were calibrated against the legacy
            # fixed-world-light shading; the headlight guide (a different
            # gradient field) shifts its scores and flipped a validated
            # declared-pose case to a wrong estimate (owl: est. el -15).
            rendered = render_mesh_views(
                mesh, azimuths=(float(azimuth),), elevation=float(elevation),
                size=size, lighting="fixed",
            )[0].convert("RGBA")
        except Exception:
            return None
        array = np.asarray(rendered, dtype=np.float32) / 255.0
        background = array[2, 2, :3]
        mask = np.abs(array[:, :, :3] - background).sum(axis=2) > 0.08
        if mask.mean() < 0.02:
            return None
        render_rows = np.nonzero(mask.any(axis=1))[0]
        if not len(render_rows):
            return None
        render_top = int(render_rows[0])
        render_bottom = int(render_rows[-1])
        render_widths = mask.sum(axis=1).astype(np.float64)
        span = min(photo_usable, render_bottom - render_top)
        if span < 10:
            return None
        photo_mean_width = float(photo_widths[photo_top : photo_top + span].mean())
        render_mean_width = float(render_widths[render_top : render_top + span].mean())
        if photo_mean_width < 2 or render_mean_width < 2:
            return None
        scale = photo_mean_width / render_mean_width
        render_centroid_x = float((mask * photo_cols[None, :]).sum() / max(mask.sum(), 1))
        # Map render -> photo frame: out = s*(in - anchor_render) + anchor_photo,
        # anchored at (subject top row, silhouette centroid x). PIL affine
        # takes the inverse map.
        inv = 1.0 / scale
        matrix = (
            inv,
            0.0,
            render_centroid_x - inv * photo_centroid_x,
            0.0,
            inv,
            render_top - inv * photo_top,
        )
        rgba_render = np.zeros((size, size, 4), dtype=np.uint8)
        rgba_render[:, :, :3] = (array[:, :, :3] * 255).astype(np.uint8)
        rgba_render[:, :, 3] = np.where(mask, 255, 0)
        warped = Image.fromarray(rgba_render).transform(
            (size, size), Image.AFFINE, matrix, resample=Image.BILINEAR, fillcolor=(0, 0, 0, 0)
        )
        render_array = np.asarray(warped, dtype=np.float32) / 255.0
        render_mask = render_array[:, :, 3] > 0.5
        return render_array[:, :, :3].mean(axis=2), render_mask

    def score_pose(azimuth: float, elevation: float) -> float:
        frame = rendered_frame(azimuth, elevation)
        if frame is None:
            return -np.inf
        gray, render_mask = frame
        overlap = photo_alpha & render_mask
        if int(overlap.sum()) < 500:
            return -np.inf
        render_gx, render_gy = gradient_field(gray, render_mask)
        return float(
            (photo_gx[overlap] * render_gx[overlap] + photo_gy[overlap] * render_gy[overlap]).mean()
        )

    def antisymmetric_map(gray: Any, mask: Any) -> Tuple[Any, Any]:
        flipped = gray[:, ::-1]
        flipped_mask = mask[:, ::-1]
        both = mask & flipped_mask
        anti = np.where(both, 0.5 * (gray - flipped), 0.0)
        rms = float(np.sqrt(np.mean(anti[both] ** 2))) if both.any() else 0.0
        return anti / max(rms, 1e-8), both

    photo_anti, photo_anti_mask = antisymmetric_map(photo[:, :, :3].mean(axis=2), photo_alpha)

    def chirality_score(azimuth: float, elevation: float) -> float:
        """Correlate ANTI-SYMMETRIC luminance components of photo and render.

        On a bilaterally symmetric mesh, renders at +az and -az are
        near-mirror images, so the full gradient correlation's sign margin
        is thin (measured: 0.1% vertex jitter flipped the argmax sign).
        The horizontal anti-symmetric luminance component isolates exactly
        the chirality carriers — the direction the subject's center line
        shifts against the silhouette, asymmetric masses like a hair
        parting — and is sign-opposite between the two mirror poses, so
        its correlation decides the sign with a margin the symmetric
        content cannot dilute.
        """
        rendered_gray_mask = rendered_frame(azimuth, elevation)
        if rendered_gray_mask is None:
            return -np.inf
        gray, mask = rendered_gray_mask
        anti, anti_mask = antisymmetric_map(gray, mask)
        both = photo_anti_mask & anti_mask
        if int(both.sum()) < 500:
            return -np.inf
        return float((photo_anti[both] * anti[both]).mean())

    steps = int(azimuth_window_deg // max(azimuth_step_deg, 1e-6))
    azimuths = np.arange(-steps, steps + 1, dtype=np.float64) * float(azimuth_step_deg)
    best_azimuth, best_elevation, best_score = 0.0, 0.0, -np.inf
    declared_score = None
    score_rows: Dict[float, Any] = {}
    for elevation in elevation_candidates:
        row = np.full(len(azimuths), -np.inf)
        for index, azimuth in enumerate(azimuths):
            score = score_pose(float(azimuth), float(elevation))
            row[index] = score
            if azimuth == 0.0 and float(elevation) == 0.0:
                declared_score = score
            if score > best_score:
                best_azimuth, best_elevation, best_score = float(azimuth), float(elevation), score
        score_rows[float(elevation)] = row
    if not np.isfinite(best_score):
        return result

    # PLATEAU CENTROID (stability): on near-symmetric subjects the score
    # curve around the optimum is a broad plateau — the argmax within it is
    # decided by sampling noise, and 0.1-0.3% vertex jitter re-rolled the
    # pose by up to 7.5 degrees between bakes. The contiguous region around
    # the argmax whose scores stay within 90% of the peak is the real
    # answer; its score-weighted centroid is stable where the argmax is a
    # lottery.
    row = score_rows[best_elevation]
    peak_index = int(np.argmax(row))
    threshold = 0.9 * float(row[peak_index])
    low = peak_index
    while low > 0 and np.isfinite(row[low - 1]) and row[low - 1] >= threshold:
        low -= 1
    high = peak_index
    while high < len(row) - 1 and np.isfinite(row[high + 1]) and row[high + 1] >= threshold:
        high += 1
    plateau_scores = row[low : high + 1]
    plateau_azimuths = azimuths[low : high + 1]
    if len(plateau_azimuths) > 1 and np.isfinite(plateau_scores).all():
        weights_plateau = plateau_scores - threshold
        total = float(weights_plateau.sum())
        if total > 1e-12:
            best_azimuth = float((plateau_azimuths * weights_plateau).sum() / total)
            centroid_score = score_pose(best_azimuth, best_elevation)
            if np.isfinite(centroid_score):
                best_score = float(centroid_score)

    # Chirality tie-break: the mirror pose of the argmax is a structural
    # near-tie on symmetric subjects; decide the SIGN with the
    # anti-symmetric correlation, which is robust where the full score is
    # jitter-fragile.
    if abs(best_azimuth) > 1e-6:
        chirality_pos = chirality_score(abs(best_azimuth), best_elevation)
        chirality_neg = chirality_score(-abs(best_azimuth), best_elevation)
        if np.isfinite(chirality_pos) and np.isfinite(chirality_neg):
            signed = abs(best_azimuth) if chirality_pos >= chirality_neg else -abs(best_azimuth)
            if signed != best_azimuth:
                flipped_score = score_pose(signed, best_elevation)
                if np.isfinite(flipped_score):
                    best_azimuth, best_score = float(signed), float(flipped_score)

    # Local refinement around the coarse argmax, both axes. Elevation
    # refinement matters because the candidate grid is coarse (7-8 degree
    # gaps): a photo shot at +15 must not stay pinned to +8 (measured on
    # the starship lane: the elevation error alone cost 3x observed
    # coverage).
    for azimuth in (best_azimuth - 2.5, best_azimuth + 2.5):
        score = score_pose(azimuth, best_elevation)
        if score > best_score:
            best_azimuth, best_score = float(azimuth), score
    for elevation in (best_elevation - 4.0, best_elevation + 4.0):
        score = score_pose(best_azimuth, elevation)
        if score > best_score:
            best_elevation, best_score = float(elevation), score

    result["score"] = round(float(best_score), 4)
    if declared_score is not None and np.isfinite(declared_score):
        result["score_at_declared"] = round(float(declared_score), 4)
    # ACCEPTANCE GATE (never silently commit a weak estimate): the
    # correlation must be positive (an anti-correlated argmax means no pose
    # matched), must clearly beat the declared pose, and must reach an
    # ABSOLUTE floor — adversarial verification measured bad commits at
    # scores 0.0043-0.0052 (a frontal statue moved to az +32.5) while
    # genuine matches score 0.012-0.038. Below the floor the declared pose
    # wins and the result carries an explicit warning for the caller's
    # metadata.
    #
    # The margin is RELATIVE when the declared pose itself matches well:
    # a declared pose scoring 0.044 was displaced by a rival at 0.053
    # (+19%) that visibly mismatched the photo (a frontal statue moved to
    # az -5/el -8 — the rival correlated with busy base-region gradients,
    # not the subject). Displacing a good declared match demands
    # proportionally strong evidence; displacing a non-match (declared
    # score near zero or negative, e.g. a turned head) needs only the
    # absolute margin.
    required_margin = float(min_margin)
    if declared_score is not None and np.isfinite(declared_score) and declared_score > 0:
        required_margin = max(required_margin, 0.25 * float(declared_score))
    accept = (
        best_score > float(min_peak_score)
        and (
            declared_score is None
            or not np.isfinite(declared_score)
            or (best_score - declared_score) > required_margin
        )
    )
    if not accept and best_score > 0:
        result["rejected_reason"] = (
            f"peak score {best_score:.4f} below floor {float(min_peak_score):.4f}"
            if best_score <= float(min_peak_score)
            else (
                f"margin over declared {best_score - (declared_score or 0.0):.4f} "
                f"below required {required_margin:.4f}"
            )
        )
    if accept and (abs(best_azimuth) > 1e-6 or abs(best_elevation) > 1e-6):
        result.update(
            {"azimuth_deg": best_azimuth, "elevation_deg": best_elevation, "estimated": True}
        )
    return result


def register_reference_by_source_overlap(
    reference_rgba: Any,
    *,
    positions_texture: Any,
    source_projection: Mapping[str, Any],
    azimuth_deg: float,
    elevation_deg: float,
    camera_distance: float,
    ortho_half_extent: float,
    normals_texture: Optional[Any] = None,
    scale_candidates: Sequence[float] = (0.98, 1.0, 1.02, 1.04),
    shift_range: float = 0.08,
    shift_step: float = 0.01,
    # Noise guard for the 3-parameter similarity fit; a few hundred texels
    # is statistically ample (production overlaps run 10^4-10^5). 800
    # proved brittle: an upstream confidence retune shaved a legitimate
    # small overlap from 905 to 745 and silently disabled registration.
    min_overlap: int = 400,
    min_source_weight: float = 0.25,
    min_improvement: float = 0.01,
    min_reference_facing: float = 0.2,
) -> Tuple[Any, Dict[str, Any]]:
    """Align a reference photo to the SOURCE'S PAINTED TRUTH on shared surface.

    Silhouette-based registration aligns outlines — on a head that means the
    HAIR contour — and can leave interior features (eye, nose, mouth)
    displaced by several percent of the frame (measured: 58 px nose-anchor
    error at 1024, which painted the profile's eye on the temple). But once
    the source view is projected, every texel the source saw with solid
    weight has a known-good color at a known surface point; a residual
    similarity transform of the reference photo only changes where that
    surface samples the photo. Minimizing the source-weighted mean |RGB|
    disagreement over the mutual-overlap texels therefore registers the
    reference's INTERIOR CONTENT to the source's, using the pipeline's own
    disagreement measure — subject-agnostic and self-validating (the warp
    is accepted only on a real improvement, `min_improvement`).

    Orthographic frames only (sample coordinates are recomputed here with
    the same formula as the projector).
    """
    import numpy as np
    from PIL import Image

    from .backends.triposr_runtime import _tripo_camera_position, _tripo_look_at_matrix

    image = reference_rgba.convert("RGBA") if hasattr(reference_rgba, "convert") else reference_rgba
    array = np.asarray(image, dtype=np.float32) / 255.0
    height, width = array.shape[:2]
    stats: Dict[str, Any] = {
        "applied": False,
        "scale": 1.0,
        "shift_x": 0.0,
        "shift_y": 0.0,
        "overlap": 0,
        "err_before": None,
        "err_after": None,
    }

    positions = np.asarray(positions_texture, dtype=np.float32)
    source_weight = np.asarray(source_projection["weight"], dtype=np.float32)
    source_rgb = np.asarray(source_projection["rgba"], dtype=np.float32)[:, :, :3]

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
    ortho_scale = 0.5 * float(height) / max(float(ortho_half_extent), 1e-6)
    sample_x = ortho_scale * camera_space[:, :, 0] + float(width) / 2.0 - 0.5
    sample_y = -ortho_scale * camera_space[:, :, 1] + float(height) / 2.0 - 0.5

    in_frame = (
        (sample_x >= 0)
        & (sample_x <= width - 1)
        & (sample_y >= 0)
        & (sample_y <= height - 1)
        & (positions[:, :, 3] > 0)
    )
    # MUTUAL VISIBILITY (adversarial audit, critical): "in frame" is not
    # "seen by this camera" — an orthographic far-side camera has every
    # front texel in frame THROUGH the body. Without a facing gate the
    # back view's "overlap" was the entire front-painted surface, the fit
    # matched the reference against content it cannot see, and every
    # generated reference was driven to the search bound (measured +-0.08
    # = 61 px at 768: the displaced tail and wing). Only texels whose
    # surface actually faces this reference's camera may witness the fit.
    if normals_texture is not None:
        normals = np.asarray(normals_texture, dtype=np.float32)[:, :, :3]
        norm = np.linalg.norm(normals, axis=2, keepdims=True)
        unit_normals = np.divide(normals, np.maximum(norm, 1e-8))
        facing = unit_normals @ np.asarray(
            eye / max(float(np.linalg.norm(eye)), 1e-8), dtype=np.float32)
        in_frame = in_frame & (facing > float(min_reference_facing))
    overlap = (source_weight > float(min_source_weight)) & in_frame
    count = int(overlap.sum())
    stats["overlap"] = count
    if count < int(min_overlap):
        return image, stats

    xs = sample_x[overlap]
    ys = sample_y[overlap]
    target = source_rgb[overlap]
    weights = source_weight[overlap]
    center_x, center_y = (width - 1) / 2.0, (height - 1) / 2.0

    def bilinear(xs2: Any, ys2: Any) -> Any:
        x0 = np.clip(np.floor(xs2), 0, width - 1).astype(np.int32)
        y0 = np.clip(np.floor(ys2), 0, height - 1).astype(np.int32)
        x1 = np.clip(x0 + 1, 0, width - 1)
        y1 = np.clip(y0 + 1, 0, height - 1)
        wx = np.clip(xs2 - x0, 0.0, 1.0).astype(np.float32)[:, None]
        wy = np.clip(ys2 - y0, 0.0, 1.0).astype(np.float32)[:, None]
        return (
            array[y0, x0] * (1 - wx) * (1 - wy)
            + array[y0, x1] * wx * (1 - wy)
            + array[y1, x0] * (1 - wx) * wy
            + array[y1, x1] * wx * wy
        )

    def error_of(scale: float, dx: float, dy: float) -> Optional[float]:
        xs2 = (xs - center_x - dx * width) / scale + center_x
        ys2 = (ys - center_y - dy * height) / scale + center_y
        in_bounds = (xs2 >= 0) & (xs2 <= width - 1) & (ys2 >= 0) & (ys2 <= height - 1)
        if int(in_bounds.sum()) < int(min_overlap) // 2:
            return None
        sampled = bilinear(xs2[in_bounds], ys2[in_bounds])
        opaque = sampled[:, 3] > 0.5
        if int(opaque.sum()) < int(min_overlap) // 2:
            return None
        difference = np.abs(sampled[opaque, :3] - target[in_bounds][opaque]).mean(axis=1)
        used_weights = weights[in_bounds][opaque]
        return float((difference * used_weights).sum() / used_weights.sum())

    baseline = error_of(1.0, 0.0, 0.0)
    stats["err_before"] = round(baseline, 4) if baseline is not None else None
    best = (baseline if baseline is not None else np.inf, 1.0, 0.0, 0.0)
    shifts = np.arange(-float(shift_range), float(shift_range) + 1e-9, float(shift_step))
    for scale in scale_candidates:
        for dy in shifts:
            for dx in shifts:
                error = error_of(float(scale), float(dx), float(dy))
                if error is not None and error < best[0]:
                    best = (error, float(scale), float(dx), float(dy))
    error, scale, dx, dy = best
    stats["err_after"] = round(float(error), 4) if np.isfinite(error) else None
    if (
        baseline is None
        or not np.isfinite(error)
        or error > baseline - float(min_improvement)
        or (scale == 1.0 and dx == 0.0 and dy == 0.0)
    ):
        return image, stats

    # SILHOUETTE GUARD: the photometric fit may not undo the silhouette
    # registration that precedes it. A warp that improves overlap-color
    # agreement by sliding texture (grain locking onto grain) drags the
    # matte off the mesh footprint — measured as near-bound shifts (46-61
    # px) on views whose IoU was already 0.93+. Surface coverage = the
    # fraction of THIS camera's facing-gated surface texels whose sample
    # lands on opaque photo; the warp may not lose more than 1% of it.
    surface_texels = in_frame
    if surface_texels.any():
        def coverage_of(scale_c: float, dx_c: float, dy_c: float) -> float:
            xs_all = (sample_x[surface_texels] - center_x - dx_c * width) / scale_c + center_x
            ys_all = (sample_y[surface_texels] - center_y - dy_c * height) / scale_c + center_y
            in_b = (xs_all >= 0) & (xs_all <= width - 1) & (ys_all >= 0) & (ys_all <= height - 1)
            if not in_b.any():
                return 0.0
            sampled_alpha = bilinear(xs_all[in_b], ys_all[in_b])[:, 3]
            return float((sampled_alpha > 0.5).sum()) / float(surface_texels.sum())

        coverage_before = coverage_of(1.0, 0.0, 0.0)
        coverage_after = coverage_of(scale, dx, dy)
        stats["coverage_before"] = round(coverage_before, 4)
        stats["coverage_after"] = round(coverage_after, 4)
        if coverage_after < coverage_before - 0.01:
            stats["rejected"] = "silhouette_coverage_drop"
            return image, stats

    inv_scale = 1.0 / scale
    matrix = (
        inv_scale,
        0.0,
        center_x - inv_scale * (center_x + dx * width),
        0.0,
        inv_scale,
        center_y - inv_scale * (center_y + dy * height),
    )
    warped = _warp_affine_rgba(image, matrix)
    stats.update(
        {
            "applied": True,
            "scale": round(scale, 4),
            "shift_x": round(dx, 4),
            "shift_y": round(dy, 4),
        }
    )
    return warped, stats


def register_view_by_width_profile(
    mesh: Any,
    *,
    observed_rgba: Any,
    azimuth_deg: float,
    elevation_deg: float,
    camera_distance: float,
    projection_model: str = "perspective",
    ortho_half_extent: Optional[float] = None,
    fovy_deg: float = 40.0,
    scale_range: Tuple[float, float] = (0.45, 1.6),
    scale_step: float = 0.02,
) -> Tuple[Any, Dict[str, float]]:
    """Register a photo to the mesh silhouette by width-profile matching.

    Area-IoU and edge-chamfer objectives both fail on photos whose CROP
    differs from the mesh's framing (a head-only profile against a
    head-plus-shoulders silhouette): they either land on a compromise scale
    or lock onto a long edge that aligns at any scale. The subject's TOP is
    almost never cropped, so the silhouette width measured row by row below
    the top is a crop-immune, scale-sensitive shape signature:

        scale* = argmin_s  mean_k | width_mesh(k) - s * width_photo(k / s) |

    evaluated only over rows the photo actually observes (above its crop
    line). Horizontal shift comes from per-row centroid alignment and
    vertical shift from aligning the tops. The warp maps the photo into the
    mesh's frame; a residual local search is the caller's business.
    """
    import numpy as np
    from PIL import Image

    image = observed_rgba.convert("RGBA") if hasattr(observed_rgba, "convert") else observed_rgba
    stats: Dict[str, Any] = {"applied": False, "scale": 1.0, "shift_x": 0.0, "shift_y": 0.0, "profile_error": None}
    array = np.asarray(image, dtype=np.float32) / 255.0
    if array.ndim != 3 or array.shape[2] < 4:
        return image, stats
    photo_mask_full = array[:, :, 3] > 0.5
    if photo_mask_full.mean() < 0.005:
        return image, stats

    size = 192
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    if len(vertices) > 20000:
        vertices = vertices[:: len(vertices) // 20000 + 1]
    mesh_mask = _splat_silhouette(
        vertices,
        azimuth_deg=float(azimuth_deg),
        elevation_deg=float(elevation_deg),
        camera_distance=float(camera_distance),
        size=size,
        fovy_deg=float(fovy_deg),
        projection_model=str(projection_model),
        ortho_half_extent=ortho_half_extent,
    )
    if not mesh_mask.any():
        return image, stats

    # Photo mask in the SAME square frame with height-normalized letterbox.
    full_h, full_w = photo_mask_full.shape
    letter_w = max(1, int(round(size * full_w / full_h)))
    resized = np.asarray(
        Image.fromarray((photo_mask_full * 255).astype(np.uint8)).resize((letter_w, size), Image.BILINEAR)
    ) > 127
    photo_mask = np.zeros((size, size), dtype=bool)
    if letter_w >= size:
        left = (letter_w - size) // 2
        photo_mask = resized[:, left : left + size]
    else:
        left = (size - letter_w) // 2
        photo_mask[:, left : left + letter_w] = resized

    def profile(mask: Any) -> Tuple[int, Any, Any]:
        rows = np.nonzero(mask.any(axis=1))[0]
        top = int(rows[0])
        widths = mask.sum(axis=1).astype(np.float64)
        cols = np.arange(mask.shape[1], dtype=np.float64)
        sums = (mask * cols[None, :]).sum(axis=1)
        counts = np.maximum(mask.sum(axis=1), 1)
        centroids = sums / counts
        return top, widths, centroids

    mesh_top, mesh_widths, mesh_centroids = profile(mesh_mask)
    photo_top, photo_widths, photo_centroids = profile(photo_mask)
    photo_rows = np.nonzero(photo_mask.any(axis=1))[0]
    photo_bottom = int(photo_rows[-1])
    # Rows at the frame border are crop lines; keep a safety margin.
    photo_usable = photo_bottom - photo_top - (2 if photo_bottom >= size - 2 else 0)
    mesh_rows = np.nonzero(mesh_mask.any(axis=1))[0]
    mesh_bottom = int(mesh_rows[-1])
    if photo_usable < 10:
        return image, stats

    best = (np.inf, 1.0)
    for scale in np.arange(float(scale_range[0]), float(scale_range[1]) + 1e-9, float(scale_step)):
        errors = []
        for k in range(0, mesh_bottom - mesh_top):
            j = k / scale
            if j >= photo_usable:
                break
            jf = photo_top + j
            j0 = int(jf)
            frac = jf - j0
            if j0 + 1 > photo_bottom:
                break
            photo_width = (1 - frac) * photo_widths[j0] + frac * photo_widths[min(j0 + 1, size - 1)]
            mesh_width = mesh_widths[mesh_top + k]
            errors.append(abs(mesh_width - scale * photo_width))
        # Require a meaningful comparison span (at least a third of the
        # mesh's height or everything the photo can offer).
        if len(errors) < min(30, photo_usable):
            continue
        error = float(np.mean(errors)) / max(float(mesh_widths.max()), 1.0)
        if error < best[0]:
            best = (error, float(scale))
    if not np.isfinite(best[0]):
        return image, stats
    profile_error, scale = best

    # Vertical shift: photo top row maps onto the mesh top row.
    # Horizontal shift: median centroid offset over the compared rows.
    offsets = []
    for k in range(0, mesh_bottom - mesh_top):
        j = k / scale
        jf = photo_top + j
        j0 = int(jf)
        if j0 > photo_bottom or (photo_bottom >= size - 2 and j0 >= photo_bottom - 2):
            break
        if photo_widths[j0] < 2 or mesh_widths[mesh_top + k] < 2:
            continue
        center = (size - 1) / 2.0
        offsets.append(
            (mesh_centroids[mesh_top + k] - center) - scale * (photo_centroids[j0] - center)
        )
    shift_x_px = float(np.median(offsets)) if offsets else 0.0
    center = (size - 1) / 2.0
    # Solve top alignment: mesh_top = scale*(photo_top - center - ?) ... in
    # the warp model used by the projector-facing transform below,
    # output_row = scale*(input_row - center) + center + shift_y_px.
    shift_y_px = float(mesh_top) - (scale * (photo_top - center) + center)

    stats.update(
        {
            "applied": True,
            "scale": round(scale, 4),
            "shift_x": round(shift_x_px / size, 4),
            "shift_y": round(shift_y_px / size, 4),
            "profile_error": round(profile_error, 4),
        }
    )

    # Apply to the ORIGINAL image: the frame math above used the square
    # height-normalized frame, and the original photo maps into it by
    # (size/full_h) with a horizontal letterbox offset. Compose into a
    # single affine in original-photo pixels: out = s*(in - c) + c + t,
    # with t scaled back by (full_h/size).
    width, height = image.size
    center_x, center_y = (width - 1) / 2.0, (height - 1) / 2.0
    tx_px = shift_x_px * (height / float(size))
    ty_px = shift_y_px * (height / float(size))
    inv_scale = 1.0 / scale
    matrix = (
        inv_scale,
        0.0,
        center_x - inv_scale * (center_x + tx_px),
        0.0,
        inv_scale,
        center_y - inv_scale * (center_y + ty_px),
    )
    warped = _warp_affine_rgba(image, matrix)
    return warped, stats


def _warp_affine_rgba(image: Any, matrix: Tuple[float, ...]) -> Any:
    """Similarity-warp an RGBA image (BILINEAR color and alpha).

    A bicubic color variant was measured and REVERTED: it retains ~5%
    more of the 2-8 px relief band per registration warp (owl back:
    13.66 bilinear vs 14.75 bicubic projected band RMS), but its
    overshoot at strong carved edges raised the whole-bake acceptance
    gate's long-strong-edge statistic by the same magnitude as the
    labeled chair seam regression (+0.03-0.04 absolute), making real
    seams indistinguishable from resampling ring in     the one metric that
    auto-protects unattended users. Until the seam metric measures
    shipped handoff steps directly (see the handoff-seam ledger in
    `blend_projections`), the warp stays bilinear.
    """
    from PIL import Image

    rgba = image.convert("RGBA")
    return rgba.transform(
        rgba.size, Image.AFFINE, matrix,
        resample=Image.BILINEAR, fillcolor=(0, 0, 0, 0))


def estimate_pose_with_silhouette_guard(
    mesh: Any,
    observed_rgba: Any,
    *,
    border_ratio: float = 0.15,
    azimuth_window_deg: float = 40.0,
    size: int = 256,
    elevations: Sequence[float] = (-15.0, -8.0, 0.0, 8.0, 15.0),
    rescue_azimuth_max: float = 50.0,
    azimuth_step_deg: float = 5.0,
    veto_tolerance: float = 0.02,
    rescue_min_riou_gap: float = 0.10,
    rescue_min_aspect_err: float = 0.15,
    rescue_min_best_riou: float = 0.75,
    plateau_band: float = 0.03,
) -> Dict[str, Any]:
    """Source-pose estimation with an independent SILHOUETTE evidence channel.

    The gradient-NCC estimator's validity assumption — photo luminance
    gradients co-located with mesh crease shading — fails on smooth glossy
    high-chroma subjects: their photo gradients are specular streaks that
    correlate ACCIDENTALLY with near-frontal renders. Measured on a 3/4
    sports-car photo: the true pose (az ~35) scored ~0 while an incoherent
    spike at az 5 (neighbors at 16% of peak height; genuine matches
    measure 40%+) passed the height-only gates and collapsed observed
    coverage to 0.055. The same signature moved a certified chair to
    az -27.5. Shape evidence — material-independent — contradicted both
    commits loudly (registered silhouette IoU 0.73 at the commit vs 0.89
    in the true basin) but was never consulted.

    Two additions around the UNCHANGED production NCC lane:

    - VETO: an accepted NCC pose must not WORSEN the registered
      silhouette IoU vs the declared pose by more than `veto_tolerance`;
      the true pose renders the geometry the photo shows, so a commit
      that degrades the silhouette fit is a photometric spike.
    - RESCUE: entered only when the NCC lane did not commit (own gates or
      veto). Moves the pose only on DOUBLE-KEYED decisive shape evidence
      (registered-IoU gap > `rescue_min_riou_gap` AND declared-pose
      aspect error > `rescue_min_aspect_err` AND best IoU above the
      quality floor). Pose selection folds the bilateral mirror with the
      production anti-symmetric chirality scorer and resolves the az/el
      aspect-trade ridge by boundary chamfer (area-IoU is measurably
      degenerate along it).

    Calibration (movers must fire, stayers must not — measured):
    car riou_gap 0.156 / aspect 0.232 fires; starship 0.290/0.264 fires;
    owl 0.000/0.002, chair 0.064/0.076, face 0.054/0.131 stay. The weak-
    evidence contract is preserved: below the double-keyed gate the
    declared pose ships exactly as before.
    """
    import math

    import numpy as np
    from PIL import Image
    from scipy.ndimage import binary_erosion, distance_transform_edt

    from .rendering import render_mesh_views

    ncc = estimate_pose_photometric(
        mesh, observed_rgba, border_ratio=border_ratio,
        azimuth_window_deg=azimuth_window_deg)
    result_trail: Dict[str, Any] = {}

    def render_mask(azimuth: float, elevation: float) -> Any:
        rendered = render_mesh_views(
            mesh, azimuths=(float(azimuth),), elevation=float(elevation),
            size=size, lighting="fixed")[0].convert("RGBA")
        array = np.asarray(rendered, dtype=np.float32) / 255.0
        background = array[2, 2, :3]
        gray = array[:, :, :3].mean(axis=2)
        return gray, np.abs(array[:, :, :3] - background).sum(axis=2) > 0.08

    def bbox_aspect(mask: Any) -> Optional[float]:
        rows = np.nonzero(mask.any(axis=1))[0]
        cols = np.nonzero(mask.any(axis=0))[0]
        if not len(rows) or not len(cols):
            return None
        return float(cols[-1] - cols[0] + 1) / max(float(rows[-1] - rows[0] + 1), 1.0)

    def register_mask(photo_mask: Any, mask: Any, *, fine: bool = False) -> Tuple[float, Optional[Any]]:
        pr, pc = np.nonzero(photo_mask)
        rr, rc = np.nonzero(mask)
        if not len(pr) or not len(rr):
            return 0.0, None
        p_cx = pc.mean()
        p_cy = pr.mean()
        r_cx = rc.mean()
        r_cy = rr.mean()
        base_scale = math.sqrt(len(pr) / len(rr))
        source = Image.fromarray((mask * 255).astype(np.uint8))
        if fine:
            scales = [base_scale * f for f in np.arange(0.90, 1.101, 0.02)]
            shifts = range(-10, 11, 2)
        else:
            scales = [base_scale * f for f in (0.92, 0.96, 1.0, 1.04, 1.08)]
            shifts = range(-8, 9, 4)
        best = (0.0, None)
        for scale in scales:
            inv = 1.0 / scale
            for dy in shifts:
                for dx in shifts:
                    matrix = (inv, 0.0, r_cx - inv * (p_cx + dx),
                              0.0, inv, r_cy - inv * (p_cy + dy))
                    warped = np.asarray(
                        source.transform((size, size), Image.AFFINE, matrix,
                                         resample=Image.NEAREST, fillcolor=0)) > 127
                    union = int((photo_mask | warped).sum())
                    if union == 0:
                        continue
                    iou = float((photo_mask & warped).sum()) / union
                    if iou > best[0]:
                        best = (iou, warped)
        return best

    def chamfer(a_mask: Any, b_mask: Optional[Any]) -> float:
        if b_mask is None:
            return float("inf")
        a_boundary = a_mask & ~binary_erosion(a_mask)
        b_boundary = b_mask & ~binary_erosion(b_mask)
        if not a_boundary.any() or not b_boundary.any():
            return float("inf")
        da = distance_transform_edt(~a_boundary)
        db = distance_transform_edt(~b_boundary)
        return float(0.5 * (da[b_boundary].mean() + db[a_boundary].mean()))

    image = observed_rgba.convert("RGBA") if hasattr(observed_rgba, "convert") else observed_rgba
    canonical = recenter_to_canonical_frame(image, border_ratio=border_ratio)
    photo = np.asarray(canonical.resize((size, size), Image.BILINEAR), dtype=np.float32) / 255.0
    photo_alpha = photo[:, :, 3] > 0.5
    if photo_alpha.mean() < 0.02:
        return dict(ncc)
    photo_gray = photo[:, :, :3].mean(axis=2)
    photo_aspect = bbox_aspect(photo_alpha)

    azimuths = [float(a) for a in np.arange(
        -float(rescue_azimuth_max), float(rescue_azimuth_max) + 1e-9,
        float(azimuth_step_deg))]
    riou: Dict[Tuple[float, float], float] = {}
    masks: Dict[Tuple[float, float], Any] = {}
    aspects: Dict[Tuple[float, float], Optional[float]] = {}
    for elevation in elevations:
        for azimuth in azimuths:
            _, mask = render_mask(azimuth, elevation)
            masks[(azimuth, elevation)] = mask
            aspects[(azimuth, elevation)] = bbox_aspect(mask)
            riou[(azimuth, elevation)], _ = register_mask(photo_alpha, mask)

    declared_key = (0.0, 0.0)
    declared_riou = riou.get(declared_key, 0.0)
    declared_aspect = aspects.get(declared_key)
    declared_aspect_err = (
        abs(math.log(photo_aspect / declared_aspect))
        if photo_aspect and declared_aspect else 0.0
    )
    best_pose = max(riou, key=lambda k: riou[k])
    best_riou = riou[best_pose]
    result_trail["silhouette"] = {
        "declared_riou": round(declared_riou, 4),
        "declared_aspect_err": round(declared_aspect_err, 4),
        "best_riou": round(best_riou, 4),
        "best_pose": [best_pose[0], best_pose[1]],
    }

    if ncc.get("estimated"):
        commit_key = (float(ncc["azimuth_deg"]), float(ncc["elevation_deg"]))
        commit_mask = masks.get(commit_key)
        if commit_mask is None:
            _, commit_mask = render_mask(*commit_key)
        committed_riou = riou.get(commit_key)
        if committed_riou is None:
            committed_riou, _ = register_mask(photo_alpha, commit_mask)
        vetoed = committed_riou < declared_riou - float(veto_tolerance)
        # Shape-decisive override (v4 car): a photometric spike can land
        # BETWEEN the declared pose and the truth — commit az 12.5 scored
        # riou 0.787, above declared 0.746, so the worse-than-declared
        # veto passed it while the true basin at az -25 measured 0.908
        # and coverage collapsed to 5.2% again. The override fires only
        # on the rescue lane's double-keyed decisive evidence, with the
        # aspect key measured at the COMMIT (a correct commit renders the
        # photo's aspect, so its own aspect error stays small and blocks
        # the override even if some far pose spuriously out-scores it).
        commit_aspect = bbox_aspect(commit_mask)
        commit_aspect_err = (
            abs(math.log(photo_aspect / commit_aspect))
            if photo_aspect and commit_aspect else 0.0
        )
        shape_override = (
            (best_riou - float(committed_riou)) > float(rescue_min_riou_gap)
            and commit_aspect_err > float(rescue_min_aspect_err)
            and best_riou > float(rescue_min_best_riou)
        )
        vetoed = vetoed or shape_override
        result_trail["veto"] = {
            "committed_riou": round(float(committed_riou), 4),
            "delta_vs_declared": round(float(committed_riou - declared_riou), 4),
            "commit_aspect_err": round(commit_aspect_err, 4),
            "shape_override": bool(shape_override),
            "fires": bool(vetoed),
        }
        if not vetoed:
            accepted = dict(ncc)
            accepted["method"] = "gradient_ncc"
            accepted["guard_trail"] = result_trail
            return accepted

    # Second key for the rescue: the declared pose's own aspect error, OR
    # a shape-decisive override veto upstream. The dead zone this closes
    # (measured, integrator program): a fresh 3/4 car draw whose NCC
    # commit was override-vetoed (commit riou 0.773 vs basin best 0.896,
    # commit aspect err 0.159) then failed the rescue's declared-aspect
    # key (0.113 < 0.15) — strong enough shape evidence to REJECT the
    # commit, then judged not strong enough to MOVE, so the bake ran at
    # declared (0,0): coverage 0.0574 vs 0.1877 at the basin pose,
    # fidelity 32.7 vs 21.6 dE (A/B in /tmp/fix3/car_pose_ab.log). The
    # override veto is itself double-keyed decisive evidence (riou gap
    # at the commit + aspect error at the commit), so counting it as the
    # rescue's second key preserves the double-key doctrine. Stayers
    # measured unmoved (their gap key alone keeps them): owl 0.000,
    # face 0.054, chair 0.064 — the chair's override veto fires but its
    # basin gap (0.064 < 0.10) still refuses the move.
    override_vetoed = bool(
        (result_trail.get("veto") or {}).get("shape_override")
        and (result_trail.get("veto") or {}).get("fires"))
    gate_ok = (
        (best_riou - declared_riou) > float(rescue_min_riou_gap)
        and (declared_aspect_err > float(rescue_min_aspect_err)
             or override_vetoed)
        and best_riou > float(rescue_min_best_riou)
    )
    result_trail["rescue_gate"] = {
        "riou_gap": round(best_riou - declared_riou, 4),
        "override_vetoed_key": override_vetoed,
        "fires": bool(gate_ok),
    }
    if not gate_ok:
        return {
            "azimuth_deg": 0.0,
            "elevation_deg": 0.0,
            "estimated": False,
            "score": ncc.get("score"),
            "score_at_declared": ncc.get("score_at_declared"),
            "method": "declared",
            "rejected_reason": (
                "ncc_vetoed_by_silhouette" if ncc.get("estimated")
                else ncc.get("rejected_reason")
            ),
            "guard_trail": result_trail,
        }

    # Rescue pose selection: mirror fold via anti-symmetric chirality.
    az_star, el_star = best_pose
    sign = 1.0
    if abs(az_star) > 1e-6:
        def antisymmetric(gray: Any, mask: Any) -> Tuple[Any, Any]:
            flipped = gray[:, ::-1]
            flipped_mask = mask[:, ::-1]
            both = mask & flipped_mask
            anti = np.where(both, 0.5 * (gray - flipped), 0.0)
            rms = float(np.sqrt(np.mean(anti[both] ** 2))) if both.any() else 0.0
            return anti / max(rms, 1e-8), both

        photo_anti, photo_anti_mask = antisymmetric(photo_gray, photo_alpha)

        def chirality(azimuth: float, elevation: float) -> float:
            gray_r, mask_r = render_mask(azimuth, elevation)
            _, warped = register_mask(photo_alpha, mask_r)
            if warped is None:
                return float("-inf")
            # Approximate anchored frame: use the registered warp of the
            # gray render (chirality only needs the SIGN of correlation).
            gray_image = Image.fromarray((gray_r * 255).astype(np.uint8))
            anti_r, anti_mask_r = antisymmetric(
                np.asarray(gray_image, dtype=np.float32) / 255.0, mask_r)
            both = photo_anti_mask & anti_mask_r
            if int(both.sum()) < 500:
                return float("-inf")
            return float((photo_anti[both] * anti_r[both]).mean())

        chir_pos = chirality(abs(az_star), el_star)
        chir_neg = chirality(-abs(az_star), el_star)
        result_trail["chirality"] = {
            "pos": round(chir_pos, 4), "neg": round(chir_neg, 4)}
        sign = 1.0 if chir_pos >= chir_neg else -1.0

    signed = {pose: value for pose, value in riou.items()
              if pose[0] * sign >= 0.0}
    best_signed = max(signed.values())
    plateau = [pose for pose, value in signed.items()
               if value >= best_signed - float(plateau_band)]
    chamfers: Dict[Tuple[float, float], float] = {}
    for pose in plateau:
        _, warped = register_mask(photo_alpha, masks[pose], fine=True)
        chamfers[pose] = chamfer(photo_alpha, warped)
    final_pose = min(chamfers, key=lambda k: chamfers[k])
    result_trail["rescue_selection"] = {
        "sign": sign,
        "plateau_chamfers": {
            f"{p[0]:.0f},{p[1]:.0f}": round(chamfers[p], 2) for p in plateau},
    }
    return {
        "azimuth_deg": float(final_pose[0]),
        "elevation_deg": float(final_pose[1]),
        "estimated": True,
        "score": round(float(riou[final_pose]), 4),
        "score_at_declared": round(float(declared_riou), 4),
        "method": "silhouette_rescue",
        "guard_trail": result_trail,
    }


def register_view_2d(
    mesh: Any,
    *,
    observed_rgba: Any,
    azimuth_deg: float,
    elevation_deg: float,
    camera_distance: float,
    scale_candidates: Optional[Sequence[float]] = None,
    shift_range: float = 0.18,
    shift_step: float = 0.03,
    projection_model: str = "perspective",
    ortho_half_extent: Optional[float] = None,
) -> Tuple[Any, Dict[str, float]]:
    """Align the observed photo to the mesh's projected silhouette in 2D.

    Even with the right camera pose and distance, the photo and the virtual
    view disagree by a similarity transform: the photo's subject is framed by
    its own crop (and may be truncated), while the mesh is centered at the
    world origin. Projecting without correcting this paints features at the
    wrong place (eyes above eye sockets, hair over the face).

    This fits scale and translation by maximizing silhouette IoU between the
    photo's alpha mask and a splatted mesh silhouette over a coarse grid with
    local refinement, then warps the photo (and its alpha) with that
    similarity transform so the projector can keep its fixed pinhole model.
    Returns `(warped_rgba_image, stats)`; when no useful alignment is found
    the original image is returned with `applied=False`.
    """
    import numpy as np
    from PIL import Image

    image = observed_rgba.convert("RGBA") if hasattr(observed_rgba, "convert") else observed_rgba
    array = np.asarray(image, dtype=np.float32) / 255.0
    stats = {"applied": False, "scale": 1.0, "shift_x": 0.0, "shift_y": 0.0, "iou_before": 0.0, "iou_after": 0.0}
    if array.ndim != 3 or array.shape[2] < 4:
        return image, stats
    alpha = array[:, :, 3]
    photo_mask_full = alpha > 0.5
    if photo_mask_full.mean() < 0.005:
        return image, stats

    size = 96
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    if len(vertices) > 15000:
        vertices = vertices[:: len(vertices) // 15000 + 1]
    mesh_mask = _splat_silhouette(
        vertices,
        azimuth_deg=float(azimuth_deg),
        elevation_deg=float(elevation_deg),
        camera_distance=float(camera_distance),
        size=size,
        projection_model=str(projection_model),
        ortho_half_extent=ortho_half_extent,
    )
    if not mesh_mask.any():
        return image, stats

    # Aspect-preserving letterbox into the square comparison frame: an
    # anisotropic resize turns circles into ellipses that no isotropic
    # scale/shift can reconcile, forcing spurious warps on correctly framed
    # photos. Both frame axes are normalized by HEIGHT (matching the
    # projector's NDC convention and the shift back-conversion below).
    full_h, full_w = photo_mask_full.shape
    letter_w = max(1, int(round(size * full_w / full_h)))
    resized = np.asarray(
        Image.fromarray((photo_mask_full * 255).astype(np.uint8)).resize(
            (letter_w, size), Image.BILINEAR
        )
    ) > 127
    photo_mask = np.zeros((size, size), dtype=bool)
    if letter_w >= size:
        left_crop = (letter_w - size) // 2
        photo_mask = resized[:, left_crop : left_crop + size]
    else:
        left_pad = (size - letter_w) // 2
        photo_mask[:, left_pad : left_pad + letter_w] = resized

    # Photos are frequently CROPPED at frame edges (a portrait cut at the
    # chest), while the mesh silhouette is complete. Rows/columns of the
    # photo mask that touch the frame border mark crop lines, not shape
    # boundaries; beyond a transformed crop line the mesh silhouette is
    # simply unobserved and must not count as mismatch. Without this, a
    # head-only photo can never align to a head-plus-shoulders silhouette:
    # the fit lands on a compromise scale that misplaces every feature.
    photo_rows = np.nonzero(photo_mask.any(axis=1))[0]
    photo_cols = np.nonzero(photo_mask.any(axis=0))[0]
    cropped_bottom = bool(len(photo_rows)) and photo_rows[-1] >= size - 2
    cropped_top = bool(len(photo_rows)) and photo_rows[0] <= 1
    cropped_left = bool(len(photo_cols)) and photo_cols[0] <= 1
    cropped_right = bool(len(photo_cols)) and photo_cols[-1] >= size - 2

    coords_y, coords_x = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    center = (size - 1) / 2.0

    def transformed_mask(scale: float, shift_x: float, shift_y: float) -> tuple:
        # Warp the photo mask about the frame center by (scale, shift);
        # also return the in-frame validity so crop lines track the warp.
        src_x = (coords_x - center - shift_x * size) / scale + center
        src_y = (coords_y - center - shift_y * size) / scale + center
        valid = (src_x >= 0) & (src_x < size) & (src_y >= 0) & (src_y < size)
        out = np.zeros((size, size), dtype=bool)
        sx = np.clip(src_x.astype(np.int32), 0, size - 1)
        sy = np.clip(src_y.astype(np.int32), 0, size - 1)
        out[valid] = photo_mask[sy[valid], sx[valid]]
        observable = valid.copy()
        if cropped_bottom:
            observable &= src_y <= float(photo_rows[-1])
        if cropped_top:
            observable &= src_y >= float(photo_rows[0])
        if cropped_left:
            observable &= src_x >= float(photo_cols[0])
        if cropped_right:
            observable &= src_x <= float(photo_cols[-1])
        return out, observable

    try:
        from scipy.ndimage import binary_erosion, distance_transform_edt
    except Exception:
        binary_erosion = None
        distance_transform_edt = None

    def edges_of(mask: Any) -> Any:
        if binary_erosion is None:
            shifted = np.zeros_like(mask)
            shifted[1:, 1:] = mask[:-1, :-1]
            return mask & ~shifted
        return mask & ~binary_erosion(mask, iterations=1, border_value=False)

    mesh_edges = edges_of(mesh_mask)
    mesh_edge_distance = (
        distance_transform_edt(~mesh_edges) if distance_transform_edt is not None else None
    )

    def score_of(mask: Any, observable: Any) -> float:
        """Symmetric edge-chamfer score (higher is better).

        Region IoU rewards degenerate solutions on cropped photos: blowing
        the photo up until only its largest blob stays in frame maximizes
        in-frame overlap while the mismatch conveniently leaves the frame.
        Edge chamfer does not: silhouette EDGES must coincide, and
        enlarging scatters photo edges into the mesh interior where they
        pay full distance penalties. Crop lines stay excluded from both
        edge sets (a crop boundary is not a shape edge).
        """
        if mesh_edge_distance is None:
            mesh_visible = mesh_mask & observable
            intersection = float(np.logical_and(mask, mesh_visible).sum())
            union = float(np.logical_or(mask, mesh_visible).sum())
            return intersection / union if union > 0 else 0.0
        photo_edges = edges_of(mask) & observable
        if not photo_edges.any():
            return -1e9
        forward = float(mesh_edge_distance[photo_edges].mean())
        photo_edge_distance = distance_transform_edt(~photo_edges)
        mesh_edges_observable = mesh_edges & observable
        if not mesh_edges_observable.any():
            return -1e9
        backward = float(photo_edge_distance[mesh_edges_observable].mean())
        return -(forward + backward) / 2.0

    def iou_of(mask: Any, observable: Any) -> float:
        mesh_visible = mesh_mask & observable
        intersection = float(np.logical_and(mask, mesh_visible).sum())
        union = float(np.logical_or(mask, mesh_visible).sum())
        return intersection / union if union > 0 else 0.0

    baseline_mask, baseline_observable = transformed_mask(1.0, 0.0, 0.0)
    baseline_score = score_of(baseline_mask, baseline_observable)
    baseline_iou = iou_of(baseline_mask, baseline_observable)
    best = (baseline_score, 1.0, 0.0, 0.0)
    scales = list(scale_candidates or np.arange(0.7, 1.45, 0.05))
    shifts = np.arange(-float(shift_range), float(shift_range) + 1e-9, float(shift_step))
    for scale in scales:
        for shift_y in shifts:
            for shift_x in shifts:
                warped, observable = transformed_mask(float(scale), float(shift_x), float(shift_y))
                candidate = score_of(warped, observable)
                if candidate > best[0]:
                    best = (candidate, float(scale), float(shift_x), float(shift_y))

    best_score, scale, shift_x, shift_y = best
    chosen_mask, chosen_observable = transformed_mask(scale, shift_x, shift_y)
    best_iou = iou_of(chosen_mask, chosen_observable)
    stats.update(
        {
            "iou_before": round(baseline_iou, 4),
            "iou_after": round(best_iou, 4),
            "score_before": round(float(baseline_score), 4),
            "score_after": round(float(best_score), 4),
        }
    )
    # Require a real improvement before warping; tiny gains are grid noise.
    # Chamfer scores are negative mean edge distances in 96-grid pixels; a
    # quarter-pixel mean improvement is a real alignment change.
    if best_score < baseline_score + 0.25 or (scale == 1.0 and shift_x == 0.0 and shift_y == 0.0):
        return image, stats

    width, height = image.size
    center_x, center_y = (width - 1) / 2.0, (height - 1) / 2.0
    # PIL affine takes the inverse map (output -> input coordinates).
    inv_scale = 1.0 / scale
    # The fit frame is height-normalized on BOTH axes (aspect-preserving
    # letterbox), so both shifts are fractions of the photo HEIGHT; using
    # width here displaced features by (W/H - 1) * shift on non-square
    # photos.
    matrix = (
        inv_scale,
        0.0,
        center_x - inv_scale * (center_x + shift_x * height),
        0.0,
        inv_scale,
        center_y - inv_scale * (center_y + shift_y * height),
    )
    warped = _warp_affine_rgba(image, matrix)
    stats.update({"applied": True, "scale": round(scale, 4), "shift_x": round(shift_x, 4), "shift_y": round(shift_y, 4)})
    return warped, stats


def refine_registration_photometric(
    mesh: Any,
    *,
    observed_rgba: Any,
    azimuth_deg: float,
    elevation_deg: float,
    camera_distance: float,
    fovy_deg: float = 40.0,
    size: int = 160,
    scale_deltas: Sequence[float] = (-0.06, -0.03, 0.0, 0.03, 0.06),
    shift_deltas: Sequence[float] = (-0.06, -0.045, -0.03, -0.015, 0.0, 0.015, 0.03, 0.045, 0.06),
) -> Tuple[Any, Dict[str, float]]:
    """Refine 2D photo-to-mesh registration using geometric edges.

    Silhouette IoU is blind to interior misalignment: on a face, a 4% vertical
    offset barely changes the outline but paints eyes onto eyelids. Interior
    geometry edges are visible in both signals we do have: depth discontinuities
    and curvature of the rendered mesh, and luminance edges of the photo (eye
    sockets, lips, and nostrils produce both). This renders the mesh depth map
    at the view pose, extracts gradient-magnitude edges from it and from the
    photo, and maximizes their normalized cross-correlation over a small
    similarity grid around identity. The refinement is accepted only when the
    correlation improves over the unshifted baseline.

    Requires the GL depth renderer; on hosts without it the input is returned
    unchanged (silhouette registration still applies).
    """
    import numpy as np
    from PIL import Image

    from .backends.triposr_runtime import _tripo_render_camera_depth_map

    stats = {"applied": False, "scale": 1.0, "shift_x": 0.0, "shift_y": 0.0, "ncc_before": 0.0, "ncc_after": 0.0}
    image = observed_rgba.convert("RGBA") if hasattr(observed_rgba, "convert") else observed_rgba
    depth_render = _tripo_render_camera_depth_map(
        mesh,
        width=int(size),
        height=int(size),
        azimuth_deg=float(azimuth_deg),
        elevation_deg=float(elevation_deg),
        camera_distance=float(camera_distance),
        fovy_deg=float(fovy_deg),
    )
    if depth_render is None:
        return image, stats
    depth_map, _far = depth_render

    def edge_map(gray: Any) -> Any:
        gy, gx = np.gradient(np.asarray(gray, dtype=np.float32))
        magnitude = np.hypot(gx, gy)
        p99 = float(np.percentile(magnitude, 99.0)) or 1.0
        return np.clip(magnitude / max(p99, 1e-8), 0.0, 1.0)

    photo = np.asarray(image.resize((size, size), Image.BILINEAR), dtype=np.float32) / 255.0
    photo_alpha = photo[:, :, 3]
    photo_gray = photo[:, :, :3].mean(axis=2) * photo_alpha
    photo_edges = edge_map(photo_gray)
    mesh_edges = edge_map(depth_map)

    # The outline dominates both edge maps and is already aligned by the
    # silhouette registration, so it drowns out the interior features this
    # refinement exists for. Mask a dilated band around both silhouettes.
    try:
        from scipy.ndimage import binary_dilation

        mesh_fg = depth_map > 0.0
        photo_fg = photo_alpha > 0.5
        outline = np.zeros_like(mesh_fg)
        for mask in (mesh_fg, photo_fg):
            boundary = mask ^ binary_dilation(mask, iterations=1)
            outline |= binary_dilation(boundary, iterations=max(2, size // 48))
        interior = mesh_fg & photo_fg & ~outline
        photo_edges = np.where(interior, photo_edges, 0.0)
        mesh_edges = np.where(interior, mesh_edges, 0.0)
    except Exception:
        pass

    coords_y, coords_x = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    center = (size - 1) / 2.0

    def ncc(scale: float, shift_x: float, shift_y: float) -> float:
        src_x = (coords_x - center - shift_x * size) / scale + center
        src_y = (coords_y - center - shift_y * size) / scale + center
        valid = (src_x >= 0) & (src_x < size) & (src_y >= 0) & (src_y < size)
        warped = np.zeros((size, size), dtype=np.float32)
        sx = np.clip(src_x.astype(np.int32), 0, size - 1)
        sy = np.clip(src_y.astype(np.int32), 0, size - 1)
        warped[valid] = photo_edges[sy[valid], sx[valid]]
        a = warped - warped.mean()
        b = mesh_edges - mesh_edges.mean()
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        return float((a * b).sum() / denom) if denom > 1e-8 else 0.0

    baseline = ncc(1.0, 0.0, 0.0)
    best = (baseline, 1.0, 0.0, 0.0)
    for scale_delta in scale_deltas:
        for shift_y in shift_deltas:
            for shift_x in shift_deltas:
                scale = 1.0 + float(scale_delta)
                score = ncc(scale, float(shift_x), float(shift_y))
                if score > best[0]:
                    best = (score, scale, float(shift_x), float(shift_y))

    score, scale, shift_x, shift_y = best
    stats.update({"ncc_before": round(baseline, 4), "ncc_after": round(score, 4)})
    if score < baseline + 0.01 or (scale == 1.0 and shift_x == 0.0 and shift_y == 0.0):
        return image, stats

    width, height = image.size
    center_x, center_y = (width - 1) / 2.0, (height - 1) / 2.0
    inv_scale = 1.0 / scale
    # Shifts were fitted in a square frame scaled by the photo height, so
    # both convert back through HEIGHT (width overshoots on landscape
    # photos).
    matrix = (
        inv_scale,
        0.0,
        center_x - inv_scale * (center_x + shift_x * height),
        0.0,
        inv_scale,
        center_y - inv_scale * (center_y + shift_y * height),
    )
    warped = _warp_affine_rgba(image, matrix)
    stats.update({"applied": True, "scale": round(scale, 4), "shift_x": round(shift_x, 4), "shift_y": round(shift_y, 4)})
    return warped, stats


def mesh_mirror_symmetry_score(mesh: Any, *, axis: int = 1, samples: int = 4000) -> float:
    """Measure how mirror-symmetric a mesh is across a world axis plane.

    Samples surface vertices, reflects them across the plane, and measures
    the nearest-vertex distance of the reflections, normalized by the mesh
    scale. Returns a score in [0, 1] where 1 is perfectly symmetric. Used to
    gate mirror-based texture completion: applying it to a mesh whose
    symmetry plane does not match produces doubled, offset features, which
    is worse than leaving hidden texels to the fill.
    """
    import numpy as np

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    if len(vertices) < 16:
        return 0.0
    rng = np.random.default_rng(11)
    picks = rng.choice(len(vertices), size=min(int(samples), len(vertices)), replace=False)
    sample = vertices[picks]
    mirrored = sample.copy()
    mirrored[:, axis] *= -1.0
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(vertices)
        distances, _ = tree.query(mirrored, k=1)
    except Exception:
        # Fallback without scipy.spatial: chunked brute force on a subsample.
        reference = vertices[rng.choice(len(vertices), size=min(4000, len(vertices)), replace=False)]
        distances = np.empty(len(mirrored), dtype=np.float32)
        for start in range(0, len(mirrored), 512):
            chunk = mirrored[start : start + 512]
            deltas = np.linalg.norm(chunk[:, None, :] - reference[None, :, :], axis=2)
            distances[start : start + 512] = deltas.min(axis=1)
    scale = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0))) or 1.0
    normalized = np.asarray(distances) / scale
    # Median distance of 1% of the diagonal maps to ~0.9; 5% to ~0.5.
    median = float(np.median(normalized))
    return float(np.clip(1.0 - median / 0.10, 0.0, 1.0))


def mirror_fill_from_observed(
    *,
    positions_texture: Any,
    observed_mask: Any,
    colors_rgb: Any,
    axis: int = 1,
    max_distance_ratio: float = 0.02,
    observed_weight: Optional[Any] = None,
    min_source_weight: float = 0.15,
    consensus_guard: bool = True,
    consensus_radius_ratio: float = 0.03,
    consensus_max_spread: float = 0.09,
    consensus_contrast: float = 0.22,
) -> Tuple[Any, Any]:
    """Fill unobserved texels from their observed mirror twins in 3D.

    For each unseen texel with world position p, look up the observed texel
    nearest to mirror(p) across the symmetry plane and take its blended
    color. Unlike re-projecting the photo through a mirrored camera, this
    formulation cannot double features on the visible side: it only writes
    texels the views never covered, and only when a genuine observed twin
    exists within `max_distance_ratio` of the mesh scale.

    When `observed_weight` is provided, only CONFIDENT texels may act as
    mirror sources (`min_source_weight`). The gate exists because grazing
    hairline/forehead rim samples fabricated a bright skin patch on the
    hidden crown when allowed as sources. 0.15 balances that against
    coverage: an ablation showed 0.35 disqualified 81% of observed texels
    and shrank mirror completion from +0.42 to +0.09 of the atlas with no
    visible quality gain over 0.15 (the crown patch is gone at both).

    `consensus_guard`: geometry is never perfectly symmetric (score 0.966
    on the face lane), so a twin lookup near a material boundary can land
    JUST ACROSS it — copying hairline hair onto observed-adjacent cheek
    skin (measured: half the dark defect pixels on the left cheek at close
    zoom were such copies). A copy is rejected only when it contradicts a
    CONFIDENT local consensus: the destination's observed 3D neighborhood
    is color-consistent (spread below `consensus_max_spread`) AND the
    copied color deviates from its mean by more than `consensus_contrast`.
    Feature-rich destinations (high spread: eye/lip boundaries) accept
    copies unconditionally, so legitimate feature completion is unaffected;
    rejected texels fall through to the harmonic fill.

    Returns `(fill_rgb, fill_mask)`.
    """
    import numpy as np

    positions = np.asarray(positions_texture, dtype=np.float32)
    observed = np.asarray(observed_mask, dtype=bool)
    colors = np.asarray(colors_rgb, dtype=np.float32)
    surface = positions[:, :, 3] > 0.0
    unseen = surface & ~observed
    fill_rgb = np.zeros_like(colors)
    fill_mask = np.zeros(observed.shape, dtype=bool)

    source_mask = observed
    if observed_weight is not None:
        weights = np.asarray(observed_weight, dtype=np.float32)
        if weights.shape == observed.shape:
            confident = observed & (weights >= float(min_source_weight))
            # Graceful degradation, never a cliff: the old behavior fell
            # back to ALL observed texels when the confident set was under
            # a hard 500-texel floor — abandoning the gate exactly when
            # confidence was scarcest (measured: a tiny upstream weight
            # rescale moved the count 570 -> 430 across the floor and
            # mirror coverage jumped 9x with no code change). Now: use the
            # confident set whatever its size; if it is thin, top up with
            # the BEST-weighted remaining observed texels (never texels an
            # order of magnitude below threshold) until a minimal anchor
            # population is reached, so fill extent varies O(1) with O(1)
            # source changes.
            minimum_anchor = 500
            if int(confident.sum()) >= minimum_anchor:
                source_mask = confident
            else:
                deficit = minimum_anchor - int(confident.sum())
                remaining = observed & ~confident & (
                    weights >= 0.5 * float(min_source_weight)
                )
                if remaining.any():
                    remaining_weights = np.where(remaining, weights, -1.0)
                    flat = remaining_weights.ravel()
                    take = min(int(remaining.sum()), deficit)
                    if take > 0:
                        top_indices = np.argpartition(flat, -take)[-take:]
                        topped = np.zeros(flat.shape, dtype=bool)
                        topped[top_indices] = flat[top_indices] > 0
                        confident = confident | topped.reshape(confident.shape)
                # With no credible anchors at all, produce NO fill rather
                # than fabricate from junk-weight sources.
                source_mask = confident
    if not unseen.any() or not source_mask.any():
        return fill_rgb, fill_mask

    observed_positions = positions[:, :, :3][source_mask]
    observed_colors = colors[source_mask]
    unseen_positions = positions[:, :, :3][unseen].copy()
    unseen_positions[:, axis] *= -1.0

    scale = float(np.linalg.norm(observed_positions.max(axis=0) - observed_positions.min(axis=0))) or 1.0
    threshold = float(max_distance_ratio) * scale
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(observed_positions)
        # Two pure-search optimizations, both output-identical by
        # construction (verified bitwise on the golden assets):
        # - workers=-1 parallelizes over query points; exact NN per point.
        # - distance_upper_bound prunes the search at the acceptance
        #   threshold: most mirror twins land nowhere near an observed
        #   texel (measured 1.6% acceptance on the owl), and unbounded
        #   exact-NN backtracking across the whole tree dominated the
        #   stage (140 s of a 167 s stage). Queries beyond the bound
        #   return (inf, n) and are dropped by the same `valid` mask that
        #   already applied the threshold; the small slack keeps the
        #   exact `<= threshold` cut with the mask, not the prune.
        distances, indices = tree.query(
            unseen_positions, k=1, workers=-1,
            distance_upper_bound=threshold * 1.001,
        )
    except Exception:
        return fill_rgb, fill_mask

    valid = np.asarray(distances) <= threshold
    unseen_rows, unseen_cols = np.nonzero(unseen)
    rows = unseen_rows[valid]
    cols = unseen_cols[valid]
    candidate_colors = observed_colors[np.asarray(indices)[valid]]

    if consensus_guard and len(rows) > 0:
        destination_points = positions[:, :, :3][rows, cols]
        neighbor_lists = tree.query_ball_point(
            destination_points, r=float(consensus_radius_ratio) * scale, workers=-1
        )
        keep = np.ones(len(rows), dtype=bool)
        for i, neighbors in enumerate(neighbor_lists):
            # Too few observed neighbors = no consensus to contradict.
            if len(neighbors) < 24:
                continue
            local = observed_colors[neighbors]
            local_mean = local.mean(axis=0)
            if float(np.abs(local - local_mean).mean()) > float(consensus_max_spread):
                continue
            if float(np.abs(candidate_colors[i] - local_mean).mean()) > float(consensus_contrast):
                keep[i] = False
        rows, cols, candidate_colors = rows[keep], cols[keep], candidate_colors[keep]

    fill_rgb[rows, cols] = candidate_colors
    fill_mask[rows, cols] = True
    return fill_rgb, fill_mask


def tone_match_completion_components(
    fill_rgb: Any,
    *,
    add_mask: Any,
    colors_rgb: Any,
    observed_mask: Any,
    surface_mask: Any,
    gain_clamp: float = 0.25,
    ring_bright_ratio: float = 0.72,
    copy_bright_ratio: float = 0.60,
    min_component_texels: int = 8,
) -> Tuple[Any, int]:
    """Tone-match mirror-completion components to their destination ring
    (cycle 6, FACE-22 mirror-copy border contours).

    THE DEFECT: mirror completion copies the twin's colors VERBATIM; on a
    lighting-asymmetric subject the twin carries its own baked shading, so
    each filled component lands at a tone offset from the observed skin
    around it (measured on the face proof: chest copies at +16/255 over
    their destination ring) and its border prints as a closed line-art
    contour. The legacy compositor's seam leveling owned exactly this
    reconciliation (mirror regions participate in `level_composed_seams`),
    but the gradient-domain path runs its solve BEFORE mirror completion,
    so nothing reconciles completion tone there — the missing handoff is
    this function.

    SCOPE (each restriction measured in the cycle-6 ladder):
    - PURE-BRIGHT copies only (`copy_bright_ratio` of the global bright
      median): a mixed-material patch that gets rescaled re-classifies
      its own dark micro-content against the shifted surround and mints
      dark_debris islands regardless of gain direction (measured:
      0.0031-0.0036 vs the 0.003 gate at az-35/-22.5).
    - BRIGHT destination rings only (`ring_bright_ratio`): against a
      dark/mixed ring the "offset" is mixture noise, not lighting.
    - Component-level log-median gain, clamped to `gain_clamp`, detail
      verbatim: one copy is one phenomenon (the shadow-apron component
      rule); per-texel matching would erase the copy's own content.

    Returns `(fill_rgb, matched_texels)`; the input array is not modified.
    """
    import numpy as np
    from scipy.ndimage import binary_dilation, binary_erosion
    from scipy.ndimage import label as cc_label

    eps = 0.02
    fill = np.asarray(fill_rgb, dtype=np.float32).copy()
    observed = np.asarray(observed_mask, dtype=bool)
    surface = np.asarray(surface_mask, dtype=bool)
    add = np.asarray(add_mask, dtype=bool)
    obs_lum = np.asarray(colors_rgb, dtype=np.float32).mean(axis=2)
    observed_lum = obs_lum[observed]
    if observed_lum.size == 0:
        return fill, 0
    bright_median = float(np.median(
        observed_lum[observed_lum >= np.median(observed_lum)]))
    if bright_median <= 0.0:
        return fill, 0
    fill_lum = fill.mean(axis=2)
    labels, count = cc_label(add, structure=np.ones((3, 3), bool))
    matched = 0
    for index in range(1, count + 1):
        component = labels == index
        if int(component.sum()) < int(min_component_texels):
            continue
        # pure-bright copies only (see docstring)
        if float(fill_lum[component].min()) < float(copy_bright_ratio) * bright_median:
            continue
        border = component & ~binary_erosion(component, iterations=2)
        ring = (binary_dilation(component, iterations=3) & ~component
                & observed & surface)
        if int(ring.sum()) < 8 or not border.any():
            continue
        ring_median = float(np.median(obs_lum[ring]))
        if ring_median < float(ring_bright_ratio) * bright_median:
            continue
        gain = float(np.exp(
            np.log(np.clip(ring_median, 0.0, 1.0) + eps)
            - np.log(np.clip(float(np.median(fill_lum[border])), 0.0, 1.0)
                     + eps)))
        gain = float(np.clip(gain, 1.0 - float(gain_clamp),
                             1.0 + float(gain_clamp)))
        if abs(gain - 1.0) < 1e-3:
            continue
        fill[component] = np.clip(fill[component] * gain, 0.0, 1.0)
        matched += int(component.sum())
    return fill, matched


def erode_view_alpha(observed_rgba: Any, *, erosion_px: Optional[int] = None) -> Any:
    """Erode the photo's alpha so contour-adjacent pixels are not projected.

    Pixels on the subject's outline mix foreground and background (or hair
    and skin) and, because the reconstructed silhouette never matches the
    photo contour exactly, grazing-angle texels otherwise sample them and
    smear outline colors across the surface. A small erosion of the sampling
    alpha removes exactly that rim while leaving interior detail unchanged.
    """
    import numpy as np
    from PIL import Image

    image = observed_rgba.convert("RGBA") if hasattr(observed_rgba, "convert") else observed_rgba
    array = np.asarray(image).copy()
    alpha = array[:, :, 3]
    if erosion_px is None:
        # Proportional to frame size, but capped so crisp high-resolution
        # inputs do not lose thin structures (a strap or chair spindle) to
        # an oversized rim erosion.
        erosion_px = min(8, max(1, int(round(min(image.size) / 256.0))))
    if int(erosion_px) <= 0:
        return image
    try:
        from scipy.ndimage import binary_erosion

        mask = alpha > 127
        eroded = binary_erosion(mask, iterations=int(erosion_px), border_value=False)
        array[:, :, 3] = np.where(eroded, alpha, 0)
        return Image.fromarray(array, mode="RGBA")
    except Exception:
        return image


def harmonize_view_exposure(
    reference_rgba: Any,
    *,
    primary_rgba: Any,
    strength: float = 1.0,
) -> Any:
    """Match a reference view's color statistics to the primary view.

    Reference photos of the same subject routinely differ in exposure and
    white balance from the primary view. Blending them without correction
    produces visible luminance seams at view boundaries. This applies the
    classic channel-wise mean/std transfer (Reinhard et al., "Color Transfer
    between Images") over foreground pixels only, which is robust, fast, and
    has no learned components.
    """
    import numpy as np

    ref = np.asarray(reference_rgba, dtype=np.float32)
    pri = np.asarray(primary_rgba, dtype=np.float32)
    if ref.ndim != 3 or pri.ndim != 3 or ref.shape[2] < 4 or pri.shape[2] < 4:
        return reference_rgba
    ref_mask = ref[:, :, 3] > 0.5
    pri_mask = pri[:, :, 3] > 0.5
    if ref_mask.sum() < 64 or pri_mask.sum() < 64:
        return reference_rgba
    out = ref.copy()
    clamped = float(np.clip(strength, 0.0, 1.0))
    for channel in range(3):
        ref_values = ref[:, :, channel][ref_mask]
        pri_values = pri[:, :, channel][pri_mask]
        ref_std = float(ref_values.std())
        pri_std = float(pri_values.std())
        if ref_std < 1e-5:
            continue
        gain = pri_std / ref_std
        # Bound the gain so a flat reference cannot explode into noise.
        gain = float(np.clip(gain, 0.5, 2.0))
        corrected = (ref[:, :, channel] - float(ref_values.mean())) * gain + float(pri_values.mean())
        out[:, :, channel] = (1.0 - clamped) * ref[:, :, channel] + clamped * corrected
    out[:, :, :3] = np.clip(out[:, :, :3], 0.0, 1.0)
    return out


def remove_speckle_weights(
    weight: Any,
    *,
    min_texels: int = 16,
    strong_weight: float = 0.5,
) -> Any:
    """Zero isolated low-confidence specks in a projection weight map.

    Genuine view coverage is spatially contiguous (a photo sees connected
    surface regions). Isolated texel islands come from numerical edge cases:
    grazing rays passing the facing test, or depth-test flicker at silhouette
    boundaries. Painted, they show up as salt-and-pepper photo samples in the
    middle of otherwise-unobserved regions, and every downstream fill then
    respects them as truth.

    A component is removed only when it is BOTH small (below `min_texels`,
    8-connected so diagonal thin coverage stays intact) AND low-confidence
    (its peak weight stays under `strong_weight`). True speckle is grazing
    coverage with facing barely above threshold, so its weights are tiny;
    a small but strongly-facing chart (a legitimately observed small UV
    island) keeps its coverage.
    """
    import numpy as np

    weights = np.asarray(weight, dtype=np.float32)
    covered = weights > 0.0
    if not covered.any() or int(min_texels) <= 1:
        return weights
    try:
        from scipy.ndimage import label, maximum

        labels, count = label(covered, structure=np.ones((3, 3), dtype=bool))
        if count <= 1:
            return weights
        sizes = np.bincount(labels.ravel())
        component_ids = np.arange(1, count + 1)
        peak_weights = np.asarray(maximum(weights, labels=labels, index=component_ids))
        small = np.zeros(count + 1, dtype=bool)
        small[1:] = (sizes[1:] < int(min_texels)) & (peak_weights < float(strong_weight))
        if small.any():
            weights = np.where(small[labels], 0.0, weights)
    except Exception:
        return weights
    return weights


def feather_projection_weight(weight: Any, *, feather_texels: float = 6.0) -> Any:
    """Feather the hard boundary of a projection's weight map.

    Projection validity is a binary cut (facing threshold, depth test, image
    bounds). Blending views whose weights end abruptly leaves visible seams.
    This attenuates weights within `feather_texels` of the projection
    boundary using a distance transform, so views hand off smoothly.
    """
    import numpy as np

    weights = np.asarray(weight, dtype=np.float32)
    covered = weights > 0.0
    if not covered.any() or covered.all() or feather_texels <= 0.0:
        return weights
    try:
        from scipy.ndimage import distance_transform_edt

        interior_distance = distance_transform_edt(covered)
        ramp = np.clip(interior_distance / float(feather_texels), 0.0, 1.0).astype(np.float32)
        return weights * ramp
    except Exception:
        return weights


def blend_projections(
    projections: Sequence[Mapping[str, Any]],
    *,
    atlas_shape: Tuple[int, int],
    sharpness: float = 3.0,
    feather_texels: float = 6.0,
    detail_fusion: str = "two_band",
    detail_sigma: float = 3.0,
    winner_smooth_sigma: float = 8.0,
) -> Dict[str, Any]:
    """Blend per-view projections with best-view-biased weights.

    A plain weighted average of overlapping views produces ghosting whenever
    the projections disagree slightly (imperfect geometry, parallax between
    the real reference photo and the reconstructed surface). The softmax
    bias over per-view weights (temperature `1/sharpness`)

        w_i' = w_i * exp(sharpness * (w_i - max_j w_j))

    suppresses ghosting only where one view clearly dominates — at WEIGHT
    TIES (two views facing the surface equally, the entire ridge between
    adjacent view cones) the bias is exp(0) for both and the blend
    degenerates to a plain average. Averaging views with residual
    registration error is convolution with the error kernel: for two views
    offset by d texels the MTF has a null at frequency 1/(2d) — a few
    texels of disagreement erase every feature finer than ~2d texels.
    Measured on the four-view owl: the equal-facing band between generated
    views turned crisp plumage into mud.

    `detail_fusion="two_band"` (multi-view only) applies the production
    answer (Baumberg BMVC'02; Metashape "mosaic"; Burt-Adelson band rule:
    transition width proportional to wavelength): split each view at
    `detail_sigma` texels; the LOW band (tone, lighting) keeps the softmax
    average with its wide smooth transitions, the HIGH band (detail) is
    winner-take-all from the single best view per texel. The high band is
    zero-mean, so ownership switches carry no visible tone step — a
    zero-width transition is safe there, and detail is never averaged
    across imperfectly-registered views. The winner is the argmax of the
    weight maps smoothed at `winner_smooth_sigma` (smooth the WEIGHTS, not
    the labels: the raw argmax dithers along the tie ridge, fragmenting
    detail ownership into slivers).

    `detail_fusion="average"` restores the pre-two-band behavior;
    single-view bakes are bit-identical under either mode.
    """
    import numpy as np

    height, width = atlas_shape
    accum_rgb = np.zeros((height, width, 3), dtype=np.float32)
    accum_weight = np.zeros((height, width), dtype=np.float32)
    accum_alpha = np.zeros((height, width), dtype=np.float32)
    view_stats: List[Dict[str, Any]] = []

    prepared: List[Tuple[Any, Any, Mapping[str, Any]]] = []
    stacked_weights: List[Any] = []
    raw_coverage = np.zeros((height, width), dtype=bool)
    for projection in projections:
        rgba = np.asarray(projection.get("rgba"), dtype=np.float32)
        weight = np.asarray(projection.get("weight"), dtype=np.float32)
        if rgba.shape[:2] != (height, width) or weight.shape != (height, width):
            continue
        # Isolated speck coverage is projection noise, not signal; drop it
        # before it pollutes both the blend and the coverage gates.
        speckle_floor = max(4, int(round((height * width) / 262144)))
        weight = remove_speckle_weights(weight, min_texels=speckle_floor)
        # Coverage gating downstream (symmetry fill, inpainting) must see the
        # true observed set; feathering only shapes the blend falloff.
        raw_coverage |= weight > 0.0
        weight = feather_projection_weight(weight, feather_texels=feather_texels)
        prepared.append((rgba, weight, projection))
        stacked_weights.append(weight)

    if not prepared:
        return {
            "rgb": accum_rgb,
            "weight": accum_weight,
            "alpha": accum_alpha,
            "coverage": raw_coverage,
            "view_stats": view_stats,
        }

    two_band = detail_fusion == "two_band" and len(prepared) > 1
    view_low: List[Any] = []
    if two_band:
        from scipy.ndimage import gaussian_filter

        # Per-view frequency split as a WEIGHT-CARRYING normalized
        # convolution: low = G(rgb*w) / G(w). A binary-coverage mask
        # admitted weight-crushed rim texels at full strength — exactly
        # the samples every other stage distrusts — and the one-sided
        # mean at coverage edges then reported the brighter interior,
        # turning rim shading into a large negative detail band
        # (measured: a flank band DARKER than both witnesses' own
        # content, Y 0.31 vs 0.52/0.57). Weights already decay toward
        # edges, so carrying them both excludes crushed rims and softens
        # the one-sided bias.
        for rgba, weight, _projection in prepared:
            low = np.empty((height, width, 3), dtype=np.float32)
            den = gaussian_filter(weight, float(detail_sigma))
            good = den > 1e-6
            for channel in range(3):
                num = gaussian_filter(
                    rgba[:, :, channel] * weight, float(detail_sigma))
                plane = np.zeros((height, width), dtype=np.float32)
                plane[good] = num[good] / den[good]
                low[:, :, channel] = plane
            view_low.append(low)

    weight_stack = np.stack(stacked_weights, axis=0)
    max_weight = weight_stack.max(axis=0)
    for index, (rgba, weight, projection) in enumerate(prepared, start=1):
        if sharpness > 0.0:
            bias = np.exp(np.clip(float(sharpness) * (weight - max_weight), -20.0, 0.0))
            biased_weight = weight * bias
        else:
            biased_weight = weight
        base_rgb = view_low[index - 1] if two_band else rgba[:, :, :3]
        accum_rgb += base_rgb * biased_weight[:, :, None]
        accum_weight += biased_weight
        accum_alpha = np.maximum(accum_alpha, rgba[:, :, 3])
        view_stats.append(
            {
                "index": index,
                "label": str(projection.get("label") or f"view_{index:02d}"),
                "azimuth_deg": float(projection.get("azimuth_deg", 0.0)),
                "elevation_deg": float(projection.get("elevation_deg", 0.0)),
                "generated": bool(projection.get("generated", False)),
                "coverage_ratio": round(float(projection.get("coverage_ratio") or 0.0), 4),
            }
        )

    observed = accum_weight > 1e-6
    blended_rgb = np.zeros_like(accum_rgb)
    if observed.any():
        blended_rgb[observed] = accum_rgb[observed] / accum_weight[observed][:, None]

    if two_band and observed.any():
        from scipy.ndimage import gaussian_filter

        # Detail ownership: argmax over MASK-NORMALIZED smoothed weights
        # for spatial coherence. Plain Gaussian smoothing averages in the
        # zeros outside each view's coverage, halving a view's smoothed
        # weight within ~sigma of its own edge — measured: a grazing back
        # view with 3x lower pointwise weight out-owned a confident side
        # view in an 8-16 texel strip inside every coverage edge.
        # Normalizing by the smoothed coverage indicator removes the
        # coverage bias; eligibility additionally requires a texel's RAW
        # weight to reach 30% of the local best, so a poor witness can
        # never own detail merely by spatial majority.
        covering = weight_stack > 0.0
        smoothed = np.stack(
            [np.divide(
                gaussian_filter(w, float(winner_smooth_sigma)),
                np.maximum(gaussian_filter(
                    c.astype(np.float32), float(winner_smooth_sigma)), 1e-6))
             for w, c in zip(stacked_weights, covering)],
            axis=0)
        quality_floor = 0.3 * weight_stack.max(axis=0)
        eligible_mask = covering & (weight_stack >= quality_floor[None, ...])
        eligible = np.where(eligible_mask, smoothed, -1.0)
        winner = eligible.argmax(axis=0)
        winner_valid = np.take_along_axis(
            eligible_mask, winner[None, ...], axis=0)[0]
        # Where no eligible smoothed winner exists, use the raw-weight best.
        raw_winner = weight_stack.argmax(axis=0)
        winner = np.where(winner_valid, winner, raw_winner)

        # The HIGH band crosses ownership boundaries through a ~3-texel
        # feather, not a hard cut: the detail band is only zero-mean at
        # scales below `detail_sigma`, so a zero-width switch still steps
        # by the band's local content and reads as a long artificial edge
        # (measured: the whole-bake seam metric tripled on hard cuts
        # while no tone seam was visible). Misaligned detail averages
        # only inside this narrow strip — invisible at 3 texels, mud at
        # the overlap scale the softmax average used to blend across.
        handoff_sigma = max(1.0, float(detail_sigma) / 2.0)
        high_rgb = np.zeros((height, width, 3), dtype=np.float32)
        high_norm = np.zeros((height, width), dtype=np.float32)
        for index, (rgba, _weight, _projection) in enumerate(prepared):
            indicator = (winner == index).astype(np.float32)
            if not indicator.any():
                continue
            feathered = gaussian_filter(indicator, handoff_sigma)
            feathered *= covering[index]
            if not feathered.any():
                continue
            high = rgba[:, :, :3] - view_low[index]
            high_rgb += high * feathered[:, :, None]
            high_norm += feathered
        has_high = high_norm > 1e-6
        blended_rgb[has_high] += (
            high_rgb[has_high] / high_norm[has_high][:, None])
        blended_rgb[~observed] = 0.0
        np.clip(blended_rgb, 0.0, 1.0, out=blended_rgb)

        # HANDOFF SEAM LEDGER: tone disagreement across ownership
        # boundaries, measured in texture space where the handoffs are
        # KNOWN. A render-space seam detector cannot separate a genuine
        # handoff seam from a crisp carved contour (measured: their
        # side-tone distributions overlap completely); here the boundary
        # set is exact and the measured quantity is the OWNERS' low-band
        # difference at the boundary texel — content detail is excluded
        # by construction. The whole-bake acceptance gate consumes this.
        boundary_h = (
            observed[:, 1:] & observed[:, :-1]
            & (winner[:, 1:] != winner[:, :-1]))
        boundary_v = (
            observed[1:, :] & observed[:-1, :]
            & (winner[1:, :] != winner[:-1, :]))
        steps: List[Any] = []
        owners_a: List[Any] = []
        owners_b: List[Any] = []
        co_witnessed: List[Any] = []
        low_stack = np.stack(view_low, axis=0)
        for boundary, (dr, dc) in ((boundary_h, (0, 1)), (boundary_v, (1, 0))):
            if not boundary.any():
                continue
            rows_b, cols_b = np.nonzero(boundary)
            owner_a = winner[rows_b, cols_b]
            owner_b = winner[rows_b + dr, cols_b + dc]
            steps.append(
                low_stack[owner_a, rows_b, cols_b]
                - low_stack[owner_b, rows_b, cols_b])
            owners_a.append(owner_a)
            owners_b.append(owner_b)
            # A boundary texel is CO-WITNESSED when both owners hold weight
            # on both sides of the switch: the low-band average can ramp
            # across it. Where one owner's coverage simply ENDS, the handoff
            # is one-sided and only as wide as the coverage feather.
            co_witnessed.append(
                (weight_stack[owner_a, rows_b + dr, cols_b + dc] > 0.0)
                & (weight_stack[owner_b, rows_b, cols_b] > 0.0))
        if steps:
            all_steps_rgb = np.concatenate(steps, axis=0)
            all_steps = np.abs(all_steps_rgb).mean(axis=1)
            pair_a = np.concatenate(owners_a)
            pair_b = np.concatenate(owners_b)
            pair_co = np.concatenate(co_witnessed)
            handoff_seams = {
                "boundary_texels": int(all_steps.size),
                "step_p50": round(float(np.percentile(all_steps, 50)), 4),
                "step_p95": round(float(np.percentile(all_steps, 95)), 4),
                "step_max": round(float(all_steps.max()), 4),
            }
            # Per-pair attribution: which view pairs carry the steps, how
            # much of each step is luminance (a lighting/tone difference the
            # delight/compositor lanes can reconcile) vs chroma, and whether
            # the boundary is co-witnessed. Diagnosis data for the whole-bake
            # gate; adds no pixel-affecting behavior.
            labels = [
                str(projection.get("label") or f"view_{index + 1:02d}")
                for index, (_r, _w, projection) in enumerate(prepared)]
            lum_steps = np.abs(all_steps_rgb @ np.array(
                [0.299, 0.587, 0.114], dtype=np.float32))
            lo = np.minimum(pair_a, pair_b)
            hi = np.maximum(pair_a, pair_b)
            pair_key = lo.astype(np.int64) * len(prepared) + hi
            pair_rows: List[Dict[str, Any]] = []
            for key in np.unique(pair_key):
                sel = pair_key == key
                pair_rows.append({
                    "views": [labels[int(key) // len(prepared)],
                              labels[int(key) % len(prepared)]],
                    "boundary_texels": int(sel.sum()),
                    "step_p50": round(float(np.percentile(all_steps[sel], 50)), 4),
                    "step_p95": round(float(np.percentile(all_steps[sel], 95)), 4),
                    "lum_share_p50": round(float(np.percentile(
                        lum_steps[sel] / np.maximum(all_steps[sel], 1e-6), 50)), 4),
                    "co_witnessed_frac": round(float(pair_co[sel].mean()), 4),
                })
            pair_rows.sort(key=lambda row: -row["boundary_texels"])
            handoff_seams["pairs"] = pair_rows
        else:
            handoff_seams = {"boundary_texels": 0}
    else:
        handoff_seams = None
    # Texels covered only by the feather-zeroed rim keep raw coverage only
    # while a colored neighbor exists nearby IN UV WITHIN THE SAME feather
    # band; farther rim texels are demoted to unseen so the 3D fill handles
    # them. An unbounded UV-nearest gather here could borrow colors from an
    # unrelated chart and then lock them in as observed truth.
    rim = raw_coverage & ~observed
    if rim.any():
        if not observed.any():
            raw_coverage = observed.copy()
        else:
            try:
                from scipy.ndimage import distance_transform_edt

                distances, nearest = distance_transform_edt(~observed, return_indices=True)
                near_rim = rim & (distances <= float(feather_texels) + 1.0)
                far_rim = rim & ~near_rim
                blended_rgb[near_rim] = blended_rgb[nearest[0][near_rim], nearest[1][near_rim]]
                if far_rim.any():
                    raw_coverage = raw_coverage & ~far_rim
            except Exception:
                raw_coverage = observed.copy()
    return {
        "rgb": blended_rgb,
        "weight": np.clip(max_weight, 0.0, 1.0),
        "alpha": accum_alpha,
        "coverage": raw_coverage,
        "view_stats": view_stats,
        "handoff_seams": handoff_seams,
    }


def mesh_graph_harmonic_fill(
    mesh: Any,
    *,
    positions_texture: Any,
    observed_mask: Any,
    colors_rgba: Any,
    max_vertices: int = 200000,
) -> Optional[Any]:
    """Fill unseen texels by harmonic color interpolation over the mesh graph.

    Euclidean nearest-neighbor borrowing jumps across surface concavities
    (the back of a head borrows neck skin through empty space). Diffusing
    along mesh edges instead respects surface connectivity, which is how the
    upstream Hunyuan `mesh_inpaint_processor` fills hidden regions.

    Formulation: solve the graph Laplace equation L x = 0 for unknown vertex
    colors with observed vertices as Dirichlet boundary values (uniform edge
    weights). The solution is the unique harmonic interpolant: smooth,
    maximum-principle-bounded (no invented colors), and it converges to
    nearby boundary colors along the surface. Observed texel colors are
    pulled onto vertices by nearest-vertex assignment; unseen texels then
    read an inverse-distance blend of their nearest vertices' solved colors.
    (Nearest-VERTEX texel assignment was measured to render as faceted
    flat-color polygon blocks at close zoom: ~59k vertices serve ~4.2M
    texels at 2048, so every texel inside a vertex's Voronoi cell got the
    identical color. IDW over the 3 nearest vertices is C0 across cell
    borders and removed the visible facets: fill-region flat-plateau
    fraction 0.45 -> 0.21 on the face proof asset.)

    Returns the filled RGBA array, or None when the solve is unavailable
    (missing scipy, oversized mesh, degenerate inputs) so callers can fall
    back to the KD-tree fill.
    """
    import numpy as np

    try:
        from scipy import sparse
        from scipy.sparse.linalg import spsolve
        from scipy.spatial import cKDTree
    except Exception:
        return None

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    if len(vertices) == 0 or len(vertices) > int(max_vertices):
        return None
    rgba = np.asarray(colors_rgba, dtype=np.float32).copy()
    positions = np.asarray(positions_texture, dtype=np.float32)
    surface = positions[:, :, 3] > 0.0
    observed = np.asarray(observed_mask, dtype=bool) & surface
    unseen = surface & ~observed
    if not unseen.any() or not observed.any():
        return None

    tree = cKDTree(vertices)

    # Pull observed texel colors onto vertices (average when several texels
    # map to one vertex).
    observed_points = positions[:, :, :3][observed]
    observed_colors = rgba[observed][:, :3]
    _, observed_vertex_idx = tree.query(observed_points, k=1, workers=-1)
    vertex_color_sum = np.zeros((len(vertices), 3), dtype=np.float64)
    vertex_color_count = np.zeros(len(vertices), dtype=np.float64)
    np.add.at(vertex_color_sum, observed_vertex_idx, observed_colors.astype(np.float64))
    np.add.at(vertex_color_count, observed_vertex_idx, 1.0)
    known = vertex_color_count > 0
    if not known.any() or known.all():
        return None
    known_colors = vertex_color_sum[known] / vertex_color_count[known][:, None]

    edges = np.asarray(mesh.edges_unique, dtype=np.int64)
    if len(edges) == 0:
        return None
    # Anisotropic (feature-preserving) conductance: edges crossing sharp
    # creases get near-zero weight. Generated meshes fuse thin shell flaps
    # (hair films) onto the surface below along crease rims; with uniform
    # weights the harmonic fill diffuses the UNDERLYING material's color
    # (forehead skin) up onto the shell. Down-weighting crease edges makes
    # color flow along each smooth sheet from its own observed region
    # instead of across the fusion seam.
    try:
        vertex_normals = np.asarray(mesh.vertex_normals, dtype=np.float64)
        normal_dot = (vertex_normals[edges[:, 0]] * vertex_normals[edges[:, 1]]).sum(axis=1)
        conductance = np.clip(normal_dot, 0.05, 1.0) ** 2
    except Exception:
        conductance = np.ones(len(edges), dtype=np.float64)
    row = np.concatenate([edges[:, 0], edges[:, 1]])
    col = np.concatenate([edges[:, 1], edges[:, 0]])
    data = np.concatenate([conductance, conductance]).astype(np.float64)
    adjacency = sparse.coo_matrix((data, (row, col)), shape=(len(vertices), len(vertices))).tocsr()
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    laplacian = sparse.diags(degree) - adjacency

    unknown = ~known
    unknown_idx = np.nonzero(unknown)[0]
    known_idx = np.nonzero(known)[0]
    # L_uu x_u = -L_uk x_k  (Dirichlet harmonic interpolation)
    l_uu = laplacian[unknown_idx][:, unknown_idx].tocsc()
    l_uk = laplacian[unknown_idx][:, known_idx]
    rhs = -l_uk @ known_colors
    try:
        solved = spsolve(l_uu, rhs)
    except Exception:
        return None
    # A fully-unobserved disconnected component makes L_uu singular:
    # depending on platform, spsolve then emits zeros (fills black) or
    # NaN/inf. Fall back to the KD fill rather than paint garbage.
    if not np.isfinite(np.asarray(solved)).all():
        return None
    solved = np.atleast_2d(np.asarray(solved, dtype=np.float64))
    if solved.shape[0] == 3 and solved.shape[1] == len(unknown_idx):
        solved = solved.T
    vertex_colors = np.zeros((len(vertices), 3), dtype=np.float64)
    vertex_colors[known_idx] = known_colors
    vertex_colors[unknown_idx] = solved
    vertex_colors = np.clip(vertex_colors, 0.0, 1.0)

    unseen_points = positions[:, :, :3][unseen]
    k_interp = int(min(3, len(vertices)))
    distances, unseen_vertex_idx = tree.query(unseen_points, k=k_interp, workers=-1)
    distances = np.atleast_2d(np.asarray(distances, dtype=np.float32))
    unseen_vertex_idx = np.atleast_2d(np.asarray(unseen_vertex_idx, dtype=np.int64))
    if distances.shape[0] == 1 and len(unseen_points) > 1:
        distances = distances.T
        unseen_vertex_idx = unseen_vertex_idx.T
    # Soft floor relative to the local vertex spacing: texels at a vertex
    # reproduce that vertex's color exactly, texels between vertices blend.
    floor = 0.1 * float(np.median(distances[:, 0])) + 1e-9
    weights = (1.0 / (distances + floor)).astype(np.float32)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-8)
    interpolated = np.einsum("nk,nkc->nc", weights, vertex_colors[unseen_vertex_idx])
    rgba[unseen, :3] = interpolated.astype(np.float32)
    rgba[unseen, 3] = 1.0
    return rgba


def level_composed_seams(
    mesh: Any,
    *,
    positions_texture: Any,
    colors_rgb: Any,
    region_map: Any,
    confidence_map: Optional[Any] = None,
    smoothness: float = 200.0,
    pin_strength: float = 2.0,
    anchor: float = 1e-3,
    max_offset: float = 0.30,
    confident_weight: float = 0.45,
    boundary_cap: float = 0.18,
    max_vertices: int = 200000,
) -> Optional[Any]:
    """Solve per-region low-frequency tone offsets that cancel seam steps.

    Multi-view composition partitions the surface into REGIONS (per-texel
    winning view, mirror completion). Exposure/shading differences between
    the regions' sources show up exactly at region boundaries as tone steps
    (the mid-face vertical seam on the face lane measured 0.06 luminance,
    ~16/255 — clearly visible), while inside a region the content is
    self-consistent. This is the classic texture-atlas seam-leveling setup
    (Ivanov & Lempitsky): solve one additive offset field per region that is
    LOW-FREQUENCY inside regions but discontinuous at region borders, such
    that corrected colors agree across every boundary edge. High-frequency
    content (pores, lashes, hair strands) rides on top unchanged.

    Formulation, one offset g_v per composed vertex (its dominant region's
    field sample), solved jointly per RGB channel:

        E = sum_{cross-region edges}  ((c_a + g_a) - (c_b + g_b))^2
          + smoothness * sum_{same-region edges} (g_a - g_b)^2
          + pin_strength * sum_{confident vertices} g_v^2
          + anchor * sum_v g_v^2

    Two safeguards are load-bearing:

    - `boundary_cap`: only boundary edges whose step is small enough to BE
      a tone seam participate. A large step (hair against skin at the
      hairline, where region ownership also changes hands) is genuine
      material content; demanding agreement there would tint both materials
      toward each other around every such border (measured: an uncapped
      solve dropped the right-profile identity's worst face window from
      +0.10 to -0.13 SSIM around the ear).
    - confidence pinning: every vertex whose winning witness is CONFIDENT
      (weight above `confident_weight`) is pinned toward zero correction.
      Each photo is ground truth on surface it saw well — the identity
      contract at that photo's own pose — so leveling may only recolor the
      weak/contested bands between confident zones, bridging tone as a ramp
      across the weak band instead of repainting anyone's truth.

    Returns per-texel RGB offsets (H, W, 3) to ADD to `colors_rgb`, or None
    when the solve is unavailable or degenerate.
    """
    import numpy as np

    try:
        from scipy import sparse
        from scipy.sparse.linalg import spsolve
        from scipy.spatial import cKDTree
    except Exception:
        return None

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    edges = np.asarray(mesh.edges_unique, dtype=np.int64)
    if len(vertices) == 0 or len(edges) == 0 or len(vertices) > int(max_vertices):
        return None
    positions = np.asarray(positions_texture, dtype=np.float32)
    colors = np.asarray(colors_rgb, dtype=np.float32)
    regions = np.asarray(region_map)
    composed = regions >= 0
    if not composed.any():
        return None
    region_count = int(regions.max()) + 1
    if region_count < 2:
        return None

    n_vert = len(vertices)
    tree = cKDTree(vertices)
    _, texel_vertex = tree.query(positions[:, :, :3][composed], k=1, workers=-1)
    texel_region = regions[composed].astype(np.int64)
    texel_color = colors[composed]

    # Dominant region per vertex; per-vertex mean color over that region's
    # texels only (mixing regions here would smear the very step being
    # measured).
    key = texel_vertex * region_count + texel_region
    counts = np.bincount(key, minlength=n_vert * region_count).reshape(n_vert, region_count)
    dominant = np.where(counts.any(axis=1), counts.argmax(axis=1), -1)
    use = texel_region == dominant[texel_vertex]
    color_sum = np.zeros((n_vert, 3), dtype=np.float64)
    color_cnt = np.zeros(n_vert, dtype=np.float64)
    np.add.at(color_sum, texel_vertex[use], texel_color[use].astype(np.float64))
    np.add.at(color_cnt, texel_vertex[use], 1.0)
    present = color_cnt > 0
    if int(present.sum()) < 16:
        return None
    mean_color = np.zeros_like(color_sum)
    mean_color[present] = color_sum[present] / color_cnt[present][:, None]

    comp_idx = np.full(n_vert, -1, dtype=np.int64)
    comp_ids = np.nonzero(present)[0]
    comp_idx[comp_ids] = np.arange(len(comp_ids))
    n_unk = len(comp_ids)

    edge_a, edge_b = edges[:, 0], edges[:, 1]
    both = present[edge_a] & present[edge_b]
    edge_a, edge_b = edge_a[both], edge_b[both]
    cross = dominant[edge_a] != dominant[edge_b]
    idx_a, idx_b = comp_idx[edge_a], comp_idx[edge_b]

    rows: List[Any] = []
    cols: List[Any] = []
    vals: List[Any] = []
    rhs = np.zeros((n_unk, 3), dtype=np.float64)

    def add_term(i: Any, j: Any, v: Any) -> None:
        rows.append(i)
        cols.append(j)
        vals.append(v)

    if not cross.any():
        return None
    color_a = mean_color[edge_a[cross]]
    color_b = mean_color[edge_b[cross]]
    seamlike = np.abs(color_b - color_a).mean(axis=1) <= float(boundary_cap)
    if not seamlike.any():
        return None
    delta = (color_b - color_a)[seamlike]
    i = idx_a[cross][seamlike]
    j = idx_b[cross][seamlike]
    ones = np.ones(len(i))
    add_term(i, i, ones)
    add_term(j, j, ones)
    add_term(i, j, -ones)
    add_term(j, i, -ones)
    np.add.at(rhs, i, delta)
    np.add.at(rhs, j, -delta)

    same = ~cross
    if same.any():
        i, j = idx_a[same], idx_b[same]
        w = np.full(len(i), float(smoothness))
        add_term(i, i, w)
        add_term(j, j, w)
        add_term(i, j, -w)
        add_term(j, i, -w)

    diagonal = np.full(n_unk, float(anchor))
    if confidence_map is not None:
        confidence = np.asarray(confidence_map, dtype=np.float32)
        if confidence.shape == composed.shape:
            vertex_conf = np.zeros(n_vert, dtype=np.float64)
            vertex_cnt = np.zeros(n_vert, dtype=np.float64)
            np.add.at(vertex_conf, texel_vertex, confidence[composed].astype(np.float64))
            np.add.at(vertex_cnt, texel_vertex, 1.0)
            with np.errstate(invalid="ignore"):
                conf_mean = np.where(vertex_cnt > 0, vertex_conf / np.maximum(vertex_cnt, 1e-9), 0.0)
            pinned = present & (conf_mean > float(confident_weight))
            diagonal[comp_idx[pinned]] += float(pin_strength)
    add_term(np.arange(n_unk), np.arange(n_unk), diagonal)

    matrix = sparse.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n_unk, n_unk),
    ).tocsc()
    try:
        solved = spsolve(matrix, rhs)
    except Exception:
        return None
    solved = np.atleast_2d(np.asarray(solved, dtype=np.float64))
    if solved.shape[0] == 3 and solved.shape[1] == n_unk:
        solved = solved.T
    if not np.isfinite(solved).all():
        return None
    solved = np.clip(solved, -float(max_offset), float(max_offset))

    field = np.zeros((n_vert, 3), dtype=np.float32)
    field[comp_ids] = solved.astype(np.float32)
    offsets = np.zeros_like(colors)
    offsets[composed] = field[texel_vertex]
    return offsets


def _texel_world_pitch(positions_texture: Any, surface_mask: Any) -> float:
    """Median 3D distance between horizontally adjacent surface texels.

    This is the atlas sampling pitch in world units — the natural length
    scale for any texel-resolution surface operation (smoothing radii,
    detail wavelengths). Chart-adjacent texel pairs are true 3D neighbors;
    pairs straddling chart boundaries are large outliers, so the median is
    the robust statistic.
    """
    import numpy as np

    surface = np.asarray(surface_mask, dtype=bool)
    positions = np.asarray(positions_texture, dtype=np.float32)[:, :, :3]
    pair = surface[:, :-1] & surface[:, 1:]
    if not pair.any():
        return 1e-3
    deltas = np.linalg.norm(positions[:, 1:][pair] - positions[:, :-1][pair], axis=1)
    return float(np.median(deltas)) + 1e-12


def texel_surface_smooth(
    colors_rgba: Any,
    *,
    positions_texture: Any,
    normals_texture: Optional[Any],
    observed_mask: Any,
    fill_mask: Optional[Any] = None,
    iterations: int = 12,
    neighbors: int = 8,
) -> Any:
    """Relax fill-texel colors to C0 over the 3D surface texel graph.

    Every fill strategy that assigns colors from a coarser proxy leaves
    texel-scale discontinuities: the harmonic solve interpolates between
    VERTICES (Voronoi plateaus without IDW, residual cell seams with it),
    and the KD fallback blends per-texel donor sets that change abruptly
    between adjacent texels (patchwork mottle). A few Jacobi iterations over
    the k-nearest-neighbor graph of texel 3D positions — observed texels
    held fixed as Dirichlet anchors — make the fill C0 at texel resolution
    while keeping the global color transport of the underlying fill.
    Neighbor weights combine inverse distance with normal agreement so
    smoothing follows the surface sheet rather than jumping gaps (same
    reasoning as the KD fill's normal term). Measured on the proof assets:
    fill-region Laplacian energy dropped 30x (harmonic base) / 8x (KD base)
    with no visible loss of legitimate content.

    Returns the smoothed RGBA array (input unchanged on missing scipy or
    empty masks).
    """
    import numpy as np

    rgba = np.asarray(colors_rgba, dtype=np.float32).copy()
    try:
        from scipy.spatial import cKDTree
    except Exception:
        return rgba

    positions = np.asarray(positions_texture, dtype=np.float32)
    surface = positions[:, :, 3] > 0.0
    observed = np.asarray(observed_mask, dtype=bool) & surface
    fill = surface & ~observed if fill_mask is None else (np.asarray(fill_mask, dtype=bool) & surface)
    if not fill.any():
        return rgba

    normals = None
    if normals_texture is not None:
        normals = np.asarray(normals_texture, dtype=np.float32)[:, :, :3]
        norms = np.linalg.norm(normals, axis=2, keepdims=True)
        normals = np.divide(normals, np.maximum(norms, 1e-8))

    participating = fill | observed
    part_pos = positions[:, :, :3][participating]
    part_col = rgba[participating][:, :3].astype(np.float32)
    part_fixed = observed[participating]
    fill_in_part = np.nonzero(fill[participating])[0]
    if len(part_pos) <= neighbors:
        return rgba

    tree = cKDTree(part_pos)
    # k+1 because each texel's nearest hit is itself.
    distances, indices = tree.query(part_pos[fill_in_part], k=int(neighbors) + 1, workers=-1)
    distances = np.asarray(distances, dtype=np.float32)[:, 1:]
    indices = np.asarray(indices, dtype=np.int64)[:, 1:]

    pitch = float(np.median(distances[:, 0])) + 1e-12
    weights = (1.0 / (distances + 0.5 * pitch)).astype(np.float32)
    if normals is not None:
        part_nrm = normals[participating]
        agreement = np.einsum("nc,nkc->nk", part_nrm[fill_in_part], part_nrm[indices])
        weights = weights * (0.05 + 0.95 * np.clip(agreement, 0.0, 1.0) ** 2)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-8)

    colors = part_col.copy()
    anchors = part_col[part_fixed]
    for _ in range(int(iterations)):
        colors[fill_in_part] = np.einsum("nk,nkc->nc", weights, colors[indices])
        colors[part_fixed] = anchors

    rows, cols = np.nonzero(participating)
    rgba[rows[fill_in_part], cols[fill_in_part], :3] = np.clip(colors[fill_in_part], 0.0, 1.0)
    rgba[rows[fill_in_part], cols[fill_in_part], 3] = 1.0
    return rgba


def _fill_value_noise_3d(points: Any, wavelength: float, seed: int) -> Any:
    """Deterministic trilinear value noise at 3D points.

    Evaluated at world positions, the field is continuous across UV chart
    boundaries by construction (adjacent surface points get near-identical
    values regardless of where their texels live in the atlas), which is
    the property the fill-detail pass needs. Integer-hash based: no state,
    reproducible across processes and platforms.
    """
    import numpy as np

    p = np.asarray(points, dtype=np.float64) / float(wavelength)
    cell = np.floor(p).astype(np.int64)
    frac = (p - cell).astype(np.float32)
    t = frac * frac * (3.0 - 2.0 * frac)  # smoothstep, C1 interpolation

    def hash_noise(ix, iy, iz):
        h = (
            ix * np.int64(374761393)
            + iy * np.int64(668265263)
            + iz * np.int64(2147483647)
            + np.int64(seed) * np.int64(999998727)
        )
        h = (h ^ (h >> 13)) * np.int64(1274126177)
        h = h ^ (h >> 16)
        return ((h & np.int64(0xFFFFFF)).astype(np.float32) / float(0xFFFFFF)) * 2.0 - 1.0

    ix, iy, iz = cell[:, 0], cell[:, 1], cell[:, 2]
    n000 = hash_noise(ix, iy, iz)
    n100 = hash_noise(ix + 1, iy, iz)
    n010 = hash_noise(ix, iy + 1, iz)
    n110 = hash_noise(ix + 1, iy + 1, iz)
    n001 = hash_noise(ix, iy, iz + 1)
    n101 = hash_noise(ix + 1, iy, iz + 1)
    n011 = hash_noise(ix, iy + 1, iz + 1)
    n111 = hash_noise(ix + 1, iy + 1, iz + 1)
    nx00 = n000 * (1 - t[:, 0]) + n100 * t[:, 0]
    nx10 = n010 * (1 - t[:, 0]) + n110 * t[:, 0]
    nx01 = n001 * (1 - t[:, 0]) + n101 * t[:, 0]
    nx11 = n011 * (1 - t[:, 0]) + n111 * t[:, 0]
    nxy0 = nx00 * (1 - t[:, 1]) + nx10 * t[:, 1]
    nxy1 = nx01 * (1 - t[:, 1]) + nx11 * t[:, 1]
    return nxy0 * (1 - t[:, 2]) + nxy1 * t[:, 2]


def _balanced_query(tree: Any, points: Any, k: int, *, workers: int = -1) -> Tuple[Any, Any]:
    """`cKDTree.query` under a randomized query order (results identical).

    scipy splits the query array into contiguous per-thread chunks. Atlas-
    ordered texel queries are spatially coherent, so whole chunks land far
    from the tree (expensive exact-NN backtracking) while others are trivial
    — one straggler thread then owns most of the wall time. A fixed random
    permutation spreads hard queries across all threads. Per-point results
    are exact NN either way; the permutation is undone before returning
    (measured 3.8x on the owl donor query, bitwise-identical output).
    """
    import numpy as np

    points = np.asarray(points)
    if len(points) < 4096:
        return tree.query(points, k=k, workers=workers)
    permutation = np.random.default_rng(11).permutation(len(points))
    inverse = np.empty_like(permutation)
    inverse[permutation] = np.arange(len(permutation))
    distances, indices = tree.query(points[permutation], k=k, workers=workers)
    return distances[inverse], indices[inverse]


def _masked_gaussian_filter(field: Any, mask: Any, sigma: float) -> Any:
    """Gaussian filter over masked pixels only (normalized convolution)."""
    import numpy as np
    from scipy.ndimage import gaussian_filter

    mask_f = np.asarray(mask, dtype=np.float32)
    num = gaussian_filter(np.asarray(field, dtype=np.float32) * mask_f, sigma=sigma)
    den = gaussian_filter(mask_f, sigma=sigma)
    out = np.zeros_like(num)
    good = den > 1e-4
    out[good] = num[good] / den[good]
    return out


def _multigrid_orientation_field(
    query_points: Any,
    anchor_points: Any,
    anchor_directions: Any,
    anchor_weights: Any,
    query_normals: Any,
    *,
    scale: float,
    cell_ratio: float = 1.0 / 150.0,
    sweeps: int = 60,
) -> Tuple[Any, Any]:
    """Propagate observed orientations across a fill domain via coarse voxels.

    Donor-local orientation transfer (k nearest observed donors) is NOISE in
    deep fill regions: the nearest donors of a texel far inside the domain
    all cluster at the same boundary spot, so adjacent fill texels flip
    between unrelated donor sets. Pooling anchor orientation tensors into
    coarse surface voxels and diffusing the TENSOR field over the voxel
    k-NN graph (seeded cells re-anchored each sweep) yields a globally
    combed topology whose neighbor alignment measures |cos| p50 0.999 in
    the deep rear of the face proof (solver-4 G3's construction). Tensor
    (sign-free) diffusion matters: eigenvector signs are arbitrary per
    anchor and averaging vectors directly cancels them.

    Returns (directions[N,3] unit tangent, ok[N] bool) for query_points.
    """
    import numpy as np
    from scipy.spatial import cKDTree

    queries = np.asarray(query_points, dtype=np.float64)
    anchors = np.asarray(anchor_points, dtype=np.float64)
    directions = np.asarray(anchor_directions, dtype=np.float64)
    weights = np.asarray(anchor_weights, dtype=np.float64)
    normals = np.asarray(query_normals, dtype=np.float64)
    n_queries = len(queries)
    if len(anchors) < 16 or n_queries == 0:
        return (np.zeros((n_queries, 3), dtype=np.float32),
                np.zeros(n_queries, dtype=bool))

    cell = float(cell_ratio) * float(scale)
    all_points = np.concatenate([queries, anchors], axis=0)
    keys = np.floor(all_points / max(cell, 1e-9)).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    n_cells = int(inverse.max()) + 1
    query_cell = inverse[:n_queries]
    anchor_cell = inverse[n_queries:]

    centroids = np.zeros((n_cells, 3), np.float64)
    counts = np.bincount(inverse, minlength=n_cells).astype(np.float64)
    for axis_index in range(3):
        centroids[:, axis_index] = np.bincount(
            inverse, weights=all_points[:, axis_index], minlength=n_cells)
    centroids /= np.maximum(counts, 1.0)[:, None]

    tensors = np.zeros((n_cells, 3, 3), np.float64)
    outer = (directions[:, :, None] * directions[:, None, :]
             * np.maximum(weights, 1e-4)[:, None, None])
    for i in range(3):
        for j in range(3):
            np.add.at(tensors[:, i, j], anchor_cell, outer[:, i, j])
    seed_weight = np.zeros(n_cells, np.float64)
    np.add.at(seed_weight, anchor_cell, np.maximum(weights, 1e-4))
    seeded = seed_weight > 0
    trace = np.trace(tensors, axis1=1, axis2=2)
    nonzero = trace > 1e-12
    tensors[nonzero] /= trace[nonzero, None, None]
    seed_tensors = tensors[seeded].copy()

    tree = cKDTree(centroids)
    _, neighbor_index = tree.query(centroids, k=min(10, n_cells), workers=-1)
    neighbor_index = np.atleast_2d(neighbor_index)
    for _ in range(int(sweeps)):
        tensors = tensors[neighbor_index].mean(axis=1)
        # re-anchoring keeps observed orientation authoritative while the
        # diffusion carries it into unseeded cells
        tensors[seeded] = 0.4 * tensors[seeded] + 0.6 * seed_tensors
        trace = np.trace(tensors, axis1=1, axis2=2)
        nonzero = trace > 1e-12
        tensors[nonzero] /= trace[nonzero, None, None]

    fine = tensors[query_cell]
    vec = fine.sum(axis=2) + 1e-9
    for _ in range(12):
        vec = np.einsum("nij,nj->ni", fine, vec)
        vec /= np.maximum(np.linalg.norm(vec, axis=1, keepdims=True), 1e-12)
    vec -= normals * np.einsum("nc,nc->n", vec, normals)[:, None]
    norm = np.linalg.norm(vec, axis=1)
    ok = norm > 1e-6
    vec = np.divide(vec, np.maximum(norm, 1e-6)[:, None])
    return vec.astype(np.float32), ok


def synthesize_fill_detail(
    colors_rgba: Any,
    *,
    positions_texture: Any,
    normals_texture: Optional[Any],
    observed_mask: Any,
    fill_mask: Optional[Any] = None,
    gain: float = 0.7,
    neighbors: int = 6,
    lowpass_sigma: float = 3.0,
    amplitude_sigma: float = 5.0,
    wavelength_texels: float = 2.0,
    octaves: int = 3,
    lic_steps: int = 16,
    seam_feather_texels: float = 6.0,
    max_log_amplitude: float = 0.25,
    color_sigma: float = 0.12,
    seed: int = 11,
    energy_calibration_max: float = 3.0,
    strand_comb: bool = False,
    strand_anisotropy_min: float = 0.40,
    strand_dark_ratio: float = 0.55,
    strand_lic_steps: int = 48,
    strand_amplitude_factor: float = 0.6,
    strand_comb_doublings: int = 8,
    stats_out: Optional[Dict[str, Any]] = None,
) -> Any:
    """Add observed-statistics micro-texture to fill regions.

    Harmonic/KD fill produces the correct average color but ZERO texture
    detail, which reads as a painted wash next to observed content (the
    owner-visible "flat mush" on hidden surfaces). This pass transfers the
    STATISTICS of observed micro-texture — local amplitude and streak
    orientation, per material — without copying structure:

    - Amplitude: per-channel robust (L1) local residual amplitude of
      observed texels around a Gaussian low-pass, transferred to each fill
      texel from its k nearest observed donors in 3D, weighted by normal
      agreement and base-color similarity (hair borrows hair amplitude,
      skin borrows skin amplitude, hull borrows hull amplitude). L1 rather
      than RMS: sparse strong edges (panel lines, feature boundaries)
      dominate an RMS estimate, and noise carrying edge-level contrast
      reads as granite; the L1 statistic tracks the dense stochastic
      component that noise CAN legitimately reproduce. The transferred
      amplitude is further capped at the observed population's p90.
    - Orientation: structure-tensor minor eigenvector of observed
      luminance (the along-streak direction), mapped to 3D through the UV
      tangent frame, transferred by tensor averaging, then imprinted with
      line-integral-convolution smearing of the noise along the direction,
      with streak length proportional to the donors' anisotropy. Hair
      renders as combed streaks, isotropic grain stays isotropic.
    - Carrier: deterministic multi-octave 3D value noise evaluated at
      texel world positions — seamless across UV charts by construction.
      The finest octave sits at ~2 texels: photo micro-texture carries
      most of its gradient energy at 1-3 texel scales, and a coarser
      carrier (the earlier 3-texel default) measured only 0.69x the
      observed per-sigma gradient energy — a structural deficit no gain
      setting could close.
    - CLOSED-LOOP ENERGY CALIBRATION: sigma-matching in log space is an
      OPEN-LOOP proxy for the quality bar that actually judges the fill —
      mean gradient energy of linear luminance, fill vs observed
      (`texture_qa.fill_character`) — and it systematically undershoots.
      Measured decomposition on the starship proof at 1024 (fill/observed
      energy 0.43 at gain 0.7): donor amplitude transfer 0.84x (color-
      similarity weighting selects donors darker/quieter than the
      observed median), carrier frequency 0.69x with the old 3-texel
      finest octave, and base luminance 0.79x (multiplicative log-detail
      on a darker base yields proportionally less LINEAR gradient). No
      per-factor correction closes this robustly — the factors multiply
      differently per asset and resolution — so the pass closes the loop
      on the realized statistic itself: it provisionally APPLIES the
      detail (clip and seam ramp included), measures the resulting fill
      gradient energy on the atlas grid, and solves a single global scale
      so the fill lands at `gain` x the observed total. Scale bounds:
      never below 1 (an already-rich fill — face hair streaks — is never
      dampened), never above `energy_calibration_max`, and never past
      the sigma guard that keeps the fill's log-sigma at or below the
      observed population's BAND-MATCHED residual sigma (residual around
      a lowpass at half the carrier's coarsest wavelength, so the
      comparison covers the same texel band the carrier occupies).
      Gradient parity may not be bought with granite: matching energy
      through amplitude alone on an edge-dominated subject would need
      sigma far above the photo's stochastic band, which reads as noise
      injection; the sigma guard caps the calibration exactly there and
      the shortfall, if any, is reported in `stats_out` rather than
      hidden.
    - Application: multiplicative in log domain (albedo variation scales
      with base brightness), zero-mean (never shifts the fill's average
      color), amplitude-clamped, and feathered to zero at the observed
      seam so real content is never touched.

    STRUCTURE transfer (copying observed residual patches) was prototyped
    and measured worse: coherent shift-map copies produced chaotic panel
    fragments and level-set banding. Statistics transfer cannot invent a
    hairline whorl or a specific panel layout — it makes hidden surface
    read as the same MATERIAL, not the same content; that limitation is
    inherent to any non-generative fill.

    STRAND COMB (`strand_comb=True`, off by default): fill whose donors are
    DARK, strongly-anisotropic streaked material (long-fiber hair) reads as
    leopard mottle under the default statistics — measured on the face
    proof rear at az180: the blotch statistic (residual around a 6 px
    Gaussian at 1000 px renders) is 6.4 with the default pass vs 4.6 for
    the raw membrane, i.e. the noise ADDS rosettes, and the membrane's own
    tone wash carries the rest. Three per-texel changes in the strand
    regime (donor anisotropy >= `strand_anisotropy_min` AND base darker
    than `strand_dark_ratio` x the observed bright-half median):
    - orientation comes from a MULTIGRID-propagated global field
      (`_multigrid_orientation_field`) instead of donor-local averaging —
      donor-local orientation is noise deep inside the fill domain;
    - the carrier keeps only the finest octave, combed with extended LIC
      (`strand_lic_steps`) — the coarse value-noise octaves ARE the
      leopard rosettes; fine carriers also buy more gradient energy per
      contrast unit, so the closed-loop calibration lands at visibly
      lower contrast for the same fill-energy gate;
    - the BASE fill tone is advected along the same field (sparse
      index-doubling kernel, strides 1..2^`strand_comb_doublings` LIC
      steps) so membrane tone blotches elongate into strand-parallel
      streams, and the transferred amplitude is scaled by
      `strand_amplitude_factor` (elongated LOW-contrast statistics).
    Measured on the face proof at 1024: rear blotch 6.4 -> 5.0 (az180),
    7.9 -> 7.4 (az-135) with fill/observed energy 0.93 (gate >= 0.5).
    Off by default: single-photo bakes in the proof set are pinned
    regression canaries this cycle, and the regime is validated on
    multi-view anchors; callers enable it explicitly.

    Deterministic for fixed inputs and seed. Returns the modified RGBA
    array (input unchanged on missing scipy or empty masks). `stats_out`,
    when given, receives the calibration measurements (energies, scale,
    sigma guard) for bake metadata.
    """
    import numpy as np

    rgba = np.asarray(colors_rgba, dtype=np.float32).copy()
    if float(gain) <= 0.0:
        return rgba
    try:
        from scipy.ndimage import distance_transform_edt
        from scipy.spatial import cKDTree
    except Exception:
        return rgba

    positions = np.asarray(positions_texture, dtype=np.float32)
    surface = positions[:, :, 3] > 0.0
    observed = np.asarray(observed_mask, dtype=bool) & surface
    fill = surface & ~observed if fill_mask is None else (np.asarray(fill_mask, dtype=bool) & surface)
    if not fill.any() or int(observed.sum()) < 64:
        return rgba

    rgb = rgba[:, :, :3]
    positions_xyz = positions[:, :, :3]
    if normals_texture is not None:
        normals = np.asarray(normals_texture, dtype=np.float32)[:, :, :3]
        norms = np.linalg.norm(normals, axis=2, keepdims=True)
        normals = np.divide(normals, np.maximum(norms, 1e-8))
    else:
        normals = np.zeros_like(positions_xyz)
        normals[:, :, 2] = 1.0

    # --- observed micro-texture statistics (log domain) ------------------
    eps = 0.02
    log_rgb = np.log(np.clip(rgb, 0.0, 1.0) + eps)
    log_low = np.stack(
        [_masked_gaussian_filter(log_rgb[:, :, c], observed, lowpass_sigma) for c in range(3)],
        axis=2,
    )
    log_res = np.where(observed[:, :, None], log_rgb - log_low, 0.0).astype(np.float32)
    # Robust L1 amplitude (sigma = E|r| / 0.798 for a Gaussian residual).
    amplitude = np.stack(
        [_masked_gaussian_filter(np.abs(log_res[:, :, c]), observed, amplitude_sigma) for c in range(3)],
        axis=2,
    ) / 0.798

    # --- observed streak orientation (structure tensor, UV -> 3D) --------
    lum = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    from scipy.ndimage import gaussian_filter

    lum_s = gaussian_filter(lum.astype(np.float32), 1.2)
    gy, gx = np.gradient(lum_s)
    gx = np.where(observed, gx, 0.0)
    gy = np.where(observed, gy, 0.0)
    jxx = _masked_gaussian_filter(gx * gx, observed, 6.0)
    jxy = _masked_gaussian_filter(gx * gy, observed, 6.0)
    jyy = _masked_gaussian_filter(gy * gy, observed, 6.0)
    trace = jxx + jyy
    disc = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy)
    lam1 = 0.5 * (trace + disc)
    lam2 = 0.5 * (trace - disc)
    anisotropy = np.where(trace > 1e-12, (lam1 - lam2) / (trace + 1e-12), 0.0).astype(np.float32)
    # Major eigenvector (across-streak), then rotate 90 deg -> along-streak.
    ev_u = lam1 - jyy
    ev_v = jxy
    ev_norm = np.hypot(ev_u, ev_v)
    ev_ok = ev_norm > 1e-9
    ev_u = np.where(ev_ok, ev_u / np.maximum(ev_norm, 1e-9), 1.0)
    ev_v = np.where(ev_ok, ev_v / np.maximum(ev_norm, 1e-9), 0.0)
    minor_u = -ev_v
    minor_v = ev_u
    # UV tangent frame: 3D derivative of surface position along atlas axes.
    pitch = _texel_world_pitch(positions, surface)
    d_u = np.zeros_like(positions_xyz)
    d_v = np.zeros_like(positions_xyz)
    d_u[:, 1:-1] = (positions_xyz[:, 2:] - positions_xyz[:, :-2]) * 0.5
    d_v[1:-1, :] = (positions_xyz[2:, :] - positions_xyz[:-2, :]) * 0.5
    ok_u = np.zeros(surface.shape, dtype=bool)
    ok_v = np.zeros(surface.shape, dtype=bool)
    ok_u[:, 1:-1] = surface[:, 2:] & surface[:, :-2]
    ok_v[1:-1, :] = surface[2:, :] & surface[:-2, :]
    ok_u &= np.linalg.norm(d_u, axis=2) < 4.0 * pitch  # cross-chart jumps out
    ok_v &= np.linalg.norm(d_v, axis=2) < 4.0 * pitch
    d_u[~ok_u] = 0.0
    d_v[~ok_v] = 0.0
    dir3 = minor_u[:, :, None] * d_u + minor_v[:, :, None] * d_v
    dir3_norm = np.linalg.norm(dir3, axis=2)
    dir_valid = observed & (dir3_norm > 1e-9)
    dir3 = np.divide(dir3, np.maximum(dir3_norm, 1e-9)[:, :, None])

    # --- transfer statistics to fill texels (surface-nearest donors) -----
    obs_pos = positions_xyz[observed]
    obs_nrm = normals[observed]
    obs_amp = amplitude[observed]
    obs_dir = dir3[observed]
    obs_anis = anisotropy[observed]
    obs_dvalid = dir_valid[observed].astype(np.float32)
    obs_low_lin = (np.exp(log_low) - eps)[observed]
    fill_pos = positions_xyz[fill]
    fill_nrm = normals[fill]
    fill_base = rgb[fill]
    n_fill = len(fill_pos)
    # The full-atlas intermediates above (~0.5 GB at 2048) are no longer
    # needed once their observed-texel extractions exist; releasing them
    # here keeps the long donor-query phase off the bake's RSS peak. The
    # two observed quantiles consumed later are computed from log_res /
    # amplitude before the release (values unchanged: same inputs, same
    # operations, merely reordered ahead of their consumers).
    amp_cap_early = (
        float(np.percentile(amplitude[observed].max(axis=1), 90.0))
        if observed.any() else None
    )
    amp_floor_early = (
        np.percentile(np.abs(log_res[observed]), 25.0, axis=0) / 0.798
        if observed.any() else None
    )
    del log_low, log_res, amplitude, dir3, dir_valid, anisotropy
    del d_u, d_v, jxx, jxy, jyy, trace, disc, lam1, lam2, ev_u, ev_v, gx, gy

    tree = cKDTree(obs_pos)
    kq = int(min(max(1, neighbors), len(obs_pos)))
    distances, indices = _balanced_query(tree, fill_pos, k=kq)
    distances = np.atleast_2d(np.asarray(distances, dtype=np.float32))
    indices = np.atleast_2d(np.asarray(indices, dtype=np.int64))
    if distances.shape[0] == 1 and n_fill > 1:
        distances, indices = distances.T, indices.T

    # Wide distance floor: donor sets deep inside fill regions cluster at
    # the observed boundary, whose residual amplitude is inflated by seam
    # content; the floor averages over the pool instead of trusting the
    # nearest boundary texels.
    w_dist = (1.0 / (distances + 8.0 * pitch)).astype(np.float32)
    agree = np.einsum("nc,nkc->nk", fill_nrm, obs_nrm[indices])
    w_nrm = (0.05 + 0.95 * np.clip(agree, 0.0, 1.0) ** 2).astype(np.float32)
    dcol = np.linalg.norm(obs_low_lin[indices] - fill_base[:, None, :], axis=2)
    w_col = np.exp(-((dcol / float(color_sigma)) ** 2)).astype(np.float32)
    weights = w_dist * w_nrm * (0.02 + 0.98 * w_col)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-8)

    target_amp = np.einsum("nk,nkc->nc", weights, obs_amp[indices]).astype(np.float32)
    if observed.any():
        amp_cap = amp_cap_early
        target_amp = np.minimum(target_amp, amp_cap)
        # Amplitude FLOOR at the observed population's low quantile (per
        # channel): donors imaged at extreme grazing carry artificially low
        # residual amplitude — the same foreshortening that smears their
        # content also erases their micro-texture statistic — and fill
        # anchored by them renders as literal flat plateaus with straight
        # chart-edge boundaries (measured on the starship 2048 bake: a
        # 11k-texel flat cell, texel facet_cellular 0.092 vs the 0.091
        # allowance; with the floor 0.009, fill energy 0.615 -> 0.636, no
        # other gate moved). The floor statistic is the RAW per-texel
        # |residual| quantile, not the Gaussian-smoothed amplitude field:
        # on edge-dominated content (flat panels + sparse lines) the
        # smoothed field spreads line energy over flat texels and a
        # quantile of it would inject line-level noise everywhere, while
        # the raw quantile stays at the flat majority's true (near-zero)
        # stochastic level. No fill may claim to be smoother than the
        # quietest quartile of the witnessed material; the closed-loop
        # energy calibration and sigma guard keep the global level honest.
        amp_floor = amp_floor_early
        target_amp = np.maximum(target_amp, amp_floor[None, :].astype(np.float32))
    target_anis = np.einsum("nk,nk->n", weights, obs_anis[indices]).astype(np.float32)
    # Orientation: sign-invariant tensor average, principal eigenvector by
    # power iteration, projected onto the local tangent plane.
    donor_dirs = obs_dir[indices]
    w_dir = weights * obs_dvalid[indices]
    tensors = np.einsum("nk,nki,nkj->nij", w_dir, donor_dirs, donor_dirs)
    vec = tensors.sum(axis=2) + 1e-9
    for _ in range(8):
        vec = np.einsum("nij,nj->ni", tensors, vec)
        vec /= np.maximum(np.linalg.norm(vec, axis=1, keepdims=True), 1e-12)
    vec -= fill_nrm * np.einsum("nc,nc->n", vec, fill_nrm)[:, None]
    vec_norm = np.linalg.norm(vec, axis=1)
    has_dir = vec_norm > 1e-6
    vec = np.divide(vec, np.maximum(vec_norm, 1e-6)[:, None])

    # --- strand regime (see docstring; empty when strand_comb is off) -----
    strand = np.zeros(n_fill, dtype=bool)
    if strand_comb and n_fill:
        base_lum = (0.299 * fill_base[:, 0] + 0.587 * fill_base[:, 1]
                    + 0.114 * fill_base[:, 2])
        obs_lum = lum[observed]
        bright_median = float(np.median(obs_lum[obs_lum >= np.median(obs_lum)]))
        strand = (target_anis >= float(strand_anisotropy_min)) & (
            base_lum < float(strand_dark_ratio) * max(bright_median, 1e-6))
        if strand.any():
            scale_world = float(np.linalg.norm(
                obs_pos.max(axis=0) - obs_pos.min(axis=0)))
            anchor_sel = (obs_dvalid > 0) & (obs_anis > 0.25)
            global_dir, global_ok = _multigrid_orientation_field(
                fill_pos, obs_pos[anchor_sel], obs_dir[anchor_sel],
                obs_anis[anchor_sel], fill_nrm, scale=scale_world)
            # the global field is applied only inside the strand regime:
            # elsewhere donor-local orientation was measured identical
            # (solver-4 G3), and hair orientation must not leak onto
            # neighboring smooth-material fill
            use_global = strand & global_ok
            vec = np.where(use_global[:, None], global_dir, vec)
            has_dir = has_dir | use_global
            strand = strand & global_ok

    # --- carrier noise + directional smearing (LIC) ----------------------
    noise = np.zeros(n_fill, dtype=np.float32)
    for octave in range(int(octaves)):
        wavelength = float(wavelength_texels) * pitch * (2.0 ** octave)
        component = (1.0 / (1.6 ** octave)) * _fill_value_noise_3d(
            fill_pos, wavelength, seed + octave)
        if octave > 0 and strand.any():
            # strand regime keeps only the finest octave: the coarse
            # value-noise blobs are the leopard rosettes (measured)
            component = np.where(strand, 0.0, component)
        noise += component
    noise /= max(float(noise.std()), 1e-6)

    fill_tree = cKDTree(fill_pos)
    step = 1.5 * pitch
    _, idx_fwd = fill_tree.query(fill_pos + vec * step, k=1, workers=-1)
    _, idx_bwd = fill_tree.query(fill_pos - vec * step, k=1, workers=-1)
    identity = np.arange(n_fill)
    idx_fwd = np.where(has_dir, idx_fwd, identity)
    idx_bwd = np.where(has_dir, idx_bwd, identity)
    steps_per_texel = np.clip(target_anis, 0.0, 1.0) * float(lic_steps)
    total_lic_steps = int(lic_steps)
    if strand.any():
        steps_per_texel = np.where(strand, float(strand_lic_steps), steps_per_texel)
        total_lic_steps = max(total_lic_steps, int(strand_lic_steps))
    lic = noise.copy()
    for iteration in range(total_lic_steps):
        active = steps_per_texel > iteration
        if not active.any():
            break
        smoothed = (lic + 0.85 * (lic[idx_fwd] + lic[idx_bwd])) / 2.7
        lic = np.where(active, smoothed, lic)
    # Directional averaging damps amplitude; restore local unit variance so
    # streaks stay crisp (fall back to global std on near-flat chains).
    local = np.stack(
        [lic, lic[idx_fwd], lic[idx_bwd], lic[idx_fwd][idx_fwd], lic[idx_bwd][idx_bwd]], axis=0
    )
    local_std = local.std(axis=0)
    global_std = max(float(lic.std()), 1e-6)
    lic = lic / np.where(local_std > 0.25 * global_std, local_std, global_std)

    # --- seam feather (needed by the calibration's realized application) --
    dist_to_obs = distance_transform_edt(~observed)
    ramp = np.clip(dist_to_obs[fill] / float(seam_feather_texels), 0.0, 1.0).astype(np.float32)

    # --- strand base-tone combing (see docstring) --------------------------
    if strand.any():
        base = fill_base.astype(np.float32).copy()
        accumulated = base.copy()
        count = np.ones(n_fill, dtype=np.float32)
        forward, backward = idx_fwd.copy(), idx_bwd.copy()
        for _ in range(int(strand_comb_doublings)):
            accumulated = accumulated + base[forward] + base[backward]
            count += 2.0
            forward = forward[forward]
            backward = backward[backward]
        combed = accumulated / count[:, None]
        comb_weight = strand.astype(np.float32) * ramp
        fill_base = base * (1.0 - comb_weight[:, None]) + combed * comb_weight[:, None]
        rgb[fill] = fill_base
        target_amp = np.where(strand[:, None],
                              target_amp * float(strand_amplitude_factor),
                              target_amp)

    # --- closed-loop energy calibration (see docstring) -------------------
    # The realized fill energy is measured with the SAME operator the QA
    # gate uses (Scharr): a central-difference proxy was measured to
    # overestimate the fine-scale carrier's energy by ~2x relative to
    # Scharr's cross-smoothing response, i.e. the operator does NOT cancel
    # between fields of different spectra. Regions are eroded so region-
    # boundary steps never contribute.
    calibration_scale = 1.0
    calibration_stats: Dict[str, Any] = {}
    try:
        from scipy.ndimage import binary_erosion, convolve1d

        def scharr_energy(field2d: Any, inner: Any) -> float:
            f = field2d.astype(np.float32)
            gx_s = convolve1d(convolve1d(f, [1.0, 0.0, -1.0], axis=1),
                              [3.0, 10.0, 3.0], axis=0)
            gy_s = convolve1d(convolve1d(f, [1.0, 0.0, -1.0], axis=0),
                              [3.0, 10.0, 3.0], axis=1)
            return float(np.hypot(gx_s, gy_s)[inner].mean())

        obs_inner = binary_erosion(observed, iterations=2)
        fill_inner = binary_erosion(fill, iterations=2)
        if int(obs_inner.sum()) >= 500 and int(fill_inner.sum()) >= 500:
            lum_lin = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
            e_observed = scharr_energy(np.where(observed, lum_lin, 0.0), obs_inner)
            target_energy = float(gain) * e_observed

            amp_lum = (
                0.299 * target_amp[:, 0] + 0.587 * target_amp[:, 1] + 0.114 * target_amp[:, 2]
            )
            detail_unit = (lic * amp_lum * float(gain)).astype(np.float32)
            log_base_fill = np.log(
                np.clip(0.299 * fill_base[:, 0] + 0.587 * fill_base[:, 1]
                        + 0.114 * fill_base[:, 2], 0.0, 1.0) + eps
            ).astype(np.float32)
            realized = lum_lin.copy()

            def realized_energy(scale: float) -> float:
                d = np.clip(detail_unit * float(scale),
                            -float(max_log_amplitude), float(max_log_amplitude)) * ramp
                realized[fill] = np.clip(np.exp(log_base_fill + d) - eps, 0.0, 1.0)
                return scharr_energy(realized, fill_inner)

            e_base = realized_energy(0.0)
            e_unit = realized_energy(1.0)
            if e_unit > e_base + 1e-9:
                if e_unit >= target_energy:
                    calibration_scale = 1.0
                else:
                    # Secant on the (clip-saturating, monotone) response.
                    s_prev, e_prev = 1.0, e_unit
                    s_cur = min(
                        1.0 + (target_energy - e_unit) / (e_unit - e_base),
                        float(energy_calibration_max),
                    )
                    for _ in range(2):
                        e_cur = realized_energy(s_cur)
                        if abs(e_cur - e_prev) < 1e-9 or abs(e_cur - target_energy) < 1e-4:
                            break
                        s_next = s_cur + (target_energy - e_cur) * (s_cur - s_prev) / (
                            e_cur - e_prev
                        )
                        s_prev, e_prev = s_cur, e_cur
                        s_cur = float(np.clip(s_next, 1.0, float(energy_calibration_max)))
                    calibration_scale = float(
                        np.clip(s_cur, 1.0, float(energy_calibration_max))
                    )

                # sigma guard: the calibrated fill's log-sigma may not
                # exceed the observed band-matched residual sigma.
                band_sigma = 0.5 * float(wavelength_texels) * (2.0 ** (int(octaves) - 1))
                log_lum_lin = np.log(np.clip(lum_lin, 0.0, 1.0) + eps)
                log_low_band = _masked_gaussian_filter(log_lum_lin, observed, band_sigma)
                res_band = np.where(observed, log_lum_lin - log_low_band, 0.0)
                sigma_observed = float(np.abs(res_band[obs_inner]).mean()) / 0.798
                sigma_scaled = float(np.abs(np.clip(
                    detail_unit * calibration_scale,
                    -float(max_log_amplitude), float(max_log_amplitude))).mean()) / 0.798
                if sigma_scaled > sigma_observed > 0.0:
                    calibration_scale = float(np.clip(
                        calibration_scale * sigma_observed / sigma_scaled,
                        1.0, float(energy_calibration_max)))
                calibration_stats = {
                    "observed_energy": round(e_observed, 3),
                    "target_energy": round(target_energy, 3),
                    "fill_energy_at_1": round(e_unit, 3),
                    "fill_energy_base": round(e_base, 3),
                    # realized energy at the FINAL scale: any shortfall
                    # against target_energy (caps, sigma guard, clip
                    # saturation) is visible here, not hidden.
                    "fill_energy_final": round(realized_energy(calibration_scale), 3),
                    "sigma_observed_band": round(sigma_observed, 4),
                    "sigma_at_scale": round(sigma_scaled, 4),
                    "scale": round(calibration_scale, 3),
                }
    except Exception:
        calibration_scale = 1.0
    if stats_out is not None:
        stats_out["energy_calibration"] = calibration_stats or {"scale": 1.0}

    # --- apply: multiplicative, clamped, feathered at the observed seam --
    detail = np.clip(
        lic[:, None] * target_amp * float(gain) * calibration_scale,
        -float(max_log_amplitude), float(max_log_amplitude)
    )
    detail *= ramp[:, None]
    filled = np.exp(np.log(np.clip(fill_base, 0.0, 1.0) + eps) + detail) - eps
    rgba[fill, :3] = np.clip(filled, 0.0, 1.0)
    rgba[fill, 3] = 1.0
    return rgba


def _voxel_neighborhood_mean(points: Any, values: Any, cell: float,
                             select: Optional[Any] = None,
                             octants: Optional[Any] = None) -> Any:
    """Per-point mean of `values` over the 3x3x3 voxel neighborhood.

    World-space box window (span ~[cell, 2*cell] depending on in-cell
    position) via integer voxel binning + separable box sums: O(points),
    no KD-tree, deterministic. `select` restricts which points CONTRIBUTE;
    all points are queried. Returns NaN where the neighborhood is empty.

    `octants` (per-point int 0..5: the DOMINANT normal axis direction,
    argmax |n| with sign — bins +x, -x, +y, -y, +z, -z) makes the
    statistic SHEET-AWARE: contributions are binned per direction and each
    point reads every bin EXCEPT the one opposite its own. A Euclidean
    ball on a thin-shell mesh otherwise mixes the two sides of the shell
    (a hull underside's "context" would include the sunlit topside
    millimeters away through the surface; the face's rear hair would be
    judged against the forward-facing skin through the head). Orthogonal
    bins stay pooled — crease walls and adjacent bulkhead faces ARE
    visible context around a defect. (A Hamming<=1 octant pooling was
    tried first and measurably broke sheet separation: a backward normal
    (-1,0,0) and a forward normal (+1,0,0) sit one sign flip apart, so
    the rear of a head pooled with the face and dark hair fill got lifted
    toward skin tones.)
    """
    import numpy as np
    from scipy.ndimage import uniform_filter

    pts = np.asarray(points, dtype=np.float64)
    lo = pts.min(axis=0)
    dims = np.maximum(((pts.max(axis=0) - lo) / float(cell)).astype(np.int64) + 1, 1)
    ijk = np.clip(((pts - lo) / float(cell)).astype(np.int64), 0, dims - 1)
    spatial = (ijk[:, 0] * dims[1] + ijk[:, 1]) * dims[2] + ijk[:, 2]
    if octants is not None:
        direction = np.asarray(octants, dtype=np.int64)
        flat = spatial * 6 + direction
        grid_shape = (int(dims[0]), int(dims[1]), int(dims[2]), 6)
        window = (3, 3, 3, 1)
    else:
        flat = spatial
        grid_shape = (int(dims[0]), int(dims[1]), int(dims[2]))
        window = 3
    n_cells = int(np.prod(grid_shape))
    contribute = flat if select is None else flat[np.asarray(select, dtype=bool)]
    contribute_values = values if select is None else np.asarray(values)[np.asarray(select, dtype=bool)]
    sums = np.bincount(contribute, weights=contribute_values, minlength=n_cells).reshape(grid_shape)
    counts = np.bincount(contribute, minlength=n_cells).astype(np.float64).reshape(grid_shape)
    sums3 = uniform_filter(sums, size=window, mode="constant") * 27.0
    counts3 = uniform_filter(counts, size=window, mode="constant") * 27.0
    if octants is not None:
        # Direction d (0..5, axis = d//2, sign = d%2) pools every direction
        # except its opposite (same axis, other sign).
        adjacency = np.ones((6, 6), dtype=np.float64)
        for d in range(6):
            adjacency[d, d ^ 1] = 0.0
        sums3 = np.einsum("xyzq,oq->xyzo", sums3, adjacency)
        counts3 = np.einsum("xyzq,oq->xyzo", counts3, adjacency)
    mean3 = np.where(counts3 > 0.5, sums3 / np.maximum(counts3, 1e-9), np.nan)
    return mean3.reshape(-1)[flat]


def _voxel_field_mean_c0(points: Any, values: Any, cell: float,
                         select: Optional[Any] = None) -> Any:
    """C0 (trilinearly interpolated) variant of `_voxel_neighborhood_mean`
    for statistics that become MULTIPLICATIVE CORRECTION FIELDS.

    `_voxel_neighborhood_mean` is piecewise-CONSTANT over its voxel
    lattice: every point in a cell reads the same 3x3x3 box mean, so the
    statistic steps at every cell boundary. That is invisible when the
    statistic only GATES a decision, but a field applied multiplicatively
    to image content prints the lattice onto the texture: measured on the
    car_bo3 live incident, the tone-consensus field (stage 2 of
    `equalize_projection_tone`) saturated its +-0.5 log cap with
    cell-to-cell steps and rendered as rectangular exposure BLOCKS over
    the roof/glass — the "image inside an image" defect class (the voxel
    lattice intersects a flat body panel in axis-aligned world-space
    rectangles; within-cell field variance measured 0.47x the total, i.e.
    most of the field's spatial structure WAS the lattice).

    Construction: the same integer binning and 3x3x3 zero-padded box sums
    as `_voxel_neighborhood_mean`, then evaluation at each query point by
    trilinear interpolation of the CELL-CENTERED means (clamp-to-edge;
    unoccupied corner cells carry zero interpolation weight). At a cell
    center this reproduces the box mean exactly; between centers it is
    continuous by construction, so a field built from it cannot print
    lattice steps at any amplitude. Deterministic, O(points), same NaN
    contract (NaN where no occupied corner is in reach).

    Deliberately NOT a replacement for `_voxel_neighborhood_mean`: the
    gate/decision users (fill floor, film band, disc detection) are
    calibrated on the box statistic and pinned by single-photo canaries;
    only correction-field builders should use this variant.
    """
    import numpy as np
    from scipy.ndimage import uniform_filter

    pts = np.asarray(points, dtype=np.float64)
    lo = pts.min(axis=0)
    step = max(float(cell), 1e-12)
    coords = (pts - lo) / step
    dims = np.maximum(coords.astype(np.int64).max(axis=0) + 1, 1)
    ijk = np.clip(coords.astype(np.int64), 0, dims - 1)
    flat = (ijk[:, 0] * dims[1] + ijk[:, 1]) * dims[2] + ijk[:, 2]
    n_cells = int(np.prod(dims))
    if select is None:
        contribute = flat
        contribute_values = np.asarray(values, dtype=np.float64)
    else:
        chosen = np.asarray(select, dtype=bool)
        contribute = flat[chosen]
        contribute_values = np.asarray(values, dtype=np.float64)[chosen]
    grid_shape = (int(dims[0]), int(dims[1]), int(dims[2]))
    sums = np.bincount(
        contribute, weights=contribute_values, minlength=n_cells
    ).reshape(grid_shape)
    counts = np.bincount(contribute, minlength=n_cells).astype(
        np.float64).reshape(grid_shape)
    sums3 = uniform_filter(sums, size=3, mode="constant") * 27.0
    counts3 = uniform_filter(counts, size=3, mode="constant") * 27.0
    occupied = counts3 > 0.5
    mean3 = np.where(occupied, sums3 / np.maximum(counts3, 1e-9), 0.0)
    mean3_flat = mean3.reshape(-1)
    occupied_flat = occupied.reshape(-1).astype(np.float64)

    centered = coords - 0.5
    base = np.floor(centered).astype(np.int64)
    frac = centered - base
    value_acc = np.zeros(len(pts), dtype=np.float64)
    weight_acc = np.zeros(len(pts), dtype=np.float64)
    for dx in (0, 1):
        wx = frac[:, 0] if dx else 1.0 - frac[:, 0]
        cx = np.clip(base[:, 0] + dx, 0, dims[0] - 1)
        for dy in (0, 1):
            wy = frac[:, 1] if dy else 1.0 - frac[:, 1]
            cy = np.clip(base[:, 1] + dy, 0, dims[1] - 1)
            for dz in (0, 1):
                wz = frac[:, 2] if dz else 1.0 - frac[:, 2]
                cz = np.clip(base[:, 2] + dz, 0, dims[2] - 1)
                corner = (cx * dims[1] + cy) * dims[2] + cz
                weight = wx * wy * wz * occupied_flat[corner]
                value_acc += weight * mean3_flat[corner]
                weight_acc += weight
    out = np.full(len(pts), np.nan, dtype=np.float64)
    reachable = weight_acc > 1e-12
    out[reachable] = value_acc[reachable] / weight_acc[reachable]
    return out


def enforce_fill_luminance_floor(
    colors_rgba: Any,
    *,
    positions_texture: Any,
    surface_mask: Any,
    synthesized_mask: Any,
    normals_texture: Optional[Any] = None,
    donor_mask: Optional[Any] = None,
    context_radius_frac: float = 0.035,
    floor_ratio: float = 0.65,
    anchor_floor_ratio: float = 0.55,
    absolute_floor: float = 26.0 / 255.0,
    minority_full: float = 0.30,
    minority_zero: float = 0.45,
    residual_depth: float = 0.10,
    consensus_depth: float = 1.2,
    max_lift: float = 1.4,
    evidence_headroom: float = 1.35,
) -> Tuple[Any, Dict[str, Any]]:
    """Lift feature-dark pockets in SYNTHESIZED texels to a context floor.

    Provenance of the dark fill fragments this removes (measured on the
    starship/owl single-view bakes): a handful of very dark observed
    anchors — real photo content (intake interiors, shadowed panel joints)
    plus occasional background-adjacent grazing samples — seed the harmonic
    fill, and the maximum-principle solve happily TRANSPORTS that darkness
    across hidden surface far from where the photo justified it. At close
    zoom those pockets read as ink smears against the surrounding hull.
    Observed texels carry photo evidence and are NEVER touched; synthesized
    texels carry none, so a luminance floor relative to their local surface
    context is a legitimate prior, not a cover-up.

    Math (per synthesized texel t, world position p_t, luminance Y_t),
    evaluated at TWO context scales s in {R, 2R} — close-zoom inspection
    judges pockets against a window substantially wider than the pocket
    itself, and a fragment hugging a large dark structure still reads as a
    defect against the wider surroundings of both:

        m1_s(t)  = mean Y of surface texels in the world ball B(p_t, s)
                   (PLAIN mean, matching the defect detector's local
                   windowed reference — a bright-half mean was measured to
                   set the floor above broad legitimate mid-dark fill on
                   the starship and flatten a third of its fill texture)
        dark_s(t)= ball fraction of texels darker than floor_ratio * m1_s
        gate_s   = smoothstep from 1 at dark_s <= minority_full
                   to 0 at dark_s >= minority_zero
        ctx      = max_s(m1_s * gate_s)

    With `normals_texture`, every ball statistic is SHEET-AWARE: points are
    additionally binned by normal octant and each texel reads only its own
    octant's neighborhood (see `_voxel_neighborhood_mean`). Generated
    meshes are thin crusts, and a Euclidean ball otherwise mixes the two
    sides of the crust — a shaded underside would be judged against the
    sunlit topside millimeters away through the surface and lifted toward
    it (measured on the starship: the sheet-blind floor flattened the
    underside fill and dropped the fill-gradient-energy gate 0.57 -> 0.48).

    plus a DONOR-CONSENSUS term over DONOR texels only (`donor_mask`,
    default = everything not synthesized; the bake passes the
    direct-observed set — the anchors the fill is supposed to interpolate,
    "donor validation"): with a_s(t) the ball mean of donor luminance and
    adark_s(t)/agate_s its own dark-minority statistics,

        anchor   = max_s(a_s * agate_s)
        target   = max(absolute_floor * gate,
                       floor_ratio * ctx,
                       anchor_floor_ratio * anchor)

    Application is PER PIXEL in log-luminance with SATURATING depth,
    restricted to COMPACT dark components. With d = max(0, log target -
    log Y) the pixel's depth below the floor:

        d'     = residual_depth * (1 - exp(-d / residual_depth))
        lift   = min(d - d', max_lift)
        log Y' = log Y + lift        (synthesized texels only)

    d' is the depth that REMAINS after the lift: monotone increasing in d
    (depth ordering preserved — no posterization plateau at the floor),
    slope 1 at d = 0 (pixels just under the floor are barely touched, so
    no visible boundary forms at the floor line), and bounded by
    `residual_depth` (every lifted pixel ends within
    exp(-residual_depth) of its floor). A base/residual split (ball
    log-mean base lifted, texel residual preserved verbatim) was measured
    to leak: dark BANDS wider than the base radius but narrower than the
    context ball (the owl's synthesized crease bands) hide their deficit
    in the residual and ship unchanged. Saturating per-pixel depth closes
    that route by construction; the gates above exempt extended legitimate
    dark structure BEFORE any lift is computed.

    (Two compactness restrictions — atlas connected components with a
    minimum size, and a 3D deep-texel density test — were prototyped to
    narrow the lift to "blob-shaped" defects and reverted: both left the
    edges of real pockets under the floor, which the close-zoom detector
    then flagged as fresh smaller fragments. The sheet-aware minority
    gates above are the load-bearing protection for legitimate texture;
    with them, lifting every gated under-floor texel costs < 0.02 of the
    fill-gradient-energy ratio on the starship.)

    Detector-margin argument for the defaults (the close-zoom QA flags
    synthesized pixels with Y < max(22/255, 0.45 * window_mean), window
    mean taken over a render window comparable to the R..2R ball): the
    guaranteed post-floor luminance approaches exp(-residual_depth) *
    floor_ratio * m1 = 0.905 * 0.65 * m1 ~ 0.59 * m1. The headroom over
    the detector's 0.45 ratio is deliberate: the render window mean mixes
    brighter content than the sheet-aware ball (other sheets visible in
    the same window, view-dependent shading) — measured on the owl's wing
    creases, the render reference exceeds the octant ball mean by up to
    ~1.2x, and 0.59 * m1 > 0.45 * 1.2 * m1 still clears the line. Pixels
    deeper than `consensus_depth` below the floor additionally blend
    toward the context consensus color (bright-half mean color scaled to
    the target luminance): near-black 8-bit pixels carry no usable chroma
    and multiplying them only amplifies quantization noise.

    The two terms catch the two measured defect classes. The context floor
    catches pockets a viewer sees against bright surroundings. The anchor
    floor catches TRANSPORTED and INTERPOLATED darkness: synthesized texels
    substantially darker than the observed donors around them (the harmonic
    solve moves darkness along the graph from remote dark anchors; measured
    on the starship underside: blob luminance 24-26 vs nearest-anchor
    61-99, and on the owl wing creases: 61-85 vs donors 73-113). Its ratio
    is deliberately STRICTER than the context ratio — in a bright-majority
    donor neighborhood (agate), a harmonic interpolant far below the local
    donor consensus is by construction importing far-away darkness, while
    the maximum principle bounds legitimate local interpolation near the
    local donor range. Legitimate continuations survive both terms: their
    own donors are just as dark as they are (agate engages the dark-donor
    minority test at floor_ratio * a, so moderately shaded donors do not
    disable the gate).

    Discriminators that protect legitimate dark content:

    - uniformly dark regions (hair mass, shaded hull side): m1 tracks the
      local darkness, target falls below Y, lift is exactly zero;
    - EXTENDED dark structure (hairline shadow bands, large intake
      interiors): the minority gate — a defect pocket is by definition a
      local anomaly, so when the context ball itself is substantially dark
      (dark fraction above `minority_zero`) the darkness is a regional
      material/shading statement and the floor stands down (measured: an
      ungated floor turned the face's hairline shadow band into a pale
      film and tripped the pale_film gate; the gate removes exactly that
      failure while keeping the ship/owl pocket lifts);
    - moderately dark detail (panel joints, synthesized micro-texture
      whose depth stays above ~floor_ratio of the bright context): d = 0,
      bit-identical. Deeper thin lines compress but remain locally darkest
      (ordering preserved); measured fill-region Scharr edge energy on the
      starship changes < +1%;
    - observed texels: excluded from the mask entirely.

    Implementation: voxel binning + 3x3x3 box sums (O(texels), no KD-tree,
    deterministic). Returns (rgba, stats); unchanged input when scipy is
    missing or nothing qualifies.
    """
    import numpy as np

    out = np.asarray(colors_rgba, dtype=np.float32).copy()
    surface = np.asarray(surface_mask, dtype=bool)
    synthesized = np.asarray(synthesized_mask, dtype=bool) & surface
    stats: Dict[str, Any] = {"applied": False, "lifted_texels": 0}
    if not synthesized.any():
        return out, stats
    try:
        from scipy.ndimage import uniform_filter  # noqa: F401 (probe import)
    except Exception:
        return out, stats

    eps = 1e-3
    rgb = out[:, :, :3]
    lum = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    positions = np.asarray(positions_texture, dtype=np.float32)[:, :, :3]
    surf_pts = positions[surface].astype(np.float64)
    surf_lum = lum[surface].astype(np.float64)
    covered = positions[surface]
    diagonal = float(np.linalg.norm(covered.max(axis=0) - covered.min(axis=0))) or 1.0
    synthesized_in_surface = synthesized[surface]

    octants: Optional[Any] = None
    if normals_texture is not None:
        normal_vectors = np.asarray(normals_texture, dtype=np.float32)[:, :, :3][surface]
        axis = np.abs(normal_vectors).argmax(axis=1)
        sign = np.take_along_axis(normal_vectors, axis[:, None], axis=1)[:, 0] < 0
        octants = axis.astype(np.int64) * 2 + sign.astype(np.int64)

    surf_rgb = rgb[surface].astype(np.float64)
    if donor_mask is not None:
        observed_in_surface = (np.asarray(donor_mask, dtype=bool) & surface)[surface]
    else:
        observed_in_surface = ~synthesized_in_surface
    span = max(float(minority_zero) - float(minority_full), 1e-6)
    gated_context = np.zeros(len(surf_pts), dtype=np.float64)
    gate_context = np.zeros(len(surf_pts), dtype=np.float64)
    gated_anchor = np.zeros(len(surf_pts), dtype=np.float64)
    gate_anchor = np.zeros(len(surf_pts), dtype=np.float64)
    context_rgb = np.zeros((len(surf_pts), 3), dtype=np.float64)

    def minority_gate(dark_fraction: Any) -> Any:
        clean = np.where(np.isfinite(dark_fraction), dark_fraction, 1.0)
        return np.clip((float(minority_zero) - clean) / span, 0.0, 1.0)

    for scale in (1.0, 2.0):
        context_cell = float(context_radius_frac) * diagonal * scale
        # --- context floor: PLAIN ball mean of all surface texels ----------
        m1 = _voxel_neighborhood_mean(surf_pts, surf_lum, context_cell, octants=octants)
        m1 = np.where(np.isfinite(m1), m1, surf_lum)
        dark_fraction = _voxel_neighborhood_mean(
            surf_pts, (surf_lum < float(floor_ratio) * m1).astype(np.float64), context_cell,
            octants=octants,
        )
        gate = minority_gate(dark_fraction)
        candidate = m1 * gate
        better = candidate > gated_context
        gated_context = np.where(better, candidate, gated_context)
        gate_context = np.where(better, gate, gate_context)
        if better.any():
            # Consensus color: bright-half mean (represents the material a
            # deep pocket should blend toward, not the pocket itself).
            bright = surf_lum >= m1
            for channel in range(3):
                channel_mean = _voxel_neighborhood_mean(
                    surf_pts, surf_rgb[:, channel], context_cell, select=bright,
                    octants=octants,
                )
                channel_mean = np.where(np.isfinite(channel_mean), channel_mean, m1)
                context_rgb[better, channel] = channel_mean[better]
        # --- donor-consensus floor: OBSERVED anchors only ------------------
        if observed_in_surface.any():
            anchor_mean = _voxel_neighborhood_mean(
                surf_pts, surf_lum, context_cell, select=observed_in_surface,
                octants=octants,
            )
            anchor_dark = _voxel_neighborhood_mean(
                surf_pts,
                (surf_lum < float(floor_ratio) * np.where(np.isfinite(anchor_mean), anchor_mean, 0.0))
                .astype(np.float64) * observed_in_surface,
                context_cell,
                select=observed_in_surface,
                octants=octants,
            )
            agate = minority_gate(anchor_dark)
            acand = np.where(np.isfinite(anchor_mean), anchor_mean, 0.0) * agate
            abetter = acand > gated_anchor
            gated_anchor = np.where(abetter, acand, gated_anchor)
            gate_anchor = np.where(abetter, agate, gate_anchor)

    overall_gate = np.maximum(gate_context, gate_anchor)
    target = np.maximum(
        float(absolute_floor) * overall_gate,
        np.maximum(
            float(floor_ratio) * gated_context,
            float(anchor_floor_ratio) * gated_anchor,
        ),
    )

    # DARK-EVIDENCE EXEMPTION (per connected dark component, in 3D):
    # a synthesized dark region that CONNECTS to observed texels of the
    # same darkness AND TRACKS their tone is a witnessed feature continued
    # into hidden surface (the owl's wing markings), not a transported
    # pocket. Lifting it to the context floor was measured to (a) flatten
    # the marking and (b) manufacture a tone seam against its own observed
    # core (owl observed|fill seam p95 29 -> 52..60 at both resolutions).
    # Exemption requires BOTH conditions per component:
    #   evidence:  >= 8 observed texels inside the dark component
    #   tracking:  mean synthesized luminance <= evidence_headroom *
    #              mean observed (evidence) luminance
    # The tracking test is the discriminator that keeps the starship's
    # engine-halo smears un-exempt (measured: their synthesized part sits
    # at 2..3.7x the near-black cavity tone — an intermediate smear that
    # follows NOTHING — while the owl's marking fill sits at 0.9x its
    # core). Exempt components keep their own tone; everything else gets
    # the full context/donor floor.
    dark_component_target = np.full(len(surf_pts), np.inf)
    try:
        from scipy import ndimage as _ndimage

        pair_cols = surface[:, :-1] & surface[:, 1:]
        if pair_cols.any():
            pos_map = np.asarray(positions_texture, dtype=np.float32)[:, :, :3]
            steps = np.linalg.norm(
                pos_map[:, 1:][pair_cols] - pos_map[:, :-1][pair_cols], axis=1
            )
            pitch = float(np.median(steps)) + 1e-12
        else:
            pitch = float(diagonal) / 1024.0
        dark_sel = surf_lum < np.maximum(target, float(absolute_floor))
        if dark_sel.any():
            # Connectivity granularity is a WORLD scale (floored by the
            # texel pitch): with a pitch-proportional cell, the same dark
            # marking that is one connected component at 1024 fragments at
            # 2048 (half the world cell), loses its observed evidence, and
            # gets lifted — measured as an owl seam regression appearing
            # ONLY at 2048 (p95 39 at 1024 vs 60 at 2048).
            cell = max(2.0 * pitch, 0.003 * float(diagonal), 1e-9)
            lo = surf_pts.min(axis=0)
            dims = np.maximum(((surf_pts.max(axis=0) - lo) / cell).astype(np.int64) + 1, 1)
            if int(dims.prod()) <= 64_000_000:
                ijk = np.clip(((surf_pts[dark_sel] - lo) / cell).astype(np.int64), 0, dims - 1)
                occupancy = np.zeros(tuple(dims), dtype=bool)
                occupancy[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True
                labels, count = _ndimage.label(occupancy, structure=np.ones((3, 3, 3), bool))
                if count > 0:
                    texel_label = labels[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
                    donor_dark = observed_in_surface[dark_sel]
                    synth_dark = synthesized_in_surface[dark_sel]
                    evidence_sum = np.bincount(
                        texel_label[donor_dark],
                        weights=surf_lum[dark_sel][donor_dark],
                        minlength=count + 1,
                    )
                    evidence_count = np.bincount(
                        texel_label[donor_dark], minlength=count + 1
                    )
                    synth_sum = np.bincount(
                        texel_label[synth_dark],
                        weights=surf_lum[dark_sel][synth_dark],
                        minlength=count + 1,
                    )
                    synth_count = np.bincount(
                        texel_label[synth_dark], minlength=count + 1
                    )
                    evidence_tone = evidence_sum / np.maximum(evidence_count, 1)
                    synth_tone = synth_sum / np.maximum(synth_count, 1)
                    exempt = (
                        (evidence_count >= 8)
                        & (synth_count > 0)
                        & (synth_tone <= float(evidence_headroom) * evidence_tone)
                    )
                    component_target = np.where(
                        exempt[texel_label],
                        float(evidence_headroom) * evidence_tone[texel_label],
                        np.inf,
                    )
                    scatter = np.full(len(surf_pts), np.inf)
                    scatter[np.nonzero(dark_sel)[0]] = component_target
                    dark_component_target = scatter
                    stats["evidence_exempt_components"] = int(exempt.sum())
    except Exception:
        pass
    target = np.minimum(target, np.maximum(dark_component_target, surf_lum))

    # Saturating per-pixel depth compression (see docstring).
    depth = np.maximum(
        0.0, np.log(np.maximum(target, eps)) - np.log(np.maximum(surf_lum, eps))
    )
    depth = np.where(synthesized_in_surface, depth, 0.0)

    remaining = float(residual_depth) * (1.0 - np.exp(-depth / float(residual_depth)))
    lift = np.minimum(depth - remaining, float(max_lift))
    lifted = lift > 1e-4
    if not lifted.any():
        return out, stats

    # Consensus blend for pixels far below the floor: the context
    # bright-half mean COLOR scaled to the target luminance. (Texels whose
    # context branch never engaged have no consensus color; they keep the
    # multiplicative lift alone.)
    context_lum = (
        0.299 * context_rgb[:, 0] + 0.587 * context_rgb[:, 1] + 0.114 * context_rgb[:, 2]
    )
    overshoot = np.clip((depth - float(consensus_depth)) / 0.5, 0.0, 1.0)
    overshoot = np.where(context_lum > eps, overshoot, 0.0)
    consensus_rgb = context_rgb * (
        target / np.maximum(context_lum, eps)
    )[:, None]

    stats.update(
        applied=True,
        lifted_texels=int(lifted.sum()),
        mean_lift=round(float(lift[lifted].mean()), 4),
        p99_lift=round(float(np.percentile(lift[lifted], 99)), 4),
        consensus_texels=int((overshoot > 0.0).sum()),
    )
    rows, cols = np.nonzero(surface)
    scale = np.exp(lift).astype(np.float32)
    multiplied = out[rows, cols, :3] * scale[:, None]
    blended = (
        multiplied * (1.0 - overshoot[:, None])
        + consensus_rgb.astype(np.float32) * overshoot[:, None]
    )
    out[rows, cols, :3] = np.clip(blended, 0.0, 1.0).astype(np.float32)
    return out, stats


def inpaint_unseen_texels(
    colors_rgba: Any,
    *,
    surface_mask: Any,
    observed_mask: Any,
    positions_texture: Optional[Any] = None,
    normals_texture: Optional[Any] = None,
    neighbors: int = 16,
) -> Any:
    """Fill surface texels that no view observed.

    The fill must happen in 3D surface space, not UV space: xatlas packs
    unrelated charts side by side, so any UV-space propagation (diffusion
    inpainting, pooled color fields) bleeds colors across chart boundaries
    and produces patchwork noise on hidden regions. This is the same
    conclusion the upstream Hunyuan texture pipeline reached (its
    `mesh_inpaint_processor` diffuses colors over the mesh graph, not the
    atlas).

    For each unseen texel with a world position, colors are borrowed from
    its `neighbors` nearest observed texels in 3D with inverse-distance
    weights (soft floor at a quarter of the median nearest-neighbor
    distance), further weighted by surface-normal agreement when normals are
    available. The normal term keeps material regions coherent: the hidden
    back of a head borrows from observed hair (backward-facing normals)
    rather than from the nearer face skin (forward-facing normals), because
    proximity across a material boundary is usually proximity across an
    orientation change too. Falls back to a UV-space EDT fill only when no
    positions are available.
    """
    import numpy as np

    rgba = np.asarray(colors_rgba, dtype=np.float32).copy()
    surface = np.asarray(surface_mask, dtype=bool)
    observed = np.asarray(observed_mask, dtype=bool) & surface
    unseen = surface & ~observed
    if not unseen.any():
        return rgba
    if not observed.any():
        # Nothing was observed at all: neutral mid-gray keeps the asset usable.
        rgba[surface, :3] = 0.5
        rgba[surface, 3] = 1.0
        return rgba

    if positions_texture is not None:
        try:
            from scipy.spatial import cKDTree

            positions = np.asarray(positions_texture, dtype=np.float32)[:, :, :3]
            observed_points = positions[observed]
            observed_colors = rgba[observed][:, :3]
            unseen_points = positions[unseen]
            k = int(min(max(1, neighbors), len(observed_points)))
            tree = cKDTree(observed_points)
            distances, indices = tree.query(unseen_points, k=k, workers=-1)
            distances = np.atleast_2d(np.asarray(distances, dtype=np.float32))
            indices = np.atleast_2d(np.asarray(indices, dtype=np.int64))
            if distances.shape[0] == 1 and len(unseen_points) > 1:
                distances = distances.T
                indices = indices.T
            # Inverse-distance weights with a soft floor so exact hits do not
            # produce infinities and far regions average smoothly.
            scale = float(np.median(distances[:, 0])) + 1e-6
            weights = (1.0 / (distances + 0.25 * scale)).astype(np.float32)
            if normals_texture is not None:
                normals = np.asarray(normals_texture, dtype=np.float32)[:, :, :3]
                norms = np.linalg.norm(normals, axis=2, keepdims=True)
                normals = np.divide(normals, np.maximum(norms, 1e-8))
                observed_normals = normals[observed]
                unseen_normals = normals[unseen]
                agreement = np.einsum(
                    "nc,nkc->nk", unseen_normals, observed_normals[indices]
                )
                # Same-facing neighbors keep full weight; orthogonal or
                # opposite-facing ones fade out. The 0.1 floor prevents a
                # fully-disagreeing neighborhood from zeroing out entirely.
                weights = weights * (0.1 + 0.9 * np.clip(agreement, 0.0, 1.0) ** 2)
            weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-8)
            filled = (observed_colors[indices] * weights[:, :, None]).sum(axis=1)
            rgba[unseen, :3] = filled
            rgba[unseen, 3] = 1.0
            return rgba
        except Exception:
            pass

    try:
        from scipy.ndimage import distance_transform_edt

        _, nearest = distance_transform_edt(~observed, return_indices=True)
        rgba[unseen, :3] = rgba[nearest[0][unseen], nearest[1][unseen], :3]
        rgba[unseen, 3] = 1.0
        return rgba
    except Exception:
        mean_color = rgba[observed, :3].mean(axis=0)
        rgba[unseen, :3] = mean_color
        rgba[unseen, 3] = 1.0
        return rgba


def _shading_sh_basis(normals: Any) -> Any:
    """Order-2 real spherical-harmonics monomial basis evaluated at normals.

    Irradiance from ANY distant light distribution on a convex Lambertian
    surface lies almost entirely (>99% of energy) in this 9-dimensional span
    of the surface normal (Ramamoorthi & Hanrahan 2001), which makes it the
    right low-frequency model for baked-in photo shading as a function of
    the normal. Plain monomial scaling (constants folded into coefficients).
    """
    import numpy as np

    n = np.asarray(normals, dtype=np.float64)
    nx, ny, nz = n[..., 0], n[..., 1], n[..., 2]
    return np.stack(
        [
            np.ones_like(nx), nx, ny, nz,
            nx * ny, ny * nz, 3.0 * nz * nz - 1.0, nx * nz, nx * nx - ny * ny,
        ],
        axis=-1,
    )


def delight_projections(
    projections: Sequence[Dict[str, Any]],
    *,
    normals_texture: Any,
    positions_texture: Optional[Any] = None,
    source_index: int = 0,
    min_overlap_texels: int = 400,
    weight_floor: float = 0.05,
    luminance_floor: float = 0.02,
    max_log_amplitude: float = 1.0,
    ridge: float = 1e-4,
    huber_k: float = 2.5,
    improvement_margin: float = 0.002,
    fade_radius_frac: float = 0.06,
    fade_density_full: float = 0.12,
) -> Dict[str, Any]:
    """Remove per-view baked-in shading DIFFERENCES before blending.

    Photos carry their own lighting: a front photo lit from front-left and a
    profile shot lit flat disagree on every shared surface point even when
    both are perfectly registered, and the disagreement is a smooth function
    of the surface NORMAL, not of the content. Downstream this shows up as
    tone steps at view handoffs that seam leveling can only bridge, never
    remove, and as doubled shading under viewer relighting.

    Model: Lambertian image formation I_v(p) = A(p) * S_v(n(p)) with albedo
    A shared across views. On OVERLAP texels (both views see the same
    surface point) the log-luminance ratio cancels the albedo exactly:

        log Y_u(p) - log Y_v(p) = B(n(p)) . (c_u - c_v)

    with log S_v(n) ~ B(n) . c_v in the order-2 SH basis (`_shading_sh_basis`).
    A joint weighted ridge least-squares over all overlapping view pairs,
    with the gauge c_source = 0, recovers each reference's shading RELATIVE
    to the source photo's light; dividing the reference by exp(B(n) . c_v)
    relights it to the source. The lighting component COMMON to all views is
    mathematically unobservable from ratios (the classic albedo/shading
    ambiguity), so the source's own shading is the reference light — the
    same identity contract the rest of the pipeline keeps (the source photo
    is ground truth at its own pose). This strictly generalizes the DC
    exposure gain in `harmonize_and_gate_projection` (its per-channel gain
    is the order-0 term; orders 1-2 add the normal-dependent field) and
    runs BEFORE it, so the gain gate sees delighted colors and handles only
    residual white balance.

    Robustness, mirroring the exposure gate's revert-on-confound design:

    - luminance-only: one scalar field per view multiplies RGB equally, so
      chroma (freckles, hull markings) is untouched; genuine albedo detail
      is also high-frequency in normal space and outside the SH span;
    - Huber IRLS with MAD-adaptive threshold: content outliers in the
      overlap (hair pixels over skin from misregistration) sit far outside
      the fitted-shading residual cloud and are downweighted, while
      legitimately strong shading ratios stay full-weight inliers;
    - extrapolation clip: a reference's exclusive texels carry normals the
      overlap never constrained, so the field is clipped to the overlap's
      own [p1, p99] range (+0.1 headroom) before the absolute amplitude cap;
    - OVERLAP-PROXIMITY FADE (with `positions_texture`): the correction is
      applied fully near the overlap surface (where cross-view consistency
      is measurable and seams form) and fades to zero deep inside the
      reference's EXCLUSIVE territory. There the reference photo is the
      only witness, and per-view identity — the render at that photo's
      pose must match THAT photo, under its own lighting — outranks
      consistency with a light the viewer never sees from there. An
      unfaded correction was measured end-to-end to relight a profile's
      whole exclusive side (identity MAE 26.4 -> 39.5 against its own
      photo) while helping only the handoff band; the fade keeps the
      handoff fix and leaves view centers bit-identical. Fade =
      clip(overlap density in the world ball B(p, fade_radius_frac *
      diagonal) / fade_density_full, 0, 1), computed with the same voxel
      ball statistics as the fill floor;
    - revert-on-confound: the correction is kept per reference only when it
      REDUCES that reference's overlap disagreement against its fitted
      partners by more than `improvement_margin`; otherwise the view
      reverts unchanged. For GENERATED views the partners are witness-
      RANKED (real photos outrank generated-mutual, the same rule as
      `equalize_projection_tone`): a relight that worsens real-photo
      agreement never ships, one that improves it ships even at
      generated-mutual cost — measured on the fresh-draw car, the
      unranked aggregate let a side reference relight toward two other
      generated views' invented lighting with zero real overlap in its
      gate mass, and source-pose fidelity paid for it.

    Mutates projection rgba in place (weights untouched) and returns stats
    (per-view disagreement before/after, field amplitude, applied flags).
    """
    import numpy as np

    stats: Dict[str, Any] = {"applied": False, "views": []}
    view_count = len(projections)
    if view_count < 2:
        return stats

    normals = np.asarray(normals_texture, dtype=np.float64)[:, :, :3]
    norm = np.linalg.norm(normals, axis=2, keepdims=True)
    normals = np.divide(normals, np.maximum(norm, 1e-8))

    weights = [np.asarray(p["weight"], dtype=np.float32) for p in projections]
    rgbs = [np.asarray(p["rgba"], dtype=np.float32)[:, :, :3] for p in projections]
    lums = [0.299 * c[:, :, 0] + 0.587 * c[:, :, 1] + 0.114 * c[:, :, 2] for c in rgbs]

    # ---- pairwise ratio equations on overlap texels -----------------------
    rows_basis: List[Any] = []
    rows_pair: List[Any] = []
    rows_target: List[Any] = []
    rows_weight: List[Any] = []
    for u in range(view_count):
        for v in range(u + 1, view_count):
            overlap = (weights[u] > weight_floor) & (weights[v] > weight_floor)
            overlap &= (lums[u] > luminance_floor) & (lums[v] > luminance_floor)
            count = int(overlap.sum())
            if count < int(min_overlap_texels):
                continue
            rows_basis.append(_shading_sh_basis(normals[overlap]))
            rows_pair.append(np.full(count, u * view_count + v, dtype=np.int64))
            rows_target.append(
                np.log(lums[u][overlap].astype(np.float64))
                - np.log(lums[v][overlap].astype(np.float64))
            )
            rows_weight.append(
                np.minimum(weights[u][overlap], weights[v][overlap]).astype(np.float64)
            )
    if not rows_basis:
        return stats

    basis_all = np.concatenate(rows_basis, axis=0)
    pair_all = np.concatenate(rows_pair, axis=0)
    target_all = np.concatenate(rows_target, axis=0)
    weight_all = np.concatenate(rows_weight, axis=0)

    n_basis = basis_all.shape[1]
    free = [i for i in range(view_count) if i != int(source_index)]
    col_of = {view: k for k, view in enumerate(free)}
    n_unknown = len(free) * n_basis

    pair_u = pair_all // view_count
    pair_v = pair_all % view_count
    design = np.zeros((len(pair_all), n_unknown), dtype=np.float64)
    for view, col in col_of.items():
        sel_u = pair_u == view
        sel_v = pair_v == view
        if sel_u.any():
            design[sel_u, col * n_basis:(col + 1) * n_basis] += basis_all[sel_u]
        if sel_v.any():
            design[sel_v, col * n_basis:(col + 1) * n_basis] -= basis_all[sel_v]

    # ---- Huber IRLS ridge solve (MAD-adaptive threshold) -------------------
    sample_weight = weight_all.copy()
    coeffs = np.zeros(n_unknown)
    for _ in range(3):
        weighted_design = design * sample_weight[:, None]
        gram = design.T @ weighted_design + float(ridge) * len(design) * np.eye(n_unknown)
        rhs = weighted_design.T @ target_all
        try:
            coeffs = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            return stats
        residual = design @ coeffs - target_all
        mad = float(np.median(np.abs(residual - np.median(residual)))) + 1e-6
        delta = float(huber_k) * 1.4826 * mad
        sample_weight = weight_all * np.where(
            np.abs(residual) <= delta, 1.0, delta / np.maximum(np.abs(residual), 1e-9)
        )

    # ---- per-view field, clipped + faded; apply behind the improvement gate
    src = int(source_index)
    basis_full = _shading_sh_basis(normals)
    applied_any = False
    surface_points: Optional[Any] = None
    surface_sel: Optional[Any] = None
    diagonal = 1.0
    if positions_texture is not None:
        positions_arr = np.asarray(positions_texture, dtype=np.float32)
        surface_sel = positions_arr[:, :, 3] > 0.0
        surface_points = positions_arr[:, :, :3][surface_sel].astype(np.float64)
        if len(surface_points) > 0:
            diagonal = float(
                np.linalg.norm(surface_points.max(axis=0) - surface_points.min(axis=0))
            ) or 1.0
    for view, col in col_of.items():
        field = basis_full @ coeffs[col * n_basis:(col + 1) * n_basis]
        # CONSENSUS APPLICATION (adversarial round 3): the joint fit above
        # already constrains every fitted view through ALL its overlapping
        # pairs (side|back rows included) — but applying only where a view
        # overlaps the SOURCE structurally excluded far-side views (a back
        # view can never share a texel with the source: the facing cones
        # end at ~66 and start at ~101 degrees), and the source-keyed fade
        # zeroed the sides' corrections exactly at the side|back junction
        # where the seams form. The application predicate is now the UNION
        # of the view's fitted overlaps: every view the solver actually
        # constrained gets its relight, faded by proximity to the surface
        # that constrained it, and the revert gate measures disagreement
        # against ALL its overlap partners.
        source_overlap = (weights[src] > weight_floor) & (weights[view] > weight_floor)
        fit_overlap = np.zeros_like(source_overlap)
        for other in range(view_count):
            if other == view:
                continue
            fit_overlap |= (
                (weights[other] > weight_floor) & (weights[view] > weight_floor)
            )
        row: Dict[str, Any] = {
            "index": view,
            "label": projections[view].get("label"),
            "overlap_texels": int(source_overlap.sum()),
            "fit_overlap_texels": int(fit_overlap.sum()),
            "applied": False,
        }
        if int(fit_overlap.sum()) < int(min_overlap_texels):
            stats["views"].append(row)
            continue
        # Views whose gauge reaches the source only through a CHAIN of
        # references (no direct source overlap) inherit the chain's error:
        # halve their amplitude budget.
        chained = int(source_overlap.sum()) < int(min_overlap_texels)
        amplitude = float(max_log_amplitude) * (0.5 if chained else 1.0)
        row["chained_gauge"] = chained
        low = float(np.percentile(field[fit_overlap], 1)) - 0.1
        high = float(np.percentile(field[fit_overlap], 99)) + 0.1
        field = np.clip(np.clip(field, low, high), -amplitude, amplitude)
        if surface_points is not None and surface_sel is not None and len(surface_points) > 0:
            # Overlap-proximity fade over the 3D surface (see docstring).
            # The density is the C0 (interpolated) voxel statistic: the
            # box-mean variant is piecewise-constant over its voxel
            # lattice, and a fade with lattice steps multiplies the field
            # into rectangular exposure blocks on flat surfaces (the
            # car_bo3 roof-block incident; see `_voxel_field_mean_c0`).
            # The former hard `fade[fit_overlap] = 1.0` override is gone
            # for the same reason: forcing full strength on scattered
            # overlap texels steps against their faded neighbors — dense
            # overlap bands reach fade 1.0 through their own density
            # (full fade at 12% ball occupancy), sparse slivers now
            # honestly keep partial correction.
            overlap_indicator = fit_overlap[surface_sel].astype(np.float64)
            density = _voxel_field_mean_c0(
                surface_points,
                overlap_indicator,
                max(float(fade_radius_frac) * diagonal, 1e-9),
            )
            density = np.where(np.isfinite(density), density, 0.0)
            fade_surface = np.clip(density / max(float(fade_density_full), 1e-9), 0.0, 1.0)
            fade = np.zeros(field.shape, dtype=np.float64)
            fade[surface_sel] = fade_surface
            field = field * fade
        correction = np.exp(-field).astype(np.float32)
        corrected = np.clip(rgbs[view] * correction[:, :, None], 0.0, 1.0)
        # Revert gate over ALL fitted partners, weighted like the fit.
        # Partners are CLASSED (real photos vs generated views) so that
        # generated views get the witness-RANKED rule below; real
        # reference views keep the aggregate (real photos reconciling
        # among themselves have no subordinate class).
        class_sums = {"real": [0.0, 0.0, 0.0], "generated": [0.0, 0.0, 0.0]}
        for other in range(view_count):
            if other == view:
                continue
            pair_overlap = (
                (weights[other] > weight_floor) & (weights[view] > weight_floor)
            )
            if not pair_overlap.any():
                continue
            pair_weight = float(np.minimum(
                weights[other][pair_overlap], weights[view][pair_overlap]).sum())
            klass = (
                "generated" if projections[other].get("generated") else "real")
            class_sums[klass][0] += pair_weight * float(
                np.abs(rgbs[view][pair_overlap] - rgbs[other][pair_overlap]).mean())
            class_sums[klass][1] += pair_weight * float(
                np.abs(corrected[pair_overlap] - rgbs[other][pair_overlap]).mean())
            class_sums[klass][2] += pair_weight
        den = class_sums["real"][2] + class_sums["generated"][2]
        before = (
            class_sums["real"][0] + class_sums["generated"][0]) / max(den, 1e-9)
        after = (
            class_sums["real"][1] + class_sums["generated"][1]) / max(den, 1e-9)
        overlap = fit_overlap
        exclusive = (weights[view] > weight_floor) & ~overlap
        row.update(
            disagreement_before=round(before, 4),
            disagreement_after=round(after, 4),
            field_p95=round(float(np.percentile(np.abs(field[overlap]), 95)), 4),
            exclusive_mean_abs_delta=round(
                float(
                    np.abs(corrected[exclusive] - rgbs[view][exclusive]).mean()
                ) if exclusive.any() else 0.0,
                4,
            ),
        )
        if projections[view].get("generated"):
            # Witness-RANKED accept for synthesized views (same rule as
            # `equalize_projection_tone`): a relight that measurably
            # worsens agreement with real photos never ships; one that
            # measurably improves it ships even when generated-mutual
            # agreement pays for it; real-absent falls back to the
            # aggregate. Measured need (fresh-draw car, /tmp/gfix2): the
            # aggregate let a side reference relight toward two OTHER
            # generated views' invented lighting (zero real overlap in
            # its gate mass) and the composite carried that light onto
            # surface visible at the source pose.
            margin = float(improvement_margin)
            accept = after < before - margin
            real_mass = class_sums["real"][2]
            if real_mass > 0.0:
                real_before = class_sums["real"][0] / real_mass
                real_after = class_sums["real"][1] / real_mass
                row.update(
                    real_disagreement_before=round(real_before, 4),
                    real_disagreement_after=round(real_after, 4),
                )
                if real_after > real_before + margin:
                    accept = False
                elif real_after < real_before - margin:
                    accept = True
        else:
            accept = after < before - float(improvement_margin)
        if accept:
            covered = weights[view] > 0.0
            # scarce witness candidates must carry the same relight as
            # the view's strict claims (they are the same photo a few
            # degrees further); they never participate in the fit
            # (overlap requires weight > weight_floor)
            scarce = projections[view].get("scarce_weight")
            if scarce is not None:
                covered = covered | (np.asarray(scarce, dtype=np.float32) > 0.0)
            out = np.asarray(projections[view]["rgba"], dtype=np.float32)
            out[:, :, :3][covered] = corrected[covered]
            projections[view]["rgba"] = out
            rgbs[view] = corrected
            row["applied"] = True
            # provenance: the write mask of this color-writing lane, so
            # a bake's stats name every writer with its footprint
            row["written_texels"] = int(np.count_nonzero(covered))
            applied_any = True
        stats["views"].append(row)
    stats["applied"] = applied_any
    return stats


def equalize_projection_tone(
    projections: Sequence[Dict[str, Any]],
    *,
    positions_texture: Optional[Any] = None,
    source_index: int = 0,
    fit_weight_floor: float = 0.02,
    min_pair_texels: int = 400,
    luminance_floor: float = 0.02,
    max_log_gain: float = 1.0,
    improvement_margin: float = 0.002,
    fade_radius_frac: float = 0.06,
    fade_density_full: float = 0.12,
    field_radius_frac: float = 0.03,
    protect_floor: float = 0.02,
) -> Dict[str, Any]:
    """Consensus tone-LEVEL equalization: one log-luminance gain per view.

    The order-0 fallback lane of `delight_projections`. The SH ratio fit is
    the right model when overlaps populate the normal sphere, but its joint
    solve trades small overlaps against large ones: measured on a
    4-generated-view low-coverage bake (car candidate), the solver spent the
    side views' 2.6k/5k rim-dominated fit texels satisfying the dominant
    back|top_rear pair (7.7k texels at log-ratio 1.1), the sides' own
    overlap disagreement WORSENED (0.250 -> 0.279, 0.231 -> 0.319) and the
    fail-closed revert correctly refused them — shipping a composite with
    HALF the views relit and half not. The handoff ledger measured the
    consequence: boundary step p50 0.214 / p95 0.446, with every dominant
    boundary pair 93-99% LUMINANCE (a tone level, not content).

    This pass solves the panorama gain-compensation problem instead
    (Brown & Lowe: one scalar gain per image over pairwise overlap
    statistics): minimize sum_pairs W_uv * (g_u - g_v - r_uv)^2 with
    r_uv = median log-luminance ratio on the (u, v) overlap and W_uv the
    overlap's summed min-weight mass. One unknown per view cannot
    shape-distort content, which is exactly why it stays reliable on the
    rim-dominated overlaps where the 9-DOF-per-view fit fails.

    Contracts shared with the delight lane:

    - gauge: every REAL photo view is gauge-fixed at zero — photos are
      evidence and DEFINE the level; synthesis conforms to them (the
      witness-ranking doctrine of `protect_observed_texels`), so only
      generated views receive gains. Views connected to the source only
      through other references inherit the chain's error, so their gain
      cap is halved (delight's chained-gauge convention). A connected
      component with NO real member keeps its weighted-mean gain at zero:
      the level common to a group is unobservable from ratios (the same
      albedo/shading ambiguity delight documents), and each image's own
      level passed the per-view material gates — relative consistency is
      the only claim the evidence supports.
    - fit floor `fit_weight_floor` 0.02 = the photo-evidence floor of
      `protect_observed_texels` (weight 0.02 ~ the 0.2 facing cutoff).
      Rim texels are legitimate evidence for a weighted STATISTIC; the
      measured justification is that at delight's 0.05 pair floor the
      source participates in ZERO pairs >= 400 texels on the low-coverage
      car (front|top_rear 0, front|side_right 55) — the gauge chain never
      reaches the photo — while at 0.02 the front|top_rear pair carries
      1.8k texels. `min_pair_texels` 400 is delight's own pair floor.
    - luminance-only (chroma untouched); white balance remains
      `harmonize_and_gate_projection`'s job.
    - overlap-proximity fade over the 3D surface: full correction near the
      overlap surface where handoffs form, zero deep inside a view's
      exclusive territory (same voxel-ball construction and constants).
    - fail-closed, witness-RANKED revert gate: a correction that
      measurably worsens agreement with REAL photos (by
      `improvement_margin`) never ships; one that measurably improves it
      ships even when generated-mutual agreement pays for it; only when
      the real class is absent or stable does generated-mutual
      improvement decide (see `witness_ranked_accept` — photos are
      evidence and define tone, mutual consistency among synthesized
      views is subordinate). Judging one aggregate instead structurally
      silences the photo: its rim overlap carries ~1% of a big view's
      pair mass BY CONSTRUCTION (facing demotions), so a correction that
      conforms a generated view to the photo where they meet — the exact
      boundary the handoff ledger flags — could never carry a
      mass-weighted gate (measured on the car candidate: top_rear's
      field improved the front pair and reverted on the aggregate).
      Every participating pair carries >= `min_pair_texels` texels, so
      neither class is ever judged on statistical noise.

    STAGE 2 — LOCAL CONSENSUS FIELD (generated views only). A single gain
    per view removes the level, but the measured pairwise disagreement is
    spatially varying: post-delight log-ratio IQRs on the car candidate
    were 0.68-1.55 across the top_rear pairs — independently synthesized
    views paint different regional shading (a roof bright in the elevated
    view, dark in the side view), which is exactly the structure the SH
    normal fit models and had to revert on the rim-dominated side
    overlaps.     Stage 2 reconciles it in POSITION space, where the evidence
    lives: per generated view, the per-texel log-luminance deviation from
    the projection-weight-averaged consensus of the texel's witnesses,
    weighted with the DOWNSTREAM composition semantics — wherever a real
    photo holds ANY positive claim, the consensus IS the photo's reading
    with full authority (generated weights are zeroed there exactly as
    `protect_observed_texels`' absolute mode will zero them, the view's
    own reading excluded); elsewhere it is the mixture the low band
    will blend, the view's own reading included (self-inclusion makes
    the deviation a shrinkage toward the shipped tone rather than full
    conformance to partners). The real-witnessed band enters the
    evidence even below the pair fit floor: grazing photo samples are
    smeared in detail but valid in REGIONAL tone, which is all this
    stage consumes (measured on the fresh-draw car: weighting the photo
    by its own 0.001-0.02 grazing weights let reference self-weights
    dominate the consensus and pulled references AWAY from the photo on
    exactly the surface they were about to own). The deviation is smoothed over the surface
    with the C0 voxel-ball statistic (`_voxel_field_mean_c0`) at
    `field_radius_frac` * diagonal — a REGIONAL statistic ~20x the
    detail-fusion scale at 2048. C0 is a hard requirement, not a nicety:
    the box-mean voxel statistic is piecewise-constant over its lattice,
    and this field MULTIPLIES image content — measured on the car_bo3
    live incident, the stepped field rendered as rectangular exposure
    blocks over the roof/glass (the "image inside an image" class), the
    field saturated at its cap with most of its spatial variance sitting
    BETWEEN lattice cells (within-cell std 0.47x total).
    Default 0.03 from the car-candidate radius sweep: overlap
    disagreement on the side_left|top_rear pair dropped
    -10.5/-10.6/-12.5/-14.2% at radius 0.06/0.04/0.03/0.02, but at 0.02
    the fitted fields saturate the amplitude cap (side_left field_p95
    0.465 vs cap 0.5; 0.301 at 0.03) — a saturated field is cap-clipped
    rather than evidence-shaped, so 0.03 keeps ~90% of the measured gain
    with clear amplitude headroom.
    Capped at half `max_log_gain` (the chained-amplitude budget: the
    local gauge is inherited through the same reference chain), faded by
    evidence density (delight's fade constants), and accepted per view
    under the same fail-closed revert gate. Real photo views are never
    field-corrected: photos are evidence and DEFINE the consensus;
    synthesis conforms to it (the witness-ranking doctrine of
    `protect_observed_texels`).

    Runs AFTER `delight_projections` (it consumes the SH-corrected colors
    and settles the residual LEVELS the SH lane could not gauge or had to
    revert) and before harmonization/protection. Mutates projection rgba in
    place (weights untouched) and returns stats.
    """
    import numpy as np

    stats: Dict[str, Any] = {"applied": False, "views": [], "pairs": []}
    view_count = len(projections)
    if view_count < 2:
        return stats

    weights = [np.asarray(p["weight"], dtype=np.float32) for p in projections]
    rgbs = [np.asarray(p["rgba"], dtype=np.float32)[:, :, :3] for p in projections]

    def luminance(rgb: Any) -> Any:
        return 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]

    lums = [luminance(rgb) for rgb in rgbs]

    # ---- pairwise overlap level ratios -----------------------------------
    pair_rows: List[Dict[str, Any]] = []
    for u in range(view_count):
        for v in range(u + 1, view_count):
            overlap = (
                (weights[u] > float(fit_weight_floor))
                & (weights[v] > float(fit_weight_floor))
                & (lums[u] > float(luminance_floor))
                & (lums[v] > float(luminance_floor))
            )
            count = int(overlap.sum())
            if count < int(min_pair_texels):
                continue
            ratio = float(np.median(
                np.log(lums[u][overlap].astype(np.float64))
                - np.log(lums[v][overlap].astype(np.float64))))
            mass = float(np.minimum(
                weights[u], weights[v])[overlap].sum())
            pair_rows.append({
                "u": u, "v": v, "ratio": ratio, "mass": mass,
                "texels": count, "overlap": overlap,
            })
    if not pair_rows:
        return stats

    # ---- weighted least squares on the pair graph ------------------------
    # Real photo views are gauge-FIXED at zero (they define the level);
    # only generated views are free unknowns. The reduced normal equations
    # eliminate the fixed variables exactly.
    src = int(source_index)
    real_views = {
        i for i in range(view_count) if not projections[i].get("generated")
    }
    real_views.add(src)
    free = [i for i in range(view_count) if i not in real_views]
    if not free:
        return stats
    col_of = {view: k for k, view in enumerate(free)}
    normal = np.zeros((len(free), len(free)), dtype=np.float64)
    rhs = np.zeros(len(free), dtype=np.float64)
    for row in pair_rows:
        u, v, mass, ratio = row["u"], row["v"], row["mass"], row["ratio"]
        cu, cv = col_of.get(u), col_of.get(v)
        if cu is not None:
            normal[cu, cu] += mass
            rhs[cu] += mass * ratio
        if cv is not None:
            normal[cv, cv] += mass
            rhs[cv] -= mass * ratio
        if cu is not None and cv is not None:
            normal[cu, cv] -= mass
            normal[cv, cu] -= mass
    # A component with no real member leaves its block singular (gains
    # only determined up to a constant); the tiny ridge makes the solve
    # well-posed and the exact mean-zero gauge is re-imposed below.
    ridge = 1e-9 * max(float(np.trace(normal)), 1.0)
    try:
        solved = np.linalg.solve(normal + ridge * np.eye(len(free)), rhs)
    except np.linalg.LinAlgError:
        return stats
    gains = np.zeros(view_count, dtype=np.float64)
    for view, col in col_of.items():
        gains[view] = solved[col]

    # connected components over the pair graph (union-find)
    parent = list(range(view_count))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for row in pair_rows:
        ra, rb = find(row["u"]), find(row["v"])
        if ra != rb:
            parent[ra] = rb
    component = [find(i) for i in range(view_count)]
    view_mass = np.zeros(view_count, dtype=np.float64)
    for row in pair_rows:
        view_mass[row["u"]] += row["mass"]
        view_mass[row["v"]] += row["mass"]
    for comp in set(component):
        members = [i for i in range(view_count) if component[i] == comp]
        if not any(m in real_views for m in members):
            mass = view_mass[members]
            gains[members] -= (
                float((gains[members] * mass).sum() / max(mass.sum(), 1e-9)))

    fitted = {i for row in pair_rows for i in (row["u"], row["v"])}
    direct_source = {
        row["u"] if row["v"] in real_views else row["v"]
        for row in pair_rows
        if (row["u"] in real_views) != (row["v"] in real_views)
    }

    def classed_disagreement(view: int, view_rgb: Any) -> Dict[str, float]:
        """Mass-weighted mean |rgb - partner| per witness class."""
        sums = {"real": 0.0, "generated": 0.0}
        masses = {"real": 0.0, "generated": 0.0}
        for row in pair_rows:
            if view not in (row["u"], row["v"]):
                continue
            other = row["v"] if row["u"] == view else row["u"]
            klass = "real" if other in real_views else "generated"
            overlap = row["overlap"]
            sums[klass] += row["mass"] * float(
                np.abs(view_rgb[overlap] - rgbs[other][overlap]).mean())
            masses[klass] += row["mass"]
        return {
            klass: sums[klass] / masses[klass]
            for klass in sums if masses[klass] > 0.0
        }

    def witness_ranked_accept(
            before: Dict[str, float], after: Dict[str, float]) -> bool:
        """RANKED witness classes: photos outrank generated-mutual.

        Lexicographic rule: a correction that measurably WORSENS the
        real-photo class never ships; one that measurably IMPROVES it
        ships even when generated-mutual agreement pays for it (photos
        are evidence and define the tone; mutual consistency among
        synthesized views is subordinate — the same ranking
        `protect_observed_texels` enforces on weights). Only when the
        real class is absent or stable does generated-mutual improvement
        decide.

        History: the first landed gate was SYMMETRIC (any class
        worsening vetoed). Measured on the fresh-draw car
        (/tmp/gfix2): the side view's consensus field improved photo
        agreement 0.208 -> 0.198 and was vetoed because generated-mutual
        moved 0.188 -> 0.204 — the veto shipped the tone error into the
        composite and the source-pose fidelity paid it. The symmetric
        form had been chosen over a mass-weighted TRADE rule (photo
        gains buying bounded generated losses) whose net effect could
        not be bounded; the ranked form is not a trade — the real class
        is still protected by its own margin, it just cannot be held
        hostage by the subordinate class. Every participating pair
        carries >= `min_pair_texels` texels, so neither class is judged
        on statistical noise.
        """
        margin = float(improvement_margin)
        real_before = before.get("real")
        real_after = after.get("real")
        if real_before is not None:
            if real_after > real_before + margin:
                return False
            if real_after < real_before - margin:
                return True
        gen_before = before.get("generated")
        gen_after = after.get("generated")
        return (
            gen_before is not None and gen_after < gen_before - margin)
    for row in pair_rows:
        stats["pairs"].append({
            "views": [projections[row["u"]].get("label"),
                      projections[row["v"]].get("label")],
            "texels": row["texels"],
            "log_ratio": round(row["ratio"], 4),
        })

    # ---- fade support (same construction as delight_projections) ---------
    surface_points: Optional[Any] = None
    surface_sel: Optional[Any] = None
    diagonal = 1.0
    if positions_texture is not None:
        positions_arr = np.asarray(positions_texture, dtype=np.float32)
        surface_sel = positions_arr[:, :, 3] > 0.0
        surface_points = positions_arr[:, :, :3][surface_sel].astype(np.float64)
        if len(surface_points) > 0:
            diagonal = float(np.linalg.norm(
                surface_points.max(axis=0) - surface_points.min(axis=0))) or 1.0

    applied_any = False
    for view in range(view_count):
        if view in real_views or view not in fitted:
            continue
        chained = view not in direct_source
        cap = float(max_log_gain) * (0.5 if chained else 1.0)
        gain = float(np.clip(gains[view], -cap, cap))
        view_pairs = [row for row in pair_rows if view in (row["u"], row["v"])]
        fit_overlap = np.zeros(weights[view].shape, dtype=bool)
        for row in view_pairs:
            fit_overlap |= row["overlap"]
        row_stats: Dict[str, Any] = {
            "index": view,
            "label": projections[view].get("label"),
            "gain_log": round(gain, 4),
            "chained_gauge": bool(chained),
            "applied": False,
        }
        if abs(gain) < 1e-4:
            stats["views"].append(row_stats)
            continue
        field = np.full(weights[view].shape, gain, dtype=np.float64)
        if (surface_points is not None and surface_sel is not None
                and len(surface_points) > 0):
            # C0 fade (see `_voxel_field_mean_c0` and the delight lane's
            # note): the box-mean density steps at voxel-lattice
            # boundaries and a stepped fade prints the lattice as
            # exposure blocks; the hard on-overlap override is removed
            # with the same rationale.
            overlap_indicator = fit_overlap[surface_sel].astype(np.float64)
            density = _voxel_field_mean_c0(
                surface_points,
                overlap_indicator,
                max(float(fade_radius_frac) * diagonal, 1e-9),
            )
            density = np.where(np.isfinite(density), density, 0.0)
            fade_surface = np.clip(
                density / max(float(fade_density_full), 1e-9), 0.0, 1.0)
            fade = np.zeros(field.shape, dtype=np.float64)
            fade[surface_sel] = fade_surface
            field = field * fade
        correction = np.exp(-field).astype(np.float32)
        corrected = np.clip(rgbs[view] * correction[:, :, None], 0.0, 1.0)
        # Fail-closed witness-ranked revert gate (see docstring; partners
        # corrected earlier in this loop are compared in their corrected
        # state — the delight lane's sequential semantics).
        before = classed_disagreement(view, rgbs[view])
        after = classed_disagreement(view, corrected)
        row_stats.update(
            disagreement_before={k: round(v, 4) for k, v in before.items()},
            disagreement_after={k: round(v, 4) for k, v in after.items()},
        )
        if witness_ranked_accept(before, after):
            covered = weights[view] > 0.0
            scarce = projections[view].get("scarce_weight")
            if scarce is not None:
                covered = covered | (np.asarray(scarce, dtype=np.float32) > 0.0)
            out = np.asarray(projections[view]["rgba"], dtype=np.float32)
            out[:, :, :3][covered] = corrected[covered]
            projections[view]["rgba"] = out
            rgbs[view] = corrected
            lums[view] = luminance(corrected)
            row_stats["applied"] = True
            row_stats["written_texels"] = int(np.count_nonzero(covered))
            applied_any = True
        stats["views"].append(row_stats)

    # ---- stage 2: local consensus field (generated views only) -----------
    field_cap = 0.5 * float(max_log_gain)
    for view in range(view_count):
        if view in real_views or view not in fitted:
            continue
        view_pairs = [row for row in pair_rows if view in (row["u"], row["v"])]
        # The consensus weighting mirrors the DOWNSTREAM composition
        # semantics: wherever a real photo holds ANY positive claim, the
        # shipped texel IS the photo's (`protect_observed_texels`
        # absolute mode zeroes generated weights there), so the consensus
        # on that whole band is the REAL-WITNESS reading with full
        # authority — no self-inclusion, no other generated view. The
        # first landed version granted the photo authority only at
        # weight >= the 0.02 floor and weighted it BY ITS OWN WEIGHT
        # elsewhere; measured on the fresh-draw car (/tmp/gfix2), the
        # photo's grazing band carries weights of 0.001-0.02 against
        # reference self-weights of 0.3-0.6, so the "consensus" there was
        # the reference's own reading and the field pulled references
        # AWAY from the photo exactly where they were about to own
        # photo-adjacent surface. Elsewhere (no real evidence) the
        # consensus stays the weighted mixture the low band will blend,
        # the view's own reading included (self-inclusion = shrinkage
        # toward the shipped tone rather than full conformance to
        # partners).
        protected = np.zeros(weights[view].shape, dtype=bool)
        real_num = np.zeros(weights[view].shape, dtype=np.float64)
        real_den = np.zeros(weights[view].shape, dtype=np.float64)
        for other in real_views:
            other_weight = weights[other].astype(np.float64)
            has_claim = (other_weight > 0.0) & (
                lums[other] > float(luminance_floor))
            protected |= has_claim
            other_lum = np.clip(
                lums[other].astype(np.float64), float(luminance_floor), None)
            real_num += np.where(
                has_claim, other_weight * np.log(other_lum), 0.0)
            real_den += np.where(has_claim, other_weight, 0.0)
        evidence = np.zeros(weights[view].shape, dtype=bool)
        consensus_num = np.zeros(weights[view].shape, dtype=np.float64)
        consensus_den = np.zeros(weights[view].shape, dtype=np.float64)
        for row in view_pairs:
            other = row["v"] if row["u"] == view else row["u"]
            overlap = row["overlap"]
            evidence |= overlap
            other_weight = weights[other].astype(np.float64)
            if other not in real_views:
                other_weight = np.where(protected, 0.0, other_weight)
            other_lum = np.clip(
                lums[other].astype(np.float64), float(luminance_floor), None)
            consensus_num += np.where(
                overlap, other_weight * np.log(other_lum), 0.0)
            consensus_den += np.where(overlap, other_weight, 0.0)
        # The real-witnessed band this view covers is tone evidence even
        # where the photo's weight sits under the pair fit floor: the
        # photo's grazing samples are smeared in DETAIL but valid in
        # REGIONAL tone, and this stage is a regional statistic (surface
        # smoothing at `field_radius_frac` x diagonal).
        own_covers = (
            (weights[view] > float(fit_weight_floor))
            & (lums[view] > float(luminance_floor)))
        real_band = protected & own_covers & (real_den > 1e-9)
        evidence |= real_band
        if int(evidence.sum()) < int(min_pair_texels):
            continue
        own_log = np.log(np.clip(
            lums[view].astype(np.float64), float(luminance_floor), None))
        own_weight = np.where(
            protected, 0.0, weights[view].astype(np.float64))
        good = evidence & (consensus_den > 1e-9)
        deviation = np.zeros(weights[view].shape, dtype=np.float64)
        total = consensus_den + np.where(evidence, own_weight, 0.0)
        consensus_log = np.zeros_like(consensus_num)
        consensus_log[good] = (
            consensus_num[good] + own_weight[good] * own_log[good]
        ) / total[good]
        # Photo authority on the real-witnessed band overrides the
        # mixture (downstream composition ships the photo there).
        consensus_log[real_band] = real_num[real_band] / real_den[real_band]
        good |= real_band
        deviation[good] = own_log[good] - consensus_log[good]
        # Bounded influence: a misregistered partner's content outlier
        # (a wheel read against an arch) must not steer the regional
        # field; the clip bounds any single texel's pull at the amplitude
        # the correction itself is allowed.
        deviation = np.clip(deviation, -2.0 * field_cap, 2.0 * field_cap)
        row_stats = {
            "index": view,
            "label": projections[view].get("label"),
            "stage": "local_field",
            "evidence_texels": int(good.sum()),
            "applied": False,
        }
        if (surface_points is None or surface_sel is None
                or len(surface_points) == 0):
            # The field is a SURFACE statistic; without positions there is
            # no honest smoothing support, so stage 2 stands down (stage 1
            # still ran — levels need no geometry).
            stats["views"].append(row_stats)
            continue
        evidence_surface = good[surface_sel].astype(np.float64)
        deviation_surface = np.where(good, deviation, 0.0)[surface_sel]
        radius = max(float(field_radius_frac) * diagonal, 1e-9)
        # THE ROOF-BLOCK FIX (car_bo3 live incident): the smoothed field
        # and its density MUST be C0. The box-mean voxel statistic is
        # piecewise-constant over its lattice, and this field multiplies
        # image content — on the car's flat roof/glass the lattice
        # rendered as rectangular exposure blocks stepping by up to the
        # full +-cap between adjacent cells ("image inside an image").
        # `_voxel_field_mean_c0` keeps the same regional scale and NaN
        # contract but interpolates between cell centers, so the applied
        # field is continuous by construction at any amplitude.
        density = _voxel_field_mean_c0(
            surface_points, evidence_surface, radius)
        smoothed = _voxel_field_mean_c0(
            surface_points, deviation_surface, radius)
        density = np.where(np.isfinite(density), density, 0.0)
        smoothed = np.where(np.isfinite(smoothed), smoothed, 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            field_surface = np.where(
                density > 1e-9, smoothed / np.maximum(density, 1e-9), 0.0)
        field_surface = np.clip(field_surface, -field_cap, field_cap)
        # SUPPORT-EDGE FADE: the ratio smoothed/density is a local mean
        # of the deviation and stays O(cap) arbitrarily close to the
        # evidence support's boundary, while outside it is defined 0 —
        # a residual hard step of up to the cap exactly at the boundary
        # (the wider fade below does not necessarily vanish there). The
        # same density, normalized by the fade-full constant, drives the
        # field to zero continuously as its own evidence runs out.
        support_fade = np.clip(
            density / max(float(fade_density_full), 1e-9), 0.0, 1.0)
        field_surface = field_surface * support_fade
        # Evidence-density fade at the delight lane's constants: the
        # correction dies away from the surface that justified it.
        fade_density = _voxel_field_mean_c0(
            surface_points, evidence_surface,
            max(float(fade_radius_frac) * diagonal, 1e-9))
        fade_density = np.where(np.isfinite(fade_density), fade_density, 0.0)
        fade_surface = np.clip(
            fade_density / max(float(fade_density_full), 1e-9), 0.0, 1.0)
        field = np.zeros(weights[view].shape, dtype=np.float64)
        field[surface_sel] = field_surface * fade_surface
        if float(np.abs(field).max()) < 1e-4:
            stats["views"].append(row_stats)
            continue
        correction = np.exp(-field).astype(np.float32)
        corrected = np.clip(rgbs[view] * correction[:, :, None], 0.0, 1.0)
        before = classed_disagreement(view, rgbs[view])
        after = classed_disagreement(view, corrected)
        row_stats.update(
            field_p95=round(float(np.percentile(
                np.abs(field[good]), 95)) if good.any() else 0.0, 4),
            disagreement_before={k: round(v, 4) for k, v in before.items()},
            disagreement_after={k: round(v, 4) for k, v in after.items()},
        )
        if witness_ranked_accept(before, after):
            covered = weights[view] > 0.0
            scarce = projections[view].get("scarce_weight")
            if scarce is not None:
                covered = covered | (np.asarray(scarce, dtype=np.float32) > 0.0)
            out = np.asarray(projections[view]["rgba"], dtype=np.float32)
            out[:, :, :3][covered] = corrected[covered]
            projections[view]["rgba"] = out
            rgbs[view] = corrected
            lums[view] = luminance(corrected)
            row_stats["applied"] = True
            row_stats["written_texels"] = int(np.count_nonzero(covered))
            applied_any = True
        stats["views"].append(row_stats)

    for row in pair_rows:
        row.pop("overlap", None)
    stats["applied"] = applied_any
    return stats


def harmonize_and_gate_projection(
    projection: Dict[str, Any],
    *,
    source_projection: Mapping[str, Any],
    harmonize: bool = True,
    min_overlap_texels: int = 400,
    attenuate_above: float = 0.16,
    reject_above: float = 0.34,
) -> Dict[str, Any]:
    """Color-harmonize a reference projection to the source and gate it.

    Both corrections run on the overlap set (texels observed by BOTH the
    source side and the reference), which compares the same physical surface
    points:

    1. Per-channel linear gain toward the source's overlap statistics
       (median-ratio estimate, clamped to [0.5, 2.0]) removes exposure and
       white-balance differences. The gain is only KEPT when it actually
       reconciles the overlap: when the post-gain disagreement stays above
       `attenuate_above`, the overlap difference is content mismatch
       (misregistration, different subject state), the exposure estimate is
       confounded, and applying it would tint the reference's correctly
       painted exclusive texels — so the colors revert.
    2. The residual mean absolute RGB disagreement is a direct
       reprojection-error measure. Registration failures, wrong declared
       angles, and identity-inconsistent synthesized views all show up here
       — and nothing downstream could previously detect that a reference
       made the texture worse. Weights attenuate linearly from
       `attenuate_above` and reach zero at `reject_above`.

    Mutates `projection["rgba"]`/`projection["weight"]` in place and returns
    a stats row.
    """
    import numpy as np

    stats: Dict[str, Any] = {
        "overlap_texels": 0,
        "harmonized": False,
        "gains": None,
        "disagreement": None,
        "weight_scale": 1.0,
    }
    source_weight = np.asarray(source_projection.get("weight"), dtype=np.float32)
    reference_weight = np.asarray(projection.get("weight"), dtype=np.float32)
    if source_weight.shape != reference_weight.shape:
        return stats
    overlap = (source_weight > 0.05) & (reference_weight > 0.05)
    overlap_count = int(overlap.sum())
    stats["overlap_texels"] = overlap_count
    if overlap_count < int(min_overlap_texels):
        return stats

    source_rgb = np.asarray(source_projection["rgba"], dtype=np.float32)[:, :, :3]
    reference_rgba = np.asarray(projection["rgba"], dtype=np.float32)
    disagreement = float(
        np.abs(reference_rgba[:, :, :3][overlap] - source_rgb[overlap]).mean()
    )
    if harmonize:
        # A true exposure/white-balance difference is ONE multiplicative
        # relation that holds across the overlap: per-texel ratios cluster
        # tightly. Content mismatch (hair pixels overlapping skin pixels)
        # produces a wide ratio spread, and a gain fitted on it tints the
        # reference's correctly painted exclusive texels. Gate on the
        # log-ratio interquartile range before trusting any gain.
        reference_overlap = reference_rgba[:, :, :3][overlap]
        source_overlap = source_rgb[overlap]
        usable = reference_overlap.mean(axis=1) > 0.02
        gains: Optional[List[float]] = None
        if int(usable.sum()) >= int(min_overlap_texels):
            ratios = np.maximum(source_overlap[usable], 1e-4) / np.maximum(
                reference_overlap[usable], 1e-4
            )
            log_ratios = np.log(ratios)
            spread = float(
                np.max(
                    np.percentile(log_ratios, 75, axis=0) - np.percentile(log_ratios, 25, axis=0)
                )
            )
            stats["gain_ratio_spread"] = round(spread, 4)
            if spread <= 0.7:
                gains = [
                    float(np.clip(float(np.median(ratios[:, channel])), 0.5, 2.0))
                    for channel in range(3)
                ]
        if gains is not None:
            harmonized_rgb = np.clip(
                reference_rgba[:, :, :3] * np.asarray(gains, dtype=np.float32), 0.0, 1.0
            )
            disagreement_after = float(
                np.abs(harmonized_rgb[overlap] - source_rgb[overlap]).mean()
            )
            if disagreement_after <= float(attenuate_above) or disagreement_after < disagreement - 0.02:
                covered = reference_weight > 0.0
                # scarce witness candidates inherit the same gain (same
                # photo, same exposure); they never influence the gain
                # estimate (overlap requires weight > 0.05)
                scarce = projection.get("scarce_weight")
                if scarce is not None:
                    covered = covered | (np.asarray(scarce, dtype=np.float32) > 0.0)
                reference_rgba[:, :, :3][covered] = harmonized_rgb[covered]
                projection["rgba"] = reference_rgba
                stats["harmonized"] = True
                stats["gains"] = [round(g, 4) for g in gains]
                disagreement = disagreement_after

    stats["disagreement"] = round(disagreement, 4)
    if disagreement > float(attenuate_above):
        span = max(float(reject_above) - float(attenuate_above), 1e-6)
        scale = float(np.clip(1.0 - (disagreement - float(attenuate_above)) / span, 0.0, 1.0))
        projection["weight"] = reference_weight * scale
        stats["weight_scale"] = round(scale, 4)
    return stats


def filter_projection_outliers(
    mesh: Any,
    *,
    positions_texture: Any,
    projections: Sequence[Mapping[str, Any]],
    blended_rgb: Any,
    observed_mask: Any,
    color_threshold: float = 0.3,
    min_neighbor_support: int = 3,
) -> Any:
    """Drop observed texels that are foreign islands on the mesh surface.

    Misprojections at silhouette rims (a hair-shell tip catching forehead
    pixels from a profile photo, a chin catching hair-fringe pixels) share
    a signature no local per-view test can catch: the texel's WINNING VIEW
    differs from the view that dominates its mesh neighborhood, AND its
    color disagrees strongly with that neighborhood's consensus. Genuine
    high-frequency detail (an iris on a cheek) is painted by the same view
    as its surroundings, so it never trips the first condition.

    Returns a boolean drop mask in atlas space; callers should treat
    dropped texels as unobserved so completion/fill replaces them.
    """
    import numpy as np

    import numpy as np  # noqa: F811 (kept close to use for clarity)

    observed = np.asarray(observed_mask, dtype=bool)
    drop = np.zeros_like(observed)
    # Single-view bakes still benefit: the foreign-VIEW condition is
    # vacuous with one witness, but the same-view "extreme" condition
    # (color wildly foreign to the surface neighborhood) catches silhouette
    # rim misprojections — dark background-adjacent pixels stamped onto
    # hull/skin at grazing angles (measured on the single-view starship:
    # the surviving dark fragments at 4x zoom are exactly this class).
    if not projections or not observed.any():
        return drop
    try:
        from scipy import sparse
        from scipy.spatial import cKDTree
    except Exception:
        return drop

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    edges = np.asarray(mesh.edges_unique, dtype=np.int64)
    if len(vertices) == 0 or len(edges) == 0:
        return drop

    positions = np.asarray(positions_texture, dtype=np.float32)
    weight_stack = np.stack([np.asarray(p["weight"], dtype=np.float32) for p in projections])
    winner = weight_stack.argmax(axis=0)
    winner_weight = weight_stack.max(axis=0)
    texel_sel = observed & (winner_weight > 0.05)
    if not texel_sel.any():
        return drop

    texel_points = positions[:, :, :3][texel_sel]
    texel_view = winner[texel_sel]
    texel_rgb = np.asarray(blended_rgb, dtype=np.float32)[texel_sel]

    tree = cKDTree(vertices)
    _, vertex_of_texel = tree.query(texel_points, k=1, workers=-1)

    row = np.concatenate([edges[:, 0], edges[:, 1]])
    col = np.concatenate([edges[:, 1], edges[:, 0]])
    adjacency = sparse.coo_matrix(
        (np.ones(len(row)), (row, col)), shape=(len(vertices), len(vertices))
    ).tocsr()
    # Two-hop reach makes the consensus wider than a misprojection patch's
    # interior, so patches larger than a one-ring still meet foreign
    # neighbors.
    reach = adjacency + adjacency @ adjacency
    # A texel must not vote in its own consensus: the diagonal of A@A equals
    # the vertex degree, which let any island >= a one-ring dominate its own
    # neighborhood histogram and dilute the color deviation below threshold
    # (measured: a planted foreign island voted 28-vs-14 for itself and was
    # never dropped). Binarize because off-diagonal 2-hop PATH COUNTS (up to
    # 2 through triangles) similarly overweighted island mutual support.
    reach.setdiag(0.0)
    reach.eliminate_zeros()
    reach.data[:] = 1.0

    view_count = len(projections)
    alive = np.ones(len(texel_view), dtype=bool)
    # Iterative erosion: each pass removes texels foreign to their current
    # surviving neighborhood, so contiguous misprojection patches erode
    # from their borders inward instead of surviving on self-support.
    for _ in range(3):
        vertex_view_weight = np.zeros((len(vertices), view_count), dtype=np.float64)
        np.add.at(
            vertex_view_weight,
            (vertex_of_texel[alive], texel_view[alive]),
            1.0,
        )
        vertex_rgb_sum = np.zeros((len(vertices), 3), dtype=np.float64)
        vertex_rgb_count = np.zeros(len(vertices), dtype=np.float64)
        np.add.at(vertex_rgb_sum, vertex_of_texel[alive], texel_rgb[alive].astype(np.float64))
        np.add.at(vertex_rgb_count, vertex_of_texel[alive], 1.0)

        neighbor_view_weight = reach @ vertex_view_weight
        neighbor_rgb_sum = reach @ vertex_rgb_sum
        neighbor_rgb_count = reach @ vertex_rgb_count
        has_support = neighbor_rgb_count >= float(min_neighbor_support)
        neighbor_dominant = neighbor_view_weight.argmax(axis=1)
        neighbor_rgb = neighbor_rgb_sum / np.maximum(neighbor_rgb_count, 1e-6)[:, None]

        texel_neighbor_dominant = neighbor_dominant[vertex_of_texel]
        texel_neighbor_rgb = neighbor_rgb[vertex_of_texel]
        texel_has_support = has_support[vertex_of_texel]
        color_deviation = np.abs(texel_rgb - texel_neighbor_rgb.astype(np.float32)).mean(axis=1)
        foreign = (
            alive
            & texel_has_support
            & (texel_view != texel_neighbor_dominant)
            & (color_deviation > float(color_threshold))
        )
        # Same-view islands need a higher bar: shell tips at silhouette
        # rims catch mismatched content from their OWN view (a hair-shell
        # tip stamped with forehead skin by the front photo). Genuine
        # same-view detail (lips, brows, eyes) recruits its own kind into
        # the consensus and stays under this threshold.
        extreme = alive & texel_has_support & (color_deviation > float(color_threshold) + 0.1)
        removed = foreign | extreme
        if not removed.any():
            break
        alive = alive & ~removed

    drop_indices = ~alive
    drop[texel_sel] = drop_indices
    return drop


def resolve_projection_conflicts(
    projections: Sequence[Dict[str, Any]],
    *,
    conflict_threshold: float = 0.25,
    min_weight: float = 0.05,
    priority_view: Optional[int] = 0,
    priority_floor: float = 0.25,
) -> Dict[str, Any]:
    """Resolve per-texel color conflicts by keeping the best witness.

    Global weight attenuation punishes a whole view for localized
    disagreement: a correct profile photo loses its exclusive side coverage
    because the front view paints hair strands over a band of shared cheek
    texels. The disagreement is real but LOCAL, so the response must be
    local too. On texels where covering views disagree strongly (any
    pairwise mean absolute RGB difference above `conflict_threshold`), one
    witness keeps its claim:

    - the `priority_view` (the SOURCE photo — the user's actual image of
      the actual subject state) wins wherever it sees the texel WELL
      (weight above `priority_floor`): on such surface the source is
      ground truth and no synthesized/auxiliary reference may overrule
      it. An ablation moved the floor from 0.45 down to 0.25: 0.45 handed
      contested cheek texels to stretched reference content and increased
      duplicate-feature blobs, while 0.25 still lets a head-on reference
      (weight ~0.8) overrule truly grazing source rim samples
      (weight ~0.2).
    - at grazing source angles the highest projection weight wins: there
      the source's own samples are stretched rim content (a forehead
      photo pixel smeared over a hair-shell top), and a reference that
      faces the surface head-on is the better witness.

    Mutates projection weights in place and returns summary stats.
    """
    import numpy as np

    stats: Dict[str, Any] = {"conflict_texels": 0, "zeroed_by_view": {}}
    if len(projections) < 2:
        return stats
    rgb_stack = np.stack(
        [np.asarray(p["rgba"], dtype=np.float32)[:, :, :3] for p in projections]
    )
    weight_stack = np.stack([np.asarray(p["weight"], dtype=np.float32) for p in projections])
    covering = weight_stack > float(min_weight)
    multi = covering.sum(axis=0) >= 2
    if not multi.any():
        return stats

    conflict = np.zeros(multi.shape, dtype=bool)
    view_count = len(projections)
    for i in range(view_count):
        for j in range(i + 1, view_count):
            both = covering[i] & covering[j] & multi
            if not both.any():
                continue
            difference = np.abs(rgb_stack[i] - rgb_stack[j]).mean(axis=2)
            conflict |= both & (difference > float(conflict_threshold))
    if not conflict.any():
        return stats

    winner = weight_stack.argmax(axis=0)
    if priority_view is not None and 0 <= int(priority_view) < view_count:
        prio = int(priority_view)
        source_credible = weight_stack[prio] > float(priority_floor)
        winner = np.where(source_credible, prio, winner)
    stats["conflict_texels"] = int(conflict.sum())
    for i, projection in enumerate(projections):
        kill = conflict & covering[i] & (winner != i)
        if kill.any():
            weights = np.asarray(projection["weight"], dtype=np.float32).copy()
            weights[kill] = 0.0
            projection["weight"] = weights
            stats["zeroed_by_view"][str(projection.get("label") or i)] = int(kill.sum())
    return stats


def protect_observed_texels(
    projections: Sequence[Dict[str, Any]],
    *,
    protect_floor: float = 0.02,
    mode: str = "ramp",
) -> Dict[str, Any]:
    """Generated views COMPLETE the surface; they never REVISE it.

    Weight subordination (generated x0.6) makes synthesized views lose
    per-texel contests, but the feathered blend still averages them into
    photo-covered texels — measured on the chair/portrait v2 bakes as rust
    mottling and skin blotches on the FRONT view, i.e. synthesis revising
    the user's own photograph. The semantic boundary is sharper than any
    weight ratio: a real photo is evidence, a generated view is plausible
    synthesis, and synthesis must contribute nothing where evidence exists.

    `protect_floor` is deliberately at the photo's own COVERAGE edge
    (weight 0.02 ~ its 0.2 facing cutoff), not at a "credible claim"
    level: a 0.25 floor left every obliquely-photographed surface (ear
    tops, brow ridges, foot tops at facing 0.2-0.5) open to generated
    content, and correctly-registered references measurably repainted the
    subject's front with drifted tone (photo fidelity deltaE 19.2 -> 25.3;
    an earlier registration bug had been hiding the leak by pushing the
    references' rims off the mesh). Stretched oblique photo content is
    still the subject's own appearance — the certified single-photo path
    ships exactly that band — while a generated view is somebody's guess
    about it.

    `mode` selects what happens BELOW the floor:

    - "ramp" (historical default): generated weights ramp back linearly,
      so the handoff sits at the photo's own fade-out.
    - "absolute": generated weight is zeroed wherever ANY real view holds
      POSITIVE weight — full single-view sovereignty. Measured
      justification (fresh-draw car decomposition, /tmp/gfix2): on a
      low-coverage subject at an estimated pose, 25% of the
      photo-witnessed atlas carried real weight BELOW the 0.02 floor
      (grazing facing, concavity demotions), and the ramp handed that
      band to generated references at up to 30x the photo's own weight —
      |candidate - baseline| on the band measured 39 dE mean over 20k
      texels, the single largest witnessed-surface contamination
      channel (photo fidelity at the true pose regressed +3.2 dE; the
      absolute mode alone recovered 2.2). The doctrine was always
      "a real photo's stretched rim content outranks plausible
      synthesis" (the source keeps single-view facing semantics when
      references are generated); the ramp contradicted it exactly in
      the band where the photo is weakest. The bake call site uses
      "absolute"; the coverage-edge handoff smoothing that the ramp
      provided is carried by the blend feather and the gradient-domain
      composite instead (measured: no handoff-ledger regression).

    Texels where a real view's claim was ZEROED upstream (per-texel
    conflict resolution: a head-on reference outranks the photo's
    stretched rim samples under strong disagreement) hold no real weight
    by the time this runs, so they stay reference-owned in both modes —
    that contest was measured net-GOOD for source-pose fidelity (the
    photo's own grazing smear loses to content that faces the surface).

    Mutates generated projections' weights in place; returns stats.
    """
    import numpy as np

    stats: Dict[str, Any] = {"applied": False, "zeroed_by_view": {}}
    generated = [p for p in projections if p.get("generated")]
    real = [p for p in projections if not p.get("generated")]
    if not generated or not real:
        return stats
    real_weight = np.stack(
        [np.asarray(p["weight"], dtype=np.float32) for p in real], axis=0
    ).max(axis=0)
    floor = max(float(protect_floor), 1e-6)
    if str(mode) == "absolute":
        # Synthesis contributes nothing wherever evidence exists, at any
        # confidence: the photo's weakest witnessed band is still the
        # subject's own appearance.
        scale = (real_weight <= 0.0).astype(np.float32)
        protected = real_weight > 0.0
    else:
        # 1.0 where no real evidence, fading to 0.0 at/above the floor.
        scale = np.clip((floor - real_weight) / floor, 0.0, 1.0)
        protected = real_weight >= floor
    stats["applied"] = True
    stats["protect_floor"] = floor
    stats["mode"] = str(mode)
    stats["protected_texels"] = int(protected.sum())
    for projection in generated:
        weight = np.asarray(projection["weight"], dtype=np.float32)
        zeroed = int(((weight > 0.0) & protected).sum())
        projection["weight"] = weight * scale
        stats["zeroed_by_view"][str(projection.get("label"))] = zeroed
    return stats


def admit_scarce_witnesses(
    projections: Sequence[Dict[str, Any]],
    *,
    surface_mask: Any,
    positions_texture: Optional[Any] = None,
    exclude_mask: Optional[Any] = None,
    consensus_weight: float = 0.35,
    consensus_radius_ratio: float = 0.03,
    consensus_max_spread: float = 0.09,
    consensus_contrast: float = 0.22,
    consensus_min_neighbors: int = 24,
    feature_moat_ratio: float = 0.044,
) -> Dict[str, Any]:
    """Admit below-threshold witness claims on texels NOBODY paints (G1).

    THE SURRENDER THIS CLOSES (measured on the certified face proof,
    cycle 7): the per-role facing thresholds (source 0.4 in ortho
    multi-view, references 0.2) are calibrated for surface where a BETTER
    witness exists — beyond them the view's samples are stretched rim
    content and the other photo is the better witness (ADR 0008). But the
    same hard gate also discards claims on texels NO view passes its
    threshold for: the band between the source's 0.4 cutoff and a
    profile's 0.2 cutoff (the owner's example — the left jaw IS in the
    left photo yet was repainted as mirror/fill), the under-chin, the
    crown. There "stretched content beats no content" — the single-view
    doctrine — applies per texel: a real observation of that exact
    surface outranks a symmetry guess or harmonic fill.

    MECHANISM: the projector emits, per view, a `scarce_weight` map —
    claims between the grazing floor (facing 0.05) and the role
    threshold, bounded by the EXACT per-texel sampling stretch (the
    texel->photo Jacobian; facing is a tilt proxy and cannot see
    collapsed mappings) and still respecting first-surface visibility,
    photo alpha, the layered-zone surrender, and the stretch/concavity
    demotion. This pass admits those claims ONLY on texels where no view
    holds a strict claim; where any strict witness exists the strict
    gates keep their measured calibration and every scarce claim stays
    discarded. Admission happens AFTER the global compositing solve
    (strictly local paint — the knowledge-base "commit local repairs
    late" rule): an early admission perturbs the screened-Poisson
    anchor set GLOBALLY, and the measured effect on the face proof was
    knife-edge photo-true dark content (the front's under-lip shadow,
    winner weight 0.63) re-shading by 1-3/255 at battery poses far from
    any rescued texel — three absolute debris detectors sitting at 73%
    utilization flipped red, and the fringe stage's pre-repair baseline
    inherited the drift (its exemption bound loosened and admitted a
    mouth stamp the certified baseline refuses). Rescued texels still
    inherit the delight/harmonization tone corrections (those stages
    apply their fields to scarce candidates' colors too), and admission
    stays AFTER per-view registration, so registration evidence remains
    strict-witness-only. Scarce weights inherit the facing-squared
    preference, so when several views' scarce claims overlap, the
    caller's merge arbitrates by best witness exactly like the blend.

    CONSENSUS GUARD (`positions_texture`; the mirror-copy guard's exact
    semantics transposed — see `mirror_fill_from_observed`): geometry
    silhouettes never match the photo contour exactly, so a grazing
    claim near a material boundary can be the OTHER material's pixel
    displaced onto this surface (a jawline hair pixel landing on chin
    skin). Measured on the face proof at 1024: unguarded admission was
    74% dark content and lifted dark_debris 0.0022 -> 0.0038 at az-22.5
    (gate 0.003) while identity improved — the admitted band is real
    photo content but its boundary mixtures are exactly the flake class.
    Two refusal rules, both consensus-based:

    1. CONTRADICTION (the mirror guard's rule): the strict-confident 3D
       neighborhood (winner weight >= `consensus_weight` within
       `consensus_radius_ratio` x scale) is color-consistent (spread <=
       `consensus_max_spread`) AND the claim deviates from its mean by
       more than `consensus_contrast`. Feature-rich neighborhoods (high
       spread) accept everything under this rule alone.
    2. LIKE-MATERIAL SUPPORT: a claim's own class (dark/bright at 0.55x
       the confident bright-half median — the pipeline's standard dark
       split) must be the MAJORITY class of its confident neighborhood,
       and a claim with NO confident neighborhood is refused outright
       (admission requires support). Both directions were measured as
       detector regressions: DARK-on-bright claims are the debris/flake
       class (rule 1 alone refused 54 of 2984 and the az-22.5 debris
       detectors stayed red at 0.0035 vs the 0.003 gate), and
       BRIGHT-on-dark claims are the FACE-07 pale-chip class displaced
       into hair (an arm without the bright direction shipped crown
       skin-flake failures 0.0000->0.0022 at az±135/180 vs the 0.0008
       gate). Near material boundaries the legitimate mixed context
       raises the ring spread above `consensus_max_spread`, which
       disarms rule 1 — the class-majority rule is what binds there.
       Refused texels fall through to mirror/fill exactly as before.

    3. DARK-MASS ADJACENCY (the material-commitment doctrine: dark
       commitment requires consensus with the DARK BODY, not local
       darkness): surviving dark claims must lie within the consensus
       radius of the strict-confident dark MASS — the largest
       world-space voxel component of confident dark texels (the
       detectors' own "the hair mass is never an island" construction).
       A dark pocket licensed only by a nearby dark FEATURE (a lip
       slit, a brow) is refused: features must not seed adjacent dark
       admission. Measured: mouth-corner dark claims passed rule 2 on
       the slit's own dark density, became fill anchors, and the
       harmonic fill continued them into ~390 px isolated dark islands
       on skin at THREE battery views — none of which touched a rescued
       texel's rendered position (the "global stages make early texel
       edits non-local" class; the fill-floor tracking exemption then
       correctly kept the islands because their fill TRACKED its own
       observed anchors). Hair-rim and mass-adjacent bands — the bulk
       of the dark orphan area — keep their leverage.

    4. FEATURE MOAT (the billboard lesson: "doubled brow/lid reads as a
       third eye — base-material witness veto inside the feature
       moat"): NO claim is admitted within `feature_moat_ratio` x scale
       of a strong dark FEATURE core (confident, locally contrastful,
       compact — the rescue detector's core signal, excluding the dark
       mass). Orphan bands beside features are where parallax-displaced
       feature content lands (measured end-state: rescued mouth-corner
       texels re-anchored the surround and shipped ~390 px lip-slit
       continuations as debris islands at three battery poses even
       when the claims themselves were bright and locally consistent —
       rules 1-3 cannot see displaced-feature adjacency). Features are
       well-witnessed by construction (strict claims exist), so the
       moat costs no leverage where nothing else could paint.

    Mutates projection weights/rgba alpha in place; returns stats.
    """
    import numpy as np

    stats: Dict[str, Any] = {"applied": False, "admitted_texels": 0,
                             "consensus_refused": 0, "views": []}
    if not projections:
        return stats
    surface = np.asarray(surface_mask, dtype=bool)
    scarce_maps = []
    for projection in projections:
        scarce = projection.get("scarce_weight")
        if scarce is None:
            return stats
        scarce_maps.append(np.asarray(scarce, dtype=np.float32))

    weight_stack = np.stack(
        [np.asarray(p["weight"], dtype=np.float32) for p in projections], axis=0)
    strict_any = (weight_stack > 0.0).any(axis=0)
    orphan = surface & ~strict_any
    if exclude_mask is not None:
        # deliberately surrendered-and-committed surface (film band) is
        # not orphaned — surrender+commit is one coupled decision
        orphan &= ~np.asarray(exclude_mask, dtype=bool)

    # confident-consensus context for the guard: strict winners at
    # confident weight, their positions and blended (max-weight) colors
    guard_tree = None
    guard_colors = None
    guard_dark = None
    guard_split = 0.0
    guard_scale = 1.0
    mass_tree = None
    feature_tree = None
    if positions_texture is not None:
        positions = np.asarray(positions_texture, dtype=np.float32)[:, :, :3]
        winner_weight = weight_stack.max(axis=0)
        winner_index = weight_stack.argmax(axis=0)
        confident = surface & (winner_weight >= float(consensus_weight))
        if int(confident.sum()) >= int(consensus_min_neighbors):
            rgb_stack = np.stack(
                [np.asarray(p["rgba"], dtype=np.float32)[:, :, :3]
                 for p in projections], axis=0)
            rows, cols = np.nonzero(confident)
            guard_colors = rgb_stack[winner_index[rows, cols], rows, cols]
            guard_points = positions[rows, cols]
            guard_scale = float(np.linalg.norm(
                guard_points.max(axis=0) - guard_points.min(axis=0))) or 1.0
            guard_lum = guard_colors.mean(axis=1)
            bright_median = float(np.median(
                guard_lum[guard_lum >= np.median(guard_lum)]))
            guard_split = 0.55 * bright_median
            guard_dark = guard_lum < guard_split
            try:
                from scipy.spatial import cKDTree

                guard_tree = cKDTree(guard_points)
            except Exception:
                guard_tree = None
            # rule 3 context: the strict-confident dark MASS (largest
            # world voxel component of confident dark texels)
            in_mass = np.zeros(len(guard_points), dtype=bool)
            if guard_tree is not None and int(guard_dark.sum()) >= int(
                    consensus_min_neighbors):
                from .feature_fringe_repair import _cluster_core_texels_world

                dark_indices = np.nonzero(guard_dark)[0]
                dark_points = guard_points[dark_indices]
                component = _cluster_core_texels_world(
                    dark_points[:, None, :],
                    np.ones((len(dark_points), 1), dtype=bool),
                    float(consensus_radius_ratio) * guard_scale,
                )[:, 0]
                counts = np.bincount(component[component >= 0])
                if counts.size:
                    mass_members = component == int(counts.argmax())
                    mass_tree = cKDTree(dark_points[mass_members])
                    in_mass[dark_indices[mass_members]] = True
            # rule 4 context: strong dark FEATURE cores (the rescue
            # detector's core signal — confident, locally contrastful
            # compact darks off the mass: iris, lash mass, lip slit)
            if guard_tree is not None:
                _, ball_mean, _ = _voxel_ball_stats(
                    guard_points, guard_lum, 0.02 * guard_scale,
                    guard_points)
                feature_core = ((guard_lum - ball_mean) <= -0.12) & ~in_mass
                if int(feature_core.sum()) >= 8:
                    feature_tree = cKDTree(guard_points[feature_core])

    admitted_union = np.zeros(surface.shape, dtype=bool)
    refused_union = np.zeros(surface.shape, dtype=bool)
    refused_total = 0
    for projection, scarce in zip(projections, scarce_maps):
        sel = orphan & (scarce > 0.0)
        count = int(sel.sum())
        refused = 0
        if count and guard_tree is not None:
            rows, cols = np.nonzero(sel)
            candidate_colors = np.asarray(
                projection["rgba"], dtype=np.float32)[rows, cols, :3]
            candidate_points = np.asarray(
                positions_texture, dtype=np.float32)[rows, cols, :3]
            neighbor_lists = guard_tree.query_ball_point(
                candidate_points,
                r=float(consensus_radius_ratio) * guard_scale, workers=-1)
            candidate_dark = candidate_colors.mean(axis=1) < guard_split
            mass_near = np.zeros(len(rows), dtype=bool)
            if mass_tree is not None and candidate_dark.any():
                mass_distance, _ = mass_tree.query(
                    candidate_points[candidate_dark], k=1, workers=-1)
                mass_near[candidate_dark] = (
                    np.asarray(mass_distance)
                    <= float(consensus_radius_ratio) * guard_scale)
            in_moat = np.zeros(len(rows), dtype=bool)
            if feature_tree is not None:
                moat_distance, _ = feature_tree.query(
                    candidate_points, k=1, workers=-1)
                in_moat = (np.asarray(moat_distance)
                           <= float(feature_moat_ratio) * guard_scale)
            keep = np.ones(len(rows), dtype=bool)
            for i, neighbors in enumerate(neighbor_lists):
                if in_moat[i]:
                    # rule 4: no admission inside a feature moat
                    keep[i] = False
                    continue
                if candidate_dark[i] and not mass_near[i]:
                    # rule 3: dark admission requires the dark MASS, not
                    # a nearby dark feature (see docstring)
                    keep[i] = False
                    continue
                if len(neighbors) < int(consensus_min_neighbors):
                    # rule 2: admission requires like-material support
                    keep[i] = False
                    continue
                local = guard_colors[neighbors]
                local_mean = local.mean(axis=0)
                dark_share = float(guard_dark[neighbors].mean())
                if candidate_dark[i] != (dark_share >= 0.5):
                    keep[i] = False  # rule 2: claim off its own material
                    continue
                if float(np.abs(local - local_mean).mean()) > float(consensus_max_spread):
                    continue  # feature-rich context accepts everything
                if float(np.abs(candidate_colors[i] - local_mean).mean()) > float(consensus_contrast):
                    keep[i] = False
            refused = int((~keep).sum())
            refused_union[rows[~keep], cols[~keep]] = True
            sel = np.zeros(surface.shape, dtype=bool)
            sel[rows[keep], cols[keep]] = True
            count = int(sel.sum())
        refused_total += refused
        stats["views"].append({
            "label": str(projection.get("label") or "view"),
            "admitted_texels": count,
            "consensus_refused": refused,
        })
        if count == 0:
            continue
        weight = np.asarray(projection["weight"], dtype=np.float32)
        weight[sel] = scarce[sel]
        projection["weight"] = weight
        rgba = np.asarray(projection["rgba"], dtype=np.float32)
        rgba[:, :, 3] = np.where(sel, 1.0, rgba[:, :, 3])
        projection["rgba"] = rgba
        admitted_union |= sel
    stats["admitted_texels"] = int(admitted_union.sum())
    stats["consensus_refused"] = refused_total
    stats["applied"] = stats["admitted_texels"] > 0
    stats["admitted_ratio"] = round(
        stats["admitted_texels"] / max(float(surface.sum()), 1.0), 4)
    stats["admitted_mask"] = admitted_union
    stats["refused_mask"] = refused_union & ~admitted_union
    return stats


def consolidate_unwitnessed_debris(
    mesh: Any,
    *,
    atlas: Any,
    colors: Any,
    positions_texture: Any,
    normals_texture: Any,
    strict_mask: Any,
    island_share_min: float = 0.6,
    size: int = 896,
) -> Tuple[Any, int]:
    """Render-informed consolidation of UNWITNESSED dark micro islands
    (the FACE-20 displaced-refill discipline, the same construction as the
    fringe lane's `_consolidate_render_specks`, applied to fill).

    THE CLASS (measured on the face proof at 1024, three admission arms):
    fill pockets anchored by locally dark observed content ship as
    isolated bright-ringed dark micro islands on skin at oblique battery
    poses. The certified baseline lives at 73% of the debris gate on
    exactly this population (az-22.5 el0: 7 items, ~all measured 71-100%
    unwitnessed fill); ANY re-partitioning of the harmonic fill's pockets
    — which witness-scarcity admission inevitably causes, anywhere on the
    surface, because the solve is global — reshuffles them, and one
    re-partitioned pocket (415 px, far mouth corner) took three battery
    views over the gate. A blast-radius scope was measured meaningless
    (the admitted band spans every view boundary, so all fill pockets are
    "near" rescue). The honest general rule has no radius:

    AN ISOLATED, BRIGHT-RINGED, SUB-FEATURE DARK ISLAND WHOSE TEXELS ARE
    PREDOMINANTLY UNWITNESSED IS A DEFECT BY DEFINITION — no photo
    witnessed its content where it shows, and its own ring context
    contradicts it. `strict_mask` is the PHOTO-EVIDENCE set: direct
    claims (rescued included), rescue-disc transplants (twin photo
    evidence), film-repainted falloff. MIRROR COMPLETION IS EXCLUDED on
    purpose: it is a symmetry GUESS, and the measured regression
    pockets were 98-100% mirror texels (the twin mouth-corner's slit
    content copied into a re-partitioned completion pocket, reading as
    an isolated island — its feature-rich destination ring disarms the
    mirror stage's own consensus guard by design). Real observation
    beats symmetry guess is the completion doctrine; a guess that
    renders as isolated debris on light material loses its tone.

    At each veto-battery pose, lift the unwitnessed members
    (`island_share_min` of first-surface texels outside `strict_mask`)
    to just above the dark class under THAT view's own shading
    (0.88 + 0.12 * facing — an albedo above the split still renders dark
    when tilted; the displaced-refill floor rule keeps the luminance
    pattern at reduced contrast). Photo-witnessed dark anatomy is never
    touched (witnessed islands fail the share gate); feature-scale blobs
    belong to the doubling detectors and are excluded by the sub-feature
    cap; and texels rendering inside a FEATURE-CLASS compact dark blob
    at ANY battery pose are protected (`_feature_blob_footprint`
    construction — the fringe lane's measured lesson: a texel lifted at
    one pose can be another pose's profile-eye lash mass).

    ISLAND CONSTRUCTION: the dark split is anchored to the LIGHT
    MATERIAL's own median — the standard YCrCb skin-chroma class when
    the subject carries one (portraits: the binding debris detectors
    normalize by the skin median, and a mouth-corner island ringed by
    mid-luminance lip material is invisible to a bright-half-median
    split, measured: threshold 113 vs 100 fused the island into the
    feature-scale slit component), with the bright-half median as the
    subject-agnostic fallback. Ring acceptance requires the island to
    sit on light-material/skin context; the largest dark component (the
    dark mass itself) is never an island.
    """
    import numpy as np
    from scipy.ndimage import binary_dilation
    from scipy.ndimage import label as cc_label

    from .feature_fringe_repair import (
        _compact_dark_blobs_px,
        _feature_blob_footprint,
        _first_surface_projection,
        _render_foreground,
        _render_with_colors,
        _renderer_camera,
    )

    def _debris_islands(rgb: Any, fg: Any) -> List[Dict[str, Any]]:
        arr = np.asarray(rgb, np.float32)
        lum = (0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1]
               + 0.114 * arr[:, :, 2])
        fg_lum = lum[fg]
        if fg_lum.size < 256:
            return []
        # light-material median: skin-chroma class when present (the
        # debris detectors' own normalization), bright half otherwise
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b
        cb = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b
        skin = (fg & (cr >= 135) & (cr <= 180) & (cb >= 80) & (cb <= 130)
                & (lum > 110))
        if int(skin.sum()) > 100:
            light_median = float(np.median(lum[skin]))
            light_context = skin
        else:
            light_median = float(np.median(fg_lum[fg_lum >= np.median(fg_lum)]))
            light_context = fg & (lum >= light_median)
        split = 0.55 * light_median
        size_px = arr.shape[0]
        rows, cols = np.nonzero(fg)
        bbox_area = float((rows.max() - rows.min() + 1)
                          * (cols.max() - cols.min() + 1))
        feature_floor = 0.0009 * bbox_area
        speck_floor = max(int(24 * (size_px / 896.0) ** 2), 8)
        dark = fg & (lum < split)
        labels, count = cc_label(dark, structure=np.ones((3, 3), bool))
        sizes = np.bincount(labels.ravel())
        if len(sizes):
            sizes[0] = 0
        mass_id = int(sizes.argmax()) if sizes.size > 1 else -1
        out: List[Dict[str, Any]] = []
        for index in range(1, count + 1):
            if index == mass_id:
                continue
            area = int(sizes[index])
            if not (speck_floor <= area < feature_floor):
                continue
            component = labels == index
            ring = binary_dilation(component, iterations=3) & ~component & fg
            if not ring.any() or float(light_context[ring].mean()) < 0.45:
                continue
            out.append({"mask": component, "area": area, "split": split})
        return out

    colors = np.asarray(colors, np.float32)
    positions = np.asarray(positions_texture, np.float32)
    surface = positions[:, :, 3] > 0.0
    if not surface.any():
        return colors, 0
    normals = np.asarray(normals_texture, np.float32)[:, :, :3]
    normals = normals / np.maximum(
        np.linalg.norm(normals, axis=2, keepdims=True), 1e-8)
    surface_rows, surface_cols = np.nonzero(surface)
    surface_points = positions[:, :, :3][surface_rows, surface_cols]
    strict_flat = np.asarray(strict_mask, dtype=bool)[surface_rows,
                                                      surface_cols]

    # the fringe lane's veto battery: the eye-gated near-frontal fan plus
    # the negative-azimuth sweep where debris regressions appear
    veto_views = [(0.0, (0.0, 22.5, -22.5, 35.0, -35.0, -45.0, -70.0,
                         -90.0, 90.0)),
                  (10.0, (0.0, 22.5, -22.5, -45.0, -70.0, -90.0))]
    # pass 1: one render per battery view on the input colors — debris
    # islands to judge + feature-class blob evidence for the footprint
    # protection (a texel lifted at one pose can be another pose's
    # profile-eye lash mass; the render battery knows where features are)
    scale = float(np.linalg.norm(surface_points.max(axis=0)
                                 - surface_points.min(axis=0))) or 1.0
    r_feat_world = 0.02 * scale
    scanned: List[Tuple[float, float, Any, Any]] = []
    pre_entries: List[Tuple[Any, List[Dict[str, Any]]]] = []
    for el, azimuths in veto_views:
        rendered = _render_with_colors(mesh, atlas, colors, azimuths, el, size)
        entries = []
        for az, rgb in zip(azimuths, rendered):
            fg = _render_foreground(rgb)
            camera = _renderer_camera(mesh, az, el, size)
            r_px = r_feat_world * camera["px_per_world"]
            blobs, _ = _compact_dark_blobs_px(rgb, fg, (0.25 * r_px, 2.6 * r_px))
            entries.append({"blobs": blobs})
            scanned.append((az, el, rgb, fg))
        pre_entries.append(((el, azimuths), entries))
    protected_flat = _feature_blob_footprint(mesh, pre_entries,
                                             surface_points, size)

    lifted_total = 0
    for az, el, rgb, fg in scanned:
        islands = _debris_islands(rgb, fg)
        if not islands:
            continue
        px, py, visible = _first_surface_projection(
            mesh, surface_points, az, el, size)
        ix = np.clip(np.round(px).astype(np.int32), 0, size - 1)
        iy = np.clip(np.round(py).astype(np.int32), 0, size - 1)
        camera = _renderer_camera(mesh, az, el, size)
        eye = camera["eye"]
        for island in islands:
            mask_px = binary_dilation(island["mask"], iterations=1)
            in_island = mask_px[iy, ix] & visible
            if not in_island.any():
                continue
            unwitnessed = in_island & ~strict_flat
            share = float(unwitnessed.sum()) / float(in_island.sum())
            if share < float(island_share_min):
                continue
            unwitnessed &= ~protected_flat
            if not unwitnessed.any():
                continue
            render_split = float(island["split"]) / 255.0
            sel_rows = surface_rows[unwitnessed]
            sel_cols = surface_cols[unwitnessed]
            level = colors[sel_rows, sel_cols, :3].mean(axis=1)
            shade = 0.88 + 0.12 * np.clip(
                normals[sel_rows, sel_cols] @ eye, 0.0, 1.0)
            target = 1.02 * render_split / np.maximum(shade, 0.5)
            need = level < target
            if not need.any():
                continue
            lift = target[need] / np.maximum(level[need], 1e-6)
            colors[sel_rows[need], sel_cols[need], :3] = np.clip(
                colors[sel_rows[need], sel_cols[need], :3]
                * lift[:, None], 0.0, 1.0)
            lifted_total += int(need.sum())
    return colors, lifted_total


def assemble_leverage_ledger(
    projections: Sequence[Mapping[str, Any]],
    *,
    surface_mask: Any,
    direct_union_mask: Any,
    shipped_direct_mask: Optional[Any] = None,
    mirror_added_mask: Optional[Any] = None,
) -> Dict[str, Any]:
    """Per-view potential/painted/won reference-leverage ledger.

    THE QUESTION THIS ANSWERS (project owner, cycle 7): how much of what
    the photos can GEOMETRICALLY witness does the bake actually paint from
    them, and which gate surrenders the rest? Measured on the certified
    face proof before this ledger existed: the views could see ~57% of the
    surface (facing > 0.05, unoccluded, photo content present) while
    direct projection painted ~21% — the difference was silently handed to
    mirror/harmonic fill by the quality gates, invisible in any stat.

    Definitions (all masks in atlas space, produced by the projector):

      potential   in-frame, first-surface unoccluded, facing > 0.05,
                  photo alpha present — the honest paintable set
      painted     final projection weight > 0 intersected with the shipped
                  direct union (post conflict/film/outlier demotions)
      won         texel's final blend winner among the painted set

    Per-view surrender attribution inside `potential` (mutually
    exclusive, in gate order):

      facing_gate  facing at or below the view's own threshold
      zone_gate    layered-zone witness surrender (mixture regions)
      downstream   projected but killed later (harmonize/conflict/film)
      union_drop   painted by the view but dropped from the union
                   (outlier filter, speckle/rim demotion)

    The ledger is accounting ONLY — it never feeds a numeric path — and
    is emitted into bake stats as `leverage` (texture_qa reports it).
    """
    import numpy as np

    surface = np.asarray(surface_mask, dtype=bool)
    surface_count = int(surface.sum())
    if surface_count == 0 or not projections or any(
            not isinstance(p, Mapping) or p.get("potential") is None
            for p in projections):
        return {"available": False}

    direct_union = np.asarray(direct_union_mask, dtype=bool) & surface
    weight_stack = np.stack(
        [np.asarray(p["weight"], dtype=np.float32) for p in projections], axis=0)
    winner_index = weight_stack.argmax(axis=0)
    winner_weight = weight_stack.max(axis=0)

    potential_union = np.zeros(surface.shape, dtype=bool)
    views: List[Dict[str, Any]] = []
    for index, projection in enumerate(projections):
        potential = np.asarray(projection["potential"], dtype=bool) & surface
        potential_union |= potential
        facing = np.asarray(projection["facing"], dtype=np.float32)
        zone = np.asarray(projection["zone"], dtype=bool)
        threshold = float(projection.get("facing_threshold", 0.2))
        weight = weight_stack[index]
        painted_final = weight > 0.0
        painted = painted_final & direct_union
        won = (winner_index == index) & (winner_weight > 1e-6) & direct_union

        above = potential & (facing > threshold)
        below = potential & ~(facing > threshold)
        # scarcity-rescued claims paint below the strict threshold; they
        # are painted, not surrendered
        rescued = below & painted_final
        facing_gate = below & ~painted_final
        zone_gate = above & zone
        downstream = above & ~zone & ~painted_final
        union_drop = above & ~zone & painted_final & ~direct_union

        def _ratio(count: int) -> float:
            return round(count / float(surface_count), 4)

        views.append({
            "label": str(projection.get("label") or f"view_{index + 1:02d}"),
            "facing_threshold": threshold,
            "potential_texels": int(potential.sum()),
            "potential_ratio": _ratio(int(potential.sum())),
            "painted_texels": int(painted.sum()),
            "painted_ratio": _ratio(int(painted.sum())),
            "won_texels": int(won.sum()),
            "won_ratio": _ratio(int(won.sum())),
            "rescued_texels": int(rescued.sum()),
            "surrendered_facing_gate": int(facing_gate.sum()),
            "surrendered_zone_gate": int(zone_gate.sum()),
            "surrendered_downstream": int(downstream.sum()),
            "surrendered_union_drop": int(union_drop.sum()),
        })

    surrendered_union = potential_union & ~direct_union
    ledger: Dict[str, Any] = {
        "available": True,
        "surface_texels": surface_count,
        "potential_union_ratio": round(
            float(potential_union.sum()) / surface_count, 4),
        "direct_painted_ratio": round(
            float(direct_union.sum()) / surface_count, 4),
        "leverage_ratio": round(
            float(direct_union.sum()) / max(float(potential_union.sum()), 1.0), 4),
        "surrendered_visible_ratio": round(
            float(surrendered_union.sum()) / surface_count, 4),
        "unobservable_ratio": round(
            float((surface & ~potential_union).sum()) / surface_count, 4),
        "views": views,
    }
    if shipped_direct_mask is not None:
        shipped = np.asarray(shipped_direct_mask, dtype=bool) & surface
        ledger["shipped_direct_ratio"] = round(
            float(shipped.sum()) / surface_count, 4)
    if mirror_added_mask is not None:
        mirror = np.asarray(mirror_added_mask, dtype=bool) & surface
        ledger["mirror_completed_ratio"] = round(
            float(mirror.sum()) / surface_count, 4)
        # G4 watch: mirror completion writing texels a photo could witness
        # (it only writes texels with NO direct claim, so any overlap here
        # is surface the gates surrendered and symmetry then guessed)
        ledger["mirror_over_photo_visible_ratio"] = round(
            float((mirror & potential_union).sum()) / surface_count, 4)
    return ledger


def bake_projection_texture(
    mesh: Any,
    *,
    observed_views: Sequence[Mapping[str, Any]],
    texture_resolution: int = 2048,
    texture_completion: str = "none",
    base_color_fn: Optional[Callable[[Any], Any]] = None,
    blend_sharpness: float = 3.0,
    feather_texels: Optional[float] = None,
    harmonize_references: bool = True,
    estimate_source_pose: bool = False,
    refine_reference_poses: bool = True,
    source_pose_window_deg: float = 75.0,
    projection_model: str = "perspective",
    canonical_border_ratio: float = 0.15,
    fill_detail_gain: float = 0.7,
    compositing: str = "auto",
    source_pose_override: Optional[Tuple[float, float]] = None,
    scarcity_rescue: str = "auto",
    generated_reference_weight: float = 0.6,
    detail_fusion: str = "two_band",
) -> Tuple[Any, Dict[str, Any]]:
    """Bake a UV base-color texture for `mesh` from observed views.

    `observed_views` entries carry `rgba` (PIL RGBA image), `azimuth_deg`,
    `elevation_deg`, `label`, and optional `role` ("source"/"reference").
    View angles are interpreted RELATIVE TO THE SOURCE VIEWPOINT (the first
    view): a `side_left` reference means "90 degrees left of wherever the
    source photo was taken from". With `estimate_source_pose=True` the
    estimated source pose offset is therefore applied to every view.
    `base_color_fn(positions_texture) -> rgba array` supplies backend color
    priors for texels views cannot reach (e.g. the TripoSR triplane field);
    without it, unseen texels are inpainted from the projected texels.

    `texture_completion` accepts "none", "mirror_symmetry", or "auto"
    (mirror completion if and only if the mesh's own left-right symmetry
    score passes the same gate that protects explicit mirror requests).
    `fill_detail_gain` scales the observed-statistics micro-texture added
    to propagated fill regions (see `synthesize_fill_detail`); 0 disables.

    `compositing` selects how per-view projections become one texture:
    "legacy" is the additive patch stack (softmax color blend + per-region
    seam-leveling offsets); "gradient_domain" composites per-view GRADIENTS
    and solves one screened Poisson system over the texel surface graph
    (see `gradient_compositing`), which removes tone seams mathematically
    while preserving witnessed edges; "auto" resolves to gradient_domain
    (measured better on both QA harnesses on the face and starship proof
    assets — A/B numbers in CHANGELOG) with legacy kept selectable.

    `scarcity_rescue` admits below-facing-threshold witness claims on
    texels NO view paints (bounded by exact per-texel sampling stretch;
    see `admit_scarce_witnesses`): "auto" enables it for multi-view bakes
    and keeps single-view bakes on their calibrated wide threshold
    ("on"/"off" force either; the single-photo proof assets are pinned
    regression canaries, so enabling there needs its own A/B).

    Returns `(textured_trimesh, stats)` where stats mirrors the TripoSR bake
    metadata contract (projection mode, coverage, per-view stats, atlas data).
    """
    import numpy as np

    from .backends.triposr_runtime import (
        _tripo_build_textured_mesh,
        _tripo_edge_bleed_texture,
        _tripo_make_texture_atlas,
        _tripo_project_observed_texture,
        _tripo_rasterize_normal_atlas,
        _tripo_rasterize_position_atlas,
        _tripo_texture_image,
        _tripo_texture_padding,
        _tripo_uv_preview,
    )

    resolution = int(texture_resolution)
    padding = _tripo_texture_padding(resolution)
    if feather_texels is None:
        feather_texels = max(4.0, resolution / 512.0 * 3.0)

    atlas = _tripo_make_texture_atlas(mesh, texture_resolution=resolution, texture_padding=padding)
    # Rasterize per-texel attributes WITHOUT border dilation. Dilating
    # positions/normals into chart gaps copies front-face attributes next to
    # back-face charts; those contaminated gap texels then pass the facing
    # and depth tests and bake shadow/background photo pixels as speckle.
    # Chart gaps are instead filled from baked colors by the final edge
    # bleed, which cannot invent false coverage.
    raster_kwargs = dict(
        atlas_vmapping=atlas["vmapping"],
        atlas_indices=atlas["indices"],
        atlas_uvs=atlas["uvs"],
        texture_resolution=resolution,
        texture_padding=0,
    )
    positions_texture = _tripo_rasterize_position_atlas(mesh, **raster_kwargs)
    normals_texture = _tripo_rasterize_normal_atlas(mesh, **raster_kwargs)
    surface_mask = np.asarray(positions_texture)[:, :, 3] > 0.0

    views = [dict(view) for view in observed_views if view.get("rgba") is not None]
    camera_distance = 1.9
    source_pose: Dict[str, Any] = {"azimuth_deg": 0.0, "elevation_deg": 0.0, "iou": 0.0}
    orthographic = str(projection_model) == "orthographic"
    if orthographic:
        # Canonical-frame path for models that reconstruct under an
        # orthographic camera from a recentered conditioning image
        # (Hunyuan-family). FRAMING becomes deterministic: recenter each
        # photo exactly as the model's own preprocessor does, and use the
        # orthographic half-extent that reproduces that framing per pose.
        # POSE does not: the backend canonicalizes the OBJECT (symmetry
        # plane onto the world axes), not the camera, so a photo of a
        # subject whose head is turned sits 15-25 degrees away from the
        # canonical front and must be projected from there. Silhouettes
        # cannot see that yaw on a head; gradient correlation against
        # untextured renders can.
        if views and source_pose_override is not None:
            # Explicit pin (reproducible verification bakes, known-pose
            # captures): bypass estimation entirely.
            pinned_azimuth, pinned_elevation = source_pose_override
            source_pose = {
                "azimuth_deg": float(pinned_azimuth),
                "elevation_deg": float(pinned_elevation),
                "estimated": False,
                "method": "override",
                "score": None,
                "score_at_declared": None,
            }
            views[0]["azimuth_deg"] = float(views[0].get("azimuth_deg", 0.0)) + float(pinned_azimuth)
            views[0]["elevation_deg"] = float(views[0].get("elevation_deg", 0.0)) + float(
                pinned_elevation
            )
        elif views:
            # Reference angles name sides of the SUBJECT ("side_left"), and
            # the canonical object frame IS the subject frame, so they hold
            # regardless of how the head was turned in the source photo;
            # only the source camera needs its pose recovered. A +/-40 deg
            # window covers plausible head turn in a "front" photo.
            photometric_pose = estimate_pose_with_silhouette_guard(
                mesh,
                views[0]["rgba"],
                border_ratio=float(canonical_border_ratio),
                azimuth_window_deg=40.0,
            )
            source_pose = {
                "azimuth_deg": photometric_pose["azimuth_deg"],
                "elevation_deg": photometric_pose["elevation_deg"],
                "estimated": bool(photometric_pose["estimated"]),
                "method": photometric_pose.get("method", "gradient_ncc"),
                "score": photometric_pose["score"],
                "score_at_declared": photometric_pose["score_at_declared"],
                "rejected_reason": photometric_pose.get("rejected_reason"),
            }
            if photometric_pose["estimated"]:
                views[0]["azimuth_deg"] = float(views[0].get("azimuth_deg", 0.0)) + float(
                    photometric_pose["azimuth_deg"]
                )
                views[0]["elevation_deg"] = float(views[0].get("elevation_deg", 0.0)) + float(
                    photometric_pose["elevation_deg"]
                )
        # Frame convention for the recenter. The orthographic projector
        # centers the WORLD ORIGIN; the canonical recenter centers the
        # photo's ALPHA BBOX. Away from the canonical front those diverge
        # by the mesh bbox's projected offset, and the plain recenter then
        # paints every sample that many pixels off the surface that imaged
        # it (SHIP-03 "nose melt": 54 px at az+30/el+15; src-pose fidelity
        # MAE 45.5 -> 18.1 with the correction). WHICH frame is registered
        # truth depends on HOW the pose was established:
        # - OVERRIDDEN pose (external capture fact): the model never
        #   consumed this frame, so the photo must be registered to the
        #   projector's own convention — `projected_frame_center_px`,
        #   deterministic, no content search.
        # - ESTIMATED pose (gradient_ncc) or canonical front: the
        #   estimator SEARCHED az/el for the best gradient alignment of
        #   the legacy-centered photo — pose and frame are co-adapted, and
        #   re-centering one side breaks the pair (measured on the face
        #   proof: verdict1 failures 2 -> 10, front SSIM 0.630 -> 0.598).
        #   At the canonical front the two conventions agree to ~1 px by
        #   construction (the model itself recentered the conditioning
        #   image), so legacy centering IS the projector frame there.
        projector_frame = source_pose_override is not None
        for view in views:
            if projector_frame:
                frame_size = 1024
                view_center = projected_frame_center_px(
                    mesh,
                    azimuth_deg=float(view.get("azimuth_deg", 0.0)),
                    elevation_deg=float(view.get("elevation_deg", 0.0)),
                    size=frame_size,
                    border_ratio=float(canonical_border_ratio),
                )
                view["frame_center_dx_px"] = round(
                    view_center[0] - frame_size / 2.0, 2)
                view["frame_center_dy_px"] = round(
                    view_center[1] - frame_size / 2.0, 2)
                view["rgba"] = recenter_to_canonical_frame(
                    view["rgba"],
                    size=frame_size,
                    border_ratio=float(canonical_border_ratio),
                    center_px=view_center,
                )
            else:
                view["rgba"] = recenter_to_canonical_frame(
                    view["rgba"], border_ratio=float(canonical_border_ratio)
                )
        estimate_source_pose = False
        refine_reference_poses = False
        camera_distance = 3.0
    if views and not orthographic:
        first_rgba = np.asarray(views[0]["rgba"].convert("RGBA"), dtype=np.float32) / 255.0
        if estimate_source_pose:
            # Backends that reconstruct in a canonical object frame (the
            # photo is not guaranteed to be the frame's front view) need the
            # source camera pose recovered before projection. The offset also
            # applies to reference views, whose named angles are relative to
            # the source viewpoint. Multi-view-conditioned geometry pins the
            # canonical front to the front input, so callers can narrow this
            # window to reflect that stronger prior.
            # NOTE: a photometric-NCC tie-break over the top silhouette
            # candidates was tried here and reverted: on faces the interior
            # edge NCC landscape is flat and structureless (empirically
            # verified), so it "decisively" preferred poses 15 degrees off
            # frontal, and every reference view inherited that error.
            # Silhouette + angular prior is the reliable general-purpose
            # signal at this stage.
            source_pose = estimate_view_pose(
                mesh,
                observed_rgba=first_rgba,
                azimuth_window_deg=float(source_pose_window_deg),
            )
            camera_distance = float(source_pose["camera_distance"])
            for view in views:
                view["azimuth_deg"] = float(view.get("azimuth_deg", 0.0)) + float(source_pose["azimuth_deg"])
                view["elevation_deg"] = float(view.get("elevation_deg", 0.0)) + float(source_pose["elevation_deg"])
        else:
            camera_distance = estimate_camera_distance(
                mesh,
                observed_rgba=first_rgba,
                azimuth_deg=float(views[0].get("azimuth_deg", 0.0)),
                elevation_deg=float(views[0].get("elevation_deg", 0.0)),
            )
    projections: List[Dict[str, Any]] = []
    registration_stats: List[Dict[str, Any]] = []
    has_real_references = any(
        not view.get("generated") for view in views[1:]
    )
    for index, view in enumerate(views, start=1):
        rgba_image = view["rgba"].convert("RGBA")
        view_azimuth = float(view.get("azimuth_deg", 0.0))
        view_elevation = float(view.get("elevation_deg", 0.0))
        view_distance = float(camera_distance)
        view_half_extent: Optional[float] = None
        if orthographic:
            view_half_extent = canonical_ortho_half_extent(
                mesh,
                azimuth_deg=view_azimuth,
                elevation_deg=view_elevation,
                border_ratio=float(canonical_border_ratio),
            )
        pose_stats: Dict[str, Any] = {"refined": False}
        if index > 1 and refine_reference_poses:
            # Named reference angles ("side_left") are approximate: real
            # photos and synthesized views are routinely 10-20 degrees off
            # the label, and projecting at the wrong pose paints features
            # onto the wrong surface. Solve each reference's pose in a
            # narrow window around its declared angle, with its own camera
            # distance (subject framing differs per photo). The refined pose
            # is accepted only when its silhouette IoU beats the declared
            # pose by a clear margin: near-symmetric subjects have flat IoU
            # landscapes where the argmax is noise, and empirically the
            # refiner otherwise drifts AWAY from the true pose as often as
            # toward it.
            refined = estimate_view_pose(
                mesh,
                observed_rgba=np.asarray(rgba_image, dtype=np.float32) / 255.0,
                center_azimuth_deg=view_azimuth,
                center_elevation_deg=view_elevation,
                azimuth_window_deg=25.0,
                azimuth_step_deg=5.0,
                elevation_candidates_deg=(
                    view_elevation - 10.0,
                    view_elevation,
                    view_elevation + 10.0,
                ),
                prior_strength=0.06,
                default_distance=float(camera_distance),
                min_iou=0.4,
            )
            accepted = float(refined["iou"]) > float(refined.get("center_iou") or 0.0) + 0.02
            if not accepted:
                refined["azimuth_deg"] = view_azimuth
                refined["elevation_deg"] = view_elevation
                refined["camera_distance"] = estimate_camera_distance(
                    mesh,
                    observed_rgba=np.asarray(rgba_image, dtype=np.float32) / 255.0,
                    azimuth_deg=view_azimuth,
                    elevation_deg=view_elevation,
                    default_distance=float(camera_distance),
                )
            pose_stats = {
                "refined": True,
                "accepted": bool(accepted),
                "declared_azimuth_deg": view_azimuth,
                "declared_elevation_deg": view_elevation,
                "azimuth_deg": refined["azimuth_deg"],
                "elevation_deg": refined["elevation_deg"],
                "camera_distance": refined["camera_distance"],
                "silhouette_iou": refined["iou"],
                "declared_iou": refined.get("center_iou"),
            }
            view_azimuth = float(refined["azimuth_deg"])
            view_elevation = float(refined["elevation_deg"])
            view_distance = float(refined["camera_distance"])
            view["azimuth_deg"] = view_azimuth
            view["elevation_deg"] = view_elevation
        if orthographic:
            # For the SOURCE view the canonical recenter IS the
            # registration: the model consumed exactly this frame, so any
            # further silhouette-fit warp only chases reconstruction error
            # in the geometry and displaces the photo's features (measured:
            # a residual scale/shift search on the source doubled the
            # duplicate-feature count at three-quarter views). REFERENCE
            # photos carry their own framing (a profile shot cropped at the
            # chest does not cover the mesh's shoulder extent), so they are
            # first aligned by crop-immune WIDTH-PROFILE matching (the
            # subject's top is essentially never cropped, making row-wise
            # silhouette widths a scale-sensitive signature that area-IoU
            # and edge-chamfer objectives lack on cropped photos), then a
            # small residual search absorbs the remainder.
            if index == 1:
                refine_stats = {"applied": False}
                reg_stats = {"applied": False, "scale": 1.0, "shift_x": 0.0, "shift_y": 0.0}
            else:
                rgba_image, refine_stats = register_view_by_width_profile(
                    mesh,
                    observed_rgba=rgba_image,
                    azimuth_deg=view_azimuth,
                    elevation_deg=view_elevation,
                    camera_distance=view_distance,
                    projection_model="orthographic",
                    ortho_half_extent=view_half_extent,
                )
                rgba_image, reg_stats = register_view_2d(
                    mesh,
                    observed_rgba=rgba_image,
                    azimuth_deg=view_azimuth,
                    elevation_deg=view_elevation,
                    camera_distance=view_distance,
                    scale_candidates=(0.96, 1.0, 1.04),
                    shift_range=0.06,
                    shift_step=0.02,
                    projection_model="orthographic",
                    ortho_half_extent=view_half_extent,
                )
        else:
            # 2D-register each photo against the mesh silhouette at its
            # assigned pose so features project onto the right surface.
            # NOTE: `refine_registration_photometric` was removed from this
            # path after an adversarial ground-truth test: injected known
            # shifts were recovered in 0 of 15 trials and the NCC objective
            # proposed nearly the same warp regardless of the true offset
            # (a constant attractor, not a signal). The function remains
            # available for callers with edge-rich subjects.
            rgba_image, reg_stats = register_view_2d(
                mesh,
                observed_rgba=rgba_image,
                azimuth_deg=view_azimuth,
                elevation_deg=view_elevation,
                camera_distance=view_distance,
            )
            refine_stats = {"applied": False}
        rgba_image = erode_view_alpha(rgba_image)
        if orthographic and index > 1 and projections and view_half_extent is not None:
            # Final registration stage: silhouette methods align OUTLINES
            # (on heads, the hair contour) and can leave interior features
            # displaced; the source's already-projected texels provide
            # ground-truth colors at known surface points, so align the
            # reference's interior content photometrically to that overlap.
            rgba_image, overlap_stats = register_reference_by_source_overlap(
                rgba_image,
                positions_texture=positions_texture,
                source_projection=projections[0],
                azimuth_deg=view_azimuth,
                elevation_deg=view_elevation,
                camera_distance=view_distance,
                ortho_half_extent=float(view_half_extent),
                normals_texture=normals_texture,
            )
            reg_stats["overlap_alignment"] = overlap_stats
            # Dense residual: global transforms cannot satisfy per-feature
            # displacements (nose/mouth/eyes each want different small
            # shifts on generated geometry), which paints ghost lip/lash
            # fragments next to the source's features. The strictly-local
            # validated lattice flow aligns the reference's interior
            # content to the source truth cell by cell and leaves
            # everything unvalidated — hair, far side, thin evidence —
            # at identically zero displacement (see reference_flow).
            from .reference_flow import estimate_reference_flow

            rgba_image, flow_stats = estimate_reference_flow(
                rgba_image,
                positions_texture=positions_texture,
                source_projection=projections[0],
                azimuth_deg=view_azimuth,
                elevation_deg=view_elevation,
                camera_distance=view_distance,
                ortho_half_extent=float(view_half_extent),
                normals_texture=normals_texture,
            )
            reg_stats["dense_flow"] = flow_stats
        reg_stats["label"] = str(view.get("label") or f"view_{index:02d}")
        reg_stats["photometric"] = refine_stats
        reg_stats["pose"] = pose_stats
        if view_half_extent is not None:
            reg_stats["ortho_half_extent"] = round(float(view_half_extent), 4)
        if "frame_center_dx_px" in view:
            # Projector-frame registration offset (see the recenter above);
            # harnesses reconstructing per-view visibility from input.png
            # must recenter with the same offset to attribute regions
            # faithfully.
            reg_stats["frame_center_dx_px"] = view["frame_center_dx_px"]
            reg_stats["frame_center_dy_px"] = view["frame_center_dy_px"]
        registration_stats.append(reg_stats)
        # Keep the registered image on the view so the mirror-symmetry pass
        # reuses the aligned pixels rather than the raw photo.
        view["rgba"] = rgba_image
        projection = _tripo_project_observed_texture(
            rgba_image,
            mesh=mesh,
            positions_texture=positions_texture,
            normals_texture=normals_texture,
            azimuth_deg=view_azimuth,
            elevation_deg=view_elevation,
            camera_distance=view_distance,
            projection_model=str(projection_model),
            ortho_half_extent=view_half_extent,
            # Per-role facing threshold (ortho multi-view only): beyond
            # ~66 degrees off-axis the source's samples are stretched rim
            # content; where reference photos exist they are the better
            # witness, so the source stops painting earlier. Single-view
            # and perspective bakes keep the wide threshold — stretched
            # content beats no content when nothing else covers the area.
            # GENERATED references do not tighten the source's gate: a
            # real photo's stretched rim content still outranks plausible
            # synthesis, so the source keeps single-view semantics unless
            # at least one REAL reference photo exists.
            facing_threshold=(
                0.4 if (orthographic and index == 1 and has_real_references) else 0.2
            ),
        )
        projection["label"] = str(view.get("label") or f"view_{index:02d}")
        projection["azimuth_deg"] = view_azimuth
        projection["elevation_deg"] = view_elevation
        projection["role"] = str(view.get("role") or ("source" if index == 1 else "reference"))
        if view.get("generated"):
            # Synthesized views are plausible witnesses, not photographs:
            # they must lose every per-texel contest against real photo
            # content, and only own texels no real view covers. A uniform
            # weight attenuation keeps them subordinate in blending and
            # conflict resolution while leaving their exclusive-region
            # coverage intact.
            projection["generated"] = True
            projection["weight"] = (
                np.asarray(projection["weight"], dtype=np.float32)
                * float(generated_reference_weight)
            )
        projections.append(projection)

    # Remove per-view baked-in shading DIFFERENCES before any tone gating:
    # photos disagree on shared surface points as a smooth function of the
    # surface normal (each photo's own lighting), and that disagreement
    # otherwise survives into view-handoff tone steps and gets doubled by
    # viewer relighting. The SH-in-normal-space ratio fit cancels albedo
    # exactly on overlap texels, corrects references to the source's light,
    # and reverts per view unless the overlap disagreement measurably drops
    # (see `delight_projections`).
    #
    # (An adversarial bisect measured the UNFADED version of this stage
    # relighting a profile's whole exclusive side — identity MAE vs its own
    # photo 26.4 -> 39.5 — because the overlap-only gate could not see the
    # exclusive-region damage. Re-enabled with the overlap-proximity fade:
    # the correction now applies fully only near the overlap surface where
    # seams form, fades to zero deep inside each reference's exclusive
    # territory (view centers stay bit-identical to their own photo), and
    # the stats row reports `exclusive_mean_abs_delta` so any residual
    # exclusive-side drift is measurable per bake.)
    delight_stats: Dict[str, Any] = {"applied": False}
    if len(projections) > 1:
        delight_stats = delight_projections(
            projections,
            normals_texture=normals_texture,
            positions_texture=positions_texture,
            source_index=0,
        )

    # CONSENSUS TONE LEVEL (order-0 fallback; see `equalize_projection_tone`):
    # independently synthesized views are mutually tone-inconsistent as a
    # LEVEL, not just as normal-dependent shading (measured on the car
    # candidate: back reads 3x brighter than top_rear on 7.7k shared
    # texels), and when the SH fit above cannot hold all pairs it reverts
    # per view, shipping a half-relit set whose ownership boundaries step
    # by the residual level (handoff p50 0.214, 93-99% luminance). The
    # scalar consensus solve settles those levels with the same gauge,
    # fade and fail-closed revert contracts. Enabled where generated views
    # participate (the measured failure class); real-photo-only multi-view
    # bakes keep today's behavior until their own A/B — the same
    # measured-validation scoping as strand_comb / tone_bottom_cap.
    tone_level_stats: Dict[str, Any] = {"applied": False}
    if len(projections) > 1 and any(p.get("generated") for p in projections):
        tone_level_stats = equalize_projection_tone(
            projections,
            positions_texture=positions_texture,
            source_index=0,
        )

    # Harmonize and quality-gate references IN ATLAS SPACE against the union
    # of everything already accepted (source first, then references in
    # order): overlap texels compare the same physical surface points,
    # unlike whole-image statistics which mix different content. Gating
    # against the union rather than the source alone also catches
    # reference-vs-reference conflicts (two photos fighting over the same
    # side of the object) that the source never observes. References that
    # still disagree after harmonization are misregistered or inconsistent
    # (common with synthesized views), and blending them would ghost the
    # texture, so their weights are attenuated and, at extreme disagreement,
    # rejected. (A mirrored-pose retry for swapped left/right labels was
    # tried here and reverted: its acceptance signal — low disagreement on a
    # small accidental overlap — also fired on correctly labeled views and
    # stole coverage from the true side.)
    consistency_stats: List[Dict[str, Any]] = []
    if len(projections) > 1:
        union_projection = {
            "rgba": np.array(projections[0]["rgba"], dtype=np.float32, copy=True),
            "weight": np.array(projections[0]["weight"], dtype=np.float32, copy=True),
        }
        for position, projection in enumerate(projections[1:], start=1):
            stats_row = harmonize_and_gate_projection(
                projection,
                source_projection=union_projection,
                harmonize=harmonize_references,
                # The per-texel conflict resolution below handles localized
                # disagreement (hair over cheek bands); the global gate here
                # only needs to catch broadly inconsistent references, so it
                # engages later than its per-texel counterpart.
                attenuate_above=0.24,
                reject_above=0.4,
            )
            stats_row["label"] = projection.get("label")
            consistency_stats.append(stats_row)
            # Fold the (possibly attenuated) reference into the union so the
            # next reference is checked against all accepted content.
            reference_weight = np.asarray(projection["weight"], dtype=np.float32)
            take = reference_weight > union_projection["weight"]
            if take.any():
                union_projection["rgba"][take] = np.asarray(projection["rgba"], dtype=np.float32)[take]
                union_projection["weight"][take] = reference_weight[take]
        conflict_stats = resolve_projection_conflicts(projections)
        if consistency_stats:
            consistency_stats[-1]["conflict_resolution"] = conflict_stats

    # Generated views are completion-only: after tone/consistency stages
    # (which need the overlap texels for their statistics), synthesized
    # weight is removed wherever real evidence exists so the feathered
    # blend can never average plausible synthesis into the user's photo
    # (see `protect_observed_texels`). ABSOLUTE mode: the historical
    # sub-floor ramp handed the photo's weakest witnessed band (grazing
    # facing, concavity demotions — 25% of the fresh-car's witnessed
    # atlas) to references at up to 30x the photo's weight, the largest
    # measured witnessed-surface contamination channel of the source-pose
    # fidelity regression (39 dE mean |candidate - baseline| over that
    # band; decomposition in /tmp/gfix2/report.md).
    generated_protection_stats = protect_observed_texels(
        projections, mode="absolute")

    # FILM-BAND COMMITMENT (multi-view only; see film_band.py): extend the
    # layered zone into fused film bands under multi-view consensus, vacate
    # mixture claims exactly where the film tone will take over, and mark
    # the extension contested (mirror-source exclusion). Runs after
    # delight/harmonization so the vacate sees final weights, before the
    # blend so vacated claims never anchor coverage.
    from .film_band import commit_film_band, demote_unwitnessed_rim, retone_film_band

    film_state = commit_film_band(
        projections,
        surface_mask=surface_mask,
        positions_texture=positions_texture,
    )
    film_stats: Dict[str, Any] = (
        dict(film_state["stats"]) if film_state else {"applied": False}
    )

    blend = blend_projections(
        projections,
        atlas_shape=surface_mask.shape,
        sharpness=float(blend_sharpness),
        feather_texels=float(feather_texels),
        detail_fusion=str(detail_fusion),
    )
    if film_state is not None:
        zone_union = np.zeros(surface_mask.shape, dtype=bool)
        for projection in projections:
            maps = projection.get("film_band")
            if isinstance(maps, dict):
                zone_union |= maps["zone_texel"] | maps["commit_texel"]
        film_stats["rim_demoted"] = demote_unwitnessed_rim(
            blend, zone_union=zone_union)
    observed_weight = np.asarray(blend["weight"], dtype=np.float32)
    # Gate completion/fill on true coverage, not the feathered blend weights,
    # so seam-adjacent observed texels are never treated as hidden surface.
    observed_mask = np.asarray(blend.get("coverage"), dtype=bool)
    if observed_mask.shape != surface_mask.shape:
        observed_mask = observed_weight > 1e-6

    # CAPTURE EFFICIENCY per view (self-healing contract): coverage means
    # nothing without the denominator of what the pose COULD have painted.
    # facing_fraction = surface texels facing the view's camera above the
    # projector's own cutoff; capture_efficiency = the view's realized
    # coverage over that fraction. Measured separation on the labeled
    # fleet: healthy sources 0.40-0.75, pose-broken bakes 0.21-0.26 — the
    # most general of the coverage floors because it normalizes for
    # subject shape (a car's single view CAN only see ~35% of its surface;
    # painting 30% of the surface is excellent there and catastrophic on
    # an owl).
    try:
        from .backends.triposr_runtime import _tripo_camera_position

        unit_normals = np.asarray(normals_texture, dtype=np.float32)[:, :, :3]
        norm_len = np.linalg.norm(unit_normals, axis=2, keepdims=True)
        unit_normals = np.divide(unit_normals, np.maximum(norm_len, 1e-8))
        surface_total = max(int(surface_mask.sum()), 1)
        for row, projection in zip(blend["view_stats"], projections):
            eye = _tripo_camera_position(
                azimuth_deg=float(projection.get("azimuth_deg", 0.0)),
                elevation_deg=float(projection.get("elevation_deg", 0.0)),
                camera_distance=float(camera_distance),
            )
            direction = eye / max(float(np.linalg.norm(eye)), 1e-8)
            facing = (unit_normals @ direction) > 0.2
            facing_fraction = float((facing & surface_mask).sum()) / surface_total
            painted = (
                np.asarray(projection["weight"], dtype=np.float32) > 0.0
            ) & surface_mask
            row["facing_fraction"] = round(facing_fraction, 4)
            row["coverage_ratio"] = round(float(painted.sum()) / surface_total, 4)
            row["capture_efficiency"] = round(
                (float(painted.sum()) / surface_total) / max(facing_fraction, 1e-6), 4)
    except Exception:
        pass

    outlier_stats: Dict[str, Any] = {"dropped_texels": 0}
    if projections:
        # Rim misprojections (a hair-shell tip catching forehead pixels
        # from a profile photo, dark background-adjacent pixels stamped on
        # a hull at grazing angles) survive every per-view test but stand
        # out on the surface: foreign winning view and/or foreign color
        # relative to the mesh neighborhood. Demote them to unobserved so
        # the completion/fill stages replace them with surface-consistent
        # color. Single-view bakes rely on the color-extreme condition
        # alone.
        drop = filter_projection_outliers(
            mesh,
            positions_texture=positions_texture,
            projections=projections,
            blended_rgb=blend["rgb"],
            observed_mask=observed_mask,
        )
        if drop.any():
            observed_mask = observed_mask & ~drop
            observed_weight = np.where(drop, 0.0, observed_weight)
            outlier_stats["dropped_texels"] = int(drop.sum())

    # The DIRECT-painted union: photo projection after every gate and the
    # outlier filter, before any completion writes. This is the honest
    # "what the photos actually painted" set the leverage ledger reports.
    direct_union_mask = np.array(observed_mask, dtype=bool)

    # Compositing mode. The gradient-domain path composites the views in
    # the gradient domain with ONE screened-Poisson solve over the
    # observed texel surface graph, replacing the legacy additive seam
    # leveling (which stays in place for compositing="legacy"). It runs
    # HERE — after outlier filtering, before completion — so mirror
    # copies and the harmonic fill propagate tone-equalized colors, the
    # same relative order the legacy path gives its leveling offsets.
    # "auto" resolves per view count: multi-view bakes carry cross-view
    # tone seams and winner-take-all handoffs — exactly what the gradient
    # solve eliminates (measured better on both QA harnesses on the face
    # lane). Single-view bakes have no cross-view seams; measured on the
    # starship lane the solve changes the composite by <1/255 there and
    # only jitters threshold-marginal detectors, so legacy stays the
    # single-view default (gradient_domain remains explicitly selectable).
    compositing_requested = str(compositing or "auto").strip().lower()
    if compositing_requested == "auto":
        compositing_resolved = (
            "gradient_domain" if len(projections) > 1 else "legacy"
        )
    else:
        compositing_resolved = compositing_requested
    compositing_stats: Dict[str, Any] = {
        "requested": compositing_requested,
        "mode": compositing_resolved,
        "applied": False,
    }
    if compositing_resolved == "gradient_domain" and projections:
        import time as _time

        from .gradient_compositing import composite_gradient_domain

        class_map = np.where(observed_mask, 0, -1).astype(np.int32)
        observed_positions = np.array(positions_texture, dtype=np.float32)
        observed_positions[:, :, 3] = np.where(
            observed_mask, observed_positions[:, :, 3], 0.0
        )
        blend_rgba = np.concatenate(
            [
                np.asarray(blend["rgb"], dtype=np.float32),
                observed_mask.astype(np.float32)[:, :, None],
            ],
            axis=2,
        )
        # PHOTO-ANCHOR PIN (generated-references bakes only): the solve's
        # screening is proportional to blend confidence, so weakly-
        # witnessed photo texels (grazing facing, weights 0.02-0.4) sit
        # on an equalization decay length of 20-60 texels — long enough
        # for the tone step at every photo|reference ownership boundary
        # to redistribute INTO the photo's own surface. Measured on the
        # fresh-draw car (/tmp/gfix2): texels witnessed ONLY by the
        # photo, with every generated weight zero, still moved 14.1 dE
        # mean from the no-references baseline, and skipping the solve
        # collapsed that channel to 1.7 — the solve was the carrier.
        # Where the only other witnesses are synthesized, equalizing the
        # photo toward them inverts the evidence ranking, so every
        # real-witnessed texel is pinned at full anchor confidence and
        # the source boost applies to the photo's whole witnessed set
        # (`source_confidence_floor` ~ 0): drift is then bounded by
        # 1/sqrt(lambda_max) ~ 9 texels at 1024 — the blend feather's own
        # handoff scale — and the boundary tone steps resolve on the
        # reference side (their anchors stay soft). Bakes with REAL
        # reference photos keep proportional screening: real cross-view
        # tone disagreement is exactly what the solve exists to
        # equalize.
        solve_anchor_confidence = observed_weight
        solve_kwargs: Dict[str, Any] = {}
        if not has_real_references and any(
                p.get("generated") for p in projections):
            real_witnessed = np.zeros(surface_mask.shape, dtype=bool)
            for projection in projections:
                if not projection.get("generated"):
                    real_witnessed |= (
                        np.asarray(projection["weight"], dtype=np.float32)
                        > 0.0)
            solve_anchor_confidence = np.where(
                real_witnessed, 1.0, observed_weight).astype(np.float32)
            solve_kwargs["source_confidence_floor"] = 1e-6
        solve_started = _time.perf_counter()
        solved = composite_gradient_domain(
            positions_texture=observed_positions,
            normals_texture=normals_texture,
            view_rgb=[np.asarray(p["rgba"], dtype=np.float32) for p in projections],
            view_weight=[np.asarray(p["weight"], dtype=np.float32) for p in projections],
            class_map=class_map,
            filled_rgb=blend_rgba,
            anchor_confidence=solve_anchor_confidence,
            view_valid=[
                np.asarray(p["rgba"], dtype=np.float32)[:, :, 3] > 0.0
                for p in projections
            ],
            resolution_reference=1024,
            # Relative residual 1e-5 leaves smooth-mode errors far below
            # one uint8 level (measured: tightening to 5e-6 changes the
            # texture by <1/255 everywhere) and saves ~25% of iterations.
            cg_tol=1e-5,
            **solve_kwargs,
        )
        if solved is not None:
            solved_rgb, solver_stats = solved
            blend["rgb"] = np.where(
                observed_mask[:, :, None], solved_rgb, blend["rgb"]
            )
            compositing_stats["applied"] = True
            compositing_stats["solve_seconds"] = round(
                _time.perf_counter() - solve_started, 2
            )
            compositing_stats["solver"] = {
                key: solver_stats.get(key)
                for key in (
                    "nodes",
                    "edges",
                    "cg_iterations",
                    "final_relative_residual",
                    "edge_kind_counts",
                    "mg_level_sizes",
                    "gradient_rule",
                    "anchor_lambda_scale",
                    "line_edge_weight",
                    "specular_reconcile",
                    "shadow_reconcile",
                )
            }

    # WITNESS-SCARCITY ADMISSION (G1; see `admit_scarce_witnesses`): on
    # texels no view claims at its strict facing threshold, admit the
    # stretch-bounded below-threshold claims — a real observation beats a
    # symmetry guess or fill wherever it is the only witness. Runs HERE,
    # after the global compositing solve and before mirror completion, as
    # a STRICTLY LOCAL paint (the "commit local repairs late" rule: an
    # earlier admission measurably perturbed the Poisson anchor set
    # globally and flipped knife-edge debris detectors far from any
    # rescued texel); before mirror, so symmetry never guesses surface a
    # real witness paints. Rescued colors carry the delight/harmonize
    # tone corrections (those stages apply their fields to scarce
    # candidates too); film-committed texels are excluded (surrender +
    # commitment is one coupled decision).
    scarcity_requested = str(scarcity_rescue or "auto").strip().lower()
    # "auto" keys on REAL reference photos: generated views must not flip
    # the single-photo canaries into the rescue regime (the bake docstring
    # pins that regime change behind its own A/B).
    scarcity_enabled = (
        (len(projections) > 1 and has_real_references)
        if scarcity_requested == "auto"
        else scarcity_requested in ("on", "true", "1", "yes")
    )
    scarcity_stats: Dict[str, Any] = {
        "requested": scarcity_requested,
        "enabled": bool(scarcity_enabled),
        "applied": False,
    }
    rescued_admitted_mask: Optional[Any] = None
    if scarcity_enabled and projections:
        scarcity_stats.update(admit_scarce_witnesses(
            projections, surface_mask=surface_mask,
            positions_texture=positions_texture,
            exclude_mask=(film_state["commit_mask"]
                          if film_state is not None else None)))
        rescued_admitted_mask = scarcity_stats.pop("admitted_mask", None)
        scarcity_stats.pop("refused_mask", None)
        if rescued_admitted_mask is not None and rescued_admitted_mask.any():
            rescue_weight_stack = np.stack(
                [np.asarray(p["weight"], dtype=np.float32)
                 for p in projections], axis=0)
            rescue_winner = rescue_weight_stack.argmax(axis=0)
            sel_rows, sel_cols = np.nonzero(rescued_admitted_mask)
            rescue_rgb = np.stack(
                [np.asarray(p["rgba"], dtype=np.float32)[:, :, :3]
                 for p in projections], axis=0)
            blend["rgb"][sel_rows, sel_cols] = rescue_rgb[
                rescue_winner[sel_rows, sel_cols], sel_rows, sel_cols]
            observed_mask = observed_mask | rescued_admitted_mask
            observed_weight = np.maximum(
                observed_weight,
                np.where(rescued_admitted_mask,
                         rescue_weight_stack.max(axis=0), 0.0))
            direct_union_mask |= rescued_admitted_mask
            del rescue_rgb

    # "auto" completion: apply mirror completion whenever the geometry
    # itself is measurably left-right symmetric. Rationale (measured on the
    # single-view starship proof): a single photo observes a thin sliver of
    # a symmetric object (6-24% of texels), and the mirrored twin of that
    # sliver is REAL content where any propagated fill is a characterless
    # wash; the same geometry gate that protects explicit mirror requests
    # (score >= 0.55, plus confident-source and contested-exclusion gates)
    # protects asymmetric subjects here. Callers that pass "none" keep
    # today's behavior.
    texture_completion_requested = str(texture_completion or "none").strip().lower()
    texture_completion_resolved = texture_completion_requested
    if texture_completion_requested == "auto":
        auto_symmetry_score = mesh_mirror_symmetry_score(mesh, axis=1)
        texture_completion_resolved = (
            "mirror_symmetry" if auto_symmetry_score >= 0.55 else "none"
        )

    symmetry_stats: Dict[str, Any] = {"mode": "none", "coverage_ratio": 0.0, "applied": False}
    mirror_added_mask: Optional[Any] = None
    if texture_completion_resolved == "mirror_symmetry" and views:
        # Mirror completion assumes the geometry really is left-right
        # symmetric; verify before copying mirrored colors.
        symmetry_score = mesh_mirror_symmetry_score(mesh, axis=1)
        symmetry_stats["geometry_symmetry_score"] = round(symmetry_score, 4)
        if symmetry_score >= 0.55:
            # Mirror sources must be CLEAN witnesses: measured on the face
            # lane, >90% of hairline flake islands were mirror/harmonic
            # COPIES of a few low-confidence mixture anchors, not direct
            # projections. Two gates: (1) texels inside any view's
            # contested layered band (see the projector's layered-zone
            # gate) may keep their own claim but are excluded as mirror
            # sources; (2) the source floor rises 0.15 -> 0.35 — with the
            # contested band surrendered, the anchors that remain are
            # confident, and the earlier coverage argument for 0.15 no
            # longer applies. (Removing mirror completion entirely was
            # measured worse: the far cheek degrades to harmonic mush.)
            mirror_source_weight = observed_weight
            contested_union = np.zeros(surface_mask.shape, dtype=bool)
            for projection in projections:
                projection_contested = np.asarray(projection.get("contested", False))
                if projection_contested.shape == contested_union.shape:
                    contested_union |= projection_contested
            if contested_union.any():
                mirror_source_weight = np.where(contested_union, 0.0, observed_weight)
            fill_rgb, fill_mask = mirror_fill_from_observed(
                positions_texture=positions_texture,
                observed_mask=observed_mask,
                colors_rgb=blend["rgb"],
                observed_weight=mirror_source_weight,
                min_source_weight=0.35,
            )
            add_mask = fill_mask & surface_mask & ~observed_mask
            if film_state is not None:
                # Mirror copies inside the COMMITTED film region read as
                # pale flakes on the dark fill (the twin side's mixtures
                # are no better witnesses than the vacated ones); the
                # retone owns those texels instead. The wider un-committed
                # band keeps mirror behavior — banning it there replaced
                # coherent pale ribbons with darker membrane and pushed
                # the rear-quarter crown-flake contrast over its gate.
                band_removed = add_mask & film_state["commit_mask"]
                if band_removed.any():
                    add_mask = add_mask & ~film_state["commit_mask"]
                    film_stats["mirror_destinations_removed"] = int(
                        band_removed.sum())
            # Completion tone match (FACE-22, multi-view only): reconcile
            # each pure-bright copy's tone to its destination ring so the
            # component border cannot print as a contour. Multi-view only
            # this cycle for the same reason as the strand comb /
            # tone_bottom_cap: the single-photo proof assets are pinned
            # regression canaries whose bakes must stay bit-identical;
            # enabling there needs its own A/B.
            tone_matched_texels = 0
            if add_mask.any() and len(projections) > 1:
                fill_rgb, tone_matched_texels = tone_match_completion_components(
                    fill_rgb,
                    add_mask=add_mask,
                    colors_rgb=blend["rgb"],
                    observed_mask=observed_mask,
                    surface_mask=surface_mask,
                )
            if add_mask.any():
                blend["rgb"][add_mask] = fill_rgb[add_mask]
                observed_mask |= add_mask
                mirror_added_mask = add_mask
                observed_weight = np.maximum(observed_weight, add_mask.astype(np.float32) * 0.85)
                symmetry_stats = {
                    "mode": "mirror_symmetry",
                    "geometry_symmetry_score": round(symmetry_score, 4),
                    "coverage_ratio": round(
                        float(np.count_nonzero(add_mask)) / float(max(int(surface_mask.sum()), 1)), 4
                    ),
                    "tone_matched_texels": int(tone_matched_texels),
                    "applied": True,
                }

    # Direct (photo-witnessed) coverage, mirror copies excluded: consumers
    # downstream (e.g. the synthesized-texel sweep) need to tell photo
    # truth from completion.
    direct_observed_mask = np.array(observed_mask, dtype=bool)
    if mirror_added_mask is not None:
        direct_observed_mask &= ~np.asarray(mirror_added_mask, dtype=bool)

    # Mirror TWIN RESCUE: mirror completion above only writes UNOBSERVED
    # texels, but on near-symmetric subjects a feature region can be
    # observed yet badly witnessed — every covering view sees it at grazing
    # incidence or through a misregistered duplicate reference — leaving a
    # smear no downstream gate can repair (all covering witnesses agree on
    # the wrong content; measured on the face lane's -90 profile eye:
    # witness ball-weight 0.16 vs the healthy twin's 0.55, harness
    # eye_count 0). Detect strong feature discs whose twin is weakly
    # witnessed AND feature-empty (`detect_mirror_rescue_discs` documents
    # the gates that keep legitimately asymmetric content untouched) and
    # transplant the healthy content tone-matched (`mirror_rescue_disc`).
    # Runs after mirror completion so detection sees final direct coverage,
    # under the same geometry-symmetry gate; transplanted texels count as
    # completion (not photo truth), exactly like mirror-filled ones.
    rescue_stats: Dict[str, Any] = {"applied": False, "discs": []}
    rescue_protected_mask: Optional[Any] = None
    rescue_symmetry_score = symmetry_stats.get("geometry_symmetry_score")
    if rescue_symmetry_score is None:
        rescue_symmetry_score = round(mesh_mirror_symmetry_score(mesh, axis=1), 4)
    rescue_stats["geometry_symmetry_score"] = rescue_symmetry_score
    if float(rescue_symmetry_score) >= 0.55 and projections:
        # Witness quality must be the RAW per-view winner weight (the best
        # photo evidence at each texel), not the blend's feathered weight:
        # feathering zeroes a 6-texel rim band and drags ball means down,
        # which both starves the detector's direct set and shifts every
        # threshold (the gates are calibrated on raw witness weights).
        raw_winner_weight = np.stack(
            [np.asarray(p["weight"], dtype=np.float32) for p in projections],
            axis=0,
        ).max(axis=0)
        direct_weight = np.where(direct_observed_mask, raw_winner_weight, 0.0)
        rescue_discs = detect_mirror_rescue_discs(
            positions_texture=positions_texture,
            colors_rgb=blend["rgb"],
            observed_mask=direct_observed_mask,
            observed_weight=direct_weight,
            axis=1,
        )
        rescue_source_mask = direct_observed_mask & (raw_winner_weight >= 0.35)
        positions_array = np.asarray(positions_texture, dtype=np.float32)
        for disc in rescue_discs:
            rescued_rgb, disc_stats = mirror_rescue_disc(
                blend["rgb"],
                positions_texture=positions_texture,
                center=disc["center"],
                radius=float(disc["radius"]),
                axis=1,
                source_mask=rescue_source_mask,
                source_shift=disc.get("placement_shift"),
            )
            if not disc_stats.get("rescued_texels"):
                continue
            blend["rgb"] = rescued_rgb
            disc_center = np.asarray(disc["center"], dtype=np.float32)
            disc_distance = np.linalg.norm(
                positions_array[:, :, :3] - disc_center[None, None, :], axis=2)
            same_side = (
                positions_array[:, :, 1] * disc_center[1]
            ) >= 0.0 if abs(float(disc_center[1])) > 1e-6 else np.ones(
                surface_mask.shape, dtype=bool)
            disc_mask = surface_mask & same_side & (disc_distance < float(disc["radius"]))
            observed_mask |= disc_mask
            direct_observed_mask &= ~disc_mask
            observed_weight = np.maximum(
                observed_weight, disc_mask.astype(np.float32) * 0.85)
            rescue_stats["discs"].append({**disc, **disc_stats})
            # transplanted content sits over the ORIGINAL trace-weight
            # witnesses; the deposit commit below must neither judge nor
            # recolor it (measured: an unprotected retone erased the
            # rescued -90 profile eye). The margin covers the feather.
            protected = surface_mask & same_side & (
                disc_distance < 1.3 * float(disc["radius"]))
            rescue_protected_mask = protected if rescue_protected_mask is None \
                else (rescue_protected_mask | protected)
        rescue_stats["applied"] = bool(rescue_stats["discs"])

    leveling_stats: Dict[str, Any] = {"applied": False}
    if len(projections) > 1 and compositing_resolved != "gradient_domain":
        # Multi-view composition stitches regions from different sources
        # (per-texel winning view, mirror completion) whose tones disagree
        # at the handoffs even after global harmonization: the residual is
        # CONTENT-level (baked-in shading, synthetic-vs-photo skin tone) and
        # shows up as a visible step at every region border (the face
        # lane's mid-face vertical seam). Solve a low-frequency per-region
        # offset field on the mesh graph that cancels those steps while
        # confident witnesses stay pinned to their own truth. Single-view
        # bakes have no cross-view seams and skip this entirely.
        weight_stack = np.stack(
            [np.asarray(p["weight"], dtype=np.float32) for p in projections], axis=0
        )
        winner_index = weight_stack.argmax(axis=0)
        winner_weight = weight_stack.max(axis=0)
        region_map = np.full(surface_mask.shape, -1, dtype=np.int32)
        direct_region = observed_mask & (winner_weight > 1e-6)
        region_map[direct_region] = winner_index[direct_region]
        if mirror_added_mask is not None:
            region_map[np.asarray(mirror_added_mask, dtype=bool)] = len(projections)
        seam_offsets = level_composed_seams(
            mesh,
            positions_texture=positions_texture,
            colors_rgb=blend["rgb"],
            region_map=region_map,
            confidence_map=winner_weight,
        )
        if seam_offsets is not None:
            blend["rgb"] = np.clip(blend["rgb"] + seam_offsets, 0.0, 1.0)
            magnitudes = np.abs(seam_offsets[region_map >= 0])
            leveling_stats = {
                "applied": True,
                "mean_offset": round(float(magnitudes.mean()), 4),
                "p90_offset": round(float(np.percentile(magnitudes, 90)), 4),
            }

    colors = np.zeros((*surface_mask.shape, 4), dtype=np.float32)
    colors[:, :, :3] = blend["rgb"]
    colors[observed_mask, 3] = 1.0

    fill_mode = "nearest_observed_3d"
    if base_color_fn is None:
        # Prefer surface-connectivity-aware harmonic diffusion over Euclidean
        # borrowing; fall back to the KD-tree fill when the solve is
        # unavailable.
        harmonic = mesh_graph_harmonic_fill(
            mesh,
            positions_texture=positions_texture,
            observed_mask=observed_mask,
            colors_rgba=colors,
        )
        if harmonic is not None:
            colors = harmonic
            fill_mode = "mesh_harmonic"
    if fill_mode != "mesh_harmonic" and base_color_fn is not None:
        try:
            base_colors = np.asarray(base_color_fn(positions_texture), dtype=np.float32)
            unseen = surface_mask & ~observed_mask
            colors[unseen, :3] = base_colors[unseen, :3]
            colors[unseen, 3] = 1.0
            fill_mode = "backend_color_field"
        except Exception:
            colors = inpaint_unseen_texels(
                colors,
                surface_mask=surface_mask,
                observed_mask=observed_mask,
                positions_texture=positions_texture,
                normals_texture=normals_texture,
            )
    elif fill_mode != "mesh_harmonic":
        colors = inpaint_unseen_texels(
            colors,
            surface_mask=surface_mask,
            observed_mask=observed_mask,
            positions_texture=positions_texture,
            normals_texture=normals_texture,
        )

    fill_detail_stats: Dict[str, Any] = {"applied": False, "gain": float(fill_detail_gain)}
    repaint_floor_exempt: Optional[Any] = None
    trace_deposit_stats: Dict[str, Any] = {"applied": False}
    pale_chip_stats: Dict[str, Any] = {"applied": False}
    bottom_cap_stats: Dict[str, Any] = {"applied": False}
    fringe_repair_stats: Dict[str, Any] = {"applied": False}
    if fill_mode in ("mesh_harmonic", "nearest_observed_3d"):
        # Both propagated fills leave texel-scale defects that the backend
        # color-field path does not have: coarse-proxy discontinuities
        # (vertex-cell seams, KD donor-set patchwork) and zero micro-texture
        # (flat painted wash). The smoothing pass makes the fill C0 at texel
        # resolution; the detail pass then restores observed-statistics
        # micro-texture so hidden surface reads as the same material as the
        # observed surface. Order matters: smoothing after detail would
        # erase it.
        colors = texel_surface_smooth(
            colors,
            positions_texture=positions_texture,
            normals_texture=normals_texture,
            observed_mask=observed_mask,
        )
        if film_state is not None:
            # Film-band tone repair runs AFTER the smoothing pass
            # (smoothing would pull the repainted tone back toward the
            # membrane) and before detail synthesis so the synthesized
            # micro-texture picks its donors by the repaired color (hair
            # statistics, not skin). The gradient REPAINT
            # (film_band_gradient.py) rebuilds the whole hairline apron —
            # source-photo content where the source witnesses it, the
            # photos' own hair-to-skin falloff elsewhere; when its
            # prerequisites are missing (no dark mass / skin ring /
            # falloff profile) the cycle-2 dark-anchor retone of the
            # committed texels remains the fallback.
            from .film_band_gradient import repaint_film_band

            repainted = repaint_film_band(
                projections,
                colors,
                positions_texture=positions_texture,
                normals_texture=normals_texture,
                observed_mask=observed_mask,
                texture_resolution=resolution,
            )
            if repainted is not None:
                colors, repaint_stats, repaint_applied_mask = repainted
                film_stats["gradient_repaint"] = repaint_stats
                # Repainted texels carry the photo's own falloff (source
                # stamps / profile tone); the statistical fill-luminance
                # floor must not overrule them (measured at 1024: the
                # floor re-lifted 17% of the darkened curtain texels into
                # pale shreds that read as skin islands).
                repaint_floor_exempt = repaint_applied_mask
            else:
                colors, retone_stats = retone_film_band(
                    colors,
                    positions_texture=positions_texture,
                    observed_mask=observed_mask,
                    commit_mask=film_state["commit_mask"],
                    body_weight=film_state["body_weight"],
                    normals_texture=normals_texture,
                )
                film_stats["retone"] = retone_stats
        # Trace-deposit commit (multi-view only; see commit_trace_deposits):
        # chips/dashes of displaced view content at trace witness weight,
        # contradicted by the multi-witness bright-material consensus of
        # their surround, are retoned from their validated ring anchors.
        # Runs HERE — after mirror completion, rescue and film retone,
        # before detail synthesis — as a strictly local recolor: committing
        # earlier measurably cascades through the global stages (Poisson
        # anchors, rescue localization, fill calibration) and flips
        # knife-edge detectors far from any chip.
        if len(projections) > 1:
            colors = commit_trace_deposits(
                colors,
                positions_texture=positions_texture,
                observed_mask=observed_mask,
                projections=projections,
                film_commit_mask=(film_state["commit_mask"]
                                  if film_state is not None else None),
                protected_mask=rescue_protected_mask,
                stats_out=trace_deposit_stats,
            )
            # Pale-chip commit (the dark-context DUAL; see
            # commit_pale_chips): isolated pale islands in hair whose
            # plain ring every witness reads uniformly dark — the
            # FACE-07 ear-band class. Same placement rationale as the
            # bright-context commit (late, strictly local), same
            # multi-view requirement (single-view ring consensus is
            # vacuous), so single-photo canaries are structurally
            # untouchable.
            colors = commit_pale_chips(
                colors,
                positions_texture=positions_texture,
                observed_mask=observed_mask,
                projections=projections,
                film_commit_mask=(film_state["commit_mask"]
                                  if film_state is not None else None),
                protected_mask=rescue_protected_mask,
                stats_out=pale_chip_stats,
            )
            # Synthetic-cut-face toning (see tone_bottom_cap): the bust
            # disc's tone comes from its own rim's observed content
            # instead of the global fill. Multi-view bakes only this
            # cycle: the single-photo proof assets are pinned regression
            # canaries; enabling there needs its own A/B (same scoping
            # precedent as the strand comb).
            colors = tone_bottom_cap(
                colors,
                positions_texture=positions_texture,
                normals_texture=normals_texture,
                observed_mask=observed_mask,
                direct_observed_mask=direct_observed_mask,
                stats_out=bottom_cap_stats,
            )
        if float(fill_detail_gain) > 0.0:
            colors = synthesize_fill_detail(
                colors,
                positions_texture=positions_texture,
                normals_texture=normals_texture,
                observed_mask=observed_mask,
                gain=float(fill_detail_gain),
                # The strand-comb regime (combed low-contrast statistics on
                # dark anisotropic-donor fill; see synthesize_fill_detail)
                # is enabled for multi-view bakes: it is validated on
                # multi-view anchors this cycle, and the single-photo proof
                # assets are pinned regression canaries whose bakes must
                # stay bit-identical; enabling it there needs its own A/B.
                strand_comb=len(projections) > 1,
                stats_out=fill_detail_stats,
            )
            fill_detail_stats["applied"] = True

    # Final consensus sweep over FILL texels: propagated fill may not ship
    # feature-dark pockets that the local surface context and the observed
    # donors contradict (dark observed anchors transported across hidden
    # surface by the harmonic solve). Runs last so every fill color
    # (harmonic, KD, backend field, detail, composite) is covered.
    #
    # Scope, re-set after adversarial review (critic 2, 2026-07-05) which
    # measured three regressions in the first version and disabled it:
    # - MIRROR-COMPLETED texels are EXEMPT. They carry evidence (their
    #   observed twin), and the local floor cannot distinguish a mirrored
    #   pupil from a defect pocket (measured: +135..628% luminance error
    #   on a constructed pupil-analog). Validating mirror copies is the
    #   mirror stage's job (source-confidence, contested-exclusion, and
    #   consensus guards).
    # - Ball statistics exclude the OPPOSITE sheet only (dominant-axis
    #   direction bins; the earlier hemisphere pooling let the face's rear
    #   hair read forward-facing skin as context and lifted hair fill
    #   toward skin tones — the "skin patches in rear hair" failure).
    # - Donors for the donor-consensus floor are the direct-observed set.
    # Observed and mirror texels are bit-identical by construction.
    # Film-band-repainted texels carry the photos' own falloff (source
    # stamps / measured profile tone) — evidence, not statistics; the
    # floor's donor-consensus prior must not overrule them (measured: it
    # re-lifted darkened curtain texels into pale shreds at 1024).
    fill_floor_mask = surface_mask & ~observed_mask
    if repaint_floor_exempt is not None:
        fill_floor_mask = fill_floor_mask & ~repaint_floor_exempt
    fill_floor_stats: Dict[str, Any] = {"applied": False}
    colors, fill_floor_stats = enforce_fill_luminance_floor(
        colors,
        positions_texture=positions_texture,
        surface_mask=surface_mask,
        synthesized_mask=fill_floor_mask,
        donor_mask=direct_observed_mask,
        normals_texture=normals_texture,
    )

    # UNWITNESSED-DEBRIS CONSOLIDATION (G1 follow-through; see
    # consolidate_unwitnessed_debris): witness-scarcity admission
    # re-partitions the global fill's pockets, and a re-partitioned
    # dark pocket can ship as an isolated debris-class island on skin
    # (measured: one 415 px pocket took three battery views over the
    # absolute debris gate while no rescued texel rendered within 20 px
    # of it). The general rule needs no rescue provenance: an isolated
    # bright-ringed sub-feature dark island that is predominantly
    # UNWITNESSED is a defect by definition — lift its unwitnessed
    # members just above the dark class under each view's own shading
    # (the displaced-refill floor discipline the fringe lane applies to
    # its own stamps). Scoped to bakes where scarcity admission ran, so
    # bakes without it (single-photo canaries, scarcity_rescue="off")
    # stay bit-identical. Runs after every color-writing fill stage and
    # before the fringe repair (whose render evidence and veto baselines
    # must see the shipped state).
    if rescued_admitted_mask is not None and rescued_admitted_mask.any():
        # photo-WITNESS set: direct claims (rescued included) and
        # rescue-disc transplants (twin photo evidence). Mirror copies
        # and film-repaint falloff are deliberately NOT witnesses here:
        # a symmetry guess or a statistical falloff field that renders
        # as an isolated dark island on light material loses its tone
        # (measured: the regression pockets were 71-100% fill/repaint —
        # the repaint's ratio-field tone extrapolated onto skin, the
        # FACE-22 support-bound class). The repaint/mirror exemptions on
        # the fill FLOOR remain untouched (their curtain-shred class is
        # ringed by the dark mass and structurally fails this pass's
        # skin-ring rule); mirrored FEATURES are protected by the
        # feature-blob footprint.
        evidence_mask = np.asarray(direct_union_mask, dtype=bool).copy()
        if rescue_protected_mask is not None:
            evidence_mask |= np.asarray(rescue_protected_mask, dtype=bool)
        colors, rescued_lifted = consolidate_unwitnessed_debris(
            mesh,
            atlas=atlas,
            colors=colors,
            positions_texture=positions_texture,
            normals_texture=normals_texture,
            strict_mask=evidence_mask,
        )
        scarcity_stats["speck_lifted_texels"] = int(rescued_lifted)

    # FEATURE-FRINGE REPAIR (multi-view only; see feature_fringe_repair.py):
    # the deposits commit_trace_deposits measurably CANNOT treat —
    # displaced-content chips/dashes INSIDE protected feature complexes
    # (tear ducts, lash lines, lip edges) whose surround consensus is
    # feature-mixed by construction — are repaired with the photo's own
    # content under the identity correspondence, rescue-transplant
    # semantics (tone match + feather + whole patch), guarded by
    # structure-preservation vetoes. Runs LAST, on the SHIPPED colors:
    # the repair evidence and the vetoes are built by rendering the
    # current texture, and a placement before detail/floor measured those
    # stages repainting fill around the repairs into new isolated
    # micro-islands at six views (the vetoes had judged a texture that
    # never shipped).
    if len(projections) > 1 and views:
        from .feature_fringe_repair import repair_feature_fringes

        # The identity correspondence must be built against the SAME
        # image the identity contract uses — the caller's un-matted
        # source photo when provided (`identity_image` on the source
        # view; bundles ship it as input.png). The matted rgba's tighter
        # silhouette measurably snaps the bbox+NCC registration into a
        # different basin (residual dx 0.065 vs 0.025, alignment SSIM
        # 0.645 vs 0.546): content aligned in the wrong basin banks
        # nothing at the gate.
        colors, fringe_mask = repair_feature_fringes(
            mesh,
            atlas=atlas,
            colors_rgba=colors,
            positions_texture=positions_texture,
            normals_texture=normals_texture,
            projections=projections,
            source_image=(views[0].get("identity_image")
                          or views[0].get("rgba")),
            source_azimuth_deg=float(views[0].get("azimuth_deg", 0.0)),
            source_elevation_deg=float(views[0].get("elevation_deg", 0.0)),
            source_index=0,
            rescue_discs=rescue_stats.get("discs", []),
            observed_mask=observed_mask,
            stats_out=fringe_repair_stats,
        )
        if fringe_mask is not None and fringe_mask.any():
            observed_mask = observed_mask | fringe_mask

    texture_image = _tripo_edge_bleed_texture(_tripo_texture_image(colors))
    textured_mesh = _tripo_build_textured_mesh(
        mesh,
        bake_output={"vmapping": atlas["vmapping"], "indices": atlas["indices"], "uvs": atlas["uvs"]},
        texture_image=texture_image,
    )
    uv_preview = _tripo_uv_preview(texture_image=texture_image, uvs=atlas["uvs"], indices=atlas["indices"])

    coverage_denominator = max(int(surface_mask.sum()), 1)
    mode = "projection_only" if len(projections) <= 1 else "projection_multiview"
    if symmetry_stats["applied"]:
        mode += "_plus_symmetry"
    mode += "_plus_" + ("color_field" if fill_mode == "backend_color_field" else "fill3d")
    leverage_stats = assemble_leverage_ledger(
        projections,
        surface_mask=surface_mask,
        direct_union_mask=direct_union_mask,
        shipped_direct_mask=direct_observed_mask,
        mirror_added_mask=mirror_added_mask,
    )
    stats: Dict[str, Any] = {
        "camera_distance": round(float(camera_distance), 4),
        "source_pose": {
            "azimuth_deg": float(source_pose.get("azimuth_deg", 0.0)),
            "elevation_deg": float(source_pose.get("elevation_deg", 0.0)),
            # HONEST field contract: a measurement that never ran exports
            # null, not 0.0 (the dead always-zero IoU cost an audit real
            # time chasing a phantom signal).
            "silhouette_iou": (
                float(source_pose["iou"]) if source_pose.get("iou") else None
            ),
            "method": source_pose.get("method", "silhouette"),
            "score": source_pose.get("score"),
            "score_at_declared": source_pose.get("score_at_declared"),
            "rejected_reason": source_pose.get("rejected_reason"),
            "estimated": bool(source_pose.get("estimated", estimate_source_pose)),
        },
        "source_registration": (
            {
                "method": "mesh_bbox_center",
                "frame_center_dx_px": views[0].get("frame_center_dx_px", 0.0),
                "frame_center_dy_px": views[0].get("frame_center_dy_px", 0.0),
                "frame_size": 1024,
            }
            if views and "frame_center_dx_px" in views[0]
            else {"method": "photo_bbox_center"}
        ),
        "view_registration": registration_stats,
        "view_consistency": consistency_stats,
        "outlier_filter": outlier_stats,
        "texture_image": texture_image,
        "uv_preview": uv_preview,
        "texture_padding": int(padding),
        "projection_mode": mode,
        "unseen_fill_mode": fill_mode,
        "observed_coverage_ratio": round(
            float(np.count_nonzero(observed_mask & surface_mask)) / float(coverage_denominator), 4
        ),
        "observed_view_stats": blend["view_stats"],
        "leverage": leverage_stats,
        "scarcity_rescue": scarcity_stats,
        "texture_completion": texture_completion_resolved,
        "texture_completion_requested": texture_completion_requested,
        "fill_detail": fill_detail_stats,
        "fill_floor": fill_floor_stats,
        "delight": delight_stats,
        "tone_consensus": tone_level_stats,
        "generated_protection": generated_protection_stats,
        "handoff_seams": blend.get("handoff_seams"),
        "symmetry_completion": symmetry_stats,
        "mirror_rescue": rescue_stats,
        "trace_deposits": {k: v for k, v in trace_deposit_stats.items()
                           if k != "blobs"} | {
            "blob_count": len(trace_deposit_stats.get("blobs", []))},
        "pale_chips": pale_chip_stats,
        "bottom_cap": bottom_cap_stats,
        "feature_fringe_repair": fringe_repair_stats,
        "film_band": film_stats,
        "seam_leveling": leveling_stats,
        "compositing": compositing_stats,
        "blend_sharpness": float(blend_sharpness),
        "feather_texels": float(feather_texels),
        "vertex_mapping_count": int(len(atlas["vmapping"])),
        "uv_vertex_count": int(len(atlas["uvs"])),
        "face_count": int(len(atlas["indices"])),
        "vmapping": atlas["vmapping"],
        "indices": atlas["indices"],
        "uvs": atlas["uvs"],
    }
    return textured_mesh, stats


def _voxel_ball_stats(points: Any, values: Any, cell: float,
                      query_points: Any) -> Tuple[Any, Any, Any]:
    """(count, mean, std) of `values` over points within the 3x3x3 voxel
    neighborhood of each query point (world-space box ball of span
    ~[cell, 2*cell]).

    Same integer-binning construction as `_voxel_neighborhood_mean`, but it
    answers at ARBITRARY query positions (needed for mirror-twin lookups,
    which are not themselves surface points) and returns the count and
    second moment alongside the mean (the rescue detector needs witness
    density and local contrast, not just averages). O(points + queries),
    deterministic, no KD-tree.
    """
    import numpy as np
    from scipy.ndimage import uniform_filter

    pts = np.asarray(points, dtype=np.float64)
    vals = np.asarray(values, dtype=np.float64)
    queries = np.asarray(query_points, dtype=np.float64)
    lo = np.minimum(pts.min(axis=0), queries.min(axis=0))
    hi = np.maximum(pts.max(axis=0), queries.max(axis=0))
    dims = np.maximum(((hi - lo) / float(cell)).astype(np.int64) + 1, 1)
    grid_shape = (int(dims[0]), int(dims[1]), int(dims[2]))
    n_cells = int(np.prod(grid_shape))

    ijk = np.clip(((pts - lo) / float(cell)).astype(np.int64), 0, dims - 1)
    flat = (ijk[:, 0] * dims[1] + ijk[:, 1]) * dims[2] + ijk[:, 2]
    sums = np.bincount(flat, weights=vals, minlength=n_cells).reshape(grid_shape)
    sums2 = np.bincount(flat, weights=vals * vals, minlength=n_cells).reshape(grid_shape)
    counts = np.bincount(flat, minlength=n_cells).astype(np.float64).reshape(grid_shape)
    sums3 = uniform_filter(sums, size=3, mode="constant") * 27.0
    sums23 = uniform_filter(sums2, size=3, mode="constant") * 27.0
    counts3 = uniform_filter(counts, size=3, mode="constant") * 27.0

    qijk = np.clip(((queries - lo) / float(cell)).astype(np.int64), 0, dims - 1)
    qflat = (qijk[:, 0] * dims[1] + qijk[:, 1]) * dims[2] + qijk[:, 2]
    count = counts3.reshape(-1)[qflat]
    mean = np.where(count > 0.5, sums3.reshape(-1)[qflat] / np.maximum(count, 1e-9), 0.0)
    m2 = np.where(count > 0.5, sums23.reshape(-1)[qflat] / np.maximum(count, 1e-9), 0.0)
    std = np.sqrt(np.clip(m2 - mean * mean, 0.0, None))
    return count, mean, std


def detect_mirror_rescue_discs(
    *,
    positions_texture: Any,
    colors_rgb: Any,
    observed_mask: Any,
    observed_weight: Any,
    axis: int = 1,
    feature_radius_ratio: float = 0.02,
    strong_weight: float = 0.35,
    strong_feature_std: float = 0.05,
    twin_coverage_min: float = 0.25,
    twin_weight_ratio: float = 0.5,
    core_contrast: float = 0.12,
    core_min_fraction: float = 3e-4,
    twin_core_ratio: float = 0.5,
    max_discs: int = 4,
    max_radius_ratio: float = 0.0575,
) -> List[Dict[str, Any]]:
    """Find feature discs whose mirror twin needs a `mirror_rescue_disc`.

    General mechanism (no feature-class knowledge): on a near-symmetric
    subject, a STRONG local feature — confidently witnessed, locally
    contrastful, with a coherent dark core — whose mirror twin region is
    observed but WEAKLY witnessed AND feature-empty marks a twin-side
    defect: every view covering the twin sees it at grazing incidence or
    through a misregistered duplicate reference, so the twin carries a
    smear where real content belongs, and no per-texel gate downstream can
    repair it (all covering views agree on the wrong content). Measured on
    the face lane's profile eyes: healthy disc ball-weight 0.55 vs twin
    0.16 with twin blob response 0.22x the healthy core's.

    Per-texel quantities (voxel ball of `feature_radius_ratio` * mesh
    diagonal over DIRECT observed texels, on surface-smoothed luminance):

      W  ball-mean witness weight        F   ball luminance std
      Wt twin ball-mean witness weight   Ct  twin/own witness density
      DoG = smoothed luminance - ball mean (compact blob response)

    A texel is a STRONG-side candidate when observed with W >=
    `strong_weight` and F >= `strong_feature_std`, off the symmetry plane,
    and its twin ball satisfies Ct >= `twin_coverage_min` (the twin is
    genuinely observed — unobserved twins belong to mirror COMPLETION, not
    rescue) and Wt <= `twin_weight_ratio` * W (the twin witnesses are
    categorically weaker; content well-witnessed on both sides is NEVER
    touched, which is what protects legitimately asymmetric content).
    Candidate components must then show a coherent DARK core (DoG <=
    -`core_contrast` over >= `core_min_fraction` of the direct texel
    count): feature-dark blobs (eyes, nostrils, markings) are the class
    whose loss reads as damage, while transplanting bright speculars was
    measured to fragment neighboring features. Finally the twin's own blob
    response at the mirrored core must be <= `twin_core_ratio` of the
    core's (the twin is feature-EMPTY; a twin with its own coherent
    structure is legitimate asymmetric content and is left alone).

    Returns disc descriptors sorted by core size: `center`/`radius` are the
    TWIN-side transplant disc (already mirrored) for `mirror_rescue_disc`;
    `source_center` is the healthy side. Fires zero discs on subjects
    without the witness asymmetry (measured: single-photo bakes cover the
    twin side by mirror completion, so Ct ~ 0 and nothing triggers).
    """
    import numpy as np
    from scipy.ndimage import gaussian_filter
    from scipy.ndimage import label as cc_label

    positions = np.asarray(positions_texture, dtype=np.float32)
    surface = positions[:, :, 3] > 0.0
    pts = positions[:, :, :3]
    observed = np.asarray(observed_mask, dtype=bool) & surface
    weight = np.asarray(observed_weight, dtype=np.float32)
    direct = observed & (weight > 0.0)
    direct_count = int(direct.sum())
    if direct_count < 1024:
        return []

    colors = np.asarray(colors_rgb, dtype=np.float32)
    luminance = colors[:, :, :3].mean(axis=2).astype(np.float32)

    scale = float(np.linalg.norm(pts[direct].max(axis=0) - pts[direct].min(axis=0)))
    if scale <= 0.0:
        return []
    radius_world = float(feature_radius_ratio) * scale

    # Surface-smoothed luminance: structure survives smoothing, projection
    # speckle averages out, and the feature scale becomes resolution-
    # invariant (sigma in texels grows with the atlas).
    sigma = max(1.0, positions.shape[0] / 512.0)
    direct_f = direct.astype(np.float32)
    denominator = gaussian_filter(direct_f, sigma)
    luminance_smooth = np.where(
        denominator > 0.2,
        gaussian_filter(luminance * direct_f, sigma) / np.maximum(denominator, 1e-6),
        0.0,
    ).astype(np.float32)

    own_points = pts[direct]
    twin_points = own_points.copy()
    twin_points[:, axis] *= -1.0
    lum_direct = luminance_smooth[direct]
    weight_direct = weight[direct]

    n_own, mean_own, std_own = _voxel_ball_stats(
        own_points, lum_direct, radius_world, own_points)
    n_twin, _, _ = _voxel_ball_stats(
        own_points, lum_direct, radius_world, twin_points)
    _, wmean_own, _ = _voxel_ball_stats(
        own_points, weight_direct, radius_world, own_points)
    _, wmean_twin, _ = _voxel_ball_stats(
        own_points, weight_direct, radius_world, twin_points)

    feature_std = np.zeros(surface.shape, dtype=np.float32)
    feature_std[direct] = std_own
    witness_own = np.zeros(surface.shape, dtype=np.float32)
    witness_own[direct] = wmean_own
    witness_twin = np.zeros(surface.shape, dtype=np.float32)
    witness_twin[direct] = wmean_twin
    twin_density = np.zeros(surface.shape, dtype=np.float32)
    twin_density[direct] = n_twin / np.maximum(n_own, 1.0)
    blob_response = np.zeros(surface.shape, dtype=np.float32)
    blob_response[direct] = lum_direct - mean_own

    off_plane = np.abs(pts[:, :, axis]) > radius_world
    strong = direct & (witness_own >= float(strong_weight)) & (
        feature_std >= float(strong_feature_std))
    trigger = strong & off_plane & (twin_density >= float(twin_coverage_min)) & (
        witness_twin <= float(twin_weight_ratio) * witness_own)
    if not trigger.any():
        return []

    # Twin blob responses must be sampled POINTWISE at the nearest direct
    # texel: a ball average dilutes a feature's edge response toward zero
    # (any blob wider than the ball is DoG-flat inside), which would blind
    # the feature-emptiness gate to real twin structure.
    try:
        from scipy.spatial import cKDTree
    except Exception:
        return []
    direct_tree = cKDTree(own_points)
    abs_response_direct = np.abs(lum_direct - mean_own)

    core_floor = max(12, int(float(core_min_fraction) * direct_count))
    component_labels, component_count = cc_label(trigger)
    discs: List[Dict[str, Any]] = []
    for component_id in range(1, component_count + 1):
        component = component_labels == component_id
        if int(component.sum()) < core_floor:
            continue
        core = component & (blob_response <= -float(core_contrast))
        core_points_all = pts[core]
        if len(core_points_all) < core_floor:
            continue
        # A component can mix several blobs (brow + eye + specular); the
        # transplant disc must center on ONE coherent dark core, so split
        # the core into its own connected clusters.
        core_labels, core_count = cc_label(core)
        for core_id in range(1, core_count + 1):
            core_cluster = core_labels == core_id
            cluster_size = int(core_cluster.sum())
            if cluster_size < core_floor:
                continue
            cluster_points = pts[core_cluster]
            core_center = np.median(cluster_points, axis=0)

            # Twin feature-emptiness: pointwise blob response of the nearest
            # direct texel at each mirrored core position.
            twin_center = core_center.copy()
            twin_center[axis] *= -1.0
            twin_query = cluster_points.copy()
            twin_query[:, axis] *= -1.0
            twin_distance, twin_index = direct_tree.query(twin_query, k=1, workers=-1)
            valid_twin = np.asarray(twin_distance) < 0.01 * scale
            if valid_twin.sum() < max(8, cluster_size // 4):
                continue
            own_response = float(np.abs(blob_response[core_cluster]).mean())
            twin_response = float(
                abs_response_direct[np.asarray(twin_index)[valid_twin]].mean())
            if twin_response > float(twin_core_ratio) * own_response:
                continue

            component_points = pts[component]
            component_radius = float(np.percentile(
                np.linalg.norm(component_points - core_center, axis=1), 95))
            disc_radius = float(np.clip(
                1.15 * component_radius,
                2.5 * radius_world,
                float(max_radius_ratio) * scale,
            ))
            # The whole disc must lie strictly on one side of the symmetry
            # plane: a feature straddling the plane has no clean "twin
            # side" — a half-transplant guarantees a mid-feature seam right
            # where the two sides must agree (measured on the face lane's
            # mouth: the clipped transplant painted a black dash and flake
            # fringe onto the front-view lips).
            if abs(float(core_center[axis])) <= disc_radius:
                continue

            # PLACEMENT anchor: the twin side's own weak witnesses carry
            # wrong CONTENT but still measured WHERE the dark feature sits
            # better than the pure geometric mirror does (mesh asymmetry
            # and per-side registration displace the mirror position;
            # measured on the face lane: the un-anchored transplant pulled
            # the source-pose identity registration 1.3% and its SSIM
            # 0.632 -> 0.601). Correction is restricted to the MIRROR AXIS
            # component of the twin's evidence-weighted feature-dark
            # centroid: across every estimator tried (weak-witness dark
            # centroid at two disc extents, the harness's own detector
            # localization) the axis component agreed within 0.002 of the
            # mesh scale while the in-plane components flipped signs — the
            # systematic mirror-placement error lives along the axis, the
            # rest is multi-cluster noise that measurably re-rolled a
            # bistable registration. Capped at 0.4x the feature radius,
            # below the anchor's own spread, so the correction is
            # effectively deterministic when evidence demands more and
            # can never move the transplant off the geometric position
            # (an uncapped shift measurably broke the repair).
            placement_shift = np.zeros(3, dtype=np.float32)
            twin_distance_map = np.linalg.norm(
                pts - twin_center[None, None, :], axis=2)
            twin_same_side = (
                pts[:, :, axis] * twin_center[axis]
            ) >= 0.0 if abs(float(twin_center[axis])) > 1e-6 else np.ones(
                surface.shape, dtype=bool)
            twin_disc = direct & twin_same_side & (twin_distance_map < disc_radius)
            twin_ring = direct & twin_same_side & (
                twin_distance_map >= disc_radius
            ) & (twin_distance_map < 1.45 * disc_radius)
            if twin_ring.sum() >= 64 and twin_disc.any():
                ring_tone = float(luminance[twin_ring].mean())
                darkness = np.clip(
                    ring_tone - 0.22 - luminance, 0.0, None).astype(np.float32)
                twin_dark = twin_disc & (darkness > 0.0)
                if int(twin_dark.sum()) >= max(12, core_floor // 2):
                    anchor_weight = weight[twin_dark] * darkness[twin_dark]
                    total = float(anchor_weight.sum())
                    if total > 1e-6:
                        anchor_axis = float(
                            (pts[twin_dark, axis] * anchor_weight).sum() / total)
                        axis_delta = anchor_axis - float(twin_center[axis])
                        cap = 0.4 * radius_world
                        placement_shift[axis] = float(
                            np.clip(axis_delta, -cap, cap))

            discs.append({
                "center": [float(v) for v in twin_center],
                "source_center": [float(v) for v in core_center],
                "radius": disc_radius,
                "placement_shift": [float(v) for v in placement_shift],
                "core_texels": cluster_size,
                "own_blob_response": round(own_response, 4),
                "twin_blob_response": round(twin_response, 4),
                "own_witness": round(float(witness_own[core_cluster].mean()), 4),
                "twin_witness": round(float(witness_twin[core_cluster].mean()), 4),
            })

    discs.sort(key=lambda d: -d["core_texels"])
    kept: List[Dict[str, Any]] = []
    for disc in discs:
        center = np.asarray(disc["center"], dtype=np.float32)
        overlapping = any(
            np.linalg.norm(center - np.asarray(k["center"], dtype=np.float32))
            < 0.75 * (disc["radius"] + k["radius"])
            for k in kept
        )
        if not overlapping:
            kept.append(disc)
        if len(kept) >= int(max_discs):
            break
    return kept


def mirror_rescue_disc(
    colors_rgb: Any,
    *,
    positions_texture: Any,
    center: Any,
    radius: float,
    axis: int = 1,
    ring_width_ratio: float = 0.45,
    feather_texels: float = 3.0,
    source_mask: Optional[Any] = None,
    source_shift: Optional[Any] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """Replace a feature disc's texels with their mirror twins' content.

    The transplant is deliberately WHOLE-DISC (no per-texel evidence
    carve-outs): the disc around a weakly-witnessed feature also contains
    content other views paint at moderate facing, and every partial-keep
    variant measured (source-witnessed exclusion, confidence ramps,
    dark-aware keeps, lateral-normal cuts) left the kept fragments and the
    transplanted band coexisting as SEPARATE blobs at intermediate views —
    a doubled-feature class strictly worse than the small source-pose
    registration sensitivity the full transplant costs (measured on the
    face lane: exclusion variants added three eye_count=3 failures while
    the full transplant's front identity delta at a FIXED alignment is
    0.004 SSIM with an identical worst window).

    `source_shift` translates WHERE the transplanted content lands inside
    the disc: destination texel x copies from mirror(x - shift), i.e. the
    twin's feature appears at (its mirrored position + shift). Callers use
    it to anchor placement on the destination's own evidence
    (`detect_mirror_rescue_discs` computes a capped shift toward the twin
    side's feature-dark centroid); zero/None keeps the pure geometric
    mirror.

    Complements `mirror_fill_from_observed`, which only writes UNOBSERVED
    texels: on near-symmetric subjects a feature region can be OBSERVED yet
    badly witnessed (all views see it at grazing incidence, or a mirrored
    duplicate reference lands misregistered), leaving displaced or smeared
    content that blocks the unobserved-only fill. Measured on the face
    lane's right eye at az -90: the disc's own witnesses average blend
    weight 0.14 vs the mirror twin's 0.50, and the painted iris band sits
    ~0.04 mesh units below its mirror-correct position — a defect no
    texture gate downstream can repair because every covering view agrees
    on the wrong content.

    The rescue copies, for every surface texel inside the world-space disc
    (`center`, `radius`), the color of the nearest surface texel to its
    mirror across `axis` (restricted to `source_mask` when given, e.g. a
    confident-witness mask). Two safeguards keep the transplant local and
    tone-neutral:

    - TONE MATCHING: the copied content is offset per channel so the twin
      ring's mean matches the destination ring's mean (ring = annulus
      `radius`..`radius*(1+ring_width_ratio)`). Composition stages
      (leveling, delight) legitimately shade the two sides differently;
      without the offset the transplant carries the twin side's tone and
      reads as a pasted patch.
    - FEATHERING: the disc edge blends over `feather_texels` so no hard
      seam is introduced.

    Callers decide WHERE to rescue (e.g. from a QA detector's localization
    or a witness-quality map); this function performs only the geometry-
    driven transplant. Returns `(rescued_rgb, stats)`; the input array is
    not modified.
    """
    import numpy as np

    colors = np.asarray(colors_rgb, dtype=np.float32).copy()
    positions = np.asarray(positions_texture, dtype=np.float32)
    surface = positions[:, :, 3] > 0.0
    pts = positions[:, :, :3]
    center = np.asarray(center, dtype=np.float32).reshape(3)
    stats: Dict[str, Any] = {"rescued_texels": 0, "tone_offset": None}

    distance = np.linalg.norm(pts - center[None, None, :], axis=2)
    # The disc must stay on the center's own side of the symmetry plane:
    # crossing it would overwrite the healthy twin.
    same_side = (pts[:, :, axis] * center[axis]) >= 0.0 if abs(center[axis]) > 1e-6 \
        else np.ones(surface.shape, dtype=bool)
    disc = surface & same_side & (distance < float(radius))
    ring = surface & same_side & (distance >= float(radius)) & (
        distance < float(radius) * (1.0 + float(ring_width_ratio))
    )
    if not disc.any():
        return colors, stats

    sources = surface & ~disc
    if source_mask is not None:
        mask = np.asarray(source_mask, dtype=bool)
        if mask.shape == surface.shape:
            sources = sources & mask
    src_pts = pts[sources]
    if len(src_pts) < 64:
        return colors, stats
    src_cols = colors[sources]

    try:
        from scipy.spatial import cKDTree
    except Exception:
        return colors, stats
    tree = cKDTree(src_pts)

    shift = np.zeros(3, dtype=np.float32)
    if source_shift is not None:
        shift = np.asarray(source_shift, dtype=np.float32).reshape(3)

    def twin_colors(mask: Any) -> Any:
        query = pts[mask] - shift[None, :]
        query[:, axis] *= -1.0
        _, indices = tree.query(query, k=1, workers=-1)
        return src_cols[indices]

    copied = twin_colors(disc)
    tone_offset = np.zeros(colors.shape[-1], dtype=np.float32)
    # The destination ring mean must be computed over texels that CARRY
    # CONTENT: inside a bake the annulus can include not-yet-filled texels
    # whose zeros bias the tone offset dark (measured: -0.066 vs -0.045
    # per channel on the same disc, enough to push transplanted skin
    # flecks across the dark-debris gate). With a source_mask the
    # confident witnesses are that content set; without one, all ring
    # texels count (the caller owns their validity).
    ring_content = ring
    if source_mask is not None:
        mask = np.asarray(source_mask, dtype=bool)
        if mask.shape == surface.shape and (ring & mask).sum() >= 64:
            ring_content = ring & mask
    if ring_content.any():
        tone_offset = colors[ring_content].mean(axis=0) - twin_colors(
            ring_content).mean(axis=0)
        copied = copied + tone_offset[None, :]

    rescued = colors.copy()
    rescued[disc] = np.clip(copied, 0.0, colors.max() if colors.max() > 1.5 else 1.0)

    # feather the disc boundary in atlas space
    weight = disc.astype(np.float32)
    try:
        from scipy.ndimage import gaussian_filter

        weight = gaussian_filter(weight, float(feather_texels))
        weight = np.clip(weight, 0.0, 1.0)
        weight[~surface] = 0.0
    except Exception:
        pass
    out = colors * (1.0 - weight[..., None]) + rescued * weight[..., None]
    stats["rescued_texels"] = int(disc.sum())
    stats["tone_offset"] = [round(float(v), 4) for v in tone_offset]
    return out, stats


def commit_trace_deposits(
    colors_rgba: Any,
    *,
    positions_texture: Any,
    observed_mask: Any,
    projections: Sequence[Mapping[str, Any]],
    film_commit_mask: Optional[Any] = None,
    protected_mask: Optional[Any] = None,
    deviation_min: float = 0.045,
    trace_w50: float = 0.30,
    trace_w90: float = 0.40,
    ring_cover_frac: float = 0.20,
    ring_single_cover: float = 0.60,
    ring_bright_min: float = 0.96,
    ring_min_texels: int = 8,
    bright_context: float = 0.55,
    min_area_frac: float = 4e-5,
    max_area_frac: float = 6e-3,
    max_blob_eval: int = 400,
    film_overlap_max: float = 0.30,
    context_ball_frac: float = 0.02,
    feature_weight: float = 0.35,
    feature_core_contrast: float = 0.12,
    feature_core_min: int = 12,
    feature_halo_ratio: float = 1.4,
    rim_feather_texels: int = 3,
    stats_out: Optional[Dict[str, Any]] = None,
) -> Any:
    """Vacate-and-retone trace-weight deposits contradicted by multi-witness
    skin consensus (the FACE-03/04/05 residual chip/dash class).

    THE DEFECT: small chips and dashes — displaced view content (lash/lip
    fragments, shading flakes, strap slivers) — deposited on smooth bright
    material at TRACE witness weight. The winning claim is real photo
    content that landed on the wrong surface: every confident witness of
    the neighborhood reads the local material as uniform bright skin,
    while the deposit's own witness weight sits categorically below the
    confident-feature population (measured on the face proof at 1024:
    chip blobs w50 0.02-0.29 / w90 <= 0.29 vs legit features w50 0.44-0.93).

    COMMIT SEMANTICS (film-band lessons, blob-level — per-texel thresholds
    measurably cannot separate chips from feature fringes, cycle-2 B/D
    negative results):
    - candidate texels: direct-witnessed, winner weight <= `trace_w50`,
      color deviating >= `deviation_min` from the 3D voxel-ball context
      mean, in a BRIGHT ball context (dark-material regions belong to the
      film-band machinery). Components above `max_blob_eval` texels are
      chip FIELDS chained by marginal halo texels and are split into
      strong-deviation cores, each judged against its own local ring.
    - a blob commits only when EVERY view imaging >= `ring_cover_frac` of
      its plain 3D ring (non-deviant direct surround; a world-space ball —
      atlas dilation crosses UV charts, measured picking up hair texels
      that veto valid commits) reads the ring >= `ring_bright_min` bright,
      at least one such witness exists, and a single witness must cover
      >= `ring_single_cover` of the ring (single-witness consensus is
      otherwise vacuous).
    - the blob's own witness must be trace (w50/w90 gates): content any
      view confidently witnesses is NEVER demoted (cycle-2 D's gate).
    - BRIGHT deposits (blob brighter than its ball context) additionally
      require median distance >= `feature_halo_ratio` x the context ball
      radius from any confident-contrast feature core (clusters of
      per-texel confidently-witnessed strong-contrast texels: lash lines,
      lip borders, sclera). A bright trace deposit near such a core is
      ambiguous with the feature's own bright fringe; committing those
      was measured to wash the eye corner and drop eye_count at az0/±90.
      Ball-mean witness (the mirror-rescue construction) cannot serve as
      the core signal here: the ball mean around a trace chip is lifted
      by its confident surround, so every chip registers as its own
      "feature" (measured: chin dash ball-weight 0.42 vs own w50 0.047).
      DARK deposits carry no such ambiguity — a dark blob whose surround
      every witness reads as uniform bright skin is exactly the chip
      class, and the ring consensus already refuses dark deposits whose
      ring contains real feature darks (measured votes 0.50-0.81 at lash
      dashes vs the 0.96 bar).
    - committed texels are RETONED from their validated bright ring
      anchors (inverse-square 3D interpolation). Verbatim refills (mirror
      twin / nearest confident) read as new pale patches, and membrane
      diffusion drags adjacent feature darkness across the hole
      (measured: dark_debris 0.0024 -> 0.0044 at az-22.5 with membrane
      refill); the ring anchors are exactly the texels whose consensus
      justified the commit, so the retone cannot inherit foreign content.
    - RIM FEATHER (cycle 6, FACE-22): the deposit's antialiased border
      mixtures sit BELOW `deviation_min` by construction (mixture
      deviation = coverage x deposit deviation), so the blob commits and
      its 1-3 texel rim keeps the old darker tone — rendered, the rim
      prints as a closed line-art outline around the retoned interior
      (the FACE-22 az-22.5 chest contour; stage difference maps in the
      cycle-6 provenance run). After each commit, texels within
      `rim_feather_texels` of the committed set that carry the SAME
      evidence class (direct, trace-weight, bright ball context, outside
      the film commit) blend toward the same ring-anchor interpolation,
      alpha decaying with distance and ONE-SIDED: only texels darker
      than the target move, so feature edges and legitimately dark
      surround can only keep or gain luminance headroom — never darken.

    PLACEMENT: the commit runs LATE (after mirror completion, rescue and
    film retone, immediately before detail synthesis) as a strictly local
    recolor. Committing at the outlier stage was measured to cascade
    through every global stage (Poisson anchors, rescue-disc localization,
    fill calibration): the whole-face render diff at the gate pose read
    mean 4.1/255 with 14% of pixels above 8/255, flipping knife-edge
    detectors unrelated to any chip. `protected_mask` (rescue-disc
    footprints) keeps transplanted content out of both detection and
    retone. Requires >= 2 projections: with a single witness the ring
    consensus collapses to the winner's own photo and is vacuous by
    construction (same reasoning as `commit_film_band`).

    Returns the recolored copy; `stats_out` receives per-blob records.
    """
    import numpy as np
    from scipy.ndimage import binary_dilation
    from scipy.ndimage import label as cc_label

    stats: Dict[str, Any] = {"applied": False, "committed_texels": 0, "blobs": []}
    if stats_out is not None:
        stats_out.update(stats)
    rgba = np.asarray(colors_rgba, dtype=np.float32)
    if len(projections) < 2:
        return rgba

    positions = np.asarray(positions_texture, dtype=np.float32)
    surface = positions[:, :, 3] > 0.0
    pts = positions[:, :, :3]
    colors = rgba[:, :, :3]
    lum = colors.mean(axis=2)

    weight_stack = np.stack(
        [np.asarray(p["weight"], dtype=np.float32) for p in projections], axis=0)
    winner_weight = weight_stack.max(axis=0)
    direct = np.asarray(observed_mask, dtype=bool) & surface & (winner_weight > 1e-6)
    if protected_mask is not None:
        protected = np.asarray(protected_mask, dtype=bool)
        if protected.shape == surface.shape:
            direct = direct & ~protected
    n_direct = int(direct.sum())
    if n_direct < 1024:
        return rgba

    direct_points = pts[direct]
    scale = float(np.linalg.norm(direct_points.max(axis=0) - direct_points.min(axis=0)))
    if scale <= 0.0:
        return rgba
    r_ctx = float(context_ball_frac) * scale

    ball_rgb = np.zeros((*surface.shape, 3), np.float32)
    for channel in range(3):
        _, channel_mean, _ = _voxel_ball_stats(
            direct_points, colors[:, :, channel][direct], r_ctx, direct_points)
        ball_rgb[direct, channel] = channel_mean
    _, lum_mean, _ = _voxel_ball_stats(
        direct_points, lum[direct], r_ctx, direct_points)
    ball_mean = np.zeros(surface.shape, np.float32)
    ball_mean[direct] = lum_mean

    deviation = np.abs(colors - ball_rgb).mean(axis=2)
    deviation[~direct] = 0.0

    direct_lum = lum[direct]
    bright_median = float(np.median(direct_lum[direct_lum >= np.median(direct_lum)]))
    if bright_median <= 0.0:
        return rgba

    candidates = (
        direct
        & (winner_weight <= float(trace_w50))
        & (deviation >= float(deviation_min))
        & (ball_mean > float(bright_context) * bright_median)
    )
    if not candidates.any():
        return rgba

    film = None
    if film_commit_mask is not None:
        film = np.asarray(film_commit_mask, dtype=bool)
        if film.shape != surface.shape:
            film = None

    # confident-contrast feature cores (protection for BRIGHT deposits)
    blob_response = np.zeros(surface.shape, np.float32)
    blob_response[direct] = lum[direct] - lum_mean
    feature_core = (
        direct
        & (winner_weight >= float(feature_weight))
        & (np.abs(blob_response) >= float(feature_core_contrast))
    )
    core_labels, core_count = cc_label(feature_core, structure=np.ones((3, 3), bool))
    feature_tree = None
    if core_count:
        sizes = np.bincount(core_labels.ravel())
        sizes[0] = 0
        strong_core = (sizes >= int(feature_core_min))[core_labels]
        if strong_core.any():
            try:
                from scipy.spatial import cKDTree

                feature_tree = cKDTree(pts[strong_core])
            except Exception:
                feature_tree = None
    halo_radius = float(feature_halo_ratio) * r_ctx

    valid_stack = [np.asarray(p["rgba"], dtype=np.float32)[:, :, 3] > 0.0
                   for p in projections]
    rgb_stack = [np.asarray(p["rgba"], dtype=np.float32)[:, :, :3]
                 for p in projections]

    min_area = max(4, int(float(min_area_frac) * n_direct))
    max_area = int(float(max_area_frac) * n_direct)

    # ISOLATION over the WHOLE surface (fill included): a dark deposit must
    # be an isolated island, not the frontier sliver of a connected dark
    # mass. At hair/skin frontiers the surround's dark side is often FILL
    # (surrendered/contested texels carry no direct witness), so a ring
    # consensus over direct texels sees only the bright side and approves
    # hair-mass fragments — measured: committed slivers at the temple
    # hairline retoned to pale streaks inside the hair at +90 el10 and
    # dropped the profile eye_count. Connectivity is evaluated on the
    # current colors (post fill), where the hair mass is contiguous.
    dark_mask = surface & (lum < float(bright_context) * bright_median)
    dark_labels, _ = cc_label(dark_mask, structure=np.ones((3, 3), bool))
    dark_sizes = np.bincount(dark_labels.ravel())
    if len(dark_sizes):
        dark_sizes[0] = 0

    blobs, blob_count = cc_label(candidates, structure=np.ones((3, 3), bool))
    # Eval units are stored as (window, local_mask) pairs and every
    # per-unit operation below runs inside the unit's bounding window or
    # on a precomputed flat domain; with hundreds of candidate units the
    # previous full-atlas masks/gathers per unit dominated this stage
    # (measured 17.7 s on the face proof; outputs verified bitwise-
    # identical against the full-atlas formulation).
    from scipy.ndimage import find_objects

    atlas_shape = surface.shape
    eval_units: List[Tuple[Tuple[slice, slice], Any]] = []
    for blob_id, bbox in enumerate(find_objects(blobs), start=1):
        if bbox is None:
            continue
        window = (
            slice(max(bbox[0].start - 2, 0), min(bbox[0].stop + 2, atlas_shape[0])),
            slice(max(bbox[1].start - 2, 0), min(bbox[1].stop + 2, atlas_shape[1])),
        )
        blob_local = blobs[window] == blob_id
        area = int(blob_local.sum())
        if area < min_area:
            continue
        if area <= int(max_blob_eval):
            eval_units.append((window, blob_local))
            continue
        cores = blob_local & (deviation[window] >= 1.5 * float(deviation_min))
        piece_labels, piece_count = cc_label(cores, structure=np.ones((3, 3), bool))
        for piece_id in range(1, piece_count + 1):
            piece = piece_labels == piece_id
            if int(piece.sum()) < min_area:
                continue
            eval_units.append(
                (window, binary_dilation(piece, iterations=2) & blob_local))

    # residue field for the whole-neighborhood rule below: trace-weight
    # texels reading BELOW SKIN (mid-gray dashes and chip shadow edges,
    # lum 0.45-0.60 on a 0.73-median skin, sit above the dark-material
    # split but count as dark micro-islands once their surround is clean)
    residue_field = (
        direct
        & (winner_weight <= float(trace_w50))
        & (lum <= 0.82 * bright_median)
    )
    if protected_mask is not None:
        protected = np.asarray(protected_mask, dtype=bool)
        if protected.shape == surface.shape:
            residue_field &= ~protected
    if film is not None:
        residue_field &= ~film

    def _sweepable(island):
        """A residue island the neighborhood consensus may retone: small,
        isolated (never a frontier sliver of a connected dark mass) and
        outside every confident-feature halo."""
        island_area = int(island.sum())
        if island_area > 12 * min_area:
            return False
        island_dark = dark_labels[island & dark_mask]
        island_dark = island_dark[island_dark > 0]
        if len(island_dark):
            if int(dark_sizes[np.unique(island_dark)].max()) > 3 * island_area + 8:
                return False
        if feature_tree is not None:
            core_distance, _ = feature_tree.query(pts[island], k=1, workers=-1)
            if float(np.median(np.asarray(core_distance))) < halo_radius:
                return False
        return True

    out = rgba.copy()
    committed_total = 0
    committed_mask = np.zeros(surface.shape, dtype=bool)

    # Flat ring domain: a ring texel must be direct with sub-threshold
    # deviation, so distances and per-view statistics are evaluated only
    # over that fixed population (row-major extraction preserves the
    # element order the full-atlas boolean indexing produced, keeping
    # every reduction bit-identical).
    ring_domain = direct & (deviation < float(deviation_min))
    dom_rows, dom_cols = np.nonzero(ring_domain)
    dom_pts = pts[dom_rows, dom_cols]
    dom_valid = [v[dom_rows, dom_cols] for v in valid_stack]
    dom_rgb = [v[dom_rows, dom_cols] for v in rgb_stack]
    dom_lum = lum[dom_rows, dom_cols]
    dom_index_map = np.full(atlas_shape, -1, dtype=np.int32)
    dom_index_map[dom_rows, dom_cols] = np.arange(len(dom_rows), dtype=np.int32)
    blob_at_dom = np.zeros(len(dom_rows), dtype=bool)

    residue_rows, residue_cols = np.nonzero(residue_field)
    residue_pts = pts[residue_rows, residue_cols]
    residue_index_map = np.full(atlas_shape, -1, dtype=np.int32)
    residue_index_map[residue_rows, residue_cols] = np.arange(
        len(residue_rows), dtype=np.int32)
    blob_at_residue = np.zeros(len(residue_rows), dtype=bool)

    def materialize(window, local_mask):
        full = np.zeros(atlas_shape, dtype=bool)
        full[window][local_mask] = True
        return full

    for window, blob_local in eval_units:
        area = int(blob_local.sum())
        if area < min_area or area > max_area:
            continue
        if film is not None and float(
                (blob_local & film[window]).sum()) / area > float(film_overlap_max):
            continue
        pts_window = pts[window]
        blob_pts = pts_window[blob_local]
        blob_center = blob_pts.mean(axis=0)
        blob_weights = winner_weight[window][blob_local]
        if float(np.percentile(blob_weights, 50)) > float(trace_w50):
            continue
        if float(np.percentile(blob_weights, 90)) > float(trace_w90):
            continue
        bright_deposit = float(np.median(blob_response[window][blob_local])) > 0.0
        if bright_deposit and feature_tree is not None:
            core_distance, _ = feature_tree.query(blob_pts, k=1, workers=-1)
            if float(np.median(np.asarray(core_distance))) < halo_radius:
                continue
        if not bright_deposit:
            # isolation: reject frontier slivers of a connected dark mass
            blob_dark_labels = dark_labels[window][blob_local & dark_mask[window]]
            blob_dark_labels = blob_dark_labels[blob_dark_labels > 0]
            if len(blob_dark_labels):
                component_size = int(dark_sizes[np.unique(blob_dark_labels)].max())
                if component_size > 3 * area + 8:
                    continue
        blob_radius = float(np.percentile(
            np.linalg.norm(blob_pts - blob_center[None, :], axis=1), 90))
        ring_radius = max(2.5 * blob_radius, 1.5 * r_ctx)
        blob_dom_ids = dom_index_map[window][blob_local]
        blob_dom_ids = blob_dom_ids[blob_dom_ids >= 0]
        blob_at_dom[blob_dom_ids] = True
        dom_distance = np.linalg.norm(dom_pts - blob_center[None, :], axis=1)
        ring_flat = (dom_distance <= ring_radius) & ~blob_at_dom
        blob_at_dom[blob_dom_ids] = False
        ring_count = int(ring_flat.sum())
        if ring_count < int(ring_min_texels):
            continue
        votes: List[float] = []
        covers: List[float] = []
        for view_index in range(len(projections)):
            selected = ring_flat & dom_valid[view_index]
            n_selected = int(selected.sum())
            cover = n_selected / ring_count
            if n_selected < int(ring_min_texels) or cover < float(ring_cover_frac):
                continue
            view_lum = dom_rgb[view_index][selected].mean(axis=1)
            votes.append(float(
                (view_lum > float(bright_context) * bright_median).mean()))
            covers.append(cover)
        if not votes:
            continue
        if len(votes) == 1 and max(covers) < float(ring_single_cover):
            continue
        if min(votes) < float(ring_bright_min):
            continue

        # WHOLE-NEIGHBORHOOD RULE: committing a chip and leaving its
        # sub-threshold residue neighbors (mid-gray dashes, shadow edges
        # just under the deviation bar) UNMASKS them — on the cleaned
        # surround they read as new isolated dark islands (measured:
        # dark_debris 0.0022 -> 0.0037 at az0 for a commit pass without
        # this rule; the flagged islands sat exactly beside the commits).
        # The blob therefore commits only if EVERY residue island inside
        # its ring is itself sweepable under the same consensus — clean
        # the neighborhood fully, or leave it exactly as witnessed.
        blob_res_ids = residue_index_map[window][blob_local]
        blob_res_ids = blob_res_ids[blob_res_ids >= 0]
        blob_at_residue[blob_res_ids] = True
        residue_distance = np.linalg.norm(
            residue_pts - blob_center[None, :], axis=1)
        residue_near_flat = (residue_distance <= ring_radius) & ~blob_at_residue
        blob_at_residue[blob_res_ids] = False
        sweep_islands: List[Any] = []
        if residue_near_flat.any():
            residue_near = np.zeros(atlas_shape, dtype=bool)
            residue_near[residue_rows[residue_near_flat],
                         residue_cols[residue_near_flat]] = True
            residue_labels, residue_count = cc_label(
                residue_near, structure=np.ones((3, 3), bool))
            refused = False
            for residue_id in range(1, residue_count + 1):
                island = residue_labels == residue_id
                if int(island.sum()) < 2:
                    continue
                if not _sweepable(island):
                    refused = True
                    break
                sweep_islands.append(island)
            if refused:
                continue

        # Ring-anchored retone (inverse-square 3D interpolation). Anchors
        # are the BRIGHT-class ring texels only: the ring's deviation
        # filter keys on each texel's OWN ball context, so near a feature
        # boundary it admits dark texels (under-lid shadow beside an
        # under-eye chip) whose inclusion was measured to wash committed
        # blobs mid-dark (dark_debris 0.0024 -> 0.0048 at az-22.5). The
        # commit's consensus evidence is that the surround is BRIGHT
        # material — the retone must draw from exactly that evidence.
        anchor_flat = ring_flat & (dom_lum > float(bright_context) * bright_median)
        if int(anchor_flat.sum()) < int(ring_min_texels):
            continue
        newly_committed = materialize(window, blob_local)
        local_rows, local_cols = np.nonzero(blob_local)
        blob_rows = local_rows + window[0].start
        blob_cols = local_cols + window[1].start
        ring_rows, ring_cols = dom_rows[anchor_flat], dom_cols[anchor_flat]
        blob_points = pts[blob_rows, blob_cols]
        ring_points = dom_pts[anchor_flat]
        ring_colors = out[ring_rows, ring_cols, :3]
        squared = ((blob_points[:, None, :] - ring_points[None, :, :]) ** 2).sum(axis=2)
        inverse_weights = 1.0 / (squared + 1e-8)
        blended = (inverse_weights @ ring_colors) / np.maximum(
            inverse_weights.sum(axis=1, keepdims=True), 1e-9)
        out[blob_rows, blob_cols, :3] = blended
        committed_total += area
        committed_mask[window][blob_local] = True
        # retone the neighborhood's sweepable residue from the same
        # validated anchors (their own contrast never re-thresholded —
        # the consensus is the blob's)
        for island in sweep_islands:
            island = island & ~committed_mask
            if not island.any():
                continue
            island_rows, island_cols = np.nonzero(island)
            squared = ((pts[island_rows, island_cols][:, None, :]
                        - ring_points[None, :, :]) ** 2).sum(axis=2)
            inverse_weights = 1.0 / (squared + 1e-8)
            out[island_rows, island_cols, :3] = (
                (inverse_weights @ ring_colors)
                / np.maximum(inverse_weights.sum(axis=1, keepdims=True), 1e-9))
            committed_mask |= island
            newly_committed |= island
            committed_total += int(island.sum())
            stats["swept_islands"] = stats.get("swept_islands", 0) + 1
        # rim feather (see docstring): pull the commit's border mixtures
        # toward the ring-anchor tone, distance-decayed, one-sided. The
        # target interpolates from anchors OUTSIDE the feather band: a rim
        # mixture bright enough to be a ring anchor itself would otherwise
        # self-dominate the inverse-square target (distance ~0) and pin
        # its own darkness in place.
        rim_exclusion = binary_dilation(
            newly_committed, iterations=int(rim_feather_texels))
        anchor_mask = np.zeros(atlas_shape, dtype=bool)
        anchor_mask[ring_rows, ring_cols] = True
        rim_anchor = anchor_mask & ~rim_exclusion
        if int(rim_anchor.sum()) >= int(ring_min_texels):
            rim_anchor_rows, rim_anchor_cols = np.nonzero(rim_anchor)
            rim_anchor_points = pts[rim_anchor_rows, rim_anchor_cols]
            rim_anchor_colors = out[rim_anchor_rows, rim_anchor_cols, :3]
            rim_band = newly_committed.copy()
            for rim_step in range(int(rim_feather_texels)):
                rim_band = binary_dilation(rim_band)
                rim = (
                    rim_band & ~committed_mask & ~newly_committed & direct
                    & (winner_weight <= float(trace_w50))
                    & (ball_mean > float(bright_context) * bright_median)
                )
                if film is not None:
                    rim &= ~film
                if not rim.any():
                    continue
                rim_rows, rim_cols = np.nonzero(rim)
                squared = ((pts[rim_rows, rim_cols][:, None, :]
                            - rim_anchor_points[None, :, :]) ** 2).sum(axis=2)
                inverse_weights = 1.0 / (squared + 1e-8)
                rim_target = (inverse_weights @ rim_anchor_colors) / np.maximum(
                    inverse_weights.sum(axis=1, keepdims=True), 1e-9)
                rim_current = out[rim_rows, rim_cols, :3]
                below = (rim_current.mean(axis=1)
                         < rim_target.mean(axis=1) - 1.0 / 255.0)
                alpha = ((1.0 - rim_step / float(rim_feather_texels))
                         * below)[:, None]
                out[rim_rows, rim_cols, :3] = (
                    rim_current * (1.0 - alpha) + rim_target * alpha)
                stats["rim_feathered"] = (
                    stats.get("rim_feathered", 0) + int(below.sum()))
                newly_committed |= rim
        stats["blobs"].append({
            "area": area,
            "center": [round(float(v), 3) for v in blob_center],
            "w50": round(float(np.percentile(blob_weights, 50)), 3),
            "w90": round(float(np.percentile(blob_weights, 90)), 3),
            "bright_deposit": bool(bright_deposit),
            "votes": [round(v, 3) for v in votes],
        })

    stats["applied"] = committed_total > 0
    stats["committed_texels"] = committed_total
    if stats_out is not None:
        stats_out.update(stats)
    return out


def commit_pale_chips(
    colors_rgba: Any,
    *,
    positions_texture: Any,
    observed_mask: Any,
    projections: Sequence[Mapping[str, Any]],
    film_commit_mask: Optional[Any] = None,
    protected_mask: Optional[Any] = None,
    deviation_min: float = 0.045,
    trace_w50: float = 0.30,
    ring_cover_frac: float = 0.20,
    ring_single_cover: float = 0.60,
    ring_dark_min: float = 0.96,
    ring_min_texels: int = 8,
    dark_context: float = 0.55,
    min_area: int = 4,
    max_area_frac: float = 1.2e-3,
    context_ball_frac: float = 0.02,
    bright_connect_min: int = 600,
    stats_out: Optional[Dict[str, Any]] = None,
) -> Any:
    """Vacate-and-retone PALE chip islands in DARK material context — the
    dual of `commit_trace_deposits` (the FACE-07 ear-band class).

    THE DEFECT: small pale blobs — skin/mixture content displaced into
    hair by profile-photo layered zones, plus completion texels that
    inherited those anchors — sitting in a DARK ball context at TRACE (or
    zero, for fill) witness weight, while every qualifying witness of the
    blob's plain 3D ring reads the surround as uniformly DARK material.
    Bright-context deposits belong to `commit_trace_deposits`; the fused
    mixture BANDS belong to the film-band machinery; this pass owns the
    isolated pale chip islands the ear bands and hairline carry.

    Commit gates (mirroring the bright-context commit's measured lessons):
    - candidate domain: winner weight <= `trace_w50` (confidently
      witnessed pale content — skin between strands the photo actually
      saw — is NEVER touched) INCLUDING fill texels (weight 0): pale
      fill islands are copies of the same displaced anchors; the ear-band
      chip population measured 35-60% fill;
    - pale deviation >= `deviation_min` above a DARK ball context
      (< `dark_context` x bright-half median);
    - plain-ring multi-witness DARK consensus: every view imaging
      >= `ring_cover_frac` of the ring must read it >= `ring_dark_min`
      dark; one witness must cover >= `ring_single_cover`; no witness =
      vacuous = refuse (single-view bakes structurally no-op);
    - isolation: chips 2-connected to a big bright component
      (>= `bright_connect_min` texels — the face/ear skin mass) are
      frontier slivers of real bright material, not islands — refused;
    - area cap `max_area_frac` (of direct texels): the chip class
      measures 12-60 texels at 1024; without the cap a 700-texel
      rear blob committed into a visibly flat gray wash (measured);
    - retone from validated DARK ring anchors (inverse-square 3D) —
      the texels whose consensus justified the commit.

    Requires >= 2 projections. Returns the recolored copy; `stats_out`
    receives counts and refusal tallies.
    """
    import numpy as np
    from scipy.ndimage import binary_dilation
    from scipy.ndimage import label as cc_label

    stats: Dict[str, Any] = {
        "applied": False, "committed_texels": 0, "blobs": 0, "refused": {}}
    rgba = np.asarray(colors_rgba, dtype=np.float32)
    if len(projections) < 2:
        if stats_out is not None:
            stats_out.update(stats)
        return rgba
    try:
        from scipy.spatial import cKDTree
    except Exception:
        if stats_out is not None:
            stats_out.update(stats)
        return rgba

    positions = np.asarray(positions_texture, dtype=np.float32)
    surface = positions[:, :, 3] > 0.0
    pts = positions[:, :, :3]
    colors = rgba[:, :, :3]
    lum = colors.mean(axis=2)

    weight_stack = np.stack(
        [np.asarray(p["weight"], dtype=np.float32) for p in projections])
    winner_weight = weight_stack.max(axis=0)
    direct = np.asarray(observed_mask, dtype=bool) & surface & (winner_weight > 1e-6)
    protected = None
    if protected_mask is not None:
        candidate_protected = np.asarray(protected_mask, dtype=bool)
        if candidate_protected.shape == surface.shape:
            protected = candidate_protected
            direct = direct & ~protected
    film = None
    if film_commit_mask is not None:
        film = np.asarray(film_commit_mask, dtype=bool)
        if film.shape != surface.shape:
            film = None
    direct_eval = direct & ~film if film is not None else direct
    if int(direct_eval.sum()) < 1024:
        if stats_out is not None:
            stats_out.update(stats)
        return rgba

    direct_points = pts[direct_eval]
    scale = float(np.linalg.norm(
        direct_points.max(axis=0) - direct_points.min(axis=0)))
    if scale <= 0.0:
        if stats_out is not None:
            stats_out.update(stats)
        return rgba
    r_ctx = float(context_ball_frac) * scale

    # candidate domain: trace-weight direct AND fill; film/rescue excluded
    eval_domain = surface & (winner_weight <= float(trace_w50))
    if protected is not None:
        eval_domain &= ~protected
    if film is not None:
        eval_domain &= ~film

    surface_points = pts[surface]
    _, lum_mean_all, _ = _voxel_ball_stats(
        surface_points, lum[surface], r_ctx, surface_points)
    ball_mean = np.zeros(surface.shape, np.float32)
    ball_mean[surface] = lum_mean_all

    direct_lum = lum[direct_eval]
    bright_median = float(np.median(direct_lum[direct_lum >= np.median(direct_lum)]))
    if bright_median <= 0.0:
        if stats_out is not None:
            stats_out.update(stats)
        return rgba
    dark_bar = float(dark_context) * bright_median

    candidates = (
        eval_domain
        & (lum - ball_mean >= float(deviation_min))
        & (ball_mean < dark_bar)
    )
    if not candidates.any():
        if stats_out is not None:
            stats_out.update(stats)
        return rgba

    bright_mask = surface & (lum >= dark_bar)
    bright_labels, _ = cc_label(bright_mask, structure=np.ones((3, 3), bool))
    bright_sizes = np.bincount(bright_labels.ravel())
    if len(bright_sizes):
        bright_sizes[0] = 0

    valid_stack = [np.asarray(p["rgba"], dtype=np.float32)[:, :, 3] > 0.0
                   for p in projections]
    lum_stack = [np.asarray(p["rgba"], dtype=np.float32)[:, :, :3].mean(axis=2)
                 for p in projections]

    n_direct = int(direct_eval.sum())
    max_area = max(int(min_area), int(float(max_area_frac) * n_direct))
    plain = direct_eval & ~candidates
    plain_points = pts[plain]
    plain_tree = cKDTree(plain_points)
    plain_is_dark = lum[plain] < dark_bar
    plain_valid = [v[plain] for v in valid_stack]
    plain_photo_dark = [pl[plain] < dark_bar for pl in lum_stack]

    blobs, blob_count = cc_label(candidates, structure=np.ones((3, 3), bool))
    out = rgba.copy()
    committed = np.zeros(surface.shape, dtype=bool)
    refused = {"area": 0, "w90": 0, "ring": 0, "isolation": 0, "cover": 0}
    n_committed_blobs = 0
    # Per-blob work happens inside each blob's bounding window (margin 2
    # covers the isolation dilation): candidate chips are tiny, so full-
    # atlas masks/dilations per blob dominated this stage (measured 44 s
    # on the face proof; output verified bitwise-identical). The plain
    # colors gather is loop-invariant (`colors` is never mutated) and is
    # hoisted for the same reason.
    from scipy.ndimage import find_objects

    blob_slices = find_objects(blobs)
    plain_colors = colors[plain]
    for blob_id in range(1, blob_count + 1):
        bbox = blob_slices[blob_id - 1]
        if bbox is None:
            continue
        window = (
            slice(max(bbox[0].start - 2, 0), min(bbox[0].stop + 2, surface.shape[0])),
            slice(max(bbox[1].start - 2, 0), min(bbox[1].stop + 2, surface.shape[1])),
        )
        blob = blobs[window] == blob_id
        area = int(blob.sum())
        if area < int(min_area) or area > max_area:
            refused["area"] += 1
            continue
        # isolation: 2-dilated blob touching a big bright component is a
        # frontier sliver of real bright material
        ring2 = binary_dilation(blob, iterations=2)
        touched = np.unique(bright_labels[window][ring2 & bright_mask[window]])
        touched = touched[touched > 0]
        if any(bright_sizes[label_id] >= int(bright_connect_min)
               for label_id in touched):
            refused["isolation"] += 1
            continue
        pts_window = pts[window]
        center = pts_window[blob].mean(axis=0)
        radius = max(
            float(np.linalg.norm(pts_window[blob] - center, axis=1).max()) * 1.5, r_ctx)
        ring_idx = np.asarray(
            plain_tree.query_ball_point(center, r=radius), dtype=np.int64)
        if len(ring_idx) < int(ring_min_texels):
            refused["ring"] += 1
            continue
        votes_ok = True
        any_witness = False
        single_cover_ok = False
        for view_index in range(len(projections)):
            view_sees = plain_valid[view_index][ring_idx]
            cover = float(view_sees.mean())
            if cover < float(ring_cover_frac):
                continue
            any_witness = True
            if cover >= float(ring_single_cover):
                single_cover_ok = True
            dark_votes = float(
                plain_photo_dark[view_index][ring_idx][view_sees].mean())
            if dark_votes < float(ring_dark_min):
                votes_ok = False
                break
        if not (votes_ok and any_witness and single_cover_ok):
            refused["cover" if any_witness else "ring"] += 1
            continue
        anchor_sel = ring_idx[plain_is_dark[ring_idx]]
        if len(anchor_sel) < int(ring_min_texels):
            refused["ring"] += 1
            continue
        anchor_points = plain_points[anchor_sel]
        anchor_rgb = plain_colors[anchor_sel]
        blob_points = pts_window[blob]
        squared = ((blob_points[:, None, :] - anchor_points[None, :, :]) ** 2).sum(axis=2)
        inverse_weights = 1.0 / np.maximum(squared, 1e-10)
        inverse_weights /= inverse_weights.sum(axis=1, keepdims=True)
        out_window = out[window]
        out_window[blob, :3] = inverse_weights @ anchor_rgb
        committed_window = committed[window]
        committed_window[blob] = True
        n_committed_blobs += 1

    stats.update(
        applied=bool(n_committed_blobs),
        committed_texels=int(committed.sum()),
        blobs=n_committed_blobs,
        refused=refused,
    )
    if stats_out is not None:
        stats_out.update(stats)
    return out


def tone_bottom_cap(
    colors_rgba: Any,
    *,
    positions_texture: Any,
    normals_texture: Optional[Any],
    observed_mask: Any,
    direct_observed_mask: Any,
    plane_alignment_min: float = 0.9,
    min_cap_frac: float = 0.005,
    max_direct_frac: float = 0.01,
    max_slab_ratio: float = 0.08,
    rim_dilate: int = 3,
    rim_neighbors: int = 48,
    target_sigma: float = 24.0,
    detail_keep: float = 0.6,
    reference_resolution: int = 1024,
    stats_out: Optional[Dict[str, Any]] = None,
) -> Any:
    """Geometry-aware toning of synthetic CUT FACES from their rim's
    observed content (the FACE-12 bust-disc class).

    Generated busts/statues are truncated by a planar cut; the cut face
    is synthetic geometry no photo ever witnessed, yet the global
    harmonic fill tones it with whatever the mesh graph happens to
    connect it to (measured on the face proof: a tan/taupe marble wash
    fed by rear-hair and neck anchors). The principled tone for a
    synthetic cut face is the CONTINUATION OF ITS OWN RIM — the observed
    material that physically borders the cut (chest skin at the front
    rim, hair curtain at the rear rim), interpolated across the plane.

    Detection is purely geometric (no asset-specific masks): a connected
    component of surface texels whose normals align with a common axis
    (|n . axis| >= `plane_alignment_min` for the dominant down axis),
    covering >= `min_cap_frac` of the surface, carrying essentially no
    direct witness (< `max_direct_frac`), and forming a thin slab
    (thickness/extent <= `max_slab_ratio`). Subjects without such a face
    (closed meshes, curved hulls) no-op structurally.

    Toning: inverse-distance interpolation (in the cap plane) of the
    rim's OBSERVED colors, smoothed at `target_sigma` (atlas texels at
    the reference resolution), with `detail_keep` of the cap's own
    log-residual micro-detail preserved so the face keeps texture.
    """
    import numpy as np
    from scipy.ndimage import binary_dilation, gaussian_filter
    from scipy.ndimage import label as cc_label

    stats: Dict[str, Any] = {"applied": False, "cap_texels": 0}
    rgba = np.asarray(colors_rgba, dtype=np.float32)
    try:
        from scipy.spatial import cKDTree
    except Exception:
        if stats_out is not None:
            stats_out.update(stats)
        return rgba

    positions = np.asarray(positions_texture, dtype=np.float32)
    surface = positions[:, :, 3] > 0.0
    if normals_texture is None or not surface.any():
        if stats_out is not None:
            stats_out.update(stats)
        return rgba
    normals = np.asarray(normals_texture, dtype=np.float32)[:, :, :3]
    norm = np.linalg.norm(normals, axis=2, keepdims=True)
    normals = np.divide(normals, np.maximum(norm, 1e-8))
    xyz = positions[:, :, :3]
    observed = np.asarray(observed_mask, dtype=bool) & surface
    direct = np.asarray(direct_observed_mask, dtype=bool) & surface

    # dominant downward axis: generated meshes are recentered/axis-aligned;
    # the cut face of a bust points -z. Detect against -z explicitly (the
    # generic "any planar axis" variant would also catch flat hull panels).
    axis = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    alignment = normals @ axis
    candidates = surface & (alignment >= float(plane_alignment_min))
    if not candidates.any():
        if stats_out is not None:
            stats_out.update(stats)
        return rgba
    labels, _ = cc_label(candidates)
    sizes = np.bincount(labels.ravel())
    if len(sizes):
        sizes[0] = 0
    n_surface = int(surface.sum())
    cap = np.zeros_like(candidates)
    for label_id in np.argsort(sizes)[::-1]:
        if sizes[label_id] < float(min_cap_frac) * n_surface:
            break
        component = labels == label_id
        if float(direct[component].mean()) >= float(max_direct_frac):
            continue
        component_points = xyz[component]
        thickness = float(np.percentile(component_points[:, 2], 98)
                          - np.percentile(component_points[:, 2], 2))
        extent = float(np.linalg.norm(
            component_points.max(axis=0) - component_points.min(axis=0)))
        if extent <= 0 or thickness / extent > float(max_slab_ratio):
            continue
        cap |= component
    if not cap.any():
        if stats_out is not None:
            stats_out.update(stats)
        return rgba

    rim = binary_dilation(cap, iterations=int(rim_dilate)) & ~cap & observed
    rim_points = xyz[rim]
    if len(rim_points) < 32:
        if stats_out is not None:
            stats_out.update(stats)
        return rgba
    rim_colors = rgba[rim][:, :3]

    resolution = surface.shape[0]
    s = resolution / float(reference_resolution)
    tree = cKDTree(rim_points[:, :2])
    cap_rows, cap_cols = np.nonzero(cap)
    query = xyz[cap][:, :2]
    k = int(min(rim_neighbors, len(rim_points)))
    distances, indices = tree.query(query, k=k, workers=-1)
    distances = np.atleast_2d(distances)
    indices = np.atleast_2d(indices)
    weights = 1.0 / np.maximum(distances, 1e-4) ** 1.5
    weights /= weights.sum(axis=1, keepdims=True)
    target_rgb = np.einsum("nk,nkc->nc", weights.astype(np.float32),
                           rim_colors[indices])

    target_field = np.zeros((*surface.shape, 3), np.float32)
    target_field[cap_rows, cap_cols] = target_rgb

    def masked_gauss(field: Any, mask: Any, sigma: float) -> Any:
        m = mask.astype(np.float32)
        density = gaussian_filter(m, sigma)
        smoothed = gaussian_filter(field * m, sigma)
        return np.where(density > 1e-6, smoothed / np.maximum(density, 1e-6), 0.0)

    target_smooth = np.stack(
        [masked_gauss(target_field[..., c], cap, float(target_sigma) * s)
         for c in range(3)], axis=2)

    eps = 0.02
    log_colors = np.log(np.clip(rgba[..., :3], 0.0, 1.0) + eps)
    log_low = np.stack(
        [masked_gauss(log_colors[..., c], cap, 8.0 * s) for c in range(3)],
        axis=2)
    detail = np.where(cap[..., None], (log_colors - log_low) * float(detail_keep), 0.0)
    new_log = np.log(np.clip(target_smooth, 0.0, 1.0) + eps) + detail
    out = rgba.copy()
    out[cap, :3] = np.clip(np.exp(new_log[cap]) - eps, 0.0, 1.0)

    stats.update(applied=True, cap_texels=int(cap.sum()),
                 rim_anchors=int(len(rim_points)))
    if stats_out is not None:
        stats_out.update(stats)
    return out
