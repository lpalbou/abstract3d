"""Generated reference views: complete single-photo texture coverage with
synthesized photos of the unseen angles.

A single photo can only witness ~25-30% of a closed surface; everything else
falls to mirror completion and harmonic fill. The validated remedy (proof
bundles under `artifacts/validation/generated-references/`) is to SYNTHESIZE
the missing views and feed them into the bake as ordinary references:

1. render the reconstructed mesh itself from the target angle (a clay
   render locks the silhouette and pose — free-form i2i without it invents
   a different object; measured on the starship underside),
2. condition an `abstractvision` image-to-image generation on that render,
3. accept only generations whose subject silhouette matches the clay
   silhouette (IoU gate; below the gate the projector would smear content
   across the surface),
4. match the generation's foreground tone to the source photo in LAB space
   (the bake's harmonization reconciles residual differences),

Generated views are plausible synthesis, not ground truth: they replace the
fill's characterless wash with coherent material, and every generated view
is marked as such in the bake stats and bundle metadata.
"""

from __future__ import annotations

import io
import math
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

# The default angle set covers the canonical reference slots: the back and
# both profiles complement a front-ish source photo. Callers with knowledge
# of the subject (e.g. a flying vehicle needing its underside) pass their
# own angle list.
DEFAULT_ANGLES: Tuple[Tuple[str, float, float], ...] = (
    ("back", 180.0, 0.0),
    ("side_left", 90.0, 0.0),
    ("side_right", -90.0, 0.0),
    # Elevated view: every elevation-0 view sees the top of the subject at
    # grazing incidence only, so crown/top texels otherwise inherit their
    # color from the worst witnesses (measured on the owl: the crown was
    # the only region still carrying baked specular after 3 views).
    ("top", 0.0, 55.0),
)

# Finish/rendering qualities only — never material nouns (any named
# material may be the CORRECT one for some subject, and negated mentions
# still activate the concept). Note: guidance-distilled FLUX.2-klein routes
# ignore negative prompts entirely (verified in the mlx backend's model
# table), so the positive wording and the texture gate carry the fix there;
# the negative earns its keep on CFG-capable editors (Qwen).
DEFAULT_NEGATIVE_PROMPT = (
    "smooth glazed finish, glossy shine, polished surface, specular "
    "highlights, reflections, airbrushed, soft focus, blurred detail, "
    "simplified surface, flat lifeless texture, CGI render, rim lighting, "
    "lens flare"
)

# Appended on texture-gate retries: smoothing is a systematic bias, so the
# retry shifts the prompt mean instead of only re-rolling the seed.
TEXTURE_ESCALATION_CLAUSE = (
    " The left photo's surface texture must be copied literally, groove for "
    "groove and mark for mark; a smoothed, cleaned-up or glazed surface is "
    "wrong."
)


def _foreground_stats(image_rgba: Any) -> Optional[Tuple[Any, Any]]:
    """Per-channel LAB mean/std over the alpha foreground."""

    import numpy as np
    from PIL import Image

    rgba = np.asarray(image_rgba.convert("RGBA"), dtype=np.float32) / 255.0
    mask = rgba[:, :, 3] > 0.5
    if int(mask.sum()) < 256:
        return None
    from skimage import color as skcolor

    lab = skcolor.rgb2lab(rgba[:, :, :3])
    fg = lab[mask]
    return fg.mean(axis=0), fg.std(axis=0) + 1e-6


def match_tone_lab(
    generated_rgba: Any,
    source_rgba: Any,
    *,
    max_shift_l: float = 15.0,
    max_shift_ab: float = 10.0,
) -> Tuple[Any, Dict[str, Any]]:
    """Move the generation's foreground LAB statistics TOWARD the source's.

    Statistics transfer (not histogram matching): the generation keeps its
    own content and contrast structure, only its global tone moves. The
    mean shift is CAPPED per channel: an unbounded transfer whitewashes a
    legitimately different unseen side (dark back on a light-fronted
    subject) into the front photo's statistics — manufacturing exactly the
    confidently-wrong content the gates exist to prevent. The pre-match
    distance and the applied (possibly clipped) shift are returned so the
    bundle records how far the generation's tone actually was.
    """

    import numpy as np
    from PIL import Image
    from skimage import color as skcolor

    stats: Dict[str, Any] = {"applied": False}
    source_stats = _foreground_stats(source_rgba)
    generated_stats = _foreground_stats(generated_rgba)
    if source_stats is None or generated_stats is None:
        return generated_rgba, stats
    src_mean, src_std = source_stats
    gen_mean, gen_std = generated_stats

    raw_shift = src_mean - gen_mean
    limit = np.array([max_shift_l, max_shift_ab, max_shift_ab], dtype=np.float64)
    shift = np.clip(raw_shift, -limit, limit)
    stats.update(
        applied=True,
        pre_match_distance=[round(float(v), 2) for v in raw_shift],
        applied_shift=[round(float(v), 2) for v in shift],
        clipped=bool((np.abs(raw_shift) > limit).any()),
    )

    rgba = np.asarray(generated_rgba.convert("RGBA"), dtype=np.float32) / 255.0
    mask = rgba[:, :, 3] > 0.5
    lab = skcolor.rgb2lab(rgba[:, :, :3])
    adjusted = (lab - gen_mean[None, None, :]) * (
        np.clip(src_std / gen_std, 0.5, 2.0)[None, None, :]
    ) + (gen_mean + shift)[None, None, :]
    # Keep the adjusted values inside the LAB gamut before conversion:
    # statistics transfer can push background pixels (excluded from the
    # stats) far outside, and lab2rgb warns per out-of-gamut pixel.
    adjusted[:, :, 0] = np.clip(adjusted[:, :, 0], 0.0, 100.0)
    adjusted[:, :, 1:] = np.clip(adjusted[:, :, 1:], -110.0, 110.0)
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*negative Z values.*")
        rgb = np.clip(skcolor.lab2rgb(adjusted), 0.0, 1.0)
    out = rgba.copy()
    out[:, :, :3] = np.where(mask[:, :, None], rgb, rgba[:, :, :3])
    return Image.fromarray((out * 255.0 + 0.5).astype(np.uint8), "RGBA"), stats


def clay_silhouette(clay_image: Any) -> Any:
    """Foreground mask of a clay render.

    The offscreen renderer emits RGB over a near-white constant background
    (0.95, 0.95, 0.93); the subject is every pixel that deviates from it.
    Distance in RGB (not luminance) so pale surfaces near the background
    tone still register through their shading gradients.
    """

    import numpy as np

    rgb = np.asarray(clay_image.convert("RGB"), dtype=np.float32) / 255.0
    background = np.array([0.95, 0.95, 0.93], dtype=np.float32)
    distance = np.abs(rgb - background[None, None, :]).max(axis=2)
    return distance > 0.04


def project_source_witness(
    mesh: Any,
    source_rgba: Any,
    *,
    source_pose: Tuple[float, float] = (0.0, 0.0),
    grid: int = 256,
    facing_min: float = 0.15,
) -> Tuple[Any, Any]:
    """Project the source photo onto the mesh: `(witnessed_mask, colors)`
    per vertex, no fill for unseen vertices.

    Projection is the clay camera's own formula (azimuth/elevation of the
    SOURCE view), with the photo's foreground bbox letterbox-mapped onto
    the mesh's projected bbox; visibility is a coarse-grid z-buffer (part-
    level correctness, not texel accuracy). Shared by the part-tinted
    conditioning guide (which fills unseen vertices) and the witnessed-
    consistency gate (which must NOT fill — its whole point is judging
    only where the photo is evidence).
    """

    import numpy as np

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    centered = vertices - vertices.mean(axis=0, keepdims=True)
    scale = float(np.max(np.linalg.norm(centered, axis=1))) or 1.0
    centered = centered / scale

    azimuth, elevation = (math.radians(float(source_pose[0])),
                          math.radians(float(source_pose[1])))
    eye = np.array([
        math.cos(elevation) * math.cos(azimuth),
        math.cos(elevation) * math.sin(azimuth),
        math.sin(elevation),
    ], dtype=np.float32)
    forward = -eye  # camera looks at the origin
    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    side = np.cross(forward, up)
    side /= max(float(np.linalg.norm(side)), 1e-8)
    cam_up = np.cross(side, forward)
    cam_x = centered @ side
    cam_y = centered @ cam_up
    cam_z = centered @ forward  # larger = farther from camera

    photo = np.asarray(source_rgba.convert("RGBA"), dtype=np.float32) / 255.0
    alpha = photo[:, :, 3] > 0.5
    colors = np.zeros((len(centered), 3), dtype=np.float32)
    if not alpha.any():
        return np.zeros(len(centered), dtype=bool), colors
    prow = np.any(alpha, axis=1)
    pcol = np.any(alpha, axis=0)
    pr0, pr1 = int(np.argmax(prow)), int(len(prow) - np.argmax(prow[::-1]))
    pc0, pc1 = int(np.argmax(pcol)), int(len(pcol) - np.argmax(pcol[::-1]))

    # Letterbox mapping: mesh-projection bbox -> photo foreground bbox,
    # one uniform scale (anisotropic stretch would smear part boundaries).
    mesh_w = float(cam_x.max() - cam_x.min()) or 1e-6
    mesh_h = float(cam_y.max() - cam_y.min()) or 1e-6
    photo_w, photo_h = float(pc1 - pc0), float(pr1 - pr0)
    fit = min(photo_w / mesh_w, photo_h / mesh_h)
    center_u = (pc0 + pc1) / 2.0
    center_v = (pr0 + pr1) / 2.0
    us = center_u + (cam_x - (cam_x.max() + cam_x.min()) / 2.0) * fit
    vs = center_v - (cam_y - (cam_y.max() + cam_y.min()) / 2.0) * fit

    # Coarse z-buffer visibility on a grid over the projected footprint.
    gx = np.clip(((cam_x - cam_x.min()) / mesh_w * (grid - 1)).astype(np.int32), 0, grid - 1)
    gy = np.clip(((cam_y - cam_y.min()) / mesh_h * (grid - 1)).astype(np.int32), 0, grid - 1)
    cell = gy * grid + gx
    order = np.argsort(cam_z)  # nearest first
    zbuf = np.full(grid * grid, np.inf, dtype=np.float32)
    np.minimum.at(zbuf, cell[order], cam_z[order])
    depth_eps = 0.03 * (float(cam_z.max() - cam_z.min()) or 1.0)
    visible = cam_z <= zbuf[cell] + depth_eps
    try:
        normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
        visible &= (normals @ (-forward)) > float(facing_min)
    except Exception:
        pass

    ui = np.clip(us.astype(np.int32), 0, photo.shape[1] - 1)
    vi = np.clip(vs.astype(np.int32), 0, photo.shape[0] - 1)
    on_subject = alpha[vi, ui]
    witnessed = visible & on_subject
    if witnessed.any():
        colors[witnessed] = photo[vi[witnessed], ui[witnessed], :3]
    return witnessed, colors


def tint_mesh_from_source(
    mesh: Any,
    source_rgba: Any,
    *,
    source_pose: Tuple[float, float] = (0.0, 0.0),
    grid: int = 256,
    facing_min: float = 0.15,
) -> Any:
    """Copy of `mesh` with vertex colors sampled from the source photo —
    the "part-tinted clay" conditioning guide.

    A neutral gray clay panel makes the generator GUESS each part's
    material, and on multi-part subjects it guesses wrong (measured: a
    wood-armed chair regenerated with upholstered arms from gray clay,
    while every part the source photo witnesses is unambiguous). Tinting
    every vertex with its observed color — and unseen vertices with their
    nearest witnessed neighbor's (mirror-symmetric candidates included) —
    hands the generator the per-part base color assignment so it refines
    materials instead of inventing them. The tint is a PRIOR, not truth:
    the acceptance gates still judge the result against the source photo.
    """

    import numpy as np
    import trimesh
    from scipy.spatial import cKDTree

    witnessed, colors = project_source_witness(
        mesh, source_rgba, source_pose=source_pose, grid=grid,
        facing_min=facing_min)
    if not witnessed.any():
        return mesh.copy()

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    centered = vertices - vertices.mean(axis=0, keepdims=True)
    scale = float(np.max(np.linalg.norm(centered, axis=1))) or 1.0
    centered = centered / scale

    tint = np.full_like(colors, 0.72)
    tint[witnessed] = colors[witnessed]
    seen_points = centered[witnessed]
    tree = cKDTree(seen_points)
    unseen = ~witnessed
    if unseen.any():
        query = centered[unseen]
        d_direct, i_direct = tree.query(query, workers=-1)
        mirrored = query.copy()
        mirrored[:, 1] *= -1.0  # canonical frame: x-z symmetry plane
        d_mirror, i_mirror = tree.query(mirrored, workers=-1)
        use_mirror = d_mirror < d_direct
        nearest = np.where(use_mirror, i_mirror, i_direct)
        tint[unseen] = tint[witnessed][nearest]

    tinted = mesh.copy()
    tinted.visual = trimesh.visual.ColorVisuals(
        tinted, vertex_colors=(np.clip(tint, 0.0, 1.0) * 255).astype(np.uint8))
    return tinted


def suppress_specular_highlights(
    image_rgba: Any,
    *,
    source_rgba: Optional[Any] = None,
    lightness_delta: float = 12.0,
    chroma_ratio: float = 0.85,
    body_sigma: float = 9.0,
    blend: float = 0.75,
    relief_band_threshold: float = 2.5,
) -> Tuple[Any, float]:
    """Pull baked specular highlights toward the local diffuse color —
    WITHOUT flattening carved relief.

    A highlight reads as LIGHTER and LESS SATURATED than the local diffuse
    body. Two adversarially-motivated guards on top of that predicate:

    - RELIEF EXEMPTION: pixels in a high band-pass-energy neighborhood are
      never corrected. Carved-ridge micro-highlights satisfy the specular
      predicate (measured: 2% of a matte carved-wood SOURCE photo flagged),
      and blending them toward a sigma-9 body estimate erases exactly the
      relief the transfer must preserve. True glaze highlights are broad
      and smooth — low band energy — so the exemption costs nothing there.
    - SOURCE CALIBRATION: when the source photo itself flags a similar
      fraction under the same predicate (its own false-positive floor),
      the correction blend is scaled down proportionally.

    Returns `(image, corrected_fraction)`.
    """

    import numpy as np
    from PIL import Image
    from scipy.ndimage import gaussian_filter
    from skimage import color as skcolor

    def flag_speculars(rgba_array: Any) -> Tuple[Any, Any, Any, Any]:
        alpha_mask = rgba_array[:, :, 3] > 0.5
        lab_array = skcolor.rgb2lab(rgba_array[:, :, :3]).astype(np.float32)
        weight = alpha_mask.astype(np.float32)

        def masked_blur(field: Any, sigma: float) -> Any:
            numerator = gaussian_filter(field * weight, sigma)
            denominator = gaussian_filter(weight, sigma)
            return numerator / np.maximum(denominator, 1e-6)

        body_est = np.stack(
            [masked_blur(lab_array[:, :, c], body_sigma) for c in range(3)], axis=2)
        chroma = np.hypot(lab_array[:, :, 1], lab_array[:, :, 2])
        body_chroma = np.hypot(body_est[:, :, 1], body_est[:, :, 2])
        flags = (
            alpha_mask
            & (lab_array[:, :, 0] > body_est[:, :, 0] + float(lightness_delta))
            & (chroma < float(chroma_ratio) * body_chroma)
        )
        # relief energy: 2-8 px band of the lightness channel
        band = masked_blur(lab_array[:, :, 0], 1.0) - masked_blur(lab_array[:, :, 0], 3.0)
        relief = gaussian_filter(np.abs(band), 3.0)
        return flags, relief, lab_array, body_est

    rgba = np.asarray(image_rgba.convert("RGBA"), dtype=np.float32) / 255.0
    mask = rgba[:, :, 3] > 0.5
    if int(mask.sum()) < 1024:
        return image_rgba, 0.0
    specular, relief, lab, body = flag_speculars(rgba)
    # Relief exemption: never correct pixels living inside carved texture.
    specular &= relief < float(relief_band_threshold)
    fraction = float(specular.mean())
    if not specular.any():
        return image_rgba, 0.0

    effective_blend = float(blend)
    if source_rgba is not None:
        try:
            src = np.asarray(source_rgba.convert("RGBA"), dtype=np.float32) / 255.0
            if int((src[:, :, 3] > 0.5).sum()) >= 1024:
                src_flags, src_relief, _, _ = flag_speculars(src)
                src_flags &= src_relief < float(relief_band_threshold)
                source_floor = float(src_flags.mean())
                if fraction <= 2.0 * source_floor:
                    # The generation flags no more than the source's own
                    # false-positive floor: correcting would only damage.
                    effective_blend *= 0.25
        except Exception:
            pass

    feather = np.clip(gaussian_filter(specular.astype(np.float32), 2.0) * 1.5, 0.0, 1.0)
    feather *= effective_blend
    corrected = lab * (1.0 - feather[:, :, None]) + body * feather[:, :, None]
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*negative Z values.*")
        rgb = np.clip(skcolor.lab2rgb(corrected.astype(np.float64)), 0.0, 1.0)
    out = rgba.copy()
    out[:, :, :3] = np.where(mask[:, :, None], rgb.astype(np.float32), rgba[:, :, :3])
    return Image.fromarray((out * 255.0 + 0.5).astype(np.uint8), "RGBA"), fraction


def register_matte_to_clay(
    generated_rgba: Any,
    clay_image: Any,
    *,
    scale_candidates: Sequence[float] = (0.88, 0.94, 1.0, 1.06, 1.12),
    shift_range: float = 0.10,
    shift_step: float = 0.02,
) -> Tuple[Any, Dict[str, Any]]:
    """Similarity-register the generation's matte onto the clay silhouette.

    Composite/rotate conditioning lets the model reframe the subject a few
    percent; the projector needs the generation in the CLAY's frame (that
    frame is what the bake reprojects). A small scale/shift search over
    silhouette IoU absorbs the reframing before the acceptance gate, so
    the gate measures shape agreement, not framing agreement.
    """

    import numpy as np
    from PIL import Image

    clay_mask_full = clay_silhouette(clay_image)
    matte = generated_rgba.convert("RGBA")
    if matte.size != clay_image.size:
        matte = matte.resize(clay_image.size, Image.LANCZOS)
    height, width = clay_mask_full.shape
    alpha_full = np.asarray(matte)[:, :, 3] > 128
    if not alpha_full.any() or not clay_mask_full.any():
        return matte, {"applied": False, "reason": "empty masks"}

    # The scale/shift search runs on downsampled BOOLEAN masks (the IoU
    # objective is insensitive to fine detail at these step sizes); only
    # the single winning transform is applied at full resolution.
    search = 160
    clay_small = np.asarray(
        Image.fromarray(clay_mask_full.astype(np.uint8) * 255)
        .resize((search, search), Image.BILINEAR)) > 127
    alpha_small_img = Image.fromarray(alpha_full.astype(np.uint8) * 255).resize(
        (search, search), Image.BILINEAR)

    def small_iou(scale: float, fx: float, fy: float) -> float:
        scaled_size = max(8, int(round(search * scale)))
        scaled = alpha_small_img.resize((scaled_size, scaled_size), Image.BILINEAR)
        canvas = Image.new("L", (search, search), 0)
        canvas.paste(scaled, ((search - scaled_size) // 2 + int(round(fx * search)),
                              (search - scaled_size) // 2 + int(round(fy * search))))
        mask = np.asarray(canvas) > 127
        union = int((mask | clay_small).sum())
        return float((mask & clay_small).sum()) / union if union else 0.0

    best = (small_iou(1.0, 0.0, 0.0), 1.0, 0.0, 0.0)
    shifts = np.arange(-shift_range, shift_range + 1e-9, shift_step)
    for scale in scale_candidates:
        for fx in shifts:
            for fy in shifts:
                score = small_iou(float(scale), float(fx), float(fy))
                if score > best[0]:
                    best = (score, float(scale), float(fx), float(fy))
    _, scale, fx, fy = best
    if scale == 1.0 and abs(fx) < 1e-9 and abs(fy) < 1e-9:
        return matte, {"applied": False, "iou": round(best[0], 4)}

    scaled_size = (max(8, int(round(width * scale))),
                   max(8, int(round(height * scale))))
    scaled = matte.resize(scaled_size, Image.LANCZOS)
    registered = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    registered.paste(scaled, ((width - scaled_size[0]) // 2 + int(round(fx * width)),
                              (height - scaled_size[1]) // 2 + int(round(fy * height))))
    final_mask = np.asarray(registered)[:, :, 3] > 128
    union = int((final_mask | clay_mask_full).sum())
    final_iou = float((final_mask & clay_mask_full).sum()) / union if union else 0.0
    return registered, {
        "applied": True, "iou": round(final_iou, 4),
        "scale": scale, "fx": round(fx, 3), "fy": round(fy, 3),
    }


def silhouette_iou(generated_rgba: Any, clay_image: Any) -> float:
    """IoU between the generation's matte and the clay render's silhouette."""

    import numpy as np
    from PIL import Image

    clay = clay_silhouette(clay_image)
    matte = generated_rgba.convert("RGBA")
    if matte.size != clay_image.size:
        matte = matte.resize(clay_image.size, Image.LANCZOS)
    generated = np.asarray(matte)[:, :, 3] > 128
    union = int((generated | clay).sum())
    if union == 0:
        return 0.0
    return float((generated & clay).sum()) / float(union)


def default_i2i_generator(owner: Any) -> Callable[..., bytes]:
    """The provider-neutral i2i callable, mirroring the t23d composition
    resolution (`scene3d_image_provider` / env / configured default)."""

    from .errors import DependencyUnavailableError
    from .image_composition import (
        COMPOSITION_INSTALL_HINT,
        _abstractvision_capability,
        _owner_vision,
    )

    vision = _owner_vision(owner)
    if vision is not None and callable(getattr(vision, "i2i", None)):
        return lambda prompt, image, **kwargs: vision.i2i(prompt, image=image, **kwargs)
    capability = _abstractvision_capability(owner)
    if not callable(getattr(capability, "i2i", None)):
        raise DependencyUnavailableError(COMPOSITION_INSTALL_HINT)
    return lambda prompt, image, **kwargs: capability.i2i(prompt, image=image, **kwargs)


def parse_generation_angles(
    raw: Any,
) -> Optional[Tuple[Tuple[str, float, float], ...]]:
    """Parse a user-facing angle list into (label, azimuth, elevation) tuples.

    Accepts a sequence (or comma-separated string) of either angle labels
    ("back", "side_left", "bottom", "top", ...) or explicit
    "label:azimuth,elevation" entries, e.g. "bottom:0,-75". Returns None for
    empty input so callers fall back to DEFAULT_ANGLES.
    """

    if raw is None:
        return None
    if isinstance(raw, str):
        parts = [part.strip() for part in raw.split(";" if ";" in raw else ",")]
        # "back, side_left" style splits on commas; explicit az,el entries
        # must use the semicolon form ("bottom:0,-75; back:180,0").
        if any(":" in part for part in parts) and ";" not in raw:
            parts = [part.strip() for part in raw.split(";")]
        entries: Sequence[Any] = [part for part in parts if part]
    else:
        entries = list(raw)
    if not entries:
        return None

    named = {
        "back": (180.0, 0.0),
        "side_left": (90.0, 0.0),
        "side_right": (-90.0, 0.0),
        "front_left": (45.0, 0.0),
        "front_right": (-45.0, 0.0),
        "back_left": (135.0, 0.0),
        "back_right": (-135.0, 0.0),
        "top": (0.0, 55.0),
        "bottom": (0.0, -75.0),
    }
    resolved: List[Tuple[str, float, float]] = []
    for entry in entries:
        if isinstance(entry, (tuple, list)) and len(entry) == 3:
            resolved.append((str(entry[0]), float(entry[1]), float(entry[2])))
            continue
        text = str(entry).strip()
        if ":" in text:
            label, _, angles_text = text.partition(":")
            azimuth_text, _, elevation_text = angles_text.partition(",")
            resolved.append(
                (label.strip(), float(azimuth_text), float(elevation_text or 0.0)))
            continue
        key = text.lower()
        if key not in named:
            raise ValueError(
                f"Unknown generation angle {text!r}. Use one of "
                f"{sorted(named)} or the explicit 'label:azimuth,elevation' form."
            )
        azimuth, elevation = named[key]
        resolved.append((key, azimuth, elevation))
    return tuple(resolved)


def auto_generation_ready(owner: Any, subject_hint: Optional[str] = None) -> Tuple[bool, str]:
    """Whether "auto" mode may fire. "on" bypasses this gate (explicit intent).

    One condition: the resolved i2i route must name an EXPLICITLY CONFIGURED
    provider (owner vision handle, `scene3d_image_provider`, or the env
    override) — the capability's fallback route is a remote API, and a
    default-on feature must never silently send a user's mesh renders (or
    money) to a provider they never chose.

    A subject hint is NOT required: the source photo is the material
    authority (composite conditioning) and the subject noun is derived
    automatically (caption -> nouns-only extraction). The earlier hint
    requirement contradicted no-human-in-the-loop operation and pushed
    users to hand-write descriptions — the proven wrong-material vector.
    """

    del subject_hint  # kept for call-site compatibility
    from .image_composition import has_image_composer, resolve_image_generation_request

    if not has_image_composer(owner):
        return False, "no image composer available"
    vision = getattr(owner, "vision", None) if owner is not None else None
    request = resolve_image_generation_request(owner)
    if not request.get("provider") and not (
        vision is not None and callable(getattr(vision, "i2i", None))
    ):
        return False, (
            "no explicitly configured image provider (set scene3d_image_provider "
            "or ABSTRACT3D_IMAGE_PROVIDER, e.g. a local mlx-gen route)"
        )
    return True, "ready"


def _view_phrase(label: str) -> str:
    return {
        "back": "seen directly from behind",
        "side_left": "seen from its left side profile",
        "side_right": "seen from its right side profile",
        "bottom": "seen from directly underneath",
        "top": "seen from a high angle, looking down at its top",
    }.get(label, f"seen from the {label.replace('_', ' ')} view")


# Person-category words (from the AUTO caption, not from a human): human
# subjects are the one category where i2i editors systematically drift to
# "sculpture" renderings of the clay panel; the clause is generic to the
# whole category. The refusal gate reuses the list, so recall errs wide:
# a false positive skips one object generation (cheap, and "on" plus the
# person acknowledgment overrides); a false negative synthesizes a face.
_PERSON_WORDS = frozenset(
    "person people human humans man men woman women boy boys girl girls "
    "child children kid kids baby babies infant toddler lady ladies guy "
    "guys gentleman gentlemen bride groom male female portrait face head "
    "bust figure selfie".split())


def is_person_subject(subject_text: Optional[str]) -> bool:
    """Whether a caption/hint/noun names a human subject.

    Human subjects are categorically excluded from unattended generation
    (`person_policy="skip"`): an adversarial review measured i2i side/back
    synthesis drifting to a DIFFERENT person's face — different age, nose,
    skin — while every material gate strict-passed. No gate in the stack
    measures identity, so no gate can defend it; until one does (e.g. a
    face-embedding similarity floor), silently altering a person's likeness
    is a product-trust failure, not a texture defect.

    Tokenization is alphabetic-runs, not whitespace ("woman's" matches
    "woman"). Word-list detection is a stopgap; the robust upgrade path is
    a face detector on the source photo (tracked in KnowledgeBase).
    """

    import re

    tokens = set(re.findall(r"[a-z]+", (subject_text or "").lower()))
    return bool(tokens & _PERSON_WORDS)


def _person_clause(subject_noun: Optional[str]) -> str:
    words = set((subject_noun or "").lower().split())
    if words & _PERSON_WORDS:
        return (
            " The subject is a living person, not a statue: real human skin "
            "with its natural color from the left photo, and real individual "
            "hair strands with the left photo's hair color. It is the SAME "
            "person as the left photo: same age, same clean unblemished skin "
            "complexion, same clothing, and the same clean dry healthy hair."
        )
    return ""


_COLOR_SECTORS = (
    (20.0, "pink"), (55.0, "red"), (90.0, "orange"), (120.0, "yellow"),
    (165.0, "green"), (230.0, "cyan"), (290.0, "blue"), (330.0, "purple"),
    (360.0, "pink"),
)


def _foreground_chroma_stats(generated_rgba: Any, source_rgba: Any) -> Dict[str, Any]:
    """Foreground mean-LAB chroma of both images + their ratio (diagnosis)."""

    import numpy as np
    from skimage import color as skcolor

    def chroma(image: Any) -> float:
        rgba = np.asarray(image.convert("RGBA"), dtype=np.float32) / 255.0
        mask = rgba[:, :, 3] > 0.5
        if not mask.any():
            return 0.0
        lab = skcolor.rgb2lab(rgba[:, :, :3]).astype(np.float32)
        return float(np.hypot(lab[:, :, 1][mask].mean(), lab[:, :, 2][mask].mean()))

    generated = chroma(generated_rgba)
    source = chroma(source_rgba)
    return {
        "generated": round(generated, 1),
        "source": round(source, 1),
        "ratio": round(generated / max(source, 1e-6), 3),
    }


def measured_color_anchor(source_rgba: Any, *, min_chroma: float = 40.0,
                          min_weight: float = 0.20) -> Optional[str]:
    """Frozen-vocabulary color term for the source's dominant chromatic
    part, or None.

    Measured rationale (sports-car incident): with a gray conditioning
    guide described as "an untextured model", the i2i editor echoed the
    CLAY's material statistics and returned achromatic cars — pixels
    alone failed to carry "saturated red" across, and the material-word
    ban had stripped the color from every text channel. The ban exists
    because HUMAN text lies; a value measured from the source pixels
    cannot lie about its own source. Safety is structural: the term comes
    from a frozen ten-word hue enum (no free text, no material/finish
    nouns — a hue does not name a material class), it is computed here
    from `source_rgba` with no caller-supplied override, and the
    acceptance gates still judge the result against the pixels.
    Thresholds are calibrated so the anchor fires only for strongly
    chromatic dominant parts (car body C*=90 fires; owl 35, chair 37,
    portrait 22, starship 6 all stay silent).
    """

    import numpy as np
    from scipy.cluster.vq import kmeans2
    from skimage import color as skcolor

    rgba = np.asarray(source_rgba.convert("RGBA"), dtype=np.float32) / 255.0
    mask = rgba[:, :, 3] > 0.5
    if not mask.any():
        return None
    lab = skcolor.rgb2lab(rgba[:, :, :3]).astype(np.float32)[mask]
    if len(lab) > 40000:
        picks = np.random.default_rng(0).choice(len(lab), 40000, replace=False)
        lab = lab[picks]
    try:
        _, labels = kmeans2(lab, 3, minit="++", seed=0)
    except Exception:
        return None
    best: Optional[Tuple[float, Any]] = None
    for index in range(3):
        member = lab[labels == index]
        if len(member) < len(lab) * float(min_weight):
            continue
        median = np.median(member, axis=0)
        chroma = float(np.hypot(median[1], median[2]))
        if chroma < float(min_chroma):
            continue
        weight = len(member) / len(lab)
        if best is None or weight > best[0]:
            best = (weight, median)
    if best is None:
        return None
    median = best[1]
    hue = float(np.degrees(np.arctan2(median[2], median[1]))) % 360.0
    term = next(name for limit, name in _COLOR_SECTORS if hue <= limit)
    chroma = float(np.hypot(median[1], median[2]))
    lightness = float(median[0])
    if chroma > 60.0:
        term = f"saturated {term}"
    elif lightness < 30.0:
        term = f"dark {term}"
    elif lightness > 70.0:
        term = f"light {term}"
    return term


def _view_prompt(label: str, subject_noun: Optional[str],
                 conditioning: str = "clay", *, tinted: bool = False,
                 color_anchor: Optional[str] = None) -> str:
    """Build the generation instruction. `subject_noun` must already be
    material-free (see `captioning.extract_subject_noun`) — the template has
    no free-text slot, so nothing a human or captioner writes can inject a
    material claim structurally. `color_anchor`, when present, is a
    MEASURED pixel readout (see `measured_color_anchor`), not text anyone
    wrote.
    """

    noun = (subject_noun or "").strip() or "object"
    phrase = _view_phrase(label)
    if conditioning == "composite":
        # Adversarially designed wording: the copy-material clause leads
        # with MATERIAL-NEUTRAL relief vocabulary (self-normalizing — for a
        # smooth subject, copying its relief exactly yields smooth), the
        # result is named "a real photograph" (naming it a render biases
        # CG-smooth output), and re-interpretation is forbidden without
        # naming any material class. The prior wording ("repaint ... the
        # surface pattern") was satisfied literally by the shipped failure:
        # pattern kept as painted decoration, carved relief lost.
        right_panel = (
            "a rough model of the SAME subject painted with flat base "
            "colors" if tinted else "an untextured model of the SAME subject")
        tint_clause = (
            " The right panel's flat colors already mark which part is "
            "which material — keep that assignment." if tinted else "")
        color_clause = (
            f" The subject's main color is {color_anchor}, the same color "
            "as in the left photo — never the gray of the model."
            if color_anchor else "")
        # Surface clause, class-conditional (measured, sports-car ladder):
        # the relief enumeration ("carving depth, grooves, grain, cracks,
        # fibers") plus "Do not smooth, polish, glaze" plus "no gloss" is
        # right for carved/fibrous subjects but measurably WRONG for the
        # vivid smooth-finish class the color anchor marks — 8 of 8
        # candidates under the relief wording rendered craquelure on a
        # crack-free car, while dropping it (candidate K) produced clean
        # paint at material strict-pass. The docstring's old claim that
        # relief vocabulary is "self-normalizing" was refuted by that
        # measurement. If a vivid subject genuinely has relief, the
        # escalation ladder restores the relief demand on attempt 2.
        if color_anchor:
            surface_clause = (
                "Reproduce the same colors and pattern, and the same "
                "surface finish as the left photo: smooth, clean and "
                "unbroken — no cracks, no weathering, no added aging. "
                "Do not change any material type. ")
            finish = "soft diffuse lighting."
        else:
            surface_clause = (
                "Reproduce the same surface relief, carving depth, grooves, "
                "grain, cracks, fibers and micro-texture, and the same "
                "colors and pattern. Match the roughness of each left "
                "surface exactly. Do not change any material type. Do not "
                "smooth, polish, glaze or simplify any surface. ")
            finish = "soft diffuse lighting, no gloss."
        return (
            f"Left panel: a photo of {noun}. Right panel: {right_panel} "
            f"{phrase}. Make the right panel a real "
            "photograph of the left subject seen from this angle. Copy the "
            "left subject's material identity exactly, PART BY PART: each "
            "part keeps its own material, color and texture from the left "
            f"photo — never spread one part's material onto a different part."
            f"{tint_clause}{color_clause} "
            + surface_clause +
            "Keep the right panel's exact shape, "
            "pose and framing. Output only the finished right view as a "
            "single image, on a plain dark background, "
            + finish + _person_clause(noun)
        )
    if conditioning == "rotate":
        return (
            f"Rotate the camera to show this exact {noun} {phrase}. Keep the "
            "same object identity: the same surface relief, grooves, grain "
            "and micro-texture, the same colors and pattern. Do not change "
            "the material type. Plain dark background, soft diffuse lighting."
        )
    return (
        f"a {noun}, {phrase}, the same physical object with consistent "
        "materials and colors, photographed on a plain dark background, "
        "soft diffuse even studio lighting, matte finish, no harsh specular "
        "highlights or reflections, product photography, sharp focus"
    )


def generate_reference_views(
    mesh: Any,
    source_rgba: Any,
    *,
    owner: Any = None,
    image_generator: Optional[Callable[..., Any]] = None,
    angles: Sequence[Tuple[str, float, float]] = DEFAULT_ANGLES,
    subject_hint: Optional[str] = None,
    silhouette_iou_min: float = 0.75,
    tone_match: bool = True,
    render_size: int = 768,
    seed: int = 11,
    max_attempts: int = 3,
    negative_prompt: Optional[str] = DEFAULT_NEGATIVE_PROMPT,
    conditioning: str = "composite",
    # A/B-measured OFF: the part-tinted guide ("painted with flat base
    # colors") biased the editor toward flat-painted output — owl back
    # relief 0.76-0.80/flat_delta 0.23-0.26 (floor FAIL) with tint vs
    # 1.09/0.06 (strict PASS) with the plain gray clay guide.
    tint_conditioning: bool = False,
    source_pose: Tuple[float, float] = (0.0, 0.0),
    steps_schedule: Sequence[Optional[int]] = (8, 12, 12),
    image_request: Optional[Mapping[str, Any]] = None,
    person_policy: str = "skip",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Synthesize reference views for `mesh` from its own clay renders.

    Fully subject-agnostic: `subject_hint` (a user's t23d prompt, when one
    exists) and the automatic caption are both reduced to a MATERIAL-FREE
    noun phrase — the source photo is the only material authority, and the
    prompt template has no free-text slot.

    Returns `(views, report)`: `views` entries are bake-ready observed-view
    dicts (`rgba`/`azimuth_deg`/`elevation_deg`/`label`/`role`/`generated`);
    `report` carries per-angle acceptance, per-attempt IoU/texture metrics,
    timings, and the prompts used, so bundles persist an honest provenance
    record.

    `person_policy` ("skip" by default) refuses human subjects outright —
    see `is_person_subject` for why identity cannot currently be defended.
    Explicit opt-in callers may pass "proceed"; the report then records a
    `person_warning`.
    """

    import hashlib

    from PIL import Image

    from .rendering import get_last_render_backend, render_mesh_views
    from .segmentation import remove_background_robust

    generator = image_generator or default_i2i_generator(owner)
    if image_request:
        request = dict(image_request)
    else:
        # Resolve provider/model exactly like the composed-t23d stage does
        # (scene3d_image_provider / ABSTRACT3D_IMAGE_PROVIDER / configured
        # abstractvision default) so the same knobs govern both synthesis
        # stages. Without this, the capability's own default route may be a
        # remote provider the operator never configured for i2i.
        from .image_composition import resolve_image_generation_request

        request = resolve_image_generation_request(owner)
    request.pop("width", None)
    request.pop("height", None)
    request.pop("seed", None)
    request = {key: value for key, value in request.items() if value is not None}

    # Subject noun: nouns-only reduction of the user's text when present,
    # else of an automatic caption of the source photo. Material words are
    # structurally banned either way (see captioning._MATERIAL_STOPLIST).
    from .captioning import caption_image, extract_subject_noun

    caption: Optional[str] = None
    if not (subject_hint or "").strip():
        caption = caption_image(source_rgba)
    subject_noun = extract_subject_noun(subject_hint or caption)

    # Human-subject policy (see `is_person_subject`): "skip" refuses to
    # synthesize views of a person — the gate stack cannot defend facial
    # identity. An explicit caller may pass "proceed" (a person-specific
    # acknowledgment, not merely "on"); the report then carries a warning
    # so the risk is on the record.
    if person_policy not in ("skip", "proceed"):
        raise ValueError(
            f"person_policy must be 'skip' or 'proceed' (got {person_policy!r})")
    person_detected = is_person_subject(subject_hint) or is_person_subject(caption)
    caption_checked = caption is not None
    if person_policy == "skip" and not person_detected and (subject_hint or "").strip():
        # A hint that doesn't name a person is not evidence of absence —
        # caption the photo itself before unattended synthesis of what
        # might be someone's face.
        caption = caption_image(source_rgba)
        caption_checked = caption is not None
        person_detected = is_person_subject(caption)
    if person_policy == "skip" and not person_detected and not caption_checked:
        # FAIL CLOSED (adversarial round 2): a None caption means "person
        # status unknown", not "not a person". An unavailable captioner
        # must never become a permission grant for unattended synthesis of
        # what might be someone's face.
        return [], {
            "angles": [],
            "accepted": 0,
            "rejected": 0,
            "skipped": (
                "captioner unavailable: the person-subject check cannot run, "
                "so generation is refused (fail closed — an unavailable "
                "check is not a permission grant). Install transformers/BLIP "
                "for automatic captioning, provide real reference photos, or "
                "pass the person acknowledgment (allow_person_subjects / "
                "texture_reference_allow_person) to attest the subject may "
                "be synthesized."
            ),
            "person_detected": None,
            "caption": None,
            "subject_hint": subject_hint,
        }
    if person_detected and person_policy == "skip":
        return [], {
            "angles": [],
            "accepted": 0,
            "rejected": 0,
            "skipped": (
                "person subject detected "
                f"(caption={caption!r}, hint={subject_hint!r}): reference "
                "generation cannot guarantee facial identity, so synthesis "
                "of people requires an explicit person acknowledgment "
                "(allow_person_subjects / texture_reference_allow_person / "
                "--texture-reference-allow-person). Prefer real reference "
                "photos for people."
            ),
            "person_detected": True,
            "caption": caption,
            "subject_hint": subject_hint,
        }

    # Part-tinted conditioning guide: one tinted copy of the mesh, rendered
    # per angle for the composite's right panel (gray clay stays the
    # geometry/silhouette authority for gating and registration).
    tinted_mesh: Optional[Any] = None
    report_tint_error: Optional[str] = None
    if conditioning == "composite" and tint_conditioning:
        try:
            tinted_mesh = tint_mesh_from_source(
                mesh, source_rgba, source_pose=source_pose)
        except Exception as exc:
            report_tint_error = f"{type(exc).__name__}: {exc}"
            tinted_mesh = None

    views: List[Dict[str, Any]] = []
    report: Dict[str, Any] = {
        "angles": [],
        "accepted": 0,
        "rejected": 0,
        # Provenance: what actually produced these pixels.
        "image_request": {k: str(v) for k, v in request.items()},
        "subject_hint": subject_hint,
        "caption": caption,
        "subject_noun": subject_noun,
        "negative_prompt": negative_prompt,
        "conditioning": conditioning,
        "tinted_conditioning": tinted_mesh is not None,
        "base_seed": int(seed),
        "silhouette_iou_min": float(silhouette_iou_min),
    }
    if person_detected:
        report["person_detected"] = True
        report["person_warning"] = (
            "person subject: generated side/back views may not preserve "
            "facial identity (no identity gate exists); requested explicitly, "
            "proceeding"
        )
    if report_tint_error:
        report["tint_error"] = report_tint_error
    # Measured color anchor: pixels-only prompt redundancy for strongly
    # chromatic subjects (see `measured_color_anchor` — the gray-car
    # incident proved a correct measured claim adds real redundancy
    # against clay echo, while the human-text ban stays intact).
    try:
        color_anchor = measured_color_anchor(source_rgba)
    except Exception:
        color_anchor = None
    report["color_anchor"] = color_anchor
    for label, azimuth, elevation in angles:
        entry: Dict[str, Any] = {
            "label": label,
            "azimuth_deg": float(azimuth),
            "elevation_deg": float(elevation),
            "attempts": [],
            "accepted": False,
        }
        started = time.perf_counter()
        clay = render_mesh_views(
            mesh, size=int(render_size), azimuths=[float(azimuth)],
            elevation=float(elevation),
        )[0].convert("RGBA")
        renderer = get_last_render_backend()
        entry["clay_renderer"] = renderer
        if renderer != "moderngl":
            # The matplotlib fallback decimates faces above 300k and uses
            # painter's-algorithm depth: its silhouette is corrupted in a
            # way the IoU gate cannot detect (the gate would compare the
            # generation against the same corrupted clay).
            entry["error"] = (
                f"clay renderer is {renderer!r}; reference generation requires "
                "the moderngl offscreen renderer for a trustworthy silhouette"
            )
            entry["seconds"] = round(time.perf_counter() - started, 1)
            report["angles"].append(entry)
            report["rejected"] += 1
            continue
        base_prompt = _view_prompt(label, subject_noun, conditioning,
                                   tinted=tinted_mesh is not None,
                                   color_anchor=color_anchor)
        entry["prompt"] = base_prompt
        entry["conditioning"] = conditioning

        # Conditioning image per strategy. "composite" pairs the SOURCE
        # PHOTO with the clay render so the model transfers the real
        # materials instead of inventing them. Both panels are LETTERBOXED
        # (no anisotropic stretch of the material the model must copy) onto
        # the same dark background.
        buffer = io.BytesIO()
        if conditioning == "composite":
            panel = int(render_size)

            def letterbox(image: Any) -> Any:
                rgb = image.convert("RGB")
                scale = panel / max(rgb.size)
                new_size = (max(1, int(round(rgb.width * scale))),
                            max(1, int(round(rgb.height * scale))))
                resized = rgb.resize(new_size, Image.LANCZOS)
                box = Image.new("RGB", (panel, panel), (16, 16, 16))
                box.paste(resized, ((panel - new_size[0]) // 2,
                                    (panel - new_size[1]) // 2))
                return box

            # The right panel is the TINTED render when available (part
            # base colors sampled from the source photo — the generator
            # refines materials instead of guessing them per part); the
            # gray clay remains the silhouette/registration authority.
            guide = clay
            if tinted_mesh is not None:
                try:
                    guide = render_mesh_views(
                        tinted_mesh, size=int(render_size),
                        azimuths=[float(azimuth)], elevation=float(elevation),
                    )[0].convert("RGBA")
                except Exception:
                    guide = clay
            guide_dark = Image.new("RGB", guide.size, (16, 16, 16))
            guide_dark.paste(guide.convert("RGB"), (0, 0),
                             Image.fromarray(
                                 (clay_silhouette(guide) * 255).astype("uint8")))
            canvas = Image.new("RGB", (panel * 2, panel), (16, 16, 16))
            canvas.paste(letterbox(source_rgba), (0, 0))
            canvas.paste(letterbox(guide_dark), (panel, 0))
            canvas.save(buffer, format="PNG")
        elif conditioning == "rotate":
            source_rgba.convert("RGB").save(buffer, format="PNG")
        else:
            clay.convert("RGB").save(buffer, format="PNG")
        conditioning_bytes = buffer.getvalue()

        # Retry ladder: IoU failures re-roll the seed with the SAME prompt
        # (shape failure is stochastic); texture/material failures escalate
        # the prompt (smoothing and material drift are systematic biases).
        # Every IoU-passing candidate is kept with its metrics, but only a
        # STRICT pass may ship (see selection below). Three complementary
        # oracles, each catching a failure family the others are blind to
        # (calibrated on the v1+v2 critic-labeled set):
        #   texture_fidelity    – relief smoothing (wood -> glaze)
        #   part_material_fidelity – palette flips (black hair -> chocolate,
        #                            upholstery -> camouflage)
        #   gate_baked_speculars – glossy highlight fields (wet-look hair)
        from .material_gates import (
            cloud_evidence_delta,
            gate_baked_speculars,
            gate_witnessed_consistency,
            part_material_fidelity,
            texture_fidelity,
        )

        candidates: List[Dict[str, Any]] = []
        escalated = False
        # Anchor-class retry ladder (measured): per-seed two-key pass is
        # ~0.44 on the hardest angle, so best-of-6 spaced seeds reaches
        # 97% angle acceptance with 94% rerun agreement (early stop keeps
        # the EXPECTED attempt count near 2). Seeds are spaced 1000 apart
        # (adjacent seeds measured uncorrelated; spacing costs nothing)
        # and steps alternate 8/12 (steps-12 is NOT better on average —
        # same-seed 8->12 moved scores up to 15 dE in BOTH directions —
        # but alternation decorrelates the ladder from any per-steps
        # bias). Non-anchor subjects keep the short ladder: every
        # recorded fleet run passed by attempt 2.
        anchor_class = bool(color_anchor)
        attempt_budget = 6 if anchor_class else int(max_attempts)
        for attempt in range(attempt_budget):
            attempt_seed = int(seed) + (1000 * attempt if anchor_class else attempt)
            prompt = base_prompt + (TEXTURE_ESCALATION_CLAUSE if escalated else "")
            call_kwargs = dict(request)
            if negative_prompt:
                call_kwargs["negative_prompt"] = negative_prompt + (
                    ", smooth featureless surface" if escalated else "")
            # The distilled klein default is 4 denoise steps — too few for
            # micro-texture; measured on the owl back, the relief ratio
            # rises with steps. The schedule escalates alongside the prompt.
            if anchor_class:
                call_kwargs["steps"] = 8 if attempt % 2 == 0 else 12
            elif attempt < len(steps_schedule) and steps_schedule[attempt]:
                call_kwargs["steps"] = int(steps_schedule[attempt])
            attempt_row: Dict[str, Any] = {"seed": attempt_seed,
                                           "escalated": escalated,
                                           "steps": call_kwargs.get("steps")}
            try:
                payload = generator(prompt, conditioning_bytes,
                                    seed=attempt_seed, **call_kwargs)
                data = payload if isinstance(payload, (bytes, bytearray)) else None
                if data is None and isinstance(payload, Mapping):
                    for key in ("data", "bytes", "content"):
                        if isinstance(payload.get(key), (bytes, bytearray)):
                            data = payload[key]
                            break
                if data is None:
                    attempt_row["error"] = "generator returned no image bytes"
                    entry["attempts"].append(attempt_row)
                    continue
                generated = Image.open(io.BytesIO(bytes(data)))
                if conditioning == "composite" and generated.width > generated.height:
                    # Any wider-than-tall echo of the two-panel canvas keeps
                    # only the repainted right panel (the old >=1.6 aspect
                    # heuristic missed 4:3 echoes and burned whole ladders).
                    generated = generated.crop(
                        (generated.width - generated.height, 0,
                         generated.width, generated.height))
                matted = remove_background_robust(generated)
                matted, registration = register_matte_to_clay(matted, clay)
                attempt_row["registration"] = registration
                iou = silhouette_iou(matted, clay)
                attempt_row["silhouette_iou"] = round(iou, 4)
                if iou < float(silhouette_iou_min):
                    attempt_row["failure_family"] = "silhouette"
                    entry["attempts"].append(attempt_row)
                    continue  # same prompt, next seed
                # Post-processing BEFORE the texture gate: the gate must
                # judge the exact pixels the bake will consume.
                processed = matted
                try:
                    processed, specular_fraction = suppress_specular_highlights(
                        processed, source_rgba=source_rgba)
                    attempt_row["specular_suppressed_fraction"] = round(
                        specular_fraction, 4)
                except Exception as exc:
                    attempt_row["specular_suppression_error"] = (
                        f"{type(exc).__name__}: {exc}")
                if tone_match:
                    try:
                        processed, tone_stats = match_tone_lab(processed, source_rgba)
                        attempt_row["tone_match"] = tone_stats
                    except Exception as exc:
                        attempt_row["tone_match"] = {
                            "applied": False,
                            "error": f"{type(exc).__name__}: {exc}"}
                # F7 (measured on the sports-car ladder): the flat_delta
                # strict threshold 0.12 is calibrated on relief subjects;
                # on the anchor-marked smooth-finish class it rejects the
                # smoothness that is CORRECT (the one clean-paint
                # material-strict candidate scored 0.138 while 7 of 8
                # craquelure candidates passed) — the source's band energy
                # there is specular structure and panel lines, not
                # micro-relief. Strict relaxes to 0.18 for this class;
                # floor (0.20) and relief_ratio strict stay unchanged.
                texture = texture_fidelity(
                    processed, source_rgba,
                    **({"flat_delta_max": 0.18} if color_anchor else {}))
                material = part_material_fidelity(processed, source_rgba)
                specular = gate_baked_speculars(processed)
                # Diagnosis stats (persist-for-diagnosis contract): the
                # gray-car class was invisible in the shipped metadata —
                # foreground chroma makes it readable at a glance.
                try:
                    attempt_row["fg_chroma"] = _foreground_chroma_stats(
                        processed, source_rgba)
                except Exception:
                    pass
                attempt_row["texture"] = {
                    k: texture.get(k)
                    for k in ("passed", "floor", "relief_ratio", "flat_delta",
                              "s50", "selection_score", "reason")}
                attempt_row["material"] = {
                    k: material.get(k)
                    for k in ("passed", "floor", "worst_part_delta_e", "reason",
                              "ensemble_delta_e", "source_chroma_dispersion",
                              "generated_chroma_dispersion")
                    if material.get(k) is not None}
                attempt_row["speculars"] = {
                    k: specular.get(k)
                    for k in ("passed", "worst_blob_fraction")}
                if anchor_class:
                    # TWO-KEY STRICT LINE for the anchor class (measured:
                    # NO threshold on the global palette score separates
                    # this corpus — correct 3.9-24.4 vs wrong 11.1-40
                    # under every aggregation; the residual wrongness is
                    # POSITIONAL). Witnessed angles: consensus fence at
                    # 22 plus the witnessed-region veto. Witness-starved
                    # angles (a back from a front photo witnesses ~600
                    # px — physics): tighter consensus 16 plus the cloud-
                    # evidence key at 11. flat_delta leaves strict
                    # entirely (zero discrimination on 71 labeled
                    # candidates; it alone blocked 3 material-passing
                    # correct backs). Relief keeps only its floor; the
                    # collapse guard (consensus 40) and speculars stay.
                    consensus = material.get("worst_part_delta_e")
                    relief_ok = (
                        "relief_ratio" not in texture
                        or float(texture.get("relief_ratio") or 0.0) >= 0.65)
                    witness = gate_witnessed_consistency(
                        processed, mesh, source_rgba,
                        azimuth_deg=azimuth, elevation_deg=elevation,
                        source_pose=source_pose)
                    attempt_row["witness"] = {
                        k: witness.get(k)
                        for k in ("witnessed", "witnessed_px", "chroma_flip",
                                  "bright_flip", "witnessed_tile_median",
                                  "passed")
                        if witness.get(k) is not None}
                    if consensus is None:
                        material_ok = False
                    elif witness["witnessed"]:
                        material_ok = (float(consensus) <= 22.0
                                       and bool(witness["passed"]))
                    else:
                        cloud = cloud_evidence_delta(processed, source_rgba)
                        attempt_row["cloud_evidence_delta"] = (
                            round(cloud, 2) if cloud is not None else None)
                        material_ok = (float(consensus) <= 16.0
                                       and cloud is not None
                                       and float(cloud) <= 11.0)
                    strict_pass = bool(material_ok and relief_ok
                                       and specular["passed"])
                else:
                    strict_pass = bool(texture["passed"] and material["passed"]
                                       and specular["passed"])
                # Triage without pixels: name the failure class in the
                # record (the gray-car diagnosis needed a full regeneration
                # to learn what one string could have said).
                if not strict_pass:
                    material_reason = str(material.get("reason") or "")
                    if "chroma collapse" in material_reason:
                        attempt_row["failure_family"] = "chroma_collapse"
                    elif anchor_class and attempt_row.get("witness", {}).get("passed") is False:
                        attempt_row["failure_family"] = "witness_veto"
                    elif anchor_class:
                        attempt_row["failure_family"] = "palette_flip"
                    elif not material["passed"]:
                        attempt_row["failure_family"] = "palette_flip"
                    elif not texture["passed"]:
                        attempt_row["failure_family"] = "texture"
                    else:
                        attempt_row["failure_family"] = "speculars"
                floor_pass = bool(texture.get("floor", texture["passed"])
                                  and material.get("floor", material["passed"]))
                # One score to rank candidates: relief fidelity minus
                # penalties for palette drift and gloss.
                score = float(texture.get("selection_score") or 0.0)
                score -= 0.05 * max(0.0, float(
                    material.get("worst_part_delta_e") or 0.0) - 13.0)
                if not specular["passed"]:
                    score -= 0.3
                candidates.append({
                    "rgba": processed,
                    "iou": iou,
                    "strict": strict_pass,
                    "floor": floor_pass,
                    "score": score,
                    "seed": attempt_seed,
                    "raw_payload_md5": hashlib.md5(bytes(data)).hexdigest(),
                })
                entry["attempts"].append(attempt_row)
                if strict_pass:
                    break  # strict pass: stop the ladder
                # Escalate the literal-copy texture clause only where it
                # targets the defect: relief smoothing on a relief subject.
                # On the anchor-marked smooth class the clause re-injects
                # the craquelure bias it exists to fight (measured: every
                # relief-worded car candidate rendered cracked paint), and
                # for material/chroma failures the mechanism change is the
                # steps/seed re-roll, not texture wording.
                if not texture["passed"] and not color_anchor:
                    escalated = True
            except Exception as exc:
                attempt_row["error"] = f"{type(exc).__name__}: {exc}"
                entry["attempts"].append(attempt_row)
                continue

        entry["seconds"] = round(time.perf_counter() - started, 1)
        # Selection: best combined score among IoU-passing candidates that
        # pass ALL gates strictly. Floor-only candidates are recorded but
        # never baked: the v2 chair measured why — a floor-accepted view
        # leaked stained fabric straight into the bake, and a wrong texture
        # on an unseen angle is a worse product defect than the featureless
        # fill it displaces (fill is dull; wrong material is broken).
        viable = [c for c in candidates if c["strict"]]
        if not viable:
            if any(c["floor"] for c in candidates):
                entry["rejection_reason"] = (
                    "floor-only candidates (strict material gates not met); "
                    "floor quality is reported but never baked"
                )
            # PERSIST-FOR-DIAGNOSIS: the gray-car diagnosis required a
            # full rerun solely because no rejected pixel survived —
            # callers write these small copies to `rejected_refs/`.
            # The exact pixels the gates judged, downscaled; raw payloads
            # stay reproducible via seed + md5.
            # Cap 15 = 5 angles x 3 attempts: the previous 12 silently
            # dropped a fifth angle's rejects.
            rejected_images = report.setdefault("rejected_images", [])
            for attempt_index, candidate in enumerate(candidates):
                if len(rejected_images) >= 15:
                    break
                small = candidate["rgba"].copy()
                small.thumbnail((512, 512))
                rejected_images.append({
                    "label": label,
                    "attempt": attempt_index,
                    "image": small,
                })
            report["angles"].append(entry)
            report["rejected"] += 1
            continue
        best = max(viable, key=lambda c: c["score"])
        entry["accepted"] = True
        entry["silhouette_iou"] = round(best["iou"], 4)
        entry["texture_gate"] = "passed"
        entry["raw_payload_md5"] = best["raw_payload_md5"]
        report["angles"].append(entry)
        report["accepted"] += 1
        views.append(
            {
                "rgba": best["rgba"],
                "azimuth_deg": float(azimuth),
                "elevation_deg": float(elevation),
                "label": label,
                "role": "reference",
                "generated": True,
                "clay_render": clay,
            }
        )
    return views, report
