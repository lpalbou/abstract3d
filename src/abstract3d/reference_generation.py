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

# Steer the i2i model away from baking its own lighting into what the bake
# must treat as albedo. Applied to every generated view.
DEFAULT_NEGATIVE_PROMPT = (
    "glossy shine, specular highlights, strong reflections, rim lighting, "
    "lens flare, harsh shadows, dramatic lighting"
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


def suppress_specular_highlights(
    image_rgba: Any,
    *,
    lightness_delta: float = 12.0,
    chroma_ratio: float = 0.85,
    body_sigma: float = 9.0,
    blend: float = 0.75,
) -> Tuple[Any, float]:
    """Pull baked specular highlights toward the local diffuse color.

    Diffusion models render glossy materials with their own studio light,
    and the bake must treat every generated pixel as albedo — a highlight
    that survives becomes a permanent pale blob on the texture (measured:
    the owl's crown, present in all four generated views, survived view
    consensus untouched). A highlight reads as LIGHTER and LESS SATURATED
    than the diffuse body around it, so: estimate the local body color with
    a heavy Gaussian of the foreground (normalized convolution), flag
    pixels that exceed it in L while falling below it in chroma, feather
    the mask, and blend flagged pixels toward the body estimate.

    Returns `(image, corrected_fraction)`.
    """

    import numpy as np
    from PIL import Image
    from scipy.ndimage import gaussian_filter
    from skimage import color as skcolor

    rgba = np.asarray(image_rgba.convert("RGBA"), dtype=np.float32) / 255.0
    alpha = rgba[:, :, 3]
    mask = alpha > 0.5
    if int(mask.sum()) < 1024:
        return image_rgba, 0.0
    lab = skcolor.rgb2lab(rgba[:, :, :3]).astype(np.float32)
    weight = mask.astype(np.float32)

    def masked_blur(field: Any) -> Any:
        numerator = gaussian_filter(field * weight, body_sigma)
        denominator = gaussian_filter(weight, body_sigma)
        return numerator / np.maximum(denominator, 1e-6)

    body = np.stack([masked_blur(lab[:, :, c]) for c in range(3)], axis=2)
    chroma = np.hypot(lab[:, :, 1], lab[:, :, 2])
    body_chroma = np.hypot(body[:, :, 1], body[:, :, 2])
    specular = (
        mask
        & (lab[:, :, 0] > body[:, :, 0] + float(lightness_delta))
        & (chroma < float(chroma_ratio) * body_chroma)
    )
    fraction = float(specular.mean())
    if not specular.any():
        return image_rgba, 0.0
    feather = np.clip(gaussian_filter(specular.astype(np.float32), 2.0) * 1.5, 0.0, 1.0)
    feather *= float(blend)
    corrected = lab * (1.0 - feather[:, :, None]) + body * feather[:, :, None]
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*negative Z values.*")
        rgb = np.clip(skcolor.lab2rgb(corrected.astype(np.float64)), 0.0, 1.0)
    out = rgba.copy()
    out[:, :, :3] = np.where(mask[:, :, None], rgb.astype(np.float32), rgba[:, :, :3])
    return Image.fromarray((out * 255.0 + 0.5).astype(np.uint8), "RGBA"), fraction


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


def auto_generation_ready(owner: Any, subject_hint: Optional[str]) -> Tuple[bool, str]:
    """Whether "auto" mode may fire. "on" bypasses these gates (explicit intent).

    Two conditions, both adversarially motivated:
    - the resolved i2i route must name an EXPLICITLY CONFIGURED provider
      (owner vision handle, `scene3d_image_provider`, or the env override) —
      the capability's fallback route is a remote API, and a default-on
      feature must never silently send a user's mesh renders (or money)
      to a provider they never chose;
    - a non-empty subject hint must exist — the i2i model conditions on an
      untextured clay render, and with no textual subject knowledge it
      invents materials and identity for exactly the default one-photo user.
    """

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
    if not (subject_hint or "").strip():
        return False, (
            "no subject hint (pass a prompt describing the subject so the "
            "synthesized views stay faithful to it)"
        )
    return True, "ready"


def _view_prompt(label: str, subject_hint: Optional[str]) -> str:
    subject = (subject_hint or "").strip() or "the exact object shown"
    view_phrase = {
        "back": "seen directly from behind",
        "side_left": "seen from its left side profile",
        "side_right": "seen from its right side profile",
        "bottom": "seen from directly underneath",
        "top": "seen from a high angle, looking down at its top",
    }.get(label, f"seen from the {label.replace('_', ' ')} view")
    return (
        f"{subject}, {view_phrase}, the same physical object with consistent "
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
    max_attempts: int = 2,
    negative_prompt: Optional[str] = DEFAULT_NEGATIVE_PROMPT,
    image_request: Optional[Mapping[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Synthesize reference views for `mesh` from its own clay renders.

    Returns `(views, report)`: `views` entries are bake-ready observed-view
    dicts (`rgba`/`azimuth_deg`/`elevation_deg`/`label`/`role`/`generated`);
    `report` carries per-angle acceptance, IoU scores, timings, and the
    prompts used, so bundles can persist an honest provenance record.
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

    views: List[Dict[str, Any]] = []
    report: Dict[str, Any] = {
        "angles": [],
        "accepted": 0,
        "rejected": 0,
        # Provenance: what actually produced these pixels.
        "image_request": {k: str(v) for k, v in request.items()},
        "subject_hint": subject_hint,
        "negative_prompt": negative_prompt,
        "base_seed": int(seed),
        "silhouette_iou_min": float(silhouette_iou_min),
    }
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
        prompt = _view_prompt(label, subject_hint)
        entry["prompt"] = prompt

        accepted_rgba: Optional[Any] = None
        for attempt in range(int(max_attempts)):
            attempt_seed = int(seed) + attempt
            buffer = io.BytesIO()
            clay.convert("RGB").save(buffer, format="PNG")
            call_kwargs = dict(request)
            if negative_prompt:
                call_kwargs["negative_prompt"] = negative_prompt
            try:
                payload = generator(prompt, buffer.getvalue(),
                                    seed=attempt_seed, **call_kwargs)
                data = payload if isinstance(payload, (bytes, bytearray)) else None
                if data is None and isinstance(payload, Mapping):
                    for key in ("data", "bytes", "content"):
                        if isinstance(payload.get(key), (bytes, bytearray)):
                            data = payload[key]
                            break
                if data is None:
                    entry["attempts"].append(
                        {"seed": attempt_seed,
                         "error": "generator returned no image bytes"})
                    continue
                generated = Image.open(io.BytesIO(bytes(data)))
                matted = remove_background_robust(generated)
            except Exception as exc:
                entry["attempts"].append(
                    {"seed": attempt_seed, "error": f"{type(exc).__name__}: {exc}"})
                continue
            iou = silhouette_iou(matted, clay)
            entry["attempts"].append({"seed": attempt_seed, "silhouette_iou": round(iou, 4)})
            if iou >= float(silhouette_iou_min):
                accepted_rgba = matted
                entry["silhouette_iou"] = round(iou, 4)
                entry["accepted_image_md5"] = hashlib.md5(bytes(data)).hexdigest()
                break

        if accepted_rgba is None:
            entry["seconds"] = round(time.perf_counter() - started, 1)
            report["angles"].append(entry)
            report["rejected"] += 1
            continue

        try:
            accepted_rgba, specular_fraction = suppress_specular_highlights(accepted_rgba)
            entry["specular_suppressed_fraction"] = round(specular_fraction, 4)
        except Exception as exc:
            entry["specular_suppression_error"] = f"{type(exc).__name__}: {exc}"

        if tone_match:
            try:
                accepted_rgba, tone_stats = match_tone_lab(accepted_rgba, source_rgba)
                entry["tone_match"] = tone_stats
            except Exception as exc:
                entry["tone_match"] = {"applied": False,
                                       "error": f"{type(exc).__name__}: {exc}"}

        entry["accepted"] = True
        entry["seconds"] = round(time.perf_counter() - started, 1)
        report["angles"].append(entry)
        report["accepted"] += 1
        views.append(
            {
                "rgba": accepted_rgba,
                "azimuth_deg": float(azimuth),
                "elevation_deg": float(elevation),
                "label": label,
                "role": "reference",
                "generated": True,
                "clay_render": clay,
            }
        )
    return views, report
