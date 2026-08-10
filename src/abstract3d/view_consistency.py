"""Row-consistency analysis for same-elevation multi-view conditioning images.

Geometry rationale (why 1-D row profiles are sufficient)
--------------------------------------------------------
All conditioning views in a turntable bundle (front/left/right/back) come from
cameras at the SAME elevation orbiting the vertical axis, with vertical
up-vectors and near-orthographic projection at subject distance. A 3D point at
world height Z then projects to the same normalized image height y in EVERY
view: the epipolar geometry between any two such views degenerates so that
epipolar lines are horizontal scanlines, and cross-view correspondence
constrains ONLY the row coordinate. The law is subject-agnostic — a mouth, a
chair's seat edge, and a mug's rim all obey it equally.

Multi-view shape models (e.g. Hunyuan3D-2mv) and texture bakes implicitly rely
on this law: a feature presented at row y_a in one view and at row y_b in
another is carved/painted as TWO hypotheses (the observed double-mouth failure
on human busts). Image-to-image synthesized views violate the law by
re-framing the subject — a vertical translation plus, sometimes, a mild
vertical rescale — relative to the clay render whose framing is exact by
construction (it is rendered from the actual geometry). Because the violation
is a 1-D remap of image rows, detection and repair both operate on per-row
horizontal-structure profiles: ``row_profile`` (measurement),
``row_alignment_score`` / ``best_row_shift`` (comparison), ``align_view_rows``
(repair), ``view_consistency_report`` (the gate).

Dependencies (numpy, Pillow) import lazily so the base install stays light;
missing dependencies raise a loud, actionable error (``mesh_ops`` policy).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from .errors import DependencyUnavailableError, InvalidRequestError

_VIEW_EXTRA_HINT = 'View-consistency analysis needs numpy + Pillow. Install them with: pip install "abstract3d[mesh]"'

# Calibration — measured on out/laurent-bust-redo/hunyuan_multiview (the
# double-mouth bundle) plus synthetic sweeps; tests/test_view_consistency.py
# re-verifies the real-bundle side whenever the bundle exists on disk.
# Aligned real pairs (texture refs vs their clay guides; clay-vs-clay opposite
# sides) score ~0.43..0.92 at zero shift with best shifts <= 0.021·H; the
# misaligned synthesized geometry views score 0.0 at zero shift with best
# remap correlation 0.67..0.75 at shifts 0.07..0.15·H; scrambled/unrelated
# subjects stay <= ~0.13 at zero shift and <= ~0.40 under the full remap
# search. The floors sit in the measured gaps, nearer the failing side, so
# borderline content degrades toward review, never toward silent acceptance.
DEFAULT_ACCEPT_FLOOR = 0.35
CORRECTABLE_FLOOR = 0.55
CONSISTENT_MAX_SHIFT_FRAC = 0.03

_REPORT_SHIFT_FRAC = 0.18  # detection searches wider than the correction default
_SMOOTH_SIGMA_FRAC = 0.008
_OCCUPANCY_WEIGHT = 0.2
_WARP_GAIN_MIN = 0.04
_WARP_SCALE_RANGE = 0.15
_WARP_SCALE_STEPS = 25
_MIN_HEIGHT = 16
_MIN_FOREGROUND_FRAC = 0.001
_ALPHA_SEGMENTING_FRAC = 0.01
_BG_COLOR_DISTANCE = 12.0
_MIN_OVERLAP_ROWS = 8
_TIE_EPS = 1e-12


def _import_numpy():
    try:
        import numpy  # noqa: PLC0415
    except Exception as e:  # pragma: no cover - exercised via error-path tests
        raise DependencyUnavailableError(_VIEW_EXTRA_HINT) from e
    return numpy


def _import_pil_image():
    try:
        from PIL import Image  # noqa: PLC0415
    except Exception as e:  # pragma: no cover
        raise DependencyUnavailableError(_VIEW_EXTRA_HINT) from e
    return Image


def _clamp01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def _as_rgba_array(image: Any) -> Any:
    """Coerce a PIL Image or numpy array to a float64 (H, W, 4) RGBA array.

    Accepted numpy shapes: (H, W), (H, W, 1) gray, (H, W, 2) LA, (H, W, 3)
    RGB, (H, W, 4) RGBA. Float inputs entirely within [0, 1] are treated as
    normalized and scaled to [0, 255]; values outside [0, 255] after that are
    refused loudly (silent clipping would hide a caller bug).
    """
    np = _import_numpy()
    Image = _import_pil_image()
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGBA"), dtype=np.float64)
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.ndim != 3 or arr.shape[2] not in (1, 2, 3, 4):
        raise InvalidRequestError(
            "Expected a PIL Image or an array shaped (H, W[, C]) with C in "
            f"{{1, 2, 3, 4}}, got shape {getattr(arr, 'shape', None)!r}."
        )
    was_float = np.issubdtype(arr.dtype, np.floating)
    arr = arr.astype(np.float64)
    if was_float and arr.size and float(arr.max()) <= 1.0 + 1e-9 and float(arr.min()) >= -1e-9:
        arr = arr * 255.0
    if arr.size and (float(arr.min()) < -1e-6 or float(arr.max()) > 255.0 + 1e-6):
        raise InvalidRequestError(
            "Array pixel values must lie in [0, 255] (or [0, 1] for float "
            f"arrays); got range [{float(arr.min()):.4g}, {float(arr.max()):.4g}]."
        )
    arr = np.clip(arr, 0.0, 255.0)
    h, w, c = arr.shape
    if c == 2:  # LA -> replicate luminance, keep alpha
        arr = np.concatenate([np.repeat(arr[..., :1], 3, axis=2), arr[..., 1:2]], axis=2)
    elif c == 1:
        arr = np.repeat(arr, 3, axis=2)
    if arr.shape[2] == 3:
        arr = np.concatenate([arr, np.full((h, w, 1), 255.0)], axis=2)
    return arr


def _foreground_mask(arr: Any) -> Any:
    """Subject mask (deterministic input contract). If the alpha channel
    actually segments something (>= 1% of pixels transparent) the mask is
    ``alpha >= 128``; otherwise the image is treated as opaque-on-uniform-
    background (the clay renders in real bundles): background = median border
    color, mask = "chromatically far from it". A defined contract, not a
    fallback — pass real alpha, a border-visible background, or a bbox.
    """
    np = _import_numpy()
    alpha = arr[..., 3]
    if float((alpha < 128.0).mean()) >= _ALPHA_SEGMENTING_FRAC:
        return alpha >= 128.0
    rgb = arr[..., :3]
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]], axis=0)
    background = np.median(border, axis=0)
    return np.abs(rgb - background).max(axis=2) > _BG_COLOR_DISTANCE


def _l1(np: Any, vector: Any) -> Any:
    total = float(vector.sum())
    return vector / total if total > 0.0 else vector


def _smooth_rows(np: Any, profile: Any, sigma: float) -> Any:
    """Gaussian smoothing: widens correlation peaks (a basin of attraction for
    the shift search); convolution commutes with translation, so it never
    biases the detected shift."""
    if sigma < 0.3:
        return profile
    radius = max(1, int(round(3.0 * sigma)))
    kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1, dtype=np.float64) / sigma) ** 2)
    return np.convolve(profile, kernel / kernel.sum(), mode="same")


def _profile_core(np: Any, rgb: Any, mask: Any, *, sigma: float) -> Any:
    """Per-row horizontal-structure energy over a masked region (L1-normalized;
    all-zero when the region carries no measurable structure at all).

    Blends two background-independent evidence channels: interior luminance
    edge energy (weight 0.8) — |vertical gradient| of horizontally pre-smoothed
    luminance on mask pixels whose vertical neighbors are also in-mask (the
    silhouette boundary is excluded: its gradient measures the arbitrary
    background) — carrying mouth/eye/chin (or seat-edge/rim) height evidence;
    and silhouette width-change energy (weight 0.2) — |d/dy of per-row mask
    occupancy| — which obeys the same law and keeps texture-poor subjects
    (clay renders) measurable. The outline channel stays subordinate because
    matching outlines with re-heighted interior features ARE the double-mouth
    defect and must not read as aligned. Each channel is sqrt-compressed
    before L1 normalization (raw energies are dominated by one or two huge
    peaks, which makes correlation brittle — measured on the real bundle); if
    one channel is identically zero the other carries the profile alone.
    """
    h = rgb.shape[0]
    if not bool(mask.any()):
        return np.zeros(h, dtype=np.float64)
    lum = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    padded = np.pad(lum, ((0, 0), (1, 1)), mode="edge")
    lum_s = (padded[:, :-2] + 2.0 * lum + padded[:, 2:]) / 4.0
    grad_y = np.zeros_like(lum)
    grad_y[1:-1] = (lum_s[2:] - lum_s[:-2]) * 0.5
    above = np.zeros_like(mask)
    above[1:] = mask[:-1]
    below = np.zeros_like(mask)
    below[:-1] = mask[1:]
    interior = mask & above & below
    edge_energy = np.where(interior, np.abs(grad_y), 0.0).sum(axis=1)
    occupancy = mask.sum(axis=1).astype(np.float64)
    silhouette = np.zeros(h, dtype=np.float64)
    silhouette[1:-1] = np.abs(occupancy[2:] - occupancy[:-2]) * 0.5

    interior_profile = _l1(np, np.sqrt(edge_energy))
    silhouette_profile = _l1(np, np.sqrt(silhouette))
    has_interior = float(interior_profile.sum()) > 0.0
    has_silhouette = float(silhouette_profile.sum()) > 0.0
    if has_interior and has_silhouette:
        blended = (
            (1.0 - _OCCUPANCY_WEIGHT) * interior_profile + _OCCUPANCY_WEIGHT * silhouette_profile
        )
    elif has_interior or has_silhouette:
        blended = interior_profile if has_interior else silhouette_profile
    else:
        return np.zeros(h, dtype=np.float64)
    return _l1(np, _smooth_rows(np, blended, sigma))


def _validate_bbox(bbox: Any, width: int, height: int) -> Tuple[int, int, int, int]:
    try:
        left, top, right, bottom = (int(v) for v in bbox)
    except (TypeError, ValueError) as e:
        raise InvalidRequestError(
            f"bbox must be (left, top, right, bottom) integers, got {bbox!r}."
        ) from e
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise InvalidRequestError(
            f"bbox {bbox!r} is out of bounds for a {width}x{height} image."
        )
    return left, top, right, bottom


def row_profile(image_rgba: Any, *, bbox: Optional[Tuple[int, int, int, int]] = None) -> Any:
    """Normalized horizontal-edge-energy per image row over the subject foreground.

    Same-elevation turntable views constrain features to identical normalized
    rows across views (module docstring), so this 1-D profile — length = image
    height, L1-normalized, zero on rows outside the subject — is a complete
    summary for cross-view height comparison. ``image_rgba`` may be a PIL
    Image or a numpy array (:func:`_as_rgba_array` lists accepted shapes);
    ``bbox`` optionally restricts analysis to a (left, top, right, bottom)
    pixel box (PIL convention). Raises :class:`InvalidRequestError` for images
    too small for scanline structure (< 16 px) or with no detectable
    foreground, :class:`DependencyUnavailableError` when numpy/Pillow are
    missing. All-zero only for a subject with no measurable horizontal
    structure at all (e.g. a perfectly uniform full-frame fill).
    """
    np = _import_numpy()
    arr = _as_rgba_array(image_rgba)
    height, width = arr.shape[:2]
    if height < _MIN_HEIGHT or width < _MIN_HEIGHT:
        raise InvalidRequestError(
            f"Image {width}x{height} is too small for row-consistency analysis "
            f"(needs at least {_MIN_HEIGHT}x{_MIN_HEIGHT})."
        )
    left, top, right, bottom = (
        _validate_bbox(bbox, width, height) if bbox is not None else (0, 0, width, height)
    )
    region = arr[top:bottom, left:right]
    mask = _foreground_mask(region)
    if float(mask.mean()) < _MIN_FOREGROUND_FRAC:
        raise InvalidRequestError(
            "No foreground detected: the alpha channel segments nothing and no "
            "pixels differ from the border background color. Pass an image with "
            "a real alpha matte, a background-contrasted subject, or a bbox."
        )
    core = _profile_core(np, region[..., :3], mask, sigma=_SMOOTH_SIGMA_FRAC * height)
    profile = np.zeros(height, dtype=np.float64)
    profile[top:bottom] = core
    return _l1(np, profile)


def _validate_profile(np: Any, profile: Any, name: str) -> Any:
    arr = np.asarray(profile, dtype=np.float64)
    if arr.ndim != 1:
        raise InvalidRequestError(f"{name} must be a 1-D row profile, got shape {arr.shape!r}.")
    if arr.shape[0] < _MIN_HEIGHT:
        raise InvalidRequestError(
            f"{name} has {arr.shape[0]} rows; at least {_MIN_HEIGHT} are required."
        )
    if not bool(np.isfinite(arr).all()):
        raise InvalidRequestError(f"{name} contains non-finite values.")
    if float(arr.min()) < 0.0:
        raise InvalidRequestError(f"{name} contains negative values; profiles are energies.")
    return arr


def _resample_profile(np: Any, profile: Any, length: int) -> Any:
    """Resample onto ``length`` rows in NORMALIZED height coordinates — the
    consistency law is stated in normalized y, so different resolutions
    compare on a common grid."""
    if profile.shape[0] == length:
        return profile
    old_grid = np.linspace(0.0, 1.0, profile.shape[0])
    new_grid = np.linspace(0.0, 1.0, length)
    return _l1(np, np.interp(new_grid, old_grid, profile))


def _pearson(np: Any, a: Any, b: Any) -> float:
    """Signed Pearson correlation over the union of supported rows.

    Restricting to rows where either profile carries mass keeps shared empty
    background rows (a framing coincidence) from inflating similarity, while
    disjoint supports still read strongly negative.
    """
    selected = (a > 0.0) | (b > 0.0)
    if int(selected.sum()) < _MIN_OVERLAP_ROWS:
        return 0.0
    x = a[selected] - a[selected].mean()
    y = b[selected] - b[selected].mean()
    nx = float(np.sqrt((x * x).sum()))
    ny = float(np.sqrt((y * y).sum()))
    if nx <= 0.0 or ny <= 0.0:
        return 0.0
    return float((x * y).sum() / (nx * ny))


def _shift_correlation(np: Any, a: Any, b: Any, shift: int) -> float:
    """Correlation of ``a`` translated DOWN by ``shift`` rows against ``b``.
    Module-wide sign convention: positive = downward (larger row indices);
    ``a[i - shift]`` is compared with ``b[i]`` over the overlap window."""
    n = a.shape[0]
    if shift >= 0:
        xa, xb = a[: n - shift], b[shift:]
    else:
        xa, xb = a[-shift:], b[: n + shift]
    if xa.shape[0] < _MIN_OVERLAP_ROWS:
        return 0.0
    return _pearson(np, xa, xb)


def _best_shift(np: Any, a: Any, b: Any, max_shift_frac: float) -> Tuple[int, float]:
    frac = float(max_shift_frac)
    if not (0.0 < frac <= 0.45):
        raise InvalidRequestError(
            f"max_shift_frac must be in (0, 0.45], got {max_shift_frac!r} "
            "(beyond 0.45 the overlap window is too small to correlate honestly)."
        )
    n = a.shape[0]
    span = max(1, int(round(frac * n)))
    best_shift, best_corr = 0, -2.0
    for shift in range(-span, span + 1):
        corr = _shift_correlation(np, a, b, shift)
        # Ties prefer the smaller |shift| (zero drift on flat landscapes).
        if corr > best_corr + _TIE_EPS or (
            abs(corr - best_corr) <= _TIE_EPS and abs(shift) < abs(best_shift)
        ):
            best_shift, best_corr = shift, corr
    return best_shift, best_corr


def _warp_profile(np: Any, profile: Any, scale: float) -> Any:
    """Profile analogue of :func:`_warp_rows`: sample at scale*(i - c) + c."""
    n = profile.shape[0]
    center = (n - 1) / 2.0
    source = scale * (np.arange(n, dtype=np.float64) - center) + center
    warped = np.interp(source, np.arange(n, dtype=np.float64), profile, left=0.0, right=0.0)
    return _l1(np, warped)


def _best_remap(
    np: Any, a: Any, b: Any, max_shift_frac: float
) -> Tuple[float, float, int, int, float]:
    """Search the full correction family (shift; then scale + shift). Returns
    ``(best_corr, best_scale, best_scale_shift, shift_only, corr_shift_only)``.
    ONE search shared by detection and repair, so "correctable" always means
    "the corrector's own model explains the disagreement"."""
    shift_only, corr_shift_only = _best_shift(np, a, b, max_shift_frac)
    best_corr, best_scale, best_shift = corr_shift_only, 1.0, shift_only
    for scale in np.linspace(1.0 - _WARP_SCALE_RANGE, 1.0 + _WARP_SCALE_RANGE, _WARP_SCALE_STEPS):
        scale = float(scale)
        if abs(scale - 1.0) < 1e-9:
            continue
        candidate_shift, candidate_corr = _best_shift(
            np, _warp_profile(np, a, scale), b, max_shift_frac
        )
        if candidate_corr > best_corr + _TIE_EPS:
            best_corr, best_scale, best_shift = candidate_corr, scale, candidate_shift
    return best_corr, best_scale, best_shift, shift_only, corr_shift_only


def row_alignment_score(profile_a: Any, profile_b: Any) -> float:
    """[0..1] row-alignment similarity of two profiles at ZERO shift: Pearson
    correlation over the union of supported rows, clamped to [0, 1]
    (anti-correlation means no more than "not aligned" here). Different
    lengths compare in normalized height (``profile_b`` is resampled)."""
    np = _import_numpy()
    a = _validate_profile(np, profile_a, "profile_a")
    b = _validate_profile(np, profile_b, "profile_b")
    b = _resample_profile(np, b, a.shape[0])
    return _clamp01(_pearson(np, a, b))


def best_row_shift(profile_a: Any, profile_b: Any, *, max_shift_frac: float = 0.15) -> int:
    """Signed row shift maximizing correlation within +/- max_shift_frac of
    height: how far ``profile_a``'s content must translate (positive = down,
    negative = up) to best match ``profile_b``. Units are ``profile_a``'s
    rows; ``profile_b`` is resampled onto that grid if lengths differ."""
    np = _import_numpy()
    a = _validate_profile(np, profile_a, "profile_a")
    b = _validate_profile(np, profile_b, "profile_b")
    b = _resample_profile(np, b, a.shape[0])
    shift, _ = _best_shift(np, a, b, max_shift_frac)
    return int(shift)


def _translate_rows(np: Any, arr: Any, shift: int) -> Any:
    """Translate rows by ``shift`` (positive = down); revealed rows transparent."""
    out = np.zeros_like(arr)
    h = arr.shape[0]
    if abs(shift) >= h:
        return out
    if shift >= 0:
        out[shift:] = arr[: h - shift]
    else:
        out[: h + shift] = arr[-shift:]
    return out


def _warp_rows(np: Any, arr: Any, scale: float, shift: int) -> Any:
    """Linear y-remap: output row j samples input row scale*(j - shift - c) + c
    (``c`` = center row), matching :func:`_warp_profile` + translation exactly.
    Rows mapping outside the source become fully transparent. Interpolation is
    along y only — columns are never resampled.
    """
    h = arr.shape[0]
    center = (h - 1) / 2.0
    source = scale * (np.arange(h, dtype=np.float64) - float(shift) - center) + center
    low = np.floor(source).astype(int)
    frac = (source - low)[:, None, None]
    valid = (source >= 0.0) & (source <= float(h - 1))
    low_c = np.clip(low, 0, h - 1)
    high_c = np.clip(low + 1, 0, h - 1)
    out = arr[low_c] * (1.0 - frac) + arr[high_c] * frac
    out[~valid] = 0.0
    return out


def align_view_rows(
    image_rgba: Any,
    reference_clay_rgba: Any,
    *,
    max_shift_frac: float = 0.15,
) -> Tuple[Any, Dict[str, Any]]:
    """Align a synthesized view's rows to the clay reference's row profile.

    Returns ``(corrected_image_rgba, report_dict)``: a PIL RGBA image at the
    input's resolution plus a report carrying at least ``{"shift_rows": int,
    "score_before": float, "score_after": float, "warped": bool}``.

    Correction model — deliberately rigid, two tiers. (1) Pure vertical
    translation, always considered: shape-preserving, and it removes the
    dominant error mode measured on real failures (~0.07-0.15 of height).
    (2) Single-segment linear y-remap (uniform vertical scale about the image
    center + translation — the one-piece case of a piecewise-linear warp),
    applied only when it beats the best pure translation by a clear margin
    (``_WARP_GAIN_MIN``): the i2i re-framing error is measurably affine in y
    on real bundles (scale 0.89-1.13), and a y-only remap moves every feature
    to its lawful row while preserving row content. Multi-segment piecewise
    warps are deliberately NOT attempted — a nonrigid remap fitted to a noisy
    1-D signal can invent distortions worse than the defect it repairs.
    Honesty gate: the corrected image is re-measured; if its score does not
    improve, the ORIGINAL is returned unchanged with ``shift_rows == 0`` (the
    rejected remap stays visible via ``"measured_shift_rows"`` /
    ``"measured_scale"``; ``"applied"`` says what happened). Revealed rows are
    fully transparent; for opaque inputs the output alpha is meaningful only
    there — keep color keying downstream.
    """
    np = _import_numpy()
    Image = _import_pil_image()
    arr = _as_rgba_array(image_rgba)
    height, width = arr.shape[:2]
    if height < _MIN_HEIGHT or width < _MIN_HEIGHT:
        raise InvalidRequestError(
            f"Image {width}x{height} is too small for row alignment "
            f"(needs at least {_MIN_HEIGHT}x{_MIN_HEIGHT})."
        )
    mask = _foreground_mask(arr)
    if float(mask.mean()) < _MIN_FOREGROUND_FRAC:
        raise InvalidRequestError(
            "No foreground detected in the image to align (see row_profile); "
            "cannot measure row placement."
        )
    sigma = _SMOOTH_SIGMA_FRAC * height
    profile_img = _profile_core(np, arr[..., :3], mask, sigma=sigma)
    profile_ref = _resample_profile(np, row_profile(reference_clay_rgba), height)

    score_before = _clamp01(_pearson(np, profile_img, profile_ref))
    corr_warp, best_scale, warp_shift, shift_only, corr_shift = _best_remap(
        np, profile_img, profile_ref, max_shift_frac
    )

    use_warp = best_scale != 1.0 and corr_warp >= corr_shift + _WARP_GAIN_MIN
    if use_warp:
        corrected = _warp_rows(np, arr, best_scale, warp_shift)
        mask_channel = mask.astype(np.float64)[:, :, None] * 255.0
        corrected_mask = _warp_rows(np, mask_channel, best_scale, warp_shift)[:, :, 0] >= 128.0
        applied_shift, applied_scale = int(warp_shift), float(best_scale)
    else:
        corrected = _translate_rows(np, arr, shift_only)
        corrected_mask = (
            _translate_rows(np, mask.astype(np.float64)[:, :, None], shift_only)[:, :, 0] >= 0.5
        )
        applied_shift, applied_scale = int(shift_only), 1.0

    profile_after = _profile_core(np, corrected[..., :3], corrected_mask, sigma=sigma)
    score_after = _clamp01(_pearson(np, profile_after, profile_ref))

    applied = score_after > score_before
    if not applied:
        corrected = arr
        score_after = score_before
        applied_shift, applied_scale = 0, 1.0

    corrected_image = Image.fromarray(
        np.clip(np.rint(corrected), 0.0, 255.0).astype(np.uint8), "RGBA"
    )
    report: Dict[str, Any] = {
        "shift_rows": int(applied_shift),
        "score_before": float(score_before),
        "score_after": float(score_after),
        "warped": bool(applied and use_warp),
        "scale": float(applied_scale if applied else 1.0),
        "shift_frac": float(abs(applied_shift) / height),
        "height_rows": int(height),
        "applied": bool(applied),
        "measured_shift_rows": int(warp_shift if use_warp else shift_only),
        "measured_scale": float(best_scale if use_warp else 1.0),
    }
    return corrected_image, report


def view_consistency_report(
    clay_rgba: Any,
    synthesized_rgba: Any,
    *,
    accept_floor: Optional[float] = None,
) -> Dict[str, Any]:
    """Judge whether a synthesized view obeys the same-elevation row law,
    comparing its row profile against the clay reference's (whose framing is
    exact by construction — rendered from the actual geometry). Shift units
    and ``shift_frac`` are the SYNTHESIZED image's rows; sign follows the
    module convention (negative = content must move UP to match the clay).

    Verdicts: ``"consistent"`` — zero-shift score >= ``accept_floor`` AND best
    shift within ``CONSISTENT_MAX_SHIFT_FRAC`` of height, use as-is;
    ``"correctable"`` — rows disagree but the corrector's remap family
    (translation, optionally + uniform vertical scale) reaches
    ``CORRECTABLE_FLOOR`` correlation (same content at unlawful heights —
    repair with :func:`align_view_rows`); ``"inconsistent"`` — no remap in the
    family explains the disagreement, drop or regenerate the view.
    ``accept_floor`` defaults to :data:`DEFAULT_ACCEPT_FLOOR`; both floors are
    calibrated on the real double-mouth bundle plus synthetic sweeps (see the
    calibration note beside the module constants).
    """
    np = _import_numpy()
    floor = DEFAULT_ACCEPT_FLOOR if accept_floor is None else float(accept_floor)
    if not (0.0 <= floor <= 1.0):
        raise InvalidRequestError(f"accept_floor must be in [0, 1], got {accept_floor!r}.")
    profile_syn = row_profile(synthesized_rgba)
    height = profile_syn.shape[0]
    profile_clay = _resample_profile(np, row_profile(clay_rgba), height)

    score = _clamp01(_pearson(np, profile_syn, profile_clay))
    corr_remap, remap_scale, remap_shift, shift_only, corr_shift = _best_remap(
        np, profile_syn, profile_clay, _REPORT_SHIFT_FRAC
    )
    score_at_best = _clamp01(corr_shift)
    score_at_remap = _clamp01(corr_remap)
    shift_frac = float(abs(shift_only) / height)

    if score >= floor and shift_frac <= CONSISTENT_MAX_SHIFT_FRAC:
        verdict = "consistent"
    elif score_at_remap >= CORRECTABLE_FLOOR:
        verdict = "correctable"
    else:
        verdict = "inconsistent"

    return {
        "score": float(score),
        "best_shift_rows": int(shift_only),
        "shift_frac": shift_frac,
        "verdict": verdict,
        "score_at_best_shift": float(score_at_best),
        "score_at_best_remap": float(score_at_remap),
        "best_remap_scale": float(remap_scale),
        "best_remap_shift_rows": int(remap_shift),
        "accept_floor": float(floor),
        "correctable_floor": float(CORRECTABLE_FLOOR),
        "consistent_max_shift_frac": float(CONSISTENT_MAX_SHIFT_FRAC),
        "search_max_shift_frac": float(_REPORT_SHIFT_FRAC),
        "height_rows": int(height),
    }


__all__ = [
    "CONSISTENT_MAX_SHIFT_FRAC",
    "CORRECTABLE_FLOOR",
    "DEFAULT_ACCEPT_FLOOR",
    "align_view_rows",
    "best_row_shift",
    "row_alignment_score",
    "row_profile",
    "view_consistency_report",
]
