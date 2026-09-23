"""General photo-referenced evaluation of a reconstructed bust (mesh + texture).

Design ruling (operator, 2026-07-21): no defect-specific detectors built from
single examples (the "mouth-open detector" class is forbidden). Every observed
defect class must fall out of a SMALL number of general comparisons, each
justified by a general argument first and only then calibrated:

TEXTURE axis (photo is ground truth for every surface point it can see)
  T1 source-structure   The photo is projected onto the mesh through the
                        front camera (feature-row alignment + occlusion
                        test); at every angle with front-visible overlap the
                        textured render is compared against that projection
                        with block-matched local NCC of luminance-equalized
                        gradient fields. Fabricated, displaced, or missing
                        content disagrees locally no matter what it depicts
                        (ghost frames, lips-on-nose, pasted patches, smears).
  T1 source-chroma      delta-E on the a,b channels only, over the same
                        projected correspondence: exposure differences cannot
                        mask wrong-colored content.
  T2 palette            At every angle including the back: after ONE global
                        exposure normalization (estimated at the front pose,
                        where correspondence is guaranteed), subject pixels
                        must be colorable from the photo's own subject
                        palette AT THE CORRESPONDING HEIGHT BAND (+-1 band
                        of registration slack). Height-banding rides the
                        same orthographic invariance as the feature rows: an
                        upright subject shows the same material strata at
                        the same heights from every azimuth. Per-band outlier
                        thresholds are self-calibrated from the photo's own
                        distance distribution, so nothing subject- or
                        defect-specific is encoded.

MESH axis (clay shading + silhouettes; priors are anatomical, not per-feature)
  M1 mirror-symmetry    A bust is near-bilaterally symmetric: the render at
                        +az must equal the mirrored render at -az (exact for
                        a symmetric mesh under this renderer). One-sided
                        bulges, slabs, and lateral deformations break it.
                        +-90 pairs are mathematically degenerate (opposite
                        orthographic views share one occluding contour) and
                        are never scored.
  M2 profile-articulation  A human profile articulates several protrusions
                        (brow, nose, lips, chin); a "soft face" reconstructs
                        only a nose bump. Score: summed prominence of
                        secondary leading-edge extrema in the face band,
                        normalized by head depth — feature-agnostic.
  M3 cavity-mass        Local shading valleys (pixels much darker than their
                        neighborhood under headlight shading) measure carved
                        cavities and crease duplications on the face band.
                        The front photo legitimizes only thin creases for a
                        closed-mouth subject; fabricated cavities (open
                        mouth, doubled lips, striation valleys) add valley
                        mass. Scope: assumes the source photo shows no large
                        open cavities (the pipeline's portrait input
                        contract).
  M4 silhouette floor   IoU-maximizing registration of the front silhouette
                        against the photo matte. NON-SEPARATING on the
                        validation set (all plausible busts land 0.88-0.91)
                        and therefore NOT a discriminating gate: kept only
                        as a floor (0.80) against out-of-family shape
                        failures, and reported as a diagnostic.

Aggregation: every measure is reduced to its WORST angle (min for
agreements, max for disagreements) and an axis passes only if every measure
passes — one bad angle fails the model, because the operator inspects the
full orbit. Never a mean.

Thresholds live in :class:`EvaluationConfig`. Defaults were calibrated on
the 8-model laurent-bust-redo validation set (2026-07-21) by placing cuts
inside the gap between operator-BAD and operator-OK models. Measures that
could not separate that set (clay-vs-photo structure agreement, feature-row
NCC, shading noise, delta-E-with-L) were REJECTED; the numbers are in
docs/research/evaluation_strategy_v2.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np
from PIL import Image

# Canonical frame: Z-up, front +X (matches abstract3d.rendering). Exported
# scene.glb files are glTF Y-up / front +Z and need this un-rotation.
GLTF_TO_CANON = np.array(
    [
        [0.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class EvaluationConfig:
    """Angles, comparator geometry, and calibrated acceptance thresholds."""

    render_size: int = 768
    elevation_deg: float = 0.0
    # Full orbit (palette); the subset with usable front-visible overlap
    # (source agreement); mirror pairs (+-90 degenerate, never added).
    orbit_azimuths: Tuple[float, ...] = (
        0.0, 30.0, -30.0, 60.0, -60.0, 90.0, -90.0, 135.0, -135.0, 180.0,
    )
    source_azimuths: Tuple[float, ...] = (0.0, 30.0, -30.0, 60.0, -60.0)
    chroma_azimuths: Tuple[float, ...] = (0.0, 30.0, -30.0)
    symmetry_pairs: Tuple[float, ...] = (30.0, 60.0, 135.0)
    cavity_azimuths: Tuple[float, ...] = (0.0, 30.0, -30.0)

    # ---- comparator geometry (general, not defect-specific) ----
    patch_px: int = 48                 # structure patch size
    patch_search_px: int = 8           # local block-matching search radius
    patch_min_coverage: float = 0.55   # min region fraction for a live patch
    patch_disagree_below: float = 0.25 # best local NCC below this = disagree
    grad_energy_floor: float = 0.08    # rel. floor: below = "flat" patch
    front_visible_cos: float = 0.25    # world-normal . +X cut ("photo sees it")
    front_depth_tol: float = 0.035     # occlusion tolerance (normalized units)
    face_band: Tuple[float, float] = (0.18, 0.62)  # subject-height fractions
    cavity_depth: float = 0.12         # local shade deficit = cavity pixel
    cavity_sigma: float = 6.0          # neighborhood scale (px at 768)
    articulation_prominence: float = 0.008  # extremum floor, head-depth units
    palette_bands: int = 6
    palette_clusters: int = 6
    palette_min_mass: float = 0.02
    palette_self_quantile: float = 99.5  # photo self-distance -> threshold
    palette_floor: float = 10.0          # LAB floor for the outlier cut
    row_scale_range: Tuple[float, float] = (0.80, 1.25)
    row_shift_range: float = 0.15

    # ---- calibrated acceptance thresholds (validation set 2026-07-21;
    #      per-threshold gap evidence in docs/research/evaluation_strategy_v2.md)
    tex_structure_max: float = 0.32    # T1 worst-angle disagreeing-patch frac
    tex_chroma_max: float = 11.5       # T1 worst-angle chroma delta-E p95
    tex_palette_max: float = 0.015     # T2 worst-angle out-of-palette frac
    mesh_symmetry_iou_min: float = 0.93          # M1 worst pair
    mesh_articulation_min: float = 0.055         # M2 profile articulation
    mesh_cavity_max: float = 0.072     # M3 worst-angle cavity fraction
    mesh_silhouette_floor: float = 0.80          # M4 out-of-family floor


# --------------------------------------------------------------------------
# Small numeric helpers (pure; unit-testable without GL)
# --------------------------------------------------------------------------


def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    from skimage import color as skcolor

    return skcolor.rgb2lab(np.clip(rgb, 0.0, 255.0) / 255.0)


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized cross-correlation of two equal-length signals/fields."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    da, db = a - a.mean(), b - b.mean()
    denom = math.sqrt(float((da * da).sum()) * float((db * db).sum()))
    return float((da * db).sum() / denom) if denom > 1e-12 else 0.0


def subject_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """(row0, row1, col0, col1), exclusive ends; raises on empty mask."""
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0:
        raise ValueError("empty subject mask")
    return int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1


def edge_row_profile(lum: np.ndarray, mask: np.ndarray, rows: int = 256) -> np.ndarray:
    """Per-row mean horizontal-edge energy over interior subject pixels,
    resampled to a fixed row count over the subject's own bbox. Mean (not
    sum) so wide shoulder rows do not dominate narrow face rows; interior
    only so the silhouette step does not drown feature edges."""
    from scipy import ndimage

    interior = ndimage.binary_erosion(mask, iterations=3)
    gy = np.abs(np.diff(lum, axis=0))
    sel = interior[1:] & interior[:-1]
    sums = (gy * sel).sum(axis=1)
    counts = sel.sum(axis=1)
    prof = np.where(counts > 4, sums / np.maximum(counts, 1), 0.0)
    r0, r1, _, _ = subject_bbox(mask)
    prof = prof[max(r0, 0) : max(r1 - 1, r0 + 1)]
    xs = np.linspace(0.0, 1.0, num=max(len(prof), 2))
    return np.interp(np.linspace(0.0, 1.0, rows), xs, prof)


def refine_row_alignment(
    photo_profile: np.ndarray,
    render_profile: np.ndarray,
    scale_range: Tuple[float, float] = (0.80, 1.25),
    shift_range: float = 0.15,
    steps: int = 25,
) -> Tuple[float, float, float]:
    """Best (scale, shift, ncc) mapping render rows onto photo rows on edge
    profiles. This is the feature-row registration the recenter-only bake
    doctrine lacks: the photo anchors to the mesh by its FEATURE rows, so a
    bake that painted displaced rows disagrees with the projection instead
    of being reproduced by it."""
    n = len(photo_profile)
    base = np.linspace(0.0, 1.0, n)
    best = (1.0, 0.0, -1.0)
    for scale in np.linspace(scale_range[0], scale_range[1], steps):
        for shift in np.linspace(-shift_range, shift_range, steps):
            xs = base * scale + shift
            valid = (xs >= 0.0) & (xs <= 1.0)
            if valid.sum() < n // 2:
                continue
            score = ncc(np.interp(xs[valid], base, photo_profile), render_profile[valid])
            if score > best[2]:
                best = (float(scale), float(shift), float(score))
    return best


def structure_agreement(
    render_lum: np.ndarray,
    target_lum: np.ndarray,
    region: np.ndarray,
    *,
    patch_px: int = 48,
    search_px: int = 8,
    min_coverage: float = 0.55,
    energy_floor: float = 0.08,
    disagree_below: float = 0.25,
) -> Dict[str, object]:
    """Block-matched structure agreement between two luminance images over
    `region` (the photo-visible pixels).

    General comparator for "is the CONTENT here the content the photo put
    here": luminance is variance-equalized inside the region (exposure/tone
    cannot mask or fake placement errors), gradient-magnitude fields are
    compared patchwise with a +-search_px local search (absorbing the few-px
    residual of the global alignment), and a patch DISAGREES when its best
    local NCC stays below `disagree_below` or when exactly one side carries
    structure (fabricated or missing content). Score = disagreeing fraction."""
    import cv2
    from scipy import ndimage

    sel = region & np.isfinite(target_lum)
    if sel.sum() < patch_px * patch_px:
        return {"disagree_frac": float("nan"), "median": float("nan"), "patches": []}
    mu_r, sd_r = float(render_lum[sel].mean()), float(render_lum[sel].std()) or 1.0
    mu_t, sd_t = float(target_lum[sel].mean()), float(target_lum[sel].std()) or 1.0
    render_eq = (render_lum - mu_r) / sd_r * sd_t + mu_t
    # Region-mean fill outside the region BEFORE the gradient, else the
    # region boundary itself injects fake edges into every boundary patch.
    render_eq = np.where(sel, render_eq, mu_t)
    target_f = np.where(sel, target_lum, mu_t)

    def grad_mag(lum: np.ndarray) -> np.ndarray:
        gx = ndimage.sobel(lum, axis=1)
        gy = ndimage.sobel(lum, axis=0)
        return np.hypot(gx, gy).astype(np.float32)

    gr, gt = grad_mag(render_eq), grad_mag(target_f)
    interior = ndimage.binary_erosion(sel, iterations=2)
    scale_ref = float(np.percentile(gt[interior], 90)) if interior.any() else 1.0
    floor = energy_floor * max(scale_ref, 1e-6)

    r0, r1, c0, c1 = subject_bbox(sel)
    h, w = sel.shape
    scores: List[Tuple[int, int, float]] = []
    for pr in range(r0, r1, patch_px):
        for pc in range(c0, c1, patch_px):
            block = np.s_[pr : min(pr + patch_px, h), pc : min(pc + patch_px, w)]
            m = interior[block]
            if m.size == 0 or m.mean() < min_coverage:
                continue
            patch_r = gr[block]
            er = float(np.sqrt((patch_r[m] ** 2).mean()))
            et = float(np.sqrt((gt[block][m] ** 2).mean()))
            if er < floor and et < floor:
                continue  # both flat: no structure to judge
            if er < floor or et < floor:
                scores.append((pr, pc, 0.0))  # one-sided structure
                continue
            wr0, wr1 = max(pr - search_px, 0), min(pr + patch_px + search_px, h)
            wc0, wc1 = max(pc - search_px, 0), min(pc + patch_px + search_px, w)
            window = gt[wr0:wr1, wc0:wc1]
            if window.shape[0] < patch_r.shape[0] or window.shape[1] < patch_r.shape[1]:
                scores.append((pr, pc, max(ncc(patch_r, gt[block]), 0.0)))
                continue
            response = cv2.matchTemplate(window, patch_r, cv2.TM_CCOEFF_NORMED)
            scores.append((pr, pc, float(max(response.max(), 0.0))))
    if not scores:
        return {"disagree_frac": float("nan"), "median": float("nan"), "patches": []}
    values = np.array([s[2] for s in scores])
    return {
        "disagree_frac": float((values < disagree_below).mean()),
        "median": float(np.median(values)),
        "patches": scores,
    }


def chroma_disagreement(
    render_rgb: np.ndarray, target_rgb: np.ndarray, region: np.ndarray
) -> Dict[str, float]:
    """delta-E restricted to the a,b (chroma) channels; p95 so a localized
    defect is not diluted by a large agreeing region. Dropping L makes the
    measure exposure-immune."""
    if region.sum() < 64:
        return {"mean": float("nan"), "p95": float("nan")}
    lab_r = _rgb_to_lab(render_rgb)
    lab_t = _rgb_to_lab(target_rgb)
    d = np.sqrt(((lab_r[..., 1:] - lab_t[..., 1:]) ** 2).sum(axis=2))
    vals = d[region]
    return {"mean": float(vals.mean()), "p95": float(np.percentile(vals, 95))}


@dataclass(frozen=True)
class BandedPaletteModel:
    """Height-banded LAB palettes of the photo's subject pixels.

    An upright subject shows the same material strata at the same subject
    heights from every azimuth (the orthographic turntable invariance), so a
    render pixel at band b must be colorable from the photo's palette of
    band b-1/b/b+1 (one band of registration slack). Per-band outlier
    thresholds are the photo's own `self_quantile` nearest-palette distance
    (never below `floor`): the photo defines both the palettes and the
    tolerances."""

    band_centers: Tuple[np.ndarray, ...]   # per band: (k, 3) LAB or empty
    band_thresholds: Tuple[float, ...]
    photo_l_median: float
    photo_l_iqr: float

    @property
    def n_bands(self) -> int:
        return len(self.band_centers)

    def exposure_map(self, rgb: np.ndarray, region: np.ndarray) -> Tuple[float, float]:
        """Global L gain/offset mapping this image's subject-L distribution
        onto the photo's (median/IQR). Estimated ONCE at the front pose —
        where content correspondence with the photo is guaranteed — and
        applied to every angle: exposure is global, fabrications are local,
        so a global map cannot hide a local out-of-palette region."""
        lab = _rgb_to_lab(rgb)[region]
        l_med = float(np.median(lab[:, 0]))
        l_iqr = float(np.percentile(lab[:, 0], 75) - np.percentile(lab[:, 0], 25)) or 1.0
        gain = self.photo_l_iqr / l_iqr
        return gain, self.photo_l_median - l_med * gain

    def outlier_fraction(
        self, rgb: np.ndarray, mask: np.ndarray, exposure: Tuple[float, float]
    ) -> float:
        if mask.sum() < 64:
            return float("nan")
        r0, r1, _, _ = subject_bbox(mask)
        h = max(r1 - r0, 1)
        lab = _rgb_to_lab(rgb)
        lab = lab.copy()
        lab[..., 0] = lab[..., 0] * exposure[0] + exposure[1]
        total, outliers = 0, 0
        for b in range(self.n_bands):
            row_lo = r0 + int(b / self.n_bands * h)
            row_hi = r0 + int((b + 1) / self.n_bands * h)
            band_mask = np.zeros_like(mask)
            band_mask[row_lo:row_hi] = True
            band_mask &= mask
            if band_mask.sum() < 100:
                continue
            sel = lab[band_mask]
            ratios = []
            for nb in (b - 1, b, b + 1):
                if not (0 <= nb < self.n_bands) or len(self.band_centers[nb]) == 0:
                    continue
                diffs = sel[:, None, :] - self.band_centers[nb][None, :, :]
                dist = np.sqrt((diffs**2).sum(axis=2)).min(axis=1)
                ratios.append(dist / self.band_thresholds[nb])
            if not ratios:
                continue
            nearest = np.min(np.stack(ratios, axis=0), axis=0)
            total += len(sel)
            outliers += int((nearest > 1.0).sum())
        return float(outliers / total) if total else float("nan")


def build_palette(
    photo_rgb: np.ndarray,
    photo_mask: np.ndarray,
    *,
    bands: int = 6,
    clusters: int = 6,
    min_mass: float = 0.02,
    self_quantile: float = 99.5,
    floor: float = 10.0,
    seed: int = 7,
) -> BandedPaletteModel:
    """K-means LAB palette per subject-height band; clusters carrying less
    than `min_mass` of a band are dropped so a tiny accessory cannot
    legitimize a large fabricated region of that color."""
    from scipy.cluster.vq import kmeans2

    lab_img = _rgb_to_lab(photo_rgb)
    lab_all = lab_img[photo_mask]
    r0, r1, _, _ = subject_bbox(photo_mask)
    h = max(r1 - r0, 1)
    rng = np.random.default_rng(seed)
    band_centers: List[np.ndarray] = []
    band_thresholds: List[float] = []
    for b in range(bands):
        band_mask = np.zeros_like(photo_mask)
        band_mask[r0 + int(b / bands * h) : r0 + int((b + 1) / bands * h)] = True
        band_mask &= photo_mask
        lab = lab_img[band_mask]
        if len(lab) < 500:
            band_centers.append(np.zeros((0, 3)))
            band_thresholds.append(floor)
            continue
        if len(lab) > 40000:
            lab = lab[rng.choice(len(lab), 40000, replace=False)]
        centers, labels = kmeans2(lab, min(clusters, len(lab)), minit="++", seed=seed)
        mass = np.bincount(labels, minlength=len(centers)) / max(len(labels), 1)
        keep = centers[mass >= min_mass]
        if len(keep) == 0:
            keep = centers
        diffs = lab[:, None, :] - keep[None, :, :]
        self_dist = np.sqrt((diffs**2).sum(axis=2)).min(axis=1)
        band_centers.append(keep)
        band_thresholds.append(max(float(np.percentile(self_dist, self_quantile)), floor))
    return BandedPaletteModel(
        band_centers=tuple(band_centers),
        band_thresholds=tuple(band_thresholds),
        photo_l_median=float(np.median(lab_all[:, 0])),
        photo_l_iqr=float(
            np.percentile(lab_all[:, 0], 75) - np.percentile(lab_all[:, 0], 25)
        )
        or 1.0,
    )


def mirror_symmetry_iou(mask_pos: np.ndarray, mask_neg: np.ndarray) -> float:
    """IoU of the +az silhouette against the horizontally mirrored -az
    silhouette — NO alignment step.

    For a bilaterally symmetric mesh this renderer produces exactly
    mirror-identical frames (same per-view half-extent, mirrored geometry),
    so any deviation — including a framing-scale difference — IS asymmetry
    signal; an alignment step would absorb real one-sided bulges. +-90
    pairs are degenerate (opposite orthographic views share one occluding
    contour: they mirror-match for ANY mesh) and must never be scored."""
    m2 = mask_neg[:, ::-1]
    union = (mask_pos | m2).sum()
    inter = (mask_pos & m2).sum()
    return float(inter / union) if union else 0.0


def face_band_mask(mask: np.ndarray, band: Tuple[float, float]) -> np.ndarray:
    """Rows between band[0] and band[1] of the subject height — a geometric
    face band (below the hair dome, above the chest), never a named feature."""
    r0, r1, _, _ = subject_bbox(mask)
    h = r1 - r0
    out = np.zeros_like(mask)
    out[r0 + int(band[0] * h) : r0 + int(band[1] * h)] = True
    return out & mask


def cavity_fraction(
    clay_shade: np.ndarray,
    mask: np.ndarray,
    band: Tuple[float, float],
    *,
    depth: float = 0.12,
    sigma: float = 6.0,
) -> float:
    """Fraction of face-band pixels lying in a LOCAL shading valley — much
    darker than their neighborhood under headlight shading.

    Local contrast (not absolute darkness) is the point: uniformly dark
    downward-facing surfaces (under-jaw at elevation 0) carry no local
    deficit, while carved cavities and duplicated creases (open mouth slot,
    doubled lips, striation valleys) are exactly local deficits. The
    neighborhood mean is mask-normalized so the background can never bleed
    into rim pixels and fake a valley."""
    from scipy import ndimage

    sel = face_band_mask(mask, band) & ndimage.binary_erosion(mask, iterations=4)
    if sel.sum() < 200:
        return float("nan")
    weight = ndimage.gaussian_filter(mask.astype(np.float64), sigma)
    smoothed = ndimage.gaussian_filter(np.where(mask, clay_shade, 0.0), sigma)
    neighborhood = smoothed / np.maximum(weight, 1e-6)
    deficit = neighborhood - clay_shade
    return float(((deficit > depth) & sel).sum() / sel.sum())


def profile_articulation(
    mask: np.ndarray,
    band: Tuple[float, float],
    *,
    prominence_floor: float = 0.008,
) -> float:
    """Summed prominence of SECONDARY leading-edge extrema in the face band
    of a profile silhouette, in head-depth units.

    A human profile articulates several protrusions (brow, nose, lips, chin
    — any face, no template); a "soft face" reconstruction shows only the
    nose bump. The largest extremum is excluded (every face has a nose);
    what remains measures articulation. Feature-agnostic by construction:
    extrema are found, never named. Orientation-neutral: both silhouette
    edges are scored and the more articulated one is taken (the back-of-head
    edge of any bust is near-featureless in the face band, so the max picks
    the face edge without knowing which side faces the camera). The +-90
    views are mirror-degenerate, so one profile measurement covers both."""
    from scipy import ndimage, signal

    r0, r1, _, _ = subject_bbox(mask)
    h = r1 - r0
    rows = np.arange(r0 + int(band[0] * h), r0 + int(band[1] * h))
    left = np.full(len(rows), np.nan)
    right = np.full(len(rows), np.nan)
    widths: List[int] = []
    for i, row in enumerate(rows):
        cols = np.flatnonzero(mask[row])
        if cols.size:
            left[i], right[i] = cols[0], cols[-1]
            widths.append(int(cols.size))
    if not widths:
        return float("nan")
    depth_scale = float(np.percentile(widths, 90)) or 1.0
    sigma = max(1.5, 0.015 * len(rows))

    def edge_articulation(xs: np.ndarray) -> float:
        valid = np.isfinite(xs)
        if valid.sum() < 16:
            return float("nan")
        filled = np.interp(np.arange(len(xs)), np.flatnonzero(valid), xs[valid])
        smooth = ndimage.gaussian_filter1d(filled / depth_scale, sigma)
        # peaks of both signs: prominence sets are invariant to edge
        # orientation, so no side/sign convention is needed.
        _, props_hi = signal.find_peaks(smooth, prominence=prominence_floor)
        _, props_lo = signal.find_peaks(-smooth, prominence=prominence_floor)
        proms = sorted(
            list(props_hi["prominences"]) + list(props_lo["prominences"]),
            reverse=True,
        )
        return float(sum(proms[1:])) if len(proms) > 1 else 0.0

    scores = [s for s in (edge_articulation(left), edge_articulation(right)) if s == s]
    return max(scores) if scores else float("nan")


def registered_silhouette_iou(
    photo_mask: np.ndarray,
    render_mask: np.ndarray,
    *,
    band: float = 0.62,
    scale_range: Tuple[float, float] = (0.75, 1.35),
    coarse: int = 4,
) -> Dict[str, float]:
    """Maximum head-band IoU of the render silhouette against the photo
    matte over a similarity search (scale + translation). The search is the
    registration; the maximized IoU is the score. Non-separating within the
    validation family (all 0.88-0.91) — used only as an out-of-family FLOOR
    and diagnostic."""
    pr0, pr1, _, _ = subject_bbox(photo_mask)
    band_rows = int((pr1 - pr0) * band)
    photo_small = photo_mask[::coarse, ::coarse]
    ph, pw = photo_small.shape
    band_r0, band_r1 = pr0 // coarse, (pr0 + band_rows) // coarse
    photo_band = photo_small[band_r0:band_r1]

    rr0, rr1, rc0, rc1 = subject_bbox(render_mask)
    crop = render_mask[rr0:rr1, rc0:rc1][::coarse, ::coarse]
    best = {"iou": 0.0, "scale": 1.0, "dy": 0, "dx": 0}
    base_scale = (pr1 - pr0) / max(rr1 - rr0, 1)
    ph_band = max(band_r1 - band_r0, 1)
    for rel in np.linspace(scale_range[0], scale_range[1], 13):
        scale = base_scale * rel
        size = (
            max(int(round(crop.shape[1] * scale)), 4),
            max(int(round(crop.shape[0] * scale)), 4),
        )
        scaled = np.asarray(
            Image.fromarray((crop * 255).astype(np.uint8)).resize(size, Image.NEAREST)
        ) > 127
        for dy in range(-ph_band // 4, ph_band // 4 + 1, max(ph_band // 16, 1)):
            for dx in range(-pw // 8, pw // 8 + 1, max(pw // 24, 1)):
                canvas = np.zeros_like(photo_band)
                y0, x0 = dy, (pw - scaled.shape[1]) // 2 + dx
                ys0, xs0 = max(y0, 0), max(x0, 0)
                ys1 = min(y0 + scaled.shape[0], ph_band)
                xs1 = min(x0 + scaled.shape[1], pw)
                if ys1 <= ys0 or xs1 <= xs0:
                    continue
                canvas[ys0:ys1, xs0:xs1] = scaled[
                    ys0 - y0 : ys1 - y0, xs0 - x0 : xs1 - x0
                ]
                union = (canvas | photo_band).sum()
                if union == 0:
                    continue
                iou = float((canvas & photo_band).sum() / union)
                if iou > best["iou"]:
                    best = {"iou": iou, "scale": float(rel), "dy": dy, "dx": dx}
    return best


# --------------------------------------------------------------------------
# Photo reference
# --------------------------------------------------------------------------


@dataclass
class PhotoReference:
    rgb: np.ndarray            # (H, W, 3) float 0..255
    mask: np.ndarray           # subject mask
    lum: np.ndarray            # luminance 0..255
    edge_profile: np.ndarray   # 256-row edge signature
    palette: BandedPaletteModel
    mask_source: str           # "alpha" | "border-differencing (#FALLBACK)"

    @classmethod
    def load(cls, path: Path, config: "EvaluationConfig") -> "PhotoReference":
        img = Image.open(path)
        arr = np.asarray(img.convert("RGBA"), dtype=np.float64)
        rgb, alpha = arr[..., :3], arr[..., 3]
        if alpha.min() < 250:
            mask, source = alpha > 128, "alpha"
        else:
            # No matte: border-median background differencing. Degraded on
            # busy backgrounds — reported, never silent.
            border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
            bg = np.median(border, axis=0)
            mask = np.abs(rgb - bg).max(axis=2) > 24
            source = "border-differencing (#FALLBACK: pass a matted photo)"
        lum = rgb @ np.array([0.299, 0.587, 0.114])
        return cls(
            rgb=rgb,
            mask=mask,
            lum=lum,
            edge_profile=edge_row_profile(lum, mask),
            palette=build_palette(
                rgb,
                mask,
                bands=config.palette_bands,
                clusters=config.palette_clusters,
                min_mass=config.palette_min_mass,
                self_quantile=config.palette_self_quantile,
                floor=config.palette_floor,
            ),
            mask_source=source,
        )


# --------------------------------------------------------------------------
# Geometry renderer (world-position / normal / texture buffers)
# --------------------------------------------------------------------------


class BufferRenderer:
    """Offscreen orthographic renderer replicating abstract3d.rendering's
    camera (eye at 3.2 units, up +Z, per-view half-extent * 1.18) while also
    producing world positions and normals — the correspondence oracle for
    photo projection. Textured output uses the production preview's
    near-flat shading (0.88 + 0.12 diffuse); clay shading is the headlight
    law (0.15 + 0.85 n.v)."""

    _VERT = """
        #version 330
        in vec3 in_pos; in vec3 in_nrm; in vec3 in_color; in vec2 in_uv;
        out vec3 v_nrm; out vec3 v_color; out vec2 v_uv; out vec3 v_pos;
        uniform mat4 u_mvp;
        void main() {
            gl_Position = u_mvp * vec4(in_pos, 1.0);
            v_nrm = in_nrm; v_color = in_color; v_uv = in_uv; v_pos = in_pos;
        }
    """
    _FRAG = """
        #version 330
        in vec3 v_nrm; in vec3 v_color; in vec2 v_uv; in vec3 v_pos;
        layout(location=0) out vec4 f_rgb;
        layout(location=1) out vec4 f_nrm;
        layout(location=2) out vec4 f_pos;
        uniform sampler2D u_tex; uniform int u_use_texture;
        uniform vec3 u_light; uniform vec3 u_view_dir;
        void main() {
            vec3 base = v_color;
            if (u_use_texture == 1)
                base = texture(u_tex, vec2(v_uv.x, 1.0 - v_uv.y)).rgb;
            vec3 n = normalize(v_nrm);
            if (dot(n, u_view_dir) < 0.0) n = -n;
            float diffuse = max(dot(n, normalize(u_light)), 0.0);
            float shade = 0.88 + 0.12 * diffuse;
            f_rgb = vec4(base * shade, 1.0);
            f_nrm = vec4(n * 0.5 + 0.5, 1.0);
            f_pos = vec4(v_pos, 1.0);
        }
    """

    def __init__(self, mesh, size: int = 768):
        import moderngl

        self._gl = moderngl
        self.size = int(size)
        m = mesh.copy()
        marker = getattr(mesh, "metadata", {}).get("abstract3d_export_frame")
        if marker == "gltf_yup_front_pz":
            m.apply_transform(GLTF_TO_CANON)
        verts = np.asarray(m.vertices, dtype=np.float32)
        faces = np.asarray(m.faces, dtype=np.int32)
        center = 0.5 * (verts.min(axis=0) + verts.max(axis=0))
        radius = float(np.max(np.linalg.norm(verts - center, axis=1))) or 1.0
        self.verts_c = ((verts - center) / radius).astype(np.float32)
        normals = np.asarray(m.vertex_normals, dtype=np.float32)

        colors, uv, tex_img = self._extract_appearance(m)
        tri = lambda a, d: np.asarray(a, dtype=np.float32)[faces].reshape(-1, d)
        packed = np.concatenate(
            [tri(self.verts_c, 3), tri(normals, 3), tri(colors, 3), tri(uv, 2)],
            axis=1,
        ).astype(np.float32)

        from .rendering import _create_standalone_context

        ctx = _create_standalone_context(moderngl)
        self.ctx = ctx
        self.prog = ctx.program(vertex_shader=self._VERT, fragment_shader=self._FRAG)
        self.vbo = ctx.buffer(packed.tobytes())
        self.vao = ctx.vertex_array(
            self.prog, [(self.vbo, "3f 3f 3f 2f", "in_pos", "in_nrm", "in_color", "in_uv")]
        )
        self.textures = []
        if tex_img is not None:
            texture = ctx.texture(tex_img.size, 3, tex_img.tobytes())
            texture.build_mipmaps()
            texture.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
            texture.use(0)
            self.textures.append(texture)
            self.prog["u_use_texture"].value = 1
        else:
            dummy = ctx.texture((1, 1), 3, bytes([180, 180, 180]))
            dummy.use(0)
            self.textures.append(dummy)
            self.prog["u_use_texture"].value = 0
        self.prog["u_light"].value = (0.45, -0.35, 0.82)
        s = self.size
        self.att_rgb = ctx.texture((s, s), 4, dtype="f4")
        self.att_nrm = ctx.texture((s, s), 4, dtype="f4")
        self.att_pos = ctx.texture((s, s), 4, dtype="f4")
        self.depth = ctx.depth_renderbuffer((s, s))
        self.fbo = ctx.framebuffer([self.att_rgb, self.att_nrm, self.att_pos], self.depth)

    @staticmethod
    def _extract_appearance(mesh):
        visual = getattr(mesh, "visual", None)
        uv = getattr(visual, "uv", None) if visual is not None else None
        tex_img = None
        material = getattr(visual, "material", None) if visual is not None else None
        for attr in ("image", "baseColorTexture"):
            candidate = getattr(material, attr, None) if material is not None else None
            if candidate is not None and isinstance(candidate, Image.Image):
                tex_img = candidate.convert("RGB")
                break
        n = len(mesh.vertices)
        if uv is None or len(uv) != n or tex_img is None:
            uv, tex_img = np.zeros((n, 2), dtype=np.float32), None
        colors = np.full((n, 3), 0.72, dtype=np.float32)
        vc = getattr(visual, "vertex_colors", None) if visual is not None else None
        if isinstance(vc, np.ndarray) and len(vc) == n:
            colors = vc[:, :3].astype(np.float32)
            if colors.max(initial=0.0) > 1.0:
                colors = colors / 255.0
        return colors, np.asarray(uv, dtype=np.float32), tex_img

    @staticmethod
    def _look_at(eye: np.ndarray) -> np.ndarray:
        target = np.zeros(3, dtype=np.float32)
        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        fwd = target - eye
        fwd /= max(float(np.linalg.norm(fwd)), 1e-8)
        side = np.cross(fwd, up)
        side /= max(float(np.linalg.norm(side)), 1e-8)
        up2 = np.cross(side, fwd)
        view = np.eye(4, dtype=np.float32)
        view[0, :3], view[1, :3], view[2, :3] = side, up2, -fwd
        view[:3, 3] = -view[:3, :3] @ eye
        return view

    def render(self, azimuth: float, elevation: float) -> Dict[str, np.ndarray]:
        az, el = math.radians(azimuth), math.radians(elevation)
        eye = 3.2 * np.array(
            [math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)],
            dtype=np.float32,
        )
        view = self._look_at(eye)
        cam = self.verts_c @ view[:3, :3].T + view[:3, 3]
        half = max(float(np.max(np.abs(cam[:, :2]))) * 1.18, 1e-3)
        proj = np.eye(4, dtype=np.float32)
        proj[0, 0] = proj[1, 1] = 1.0 / half
        proj[2, 2], proj[2, 3] = -2.0 / 15.9, -16.1 / 15.9
        self.prog["u_mvp"].write((proj @ view).astype(np.float32).T.tobytes())
        eye_n = eye / float(np.linalg.norm(eye))
        self.prog["u_view_dir"].value = tuple(float(v) for v in eye_n)
        self.fbo.use()
        self.ctx.enable(self._gl.DEPTH_TEST)
        self.ctx.disable(self._gl.CULL_FACE)
        self.fbo.clear(0.0, 0.0, 0.0, 0.0)
        self.vao.render()
        s = self.size
        read = lambda att: np.frombuffer(att.read(), dtype=np.float32).reshape(s, s, 4)[::-1]
        rgb = read(self.att_rgb)
        nrm = read(self.att_nrm)
        pos = read(self.att_pos)
        mask = nrm[..., 3] > 0.5
        normals = nrm[..., :3] * 2.0 - 1.0
        n_view = normals @ view[:3, :3].T
        clay_shade = 0.15 + 0.85 * np.clip(n_view[..., 2], 0.0, 1.0)
        return {
            "rgb": np.clip(rgb[..., :3], 0.0, 1.0) * 255.0,
            "mask": mask,
            "normals_world": normals,
            "world": pos[..., :3],
            "clay_shade": np.where(mask, clay_shade, 1.0),
            "half_extent": half,
            "view": view,
        }

    def release(self) -> None:
        for r in (
            self.vao, self.vbo, self.prog, self.att_rgb, self.att_nrm,
            self.att_pos, self.depth, self.fbo, *self.textures, self.ctx,
        ):
            try:
                r.release()
            except Exception:
                pass


# --------------------------------------------------------------------------
# Photo projection through the front camera (mesh as correspondence oracle)
# --------------------------------------------------------------------------


@dataclass
class FrontProjection:
    """Maps world points to photo pixels through the front camera plus the
    feature-row alignment, and answers front-visibility (normal + occlusion).
    The photo's pixels never move; only the map is evaluated."""

    photo: PhotoReference
    front: Dict[str, np.ndarray]
    row_scale: float
    row_shift: float           # in units of the photo subject height
    config: EvaluationConfig
    front_x_map: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        # Frontmost surface x per front-image pixel (orthographic along +X:
        # the front camera sees the max-x surface) for occlusion tests.
        self.front_x_map = np.where(
            self.front["mask"], self.front["world"][..., 0], -np.inf
        )
        pr0, pr1, pc0, pc1 = subject_bbox(self.photo.mask)
        rr0, rr1, rc0, rc1 = subject_bbox(self.front["mask"])
        sy = (pr1 - pr0) / max(rr1 - rr0, 1) * self.row_scale
        sx = (pc1 - pc0) / max(rc1 - rc0, 1)
        self._row_map = (sy, pr0 + self.row_shift * (pr1 - pr0) - rr0 * sy)
        pcc, rcc = 0.5 * (pc0 + pc1), 0.5 * (rc0 + rc1)
        self._col_map = (sx, pcc - rcc * sx)

    def project(
        self, buffers: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(photo colors, photo luminance, validity) for every pixel of
        another view of the same mesh."""
        world = buffers["world"]
        mask = buffers["mask"]
        size = self.front["mask"].shape[0]
        half = float(self.front["half_extent"])
        view = self.front["view"]
        cam = world @ view[:3, :3].T + view[:3, 3]
        u = (cam[..., 0] / half * 0.5 + 0.5) * (size - 1)
        v = (1.0 - (cam[..., 1] / half * 0.5 + 0.5)) * (size - 1)
        vi = np.clip(np.round(v).astype(int), 0, size - 1)
        ui = np.clip(np.round(u).astype(int), 0, size - 1)
        front_visible = (
            mask
            & (buffers["normals_world"][..., 0] > self.config.front_visible_cos)
            & (world[..., 0] >= self.front_x_map[vi, ui] - self.config.front_depth_tol)
        )
        pv = vi * self._row_map[0] + self._row_map[1]
        pu = ui * self._col_map[0] + self._col_map[1]
        ph, pw = self.photo.mask.shape
        inside = (pv >= 0) & (pv <= ph - 1) & (pu >= 0) & (pu <= pw - 1)
        pvi = np.clip(np.round(pv).astype(int), 0, ph - 1)
        pui = np.clip(np.round(pu).astype(int), 0, pw - 1)
        valid = front_visible & inside & self.photo.mask[pvi, pui]
        colors = np.zeros(world.shape[:2] + (3,))
        lum = np.full(world.shape[:2], np.nan)
        colors[valid] = self.photo.rgb[pvi[valid], pui[valid]]
        lum[valid] = self.photo.lum[pvi[valid], pui[valid]]
        return colors, lum, valid


# --------------------------------------------------------------------------
# Evaluation orchestration
# --------------------------------------------------------------------------


def load_mesh_for_evaluation(glb_path: Path):
    import trimesh

    scene = trimesh.load(str(glb_path), process=False)
    mesh = scene.to_mesh() if isinstance(scene, trimesh.Scene) else scene
    if glb_path.suffix.lower() in {".glb", ".gltf"}:
        mesh.metadata["abstract3d_export_frame"] = "gltf_yup_front_pz"
    return mesh


def _measure(
    per_angle: Mapping[str, float], threshold: float, direction: str
) -> Dict[str, object]:
    finite = {k: v for k, v in per_angle.items() if v == v}
    if not finite:
        return {
            "per_angle": dict(per_angle), "worst": None, "worst_angle": None,
            "threshold": threshold, "direction": direction, "pass": False,
            "note": "no angle produced a finite score",
        }
    if direction == "min":
        worst_angle = min(finite, key=lambda k: finite[k])
        ok = finite[worst_angle] >= threshold
    else:
        worst_angle = max(finite, key=lambda k: finite[k])
        ok = finite[worst_angle] <= threshold
    return {
        "per_angle": {k: (round(v, 4) if v == v else None) for k, v in per_angle.items()},
        "worst": round(finite[worst_angle], 4),
        "worst_angle": worst_angle,
        "threshold": threshold,
        "direction": direction,
        "pass": bool(ok),
    }


def aggregate_verdict(
    measures: Mapping[str, Mapping[str, Mapping[str, object]]],
) -> Dict[str, object]:
    """Boolean acceptance with named sub-verdicts for a one-shot loop.

    Worst-angle semantics: each measure is already reduced to its worst
    angle; an axis passes only if EVERY measure passes (min-aggregation,
    never mean)."""
    failing: List[str] = []
    axis_ok: Dict[str, bool] = {}
    worst: Dict[str, Dict[str, object]] = {}
    for axis, group in measures.items():
        ok = True
        for name, m in group.items():
            if not m["pass"]:
                ok = False
                failing.append(f"{axis}.{name}@{m.get('worst_angle')}")
        axis_ok[axis] = ok
        worst[axis] = {
            name: {"angle": m.get("worst_angle"), "value": m.get("worst")}
            for name, m in group.items()
        }
    return {
        "mesh_ok": axis_ok.get("mesh", False),
        "texture_ok": axis_ok.get("texture", False),
        "accept": bool(axis_ok.get("mesh") and axis_ok.get("texture")),
        "failing_measures": failing,
        "worst_angle": worst,
    }


def evaluate_model(
    glb_path: Path,
    photo: PhotoReference,
    config: Optional[EvaluationConfig] = None,
    *,
    save_debug_dir: Optional[Path] = None,
) -> Dict[str, object]:
    """Full evaluation of one GLB against the photo: renders the orbit
    buffers, computes every measure per angle, and aggregates the verdict."""
    config = config or EvaluationConfig()
    mesh = load_mesh_for_evaluation(glb_path)
    renderer = BufferRenderer(mesh, size=config.render_size)
    try:
        buffers = {
            az: renderer.render(az, config.elevation_deg)
            for az in config.orbit_azimuths
        }
    finally:
        renderer.release()

    front = buffers[0.0]
    clay_profile = edge_row_profile(front["clay_shade"] * 255.0, front["mask"])
    row_scale, row_shift, row_ncc = refine_row_alignment(
        photo.edge_profile, clay_profile,
        scale_range=config.row_scale_range, shift_range=config.row_shift_range,
    )
    projection = FrontProjection(
        photo=photo, front=front, row_scale=row_scale, row_shift=row_shift,
        config=config,
    )

    def render_lum(buf: Dict[str, np.ndarray]) -> np.ndarray:
        return buf["rgb"] @ np.array([0.299, 0.587, 0.114])

    # ---- T1 source agreement over the front hemisphere -------------------
    t1_structure: Dict[str, float] = {}
    t1_chroma: Dict[str, float] = {}
    for az in config.source_azimuths:
        buf = buffers[az]
        target_rgb, target_lum, valid = projection.project(buf)
        result = structure_agreement(
            render_lum(buf), target_lum, valid,
            patch_px=config.patch_px, search_px=config.patch_search_px,
            min_coverage=config.patch_min_coverage,
            energy_floor=config.grad_energy_floor,
            disagree_below=config.patch_disagree_below,
        )
        t1_structure[str(az)] = result["disagree_frac"]
        if az in config.chroma_azimuths:
            t1_chroma[str(az)] = chroma_disagreement(buf["rgb"], target_rgb, valid)["p95"]
        if save_debug_dir is not None:
            save_debug_dir.mkdir(parents=True, exist_ok=True)
            side = np.concatenate([buf["rgb"], target_rgb], axis=1).astype(np.uint8)
            Image.fromarray(side).save(save_debug_dir / f"projection_az{az:+.0f}.png")

    # ---- T2 palette consistency over the full orbit ------------------------
    exposure = photo.palette.exposure_map(front["rgb"], front["mask"])
    t2_palette = {
        str(az): photo.palette.outlier_fraction(buf["rgb"], buf["mask"], exposure)
        for az, buf in buffers.items()
    }

    # ---- M1 mirror symmetry -------------------------------------------------
    m1 = {
        str(az): mirror_symmetry_iou(buffers[az]["mask"], buffers[-az]["mask"])
        for az in config.symmetry_pairs
    }

    # ---- M2 profile articulation (+-90 mirror-degenerate: one measurement) --
    m2 = {
        "90": profile_articulation(
            buffers[90.0]["mask"], config.face_band,
            prominence_floor=config.articulation_prominence,
        )
    }

    # ---- M3 cavity mass ------------------------------------------------------
    m3 = {
        str(az): cavity_fraction(
            buffers[az]["clay_shade"], buffers[az]["mask"], config.face_band,
            depth=config.cavity_depth, sigma=config.cavity_sigma,
        )
        for az in config.cavity_azimuths
    }

    # ---- M4 silhouette floor (diagnostic + out-of-family gate) --------------
    sil = registered_silhouette_iou(photo.mask, front["mask"])

    measures = {
        "texture": {
            "source_structure": _measure(t1_structure, config.tex_structure_max, "max"),
            "source_chroma": _measure(t1_chroma, config.tex_chroma_max, "max"),
            "palette_outliers": _measure(t2_palette, config.tex_palette_max, "max"),
        },
        "mesh": {
            "mirror_symmetry": _measure(m1, config.mesh_symmetry_iou_min, "min"),
            "profile_articulation": _measure(m2, config.mesh_articulation_min, "min"),
            "cavity_mass": _measure(m3, config.mesh_cavity_max, "max"),
            "silhouette_floor": _measure(
                {"0": sil["iou"]}, config.mesh_silhouette_floor, "min"
            ),
        },
    }
    return {
        "model": glb_path.parent.name if glb_path.name == "scene.glb" else glb_path.stem,
        "glb": str(glb_path),
        "photo_mask_source": photo.mask_source,
        "row_alignment": {
            "scale": round(row_scale, 4), "shift": round(row_shift, 4),
            "ncc": round(row_ncc, 4),
        },
        "silhouette_registration": {
            k: (round(v, 4) if isinstance(v, float) else v) for k, v in sil.items()
        },
        "exposure_gain": [round(v, 4) for v in exposure],
        "measures": measures,
        "verdict": aggregate_verdict(measures),
    }
