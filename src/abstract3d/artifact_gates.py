"""Measured artifact-detector battery for baked textures.

Every detector in this module targets an artifact class this project has
actually shipped, and every constant carries the margins measured on the
preserved corpus (calibration program 2026-07-14, /tmp/afix2/report.md;
good corpus = owl standards, face proof, starship, chair, portrait,
sportscar_v7, gfix3 car_a/car_b, car_bo3_refs minus its roof-block
region; bad exemplars per class below). The doctrine, from the fix-gate
program: an axis may VOTE (refuse a bake) only where it has ZERO false
fires on the good corpus with a real margin; anything weaker ships as a
recorded measurement plus a loud warning, never a refusal.

Artifact classes and where each is caught:

1a IMAGE-IN-IMAGE (duplicated photo stamp): a miniature of the whole
   source photo frame - background included - baked onto the surface
   (sportscar_v7_candidate hood, feature-fringe-repair incident
   2026-07). The stamp lands WARPED (its registration was degenerate by
   definition), which defeats rigid template matching: five measured
   variants (raw and gradient-domain full-frame NCC, part-tile
   consistency voting, own-scale-excluded search, texture- and
   render-space) all failed to separate - the legitimate bake IS a
   photo copy at its own scale, texture space repeats subject content
   in every UV island, and near-flat tiles or 1D stripes produce
   degenerate high-NCC matches (all dead ends recorded in the report).
   What survives every warp is the stamp's PAYLOAD: photo-background
   colored / foreign pale matter on the subject -> the `pale_blotch`
   detector (voting, A/B) carries this class. Measured on the rebuilt
   incident pair: added max-component 0.0244 vs worst legitimate pair
   delta 0.0036 (6.8x apart); budget 0.009 sits 2.5x above the accepts
   and 2.7x under the incident.
1b RECTANGULAR PATCH-BLOCK GRIDS (low-res latent blocks over roof/glass,
   car_bo3_refs live defect): translucent axis-aligned rectangle cells
   whose interiors continue the underlying content. `rect_haze_cells`
   measures them; it CANNOT vote: the good corpus reaches 1 cell by
   noise (portrait, chairs, car_a) vs the defect's 2, and the pinned
   car_bo3 pair verdict is ACCEPT (the hue program shipped it with the
   blocks as a known cosmetic residual owned by the fringe-repair
   line). Ships as a recorded warning at >= 2 cells.
2  WHITE / FOREIGN RIM BLOTCHES at coverage edges (diagsub_* renders,
   maxvis-era candidates): compact bright desaturated blobs vs the
   local surface context -> `pale_blotch` (same detector as 1a; the
   stamp's background ring IS a rim blotch photometrically).
3  FILL MOTTLE / dark patchwork (sportscar_v5/v6/v7_maxvis, the 1024
   fill-cap class): `dark_patchwork` measures the mid-band dark+desat
   camouflage, but the labeled-good v7 (2048 baseline, 88% fill)
   measures 0.0926-0.1015 on the SAME statistic as the bad maxvis
   (0.0953) - the class is a property of every low-coverage baseline
   and differs only in degree, so no zero-false-fire threshold exists
   in render space. Recorded always; the deterministic CAUSE is gated
   instead: `fill_cap_mottle_risk` warns when the bake stats show the
   fill-detail energy calibration pegged at its 3.0 hard cap on a
   majority-fill subject (measured: maxvis scale 3.0 / unobservable
   0.839 vs v7@2048 scale 1.109 - the parity audit's root cause).
4  HANDOFF SEAM STEPS (mis-toned view handoffs): owned by the voting
   composition tone/hue axes in `bake_acceptance` (directional low-band
   damage). The battery adds NO detector for this class; the
   calibration proves non-interference (all 12 fix1 fixture verdicts
   unchanged with the battery active).
5  BAKED SPECULAR STREAKS / clear-coat smears (v4-v6 hood fog, v6
   milky wash): `pale_wash` measures broad-scale chroma collapse with
   luminance lift vs local context. It cannot vote: the source photo's
   own baked speculars are the same physics (labeled-good v7 measures
   0.0454 at its photo-facing views vs v6's 0.0382 - overlapping), and
   the pinned-ACCEPT car_bo3 pair adds +0.0123 of wash (its roof
   blocks read as wash). Ships as a recorded warning (added > 0.010 in
   the A/B gate = 5.9x above the worst legitimate pair, and it fires
   exactly on the known car_bo3 defect; absolute > 0.055 standalone).
6  GHOST / MELTED texture at a wrong pose (registration collapse,
   sportscar_v4 at coverage 0.05): invisible to A/B (both sides share
   the registration) and already gated at ship time by
   `evaluate_single_view_bake`; `registration_floor_check` surfaces
   the same class standalone from bundle stats (v4: source coverage
   0.0498 / capture efficiency 0.166 vs good-corpus minima 0.1088 /
   0.294 - margins 1.09x/1.18x are too thin to fail on, so it warns).

A/B contract (the gate): the battery punishes only ADDED artifacts -
per view, candidate measurement minus baseline measurement, worst view
against the budget. Subject-intrinsic bright structure (headlights,
chrome, white sills) cancels in the difference; a candidate that
inherits the baseline's artifacts unchanged is not refused for them.

Standalone contract (texture_qa): no baseline exists, so every render
detector reports against absolute warn lines calibrated on the good
corpus; none of them fails the run (measured margins are 1.14-1.8x -
below the project's voting bar). The two stats-based checks (classes 3
and 6) warn from bundle metadata when present.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "photo_reference",
    "measure_render_artifacts",
    "evaluate_artifact_battery_ab",
    "evaluate_bundle_artifact_battery",
    "registration_floor_check",
    "fill_cap_mottle_risk",
    "ADDED_BLOTCH_BUDGET",
    "ADDED_WASH_WARN",
    "ADDED_RECT_CELLS_WARN",
    "STANDALONE_WARN_LINES",
]

# ---------------------------------------------------------------------------
# Calibration constants. Every value is measured; see the module docstring
# for the corpus and /tmp/afix2/report.md for the full per-item table.
# ---------------------------------------------------------------------------

# pale_blotch: compact foreign pale matter (classes 1a + 2).
BLOTCH_CONTEXT_SIGMA_RATIO = 0.06   # local-context scale (fraction of the
#   subject diagonal): big enough that a blotch does not dominate its own
#   context, small enough that body-scale shading does not read as contrast.
BLOTCH_MIN_DELTA_L = 18.0           # blob is brighter than its context by
#   more than any measured legitimate local brightening of the good corpus
#   interiors at this scale (specular grain decorrelates below it).
BLOTCH_MIN_DELTA_CHROMA = 18.0      # and desaturated against its context by
#   the same amount - the "foreign matter" signature; keeps dark glass,
#   saturated trim and shading quiet. With (18, 18) the good corpus worst
#   view measures 0.0162 total / 0.0110 max-component (v7 white sills,
#   car_b headlight lens) vs the stamp incident 0.0350 / 0.0279.
BLOTCH_MIN_AREA_RATIO = 0.0015      # component floor (fraction of subject
#   pixels): speckle and highlight grain never aggregate.
BLOTCH_MAX_ELONGATION = 6.0         # principal-axis ratio (sqrt of the
#   PCA eigenvalue ratio): the artifact class is a compact-to-oval BLOB
#   (measured: stamp payload components 1.6-5.6, diagsub splashes
#   1.0-4.7), while the brightest LEGITIMATE bright-desat features are
#   strips (fx_car's reference-painted tail-light band 17.8, crisp
#   contour highlights 8.1-10.8, car_b trim 14.9). Without this cut the
#   fx_car ACCEPT pair measures +0.0077 of added "blotch" from its own
#   tail-light band and the voting budget loses its accept margin; with
#   it the pair drops to +0.0011 while every measured bad-class
#   component survives. Strips are still recorded (`elongated_frac`),
#   they just cannot vote.

# pale_wash: broad desaturating lift (class 5).
WASH_CONTEXT_SIGMA_RATIO = 0.14     # the wash is body-panel scale; its
#   context must be broader still.
WASH_LOCAL_SIGMA_RATIO = 0.02       # the compared field is itself smoothed
#   so texture grain does not vote.
WASH_MIN_DELTA_L = 6.0              # milky overlays lift L only mildly;
WASH_MIN_DELTA_CHROMA = 16.0        # the chroma collapse is the signature.

# rect_haze_cells: translucent axis-aligned rectangle cells (class 1b).
HAZE_CONTEXT_SIGMA_RATIO = 0.05
HAZE_MIN_DELTA_L = 1.5              # the cells are subtle (2-8 L offsets);
HAZE_MIN_DELTA_CHROMA = 3.0         # they desaturate what they overlay.
HAZE_MIN_SIDE_RATIO = 0.02          # cell side band (fraction of subject
HAZE_MAX_SIDE_RATIO = 0.25          # diagonal): latent-block scale.
HAZE_RECT_FILL_MIN = 0.66           # component must fill its axis-aligned
#   bounding box - organic mottle blobs do not.

# dark_patchwork: mid-band dark+desat camouflage (class 3).
PATCH_BAND_LO_RATIO = 0.008         # bandpass between mottle-cell scale
PATCH_BAND_HI_RATIO = 0.024         # and panel scale.
PATCH_MIN_DARK_L = 6.0
PATCH_MIN_DARK_CHROMA = 6.0

# --- A/B budgets (the gate). Measured on 18 baseline/candidate pairs:
# the 12 fix1 fixtures, the hue1 car_bo3 (pinned ACCEPT) and car_final
# pairs, the three fix3 chroma-rotation pairs, the hue15 probe, and the
# rebuilt sportscar_v7_candidate stamp incident pair.
ADDED_BLOTCH_BUDGET = 0.009
#   Worst-view added max-blotch-component. Labeled-accept pairs measure
#   <= 0.0036 (fx_car / fx_car_recheck, a borderline-elongation
#   asymmetry at az-90_el50; 9 of 12 fix1 pairs measure exactly 0.0000,
#   the pinned-ACCEPT car_bo3 pair 0.0026 total) and the rebuilt stamp
#   incident measures 0.0244: the budget sits 2.5x above the worst
#   accept and 2.7x under the incident (geometric middle of a 6.8x
#   separation). VOTES.
ADDED_WASH_WARN = 0.010
#   Worst-view added wash fraction. Labeled-accept fix1 pairs <= 0.0017
#   (5.9x under); the pinned-ACCEPT car_bo3 pair measures 0.0123 (its
#   roof patch-blocks read as wash - the exact live defect the warning
#   should name), so this CANNOT vote without flipping a pinned verdict.
#   WARNS.
ADDED_RECT_CELLS_WARN = 2
#   Added rectangle-cell count. Good pairs reach +1 by noise (portrait,
#   chairs, car_a); the block defect measures 2 absolute cells. Integer
#   count with a 1-cell noise floor cannot carry a vote. WARNS.

# --- Standalone warn lines (texture_qa; worst turnaround view). Good
# corpus maxima in parentheses; margins are below the voting bar, so all
# of these warn and none fails the run.
STANDALONE_WARN_LINES: Dict[str, float] = {
    "blotch_total_frac": 0.021,     # good max 0.0162 (v7 az90_el50 sills);
    #   fires: v7_candidate 0.0350, sportscar_v5 0.0291, diagsub_None
    #   0.0285 / back 0.0243 / side_left 0.0277 (misses top_rear 0.0120,
    #   side_right 0.0086 - documented false-negatives of the absolute
    #   form; the A/B form catches their class by construction).
    "bg_blotch_frac": 0.020,        # photo-background-colored blobs; good
    #   max 0.0110 (car_b headlight lens is honestly backdrop-gray);
    #   fires: v7_candidate 0.0279 (the stamp's background ring).
    "wash_frac": 0.055,             # good max 0.0450 (v7's own baked photo
    #   speculars; half-res values drift <= 0.0004 from the full-res
    #   calibration); fires: maxvis 0.0703. v6 (0.0379) and the bo3 roof
    #   (0.0376) sit BELOW the good max: the absolute form measurably
    #   cannot catch them (documented miss; the A/B lane can).
    "rect_haze_cells": 2,           # good max 1 (noise floor).
    "patch_frac": 0.12,             # good max 0.1015 (car_b front intakes)
    #   OVERLAPS the bad exemplars (maxvis 0.0953): recorded, effectively
    #   never warns - kept as fleet-drift data; the class is gated by
    #   fill_cap_mottle_risk on the CAUSE instead.
}

# Class 6 floors (standalone; the ship-time gate lives in
# evaluate_single_view_bake with its own calibrated floors).
REGISTRATION_SOURCE_COVERAGE_FLOOR = 0.10   # v4 incident 0.0498; good-corpus
REGISTRATION_EFFICIENCY_FLOOR = 0.25        # min 0.1088 / 0.294 (v7). Both
#   floors must fail together (AND): margins 1.09x/1.18x are thin, hence
#   warn-tier.

# Class 3 cause check: the 1024 fill-cap mottle (parity audit tex1 E1/E2).
FILL_CAP_SCALE_MIN = 2.95           # the calibration cap is exactly 3.0
FILL_CAP_UNOBSERVABLE_MIN = 0.5     # majority-fill subjects only (maxvis
#   0.839 fires; the owl class at 0.17 fill never can).


def _lab_fields(image: Any) -> Tuple[Any, Any, Any]:
    """L, a, b float32 planes of a PIL image or HxWx3 uint8 array."""

    import numpy as np
    from skimage import color as skcolor

    if hasattr(image, "convert"):
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    else:
        rgb = np.asarray(image, dtype=np.float32)
        if rgb.max() > 1.0:
            rgb = rgb / 255.0
    lab = skcolor.rgb2lab(rgb).astype(np.float32)
    return lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]


def _render_foreground(image: Any) -> Any:
    """Foreground of an offscreen render (constant near-white background,
    same rule as `reference_generation.clay_silhouette`)."""

    import numpy as np

    if hasattr(image, "convert"):
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    else:
        rgb = np.asarray(image, dtype=np.float32)
        if rgb.max() > 1.0:
            rgb = rgb / 255.0
    background = np.array([0.95, 0.95, 0.93], dtype=np.float32)
    return np.abs(rgb - background[None, None, :]).max(axis=2) > 0.04


def _masked_smooth(field: Any, mask: Any, sigma: float) -> Any:
    """Gaussian low-pass that ignores non-mask pixels (normalized
    convolution) - background never bleeds into the subject band.

    Truncated at 3 sigma: the tail mass cancels to first order in the
    numerator/denominator ratio, and the battery's largest kernel
    (wash context, 0.14 x subject diagonal) dominates the per-view cost
    (measured: 1.6 s -> ~1.1 s per 512 px view; calibration values move
    at the 4th decimal, far inside every margin)."""

    import numpy as np
    from scipy import ndimage

    weights = mask.astype(np.float32)
    numerator = ndimage.gaussian_filter(field * weights, sigma, truncate=3.0)
    denominator = ndimage.gaussian_filter(weights, sigma, truncate=3.0)
    return numerator / np.maximum(denominator, 1e-6)


def _foreground_diagonal(mask: Any) -> float:
    import numpy as np

    rows, cols = np.nonzero(mask)
    if len(rows) == 0:
        return 0.0
    return float(np.hypot(int(rows.max()) - int(rows.min()),
                          int(cols.max()) - int(cols.min())))


def photo_reference(source_image: Any) -> Optional[Dict[str, Any]]:
    """Source-photo statistics the battery conditions on.

    - RGBA input (the gate's matted `source_rgba`): the subject mask is
      the alpha; the original background is unrecoverable (matting zeroes
      it), so `bg` is None and background classification is skipped.
    - RGB input (texture_qa's raw `input.png`): the border median is the
      background reference (the same rule that fixed the gray-backdrop
      mini-photo incident in feature-fringe repair), and the subject is
      everything beyond a 12 LAB-distance band around it.

    Returns None when no usable subject mass exists.
    """

    import numpy as np
    from scipy import ndimage

    L, A, B = _lab_fields(source_image)
    h, w = L.shape
    bg: Optional[Tuple[float, float, float]] = None
    if hasattr(source_image, "mode") and source_image.mode == "RGBA":
        alpha = np.asarray(source_image)[:, :, 3]
        subject = alpha > 128
    else:
        margin = max(3, int(0.02 * min(h, w)))
        border = np.zeros((h, w), bool)
        border[:margin, :] = border[-margin:, :] = True
        border[:, :margin] = border[:, -margin:] = True
        bg = (float(np.median(L[border])), float(np.median(A[border])),
              float(np.median(B[border])))
        distance = np.sqrt((L - bg[0]) ** 2 + (A - bg[1]) ** 2
                           + (B - bg[2]) ** 2)
        subject = distance > 12.0
    subject = ndimage.binary_erosion(subject, iterations=2)
    if int(subject.sum()) < 500:
        return None
    chroma = np.hypot(A, B)
    return {
        "subject_l_med": float(np.median(L[subject])),
        "subject_chroma_med": float(np.median(chroma[subject])),
        "bg": bg,
    }


def measure_render_artifacts(render: Any,
                             photo_ref: Optional[Dict[str, Any]] = None,
                             ) -> Dict[str, Any]:
    """Run every render-space detector on one view; raw measurements only
    (thresholding happens in the A/B or standalone evaluators).

    Deterministic, CPU-only (numpy/scipy/skimage), ~0.4 s at 512 px.
    """

    import numpy as np
    from scipy import ndimage

    L, A, B = _lab_fields(render)
    chroma = np.hypot(A, B)
    fg = _render_foreground(render)
    interior = ndimage.binary_erosion(fg, iterations=4)
    empty = {
        "pale_blotch": {"total_frac": 0.0, "max_component_frac": 0.0,
                        "count": 0, "bg_frac": None, "elongated_frac": 0.0},
        "pale_wash": {"frac": 0.0},
        "rect_haze": {"cells": 0, "cell_area_frac": 0.0},
        "dark_patchwork": {"patch_frac": 0.0},
        "measured": False,
    }
    if int(interior.sum()) < 500:
        return empty
    diag = _foreground_diagonal(fg)
    fg_px = float(max(int(fg.sum()), 1))

    # ---- pale_blotch (classes 1a + 2): compact bright desaturated blobs
    # against their local context.
    sigma_blotch = max(BLOTCH_CONTEXT_SIGMA_RATIO * diag, 4.0)
    l_ctx = _masked_smooth(L, fg, sigma_blotch)
    c_ctx = _masked_smooth(chroma, fg, sigma_blotch)
    blotch = ((L - l_ctx > BLOTCH_MIN_DELTA_L)
              & (c_ctx - chroma > BLOTCH_MIN_DELTA_CHROMA) & interior)
    labels, count = ndimage.label(blotch, structure=np.ones((3, 3), bool))
    total = 0.0
    max_component = 0.0
    elongated = 0.0
    kept = 0
    bg_total: Optional[float] = None
    bg = (photo_ref or {}).get("bg")
    if bg is not None:
        bg_total = 0.0
    for index in range(1, count + 1):
        component = labels == index
        area = int(component.sum())
        if area < BLOTCH_MIN_AREA_RATIO * fg_px:
            continue
        frac = area / fg_px
        rows_c, cols_c = np.nonzero(component)
        eigen = np.linalg.eigvalsh(np.cov(np.stack(
            [rows_c, cols_c]).astype(np.float64)))
        eigen = np.maximum(eigen, 1e-6)
        if float(np.sqrt(eigen[1] / eigen[0])) > BLOTCH_MAX_ELONGATION:
            # A strip, not a blob: tail-light bands, sills, chrome trim
            # (see BLOTCH_MAX_ELONGATION). Recorded, not counted.
            elongated += frac
            continue
        total += frac
        kept += 1
        max_component = max(max_component, frac)
        if bg is not None:
            distance = np.sqrt(
                (float(np.median(L[component])) - bg[0]) ** 2
                + (float(np.median(A[component])) - bg[1]) ** 2
                + (float(np.median(B[component])) - bg[2]) ** 2)
            # 16 LAB units: the photo backdrop as re-rendered through the
            # bake's tone chain and the preview shading stays inside this
            # band (measured on the stamp incident: median distance 9).
            if distance <= 16.0:
                bg_total += frac

    # ---- pale_wash (class 5): broad-scale desaturating lift. Both
    # compared fields are low-passed (>= 2% of the subject diagonal), so
    # the measure is computed on a 2x-subsampled grid: the area FRACTION
    # is resolution-invariant and the broad kernel (14% of the diagonal)
    # dominates the battery's per-view cost otherwise (measured drift
    # <= 0.0013 absolute on the calibration corpus, inside every margin;
    # per-view cost 1.6 s -> 0.7 s).
    L2, C2, fg2 = L[::2, ::2], chroma[::2, ::2], fg[::2, ::2]
    interior2 = interior[::2, ::2]
    sigma_wash = max(WASH_CONTEXT_SIGMA_RATIO * diag, 6.0) / 2.0
    sigma_local = max(WASH_LOCAL_SIGMA_RATIO * diag, 2.0) / 2.0
    l_local = _masked_smooth(L2, fg2, sigma_local)
    c_local = _masked_smooth(C2, fg2, sigma_local)
    l_broad = _masked_smooth(L2, fg2, sigma_wash)
    c_broad = _masked_smooth(C2, fg2, sigma_wash)
    wash = ((l_local - l_broad > WASH_MIN_DELTA_L)
            & (c_broad - c_local > WASH_MIN_DELTA_CHROMA) & interior2)
    wash_frac = float(wash.sum()) / float(max(int(fg2.sum()), 1))

    # ---- rect_haze (class 1b): translucent axis-aligned rectangle cells.
    sigma_haze = max(HAZE_CONTEXT_SIGMA_RATIO * diag, 4.0)
    l_res = L - _masked_smooth(L, fg, sigma_haze)
    c_res = chroma - _masked_smooth(chroma, fg, sigma_haze)
    haze = ((l_res > HAZE_MIN_DELTA_L) & (c_res < -HAZE_MIN_DELTA_CHROMA)
            & ndimage.binary_erosion(fg, iterations=3))
    haze_labels, haze_count = ndimage.label(
        haze, structure=np.ones((3, 3), bool))
    min_side = max(HAZE_MIN_SIDE_RATIO * diag, 7.0)
    max_side = HAZE_MAX_SIDE_RATIO * diag
    cells = 0
    cell_px = 0
    for sl in ndimage.find_objects(haze_labels):
        if sl is None:
            continue
        height = sl[0].stop - sl[0].start
        width = sl[1].stop - sl[1].start
        if (width < min_side or height < min_side
                or width > max_side or height > max_side):
            continue
        area = int(np.count_nonzero(haze_labels[sl]))
        if area < HAZE_RECT_FILL_MIN * width * height:
            continue
        cells += 1
        cell_px += area

    # ---- dark_patchwork (class 3): mid-band dark+desat camouflage.
    band_lo = max(PATCH_BAND_LO_RATIO * diag, 1.5)
    band_hi = max(PATCH_BAND_HI_RATIO * diag, 4.0)
    l_band = (_masked_smooth(L, fg, band_lo)
              - _masked_smooth(L, fg, band_hi))
    c_band = (_masked_smooth(chroma, fg, band_lo)
              - _masked_smooth(chroma, fg, band_hi))
    patches = ((l_band < -PATCH_MIN_DARK_L)
               & (c_band < -PATCH_MIN_DARK_CHROMA) & interior)

    return {
        "pale_blotch": {"total_frac": round(total, 5),
                        "max_component_frac": round(max_component, 5),
                        "count": kept,
                        "bg_frac": (round(bg_total, 5)
                                    if bg_total is not None else None),
                        "elongated_frac": round(elongated, 5)},
        "pale_wash": {"frac": round(wash_frac, 5)},
        "rect_haze": {"cells": cells,
                      "cell_area_frac": round(cell_px / fg_px, 5)},
        "dark_patchwork": {
            "patch_frac": round(float(patches.sum()) / fg_px, 5)},
        "measured": True,
    }


def evaluate_artifact_battery_ab(
    candidate_renders: Sequence[Tuple[str, Any]],
    baseline_renders: Sequence[Tuple[str, Any]],
    *,
    photo_ref: Optional[Dict[str, Any]] = None,
    added_blotch_budget: float = ADDED_BLOTCH_BUDGET,
    added_wash_warn: float = ADDED_WASH_WARN,
    added_rect_cells_warn: int = ADDED_RECT_CELLS_WARN,
) -> Dict[str, Any]:
    """A/B battery over a shared turnaround render set.

    Directional non-regression, same doctrine as the tone axes: only
    ADDED artifact mass counts (per view candidate minus baseline, worst
    view vs budget), so subject-intrinsic bright structure and inherited
    baseline damage never refuse a candidate.

    Returns `{"reasons": [...], "warnings": [...], "metrics": {...}}`;
    reasons refuse, warnings are recorded loudly, metrics carry every
    per-detector worst delta with its budget and vote flag.
    """

    worst = {
        "blotch": (0.0, None),   # added max-component
        "wash": (0.0, None),
        "cells": (0, None),
        "patch": (0.0, None),
    }
    measured_views = 0
    for (label, candidate), (_, baseline) in zip(
            candidate_renders, baseline_renders):
        cand = measure_render_artifacts(candidate, photo_ref)
        base = measure_render_artifacts(baseline, photo_ref)
        if not (cand["measured"] and base["measured"]):
            continue
        measured_views += 1
        deltas = {
            "blotch": (cand["pale_blotch"]["max_component_frac"]
                       - base["pale_blotch"]["max_component_frac"]),
            "wash": cand["pale_wash"]["frac"] - base["pale_wash"]["frac"],
            "cells": cand["rect_haze"]["cells"] - base["rect_haze"]["cells"],
            "patch": (cand["dark_patchwork"]["patch_frac"]
                      - base["dark_patchwork"]["patch_frac"]),
        }
        for key, value in deltas.items():
            if value > worst[key][0]:
                worst[key] = (value, label)

    reasons: List[str] = []
    warnings: List[str] = []
    if measured_views and worst["blotch"][0] > float(added_blotch_budget):
        reasons.append(
            f"artifact battery (pale blotch): candidate adds a foreign "
            f"bright desaturated component of {worst['blotch'][0]:.4f} of "
            f"the subject at {worst['blotch'][1]} (budget "
            f"{added_blotch_budget}; the image-in-image stamp class "
            f"measures 0.0244, legitimate reference work <= 0.0036)")
    if measured_views and worst["wash"][0] > float(added_wash_warn):
        warnings.append(
            f"artifact battery (pale wash, non-voting): candidate adds "
            f"{worst['wash'][0]:.4f} of broad desaturating lift at "
            f"{worst['wash'][1]} (warn line {added_wash_warn}; baked "
            f"specular / patch-block class - fix1-accept pairs measure "
            f"<= 0.0017)")
    if measured_views and worst["cells"][0] >= int(added_rect_cells_warn):
        warnings.append(
            f"artifact battery (rect haze cells, non-voting): candidate "
            f"adds {worst['cells'][0]} translucent axis-aligned rectangle "
            f"cells at {worst['cells'][1]} (patch-block grid class)")

    metrics = {
        "added_pale_blotch": {
            "worst": round(worst["blotch"][0], 5),
            "worst_view": worst["blotch"][1],
            "max_allowed": float(added_blotch_budget),
            "votes": True},
        "added_pale_wash": {
            "worst": round(worst["wash"][0], 5),
            "worst_view": worst["wash"][1],
            "warn_line": float(added_wash_warn),
            "votes": False},
        "added_rect_haze_cells": {
            "worst": int(worst["cells"][0]),
            "worst_view": worst["cells"][1],
            "warn_line": int(added_rect_cells_warn),
            "votes": False},
        "added_dark_patchwork": {
            "worst": round(worst["patch"][0], 5),
            "worst_view": worst["patch"][1],
            # Legitimate references add up to +0.0629 of mid-band
            # structure (fx_car: real glass frames over blank fill), far
            # above any damage exemplar - recorded for drift only.
            "votes": False},
        "measured_views": measured_views,
        "photo_background_available": bool((photo_ref or {}).get("bg")),
    }
    return {"reasons": reasons, "warnings": warnings, "metrics": metrics}


def evaluate_bundle_artifact_battery(
    views: Sequence[Tuple[str, Any]],
    *,
    photo_ref: Optional[Dict[str, Any]] = None,
    stats: Optional[Dict[str, Any]] = None,
    warn_lines: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Standalone battery for a single bake (no baseline): worst-view
    absolute measurements against the good-corpus warn lines, plus the
    stats-based class 3 / class 6 checks when bundle stats are given.

    Nothing here fails a run - the measured absolute margins (1.14-1.8x)
    are below the project's voting bar; see the module docstring.
    """

    lines = dict(STANDALONE_WARN_LINES)
    if warn_lines:
        lines.update(warn_lines)

    per_view: Dict[str, Any] = {}
    worst = {"blotch_total_frac": (0.0, None), "bg_blotch_frac": (0.0, None),
             "max_component_frac": (0.0, None), "wash_frac": (0.0, None),
             "rect_haze_cells": (0, None), "patch_frac": (0.0, None)}
    for label, render in views:
        row = measure_render_artifacts(render, photo_ref)
        per_view[label] = row
        if not row["measured"]:
            continue
        candidates = {
            "blotch_total_frac": row["pale_blotch"]["total_frac"],
            "bg_blotch_frac": row["pale_blotch"]["bg_frac"] or 0.0,
            "max_component_frac": row["pale_blotch"]["max_component_frac"],
            "wash_frac": row["pale_wash"]["frac"],
            "rect_haze_cells": row["rect_haze"]["cells"],
            "patch_frac": row["dark_patchwork"]["patch_frac"],
        }
        for key, value in candidates.items():
            if value > worst[key][0]:
                worst[key] = (value, label)

    warnings: List[str] = []
    for key in ("blotch_total_frac", "bg_blotch_frac", "wash_frac",
                "rect_haze_cells"):
        value, view = worst[key]
        line = lines[key]
        if value >= line and view is not None:
            warnings.append(
                f"artifact battery ({key}): worst view {view} measures "
                f"{value:.4f} vs warn line {line} (good-corpus "
                f"calibration, /tmp/afix2 program)"
                if not isinstance(value, int) else
                f"artifact battery ({key}): worst view {view} measures "
                f"{value} vs warn line {int(line)}")

    checks: Dict[str, Any] = {}
    if stats is not None:
        checks["registration_floors"] = registration_floor_check(stats)
        if checks["registration_floors"]["fired"]:
            checks_row = checks["registration_floors"]
            warnings.append(
                "artifact battery (registration floors): source coverage "
                f"{checks_row['source_coverage']} and capture efficiency "
                f"{checks_row['capture_efficiency']} both under their "
                "floors - the ghost/melted wrong-pose class "
                "(sportscar_v4: 0.0498 / 0.166)")
        checks["fill_cap_mottle_risk"] = fill_cap_mottle_risk(stats)
        if checks["fill_cap_mottle_risk"]["fired"]:
            row = checks["fill_cap_mottle_risk"]
            warnings.append(
                "artifact battery (fill-cap mottle risk): fill-detail "
                f"scale {row['fill_scale']} at the calibration cap on a "
                f"{row['unobservable_ratio']:.0%}-fill subject - the 1024 "
                "fill-cap patchwork class (parity audit: 5.8x darker "
                "patches than the same bake at 2048)")

    return {
        "warnings": warnings,
        "worst": {key: {"value": value, "view": view}
                  for key, (value, view) in worst.items()},
        "warn_lines": lines,
        "checks": checks,
        "per_view": per_view,
    }


def registration_floor_check(stats: Dict[str, Any]) -> Dict[str, Any]:
    """Class 6 (ghost/melted wrong-pose bake), from the bake's own stats.

    Fires only when BOTH the source view's coverage and its capture
    efficiency sit under their floors (the incident measures 0.0498 /
    0.166; the weakest good-corpus bundle - sportscar_v7 - measures
    0.1088 / 0.294, so single floors would carry 1.09x/1.18x margins:
    too thin for anything but a joint warning). Missing stats (older
    bundles have no capture_efficiency) disable the check honestly.
    """

    rows = (stats or {}).get("observed_view_stats") or []
    source = next(
        (row for row in rows if row.get("index") == 1
         or row.get("label") in ("front", "source", "view_01")),
        rows[0] if rows else None)
    coverage = (source or {}).get("coverage_ratio")
    efficiency = (source or {}).get("capture_efficiency")
    available = coverage is not None and efficiency is not None
    fired = bool(
        available
        and float(coverage) < REGISTRATION_SOURCE_COVERAGE_FLOOR
        and float(efficiency) < REGISTRATION_EFFICIENCY_FLOOR)
    return {"available": available, "fired": fired,
            "source_coverage": coverage, "capture_efficiency": efficiency,
            "coverage_floor": REGISTRATION_SOURCE_COVERAGE_FLOOR,
            "efficiency_floor": REGISTRATION_EFFICIENCY_FLOOR}


def fill_cap_mottle_risk(stats: Dict[str, Any]) -> Dict[str, Any]:
    """Class 3 cause check: fill-detail energy calibration pegged at its
    hard cap on a majority-fill subject (the measured mechanism of the
    1024 mottle: scale 3.0 / sigma 0.2075 vs 1.109 / 0.1291 at 2048)."""

    fill = (stats or {}).get("fill_detail") or {}
    calibration = fill.get("energy_calibration") or {}
    scale = calibration.get("scale")
    unobservable = ((stats or {}).get("leverage") or {}).get(
        "unobservable_ratio")
    available = scale is not None and unobservable is not None
    fired = bool(available and float(scale) >= FILL_CAP_SCALE_MIN
                 and float(unobservable) >= FILL_CAP_UNOBSERVABLE_MIN)
    return {"available": available, "fired": fired,
            "fill_scale": scale, "unobservable_ratio": unobservable,
            "scale_min": FILL_CAP_SCALE_MIN,
            "unobservable_min": FILL_CAP_UNOBSERVABLE_MIN}
