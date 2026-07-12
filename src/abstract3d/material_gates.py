"""Material-fidelity gates for generated reference views (critic gates G1-G5).

The results critic's audit proved the existing acceptance gates (silhouette
IoU, tone distance, texture QA) are structurally blind to MATERIAL IDENTITY
loss: a carved-wood subject regenerated as glazed ceramic passed every
shipped gate. These gates compare the generated view's foreground against
the SOURCE PHOTO's foreground as material evidence, per tile rather than
globally (global averages are how "cream base + brown lines" impersonated
"mid-brown wood").

Correspondence problem: the unseen side's layout legitimately differs from
the source's, so tiles are matched by nearest MATERIAL (median LAB), not by
position. A tile must resemble SOME source material region; its
microstructure requirement follows the matched region's.

All gates return measurements plus pass/fail so callers can log every
number; `evaluate_material_fidelity` aggregates them into one verdict.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

TILE_GRID = 8
MIN_TILE_FOREGROUND = 0.35


def _foreground_tiles(image_rgba: Any, grid: int = TILE_GRID) -> Tuple[Any, Any, Any]:
    """Per-tile (median LAB, high-frequency energy, validity) over the matte."""

    import cv2
    import numpy as np
    from skimage import color as skcolor

    rgba = np.asarray(image_rgba.convert("RGBA"), dtype=np.float32) / 255.0
    mask = rgba[:, :, 3] > 0.5
    lab = skcolor.rgb2lab(rgba[:, :, :3]).astype(np.float32)
    lum = (0.299 * rgba[:, :, 0] + 0.587 * rgba[:, :, 1]
           + 0.114 * rgba[:, :, 2]).astype(np.float32)
    gx = cv2.Scharr(lum, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(lum, cv2.CV_32F, 0, 1)
    magnitude = np.hypot(gx, gy)

    height, width = mask.shape
    tile_h, tile_w = height // grid, width // grid
    medians: List[Any] = []
    energies: List[float] = []
    valid: List[bool] = []
    for row in range(grid):
        for col in range(grid):
            tile_mask = mask[row * tile_h:(row + 1) * tile_h,
                             col * tile_w:(col + 1) * tile_w]
            coverage = float(tile_mask.mean()) if tile_mask.size else 0.0
            if coverage < MIN_TILE_FOREGROUND:
                medians.append(None)
                energies.append(0.0)
                valid.append(False)
                continue
            tile_lab = lab[row * tile_h:(row + 1) * tile_h,
                           col * tile_w:(col + 1) * tile_w][tile_mask]
            tile_energy = magnitude[row * tile_h:(row + 1) * tile_h,
                                    col * tile_w:(col + 1) * tile_w][tile_mask]
            medians.append(np.median(tile_lab, axis=0))
            energies.append(float(tile_energy.mean()))
            valid.append(True)
    return medians, np.asarray(energies, dtype=np.float32), np.asarray(valid, dtype=bool)


def _foreground_lightness(image_rgba: Any) -> Any:
    import numpy as np
    from skimage import color as skcolor

    rgba = np.asarray(image_rgba.convert("RGBA"), dtype=np.float32) / 255.0
    mask = rgba[:, :, 3] > 0.5
    lab = skcolor.rgb2lab(rgba[:, :, :3])
    return lab[:, :, 0][mask]


def gate_tile_albedo(generated: Any, source: Any,
                     *, max_tile_delta: float = 20.0,
                     max_median_delta: float = 12.0,
                     min_pass_fraction: float = 0.9) -> Dict[str, Any]:
    """G1: every generated tile must resemble SOME source material tile."""

    import numpy as np

    gen_medians, _, gen_valid = _foreground_tiles(generated)
    src_medians, _, src_valid = _foreground_tiles(source)
    src_stack = np.stack([m for m, ok in zip(src_medians, src_valid) if ok])
    deltas: List[float] = []
    for median, ok in zip(gen_medians, gen_valid):
        if not ok:
            continue
        deltas.append(float(np.linalg.norm(src_stack - median[None, :], axis=1).min()))
    deltas_arr = np.asarray(deltas, dtype=np.float32)
    if deltas_arr.size == 0:
        return {"name": "G1_tile_albedo", "passed": False, "reason": "no tiles"}
    pass_fraction = float((deltas_arr <= max_tile_delta).mean())
    median_delta = float(np.median(deltas_arr))
    return {
        "name": "G1_tile_albedo",
        "passed": pass_fraction >= min_pass_fraction and median_delta <= max_median_delta,
        "tile_pass_fraction": round(pass_fraction, 3),
        "median_delta_e": round(median_delta, 2),
        "worst_delta_e": round(float(deltas_arr.max()), 2),
    }


def gate_baked_speculars(generated: Any,
                         *, lightness_delta: float = 60.0,
                         max_blob_fraction: float = 0.005) -> Dict[str, Any]:
    """G2: no large near-white blobs above the foreground's own median."""

    import numpy as np
    from scipy.ndimage import label as cc_label
    from skimage import color as skcolor

    rgba = np.asarray(generated.convert("RGBA"), dtype=np.float32) / 255.0
    mask = rgba[:, :, 3] > 0.5
    lab = skcolor.rgb2lab(rgba[:, :, :3])
    lightness = lab[:, :, 0]
    if not mask.any():
        return {"name": "G2_baked_speculars", "passed": False, "reason": "empty matte"}
    median_l = float(np.median(lightness[mask]))
    hot = mask & (lightness > median_l + lightness_delta)
    labels, count = cc_label(hot)
    foreground = int(mask.sum())
    worst = 0.0
    for index in range(1, count + 1):
        worst = max(worst, float((labels == index).sum()) / foreground)
    return {
        "name": "G2_baked_speculars",
        "passed": worst <= max_blob_fraction,
        "worst_blob_fraction": round(worst, 5),
        "hot_components": count,
        "median_lightness": round(median_l, 1),
    }


def gate_tile_microstructure(generated: Any, source: Any,
                             *, min_energy_ratio: float = 0.5,
                             min_pass_fraction: float = 0.9,
                             smooth_source_quantile: float = 0.3) -> Dict[str, Any]:
    """G3: tiles matched to TEXTURED source material must carry texture.

    Tiles whose nearest source material is itself smooth (below the source's
    own low-energy quantile) are exempt — a porcelain vase must not be
    rejected for generating smooth porcelain.
    """

    import numpy as np

    gen_medians, gen_energy, gen_valid = _foreground_tiles(generated)
    src_medians, src_energy, src_valid = _foreground_tiles(source)
    src_stack = np.stack([m for m, ok in zip(src_medians, src_valid) if ok])
    src_energies = src_energy[src_valid]
    smooth_floor = float(np.quantile(src_energies, smooth_source_quantile))

    checked = 0
    passed = 0
    ratios: List[float] = []
    for median, energy, ok in zip(gen_medians, gen_energy, gen_valid):
        if not ok:
            continue
        nearest = int(np.linalg.norm(src_stack - median[None, :], axis=1).argmin())
        matched_energy = float(src_energies[nearest])
        if matched_energy <= smooth_floor:
            continue  # matched material is legitimately smooth
        checked += 1
        ratio = energy / max(matched_energy, 1e-6)
        ratios.append(ratio)
        if ratio >= min_energy_ratio:
            passed += 1
    if checked == 0:
        return {"name": "G3_tile_microstructure", "passed": True,
                "reason": "source has no textured tiles", "checked_tiles": 0}
    fraction = passed / checked
    return {
        "name": "G3_tile_microstructure",
        "passed": fraction >= min_pass_fraction,
        "tile_pass_fraction": round(fraction, 3),
        "checked_tiles": checked,
        "median_energy_ratio": round(float(np.median(ratios)), 3),
    }


def gate_forbidden_modes(generated: Any, source: Any,
                         *, max_outside_fraction: float = 0.05) -> Dict[str, Any]:
    """G5: generated lightness must live inside the source's L* range."""

    import numpy as np

    src_l = _foreground_lightness(source)
    gen_l = _foreground_lightness(generated)
    if src_l.size == 0 or gen_l.size == 0:
        return {"name": "G5_forbidden_modes", "passed": False, "reason": "empty matte"}
    low, high = np.percentile(src_l, [1, 99])
    outside = float(((gen_l < low) | (gen_l > high)).mean())
    return {
        "name": "G5_forbidden_modes",
        "passed": outside <= max_outside_fraction,
        "outside_fraction": round(outside, 4),
        "source_l_range": [round(float(low), 1), round(float(high), 1)],
    }


def gate_cross_view_coherence(views: Sequence[Any],
                              *, max_pairwise_delta: float = 8.0) -> Dict[str, Any]:
    """G4: accepted views must agree on the base tone with each other."""

    import numpy as np
    from skimage import color as skcolor

    medians = []
    for view in views:
        rgba = np.asarray(view.convert("RGBA"), dtype=np.float32) / 255.0
        mask = rgba[:, :, 3] > 0.5
        if not mask.any():
            continue
        lab = skcolor.rgb2lab(rgba[:, :, :3])
        medians.append(np.median(lab[mask], axis=0))
    if len(medians) < 2:
        return {"name": "G4_cross_view_coherence", "passed": True,
                "reason": "fewer than two views"}
    worst = 0.0
    for i in range(len(medians)):
        for j in range(i + 1, len(medians)):
            worst = max(worst, float(np.linalg.norm(medians[i] - medians[j])))
    return {
        "name": "G4_cross_view_coherence",
        "passed": worst <= max_pairwise_delta,
        "worst_pairwise_delta_e": round(worst, 2),
    }


def texture_fidelity(
    generated_rgba: Any,
    source_rgba: Any,
    *,
    relief_ratio_min: float = 0.90,
    flat_delta_max: float = 0.12,
    floor_relief_ratio: float = 0.65,
    floor_flat_delta: float = 0.20,
    smooth_source_s50: float = 2.0,
) -> Dict[str, Any]:
    """Band-pass texture-identity gate (adversary-calibrated on the shipped
    failure set: both real material-loss failures caught, zero false
    rejections including a legitimately plainer ship underside).

    Measures the 2-8 px relief/grain band (G(L,1) - G(L,3)) in tiles:
    `relief_ratio` = generation's 90th-percentile tile RMS over the source's
    MEDIAN tile RMS — "does the generation's best-textured decile anywhere
    reach the source's typical texture" — layout-invariant, one-sided
    (excess texture never rejects). `flat_delta` = growth in the fraction of
    near-flat pixels, catching partial glazing. Sources that are themselves
    smooth (S50 below `smooth_source_s50`) auto-pass: a porcelain vase must
    generate smooth. Thresholds are provisional (calibrated on n=11 views);
    both metrics are always reported for recalibration.
    """

    import numpy as np
    from PIL import Image
    from scipy.ndimage import binary_erosion, gaussian_filter

    def prepare(image_rgba: Any) -> Optional[Tuple[Any, Any]]:
        rgba = np.asarray(image_rgba.convert("RGBA"), dtype=np.float32) / 255.0
        mask = rgba[:, :, 3] > 0.5
        if not mask.any():
            return None
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        r0, r1 = np.argmax(rows), len(rows) - np.argmax(rows[::-1])
        c0, c1 = np.argmax(cols), len(cols) - np.argmax(cols[::-1])
        crop = rgba[r0:r1, c0:c1]
        height, width = crop.shape[:2]
        scale = 512.0 / max(height, width)
        new_size = (max(24, int(round(width * scale))),
                    max(24, int(round(height * scale))))
        resized = np.asarray(
            Image.fromarray((crop * 255).astype(np.uint8), "RGBA").resize(
                new_size, Image.LANCZOS), dtype=np.float32) / 255.0
        mask_r = resized[:, :, 3] > 0.5
        erode_px = max(2, int(round(0.03 * max(new_size))))
        mask_r = binary_erosion(mask_r, iterations=erode_px)
        if not mask_r.any():
            return None
        lum = (0.2126 * resized[:, :, 0] + 0.7152 * resized[:, :, 1]
               + 0.0722 * resized[:, :, 2]) * 100.0
        # Mask-normalized Gaussians: transparent padding (black) must not
        # bleed into the band statistics near the silhouette.
        weight = (resized[:, :, 3] > 0.5).astype(np.float32)

        def masked_blur(sigma: float) -> Any:
            numerator = gaussian_filter(lum * weight, sigma)
            denominator = gaussian_filter(weight, sigma)
            return numerator / np.maximum(denominator, 1e-6)

        band = masked_blur(1.0) - masked_blur(3.0)
        return band, mask_r

    def tile_rms(band: Any, mask: Any) -> Any:
        import numpy as np

        tile = 24
        values = []
        for row in range(0, band.shape[0] - tile + 1, tile):
            for col in range(0, band.shape[1] - tile + 1, tile):
                tile_mask = mask[row:row + tile, col:col + tile]
                if float(tile_mask.mean()) < 0.85:
                    continue
                patch = band[row:row + tile, col:col + tile][tile_mask]
                values.append(float(np.sqrt(np.mean(patch ** 2))))
        return np.asarray(values, dtype=np.float32)

    src = prepare(source_rgba)
    gen = prepare(generated_rgba)
    if src is None or gen is None:
        return {"name": "texture_fidelity", "passed": True, "floor": True,
                "reason": "empty matte", "measured": None}
    src_band, src_mask = src
    gen_band, gen_mask = gen
    src_tiles = tile_rms(src_band, src_mask)
    gen_tiles = tile_rms(gen_band, gen_mask)
    import numpy as np

    if len(src_tiles) < 12 or len(gen_tiles) < 12:
        return {"name": "texture_fidelity", "passed": True, "floor": True,
                "reason": f"too few tiles (src {len(src_tiles)}, gen {len(gen_tiles)})"}
    s50 = float(np.median(src_tiles))
    result: Dict[str, Any] = {"name": "texture_fidelity", "s50": round(s50, 3)}
    if s50 < float(smooth_source_s50):
        result.update(passed=True, reason="smooth source (auto-pass)")
        return result
    g90 = float(np.percentile(gen_tiles, 90))
    flat_src = float((np.abs(src_band[src_mask]) < 1.2).mean())
    flat_gen = float((np.abs(gen_band[gen_mask]) < 1.2).mean())
    relief_ratio = g90 / max(s50, 1e-6)
    flat_delta = flat_gen - flat_src
    result.update(
        relief_ratio=round(relief_ratio, 3),
        flat_delta=round(flat_delta, 3),
        passed=bool(relief_ratio >= relief_ratio_min and flat_delta <= flat_delta_max),
        floor=bool(relief_ratio >= floor_relief_ratio and flat_delta <= floor_flat_delta),
        selection_score=round(relief_ratio - 2.0 * max(0.0, flat_delta), 3),
    )
    return result


def part_material_fidelity(
    generated_rgba: Any,
    source_rgba: Any,
    *,
    clusters: int = 3,
    strict_delta_e: float = 12.0,
    floor_delta_e: float = 16.0,
    min_part_weight: float = 0.10,
    lightness_tolerance: float = 25.0,
) -> Dict[str, Any]:
    """G6: part-aware material identity. Every MAJOR color-part of the
    generation must exist in the source's part palette.

    The band-pass texture gate is blind to material flips that keep relief
    plausible (upholstery regenerated as camouflage mottling), and
    tile-matching gates forgive them because SOME source tile is always
    close. Clustering foreground LAB into a small part palette and scoring
    the WORST major generated part's distance to its nearest source part
    catches them. Distance is CHROMA-FIRST: a*/b* carry material hue
    identity, while L carries shading, which legitimately differs on an
    unseen side — L contributes only beyond a tolerance band (measured: the
    full-LAB variant scored an accepted profile view (15.7) above a
    confirmed camo flip's margin and would have rejected every portrait
    view for shading alone). On the v1+v2 critic-labeled set the ab-metric
    separates palette flips (chair camo 22.3, owl ceramic 9.6-10.9) from
    all confirmed passes (<= 9.6) when combined with the texture gate,
    which independently catches the ceramic pair via flat_delta.

    KNOWN BLIND SPOT (documented, not hidden): semantic re-rendering that
    keeps the palette AND relief energy — v1's "sculpted goo" hair scored
    6.7-7.4 here, indistinguishable from real hair. No foreground statistic
    tested separates it (palette, per-part relief, shine fraction all
    overlap); the countermeasure is generator quality, not gating.
    """

    import numpy as np
    from scipy.cluster.vq import kmeans2
    from skimage import color as skcolor

    def palette(image_rgba: Any, cluster_count: int,
                restart_seed: int) -> List[Dict[str, Any]]:
        rgba = np.asarray(image_rgba.convert("RGBA"), dtype=np.float32) / 255.0
        mask = rgba[:, :, 3] > 0.5
        lab = skcolor.rgb2lab(rgba[:, :, :3]).astype(np.float32)[mask]
        if len(lab) == 0:
            return []
        if len(lab) > 40000:
            picks = np.random.default_rng(restart_seed).choice(
                len(lab), 40000, replace=False)
            lab = lab[picks]
        _, labels = kmeans2(lab, cluster_count, minit="++", seed=restart_seed)
        parts = []
        for index in range(cluster_count):
            member = lab[labels == index]
            if len(member) < len(lab) * 0.05:
                continue
            parts.append({"median": np.median(member, axis=0),
                          "weight": len(member) / len(lab)})
        return parts

    # CHROMA-COLLAPSE GUARD, dispersion form (measured on the sports-car
    # incident + adversarial recheck): a MONOCHROME generation of a
    # colorful subject previously strict-PASSED the forward-only part
    # test raw (gray hides within the lightness tolerance of any dark
    # source part; the gray car scored 11.72) and was only caught through
    # a tone-match interaction — luck, and `tone_match=False` is a public
    # kwarg. The guard measures chroma DISPERSION (std of the foreground
    # ab magnitudes), not chroma-of-mean: the mean form left a gray
    # PORTRAIT strict-passing the whole battery (skin's mean chroma sits
    # below its arming floor) and cancels complementary hues. Measured
    # separation: collapsed candidates 0.02-0.18 (gray portrait 0.06),
    # every accepted fleet view >= 0.53.
    def chroma_dispersion(image_rgba: Any) -> float:
        rgba = np.asarray(image_rgba.convert("RGBA"), dtype=np.float32) / 255.0
        mask = rgba[:, :, 3] > 0.5
        if not mask.any():
            return 0.0
        lab = skcolor.rgb2lab(rgba[:, :, :3]).astype(np.float32)
        chroma = np.hypot(lab[:, :, 1][mask], lab[:, :, 2][mask])
        return float(chroma.std())

    source_dispersion = chroma_dispersion(source_rgba)
    generated_dispersion = chroma_dispersion(generated_rgba)
    if source_dispersion >= 5.0 and (
            generated_dispersion / max(source_dispersion, 1e-6) < 0.35):
        return {"name": "G6_part_material", "passed": False, "floor": False,
                # A large constant, not None: the selection score treats
                # None as zero palette penalty, letting a collapsed
                # candidate outrank floor-band ones in the recorded
                # ranking (nothing ships, but the record misleads).
                "worst_part_delta_e": 40.0,
                "source_chroma_dispersion": round(source_dispersion, 2),
                "generated_chroma_dispersion": round(generated_dispersion, 2),
                "reason": "chroma collapse: generation is near-monochrome "
                          "for a chromatic subject"}

    # PART-CORRESPONDENCE ENSEMBLE (measured): a single-k clustering adds
    # 14-24 points of pure correspondence noise near boundaries — a
    # PERFECT copy plus one realistic windshield reflection scored 25.3
    # at k=3 (the reflection steals a centroid) and 1.75 under the
    # ensemble. The subject's true part count is unknown a priori, so the
    # gate scores the MINIMUM worst-part distance over k in {3,4,5}: a
    # true palette flip stays far from the source under every k (all
    # measured true failures remain above strict), while correspondence
    # artifacts collapse under at least one k.
    def worst_for_clusters(cluster_count: int, restart_seed: int) -> Optional[float]:
        source_parts = palette(source_rgba, cluster_count, restart_seed)
        generated_parts = palette(generated_rgba, cluster_count, restart_seed)
        if not source_parts or not generated_parts:
            return None
        worst = 0.0
        for part in generated_parts:
            if part["weight"] < float(min_part_weight):
                continue
            best = float("inf")
            for source_part in source_parts:
                chroma_distance = float(np.hypot(
                    part["median"][1] - source_part["median"][1],
                    part["median"][2] - source_part["median"][2]))
                lightness_excess = max(
                    0.0, abs(float(part["median"][0] - source_part["median"][0]))
                    - float(lightness_tolerance))
                best = min(best, chroma_distance + 0.5 * lightness_excess)
            worst = max(worst, best)
        return worst

    # RESTART CONSENSUS (measured): even the k-ensemble carries 3.15 dE
    # median RNG noise per candidate (up to 6.85) from the k-means draw —
    # a recorded 24.12 REJECT re-scored at 15.2 mean under resampling,
    # i.e. a verdict was condemned by clustering luck. The median over 5
    # fixed restarts cuts the noise to 1.13 dE at ~2s CPU against a
    # ~200s generation. Restart seeds are fixed so the gate stays
    # deterministic.
    restart_scores: List[float] = []
    for restart_seed in range(5):
        ensemble = [value for value in (
            worst_for_clusters(k, restart_seed)
            for k in (int(clusters), int(clusters) + 1, int(clusters) + 2))
            if value is not None]
        if ensemble:
            restart_scores.append(min(ensemble))
    if not restart_scores:
        return {"name": "G6_part_material", "passed": True, "floor": True,
                "reason": "empty palette"}
    consensus = float(np.median(restart_scores))
    return {
        "name": "G6_part_material",
        "worst_part_delta_e": round(consensus, 2),
        "restart_delta_e": [round(value, 2) for value in restart_scores],
        "passed": consensus <= float(strict_delta_e),
        "floor": consensus <= float(floor_delta_e),
    }


def cloud_evidence_delta(
    generated_rgba: Any,
    source_rgba: Any,
    *,
    min_part_weight: float = 0.10,
    lightness_tolerance: float = 25.0,
    trim_quantile: float = 0.05,
) -> Optional[float]:
    """Worst generated part's distance to the nearest EVIDENCE in the
    source's pixel cloud (chroma-first, trim-quantile), min over k.

    Motivation (measured on unseen car backs): a visually-correct "dark
    shaded red" part scores 18-24 under the part-MEDIAN metric because
    k-means absorbs the source's dark-red shadow pixels into a neutral
    black cluster — the evidence exists in the source CLOUD but not among
    the k cluster medians. The trim quantile keeps a few stray source
    pixels from vouching for a whole generated part. True flips stay far
    (camouflage/gold pixels have no source evidence). Calibrated as a
    SECONDARY key only: correct backs 4.6-13.0, wrong v2-era backs
    11.5-21.9 — the lines cross at 11-13, so it never accepts alone.
    """

    import numpy as np
    from scipy.cluster.vq import kmeans2
    from skimage import color as skcolor

    def fg_lab(image_rgba: Any, cap: int) -> Any:
        rgba = np.asarray(image_rgba.convert("RGBA"), dtype=np.float32) / 255.0
        mask = rgba[:, :, 3] > 0.5
        lab = skcolor.rgb2lab(rgba[:, :, :3]).astype(np.float32)[mask]
        if len(lab) > cap:
            picks = np.random.default_rng(0).choice(len(lab), cap, replace=False)
            lab = lab[picks]
        return lab

    source_cloud = fg_lab(source_rgba, 20000)
    generated_lab = fg_lab(generated_rgba, 40000)
    if len(source_cloud) == 0 or len(generated_lab) == 0:
        return None
    values = []
    for k in (3, 4, 5):
        _, labels = kmeans2(generated_lab, k, minit="++", seed=0)
        worst = 0.0
        for index in range(k):
            member = generated_lab[labels == index]
            if len(member) < len(generated_lab) * 0.05:
                continue
            if len(member) / len(generated_lab) < float(min_part_weight):
                continue
            median = np.median(member, axis=0)
            ab = np.hypot(source_cloud[:, 1] - median[1],
                          source_cloud[:, 2] - median[2])
            l_excess = np.maximum(
                0.0, np.abs(source_cloud[:, 0] - median[0])
                - float(lightness_tolerance))
            evidence = float(np.quantile(ab + 0.5 * l_excess, trim_quantile))
            worst = max(worst, evidence)
        values.append(worst)
    return min(values) if values else None


def gate_witnessed_consistency(
    generated_rgba: Any,
    mesh: Any,
    source_rgba: Any,
    *,
    azimuth_deg: float,
    elevation_deg: float,
    source_pose: Tuple[float, float] = (0.0, 0.0),
    min_witnessed_px: int = 10000,
    chroma_flip_max: float = 0.65,
    bright_flip_max: float = 0.25,
    witnessed_tile_median_max: float = 20.0,
    render_size: int = 768,
    tile_grid: int = 12,
) -> Dict[str, Any]:
    """Positional consistency against the surface the SOURCE PHOTO
    witnessed, rendered from the candidate's own angle.

    Global foreground statistics measurably cannot separate the residual
    anchor-class failure modes (glass canopy painted body-red scored
    11.97 — inside the old strict line — while clean red backs scored
    23.99/24.12): the wrongness is POSITIONAL. Projecting the source onto
    the mesh (witnessed vertices only, NO fill) and rendering the
    expected colors + witness mask from the target angle gives per-pixel
    expectations exactly where the photo is evidence. Physically
    grounded: the bake paints witnessed texels from the photo regardless,
    so a reference contradicting them is objectively inconsistent — no
    subject knowledge involved. Calibrated: fires on 12/14 wrong tops
    with 0/10 false fires on correct tops; backs at az 180 witness ~600
    px from a front source (physics), so the gate reports
    `witnessed: False` and the caller falls to the starved-angle keys.

    Veto keys (all measured bands): `chroma_flip` — witnessed dark-glass
    pixels that came out chromatic (canopy painted body color);
    `bright_flip` — witnessed dark pixels that came out >= 30 L brighter
    and achromatic (gray-clay canopy); witnessed tile-median distance
    (broad drift).
    """

    import numpy as np
    from PIL import Image
    from skimage import color as skcolor

    from .rendering import render_mesh_views
    from .reference_generation import clay_silhouette, project_source_witness

    import trimesh as _trimesh

    witnessed, colors = project_source_witness(
        mesh, source_rgba, source_pose=source_pose)

    def render_vertex_colors(values: Any) -> Any:
        colored = mesh.copy()
        colored.visual = _trimesh.visual.ColorVisuals(
            colored, vertex_colors=values)
        return render_mesh_views(
            colored, size=render_size, azimuths=[float(azimuth_deg)],
            elevation=float(elevation_deg))[0]

    vertex_rgba = np.zeros((len(colors), 4), dtype=np.uint8)
    vertex_rgba[:, :3] = (np.clip(colors, 0.0, 1.0) * 255).astype(np.uint8)
    vertex_rgba[:, 3] = 255
    expected = np.asarray(
        render_vertex_colors(vertex_rgba).convert("RGB"),
        dtype=np.float32) / 255.0

    marker = np.zeros((len(colors), 4), dtype=np.uint8)
    marker[witnessed] = (255, 255, 255, 255)
    marker[~witnessed, 3] = 255
    witness_mask = np.asarray(
        render_vertex_colors(marker).convert("L"), dtype=np.float32) / 255.0

    clay = render_mesh_views(
        mesh, size=render_size, azimuths=[float(azimuth_deg)],
        elevation=float(elevation_deg))[0].convert("RGBA")
    silhouette = clay_silhouette(clay)

    generated = generated_rgba.convert("RGBA")
    if generated.size != (witness_mask.shape[1], witness_mask.shape[0]):
        generated = generated.resize(
            (witness_mask.shape[1], witness_mask.shape[0]), Image.LANCZOS)
    generated_array = np.asarray(generated, dtype=np.float32) / 255.0
    generated_mask = generated_array[:, :, 3] > 0.5

    valid = generated_mask & silhouette & (witness_mask > 0.85)
    result: Dict[str, Any] = {
        "name": "witnessed_consistency",
        "witnessed_px": int(valid.sum()),
        "witnessed": bool(valid.sum() >= int(min_witnessed_px)),
    }
    if not result["witnessed"]:
        return result

    generated_lab = skcolor.rgb2lab(generated_array[:, :, :3]).astype(np.float32)
    expected_lab = skcolor.rgb2lab(expected).astype(np.float32)
    g = generated_lab[valid]
    e = expected_lab[valid]
    g_chroma = np.hypot(g[:, 1], g[:, 2])
    e_chroma = np.hypot(e[:, 1], e[:, 2])
    expected_dark = (e_chroma < 20.0) & (e[:, 0] < 45.0)
    chroma_flip = (float((g_chroma[expected_dark] > 30.0).mean())
                   if expected_dark.any() else 0.0)
    bright_flip = (float(((g[expected_dark, 0] - e[expected_dark, 0] >= 30.0)
                          & (g_chroma[expected_dark] <= 30.0)).mean())
                   if expected_dark.any() else 0.0)

    # Tile medians on bbox-normalized 480px crops (the calibration
    # pipeline's exact frame: thresholds transfer only with the pipeline).
    def crop_to_bbox(array: Any, mask: Any) -> Tuple[Any, Any]:
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        r0, r1 = int(np.argmax(rows)), int(len(rows) - np.argmax(rows[::-1]))
        c0, c1 = int(np.argmax(cols)), int(len(cols) - np.argmax(cols[::-1]))
        return array[r0:r1, c0:c1], mask[r0:r1, c0:c1]

    def tiles(image_rgb: Any, fg_mask: Any, weight: Optional[Any]) -> List[Optional[Any]]:
        lab = skcolor.rgb2lab(image_rgb.astype(np.float32))
        height, width = fg_mask.shape
        th, tw = height // tile_grid, width // tile_grid
        out: List[Optional[Any]] = []
        for row in range(tile_grid):
            for col in range(tile_grid):
                fm = fg_mask[row * th:(row + 1) * th, col * tw:(col + 1) * tw]
                if fm.mean() < 0.6:
                    out.append(None)
                    continue
                if weight is not None:
                    wm = weight[row * th:(row + 1) * th, col * tw:(col + 1) * tw]
                    if wm[fm].mean() < 0.7:
                        out.append(None)
                        continue
                out.append(np.median(
                    lab[row * th:(row + 1) * th, col * tw:(col + 1) * tw][fm],
                    axis=0))
        return out

    def resize_pair(array: Any, mask: Any, *, nearest: bool = False) -> Tuple[Any, Any]:
        size = (480, 480)
        image = Image.fromarray((array * 255).astype(np.uint8)).resize(size, Image.LANCZOS)
        mask_image = Image.fromarray((mask * 255).astype(np.uint8)).resize(size, Image.NEAREST)
        return (np.asarray(image, dtype=np.float32) / 255.0,
                np.asarray(mask_image) > 128)

    gen_crop, gen_crop_mask = crop_to_bbox(generated_array[:, :, :3], generated_mask)
    exp_crop, exp_crop_mask = crop_to_bbox(expected, silhouette)
    wit_crop, _ = crop_to_bbox(witness_mask[:, :, None], silhouette)
    gen_480, gen_mask_480 = resize_pair(gen_crop, gen_crop_mask)
    exp_480, exp_mask_480 = resize_pair(exp_crop, exp_crop_mask)
    wit_480 = np.asarray(
        Image.fromarray((wit_crop[:, :, 0] * 255).astype(np.uint8)).resize(
            (480, 480), Image.NEAREST), dtype=np.float32) / 255.0

    generated_tiles = tiles(gen_480, gen_mask_480, None)
    expected_tiles = tiles(exp_480, exp_mask_480, wit_480)
    distances = []
    for gt, et in zip(generated_tiles, expected_tiles):
        if gt is None or et is None:
            continue
        chroma_distance = float(np.hypot(gt[1] - et[1], gt[2] - et[2]))
        lightness_excess = max(0.0, abs(float(gt[0] - et[0])) - 25.0)
        distances.append(chroma_distance + 0.5 * lightness_excess)
    tile_median = float(np.median(distances)) if distances else None

    vetoed = bool(
        chroma_flip > float(chroma_flip_max)
        or bright_flip > float(bright_flip_max)
        or (tile_median is not None
            and tile_median > float(witnessed_tile_median_max))
    )
    result.update(
        chroma_flip=round(chroma_flip, 3),
        bright_flip=round(bright_flip, 3),
        witnessed_tile_median=(
            round(tile_median, 2) if tile_median is not None else None),
        passed=not vetoed,
    )
    return result


def evaluate_material_fidelity(generated: Any, source: Any) -> Dict[str, Any]:
    """Run the per-view gates (G1, G2, G3, G5) and aggregate."""

    gates = [
        gate_tile_albedo(generated, source),
        gate_baked_speculars(generated),
        gate_tile_microstructure(generated, source),
        gate_forbidden_modes(generated, source),
    ]
    return {
        "passed": all(g["passed"] for g in gates),
        "gates": gates,
        "failed": [g["name"] for g in gates if not g["passed"]],
    }
