#!/usr/bin/env python3
"""Gate calibration round 2 (2026-07-22): interior-aware signals.

Round-1 findings (/tmp/view_gate_calibration.json):
- semi_transparent_frac separates the side_left debris (0.19) but NOT the
  back (0.051 vs viewgen back 0.050): rim wisps dominate both.
- banded palette outliers were POISONED by matte-edge white-bleed: the
  source's own boundary blend pixels (white background) entered the band
  palettes and legitimized the straw-blond debris hair. Interior erosion
  is required on BOTH sides.
- bench-B must-pass views carry up to 11 components / raggedness 1.74, so
  component/raggedness counts cannot gate alone.

Round-2 signals:
  interior_semi_frac  semi-transparent pixels AWAY from the silhouette rim
                      (alpha tears INSIDE the body) / subject area
  palette_outlier     banded LAB outliers, interior-eroded palette + pixels
  dark_grad_ratio     per-band gradient energy over interior DARK pixels
                      (the subject's own dark strata: shirt, hair) as a
                      ratio to the source's same band; paint-streak texture
                      on smooth dark fabric fires it
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
SOURCE = Path("/tmp/laurent_front_clean4.png")


def load_matted(path: Path):
    from abstract3d.segmentation import clean_alpha_mask, remove_background_robust

    img = Image.open(path).convert("RGBA")
    if img.getchannel("A").getextrema()[0] < 255:
        return clean_alpha_mask(img)
    return remove_background_robust(img)


def interior_erosion_px(mask: np.ndarray) -> int:
    rows = np.flatnonzero(mask.any(axis=1))
    height = int(rows[-1] - rows[0] + 1) if rows.size else mask.shape[0]
    return max(2, int(round(0.008 * height)))


def matte_interior_stats(rgba) -> dict:
    from scipy.ndimage import binary_dilation, binary_erosion

    arr = np.asarray(rgba, dtype=np.float64)
    alpha = arr[..., 3] / 255.0
    support = alpha > 0.06
    mask = alpha > 0.5
    if not mask.any():
        return {"error": "empty"}
    semi = (alpha > 0.06) & (alpha < 0.94)
    rim_w = interior_erosion_px(mask)
    rim = support & ~binary_erosion(support, iterations=rim_w)
    interior_semi = semi & ~rim
    # also: fully transparent holes INSIDE the opaque body (torn alpha)
    from scipy.ndimage import binary_fill_holes

    filled = binary_fill_holes(mask)
    holes = filled & ~mask
    return {
        "semi_frac": round(float(semi.sum() / max(mask.sum(), 1)), 5),
        "interior_semi_frac": round(float(interior_semi.sum() / max(mask.sum(), 1)), 5),
        "hole_frac": round(float(holes.sum() / max(mask.sum(), 1)), 5),
    }


def band_frame(rgba):
    from abstract3d.loop_conditioning import window_anchors

    anchors = window_anchors(rgba)
    span = anchors["shoulder"] - anchors["head_top"]
    cut = int(round(anchors["shoulder"] + 0.22 * span))
    arr = np.asarray(rgba)
    mask = arr[..., 3] > 128
    return mask, anchors["head_top"], min(cut, mask.shape[0] - 1)


def banded_interior(view_rgba, source_rgba, n_bands: int = 6) -> dict:
    from scipy.ndimage import binary_erosion

    from abstract3d.model_evaluation import _rgb_to_lab, build_palette

    v_arr = np.asarray(view_rgba.convert("RGBA"), dtype=np.float64)
    s_arr = np.asarray(source_rgba.convert("RGBA"), dtype=np.float64)
    v_mask, v_lo, v_hi = band_frame(view_rgba)
    s_mask, s_lo, s_hi = band_frame(source_rgba)
    v_int = binary_erosion(v_mask, iterations=interior_erosion_px(v_mask))
    s_int = binary_erosion(s_mask, iterations=interior_erosion_px(s_mask))

    s_window = s_int.copy()
    s_window[:s_lo] = False
    s_window[s_hi:] = False
    palette = build_palette(s_arr[..., :3], s_window)

    v_lab = _rgb_to_lab(v_arr[..., :3])
    s_lab = _rgb_to_lab(s_arr[..., :3])

    def lum_grad(arr):
        lum = arr[..., 0] * 0.299 + arr[..., 1] * 0.587 + arr[..., 2] * 0.114
        g = np.zeros_like(lum)
        g[1:, :] += np.abs(np.diff(lum, axis=0))
        g[:, 1:] += np.abs(np.diff(lum, axis=1))
        return g

    v_g, s_g = lum_grad(v_arr[..., :3]), lum_grad(s_arr[..., :3])
    # dark stratum: L below the source's own subject median (dark-dominant
    # subject; general: the darker half of the subject's own L range)
    s_l_vals = s_lab[s_int][:, 0]
    dark_line = float(np.percentile(s_l_vals, 50))
    v_dark = v_int & (v_lab[..., 0] < dark_line)
    s_dark = s_int & (s_lab[..., 0] < dark_line)

    out_fracs, dark_ratios = [], []
    for b in range(n_bands):
        vb = np.zeros_like(v_mask)
        vb[v_lo + int(b / n_bands * (v_hi - v_lo)):
           v_lo + int((b + 1) / n_bands * (v_hi - v_lo))] = True
        sb = np.zeros_like(s_mask)
        sb[s_lo + int(b / n_bands * (s_hi - s_lo)):
           s_lo + int((b + 1) / n_bands * (s_hi - s_lo))] = True
        v_band = vb & v_int
        s_band = sb & s_int
        if v_band.sum() < 200 or s_band.sum() < 200:
            out_fracs.append(None)
            dark_ratios.append(None)
            continue
        sel = v_lab[v_band]
        ratios = []
        for nb in (b - 1, b, b + 1):
            if not (0 <= nb < palette.n_bands) or len(palette.band_centers[nb]) == 0:
                continue
            diffs = sel[:, None, :] - palette.band_centers[nb][None, :, :]
            dist = np.sqrt((diffs**2).sum(axis=2)).min(axis=1)
            ratios.append(dist / palette.band_thresholds[nb])
        out_fracs.append(
            round(float((np.min(np.stack(ratios, axis=0), axis=0) > 1.0).mean()), 4)
            if ratios else None)
        v_dark_band = vb & v_dark
        s_dark_band = sb & s_dark
        if v_dark_band.sum() > 400 and s_dark_band.sum() > 400:
            ve = float(np.sqrt((v_g[v_dark_band] ** 2).mean()))
            se = float(np.sqrt((s_g[s_dark_band] ** 2).mean()))
            dark_ratios.append(round(ve / max(se, 1e-6), 2))
        else:
            dark_ratios.append(None)
    valid_out = [o for o in out_fracs if o is not None]
    valid_dark = [d for d in dark_ratios if d is not None]
    return {
        "outlier_bands": out_fracs,
        "outlier_max": max(valid_out) if valid_out else None,
        "dark_grad_bands": dark_ratios,
        "dark_grad_max": max(valid_dark) if valid_dark else None,
        "dark_line_l": round(dark_line, 1),
    }


def main() -> None:
    source = load_matted(SOURCE)
    groups = {
        "REJ_win_sl": REPO / "out/bust/e22_oneshot_v2/geometry_view_windowed_side_left.png",
        "REJ_win_bk": REPO / "out/bust/e22_oneshot_v2/geometry_view_windowed_back.png",
        "REJ_syn_sl": REPO / "out/bust/e22_oneshot_v2/geometry_view_synthesized_side_left.png",
        "REJ_syn_bk": REPO / "out/bust/e22_oneshot_v2/geometry_view_synthesized_back.png",
        "REJ_syn_sr": REPO / "out/bust/e22_oneshot_v2/geometry_view_synthesized_side_right.png",
        "PASS_E_bk": REPO / "out/bust-viewbench/E/back.png",
        "PASS_E_pl": REPO / "out/bust-viewbench/E/profile_left.png",
        "PASS_E_pr": REPO / "out/bust-viewbench/E/profile_right.png",
        "PASS_E_s65l": REPO / "out/bust-viewbench/E/side65_left.png",
        "PASS_E_s65r": REPO / "out/bust-viewbench/E/side65_right.png",
        "PASS_B_bk": REPO / "out/bust-viewbench/B/back.png",
        "PASS_B_pl": REPO / "out/bust-viewbench/B/profile_left.png",
        "PASS_B_pr": REPO / "out/bust-viewbench/B/profile_right.png",
        "PASS_B_s65l": REPO / "out/bust-viewbench/B/side65_left.png",
        "PASS_B_s65r": REPO / "out/bust-viewbench/B/side65_right.png",
        "PASS_vf_sl": REPO / "out/laurent-bust-redo/viewgen_fixed/side_left.png",
        "PASS_vf_sr": REPO / "out/laurent-bust-redo/viewgen_fixed/side_right.png",
        "PASS_vf_bk": REPO / "out/laurent-bust-redo/viewgen_fixed/back.png",
    }
    results = {}
    for key, path in groups.items():
        rgba = Image.open(path).convert("RGBA")
        row = {"matte": matte_interior_stats(rgba)}
        try:
            row["banded"] = banded_interior(rgba, source)
        except Exception as exc:
            row["banded"] = {"error": f"{type(exc).__name__}: {exc}"}
        results[key] = row
        m, b = row["matte"], row["banded"]
        print(f"{key:12s} semi={m.get('semi_frac')} int_semi={m.get('interior_semi_frac')} "
              f"holes={m.get('hole_frac')} out_max={b.get('outlier_max')} "
              f"dark_max={b.get('dark_grad_max')}")
        print(f"             out_bands={b.get('outlier_bands')}")
        print(f"             dark_bands={b.get('dark_grad_bands')}")
    Path("/tmp/view_gate_calibration2.json").write_text(json.dumps(results, indent=1))
    print("wrote /tmp/view_gate_calibration2.json")


if __name__ == "__main__":
    main()
