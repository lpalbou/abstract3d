#!/usr/bin/env python3
"""Calibration probe for the two loop view-quality gates (2026-07-22).

Measures candidate signals over the on-disk evidence so the gate
thresholds are chosen from labeled data, never invented:

MUST-REJECT (e22v2's accepted-but-garbage conditioning views — wrong
identity + catastrophic matte debris carved into the mesh):
  out/bust/e22_oneshot_v2/geometry_view_windowed_side_left.png
  out/bust/e22_oneshot_v2/geometry_view_windowed_back.png
  (+ the synthesized full-span variants as secondary evidence)

MUST-PASS (proven-clean views of the same man):
  out/bust-viewbench/E/*.png   (clay-guided bench arm, pose 0-27 deg)
  out/bust-viewbench/B/*.png   (operator-recipe bench arm)
  out/laurent-bust-redo/viewgen_fixed/*.png (the e20 champion's views)

Signals per view (vs the source photo /tmp/laurent_front_clean4.png):
  matte:   connected components (count + off-main mass), semi-transparent
           fraction, alpha edge raggedness (perimeter / sqrt(area)) as a
           ratio to the source matte's own raggedness
  palette: banded-LAB outlier fraction over the anatomical window span
           (model_evaluation.build_palette machinery on the VIEW)
  structure: per-band gradient-energy ratio (view band / source band)
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


def load_raw(path: Path):
    """The exact pixels on disk (already matted artifacts)."""
    return Image.open(path).convert("RGBA")


def matte_stats(rgba) -> dict:
    from scipy import ndimage

    arr = np.asarray(rgba, dtype=np.float64)
    alpha = arr[..., 3] / 255.0
    mask = alpha > 0.5
    if not mask.any():
        return {"error": "empty"}
    labels, n = ndimage.label(mask)
    sizes = np.bincount(labels.ravel())[1:]
    main = sizes.max()
    off_main = float((sizes.sum() - main) / sizes.sum())
    # perimeter: boundary pixels of the mask
    from scipy.ndimage import binary_erosion

    boundary = mask & ~binary_erosion(mask)
    perimeter = float(boundary.sum())
    raggedness = perimeter / max(np.sqrt(float(mask.sum())), 1.0)
    semi = float(((alpha > 0.06) & (alpha < 0.94)).sum() / max(mask.sum(), 1))
    return {
        "components": int(n),
        "components_over_16px": int((sizes >= 16).sum()),
        "off_main_mass": round(off_main, 5),
        "raggedness": round(raggedness, 3),
        "semi_transparent_frac": round(semi, 5),
    }


def window_band_frame(rgba, bands: int = 6):
    """(mask, row_lo, row_hi): the anatomical window span for banding."""
    from abstract3d.loop_conditioning import window_anchors

    anchors = window_anchors(rgba)
    span = anchors["shoulder"] - anchors["head_top"]
    cut = int(round(anchors["shoulder"] + 0.22 * span))
    arr = np.asarray(rgba)
    mask = arr[..., 3] > 128
    return mask, anchors["head_top"], min(cut, mask.shape[0] - 1)


def banded_signals(view_rgba, source_rgba, palette) -> dict:
    from abstract3d.model_evaluation import _rgb_to_lab

    v_arr = np.asarray(view_rgba.convert("RGBA"), dtype=np.float64)
    s_arr = np.asarray(source_rgba.convert("RGBA"), dtype=np.float64)
    v_mask, v_lo, v_hi = window_band_frame(view_rgba)
    s_mask, s_lo, s_hi = window_band_frame(source_rgba)

    v_rgb = v_arr[..., :3]
    s_rgb = s_arr[..., :3]
    v_lab = _rgb_to_lab(v_rgb)
    s_lab = _rgb_to_lab(s_rgb)

    def grad_energy(rgb, mask):
        lum = rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114
        gy = np.abs(np.diff(lum, axis=0))
        gx = np.abs(np.diff(lum, axis=1))
        g = np.zeros_like(lum)
        g[1:, :] += gy
        g[:, 1:] += gx
        from scipy.ndimage import binary_erosion

        interior = binary_erosion(mask, iterations=2)
        return g, interior

    v_g, v_int = grad_energy(v_rgb, v_mask)
    s_g, s_int = grad_energy(s_rgb, s_mask)

    n_bands = 6
    out_fracs, grad_ratios, chroma_shift = [], [], []
    for b in range(n_bands):
        vb_lo = v_lo + int(b / n_bands * (v_hi - v_lo))
        vb_hi = v_lo + int((b + 1) / n_bands * (v_hi - v_lo))
        sb_lo = s_lo + int(b / n_bands * (s_hi - s_lo))
        sb_hi = s_lo + int((b + 1) / n_bands * (s_hi - s_lo))
        v_band = np.zeros_like(v_mask)
        v_band[vb_lo:vb_hi] = True
        v_band &= v_mask
        s_band = np.zeros_like(s_mask)
        s_band[sb_lo:sb_hi] = True
        s_band &= s_mask
        if v_band.sum() < 200 or s_band.sum() < 200:
            continue
        # banded palette outliers with one-band slack (palette is built on
        # the source's window frame; band index b maps by anatomy)
        sel = v_lab[v_band]
        ratios = []
        for nb in (b - 1, b, b + 1):
            if not (0 <= nb < palette.n_bands) or len(palette.band_centers[nb]) == 0:
                continue
            diffs = sel[:, None, :] - palette.band_centers[nb][None, :, :]
            dist = np.sqrt((diffs**2).sum(axis=2)).min(axis=1)
            ratios.append(dist / palette.band_thresholds[nb])
        if ratios:
            nearest = np.min(np.stack(ratios, axis=0), axis=0)
            out_fracs.append(float((nearest > 1.0).mean()))
        # gradient energy ratio (interior pixels only)
        v_sel = v_band & v_int
        s_sel = s_band & s_int
        if v_sel.sum() > 200 and s_sel.sum() > 200:
            ve = float(np.sqrt((v_g[v_sel] ** 2).mean()))
            se = float(np.sqrt((s_g[s_sel] ** 2).mean()))
            grad_ratios.append(ve / max(se, 1e-6))
        # band mean chroma shift (a,b LAB)
        v_ab = v_lab[v_band][:, 1:].mean(axis=0)
        s_ab = s_lab[s_band][:, 1:].mean(axis=0)
        chroma_shift.append(float(np.sqrt(((v_ab - s_ab) ** 2).sum())))
    return {
        "palette_outlier_max": round(max(out_fracs), 4) if out_fracs else None,
        "palette_outlier_mean": round(float(np.mean(out_fracs)), 4) if out_fracs else None,
        "grad_ratio_max": round(max(grad_ratios), 3) if grad_ratios else None,
        "grad_ratio_bands": [round(g, 2) for g in grad_ratios],
        "outlier_bands": [round(o, 3) for o in out_fracs],
        "chroma_shift_max": round(max(chroma_shift), 2) if chroma_shift else None,
    }


def main() -> None:
    from abstract3d.model_evaluation import build_palette

    source = load_matted(SOURCE)
    s_arr = np.asarray(source, dtype=np.float64)
    s_mask_full, s_lo, s_hi = window_band_frame(source)
    # palette over the WINDOW span only (the frame that conditions)
    window_mask = s_mask_full.copy()
    window_mask[:s_lo] = False
    window_mask[s_hi:] = False
    palette = build_palette(s_arr[..., :3], window_mask)
    src_matte = matte_stats(source)
    print(f"SOURCE matte: {src_matte}")

    groups = {
        "REJECT_e22v2_windowed": [
            REPO / "out/bust/e22_oneshot_v2/geometry_view_windowed_side_left.png",
            REPO / "out/bust/e22_oneshot_v2/geometry_view_windowed_back.png",
        ],
        "REJECT_e22v2_synth_fullspan": [
            REPO / "out/bust/e22_oneshot_v2/geometry_view_synthesized_side_left.png",
            REPO / "out/bust/e22_oneshot_v2/geometry_view_synthesized_back.png",
            REPO / "out/bust/e22_oneshot_v2/geometry_view_synthesized_side_right.png",
        ],
        "PASS_bench_E": sorted((REPO / "out/bust-viewbench/E").glob("*[!w].png")),
        "PASS_bench_B": sorted((REPO / "out/bust-viewbench/B").glob("*[!w].png")),
        "PASS_viewgen_fixed": [
            REPO / "out/laurent-bust-redo/viewgen_fixed/side_left.png",
            REPO / "out/laurent-bust-redo/viewgen_fixed/side_right.png",
            REPO / "out/laurent-bust-redo/viewgen_fixed/back.png",
        ],
        "PASS_control_inputs_windowed": [
            REPO / "out/bust/control_cleanviews/inputs/windowed_side_left.png",
            REPO / "out/bust/control_cleanviews/inputs/windowed_back.png",
        ],
    }
    results = {}
    for group, paths in groups.items():
        for path in paths:
            if not path.exists() or path.name.endswith("_raw.png"):
                continue
            rgba = load_raw(path)
            row = {"matte": matte_stats(rgba)}
            row["matte"]["raggedness_ratio"] = round(
                row["matte"]["raggedness"] / src_matte["raggedness"], 3)
            try:
                row["banded"] = banded_signals(rgba, source, palette)
            except Exception as exc:
                row["banded"] = {"error": f"{type(exc).__name__}: {exc}"}
            key = f"{group}/{path.name}"
            results[key] = row
            m, b = row["matte"], row["banded"]
            print(f"{key}\n  matte: comps16={m.get('components_over_16px')} "
                  f"off_main={m.get('off_main_mass')} ragged_ratio={m.get('raggedness_ratio')} "
                  f"semi={m.get('semi_transparent_frac')}\n  banded: {b}")
    out = Path("/tmp/view_gate_calibration.json")
    out.write_text(json.dumps(results, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
