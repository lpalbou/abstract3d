#!/usr/bin/env python3
"""Width-profile view harmonization v2 (double-mouth fix, experiment lane).

Root cause (measured on out/laurent-bust-redo/hunyuan_multiview): the front
conditioning image is a tight head crop while i2i-synthesized side/back
views include extra torso. Hunyuan3D-2mv recenters each view by its own
alpha bbox, so shared features land on different normalized rows across
views (mouth at 62% in front vs ~45% in sides) and the shape DiT carves
both hypotheses -> double mouth.

Law: for same-elevation turntable views (near-orthographic), a world point
projects to the SAME image row in every view. Consistency therefore
requires all views to be cut at the same world heights — top of subject is
naturally shared; the BOTTOM cut must correspond to the same anatomical
line in every view.

Registration signal: subject WIDTH-PER-ROW. Unlike edge energy (content
dependent: glasses dominate front, hair the back), silhouette width is
comparable across turntable views of the same object (head bulge, neck
waist, shoulder flare occur at the same world heights in all views).

Procedure per view:
  1. width profile w(y) over the subject bbox, resampled to N rows;
  2. search (scale, shift) mapping the view's profile onto the reference
     (front) profile, maximizing overlap NCC of sqrt-width (sqrt softens
     silhouette area dominance);
  3. the reference's [0,1] span maps to view rows [lo, hi]; alpha-zero
     everything outside -> after downstream bbox recentering all views
     share the anatomical window.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

N_ROWS = 512


def load_rgba(path: Path) -> Image.Image:
    return Image.open(path).convert("RGBA")


def subject_alpha(image: Image.Image) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float64)
    alpha = arr[..., 3] / 255.0
    if alpha.min() > 0.99:
        lum = arr[..., :3].mean(axis=2)
        alpha = (np.abs(lum - lum[0, 0]) > 12).astype(float)
    return alpha


def width_profile(image: Image.Image, rows: int = N_ROWS) -> np.ndarray:
    alpha = subject_alpha(image)
    ys, xs = np.where(alpha > 0.5)
    if ys.size == 0:
        raise ValueError("no subject pixels")
    y0, y1 = ys.min(), ys.max()
    sub = alpha[y0 : y1 + 1]
    w = (sub > 0.5).sum(axis=1).astype(np.float64)
    w = np.interp(np.linspace(0, 1, rows), np.linspace(0, 1, len(w)), w)
    peak = w.max()
    return np.sqrt(w / peak) if peak > 0 else w


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    da, db = a - a.mean(), b - b.mean()
    d = np.sqrt((da * da).sum() * (db * db).sum())
    return float((da * db).sum() / d) if d > 0 else 0.0


def register(
    src: np.ndarray,
    ref: np.ndarray,
    scales: np.ndarray = np.linspace(0.55, 1.6, 106),
    max_shift_frac: float = 0.4,
    min_overlap_frac: float = 0.5,
) -> tuple[float, int, float]:
    n = len(ref)
    best = (1.0, 0, -1.0)
    for sc in scales:
        m = int(round(n * sc))
        if m < 32:
            continue
        src_s = np.interp(np.linspace(0, 1, m), np.linspace(0, 1, n), src)
        for sh in range(-int(n * max_shift_frac), int(n * max_shift_frac) + 1):
            lo, hi = max(0, sh), min(n, sh + m)
            if hi - lo < n * min_overlap_frac:
                continue
            c = ncc(src_s[lo - sh : hi - sh], ref[lo:hi])
            if c > best[2]:
                best = (float(sc), int(sh), float(c))
    return best


def harmonize(view_path: Path, ref: np.ndarray, out_path: Path) -> dict:
    image = load_rgba(view_path)
    prof = width_profile(image)
    scale, shift, corr_reg = register(prof, ref)
    corr_before = ncc(prof, ref)

    alpha = subject_alpha(image)
    ys = np.where(alpha.max(axis=1) > 0.5)[0]
    y0, y1 = int(ys.min()), int(ys.max())
    subj_h = y1 - y0 + 1

    # ref row r maps to src normalized (r - shift) / (n * scale)
    n = N_ROWS
    lo_norm = max((0 - shift) / (n * scale), 0.0)
    hi_norm = min((n - shift) / (n * scale), 1.0)
    if hi_norm - lo_norm < 0.2:
        raise ValueError(f"{view_path.name}: degenerate window {lo_norm:.2f}..{hi_norm:.2f}")
    crop_y0 = y0 + int(round(lo_norm * subj_h))
    crop_y1 = y0 + int(round(hi_norm * subj_h))

    arr = np.asarray(image).copy()
    arr[:crop_y0, :, 3] = 0
    arr[crop_y1 + 1 :, :, 3] = 0
    out = Image.fromarray(arr, "RGBA")
    out.save(out_path)

    return {
        "view": view_path.name,
        "scale": round(scale, 3),
        "shift_rows": shift,
        "corr_before": round(corr_before, 3),
        "corr_registered": round(corr_reg, 3),
        "corr_after_crop": round(ncc(width_profile(out), ref), 3),
        "window_norm": [round(lo_norm, 3), round(hi_norm, 3)],
        "output": str(out_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--front", required=True, type=Path)
    parser.add_argument("--views", nargs="+", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ref = width_profile(load_rgba(args.front))
    reports = []
    for view in args.views:
        report = harmonize(view, ref, args.output_dir / f"h2_{view.stem}.png")
        reports.append(report)
        print(json.dumps(report))
    (args.output_dir / "harmonize_v2_report.json").write_text(json.dumps(reports, indent=1))


if __name__ == "__main__":
    main()
