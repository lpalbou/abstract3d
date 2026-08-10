#!/usr/bin/env python3
"""Harmonize a conditioning-view SET to one anatomical window (cycle-1 math).

Mechanism being fixed (measured, /tmp/striation_audit/landmarks.json): the
Hunyuan3D-2mv preprocessing recenters each view by its own alpha bbox, so a
feature's normalized row is (r - head_top) / (bottom_cut - head_top). Views
cut at different torso depths put the SAME anatomy at DIFFERENT normalized
rows (e11 set: lips at 0.31 in the front vs 0.47 in the sides), and the
shape DiT carves every hypothesis -> striated mouths. Uniform y-scaling
cannot fix a ratio; the WINDOW can: cut every view (front included) to the
same anatomical span [head_top, shoulder + k*(shoulder - head_top)], and the
ratio structure equalizes by construction.

Anchors are subject-agnostic and measurable in EVERY view including the
back: head_top = first mask row; shoulder = first row below head_top where
the mask width reaches SHOULDER_FRAC of the row-wise max width (the
shoulder flare is the widest structure of a bust in any azimuth).

Only equatorial views belong in this harmonization (the same-row law
degenerates from equal-elevation turntable geometry); elevated views are
texture-lane material and must not be windowed here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

SHOULDER_FRAC = 0.92  # fraction of the max row width that counts as "shoulder reached"
DEFAULT_K = 0.22      # torso kept below the shoulder line, in head-to-shoulder spans


def subject_alpha(image: Image.Image) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float64)
    alpha = arr[..., 3] / 255.0
    if alpha.min() > 0.99:
        lum = arr[..., :3].mean(axis=2)
        alpha = (np.abs(lum - lum[0, 0]) > 12).astype(float)
    return alpha


def anchors(image: Image.Image) -> dict:
    """head_top and shoulder rows from the alpha support."""
    alpha = subject_alpha(image)
    mask = alpha > 0.5
    rows = np.flatnonzero(mask.any(axis=1))
    if rows.size == 0:
        raise ValueError("no subject pixels")
    head_top = int(rows[0])
    widths = mask.sum(axis=1).astype(np.float64)
    wmax = widths[rows].max()
    # First row (scanning down) where width reaches SHOULDER_FRAC of max.
    # For a bust the max width IS the shoulder region, so this finds the
    # row where the flare has essentially arrived.
    cand = np.flatnonzero(widths >= SHOULDER_FRAC * wmax)
    cand = cand[cand > head_top]
    if cand.size == 0:
        raise ValueError("no shoulder line found")
    shoulder = int(cand[0])
    return {"head_top": head_top, "shoulder": shoulder, "bottom": int(rows[-1])}


def window_view(image: Image.Image, k: float) -> tuple[Image.Image, dict]:
    """Alpha-cut the view to [head_top, shoulder + k*(shoulder-head_top)]."""
    a = anchors(image)
    span = a["shoulder"] - a["head_top"]
    if span <= 8:
        raise ValueError(f"degenerate head-to-shoulder span {span}")
    cut = int(round(a["shoulder"] + k * span))
    arr = np.asarray(image.convert("RGBA")).copy()
    if cut + 1 < arr.shape[0]:
        arr[cut + 1 :, :, 3] = 0
    out = Image.fromarray(arr, "RGBA")
    report = dict(a)
    report["cut_row"] = cut
    report["kept_ratio_shoulder"] = round(span / max(cut - a["head_top"], 1), 4)
    return out, report


def normalized_ratios(image: Image.Image) -> dict:
    """Post-window verification: anchor ratios in the bbox-normalized frame."""
    a = anchors(image)
    alpha = subject_alpha(image)
    rows = np.flatnonzero((alpha > 0.5).any(axis=1))
    y0, y1 = int(rows[0]), int(rows[-1])
    h = max(y1 - y0, 1)
    return {
        "shoulder_norm": round((a["shoulder"] - y0) / h, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--front", required=True, type=Path)
    parser.add_argument("--views", nargs="+", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--k", type=float, default=DEFAULT_K)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    reports = {}
    for path in [args.front, *args.views]:
        image = Image.open(path).convert("RGBA")
        out, report = window_view(image, args.k)
        dest = args.output_dir / f"win_{path.stem}.png"
        out.save(dest)
        report["output"] = str(dest)
        report["post_window"] = normalized_ratios(out)
        reports[path.stem] = report
        print(path.stem, json.dumps(report))

    shoulders = [r["post_window"]["shoulder_norm"] for r in reports.values()]
    spread = max(shoulders) - min(shoulders)
    summary = {"shoulder_norm_spread": round(spread, 4), "k": args.k}
    print(json.dumps(summary))
    (args.output_dir / "harmonize_set_report.json").write_text(
        json.dumps({"views": reports, "summary": summary}, indent=1)
    )


if __name__ == "__main__":
    main()
