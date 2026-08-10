#!/usr/bin/env python
"""Striation audit — stage 2: objective feature-row landmarks in the DiT frame.

Landmarks (each computed on the 512x512 recentered frame from stage 1, mask =
alpha > 127):
  head_top      first mask row (top of hair)                    [all views]
  glasses       darkness-weighted centroid of the sunglasses     [front/sides]
                band: within-mask pixels with luminance < 60 in
                the upper half of the head; the band is the
                largest dark run of rows
  nose_tip      profile silhouette extremum: row of the max      [sides]
                horizontal extent toward the facing direction,
                searched between glasses centroid and glasses
                centroid + 0.35 * head span (the nose lives
                there; excludes chin/shoulder extrema)
  lips          redness peak: max row-mean of (R - (G+B)/2)      [front/sides]
                over mask pixels, searched between glasses
                bottom + 8 and beard_bottom - 8
  beard_bottom  strongest dark->light row step of within-mask    [front/sides]
                mean luminance below the glasses (the beard to
                neck/shirt boundary is the strongest such step)
  shoulder      first row (below glasses) where mask width       [all views]
                >= 55% of the frame width
  bottom_cut    last mask row (where the frame cuts the torso)   [all views]

Emits /tmp/striation_audit/landmarks.json and per-view overlay PNGs
(*_landmarks.png) for visual verification of every number.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

OUT = Path("/tmp/striation_audit")
PARAMS = json.loads((OUT / "recenter_params.json").read_text())

DARK_LUM = 60.0
SHOULDER_FRAC = 0.55


def luminance(rgb: np.ndarray) -> np.ndarray:
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def largest_true_run(rows: np.ndarray) -> tuple[int, int] | None:
    """(start, stop) of the longest consecutive True run, or None."""
    idx = np.flatnonzero(rows)
    if idx.size == 0:
        return None
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate([[0], breaks + 1])
    stops = np.concatenate([breaks, [idx.size - 1]])
    lengths = idx[stops] - idx[starts]
    k = int(np.argmax(lengths))
    return int(idx[starts[k]]), int(idx[stops[k]]) + 1


def measure(view_name: str, frame: np.ndarray) -> dict:
    rgb = frame[..., :3].astype(np.float64)
    mask = frame[..., 3] > 127
    h, w = mask.shape
    lum = luminance(rgb)
    rows_any = np.flatnonzero(mask.any(axis=1))
    head_top = int(rows_any[0])
    bottom_cut = int(rows_any[-1])
    out: dict = {"head_top": head_top, "bottom_cut": bottom_cut}

    is_back = view_name == "back"
    if not is_back:
        # Glasses: darkest wide band in the upper part of the subject.
        dark = mask & (lum < DARK_LUM)
        dark_count = dark.sum(axis=1).astype(np.float64)
        upper = np.zeros(h, dtype=bool)
        span = bottom_cut - head_top
        upper[head_top : head_top + int(0.45 * span)] = True
        candidate_rows = (dark_count > 0.02 * w) & upper
        run = largest_true_run(candidate_rows)
        if run is None:
            raise RuntimeError(f"{view_name}: no glasses band found")
        g0, g1 = run
        band = np.arange(g0, g1)
        weights = dark_count[g0:g1]
        glasses = float((band * weights).sum() / weights.sum())
        out["glasses"] = glasses
        out["glasses_band"] = [g0, g1]

        # Beard bottom: strongest dark->light step of within-mask mean
        # luminance below the glasses band. Mean over mask pixels only.
        mean_lum = np.full(h, np.nan)
        for y in range(h):
            m = mask[y]
            if m.any():
                mean_lum[y] = lum[y, m].mean()
        search0 = int(g1 + 0.05 * span)
        search1 = min(bottom_cut - 2, int(g1 + 0.55 * span))
        step = np.full(h, -np.inf)
        for y in range(search0, search1):
            above = mean_lum[max(0, y - 6) : y]
            below = mean_lum[y + 1 : y + 7]
            if np.isnan(above).all() or np.isnan(below).all():
                continue
            step[y] = np.nanmean(below) - np.nanmean(above)
        beard_bottom = int(np.argmax(step))
        out["beard_bottom"] = beard_bottom
        out["beard_step_strength"] = float(step[beard_bottom])

        # Lips: redness peak between glasses bottom and beard bottom.
        redness = rgb[..., 0] - 0.5 * (rgb[..., 1] + rgb[..., 2])
        red_rows = np.full(h, -np.inf)
        for y in range(int(g1) + 8, beard_bottom - 8):
            m = mask[y]
            if m.sum() > 4:
                vals = np.sort(redness[y, m])[-10:]  # top-10 reddest pixels
                red_rows[y] = vals.mean()
        lips = int(np.argmax(red_rows))
        out["lips"] = lips
        out["lips_strength"] = float(red_rows[lips])

        if view_name in ("side_left", "side_right"):
            # Nose tip: silhouette extremum toward the facing direction.
            cols = np.arange(w)
            n0 = int(glasses)
            n1 = min(beard_bottom, int(glasses + 0.35 * span))
            extent = np.full(h, -np.inf)
            for y in range(n0, n1):
                m = mask[y]
                if not m.any():
                    continue
                c = cols[m]
                extent[y] = (w - c.min()) if view_name == "side_left" else c.max()
            nose_tip = int(np.argmax(extent))
            out["nose_tip"] = nose_tip

    width_frac = mask.sum(axis=1) / float(w)
    below = np.flatnonzero((width_frac >= SHOULDER_FRAC) & (np.arange(h) > head_top))
    out["shoulder"] = int(below[0]) if below.size else None
    return out


def overlay(frame: np.ndarray, marks: dict, path: Path) -> None:
    im = Image.fromarray(frame[..., :3]).convert("RGB").resize((1024, 1024), Image.LANCZOS)
    d = ImageDraw.Draw(im)
    colors = {
        "head_top": (255, 255, 0),
        "glasses": (0, 255, 255),
        "nose_tip": (255, 128, 0),
        "lips": (255, 0, 0),
        "beard_bottom": (0, 255, 0),
        "shoulder": (255, 0, 255),
        "bottom_cut": (128, 128, 255),
    }
    for key, color in colors.items():
        v = marks.get(key)
        if v is None:
            continue
        yy = int(round(float(v) * 2))
        d.line([(0, yy), (1024, yy)], fill=color, width=2)
        d.text((6, yy + 2), f"{key}={float(v):.1f}", fill=color)
    im.save(path)


def main() -> None:
    results: dict = {}
    for set_name, views in PARAMS.items():
        results[set_name] = {}
        for view_name in views:
            frame = np.asarray(Image.open(OUT / f"{set_name}_{view_name}_dit512.png"))
            marks = measure(view_name, frame)
            results[set_name][view_name] = marks
            overlay(frame, marks, OUT / f"{set_name}_{view_name}_landmarks.png")
            printable = {
                k: (round(v, 1) if isinstance(v, float) else v)
                for k, v in marks.items()
                if not k.endswith("_strength") and k != "glasses_band"
            }
            print(f"{set_name}/{view_name}: {printable}")
    (OUT / "landmarks.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    sys.exit(main())
