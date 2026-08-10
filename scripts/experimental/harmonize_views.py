#!/usr/bin/env python3
"""Row-profile view harmonization experiment (double-mouth fix validation).

Hypothesis: Hunyuan3D-2mv carves duplicate features (double mouth) because
independently synthesized conditioning views place facial features at
different normalized image heights than the real front photo. For
same-elevation turntable views under near-orthographic projection, the
epipolar constraint degenerates to "same feature -> same image row"
(Era3D's row-wise attention exploits exactly this). Independent i2i
generations violate it.

Fix under test: register each synthesized view's vertical row-profile
(horizontal-edge energy over the subject) to the front photo's profile via
a (scale, shift) search, then crop each subject so that after standard
bbox recentering all views agree about feature heights.

This is an out-of-tree experiment script; the durable implementation lives
in abstract3d.view_consistency + the geometry-view gates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

PROFILE_ROWS = 512


def load_rgba(path: Path) -> Image.Image:
    return Image.open(path).convert("RGBA")


def subject_alpha(image: Image.Image) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float64)
    alpha = arr[..., 3] / 255.0
    if alpha.min() > 0.99:  # opaque image: estimate foreground from corners
        lum = arr[..., :3].mean(axis=2)
        alpha = (np.abs(lum - lum[0, 0]) > 12).astype(float)
    return alpha


def subject_bbox(alpha: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(alpha > 0.5)
    if ys.size == 0:
        raise ValueError("no subject pixels found")
    return int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())


def row_profile(image: Image.Image, rows: int = PROFILE_ROWS) -> np.ndarray:
    """Horizontal-edge energy per subject row, bbox-normalized, resampled."""
    arr = np.asarray(image, dtype=np.float64)
    alpha = subject_alpha(image)
    y0, y1, x0, x1 = subject_bbox(alpha)
    crop = arr[y0 : y1 + 1, x0 : x1 + 1]
    am = alpha[y0 : y1 + 1, x0 : x1 + 1]
    lum = crop[..., :3] @ np.array([0.299, 0.587, 0.114])
    gy = np.abs(np.diff(lum, axis=0)) * np.minimum(am[1:], am[:-1])
    prof = gy.sum(axis=1)
    prof = np.interp(np.linspace(0, 1, rows), np.linspace(0, 1, len(prof)), prof)
    total = prof.sum()
    return prof / total if total > 0 else prof


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    da, db = a - a.mean(), b - b.mean()
    denom = np.sqrt((da * da).sum() * (db * db).sum())
    return float((da * db).sum() / denom) if denom > 0 else 0.0


def register_scale_shift(
    src: np.ndarray,
    ref: np.ndarray,
    scales: np.ndarray = np.linspace(0.6, 1.4, 81),
    max_shift_frac: float = 0.35,
) -> tuple[float, int, float]:
    """Best (scale, shift, corr): overlay src scaled by `scale` at row offset
    `shift` onto the ref grid, maximizing masked correlation. Requires at
    least 40% overlap so a sliver match can't win."""
    n = len(ref)
    best = (1.0, 0, -1.0)
    max_shift = int(n * max_shift_frac)
    for sc in scales:
        m = int(round(n * sc))
        if m < 32:
            continue
        src_s = np.interp(np.linspace(0, 1, m), np.linspace(0, 1, n), src)
        for sh in range(-max_shift, max_shift + 1):
            lo, hi = max(0, sh), min(n, sh + m)
            if hi - lo < n * 0.4:
                continue
            c = ncc(src_s[lo - sh : hi - sh], ref[lo:hi])
            if c > best[2]:
                best = (float(sc), int(sh), float(c))
    return best


def harmonize(
    view_path: Path, ref_profile: np.ndarray, out_path: Path
) -> dict:
    """Crop the view's subject so its normalized rows align with the
    reference profile. Given the registration ref_row = src_norm*scale*n + shift,
    the src rows that cover ref rows [0, n] are
    src_norm in [(-shift)/(n*scale), (n-shift)/(n*scale)] — crop that window
    (clamped to the subject) so bbox recentering reproduces the alignment."""
    image = load_rgba(view_path)
    prof = row_profile(image)
    scale, shift, corr_after = register_scale_shift(prof, ref_profile)
    corr_before = ncc(prof, ref_profile)

    alpha = subject_alpha(image)
    y0, y1, x0, x1 = subject_bbox(alpha)
    subj_h = y1 - y0 + 1

    n = PROFILE_ROWS
    lo_norm = (0 - shift) / (n * scale)
    hi_norm = (n - shift) / (n * scale)
    # Clamp to available subject extent; report the uncovered remainder.
    lo_clamped = min(max(lo_norm, 0.0), 0.9)
    hi_clamped = max(min(hi_norm, 1.0), lo_clamped + 0.1)
    crop_y0 = y0 + int(round(lo_clamped * subj_h))
    crop_y1 = y0 + int(round(hi_clamped * subj_h))

    arr = np.asarray(image).copy()
    # Zero alpha outside the harmonized window: bbox recentering then sees
    # only the aligned span.
    arr[: crop_y0, :, 3] = 0
    arr[crop_y1 + 1 :, :, 3] = 0
    out = Image.fromarray(arr, "RGBA")
    out.save(out_path)

    check = ncc(row_profile(out), ref_profile)
    report = {
        "view": view_path.name,
        "scale": round(scale, 3),
        "shift_rows": shift,
        "shift_frac": round(shift / n, 3),
        "corr_before": round(corr_before, 3),
        "corr_registered": round(corr_after, 3),
        "corr_after_crop": round(check, 3),
        "window_norm": [round(lo_norm, 3), round(hi_norm, 3)],
        "window_applied": [round(lo_clamped, 3), round(hi_clamped, 3)],
        "crop_rows": [int(crop_y0), int(crop_y1)],
        "output": str(out_path),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--front", required=True, type=Path)
    parser.add_argument("--views", nargs="+", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ref_profile = row_profile(load_rgba(args.front))

    reports = []
    for view in args.views:
        out_path = args.output_dir / f"harmonized_{view.stem}.png"
        report = harmonize(view, ref_profile, out_path)
        reports.append(report)
        print(json.dumps(report, indent=1))

    (args.output_dir / "harmonize_report.json").write_text(
        json.dumps(reports, indent=1)
    )


if __name__ == "__main__":
    main()
