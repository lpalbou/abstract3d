#!/usr/bin/env python3
"""Top-anchored view harmonization v3 (double-mouth fix, experiment lane).

v2 failed because free (scale, shift) width-profile registration matched
the front's head+shoulders onto the sides' chest+waist (beheaded crops).

The turntable constraint gives a stronger anchor: for same-elevation views
of an upright subject, the TOP of the subject (top of head) is the same
world height in every view — its projected row after bbox-normalization is
row 0 everywhere. Only the BOTTOM differs (how much torso each view
includes). So the unknown is a single bottom-cut fraction x: the front's
full subject span [0,1] corresponds to the synthesized view's span [0,x].

Search x in [0.35,1.0] (+ small top slack for hair variance), score by NCC
of a 2-channel profile (sqrt silhouette width + horizontal-edge energy) —
width pins the shoulder flare, edge energy pins the chin/feature lines.
Keep the synthesized subject's rows [0,x], alpha-zero the rest.
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


def profiles(image: Image.Image, rows: int = N_ROWS) -> np.ndarray:
    """(rows, 2): [sqrt-width, edge-energy], each L2-normalized over the
    subject bbox."""
    arr = np.asarray(image, dtype=np.float64)
    alpha = subject_alpha(image)
    ys, xs = np.where(alpha > 0.5)
    if ys.size == 0:
        raise ValueError("no subject pixels")
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    sub_a = alpha[y0 : y1 + 1, x0 : x1 + 1]
    sub_c = arr[y0 : y1 + 1, x0 : x1 + 1]

    width = (sub_a > 0.5).sum(axis=1).astype(np.float64)
    width = np.sqrt(width / max(width.max(), 1.0))

    lum = sub_c[..., :3] @ np.array([0.299, 0.587, 0.114])
    gy = np.abs(np.diff(lum, axis=0)) * np.minimum(sub_a[1:], sub_a[:-1])
    edge = np.concatenate([[0.0], gy.sum(axis=1)])
    if edge.max() > 0:
        edge = edge / edge.max()

    grid = np.linspace(0, 1, rows)
    src = np.linspace(0, 1, len(width))
    out = np.stack(
        [np.interp(grid, src, width), np.interp(grid, src, edge)], axis=1
    )
    return out


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    total, chans = 0.0, 0
    for c in range(a.shape[1]):
        da, db = a[:, c] - a[:, c].mean(), b[:, c] - b[:, c].mean()
        d = np.sqrt((da * da).sum() * (db * db).sum())
        if d > 0:
            total += float((da * db).sum() / d)
            chans += 1
    return total / max(chans, 1)


def register_top_anchored(
    src: np.ndarray,
    ref: np.ndarray,
    cuts: np.ndarray = np.linspace(0.35, 1.0, 66),
    top_slack: np.ndarray = np.linspace(0.0, 0.06, 4),
) -> tuple[float, float, float]:
    """Find (cut, top, score): ref[0,1] corresponds to src[top, cut]."""
    n = len(ref)
    best = (1.0, 0.0, -1.0)
    grid = np.linspace(0, 1, n)
    for top in top_slack:
        for cut in cuts:
            if cut - top < 0.25:
                continue
            span = np.linspace(top, cut, n)
            seg = np.stack(
                [np.interp(span, grid, src[:, c]) for c in range(src.shape[1])],
                axis=1,
            )
            score = ncc(seg, ref)
            if score > best[2]:
                best = (float(cut), float(top), float(score))
    return best


def harmonize(view_path: Path, ref: np.ndarray, out_path: Path) -> dict:
    image = load_rgba(view_path)
    prof = profiles(image)
    cut, top, score = register_top_anchored(prof, ref)
    score_before = ncc(prof, ref)

    alpha = subject_alpha(image)
    ys = np.where(alpha.max(axis=1) > 0.5)[0]
    y0, y1 = int(ys.min()), int(ys.max())
    subj_h = y1 - y0 + 1
    crop_y0 = y0 + int(round(top * subj_h))
    crop_y1 = y0 + int(round(cut * subj_h))

    arr = np.asarray(image).copy()
    arr[:crop_y0, :, 3] = 0
    arr[crop_y1 + 1 :, :, 3] = 0
    out = Image.fromarray(arr, "RGBA")
    out.save(out_path)

    return {
        "view": view_path.name,
        "window": [round(top, 3), round(cut, 3)],
        "score_before": round(score_before, 3),
        "score_registered": round(score, 3),
        "score_after_crop": round(ncc(profiles(out), ref), 3),
        "output": str(out_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--front", required=True, type=Path)
    parser.add_argument("--views", nargs="+", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ref = profiles(load_rgba(args.front))
    reports = []
    for view in args.views:
        report = harmonize(view, ref, args.output_dir / f"h3_{view.stem}.png")
        reports.append(report)
        print(json.dumps(report))
    (args.output_dir / "harmonize_v3_report.json").write_text(json.dumps(reports, indent=1))


if __name__ == "__main__":
    main()
