#!/usr/bin/env python3
"""First-draft azimuth ruler — KEPT AS COUNTER-EVIDENCE, do not use for poses.

Its band (top 62% of the BUST bbox) still includes the shoulders, so it
inherits the full-silhouette pose blindness the viewgen audit documented:
on e20's side_left it reads a flat sweep MAXIMIZED at the declared 90
(0.957 @90 vs 0.920 @50) while the audit-grade head-band instrument
(texture_forensics_ruler2.py, band from the clay's shoulder flare) puts
the true azimuth at 65. Quoted in docs/research/texture_forensics.md as
the measurement showing why the bake's own full-silhouette refiner could
never have caught the pose error. Use texture_forensics_ruler2.py for
actual pose measurements.

Per azimuth candidate:
  mesh silhouette = orthographic vertex splat at canonical framing
  view mask       = recenter_to_canonical_frame(view).alpha
  IoU(az)         = max over scale in [0.85,1.15], integer shifts +-24 px
                    (FFT correlation gives all shifts at once)
computed on rows [top, top + 0.62*span] of the mesh silhouette bbox.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
from PIL import Image

CANVAS = 320
SHIFT_WIN = 26


def mesh_silhouette(vertices: np.ndarray, azimuth_deg: float, elevation_deg: float,
                    half_extent: float, canvas: int) -> np.ndarray:
    from abstract3d.backends.triposr_runtime import (
        _tripo_camera_position, _tripo_look_at_matrix)
    from scipy.ndimage import binary_closing, binary_dilation

    eye = _tripo_camera_position(
        azimuth_deg=azimuth_deg, elevation_deg=elevation_deg, camera_distance=3.0)
    view = _tripo_look_at_matrix(
        eye, np.zeros(3, dtype=np.float32), np.array([0, 0, 1], dtype=np.float32))
    cam = vertices @ view[:3, :3].T + view[:3, 3]
    scale = 0.5 * canvas / max(half_extent, 1e-6)
    x = scale * cam[:, 0] + canvas / 2.0 - 0.5
    y = -scale * cam[:, 1] + canvas / 2.0 - 0.5
    grid = np.zeros((canvas, canvas), dtype=bool)
    xi = np.clip(np.round(x).astype(np.int32), 0, canvas - 1)
    yi = np.clip(np.round(y).astype(np.int32), 0, canvas - 1)
    grid[yi, xi] = True
    grid = binary_dilation(grid, iterations=2)
    grid = binary_closing(grid, structure=np.ones((5, 5), dtype=bool))
    return grid


def band_rows(mask: np.ndarray, frac: float = 0.62) -> tuple[int, int]:
    rows = np.nonzero(mask.any(axis=1))[0]
    top, bottom = int(rows[0]), int(rows[-1])
    return top, top + int(frac * (bottom - top))


def best_iou_over_scale_shift(mesh_band: np.ndarray, view_alpha: np.ndarray,
                              band: tuple[int, int]) -> dict:
    from scipy.signal import fftconvolve

    lo, hi = band
    mesh_crop = mesh_band[lo:hi + 1].astype(np.float32)
    mesh_area = float(mesh_crop.sum())
    best = {"iou": -1.0, "scale": None, "dx": None, "dy": None}
    base = Image.fromarray((view_alpha * 255).astype(np.uint8))
    canvas = view_alpha.shape[0]
    for scale in np.arange(0.85, 1.1501, 0.025):
        new = base.resize(
            (max(1, int(round(canvas * scale))),) * 2, Image.BILINEAR)
        scaled = np.zeros_like(view_alpha, dtype=np.float32)
        arr = np.asarray(new, dtype=np.float32) / 255.0
        off = (canvas - arr.shape[0]) // 2
        if off >= 0:
            scaled[off:off + arr.shape[0], off:off + arr.shape[1]] = arr
        else:
            scaled = arr[-off:-off + canvas, -off:-off + canvas]
        view_bin = scaled > 0.5
        # correlation over all shifts; read only the +-SHIFT_WIN window
        corr = fftconvolve(
            mesh_band.astype(np.float32),
            view_bin[::-1, ::-1].astype(np.float32), mode="same")
        center = canvas // 2
        for dy in range(-SHIFT_WIN, SHIFT_WIN + 1, 2):
            for dx in range(-SHIFT_WIN, SHIFT_WIN + 1, 2):
                # intersection restricted to the band: recompute exactly for
                # the correlation argmax candidates only would be cheaper,
                # but exact per-shift band IoU keeps the ruler honest.
                inter_full = corr[center + dy, center + dx]
                if inter_full <= 0:
                    continue
                shifted = np.roll(np.roll(view_bin, dy, axis=0), dx, axis=1)
                shifted_crop = shifted[lo:hi + 1]
                inter = float(np.logical_and(mesh_crop > 0, shifted_crop).sum())
                union = mesh_area + float(shifted_crop.sum()) - inter
                iou = inter / max(union, 1e-6)
                if iou > best["iou"]:
                    best = {"iou": iou, "scale": round(float(scale), 3),
                            "dx": dx, "dy": dy}
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--view", type=Path, required=True)
    parser.add_argument("--center", type=float, required=True, help="declared azimuth")
    parser.add_argument("--window", type=float, default=40.0)
    parser.add_argument("--step", type=float, default=2.5)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    import trimesh

    from abstract3d.texturing import canonical_ortho_half_extent, recenter_to_canonical_frame

    mesh = trimesh.load(str(args.mesh), force="mesh", process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float32)

    view_img = Image.open(args.view).convert("RGBA")
    canonical = recenter_to_canonical_frame(view_img, size=CANVAS, border_ratio=0.15)
    view_alpha = np.asarray(canonical)[:, :, 3] > 128

    rows = []
    azimuths = np.arange(args.center - args.window, args.center + args.window + 0.01, args.step)
    for az in azimuths:
        half = canonical_ortho_half_extent(
            mesh, azimuth_deg=float(az), elevation_deg=0.0, border_ratio=0.15)
        sil = mesh_silhouette(vertices, float(az), 0.0, half, CANVAS)
        band = band_rows(sil)
        best = best_iou_over_scale_shift(sil, view_alpha, band)
        rows.append({"azimuth_deg": round(float(az), 2), **best})
        print(f"az {az:8.2f}  head-band IoU {best['iou']:.4f} "
              f"(scale {best['scale']}, shift {best['dx']},{best['dy']})", flush=True)

    best_row = max(rows, key=lambda r: r["iou"])
    at_center = next(r for r in rows if abs(r["azimuth_deg"] - args.center) < 1e-6)
    report = {
        "view": str(args.view), "declared_azimuth_deg": args.center,
        "measured_azimuth_deg": best_row["azimuth_deg"],
        "iou_at_measured": round(best_row["iou"], 4),
        "iou_at_declared": round(at_center["iou"], 4),
        "delta_deg": round(best_row["azimuth_deg"] - args.center, 2),
        "sweep": rows,
    }
    print(json.dumps({k: v for k, v in report.items() if k != "sweep"}, indent=1))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
