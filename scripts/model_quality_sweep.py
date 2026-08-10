#!/usr/bin/env python3
"""Four-metric model evaluation: mesh and texture, each at two angles.

Operator-ruled criteria (2026-07-21): mesh quality and texture quality are
INDEPENDENT axes measured at TWO angles each — e10/e15 class = texture
passable but mesh wrong; e20 class = mesh good but texture misplaced. One
conflated number hides both failure modes.

Metrics (raw, no opaque composite):
  mesh_front      NCC of horizontal-edge row profiles: front CLAY render vs
                  the source photo (are feature lines carved at the photo's
                  heights?). Higher is better; 1.0 = perfect placement.
  mesh_oblique    1 - duplication peak ratio on the 55-degree clay render
                  (striation/multi-ridge detector, raking-light analog),
                  minus a bumpiness penalty (high-frequency shading noise in
                  the face band). Higher is better.
  tex_front       CIE76 delta-E of the textured front render vs the photo
                  over the shared face band. Lower is better.
  tex_oblique     ghost-structure score on the 30-degree textured render:
                  extra dark glasses-like bands beyond the widest one
                  + saturation-outlier speckle ratio. Lower is better.

Photo is ground truth at the front; anatomy priors (one of each feature)
serve where no photo exists.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from PIL import Image


# ---------------------------------------------------------------- helpers
def subject_mask(arr: np.ndarray) -> np.ndarray:
    if arr.shape[2] == 4 and arr[..., 3].min() < 250:
        return arr[..., 3] > 10
    lum = arr[..., :3].mean(axis=2)
    return np.abs(lum - lum[0, 0]) > 8


def face_band(arr: np.ndarray, mask: np.ndarray, top_frac: float = 0.62):
    ys, xs = np.where(mask)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    h = y1 - y0 + 1
    y_cut = y0 + int(h * top_frac)
    return arr[y0:y_cut, x0 : x1 + 1], mask[y0:y_cut, x0 : x1 + 1]


def edge_profile(rgb: np.ndarray, mask: np.ndarray, rows: int = 384) -> np.ndarray:
    lum = rgb @ np.array([0.299, 0.587, 0.114])
    gy = np.abs(np.diff(lum, axis=0)) * np.minimum(mask[1:], mask[:-1])
    prof = gy.sum(axis=1)
    prof = np.interp(np.linspace(0, 1, rows), np.linspace(0, 1, max(len(prof), 2)), prof)
    s = prof.sum()
    return prof / s if s > 0 else prof


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    da, db = a - a.mean(), b - b.mean()
    d = np.sqrt((da * da).sum() * (db * db).sum())
    return float((da * db).sum() / d) if d > 0 else 0.0


def load_meshes(glb_path: Path):
    import trimesh

    scene = trimesh.load(str(glb_path), process=False)
    mesh_tex = scene.to_mesh() if isinstance(scene, trimesh.Scene) else scene
    mesh_tex.metadata["abstract3d_export_frame"] = "gltf_yup_front_pz"
    mesh_clay = trimesh.Trimesh(
        vertices=np.asarray(mesh_tex.vertices),
        faces=np.asarray(mesh_tex.faces),
        process=False,
    )
    mesh_clay.metadata["abstract3d_export_frame"] = "gltf_yup_front_pz"
    return mesh_clay, mesh_tex


def render(mesh, azimuth: float, elevation: float, size: int = 1536):
    from abstract3d.rendering import render_mesh_views

    return render_mesh_views(mesh, size=size, azimuths=[azimuth], elevation=elevation)[0]


# ---------------------------------------------------------------- metrics
def mesh_front_score(clay_front: Image.Image, photo: Image.Image) -> float:
    ca = np.asarray(clay_front.convert("RGBA"), dtype=np.float64)
    pa = np.asarray(photo.convert("RGBA"), dtype=np.float64)
    cb, cm = face_band(ca, subject_mask(ca))
    pb, pm = face_band(pa, subject_mask(pa))
    return round(ncc(edge_profile(cb[..., :3], cm), edge_profile(pb[..., :3], pm)), 4)


def duplication_ratio(clay_img: Image.Image) -> float:
    arr = np.asarray(clay_img.convert("RGBA"), dtype=np.float64)
    band, bm = face_band(arr, subject_mask(arr), top_frac=0.55)
    prof = edge_profile(band[..., :3], bm, rows=512)
    prof = prof - prof.mean()
    denom = float((prof * prof).sum()) or 1.0
    n = len(prof)
    lo, hi = int(n * 0.03), int(n * 0.15)
    best = 0.0
    for lag in range(lo, hi):
        c = float((prof[:-lag] * prof[lag:]).sum()) / denom
        best = max(best, c)
    return best


def bumpiness(clay_img: Image.Image) -> float:
    """High-frequency shading noise in the face band (cheek pimple class)."""
    arr = np.asarray(clay_img.convert("RGBA"), dtype=np.float64)
    band, bm = face_band(arr, subject_mask(arr))
    lum = band[..., :3] @ np.array([0.299, 0.587, 0.114])
    from scipy import ndimage

    smooth = ndimage.gaussian_filter(lum, 6.0)
    resid = np.abs(lum - smooth)[bm]
    return float(resid.mean()) if resid.size else 0.0


def mesh_oblique_score(clay_obl: Image.Image) -> float:
    dup = duplication_ratio(clay_obl)
    bump = bumpiness(clay_obl)
    return round(max(0.0, 1.0 - dup - min(bump / 12.0, 0.5)), 4)


def tex_front_delta(tex_front: Image.Image, photo: Image.Image) -> float:
    from skimage import color as skcolor

    pair = []
    for img in (tex_front, photo):
        arr = np.asarray(img.convert("RGBA"), dtype=np.float64)
        band, bm = face_band(arr, subject_mask(arr))
        rgb = Image.fromarray(band[..., :3].astype(np.uint8)).resize((384, 384), Image.LANCZOS)
        mm = Image.fromarray((bm * 255).astype(np.uint8)).resize((384, 384), Image.NEAREST)
        pair.append((np.asarray(rgb, dtype=np.float64), np.asarray(mm) > 127))
    (ra, rm), (pa, pm) = pair
    shared = rm & pm
    if not shared.any():
        return float("nan")
    la, lb = skcolor.rgb2lab(ra / 255.0), skcolor.rgb2lab(pa / 255.0)
    return round(float(np.sqrt(((la - lb) ** 2).sum(axis=2))[shared].mean()), 2)


def tex_oblique_ghost(tex_obl: Image.Image) -> float:
    """Extra dark glasses-like bands + saturation speckles, face band."""
    arr = np.asarray(tex_obl.convert("RGBA"), dtype=np.float64)
    band, bm = face_band(arr, subject_mask(arr))
    lum = band[..., :3] @ np.array([0.299, 0.587, 0.114])
    dark = (lum < 60) & bm
    width = bm.sum(axis=1)
    frac = np.divide(dark.sum(axis=1), np.maximum(width, 1))
    rows = (frac > 0.30) & (width > 0)
    # count distinct dark bands (runs) — one pair of glasses = 1 band
    runs, in_run = 0, False
    h = len(rows)
    for i, r in enumerate(rows):
        if r and not in_run:
            runs += 1
            in_run = True
        elif not r and in_run:
            in_run = False
    extra_bands = max(0, runs - 1)
    # saturation speckles: tiny high-saturation islands on skin
    mx = band[..., :3].max(axis=2)
    mn = band[..., :3].min(axis=2)
    sat = (mx - mn) / np.maximum(mx, 1)
    speckle = ((sat > 0.55) & (mx > 120) & bm).sum() / max(bm.sum(), 1)
    return round(extra_bands + float(speckle) * 20.0, 3)


# ---------------------------------------------------------------- sweep
def evaluate(glb: Path, photo: Image.Image) -> dict:
    mesh_clay, mesh_tex = load_meshes(glb)
    clay_front = render(mesh_clay, 0.0, 0.0)
    clay_obl = render(mesh_clay, 55.0, 12.0)
    tex_front = render(mesh_tex, 0.0, 0.0)
    tex_obl = render(mesh_tex, 30.0, 12.0)
    return {
        "mesh_front": mesh_front_score(clay_front, photo),
        "mesh_oblique": mesh_oblique_score(clay_obl),
        "tex_front_dE": tex_front_delta(tex_front, photo),
        "tex_oblique_ghost": tex_oblique_ghost(tex_obl),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--photo", required=True, type=Path)
    parser.add_argument("--glb", nargs="+", required=True, type=Path)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    photo = Image.open(args.photo)
    results = {}
    for glb in args.glb:
        label = glb.parent.name if glb.name == "scene.glb" else glb.stem
        try:
            results[label] = evaluate(glb, photo)
        except Exception as exc:
            results[label] = {"error": f"{type(exc).__name__}: {exc}"}
        print(f"{label:28s}", json.dumps(results[label]))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
