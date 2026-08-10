#!/usr/bin/env python3
"""Texture PLACEMENT score — the criterion coverage cannot measure.

Root failure this fixes (e20/e21 incident): the bake reports coverage
(how much surface got painted) while the operator sees ghost glasses and
lips painted on the nose (paint in the WRONG PLACE). Placement has ground
truth: the source photo. Render the textured mesh from the source pose
and measure face-band disagreement against the photo itself; misplaced
paint that is visible from the front shows as structural error no matter
what my eye thinks of a render.

Score: masked, bbox-normalized comparison on the subject's face band
(top 62% of subject rows):
  - lab_delta: mean CIE76 delta-E over the shared support (color drift)
  - edge_disagreement: 1 - NCC of horizontal-edge row profiles (feature
    lines at wrong heights -> high)
  - ghost_band_ratio: dark glasses-like bands OUTSIDE the photo's own
    glasses rows (ghost frames painted low)
Lower is better for all three.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from PIL import Image


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


def normalize_pair(a_img, b_img, size: int = 384):
    """bbox-crop both to their subject face bands and resample to a common
    grid so pixelwise comparison is registration-tolerant at coarse scale."""
    out = []
    for img in (a_img, b_img):
        arr = np.asarray(img.convert("RGBA"), dtype=np.float64)
        m = subject_mask(arr)
        band, bm = face_band(arr, m)
        rgb = band[..., :3]
        im = Image.fromarray(rgb.astype(np.uint8)).resize((size, size), Image.LANCZOS)
        mm = Image.fromarray((bm * 255).astype(np.uint8)).resize((size, size), Image.NEAREST)
        out.append((np.asarray(im, dtype=np.float64), np.asarray(mm) > 127))
    return out


def lab_delta(a, b, mask):
    try:
        from skimage import color as skcolor
    except Exception as exc:  # pragma: no cover
        raise SystemExit(f"scikit-image required: {exc}")
    la = skcolor.rgb2lab(a / 255.0)
    lb = skcolor.rgb2lab(b / 255.0)
    d = np.sqrt(((la - lb) ** 2).sum(axis=2))
    return float(d[mask].mean()) if mask.any() else float("nan")


def edge_profile(a, mask):
    lum = a @ np.array([0.299, 0.587, 0.114])
    gy = np.abs(np.diff(lum, axis=0)) * np.minimum(mask[1:], mask[:-1])
    prof = gy.sum(axis=1)
    s = prof.sum()
    return prof / s if s > 0 else prof


def edge_disagreement(a, b, mask_a, mask_b):
    pa, pb = edge_profile(a, mask_a), edge_profile(b, mask_b)
    da, db = pa - pa.mean(), pb - pb.mean()
    denom = np.sqrt((da * da).sum() * (db * db).sum())
    ncc = float((da * db).sum() / denom) if denom > 0 else 0.0
    return 1.0 - max(ncc, 0.0)


def glasses_rows(a, mask, dark_thresh: float = 60.0, min_width_frac: float = 0.25):
    lum = a @ np.array([0.299, 0.587, 0.114])
    dark = (lum < dark_thresh) & mask
    width = mask.sum(axis=1)
    frac = np.divide(dark.sum(axis=1), np.maximum(width, 1))
    return (frac > min_width_frac) & (width > 0)


def ghost_band_ratio(render, photo, mask_r, mask_p):
    """Dark wide bands in the render at rows where the photo has none."""
    rows_r = glasses_rows(render, mask_r)
    rows_p = glasses_rows(photo, mask_p)
    # dilate photo rows by 5% height (registration tolerance)
    k = max(1, int(0.05 * len(rows_p)))
    dil = rows_p.copy()
    for i in np.flatnonzero(rows_p):
        dil[max(0, i - k) : i + k + 1] = True
    ghost = rows_r & ~dil
    return float(ghost.sum() / max(rows_r.sum(), 1))


def render_front(glb_path: Path, size: int = 2048):
    import trimesh

    from abstract3d.rendering import render_mesh_views

    scene = trimesh.load(str(glb_path), process=False)
    mesh = scene.to_mesh() if isinstance(scene, trimesh.Scene) else scene
    mesh.metadata["abstract3d_export_frame"] = "gltf_yup_front_pz"
    return render_mesh_views(mesh, size=size, azimuths=[0.0], elevation=0.0)[0]


def score(glb: Path, photo: Path) -> dict:
    render = render_front(glb)
    photo_img = Image.open(photo)
    (ra, rm), (pa, pm) = normalize_pair(render, photo_img)
    shared = rm & pm
    return {
        "lab_delta": round(lab_delta(ra, pa, shared), 2),
        "edge_disagreement": round(edge_disagreement(ra, pa, rm, pm), 4),
        "ghost_band_ratio": round(ghost_band_ratio(ra, pa, rm, pm), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--photo", required=True, type=Path)
    parser.add_argument("--glb", nargs="+", required=True, type=Path)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    results = {}
    for glb in args.glb:
        label = glb.parent.name if glb.name == "scene.glb" else glb.stem
        try:
            results[label] = score(glb, args.photo)
        except Exception as exc:
            results[label] = {"error": f"{type(exc).__name__}: {exc}"}
        print(label, json.dumps(results[label]))
    if args.json_out:
        args.json_out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
