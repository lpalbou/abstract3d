#!/usr/bin/env python3
"""Oblique face close-ups at high resolution — the verification surface the
operator actually checks.

The striated-mouth defect shipped twice because verification rendered
front/45-degree views where raking light does not catch parallel ridges.
Oblique 30/55-degree views at 2048px, cropped to the face, make
multi-hypothesis carving visible. Run this on EVERY candidate before any
quality claim.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import trimesh
from PIL import Image


def load_meshes(glb_path: Path):
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


def face_crop(img: Image.Image, top_frac: float = 0.62):
    arr = np.asarray(img.convert("RGBA"))
    if arr[..., 3].min() < 250:
        mask = arr[..., 3] > 10
    else:
        lum = arr[..., :3].mean(axis=2)
        mask = np.abs(lum - lum[0, 0]) > 8
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    h = y1 - y0 + 1
    crop = img.crop((int(x0), int(y0), int(x1) + 1, int(y0 + h * top_frac)))
    if crop.width > 1200:
        crop = crop.resize((1200, int(1200 * crop.height / crop.width)), Image.LANCZOS)
    return crop


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glb", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--label", default="model")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    from abstract3d.rendering import render_mesh_views

    mesh_clay, mesh_tex = load_meshes(args.glb)
    angles = [("obl_l30", 30.0), ("obl_l55", 55.0), ("obl_r30", 330.0), ("obl_r55", 305.0)]
    for name, az in angles:
        for kind, mesh in (("clay", mesh_clay), ("tex", mesh_tex)):
            img = render_mesh_views(mesh, size=2048, azimuths=[az], elevation=12.0)[0]
            crop = face_crop(img)
            if crop is None:
                continue
            path = args.output_dir / f"{args.label}_{name}_{kind}.png"
            crop.save(path)
            print(path)


if __name__ == "__main__":
    main()
